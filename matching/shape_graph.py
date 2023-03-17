import numpy as np
from utils.data import ShapeDatasetCombine
import scipy.sparse.csgraph
from utils.tools import my_long_tensor
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing


def get_undirected_path(graph, s, t):
    _, (pred,) = scipy.sparse.csgraph.shortest_path(graph, return_predecessors=True, directed=False, indices=[s])
    path = [t]
    i = t
    while s != i:
        path = [pred[i]] + path
        i = pred[i]
    return path


class ShapeGraph:
    def __init__(self, dataset: ShapeDatasetCombine, adj, assign_x_col, assign_y_col):
        self.dataset = dataset
        self.adj = adj
        self.graph = None
        self.assign_x_col = assign_x_col
        self.assign_y_col = assign_y_col

        self.n = self.adj.shape[0]

        if assign_x_col is not None and assign_y_col is not None:
            self.n_x = [assign_x_col[i][0].shape[0] for i in range(self.n)]
            self.n_y = [assign_y_col[0][i].shape[0] for i in range(self.n)]

        self._init_graph(adj)

        print("Using '{}' shape graph".format(str(self)))

    def __str__(self):
        raise NotImplementedError

    def _init_graph(self, adj):
        raise NotImplementedError

    def shortest_path(self, s, t):
        return get_undirected_path(self.graph, s, t)

    def get_match(self, s, t):
        path = self.shortest_path(s, t)

        idx_x = np.arange(0, self.n_x[s], dtype=int)
        idx_y = idx_x

        for i_p in range(len(path)-1):
            i, j = path[i_p], path[i_p+1]

            if self.adj[i, j] < self.adj[j, i]:
                assign = self.assign_x_col[i][j]
            else:
                assign = self.assign_y_col[j][i]
            idx_y = assign[idx_y]

        return idx_x, idx_y

    def get_all_matches(self, two_sided=False):
        assign_x_col = [[None] * self.n for _ in range(self.n)]
        if two_sided:
            assign_y_col = [[None] * self.n for _ in range(self.n)]
        for s in range(self.n):
            for t in range(self.n):
                _, assign_x = self.get_match(s, t)
                assign_x_col[s][t] = assign_x
                if two_sided:
                    assign_y_col[t][s] = assign_x

        if two_sided:
            return assign_x_col, assign_y_col
        else:
            return assign_x_col


class FullShapeGraph(ShapeGraph):
    def __init__(self, dataset: ShapeDatasetCombine, adj, assign_x_col, assign_y_col):
        super().__init__(dataset, adj, assign_x_col, assign_y_col)

    def __str__(self):
        return "full"

    def _init_graph(self, adj):
        adj = np.minimum(adj, adj.T)
        self.graph = scipy.sparse.csgraph.csgraph_from_dense(adj)


class MstShapeGraph(ShapeGraph):
    def __init__(self, dataset: ShapeDatasetCombine, adj, assign_x_col, assign_y_col):
        super().__init__(dataset, adj, assign_x_col, assign_y_col)

    def __str__(self):
        return "mst"

    def _init_graph(self, adj):
        adj = np.minimum(adj, adj.T)
        self.graph = scipy.sparse.csgraph.csgraph_from_dense(adj)
        self.graph = scipy.sparse.csgraph.minimum_spanning_tree(self.graph)


class StarShapeGraph(ShapeGraph):
    def __init__(self, dataset: ShapeDatasetCombine, adj, assign_x_col, assign_y_col):
        super().__init__(dataset, adj, assign_x_col, assign_y_col)

    def __str__(self):
        return "star"

    def _init_graph(self, adj):
        adj = np.minimum(adj, adj.T)
        i_center = np.argmin(np.mean(adj, axis=1))
        print("Canonical pose index of star graph = ", i_center)
        adj_star = np.zeros_like(adj)
        adj_star[i_center, :] = adj[i_center, :]
        adj_star[:, i_center] = adj[:, i_center]
        self.graph = scipy.sparse.csgraph.csgraph_from_dense(adj_star)


class TspShapeGraph(ShapeGraph):
    def __init__(self, dataset: ShapeDatasetCombine, adj, assign_x_col, assign_y_col):
        super().__init__(dataset, adj, assign_x_col, assign_y_col)

    def __str__(self):
        return "tsp"

    def _init_graph(self, adj):
        adj = np.minimum(adj, adj.T)

        adj_inf = adj
        adj_inf[adj_inf == 0] = np.inf
        adj_tsp = np.zeros((adj_inf.shape[0]+1, adj_inf.shape[1]+1))
        adj_tsp[:self.n, :self.n] = adj_inf

        idx, _ = solve_tsp_simulated_annealing(adj_tsp)

        i_n = idx.index(self.n)
        idx = idx[i_n+1:] + idx[:i_n]

        adj_graph = np.zeros_like(adj)

        for k in range(len(idx)-1):
            i, j = idx[k], idx[k+1]
            adj_graph[i, j] = adj[i, j]
            adj_graph[j, i] = adj[j, i]

        self.graph = scipy.sparse.csgraph.csgraph_from_dense(adj_graph)


graph_mode_to_type = {"full": FullShapeGraph, "mst": MstShapeGraph, "star": StarShapeGraph, "tsp": TspShapeGraph}

