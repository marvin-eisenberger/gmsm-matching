# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import scipy.io
from torch_geometric.nn import fps, knn_graph
import matplotlib.pyplot as plt
from utils.param import *
from utils.tools import *
import random


def save_alignment_plt(folder_path, i, shape_x, shape_y, vert_x_star, vert_x_smooth, vert_y_smooth):
    vert_arr = [shape_x.vert, vert_x_smooth, vert_x_star, vert_y_smooth, shape_y.vert]
    triv_arr = [shape_x.triv, shape_x.triv, shape_x.triv, shape_y.triv, shape_y.triv]
    acr_arr = ["x", "x_smooth", "x_star", "y_smooth", "y"]
    save_seq_plt(folder_path, i, vert_arr, triv_arr, acr_arr)


def save_seq_plt(folder_path, i, vert_arr, triv_arr, acr_arr):
    for j in range(len(vert_arr)):
        vert = vert_arr[j].detach().cpu().numpy()
        triv = triv_arr[j].detach().cpu().numpy()
        acr = acr_arr[j]

        if acr:
            file_out = os.path.join(folder_path, "pair_{:03d}_{:03d}_{}.png".format(i, j, acr))
        else:
            file_out = os.path.join(folder_path, "pair_{:03d}_{:03d}.png".format(i, j))

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        max_bound = np.abs(vert).max()
        ax.plot_trisurf(vert[:, 0], vert[:, 1], vert[:, 2],
                     triangles=triv,
                     cmap='viridis',
                     linewidths=0.2)
        ax.set_xlim(-max_bound, max_bound)
        ax.set_ylim(-max_bound, max_bound)
        ax.set_zlim(-max_bound, max_bound)

        plt.savefig(file_out)
        plt.clf()

def plot_shape_pair(shape_x, shape_y, vert_x, vert_y, tit=None):
    vert_x = vert_x.detach().cpu().numpy()
    vert_y = vert_y.detach().cpu().numpy()
    max_bound = max([np.abs(vert_x).max(), np.abs(vert_y).max()])

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_trisurf(vert_x[:, 0], vert_x[:, 1], vert_x[:, 2], triangles=shape_x.get_triv_np(), cmap='viridis', linewidths=0.2)
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.set_zlim(-max_bound, max_bound)

    ax = fig.add_subplot(122, projection='3d')
    ax.plot_trisurf(vert_y[:, 0], vert_y[:, 1], vert_y[:, 2], triangles=shape_y.get_triv_np(), cmap='viridis', linewidths=0.2)
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.set_zlim(-max_bound, max_bound)

    if tit is not None:
        plt.title(tit)

    plt.show()


def plot_shape_triplet(shape_x, shape_y, vert_new):
    vert_x = shape_x.vert.detach().cpu().numpy()
    vert_y = shape_y.vert.detach().cpu().numpy()
    vert_new = vert_new.detach().cpu().numpy()

    vert_x = vert_x - vert_x.mean(0, keepdims=True)
    vert_y = vert_y - vert_y.mean(0, keepdims=True)
    vert_new = vert_new - vert_new.mean(0, keepdims=True)
    max_bound = max([np.abs(vert_x).max(), np.abs(vert_y).max(), np.abs(vert_new).max()])

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_trisurf(vert_x[:, 0], vert_x[:, 1], vert_x[:, 2], triangles=shape_x.get_triv_np(), cmap='viridis', linewidths=0.2)
    plt.title("X")
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.set_zlim(-max_bound, max_bound)

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_trisurf(vert_y[:, 0], vert_y[:, 1], vert_y[:, 2], triangles=shape_y.get_triv_np(), cmap='viridis', linewidths=0.2)
    plt.title("Y")
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.set_zlim(-max_bound, max_bound)

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_trisurf(vert_new[:, 0], vert_new[:, 1], vert_new[:, 2], triangles=shape_x.get_triv_np(), cmap='viridis', linewidths=0.2)
    plt.title("X*")
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.set_zlim(-max_bound, max_bound)

    plt.show()


def shape_from_dict(mat_dict):
    shape = Shape(torch.from_numpy(mat_dict["vert"][0].astype(np.float32)).to(device),
                  torch.from_numpy(mat_dict["triv"][0].astype(np.int64)).to(device) - 1)

    for attr in ["evecs", "evals", "normal", "area", "SHOT"]:
        setattr(shape, attr, torch.tensor(mat_dict[attr][0], device=device, dtype=torch.float32))

    for attr in ["A"]:
        mat = mat_dict[attr][0].diagonal()
        setattr(shape, attr, torch.tensor(mat, device=device, dtype=torch.float32))

    shape.compute_xi_()

    return shape


def load_shape_pair(file_load):
    mat_dict = scipy.io.loadmat(file_load)

    print("Loaded file ", file_load, "")

    shape_x = shape_from_dict(mat_dict["X"][0])
    shape_y = shape_from_dict(mat_dict["Y"][0])

    return shape_x, shape_y


def compute_outer_normal(vert, triv, samples):
    edge_1 = torch.index_select(vert, 0, triv[:, 1]) - torch.index_select(vert, 0, triv[:, 0])
    edge_2 = torch.index_select(vert, 0, triv[:, 2]) - torch.index_select(vert, 0, triv[:, 0])

    face_norm = torch.cross(1e4*edge_1, 1e4*edge_2)

    normal = my_zeros(vert.shape)
    for d in range(3):
        normal = torch.index_add(normal, 0, triv[:, d], face_norm)
    normal = normal / (1e-5 + normal.norm(dim=1, keepdim=True))

    return normal[samples, :]


class Shape:
    def __init__(self, vert=None, triv=None):
        self.vert = vert
        self.triv = triv
        self.samples = None
        self.reset_sampling()
        self.neigh = None
        self.neigh_hessian = None
        self.mahal_cov_mat = None
        self.evecs = None
        self.evals = None
        self.A = None
        self.W = None
        self.basisfeatures = None
        self.SHOT = None
        self.normal = None
        self.area = None
        self.xi = None
        self.diffusion_net_ops = None

    def subsample_fps(self, n_vert):
        assert n_vert <= self.vert.shape[0], "you can only subsample to less vertices than before"

        ratio = n_vert / self.vert.shape[0]
        self.samples = fps(self.vert.detach().to(device_cpu), ratio=ratio).to(device)

    def subsample_random(self, n_vert):
        self.samples = my_long_tensor([random.randint(0, self.vert.shape[0]-1) for _ in range(n_vert)])

    def reset_sampling(self):
        self.samples = my_long_tensor(list(range(self.vert.shape[0])))
        self.neigh = None

    def compute_xi_(self):
        if self.evecs is not None and self.A is not None and self.vert is not None:
            self.xi = torch.mm(self.evecs.transpose(0, 1), self.vert * self.A.unsqueeze(1))

    def get_vert(self):
        return self.vert[self.samples, :]

    def get_vert_shape(self):
        return self.get_vert().shape

    def get_triv(self):
        return self.triv

    def get_triv_np(self):
        return self.triv.detach().cpu().numpy()

    def get_vert_np(self):
        return self.vert[self.samples, :].detach().cpu().numpy()

    def get_vert_full_np(self):
        return self.vert.detach().cpu().numpy()

    def get_edge_index(self, num_knn=5):
        edge_index_one = self.get_neigh(num_knn).t()
        edge_index = torch.zeros(
            [2, edge_index_one.shape[1] * 2], dtype=torch.long, device=self.vert.device
        )
        edge_index[:, : edge_index_one.shape[1]] = edge_index_one
        edge_index[0, edge_index_one.shape[1] :] = edge_index_one[1, :]
        edge_index[1, edge_index_one.shape[1] :] = edge_index_one[0, :]
        return edge_index

    def get_neigh(self, num_knn=5):
        if self.neigh is None:
            self.compute_neigh(num_knn=num_knn)

        return self.neigh

    def compute_neigh(self, num_knn=5):
        self._triv_neigh()

    def _triv_neigh(self):
        self.neigh = torch.cat((self.triv[:, [0, 1]], self.triv[:, [0, 2]], self.triv[:, [1, 2]]), 0)

    def to(self, device):
        self.vert = self.vert.to(device)
        self.triv = self.triv.to(device)
