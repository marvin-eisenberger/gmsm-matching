import torch.nn
import scipy.io
import numpy as np
import time
from matching.diff_matching import *
from data.data import *
import diffusion_net as diffusion_net
from matching.shape_graph import *


class FeatModuleDiffusionNet(torch.nn.Module):
    def __init__(self, C_in=3, C_out=128, C_width=128):
        super().__init__()

        self.diffnet = diffusion_net.layers.DiffusionNet(
            C_in=C_in,
            C_out=C_out,
            C_width=C_width,
            outputs_at='vertices')

        print("Using FeatModuleDiffusionNet")

    def forward(self, shape_x: Shape):
        if shape_x.diffusion_net_ops is not None:
            frames, mass, L, evals, evecs, gradX, gradY = \
                shape_x.diffusion_net_ops
        else:
            frames, mass, L, evals, evecs, gradX, gradY = \
                diffusion_net.geometry.get_operators(shape_x.vert.to(device_cpu), shape_x.triv.to(device_cpu))

        frames, mass, L, evals, evecs, gradX, gradY = \
            frames.to(device), mass.to(device), L.to(device), evals.to(device), evecs.to(device), gradX.to(device), gradY.to(device)

        emb = self.diffnet(shape_x.vert, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=shape_x.triv)

        return emb[shape_x.samples, :]


class GraphMultiShapeMatching:
    def __init__(self, dataset=None, save_path=None, param=None):
        self.save_path = save_path
        self.dataset = dataset
        if dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        if param is None:
            self.param = DeepParam()
        else:
            self.param = param

        self._init_feat_mod()

        self.shells = OTSmoothShells(self.param)
        self.i_epoch = 0

        print("Using ", self.__class__.__name__)

    def _init_feat_mod(self):
        self.feat_mod = FeatModuleDiffusionNet().to(device)
        if self.dataset:
            self.dataset.init_diffusion_net_ops()

        self.optimizer = torch.optim.Adam(self.feat_mod.parameters(), lr=self.param.lr)

    def save_self(self):
        folder_path = os.path.join(self.save_path, "chkpt")

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        ckpt = {'i_epoch': self.i_epoch,
                'feat_mod': self.feat_mod.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'par': self.param.__dict__}

        ckpt_name = 'ckpt_ep{}.pth'.format(self.i_epoch)
        ckpt_path = os.path.join(folder_path, ckpt_name)

        ckpt_last_name = 'ckpt_last.pth'
        ckpt_last_path = os.path.join(folder_path, ckpt_last_name)

        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, ckpt_last_path)

    def load_self(self, folder_path, num_epoch=None):
        if num_epoch is None:
            ckpt_name = 'ckpt_last.pth'
            ckpt_path = os.path.join(folder_path, "chkpt", ckpt_name)
        else:
            ckpt_name = 'ckpt_ep{}.pth'.format(num_epoch)
            ckpt_path = os.path.join(folder_path, "chkpt", ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)

        if 'par' in ckpt:
            self.param.from_dict(ckpt['par'])
            self.param.print_self()
            self._init_feat_mod()

        self.i_epoch = ckpt['i_epoch']
        self.feat_mod.load_state_dict(ckpt['feat_mod'])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.feat_mod.train()

        if num_epoch is None:
            print("Loaded model from ", folder_path, " with the latest weights")
        else:
            print("Loaded model from ", folder_path, " with the weights from epoch ", num_epoch)

    def train(self, num_epochs=int(1e5)):
        self.param.print_self()
        print("start training ...")
        self.feat_mod.train()

        if self.param.use_graph and self.i_epoch >= self.param.graph_burn_in and self.param.graph_mode != "vanilla":
            graph = self.construct_shape_graph(self.dataset, self.param.graph_mode)
            assign_x_col, assign_y_col = graph.get_all_matches(two_sided=True)
        else:
            graph = None

        while self.i_epoch < num_epochs:
            tot_loss = 0
            loss_comp = [0] * (1 if graph is None else 2)
            i_tot = 0
            for i, data in enumerate(self.train_loader):
                i_tot += 1
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                if self.param.subsample_num is not None:
                    shape_x.subsample_random(self.param.subsample_num)
                    shape_y.subsample_random(self.param.subsample_num)

                loss_ds = self.match_pair(shape_x, shape_y)

                loss = loss_ds
                loss_comp[0] += loss_ds.item() / self.dataset.__len__()

                if graph is not None:
                    i_x = data["i_x"]
                    i_y = data["i_y"]
                    assign_x = my_long_tensor(assign_x_col[i_x][i_y])
                    assign_y = my_long_tensor(assign_y_col[i_x][i_y])

                    shape_x.reset_sampling()
                    shape_y.reset_sampling()

                    samples_x = torch.cat((shape_x.samples, assign_y), 0)
                    samples_y = torch.cat((assign_x, shape_y.samples), 0)

                    loss_graph = self.param.lambda_graph * self.param.subsample_num * \
                                 ((self.shells.vert_x_star[samples_x, :] - self.shells.vert_y_smooth[samples_y, :]) ** 2).sum(dim=1).mean()

                    loss = loss + loss_graph
                    loss_comp[1] += loss_graph.item() / self.dataset.__len__()

                loss.backward()

                if i_tot % self.param.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                tot_loss += loss.item() / self.dataset.__len__()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if graph is not None:
                print("epoch {:d}, loss = {:.4f} (loss_ds = {:.4f}, loss_graph ={:.4f})".format(self.i_epoch, tot_loss, loss_comp[0], loss_comp[1]))
            else:
                print("epoch {:d}, loss = {:.4f}".format(self.i_epoch, tot_loss))
            if self.i_epoch % self.param.log_freq == 0 and self.save_path is not None:
                self.save_self()
                print_memory_status(self.i_epoch)

            if self.param.use_graph and \
                    self.i_epoch % self.param.graph_update_freq == 0 and \
                    self.i_epoch >= self.param.graph_burn_in and \
                    self.param.graph_mode != "vanilla":
                graph = self.construct_shape_graph(self.dataset, self.param.graph_mode)
                assign_x_col, assign_y_col = graph.get_all_matches(two_sided=True)

            self.i_epoch += 1

    def get_val_folder_name(self):
        return "val"

    def validate(self, dataset_val, graph_mode_arr=None, val_highres=False, two_sided=False, idx_pairs=None):
        self.param.print_self()
        print("Validate dataset: ", dataset_val.get_name())

        if val_highres:
            print("High resolution validation")

        if self.save_path is None:
            print("Save path not specified!")
            return

        if graph_mode_arr is None or not len(graph_mode_arr):
            print("graph_mode_arr={}, nothing to do here!".format(graph_mode_arr))
            return

        start_time = time.time()

        mat_vanilla = self.get_all_matches(dataset_val, val_highres=val_highres)

        t_elapsed_network_query = (time.time() - start_time)
        print("network query time = {:05d}s".format(int(t_elapsed_network_query)))

        for graph_mode in graph_mode_arr:
            print("graph_mode={}".format(graph_mode))
            val_folder_name = self.get_val_folder_name()
            folder_path = os.path.join(self.save_path,
                                       val_folder_name + "_highres" if val_highres else val_folder_name,
                                       "ep_{:04d}".format(self.i_epoch),
                                       dataset_val.get_name(),
                                       graph_mode)

            print("Saving to: ", folder_path)

            os.makedirs(folder_path, exist_ok=True)

            loss_matrix, assign_x_col, assign_y_col = mat_vanilla
            if graph_mode != "vanilla":
                start_time = time.time()
                graph = graph_mode_to_type[graph_mode](dataset_val, loss_matrix, assign_x_col, assign_y_col)

                t_elapsed_graph_construct = (time.time() - start_time)
                print("graph construct time = {:05d}s".format(int(t_elapsed_graph_construct)))

                loss_matrix = graph.adj
                a = graph.get_all_matches(two_sided=two_sided)
                if two_sided:
                    assign_x_col, assign_y_col = a
                else:
                    assign_x_col, assign_y_col = a, None

                t_elapsed_graph_query = (time.time() - start_time)
                print("graph query time = {:05d}s".format(int(t_elapsed_graph_query)))
            n = len(assign_x_col)


            for i_x in range(n):
                for i_y in range(n):
                    if idx_pairs is not None and (i_x, i_y) not in idx_pairs:
                        continue
                    assign_x = assign_x_col[i_x][i_y]
                    mat_dict = {'i_x': i_x,
                                'i_y': i_y,
                                'assign_x': np.expand_dims(assign_x, 1) + 1,
                                }

                    if two_sided:
                        assign_y = assign_y_col[i_x][i_y]
                        mat_dict["assign_y"] = np.expand_dims(assign_y, 1) + 1,

                    file_mat = "pair__{:03d}_{:03d}.mat".format(i_x, i_y)

                    scipy.io.savemat(os.path.join(folder_path, file_mat), mat_dict)

            scipy.io.savemat(os.path.join(folder_path, "loss_{:d}.mat".format(1)),
                             {
                                 "loss": loss_matrix
                             })

            if graph_mode != "vanilla":
                scipy.io.savemat(os.path.join(folder_path, "graph.mat"),
                                 {
                                     "graph": graph.graph.toarray(),
                                     "t_elapsed_network_query": t_elapsed_network_query,
                                     "t_elapsed_graph_construct": t_elapsed_graph_construct,
                                     "t_elapsed_graph_query": t_elapsed_graph_query,
                                 })

            print_memory_status(self.i_epoch)

        print("Validate, epoch {:d}".format(self.i_epoch))

    def get_all_matches(self, dataset, val_highres=False):
        graph_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        n = dataset.num_shapes

        loss_matrix = np.zeros((n, n))
        assign_x_col = [[None]*n for _ in range(n)]
        assign_y_col = [[None]*n for _ in range(n)]
        tot_loss_arr = [0] * 2

        with torch.no_grad():
            print("Querying network for all val pairs...")
            for i, data in enumerate(graph_loader):

                # load shapes from dataset
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])
                i_x = data["i_x"].item()
                i_y = data["i_y"].item()

                # query network
                assign_x, assign_y, loss_arr = self.test_model_fast(shape_x, shape_y, val_highres)

                for i in range(len(tot_loss_arr)):
                    tot_loss_arr[i] += loss_arr[i] / dataset.__len__()

                loss_matrix[i_x, i_y] = loss_arr[1]

                assign_x_col[i_x][i_y] = assign_x.detach().cpu().numpy()
                assign_y_col[i_x][i_y] = assign_y.detach().cpu().numpy()

        for i in range(len(tot_loss_arr)):
            print("val_loss_{:d} = {:f}".format(i, tot_loss_arr[i]))

        return loss_matrix, assign_x_col, assign_y_col

    def construct_shape_graph(self, dataset, graph_mode, val_highres=False) -> ShapeGraph:
        print("Constructing shape graph...")

        loss_matrix, assign_x_col, assign_y_col = self.get_all_matches(dataset, val_highres=val_highres)

        return graph_mode_to_type[graph_mode](dataset, loss_matrix, assign_x_col, assign_y_col)

    def test(self, dataset_val):
        self.param.print_self()
        print("Test dataset: ", dataset_val.get_name())

        if self.save_path is None:
            print("Save path not specified!")
            return

        folder_path = os.path.join(self.save_path, "test", "ep_{:04d}".format(self.i_epoch), dataset_val.get_name())

        print("Saving to: ", folder_path)

        os.makedirs(folder_path, exist_ok=True)

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

        loss_matrix = np.zeros((dataset_val.num_shapes, dataset_val.num_shapes))

        start_time = time.time()

        with torch.no_grad():
            tot_loss = 0
            for i, data in enumerate(val_loader):
                t_elapsed = (time.time() - start_time)
                print("pair = {:04d}/{:04d}; time elapsed = {:05d}s".format(i, len(dataset_val), int(t_elapsed)))
				
                # load shapes from dataset
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])
                i_x = data["i_x"].item()
                i_y = data["i_y"].item()

                assignment, assignmentinv, loss = self.test_model(shape_x, shape_y, loss_out=True)

                tot_loss += loss.detach() / dataset_val.__len__()

                loss_matrix[i_x, i_y] = loss.item()

                # save results in mat file
                mat_dict = {'ass_x': assignment.unsqueeze(1).detach().cpu().numpy()+1,
                            'ass_y': assignmentinv.unsqueeze(1).detach().cpu().numpy()+1,
                            'loss': loss.item(),
                            'i_x': i_x,
                            'i_y': i_y,
                            'shape_file_x': data['X']['file_name'],
                            'shape_file_y': data['Y']['file_name'],
                            }

                file_mat = "pair__{:03d}_{:03d}.mat".format(i_x, i_y)

                scipy.io.savemat(os.path.join(folder_path, file_mat), mat_dict)

            scipy.io.savemat(os.path.join(folder_path, "loss.mat"), {"loss": loss_matrix})

            print_memory_status(self.i_epoch)

            print("Test, epoch {:d}".format(self.i_epoch), ", loss = {:f}".format(tot_loss))

    def feat_embedding(self, shape_x, shape_y):
        emb_x = self.feat_mod(shape_x)
        emb_y = self.feat_mod(shape_y)
        return emb_x, emb_y

    def feat_corr_pair(self, shape_x, shape_y):
        emb_x, emb_y = self.feat_embedding(shape_x, shape_y)

        self.shells.feat_correspondences(shape_x, shape_y, emb_x, emb_y)

    def match_pair(self, shape_x, shape_y):
        # compute learned correspondences
        self.feat_corr_pair(shape_x, shape_y)

        # match pair
        loss = self.shells.hierarchical_matching(shape_x, shape_y)

        return loss

    def test_model_fast(self, shape_x, shape_y, val_highres=False):
        if self.param.subsample_num is not None:
            shape_x.subsample_random(self.param.subsample_num)
            shape_y.subsample_random(self.param.subsample_num)

        loss_0 = self.match_pair(shape_x, shape_y)

        shape_x.reset_sampling()
        shape_y.reset_sampling()

        self.shells.smooth_shape(shape_x, shape_y)
        emb_x, emb_y = self.shells.product_embedding(shape_x, shape_y)

        # fine alignment
        ass_shells = AssSmoothShells()
        ass_shells.param.status_log = False
        if not val_highres:
            ass_shells.param.k_array_len = 1
        ass_shells.param.k_min = self.shells.param.k_max
        if not val_highres:
            ass_shells.param.k_max = ass_shells.param.k_min
        ass_shells.param.compute_karray()

        ass_shells.embedding_correspondences(shape_x, shape_y, emb_x, emb_y)
        loss_1 = ass_shells.hierarchical_matching(shape_x, shape_y)

        assign_x = ass_shells.samples_y[:shape_x.vert.shape[0]]
        assign_y = ass_shells.samples_x[shape_x.vert.shape[0]:]

        return assign_x, assign_y, [loss_0.item(), loss_1.item()]
