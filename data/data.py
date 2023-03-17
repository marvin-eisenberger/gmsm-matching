# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List
import os
import scipy.io
import diffusion_net
from utils.shape_utils import *


def input_to_batch(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    for attr in ["A"]:
        dict_out[attr] = np.asarray(mat_dict[attr][0].diagonal(), dtype=np.float32)

    return dict_out


def batch_to_shape(batch):
    shape = Shape(batch["vert"].squeeze().to(device), batch["triv"].squeeze().to(device, torch.long) - 1)

    for attr in ["evecs", "evals", "SHOT", "A"]:
        setattr(shape, attr, batch[attr].squeeze().to(device))

    if "diffusion_net_ops" in batch:
        for i in range(len(batch["diffusion_net_ops"])):
            batch["diffusion_net_ops"][i] = \
                batch["diffusion_net_ops"][i][0].to(device)
        setattr(shape, "diffusion_net_ops", batch["diffusion_net_ops"])

    shape.compute_xi_()

    return shape


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, file_fct, num_shapes):
        self.file_fct = file_fct
        self.num_shapes = num_shapes

        self.data = []

        self._init_data()

    def _init_data(self):
        for i in range(self.num_shapes):
            file_name = self.file_fct(self._get_index(i))
            load_data = scipy.io.loadmat(file_name)

            data_curr = input_to_batch(load_data["X"][0])
            data_curr["file_name"] = file_name

            self.data.append(data_curr)

            print("Loaded file ", file_name, "")

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_name(self):
        return self.__class__.__name__

    def init_diffusion_net_ops(self):
        print("Init diffusion_net_ops...")
        for d in self.data:
            vert = torch.tensor(d["vert"], device=device_cpu, dtype=torch.float32)
            triv = torch.tensor(d["triv"], device=device_cpu, dtype=torch.long) - 1
            d["diffusion_net_ops"] = \
                diffusion_net.geometry.get_operators(vert, triv)

    def save_data(self, folder_out):
        os.makedirs(folder_out, exist_ok=True)
        for i in range(len(self.data)):
            file_out = os.path.join(folder_out, "shape_{:03d}.mat".format(i))

            mat_dict = {'X': {
                'vert': self.data[i]["vert"],
                'triv': self.data[i]["triv"]},
                'i': i}

            scipy.io.savemat(file_out, mat_dict)


class ShapeDatasetCombine(ShapeDataset):
    def __init__(self, file_fct, num_shapes):
        super().__init__(file_fct, num_shapes)
        self.num_pairs = num_shapes ** 2

        print("loaded ", self.get_name(), " with ", self.num_pairs, " pairs")

    def __getitem__(self, index):
        i1 = int(index / self.num_shapes)
        i2 = int(index % self.num_shapes)
        data_curr = dict()
        data_curr["X"] = self.data[i1]
        data_curr["Y"] = self.data[i2]

        data_curr["i_x"] = i1
        data_curr["i_y"] = i2

        return data_curr

    def __len__(self):
        return self.num_pairs


class ShapeDatasetCombineMulti(ShapeDatasetCombine):
    def __init__(self, datasets: List[ShapeDatasetCombine]):
        self.datasets = datasets
        num_shapes = sum([d.num_shapes for d in datasets])
        super().__init__(None, num_shapes)

    def _init_data(self):
        for d in self.datasets:
            self.data += d.data

    def get_name(self):
        ans = "Combine_"
        for d in self.datasets:
            ans += "_" + d.get_name()
        return ans


def get_shrec20_file(i):
    folder_path = "./data/shrec20"

    file_arr = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_arr.sort()
    return os.path.join(folder_path, file_arr[i])

class Shrec20_full(ShapeDatasetCombine):
    def __init__(self):
        super().__init__(get_shrec20_file, 14)

    def get_name(self):
        return "Shrec20_full"