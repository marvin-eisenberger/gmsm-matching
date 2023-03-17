import os
import scipy.io
import time
from model import *


def run_validation(method_array, dataset_arr, graph_mode_arr, val_highres=False, num_epoch=None, idx_pairs=None):
    print("val_highres =", val_highres)
    print("graph_mode_arr =", graph_mode_arr)
    print("num_epoch =", num_epoch)
    print("idx_pairs =", idx_pairs)

    for dataset in dataset_arr:
        dataset_val = dataset()
        dataset_val.init_diffusion_net_ops()
        for stamp in method_array:
            folder_path = save_path(stamp)

            model = GraphMultiShapeMatching(None, folder_path)
            model.load_self(folder_path, num_epoch=num_epoch)
            model.validate(dataset_val, graph_mode_arr=graph_mode_arr, val_highres=val_highres, idx_pairs=idx_pairs)


if __name__ == "__main__":
    method_array = [
        "shrec20_pretrained",
    ]

    dataset_arr = [
        Shrec20_full,
    ]
	
    graph_mode_arr = ["full"]
	
    run_validation(method_array, dataset_arr, graph_mode_arr)
