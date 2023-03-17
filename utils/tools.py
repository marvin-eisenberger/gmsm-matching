# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from utils.param import device


def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)


def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def my_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.float32)


def my_long_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.long)


def my_range(start, end, step=1):
    return torch.arange(start=start, end=end, step=step, device=device, dtype=torch.float32)


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)


def print_memory_status(pos=None):
    if torch.torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        res = torch.cuda.memory_reserved(0) // (1024**2)
        all = torch.cuda.memory_allocated(0) // (1024**2)
        free = res - all
        if pos is not None:
            print("Memory status at pos", pos, ": tot =", tot, ", res =", res, ", all =", all, ", free =", free)
        else:
            print("Memory status: tot =", tot, ", res =", res, ", all =", all, ", free =", free)
        return torch.cuda.memory_reserved(0) / 1024**2
    else:
        print("Memory status: CUDA NOT AVAILABLE")