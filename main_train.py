from model import *
from utils.param import *


def main_train():
    folder_out = save_path("shrec20")
	
    dataset = Shrec20_full()

    model = GraphMultiShapeMatching(dataset, folder_out)
    model.train()


if __name__ == "__main__":
    main_train()
