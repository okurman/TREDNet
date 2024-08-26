#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
from time import perf_counter

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lib.trainer import train_model
from torchvision import datasets, transforms
from torchsummary import summary
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters
# NUM_EPOCHS = 30
NUM_EPOCHS = 100
BATCH_SIZE = 256
# BATCH_SIZE = 12

torch.manual_seed(1024)

import lib.models as model_builder
from lib import trainer


class HDF5Dataset(Dataset):
    def __init__(self, data_file, x_name, y_name):
        self.file = data_file
        self.inf = h5py.File(data_file, "r")
        self.X = self.inf[x_name]
        self.Y = self.inf[y_name]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        _x = np.asarray(self.X[idx], dtype=np.float32).swapaxes(0, 1)
        _y = np.asarray(self.Y[idx], dtype=np.float32)

        return _x, _y


def load_dataset_wrappers(data_file, label_num=None):

    if label_num is not None:

        train_dataset = HDF5Dataset(data_file, "X_train", f"Y_train_{label_num}")
        val_dataset   = HDF5Dataset(data_file, "X_val", f"Y_val_{label_num}")
        test_dataset  = HDF5Dataset(data_file, "X_test", f"Y_test_{label_num}")

    else:

        train_dataset = HDF5Dataset(data_file, "X_train", f"Y_train")
        val_dataset   = HDF5Dataset(data_file, "X_val", f"Y_val")
        test_dataset  = HDF5Dataset(data_file, "X_test", f"Y_test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, prefetch_factor=2)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    return train_loader, val_loader, test_loader


def mnist_train():

    torch.manual_seed(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAVE_DIR = "/home/ubuntu/TREDNet/data/phase_one/model_runs/mnist"

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST(SAVE_DIR, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(SAVE_DIR, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = model_builder.Net().to(device)
    trainer.train_model(model, train_loader, test_loader, SAVE_DIR)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA_FILE = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
    # SAVE_DIR = "/Users/okurman/Projects/TREDNet/data/phase_one/model_runs/model_v1/"

    DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
    SAVE_DIR  = "/home/ubuntu/TREDNet/data/phase_one/model_runs/model_v4/"

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE)

    # model = model_builder.sequential_model_small(label_size=10)
    # model = model_builder.sequential_model(label_size=4560)

    model = model_builder.PhaseOne()
    model.to(device)
    summary(model, (4, 1000))

    train_model(model, train_loader, val_loader, device, SAVE_DIR)


def eval():

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
    SAVE_DIR  = "/home/ubuntu/TREDNet/data/phase_one/model_runs/model_v3/"

    # train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE, label_num=100)
    train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE)

    # model = model_builder.sequential_model_small(label_size=10)
    # model = model_builder.sequential_model(label_size=100)
    model = model_builder.sequential_model(label_size=4560)
    model.to(device)

    checkpoint = torch.load(os.path.join(SAVE_DIR, "checkpoint.pt"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # aurocs = trainer.get_auroc(model, val_loader, device)
    # print("val", aurocs.mean())

    aurocs = trainer.get_auroc(model, test_loader, device, on_cpu=True)
    print("test", aurocs.mean())

    # aurocs = trainer.get_auroc(model, train_loader, device)
    # print("train", aurocs.mean())


if __name__ == "__main__":

    # mnist_train()
    # eval()
    main()

    # parser = argparse.ArgumentParser(description='Train phase-one model.')
    # parser.add_argument('--model-dir', type=str, default="./",
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=60, metavar='E',
    #                     help='number of epochs to train (default: 60)')
    # parser.add_argument('--patience', type=int, default=10, metavar='P',
    #                     help='number of epochs to train (default: 60)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Don\'t use CUDA')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()
    #
    # main(args)

