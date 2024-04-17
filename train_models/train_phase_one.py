#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import h5py
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# Hyper-parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001


sorted_ix_200 = [2067,  623, 1513,  622, 1332, 1911,  624, 1910, 1751, 1917, 1147,
       2068, 2073, 2070, 1987, 2066, 1331, 1326, 1418,  872, 1981,  867,
       1986, 1570, 1095,  627, 1395, 1375, 1370, 1853, 2074, 1413,  633,
        575,  730, 1515,  893, 1510, 1325, 1028, 1137, 1916,  834, 1600,
       1511, 1750,  588, 1412,  841, 1870, 2063,  997, 1993,  727, 1550,
        848,  959,  740,  590,  576,  926, 1019, 1416,  692, 1368, 1297,
       1041, 1345, 1836, 1512, 1429, 1710, 1107,  628, 2052, 1672, 1767,
       1383, 1554, 1886,  847, 1755,  936, 1892,  968, 1944, 1077,  537,
       2062, 1764,  634, 1199, 1818, 1090,  919,  906, 1006, 2054, 1984,
       1507,  991, 1146,  723, 1852, 2048,  680, 1372, 1141, 1532,  840,
       1930, 1844, 1094, 1896, 1753,  735, 1301,  946,  743, 1551, 1369,
       1616, 1190, 1423,  693, 1608, 1723, 2065,  933, 1464,  833, 1039,
        426,  330, 1901,  990, 1493, 1909, 1393, 1850,  577, 1868, 1387,
        532,  300, 1978, 1713, 1103, 1089,  554, 1914,  608, 1264, 1655,
       1898, 1527, 1382, 1140, 1533, 2064, 1473, 1757, 1075, 1155, 1052,
        823,  814,  482, 1329,  489, 1456, 1878, 1125,  528,  369, 1756,
       1188,  856,  492, 1943,  530,  696, 1990, 1908, 1054,  649,  939,
       1315, 1348, 1642,  866, 1337,  527, 1575, 1576, 1074,  604,  888,
        571, 1042]

sorted_ix_200 = np.asarray(sorted_ix_200)
feat_idx = np.sort(sorted_ix_200[:10])


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=20, kernel_size=1)
        self.bn = nn.BatchNorm1d(10)
        self.pool = nn.MaxPool1d(2, 2)
        
        self.conv2 = nn.Conv1d(20, 10, 5)
        self.fc1 = nn.Linear(2460, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x)) # -> N, 20, 993
        x = self.pool(x)          # -> N, 20, 496
        x = F.relu(self.conv2(x)) # -> N, 10, 492
        x = self.bn(x),
        x = self.pool(x)          # -> N, 10, 246
        x = torch.flatten(x, 1)   # -> N, 2460
        x = F.relu(self.fc1(x))   # -> N, 64
        x = F.relu(self.fc2(x))   # -> N, 10
        x = torch.sigmoid(x)      # -> N, 10
        return x


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
        _y = np.asarray(self.Y[idx, feat_idx], dtype=np.float32)

        return _x, _y


def checkpoint(model, optimizer, filename):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def load_dataset_wrappers():

    # train_file = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.train.h5"
    # val_file   = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.val.h5"
    # test_file  = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.test.h5"
    data_file = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"

    train_dataset = HDF5Dataset(data_file, "X_train", "Y_train")
    val_dataset = HDF5Dataset(data_file, "X_val", "Y_val")
    test_dataset = HDF5Dataset(data_file, "X_test", "Y_test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    train_loader = iter(train_loader)
    val_loader = iter(val_loader)
    test_loader = iter(test_loader)

    return train_loader, val_loader, test_loader


def main(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    DATA_FILE = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.h5"
    SAVE_DIR = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/"

    train_loader, val_loader, test_loader = load_dataset_wrappers()

    model = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    n_total_steps = len(train_loader)
    for epoch in range(10):

        with tqdm.tqdm(train_loader, unit="batch") as bar:
            bar.set_description(f"Epoch {epoch}")

            running_loss = 0.0

            for i, (x, y) in enumerate(bar):

                if i == 100:
                    break

                y_pred = model(x)
                acc = (y_pred.round() == y).float().mean()
                loss = criterion(y_pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                bar.set_postfix(loss=float(loss), acc=f"{float(acc) * 100:.2f}%")

        print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.6f}')

        epoch_model_file = SAVE_DIR + f"/epoch-{epoch}.pth"
        print("Saving the epoch model file to:")
        print(epoch_model_file)
        checkpoint(model, epoch_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train phase-one model.')
    parser.add_argument('--model-dir', type=str, default="./",
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Don\'t use CUDA')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    main(args)

