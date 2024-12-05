#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm

# from lib.trainer import train_model
# from torchsummary import summary

# import warnings
# warnings.filterwarnings("ignore")
# torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters
# NUM_EPOCHS = 30
NUM_EPOCHS = 100
BATCH_SIZE = 640
# BATCH_SIZE = 12

torch.manual_seed(1024)


class PhaseOne(nn.Module):
    def __init__(self, label_size=1924):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=320, out_channels=320, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=480, out_channels=480, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=480, out_channels=640, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=640, out_channels=640, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),

            # in the original TREDNet there's no MaxPooling here. This is to reduce the number of params.
            # nn.MaxPool1d(3),
            
            nn.Flatten(),
            nn.LazyLinear(label_size),
            nn.ReLU(),
            nn.Linear(label_size, label_size),
            nn.Sigmoid())

    def forward(self, x):

        y = self.net(x)

        return y


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


def save_checkpoint(model, optimizer, epoch, filename, val_loss, val_acc):
    print(f"Saving a checkpoint to: {filename}")
    torch.save({
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_class': model.__class__,
        'model_state_dict': model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc},
    filename)


def load_checkpoint(model, optimizer, filename):

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint["epoch"]

    return epoch


def load_dataset_wrappers(data_file):

    train_dataset = HDF5Dataset(data_file, "train_data", f"train_labels")
    val_dataset   = HDF5Dataset(data_file, "validation_data", f"validation_labels")
    test_dataset  = HDF5Dataset(data_file, "test_data", f"test_labels")

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, prefetch_factor=2)
    # val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, prefetch_factor=2)
    # test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, prefetch_factor=2)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, prefetch_factor=2)
    # test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,  num_workers=2, prefetch_factor=2)
    test_loader  = None

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device,  model_dir, max_epoch=100):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=5e-4)
    
    checkpoint_file = os.path.join(model_dir, "checkpoint.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_file):
        _epoch = load_checkpoint(model, optimizer, checkpoint_file)
        start_epoch = _epoch + 1

    history_file = os.path.join(model_dir, "history.tsv")
    if os.path.exists(history_file):
        history = pd.read_csv(history_file, sep="\t")
    else:
        history = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    model.to(device)

    history["epoch"] = pd.to_numeric(history["epoch"]).astype(int)
    history.to_csv(history_file, sep="\t", index=False)

    best_model_file = os.path.join(model_dir, "best_model.pt")
    
    if os.path.exists(best_model_file):
        _checkpoint = torch.load(best_model_file)
        best_epoch = _checkpoint["epoch"]
        best_loss = _checkpoint["val_loss"]
    else:
        best_epoch = -1
        best_loss = 10**8

    early_stop_thresh = 10

    for epoch in range(start_epoch, max_epoch):

        model.train()
        acc_list = []
        train_loss = 0

        with tqdm(train_loader, unit="batch", position=0, leave=True) as bar:
            bar.set_description(f"Epoch {epoch}")

            for i, (x, y) in enumerate(bar):
                
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()

                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                batch_acc = (y_pred.round() == y).float().mean()
                acc_list.append(batch_acc.to("cpu"))

                # if i % 10000 == 0:
                if i == 500:
                    break
                
                # if i % 100 == 0:
                bar.set_postfix(loss=f"{float(loss):4.6f}", acc=f"{100 * batch_acc:2.2f}")

        train_loss /= len(train_loader)
        train_acc = 100 * np.mean(acc_list)

        with torch.no_grad():
            val_acc, val_loss = run_validation(model, val_loader, device, criterion)
            aurocs = get_auroc(model, val_loader, device)
            val_auroc = aurocs.mean().float()

        print(f"tr acc: {train_acc:2.2f} "
              f"val acc:{val_acc:2.2f} "
              f"tr loss: {train_loss:.8f} "
              f"val loss: {val_loss:.8f}."
              f"val auroc: {val_auroc:.3f}.")

        history.loc[epoch] = [epoch, train_loss, train_acc, val_loss, val_acc]
        history.to_csv(history_file, sep="\t", index=False)
        
        if epoch == 0 or val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_model_file, val_loss, val_acc)

        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break


def eval():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/data_all_uncompressed.hdf5"
    SAVE_DIR  = "/home/ubuntu/TREDNet/data/phase_one/model_runs/model_v4/"

    test_dataset  = HDF5Dataset(DATA_FILE, "test_data", "test_labels")
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)

    checkpoint_file = os.path.join(SAVE_DIR, "checkpoint.pt")
    
    model = PhaseOne()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=5e-4)
    epoch = load_checkpoint(model, optimizer, checkpoint_file)
    
    model.to(device)

    aurocs = get_auroc(model, test_loader, device)
    print("Test set auROC:", aurocs.mean())


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhaseOne()
    model.to(device)
    # summary(model, (4, 1000))

    # DATA_FILE = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
    # SAVE_DIR = "/Users/okurman/Projects/TREDNet/data/phase_one/model_runs/model_v1/"

    # DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
    DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/data_all_uncompressed.hdf5"
    SAVE_DIR  = "/home/ubuntu/TREDNet/data/phase_one/model_runs/model_v1/"
    
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE)

    train_model(model, train_loader, val_loader, device, SAVE_DIR)




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


    # test 
    # test2a


