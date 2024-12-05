#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import BinaryBinnedAUROC
import pandas as pd
from tqdm.auto import tqdm
import gc

# from lib.trainer import train_model
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 640


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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,  num_workers=2, prefetch_factor=2)

    return train_loader, val_loader, test_loader


def get_auroc(model, data_loader, device, bins=100, on_cpu=False, max_iter=-1):
    model.eval()

    y_pool = []
    y_pred_pool = []

    with torch.no_grad():
        with tqdm(data_loader, unit="batch") as bar:
            bar.set_description(f"y_pred")
            for i, (x, y) in enumerate(bar):

                if max_iter > 0 and max_iter == i: break

                x = x.to(device)
                y_pred = model(x)

                if on_cpu:
                    y = y.to("cpu")
                    y_pred = y_pred.to("cpu")

                y_pool.append(y)
                y_pred_pool.append(y_pred)

    y = torch.cat(y_pool, axis=0).t()
    y_pred = torch.cat(y_pred_pool, axis=0).t()

    if not on_cpu:
        y = y.to("cuda")

    aurocs = []
    with tqdm(range(y.shape[0]), unit="columns") as bar:
        bar.set_description(f"auROCs")
        for i in bar:
            _y = y[i, :]
            _y_pred = y_pred[i, :]
            metric = BinaryBinnedAUROC(threshold=bins)
            metric.update(_y_pred, _y)
            (_auroc, _) = metric.compute()
            aurocs.append(_auroc)

    aurocs = torch.FloatTensor(aurocs)

    return aurocs


def run_validation(model, data_loader, device, criterion, max_iter=-1):
    model.eval()

    val_loss = 0
    acc_list = []

    with tqdm(data_loader, unit="batch") as bar:
        bar.set_description(f"Validation")
        for i, (x, y) in enumerate(bar):

            if max_iter > 0 and max_iter == i: break

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            val_loss += loss.item()
            batch_acc = (y_pred.round() == y).float().mean()
            acc_list.append(batch_acc.to("cpu"))

    acc = 100 * np.mean(acc_list)
    val_loss /= len(data_loader)

    return acc, val_loss


def train_model(model, train_loader, val_loader, device,  model_dir, max_epoch=100, max_iter=-1):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    checkpoint_file = os.path.join(model_dir, "checkpoint.base.pt")
    save_checkpoint(model, optimizer, 0, checkpoint_file, 0, 0)

    history_file = os.path.join(model_dir, "history.tsv")
    if os.path.exists(history_file):
        history = pd.read_csv(history_file, sep="\t")
    else:
        history = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auroc"])

    model.to(device)

    history.to_csv(history_file, sep="\t", index=False)

    for epoch in range(1, max_epoch+1):

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

                if max_iter > 0 and max_iter == i: break

                bar.set_postfix(loss=f"{float(loss):4.6f}", acc=f"{100 * batch_acc:2.2f}")

        train_loss /= len(train_loader)
        train_acc = 100 * np.mean(acc_list)

        with torch.no_grad():
            val_acc, val_loss = run_validation(model, val_loader, device, criterion, max_iter)
            aurocs = get_auroc(model, val_loader, device, max_iter=max_iter)
            val_auroc = aurocs.mean().numpy()

        print(f"tr acc: {train_acc:2.2f} "
              f"val acc:{val_acc:2.2f} "
              f"tr loss: {train_loss:.8f} "
              f"val loss: {val_loss:.8f}. "
              f"val auroc: {val_auroc:.3f}.")

        checkpoint_file = os.path.join(model_dir, f"checkpoint.{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, checkpoint_file, val_loss, val_acc)

        history.loc[epoch] = [epoch, train_loss, train_acc, val_loss, val_acc, val_auroc]
        history.to_csv(history_file, sep="\t", index=False)

        gc.collect()

        print("\n\n")


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhaseOne()
    model.to(device)
    # summary(model, (4, 1000))

    # DATA_FILE = "/data/Dcode/sanjar/Projects/TREDNet/data/tensors/data_0.25.hdf5"
    # SAVE_DIR  = "/data/Dcode/sanjar/Projects/TREDNet/data/model_runs/v1"

    DATA_FILE = args.f
    SAVE_DIR = args.d
    

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE)
    
    train_model(model, train_loader, val_loader, device, SAVE_DIR, max_iter=-1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train phase-one model.')
    parser.add_argument('-d', type=str, default="./",
                        help='Directory to save model')
    parser.add_argument('-f', type=str, default="./",
                        help='Dataset file')
    args = parser.parse_args()

    main(args)


    # test 
    # test2a


