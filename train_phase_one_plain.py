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
from torchsummary import summary
from torch.amp import autocast, GradScaler
import pandas as pd
from tqdm.auto import tqdm
import gc

import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 1000
EARLY_STOP_THRESH = 15


torch.manual_seed(1024)



class PhaseOne(nn.Module):
    def __init__(self, label_size=1924):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding="same"),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(in_channels=320, out_channels=320, kernel_size=8, padding="same"),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding="same"),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Conv1d(in_channels=480, out_channels=480, kernel_size=8, padding="same"),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=480, out_channels=640, kernel_size=8, padding="same"),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Conv1d(in_channels=640, out_channels=640, kernel_size=8, padding="same"),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),

            nn.Linear(19840, label_size),
            nn.ReLU(),
            nn.Linear(label_size, label_size))

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


def load_dataset_wrappers(data_file, batch_size=BATCH_SIZE):

    train_dataset = HDF5Dataset(data_file, "train_data", f"train_labels")
    val_dataset   = HDF5Dataset(data_file, "validation_data", f"validation_labels")
    test_dataset  = HDF5Dataset(data_file, "test_data", f"test_labels")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True)
    # test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True)

    return train_loader, val_loader, None


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

    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adadelta(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.1, patience=5, 
                                                         verbose=True)
    
    scaler = GradScaler('cuda')
    
    checkpoint_file = os.path.join(model_dir, "checkpoint.base.pt")
    save_checkpoint(model, optimizer, 0, checkpoint_file, 0, 0)

    best_model_file = os.path.join(model_dir, "best_model.pt")
    best_epoch = -1
    best_loss = 10**8
    
    history_file = os.path.join(model_dir, "history.tsv")
    if os.path.exists(history_file):
        history = pd.read_csv(history_file, sep="\t")
    else:
        history = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auroc"])
    history.to_csv(history_file, sep="\t", index=False)

    model.to(device)

    for epoch in range(1, max_epoch+1):
        
        model.train()
        acc_list = []
        train_loss = 0
        
        with tqdm(train_loader, unit="batch", position=0, leave=True) as bar:
            bar.set_description(f"Epoch {epoch}")

            for i, (x, y) in enumerate(bar):
                
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()

                with autocast('cuda'):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                
                scaler.update()

                train_loss += loss.item()

                batch_acc = (y_pred.round() == y).float().mean()
                acc_list.append(batch_acc.to("cpu"))

                if max_iter > 0 and max_iter == i: break

                bar.set_postfix(loss=f"{float(loss):4.6f}", acc=f"{100 * batch_acc:2.2f}")

        train_loss /= len(train_loader)
        train_acc = 100 * np.mean(acc_list)

        with torch.no_grad():
            val_acc, val_loss = run_validation(model, val_loader, device, criterion, max_iter)
            # aurocs = get_auroc(model, val_loader, device, max_iter=max_iter)
            # val_auroc = aurocs.mean().numpy()
            val_auroc = -1

        scheduler.step(val_loss)
        print(f"tr acc: {train_acc:2.2f} "
              f"val acc:{val_acc:2.2f} "
              f"tr loss: {train_loss:.8f} "
              f"val loss: {val_loss:.8f}. "
              f"val auroc: {val_auroc:.3f}.")
        
        history.loc[epoch] = [epoch, train_loss, train_acc, val_loss, val_acc, val_auroc]
        history.to_csv(history_file, sep="\t", index=False)

        if epoch == 0 or val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_model_file, val_loss, val_acc)

        elif epoch - best_epoch > EARLY_STOP_THRESH:
            print("Early stopped training at epoch %d" % epoch)
            break
        
        gc.collect()

        print("\n\n")


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = torch.cuda.device_count()

    model = PhaseOne()

    if gpus > 1:
        model = nn.DataParallel(model)

    model.to(device)
    # summary(model, (4, 1000))

    # DATA_FILE = "/data/Dcode/sanjar/Projects/TREDNet/data/tensors/data_0.25.hdf5"
    # SAVE_DIR  = "/data/Dcode/sanjar/Projects/TREDNet/data/model_runs/v1"

    DATA_FILE = args.f
    SAVE_DIR = args.d
    

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    train_loader, val_loader, test_loader = load_dataset_wrappers(DATA_FILE, batch_size=BATCH_SIZE*gpus)

    # for x, y in train_loader:
    #     y_pred = model(x.to(device))
    #     print(x.shape, y.shape, y_pred.shape)
    # return
    
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


