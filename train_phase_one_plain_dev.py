#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
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
import sys
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 600
EARLY_STOP_THRESH = 10


torch.manual_seed(1024)


class PhaseOne(nn.Module):
    def __init__(self, label_size=1924, mp=0, do=0.2):
        super().__init__()

        modules = [nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding="same"),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(in_channels=320, out_channels=320, kernel_size=8, padding="same"),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(do),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding="same"),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Conv1d(in_channels=480, out_channels=480, kernel_size=8, padding="same"),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Dropout(do),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding="same"),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.Conv1d(in_channels=960, out_channels=960, kernel_size=8, padding="same"),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.Dropout(do)]

        if mp == 2:
            modules += [
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(59520, label_size)]
        elif mp == 3:
            modules += [
                nn.MaxPool1d(3),
                nn.Flatten(),
                nn.Linear(39360, label_size)]
        elif mp == 4:
            modules += [
                nn.MaxPool1d(4),
                nn.Flatten(),
                nn.Linear(29760, label_size)]
        else:
            modules += [
                nn.Flatten(),
                nn.Linear(120000, label_size)
            ]
        
        modules += [nn.ReLU(),
            nn.Linear(label_size, label_size)]
        
        self.net = nn.Sequential(*modules)
    
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


def save_checkpoint(model, optimizer, epoch, filename, val_loss, train_loss):

    print(f"Saving a checkpoint to: {filename}")

    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_class': model.__class__,
        'model_state_dict': model_state_dict,
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_loss},
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True)
    
    return train_loader, val_loader


def get_auroc(model, data_loader, device, bins=20, on_cpu=False, max_iter=-1):
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
    
    with tqdm(data_loader, unit="batch") as bar:
        bar.set_description(f"Validation")
        for i, (x, y) in enumerate(bar):

            if max_iter > 0 and max_iter == i: break

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            val_loss += loss.item()
            
    val_loss /= len(data_loader)

    return val_loss


def train_model(model, optimizer, train_loader, val_loader, device, args):

    model_dir = args.d

    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler('cuda')
    
    best_model_file = os.path.join(model_dir, "best_model.pt")
    best_epoch = -1
    best_loss = 10**8
    
    history_file = os.path.join(model_dir, "history.tsv")
    if os.path.exists(history_file):
        history = pd.read_csv(history_file, sep="\t")
    else:
        history = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    history.to_csv(history_file, sep="\t", index=False)

    model.to(device)

    for epoch in range(1, NUM_EPOCHS+1):
        
        model.train()
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
                
                bar.set_postfix(loss=f"{float(loss):4.6f}")

        train_loss /= len(train_loader)
        
        with torch.no_grad():
            val_loss = run_validation(model, val_loader, device, criterion)
            # aurocs = get_auroc(model, val_loader, device)
            # val_auroc = aurocs.mean().numpy()
            # val_auroc = -1
        
        print(f"tr loss: {train_loss:.8f} "
              f"val loss: {val_loss:.8f}. ")
        
        history.loc[epoch] = [epoch, train_loss, val_loss]
        history.to_csv(history_file, sep="\t", index=False)

        if epoch == 0 or val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_model_file, val_loss, train_loss)

        elif epoch - best_epoch > EARLY_STOP_THRESH:
            print("Early stopped training at epoch %d" % epoch)
            break
        
        gc.collect()

        print("\n\n")


def eval(args):

    model_dir = args.d

    checkpoint_file = os.path.join(model_dir, "best_model.pt")
    assert os.path.exists(checkpoint_file)

    data_file = args.f
    assert os.path.exists(data_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset  = HDF5Dataset(data_file, "test_data", "test_labels")
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)
    
    model = PhaseOne()
    checkpoint = torch.load(checkpoint_file, encoding="ascii", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    aurocs = get_auroc(model, test_loader, device)
    print("Test set auROC:", aurocs.mean())


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = torch.cuda.device_count()

    model = PhaseOne(mp=args.mp, do=args.do)
    
    # print(model.net)
    # summary(model, (4, 1000))
    
    if gpus > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    if not os.path.exists(args.d):
        os.mkdir(args.d)

    train_loader, val_loader = load_dataset_wrappers(args.f, batch_size=BATCH_SIZE*gpus)
    
    if args.opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.opt == "SGD":
        # from Sei model: 
        # lr=0.1, "weight_decay": 1e-7, "momentum": 0.9
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, 
                                    weight_decay=args.l2,
                                    momentum=0.9,
                                    nesterov=True)
    if args.opt == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
    
    train_model(model, optimizer, train_loader, val_loader, device, args)
    

def parse_args():

    parser = argparse.ArgumentParser(description='Train phase-one model.')
    
    parser.add_argument('-d', type=str, required=True,help='Directory to save model')
    parser.add_argument('-f', type=str, required=True, help='Dataset file')
    parser.add_argument('-eval', action="store_true", help='Run evaluation of the trained model')

    parser.add_argument('-mp', type=int, default=0, help='Max pooling layer at the end')
    parser.add_argument('-opt', choices=['SGD', 'Adam', 'Adadelta'], default="Adam")
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-l2', type=float, default=1e-4, help='L2 regularizer')
    parser.add_argument('-do', type=float, default=0.3, help='Dropout')

    args = parser.parse_args()
    assert os.path.exists(args.d) and os.path.exists(args.f)

    args_list = [[arg, getattr(args, arg)] for arg in vars(args)]
    print("\nRunning with args:")
    print(tabulate(args_list))
    print("\n\n")

    return args


if __name__ == "__main__":

    args = parse_args()

    if args.eval:
        eval(args)
        sys.exit()
    
    main(args)
    