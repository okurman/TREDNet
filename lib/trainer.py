import torch
import torch.nn as nn
import tqdm
import numpy as np
from torcheval.metrics import BinaryBinnedAUROC
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import os
import pandas as pd


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


def get_auroc(model, data_loader, device, bins=100, on_cpu=False):

    model.eval()

    y_pool = []
    y_pred_pool = []

    with torch.no_grad():
        with tqdm.tqdm(data_loader, unit="batch") as bar:
            bar.set_description(f"auROC")
            for i, (x, y) in enumerate(bar):

                if i == 10: break

                x = x.to(device)
                y_pred = model(x)

                if on_cpu:
                    y = y.to("cpu")
                    y_pred = y_pred.to("cpu")

                y_pool.append(y)
                y_pred_pool.append(y_pred)

    y = torch.cat(y_pool, axis=0).t()
    y_pred = torch.cat(y_pred_pool, axis=0).t()

    # try:
    #     metric = BinaryBinnedAUROC(num_tasks=y.shape[0], threshold=bins)
    #     metric.update(y_pred, y)
    #     (aurocs, thr) = metric.compute()
    #
    # except RuntimeError:

    aurocs = []
    for i in range(y.shape[0]):
        _y = y[i, :]
        _y_pred = y_pred[i, :]
        metric = BinaryBinnedAUROC(num_tasks=1, threshold=bins)
        metric.update(_y_pred, _y)
        (_auroc, _) = metric.compute()
        aurocs.append(_auroc)

    aurocs = torch.FloatTensor(aurocs)

    return aurocs


def run_validation(model, data_loader, device, criterion, stop_at=None):

    model.eval()

    val_loss = 0
    acc_list = []

    with tqdm.tqdm(data_loader, unit="batch") as bar:
        bar.set_description(f"Validation")
        for i, (x, y) in enumerate(bar):

            if stop_at and stop_at == i: break

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            val_loss += loss.item()
            batch_acc = (y_pred.round() == y).float().mean()
            acc_list.append(batch_acc.to("cpu"))

    acc = 100 * np.mean(acc_list)
    val_loss /= len(data_loader)

    return acc, val_loss


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
        best_loss = -1

    early_stop_thresh = 10

    for epoch in range(start_epoch, max_epoch):

        model.train()
        acc_list = []
        train_loss = 0

        with tqdm.tqdm(train_loader, unit="batch") as bar:
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

                if i % 100 == 0:
                    bar.set_postfix(loss=f"{float(loss):4.2f}", acc=f"{100 * batch_acc:2.3f}")

        train_loss /= len(train_loader)
        train_acc = 100 * np.mean(acc_list)

        with torch.no_grad():
            val_acc, val_loss = run_validation(model, val_loader, device, criterion, stop_at=None)
            # aurocs = get_auroc(model, val_loader, device)
            # val_auroc = aurocs.mean().float()

        print(f"tr acc: {train_acc:2.2f} "
              f"val acc:{val_acc:2.2f} "
              f"tr loss: {train_loss:.8f} "
              f"val loss: {val_loss:.8f}.")
              # f"val auroc: {val_auroc:.3f}.")

        history.loc[epoch] = [epoch, train_loss, train_acc, val_loss, val_acc]
        history.to_csv(history_file, sep="\t", index=False)

        save_checkpoint(model, optimizer, epoch, checkpoint_file, val_loss, val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_model_file, val_loss, val_acc)

        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break
