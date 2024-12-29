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
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchsummary import summary
import gc
from tqdm import tqdm
import sys
from tabulate import tabulate
import shutil

import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from lightning.pytorch import seed_everything
seed_everything(1024, workers=True)


## Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 1000
EARLY_STOP_THRESH = 30
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


class PhaseOneDataModule(L.LightningDataModule):
    def __init__(self, data_file, batch_size=800):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = HDF5Dataset(self.data_file, "train_data", "train_labels")
            self.val_dataset = HDF5Dataset(self.data_file, "validation_data", "validation_labels")
        if stage == "test":
            self.test_dataset = HDF5Dataset(self.data_file, "test_data", "test_labels")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                        num_workers=8, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                        num_workers=2, prefetch_factor=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                        num_workers=4, prefetch_factor=2)


class PhaseOneLightning(L.LightningModule):
    def __init__(self, label_size=1924, mp=0, do=0.2, learning_rate=1e-4, weight_decay=1e-4, optimizer="Adam"):
        super().__init__()
        self.save_hyperparameters()
        self.model = PhaseOne(label_size=label_size, mp=mp, do=do)
        self.criterion = nn.BCEWithLogitsLoss()

        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        y_pred = torch.sigmoid(y_pred)
        precision = self.precision(y_pred, y)
        self.log('train_prec', precision, prog_bar=False, on_step=False, on_epoch=True)
        
        recall = self.recall(y_pred, y)
        self.log('train_recall', recall, prog_bar=False, on_step=False, on_epoch=True)

        f1 = self.f1(y_pred, y)
        self.log('train_f1', f1, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        y_pred = torch.sigmoid(y_pred)
        
        precision = self.precision(y_pred, y)
        self.log('val_prec', precision, prog_bar=False, on_step=False, on_epoch=True)

        recall = self.recall(y_pred, y)
        self.log('val_recall', recall, prog_bar=False, on_step=False, on_epoch=True)
        
        f1 = self.f1(y_pred, y)
        self.log('val_f1', f1, prog_bar=False, on_step=False, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay,
                # defaulf value of 1e-08 caused NaN loss values when used with lr=0.1
                # as a result of 16-bit mixed presicion training.
                eps=1e-06 if self.hparams.learning_rate == 0.1 else 1e-08
            )
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        elif self.hparams.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(), 
                lr = self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay)
        
        print("\n\n")
        print("##############################")
        print("####### OPTIMIZER  ###########")
        print(optimizer)
        print("##############################")
        print("\n\n")

        if self.hparams.optimizer == "Adadelta":

            return optimizer
        
        else:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   mode='min', 
                                                                   factor=0.3, 
                                                                   patience=5,
                                                                   verbose=True)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch", # default
                    "frequency": 1, # default
                },
            }


def get_auroc(model, data_loader, device, on_cpu=False, max_iter=-1):

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
                y_pred = torch.sigmoid(y_pred)
                
                if on_cpu:
                    y = y.to("cpu")
                    
                y_pool.append(y)
                y_pred_pool.append(y_pred)

    y = torch.cat(y_pool, axis=0).t()
    y_pred = torch.cat(y_pred_pool, axis=0).t()

    if not on_cpu:
        y = y.to("cuda")
        y_pred = y_pred.to("cuda")

    aurocs = []
    with tqdm(range(y.shape[0]), unit="columns") as bar:
        bar.set_description(f"auROCs")
        for i in bar:
            _y = y[i, :]
            _y_pred = y_pred[i, :]
            
            metric = BinaryAUROC()
            _y = _y.to(device)
            _auroc = metric(_y_pred, _y)

            aurocs.append(_auroc)

    aurocs = torch.FloatTensor(aurocs)

    return aurocs


def eval(args, data_prefix="test"):

    assert os.path.exists(args.d) and os.path.exists(args.f)

    checkpoint_file = os.path.join(args.d, "best_model.ckpt")
    save_file = os.path.join(args.d, f"{data_prefix}_set.ROC.txt")
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file doesn't exist: \n {checkpoint_file}")

    if os.path.exists(save_file):
        checkpoint_mtime = os.path.getmtime(checkpoint_file)
        aucs_mtime = os.path.getmtime(save_file)
        if checkpoint_mtime < aucs_mtime:
            print(f"No new model has been saved in best_model.ckpt since the {save_file} was generated")
            return
    
    model = PhaseOneLightning.load_from_checkpoint(checkpoint_file)
    
    dataset  = HDF5Dataset(args.f, f"{data_prefix}_data", f"{data_prefix}_labels")
    data_loader  = DataLoader(dataset,  batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)
    
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    
    aurocs = get_auroc(model, data_loader, device)
    print("Test set auROC:", aurocs.mean())
    
    with open(save_file, "w") as of:
        of.write(f"auROC \t{aurocs.mean()}\n")


def copy_to_ssd(args):

    ssd_path = "/lscratch/%s" % (os.environ.get("SLURM_JOB_ID"))
    if os.path.exists(ssd_path):
        new_f = os.path.join(ssd_path, os.path.basename(args.f))
        if not os.path.exists(new_f):
            print("Copying the data file")
            print(f"From: {args.f}")
            print(f"To: {new_f}\n")
            shutil.copyfile(args.f, new_f)

        return new_f
    
    return args.f


def main(args):
    
    # data_file = copy_to_ssd(args)
    data_file = args.f

    data_module = PhaseOneDataModule(data_file, batch_size=args.batch)
    
    model = PhaseOneLightning(mp=args.mp, do=args.do, learning_rate=args.lr, weight_decay=args.l2, optimizer=args.opt)

    early_stopping =  EarlyStopping(monitor='val_loss', 
                                    patience=args.estop, 
                                    mode='min', 
                                    verbose=True)
    checkpoint_callback = ModelCheckpoint(dirpath=args.d, 
                                          filename='best_model',
                                          save_top_k=1, 
                                          monitor='val_loss', 
                                          mode='min', 
                                          verbose=True,
                                          save_last=True)
    
    callbacks = [checkpoint_callback] if args.ne else [early_stopping, checkpoint_callback]
    csv_logger = CSVLogger(args.d, name="training_logs")
    tb_logger = TensorBoardLogger(args.d, name="training_logs")

    ###################################
    ##########  Resume block ##########
    ###################################
    if args.resume:
        checkpoint_file = os.path.join(args.d, "best_model.ckpt")
        trainer = L.Trainer(max_epochs=args.epochs, 
                            callbacks=callbacks,
                            logger=[tb_logger, csv_logger],
                            detect_anomaly=False)
        trainer.fit(model, data_module, ckpt_path=checkpoint_file)

        return

    ##############################################
    ########## New model training block ##########
    ##############################################

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        strategy='ddp', # it is the default value.
        devices='auto',
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        precision='16-mixed',
        log_every_n_steps=500,
        # detect_anomaly=True,
        deterministic=True
    )
    
    print("\n\n")
    print("===== Layers of built model: =====")
    print("==================================")
    print(model.model.net)
    print("==================================\n")
    print("\n\n")

    # Train the model
    trainer.fit(model, data_module)


def parse_args():

    parser = argparse.ArgumentParser(description='Train phase-one model.')
    
    parser.add_argument('-d', type=str, required=True,help='Directory to save model')
    parser.add_argument('-f', type=str, required=True, help='Dataset file')
    parser.add_argument('-eval', action="store_true", help='Run evaluation of the trained model')
    parser.add_argument('-resume', action="store_true", help='Resume the training task from the checkpoint')

    parser.add_argument('-mp', type=int, default=0, help='Max pooling layer at the end')
    parser.add_argument('-lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('-l2', type=float, default=1e-5, help='L2 regularizer')
    parser.add_argument('-do', type=float, default=0.2, help='Dropout')
    parser.add_argument('-ne', action="store_true", help='Exclude early stopping callback from training')
    
    parser.add_argument('-opt', choices=['SGD', 'Adam', 'Adadelta', 'AdamW'], default="Adam")

    parser.add_argument('-epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-estop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('-batch', type=int, default=1000, help='Batch size')
    
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()
    assert os.path.exists(args.d) and os.path.exists(args.f)

    args_list = [[arg, getattr(args, arg)] for arg in vars(args)]
    print("\nRunning with args:")
    print(tabulate(args_list))
    print("\n\n")

    return args


if __name__ == "__main__":

    print("#########################")
    print("### Python/CUDA envs  ###")
    print(f"Python interpreter: {sys.executable}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print("#########################")
    print("\n\n")
    
    args = parse_args()
    checkpoint_file = os.path.join(args.d, "best_model.ckpt")

    if args.eval:
        if not os.path.exists(checkpoint_file):
            print(f"No checkpoint file was found to run evaluation: {checkpoint_file}")
            sys.exit(1)

        eval(args, data_prefix="test")
        # eval(args, data_prefix="validation")
        sys.exit(1)
    
    if not os.path.exists(args.f):
        print(f"Data file path doesn't exist: {args.f}")
        sys.exit(1)
    
    if not os.path.exists(args.d):
        print(f"Directory path doesn't exist: {args.d}")
        sys.exit(1)

    if args.resume:
        
        if not os.path.exists(checkpoint_file):
            print(f"No checkpoint file was found to resume from: {checkpoint_file}")
            sys.exit(1)

    main(args)



