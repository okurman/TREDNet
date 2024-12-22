from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Any
import h5py
import argparse
import os
from time import perf_counter
from tqdm.auto import tqdm
import gc
import sys
from tabulate import tabulate


# import h5py
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torcheval.metrics import BinaryBinnedAUROC, BinaryAUROC
# from torchsummary import summary
# from torch.amp import autocast, GradScaler
# import pandas as pd


import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Hyper-parameters 
NUM_EPOCHS = 100
BATCH_SIZE = 100
EARLY_STOP_THRESH = 15


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


class DNASequenceDataset:
    def __init__(self, data_file: str, split: str):
        self.inf = h5py.File(data_file, "r")
        self.X = self.inf[f"{split}_data"]
        self.Y = self.inf[f"{split}_labels"]

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        return {
            "data": np.asarray(self.X[idx], dtype=np.float32).swapaxes(0, 1),
            "labels": np.asarray(self.Y[idx], dtype=np.float32)
        }

    def to_huggingface_dataset(self):
        # Convert to format expected by HF Trainer
        return Dataset.from_generator(
            self._generator,
            features={
                "data": {"shape": (4, self.X.shape[1]), "dtype": "float32"},
                "labels": {"shape": (1924,), "dtype": "float32"}
            }
        )
    
    def _generator(self):
        for i in range(len(self)):
            yield self[i]


class DNATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_values = inputs["data"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)
        
        outputs = model(input_values)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss


def train_with_huggingface(args):
    
    print("Dataset creation")
    train_dataset = DNASequenceDataset(args.f, "train").to_huggingface_dataset()
    val_dataset = DNASequenceDataset(args.f, "validation").to_huggingface_dataset()
    print("Done")
    
    # train_dataset_batched = train_dataset.map(lambda x: x, batched=True)
    # val_dataset_batched = val_dataset.map(lambda x: x, batched=True)
    
    print("Model creation")
    model = PhaseOne(mp=args.mp, do=args.do).to(torch.device("cuda"))
    print("Done")

    training_args = TrainingArguments(
        output_dir=args.d,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=args.lr,
        weight_decay=args.l2,
        logging_dir=f"{args.d}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Enable mixed precision training
        optim=args.opt.lower(),
    )
    
    trainer = DNATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(f"{args.d}/best_model")



def main(args):
    
    if not os.path.exists(args.d):
        os.makedirs(args.d)
    
    train_with_huggingface(args)



def parse_args():

    parser = argparse.ArgumentParser(description='Train phase-one model.')
    
    parser.add_argument('-d', type=str, required=True,help='Directory to save model')
    parser.add_argument('-f', type=str, required=True, help='Dataset file')
    parser.add_argument('-eval', action="store_true", help='Run evaluation of the trained model')

    parser.add_argument('-mp', type=int, default=0, help='Max pooling layer at the end')
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-l2', type=float, default=1e-4, help='L2 regularizer')
    parser.add_argument('-do', type=float, default=0.2, help='Dropout')
    
    parser.add_argument('-opt', choices=['SGD', 'Adam', 'Adadelta'], default="Adam")

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

