
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseOne(nn.Module):
    def __init__(self, label_size=4560):
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


def sequential_model(label_size=10):

    model = nn.Sequential(
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
        nn.MaxPool1d(2),

        nn.Flatten(),
        nn.LazyLinear(label_size),
        nn.ReLU(),
        nn.Linear(label_size, label_size),
        nn.Sigmoid()
    )

    return model

