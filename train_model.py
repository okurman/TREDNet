import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-data-file', dest="data_file", default="data/phase_two_dataset.hdf5")
parser.add_argument('-save-dir', dest="save_dir", default="trained_model")
args = parser.parse_args()

data_file = args.data_file
save_dir = args.save_dir

if not os.path.exists(data_file):
    print("Data file not found: ", data_file)
    sys.exit()

if os.path.exists(save_dir):
    print("Directory already exists: ", data_file)
    print("Choose another destination.")
    sys.exit()
else:
    os.mkdir(save_dir)

from src import models
from src import data_prep

print("Loading the dataset from:", data_file)
data = data_prep.load_dataset(data_file)

print("Launching the training of model")
print("Model files and performance evaluation results will be written in:")
print("        " + save_dir)
models.run_model(data, save_dir)

