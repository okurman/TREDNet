import argparse
import os
import sys
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-data-file', dest="data_file", default="data/phase_two_dataset.hdf5", help="Dataset file created by data_prep.py")
parser.add_argument('-save-dir', dest="save_dir", default="trained_model")
args = parser.parse_args()

if len(sys.argv) < 3:
    parser.print_help()
    sys.exit()

data_file = args.data_file
save_dir = args.save_dir

if not os.path.exists(data_file):
    print("Data file not found: ", data_file)
    sys.exit()

if os.path.exists(save_dir):
    print("Directory already exists: ", save_dir)
    print("Choose another destination.")
    sys.exit()
else:
    os.mkdir(save_dir)

from src import models

print("Launching the training of model")
print("Model files and performance evaluation results will be written in:")
print("        " + save_dir)
with h5py.File(data_file, "r") as data:
    models.run_model(data, save_dir)

