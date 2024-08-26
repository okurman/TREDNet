import argparse
import os
import sys
import h5py

from lib.v1 import models

parser = argparse.ArgumentParser()

# I/O logistics.
parser.add_argument('-data-file',
                    dest="data_file",
                    default="data/phase_two_dataset.hdf5",
                    help="Dataset file created by data_prep.py")
parser.add_argument('-save-dir',
                    dest="save_dir",
                    default="trained_model",
                    help="Directory where the files will be saved. Will exit if the directory exists.")

args = parser.parse_args()

if not os.path.exists(args.data_file):
    print("Data file not found: ", args.data_file)
    sys.exit()

with h5py.File(args.data_file, "r") as data:

    model = models.define_model()

    models.run_model(data=data, save_dir=args.save_dir)


print("Training completed!")