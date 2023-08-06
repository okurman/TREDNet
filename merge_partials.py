#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import os
import glob
import argparse

import numpy as np


def merge_deltas(data_dir, save_file=None):

	partial_files = glob.glob(os.path.join(data_dir, "partial*.hdf5"))
	save_file = os.path.join(data_dir, save_file if save_file else "normalized_deltas.hdf5")

	print("Saving to file: %s" % save_file)

	with h5py.File(save_file, "w") as of:
		for f in partial_files:
			print("\t\t Reading file: %s" % f)
			with h5py.File(f, "r") as inf:
				for k in inf.keys():
					of.create_dataset(name=k, data=inf[k])


# def merge_features(data_dir, save_file=None):




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-data-dir', dest="data_dir", required=True)
	parser.add_argument('-delta', action="store_true", help="If merging the delta files")
	parser.add_argument('-features', action="store_true", help="If merging the features for peak detection")
	parser.add_argument('-save-file', dest="save_file", default=None, help="Name of the file to be saved in -data-dir")

	args = parser.parse_args()

	if args.delta:
		merge_deltas(args.data_dir, args.save_file)
	elif args.features:
		pass

	# parser.add_argument('-delta-file', dest="delta_file", default=None, help="hdf5 file containing the delta scores")
