#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import sys
import numpy as np
import os


def create_dataset(data_dir):

    chroms = set(["chr%d" % i for i in range(1, 23)])
    test_chroms = ["chr8", "chr9"]
    train_chroms = chroms.difference(test_chroms)

    print("Loading training set partials.")
    peak_features, dip_features, control_features = [], [], []
    train_files = [os.path.join(data_dir, "partial_%s.features.hdf5" % chrom) for chrom in train_chroms]
    for f in train_files:
        print("\t%s" % f)
        with h5py.File(f, "r") as inf:
            peak_features.append(inf["X_peak"][()])
            dip_features.append(inf["X_dip"][()])
            control_features.append(inf["X_control"][()])

    train_X_controls = np.vstack(control_features)
    train_X_peak = np.vstack(peak_features)
    train_X_dip = np.vstack(dip_features)
    print("Training set shapes")
    print(train_X_controls.shape, train_X_peak.shape, train_X_dip.shape)

    peak_features, dip_features, control_features = [], [], []
    train_files = [os.path.join(data_dir, "partial_%s.features.hdf5" % chrom) for chrom in test_chroms]
    print("Loading test set partials.")
    for f in train_files:
        print("\t%s" % f)
        with h5py.File(f, "r") as inf:
            peak_features.append(inf["X_peak"][()])
            dip_features.append(inf["X_dip"][()])
            control_features.append(inf["X_control"][()])

    test_X_controls = np.vstack(control_features)
    test_X_peak = np.vstack(peak_features)
    test_X_dip = np.vstack(dip_features)
    print("Test set shapes")
    print(test_X_controls.shape, test_X_peak.shape, test_X_dip.shape)

    for peak_type, _X_train_pos, _X_train_control, _X_test_pos, _X_test_control in zip(["peak", "dip"],
                                                                                   [train_X_peak, train_X_dip],
                                                                                   [train_X_controls, train_X_controls],
                                                                                   [test_X_peak, test_X_dip],
                                                                                   [test_X_controls, test_X_controls]):

        X_train = np.vstack((_X_train_pos, _X_train_control))
        Y_train = np.concatenate((np.ones(_X_train_pos.shape[0]), np.zeros(_X_train_control.shape[0])))

        ix = np.arange(_X_test_control.shape[0])
        np.random.shuffle(ix)
        ix = ix[:_X_test_pos.shape[0]]
        _X_test_control = _X_test_control[ix, :]

        assert _X_test_pos.shape == _X_test_control.shape

        X_test = np.vstack((_X_test_pos, _X_test_control))
        Y_test = np.concatenate((np.ones(_X_test_pos.shape[0]), np.zeros(_X_test_pos.shape[0])))

        ix = np.arange(X_train.shape[0])
        np.random.shuffle(ix)
        X_train = X_train[ix, :]
        Y_train = Y_train[ix]

        ix = np.arange(X_test.shape[0])
        np.random.shuffle(ix)
        X_test = X_test[ix, :]
        Y_test = Y_test[ix]

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        save_file = os.path.join(data_dir, "dataset_%s.hdf5" % peak_type)
        print(save_file)
        with h5py.File(save_file, "w") as of:
            of.create_dataset(name="X_train", data=X_train, compression="gzip")
            of.create_dataset(name="Y_train", data=Y_train, compression="gzip")
            of.create_dataset(name="X_test", data=X_test, compression="gzip")
            of.create_dataset(name="Y_test", data=Y_test, compression="gzip")


def sanity_check(data_dir):

    """Compare the sizes of the matrices in dataset file with the number from partial files"""

    chroms = set(["chr%d" % i for i in range(1, 23)])
    test_chroms = ["chr8", "chr9"]
    train_chroms = chroms.difference(test_chroms)

    peak_features, dip_features, control_features = 0, 0, 0
    train_files = [os.path.join(data_dir, "partial_%s.features.hdf5" % chrom) for chrom in train_chroms]
    for f in train_files:
        with h5py.File(f, "r") as inf:
            peak_features += inf["X_peak"].shape[0]
            dip_features += inf["X_dip"].shape[0]
            control_features += inf["X_control"].shape[0]
    train_sizes = [peak_features, dip_features, control_features]

    peak_features, dip_features, control_features = 0, 0, 0
    train_files = [os.path.join(data_dir, "partial_%s.features.hdf5" % chrom) for chrom in test_chroms]
    for f in train_files:
        with h5py.File(f, "r") as inf:
            peak_features += inf["X_peak"].shape[0]
            dip_features += inf["X_dip"].shape[0]
            control_features += inf["X_control"].shape[0]

    test_sizes = [peak_features, dip_features, control_features]

    dataset_files = [os.path.join(data_dir, "dataset_peak.hdf5"),
                     os.path.join(data_dir, "dataset_dip.hdf5")]
    feature_indices = [0, 1]

    for peak_file, feature_ind in zip(dataset_files, feature_indices):
        print("Saving to:", peak_file)
        with h5py.File(peak_file, "r") as inf:
            assert inf["Y_train"].shape[0] == inf["X_train"].shape[0]
            assert inf["Y_test"].shape[0] == inf["X_test"].shape[0]

            Y_train = inf["Y_train"][()]
            Y_test = inf["Y_test"][()]

            assert Y_train.shape[0] - np.sum(Y_train) == train_sizes[2]
            assert np.sum(Y_train) == train_sizes[feature_ind]
            assert np.sum(Y_test) == test_sizes[feature_ind]


if __name__ == "__main__":

    data_dir = sys.argv[1]
    create_dataset(data_dir)