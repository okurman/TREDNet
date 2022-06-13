#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sanjarbek Hudaiberdiev"

import argparse
from itertools import groupby
from operator import itemgetter

import h5py
from pybedtools import BedTool

import features
import numpy as np
from tensorflow import keras
from os.path import join


def load_model(model_dir, fpr_level=5):
    weights_file = join(model_dir, "best_model_weights.hdf5")
    model = keras.models.load_model(weights_file)
    model.load_weights(weights_file)

    thresholds_file = join(model_dir, "fpr_threshold_scores.txt")
    for line in open(thresholds_file):
        parts = line.rstrip().split()
        if int(parts[0]) == fpr_level:
            thr = float(parts[1])

    return model, thr


def annotate(model_dir, deltas_file, save_file):
    with h5py.File(deltas_file, "r") as inf:
        enh_id2deltas = {k: inf[k][()] for k in inf.keys() if k.startswith("chr22")}

    model, thr = load_model(model_dir)

    bed_list = []

    cnt = 1
    total = len(enh_id2deltas)

    for enh_id, deltas in enh_id2deltas.items():

        print(cnt, total)
        cnt += 1

        [chrom, start, stop] = enh_id.split("-")
        start = int(start)
        stop = int(stop)
        length = stop - start

        X_list = []

        for pos in range(11, length - 11):
            pos_features = features.convert_pos_to_features(deltas, pos)
            X_list.append(pos_features)

        X = np.vstack(X_list)
        Y = model.predict(X)

        positive_ix = np.argwhere(Y[:, 0] > thr)

        peaks = []
        for k, g in groupby(enumerate(positive_ix[:, 0]), lambda x: x[0] - x[1]):
            _peak = np.asarray(list(map(itemgetter(1), g)))
            if len(_peak) > 3:
                peaks.append(_peak)

        for peak in peaks:
            peak += 11
            peak_deltas = deltas[peak]

            peak_start = peak[0] + start
            peak_stop = peak[-1] + start

            _bed = "%s\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f" % (
                chrom, start, stop, peak_start, peak_stop,
                np.mean(peak_deltas),
                np.max(peak_deltas),
                np.min(peak_deltas),
                np.std(peak_deltas))

            bed_list.append(_bed)

    peaks_bed = BedTool("\n".join(bed_list), from_string=True)
    peaks_bed.saveas(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-delta-file', dest="delta_file", required=True)
    parser.add_argument('-model-dir', dest="model_dir", required=True)
    parser.add_argument('-save-file', dest="save_file", required=True)

    args = parser.parse_args()
    annotate(args.model_dir, args.delta_file, args.save_file)
