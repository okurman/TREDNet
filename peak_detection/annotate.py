#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sanjarbek Hudaiberdiev"

import argparse
from itertools import groupby
import pybedtools
pybedtools.helpers.set_bedtools_path("/data/Dcode/sanjar/progs/bedtools/bin/")
import sys
sys.path.append("/data/Dcode/sanjar/progs/bedtools/bin/")

from pybedtools import BedTool
import features
import numpy as np
from tensorflow import keras
from os.path import join
from tqdm.auto import tqdm, trange
import h5py


def load_model(model_dir):

    weights_file = join(model_dir, "model_weights.hdf5")
    model = keras.models.load_model(weights_file)
    model.load_weights(weights_file)

    thresholds_file = join(model_dir, "fpr_threshold_scores.txt")
    frp2thr = {l.split("\t")[0].strip(): float(l.split("\t")[1].rstrip()) for l in open(thresholds_file)}

    return model, frp2thr


def get_consecutive_positives(Y, thr):

    positive_ix = np.argwhere(Y[:, 0] > thr)
    peaks = []
    for k, g in groupby(enumerate(positive_ix[:, 0]), lambda x: x[0] - x[1]):
        # each group is a list of tuples (cnt, pos).
        _peak = np.asarray(list(map(lambda x: x[1], g)))
        if len(_peak) >= 5:
            peaks.append(_peak)

    return peaks


def annotate(args):

    enhancer_file = args.enhancer_file
    deltas_file = args.delta_file
    model_dir = args.model_dir
    save_file = args.save_file
    fpr = args.fpr

    with h5py.File(deltas_file, "r") as inf:
        # enh_id2deltas = {k: inf[k][()] for k in inf.keys() if k.startswith("chr22")}
        enh_id2deltas = {k: inf[k][()] for k in inf.keys()}

    model, fpr2thr = load_model(model_dir)

    if fpr == "all":
        fprs = ["1", "3", "5", "10"]
        save_files = [save_file.replace(".bed", ".fpr_%s.bed" % _) for _ in fprs]
    else:
        fprs = [str(fpr)]
        save_files = [save_file]

    peaks_pool = [[] for _ in fprs]

    enh_ids = ["%s-%d-%d" % (r.chrom, r.start, r.stop) for r in BedTool(enhancer_file)]

    for enh_id in tqdm(enh_ids, position=0, leave=True, desc="Annotating enhancers"):

        deltas = enh_id2deltas[enh_id]

        [chrom, start, stop] = enh_id.split("-")
        start = int(start)
        stop = int(stop)
        length = stop - start

        X_list = []

        # feature windows span from -10 to +10
        for pos in range(11, length - 11):
            pos_features = features.extract_features_for_position(deltas, pos)
            X_list.append(pos_features)

        X = np.vstack(X_list)
        X = X[..., np.newaxis]
        Y = model.predict(X)

        for ind, fpr in enumerate(fprs):

            thr = fpr2thr[fpr]
            peaks = get_consecutive_positives(Y, thr)

            for peak in peaks:
                peak += 11
                peak_deltas = deltas[peak]

                peak_start = peak[0] + start
                peak_stop = peak[-1] + start

                bed_parts = [chrom, peak_start, peak_stop, start, stop,
                            np.round(np.mean(peak_deltas), 3),
                            np.round(np.max(peak_deltas), 3),
                            np.round(np.min(peak_deltas), 3)]

                peaks_pool[ind].append(bed_parts)

    for peaks, save_file in zip(peaks_pool, save_files):

        if not peaks:
            continue

        print(save_file)
        bed_str = "\n".join(["\t".join(list(map(str, peak))) for peak in peaks])
        BedTool(bed_str, from_string=True).sort().saveas(save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-enhancer-file', dest="enhancer_file", required=True)
    parser.add_argument('-delta-file', dest="delta_file", required=True)
    parser.add_argument('-model-dir', dest="model_dir", required=True)
    parser.add_argument('-save-file', dest="save_file", required=True, help="BED file to save the annotations")
    parser.add_argument('-fpr', dest="fpr", default="5", choices=["1", "3", "5", "10", "all"], required=True,
                  help="false-positive rate for annotating the region as EDR or ESR.")

    args = parser.parse_args()
    annotate(args)
