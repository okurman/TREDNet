#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys

import h5py

from pybedtools import BedTool
from collections import defaultdict
import numpy as np

from lib.v1.tools import get_chrom2seq
from lib.v1.tools import run_two_phases
from lib.v1.tools import get_models
from lib.v1.tools import generate_mutation_vectors


NUCLEOTIDES = ["A", "C", "G", "T"]
INPUT_LENGTH = 2000


def extract_deltas_for_enhancer(r, models, r_seq, mutation_window_range, output_field=None):

    delta_window_length = mutation_window_range[1] - mutation_window_range[0]

    pos2nt2mat = dict()

    for i, pos in enumerate(range(r.start, r.stop)[mutation_window_range[0]: mutation_window_range[1]]):
        nt2mat = generate_mutation_vectors(r, pos, r_seq, add_reference=True if i == 0 else False)
        pos2nt2mat[i] = nt2mat

    number_of_matrices = delta_window_length * 3 + 1

    X = np.zeros((number_of_matrices, 2000, 4))

    pos2nt2ind = defaultdict(dict)
    pivot = 0
    for pos, nt2mat in pos2nt2mat.items():
        for nt, mat in nt2mat.items():
            if nt == "ref":
                continue
            pos2nt2ind[pos][nt] = pivot
            X[pivot, :, :] = mat
            pivot += 1

    Y = run_two_phases(X, models, verbose=0)
    if output_field:
        Y = Y[:, output_field]

    M = np.zeros((delta_window_length, 3))

    for i, pos in enumerate(range(r.start, r.stop)[mutation_window_range[0]: mutation_window_range[1]]):

        if r_seq[i] == "N":
            continue
        j = 0
        for nt in NUCLEOTIDES:
            if i == 0:
                ref_nt = pos2nt2mat[0]["ref"]
                if nt == ref_nt:
                    continue
            if nt not in pos2nt2ind[i]:
                continue
            ind = pos2nt2ind[i][nt]
            M[i, j] = Y[ind]
            j += 1

    ref_nt = pos2nt2mat[0]["ref"]
    ind = pos2nt2ind[0][ref_nt]
    M_ref = Y[ind]

    return M, M_ref


def generate_scores(args, chrom2seq=None):

    if not chrom2seq:
        chrom2seq = get_chrom2seq(args.genome_fasta_file)
        # return chrom2seq

    models = get_models(args.phase_one_file, args.phase_two_file)
    enhancers = BedTool(args.enhancers_file)

    lens = set([r.length for r in enhancers])

    if len(lens) > 1:
        print("All input regions must be of same length!", sys.stderr)
        print("Detected input lengths:", lens, sys.stderr)
        print("Exiting %s" % sys.argv[0])
        sys.exit()

    input_length = lens.pop()

    flank_length = int((INPUT_LENGTH - input_length) / 2)

    enh_id2deltas = dict()

    total_count = enhancers.count()

    for i, enh in enumerate(enhancers):

        # if i % 100 == 0:
        print(i + 1, "/", total_count, str(enh).strip())

        enh_id = "%s-%d-%d" % (enh.chrom, enh.start, enh.stop)

        enh.start -= flank_length
        enh.stop += flank_length
        enh_seq = chrom2seq[enh.chrom][enh.start:enh.stop].upper()

        if not len(enh_seq) == INPUT_LENGTH:
            print("Skipping the enhancer: %s" % enh)
            continue

        mutation_window_range = [flank_length, flank_length + input_length]

        M, M_ref = extract_deltas_for_enhancer(enh, models, enh_seq, mutation_window_range=mutation_window_range)
        diff = np.average(M_ref - M, 1)
        diff /= np.median(np.abs(diff))

        enh_id2deltas[enh_id] = diff

    with h5py.File(args.save_file, "w") as of:
        print("Saving to file: " + args.save_file)
        for enh_id, deltas in enh_id2deltas.items():
            of.create_dataset(name=enh_id, data=deltas, compression="gzip")


def argument_parser(p_args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('-enhancers-file', dest="enhancers_file",
                        help="BED file containing enhancers.", required=True)
    parser.add_argument('-phase-one-file', dest="phase_one_file", help="Phase one model weights (hdf5) file.",
                        required=True)
    parser.add_argument('-phase-two-file', dest="phase_two_file", help="Phase two model weights (hdf5) file.",
                        required=True)
    parser.add_argument('-save-file', dest="save_file", help="File to save the scores (h5)", required=True)
    parser.add_argument('-genome-fasta', dest="genome_fasta_file", help="Fasta file to genome assembly (hg19 of hg38)",
                        required=True)

    if p_args:
        return parser.parse_args(p_args)
    else:
        return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    if os.path.exists(args.save_file):
        print("The file for saving the results already exists!", sys.stderr)
        print(args.save_file, sys.stderr)
        print("Exiting %s" % sys.argv[0])
        sys.exit()

    generate_scores(args)
