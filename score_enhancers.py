# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse
import sys
import os

from lib.v1.tools import seq2one_hot
from lib.v1.tools import run_two_phases
from lib.v1.tools import get_models
import numpy as np
from Bio import SeqIO

NUCLEOTIDES = ["A", "C", "G", "T"]


class Model:

    def __init__(self, args):

        if not os.path.exists(args.phase_one_file):
            print("The phase-one model file doesn't exist: %s" % args.phase_one_file)
            sys.exit()
        if not os.path.exists(args.phase_two_file):
            print("The phase-two model file doesn't exist: %s" % args.phase_two_file)
            sys.exit()
        self.model = get_models(args.phase_one_file, args.phase_two_file)

    def predict(self, X):

        Y = run_two_phases(X, self.model)

        return Y


def score_fasta(fasta_file, model):

    print("Converting the fasta to matrix.")
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    X = np.zeros((len(seqs), 2000, 4))
    for i, seq in enumerate(seqs):
        X[i, :, :] = seq2one_hot(seq.seq)

    print("Running predictions.")
    Y = model.predict(X)

    return Y


def generate_scores(args):

    model = Model(args)

    if args.input_file.endswith(".fa"):
        Y = score_fasta(args.input_file, model)
        np.savetxt(args.save_file, Y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input_file",
                        help="Input file.", required=True)
    parser.add_argument('-phase-one-file', dest="phase_one_file", required=True)
    parser.add_argument('-phase-two-file', dest="phase_two_file", required=True)
    parser.add_argument('-save-file', dest="save_file", required=True)
    parser.add_argument('-hg19-fasta', dest="hg19_fasta_file", help="Fasta file to hg19 assembly")

    args = parser.parse_args()

    generate_scores(args)

