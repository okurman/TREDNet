#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from os.path import join, exists

import kipoi
import vcf
from pybedtools import BedTool
from src.data_prep import seq2one_hot
from src.models import run_two_phases, get_models
from collections import defaultdict
import numpy as np

NUCLEOTIDES = ["A", "C", "G", "T"]


class Model:

    def __init__(self, args):

        self.use_kipoi = args.use_kipoi

        if args.use_kipoi:
            self.model = kipoi.get_model("TREDNet/phase_two_%s" % args.phase_two_name)
        else:
            if args.phase_two_name == "islet":
                phase_two_file = join("data/TREDNet_weights_phase_two_islets.hdf5")
            else:
                phase_two_file = join("data/TREDNet_weights_phase_two_%s.hdf5" % args.phase_two_name)

            phase_one_file = "data/TREDNet_weights_phase_one.hdf5"

            if not exists(phase_one_file):
                print("The phase-one model file doesn't exist: %s" % phase_one_file)
                print("Download the model using data_download.sh script first and make tha the file exists.")
                sys.exit()

            if not exists(phase_two_file):
                print("The phase-two model file doesn't exist: %s" % phase_two_file)
                print("Download the model using data_download.sh script first and make tha the file exists.")
                sys.exit()

            self.model = get_models(phase_one_file, phase_two_file)

    def predict(self, X):

        if self.use_kipoi:
            Y = self.model.predict_on_batch(X)
        else:
            Y = run_two_phases(X, self.model)

        return Y


def extract_deltas_for_enhancer(r, model, r_seq, mutation_window_range=[0, 2000]):

    r_seq = r_seq.upper()

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

    Y = model.predict(X)
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


def generate_mutation_vectors(r, pos, r_seq, add_reference=False):

    relative_position = pos - r.start
    ref_nt = r_seq[relative_position]

    nt2matrix = dict()
    nt2matrix["ref"] = ref_nt

    if add_reference:
        nt2matrix[ref_nt] = seq2one_hot(r_seq)

    if not ref_nt == "N":
        for nt in set(NUCLEOTIDES).difference(ref_nt):
            alt_seq = r_seq[:relative_position] + nt.upper() + r_seq[relative_position + 1:]
            nt2matrix[nt] = seq2one_hot(alt_seq)

    return nt2matrix


def generate_scores(args):

    model = Model(args)

    vcf_records = list(vcf.Reader(open(args.vcf_file)))
    total_records = len(vcf_records)

    out_pool = []
    for cnt, vcf_rec in enumerate(vcf_records):

        chrom = "chr" + vcf_rec.CHROM
        _str = "%s\t%d\t%d\t%s" % (chrom, vcf_rec.POS - 1000, vcf_rec.POS + 1000, vcf_rec.ID)

        print("\nVariant: (%d/%d)" % (cnt, total_records), _str)
        _bed = BedTool(_str, from_string=True)

        _bed = _bed.sequence(fi=args.hg19_fasta_file)
        seq = open(_bed.seqfn).readlines()[1].strip()
        _bed = _bed[0]

        if args.score_iep:

            ref_seq = seq[:999] + vcf_rec.REF + seq[1000:]
            alt_seqs = [seq[:999] + str(alt) + seq[1000:] for alt in vcf_rec.ALT]

            X_ref = seq2one_hot(ref_seq)
            X_alt = seq2one_hot(alt_seqs[0])
            X = np.stack((X_ref, X_alt))

            Y = model.predict(X)

            Y = Y[:, 0]
            score = np.max(Y) * np.abs(Y[0] - Y[1])

        elif args.score_delta:

            M, M_ref = extract_deltas_for_enhancer(_bed, model, seq, mutation_window_range=[990, 1010])
            diff = np.average(M_ref - M, 1)
            diff /= np.median(np.abs(diff))
            score = ",".join([str(_) for _ in diff])

        out_pool.append([chrom, vcf_rec.POS, vcf_rec.REF, ",".join([str(_) for _ in vcf_rec.ALT]), score])

    with open(args.save_file, "w") as of:
        header = "#CHROM\tPOS\tREF\tALT\t%s score\n" % ("IEP" if args.score_iep else "Delta")
        of.write(header)
        for _arr in out_pool:
            out_line = "\t".join([str(_) for _ in _arr]) + "\n"
            of.write(out_line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-vcf-file', dest="vcf_file",
                        help="VCF file containing variants for calculating the mutational scores.", required=True)
    parser.add_argument('-phase-one-file', dest="phase_one_file", help="Phase one model weights (hdf5) file.",
                        default="data/TREDNet_weights_phase_one.hdf5")
    parser.add_argument('-phase-two-name', dest="phase_two_name", help="Phase two model's cell name.",
                        default="islet", choices=["islet", "HepG2", "K562"], required=True)
    parser.add_argument('-save-file', dest="save_file", default="variant_scores.txt", help="File to save the scores",
                        required=True)
    parser.add_argument('-hg19-fasta', dest="hg19_fasta_file", help="Fasta file to hg19 assembly", required=True)
    parser.add_argument('-score-delta', default=True, help="Generate delta scores",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-score-iep', default=False, help="Generate IEP scores",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-use-kipoi', default=False, help="Use Kipoi models for running the models",
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if not args.use_kipoi:
        USE_KIPOI = False

    if args.score_delta:
        generate_scores(args)
