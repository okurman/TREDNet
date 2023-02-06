import argparse
import os
import sys
import numpy as np
from Bio import SeqIO
from pybedtools import BedTool
from models import get_phase_one_model
import h5py
import tempfile

train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", "chr11", "chr12", "chr13",
                     "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
validation_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]

BIN_LENGTH = 200
INPUT_LENGTH = 2000
EPOCH = 200
BATCH_SIZE = 64
GPUS = 4

NUCLEOTIDES = np.array(['A', 'C', 'G', 'T'])


def get_chrom2seq(hg19_fasta_file="hg19.fa", capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(hg19_fasta_file, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq


def seq2one_hot(seq):

    m = np.zeros((len(seq), 4), dtype=bool)
    seq = seq.upper()
    for i in range(len(seq)):
        m[i, :] = (NUCLEOTIDES == seq[i])

    return m


def is_interactive():

    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        return True
    else:
        return False


def create_dataset_phase_two(positive_bed_file, negative_bed_file, dataset_save_file, chrom2seq=None,
                                      model=None, phase_one_weights_file=None, hg19_file=None):

    if not model:
        print("Loading the phase_one model\n")
        model = get_phase_one_model(phase_one_weights_file)
        # return model

    if not chrom2seq:
        print("Loading hg19 fasta into memory\n")
        chrom2seq = get_chrom2seq(hg19_file)
        # return chrom2seq

    print("Splitting the regions to train/val/test\n")

    pos_beds = list(BedTool(positive_bed_file))
    neg_beds = list(BedTool(negative_bed_file))
    for bed in [pos_beds, neg_beds]:
        for r in bed:
            if r.length == INPUT_LENGTH:
                continue
            flank = (INPUT_LENGTH - r.length) // 2
            r.start -= flank
            r.stop += flank
            if not r.length == INPUT_LENGTH:
                r.stop += (INPUT_LENGTH - r.length)

    pos_beds_split = list()
    pos_beds_split.append([r for r in pos_beds if r.chrom in train_chromosomes])
    pos_beds_split.append([r for r in pos_beds if r.chrom in validation_chromosomes])
    pos_beds_split.append([r for r in pos_beds if r.chrom in test_chromosomes])

    neg_beds_split = list()
    neg_beds_split.append([r for r in neg_beds if r.chrom in train_chromosomes])
    neg_beds_split.append([r for r in neg_beds if r.chrom in validation_chromosomes])
    neg_beds_split.append([r for r in neg_beds if r.chrom in test_chromosomes])

    tmp_file = tempfile.NamedTemporaryFile(prefix="/data/Dcode/common/tmp.TREDnet.", suffix=".hdf5")
    ph1_data = h5py.File(tmp_file.name, "w")
    ph2_data = h5py.File(dataset_save_file, "w")

    print("Writing the regions to one-hot\n")
    for name, pos_beds, neg_beds in zip(["train", "val", "test"],
                                         pos_beds_split,
                                         neg_beds_split):

        total_length = len(pos_beds) + len(neg_beds)
        print(" %s  size: %d" % (name, total_length))
        data = ph1_data.create_dataset(name="%s_data" % name, shape=(total_length, 2000, 4), dtype=bool)
        labels = ph2_data.create_dataset(name="%s_labels" % name, shape=(total_length, 1), dtype=int, compression="gzip")

        cnt = -1
        for label, beds in zip([1, 0], [pos_beds, neg_beds]):
            for r in beds:
                cnt += 1

                if cnt % 50000 == 0:
                    print("   progress: %10d / %10d" % (cnt, total_length))

                _seq = chrom2seq[r.chrom][r.start:r.stop]
                if not len(_seq) == 2000:
                    print("Skipping the regions with <2kb:", r)
                    continue
                _vector = seq2one_hot(_seq)

                data[cnt, :, :] = _vector
                labels[cnt] = label

    print("\nRunning predictions on one-hot data")
    for name in ["train_data", "val_data", "test_data"]:

        print(name)

        in_data = ph1_data[name]
        out_data = ph2_data.create_dataset(name=name, shape=(in_data.shape[0], 1924, 1), compression="gzip")

        chunk_size = 50000

        for i in range(0, in_data.shape[0], chunk_size):
            print("Batch  %d / %d" % (i/chunk_size, in_data.shape[0]/chunk_size))
            chunk_in = in_data[i: i + chunk_size, :, :]
            chunk_out = model.predict(chunk_in, verbose=1 if is_interactive() else 2)
            out_data[i: i + chunk_size, :, :] = chunk_out[..., np.newaxis]

    ph1_data.close()
    ph2_data.close()


def load_dataset(data_file):

    data = {}

    with h5py.File(data_file, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]

    if len(data["train_data"].shape) == 2:
        data["train_data"] = data["train_data"][..., np.newaxis]
        data["test_data"] = data["test_data"][..., np.newaxis]
        data["val_data"] = data["val_data"][..., np.newaxis]

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset for phase two")
    parser.add_argument('-pos-file', dest="pos_file", required=True, help="Positive regions bed file")
    parser.add_argument('-neg-file', dest="neg_file", required=True, help="Control regions bed file")
    parser.add_argument('-save-file', dest="save_file", required=True, help="Dataset file (.hdf5) to save")
    parser.add_argument('-phase-one-weights', dest="phase_one_weights", default="../data/phase_one_weights.hdf5",
                        help="Weights file of the phase one model")

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()

    pos_file = args.pos_file
    neg_file = args.neg_file
    save_file = args.save_file

    if not os.path.exists(pos_file):
        print("Data file not found: %s " % pos_file)
        sys.exit()

    if not os.path.exists(neg_file):
        print("Data file not found: %s" % neg_file)
        sys.exit()

    if os.path.exists(save_file):
        print("Data save file already exists: %s " % save_file)
        sys.exit()

    create_dataset_phase_two(pos_file, neg_file, save_file)