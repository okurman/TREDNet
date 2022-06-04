import os
import numpy as np
from os.path import join
from Bio import SeqIO
from pybedtools import BedTool
from .models import get_phase_one_model
import h5py

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


def get_chrom2seq(hg19_fasta_file="/data/Dcode/common/hg19.fa", capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(hg19_fasta_file, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq


def seq2one_hot(seq):

    m = np.zeros((len(seq), 4), dtype=np.bool)
    seq = seq.upper()
    for i in range(len(seq)):
        m[i, :] = (NUCLEOTIDES == seq[i])

    return m


def create_dataset_phase_two(positive_bed_file, negative_bed_file, dataset_save_file, chrom2seq=None,
                                      model=None):

    if not model:
        print("Loading the phase_one model")
        model = get_phase_one_model()
        # return model

    if not chrom2seq:
        print("Loading hg19 fasta into memory")
        chrom2seq = get_chrom2seq()
        # return chrom2seq

    print("Generating the positive dataset")

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

    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in validation_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    for bed_list, data_list in zip([pos_train_bed, pos_val_bed, pos_test_bed],
                                   [pos_train_data, pos_val_data, pos_test_data]):

        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print("Generating the negative dataset")

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in validation_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip([neg_train_bed, neg_val_bed, neg_test_bed],
                                   [neg_train_data, neg_val_data, neg_test_data]):

        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print("Merging positive and negative to single matrices")
    print("Positive training set")
    pos_train_data_matrix = np.zeros((len(pos_train_data), INPUT_LENGTH, 4))
    for i in range(len(pos_train_data)):
        pos_train_data_matrix[i, :, :] = pos_train_data[i]
    print("Positive validation set")
    pos_val_data_matrix = np.zeros((len(pos_val_data), INPUT_LENGTH, 4))
    for i in range(len(pos_val_data)):
        pos_val_data_matrix[i, :, :] = pos_val_data[i]
    print("Positive test set")
    pos_test_data_matrix = np.zeros((len(pos_test_data), INPUT_LENGTH, 4))
    for i in range(len(pos_test_data)):
        pos_test_data_matrix[i, :, :] = pos_test_data[i]

    print("Negative training set")
    neg_train_data_matrix = np.zeros((len(neg_train_data), INPUT_LENGTH, 4))
    for i in range(len(neg_train_data)):
        neg_train_data_matrix[i, :, :] = neg_train_data[i]
    print("Negative validation set")
    neg_val_data_matrix = np.zeros((len(neg_val_data), INPUT_LENGTH, 4))
    for i in range(len(neg_val_data)):
        neg_val_data_matrix[i, :, :] = neg_val_data[i]
    print("Negative test set")
    neg_test_data_matrix = np.zeros((len(neg_test_data), INPUT_LENGTH, 4))
    for i in range(len(neg_test_data)):
        neg_test_data_matrix[i, :, :] = neg_test_data[i]

    print("Stacking up the positive and negative set matrices")
    test_data = np.vstack((pos_test_data_matrix, neg_test_data_matrix))
    test_labels = np.concatenate((np.ones(len(pos_test_data)), np.zeros(len(neg_test_data))))
    train_data = np.vstack((pos_train_data_matrix, neg_train_data_matrix))
    train_labels = np.concatenate((np.ones(len(pos_train_data)), np.zeros(len(neg_train_data))))
    val_data = np.vstack((pos_val_data_matrix, neg_val_data_matrix))
    val_labels = np.concatenate((np.ones(len(pos_val_data)), np.zeros(len(neg_val_data))))

    print("Running predictions on phase-one model")
    test_data = model.predict(test_data)
    train_data = model.predict(train_data)
    val_data = model.predict(val_data)

    print("Saving to file:", dataset_save_file)

    with h5py.File(dataset_save_file, "w") as of:
        print("Test data")
        of.create_dataset(name="test_data", data=test_data, compression="gzip")
        of.create_dataset(name="test_labels", data=test_labels, compression="gzip")
        print("Train data")
        of.create_dataset(name="train_data", data=train_data, compression="gzip")
        of.create_dataset(name="train_labels", data=train_labels, compression="gzip")
        print("Validation data")
        of.create_dataset(name="val_data", data=val_data, compression="gzip")
        of.create_dataset(name="val_labels", data=val_labels, compression="gzip")


def load_dataset(data_file):

    data = {}

    with h5py.File(data_file, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]

    data["train_data"] = data["train_data"][..., np.newaxis]
    data["test_data"] = data["test_data"][..., np.newaxis]
    data["val_data"] = data["val_data"][..., np.newaxis]

    return data