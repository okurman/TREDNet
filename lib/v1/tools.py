#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt


NUCLEOTIDES = np.array(['A', 'C', 'G', 'T'])


def get_chrom2seq(hg19_fasta_file="hg19.fa", capitalize=True):

	chrom2seq = {}
	for seq in SeqIO.parse(hg19_fasta_file, "fasta"):
		chrom2seq[seq.description.split()[0]] = str(seq.seq.upper()) if capitalize else str(seq.seq)

	return chrom2seq


def seq2one_hot(seq):
	m = np.zeros((len(seq), 4), dtype=bool)
	seq = seq.upper()
	for i in range(len(seq)):
		m[i, :] = (NUCLEOTIDES == seq[i])

	return m


def get_models(phase_one_file, phase_two_file):

	model_1 = load_model(phase_one_file)
	model_2 = load_model(phase_two_file)

	return [model_1, model_2]


def get_phase_one_model(weights_file):
	model = load_model(weights_file)

	return model


def run_two_phases(X, models, verbose=1):

	Y_1 = models[0].predict(X, verbose=verbose)
	Y_1 = Y_1[..., np.newaxis]
	Y_2 = models[1].predict(Y_1, verbose=verbose)

	return Y_2


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


def generate_plot_history(history, save_file, title=None):

	if type(history) == str:
		history = pd.read_table(history)
	else:
		history = history.history

	loss = history["loss"]
	val_loss = history["val_loss"]
	acc = history["acc"]
	val_acc = history["val_acc"]

	x_range = np.arange(len(loss))

	fig, axs = plt.subplots(1, 2)

	axs[0].plot(x_range, loss, label="train")
	axs[0].plot(x_range, val_loss, label="val")
	axs[0].set_xlabel("Epochs")
	axs[0].set_ylabel("Loss")
	axs[0].set_title("Loss")
	axs[0].legend(loc="upper right")

	axs[1].plot(x_range, acc, label="train")
	axs[1].plot(x_range, val_acc, label="val")
	axs[1].set_xlabel("Epochs")
	axs[1].set_ylabel("Accuracy")
	axs[1].set_title("Accuracy")
	axs[1].legend(loc="upper left")

	if title:
		plt.suptitle(title)

	plt.tight_layout()
	plt.savefig(save_file)
	plt.close()