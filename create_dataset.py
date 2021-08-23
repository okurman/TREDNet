import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-pos-bed', dest="pos_bed_file", default="data/E118.H3K27ac.enhancers.bed")
parser.add_argument('-neg-bed', dest="neg_bed_file", default="data/E118.H3K27ac.controls.bed")
parser.add_argument('-save-file', dest="save_file", default="data/phase_two_dataset.hdf5")
parser.add_argument('-hg19-fasta', dest="hg19_fasta_file", default="/data/Dcode/common/hg19.fa")

args = parser.parse_args()

pos_bed_file = args.pos_bed_file
neg_bed_file = args.neg_bed_file
save_file = args.save_file

print("Starting with the parameters:")
print("Positive bed file:", pos_bed_file)
print("Negative bed file:", neg_bed_file)
print("Dataset file:", save_file)

if not os.path.exists(pos_bed_file):
    print("Positive bed file not found: ", pos_bed_file)
    sys.exit()
if not os.path.exists(neg_bed_file):
    print("Negative bed file not found: ", neg_bed_file)
    sys.exit()

if os.path.exists(save_file):
    print("File exists: ", save_file)
    print("Select a different file name")
    sys.exit()

from src import data_prep

data_prep.create_dataset_phase_two(pos_bed_file, neg_bed_file, save_file)
