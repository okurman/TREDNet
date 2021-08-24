import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-vcf-file', dest="vcf_file", default="data/T2D_GWAS_SNPs.vcf")
parser.add_argument('-model-file', dest="model_file", default="data/phase_two_weights.hdf5")
parser.add_argument('-save-file', dest="save_file", default="variant_scores.txt")
parser.add_argument('-hg19-fasta', dest="hg19_fasta_file", default="/data/Dcode/common/hg19.fa")

args = parser.parse_args()

vcf_file = args.vcf_file
model_file = args.model_file
save_file = args.save_file

print("Starting with the parameters:")
print("Variants input file:", vcf_file)
print("Phase-two model file:", model_file)
print("Save file:", save_file)

if not os.path.exists(vcf_file):
    print("Input file not found: ", vcf_file)
    sys.exit()
if not os.path.exists(model_file):
    print("Model file not found: ", model_file)
    sys.exit()


# TODO: implement