#!/usr/bin/env bash

all_enh_file=$1
motifs_file=$2
delta_file=$3

mkdir -p logs

cut -f1 $all_enh_file | sort | uniq | grep -v "_" | grep -v chrX | grep -v chrY | while read chrom
do

  echo $chrom

  enh_file=partial_$chrom.bed
  grep -P "^$chrom\t" $all_enh_file > $enh_file

  save_file=partial_$chrom.features.hdf5

  sbatch --time=10:00:00 \
         --cpus-per-task=8 \
         --mem=100g \
         --error=logs/$chrom.err \
         --output=logs/$chrom.out \
         --job-name="ft_"$chrom \
         create_dataset.sh $enh_file $motifs_file $delta_file $save_file

done



