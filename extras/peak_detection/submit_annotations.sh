#!/usr/bin/env bash

enhancers_file=$1
delta_file=$2
model_dir=$3
save_dir=$4

mkdir -p $save_dir/logs

#split --additional-suffix=.bed -n l/20 $enhancers_file $save_dir/partial_

for f in $save_dir/partial_*.bed
do
  save_file=`echo $f | sed 's/.bed/.PAS.bed/g'`
  n=`basename $f | cut -d"." -f1 | cut -d"_" -f2`

#  echo $f
#  echo $save_file

  sbatch --time=10:00:00 \
         --partition=gpu \
         --gres=gpu:k80:1 \
         --cpus-per-task=8 \
         --mem=100g \
         --error=$save_dir/logs/$n.err \
         --output=$save_dir/logs/$n.out \
         --job-name=$n \
         annotate.sh $f $delta_file $model_dir $save_file

done
