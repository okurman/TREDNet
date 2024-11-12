#!/usr/bin/env bash

# $f $delta_file $model_dir $save_file

/data/hudaiber/keras python annotate.py -enhancer-file $1 -delta-file $2 -model-dir $3 -save-file $4 -fpr all
