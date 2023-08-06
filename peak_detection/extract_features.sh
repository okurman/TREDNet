#!/usr/bin/env bash

/data/Dcode/sanjar/miniconda3/bin/python /data/Dcode/sanjar/TREDNet/peak_detection/extract_features.py -enhancers-file $1 -motifs-file $2 -delta-file $3 -save-file $4

echo "Done!" $1