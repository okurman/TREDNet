import argparse
import os
import sys
from os.path import join
import h5py
import numpy as np
import multiprocessing

import pybedtools
pybedtools.helpers.set_bedtools_path("/data/Dcode/sanjar/progs/bedtools/bin/")
from pybedtools import BedTool

import features

# WORK_DIR = "/panfs/pan1/devdcode/sanjar/peak_detection/v1/"
# ENHANCERS_FILE = join(WORK_DIR, "E087_atac_k27_1k.unique.core.bed")
# TFBS_FILE = join(WORK_DIR, "list_motif_in_enhancer.bed")
# DELTAS_FILE = "/panfs/pan1/devdcode/common/T2D_project/DL_peakNdip/file_enhancer_score/ATAC_27ac_v4/normalized_scores.hdf5"


def process_batch(enhancers_with_motifs, save_file, enh_id2deltas):

    print("\tStarting the batch for: %s" % save_file)
    control_features = []
    peak_features = []
    dip_features = []

    counts_array = []

    for enh_cnt, enh in enumerate(enhancers_with_motifs):

        control_cnt, peak_cnt, dip_cnt = 0, 0, 0
        enh_id = "%s-%d-%d" % (enh.chrom, enh.start, enh.stop)

        if not enh_id in enh_id2deltas:
            print("enh_id not found in deltas map", enh_id)
            continue

        deltas = enh_id2deltas[enh_id]

        motif_starts = np.asarray(enh.fields[3].split(","), dtype=int)
        motif_stops = np.asarray(enh.fields[4].split(","), dtype=int)

        motif_starts -= enh.start
        motif_stops -= enh.start

        motif_leftmost = min(motif_starts)
        motif_rightmost = max(motif_stops)

        control_regions = []
        control_left = [10, motif_leftmost - 20]
        if control_left[1] - control_left[0] >= 50:
            control_regions.append(control_left)
        control_right = [motif_rightmost + 20, enh.length - 10]
        if control_right[1] - control_right[0] >= 50:
            control_regions.append(control_right)

        if control_regions:
            _features = features.extract_features_for_enhancer(deltas, control_regions, control=True)
            control_cnt = len(_features)
            control_features += [_[1] for _ in _features]

        motifs = [[_start, _stop] for _start, _stop in zip(motif_starts, motif_stops)]

        features_pool = features.extract_features_for_enhancer(deltas, motifs, only_strong=True, control=False)

        for [sign, _features] in features_pool:

            if _features is None:
                continue

            if sign == 1:
                peak_features.append(_features)
                peak_cnt += 1
            elif sign == -1:
                dip_features.append(_features)
                dip_cnt += 1
            else:
                raise ValueError("Motif sign should not be %d here." % sign)

        counts_array.append([control_cnt, peak_cnt, dip_cnt])

    print("\tSaving to file: %s " % save_file)
    with h5py.File(save_file, "w") as of:
        of.create_dataset(name="X_control", data=np.vstack(control_features), compression="gzip")
        of.create_dataset(name="X_peak", data=np.vstack(peak_features), compression="gzip")
        of.create_dataset(name="X_dip", data=np.vstack(dip_features), compression="gzip")


def process_in_parallel(args, enhancers_with_motifs, enh_id2deltas):

    chroms = ["chr%d" % i for i in range(1, 23)]

    batch_list = [[r for r in enhancers_with_motifs if r.chrom == chrom] for chrom in chroms]

    args_pool = []
    for chrom, batch in zip(chroms, batch_list):
        partial_save_file = os.path.join(args.save_dir, "partial_%s.features.hdf5" % chrom)
        args_pool.append((batch, partial_save_file, enh_id2deltas))

    print("Launching processes in pool")
    pool = multiprocessing.Pool(args.cpus)
    [pool.apply_async(process_batch, args) for args in args_pool]
    pool.close()
    pool.join()


def main(args):

    if args.enhancers_file:
        enhancers = BedTool(args.enhancers_file)
    else:
        with h5py.File(args.delta_file, "r") as inf:

            bed_str = "\n".join([k.replace("-", "\t") for k in list(inf.keys())])
            enhancers = BedTool(bed_str, from_string=True).sort()

    motifs = BedTool(args.motifs_file).sort()

    print("Extracting the regions")
    enhancers_with_motifs = enhancers.intersect(motifs, F=1.0, wo=True).groupby(c=[5, 6], o="collapse")
    enhancers_wo_motifs = enhancers.intersect(motifs, v=True)
    print("Enhancers without motifs:", enhancers_wo_motifs.count())
    print("Loading deltas to map")
    with h5py.File(args.delta_file, "r") as inf:
        enh_id2deltas = {k: inf[k][()] for k in inf.keys()}

    enhancers_with_motifs = list(enhancers_with_motifs)

    if args.save_dir:
        process_in_parallel(args, enhancers_with_motifs, enh_id2deltas)
    else:
        process_batch(enhancers_with_motifs, args.save_file, enh_id2deltas)


def argument_parser(p_args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('-enhancers-file', dest="enhancers_file", default=None,
                        help="BED file containing enhancers. If not supplied, enhancers are extracted from delta file.")
    parser.add_argument('-motifs-file', dest="motifs_file", help="BED file with motif hit coordinates",
                        required=True)
    parser.add_argument('-delta-file', dest="delta_file", default=None, help="hdf5 file containing the delta scores")
    parser.add_argument('-save-file', dest="save_file", default=None, help="hdf5 file to save the results")
    parser.add_argument('-save-dir', dest="save_dir",
                        help="Directory where the feature files for each chromosome will be created. Will work only"
                             "if -save-file is not provided.")
    parser.add_argument('-cpus', dest="cpus", default=10,
                        help="Number of CPUs to use for running the data createion in parallel")
    
    args = parser.parse_args(p_args) if p_args else parser.parse_args()

    if not (args.enhancers_file or args.delta_file):
        print("Provide either enhancers file or deltas file", sys.stderr)
        print("Exiting %s" % sys.argv[0])

    return args



if __name__ == "__main__":

    args = argument_parser()

    main(args)