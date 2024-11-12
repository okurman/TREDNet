
import sys
from os.path import join
import h5py
import numpy as np
from pybedtools import BedTool

import features

WORK_DIR = "/panfs/pan1/devdcode/sanjar/peak_detection/v1/"
ENHANCERS_FILE = join(WORK_DIR, "E087_atac_k27_1k.unique.core.bed")
TFBS_FILE = join(WORK_DIR, "list_motif_in_enhancer.bed")
DELTAS_FILE = "/panfs/pan1/devdcode/common/T2D_project/DL_peakNdip/file_enhancer_score/ATAC_27ac_v4/normalized_scores.hdf5"


def extract_dataset_chrom(chrom="chr1"):
    
    save_file = join(WORK_DIR, "partials/%s_data.hdf5" % chrom)
    
    print("Extracting the regions")
    enhancers_with_motifs = BedTool(ENHANCERS_FILE).intersect(TFBS_FILE, wo=True).groupby(c=[5, 6], o="collapse")
    enhancers_with_motifs = [r for r in enhancers_with_motifs if r.chrom == chrom]
    enhancers_wo_motifs = BedTool(ENHANCERS_FILE).intersect(TFBS_FILE, v=True)
    enhancers_wo_motifs = [r for r in enhancers_wo_motifs if r.chrom == chrom]

    print("Loading deltas to map")
    with h5py.File(DELTAS_FILE, "r") as inf:
        enh_id2deltas = {k: inf[k][()] for k in inf.keys() if k.startswith(chrom)}

    control_features = []
    control_info = []
    peak_features = []
    peak_info = []
    dip_features = []
    dip_info = []

    print("Starting to scan the enhancers")
    for enh in enhancers_with_motifs:

        motif_starts = np.asarray(enh.fields[3].split(","), dtype=int)
        motif_stops = np.asarray(enh.fields[4].split(","), dtype=int)

        leftmost = min(motif_starts)
        rightmost = max(motif_stops)

        control_range = list(range(enh.start + 11, leftmost - 20))
        control_range += list(range(rightmost + 20, enh.stop - 11))

        enh_id = "%s-%d-%d" % (enh.chrom, enh.start, enh.stop)
        if not enh_id in enh_id2deltas:
            print("enh_id not found in deltas map", enh_id)
            continue

        deltas = enh_id2deltas[enh_id]

        for m_start, m_stop in zip(motif_starts, motif_stops):

            if m_start - enh.start < 11 or enh.stop - m_stop < 11:
                continue

            m_start -= enh.start
            m_stop -= enh.start

            m_deltas = deltas[m_start: m_stop]
            flank_deltas = np.concatenate((
                deltas[m_start-10: m_start],
                deltas[m_stop: m_stop+10]
            ))

            # if the peak is strong
            if np.mean(m_deltas) > 0 and np.mean(m_deltas)/np.mean(flank_deltas) > 5:
                for pos in range(m_start, m_stop):

                    pos_info = enh_id + "-%d" % (pos + enh.start)
                    pos_features = features.convert_pos_to_features(deltas, pos)

                    peak_info.append(pos_info)
                    peak_features.append(pos_features)

            # if strong dip
            if np.mean(m_deltas) < 0 and np.abs(np.mean(m_deltas)/np.mean(flank_deltas)) > 5:
                for pos in range(m_start, m_stop):

                    pos_info = enh_id + "-%d" % (pos + enh.start)
                    pos_features = features.convert_pos_to_features(deltas, pos)

                    dip_info.append(pos_info)
                    dip_features.append(pos_features)

        for pos in control_range:

            pos_info = enh_id + "-%d" % pos
            _pos = pos - enh.start
            pos_features = features.convert_pos_to_features(deltas, _pos)
            control_features.append(pos_features)
            control_info.append(pos_info)

    for enh in enhancers_wo_motifs:
        control_range = list(range(enh.start + 11, enh.stop - 11))
        enh_id = "%s-%d-%d" % (enh.chrom, enh.start, enh.stop)
        deltas = enh_id2deltas[enh_id]
        for pos in control_range:
            pos_info = enh_id + "-%d" % pos
            _pos = pos - enh.start
            pos_features = features.convert_pos_to_features(deltas, _pos)
            control_features.append(pos_features)
            control_info.append(pos_info)

    print("Saving to file:", save_file)
    with h5py.File(save_file, "w") as of:
        of.create_dataset(name="X_control", data=np.asarray(control_features), compression="gzip")
        of.create_dataset(name="info_control", data=np.asarray(control_info, dtype="S2"), compression="gzip")
        of.create_dataset(name="X_peak", data=np.asarray(peak_features), compression="gzip")
        of.create_dataset(name="info_peak", data=np.asarray(peak_info, dtype="S2"), compression="gzip")
        of.create_dataset(name="X_dip", data=np.asarray(dip_features), compression="gzip")
        of.create_dataset(name="info_dip", data=np.asarray(dip_info, dtype="S2"), compression="gzip")

    print("Chromosome %s completed" % chrom)


def merge_partials():

    test_chroms = [8, 9]
    train_chroms = set(range(1, 23)).difference(test_chroms)

    train_files = [join(WORK_DIR, "partials/chr%d_data.hdf5") % d for d in train_chroms]
    test_files = [join(WORK_DIR, "partials/chr%d_data.hdf5") % d for d in test_chroms]

    peak_dataset_file = join(WORK_DIR, "DL_dataset_peaks.hdf5")
    dip_dataset_file = join(WORK_DIR, "DL_dataset_dips.hdf5")

    def _append(m, target):
        target.resize((target.shape[0] + m.shape[0]), axis=0)
        target[-m.shape[0]:] = m

    with h5py.File(peak_dataset_file, "w") as pf, h5py.File(dip_dataset_file, "w") as df:

        for file_array, suffix in zip([train_files, test_files], ["train", "test"]):
            print(suffix)
            X_name = "X_%s" % suffix
            Y_name = "Y_%s" % suffix
            info_name = "info_%s" % suffix

            for f in file_array[::-1]:
                print(f)
                inf = h5py.File(f, "r")

                X_control = inf["X_control"]
                info_control = inf["info_control"]
                X_peak = inf["X_peak"]
                info_peak = inf["info_peak"]
                X_dip = inf["X_dip"]
                info_dip = inf["info_dip"]

                X = np.vstack((X_peak, X_control))
                Y = np.vstack((np.ones((X_peak.shape[0], 1)), np.zeros((X_control.shape[0], 1))))
                info = np.concatenate((info_peak, info_control))

                if X_name not in pf:
                    pf.create_dataset(name=X_name, maxshape=(None, X.shape[1]), data=X, compression="gzip")
                    pf.create_dataset(name=Y_name, maxshape=(None, Y.shape[1]), data=Y, compression="gzip")
                    pf.create_dataset(name=info_name, maxshape=(None,), data=info, compression="gzip")
                else:
                    _append(X, pf[X_name])
                    _append(Y, pf[Y_name])
                    _append(info, pf[info_name])

                X = np.vstack((X_dip, X_control))
                Y = np.vstack((np.ones((X_dip.shape[0], 1)), np.zeros((X_control.shape[0], 1))))
                info = np.concatenate((info_dip, info_control))

                if X_name not in df:
                    df.create_dataset(name=X_name, maxshape=(None, X.shape[1]), data=X, compression="gzip")
                    df.create_dataset(name=Y_name, maxshape=(None, Y.shape[1]), data=Y, compression="gzip")
                    df.create_dataset(name=info_name, maxshape=(None,), data=info, compression="gzip")
                else:
                    _append(X, df[X_name])
                    _append(Y, df[Y_name])
                    _append(info, df[info_name])



if __name__ == "__main__":

    chrom = "chr" + sys.argv[1]
    extract_dataset_chrom(chrom)
