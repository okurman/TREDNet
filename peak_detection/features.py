#!/usr/bin/python
import sys

import numpy as np


def extract_features_for_position(deltas, pos):

    """
    Extract features for a single position on enhancer

    :param deltas: delta scores of an enhancer region
    :param pos: relative position within enhancer
    :return: numpy array of length 220
    """

    window_lengths = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    core_lengths = [6, 6, 6, 6, 6, 5, 4, 3, 2, 1]

    features = []

    for win_len, core_len in zip(window_lengths, core_lengths):

        for pivot in range(win_len):

            win_start = pos - win_len + pivot + 1

            # print(list(range(win_start, win_start + win_len)))
            win_deltas = [0 if (i < 0 or i >= len(deltas)) else deltas[i]
                              for i in range(win_start, win_start + win_len)]

            win_deltas = np.asarray(win_deltas)

            win_avg = np.mean(win_deltas)
            win_sign = np.sign(win_avg)
            win_max = max(win_sign * win_deltas)

            win_deltas *= win_sign

            core_start = int((win_len - core_len) / 2)
            core_end = core_start + core_len
            core_deltas = win_deltas[core_start: core_end]

            w_neg_ix = np.argwhere(win_deltas < 0)
            c_neg_ix = np.argwhere(core_deltas < 0)
            c_zero_ix = np.argwhere(core_deltas == 0)

            w_neg_frac = win_sign * (len(w_neg_ix) / win_len)
            c_neg_frac = win_sign * 0.5 if core_len == len(c_zero_ix) else len(c_neg_ix) / core_len
            if win_sign < 0:
                w_neg_frac -= 1
                c_neg_frac -= 1

            features += [win_avg, win_max, w_neg_frac, c_neg_frac]

    features = np.asarray(features)

    return features


def extract_features_for_motif(enhancer_delta_scores, motif_start, motif_stop):

    """
    Extract features for each position of a motif.

    :param enhancer_delta_scores: delta scores for enhancer (usually 1kb.)
    :param motif_start: start of motif (int, relative coordinate)
    :param motif_stop: stop of motif (int, relative coordinate)
    :return: numpy array of size <no of positions in motif> x 220
    """

    motif_features = []

    for pos in range(motif_start, motif_stop):

        pos_features = extract_features_for_position(enhancer_delta_scores, pos)
        motif_features.append(pos_features)

    motif_features = np.vstack(motif_features)

    return motif_features


def extract_features_for_enhancer(enhancer_deltas, motif_coordinates, only_strong=True, control=False):

    """

    :param enhancer_deltas: delta scores for enhancer (usually 1kb.)
    :param motif_coordinates: list of [motif_start, motif_stop] pairs of relative coordinates.
    :param only_strong: Extract features only motifs with motif/flank > 5.0.
            For motifs failing the requirement, None will be returned.
    :param control: Supply True if the features are being generated for control regions.

    :return: a list of (region_type, features) tuples.

    region_type:
        0:  control
        1:  peak
        -1: dip
    """

    features_pool = []

    for [motif_start, motif_stop] in motif_coordinates:

        if control:
            motif_features = extract_features_for_motif(enhancer_deltas, motif_start, motif_stop)
            features_pool.append([0, motif_features])
            continue

        motif_deltas = enhancer_deltas[motif_start: motif_stop]
        motif_sign = np.sign(np.mean(motif_deltas))

        if only_strong:

            flank_ranges = list(range(max(motif_start - 10, 0), motif_start)) + \
                           list(range(motif_stop, min(motif_stop + 10, len(enhancer_deltas))))
            flank_deltas = [enhancer_deltas[i] for i in flank_ranges]
            avg_flank_deltas = np.mean(flank_deltas)

            if avg_flank_deltas == 0 or np.abs(np.mean(motif_deltas) / avg_flank_deltas) < 5.0:
                features_pool.append([motif_sign, None])
                continue

        motif_features = extract_features_for_motif(enhancer_deltas, motif_start, motif_stop)
        features_pool.append([motif_sign, motif_features])

    return features_pool

