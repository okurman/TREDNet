import numpy as np

def convert_pos_to_features(deltas, pos):

    overall_sign = np.sign(np.mean(deltas))

    def extract_four_values(scores):
        _len = len(scores)
        core_start = int((_len-6)/2)
        core_scores = scores if _len <= 6 else scores[core_start: core_start+6]

        values = [np.average(scores),
                  np.max(scores) if overall_sign > 0 else np.min(scores),
                  overall_sign * np.sum(np.sign(scores) == overall_sign)/_len,
                  overall_sign * np.sum(np.sign(core_scores) == overall_sign)/len(core_scores)]
        return values

    total_length = len(deltas)
    assert 10 < pos < total_length - 10

    window_lengths = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    features = []

    for win_len in window_lengths:
        for pivot in range(win_len):
            win_start = pos - win_len + pivot + 1
            win_deltas = deltas[win_start: win_start + win_len]
            win_features = extract_four_values(win_deltas)
            features += win_features

    return features

