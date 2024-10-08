#!/usr/bin/env python

import h5py
import numpy as np

sorted_ix_200 = [2067,  623, 1513,  622, 1332, 1911,  624, 1910, 1751, 1917, 1147,
       2068, 2073, 2070, 1987, 2066, 1331, 1326, 1418,  872, 1981,  867,
       1986, 1570, 1095,  627, 1395, 1375, 1370, 1853, 2074, 1413,  633,
        575,  730, 1515,  893, 1510, 1325, 1028, 1137, 1916,  834, 1600,
       1511, 1750,  588, 1412,  841, 1870, 2063,  997, 1993,  727, 1550,
        848,  959,  740,  590,  576,  926, 1019, 1416,  692, 1368, 1297,
       1041, 1345, 1836, 1512, 1429, 1710, 1107,  628, 2052, 1672, 1767,
       1383, 1554, 1886,  847, 1755,  936, 1892,  968, 1944, 1077,  537,
       2062, 1764,  634, 1199, 1818, 1090,  919,  906, 1006, 2054, 1984,
       1507,  991, 1146,  723, 1852, 2048,  680, 1372, 1141, 1532,  840,
       1930, 1844, 1094, 1896, 1753,  735, 1301,  946,  743, 1551, 1369,
       1616, 1190, 1423,  693, 1608, 1723, 2065,  933, 1464,  833, 1039,
        426,  330, 1901,  990, 1493, 1909, 1393, 1850,  577, 1868, 1387,
        532,  300, 1978, 1713, 1103, 1089,  554, 1914,  608, 1264, 1655,
       1898, 1527, 1382, 1140, 1533, 2064, 1473, 1757, 1075, 1155, 1052,
        823,  814,  482, 1329,  489, 1456, 1878, 1125,  528,  369, 1756,
       1188,  856,  492, 1943,  530,  696, 1990, 1908, 1054,  649,  939,
       1315, 1348, 1642,  866, 1337,  527, 1575, 1576, 1074,  604,  888,
        571, 1042]


sorted_ix_200 = np.asarray(sorted_ix_200)
feat_idx_10 = np.sort(sorted_ix_200[:10])
feat_idx_50 = np.sort(sorted_ix_200[:50])
feat_idx_100 = np.sort(sorted_ix_200[:100])

DATA_FILE = "/home/ubuntu/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"
DATA_FILE = "/Users/okurman/Projects/TREDNet/data/phase_one/datasets/phase_one.dataset.400_1K_4560.unc.h5"

def create_compact_labels():

    with h5py.File(DATA_FILE, "a") as inf:

        for n_feat in [10, 50, 100]:
            feature_idx = np.sort(sorted_ix_200[:n_feat])
            for y_label in ["Y_test", "Y_val", "Y_train"]:
                y = inf[y_label][:, feature_idx]
                print(y_label, y.shape)
                inf.create_dataset(name=y_label+f"_{n_feat}", data=y)


if __name__ == "__main__":

    create_compact_labels()
