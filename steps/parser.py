import numpy as np


def parse_data(basepath="data/corners_", ext=".dat"):
    sensed = []
    for i in range(1, 6):
        sensed.append(np.loadtxt(basepath + str(i) + ext).reshape((64, 4, 2)))

    return {
        'real': np.loadtxt(basepath + "real" + ext).reshape((64, 4, 2)),
        'sensed': sensed
    }
