import numpy as np
from utils.timer import timer



def v(p, q, H):

    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])


def get_camera_intrinsics(homographies):

    h_count = len(homographies)

    vec = []

    for i in range(0, h_count):
        curr = np.reshape(homographies[i], (3, 3))

        vec.append(v(0, 1, curr))
        vec.append(v(0, 0, curr) - v(1, 1, curr))

    vec = np.array(vec)

    b = np.linalg.lstsq(
        vec,
        np.zeros(h_count * 2),
    )[-1]

    w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2 * b[1] * b[3] * b[4] - b[2] * b[3]**2
    d = b[0] * b[2] - b[1]**2

    # if (d < 0):
    #     d = 0.01
    d = -d

    #
    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ])
