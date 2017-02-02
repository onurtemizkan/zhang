import numpy as np


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

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    tmpX = B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]
    tmpY = B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1]

    v0 = tmpX / tmpY
    ld = B[2, 2] - (B[0, 2] * B[0, 2] + v0 * tmpX) / B[0, 0]

    if ld < 0:
        ld = 0.001

    alpha = np.sqrt(ld / B[0, 0])
    beta = np.sqrt(ld * B[0, 0] / tmpY)
    gamma = -B[0, 1] * alpha * alpha * beta / ld
    u0 = gamma * v0 / beta - B[0, 2] * alpha * alpha / ld

    return np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,      1]
    ])
