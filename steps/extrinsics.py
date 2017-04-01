import numpy as np


def estimate_view_transform(intrinsics, homography):

    homography = homography.reshape(3, 3)
    inv_intrinsics = np.linalg.inv(intrinsics)

    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]

    ld1 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h1))
    ld2 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h2))
    ld3 = (ld1 + ld2) / 2

    r0 = ld1 * np.dot(inv_intrinsics, h1)
    r1 = ld2 * np.dot(inv_intrinsics, h2)
    r2 = np.cross(r0, r1)

    t = np.array(ld3 * np.dot(inv_intrinsics, h3)).transpose()

    Rt = np.array(
            [r0.transpose(), r1.transpose(), r2.transpose(), t.transpose()]
         ).transpose()

    # todo: Add optimization?
    return Rt


def get_camera_extrinsics(intrinsics, homographies):

    extrinsics = []
    for i in range(0, len(homographies)):
        extrinsics.append(estimate_view_transform(intrinsics, homographies[i]))

    return extrinsics
