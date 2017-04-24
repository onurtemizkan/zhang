import numpy as np


def estimate_lens_distortion(intrinsics, extrinsics, model, sensor):

    uc = intrinsics[0, 2]
    vc = intrinsics[1, 2]

    D = []
    d = []

    l = 0

    for i in range(0, len(extrinsics)):
        for j in range(0, model.size/2):

            homog_model_coords = np.array([model[j][0], model[j][1], 0, 1])
            homog_coords = np.dot(extrinsics[i], homog_model_coords)

            coords = homog_coords / homog_coords[-1]
            [x, y, hom] = coords

            r = np.sqrt(x*x + y*y)

            P = np.dot(intrinsics, homog_coords)
            P = P / P[2]

            [u, v, trash] = P

            du = u - uc
            dv = v - vc

            D.append(
                np.array([
                    du * r**2, du * r**4
                ])
            )

            D.append(
                np.array([
                    dv * r**2, dv * r**4
                ])
            )

            up = sensor[i][j][0]
            vp = sensor[i][j][1]

            d.append(up - u)
            d.append(vp - v)

    k = np.linalg.lstsq(
        np.array(D),
        np.array(d)
    )

    return k
