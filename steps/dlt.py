import numpy as np
from scipy import optimize as opt


def get_normalisation_matrix(flattened_corners):

    avg_x = flattened_corners[0].mean()
    avg_y = flattened_corners[1].mean()

    s_x = np.sqrt(2 / flattened_corners[0].std())
    s_y = np.sqrt(2 / flattened_corners[1].std())

    return np.matrix([
        [s_x,   0,   -s_x * avg_x],
        [0,   s_y,   -s_y * avg_y],
        [0,     0,              1]
    ])


def unhomogeneous(x):
    return np.array([x[0], x[1]])


def estimate_homography(sensed, real):

    first_normalisation_matrix = get_normalisation_matrix(sensed)
    second_normalisation_matrix = get_normalisation_matrix(real)  # note: can be moved out

    M = np.array([])
    for j in range(0, sensed.size / 2):
        homogeneous_first = np.array([
            sensed[0][j],
            sensed[1][j],
            1
        ])

        homogeneous_second = np.array([
            real[0][j],
            real[1][j],
            1
        ])

        # prime is " ' " like a'
        # first prime
        pr_1 = np.apply_along_axis(
            unhomogeneous,
            1,
            np.dot(first_normalisation_matrix, homogeneous_first)[0]
        )[0]

        # second prime
        pr_2 = np.apply_along_axis(
            unhomogeneous,
            1,
            np.dot(second_normalisation_matrix, homogeneous_second)[0]
        )[0]

        M = np.append(M, np.array([
            pr_1[0], pr_1[1], 1,
            0, 0, 0,
            -pr_1[0]*pr_2[0], -pr_1[1]*pr_2[0], -pr_2[0]
        ]), axis=0)
        M = np.append(M, np.array([
            0, 0, 0, pr_1[0], pr_1[1],
            1, -pr_1[0]*pr_2[1], -pr_1[1]*pr_2[1], -pr_2[1]
        ]), axis=0)

    # U, S, V = np.linalg.svd()
    U, S, Vh = np.linalg.svd(M.reshape((512, 9)))

    L = Vh[-1]

    H = L.reshape(3, 3)

    denormalised = np.dot(
        np.dot(
            np.linalg.inv(second_normalisation_matrix),
            H
        ),
        first_normalisation_matrix
    )

    return denormalised


def cost(homography, points):

    Y = []

    for i in range(0, points.size / 2):
        x = points[0][i]
        y = points[1][i]

        w = homography[6] * x + homography[7] * y + homography[8]

        M = np.array([
            [homography[0], homography[1], homography[2]],
            [homography[3], homography[4], homography[5]]
        ])

        homog = np.transpose(np.array([x, y, 1]))
        [u, v] = (1/w) * np.dot(M, homog)

        Y.append(u)
        Y.append(v)

    return np.array(Y)


def jac(homography, points):
    J = []

    for i in range(0, points.size / 2):
        x = points[0][i]
        y = points[1][i]

        s_x = homography[0] * x + homography[1] * y + homography[2]
        s_y = homography[3] * x + homography[4] * y + homography[5]
        w = homography[6] * x + homography[7] * y + homography[8]

        J.append(
            np.array([
                x / w, y / w, 1/w,
                0, 0, 0,
                (-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
            ])
        )

        J.append(
            np.array([
                0, 0, 0,
                x / w, y / w, 1 / w,
                (-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
            ])
        )

    return np.array(J)


def refine_homography(homography, sensed):
    return opt.root(cost, homography, jac=jac, args=sensed, method='lm').x


def compute_homography(data):
    real = data['real'].reshape((2, 256))

    refined_homographies = []

    for i in range(0, len(data['sensed'])):
        sensed = data['sensed'][i].reshape((2, 256))
        estimated = estimate_homography(sensed, real)
        refined = refine_homography(estimated, sensed)
        refined = refined / refined[-1]
        refined_homographies.append(refined)

    return np.array(refined_homographies)
