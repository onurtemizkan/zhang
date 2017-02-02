from steps.parser import parse_data
from steps.dlt import compute_homography
from steps.intrinsics import get_camera_intrinsics
from steps.extrinsics import get_camera_extrinsics
from steps.distortion import estimate_lens_distortion


def calibrate():
    data = parse_data()

    homographies = compute_homography(data)

    intrinsics = get_camera_intrinsics(homographies)

    # print "intrinsics"
    # print intrinsics

    extrinsics = get_camera_extrinsics(intrinsics, homographies)

    # print "extrinsics"
    # print extrinsics

    distortion = estimate_lens_distortion(
        intrinsics,
        extrinsics,
        data["real"],
        data["sensed"]
    )

    # print "distortion"
    # print distortion

    return

calibrate()
