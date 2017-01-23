from steps.Parser import parse_data
from steps.DLT import compute_homography
from steps.Intrinsics import get_camera_intrinsics
from steps.Extrinsics import get_camera_extrinsics


def calibrate():
    homographies = compute_homography(parse_data())

    intrinsics = get_camera_intrinsics(homographies)

    print "intrinsics"
    print intrinsics

    get_camera_extrinsics(intrinsics, homographies)
    return

calibrate()
