from steps.parser import parse_data
from steps.dlt import compute_homography
from steps.intrinsics import get_camera_intrinsics
from steps.extrinsics import get_camera_extrinsics
from steps.distortion import estimate_lens_distortion
from utils.timer import timer


def calibrate():
    data = parse_data()

    end = timer()
    homographies = compute_homography(data)
    end("Homography Estimation")
    print "homographies"
    print homographies

    end = timer()
    intrinsics = get_camera_intrinsics(homographies)
    end("Intrinsics")

    print "intrinsics"
    print intrinsics

    end = timer()
    extrinsics = get_camera_extrinsics(intrinsics, homographies)
    end("Extrinsics")

    print "extrinsics"
    print extrinsics

    end = timer()
    distortion = estimate_lens_distortion(
        intrinsics,
        extrinsics,
        data["real"],
        data["sensed"]
    )
    end("Distortion")

    return

calibrate()
