import cv2
import numpy as np


SENSOR_RESOLUTION = {
    "OHR": 0.30,
    "NRC": 1.0,
    "NAC": 1.0,
    "TMC": 5.0,
    "IIRS": 80.0,
    "WAC": 100.0,
}


def detect_sensor_from_name(filename):
    name = str(filename).upper()

    for key in ("OHR", "NRC", "NAC", "TMC", "IIRS", "WAC"):
        if key in name:
            return key

    raise ValueError(f"Could not detect sensor from filename: {filename}")


def compute_new_size(width, height, source_resolution, target_resolution):
    scale = source_resolution / target_resolution

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return new_width, new_height


def Resolution_Resampling(
    source_image,
    reference_image,
    source_filename,
    reference_filename,
    method="bilinear",
):
    """
    Resample source and reference NumPy images to a common spatial resolution.

    Images are passed as NumPy arrays.
    Sensor resolutions are determined from the filenames.
    """

    source_sensor = detect_sensor_from_name(source_filename)
    reference_sensor = detect_sensor_from_name(reference_filename)

    source_resolution = SENSOR_RESOLUTION[source_sensor]
    reference_resolution = SENSOR_RESOLUTION[reference_sensor]

    # Use the coarser resolution as the common target.
    target_resolution = max(
        source_resolution,
        reference_resolution
    )

    print(
        f"Source sensor: {source_sensor} "
        f"({source_resolution} m/px)"
    )

    print(
        f"Reference sensor: {reference_sensor} "
        f"({reference_resolution} m/px)"
    )

    print(
        f"Compatible target resolution: "
        f"{target_resolution} m/px"
    )

    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    interpolation = interpolation_map[method]

    # -------------------------
    # Source
    # -------------------------

    source_h, source_w = source_image.shape[:2]

    if source_resolution != target_resolution:

        new_w, new_h = compute_new_size(
            source_w,
            source_h,
            source_resolution,
            target_resolution,
        )

        source_image = cv2.resize(
            source_image,
            (new_w, new_h),
            interpolation=interpolation,
        )

        print(
            f"Source: "
            f"{source_w}x{source_h} "
            f"-> "
            f"{new_w}x{new_h}"
        )

    else:

        print(
            f"Source: already at "
            f"{target_resolution} m/px"
        )

    # -------------------------
    # Reference
    # -------------------------

    reference_h, reference_w = reference_image.shape[:2]

    if reference_resolution != target_resolution:

        new_w, new_h = compute_new_size(
            reference_w,
            reference_h,
            reference_resolution,
            target_resolution,
        )

        reference_image = cv2.resize(
            reference_image,
            (new_w, new_h),
            interpolation=interpolation,
        )

        print(
            f"Reference: "
            f"{reference_w}x{reference_h} "
            f"-> "
            f"{new_w}x{new_h}"
        )

    else:

        print(
            f"Reference: already at "
            f"{target_resolution} m/px"
        )

    return source_image, reference_image