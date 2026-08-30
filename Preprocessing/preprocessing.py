import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def align_images(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align source image to reference image coordinate system.
    """

    src_h, src_w = source.shape[:2]
    ref_h, ref_w = reference.shape[:2]
    

    if src_h == ref_h and src_w == ref_w:
        return source

    aligned = cv2.resize(source, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    
    return aligned


def reproject_by_bounds(image: np.ndarray, 
                       source_bounds: Dict, 
                       target_bounds: Dict,
                       target_shape: Tuple) -> np.ndarray:
    """
    Reproject image based on geographic bounds.
    """
    src_h, src_w = image.shape[:2]
    tgt_h, tgt_w = target_shape
    

    tgt_y, tgt_x = np.mgrid[0:tgt_h, 0:tgt_w]
    
    # Normalize target coordinates to [0, 1]
    norm_y = tgt_y / (tgt_h - 1) if tgt_h > 1 else tgt_y
    norm_x = tgt_x / (tgt_w - 1) if tgt_w > 1 else tgt_x
    
    src_lat_min = source_bounds['south']
    src_lat_max = source_bounds['north']
    src_lon_min = source_bounds['west']
    src_lon_max = source_bounds['east']
    
    tgt_lat_min = target_bounds['south']
    tgt_lat_max = target_bounds['north']
    tgt_lon_min = target_bounds['west']
    tgt_lon_max = target_bounds['east']
    
    # Geographic coordinates in target system
    tgt_lats = tgt_lat_max - norm_y * (tgt_lat_max - tgt_lat_min)
    tgt_lons = tgt_lon_min + norm_x * (tgt_lon_max - tgt_lon_min)
    
    # Map to source pixel coordinates
    src_py = (src_lat_max - tgt_lats) / (src_lat_max - src_lat_min) * (src_h - 1)
    src_px = (tgt_lons - src_lon_min) / (src_lon_max - src_lon_min) * (src_w - 1)
    
    # Clip to valid range
    src_px = np.clip(src_px, 0, src_w - 1).astype(np.float32)
    src_py = np.clip(src_py, 0, src_h - 1).astype(np.float32)
    
    # Apply remap with bilinear interpolation
    reprojected = cv2.remap(
        image.astype(np.float32),
        src_px, src_py,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return reprojected


def georeferencing(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align source image to reference image.
    """
    return align_images(source, reference)



def intensity_normalization(image, low_percentile=2, high_percentile=98):

    image_float = image.astype(np.float32)

    original_min = image_float.min()
    original_max = image_float.max()

    low = np.percentile(image_float, low_percentile)
    high = np.percentile(image_float, high_percentile)

    lower_pixels = np.sum(image_float < low)
    upper_pixels = np.sum(image_float > high)

    total_pixels = image_float.size

    lower_percentage = (lower_pixels / total_pixels) * 100
    upper_percentage = (upper_pixels / total_pixels) * 100

    if high <= low:
        return image.astype(np.uint8)

    image_clipped = np.clip(image_float, low, high)

    normalized = (image_clipped - low) / (high - low)
    normalized = (normalized * 255).astype(np.uint8)


    return normalized

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

def process_all(source, reference):
    source_original = source.copy()
    reference_original = reference.copy()

    source_normalized = normalize(source)
    reference_normalized = normalize(reference)

    source_georeferenced = georeferencing(
    source_normalized,
    reference_normalized
    )

    source_resampled, reference_resampled = Resolution_Resampling(
    source_georeferenced,
    reference_normalized,
    source_original,
    reference_original,
    method="bilinear"
    )


    source_final = intensity_normalization(
    (source_resampled * 255).astype(np.uint8),
    low_percentile=2,
    high_percentile=98
    )

    reference_final = intensity_normalization(
    (reference_resampled * 255).astype(np.uint8),
    low_percentile=2,
    high_percentile=98
    )

    return [source_final, reference_final]