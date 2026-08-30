import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.core.loader import load_uploaded_image
from app.core.preprocess import process_all
from SuperGluePretrainedNetwork.superglue_matching import match_images


import base64
import cv2
import time


def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def run_pipeline(source_bytes, reference_bytes, source_sensor, reference_sensor):
    src = load_uploaded_image(source_bytes)
    ref = load_uploaded_image(reference_bytes)

    processed = process_all(
        src,
        ref,
        source_sensor=source_sensor,
        reference_sensor=reference_sensor
    )

    src_final = processed[0]
    ref_final = processed[1]

    start = time.perf_counter()
    result = match_images(src_final, ref_final)
    compute_time = time.perf_counter() - start

    match_img = result["image"]
    rmse = result["rmse"]
    inlier_count = result["inlier_count"]
    inlier_ratio = result["inlier_ratio"]

    return {
        "source_shape": list(src.shape),
        "reference_shape": list(ref.shape),
        "match_image": encode_image(match_img),
        "rmse": rmse,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "compute_time": compute_time,
    }