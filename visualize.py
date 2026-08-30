from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from loader import load_image, normalize
from Preprocessing.Georeferencing import georeferencing
from Preprocessing.intensity_normalisation import intensity_normalization


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data" / "img" / "ch2_nac_58947656958.png"
REFERENCE_PATH = ROOT / "data" / "img" / "ch2_nac_58947656958 copy.png"


def resample_to_reference(source_image: np.ndarray, reference_image: np.ndarray):
    """Resize the source image to match the reference dimensions for alignment."""
    target_height, target_width = reference_image.shape[:2]
    source_resampled = cv2.resize(
        source_image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return source_resampled, reference_image


def show_pair(fig, left_ax, right_ax, left_image, right_image, left_title, right_title):
    left_ax.imshow(left_image, cmap="gray")
    left_ax.axis("off")
    fig.text(
        left_ax.get_position().x0 - 0.02,
        0.5 * (left_ax.get_position().y0 + left_ax.get_position().y1),
        left_title,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    right_ax.imshow(right_image, cmap="gray")
    right_ax.axis("off")
    fig.text(
        right_ax.get_position().x0 - 0.02,
        0.5 * (right_ax.get_position().y0 + right_ax.get_position().y1),
        right_title,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def preprocess_images(source_image, reference_image):
    """Apply the preprocessing steps in the correct order for visualization."""
    source_normalized = normalize(source_image)
    reference_normalized = normalize(reference_image)

    source_georeferenced = georeferencing(source_normalized, reference_normalized)
    source_resampled, reference_resampled = resample_to_reference(
        source_georeferenced,
        reference_normalized,
    )

    source_processed = intensity_normalization(
        (source_resampled * 255).astype(np.uint8),
        low_percentile=2,
        high_percentile=98,
    )
    reference_processed = intensity_normalization(
        (reference_resampled * 255).astype(np.uint8),
        low_percentile=2,
        high_percentile=98,
    )

    final_overlay = cv2.addWeighted(reference_processed, 0.5, source_processed, 0.5, 0)
    return (
        source_image,
        reference_image,
        source_normalized,
        reference_normalized,
        source_georeferenced,
        reference_normalized,
        source_resampled,
        reference_resampled,
        source_processed,
        reference_processed,
        final_overlay,
    )


source = load_image(str(SOURCE_PATH), gray=True)
reference = load_image(str(REFERENCE_PATH), gray=True)
print(f"Source image loaded: {source.shape}")
print(f"Reference image loaded: {reference.shape}")

(
    original_source,
    original_reference,
    source_normalized,
    reference_normalized,
    source_georeferenced,
    reference_after_georeferencing,
    source_resampled,
    reference_resampled,
    source_final,
    reference_final,
    final_overlay,
) = preprocess_images(source, reference)

print("Preprocessing complete")
print(f"  Source shape: {source.shape} -> {source_georeferenced.shape}")
print(f"  Reference shape: {reference.shape}")

cv2.imwrite("final_preprocessed_result.png", final_overlay)

stage_rows = [
    ("Original", original_source, original_reference),
    ("Normalized", source_normalized, reference_normalized),
    ("After Georeferencing", source_georeferenced, reference_after_georeferencing),
    ("After Resolution Resampling", source_resampled, reference_resampled),
    ("After Intensity Normalization", source_final, reference_final),
]

fig = plt.figure(figsize=(16, 4.3 * (len(stage_rows) + 1)))
fig.suptitle("Lunar Image Preprocessing Pipeline", fontsize=17, fontweight="bold")

left_margin = 0.12
right_margin = 0.12
used_width = 1.0 - left_margin - right_margin

gs = fig.add_gridspec(len(stage_rows) + 1, 2, left=left_margin, right=1.0 - right_margin, top=0.92, bottom=0.06, hspace=0.42, wspace=0.18)

for idx, (title, left_image, right_image) in enumerate(stage_rows, start=0):
    left_ax = fig.add_subplot(gs[idx, 0])
    right_ax = fig.add_subplot(gs[idx, 1])
    show_pair(
        fig,
        left_ax,
        right_ax,
        left_image,
        right_image,
        f"{title} - Source",
        f"{title} - Reference",
    )

final_ax = fig.add_subplot(gs[len(stage_rows), 0])
final_ax.imshow(final_overlay, cmap="gray")
final_ax.axis("off")
fig.text(
    final_ax.get_position().x0 - 0.02,
    final_ax.get_position().y0 + final_ax.get_position().height / 2,
    "Final Registered Result",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
)

blank_ax = fig.add_subplot(gs[len(stage_rows), 1])
blank_ax.axis("off")

output_path = ROOT / "lunar_preprocessing_pipeline.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

if plt.get_backend().lower() != "agg":
    plt.show()
else:
    print(f"Saved pipeline visualization: {output_path}")