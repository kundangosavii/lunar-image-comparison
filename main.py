from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from loader import load_image, normalize
from Preprocessing.Georeferencing import georeferencing
from Preprocessing.intensity_normalisation import intensity_normalization
from Preprocessing.Resolution_Resampling import Resolution_Resampling


ROOT = Path(__file__).resolve().parent

SOURCE_PATH = ROOT / "data" / "img_data" / "ohr_lunar_src_1.jpeg"
REFERENCE_PATH = ROOT / "data" / "img_data" / "nac_lunar_ref_1.png"


# ============================================================
# 1. LOAD
# ============================================================

source = load_image(str(SOURCE_PATH), gray=True)
reference = load_image(str(REFERENCE_PATH), gray=True)

print(f"✓ Source image loaded: {source.shape}")
print(f"✓ Reference image loaded: {reference.shape}")


# ============================================================
# 2. NORMALIZATION
# ============================================================

source_normalized = normalize(source)
reference_normalized = normalize(reference)

print("✓ Images normalized to [0, 1]")


# ============================================================
# 3. GEOREFERENCING
# ============================================================

print("\nApplying georeferencing...")

source_georeferenced = georeferencing(
    source_normalized,
    reference_normalized
)

print("✓ Georeferencing complete")

print(
    f"  Source: "
    f"{source_normalized.shape} "
    f"→ "
    f"{source_georeferenced.shape}"
)

print(
    f"  Reference: "
    f"{reference_normalized.shape}"
)


# ============================================================
# 4. RESOLUTION RESAMPLING
# ============================================================

print("\nApplying resolution resampling...")

source_resampled, reference_resampled = Resolution_Resampling(
    source_georeferenced,
    reference_normalized,
    SOURCE_PATH.name,
    REFERENCE_PATH.name,
    method="bilinear"
)

print("✓ Resolution resampling complete")

print(
    f"  Source: "
    f"{source_georeferenced.shape} "
    f"→ "
    f"{source_resampled.shape}"
)

print(
    f"  Reference: "
    f"{reference_normalized.shape} "
    f"→ "
    f"{reference_resampled.shape}"
)


# ============================================================
# 5. INTENSITY NORMALIZATION
# ============================================================

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

print("✓ Intensity normalization complete")

cv2.imwrite(
    str(ROOT / "source_preprocessed_result.png"),
    source_final
)

cv2.imwrite(
    str(ROOT / "reference_preprocessed_result.png"),
    reference_final
)

# ============================================================
# 6. FINAL OVERLAY
# ============================================================


print("Before overlay:")
print("  Source:", source_final.shape)
print("  Reference:", reference_final.shape)

if source_final.shape != reference_final.shape:

    reference_final = cv2.resize(
        reference_final,
        (
            source_final.shape[1],
            source_final.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

print("After resizing:")
print("  Source:", source_final.shape)
print("  Reference:", reference_final.shape)

final_overlay = cv2.addWeighted(
    reference_final,
    0.5,
    source_final,
    0.5,
    0
)













final_overlay = cv2.addWeighted(
    reference_final,
    0.5,
    source_final,
    0.5,
    0
)

cv2.imwrite(
    str(ROOT / "final_preprocessed_result.png"),
    final_overlay
)

print("✓ Final result saved")


# ============================================================
# 7. VISUALIZATION
# ============================================================

stage_rows = [
    (
        "Original",
        source,
        reference
    ),
    (
        "Normalized",
        source_normalized,
        reference_normalized
    ),
    (
        "After Georeferencing",
        source_georeferenced,
        reference_normalized
    ),
    (
        "After Resolution Resampling",
        source_resampled,
        reference_resampled
    ),
    (
        "After Intensity Normalization",
        source_final,
        reference_final
    ),
]


fig = plt.figure(
    figsize=(16, 4.3 * (len(stage_rows) + 1))
)

fig.suptitle(
    "Lunar Image Preprocessing Pipeline",
    fontsize=17,
    fontweight="bold"
)

gs = fig.add_gridspec(
    len(stage_rows) + 1,
    2,
    left=0.12,
    right=0.88,
    top=0.92,
    bottom=0.06,
    hspace=0.42,
    wspace=0.18
)


for idx, (title, left_image, right_image) in enumerate(stage_rows):

    left_ax = fig.add_subplot(gs[idx, 0])
    right_ax = fig.add_subplot(gs[idx, 1])

    left_ax.imshow(
        left_image,
        cmap="gray"
    )

    left_ax.set_title(
        f"{title} - Source"
    )

    left_ax.axis("off")

    right_ax.imshow(
        right_image,
        cmap="gray"
    )

    right_ax.set_title(
        f"{title} - Reference"
    )

    right_ax.axis("off")


# Final result

final_ax = fig.add_subplot(
    gs[len(stage_rows), 0]
)

final_ax.imshow(
    final_overlay,
    cmap="gray"
)

final_ax.set_title(
    "Final Registered Result"
)

final_ax.axis("off")


blank_ax = fig.add_subplot(
    gs[len(stage_rows), 1]
)

blank_ax.axis("off")


output_path = (
    ROOT /
    "lunar_preprocessing_pipeline.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(
    f"✓ Saved pipeline visualization: "
    f"{output_path}"
)