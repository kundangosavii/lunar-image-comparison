import cv2
import torch
import numpy as np

from models.matching import Matching
from models.utils import frame2tensor


# ============================================================
# 1. IMAGE PATHS
# ============================================================

SOURCE_IMAGE = (
    r"C:\Users\JAHANVI\lunar-image-comparison\Matching\SuperGluePretrainedNetwork\source_preprocessed_result.png"
)

REFERENCE_IMAGE = (
    r"C:\Users\JAHANVI\lunar-image-comparison\Matching\SuperGluePretrainedNetwork\reference_preprocessed_result.png"
)


# ============================================================
# 2. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("======================================")
print("       SUPERGLUE IMAGE MATCHING")
print("======================================")

print("Device:", device)


# ============================================================
# 3. LOAD IMAGES
# ============================================================

source = cv2.imread(
    SOURCE_IMAGE,
    cv2.IMREAD_GRAYSCALE
)

reference = cv2.imread(
    REFERENCE_IMAGE,
    cv2.IMREAD_GRAYSCALE
)


if source is None:
    raise FileNotFoundError(
        "Source image not found. Check SOURCE_IMAGE path."
    )

if reference is None:
    raise FileNotFoundError(
        "Reference image not found. Check REFERENCE_IMAGE path."
    )


print("\nOriginal source size:")
print("Height:", source.shape[0])
print("Width :", source.shape[1])

print("\nOriginal reference size:")
print("Height:", reference.shape[0])
print("Width :", reference.shape[1])


# ============================================================
# 4. RESIZE FOR SUPERGLUE
# ============================================================

# This is ONLY for SuperGlue processing.
# The original images are not modified.

MAX_SIZE = 1200


def resize_image(image, max_size):

    height, width = image.shape

    largest_dimension = max(
        height,
        width
    )

    if largest_dimension <= max_size:
        return image

    scale = max_size / largest_dimension

    new_width = int(width * scale)

    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


source = resize_image(
    source,
    MAX_SIZE
)

reference = resize_image(
    reference,
    MAX_SIZE
)


print("\nImage size used by SuperGlue:")

print(
    "Source    :",
    source.shape
)

print(
    "Reference :",
    reference.shape
)

# ============================================================
# 5. SUPERPOINT + SUPERGLUE CONFIGURATION
# ============================================================

config = {

    "superpoint": {

        # Keep NMS unchanged
        "nms_radius": 3,

        # Lower threshold -> more detected keypoints
        "keypoint_threshold": 0.0001,

        # More keypoints available for matching
        "max_keypoints": 4096
    },

    "superglue": {

        "weights": "outdoor",

        "sinkhorn_iterations": 20,

        # Very permissive threshold to obtain more
        # candidate correspondences
        "match_threshold": 0.0
    }
}

# ============================================================
# 6. LOAD SUPERGLUE
# ============================================================

print("\nLoading SuperPoint + SuperGlue...")

matching = Matching(config)

matching = matching.eval().to(device)


# ============================================================
# 7. CONVERT IMAGES TO TENSORS
# ============================================================

source_tensor = frame2tensor(
    source,
    device
)

reference_tensor = frame2tensor(
    reference,
    device
)


# ============================================================
# 8. RUN SUPERPOINT + SUPERGLUE
# ============================================================

print("Running matching...")

with torch.no_grad():

    prediction = matching({
        "image0": source_tensor,
        "image1": reference_tensor
    })


# ============================================================
# 9. EXTRACT KEYPOINTS
# ============================================================

keypoints_source = (
    prediction["keypoints0"][0]
    .cpu()
    .numpy()
)

keypoints_reference = (
    prediction["keypoints1"][0]
    .cpu()
    .numpy()
)


# ============================================================
# 10. EXTRACT MATCHES
# ============================================================

matches = (
    prediction["matches0"][0]
    .cpu()
    .numpy()
)

matching_scores = (
    prediction["matching_scores0"][0]
    .cpu()
    .numpy()
)


# ============================================================
# 11. SELECT VALID SUPREGLUE MATCHES
# ============================================================

valid_matches = matches >= 0


matched_source = keypoints_source[
    valid_matches
]

matched_reference = keypoints_reference[
    matches[valid_matches]
]

matched_scores = matching_scores[
    valid_matches
]


# ============================================================
# 12. SORT BY SUPREGLUE CONFIDENCE
# ============================================================

if len(matched_scores) > 0:

    sort_indices = np.argsort(
        matched_scores
    )[::-1]

    matched_source = matched_source[
        sort_indices
    ]

    matched_reference = matched_reference[
        sort_indices
    ]

    matched_scores = matched_scores[
        sort_indices
    ]


# ============================================================
# 13. MATCH COUNT
# ============================================================

match_count = len(
    matched_source
)


print("\n======================================")
print("          MATCHING RESULTS")
print("======================================")

print(
    "Source keypoints    :",
    len(keypoints_source)
)

print(
    "Reference keypoints :",
    len(keypoints_reference)
)

print(
    "Total matches       :",
    match_count
)
# ============================================================
# 14. RANSAC INLIER CALCULATION
# ============================================================

inlier_mask = None
homography = None

if match_count >= 4:

    homography, inlier_mask = cv2.findHomography(
        matched_source,
        matched_reference,
        cv2.RANSAC,
        5.0
    )


# ============================================================
# 15. INLIER COUNT + INLIER RATIO + RMSE
# ============================================================

if inlier_mask is not None:

    inlier_mask = (
        inlier_mask
        .ravel()
        .astype(bool)
    )

    inlier_count = int(
        np.sum(inlier_mask)
    )

    inlier_ratio = (
        inlier_count / match_count
    )

else:

    inlier_count = 0

    inlier_ratio = 0.0


# ============================================================
# ACCURATE RMSE CALCULATION
# ============================================================

rmse = None


if (
    homography is not None
    and inlier_mask is not None
    and inlier_count >= 1
):

    # Take only RANSAC inliers
    inlier_source_points = (
        matched_source[inlier_mask]
        .reshape(-1, 1, 2)
        .astype(np.float32)
    )

    inlier_reference_points = (
        matched_reference[inlier_mask]
        .reshape(-1, 1, 2)
        .astype(np.float32)
    )


    # Transform source points using homography
    projected_source_points = cv2.perspectiveTransform(
        inlier_source_points,
        homography
    )


    # Convert back to Nx2
    projected_source_points = (
        projected_source_points
        .reshape(-1, 2)
    )

    actual_reference_points = (
        inlier_reference_points
        .reshape(-1, 2)
    )


    # Euclidean error for every inlier
    errors = np.linalg.norm(
        projected_source_points
        - actual_reference_points,
        axis=1
    )


    # RMSE
    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


print(
    "Inlier count        :",
    inlier_count
)

print(
    "Inlier ratio        :",
    f"{inlier_ratio:.4f}"
)

print(
    "Inlier percentage   :",
    f"{inlier_ratio * 100:.2f}%"
)


# ============================================================
# PRINT RMSE
# ============================================================

if rmse is not None:

    print(
        "RMSE                :",
        f"{rmse:.4f} pixels"
    )

else:

    print(
        "RMSE                :",
        "N/A"
    )


# ============================================================
# RMSE QUALITY INDICATION
# ============================================================

if rmse is not None:

    if rmse <= 3.0:

        rmse_status = "GOOD"

    elif rmse <= 5.0:

        rmse_status = "ACCEPTABLE"

    else:

        rmse_status = "HIGH"

else:

    rmse_status = "UNAVAILABLE"


print(
    "RMSE status         :",
    rmse_status
)
# ============================================================
# 16. SAVE MATCH DATA
# ============================================================

np.savez(
    "superglue_match_data.npz",

    source_keypoints=keypoints_source,

    reference_keypoints=keypoints_reference,

    source_matched_points=matched_source,

    reference_matched_points=matched_reference,

    match_scores=matched_scores,

    matches=matches,

    inlier_mask=inlier_mask,

    homography=homography,

    rmse=rmse
)


print(
    "\nMatch data saved as:"
)

print(
    "superglue_match_data.npz"
)

# ============================================================
# 17. CREATE MATCHING VISUALIZATION
# ============================================================

print("\nCreating matching visualization...")


# ------------------------------------------------------------
# VISUALIZATION SETTINGS
# ------------------------------------------------------------

# Maximum display height.
# Aspect ratio will be preserved.
DISPLAY_HEIGHT = 600

# Space between the two images
IMAGE_GAP = 300

# Outer margins
SIDE_MARGIN = 150
TOP_MARGIN = 120
BOTTOM_MARGIN = 100


# ------------------------------------------------------------
# FUNCTION TO RESIZE WITHOUT STRETCHING
# ------------------------------------------------------------

def resize_keep_aspect(image, target_height):

    h, w = image.shape[:2]

    scale = target_height / h

    new_width = int(
        w * scale
    )

    resized = cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


# ------------------------------------------------------------
# RESIZE WITHOUT DISTORTION
# ------------------------------------------------------------

source_display = resize_keep_aspect(
    source,
    DISPLAY_HEIGHT
)

reference_display = resize_keep_aspect(
    reference,
    DISPLAY_HEIGHT
)


# ------------------------------------------------------------
# GET DISPLAY DIMENSIONS
# ------------------------------------------------------------

source_display_height = source_display.shape[0]
source_display_width = source_display.shape[1]

reference_display_height = reference_display.shape[0]
reference_display_width = reference_display.shape[1]


# ------------------------------------------------------------
# SCALE FACTORS
# ------------------------------------------------------------

source_scale = (
    source_display_width /
    source.shape[1]
)

reference_scale = (
    reference_display_width /
    reference.shape[1]
)


# ------------------------------------------------------------
# CANVAS DIMENSIONS
# ------------------------------------------------------------

canvas_height = (
    DISPLAY_HEIGHT
    + TOP_MARGIN
    + BOTTOM_MARGIN
)

canvas_width = (
    SIDE_MARGIN
    + source_display_width
    + IMAGE_GAP
    + reference_display_width
    + SIDE_MARGIN
)


# ------------------------------------------------------------
# CREATE BLACK CANVAS
# ------------------------------------------------------------

canvas = np.ones(
    (
        canvas_height,
        canvas_width,
        3
    ),
    dtype=np.uint8
)*255


# ------------------------------------------------------------
# IMAGE POSITIONS
# ------------------------------------------------------------

source_x = SIDE_MARGIN
source_y = TOP_MARGIN


reference_x = (
    SIDE_MARGIN
    + source_display_width
    + IMAGE_GAP
)

reference_y = TOP_MARGIN


# ------------------------------------------------------------
# CONVERT GRAYSCALE TO COLOR
# ------------------------------------------------------------

source_color = cv2.cvtColor(
    source_display,
    cv2.COLOR_GRAY2BGR
)

reference_color = cv2.cvtColor(
    reference_display,
    cv2.COLOR_GRAY2BGR
)


# ------------------------------------------------------------
# PLACE SOURCE IMAGE
# ------------------------------------------------------------

canvas[
    source_y:
    source_y + source_display_height,

    source_x:
    source_x + source_display_width
] = source_color


# ------------------------------------------------------------
# PLACE REFERENCE IMAGE
# ------------------------------------------------------------

canvas[
    reference_y:
    reference_y + reference_display_height,

    reference_x:
    reference_x + reference_display_width
] = reference_color


# ============================================================
# 18. DRAW MATCHING LINES
# ============================================================

if match_count > 0:

    # We display up to 100 highest-confidence
    # SuperGlue matches.
    DISPLAY_MATCHES = min(
        match_count,
        100
    )


    # Multiple colors
    colors = [

        (0, 0, 255),       # Red

        (0, 165, 255),     # Orange

        (0, 255, 255),     # Yellow

        (0, 255, 0),       # Green

        (255, 255, 0),     # Cyan

        (255, 0, 0),       # Blue

        (255, 0, 255),     # Magenta

        (128, 0, 255),     # Purple

        (255, 128, 0),     # Light blue

        (0, 128, 255)      # Orange-red
    ]


    for i in range(DISPLAY_MATCHES):

        # Original source coordinate
        x1, y1 = matched_source[i]


        # Original reference coordinate
        x2, y2 = matched_reference[i]


        # ----------------------------------------------------
        # SCALE SOURCE COORDINATE
        # ----------------------------------------------------

        x1 = int(
            x1 * source_scale
        )

        y1 = int(
            y1 * source_scale
        )


        # ----------------------------------------------------
        # SCALE REFERENCE COORDINATE
        # ----------------------------------------------------

        x2 = int(
            x2 * reference_scale
        )

        y2 = int(
            y2 * reference_scale
        )


        # ----------------------------------------------------
        # MOVE TO CANVAS
        # ----------------------------------------------------

        x1 += source_x
        y1 += source_y

        x2 += reference_x
        y2 += reference_y


        # ----------------------------------------------------
        # SELECT COLOR
        # ----------------------------------------------------

        line_color = colors[
            i % len(colors)
        ]


        # ----------------------------------------------------
        # DRAW LINE
        # ----------------------------------------------------

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            line_color,
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # DRAW SOURCE MATCH POINT
        # ----------------------------------------------------

        cv2.circle(
            canvas,
            (x1, y1),
            2,
            line_color,
            -1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # DRAW REFERENCE MATCH POINT
        # ----------------------------------------------------

        cv2.circle(
            canvas,
            (x2, y2),
            2,
            line_color,
            -1,
            cv2.LINE_AA
        )


# ============================================================
# 19. ADD LABELS
# ============================================================

cv2.putText(
    canvas,
    "SOURCE",
    (
        source_x,
        TOP_MARGIN - 35
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


cv2.putText(
    canvas,
    "REFERENCE",
    (
        reference_x,
        TOP_MARGIN - 35
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# 20. RESULT INFORMATION
# ============================================================
# ============================================================
# RESULT INFORMATION
# ============================================================

if rmse is not None:

    result_text = (
        f"Matches: {match_count}    "
        f"Inliers: {inlier_count}    "
        f"Inlier Ratio: {inlier_ratio:.2%}    "
        f"RMSE: {rmse:.2f} px"
    )

else:

    result_text = (
        f"Matches: {match_count}    "
        f"Inliers: {inlier_count}    "
        f"Inlier Ratio: {inlier_ratio:.2%}    "
        f"RMSE: N/A"
    )


cv2.putText(
    canvas,
    result_text,
    (
        SIDE_MARGIN,
        canvas_height - 35
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# ============================================================
# RMSE QUALITY LABEL
# ============================================================

if rmse_status == "GOOD":

    rmse_label = "RMSE: GOOD"

    rmse_label_color = (
        0,
        150,
        0
    )

elif rmse_status == "ACCEPTABLE":

    rmse_label = "RMSE: ACCEPTABLE"

    rmse_label_color = (
        0,
        140,
        200
    )

elif rmse_status == "HIGH":

    rmse_label = "RMSE: HIGH"

    rmse_label_color = (
        0,
        0,
        200
    )

else:

    rmse_label = "RMSE: UNAVAILABLE"

    rmse_label_color = (
        80,
        80,
        80
    )


cv2.putText(
    canvas,
    rmse_label,
    (
        canvas_width - 350,
        canvas_height - 35
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    rmse_label_color,
    2,
    cv2.LINE_AA
)

# ============================================================
# 21. SAVE RESULT
# ============================================================

cv2.imwrite(
    "superglue_matching_result.png",
    canvas
)


print(
    "\nMatching visualization saved as:"
)

print(
    "superglue_matching_result.png"
)


print(
    "\nMatches displayed:",
    min(match_count, 100)
)


# ============================================================
# 22. FINISHED
# ============================================================

print("\n======================================")
print("          MATCHING COMPLETED")
print("======================================")