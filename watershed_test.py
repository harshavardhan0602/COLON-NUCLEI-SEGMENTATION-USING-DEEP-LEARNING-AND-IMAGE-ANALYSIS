import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.metrics import mean_absolute_error
from skimage.measure import regionprops


# -----------------------------
# Load dataset
# -----------------------------
images = np.load("images.npy")
labels = np.load("labels.npy")

print("Dataset size:", images.shape)


# -----------------------------
# Metric Functions
# -----------------------------
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()

    return (2 * intersection) / (
        y_true.sum() + y_pred.sum() + 1e-8
    )


# -----------------------------
# Watershed Function
# -----------------------------
def apply_watershed(image):

    # Step 1: Grayscale threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, gray_thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 2: HSV purple filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_purple = np.array([90, 30, 20])
    upper_purple = np.array([180, 255, 255])

    purple_mask = cv2.inRange(
        hsv,
        lower_purple,
        upper_purple
    )

    # Step 3: Combine masks
    thresh = cv2.bitwise_and(
        gray_thresh,
        purple_mask
    )

    # Step 4: Morphological cleanup
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Step 5: Distance transform
    distance = ndimage.distance_transform_edt(
        opening
    )

    # Step 6: Marker detection
    coords = peak_local_max(
        distance,
        min_distance=7,
        threshold_abs=0.18 * distance.max(),
        labels=opening
    )

    mask = np.zeros(
        distance.shape,
        dtype=bool
    )

    if len(coords) > 0:
        mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(mask)

    # Step 7: Watershed
    labels_ws = watershed(
        -distance,
        markers,
        mask=opening
    )

    # Step 8: Count valid nuclei
    valid_count = 0

    for region in regionprops(labels_ws):
        if 15 < region.area < 400:
            valid_count += 1

    pred_count = valid_count

    pred_mask = (labels_ws > 0).astype(np.uint8)

    return thresh, labels_ws, pred_mask, pred_count


# -----------------------------
# Dataset Evaluation
# -----------------------------
iou_scores = []
dice_scores = []

gt_counts = []
pred_counts = []

for idx in range(len(images)):

    image = images[idx]

    gt_instance = labels[idx, :, :, 0]
    gt_mask = (gt_instance > 0).astype(np.uint8)

    gt_count = len(np.unique(gt_instance)) - 1

    thresh, labels_ws, pred_mask, pred_count = apply_watershed(image)

    gt_counts.append(gt_count)
    pred_counts.append(pred_count)

    iou_scores.append(
        calculate_iou(gt_mask, pred_mask)
    )

    dice_scores.append(
        calculate_dice(gt_mask, pred_mask)
    )


# -----------------------------
# Final Results
# -----------------------------
avg_iou = np.mean(iou_scores)
avg_dice = np.mean(dice_scores)

count_mae = mean_absolute_error(
    gt_counts,
    pred_counts
)

print("\n===================================")
print("WATERSHED FINAL RESULTS")
print("===================================")

print(f"Average IoU   : {avg_iou:.4f}")
print(f"Average Dice  : {avg_dice:.4f}")
print(f"Count MAE     : {count_mae:.4f}")

print(f"Average GT Count   : {np.mean(gt_counts):.2f}")
print(f"Average Pred Count : {np.mean(pred_counts):.2f}")

print("===================================\n")


# -----------------------------
# Visualize 5 samples
# -----------------------------
sample_indices = [10, 25, 50, 100, 150]

fig, axes = plt.subplots(
    len(sample_indices),
    4,
    figsize=(24, 6 * len(sample_indices))
)

for i, sample_idx in enumerate(sample_indices):

    image = images[sample_idx]

    gt_instance = labels[sample_idx, :, :, 0]
    gt_count = len(np.unique(gt_instance)) - 1

    thresh, labels_ws, pred_mask, pred_count = apply_watershed(image)

    # Original image
    axes[i,0].imshow(image)
    axes[i,0].set_title(
        f"Original Image\nSample {sample_idx}",
        fontsize=12
    )
    axes[i,0].axis("off")

    # Ground truth
    axes[i,1].imshow(
        gt_instance,
        cmap='nipy_spectral'
    )
    axes[i,1].set_title(
        f"Ground Truth\nCount: {gt_count}",
        fontsize=12
    )
    axes[i,1].axis("off")

    # Threshold mask
    axes[i,2].imshow(
        thresh,
        cmap='gray'
    )
    axes[i,2].set_title(
        "Threshold Mask",
        fontsize=12
    )
    axes[i,2].axis("off")

    # Watershed output
    axes[i,3].imshow(
        labels_ws,
        cmap='nipy_spectral'
    )
    axes[i,3].set_title(
        f"Watershed Output\nCount: {pred_count}",
        fontsize=12
    )
    axes[i,3].axis("off")


plt.subplots_adjust(
    hspace=0.5,
    wspace=0.3
)

plt.savefig(
    "watershed_final_results.png",
    dpi=300,
    bbox_inches='tight'
)

print("Visualization saved successfully")

plt.show()