import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.metrics import mean_absolute_error

images = np.load("images.npy")
labels = np.load("labels.npy")

print("Dataset size:", images.shape)


# -----------------------------------
# Metric Functions
# -----------------------------------
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

iou_scores = []
dice_scores = []

gt_counts = []
pred_counts = []

for idx in range(len(images)):

    image = images[idx]

    # Ground truth instance labels
    gt_instance = labels[idx, :, :, 0]

    # Ground truth binary mask
    gt_mask = (gt_instance > 0).astype(np.uint8)

    # Exact ground truth nuclei count
    gt_count = len(np.unique(gt_instance)) - 1

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological opening
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Distance transform
    distance = ndimage.distance_transform_edt(opening)

    # Marker detection
    coords = peak_local_max(
        distance,
        footprint=np.ones((3,3)),
        labels=opening
    )

    mask = np.zeros(distance.shape, dtype=bool)

    if len(coords) > 0:
        mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(mask)

    # Watershed segmentation
    labels_ws = watershed(
        -distance,
        markers,
        mask=opening
    )

    # Predicted binary mask
    pred_mask = (labels_ws > 0).astype(np.uint8)

    # Exact predicted nuclei count
    pred_count = len(np.unique(labels_ws)) - 1

    # Store counts
    gt_counts.append(gt_count)
    pred_counts.append(pred_count)

    # Store segmentation metrics
    iou_scores.append(
        calculate_iou(gt_mask, pred_mask)
    )

    dice_scores.append(
        calculate_dice(gt_mask, pred_mask)
    )

avg_iou = np.mean(iou_scores)
avg_dice = np.mean(dice_scores)
count_mae = mean_absolute_error(gt_counts, pred_counts)

print("\n----- Overall Results -----")
print("Average IoU:", avg_iou)
print("Average Dice:", avg_dice)
print("Count MAE:", count_mae)

sample_idx = 10

image = images[sample_idx]

# Ground truth labels
gt_instance = labels[sample_idx, :, :, 0]

# Exact GT count for sample
gt_count = len(np.unique(gt_instance)) - 1


# Grayscale only for processing
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Thresholding
_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)


# Morphological cleanup
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


# Distance transform
distance = ndimage.distance_transform_edt(opening)


# Marker detection
coords = peak_local_max(
    distance,
    footprint=np.ones((3,3)),
    labels=opening
)

mask = np.zeros(distance.shape, dtype=bool)

if len(coords) > 0:
    mask[tuple(coords.T)] = True

markers, _ = ndimage.label(mask)


# Watershed output
labels_ws = watershed(
    -distance,
    markers,
    mask=opening
)

pred_count = len(np.unique(labels_ws)) - 1


print("\n----- Sample Result -----")
print(f"Sample Index: {sample_idx}")
print(f"Ground Truth Nuclei Count: {gt_count}")
print(f"Watershed Predicted Count: {pred_count}")

#OUTPUTS
plt.figure(figsize=(20,5))


# Original Image
plt.subplot(1,4,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")


# Ground Truth Labels
plt.subplot(1,4,2)
plt.imshow(gt_instance, cmap='nipy_spectral')
plt.title(f"Ground Truth Labels\nCount: {gt_count}")
plt.axis("off")


# Thresholded Mask
plt.subplot(1,4,3)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Mask")
plt.axis("off")


# Watershed Output
plt.subplot(1,4,4)
plt.imshow(labels_ws, cmap='nipy_spectral')
plt.title(f"Watershed Output\nCount: {pred_count}")
plt.axis("off")


plt.tight_layout()
plt.savefig("watershed_final_output.png")

print("Visualization saved successfully")