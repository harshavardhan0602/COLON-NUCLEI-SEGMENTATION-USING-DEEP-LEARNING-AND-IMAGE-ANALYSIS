import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    mean_absolute_error
)


# -----------------------------------
# Load dataset
# -----------------------------------
images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32)/255.0
masks = (labels[:,:,:,0] > 0).astype(np.uint8)


# -----------------------------------
# Same split as training
# -----------------------------------
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    images,
    masks,
    labels,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Load model
# -----------------------------------
model = tf.keras.models.load_model(
    "unet_model_15epochs.h5",
    compile=False
)

print("U-Net model loaded successfully")


# -----------------------------------
# Predict full test set
# -----------------------------------
preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(np.uint8)


# -----------------------------------
# Metrics
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

for i in range(len(y_test)):

    gt_mask = y_test[i]
    pred_mask = preds_binary[i].squeeze()

    iou_scores.append(
        calculate_iou(gt_mask, pred_mask)
    )

    dice_scores.append(
        calculate_dice(gt_mask, pred_mask)
    )

    gt_count = len(
        np.unique(
            labels_test[i,:,:,0]
        )
    ) - 1

    _, pred_count = ndimage.label(pred_mask)

    gt_counts.append(gt_count)
    pred_counts.append(pred_count)


precision = precision_score(
    y_test.flatten(),
    preds_binary.flatten()
)

recall = recall_score(
    y_test.flatten(),
    preds_binary.flatten()
)

count_mae = mean_absolute_error(
    gt_counts,
    pred_counts
)


# -----------------------------------
# Final Results
# -----------------------------------
print("\n===================================")
print("U-NET FINAL RESULTS (15 EPOCHS)")
print("===================================")

print(f"Train Samples: {len(X_train)}")
print(f"Test Samples : {len(X_test)}")

print("\nSegmentation Metrics")
print(f"Average IoU : {np.mean(iou_scores):.4f}")
print(f"Average Dice: {np.mean(dice_scores):.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")

print("\nCounting Metrics")
print(f"Count MAE   : {count_mae:.4f}")

print("===================================\n")


# -----------------------------------
# Visualization
# -----------------------------------
sample_indices = [10,25,50,100,150]

fig, axes = plt.subplots(
    len(sample_indices),
    3,
    figsize=(18,5*len(sample_indices))
)

for i, sample_idx in enumerate(sample_indices):

    image = images[sample_idx]
    gt_instance = labels[sample_idx,:,:,0]

    gt_mask = (gt_instance > 0).astype(np.uint8)

    gt_count = len(np.unique(gt_instance)) - 1

    pred = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0]

    pred_mask = (pred > 0.5).astype(np.uint8).squeeze()

    _, pred_count = ndimage.label(pred_mask)

    # Original
    axes[i,0].imshow(image)
    axes[i,0].set_title(
        f"Original Image\nSample {sample_idx}"
    )
    axes[i,0].axis("off")

    # GT
    axes[i,1].imshow(
        gt_mask,
        cmap="gray"
    )
    axes[i,1].set_title(
        f"Ground Truth\nCount: {gt_count}"
    )
    axes[i,1].axis("off")

    # Prediction
    axes[i,2].imshow(
        pred_mask,
        cmap="gray"
    )
    axes[i,2].set_title(
        f"Detected Nuclei\nCount: {pred_count}"
    )
    axes[i,2].axis("off")


plt.subplots_adjust(
    hspace=0.6,
    wspace=0.3
)

plt.savefig(
    "unet_15epoch_results.png",
    dpi=300,
    bbox_inches="tight"
)

print("Visualization saved successfully")

plt.show()