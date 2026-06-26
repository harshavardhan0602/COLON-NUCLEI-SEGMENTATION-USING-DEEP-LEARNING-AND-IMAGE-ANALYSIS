import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import mean_absolute_error
from skimage.measure import label


# =====================================================
# Load Dataset
# =====================================================
print("Loading dataset...")

images = np.load("images.npy")
labels_data = np.load("labels.npy")

images = images.astype(np.float32) / 255.0

true_masks = (labels_data[:, :, :, 0] > 0).astype(np.float32)
true_masks = np.expand_dims(true_masks, axis=-1)

print("Images shape:", images.shape)
print("Masks shape:", true_masks.shape)


# =====================================================
# Dice Metric
# =====================================================
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2 * intersection + 1e-7
    ) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) + 1e-7
    )


# =====================================================
# Dice Loss
# =====================================================
def dice_loss(y_true, y_pred):
    smooth = 1e-7

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - (
        (2 * intersection + smooth) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )


# =====================================================
# Focal Loss
# =====================================================
def focal_loss(y_true, y_pred,
               alpha=0.25,
               gamma=2.0):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_pred = K.clip(y_pred, 1e-7, 1-1e-7)

    pt = tf.where(
        K.equal(y_true,1),
        y_pred,
        1-y_pred
    )

    loss = -alpha * K.pow(
        1-pt,
        gamma
    ) * K.log(pt)

    return K.mean(loss)


# =====================================================
# Hybrid Loss
# =====================================================
def hybrid_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    d_loss = dice_loss(y_true, y_pred)
    f_loss = focal_loss(y_true, y_pred)

    return (
        0.3*K.mean(bce) +
        0.4*d_loss +
        0.3*f_loss
    )


# =====================================================
# Load Model
# =====================================================
print("Loading trained model...")

model = tf.keras.models.load_model(
    "best_multiscale_jpfm_unetpp.keras",
    custom_objects={
        "hybrid_loss": hybrid_loss,
        "dice_coef": dice_coef
    }
)

print("Model loaded successfully")


# =====================================================
# Metric Functions
# =====================================================
def calculate_iou(y_true, y_pred):

    intersection = np.logical_and(
        y_true,
        y_pred
    ).sum()

    union = np.logical_or(
        y_true,
        y_pred
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(y_true, y_pred):

    intersection = np.logical_and(
        y_true,
        y_pred
    ).sum()

    return (
        2 * intersection
    ) / (
        y_true.sum() +
        y_pred.sum() +
        1e-8
    )


def calculate_precision(y_true, y_pred):

    tp = np.logical_and(
        y_true==1,
        y_pred==1
    ).sum()

    fp = np.logical_and(
        y_true==0,
        y_pred==1
    ).sum()

    return tp / (tp + fp + 1e-8)


def calculate_recall(y_true, y_pred):

    tp = np.logical_and(
        y_true==1,
        y_pred==1
    ).sum()

    fn = np.logical_and(
        y_true==1,
        y_pred==0
    ).sum()

    return tp / (tp + fn + 1e-8)


# =====================================================
# Predict Entire Dataset
# =====================================================
print("Generating predictions...")

predictions = model.predict(
    images,
    batch_size=8,
    verbose=1
)

pred_masks = (
    predictions > 0.5
).astype(np.uint8)


# =====================================================
# Evaluate
# =====================================================
iou_scores = []
dice_scores = []
precision_scores = []
recall_scores = []

gt_counts = []
pred_counts = []

for i in range(len(images)):

    gt_mask = true_masks[i].squeeze().astype(np.uint8)
    pred_mask = pred_masks[i].squeeze().astype(np.uint8)

    iou_scores.append(
        calculate_iou(
            gt_mask,
            pred_mask
        )
    )

    dice_scores.append(
        calculate_dice(
            gt_mask,
            pred_mask
        )
    )

    precision_scores.append(
        calculate_precision(
            gt_mask,
            pred_mask
        )
    )

    recall_scores.append(
        calculate_recall(
            gt_mask,
            pred_mask
        )
    )

    # Ground truth nuclei count
    gt_instance = labels_data[i,:,:,0]
    gt_count = len(
        np.unique(gt_instance)
    ) - 1

    gt_counts.append(gt_count)

    # Predicted nuclei count
    labeled_pred = label(pred_mask)
    pred_count = len(
        np.unique(labeled_pred)
    ) - 1

    pred_counts.append(pred_count)


count_mae = mean_absolute_error(
    gt_counts,
    pred_counts
)


# =====================================================
# Final Results
# =====================================================
print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")
print(f"Average IoU : {np.mean(iou_scores):.4f}")
print(f"Average Dice: {np.mean(dice_scores):.4f}")
print(f"Precision   : {np.mean(precision_scores):.4f}")
print(f"Recall      : {np.mean(recall_scores):.4f}")
print(f"Count MAE   : {count_mae:.4f}")
print("===================================")


# =====================================================
# Visualize 5 Samples
# =====================================================
sample_indices = [10, 25, 50, 100, 150]

fig, axes = plt.subplots(
    len(sample_indices),
    3,
    figsize=(18, 6*len(sample_indices))
)

for i, idx in enumerate(sample_indices):

    original = images[idx]
    gt_mask = true_masks[idx].squeeze()
    pred_mask = pred_masks[idx].squeeze()

    gt_instance = labels_data[idx,:,:,0]
    gt_count = len(np.unique(gt_instance)) - 1

    pred_count = len(
        np.unique(
            label(pred_mask)
        )
    ) - 1


    # Original image
    axes[i,0].imshow(original)
    axes[i,0].set_title(
        f"Original Image\nSample {idx}",
        fontsize=12
    )
    axes[i,0].axis("off")


    # Ground Truth
    axes[i,1].imshow(
        gt_mask,
        cmap="gray"
    )
    axes[i,1].set_title(
        f"Ground Truth\nCount: {gt_count}",
        fontsize=12
    )
    axes[i,1].axis("off")


    # Prediction
    axes[i,2].imshow(
        pred_mask,
        cmap="gray"
    )
    axes[i,2].set_title(
        f"Prediction\nCount: {pred_count}",
        fontsize=12
    )
    axes[i,2].axis("off")


plt.subplots_adjust(
    hspace=0.6,
    wspace=0.3
)

plt.savefig(
    "multiscale_jpfm_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved successfully!")