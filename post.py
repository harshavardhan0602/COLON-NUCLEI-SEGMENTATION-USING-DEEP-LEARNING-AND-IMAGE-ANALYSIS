import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
from tensorflow.keras.models import load_model


# -----------------------------------
# Custom Functions
# -----------------------------------
def dice_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)

    return (2 * intersection + 1e-7) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) + 1e-7
    )


def dice_loss(y_true, y_pred):
    smooth = 1e-7

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - (
        (2 * intersection + smooth) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_pred = K.clip(y_pred, 1e-7, 1-1e-7)

    pt = tf.where(
        K.equal(y_true,1),
        y_pred,
        1-y_pred
    )

    loss = -alpha * K.pow(1-pt, gamma) * K.log(pt)

    return K.mean(loss)


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


# -----------------------------------
# Load Dataset
# -----------------------------------
print("Loading dataset...")

images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32)/255.0

masks = (labels[:,:,:,0] > 0).astype(np.uint8)
masks = np.expand_dims(masks, axis=-1)


X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    images,
    masks,
    labels,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Load trained model
# -----------------------------------
model = load_model(
    "best_attention_unetpp_hybrid.keras",
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "hybrid_loss": hybrid_loss
    }
)

print("Model loaded successfully")


# -----------------------------------
# Predict
# -----------------------------------
preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(np.uint8)


# -----------------------------------
# Watershed on Prediction Mask
# -----------------------------------
def watershed_on_prediction(pred_mask):

    pred_mask = pred_mask.astype(np.uint8)

    # distance transform
    distance = ndimage.distance_transform_edt(pred_mask)

    # detect local peaks
    coords = peak_local_max(
        distance,
        min_distance=4,
        labels=pred_mask
    )

    marker_mask = np.zeros(distance.shape, dtype=bool)

    if len(coords) > 0:
        marker_mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(marker_mask)

    # watershed split
    labels_ws = watershed(
        -distance,
        markers,
        mask=pred_mask
    )

    final_mask = (labels_ws > 0).astype(np.uint8)

    pred_count = len(np.unique(labels_ws)) - 1

    return final_mask, labels_ws, pred_count


# -----------------------------------
# Metrics
# -----------------------------------
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

    return (2*intersection)/(
        y_true.sum() +
        y_pred.sum() +
        1e-8
    )


iou_scores = []
dice_scores = []

gt_counts = []
pred_counts = []

watershed_predictions = []


for i in range(len(y_test)):

    gt_mask = y_test[i].squeeze()
    pred_mask = preds_binary[i].squeeze()

    final_mask, labels_ws, pred_count = watershed_on_prediction(pred_mask)

    watershed_predictions.append(final_mask)

    iou_scores.append(
        calculate_iou(
            gt_mask,
            final_mask
        )
    )

    dice_scores.append(
        calculate_dice(
            gt_mask,
            final_mask
        )
    )

    gt_count = len(
        np.unique(
            labels_test[i,:,:,0]
        )
    ) - 1

    gt_counts.append(gt_count)
    pred_counts.append(pred_count)


precision = precision_score(
    y_test.flatten(),
    np.array(watershed_predictions).flatten()
)

recall = recall_score(
    y_test.flatten(),
    np.array(watershed_predictions).flatten()
)

count_mae = mean_absolute_error(
    gt_counts,
    pred_counts
)


# -----------------------------------
# Final Results
# -----------------------------------
print("\n===================================")
print("POST PROCESSED RESULTS")
print("Watershed on Predictions")
print("===================================")

print(f"Average IoU : {np.mean(iou_scores):.4f}")
print(f"Average Dice: {np.mean(dice_scores):.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Count MAE   : {count_mae:.4f}")

print("===================================")


# -----------------------------------
# Visualization
# -----------------------------------
sample_indices = [10,25,50,100,150]

fig, axes = plt.subplots(
    len(sample_indices),
    4,
    figsize=(20,25)
)

for i, idx in enumerate(sample_indices):

    image = images[idx]
    gt_mask = masks[idx].squeeze()

    raw_pred = preds_binary[idx].squeeze()

    final_mask, labels_ws, pred_count = watershed_on_prediction(raw_pred)

    gt_count = len(
        np.unique(
            labels[idx,:,:,0]
        )
    ) - 1

    # Original
    axes[i,0].imshow(image)
    axes[i,0].set_title(
        f"Original\nSample {idx}"
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

    # Raw prediction
    axes[i,2].imshow(
        raw_pred,
        cmap="gray"
    )
    axes[i,2].set_title(
        "Raw Prediction"
    )
    axes[i,2].axis("off")

    # Watershed output
    axes[i,3].imshow(
        labels_ws,
        cmap="nipy_spectral"
    )
    axes[i,3].set_title(
        f"Watershed Split\nCount: {pred_count}"
    )
    axes[i,3].axis("off")


plt.subplots_adjust(
    hspace=0.6,
    wspace=0.3
)

plt.savefig(
    "watershed_prediction_results.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()