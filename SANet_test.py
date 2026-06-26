import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# -----------------------------------
# Load dataset
# -----------------------------------
images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32) / 255.0


# -----------------------------------
# Generate density maps
# -----------------------------------
def generate_density_map(instance_mask):
    h, w = instance_mask.shape
    density = np.zeros((h,w), dtype=np.float32)

    nuclei_ids = np.unique(instance_mask)
    nuclei_ids = nuclei_ids[nuclei_ids != 0]

    for nucleus_id in nuclei_ids:

        nucleus_mask = (instance_mask == nucleus_id)

        props = regionprops(
            nucleus_mask.astype(int)
        )

        if len(props) == 0:
            continue

        y, x = props[0].centroid
        y = int(y)
        x = int(x)

        density[y,x] = 1

    density = gaussian_filter(
        density,
        sigma=1.2
    )

    if density.sum() > 0:
        density = density * (
            len(nuclei_ids)/density.sum()
        )

    return density


density_maps = []

for i in range(len(labels)):
    density_maps.append(
        generate_density_map(
            labels[i,:,:,0]
        )
    )

density_maps = np.array(density_maps)
density_maps = np.expand_dims(
    density_maps,
    axis=-1
)


# -----------------------------------
# Same split as training
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images,
    density_maps,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Load model
# -----------------------------------
model = tf.keras.models.load_model(
    "sanet_model_15epochs.h5",
    compile=False
)

print("SANet model loaded successfully")


# -----------------------------------
# Predict
# -----------------------------------
preds = model.predict(X_test)

preds = np.maximum(preds, 0)


# -----------------------------------
# Count evaluation
# -----------------------------------
true_counts = []
pred_counts = []

for i in range(len(y_test)):
    true_count = y_test[i].sum()
    pred_count = preds[i].sum()

    true_counts.append(true_count)
    pred_counts.append(pred_count)


mae = mean_absolute_error(
    true_counts,
    pred_counts
)

rmse = np.sqrt(
    mean_squared_error(
        true_counts,
        pred_counts
    )
)


# -----------------------------------
# Final Results
# -----------------------------------
print("\n===================================")
print("SANET FINAL RESULTS (15 EPOCHS)")
print("===================================")

print(f"Train Samples: {len(X_train)}")
print(f"Test Samples : {len(X_test)}")

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

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

    gt_density = density_maps[sample_idx].squeeze()

    pred_density = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0].squeeze()

    pred_density = np.maximum(
        pred_density,
        0
    )

    gt_count = gt_density.sum()
    pred_count = pred_density.sum()

    # Original
    axes[i,0].imshow(image)
    axes[i,0].set_title(
        f"Original\nSample {sample_idx}"
    )
    axes[i,0].axis("off")

    # GT Density
    axes[i,1].imshow(
        gt_density,
        cmap='jet'
    )
    axes[i,1].set_title(
        f"GT Density\nCount: {int(gt_count)}"
    )
    axes[i,1].axis("off")

    # Pred Density
    axes[i,2].imshow(
        pred_density,
        cmap='jet'
    )
    axes[i,2].set_title(
        f"Pred Density\nCount: {int(pred_count)}"
    )
    axes[i,2].axis("off")


plt.subplots_adjust(
    hspace=0.6,
    wspace=0.3
)

plt.savefig(
    "sanet_15epoch_results.png",
    dpi=300,
    bbox_inches="tight"
)

print("Visualization saved successfully")

plt.show()