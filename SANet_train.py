import numpy as np
import tensorflow as tf

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# Load dataset
# -----------------------------------
images = np.load("images.npy")
labels = np.load("labels.npy")

print("Images:", images.shape)
print("Labels:", labels.shape)

images = images.astype(np.float32) / 255.0


# -----------------------------------
# Generate density maps
# -----------------------------------
def generate_density_map(instance_mask):
    h, w = instance_mask.shape
    density = np.zeros((h, w), dtype=np.float32)

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

        density[y, x] = 1

    density = gaussian_filter(
        density,
        sigma=1.2
    )

    if density.sum() > 0:
        density = density * (
            len(nuclei_ids) / density.sum()
        )

    return density


density_maps = []

for i in range(len(labels)):
    instance_mask = labels[i, :, :, 0]

    density = generate_density_map(
        instance_mask
    )

    density_maps.append(density)

    if i % 500 == 0:
        print(f"Processed {i}")


density_maps = np.array(density_maps)
density_maps = np.expand_dims(
    density_maps,
    axis=-1
)

print("Density Maps:", density_maps.shape)


# -----------------------------------
# Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images,
    density_maps,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Build SANet
# -----------------------------------
def build_sanet(input_shape=(256,256,3)):
    inputs = Input(input_shape)

    b1 = Conv2D(
        32,
        3,
        activation='relu',
        padding='same'
    )(inputs)

    b2 = Conv2D(
        32,
        5,
        activation='relu',
        padding='same'
    )(inputs)

    b3 = Conv2D(
        32,
        7,
        activation='relu',
        padding='same'
    )(inputs)

    merged = concatenate([b1,b2,b3])

    x = Conv2D(
        64,
        3,
        activation='relu',
        padding='same'
    )(merged)

    x = MaxPooling2D()(x)

    x = Conv2D(
        128,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = Conv2D(
        128,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = UpSampling2D()(x)

    outputs = Conv2D(
        1,
        1,
        activation='linear',
        padding='same'
    )(x)

    model = Model(inputs, outputs)

    return model


model = build_sanet()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

model.summary()


# -----------------------------------
# Train
# -----------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=8,
    callbacks=[early_stop]
)


# -----------------------------------
# Save model
# -----------------------------------
model.save("sanet_model_15epochs.h5")

print("SANet model saved successfully")