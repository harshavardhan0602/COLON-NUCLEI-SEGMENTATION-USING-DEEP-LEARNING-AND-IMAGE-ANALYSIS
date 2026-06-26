import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# Load Dataset
# -----------------------------------
images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32) / 255.0

# Binary masks
masks = (labels[:, :, :, 0] > 0).astype(np.float32)
masks = np.expand_dims(masks, axis=-1)

print("Images Shape:", images.shape)
print("Masks Shape:", masks.shape)


# -----------------------------------
# Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)

print("Train Samples:", len(X_train))
print("Test Samples:", len(X_test))


# -----------------------------------
# Dice Metric
# -----------------------------------
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2 * intersection + 1e-7) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) + 1e-7
    )


# -----------------------------------
# Convolution Block
# -----------------------------------
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x


# -----------------------------------
# Build U-Net++
# -----------------------------------
def build_unetpp(input_shape=(256,256,3)):
    inputs = Input(input_shape)

    x00 = conv_block(inputs, 32)
    p0 = MaxPooling2D()(x00)

    x10 = conv_block(p0, 64)
    p1 = MaxPooling2D()(x10)

    x20 = conv_block(p1, 128)
    p2 = MaxPooling2D()(x20)

    x30 = conv_block(p2, 256)

    # Nested skip connections
    x01 = conv_block(
        concatenate([
            x00,
            UpSampling2D()(x10)
        ]),
        32
    )

    x11 = conv_block(
        concatenate([
            x10,
            UpSampling2D()(x20)
        ]),
        64
    )

    x21 = conv_block(
        concatenate([
            x20,
            UpSampling2D()(x30)
        ]),
        128
    )

    x02 = conv_block(
        concatenate([
            x00,
            x01,
            UpSampling2D()(x11)
        ]),
        32
    )

    outputs = Conv2D(
        1,
        1,
        activation='sigmoid'
    )(x02)

    model = Model(inputs, outputs)

    return model


model = build_unetpp()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef]
)

model.summary()


# -----------------------------------
# Early stopping
# -----------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# -----------------------------------
# Train
# -----------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=8,
    callbacks=[early_stop]
)


# -----------------------------------
# Save Model
# -----------------------------------
model.save("unetpp_model_15epochs.h5")

print("U-Net++ model saved successfully")