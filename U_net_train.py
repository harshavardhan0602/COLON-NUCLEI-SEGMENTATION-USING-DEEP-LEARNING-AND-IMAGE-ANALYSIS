import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# Load dataset
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
# Train Test Split
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
# U-Net Model
# -----------------------------------
def build_unet(input_shape=(256,256,3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32,3,activation='relu',padding='same')(inputs)
    c1 = Conv2D(32,3,activation='relu',padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64,3,activation='relu',padding='same')(p1)
    c2 = Conv2D(64,3,activation='relu',padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128,3,activation='relu',padding='same')(p2)
    c3 = Conv2D(128,3,activation='relu',padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(256,3,activation='relu',padding='same')(p3)
    c4 = Conv2D(256,3,activation='relu',padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    # Bottleneck
    c5 = Conv2D(512,3,activation='relu',padding='same')(p4)
    c5 = Conv2D(512,3,activation='relu',padding='same')(c5)

    # Decoder
    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6,c4])
    c6 = Conv2D(256,3,activation='relu',padding='same')(u6)
    c6 = Conv2D(256,3,activation='relu',padding='same')(c6)

    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7,c3])
    c7 = Conv2D(128,3,activation='relu',padding='same')(u7)
    c7 = Conv2D(128,3,activation='relu',padding='same')(c7)

    u8 = UpSampling2D()(c7)
    u8 = concatenate([u8,c2])
    c8 = Conv2D(64,3,activation='relu',padding='same')(u8)
    c8 = Conv2D(64,3,activation='relu',padding='same')(c8)

    u9 = UpSampling2D()(c8)
    u9 = concatenate([u9,c1])
    c9 = Conv2D(32,3,activation='relu',padding='same')(u9)
    c9 = Conv2D(32,3,activation='relu',padding='same')(c9)

    outputs = Conv2D(1,1,activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model


model = build_unet()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef]
)

model.summary()


# -----------------------------------
# Early Stopping
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
# Save model
# -----------------------------------
model.save("unet_model_15epochs.h5")

print("U-Net model saved successfully")