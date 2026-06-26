import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

images = np.load("images.npy")
labels = np.load("labels.npy")



instance_masks = labels[:, :, :, 0]
binary_masks = (instance_masks > 0).astype(np.float32)

print("Mask shape:", binary_masks.shape)


images = images / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    images,
    binary_masks,
    test_size=0.2,
    random_state=42
)

# Add channel dimension
y_train = np.expand_dims(y_train, axis=-1).astype(np.float32)
y_test = np.expand_dims(y_test, axis=-1).astype(np.float32)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) +
        tf.keras.backend.sum(y_pred_f) +
        smooth
    )


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x


def build_unetplusplus(input_shape=(256,256,3)):
    inputs = Input(input_shape)

    # Encoder
    x00 = conv_block(inputs, 32)
    p0 = MaxPooling2D()(x00)

    x10 = conv_block(p0, 64)
    p1 = MaxPooling2D()(x10)

    x20 = conv_block(p1, 128)
    p2 = MaxPooling2D()(x20)

    x30 = conv_block(p2, 256)
    p3 = MaxPooling2D()(x30)

    x40 = conv_block(p3, 512)

    # Decoder (nested skip connections)

    x31 = conv_block(
        concatenate([UpSampling2D()(x40), x30]), 256
    )

    x21 = conv_block(
        concatenate([UpSampling2D()(x30), x20]), 128
    )

    x22 = conv_block(
        concatenate([UpSampling2D()(x31), x20, x21]), 128
    )

    x11 = conv_block(
        concatenate([UpSampling2D()(x20), x10]), 64
    )

    x12 = conv_block(
        concatenate([UpSampling2D()(x21), x10, x11]), 64
    )

    x13 = conv_block(
        concatenate([UpSampling2D()(x22), x10, x11, x12]), 64
    )

    x01 = conv_block(
        concatenate([UpSampling2D()(x10), x00]), 32
    )

    x02 = conv_block(
        concatenate([UpSampling2D()(x11), x00, x01]), 32
    )

    x03 = conv_block(
        concatenate([UpSampling2D()(x12), x00, x01, x02]), 32
    )

    x04 = conv_block(
        concatenate([UpSampling2D()(x13), x00, x01, x02, x03]), 32
    )

    outputs = Conv2D(1, 1, activation='sigmoid')(x04)

    model = Model(inputs, outputs)

    return model


model = build_unetplusplus()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef]
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    batch_size=8,
    epochs=1
)


preds = model.predict(X_test)
pred_masks = (preds > 0.5).astype(np.uint8)

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    if union == 0:
        return 1.0

    return intersection / union


iou_scores = []

for i in range(len(y_test)):
    score = calculate_iou(
        y_test[i].squeeze(),
        pred_masks[i].squeeze()
    )
    iou_scores.append(score)

print("Average IoU:", np.mean(iou_scores))



#OUTPUTS
idx = 5

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(X_test[idx])
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(y_test[idx].squeeze(), cmap='gray')
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred_masks[idx].squeeze(), cmap='gray')
plt.title("U-Net++ Prediction")

plt.savefig("unetplusplus_output.png")
print("U-Net++ output saved successfully")