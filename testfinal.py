import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# GPU Check
# =====================================================
print("GPU Available:",
      tf.config.list_physical_devices('GPU'))

# =====================================================
# Load Dataset
# =====================================================
print("Loading dataset...")

images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32) / 255.0

masks = (labels[:, :, :, 0] > 0).astype(np.float32)
masks = np.expand_dims(masks, axis=-1)

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)


# =====================================================
# Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)


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
        (2.0 * intersection + smooth) /
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
        K.equal(y_true, 1),
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
        0.3 * K.mean(bce) +
        0.4 * d_loss +
        0.3 * f_loss
    )


# =====================================================
# Conv Block
# =====================================================
def conv_block(x, filters):
    x = Conv2D(
        filters,
        3,
        padding='same',
        activation='relu'
    )(x)

    x = Conv2D(
        filters,
        3,
        padding='same',
        activation='relu'
    )(x)

    return x


# =====================================================
# Multi-scale Attention Gate
# =====================================================
def multiscale_attention_gate(
    x,
    g,
    inter_channels
):

    # Multi-scale feature extraction
    branch1 = Conv2D(
        inter_channels,
        1,
        padding='same'
    )(x)

    branch2 = Conv2D(
        inter_channels,
        3,
        padding='same'
    )(x)

    branch3 = Conv2D(
        inter_channels,
        5,
        padding='same'
    )(x)

    multi_features = Add()([
        branch1,
        branch2,
        branch3
    ])

    gating = Conv2D(
        inter_channels,
        1,
        padding='same'
    )(g)

    merged = Add()([
        multi_features,
        gating
    ])

    merged = Activation('relu')(merged)

    psi = Conv2D(
        1,
        1,
        padding='same'
    )(merged)

    psi = Activation('sigmoid')(psi)

    output = Multiply()([
        x,
        psi
    ])

    return output


# =====================================================
# JPFM Bottleneck
# =====================================================
def jpfm_bottleneck(x):

    d1 = Conv2D(
        256,
        3,
        dilation_rate=1,
        padding='same',
        activation='relu'
    )(x)

    d2 = Conv2D(
        256,
        3,
        dilation_rate=2,
        padding='same',
        activation='relu'
    )(x)

    d4 = Conv2D(
        256,
        3,
        dilation_rate=4,
        padding='same',
        activation='relu'
    )(x)

    d8 = Conv2D(
        256,
        3,
        dilation_rate=8,
        padding='same',
        activation='relu'
    )(x)

    merged = concatenate([
        d1,
        d2,
        d4,
        d8
    ])

    merged = Conv2D(
        256,
        1,
        padding='same',
        activation='relu'
    )(merged)

    return merged


# =====================================================
# Build Model
# =====================================================
def build_multiscale_jpfm_unetpp(
    input_shape=(256,256,3)
):

    inputs = Input(input_shape)

    # ---------------- Encoder ----------------
    x00 = conv_block(inputs, 32)
    p0 = MaxPooling2D()(x00)

    x10 = conv_block(p0, 64)
    p1 = MaxPooling2D()(x10)

    x20 = conv_block(p1, 128)
    p2 = MaxPooling2D()(x20)

    # ---------------- JPFM Bottleneck ----------------
    x30 = jpfm_bottleneck(p2)

    # ---------------- Decoder ----------------

    # Skip 1
    up_x10 = UpSampling2D()(x10)

    att_x00 = multiscale_attention_gate(
        x00,
        up_x10,
        32
    )

    x01 = conv_block(
        concatenate([
            att_x00,
            up_x10
        ]),
        32
    )


    # Skip 2
    up_x20 = UpSampling2D()(x20)

    att_x10 = multiscale_attention_gate(
        x10,
        up_x20,
        64
    )

    x11 = conv_block(
        concatenate([
            att_x10,
            up_x20
        ]),
        64
    )


    # Skip 3
    up_x30 = UpSampling2D()(x30)

    att_x20 = multiscale_attention_gate(
        x20,
        up_x30,
        128
    )

    x21 = conv_block(
        concatenate([
            att_x20,
            up_x30
        ]),
        128
    )


    # Final Nested Skip
    up_x11 = UpSampling2D()(x11)

    att_x00_final = multiscale_attention_gate(
        x00,
        up_x11,
        32
    )

    x02 = conv_block(
        concatenate([
            att_x00_final,
            x01,
            up_x11
        ]),
        32
    )


    # Output
    outputs = Conv2D(
        1,
        1,
        activation='sigmoid'
    )(x02)

    model = Model(inputs, outputs)

    return model


# =====================================================
# Build Model
# =====================================================
model = build_multiscale_jpfm_unetpp()

model.compile(
    optimizer='adam',
    loss=hybrid_loss,
    metrics=['accuracy', dice_coef]
)

model.summary()
print("Total Parameters:",
      model.count_params())


# =====================================================
# Callbacks
# =====================================================
early_stop = EarlyStopping(
    monitor='val_dice_coef',
    mode='max',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_multiscale_jpfm_unetpp.keras",
    monitor='val_dice_coef',
    mode='max',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_dice_coef',
    mode='max',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# =====================================================
# Train
# =====================================================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=8,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# =====================================================
# Final Evaluation
# =====================================================
results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")
print("Loss:", results[0])
print("Accuracy:", results[1])
print("Dice Score:", results[2])
print("===================================")