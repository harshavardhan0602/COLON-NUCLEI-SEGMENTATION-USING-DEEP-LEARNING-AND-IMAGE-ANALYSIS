import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# =========================================================
# Load Dataset
# =========================================================
images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype(np.float32) / 255.0

# Instance mask -> binary mask
masks = (labels[:, :, :, 0] > 0).astype(np.float32)
masks = np.expand_dims(masks, axis=-1)

print("Images shape:", images.shape)
print("Masks shape :", masks.shape)


# =========================================================
# Train Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)


# =========================================================
# Dice Metric
# =========================================================
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


# =========================================================
# Dice Loss
# =========================================================
def dice_loss(y_true, y_pred):
    smooth = 1e-7

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - (
        (2 * intersection + smooth) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )


# =========================================================
# Focal Loss
# =========================================================
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
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


# =========================================================
# Hybrid Loss
# =========================================================
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


# =========================================================
# Convolution Block
# =========================================================
def conv_block(x, filters):

    x = Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x


# =========================================================
# Attention Block
# =========================================================
def attention_block(g, x, filters):

    g1 = Conv2D(filters, 1, padding="same")(g)
    x1 = Conv2D(filters, 1, padding="same")(x)

    psi = Add()([g1, x1])
    psi = Activation("relu")(psi)

    psi = Conv2D(1, 1, padding="same")(psi)
    psi = Activation("sigmoid")(psi)

    return Multiply()([x, psi])


# =========================================================
# Build Model
# =========================================================
def build_model(input_shape=(256,256,3)):

    inputs = Input(input_shape)

    # ---------------- Encoder ----------------
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, 512)

    # ---------------- Decoder ----------------
    u6 = UpSampling2D()(c5)
    a6 = attention_block(u6, c4, 256)
    u6 = concatenate([u6, a6])
    c6 = conv_block(u6, 256)

    u7 = UpSampling2D()(c6)
    a7 = attention_block(u7, c3, 128)
    u7 = concatenate([u7, a7])
    c7 = conv_block(u7, 128)

    u8 = UpSampling2D()(c7)
    a8 = attention_block(u8, c2, 64)
    u8 = concatenate([u8, a8])
    c8 = conv_block(u8, 64)

    u9 = UpSampling2D()(c8)
    a9 = attention_block(u9, c1, 32)
    u9 = concatenate([u9, a9])
    c9 = conv_block(u9, 32)


    # ====================================================
    # Deep Supervision Outputs
    # ====================================================

    out1 = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c6)
    out1 = UpSampling2D(size=(8,8))(out1)


    out2 = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c7)
    out2 = UpSampling2D(size=(4,4))(out2)


    out3 = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c8)
    out3 = UpSampling2D(size=(2,2))(out3)


    final_out = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c9)


    # Average all outputs
    final_avg = Average()([
        out1,
        out2,
        out3,
        final_out
    ])


    model = Model(inputs, final_avg)

    return model


# =========================================================
# Build + Compile
# =========================================================
model = build_model()

model.compile(
    optimizer="adam",
    loss=hybrid_loss,
    metrics=["accuracy", dice_coef]
)

model.summary()


# =========================================================
# Callbacks
# =========================================================
checkpoint = ModelCheckpoint(
    "best_attention_deepsupervision.keras",
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_dice_coef",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


# =========================================================
# Train
# =========================================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)


print("\nTraining completed successfully!")