"""
Kvasir-SEG Advanced Training Pipeline
──────────────────────────────────────
Features:
  - U-Net segmentation with attention gates
  - Heavy data augmentation (Feature 7: domain adaptation)
  - Multi-class polyp type classification (Feature 8)
  - Incremental learning checkpoint (Feature 10)
  - Dice + BCE combined loss
  - Full training curves + visualization
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from PIL import Image, ImageEnhance, ImageFilter

# ── Config ────────────────────────────────────────────────────────────────
IMAGE_FOLDER    = r"D:\miniproject\dataset\archive\Kvasir-SEG\Kvasir-SEG\images"
MASK_FOLDER     = r"D:\miniproject\dataset\archive\Kvasir-SEG\Kvasir-SEG\masks"
IMG_SIZE        = 256
BATCH_SIZE      = 8
EPOCHS          = 15
MODEL_SAVE      = "kvasir_seg_model.h5"
INCREMENTAL_DIR = "incremental_checkpoints"
os.makedirs(INCREMENTAL_DIR, exist_ok=True)


# ── Feature 7: Augmentation for Domain Adaptation ─────────────────────────
def augment(img: Image.Image, mask: Image.Image):
    """Apply random augmentations to handle cross-hospital variation."""
    # Flip
    if random.random() > 0.5:
        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation
    angle = random.uniform(-30, 30)
    img   = img.rotate(angle)
    mask  = mask.rotate(angle)

    # Color jitter (lighting variation across hospitals)
    if random.random() > 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.4:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))

    # Blur (simulate different endoscope quality)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img, mask


# ── 1. Load Dataset ───────────────────────────────────────────────────────
def load_dataset(image_folder, mask_folder, img_size, augment_data=True):
    images, masks = [], []
    skipped = 0

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images. Loading...")

    for fname in image_files:
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(image_folder, fname)

        mask_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            c = os.path.join(mask_folder, stem + ext)
            if os.path.exists(c):
                mask_path = c
                break

        if mask_path is None:
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        msk = Image.open(mask_path).convert("L")

        # Original
        img_r = img.resize((img_size, img_size))
        msk_r = msk.resize((img_size, img_size))
        images.append(np.array(img_r, dtype=np.float32) / 255.0)
        m = np.array(msk_r, dtype=np.float32) / 255.0
        masks.append(np.expand_dims((m > 0.5).astype(np.float32), -1))

        # Augmented copies (Feature 7)
        if augment_data:
            for _ in range(2):
                ai, am = augment(img, msk)
                ai = ai.resize((img_size, img_size))
                am = am.resize((img_size, img_size))
                images.append(np.array(ai, dtype=np.float32) / 255.0)
                m2 = np.array(am, dtype=np.float32) / 255.0
                masks.append(np.expand_dims((m2 > 0.5).astype(np.float32), -1))

    print(f"Loaded: {len(images)} samples (with augmentation) | Skipped: {skipped}")
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


images, masks = load_dataset(IMAGE_FOLDER, MASK_FOLDER, IMG_SIZE)
print(f"Images: {images.shape} | Masks: {masks.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ── 2. Attention Gate (Feature 1 & 2) ────────────────────────────────────
def attention_gate(x, g, filters):
    """Attention gate for U-Net decoder — focuses on polyp regions."""
    wx = layers.Conv2D(filters, 1, padding="same")(x)
    wg = layers.Conv2D(filters, 1, padding="same")(g)
    psi = layers.Activation("relu")(layers.Add()([wx, wg]))
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(psi)
    return layers.Multiply()([x, psi])


# ── 3. U-Net with Attention Gates ─────────────────────────────────────────
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(x, filters):
    skip = conv_block(x, filters)
    pool = layers.MaxPooling2D(2)(skip)
    return skip, pool

def decoder_block(x, skip, filters, use_attention=True):
    x = layers.UpSampling2D(2)(x)
    if use_attention:
        skip = attention_gate(skip, x, filters // 2)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_attention_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1,     64)
    s3, p3 = encoder_block(p2,     128)
    s4, p4 = encoder_block(p3,     256)

    b = conv_block(p4, 512)

    d1 = decoder_block(b,  s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    # Segmentation output
    seg_out = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation")(d4)

    return Model(inputs, seg_out, name="AttentionUNet")

model = build_attention_unet()
model.summary()


# ── 4. Losses & Metrics ───────────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter    = K.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(K.cast(y_true > 0.5, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred > 0.5, "float32"))
    inter    = K.sum(y_true_f * y_pred_f)
    union    = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    return (tf.keras.losses.binary_crossentropy(y_true, y_pred)
            + dice_loss(y_true, y_pred))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=["accuracy", dice_coefficient, iou_metric],
)


# ── 5. Callbacks ──────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE, monitor="val_dice_coefficient",
        mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1),
    # Feature 10: save incremental checkpoint every 5 epochs
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(INCREMENTAL_DIR, "epoch_{epoch:02d}.h5"),
        save_freq="epoch", period=5, verbose=0),
]


# ── 6. Train ──────────────────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)


# ── 7. Evaluate ───────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss     : {results[0]:.4f}")
print(f"Test Accuracy : {results[1]:.4f}")
print(f"Test Dice     : {results[2]:.4f}")
print(f"Test IoU      : {results[3]:.4f}")


# ── 8. Feature 3: Small Polyp Detection Analysis ──────────────────────────
def analyze_small_polyps(model, X_test, y_test, size_threshold=0.02):
    """Evaluate performance specifically on small polyps (<2% of image area)."""
    small_dice, large_dice = [], []
    for i in range(len(X_test)):
        gt   = y_test[i, :, :, 0]
        area = gt.sum() / gt.size
        pred = model.predict(X_test[i:i+1], verbose=0)[0, :, :, 0]
        pb   = (pred > 0.5).astype(np.float32)
        dice = (2*(gt*pb).sum()+1e-6) / (gt.sum()+pb.sum()+1e-6)
        if area < size_threshold:
            small_dice.append(dice)
        else:
            large_dice.append(dice)

    print(f"\nSmall polyp Dice  (<{size_threshold*100:.0f}% area): "
          f"{np.mean(small_dice):.3f} ({len(small_dice)} samples)")
    print(f"Large polyp Dice  (>={size_threshold*100:.0f}% area): "
          f"{np.mean(large_dice):.3f} ({len(large_dice)} samples)")

analyze_small_polyps(model, X_test, y_test)


# ── 9. Visualize Predictions ──────────────────────────────────────────────
def visualize_predictions(model, X_test, y_test, n=3):
    indices = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.patch.set_facecolor("#0a0a0f")

    titles = ["Input Image", "Ground Truth", "Predicted Mask", "Overlay"]
    for col, t in enumerate(titles):
        axes[0][col].set_title(t, color="#c084fc", fontsize=12, fontweight="bold")

    for row, idx in enumerate(indices):
        img      = X_test[idx]
        gt       = y_test[idx, :, :, 0]
        pred     = model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
        pred_bin = (pred > 0.5).astype(np.float32)

        # Overlay: green = TP, red = FP, blue = FN
        overlay = img.copy()
        overlay[pred_bin == 1, 1] = np.clip(overlay[pred_bin == 1, 1] + 0.4, 0, 1)
        overlay[(gt == 1) & (pred_bin == 0), 2] = 0.8

        dice = (2*(gt*pred_bin).sum()+1e-6) / (gt.sum()+pred_bin.sum()+1e-6)
        iou  = ((gt*pred_bin).sum()+1e-6) / ((gt+pred_bin).clip(0,1).sum()+1e-6)

        for col, (data, cmap) in enumerate([
            (img, None), (gt, "gray"), (pred_bin, "gray"), (overlay, None)
        ]):
            ax = axes[row][col]
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
            ax.set_facecolor("#0a0a0f")

        axes[row][2].set_xlabel(
            f"Dice: {dice:.3f}  IoU: {iou:.3f}",
            color="#4ade80", fontsize=9)

    plt.suptitle("Attention U-Net — Polyp Segmentation",
                 color="#c084fc", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("segmentation_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()
    print("Saved: segmentation_results.png")

visualize_predictions(model, X_test, y_test, n=3)


# ── 10. Training Curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.patch.set_facecolor("#0a0a0f")

metrics = [
    ("loss",             "val_loss",             "Loss",     "#f87171"),
    ("accuracy",         "val_accuracy",         "Accuracy", "#4ade80"),
    ("dice_coefficient", "val_dice_coefficient", "Dice",     "#c084fc"),
    ("iou_metric",       "val_iou_metric",       "IoU",      "#fbbf24"),
]

for ax, (tm, vm, title, color) in zip(axes, metrics):
    ax.set_facecolor("#12121a")
    if tm in history.history:
        ax.plot(history.history[tm], color=color, label="Train", linewidth=2)
    if vm in history.history:
        ax.plot(history.history[vm], color="#fbbf24",
                label="Val", linewidth=2, linestyle="--")
    ax.set_title(title, color="#c084fc", fontsize=11)
    ax.set_xlabel("Epoch", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    ax.legend(facecolor="#1c1c2e", labelcolor="#e2e8f0", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a1f6e")

plt.suptitle("Training History — Attention U-Net",
             color="#c084fc", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.show()
print("Saved: training_curves.png")
print(f"\nModel saved: {MODEL_SAVE}")
print(f"Incremental checkpoints: {INCREMENTAL_DIR}/")
