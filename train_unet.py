# ===========================================
# 🧠 U-NET TRAINING (Google Colab Version)
# ===========================================

# ✅ STEP 1: INSTALL DEPENDENCIES
# !pip install tensorflow opencv-python-headless matplotlib --quiet

# ✅ STEP 2: IMPORT LIBRARIES
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ✅ STEP 3: CONNECT TO GOOGLE DRIVE (optional if dataset in Drive)
from google.colab import drive
drive.mount('/content/drive')

# ✅ STEP 4: DEFINE PATHS (Update this to your dataset folder path in Drive)
# Example: "/content/drive/MyDrive/Sahana_3D_project/Dataset"
DATA_ROOT = "/content/drive/MyDrive/Dataset"

IMG_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
MODEL_OUT = os.path.join(DATA_ROOT, "model_output")
os.makedirs(MODEL_OUT, exist_ok=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 10

# ✅ STEP 5: LOAD IMAGE-MASK PAIRS
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
mask_files = [f"mask_{f}" for f in img_files]  # expects naming: mask_<imgname>.jpg

pairs = []
for img_f, mask_f in zip(img_files, mask_files):
    mask_path = os.path.join(MASK_DIR, mask_f)
    if os.path.exists(mask_path):
        pairs.append((os.path.join(IMG_DIR, img_f), mask_path))

if len(pairs) == 0:
    raise SystemExit("❌ No valid image-mask pairs found. Check naming (mask_<imgname>.*).")

print(f"📸 Found {len(pairs)} valid pairs.")

# ✅ STEP 6: TRAIN/VAL SPLIT
np.random.shuffle(pairs)
split_idx = int(0.8 * len(pairs))
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

# ✅ STEP 7: DATA GENERATOR
def data_generator(pairs, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    while True:
        np.random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            imgs, masks = [], []
            for img_path, mask_path in batch:
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, img_size)
                mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

                img = img.astype(np.float32) / 255.0
                mask = (mask > 127).astype(np.float32)

                imgs.append(img)
                masks.append(np.expand_dims(mask, axis=-1))
            yield np.array(imgs), np.array(masks)

train_gen = data_generator(train_pairs)
val_gen = data_generator(val_pairs)

# ✅ STEP 8: BUILD U-NET MODEL
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(c4)
    c5 = conv_block(p4, 1024)

    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.UpSampling2D(2)(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.UpSampling2D(2)(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs, outputs)
    return model

model = build_unet()
model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ STEP 9: CALLBACKS
callbacks = [
    ModelCheckpoint(os.path.join(MODEL_OUT, "unet_best.keras"), save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

# ✅ STEP 10: TRAINING
steps_per_epoch = max(1, len(train_pairs) // BATCH_SIZE)
val_steps = max(1, len(val_pairs) // BATCH_SIZE)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks
)

# ✅ STEP 11: SAVE MODEL
final_path = os.path.join(MODEL_OUT, "unet_final.keras")
model.save(final_path)
print(f"✅ Training finished. Model saved at: {final_path}")

# ✅ STEP 12: FINAL ACCURACY
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"🏁 Final Training Accuracy: {final_acc:.4f} | Validation Accuracy: {final_val_acc:.4f}")
