"""
FINE-TUNE REALISTIC METER MODEL ON REAL IMAGES
==============================================
Transfer learning: Fine-tune the pre-trained model on 35 labeled real images.
- Loads:  realistic_meter_model/best_model.keras   ← NEVER MODIFIED
- Saves:  finetuned_meter_model/best_finetuned_model.keras  ← NEW FILE
- Freezes early CNN layers, trains only RNN + output layers
- Heavy augmentation (5x) to prevent overfitting on 28 train images
- Target: 40-60% accuracy on real images
"""

import os
# ── Fix working directory so relative paths always resolve correctly ──────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random

# ==================== CONFIGURATION ====================
PRETRAINED_MODEL  = "realistic_meter_model/best_model.keras"   # ← read-only source
CHARS_PATH        = "realistic_meter_model/chars.json"
REAL_LABELS_PATH  = "real_image_labels.json"
OUTPUT_DIR        = "finetuned_meter_model"                     # ← separate output folder

IMAGE_WIDTH  = 800
IMAGE_HEIGHT = 200
BATCH_SIZE   = 4    # small batch for small dataset
MAX_EPOCHS   = 100
EARLY_STOP_PATIENCE = 15
TRAIN_SPLIT  = 0.8  # 28 train / 7 val

print("=" * 60)
print("🔄 FINE-TUNING ON REAL METER IMAGES (TRANSFER LEARNING)")
print("=" * 60)
print(f"📁 Source model  : {PRETRAINED_MODEL}  ← will NOT be overwritten")
print(f"💾 Output folder : {OUTPUT_DIR}/        ← all new files go here")
print(f"📄 Real labels   : {REAL_LABELS_PATH}")
print(f"🎯 Target        : 40-60 % real-world accuracy")
print("=" * 60)

# ==================== LOAD CHARACTER MAPPING ====================
# Embedded fallback so the script works even if chars.json is missing
_FALLBACK_CHARS = "0123456789.-"
_FALLBACK_CHAR_TO_IDX = {c: i for i, c in enumerate(_FALLBACK_CHARS)}

if os.path.exists(CHARS_PATH):
    with open(CHARS_PATH, 'r') as f:
        char_data = json.load(f)
    CHARS       = char_data['chars']
    CHAR_TO_IDX = char_data['char_to_idx']
else:
    print(f"⚠️  {CHARS_PATH} not found — using built-in fallback charset")
    CHARS       = _FALLBACK_CHARS
    CHAR_TO_IDX = _FALLBACK_CHAR_TO_IDX

IDX_TO_CHAR = {int(idx): char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS) + 1   # +1 for CTC blank

print(f"📊 Vocab size: {len(CHARS)},  classes (incl. CTC blank): {NUM_CLASSES}")

# ==================== LOAD REAL DATA ====================
print(f"\n🔄 Loading real images from {REAL_LABELS_PATH}...")

with open(REAL_LABELS_PATH, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

real_data = []
for item in gt_data:
    img_path = item.get('image_path', '')
    filename = item.get('filename', '')
    label    = item.get('full_reading', item.get('label', ''))

    if os.path.exists(img_path):
        real_data.append((img_path, label))
    else:
        # Search recursively from current directory
        found = False
        for root, dirs, files in os.walk("."):
            if filename in files:
                candidate = os.path.join(root, filename)
                real_data.append((candidate, label))
                found = True
                break
        if not found:
            print(f"   ⚠️  Not found: {filename}")

print(f"✅ Found {len(real_data)} real images")

# Shuffle and split
np.random.seed(42)
np.random.shuffle(real_data)
split_idx  = int(len(real_data) * TRAIN_SPLIT)
train_data = real_data[:split_idx]
val_data   = real_data[split_idx:]

print(f"   Train: {len(train_data)} images (×5 augmented = {len(train_data)*5} samples per epoch)")
print(f"   Val  : {len(val_data)} images")

# ==================== AUGMENTATION ====================
def augment_image(img):
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-10, 10), fillcolor=0)
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() > 0.6:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))
    if random.random() > 0.5:
        arr   = np.array(img)
        noise = np.random.normal(0, random.randint(5, 15), arr.shape)
        img   = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img

# ==================== PREPROCESSING ====================
def encode_label(label):
    return [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]

def load_and_preprocess_image(img_path, augment=False):
    img = Image.open(img_path).convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    if augment:
        img = augment_image(img)
    return np.array(img, dtype=np.float32) / 255.0

def create_dataset(data, batch_size, shuffle=True, augment=False):
    def generator():
        epoch_data = list(data)
        if augment:
            epoch_data = epoch_data * 5   # 5× repetition, different augment each time
        if shuffle:
            np.random.shuffle(epoch_data)
        for img_path, label in epoch_data:
            yield img_path, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )

    MAX_LABEL = 15

    def process_sample(img_path, label):
        img          = load_and_preprocess_image(img_path.numpy().decode(), augment=augment)
        label_enc    = encode_label(label.numpy().decode())
        label_len    = len(label_enc)
        label_padded = label_enc + [NUM_CLASSES - 1] * (MAX_LABEL - label_len)
        return (img,
                np.array(label_padded, dtype=np.int32),
                np.array([label_len],  dtype=np.int32))

    dataset = dataset.map(
        lambda p, l: tf.py_function(process_sample, [p, l],
                                    [tf.float32, tf.int32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(lambda img, lbl, llen: (
        tf.ensure_shape(img,  [IMAGE_HEIGHT, IMAGE_WIDTH]),
        tf.ensure_shape(lbl,  [MAX_LABEL]),
        tf.ensure_shape(llen, [1])
    ))
    dataset = dataset.map(lambda img, lbl, llen: (
        tf.expand_dims(img, axis=-1), lbl, llen
    ))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ==================== LOAD & CONFIGURE PRE-TRAINED MODEL ====================
print(f"\n🔄 Loading pre-trained model from {PRETRAINED_MODEL} ...")
base_model = keras.models.load_model(PRETRAINED_MODEL)
print("✅ Loaded (original files untouched)")

# Freeze first 3 Conv2D layers
freeze_names = {'conv2d', 'conv2d_1', 'conv2d_2'}
frozen = trainable = 0
for layer in base_model.layers:
    if layer.name in freeze_names:
        layer.trainable = False
        frozen += 1
        print(f"   🔒 Frozen : {layer.name}")
    else:
        layer.trainable = True
        trainable += 1
print(f"\n   Frozen layers   : {frozen}")
print(f"   Trainable layers: {trainable}")

# ==================== CUSTOM CTC TRAINING LOOP ====================
class CTCFineTuner:
    def __init__(self, model, lr=1e-4):
        self.model     = model
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.train_tracker = keras.metrics.Mean(name='train_loss')
        self.val_tracker   = keras.metrics.Mean(name='val_loss')

    @tf.function
    def train_step(self, images, labels, label_lengths):
        with tf.GradientTape() as tape:
            y_pred  = self.model(images, training=True)
            seq_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
            loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels        = labels,
                logits        = y_pred,
                label_length  = tf.cast(tf.squeeze(label_lengths, -1), tf.int32),
                logit_length  = tf.cast(seq_len, tf.int32),
                blank_index   = NUM_CLASSES - 1,
                logits_time_major = False
            ))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_tracker.update_state(loss)
        return loss

    @tf.function
    def val_step(self, images, labels, label_lengths):
        y_pred  = self.model(images, training=False)
        seq_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels        = labels,
            logits        = y_pred,
            label_length  = tf.cast(tf.squeeze(label_lengths, -1), tf.int32),
            logit_length  = tf.cast(seq_len, tf.int32),
            blank_index   = NUM_CLASSES - 1,
            logits_time_major = False
        ))
        self.val_tracker.update_state(loss)
        return loss

    def fit(self, train_ds, val_ds, epochs, patience):
        best_val   = float('inf')
        wait       = 0
        history    = {'train_loss': [], 'val_loss': []}
        best_path  = os.path.join(OUTPUT_DIR, 'best_finetuned_model.keras')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("─" * 60)

            self.train_tracker.reset_state()
            for step, (imgs, lbls, llens) in enumerate(train_ds, 1):
                loss = self.train_step(imgs, lbls, llens)
                if step % 10 == 0:
                    print(f"  Step {step:3d}: loss = {loss.numpy():.4f}")

            train_loss = self.train_tracker.result().numpy()

            self.val_tracker.reset_state()
            for imgs, lbls, llens in val_ds:
                self.val_step(imgs, lbls, llens)
            val_loss = self.val_tracker.result().numpy()

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            print(f"\n  📊 train_loss: {train_loss:.4f}  |  val_loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                self.model.save(best_path)
                print(f"  ✅ Best model saved → {best_path}")
            else:
                wait += 1
                print(f"  ⏳ No improvement ({wait}/{patience})")
                if wait >= patience:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                    break

        return history

# ==================== MAIN ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n🔄 Building data pipelines...")
    train_ds = create_dataset(train_data, BATCH_SIZE, shuffle=True,  augment=True)
    val_ds   = create_dataset(val_data,   BATCH_SIZE, shuffle=False, augment=False)

    print(f"\n🏋️  Starting fine-tuning  (LR=0.0001, batch={BATCH_SIZE}, patience={EARLY_STOP_PATIENCE})")
    print("=" * 60)

    tuner   = CTCFineTuner(base_model, lr=1e-4)
    history = tuner.fit(train_ds, val_ds, epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE)

    # Save final model and history
    final_path = os.path.join(OUTPUT_DIR, 'final_finetuned_model.keras')
    base_model.save(final_path)
    print(f"\n💾 Final model saved → {final_path}")

    with open(os.path.join(OUTPUT_DIR, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'chars.json'), 'w') as f:
        json.dump({'chars': CHARS, 'char_to_idx': CHAR_TO_IDX}, f, indent=2)

    # Plot curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'],   label='Val Loss',   linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('CTC Loss')
    plt.title('Fine-Tuning History – Real Meter Images', fontsize=14, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, 'finetune_curves.png')
    plt.savefig(curve_path, dpi=150)
    print(f"📊 Training curves saved → {curve_path}")

    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"   Original model  : {PRETRAINED_MODEL}  ← UNTOUCHED")
    print(f"   Best fine-tuned : {OUTPUT_DIR}/best_finetuned_model.keras")
    print(f"   Final fine-tuned: {OUTPUT_DIR}/final_finetuned_model.keras")
    print(f"   Chars mapping   : {OUTPUT_DIR}/chars.json")
    print(f"   Loss curves     : {OUTPUT_DIR}/finetune_curves.png")
    print("=" * 60)
    print("\n🚀 NEXT STEP:")
    print("   python test_finetuned_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
