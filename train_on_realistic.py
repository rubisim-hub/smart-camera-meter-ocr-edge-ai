"""
TRAIN CRNN MODEL ON REALISTIC METER DATASET
===========================================
Trains a CRNN sequence model on the realistic synthetic meter images.
- Dataset: realistic_meter_dataset/validation.csv
- Architecture: CNN + Bidirectional LSTM + CTC Loss
- Training: 90K train, 10K validation
- Expected: High synthetic accuracy + better real-world generalization
"""

import os
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== CONFIGURATION ====================
DATASET_DIR = "realistic_meter_dataset"
CSV_PATH = os.path.join(DATASET_DIR, "validation.csv")
OUTPUT_DIR = "realistic_meter_model"
BATCH_SIZE = 16
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 200

# Character set (digits + symbols)
CHARS = "0123456789.-"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

print("=" * 60)
print("🚀 TRAINING CRNN MODEL ON REALISTIC METER DATASET")
print("=" * 60)
print(f"📁 Dataset: {CSV_PATH}")
print(f"🎯 Character set: {CHARS}")
print(f"📊 Vocab size: {len(CHARS)}")
print(f"🔢 Classes (with CTC blank): {NUM_CLASSES}")
print("=" * 60)

# ==================== DATA LOADING ====================
def load_dataset(csv_path, train_split=0.9):
    """Load dataset from validation.csv"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(DATASET_DIR, row['filename'])
            label = row['text']
            data.append((img_path, label))
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(data)
    
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"✅ Loaded dataset:")
    print(f"   Train: {len(train_data):,}")
    print(f"   Val:   {len(val_data):,}")
    
    return train_data, val_data

# ==================== DATA PREPROCESSING ====================
def encode_label(label):
    """Encode text label to sequence of indices"""
    return [CHAR_TO_IDX[char] for char in label if char in CHAR_TO_IDX]

def decode_prediction(indices):
    """Decode CTC prediction indices to text"""
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx and idx < len(CHARS):  # Remove repeats and blank
            chars.append(IDX_TO_CHAR[idx])
        prev_idx = idx
    return ''.join(chars)

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def create_dataset(data, batch_size, shuffle=True):
    """Create tf.data.Dataset pipeline"""
    
    def generator():
        if shuffle:
            np.random.shuffle(data)
        for img_path, label in data:
            yield img_path, label
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    
    def process_sample(img_path, label):
        img = load_and_preprocess_image(img_path)
        
        # Encode label
        label_encoded = encode_label(label.numpy().decode('utf-8'))
        label_length = len(label_encoded)
        
        # Pad label to max length
        max_label_len = 15
        label_padded = label_encoded + [NUM_CLASSES-1] * (max_label_len - label_length)
        
        return img, np.array(label_padded, dtype=np.int32), np.array([label_length], dtype=np.int32)
    
    dataset = dataset.map(
        lambda img_path, label: tf.py_function(
            process_sample,
            [img_path, label],
            [tf.float32, tf.int32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes
    dataset = dataset.map(lambda img, label, label_len: (
        tf.ensure_shape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 1]),
        tf.ensure_shape(label, [15]),
        tf.ensure_shape(label_len, [1])
    ))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ==================== MODEL ARCHITECTURE ====================
def build_model():
    """Build CRNN model for meter reading OCR"""
    
    # Input
    input_img = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='image')
    
    # CNN Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)  # 400x100
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 200x50
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 100x25
    
    # Reshape for RNN: (batch, width, height*channels)
    x = layers.Reshape((100, 25*128))(x)  # (batch, 100, 3200)
    
    # Dense projection
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Output layer (CTC) - NO activation, CTC loss needs raw logits
    output = layers.Dense(NUM_CLASSES, activation=None, name='output')(x)
    
    model = keras.Model(inputs=input_img, outputs=output, name='crnn_realistic_meter')
    return model

# ==================== CUSTOM TRAINING WITH CTC LOSS ====================
class CTCTrainer:
    """Custom trainer with CTC loss"""
    
    def __init__(self, model):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.train_loss_tracker = keras.metrics.Mean(name='train_loss')
        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')
    
    @tf.function
    def train_step(self, images, labels, label_lengths):
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(images, training=True)
            
            # Prepare for CTC loss
            input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
            
            # CTC loss (using log_softmax internally)
            loss = tf.nn.ctc_loss(
                labels=labels,
                logits=y_pred,
                label_length=tf.cast(tf.squeeze(label_lengths, axis=-1), tf.int32),
                logit_length=tf.cast(input_length, tf.int32),
                blank_index=NUM_CLASSES-1,
                logits_time_major=False
            )
            loss = tf.reduce_mean(loss)
        
        # Backward pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss_tracker.update_state(loss)
        return loss
    
    @tf.function
    def val_step(self, images, labels, label_lengths):
        # Forward pass
        y_pred = self.model(images, training=False)
        
        # Prepare for CTC loss
        input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        
        # CTC loss
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=y_pred,
            label_length=tf.squeeze(label_lengths, axis=-1),
            logit_length=input_length,
            blank_index=NUM_CLASSES-1,
            logits_time_major=False
        )
        loss = tf.reduce_mean(loss)
        
        self.val_loss_tracker.update_state(loss)
        return loss
    
    def fit(self, train_dataset, val_dataset, epochs, patience=5):
        """Training loop with early stopping"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("─" * 60)
            
            # Training
            self.train_loss_tracker.reset_state()
            step = 0
            for images, labels, label_lengths in train_dataset:
                loss = self.train_step(images, labels, label_lengths)
                step += 1
                if step % 150 == 0:
                    print(f"Step {step}: loss={loss.numpy():.4f}")
            
            train_loss = self.train_loss_tracker.result().numpy()
            
            # Validation
            self.val_loss_tracker.reset_state()
            for images, labels, label_lengths in val_dataset:
                self.val_step(images, labels, label_lengths)
            
            val_loss = self.val_loss_tracker.result().numpy()
            
            # Record history
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            
            print(f"\n📊 train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.model.save(os.path.join(OUTPUT_DIR, 'best_model.keras'))
                print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ No improvement ({patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
        
        return history

# ==================== TRAINING FUNCTION ====================
def train_model():
    """Main training function"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    train_data, val_data = load_dataset(CSV_PATH)
    
    # Create datasets
    print("\n🔄 Creating data pipelines...")
    train_dataset = create_dataset(train_data, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_data, BATCH_SIZE, shuffle=False)
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_model()
    model.summary()
    
    # Save model architecture
    print(f"\n💾 Total params: {model.count_params():,}")
    
    # Train
    print("\n🏋️  Starting training...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Early stopping patience: {EARLY_STOP_PATIENCE}")
    print("=" * 60)
    
    trainer = CTCTrainer(model)
    history = trainer.fit(train_dataset, val_dataset, epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE)
    
    # Save final model
    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
    
    # Save training history
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save character mapping
    with open(os.path.join(OUTPUT_DIR, 'chars.json'), 'w') as f:
        json.dump({'chars': CHARS, 'char_to_idx': CHAR_TO_IDX}, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"💾 Best model: {OUTPUT_DIR}/best_model.keras")
    print(f"📊 Training curves: {OUTPUT_DIR}/curves.png")
    print(f"📄 History: {OUTPUT_DIR}/history.json")
    print("=" * 60)
    print("\n🚀 NEXT STEP: Test on real images")
    print("   python test_on_real.py")
    print("=" * 60)

def plot_training_curves(history):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CTC Loss', fontsize=12)
    plt.title('Training History - Realistic Meter Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'curves.png'), dpi=150)
    print(f"📊 Training curves saved: {OUTPUT_DIR}/curves.png")

# ==================== MAIN ====================
if __name__ == "__main__":
    train_model()
