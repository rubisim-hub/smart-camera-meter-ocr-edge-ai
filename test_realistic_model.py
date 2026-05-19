"""
TEST REALISTIC METER MODEL ON REAL IMAGES
=========================================
Tests the trained realistic meter model on 35 labeled real meter images.
- Model: realistic_meter_model/best_model.keras
- Real data: real_image_labels.json (35 labeled images)
- Reports: Accuracy by condition and overall
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ==================== CONFIGURATION ====================
MODEL_PATH = "realistic_meter_model/best_model.keras"
CHARS_PATH = "realistic_meter_model/chars.json"
REAL_LABELS_PATH = "real_image_labels.json"
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 200

print("=" * 60)
print("🧪 TESTING REALISTIC METER MODEL ON REAL IMAGES")
print("=" * 60)

# Load character mapping
with open(CHARS_PATH, 'r') as f:
    char_data = json.load(f)
    CHARS = char_data['chars']
    CHAR_TO_IDX = char_data['char_to_idx']
    IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
    NUM_CLASSES = len(CHARS) + 1

print(f"📁 Model: {MODEL_PATH}")
print(f"📄 Labels: {REAL_LABELS_PATH}")
print(f"🎯 Character set: {CHARS}")
print("=" * 60)

# ==================== LOAD MODEL ====================
print("\n🔄 Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ==================== LOAD GROUND TRUTH ====================
print(f"\n🔄 Loading ground truth from {REAL_LABELS_PATH}...")

with open(REAL_LABELS_PATH, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

# Parse ground truth (handle different JSON formats)
ground_truth = {}
conditions = {}

if isinstance(gt_data, list):
    # Format: [{"image_path": ..., "filename": ..., "full_reading": ..., "condition": ...}, ...]
    for item in gt_data:
        filename = item['filename']
        label = item['full_reading']
        condition = item.get('condition', 'Unknown')
        ground_truth[filename] = label
        conditions[filename] = condition
else:
    raise ValueError(f"Unexpected JSON format. Expected list, got {type(gt_data)}")

print(f"✅ Loaded {len(ground_truth)} labeled images")
print(f"   Conditions: {set(conditions.values())}")

# ==================== PREPROCESSING ====================
def load_and_preprocess_image(img_path):
    """Load and preprocess image for model"""
    img = Image.open(img_path).convert('L')  # Grayscale
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

def decode_prediction(prediction):
    """Decode CTC prediction to text"""
    # Get the most likely class at each timestep
    input_length = prediction.shape[1]
    
    # Greedy decoding
    decoded = []
    prev_idx = -1
    
    for t in range(input_length):
        # Get argmax at this timestep
        idx = np.argmax(prediction[0, t, :])
        
        # CTC decoding: remove blanks and repeated characters
        if idx != NUM_CLASSES - 1 and idx != prev_idx:  # Not blank and not repeat
            if idx < len(CHARS):
                decoded.append(IDX_TO_CHAR[idx])
        
        prev_idx = idx
    
    return ''.join(decoded)

# ==================== TESTING ====================
print("\n🧪 Processing 35 labeled images...\n")

results_by_condition = {}
all_results = []

for filename, gt_label in ground_truth.items():
    condition = conditions[filename]
    
    # Find image path
    img_path = None
    for root, dirs, files in os.walk("."):
        if filename in files:
            img_path = os.path.join(root, filename)
            break
    
    if img_path is None:
        print(f"⚠️  Image not found: {filename}")
        continue
    
    try:
        # Load and preprocess
        img_array = load_and_preprocess_image(img_path)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        pred_text = decode_prediction(prediction)
        
        # Evaluate
        correct = (pred_text == gt_label)
        
        # Store result
        result = {
            'filename': filename,
            'condition': condition,
            'ground_truth': gt_label,
            'prediction': pred_text,
            'correct': correct
        }
        all_results.append(result)
        
        # Group by condition
        if condition not in results_by_condition:
            results_by_condition[condition] = []
        results_by_condition[condition].append(result)
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")

# ==================== REPORT RESULTS ====================
print("\n" + "=" * 60)
print("📊 RESULTS BY CONDITION:")
print("=" * 60)

total_correct = 0
total_tested = 0

for condition, results in sorted(results_by_condition.items()):
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    total_correct += correct
    total_tested += total
    
    print(f"\n🏷️  {condition}:")
    print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Show first 3 examples
    for i, result in enumerate(results[:3]):
        status = "✅" if result['correct'] else "❌"
        print(f"   {status} {result['filename']:15s} GT: {result['ground_truth']:12s} Pred: {result['prediction'] or '(empty)':12s}")

# Overall accuracy
overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0

print("\n" + "=" * 60)
print(f"🎯 OVERALL: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
print("=" * 60)

# Save detailed results
output_path = "realistic_test_results.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_tested': total_tested,
        'by_condition': {
            cond: {
                'accuracy': sum(1 for r in results if r['correct']) / len(results) * 100,
                'correct': sum(1 for r in results if r['correct']),
                'total': len(results)
            }
            for cond, results in results_by_condition.items()
        },
        'detailed_results': all_results
    }, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

# ==================== RECOMMENDATIONS ====================
print("\n" + "=" * 60)
print("📈 RECOMMENDATIONS:")
print("=" * 60)

if overall_accuracy >= 80:
    print("🎉 EXCELLENT! Model works very well on real images!")
    print("   → Ready for production use")
    print("   → Consider scaling to more real data for 90%+")
elif overall_accuracy >= 60:
    print("✅ GOOD! Significant improvement over previous 0%")
    print("   → Realistic synthetic data approach worked!")
    print("   → Fine-tune on more real images to reach 80%+")
elif overall_accuracy >= 40:
    print("📊 MODERATE: Better than baseline, but needs improvement")
    print("   → Add more realistic effects to synthetic data")
    print("   → Fine-tune on 100+ labeled real images")
    print("   → Analyze failure cases to improve generator")
elif overall_accuracy >= 20:
    print("⚠️  LOW: Some learning, but large sim-to-real gap remains")
    print("   → Analyze real vs synthetic visual differences")
    print("   → Consider hybrid approach: synthetic + real fine-tuning")
else:
    print("❌ NEEDS WORK: Minimal improvement")
    print("   → Synthetic data may not match real meters well")
    print("   → Collect and label 200+ real images for supervised training")

print("=" * 60)
