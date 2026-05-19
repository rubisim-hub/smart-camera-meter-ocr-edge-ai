"""
TEST FINE-TUNED MODEL ON REAL IMAGES
====================================
Tests the fine-tuned model on all 35 real labeled images.
- Model: finetuned_meter_model/best_finetuned_model.keras
- Real data: real_image_labels.json (35 labeled images)
- Reports: Accuracy by condition and overall
"""

import os
# Fix working directory so relative paths always resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ==================== CONFIGURATION ====================
MODEL_PATH       = "finetuned_meter_model/best_finetuned_model.keras"
CHARS_PATH       = "finetuned_meter_model/chars.json"
REAL_LABELS_PATH = "real_image_labels.json"
IMAGE_WIDTH      = 800
IMAGE_HEIGHT     = 200

print("=" * 60)
print("🧪 TESTING FINE-TUNED MODEL ON REAL IMAGES")
print("=" * 60)

# ==================== LOAD CHARACTER MAPPING ====================
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
NUM_CLASSES = len(CHARS) + 1

print(f"📁 Model  : {MODEL_PATH}")
print(f"📄 Labels : {REAL_LABELS_PATH}")
print(f"🎯 Chars  : {CHARS}")
print("=" * 60)

# ==================== LOAD MODEL ====================
print("\n🔄 Loading fine-tuned model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ==================== LOAD GROUND TRUTH ====================
print(f"\n🔄 Loading ground truth from {REAL_LABELS_PATH}...")

with open(REAL_LABELS_PATH, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

ground_truth = {}
conditions   = {}

if isinstance(gt_data, list):
    for item in gt_data:
        filename  = item['filename']
        label     = item.get('full_reading', item.get('label', ''))
        condition = item.get('condition', 'Unknown')
        ground_truth[filename] = label
        conditions[filename]   = condition
else:
    raise ValueError(f"Unexpected JSON format: expected list, got {type(gt_data)}")

print(f"✅ Loaded {len(ground_truth)} labeled images")
print(f"   Conditions: {set(conditions.values())}")

# ==================== PREPROCESSING ====================
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # channel
    arr = np.expand_dims(arr, axis=0)    # batch
    return arr

def decode_prediction(prediction):
    decoded  = []
    prev_idx = -1
    for t in range(prediction.shape[1]):
        idx = int(np.argmax(prediction[0, t, :]))
        if idx != NUM_CLASSES - 1 and idx != prev_idx and idx < len(CHARS):
            decoded.append(IDX_TO_CHAR[idx])
        prev_idx = idx
    return ''.join(decoded)

# ==================== TESTING ====================
print(f"\n🧪 Processing {len(ground_truth)} labeled images...\n")

results_by_condition = {}
all_results          = []

for filename, gt_label in ground_truth.items():
    condition = conditions[filename]

    # Find image file recursively from project root
    img_path = None
    for root, dirs, files in os.walk("."):
        if filename in files:
            img_path = os.path.join(root, filename)
            break

    if img_path is None:
        print(f"⚠️  Image not found: {filename}")
        continue

    try:
        arr        = load_and_preprocess_image(img_path)
        prediction = model.predict(arr, verbose=0)
        pred_text  = decode_prediction(prediction)
        correct    = (pred_text == gt_label)

        result = {
            'filename':    filename,
            'condition':   condition,
            'ground_truth': gt_label,
            'prediction':  pred_text,
            'correct':     correct
        }
        all_results.append(result)

        if condition not in results_by_condition:
            results_by_condition[condition] = []
        results_by_condition[condition].append(result)

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# ==================== REPORT RESULTS ====================
print("\n" + "=" * 60)
print("📊 RESULTS BY CONDITION:")
print("=" * 60)

total_correct = 0
total_tested  = 0

for condition, results in sorted(results_by_condition.items()):
    correct  = sum(1 for r in results if r['correct'])
    total    = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    total_correct += correct
    total_tested  += total

    print(f"\n🏷️  {condition}:")
    print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"   {status} {r['filename']:20s}  GT: {r['ground_truth']:12s}  Pred: {r['prediction'] or '(empty)':12s}")

overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0

print("\n" + "=" * 60)
print(f"🎯 OVERALL: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
print("=" * 60)
print(f"\n📊 COMPARISON:")
print(f"   Before fine-tuning : 0/{total_tested} (0.0%)")
print(f"   After fine-tuning  : {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
print(f"   Improvement        : +{overall_accuracy:.1f}%")
print("=" * 60)

# Save results
output_path = "finetuned_test_results.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'overall_accuracy': overall_accuracy,
        'total_correct':    total_correct,
        'total_tested':     total_tested,
        'by_condition': {
            cond: {
                'accuracy': sum(1 for r in res if r['correct']) / len(res) * 100,
                'correct':  sum(1 for r in res if r['correct']),
                'total':    len(res)
            }
            for cond, res in results_by_condition.items()
        },
        'detailed_results': all_results
    }, f, indent=2)

print(f"\n💾 Detailed results saved to: {output_path}")
print("=" * 60)
