# Smart Camera Meter OCR: Ultra-Lightweight Edge AI

An ultra-lightweight OCR pipeline for reading analog utility meter values on resource-constrained edge devices. This project explores an end-to-end sequence-recognition approach using a **CRNN (Convolutional Recurrent Neural Network)** with **CTC loss**, targeting deployment on an **ESP32-C6** microcontroller. The work combines large-scale synthetic data generation, training on realistic synthetic meter images, and fine-tuning on a small labeled real-image set.  
Source project report: [OCR Edge AI PDF](https://www.genspark.ai/api/files/s/jfqN8PC1)

---

## Project Overview

This repository contains the code and artifacts for a smart camera OCR system for electricity meter reading.

### Main idea
- Generate a large realistic synthetic dataset of meter images
- Train a CRNN + CTC model on the synthetic dataset
- Fine-tune the trained model on labeled real meter images
- Evaluate the model by real-world image condition
- Prepare the pipeline for future lightweight edge deployment

### Key reported project details
- Target device: **ESP32-C6**
- Model family: **CRNN + CTC**
- Synthetic training set: **100,000 images**
- Real labeled photos: **35 manually labeled images** (with **31 usable labeled examples** reported in the paper)
- Fine-tuned real-image accuracy:
  - **Overall:** 6.45%
  - **Night:** 22.22%
- Base realistic model accuracy on real images before fine-tuning:
  - **Overall:** 0.0%

Sources: [Project report](https://www.genspark.ai/api/files/s/jfqN8PC1) · [Fine-tuned results](https://www.genspark.ai/api/files/s/2weNtZEQ) · [Pre-fine-tuning results](https://www.genspark.ai/api/files/s/JSX7xoi0)

---

## Repository Structure

```text
smart-camera-meter-ocr-edge-ai/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── generate_realistic_meters.py
├── train_on_realistic.py
├── finetune_on_real.py
├── test_realistic_model.py
├── test_finetuned_model.py
│
├── export-coco-2019.json
├── real_image_labels.json
│
├── realistic_meter_dataset/
│   ├── images/
│   ├── validation.csv
│   └── preview_grid.png
│
├── realistic_meter_model/
│   ├── best_model.keras
│   ├── chars.json
│   └── ...
│
├── finetuned_meter_model/
│   ├── best_finetuned_model.keras
│   ├── final_finetuned_model.keras
│   ├── chars.json
│   └── ...
│
├── real_images/
│   ├── .gitkeep
│   └── README.md
│
├── results/
│   ├── realistic_test_results.json
│   └── finetuned_test_results.json
│
├── reports/
│   └── final_project_paper.pdf

