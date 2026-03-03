# 🩺 Diabetic Retinopathy Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-82.4%25-success)

> A deep learning pipeline for automated **diabetic retinopathy severity grading** from retinal fundus images using two-phase transfer learning with EfficientNetB3. Achieves **82.4% validation accuracy** and **0.81 Quadratic Weighted Kappa** on 5-class grading.

---

## 📌 Problem Statement

Diabetic retinopathy (DR) is one of the leading causes of preventable blindness worldwide, affecting over 100 million people with diabetes. Manual screening by ophthalmologists is expensive, slow, and unavailable in many regions. This project builds an automated multi-class classifier to grade DR severity from retinal fundus photographs — enabling faster triage and wider access to screening.

**Severity Classes (0–4):**

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No signs of retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms, less than severe |
| 3 | Severe | 20+ intraretinal hemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularization, highest risk of blindness |

---

## 📂 Dataset

- **Source:** [APTOS 2019 Blindness Detection — Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **Size:** ~3,662 labeled retinal fundus images
- **Format:** High-resolution JPEG + CSV labels
- **Class Distribution:** Heavily skewed toward Grade 0 (~49%)
- **Split:** 80% train / 10% validation / 10% test (stratified)

> ⚠️ Data is **not** included in this repo. Download from Kaggle and place inside `data/raw/`.

---

## 🧠 Approach

```
Raw Images
    ↓
Ben Graham Preprocessing (CLAHE + circular crop + Gaussian subtract)
    ↓
Albumentations Augmentation Pipeline
    ↓
EfficientNetB3 Backbone (ImageNet pretrained)
    ↓
Phase 1: Frozen backbone → Train classifier head only (10 epochs, lr=1e-3)
    ↓
Phase 2: Full fine-tuning (40 epochs, lr=1e-5, ReduceLROnPlateau)
    ↓
Evaluation: QWK + Accuracy + Confusion Matrix + GradCAM
```

### Key Design Decisions

**Ben Graham Preprocessing** — A retinal image technique that subtracts a Gaussian-blurred version of the image from itself, revealing fine vascular details that standard normalization misses. This single step gave ~4% accuracy improvement in experiments.

**Two-Phase Fine-Tuning** — Training only the head first prevents large gradient updates from corrupting pretrained features before the classifier stabilizes. Phase 1 warms up the head; Phase 2 fine-tunes the entire network at a low learning rate.

**Quadratic Weighted Kappa (QWK)** — The APTOS competition metric. It penalizes predictions proportionally to how far off they are (0→4 is worse than 0→1), which is more clinically meaningful than plain accuracy.

**Class-Weighted Loss** — Addresses severe class imbalance without oversampling, which caused memorization of minority-class augmentations in experiments.

---

## 🏗️ Model Architecture

```
Input Image (224 × 224 × 3)
        │
EfficientNetB3 Backbone (pretrained on ImageNet)
  ├─ Phase 1: All layers FROZEN
  └─ Phase 2: All layers UNFROZEN
        │
GlobalAveragePooling2D
        │
Dense(512) → BatchNormalization → Dropout(0.4)
        │
Dense(256) → BatchNormalization → Dropout(0.3)
        │
Dense(5, activation='softmax')
        │
  Grade 0  1  2  3  4
```

**Training Config:**

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Backbone | Frozen | Unfrozen |
| Learning Rate | 1e-3 | 1e-5 |
| Epochs | 10 | 40 |
| Loss | Weighted Categorical CE | Weighted Categorical CE |
| Batch Size | 32 | 32 |

Callbacks: `EarlyStopping` (patience=7), `ReduceLROnPlateau` (factor=0.3, patience=3), `ModelCheckpoint` (monitor QWK)

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | **82.4%** |
| Quadratic Weighted Kappa | **0.81** |
| Macro AUC (OvR) | **0.94** |
| Test Accuracy | **81.1%** |

Plots saved in `results/figures/`: confusion matrix, training curves, GradCAM visualizations.

---

## 📁 Project Structure

```
diabetic-retinopathy-detection/
│
├── data/
│   ├── raw/                      # Original images & CSVs (gitignored)
│   └── processed/                # Preprocessed images, split CSVs
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb   # Image preprocessing pipeline
│   ├── 03_Model_Training.ipynb  # Two-phase transfer learning
│   └── 04_Evaluation.ipynb      # Metrics, confusion matrix, GradCAM
│
├── src/
│   ├── dataset.py                # tf.data pipeline & augmentation
│   ├── model.py                  # EfficientNetB3 architecture
│   ├── train.py                  # CLI training script
│   ├── evaluate.py               # QWK, AUC, confusion matrix
│   └── utils.py                  # Ben Graham preprocessing, GradCAM
│
├── models/                       # Best model checkpoints (.keras)
├── results/
│   └── figures/                  # All plots and visualizations
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone & set up environment
```bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset
```bash
kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
unzip data/raw/aptos2019-blindness-detection.zip -d data/raw/
```

### 3. Run notebooks in order
```
01_EDA → 02_Preprocessing → 03_Model_Training → 04_Evaluation
```

### 4. Or run CLI training
```bash
python src/train.py --epochs 50 --batch_size 32 --img_size 224
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow 2.x / Keras | Model building & training |
| EfficientNetB3 | Pretrained backbone |
| Albumentations | Image augmentation |
| OpenCV | Ben Graham preprocessing |
| Scikit-learn | QWK, stratified splits |
| Matplotlib / Seaborn | Visualizations |

---

## 📈 Key Learnings

- Ben Graham preprocessing was the single most impactful change — more than swapping architectures
- Two-phase fine-tuning is critical; skipping Phase 1 caused training instability
- QWK better reflects clinical severity than accuracy for ordinal grading tasks
- GradCAM confirmed the model focuses on clinically meaningful regions (hemorrhages, exudates)

---

## 🔗 References

- [APTOS 2019 Kaggle Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [EfficientNet Paper (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Ben Graham's Preprocessing (Kaggle Discussion)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801)

---



