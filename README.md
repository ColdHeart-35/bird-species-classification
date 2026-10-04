# Bird Species Identification — Deep Learning (CNN + Transfer Learning)

> **25 Indian Bird Species | EfficientNetB0 Transfer Learning | Baseline CNN | HOG+SVM Comparison**

---

## Problem Statement

Given a photograph of a bird, identify its **species** from 25 common Indian birds. This is a 25-class image classification problem with ~37,000 labelled images.

## Objective

Build a robust bird species classifier that:
1. Achieves high accuracy using deep learning (CNN + transfer learning)
2. Compares against a classical baseline (HOG + SVM)
3. Demonstrates key ML concepts for educational use
4. Produces a ready-to-use inference system

---

## Dataset

**Dataset**: Birds-25 (25 Indian Bird Species)

| Split | Images | Notes |
|-------|--------|-------|
| Train | 29,947 | Official train folder; augmented during training |
| Validation | 6,000 | 80% of official valid/ folder (stratified, seed=42) |
| Test | 1,500 | 20% of official valid/ folder — **untouched until final eval** |
| **Total** | **37,447** | |

**Classes** (25 species, alphabetically sorted):
```
Asian-Green-Bee-Eater, Brown-Headed-Barbet, Cattle-Egret,
Common-Kingfisher, Common-Myna, Common-Rosefinch, Common-Tailorbird,
Coppersmith-Barbet, Forest-Wagtail, Gray-Wagtail, Hoopoe, House-Crow,
Indian-Grey-Hornbill, Indian-Peacock, Indian-Pitta, Indian-Roller,
Jungle-Babbler, Northern-Lapwing, Red-Wattled-Lapwing, Ruddy-Shelduck,
Rufous-Treepie, Sarus-Crane, White-Breasted-Kingfisher,
White-Breasted-Waterhen, White-Wagtail
```

**Class balance**: Nearly balanced — ~1,147–1,200 images per class in train.

---

## Preprocessing & Data Augmentation

### Preprocessing (ALL splits)
1. Read JPEG/PNG → RGB (3 channels)
2. Resize to model input size (224×224 for EfficientNetB0; 128×128 for baseline CNN)
3. Scale pixels: `[0, 255] → [0.0, 1.0]`
4. EfficientNetB0: internal `Rescaling(255)` layer converts back to [0,255] for the pretrained backbone

### Augmentation (TRAINING only — never on val/test)
| Transform | Parameters | Why |
|-----------|-----------|-----|
| RandomFlip | horizontal | Birds appear from both sides |
| RandomRotation | ±~15° (factor=0.08) | Slight pose variation |
| RandomZoom | -5% to +10% | Scale variation |
| RandomTranslation | ±5% | Position variation |
| RandomBrightness | ±15% | Lighting variation |
| RandomContrast | ±10% | Contrast variation |

**Why these augmentations?** They replicate realistic natural variation in bird photos. Extreme flips (vertical), heavy crops, or colour distortions would create unrealistic images that could confuse the model.

---

## Model Architecture

### Model 1: HOG + Linear SVM (Baseline, Classical ML)

```
Image (any size)
     ↓
Resize → 64×64, Grayscale, Histogram Equalization
     ↓
HOG Descriptor (shape/edge features)
     ↓
StandardScaler
     ↓
LinearSVC (C=1.0)
     ↓
25-class prediction
```

**Result**: ~17.8% test accuracy — near random for 25 classes. HOG loses colour information crucial for distinguishing bird species.

---

### Model 2: Baseline CNN (Scratch, Educational)

Architecture trained from scratch to demonstrate core CNN concepts:

```
Input: 128×128×3
  ↓
Block 1: Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)   [64×64×32]
  ↓
Block 2: Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2)   [32×32×64]
  ↓
Block 3: Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2)  [16×16×128]
  ↓
Block 4: Conv2D(256, 3×3, relu) → BatchNorm → GlobalAvgPool [256]
  ↓
Dropout(0.5)
  ↓
Dense(25, softmax)
```

**Parameters**: ~1.6M trainable  
**Optimizer**: Adam (lr=1e-3) with ReduceLROnPlateau  
**Trained on**: 200 images/class (5,000 total) subset for CPU feasibility

---

### Model 3: EfficientNetB0 — Transfer Learning (Final Model)

**Two-Stage Training Strategy:**

#### Stage 1: Head-Only Training (Feature Extraction)
```
Input: 224×224×3
  ↓
Rescaling(255) — convert [0,1] → [0,255] for EfficientNet
  ↓
EfficientNetB0 backbone (ALL LAYERS FROZEN)
  → Pretrained on 1.28M ImageNet images
  → Outputs 7×7×1280 feature maps
  ↓
GlobalAveragePooling2D → [1280]
  ↓
BatchNormalization
  ↓
Dropout(0.3)
  ↓
Dense(256, relu)
  ↓
Dropout(0.3)
  ↓
Dense(25, softmax)   ← 25 bird species
```
- **Trainable**: ~330K (head only) of 4M total parameters
- **lr**: 1e-3, up to 15 epochs, EarlyStopping(patience=6)

#### Stage 2: Fine-Tuning
- Unfreeze last **30 backbone layers** (high-level feature detectors)
- **lr**: 1e-5 (10× smaller to prevent catastrophic forgetting)
- Up to 25 epochs, EarlyStopping(patience=8)

**Why EfficientNetB0?**
- Pretrained on 1.28M ImageNet images → rich visual features (edges, textures, shapes, parts)
- Compound scaling: optimal balance of depth, width, and resolution
- ~4M parameters: practical for CPU inference
- Strong baseline: 77.1% ImageNet top-1 accuracy

---

## Hyperparameters

| Hyperparameter | Baseline CNN | EfficientNetB0 Stage 1 | EfficientNetB0 Stage 2 |
|---------------|-------------|----------------------|----------------------|
| Input size | 128×128 | 224×224 | 224×224 |
| Batch size | 32 | 32 | 32 |
| Learning rate | 1e-3 | 1e-3 | 1e-5 |
| Optimizer | Adam | Adam | Adam |
| Loss | Categorical CE | Categorical CE | Categorical CE |
| Dropout | 0.5 | 0.3+0.3 | 0.3+0.3 |
| Max epochs | 25 | 15 | 25 |
| Early stopping patience | 7 | 6 | 8 |
| LR reduction patience | 3 | 3 | 4 |
| LR reduction factor | 0.5 | 0.5 | 0.3 |

---

## Training Procedure

### Preventing Overfitting
1. **Data augmentation** (training only)
2. **Dropout** layers in classification head
3. **Early stopping** — stops when val_accuracy stops improving
4. **ReduceLROnPlateau** — halves LR if val_loss plateaus
5. **Transfer learning** — starts with rich pretrained features
6. **Fine-tuning with very low LR** — prevents catastrophic forgetting

### Monitoring
- Training accuracy & loss
- Validation accuracy & loss
- Best model checkpoint saved automatically

---

## Project Structure

```
bird-species-classification/
├── data/
│   ├── archive/Birds_25/
│   │   ├── train/        ← 29,947 training images (25 classes)
│   │   └── valid/        ← 7,500 images (split 80/20 → val + test)
│   └── processed/        ← HOG features (manifest, .npz files)
├── models/
│   ├── class_labels.json          ← index → species name mapping
│   ├── bird_svm.joblib            ← HOG+SVM trained model
│   ├── baseline_cnn.keras         ← Baseline CNN trained model
│   ├── efficientnet_stage1.keras  ← Stage 1 checkpoint
│   └── efficientnet_finetuned.keras ← Final fine-tuned model
├── results/
│   ├── plots/             ← Training curves, per-class accuracy, comparison
│   ├── metrics/           ← JSON + CSV with all metrics
│   └── confusion_matrix/  ← Confusion matrix heatmaps
├── src/
│   ├── data_loader.py     ← tf.data pipeline, augmentation, class labels
│   ├── train_quick.py     ← CPU-optimised training (baseline CNN + EfficientNet)
│   ├── train_efficientnet.py  ← Full EfficientNetB0 trainer
│   ├── train_baseline_cnn.py  ← Full baseline CNN trainer
│   ├── evaluate.py        ← Full evaluation, confusion matrix, comparison
│   ├── predict_cnn.py     ← Single-image inference
│   ├── train.py           ← HOG+SVM trainer
│   ├── features.py        ← HOG feature extraction
│   └── predict.py         ← HOG+SVM inference
├── scripts/
│   ├── dataset_analysis.py
│   └── prepare_birds25_dataset.py
├── outputs/
│   ├── metrics.json       ← HOG+SVM results
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### Train both models (CPU-optimised)
```bash
# Train baseline CNN (200 img/class) + EfficientNetB0 (400 img/class)
python -m src.train_quick

# Train baseline CNN only
python -m src.train_quick --cnn-only

# Train EfficientNetB0 only (full dataset)
python -m src.train_quick --eff-only --eff-subset 0

# Custom settings
python -m src.train_quick --cnn-subset 200 --eff-subset 400 --stage1-epochs 15 --stage2-epochs 25
```

### Train with full dataset (GPU recommended)
```bash
python -m src.train_efficientnet --stage1-epochs 20 --stage2-epochs 40 --batch-size 32
```

### Evaluate models
```bash
# Evaluate both models and show comparison
python -m src.evaluate --compare

# Evaluate EfficientNetB0 only
python -m src.evaluate --model efficientnet

# Evaluate baseline CNN only
python -m src.evaluate --model baseline_cnn
```

---

## How to Predict a New Image

```bash
# Using EfficientNetB0 (recommended)
python -m src.predict_cnn path/to/bird.jpg

# Using baseline CNN
python -m src.predict_cnn path/to/bird.jpg --model baseline_cnn

# Show top-5 predictions
python -m src.predict_cnn path/to/bird.jpg --top 5
```

**Example output:**
```
──────────────────────────────────────────────────
  Model              : EfficientNetB0 (fine-tuned)
  Image              : kingfisher.jpg
──────────────────────────────────────────────────
  Predicted species  : Common Kingfisher
  Confidence         : 94.2%

  Top 3 predictions:
    1. Common Kingfisher           94.2%  ██████████████████████████████
    2. White Breasted Kingfisher    3.8%  █
    3. Indian Roller                1.2%  ░
──────────────────────────────────────────────────
```

---

## Results

*(Updated after training completes)*

### Model Comparison

| Model | Test Accuracy | Macro F1 | Parameters | Notes |
|-------|-------------|----------|-----------|-------|
| HOG + LinearSVM | 17.8% | 0.174 | N/A | Classical baseline |
| Baseline CNN | TBD | TBD | ~1.6M | Scratch, 5K images |
| EfficientNetB0 Stage 1 | TBD | TBD | ~4M (head: ~330K) | Frozen backbone |
| EfficientNetB0 Fine-Tuned | TBD | TBD | ~4M (partial) | Best model |

---

## Evaluation Metrics

The final model is evaluated on the **untouched test set** (1,500 images):

- **Accuracy**: Overall fraction correctly classified
- **Precision** (per class): TP / (TP + FP) — How often predicted class is correct
- **Recall** (per class): TP / (TP + FN) — How often true class is identified
- **F1-score**: Harmonic mean of Precision and Recall
- **Macro F1**: F1 averaged equally across all 25 classes
- **Weighted F1**: F1 weighted by class support
- **Confusion matrix**: 25×25 grid showing where the model confuses species

---

## Limitations

1. **CPU training** — subset used for feasibility; GPU training on full dataset would improve accuracy
2. **Fine-grained classification** — some species (wagtails, barbets) look similar; errors in those classes are expected
3. **Wild images** — model trained on specific photography conditions; may degrade on very different backgrounds

---

## VIVA EXPLANATION

### 1. Why CNN?
Traditional ML (HOG+SVM) extracts **hand-crafted features** — it doesn't know what's important. A CNN **learns its own features** directly from pixel data through gradient descent. For images, learned convolutional filters can capture colour patterns, shapes, and textures that are far more discriminative than HOG gradients.

### 2. What does convolution do?
A convolution slides a small **filter/kernel** (e.g., 3×3 matrix) across the image, computing a dot product at each position. This produces a **feature map** that highlights where that filter's pattern (e.g., horizontal edge, colour patch) appears. Crucially, the same filter is applied across the whole image — this is **translation invariance**: a wing edge is detected regardless of where it is in the image.

**Formula**: Output[i,j] = Σ(k,l) Input[i+k, j+l] × Filter[k,l]

### 3. What is a kernel/filter?
A kernel is a small matrix of learnable weights (e.g., 3×3 = 9 parameters). During training, kernels are updated via backpropagation to detect the most useful patterns for the task. Early layers learn edges and colours; later layers learn complex patterns like "feathers" or "beak shapes."

### 4. What is a feature map?
After applying a kernel to the input, we get a 2D grid showing the **response strength** at each location. Higher values = the filter's pattern was found there. With 32 filters in the first layer, we get 32 feature maps — the model learns 32 different "detectors."

### 5. Why ReLU?
ReLU (Rectified Linear Unit): `f(x) = max(0, x)`.

- **Introduces non-linearity** — without it, stacking linear layers is still just linear.
- **Fixes vanishing gradients** — unlike sigmoid/tanh which saturate, ReLU has gradient 1 for positive values, so gradients flow well.
- **Sparse activation** — negative values become 0, which is efficient.

### 6. Why pooling?
MaxPooling (e.g., 2×2) takes the maximum value in each 2×2 block:
- **Reduces spatial dimensions** (28×28 → 14×14) — fewer parameters
- **Translation invariance** — the model cares that a wing edge exists, not its exact pixel position
- **Reduces overfitting** — by reducing information, it forces the model to focus on the most important features

### 7. What is backpropagation?
Backpropagation computes the gradient of the loss function with respect to every parameter using the **chain rule** of calculus. Starting from the output loss, it works backwards through each layer, computing how much each weight contributed to the error. The optimizer then updates each weight to reduce the loss.

**Key insight**: We need the loss to be differentiable everywhere — that's why we use ReLU (not step function) and softmax (not argmax).

### 8. Why Adam?
Adam (Adaptive Moment Estimation) combines two improvements over basic SGD:
- **Momentum** (1st moment): Smooths gradient updates, avoids oscillation
- **Adaptive learning rates** (2nd moment): Each parameter gets its own learning rate based on historical gradients
- **Bias correction**: Corrects for initialization bias in early training

Adam requires less hyperparameter tuning than SGD and converges faster in practice.

### 9. What is transfer learning?
Transfer learning reuses a model trained on one large task (ImageNet: 1.28M images, 1,000 classes) for a different but related task (bird species). The intuition: low-level features (edges, textures, colours) are useful for ANY image task. We don't need to relearn them from scratch.

**Why it helps with small datasets**: Our bird dataset has ~30K images. Training a 4M-parameter network from scratch would severely overfit. With ImageNet weights, we start from a strong feature extractor and only adapt the final layers.

### 10. Why fine-tuning?
After training the head on frozen backbone (Stage 1), the high-level backbone features are still tuned for ImageNet. Fine-tuning (Stage 2) allows the backbone's **later layers** to adapt to bird-specific features (plumage patterns, beak shapes) at a very low learning rate.

**Why low LR in fine-tuning?** A high LR would overwrite the valuable pretrained features (**catastrophic forgetting**). A small LR (1e-5 vs 1e-3) makes tiny adjustments that improve discrimination without losing general features.

### 11. What is overfitting?
Overfitting occurs when the model **memorizes training data** instead of learning generalizable patterns. Signs:
- Training accuracy: 99%
- Validation accuracy: 60%
- Large gap between train and val curves

A model that overfits will fail on new, unseen images.

### 12. How did we prevent overfitting?
| Technique | How it helps |
|-----------|-------------|
| Data augmentation | Creates varied training examples → harder to memorize |
| Dropout (0.3, 0.5) | Randomly zeroes neurons → prevents co-adaptation |
| Early stopping | Stops training before the model starts memorizing |
| Transfer learning | Starts with strong features → less training needed |
| ReduceLROnPlateau | Fine-grained optimization when nearing convergence |
| BatchNormalization | Stabilizes and regularizes training |

### 13. Why data augmentation?
The network should recognize a Kingfisher whether the photo is taken from the left or right, slightly tilted, or in different lighting. By applying random flips, rotations, and brightness changes to training images, we **artificially expand** the dataset and expose the model to more varied examples — reducing overfitting and improving generalization.

**Critical rule**: Augmentation is applied ONLY to training data. Applying it to validation/test would contaminate the evaluation.

### 14. What does the confusion matrix show?
A 25×25 matrix where:
- **Row** = true species
- **Column** = predicted species
- **Diagonal** = correct predictions
- **Off-diagonal** = confusions

For example, if the model often confuses White-Wagtail with Gray-Wagtail (both are small grey/white birds), you'll see high off-diagonal values there. This reveals which species are visually similar and hard for the model to distinguish.

### 15. Difference between Precision, Recall and F1?
For each class (say "Indian Peacock"):

- **Precision** = Of all images the model labeled as Peacock, what fraction actually ARE Peacocks? (Measures: how often is the model right when it says "Peacock")
- **Recall** = Of all actual Peacock images, what fraction did the model correctly find? (Measures: how good is the model at finding all Peacocks)
- **F1** = Harmonic mean: `2 × (Precision × Recall) / (Precision + Recall)`. Balances both — useful when classes are imbalanced.

**Example**: A model that always predicts "Peacock" has high recall for Peacock (catches all) but very low precision (wrong most of the time).

### 16. Why was HOG + SVM weaker?
| Aspect | HOG + SVM | CNN |
|--------|-----------|-----|
| Feature extraction | Hand-crafted (edges/gradients) | Learned from data |
| Colour information | Grayscale HOG loses colour | Full RGB |
| Spatial hierarchy | One-level feature | Multi-level: edges → parts → species |
| Scalability | Fixed descriptor size | Adapts via convolution |
| With transfer learning | No equivalent | Leverages 1.28M image features |

HOG was designed for pedestrian detection where shape is dominant. Bird species often differ by **colour patterns** (e.g., Indian Roller's blue wings), which HOG ignores entirely.

### 17. How does the final prediction work?
1. Load bird image from disk
2. Resize to 224×224 pixels
3. Scale pixels to [0.0, 1.0]
4. Add batch dimension: shape becomes (1, 224, 224, 3)
5. Pass through EfficientNetB0 → dense head
6. Softmax output: 25 probabilities summing to 1.0
7. `argmax` → index of highest probability
8. Look up index in `class_labels.json` → species name
9. Report top-3 predictions with confidence percentages

**Key requirement**: Inference preprocessing MUST exactly match training preprocessing. Using different resize or scaling would degrade accuracy significantly.

---

## Citation

Dataset: Birds 25 Species Image Classification (Kaggle)

---

*Project for ML Mini Project — 5th Semester. Demonstrates CNN, transfer learning, fine-tuning, data augmentation, evaluation metrics, and model comparison.*
