

# 🧠 EEG-Based ADHD Detection for Adults

**An end-to-end EEG signal processing and machine learning framework for adult ADHD detection, across 11 experimental tasks.**

[Overview](#-overview) • [Dataset](#-dataset) • [Pipeline](#-pipeline-architecture) • [Models](#-models) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [XAI](#-explainability-xai)

</div>

---

## 📌 Overview

This project implements a full-stack EEG-based ADHD classification system for adult populations. It covers the complete workflow from raw `.mat` signal files through preprocessing, feature engineering, and classification using both classical machine learning and modern deep learning approaches.

**Key Highlights:**
- 11 experimental EEG tasks per subject (eyes-open/closed, cognitive challenge, Omni Harmonic, etc.)
- 4 participant groups: (FC),(MC),(FADHD),(MADHD)
- Binary classification: Control (0) vs ADHD (1)
- 10-model ML sweep + 4 deep learning architectures
- Ensemble soft-vote fusion achieving ~95%+ on 11-task subject-wise fusion
- SHAP + ExtraTrees explainability analysis confirming theta-band biomarkers

---

## 📂 Dataset

Each `.mat` file is a cell array of shape `11 tasks × N_subjects × N_samples × 2 channels`.

**Sampling Rate:** 256 Hz | **Channels per task:** 2 (varies by task)

| Task ID | Name | Channels | Duration |
|---------|------|----------|----------|
| T1 | Eyes Open Baseline 1 | Cz, F4 | 30 s |
| T2 | Eyes Closed 1 | Cz, F4 | 20 s |
| T3 | Eyes Open 1 | Cz, F4 | 20 s |
| T4 | Cognitive Challenge | Cz, F4 | 45 s |
| T5 | Pre-Omni Baseline | Cz, F4 | 15 s |
| T6 | Omni Harmonic | Cz, F4 | 30 s |
| T7 | Eyes Open Baseline 2 | O1, F4 | 30 s |
| T8 | Eyes Closed 2 | O1, F4 | 30 s |
| T9 | Eyes Open 2 | O1, F4 | 30 s |
| T10 | Eyes Closed F3/F4 | F3, F4 | 45 s |
| T11 | Eyes Closed Fz/F4 | Fz, F4 | 45 s |

**Expected directory structure:**
```
EEG/
├── FC.mat      
├── MC.mat       
├── FADHD.mat    
└── MADHD.mat    
```

---

## 🏗️ Pipeline Architecture

```
Raw .mat Files
      │
      ▼
  Preprocessing
  (re-reference → bandpass 0.5–40 Hz → notch 50 Hz → ICA)
      │
      ├──► Feature Extraction (48+ statistical & spectral features)
      │          │
      │          └── ML Models: 9-model sweep + LightGBM IMPROVED
      │                         (Optuna + SMOTE + TTA)
      │
      └──► Raw Sequences (512 samples × 2 channels)
                 │
                 ├── 1D-CNN
                 ├── LSTM (Bidirectional + Attention)
                 ├── CNN-LSTM
                 └── EEGNet+SE v6
                           │
                           ▼
                    Ensemble Soft-Vote (ML + DL)
                           │
                           ▼
                  XAI — SHAP + Feature Importance
```

---

## ⚙️ Signal Preprocessing

Each raw EEG signal passes through four sequential steps:

1. **Re-referencing** — subtract per-channel mean
2. **Bandpass filter** — zero-phase Butterworth (0.5–40 Hz, order 4)
3. **Notch filter** — 50 Hz power-line noise removal (IIR, Q=30)
4. **ICA** — MNE FastICA; components with |kurtosis| > 5 flagged as artifacts and removed

---

## 🔬 Feature Engineering

### Base Features (34 per sample)
For each of the 2 EEG channels:

| Feature Type | Features |
|---|---|
| Band powers | Delta, Theta, Alpha, Beta, Gamma (Welch PSD) |
| Time-domain stats | Mean, Std, Skewness, Kurtosis, RMS |
| Clinical ratios | Theta/Alpha, Theta/Beta, Beta/Alpha, Delta/Theta |
| Entropy & Complexity | Spectral entropy, ApEn, Peak-to-peak amplitude |

### Augmented Features (56 per sample — Part B)
Additional cross-channel and log-transformed features:
- Laterality indices (theta, beta, alpha)
- Global band ratios & log-power features
- Asymmetry scores
- Cross-channel coherence products

### Windowed Features (Subject-wise study)
- 4-second sliding windows, 50% overlap (stride = 512 samples)
- Per window: all 34 base features + Hjorth parameters (activity, mobility, complexity) + cross-channel stats
- Aggregated per subject-task: mean, std, min, max across windows + window count

---

## 🤖 Models

### Machine Learning — 10-Model Sweep (5-fold CV)

| Model | Key Configuration |
|-------|------------------|
| SVM-RBF | C=100, γ=0.001, PCA(0.98) |
| SVM-Linear | C=50, PCA(0.98) |
| Random Forest | 500 trees, max_depth=20 |
| ExtraTrees | 500 trees, max_depth=20 |
| Gradient Boosting | 400 estimators, lr=0.02 |
| KNN | k=5, distance-weighted |
| Logistic Regression | C=0.1, L-BFGS |
| MLP | (256, 128, 64), adaptive LR |
| XGBoost | 500 estimators, max_depth=8 |
| **LightGBM IMPROVED** | **1200 est., Optuna-tuned, SMOTE, TTA** |

**LightGBM IMPROVED Enhancements:**
- 56 engineered features (34 base + cross-channel, log-power, asymmetry, coherence)
- SMOTE oversampling for class imbalance correction
- Optuna hyperparameter optimization (100 trials, TPE sampler)
- Test-Time Augmentation (TTA) — 10 passes with low Gaussian noise
- Fine threshold tuning — grid search over [0.10, 0.91] at 0.005 step

### Deep Learning Models

All models use **5-fold StratifiedKFold**, **AdamW optimizer**, **cosine LR with warm restarts**, **label smoothing**, and **early stopping** (patience=20/60).

#### 1D-CNN
4 convolutional blocks (64→128→256→256 filters) with BatchNorm, GELU activations, MaxPooling, and Dropout. Adaptive average pooling into a 3-layer classifier head.

#### LSTM (Bidirectional + Attention)
3-layer bidirectional LSTM (hidden=128), soft attention pooling over time steps, followed by a classifier head.

#### CNN-LSTM
CNN frontend (64→128→256 conv filters) followed by a 2-layer bidirectional LSTM; uses the last hidden state for classification.

#### EEGNet+SE v6 ⭐ Best Performer
A compact architecture tailored for EEG signals:

| Block | Description |
|-------|-------------|
| Block 1 | Temporal convolution (F1=16, kernel=T/4) |
| Block 2 | Depthwise spatial conv (D=4), Squeeze-and-Excitation attention, AvgPool |
| Block 3 | Separable temporal conv (F2=32), SE attention, AvgPool |
| Classifier | 2-layer FC with dropout |

**Training extras:** Mixup augmentation (α=0.3), time-shift, scale jitter, channel flip, temporal masking, class-weighted loss, cosine annealing with warm restarts, ensemble of N=5 models per fold, TTA (10 passes).

---

## 📊 Results

### EEGNet+SE v6 — Per-Task Performance

| Task | Accuracy | F1 | AUC |
|------|----------|----|-----|
| T1 — Eyes Open Baseline 1 | 0.8992 | 0.8967 | 0.8441 |
| T2 — Eyes Closed 1 | 0.9108 | 0.9101 | 0.8779 |
| T3 — Eyes Open 1 | 0.9367 | 0.9362 | 0.9096 |
| T4 — Cognitive Challenge | 0.9242 | 0.9235 | 0.9005 |
| T5 — Pre-Omni Baseline | **0.9625** | **0.9616** | **0.9355** |
| T6 — Omni Harmonic | 0.9367 | 0.9366 | 0.9371 |
| T7 — Eyes Open Baseline 2 | 0.8983 | 0.8977 | 0.8942 |
| T8 — Eyes Closed 2 | 0.8983 | 0.8975 | 0.8645 |
| T9 — Eyes Open 2 | 0.9242 | 0.9235 | 0.9126 |
| T10 — Eyes Closed F3/F4 | 0.9117 | 0.9111 | 0.9202 |
| T11 — Eyes Closed Fz/F4 | 0.9492 | 0.9483 | 0.9224 |
| **Mean** | **0.9230** | **0.9215** | **0.9108** |

### Subject-wise Fusion (Voting Ensemble, 11 Tasks)

| Setting | Accuracy |
|---------|----------|
| Best single task | ~0.93 |
| 11-task fusion | ~0.95+ |
| Unseen subjects (80/20 split) | See notebook |

---

## 🔗 Ensemble & Fusion

The final ensemble soft-votes probabilities from:
- **LightGBM IMPROVED** — Part B, per-task OOF probabilities
- **EEGNet+SE v6** — per-task OOF scores

Weights are proportional to each model's cross-validated accuracy. A fine-grained threshold search (step=0.005) is applied before final prediction.

---

## 💡 Explainability (XAI)

Three complementary interpretability analyses are applied on the full 11-task fusion model:

1. **ExtraTrees feature importance** — aggregated at task level and feature-family level
2. **SHAP TreeExplainer** — mean |SHAP| values on a 50-sample representative subset
3. **SHAP Beeswarm plot** — top-20 features with direction of effect (ADHD class)

**Feature families analyzed:** band powers (δ/θ/α/β/γ), clinical ratios (θ/β etc.), spectral entropy, Hjorth parameters, cross-channel statistics.

> 🔑 **Key finding:** Theta-band features and the theta/beta ratio consistently rank highest across most tasks and both importance methods, confirming well-established EEG biomarkers of ADHD.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU *(optional but recommended for DL)*
- ~8 GB RAM minimum; 16 GB recommended

### Install Dependencies

```bash
# Core dependencies
pip install mne scipy h5py scikit-learn xgboost lightgbm torch torchvision
pip install tqdm seaborn matplotlib pandas numpy shap optuna imbalanced-learn
pip install decorator==4.4.2
```

> Alternatively, run the **first cell** of the notebook — it handles all installation automatically.

---

## 🚀 Usage

### 1. Configure Paths

Edit the constants at the top of the notebook:

| Variable | Default | Description |
|----------|---------|-------------|
| `FS` | 256 | Sampling frequency (Hz) |
| `LOWCUT` | 0.5 | Bandpass lower cutoff (Hz) |
| `HIGHCUT` | 40.0 | Bandpass upper cutoff (Hz) |
| `NOTCH_HZ` | 50.0 | Notch filter frequency (Hz) |
| `N_FOLDS` | 5 | Cross-validation folds |
| `BATCH_SIZE` | 32 (ML) / 16 (DL) | Training batch size |
| `DL_EPOCHS` | 100 / 400 | Max training epochs |
| `LR` | 1e-3 | Learning rate |
| `SEED` | 42 | Random seed |
| `MAX_LEN` | 512 | DL input sequence length (samples) |
| `WINDOW_SEC` | 4 | Sliding window duration (seconds) |
| `WINDOW_OVERLAP` | 0.5 | Sliding window overlap fraction |
| `DATA_PATH` | `./EEG` | Path to `.mat` data files |
| `OUTPUT_DIR` | `./ml_dl_results` | Path to save outputs |

### 2. Run the Notebook

```bash
jupyter notebook EEG_ADHD_Detection_For_ADULT.ipynb
```

Execute cells sequentially. Each section is independently runnable after preprocessing is complete.

---

## 📁 Project Structure

```
project/
├── EEG_ADHD_Detection_For_ADULT.ipynb   # Main notebook
├── EEG/                                  # Dataset directory
│   ├── FC.mat                            # Female Control
│   ├── MC.mat                            # Male Control
│   ├── FADHD.mat                         # Female ADHD
│   └── MADHD.mat                         # Male ADHD
└── ml_dl_results/                        # Auto-created output directory
    ├── ml/                               # ML results, CM, ROC, feature importance
    ├── dl/                               # DL results, loss curves, heatmaps
    ├── comparison/                       # Cross-model comparison charts
    └── summary/                          # Unified CSVs, SHAP plots, ablation results
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `mne` | EEG signal processing, ICA |
| `scipy` | Filtering, signal processing |
| `scikit-learn` | ML models, cross-validation, SMOTE |
| `xgboost` / `lightgbm` | Gradient boosting classifiers |
| `torch` | Deep learning (CNN, LSTM, EEGNet) |
| `optuna` | Hyperparameter optimization |
| `imbalanced-learn` | SMOTE oversampling |
| `shap` | Model explainability |
| `h5py` | Loading `.mat` files |
| `matplotlib` / `seaborn` | Visualization |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for advancing EEG-based neurological diagnostics
</div>
