# AIFX010 — Hoax Hertz (Audio Integrity & Truth Tone)

**Author:** Felipe Camargo de Pauli  
**Email:** fcdpauli@gmail.com

A comprehensive pipeline and library for **audio forgery/manipulation detection** with focus on *clip-level classification* (real vs fake) and acoustic artifacts analysis using MFCC, Ambient Environment Signature (AES), and ENF features.

## Features

- **`hoaxhertz/`** — Complete preprocessing and feature extraction library
- **`ml/factory/pipeline/`** — Modular ML pipeline (data → preprocessing → training → evaluation)
- **GPU acceleration** — Optional cuML/RAPIDS support for faster training
- **Ready-to-use scripts** — Extract features once, train quickly, and run batch or single-file inference
- **Multiple datasets** — Support for ADD, HAD, and PartialSpoof datasets
- **Sidecar architecture** — Efficient feature caching system

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Overview](#dataset-overview)
- [Pipeline Workflow](#pipeline-workflow)
- [GPU Support](#gpu-support)
- [Repository Structure](#repository-structure)
- [Detailed Usage](#detailed-usage)
- [Performance Results](#performance-results)
- [Troubleshooting](#troubleshooting)
- [License & Credits](#license--credits)

## Quick Start

### Prerequisites
- Linux OS
- Python 3.12
- FFmpeg (for audio decoding: .wav/.mp3/.m4a/.ogg)

### Installation

```bash
# 1) Clone and setup environment
git clone https://github.com/felipedepauli/AIFX010_Truth_Tone.git
cd AIFX010_Truth_Tone
source setenv.sh
uv sync
uv pip install -e .     # Install hoaxhertz lib in editable mode
```

### End-to-End Pipeline

```bash
# 2) Download and prepare datasets
python ml/factory/pipeline/task_0_data/download_datasets.py \
  --out ml/data/raw \
  --datasets ADD HAD PartialSpoof DEMAND ESC-50 MUSAN SLR28_RIRS_NOISES TAU_urban_asc_2022_mobile_dev UrbanSound8k VCTK

# 3) Generate data manifests (CSV files with path,label columns)
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --had-root ml/data/raw/HAD/
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --add-root ml/data/raw/ADD/ADD_train_dev/
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --partialspoof-root ml/data/raw/PartialSpoof/

# 4) Extract features (generates sidecar *.feat.npz files)
python ml/factory/pipeline/task_1_preprocess/extract_features.py \
  --csv ml/factory/pipeline/task_0_data/data_processed/train_clips.csv \
  --suffix .feat.npz \
  --use-mfcc --use-ambient --use-enf \
  --sr 16000

# 5) Train classifier with hyperparameter optimization
./ml/factory/pipeline/task_3_train/run.sh

# 6) Evaluate model performance
pushd ml/factory/pipeline/task_4_eval
./run_batck.sh ml/factory/pipeline/task_0_data/data_processed/dev_clips.csv
popd

# 7) Single file inference
python ml/factory/pipeline/task_4_eval/predict_one.py \
  --audio <YOUR_AUDIO.wav> \
  --model ml/factory/experiments/run0/clipclf.joblib \
  --suffix .feat.npz \
  --out-json ml/factory/experiments/run0/pred_one.json
```

## GPU Support (Optional)

Training (SVM/LogisticRegression + StandardScaler) supports **GPU acceleration** through RAPIDS cuML. Feature extraction runs on CPU.

### Requirements
- **CUDA 12.x** toolkit accessible in `LD_LIBRARY_PATH`
- Example: `/usr/local/cuda/lib64` containing `libnvrtc.so.12`, `libcublas.so.12`, etc.

### Installation
```bash
uv pip install --extra-index-url https://pypi.nvidia.com \
  "cupy-cuda12x>=13.0.0" \
  "cuml-cu12==25.8.*"
```

> **Note:** If CUDA libraries are missing, cuML may show warnings during SVM teardown. This doesn't affect results. The system automatically falls back to CPU if GPU libraries are unavailable.

## Dataset Overview

This project supports three main audio forgery detection datasets:

| Dataset | Type of Forgery | Scope | Annotation Level |
|---------|----------------|-------|------------------|
| **ADD** | Complete + partial fake audio | Full cases with noise, compression variations | Clip-level detection + manipulation localization |
| **PartialSpoof** | Partial forgery embedded in genuine audio | Variable forgery proportions | Utterance-level + segment-level detection |
| **HAD** (Half-Truth) | Localized forgery (few words) | Light manipulation, controlled insertion | Altered segment localization + general detection |

### Data Preparation

Each dataset requires specific preprocessing:

1. **ADD (Audio Deepfake Detection)**
   - Contains `train/` and `dev/` directories
   - Labels: `genuine` or `fake`
   - Includes segment-level annotations

2. **HAD (Half-Truth Audio Detection)**
   - Similar structure with temporal segment annotations
   - Format: `start-end-T/start-end-F` (T=true, F=fake)

3. **PartialSpoof**
   - Focused on partially spoofed audio within genuine speech
   - Mixed real and synthesized segments

## Repository Structure

```
├── src/hoaxhertz/             # Core library (pip install -e .)
│   ├── preproc/               # Audio preprocessing utilities
│   ├── features/              # Feature extraction modules
│   │   ├── mfcc.py           # MFCC extraction
│   │   ├── ambient.py        # Ambient Environment Signature (AES)
│   │   └── enf.py            # Electrical Network Frequency (ENF)
│   └── detectors/            # Detection algorithms
├── ml/
│   ├── data/
│   │   ├── raw/              # Original datasets (ADD, HAD, PartialSpoof)
│   │   └── processed/        # Processed and combined data
│   └── factory/
│       ├── experiments/      # Training results and models
│       └── pipeline/         # Modular ML pipeline
│           ├── task_0_data/  # Data preparation and manifests
│           ├── task_1_preprocess/ # Feature extraction
│           ├── task_3_train/ # Model training with hyperparameter optimization
│           └── task_4_eval/  # Model evaluation and inference
├── apps/                     # Application examples
├── research/                 # Research notebooks and papers
└── scripts/                  # Utility scripts
```

## Pipeline Workflow

The audio forgery detection pipeline follows these main stages:

### 1. Data Preparation
- **Input**: Raw audio datasets (ADD, HAD, PartialSpoof)
- **Process**: Generate standardized CSV manifests with `path,label` columns
- **Output**: Consolidated training and evaluation datasets

```bash
# Generate manifests for each dataset
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --had-root ml/data/raw/HAD/
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --add-root ml/data/raw/ADD/ADD_train_dev/
python ml/factory/pipeline/task_0_data/tools/build_manifests.py --partialspoof-root ml/data/raw/PartialSpoof/
```

The system scans raw dataset folders, understands each protocol, and exports standardized CSV files describing:
- Audio file locations
- Real vs fake labels  
- Temporal segments (in seconds) that are falsified

### 2. Feature Extraction
- **Input**: Audio files + manifest CSVs
- **Process**: Extract numerical feature vectors and save as sidecar files
- **Output**: `.feat.npz` files containing feature vectors and metadata

```bash
python ml/factory/pipeline/task_1_preprocess/extract_features.py \
  --csv ml/factory/pipeline/task_0_data/data_processed/train_clips.csv \
  --suffix .feat.npz \
  --use-mfcc --use-ambient --use-enf \
  --sr 16000
```

**Features extracted:**
- **MFCC**: Mel-Frequency Cepstral Coefficients (spectral characteristics)
- **AES**: Ambient Environment Signature (recording environment acoustics)
- **ENF**: Electrical Network Frequency (subtle power grid variations indicating tampering)

**Sidecar Architecture:**
- Creates `.feat.npz` files alongside audio files (doesn't replace original audio)
- Contains: `vector` (np.float32, shape `(D,)`) and `meta` (JSON with cfg_hash, extraction parameters, sr)
- Uses configuration hash for intelligent reprocessing (only recomputes when settings change)

### 3. Model Training
- **Input**: Feature sidecars + labels
- **Process**: Train classifiers with hyperparameter optimization
- **Output**: Trained models (`.joblib`) and performance reports

```bash
# Automatic training with Optuna hyperparameter optimization
./ml/factory/pipeline/task_3_train/run.sh
```

**Training Process:**
1. Load CSV manifests and corresponding `.feat.npz` vectors
2. Filter compatible vectors (same cfg_hash and dimensions)
3. Stack into feature matrix X (N samples × D features)
4. Encode labels (fake=0, real=1)
5. Train/validation split (80/20, stratified)
6. Feature normalization with StandardScaler
7. Train classifier (SVM RBF or LogisticRegression) with `class_weight="balanced"`
8. Hyperparameter optimization using Optuna
9. Evaluate with accuracy_score and classification_report

**Supported Classifiers:**
- **SVM** (default): RBF kernel, handles class imbalance
- **Logistic Regression**: Alternative option with balanced class weights

### 4. Model Evaluation
- **Input**: Trained models + test data
- **Process**: Batch inference and metrics computation
- **Output**: Predictions CSV and comprehensive metrics JSON

```bash
pushd ml/factory/pipeline/task_4_eval
./run_batck.sh ml/factory/pipeline/task_0_data/data_processed/dev_clips.csv
popd
```

**Generates:**
- `dev_predictions.csv`: Per-sample predictions (path, true_label, pred_label, probabilities)
- `dev_metrics.json`: Accuracy, precision, recall, F1-score, ROC-AUC (when applicable)
- Confusion matrices and performance visualizations

## Detailed Usage

### Manifest Generation

The manifest generation process creates standardized CSV files from different dataset formats:

**Output Files:**
- `ml/data/raw/ADD/_manifests/train_clips.csv`
- `ml/data/raw/ADD/_manifests/train_segments.csv`
- `ml/data/raw/PartialSpoof/_manifests/train_clips.csv`
- `ml/data/raw/HAD/_manifests/train_clips.csv`
- `ml/data/processed/train_clips.csv` (combined from all datasets)

### Feature Configuration

The feature extraction system supports multiple acoustic features:

| Feature | Description | Use Case |
|---------|-------------|----------|
| **MFCC** | Mel-frequency cepstral coefficients | Speech/audio spectral characteristics |
| **AES** | Ambient Environment Signature | Recording environment detection |
| **ENF** | Electrical Network Frequency | Power grid artifacts for tampering detection |

### Model Performance

**Example Results** (MFCC+Ambient features, SVM with GPU acceleration):
- Feature vector dimension: **162**
- Validation accuracy: **≈ 0.998**
- Macro average F1: **≈ 0.99** (on imbalanced dataset)

> ⚠️ **Note on Metrics**: Performance metrics may appear inflated on heavily imbalanced datasets. Always evaluate recall of minority class ("real"), precision-recall curves, and consider threshold tuning.

## Performance Results

### Benchmark Results

Training configuration: **MFCC+Ambient** features, SVM (GPU/cuML), aggregated data, 162-dimensional vectors

| Metric | Value | Notes |
|--------|-------|-------|
| Validation Accuracy | ~99.8% | Reproduced in project environment |
| Macro Avg F1 | ~99% | Heavily imbalanced dataset |
| Feature Dimensions | 162 | MFCC + Ambient features combined |

### Important Considerations

- **Class Imbalance**: Datasets are typically heavily imbalanced (more fake than real samples)
- **Evaluation Strategy**: Focus on recall of minority class and precision-recall analysis
- **Threshold Tuning**: Use `predict_batch.py` threshold parameters for operational deployment

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `"No usable examples (all missing/invalid sidecars)"` | Missing feature files | Run `extract_features.py` with same `--suffix` on target CSV paths |
| `"Incompatible dimension when stacking vectors"` | Mixed feature configurations | Re-extract features with consistent configuration (automatic cleanup by `cfg_hash`) |
| `cuML warning (SVMBase.__del__)` | CUDA runtime shutdown during garbage collection | Harmless warning; doesn't affect results. Optionally release CuPy pools in scripts |
| `"Undefined AUC (UndefinedMetricWarning)"` | Dataset contains only one class | Use dataset with both real and fake samples for AUC computation |
| `Python 2.7 syntax errors` | Wrong Python version | Use `source setenv.sh` to activate Python 3.12 environment |
| `Path mismatch in CSV files` | Incorrect repository paths in manifests | Update CSV paths or use corrected manifests |

### CUDA/GPU Issues

- **Missing CUDA libraries**: Ensure `LD_LIBRARY_PATH` includes CUDA toolkit paths
- **cuML installation**: Use NVIDIA PyPI index for proper GPU package versions
- **Fallback behavior**: System automatically falls back to CPU if GPU unavailable

## License & Credits

**Original Authors:** @felipedepauli and collaborators

This project consolidates audio forgery detection research with practical ML pipelines, optional GPU acceleration, and comprehensive evaluation frameworks for audio integrity assessment.

### Citation

If you use this work, please consider citing the original research and datasets:
- ADD Dataset: [Audio Deepfake Detection Challenge](https://zenodo.org/record/4277876)
- HAD Dataset: [Half-Truth Audio Detection](https://zenodo.org/record/4060433)  
- PartialSpoof Dataset: [Partial Spoof Detection](https://zenodo.org/record/4060435)