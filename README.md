# Retinopathy of Prematurity (ROP) Detection & Severity Classification

> **Undergraduate Research Opportunities Program (UROP) Project**  
> *Department of Computer Science & Engineering, SRM University-AP*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, leak-free deep learning system designed in PyTorch to address **multi-center domain shift** and **extreme class imbalance** in automated Retinopathy of Prematurity (ROP) screening and severe-stage detection.

---

## Table of Contents
1. [Overview & Clinical Motivation](#overview--clinical-motivation)
2. [Quickstart: Run Inference in 60 Seconds](#quickstart-run-inference-in-60-seconds)
3. [Environment Setup & Installation](#environment-setup--installation)
4. [Repository Structure](#repository-structure)
5. [Datasets & Folder Organization](#datasets--folder-organization)
6. [Reproducing the Full Pipeline (Step-by-Step)](#reproducing-the-full-pipeline-step-by-step)
7. [Training on Kaggle / Cloud GPUs](#training-on-kaggle--cloud-gpus)
8. [Experimental Results & Benchmark Performance](#experimental-results--benchmark-performance)
9. [Repository Maintenance: File Hygiene & Cleanup](#repository-maintenance-file-hygiene--cleanup)
10. [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Clinical Motivation

Retinopathy of Prematurity (ROP) is a vasoproliferative retinal disorder and a primary cause of preventable childhood blindness worldwide. Clinical deployment of AI screening models faces two major roadblocks:

1. **Multi-Center & Multi-Camera Domain Shift**: Fundus cameras vary widely (e.g., RetCam 3, Phoenix ICON, Forus 3nethra neo, RetCam Envision). Models trained on single-center cohorts exhibit sharp performance drops when tested on external hospitals.
2. **Extreme Class Imbalance on Urgent Sight-Threatening Cases**: Severe stages—**Stage 4** (subtotal retinal detachment), **Stage 5** (total retinal detachment), and **Aggressive ROP (A-ROP)**—are inherently rare compared to normal or mild cases. Standard cross-entropy optimization yields misleadingly high global accuracy while failing to detect sight-threatening cases.

### Two-Task Formulation
* **Task A (Binary Screening)**: Normal vs. ROP (Determines whether an infant requires specialist referral).
* **Task B (Severity Staging)**: 3-Class grading (`Mild`, `Moderate`, `Severe`) using imbalance-aware loss functions (Class-Weighted Cross-Entropy, Focal Loss).

---

## Quickstart: Run Inference in 60 Seconds

You can evaluate any fundus image immediately using our trained ResNet-50 checkpoints.

### 1. Task A — Binary Screening (Normal vs. ROP)
Uses the trained screening model located at `checkpoints/taskA_resnet50_ce_fold0_best.pt`.

```bash
# Provide the image path via command-line argument:
python predict_taskA.py "ROP-VL/Image/A-ROP/305_2014-12-03_L_105.jpg"

# Or run interactively (it will prompt for the image path):
python predict_taskA.py
```

**Sample Output:**
```text
Task A binary model loaded successfully!
Device: cpu

========== TASK A PREDICTION ==========
Image: ROP-VL/Image/A-ROP/305_2014-12-03_L_105.jpg
Predicted class: ROP
  Normal  :    5.11%
  ROP     :   94.89%
========================================
Research-use output only. Not for standalone clinical diagnosis.
```

---

### 2. Task B — 3-Class Severity Classification (Mild / Moderate / Severe)
Uses the Kaggle-trained model located at `rop_severity_model.pt`.

```bash
# Provide the image path via command-line argument:
python predict.py "ROP-VL/Image/A-ROP/305_2014-12-03_L_105.jpg"

# Or run interactively:
python predict.py
```

**Sample Output:**
```text
Using device: cpu

========== ROP SEVERITY PREDICTION ==========
Image: ROP-VL/Image/A-ROP/305_2014-12-03_L_105.jpg
Predicted class: Severe
Model output scores:
  Mild     :   9.20%
  Moderate :   9.41%
  Severe   :  81.39%
=============================================
Research-use output only. These scores are model probabilities, not calibrated clinical certainty or a medical diagnosis.
```

---

## Environment Setup & Installation

### 1. Prerequisites
- **Python**: Recommended version **3.10, 3.11, or 3.12** (64-bit).
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS.
- **Hardware**: GPU (NVIDIA CUDA) is strongly recommended for full training; CPU is fully supported for inference.

### 2. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/ROP_UROP.git
cd ROP_UROP

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

> [!TIP]
> If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch by following the instructions at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), e.g.:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Repository Structure

```
ROP_UROP/
├── configs/
│   └── config.py                   # Central configuration (paths, hyperparameters, Kaggle/local toggles)
├── src/
│   ├── audit.py                    # 12-point dataset inspection, format verification, and reporting
│   ├── manifest.py                 # Unified harmonization and patient-level fold splitting
│   ├── preprocessing.py            # Retinal field detection, aspect-ratio padding, and augmentations
│   ├── dataset.py                  # PyTorch ROPDataset and DataLoader builders
│   ├── models.py                   # Model architecture factory (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny)
│   ├── losses.py                   # CrossEntropy, Class-Weighted CE, and Focal Loss (gamma=2.0)
│   ├── training.py                 # Reproducible trainer with modern AMP mixed precision and early stopping
│   ├── evaluation.py               # AUROC, AUPRC, Sens, Spec, F1, and 95% bootstrap confidence intervals
│   └── visualization.py            # Learning curves, confusion matrices, ROC/PR curves
├── notebooks/
│   ├── 01_dataset_audit.ipynb      # Stage 1 & 2: Dataset inspection & audit execution
│   ├── 02_dataset_harmonization.ipynb # Stage 3: Harmonization and master manifest generation
│   ├── 03_preprocessing.ipynb      # Stage 4: Preprocessing & PyTorch DataLoader validation
│   ├── 04_binary_classification.ipynb # Stage 7: Task A Normal vs ROP screening experiments
│   ├── 05_severe_stage_classification.ipynb # Stage 8: Task B Severe ROP detection & imbalance handling
│   ├── 06_cross_dataset_evaluation.ipynb # Stage 9: Domain shift & zero-shot external testing
│   └── 07_results_analysis.ipynb   # Stage 10: Publication comparisons & clinical synthesis
├── manifests/                      # Pre-computed, leak-free patient-stratified split manifests
│   ├── master_manifest.csv         # Full registry of audited cohort images
│   ├── taskA_manifest.csv          # 6,145 Task A screening images (3,703 Normal, 2,442 ROP)
│   └── taskB_manifest.csv          # 1,052 Task B staging images across mild, moderate, and severe
├── checkpoints/                    # Local training model checkpoints (.pt)
│   └── taskA_resnet50_ce_fold0_best.pt
├── rop_severity_model.pt           # Kaggle-trained Task B severity checkpoint (82.3% Acc, 90.6% AUROC)
├── rop_severity_metrics.json       # Detailed evaluation metrics for rop_severity_model.pt
├── reports/                        # Exported audit reports (CSV, Excel, Markdown)
├── figures/                        # Generated figures and distribution plots
├── predict.py                      # Single-image inference for Task B Severity Staging
├── predict_taskA.py                # Single-image inference for Task A Binary Screening
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules (ignores raw datasets, zips, cache)
└── README.md
```

---

## Datasets & Folder Organization

The benchmark pools **multi-national pediatric retinal cohorts**:

| Dataset | Origin / Clinical Site | Camera Models | Images | Role in Benchmark |
| :--- | :--- | :--- | :---: | :--- |
| **ROP-VL** | Guangdong Women & Children Hospital, China | 3 camera models (<900px, 900–1200px, >1200px) | 2,020 | **Core Training (Task A & Task B)**. Contains Normal, active ROP Stages 1–5, and A-ROP. |
| **Ostrava** | University Hospital Ostrava, Czech Republic | Clarity RetCam 3, Phoenix ICON, Natus RetCam Envision | 6,004 | **Core Training (Task A) & Domain Shift Benchmark**. |
| **HVDROPDB (v3)**| Aravind Eye Hospital / HVROP, India | Forus 3nethra neo (2040×2040), Clarity RetCam (640×480) | 185 | **Zero-Shot External Benchmark**. Completely isolated test cohort (`fold = -1`). |
| **FARFUM-RoP** | Farabi Eye Hospital, Tehran, Iran | Clarity RetCam 3 (1600×1200) | 1,533 | **Auxiliary Plus-Disease Cohort** (Normal, Pre-Plus, Plus). |
| **Shenzhen 2024** | Shenzhen Eye Hospital, China | Cropped fundus (512×512) | 1,099 | **Excluded**: Auditing detected duplicate overlaps with ROP-VL with contradictory labels. Excluded to prevent test leakage. |

### Expected Folder Structure
If you acquire raw images, place them in the project root:
```
ROP_UROP/
├── ROP-VL/
│   ├── Image/
│   └── Label/
├── Ostrava/
│   └── images_stack_without_captions/
└── HVDROPDB_v3/
```
*(The pre-computed CSV manifests in `manifests/` already contain all patient-level 5-fold cross-validation splits and labels.)*

### Strict Patient-Level Leakage Prevention
Multiple fundus images from the same neonate must never cross between training and validation splits.
* **Patient-Level Grouping**: Infant ID (`ROPVL_xxxx`, `OST_xxx`) is the exclusive grouping unit.
* **Stratified Group 5-Fold Cross-Validation (`StratifiedGroupKFold`)**: Guarantees zero patient overlap between folds (`len(train_patients & val_patients) == 0`).
* **Untouched Zero-Shot Holdout**: HVDROPDB is reserved as an independent external test set (`fold = -1`).

---

## Reproducing the Full Pipeline (Step-by-Step)

### Step 1: Dataset Audit
Run a comprehensive 12-point health audit across image dimensions, corrupted files, and class counts:
```bash
python src/audit.py
# Or open notebooks/01_dataset_audit.ipynb
```

### Step 2: Harmonize Labels & Build Stratified Folds
Generates `manifests/master_manifest.csv`, `manifests/taskA_manifest.csv`, and `manifests/taskB_manifest.csv`:
```bash
python src/manifest.py
# Or open notebooks/02_dataset_harmonization.ipynb
```

### Step 3: Validate Preprocessing & DataLoaders
Verifies retinal field cropping, black-border suppression, and data augmentation pipelines:
```bash
jupyter notebook notebooks/03_preprocessing.ipynb
```

### Step 4: Train Task A (Binary Screening)
Trains ResNet-50 on 5-fold cross-validation for Normal vs. ROP:
```bash
jupyter notebook notebooks/04_binary_classification.ipynb
```

### Step 5: Train Task B (Severity Staging with Imbalance Handling)
Trains ResNet-50 on Mild vs. Moderate vs. Severe using Class-Weighted Cross-Entropy or Focal Loss ($\gamma = 2.0$):
```bash
jupyter notebook notebooks/05_severe_stage_classification.ipynb
```

### Step 6: Cross-Dataset & Domain Shift Benchmark
Evaluates cross-country zero-shot generalization (China $\rightarrow$ Czech Republic, Czech Republic $\rightarrow$ China, and India external holdout):
```bash
jupyter notebook notebooks/06_cross_dataset_evaluation.ipynb
```

### Step 7: Publication Analysis & Metrics
Generates confusion matrices, ROC/PR curves, and bootstrap 95% confidence intervals:
```bash
jupyter notebook notebooks/07_results_analysis.ipynb
```

---

## Training on Kaggle / Cloud GPUs

If local GPU acceleration is not available, train seamlessly on Kaggle:

1. **Upload Code**: Upload `src/`, `configs/`, `manifests/`, and `notebooks/` to a Kaggle Notebook.
2. **Attach Dataset**: Add the ROP datasets as Kaggle input datasets.
3. **Set Accelerator**: In Kaggle, select `Settings > Accelerator > GPU T4 x2` or `GPU P100`.
4. **Run Training**: Execute `05_severe_stage_classification.ipynb`. `configs/config.py` automatically detects Kaggle environments (`/kaggle/working/`) and saves outputs to the working directory.
5. **Download Artifacts**: Download `rop_severity_model.pt` and `rop_severity_metrics.json` into your local project root.

---

## Experimental Results & Benchmark Performance

### Task B — Severe-Stage ROP Classification (Kaggle GPU Run)
* **Architecture**: ResNet-50 (Fine-tuned, Dropout 0.2)
* **Objective**: Weighted Cross-Entropy with inverse class frequencies
* **Validation Cohort**: 209 patient-isolated retinal images

| Metric | Result |
| :--- | :---: |
| **Overall Accuracy** | **82.30%** |
| **Macro F1-Score** | **78.91%** |
| **Macro OvR AUROC** | **90.58%** |

#### Per-Class Breakdown
| Class | Support | Precision | Recall | F1-Score | OvR AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mild** (Stage 1 & 2) | 114 | 86.78% | 92.11% | **89.36%** | **92.63%** |
| **Moderate** (Stage 3) | 48 | 65.31% | 66.67% | **65.98%** | **85.36%** |
| **Severe** (Stage 4, 5, A-ROP) | 47 | 89.74% | 74.47% | **81.40%** | **93.75%** |

> [!NOTE]
> Severe stage detection achieved an **AUROC of 93.75%** and **89.74% precision**, demonstrating strong sensitivity to urgent, sight-threatening pathology despite severe class imbalance.

---

## Repository Maintenance: File Hygiene & Cleanup

When organizing or preparing the repository for GitHub, keep the following distinction in mind:

### Essential Files to Keep in Git
- `src/*.py` — All modular production pipeline code.
- `configs/config.py` — Central project settings.
- `manifests/*.csv` — Pre-computed patient-stratified split manifests.
- `notebooks/*.ipynb` — Complete reproduction notebooks (Stages 1 through 7).
- `predict.py` & `predict_taskA.py` — Easy-to-run inference scripts.
- `rop_severity_metrics.json` — Evaluation benchmarks.
- `requirements.txt` & `.gitignore` — Environment specifications and exclusion rules.
- `README.md` — Project documentation.

### Checkpoints (>100MB)
- `rop_severity_model.pt` (282 MB) and `checkpoints/*.pt` (282 MB) are essential for running local inference.
- Because GitHub rejects files larger than 100 MB via standard `git push`, do **not** stage them into Git directly. Store them via [Git LFS](https://git-lfs.com/) or attach them as binary assets on a GitHub Release page.

### Temporary & Scratch Files Safe to Remove
The following files were created during initial debugging and are safe to delete to keep your workspace clean:
- `archive (2).zip` (~8 GB) — Downloaded raw zip archive (delete to reclaim 8 GB of disk space).
- `25514449.zip` (~200 MB) — Downloaded raw archive.
- `ROP_VSCode_Model_Package.zip` (~260 MB) — Redundant archive (already extracted to `rop_severity_model.pt` and `rop_severity_metrics.json`).
- `audit2.py`, `audit3.py`, `audit4.py`, `audit5.py`, `audit6.py`, `audit_hv.py` — Early exploration scripts (all superseded by `src/audit.py`).
- `audit*_output.txt`, `verify_output.txt`, `manifest_summary.txt`, `env_info.txt` — Temporary terminal logs.
- `build_manifest.py`, `preprocess.py`, `verify_all.py` — Interim one-off scripts (superseded by `src/manifest.py` and `src/preprocessing.py`).
- `dup_montage.jpg`, `dup_pairs.csv`, `hv_vs_train.jpg` — Interim audit visualizer outputs.
- `manifest.csv` — Old unorganized root manifest (superseded by `manifests/master_manifest.csv`).
- Empty text files: `project_files.txt`, `top_level_files.txt`, `top_level_folders.txt`.

---

## Citation & Acknowledgments

This research is conducted under the **Undergraduate Research Opportunities Program (UROP)** at SRM University-AP.

```bibtex
@misc{rop_imbalance_aware_2026,
  author = {UROP ROP Research Team},
  title = {Improving Cross-Dataset Generalization and Severe-Stage Detection in Retinopathy of Prematurity Using Imbalance-Aware Deep Learning},
  year = {2026},
  publisher = {Department of Computer Science & Engineering, SRM University-AP}
}
```
