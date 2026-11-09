"""
Configuration file for ROP UROP Project.
Supports local Windows/Linux, Google Colab, and Kaggle environments with GPU support.
All dataset directories are configurable.
"""

import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(os.environ.get("ROP_PROJECT_ROOT", r"D:\UROP"))

# Detect Colab environment
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

# Detect Kaggle environment
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle")

if IS_KAGGLE:
    # On Kaggle: input datasets are read-only at /kaggle/input/<dataset-name>/
    # Output/working directory is /kaggle/working/
    # The dataset name you create on Kaggle (e.g., "rop-urop-data")
    _kaggle_input = Path("/kaggle/input")
    _kaggle_ds = _kaggle_input / "rop-urop-data"  # adjust to your Kaggle dataset name
    if _kaggle_ds.exists():
        PROJECT_ROOT = _kaggle_ds
    else:
        # fallback: check for any dataset folder in /kaggle/input/
        for p in _kaggle_input.iterdir():
            if p.is_dir():
                PROJECT_ROOT = p
                break
    # Writable output directory for Kaggle
    KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
elif IS_COLAB:
    # On Colab, check for mounted Google Drive or local content folder
    if os.path.exists("/content/drive/MyDrive/ROP_UROP"):
        PROJECT_ROOT = Path("/content/drive/MyDrive/ROP_UROP")
    elif os.path.exists("/content/ROP_UROP"):
        PROJECT_ROOT = Path("/content/ROP_UROP")
    else:
        PROJECT_ROOT = Path("/content")

# Dataset Directories (configurable via environment variables)
ROPVL_DIR = Path(os.environ.get("ROPVL_DIR", PROJECT_ROOT / "ROP-VL"))
OSTRAVA_DIR = Path(os.environ.get("OSTRAVA_DIR", PROJECT_ROOT / "Ostrava"))
FARFUM_DIR = Path(os.environ.get("FARFUM_DIR", PROJECT_ROOT / "FARFUM_RoP"))

# HVDROPDB: check both standard and v3 classification locations
_hv_v3 = PROJECT_ROOT / "HVDROPDB_v3" / "HVDROPDB_RetCam_Neo_Classification"
_hv_direct = PROJECT_ROOT / "HVDROPDB" / "HVDROPDB_RetCam_Neo_Classification"
_hv_v3_root = PROJECT_ROOT / "HVDROPDB_v3"
HVDROPDB_DIR = Path(os.environ.get("HVDROPDB_DIR", _hv_v3 if _hv_v3.exists() else (_hv_direct if _hv_direct.exists() else _hv_v3_root)))

SHENZHEN_DIR = Path(os.environ.get("SHENZHEN_DIR", PROJECT_ROOT / "Shenzhen"))

# Project Output Directories
# On Kaggle, input datasets are READ-ONLY, so outputs go to /kaggle/working/
_output_root = KAGGLE_OUTPUT_DIR if IS_KAGGLE else PROJECT_ROOT

MANIFESTS_DIR = PROJECT_ROOT / "manifests"      # read from input dataset
REPORTS_DIR = _output_root / "reports"
FIGURES_DIR = _output_root / "figures"
CHECKPOINTS_DIR = _output_root / "checkpoints"
RESULTS_DIR = _output_root / "results"
PROCESSED_DIR = _output_root / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure writable output directories exist
for p in [REPORTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)
# Only create manifests dir if writable (not on Kaggle input)
if not IS_KAGGLE:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment & Model Configurations
SEED = 42
IMG_SIZE = 224
PREPROCESS_SIZE = 256
HAS_GPU = IS_COLAB or IS_KAGGLE  # both provide GPUs
BATCH_SIZE = 32 if HAS_GPU else 16
NUM_WORKERS = 4 if HAS_GPU else 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported Backbones
BACKBONES = ["resnet50", "efficientnet_b0", "convnext_tiny"]

# Task A Mapping
TASK_A_SOURCES = ["ROPVL", "OSTRAVA"]
TASK_A_EXTERNAL = "HVDROPDB"

# Task B Severity Categories (ROP-VL)
TASK_B_CLASSES = ["mild", "moderate", "severe"]
TASK_B_BINARY_CLASSES = ["non_severe", "severe"]  # non_severe = mild + moderate; severe = severe
