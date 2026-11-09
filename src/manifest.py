"""
Stage 3: Dataset Harmonization and Master Manifest Builder.

Standardizes all ROP datasets into a unified schema with:
- Strict patient-level identifiers (e.g. ROPVL_0001, OST_043, FARFUM_012).
- Preservation of original labels and harmonized binary / severity labels.
- Patient-level leak-free Stratified Group 5-Fold cross-validation splits.
- Automated leak-prevention assertions (0 patient overlap across splits).
- Master manifest, Task A manifest, and Task B manifest generation.
"""

import os
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Import configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config

HOSPITALS = {
    "ROPVL": "Guangdong Women & Children Hospital",
    "OSTRAVA": "University Hospital Ostrava",
    "FARFUM": "Farabi Eye Hospital, Tehran",
    "HVDROPDB": "Aravind Eye Hospital / HVROP",
    "SHENZHEN": "Shenzhen Eye Hospital"
}


def build_manifests():
    print("=" * 60, flush=True)
    print("BUILDING MASTER MANIFEST AND PATIENT-LEVEL FOLDS", flush=True)
    
    existing_manifest = config.PROJECT_ROOT / "manifest.csv"
    if not existing_manifest.exists():
        raise FileNotFoundError(f"Source manifest not found at {existing_manifest}")

    df_raw = pd.read_csv(existing_manifest)
    print(f"Loaded {len(df_raw)} source rows from {existing_manifest}", flush=True)

    # Harmonize schema
    df = pd.DataFrame()
    df["image_path"] = df_raw["path"]
    df["dataset_source"] = df_raw["source"]
    df["country"] = df_raw["country"]
    df["hospital"] = df_raw["source"].map(HOSPITALS)
    
    # Camera device
    def get_device(row):
        src = row["source"]
        dev = row["device"]
        acq = row["acq_group"]
        if src == "OSTRAVA":
            return dev
        elif src == "HVDROPDB":
            return "Forus 3nethra neo" if acq == "Neo" else "Clarity RetCam"
        elif src == "FARFUM":
            return "Clarity RetCam 3"
        elif src == "ROPVL":
            return f"Camera_{acq}"
        return dev

    df["camera_device"] = df_raw.apply(get_device, axis=1)

    # Patient IDs
    def parse_orig_pid(row):
        uid = str(row["patient_uid"])
        if "_" in uid:
            return uid.split("_", 1)[1]
        return uid

    df["original_patient_id"] = df_raw.apply(parse_orig_pid, axis=1)
    df["unique_patient_id"] = df_raw["patient_uid"]
    df["exam_id"] = df_raw["exam_id"]
    df["eye"] = df_raw["eye"]
    df["visit_date"] = df_raw["visit_date"]
    df["original_label"] = df_raw["orig_label"]
    df["binary_label"] = df_raw["binary"]
    df["severity_label"] = df_raw["sev_ext"]
    df["image_width"] = df_raw["width"]
    df["image_height"] = df_raw["height"]
    df["ga"] = df_raw["ga"]
    df["bw"] = df_raw["bw"]
    df["exclude_reason"] = df_raw["exclude_reason"]
    df["split"] = df_raw["test_only"].map(lambda t: "test" if t else "train_val")
    df["fold_taskA"] = -1
    df["fold_taskB"] = -1

    # 1. Compute Task A Patient-Level 5-Fold Split
    # Core training pool: ROPVL + OSTRAVA with binary_label not null
    core_mask = df["dataset_source"].isin(["ROPVL", "OSTRAVA"]) & df["binary_label"].notna()
    idx_core = df[core_mask].index.values

    patient_groups = df.loc[idx_core, "unique_patient_id"].values
    strat_labels = (df.loc[idx_core, "dataset_source"] + "_" + df.loc[idx_core, "binary_label"].astype(int).astype(str)).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for fold, (_, val_idx) in enumerate(sgkf.split(idx_core, strat_labels, patient_groups)):
        df.loc[idx_core[val_idx], "fold_taskA"] = fold

    # 2. Compute Task B Patient-Level 5-Fold Split (ROP-VL Severity)
    taskB_mask = (df["dataset_source"] == "ROPVL") & df["severity_label"].notna()
    idx_b = df[taskB_mask].index.values
    b_patients = df.loc[idx_b, "unique_patient_id"].values
    b_labels = df.loc[idx_b, "severity_label"].values

    sgkf_b = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for fold, (_, val_idx) in enumerate(sgkf_b.split(idx_b, b_labels, b_patients)):
        df.loc[idx_b[val_idx], "fold_taskB"] = fold

    # 3. Leak-prevention assertions
    print("\nExecuting automated data-leakage assertions...", flush=True)
    for fold in range(5):
        train_pts = set(df[core_mask & (df["fold_taskA"] != fold) & (df["fold_taskA"] >= 0)]["unique_patient_id"])
        val_pts = set(df[core_mask & (df["fold_taskA"] == fold)]["unique_patient_id"])
        assert len(train_pts & val_pts) == 0, f"Task A fold {fold} patient leakage: {train_pts & val_pts}"

    for fold in range(5):
        train_pts_b = set(df[taskB_mask & (df["fold_taskB"] != fold) & (df["fold_taskB"] >= 0)]["unique_patient_id"])
        val_pts_b = set(df[taskB_mask & (df["fold_taskB"] == fold)]["unique_patient_id"])
        assert len(train_pts_b & val_pts_b) == 0, f"Task B fold {fold} patient leakage: {train_pts_b & val_pts_b}"

    assert df["image_path"].duplicated().sum() == 0, "Duplicate image paths detected!"
    print("[PASS] Automated assertions verified: 0 patient overlap across all folds.", flush=True)

    # 4. Save Master and Subtask Manifests
    config.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    master_path = config.MANIFESTS_DIR / "master_manifest.csv"
    df.to_csv(master_path, index=False)
    print(f"\n[OK] Master manifest saved to {master_path} ({len(df)} rows)", flush=True)

    # Task A Manifest
    taskA_mask = df["dataset_source"].isin(["ROPVL", "OSTRAVA", "HVDROPDB"]) & df["binary_label"].notna()
    taskA_df = df[taskA_mask].copy()
    taskA_path = config.MANIFESTS_DIR / "taskA_manifest.csv"
    taskA_df.to_csv(taskA_path, index=False)
    print(f"[OK] Task A manifest saved to {taskA_path} ({len(taskA_df)} images: {sum(taskA_df['binary_label']==0)} Normal, {sum(taskA_df['binary_label']==1)} ROP)", flush=True)

    # Task B Manifest
    taskB_df = df[taskB_mask].copy()
    taskB_path = config.MANIFESTS_DIR / "taskB_manifest.csv"
    taskB_df.to_csv(taskB_path, index=False)
    print(f"[OK] Task B manifest saved to {taskB_path} ({len(taskB_df)} images: mild/moderate/severe)", flush=True)

    # Print summary tables
    print("\nTask A Fold Distribution (Images per Fold):", flush=True)
    tabA = pd.crosstab(taskA_df["fold_taskA"], taskA_df["dataset_source"] + "_Class" + taskA_df["binary_label"].astype(int).astype(str))
    print(tabA.to_string(), flush=True)

    print("\nTask B Fold Distribution (Images per Fold):", flush=True)
    tabB = pd.crosstab(taskB_df["fold_taskB"], taskB_df["severity_label"])
    print(tabB.to_string(), flush=True)


if __name__ == "__main__":
    build_manifests()
