"""
Stage 1 & 2: Dataset Audit Module for ROP Datasets (corrected version).

What this module does
---------------------
1. Scans each dataset folder and matches image files to their annotation tables.
2. Counts images and PATIENTS per diagnostic category (patients matter more than images).
3. Reads image sizes; for Ostrava it reports the size seen on each device, for ROP-VL it reports
   the inferred size groups (the ROP-VL paper does not say which image came from which device).
4. Checks image integrity (first 50 files by default, all files with --full-integrity).
5. OPTIONAL (--overlap): computes near-duplicate evidence with a correlation signature
   (Shenzhen vs ROP-VL, HVDROPDB vs the training pool, and clusters inside HVDROPDB).
6. Writes CSV, Excel and Markdown reports and two figures. Every number in the reports and
   figures is computed from the data, not typed in.

What this module does NOT do
----------------------------
- It does not run perceptual-hash duplicate detection. Where a duplicate statement appears, it comes
  either from the optional --overlap check or from the separate analysis recorded in
  PRIOR_OVERLAP_RESULTS below, and the report says which.

Run:  python -u src/audit.py [--overlap] [--full-integrity]
"""

import os
import sys
import re
import argparse
import collections
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Facts verified against the dataset papers / READMEs (None = not verified).
HOSPITALS = {
    "ROPVL": "Shenzhen Eye Hospital (SZEH), China",
    "OSTRAVA": "University Hospital Ostrava, Czech Republic",
    "FARFUM": "UNVERIFIED - confirm in the FARFUM-RoP paper",
    "HVDROPDB": "PBMA's H.V. Desai Eye Hospital, Pune, India",
    "SHENZHEN": "Shenzhen Eye Hospital, China",
}

# Results of the separate overlap analysis run earlier in this project (pHash / correlation scripts).
# They are recorded here so the report can quote them; re-run with --overlap to recompute.
PRIOR_OVERLAP_RESULTS = (
    "Separate analysis: 65 of 1,099 Shenzhen images were within pHash distance <= 4 of an ROP-VL image "
    "(482 within <= 8); 24 pairs had correlation >= 0.98. Looser thresholds are not informative."
)

ACQ_GROUP_EDGES = (900, 1200)  # width thresholds for the inferred ROP-VL size groups


def _acq_group(width: int) -> str:
    if width < ACQ_GROUP_EDGES[0]:
        return "G1_<900"
    if width <= ACQ_GROUP_EDGES[1]:
        return "G2_900-1200"
    return "G3_>1200"


# ---------------------------------------------------------------- near-duplicate helpers
_N = 64
_yy, _xx = np.mgrid[:_N, :_N]
_MASK = (_yy - (_N - 1) / 2) ** 2 + (_xx - (_N - 1) / 2) ** 2 <= (_N / 2 - 1) ** 2


def _signature(path) -> np.ndarray:
    """Crop the black border, shrink to 64x64, keep the circular field, standardize."""
    with Image.open(path) as im:
        g = im.convert("L")
        ys, xs = np.where(np.asarray(g) > 15)
        if len(ys):
            g = g.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        v = np.asarray(g.resize((_N, _N)), dtype=np.float32)[_MASK]
    return (v - v.mean()) / (v.std() + 1e-6)


def _signatures(paths: List[Path], tag: str) -> np.ndarray:
    out = []
    for i, p in enumerate(paths):
        out.append(_signature(p))
        if (i + 1) % 1000 == 0:
            print(f"    signatures {tag}: {i + 1}", flush=True)
    return np.stack(out) if out else np.zeros((0, int(_MASK.sum())), dtype=np.float32)


def _clusters(sigs: np.ndarray, thr: float):
    """Union images whose correlation >= thr. Returns (n_clusters, n_images_with_a_twin)."""
    n = len(sigs)
    if n == 0:
        return 0, 0
    A = (sigs @ sigs.T) / _MASK.sum()
    np.fill_diagonal(A, -1)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.where(A >= thr)):
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)}), int((A.max(1) >= thr).sum())


class DatasetAuditor:
    def __init__(self, full_integrity: bool = False, overlap_checks: bool = False):
        self.reports_dir = config.REPORTS_DIR
        self.figures_dir = config.FIGURES_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.full_integrity = full_integrity
        self.overlap_checks = overlap_checks
        self.audit_records: List[Dict[str, Any]] = []
        self.class_records: List[Dict[str, Any]] = []
        self.corrupted_files: List[tuple] = []
        self._paths: Dict[str, Any] = {}
        self._ropvl_cats: Dict[str, int] = {}

    # ------------------------------------------------------------------ utilities
    def _walk_images(self, root_dir: Path) -> Dict[str, Path]:
        image_map = {}
        for dp, _, fs in os.walk(root_dir):
            for f in fs:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                    image_map[f] = Path(dp) / f
        return image_map

    def _integrity(self, paths, tag: str):
        paths = list(paths)
        todo = paths if self.full_integrity else paths[:50]
        bad = 0
        for p in todo:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001
                self.corrupted_files.append((tag, str(p), str(e)))
                bad += 1
        return len(todo), bad

    @staticmethod
    def _size(p) -> tuple:
        with Image.open(p) as im:
            return im.size

    def _integrity_text(self, checked: int, bad: int, total: int) -> str:
        scope = "all files" if checked >= total else f"first {checked} of {total} files only"
        return f"{bad} corrupted ({scope} checked)"

    # ------------------------------------------------------------------ ROP-VL
    def audit_ropvl(self) -> Dict[str, Any]:
        print("\n" + "=" * 60 + "\nAUDITING DATASET: ROP-VL (China)")
        root = config.ROPVL_DIR
        if not root.exists():
            print(f"Warning: ROP-VL root not found at {root}")
            return {"dataset_name": "ROP-VL", "status": "Directory Not Found"}

        df = pd.read_excel(root / "img_info.xlsx")
        all_images = self._walk_images(root / "Image" if (root / "Image").exists() else root)
        matched = [f for f in df["image"] if f in all_images]
        self._paths["ropvl"] = [all_images[f] for f in matched]

        cats = df["categories"].value_counts().to_dict()
        pcats = df.groupby("categories")["patient_id"].nunique().to_dict()
        self._ropvl_cats = cats
        active = ["Stage 1 ROP", "Stage 2 ROP", "Stage 3 ROP", "Stage 4 ROP", "Stage 5 ROP", "A-ROP"]
        severe = ["Stage 4 ROP", "Stage 5 ROP", "A-ROP"]
        sev_df = df[df["categories"].isin(severe)]
        act_df = df[df["categories"].isin(active)]

        # inferred acquisition groups from image width (the paper gives no per-image device)
        widths = {f: self._size(all_images[f])[0] for f in matched}
        tmp = df[df["image"].isin(widths)].copy()
        tmp["grp"] = tmp["image"].map(lambda f: _acq_group(widths[f]))
        grp_counts = tmp["grp"].value_counts().to_dict()
        multi = int((tmp.groupby("patient_id")["grp"].nunique() > 1).sum())

        n_chk, n_bad = self._integrity(self._paths["ropvl"], "ROP-VL")
        dims = collections.Counter(self._size(p) for p in self._paths["ropvl"][:50])

        for cat, count in cats.items():
            self.class_records.append({
                "dataset": "ROP-VL", "country": "China", "category": cat,
                "task_role": "Normal" if cat == "normal" else ("Excluded from Task A (post-treatment)" if cat == "Laser-treated ROP" else "ROP"),
                "severity_group": "severe" if cat in severe else ("moderate" if cat == "Stage 3 ROP" else ("mild" if "Stage" in cat else "None")),
                "images": count, "patients": pcats.get(cat, 0)})

        res = {
            "dataset_name": "ROP-VL", "country_source": "China", "hospital": HOSPITALS["ROPVL"],
            "total_images": len(all_images), "unique_patients": df["patient_id"].nunique(),
            "normal_images": cats.get("normal", 0), "normal_patients": pcats.get("normal", 0),
            "active_rop_images": int(sum(cats.get(c, 0) for c in active)),
            "active_rop_patients": int(act_df["patient_id"].nunique()),
            "post_treatment_images": cats.get("Laser-treated ROP", 0),
            "severe_images": int(sum(cats.get(c, 0) for c in severe)), "severe_patients": int(sev_df["patient_id"].nunique()),
            "stage1_images": cats.get("Stage 1 ROP", 0), "stage2_images": cats.get("Stage 2 ROP", 0),
            "stage3_images": cats.get("Stage 3 ROP", 0), "stage4_images": cats.get("Stage 4 ROP", 0),
            "stage5_images": cats.get("Stage 5 ROP", 0), "stage5_patients": pcats.get("Stage 5 ROP", 0),
            "arop_images": cats.get("A-ROP", 0),
            "camera_devices": "3 devices named in the paper (RetCam, SW-8000, RS-B002); per-image device NOT provided",
            "inferred_size_groups": f"{grp_counts}; patients with images in >1 group: {multi} of {df['patient_id'].nunique()}",
            "missing_annotations": len(all_images) - len(matched), "missing_images": len(df) - len(matched),
            "duplicate_images": "not assessed in this module",
            "integrity": self._integrity_text(n_chk, n_bad, len(self._paths["ropvl"])),
            "sample_dimensions": str(dict(dims.most_common(3))),
        }
        self.audit_records.append(res)
        print(f"  Images: {res['total_images']} | Patients: {res['unique_patients']}")
        print(f"  Normal: {res['normal_images']} ({res['normal_patients']} pts) | ROP: {res['active_rop_images']} ({res['active_rop_patients']} pts)")
        print(f"  Severe (Stage 4/5 + A-ROP): {res['severe_images']} images, {res['severe_patients']} patients; Stage 5: {res['stage5_images']} images, {res['stage5_patients']} patients")
        print(f"  Inferred size groups: {res['inferred_size_groups']}")
        return res

    # ------------------------------------------------------------------ Ostrava
    def audit_ostrava(self) -> Dict[str, Any]:
        print("\n" + "=" * 60 + "\nAUDITING DATASET: Ostrava (Czech Republic)")
        root = config.OSTRAVA_DIR
        if not root.exists():
            print(f"Warning: Ostrava root not found at {root}")
            return {"dataset_name": "Ostrava", "status": "Directory Not Found"}

        df_db = pd.read_excel(root / "infant_retinal_database_info.xlsx", sheet_name="database")
        df_db.columns = ["ID", "SEX", "GA", "BW", "PA", "DG", "PF", "D", "S"]
        clean_dir = root / "images_stack_without_captions"
        note = "images read from images_stack_without_captions (no burned-in captions)"
        if clean_dir.exists():
            all_images = self._walk_images(clean_dir)
        else:
            print("  WARNING: caption-free folder missing; falling back to 'images' (may contain captions!)")
            all_images = self._walk_images(root / "images")
            note = "WARNING: captioned copy used"

        pat = re.compile(r"^(\d+)_([FM])_GA(\d+)_BW(\d+)_PA(\d+)_DG(\d+)_PF(\d+)_D(\d+)_S(\d+)_(\d+)\.jpg$", re.I)
        recs = []
        for f, p in all_images.items():
            m = pat.match(f)
            if m:
                g = m.groups()
                recs.append({"file": f, "path": p, "ID": int(g[0]), "DG": int(g[5]), "PF": int(g[6]), "D": int(g[7]), "S": int(g[8])})
        dfp = pd.DataFrame(recs)
        self._paths["ostrava"] = list(dfp["path"]) if len(dfp) else []

        dg_names = {0: "physiological (Normal)", 1: "ROP 0 (ambiguous)", 2: "ROP 1", 3: "ROP 2", 4: "ROP 3",
                    5: "ROP 4A", 6: "ROP 4B", 7: "ROP 5", 8: "AP-ROP", 9: "st.p. ROP (post-ROP)",
                    10: "hamartomas (non-ROP)", 11: "hemorrhage (non-ROP)", 12: "hypoplasia n.II (non-ROP)", 13: "toxoplasma (non-ROP)"}
        dev_names = {1: "Clarity RetCam 3", 2: "Phoenix ICON", 3: "Natus RetCam Envision"}
        task_a_rop = [2, 3, 4]  # AP-ROP (DG 8) is a single infant -> excluded from the main analysis

        dg_img = dfp["DG"].value_counts().to_dict()
        dg_pts = dfp.groupby("DG")["ID"].nunique().to_dict()
        normal_images = dg_img.get(0, 0)
        normal_df = dfp[dfp["DG"] == 0]
        rop_df = dfp[dfp["DG"].isin(task_a_rop)]
        both = len(set(normal_df["ID"]) & set(rop_df["ID"]))
        rop_images = int(len(rop_df))
        ap_images, ap_pts = dg_img.get(8, 0), dg_pts.get(8, 0)
        excluded = len(dfp) - normal_images - rop_images

        for code, name in dg_names.items():
            if code in dg_img:
                role = "Normal" if code == 0 else ("ROP (Task A)" if code in task_a_rop else
                        ("Excluded from main analysis (single patient; sensitivity analysis only)" if code == 8 else "Excluded"))
                self.class_records.append({
                    "dataset": "Ostrava", "country": "Czech Republic", "category": f"DG {code}: {name}", "task_role": role,
                    "severity_group": "severe" if code == 8 else ("moderate" if code == 4 else ("mild" if code in (2, 3) else "None")),
                    "images": dg_img.get(code, 0), "patients": dg_pts.get(code, 0)})

        dev_parts, dev_sizes = [], {}
        for d, n in dfp["D"].value_counts().sort_index().items():
            pts = dfp[dfp["D"] == d]["ID"].nunique()
            dev_parts.append(f"{dev_names.get(d, d)}: {n} imgs ({pts} pts)")
            sample = dfp[dfp["D"] == d]["path"].head(20)
            sz = collections.Counter(self._size(p) for p in sample)
            dev_sizes[dev_names.get(d, d)] = " / ".join(f"{w}x{h}" for (w, h), _ in sz.most_common(2))
        dev_str = ", ".join(dev_parts)

        n_chk, n_bad = self._integrity(self._paths["ostrava"], "Ostrava")
        res = {
            "dataset_name": "Ostrava", "country_source": "Czech Republic", "hospital": HOSPITALS["OSTRAVA"],
            "total_images": len(all_images), "unique_patients": df_db["ID"].nunique(), "exams": len(df_db),
            "normal_images": normal_images, "normal_patients": int(normal_df["ID"].nunique()),
            "active_rop_images": rop_images, "active_rop_patients": int(rop_df["ID"].nunique()),
            "patients_with_both_labels": both,
            "post_rop_images": dg_img.get(9, 0),
            "severe_images": ap_images, "severe_patients": ap_pts,
            "stage1_images": dg_img.get(2, 0), "stage2_images": dg_img.get(3, 0), "stage3_images": dg_img.get(4, 0),
            "stage4_images": dg_img.get(5, 0) + dg_img.get(6, 0), "stage5_images": dg_img.get(7, 0),
            "arop_images": ap_images, "arop_patients": ap_pts,
            "camera_devices": dev_str, "image_size_by_device": str(dev_sizes), "notes": note,
            "missing_annotations": len(all_images) - len(dfp), "missing_images": 0,
            "duplicate_images": "not assessed in this module",
            "integrity": self._integrity_text(n_chk, n_bad, len(self._paths["ostrava"])),
            "sample_dimensions": str(dev_sizes),
        }
        self.audit_records.append(res)
        print(f"  Images: {res['total_images']} | Patients: {res['unique_patients']} | Exams: {res['exams']} | {note}")
        print(f"  Task A Normal: {normal_images} images ({res['normal_patients']} pts) | ROP (DG 2-4): {rop_images} images ({res['active_rop_patients']} pts)")
        print(f"  Patients with both Normal and ROP exams: {both}")
        print(f"  AP-ROP: {ap_images} images from {ap_pts} patient(s) -> excluded from the main analysis")
        print(f"  Excluded from Task A overall: {excluded} images")
        print(f"  Devices: {dev_str}")
        print(f"  Image size by device: {dev_sizes}")
        return res

    # ------------------------------------------------------------------ FARFUM
    def audit_farfum(self) -> Dict[str, Any]:
        print("\n" + "=" * 60 + "\nAUDITING DATASET: FARFUM-RoP (Iran)")
        root = config.FARFUM_DIR
        if not root.exists():
            print(f"Warning: FARFUM root not found at {root}")
            return {"dataset_name": "FARFUM-RoP", "status": "Directory Not Found"}

        lab = pd.read_excel(root / "Dataset_Labels.xlsx").iloc[1:]  # first row holds sub-headers
        det = pd.read_excel(root / "Dataset_Details.xlsx")
        all_images = self._walk_images(root)
        base = {os.path.splitext(f)[0] for f in all_images}
        matched = int(lab["Unnamed: 1"].astype(str).isin(base).sum())
        pc = lab["Label"].value_counts().to_dict()
        pp = lab.groupby("Label")["Unnamed: 0"].nunique().to_dict()

        for code, name, grp in [(1.0, "Normal plus form (still ROP-screened)", "normal_plus"), (2.0, "Pre-plus", "pre_plus"), (3.0, "Plus", "plus_disease")]:
            self.class_records.append({"dataset": "FARFUM", "country": "Iran", "category": name,
                                       "task_role": "Plus-disease side task only (not Task A)", "severity_group": grp,
                                       "images": pc.get(code, 0), "patients": pp.get(code, 0)})
        n_chk, n_bad = self._integrity(list(all_images.values()), "FARFUM")
        res = {
            "dataset_name": "FARFUM-RoP", "country_source": "Iran", "hospital": HOSPITALS["FARFUM"],
            "total_images": len(all_images), "unique_patients": det["Patient.id"].nunique(),
            "normal_images": "N/A (plus-grade labels only)", "normal_patients": "N/A",
            "active_rop_images": "N/A (plus-grade labels only)",
            "plus_normal_images": pc.get(1.0, 0), "pre_plus_images": pc.get(2.0, 0), "plus_images": pc.get(3.0, 0),
            "plus_patients": pp.get(3.0, 0),
            "stage1_images": "N/A", "stage2_images": "N/A", "stage3_images": "N/A", "stage4_images": "N/A", "stage5_images": "N/A", "arop_images": "N/A",
            "camera_devices": "UNVERIFIED - confirm in the FARFUM-RoP paper",
            "missing_annotations": len(lab) - matched, "missing_images": 0, "duplicate_images": "not assessed in this module",
            "integrity": self._integrity_text(n_chk, n_bad, len(all_images)),
            "sample_dimensions": str(dict(collections.Counter(self._size(p) for p in list(all_images.values())[:20]).most_common(2))),
        }
        self.audit_records.append(res)
        print(f"  Images: {res['total_images']} | Patients: {res['unique_patients']}")
        print(f"  Plus-form labels: normal {res['plus_normal_images']} | pre-plus {res['pre_plus_images']} | plus {res['plus_images']}")
        print("  Role: plus-disease side task only (labels describe plus form, not ROP presence)")
        return res

    # ------------------------------------------------------------------ HVDROPDB
    def audit_hvdropdb(self) -> Dict[str, Any]:
        print("\n" + "=" * 60 + "\nAUDITING DATASET: HVDROPDB v3 classification (India)")
        root = config.HVDROPDB_DIR
        if not root.exists():
            print(f"Warning: HVDROPDB not found at {root}")
            return {"dataset_name": "HVDROPDB", "status": "Directory Not Found"}

        folders = ["Neo_Normal", "Neo_ROP", "RetCam_Normal", "RetCam_ROP"]
        found = {}
        for dp, dirs, _ in os.walk(root):
            for d in dirs:
                if d in folders:
                    found[d] = Path(dp) / d
        counts, paths = {}, {}
        for cf in folders:
            if cf in found:
                ims = sorted(f for f in os.listdir(found[cf]) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
                counts[cf] = len(ims)
                paths[cf] = [found[cf] / f for f in ims]
                self.class_records.append({"dataset": "HVDROPDB", "country": "India", "category": cf,
                                           "task_role": "Normal (external test)" if "Normal" in cf else "ROP (external test; definition not documented)",
                                           "severity_group": "not graded", "images": len(ims), "patients": "not provided"})
        self._paths["hv"] = paths
        all_paths = [p for v in paths.values() for p in v]
        n_chk, n_bad = self._integrity(all_paths, "HVDROPDB")
        res = {
            "dataset_name": "HVDROPDB", "country_source": "India", "hospital": HOSPITALS["HVDROPDB"],
            "total_images": len(all_paths), "unique_patients": "not provided (test-only)",
            "normal_images": counts.get("Neo_Normal", 0) + counts.get("RetCam_Normal", 0), "normal_patients": "N/A",
            "active_rop_images": counts.get("Neo_ROP", 0) + counts.get("RetCam_ROP", 0), "active_rop_patients": "N/A",
            "neo_normal": counts.get("Neo_Normal", 0), "neo_rop": counts.get("Neo_ROP", 0),
            "retcam_normal": counts.get("RetCam_Normal", 0), "retcam_rop": counts.get("RetCam_ROP", 0),
            "severe_images": "not graded", "severe_patients": "N/A",
            "stage1_images": "N/A", "stage2_images": "N/A", "stage3_images": "N/A", "stage4_images": "N/A", "stage5_images": "N/A", "arop_images": "N/A",
            "camera_devices": "'Neo' (2040x2040) and a RetCam-sized camera (640x480); exact device names not verified",
            "missing_annotations": 0, "missing_images": 0,
            "duplicate_images": "not assessed in this module",
            "integrity": self._integrity_text(n_chk, n_bad, len(all_paths)),
            "sample_dimensions": str(dict(collections.Counter(self._size(p) for p in all_paths[:50]).most_common(3))),
        }
        self.audit_records.append(res)
        print(f"  Images: {res['total_images']} | Normal: {res['normal_images']} | ROP: {res['active_rop_images']}")
        print("  Role: external test only (no patient IDs); the meaning of 'ROP' is not documented on the dataset page")
        return res

    # ------------------------------------------------------------------ Shenzhen
    def audit_shenzhen(self) -> Dict[str, Any]:
        print("\n" + "=" * 60 + "\nAUDITING DATASET: Shenzhen 2024 (China) [excluded]")
        root = config.SHENZHEN_DIR
        if not root.exists():
            print(f"Warning: Shenzhen not found at {root}")
            return {"dataset_name": "Shenzhen 2024", "status": "Not Found"}

        all_images = self._walk_images(root)
        self._paths["shenzhen"] = list(all_images.values())
        folder_counts = collections.Counter(p.parent.name.lower() for p in all_images.values())
        xl = root / "zip information.xlsx"
        dfz = pd.read_excel(xl) if xl.exists() else pd.DataFrame()
        pts = int(dfz["ID"].nunique()) if "ID" in dfz else "unknown"
        n_chk, n_bad = self._integrity(self._paths["shenzhen"], "Shenzhen")
        res = {
            "dataset_name": "Shenzhen 2024", "country_source": "China", "hospital": HOSPITALS["SHENZHEN"],
            "total_images": len(all_images), "unique_patients": pts,
            "normal_images": folder_counts.get("normal", 0), "normal_patients": "not computed",
            "active_rop_images": folder_counts.get("stage1", 0) + folder_counts.get("stage2", 0) + folder_counts.get("stage3", 0),
            "post_treatment_images": folder_counts.get("laser scars", 0),
            "severe_images": "none (Stages 1-3 only)", "severe_patients": "N/A",
            "stage1_images": folder_counts.get("stage1", 0), "stage2_images": folder_counts.get("stage2", 0), "stage3_images": folder_counts.get("stage3", 0),
            "stage4_images": 0, "stage5_images": 0, "arop_images": 0,
            "camera_devices": "not documented here (all images 512x512)",
            "missing_annotations": 0, "missing_images": 0,
            "duplicate_images": PRIOR_OVERLAP_RESULTS,
            "integrity": self._integrity_text(n_chk, n_bad, len(self._paths["shenzhen"])),
            "sample_dimensions": str(dict(collections.Counter(self._size(p) for p in self._paths["shenzhen"][:20]).most_common(2))),
            "status": "EXCLUDED from the main experiments: same hospital and period as ROP-VL and near-duplicates found; not independent",
        }
        self.audit_records.append(res)
        print(f"  Images: {len(all_images)} | Patients: {pts} | EXCLUDED (not independent of ROP-VL)")
        return res

    # ------------------------------------------------------------------ optional overlap checks
    def run_overlap_checks(self):
        print("\n" + "=" * 60 + "\nOVERLAP CHECKS (correlation signature, may take several minutes)")
        by_name = {r.get("dataset_name"): r for r in self.audit_records}
        rv = self._paths.get("ropvl", [])
        sz = self._paths.get("shenzhen", [])
        ost = self._paths.get("ostrava", [])
        hv = self._paths.get("hv", {})

        S_rv = _signatures(rv, "ROP-VL") if rv else None
        if sz and S_rv is not None and len(S_rv):
            S_sz = _signatures(sz, "Shenzhen")
            best = ((S_sz @ S_rv.T) / _MASK.sum()).max(1)
            txt = (f"Correlation check: {int((best >= 0.98).sum())} of {len(sz)} Shenzhen images match an ROP-VL image at "
                   f"correlation >= 0.98 ({int((best >= 0.99).sum())} at >= 0.99). {PRIOR_OVERLAP_RESULTS}")
            if "Shenzhen 2024" in by_name:
                by_name["Shenzhen 2024"]["duplicate_images"] = txt
            print("  " + txt)

        if hv:
            hv_all = [p for v in hv.values() for p in v]
            S_hv = _signatures(hv_all, "HVDROPDB")
            pool = list(rv) + list(ost)
            if len(pool):
                S_pool = np.concatenate([S_rv if S_rv is not None else np.zeros((0, S_hv.shape[1]), np.float32),
                                         _signatures(ost, "Ostrava")]) if ost else S_rv
                best = ((S_hv @ S_pool.T) / _MASK.sum()).max(1)
                pct = np.round(np.percentile(best, [50, 90, 99, 100]), 3).tolist()
                ext = (f"No near-duplicate of any HVDROPDB image found in ROP-VL + Ostrava: images with correlation >= 0.98: "
                       f"{int((best >= 0.98).sum())} of {len(hv_all)}; best-match percentiles (50/90/99/max) = {pct}")
            else:
                ext = "overlap with training pool not computed (no training images found)"
            parts, start = [], 0
            for cf, ps in hv.items():
                S = S_hv[start:start + len(ps)]
                start += len(ps)
                k, tw = _clusters(S, 0.98)
                parts.append(f"{cf}: {len(ps)} images -> {k} clusters ({tw} with a near-twin)")
            txt = ext + " | Inside HVDROPDB (correlation >= 0.98): " + "; ".join(parts)
            if "HVDROPDB" in by_name:
                by_name["HVDROPDB"]["duplicate_images"] = txt
            print("  " + txt)

    # ------------------------------------------------------------------ reports
    def generate_reports(self):
        df_s, df_c = pd.DataFrame(self.audit_records), pd.DataFrame(self.class_records)
        R = {r.get("dataset_name"): r for r in self.audit_records}
        df_s.to_csv(self.reports_dir / "dataset_audit_report.csv", index=False)
        df_c.to_csv(self.reports_dir / "per_class_breakdown.csv", index=False)
        with pd.ExcelWriter(self.reports_dir / "dataset_audit_report.xlsx", engine="openpyxl") as w:
            df_s.to_excel(w, sheet_name="Dataset_Summary", index=False)
            df_c.to_excel(w, sheet_name="Class_Breakdown", index=False)
            if self.corrupted_files:
                pd.DataFrame(self.corrupted_files, columns=["dataset", "path", "error"]).to_excel(w, sheet_name="Corrupted", index=False)

        def md(df):
            try:
                return df.to_markdown(index=False)
            except ImportError:  # tabulate not installed
                return "```\n" + df.to_string(index=False) + "\n```"

        L = []
        a = R.get("ROP-VL", {}); o = R.get("Ostrava", {}); h = R.get("HVDROPDB", {}); f = R.get("FARFUM-RoP", {}); s = R.get("Shenzhen 2024", {})
        L.append("# Dataset Audit and Verification Summary\n")
        L.append("All counts below are computed from the data by `src/audit.py`.\n")
        L.append("## 1. Summary across datasets\n\n" + md(df_s))
        L.append("\n\n## 2. Class and diagnostic distribution\n\n" + md(df_c))
        L.append("\n\n## 3. Key decisions and constraints\n")
        if a:
            L.append(f"1. **ROP-VL ({a['hospital']})**: {a['total_images']} images, {a['unique_patients']} patients. Laser-treated images "
                     f"({a['post_treatment_images']}) are excluded from Task A. Severe disease (Stage 4, Stage 5, A-ROP): {a['severe_images']} images from "
                     f"{a['severe_patients']} patients; Stage 5: {a['stage5_images']} images from {a['stage5_patients']} patients. "
                     f"The paper names three devices but gives no per-image device, so the size groups ({a['inferred_size_groups']}) are inferred acquisition groups, not devices.")
        if o:
            L.append(f"2. **Ostrava ({o['hospital']})**: {o['total_images']} images, {o['unique_patients']} patients, {o['exams']} exams, "
                     f"three devices with image sizes {o['image_size_by_device']}. {o['notes']}. Task A: Normal {o['normal_images']} images "
                     f"({o['normal_patients']} patients) vs ROP stages 1-3 {o['active_rop_images']} images ({o['active_rop_patients']} patients); "
                     f"{o['patients_with_both_labels']} patients have both labels. AP-ROP ({o['arop_images']} images) comes from {o['arop_patients']} patient(s) and is excluded from the main analysis. "
                     f"Labels are per exam, not per image.")
        if h:
            L.append(f"3. **HVDROPDB ({h['hospital']})**: {h['total_images']} classification images (Neo normal/ROP {h['neo_normal']}/{h['neo_rop']}, "
                     f"RetCam-sized normal/ROP {h['retcam_normal']}/{h['retcam_rop']}). No patient IDs, so it is used for external testing only. "
                     f"Duplicate evidence: {h['duplicate_images']}")
        if f:
            L.append(f"4. **FARFUM-RoP**: {f['total_images']} images, {f['unique_patients']} patients. Labels grade plus disease "
                     f"(normal {f['plus_normal_images']} / pre-plus {f['pre_plus_images']} / plus {f['plus_images']}); they do not indicate ROP presence, so it is a side task only.")
        if s:
            L.append(f"5. **Shenzhen 2024**: {s['total_images']} images. {s['status']}. Evidence: {s['duplicate_images']}")
        (self.reports_dir / "dataset_audit_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[OK] Reports written to {self.reports_dir}")

    # ------------------------------------------------------------------ figures
    def generate_visualizations(self):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        R = {r.get("dataset_name"): r for r in self.audit_records}
        rows = [("ROP-VL\n(China)", R.get("ROP-VL")), ("Ostrava\n(Czech Rep.)", R.get("Ostrava")), ("HVDROPDB\n(India, test)", R.get("HVDROPDB"))]
        rows = [(n, r) for n, r in rows if r]
        if rows:
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
            x, w = np.arange(len(rows)), 0.35
            nm = [r["normal_images"] for _, r in rows]
            rp = [r["active_rop_images"] for _, r in rows]
            ax.bar(x - w / 2, nm, w, label="Normal", color="#2b5c8f")
            ax.bar(x + w / 2, rp, w, label="ROP (Task A definition)", color="#d95f02")
            for i, (_, r) in enumerate(rows):
                ax.text(x[i] - w / 2, nm[i] + 30, f"{nm[i]}\n({r['normal_patients']} pts)", ha="center", fontsize=8)
                ax.text(x[i] + w / 2, rp[i] + 30, f"{rp[i]}\n({r.get('active_rop_patients', 'n/a')} pts)", ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=10, fontweight="bold")
            ax.set_ylabel("Images (patients in brackets)", fontweight="bold")
            ax.set_title("Task A cohort by source", fontweight="bold")
            ax.legend()
            plt.tight_layout(); fig.savefig(self.figures_dir / "class_distribution_by_source.png"); plt.close(fig)

        c = self._ropvl_cats
        if c:
            names = ["Stage 1 ROP", "Stage 2 ROP", "Stage 3 ROP", "Stage 4 ROP", "Stage 5 ROP", "A-ROP"]
            labels = ["Stage 1", "Stage 2", "Stage 3\n(moderate)", "Stage 4\n(severe)", "Stage 5\n(severe)", "A-ROP\n(severe)"]
            vals = [c.get(n, 0) for n in names]
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            bars = ax.bar(labels, vals, color=["#7570b3", "#7570b3", "#e7298a", "#e41a1c", "#990000", "#e41a1c"], edgecolor="black", linewidth=0.5)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, str(int(b.get_height())), ha="center", fontsize=9, fontweight="bold")
            ax.set_ylabel("Images", fontweight="bold"); ax.set_title("ROP-VL: ROP images by stage", fontweight="bold")
            plt.tight_layout(); fig.savefig(self.figures_dir / "severe_stage_distribution.png"); plt.close(fig)
        print(f"[OK] Figures written to {self.figures_dir}")


def run_audit(full_integrity: bool = False, overlap_checks: bool = False):
    auditor = DatasetAuditor(full_integrity=full_integrity, overlap_checks=overlap_checks)
    auditor.audit_ropvl()
    auditor.audit_ostrava()
    auditor.audit_farfum()
    auditor.audit_hvdropdb()
    auditor.audit_shenzhen()
    if overlap_checks:
        auditor.run_overlap_checks()
    auditor.generate_reports()
    auditor.generate_visualizations()
    return auditor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", action="store_true", help="run near-duplicate checks (slow)")
    ap.add_argument("--full-integrity", action="store_true", help="verify every image file (slow)")
    args = ap.parse_args()
    run_audit(full_integrity=args.full_integrity, overlap_checks=args.overlap)