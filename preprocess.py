import os, random
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold

R = r"D:\UROP"
OUT = os.path.join(R, "processed")
SIZE, SEED = 256, 42
os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(R, "preprocess_summary.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

m = pd.read_csv(os.path.join(R, "manifest.csv"))
m = m[m["source"].isin(["ROPVL", "OSTRAVA", "HVDROPDB"]) & m["binary"].notna()].reset_index(drop=True)
m["binary"] = m["binary"].astype(int)
hv = m["source"] == "HVDROPDB"
m.loc[hv, "patient_uid"] = ["HV_%03d" % i for i in range(int(hv.sum()))]   # placeholder IDs, test-only
m["grp"] = m["source"] + "_" + m["acq_group"].astype(str)
log("images selected:", len(m), "(expected 6145)")

def prep(path):
    im = Image.open(path).convert("RGB")
    ys, xs = np.where(np.asarray(im).max(axis=2) > 15)
    if len(ys): im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = im.size; s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((SIZE, SIZE), Image.LANCZOS)

# ---- resize everything ----
rel = []
for i, r in m.iterrows():
    d = os.path.join(OUT, r["source"]); os.makedirs(d, exist_ok=True)
    name = f"{i:05d}.jpg"; prep(r["path"]).save(os.path.join(d, name), quality=95)
    rel.append(f"{r['source']}/{name}")
    if (i + 1) % 500 == 0: log("   processed", i + 1)
m["img"] = rel

# ---- patient-level folds ----
core = m["source"].isin(["ROPVL", "OSTRAVA"]).values
m["fold"] = -1; m["fold_B"] = -1
idx = np.where(core)[0]
strat = (m["source"] + "_" + m["binary"].astype(str)).values
grp = m["patient_uid"].values
for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(idx, strat[idx], grp[idx])):
    m.loc[idx[te], "fold"] = k
b = np.where((m["source"] == "ROPVL") & (m["binary"] == 1))[0]
for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(b, m["sev_ext"].values[b], grp[b])):
    m.loc[b[te], "fold_B"] = k

assert (m[core].groupby("patient_uid")["fold"].nunique() == 1).all(), "a patient is in two folds!"
log("\nTask A: images per fold (rows) by source_label (columns); fold -1 = HVDROPDB test set")
log(pd.crosstab(m["fold"], m["source"] + "_" + m["binary"].astype(str)).to_string())
log("\nTask B (ROP-VL ROP images): severity per fold_B")
log(pd.crosstab(m[m["fold_B"] >= 0]["fold_B"], m[m["fold_B"] >= 0]["sev_ext"]).to_string())

cols = ["img", "source", "country", "device", "acq_group", "grp", "patient_uid", "exam_id", "binary",
        "sev_paper", "sev_ext", "orig_label", "fold", "fold_B", "path"]
m[cols].to_csv(os.path.join(R, "processed_manifest.csv"), index=False)

# ---- montages: raw (left) and processed, 5 random images per group ----
def montage(rows, out_path, use_processed, thumb=192, n=5):
    grps = sorted(m["grp"].unique())
    W, H = thumb * n, thumb * len(grps)
    M = Image.new("RGB", (W, H), (40, 40, 40)); dr = ImageDraw.Draw(M)
    rng = random.Random(0)
    for gi, g in enumerate(grps):
        for j, i in enumerate(rng.sample(list(m.index[m["grp"] == g]), n)):
            p = os.path.join(OUT, m.at[i, "img"]) if use_processed else m.at[i, "path"]
            t = Image.open(p).convert("RGB"); t.thumbnail((thumb, thumb))
            M.paste(t, (j * thumb, gi * thumb))
        dr.text((4, gi * thumb + 4), g, fill=(255, 255, 0))
    M.save(out_path, quality=90)
montage(m, os.path.join(R, "raw_check.jpg"), False)
montage(m, os.path.join(R, "processed_check.jpg"), True)
log("\nDone. Files: processed\\, processed_manifest.csv, raw_check.jpg, processed_check.jpg")
logf.close()