# ================================================================
# TAMBAN MODEL 1 - PARENT FISH DETECTOR / COUNTING PIPELINE
# Google Colab: paste this entire file into ONE code cell and run.
#
# Dataset expected:
#   Fish_Model1_Cleaned_Training_Pool.zip
#
# What this code does:
#   1) Verifies the cleaned dataset before training.
#   2) Builds a capture-session stress test instead of a random split.
#   3) Trains YOLO26s in both session directions for internal checking.
#   4) Compares default NMS and YOLO26 NMS-free inference for counting.
#   5) Tunes confidence / NMS IoU on INTERNAL validation only.
#   6) Retrains one final parent-fish detector on all 213 legacy photos.
#   7) Saves the model, policy, graphs, metrics, and training metadata.
#   8) Optionally predicts NEW photos and extracts parent-fish geometry
#      / crowding features for later fusion with the 12-class part model.
#
# IMPORTANT SCIENTIFIC LIMITS:
#   - The 213 legacy photos are TRAINING material, not an independent test set.
#   - Session stress-test metrics are INTERNAL diagnostics, not thesis accuracy.
#   - The legacy images are already stretched to 880x880. Do not use their
#     pixel width/length ratios as anatomical truth.
#   - Geometry extracted from NEW photos is computed on the original uploaded
#     pixels, not on the network-resized tensor.
#   - Parent boxes do not by themselves prove head/body/tail ownership.
# ================================================================

# -----------------------------
# 0. INSTALL + IMPORT
# -----------------------------
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"])

import os, re, json, math, time, shutil, hashlib, zipfile, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import yaml

from ultralytics import YOLO
import ultralytics

warnings.filterwarnings("ignore")

# -----------------------------
# 1. SETTINGS
# -----------------------------
EXPECTED_ZIP_NAME = "Fish_Model1_Cleaned_Training_Pool.zip"
EXPECTED_ZIP_SHA256 = "e6d80dd910c445749dbd0915c4d6f66a31409efa1a20f4cb138f5d9eb6a84406"
EXPECTED_IMAGES = 213
EXPECTED_BOXES = 1192
EXPECTED_CLASS_NAME = "Fish"

MODEL_NAME = "yolo26s.pt"          # strong practical starting point for T4
IMGSZ = 768                         # source is 880x880; 768 keeps detail without upsampling
FOLD_MAX_EPOCHS = 100               # early stopping is enabled for stress-test folds
FOLD_PATIENCE = 20
MIN_FINAL_EPOCHS = 40
MAX_FINAL_EPOCHS = 120
MAX_DET = 30                        # GT maximum is 11; leaves room for diagnostics
SEED = 42
RUN_SESSION_STRESS_TEST = True      # recommended
RUN_FINAL_TRAINING = True

# Moderate augmentation. Avoid strong perspective/shear because they distort shape.
TRAIN_AUG = dict(
    degrees=25.0,
    translate=0.08,
    scale=0.30,
    shear=0.0,
    perspective=0.0,
    fliplr=0.50,
    flipud=0.50,
    hsv_h=0.015,
    hsv_s=0.30,
    hsv_v=0.25,
    mosaic=0.60,
    mixup=0.0,
    close_mosaic=10,
)

# Count-policy search. AP is still computed separately at low conf.
CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
NMS_IOU_GRID = [0.50, 0.60, 0.70, 0.80]

# -----------------------------
# 2. COLAB + DRIVE
# -----------------------------
try:
    from google.colab import drive, files
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    files = None

if IN_COLAB:
    drive.mount("/content/drive")

if torch.cuda.is_available():
    DEVICE = 0
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("WARNING: GPU not detected. In Colab choose Runtime > Change runtime type > T4 GPU.")

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
WORK = Path("/content/tamban_model1_work" if IN_COLAB else "./tamban_model1_work").resolve()
WORK.mkdir(parents=True, exist_ok=True)

if IN_COLAB:
    OUT = Path(f"/content/drive/MyDrive/Dried_Fish_Models/tamban_model1_parent_{stamp}")
else:
    OUT = Path(f"./tamban_model1_parent_{stamp}").resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

# Save software versions early.
(OUT / "environment.txt").write_text(
    f"created_utc={stamp}\n"
    f"python={sys.version}\n"
    f"ultralytics={ultralytics.__version__}\n"
    f"torch={torch.__version__}\n"
    f"opencv={cv2.__version__}\n"
    f"device={DEVICE}\n",
    encoding="utf-8",
)

# -----------------------------
# 3. UPLOAD / LOCATE ZIP
# -----------------------------
def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

zip_path = Path("/content") / EXPECTED_ZIP_NAME if IN_COLAB else Path(EXPECTED_ZIP_NAME)

if not zip_path.exists():
    if not IN_COLAB:
        raise FileNotFoundError(f"Place {EXPECTED_ZIP_NAME} beside this script.")
    print(f"\nUpload: {EXPECTED_ZIP_NAME}")
    uploaded = files.upload()
    if EXPECTED_ZIP_NAME not in uploaded:
        # Allow Colab to rename on duplicate upload, but find exact prefix.
        candidates = list(Path("/content").glob("Fish_Model1_Cleaned_Training_Pool*.zip"))
        if not candidates:
            raise FileNotFoundError("The cleaned Model 1 ZIP was not uploaded.")
        zip_path = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        zip_path = Path("/content") / EXPECTED_ZIP_NAME

actual_sha = sha256_file(zip_path)
print("ZIP:", zip_path)
print("SHA256:", actual_sha)
if actual_sha != EXPECTED_ZIP_SHA256:
    print("WARNING: ZIP hash differs from the exact cleaned copy created earlier.")
    print("The structural audit below will decide whether training can continue.")

# -----------------------------
# 4. EXTRACT + DISCOVER ROOT
# -----------------------------
extract_dir = WORK / "dataset_extract"
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True)
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

roots = [p.parent for p in extract_dir.rglob("cleanup_summary.json")]
roots = [p for p in roots if (p / "data.yaml").exists() and (p / "train/images").exists()]
if not roots:
    raise RuntimeError("Could not find cleaned dataset root after extraction.")
ROOT = roots[0]
print("Dataset root:", ROOT)

# -----------------------------
# 5. STRICT DATASET AUDIT
# -----------------------------
summary = json.loads((ROOT / "cleanup_summary.json").read_text())
status = json.loads((ROOT / "dataset_status.json").read_text())

def image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

imgs = image_files(ROOT / "train/images")
labels = sorted((ROOT / "train/labels").glob("*.txt"))
label_by_stem = {p.stem: p for p in labels}

errors = []
if len(imgs) != EXPECTED_IMAGES:
    errors.append(f"Expected {EXPECTED_IMAGES} train images, found {len(imgs)}")
if len(labels) != EXPECTED_IMAGES:
    errors.append(f"Expected {EXPECTED_IMAGES} train labels, found {len(labels)}")
if summary.get("retained_images") != EXPECTED_IMAGES:
    errors.append("cleanup_summary retained_images mismatch")
if status.get("task") != "single_class_fish_detection":
    errors.append("dataset_status task mismatch")

box_rows = []
missing_labels = []
for im in imgs:
    lp = label_by_stem.get(im.stem)
    if lp is None:
        missing_labels.append(im.name)
        continue
    lines = [x.strip() for x in lp.read_text().splitlines() if x.strip()]
    for row_i, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Non-detection row: {lp.name} row {row_i}")
            continue
        try:
            cls, x, y, w, h = map(float, parts)
        except Exception:
            errors.append(f"Non-numeric row: {lp.name} row {row_i}")
            continue
        if int(cls) != 0 or cls != 0:
            errors.append(f"Unexpected class id in {lp.name} row {row_i}: {cls}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"Invalid normalized box in {lp.name} row {row_i}")
        if x - w/2 < -1e-6 or x + w/2 > 1 + 1e-6 or y - h/2 < -1e-6 or y + h/2 > 1 + 1e-6:
            errors.append(f"Out-of-bounds box in {lp.name} row {row_i}")
        box_rows.append((im.name, x, y, w, h))

if missing_labels:
    errors.append(f"Missing label files for {len(missing_labels)} images")
if len(box_rows) != EXPECTED_BOXES:
    errors.append(f"Expected {EXPECTED_BOXES} boxes, found {len(box_rows)}")

if errors:
    print("\nAUDIT FAILED:")
    for e in errors[:30]:
        print(" -", e)
    raise RuntimeError("Dataset audit failed. Training stopped before modifying anything.")

counts = []
edge_count = 0
areas = []
for im in imgs:
    lp = label_by_stem[im.stem]
    rows = [x for x in lp.read_text().splitlines() if x.strip()]
    counts.append(len(rows))
    for line in rows:
        _, x, y, w, h = map(float, line.split())
        areas.append(w*h)
        if x-w/2 <= 1e-6 or x+w/2 >= 1-1e-6 or y-h/2 <= 1e-6 or y+h/2 >= 1-1e-6:
            edge_count += 1

print("\nAUDIT PASSED")
print(" Images:", len(imgs))
print(" Fish boxes:", len(box_rows))
print(" Fish/image: median", float(np.median(counts)), "range", (min(counts), max(counts)))
print(" Median normalized box area:", round(float(np.median(areas)), 4))
print(" Boxes touching image edge:", edge_count, f"({100*edge_count/len(box_rows):.1f}%)")
print(" Independent validation available:", status.get("independent_validation"))

# Save audit summary.
audit_json = {
    "zip_sha256": actual_sha,
    "images": len(imgs),
    "boxes": len(box_rows),
    "median_fish_per_image": float(np.median(counts)),
    "min_fish_per_image": int(min(counts)),
    "max_fish_per_image": int(max(counts)),
    "edge_boxes": int(edge_count),
    "edge_box_fraction": float(edge_count/len(box_rows)),
    "independent_validation": False,
    "independent_test": False,
}
(OUT / "metrics/dataset_audit.json").write_text(json.dumps(audit_json, indent=2))

# -----------------------------
# 6. SESSION-AWARE INTERNAL SPLITS
# -----------------------------
# Natural 20-minute gap in the filenames separates two capture sessions:
# early: 19:55-20:25 (~106 retained photos)
# late : 20:45-20:54 (~107 retained photos)
# This is more honest than random frame splitting, but still NOT an independent test.
TS_RE = re.compile(r"WIN_20260823_(\d{2})_(\d{2})_(\d{2})")

def get_session(path):
    m = TS_RE.search(path.name)
    if not m:
        return "unknown"
    hh, mm, ss = map(int, m.groups())
    sec = hh*3600 + mm*60 + ss
    cutoff = 20*3600 + 35*60  # between the two large capture blocks
    return "early" if sec < cutoff else "late"

sessions = {im.stem: get_session(im) for im in imgs}
print("\nSession counts:", pd.Series(list(sessions.values())).value_counts().to_dict())
if "unknown" in sessions.values():
    raise RuntimeError("Could not parse capture session for every retained image.")

split_csv = pd.DataFrame({"image": [p.name for p in imgs], "session": [sessions[p.stem] for p in imgs]})
split_csv.to_csv(OUT / "metrics/session_registry.csv", index=False)

fold_root = WORK / "session_folds"
if fold_root.exists():
    shutil.rmtree(fold_root)


def make_fold(name, train_session, val_session):
    base = fold_root / name
    for split in ["train", "val"]:
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)

    ntr = nva = 0
    for im in imgs:
        ses = sessions[im.stem]
        if ses == train_session:
            split = "train"; ntr += 1
        elif ses == val_session:
            split = "val"; nva += 1
        else:
            continue
        lp = label_by_stem[im.stem]
        shutil.copy2(im, base / split / "images" / im.name)
        shutil.copy2(lp, base / split / "labels" / lp.name)

    data = {
        "path": str(base),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": [EXPECTED_CLASS_NAME],
    }
    y = base / "data.yaml"
    y.write_text(yaml.safe_dump(data, sort_keys=False))
    return y, ntr, nva

folds = [
    ("fold_A_early_to_late", "early", "late"),
    ("fold_B_late_to_early", "late", "early"),
]
fold_info = {}
for name, tr_s, va_s in folds:
    y, ntr, nva = make_fold(name, tr_s, va_s)
    fold_info[name] = {"yaml": y, "train_n": ntr, "val_n": nva, "train_session": tr_s, "val_session": va_s}
    print(f"{name}: train={ntr} ({tr_s}), val={nva} ({va_s})")

# -----------------------------
# 7. TRAINING HELPERS
# -----------------------------
def training_kwargs(data_yaml, project, name, epochs, val=True, patience=20):
    kw = dict(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=IMGSZ,
        batch=-1,
        device=DEVICE,
        workers=2,
        cache="disk",
        pretrained=True,
        single_cls=True,
        optimizer="auto",
        patience=int(patience),
        seed=SEED,
        deterministic=True,
        amp=True,
        plots=True,
        save=True,
        save_period=10,
        max_det=MAX_DET,
        project=str(project),
        name=name,
        exist_ok=True,
        val=bool(val),
    )
    kw.update(TRAIN_AUG)
    return kw


def best_epoch_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    metric_cols = [c for c in df.columns if "mAP50-95" in c]
    if metric_cols:
        col = metric_cols[0]
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            idx = vals.idxmax()
            return int(df.loc[idx, "epoch"]) + 1, col, float(vals.loc[idx])
    return int(len(df)), None, None

# -----------------------------
# 8. COUNT POLICY EVALUATION
# -----------------------------
def gt_count_for_image(image_path, val_image_dir, val_label_dir):
    lp = val_label_dir / (Path(image_path).stem + ".txt")
    return len([x for x in lp.read_text().splitlines() if x.strip()])


def metrics_from_counts(gt, pred):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - gt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "exact_accuracy": float(np.mean(pred == gt)),
        "within1_accuracy": float(np.mean(np.abs(err) <= 1)),
    }


def count_policy_grid(weight_path, fold_yaml, fold_name):
    y = yaml.safe_load(Path(fold_yaml).read_text())
    base = Path(y["path"])
    val_img_dir = base / y["val"]
    val_label_dir = base / "val/labels"
    val_paths = image_files(val_img_dir)
    gt = {p.stem: gt_count_for_image(p, val_img_dir, val_label_dir) for p in val_paths}

    model = YOLO(str(weight_path))
    rows = []

    # A) default one-to-many head + NMS. Run at low confidence once per NMS IoU,
    # then threshold returned confidences offline.
    for nms_iou in NMS_IOU_GRID:
        results = model.predict(
            source=[str(p) for p in val_paths],
            imgsz=IMGSZ,
            conf=0.01,
            iou=nms_iou,
            max_det=MAX_DET,
            device=DEVICE,
            nms=None,
            verbose=False,
        )
        conf_by_stem = {Path(r.path).stem: r.boxes.conf.detach().cpu().numpy() for r in results}
        for thr in CONF_GRID:
            gts, prs = [], []
            for p in val_paths:
                gts.append(gt[p.stem])
                prs.append(int(np.sum(conf_by_stem[p.stem] >= thr)))
            m = metrics_from_counts(gts, prs)
            rows.append({
                "fold": fold_name,
                "head": "one_to_many_nms",
                "nms": "default",
                "iou": float(nms_iou),
                "conf": float(thr),
                **m,
            })

    # B) YOLO26 one-to-one NMS-free head. IoU is not a suppression threshold here.
    results = model.predict(
        source=[str(p) for p in val_paths],
        imgsz=IMGSZ,
        conf=0.01,
        max_det=MAX_DET,
        device=DEVICE,
        nms=False,
        verbose=False,
    )
    conf_by_stem = {Path(r.path).stem: r.boxes.conf.detach().cpu().numpy() for r in results}
    for thr in CONF_GRID:
        gts, prs = [], []
        for p in val_paths:
            gts.append(gt[p.stem])
            prs.append(int(np.sum(conf_by_stem[p.stem] >= thr)))
        m = metrics_from_counts(gts, prs)
        rows.append({
            "fold": fold_name,
            "head": "one_to_one_nms_free",
            "nms": "false",
            "iou": np.nan,
            "conf": float(thr),
            **m,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"metrics/{fold_name}_count_policy_grid.csv", index=False)
    return df

# -----------------------------
# 9. RUN TWO SESSION STRESS TESTS
# -----------------------------
fold_summaries = []
all_policy_rows = []
fold_best_epochs = []

if RUN_SESSION_STRESS_TEST:
    for fold_name, tr_s, va_s in folds:
        info = fold_info[fold_name]
        print("\n" + "="*72)
        print("TRAINING", fold_name)
        print("="*72)

        model = YOLO(MODEL_NAME)
        model.train(**training_kwargs(
            info["yaml"],
            OUT / "runs",
            fold_name,
            FOLD_MAX_EPOCHS,
            val=True,
            patience=FOLD_PATIENCE,
        ))
        run_dir = Path(model.trainer.save_dir)
        best_pt = run_dir / "weights/best.pt"
        last_pt = run_dir / "weights/last.pt"
        if not best_pt.exists():
            best_pt = last_pt

        csv_path = run_dir / "results.csv"
        best_epoch, metric_col, best_metric = best_epoch_from_csv(csv_path)
        fold_best_epochs.append(best_epoch)

        eval_model = YOLO(str(best_pt))
        # Default one-to-many + NMS localization metrics.
        m_nms = eval_model.val(
            data=str(info["yaml"]),
            split="val",
            imgsz=IMGSZ,
            batch=8,
            device=DEVICE,
            conf=0.001,
            max_det=MAX_DET,
            nms=None,
            plots=True,
            project=str(OUT / "validation"),
            name=fold_name + "_default_nms",
            exist_ok=True,
            verbose=False,
        )
        # NMS-free head is evaluated separately because crowded fish may behave differently.
        m_e2e = eval_model.val(
            data=str(info["yaml"]),
            split="val",
            imgsz=IMGSZ,
            batch=8,
            device=DEVICE,
            conf=0.001,
            max_det=MAX_DET,
            nms=False,
            plots=False,
            project=str(OUT / "validation"),
            name=fold_name + "_nms_free",
            exist_ok=True,
            verbose=False,
        )

        summary_row = {
            "fold": fold_name,
            "train_session": tr_s,
            "val_session": va_s,
            "train_images": info["train_n"],
            "val_images": info["val_n"],
            "best_epoch_internal": best_epoch,
            "default_precision": float(m_nms.box.mp),
            "default_recall": float(m_nms.box.mr),
            "default_map50": float(m_nms.box.map50),
            "default_map50_95": float(m_nms.box.map),
            "nmsfree_precision": float(m_e2e.box.mp),
            "nmsfree_recall": float(m_e2e.box.mr),
            "nmsfree_map50": float(m_e2e.box.map50),
            "nmsfree_map50_95": float(m_e2e.box.map),
            "checkpoint": str(best_pt),
        }
        fold_summaries.append(summary_row)

        grid = count_policy_grid(best_pt, info["yaml"], fold_name)
        all_policy_rows.append(grid)

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(OUT / "metrics/session_stress_test_detection_metrics.csv", index=False)

    policy_df = pd.concat(all_policy_rows, ignore_index=True)
    # Create a stable key so policies can be averaged across the two folds.
    policy_df["iou_key"] = policy_df["iou"].fillna(-1.0)
    agg = (policy_df.groupby(["head", "nms", "iou_key", "conf"], as_index=False)
           .agg(mean_mae=("mae", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_bias=("bias", "mean"),
                mean_exact_accuracy=("exact_accuracy", "mean"),
                mean_within1_accuracy=("within1_accuracy", "mean")))
    agg["mean_abs_bias"] = agg["mean_bias"].abs()
    agg = agg.sort_values(
        ["mean_mae", "mean_abs_bias", "mean_exact_accuracy", "mean_within1_accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    agg.to_csv(OUT / "metrics/combined_count_policy_ranking.csv", index=False)
    best_policy = agg.iloc[0].to_dict()

    print("\nBest INTERNAL count policy across the two session directions:")
    print(best_policy)

    # --- graphs ---
    plt.figure(figsize=(8, 5))
    x = np.arange(len(fold_df))
    plt.bar(x-0.18, fold_df["default_map50_95"], 0.36, label="Default NMS")
    plt.bar(x+0.18, fold_df["nmsfree_map50_95"], 0.36, label="NMS-free")
    plt.xticks(x, ["Early -> Late", "Late -> Early"])
    plt.ylim(0, 1)
    plt.ylabel("Internal mAP50-95")
    plt.title("Capture-session stress test (NOT final test accuracy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figures/session_stress_map50_95.png", dpi=200)
    plt.show(); plt.close()

    # Plot MAE vs confidence for best default-NMS IoU and NMS-free head.
    plt.figure(figsize=(8, 5))
    for head in agg["head"].unique():
        d = agg[agg["head"] == head].copy()
        if head == "one_to_many_nms":
            # choose the IoU with the best average MAE for clarity
            iou_scores = d.groupby("iou_key")["mean_mae"].min()
            best_iou = iou_scores.idxmin()
            d = d[d["iou_key"] == best_iou]
            label = f"NMS (IoU={best_iou:.2f})"
        else:
            label = "NMS-free"
        d = d.sort_values("conf")
        plt.plot(d["conf"], d["mean_mae"], marker="o", label=label)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Mean count MAE across session directions")
    plt.title("Internal counting threshold sensitivity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figures/count_mae_vs_confidence.png", dpi=200)
    plt.show(); plt.close()

else:
    print("Session stress test skipped by setting.")
    best_policy = {
        "head": "one_to_many_nms",
        "nms": "default",
        "iou_key": 0.70,
        "conf": 0.25,
        "note": "Fallback only; not tuned because stress test was disabled.",
    }

# -----------------------------
# 10. FINAL TRAINING ON ALL 213 LEGACY PHOTOS
# -----------------------------
# The stress-test folds are for internal diagnostics only. The final parent detector
# uses all screened legacy photos. It has no independent validation metric.
if fold_best_epochs:
    final_epochs = int(np.clip(round(float(np.median(fold_best_epochs))), MIN_FINAL_EPOCHS, MAX_FINAL_EPOCHS))
else:
    final_epochs = 80

print("\nFinal all-data epoch count:", final_epochs)

# Root data.yaml has an empty val folder. We disable validation for final all-data training.
final_checkpoint = None
if RUN_FINAL_TRAINING:
    print("\n" + "="*72)
    print("FINAL TRAINING ON ALL 213 SCREENED LEGACY IMAGES")
    print("No final-test metric will be claimed from this run.")
    print("="*72)

    final_model = YOLO(MODEL_NAME)
    final_model.train(**training_kwargs(
        ROOT / "data.yaml",
        OUT / "runs",
        "final_all_legacy",
        final_epochs,
        val=False,
        patience=0,
    ))
    final_run = Path(final_model.trainer.save_dir)
    last_pt = final_run / "weights/last.pt"
    if not last_pt.exists():
        raise RuntimeError("Final last.pt was not created.")
    final_checkpoint = OUT / "model1_fish_parent_detector.pt"
    shutil.copy2(last_pt, final_checkpoint)
    print("Final checkpoint:", final_checkpoint)
else:
    print("Final training skipped by setting.")

# -----------------------------
# 11. SAVE DEPLOYMENT / COUNT POLICY
# -----------------------------
policy = {
    "created_utc": stamp,
    "model_role": "parent_fish_detection_and_counting",
    "class_map": {"0": "Fish"},
    "model_architecture_start": MODEL_NAME,
    "imgsz": IMGSZ,
    "max_det": MAX_DET,
    "confidence": float(best_policy.get("conf", 0.25)),
    "inference_head": str(best_policy.get("head", "one_to_many_nms")),
    "nms": False if str(best_policy.get("head")) == "one_to_one_nms_free" else None,
    "nms_iou": None if str(best_policy.get("head")) == "one_to_one_nms_free" else float(best_policy.get("iou_key", 0.70)),
    "final_epochs": final_epochs,
    "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
    "policy_source": "two-direction capture-session internal stress test" if RUN_SESSION_STRESS_TEST else "untuned fallback",
    "independent_validation": False,
    "independent_test": False,
    "warning": "Do not report the internal session metrics as final thesis accuracy. Obtain a genuinely new specimen/session validation/test set.",
}
(OUT / "model1_policy.json").write_text(json.dumps(policy, indent=2))

# -----------------------------
# 12. GEOMETRY / CROWDING FEATURE ENGINE FOR NEW PHOTOS
# -----------------------------
def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    aa = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    return inter / (aa + bb - inter + 1e-9)


def estimate_pixel_geometry(image_bgr, box_xyxy):
    """Experimental foreground geometry from ORIGINAL image pixels.

    It estimates local background color from a padded crop, segments pixels that
    differ from that background, keeps the largest component, and computes PCA-based
    orientation and width profile. It deliberately returns geometry_reliable=False
    when quality checks fail. This is supporting evidence, not ground truth.
    """
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box_xyxy)
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    pad_x, pad_y = 0.10*bw, 0.10*bh
    cx1 = int(max(0, math.floor(x1-pad_x)))
    cy1 = int(max(0, math.floor(y1-pad_y)))
    cx2 = int(min(W, math.ceil(x2+pad_x)))
    cy2 = int(min(H, math.ceil(y2+pad_y)))
    crop = image_bgr[cy1:cy2, cx1:cx2]

    out = {
        "geometry_reliable": False,
        "orientation_axis_deg": np.nan,
        "pca_length_px": np.nan,
        "pca_width_median_px": np.nan,
        "pca_width_max_px": np.nan,
        "width_length_ratio": np.nan,
        "foreground_area_length2": np.nan,
        "solidity": np.nan,
        "foreground_fraction": np.nan,
        "component_dominance": np.nan,
        "width_profile": None,
    }
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return out

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = crop.shape[:2]
    border_t = max(2, int(round(min(h,w)*0.05)))
    border_mask = np.zeros((h,w), np.uint8)
    border_mask[:border_t,:] = 1
    border_mask[-border_t:,:] = 1
    border_mask[:,:border_t] = 1
    border_mask[:,-border_t:] = 1
    bg_pixels = lab[border_mask.astype(bool)]
    if len(bg_pixels) < 20:
        return out
    bg = np.median(bg_pixels, axis=0)
    dist = np.linalg.norm(lab-bg, axis=2)
    dmax = np.percentile(dist, 99)
    if dmax <= 1e-6:
        return out
    d8 = np.clip(dist/dmax*255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, int(round(min(h,w)*0.015)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, cc, stats, cent = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), 8)
    if n <= 1:
        return out
    areas_cc = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas_cc))
    comp = (cc == idx).astype(np.uint8)
    comp_area = int(comp.sum())
    total_fg = int((mask>0).sum())
    fg_frac = comp_area / float(h*w)
    dominance = comp_area / float(total_fg + 1e-9)
    out["foreground_fraction"] = float(fg_frac)
    out["component_dominance"] = float(dominance)

    ys, xs = np.where(comp > 0)
    if len(xs) < 80:
        return out
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    major = vecs[:,0]
    minor = vecs[:,1]
    u = centered @ major
    v = centered @ minor
    u1, u99 = np.percentile(u, [1,99])
    length = float(max(1e-6, u99-u1))

    # Width profile in 10 longitudinal bins, using robust transverse percentiles.
    edges = np.linspace(u1, u99, 11)
    widths = []
    for i in range(10):
        sel = (u >= edges[i]) & (u < edges[i+1] if i < 9 else u <= edges[i+1])
        vv = v[sel]
        if len(vv) >= 10:
            lo, hi = np.percentile(vv, [5,95])
            widths.append(float(max(0.0, hi-lo)))
        else:
            widths.append(np.nan)
    valid_w = np.asarray([x for x in widths if np.isfinite(x) and x > 0], dtype=float)
    if len(valid_w) < 5:
        return out

    hull = cv2.convexHull(pts.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    solidity = float(comp_area / (hull_area + 1e-9)) if hull_area > 0 else np.nan
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    med_w = float(np.median(valid_w))
    max_w = float(np.max(valid_w))

    out.update({
        "orientation_axis_deg": angle,
        "pca_length_px": length,
        "pca_width_median_px": med_w,
        "pca_width_max_px": max_w,
        "width_length_ratio": med_w/length,
        "foreground_area_length2": float(comp_area/(length*length + 1e-9)),
        "solidity": solidity,
        "width_profile": [None if not np.isfinite(x) else float(x/length) for x in widths],
    })

    frame_clipped = x1 <= 2 or y1 <= 2 or x2 >= W-2 or y2 >= H-2
    # Conservative quality gate. Overlap with another fish is checked outside this function.
    good = (
        (not frame_clipped)
        and 0.08 <= fg_frac <= 0.85
        and dominance >= 0.60
        and 0.45 <= solidity <= 1.20
        and length >= 20
    )
    out["geometry_reliable"] = bool(good)
    return out


def extract_parent_features(image_bgr, boxes_xyxy, confs):
    H, W = image_bgr.shape[:2]
    boxes = np.asarray(boxes_xyxy, dtype=float)
    rows = []
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box
        bw, bh = max(0.0,x2-x1), max(0.0,y2-y1)
        ious = [box_iou_xyxy(box, boxes[j]) for j in range(len(boxes)) if j != i]
        max_iou = max(ious) if ious else 0.0
        overlap_neighbors = sum(v > 0.05 for v in ious)
        frame_clipped = bool(x1 <= 2 or y1 <= 2 or x2 >= W-2 or y2 >= H-2)
        geom = estimate_pixel_geometry(image_bgr, box)
        # Geometry is more suspect when parent boxes overlap strongly.
        if max_iou > 0.20:
            geom["geometry_reliable"] = False
        row = {
            "fish_id": i+1,
            "det_conf": float(confs[i]),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "center_x_norm": float(((x1+x2)/2)/W),
            "center_y_norm": float(((y1+y2)/2)/H),
            "bbox_width_norm": float(bw/W),
            "bbox_height_norm": float(bh/H),
            "bbox_area_fraction": float((bw*bh)/(W*H + 1e-9)),
            "bbox_long_short_ratio": float(max(bw,bh)/(min(bw,bh)+1e-9)),
            "frame_clipped": frame_clipped,
            "max_parent_iou": float(max_iou),
            "overlap_neighbors_iou_gt_0_05": int(overlap_neighbors),
            **geom,
        }
        rows.append(row)
    return rows

# -----------------------------
# 13. OPTIONAL NEW-PHOTO DEMO
# -----------------------------
if final_checkpoint and IN_COLAB:
    answer = input("\nInspect genuinely NEW photos with Model 1 now? y/N: ").strip().lower()
    if answer == "y":
        print("Upload new photos. Do not use these results as final accuracy unless they have independent reviewed ground truth.")
        uploaded = files.upload()
        new_paths = []
        for name in uploaded:
            p = Path("/content") / name
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                new_paths.append(p)
        if not new_paths:
            print("No supported image files uploaded.")
        else:
            pred_dir = OUT / "new_photo_predictions"
            pred_dir.mkdir(exist_ok=True)
            feature_rows = []
            parent_json = []

            pred_model = YOLO(str(final_checkpoint))
            nms_arg = policy["nms"]
            pred_kwargs = dict(
                source=[str(p) for p in new_paths],
                imgsz=IMGSZ,
                conf=policy["confidence"],
                max_det=MAX_DET,
                device=DEVICE,
                nms=nms_arg,
                verbose=False,
            )
            if nms_arg is not False and policy["nms_iou"] is not None:
                pred_kwargs["iou"] = policy["nms_iou"]
            results = pred_model.predict(**pred_kwargs)

            for r in results:
                image_path = Path(r.path)
                im = cv2.imread(str(image_path))
                boxes = r.boxes.xyxy.detach().cpu().numpy() if r.boxes is not None else np.zeros((0,4))
                confs = r.boxes.conf.detach().cpu().numpy() if r.boxes is not None else np.zeros((0,))

                # Stable display IDs by top-to-bottom then left-to-right box centers.
                if len(boxes):
                    centers = np.column_stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2])
                    order = np.lexsort((centers[:,0], centers[:,1]))
                    boxes, confs = boxes[order], confs[order]

                rows = extract_parent_features(im, boxes, confs)
                for row in rows:
                    row["image"] = image_path.name
                    row["legacy_square_geometry_warning"] = bool(im.shape[0] == 880 and im.shape[1] == 880)
                    feature_rows.append(row)

                parent_json.append({
                    "image": image_path.name,
                    "fish_count": int(len(boxes)),
                    "fish": rows,
                })

                # Draw parent boxes + stable IDs.
                vis = im.copy()
                for row in rows:
                    x1,y1,x2,y2 = [int(round(row[k])) for k in ["x1","y1","x2","y2"]]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
                    label = f"Fish {row['fish_id']} {row['det_conf']:.2f}"
                    cv2.putText(vis, label, (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                out_img = pred_dir / f"{image_path.stem}_parents.jpg"
                cv2.imwrite(str(out_img), vis)
                print(image_path.name, "->", len(boxes), "Fish detections")

            pd.DataFrame(feature_rows).to_csv(pred_dir / "parent_fish_features.csv", index=False)
            (pred_dir / "parent_fish_predictions.json").write_text(json.dumps(parent_json, indent=2))
            print("Prediction outputs:", pred_dir)

# -----------------------------
# 14. FINAL README
# -----------------------------
readme = f"""# Tamban Model 1 - Parent Fish Detector\n\nCreated: {stamp}\n\nDataset: Fish_Model1_Cleaned_Training_Pool.zip\nVerified legacy training photos: {EXPECTED_IMAGES}\nVerified active Fish boxes: {EXPECTED_BOXES}\nStarting model: {MODEL_NAME}\nImage size: {IMGSZ}\nFinal training epochs: {final_epochs}\nFinal checkpoint: {final_checkpoint}\n\n## Interpretation\n- The two session directions are INTERNAL stress tests only.\n- The final all-data checkpoint has no independent validation/test accuracy.\n- Do not report training curves or internal session metrics as final thesis accuracy.\n- Get genuinely new reviewed specimen/session data for final parent detection, counting and end-to-end grading metrics.\n- Parent Fish boxes are intended for counting, cropping and part-to-fish association. They do not alone prove part ownership.\n- Experimental pixel geometry is quality-gated and should only be used as supporting features until validated.\n- Legacy 880x880 stretched images must not be used for anatomical width/length conclusions.\n"""
(OUT / "README_MODEL1.md").write_text(readme, encoding="utf-8")

print("\n" + "="*72)
print("MODEL 1 PIPELINE COMPLETE")
print("Output folder:", OUT)
if final_checkpoint:
    print("Final parent detector:", final_checkpoint)
print("Policy:", OUT / "model1_policy.json")
print("Metrics:", OUT / "metrics")
print("Figures:", OUT / "figures")
print("="*72)
