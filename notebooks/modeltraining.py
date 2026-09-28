# ================================================================
# TAMBAN MODEL 2 — THREE-MODEL TRAINING + COMPARISON (SINGLE CELL)
# Revision: r03
# Models:
#   1) Ultralytics YOLO26m detector
#   2) Faster R-CNN ResNet50-FPN V2 detector
#   3) EfficientNetV2-M + FPN + RetinaNet detector
#
# MODEL 2 ONLY. Model 1 is NOT trained or modified here.
#
# IMPORTANT SCIENTIFIC NOTE:
# The current cleaned pool contains 846 images but does NOT contain a
# truly specimen-independent thesis test set. This cell therefore creates
# an INTERNAL DEVELOPMENT split for engineering/model selection only.
# Final thesis metrics must be reported later on newly captured,
# specimen-independent validation/test images.
# ================================================================

# -----------------------------
# 0. INSTALL / IMPORT
# -----------------------------
import os, sys, re, math, json, time, random, shutil, zipfile, subprocess, warnings
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
warnings.filterwarnings('ignore')

PKGS = [
    'ultralytics>=8.4.0',
    'torchmetrics>=1.6.0',
    'pycocotools>=2.0.7',
    'albumentations>=1.4.20',
    'opencv-python-headless>=4.10.0.84',
    'scikit-learn>=1.4.0',
    'pandas>=2.0.0',
    'matplotlib>=3.8.0',
    'pyyaml>=6.0.0'
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', *PKGS])

import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import (
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
)
from ultralytics import YOLO

try:
    from google.colab import drive, files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

print('=' * 84)
print('TAMBAN MODEL 2 — 3-MODEL TRAINING / COMPARISON')
print('=' * 84)
print('Python      :', sys.version.split()[0])
print('PyTorch     :', torch.__version__)
print('Torchvision :', torchvision.__version__)
try:
    import ultralytics
    print('Ultralytics :', ultralytics.__version__)
except Exception:
    pass
print('CUDA        :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU         :', torch.cuda.get_device_name(0))
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM        : {VRAM_GB:.1f} GB')
else:
    VRAM_GB = 0
    print('\nERROR: GPU is not enabled. In Colab select Runtime > Change runtime type > GPU, then rerun.')
    raise RuntimeError('GPU required for practical training of all three detectors.')

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

CLASS_NAMES = [
    'Grade_A_Body','Grade_A_Head','Grade_A_Tail',
    'Grade_B_Body','Grade_B_Head','Grade_B_Tail',
    'Grade_C_Body','Grade_C_Head','Grade_C_Tail',
    'Rejected_Body','Rejected_Head','Rejected_Tail'
]
NC = len(CLASS_NAMES)
REVISION = 'r03'

# --- Main training knobs ---
YOLO_MODEL = 'yolo26m.pt'
YOLO_EPOCHS = 70
YOLO_IMGSZ = 768
YOLO_PATIENCE = 18

RCNN_EPOCHS = 40
EFFNET_EPOCHS = 40
EVAL_EVERY = 2
EARLY_STOP_EVALS = 6            # 6 evals x every 2 epochs = ~12 epochs with no improvement
MIN_SIZE = 768
MAX_SIZE = 1024
DEV_FRACTION = 0.20

# EfficientNetV2 family is the current EfficientNetV2 generation in torchvision.
# M is realistic on a typical 16 GB Colab GPU. Change to 'l' on >=30 GB VRAM if desired.
EFFNET_VARIANT = 'm'
if EFFNET_VARIANT == 'l' and VRAM_GB < 28:
    print('WARNING: EfficientNetV2-L is very large. Falling back to V2-M for this GPU.')
    EFFNET_VARIANT = 'm'

# Torch detector batch sizing / accumulation
if VRAM_GB >= 30:
    RCNN_BATCH, EFF_BATCH = 4, 2
elif VRAM_GB >= 14:
    RCNN_BATCH, EFF_BATCH = 2, 1
else:
    RCNN_BATCH, EFF_BATCH = 1, 1
TARGET_EFFECTIVE_BATCH = 4
NUM_WORKERS = 2
AMP = True

# -----------------------------
# 2. GOOGLE DRIVE + DATASET LOCATION
# -----------------------------
if IN_COLAB:
    drive.mount('/content/drive', force_remount=False)

DEFAULT_ZIP = Path('/content/drive/MyDrive/Dried_Fish_Models/Tamban_v9_Cleaned_Training_Pool.zip')
DATASET_ZIP = DEFAULT_ZIP

if not DATASET_ZIP.exists():
    print('\nDataset ZIP was not found at:')
    print(DATASET_ZIP)
    print('\nUpload Tamban_v9_Cleaned_Training_Pool.zip now...')
    if not IN_COLAB:
        raise FileNotFoundError('Set DATASET_ZIP to the cleaned Model 2 ZIP.')
    uploaded = files.upload()
    zips = [Path('/content') / n for n in uploaded if n.lower().endswith('.zip')]
    if not zips:
        raise FileNotFoundError('No ZIP uploaded.')
    DATASET_ZIP = zips[0]

OUT_ROOT = Path('/content/drive/MyDrive/Dried_Fish_Models/model2_r03_three_model_comparison')
WORK = Path('/content/tamban_model2_r03')
OUT_ROOT.mkdir(parents=True, exist_ok=True)
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

print('\nDataset:', DATASET_ZIP)
print('Outputs:', OUT_ROOT)

# -----------------------------
# 3. EXTRACT + VERIFY CLEANED MODEL 2 DATASET
# -----------------------------
EXTRACT = WORK / 'extracted'
EXTRACT.mkdir(parents=True)
with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
    z.extractall(EXTRACT)

roots = [p for p in EXTRACT.rglob('data.yaml')]
if not roots:
    raise FileNotFoundError('Could not find data.yaml inside the ZIP.')
DATA_ROOT = roots[0].parent
IMG_DIR = DATA_ROOT / 'train' / 'images'
LBL_DIR = DATA_ROOT / 'train' / 'labels'
if not IMG_DIR.exists() or not LBL_DIR.exists():
    raise FileNotFoundError('Expected train/images and train/labels in cleaned Model 2 pool.')

with open(DATA_ROOT/'data.yaml', 'r') as f:
    original_yaml = yaml.safe_load(f)
found_names = original_yaml.get('names', [])
if isinstance(found_names, dict):
    found_names = [found_names[i] for i in sorted(found_names)]
if list(found_names) != CLASS_NAMES:
    raise ValueError(f'Class mapping mismatch. Found: {found_names}')

image_exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in image_exts])
if len(images) != 846:
    print(f'WARNING: expected 846 cleaned pool images, found {len(images)}. Continuing after audit.')

# Parse YOLO 5-column labels and collect data audit
records = []
obj_counts = Counter()
invalid_rows = []
for ip in images:
    lp = LBL_DIR / f'{ip.stem}.txt'
    if not lp.exists():
        invalid_rows.append((ip.name, 'missing_label'))
        continue
    rows = []
    txt = lp.read_text().strip()
    if txt:
        for line_no, line in enumerate(txt.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                invalid_rows.append((ip.name, f'line {line_no}: expected 5 values, got {len(parts)}'))
                continue
            c, x, y, w, h = map(float, parts)
            c = int(c)
            if c < 0 or c >= NC or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid_rows.append((ip.name, f'line {line_no}: invalid values'))
                continue
            rows.append((c,x,y,w,h)); obj_counts[c] += 1
    records.append({'image_path':str(ip), 'label_path':str(lp), 'classes':sorted(set(r[0] for r in rows)), 'n_objects':len(rows)})

if invalid_rows:
    print('First annotation issues:', invalid_rows[:10])
    raise ValueError(f'Found {len(invalid_rows)} invalid annotation rows. Stop rather than train corrupted labels.')

df = pd.DataFrame(records)
print(f'\nVerified images: {len(df)}')
print('Object count   :', sum(obj_counts.values()))
print('\nClass counts:')
for i,n in enumerate(CLASS_NAMES):
    print(f'{i:2d} {n:20s} {obj_counts[i]:5d}')

# -----------------------------
# 4. GROUP-AWARE INTERNAL DEVELOPMENT SPLIT
# -----------------------------
# This attempts to keep near-time captures together. It is deliberately called a
# DEVELOPMENT split because physical specimen IDs are not available in the cleaned pool.
def make_group(stem):
    base = re.sub(r'\.rf\.[0-9a-fA-F]+$', '', stem)
    # WIN_20260823_20_24_38_Pro_jpg -> 2-minute capture bin
    m = re.search(r'WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})', base)
    if m:
        date, hh, mm, ss = m.groups()
        sec = int(hh)*3600 + int(mm)*60 + int(ss)
        return f'{date}_bin{sec//120:04d}'
    # Non-WIN images: use original source stem (Roboflow hash already removed)
    return base

df['group'] = [make_group(Path(p).stem) for p in df.image_path]

# Try many GroupShuffleSplit candidates; select the one that most closely preserves
# all 12 class object distributions while keeping groups separated.
class_matrix = np.zeros((len(df), NC), dtype=int)
for idx, lp in enumerate(df.label_path):
    for line in Path(lp).read_text().splitlines():
        if line.strip():
            class_matrix[idx, int(line.split()[0])] += 1

total_per_class = class_matrix.sum(0)
best = None
for rs in range(800):
    gss = GroupShuffleSplit(n_splits=1, test_size=DEV_FRACTION, random_state=rs)
    tr, dv = next(gss.split(df, groups=df.group))
    trc, dvc = class_matrix[tr].sum(0), class_matrix[dv].sum(0)
    if np.any(trc == 0) or np.any(dvc == 0):
        continue
    ratio = dvc / np.maximum(total_per_class, 1)
    score = float(np.mean(np.abs(ratio - DEV_FRACTION))) + 0.5*abs(len(dv)/len(df)-DEV_FRACTION)
    if best is None or score < best[0]:
        best = (score, tr, dv, rs)
if best is None:
    raise RuntimeError('Could not create a group-aware split containing all 12 classes.')
_, tr_idx, dv_idx, split_seed = best
df['split'] = 'train'
df.loc[dv_idx, 'split'] = 'dev'

split_csv = OUT_ROOT / 'model2_r03_internal_development_split.csv'
df[['image_path','label_path','group','split','n_objects']].to_csv(split_csv, index=False)

print('\nInternal development split (NOT final thesis test):')
print('Train images:', int((df.split=='train').sum()))
print('Dev images  :', int((df.split=='dev').sum()))
print('Groups train:', df[df.split=='train'].group.nunique())
print('Groups dev  :', df[df.split=='dev'].group.nunique())
print('Split seed  :', split_seed)
assert set(df[df.split=='train'].group).isdisjoint(set(df[df.split=='dev'].group))

# Create YOLO train/dev image lists using the exact same split used by the torch detectors.
train_txt = WORK/'train.txt'; dev_txt = WORK/'dev.txt'
train_txt.write_text('\n'.join(df.loc[df.split=='train','image_path'])+'\n')
dev_txt.write_text('\n'.join(df.loc[df.split=='dev','image_path'])+'\n')
yolo_yaml = WORK/'model2_r03_dev.yaml'
yolo_yaml.write_text(yaml.safe_dump({
    'path': str(DATA_ROOT),
    'train': str(train_txt),
    'val': str(dev_txt),
    'nc': NC,
    'names': CLASS_NAMES
}, sort_keys=False))

# -----------------------------
# 5. DATASET + SAFE AUGMENTATION FOR QUALITY PARTS
# -----------------------------
# We intentionally avoid strong hue/saturation/value augmentation in the torch
# models because B vs C may depend on real color/reflectance/brightness cues.
import albumentations as A

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.45),
    A.Affine(scale=(0.92,1.08), translate_percent=(-0.04,0.04), rotate=(-12,12), shear=(-3,3), p=0.35),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.20, clip=True))

class YoloPartDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = cv2.imread(row.image_path)
        if img is None: raise FileNotFoundError(row.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H,W = img.shape[:2]
        boxes=[]; labels=[]
        for line in Path(row.label_path).read_text().splitlines():
            if not line.strip(): continue
            c,x,y,w,h = map(float,line.split())
            x1=(x-w/2)*W; y1=(y-h/2)*H; x2=(x+w/2)*W; y2=(y+h/2)*H
            x1=max(0,min(W-1,x1)); y1=max(0,min(H-1,y1)); x2=max(1,min(W,x2)); y2=max(1,min(H,y2))
            if x2>x1 and y2>y1:
                boxes.append([x1,y1,x2,y2]); labels.append(int(c)+1)  # +1 reserves class 0 as background
        if self.augment and boxes:
            out = TRAIN_AUG(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = out['image'], list(out['bboxes']), list(out['labels'])
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()/255.0
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1,4)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        area = ((boxes_t[:,2]-boxes_t[:,0])*(boxes_t[:,3]-boxes_t[:,1])) if len(boxes_t) else torch.zeros((0,),dtype=torch.float32)
        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([idx],dtype=torch.int64),
            'area': area,
            'iscrowd': torch.zeros((len(boxes_t),),dtype=torch.int64)
        }
        return tensor, target

def collate_fn(batch): return tuple(zip(*batch))

train_df = df[df.split=='train'].copy(); dev_df = df[df.split=='dev'].copy()
train_ds = YoloPartDataset(train_df, augment=True)
dev_ds = YoloPartDataset(dev_df, augment=False)

# Mild rare-class-aware image sampling for R-CNN/EfficientNet.
# Use inverse-square-root rather than extreme inverse frequency and cap the weight.
train_obj_counts = np.zeros(NC, dtype=float)
for lp in train_df.label_path:
    for line in Path(lp).read_text().splitlines():
        if line.strip(): train_obj_counts[int(line.split()[0])] += 1
inv = np.sqrt(train_obj_counts.max()/np.maximum(train_obj_counts,1.0))
weights=[]
for classes in train_df.classes:
    w = max([inv[c] for c in classes], default=1.0)
    weights.append(min(float(w), 3.0))
weights = torch.tensor(weights,dtype=torch.double)

# -----------------------------
# 6. COMMON EVALUATION
# -----------------------------
def box_iou_np(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),dtype=np.float32)
    a=np.asarray(a,float); b=np.asarray(b,float)
    tl=np.maximum(a[:,None,:2],b[None,:,:2]); br=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.clip(br-tl,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip(a[:,2]-a[:,0],0,None)*np.clip(a[:,3]-a[:,1],0,None)
    bb=np.clip(b[:,2]-b[:,0],0,None)*np.clip(b[:,3]-b[:,1],0,None)
    return inter/(aa[:,None]+bb[None,:]-inter+1e-9)

def prf_at_threshold(preds, tgts, conf_thr, iou_thr=0.5):
    tp=fp=fn=0
    for p,t in zip(preds,tgts):
        pb=p['boxes'].cpu().numpy(); ps=p['scores'].cpu().numpy(); pl=p['labels'].cpu().numpy()
        gb=t['boxes'].cpu().numpy(); gl=t['labels'].cpu().numpy()
        keep=np.where(ps>=conf_thr)[0]; keep=keep[np.argsort(-ps[keep])]
        used=set()
        for j in keep:
            candidates=[k for k in range(len(gb)) if k not in used and gl[k]==pl[j]]
            if not candidates:
                fp+=1; continue
            ious=box_iou_np(pb[j:j+1],gb[candidates])[0]
            k_local=int(np.argmax(ious)); best_gt=candidates[k_local]
            if ious[k_local]>=iou_thr:
                tp+=1; used.add(best_gt)
            else: fp+=1
        fn += len(gb)-len(used)
    precision=tp/(tp+fp+1e-9); recall=tp/(tp+fn+1e-9)
    f1=2*precision*recall/(precision+recall+1e-9)
    return precision,recall,f1,tp,fp,fn

def summarize_detection_lists(preds,tgts):
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    metric50 = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.5], class_metrics=True)
    metric.update(preds,tgts); metric50.update(preds,tgts)
    a=metric.compute(); b=metric50.compute()
    best_thr,best=(None,(-1,-1,-1,0,0,0))
    for thr in np.arange(0.05,0.76,0.05):
        cur=prf_at_threshold(preds,tgts,float(thr),0.5)
        if cur[2]>best[2]: best_thr,best=float(thr),cur
    ap50_pc=np.full(NC,np.nan,float)
    classes=b.get('classes',torch.tensor([])).cpu().numpy().astype(int) if torch.is_tensor(b.get('classes',None)) else np.array([])
    vals=b.get('map_per_class',torch.tensor([])).cpu().numpy() if torch.is_tensor(b.get('map_per_class',None)) else np.array([])
    for cls,val in zip(classes,vals):
        c=int(cls)-1
        if 0<=c<NC: ap50_pc[c]=float(val)
    p,r,f1,tp,fp,fn=best
    return {
        'mAP50_95':float(a['map']), 'mAP50':float(a['map_50']), 'mAP75':float(a['map_75']),
        'precision':p,'recall':r,'f1':f1,'best_conf':best_thr,
        'tp':tp,'fp':fp,'fn':fn,'ap50_per_class':ap50_pc.tolist()
    }

@torch.inference_mode()
def evaluate_torch_detector(model, loader, device, score_floor=0.001):
    model.eval(); preds=[]; tgts=[]; times=[]
    for images,targets in loader:
        images=[im.to(device) for im in images]
        if device.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter(); outputs=model(images)
        if device.type=='cuda': torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/len(images)
        for o,t in zip(outputs,targets):
            keep=o['scores'].detach().cpu()>=score_floor
            preds.append({'boxes':o['boxes'].detach().cpu()[keep], 'scores':o['scores'].detach().cpu()[keep], 'labels':o['labels'].detach().cpu()[keep]})
            tgts.append({'boxes':t['boxes'].cpu(), 'labels':t['labels'].cpu()})
            times.append(dt)
    metrics=summarize_detection_lists(preds,tgts)
    metrics['inference_ms']=1000*float(np.mean(times)) if times else np.nan
    return metrics

@torch.inference_mode()
def evaluate_yolo_detector(model, frame):
    preds=[]; tgts=[]; times=[]
    for _,row in frame.reset_index(drop=True).iterrows():
        img=cv2.imread(row.image_path); H,W=img.shape[:2]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.perf_counter()
        rr=model.predict(row.image_path,imgsz=YOLO_IMGSZ,conf=0.001,iou=0.7,max_det=300,verbose=False,device=0)[0]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
        if rr.boxes is None or len(rr.boxes)==0:
            pb=torch.zeros((0,4)); ps=torch.zeros((0,)); pl=torch.zeros((0,),dtype=torch.long)
        else:
            pb=rr.boxes.xyxy.detach().cpu(); ps=rr.boxes.conf.detach().cpu(); pl=rr.boxes.cls.detach().cpu().long()+1
        gb=[];gl=[]
        for line in Path(row.label_path).read_text().splitlines():
            if not line.strip():continue
            c,x,y,w,h=map(float,line.split()); gb.append([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H]); gl.append(int(c)+1)
        preds.append({'boxes':pb,'scores':ps,'labels':pl})
        tgts.append({'boxes':torch.tensor(gb,dtype=torch.float32).reshape(-1,4),'labels':torch.tensor(gl,dtype=torch.long)})
    metrics=summarize_detection_lists(preds,tgts); metrics['inference_ms']=1000*float(np.mean(times))
    return metrics

# -----------------------------
# 7. TRAIN MODEL A — YOLO26m
# -----------------------------
MODEL_RESULTS={}; TRAIN_HIST={}
yolo_name=f'model2_tamban_yolo26m_partdet_{REVISION}'
yolo_project=OUT_ROOT/'yolo26m_training'
print('\n'+'='*84+'\nTRAINING 1/3:',yolo_name+'\n'+'='*84)

t0=time.perf_counter()
yolo=YOLO(YOLO_MODEL)
yolo.train(
    data=str(yolo_yaml), project=str(yolo_project), name=yolo_name,
    epochs=YOLO_EPOCHS, patience=YOLO_PATIENCE, imgsz=YOLO_IMGSZ,
    batch=-1, device=0, workers=NUM_WORKERS, seed=SEED, deterministic=False,
    pretrained=True, optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=5e-4,
    # Small dataset: modest spatial augmentation, minimal color distortion.
    mosaic=0.5, close_mosaic=10, mixup=0.0, copy_paste=0.0,
    hsv_h=0.0, hsv_s=0.04, hsv_v=0.04,
    degrees=20.0, translate=0.06, scale=0.20, shear=2.0,
    fliplr=0.5, flipud=0.5,
    cache=False, amp=True, plots=True, save=True, verbose=True
)
yolo_train_sec=time.perf_counter()-t0
run_dir=yolo_project/yolo_name
raw_best=run_dir/'weights'/'best.pt'
if not raw_best.exists(): raw_best=run_dir/'weights'/'last.pt'
yolo_best=OUT_ROOT/f'{yolo_name}_devbest.pt'
shutil.copy2(raw_best,yolo_best)
yolo_eval=YOLO(str(yolo_best))
MODEL_RESULTS['YOLO26m']=evaluate_yolo_detector(yolo_eval,dev_df)
MODEL_RESULTS['YOLO26m']['training_minutes']=yolo_train_sec/60
MODEL_RESULTS['YOLO26m']['checkpoint']=str(yolo_best)
# Capture YOLO loss history for plotting
csvp=run_dir/'results.csv'
if csvp.exists():
    yd=pd.read_csv(csvp); yd.columns=[c.strip() for c in yd.columns]
    loss_cols=[c for c in yd.columns if c.startswith('train/') and c.endswith('_loss')]
    if loss_cols: TRAIN_HIST['YOLO26m']=yd[loss_cols].sum(axis=1).tolist()

print('YOLO26m dev metrics:',{k:round(v,4) if isinstance(v,float) else v for k,v in MODEL_RESULTS['YOLO26m'].items() if k not in ['ap50_per_class','checkpoint']})

del yolo,yolo_eval; torch.cuda.empty_cache()

# -----------------------------
# 8. TRAINING HELPERS FOR TORCHVISION DETECTORS
# -----------------------------
DEVICE=torch.device('cuda')

def make_loaders(batch_size):
    sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    tr=DataLoader(train_ds,batch_size=batch_size,sampler=sampler,num_workers=NUM_WORKERS,collate_fn=collate_fn,pin_memory=True,persistent_workers=(NUM_WORKERS>0))
    dv=DataLoader(dev_ds,batch_size=1,shuffle=False,num_workers=NUM_WORKERS,collate_fn=collate_fn,pin_memory=True,persistent_workers=(NUM_WORKERS>0))
    return tr,dv

def set_bn_eval(m):
    if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.SyncBatchNorm)): m.eval()

def train_torch_detector(model, model_key, canonical_name, epochs, batch_size, lr, freeze_bn=False):
    train_loader,dev_loader=make_loaders(batch_size)
    model=model.to(DEVICE)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(epochs,1),eta_min=lr*0.05)
    scaler=torch.cuda.amp.GradScaler(enabled=AMP)
    accum=max(1,math.ceil(TARGET_EFFECTIVE_BATCH/batch_size))
    history=[]; best_map=-1; stale=0
    best_path=OUT_ROOT/f'{canonical_name}_devbest.pth'
    last_path=OUT_ROOT/f'{canonical_name}_last.pth'
    start=time.perf_counter()
    print(f'Batch={batch_size}, grad accumulation={accum}, effective~{batch_size*accum}, lr={lr}')
    for epoch in range(1,epochs+1):
        model.train()
        if freeze_bn: model.apply(set_bn_eval)
        optimizer.zero_grad(set_to_none=True)
        running=0.0; steps=0
        for bi,(images,targets) in enumerate(train_loader,1):
            images=[im.to(DEVICE,non_blocking=True) for im in images]
            targets=[{k:v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=AMP):
                loss_dict=model(images,targets)
                loss=sum(loss_dict.values())/accum
            scaler.scale(loss).backward()
            if bi%accum==0 or bi==len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params,10.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu())*accum; steps+=1
        scheduler.step()
        avg=running/max(steps,1); history.append(avg)
        msg=f'[{model_key}] epoch {epoch:03d}/{epochs} loss={avg:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}'
        if epoch%EVAL_EVERY==0 or epoch==epochs:
            ev=evaluate_torch_detector(model,dev_loader,DEVICE)
            msg+=f' | dev mAP50-95={ev["mAP50_95"]:.4f} mAP50={ev["mAP50"]:.4f} F1={ev["f1"]:.4f}'
            if ev['mAP50_95']>best_map+1e-5:
                best_map=ev['mAP50_95']; stale=0
                torch.save({'model_state':model.state_dict(),'classes':CLASS_NAMES,'revision':REVISION,'epoch':epoch,'dev_metrics':ev},best_path)
            else:
                stale+=1
            print(msg)
            if stale>=EARLY_STOP_EVALS:
                print(f'[{model_key}] early stopping: no mAP50-95 improvement for {stale} evaluations.')
                break
        else:
            print(msg)
    train_sec=time.perf_counter()-start
    torch.save({'model_state':model.state_dict(),'classes':CLASS_NAMES,'revision':REVISION,'epoch':epoch},last_path)
    ck=torch.load(best_path,map_location=DEVICE,weights_only=False); model.load_state_dict(ck['model_state'])
    final=evaluate_torch_detector(model,dev_loader,DEVICE)
    final['training_minutes']=train_sec/60; final['checkpoint']=str(best_path); final['best_epoch']=ck.get('epoch')
    TRAIN_HIST[model_key]=history
    return model,final

# -----------------------------
# 9. TRAIN MODEL B — FASTER R-CNN RESNET50-FPN V2
# -----------------------------
rcnn_name=f'model2_tamban_fasterrcnn_r50fpn_v2_partdet_{REVISION}'
print('\n'+'='*84+'\nTRAINING 2/3:',rcnn_name+'\n'+'='*84)
rcnn=fasterrcnn_resnet50_fpn_v2(weights='DEFAULT',min_size=MIN_SIZE,max_size=MAX_SIZE)
in_features=rcnn.roi_heads.box_predictor.cls_score.in_features
rcnn.roi_heads.box_predictor=FastRCNNPredictor(in_features,NC+1)
rcnn,MODEL_RESULTS['Faster R-CNN V2']=train_torch_detector(
    rcnn,'Faster R-CNN V2',rcnn_name,RCNN_EPOCHS,RCNN_BATCH,lr=2e-4,freeze_bn=False
)
print('Faster R-CNN V2 dev metrics:',{k:round(v,4) if isinstance(v,float) else v for k,v in MODEL_RESULTS['Faster R-CNN V2'].items() if k not in ['ap50_per_class','checkpoint']})
del rcnn; torch.cuda.empty_cache()

# -----------------------------
# 10. EFFICIENTNETV2 + FPN BACKBONE
# -----------------------------
class EfficientNetV2FPN(nn.Module):
    def __init__(self,variant='m',out_channels=256):
        super().__init__()
        if variant=='s': base=efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        elif variant=='l': base=efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        else: base=efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.body=base.features
        # Dynamically find the last stage at each spatial resolution, then use the final 4 scales.
        was_training=self.body.training; self.body.eval()
        with torch.no_grad():
            x=torch.zeros(1,3,256,256); stages=[]
            for i,layer in enumerate(self.body):
                x=layer(x); stages.append((i,int(x.shape[1]),int(x.shape[-2]),int(x.shape[-1])))
        if was_training:self.body.train()
        by_hw=OrderedDict()
        for item in stages: by_hw[(item[2],item[3])]=item  # keep last stage at same resolution
        chosen=list(by_hw.values())[-4:]
        self.indices=[x[0] for x in chosen]
        self.in_channels=[x[1] for x in chosen]
        self.fpn=FeaturePyramidNetwork(self.in_channels,out_channels,extra_blocks=LastLevelMaxPool())
        self.out_channels=out_channels
        print('EfficientNetV2-FPN stages:',chosen)
    def forward(self,x):
        feats=OrderedDict(); j=0
        for i,layer in enumerate(self.body):
            x=layer(x)
            if i in self.indices:
                feats[str(j)]=x; j+=1
        return self.fpn(feats)

def make_effnet_retinanet(variant='m'):
    backbone=EfficientNetV2FPN(variant,256)
    sizes=((24,),(48,),(96,),(192,),(384,))
    aspects=((0.4,0.6,1.0,1.7,2.5),)*5
    anchors=AnchorGenerator(sizes=sizes,aspect_ratios=aspects)
    return RetinaNet(backbone,num_classes=NC+1,anchor_generator=anchors,min_size=MIN_SIZE,max_size=MAX_SIZE,
                     score_thresh=0.001,topk_candidates=1000,detections_per_img=300)

# -----------------------------
# 11. TRAIN MODEL C — EFFICIENTNETV2-M + RETINANET
# -----------------------------
eff_tag={'s':'effnetv2s','m':'effnetv2m','l':'effnetv2l'}[EFFNET_VARIANT]
eff_name=f'model2_tamban_{eff_tag}_retinanet_partdet_{REVISION}'
print('\n'+'='*84+'\nTRAINING 3/3:',eff_name+'\n'+'='*84)
eff=make_effnet_retinanet(EFFNET_VARIANT)
eff,MODEL_RESULTS[f'EfficientNetV2-{EFFNET_VARIANT.upper()} RetinaNet']=train_torch_detector(
    eff,f'EfficientNetV2-{EFFNET_VARIANT.upper()} RetinaNet',eff_name,EFFNET_EPOCHS,EFF_BATCH,lr=1.5e-4,freeze_bn=True
)
print('EfficientNet RetinaNet dev metrics:',{k:round(v,4) if isinstance(v,float) else v for k,v in MODEL_RESULTS[f'EfficientNetV2-{EFFNET_VARIANT.upper()} RetinaNet'].items() if k not in ['ap50_per_class','checkpoint']})
del eff; torch.cuda.empty_cache()

# -----------------------------
# 12. BUILD FAIR THREE-MODEL COMPARISON TABLE
# -----------------------------
rows=[]
for name,m in MODEL_RESULTS.items():
    rows.append({
        'Model':name,
        'mAP50-95':m['mAP50_95'],
        'mAP50':m['mAP50'],
        'mAP75':m['mAP75'],
        'Precision@bestF1':m['precision'],
        'Recall@bestF1':m['recall'],
        'F1@IoU0.5':m['f1'],
        'Dev_conf_threshold':m['best_conf'],
        'Inference_ms_per_image':m['inference_ms'],
        'Training_minutes':m['training_minutes'],
        'Checkpoint':m['checkpoint']
    })
comparison=pd.DataFrame(rows).sort_values('mAP50-95',ascending=False).reset_index(drop=True)
comparison.to_csv(OUT_ROOT/'model2_three_model_comparison.csv',index=False)
print('\n'+'='*84)
print('FINAL INTERNAL-DEVELOPMENT COMPARISON')
print('='*84)
print(comparison.to_string(index=False))

# -----------------------------
# 13. GRAPHS — COMPARISON OF THE THREE MODELS
# -----------------------------
# Graph 1: quality metrics. This is the main model comparison figure.
plot_df=comparison.set_index('Model')
metric_cols=['mAP50-95','mAP50','Precision@bestF1','Recall@bestF1','F1@IoU0.5']
ax=(plot_df[metric_cols]*100).T.plot(kind='bar',figsize=(15,7))
ax.set_title('Tamban Model 2 — Three-Model Detection Comparison (Internal Development Split)',fontsize=14,fontweight='bold')
ax.set_ylabel('Percent (%)'); ax.set_xlabel('Metric'); ax.set_ylim(0,100); ax.grid(axis='y',alpha=.25)
plt.xticks(rotation=0); plt.legend(title='Architecture',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout()
plt.savefig(OUT_ROOT/'01_model2_three_model_quality_comparison.png',dpi=220,bbox_inches='tight'); plt.show()

# Graph 2: accuracy vs speed trade-off.
fig,ax=plt.subplots(figsize=(10,7))
for _,r in comparison.iterrows():
    ax.scatter(r['Inference_ms_per_image'],r['mAP50-95']*100,s=130)
    ax.annotate(r['Model'],(r['Inference_ms_per_image'],r['mAP50-95']*100),xytext=(7,7),textcoords='offset points',fontsize=9)
ax.set_title('Model 2 — Accuracy vs Inference Latency',fontweight='bold')
ax.set_xlabel('Inference latency (ms/image, current Colab GPU)'); ax.set_ylabel('mAP50-95 (%)'); ax.grid(alpha=.25); plt.tight_layout()
plt.savefig(OUT_ROOT/'02_model2_accuracy_vs_speed.png',dpi=220,bbox_inches='tight'); plt.show()

# Graph 3: AP50 by each of the 12 classes.
pc=pd.DataFrame({'Class':CLASS_NAMES})
for name,m in MODEL_RESULTS.items(): pc[name]=np.array(m['ap50_per_class'])*100
pc.to_csv(OUT_ROOT/'model2_per_class_ap50.csv',index=False)
ax=pc.set_index('Class').plot(kind='barh',figsize=(13,10))
ax.set_title('Model 2 — Per-Class AP50 Comparison',fontweight='bold'); ax.set_xlabel('AP50 (%)'); ax.set_xlim(0,100); ax.grid(axis='x',alpha=.25)
plt.legend(title='Architecture',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout()
plt.savefig(OUT_ROOT/'03_model2_per_class_ap50_comparison.png',dpi=220,bbox_inches='tight'); plt.show()

# Graph 4: training-loss traces. Loss scales differ between detector families,
# so normalize each curve to its first epoch for SHAPE comparison only.
fig,ax=plt.subplots(figsize=(12,6))
for name,h in TRAIN_HIST.items():
    if len(h):
        arr=np.array(h,float); ax.plot(np.arange(1,len(arr)+1),arr/(arr[0]+1e-9),label=name)
ax.set_title('Training Loss Trend (Normalized; scales are not directly comparable)',fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss / first-epoch loss'); ax.grid(alpha=.25); ax.legend(); plt.tight_layout()
plt.savefig(OUT_ROOT/'04_model2_normalized_training_loss.png',dpi=220,bbox_inches='tight'); plt.show()

# -----------------------------
# 14. SAVE COMPLETE METADATA / MODEL REGISTRY
# -----------------------------
registry={
    'project':'Tamban Model 2 quality-part detection',
    'revision':REVISION,
    'classes':CLASS_NAMES,
    'dataset_zip':str(DATASET_ZIP),
    'dataset_images':int(len(df)),
    'dataset_objects':int(sum(obj_counts.values())),
    'split_type':'internal group-aware development split; NOT specimen-independent final test',
    'train_images':int((df.split=='train').sum()),
    'dev_images':int((df.split=='dev').sum()),
    'models':MODEL_RESULTS,
    'notes':[
        'Ungraded is not a training class; it belongs to downstream evidence/fusion logic.',
        'No strong HSV augmentation was used because quality classes may depend on real color/reflectance cues.',
        'Model selection here is provisional until new specimen-independent validation/test captures exist.',
        'All three models were compared on exactly the same internal development image list.'
    ]
}
with open(OUT_ROOT/'model2_r03_model_registry.json','w') as f: json.dump(registry,f,indent=2)

# Plain-text thesis-friendly summary
winner=comparison.iloc[0]
summary=f'''TAMBAN MODEL 2 — THREE-MODEL INTERNAL DEVELOPMENT COMPARISON\n\nDataset: {len(df)} cleaned images, {sum(obj_counts.values())} part boxes, 12 grade-part classes.\nTrain/dev: {(df.split=='train').sum()} / {(df.split=='dev').sum()} images, group-aware engineering split.\n\nHighest internal-dev mAP50-95: {winner['Model']} = {winner['mAP50-95']:.4f}\n\nCAUTION: This is NOT a final thesis winner. The cleaned pool lacks specimen-independent ground truth.\nThe three checkpoints must next be tested on newly captured, mixed-grade, specimen-independent images.\n'''
(OUT_ROOT/'READ_ME_AFTER_TRAINING.txt').write_text(summary)

print('\n'+'='*84)
print('TRAINING COMPLETE')
print('='*84)
print(summary)
print('Saved to:',OUT_ROOT)
print('\nFiles created:')
for p in sorted(OUT_ROOT.iterdir()):
    if p.is_file(): print(' -',p.name)
print('\nNEXT STEP: use newly captured independent validation images before choosing the production Model 2.')
