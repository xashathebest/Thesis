"""Tamban V3: evidence-gated individual-fish detection / grading workbench.

Complete source is embedded in the Colab notebook; only data/checkpoints are uploaded.
Preserves old checkpoints. Does not generate trustworthy whole-fish labels from parts.
Default Colab mode audits the legacy training pool and optionally runs its saved model.
Training/evaluation mode requires human-reviewed whole-instance labels and group metadata.

No performance claims are made by this source. Read VERIFICATION.json for tested stages.
Uses Ultralytics YOLO26 architecture, pretrained weights and loss; explicit torch loop.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, itertools, json, math, os, random, re
import shutil, stat, subprocess, sys, time, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

VERSION = 'tamban_v3.1'
ULTRALYTICS_VERSION = '8.4.153'
GRADES = ['Class_A', 'Class_B', 'Class_C', 'Rejected']
PART_NAMES = [f'Grade_{g}_{p}' for g in 'ABC' for p in ('Body','Head','Tail')] + [f'Rejected_{p}' for p in ('Body','Head','Tail')]
LEGACY_HASH = '4061dcd9d940b86f0c957acbe024766d3b5812eb15776b8b19d96b19274ca4a2'
IMAGE_EXT = {'.jpg','.jpeg','.png','.bmp','.webp'}
DEFAULT = {
    'model': 'yolo26s.pt', 'epochs': 40, 'imgsz': 640, 'batch': 4,
    'accumulate': 8, 'lr': .0003, 'weight_decay': .0001, 'warmup_epochs': 3,
    'seed': 42, 'amp': True, 'validate_every': 5, 'patience_checks': 6,
    'class_sampling': False, 'rot90': False, 'device': '0',
}
BASE_INFER = {'imgsz':640,'conf':.25,'iou':.7,'max_det':300,'head':'many',
              'tiles':False,'tile_size':640,'tile_overlap':.25,'merge_iou':.75,
              'exclusive_grades':False,'max_tiles':64}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def jsonable(v):
    if isinstance(v,dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray): return jsonable(v.tolist())
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,Path):return str(v)
    return v

def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(jsonable(obj),indent=2,allow_nan=False),encoding='utf-8')

def atomic_torch_save(obj,path):
    import torch
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temp);os.replace(temp,path)

def unique_run(base,label):
    p=Path(base)/(label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    p.mkdir(parents=True,exist_ok=False);return p

def install_colab_dependencies():
    # Do not replace Colab's working CUDA torch intentionally.
    pkgs=[f'ultralytics=={ULTRALYTICS_VERSION}','scikit-image>=0.24,<0.27',
          'scikit-learn>=1.5,<1.10','PyYAML>=6,<7','joblib>=1.3,<2','tqdm>=4.65,<5']
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--upgrade-strategy','only-if-needed',*pkgs])
    import ultralytics
    if ultralytics.__version__!=ULTRALYTICS_VERSION:
        raise RuntimeError('Restart this session; another Ultralytics version was already imported.')

def safe_extract(archive,destination):
    base=Path(destination).resolve();base.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>200000 or sum(i.file_size for i in infos)>30_000_000_000:
            raise ValueError('Archive too large for this automatic import.')
        for info in infos:
            if not (base/info.filename).resolve().is_relative_to(base) or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe ZIP path/link.')
        z.extractall(base)
    return base

def image_paths(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXT)

def dataset_root(base):
    candidates=[p.parent for p in Path(base).rglob('data.yaml') if (p.parent/'train/images').is_dir()]
    if len(candidates)!=1: raise ValueError('Expected one data.yaml beside train/images.')
    return candidates[0]

def names_from_yaml(path):
    raw=yaml.safe_load(Path(path).read_text())
    n=raw.get('names',[])
    if isinstance(n,dict):
        if sorted(map(int,n))!=list(range(len(n))):raise ValueError('Class IDs must be contiguous from zero.')
        n=[n[k] for k in sorted(n,key=int)]
    if not isinstance(n,list) or not all(isinstance(x,str) for x in n):raise ValueError('Invalid class names.')
    return n

def canonical_grade(name):
    s=re.sub('[^a-z0-9]','',name.lower())
    lookup={'a':0,'classa':0,'gradea':0,'b':1,'classb':1,'gradeb':1,
            'c':2,'classc':2,'gradec':2,'rejected':3,'reject':3}
    if s not in lookup: raise ValueError(f'{name!r} is NOT a whole-instance grade. Part labels cannot be renamed into fish boxes.')
    return lookup[s]

def read_boxes(path,nc):
    if not Path(path).is_file():raise ValueError('Missing label file: '+str(path))
    rows=[]
    for j,line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():continue
        a=np.array([float(v) for v in line.split()])
        if len(a)!=5:raise ValueError(f'{path}:{j+1}: use reviewed detection BOX labels, not part polygons.')
        if not np.isfinite(a).all() or a[0]!=int(a[0]) or not 0<=a[0]<nc or min(a[3:])<=0:
            raise ValueError(f'Invalid label at {path}:{j+1}.')
        if min(a[1]-a[3]/2,a[2]-a[4]/2)<-1e-5 or max(a[1]+a[3]/2,a[2]+a[4]/2)>1+1e-5:
            raise ValueError(f'Out-of-bounds label at {path}:{j+1}.')
        rows.append(a)
    return np.array(rows,dtype=np.float32).reshape(-1,5)

def xywh_to_xyxy(labels,w,h):
    a=np.asarray(labels).reshape(-1,5)
    out=np.zeros((len(a),4),np.float32)
    if len(a):
        out[:,0]=(a[:,1]-a[:,3]/2)*w;out[:,2]=(a[:,1]+a[:,3]/2)*w
        out[:,1]=(a[:,2]-a[:,4]/2)*h;out[:,3]=(a[:,2]+a[:,4]/2)*h
    return out

def xyxy_to_xywh(boxes,classes,w,h):
    a=np.zeros((len(boxes),5),np.float32)
    if len(a):
        a[:,0]=classes;a[:,1:3]=(boxes[:,:2]+boxes[:,2:])/2/[w,h]
        a[:,3:5]=(boxes[:,2:]-boxes[:,:2])/[w,h]
    return a

def iou_matrix(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(0,hi-lo),axis=2)
    aa=np.prod(np.maximum(0,a[:,2:]-a[:,:2]),axis=1)
    bb=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def crop_legacy_bars(image,labels=None):
    """Exact legacy transform only. Never inferred for a newly captured image."""
    h,w=image.shape[:2]
    valid=h==w==640 and (image[:120].min(2)>230).mean()>.97 and (image[520:].min(2)>230).mean()>.97
    if not valid:return image.copy(),None if labels is None else labels.copy(),0
    result=image[140:500].copy()
    if labels is None:return result,None,140
    b=xywh_to_xyxy(labels,w,h);b[:,[1,3]]-=140
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,360)
    if np.any(b[:,2:]<=b[:,:2]):raise ValueError('Border removal would erase a labeled part.')
    return result,xyxy_to_xywh(b,labels[:,0],w,360),140

def morphology(mask):
    """Rotation-aware visible silhouette proxies. Not thickness in 3D or true physical size.
    Do not infer a missing part from a gap or from a rejected mask-quality flag.
    """
    mask=np.asarray(mask,dtype=np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    if n<2:return {'shape_available':False}
    idx=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));component=lab==idx
    ys,xs=np.nonzero(component);p=np.column_stack((xs,ys)).astype(np.float32)
    if len(p)<32:return {'shape_available':False}
    vals,vec=np.linalg.eigh(np.cov(p.T));axis=vec[:,np.argmax(vals)];perp=np.array([-axis[1],axis[0]])
    along=(p-p.mean(0))@axis;across=(p-p.mean(0))@perp
    low,high=np.percentile(along,[1,99]);length=high-low
    widths=[]
    for l,r in zip(np.linspace(low,high,17)[:-1],np.linspace(low,high,17)[1:]):
        s=across[(along>=l)&(along<r)]
        if len(s)>=5:widths.append(np.percentile(s,95)-np.percentile(s,5))
    (_, _),(ra,rb),_=cv2.minAreaRect(p)
    hull=cv2.contourArea(cv2.convexHull(p));area=len(p)
    return {'shape_available':True,'width_length_rotated':min(ra,rb)/max(ra,rb,1),
            'width90_length_pca':float(np.percentile(widths,90)/max(length,1)) if widths else np.nan,
            'width50_length_pca':float(np.median(widths)/max(length,1)) if widths else np.nan,
            'area_length2':area/max(length**2,1),'solidity_proxy':min(area/max(hull,1),1.),
            'axis_angle_deg':math.degrees(math.atan2(axis[1],axis[0]))%180,
            'largest_component_fraction':area/max(mask.sum(),1),'length_proxy_px':float(length),
            'mask_area_px':area,'component_touches_crop':bool(component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any())}

COLOR_KEYS=['lab_L_mean','lab_a_mean','lab_b_mean','lab_L_std','lab_a_std','lab_b_std',
            'value_mean','value_p95','value_std','saturation_mean','saturation_std',
            'hue_sin','hue_cos','yellow_ratio','highlight_ratio','gray_std']
TEXTURE_KEYS=['glcm_contrast','glcm_homogeneity','glcm_energy','edge_density']+[f'lbp_{k}' for k in range(18)]
SHAPE_KEYS=['width_length_rotated','width90_length_pca','width50_length_pca','area_length2','solidity_proxy']
FEATURE_SETS={'color':COLOR_KEYS,'color_texture':COLOR_KEYS+TEXTURE_KEYS,
              'color_texture_on_shape_eligible':COLOR_KEYS+TEXTURE_KEYS,
              'color_texture_shape':COLOR_KEYS+TEXTURE_KEYS+SHAPE_KEYS}

def region_features(crop,other_overlap=0.,frame_clipped=False):
    """Dark-background approximation with logged quality checks. No background/size metadata in model inputs."""
    if crop.size==0 or min(crop.shape[:2])<8:return None,None
    from skimage.feature import local_binary_pattern
    scale=min(1,256/max(crop.shape[:2]));im=crop if scale==1 else cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    mask=cv2.morphologyEx((gray>40).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    inside=cv2.erode(mask,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    if inside.sum()<64:return None,mask
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(im.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
    H,S,V=[hsv[:,:,j][inside].astype(float) for j in range(3)]
    a=H[S>=40]*2*np.pi/180
    f={'value_mean':V.mean(),'value_p95':np.percentile(V,95),'value_std':V.std(),
       'saturation_mean':S.mean(),'saturation_std':S.std(),'hue_sin':np.sin(a).mean() if len(a) else 0.,
       'hue_cos':np.cos(a).mean() if len(a) else 0.,'yellow_ratio':((H>=15)&(H<=40)&(S>=40)).mean(),
       'highlight_ratio':((V>=220)&(S<=80)).mean(),'gray_std':gray[inside].std()}
    for i,name in enumerate(['L','a','b']):
        f[f'lab_{name}_mean']=lab[:,:,i][inside].mean();f[f'lab_{name}_std']=lab[:,:,i][inside].std()
    q=(gray.astype(int)*16//256);P=np.zeros((16,16),float)
    for dx,dy in [(1,0),(0,1),(1,1),(-1,1),(2,0),(0,2),(2,2),(-2,2)]:
        h,w=q.shape;xa,xb=max(0,-dx),min(w,w-dx);ya,yb=max(0,-dy),min(h,h-dy)
        ok=inside[ya:yb,xa:xb]&inside[ya+dy:yb+dy,xa+dx:xb+dx]
        x=q[ya:yb,xa:xb][ok];y=q[ya+dy:yb+dy,xa+dx:xb+dx][ok]
        if len(x):
            m=np.bincount(x*16+y,minlength=256).reshape(16,16);P+=(m+m.T)/(2*len(x))
    P/=max(P.sum(),1);ii,jj=np.indices((16,16))
    f.update(glcm_contrast=(((ii-jj)**2)*P).sum(),glcm_homogeneity=(P/(1+(ii-jj)**2)).sum(),glcm_energy=np.sqrt((P*P).sum()))
    L=local_binary_pattern(gray,16,2,'uniform');hist=np.bincount(L[inside].astype(int),minlength=18).astype(float);hist/=hist.sum()
    f.update({f'lbp_{i}':x for i,x in enumerate(hist)})
    f['edge_density']=((cv2.Canny(gray,50,150)>0)&inside).sum()/inside.sum()
    dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    f['darkline_candidate_ratio']=((dark>15)&inside).sum()/inside.sum() # Not used in classifiers.
    f.update(morphology(mask));f['foreground_fraction']=float(mask.mean());f['processing_scale']=scale
    f['shape_usable_proxy']=bool(f.get('shape_available',False) and f['largest_component_fraction']>=.9 and
                               f['length_proxy_px']>=30 and not frame_clipped and other_overlap<.1)
    f['mask_caution']='threshold mask, not confirmed instance segmentation'
    return jsonable(f),mask

# --------------------------- DATA AUDIT --------------------------------
def audit_legacy(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if names_from_yaml(root/'data.yaml')!=PART_NAMES:raise ValueError('Expected the preserved 12-class part dataset.')
    rows=[];anns=[];hashbits=[];issues=[]
    paths=image_paths(root/'train/images')
    image_stems={p.stem for p in paths};label_stems={p.stem for p in (root/'train/labels').glob('*.txt')}
    if len(image_stems)!=len(paths):raise ValueError('Duplicate image basenames in legacy pool.')
    for stem in sorted(label_stems-image_stems):issues.append({'source':stem,'issue':'orphan label without image'})
    for p in paths:
        im=cv2.imread(str(p));
        if im is None:raise ValueError('Unreadable image: '+str(p))
        labels=read_boxes(root/'train/labels'/f'{p.stem}.txt',12)
        photo,b,top=crop_legacy_bars(im,labels);h,w=photo.shape[:2];boxes=xywh_to_xyxy(b,w,h)
        src=p.name.split('.rf.')[0];g=sorted(set(int(x)//3 for x in labels[:,0]))
        R=re.search(r'WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})',src)
        stamp=pd.to_datetime(''.join(R.groups()),format='%Y%m%d%H%M%S').isoformat() if R else ''
        gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY);v=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8]
        hashbits.append((v>np.median(v[1:])).ravel())
        body=np.where(b[:,0].astype(int)%3==0)[0];ov=iou_matrix(boxes[body],boxes[body]);np.fill_diagonal(ov,0)
        for j,row in enumerate(labels):
            pw,ph=row[3]*im.shape[1],row[4]*im.shape[0]
            anns.append({'source':src,'class_id':int(row[0]),'part':PART_NAMES[int(row[0])],
                         'min_side_px':float(min(pw,ph)),'area_px':float(pw*ph),
                         'touches_photo':bool(boxes[j,0]<=1 or boxes[j,1]<=1 or boxes[j,2]>=w-1 or boxes[j,3]>=h-1)})
        if len(labels)==0:issues.append({'source':src,'issue':'empty label: inspect whether legitimate negative'})
        if len(labels)!=len(set(tuple(x) for x in labels.tolist())):issues.append({'source':src,'issue':'exact duplicate label'})
        rows.append({'source':src,'image':str(p),'timestamp':stamp,'grades':'|'.join(GRADES[x] for x in g),
                     'part_boxes':len(labels),'body_appearances':len(body),'max_body_box_iou':float(ov.max()) if ov.size else 0.,
                     'width':im.shape[1],'height':im.shape[0],'photo_width':w,'photo_height':h,
                     'sha256':sha256(p),'pixels_sha256':hashlib.sha256(im.tobytes()).hexdigest()})
    R=pd.DataFrame(rows);A=pd.DataFrame(anns)
    R.to_csv(out/'images.csv',index=False);A.to_csv(out/'parts.csv',index=False)
    D=squareform(pdist(np.asarray(hashbits),metric='hamming'))*64 if len(R)>1 else np.zeros((len(R),len(R)))
    pairs=[{'source_a':R.iloc[i].source,'source_b':R.iloc[j].source,'hash_bits':int(D[i,j]),'status':'candidate, not proven same fish'}
           for i,j in zip(*np.where(np.triu(D<=4,1)))]
    pd.DataFrame(pairs).to_csv(out/'similar_scene_candidates.csv',index=False)
    # This is a metadata template, not an invented specimen split.
    group=R[['source','image','timestamp']].copy();group['capture_batch']='UNKNOWN';group['specimen_ids']=''
    group['leakage_group']='UNKNOWN';group['split']='train';group['identity_confirmed']=False
    group.to_csv(out/'legacy_identity_template.csv',index=False)
    summary={'images':len(R),'part_annotations':len(A),'whole_fish_annotations':0,
             'class_counts':A.class_id.value_counts().sort_index().to_dict(),
             'max_part_boxes':int(R.part_boxes.max()),'max_body_appearances':int(R.body_appearances.max()),
             'short_side_below8px':int((A.min_side_px<8).sum()),'photo_edge_parts':int(A.touches_photo.sum()),
             'similar_scene_candidate_pairs':len(pairs),'exact_file_duplicates':int(R.sha256.duplicated().sum()),
             'orphan_labels':len(label_stems-image_stems),'missing_labels':len(image_stems-label_stems),
             'edge_definition':'Continuous transformed box within one pixel of photo boundary',
             'validation_available':False,'test_available':False,'issues':issues,
             'warning':'Legacy data are suitable for part-debugging, NOT ground truth for individual-fish grading.'}
    write_json(out/'audit.json',summary);return R,A,summary

# ---------------------- REVIEWED FISH-LEVEL DATA ------------------------
REQUIRED_SPEC={
    'annotation_unit':'fish_instance', 'whole_instance_labels_reviewed':True,
    'exhaustive_visible_items':True, 'specimen_groups_confirmed':True,
    'test_untouched':True, 'no_ignored_regions':True,
}

def validate_group_table(meta):
    needed={'image','split','capture_batch','specimen_ids','leakage_group'}
    if not needed<=set(meta):raise ValueError('groups.csv is missing: '+str(needed-set(meta)))
    if meta.image.duplicated().any():raise ValueError('groups.csv contains duplicate image keys.')
    if not set(meta.split)<= {'train','val','test'}:raise ValueError('Use train / val / test in groups.csv.')
    for col in ['capture_batch','leakage_group']:
        if meta[col].astype(str).str.strip().isin(['','UNKNOWN','nan','None']).any():raise ValueError(f'Unresolved {col}.')
        crossing=meta.groupby(col).split.nunique()
        if (crossing>1).any():raise ValueError(f'{col} crosses splits: '+str(crossing[crossing>1].index.tolist()[:10]))
    sightings=defaultdict(set)
    for _,r in meta.iterrows():
        ids=[x.strip() for x in str(r.specimen_ids).split('|') if x.strip()]
        negative=str(r.get('is_negative',False)).lower()=='true'
        if not ids and negative:continue
        if not ids or any(x.upper() in {'UNKNOWN','NAN','NONE'} for x in ids):raise ValueError('Fish-containing images need known specimen IDs; verified empty images need is_negative=True.')
        for s in ids:sightings[s].add(r.split)
    bad=[s for s,sp in sightings.items() if len(sp)>1]
    if bad:raise ValueError('Same physical specimens cross splits: '+str(bad[:10]))
    return True

def prepare_fish_data(root,work,out):
    """Strictly require real 4-grade INSTANCE labels, never union guessed part boxes."""
    root=Path(root);spec_path=root/'dataset_spec.json';groups_path=root/'groups.csv'
    if not spec_path.exists() or not groups_path.exists():
        raise ValueError('Fish-instance data need reviewed boxes, dataset_spec.json and groups.csv. See supplied template. Legacy part labels cannot train this target.')
    spec=json.loads(spec_path.read_text())
    for k,v in REQUIRED_SPEC.items():
        if spec.get(k)!=v:raise ValueError(f'dataset_spec.json requires confirmed {k}={v!r}. Do not mark true until checked.')
    if not str(spec.get('fragment_counting_policy','')).strip():raise ValueError('Define how separate rejected fragments are counted.')
    original_names=names_from_yaml(root/'data.yaml');mapping={i:canonical_grade(n) for i,n in enumerate(original_names)}
    if len(mapping)!=4 or set(mapping.values())!=set(range(4)):raise ValueError('Four unique whole-instance grades required.')
    meta=pd.read_csv(groups_path,keep_default_na=False);validate_group_table(meta)
    entries=[];checks=[];counts=Counter();source_key=defaultdict(set);file_hashes=defaultdict(set);pixel_hashes=defaultdict(set)
    actual=set()
    for sp in ['train','val','test']:
        sub='valid' if sp=='val' and (root/'valid/images').exists() else sp
        for p in image_paths(root/sub/'images'):
            rel=p.relative_to(root).as_posix();actual.add(rel)
            row=meta.loc[meta.image==rel]
            if len(row)!=1 or row.iloc[0].split!=sp:raise ValueError('Missing/inconsistent metadata for '+rel)
            row=row.iloc[0]
            label=root/sub/'labels'/p.relative_to(root/sub/'images').with_suffix('.txt')
            a=read_boxes(label,4)
            neg=str(row.get('is_negative',False)).lower()=='true'
            if neg != (len(a)==0):raise ValueError('Empty labels must be explicitly verified as is_negative=True, and negatives cannot contain labels: '+rel)
            if len(a):a[:,0]=[mapping[int(x)] for x in a[:,0]]
            im=cv2.imread(str(p));
            if im is None:raise ValueError('Unreadable '+rel)
            if len(a)!=len(set(map(tuple,a.tolist()))):raise ValueError('Duplicate labels in '+rel)
            # Native-resolution originals retained. YOLO handles aspect-preserving resize.
            # No guessed auto-crop, CLAHE, saturation boost or irreversible background removal.
            dest=Path(work)/'fish'/sp/'images'/p.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('Duplicate image basenames across subfolders: '+p.name)
            shutil.copy2(p,dest)
            lp=Path(work)/'fish'/sp/'labels'/f'{p.stem}.txt';lp.parent.mkdir(parents=True,exist_ok=True)
            if lp.exists():raise ValueError('Two image files map to the same label stem: '+p.stem)
            lp.write_text(''.join(f'{int(z[0])} '+ ' '.join(f'{v:.9f}' for v in z[1:])+'\n' for z in a))
            key=p.name.split('.rf.')[0];key=re.sub(r'-\d+-?(?=_jpg|_png|\.)','',key)
            source_key[key].add(sp);fh=sha256(p);ph=hashlib.sha256(im.tobytes()).hexdigest()
            file_hashes[fh].add(sp);pixel_hashes[ph].add(sp)
            boxes=xywh_to_xyxy(a,im.shape[1],im.shape[0]);over=iou_matrix(boxes,boxes);np.fill_diagonal(over,0)
            entries.append({'id':rel,'path':dest,'labels':a,'width':im.shape[1],'height':im.shape[0],
                'split':sp,'capture_batch':row.capture_batch,'specimen_ids':row.specimen_ids,
                'leakage_group':row.leakage_group,'sha256':fh,'max_box_iou':float(over.max()) if over.size else 0})
            for c in a[:,0].astype(int):counts[(sp,c)]+=1
    if actual!=set(meta.image):raise ValueError('groups.csv has absent or unlisted images.')
    for label,groups in [('source name',source_key),('file hash',file_hashes),('pixel hash',pixel_hashes)]:
        if any(len(v)>1 for v in groups.values()):raise ValueError(f'Cross-split duplicate by {label}.')
    for sp in ['train','val','test']:
        for c in range(4):
            if counts[(sp,c)]==0:raise ValueError(f'{sp} lacks {GRADES[c]} ground truth.')
    # Screen near-duplicates across splits, but never call hash identity physical-specimen identity.
    hashes=[]
    for e in entries:
        g=cv2.imread(str(e['path']),0);d=cv2.dct(cv2.resize(g,(32,32)).astype(np.float32))[:8,:8]
        hashes.append((d>np.median(d[1:])).ravel())
    candidates=[]
    for i in range(len(entries)):
        for j in range(i+1,len(entries)):
            if entries[i]['split']==entries[j]['split']:continue
            d=int(np.count_nonzero(hashes[i]!=hashes[j]))
            if d<=4:candidates.append({'image_a':entries[i]['id'],'image_b':entries[j]['id'],'hash_bits':d})
    write_json(Path(out)/'cross_split_near_duplicates.json',candidates)
    if candidates:
        approved=spec.get('reviewed_near_duplicate_pairs',[])
        known={tuple(sorted(x)) for x in approved}
        unresolved=[x for x in candidates if tuple(sorted([x['image_a'],x['image_b']])) not in known]
        if unresolved:raise ValueError('Cross-split near-duplicate candidates need review. See report; only whitelist visually distinct pairs, never real duplicates.')
    pd.DataFrame([{k:v for k,v in e.items() if k!='labels'} for e in entries]).to_csv(Path(out)/'dataset_manifest.csv',index=False)
    fingerprint=hashlib.sha256(json.dumps(sorted((e['id'],e['sha256'],e['split'],e['leakage_group'],e['capture_batch'],e['specimen_ids'],e['labels'].tolist()) for e in entries)).encode()).hexdigest()
    write_json(Path(out)/'dataset_fingerprint.json',{'sha256':fingerprint,'spec':spec})
    return entries,fingerprint

# ------------------------------ INFERENCE -------------------------------
def stable_nms(preds,iou=.75,groups=None,max_det=300):
    """Prediction rows: x1 y1 x2 y2 score class. Return rows and original kept indices.
    Used to merge repeated TILE detections; aggressive suppression can erase real overlap.
    """
    a=np.asarray(preds,float).reshape(-1,6)
    if len(a)==0:return a,np.empty(0,int)
    group=a[:,5].astype(int) if groups is None else np.asarray(groups)
    order=np.argsort(-a[:,4],kind='stable');keep=[]
    while len(order) and len(keep)<max_det:
        i=order[0];keep.append(i);rest=order[1:]
        if not len(rest):break
        ov=iou_matrix(a[i:i+1,:4],a[rest,:4])[0]
        order=rest[~((group[rest]==group[i])&(ov>iou))]
    return a[keep],np.array(keep,int)

def tile_windows(w,h,size,overlap):
    if size<=0 or not 0<=overlap<.8:raise ValueError('Invalid tile geometry.')
    def starts(n):
        if n<=size:return [0]
        seq=list(range(0,n-size+1,max(1,int(size*(1-overlap)))))
        if seq[-1]!=n-size:seq.append(n-size)
        return seq
    return [(x,y,min(w,x+size),min(h,y+size)) for y in starts(h) for x in starts(w)]

class Predictor:
    """Separate YOLO instance per head; do not switch heads after destructive fusion."""
    def __init__(self,checkpoint,unit='fish_instance',device='0'):
        self.checkpoint=Path(checkpoint);self.unit=unit;self.device=device;self.models={}
        if not self.checkpoint.exists():raise FileNotFoundError(str(checkpoint))
    def model_for(self,head):
        from ultralytics import YOLO
        if head not in self.models:
            m=YOLO(str(self.checkpoint));names=list(m.names.values()) if isinstance(m.names,dict) else list(m.names)
            expected=GRADES if self.unit=='fish_instance' else PART_NAMES
            if names!=expected:raise ValueError('Checkpoint names/unit mismatch: '+str(names))
            if getattr(m,'task','detect')!='detect':raise ValueError('This version evaluates BOX detection checkpoints.')
            self.models[head]=m
        return self.models[head]
    def __call__(self,path_or_image,policy,candidate_floor=None,legacy_export=False):
        im=cv2.imread(str(path_or_image)) if isinstance(path_or_image,(str,Path)) else np.asarray(path_or_image).copy()
        if im is None:raise ValueError('Unreadable inference image.')
        top=0
        if legacy_export:im,_,top=crop_legacy_bars(im)
        h,w=im.shape[:2];policy={**BASE_INFER,**policy};model=self.model_for(policy['head'])
        floor=policy['conf'] if candidate_floor is None else candidate_floor
        windows=[(0,0,w,h)]
        if policy['tiles'] and max(w,h)>policy['tile_size']:
            tiles=tile_windows(w,h,policy['tile_size'],policy['tile_overlap'])
            if len(tiles)>policy['max_tiles']:raise ValueError('Too many tiles; increase tile size instead of silently dropping coverage.')
            windows.extend(t for t in tiles if t!=(0,0,w,h))
        preds=[];timings=[]
        for wi,(x,y,x2,y2) in enumerate(windows):
            patch=im[y:y2,x:x2]
            t=time.perf_counter()
            result=model.predict(patch,imgsz=policy['imgsz'],rect=False,conf=floor,
                iou=policy['iou'],max_det=policy['max_det'],nms=(False if policy['head']=='one' else True),
                agnostic_nms=False,device=self.device,verbose=False)[0]
            timings.append(time.perf_counter()-t)
            a=result.boxes.data.detach().cpu().numpy()[:,:6].copy()
            if wi>0 and len(a):
                # Full-image pass supplies objects cut by an INTERNAL tile edge.
                margin=2
                keep=~(((x>0)&(a[:,0]<=margin))|((y>0)&(a[:,1]<=margin))|
                       ((x2<w)&(a[:,2]>=x2-x-margin))|((y2<h)&(a[:,3]>=y2-y-margin)))
                a=a[keep]
            if len(a):a[:,[0,2]]+=x;a[:,[1,3]]+=y;preds.extend(a)
        a=np.asarray(preds,float).reshape(-1,6);raw_count=len(a)
        if len(windows)>1 or policy['exclusive_grades']:
            groups=a[:,5].astype(int)
            if policy['exclusive_grades']:
                # Never suppress a legitimate head against its body in legacy mode.
                groups=(groups%3) if self.unit=='legacy_parts' else np.zeros(len(a),int)
            a,_=stable_nms(a,policy['merge_iou'],groups,policy['max_det'])
        elif len(a)>policy['max_det']:
            a=a[np.argsort(-a[:,4],kind='stable')[:policy['max_det']]]
        if len(a):a[:,[1,3]]+=top
        return a,{'seconds_including_prepost':sum(timings),'passes':len(windows),'premerge_detections':raw_count,
                  'returned':len(a),'limit_reached':len(a)>=policy['max_det'],'unit':self.unit,
                  'head':policy['head'],'legacy_crop_top':top}

def draw_predictions(image,preds,names,unit,out_file,title='PREDICTIONS - not ground truth'):
    from PIL import Image,ImageDraw,ImageFont
    if isinstance(image,(str,Path)):im=Image.open(image).convert('RGB')
    else:im=Image.fromarray(np.asarray(image)[:,:,::-1])
    w,h=im.size;bar=75
    canvas=Image.new('RGB',(w,h+bar),'white');canvas.paste(im,(0,bar));d=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',max(12,int(w/70)))
    except OSError:font=ImageFont.load_default()
    palette=['#f6bd16','#00aaff','#d34eff','#ff5d5d']
    counts=Counter()
    for i,row in enumerate(preds):
        x1,y1,x2,y2,score,c=row;c=int(c);color=palette[c if unit=='fish_instance' else c//3]
        d.rectangle([x1,y1+bar,x2,y2+bar],outline=color,width=max(2,w//500))
        text=f'{i+1}: {names[c]} {score:.2f}'
        d.text((max(0,x1),max(bar,y1+bar)),text,fill=color,font=font,stroke_width=1,stroke_fill='black')
        counts[names[c]]+=1
    if unit=='fish_instance':label=f'Visible fish/items: {len(preds)} | '+ '  '.join(f'{g}: {counts[g]}' for g in names)
    else:label=f'PART boxes: {len(preds)} | body candidates: {sum(int(r[5])%3==0 for r in preds)} | fish count NOT established'
    d.text((8,8),title,fill='black',font=font);d.text((8,35),label,fill='black',font=font)
    Path(out_file).parent.mkdir(parents=True,exist_ok=True);canvas.save(out_file)
    return {'prediction_count':len(preds),'count_by_label':{name:counts[name] for name in names},'counting_unit':unit,
            'visible_grading_item_count':len(preds) if unit=='fish_instance' else None,
            'intact_fish_count':None, 'whole_fish_count':None}

# -------------------------- EXPLICIT EVALUATION -------------------------
def spatial_matches(gt,pred,threshold=.5):
    """Maximum-cardinality IoU matching independent of grade; deterministic secondary IoU objective."""
    ov=iou_matrix(gt,pred)
    if not ov.size:return [],list(range(len(gt))),list(range(len(pred)))
    score=(ov>=threshold).astype(float)*(min(len(gt),len(pred))+1)+ov
    a,b=linear_sum_assignment(-score)
    matches=[(int(i),int(j),float(ov[i,j])) for i,j in zip(a,b) if ov[i,j]>=threshold]
    usedg={i for i,j,v in matches};usedp={j for i,j,v in matches}
    return matches,[i for i in range(len(gt)) if i not in usedg],[j for j in range(len(pred)) if j not in usedp]

def average_precision_101(recall,precision):
    if not len(recall):return 0.
    envelope=np.maximum.accumulate(np.asarray(precision)[::-1])[::-1]
    thresholds=np.linspace(0,1,101);idx=np.searchsorted(recall,thresholds,side='left')
    values=np.zeros(101);valid=idx<len(envelope);values[valid]=envelope[idx[valid]]
    return float(values.mean())

def ap_report(truth,predictions,nc,max_det=300):
    """COCO-style 101-point AP, IoU .50:.95, all areas. No crowd/ignore-region semantics.
    Each image is capped at max_det before scoring. Undefined absent-class AP is null.
    This is not advertised as the full official COCO evaluator.
    """
    ths=np.round(np.arange(.5,.951,.05),2);result=[]
    capped={k:np.asarray(v).reshape(-1,6)[np.argsort(-np.asarray(v).reshape(-1,6)[:,4],kind='stable')[:max_det]] for k,v in predictions.items()}
    for c in range(nc):
        gt={k:v[v[:,4].astype(int)==c,:4] for k,v in truth.items()}
        ng=sum(len(v) for v in gt.values());aps=[]
        cand=[]
        for k,a in capped.items():
            cand.extend((float(r[4]),k,j,r[:4]) for j,r in enumerate(a) if int(r[5])==c)
        cand.sort(key=lambda t:(-t[0],t[1],t[2]))
        for threshold in ths:
            used={k:set() for k in gt};tp=[];fp=[]
            for score,k,j,box in cand:
                overlaps=iou_matrix(np.asarray(box).reshape(1,4),gt[k])[0]
                available=[i for i in range(len(overlaps)) if i not in used[k] and overlaps[i]>=threshold]
                if available:
                    ix=max(available,key=lambda i:overlaps[i]);used[k].add(ix);tp.append(1);fp.append(0)
                else:tp.append(0);fp.append(1)
            if ng:
                ct=np.cumsum(tp);cf=np.cumsum(fp);aps.append(average_precision_101(ct/ng,ct/np.maximum(ct+cf,1)))
            else:aps.append(np.nan)
        result.append({'class_id':c,'ground_truth':ng,'AP50':aps[0], 'AP50_95':float(np.mean(aps)),
                       **{f'AP{int(t*100)}':v for t,v in zip(ths,aps)}})
    present=[r for r in result if r['ground_truth']>0]
    return {'mAP50':float(np.mean([r['AP50'] for r in present])) if present else None,
            'mAP50_95':float(np.mean([r['AP50_95'] for r in present])) if present else None,
            'per_class':result,'max_det':max_det,'definition':'101-point AP, IoU .50:.95, no ignore/crowd labels'}

def fixed_metrics(truth,predictions,names,conf=.25,metadata=None):
    n=len(names);cm=np.zeros((n+1,n+1),int);per_image=[];examples=[]
    for key,gt in truth.items():
        a=np.asarray(predictions.get(key,[]),float).reshape(-1,6);a=a[a[:,4]>=conf]
        matched,missing,extra=spatial_matches(gt[:,:4],a[:,:4],.5)
        local=np.zeros_like(cm);wrong=0
        for i,j,v in matched:
            tc,pc=int(gt[i,4]),int(a[j,5]);local[tc,pc]+=1;wrong+=tc!=pc
        for i in missing:local[int(gt[i,4]),n]+=1
        for j in extra:local[n,int(a[j,5])]+=1
        cm+=local
        # A duplicate candidate is an unmatched prediction overlapping already matched GT.
        # It is not proof of a duplicate when ground truth itself is incomplete.
        dup=sum(any(iou_matrix(a[j:j+1,:4],gt[i:i+1,:4])[0,0]>=.5 for i,_,_ in matched) for j in extra)
        count=len(a)-len(gt);md=(metadata or {}).get(key,{})
        ov=iou_matrix(gt[:,:4],gt[:,:4]);np.fill_diagonal(ov,0)
        crowded=len(gt)>=8 or (ov.size and ov.max()>=.1)
        per_image.append({'image':key,'gt_items':len(gt),'pred_items':len(a),'count_error':count,'abs_count_error':abs(count),
              'localized':len(matched),'missed':len(missing),'extra':len(extra),'wrong_grades':int(wrong),'duplicate_candidates':int(dup),
              'crowded_proxy':bool(crowded),'leakage_group':md.get('leakage_group','UNKNOWN'),
              'confusion':local.tolist(),
              **{f'count_error_{g}':int((a[:,5]==i).sum()-(gt[:,4]==i).sum()) for i,g in enumerate(names)}})
        examples.append({'image':key,'missed_gt_indices':missing,'unmatched_prediction_indices':extra,
                         'wrong_grade_pairs':[(i,j) for i,j,_ in matched if int(gt[i,4])!=int(a[j,5])],
                         'review_score':len(missing)+len(extra)+wrong,'threshold':conf})
    cls=[]
    for i,name in enumerate(names):
        tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp)
        # End-to-end grade metrics include missed objects and spurious objects.
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if tp+fn else None
        cls.append({'grade':name,'TP':tp,'FP':fp,'FN':fn,'precision':precision,'recall':recall,'F1':f1,'support':int(cm[i,:].sum())})
    localized=int(cm[:n,:n].sum());total_gt=int(cm[:n,:].sum());total_pred=int(cm[:,:n].sum())
    mae=float(np.mean([x['abs_count_error'] for x in per_image])) if per_image else None
    report={'confidence_threshold':conf,'matching_iou':.5,'class_agnostic_precision':localized/total_pred if total_pred else 0.,
            'class_agnostic_recall':localized/total_gt if total_gt else None,
            'grade_accuracy_given_localized':int(np.trace(cm[:n,:n]))/localized if localized else None,
            'macro_joint_grade_F1':float(np.mean([x['F1'] for x in cls if x['F1'] is not None])) if any(x['F1'] is not None for x in cls) else None,
            'count_MAE':mae,'count_bias':float(np.mean([x['count_error'] for x in per_image])) if per_image else None,
            'count_RMSE':float(np.sqrt(np.mean([x['count_error']**2 for x in per_image]))) if per_image else None,
            'per_class':cls,'confusion_gt_rows_pred_columns':cm.tolist(),'matrix_labels':list(names)+['BACKGROUND'],
            'n_images':len(per_image),'gt_instances':total_gt,'pred_instances':total_pred}
    return report,per_image,sorted(examples,key=lambda e:-e['review_score'])

def grouped_count_interval(per_image,seed=42,repeats=1000):
    groups=defaultdict(list)
    for r in per_image:groups[r['leakage_group']].append(r['abs_count_error'])
    keys=list(groups)
    if 'UNKNOWN' in groups or len(keys)<2:return {'available':False,'reason':'Insufficient known independent groups.'}
    rng=np.random.default_rng(seed);means=[]
    for _ in range(repeats):
        chosen=rng.choice(keys,len(keys),replace=True);vals=[x for k in chosen for x in groups[k]];means.append(np.mean(vals))
    return {'available':True,'n_groups':len(keys),'count_MAE_cluster_bootstrap_95_percentile':np.percentile(means,[2.5,97.5]).tolist(),
            'caution':'Very few groups; interval unstable' if len(keys)<5 else 'Independence depends on correct group metadata.'}

def truth_from_entries(entries):
    return {e['id']:np.column_stack([xywh_to_xyxy(e['labels'],e['width'],e['height']),e['labels'][:,0]]) for e in entries}

def evaluate_predictions(entries,preds,out,policy,names=GRADES):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);truth=truth_from_entries(entries)
    pred={k:np.asarray(preds.get(k,[]),float).reshape(-1,6) for k in truth}
    m,per,examples=fixed_metrics(truth,pred,names,policy['conf'],{e['id']:e for e in entries})
    ap=ap_report(truth,pred,len(names),policy['max_det']);m['AP_per_class']=ap.pop('per_class');m.update(ap)
    locgt={k:np.column_stack([v[:,:4],np.zeros(len(v))]) for k,v in truth.items()}
    locpred={k:v.copy() for k,v in pred.items()}
    for v in locpred.values():v[:,5]=0
    m['localization_AP']=ap_report(locgt,locpred,1,policy['max_det'])
    m['count_uncertainty']=grouped_count_interval(per)
    write_json(out/'metrics.json',m);write_json(out/'failure_examples.json',examples)
    pd.DataFrame(m['per_class']).to_csv(out/'per_class_detection_grade.csv',index=False)
    pd.DataFrame(per).drop(columns='confusion').to_csv(out/'per_image_and_counts.csv',index=False)
    pd.DataFrame(m['confusion_gt_rows_pred_columns'],index=m['matrix_labels'],columns=m['matrix_labels']).to_csv(out/'confusion_matrix.csv')
    cm=np.asarray(m['confusion_gt_rows_pred_columns'])[:-1,:-1];conditional=[]
    for i,name in enumerate(names):
        tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp)
        conditional.append({'grade':name,'precision_given_localization':tp/(tp+fp) if tp+fp else 0.,
            'recall_given_localization':tp/(tp+fn) if tp+fn else None,
            'F1_given_localization':2*tp/(2*tp+fp+fn) if tp+fn else None,
            'matched_support':int(cm[i].sum()),'caution':'Excludes missed/spurious objects; not end-to-end performance.'})
    pd.DataFrame(conditional).to_csv(out/'conditional_grading_only_localized.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6));matrix=np.asarray(m['confusion_gt_rows_pred_columns'])
    handle=ax.imshow(matrix);fig.colorbar(handle,ax=ax)
    ax.set_xticks(range(len(names)+1),m['matrix_labels'],rotation=45,ha='right')
    ax.set_yticks(range(len(names)+1),m['matrix_labels']);ax.set(xlabel='Predicted',ylabel='Ground truth',title='IoU-matched object / grade confusion')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=140);plt.close(fig)
    # Crowded-only mAP and metrics, never class an annotation overlap as proven physical occlusion.
    for name,test in [('crowded',[x['image'] for x in per if x['crowded_proxy']]),('less_crowded',[x['image'] for x in per if not x['crowded_proxy']])]:
        if test:
            sub={k:truth[k] for k in test};sp={k:pred[k] for k in test};sm,_,_=fixed_metrics(sub,sp,names,policy['conf'])
            sa=ap_report(sub,sp,len(names),policy['max_det']);sm['AP_per_class']=sa.pop('per_class');sm.update(sa);write_json(out/(name+'_metrics.json'),sm)
    # Visual failure cases use real model outputs, not attractive cherry-picked demos.
    byid={e['id']:e for e in entries}
    for j,x in enumerate(examples[:12]):
        e=byid[x['image']];a=pred[e['id']];a=a[a[:,4]>=policy['conf']]
        draw_predictions(e['path'],a,names,'fish_instance' if len(names)==4 else 'legacy_parts',out/'failure_gallery'/f'{j:02d}_pred.jpg',title=x['image'])
        gt=truth[e['id']];ann=np.column_stack([gt[:,:4],np.ones(len(gt)),gt[:,4]])
        draw_predictions(e['path'],ann,names,'fish_instance' if len(names)==4 else 'legacy_parts',out/'failure_gallery'/f'{j:02d}_truth.jpg',title='GROUND TRUTH: '+x['image'])
    return m

# --------------------- EXPLICIT FINETUNING LOOP -------------------------
def transform_training(image,labels,size,flip_x=False,flip_y=False,rotations=0):
    h,w=image.shape[:2];scale=min(size/w,size/h);nw,nh=round(w*scale),round(h*scale)
    dx,dy=(size-nw)//2,(size-nh)//2
    im=np.full((size,size,3),114,np.uint8);im[dy:dy+nh,dx:dx+nw]=cv2.resize(image,(nw,nh))
    a=labels.copy();a[:,1]=(a[:,1]*nw+dx)/size;a[:,2]=(a[:,2]*nh+dy)/size;a[:,3]*=nw/size;a[:,4]*=nh/size
    if flip_x:im=im[:,::-1];a[:,1]=1-a[:,1]
    if flip_y:im=im[::-1];a[:,2]=1-a[:,2]
    for _ in range(rotations%4):
        im=np.rot90(im);x=a[:,1].copy();a[:,1]=a[:,2];a[:,2]=1-x;a[:,[3,4]]=a[:,[4,3]]
    return np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)),a

def make_loader(entries,config):
    import torch
    class FishDataset(torch.utils.data.Dataset):
        def __len__(self):return len(entries)
        def __getitem__(self,i):
            e=entries[i];im=cv2.imread(str(e['path']))
            x,a=transform_training(im,e['labels'],config['imgsz'],random.random()<.5,random.random()<.5,
                                   random.randrange(4) if config['rot90'] else 0)
            return torch.from_numpy(x).float()/255.,torch.from_numpy(a)
    def collate(batch):
        x,a=zip(*batch);labels=torch.cat(a)
        return {'img':torch.stack(x),'cls':labels[:,:1],'bboxes':labels[:,1:],
                'batch_idx':torch.cat([torch.full((len(z),),i,dtype=torch.float32) for i,z in enumerate(a)])}
    sampler=None
    if config['class_sampling']:
        freq=Counter(c for e in entries for c in set(e['labels'][:,0].astype(int)))
        weights=[max([1/math.sqrt(freq[c]) for c in set(e['labels'][:,0].astype(int))] or [min(1/math.sqrt(v) for v in freq.values())]) for e in entries]
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(entries),replacement=True)
    return torch.utils.data.DataLoader(FishDataset(),batch_size=config['batch'],shuffle=sampler is None,
        sampler=sampler,num_workers=0,pin_memory=True,collate_fn=collate,drop_last=False)

def train_fish(entries,fingerprint,out,config=None,resume_dir=None):
    """Train reviewed fish INSTANCE boxes. Cannot run on the legacy part-only ZIP."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import ModelEMA
    cfg={**DEFAULT,**(config or {})};out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Training needs a CUDA GPU; select T4 in Colab.')
    train=[e for e in entries if e['split']=='train'];val=[e for e in entries if e['split']=='val']
    if not train or not val:raise ValueError('Real train and validation groups are required. No val=train workaround.')
    if any(len(e['labels']) and (e['labels'][:,0].max()>3 or e['labels'][:,0].min()<0) for e in entries):
        raise ValueError('Fish trainer requires four reviewed instance-grade targets, not part classes.')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    loader=make_loader(train,cfg)
    pretrained=YOLO(cfg['model']);net=DetectionModel(copy.deepcopy(pretrained.model.yaml),ch=3,nc=4,verbose=False)
    net.load(pretrained.model);del pretrained
    net.names=dict(enumerate(GRADES));net.task='detect'
    argdict={'task':'detect','model':cfg['model'],'epochs':cfg['epochs'],'imgsz':cfg['imgsz'],'batch':cfg['batch'],
             'device':cfg['device'],'optimizer':'AdamW','lr0':cfg['lr'],'weight_decay':cfg['weight_decay'],
             'seed':cfg['seed'],'val':True,'nms':True,'max_det':300}
    net.args=get_cfg(overrides=argdict);net=net.to('cuda');net.criterion=net.init_criterion()
    decay=[];nodecay=[]
    for p in net.parameters():
        if p.requires_grad:(decay if p.ndim>1 else nodecay).append(p)
    optim=torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},{'params':nodecay,'weight_decay':0}],lr=cfg['lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp']);ema=ModelEMA(net)
    signature={'protocol':VERSION,'ultralytics':ULTRALYTICS_VERSION,'dataset_sha256':fingerprint,'config':cfg}
    write_json(out/'training_config.json',signature)
    history=[];start=0;best=-1.;bad=0
    if resume_dir:
        # Only load checkpoints YOU trust: torch pickle can execute code.
        old=Path(resume_dir);s=torch.load(old/'resume_state.pt',map_location='cpu',weights_only=False)
        if s['signature']!=signature:raise ValueError('Resume data/config/version mismatch. Start a new pretrained run.')
        net.load_state_dict(s['net']);ema.ema.load_state_dict(s['ema']);ema.updates=s['ema_updates']
        optim.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler']);history=s['history']
        start=s['completed'];best=s['best'];bad=s['bad_checks']
        for _ in range(start):
            if hasattr(net.criterion,'update'):net.criterion.update()
        random.setstate(s['rng_python']);np.random.set_state(s['rng_numpy']);torch.set_rng_state(s['rng_torch']);torch.cuda.set_rng_state_all(s['rng_cuda'])
        if (old/'best_val.pt').exists():shutil.copy2(old/'best_val.pt',out/'best_val.pt')
        del s
    def save_weights(completed):
        ema.update_attr(net,include=['yaml','nc','args','names','stride'])
        snapshot=copy.deepcopy(ema.ema).cpu().float().eval();snapshot.criterion=None
        atomic_torch_save({'epoch':completed-1,'model':snapshot,'ema':None,'train_args':argdict,'train_metrics':{},
                          'version':ULTRALYTICS_VERSION,'date':datetime.now(timezone.utc).isoformat()},out/'last.pt')
    def save_resume(completed):
        atomic_torch_save({'signature':signature,'net':net.state_dict(),'ema':ema.ema.state_dict(),'ema_updates':ema.updates,
                          'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),'history':history,'completed':completed,'best':best,'bad_checks':bad,
                          'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()},out/'resume_state.pt')
    from tqdm.auto import tqdm
    for ep in range(start,cfg['epochs']):
        net.train();optim.zero_grad(set_to_none=True);total=0.;seen=0;components=Counter();updates=0;t0=time.time()
        gains={'o2m_gain':float(getattr(net.criterion,'o2m',1)), 'o2o_gain':float(getattr(net.criterion,'o2o',0))}
        for step,batch in enumerate(tqdm(loader,desc=f'Fish-instance epoch {ep+1}/{cfg["epochs"]}')):
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};n=batch['img'].shape[0]
            fraction=ep+step/len(loader);warm=cfg['warmup_epochs']
            f=min(1,(fraction+1/max(len(loader),1))/max(warm,1e-9)) if fraction<warm else .1+.9*(1+math.cos(math.pi*(fraction-warm)/max(cfg['epochs']-warm,1)))/2
            for group in optim.param_groups:group['lr']=cfg['lr']*f
            block=(step//cfg['accumulate'])*cfg['accumulate'];block_images=min((block+cfg['accumulate'])*cfg['batch'],len(train))-block*cfg['batch']
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                loss,items=net(batch);meanloss=loss.sum()/n;scaled=meanloss*n/block_images
            if not torch.isfinite(meanloss):raise RuntimeError('Nonfinite loss. Preserve checkpoint, inspect data/AMP; do not continue silently.')
            scaler.scale(scaled).backward()
            if (step+1)%cfg['accumulate']==0 or step+1==len(loader):
                scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
                oldscale=scaler.get_scale();scaler.step(optim);scaler.update()
                if scaler.get_scale()>=oldscale:ema.update(net);updates+=1
                optim.zero_grad(set_to_none=True)
            total+=float(meanloss.detach())*n;seen+=n
            if isinstance(items,dict):
                for k,v in items.items():components['one2one_'+k]+=float(v.detach())*n
            elif torch.is_tensor(items):
                for i,v in enumerate(items.detach().flatten()):components[f'one2one_component_{i}']+=float(v)*n
        if updates==0:raise RuntimeError('No successful optimizer steps.')
        row={'epoch':ep+1,'weighted_training_loss':total/seen,'lr':optim.param_groups[0]['lr'],'optimizer_steps':updates,
             'seconds':time.time()-t0,**gains,**{k:v/seen for k,v in components.items()}}
        if hasattr(net.criterion,'update'):net.criterion.update()
        check=ep==0 or (ep+1)%cfg['validate_every']==0 or ep+1==cfg['epochs']
        if check:
            save_weights(ep+1)
            # Validation uses only the fixed validation split; test labels never select weights.
            P=Predictor(out/'last.pt','fish_instance',cfg['device']);pol={**BASE_INFER,'imgsz':cfg['imgsz']}
            predictions={e['id']:P(e['path'],pol,candidate_floor=.001)[0] for e in val}
            metrics=evaluate_predictions(val,predictions,out/'validation'/f'epoch_{ep+1:03d}',pol)
            row['val_mAP50_95']=metrics['mAP50_95'];row['val_macro_joint_F1']=metrics['macro_joint_grade_F1']
            score=metrics['mAP50_95']
            if score is not None and score>best+1e-6:
                best=score;bad=0;shutil.copy2(out/'last.pt',out/'best_val.pt')
            else:bad+=1
            del P,predictions;torch.cuda.empty_cache()
        history.append(row);pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        if check:save_resume(ep+1)
        print(row)
        if check and bad>=cfg['patience_checks']:print('Stopping by validation patience, not by training loss.');break
    write_json(out/'training_complete.json',{'dataset':fingerprint,'best_validation_mAP':best,'independent_test_run':False,
             'selection':'validation mAP50-95','no_claim_of_improvement':True})
    del net,optim,ema,scaler;torch.cuda.empty_cache()
    return out/'best_val.pt'

# ------------------------ FEATURE ABLATIONS -----------------------------
def feature_table(entries,out):
    rows=[];out=Path(out)
    for e in entries:
        im=cv2.imread(str(e['path']));boxes=xywh_to_xyxy(e['labels'],e['width'],e['height'])
        ov=iou_matrix(boxes,boxes);np.fill_diagonal(ov,0)
        for j,b in enumerate(boxes):
            x1,y1,x2,y2=np.rint(b).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(e['width'],x2);y2=min(e['height'],y2)
            clipped=x1<=1 or y1<=1 or x2>=e['width']-1 or y2>=e['height']-1
            f,_=region_features(im[y1:y2,x1:x2],float(ov[j].max()) if ov.size else 0,clipped)
            if f:
                rows.append({'image':e['id'],'object_index':j,'grade_id':int(e['labels'][j,0]),'split':e['split'],
                             'leakage_group':e['leakage_group'],**f})
    table=pd.DataFrame(rows);table.to_csv(out/'instance_feature_measurements.csv',index=False);return table

def train_feature_ablations(entries,out):
    """Train new whole-instance classifiers only on training GT crops.
    Grade re-evaluation must use DETECTED crops, not perfect validation GT crops.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    out=Path(out);table=feature_table([e for e in entries if e['split']=='train'],out);bundles={}
    if table.empty:
        write_json(out/'feature_branch_skipped.json',{'reason':'No regions pass the foreground checks.'});return {}
    for name,columns in FEATURE_SETS.items():
        train=table.copy()
        requires_shape = name in ['color_texture_on_shape_eligible','color_texture_shape']
        if requires_shape:train=train[train.shape_usable_proxy==True]
        train=train[np.isfinite(train[columns].to_numpy(float)).all(axis=1)].copy()
        if set(train.grade_id)!=set(range(4)):
            write_json(out/f'feature_{name}_skipped.json',{'reason':'Insufficient valid training regions across all four grades.'});continue
        # Equal total weight per grade and capture group within grade, not per repeated crop.
        per_group=train.groupby(['leakage_group','grade_id']).grade_id.transform('size')
        n_groups=train.groupby('grade_id').leakage_group.transform('nunique')
        weights=1/(per_group*n_groups);weights/=weights.mean()
        clf=RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=5,n_jobs=-1,random_state=42)
        clf.fit(train[columns],train.grade_id,sample_weight=weights)
        bundle={'classifier':clf,'features':columns,'feature_set':name,'unit':'fish_instance','extractor_version':VERSION,
                'validated':False,'requires_shape_mask':requires_shape,'mask':'threshold approximation, unsuitable for some backgrounds'}
        joblib.dump(bundle,out/f'feature_{name}.joblib');bundles[name]=bundle
    return bundles

def regrade_predictions(entries,predictions,bundle):
    """Alternative grader on fixed detector boxes. Never invent additional objects.
    Retains detector confidence as ranking score, NOT a calibrated grade probability.
    Shape model falls back to detector when silhouette/overlap checks fail.
    """
    result={};assessments=[]
    for e in entries:
        im=cv2.imread(str(e['path']));a=np.asarray(predictions[e['id']]).copy().reshape(-1,6)
        ov=iou_matrix(a[:,:4],a[:,:4]);np.fill_diagonal(ov,0)
        for j,row in enumerate(a):
            # Duplicate low-score proposals should not create a false mask-overlap alarm.
            eligible=np.where((a[:,4]>=.25)&(np.arange(len(a))!=j))[0]
            overlap=float(ov[j,eligible].max()) if len(eligible) else 0.
            x1,y1,x2,y2=np.rint(row[:4]).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(im.shape[1],x2);y2=min(im.shape[0],y2)
            f,_=region_features(im[y1:y2,x1:x2],overlap,x1<=1 or y1<=1 or x2>=im.shape[1]-1 or y2>=im.shape[0]-1)
            valid=f is not None and all(f.get(k) is not None and np.isfinite(f[k]) for k in bundle['features'])
            if valid and bundle.get('requires_shape_mask',False):valid=f['shape_usable_proxy']
            record={'image':e['id'],'detection':j,'old_grade':int(row[5]),'method':'detector_fallback'}
            if valid:
                z=pd.DataFrame([{k:f[k] for k in bundle['features']}]);probs=bundle['classifier'].predict_proba(z)[0]
                k=int(np.argmax(probs));grade=int(bundle['classifier'].classes_[k]);a[j,5]=grade
                record.update(method='feature_grader',new_grade=grade,feature_score_uncalibrated=float(probs[k]))
            assessments.append(record)
        result[e['id']]=a
    return result,assessments

# -------------------------- VALIDATION PLAN -----------------------------
def validation_experiments(entries,checkpoint,fingerprint,out,feature_bundles=None):
    """Choose settings ONLY using validation data. Export a locked deployment policy."""
    out=Path(out);val=[e for e in entries if e['split']=='val']
    if not val:raise ValueError('No independent validation split.')
    variants={
        'baseline640':{**BASE_INFER},
        'resolution960':{**BASE_INFER,'imgsz':960},
        'one_to_one640':{**BASE_INFER,'head':'one'},
        'exclusive_grade_merge':{**BASE_INFER,'exclusive_grades':True},
        'nms_iou055':{**BASE_INFER,'iou':.55},
        'nms_iou085':{**BASE_INFER,'iou':.85},
    }
    if any(max(e['width'],e['height'])>960 for e in val):
        variants['tiles_plus_full']={**BASE_INFER,'tiles':True,'tile_size':640}
    else:write_json(out/'tiling_not_run.json',{'reason':'No validation source exceeds 960 pixels; no automatic tiling of low-resolution legacy exports.'})
    P=Predictor(checkpoint);all_scores=[];raw_cache={};best=None
    for name,pol in variants.items():
        # Untimed warm-up prevents the first policy paying model/backend startup in the speed comparison.
        P(val[0]['path'],pol,candidate_floor=.001)
        raw={};runtime=[]
        for e in val:
            raw[e['id']],info=P(e['path'],pol,candidate_floor=.001);runtime.append(info['seconds_including_prepost'])
        raw_cache[name]=raw
        for threshold in [.05,.1,.25,.4]:
            selected={**pol,'conf':threshold};label=f'{name}_conf{threshold:.2f}'
            m=evaluate_predictions(val,raw,out/'validation_sweep'/label,selected)
            score={'experiment':label,'variant':name,'feature_set':None,'conf':threshold,
                   'macro_joint_F1':m['macro_joint_grade_F1'],'mAP50_95':m['mAP50_95'],'count_MAE':m['count_MAE'],
                   'mean_seconds':float(np.mean(runtime)),'policy':selected}
            all_scores.append(score)
    # Controlled regrading: same baseline detections / confidence / geometry.
    for fname,bundle in (feature_bundles or {}).items():
        changed,assess=regrade_predictions(val,raw_cache['baseline640'],bundle)
        write_json(out/f'feature_{fname}_validation_assessments.json',assess)
        for threshold in [.05,.1,.25,.4]:
            pol={**BASE_INFER,'conf':threshold};label=f'baseline640_{fname}_conf{threshold:.2f}'
            m=evaluate_predictions(val,changed,out/'validation_sweep'/label,pol)
            all_scores.append({'experiment':label,'variant':'baseline640','feature_set':fname,'conf':threshold,
                               'macro_joint_F1':m['macro_joint_grade_F1'],'mAP50_95':m['mAP50_95'],'count_MAE':m['count_MAE'],
                               'policy':pol})
    # Explicit selection objective. Recall/precision/count tradeoffs remain visible.
    best=max(all_scores,key=lambda s:(s['macro_joint_F1'] if s['macro_joint_F1'] is not None else -1,-s['count_MAE']))
    lock={'protocol':VERSION,'unit':'fish_instance','dataset_sha256':fingerprint,'weights_sha256':sha256(checkpoint),
          'selected_on':'validation ONLY','selection_metric':'macro end-to-end grade F1, then count MAE',
          'chosen':best,'test_used':False}
    if best['feature_set']:
        file=out/f'feature_{best["feature_set"]}.joblib'
        lock['feature_file']=str(file);lock['feature_sha256']=sha256(file)
    write_json(out/'deployment_policy_LOCKED.json',lock)
    pd.DataFrame([{k:v for k,v in s.items() if k!='policy'} for s in all_scores]).to_csv(out/'validation_experiments.csv',index=False)
    return lock

def final_test(entries,checkpoint,fingerprint,lock_path,out):
    """Explicit opt-in, frozen config. The lock is not a guarantee humans have never seen test data."""
    import joblib
    lock_path=Path(lock_path);lock=json.loads(lock_path.read_text());out=Path(out)
    if lock['dataset_sha256']!=fingerprint or lock['weights_sha256']!=sha256(checkpoint):raise ValueError('Test dataset/weights differ from validation lock.')
    if lock.get('test_used') or (lock_path.parent/'TEST_ALREADY_RUN.json').exists():raise ValueError('This lock has already been tested. Preserve the recorded evaluation; do not tune on test.')
    test=[e for e in entries if e['split']=='test'];policy=lock['chosen']['policy'];P=Predictor(checkpoint)
    raw={e['id']:P(e['path'],policy,candidate_floor=.001)[0] for e in test}
    if lock['chosen']['feature_set']:
        f=Path(lock['feature_file'])
        if sha256(f)!=lock['feature_sha256']:raise ValueError('Feature model differs from validated model.')
        raw,_=regrade_predictions(test,raw,joblib.load(f))
    m=evaluate_predictions(test,raw,out/'final_test',policy)
    write_json(lock_path.parent/'TEST_ALREADY_RUN.json',{'time':datetime.now(timezone.utc).isoformat(),'locked_policy':str(lock_path),'results':str(out/'final_test')})
    return m

# ------------------- AUTOMATIC ANNOTATION DRAFTS -------------------------
def draft_fish_boxes(root,out):
    """Annotation-assistance ONLY. Geometry cannot prove ownership in overlap.
    Never writes trainable YOLO fish labels. Every proposal is reviewed=False.
    Unassigned heads/tails remain candidates, not automatically separate fish.
    """
    out=Path(out);images=[];annotations=[];review=[];aid=1
    for image_id,p in enumerate(image_paths(Path(root)/'train/images'),1):
        im=cv2.imread(str(p));h,w=im.shape[:2]
        labels=read_boxes(Path(root)/'train/labels'/f'{p.stem}.txt',12);boxes=xywh_to_xyxy(labels,w,h)
        body=list(np.where(labels[:,0].astype(int)%3==0)[0]);attachments={i:[] for i in body};used=set(body)
        ambiguities=defaultdict(list)
        for part in [1,2]:
            ids=list(np.where(labels[:,0].astype(int)%3==part)[0])
            if not body or not ids:continue
            cost=np.full((len(body),len(ids)),1e5,float)
            for bi,i in enumerate(body):
                a=boxes[i];length=max(a[2]-a[0],a[3]-a[1]);center=(a[:2]+a[2:])/2
                for pi,j in enumerate(ids):
                    b=boxes[j]
                    if int(labels[i,0])//3!=int(labels[j,0])//3:continue
                    gap=np.linalg.norm(np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:])))
                    distance=np.linalg.norm((b[:2]+b[2:])/2-center)/max(length,1)
                    if gap<=.25*length and distance<=1.2:cost[bi,pi]=distance+gap/max(length,1)
            for bi,pi in zip(*linear_sum_assignment(cost)):
                if cost[bi,pi]>=1e4:continue
                sortedrow=np.sort(cost[bi]);sortedcol=np.sort(cost[:,pi]);margin=min(sortedrow[1]-sortedrow[0] if len(sortedrow)>1 else 1e5,
                                                                                  sortedcol[1]-sortedcol[0] if len(sortedcol)>1 else 1e5)
                i,j=body[bi],ids[pi]
                if margin<.12:ambiguities[i].append('ambiguous nearby '+('head' if part==1 else 'tail'));continue
                attachments[i].append(j);used.add(j)
        groups=[]
        for i in body:
            groups.append((i,[i]+attachments[i], 'body_anchor_candidate'))
        for j in range(len(labels)):
            if j not in used:groups.append((j,[j],'unassigned_part_or_fragment'))
        images.append({'id':image_id,'file_name':p.name,'width':w,'height':h})
        for anchor,ids,status in groups:
            bb=boxes[ids];a=np.r_[bb[:,:2].min(0),bb[:,2:].max(0)]
            attr={'reviewed':False,'proposal_only':True,'status':status,'source_part_rows':ids,
                  'notes':ambiguities[anchor]+['Verify same physical fish, whole extent, grade, fragments and missing items.']}
            annotations.append({'id':aid,'image_id':image_id,'category_id':int(labels[anchor,0])//3+1,
                                'bbox':[float(a[0]),float(a[1]),float(a[2]-a[0]),float(a[3]-a[1])],
                                'area':float((a[2]-a[0])*(a[3]-a[1])),'iscrowd':0,'attributes':attr})
            review.append({'proposal_id':aid,'image':p.name,'grade_hint':GRADES[int(labels[anchor,0])//3],
                           'status':status,'linked_part_count':len(ids),'reviewed':False})
            aid+=1
    coco={'info':{'description':'UNREVIEWED DRAFTS. NOT ground truth. Never train directly.'},'images':images,'annotations':annotations,
          'categories':[{'id':i+1,'name':n} for i,n in enumerate(GRADES)]}
    write_json(out/'DRAFT_NOT_TRAINING_fish_boxes.coco.json',coco)
    pd.DataFrame(review).to_csv(out/'DRAFT_review_queue.csv',index=False)
    return coco

# -------------------------- COLAB INTERFACE -----------------------------
def upload_file(suffix,prompt_text):
    from google.colab import files
    print(prompt_text)
    while True:
        uploaded=files.upload();found=[Path(name).resolve() for name in uploaded if name.lower().endswith(suffix)]
        del uploaded
        if len(found)==1:return found[0]
        print(f'Please select exactly ONE {suffix} file, not a verification JSON or notebook.')

def acquire_legacy():
    for p in Path('/content').glob('Tamban_v9_Cleaned_Training_Pool*.zip'):
        if sha256(p)==LEGACY_HASH:return p
    p=upload_file('.zip','Upload Tamban_v9_Cleaned_Training_Pool.zip only.')
    if sha256(p)!=LEGACY_HASH:raise ValueError('This audit expects the original cleaned legacy ZIP. Use fish-data mode for NEW reviewed data.')
    return p

def checkpoint_metadata(checkpoint,unit,out):
    """Inspect only your own trusted checkpoint, without editing it."""
    from ultralytics import YOLO
    m=YOLO(str(checkpoint));head=m.model.model[-1]
    d={'path':str(checkpoint),'sha256':sha256(checkpoint),'names':m.names,'task':m.task,
       'parameters':sum(p.numel() for p in m.model.parameters()),'head_class':type(head).__name__,
       'has_one_to_one':getattr(head,'one2one_cv2',None) is not None,
       'has_one_to_many':getattr(head,'cv2',None) is not None,'unit':unit}
    expected=PART_NAMES if unit=='legacy_parts' else GRADES
    if list(m.names.values())!=expected:raise ValueError('Checkpoint annotation target does not match this mode.')
    write_json(Path(out)/'checkpoint_inspection.json',d);return d

def archive_outputs(folder,destination):
    folder=Path(folder);destination=Path(destination)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file() and p.name!='resume_state.pt' and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(folder))
    return destination

def colab_main():
    """One embedded code cell. Uploads DATA/model files, never an external Python launcher."""
    from google.colab import drive,files
    from IPython.display import display,Image
    install_colab_dependencies();drive.mount('/content/drive')
    base=Path('/content/drive/MyDrive/Dried_Fish_Models')
    print('1 = audit existing part model/data (NO retraining; safe starting point)')
    print('2 = train reviewed whole-fish instance data + tune on validation (NO test)')
    print('3 = run FINAL TEST using an existing locked policy (explicit opt-in)')
    print('4 = predict new photos using a locked whole-fish model/policy')
    mode=input('Choose 1, 2, 3 or 4 [default 1]: ').strip() or '1'
    out=unique_run(base,'tamban_v3_'+mode)
    work=unique_run('/content','tamban_v3_work')
    write_json(out/'environment.json',{'protocol':VERSION,'ultralytics':ULTRALYTICS_VERSION,'python':sys.version})
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    if mode=='1':
        z=acquire_legacy();root=dataset_root(safe_extract(z,work/'legacy'))
        R,A,summary=audit_legacy(root,out);print(json.dumps(summary,indent=2))
        draft_fish_boxes(root,out)
        old=Path('/content/drive/MyDrive/Dried_Fish_Models/tamban_yolo26_complete_v2/last.pt')
        typed=input(f'Existing trained checkpoint path [Enter uses {old}]: ').strip()
        if typed:old=Path(typed)
        if not old.exists():
            print('No checkpoint at that path. Audit and annotation drafts saved; model performance was NOT evaluated.')
        else:
            checkpoint_metadata(old,'legacy_parts',out)
            P=Predictor(old,'legacy_parts')
            chosen=[]
            for grade in GRADES:
                subset=R[R.grades==grade].sort_values('max_body_box_iou',ascending=False).head(2)
                chosen.extend(Path(x) for x in subset.image)
            for j,p in enumerate(chosen):
                pol={**BASE_INFER,'conf':.25,'max_det':100} # Original policy for baseline replay.
                pred,info=P(p,pol,legacy_export=True)
                target=out/'legacy_training_demos'/f'{j:02d}_{p.stem}.jpg'
                counts=draw_predictions(p,pred,PART_NAMES,'legacy_parts',target,'TRAINING DEMO - no independent accuracy')
                write_json(target.with_suffix('.json'),{'predictions':pred,'runtime':info,'counts':counts})
                if j<4:display(Image(filename=str(target),width=850))
            print('These are part-model TRAINING demos. There are no verified whole-fish predictions or independent metrics.')
            if input('Inspect new photos with this PART model? y/N: ').strip().lower()=='y':
                uploaded=files.upload()
                for name in uploaded:
                    if Path(name).suffix.lower() not in IMAGE_EXT:continue
                    pred,info=P(name,BASE_INFER)
                    target=out/'new_photo_part_demos'/(Path(name).stem+'.jpg')
                    counts=draw_predictions(name,pred,PART_NAMES,'legacy_parts',target)
                    write_json(target.with_suffix('.json'),{'predictions':pred,'runtime':info,'counts':counts})
                    display(Image(filename=str(target),width=850))
                del uploaded
    elif mode=='2':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Select Runtime > Change runtime type > T4 GPU.')
        z=upload_file('.zip','Upload reviewed FISH-INSTANCE dataset ZIP (not the 12-part legacy ZIP).')
        root=dataset_root(safe_extract(z,work/'fish_input'))
        entries,fingerprint=prepare_fish_data(root,work,out)
        resume=input('Optional V3 run folder to resume SAME data/settings [Enter starts pretrained]: ').strip()
        ckpt=train_fish(entries,fingerprint,out,DEFAULT,resume_dir=resume or None)
        bundles={}
        if input('Run controlled feature-grading ablations? y/N: ').strip().lower()=='y':bundles=train_feature_ablations(entries,out)
        lock=validation_experiments(entries,ckpt,fingerprint,out,bundles)
        print('Training and validation complete. Final test NOT run.')
        print('Chosen validation policy:',lock['chosen'])
    elif mode in ['3','4']:
        checkpoint=Path(input('Path to the V3 best_val.pt checkpoint: ').strip())
        lock_path=Path(input('Path to deployment_policy_LOCKED.json: ').strip())
        lock=json.loads(lock_path.read_text())
        if sha256(checkpoint)!=lock['weights_sha256']:raise ValueError('Checkpoint differs from the validated policy.')
        if mode=='3':
            if input('Type FINAL TEST to use the untouched test set now: ').strip()!='FINAL TEST':return out
            z=upload_file('.zip','Upload the SAME reviewed fish-instance dataset used in validation.')
            root=dataset_root(safe_extract(z,work/'fish_input'));entries,fingerprint=prepare_fish_data(root,work,out)
            print(final_test(entries,checkpoint,fingerprint,lock_path,out))
        else:
            import joblib
            P=Predictor(checkpoint);bundle=None
            if lock['chosen']['feature_set']:
                p=Path(lock['feature_file'])
                if sha256(p)!=lock['feature_sha256']:raise ValueError('Feature model hash mismatch.')
                bundle=joblib.load(p)
            print('Upload images for individual fish/item grading. No annotations required for inference.')
            uploaded=files.upload()
            for name in uploaded:
                if Path(name).suffix.lower() not in IMAGE_EXT:continue
                a,info=P(name,lock['chosen']['policy'],candidate_floor=.001)
                assess=[]
                if bundle:
                    im=cv2.imread(name);e={'id':name,'path':Path(name),'width':im.shape[1],'height':im.shape[0]}
                    result,assess=regrade_predictions([e],{name:a},bundle);a=result[name]
                a=a[a[:,4]>=lock['chosen']['policy']['conf']]
                target=out/'predictions'/(Path(name).stem+'.jpg')
                counts=draw_predictions(name,a,GRADES,'fish_instance',target)
                write_json(target.with_suffix('.json'),{'predictions':a,'counts':counts,'runtime':info,
                           'feature_assessments':assess, 'grading_method':lock['chosen']['feature_set'] or 'detector_only',
                           'note':'Displayed confidence is detector score, not calibrated fused-grade probability. Rejected is a grade, not explicit defect localization.'})
                display(Image(filename=str(target),width=900))
            del uploaded
    else:raise ValueError('Choose one of the four modes.')
    z=archive_outputs(out,Path('/content')/(out.name+'_results.zip'));shutil.copy2(z,out.parent/z.name)
    print('Original files/checkpoints untouched. Outputs:',out)
    files.download(str(z));return out

if __name__=='__main__':
    try: in_colab=importlib.util.find_spec('google.colab') is not None
    except ModuleNotFoundError: in_colab=False
    if in_colab:
        OUTPUT_DIRECTORY=colab_main()
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--audit-legacy',type=Path);parser.add_argument('--out',type=Path,default=Path('tamban_v3_local_audit'))
        args=parser.parse_args()
        if args.audit_legacy:
            args.out.mkdir(parents=True,exist_ok=True);root=dataset_root(safe_extract(args.audit_legacy,args.out/'data'));audit_legacy(root,args.out)
        else:print('Open the accompanying Colab notebook, or import these functions for tests.')
