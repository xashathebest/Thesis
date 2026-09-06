"""Generate evidence-only comparison CSV and deterministic Markdown report."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from src.evaluation.comparison_protocol import MODELS, METRICS, PROTOCOL, fingerprint, write_csv, write_json
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root

FIELDS = ['Model','mAP50','mAP50_95','Precision','Recall','F1','InferenceTime','FPS','ModelSize',
          'variant','task','input_size','epochs_completed','best_epoch','mask_precision','mask_recall','mask_map50',
          'mask_map50_95','median_inference_ms','parameter_count','training_time_minutes','evaluation_split','status','notes']
NUMERIC = FIELDS[1:9]+FIELDS[11:21]


def result_path(root, model):
    if model not in MODELS:
        raise ValueError('Unknown model family')
    return Path(root)/model/'metrics.json'


def load_result(root, model, expected_fingerprint):
    path = result_path(root,model)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('model') != model:
        raise ValueError(f'Model identity mismatch: {path}')
    if payload.get('status') == 'completed':
        if payload.get('protocol') != PROTOCOL or payload.get('evaluation_split') != 'test' or payload.get('dataset_fingerprint') != expected_fingerprint:
            raise ValueError(f'Incompatible evaluation provenance: {path}')
        if not payload.get('checkpoint_sha256') or not payload.get('hardware') or not payload.get('timing_definition'):
            raise ValueError(f'Missing checkpoint/timing provenance: {path}')
    elif any(value is not None for value in payload.get('metrics',{}).values()):
        raise ValueError(f'Non-completed results cannot carry comparison measurements: {path}')
    for key,value in payload.get('metrics',{}).items():
        if value is None:
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid metric {key}: {path}')
        if key in (*METRICS,'F1','mask_precision','mask_recall','mask_map50','mask_map50_95') and value > 1:
            raise ValueError(f'Metric outside [0,1]: {key}')
    return payload


def display(value):
    return 'N/A' if value is None or value == '' else (f'{value:.4f}' if isinstance(value,float) else str(value))


def generate(output,data,environment=None):
    output,data = Path(output),Path(data)
    environment = environment or {}
    dataset_id = fingerprint(data)
    audit = json.loads((data/'dataset_report.json').read_text())
    rows,payloads = [],[]
    for family,variant in MODELS.items():
        payload = load_result(output,family,dataset_id)
        if payload is None:
            notes = 'No canonical trained checkpoint or measured test results. Full training not run.'
            if environment.get('cuda_available') is False:
                notes += ' CPU-only: full three-model training impractical in this session.'
            if family=='efficientdet_d3' and not environment.get('packages',{}).get('effdet'):
                notes += ' effdet dependency unavailable.'
            payload = dict(model=family,variant=variant,status='not_trained',metrics={},notes=notes,
                           dataset_fingerprint=dataset_id,hardware=environment)
            write_json(result_path(output,family),payload)
            write_csv(output/family/'per_class.csv',
                      [dict(class_id=i,**{'class':name},support=audit['splits']['test']['classes'][name]) for i,name in SOURCE_CLASSES.items()],
                      ['class_id','class','support',*METRICS])
            write_json(output/family/'groups.json',{group:{key:None for key in METRICS} for group in ('Class A','Class B','Class C','Rejected','Body','Head','Tail')})
            write_json(output/family/'error_analysis.json',{'status':'unavailable','reason':'No trained test predictions; no observed model errors.'})
            folder = output/family/'predictions'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'README.md').write_text('No trained predictions available. Successful evaluation writes ALL 115 held-out images (green ground truth, red predictions), including failures. Whole-fish counts and overlap scenario tags need human review.\n',encoding='utf-8')
        row = {key:None for key in FIELDS}
        row.update(Model=variant,variant=variant,task='part segmentation' if family=='yolo' else 'part detection',
                   input_size=payload.get('input_size'),evaluation_split=payload.get('evaluation_split'),status=payload['status'],notes=payload.get('notes',''))
        row.update({key:value for key,value in payload.get('metrics',{}).items() if key in FIELDS})
        p,r = row['Precision'],row['Recall']
        if p is not None and r is not None:
            row['F1'] = 2*p*r/(p+r) if p+r else 0.
        rows.append(row)
        payloads.append(payload)
    write_csv(output/'model_comparison.csv',rows,FIELDS)
    complete = [r for r in rows if r['status']=='completed']
    comparable = len(complete)==3
    timing_keys = {(p.get('benchmark_device'),json.dumps(p.get('hardware'),sort_keys=True),p.get('timing_definition')) for p in payloads if p['status']=='completed'}
    def winner(key,reverse=True):
        if not comparable or any(r[key] is None for r in complete):
            return 'Undetermined: three completed comparable measurements required.'
        if key=='FPS' and len(timing_keys)!=1:
            return 'Undetermined: benchmark hardware/method differs.'
        best = (max if reverse else min)(r[key] for r in complete)
        return ', '.join(r['Model'] for r in complete if r[key]==best)
    table_keys = ['Model','Precision','Recall','mAP50','mAP50_95','FPS','InferenceTime','ModelSize','status']
    table = ['| '+' | '.join(table_keys)+' |','| '+' | '.join(['---']*len(table_keys))+' |']
    table += ['| '+' | '.join(display(r[k]) for k in table_keys)+' |' for r in rows]
    lines = ['# Model Comparison','', '## Experimental Setup','',
        'Target: 12 anatomical-part/grade source classes, never four whole-fish boxes. Dataset: canonical_v7_parts, derived from Roboflow v7 COCO segmentation. Source class IDs are unchanged.',
        f"Split sizes (stored manifest/report): train {audit['splits']['train']['images']}, validation {audit['splits']['val']['images']}, test {audit['splits']['test']['images']}; {audit['leakage_groups']} leakage groups. Seed 42. Source-group and exact-duplicate boundaries are preserved.",
        f'Dataset fingerprint: `{dataset_id}`. The runner validates every image hash and canonical safeguards before training/evaluation.',
        'Planned resolution: 640 × 640 for all three. EfficientDet-D3 uses a 640 override rather than its usual larger input. Detector adapters derive XYXY boxes from the same canonical polygons; internal foreground labels are 1–12 and exported IDs remain 0–11.',
        'Observed environment: `'+json.dumps(environment,sort_keys=True)+'`.',
        'Common box protocol: class-aware greedy matching; IoU .50:.05:.95; 101 recall-point interpolated AP; macro class mean; maximum 100 detections/image. No crowd/ignore semantics or COCO size-stratified AP are claimed. P/R use fixed confidence .25 and IoU .50; prediction floor .001. Missing ground-truth classes are excluded, never assigned invented scores. F1 is the harmonic mean of macro P/R.',
        'Timing: batch 1, ten warmups, every test image once, CUDA synchronized when available. Includes preprocessing, transfer, forward, postprocessing and CPU box conversion; excludes checkpoint loading, image decoding, rendering and metric computation. FPS = 1000 / mean latency in ms. Checkpoint MB uses decimal bytes/1,000,000.',
        'Training defaults: 100 epochs; detector batch 2. Faster R-CNN uses pretrained ResNet50-FPN, SGD .005/momentum .9; EfficientDet-D3 uses pretrained tf_efficientdet_d3, AdamW .0002. Both decay learning rate by .1 each third of training and select validation box mAP50-95. YOLO preserves existing augmentation, optimizer and early-stopping settings, and selects native combined box/mask fitness. These are documented architecture-specific baselines, not equally tuned searches. One seed cannot establish statistical superiority.',
        '', '## Comparison Table','',*table,'',
        'All absent measurements are N/A, not zero. CSV additionally contains completed epochs, best epoch, training minutes, parameter count, median latency and separate mask metrics. Input-size cells remain unavailable until a run actually records them.','']
    descriptions = {
        'yolo':('YOLO','YOLOv8n-seg is selected by configs/yolo_parts.yaml; the renamed yolov26 placeholder directory does not override that config. Native masks and an existing auxiliary part-fusion path are design strengths. Part predictions still require association; they must not become whole-fish tracker detections. Speed, compactness and localization superiority have not been measured.'),
        'faster_rcnn':('Faster R-CNN','ResNet50-FPN uses region proposals and a 12-class foreground head plus background. Proposal-based localization is a design capability; accuracy gains are unproven. This branch is box-only and needs a separate deployment adapter; latency and size penalties have not been measured.'),
        'efficientdet_d3':('EfficientDet','EfficientDet-D3 uses compound scaling and bidirectional feature fusion. It offers an efficiency-oriented design, but no observed accuracy/speed advantage. The additional effdet dependency and its YXYX training-label convention add integration work; this branch is box-only.')}
    for family,row in zip(MODELS,rows):
        title,description = descriptions[family]
        lines += ['## '+title,'',description, '',f"Status: {row['status']}. {row['notes']}",'',
            'Run details: '+ '; '.join(f'{k}={display(row[k])}' for k in ('epochs_completed','best_epoch','training_time_minutes','parameter_count'))+'.',
            'Mask metrics (additional capability; excluded from box ranking): '+ '; '.join(f'{k}={display(row[k])}' for k in ('mask_precision','mask_recall','mask_map50','mask_map50_95'))+'.',
            'Per-class, quality and anatomy macro views are saved alongside metrics. Error analysis reports observed false positives and missed instances at the fixed operating point; specific confusion/overlap narratives require visual review. No error categories are asserted without predictions.','']
    lines += ['## Recommended Model','', 'Highest detection accuracy: '+winner('mAP50_95'),
        'Fastest inference: '+winner('FPS'),'Smallest measured checkpoint: '+winner('ModelSize',False),
        'Conveyor recommendation: not yet established. Require complete held-out results, acceptable recall for every quality/part class (especially Rejected), a measured latency below the actual conveyor frame budget, and acceptable deployment memory. Select among the accuracy/speed/size Pareto alternatives after those application constraints are specified; no arbitrary weighted score is used.',
        'Academically strongest: not established. Compare class consistency, repeated seeds and uncertainty, as well as held-out AP; one incomplete or single-seed experiment cannot prove general superiority.','',
        '## Alternative Use Cases','',
        'YOLO is a candidate when part masks are required; Faster R-CNN is a candidate for proposal-based detection experiments; EfficientDet is a candidate for studying feature-pyramid efficiency. These are design-motivated uses, not measured rankings.','',
        '## Limitations','',
        'Only 910 images / 337 source groups; imbalance includes 39 Grade_C_Tail test instances versus 167 Rejected_Body instances. Part-level annotations and limited true segmentation geometry cannot establish whole-fish grading accuracy. Whole-fish ground truth is missing. Static images may differ substantially from conveyor video in blur, lighting, occlusion and fish presentation. CPU-only local hardware makes full training impractical; no pretrained trained checkpoints were found. EfficientDet dependency is missing in the inspected environment. No production-readiness claim is supported.',
        'All held-out images will be saved to avoid success cherry-picking. Easy single fish, multiple fish and overlap tags require reviewed whole-fish scene labels; part counts are not fish counts. Minority and rejected parts remain represented in the full prediction set.','',
        '## Reproduction','', 'Run from the Thesis repository in an environment with a working CUDA-enabled PyTorch installation. Package installation may download pretrained weights during training. Run names must be new; existing weights are not overwritten.','', '```powershell',
        'python -m pip install -r requirements-comparison.txt',
        *[f'python -m src.training.train_comparison --model {family} --dry-run' for family in MODELS],
        *[f'python -m src.training.train_comparison --model {family} --run-name comparison_v1 --epochs 100 --batch-size {8 if family=="yolo" else 2} --input-size 640 --device cuda:0' for family in MODELS],
        'python -m src.evaluation.compare_models','```','',
        'To retry evaluation after a completed training run, repeat its command with --evaluate-only. Checkpoint provenance must match the canonical fingerprint and input size. Successful runs populate all common metrics, per-class/group results, error counts, predictions, size, parameter count, timing and training metadata automatically. YOLO additionally populates native mask metrics. Detector mask fields stay N/A. Native mask P/R use Ultralytics operating-point semantics and must not be substituted for shared box P/R. Interrupted runs remain incomplete in experiment_info.json and cannot enter the completed comparison.',
        '', 'Implementation reference: [EfficientDet upstream benches](https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/bench.py) and [model factory](https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/factory.py).',
        '', 'Local implementation checks and unverified execution paths are documented in [COMPARISON_VALIDATION.md](COMPARISON_VALIDATION.md).','']
    (output/'MODEL_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=project_root()/'results')
    parser.add_argument('--data',type=Path,default=project_root()/'dataset/canonical_v7_parts')
    args = parser.parse_args()
    from src.training.train_comparison import hardware
    generate(args.output,args.data,hardware())


if __name__ == "__main__":
    main()
