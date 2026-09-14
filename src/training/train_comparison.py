"""Canonical comparison runner. Dry runs validate data without training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.evaluation.comparison_protocol import MODELS, PROTOCOL, evaluate_boxes, fingerprint, preflight, records, write_csv, write_json
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root


def hardware():
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'effdet', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(processor=platform.processor(), cuda_available=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', packages=packages)


def make_model(family, size, pretrained):
    if family == 'faster_rcnn':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, min_size=size, max_size=size, box_score_thresh=.001, box_detections_per_img=100)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 13)
        return model
    from effdet import create_model
    return create_model('tf_efficientdet_d3', num_classes=12, pretrained=pretrained, pretrained_backbone=False,
                        image_size=(size, size), max_det_per_image=100)


def tensor_image(image, size, device, normalize=False):
    import torch
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()).to(device) / 255
    if normalize:
        tensor = (tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]
    return tensor


class Predictor:
    def __init__(self, family, model, size, device):
        self.family, self.model, self.size, self.device = family, model, size, device
        if family == 'efficientdet_d3':
            from effdet import DetBenchPredict
            self.model = DetBenchPredict(model).to(device).eval()
        elif family == 'faster_rcnn':
            model.eval()

    def __call__(self, image):
        import torch
        width, height = image.size
        with torch.inference_mode():
            if self.family == 'yolo':
                result = self.model.predict(image, imgsz=self.size, conf=.001, max_det=100, device=self.device, verbose=False)[0].boxes
                return dict(boxes=result.xyxy.cpu().tolist(), labels=result.cls.int().cpu().tolist(), scores=result.conf.cpu().tolist())
            tensor = tensor_image(image, self.size, self.device, self.family == 'efficientdet_d3')
            if self.family == 'faster_rcnn':
                result = self.model([tensor])[0]
                boxes, labels, scores = result['boxes'], result['labels']-1, result['scores']
            else:
                result = self.model(tensor[None])[0]
                result = result[result[:,4] >= .001]
                boxes, scores, labels = result[:,:4], result[:,4], result[:,5].long()-1
            boxes = boxes * boxes.new_tensor([width/self.size,height/self.size]*2)
            return dict(boxes=boxes.cpu().tolist(), labels=labels.cpu().tolist(), scores=scores.cpu().tolist())


def predict_split(predictor, rows, benchmark=False):
    import torch
    def sync():
        if str(predictor.device).startswith('cuda'):
            torch.cuda.synchronize()
    if benchmark:
        with Image.open(rows[0]['path']) as source:
            warmup = source.convert('RGB')
        for _ in range(10):
            predictor(warmup)
        sync()
    predictions, timings = [], []
    for row in rows:
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        sync()
        start = time.perf_counter()
        prediction = predictor(image)
        sync()
        timings.append((time.perf_counter()-start)*1000)
        predictions.append(dict(image=row['image'], **prediction))
    return predictions, dict(InferenceTime=float(np.mean(timings)), median_inference_ms=float(np.median(timings)), FPS=1000/float(np.mean(timings)))


def train_detector(args, run, info, device):
    import torch
    model = make_model(args.model, args.input_size, True).to(device)
    if args.model == 'efficientdet_d3':
        from effdet import DetBenchTrain
        bench = DetBenchTrain(model, create_labeler=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or .0002, weight_decay=.0001)
    else:
        bench = model
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr or .005, momentum=.9, weight_decay=.0005)
    info['optimizer'] = type(optimizer).__name__
    info['learning_rate'] = optimizer.param_groups[0]['lr']
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,args.epochs//3), gamma=.1)
    train, val = records(args.data,'train'), records(args.data,'val')
    best, history, start = -1., [], time.perf_counter()
    try:
        for epoch in range(1,args.epochs+1):
            bench.train()
            order = list(range(len(train)))
            random.Random(args.seed+epoch).shuffle(order)
            losses = []
            for offset in range(0,len(order),args.batch_size):
                batch = [train[i] for i in order[offset:offset+args.batch_size]]
                images, targets = [], []
                for row in batch:
                    with Image.open(row['path']) as source:
                        images.append(tensor_image(source.convert('RGB'),args.input_size,device,args.model=='efficientdet_d3'))
                    boxes = torch.tensor(row['boxes'],dtype=torch.float32,device=device).reshape(-1,4)
                    boxes *= boxes.new_tensor([args.input_size/row['width'],args.input_size/row['height']]*2)
                    targets.append(dict(boxes=boxes,labels=torch.tensor(row['labels'],dtype=torch.long,device=device)+1))
                optimizer.zero_grad()
                if args.model == 'efficientdet_d3':
                    loss = bench(torch.stack(images),dict(bbox=[t['boxes'][:,[1,0,3,2]] for t in targets],cls=[t['labels'] for t in targets]))['loss']
                else:
                    loss = sum(bench(images,targets).values())
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            predictions,_ = predict_split(Predictor(args.model,model,args.input_size,device),val)
            score = evaluate_boxes(val,predictions)['overall']['mAP50_95']
            info.update(epochs_completed=epoch,training_time_minutes=(time.perf_counter()-start)/60)
            if score > best:
                best = score
                info['best_epoch'] = epoch
                torch.save(model.state_dict(),run/'best.pth')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),val_box_map50_95=score))
            write_json(run/'history.json',history)
            write_json(run/'experiment_info.json',info)
            print(f'epoch={epoch} val_box_map50_95={score:.6f}',flush=True)
    finally:
        info['training_time_minutes'] = (time.perf_counter()-start)/60
        write_json(run/'experiment_info.json',info)
    return run/'best.pth'


def train_yolo(args,run,info,device):
    from src.training.train_yolo_parts import main as existing_train
    from src.preprocessing.dataset_utils import load_yaml_file
    import yaml
    config = load_yaml_file(project_root()/'configs/yolo_parts.yaml')
    config.update(data=str(args.data/'data.yaml'),imgsz=args.input_size,epochs=args.epochs,
                  batch_size=args.batch_size,seed=args.seed,project_dir=str(run/'training'))
    config_path = run/'yolo.yaml'
    config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
    start = time.perf_counter()
    try:
        code = existing_train(['--config',str(config_path),'--acknowledge-part-only','--run-name','fit','--device',device])
        if code:
            raise RuntimeError(f'YOLO trainer exited {code}')
    finally:
        info['training_time_minutes'] = (time.perf_counter()-start)/60
        write_json(run/'experiment_info.json',info)
    with (run/'training/fit/results.csv').open() as stream:
        history = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(stream)]
    info['epochs_completed'] = len(history)
    info['selection_rule'] = 'Ultralytics native segmentation fitness; detector branches select validation box mAP50-95'
    # Use the actual best-epoch callback sidecar written by the existing trainer.
    sidecar = run/'training/fit/best_epoch.json'
    info['best_epoch'] = json.loads(sidecar.read_text())['best_epoch'] if sidecar.exists() else None
    return run/'training/fit/weights/best.pt'


def evaluate_run(args,run,info,device):
    import torch
    checkpoint = Path(info['checkpoint'])
    if args.model == 'yolo':
        from ultralytics import YOLO
        model = YOLO(str(checkpoint))
        parameters = sum(p.numel() for p in model.model.parameters())
    else:
        model = make_model(args.model,args.input_size,False).to(device)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        parameters = sum(p.numel() for p in model.parameters())
    truth = records(args.data,'test')
    predictions,timing = predict_split(Predictor(args.model,model,args.input_size,device),truth,True)
    metrics = evaluate_boxes(truth,predictions)
    output = args.output/args.model
    write_json(output/'predictions/test.json',predictions)
    for row,prediction in zip(truth,predictions):
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for box in row['boxes']:
            draw.rectangle(box,outline='lime',width=1)
        for box,label,score in zip(prediction['boxes'],prediction['labels'],prediction['scores']):
            if score >= .25:
                draw.rectangle(box,outline='red',width=2)
                draw.text((box[0],box[1]),f'{SOURCE_CLASSES[label]} {score:.2f}',fill='red')
        image.save(output/'predictions'/row['image'])
    values = {**metrics['overall'],**timing,'ModelSize':checkpoint.stat().st_size/1_000_000,'parameter_count':parameters,
              **{k:info.get(k) for k in ('epochs_completed','best_epoch','training_time_minutes')}}
    if args.model == 'yolo':
        mask = model.val(data=str(args.data/'data.yaml'),split='test',imgsz=args.input_size,conf=.001,max_det=100,
                         device=device,project=str(output),name='mask_test',exist_ok=True)
        values.update(mask_precision=float(mask.seg.mp),mask_recall=float(mask.seg.mr),mask_map50=float(mask.seg.map50),mask_map50_95=float(mask.seg.map))
        write_json(output/'mask_metrics.json',dict(definition='Ultralytics native mask metrics; P/R uses native best-F1, not shared box operating point',results=mask.results_dict))
        mask_rows = {int(class_id):index for index,class_id in enumerate(mask.seg.ap_class_index)}
        for row in metrics['per_class']:
            index = mask_rows.get(row['class_id'])
            for key,values_array in [('mask_precision',mask.seg.p),('mask_recall',mask.seg.r),('mask_map50',mask.seg.ap50),('mask_map50_95',mask.seg.ap)]:
                row[key] = float(values_array[index]) if index is not None else None
    payload = dict(model=args.model,variant=MODELS[args.model],status='completed',evaluation_split='test',protocol=PROTOCOL,
        dataset_fingerprint=info['dataset_fingerprint'],input_size=args.input_size,hardware=hardware(),benchmark_device=device,
        timing_definition='batch1 PIL RGB in memory -> preprocess + transfer + forward + postprocess + CPU boxes; 10 warmups; all test images; load/decode excluded',
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),metrics=values,
        notes=info.get('selection_rule','validation box mAP50-95 selects best epoch'))
    write_json(output/'metrics.json',payload)
    write_csv(output/'per_class.csv',metrics['per_class'],['class_id','class','support','Precision','Recall','mAP50','mAP50_95','mask_precision','mask_recall','mask_map50','mask_map50_95'])
    write_json(output/'groups.json',metrics['groups_macro_mean'])
    write_json(output/'error_analysis.json',metrics['errors'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--data',type=Path,default=project_root()/'dataset/canonical_v7_parts')
    parser.add_argument('--output',type=Path,default=project_root()/'results')
    parser.add_argument('--run-name',default='comparison_v1')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--input-size',type=int,default=640)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--device',default=None)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    args = parser.parse_args(argv)
    if args.epochs<1 or args.batch_size<1 or args.input_size<128 or args.input_size%128:
        parser.error('Positive epochs/batch and input size divisible by 128 required')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        parser.error('run-name must be one directory name')
    args.data = args.data.resolve()
    environment = hardware()
    audit = preflight(args.data)
    print(json.dumps(dict(hardware=environment,splits={s:r['images'] for s,r in audit['splits'].items()},dataset_fingerprint=fingerprint(args.data)),indent=2))
    if args.dry_run:
        print('DATA PREFLIGHT PASS; no training or model-forward pass performed')
        if args.model=='efficientdet_d3' and not environment['packages']['effdet']:
            print('TRAINING BLOCKED: effdet missing; install requirements-comparison.txt')
        return 0
    device = args.device or ('cuda:0' if environment['cuda_available'] else 'cpu')
    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = project_root()/'models'/('yolo_parts' if args.model=='yolo' else args.model)/args.run_name
    if args.evaluate_only:
        info = json.loads((run/'experiment_info.json').read_text())
        if info['dataset_fingerprint']!=fingerprint(args.data) or info['input_size']!=args.input_size or info['status']!='completed' or info['model']!=args.model:
            raise ValueError('Checkpoint provenance does not match completed canonical experiment')
        if hashlib.sha256(Path(info['checkpoint']).read_bytes()).hexdigest()!=info.get('checkpoint_sha256'):
            raise ValueError('Checkpoint hash differs from completed experiment')
    else:
        run.mkdir(parents=True,exist_ok=False)
        info = dict(model=args.model,variant=MODELS[args.model],input_size=args.input_size,dataset_fingerprint=fingerprint(args.data),
                    hardware=environment,status='incomplete',seed=args.seed,requested_epochs=args.epochs,batch_size=args.batch_size,
                    epochs_completed=0,best_epoch=None,training_time_minutes=None)
        write_json(run/'experiment_info.json',info)
        checkpoint = (train_yolo if args.model=='yolo' else train_detector)(args,run,info,device)
        info.update(status='completed',checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        write_json(run/'experiment_info.json',info)
    evaluate_run(args,run,info,device)
    from src.evaluation.compare_models import generate
    generate(args.output,args.data,environment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
