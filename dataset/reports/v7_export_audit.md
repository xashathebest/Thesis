# Roboflow v7 export audit

Audit status: `BLOCKED_PENDING_WHOLE_FISH_REVIEW`

The two exports are valid alternative representations of the same images and
annotations, but their annotation unit is a fish part—not a physical fish. They
cannot directly produce the four-class, per-fish model required by the webcam
and ByteTrack application.

## Inventory

| Split | Derived images | Original-source groups | Annotations | Empty images | Max instances/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 861 | 288 | 9,099 | 3 | 21 |
| Validation | 32 | 32 | 399 | 0 | 18 |
| Test | 17 | 17 | 225 | 0 | 19 |
| Total | 910 | 337 | 9,723 | 3 | 21 |

All 910 images are readable 640×640 JPEGs. Every YOLO image has a same-stem
label, every label has an image, and every COCO image reference resolves. No
invalid class IDs, out-of-bounds polygon coordinates, zero-area boxes, duplicate
generated filenames, duplicate COCO IDs, or annotation-to-image reference
errors were found. Seven annotations cover less than 0.1% of image area and
need manual review. The three empty files are augmented versions of
`WIN_20260823_21_45_06_Pro.jpg`.

The directory `Dried Fish Quality Grading.v7i.coco-segmentation/` at repository
root and `dataset/Dried Fish Quality Grading.v7i.coco-segmentation/` contain the
same 915 files with identical SHA-256 content. This audit does not delete either
copy.

## Class definitions and distribution

The exact semantic mapping is by quality prefix, while retaining that the
source instance is a part:

| Target quality | Source part class IDs (YOLO / COCO) | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Class A | 0–2 / 1–3 | 1,625 | 5 | 0 | 1,630 |
| Class B | 3–5 / 4–6 | 3,181 | 2 | 4 | 3,187 |
| Class C | 6–8 / 7–9 | 1,446 | 127 | 90 | 1,663 |
| Rejected | 9–11 / 10–12 | 2,847 | 265 | 131 | 3,243 |

COCO category `0 = dried-fish-quality-grades` is a supercategory placeholder
and has no annotations. COCO annotation categories use IDs 1–12; YOLO uses
0–11. The source classes are `Grade_{A|B|C}_{Body|Head|Tail}` and
`Rejected_{Body|Head|Tail}`.

This is severely unsuitable for final evaluation: the test set cannot measure
Class A at all and has only four Class B part annotations. Validation has only
five Class A and two Class B parts. A reviewed whole-fish dataset must be
re-split at source/specimen group level with all four classes represented.

## Annotation-format comparison

The YOLO26 export is dataset format only; it does not require a YOLO26 runtime.
It contains normalized polygon coordinates, class IDs, a `data.yaml`, and
train/valid/test image/label directories. All 9,723 records have more than the
five fields used by ordinary YOLO boxes, so they are segmentation-format rows.

The COCO export stores image IDs, category IDs, bounding boxes, `iscrowd`, area,
and instance segmentation. It contains 9,717 polygon annotations and six
compressed RLE masks. No annotation has multiple polygon rings and every
`iscrowd` value is false. The images and ordinary polygon geometry agree with
the YOLO export; the six RLE masks are approximated in YOLO and make COCO the
preferred source representation.

Segmentation quality is limited. Of 9,717 polygon annotations, 9,405 are
axis-aligned rectangles (96.8%); only 312 contain non-rectangular geometry.
COCO `area` equals bounding-box area throughout, producing 451 disagreements
with actual polygon area. Consequently, mask mAP from this export would mostly
measure rectangular part localization, not tight fish silhouettes.

## Multi-instance, overlap, and visibility findings

There are multiple part instances in 907 of 910 images. COCO includes 1,490
annotation-box pairs with IoU ≥ 0.1 and 15 pairs with IoU ≥ 0.5. Those figures
show crowded/overlapping labeled regions, but they do not prove distinct whole
fish overlap because part-to-fish association IDs are absent. Boundary-touching
parts and visibly fragmented fish occur; they require truncation, occlusion, and
gradability review under `dataset/ANNOTATION_GUIDELINES.md`.

Sixteen train images, two validation images, and one test image contain more
than one quality prefix. Direct inspection confirmed that an intact physical
fish can have, for example, Grade A head/tail regions and a Grade B body region.
A final fish class therefore cannot be inferred safely by renaming categories.

## Duplicate and leakage audit

The YOLO and COCO images match byte-for-byte for every split (910/910), so
combining the exports would double the dataset without adding observations.
There are no exact image hashes across train, validation, and test. Grouping by
COCO `images[].extra.name` yields 337 original-source groups and no group spans
more than one split. A 256-bit difference-hash screen found no cross-split pair
within Hamming distance 12.

Train contains exposure-augmented copies: 285 original groups have three
derived images and three groups have two. Validation and test contain one image
per original. These copies are legitimate within-train augmentation, not
independent source specimens, and must remain grouped in future splitting.
Adjacent capture frames/specimen identity cannot be ruled out from filenames
alone because the export has no specimen manifest; that remains a leakage risk.

## Canonical dataset decision

COCO is the canonical *source annotation* representation. No final canonical
four-class dataset is materialized yet because doing so would invent fish
boundaries, part associations, and whole-fish grading policy. The already
materialized `dataset/splits` directory is an exact `source_class_id // 3`
collapse of all 910 YOLO files. Its geometry still describes independent parts,
so it is marked unverified and blocked by the segmentation preflight.

The review gate is documented in
`dataset/annotations/v7_whole_fish_review/README.md`. After review, materialize
one image once (from COCO, not both exports), one mask per physical fish, the
four exact thesis class IDs, and a source/specimen-group split manifest.

## Model and training strategy

The primary final task should be YOLO instance segmentation, with bounding boxes
retained automatically for ByteTrack. Recommended first model:

- `yolov8n-seg.pt`, pretrained, 640 px, 100 epochs maximum, patience 20,
  batch 8, auto device, two workers;
- restrained geometric/color augmentation because grading uses subtle surface
  and color evidence; mosaic/copy-paste are low-probability and must be reviewed;
- compare `yolov8s-seg.pt` only after the nano baseline and only if measured
  laptop latency/FPS remains acceptable;
- keep YOLOv8n detection as a speed baseline, not the primary crowded-scene
  model. A segmentation model still emits boxes, classes, confidence, and track
  IDs through the existing Ultralytics/ByteTrack path.

Training from scratch is inappropriate for 337 original-source images.
Transfer learning is required. Do not start final training until the review
gate passes and validation/test are representative.

Once reviewed data replaces the blocked split and `annotation_unit` is changed
to `whole_fish`, materialize and train it with:

```powershell
python -m src.preprocessing.prepare_reviewed_coco `
  --source dataset/annotations/v7_whole_fish_review/reviewed_coco `
  --output dataset/canonical_v7_whole_fish
python -m src.training.train_yolo_segmentation `
  --data dataset/canonical_v7_whole_fish/data.yaml `
  --manifest dataset/canonical_v7_whole_fish/split_manifest.csv `
  --dry-run
python -m src.training.train_yolo_segmentation `
  --data dataset/canonical_v7_whole_fish/data.yaml `
  --manifest dataset/canonical_v7_whole_fish/split_manifest.csv `
  --run-name v7_whole_fish_yolov8n_seg
```

Expected weights:
`models/yolo_segmentation/v7_whole_fish_yolov8n_seg/weights/best.pt`.

Evaluation must record box and mask mAP@0.5 and mAP@0.5:0.95, precision,
recall, per-class metrics/confusion matrix, model size, latency, and FPS. Tag a
review subset as isolated, multi-fish, and overlapping so performance can be
reported separately. The first recommended experiment is a 640 px
YOLOv8n-seg transfer-learning baseline after manual whole-fish review; then
compare 960 px only if small/overlap failures justify its latency cost.
