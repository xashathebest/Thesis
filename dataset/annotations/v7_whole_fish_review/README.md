# V7 whole-fish review gate

The two Roboflow v7 exports are immutable sources. Their annotations describe
`Body`, `Head`, and `Tail` regions independently and do not identify which
regions belong to the same physical fish. They must not be relabeled by prefix
and treated as four-class fish instances.

Before materializing a canonical training dataset, create reviewed COCO
instance-segmentation annotations with exactly these categories:

- `0 = Class A`
- `1 = Class B`
- `2 = Class C`
- `3 = Rejected`

Each annotation must be one tight polygon or mask around one physical fish (or
an independently sortable fragment if the research protocol explicitly permits
that unit). Record `specimen_id`, `scene_id`, occlusion/truncation status,
gradability, annotation confidence, and the adjudicated whole-fish grade. Mixed
part grades require expert adjudication; no worst-part, majority, or body-only
rule is approved by this repository.

Use the COCO export as the visual/geometry reference because it retains six RLE
masks that the YOLO export approximates as polygons. Do not copy images from both
exports. Group all augmented versions with the original name in COCO
`images[].extra.name`, and keep the complete group in one split.

The final reviewed split must contain every thesis class in train, validation,
and test. The exported v7 split does not meet that requirement: its test set has
no Class A parts and only four Class B parts.

Add `extra.leakage_group_id` to each reviewed COCO image record, using the same
value for augmented versions, adjacent frames, and repeated views of the same
specimen. Then materialize the reviewed dataset without touching this source:

```powershell
python -m src.preprocessing.prepare_reviewed_coco `
  --source dataset/annotations/v7_whole_fish_review/reviewed_coco `
  --output dataset/canonical_v7_whole_fish
```

The converter rejects the current part categories, missing classes in a split,
empty images, cross-split leakage groups, RLE masks that would otherwise be
replaced by boxes, and multi-ring instances that YOLO cannot represent safely.
It writes one YOLO row per reviewed fish plus `data.yaml`, `split_manifest.csv`,
and `provenance.json`.
