# Dataset Metadata Schema

These CSV files are blank templates. Fill them during collection and annotation;
do not infer physical specimen identity from filenames or timestamps alone.

## Image metadata

`dataset/metadata.csv` contains one row per source frame. `image_id` is the
extensionless image filename and must be unique. Use `|` between multiple IDs in
one field (for example, `fish_0042|fish_0043`). Do not use commas inside an ID.

Fields required before an annotated image is eligible for splitting:

- `image_id`: extensionless filename of the annotated frame.
- `capture_session`: one uninterrupted camera/setup session.
- `batch_id`: independent collection or catch batch.
- `scene_id`: one physical conveyor arrangement/pile. All nearby frames showing
  the same arrangement must share this value.
- `specimen_ids`: every physical fish visible in the frame, pipe-separated. For
  multi-fish frames, also create one normalized row per annotation in
  `manifests/specimen_groups.csv`.
- `split_eligible`: explicit `yes` or `no`. A `no` row also requires
  `exclusion_reason`; excluded frames remain untouched.

Use `capture_sequence` for a run of temporally adjacent/rearranged frames and
`split_group_id` when reviewers know that records must stay together for a reason
not captured by another field. `duplicate_group_id` may hold pipe-separated audit
group IDs. These fields only make grouping stricter; they never override specimen,
scene, batch, or session grouping.

Camera and acquisition fields are descriptive but important for detecting
shortcut learning: `camera_id`, position/height, resolution, lighting, conveyor
state/speed, exposure, gain/ISO, shutter, white balance, and focus mode. Use fixed
settings where practical and record actual values rather than guesses.

`fish_class` is optional at frame level because one frame may contain multiple
classes. Final labels and biological attributes belong in the instance table.

## Specimen grouping

`dataset/manifests/specimen_groups.csv` is a normalized one-row-per-annotation
link table:

- `image_id`: source frame ID matching `metadata.csv`.
- `instance_id`: stable annotation ID within the frame, such as `fish_0001`.
- `specimen_id`: stable physical fish identity across every frame where it occurs.
- `scene_id`: must match the source frame's scene when supplied.
- `association_confidence`: controlled value such as `high`, `medium`, or `low`.
- `review_status`: for example `KEEP` or `MANUAL_REVIEW`.

Do not invent specimen IDs when identity is uncertain. Mark the association for
manual review and do not split the dataset until it is resolved.

## Instance attributes

`dataset/annotations/attributes/instance_attributes_template.csv` is the blank
schema for a one-row-per-fish annotation table.
It separates observed attributes (structure, morphology, discoloration, defects,
occlusion, truncation, and confidence) from the expert-approved final grade. Keep
the two independent expert labels and adjudicated label so inter-rater agreement
can be measured.

Use consistent controlled values for booleans (`yes`, `no`, `uncertain`) and
document any numeric severity/fullness scale before annotation starts. Never fill
unknown values by guessing.

## Leakage-safe split contract

Run the audit first so both exact- and near-duplicate manifests exist. Then run:

```bash
python -m src.preprocessing.split_dataset
```

Polygon instance-segmentation labels are required by default. The explicit
`--annotation-format box` option exists only to reproduce a documented legacy
detector experiment; a manifest cannot mix boxes and polygons.

The command refuses to create a manifest unless every eligible annotated frame
has sufficient independence metadata, every multi-fish annotation is associated
with a specimen, duplicate reviews are resolved, at least three independent
connected groups exist, and every class occurs in at least three groups. A valid
run writes `dataset/manifests/split_manifest.csv`; it does not copy images unless
`--materialize` is supplied.

The splitter joins all frames connected by any batch, session, scene, sequence,
specimen, explicit split group, or duplicate group. This transitive component is
the atomic `leakage_group_id`. Consequently, no related frame can cross training,
validation, and test even through an indirect relationship.
