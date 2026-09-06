# V7 part fusion and structural reasoning

Status: `PROTOTYPE_IMPLEMENTED; MODEL_TRAINING_AND_RUNTIME_ENABLEMENT_GATED`

This design preserves the verified 12 source classes while preventing
anatomical regions from becoming separately tracked fish. It does not weaken
the whole-fish review gate or create whole-fish ground truth from inferred
associations.

## Source mapping

The exact IDs remain those in `configs/v7_source_mapping.yaml`:

| YOLO IDs | COCO IDs | Region classes | Fish-quality evidence |
| --- | --- | --- | --- |
| 0, 1, 2 | 1, 2, 3 | Grade_A_Body, Head, Tail | Class A |
| 3, 4, 5 | 4, 5, 6 | Grade_B_Body, Head, Tail | Class B |
| 6, 7, 8 | 7, 8, 9 | Grade_C_Body, Head, Tail | Class C |
| 9, 10, 11 | 10, 11, 12 | Rejected_Body, Head, Tail | Rejected |

The part-model adapter rejects an unfamiliar name/ID combination rather than
guessing its meaning.

## Rejected-sample investigation

| Rejected region | Annotations |
| --- | ---: |
| Body | 1,310 |
| Head | 1,276 |
| Tail | 657 |

Rejected annotations occur in 240 exported images belonging to 100 unique
pre-augmentation source images. Image-level category presence is:

| Rejected region types present somewhere in image | Derived images | Unique sources |
| --- | ---: | ---: |
| Head + Body + Tail | 223 | 93 |
| Head + Body | 14 | 6 |
| Head + Tail | 3 | 1 |
| One region type only | 0 | 0 |

These are image-level counts, not fish-completeness counts. A frame can contain
many fish and there are no source part-to-fish IDs. They must not be used to
claim that 93 fish are complete or that six fish lack tails.

Manual inspection of representatives found clearly fragmented fish, exposed
spines/tissue, head loss, tail loss, and severe body truncation. It also found
substantially complete fish among Rejected scenes with serious discoloration,
surface/tissue damage, or deformation. Rejected therefore has multiple visible
causes; missing anatomy is neither necessary nor safely inferred from a missing
part label. Crowding and touching fish are common. Only one Rejected annotation
touches the nominal 640×640 boundary, but fitted white borders mean this is not
a reliable observability measure for source images.

The export can train a part/grade recognizer, but it cannot supervise physical
absence, occlusion probability, termination visibility, or part association.
Those properties need reviewed whole-fish/specimen data.

## Architecture comparison

| Option | Strengths | Main risks | Decision |
| --- | --- | --- | --- |
| A: 12-class parts → association → synthetic box | Available before whole-fish labels; explainable; one model | Physical identity is ambiguous in overlap/fragmentation; synthetic boxes cannot be cleanly passed into Ultralytics `model.track()` | Offline/provisional fallback only |
| B: tracked whole-fish segmentation + 12-class parts | ByteTrack operates on physical fish; masks constrain assignment; robust identity and explainable part evidence | Requires reviewed whole-fish annotations and two-model latency budget | Preferred final architecture |
| C: whole-fish four-class segmentation only | Simplest and fastest runtime | Discards explicit regional evidence and structural explainability | Retain as baseline |

Option B is selected. The existing whole-fish model remains the sole source of
ByteTrack IDs. The part model runs with ordinary prediction (not tracking), and
its outputs are assigned to whole-fish anchors. This avoids depending on the
installed Ultralytics release's internal `BYTETracker.update` API and prevents
part IDs from leaking into inspection counts.

Before whole-fish weights exist, Body detections may anchor provisional
candidates for offline evaluation. Unmatched parts remain separate uncertain
candidates instead of being forced into a neighboring fish. This mode is not
enabled in the dashboard and must not create production inspection events.

## Association and orientation

The prototype performs a separate global Hungarian assignment for each region,
with at most one Head, Body, and Tail per whole-fish anchor. Body is the anchor
in fallback mode. Scores combine:

- containment/intersection for reviewed whole-fish boxes;
- center distance normalized by fish/body scale;
- lateral alignment with the principal axis;
- expected longitudinal adjacency;
- relative part area;
- detector confidence.

Assignments below `part_association_threshold` are rejected. A part cannot be
assigned to multiple fish. Head and Tail links on the same side of a Body are
anatomically incompatible; the lower-confidence link is removed. This is still
a prototype: dense crossing fish and badly fragmented scenes require masks and
reviewed association ground truth to validate the score.

Orientation is estimated in this order:

1. Tail-to-Head center vector when both are observed;
2. PCA of a Body mask;
3. PCA of a whole-fish anchor mask;
4. the Body/anchor bounding box's major axis.

The whole-fish box is the union of all accepted part boxes and, when supplied,
the tracked whole-fish anchor. The anchor mask is retained; disconnected part
masks are not joined with invented background-spanning polygons.

## Structural reasoning

All Head, Body, and Tail fields are optional. Missing detections initially
contribute exactly zero structural rejection evidence.

For a missing Head or Tail, the assessor estimates its expected endpoint from
the Body axis and the observed opposite part. If that endpoint falls within the
configured frame margin, it is potentially observable. If it intersects the
frame margin, it is unobservable and receives no physical-loss evidence. If an
other fish candidate covers the expected endpoint above the occlusion
threshold, the state is uncertain and receives no physical-loss evidence.

Structural rejection is accepted only from an explicit physical-absence input,
such as a reviewed termination/damage model or human-confirmed structural cue,
and only when the expected region is visible and occlusion is low. Repeated
nondetection alone never creates that input. Observed recovery multiplies the
temporal structural confidence by `structural_recovery_decay`.

Statuses are `complete`, `partial`, `uncertain`, `likely_damaged`, and
`structurally_rejected`. The latter is an evidence status, not a mandatory
verdict override under the default policy.

## Quality and temporal fusion

For visible part `p` with predicted categorical grade `g`:

```text
part_evidence[g] += confidence[p] × anatomical_weight[p]
```

Verified structure contributes:

```text
evidence[Rejected] += structural_rejection_confidence × structural_rejection_weight
```

For each of the four thesis classes:

```text
percentage[class] = evidence[class] / sum(all evidence) × 100
```

The highest percentage wins. Numeric class IDs are never averaged. The default
`weighted_vote` policy does not override the winner merely because structural
evidence exists. `strict_structural_rejection` is implemented as a disabled
future policy.

Temporal state is keyed by the whole-fish ByteTrack ID and accumulates complete
four-class part-evidence vectors, not only the frame winner. Structural evidence
is maintained separately so a transient old damage hypothesis cannot grow once
per frame and dominate indefinitely. Reappearance decays it. An uncertain
result is provisional and is not a fifth class.

## Prototype configuration

`configs/part_fusion.yaml` contains non-authoritative defaults:

| Setting | Default |
| --- | ---: |
| Head / Body / Tail weights | 1.0 / 1.0 / 1.0 |
| Structural rejection weight | 1.0 |
| Minimum parts for verdict | 1 |
| Minimum total frame evidence | 1.5 |
| Minimum track observations | 2 |
| Part association threshold | 0.45 |
| Minimum association confidence | 0.45 |
| Structural rejection threshold | 0.80 |
| Temporal evidence limit | 120 |
| Image boundary margin | 0.05 of frame |
| Occlusion threshold | 0.30 |
| Structural recovery decay | 0.25 |
| Verdict policy | weighted_vote |

One part is permitted because Body-only or fragment candidates can be valid,
but a single ordinary confidence observation remains provisional under the 1.5
minimum evidence and two-observation temporal default. Thresholds require
validation and are not claims about fish biology or grading standards.

## Part-model training

The source YOLO export can train an auxiliary 12-class YOLOv8n-seg model. COCO
remains the richer archival source, but the YOLO labels are training-ready and
the six COCO RLE masks do not justify an unsafe lossy conversion. Most source
polygons are rectangular, so mask quality will be limited; a YOLOv8n detection
baseline should also be measured.

Preflight only:

```powershell
python -m src.training.train_yolo_parts --dry-run
```

The data itself passes, but evaluation is not class-complete: validation lacks
source IDs 3 and 4, while test lacks IDs 0, 1, and 2. The recommended action is
to create a source/specimen-group-safe, class-complete split before training.
For an explicitly exploratory run on the exported split:

```powershell
python -m src.training.train_yolo_parts `
  --acknowledge-part-only `
  --allow-incomplete-evaluation `
  --run-name v7_parts_yolov8n_seg_exploratory
```

Expected weights:
`models/yolo_v7_parts/v7_parts_yolov8n_seg_exploratory/weights/best.pt`.
These weights must never be supplied as the dashboard's whole-fish
`LEMURU_WEIGHTS` model.

## Required validation

The next experiment should label a balanced video subset with whole-fish masks,
persistent specimen IDs, part-to-fish links, occlusion/truncation flags, and
expert-confirmed physical-loss evidence. Compare:

1. whole-fish YOLOv8n-seg alone;
2. Option B with the auxiliary part model;
3. the provisional Option A association output offline.

Report association precision/recall, part-to-wrong-fish rate, ID switches,
false structural-rejection rate, four-class calibration, line-event accuracy,
latency, and FPS for isolated, touching, overlapping, boundary-clipped,
complete, and genuinely damaged fish. Known failure cases are overlapping
parallel fish, missing Body anchors, several fragments close together,
rectangular masks, unmodeled white/padded usable-image boundaries, and physical
loss without an explicit termination/damage cue.

## Implementation inventory and verification

Created for the part-fusion prototype:

- `src/inference/part_fusion.py`: candidate representation, global part
  association, orientation, conservative structural assessment, categorical
  evidence fusion, and track-keyed temporal accumulation.
- `src/inference/part_model.py`: strict Ultralytics adapter for untracked
  12-class part predictions.
- `src/training/train_yolo_parts.py`: guarded auxiliary-model preflight and
  trainer; training requires explicit part-only acknowledgement.
- `configs/part_fusion.yaml`: centralized evidence weights and prototype
  thresholds.
- `configs/yolo_parts.yaml`: auxiliary YOLOv8n-seg training configuration.
- `tests/test_part_fusion.py`: CPU-only association, structure, fusion,
  temporal identity, and one-event tests.

No existing API or frontend file was changed for this prototype. The current
dashboard and ByteTrack path therefore remain intact. Runtime activation waits
for both a reviewed whole-fish model and trained auxiliary part weights; at
that point the existing tracked whole-fish ID becomes the fusion key.

Verification on 2026-09-06:

- `test_part_fusion.py`: 15/15 passed.
- `test_v7_export_audit.py`: 12/12 passed.
- `test_segmentation_dataset.py`: 6/6 passed.
- `test_operator_dashboard.py`: 21/21 passed.
- Python syntax compilation passed for the new inference/training modules.
- Part trainer dry-run passed without creating model or dataset files; it
  correctly reported the incomplete exported evaluation splits.
- Full discovery: 94/96 passed. The two failures are pre-existing and outside
  this implementation: the handcrafted-feature diagonal-orientation assertion
  and a missing `annotation_index` constructor argument in specimen-manifest
  loading.
