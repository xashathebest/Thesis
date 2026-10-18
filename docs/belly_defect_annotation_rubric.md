# Belly-region annotation rubric — `tamban-belly-draft-v0.1`

**Status: PENDING EXPERT VERIFICATION.** This is a proposed research annotation form, not an accepted grading rule or detector specification. An apparent opening in an image may be normal processing, a hole, a tear, exposed tissue, or something unclear. Neither the cause nor its grading consequence can be determined from the appearance alone. No quality threshold, class consequence, defect label, or expert decision is established by this draft.

## Unit of review and provenance

Review one physical fish in one recorded image or video frame. Give it a specimen ID and a fish/image instance ID, identify the image or video and frame, image dimensions, and the annotator. Use a separate observation row for each localized belly finding, and link all rows to the same fish/image instance. For a fish with no localizable finding, use one observation row and leave the location empty. Keep source images immutable, and record any edits or resizing so coordinates can be mapped back to the source. Expert or second-review decisions are recorded with reviewer ID, consultation reference and actual date only when they occur. Protect reviewer and specimen identifiers according to the study protocol; these IDs alone do not authenticate an expert.

## Observation fields

| Field | Proposed allowed values | Meaning / use |
|---|---|---|
| `belly_defect_present` | `yes`, `no`, `uncertain` | Whether an expert considers a visible finding a belly defect. Use `uncertain` when the distinction from normal preparation or another appearance cannot be resolved; do not use `no` solely because the belly is hidden. |
| `belly_defect_type` | `hole`, `tear`, `exposed_tissue`, `acceptable_processed_opening`, `other`, `uncertain` | Expert's description of the visible finding, if any. `acceptable_processed_opening` is a separate observation, not automatically a defect. For no visible finding leave blank. Give an explanatory note for `other`. |
| `belly_defect_severity` | `none`, `minor`, `moderate`, `severe`, `uncertain` | Proposed ordinal categories only. The distinctions and any relationship to quality are **not defined or approved**. Use `none` only for an expert's no-defect determination; use `uncertain` when severity cannot be assessed. Do not assign physical cutoffs. |
| `visibility_state` | `visible`, `partially_visible`, `occluded`, `out_of_frame`, `image_quality_insufficient`, `uncertain` | Visibility of the relevant belly area, separate from whether a defect exists. A partially visible area may require `uncertain` even if other regions appear normal. |
| `expert_grading_consequence` | `no_effect`, `downgrade`, `reject`, `unassessable`, `pending` | Expert-reported effect, not an automatic system decision. Use `pending` before consultation; `unassessable` only when the expert says so. |
| `proposed_origin` | `accidental_tear`, `normal_preparation`, `other`, `uncertain` | A separate expert-reported hypothesis, not a fact inferred from a photo. Leave blank until assessed or use `uncertain`. |
| `head_physical_state`, `tail_physical_state` | `present`, `confirmed_absent`, `unknown_or_occluded` | Physical anatomy evidence only. Unobserved is not absent; use `confirmed_absent` only with separately recorded, adequate expert evidence. |
| `head_visibility_state`, `tail_visibility_state` | same vocabulary as belly `visibility_state` | Visibility is distinct from physical state. |
| `class_a_effect`, `class_b_effect`, `class_c_effect` | `permitted`, `limited`, `not_permitted`, `unassessable`, `pending` | Expert-reported applicability for each class. These are draft questions, not known answers. Rejection is separately recorded in `expert_grading_consequence`. |

If multiple findings exist, annotate each individually and record a fish-level conclusion only after the expert determines how they interact. A finding may require multiple views or physical inspection; record evidence and limitations in `notes`, and leave unresolved fields uncertain or pending. Do not infer severe damage, missing anatomy, or a grade from another field or detector output.

## Location

Use a polygon around the visible opening/tear or a tight rectangle when a polygon is unavailable. Coordinates are in original-frame pixels, with origin at top left and image dimensions recorded. Polygon format is a JSON array `[[x,y], ...]`; the box format is JSON `[x_min,y_min,x_max,y_max]` with `x_max > x_min`, `y_max > y_min`, bounded by image dimensions. Store one geometry per finding; use blank geometry when the finding cannot be localized. If the belly is outside the frame, do not invent a location. This geometry is a proposed labeling convention and needs expert and annotator approval.

## Independent second review and adjudication

1. Two trained reviewers independently annotate uncertain, borderline, consequential, or disputed findings, and a predeclared sample of apparently clear cases. Neither sees the other's answer before submitting. Record the chosen sampling protocol; do not manufacture agreement statistics.
2. Compare presence, type, visibility, severity, anatomy, consequence and geometry separately. Record disagreements and the evidence each reviewer used; do not resolve them by averaging grades or treating missing data as no defect.
3. Refer unresolved findings to a qualified adjudicator with access to the source and, if necessary, physical specimen or additional views. Record adjudicator, actual date, reference, outcome and rationale. Until then retain `uncertain`/`pending`; preserve both original reviews and the adjudication as separate immutable records linked by fish and finding IDs.
4. Version any expert-approved rubric separately. Archive the prior draft and version the schema, expert-approved category definitions, allowed grades, examples, and date of approval. Consult the expert again after substantive changes.

The [consultation checklist](belly_defect_expert_checklist.md) lists the unanswered decisions. The [blank CSV template](belly_defect_annotation_template.csv) contains column names only; its empty state is not a dataset or expert response.
