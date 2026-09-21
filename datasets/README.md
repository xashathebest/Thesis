# Future independent-validation datasets

This tree intentionally contains no images or labels. It reserves separate
locations for future data collection and must never be populated by copying,
augmenting, or deriving files from a different split.

- `development/` — data eligible for development-only experiments.
- `validation_independent/` — independently acquired data for protocol and
  threshold selection only.
- `test_locked/` — independently acquired data that stays untouched until the
  final locked evaluation.

Use `python -m src.evaluation.dataset_independence` before treating any split as
independent. A proposed manifest is for review only and never moves data.

For each future image/frame manifest row, record at least `sample_id`,
`source_group_id`, `image_path`, `annotation_path`, `class_information`, `split`,
and `original_source` when known. Also retain recording/camera session, physical
fish ID where available, and augmentation-parent relationships. A new filename is
not evidence of a new physical sample.
