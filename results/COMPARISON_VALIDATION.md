# Comparison implementation validation

- All three canonical data preflights passed: 680 train, 115 validation, 115 test images, with image hashes checked against the manifest.
- Hardware: Intel64 Family 6 Model 141 Stepping 1; PyTorch 2.14.0+cpu; CUDA unavailable. torchvision 0.29.0, Ultralytics 8.4.142. effdet is unavailable.
- Final comparison unit suite: 10 tests passed. Covers AP/matching, IoU thresholds, fixed confidence, measured zero versus unavailable, provenance, CSV paths/schema and deterministic reporting.
- Every comparison notebook code cell executed successfully with the Agg plotting backend. The expected noninteractive-display warning does not affect notebook plots in Jupyter.
- Faster R-CNN adapter: untrained 128-pixel CPU forward smoke check passed. This is not a trained-model evaluation, and no measurements were entered into the comparison.
- Full repository suite before the last two comparison tests were added: 110 tests ran, 108 passed, one failure and one error. The affected code/tests have no changes in this implementation:
  - `test_diagonal_major_axis_is_rotated_horizontal`: major_vector_x was 0.0 rather than above 0.98.
  - `test_specimen_manifest_rejects_duplicate_instance_link`: SpecimenAssociation construction omitted required annotation_index.
- `git diff --check` passed.

No full training was attempted on CPU. No pretrained trained checkpoints were present at initial inspection. EfficientDet training/forward execution remains unverified because effdet is missing. Full training, checkpoint selection, held-out inference and GPU timing must be verified on the intended training machine. The provided commands and dependency file prepare that work; they do not constitute measured experiments.
