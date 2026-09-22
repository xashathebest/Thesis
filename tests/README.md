# Model-free test suite

The normal suite exercises the canonical local inspection system without a
webcam, GPU, network service, secret, or model checkpoint. It covers the shared
live/offline grading policy, fixed anatomy weights, temporal aggregation,
tracking/counting invariants, camera-control readbacks, SQLite persistence, and
CSV/XLSX exports.

Run it from the repository root:

```powershell
py -m unittest discover -s tests -v
```

Tests that can use an actual camera or supplied checkpoint are opt-in smoke tests.
They skip cleanly unless their documented local assets and environment variables
are present. The normal GitHub Actions workflow runs only the model-free suite.

`test_dataset_independence.py`, `test_camera_calibration.py`,
`test_camera_calibration_ui.py`, `test_validation_study.py`, and
`test_end_to_end_validation.py` also exercise the
research-only independent-study safeguards: lineage leakage, camera confirmation,
locked manifests, blind truth, retained Ungraded outcomes, and performance-label
gating. They do not run a checkpoint or certify a model.
