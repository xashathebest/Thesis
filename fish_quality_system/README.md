# Dried Tamban Inspection System

This is a local, modular conveyor-inspection foundation. It deliberately does
not fabricate production predictions: Model 1 and Model 2 adapters raise a
clear error until trained weights and their local runners are supplied.

## What it does now

`vision/` performs image-quality checks, fish-mask association/cropping,
PCA-based alignment, mask-based pixel measurements, shininess estimation, and
temporary-occlusion-aware tracking. `grading/` applies explainable `D > C > B
> A` rules. A missing anatomical mask is **not** treated as a missing part;
only `confirmed_missing_parts` from a clear real-model/temporal evidence layer
can produce Grade D. Uncertain anatomy is `NEEDS_REVIEW`.

`database/` saves one SQLite record per persistent fish ID. `reports/` writes a
CSV and an XLSX workbook with `Individual Fish` and `Batch Summary` sheets.
All yellowing, wide-body, deformation, dark-discoloration, and shininess grade
thresholds default to `None` in `config.py`. Set them only from a labeled
calibration study; until then ambiguous threshold-dependent results are
reviewed rather than silently classified.

## Windows setup

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r fish_quality_system\requirements.txt
```

Python 3.11 or 3.12 is the supported target. If PowerShell blocks activation,
run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal first.

## Run

With real local adapters configured and weights installed:

```powershell
python -m fish_quality_system.main --image path\to\frame.jpg --batch-id BATCH-2026-09-06-001
python -m fish_quality_system.main --video path\to\conveyor.mp4 --batch-id BATCH-2026-09-06-001
```

To test the database, tracking finalization, grading and report path before
models are trained, use the deliberately simulated test-only mode:

```powershell
python -m fish_quality_system.main --demo --mock-mode --batch-id BATCH-2026-09-06-001
python -m unittest discover -s fish_quality_system\tests -v
```

`--mock-mode` never loads or represents a trained model. Its output is marked
simulated in the code and must not be used for inspection decisions.

## Weight integration

Place Model 1 whole-fish/surface-defect instance-segmentation weights at
`fish_quality_system/weights/model1_surface.pt`, and Model 2 anatomical
segmentation weights at `fish_quality_system/weights/model2_parts.pt`.

Then implement the model-format-specific local calls in
`models/model1_surface.py` and `models/model2_parts.py`. Both adapters already
enforce a local-file check and return shared `MaskDetection` contracts. Model 1
must output `fish`, `crack`, `yellowing`, and optional dark/broken-surface masks;
Model 2 must output `head`, `body`, and `tail` masks plus explicit clear-view
missing-part evidence where appropriate. No Roboflow cloud inference is needed.

Exports and the default SQLite database are written to
`fish_quality_system/outputs/`.
