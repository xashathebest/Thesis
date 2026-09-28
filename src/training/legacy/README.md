# legacy/

Files in this directory are **archived** — they are not part of the active Tamban runtime inference path.

## tamban_v3_colab_workbench.py

Original Tamban V3 Colab training/evaluation workbench script.

- Architecture: YOLOv26-M (Ultralytics 8.4.153)
- Intended environment: Google Colab
- Purpose: Tamban-graded individual-fish detection + part quality workbench
- Status: Archived; not imported or called by any active module
- Notes: Contains the original training loop, loss, and evaluation logic for the
  evidence-gated grading experiment. Reference only — do not run locally without
  the Colab environment and required GPU.

> **Important**: This file uses rom google.colab import files and scipy/pandas
> APIs that are not installed in the local virtual environment. It will fail to
> import locally; this is expected.
