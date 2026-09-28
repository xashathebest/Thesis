# YOLOv26 Models

This folder contains YOLOv26-M checkpoints trained as **Model 2** (part-quality detector)
for the Tamban Sardinella lemuru grading system.

## Checkpoints

| File | SHA-256 (prefix) | Size | Status |
|------|-----------------|------|--------|
| model2_tamban_yolo26m_partdet_r03_devbest.pt | 74cd336b6ba | 42 MB | Experimental — not in active runtime |

## Classes (12-class part schema)

| ID | Name |
|----|------|
| 0 | Grade_A_Body |
| 1 | Grade_A_Head |
| 2 | Grade_A_Tail |
| 3 | Grade_B_Body |
| 4 | Grade_B_Head |
| 5 | Grade_B_Tail |
| 6 | Grade_C_Body |
| 7 | Grade_C_Head |
| 8 | Grade_C_Tail |
| 9 | Rejected_Body |
| 10 | Rejected_Head |
| 11 | Rejected_Tail |

## Usage

To test model2_tamban_yolo26m_partdet_r03_devbest.pt against the active runtime:

`powershell
# Windows PowerShell (from repo root)
 = "models\yolov26\model2_tamban_yolo26m_partdet_r03_devbest.pt"
py -m src.api
`

See models/MODEL_REGISTRY.json for full SHA-256 checksums and provenance.
