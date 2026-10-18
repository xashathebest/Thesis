# Independent end-to-end study — BLOCKED

No real fish-level accuracy, counting, latency or fusion performance has been measured by this preflight.

Inspected COCO test: 17 images; part-annotation counts: `{"Grade_C_Head": 30, "Grade_C_Body": 43, "Grade_C_Tail": 17, "Rejected_Body": 56, "Rejected_Tail": 28, "Rejected_Head": 47, "Grade_B_Head": 2, "Grade_B_Body": 1, "Grade_B_Tail": 1}`. These are not fish-level ground truth.
Historical dataset-independence audit timestamp: `2026-09-22T04:17:38.486286+00:00`. Its recorded validation/test outcomes are in `preflight.json`; re-audit current data.

| Item | Status | Reason |
|---|---|---|
| independent_fish_level_study | BLOCKED | No verified independent expert-labeled fish manifest or locked session was supplied. |
| model1_detection_and_counting | NOT_RUN | No independent counting/detection truth was supplied. |
| E0_end_to_end_grading | NOT_RUN | No independent fish-level test was run. |
| E1_E2_fusion_comparison | BLOCKED | No validated same-fish feature artifact and holdout set were supplied. |
| E3_E4_structural_fusion | BLOCKED | Expert structural criteria and validated evidence are unavailable. |
| real_camera_latency | NOT_RUN | No locked real-camera latency study was supplied. |

`preflight.json` records actual file hashes and installed versions. A hash snapshot here is not a frozen study session. No real-data plot is generated without measured data.
