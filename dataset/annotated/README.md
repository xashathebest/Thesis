# Annotated Dataset

This folder stores the labeled dataset used for object detection experiments.

Structure:
- `images/` for annotated images
- `labels/` for YOLO-compatible label files

Rules:
- Each image must have a matching label file with the same base name.
- Label format must be `class_id x_center y_center width height`.
- Bounding box coordinates must be normalized between 0 and 1.
- Do not place the same image in multiple splits.
