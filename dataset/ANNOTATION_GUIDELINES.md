# Annotation Guidelines

These guidelines describe how the research team should annotate Sardinella lemuru images for the thesis dataset.

## Basic Rules

- Draw one bounding box around each visible fish.
- Keep the bounding box as tight as reasonably possible.
- Assign exactly one quality class per fish.
- Do not include large amounts of conveyor belt or background inside the box.
- Keep annotation style consistent across all annotators.
- Do not change labels intentionally just to balance the dataset.

## Class Labels

- `0 = First Class`
- `1 = Second Class`
- `2 = Fatty/Oily`
- `3 = Rejected`

## Special Cases

### Partially Visible Fish

Annotate a partially visible fish only if the fish can still be identified with reasonable confidence.

RESEARCH TEAM DECISION REQUIRED: whether a specific partial view is acceptable for annotation in borderline cases.

### Overlapping Fish

If two fish can still be separated visually, draw separate boxes for each fish.

RESEARCH TEAM DECISION REQUIRED: how to label cases where overlap is so strong that the fish boundaries cannot be separated reliably.

### Blurry or Unusable Images

Do not guess labels for fish that cannot be identified with confidence.

RESEARCH TEAM DECISION REQUIRED: whether a blurry or unusable image should be excluded entirely or kept for review.

### Uncertain Quality Class

If the quality class cannot be determined confidently, do not invent a label.

RESEARCH TEAM DECISION REQUIRED: the final policy for uncertain class assignment.

## YOLO Label Format

Each image must have a matching label file with the same base name.

Example:

- `dataset/annotated/images/fish_0001.jpg`
- `dataset/annotated/labels/fish_0001.txt`

Each YOLO line must follow:

`class_id x_center y_center width height`

All bounding box coordinates must be normalized between `0` and `1`.
