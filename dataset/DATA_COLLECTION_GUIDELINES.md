# Data Collection Guidelines

This document describes the recommended metadata to record when collecting real 4K camera images for the Sardinella lemuru thesis project.

## Recommended Metadata Fields

Record one metadata row per image where possible.

- `image_id`: a unique image identifier
- `capture_session`: the capture session or batch name
- `capture_date`: the date the image was captured
- `camera_resolution`: the original camera resolution
- `camera_position`: notes about the fixed 4K camera setup
- `lighting_condition`: lighting description at capture time
- `conveyor_condition`: conveyor state or speed if relevant
- `fish_class`: the assigned fish class, if known
- `notes`: free-text notes for anomalies or review comments

## Suggested CSV Layout

A simple CSV file is sufficient for the initial project stage.

Example header only:

```csv
image_id,capture_session,camera_resolution,camera_position,lighting_condition,conveyor_condition,fish_class,notes
```

## Collection Notes

- Keep the original raw images untouched in `dataset/raw/`.
- Do not encode metadata into filenames unless the research team explicitly agrees that it is necessary.
- Store metadata separately so the image source data stays clean and reusable.
- Use the same capture and annotation rules for every session to reduce inconsistency.
