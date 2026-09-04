# Dataset Manifests

Generated audit and provenance tables live here. Raw images are never deleted
or overwritten. A status such as `EXCLUDE_FROM_TRAINING` changes only canonical
dataset membership.

Expected outputs include:

- `image_inventory.csv`
- `exact_duplicates.csv`
- `near_duplicates.csv`
- `specimen_groups.csv`
- `split_manifest.csv`

Near-duplicate matches are review candidates. They must be grouped for splitting
but should not be deleted automatically.
