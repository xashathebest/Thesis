# Dataset Version

Current dataset version: not assigned yet

Date: not assigned yet

Images: 0

Annotation status: pending

Train/validation/test status: not created yet

Important changes:
- Initial canonical dataset layout and validation tooling are in place.
- No research dataset has been frozen yet.
- The final dataset version should be changed only after the research team approves the canonical annotated dataset.

## Dataset Freeze

A dataset freeze means the approved dataset version is locked for the model comparison stage.

Rules after the freeze:
- Do not silently change annotations in the frozen version.
- Do not change the train, validation, or test split for the frozen version.
- If an annotation error is found later, create a new dataset version instead of altering the frozen research data.
