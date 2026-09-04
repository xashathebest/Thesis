# Instance Annotation and Grading Guidelines

These rules apply to dried *Sardinella lemuru* images used by the thesis. The
annotation unit is one physical fish or, after the research team defines the
sorting policy, one independently sortable rejected fragment. Raw folder names
are provenance hints and must not be copied blindly to every visible object.

## Final classes

- `0 = Class A`: substantially complete, normal/slender morphology, relatively
  clean silver/gray surface, and low defect burden. Small localized yellow or
  brown regions around the head, gills, belly edge, or tail are permitted.
- `1 = Class B`: substantially complete, with moderate discoloration, damaged or
  missing scales, scraping, rough texture, small holes/tears, or moderate local
  deformation. Yellow pixels alone are insufficient.
- `2 = Class C`: substantially complete with clearly broader/full middle-body
  morphology relative to length. Gloss is supporting evidence only.
- `3 = Rejected`: severe anatomical loss, major fragmentation, missing major
  body portions, exposed skeleton/spine, or major structural discontinuity.

The tentative `Rejected -> C -> B -> A` conflict priority must not be applied
until domain experts approve it. Record independent attributes first so policy
changes do not require re-annotating the visible evidence.

## Required instance geometry

- Draw one tight polygon mask around each separable fish. A bounding box may be
  exported as an additional annotation, but it does not replace the mask.
- Exclude neighboring fish and as much background as possible.
- Use a persistent `instance_id` and `specimen_id` across repeated frames.
- Do not merge touching fish into one instance.
- When boundaries cannot be resolved, set `occluded=true` and normally
  `gradable=false`; do not invent a contour.
- Mark fish clipped by the image boundary as `truncated=true`. Image truncation
  is not anatomical damage and must not independently cause a Rejected label.
- Do not infer that a hidden head or tail is physically missing.

## Required attributes

Record the fields in
`dataset/annotations/attributes/instance_attributes_template.csv`, including:

- provenance: specimen, batch, capture session, and scene;
- structural attributes: head/tail presence, body completeness/continuity, and
  severe structural damage;
- morphology: full-body indicator and fullness score;
- surface/color: defect and discoloration presence/severity;
- quality controls: occlusion, truncation, gradability, and confidence;
- independent expert labels and the adjudicated label.

Use an explicit unknown value when anatomy is not visible. Do not encode unknown
as `false`.

## Agreement and adjudication

1. Two qualified annotators independently label the gold-standard subset.
2. Calculate raw agreement and Cohen's kappa for final class. Use a suitable
   weighted agreement statistic for ordinal severity scores.
3. Send disagreements, low-confidence examples, B/C conflicts, and possible
   B/Rejected boundary cases to an expert adjudicator.
4. Preserve both original opinions and the adjudicated outcome.
5. Revise the rubric if qualified humans cannot distinguish a class pair
   consistently.

## YOLO segmentation label format

Each image in a materialized split has a same-stem `.txt` file. Each line is:

`class_id x1 y1 x2 y2 x3 y3 ...`

Coordinates are normalized to `[0, 1]` and contain at least three polygon
points. Run the segmentation preflight before training; five-value YOLO boxes
are deliberately rejected by that preflight.

## Manual-review conditions

Use `MANUAL_REVIEW` rather than guessing when:

- B and C characteristics coexist;
- moderate damage approaches severe anatomical loss;
- glare hides morphology or surface condition;
- overlap prevents a trustworthy individual mask;
- head or tail status is unknown because of framing;
- a detached fragment cannot be associated with its source fish.

The research team must document whether detached pieces are individual sortable
objects before annotation begins.
