# Handcrafted feature inventory and QA

Audit date: 2026-10-17. These measurements are not validated fish grades or localized crack/hole detections. **Automatic / offline research** below means computed from the supplied image and mask in the offline pipeline; **manual / offline research** means supplied by a reviewer; **live diagnostic** means reported without changing the verdict. No listed feature is grade-changing in the current two-YOLO runtime.

## Lineage and limits

`src/preprocessing/instance_processing.py:process_instance_manifest` consumes existing approved/adjudicated individual masks (unless a recorded research override is used), extracts source crops, normalizes their principal axis, records mask/crop hashes and reviewer/direction metadata. `src/features/handcrafted.py:extract_manifest_features` reads the normalized RGB crop and same-size binary mask, validates hashes when supplied, defaults to 500 mask pixels, and skips non-gradable, occluded and truncated records unless explicitly overridden. It writes morphology, color, texture, structure and combined CSVs plus a status manifest and settings/hash JSON. `src/training/feature_dataset.py:load_feature_table` excludes `annotated_*`, manual anatomy fields and annotation-completeness fields; it checks specimen/batch/session split leakage. `src/training/train_feature_models.py` fits median imputation and model preprocessing on training rows, evaluates validation separately, and stores the feature names and input hash. `src/evaluation/evaluate_feature_model.py` uses the locked test split separately. Feature-column discovery does inspect all eligible rows for numeric/finite availability; it is not a train-only schema selection and deserves review before formal locked-test experiments. Unknown numeric metadata fields could also be picked up unless explicitly excluded; use a reviewed schema for new inputs.

The live `src/inference/grading_engine.py:_measurement_payload` calls `extract_color_features` with an **all-ones rectangular** Model 1 crop or Model 2 box. It reports mean/median hue, mean saturation/value, and yellow/brown/dark/discoloration proxies as **live diagnostics**. These ROIs may include conveyor background or another fish and are not equivalent to masked offline measurements. No live mask, morphology, texture, structural model, or feature-trained classifier changes grading.

## Color and brightness — `src/features/color.py:extract_color_features`

All rows in this table are **automatic / offline research** unless identified as live diagnostics above. RGB is normalized to [0,1]; HSV hue is degrees [0,360), saturation/value [0,1]; CIE Lab uses D65 (L approximately 0–100 for valid sRGB, a/b signed). Counts are non-negative integers and coverage ratios are [0,1].

| Exact output names | Calculation / unit and conditions |
|---|---|
| `color_fish_pixels` | Foreground-pixel count. |
| `mean_r`, `mean_g`, `mean_b_rgb` | Masked channel means in [0,1]. |
| `mean_lab_l`, `median_lab_l`, `std_lab_l`, `mean_lab_a`, `mean_lab_b`, `lab_b_p90` | Masked Lab means, median, standard deviation or 90th percentile; Lab units. |
| `circular_mean_hue_deg`, `median_hue_deg` | Saturation-weighted circular hue mean and ordinary hue median; circular mean is NaN when all pixels have zero saturation. |
| `mean_saturation`, `median_saturation`, `mean_value`, `median_value` | Masked HSV saturation/value summaries; value is a camera-dependent brightness proxy. |
| `yellow_ratio_proxy`, `brown_ratio_proxy`, `dark_patch_ratio_proxy`, `discoloration_coverage_proxy` | Fraction of fish pixels satisfying configured HSV yellow/brown, Lab dark, or their union; overlapping masks are unioned, not summed. Thresholds in `configs/preprocessing.yaml` define proxies, not biological or grading cutoffs. |
| `largest_yellow_region_ratio_proxy`, `largest_brown_region_ratio_proxy`, `largest_discoloration_region_ratio_proxy`, `yellow_region_count_proxy`, `brown_region_count_proxy`, `discoloration_region_count_proxy` | Largest connected proxy component divided by fish pixels; component counts. |
| `specular_ratio_proxy`, `largest_highlight_ratio_proxy`, `highlight_region_count_proxy` | High-value/low-saturation pixel fraction, largest component fraction and component count. Highlight pixels are a reflectance proxy, not a calibrated reflectance measurement. |
| `head_tail_direction_known` | Boolean metadata; not a color value. |
| `{yellow,brown,dark,discoloration}_{head,anterior,mid,posterior,tail}_ratio_proxy` | 20 columns: fraction of foreground pixels in each crop-width fifth satisfying each proxy. Only calculated when metadata confirms head-left direction; missing/empty regions are NaN. Crop columns and PCA orientation are not automatically anatomical landmarks. |

## Texture and gradients — `src/features/texture.py:extract_texture_features`

All are **automatic / offline research**, dimensionless unless a pixel count is stated. Grayscale is a weighted sum of normalized RGB. GLCM counts pairs with both endpoints inside the mask at four configured offsets; defaults are 16 gray levels and one-pixel distance. LBP uses fully masked 3×3 neighborhoods. Gradients use fully masked eroded interior pixels; thin masks with no such neighborhood now report NaN instead of background-contaminated gradient values.

| Exact output names | Calculation / unit and conditions |
|---|---|
| `texture_fish_pixels`, `glcm_valid_pair_count`, `lbp_valid_pixel_count`, `texture_interior_pixel_count` | Counts of mask pixels, valid GLCM pairs, valid LBP centers, and eroded-interior gradient pixels. |
| `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_entropy`, `glcm_correlation` | Symmetric normalized co-occurrence statistics: squared gray-level difference, inverse squared-difference homogeneity, sum of squared probabilities, base-2 entropy and correlation. Contrast is 0–(levels−1)², homogeneity/energy [0,1], entropy nonnegative; zero pairs yield NaN; zero-variance correlation is conventionally 1. |
| `lbp_uniform_bin_0` through `lbp_uniform_bin_8`, `lbp_uniform_bin_nonuniform`, `lbp_entropy`, `lbp_nonuniform_ratio` | Ten normalized 8-neighbor uniform-pattern bins, base-2 histogram entropy, and nonuniform bin fraction. Bins/ratio [0,1]; no valid neighborhood yields NaN. |
| `edge_density_proxy`, `mean_gradient_proxy`, `gradient_std_proxy`, `gradient_p90_proxy`, `local_edge_variance_proxy` | Fraction of interior gradients above the configured threshold, mean, standard deviation, p90 and variance of normalized grayscale gradient magnitude. They are texture proxies, not confirmed surface defects. |

## Morphology — `src/features/morphology.py:extract_morphology_features`

All are **automatic / offline research**, based on a single supplied mask; native pixels are not physical lengths and two-pixel minimum is required. PCA projection and a skeleton geodesic approximation determine axes/length; a PCA fallback is documented in `length_method`. Widths are local transverse spans near 20–80% of the PCA axis, not verified anatomical widths.

| Exact output names | Calculation / unit and conditions |
|---|---|
| `fish_area_pixels`, `fish_length_pixels`, `major_axis_pixels`, `minor_axis_pixels`, `max_width_pixels`, `bbox_width_pixels`, `bbox_height_pixels`, `fish_perimeter_pixels_4n`, `convex_hull_area_pixels`, `convex_hull_perimeter_pixels` | Foreground/hull areas in pixel² and lengths in pixels; perimeter uses a four-neighbor digital boundary. |
| `width_20_pixels`, `width_30_pixels`, `width_40_pixels`, `width_50_pixels`, `width_60_pixels`, `width_70_pixels`, `width_80_pixels` | Transverse pixel spans in configured PCA-axis windows; nearest pixel fallback. |
| `normalized_width_20`, `normalized_width_30`, `normalized_width_40`, `normalized_width_50`, `normalized_width_60`, `normalized_width_70`, `normalized_width_80`, `width_length_ratio`, `normalized_area`, `middle_body_fullness` | Dimensionless widths divided by centerline length, area divided by length², and median middle width divided by length. |
| `solidity`, `contour_irregularity`, `convexity`, `eccentricity`, `symmetry_iou`, `centerline_tortuosity`, `curvature_proxy` | Area/hull area, digital perimeter/hull perimeter, inverse perimeter ratio, PCA eccentricity, reflection IoU, length/major-axis span, and max(0, tortuosity−1). Solidity, eccentricity and IoU are [0,1]; digital perimeter ratios depend on rasterization. |
| `length_method`, `mask_component_count`, `largest_component_fraction`, `skeleton_converged`, `skeleton_iterations`, `skeleton_pixels` | Method text, component count/fraction, convergence flag, iteration count and skeleton pixel count; diagnostics rather than physical defect labels. |

## Structure — `src/features/structure.py:extract_structural_features`

`structural_mask_component_count`, `structural_largest_component_fraction`, `structural_second_component_fraction`, `structural_mask_touches_border` are **automatic / offline research** mask diagnostics. They cannot establish anatomical absence. `annotated_head_present`, `annotated_tail_present`, `annotated_body_complete`, `annotated_body_continuity`, `annotated_severe_structural_damage`, `annotated_full_body_morphology`, `annotated_moderate_surface_defect`, `annotated_discoloration_present`, `annotated_body_fullness_score`, `annotated_surface_defect_severity`, `annotated_discoloration_severity`, `annotated_missing_body_ratio`, `annotated_exposed_skeleton_ratio` are **manual / offline research**; missing values are NaN. Boolean fields use 0/1, ordinal fields map none/low/moderate/high to 0–3 or accept reviewed numeric input. `structural_annotation_fields_available` and `structural_annotations_complete` describe annotation coverage and are excluded from model inputs. The live explicit structural-review API is separate and never changes the grade.

## Robustness findings

- Empty masks and mismatched/grayscale/alpha arrays are rejected by low-level color/texture functions. The offline loader converts decoded files to RGB: grayscale is replicated and alpha is discarded rather than used as a mask; inspect transparent images before use. Nonfinite or out-of-range float RGB is rejected. NaN can intentionally represent an undefined feature; infinite feature values are rejected by the training loader.
- Very thin/tiny masks can have no GLCM pairs, valid LBP centers or eroded gradient interior; affected outputs are NaN. In production extraction the default minimum is 500 foreground pixels, but area alone does not guarantee a valid interior.
- Touching fish or mask contamination cannot be reliably detected by the feature code; component diagnostics are not proof that a mask contains one fish. Review masks and source lineage. A crop touching a border can truncate shape, and preprocessing records border contact; standard table extraction skips manifest-marked truncated/occluded rows.
- Overexposure, shadow, camera white balance, lighting and JPEG artifacts affect color, brightness, highlight and texture proxies. No calibration establishes biological thresholds. Pixel-based widths, perimeter, GLCM distance, LBP and gradients depend on image resolution and resampling; normalized shape ratios do not eliminate all scale effects.
- Existing tests cover known color references, a uniform surface and mask exclusion. Additional QA covers deterministic background isolation, malformed inputs, tiny masks, infinities and exclusion of reviewed annotations. Model-free baseline verdict tests confirm this audit did not enable feature grading.
