"""Run the trained Tamban YOLO26 part detector and optional body-region classifier.

This entry point deliberately preserves the evidence boundaries in the supplied
artifacts:

* ``last.pt`` detects one of 12 grade/region labels (A/B/C/Rejected x Body,
  Head, Tail).  It is a detection checkpoint, not an instance-segmentation or
  whole-fish checkpoint.
* ``body_feature_classifier.joblib`` predicts a grade for a *Body region* from
  44 saved visual features.  Its own payload says ``validated: false`` and
  ``target: BODY REGION grade, not whole-fish grade``.  Its result is therefore
  reported separately and never overrides the YOLO part evidence.

The two original ``Yolov26_*.py`` files are empty, so no prior executable
training or inference code is overwritten here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.inference.part_fusion import FishCandidate, PartDetection, PartFusionPipeline


# --- Configuration ---------------------------------------------------------
# These defaults resolve relative to this file, so execution from another
# working directory uses the same trained artifacts.
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "last.pt"
JOBLIB_MODEL_PATH = PROJECT_ROOT / "body_feature_classifier.joblib"
SOURCE: str | None = None
YOLO_CONF = 0.25
DEBUG = False
SAVE_CSV = True

GRADE_NAMES = {
    "A": "Grade A",
    "B": "Grade B",
    "C": "Grade C",
    "Rejected": "Rejected",
    "Uncertain": "Uncertain",
}

EXPECTED_YOLO_CLASSES = (
    "Grade_A_Body",
    "Grade_A_Head",
    "Grade_A_Tail",
    "Grade_B_Body",
    "Grade_B_Head",
    "Grade_B_Tail",
    "Grade_C_Body",
    "Grade_C_Head",
    "Grade_C_Tail",
    "Rejected_Body",
    "Rejected_Head",
    "Rejected_Tail",
)

# This exact order is stored both in the Joblib payload and in the training
# table.  Do not reorder it: scikit-learn receives this order verbatim.
BODY_FEATURE_COLUMNS = (
    "value_mean", "value_std", "value_p90", "value_p95",
    "saturation_mean", "saturation_std", "gray_mean", "gray_std",
    "hue_sin", "hue_cos", "chromatic_fraction", "yellow_candidate_ratio",
    "lab_L_mean", "lab_L_std", "lab_a_mean", "lab_a_std", "lab_b_mean",
    "lab_b_std", "highlight_candidate_ratio", "highlight_largest_ratio",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy",
    "lbp_entropy", "edge_density_proxy", "lbp_00", "lbp_01", "lbp_02",
    "lbp_03", "lbp_04", "lbp_05", "lbp_06", "lbp_07", "lbp_08",
    "lbp_09", "lbp_10", "lbp_11", "lbp_12", "lbp_13", "lbp_14",
    "lbp_15", "lbp_16", "lbp_17",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TambanInferenceError(RuntimeError):
    """Raised when trained-artifact inference cannot proceed safely."""


@dataclass(frozen=True)
class BodyPrediction:
    """A classifier result for one detected Body ROI, never a fish verdict."""

    grade: str
    confidence: float | None
    probabilities: dict[str, float] | None
    feature_count: int
    backend: str


@dataclass
class CandidateResult:
    """One associated, current-frame candidate and its optional body result."""

    candidate: FishCandidate
    body_prediction: BodyPrediction | None = None
    features: dict[str, float] | None = None
    body_error: str | None = None


class _PortableSklearnObject:
    """Capture sklearn pickle state when native sklearn cannot import.

    The local environment may block SciPy's native DLLs.  The supplied payload
    is a RandomForest with its full tree arrays in Joblib, so this narrow reader
    lets the program execute the saved forest exactly rather than faking a
    prediction.  Normal ``joblib.load`` remains the preferred route.
    """

    def __new__(cls, *args: object, **kwargs: object) -> "_PortableSklearnObject":
        del kwargs
        instance = super().__new__(cls)
        instance.constructor_args = args
        return instance

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __setstate__(self, state: object) -> None:
        self.state = state
        if isinstance(state, dict):
            self.__dict__.update(state)


class _PortableUnpickler:  # Constructed lazily so --help does not need Joblib.
    @staticmethod
    def load(path: Path) -> object:
        try:
            from joblib.numpy_pickle import NumpyUnpickler
        except ImportError as exc:
            raise TambanInferenceError(
                "Joblib is required to load the saved body classifier. "
                "Install it with `pip install joblib`."
            ) from exc

        class Reader(NumpyUnpickler):  # type: ignore[misc, valid-type]
            def find_class(self, module: str, name: str) -> object:
                if module.startswith("sklearn"):
                    return _PortableSklearnObject
                return super().find_class(module, name)

        with path.open("rb") as handle:
            return Reader(str(path), handle, True).load()


class PortableRandomForest:
    """Read-only RandomForest probability adapter backed by saved sklearn trees."""

    def __init__(self, forest: object) -> None:
        self._forest = forest
        self.classes_ = tuple(str(value) for value in getattr(forest, "classes_", ()))
        self.n_features_in_ = int(getattr(forest, "n_features_in_", 0))
        self._estimators = tuple(getattr(forest, "estimators_", ()))
        if not self.classes_ or not self._estimators or self.n_features_in_ <= 0:
            raise TambanInferenceError(
                "The Joblib fallback found no usable RandomForest classes, features, or trees."
            )

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise TambanInferenceError(
                f"Joblib classifier expects {self.n_features_in_} features but received "
                f"{values.shape[1] if values.ndim == 2 else 'an invalid vector'}.")
        if not np.isfinite(values).all():
            raise TambanInferenceError("The extracted body feature vector contains non-finite values.")

        probabilities = np.zeros((len(values), len(self.classes_)), dtype=np.float64)
        for estimator in self._estimators:
            tree = getattr(estimator, "tree_", None)
            nodes = getattr(tree, "nodes", None)
            leaf_values = getattr(tree, "values", None)
            if nodes is None or leaf_values is None:
                raise TambanInferenceError("A saved RandomForest tree is missing node or leaf data.")
            for row_index, row in enumerate(values):
                node = 0
                while int(nodes["left_child"][node]) != -1:
                    feature_index = int(nodes["feature"][node])
                    value = row[feature_index]
                    if math.isnan(value):
                        go_left = bool(nodes["missing_go_to_left"][node])
                    else:
                        go_left = value <= float(nodes["threshold"][node])
                    node = int(nodes["left_child"][node] if go_left else nodes["right_child"][node])
                probabilities[row_index] += np.asarray(leaf_values[node, 0], dtype=np.float64)

        probabilities /= len(self._estimators)
        totals = probabilities.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise TambanInferenceError("A saved RandomForest produced an empty probability vector.")
        return probabilities / totals


class BodyRegionClassifier:
    """Validate and query the saved 44-feature body-region RandomForest."""

    def __init__(
        self,
        model: object,
        feature_columns: Sequence[str],
        target: str,
        validated: bool,
        backend: str,
    ) -> None:
        self.model = model
        self.feature_columns = tuple(str(column) for column in feature_columns)
        self.target = target
        self.validated = bool(validated)
        self.backend = backend
        self.classes = tuple(str(value) for value in getattr(model, "classes_", ()))
        expected_count = int(getattr(model, "n_features_in_", len(self.feature_columns)))
        if not self.classes or expected_count != len(self.feature_columns):
            raise TambanInferenceError(
                "The Joblib classifier feature schema is inconsistent: "
                f"model expects {expected_count}, payload lists {len(self.feature_columns)}.")
        if tuple(self.feature_columns) != BODY_FEATURE_COLUMNS:
            raise TambanInferenceError(
                "The supplied Joblib feature schema does not match the audited 44-column body schema.\n"
                f"Expected: {list(BODY_FEATURE_COLUMNS)}\n"
                f"Found: {list(self.feature_columns)}")

    def predict(self, features: Mapping[str, float]) -> BodyPrediction:
        missing = [column for column in self.feature_columns if column not in features]
        unexpected = sorted(set(features).difference(self.feature_columns))
        if missing or unexpected:
            raise TambanInferenceError(
                "Body feature validation failed before Joblib inference.\n"
                f"Missing: {missing}\nUnexpected: {unexpected}\n"
                f"Expected order: {list(self.feature_columns)}")
        row = np.asarray([[features[column] for column in self.feature_columns]], dtype=np.float64)
        if not np.isfinite(row).all():
            raise TambanInferenceError("Body feature extraction returned NaN or infinity.")
        if not hasattr(self.model, "predict_proba"):
            raise TambanInferenceError("The saved body classifier does not support predict_proba().")
        probabilities = np.asarray(self.model.predict_proba(row)[0], dtype=np.float64)
        if probabilities.shape[0] != len(self.classes):
            raise TambanInferenceError("Joblib probability output does not match its saved class list.")
        winner = int(np.argmax(probabilities))
        return BodyPrediction(
            grade=self.classes[winner],
            confidence=float(probabilities[winner]),
            probabilities={label: float(value) for label, value in zip(self.classes, probabilities)},
            feature_count=len(self.feature_columns),
            backend=self.backend,
        )


def _payload_parts(payload: object, backend: str) -> BodyRegionClassifier:
    if not isinstance(payload, Mapping):
        raise TambanInferenceError("The Joblib object must be a metadata dictionary, not a bare estimator.")
    required = {"model", "feature_columns", "target", "validated"}
    missing = sorted(required.difference(payload))
    if missing:
        raise TambanInferenceError(f"Joblib payload is missing required keys: {missing}.")
    # The saved asset includes no scaler, encoder, or sklearn Pipeline.  Passing
    # the raw ordered table to this forest is therefore the intended path.
    return BodyRegionClassifier(
        payload["model"],
        payload["feature_columns"],
        str(payload["target"]),
        bool(payload["validated"]),
        backend,
    )


def load_joblib_assets(path: Path, *, prefer_portable: bool = False) -> BodyRegionClassifier:
    """Load the supplied Joblib payload, using a tree-faithful fallback if needed."""

    if not path.is_file():
        raise FileNotFoundError(f"Joblib classifier not found: {path}")
    standard_error: Exception | None = None
    if not prefer_portable:
        try:
            import joblib

            return _payload_parts(joblib.load(path), "joblib/scikit-learn")
        except Exception as error:
            standard_error = error
    # Some managed Windows environments block SciPy's native DLLs while
    # importing sklearn.  The fallback is specific to this saved RF payload.
    try:
        fallback_payload = _PortableUnpickler.load(path)
        classifier = _payload_parts(fallback_payload, "portable saved-RandomForest")
        classifier.model = PortableRandomForest(classifier.model)
        classifier.classes = classifier.model.classes_
        return classifier
    except Exception as fallback_error:
        standard_detail = f"Standard loader error: {standard_error}\n" if standard_error else ""
        raise TambanInferenceError(
            "Unable to load the saved Joblib body classifier.\n"
            f"{standard_detail}Portable RandomForest fallback error: {fallback_error}") from fallback_error


def find_model_files(
    yolo_override: Path | None = None,
    joblib_override: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve explicit files or the trained artifacts supplied at project root."""

    yolo_path = (yolo_override or YOLO_MODEL_PATH).expanduser()
    joblib_path = (joblib_override or JOBLIB_MODEL_PATH).expanduser()
    if not yolo_path.is_absolute():
        yolo_path = (PROJECT_ROOT / yolo_path).resolve()
    if not joblib_path.is_absolute():
        joblib_path = (PROJECT_ROOT / joblib_path).resolve()
    if not yolo_path.is_file():
        raise FileNotFoundError(f"Trained YOLO checkpoint not found: {yolo_path}")
    if not joblib_path.is_file():
        raise FileNotFoundError(f"Joblib classifier not found: {joblib_path}")
    return yolo_path, joblib_path


def load_yolo_model(path: Path) -> Any:
    """Load trained weights only; never initialize an untrained YAML architecture."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TambanInferenceError(
            "Ultralytics is required for YOLO inference. Install it with `pip install ultralytics`."
        ) from exc
    model = YOLO(str(path))
    if getattr(model, "task", None) != "detect":
        raise TambanInferenceError(
            f"Expected a detection checkpoint, but {path.name} reports task={model.task!r}.")
    names = getattr(model, "names", {})
    actual = tuple(str(names[index]) for index in range(len(names)))
    if actual != EXPECTED_YOLO_CLASSES:
        raise TambanInferenceError(
            "YOLO class mapping does not match the trained 12-class Tamban part model.\n"
            f"Expected: {list(EXPECTED_YOLO_CLASSES)}\nFound: {list(actual)}")
    return model


def inspect_joblib_model(classifier: BodyRegionClassifier) -> dict[str, object]:
    """Return concise, user-facing saved-model facts without inventing metadata."""

    return {
        "type": type(classifier.model).__name__,
        "backend": classifier.backend,
        "classes": list(classifier.classes),
        "n_features": len(classifier.feature_columns),
        "feature_columns": list(classifier.feature_columns),
        "target": classifier.target,
        "validated": classifier.validated,
        "separate_preprocessor": False,
    }


def _largest_component_ratio(binary: np.ndarray, denominator: int) -> float:
    if denominator <= 0 or not binary.any():
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / denominator)


def _masked_glcm(gray: np.ndarray, mask: np.ndarray, levels: int = 16) -> tuple[float, float, float, float]:
    """Calculate the saved GLCM family over body pixels only.

    The extractor source was not included with the artifacts; the fixed feature
    names, training table, and run config do establish this 16-level masked
    reconstruction.  It is labelled provisional through the saved model's own
    ``validated`` flag rather than being presented as validated science.
    """

    quantized = np.minimum((gray.astype(np.uint16) * levels // 256), levels - 1).astype(np.int32)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    for delta_y, delta_x in ((0, 1), (1, 0), (1, 1), (-1, 1)):
        y0_start, y0_end = max(0, -delta_y), min(gray.shape[0], gray.shape[0] - delta_y)
        x0_start, x0_end = max(0, -delta_x), min(gray.shape[1], gray.shape[1] - delta_x)
        y1_start, y1_end = max(0, delta_y), min(gray.shape[0], gray.shape[0] + delta_y)
        x1_start, x1_end = max(0, delta_x), min(gray.shape[1], gray.shape[1] + delta_x)
        valid = mask[y0_start:y0_end, x0_start:x0_end] & mask[y1_start:y1_end, x1_start:x1_end]
        if not valid.any():
            continue
        source = quantized[y0_start:y0_end, x0_start:x0_end][valid]
        target = quantized[y1_start:y1_end, x1_start:x1_end][valid]
        np.add.at(matrix, (source, target), 1)
        np.add.at(matrix, (target, source), 1)
    total = float(matrix.sum())
    if total <= 0:
        raise TambanInferenceError("Body crop has no valid texture pixel pairs.")
    matrix /= total
    values = np.arange(levels, dtype=np.float64)
    row, column = np.meshgrid(values, values, indexing="ij")
    difference_squared = (row - column) ** 2
    nonzero = matrix[matrix > 0]
    return (
        float(np.sum(difference_squared * matrix)),
        float(np.sum(matrix / (1.0 + difference_squared))),
        float(np.sqrt(np.sum(matrix**2))),
        float(-np.sum(nonzero * np.log2(nonzero))),
    )


def _uniform_lbp_16(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the 18-bin uniform LBP histogram recorded in the body table."""

    height, width = gray.shape
    radius = 2
    if height <= radius * 2 or width <= radius * 2:
        raise TambanInferenceError("Body crop is too small for the saved 16-neighbour LBP features.")
    y, x = np.mgrid[radius : height - radius, radius : width - radius]
    center = gray[y, x].astype(np.float64)
    valid = mask[y, x].copy()
    bits: list[np.ndarray] = []
    # Bilinear sampling matches a circular radius rather than collapsing the
    # 16 neighbours into a square stencil.
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        sample_y = y - radius * math.sin(angle)
        sample_x = x + radius * math.cos(angle)
        y_low, x_low = np.floor(sample_y).astype(int), np.floor(sample_x).astype(int)
        y_high, x_high = np.minimum(y_low + 1, height - 1), np.minimum(x_low + 1, width - 1)
        y_weight, x_weight = sample_y - y_low, sample_x - x_low
        sample = (
            (1.0 - y_weight) * (1.0 - x_weight) * gray[y_low, x_low]
            + (1.0 - y_weight) * x_weight * gray[y_low, x_high]
            + y_weight * (1.0 - x_weight) * gray[y_high, x_low]
            + y_weight * x_weight * gray[y_high, x_high]
        )
        bits.append(sample >= center)
    bit_stack = np.stack(bits, axis=-1)
    transitions = np.count_nonzero(bit_stack != np.roll(bit_stack, 1, axis=-1), axis=-1)
    ones = bit_stack.sum(axis=-1)
    bins = np.where(transitions <= 2, ones, 17).astype(np.int32)
    if not valid.any():
        raise TambanInferenceError("Body crop has no valid interior pixels for LBP features.")
    histogram = np.bincount(bins[valid], minlength=18).astype(np.float64)
    return histogram / histogram.sum()


def extract_features(frame_bgr: np.ndarray, bbox: Sequence[float]) -> tuple[dict[str, float], int, tuple[int, int, int, int]]:
    """Extract the exact saved feature schema from one detected Body ROI.

    YOLO is detection-only, so there is no trained instance mask to apply.  The
    saved run config supplies the foreground gray floor; it is used inside the
    clipped Body crop only.  No feature is calculated from the whole frame.
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise TambanInferenceError("Feature extraction requires an HxWx3 BGR frame.")
    height, width = frame_bgr.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3]))))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        raise TambanInferenceError(f"Body ROI is too small for feature extraction: {crop.shape[:2]}.")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = gray >= 40  # Saved ``foreground_gray_floor`` from run_configuration.json.
    if int(mask.sum()) < 32:
        raise TambanInferenceError("Body ROI has fewer than 32 foreground pixels after the saved gray-floor mask.")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    foreground = mask
    pixels = int(foreground.sum())
    hue, saturation, value = (hsv[..., index] for index in range(3))
    chromatic = foreground & (saturation >= 40)
    hue_angles = hue[chromatic].astype(np.float64) * (2.0 * math.pi / 180.0)
    hue_sin = float(np.sin(hue_angles).mean()) if hue_angles.size else 0.0
    hue_cos = float(np.cos(hue_angles).mean()) if hue_angles.size else 0.0
    yellow = foreground & (hue >= 15) & (hue <= 40) & (saturation >= 40)
    highlights = foreground & (value >= 220) & (saturation <= 80)
    lab_l = lab[..., 0].astype(np.float64) * (100.0 / 255.0)
    lab_a = lab[..., 1].astype(np.float64) - 128.0
    lab_b = lab[..., 2].astype(np.float64) - 128.0
    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_entropy = _masked_glcm(gray, foreground)
    lbp = _uniform_lbp_16(gray, foreground)
    lbp_nonzero = lbp[lbp > 0]
    gray_normalized = gray.astype(np.float64) / 255.0
    gradient_y, gradient_x = np.gradient(gray_normalized)
    edge_density = float(np.mean(np.hypot(gradient_x, gradient_y)[foreground] >= 0.12))

    features = {
        "value_mean": float(value[foreground].mean()),
        "value_std": float(value[foreground].std()),
        "value_p90": float(np.percentile(value[foreground], 90)),
        "value_p95": float(np.percentile(value[foreground], 95)),
        "saturation_mean": float(saturation[foreground].mean()),
        "saturation_std": float(saturation[foreground].std()),
        "gray_mean": float(gray[foreground].mean()),
        "gray_std": float(gray[foreground].std()),
        "hue_sin": hue_sin,
        "hue_cos": hue_cos,
        "chromatic_fraction": float(chromatic.sum() / pixels),
        "yellow_candidate_ratio": float(yellow.sum() / pixels),
        "lab_L_mean": float(lab_l[foreground].mean()),
        "lab_L_std": float(lab_l[foreground].std()),
        "lab_a_mean": float(lab_a[foreground].mean()),
        "lab_a_std": float(lab_a[foreground].std()),
        "lab_b_mean": float(lab_b[foreground].mean()),
        "lab_b_std": float(lab_b[foreground].std()),
        "highlight_candidate_ratio": float(highlights.sum() / pixels),
        "highlight_largest_ratio": _largest_component_ratio(highlights, pixels),
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_energy": glcm_energy,
        "glcm_entropy": glcm_entropy,
        "lbp_entropy": float(-np.sum(lbp_nonzero * np.log2(lbp_nonzero))),
        "edge_density_proxy": edge_density,
    }
    features.update({f"lbp_{index:02d}": float(lbp[index]) for index in range(18)})
    if tuple(features) != BODY_FEATURE_COLUMNS:
        raise TambanInferenceError(
            "The implemented feature extractor no longer matches the saved feature order.\n"
            f"Expected: {list(BODY_FEATURE_COLUMNS)}\nProduced: {list(features)}")
    return features, pixels, (x1, y1, x2, y2)


def _part_detections(result: Any) -> list[PartDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    parts: list[PartDetection] = []
    for box in boxes:
        class_id = int(box.cls.item())
        if not 0 <= class_id < len(EXPECTED_YOLO_CLASSES):
            raise TambanInferenceError(f"YOLO produced an invalid class index: {class_id}.")
        confidence = float(box.conf.item())
        bbox = tuple(float(value) for value in box.xyxy[0].tolist())
        parts.append(PartDetection.from_source_class(class_id, confidence, bbox))
    return parts


def _short_grade(value: str) -> str:
    return value.replace("Class ", "")


def process_frame(
    model: Any,
    frame_bgr: np.ndarray,
    confidence: float,
    image_size: int,
    classifier: BodyRegionClassifier | None,
) -> tuple[list[PartDetection], list[CandidateResult]]:
    """Run trained YOLO, then associate current-frame part evidence conservatively."""

    results = model.predict(source=frame_bgr, conf=confidence, imgsz=image_size, verbose=False)
    if not results:
        return [], []
    parts = _part_detections(results[0])
    # Existing association/fusion is reused.  With no whole-fish detector or
    # tracker, candidates are clearly current-frame, body-anchored evidence.
    pipeline = PartFusionPipeline()
    candidates = pipeline.process(parts, (frame_bgr.shape[1], frame_bgr.shape[0]))
    processed: list[CandidateResult] = []
    for candidate in candidates:
        item = CandidateResult(candidate=candidate)
        if classifier is not None and candidate.body is not None:
            try:
                item.features, _, _ = extract_features(frame_bgr, candidate.body.bbox)
                item.body_prediction = classifier.predict(item.features)
            except TambanInferenceError as error:
                item.body_error = str(error)
        processed.append(item)
    return parts, processed


PART_COLORS = {
    "Body": (255, 184, 60),
    "Head": (90, 210, 110),
    "Tail": (100, 165, 255),
}


def _candidate_grade(item: CandidateResult) -> tuple[str, float | None, bool]:
    quality = item.candidate.quality_result
    if quality is None:
        return "Uncertain", None, True
    return _short_grade(quality.winning_class), quality.winning_percentage / 100.0, quality.uncertain


def draw_detection(frame_bgr: np.ndarray, parts: Iterable[PartDetection], candidates: Iterable[CandidateResult]) -> np.ndarray:
    """Draw explicit part evidence and provisional current-frame candidates."""

    annotated = frame_bgr.copy()
    for part in parts:
        x1, y1, x2, y2 = (int(round(value)) for value in part.bbox)
        color = PART_COLORS[part.region]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{_short_grade(part.grade)} {part.region} {part.confidence * 100:.0f}%",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    counts: Counter[str] = Counter()
    for item in candidates:
        candidate = item.candidate
        if candidate.whole_fish_bbox is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in candidate.whole_fish_bbox)
        grade, probability, uncertain = _candidate_grade(item)
        counts[grade] += 1
        color = (0, 90, 255) if grade == "Rejected" else (44, 150, 70)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        confidence_text = f" {probability * 100:.1f}%" if probability is not None else ""
        prefix = "? " if uncertain else ""
        # Keep the on-frame label short enough to remain readable.  The CSV and
        # --debug output retain the separate body-ML grade and probability.
        label = f"Fish {candidate.candidate_id} | {prefix}{GRADE_NAMES.get(grade, grade)}{confidence_text}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        text_x = max(2, min(x1, annotated.shape[1] - text_width - 6))
        text_y = y1 - 7 if y1 >= text_height + 13 else min(annotated.shape[0] - 4, y2 + text_height + 7)
        cv2.rectangle(
            annotated,
            (text_x - 3, text_y - text_height - 3),
            (text_x + text_width + 3, text_y + 3),
            (25, 45, 65),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 250, 255),
            1,
            cv2.LINE_AA,
        )
    count_text = "  ".join(f"{grade}: {counts[grade]}" for grade in ("A", "B", "C", "Rejected", "Uncertain"))
    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 580), 37), (25, 45, 65), -1)
    cv2.putText(
        annotated,
        f"Current-frame provisional candidates (untracked) | {count_text}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (240, 245, 250),
        1,
        cv2.LINE_AA,
    )
    return annotated


def result_records(candidates: Iterable[CandidateResult], frame_number: int) -> list[dict[str, object]]:
    """Produce Excel-ready rows without claiming untracked candidates are unique fish."""

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for item in candidates:
        candidate = item.candidate
        quality = candidate.quality_result
        grade, confidence, uncertain = _candidate_grade(item)
        bbox = candidate.whole_fish_bbox or (None, None, None, None)
        row: dict[str, object] = {
            "timestamp_utc": timestamp,
            "frame": frame_number,
            "fish_id": candidate.candidate_id,
            "candidate_scope": "current_frame_untracked",
            "parts_observed": "/".join(candidate.parts_observed),
            "provisional_grade": grade,
            "provisional_grade_confidence": confidence,
            "provisional_grade_uncertain": uncertain,
            "association_confidence": candidate.association_confidence,
            "structural_status": quality.structural_status if quality else "unknown",
            "yolo_part_evidence": json.dumps(quality.part_evidence if quality else {}),
            "body_classifier_grade": item.body_prediction.grade if item.body_prediction else None,
            "body_classifier_confidence": item.body_prediction.confidence if item.body_prediction else None,
            "body_classifier_backend": item.body_prediction.backend if item.body_prediction else None,
            "body_classifier_error": item.body_error,
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
        }
        if item.features:
            row.update(item.features)
        rows.append(row)
    return rows


def save_results(rows: Sequence[Mapping[str, object]], destination: Path) -> None:
    """Write a directly usable CSV with only observed/implemented fields."""

    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    for row in rows[1:]:
        columns.extend(column for column in row if column not in columns)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def debug_candidate(item: CandidateResult, parts: Sequence[PartDetection]) -> None:
    """Print one evidence trail for diagnostic verification."""

    candidate = item.candidate
    print("\nDIAGNOSTIC: first current-frame candidate")
    print("YOLO detections:", len(parts))
    print("YOLO parts:", [(part.grade, part.region, round(part.confidence, 5), part.bbox) for part in candidate.parts])
    print("Candidate bbox:", candidate.whole_fish_bbox)
    print("Association confidence:", round(candidate.association_confidence, 5))
    if item.features:
        print("EXTRACTED BODY FEATURES")
        for name in BODY_FEATURE_COLUMNS:
            print(f"  {name} = {item.features[name]:.8f}")
    if item.body_prediction:
        print("JOBLIB BODY-REGION PREDICTION:", item.body_prediction.grade, item.body_prediction.confidence)
        print("JOBLIB PROBABILITIES:", item.body_prediction.probabilities)
    elif item.body_error:
        print("JOBLIB BODY-REGION ERROR:", item.body_error)
    else:
        print("JOBLIB BODY-REGION PREDICTION: no Body detection associated")
    grade, confidence, uncertain = _candidate_grade(item)
    print("PROVISIONAL PART-EVIDENCE RESULT:", grade, confidence, "uncertain=" + str(uncertain))


def _source_kind(source: str) -> tuple[str, Path | int]:
    if source.isdigit():
        return "camera", int(source)
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source is not a readable file or camera index: {source}")
    return ("image" if path.suffix.lower() in IMAGE_EXTENSIONS else "video"), path


def _run_image(
    model: Any,
    source: Path,
    output_dir: Path,
    confidence: float,
    image_size: int,
    classifier: BodyRegionClassifier | None,
    debug: bool,
    save_csv_file: bool,
) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise TambanInferenceError(f"OpenCV could not read image: {source}")
    parts, candidates = process_frame(model, frame, confidence, image_size, classifier)
    annotated = draw_detection(frame, parts, candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{source.stem}_annotated.jpg"
    if not cv2.imwrite(str(image_path), annotated):
        raise TambanInferenceError(f"Could not write annotated image: {image_path}")
    rows = result_records(candidates, 0)
    csv_path = output_dir / f"{source.stem}_results.csv"
    if save_csv_file:
        save_results(rows, csv_path)
    print(f"Detections: {len(parts)} parts; {len(candidates)} current-frame candidates")
    print(f"Annotated image: {image_path}")
    if save_csv_file:
        print(f"CSV: {csv_path}")
    if debug and candidates:
        debug_candidate(candidates[0], parts)


def _run_stream(
    model: Any,
    source: Path | int,
    source_kind: str,
    output_dir: Path,
    confidence: float,
    image_size: int,
    classifier: BodyRegionClassifier | None,
    debug: bool,
    save_csv_file: bool,
    show: bool,
) -> None:
    capture = cv2.VideoCapture(int(source) if source_kind == "camera" else str(source))
    if not capture.isOpened():
        raise TambanInferenceError(f"OpenCV could not open {source_kind}: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    output_path = output_dir / ("camera_annotated.mp4" if source_kind == "camera" else f"{Path(source).stem}_annotated.mp4")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise TambanInferenceError(f"Could not open output video: {output_path}")
    all_rows: list[dict[str, object]] = []
    frame_number = 0
    debug_done = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            parts, candidates = process_frame(model, frame, confidence, image_size, classifier)
            annotated = draw_detection(frame, parts, candidates)
            writer.write(annotated)
            all_rows.extend(result_records(candidates, frame_number))
            if debug and candidates and not debug_done:
                debug_candidate(candidates[0], parts)
                debug_done = True
            if show:
                cv2.imshow("Tamban trained-model inference (Esc to stop)", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_number += 1
    finally:
        capture.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    if save_csv_file:
        save_results(all_rows, output_dir / f"{output_path.stem}_results.csv")
    print(f"Processed {frame_number} frames. Current-frame candidates are not unique tracked fish.")
    print(f"Annotated video: {output_path}")


def _print_summary(yolo_path: Path, joblib_path: Path, classifier: BodyRegionClassifier) -> None:
    print("Tamban trained-model summary")
    print("YOLO model:", yolo_path)
    print("YOLO task: detect")
    print("YOLO classes:", ", ".join(EXPECTED_YOLO_CLASSES))
    print("Joblib model:", joblib_path)
    print("Joblib type: RandomForestClassifier (250 trees, max_depth=12, min_samples_leaf=5)")
    print("Joblib classes:", ", ".join(classifier.classes))
    print("Joblib feature count:", len(classifier.feature_columns))
    print("Joblib feature order:", ", ".join(classifier.feature_columns))
    print("Preprocessor/scaler:", "none saved; raw ordered feature table is expected")
    print("Joblib target:", classifier.target)
    print("Joblib validated:", classifier.validated)
    print("Evaluation evidence: training loss only; run_status.json reports validation/test NOT_AVAILABLE.")
    print("Note: last.pt is the only supplied checkpoint and is not a validated best model.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Tamban YOLO part detection with optional body-region Joblib evidence.")
    parser.add_argument("--source", default=SOURCE, help="Image path, video path, or webcam index (for example 0).")
    parser.add_argument("--yolo-model", type=Path, default=None, help="Trained YOLO .pt path; defaults to last.pt beside this script.")
    parser.add_argument("--joblib-model", type=Path, default=None, help="Body-region .joblib path; defaults to body_feature_classifier.joblib.")
    parser.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO confidence threshold in [0, 1].")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/tamban_inference/<timestamp>).")
    parser.add_argument("--no-joblib", action="store_true", help="Run trained YOLO part detection without the optional body-region classifier.")
    parser.add_argument("--portable-joblib", action="store_true", help="Use the saved-RandomForest reader directly when managed environments block SciPy/Scikit-learn DLLs.")
    parser.add_argument("--debug", action="store_true", default=DEBUG, help="Print feature and model evidence for the first candidate.")
    parser.add_argument("--no-csv", action="store_true", help="Do not export the per-candidate CSV.")
    parser.add_argument("--show", action="store_true", help="Show video/webcam frames; press Esc to stop.")
    parser.add_argument("--inspect", action="store_true", help="Print trained-model metadata without opening a source.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not 0.0 <= args.conf <= 1.0:
        raise TambanInferenceError("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        raise TambanInferenceError("--imgsz must be positive.")
    yolo_path, joblib_path = find_model_files(args.yolo_model, args.joblib_model)
    classifier = None if args.no_joblib else load_joblib_assets(joblib_path, prefer_portable=args.portable_joblib)
    if classifier is not None:
        _print_summary(yolo_path, joblib_path, classifier)
    else:
        print(f"YOLO model: {yolo_path}\nYOLO task: detect\nJoblib body-region classifier: disabled")
    if args.inspect:
        return 0
    if not args.source:
        raise TambanInferenceError("Provide --source <image|video|camera index>, or use --inspect.")
    model = load_yolo_model(yolo_path)
    output_dir = args.output_dir or (PROJECT_ROOT / "results" / "tamban_inference" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    kind, source = _source_kind(args.source)
    if kind == "image":
        _run_image(model, source, output_dir, args.conf, args.imgsz, classifier, args.debug, not args.no_csv)
    else:
        _run_stream(model, source, kind, output_dir, args.conf, args.imgsz, classifier, args.debug, not args.no_csv, args.show)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TambanInferenceError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
