"""Locked-test evaluation with specimen-group bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.preprocessing.dataset_utils import project_root
from src.training.feature_dataset import CLASS_NAMES, load_feature_table, validate_split_class_coverage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_values(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

    rejected_recall = float(recall_score(y_true, y_pred, labels=["Rejected"], average=None, zero_division=0)[0])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0)),
        "rejected_recall": rejected_recall,
        "rejected_false_accept_rate": 1.0 - rejected_recall,
    }


def _group_bootstrap(y_true, y_pred, groups, seed: int, repetitions: int) -> dict[str, dict[str, float]]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive.")
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    if unique_groups.size == 0:
        raise ValueError("at least one test specimen/group is required for bootstrap intervals.")
    samples: dict[str, list[float]] = {}
    for _ in range(repetitions):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected_indices = np.concatenate([np.flatnonzero(groups == group) for group in selected_groups])
        values = _metric_values(y_true[selected_indices], y_pred[selected_indices])
        for name, value in values.items():
            samples.setdefault(name, []).append(value)
    return {
        name: {"lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}
        for name, values in samples.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a feature model once on the locked test split.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("ERROR: scikit-learn and joblib are required. Install requirements.txt first.")
        return 1

    root = project_root()
    model_path = args.run_dir / "model.joblib"
    metadata_path = args.run_dir / "experiment_info.json"
    if not model_path.is_file() or not metadata_path.is_file():
        print("ERROR: Run directory must contain model.joblib and experiment_info.json.")
        return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_path = args.features or Path(metadata["feature_table"])
    try:
        table = load_feature_table(feature_path)
        validate_split_class_coverage(table, ("test",))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if _sha256(feature_path) != metadata.get("feature_table_sha256"):
        print("ERROR: Feature table changed after training; evaluation provenance is no longer identical.")
        return 1
    if table.feature_names != metadata.get("feature_names"):
        print("ERROR: Feature columns differ from those used during training.")
        return 1

    test_idx = table.indices("test")
    model = joblib.load(model_path)
    predictions = model.predict(table.values[test_idx])
    true_labels = table.labels[test_idx]
    metrics = _metric_values(true_labels, predictions)
    payload = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "feature_table": str(feature_path.resolve()),
        "feature_table_sha256": table.source_sha256,
        "test_instances": int(len(test_idx)),
        "test_groups": int(len(set(table.groups[test_idx]))),
        "metrics": metrics,
        "confidence_intervals": _group_bootstrap(true_labels, predictions, table.groups[test_idx], args.seed, args.bootstrap_repetitions),
        "classification_report": classification_report(true_labels, predictions, labels=list(CLASS_NAMES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=list(CLASS_NAMES)).tolist(),
        "class_order": list(CLASS_NAMES),
        "bootstrap_unit": "specimen/group",
        "warning": "Do not use these final-test results to tune any model, threshold, preprocessing, or augmentation choice.",
    }
    output = args.output or args.run_dir / "locked_test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (args.run_dir / "locked_test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "group_id", "actual", "predicted"])
        for index, predicted in zip(test_idx, predictions, strict=True):
            writer.writerow([table.instance_ids[index], table.groups[index], table.labels[index], predicted])
    print(json.dumps(payload, indent=2))
    print(f"Locked-test report saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
