"""Train an interpretable handcrafted-feature baseline without touching test data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.dataset_utils import project_root
from src.training.feature_dataset import CLASS_NAMES, load_feature_table, validate_split_class_coverage


def _classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=list(CLASS_NAMES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES)).tolist(),
        "class_order": list(CLASS_NAMES),
    }


def _build_pipeline(model_name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC

    if model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", estimator)])
    if model_name == "svm":
        estimator = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("classifier", estimator)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an instance-level engineered-feature baseline.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--model", choices=("random_forest", "svm", "logistic_regression"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    try:
        import joblib
        import sklearn
    except ImportError:
        print("ERROR: scikit-learn and joblib are required. Install requirements.txt first.")
        return 1

    root = project_root()
    feature_path = args.features or root / "dataset" / "features" / "combined_features.csv"
    try:
        table = load_feature_table(feature_path)
        validate_split_class_coverage(table, ("train", "validation"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        print("Training was blocked; provide reviewed per-instance features and group-safe split metadata.")
        return 1

    train_idx = table.indices("train")
    validation_idx = table.indices("validation")
    pipeline = _build_pipeline(args.model, args.seed)
    pipeline.fit(table.values[train_idx], table.labels[train_idx])
    predictions = pipeline.predict(table.values[validation_idx])
    metrics = _classification_metrics(table.labels[validation_idx], predictions)

    run_name = args.run_name or f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = root / "models" / "feature_baselines" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, run_dir / "model.joblib")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "feature_table": str(feature_path.resolve()),
        "feature_table_sha256": table.source_sha256,
        "feature_names": table.feature_names,
        "train_instances": int(len(train_idx)),
        "validation_instances": int(len(validation_idx)),
        "train_groups": int(len(set(table.groups[train_idx]))),
        "validation_groups": int(len(set(table.groups[validation_idx]))),
        "sklearn_version": sklearn.__version__,
        "validation_metrics": metrics,
        "test_data_used": False,
    }
    (run_dir / "experiment_info.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    with (run_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "group_id", "actual", "predicted"])
        for index, predicted in zip(validation_idx, predictions, strict=True):
            writer.writerow([table.instance_ids[index], table.groups[index], table.labels[index], predicted])
    print(json.dumps(metrics, indent=2))
    print(f"Model and validation results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
