"""Imbalanced binary classification with a validation-selected F1 threshold."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_splits(seed=42):
    X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, n_redundant=2,
                               weights=[.85, .15], flip_y=.02, class_sep=1.1, random_state=seed)
    development, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    train, validation = train_test_split(development, test_size=.25, stratify=y[development], random_state=seed)
    return {name: (X[ids], y[ids], ids) for name, ids in
            (("train", train), ("validation", validation), ("test", test))}


def classification_metrics(actual, predicted):
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist()}


def choose_threshold(validation_y, validation_probabilities, thresholds=(.2, .35, .5, .65)):
    rows = [{"threshold": float(threshold), **classification_metrics(
        validation_y, np.asarray(validation_probabilities) >= threshold)} for threshold in thresholds]
    if not rows:
        raise ValueError("Need at least one candidate threshold")
    # F1 is declared before evaluation; ties prefer proximity to .5, then smaller threshold.
    best = max(rows, key=lambda row: (row["f1"], -abs(row["threshold"] - .5), -row["threshold"]))
    return best["threshold"], rows


def run_experiment(splits=None):
    splits = make_splits() if splits is None else splits
    X_train, y_train, _ = splits["train"]
    X_validation, y_validation, _ = splits["validation"]
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    model.fit(X_train, y_train)
    threshold, validation_rows = choose_threshold(y_validation, model.predict_proba(X_validation)[:, 1])
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    X_test, y_test, _ = splits["test"]
    report = {"seed": 42, "selection_metric": "validation F1", "threshold": threshold,
              "validation_thresholds": validation_rows,
              "split_sizes": {name: len(part[1]) for name, part in splits.items()},
              "positive_fraction": {name: float(part[1].mean()) for name, part in splits.items()},
              "split_ids": {name: part[2].tolist() for name, part in splits.items()},
              "test": {"majority": classification_metrics(y_test, baseline.predict(X_test)),
                       "logistic": classification_metrics(y_test, model.predict_proba(X_test)[:, 1] >= threshold)}}
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/03-logistic-regression")
    args = parser.parse_args()
    _, report = run_experiment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["validation_thresholds"]:
        print(f"Validation threshold={row['threshold']:.2f}: precision={row['precision']:.3f} recall={row['recall']:.3f} F1={row['f1']:.3f}")
    print(f"Selected threshold: {report['threshold']}")
    for name, metrics in report["test"].items():
        print(f"Test {name}: accuracy={metrics['accuracy']:.3f} F1={metrics['f1']:.3f} confusion={metrics['confusion_matrix']}")
    print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
