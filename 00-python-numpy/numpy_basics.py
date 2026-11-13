"""List and NumPy implementations of prediction and mean absolute error."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def predict_python(X, weights, bias):
    if not X or not weights or any(len(row) != len(weights) for row in X):
        raise ValueError("X must be nonempty with one column per weight")
    return [sum(x * w for x, w in zip(row, weights)) + bias for row in X]


def mae_python(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal nonzero length")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def predict_numpy(X, weights, bias):
    X, weights = np.asarray(X, dtype=float), np.asarray(weights, dtype=float)
    if X.ndim != 2 or weights.ndim != 1 or X.shape[0] == 0 or X.shape[1] != len(weights):
        raise ValueError("Expected X (n,d) and weights (d,) with n > 0")
    return X @ weights + bias


def mae_numpy(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or actual.size == 0:
        raise ValueError("Expected equal nonempty vectors (n,); avoid broadcasting")
    return float(np.abs(actual - predicted).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/00-python-numpy")
    args = parser.parse_args()
    X = np.array([[1., 2.], [2., 1.], [3., 4.]])
    weights, bias = np.array([2., -1.]), 3.
    y = np.array([4., 5., 6.])
    predicted = predict_numpy(X, weights, bias)
    report = {"X_shape": list(X.shape), "y_shape": list(y.shape),
              "weights_shape": list(weights.shape), "prediction_shape": list(predicted.shape),
              "prediction": predicted.tolist(), "mae_numpy": mae_numpy(y, predicted),
              "mae_python": mae_python(y.tolist(), predict_python(X.tolist(), weights.tolist(), bias)),
              "bad_broadcast_shape": list((y[:, None] - predicted).shape),
              "squared_error_derivative": (2 * (predicted - y)).tolist()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
