"""Compare gradient descent, closed form and cross-validated regularization."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mse(X, y, weights, bias):
    return float(np.mean((X @ weights + bias - y) ** 2))


def mse_gradient(X, y, weights, bias):
    residual = X @ weights + bias - y
    return 2 * X.T @ residual / len(y), float(2 * residual.mean())


def gradient_descent(X, y, learning_rate=.1, steps=300):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],) or len(y) == 0:
        raise ValueError("Expected nonempty X (n,d), y (n,)")
    if learning_rate <= 0 or not np.isfinite(learning_rate) or steps < 1:
        raise ValueError("learning_rate must be finite and positive; steps >= 1")
    weights, bias, losses = np.zeros(X.shape[1]), 0., []
    # Keep the initial loss, then one loss after each update for an honest curve.
    losses.append(mse(X, y, weights, bias))
    for _ in range(steps):
        dw, db = mse_gradient(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        loss = mse(X, y, weights, bias)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e100:
            break
    return weights, bias, losses


def univariate_experiment():
    # Import the original lesson instead of silently making a different fixture.
    spec = importlib.util.spec_from_file_location("house_price_source", ROOT / "01-linear-regression/house_price.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    samples = source.DATA.copy()
    random.Random(42).shuffle(samples)
    train, test = np.array(samples[:16], dtype=float), np.array(samples[16:], dtype=float)
    mean, scale = train[:, 0].mean(), train[:, 0].std()
    X, y = ((train[:, :1] - mean) / scale), train[:, 1]
    target_mean = y.mean()
    closed_slope, closed_bias = source.fit(samples[:16])
    curves, summaries = {}, {}
    for name, rate in (("slow", .001), ("normal", .1), ("diverging", 1.1)):
        weights, bias, losses = gradient_descent(X, y - target_mean, rate, 300)
        slope = float(weights[0] / scale)
        intercept = float(bias + target_mean - slope * mean)
        curves[name] = losses
        summaries[name] = {"learning_rate": rate, "initial_mse": losses[0], "final_mse": losses[-1],
                           "slope": slope, "intercept": intercept}
    closed_prediction = closed_slope * test[:, 0] + closed_bias
    normal = summaries["normal"]
    gd_prediction = normal["slope"] * test[:, 0] + normal["intercept"]
    report = {"train_size": 16, "test_size": 4, "closed_form": {"slope": closed_slope, "intercept": closed_bias},
              "gd": summaries, "test_closed_predictions": closed_prediction.tolist(),
              "test_gd_predictions": gd_prediction.tolist(),
              "max_prediction_difference": float(np.max(np.abs(closed_prediction - gd_prediction)))}
    return report, curves


def regularization_experiment():
    X, y = make_regression(n_samples=180, n_features=40, n_informative=5, noise=25., random_state=42)
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=.25, random_state=42)
    X_train, y_train, X_test, y_test = X[train_ids], y[train_ids], X[test_ids], y[test_ids]
    folds = KFold(n_splits=4, shuffle=True, random_state=42)
    searches, rows = {}, []
    for name, model, grid in (("linear", LinearRegression(), {}),
                              ("ridge", Ridge(), {"model__alpha": [.1, 1., 10., 100.]}),
                              ("lasso", Lasso(max_iter=10000), {"model__alpha": [.1, 1., 10.]})):
        search = GridSearchCV(Pipeline([("scale", StandardScaler()), ("model", model)]), grid,
                              scoring="neg_mean_squared_error", cv=folds, n_jobs=1)
        search.fit(X_train, y_train)
        searches[name] = search
        index = search.best_index_
        rows.append({"model": name, "cv_mse_mean": -float(search.best_score_),
                     "cv_mse_std": float(search.cv_results_["std_test_score"][index]),
                     "best_params": search.best_params_})
    winner = min(rows, key=lambda row: row["cv_mse_mean"])["model"]
    # Exactly one final test evaluation after selecting by CV; no test-guided tuning.
    test_mse = float(mean_squared_error(y_test, searches[winner].predict(X_test)))
    return {"seed": 42, "shape": list(X.shape), "informative_features": 5, "noise_std": 25.,
            "train_size": len(train_ids), "test_size": len(test_ids), "cv_folds": 4,
            "cv_results": rows, "winner": winner, "winner_test_mse": test_mse,
            "train_ids": train_ids.tolist(), "test_ids": test_ids.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/02-gradient-descent")
    args = parser.parse_args()
    univariate, curves = univariate_experiment()
    report = {"univariate": univariate, "regularization": regularization_experiment()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *curves])
        for step in range(max(map(len, curves.values()))):
            writer.writerow([step, *(losses[step] if step < len(losses) else "" for losses in curves.values())])
    print(f"GD vs closed max prediction difference: {univariate['max_prediction_difference']:.8f}")
    for row in report["regularization"]["cv_results"]:
        print(f"{row['model']}: CV MSE {row['cv_mse_mean']:.3f} +/- {row['cv_mse_std']:.3f}")
    print(f"Winner: {report['regularization']['winner']}; test MSE: {report['regularization']['winner_test_mse']:.3f}")
    print(f"Saved reports to {args.output_dir}")


if __name__ == "__main__":
    main()
