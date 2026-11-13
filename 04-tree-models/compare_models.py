"""Fixed-split model comparison with development CV and one final test."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "03-logistic-regression"))
from logistic_regression import classification_metrics, make_splits


def run_experiment():
    splits = make_splits()
    # Reuse lesson 03's exact held-out test. Train+validation form development CV data.
    X = np.concatenate([splits["train"][0], splits["validation"][0]])
    y = np.concatenate([splits["train"][1], splits["validation"][1]])
    selection_ids = np.concatenate([splits["train"][2], splits["validation"][2]])
    X_test, y_test, test_ids = splits["test"]
    models = [
        ("majority", DummyClassifier(strategy="most_frequent"), {}),
        ("logistic", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
         {"model__C": [.1, 1.]}),
        ("tree", DecisionTreeClassifier(random_state=42), {"max_depth": [3, 6]}),
        ("forest", RandomForestClassifier(n_estimators=60, min_samples_leaf=2, random_state=42, n_jobs=1),
         {"max_depth": [4, None]}),
        ("boosting", GradientBoostingClassifier(n_estimators=60, learning_rate=.1, random_state=42),
         {"max_depth": [1, 2]}),
    ]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows, searches = [], {}
    for name, model, parameters in models:
        search = GridSearchCV(model, parameters, scoring="f1", cv=folds, n_jobs=1)
        search.fit(X, y)
        searches[name] = search
        rows.append({"model": name, "cv_f1_mean": float(search.best_score_),
                     "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                     "best_params": search.best_params_, "candidate_count": len(search.cv_results_["params"])})
    winner = max(rows, key=lambda row: row["cv_f1_mean"])["model"]
    fitted = searches[winner].best_estimator_
    # Threshold .5 is fixed for every candidate; lesson 03 is the separate threshold experiment.
    predicted = fitted.predict(X_test)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    errors = [{"sample_id": int(test_ids[i]), "actual": int(y_test[i]), "predicted": int(predicted[i]),
               "probability_positive": float(probabilities[i]), "features": X_test[i].tolist(),
               "kind": "false_negative" if y_test[i] else "false_positive"}
              for i in np.flatnonzero(predicted != y_test)[:5]]
    importances = None
    if hasattr(fitted, "feature_importances_"):
        importances = [{"feature": int(i), "importance": float(fitted.feature_importances_[i])}
                       for i in np.argsort(fitted.feature_importances_)[::-1][:5]]
    return {"seed": 42, "selection_metric": "3-fold development F1", "threshold": .5,
            "cv_results": rows, "winner": winner, "test": classification_metrics(y_test, predicted),
            "development_size": len(y), "test_size": len(y_test), "selection_ids": selection_ids.tolist(),
            "test_ids": test_ids.tolist(), "errors": errors, "total_errors": int(np.sum(predicted != y_test)),
            "top_impurity_importances": importances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/04-tree-models")
    args = parser.parse_args()
    report = run_experiment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["cv_results"]:
        print(f"{row['model']}: CV F1 {row['cv_f1_mean']:.3f} +/- {row['cv_f1_std']:.3f}; {row['best_params']}")
    print(f"Winner: {report['winner']}; final test: {report['test']}")
    for error in report["errors"]:
        print(f"Error id={error['sample_id']} {error['kind']} p(positive)={error['probability_positive']:.3f}")
    print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
