"""Behavior checks: shape safety, analytic gradient, optimization and data boundaries."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_lesson(folder, filename):
    path = ROOT / folder / filename
    if not path.exists():
        raise AssertionError(f"Missing runnable lesson: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class NumpyTests(unittest.TestCase):
    def test_python_numpy_agree_and_column_target_is_rejected(self):
        lesson = load_lesson("00-python-numpy", "numpy_basics.py")
        X = np.array([[1., 2.], [3., 4.], [5., 6.]])
        weights = np.array([2., -1.])
        np.testing.assert_array_equal(lesson.predict_numpy(X, weights, 3), [3., 5., 7.])
        self.assertEqual(lesson.predict_python(X.tolist(), weights.tolist(), 3), [3., 5., 7.])
        self.assertEqual(lesson.mae_numpy([2., 6., 7.], [3., 5., 7.]), 2 / 3)
        self.assertEqual(lesson.mae_python([2., 6., 7.], [3., 5., 7.]), 2 / 3)
        with self.assertRaises(ValueError):
            lesson.mae_numpy(np.zeros((3, 1)), np.ones(3))
        with self.assertRaises(ValueError):
            lesson.predict_numpy(X, np.ones(3), 0)


class GradientTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        lesson = load_lesson("02-gradient-descent", "gradient_descent.py")
        X = np.array([[1., 2.], [-1., 3.], [2., 0.]])
        y = np.array([1., 4., -2.])
        weights, bias, eps = np.array([.2, -.4]), .3, 1e-6
        dw, db = lesson.mse_gradient(X, y, weights, bias)
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = eps
            numeric = (lesson.mse(X, y, weights + delta, bias) -
                       lesson.mse(X, y, weights - delta, bias)) / (2 * eps)
            self.assertAlmostEqual(dw[j], numeric, places=6)
        numeric_bias = (lesson.mse(X, y, weights, bias + eps) -
                        lesson.mse(X, y, weights, bias - eps)) / (2 * eps)
        self.assertAlmostEqual(db, numeric_bias, places=6)

    def test_gradient_descent_converges_and_detects_divergence(self):
        lesson = load_lesson("02-gradient-descent", "gradient_descent.py")
        X = np.array([[-1.], [0.], [1.]])
        y = np.array([1., 3., 5.])
        weights, bias, losses = lesson.gradient_descent(X, y, learning_rate=.1, steps=300)
        np.testing.assert_allclose(X @ weights + bias, y, atol=1e-7)
        self.assertLess(losses[-1], losses[0] * 1e-10)
        _, _, unstable = lesson.gradient_descent(X, y, learning_rate=2., steps=50)
        self.assertGreater(unstable[-1], unstable[0])


class ClassificationTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_reproducible(self):
        lesson = load_lesson("03-logistic-regression", "logistic_regression.py")
        a, b = lesson.make_splits(), lesson.make_splits()
        ids = [set(a[name][2].tolist()) for name in ("train", "validation", "test")]
        self.assertEqual(len(set.union(*ids)), 1200)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(a[name][2], b[name][2])
            self.assertLess(a[name][1].mean(), .25)
            self.assertGreater(a[name][1].mean(), .1)

    def test_threshold_uses_validation_f1_and_metrics_handle_no_positives(self):
        lesson = load_lesson("03-logistic-regression", "logistic_regression.py")
        y, probabilities = np.array([0, 0, 1, 1]), np.array([.1, .4, .45, .8])
        threshold, rows = lesson.choose_threshold(y, probabilities, [.3, .5, .7])
        self.assertEqual(threshold, .3)  # F1 .8, versus 2/3 at .5 and .7.
        self.assertEqual(len(rows), 3)
        metrics = lesson.classification_metrics(y, np.zeros(4, dtype=int))
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [2, 0]])
        self.assertEqual(metrics["precision"], 0.)
        self.assertEqual(metrics["recall"], 0.)

    def test_full_experiment_scaler_uses_training_only(self):
        lesson = load_lesson("03-logistic-regression", "logistic_regression.py")
        splits = lesson.make_splits()
        model, report = lesson.run_experiment(splits)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, splits["train"][0].mean(axis=0))
        self.assertGreater(report["test"]["logistic"]["f1"], report["test"]["majority"]["f1"])


class TreeTests(unittest.TestCase):
    def test_cv_selection_excludes_test_and_error_examples_are_errors(self):
        lesson = load_lesson("04-tree-models", "compare_models.py")
        report = lesson.run_experiment()
        self.assertEqual(len(report["cv_results"]), 5)
        self.assertFalse(set(report["selection_ids"]) & set(report["test_ids"]))
        self.assertGreater(report["test"]["f1"], 0.)
        for error in report["errors"]:
            self.assertNotEqual(error["actual"], error["predicted"])
            self.assertIn(error["sample_id"], report["test_ids"])


if __name__ == "__main__":
    unittest.main()
