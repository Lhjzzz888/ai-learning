"""标准库课程的行为测试；不需要安装第三方依赖。"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / '01-linear-regression' / 'house_price.py'
spec = importlib.util.spec_from_file_location('house_price', SCRIPT)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


class LinearRegressionTests(unittest.TestCase):
    def test_closed_form_recovers_line(self):
        self.assertEqual(lesson.fit([(1, 5), (2, 8), (3, 11)]), (3, 2))

    def test_invalid_training_data_has_clear_error(self):
        for samples in ([], [(1, 2)], [(1, 2), (1, 3)], [(1, 2), (float('nan'), 3)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                lesson.fit(samples)

    def test_baseline_uses_only_training_labels(self):
        result = lesson.evaluate_models([(1, 3), (2, 5), (3, 7)], [(4, 9), (5, 11)])
        self.assertEqual(result['baseline_prediction'], 5)
        self.assertEqual(result['baseline_mae'], 5)
        self.assertAlmostEqual(result['linear_mae'], 0)

    def test_empty_test_set_has_clear_error(self):
        with self.assertRaises(ValueError):
            lesson.evaluate_models([(1, 3), (2, 5)], [])

    def test_cli_rejects_nonfinite_area(self):
        for value in ('nan', 'inf', '-1'):
            with self.subTest(value=value):
                result = subprocess.run([sys.executable, str(SCRIPT), '--area', value], capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
