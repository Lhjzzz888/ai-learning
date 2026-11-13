"""Behavioral checks for the offline digits training loop (CPU)."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from learning import vision


class VisionTests(unittest.TestCase):
    def setUp(self):
        vision.configure_runtime(seed=7, threads=1)

    def test_split_is_disjoint_reproducible_and_uses_fixed_pixel_scale(self):
        a = vision.load_digits_splits(seed=7)
        b = vision.load_digits_splits(seed=7)
        sets = [set(a[f"{part}_ids"].tolist()) for part in ("train", "val", "test")]
        self.assertEqual(len(set.union(*sets)), 1797)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        np.testing.assert_array_equal(a["train_ids"], b["train_ids"])
        self.assertEqual(tuple(a["train_x"].shape[1:]), (1, 8, 8))
        self.assertEqual(a["train_x"].dtype, torch.float32)
        # load_digits pixels are known integers 0..16: no split-wide fitted scaler.
        scaled = a["train_x"] * 16
        self.assertTrue(torch.equal(scaled, scaled.round()))
        self.assertGreaterEqual(float(a["train_x"].min()), 0)
        self.assertLessEqual(float(a["train_x"].max()), 1)

    def test_evaluation_weights_short_last_batch_by_samples(self):
        logits = torch.tensor([[math.log(3), 0], [0, math.log(3)], [0, math.log(3)]])
        labels = torch.tensor([0, 1, 0])
        model = nn.Identity()
        model.train()
        metrics = vision.evaluate(model, vision.make_loader(logits, labels, batch_size=2))
        expected = (-2 * math.log(0.75) - math.log(0.25)) / 3
        self.assertAlmostEqual(metrics["loss"], expected, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["samples"], 3)
        self.assertFalse(model.training)

    def test_models_keep_batch_dimension_and_emit_ten_logits(self):
        for kind in ("mlp", "cnn"):
            model = vision.make_model({"kind": kind, "hidden_dim": 32, "channels": 8})
            for size in (1, 5):
                logits = model(torch.zeros(size, 1, 8, 8))
                self.assertEqual(tuple(logits.shape), (size, 10))
                # A softmax output would sum to one; these are raw scores.
                self.assertFalse(torch.allclose(logits.sum(-1), torch.ones(size)))

    def test_training_metrics_weight_short_last_batch_by_samples(self):
        logits = torch.tensor([[math.log(3), 0], [0, math.log(3)], [0, math.log(3)]])
        model = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(logits.T)
        loader = vision.make_loader(torch.eye(3), torch.tensor([0, 1, 0]), batch_size=2)
        metrics = vision.train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0))
        self.assertAlmostEqual(metrics["loss"], (-2 * math.log(0.75) - math.log(0.25)) / 3, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_training_can_overfit_a_tiny_subset_and_reload_best_checkpoint(self):
        data = vision.load_digits_splits(seed=7)
        x, y = data["train_x"][:32], data["train_y"][:32]
        config = {"kind": "mlp", "hidden_dim": 64}
        model = vision.make_model(config)
        loader = vision.make_loader(x, y, batch_size=32, shuffle=True, seed=7)
        validation = vision.make_loader(data["val_x"], data["val_y"], batch_size=91)
        initial = vision.evaluate(model, vision.make_loader(x, y))["loss"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "best.pt"
            result = vision.train_model(model, loader, validation, config=config,
                                        epochs=80, lr=0.02, checkpoint_path=checkpoint,
                                        split_seed=7)
            final = vision.evaluate(model, vision.make_loader(x, y))
            self.assertLess(final["loss"], initial / 10)
            self.assertGreater(final["accuracy"], 0.95)
            restored, metadata = vision.load_checkpoint(checkpoint)
            self.assertEqual(metadata["config"], config)
            self.assertEqual(metadata["epoch"], result["best_epoch"])
            self.assertAlmostEqual(metadata["val_loss"], min(row["val_loss"] for row in result["history"]))
            self.assertAlmostEqual(vision.evaluate(restored, validation)["loss"], metadata["val_loss"], places=6)
            # Check a direct save of the final model as well as the selected model.
            final_path = Path(directory) / "final.pt"
            vision.save_checkpoint(final_path, model, config, epoch=80, val_loss=1.0, split_seed=7)
            reloaded, _ = vision.load_checkpoint(final_path)
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), reloaded(x), rtol=0, atol=0)
                self.assertEqual(tuple(restored(x).shape), (32, 10))

    def test_cnn_learns_with_shared_training_loop(self):
        data = vision.load_digits_splits(seed=7)
        config = {"kind": "cnn", "hidden_dim": 32, "channels": 8}
        model = vision.make_model(config)
        train = vision.make_loader(data["train_x"], data["train_y"], batch_size=128,
                                   shuffle=True, seed=7)
        val = vision.make_loader(data["val_x"], data["val_y"], batch_size=128)
        initial = vision.evaluate(model, val)["loss"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            result = vision.train_model(model, train, val, config=config, epochs=5,
                                        lr=0.01, checkpoint_path=Path(directory) / "best.pt")
        self.assertLess(result["best_val_loss"], initial * 0.75)

    def test_invalid_training_budget_is_rejected(self):
        model = vision.make_model({"kind": "mlp"})
        loader = vision.make_loader(torch.zeros(2, 1, 8, 8), torch.zeros(2, dtype=torch.long))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with self.assertRaises(ValueError):
                vision.train_model(model, loader, loader, config={"kind": "mlp"},
                                   epochs=0, checkpoint_path=Path(directory) / "x.pt")


if __name__ == "__main__":
    unittest.main()
