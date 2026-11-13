"""Language contracts: boundary leaks, future leaks, wrong labels, lost weights."""
import math
from pathlib import Path
import tempfile
import unittest

import torch
from learning import language as lm


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_vocabulary_and_document_boundaries(self):
        data = lm.prepare_data()
        ids = [set(data["metadata"]["document_ids"][s]) for s in ("train", "val", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        groups = [set(data["metadata"]["source_groups"][s]) for s in ("train", "val", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        train_chars = set("".join(d["text"] for d in data["documents"]["train"]))
        self.assertEqual(set(data["vocab"][1:]), train_chars)
        self.assertEqual(lm.encode("甲火星", ["<unk>", "甲"]), [1, 0, 0])
        self.assertAlmostEqual(lm.unknown_rate("甲火星", ["<unk>", "甲"]), 2 / 3)

    def test_label_shift_and_no_window_crosses_documents(self):
        windows = lm.training_windows([[1, 2, 3, 4], [8, 9, 8, 9]], 2)
        self.assertEqual([(x.tolist(), y.tolist()) for x, y in windows], [
            ([1, 2], [2, 3]), ([2, 3], [3, 4]),
            ([8, 9], [9, 8]), ([9, 8], [8, 9])])
        with self.assertRaises(ValueError):
            lm.training_windows([[1]], 2)

    def test_full_evaluation_counts_each_target_once(self):
        model = lm.BigramLM(vocab_size=3, block_size=3)
        with torch.no_grad():
            model.table.weight.zero_()
        score = lm.evaluate(model, [[0, 1, 2, 1, 0], [2, 1, 0]])
        self.assertEqual(score["tokens"], 6)
        self.assertAlmostEqual(score["loss"], math.log(3), places=6)
        self.assertAlmostEqual(score["perplexity"], 3.0, places=5)
        # 不同块的损失不同且块长为 3/1/2，能抓住错误的“块平均再平均”。
        with torch.no_grad():
            model.table.weight[0] = torch.tensor([0.0, math.log(2), 0.0])
            model.table.weight[1] = torch.tensor([math.log(3), 0.0, 0.0])
        weighted = lm.evaluate(model, [[0, 1, 2, 1, 0], [2, 1, 0]])
        expected = (math.log(2) + math.log(5) + 2 * math.log(3) + 2 * math.log(5 / 3)) / 6
        self.assertAlmostEqual(weighted["loss"], expected, places=6)

    def test_transformer_cannot_read_future_tokens(self):
        torch.manual_seed(9)
        model = lm.TinyTransformer(vocab_size=8, block_size=8, d_model=16, heads=4, layers=2).eval()
        x = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 7, 6]])
        with torch.no_grad():
            a, b = model(x), model(changed)
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(a[:, 3:], b[:, 3:]))
        weights = model.blocks[0].attention.attention_weights(model.embedding(x) + model.position(torch.arange(5)))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)

    def test_models_overfit_a_tiny_predictable_sequence(self):
        for kind in ("bigram", "transformer"):
            with self.subTest(kind=kind):
                torch.manual_seed(12)
                config = {"vocab_size": 3, "block_size": 4}
                if kind == "transformer":
                    config.update(d_model=16, heads=2, layers=1)
                model = lm.make_model(kind, config)
                x, y = torch.tensor([[1, 2, 1, 2]]), torch.tensor([[2, 1, 2, 1]])
                initial = torch.nn.functional.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1)).item()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.1 if kind == "bigram" else 0.03, weight_decay=0)
                for _ in range(70):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
                    loss.backward()
                    optimizer.step()
                self.assertLess(loss.item(), 0.08)
                self.assertLess(loss.item(), initial * 0.1)

    def test_checkpoint_reload_and_seeded_generation(self):
        config = {"vocab_size": 3, "block_size": 4, "d_model": 16, "heads": 2, "layers": 1}
        model = lm.make_model("transformer", config).eval()
        x = torch.tensor([[1, 2, 1]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            checkpoint = Path(directory) / "best.pt"
            lm.save_checkpoint(checkpoint, model, "transformer", config, ["<unk>", "甲", "乙"], {"document_ids": {}}, 1, 2.0)
            loaded, payload = lm.load_checkpoint(checkpoint)
            torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)
            a = lm.generate(loaded, payload["vocab"], "甲陌", seed=4, max_new_tokens=8)
            b = lm.generate(loaded, payload["vocab"], "甲陌", seed=4, max_new_tokens=8)
            self.assertEqual(a, b)
            self.assertEqual(a["unknown_prompt_characters"], 1)
            self.assertEqual(a["generated_tokens"], 8)
            self.assertTrue(a["text"].startswith("甲陌"))
            with self.assertRaises(ValueError):
                lm.generate(loaded, payload["vocab"], "", seed=4, max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
