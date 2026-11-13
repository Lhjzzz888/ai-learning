"""Compare predeclared MLP/CNN configurations on the same offline digits split."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from learning.vision import (configure_runtime, evaluate, load_checkpoint, load_digits_splits,
                             make_loader, make_model, train_model, verify_reload,
                             write_errors, write_history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="same epoch budget for both models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/06-cnn"))
    args = parser.parse_args()
    configure_runtime(args.seed, args.threads)
    data = load_digits_splits(args.seed)
    configs = [{"kind": "mlp", "hidden_dim": 64},
               {"kind": "cnn", "hidden_dim": 32, "channels": 8}]
    val = make_loader(data["val_x"], data["val_y"], args.batch_size)
    comparisons, selected_models = [], []
    for config in configs:
        # Reset both initialization and sample order for a reproducible comparison.
        configure_runtime(args.seed, args.threads)
        model = make_model(config)
        output = args.output_dir / config["kind"]
        checkpoint = output / "best.pt"
        train = make_loader(data["train_x"], data["train_y"], args.batch_size, shuffle=True, seed=args.seed)
        result = train_model(model, train, val, config=config, epochs=args.epochs,
                             lr=args.lr, weight_decay=args.weight_decay,
                             checkpoint_path=checkpoint, split_seed=args.seed)
        write_history(result["history"], output)
        selected, metadata = load_checkpoint(checkpoint)
        delta = verify_reload(selected, checkpoint, config, data["val_x"][:16],
                              epoch=metadata["epoch"], val_loss=metadata["val_loss"], split_seed=args.seed)
        trace, sample = [{"layer": "input", "shape": [1, 1, 8, 8]}], data["val_x"][:1]
        with torch.no_grad():
            for name, layer in selected.named_children():
                sample = layer(sample)
                trace.append({"layer": name, "shape": list(sample.shape)})
        comparisons.append({"config": config, "parameter_count": sum(p.numel() for p in selected.parameters()),
                            "best_epoch": result["best_epoch"], "seconds": result["seconds"],
                            "validation": evaluate(selected, val), "shape_trace": trace,
                            "reload_max_abs_difference": delta})
        selected_models.append((selected, output))
    # Pick by validation only. Freeze BOTH comparison configurations/checkpoints
    # before any final test evaluation. These test metrics never select a model.
    winner = min(comparisons, key=lambda item: item["validation"]["loss"])["config"]["kind"]
    for row, (model, output) in zip(comparisons, selected_models):
        row["final_test"] = evaluate(model, make_loader(data["test_x"], data["test_y"], args.batch_size))
        errors = write_errors(model, data["test_x"], data["test_y"], data["test_ids"], output)
        row["error_analysis"] = {"total_errors": errors["total_errors"], "shown": errors["shown"]}
    report = {"seed": args.seed, "device": "cpu", "threads": args.threads,
              "torch_version": str(torch.__version__), "epochs_each": args.epochs, "lr": args.lr,
              "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "split_sizes": {part: len(data[f"{part}_y"]) for part in ("train", "val", "test")},
              "selected_by_validation_loss": winner, "models": comparisons}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
