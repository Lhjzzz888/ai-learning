"""Train an offline digits MLP, or load its validation-best checkpoint."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from learning.vision import (configure_runtime, evaluate, load_checkpoint, load_digits_splits,
                             make_loader, make_model, train_model, verify_reload, write_history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tiny-overfit", type=int, default=0, metavar="N",
                        help="train only on the first N training samples; skip final test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/05-pytorch-mlp"))
    parser.add_argument("--checkpoint", type=Path, help="load without training; predict a validation sample")
    parser.add_argument("--predict-index", type=int, default=0, help="index inside the validation split")
    args = parser.parse_args()
    configure_runtime(args.seed, args.threads)
    if args.checkpoint:
        model, metadata = load_checkpoint(args.checkpoint)
        data = load_digits_splits(metadata["split_seed"])
        if not 0 <= args.predict_index < len(data["val_y"]):
            parser.error("--predict-index must be inside the validation split")
        index = args.predict_index
        with torch.no_grad():
            probabilities = model(data["val_x"][index:index + 1]).softmax(-1)[0]
        print(json.dumps({"source": "validation", "original_id": int(data["val_ids"][index]),
                          "true": int(data["val_y"][index]), "predicted": int(probabilities.argmax()),
                          "probabilities": probabilities.tolist(), "checkpoint": metadata}, indent=2))
        return
    if args.tiny_overfit < 0:
        parser.error("--tiny-overfit must be nonnegative")
    data = load_digits_splits(args.seed)
    if args.tiny_overfit > len(data["train_y"]):
        parser.error("--tiny-overfit exceeds the training split size")
    x, y = data["train_x"], data["train_y"]
    if args.tiny_overfit:
        x, y = x[:args.tiny_overfit], y[:args.tiny_overfit]
    config = {"kind": "mlp", "hidden_dim": args.hidden_dim}
    model = make_model(config)
    train = make_loader(x, y, args.batch_size, shuffle=True, seed=args.seed)
    val = make_loader(data["val_x"], data["val_y"], args.batch_size)
    checkpoint = args.output_dir / "best.pt"
    result = train_model(model, train, val, config=config, epochs=args.epochs, lr=args.lr,
                         checkpoint_path=checkpoint, split_seed=args.seed)
    last_train = evaluate(model, make_loader(x, y, args.batch_size))
    write_history(result["history"], args.output_dir)
    selected, metadata = load_checkpoint(checkpoint)
    delta = verify_reload(selected, checkpoint, config, data["val_x"][:16],
                          epoch=metadata["epoch"], val_loss=metadata["val_loss"], split_seed=args.seed)
    # Hand derivative: x=2, w=3, b=1, target=5 -> (wx+b-y)^2=4,
    # dL/dw=2*(7-5)*2=8, dL/db=2*(7-5)=4.
    w, b = torch.tensor(3.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)
    loss = (w * 2 + b - 5).square()
    loss.backward()
    summary = {"config": config, "seed": args.seed, "device": "cpu", "threads": args.threads,
               "torch_version": str(torch.__version__), "epochs": args.epochs, "lr": args.lr,
               "batch_size": args.batch_size, "train_samples": len(y),
               "val_samples": len(data["val_y"]), "test_samples": len(data["test_y"]),
               "parameter_count": sum(p.numel() for p in selected.parameters()),
               "best_epoch": result["best_epoch"], "seconds": result["seconds"],
               "last_epoch_train": last_train, "selected_validation": evaluate(selected, val),
               "reload_max_abs_difference": delta,
               "hand_gradient_check": {"loss": loss.item(), "dw": w.grad.item(), "db": b.grad.item()}}
    if args.tiny_overfit:
        summary["mode"] = "tiny training subset diagnostic; no final test"
    else:
        # Configuration and checkpoint are fixed before the test loader is used.
        summary["final_test"] = evaluate(selected, make_loader(data["test_x"], data["test_y"], args.batch_size))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
