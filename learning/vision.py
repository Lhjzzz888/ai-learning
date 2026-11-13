"""Shared, offline CPU training tools for lessons 05 and 06.

All classifiers accept (N, 1, 8, 8) float tensors and return raw (N, 10)
logits. No preprocessing statistics are learned from validation/test data.
"""

import csv
import json
import random
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def configure_runtime(seed=42, threads=1):
    """Keep these small examples on CPU, with one thread by default."""
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def load_digits_splits(seed=42):
    """Stratified 60/20/20 split; divide known integer pixels by 16.

    Original row IDs make the shared partition inspectable. Splitting happens
    before conversion, though this fixed physical scale fits no statistics.
    """
    digits = load_digits()
    ids = np.arange(len(digits.target))
    train_val, test = train_test_split(ids, test_size=0.2, random_state=seed,
                                     stratify=digits.target)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed,
                                 stratify=digits.target[train_val])
    result = {}
    for part, indices in (("train", train), ("val", val), ("test", test)):
        result[f"{part}_ids"] = indices
        result[f"{part}_x"] = torch.tensor(digits.images[indices, None] / 16, dtype=torch.float32)
        result[f"{part}_y"] = torch.tensor(digits.target[indices], dtype=torch.long)
    return result


def make_loader(x, y, batch_size=64, shuffle=False, seed=42):
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x and y must have the same nonzero sample count")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0, drop_last=False)


def make_model(config):
    kind = config.get("kind", "mlp")
    hidden = int(config.get("hidden_dim", 64 if kind == "mlp" else 32))
    if hidden < 1:
        raise ValueError("hidden_dim must be positive")
    if kind == "mlp":
        layers = [("flatten", nn.Flatten()), ("hidden", nn.Linear(64, hidden)),
                  ("relu", nn.ReLU()), ("logits", nn.Linear(hidden, 10))]
    elif kind == "cnn":
        channels = int(config.get("channels", 8))
        if channels < 1:
            raise ValueError("channels must be positive")
        layers = [("conv1", nn.Conv2d(1, channels, kernel_size=3, padding=1)),
                  ("relu1", nn.ReLU()), ("pool", nn.MaxPool2d(2)),
                  ("conv2", nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)),
                  ("relu2", nn.ReLU()), ("flatten", nn.Flatten()),
                  ("hidden", nn.Linear(channels * 2 * 4 * 4, hidden)),
                  ("relu3", nn.ReLU()), ("logits", nn.Linear(hidden, 10))]
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return nn.Sequential(OrderedDict(layers)).cpu()


def evaluate(model, loader):
    """eval() sets layer behavior; no_grad() separately disables autograd."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.numel()
    if not count:
        raise ValueError("cannot evaluate an empty loader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def train_epoch(model, loader, optimizer):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        # CrossEntropyLoss's default mean is over this batch, so weight by N.
        loss_sum += loss.item() * y.numel()
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
        count += y.numel()
    if not count:
        raise ValueError("cannot train on an empty loader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def save_checkpoint(path, model, config, *, epoch, val_loss, split_seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": dict(config), "state_dict": model.state_dict(),
                "epoch": int(epoch), "val_loss": float(val_loss), "split_seed": int(split_seed),
                "pixel_scale": 16}, path)


def load_checkpoint(path):
    # Only tensors and primitive metadata are stored; never unpickle model code.
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    if metadata.get("format_version") != 1 or metadata.get("pixel_scale") != 16:
        raise ValueError("unsupported checkpoint format or preprocessing")
    model = make_model(metadata["config"])
    model.load_state_dict(metadata["state_dict"])
    model.eval()
    return model, {key: value for key, value in metadata.items() if key != "state_dict"}


def train_model(model, train_loader, val_loader, *, config, epochs, checkpoint_path,
                lr=0.01, weight_decay=0.0, split_seed=42):
    """Select minimum validation cross entropy, including the epoch-0 baseline.

    The returned model remains at the last epoch. Call load_checkpoint for the
    selected model. These inference checkpoints do not support exact resumption.
    """
    if epochs < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("epochs/lr must be positive and weight_decay nonnegative")
    # Use a separate non-shuffled train evaluation loader for the epoch-0 curve;
    # consuming it must not advance the training shuffle generator.
    baseline_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                 shuffle=False, num_workers=0)
    initial_train, initial_val = evaluate(model, baseline_loader), evaluate(model, val_loader)
    history = [{"epoch": 0, "train_loss": initial_train["loss"],
                "train_accuracy": initial_train["accuracy"], "val_loss": initial_val["loss"],
                "val_accuracy": initial_val["accuracy"]}]
    best_loss, best_epoch = initial_val["loss"], 0
    save_checkpoint(checkpoint_path, model, config, epoch=0, val_loss=best_loss, split_seed=split_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]})
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch = val_metrics["loss"], epoch
            save_checkpoint(checkpoint_path, model, config, epoch=epoch,
                            val_loss=best_loss, split_seed=split_seed)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "seconds": perf_counter() - started}


def verify_reload(model, path, config, x, *, epoch, val_loss, split_seed=42):
    """Save and reload an inference checkpoint, checking actual predictions."""
    save_checkpoint(path, model, config, epoch=epoch, val_loss=val_loss, split_seed=split_seed)
    reloaded, _ = load_checkpoint(path)
    model.eval()
    with torch.no_grad():
        before, after = model(x), reloaded(x)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        return float((before - after).abs().max())


def write_history(history, output_dir):
    """Write editable CSV plus a dependency-free SVG loss curve."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    ceiling = max(row[key] for row in history for key in ("train_loss", "val_loss")) * 1.05
    max_epoch = max(row["epoch"] for row in history) or 1
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">',
             '<rect width="720" height="360" fill="white"/>',
             '<path d="M60 30V300H690" fill="none" stroke="#444"/>',
             '<text x="20" y="20" font-family="sans-serif">Cross entropy</text>',
             '<text x="620" y="340" font-family="sans-serif">Epoch</text>']
    for key, color, label, y in (("train_loss", "#2563eb", "Train (online batches)", 48),
                                 ("val_loss", "#dc2626", "Validation", 69)):
        points = " ".join(f'{60 + row["epoch"] / max_epoch * 630:.1f},{300 - row[key] / ceiling * 260:.1f}'
                          for row in history)
        parts.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2"/>')
        parts.append(f'<text x="390" y="{y}" fill="{color}" font-family="sans-serif">{label}</text>')
    for i in range(6):
        y = 300 - i / 5 * 260
        parts.append(f'<text x="10" y="{y:.1f}" font-family="sans-serif" font-size="12">{ceiling * i / 5:.2f}</text>')
        epoch = round(max_epoch * i / 5)
        parts.append(f'<text x="{60 + i / 5 * 630:.1f}" y="320" font-family="sans-serif" font-size="12">{epoch}</text>')
    parts.append("</svg>")
    (output / "curves.svg").write_text("\n".join(parts), encoding="utf-8")


def write_errors(model, x, y, ids, output_dir):
    """Save first five test errors, or every error when fewer than five exist."""
    model.eval()
    with torch.no_grad():
        probabilities = model(x).softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
    wrong = torch.where(predictions != y)[0].tolist()
    cases, svg = [], ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="210">',
                      '<rect width="800" height="210" fill="white"/>']
    for column, index in enumerate(wrong[:5]):
        case = {"original_id": int(ids[index]), "true": int(y[index]),
                "predicted": int(predictions[index]),
                "confidence": float(probabilities[index, predictions[index]]),
                "pixels_0_to_16": (x[index, 0] * 16).int().tolist()}
        cases.append(case)
        left = column * 160
        svg.append(f'<text x="{left + 8}" y="20" font-size="12" font-family="sans-serif">id={case["original_id"]}</text>')
        svg.append(f'<text x="{left + 8}" y="40" font-size="12" font-family="sans-serif">true={case["true"]}, pred={case["predicted"]}</text>')
        for row in range(8):
            for col in range(8):
                shade = round(float(x[index, 0, row, col]) * 255)
                svg.append(f'<rect x="{left + 8 + col * 16}" y="{55 + row * 16}" width="16" height="16" fill="rgb({shade},{shade},{shade})"/>')
    if not cases:
        svg.append('<text x="20" y="40" font-family="sans-serif">No errors in this test split.</text>')
    svg.append("</svg>")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"total_errors": len(wrong), "shown": len(cases), "selection": "first five in fixed test order, or all", "cases": cases}
    (output / "errors.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "errors.svg").write_text("\n".join(svg), encoding="utf-8")
    return report
