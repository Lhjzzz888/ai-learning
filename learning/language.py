"""字符级语言模型：先按来源组划分，再建词表和窗口；仅用 CPU。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F


CORPUS = Path(__file__).resolve().parents[1] / "data" / "text" / "corpus.json"
SPLIT_SEED = 17


def encode(text, vocab):
    lookup = {character: index for index, character in enumerate(vocab)}
    return [lookup.get(character, 0) for character in text]


def unknown_rate(text, vocab):
    return sum(token == 0 for token in encode(text, vocab)) / len(text) if text else 0.0


def prepare_data(path=CORPUS, split_seed=SPLIT_SEED):
    raw = Path(path).read_bytes()
    documents = json.loads(raw.decode("utf-8"))
    seen_ids, seen_texts = set(), set()
    for document in documents:
        document["text"] = unicodedata.normalize("NFC", document["text"]).strip()
        if document["id"] in seen_ids or document["text"] in seen_texts:
            raise ValueError("语料含重复 ID 或完全重复正文；请去重并检查来源组")
        if len(document["text"]) < 2 or not document["source_group"]:
            raise ValueError("每篇正文至少两个字符，且必须标注 source_group")
        seen_ids.add(document["id"])
        seen_texts.add(document["text"])
    groups = sorted({document["source_group"] for document in documents})
    if len(groups) < 3:
        raise ValueError("至少需要三个独立来源组来划分训练、验证和测试")
    random.Random(split_seed).shuffle(groups)
    heldout = max(1, len(groups) // 6)
    group_splits = {"train": groups[2 * heldout:], "val": groups[:heldout], "test": groups[heldout:2 * heldout]}
    splits = {name: [d for d in documents if d["source_group"] in selected] for name, selected in group_splits.items()}
    vocab = ["<unk>"] + sorted(set("".join(d["text"] for d in splits["train"])))
    encoded = {name: [encode(d["text"], vocab) for d in docs] for name, docs in splits.items()}
    counts = {}
    for name, docs in encoded.items():
        characters = sum(len(d) for d in docs)
        unknown = sum(d.count(0) for d in docs)
        counts[name] = {"characters": characters, "scored_tokens": sum(len(d) - 1 for d in docs),
                        "unknown_characters": unknown, "unknown_rate": unknown / characters}
    metadata = {"corpus_sha256": hashlib.sha256(raw).hexdigest(), "split_seed": split_seed,
                "document_ids": {s: [d["id"] for d in docs] for s, docs in splits.items()},
                "source_groups": group_splits, "counts": counts,
                "normalization": "Unicode NFC, strip ends, preserve internal whitespace and punctuation",
                "evaluation": "per-document consecutive blocks; every target after first character counted once",
                "data_license": "CC0-1.0; original repository-authored corpus only"}
    return {"documents": splits, "encoded": encoded, "vocab": vocab, "metadata": metadata}


def training_windows(documents, block_size):
    if block_size < 1:
        raise ValueError("block_size 必须为正整数")
    windows = []
    for document in documents:
        # 一篇文档内允许重叠训练，但从不跨文档，也不跨集合。
        for start in range(len(document) - block_size):
            tokens = torch.tensor(document[start:start + block_size + 1], dtype=torch.long)
            windows.append((tokens[:-1], tokens[1:]))
    if not windows:
        raise ValueError("没有足够长的训练文档；请减小 block_size")
    return windows


class BigramLM(nn.Module):
    """每个输入字符查一行可训练 logits，输出下一字符的分布。"""
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens):
        return self.table(tokens)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads, block_size):
        super().__init__()
        if heads < 1 or d_model < 1 or d_model % heads:
            raise ValueError("d_model 必须为正数且能被 heads 整除")
        self.heads, self.head_dim = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)), persistent=False)

    def _qkv(self, x):
        batch, length, width = x.shape
        # [B,T,3D] -> 三个 [B,H,T,D/H]；没有调用现成 Transformer/Attention 层。
        return self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

    def _weights(self, q, k):
        length = q.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        return F.softmax(scores, dim=-1)

    def attention_weights(self, x):
        q, k, _ = self._qkv(x)
        return self._weights(q, k)

    def forward(self, x):
        q, k, v = self._qkv(x)
        attended = self._weights(q, k) @ v
        batch, _, length, _ = attended.shape
        return self.output(attended.transpose(1, 2).contiguous().reshape(batch, length, -1))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, block_size):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads, block_size)
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=32, heads=4, layers=2):
        super().__init__()
        if block_size < 1 or layers < 1:
            raise ValueError("block_size 和 layers 必须为正整数")
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, heads, block_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        length = tokens.shape[1]
        if not 1 <= length <= self.block_size:
            raise ValueError("输入长度必须在 1 与 block_size 之间")
        x = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def make_model(kind, config):
    if kind == "bigram":
        return BigramLM(**config)
    if kind == "transformer":
        return TinyTransformer(**config)
    raise ValueError(f"未知模型类型：{kind}")


@torch.no_grad()
def evaluate(model, documents):
    """完整、确定性计分，不随机抽验证窗口；每个 target 只计一次。"""
    model.eval()
    total_loss, count = 0.0, 0
    for document in documents:
        # 输入块 x[0:T] 对应 y[1:T+1]；最后短块也参与，按 token 数加权。
        for start in range(0, len(document) - 1, model.block_size):
            chunk = document[start:start + model.block_size + 1]
            x = torch.tensor([chunk[:-1]], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            logits = model(x)
            total_loss += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y, reduction="sum").item()
            count += len(chunk) - 1
    if count == 0:
        raise ValueError("评估至少需要一个有效 target")
    loss = total_loss / count
    return {"loss": loss, "perplexity": math.exp(min(loss, 700)), "tokens": count}


def save_checkpoint(path, model, kind, config, vocab, metadata, step, val_loss):
    torch.save({"format_version": 1, "model_kind": kind, "config": config,
                "state_dict": model.state_dict(), "vocab": vocab, "data": metadata,
                "step": step, "validation_loss": val_loss}, path)


def load_checkpoint(path):
    # 只保存张量、数字、字符串和容器，不依赖任意 Python 类反序列化。
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("不支持的检查点版本")
    model = make_model(payload["model_kind"], payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def generate(model, vocab, prompt, seed=17, max_new_tokens=60, temperature=1.0):
    if not prompt:
        raise ValueError("prompt 不能为空；本例没有 BOS token")
    if max_new_tokens < 0 or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("max_new_tokens 不能为负，temperature 必须是有限正数")
    model.eval()
    tokens = encode(prompt, vocab)
    unknown = tokens.count(0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    added = []
    for _ in range(max_new_tokens):
        context = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
        logits = model(context)[0, -1] / temperature
        token = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).item())
        tokens.append(token)
        added.append(vocab[token])
    return {"text": prompt + "".join(added), "generated_tokens": len(added),
            "unknown_prompt_characters": unknown,
            "prompt_context_characters": min(len(prompt), model.block_size), "seed": seed,
            "temperature": temperature}


def training_main(kind):
    parser = argparse.ArgumentParser(description="CPU 单线程字符级语言模型，从随机权重训练")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / ("07" if kind == "bigram" else "08"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.1 if kind == "bigram" else 0.003)
    if kind == "transformer":
        parser.add_argument("--d-model", type=int, default=32)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.block_size, args.batch_size, args.eval_every) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("步数、上下文长度、批次、评估间隔与学习率必须为正数")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = prepare_data()
    windows = training_windows(data["encoded"]["train"], args.block_size)
    config = {"vocab_size": len(data["vocab"]), "block_size": args.block_size}
    if kind == "transformer":
        config.update(d_model=args.d_model, heads=args.heads, layers=args.layers)
    model = make_model(kind, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pt"
    initial_train = evaluate(model, data["encoded"]["train"])
    initial_val = evaluate(model, data["encoded"]["val"])
    best_loss = initial_val["loss"]
    save_checkpoint(checkpoint, model, kind, config, data["vocab"], data["metadata"], 0, best_loss)
    history = [{"step": 0, "train": initial_train, "validation": initial_val}]
    print(json.dumps({"data": data["metadata"], "vocab_size": len(data["vocab"]), "parameters": sum(p.numel() for p in model.parameters())}, ensure_ascii=False))
    sample_x, sample_y = windows[0]
    print("首个窗口的输入 ID:", sample_x.tolist(), "标签 ID:", sample_y.tolist())
    with torch.no_grad():
        probabilities = F.softmax(model(sample_x.unsqueeze(0))[0, 0], dim=-1)
        values, indices = probabilities.topk(min(3, len(data["vocab"])))
    print("形状: x=[B,T], y=[B,T], logits=[B,T,V]; 首位置初始预测概率前三:",
          [(data["vocab"][i], round(p, 4)) for i, p in zip(indices.tolist(), values.tolist())])
    for step in range(1, args.steps + 1):
        model.train()
        batch = [windows[sampler.randrange(len(windows))] for _ in range(args.batch_size)]
        x, y = torch.stack([pair[0] for pair in batch]), torch.stack([pair[1] for pair in batch])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(data["vocab"])), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            train, val = evaluate(model, data["encoded"]["train"]), evaluate(model, data["encoded"]["val"])
            history.append({"step": step, "train": train, "validation": val})
            print(f"step={step} train_nll={train['loss']:.4f} val_nll={val['loss']:.4f} val_ppl={val['perplexity']:.2f}")
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                save_checkpoint(checkpoint, model, kind, config, data["vocab"], data["metadata"], step, best_loss)
    # 只在验证选择结束后加载最佳权重，对最终测试集做一次完整计分。
    selected, payload = load_checkpoint(checkpoint)
    final_test = evaluate(selected, data["encoded"]["test"])
    metrics = {"model_kind": kind, "config": config, "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "parameters": sum(p.numel() for p in selected.parameters()), "history": history,
               "selected_step": payload["step"], "selected_validation_loss": payload["validation_loss"], "test": final_test,
               "data": data["metadata"], "elapsed_seconds": time.perf_counter() - started,
               "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__), "device": "cpu", "threads": 1, "platform": platform.platform(), "processor": platform.processor()}}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_step": payload["step"], "test": final_test, "elapsed_seconds": metrics["elapsed_seconds"], "checkpoint": str(checkpoint)}, ensure_ascii=False))
