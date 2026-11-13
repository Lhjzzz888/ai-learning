"""加载 07 或 08 的最佳验证检查点并自回归采样。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from learning.language import generate, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="CPU 检查点推理，无数据或权重下载")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, payload = load_checkpoint(args.checkpoint)
    try:
        result = generate(model, payload["vocab"], args.prompt, args.seed, args.max_new_tokens, args.temperature)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"model_kind": payload["model_kind"], "selected_step": payload["step"], **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
