"""打印一个随机注意力头，检查未来位置的概率为零。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from learning.language import TinyTransformer


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    text = "清晨有风。"
    model = TinyTransformer(vocab_size=6, block_size=5, d_model=8, heads=2, layers=1).eval()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        embedded = model.embedding(x) + model.position(torch.arange(5))
        normalized = model.blocks[0].norm1(embedded)
        weights = model.blocks[0].attention.attention_weights(normalized)[0, 0]
    print("列=被关注的 key，行=当前 query；字符:", list(text))
    print("softmax(QK^T / sqrt(head_dim) + causal_mask)，head_dim=4")
    for character, row in zip(text, weights.tolist()):
        print(character, " ".join(f"{value:.3f}" for value in row))
    print("每行和为 1；上三角为 0。随机注意力不代表学到了语言知识。")


if __name__ == "__main__":
    main()
