"""从随机权重训练手写多头因果 decoder。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.language import training_main


if __name__ == "__main__":
    training_main("transformer")
