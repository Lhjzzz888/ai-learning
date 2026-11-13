"""从仓库根目录执行：python 07-character-language-model/train.py。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.language import training_main


if __name__ == "__main__":
    training_main("bigram")
