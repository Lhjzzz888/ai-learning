# 08｜从 Q/K/V 到小型因果 Transformer

先完成 [07 的数据与评估](../07-character-language-model/README.md)，再从根目录运行：

```bash
python 08-tiny-transformer/inspect_attention.py
python 08-tiny-transformer/train.py --steps 200 --output-dir artifacts/08
python 09-inference/generate.py --checkpoint artifacts/08/best.pt --prompt 清晨 --seed 17 --max-new-tokens 60
```

默认 CPU 单线程，2 层、隐藏维度 32、4 个头、上下文 32、批次 8，学习率 0.003；这是比路线建议更小的起跑配置。先用 `--steps 10` 看一轮是否可运行。所有权重随机初始化，没有预训练权重和远程数据下载。代码集中在 [`learning/language.py`](../learning/language.py)，两个课程入口调用共同训练循环，保证词表、数据划分和计分方式相同。

## 顺着张量走一遍

字符 ID `[B,T]` 经字符 embedding 与可学习的位置 embedding 相加，得到 `[B,T,D]`。每个 PreNorm decoder 块包含：

```text
x -> LayerNorm -> 因果多头注意力 -> + x
  -> LayerNorm -> Linear(D,4D) -> GELU -> Linear(4D,D) -> + 残差
最后 LayerNorm -> Linear(D,V) -> [B,T,V] logits
```

自注意力由张量操作手写：一层线性投影产生 Q/K/V，变形为 `[B,H,T,D/H]`；分数 `Q @ K.transpose(-2,-1) / sqrt(D/H)` 为 `[B,H,T,T]`。主对角线以上填负无穷，softmax 后恰为零，位置 t 只能看 0..t（可以看当前位置，因为标签是下一个字符）。权重乘 V，各头拼接后再线性投影。`D` 必须能被头数 `H` 整除。程序未调用现成 `nn.Transformer`、`nn.MultiheadAttention` 或融合注意力。

`inspect_attention.py` 打印“清晨有风。”的随机模型第一个头的 5×5 矩阵。第一行只允许关注自己，各行和为 1，上三角为零；它解释 mask，并不是对语义关系的结论。位置向量区分字符出现的先后；残差为信息和梯度提供路径；LayerNorm 稳定每个位置的隐藏表示。本最小实现无 dropout，适合先学习核心流程，再单独扩展。

## 验证与对照

运行 `python -m unittest discover -s tests -p test_language.py -v`。测试固定随机种子，在极小的交替字符序列上训练 bigram 和 Transformer，要求训练 NLL 明显下降并低于小阈值；把未来输入换掉，前面位置的 logits 应保持一致；另查标签错位、未知字符、文档边界、完整 target 计分和检查点重载。极小过拟合证明训练循环能学，不是泛化能力证明。

训练入口支持 `--d-model 32 --heads 4 --layers 2`，以及与 07 相同的 `--steps --output-dir --seed --block-size --batch-size --eval-every --learning-rate`。负/零预算及不合法维度会报错。每次验证均 `eval()` 并关闭梯度；训练继续前恢复 `train()`。初始及每次完整验证中最低 NLL 对应 `best.pt`；然后统一测试，不能根据测试结果回头挑预算或改模型。需要多次比较时，把最终测试延后到方案锁定，并不要把每轮脚本附带的测试分数用于决策。

比较 `artifacts/07/metrics.json` 与 `artifacts/08/metrics.json` 的语料哈希、划分、词表大小、block_size、验证/测试 token 数后，再看 NLL/PPL。记录参数量、步数、时间；Transformer 结构更复杂，但参数量和耗时要以实测为准：bigram 的表有 `V²` 个参数，本例甚至比小 Transformer 的参数更多。不预设 Transformer 一定胜过 bigram。两者都只在极小且高度重复的原创中文上训练，可能记住片段、重复输出或组合出没有意义的话。

## 练习

1. 将一行 Q/K 数值代入缩放点积，算出 mask 前后分数与 softmax；解释为何 mask 不能在 softmax 后仅清零而不重新归一化。
2. 故意移除因果 mask，再运行未来扰动测试，确认它确实能发现信息泄漏；恢复实现后再训练。
3. 固定所有其他条件，只比较隐藏维度 16 与 32（都能整除 4 个头）。只根据验证集选择，再记录耗时与参数代价。
4. 保持相同词表与上下文，对照 bigram 的生成；记下各自至少一个失败案例，不因语言流畅就宣称更低测试损失。

保存与推理细节、结课报告模板见 [09](../09-inference/README.md)。
