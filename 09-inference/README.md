# 09｜加载、生成与复现

先按 [07](../07-character-language-model/README.md) 或 [08](../08-tiny-transformer/README.md) 训练，再在仓库根目录执行：

```bash
python 09-inference/generate.py --checkpoint artifacts/08/best.pt --prompt 清晨 --seed 17 --max-new-tokens 60
python 09-inference/generate.py --checkpoint artifacts/07/best.pt --prompt 清晨 --seed 17 --max-new-tokens 60 --temperature 0.8
```

一个推理入口支持两种模型；检查点中的 `model_kind` 决定实例化什么网络，`config` 决定尺寸，`vocab` 决定字符 ID。加载先在 CPU 用 `torch.load(..., weights_only=True)` 读取，再 `load_state_dict`，最后 `eval()`、关闭梯度。推理不重新读取语料或拟合词表，因此复制 `best.pt` 就带上了必要的模型信息。环境需要兼容的 PyTorch，具体安装见[首页](../README.md)。仅加载可信来源的文件。

## 提示与采样的边界

每步取当前上下文的最后一个位置 logits，除以正温度，softmax 后用带独立种子的 CPU 生成器抽样；把新 token 追加，再重复。模型每次最多看最后 `block_size` 个字符，超长历史从左侧截断，位置从 0 重新编号；返回的 `text` 保留完整原提示，`prompt_context_characters` 表示初始真正送入模型的字符数。bigram 即使收到多个字符，仍只利用前一个。

未见提示字符编码为 `<unk>`，JSON 显示 `unknown_prompt_characters`；显示文本仍保留原提示，所以“显示了这个字”不等于模型认识这个字。模型采样到 ID 0 时显示字面量 `<unk>`。空提示报错，因为没有 BOS；没有 EOS 自动终止，恰好追加 `--max-new-tokens` 个 token（可为 0，不能负数）。`<unk>` 的显示长度大于一个字符，因此字符串长度不等于生成 token 数。温度必须是有限正数；温度改变采样分布，不改变固定检查点的评估 NLL/PPL。

相同检查点、提示、种子、温度和生成长度在同一环境中可重复；跨硬件、PyTorch 版本的逐字符一致性不作保证。默认 CPU 单线程，没有 GPU 自动选择或隐藏下载。

## 检查点与实验文件

`best.pt` 包含格式版本、模型类型/配置、`state_dict`、词表（ID 0 为 `<unk>`）、语料 SHA-256、来源与文档划分、规范化和计分规则、未知率、最佳验证步数及损失。它是**推理检查点**，没有优化器与采样器/随机状态，不能精确接着原训练继续；如需恢复训练，练习补全这些状态并验证下一步损失一致。

`metrics.json` 还包含训练参数、初始及后续训练/验证曲线点、最终测试结果、参数量、耗时以及 Python/PyTorch/平台/CPU 信息。输出目录可用 `--output-dir` 重定向；重新训练到同一路径会覆盖这两个文件，请用不同目录保存对照实验。

## 干净环境复现清单

1. 复制仓库与所需 `best.pt`；按首页创建虚拟环境，安装深度学习依赖，记录实际 `python --version` 和 `python -m pip freeze`。
2. 先运行上面的加载命令，再运行 `python -m unittest discover -s tests -p test_language.py -v` 检查重载预测一致性、因果性和训练行为。
3. 用 `python 08-tiny-transformer/train.py --steps 10 --output-dir artifacts/reproduce-smoke` 复现一个短训练闭环；正式复现时恢复报告中的全部配置与训练预算。
4. 核对语料哈希、划分、未知率、词表与 token 数。把新输出与原记录比较，不把短训练等同于原预算的性能复现。

## 结课报告模板

复制 [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md)，填入实测值。报告应包含数据/许可、基线、验证选择、最终测试、曲线、环境、成功及失败例子；不能仅展示好看的生成片段。

练习：改变采样温度并固定其他条件，对比重复与连贯性；给模型一个含未见字的提示，解释未知数；尝试超过上下文长度的提示，检查 JSON 中的有效上下文数。下一步只选一个改动，把原因、预算与验证结果写清楚。
