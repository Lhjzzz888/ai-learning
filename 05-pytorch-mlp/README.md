# 05｜PyTorch MLP：把训练闭环跑通

[返回学习目录](../README.md) · [阶段要求](../docs/ROADMAP.md#05把神经网络原理与-pytorch-训练循环一起学) · [下一课 CNN](../06-cnn/README.md)

本课用 scikit-learn 包内的 `load_digits` 训练一个小型数字分类器，不下载数据。先安装根目录说明中的神经网络依赖，在仓库根目录运行：

```powershell
python 05-pytorch-mlp/train.py --epochs 20
```

先检查极小子集能否学会，再扩大数据：

```powershell
python 05-pytorch-mlp/train.py --tiny-overfit 32 --epochs 120 --output-dir artifacts/05-tiny
python 05-pytorch-mlp/train.py --epochs 3 --output-dir artifacts/05-smoke
```

`--tiny-overfit` 仅截取训练部分的前 N 个样本，仍使用原来的验证集选择检查点，跳过最终测试。观察 `last_epoch_train` 的损失是否接近零；极小集记忆成功只能说明训练流程能学，不能说明泛化好。验证最佳检查点可能早于最后一轮，因此它不一定是训练损失最低的版本。`--epochs 3` 是流程试跑，不代表达到验收表现。

## 数据和形状

数据共 1797 张 8×8 灰度图、10 类数字。`learning/vision.py` 先按固定种子进行两次分层划分，得到 1077/360/360 个训练／验证／测试样本。两课使用完全相同的样本 ID。像素原值为 0–16，统一除以已知常数 16，转换成 `float32`；这一步不从任何数据集拟合均值、方差或极值。标签是 `long` 类型的类别编号。

```text
输入       [N, 1, 8, 8]
Flatten    [N, 64]
Linear     [N, 64]      由 --hidden-dim 设置宽度
ReLU       [N, 64]
Linear     [N, 10]      原始 logits
```

默认网络有 4810 个可训练参数，使用 Adam、学习率 0.01、批次 64、CPU 单线程。`--seed` 同时控制划分、初始化和训练顺序；相同种子不保证跨硬件或库版本完全相同。

## 读懂每一步

`train_epoch` 先 `model.train()`，对每批执行 `zero_grad → forward → CrossEntropy → backward → step`。梯度默认会累加，所以每批更新前必须清零。交叉熵直接接收原始 logits；它内部完成所需的 log-softmax，提前再加 softmax 会改变目标。

`evaluate` 用 `model.eval()` 设置层的推理行为，再用 `torch.no_grad()` 关闭梯度记录。两者负责不同事情，`eval()` 本身不会关闭自动求导。当前 MLP 没有 Dropout 或 BatchNorm，但仍保留完整的模式切换习惯。

每批平均损失乘以该批样本数，再除以总样本数；准确率也按总正确数计算。最后一个批次较短时，不能把每批指标简单平均。历史中第 0 轮是更新前的评估；后续 `train_loss` 是该轮在线批次损失，参数在批次间变化。`val_loss` 则来自整轮更新后的固定模型，两条曲线不是完全相同测量条件。

手算一个标量例子：`x=2, w=3, b=1, y=5`，预测为 7，平方误差为 4。链式法则得到 `dL/dw=8, dL/db=4`。脚本用自动求导打印同一结果 `hand_gradient_check`，可先遮住结果自己推导。

## 产物与重载

默认产物在 `artifacts/05-pytorch-mlp/`：

- `curves.csv`、`curves.svg`：包含第 0 轮的训练／验证曲线，可直接打开 SVG 或导入 CSV 作图。
- `best.pt`：按验证交叉熵最低选择，包含模型配置、`state_dict`、轮数、验证损失、划分种子和像素尺度。
- `summary.json`：实际配置、参数量、耗时、验证指标、最终测试指标与重载差异。

检查点选择包含第 0 轮；训练配置和选择规则固定后才评估一次最终测试集。保存前后的同一批验证样本预测用精确一致断言检查，`reload_max_abs_difference` 应为 0。

加载已有检查点，不训练，预测验证集中的一张图：

```powershell
python 05-pytorch-mlp/train.py --checkpoint artifacts/05-pytorch-mlp/best.pt --predict-index 0
```

共享训练工具的行为测试可单独运行：

```powershell
python -m unittest discover -s tests -p test_vision.py -v
```

输出保留原始样本 ID、真实标签、预测标签与 10 类概率。重载用 `torch.load(..., weights_only=True)`，在代码中按配置重建网络。产物只用于推理；未保存优化器和随机状态，不承诺从中精确恢复训练。

## 练习与局限

1. 先用 32 个样本检查损失明显下降；若做不到，检查标签、梯度清零、学习率和 logits。
2. 一次只改一个因素，例如隐藏宽度或学习率，用验证曲线记录结论；不要依据最终测试结果反复调参。
3. 解释训练下降而验证上升的情况，再进入 CNN 的结构比较。

8×8 小图、随机独立划分和有限样本只能作为教学实验，不能证明对手机照片或不同书写来源有效。数据来源是 scikit-learn 随安装包提供的 digits 数据，详细说明可在 Python 中查看 `sklearn.datasets.load_digits().DESCR`；不要把它当成自己采集的数据。
