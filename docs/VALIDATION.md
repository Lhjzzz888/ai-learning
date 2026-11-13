# 代码案例验证记录

验证日期：2026-11-13。范围：模块 00–09 的 CPU 教学案例。以下是一次固定配置的实际结果，用于确认流程，不是模型排行榜或现实任务效果承诺。

## 环境与测试

- Windows，Python 3.12.14，NumPy 2.2.6，scikit-learn 1.7.2，PyTorch 2.9.1+cpu。
- 在独立依赖目录安装并验证；神经网络默认 CPU 单线程，无数据下载、GPU 或预训练权重。
- `python -m unittest discover -s tests -v`：**25 项通过**。覆盖训练集限定预处理/词表、数据与来源组隔离、有限差分梯度、分类阈值、短批次/短文本块加权、小样本过拟合、因果性及保存重载。
- 本次实测平台是 Windows；没有声称验证过所有操作系统、Python 版本或 GPU。

## 实际运行的命令

在仓库根目录、安装完整依赖后运行；以下命令均退出成功：

```bash
python 00-python-numpy/numpy_basics.py
python 01-linear-regression/house_price.py --area 85
python 02-gradient-descent/gradient_descent.py
python 03-logistic-regression/logistic_regression.py
python 04-tree-models/compare_models.py
python 05-pytorch-mlp/train.py --epochs 12
python 05-pytorch-mlp/train.py --tiny-overfit 16 --epochs 150 --output-dir artifacts/05-overfit
python 06-cnn/train.py --epochs 12
python 07-character-language-model/train.py --steps 200
python 08-tiny-transformer/train.py --steps 200
python 09-inference/generate.py --checkpoint artifacts/08/best.pt --prompt 清晨 --seed 17 --max-new-tokens 60
```

完整机器可读结果由各脚本写入 `artifacts/`，不随源码提交。不要将不同训练预算的指标直接当成完全相同的复现实验。

## 观察到的结果

| 案例 | 预算/选择方式 | 实测观察 |
| --- | --- | --- |
| 00 NumPy | 固定三条样本 | Python/NumPy 预测一致，MAE=1；错误广播演示为 3×3 |
| 01 线性回归 | 固定 16/4 划分 | 常数基线 MAE=64.00，线性回归 MAE=3.40（虚构万元） |
| 02 梯度下降 | 正常学习率 0.1，300 步 | 与闭式解的最大测试预测差约 2.84e-14；CV 选择 Lasso，最终测试 MSE 约 772.70 |
| 03 逻辑回归 | 验证 F1 选择阈值 0.2 | 测试 F1 约 0.568；多数类基线 accuracy 约 0.846，但 F1=0 |
| 04 树模型 | 三折 CV | 选中随机森林，开发集 CV F1 约 0.674±0.047；输出最终测试与错误样本报告 |
| 05 MLP | 12 epochs | 验证 accuracy 约 0.961，测试 accuracy 约 0.958；重载 logits 最大差为 0 |
| 05 小样本诊断 | 16 样本、150 epochs | 末轮训练损失约 0.000460、训练 accuracy=1.0；未进行最终测试 |
| 06 MLP/CNN | 各 12 epochs，按验证交叉熵选择 | 选中 MLP；两模型分别导出 5 个错分图像，重载 logits 最大差均为 0 |
| 07 bigram | 200 steps，验证选择第 50 步 | 验证 NLL 从约 6.038 降至 4.956；测试 NLL 约 5.069，PPL 约 159.06 |
| 08 Transformer | 200 steps，验证选择第 100 步 | 验证 NLL 从约 5.720 降至 4.603；测试 NLL 约 4.905，PPL 约 134.98 |
| 09 推理 | 选中的 Transformer，种子 17 | 成功加载检查点，按提示“清晨”生成 60 个新 token |

06 中 CNN 的测试 accuracy 约为 0.972，虽然高于 MLP，但仍保留按预先指定的**验证交叉熵**做出的 MLP 选择，不能看到测试分数后换赢家。模型选择指标不同于单纯 accuracy；本次预算也不足以给出模型类型的一般结论。

## 语言模型的能力边界

本次语料总共只有 1568 字符，训练/验证/测试字符数为 1044/259/265；验证计分 255 token、测试 261 token。词表仅从训练集建立，验证未知字符约占 12.36%，测试约占 12.45%；这些字符被编码为 `<unk>` 并纳入评分。bigram 为 60,516 参数，小 Transformer 为 42,486 参数；复杂结构不一定意味着本例参数更多。

推理结果开头为“清晨过的果画之。朋友听朋点子上单们。”，虽然程序完成了生成循环，文本仍明显不连贯。它说明训练、验证、保存和采样流程已打通，并不说明模型具备通用中文写作或问答能力。后续应先增加许可明确且来源分离的语料，只根据验证集决定修改，再保留独立最终测试。
