# 02｜梯度下降、多特征与正则化

[返回目录](../README.md) · [上一课](../01-linear-regression/README.md) · [下一课](../03-logistic-regression/README.md)

本课同时回答两个问题：梯度下降如何逼近闭式解，以及复杂模型怎样只用训练数据选择正则化。需要 NumPy、scikit-learn；按根目录 README 安装后无需联网或 GPU。

```bash
python 02-gradient-descent/gradient_descent.py
python 02-gradient-descent/gradient_descent.py --output-dir artifacts/my-gradient
```

结果保存在 `artifacts/02-gradient-descent/report.json`；`loss_curve.csv` 是可用表格软件画折线图的真实训练曲线，首行包含更新前的损失，后续每行对应一次更新。标准输出只显示核心比较，完整参数和划分索引保存在 JSON。

## 一元回归：保持数据和目标一致

脚本直接读取第一课 `house_price.py` 的 `DATA`，使用相同的 `random.Random(42)` 洗牌和 16/4 划分。`X` 为 `(16,1)`，`y` 为 `(16,)`，权重为 `(1,)`，预测为 `(16,)`。只在训练集计算面积均值和标准差，将面积标准化；标签减训练标签均值，让教学优化更容易观察。训练后把斜率和截距还原到原始单位，与第一课最小二乘 `fit` 的测试预测对照。

目标 `MSE = mean((Xw+b-y)²)`，残差 `r = Xw+b-y`，梯度为 `dw = 2 X.T @ r / n`、`db = 2 mean(r)`，更新为 `w -= learning_rate * dw`、`b -= learning_rate * db`。最小二乘和这里的梯度下降优化同一个 MSE；它们不是用 MAE 拟合。

在同一训练集上固定演示三种学习率：`0.001` 较慢、`0.1` 正常下降、`1.1` 发散。这里的结论依赖标准化后的二次目标，不能照抄到任意数据。极大或非有限损失会提前终止，CSV 中后续留空，避免用假数据补齐曲线。比较正常下降与闭式解的 `max_prediction_difference`；它应接近数值精度。不要根据四个测试样本反复选择学习率；这些学习率已在教学示例中固定。

## 多特征：把标准化放进交叉验证

独立实验由 `make_regression` 生成 180 个样本、40 个特征，其中 5 个为生成目标提供信息，标签叠加标准差 25 的高斯噪声，种子为 42。该生成器的结构是线性的，这会偏向线性模型；结论不适用于真实房价。

1. 先留出 25% 测试样本，余下 135 个样本参与选择。
2. 固定相同的 4 折 `KFold(shuffle=True, random_state=42)`，主指标为 MSE（越小越好）。
3. 每个候选都用 `Pipeline(StandardScaler, model)`，确保标准化只在当前折的训练部分拟合。
4. 比较普通线性回归、Ridge 的 `alpha=[0.1,1,10,100]`、Lasso 的 `alpha=[0.1,1,10]`。
5. 每类报告最优候选的 CV MSE 均值及折间标准差；按均值锁定一个赢家，才在全部训练样本重拟合并进行一次最终测试。

scikit-learn 的评分器使用负 MSE，因为搜索器统一选择最大分数；输出时已取负还原为正 MSE。折间标准差表示这次划分的波动，不是置信区间。JSON 的 `winner_test_mse` 只属于选定赢家，其他模型没有在测试集排名。

Ridge 用 L2 惩罚限制权重幅度；Lasso 用 L1 惩罚，可能把无关特征权重压为零。`alpha` 越大惩罚越强，但强度必须在 CV 中选择。正则化会引入偏差，不能保证每次得分都更好，也不能靠挑选最好测试结果证明有效。

## 练习与验收

- 用 `loss_curve.csv` 画三条曲线（跨度大时使用对数纵轴），解释每条曲线形状；手推一个样本的梯度。
- 修改一个因素，例如噪声或训练样本数，先记录假设，再比较 CV 表。产生新实验时保留原报告，不覆盖证据。
- 检查普通线性回归与正则化模型使用相同样本、折和指标；指出把 `StandardScaler.fit_transform(X)` 放在划分前会泄漏什么信息。
- 完成路线中的验收：解释学习率、还原后参数、CV 选择和正则化局限。小样本的合成结果不是现实效果承诺。

测试入口：`python -m unittest discover -s tests -p test_tabular.py`，包括有限差分梯度与已知直线的收敛检查。
