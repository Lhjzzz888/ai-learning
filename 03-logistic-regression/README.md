# 03｜逻辑回归、概率与分类评估

[返回目录](../README.md) · [上一课](../02-gradient-descent/README.md) · [下一课](../04-tree-models/README.md)

在类别不均衡的二分类任务中，模型输出概率还不够：需要明确阈值、错误代价和评估边界。本课使用 NumPy 与 scikit-learn，CPU 离线运行。

```bash
python 03-logistic-regression/logistic_regression.py
python 03-logistic-regression/logistic_regression.py --output-dir artifacts/my-logistic
```

输出每个候选阈值的验证集 precision、recall、F1，以及固定阈值后的最终测试指标；完整结果、每部分的正类比例与样本索引保存在 `artifacts/03-logistic-regression/report.json`。查看实际运行的报告，不把合成数据指标当作业务结论。

## 数据与划分

`make_classification` 用种子 42 生成 1200 个独立样本、12 个特征：6 个有信息、2 个冗余，其余为无关特征；类别权重为 `[0.85,0.15]`，随机翻转标签比例参数为 `0.02`，`class_sep=1.1`。翻转后实际正类比例不必恰好为 15%。

先分层留出 20% 测试集，再在余下部分分层留出 25% 作为验证集，得到训练/验证/测试 720/240/240。`X_train` 为 `(720,12)`，`y_train` 为 `(720,)`。所有索引不重叠；相同 seed 可复现这次划分。`Pipeline(StandardScaler, LogisticRegression)` 仅在训练集拟合，验证集选择阈值时不重拟合。多数类基线也只根据训练标签确定类别。

这是随机独立样本的教学任务。有同一个人的多条记录时应按人分组；有时间顺序时应按时间划分，不能直接照用随机分层。

## 概率、损失与阈值

逻辑回归计算 `z = Xw+b`，通过 `sigmoid(z)=1/(1+exp(-z))` 得到正类概率。二元交叉熵为 `-mean(y*log(p)+(1-y)*log(1-p))`；scikit-learn 在数值稳定实现中拟合这个目标并带默认 L2 正则化。`predict_proba(X)[:,1]` 的形状是 `(n,)`，表示类别 1 的概率。

预先声明主指标为验证 F1，尝试 `[0.2,0.35,0.5,0.65]` 四个阈值。`p >= threshold` 判为正类。F1 相同时优先选择接近 0.5 的阈值，再选较小阈值；规则写在代码中，不能看完测试集再改。这里把漏报与误报都视为需要控制；如果漏报更昂贵，下一轮应事先定义 recall 约束或代价指标，而不是假定 F1 适合所有场景。降低阈值通常能减少漏报，却会增加误报。

| 指标 | 计算 | 含义 |
| --- | --- | --- |
| confusion matrix | `[[TN, FP], [FN, TP]]` | 行为真实类，列为预测类 |
| precision | `TP/(TP+FP)` | 报为正类时有多少是真的 |
| recall | `TP/(TP+FN)` | 真正的正类中找回多少 |
| F1 | `2TP/(2TP+FP+FN)` | precision/recall 的调和平均 |
| accuracy | `(TP+TN)/全部` | 总体正确率 |

当分母为零时，precision/recall/F1 明确返回 0。多数类模型的 accuracy 可以很高，但正类 recall 和 F1 为零；所以不能只看 accuracy。阈值锁定后，报告基线与逻辑回归在同一个测试集上的表现；没有根据测试集重新挑阈值或训练模型。

多分类下一步会把 sigmoid 扩展为 softmax：对每个类别的 logit 取指数后归一化，再用真实类别的负对数概率作交叉熵。多分类通常选择最大概率类别；它与这里单个正类阈值的机制不同。

## 练习与验收

1. 从 JSON 中的混淆矩阵手算 precision、recall、F1，与程序结果核对。
2. 只看验证表，解释阈值降低时哪些错误变化，以及为什么多数类基线的 accuracy 会误导。
3. 在实验前写下误报和漏报的代价，定义新的选择规则后开启一次新实验；最终测试不能回流到选择过程。
4. 能指明标准化、模型拟合、阈值选择分别读取哪部分数据，解释合成数据和默认正则化的局限。

行为测试覆盖数据索引边界、阈值选择、无正类预测时指标，以及标准化确实只学习训练均值。
