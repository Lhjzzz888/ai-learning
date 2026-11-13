"""第一课：用一元线性回归预测房价（示例数据，单位：平方米、万元）。"""
import argparse
import math
import random

# 虚构数据，仅用于学习，不能用来估算真实房价。
DATA = [(45, 92), (50, 105), (55, 110), (60, 125), (65, 128),
        (70, 145), (75, 150), (80, 168), (85, 170), (90, 185),
        (95, 193), (100, 205), (105, 220), (110, 224), (115, 239),
        (120, 245), (125, 260), (130, 267), (135, 280), (140, 288)]


def fit(samples):
    """最小二乘法：找到让训练数据的平方误差之和最小的直线。"""
    if len(samples) < 2:
        raise ValueError('训练数据至少需要两条')
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in samples):
        raise ValueError('训练数据必须是有限数值')
    mean_x = sum(x for x, _ in samples) / len(samples)
    mean_y = sum(y for _, y in samples) / len(samples)
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    if denominator == 0:
        raise ValueError('训练数据的面积不能全部相同')
    slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def evaluate_models(train, test):
    """在同一测试集上比较直线与常数基线；所有参数只从训练集计算。"""
    if not test:
        raise ValueError('测试集不能为空')
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in test):
        raise ValueError('测试数据必须是有限数值')
    slope, intercept = fit(train)
    baseline = sum(y for _, y in train) / len(train)
    return {
        'slope': slope,
        'intercept': intercept,
        'baseline_prediction': baseline,
        'baseline_mae': sum(abs(y - baseline) for _, y in test) / len(test),
        'linear_mae': sum(abs(y - (slope * x + intercept)) for x, y in test) / len(test),
    }


def main():
    parser = argparse.ArgumentParser(description='用房屋面积预测示例房价')
    parser.add_argument('--area', type=float, default=100, help='房屋面积（平方米），默认 100')
    args = parser.parse_args()
    if not math.isfinite(args.area) or args.area <= 0:
        parser.error('面积必须是大于 0 的有限数值')

    # 固定随机种子，确保每次划分的训练集和测试集相同。
    samples = DATA.copy()
    random.Random(42).shuffle(samples)
    train, test = samples[:16], samples[16:]
    results = evaluate_models(train, test)
    slope, intercept = results['slope'], results['intercept']
    predictions = [(x, y, slope * x + intercept) for x, y in test]

    print(f'训练集：{len(train)} 条，测试集：{len(test)} 条')
    print(f'模型：预测房价 = {slope:.3f} × 面积 + {intercept:.3f}（万元）')
    print('\n测试集（面积 / 实际价格 / 预测价格）：')
    for area, actual, predicted in predictions:
        print(f'{area:>3} 平方米 / {actual:>6.1f} 万元 / {predicted:>6.1f} 万元')
    print(f"\n常数基线（训练标签均值）：{results['baseline_prediction']:.2f} 万元")
    print(f"常数基线测试 MAE：{results['baseline_mae']:.2f} 万元")
    print(f"线性回归测试 MAE：{results['linear_mae']:.2f} 万元")
    print(f'{args.area:g} 平方米的示例预测价格：{slope * args.area + intercept:.2f} 万元')


if __name__ == '__main__':
    main()

