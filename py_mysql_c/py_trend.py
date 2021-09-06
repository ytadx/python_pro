"""
1.做最小二乘拟合，把序列拟合成一条直线;
2.根据直线的斜率k可以得知序列的主要走势：
例如：(1)k > 0.1763 上升  (2) k < -0.1763 下降 (3)其他
3.然后计算序列各点到直线的距离（和方差一样）
设定一个阈值L，统计超过L的点数目，点数目越多说明序列震荡越厉害
"""

import numpy as np
import math


def trendline(data):  # 拟合曲线
    order = 1
    index = [i for i in range(1, len(data) + 1)]  # x轴坐标
    coeffs = np.polyfit(index, list(data), order)  # 曲线拟合
    return coeffs


def judge_slope(coeffs, data, degree, shake=1):
    tan_k = math.tan(degree * math.pi / 180)  # 注意弧度转化
    # print(coeffs[0])
    # print(tan_k)
    count = 0
    for i, d in enumerate(data):  # i+1相当于横坐标，从1开始
        y = np.polyval(coeffs, i + 1)
        count += (y - d) ** 2
    if coeffs[0] >= tan_k:
        return "上升，残差：" + str(count)
    elif coeffs[0] <= -tan_k:
        return "下降，残差：" + str(count)
    else:
        return get_shake(coeffs, data, shake)


def slope_distance(coeffs, coeffs2):
    tan_k1 = math.tan(coeffs[0] * math.pi / 180)  # 注意弧度转化
    tan_k2 = math.tan(coeffs2[0] * math.pi / 180)
    Cobb = float(
        math.fabs(np.arctan((coeffs[0] - coeffs2[0]) / (float(1 + coeffs[0] * coeffs2[0]))) * 180 / np.pi) + 0.5)
    return Cobb


def get_shake(coeffs, data, shake):
    count = 0
    for i, d in enumerate(data):  # i+1相当于横坐标，从1开始
        y = np.polyval(coeffs, i + 1)
        count += (y - d) ** 2
    # print("count: ",count)
    if count > shake:
        return "波动，残差：" + str(count)
    else:
        return "平稳，残差：" + str(count)


if __name__ == '__main__':
    data1 = [1,-3,5,-1]
    data2 = [4,1,4.5,4.5]
    print(trendline(data1))
    print(trendline(data2))
    print(slope_distance(trendline(data1), trendline(data2)))
    # res = judge_slope(trendline(data), data, degree=1, shake=1)
