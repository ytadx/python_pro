import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

# Krzanowski, W. J. (2000). “Principles of Multivariate analysis”.
# Gower, J. C. (1975). “Generalized procrustes analysis”

# Procrustes 分析，对两个数据集的相似性测试
# 给定两个相同大小的矩阵，procrustes 对两者进行标准化，使得：
# Procrustes 将 最优变换应用于第二个矩阵（包括缩放/膨胀、旋转和反射）以最小化，或两个输入数据集之间逐点差异的平方和。
# 如果两个数据集具有不同的维度（不同的列数），只需将零列添加到两者中的较小者。
# M^2=sum(data1-data2)^2 差的平方和

a = np.array([[2, 2, 2], [4, 2, 1], [7, 1, 2.5], [4.5, 2, 1.5]])
b = np.array([[2, 1, 2], [4, 3, 4], [11, 1, 7], [1, 2, 2.8]])
mtx1, mtx2, disparity = procrustes(a, b)
print(mtx1)
print(mtx2)
print(disparity)

A = np.array([[2, 2, 2], [4, 2, 1], [7, 1, 2.5], [4.5, 2, 1.5]])
R, sca = orthogonal_procrustes(A, np.fliplr(A))
print(R)
print(sca)