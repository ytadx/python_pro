from scipy.spatial.distance import cdist
import numpy as np
import scipy.stats

x = [(1, 2, 4), (2, 4, 8), (1, 2, 6), (1, 4, 4)]

a = ['seuclidean', 'canberra']

# rule1: 第一对距离为0，去掉
# rule2: 第1，第3，第4 距离一样，去掉

for i in a:
    dist = cdist(x, x, metric=i)
    print(i + ': ' + str(dist))
    print('--------------------------------------')

