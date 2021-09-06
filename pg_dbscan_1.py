import pandas as pd
from matplotlib import colors
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt


beer = pd.read_excel('gg.xlsx')
# print(beer)

X = beer[["甜", "酸", "苦", "涩", "鲜", "咸"]]
# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)


labels = db.labels_
beer['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
beer.sort_values('cluster_db')

print(beer.groupby('cluster_db').mean())

print(pd.plotting.scatter_matrix(X, c=beer.cluster_db, figsize=(10, 10), s=100))
plt.show()
