import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt

rel = pd.read_csv('rel.csv', encoding='gb18030')
# data = pd.read_csv('gg_all_data.csv', encoding='gb18030', sep=',')

model_sample = pd.read_excel('gg.xlsx')
columns = [c for c in model_sample.columns if c not in ['序号', '样品']]

for col in columns:
    try:
        model_sample[col] = model_sample[col].astype('float64')
    except:
        print(col)


row = model_sample.shape[0]
col = rel.shape[0]
qm = np.zeros((row, col))
t_name = []

for index, row in rel.iterrows():
    a = row['r1']
    b = row['r2']
    t_name.append(a + '_' + b)
    c = row['col']
    qm[:, index] = qm[:, index] + c

qm = pd.DataFrame(qm, index=model_sample['样品'], columns=t_name)
print(qm)


# def mean_std(data, col, epsilon=1e-5, fillna=0):
#     data_norm = data.copy()
#     for col_i in col:
#         data_norm[col_i] = (data[col_i] - data[col_i].mean()) / (data[col_i].std() + epsilon)
#     data_norm = data_norm.fillna(fillna)
#     return data_norm


# data_norm = mean_std(model_sample, columns)  # 标准化
data_norm = model_sample  # 原始矩阵


data_norm2 = data_norm[columns].T.rename(columns=model_sample['样品'])
# pandas中相关矩阵
corr = data_norm2.corr('pearson')
corr.to_excel('corr.xlsx')

data_norm3 = pd.DataFrame(np.array(data_norm.iloc[:, 2:]), index=model_sample['样品'])
# data_norm3.isnull().to_excel('test.xlsx')

# 使用scipy 计算距离

a = ['cosine', 'euclidean', 'cityblock', 'jaccard', 'mahalanobis', 'correlation']

print(data_norm3)
#
# for i in a:
#     dist = cdist(data_norm3, data_norm3, metric=i)
#     file_mame = 'dist_' + i + '.xlsx'
#     dist = pd.DataFrame(dist, index=model_sample['样品'], columns=model_sample['样品'])
#     dist.to_excel(file_mame)

# sk计算距离
# dist = pairwise_distances(data_norm3, metric="euclidean")


# for i in columns:
#     print(data_norm[i])
#     print(data_norm.iloc[0, 2:])
#     # print(corr_(data_norm[i], data_norm.iloc[0, 2:].T))
#
#
# def corr_col(data_norm, col, y='样品', percentile=0.25):
#     col_corr = []
#     for col_i in col:
#         col_corr.append(corr_(data_norm[col_i], data_norm[y]))
#     df_corr = pd.DataFrame({'col': col, 'corr': col_corr}).dropna()
#     col_corr_ = df_corr['corr']
#     col_corr_percentile = np.percentile(col_corr_, percentile)
#     return df_corr, col_corr, col_corr_percentile

#
# corr_col(data_norm, columns)
# print(corr_col)


# print(rel)  # 读取专家建议
# # print(data.T[1:])
# # data2 = data.T[1:]
# # data2 = pd.DataFrame(data.iloc[:, 1:].T)
# # print(data.iloc[:, 1:].corr(method='pearson', min_periods=1))
# # print()
# # print(np.corrcoef(data2))
#
# for index, row in rel.iterrows():
#     # print(row['r1'], row['r2'], row['col'])
#     a = row['r1']
#     b = row['r2']
#     c = row['col']
#     for i in range(len(data.columns)):
#         if data.columns[i] == a:
#             for j in range(len(data.columns)):
#                 if data.columns[j] == b:
#                     print(data.columns[j] + ' ' + str(np.mean(data[str(data.columns[j])])))
#                     # print(data[str(data.columns[j])] * c)
#                     print(data.columns[j] + ' ' + str(np.mean(data[str(data.columns[j])] * c)))
#
# # attributes = data.columns
# #
# # weights = [10, 5, 2]
# # data['Score'] = data[attributes].mul(weights).sum(1)



