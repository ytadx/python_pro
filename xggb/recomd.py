import pandas as pd
from scipy.spatial.distance import cdist

df1 = pd.read_excel(r'产品数据20210818_2.xlsx')

d = pd.get_dummies(df1["品类"], prefix="s")

for i in range(len(d.columns)):
    df1[d.columns[i]] = d[d.columns[i]]

x = df1.iloc[:, 2:]

a = ['cosine', 'euclidean']

for i in a:
    dist = cdist(x, x, metric=i)
    file_mame = 'dist_' + i + '.xlsx'
    dd = pd.DataFrame(dist, index=df1.iloc[:, 0], columns=df1.iloc[:, 0])
    dd.to_excel(file_mame)
