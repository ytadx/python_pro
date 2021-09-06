import numpy as np
import pymysql
import pandas as pd
from sqlalchemy import create_engine

from scipy.spatial.distance import cdist

db = pymysql.connect(host='47.100.53.205', port=7009, user='dataanalyst',
                     passwd='123456', db='idc', charset='utf8')

# 初始化数据库连接，使用pymysql模块
# create_engine()初始化数据库连接时需要提供：'数据库类型+数据库驱动名称://用户名:密码@地址:端口号/数据库名'
# MySQL的用户：root, 密码:147369, 端口：3306,数据库：infectious
engine = create_engine('mysql+pymysql://dataanalyst:123456@47.100.53.205:7009/idc')

# 查询语句，选出goods表中的所有数据
sql = '''select * from goods;'''

# read_sql_query的两个参数: sql语句， 数据库连接
df = pd.read_sql_query(sql, engine)

# 输出employee表的查询结果
# print(df)

# 列名
colums = []
row = []
for index, row in df.iteritems():
    colums.append(index)
    # print(index,row[0],row[1],row[2])

a = df.shape[0]
b = df.shape[1]
data = np.empty(shape=[a, b])

df2 = pd.DataFrame(data, columns=colums, index=df.index)

for i in range(a):
    for j in range(b):
        if j >= 4:
            if float(df.iloc[i, j]):
                df2.iloc[i, j] = float(df.iloc[i, j])
            else:
                df2.iloc[i, j] = float(str(df.iloc[i, j]).strip())
        else:
            df2.iloc[i, j] = df.iloc[i, j]


# df2.to_excel('tt.xlsx')

# print(df[df.iloc[:, 4:].isnull().T.any()])

df3 = pd.get_dummies(df2["品类"], prefix="_")


for i in range(len(df3.columns)):
    # pd.concat([df2, pd.DataFrame(columns=[df3.columns[i]])])
    df2.insert(204+i, df3.columns[i], df3[df3.columns[i]]*6)

df2.to_excel('tt.xlsx')

columns = [c for c in df2.columns if c not in ['名称', '公司', '品牌', '品类']]
df3 = pd.DataFrame(np.array(df2.iloc[:, 4:]), index=df2['名称'])
a = ['euclidean', 'correlation']


for i in a:
    dist = cdist(df3, df3, metric=i)
    file_mame = 'no_dist_' + i + '.xlsx'
    dist = pd.DataFrame(dist, index=df2['名称'], columns=df2['名称'])
    dist.to_excel(file_mame)
