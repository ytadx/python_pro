import pandas as pd

df = pd.DataFrame({"key": ['green', 'red', 'blue'],
                   "data1": ['a', 'b', 'c'], "sorce": [33, 61, 99]})
# get_dummies(data,....) 在不指定新列的列名的情况下，将以data原标签对为列名
print("-------df---------")
# 原始矩阵
print(df)
df_dummies1 = pd.get_dummies(df["key"])
print('''-------pd.get_dummies(df["key"])--df_dummies1-------''')
# 在不指定新列的列名的情况下，将以data原标签对为列名
print(df_dummies1)


# prefix参数可以给哑变量的名字加上一个前缀
df_dummies2 = pd.get_dummies(df["key"], prefix="key")
print('''---=pd.get_dummies(df["key"],prefix="key")----df_dummies2-----''')
print(df_dummies2)


# 如果不指定data列的话，默认是所有的分类变量进行one_hot处理
df_dummies3 = pd.get_dummies(df)
print("-------pd.get_dummies(df)---df_dummies3------")
print(df_dummies3)


# prefix参数可以给哑变量的名字加上一个前缀,如果是多个则需要一个列参数
df_dummies4 = pd.get_dummies(df, prefix=["class", "like"])
print('''-------pd.get_dummies(df,prefix=["class","like"])----df_dummies4-----''')
print(df_dummies4)

df_dummies5 = pd.get_dummies(df, columns=["key", "sorce"])
print('''---=pd.get_dummies(df,columns=["key","sorce"])----df_dummies5-----''')
print(df_dummies5)
