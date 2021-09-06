import matplotlib
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import graphviz

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, plot_importance
import warnings

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

dfx = pd.read_excel(r'产品数据20210818_1.xlsx')
# one_hot
dd = pd.get_dummies(dfx["品类"], prefix="s")

# X的数据
for i in range(len(dd.columns)):
    dfx[dd.columns[i]] = dd[dd.columns[i]]
# print(dfx)

# Y数据从数据库来

engine = create_engine('mysql+pymysql://dataanalyst:123456@47.100.53.205:7009/idc')
sql = '''select * from user_like_copy where user_id=19'''
dfy = pd.read_sql_query(sql, engine)
ddxy = pd.merge(dfx, dfy, left_on='名字', right_on='product_name')

columns = [c for c in ddxy.columns if c not in ['品类', 'user_id', 'product_name', 'product_id']]
columns_y = [c for c in ddxy.columns if c not in ['品类', 'user_id', 'product_name', 'product_id', 'level']]

ddxx = pd.DataFrame(columns=columns)
ddxt = pd.DataFrame(columns=columns_y)  # 预测使用数据集

for col in columns:
    ddxx[col] = ddxy[col]

for col in columns_y:
    ddxt[col] = dfx[col]

# ddxx.to_excel('tt_data.xlsx')
# ddxt.to_excel('tt_pre.xlsx')

X = ddxx.iloc[:, 1:26]
y = ddxx.iloc[:, 27]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123457)

#
# def xg_eval_mae(yhat, dtrain):
#     y = dtrain.get_label()
#     return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


dtrain = xgb.DMatrix(X_train, y_train)

# xgb_params = {
#     'seed': 0,
#     'eta': 0.1,
#     'colsample_bytree': 0.5,
#     'silent': 1,
#     'subsample': 0.5,
#     'objective': 'reg:linear',
#     'max_depth': 5,
#     'min_child_weight': 3
# }

#
# # 使用交叉验证，这里没有调节参数
# bst_cvl = xgb.cv(xgb_params, dtrain,
#                  num_boost_round=100,  # 最大迭代次数,树个数
#                  nfold=3, seed=5,  # 表示几折
#                  feval=xg_eval_mae,  # 以这个标准进行衡量（损失吧？）
#                  maximize=False,
#                  early_stopping_rounds=10,
#                  # 连着10次round没有提升，迭代停止，输出最好的轮数
#                  verbose_eval=10,
#                  # 每10轮打印一次评价指标
#                  )
# print('CV score:', bst_cvl.iloc[-1, :]['test-mae-mean'])
#
# plt.figure()
# bst_cvl[['train-mae-mean','test-mae-mean']].plot()
# plt.show()


model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:linear')
model.fit(X_train, y_train)

ans = model.predict(X_test)

# ans_len = len(ans)
# id_list = np.arange(10441, 17441)
# data_arr = []
# for row in range(0, ans_len):
#     data_arr.append([int(id_list[row]), ans[row]])
# np_data = np.array(data_arr)

# 显示重要特征
plot_importance(model)
plt.show()

print(mean_squared_error(y_test, model.predict(X_test)))

xgb.plot_tree(model, num_trees=2)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
xgb.to_graphviz(model, num_trees=80, rankdir='LR')