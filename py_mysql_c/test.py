import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import pandas as pd
from sklearn import linear_model


import matplotlib
print(matplotlib.get_data_path())  # 数据路径

def li_list(x):
    Newx = [[x] for x in x]
    return Newx


y = [5, 6.3, 2.7, 3.6]
x = [0, 1, 2, 3]
li_list(x)

model = linear_model.LinearRegression()
model.fit(li_list(x), li_list(y))
print('intercept_:%.3f' % model.intercept_)
print('coef_:%.3f' % model.coef_)
print('Mean squared error: %.3f' % mean_squared_error(li_list(y), model.predict(li_list(x))))
print('Variance score: %.3f' % r2_score(li_list(y), model.predict(li_list(x))))
# 1-((y_test - LR.predict(X_test)) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
print('score: %.3f' % model.score(li_list(x), li_list(y)))
plt.scatter(li_list(x), li_list(y), color='green')
plt.plot(li_list(x), model.predict(li_list(x)), color='red', linewidth=3)
plt.show()


