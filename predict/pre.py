
# 用pmml进行模型预测
from pypmml import Model
from sklearn import metrics
import pandas as pd


# 应用PMML模型
model = Model.load("1.xml")
model_sample = pd.read_excel('果茶_all.xlsx')
# print(model_sample)

y_pred = pd.DataFrame(model.predict(model_sample))
print(y_pred)
# y_pred = y_pred["predicted_second_heart_attack"].values.tolist()
# print('accuracy is', metrics.accuracy_score(y_pred, y_test))
