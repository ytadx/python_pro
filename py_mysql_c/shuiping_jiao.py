import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sqlalchemy import create_engine
import pandas as pd
from sklearn import linear_model
from scipy.spatial import distance


def li_list(x):
    Newx = [[x] for x in x]
    return Newx


def trendline(y):  # 拟合曲线
    order = 1
    x = [i for i in range(len(y))]  # x轴坐标
    reg = linear_model.LinearRegression()
    reg.fit(li_list(x), li_list(y))
    # print('intercept_:%.3f' % reg.intercept_)
    # print('coef_:%.3f' % reg.coef_)
    mes = mean_squared_error(li_list(y), reg.predict(li_list(x)))
    # print('Mean squared error: %.3f' % mes)
    r2 = r2_score(li_list(y), reg.predict(li_list(x)))
    # print('Variance score: %.3f' % r2)
    # 1-((y_test - LR.predict(X_test)) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    # rs = reg.score(li_list(x), li_list(y))
    # print('score: %.3f' % rs)
    return reg.coef_, reg.intercept_, mes, r2


def proess_cal(x1):
    slope = trendline(x1)[0][0]
    angle = np.rad2deg(np.arctan(slope))
    # print('slope: ' + str(slope))
    # print('angle: ' + str(angle))
    return slope, angle


if __name__ == '__main__':

    engine = create_engine('mysql+pymysql://dataanalyst:123456@47.100.53.205:7009/idc')

    # s1230: '5%蔗糖水溶液',s1273 '8%蔗糖水溶液',s1219 '10%蔗糖水溶液',s1288 '12%蔗糖水溶液'
    # select user_id,s1221,s1273,s1219,s1288
    # from user_product_evaluation_pro
    # where feature='甜' and feature<>'user_total'
    # and s1221+s1273+s1219+s1288>0

    # s1230 : 三得利沁柠水 甜：3.5
    # s1245 : 水溶C100柠檬味 甜：5.1
    # s1235 : 康师傅冰红茶 甜：7.5

    # s1230 : 三得利沁柠水 酸：3
    # s1235 : 康师傅冰红茶 酸：6.2
    # s1245 : 水溶C100柠檬味 酸：9.6

    # s1230 : 三得利沁柠水 甜：2.5
    # s1235 : 康师傅冰红茶 甜：4.5
    # s1245 : 水溶C100柠檬味 甜：6.5

    sql = '''
        SELECT user_id,feature,s1230,s1235,s1245
        FROM idc.user_product_evaluation_pro
        WHERE feature = '柠檬' AND feature <> 'user_total'
        AND Total>0
    '''

    # read_sql_query的两个参数: sql语句， 数据库连接
    df = pd.read_sql_query(sql, engine)

    # mm = (list(['std', '甜', [0],np.mean(df.iloc[:, 2:], 0)[1],np.mean(df.iloc[:, 2:], 0)[2]]))

    # 取列均为 标准
    # new = pd.DataFrame({'user_id': -100000, 'feature': '甜', 's1230': np.mean(df.iloc[:, 2:], 0)[0],
    #                     's1245': np.mean(df.iloc[:, 2:], 0)[1], 's1235': np.mean(df.iloc[:, 2:], 0)[2], }, index=[0]
    #                    )

    # 取 数据库内标准

    sql = '''
        SELECT CONCAT('s',a.product_id) product_id,b.`甜`,b.`酸`,b.柠檬,product_name from
        (select DISTINCT product_id,product_name from idc.user_product_evaluation
        ) a left join 
        (select * from idc.goods) b on a.product_name=b.名称 order by b.`柠檬` asc
    '''

    df2 = pd.read_sql_query(sql, engine)
    temp = df2.iloc[:, :4].T

    new = pd.DataFrame({'user_id': -100000, 'feature': '柠檬', 's1230': temp.iloc[3][0],
                        's1235': temp.iloc[3][1], 's1245': temp.iloc[3][2]}, index=[0]
                       )

    df = df.append(new, ignore_index=True)
    print(new)

    result = pd.DataFrame(
        columns=('user_id', 'feature', 's1230', 's1235', 's1245', 'slope', 'intercept', 'mes', 'r2'))

    for i in range(df.shape[0]):
        x = df.iloc[i, 2:]  # X数据
        a = list(np.array(df.iloc[i, :]))
        coef, intercept, mes, r2 = trendline(x)
        b = list([round(coef[0][0], 3), round(intercept[0], 3), round(mes, 3), round(r2, 3)])
        # plt.plot(x1, label=str(df.iloc[i, 0]) + ': slope:' + str(round(b[0], 1)) + ': angle:' + str(int(b[1])))
        a.extend(b)
        result = result.append(pd.DataFrame(
            {'user_id': a[0], 'feature': a[1], 's1230': a[2], 's1235': a[3], 's1245': a[4],
             'slope': a[5],
             'intercept': a[6], 'mes': a[7], 'r2': a[8]}, index=[i]))

    result['dist'] = distance.cdist(result.iloc[:, 2:5], new.iloc[:, 2:], 'euclidean')

    result.to_sql('user_product_evaluation_lemon', con=engine, index=False, if_exists='replace')

    # plt.plot(df.iloc[0, 1:], '-', color='r', linewidth=3,
    #          label='avg' + ': slope:' + str(round(proess_cal(df.iloc[0, 1:])[0], 1)) + ': angle:' + str(int(proess_cal(df.iloc[0, 1:])[1])))
    # plt.legend(loc='upper left')
    # plt.show()
    # a = list(df.columns)
    # b = a.extend(['slope', 'angle'])
    # pd.DataFrame(df_2, columns=a).to_sql(name='user_product_evaluation_cal',
    #                                      con=engine.connect(), if_exists="append",
    #                                      index=False)
