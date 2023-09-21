#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/7/13 10:46

from urban_ms_predict import CalculateMS
import xgboost as xgb
from geopandas import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred):
    """
    calculate mape
    :param y_true:
    :param y_pred:
    :return:
    """
    n = len(y_true)
    res = 0
    y_true = y_true.tolist()
    for i in range(n):
        if y_true[i] == 0:
            continue
        res += abs(y_true[i] - y_pred[i]) / y_true[i]

    return res / n


def train_model(x, y, test_x, test_y, case_num):
    rfr = RandomForestRegressor(n_estimators=300,
                                max_depth=15,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                oob_score=True,
                                random_state=10)
    rfr.fit(x, y)
    rfr_predict = rfr.predict(test_x)
    rfr_mse = mean_squared_error(test_y, rfr_predict) / (max(y) * max(y))
    rfr_mae = mean_absolute_error(test_y, rfr_predict) / max(y)
    rfr_mape = mape(test_y, rfr_predict)

    print('The mean squared error of RFR is', rfr_mse)
    print('The mean absolute error of RFR is', rfr_mae)
    print('The mape of RFR is', rfr_mape)

    f.write('Random forest result:\n')
    f.write(f'The mean squared error of RFR is {rfr_mse}\n')
    f.write(f'The mean absolute error of RFR is {rfr_mae}\n')
    f.write(f'The mape of RFR is {rfr_mape}\n')
    f.write('\n')

    ###################

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    reg_predict = reg.predict(test_x)

    reg_mse = mean_squared_error(test_y, reg_predict) / (max(y) * max(y))
    reg_mae = mean_absolute_error(test_y, reg_predict) / max(y)
    reg_mape = mape(test_y, reg_predict)

    print('The mean squared error of LinearRegression is', reg_mse)
    print('The mean absolute error of LinearRegression is', reg_mae)
    print('The mape of LinearRegression is', reg_mape)

    f.write('Linear Regression result:\n')
    f.write(f'The mean squared error of LinearRegression is {reg_mse}\n')
    f.write(f'The mean absolute error of LinearRegression is {reg_mae}\n')
    f.write(f'The mape of LinearRegression is {reg_mape}\n')
    f.write('\n')

    ###################
    xlr = xgb.XGBRegressor(max_depth=20,
                           learning_rate=0.1,
                           n_estimators=300,
                           verbosity=1,
                           silent=None,
                           objective='reg:squarederror',
                           booster='gbtree',
                           n_jobs=20,
                           nthread=None,
                           gamma=0,
                           min_child_weight=1,
                           max_delta_step=0,
                           subsample=0.8,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           colsample_bynode=1,
                           reg_alpha=0,
                           reg_lambda=1,
                           scale_pos_weight=1,
                           base_score=0.5,
                           random_state=0,
                           seed=None,
                           missing=None)

    xlr.fit(x, y)
    xgbr_predict = xlr.predict(test_x)

    xgbr_mse = mean_squared_error(test_y, xgbr_predict) / (max(y) * max(y))
    xgbr_mae = mean_absolute_error(test_y, xgbr_predict) / max(y)
    xgbr_mape = mape(test_y, xgbr_predict)

    print('The mean squared error of XGBR is', xgbr_mse)
    print('The mean absolute error of XGBR is', xgbr_mae)
    print('The mape of XGBR is', xgbr_mape)

    f.write('XGBoost result:\n')
    f.write(f'The mean squared error of XGBR is {xgbr_mse}\n')
    f.write(f'The mean absolute error of XGBR is {xgbr_mae}\n')
    f.write(f'The mape of XGBR is {xgbr_mape}\n')
    f.write('\n')

    ###################
    f.write(f'Total True value {sum(test_y)}\n')
    f.write(f'Total LinearRegression {sum(reg_predict)}\n')
    f.write(f'Total RFR {sum(rfr_predict)}\n')
    f.write(f'Total XGBR {sum(xgbr_predict)}\n')
    f.write('\n')

    ###################
    plt.scatter(test_y, test_y, color='darkorange', label='Input data', s=10.)

    plt.scatter(test_y, reg_predict, color='navy', label='reg', s=10.)
    plt.scatter(test_y, xgbr_predict, color='cornflowerblue', label='xgb', s=10.)
    plt.scatter(test_y, rfr_predict, color='c', label='rfr_predict', s=10.)

    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.title(f'Case ({case_num})')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'./model comparision statistics/Case{case_num}.png', dpi=300)
    plt.close()


def case1():
    """
    One city for training, another for validation.
    We take the data in Beijing to train the model and the data in Shenzhen to valid
    :return:
    """
    train_df = beijing_feature
    test_X, test_y = shenzhen_feature.iloc[:, 1:-1], shenzhen_feature.iloc[:, -1]
    X, y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
    train_model(X, y, test_X, test_y, case_num=1)


def case2():
    """
    Two cities for training, the rest one for validation.
    We take Beijing and Guangzhou for training and shenzhen for validation
    :return:
    """
    train_df = pd.concat([beijing_feature, guangzhou_feature])
    test_X, test_y = shenzhen_feature.iloc[:, 1:-1], shenzhen_feature.iloc[:, -1]
    X, y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
    train_model(X, y, test_X, test_y, case_num=2)


def case3():
    """
    Randomly select 80% of grid within one city to train the model,
    and the remaining 20% for validation.

    We take Beijing as an example
    :return:
    """
    train_df = beijing_feature
    ori_X, ori_y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
    # set random_state=42
    X, test_X, y, test_y = train_test_split(ori_X, ori_y, test_size=0.2, random_state=42)
    train_model(X, y, test_X, test_y, case_num=3)


def case4():
    """
    Mix the grids of cities and randomly select 80% of the grid
    to train the model, the remaining 20% for validation.

    (e.g., Beijing, Guangzhou, and Shenzhen in this case)
    :return:
    """
    train_df = pd.concat([beijing_feature, guangzhou_feature, shenzhen_feature])
    ori_X, ori_y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
    # set random_state=42
    X, test_X, y, test_y = train_test_split(ori_X, ori_y, test_size=0.2, random_state=42)
    train_model(X, y, test_X, test_y, case_num=4)


if __name__ == '__main__':
    # get true ms
    beijing_feature = CalculateMS('beijing').feature()
    guangzhou_feature = CalculateMS('guangzhou').feature()
    shenzhen_feature = CalculateMS('shenzhen').feature()

    f = open("./model comparision statistics/model_comparision_statistics.txt", 'a')

    # case 1
    f.write('Case (1) ' + '*' * 20 + '\n')
    case1()

    # case 2
    f.write('Case (2) ' + '*' * 20 + '\n')
    case2()

    # case 1
    f.write('Case (3) ' + '*' * 20 + '\n')
    case3()

    # case 1
    f.write('Case (4) ' + '*' * 20 + '\n')
    case4()

    f.close()
