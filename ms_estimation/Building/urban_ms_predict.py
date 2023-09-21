#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/3/24 23:51

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split


class CalculateMS:
    def __init__(self, city, ms='Total MS(t)'):
        self.city = city
        self.ms = ms

    def handle_ms(self):
        """
        calculate the density of building_id,
        so that we can get the ms if the buildings are separated by girds
        """
        df = pd.read_csv(f'./ground truth/{self.city}/{self.city}_ms.csv')
        df['density'] = df[self.ms] * 1000 / df['Floor Area (m2)']
        return df[['building_id', 'density']]

    def calculate_ms(self):
        """
        calculate the ms on grids
        """
        df_density = self.handle_ms()
        df = pd.read_csv(f'./ground truth/{self.city}/{self.city}.csv')
        df_merge = pd.merge(df, df_density, on='building_id')

        # according to the formula y = 0.1427x + 1.4141 to calculate the true floor
        df_merge['floor'] = df_merge['height'].apply(lambda x: round(0.1427 * x + 1.4141))
        # calculate the ms according to floor area and density
        df_merge['floor_area'] = df_merge['inter_shape_area'] * df_merge['floor']
        df_merge['inter_ms'] = df_merge['density'] * df_merge['floor_area']

        return df_merge

    def group_ms(self):
        """
        calculate the indicators in the grid. group by grid id
        :return:
        """
        df = self.calculate_ms()[['id', 'inter_shape_area', 'inter_perimeter', 'floor', 'floor_area', 'inter_ms']]
        df_agg = df.groupby('id').agg([np.sum, np.mean, np.size]).reset_index()

        building_ms = df_agg['inter_ms']['sum']

        # select features
        building_id = df_agg['id']
        building_size = df_agg['inter_shape_area']['size']
        sum_shape_area = df_agg['inter_shape_area']['sum']
        mean_shape_area = df_agg['inter_shape_area']['mean']
        sum_perimeter = df_agg['inter_perimeter']['sum']
        mean_perimeter = df_agg['inter_perimeter']['mean']
        sum_floor = df_agg['floor']['sum']
        mean_floor = df_agg['floor']['mean']
        sum_floor_area = df_agg['floor_area']['sum']
        mean_floor_area = df_agg['floor_area']['mean']

        ratio_floor_area = sum_floor_area / sum_shape_area
        ratio_perimeter_shape = sum_perimeter / sum_shape_area

        # get building feature
        df_feature = pd.DataFrame({
            'id': building_id,
            'building_size': building_size,
            'sum_shape_area': sum_shape_area,
            'mean_shape_area': mean_shape_area,
            'sum_perimeter': sum_perimeter,
            'mean_perimeter': mean_perimeter,
            'sum_floor': sum_floor,
            'mean_floor': mean_floor,
            'sum_floor_area': sum_floor_area,
            'mean_floor_area': mean_floor_area,
            'ratio_floor_area': ratio_floor_area,
            'ratio_perimeter_shape': ratio_perimeter_shape,
            'ms': building_ms
        })

        return df_feature

    def split_poi(self):
        """
        poi classification
        :return:
        """
        tmp = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20}
        df = pd.read_csv(f'./ground truth/{self.city}/{self.city}_poi.csv')
        df = df[df['typecode'].isin(tmp)]
        one_hot = pd.get_dummies(df['typecode'])
        one_hot = pd.concat([one_hot, df['id']], axis=1)
        group = one_hot.groupby('id').sum()
        group_sum = group.sum(axis=1)
        group = group.div(group_sum, axis='rows')
        final = pd.concat([group, group_sum], axis=1).reset_index()

        return final

    def feature(self):
        """
        get the final feature
        :return:
        """
        tmp = pd.merge(self.split_poi(), self.group_ms(), how='right')
        tmp.fillna(0, inplace=True)
        return tmp


class PredictMS:
    def __init__(self, city):
        """
        prediction class, get the same feature as train model
        :param city:
        """
        self.city = city

    def group_building(self):
        """
        :return:
        """
        df = pd.read_csv(f'./data/{self.city}/{self.city}.csv')
        df['floor'] = df['height'].apply(lambda x: round(0.1427 * x + 1.4141))
        df['floor_area'] = df['inter_shape_area'] * df['floor']
        df_agg = df.groupby('id').agg([np.sum, np.mean, np.size]).reset_index()

        # select features
        building_id = df_agg['id']
        building_size = df_agg['inter_shape_area']['size']
        sum_shape_area = df_agg['inter_shape_area']['sum']
        mean_shape_area = df_agg['inter_shape_area']['mean']
        sum_perimeter = df_agg['inter_perimeter']['sum']
        mean_perimeter = df_agg['inter_perimeter']['mean']
        sum_floor = df_agg['floor']['sum']
        mean_floor = df_agg['floor']['mean']
        sum_floor_area = df_agg['floor_area']['sum']
        mean_floor_area = df_agg['floor_area']['mean']

        ratio_floor_area = sum_floor_area / sum_shape_area
        ratio_perimeter_shape = sum_perimeter / sum_shape_area

        # get building feature
        df_feature = pd.DataFrame({
            'id': building_id,
            'building_size': building_size,
            'sum_shape_area': sum_shape_area,
            'mean_shape_area': mean_shape_area,
            'sum_perimeter': sum_perimeter,
            'mean_perimeter': mean_perimeter,
            'sum_floor': sum_floor,
            'mean_floor': mean_floor,
            'sum_floor_area': sum_floor_area,
            'mean_floor_area': mean_floor_area,
            'ratio_floor_area': ratio_floor_area,
            'ratio_perimeter_shape': ratio_perimeter_shape,
        })

        return df_feature

    def split_poi(self):
        """
        :return:
        """
        tmp = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20}
        df = pd.read_csv(f'./data/{self.city}/{self.city}_poi.csv')
        df = df[df['typecode'].isin(tmp)]
        one_hot = pd.get_dummies(df['typecode'])
        one_hot = pd.concat([one_hot, df['id']], axis=1)
        group = one_hot.groupby('id').sum()
        group_sum = group.sum(axis=1)
        group = group.div(group_sum, axis='rows')
        final = pd.concat([group, group_sum], axis=1).reset_index()

        return final

    def feature(self):
        """
        合并景观特征和POI，构造特征
        :return:
        """
        tmp = pd.merge(self.split_poi(), self.group_building(), how='right')
        tmp.fillna(0, inplace=True)
        return tmp

    def predict(self):
        """
        预测存量
        :return:
        """
        feature = self.feature()
        building_id, X = feature.iloc[:, 0], feature.iloc[:, 1:]
        return building_id, X


class StatisticsMS:
    def __init__(self, material):
        self.material = material
        self.path = f'./material/{material}/'

    @staticmethod
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

    @staticmethod
    def add_feature(result: str, shpdatafile, outfile):
        """
        add attribute to shp
        :param result
        :param shpdatafile:
        :param outfile:
        :return:
        """
        shpdata = gpd.GeoDataFrame.from_file(shpdatafile)
        print(len(shpdata))

        result_dict = {}
        result_list = []
        with open(result, 'r') as f:
            for r in f.readlines()[1:]:
                result_dict[int(r.split(',')[0])] = float(r.strip().split(',')[1])

        for i in range(0, len(shpdata)):
            if (i in result_dict) and (result_dict[i] > 0):
                result_list.append(result_dict[i])
            else:
                result_list.append(0)

        shpdata['result'] = result_list
        shpdata.to_file(outfile)

    def generate(self):
        """
        get train and test data
        :return:
        """
        # get train feature from beijing guangzhou and shenzhen
        beijing_feature = CalculateMS('beijing', self.material).feature()
        guangzhou_feature = CalculateMS('guangzhou', self.material).feature()
        shenzhen_feature = CalculateMS('shenzhen', self.material).feature()

        # train_test_split, random_state=42
        train_df = pd.concat([beijing_feature, guangzhou_feature, shenzhen_feature])
        ori_X, ori_y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
        X, test_X, y, test_y = train_test_split(ori_X, ori_y, test_size=0.2, random_state=42)

        return X, y, test_X, test_y

    def train(self):
        """
        train model
        :return:
        """
        # create dir and file
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        X, y, test_X, test_y = self.generate()
        f = open(self.path + self.material + f'_statistics.txt', 'a')

        # use random forest to predict the result
        print("Using random forest...")
        rfr = RandomForestRegressor(n_estimators=300,
                                    max_depth=15,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    oob_score=True,
                                    random_state=10)
        rfr.fit(X, y)
        rfr_predict = rfr.predict(test_X)
        # rfr_mse = mean_squared_error(test_y, rfr_predict)
        rfr_rmse = mean_squared_error(test_y, rfr_predict, squared=False)
        rfr_mae = mean_absolute_error(test_y, rfr_predict)
        rfr_mape = self.mape(test_y, rfr_predict)
        rfr_r2 = r2_score(test_y, rfr_predict)

        print('The root mean squared error of RFR is', rfr_rmse)
        print('The mean absolute error of RFR is', rfr_mae)
        print('The mape of RFR is', rfr_mape)
        print('The R2 of RFR is', rfr_r2)

        f.write('Random forest result:\n')
        f.write(f'The root mean squared error of RFR is {rfr_rmse}\n')
        f.write(f'The mean absolute error of RFR is {rfr_mae}\n')
        f.write(f'The mape of RFR is {rfr_mape}\n')
        f.write(f'The R2 of RFR is {rfr_r2}\n')
        f.write('\n')

        # linear regression to predict the result
        print("Using linear regression...")
        reg = linear_model.LinearRegression()
        reg.fit(X, y)
        reg_predict = reg.predict(test_X)

        # reg_mse = mean_squared_error(test_y, reg_predict)
        reg_rmse = mean_squared_error(test_y, reg_predict, squared=False)
        reg_mae = mean_absolute_error(test_y, reg_predict)
        reg_mape = self.mape(test_y, reg_predict)
        reg_r2 = r2_score(test_y, reg_predict)

        print('The root mean squared error of LinearRegression is', reg_rmse)
        print('The mean absolute error of LinearRegression is', reg_mae)
        print('The mape of LinearRegression is', reg_mape)
        print('The R2 of LinearRegression is', reg_r2)

        f.write('Linear Regression result:\n')
        f.write(f'The root mean squared error of LinearRegression is {reg_rmse}\n')
        f.write(f'The mean absolute error of LinearRegression is {reg_mae}\n')
        f.write(f'The mape of LinearRegression is {reg_mape}\n')
        f.write(f'The R2 of LinearRegression is {reg_r2}\n')
        f.write('\n')

        # xgboost result
        print("Using XGBoost regression...")
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

        xlr.fit(X, y)
        xgbr_predict = xlr.predict(test_X)

        # xgbr_mse = mean_squared_error(test_y, xgbr_predict)
        xgbr_rmse = mean_squared_error(test_y, xgbr_predict, squared=False)
        xgbr_mae = mean_absolute_error(test_y, xgbr_predict)
        xgbr_mape = self.mape(test_y, xgbr_predict)
        xgbr_r2 = r2_score(test_y, xgbr_predict)

        print('The root mean squared error of XGBR is', xgbr_rmse)
        print('The mean absolute error of XGBR is', xgbr_mae)
        print('The mape of XGBR is', xgbr_mape)
        print('The R2 of XGBR is', xgbr_r2)

        f.write('XGBoost result:\n')
        f.write(f'The root mean squared error of XGBR is {xgbr_rmse}\n')
        f.write(f'The mean absolute error of XGBR is {xgbr_mae}\n')
        f.write(f'The mape of XGBR is {xgbr_mape}\n')
        f.write(f'The R2 of XGBR is {xgbr_r2}\n')
        f.write('\n')

        # calculate the total ms
        f.write(f'Total True value {sum(test_y)}\n')
        f.write(f'Total LinearRegression {sum(reg_predict)}\n')
        f.write(f'Total RFR {sum(rfr_predict)}\n')
        f.write(f'Total XGBR {sum(xgbr_predict)}\n')
        f.write(f'Total error is {abs(sum(test_y) - sum(rfr_predict))/sum(test_y)}\n')
        f.close()

        # plt.scatter(test_y, test_y, color='darkorange', label='Input data', s=10.)
        # plt.scatter(test_y, reg_predict, color='navy', label='reg', s=10.)
        # plt.scatter(test_y, xgbr_predict, color='cornflowerblue', label='xgb', s=10.)
        # plt.scatter(test_y, rfr_predict, color='c', label='rfr_predict', s=10.)
        #
        # plt.xlabel('True')
        # plt.ylabel('Predict')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.legend()
        # plt.show()
        # plt.savefig(self.path + self.material + f'_train_result.jpg', dpi=300)
        # plt.close()

        # return random forest
        return rfr

    def predict(self):
        """
        predict according to the train model
        :return:
        """
        # get the train model
        model = self.train()

        city_list = []
        for city in os.listdir('./data'):
            if city.startswith('.') or city.endswith('.csv'):
                pass
            else:
                city_list.append(city)

        for city in city_list:
            # create dir
            os.mkdir(f'./material/{self.material}/{city}/')
            city_predict = PredictMS(city)
            city_building_id, city_x = city_predict.predict()
            result = model.predict(city_x)

            df = pd.DataFrame({
                'id': city_building_id,
                'result': result
            })
            # save the result
            df.to_csv(f'./material/{self.material}/{city}/{city}_result.csv', index=False)

            # save to shp
            city_result = f'./material/{self.material}/{city}/{city}_result.csv'
            city_shp = f'./data/{city}/{city}.shp'
            city_out = f'./material/{self.material}/{city}/{city}_outfile.shp'
            self.add_feature(city_result, city_shp, city_out)
            print(city + '  finish')


if __name__ == '__main__':
    material_list = [
        'Total MS(t)',
        'Cement(kg)',
        'Steel(kg)',
        'Timber(kg)',
        'Brick(kg)',
        'Gravel(kg)',
        'Sand(kg)',
        'Asphalt(kg)',
        'Lime(kg)',
        'Glass(kg)',
        'Ceramic(kg)',
    ]
    if os.path.exists('material'):
        read_material = os.listdir('./material')
    else:
        read_material = []
        os.mkdir('material')

    for m in material_list:
        if m not in read_material:
            static = StatisticsMS(material=m)
            static.predict()
    # beijing_y = CalculateMS('shenzhen').feature().iloc[:, -1]
    # beijing_y.to_csv('shenzhen.csv')
