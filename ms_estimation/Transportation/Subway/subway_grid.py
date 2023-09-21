#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/20 21:21

import geopandas as gpd
import pandas as pd
import os
import numpy as np


def cal_data(city):
    """
    calculate the subway ms on grid scale
    :param city:
    :return:
    """
    if city not in city_dict:
        grid_path = f'../../Building/result/{city}/{city}.shp'
    else:
        grid_path = f'../../Building/result/{city_dict[city]}/{city_dict[city]}.shp'
    line_path = f'./data/{city}/{city}.shp'
    station_path = f'./data/{city}/{city}_station.shp'

    grid_shp = gpd.read_file(grid_path)
    line_shp = gpd.read_file(line_path, encoding='utf-8')
    station_shp = gpd.read_file(station_path, encoding='utf-8')

    line_shp.crs = "EPSG:4326"
    line_shp = line_shp.to_crs(crs='EPSG:3857')
    station_shp.crs = 'EPSG:4326'
    station_shp = station_shp.to_crs(crs='EPSG:3857')

    # calculate the line
    line_intersection = gpd.overlay(line_shp, grid_shp, how='intersection')
    line_intersection['inter_length'] = line_intersection.length

    subway_mci = pd.read_csv('./mci/subway_mci.csv')
    subway_mci.fillna(0, inplace=True)

    # create dir if not exist
    if not os.path.exists(f'grid/{city}'):
        os.mkdir(f'grid/{city}')

    line_intersection['Cement'] = line_intersection['inter_length'] * subway_mci[0:1]['Cement'][0] / 1000
    line_intersection['Steel'] = line_intersection['inter_length'] * float(subway_mci[0:1]['Steel'][0]) / 1000
    line_intersection['Total'] = line_intersection['inter_length'] * subway_mci[0:1]['Total'][0] / 1000

    out = line_intersection[['id', 'line', 'inter_length', 'Cement', 'Steel', 'Total']].groupby('id')
    out = out.agg(np.sum).reset_index()
    out.to_csv(f'./grid/{city}/{city}_line_grid.csv', index=False)

    # calculate the station
    station_intersection = gpd.sjoin(station_shp, grid_shp, op='intersects')

    count = len(station_intersection['line'])
    station_intersection['Cement'] = [subway_mci[1:2]['Cement'][1]] * count
    station_intersection['Steel'] = [subway_mci[1:2]['Steel'][1]] * count
    station_intersection['Gravel'] = [subway_mci[1:2]['Gravel'][1]] * count
    station_intersection['Sand'] = [subway_mci[1:2]['Sand'][1]] * count
    station_intersection['Asphalt'] = [subway_mci[1:2]['Asphalt'][1]] * count
    station_intersection['Copper'] = [subway_mci[1:2]['Copper'][1]] * count
    station_intersection['Total'] = [subway_mci[1:2]['Total'][1]] * count

    s_out = station_intersection[
        ['id', 'line', 'Cement', 'Steel', 'Gravel', 'Sand', 'Asphalt', 'Copper', 'Total']].groupby('id')

    s_out = s_out.agg(np.sum).reset_index()
    s_out.to_csv(f'./grid/{city}/{city}_station_grid.csv', index=False)

    # line grid to shape
    line_grid_shp = grid_shp.merge(out, on='id')
    line_grid_shp.to_file(f'./grid/{city}/{city}_line.shp')

    # station grid to shape
    station_grid_shp = grid_shp.merge(s_out, on='id')
    station_grid_shp.to_file(f'./grid/{city}/{city}_station.shp')


if __name__ == '__main__':
    city_dict = {
        'urumqi': 'wulumuqi',
        'lhasa': 'lasa',
        'hohhot': 'huhehaote'
    }
    if not os.path.exists('grid'):
        os.mkdir('grid')
    city_list = [i for i in os.listdir('data')]

    for city in city_list:
        print(city)
        cal_data(city)
