#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/20 22:09

import geopandas as gpd
import pandas as pd
import os
import numpy as np


def cal_data(city):
    """
    calculate the ms of each grid in city
    :param city:
    :return:
    """
    if city not in city_dict:
        grid_path = f'../../Building/result/{city}/{city}.shp'
    else:
        grid_path = f'../../Building/result/{city_dict[city]}/{city_dict[city]}.shp'

    line_path = f'./data/{city}/{city}.shp'

    grid_shp = gpd.read_file(grid_path)
    line_shp = gpd.read_file(line_path, encoding='utf-8')
    line_shp = line_shp[['gml_id', 'geometry']]

    line_shp.crs = "EPSG:4326"
    line_shp = line_shp.to_crs(crs='EPSG:3857')

    line_intersection = gpd.overlay(line_shp, grid_shp, how='intersection')
    line_intersection['length'] = line_intersection.length

    rail_mci = pd.read_csv('./mci/railway_mci.csv')

    line_intersection['Cement'] = line_intersection['length'] * rail_mci[0:1]['Cement'][0] / 1000
    line_intersection['Steel'] = line_intersection['length'] * rail_mci[0:1]['Steel'][0] / 1000
    line_intersection['Gravel'] = line_intersection['length'] * rail_mci[0:1]['Gravel'][0] / 1000
    line_intersection['Sand'] = line_intersection['length'] * rail_mci[0:1]['Sand'][0] / 1000
    line_intersection['Lime'] = line_intersection['length'] * rail_mci[0:1]['Lime'][0] / 1000
    line_intersection['Fly ash'] = line_intersection['length'] * rail_mci[0:1]['Fly ash'][0] / 1000
    line_intersection['Copper'] = line_intersection['length'] * rail_mci[0:1]['Copper'][0] / 1000
    line_intersection['Aluminum'] = line_intersection['length'] * rail_mci[0:1]['Aluminum'][0] / 1000
    line_intersection['Total'] = line_intersection['length'] * rail_mci[0:1]['Total'][0] / 1000

    # create dir if not exist
    if not os.path.exists(f'grid/{city}'):
        os.mkdir(f'grid/{city}')

    out = line_intersection[
        ['id', 'gml_id', 'length', 'Cement', 'Steel', 'Gravel', 'Sand', 'Lime', 'Fly ash', 'Copper', 'Aluminum',
         'Total']].groupby('id')
    out = out.agg(np.sum).reset_index()
    out.to_csv(f'./grid/{city}/{city}_grid.csv', index=False)

    grid_shp = grid_shp.merge(out, on='id')
    if not grid_shp.empty:
        grid_shp.to_file(f'./grid/{city}/{city}.shp')


if __name__ == '__main__':
    city_dict = {
        'urumqi': 'wulumuqi',
        'lhasa': 'lasa',
        'hohhot': 'huhehaote'
    }
    if not os.path.exists('grid'):
        os.mkdir('grid')
    city_list = [i for i in os.listdir('./data')]

    for city in city_list:
        print(city)
        cal_data(city)
