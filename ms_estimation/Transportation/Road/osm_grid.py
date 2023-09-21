#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/4 21:13

"""
map the ms to grid
"""

import geopandas as gpd
import pandas as pd
import os
import numpy as np


def cal_data(city):
    if city not in city_dict:
        grid_path = f'../../Building/result/{city}/{city}.shp'
    else:
        grid_path = f'../../Building/result/{city_dict[city]}/{city_dict[city]}.shp'
    road_path = f'./data/{city}/{city}_ms.shp'

    # read grid data and osm data
    grid_shp = gpd.read_file(grid_path)
    road_shp = gpd.read_file(road_path)

    # project
    road_shp = road_shp.to_crs(crs='EPSG:3857')
    road_shp['road_length'] = road_shp.length

    # calculate the intersection length of road and grid
    road_intersection = gpd.overlay(road_shp, grid_shp, how='intersection')
    road_intersection['inter_length'] = road_intersection.length

    # calculate  mci
    road_type = pd.read_csv('./mci/osm_road_type_reclassification.csv')
    road_mci = pd.read_csv('./mci/road_mci.csv')
    road = pd.merge(road_type, road_mci, how='left', on='Classification')

    # calculate the ms according to the inter length and mci
    road_intersection['highway'] = road_intersection['highway'].apply(
        lambda x: x.strip('[]').replace('\'', '').split(',')[0])
    road_intersection = road_intersection[['oid', 'osmid', 'highway', 'inter_length', 'road_length', 'id']]
    road_merge = pd.merge(road_intersection, road, on='highway', how='left')
    road_merge['inter_length'] = road_merge['inter_length'].astype('float64')
    road_merge['id'] = road_merge['id'].astype('int')
    road_merge['oid'] = road_merge['oid'].astype('int')

    road_merge['Cement'] = road_merge['inter_length'] * road_merge['Cement'] / 1000
    road_merge['Steel'] = road_merge['inter_length'] * road_merge['Steel'] / 1000
    road_merge['Gravel'] = road_merge['inter_length'] * road_merge['Gravel'] / 1000
    road_merge['Sand'] = road_merge['inter_length'] * road_merge['Sand'] / 1000
    road_merge['Asphalt'] = road_merge['inter_length'] * road_merge['Asphalt'] / 1000
    road_merge['Mineral Powder'] = road_merge['inter_length'] * road_merge['Mineral Powder'] / 1000
    road_merge['Total'] = road_merge['inter_length'] * road_merge['Total'] / 1000

    out = road_merge[[
        'oid',
        'osmid',
        'highway',
        'inter_length',
        'road_length',
        'Classification',
        'Cement',
        'Steel',
        'Gravel',
        'Sand',
        'Asphalt',
        'Mineral Powder',
        'Total',
        'id'
    ]]

    # create dir if not exist
    if not os.path.exists(f'grid/{city}'):
        os.mkdir(f'grid/{city}')

    # grid data to shp
    ms = out[['id', 'Cement', 'Steel', 'Gravel', 'Sand', 'Asphalt', 'Mineral Powder', 'Total']].groupby('id')
    ms = ms.agg(np.sum).reset_index()
    ms.to_csv(f'./grid/{city}/{city}_grid.csv', index=False)

    grid_shp = grid_shp.merge(ms, on='id')
    grid_shp.to_file(f'./grid/{city}/{city}.shp')


if __name__ == '__main__':
    city_dict = {
        'urumqi': 'wulumuqi',
        'lhasa': 'lasa',
        'hohhot': 'huhehaote'
    }
    if not os.path.exists('grid'):
        os.mkdir('grid')
    city_list = [i.split('.csv')[0] for i in os.listdir('./data')]

    for city in city_list:
        print(city)
        cal_data(city)
