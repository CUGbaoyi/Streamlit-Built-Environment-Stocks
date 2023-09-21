#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/1 11:36

import os

import pandas as pd
import geopandas as gpd


def calculate_road(city):
    """

    :param city:
    :return:
    """
    # road type reclassification
    road_type = pd.read_csv('./mci/osm_road_type_reclassification.csv')
    # read road mci
    road_mci = pd.read_csv('./mci/road_mci.csv')
    road = pd.merge(road_type, road_mci, how='left', on='Classification')
    shapefile = f'./data/{city}/{city}.shp'

    gdf = gpd.read_file(shapefile)
    # projection
    gdf = gdf.to_crs(crs='EPSG:3857')
    gdf['oid'] = range(len(gdf))
    gdf['road_length'] = gdf.length
    gdf['highway'] = gdf['highway'].apply(lambda x: x.strip('[]').replace('\'', '').split(',')[0])
    city_df = pd.merge(gdf, road, on='highway', how='left')
    city_df['length'] = city_df['road_length'].astype('float64')

    # calculate road ms
    city_df['Cement'] = city_df['length'] * city_df['Cement'] / 1000
    city_df['Steel'] = city_df['length'] * city_df['Steel'] / 1000
    city_df['Gravel'] = city_df['length'] * city_df['Gravel'] / 1000
    city_df['Sand'] = city_df['length'] * city_df['Sand'] / 1000
    city_df['Asphalt'] = city_df['length'] * city_df['Asphalt'] / 1000
    city_df['Mineral Powder'] = city_df['length'] * city_df['Mineral Powder'] / 1000
    city_df['Total'] = city_df['length'] * city_df['Total'] / 1000

    # save to road ms to shapefile named city_ms.shp
    city_df.to_file(f'./data/{city}/{city}_ms.shp')

    out = city_df[['oid', 'osmid', 'highway', 'length', 'Classification', 'Cement', 'Steel', 'Gravel',
                   'Sand', 'Asphalt', 'Mineral Powder', 'Total']]
    out.to_csv(f'./data/{city}/{city}_ms.csv', index=False)


if __name__ == '__main__':
    city_list = [i for i in os.listdir('./data')]

    for city in city_list:
        print(city)
        calculate_road(city)
