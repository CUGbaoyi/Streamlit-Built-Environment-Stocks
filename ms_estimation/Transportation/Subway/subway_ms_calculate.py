#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/20 15:32

import os

import pandas as pd
import geopandas as gpd


def calculate_sub(city):
    subway_mci = pd.read_csv('./mci/subway_mci.csv')
    subway_mci.fillna(0, inplace=True)
    line_shapefile = f'./data/{city}/{city}.shp'
    station_shapefile = f'./data/{city}/{city}_station.shp'

    line_gdf = gpd.read_file(line_shapefile, encoding='utf-8')
    line_gdf.crs = "EPSG:4326"
    line_gdf = line_gdf.to_crs(crs='EPSG:3857')

    station_gdf = gpd.read_file(station_shapefile, encoding='utf-8')
    station_gdf.crs = 'EPSG:4326'
    station_gdf = station_gdf.to_crs(crs='EPSG:3857')

    # calculate the line ms
    line_gdf['length'] = line_gdf.length
    line_gdf['Cement'] = line_gdf['length'] * subway_mci[0:1]['Cement'][0] / 1000
    line_gdf['Steel'] = line_gdf['length'] * float(subway_mci[0:1]['Steel'][0]) / 1000
    line_gdf['Total'] = line_gdf['length'] * subway_mci[0:1]['Total'][0] / 1000

    line_gdf[['line', 'length', 'Cement', 'Steel', 'Total']].to_csv(f'./data/{city}/{city}_line_ms.csv', index=False)
    line_gdf.to_file(f'./data/{city}/{city}_line_ms.shp')

    # calculate the station ms
    count = len(station_gdf['line'])
    station_gdf['Cement'] = [subway_mci[1:2]['Cement'][1]] * count
    station_gdf['Steel'] = [subway_mci[1:2]['Steel'][1]] * count
    station_gdf['Gravel'] = [subway_mci[1:2]['Gravel'][1]] * count
    station_gdf['Sand'] = [subway_mci[1:2]['Sand'][1]] * count
    station_gdf['Asphalt'] = [subway_mci[1:2]['Asphalt'][1]] * count
    station_gdf['Copper'] = [subway_mci[1:2]['Copper'][1]] * count
    station_gdf['Total'] = [subway_mci[1:2]['Total'][1]] * count

    station_gdf.to_csv(f'./data/{city}/{city}_station_ms.csv', index=False)
    station_gdf.to_file(f'./data/{city}/{city}_station_ms.shp')


if __name__ == '__main__':
    p = [i for i in os.listdir('./data')]
    for c in p:
        print(c)
        calculate_sub(c)
