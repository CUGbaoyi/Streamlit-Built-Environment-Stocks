#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/20 20:38

import os

import pandas as pd
import geopandas as gpd


def calculate_rail(city):
    """
    calculate the ms of railway in city
    :param city:
    :return:
    """
    rail_mci = pd.read_csv('./mci/railway_mci.csv')
    line_shapefile = f'./data/{city}/{city}.shp'

    line_gdf = gpd.read_file(line_shapefile, encoding='utf-8')
    line_gdf = line_gdf.to_crs(crs='EPSG:3857')

    line_gdf['length'] = line_gdf.length
    line_gdf['Cement'] = line_gdf['length'] * rail_mci[0:1]['Cement'][0] / 1000
    line_gdf['Steel'] = line_gdf['length'] * rail_mci[0:1]['Steel'][0] / 1000
    line_gdf['Gravel'] = line_gdf['length'] * rail_mci[0:1]['Gravel'][0] / 1000
    line_gdf['Sand'] = line_gdf['length'] * rail_mci[0:1]['Sand'][0] / 1000
    line_gdf['Lime'] = line_gdf['length'] * rail_mci[0:1]['Lime'][0] / 1000
    line_gdf['Fly ash'] = line_gdf['length'] * rail_mci[0:1]['Fly ash'][0] / 1000
    line_gdf['Copper'] = line_gdf['length'] * rail_mci[0:1]['Copper'][0] / 1000
    line_gdf['Aluminum'] = line_gdf['length'] * rail_mci[0:1]['Aluminum'][0] / 1000
    line_gdf['Total'] = line_gdf['length'] * rail_mci[0:1]['Total'][0] / 1000

    # save to road ms to shapefile named city_ms.shp
    line_gdf = line_gdf[['gml_id', 'geometry', 'length', 'Cement', 'Steel', 'Gravel', 'Sand', 'Lime', 'Fly ash', 'Copper', 'Aluminum', 'Total']]
    line_gdf.to_file(f'./data/{city}/{city}_ms.shp')

    line_gdf.drop('geometry', axis=1, inplace=True)
    line_gdf.to_csv(f'./data/{city}/{city}_ms.csv', index=False)


if __name__ == '__main__':
    p = [i for i in os.listdir('./data')]
    for c in p:
        print(c)
        calculate_rail(c)
