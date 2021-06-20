#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2021/6/12 15:05

import os
import geopandas as gpd

city_list = os.listdir('data')

for city in city_list:
    print(city)
    gdf = gpd.read_file(f"./data/{city}/{city}.shp")
    gdf = gdf.to_crs("epsg:4326")
    gdf.to_file(f"./data/{city}/{city}.shp")
