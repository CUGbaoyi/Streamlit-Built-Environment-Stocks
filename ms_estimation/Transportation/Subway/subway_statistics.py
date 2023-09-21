#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/21 10:02

import pandas as pd
import os


def get_result(city):
    print(city)
    line_tmp = {}
    line_df = pd.read_csv(f'./data/{city}/{city}_line_ms.csv')
    line_tmp['City'] = city.split('_line.csv')[0]
    line_tmp['Cement'] = line_df['Cement'].sum()
    line_tmp['Steel'] = line_df['Steel'].sum()
    line_tmp['Total'] = line_df['Total'].sum()
    line.append(line_tmp)

    station_tmp = {}
    station_df = pd.read_csv(f'./data/{city}/{city}_station_ms.csv')
    station_tmp['City'] = city.split('_station.csv')[0]
    station_tmp['Cement'] = station_df['Cement'].sum()
    station_tmp['Steel'] = station_df['Steel'].sum()
    station_tmp['Gravel'] = station_df['Gravel'].sum()
    station_tmp['Sand'] = station_df['Sand'].sum()
    station_tmp['Asphalt'] = station_df['Asphalt'].sum()
    station_tmp['Copper'] = station_df['Copper'].sum()
    station_tmp['Total'] = station_df['Total'].sum()
    station.append(station_tmp)


if __name__ == '__main__':
    city_list = [i for i in os.listdir('./data')]
    line = []
    station = []

    for c in city_list:
        get_result(c)

    line_out = pd.DataFrame(line)
    station_out = pd.DataFrame(station)

    line_out.to_csv('subway_line_statistics.csv', index=False)
    station_out.to_csv('subway_station_statistics.csv', index=False)
