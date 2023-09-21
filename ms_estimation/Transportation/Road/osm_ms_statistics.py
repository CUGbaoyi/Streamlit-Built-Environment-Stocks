#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/4 20:54

import pandas as pd
import os

city_list = [i for i in os.listdir('data')]
result = []


def get_result(city):
    print(city)
    tmp = {}
    df = pd.read_csv(f'./data/{city}/{city}_ms.csv')
    tmp['City'] = city.title()
    tmp['Cement'] = df['Cement'].sum()
    tmp['Steel'] = df['Steel'].sum()
    tmp['Gravel'] = df['Gravel'].sum()
    tmp['Sand'] = df['Sand'].sum()
    tmp['Asphalt'] = df['Asphalt'].sum()
    tmp['Mineral'] = df['Mineral Powder'].sum()
    tmp['Total'] = df['Total'].sum()

    result.append(tmp)


if __name__ == '__main__':
    for c in city_list:
        get_result(c)

    df_out = pd.DataFrame(result)
    df_out.to_csv('osm_statistics.csv', index=False)
