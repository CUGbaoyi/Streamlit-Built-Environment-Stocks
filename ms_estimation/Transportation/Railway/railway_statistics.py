#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/5/21 10:41

import pandas as pd
import os


def get_result(city):
    print(city)
    line_tmp = {}
    line_df = pd.read_csv(f'./data/{city}/{city}_ms.csv')
    line_tmp['City'] = city.split('_line.csv')[0]
    line_tmp['Cement'] = line_df['Cement'].sum()
    line_tmp['Steel'] = line_df['Steel'].sum()
    line_tmp['Gravel'] = line_df['Gravel'].sum()
    line_tmp['Sand'] = line_df['Sand'].sum()
    line_tmp['Lime'] = line_df['Lime'].sum()
    line_tmp['Fly ash'] = line_df['Fly ash'].sum()
    line_tmp['Copper'] = line_df['Copper'].sum()
    line_tmp['Aluminum'] = line_df['Aluminum'].sum()
    line_tmp['Total'] = line_df['Total'].sum()
    line.append(line_tmp)


if __name__ == '__main__':
    city_list = [i for i in os.listdir('./data')]
    line = []

    for c in city_list:
        get_result(c)

    line_out = pd.DataFrame(line)
    line_out.to_csv('railway_statistics.csv', index=False)
