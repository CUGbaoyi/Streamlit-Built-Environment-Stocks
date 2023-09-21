#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2020/4/20 16:59

"""
Statistical results of various materials
"""

import pandas as pd
import os
from collections import defaultdict


class MaterialSta:
    def __init__(self):
        self.path = f'./material/'
        self.filter_name = {'Total MS(t)_statistics.txt', '.DS_Store'}
        self.material_list = [
            'Cement(kg)',
            'Steel(kg)',
            'Timber(kg)',
            'Brick(kg)',
            'Gravel(kg)',
            'Sand(kg)',
            'Asphalt(kg)',
            'Lime(kg)',
            'Glass(kg)',
            'Ceramic(kg)',
            'Total MS(t)'
        ]
        self.dic = defaultdict(list)
        self.city_list = [city for city in os.listdir(self.path + self.material_list[-1]) if
                          city not in self.filter_name]

    def get_data(self):
        for city in self.city_list:
            self.dic['city'].append(city)
            for material in self.material_list:
                path = self.path + material + '/' + city + '/' + city + '_result.csv'
                df = pd.read_csv(path)
                self.dic[material].append(sum(df['result']))

        material_df = pd.DataFrame(self.dic)
        material_df['material_sum'] = material_df[
            ['Cement(kg)', 'Steel(kg)', 'Timber(kg)', 'Brick(kg)', 'Gravel(kg)', 'Sand(kg)', 'Asphalt(kg)', 'Lime(kg)',
             'Glass(kg)', 'Ceramic(kg)']].apply(lambda x: x.sum(), axis=1)
        material_df.to_csv(f'statistics.csv', index=False)


if __name__ == '__main__':
    materialSta = MaterialSta()
    materialSta.get_data()
