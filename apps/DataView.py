#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2021/6/9 15:10

import numpy as np
import streamlit as st
import geopandas as gpd
import pandas as pd
import altair as alt
import os
import pwlf
import scipy.stats as stats
import pyecharts.options as opts
import plotly.figure_factory as ff
import plotly.express as px

from sklearn.metrics import r2_score


@st.cache(allow_output_mutation=True)
def read_shp_file(city):
    """
    read shapefile (use st.cache)
    :param city: The city you choose
    :return:
    """
    filepath = f"./data/{city}/{city}.shp"
    gdf = gpd.read_file(filename=filepath)
    gdf = gdf.fillna(0)
    # calculate the infrastructure ms
    gdf['infrastructure'] = gdf['ms_sum'] - gdf['b_total']

    return gdf


def building_material_category_pie(gdf: gpd.GeoDataFrame, city):
    """
    get the pie plot of building material
    :param gdf:
    :param city:
    :return:
    """
    material_list = [
        'Asphalt',
        'Brick',
        'Cement',
        'Ceramic',
        'Glass',
        'Gravel',
        'Lime',
        'Sand',
        'Steel',
        'Timber'
    ]

    material_value = [sum(gdf[f"b_{i.lower()}"]) for i in material_list]
    data_pair = [list(z) for z in zip(material_list, material_value)]
    data_pair.sort(key=lambda x: x[1])

    fig = px.pie(values=[i[1] for i in data_pair], names=[i[0] for i in data_pair])
    st.markdown(f"### The proportion of various materials in the building stocks of {city}")
    st.plotly_chart(fig, use_container_width=True)


def building_stocks_distribution(gdf: gpd.GeoDataFrame, city):
    """
    get the building stocks distribution (log normal)
    :param gdf:
    :param city:
    :return:
    """
    gdf = gdf[gdf['b_total'] > 0]
    result = gdf['b_total'].tolist()
    result = [i / 1000 for i in result]
    # st.write(sum(result))
    xdata = sorted(result)
    ydata = [i / len(xdata) for i in range(len(xdata), 0, -1)]
    ydata_log = [np.log(i / len(xdata)) for i in range(len(xdata), 0, -1)]
    # fitted line.
    z = np.polyfit(xdata, ydata_log, 1)
    f = np.poly1d(z)
    y_fit = [f(i) for i in xdata]
    r2 = round(r2_score(ydata_log, y_fit), 3)
    mu = z[1] / z[0]
    theta = 1 / -z[0]

    # draw ccdf and fitted line.
    df = pd.DataFrame({'x': xdata, 'y': np.log(ydata), 'y_fit': y_fit})
    st.subheader(f'The ccdf curves for building MS in {city}: (R-square={r2})')

    base = alt.Chart(df).encode(
        alt.X('x:Q', axis=alt.Axis(title='Building Stocks(t)'))
    )

    y_line = base.mark_point(size=10).encode(
        alt.Y('y:Q', axis=alt.Axis(orient='left', title='log(P(X > Building Stocks))'))
    )

    yfit_line = base.mark_line(color='red').encode(
        alt.Y('y_fit:Q', axis=alt.Axis(labels=True, title=''))
    )

    st.altair_chart(y_line + yfit_line, use_container_width=True)

    # add pdf formula.
    st.markdown(f"#### The pdf for building MS in {city}:")
    st.latex(r'''pdf(x ; \mu, \theta)=\frac{1}{\theta} e^{-\frac{x-\mu}{\theta}}''')
    st.latex(r'''\mu={0}, \theta={1}'''.format(round(mu, 3), round(theta, 3)))


def building_stocks_gini(gdf: gpd.GeoDataFrame, city):
    """
    calculate gini index and the Lorentz curve of average building stocks
    :param gdf:
    :param city:
    :return:
    """
    gdf = gdf[gdf['b_total'] > 0]
    result = gdf['b_total'].tolist()
    p = sorted(result)
    cum = np.cumsum(sorted(np.append(p, 0)))
    _sum = cum[-1]
    x = np.array(range(len(cum))) / len(p)
    y = cum / _sum

    # calculate the GINI index
    B = np.trapz(y, x=x)
    A = 0.5 - B
    G = A / (A + B)

    # draw Lorentz curve
    st.subheader(f"The Lorentz curves for building MS in {city}")
    st.markdown(f"#### GINI index is *{round(G, 4)}*")
    df = pd.DataFrame({'x': x, 'y': y})

    c1 = alt.Chart(df).mark_area(line={'color': '#000000'},).encode(
        alt.X('x:Q'),
        alt.Y('x:Q'),
        color=alt.value("#C5DBEA")
    )

    c2 = alt.Chart(df).mark_area(line={'color': '#000000'},).encode(
        alt.X('x:Q', axis=alt.Axis(title='Cumulative grid')),
        alt.Y('y:Q', axis=alt.Axis(orient='left', title='Cumulative building MS')),
        color=alt.value("#D7C9E0")
    )

    st.altair_chart(c1 + c2, use_container_width=True)


def building_stock_pop_rank_line(gdf: gpd.GeoDataFrame, city):
    """
    plot the relationship between stock and pop
    :param gdf:
    :param city:
    :return:
    """
    gdf = gdf[(gdf['b_total'] > 0) & (gdf['population'] > 0)]
    gdf = gdf[['id', 'b_total', 'population']]
    # sorted list
    stock = [i / 1000 for i in sorted(gdf['b_total'])]
    pop = sorted(gdf['population'])

    # hangzhou has an outlier
    if city == 'hangzhou':
        pop = pop[:-1]
        stock = stock[:-1]


    # use pwlf get two line, doc: https://jekel.me/piecewise_linear_fit_py/pwlf.html
    my_pwlf = pwlf.PiecewiseLinFit(pop, stock)
    # get two line break value
    break_value = my_pwlf.fit(2)[1]

    # get break value index
    index = 0
    for break_index, v in enumerate(pop):
        if v > break_value:
            index = break_index
            break
        else:
            pass

    pop_1 = pop[:index]
    pop_2 = pop[index:]
    stock_1 = stock[:index]
    stock_2 = stock[index:]

    line_1 = np.poly1d(np.polyfit(pop_1, stock_1, 1))
    line_2 = np.poly1d(np.polyfit(pop_2, stock_2, 1))
    line_3 = np.poly1d(np.polyfit(pop, stock, 1))

    k1 = line_1[1]
    k2 = line_2[1]
    k3 = line_3[1]

    pred_line_1 = [line_1(_) for _ in pop_1]
    pred_line_2 = [line_2(_) for _ in pop_2]
    pred_line_3 = [line_3(_) for _ in pop]

    line_1_r2 = r2_score(stock_1, pred_line_1)
    line_2_r2 = r2_score(stock_2, pred_line_2)
    line_3_r2 = r2_score(stock, pred_line_3)

    category = ['Low'] * index + ["High"] * len(pop_2)
    df = pd.DataFrame({
        'pop': pop,
        'stock': stock,
        'category': category,
        'pred': pred_line_1 + pred_line_2
    })

    base = alt.Chart(df).encode(
        alt.X('pop:Q', axis=alt.Axis(title='Pop'))
    )

    y_point = base.mark_point(size=20).encode(
        alt.Y('stock:Q', axis=alt.Axis(orient='left', title='Building Stocks(t)')),
        color='category:N',
        shape=alt.Shape('category'),
    ).interactive()

    y_line = base.mark_line(strokeDash=[10, 20], strokeWidth=3).encode(
        alt.Y('pred:Q', axis=alt.Axis(orient='left', title='Building Stocks(t)')),
        alt.Color('category:N', scale=alt.Scale(scheme='dark2')),
    )

    # get break line
    break_line = alt.Chart(pd.DataFrame({'x': [break_value], 'text': f"Pop = {int(break_value)}"})).mark_rule(strokeDash=[10, 10]).encode(
        x='x',
        color=alt.value('gray'),
        size=alt.value(2)
    )

    pop_text = break_line.mark_text(
        align='left',
        baseline='middle',
        dx=10,
    ).encode(
        text='text:N',
        size=alt.value(15)
    )

    c = alt.layer(
            y_point, y_line, break_line, pop_text
        ).resolve_scale(
            color='independent',
            shape='independent'
        )

    st.markdown(f"### The correlations between the arranged population and building MS at grid level in {city}")
    st.altair_chart(c, use_container_width=True)

    st.latex(f'R_{{Low}}^2 = {round(line_1_r2, 3)}, K_{{Low}} = {int(k1)}')
    st.latex(f'R_{{High}}^2 = {round(line_2_r2, 3)}, K_{{High}} = {int(k2)}')
    st.latex(f'R_{{All}}^2 = {round(line_3_r2, 3)}, K_{{All}} = {int(k3)}')


def b2r_distribution(gdf: gpd.GeoDataFrame, city):
    """
    get the lognormal distribution of building to road ratio
    :param gdf:
    :param city:
    :return:
    """
    gdf = gdf[(gdf['b_total'] > 0) & (gdf['road'] > 0)]
    gdf['ratio'] = gdf['b_total'] / gdf['road']
    log_ratio = np.log(gdf['ratio'].tolist())

    # get mu and sigma
    mu = np.mean(log_ratio)
    sigma = np.std(log_ratio)
    skew = stats.skew(log_ratio)
    kurtosis = stats.kurtosis(log_ratio)

    # get normal fitting data according to mu and sigma
    s = np.random.normal(mu, sigma, 10000)

    group_labels = ['Normal Fit', 'BtR ratio', ]
    colors = ['#F66095', '#2BCDC1', ]

    fig = ff.create_distplot(
        [s, log_ratio],
        group_labels,
        colors=colors,
        bin_size=.1,
        # curve_type='normal',
        show_curve=True
    )

    fig.update_layout(xaxis_title='log(BtR ratio)', yaxis_title='p')

    st.markdown(f"### The pdf curves for BtR ratio in {city}")
    st.plotly_chart(fig, use_container_width=True)
    st.latex(r"pdf(ratio)=\frac{1}{x \sigma \sqrt{2 \pi}} \exp \left(-\frac{(\ln x-\mu)^{2}}{2 \sigma^{2}}\right)")
    st.latex(r'''\mu={0}, \sigma={1}, Skewness={2}, Kurtosis={3}'''.format(
        round(float(mu), 3),
        round(float(sigma), 3),
        round(float(skew), 3),
        round(float(kurtosis), 3)
    ))


def app():
    # Get the city list
    city_list = [i for i in os.listdir('./data')]
    city_list.sort()

    # set page title
    st.title('Material Stocks Data Explore')

    # get city name
    city = st.selectbox("Choose a city", city_list, index=1)
    # start_button = st.button("Start!")
    # if start_button:

    gdf = read_shp_file(city)
    # st.write(gdf.head())
    building_material_category_pie(gdf, city)
    building_stocks_distribution(gdf, city)
    building_stocks_gini(gdf, city)
    building_stock_pop_rank_line(gdf, city)
    b2r_distribution(gdf, city)
