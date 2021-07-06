#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: baoyi
# Datetime: 2021/6/8 14:58

import matplotlib
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import pandas_bokeh
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time

from utils import config
from bokeh.plotting import figure


def color_map_color(value, cmap_name=config.COLOR_MAP, vmin=0, vmax=1):
    """
    get the color rgb value though a color map and ms_value
    :param value: ms_value
    :param cmap_name: default value config.COLOR_MAP = 'RdYlBu_r'
    :param vmin: default value 0
    :param vmax: default value 1
    :return: rgb list
    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return [i * 255 for i in list(rgb)]


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


def data_view_3d(gdf: gpd.GeoDataFrame, material):
    """
    Show Map thought pydeck
    :param gdf:
    :param material:
    :return:
    """
    # select the grid which is not none
    gdf = gdf[gdf[material] > 0]
    # norm material and define the color
    gdf["ms_norm"] = (gdf[material] - np.min(gdf[material])) / (np.max(gdf[material]) - np.min(gdf[material]))
    gdf['color'] = gdf['ms_norm'].apply(lambda x: color_map_color(x))

    # get the init view location
    center_latitude = np.mean(gdf.geometry.centroid.y)
    center_longitude = np.mean(gdf.geometry.centroid.x)

    # set the highest bar
    max_material_stock = max(gdf[material])
    high = 6000

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=7,
        max_zoom=18,
        pitch=45,
        bearing=0
    )

    # default layer setting
    INITIAL_LAYER = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation=f"ms_norm * {high}",
        elevation_scale=10,
        auto_highlight=True,
        elevation_range=[0, 3000],
        get_fill_color="color",
        get_line_color="color",
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[INITIAL_LAYER],
            initial_view_state=INITIAL_VIEW_STATE,
            api_keys={'mapbox': config.MAPBOX_API_KEY},
            map_provider='mapbox',
            map_style=config.MAP_STYLE
        )
    )


def data_view_2d(gdf: gpd.GeoDataFrame, material):
    """
    Show Map thought bokeh
    :param gdf:
    :param material:
    :return:
    """
    gdf = gdf[gdf[material] > 0]

    # get the init view location
    center_latitude = np.mean(gdf.geometry.centroid.y)
    center_longitude = np.mean(gdf.geometry.centroid.x)

    fig = figure()
    p = gdf.plot_bokeh(
            figure=fig,
            show_figure=True,
            xlim=[center_longitude - 0.5, center_longitude + 0.5],
            ylim=[center_latitude - 0.5, center_latitude + 0.5],
            category=material,
            #     colormap_uselog=True,
            colormap="RdYlBu",
            show_colorbar=True,
            line_color=None,
            tile_provider="CARTODBPOSITRON_RETINA"
        )

    st.bokeh_chart(p, use_container_width=True)


def data_view_2d_plotly(gdf: gpd.GeoDataFrame, material):
    """
    Show Map thought plotly
    :param gdf:
    :param city:
    :param material:
    :return:
    """
    gdf = gdf[gdf[material] > 0]

    # get the init view location
    center_latitude = np.mean(gdf.geometry.centroid.y)
    center_longitude = np.mean(gdf.geometry.centroid.x)

    fig = px.choropleth_mapbox(gdf,
                               geojson=gdf.geometry,
                               locations=gdf.index,
                               color=material,
                               hover_name=gdf.landuse,
                               color_continuous_scale="RdYlBu_r",
                               center={"lat": center_latitude, "lon": center_longitude},
                               # mapbox_style="dark",
                               mapbox_style="open-street-map",
                               height=1000,
                               opacity=0.8,
                               zoom=8.5)

    fig.update_traces(marker_line_width=0)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)


def app():
    # Get the city list
    city_list = [i for i in os.listdir('./data')]
    city_list.sort()

    # set page title
    st.title('Map of Material Stocks')

    # select list match gdf columns name
    material_map = {
        'Building Stocks': 'b_total',
        'Infrastructure Stocks': 'infrastructure',
        'Built-environment Stocks': 'ms_sum',
        'Building Asphalt': 'b_asphalt',
        'Building Brick': 'b_brick',
        'Building Cement': 'b_cement',
        'Building Ceramic': 'b_ceramic',
        'Building Glass': 'b_glass',
        'Building Gravel': 'b_gravel',
        'Building Lime': 'b_lime',
        'Building Sand': 'b_sand',
        'Building Steel': 'b_steel',
        'Building Timber': 'b_timber'
    }

    # get city name
    city = st.selectbox("Choose a city", city_list, index=1)
    # get material
    material_select = st.selectbox("Choose a sector or material", list(material_map.keys()), index=2)
    start_button = st.button("Start!")
    if start_button:
        # read shapefile
        gdf = read_shp_file(city)
        st.subheader(f'{material_select} Spatial Distribution of {city.title()}')
        with st.warning('It may take several seconds to process map data, please wait ... '):
            data_view_2d_plotly(gdf, material_map[material_select])

        st.markdown("#### Data Example")
        gdf_view = gdf.drop('geometry', axis=1)
        gdf_view = gdf_view[(gdf_view['b_total'] > 0) & (gdf_view['road'] > 0)]
        st.write(pd.DataFrame(gdf_view)[:20])
    else:
        pass
