# Patterns of urban built environment stocks across and within cities in China

## Introduction
Rapid urbanization and booming construction activities exert resource and environmental pressures for global sustainability, especially for cities in developing countries. By estimating urban built environment stocks for 50 selected Chinese cities with big data mining and analytics techniques, we found their stock quantities are positively correlated with urban socioeconomic. We developed a website (based on [Streamlit](https://streamlit.io/)) for visualizing the spatial distribution of built-environment stocks across Chinese cities and their relationship to socioeconomic status to better understand the results discussed in the article **Patterns of urban built environment stocks across and within cities in China**.

## Usage
### Prepare Data
First of all, you should download the [shapefile](https://github.com/CUGbaoyi/Streamlit-Built-Environment-Stocks) dataset generated in this article. The dataset should move to the root path of this project, with the following folder structure.

```
├── app.py
├── apps
│   ├── DataView.py
│   ├── home.py
│   └── SpatialView.py
├── data
│   ├── baoding
│   │   ├── baoding.cpg
│   │   ├── baoding.dbf
│   │   ├── baoding.prj
│   │   ├── baoding.shp
│   │   └── baoding.shx
│   ├── beijing
        ...

├── multiapp.py
├── README.md
├── requirements.txt
└── utils
    ├── config.py
    ├── __init__.py

```

### Install the requirements
```shell

pip install -r requirements.txt

```

### Run Streamlit
```shell

streamlit run app.py

```

### Preview

![Example of the website](https://i.loli.net/2021/06/28/FDfqdwACiKcXBQZ.png)

