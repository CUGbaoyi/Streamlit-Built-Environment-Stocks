# Patterns of urban built environment stocks across and within cities in China

## Introduction
Rapid urbanization and booming construction activities exert resource and environmental pressures for global sustainability, especially for cities in developing countries. By estimating urban built environment stocks for 50 selected Chinese cities with big data mining and analytics techniques, we found their stock quantities are positively correlated with urban socioeconomic. We developed a website (based on [Streamlit](https://streamlit.io/)) for visualizing the spatial distribution of built-environment stocks across Chinese cities and their relationship to socioeconomic status to better understand the results discussed in the article **Patterns of urban built environment stocks across and within cities in China**. The example website: http://www.cityms.info

## MS estimation
This repository contains code and data for estimating material stock (MS) related to various infrastructure elements, such as buildings, transportation networks, and more. Below is a detailed overview of the repository structure and the purpose of each directory and file.

```
ms_estimation/
├── Building/
│   ├── data/                 # Raw data for buildings, organized by city
│   │   ├── baoding/
│   │   ├── beijing/
│   │   ├── ... (and other cities)
│   ├── ground truth/        # Training sample data
│   │   ├── beijing/
│   │   ├── guangzhou/
│   │   ├── shenzhen/
│   ├── material/            # Predicted results for various materials
│   │   ├── Asphalt(kg)/
│   │   ├── Brick(kg)/
│   │   ├── ... (and other materials)
│   ├── material_statistics.py   # Python script for material statistics calculation
│   ├── model comparision statistics/  # Statistics for model comparison, both visual and textual
│   │   ├── Case1.png/
│   │   ├── Case2.png/
│   │   ├── ... (and other cases)
│   ├── model_comparison.py       # Python script for model comparison
│   ├── statistics.csv            # CSV file containing various statistics
│   └── urban_ms_predict.py       # Python script for urban material stock prediction
└── Transportation/
    ├── Railway/
    │   ├── data/                 # Raw data for railways
    │   ├── ... (other subdirectories and files for railway MS estimation)
    ├── Road/
    │   ├── data/                 # Raw data for roads
    │   ├── ... (other subdirectories and files for road MS estimation)
    └── Subway/
        ├── data/                 # Raw data for subways
        ├── ... (other subdirectories and files for subway MS estimation)
```

###
-data/: Contains the raw data for buildings and is organized by cities. For instance, data for buildings in Beijing can be found in the beijing/ subdirectory. Due to the storage limitation, we provide a sample city.

-ground truth/: Contains the training sample data for the MS estimation. This is where the ground truth or reference data resides for different cities.

-material/: This directory contains the predicted results for different materials, such as asphalt, brick, cement, and so on.

-Transportation/: Contains subdirectories related to different transportation infrastructures - Railway, Road, and Subway. Each of these subdirectories contains similar structured data, grids, and scripts related to that particular infrastructure's MS estimation.

## Website Usage
### Prepare Data
First of all, you should download the [geotiff](https://figshare.com/articles/journal_contribution/High-resolution_built_environment_stocks_within_cities_in_China/19387439) data and translate to shapefile (In this github repository, we provide some sample cities). The dataset should move to the root path of this project, with the following folder structure.

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

