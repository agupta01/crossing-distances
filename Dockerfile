FROM continuumio/miniconda3

RUN conda install numpy gdal fiona osmnx geopandas -c conda-forge
