FROM continuumio/miniconda3

RUN conda install numpy gdal fiona osmnx geopandas -c conda-forge
RUN pip install 'segment-geospatial' 'timm==1.0.9'
