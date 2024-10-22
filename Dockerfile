FROM continuumio/miniconda3

RUN conda install numpy gdal osmnx segment-geospatial 'timm==1.0.9' -c conda-forge
