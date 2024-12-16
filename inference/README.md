# Inference Pipeline

## Sources
- OpenStreetMap - No API key required
- Google Maps Tile Server - No API key required

## Environment Variables
These define key hyperparameters that will be used in the inference process.
- CITY_NAME: full city name in analysis
- CITY_CODE: city code (i.e. sf for San Francisco, sna for Irvine) that will be used to look up data artifacts and connect to the correct Modal environment

## Tasks
The inference pipeline is broken down into a series of tasks, each encapsulated into a notebook.
The PCD Inference Process diagram in this folder describes the structure of these tasks (1 color = 1 task)
The `entrypoint.py` file is used to orchestrate these tasks using Modal and papermill to run on data given a city.

The tasks are as follows:
- `osm_ingest.ipynb` (Orange + Blue): Pulls raw data from OpenStreetMap for the city and modifies the OSM results to produce a list of crosswalks that need to be analyzed
  - Inputs: None
  - Outputs: crosswalk_edges.shp (shapefile of lines), intersection_coordinates.csv (index, x, y points for all intersections with crosswalks in study area), raw_intersection_coordinates.csv (index, x, y points for all intersections in study area)
  - Example: `MODAL_ENVIRONMENT=sfo uv run modal run inference/osm_ingest.py::osm_ingest --place "San Francisco, CA, USA"`
- `crosswalk_image_ingest.py` (Purple): pulls JPEGs of crosswalk images into Modal volume for city
  - Inputs: coordinates.csv
  - Outputs: None (images committed to modal volume `crosswalk-data-{CITY_CODE}` in `CITY_CODE` environment)
- `sam_inference.ipynb` (Red + Yellow): produces masks of each image in the Modal volume by running inference on fine-tuned SAM
  - Inputs: None (pulls SAM weights from Modal)
  - Outputs: crosswalk_masks.shp (contains polygons of each drivable area mask in study area)
- `grow_cut.ipynb` (Green): runs the grow-cut algorithm on the polygons to fix crosswalks
  - Inputs: cross_walks.geojson, crosswalk_edges.shp
  - Outputs: refined_crosswalks.geojson
  - Example: `MODAL_ENVIRONMENT=sfo uv run modal run inference/grow_cut.py::grow_cut`

Generally, all data artifacts will go in a city's `scratch` volume.
