import modal
import os
from typing import Optional, Tuple
from collections import namedtuple

app = modal.App("crossing-distances")

image = (
    modal.Image.from_registry("gboeing/osmnx:latest")
    .pip_install("segment-geospatial", "timm==1.0.9")
)

dataset_volume = modal.Volume.from_name("crosswalk-data", create_if_missing=True)

Coordinate = namedtuple("Coordinate", ["lat", "long"])

def coords_from_distance(lat: float, long: float, dist: float, heading: float) -> Coordinate:
    """
    Return the lat/long coordinates after traveling a certain distance from
    some original coordinates at a specified compass heading.
    """
    import math
    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    long_rad = math.radians(long)

    # Convert heading to radians
    heading_rad = math.radians(heading)

    # Earth's radius in meters
    earth_radius = 6371000

    # Calculate angular distance
    angular_distance = dist / earth_radius

    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(heading_rad)
    )

    # Calculate new longitude
    new_long_rad = long_rad + math.atan2(
        math.sin(heading_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )

    # Convert new latitude and longitude back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_long = math.degrees(new_long_rad)

    return Coordinate(lat=new_lat, long=new_long)

def create_logger():
    import logging

    logger = logging.getLogger(__name__)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(module)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )
    handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


@app.function(image=image, volumes={"/data": dataset_volume})
def get_image_for_crosswalk(lat: float, long: float):
    from datetime import datetime
    from uuid import uuid4
    import math
    from samgeo import tms_to_geotiff
    logger = create_logger()
    logger.info(f"Executing function for coordinate ({lat}, {long})")

    radius = 25.0 # meters

    crosswalk_id = str(uuid4())
    filename = os.path.join("/", "data", f"crosswalk_{crosswalk_id}.tif")

    # Build bounding box based on radius
    diag_radius = math.sqrt(2) * radius
    top_left = coords_from_distance(lat, long, diag_radius, 315)
    bottom_right = coords_from_distance(lat, long, diag_radius, 135)
    bounding_box = [bottom_right.long, bottom_right.lat, top_left.long, top_left.lat]

    tms_to_geotiff(
        output=filename, 
        bbox=bounding_box, 
        crs="EPSG:3857", 
        zoom=22, 
        source="Satellite", 
        overwrite=True, 
        quiet=True
    )
    dataset_volume.commit()

    return crosswalk_id


@app.local_entrypoint()
def main(mode: str, sample: int, input: str):
    import os
    import pandas as pd
    coords = pd.read_csv(input).sample(n=sample).apply(lambda c: (c.y, c.x), axis=1).tolist()
    if mode.lower() == "remote":
        ids = get_image_for_crosswalk.map(*zip(*coords), return_exceptions=True)
        # get_image_for_crosswalk.remote(0.0, 0.0)
    else:
        ids = list(map(get_image_for_crosswalk.local, *zip(*coords)))

    # dump IDs to json
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(index=ids, data=coords, columns=["lat", "long"]).to_csv(f"outputs/id_to_coords_sf.csv")

