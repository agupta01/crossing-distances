import modal
import os
from typing import Optional, Tuple
from collections import namedtuple

app = modal.App("crossing-distances")

image = (
    modal.Image.debian_slim(python_version="3.11")
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
    from samgeo import tms_to_geotiff, choose_device, geotiff_to_jpg

    logger = create_logger()
    
    logger.info(f"Executing function for coordinate ({lat}, {long})")

    new_coords = coords_from_distance(lat, long, dist=25.0, heading=180)
    logger.warning(f"New Coords: ({new_coords.lat}, {new_coords.long})")

@app.local_entrypoint()
def main(mode: str):
    coords = [Coordinate(lat=0.0, long=0.0), Coordinate(lat=1.0, long=-1.0)]
    if mode.lower() == "remote":
        get_image_for_crosswalk.for_each(*zip(*coords))
        # get_image_for_crosswalk.remote(0.0, 0.0)
    else:
        list(map(get_image_for_crosswalk.local, *zip(*coords)))
