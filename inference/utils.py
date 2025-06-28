import logging
import math
import os
from collections import namedtuple
from io import BytesIO
import pandas as pd
from typing import Literal
import geopandas as gpd
from pyproj import CRS

import modal

APP_NAME = "crossing-distances"
app = modal.App(APP_NAME)

PRECISION = 6  # decimal points = 111mm resolution
RADIUS = 25.0  # meters. Default size of an intersection
EPSILON = 1e-6

Coordinate = namedtuple("Coordinate", ["lat", "long"])

trunc_explanation = (
    "Values must be truncated so that decoding returns the original value."
)

osmnx_image = (
    modal.Image.from_registry("gboeing/osmnx:latest")
    .pip_install("geopy", "semver>=3.0.4")
    .env({"TINI_SUBREAPER": "1"})
)


sam_image = (
    modal.Image.from_registry("gboeing/osmnx:latest")
    .pip_install("segment-geospatial", "timm==1.0.9")
    .env({"TINI_SUBREAPER": "1"})
)

committer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pillow", "pandas")
    .env({"TINI_SUBREAPER": "1"})
)

dataset_volume = modal.Volume.from_name(
    f"crosswalk-data-{os.environ['MODAL_ENVIRONMENT']}", create_if_missing=True
)

redrive_dict = modal.Dict.from_name(
    f"crosswalk-data-{os.environ['MODAL_ENVIRONMENT']}-redrive", create_if_missing=True
)

main_scratch = modal.Volume.from_name(
    "scratch", environment_name="main", create_if_missing=True
)


def coords_from_distance(
    lat: float,
    long: float,
    dist: float,
    heading: float,
) -> Coordinate:
    """
    Return the lat/long coordinates after traveling a certain distance from
    some original coordinates at a specified compass heading.
    """
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
        math.sin(lat_rad) * math.cos(angular_distance)
        + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(heading_rad)
    )

    # Calculate new longitude
    new_long_rad = long_rad + math.atan2(
        math.sin(heading_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    # Convert new latitude and longitude back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_long = math.degrees(new_long_rad)

    return Coordinate(lat=new_lat, long=new_long)


def create_logger():
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(module)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )
    handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    raw_log_level = os.environ.get("LOG_LEVEL", "INFO").strip()
    try:
        log_level = int(raw_log_level)
    except ValueError:
        log_level = raw_log_level.upper()  # fall back to named level
    logger.setLevel(log_level)

    return logger


def get_crosswalk_id(lat: float, long: float) -> str:
    # ensure truncated lat/long first to PRECISION decimal points
    assert lat == round(
        lat, PRECISION
    ), f"Latitude not truncated to {PRECISION} decimal points. {trunc_explanation}"
    assert long == round(
        long, PRECISION
    ), f"Longitude not truncated to {PRECISION} decimal points. {trunc_explanation}"

    return (
        str(int(lat * (10**PRECISION) * (1 if lat > 0 else -1)))
        + ("N" if lat > 0 else "S")
        + "_"
        + str(int(long * (10**PRECISION) * (1 if long > 0 else -1)))
        + ("E" if long > 0 else "W")
    )


def decode_crosswalk_id(crosswalk_id: str) -> Coordinate:
    raw_lat, raw_long = tuple(crosswalk_id.split("_"))
    n_s_hemisphere = 1 if raw_lat.endswith("N") else -1
    e_w_hemisphere = 1 if raw_long.endswith("E") else -1
    return Coordinate(
        lat=n_s_hemisphere * int(raw_lat[:-1]) / (10**PRECISION),
        long=e_w_hemisphere * int(raw_long[:-1]) / (10**PRECISION),
    )


def bounding_box_from_filename(filename: str) -> tuple[float, float, float, float]:
    """Returns a bounding box from the file name, assuming a 25 meter radius around the center.

    Returns as tuple (
            bottom_right.long,
            bottom_right.lat,
            top_left.long,
            top_left.lat,
        )
    """
    center = decode_crosswalk_id(filename.split(".")[0].split("_", 1)[1])

    diag_radius = math.sqrt(2) * RADIUS
    top_left = coords_from_distance(center.lat, center.long, diag_radius, 315)
    bottom_right = coords_from_distance(center.lat, center.long, diag_radius, 135)

    return (
        bottom_right.long,
        bottom_right.lat,
        top_left.long,
        top_left.lat,
    )


def get_from_volume(
    volume_name: str, file_path: str, environment_name: str | None = None
) -> BytesIO:
    volume = modal.Volume.from_name(volume_name, environment_name=environment_name)
    data = b""
    for chunk in volume.read_file(file_path):
        data += chunk
    return BytesIO(data)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def filter_coordinates(
    df, city_code, sample, logger, return_length=False, return_df=False
) -> pd.DataFrame:
    # Sample the dataframe and round coordinates
    sampled_df = df.sample(n=sample if sample > 0 else len(df))
    sampled_df["coords"] = sampled_df.apply(
        lambda row: (round(row.y, PRECISION), round(row.x, PRECISION)), axis=1
    )

    # Generate filenames for each coordinate
    sampled_df["filename"] = sampled_df["coords"].apply(
        lambda coord: f"crosswalk_{get_crosswalk_id(coord[0], coord[1])}.jpeg"
    )

    # Get the volume
    dataset_volume = modal.Volume.from_name(
        f"crosswalk-data-{city_code}",
        environment_name=city_code,
    )

    # Get existing files in volume
    existing_files = set(
        entry.path.split("/")[-1] for entry in dataset_volume.iterdir("/")
    )

    # Filter out coordinates whose files already exist
    filtered_df = sampled_df[~sampled_df["filename"].isin(existing_files)]

    # Get the final set of coordinates
    filtered_coords = set(filtered_df["coords"].tolist())

    logger.info(
        f"Found {len(sampled_df) - len(filtered_coords)}, need {len(filtered_coords)} coordinates"
    )

    if return_length:
        return len(filtered_coords)
    elif return_df:
        return filtered_df
    return filtered_coords


def fuzzy_search_optimized(
    query: str, string_set: set[str], string_list: list[str]
) -> str:
    """
    Performs a fuzzy search to find the string with most matching characters in same positions.

    Args:
        query: The search string
        string_list: List of strings to search through (all same length as query)

    Returns:
        Best matching string from the list
    """
    if query in string_set:
        return query

    query_length = len(query)
    max_matches = 0
    best_match = string_list[0]

    # Pre-compute query bytes
    query_bytes = query.encode()

    # Pre-compute encoded strings
    encoded_strings = [s.encode() for s in string_list]

    for candidate_bytes in encoded_strings:
        # Use byte-level comparison
        matches = sum(a == b for a, b in zip(query_bytes, candidate_bytes))

        if matches > max_matches:
            max_matches = matches
            # Find original string
            best_match = string_list[encoded_strings.index(candidate_bytes)]

            if matches == query_length:
                break

    return best_match


def get_equidistant_crs(
    gdf: gpd.GeoDataFrame,
    units: Literal["Foot", "Meter"] = "Meter",
) -> CRS:
    """
    Creates a CRS that is azimuthally equidistant with respect to the center of the
    geometries in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame with geometries to create a CRS for
        units: Units to use for the CRS. Defaults to "Meter"

    Returns:
        CRS: CRS object
    """
    # Get the center of the geometries
    bounding_box = gdf.to_crs("EPSG:4326").total_bounds

    central_meridian = (bounding_box[0] + bounding_box[2]) / 2
    latitude_of_origin = (bounding_box[1] + bounding_box[3]) / 2

    _WKT = f"""
    PROJCS["ProjWiz_Custom_Azimuthal_Equidistant",
    GEOGCS["GCS_WGS_1984",
    DATUM["D_WGS_1984",
    SPHEROID["WGS_1984",6378137.0,298.257223563]],
    PRIMEM["Greenwich",0.0],
    UNIT["Degree",0.0174532925199433]],
    PROJECTION["Azimuthal_Equidistant"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",{central_meridian}],
    PARAMETER["Latitude_Of_Origin",{latitude_of_origin}],
    UNIT["{units}",1.0]]
    """

    # Create a CRS that is azimuthally equidistant with respect to the center
    return CRS.from_wkt(_WKT)


def set_equidistant_crs(
    gdf: gpd.GeoDataFrame,
    units: Literal["Foot", "Meter"] = "Meter",
    return_crs: bool = False,
):
    """
    Sets the CRS of a GeoDataFrame to an azimuthally equidistant CRS with respect to the
    center of the geometries in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame to set the CRS of
        units: Units to use for the CRS. Defaults to "Foot"
        return_crs: Whether to return the CRS object. Defaults to False

    Returns:
        GeoDataFrame: GeoDataFrame with the CRS set
        CRS: CRS object if return_crs is True
    """
    crs = get_equidistant_crs(gdf, units)
    if return_crs:
        return gdf.to_crs(crs), crs
    return gdf.to_crs(crs)
