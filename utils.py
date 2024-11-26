import logging
import math
from collections import namedtuple

Coordinate = namedtuple("Coordinate", ["lat", "long"])
PRECISION = 6  # decimal points = 111mm resolution
trunc_explanation = (
    "Values must be truncated so that decoding returns the original value."
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
    logger.setLevel(logging.INFO)

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
