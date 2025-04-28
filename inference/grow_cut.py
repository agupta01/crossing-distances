import os
from concurrent.futures.thread import ThreadPoolExecutor

import modal
import semver
from shapely.geometry import LineString, MultiLineString, Polygon
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from inference.utils import EPSILON, app, create_logger, osmnx_image

GROW_RATE = 1000
RETRIES = 1

scratch_volume = modal.Volume.from_name("scratch", create_if_missing=True)
geofiles_volume = modal.Volume.from_name(
    f"crosswalk-data-{os.environ['MODAL_ENVIRONMENT'].lower() if os.environ['MODAL_ENVIRONMENT'].lower() != 'irv' else 'sna'}-results"
)

logger = create_logger()


def get_spanning_line(multiline):
    import numpy as np
    from shapely.geometry import LineString, MultiLineString

    if not isinstance(multiline, MultiLineString):
        raise ValueError("Input must be a shapely MultiLineString")

    # Extract all coordinates from the MultiLineString
    all_coords = [coord for line in multiline.geoms for coord in line.coords]

    # Convert to numpy array for easier calculations
    coords_array = np.array(all_coords)

    # Find the points with minimum and maximum x-coordinates
    try:
        min_x_idx = np.argmin(coords_array[:, 0])
        max_x_idx = np.argmax(coords_array[:, 0])
    except Exception as e:
        logger.error(f"Error finding min/max x-coordinates: {e}")
        logger.error(multiline)
        raise

    # Create the spanning LineString
    return LineString([tuple(coords_array[min_x_idx]), tuple(coords_array[max_x_idx])])


def _grow(line: LineString, buffer: float, sides: tuple[bool, bool]):
    """Helper function to run grow step of grow-cut."""
    import math

    from shapely.geometry import LineString

    # Extract line coordinates
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    dy = y2 - y1
    dx = x2 - x1
    length = math.sqrt(dx**2 + dy**2)

    # Calculate new length
    new_length = (1 + buffer) * length

    # Calulate components of new length
    new_dx = (dx * new_length) / (length + EPSILON)
    new_dy = (dy * new_length) / (length + EPSILON)

    # Calculate center coordinate
    xc = x1 + (dx / 2)
    yc = y1 + (dy / 2)

    # Calculate new coordinates
    if sides[0]:
        new_x1, new_y1 = xc - (new_dx / 2), yc - (new_dy / 2)
    else:
        new_x1, new_y1 = x1, y1
    if sides[1]:
        new_x2, new_y2 = xc + (new_dx / 2), yc + (new_dy / 2)
    else:
        new_x2, new_y2 = x2, y2

    # Construct new LineString
    extended_line = LineString([(new_x1, new_y1), (new_x2, new_y2)])
    return extended_line


def _cut(polygon: Polygon, extended_line: LineString, limit: int = 4):
    """Helper function to run cut step of grow-cut."""
    from shapely.geometry import LineString, MultiLineString

    # Compute intersection
    intersection = extended_line.intersection(polygon)

    spans = []

    if intersection.is_empty:
        logger.warning("empty intersection")
        return []
    elif isinstance(intersection, LineString):
        spans.append(intersection)
    elif isinstance(intersection, MultiLineString):
        if len(intersection.geoms) > limit:
            # logger.info(f"{len(intersection.geoms)} segments detected. Simplifying...")
            spans.append(get_spanning_line(intersection))
        else:
            for segment in intersection.geoms:
                # only keep segments that are > 1 meter in length
                if segment.length > 1:
                    spans.append(segment)
    else:
        raise ValueError("Unexpected intersection type.")
    return spans


def _is_span_too_long(
    span_line: LineString, original_line: LineString
) -> tuple[bool, bool]:
    """Helper to determine if the spanning line is too long, and if so, what side."""
    from shapely.geometry import Point

    span_line_1, span_line_2 = Point(span_line.coords[0]), Point(span_line.coords[-1])
    original_line_1, original_line_2 = (
        Point(original_line.coords[0]),
        Point(original_line.coords[-1]),
    )
    logger.debug(
        f"Points: {span_line_1}, {span_line_2}, {original_line_1}, {original_line_2}"
    )
    values = (
        span_line_1.distance(original_line_1) > 0.5,
        span_line_2.distance(original_line_2) > 0.5,
    )
    logger.debug(
        f"Span too long: {values}. Distances are: {span_line_1.distance(original_line_1)}, {span_line_2.distance(original_line_2)}"
    )
    return values


def get_line_spans_within_polygon(
    polygon: Polygon,
    line: LineString,
    buffer: float = float(GROW_RATE / 100),
    limit: int = 4,
) -> list:
    """
    Computes all segments of a line that lie within a polygon, accounting for non-convex shapes and holes.

    Parameters:
    - polygon (Polygon): The Shapely polygon, which can be non-convex and contain holes.
    - line (LineString): The Shapely line to intersect with the polygon. Guaranteed to be a straight line.
    - buffer (float): Proportion of original length to magnify crosswalk by during "grow" phase
    - limit (int): Maximum number of discrete crosswalks to split this length into. When the value
        exceeds this amount, fuse all segments back into a single one. Used to counter zebra crossing misclassification.

    Returns:
    - list of LineString: A list of LineStrings representing each segment of the line within the polygon.
    """
    from shapely.geometry import MultiLineString

    # New algorithm
    left_finished, right_finished = False, False
    extended_line = line
    retry_limit = RETRIES
    try_count = 0
    cut_lines = []
    while not (left_finished and right_finished) and try_count < retry_limit:
        try_count += 1
        logger.debug(
            f"Try count: {try_count}, left_finished={left_finished}, right_finished={right_finished}"
        )
        extended_line = _grow(
            extended_line, buffer, sides=(not left_finished, not right_finished)
        )
        cut_lines = _cut(
            polygon, extended_line, limit
        )  # list of LineStrings after cut operation
        if all(map(lambda x: x.length < 1, cut_lines)):
            logger.debug(
                f"All lines less than 1m long. Lengths: {list(map(lambda x: x.length, cut_lines))}"
            )
            left_finished, right_finished = True, True
        else:
            left_finished, right_finished = _is_span_too_long(
                get_spanning_line(MultiLineString(cut_lines)), extended_line
            )

    return cut_lines
    # Original algorithm
    # extended_line = _grow(line, buffer)
    # cut_lines = _cut(polygon, extended_line, limit)
    # return cut_lines


@app.function(image=osmnx_image)
def compute_grow_cut(row):
    # Extract the row
    id, row = row

    # Extract the crosswalk polygon and edge
    crosswalk_polygon = row["crosswalk_polygon"]
    crosswalk_edge = row["geometry"]

    # Split up "supercrosswalks" where multiple crossings are placed in the same MultiLineString
    if isinstance(crosswalk_edge, MultiLineString):
        logger.info("Crosswalk is multilinestring, splitting")
        crosswalk_edges = list(crosswalk_edge.geoms)
    elif len(crosswalk_edge.coords) > 2:
        logger.debug(
            f"More than 2 endpoints in edge, splitting: {list(crosswalk_edge.coords)}"
        )
        # split using sliding window on normalized edge
        crosswalk_edges = [
            LineString([a, b])
            for a, b in [
                crosswalk_edge.normalize().coords[i : i + 2]
                for i in range(len(crosswalk_edge.coords) - 1)
            ]
        ]
    else:
        crosswalk_edges = [crosswalk_edge]
    logger.log(
        5,
        f"{id}:\tGeometric intersection length {crosswalk_edge.intersection(crosswalk_polygon).length}",
    )
    logger.log(5, f"{id}:\tLength before correction: {crosswalk_edge.length}")
    # Compute the spans
    spans = []
    for _crosswalk_edge in crosswalk_edges:
        if (
            not crosswalk_edge.intersection(crosswalk_polygon).is_empty
            and _crosswalk_edge.length > 1
        ):
            spans.append(
                get_line_spans_within_polygon(crosswalk_polygon, _crosswalk_edge)
            )
        else:
            spans.append([_crosswalk_edge])

    # Flatten spans into a list
    final_spans = []
    for span in spans:
        final_spans.extend(span)

    logger.log(
        5, f"{id}:\tLength after correction: {sum(span.length for span in final_spans)}"
    )

    return final_spans


@app.function(
    image=osmnx_image,
    volumes={"/scratch": scratch_volume, "/geofiles": geofiles_volume},
    timeout=86400,
    cpu=4,
)
def grow_cut(
    version_in: str = "2.0.0", version_out: str = f"2.1.0-grow{GROW_RATE}+{RETRIES}"
):
    """Runs the grow-cut algorithm on the crosswalks to refine their boundaries.

    Arguments:
        version_in: The version of the crosswalk edges (SemVer). Defaults to 2.0.0.
        version_out: The output version to set for this run (SemVer). Defaults to 2.0.2.

    Inputs:
        cross_walks.geojson: GeoJSON file containing the crosswalk masks
        crosswalk_edges.shp: Shapefile containing the (raw) crosswalk edges

    Outputs:
        refined_crosswalks.geojson: GeoJSON file containing the refined crosswalks
    """
    import time

    import geopandas as gpd
    from shapely.ops import unary_union

    if modal.is_local():
        logger.info("Running locally")
        # Set local filepaths since local runs don't have access to modal volumes
        city_path = f"./data/{os.environ['MODAL_ENVIRONMENT']}_crossings"
        masks_path = city_path
        crosswalks_path = city_path
        output_path = city_path
    else:
        masks_path = "/geofiles/train_9/geofiles"
        crosswalks_path = "/scratch"
        output_path = crosswalks_path

    # Parse versions
    def _parse_semver(version):
        semver_version = semver.Version.parse(version)
        version_tag = f"v{semver_version.major}_{semver_version.minor}"
        if semver_version.patch != 0:
            version_tag += f"_{semver_version.patch}"
        if semver_version.prerelease:
            version_tag += f"-{semver_version.prerelease}"
        if semver_version.build:
            version_tag += f"+{semver_version.build}"
        return version_tag

    version_in, version_out = tuple(map(_parse_semver, (version_in, version_out)))

    masks = gpd.read_file(f"{masks_path}/cross_walks.geojson").to_crs("EPSG:3857")
    crosswalks = gpd.read_file(
        f"{crosswalks_path}/crosswalk_edges_{version_in}.shp"
    ).to_crs("EPSG:3857")

    logger.info(
        f"Found {len(crosswalks)} crosswalks to refine. Matching to intersections..."
    )
    start = time.time()
    crosswalks_to_intersections = (
        crosswalks[["osmid", "geometry"]]
        .sjoin(masks, how="left", predicate="intersects")
        .dropna(subset=["index_right"])
        .merge(
            masks.reset_index().rename(
                columns={"geometry": "crosswalk_polygon", "index": "index_right"}
            )[["index_right", "crosswalk_polygon"]],
            on="index_right",
        )
    )
    # Deduplicate on the index by unary unioning all crosswalk_polygons for that (left) index
    crosswalks_to_intersections = crosswalks_to_intersections.groupby("osmid").agg(
        {
            "crosswalk_polygon": unary_union,
            "geometry": lambda x: x.iloc[0],
        }
    )
    logger.info(
        f"Matched {len(crosswalks_to_intersections)} crosswalks to intersections. Time: {time.time() - start:.2f}s"
    )

    if modal.is_local():
        # sample_size = 100
        # logger.info(f"Running locally, so sampling to {sample_size} crosswalks")
        # crosswalks_to_intersections = crosswalks_to_intersections.sample(sample_size, random_state=77)

        # Run comput_grow_cut.local on crosswalk_to_intersections using threadpoolexecutor
        with ThreadPoolExecutor(max_workers=5) as executor, logging_redirect_tqdm():
            spans = list(
                executor.map(
                    compute_grow_cut.local,
                    tqdm(
                        crosswalks_to_intersections.iterrows(),
                        total=len(crosswalks_to_intersections),
                    ),
                )
            )
    else:
        spans = compute_grow_cut.map(
            crosswalks_to_intersections.iterrows(),
            return_exceptions=True,
        )

    # Flatten the spans, logging any errors
    final_spans = []
    for span in spans:
        if isinstance(span, Exception):
            logger.warning(f"Error found: {span}")
        elif isinstance(span, list):
            final_spans.extend(span)
        else:
            final_spans.append(span)

    # Save the refined crosswalks
    spans_gdf = gpd.GeoDataFrame(geometry=final_spans, crs="EPSG:3857").to_crs(
        "EPSG:4326"
    )
    spans_gdf.to_file(
        f"{output_path}/refined_crosswalks_{version_out}.geojson", driver="GeoJSON"
    )


if __name__ == "__main__":
    grow_cut.local()
