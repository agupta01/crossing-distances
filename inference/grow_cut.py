import os
from concurrent.futures.thread import ThreadPoolExecutor

import modal
import semver
from shapely.geometry import LineString, MultiLineString, Polygon
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from inference.utils import EPSILON, app, create_logger, osmnx_image

GROW_RATE = 30
RETRIES = 15
TOLERANCE = 0.5  # metres – how far extended line may protrude beyond polygon before we stop growing

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
    """Grow `line` by `buffer` proportion of its *original* length on the requested `sides`.

    When `sides == (True, True)` both endpoints extend outwards by `buffer*length`.
    When only one boolean is *True* we still extend that *one* endpoint by the full
    `buffer*length` instead of half – the previous implementation extended by only
    half, making growth stall prematurely. The function is idempotent when
    ``buffer == 0``.
    """
    import math
    from shapely.geometry import LineString

    # Vector components and length
    (x1, y1), (x2, y2) = line.coords[0], line.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) + EPSILON  # avoid division-by-zero

    # Unit direction of the line
    ux, uy = dx / length, dy / length

    # Offset we have to add per growing side (absolute metres)
    grow_dist = buffer * length
    off_x, off_y = ux * grow_dist, uy * grow_dist

    # Conditionally move endpoints
    new_x1 = x1 - off_x if sides[0] else x1
    new_y1 = y1 - off_y if sides[0] else y1
    new_x2 = x2 + off_x if sides[1] else x2
    new_y2 = y2 + off_y if sides[1] else y2

    return LineString([(new_x1, new_y1), (new_x2, new_y2)])


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
    """Return a *finished* flag for each side.

    An endpoint is considered *finished* (i.e. we have grown far enough) if the
    closest distance between that original endpoint and *either* endpoint of the
    span that still lies inside the polygon is greater than ``TOLERANCE``.

    Using the *closest* span endpoint makes this test orientation-agnostic and
    fixes cases where the two lines have opposite direction, which previously
    caused both sides to be marked finished after a single iteration.
    """
    from shapely.geometry import Point

    span_pts = [Point(c) for c in span_line.coords]
    orig_pts = [Point(c) for c in original_line.coords]

    # Compute minimal distance from each original endpoint to *any* span endpoint
    dists = [min(p_orig.distance(p_span) for p_span in span_pts) for p_orig in orig_pts]

    left_finished = dists[0] > EPSILON
    right_finished = dists[1] > EPSILON

    logger.log(
        0,
        f"Span distances -> left: {dists[0]:.2f} m, right: {dists[1]:.2f} m | finished: {(left_finished, right_finished)}",
    )
    return left_finished, right_finished


def _split_line_by_heading(
    line: LineString, angle_tol_deg: float = 10
) -> list[LineString]:
    """
    Split a LineString into segments grouped by compass heading tolerance.
    """
    import math

    coords = list(line.coords)
    if len(coords) < 2:
        return []

    def heading(p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        theta = math.degrees(math.atan2(dy, dx))
        if theta < 0:
            theta += 360
        return theta

    def angular_diff(a, b):
        diff = abs(a - b) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    headings = [heading(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    groups = []
    group_start = 0
    ref_heading = headings[0]
    for idx in range(1, len(headings)):
        h = headings[idx]
        if angular_diff(h, ref_heading) <= angle_tol_deg:
            continue
        groups.append(LineString(coords[group_start : idx + 1]))
        group_start = idx
        ref_heading = h
    # last group
    final_coords = coords[group_start:]
    if len(final_coords) >= 2:
        groups.append(LineString(final_coords))
    return groups


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
        logger.log(
            0,
            f"Try count: {try_count}, left_finished={left_finished}, right_finished={right_finished}",
        )
        extended_line = _grow(
            extended_line, buffer, sides=(not left_finished, not right_finished)
        )
        cut_lines = _cut(
            polygon, extended_line, limit
        )  # list of LineStrings after cut operation
        if all(map(lambda x: x.length < 1, cut_lines)):
            logger.log(
                0,
                f"All lines less than 1m long. Lengths: {list(map(lambda x: x.length, cut_lines))}",
            )
            left_finished, right_finished = True, True
        else:
            left_finished, right_finished = _is_span_too_long(
                get_spanning_line(MultiLineString(cut_lines)), extended_line
            )
            # Safeguard: if only one endpoint has finished, halve remaining retries
            if left_finished != right_finished:
                remaining = retry_limit - try_count
                retry_limit = try_count + (remaining // 2)
                logger.log(
                    0, f"One endpoint finished; adjusted retry_limit to {retry_limit}"
                )

    # Safeguard: if retry limit reached and both endpoints unfinished, revert to original line and cut once
    if try_count >= retry_limit and not left_finished and not right_finished:
        logger.warning(
            "Retry limit reached without finishing growth. Reverting to original line and applying single cut."
        )
        return _cut(polygon, line, limit)
    return cut_lines


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
        logger.log(
            0, f"More than 2 coordinates in edge, splitting by heading tolerance"
        )
        crosswalk_edges = _split_line_by_heading(crosswalk_edge, angle_tol_deg=10)
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
    version_in: str = "2.0.0", version_out: str = f"2.1.3-grow{GROW_RATE}+{RETRIES}"
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
    import uuid

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
        masks_path = "/geofiles/train_10/geofiles"
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

    # Preprocess crosswalk geometries: normalize, drop duplicates, and assign unique UUIDs to null osmids
    crosswalks["geometry"] = crosswalks["geometry"].apply(lambda g: g.normalize())
    crosswalks = crosswalks.drop_duplicates(subset=["geometry"])
    null_mask = crosswalks["osmid"].isna()
    crosswalks.loc[null_mask, "osmid"] = [uuid.uuid4() for _ in range(null_mask.sum())]

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
    logger.debug(crosswalks_to_intersections.info())
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

        # Run compute_grow_cut.local on crosswalk_to_intersections using threadpoolexecutor
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
