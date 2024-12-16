import modal
from inference.utils import app, osmnx_image, create_logger
from shapely.geometry import LineString, MultiLineString, Polygon

scratch_volume = modal.Volume.from_name("scratch", create_if_missing=True)

def get_spanning_line(multiline):
    from shapely.geometry import MultiLineString, LineString
    import numpy as np

    if not isinstance(multiline, MultiLineString):
        raise ValueError("Input must be a shapely MultiLineString")

    # Extract all coordinates from the MultiLineString
    all_coords = [coord for line in multiline.geoms for coord in line.coords]

    # Convert to numpy array for easier calculations
    coords_array = np.array(all_coords)

    # Find the points with minimum and maximum x-coordinates
    min_x_idx = np.argmin(coords_array[:, 0])
    max_x_idx = np.argmax(coords_array[:, 0])

    # Create the spanning LineString
    return LineString([tuple(coords_array[min_x_idx]), tuple(coords_array[max_x_idx])])

def get_line_spans_within_polygon(polygon: Polygon, line: LineString, buffer: float = 0.25, limit: int = 4) -> list:
    """
    Computes all segments of a line that lie within a polygon, accounting for non-convex shapes and holes.

    Parameters:
    - polygon (Polygon): The Shapely polygon, which can be non-convex and contain holes.
    - line (LineString): The Shapely line to intersect with the polygon.
    - buffer (float): Proportion of original length to magnify crosswalk by during "grow" phase
    - limit (int): Maximum number of discrete crosswalks to split this length into. When the value
        exceeds this amount, fuse all segments back into a single one. Used to counter zebra crossing misclassification.

    Returns:
    - list of LineString: A list of LineStrings representing each segment of the line within the polygon.
    """
    from shapely.geometry import LineString, MultiLineString
    import math

    logger = create_logger()

    # Extract polygon bounds
    minx, miny, maxx, maxy = polygon.bounds

    # Extract line coordinates
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]

    # Calculate direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        raise ValueError("The provided line is a single point.")

    # Normalize direction
    dx /= length
    dy /= length

    # Extend the line far beyond the polygon bounds
    extension_length = max(maxx - minx, maxy - miny) * buffer  # Arbitrary large extension

    new_start = (x1 - dx * extension_length, y1 - dy * extension_length)
    new_end = (x2 + dx * extension_length, y2 + dy * extension_length)

    extended_line = LineString([new_start, new_end])

    # Compute intersection
    intersection = extended_line.intersection(polygon)

    spans = []

    if intersection.is_empty:
        logger.warning("empty intersection")
        return spans
    elif isinstance(intersection, LineString):
        spans.append(intersection)
    elif isinstance(intersection, MultiLineString):
        if len(intersection.geoms) > limit:
            logger.info(f"{len(intersection.geoms)} segments detected. Simplifying...")
            spans.append(get_spanning_line(intersection))
        else:
            for segment in intersection.geoms:
                spans.append(segment)
    else:
        raise ValueError("Unexpected intersection type.")

    return spans

@app.function(
    image=osmnx_image,
    volumes={"/scratch": scratch_volume},
)
def grow_cut():
    """Runs the grow-cut algorithm on the crosswalks to refine their boundaries.

    Inputs:
        cross_walks.geojson: GeoJSON file containing the crosswalk masks
        crosswalk_edges.shp: Shapefile containing the (raw) crosswalk edges

    Outputs:
        refined_crosswalks.geojson: GeoJSON file containing the refined crosswalks
    """
    import geopandas as gpd

    logger = create_logger()

    masks = gpd.read_file("/scratch/cross_walks.geojson").to_crs("EPSG:3857")
    crosswalks = gpd.read_file("/scratch/crosswalk_edges.shp").to_crs("EPSG:3857")

    spans = []

    # TODO: map this if scalability becomes a concern
    for idx, row in crosswalks.iterrows():
        # Find the polygon which contains any part of the linestring
        match = masks.loc[masks.geometry.intersects(row.geometry)]
        if len(match):
            logger.info(f"found one! @ {idx}, {row['name']}")
            spans.extend(get_line_spans_within_polygon(match.iloc[0].geometry, row.geometry))

    # Save the refined crosswalks
    spans_gdf = gpd.GeoDataFrame(geometry=spans, crs="EPSG:3857").to_crs("EPSG:4326")
    spans_gdf.to_file("/scratch/refined_crosswalks.geojson", driver="GeoJSON")
