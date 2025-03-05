import math

import geopandas as gpd
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform
from shapely.prepared import prep
from tqdm.auto import tqdm


def map_nodes_to_closest_edges(crosswalk_nodes_gdf, drive_edges, tol=5):
    # Create copies to avoid modifying the original GeoDataFrames
    nodes = crosswalk_nodes_gdf.copy()
    edges = drive_edges.copy()

    # Ensure both GeoDataFrames are in the same CRS
    if nodes.crs != edges.crs:
        edges = edges.to_crs(nodes.crs)

    # Check if the CRS is projected; if not, convert to a projected CRS (EPSG:3857 as an example)
    if not nodes.crs.is_projected:
        projected_crs = "EPSG:3857"
        nodes = nodes.to_crs(projected_crs)
        edges = edges.to_crs(projected_crs)

    # Perform a spatial join to find the nearest edge for each node
    joined = gpd.sjoin_nearest(nodes, edges, how="left", distance_col="distance")

    # Filter out nodes where the closest edge is beyond the tolerance distance
    filtered = joined[joined["distance"] <= tol]

    # Create the dictionary mapping node osmid to the closest edge osmid
    node_to_edge = dict(zip(filtered.index, filtered["index_right"]))

    return node_to_edge


def create_perpendicular_lines(nodes, edges, mapping, length=8):
    perpendicular_lines = {"id": [], "geometry": []}

    for node_idx, edge_idx in tqdm(mapping.items()):
        # Get geometries from respective GeoDataFrames
        node_geom = nodes.loc[node_idx].geometry
        edge_geom = edges.loc[edge_idx].geometry

        # Project node onto edge and get closest point
        projected_point = edge_geom.interpolate(edge_geom.project(node_geom))

        # Handle different geometry types
        if edge_geom.geom_type == "LineString":
            # Find nearest segment in the LineString
            coords = list(edge_geom.coords)
            min_dist = float("inf")
            tangent_vector = np.array([0, 0])

            # Iterate through segments to find closest one
            for i in range(len(coords) - 1):
                seg_start = np.array(coords[i])
                seg_end = np.array(coords[i + 1])
                segment = LineString([seg_start, seg_end])
                dist = segment.distance(projected_point)

                if dist < min_dist:
                    min_dist = dist
                    tangent_vector = seg_end - seg_start  # Calculate actual vector
        else:
            raise ValueError("Edge geometry must be LineString")

        # Normalize tangent vector and get perpendicular
        if np.linalg.norm(tangent_vector) == 0:
            tangent_vector = np.array([1, 0])  # Handle zero-length vectors
        else:
            tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)

        # Get the unit vector of 90 degree rotation of tangent
        perp_vector = np.array([-tangent_vector[1], tangent_vector[0]])

        # Get offset in both directions
        offset = perp_vector * (length / 2)

        # Create the actual line
        start_point = Point(node_geom.x - offset[0], node_geom.y - offset[1])
        end_point = Point(node_geom.x + offset[0], node_geom.y + offset[1])
        perpendicular_lines["geometry"].append(LineString([start_point, end_point]))
        perpendicular_lines["id"].append(node_idx)

    return (
        gpd.GeoDataFrame(data=perpendicular_lines, crs=nodes.crs)
        .set_geometry("geometry")
        .set_index("id")
    )


def compute_heading(linestring):
    """
    Computes compass heading of LineString.

    Assumes LineString coordinates are in lat/long.

    In the event of a non-linear LineString, takes the tip-to-tip heading.

    Outputs are constrained [0, 180] to avoid confusing phase-separated headings as different.
    """
    if len(linestring.coords) < 2:
        raise ValueError("Linestring must have at least two points")
    start = linestring.coords[0]
    end = linestring.coords[-1]
    lon1, lat1 = start
    lon2, lat2 = end

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon_rad = lon2_rad - lon1_rad

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(dlon_rad)

    # Check if the direction is undefined (points are coincident or calculation results in zero vector)
    epsilon = 1e-15  # Tolerance for floating point comparison
    if abs(x) < epsilon and abs(y) < epsilon:
        return 0.0

    bearing = math.atan2(y, x)
    bearing_deg = math.degrees(bearing)
    heading = (bearing_deg + 360) % 360
    return heading % 180


def magnify_line(linestring: LineString, multiplier: float):
    """Magnifies a linestring by multiplier, preserving direction. Works with EPSG:3857 coordinates."""
    # Create transformers
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform linestring to EPSG:4326
    linestring_4326 = transform(to_4326.transform, linestring)

    coords = list(linestring_4326.coords)
    coords_3857 = list(linestring.coords)
    start, end = coords[0], coords[-1]
    start_3857, end_3857 = coords_3857[0], coords_3857[-1]

    # Calculate length and direction in EPSG:4326
    length = geodesic(start[::-1], end[::-1]).meters
    azimuth = math.degrees(
        math.atan2(end_3857[0] - start_3857[0], end_3857[1] - start_3857[1])
    )

    # Calculate new start and end points
    center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    new_start = geodesic(meters=length * multiplier / 2).destination(
        center[::-1], azimuth - 180
    )
    new_end = geodesic(meters=length * multiplier / 2).destination(
        center[::-1], azimuth
    )

    # Create new linestring in EPSG:4326
    new_linestring_4326 = LineString(
        [
            (new_start.longitude, new_start.latitude),
            (new_end.longitude, new_end.latitude),
        ]
    )

    # Transform back to EPSG:3857
    new_linestring_3857 = transform(to_3857.transform, new_linestring_4326)

    return new_linestring_3857


def circular_mean(angles_degrees):
    """Computes mean of array of compass headings."""
    angles_rad = np.deg2rad(angles_degrees)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_deg = np.rad2deg(mean_rad)
    return mean_deg % 360


def merge_crosswalks(buffers_gdf, crosswalks_gdf, heading_tol=5, export_graph=False):
    """
    Combine crosswalks if they are:
        1. Not mapped to the same drive edges
        2. Compass heading is +/- `heading_tol` of each other
        3. Have buffers that intersect

    Takes two GeoDataFrames that contain the columns:
    buffers_gdf: contains buffer information for crosswalks (in EPSG:3857)
        id - unique id column (usually osmid). Should be set as index
        heading - compass heading (0 - 359) of crosswalk
        geometry - Polygon buffers around crosswalks

    crosswalks_gdf: contains original crosswalks (in EPSG:4326)
        id - unique id column (usually osmid). Should be set as index
        geometry - LineString representing crosswalk
        drive_edge_mapping - crosswalk -> drive edge id (usually osmid)

    Result is GeoDataFrame (EPSG:4326) with the columns:
        id - unique id column (usually osmid)
        geometry - LineString representing crosswalk
    """
    if buffers_gdf.crs and not buffers_gdf.crs.is_projected:
        print(
            "Buffers GeoDataFrame must be projected to accurately compute intersections. Projecting to EPSG:3857..."
        )
        buffers_gdf = buffers_gdf.to_crs("EPSG:3857")

    # Initialize a graph to track connected features
    G = nx.Graph()
    G.add_nodes_from(buffers_gdf.index)

    # Create a spatial index for efficient intersection queries
    sindex = buffers_gdf.sindex

    # Iterate over each feature to find possible matches
    for idx in tqdm(buffers_gdf.index, total=len(buffers_gdf.index)):
        geom = buffers_gdf.loc[idx, "geometry"]
        heading = buffers_gdf.loc[idx, "heading"]
        mapped_street = crosswalks_gdf.loc[idx, "drive_edge_mapping"]
        prepared_geom = prep(geom)  # Prepared geometry for faster intersection checks

        # Find candidate features via spatial index (bounding box intersection)
        possible_matches_indices = list(sindex.intersection(geom.bounds))
        possible_matches = buffers_gdf.iloc[possible_matches_indices]
        possible_matches = possible_matches[
            possible_matches.index != idx
        ]  # Remove current edge

        # Check each candidate for actual geometry intersection and heading condition
        for match_idx, match_row in possible_matches.iterrows():
            similar_headings = abs(heading - match_row["heading"]) <= heading_tol
            if (
                prepared_geom.intersects(match_row["geometry"])
                and similar_headings
                and (
                    mapped_street != crosswalks_gdf.loc[match_idx, "drive_edge_mapping"]
                )
            ):
                G.add_edge(idx, match_idx)

    # Identify connected components (groups of features to merge)
    groups = list(nx.connected_components(G))

    # Aggregate each group into a new feature with spanning line
    new_rows = []
    for group in groups:
        subgroup = buffers_gdf.loc[list(group)]
        combined_geom = subgroup.geometry.unary_union
        if combined_geom.is_empty:
            continue  # Skip groups with empty geometries

        # Compute average heading using circular mean
        avg_heading = circular_mean(subgroup["heading"].values)
        combined_id = "_".join(subgroup.index.astype(str))

        # Get lines for ids in subgroup
        line_group = crosswalks_gdf.loc[subgroup.index.tolist()]["geometry"]

        # Collect all points from original_line geometries
        all_points = []
        for line in line_group:
            if line is not None and not line.is_empty:
                all_points.extend(line.coords)

        # Create spanning line along the average heading direction
        if len(all_points) >= 2:
            theta = np.deg2rad(avg_heading)
            dx = np.sin(theta)
            dy = np.cos(theta)
            projections = [x * dx + y * dy for x, y in all_points]
            min_proj = min(projections)
            max_proj = max(projections)
            min_idx = projections.index(min_proj)
            max_idx = projections.index(max_proj)
            spanning_line = LineString([all_points[min_idx], all_points[max_idx]])
        else:
            spanning_line = LineString()  # Empty line if insufficient points

        new_rows.append(
            {
                "combined_id": combined_id,
                "heading": avg_heading,
                "geometry": combined_geom,
                "original_line": spanning_line,
            }
        )

    # Create and return the new GeoDataFrame
    res_gdf = gpd.GeoDataFrame(new_rows, crs=crosswalks_gdf.crs).set_index(
        "combined_id"
    )
    if export_graph:
        return res_gdf, G
    return res_gdf


def combine_similar_crosswalks(gdf, heading_tol=5, buffer=3, magnify=1.5):
    """
    Combine crosswalks that appear to be the same, but on different sides of the road.

    Takes a single geodataframe of crosswalk edges with columns:
        id - unique id column (usually osmid). Should be set as index
        geometry - LineString representing crosswalk
        drive_edge_mapping - crosswalk -> drive edge id (usually osmid)

    Also takes heading_tol (default 5), buffer (default 3) and magnify (default 1.5) as
    tuning parameters for the preparation/merge step:
        heading_tol - defines window of compass heading for mergeable crosswalks
        buffer - defines how large to make buffer around crosswalks
        magnify - defines how much to stretch the crosswalk by before applying buffer

    Result is GeoDataFrame (EPSG:4326) with the columns:
        id - unique id column (usually osmid)
        geometry - LineString representing crosswalk
    """
    gdf_3857 = gdf.to_crs("EPSG:3857")
    gdf_4326 = gdf.to_crs("EPSG:4326")
    buffers_gdf = gdf_3857.assign(
        heading=gdf_3857["geometry"].apply(compute_heading),
        geometry=gdf_3857["geometry"]
        .apply(lambda x: magnify_line(x, multiplier=magnify))
        .buffer(buffer),
    ).drop(columns=["drive_edge_mapping"])

    return merge_crosswalks(
        buffers_gdf,
        crosswalks_gdf=gdf_4326,
        heading_tol=heading_tol,
    )
