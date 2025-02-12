import modal

from inference.osm_utils import (
    combine_similar_crosswalks,
    create_perpendicular_lines,
    map_nodes_to_closest_edges,
)
from inference.utils import RADIUS, app, osmnx_image

scratch_volume = modal.Volume.from_name("scratch", create_if_missing=True)


@app.function(volumes={"/scratch": scratch_volume}, image=osmnx_image, timeout=3600)
def osm_ingest(place: str):
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd

    # Set useful tags for walking network
    ox.settings.useful_tags_way = ox.settings.useful_tags_way + [
        "crossing",
        "crossing:uncontrolled",
        "crossing:zebra",
        "crossing:marked",
        "crossing:traffic_signals",
        "crossing:school",
        "crossing:island",
        "crossing:refuge_island",
        "crossing:island:central",
        "crossing:island:central:traffic_signals",
        "crossing:island:central:marked",
        "crossing:island:central:zebra",
        "crossing:unmarked",
        "highway:crossing",
        "pedestrian",
        "footway",
        "pedestrian_crossing",
    ]

    ox.settings.use_cache = True
    ox.settings.log_console = True

    highway_subtypes = [
        "primary",
        "primary_link",
        "unclassified",
        "residential",
        "secondary",
        "secondary_link",
        "service",
        "tertiary",
        "tertiary_link",
        "trunk",
        "trunk_link",
        "busway",
        "construction",
        "corridor",
    ]

    service_exclusion = [
        "parking_aisle",
        "driveway",
    ]

    # Download the network with specified tags
    G = ox.graph_from_place(
        query=place, network_type="all", simplify=False, retain_all=True
    )

    # Identify and remove non-walk edges
    non_walk = []
    for u, v, k, d in G.edges(keys=True, data=True):
        is_walk = "walk" in d and d["walk"] == "designated"
        is_crossing = (
            d.get("highway") in ["crossing", "pedestrian"]
            or d.get("footway")
            == "crossing"  # Change crossing in d to a stricter filter, excluding cycleway crossings
            or any(tag.lower() == "crossing:uncontrolled" for tag in d)
            or "crossing:raised" in d
            or "crossing:speed_table" in d
            or "crossing:hump" in d
            or "crossing:zebra" in d
            or "crossing:marked" in d
            or "crossing:traffic_signals" in d
            or "crossing:school" in d
            or "crossing:island" in d
            or "crossing:refuge_island" in d
            or "crossing:island:central" in d
            or "crossing:central_island" in d
            or "crossing:island:central:traffic_signals" in d
            or "crossing:island:central:marked" in d
            or "crossing:island:central:zebra" in d
            or "crossing:island:central:uncontrolled" in d
            or "crossing:island:central:unmarked" in d
            or "crossing:unmarked" in d
            or "highway:crossing" in d
            or "pedestrian" in d
        )

        # Include pedestrian crossings at intersections without traffic signals
        is_intersection = "highway" in d and d["highway"] == "uncontrolled_intersection"
        if is_intersection and not is_crossing:
            is_crossing = True

        # Exclude pedestrian sidewalks
        is_sidewalk = "sidewalk" in d
        if not is_walk and not is_crossing and not is_sidewalk:
            non_walk.append((u, v, k))

    G.remove_edges_from(non_walk)
    G = ox.simplify_graph(G, edge_attrs_differ=["osmid"])

    # Next, get the intersections
    G_drive = ox.graph_from_place(
        query=place,
        network_type="drive",
        custom_filter=f'["highway"~"{"|".join(highway_subtypes)}"]["service"!~"{"|".join(service_exclusion)}"]["access"!~"private"]["aeroway"!~".*"]["indoor"!~"yes"]',
    )
    G_drive = ox.project_graph(G_drive)
    all_intersections = ox.consolidate_intersections(
        G_drive, rebuild_graph=False, dead_ends=False
    )

    # Save the intersection coordinates
    all_intersections.to_crs("EPSG:4326").get_coordinates().to_csv(
        "/scratch/raw_intersection_coordinates_v2_0.csv"
    )

    crosswalk_features_gdf = ox.features_from_place(
        query=place,
        tags={"highway": "crossing", "crossing": True, "disused": False},
    )

    crosswalk_nodes_gdf = crosswalk_features_gdf.query(
        "element_type == 'node'"
    ).droplevel("element_type")

    # Filter intersections to only those with pedestrian crossings within 15m
    # of the intersection. This is done using a spatial join between the nodes GDF from intersections and the edges GDF from G
    crosswalk_edges_gdf = pd.concat(
        [
            ox.graph_to_gdfs(G, nodes=False, edges=True),
            crosswalk_features_gdf.query("element_type == 'way'").droplevel(
                "element_type"
            ),
        ]
    )
    crosswalk_edges_gdf = crosswalk_edges_gdf.to_crs("EPSG:3857")
    crosswalk_nodes_gdf = crosswalk_nodes_gdf.to_crs("EPSG:3857")
    crosswalk_nodes_gdf = crosswalk_nodes_gdf.loc[
        ~crosswalk_nodes_gdf.geometry.within(
            crosswalk_edges_gdf.buffer(2.5).unary_union
        )
    ]
    crosswalk_edges_and_nodes_gdf = pd.concat(
        [
            crosswalk_edges_gdf.reset_index()[["osmid", "geometry"]],
            crosswalk_nodes_gdf.reset_index()[["osmid", "geometry"]],
        ]
    )

    crosswalk_edges_gdf.drop_duplicates("osmid", inplace=True)
    crosswalk_edges_and_nodes_gdf.drop_duplicates("osmid", inplace=True)

    drive_edges = ox.graph_to_gdfs(G_drive, nodes=False, edges=True)
    drive_edges = (
        drive_edges.reset_index().set_index(
            drive_edges.index.to_frame().apply(
                lambda x: f"{x['u']}-{x['v']}-{x['key']}", axis=1
            )
        )
        # .drop_duplicates("osmid")
    )

    # Map crosswalk nodes (for crosswalks that don't already have an edge)
    # to driving roads
    mapping = map_nodes_to_closest_edges(crosswalk_nodes_gdf, drive_edges)

    # Build perpendicular crosswalks for each node
    artificial_crossings = create_perpendicular_lines(
        nodes=crosswalk_nodes_gdf.to_crs("EPSG:3857"),
        edges=drive_edges.to_crs("EPSG:3857"),
        mapping=mapping,
    ).to_crs(epsg=4326)

    # Combine crosswalks that are actually the same,
    # just on different sides of the road
    # (for example, for roads separated by a median strip)
    merged_crosswalks = combine_similar_crosswalks(
        gdf=artificial_crossings.assign(
            drive_edge_mapping=list(
                map(lambda x: mapping[x], artificial_crossings.index)
            )
        ),
        heading_tol=5,
        buffer=3,
        magnify=3,
    )

    # Append merged crosswalks to existing crosswalks
    all_crosswalk_edges = pd.concat(
        [
            merged_crosswalks[["original_line"]]
            .reset_index()
            .rename(columns={"original_line": "geometry", "combined_id": "osmid"})
            .set_geometry("geometry", crs="EPSG:4326")
            .to_crs("EPSG:3857"),
            crosswalk_edges_gdf[["osmid", "geometry"]],
        ]
    )

    # Save crosswalk edges
    all_crosswalk_edges.to_file("/scratch/crosswalk_edges_v2_0.shp", index=False)

    all_intersections_gdf = gpd.GeoDataFrame(
        all_intersections, columns=["geometry"], crs=all_intersections.crs
    ).to_crs("EPSG:3857")

    # Perform the spatial join
    intersections_with_crosswalks = gpd.sjoin_nearest(
        all_intersections_gdf[["geometry"]],
        crosswalk_edges_and_nodes_gdf[["geometry"]],
        how="inner",
        max_distance=RADIUS,
        distance_col="d",
    )

    # Save result as csv
    intersections_with_crosswalks = intersections_with_crosswalks.drop_duplicates(
        "geometry"
    ).to_crs("EPSG:4326")
    intersections_with_crosswalks["x"] = intersections_with_crosswalks.geometry.x
    intersections_with_crosswalks["y"] = intersections_with_crosswalks.geometry.y

    intersections_with_crosswalks[["x", "y"]].to_csv(
        "/scratch/intersection_coordinates_v2_0.csv"
    )
