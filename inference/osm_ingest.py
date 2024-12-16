import modal
from inference.utils import osmnx_image, RADIUS

app = modal.App("crossing-distance-inference")
scratch_volume = modal.Volume.from_name("scratch", create_if_missing=True)

@app.function(
    volumes={"/scratch": scratch_volume},
    image=osmnx_image
)
def osm_ingest(place: str):
    import osmnx as ox
    import geopandas as gpd

    # Set useful tags for walking network
    useful_tags = ox.settings.useful_tags_way + [
        'crossing', 'crossing:uncontrolled', 'crossing:zebra', 'crossing:marked',
        'crossing:traffic_signals', 'crossing:school', 'crossing:island',
        'crossing:refuge_island', 'crossing:island:central',
        'crossing:island:central:traffic_signals', 'crossing:island:central:marked',
        'crossing:island:central:zebra', 'crossing:unmarked', 'highway:crossing',
        'pedestrian', 'footway', 'pedestrian_crossing'
    ]

    highway_subtypes = [
        "primary",
        "primary_link",
        "unclassified",
        "motorway",
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
        "corridor"
    ]

    ox.config(use_cache=True, log_console=True, useful_tags_way=useful_tags)

    # Download the network with specified tags
    G = ox.graph_from_place(query=place, network_type='all', simplify=False, retain_all=True)


    # Identify and remove non-walk edges
    non_walk = []
    for u, v, k, d in G.edges(keys=True, data=True):
        is_walk = "walk" in d and d["walk"] == "designated"
        is_crossing = (
            d.get("highway") in ["crossing", "pedestrian"] or
            d.get("footway") == "crossing" or # Change crossing in d to a stricter filter, excluding cycleway crossings
            any(tag.lower() == "crossing:uncontrolled" for tag in d) or
            "crossing:raised" in d or
            "crossing:speed_table" in d or
            "crossing:hump" in d or
            "crossing:zebra" in d or
            "crossing:marked" in d or
            "crossing:traffic_signals" in d or
            "crossing:school" in d or
            "crossing:island" in d or
            "crossing:refuge_island" in d or
            "crossing:island:central" in d or
            "crossing:central_island" in d or
            "crossing:island:central:traffic_signals" in d or
            "crossing:island:central:marked" in d or
            "crossing:island:central:zebra" in d or
            "crossing:island:central:uncontrolled" in d or
            "crossing:island:central:unmarked" in d or
            "crossing:unmarked" in d or
            "highway:crossing" in d or
            "pedestrian" in d
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
    G = ox.simplify_graph(G, edge_attrs_differ=['osmid'])

    # Next, get the intersections
    G_drive = ox.graph_from_place(
        query=place,
        network_type="drive",
        custom_filter=f'[\"highway\"~\"{"|".join(highway_subtypes)}\"]'
    )
    G_drive = ox.project_graph(G_drive)
    all_intersections = ox.consolidate_intersections(G_drive, rebuild_graph=False, dead_ends=False)

    # Save the intersection coordinates
    all_intersections.to_crs("EPSG:4326").get_coordinates().to_csv("/scratch/raw_intersection_coordinates.csv")

    # Filter intersections to only those with pedestrian crossings within 15m
    # of the intersection. This is done using a spatial join between the nodes GDF from intersections and the edges GDF from G
    crosswalk_edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs("EPSG:3857")
    crosswalk_edges_gdf.drop_duplicates("osmid", inplace=True)

    # Save crosswalk edges
    crosswalk_edges_gdf.to_file("/scratch/crosswalk_edges.shp")

    all_intersections_gdf = gpd.GeoDataFrame(all_intersections, columns=["geometry"], crs=all_intersections.crs).to_crs("EPSG:3857")

    # Perform the spatial join
    intersections_with_crosswalks = gpd.sjoin_nearest(
        all_intersections_gdf[["geometry"]],
        crosswalk_edges_gdf[["geometry"]],
        how="inner",
        max_distance=RADIUS,
        distance_col="d"
    )

    # Save result as csv
    intersections_with_crosswalks = intersections_with_crosswalks.drop_duplicates("geometry").to_crs("EPSG:4326")
    intersections_with_crosswalks["x"] = intersections_with_crosswalks.geometry.x
    intersections_with_crosswalks["y"] = intersections_with_crosswalks.geometry.y

    intersections_with_crosswalks[["x", "y"]].to_csv("/scratch/intersection_coordinates.csv")
