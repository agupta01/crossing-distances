import modal
from main import osmnx_image

app = modal.App("crossing-distance-inference")
scratch_volume = modal.Volume.from_name("scratch", create_if_missing=True)

@app.function(
    volumes={"/scratch": scratch_volume},
    mounts=[modal.Mount.from_local_dir("./", remote_path="/src")],
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

    # Next, get the intersections
    G_drive = ox.graph_from_place(
        query=place,
        network_type="drive",
        custom_filter=f'[\"highway\"~\"{"|".join(highway_subtypes)}\"]'
    )
    all_intersections = ox.consolidate_intersections(G_drive, rebuild_graph=False, dead_ends=False)

    # Filter intersections to only those with pedestrian crossings within 15m
    # of the intersection. This is done using a spatial join between the nodes GDF from intersections and the edges GDF from G
    crosswalk_edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    all_intersections_gdf = gpd.GeoDataFrame(all_intersections, columns=["geometry"], crs=all_intersections.crs)

    # Buffer the crosswalk edges by 15m
    crosswalk_edges_gdf["geometry"] = crosswalk_edges_gdf["geometry"].buffer(15)

    # Perform the spatial join
    intersections_with_crosswalks = gpd.sjoin(all_intersections_gdf, crosswalk_edges_gdf, how="inner", op="intersects")

    # Save result as shapefile
    intersections_with_crosswalks.to_file("intersections_with_crosswalks.shp")
