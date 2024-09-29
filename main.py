import modal

app = modal.App("crossing-distances")

osmnx_image = modal.Image.from_dockerfile(path="./Dockerfile")


def setup_argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Run crossing-distances")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Run local function")
    group.add_argument("--remote", action="store_true", help="Run remote function")
    return parser


def create_logger():
    import logging

    logger = logging.getLogger(__name__)

    formatter = logging.Formatter(
        fmt="$(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.",
    )
    handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@app.function(image=osmnx_image)
def get_osm_crossings(city: str):
    import osmnx as ox

    logger = create_logger()

    if city == "local":
        logger.info("Testing out logging")
        return

    logger.info(f"Fetching crosswalk data for {city} from OpenStreetMap...")

    # Set console logging to True
    ox.settings.log_console = True

    # Set useful tags for walking network
    useful_tags = ox.settings.useful_tags_way + [
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

    ox.config(use_cache=True, log_console=True, useful_tags_way=useful_tags)

    # Download the network with specified tags
    G = ox.graph_from_place(
        query=city, network_type="walk", simplify=False, retain_all=True
    )

    # Identify and remove non-walk edges
    non_walk = []
    for u, v, k, d in G.edges(keys=True, data=True):
        is_walk = "walk" in d and d["walk"] == "designated"
        is_crossing = (
            d.get("highway") in ["crossing", "pedestrian"]
            or "crossing" in d
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
    G = ox.utils_graph.remove_isolated_nodes(G)
    G = ox.simplify_graph(G)

    # Calculate and print total edge length
    stats = ox.stats.basic_stats(G)
    logger.info("Total Edge Length:", stats["edge_length_total"])

    # Plot the graph
    # ox.plot_graph(G, node_color="w", node_size=15, edge_color="b", edge_linewidth=0.5, figsize=(20, 20))


@app.local_entrypoint()
def main(mode: str):
    if mode.lower() == "remote":
        get_osm_crossings.remote("Boston, Massachusetts, USA")
    else:
        get_osm_crossings.local("local")
