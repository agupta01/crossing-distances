import argparse
import geopandas as gpd
import numpy as np
from modal import Volume
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from inference.utils import decode_crosswalk_id, get_from_volume

RAW_CROSSWALK_FILE_NAME = "crosswalk_edges_v2_0_{code}.geojson"
CROSSWALK_FILE_NAME = "refined_crosswalks_v2_1_3-grow30+15.geojson"

def parse_filenames_into_coords(filename: str) -> tuple[float, float]:
    """Parses a filename into a (lat, long) coordinate tuple."""
    return decode_crosswalk_id(filename.split("_", 1)[-1].split(".", 1)[0])

def merge_nearby_coordinates(coords_gdf, distance_threshold=12):
    """
    Merge coordinates that are within distance_threshold meters of each other,
    using the centroid of merged clusters as the new coordinate.

    Args:
        coords_gdf: GeoDataFrame with coordinates in EPSG:4326
        distance_threshold: Distance in meters for merging (default 12)

    Returns:
        GeoDataFrame with merged coordinates
    """

    if len(coords_gdf) == 0:
        return coords_gdf

    # Convert to EPSG:3857 for accurate distance calculations in meters
    coords_utm = coords_gdf.to_crs("EPSG:3857")

    # Extract x,y coordinates for clustering
    coordinates_array = np.column_stack(
        [coords_utm.geometry.x.values, coords_utm.geometry.y.values]
    )

    # Use DBSCAN clustering to group nearby points
    # eps = distance threshold, min_samples = 1 (every point can form a cluster)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric="euclidean")
    cluster_labels = clustering.fit_predict(coordinates_array)

    # Create merged coordinates by taking centroid of each cluster
    merged_coords = []
    cluster_info = []

    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        # Get all points in this cluster
        cluster_mask = cluster_labels == label
        cluster_points = coordinates_array[cluster_mask]
        original_indices = coords_utm.index[cluster_mask].tolist()

        # Calculate centroid
        centroid_x = np.mean(cluster_points[:, 0])
        centroid_y = np.mean(cluster_points[:, 1])

        merged_coords.append([centroid_x, centroid_y])
        cluster_info.append(
            {
                "cluster_id": label,
                "point_count": len(cluster_points),
                "original_indices": original_indices,
                "centroid_x_utm": centroid_x,
                "centroid_y_utm": centroid_y,
            }
        )

    # Create new GeoDataFrame with merged coordinates
    merged_coords_array = np.array(merged_coords)

    merged_gdf = gpd.GeoDataFrame(
        {
            "cluster_id": [info["cluster_id"] for info in cluster_info],
            "merged_point_count": [info["point_count"] for info in cluster_info],
            "original_indices": [info["original_indices"] for info in cluster_info],
        },
        geometry=gpd.points_from_xy(
            merged_coords_array[:, 0], merged_coords_array[:, 1]
        ),
        crs="EPSG:3857",
    )

    # Convert back to EPSG:4326
    merged_gdf = merged_gdf.to_crs("EPSG:4326")

    # Update x,y columns for final coordinates
    merged_gdf["x"] = merged_gdf.geometry.x
    merged_gdf["y"] = merged_gdf.geometry.y

    # Report merging statistics
    total_original = len(coords_gdf)
    total_merged = len(merged_gdf)
    points_merged = total_original - total_merged

    if points_merged > 0:
        print(f"Merged {points_merged} coordinates into {points_merged} fewer points.")
        print(f"Original count: {total_original}, Final count: {total_merged}")

        # Show details of multi-point clusters
        multi_point_clusters = merged_gdf[merged_gdf["merged_point_count"] > 1]
        if len(multi_point_clusters) > 0:
            print(f"Created {len(multi_point_clusters)} merged clusters:")
            for _, cluster in multi_point_clusters.iterrows():
                print(
                    f"  Cluster {cluster['cluster_id']}: {cluster['merged_point_count']} points merged"
                )
    else:
        print("No coordinates were close enough to merge.")

    return merged_gdf[["x", "y", "geometry"]]  # Return clean result


def get_raw_coords_lost(env_name):
    """Finds the coordinates of crossings that exist in raw_crosswalks but not in final crosswalks."""
    raw_crosswalks = gpd.read_file(
        get_from_volume(
            "scratch",
            RAW_CROSSWALK_FILE_NAME.format(code=env_name),
            environment_name=env_name,
        )
    ).to_crs("EPSG:3857")
    crosswalks = gpd.read_file(
        get_from_volume("scratch", CROSSWALK_FILE_NAME, environment_name=env_name)
    ).to_crs("EPSG:3857")

    missing = gpd.sjoin(
        raw_crosswalks.assign(geometry=raw_crosswalks["geometry"].buffer(1)),
        crosswalks,
        how="left",
    )

    missing = missing.loc[missing.isnull()["index_right"]]
    missing = missing.assign(geometry=missing["geometry"].centroid)
    missing = merge_nearby_coordinates(missing, distance_threshold=25)
    return missing


def get_available_image_coords(env_name):
    """Fetches coordinates of all available images for a city."""
    image_volume = Volume.from_name(
        f"crosswalk-data-{env_name}", environment_name=env_name
    )
    images = set(filter(lambda x: x.path.endswith(".jpeg"), image_volume.listdir("/")))
    image_coords = list(map(lambda x: parse_filenames_into_coords(x.path), images))

    image_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            x=list(map(lambda coord: coord.long, image_coords)),
            y=list(map(lambda coord: coord.lat, image_coords)),
            crs="EPSG:4326",
        )
    )
    return image_gdf


def get_balltree(missing, image_gdf):
    """
    Consolidates missing images by finding those that are
    not within a 10 meter radius of an available image coordinate.
    """
    missing_proj = missing.to_crs("EPSG:3857")
    image_gdf_proj = image_gdf.to_crs("EPSG:3857")

    missing_coords = np.vstack([missing_proj.geometry.x, missing_proj.geometry.y]).T
    image_coords = np.vstack([image_gdf_proj.geometry.x, image_gdf_proj.geometry.y]).T

    tree = BallTree(image_coords, metric="euclidean")
    radius = 10  # in meters

    neighbor_indices = tree.query_radius(missing_coords, r=radius)

    mask_no_match = np.array([len(hits) == 0 for hits in neighbor_indices])
    unmatched_missing = missing[mask_no_match]
    return unmatched_missing


def main(env_name):
    missing = get_raw_coords_lost(env_name)
    image_gdf = get_available_image_coords(env_name)

    final_missing = get_balltree(missing, image_gdf)

    return final_missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="city code to run")
    args = parser.parse_args()
    print(main(args.env_name).info())
