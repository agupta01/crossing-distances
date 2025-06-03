from figures.utils import *
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
import os
import matplotlib.patheffects as PathEffects
import requests
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm
from figures.clustering import cluster_cities_by_crosswalk_distributions


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
import os
import matplotlib.patheffects as PathEffects
import requests
from io import BytesIO
from zipfile import ZipFile


def plot_clusters_on_us_map_fixed(
    cluster_df,
    all_city_codes,
    output_dir="./figures/output",
):
    """
    Plot cities on a US map with cluster-based coloring and non-overlapping points.
    Fixed version to ensure points are displayed.
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-quality parameters
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    # Merge cluster data with city codes to get state information
    merged_data = pd.merge(
        cluster_df,
        all_city_codes[["City", "State"]],
        on="City",
        how="left",
    )
    print(f"Merged data: {merged_data.head()}")
    # Use GeoPandas to geocode cities
    import geopandas as gpd
    from geopandas.tools import geocode

    # Create a location string for geocoding
    merged_data["location"] = (
        merged_data["City"].astype(str) + ", " + merged_data["State"].fillna("")
    )
    print(f"Sample locations for geocoding: {merged_data['location'].head()}")

    # Run geocoding (may take a while if many cities)
    try:
        geocoded = geocode(
            merged_data["location"],
            provider="nominatim",
            user_agent="crosswalk-mapper",
            timeout=10,
        )
        # Merge geocoded geometry back
        merged_data = merged_data.join(geocoded.set_geometry("geometry"), how="left")
        merged_data["latitude"] = merged_data.geometry.apply(
            lambda geom: geom.y if geom else None
        )
        merged_data["longitude"] = merged_data.geometry.apply(
            lambda geom: geom.x if geom else None
        )
        if (
            merged_data["latitude"].isnull().any()
            or merged_data["longitude"].isnull().any()
        ):
            print(
                "Warning: Some cities could not be geocoded and will not appear on the map."
            )
    except Exception as e:
        print(f"Geocoding failed: {e}")
        merged_data["latitude"] = None
        merged_data["longitude"] = None
    city_coords = merged_data

    # Print debugging info
    print(f"Number of cities with coordinates: {len(city_coords)}")
    if len(city_coords) > 0:
        print(
            f"Sample coordinates: {city_coords[['City', 'latitude', 'longitude']].head()}"
        )

    # Download and load US map data
    try:
        usa = download_us_shapefile()
    except Exception as e:
        print(f"Error downloading shapefile: {e}")
        print("Falling back to simplified US outline")
        usa = None

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot US outline
    if usa is not None:
        usa.boundary.plot(ax=ax, linewidth=0.8, color="#777777", zorder=1)
        usa.plot(ax=ax, color="#F5F5F5", alpha=0.6, zorder=0)
    else:
        # Fallback to drawing simplified outline
        draw_simplified_us_outline(ax)

    # Set plot limits to continental US
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    # Get number of clusters for color mapping
    num_clusters = cluster_df["Cluster"].nunique()

    # Use a colorblind-friendly palette
    palette = sns.color_palette("viridis", num_clusters)

    # Apply point displacement to prevent overlaps
    city_coords = displace_overlapping_points(city_coords, min_distance=1.0)

    # IMPORTANT FIX: Ensure we're actually plotting points
    # Plot each city, colored by cluster
    for cluster_id in range(1, num_clusters + 1):
        # Filter cities in this cluster
        cluster_cities = city_coords[city_coords["Cluster"] == cluster_id]
        print(f"Plotting Cluster {cluster_id} with {len(cluster_cities)} cities")

        if len(cluster_cities) > 0:
            # Plot points with larger size to ensure visibility
            scatter = ax.scatter(
                cluster_cities["longitude"],
                cluster_cities["latitude"],
                s=120,  # Increased size for visibility
                c=[palette[cluster_id - 1]],
                label=f"Cluster {cluster_id}",
                edgecolors="white",
                linewidth=1.0,
                alpha=1.0,  # Full opacity
                zorder=100,  # Ensure points are on top
            )

            # Add city labels with white outline for better visibility
            for _, city in cluster_cities.iterrows():
                txt = ax.text(
                    city["longitude"],
                    city["latitude"] + 0.3,
                    city["City"],
                    fontsize=3,
                    ha="center",
                    va="center",
                    fontweight="light",
                    color="black",
                    zorder=101,
                )
                # Add white outline to text for better visibility
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=1, foreground="white")]
                )

    # Remove axis ticks and spines for a cleaner map
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title and legend
    ax.set_title(
        "Geographic Distribution of Crosswalk Length Clusters",
        fontweight="bold",
        fontsize=14,
    )

    # Add legend with professional styling
    legend = ax.legend(
        title="Clusters",
        title_fontsize=12,
        fontsize=10,
        markerscale=1.2,
        frameon=True,
        framealpha=0.9,
        edgecolor="dimgray",
        loc="lower right",
    )

    # Add scale bar
    scalebar_length = 500  # km
    add_scale_bar(ax, length=scalebar_length)

    # Add compass rose
    add_compass_rose(ax)

    # Save high-quality outputs
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/geographic_cluster_map.png", dpi=600, bbox_inches="tight"
    )

    return fig, ax


# Helper function for drawing simplified US outline
def draw_simplified_us_outline(ax):
    """Draw a simplified outline of the continental US."""
    from matplotlib.path import Path
    import matplotlib.patches as patches

    # Simplified outline of continental US as vertices (more detailed than previous version)
    us_outline = [
        (-124.7, 48.2),  # Northwest corner (Washington)
        (-123.0, 46.0),  # Oregon coast
        (-122.0, 42.0),  # Northern CA
        (-122.5, 37.5),  # San Francisco
        (-118.5, 34.0),  # Los Angeles
        (-117.0, 32.5),  # San Diego
        (-114.8, 32.5),  # Southern AZ
        (-106.5, 31.8),  # Southern NM
        (-103.0, 29.0),  # Southern TX
        (-97.0, 26.0),  # Gulf coast TX
        (-94.0, 29.5),  # Gulf coast TX/LA
        (-89.6, 29.2),  # Gulf coast LA
        (-84.3, 29.8),  # Florida panhandle
        (-82.0, 27.5),  # Florida west coast
        (-80.0, 25.2),  # Southern Florida
        (-80.5, 28.0),  # Florida east coast
        (-79.9, 32.0),  # SC coast
        (-76.0, 35.0),  # NC outer banks
        (-75.4, 39.2),  # Delaware
        (-71.0, 41.5),  # Rhode Island
        (-69.8, 41.3),  # Cape Cod
        (-68.0, 44.0),  # Maine southern coast
        (-66.9, 44.8),  # Maine eastern point
        (-67.5, 47.0),  # Maine/Canada border
        (-73.3, 45.0),  # Vermont/Canada
        (-77.0, 47.0),  # Upper NY
        (-82.0, 46.5),  # Upper MI
        (-86.5, 47.5),  # Upper MI western point
        (-90.0, 47.0),  # Northern WI
        (-92.0, 46.7),  # Northern MN
        (-97.2, 49.0),  # Northern ND
        (-104.0, 49.0),  # Northern MT
        (-111.0, 49.0),  # Northern ID
        (-122.0, 49.0),  # Northern WA
        (-124.7, 48.2),  # Back to start
    ]

    # Create path
    codes = [Path.MOVETO] + [Path.LINETO] * (len(us_outline) - 2) + [Path.CLOSEPOLY]
    path = Path(us_outline, codes)
    patch = patches.PathPatch(
        path,
        facecolor="#F5F5F5",
        edgecolor="#777777",
        linewidth=0.8,
        alpha=0.6,
        zorder=1,
    )
    ax.add_patch(patch)

    return ax


def download_us_shapefile():
    """Download US shapefile data from Natural Earth website."""
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )

    # Download the zip file
    response = requests.get(url)
    response.raise_for_status()

    # Extract the shapefile from zip
    with ZipFile(BytesIO(response.content)) as zipf:
        # Extract to a temporary directory
        temp_dir = os.path.join(os.getcwd(), "temp_ne_data")
        os.makedirs(temp_dir, exist_ok=True)
        zipf.extractall(temp_dir)

        # Load the shapefile
        shapefile_path = os.path.join(temp_dir, "ne_110m_admin_0_countries.shp")
        world = gpd.read_file(shapefile_path)

        # Filter for USA
        usa = world[world["NAME"] == "United States of America"]
        return usa


def geocode_cities(city_data):
    """Geocode cities using GeoPandas."""
    try:
        geocoded_cities = gpd.tools.geocode(city_data["City"], provider="nominatim")
        city_data["latitude"] = geocoded_cities.geometry.y
        city_data["longitude"] = geocoded_cities.geometry.x
    except Exception as e:
        print(f"Error geocoding cities: {e}")
        city_data["latitude"] = np.nan
        city_data["longitude"] = np.nan

    # Add columns for coordinates
    city_data["latitude"] = np.nan
    city_data["longitude"] = np.nan

    # Set random seed for reproducibility
    np.random.seed(42)

    # Assign coordinates based on state with jitter
    for i, row in city_data.iterrows():
        state = row["State"]
        if state in state_centroids:
            # Add random jitter to prevent perfect overlaps
            lat_jitter = np.random.normal(0, 0.3)
            lon_jitter = np.random.normal(0, 0.3)

            city_data.at[i, "latitude"] = state_centroids[state][0] + lat_jitter
            city_data.at[i, "longitude"] = state_centroids[state][1] + lon_jitter

    return city_data


def displace_overlapping_points(city_data, min_distance=1.0, iterations=50):
    """Apply force-directed algorithm to separate overlapping points."""
    if len(city_data) <= 1:
        return city_data

    # Extract coordinates
    coords = city_data[["longitude", "latitude"]].values.copy()
    n_points = len(coords)

    # Iterative repulsion algorithm
    for _ in range(iterations):
        # Calculate all pairwise distances
        dist_matrix = distance.squareform(distance.pdist(coords))
        np.fill_diagonal(dist_matrix, np.inf)  # Avoid self-interactions

        # Calculate displacement for each point
        displacement = np.zeros_like(coords)

        for i in range(n_points):
            # Find points too close to current point
            too_close = dist_matrix[i] < min_distance

            if not np.any(too_close):
                continue

            for j in np.where(too_close)[0]:
                # Vector from j to i
                direction = coords[i] - coords[j]
                dist = dist_matrix[i, j]

                if dist > 0:
                    # Normalize and scale by how much overlap there is
                    force = min(0.3, (min_distance - dist) / min_distance)
                    direction = direction / dist * force
                    displacement[i] += direction

        # Apply displacement with damping factor
        coords += displacement * 0.5

    # Update the DataFrame with new coordinates
    city_data["longitude"] = coords[:, 0]
    city_data["latitude"] = coords[:, 1]

    return city_data


def add_scale_bar(ax, length=500, location=(0.78, 0.05)):
    """Add a scale bar to the map in kilometers."""
    # Convert km to degrees longitude (approximate)
    mean_latitude = 38  # Mid-US latitude
    lon_scale = length / (111.32 * np.cos(np.radians(mean_latitude)))

    # Draw scale bar
    x_start = location[0]
    x_end = location[0] + lon_scale / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_pos = location[1]

    # Draw the bar with border
    ax.plot(
        [x_start, x_end],
        [y_pos, y_pos],
        "k-",
        linewidth=2.5,
        solid_capstyle="butt",
        transform=ax.transAxes,
        zorder=5,
    )

    # Add text label
    ax.text(
        (x_start + x_end) / 2,
        y_pos + 0.01,
        f"{length} km",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=ax.transAxes,
        zorder=5,
    )


def add_compass_rose(ax, position=(0.92, 0.92)):
    """Add a simple compass rose to the map."""
    x, y = position
    ax.text(
        x,
        y,
        "N",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        transform=ax.transAxes,
    )

    # Arrow pointing north
    ax.annotate(
        "",
        xy=(x, y - 0.03),
        xytext=(x, y - 0.01),
        arrowprops=dict(facecolor="black", width=1.5, headwidth=6, headlength=6),
        transform=ax.transAxes,
    )


if __name__ == "__main__":
    # Load city data
    df = pd.DataFrame(
        [get_measured_crosswalks_for_city(c) for c in tqdm(CITY_CODES)]
    ).sort_values("median", ascending=False, ignore_index=True)
    lengths_dict = {row["city"]: row["lengths"] for _, row in df.iterrows()}

    # Perform clustering
    cluster_results, linkage_matrix, distance_matrix = (
        cluster_cities_by_crosswalk_distributions(lengths_dict, num_clusters=5)
    )

    # Plot the map with both City and State information
    fig, ax = plot_clusters_on_us_map_fixed(
        cluster_results,
        all_city_codes.rename(columns={"Title": "City"}),
    )
