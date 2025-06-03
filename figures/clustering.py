from figures.utils import *


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS
import os
from tqdm import tqdm

# Set publication-quality parameters
plt.rcParams.update(
    {
        # Font settings
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        # Figure settings
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        # Line settings
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.5,
    }
)


def cluster_cities_by_crosswalk_distributions(
    lengths_dict, num_clusters=5, output_dir="./figures/output"
):
    """
    Cluster cities based on their crosswalk length distributions using
    Earth Mover's Distance with publication-quality visualizations.

    Parameters:
    lengths_dict: Dictionary mapping city names to arrays of crosswalk lengths
    num_clusters: Number of clusters to form
    output_dir: Directory to save publication-quality figures

    Returns:
    cluster_df: DataFrame showing city-to-cluster assignments
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of cities
    cities = list(lengths_dict.keys())
    n_cities = len(cities)

    # Create distance matrix
    dist_matrix = np.zeros((n_cities, n_cities))

    # Compute pairwise Wasserstein distances
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = wasserstein_distance(
                lengths_dict[cities[i]], lengths_dict[cities[j]]
            )
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Perform hierarchical clustering
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method="ward")
    clusters = fcluster(Z, num_clusters, criterion="maxclust")

    # Create a DataFrame to show which city belongs to which cluster
    raw_cluster_df = pd.DataFrame({"City": cities, "Cluster": clusters})
    
    # Calculate median length for each cluster to reorder them
    cluster_medians = []
    for cluster_id in range(1, num_clusters + 1):
        cities_in_cluster = raw_cluster_df[raw_cluster_df["Cluster"] == cluster_id]["City"].values
        cluster_lengths = np.concatenate([lengths_dict[city] for city in cities_in_cluster])
        cluster_medians.append((cluster_id, np.median(cluster_lengths)))
    
    # Sort clusters by median length (ascending)
    sorted_clusters = sorted(cluster_medians, key=lambda x: x[1])
    cluster_map = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters, 1)}
    
    # Remap cluster IDs so that cluster 1 has smallest median, cluster n has largest
    clusters = np.array([cluster_map[c] for c in clusters])
    cluster_df = pd.DataFrame({"City": cities, "Cluster": clusters})

    # --- Generate publication-quality dendrogram ---
    fig_dendro = plt.figure(figsize=(10, 7))

    # Calculate appropriate leaf font size based on number of cities
    leaf_font_size = max(6, min(9, 12 - 0.05 * n_cities))

    # Set color threshold for automatic coloring of clusters
    color_threshold = 0.7 * max(Z[:, 2])

    dendrogram(
        Z,
        labels=cities,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        color_threshold=color_threshold,
        above_threshold_color="gray",
    )

    plt.title(
        "Hierarchical Clustering of Cities by Crosswalk Length Distribution",
        fontweight="bold",
    )
    plt.xlabel("Cities", fontweight="bold")
    plt.ylabel("Ward Linkage Distance", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dendrogram_crosswalk_clustering.png", dpi=600)

    # --- Generate publication-quality distance heatmap ---
    fig_heatmap = plt.figure(figsize=(10, 8))

    # Use viridis colormap (perceptually uniform and colorblind-friendly)
    ax = sns.heatmap(
        dist_matrix,
        cmap="viridis",
        square=True,
        xticklabels=cities,
        yticklabels=cities,
        cbar_kws={"label": "Wasserstein Distance", "shrink": 0.8},
    )

    # Adjust tick labels
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)

    # Adjust tick font sizes if there are many cities
    if n_cities > 30:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.title("Pairwise Earth Mover's Distances Between Cities", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distance_heatmap_crosswalk.png", dpi=600)

    # --- Generate MDS visualization of city clusters ---
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    pos = mds.fit_transform(dist_matrix)

    # Create MDS plot with publication quality
    fig_mds = plt.figure(figsize=(10, 8))

    # Set up a sequential/gradual color palette
    palette = sns.color_palette("viridis", num_clusters)

    # Plot each cluster with its cities
    for cluster_id in range(1, num_clusters + 1):
        idx = np.where(clusters == cluster_id)[0]
        plt.scatter(
            pos[idx, 0],
            pos[idx, 1],
            s=120,
            c=[palette[cluster_id - 1]],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            edgecolors="w",
            linewidths=1,
        )

    # Add city labels with better positioning
    for i, (x, y) in enumerate(pos):
        # Get the cluster for this city
        city_cluster = clusters[i]
        plt.annotate(
            cities[i],
            (x, y),
            fontsize=9,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.title(
        "Multidimensional Scaling Projection of Cities by Crosswalk Distribution\n(Clusters Ordered by Median Length)",
        fontweight="bold",
    )
    plt.xlabel("Dimension 1", fontweight="bold")
    plt.ylabel("Dimension 2", fontweight="bold")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(
        title="Cluster",
        title_fontsize=11,
        markerscale=1.2,
        frameon=True,
        framealpha=0.9,
        edgecolor="dimgray",
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mds_visualization.png", dpi=600)

    # --- Generate distribution comparison plots ---
    # Create KDE plots for each cluster
    fig_dist = plt.figure(figsize=(10, 6))

    # Find common x-axis limits (5th and 95th percentiles to avoid outliers)
    all_lengths = np.concatenate([lengths_dict[city] for city in cities])
    x_min, x_max = np.percentile(all_lengths, [1, 99])

    for cluster_id in range(1, num_clusters + 1):
        # Get cities in this cluster
        cities_in_cluster = cluster_df[cluster_df["Cluster"] == cluster_id][
            "City"
        ].values

        # Combine all lengths from cities in this cluster
        cluster_lengths = np.concatenate(
            [lengths_dict[city] for city in cities_in_cluster]
        )

        # Plot KDE for this cluster with additional median information
        median_length = np.median(cluster_lengths)
        sns.kdeplot(
            cluster_lengths,
            label=f"Cluster {cluster_id} (n={len(cities_in_cluster)}, median={median_length:.1f}ft)",
            color=palette[cluster_id - 1],
            linewidth=2.5,
        )

    plt.xlim(x_min, x_max)
    plt.xlabel("Crosswalk Length (feet)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.title("Crosswalk Length Distributions by Cluster (Ordered by Median Length)", fontweight="bold")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(title="Clusters", frameon=True, framealpha=0.9, edgecolor="dimgray")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_distributions.png", dpi=600)

    return cluster_df, Z, dist_matrix


if __name__ == "__main__":
    df = pd.DataFrame(
        [get_measured_crosswalks_for_city(c) for c in tqdm(CITY_CODES)]
    ).sort_values("median", ascending=False, ignore_index=True)
    lengths_dict = {row["city"]: row["lengths"] for _, row in df.iterrows()}
    cluster_results, linkage_matrix, distance_matrix = (
        cluster_cities_by_crosswalk_distributions(lengths_dict, num_clusters=5)
    )
