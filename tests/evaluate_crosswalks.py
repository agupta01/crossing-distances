import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy


def compute_bins(set1, set2, bins):
    """
    Create bins and count the number of values in each bin for two sets of values.

    Parameters:
        set1 (list or array): First set of values.
        set2 (list or array): Second set of values.
        bins (int): Number of bins to use.

    Returns:
        hist1 (array): Histogram counts for set1.
        hist2 (array): Histogram counts for set2.
    """
    # Combine the two sets to determine the overall range for binning
    combined = np.concatenate([set1, set2])
    bin_edges = np.linspace(np.min(combined), np.max(combined), bins + 1)

    # Compute histogram counts for each set using the same bin edges
    hist1, _ = np.histogram(set1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(set2, bins=bin_edges, density=True)
    return hist1, hist2


def kl_divergence(set1, set2, bins):
    """
    Compute the KL divergence between two sets of values.

    This function first creates histograms (bins) for the two datasets and then
    converts these counts into probability distributions. Finally, it calculates
    the KL divergence using scipy.stats.entropy.

    Parameters:
        set1 (list or array): First set of values.
        set2 (list or array): Second set of values.
        bins (int): Number of bins to use.

    Returns:
        kl_divergence (float): The KL divergence between the two sets.
    """
    # Step 1: Get histogram counts from the provided sets
    hist1, hist2 = compute_bins(set1, set2, bins)

    epsilon = 1e-10
    hist1 = np.clip(hist1, epsilon, None)
    hist2 = np.clip(hist2, epsilon, None)

    # Step 2: Convert counts to probability distributions
    prob1 = hist1 / np.sum(hist1)
    prob2 = hist2 / np.sum(hist2)

    # Step 3: Calculate the KL divergence using scipy.stats.entropy
    kl_div = entropy(prob1, prob2)

    return kl_div


def plot_length_histograms(
    before_lengths,
    after_lengths,
    paired,
    before_name,
    after_name,
    save_path,
):
    """Plot histograms of crosswalk lengths before and after."""
    # Filter before_lengths to those in paired
    before_lengths = before_lengths[paired.index_before]
    after_lengths = after_lengths[paired.index_after]

    plt.figure(figsize=(10, 6))
    plt.gcf().set_facecolor("#D3D3D3")
    plt.hist(
        before_lengths,
        bins=175,
        alpha=0.95,
        color="#7B3294",
        label=before_name,
    )
    plt.hist(
        after_lengths,
        bins=175,
        alpha=0.95,
        histtype="step",
        color="#4DFFFC",
        label=after_name,
    )
    # plt.xscale("log")
    plt.xlabel("Length (feet)")
    plt.ylabel("Frequency")
    plt.title("Crosswalk Length Distribution")
    plt.legend()
    plt.savefig(str(save_path / "length_distribution.png"))
    plt.close()


def plot_kl_divergence(before_lengths, after_lengths, paired, save_path):
    """Plot KL divergence over different bin counts."""
    # Filter before_lengths to those in paired
    before_lengths = before_lengths[paired.index_before]
    after_lengths = after_lengths[paired.index_after]

    bins = list(range(10, 5000, 10))
    divergences = [kl_divergence(before_lengths, after_lengths, b) for b in bins]

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=bins, y=divergences)
    plt.xlabel("# of bins")
    plt.ylabel("KL divergence")
    plt.title("Divergence over bin count")
    plt.savefig(str(save_path / "kl_divergence.png"))
    plt.close()


def filter_crosswalks_in_area(after_gdf, after_lengths, save_path, low, high):
    """Filter all crosswalks that are in the length range [low, high], in feet.

    Args:
        after_gdf (GeoDataFrame): "after" GeoDataSeries with length in feet
        save_path (Path): Directory to save filtered GeoJSONs
        low (float): Lower bound of length range
        high (float): Upper bound of length range
    """
    after_gdf["length_ft"] = after_lengths
    after_gdf = after_gdf[after_gdf["length_ft"] >= low]
    after_gdf = after_gdf[after_gdf["length_ft"] <= high]
    after_gdf = after_gdf.reset_index()[["index", "geometry"]]
    after_gdf.to_file(save_path / "after_filtered.shp")


def pair_crosswalks(before_gdf, after_gdf):
    before_gdf = (
        before_gdf.reset_index()[["index", "geometry"]]
        .rename(columns={"index": "index_before"})
        .to_crs("EPSG:3857")
    )
    after_gdf = (
        after_gdf.reset_index()[["index", "geometry"]]
        .rename(columns={"index": "index_after"})
        .to_crs("EPSG:3857")
    )

    joined = gpd.sjoin_nearest(
        before_gdf,
        after_gdf,
        how="right",
        distance_col="distance",
    )

    joined = joined.set_index("index_left").join(
        before_gdf.rename(columns={"geometry": "before_geom"}).set_index("index_before")
    )
    joined = joined.assign(after_length_ft=joined["geometry"].length / 0.3695).assign(
        before_length_ft=joined["before_geom"].length / 0.3695
    )
    joined = joined.loc[
        (joined["after_length_ft"] > 6)
        & (joined["before_length_ft"] > 6)
        & (joined["distance"] < 1)
    ]
    joined = joined.assign(
        difference=np.abs(joined["after_length_ft"] - joined["before_length_ft"])
    )
    return joined


def plot_paired_differences(joined, save_path):
    """Plot histogram of paired differences between crosswalks."""
    differences = joined["difference"]
    plt.figure(figsize=(10, 6))
    plt.hist(differences, bins=100)
    plt.xlabel("Length Difference (feet)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Length Differences")
    plt.savefig(str(save_path / "length_differences.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate crosswalk data")
    parser.add_argument(
        "--before-path", required=True, help="Path to before GeoJSON file"
    )
    parser.add_argument("--before", required=True, help="Name for before dataset")
    parser.add_argument(
        "--after-path", required=True, help="Path to after GeoJSON file"
    )
    parser.add_argument("--after", required=True, help="Name for after dataset")
    parser.add_argument("--plots-path", help="Directory to save plots")
    parser.add_argument("--save-path", help="Directory to save filtered crosswalks")

    args = parser.parse_args()

    # Read GeoJSON files
    before_gdf = gpd.read_file(args.before_path).to_crs("EPSG:3857")
    after_gdf = gpd.read_file(args.after_path).to_crs("EPSG:3857")

    # Extract lengths in feet (assuming the GeoJSON is in a projected CRS)
    before_lengths = before_gdf.geometry.length / 0.3695  # Convert meters to feet
    after_lengths = after_gdf.geometry.length / 0.3695

    # Calculate statistics
    print(f"\nStatistics for {args.before}:")
    print(f"Number of crosswalks: {len(before_gdf)}")
    print(f"Mean: {before_lengths.mean():.2f} feet")
    print(f"Median: {before_lengths.median():.2f} feet")
    print(f"Standard Deviation: {before_lengths.std():.2f} feet")

    print(f"\nStatistics for {args.after}:")
    print(f"Number of crosswalks: {len(after_gdf)}")
    print(f"Mean: {after_lengths.mean():.2f} feet")
    print(f"Median: {after_lengths.median():.2f} feet")
    print(f"Standard Deviation: {after_lengths.std():.2f} feet")

    # Calculate KL divergence with 1000 bins
    kl_div = kl_divergence(before_lengths.values, after_lengths.values, 1000)
    print(f"\nKL Divergence (1000 bins): {kl_div:.4f}")

    # Pair crosswalks
    paired = pair_crosswalks(before_gdf, after_gdf)

    # Print paired statistics for differences
    print(f"\nStatistics for paired differences:")
    print(f"Mean: {paired['difference'].mean():.2f} feet")
    print(f"Median: {paired['difference'].median():.2f} feet")
    print(f"Standard Deviation: {paired['difference'].std():.2f} feet")
    print(f"Max: {paired['difference'].max():.2f} feet")
    print(f"Min: {paired['difference'].min():.4f} feet")

    if args.save_path:
        save_path = Path(args.save_path)
        after_lengths.to_csv(save_path / "after_lengths.csv", index=True)
        # filter_crosswalks_in_area(after_gdf, after_lengths, save_path, 35, 40)

    # Generate plots if path is specified
    if args.plots_path:
        plots_path = Path(args.plots_path)
        plots_path.mkdir(parents=True, exist_ok=True)

        plot_length_histograms(
            before_lengths,
            after_lengths,
            paired,
            args.before,
            args.after,
            plots_path,
        )
        plot_kl_divergence(
            before_lengths.values,
            after_lengths.values,
            paired,
            plots_path,
        )
        plot_paired_differences(paired, plots_path)
        print(f"\nPlots saved to: {plots_path}")


if __name__ == "__main__":
    main()
