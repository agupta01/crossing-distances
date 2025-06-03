import sys
import os
import warnings

warnings.filterwarnings("ignore")
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(".."))

# Set the modal environment
os.environ["MODAL_ENVIRONMENT"] = "sfo"  # just need one in here

from inference.utils import get_from_volume
import geopandas as gpd
import pandas as pd

all_city_codes = pd.read_csv("./data/envs_100_cities.csv")
CITY_CODES = list(set(all_city_codes["Code"].str.lower().tolist()) - {"anc"})

CROSSWALK_FILE_NAME = "refined_crosswalks_v2_1_3-grow30+15.geojson"


def get_measured_crosswalks_for_city(selected_city_code: str):
    selected_city = all_city_codes.loc[
        all_city_codes["Code"] == selected_city_code.upper()
    ].iloc[0]

    crosswalks = gpd.read_file(
        get_from_volume(
            "scratch", CROSSWALK_FILE_NAME, environment_name=selected_city_code
        )
    ).to_crs("EPSG:3857")
    lengths = crosswalks.geometry.length / 0.3695  # Convert meters to feet
    lengths = lengths[lengths > 3]  # Only keep those above 3 feet
    lengths = lengths[lengths < 155]  # Remove those above 155 feet

    # print(f"\nStatistics for {selected_city.Title}, {selected_city.State}:")
    # print(f"Mean: {lengths.mean():.2f} feet")
    # print(f"Median: {lengths.median():.2f} feet")
    # print(f"Standard Deviation: {lengths.std():.2f} feet")

    return {
        "city": selected_city.Title,
        "state": selected_city.State,
        "count": len(lengths.values),
        "mean": lengths.mean(),
        "median": lengths.median(),
        "stdev": lengths.std(),
        "lengths": lengths.values,
    }
