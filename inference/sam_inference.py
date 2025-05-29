import os

# Standard library imports
import csv
import gc
import json
import math
import pickle
import random
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

# Third-party imports
import cv2
import geopandas as gpd
import modal
import numpy as np
import osmnx as ox
import psutil
import scipy.interpolate
import supervision as sv
import torch
import wandb
from dotenv import load_dotenv
from shapely.geometry import LineString, Polygon, box, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid, explain_validity
import pandas as pd
from osmnx._errors import InsufficientResponseError as OSMnxInsufficientResponseError

# Set OSMnx network timeout globally
import logging
ox.settings.log_console = True
ox.settings.log_level = logging.DEBUG
ox.settings.requests_timeout = 600

load_dotenv()

# Project imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import (
    PRECISION, 
    get_crosswalk_id, 
    decode_crosswalk_id, 
    fuzzy_search_optimized,
    bounding_box_from_filename
)



# Configuration constants
# ===================================================================================================
# Environment setup
ENV_NAME = os.environ.get('MODAL_ENVIRONMENT', 'local') # Defines the environment (e.g., specific city for Modal, or 'local')
CITY_CODE = os.getenv("CITY_CODE") # Specific city code, often used for data paths and filtering

# Volume naming
INPUTS_VOLUME_NAME = f"crosswalk-data-{CITY_CODE}" # Name of the Modal Volume for input image data
OUTPUTS_VOLUME_NAME = f"{INPUTS_VOLUME_NAME}-results" # Name of the Modal Volume for storing results

# File paths
CITY_CODE_MAPPING_CSV_PATH = "data/city_code_mapping.csv" # Path to the CSV mapping city codes to OSM place names

# Image processing
IMAGE_SIZE = 1024 # Target size (width and height) for resizing input images before SAM processing
CV2_STRUCTURING_ELEMENT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Structuring element for morphological operations

# SAM2 model configuration
# Dictionary of parameters for the SAM2AutomaticMaskGenerator
MASK_GENERATOR_CONFIG = {
    "points_per_side": 16,          # Number of points to sample along each side of the image
    "pred_iou_thresh": 0.86,        # Prediction IoU threshold for filtering masks
    "stability_score_thresh": 0.91, # Stability score threshold for filtering masks
    "crop_n_layers": 1,             # Number of layers for multi-level cropping
    "min_mask_region_area": 500,    # Minimum area of a mask region to be considered
    "points_per_batch": 512,        # Number of points to process in a batch
    "box_nms_thresh": 0.78,         # Box NMS threshold for suppressing overlapping boxes
    "crop_overlap_ratio": 0.66,     # Overlap ratio for cropped regions
    "crop_n_points_downscale_factor": 2, # Downscale factor for points in cropped regions
    "output_mode": "binary_mask"    # Output mode for the mask generator (e.g., binary_mask, uncompressed_rle)
}

# Filtering parameters
MAX_RELATIVE_OVERLAP = 0.2   # Maximum relative overlap allowed between a SAM mask and OSM drivable areas
MAX_ABSOLUTE_OVERLAP = 15.0  # meters² - Maximum absolute overlap (in square meters) allowed with OSM drivable areas
MIN_CROSSWALK_AREA = 5.0     # meters² - Minimum real-world area for a mask to be considered a valid crosswalk
MAX_OVERLAP_PERCENT = 10.0   # Maximum allowed SAM overlap percentage for a building footprint to be included in fallback

# Modal setup
# ===================================================================================================
# Setup Modal app
app = modal.App(name="crossing-distance-sam2-inference")

# Configure Modal image
infer_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04", add_python="3.11")
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "CXX": "g++",
        "CC": "gcc",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0", 
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda"
    })
    .apt_install(
        "git", 
        "wget", 
        "python3-opencv", 
        "ffmpeg", 
        "build-essential", 
        "ninja-build", 
        "g++", 
        "libgl1-mesa-glx"
    )
    .pip_install("numpy")
    .run_commands("pip install torch torchvision torchaudio")
    .pip_install(
        "opencv-python==4.10.0.84",
        "pycocotools~=2.0.8",
        "matplotlib~=3.9.2",
        "supervision",
        "fiona",
        "geopandas",
        "shapely",
        "python-dotenv",
        "wandb",
        "osmnx",
    )
    .run_commands(
        # Clone SAM2
        "git clone https://git@github.com/facebookresearch/sam2.git",
        "cd sam2 && rm -rf build && rm -f ./sam2/*.so",
        # Install with CUDA flags set
        "cd sam2 && TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' FORCE_CUDA=1 pip install -v -e .",
        "cd sam2 && TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' FORCE_CUDA=1 pip install -v -e '.[dev]'",
        "cd sam2 && TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' FORCE_CUDA=1 pip install -v -e '.[notebooks]'",
        # Verify CUDA installation
        "nvcc --version || echo 'nvcc not found'",
        "echo $CUDA_HOME",
        "echo $PATH",
        "echo $LD_LIBRARY_PATH",
    )
    .add_local_python_source("utils")
    .add_local_python_source("sam2")
    .add_local_python_source("_remote_module_non_scriptable")
    .add_local_dir("../data/mapping", remote_path="/data")
)

# Setup Modal volumes
weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")
inputs_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, environment_name=ENV_NAME)
scratch_volume = modal.Volume.from_name("scratch", environment_name=ENV_NAME)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True, environment_name=ENV_NAME)

# Initialize thread pool for I/O operations
io_executor = ThreadPoolExecutor(max_workers=16)

# Utility Functions
# ===================================================================================================

# I/O utilities
# ---------------------------------------------------------------------------------------------------
def async_write(path, image):
    """Non-blocking image write using thread pool executor"""
    return io_executor.submit(cv2.imwrite, path, image)

def load_image(image_path):
    """Load image with error handling and auto-conversion"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def gc_collect():
    """Force garbage collection and release memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print memory usage
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"GPU memory: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB allocated, "
          f"{torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB reserved")

# Image processing utilities
# ---------------------------------------------------------------------------------------------------
def bgr_to_rgb(image):
    """Ensure consistent color conversion from BGR to RGB"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def rgb_to_bgr(image):
    """Ensure consistent color conversion from RGB to BGR"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# Geographic utilities
# ---------------------------------------------------------------------------------------------------
def get_utm_zone(longitude):
    """Get the UTM zone for a given longitude"""
    return int((longitude + 180) // 6) + 1

def get_utm_projection(lon, lat):
    """Get the EPSG code for UTM projection at given coordinates"""
    utm_zone = get_utm_zone(lon)
    northern = lat >= 0
    epsg_code = 32600 + utm_zone if northern else 32700 + utm_zone
    return f"EPSG:{epsg_code}"

# Data Processing Classes
# ===================================================================================================

class DataLoader:
    """Class to handle data loading and preparation operations"""
    
    @staticmethod
    def load_and_filter_images(csv_path: str, image_dir: str) -> list:
        """
        Combined function that loads coordinates from CSV and finds matching images.
        Includes caching for fuzzy search results to improve performance.
        Returns a list of valid image paths and tracks original coordinates for missing images.
        """
        # Check if cache file exists
        cache_dir = "/outputs/search_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"fuzzy_search_cache.pkl")
        
        # Get stats for current data to detect changes
        csv_mtime = os.path.getmtime(csv_path)
        with open(csv_path, 'r') as f:
            csv_line_count = sum(1 for _ in f)
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_count = len(image_files)
        
        # Create a unique hash for the current data state
        data_state_hash = f"{csv_path}_{csv_mtime}_{csv_line_count}_{image_dir}_{image_count}"
        
        # Check if cache exists and is valid
        cache_valid = False
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    if cache.get('data_state_hash') == data_state_hash:
                        print("Using cached fuzzy search results")
                        cache_valid = True
                        valid_images = cache.get('valid_images', [])
                        return valid_images
                    else:
                        print("Cache exists but data has changed - rebuilding")
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Continue with normal processing if cache loading fails
        
        # If cache is invalid or doesn't exist, proceed with normal processing
        valid_images = []
        coordinate_to_id = {}
        missing_coordinates = []
        
        # Add counters to track match types
        exact_match_count = 0
        fuzzy_match_count = 0
        
        # Index all images by ID for faster lookup
        image_id_map = {}
        all_crosswalk_ids = []  # Store all IDs for fuzzy matching later
        for filename in os.listdir(image_dir):
            if filename.endswith('.json'):
                continue
            
            filename_without_ext = os.path.splitext(filename)[0]
            parts = filename_without_ext.split('_', 1)
            
            if len(parts) == 2 and parts[0] == "crosswalk":
                crosswalk_id = parts[1]
                image_id_map[crosswalk_id] = os.path.join(image_dir, filename)
                all_crosswalk_ids.append(crosswalk_id)  # Add to list for fuzzy matching
        
        # Process coordinates from CSV
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                longitude = float(row['x'])
                latitude = float(row['y'])
                orig_coords = (latitude, longitude)
                
                # Round to PRECISION to match filename encoding
                rounded_lat = round(latitude, PRECISION)
                rounded_lon = round(longitude, PRECISION)
                
                try:
                    # Generate crosswalk ID
                    crosswalk_id = get_crosswalk_id(rounded_lat, rounded_lon)
                    coordinate_to_id[orig_coords] = crosswalk_id
                    
                    # Check if image exists with this ID
                    if crosswalk_id in image_id_map:
                        valid_images.append(image_id_map[crosswalk_id])
                        exact_match_count += 1  # Direct ID match
                    else:
                        # Try alternative formats
                        expected_filename = f"crosswalk_{crosswalk_id}"
                        found = False
                        
                        for filename in os.listdir(image_dir):
                            if filename.endswith(('.jpeg', '.jpg', '.png')) and expected_filename in filename:
                                valid_images.append(os.path.join(image_dir, filename))
                                found = True
                                exact_match_count += 1  # Alternative format match is still exact
                                break
                        
                        # If exact match not found, try fuzzy matching
                        if not found and all_crosswalk_ids:
                            # Use fuzzy matching to find the closest crosswalk ID
                            closest_id = fuzzy_search_optimized(crosswalk_id, all_crosswalk_ids)
                            
                            # Calculate character-level matching error (number of mismatched characters)
                            char_errors = sum(1 for a, b in zip(crosswalk_id, closest_id) if a != b)
                            error_pct = char_errors / len(crosswalk_id)
                            
                            # Decode the coordinates from both IDs to calculate geographic distance
                            orig_coords_latlon = decode_crosswalk_id(crosswalk_id)
                            fuzzy_coords_latlon = decode_crosswalk_id(closest_id)
                            
                            # Calculate Haversine distance between the coordinates
                            lat1, lon1 = math.radians(orig_coords_latlon.lat), math.radians(orig_coords_latlon.long)
                            lat2, lon2 = math.radians(fuzzy_coords_latlon.lat), math.radians(fuzzy_coords_latlon.long)
                            
                            # Haversine formula
                            dlon = lon2 - lon1
                            dlat = lat2 - lat1
                            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                            c = 2 * math.asin(math.sqrt(a))
                            
                            # Earth radius in meters
                            r = 6371000
                            ground_distance = c * r  # Distance in meters
                            
                            # Only use match if ground distance is less than 1 meter
                            if ground_distance <= 1.0:
                                # Get the path to the image with this ID
                                matched_img_path = image_id_map[closest_id]
                                valid_images.append(matched_img_path)
                                found = True
                                fuzzy_match_count += 1  # Count successful fuzzy matches
                                
                                print(f"Found fuzzy match for {crosswalk_id}:")
                                print(f"  Matched ID: {closest_id}")
                                print(f"  Character errors: {char_errors}/{len(crosswalk_id)} ({error_pct:.2%})")
                                print(f"  Ground distance: {ground_distance:.2f}m")
                            else:
                                print(f"Rejected fuzzy match (distance > 1m):")
                                print(f"  Original ID: {crosswalk_id}")
                                print(f"  Matched ID: {closest_id}")
                                print(f"  Ground distance: {ground_distance:.2f}m")
                        
                        if not found:
                            missing_coordinates.append((orig_coords, crosswalk_id))
                except AssertionError as e:
                    print(f"Skipping invalid coordinate ({rounded_lat}, {rounded_lon}): {e}")
                    missing_coordinates.append((orig_coords, None))
        
        # Print detailed statistics
        print(f"Processed {len(coordinate_to_id)} valid coordinates from CSV")
        print(f"Found {len(valid_images)} matching images")
        print(f"  - Exact matches: {exact_match_count} ({exact_match_count / len(valid_images) * 100:.1f}% of total)")
        print(f"  - Fuzzy matches: {fuzzy_match_count} ({fuzzy_match_count / len(valid_images) * 100:.1f}% of total)")
        print(f"  - Fuzzy matching added {fuzzy_match_count} images ({fuzzy_match_count / (exact_match_count or 1) * 100:.1f}% increase)")
        print(f"Missing {len(missing_coordinates)} coordinates")
        
        # Output details of missing coordinates
        if missing_coordinates:
            print("\nSample of missing coordinates (first 10):")
            for i, (coords, crosswalk_id) in enumerate(missing_coordinates[:10]):
                lat, lon = coords
                if crosswalk_id:
                    expected_filename = f"crosswalk_{crosswalk_id}.jpg"
                    print(f"  {i+1}. Coordinates: {lat}, {lon}")
                    print(f"     ID: {crosswalk_id}")
                    print(f"     Expected filename: {expected_filename}")
                else:
                    print(f"  {i+1}. Coordinates: {lat}, {lon} (couldn't generate valid ID)")
            
            # Save all missing coordinates to file
            missing_file = os.path.join(image_dir, "missing_coordinates.csv")
            with open(missing_file, 'w') as f:
                f.write("Latitude,Longitude,CrosswalkID,ExpectedFilename,ClosestMatch,ErrorChars,GroundDistanceM\n")
                for coords, crosswalk_id in missing_coordinates:
                    lat, lon = coords
                    if crosswalk_id:
                        # Try to find closest match for reporting purposes
                        closest_match = ""
                        error_chars = ""
                        ground_distance = ""
                        
                        if all_crosswalk_ids:
                            closest_id = fuzzy_search_optimized(crosswalk_id, all_crosswalk_ids)
                            error_chars = sum(1 for a, b in zip(crosswalk_id, closest_id) if a != b)
                            
                            # Calculate ground distance
                            orig_coords_latlon = decode_crosswalk_id(crosswalk_id)
                            fuzzy_coords_latlon = decode_crosswalk_id(closest_id)
                            
                            lat1, lon1 = math.radians(orig_coords_latlon.lat), math.radians(orig_coords_latlon.long)
                            lat2, lon2 = math.radians(fuzzy_coords_latlon.lat), math.radians(fuzzy_coords_latlon.long)
                            
                            dlon = lon2 - lon1
                            dlat = lat2 - lat1
                            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                            c = 2 * math.asin(math.sqrt(a))
                            
                            r = 6371000
                            ground_distance = c * r
                            
                            closest_match = closest_id
                        
                        f.write(f"{lat},{lon},{crosswalk_id},crosswalk_{crosswalk_id}.jpg,{closest_match},{error_chars},{ground_distance}\n")
                    else:
                        f.write(f"{lat},{lon},N/A,N/A,,,\n")
            print(f"\nSaved all {len(missing_coordinates)} missing coordinates to {missing_file}")
        
        # Add a summary of match statistics to the missing coordinates file
        if os.path.exists(missing_file):
            with open(missing_file, 'a') as f:
                f.write("\n\n--- Match Statistics ---\n")
                f.write(f"Total Valid Images: {len(valid_images)}\n")
                f.write(f"Exact Matches: {exact_match_count}\n")
                f.write(f"Fuzzy Matches: {fuzzy_match_count}\n")
                f.write(f"Exact Match Percentage: {exact_match_count / len(valid_images) * 100:.1f}%\n")
                f.write(f"Fuzzy Match Percentage: {fuzzy_match_count / len(valid_images) * 100:.1f}%\n")
                f.write(f"Fuzzy Matching Improvement: {fuzzy_match_count / (exact_match_count or 1) * 100:.1f}%\n")
                f.write(f"Missing Coordinates: {len(missing_coordinates)}\n")
        
        # Cache the results before returning
        try:
            cache_data = {
                'data_state_hash': data_state_hash,
                'valid_images': valid_images,
                'stats': {
                    'exact_match_count': exact_match_count,
                    'fuzzy_match_count': fuzzy_match_count,
                    'missing_count': len(missing_coordinates),
                    'total_valid': len(valid_images)
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Fuzzy search results cached to {cache_file}")
        except Exception as e:
            print(f"Error caching fuzzy search results: {e}")
        
        return valid_images

    @staticmethod
    def load_city_code_mapping_from_csv(csv_filepath):
        """Loads city code to OSM place mapping from a CSV file."""
        city_code_to_place_mapping = {}
        try:
            with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile: # Use utf-8-sig to handle BOM if present
                reader = csv.DictReader(csvfile)
                for row in reader:
                    city_code = row['Code'].lower()
                    city_name = row['City'] if row['City'] else None
                    county_name = row['County'] if row['County'] else None
                    state_name = row['State'] if row['State'] else None # Handle empty state
                    country_name = row['Country']

                    place_dict = {'country': country_name} # Base dict
                    if city_name: # Add city only if it's not empty
                        place_dict['city'] = city_name
                    if county_name: # Add county only if it's not empty
                        place_dict['county'] = county_name
                    if state_name: # Add state only if it's not empty
                        place_dict['state'] = state_name
                    

                    city_code_to_place_mapping[city_code] = place_dict
            print(f"City code mapping loaded from CSV: {csv_filepath}")
            return city_code_to_place_mapping
        except FileNotFoundError:
            print(f"Error: CSV mapping file not found at: {csv_filepath}")
            return {} # Return empty dict in case of error
        except Exception as e:
            print(f"Error loading city code mapping from CSV: {e}")
            return {} # Return empty dict in case of error

    @staticmethod
    def load_city_osm_data(city_code, city_code_to_place_mapping):
        """
        Loads city-wide OSM drivable edges and spatial index.
        Now includes caching for faster repeated lookups.
        """

        try:
            # First try to load from cache
            cache_dir = "/outputs/osm_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{city_code}_osm_data.pkl")
            
            if os.path.exists(cache_file):
                print(f"Loading OSM data from cache for {city_code}...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify CRS is still correct and check if cache has all required fields
                    if (cached_data['city_edges_gdf'].crs == "EPSG:4326" and
                        'non_drivable_gdf' in cached_data and 
                        'non_drivable_sindex' in cached_data):
                        print(f"OSM data loaded from cache for {city_code}")
                        return cached_data['city_edges_gdf'], cached_data['city_edges_sindex'], cached_data['non_drivable_gdf'], cached_data['non_drivable_sindex']
                    else:
                        if 'non_drivable_gdf' not in cached_data:
                            print(f"Outdated cache detected (missing non-drivable features). Rebuilding OSM data.")
                        else:
                            print(f"Cached OSM data had invalid CRS, rebuilding...")
            
            print(f"Querying OSM for drivable edges in {city_code}...")
            place = city_code_to_place_mapping.get(city_code) # Use the mapping
            if place is None: # Handle unknown city code
                raise ValueError(f"City code '{city_code}' not found in city code mapping CSV.")

            # Tags for drivable roads
            drivable_tags = {
                "highway": True,
                "boundary": True,
                "landuse": True,
                "leisure": True
            }

            # Tags for non-drivable areas (buildings, parks, etc.)
            non_drivable_tags = {
                "building": True,
                "amenity": ["parking"],
                "parking": ["surface"],
                "leisure": ["park", "stadium", "miniature_golf", "pitch", "playground"],
                "landuse": ["grass", "brownfield"],
                "surface": ["grass"],
                "place": ["square"],
            }

            # Get drivable road data
            city_edges_gdf = ox.features_from_place(place, tags=drivable_tags)

            # Get non-drivable area data
            try:
                non_drivable_gdf = ox.features_from_place(place, tags=non_drivable_tags)
            except OSMnxInsufficientResponseError as e_main:
                print(f"Warning: Main query for non-drivable features for {city_code} failed: {e_main}. Attempting subdivision...")
                
                # Get the polygon for the place to subdivide
                try:
                    place_gdf = ox.geocode_to_gdf(place)
                    place_polygon = place_gdf.unary_union # Dissolve to a single polygon/multipolygon
                except Exception as e_geocode:
                    print(f"Error: Could not geocode place '{place}' for subdivision: {e_geocode}. Non-drivable features will be considered unloaded.")
                    # If geocoding fails, subdivision is not possible. Re-raise the main error.
                    raise e_main

                all_sub_gdfs = []
                minx, miny, maxx, maxy = place_polygon.bounds
                x_step = (maxx - minx) / 2 # Dividing into 2 parts along x-axis
                y_step = (maxy - miny) / 2 # Dividing into 2 parts along y-axis

                sub_polygons_to_query = []
                for i in range(2): # 0, 1 for x-axis
                    for j in range(2): # 0, 1 for y-axis
                        # Create a bounding box for the sub-region
                        sub_poly_bbox = box(minx + i * x_step, 
                                            miny + j * y_step, 
                                            minx + (i + 1) * x_step, 
                                            miny + (j + 1) * y_step)
                        # Only query sub-polygons that actually intersect the main place_polygon
                        # and ensure the intersection itself is a valid Polygon or MultiPolygon (not empty or just a line)
                        intersection_with_place = sub_poly_bbox.intersection(place_polygon)
                        if not intersection_with_place.is_empty and isinstance(intersection_with_place, (Polygon, MultiPolygon)):
                            sub_polygons_to_query.append(intersection_with_place)
                
                if not sub_polygons_to_query:
                    print(f"Warning: No valid intersecting sub-polygons created for {city_code} after attempting to subdivide. Re-raising original error.")
                    raise e_main

                print(f"Querying {len(sub_polygons_to_query)} sub-regions for non-drivable features in {city_code}...")
                sub_queries_failed_count = 0
                successful_sub_queries = 0
                for idx, sub_poly_geom in enumerate(sub_polygons_to_query):
                    try:
                        print(f"  Querying sub-region {idx+1}/{len(sub_polygons_to_query)}...")
                        # Use the actual intersection geometry for the query
                        sub_gdf = ox.features_from_polygon(sub_poly_geom, tags=non_drivable_tags)
                        if not sub_gdf.empty:
                            all_sub_gdfs.append(sub_gdf)
                            successful_sub_queries +=1
                            print(f"    Found {len(sub_gdf)} features in sub-region {idx+1}.")
                        else:
                            print(f"    No features found in sub-region {idx+1}.")
                    except ox._errors.InsufficientResponseError as e_sub:
                        print(f"    Warning: Sub-query {idx+1} for {city_code} failed (Overpass API error): {e_sub}. Skipping this sub-region.")
                        sub_queries_failed_count += 1
                    except Exception as e_sub_other:
                        print(f"    Warning: Unexpected error in sub-query {idx+1} for {city_code}: {e_sub_other}. Skipping this sub-region.")
                        sub_queries_failed_count += 1
                
                if all_sub_gdfs:
                    print(f"Concatenating features from {len(all_sub_gdfs)} successful sub-queries ({successful_sub_queries} out of {len(sub_polygons_to_query)}).")
                    non_drivable_gdf = pd.concat(all_sub_gdfs, ignore_index=True)
                    if 'element_type' in non_drivable_gdf.columns and 'osmid' in non_drivable_gdf.columns:
                        # Keep track of original count before dropping duplicates
                        original_feature_count = len(non_drivable_gdf)
                        non_drivable_gdf = non_drivable_gdf.drop_duplicates(subset=['element_type', 'osmid'], keep='first')
                        deduplicated_count = len(non_drivable_gdf)
                        print(f"    Deduplicated features from {original_feature_count} to {deduplicated_count}.")
                    else: # Fallback if osmid or element_type not present, though unlikely for OSMnx features
                        print("    Warning: 'element_type' or 'osmid' not in columns, cannot deduplicate precisely.")
                    print(f"Total {len(non_drivable_gdf)} non-drivable features after subdivision and concatenation.")
                elif sub_queries_failed_count == len(sub_polygons_to_query) and not all_sub_gdfs: # All sub-queries failed
                    print(f"Error: All {len(sub_polygons_to_query)} sub-queries for non-drivable features failed for {city_code}. Re-raising original error from main query.")
                    raise e_main 
                else: # Some sub-queries might have succeeded but returned no data, or some failed but others succeeded with no data.
                    print(f"Warning: No non-drivable features collected after subdivision for {city_code}. This might be due to all successful sub-queries returning no features, or a mix of failures and no-data successes. Non-drivable GDF will be empty.")
                    non_drivable_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            except Exception as e_other_initial: 
                # This catches other errors from the *initial* ox.features_from_place(place, ...) call,
                # or if e_main was re-raised from the subdivision block.
                print(f"Error: Failed to retrieve non-drivable features for {city_code} due to: {e_other_initial}. Non-drivable features will be considered unloaded.")
                # To ensure the script exits as per user requirement, we must re-raise the error.
                # Or, ensure non_drivable_gdf is in a state that run_inference() will treat as a failure.
                # The most direct way to ensure failure is to re-raise.
                raise e_other_initial

            # Verification and enforcement of CRS
            if city_edges_gdf.crs is None:
                print("Warning: city_edges_gdf CRS was None, setting to EPSG:4326")
                city_edges_gdf.crs = "EPSG:4326"
            elif city_edges_gdf.crs != "EPSG:4326":
                print(f"Warning: city_edges_gdf CRS was not EPSG:4326, reprojecting from {city_edges_gdf.crs} to EPSG:4326")
                city_edges_gdf = city_edges_gdf.to_crs("EPSG:4326")

            if non_drivable_gdf.crs is None:
                print("Warning: non_drivable_gdf CRS was None, setting to EPSG:4326")
                non_drivable_gdf.crs = "EPSG:4326"
            elif non_drivable_gdf.crs != "EPSG:4326":
                print(f"Warning: non_drivable_gdf CRS was not EPSG:4326, reprojecting from {non_drivable_gdf.crs} to EPSG:4326")
                non_drivable_gdf = non_drivable_gdf.to_crs("EPSG:4326")

            # Filter drivable edges
            pedestrian_road_types = [
                'primary', 'primary_link',
                'secondary', 'secondary_link', 
                'tertiary', 'tertiary_link',
                'unclassified', 'residential',
            ]
            
            # Pre-compute common conditions to improve speed
            has_highway = city_edges_gdf['highway'].notna()
            is_pedestrian_road = city_edges_gdf['highway'].isin(pedestrian_road_types)
            
            # More efficient filtering based on column presence
            filter_conditions = [has_highway, is_pedestrian_road]
            
            # Only add filters for columns that exist
            column_filters = {
                'service': lambda df: ~df['service'].isin(['parking_aisle', 'driveway', 'drive-through', 'parking']), 
                'access': lambda df: ~df['access'].isin(['private', 'no']),
                'amenity': lambda df: ~df['amenity'].isin(['parking', 'parking_entrance', 'parking_space']),
                'landuse': lambda df: ~df['landuse'].isin(['park', 'garden', 'forest', 'recreation_ground', 'grass', 'meadow']),
                'leisure': lambda df: ~df['leisure'].isin(['park', 'garden', 'nature_reserve', 'playground', 'pitch', 'golf_course']),
                'boundary': lambda df: df['boundary'].isna(),
                'footway': lambda df: ~df['footway'].isin(['sidewalk', 'access_aisle']) | (df['footway'].isna()),
                'area': lambda df: (df['area'] != 'yes') | df['area'].isna()
            }
            
            for col, filter_func in column_filters.items():
                if col in city_edges_gdf.columns:
                    filter_conditions.append(filter_func(city_edges_gdf))
            
            # Combine all filters more efficiently
            mask = np.logical_and.reduce(filter_conditions)
            city_edges_gdf = city_edges_gdf[mask]

            # Create spatial index for drivable edges
            city_edges_sindex = city_edges_gdf.sindex
            
            # Filter non-drivable areas (only keep buildings, parks, etc.)
            filter_conditions = []

            # Check if each column exists before filtering on it
            if 'building' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['building'].notna())
            if 'amenity' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['amenity'].isin(['parking']))
            if 'parking' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['parking'].isin(['surface']))
            if 'leisure' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['leisure'].isin(['park', 'stadium', 'miniature_golf', 'pitch', 'playground']))
            if 'landuse' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['landuse'].isin(['grass', 'brownfield']))
            if 'surface' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['surface'].isin(['grass']))
            if 'place' in non_drivable_gdf.columns:
                filter_conditions.append(non_drivable_gdf['place'].isin(['square']))

            # Only apply filtering if there are conditions
            if filter_conditions:
                mask = np.logical_or.reduce(filter_conditions)
                non_drivable_gdf = non_drivable_gdf[mask]
            else:
                # If none of the columns exist, return an empty GeoDataFrame
                non_drivable_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            # Create spatial index for non-drivable areas
            non_drivable_sindex = non_drivable_gdf.sindex
            
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'city_edges_gdf': city_edges_gdf,
                    'city_edges_sindex': city_edges_sindex,
                    'non_drivable_gdf': non_drivable_gdf,
                    'non_drivable_sindex': non_drivable_sindex
                }, f)
            
            print(f"OSM data queried, filtered, and cached for {city_code}.")
            return city_edges_gdf, city_edges_sindex, non_drivable_gdf, non_drivable_sindex

        except Exception as e:
            print(f"Error querying OSM data: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return gpd.GeoDataFrame(), None, gpd.GeoDataFrame(), None

    @staticmethod
    def clear_fuzzy_search_cache():
        """Clear the fuzzy search cache"""
        cache_dir = "/outputs/search_cache"
        if os.path.exists(cache_dir):
            cache_file = os.path.join(cache_dir, "fuzzy_search_cache.pkl")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"Fuzzy search cache cleared: {cache_file}")
                except Exception as e:
                    print(f"Error clearing fuzzy search cache: {e}")
            else:
                print("No fuzzy search cache found to clear")
        else:
            print("Cache directory does not exist")

# Initialize the DataLoader for convenience
data_loader = DataLoader()

class OSMProcessor:
    """Class to handle OSM data processing and manipulation"""

    HIGHWAY_BUFFER_MAPPING = {
        # Wider Roads
        'primary': 2.0,
        'primary_link': 1.7,
        'secondary': 1.7,
        'secondary_link': 1.5,
        # Medium Roads (Baseline)
        'tertiary': 1.5,
        # Narrower Roads
        'tertiary_link': 1.2,
        'unclassified': 1.2,
        'residential': 1.2,
        # Default for any other types or if tag is missing/unexpected
        '_default_': 1.2
    }
    
    QUERY_EXPANSION_METERS = 5.0 

    @staticmethod
    def get_osm_drivable_edges(bbox_polygon, city_edges_gdf, city_edges_sindex):
        """Get drivable edges from OSM within a bounding box with dynamic buffering."""
        
        centroid = bbox_polygon.centroid
        utm_crs = get_utm_projection(centroid.x, centroid.y)
        
        bbox_utm_ser = gpd.GeoSeries([bbox_polygon], crs="EPSG:4326").to_crs(utm_crs)
        expanded_bbox_utm = bbox_utm_ser.buffer(OSMProcessor.QUERY_EXPANSION_METERS).iloc[0]
        expanded_bbox_for_query_wgs84 = gpd.GeoSeries([expanded_bbox_utm], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

        possible_matches_index = list(city_edges_sindex.intersection(expanded_bbox_for_query_wgs84.bounds))
        possible_matches = city_edges_gdf.iloc[possible_matches_index]
        
        precise_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
        
        original_osm_edges = precise_matches['geometry'].tolist() 
        buffered_osm_edges_final_wgs84 = []

        if not precise_matches.empty:
            precise_matches_utm = precise_matches.to_crs(utm_crs) 
            
            buffered_geometries_in_utm_list = []
            for index, row in precise_matches_utm.iterrows():
                highway_tag_value = row.get('highway') 
                
                lookup_key = '_default_' 
                if isinstance(highway_tag_value, str):
                    lookup_key = highway_tag_value
                elif isinstance(highway_tag_value, list) and highway_tag_value:
                    first_element = highway_tag_value[0]
                    if isinstance(first_element, str):
                        lookup_key = first_element

                buffer_width_meters = OSMProcessor.HIGHWAY_BUFFER_MAPPING.get(
                    lookup_key, 
                    OSMProcessor.HIGHWAY_BUFFER_MAPPING['_default_'] 
                )
                
                if row.geometry and not row.geometry.is_empty:
                    buffered_geom = row.geometry.buffer(buffer_width_meters)
                    if not buffered_geom.is_empty:
                        buffered_geometries_in_utm_list.append(buffered_geom)
            
            if buffered_geometries_in_utm_list:
                buffered_geoms_utm_gs = gpd.GeoSeries(buffered_geometries_in_utm_list, crs=utm_crs)
                buffered_geoms_wgs84_gs = buffered_geoms_utm_gs.to_crs("EPSG:4326")
                buffered_osm_edges_final_wgs84 = [geom for geom in buffered_geoms_wgs84_gs if not geom.is_empty]
    
        return original_osm_edges, buffered_osm_edges_final_wgs84

    @staticmethod
    def get_osm_non_drivable_features(bbox_polygon, non_drivable_gdf, non_drivable_sindex):
        """Get non-drivable features (buildings, parks, etc.) from OSM within a bounding box"""
        if non_drivable_gdf.empty or non_drivable_sindex is None:
            return []
            
        buffer_distance = 0.5
        
        centroid = bbox_polygon.centroid
        utm_crs = get_utm_projection(centroid.x, centroid.y)
        
        possible_matches_index = list(non_drivable_sindex.intersection(bbox_polygon.bounds))
        possible_matches = non_drivable_gdf.iloc[possible_matches_index]
        
        precise_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
        
        non_drivable_features = []
        if not precise_matches.empty:
            precise_matches_utm = precise_matches.to_crs(utm_crs)
            
            buffered_geoms_utm_list = []
            for geom in precise_matches_utm.geometry:
                if geom and not geom.is_empty:
                    buffered_geom = geom.buffer(buffer_distance)
                    if buffered_geom and not buffered_geom.is_empty:
                        buffered_geoms_utm_list.append(buffered_geom)
            
            if buffered_geoms_utm_list:
                buffered_geoms_utm_gs = gpd.GeoSeries(buffered_geoms_utm_list, crs=utm_crs)
                buffered_geoms_wgs84_gs = buffered_geoms_utm_gs.to_crs("EPSG:4326")
                non_drivable_features = [g for g in buffered_geoms_wgs84_gs if g and not g.is_empty]
        
        return non_drivable_features

    @staticmethod
    def create_osm_mask(geometries, image_size, bbox, is_lines=False):
        """Create binary mask of OSM edges in image coordinates (reverted to original logic)."""
        if not geometries:
            return np.zeros(image_size, dtype=np.uint8)
            
        osm_mask = np.zeros(image_size, dtype=np.uint8)
        width, height = image_size
        
        # Original scale and offset calculations
        # Assumes bbox is (min_lon, max_lat, max_lon, min_lat)
        # Original x_scale was width / (bbox[0] - bbox[2]), which implies bbox[0] > bbox[2] for positive scale, or results in a flip.
        # Original y_scale was height / (bbox[1] - bbox[3]), which implies bbox[1] > bbox[3] for positive scale.
        # Original x_offset = bbox[2]
        # Original y_offset = bbox[3]
        
        # To prevent division by zero if bbox[0] == bbox[2] or bbox[1] == bbox[3]
        # These are signs of a degenerate bounding box.
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            print(f"Warning: Degenerate bounding box encountered in create_osm_mask (original logic): {bbox}")
            return osm_mask # Return empty mask

        x_scale = width / (bbox[0] - bbox[2]) 
        y_scale = height / (bbox[1] - bbox[3])
        x_offset = bbox[2]
        y_offset = bbox[3]
        
        def transform_point(x, y): # x is lon, y is lat
            pixel_x = int((x - x_offset) * x_scale)
            pixel_y = int((y - y_offset) * y_scale)
            return pixel_x, pixel_y

        def process_line(line):
            if line.is_empty:
                return
            coords = []
            for x_geo, y_geo in line.coords:
                pixel_x, pixel_y = transform_point(x_geo, y_geo)
                coords.append((pixel_x, pixel_y))
            if len(coords) >= 2:
                coords_array = np.array(coords, dtype=np.int32)
                # Original: Draw lines between consecutive points individually
                for i in range(len(coords_array) - 1):
                    cv2.line(osm_mask, 
                             (coords_array[i][0], coords_array[i][1]),
                             (coords_array[i+1][0], coords_array[i+1][1]), 
                             1, thickness=2)
        
        def process_polygon(polygon):
            if polygon.is_empty:
                 return
            # Original area check threshold was 1e-9
            if polygon.area < 1e-9: 
                return
                
            ext_coords_pixel = []
            for x_geo, y_geo in polygon.exterior.coords:
                pixel_x, pixel_y = transform_point(x_geo, y_geo)
                ext_coords_pixel.append((pixel_x, pixel_y))
            
            if len(ext_coords_pixel) > 2:
                cv2.fillPoly(osm_mask, [np.array(ext_coords_pixel, dtype=np.int32)], 1)
                
            for interior in polygon.interiors:
                int_coords_pixel = []
                for x_geo, y_geo in interior.coords:
                    pixel_x, pixel_y = transform_point(x_geo, y_geo)
                    int_coords_pixel.append((pixel_x, pixel_y))
                if len(int_coords_pixel) > 2:
                    cv2.fillPoly(osm_mask, [np.array(int_coords_pixel, dtype=np.int32)], 0)
        
        # Original geometry processing loop structure
        for geom in geometries:
            if isinstance(geom, LineString):
                # The original logic always processed LineStrings with process_line,
                # regardless of is_lines flag in this specific loop.
                # The is_lines flag in the function signature might have been intended for other uses
                # or for how callers decide what geometries to pass.
                process_line(geom) 
            elif isinstance(geom, MultiPolygon):
                for p in geom.geoms: # Using 'p' as in some earlier versions
                    process_polygon(p)
            elif isinstance(geom, Polygon):
                process_polygon(geom)

        return osm_mask

class MaskProcessor:
    """Class for processing and managing segmentation masks"""
    
    @staticmethod
    def process_masks_parallel(valid_masks, original_size):
        """Process multiple masks in parallel using NumPy vectorization when possible"""
        if not valid_masks:
            return np.zeros(original_size[::-1], dtype=np.uint8)
            
        # Initialize a single array for all masks
        final_mask = np.zeros(original_size[::-1], dtype=np.uint8)
        
        # Group operations for vectorization
        mask_batch = [mask_data["segmentation"] for mask_data in valid_masks]
        
        # Process masks in batches to avoid memory issues
        batch_size = min(8, len(mask_batch))
        for i in range(0, len(mask_batch), batch_size):
            batch = mask_batch[i:i+batch_size]
            
            # Vectorized resizing
            for mask in batch:
                # Faster resize using direct cv2 call for binary masks
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                original_size[::-1], 
                                interpolation=cv2.INTER_NEAREST)
                
                # Faster mask combination with bitwise OR
                final_mask = cv2.bitwise_or(final_mask, resized_mask)
        
        # Apply post-processing in a single pipeline for efficiency
        final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (3, 3), 0)
        _, final_mask = cv2.threshold(final_mask, 0.5, 1, cv2.THRESH_BINARY)
        
        # Use pre-computed structuring element
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, CV2_STRUCTURING_ELEMENT)
        final_mask = cv2.medianBlur(final_mask, 3)
        
        # Final blur and threshold in one step
        final_mask = cv2.GaussianBlur(final_mask, (5,5), 1)
        final_mask = (final_mask > 0.1).astype(np.uint8)
        
        return final_mask
    
    @staticmethod
    def is_mask_coverage_insufficient(mask, image_size, min_coverage_percent=1.0):
        """
        Detect if mask coverage is insufficient by checking if it covers at least some 
        minimal percentage of the image. Returns True if mask is insufficient.
        
        Args:
            mask: The binary mask (numpy array)
            image_size: Tuple (width, height) of the original image
            min_coverage_percent: Minimum percentage of image that should be covered by masks
            
        Returns:
            bool: True if mask coverage is insufficient
        """
        # Calculate image area and masked area
        total_pixels = image_size[0] * image_size[1]
        masked_pixels = np.count_nonzero(mask)
        
        # Calculate coverage percentage
        coverage_percent = (masked_pixels / total_pixels) * 100
        
        # Check if coverage is below threshold
        return coverage_percent < min_coverage_percent
    
    @staticmethod
    def merge_masks_with_fallback(sam_mask, fallback_mask, image_size, osm_drivable_mask=None, max_overlap_percent=MAX_OVERLAP_PERCENT):
        """
        Merge SAM-generated masks with fallback OSM non-drivable feature masks.
        Prioritizes SAM masks over building footprints in areas where valid segmentation exists.
        Now evaluates each building polygon individually rather than the whole mask.
        
        Args:
            sam_mask: Binary mask from SAM model
            fallback_mask: Binary mask from OSM non-drivable features (buildings, etc.)
            image_size: Tuple (width, height) of the original image
            osm_drivable_mask: Binary mask of drivable areas from OSM (optional)
            max_overlap_percent: Maximum allowed overlap between a building and ANY SAM mask
                                before the building is excluded (default: 10%)
            
        Returns:
            numpy.ndarray: Combined binary mask
        """
        # Resize masks to match image size
        if sam_mask.shape[::-1] != image_size:
            sam_mask = cv2.resize(sam_mask, image_size, interpolation=cv2.INTER_NEAREST)
        
        if fallback_mask.shape[::-1] != image_size:
            fallback_mask = cv2.resize(fallback_mask, image_size, interpolation=cv2.INTER_NEAREST)
            
        if osm_drivable_mask is not None and osm_drivable_mask.shape[::-1] != image_size:
            osm_drivable_mask = cv2.resize(osm_drivable_mask, image_size, interpolation=cv2.INTER_NEAREST)
        
        # Start with just the SAM mask
        combined_mask = sam_mask.copy()
        
        # Extract individual building contours
        building_contours, _ = cv2.findContours(
            fallback_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process each building individually
        retained_buildings_mask = np.zeros_like(fallback_mask)
        
        # Define the building buffer in meters (5 feet ≈ 1.5 meters)
        building_buffer_meters = 1.5  # 5 feet in meters
        
        # Extract width and height of the image to estimate pixel-to-meter conversion
        width, height = image_size
        
        # Calculate approximate meters per pixel (assuming a typical aerial image)
        # This is a rough approximation using the image diagonal
        diagonal_pixels = np.sqrt(width**2 + height**2)
        diagonal_meters = 50  # Rough estimate of diagonal coverage in meters
        meters_per_pixel = diagonal_meters / diagonal_pixels
        
        # Convert buffer from meters to pixels
        buffer_pixels = int(building_buffer_meters / meters_per_pixel)
        
        # Ensure buffer is at least 1 pixel
        buffer_pixels = max(buffer_pixels, 5)  # Minimum 5 pixels for visibility
        
        print(f"Applying buffer of {buffer_pixels} pixels to accepted building footprints")
        
        for contour in building_contours:
            # Create a mask for this individual building
            building_mask = np.zeros_like(fallback_mask)
            cv2.drawContours(building_mask, [contour], 0, 1, -1)  # Fill the contour
            
            # First check: Remove building if it overlaps with roads
            if osm_drivable_mask is not None:
                road_overlap = cv2.bitwise_and(building_mask, osm_drivable_mask)
                # If the building has ANY significant overlap with roads, skip it
                road_overlap_pixels = np.count_nonzero(road_overlap)
                building_pixels = np.count_nonzero(building_mask)
                
                if building_pixels > 0 and (road_overlap_pixels / building_pixels) * 100 > 5.0:
                    # Skip this building as it has too much road overlap
                    continue
            
            # Second check: Does this building have significant overlap with ANY SAM mask?
            # If so, exclude it completely
            sam_overlap = cv2.bitwise_and(building_mask, sam_mask)
            sam_overlap_pixels = np.count_nonzero(sam_overlap)
            building_pixels = np.count_nonzero(building_mask)
            
            # Calculate overlap percentage 
            overlap_percent = 0
            if building_pixels > 0:
                overlap_percent = (sam_overlap_pixels / building_pixels) * 100
            
            # Only include buildings with little to no SAM mask overlap
            if overlap_percent <= max_overlap_percent:
                # Accept this building and apply buffer
                buffered_building_mask = cv2.dilate(
                    building_mask, 
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_pixels, buffer_pixels))
                )
                # Add the buffered building to our retained buildings mask
                retained_buildings_mask = cv2.bitwise_or(retained_buildings_mask, buffered_building_mask)
        
        # Combine SAM mask with the retained buildings
        combined_mask = cv2.bitwise_or(combined_mask, retained_buildings_mask)
        
        # Apply some post-processing to smooth the combined mask
        combined_mask = cv2.GaussianBlur(combined_mask.astype(np.float32), (3, 3), 0)
        _, combined_mask = cv2.threshold(combined_mask, 0.5, 1, cv2.THRESH_BINARY)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, CV2_STRUCTURING_ELEMENT)
        
        return combined_mask.astype(np.uint8)

    @staticmethod
    def calculate_real_world_area(mask, bbox, image_size):
        """
        Optimized version of calculate_real_world_area with caching of common calculations.
        """
        # Get image dimensions
        height, width = image_size
        
        # Calculate constants for this bbox (cache if calling multiple times with same bbox)
        lon_per_pixel = (bbox[0] - bbox[2]) / width
        lat_per_pixel = (bbox[1] - bbox[3]) / height
        
        # Earth radius in meters
        R = 6378137
        
        # Calculate area per pixel at this latitude (approximation)
        avg_lat = (bbox[1] + bbox[3]) / 2
        cos_lat = np.cos(np.radians(avg_lat))
        pixel_width = 2 * np.pi * R * cos_lat * lon_per_pixel / 360
        pixel_height = 2 * np.pi * R * lat_per_pixel / 360
        area_per_pixel = abs(pixel_width * pixel_height)
        
        # Fast counting of overlap pixels using NumPy sum
        overlap_pixels = np.count_nonzero(mask)
        return overlap_pixels * area_per_pixel

class GeometryProcessor:
    """Class for processing and manipulating geometries"""
    
    @staticmethod
    def process_geometry(contours, original_size, bounding_box, utm_crs):
        """Process contours into valid geographic polygons"""
        original_width, original_height = original_size
        valid_polygons = []
        
        for contour in contours:
            contour = contour.squeeze(axis=1)
            if len(contour) < 3:
                continue

            # 1. Contour Preprocessing with Adaptive Approximation
            contour_perimeter = cv2.arcLength(contour, closed=True)
            epsilon = 0.0007 * contour_perimeter # More conservative approximation
            approx = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze()
            
            if len(approx) < 3 or approx.ndim != 2:
                continue

            # 2. Spline Smoothing with Tension Control
            try:
                x = approx[:, 0].astype(float)
                y = approx[:, 1].astype(float)
                
                # Add periodic boundary conditions for closed shapes
                t = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, 150)  # Increased interpolation points
                
                # Use cubic spline with smoothing
                spl_x = scipy.interpolate.CubicSpline(t, x, bc_type='periodic')
                spl_y = scipy.interpolate.CubicSpline(t, y, bc_type='periodic')
                
                # Generate smoothed points
                new_x = spl_x(t_new)
                new_y = spl_y(t_new)
                points = np.column_stack((new_x, new_y))
            except Exception as e:
                points = approx  # Fallback to approximated points

            # 3. Accurate Coordinate Conversion
            try:
                # Convert to geographic coordinates (fixed latitude calculation)
                lon = (points[:, 0] / original_width) * (bounding_box[0] - bounding_box[2]) + bounding_box[2]
                lat = bounding_box[3] - (points[:, 1] / original_height) * (bounding_box[3] - bounding_box[1])

                lon = np.clip(lon, bounding_box[2], bounding_box[0])
                lat = np.clip(lat, bounding_box[1], bounding_box[3])
            except IndexError:
                continue

            # 4. Geometry Creation with Gentle Simplification
            try:
                polygon = Polygon(np.column_stack((lon, lat)))
                
                # Initial validation and repair
                if not polygon.is_valid:
                    print(f"Invalid polygon detected: {explain_validity(polygon)}")
                    polygon = make_valid(polygon).buffer(0.000005)
                    
                # Simplify in geographic coordinates (0.5m tolerance at equator)
                polygon = polygon.simplify(0.0000045, preserve_topology=True)
                
                if polygon.is_empty:
                    continue

                # 5. UTM-based Refinement
                polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
                polygon_utm = polygon_gdf.to_crs(utm_crs)
                
                buffered_utm = polygon_utm.buffer(0.05).simplify(0.1)
                
                # Convert back to WGS84 with shape preservation
                buffered_wgs84 = buffered_utm.to_crs("EPSG:4326").geometry.iloc[0]
                buffered_wgs84 = buffered_wgs84.simplify(0.000003, preserve_topology=True)  # ~0.3m
                
                if buffered_wgs84.is_valid:
                    valid_polygons.append(buffered_wgs84)
            except Exception as e:
                print(f"Geometry processing error: {e}")

        return valid_polygons

    @staticmethod
    def export_geodata(gdf, output_dir):
        """Export geodata with optimized I/O operations"""
        # Validate and simplify geometries
        def get_utm_area(geom):
            try:
                centroid = geom.centroid
                utm_crs = get_utm_projection(centroid.x, centroid.y)
                return gpd.GeoSeries([geom], crs=4326).to_crs(utm_crs).area[0]
            except Exception:
                return 0
        
        print(f"Processing {len(gdf)} geometries for export...")
        initial_geometry_count = len(gdf)
        print(f"Initial number of geometries for export: {initial_geometry_count}")
        
        # Filter small artifacts more efficiently
        gdf['area_m2'] = gdf.geometry.apply(get_utm_area)
        print(f"Number of geometries after area calculation: {len(gdf)}")
        # Disabled the area filter to retain all geometries during export
        # gdf = gdf[gdf.area_m2 > 0.5]
        # print(f"Number of geometries after area filter (if enabled): {len(gdf)}") # Add this if filter is re-enabled
        
        # Final simplification with tolerance optimized for visual quality
        gdf.geometry = gdf.geometry.simplify(0.000003, preserve_topology=True)
        print(f"Number of geometries after final simplification: {len(gdf)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define export formats
        formats = {"geojson": "GeoJSON", "kml": "KML", "shp": "ESRI Shapefile"}
        
        # Export concurrently
        futures = []

        def export_file(gdf_to_export, final_filepath, driver_name, extension):
            try:
                # Ensure output directory exists (os.makedirs in export_geodata already does this)
                # os.makedirs(os.path.dirname(final_filepath), exist_ok=True)

                if extension == "shp":
                    shp_output_dir = os.path.dirname(final_filepath)
                    final_base_name = os.path.splitext(os.path.basename(final_filepath))[0]
                    
                    known_shp_extensions = [
                        '.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.xml',
                        '.idx', '.fbn', '.fbx', '.atx', '.ixs', '.mxs' # Common shapefile extensions
                    ]
                    
                    # Clean up all potential existing components first
                    for shp_ext in known_shp_extensions:
                        file_to_remove = os.path.join(shp_output_dir, f"{final_base_name}{shp_ext}")
                        if os.path.exists(file_to_remove):
                            try:
                                os.remove(file_to_remove)
                            except OSError: # Ignore minor issues
                                pass 
                    
                    gdf_to_export.to_file(final_filepath, driver=driver_name)
                    return f"Successfully exported {final_filepath} (and its components)"
                
                else: # For GeoJSON, KML, etc. (single file formats)
                    if os.path.exists(final_filepath):
                        try:
                            os.remove(final_filepath)
                        except OSError as e_remove:
                            print(f"Warning: Could not remove existing file {final_filepath} before overwrite: {e_remove}")

                    gdf_to_export.to_file(final_filepath, driver=driver_name)
                    return f"Successfully exported {final_filepath}"

            except Exception as e:
                error_type_info = "(shapefile and its components)" if extension == "shp" else ""
                if extension == "shp":
                    shp_output_dir = os.path.dirname(final_filepath)
                    final_base_name = os.path.splitext(os.path.basename(final_filepath))[0]
                    known_shp_extensions = [
                        '.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.xml',
                        '.idx', '.fbn', '.fbx', '.atx', '.ixs', '.mxs'
                    ]
                    for shp_ext in known_shp_extensions:
                        file_to_clean = os.path.join(shp_output_dir, f"{final_base_name}{shp_ext}")
                        if os.path.exists(file_to_clean):
                            try: 
                                os.remove(file_to_clean)
                            except OSError: 
                                pass 
                
                return f"Error exporting {final_filepath} {error_type_info}: {e}. Artifacts may exist if export was partial."

        for ext, driver in formats.items():
            final_path = os.path.join(output_dir, f"cross_walks.{ext}")
            # temp_path is no longer used or needed here
            print(f"Exporting to {final_path}...")
            
            # Pass final_path as the target path to the modified export function
            futures.append(io_executor.submit(export_file, gdf.copy(), final_path, driver, ext))
        
        # Wait for all exports to complete
        for future in as_completed(futures):
            print(future.result())

# Initialize processors
# ===================================================================================================
data_loader = DataLoader()
osm_processor = OSMProcessor()
mask_processor = MaskProcessor()
geometry_processor = GeometryProcessor()

@app.function(
    volumes={
        "/weights": weights_volume, 
        "/inputs": inputs_volume,
        "/outputs": outputs_volume,
        "/scratch": scratch_volume
    }, 
    image=infer_image, 
    gpu="L4",
    secrets=[modal.Secret.from_name("wandb-secret", environment_name="sam_test")],
    timeout=43200
)
def run_inference(model_path: str, mode: str, city_code: str, rebuild_cache: bool = False):
    """
    Main inference function running on Modal.
    
    Args:
        model_path: Path to the SAM2 model weights
        mode: Running mode ('test' or 'production')
        city_code: City code to process
        rebuild_cache: Whether to force rebuild of the fuzzy search cache
    """
    
    # Set environment variables for optimization
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Print CUDA information for debugging
    print("=== Detailed CUDA Info ===")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (from torch):", torch.version.cuda)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    print("=== End of CUDA Info ===")

    # Clear any stale GPU memory before starting
    torch.cuda.empty_cache()

    # Model initialization with performance optimizations
    with torch.device('cuda'):
        # Use automatic mixed precision for faster inference
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
        autocast_context.__enter__()
        
        # Enable TF32 on Ampere and newer GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch._C._jit_set_texpr_fuser_enabled(False)

        torch.cuda.set_per_process_memory_fraction(0.95)  # Use most of GPU memory
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        
        # Build model with full CUDA optimization
        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            f"../../weights/{model_path}/checkpoints/checkpoint.pt",
            device="cuda"
        )
        
        # Initialize mask generator with configuration
        mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_CONFIG)
        
        # Precompile some operations by running on a small dummy input
        try:
            dummy_input = torch.zeros((1, 3, 64, 64), device='cuda')
            sam2.image_encoder(dummy_input)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Precompilation failed (non-critical): {e}")

    # Setup data directories
    input_set = os.listdir("/inputs")

    # Determine base output path based on mode
    base_output_path_suffix = "test" if mode == "test" else ""
    base_output_dir = os.path.join(f"/outputs/{model_path}", base_output_path_suffix)

    output_dir = os.path.join(base_output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load city code mapping and OSM data
    CITY_CODE_MAPPING_CSV_PATH = "/data/city_code_mapping.csv"
    city_code_to_place_mapping = DataLoader.load_city_code_mapping_from_csv(CITY_CODE_MAPPING_CSV_PATH)
    if not city_code_to_place_mapping:
        print("City code mapping loading failed. Using fallback approach.")

    city_edges_gdf, city_edges_sindex, non_drivable_gdf, non_drivable_sindex = DataLoader.load_city_osm_data(city_code, city_code_to_place_mapping)
    if city_edges_gdf.empty or city_edges_sindex is None:
        print("City OSM data loading failed. Exiting.")
        return
    
    if non_drivable_gdf.empty or non_drivable_sindex is None:
        print("Warning: Non-drivable OSM features not loaded. Building fallback will not be available.")
        # Create empty DataFrames as placeholders
        non_drivable_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        non_drivable_sindex = None

    # Load intersection coordinates if available
    csv_path = "/scratch/intersection_coordinates_v2_0.csv"
    if os.path.exists(csv_path):
        print(f"Loading allowed crosswalk IDs from {csv_path}")
        
        # Clear cache if rebuild_cache is True
        if rebuild_cache:
            print("Force rebuilding fuzzy search cache")
            DataLoader.clear_fuzzy_search_cache()
            
        valid_images = DataLoader.load_and_filter_images(csv_path, "/inputs")
        print(f"Found {len(valid_images)} matching images")
        
        # Replace the original input_set with the filtered one
        input_set = valid_images
        
        # If in test mode, sample from the filtered set
        if mode == "test":
            random.seed(sum(ord(c) for c in city_code))
            input_set = random.sample(input_set, min(100, len(input_set)))
    else:
        print(f"Warning: Could not find CSV file at {csv_path}. Processing all images.")
        # If in test mode, handle sampling as before
        if mode == "test":
            random.seed(sum(ord(c) for c in city_code))
            input_set = random.sample(input_set, min(100, len(input_set)))

    # Initialize wandb for tracking
    wandb.init(
        project="pedestrian-crossing-distance",
        name=f"{city_code}-{model_path}-L4-optimized",
        tags=[city_code, model_path, "drivable_filter", "spatial_index", "L40S-optimized", "building-fallback", "smart-combine"],
        config={
            **MASK_GENERATOR_CONFIG,
            # SAM Parameters
            "sam_points_per_side": MASK_GENERATOR_CONFIG["points_per_side"],
            "sam_pred_iou_thresh": MASK_GENERATOR_CONFIG["pred_iou_thresh"],
            "sam_stability_score_thresh": MASK_GENERATOR_CONFIG["stability_score_thresh"],
            "sam_min_mask_region_area": MASK_GENERATOR_CONFIG["min_mask_region_area"],
            
            # OSM Filtering Parameters (using constants for MAX_RELATIVE_OVERLAP etc.)
            "max_relative_overlap": MAX_RELATIVE_OVERLAP,
            "max_absolute_overlap": MAX_ABSOLUTE_OVERLAP,
            "min_crosswalk_area": MIN_CROSSWALK_AREA,
            "query_expansion_meters": OSMProcessor.QUERY_EXPANSION_METERS, # Log the query expansion
            
            # Fallback Parameters
            "use_building_fallback": True,
            "always_apply_fallback": True,
            "smart_building_combination": True,
            "prioritize_sam_over_buildings": True,
            "remove_building_road_overlaps": True,
            "fallback_coverage_threshold": 1.0,  # Kept for reference but no longer used
            
            # Performance Optimization
            "fuzzy_search_cache_enabled": True,
            "fuzzy_search_cache_forced_rebuild": rebuild_cache,
            
            # Image Processing
            "image_size": IMAGE_SIZE,
            "output_resolution": "original",
            
            # Environment
            "cuda_version": "12.1",
            "torch_version": torch.__version__,
            "model_path": model_path,
            "mode": mode,
            "city_code": city_code,
            "osm_method": "spatial_index",
            "city_mapping_source": "city_code_mapping.csv",
            "platform": "modal-L4",
            "max_overlap_percent": MAX_OVERLAP_PERCENT
        }
    )

    print(f"Total images to check: {len(input_set)}")
    
    # --- Checkpointing Setup ---
    CHECKPOINT_INTERVAL = 100 # Save every 100 images
    checkpoint_dir = os.path.join(base_output_dir, "geofile_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    shapefile_data = []
    processed_image_names = set() # To avoid duplicates when loading checkpoints

    # Load existing checkpoints
    initial_checkpoint_files_for_counter = [] # Used to set the batch counter correctly

    # Always try to load checkpoints
    checkpoint_files_to_load = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_batch_*.pkl")))
    initial_checkpoint_files_for_counter = checkpoint_files_to_load # Store for counter init
    if checkpoint_files_to_load:
        print(f"Found {len(checkpoint_files_to_load)} existing checkpoint files. Loading data...")
        for cp_file in checkpoint_files_to_load:
            try:
                with open(cp_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    for item in batch_data:
                        if item['image_name'] not in processed_image_names:
                            shapefile_data.append(item)
                            processed_image_names.add(item['image_name'])
                print(f"Loaded {len(batch_data)} items from {cp_file}")
            except Exception as e:
                print(f"Error loading checkpoint file {cp_file}: {e}")
        print(f"Loaded a total of {len(shapefile_data)} items from checkpoints.")
    else:
        print("No existing checkpoint files found to load. Starting fresh run for this model/mode/city combination.")
    # --- End Checkpointing Setup ---

    # Set up counters for tracking
    processed_count = 0
    skipped_count = 0
    fallback_count = 0
    
    # Initialize collection for test mode visualization
    collected_images = [] if mode == "test" else None
    
    # Main processing loop
    current_batch_data = [] # For periodic checkpointing
    checkpoint_batch_counter = len(initial_checkpoint_files_for_counter) # Start batch counter

    for image_file in input_set:
        print(f"Processing: {image_file}")
        image_name = os.path.splitext(image_file)[0].split("/")[-1]
        image_path = os.path.join("/inputs", image_file)
        
        # Check if image was already processed and its geometry loaded from checkpoint
        output_path = os.path.join(output_dir, f"{image_name}_masked.jpg") # Still need output_path for saving later
        if image_name in processed_image_names:
            print(f"Image {image_name} has its geometry loaded from checkpoint, skipping full processing.")
            skipped_count += 1
            continue # Skip to next image

        # Get bounding box from filename
        bbox = bounding_box_from_filename(image_name)
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping")
            continue
            
        original_size = (img.shape[1], img.shape[0])  # width, height
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Clear CUDA cache before processing
        torch.cuda.empty_cache()
        
        # Run model inference
        with torch.no_grad():
            try:
                masks = mask_generator.generate(resized_img)
            except IndexError as e:
                if "too many indices for tensor of dimension 1" in str(e):
                    masks = []
                    print(f"No masks detected for image {image_path}")
                else:
                    raise
        
        # Process OSM data for this image
        bbox_poly = gpd.GeoSeries([box(bbox[2], bbox[1], bbox[0], bbox[3])], crs="EPSG:4326").iloc[0]
        original_osm_edges, buffered_osm_edges = OSMProcessor.get_osm_drivable_edges(
            bbox_poly, city_edges_gdf, city_edges_sindex)
        osm_mask = OSMProcessor.create_osm_mask(buffered_osm_edges, (IMAGE_SIZE, IMAGE_SIZE), bbox)
        
        # Get non-drivable features (buildings, parks, etc.) as fallback
        non_drivable_features = OSMProcessor.get_osm_non_drivable_features(
            bbox_poly, non_drivable_gdf, non_drivable_sindex)
        non_drivable_mask = OSMProcessor.create_osm_mask(non_drivable_features, (IMAGE_SIZE, IMAGE_SIZE), bbox)

        # Filter masks based on OSM overlap
        valid_masks = []
        excluded_masks = []

        for mask_data in masks:
            mask = mask_data["segmentation"]
            
            # Calculate real-world areas
            overlap_area = MaskProcessor.calculate_real_world_area(
                np.logical_and(mask, osm_mask), bbox, (IMAGE_SIZE, IMAGE_SIZE))
            mask_area = MaskProcessor.calculate_real_world_area(mask, bbox, (IMAGE_SIZE, IMAGE_SIZE))
            
            if mask_area < MIN_CROSSWALK_AREA:  # Preserve small valid crosswalks
                valid_masks.append(mask_data)
                continue
            
            relative = overlap_area / mask_area if mask_area > 0 else 0
            if (overlap_area > MAX_ABSOLUTE_OVERLAP) or (relative > MAX_RELATIVE_OVERLAP):
                excluded_masks.append(mask_data)
            else:
                valid_masks.append(mask_data)
        
        # Create visualization
        valid_detections = sv.Detections.from_sam(valid_masks)
        excluded_detections = sv.Detections.from_sam(excluded_masks)
        excluded_detections.class_id = np.array([-1] * len(excluded_detections))

        # Annotate valid masks with index colors
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated = mask_annotator.annotate(resized_img, valid_detections)

        # Add excluded masks in blue outline
        excluded_annotator = sv.PolygonAnnotator(color=sv.Color(r=0, g=0, b=255), thickness=2)
        annotated = excluded_annotator.annotate(annotated, excluded_detections)
        
        # Free memory
        torch.cuda.empty_cache()
        print(f"Valid Masks: {len(valid_masks)}, Excluded Masks: {len(excluded_masks)}")
        
        # Create OSM masks
        original_osm_mask = OSMProcessor.create_osm_mask(
            original_osm_edges, (IMAGE_SIZE, IMAGE_SIZE), bbox, is_lines=True)
        buffered_osm_mask = osm_mask

        # Draw original OSM edges (blue)
        osm_lines_contours, _ = cv2.findContours(
            original_osm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, osm_lines_contours, -1, (0, 0, 255), 2)

        # Draw buffered OSM areas (green)
        osm_buffer_contours, _ = cv2.findContours(
            buffered_osm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, osm_buffer_contours, -1, (0, 255, 0), 2)
        
        # Process final mask
        sam_mask = MaskProcessor.process_masks_parallel(valid_masks, original_size)
        
        # Always use building footprints as fallback regardless of SAM mask coverage
        print(f"Applying OSM non-drivable features for {image_name}.")
        # Resize non-drivable mask to match original image size
        resized_non_drivable_mask = cv2.resize(non_drivable_mask, original_size, interpolation=cv2.INTER_NEAREST)
        # Resize OSM drivable area mask to match original image size
        resized_osm_drivable_mask = cv2.resize(osm_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Create a downscaled version of the sam_mask for visualization
        downscaled_sam_mask = cv2.resize(sam_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Extract individual building contours for visualization
        building_contours, _ = cv2.findContours(
            non_drivable_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate buffer size for visualization (same as in merge_masks_with_fallback)
        building_buffer_meters = 1.5  # 5 feet in meters
        width, height = IMAGE_SIZE, IMAGE_SIZE
        diagonal_pixels = np.sqrt(width**2 + height**2)
        diagonal_meters = 50  # Rough estimate of diagonal coverage in meters
        meters_per_pixel = diagonal_meters / diagonal_pixels
        buffer_pixels = int(building_buffer_meters / meters_per_pixel)
        buffer_pixels = max(buffer_pixels, 5)  # Minimum 5 pixels for visibility
        
        # Process each building polygon individually for visualization
        standalone_buildings_mask = np.zeros_like(non_drivable_mask)
        excluded_buildings_mask = np.zeros_like(non_drivable_mask)
        buffered_buildings_mask = np.zeros_like(non_drivable_mask)  # For visualization with buffer
        
        for contour in building_contours:
            # Create a mask for this individual building
            building_mask = np.zeros_like(non_drivable_mask)
            cv2.drawContours(building_mask, [contour], 0, 1, -1)  # Fill the contour
            
            # Check road overlap
            road_overlap = cv2.bitwise_and(building_mask, osm_mask)
            road_overlap_pixels = np.count_nonzero(road_overlap)
            building_pixels = np.count_nonzero(building_mask)
            road_overlap_percent = 0
            if building_pixels > 0:
                road_overlap_percent = (road_overlap_pixels / building_pixels) * 100
            
            # Check SAM mask overlap
            sam_overlap = cv2.bitwise_and(building_mask, downscaled_sam_mask)
            sam_overlap_pixels = np.count_nonzero(sam_overlap)
            sam_overlap_percent = 0
            if building_pixels > 0:
                sam_overlap_percent = (sam_overlap_pixels / building_pixels) * 100
            
            if road_overlap_percent > 5.0 or sam_overlap_percent > MAX_OVERLAP_PERCENT:
                # Building is excluded either due to road overlap or SAM overlap
                excluded_buildings_mask = cv2.bitwise_or(excluded_buildings_mask, building_mask)
            else:
                # Standalone building with little to no overlap - will be used
                standalone_buildings_mask = cv2.bitwise_or(standalone_buildings_mask, building_mask)
                
                # Apply the same buffer used in merge_masks_with_fallback for visualization
                buffered_building_mask = cv2.dilate(
                    building_mask, 
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_pixels, buffer_pixels))
                )
                buffered_buildings_mask = cv2.bitwise_or(buffered_buildings_mask, buffered_building_mask)
        
        # 1. Excluded buildings that won't be used (orange) - includes both road overlaps and SAM overlaps
        if np.any(excluded_buildings_mask):
            excluded_contours, _ = cv2.findContours(
                excluded_buildings_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, excluded_contours, -1, (255, 165, 0), 2)  # Orange
            
        # 2. Standalone buildings original footprints (thin magenta)
        if np.any(standalone_buildings_mask):
            standalone_contours, _ = cv2.findContours(
                standalone_buildings_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, standalone_contours, -1, (180, 100, 180), 1)  # Thin magenta
            
        # 3. Buffered buildings that will be used (bright cyan) - shows the actual buffer applied
        if np.any(buffered_buildings_mask):
            buffered_contours, _ = cv2.findContours(
                buffered_buildings_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, buffered_contours, -1, (0, 255, 255), 2)  # Cyan, thicker
        
        # Save the annotated image
        async_write(output_path, rgb_to_bgr(annotated))
        print(f"Saved output to: {output_path}")
        processed_count += 1

        # Store image for wandb visualization in test mode
        if mode == "test":
            collected_images.append(
                wandb.Image(
                    annotated,
                    caption=f"Processed image: {image_name}"
                )
            )

        # Merge SAM masks with non-drivable features while avoiding road overlaps
        final_mask = MaskProcessor.merge_masks_with_fallback(
            sam_mask, 
            resized_non_drivable_mask, 
            original_size,
            osm_drivable_mask=resized_osm_drivable_mask,
            max_overlap_percent=MAX_OVERLAP_PERCENT
        )
        
        # Log metrics
        sam_coverage = (np.count_nonzero(sam_mask) / (original_size[0] * original_size[1])) * 100
        building_coverage = (np.count_nonzero(resized_non_drivable_mask) / (original_size[0] * original_size[1])) * 100
        combined_coverage = (np.count_nonzero(final_mask) / (original_size[0] * original_size[1])) * 100
        
        # Calculate how many building pixels were removed due to road overlap
        naively_combined = cv2.bitwise_or(sam_mask, resized_non_drivable_mask)
        naive_coverage = (np.count_nonzero(naively_combined) / (original_size[0] * original_size[1])) * 100
        road_overlap_pct = naive_coverage - combined_coverage
        
        print(f"SAM mask coverage: {sam_coverage:.2f}%")
        print(f"Building footprint coverage: {building_coverage:.2f}%")
        print(f"Combined mask coverage: {combined_coverage:.2f}%")
        print(f"Road overlap correction: {road_overlap_pct:.2f}%")
        
        wandb.log({
            "fallback_used": True,
            "image_name": image_name,
            "sam_mask_coverage_percent": sam_coverage,
            "fallback_mask_coverage_percent": building_coverage,
            "combined_mask_coverage_percent": combined_coverage,
            "road_overlap_correction_percent": road_overlap_pct
        })
        fallback_count += 1
        
        # Find contours from mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out tiny contours early
        MIN_CONTOUR_AREA = 10  # in pixels
        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        
        # Get coordinate system for geometry processing
        centroid_lon = (bbox[0] + bbox[2])/2
        centroid_lat = (bbox[1] + bbox[3])/2
        utm_crs = get_utm_projection(centroid_lon, centroid_lat)
        
        # Process geometry from contours
        if contours:
            valid_polys = GeometryProcessor.process_geometry(contours, original_size, bbox, utm_crs)
            
            if valid_polys:
                try:
                    # Use unary_union with buffer=0 for faster processing
                    union = unary_union(valid_polys).buffer(0)
                    # Clip union to image bounds
                    union = union.intersection(bbox_poly)
                    
                    if union.is_empty:
                        inverse_geometry = bbox_poly
                    else:
                        # More efficient difference operation
                        inverse_geometry = bbox_poly.difference(union)
                        # Use relative area threshold
                        if inverse_geometry.area < bbox_poly.area * 0.01:
                            inverse_geometry = bbox_poly
                except Exception as e:
                    print(f"Geometry processing error: {e}")
                    inverse_geometry = bbox_poly
            else:
                inverse_geometry = bbox_poly
        else:
            inverse_geometry = bbox_poly
            
        # Final geometry cleanup
        inverse_geometry = make_valid(inverse_geometry) if not inverse_geometry.is_valid else inverse_geometry
        inverse_geometry = inverse_geometry.simplify(0.000003, preserve_topology=True)
        
        # Store geometry for export
        shapefile_data.append({
            "image_name": image_name,
            "geometry": inverse_geometry
        })
        current_batch_data.append({
            "image_name": image_name,
            "geometry": inverse_geometry
        })
        processed_image_names.add(image_name) # Add to set of processed names
        
        print(f"Image: {image_name}")
        print(f"Contours found: {len(contours)}")
        print(f"Valid polygons: {len(valid_polys) if 'valid_polys' in locals() else 0}")

        # --- Periodic Checkpointing --- 
        if len(current_batch_data) >= CHECKPOINT_INTERVAL:
            checkpoint_file_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{checkpoint_batch_counter}.pkl")
            try:
                with open(checkpoint_file_path, 'wb') as f:
                    pickle.dump(current_batch_data, f)
                print(f"Saved checkpoint with {len(current_batch_data)} items to {checkpoint_file_path}")
                current_batch_data = [] # Reset batch
                checkpoint_batch_counter += 1
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        # --- End Periodic Checkpointing ---

    # --- Final Checkpoint for remaining data ---
    if current_batch_data: # Save any remaining data
        checkpoint_file_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{checkpoint_batch_counter}.pkl")
        try:
            with open(checkpoint_file_path, 'wb') as f:
                pickle.dump(current_batch_data, f)
            print(f"Saved final checkpoint with {len(current_batch_data)} items to {checkpoint_file_path}")
            checkpoint_batch_counter += 1
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")
    # --- End Final Checkpoint ---

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images checked: {len(input_set)}")
    print(f"Images processed: {processed_count}")
    print(f"Images skipped (already processed): {skipped_count}")
    print(f"Images with building footprints applied: {fallback_count} (100.0%)")
    
    # Export results to GeoDataFrame
    if shapefile_data:
        gdf = gpd.GeoDataFrame(shapefile_data, geometry="geometry", crs="EPSG:4326")
        
        # Clean up before GeoDataFrame operations (which can be memory intensive)
        gc_collect()
        
        # Export data to files
        geofiles_output_dir = os.path.join(base_output_dir, "geofiles")
        GeometryProcessor.export_geodata(gdf, geofiles_output_dir)
    
    # Cleanup and shutdown executors
    io_executor.shutdown(wait=True)
    
    # Final cleanup of any temporary files or memory
    gc_collect()
    
    # Log results to wandb in test mode
    if mode == "test":
        wandb.log({
            "processed_images": collected_images,
            "images_processed": processed_count,
            "images_skipped": skipped_count,
            "images_with_building_footprints": fallback_count
        })
        wandb.finish()

# Entry point for local execution
# ===================================================================================================
@app.local_entrypoint()
def main(model: str, mode: str, rebuild_cache: bool = False):
    """Main entry point for local execution"""
    run_inference.remote(
        model_path=model,
        mode=mode,
        city_code=ENV_NAME,  # Use the same ENV_NAME from the top of the file
        rebuild_cache=rebuild_cache
    )