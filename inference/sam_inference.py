import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import cv2 
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Polygon, box, MultiPolygon
from shapely.ops import unary_union
import torch
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from threading import Thread
import pickle
import csv
import json
from shapely.validation import make_valid, explain_validity
import scipy.interpolate
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
import math

import modal

load_dotenv()

from utils import PRECISION, get_crosswalk_id, decode_crosswalk_id, coords_from_distance, bounding_box_from_filename

IMAGE_SIZE = 1024
CITY_CODE = os.getenv("CITY_CODE")
INPUTS_VOLUME_NAME = f"crosswalk-data-{CITY_CODE}"
OUTPUTS_VOLUME_NAME = f"{INPUTS_VOLUME_NAME}-results"
CITY_CODE_MAPPING_CSV_PATH = "data/city_code_mapping.csv"

MASK_GENERATOR_CONFIG = {
    "points_per_side": 16,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.91,
    "crop_n_layers": 1,
    "min_mask_region_area": 500,
    "points_per_batch": 512,
    "box_nms_thresh": 0.78,
    "crop_overlap_ratio": 0.66,
    "crop_n_points_downscale_factor": 2, 
    "output_mode": "binary_mask"
}

MAX_RELATIVE_OVERLAP = 0.05
MAX_ABSOLUTE_OVERLAP = 10.0  # meters²
MIN_CROSSWALK_AREA = 5.0     # meters²
OSM_BUFFER_METERS = 1.5

app = modal.App(name="crossing-distance-sam2-inference")

infer_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "CXX": "g++",
        "CC": "gcc",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0", 
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda"
    })
    .apt_install("git", "wget", "python3-opencv", "ffmpeg", "build-essential", "ninja-build",  "g++", "libgl1-mesa-glx")
    .pip_install("numpy")
    .run_commands("pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
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
        "nvcc --version",
        "echo $CUDA_HOME",
        "echo $PATH",
        "echo $LD_LIBRARY_PATH",
        "ldd --version"
    ))

weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")
inputs_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, environment_name=CITY_CODE)
scratch_volume = modal.Volume.from_name("scratch", environment_name=CITY_CODE)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True, environment_name=CITY_CODE)

io_executor = ThreadPoolExecutor(max_workers=16)

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

def get_utm_zone(longitude):
    return int((longitude + 180) // 6) + 1

def get_utm_projection(lon, lat):
    utm_zone = get_utm_zone(lon)
    northern = lat >= 0
    epsg_code = 32600 + utm_zone if northern else 32700 + utm_zone
    return f"EPSG:{epsg_code}"

def load_and_filter_images(csv_path: str, image_dir: str) -> list:
    """
    Combined function that loads coordinates from CSV and finds matching images.
    Returns a list of valid image paths and tracks original coordinates for missing images.
    """
    valid_images = []
    coordinate_to_id = {}
    missing_coordinates = []
    
    # First, find and load the decoder JSON file
    decoder_file = None
    for file in os.listdir(image_dir):
        if file.endswith('.json'):
            decoder_file = os.path.join(image_dir, file)
            print(f"Found decoder file: {decoder_file}")
            break
    
    # Load the decoder JSON if found
    decoder = {}
    if decoder_file:
        try:
            with open(decoder_file, 'r') as f:
                decoder = json.load(f)
            print(f"Loaded decoder with {len(decoder)} entries")
        except Exception as e:
            print(f"Error loading decoder file: {e}")
    
    # Index all images by ID for faster lookup
    image_id_map = {}
    for filename in os.listdir(image_dir):
        if filename.endswith('.json'):
            continue
        
        filename_without_ext = os.path.splitext(filename)[0]
        parts = filename_without_ext.split('_', 1)
        
        if len(parts) == 2 and parts[0] == "crosswalk":
            crosswalk_id = parts[1]
            image_id_map[crosswalk_id] = os.path.join(image_dir, filename)
    
    # Pre-process decoder keys for proximity matching
    if decoder:
        decoder_coords = []
        for coord_key in decoder.keys():
            try:
                lat, lon = map(float, coord_key.split(','))
                decoder_coords.append((lat, lon, coord_key))
            except Exception:
                continue
    
    # Process coordinates from CSV
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            longitude = float(row['x'])
            latitude = float(row['y'])
            orig_coords = (latitude, longitude)
            
            # First try direct lookup in decoder using original coordinates
            coord_key = f"{latitude},{longitude}"
            found_via_decoder = False
            
            if decoder:
                # Exact match check
                if coord_key in decoder:
                    # Found direct match in decoder
                    img_path = decoder[coord_key].replace("/data/", "")
                    full_path = os.path.join(image_dir, img_path)
                    if os.path.exists(full_path):
                        valid_images.append(full_path)
                        found_via_decoder = True
                
                # If not found with exact match, try proximity match (within ~1 meter)
                if not found_via_decoder:
                    # Convert ~1 meter to decimal degrees (approximate, varies by latitude)
                    # At equator, 1 meter ≈ 0.000009 degrees
                    tolerance = 0.000009  # ~1 meter at equator
                    
                    # Adjust tolerance for latitude (1 meter gets smaller in longitude as we move toward poles)
                    # cos(latitude) adjustment accounts for longitude compression at higher latitudes
                    lon_tolerance = tolerance / math.cos(math.radians(abs(latitude)))
                    
                    closest_match = None
                    min_distance = float('inf')
                    
                    for dc_lat, dc_lon, dc_key in decoder_coords:
                        # Calculate approximate distance in degrees
                        lat_diff = abs(latitude - dc_lat)
                        lon_diff = abs(longitude - dc_lon)
                        
                        # Quick filter to avoid expensive calculations
                        if lat_diff > tolerance or lon_diff > lon_tolerance:
                            continue
                            
                        # Haversine for more precise distance on curved earth surface
                        # Convert lat/lon from degrees to radians
                        lat1, lon1 = math.radians(latitude), math.radians(longitude)
                        lat2, lon2 = math.radians(dc_lat), math.radians(dc_lon)
                        
                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        
                        # Earth radius in meters
                        r = 6371000
                        distance = c * r  # Distance in meters
                        
                        if distance < min_distance and distance <= 1.0:  # 1.0 meter threshold
                            min_distance = distance
                            closest_match = dc_key
                    
                    if closest_match is not None:
                        img_path = decoder[closest_match].replace("/data/", "")
                        full_path = os.path.join(image_dir, img_path)
                        if os.path.exists(full_path):
                            valid_images.append(full_path)
                            found_via_decoder = True
                            print(f"Found proximity match ({min_distance:.2f}m) for {latitude},{longitude} -> {closest_match}")
            
            # If not found via decoder, try rounded coordinates and ID matching
            if not found_via_decoder:
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
                    else:
                        # Try alternative formats
                        expected_filename = f"crosswalk_{crosswalk_id}"
                        found = False
                        
                        for filename in os.listdir(image_dir):
                            if filename.endswith(('.jpeg', '.jpg', '.png')) and expected_filename in filename:
                                valid_images.append(os.path.join(image_dir, filename))
                                found = True
                                break
                        
                        if not found:
                            missing_coordinates.append((orig_coords, crosswalk_id))
                except AssertionError as e:
                    print(f"Skipping invalid coordinate ({rounded_lat}, {rounded_lon}): {e}")
                    missing_coordinates.append((orig_coords, None))
    
    # Print detailed statistics
    print(f"Processed {len(coordinate_to_id)} valid coordinates from CSV")
    print(f"Found {len(valid_images)} matching images")
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
            f.write("Latitude,Longitude,CrosswalkID,ExpectedFilename\n")
            for coords, crosswalk_id in missing_coordinates:
                lat, lon = coords
                if crosswalk_id:
                    f.write(f"{lat},{lon},{crosswalk_id},crosswalk_{crosswalk_id}.jpg\n")
                else:
                    f.write(f"{lat},{lon},N/A,N/A\n")
        print(f"\nSaved all {len(missing_coordinates)} missing coordinates to {missing_file}")
    
    return valid_images

def load_city_code_mapping_from_csv(csv_filepath):
    """Loads city code to OSM place mapping from a CSV file."""
    city_code_to_place_mapping = {}
    try:
        with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile: # Use utf-8-sig to handle BOM if present
            reader = csv.DictReader(csvfile)
            for row in reader:
                city_code = row['Code'].lower()
                city_name = row['City']
                state_name = row['State'] if row['State'] else None # Handle empty state
                country_name = row['Country']

                place_dict = {'city': city_name, 'country': country_name} # Base dict
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
                # Verify CRS is still correct
                if cached_data['city_edges_gdf'].crs == "EPSG:4326":
                    print(f"OSM data loaded from cache for {city_code}")
                    return cached_data['city_edges_gdf'], cached_data['city_edges_sindex']
                else:
                    print(f"Cached OSM data had invalid CRS, rebuilding...")
        
        print(f"Querying OSM for drivable edges in {city_code}...")
        place = city_code_to_place_mapping.get(city_code) # Use the mapping
        if place is None: # Handle unknown city code
            raise ValueError(f"City code '{city_code}' not found in city code mapping CSV.")

        tags = {
            "highway": True,
            "boundary": True,
            "landuse": True,
            "leisure": True
        }

        city_edges_gdf = ox.features_from_place(place, tags=tags)

        # Verification and enforcement of CRS
        if city_edges_gdf.crs is None:
            print("Warning: city_edges_gdf CRS was None, setting to EPSG:4326")
            city_edges_gdf.crs = "EPSG:4326"
        elif city_edges_gdf.crs != "EPSG:4326":
            print(f"Warning: city_edges_gdf CRS was not EPSG:4326, reprojecting from {city_edges_gdf.crs} to EPSG:4326")
            city_edges_gdf = city_edges_gdf.to_crs("EPSG:4326")

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

        # Create spatial index once and cache it
        city_edges_sindex = city_edges_gdf.sindex
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'city_edges_gdf': city_edges_gdf,
                'city_edges_sindex': city_edges_sindex
            }, f)
        
        print(f"OSM data queried, filtered, and cached for {city_code}.")
        return city_edges_gdf, city_edges_sindex

    except Exception as e:
        print(f"Error querying OSM data: {e}")
        return gpd.GeoDataFrame(), None

def get_osm_drivable_edges(bbox_polygon, city_edges_gdf, city_edges_sindex):
    buffered_distance = OSM_BUFFER_METERS
    
    # Get UTM CRS for this bbox
    centroid = bbox_polygon.centroid
    utm_crs = get_utm_projection(centroid.x, centroid.y)
    
    # Convert and buffer in UTM
    bbox_utm = gpd.GeoSeries([bbox_polygon], crs="EPSG:4326").to_crs(utm_crs)
    buffered_bbox_utm = bbox_utm.buffer(buffered_distance).iloc[0]
    buffered_bbox_polygon = gpd.GeoSeries([buffered_bbox_utm], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

    # Query OSM data
    possible_matches_index = list(city_edges_sindex.intersection(buffered_bbox_polygon.bounds))
    possible_matches = city_edges_gdf.iloc[possible_matches_index]
    
    precise_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
    
    # Buffer OSM edges in UTM
    if not precise_matches.empty:
        precise_matches_utm = precise_matches.to_crs(utm_crs)
        buffered_matches_utm = precise_matches_utm.copy()
        buffered_matches_utm['geometry'] = buffered_matches_utm.geometry.buffer(buffered_distance)
        buffered_matches = buffered_matches_utm.to_crs("EPSG:4326")
        buffered_osm_edges = buffered_matches['geometry'].tolist()
    else:
        buffered_osm_edges = []
    
    return precise_matches['geometry'].tolist(), buffered_osm_edges


def create_osm_mask(geometries, image_size, bbox, is_lines=False):
    """Create binary mask of OSM edges in image coordinates with optimized processing."""
    if not geometries:
        return np.zeros(image_size, dtype=np.uint8)
        
    osm_mask = np.zeros(image_size, dtype=np.uint8)
    width, height = image_size
    
    # Precompute scale factors for coordinate conversion (once, not per-point)
    x_scale = width / (bbox[0] - bbox[2])
    y_scale = height / (bbox[1] - bbox[3])
    x_offset = bbox[2]
    y_offset = bbox[3]
    
    def transform_point(x, y):
        """Fast coordinate transformation"""
        pixel_x = int((x - x_offset) * x_scale)
        pixel_y = int((y - y_offset) * y_scale)
        return pixel_x, pixel_y

    def process_line(line):
        if line.is_empty:
            return
            
        coords = []
        for x, y in line.coords:
            pixel_x, pixel_y = transform_point(x, y)
            coords.append((pixel_x, pixel_y))
            
        if len(coords) >= 2:
            # Draw lines between consecutive points - use NumPy for vectorization
            coords_array = np.array(coords, dtype=np.int32)
            for i in range(len(coords_array) - 1):
                cv2.line(osm_mask, 
                         (coords_array[i][0], coords_array[i][1]),
                         (coords_array[i+1][0], coords_array[i+1][1]), 
                         1, thickness=2)
    
    def process_polygon(polygon):
        """Optimized polygon processing"""
        if polygon.is_empty:
            return
            
        # Skip small polygons
        if polygon.area < 1e-9:
            return
            
        # Convert exterior coordinates efficiently
        coords = []
        for x, y in polygon.exterior.coords:
            pixel_x, pixel_y = transform_point(x, y)
            coords.append((pixel_x, pixel_y))
        
        if len(coords) > 2:
            # Use int32 for better performance with OpenCV
            cv2.fillPoly(osm_mask, [np.array(coords, dtype=np.int32)], 1)
            
        # Also process interior holes for complex polygons
        for interior in polygon.interiors:
            int_coords = []
            for x, y in interior.coords:
                pixel_x, pixel_y = transform_point(x, y)
                int_coords.append((pixel_x, pixel_y))
                
            if len(int_coords) > 2:
                cv2.fillPoly(osm_mask, [np.array(int_coords, dtype=np.int32)], 0)
    
    # Process geometries based on their type
    for geom in geometries:
        if isinstance(geom, LineString):
            process_line(geom)
        elif isinstance(geom, MultiPolygon):
            for polygon in geom.geoms:
                process_polygon(polygon)
        elif isinstance(geom, Polygon):
            process_polygon(geom)

    return osm_mask

# Add these constants at the beginning of the file with other constants
CV2_STRUCTURING_ELEMENT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Helper function for multi-threaded image processing
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

def process_geometry(contours, original_size, bounding_box, utm_crs):
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
    
    # Filter small artifacts more efficiently
    gdf['area_m2'] = gdf.geometry.apply(get_utm_area)
    gdf = gdf[gdf.area_m2 > 0.5]
    
    # Final simplification with tolerance optimized for visual quality
    gdf.geometry = gdf.geometry.simplify(0.000003, preserve_topology=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define export formats
    formats = {"geojson": "GeoJSON", "kml": "KML", "shp": "ESRI Shapefile"}
    
    # Export concurrently
    futures = []
    for ext, driver in formats.items():
        path = os.path.join(output_dir, f"cross_walks.{ext}")
        print(f"Exporting to {path}...")
        
        # Use a function to capture loop variables
        def export_file(gdf, path, driver):
            try:
                gdf.to_file(path, driver=driver)
                return f"Successfully exported {path}"
            except Exception as e:
                return f"Error exporting {path}: {e}"
        
        # Submit to thread pool
        futures.append(io_executor.submit(export_file, gdf, path, driver))
    
    # Wait for all exports to complete
    for future in as_completed(futures):
        print(future.result())

@app.function(volumes={"/weights": weights_volume, 
                       "/inputs": inputs_volume,
                       "/outputs": outputs_volume,
                       "/scratch": scratch_volume
                       }, 
              image=infer_image, gpu="L40S", 
              secrets=[modal.Secret.from_name("wandb-secret", environment_name="sam_test")],
              mounts=[modal.Mount.from_local_dir("../data/mapping", remote_path="/data")], timeout=21600)
def run_inference(model_path: str, mode: str, city_code: str, force: bool = False):
    import os
    import json
    from dotenv import load_dotenv
    import numpy as np
    from PIL import Image
    import cv2
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import LineString, Polygon, box, MultiPolygon
    from shapely.ops import unary_union
    import torch
    import supervision as sv
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import argparse
    import wandb
    import random
    from threading import Thread
    import pickle
    import csv
    from shapely.validation import make_valid, explain_validity
    import scipy.interpolate

    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
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
        
        # Optimize the mask generator parameters
        mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_CONFIG)
        
        # Precompile some operations by running on a small dummy input
        try:
            dummy_input = torch.zeros((1, 3, 64, 64), device='cuda')
            sam2.image_encoder(dummy_input)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Precompilation failed (non-critical): {e}")

    # Data setup
    input_set = os.listdir("/inputs")
    output_dir = f"/outputs/{model_path}/images"
    os.makedirs(output_dir, exist_ok=True)
    
    CITY_CODE_MAPPING_CSV_PATH = "/data/city_code_mapping.csv"
    city_code_to_place_mapping = load_city_code_mapping_from_csv(CITY_CODE_MAPPING_CSV_PATH)
    if not city_code_to_place_mapping:
        print("City code mapping loading failed. Using fallback approach.")

    city_edges_gdf, city_edges_sindex = load_city_osm_data(city_code, city_code_to_place_mapping)
    if city_edges_gdf.empty or city_edges_sindex is None:
        print("City OSM data loading failed. Exiting.")
        return

    # Path to the intersection coordinates CSV file
    csv_path = "/scratch/intersection_coordinates_v2_0.csv"
    if os.path.exists(csv_path):
        print(f"Loading allowed crosswalk IDs from {csv_path}")
        valid_images = load_and_filter_images(csv_path, "/inputs")
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

    # WANDB initialization
    wandb.init(
        project="pedestrian-crossing-distance",
        name=f"{city_code}-{model_path}-L40S-optimized",
        tags=[city_code, model_path, "drivable_filter", "spatial_index", "L40S-optimized"],
        config={
            **MASK_GENERATOR_CONFIG,
            # SAM Parameters
            "sam_points_per_side": MASK_GENERATOR_CONFIG["points_per_side"],
            "sam_pred_iou_thresh": MASK_GENERATOR_CONFIG["pred_iou_thresh"],
            "sam_stability_score_thresh": MASK_GENERATOR_CONFIG["stability_score_thresh"],
            "sam_min_mask_region_area": MASK_GENERATOR_CONFIG["min_mask_region_area"],
            
            # OSM Filtering Parameters
            "osm_buffer_meters": OSM_BUFFER_METERS,
            "max_relative_overlap": MAX_RELATIVE_OVERLAP,
            "max_absolute_overlap": MAX_ABSOLUTE_OVERLAP,
            "min_crosswalk_area": MIN_CROSSWALK_AREA,
            
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
            "platform": "modal-L40S",
            "force_processing": force
        }
    )

    print(f"Total images to check: {len(input_set)}")
    shapefile_data = []
    
    # Set up counters for tracking
    processed_count = 0
    skipped_count = 0
    
    # If in test mode, initialize collection for images
    if mode == "test":
        collected_images = []
    
    for image_file in input_set:
        print(f"Processing: {image_file}")
        image_name = os.path.splitext(image_file)[0].split("/")[-1]
        image_path = os.path.join("/inputs", image_file)
        
        # Check if output already exists - skip this check in test mode for consistency
        output_path = os.path.join(output_dir, f"{image_name}_masked.jpg")
        if mode != "test" and not force and os.path.exists(output_path):
            print(f"Output already exists for {image_name}, skipping (use --force to override)")
            skipped_count += 1
            continue
            
        bbox = bounding_box_from_filename(image_name)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping")
            continue
            
        original_size = (img.shape[1], img.shape[0])  # width, height
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Clear CUDA cache before processing
        torch.cuda.empty_cache()
        
        # Run model on single image
        with torch.no_grad():
            try:
                masks = mask_generator.generate(resized_img)
            except IndexError as e:
                if "too many indices for tensor of dimension 1" in str(e):
                    masks = []
                    print(f"No masks detected for image {image_path}")
                else:
                    raise
        
        bbox_poly = gpd.GeoSeries([box(bbox[2], bbox[1], bbox[0], bbox[3])], crs="EPSG:4326").iloc[0]
        osm_edges = get_osm_drivable_edges(bbox_poly, city_edges_gdf, city_edges_sindex)
        original_osm_edges, buffered_osm_edges = osm_edges
        osm_mask = create_osm_mask(buffered_osm_edges, (IMAGE_SIZE, IMAGE_SIZE), bbox)

        valid_masks = []
        excluded_masks = []

        for mask_data in masks:
            mask = mask_data["segmentation"]
            
            # Calculate real-world areas
            overlap_area = calculate_real_world_area(np.logical_and(mask, osm_mask), bbox, (IMAGE_SIZE, IMAGE_SIZE))
            mask_area = calculate_real_world_area(mask, bbox, (IMAGE_SIZE, IMAGE_SIZE))
            
            if mask_area < MIN_CROSSWALK_AREA:  # Preserve small valid crosswalks
                valid_masks.append(mask_data)
                continue
            
            relative = overlap_area / mask_area if mask_area > 0 else 0
            if (overlap_area > MAX_ABSOLUTE_OVERLAP) or (relative > MAX_RELATIVE_OVERLAP):
                excluded_masks.append(mask_data)
            else:
                valid_masks.append(mask_data)
        
        # After separating valid/excluded masks:
        valid_detections = sv.Detections.from_sam(valid_masks)
        excluded_detections = sv.Detections.from_sam(excluded_masks)
        excluded_detections.class_id = np.array([-1] * len(excluded_detections))

        # Annotate valid masks first (index colors)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated = mask_annotator.annotate(resized_img, valid_detections)

        excluded_annotator = sv.PolygonAnnotator(color=sv.Color(r=0, g=0, b=255), thickness=2)
        annotated = excluded_annotator.annotate(annotated, excluded_detections)
        torch.cuda.empty_cache()
        print(f"Valid Masks: {len(valid_masks)}, Excluded Masks: {len(excluded_masks)}")
        
        # After annotating masks:
        original_osm_mask = create_osm_mask(original_osm_edges, (IMAGE_SIZE, IMAGE_SIZE), bbox, is_lines=True)
        buffered_osm_mask = osm_mask

        # Draw original OSM edges (blue)
        osm_lines_contours, _ = cv2.findContours(original_osm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, osm_lines_contours, -1, (0, 0, 255), 2)

        # Draw buffered OSM areas (green)
        osm_buffer_contours, _ = cv2.findContours(buffered_osm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, osm_buffer_contours, -1, (0, 255, 0), 2)

        # Save the annotated image
        async_write(output_path, rgb_to_bgr(annotated))
        print(f"Saved output to: {output_path}")
        processed_count += 1

        if mode == "test":
            collected_images.append(
                wandb.Image(
                    annotated,
                    caption=f"Processed image: {image_name}"
                )
            )

        final_mask = process_masks_parallel(valid_masks, original_size)
        
        # Find contours more efficiently
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out tiny contours early
        MIN_CONTOUR_AREA = 10  # in pixels
        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        
        centroid_lon = (bbox[0] + bbox[2])/2
        centroid_lat = (bbox[1] + bbox[3])/2
        utm_crs = get_utm_projection(centroid_lon, centroid_lat)
        
        # Only process geometry if we have valid contours
        if contours:
            valid_polys = process_geometry(contours, original_size, bbox, utm_crs)
            
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
            
        # Final simplification with optimized parameters
        inverse_geometry = make_valid(inverse_geometry) if not inverse_geometry.is_valid else inverse_geometry
        inverse_geometry = inverse_geometry.simplify(0.000003, preserve_topology=True)
        shapefile_data.append({
            "image_name": image_name,
            "geometry": inverse_geometry
        })
        print(f"Image: {image_name}")
        print(f"Contours found: {len(contours)}")
        print(f"Valid polygons: {len(valid_polys) if 'valid_polys' in locals() else 0}")

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images checked: {len(input_set)}")
    print(f"Images processed: {processed_count}")
    print(f"Images skipped (already processed): {skipped_count}")
    
    if shapefile_data:
        gdf = gpd.GeoDataFrame(shapefile_data, geometry="geometry", crs="EPSG:4326")
        
        # Clean up before GeoDataFrame operations (which can be memory intensive)
        gc_collect()
        
        export_geodata(gdf, f"/outputs/{model_path}/geofiles")
    
    # Cleanup and shutdown executors
    io_executor.shutdown(wait=True)
    
    # Final cleanup of any temporary files or memory
    gc_collect()
    
    if mode == "test":
        wandb.log({
            "processed_images": collected_images,
            "images_processed": processed_count,
            "images_skipped": skipped_count
        })
        wandb.finish()
                
@app.local_entrypoint()
def main(model: str, mode: str, force: bool = False):
    city_code = os.environ.get("MODAL_ENVIRONMENT")
    run_inference.remote(
        model_path=model,
        mode=mode,
        city_code=city_code,
        force=force
    )