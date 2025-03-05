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


load_dotenv()

from utils import PRECISION, get_crosswalk_id, decode_crosswalk_id, coords_from_distance, bounding_box_from_filename

IMAGE_SIZE = 1024
CITY_CODE = os.getenv("CITY_CODE")
INPUTS_VOLUME_NAME = f"crosswalk-data-{CITY_CODE}"
OUTPUTS_VOLUME_NAME = f"{INPUTS_VOLUME_NAME}-results"
CITY_CODE_MAPPING_CSV_PATH = "../data/city_code_mapping.csv" # Path to CSV mapping file

MASK_GENERATOR_CONFIG = {
    "points_per_side": 16,
    "points_per_batch": 64,
    "crop_n_layers": 1,
    "pred_iou_thresh": 0.89,
    "stability_score_thresh": 0.91,
    "min_mask_region_area": 800,
    "box_nms_thresh": 0.78,
}

MAX_RELATIVE_OVERLAP = 0.1
MAX_ABSOLUTE_OVERLAP = 15.0  # meters²
MIN_CROSSWALK_AREA = 5.0  # meters²
OSM_BUFFER_METERS = 1

def async_write(path, image):
    Thread(target=cv2.imwrite, args=(path, image)).start()

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

def load_allowed_crosswalk_ids(csv_path: str) -> set:
    """
    Load the CSV file and generate a set of crosswalk IDs from the coordinates.
    """
    allowed_ids = set()
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            longitude = float(row['x'])
            latitude = float(row['y'])
            
            # Round to PRECISION to match filename encoding
            rounded_lat = round(latitude, PRECISION)
            rounded_lon = round(longitude, PRECISION)
            
            try:
                crosswalk_id = get_crosswalk_id(rounded_lat, rounded_lon)
                allowed_ids.add(crosswalk_id)
            except AssertionError as e:
                print(f"Skipping invalid coordinate ({rounded_lat}, {rounded_lon}): {e}")
    return allowed_ids

def filter_images(image_dir: str, allowed_ids: set) -> list:
    """
    Return a list of image paths whose crosswalk ID is in allowed_ids.
    """
    valid_images = []
    for filename in os.listdir(image_dir):
        # Remove file extension
        filename_without_ext = os.path.splitext(filename)[0]
        
        parts = filename_without_ext.split('_', 1)  # Split at the first underscore
        
        # Validate filename format
        if len(parts) != 2 or parts[0] != "crosswalk":
            print(f"Skipping invalid filename format: {filename}")
            continue
        
        crosswalk_id = parts[1]  # Actual ID (e.g., "37708269N_122423822W")
        
        if crosswalk_id in allowed_ids:
            valid_images.append(os.path.join(image_dir, filename))
    
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
    Loads city-wide OSM drivable edges and spatial index, or queries them directly every time without caching.
    Caching is now disabled.
    """

    print(f"Querying OSM for drivable edges in {city_code}... (Caching disabled)")
    try:
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
        
        # Create a comprehensive filter
        # 1. Must have highway tag
        has_highway = city_edges_gdf['highway'].notna()
        
        # 2. Must be of a pedestrian-compatible road type
        is_pedestrian_road = city_edges_gdf['highway'].isin(pedestrian_road_types)
        
        # 3. Exclude specific service roads
        not_service_road = True
        if 'service' in city_edges_gdf.columns:
            not_service_road = ~city_edges_gdf['service'].isin(['parking_aisle', 'driveway', 'drive-through', 'parking'])
        
        # 4. Exclude private roads
        not_private = True
        if 'access' in city_edges_gdf.columns:
            not_private = ~city_edges_gdf['access'].isin(['private', 'no'])
        
        # 5. Exclude parking amenities
        not_parking = True
        if 'amenity' in city_edges_gdf.columns:
            not_parking = ~city_edges_gdf['amenity'].isin(['parking', 'parking_entrance', 'parking_space'])
        
        # 6. Exclude park paths and boundaries
        not_park_path = True
        if 'landuse' in city_edges_gdf.columns:
            not_park_path = ~city_edges_gdf['landuse'].isin(['park', 'garden', 'forest', 'recreation_ground', 'grass', 'meadow'])
        
        # 7. Exclude leisure areas
        not_leisure = True
        if 'leisure' in city_edges_gdf.columns:
            not_leisure = ~city_edges_gdf['leisure'].isin(['park', 'garden', 'nature_reserve', 'playground', 'pitch', 'golf_course'])
        
        # 8. Exclude boundary features
        not_boundary = True
        if 'boundary' in city_edges_gdf.columns:
            not_boundary = city_edges_gdf['boundary'].isna()
        
        # 9. Exclude paths not used for crossing
        valid_footway = True
        if 'footway' in city_edges_gdf.columns:
            valid_footway = ~city_edges_gdf['footway'].isin(['sidewalk', 'access_aisle']) | (city_edges_gdf['footway'].isna())

        not_area = True
        if 'area' in city_edges_gdf.columns:
            not_area = (city_edges_gdf['area'] != 'yes') | city_edges_gdf['area'].isna()
        
        # Combine all filters
        mask = (
            has_highway & 
            is_pedestrian_road & 
            not_service_road & 
            not_private & 
            not_parking & 
            not_park_path & 
            not_leisure & 
            not_boundary &
            valid_footway &
            not_area
        )
        
        city_edges_gdf = city_edges_gdf[mask]

        # Create spatial index
        city_edges_sindex = city_edges_gdf.sindex

        print(f"OSM data queried and spatial index created for {city_code} (No caching).")
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
    """Create binary mask of OSM edges in image coordinates."""
    osm_mask = np.zeros(image_size, dtype=np.uint8)
    width, height = image_size

    def process_line(line):
        coords = []
        for x, y in line.coords:
            pixel_x = int((x - bbox[2]) / (bbox[0] - bbox[2]) * width)
            pixel_y = int((y - bbox[3]) / (bbox[1] - bbox[3]) * height)
            coords.append((pixel_x, pixel_y))
        if len(coords) >= 2:
            # Draw lines between consecutive points
            for i in range(len(coords) - 1):
                cv2.line(osm_mask, (int(coords[i][0]), int(coords[i][1])),
                        (int(coords[i+1][0]), int(coords[i+1][1])), 1, thickness=2)
    
    def process_polygon(polygon):
        """Helper function to process a single polygon"""
        if polygon.is_empty:
            return
            
        # Convert geographic coordinates to image pixels
        coords = []
        for x, y in polygon.exterior.coords:
            pixel_x = int((x - bbox[2]) / (bbox[0] - bbox[2]) * width)
            pixel_y = int((y - bbox[3]) / (bbox[1] - bbox[3]) * height)
            coords.append((pixel_x, pixel_y))
        
        # Draw on both masks if coords exist
        if len(coords) > 2:  # Need at least 3 points for a polygon
            cv2.fillPoly(osm_mask, [np.array(coords)], 1)
    
    for geom in geometries:
        if isinstance(geom, LineString):
            process_line(geom)
        elif isinstance(geom, MultiPolygon):
            for polygon in geom.geoms:
                process_polygon(polygon)
        elif isinstance(geom, Polygon):
            process_polygon(geom)

    return osm_mask

def calculate_real_world_area(mask, bbox, image_size):
    """
    Calculate real-world area of overlapping regions in square meters.
    Uses Haversine formula for geographic area calculation.
    """
    # Get image dimensions
    height, width = image_size
    
    # Calculate degrees per pixel
    lon_per_pixel = (bbox[0] - bbox[2]) / width
    lat_per_pixel = (bbox[1] - bbox[3]) / height
    
    # Earth radius in meters
    R = 6378137
    
    # Calculate area per pixel at this latitude (approximation)
    avg_lat = (bbox[1] + bbox[3]) / 2
    pixel_width = 2 * np.pi * R * np.cos(np.radians(avg_lat)) * lon_per_pixel / 360
    pixel_height = 2 * np.pi * R * lat_per_pixel / 360
    area_per_pixel = abs(pixel_width * pixel_height)
    
    # Calculate overlapping area in square meters
    overlap_pixels = np.sum(mask)
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
    # Validate and simplify geometries
    def get_utm_area(geom):
        centroid = geom.centroid
        utm_crs = get_utm_projection(centroid.x, centroid.y)
        return gpd.GeoSeries([geom], crs=4326).to_crs(utm_crs).area[0]
    
    gdf['area_m2'] = gdf.geometry.apply(get_utm_area)
    gdf = gdf[gdf.area_m2 > 0.5]  # Filter small artifacts
    
    # Final simplification pass
    gdf.geometry = gdf.geometry.simplify(0.000003, preserve_topology=True)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    formats = {"geojson": "GeoJSON", "kml": "KML", "shp": "ESRI Shapefile"}
    for ext, driver in formats.items():
        path = os.path.join(output_dir, f"cross_walks.{ext}")
        gdf.to_file(path, driver=driver)
                
def main():
    parser = argparse.ArgumentParser(description='SAM2 Crosswalk Detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'], 
                       help='Run mode: "test" (sample 100 images) or "full"')
    args = parser.parse_args()

    # Get environment variables
    city_code = os.getenv("CITY_CODE")
    if not city_code:
        raise ValueError("CITY_CODE environment variable not set")
    
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY environment variable not set")

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        args.model_path,
        device=device
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_CONFIG)

    csv_path = os.path.join(args.input_dir, f"{city_code}/scratch/intersection_coordinates_v2_0.csv")
    image_dir = os.path.join(args.input_dir, f"{city_code}/crosswalk-data-{city_code}")
    allowed_ids = load_allowed_crosswalk_ids(csv_path)
    input_set = filter_images(image_dir, allowed_ids)
    print(f"Found {len(input_set)} images to process.")

    image_output_dir = os.path.join(args.output_dir, f"{city_code}/images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Load city code mapping from CSV
    city_code_to_place_mapping = load_city_code_mapping_from_csv(CITY_CODE_MAPPING_CSV_PATH)
    if not city_code_to_place_mapping: # Check if loading failed
        print("City code mapping loading failed. Exiting.")
        return

    # Load city-wide OSM data and spatial index (or create and cache if not present)
    city_edges_gdf, city_edges_sindex = load_city_osm_data(city_code, city_code_to_place_mapping)

    if city_edges_gdf.empty or city_edges_sindex is None:
        print("City OSM data loading failed. Exiting.")
        return  # Exit if city OSM data couldn't be loaded


    wandb.init(
        project="pedestrian-crossing-distance",
        name=f"{city_code}-local-run",
        tags=[city_code, "local-run", "drivable-filter", "spatial-index-optim", args.mode],
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
            "model_path": args.model_path,
            "mode": args.mode,
            "city_code": city_code,
            "osm_method": "spatial_index",
            "city_mapping_source": "city_code_mapping.csv"
        }
    )
    
    if args.mode == "test":
        collected_images = []
        random.seed(sum(ord(c) for c in city_code))
        input_set = random.sample(input_set, min(100, len(input_set)))

    shapefile_data = []

    # Process images
    for image_file in input_set:
        print(f"Processing: {image_file}")
        image_name = os.path.splitext(image_file)[0].split("/")[-1]
        image_path = os.path.join(args.input_dir, image_file)
        bbox = bounding_box_from_filename(image_name)

        img = cv2.imread(image_path)
        original_size = (img.shape[0], img.shape[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
            resized_img_rgb = bgr_to_rgb(resized_img)
        else:
            resized_img_rgb = resized_img

        with torch.no_grad():
            masks = mask_generator.generate(resized_img)
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
        output_path = os.path.join(image_output_dir, f"{image_name}_masked.jpg")
        
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

        if args.mode == "test":
            collected_images.append(
                wandb.Image(
                    annotated,
                    caption=f"Processed image: {image_name}"
                )
            )

        final_mask = np.zeros(original_size[::-1], dtype=np.uint8)
        if valid_detections.mask is not None:
            for mask in valid_detections.mask:
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                original_size, 
                                interpolation=cv2.INTER_NEAREST)
                np.putmask(final_mask, resized_mask > 0.5, 1)
        
        final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (3, 3), 0)
        _, final_mask = cv2.threshold(final_mask, 0.5, 1, cv2.THRESH_BINARY)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, 
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        final_mask = cv2.medianBlur(final_mask, 3)
        final_mask = final_mask.astype(np.uint8)
        final_mask = cv2.GaussianBlur(final_mask, (5,5), 1)  # Larger blur
        final_mask = (final_mask > 0.1).astype(np.uint8)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        centroid_lon = (bbox[0] + bbox[2])/2
        centroid_lat = (bbox[1] + bbox[3])/2
        utm_crs = get_utm_projection(centroid_lon, centroid_lat)
        
        # Process geometry with UTM CRS
        valid_polys = process_geometry(contours, original_size, bbox, utm_crs)
        
        if valid_polys:
            try:
                union = unary_union(valid_polys)
                # Clip union to image bounds with buffer
                union = union.intersection(bbox_poly.buffer(0.00001))  # 1m buffer
                if union.is_empty:
                    inverse_geometry = bbox_poly
                else:
                    inverse_geometry = bbox_poly.difference(union)
                    # Ensure minimum size
                    if inverse_geometry.area < bbox_poly.area * 0.01:
                        inverse_geometry = bbox_poly
            except:
                inverse_geometry = bbox_poly
        else:
            inverse_geometry = bbox_poly

        # Final validation and repair
        inverse_geometry = make_valid(inverse_geometry) if not inverse_geometry.is_valid else inverse_geometry
        inverse_geometry = inverse_geometry.simplify(0.000003, preserve_topology=True)
        shapefile_data.append({
            "image_name": image_name,
            "geometry": inverse_geometry
        })
        print(f"Image: {image_name}")
        print(f"Contours found: {len(contours)}")
        print(f"Valid polygons: {len(valid_polys)}")

    # Data export
    if shapefile_data:
        gdf = gpd.GeoDataFrame(shapefile_data, geometry="geometry", crs="EPSG:4326")
        geo_output_dir = os.path.join(args.output_dir, f"{city_code}/geofiles")
        export_geodata(gdf, geo_output_dir)

    if args.mode == "test":
        wandb.log({
            "processed_images": collected_images
        })
        wandb.finish()

if __name__ == "__main__":
    load_dotenv()
    main()