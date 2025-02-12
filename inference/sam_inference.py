import modal
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

load_dotenv()

from utils import decode_crosswalk_id, coords_from_distance, bounding_box_from_filename

IMAGE_SIZE = 1024
CITY_CODE = os.getenv("CITY_CODE")
INPUTS_VOLUME_NAME = f"crosswalk-data-{CITY_CODE}"
OUTPUTS_VOLUME_NAME = f"{INPUTS_VOLUME_NAME}-results"

MASK_GENERATOR_CONFIG = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.90,
    "crop_n_layers": 0,
    "min_mask_region_area": 500,
}

app = modal.App(name="crossing-distance-sam2-inference")

infer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "python3-opencv", "ffmpeg", "build-essential", "ninja-build", "cmake", "g++")
    .run_commands(
        # Add NVIDIA CUDA repository
        "wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb",
        "dpkg -i cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb",
        "cp /var/cuda-repo-debian11-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-1",
        "rm cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "SAM2_BUILD_ALLOW_ERRORS": "0",  # Force CUDA extension build
        "PATH": "/usr/local/cuda-12.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6"  # Specify CUDA architectures
    })
    # Install numpy first to avoid the warning
    .pip_install("numpy")
    .pip_install(
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "torchaudio",
        "opencv-python==4.10.0.84",
        "pycocotools~=2.0.8",
        "matplotlib~=3.9.2",
        "supervision",
        "fiona",
        "geopandas",
        "shapely",
        "python-dotenv",
        "wandb",
        "osmnx"
    )
    .run_commands(
        # Clean any previous installations
        "rm -rf sam2",
        # Clone and install SAM2
        "git clone https://git@github.com/facebookresearch/sam2.git",
        # Force rebuild of CUDA extension
        "cd sam2 && rm -f ./sam2/*.so",
        "cd sam2 && pip install -e .",
        "cd sam2 && pip install -e '.[dev]'",
        "cd sam2 && pip install -v -e '.[notebooks]'"  # Added -v for verbose output
    ))

weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")
inputs_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, environment_name=CITY_CODE)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True, environment_name=CITY_CODE)

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

def get_osm_drivable_edges(bbox_polygon):
    """Fetch OSM roads while excluding parking lots and private roads"""
    try:
        # First get all highway features
        gdf = ox.features_from_polygon(
            bbox_polygon,
            tags={"highway": True},
        )
        
        # Define acceptable road types for pedestrian crossings
        pedestrian_roads = [
            'primary', 'secondary', 'tertiary', 
            'unclassified', 'residential', 
            'living_street', 'pedestrian'
        ]
        
        # Start with highway type filter
        mask = gdf['highway'].isin(pedestrian_roads)
        
        # Add additional filters only if the columns exist
        if 'service' in gdf.columns:
            mask &= ~gdf['service'].isin(['parking_aisle', 'driveway'])
        
        if 'access' in gdf.columns:
            mask &= (gdf['access'] != 'private')
            
        if 'amenity' in gdf.columns:
            mask &= (gdf['amenity'] != 'parking')
        
        filtered_gdf = gdf[mask]
        
        center_point = bbox_polygon.centroid
        utm_zone = int((center_point.x + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if center_point.y >= 0 else f"EPSG:327{utm_zone:02d}"
        
        # Project, buffer, and project back
        projected_gdf = filtered_gdf.to_crs(utm_crs)
        buffered = projected_gdf.geometry.buffer(1.5)  # 1.5 meters buffer
        return buffered.to_crs(filtered_gdf.crs)  # Return to original CRS
        
    except Exception as e:
        print(f"OSM query failed: {str(e)}")
        return []

def create_osm_mask(osm_edges, image_size, bbox):
    """Create binary mask of OSM edges in image coordinates."""
    osm_mask = np.zeros(image_size, dtype=np.uint8)
    width, height = image_size
    
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
    
    for geom in osm_edges:
        if isinstance(geom, MultiPolygon):
            # Process each polygon in the MultiPolygon
            for polygon in geom.geoms:
                process_polygon(polygon)
        elif isinstance(geom, Polygon):
            # Process single polygon
            process_polygon(geom)
        else:
            print(f"Unsupported geometry type: {type(geom)}")
            continue

    return osm_mask

def draw_negative_buffers(image, negative_points, buffer_size, color=(0, 0, 255)):
    """Draw negative points and their buffers on the image"""
    vis_image = image.copy()
    
    if negative_points:
        for x, y in negative_points:
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                # Draw the buffer circle
                cv2.circle(vis_image, (x, y), buffer_size, color, 2)
                # Draw the center point
                cv2.circle(vis_image, (x, y), 3, color, -1)
    
    return vis_image

def process_geometry(contours, original_size, bounding_box):
    original_width, original_height = original_size
    valid_polygons = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        geo_points = []
        for point in contour.squeeze(1):
            rel_x = point[0] / original_width
            rel_y = point[1] / original_height
            lon = rel_x * (bounding_box[0] - bounding_box[2]) + bounding_box[2]
            lat = rel_y * (bounding_box[1] - bounding_box[3]) + bounding_box[3]
            geo_points.append((lon, lat))
        
        try:
            polygon = Polygon(geo_points).simplify(0.00001).buffer(0.000001)
            if polygon.is_valid:
                valid_polygons.append(polygon)
        except Exception as e:
            print(f"Error creating polygon: {e}")
            continue
    
    return valid_polygons

def export_geodata(gdf, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    formats = {
        "geojson": "GeoJSON",
        "kml": "KML",
        "shp": "ESRI Shapefile"
    }
    
    for ext, driver in formats.items():
        path = os.path.join(output_dir, f"cross_walks.{ext}")
        gdf.to_file(path, driver=driver)

@app.function(volumes={"/weights": weights_volume, 
                       "/inputs": inputs_volume,
                       "/outputs": outputs_volume}, 
              image=infer_image, gpu="A100", 
              secrets=[modal.Secret.from_name("wandb-secret")],
              timeout=36000)
def run_inference(model_path: str, mode: str, city_code: str, negative_points: list = None, buffer_size: int = 20):
    import torch
    import cv2
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import supervision as sv
    import os
    import random
    from PIL import Image
    import numpy as np
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import LineString, Polygon, box, MultiPolygon
    from shapely.validation import make_valid
    from shapely.ops import unary_union
    import json
    import wandb

    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # Model initialization
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        f"../../weights/{model_path}/checkpoints/checkpoint.pt",
        device="cuda"
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_CONFIG)
    
    # Data setup
    input_set = os.listdir("/inputs")
    output_dir = f"/outputs/{model_path}/images"
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == "test":
        collected_images = [] # List to store images for wandb logging
        random.seed(sum(ord(c) for c in city_code))
        input_set = random.sample(input_set, min(100, len(input_set)))

    # WANDB initialization
    if mode == "test":
        wandb.init(
            project="pedestrian-crossing-distance",
            name=f"{city_code}-{model_path}-{'with_drive_buffers'}-test",
            config={"model_path": model_path, "mode": mode, "city_code": city_code}
        )

    shapefile_data = []

    for image in input_set:
        image_name = os.path.splitext(os.path.basename(image))[0]
        image_path = os.path.join("/inputs", image)
        bbox = bounding_box_from_filename(image_name)

        with Image.open(image_path) as img:
            original_size = img.size
            resized_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
            resized_img_rgb = bgr_to_rgb(resized_img)
        else:
            resized_img_rgb = resized_img

        masks = mask_generator.generate(resized_img)
        bbox_poly = box(bbox[2], bbox[1], bbox[0], bbox[3])
        osm_edges = get_osm_drivable_edges(bbox_poly)
        osm_mask= create_osm_mask(osm_edges, (IMAGE_SIZE, IMAGE_SIZE), bbox)

        valid_masks = []
        for mask_data in masks:
            mask = mask_data["segmentation"]
            
            # Calculate overlap with OSM drivable areas
            overlap = np.sum(np.logical_and(mask, osm_mask))
            total_area = np.sum(mask)
            
            # Reject masks with >10% overlap with drivable surfaces
            if total_area > 0 and (overlap / total_area) < 0.1:
                valid_masks.append(mask_data)

        if negative_points:
            filtered_masks = []
            for mask_data in valid_masks:  # Start with OSM-filtered masks
                mask_intersects_buffer = False
                segmentation = mask_data["segmentation"]
                
                for x, y in negative_points:
                    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                        # Create circular buffer
                        y_indices, x_indices = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                        distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
                        buffer_mask = distances <= buffer_size
                        
                        if np.any(segmentation & buffer_mask):
                            mask_intersects_buffer = True
                            break
                
                if not mask_intersects_buffer:
                    filtered_masks.append(mask)
            
            valid_masks = filtered_masks
        
        detections = sv.Detections.from_sam(valid_masks)
        
        # Mask annotation
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated = mask_annotator.annotate(resized_img, detections)
        output_path = os.path.join(output_dir, f"{image_name}_masked.jpg")

        if negative_points:
            annotated = draw_negative_buffers(annotated, negative_points, buffer_size)
        
        # Draw included OSM roads in green
        osm_contours, _ = cv2.findContours(osm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, osm_contours, -1, (0, 255, 0), 1)  # BGR color for green

        # Save the annotated image
        cv2.imwrite(output_path, rgb_to_bgr(annotated))

        if mode == "test":
            collected_images.append(
                wandb.Image(
                    annotated,
                    caption=f"Processed image: {image_name}"
                )
            )

        final_mask = np.zeros(original_size[::-1], dtype=np.uint8)
        if detections.mask is not None:
            for mask in detections.mask:
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                original_size, 
                                interpolation=cv2.INTER_LINEAR)
                np.putmask(final_mask, resized_mask > 0.5, 1)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_polys = process_geometry(contours, original_size, bbox)
        
        if valid_polys:
            union = unary_union(valid_polys)
            bbox_poly = box(bbox[2], bbox[1], bbox[0], bbox[3])
            shapefile_data.append({
                "image_name": image_name,
                "geometry": bbox_poly.difference(union)
            })

    # Data export
    if shapefile_data:
        gdf = gpd.GeoDataFrame(shapefile_data, geometry="geometry", crs="EPSG:4326")
        export_geodata(gdf, f"/outputs/{model_path}/geofiles")

    if mode == "test":
        wandb.log({
            "processed_images": collected_images
        })
        wandb.finish()
                
@app.local_entrypoint()
def main(model: str, mode: str, negative_points: str = None, buffer_size: int = 20):
    city_code = os.environ.get("MODAL_ENVIRONMENT")
    run_inference.remote(
        model_path=model,
        mode=mode,
        city_code=city_code,
    )