import modal
import os
from dotenv import load_dotenv
import math
from utils import decode_crosswalk_id, coords_from_distance, bounding_box_from_filename

load_dotenv()

CITY_CODE = os.getenv("CITY_CODE")

INPUTS_VOLUME_NAME = f"crosswalk-data-{CITY_CODE}"
OUTPUTS_VOLUME_NAME = f"{INPUTS_VOLUME_NAME}-results"

app = modal.App(name="crossing-distance-sam2-inference")

infer_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "python3-opencv", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python==4.10.0.84",
        "pycocotools~=2.0.8",
        "matplotlib~=3.9.2",
        "supervision",
        "fiona",
        "geopandas",
        "shapely",
        "python-dotenv"
    )
    .run_commands(f"git clone https://git@github.com/facebookresearch/sam2.git")
    .run_commands("pip install -e sam2/.")
    .run_commands("pip install -e 'sam2/.[dev]'")
    .run_commands("cd 'sam2/checkpoints'; ./download_ckpts.sh")
)

weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")
inputs_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, environment_name=CITY_CODE)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True, environment_name=CITY_CODE)


@app.function(volumes={"/weights": weights_volume, 
                       "/inputs": inputs_volume,
                       "/outputs": outputs_volume}, 
              image=infer_image, gpu="A100", timeout=36000)
def run_inference(model_path: str, mode: str):
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
    from shapely.geometry import Polygon, box, MultiPolygon
    from shapely.validation import make_valid
    from shapely.ops import unary_union
    import json

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = f"../../weights/{model_path}/checkpoints/checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    input_set = os.listdir("/inputs")

    if mode == "test":
        input_set = random.sample(input_set, min(100, len(input_set)))
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be test or full.")

    output_dir = f"/outputs/{model_path}"
    geofile_dir = f"/outputs/{model_path}/geofiles" 
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(geofile_dir, exist_ok=True)

    shapefile_data = []

    for image in input_set:
        image_name = os.path.splitext(os.path.basename(image))[0]
        image_path = os.path.join("/inputs", image)

        bounding_box_coords = bounding_box_from_filename(image_name)

        original_image = Image.open(image_path).convert("RGB")
        original_width, original_height = original_image.size

        resized_image = original_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        resized_image = np.array(resized_image)

        result = mask_generator.generate(resized_image)
        detections = sv.Detections.from_sam(sam_result=result)

        if detections.mask is None or len(detections.mask) == 0:
            print(f"No detections generated for image: {image_name}. Skipping.")
            continue
        
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
        annotated_image = resized_image.copy()
        annotated_image = mask_annotator.annotate(annotated_image, detections=detections)

        annotated_image_path = os.path.join(output_dir, f"{image_name}_masked.jpg")
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(annotated_image_path, annotated_image_bgr)

        final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for mask in detections.mask:
            mask_resized = cv2.resize((mask>0).astype(np.uint8), 
                                    (original_width, original_height), 
                                    interpolation=cv2.INTER_NEAREST)
            final_mask = cv2.bitwise_or(final_mask, mask_resized)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and validate polygons
        valid_polygons = []
        invalid_polygons = []
            
        for contour in contours: 
            if len(contour) >= 3:
                geo_points = []
                for point in contour.squeeze(1):
                    # Convert relative to geographic coordinates
                    rel_x = point[0] / original_width
                    rel_y = point[1] / original_height
                    
                    # Calculate geographic coordinates
                    lon = rel_x * (bounding_box_coords[0] - bounding_box_coords[2]) + bounding_box_coords[2]
                    lat = rel_y * (bounding_box_coords[1] - bounding_box_coords[3]) + bounding_box_coords[3]
                    
                    geo_points.append((lon, lat))
            
                polygon = Polygon(geo_points)
                if polygon.is_valid:
                    valid_polygons.append(polygon)
                else:
                    invalid_polygons.append(polygon)

        if valid_polygons:
            try:
                mask_union = unary_union(valid_polygons)
            except Exception as e:
                print(f"Error during union of valid polygons: {e}")
                mask_union = None
        else:
            mask_union = None
        
        final_union = mask_union  # Start with the unioned result
        if invalid_polygons:
            for poly in invalid_polygons:
                try:
                    # Merge invalid polygons one by one (unioning individually to avoid failure)
                    if final_union:
                        final_union = final_union.union(poly)
                    else:
                        final_union = poly
                except Exception as e:
                    print(f"Skipping invalid geometry due to error: {e}")

        bounding_box_polygon = box(
        bounding_box_coords[2],  # min longitude (west)
        bounding_box_coords[1],  # min latitude (south)
        bounding_box_coords[0],  # max longitude (east)
        bounding_box_coords[3],  # max latitude (north)
        )
        if not final_union.is_valid:
            print("Final union has invalid geometries, cleaning for difference operation...")
            final_union = make_valid(final_union)
        inverted_selection = bounding_box_polygon.difference(final_union)
    
        if isinstance(inverted_selection, MultiPolygon):
            largest_polygon = max(inverted_selection.geoms, key=lambda p: p.area)  # Find the largest polygon
        else:
            largest_polygon = inverted_selection

        shapefile_data.append({"image_name": image_name, "geometry": largest_polygon})

    # After all masks are processed, create the GeoDataFrame
    gdf = gpd.GeoDataFrame(shapefile_data, crs="EPSG:4326")

    # Export to GeoJSON
    geojson_path = f"{geofile_dir}/cross_walks.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    # Export to KML
    kml_path = f"{geofile_dir}/cross_walks.kml"
    gdf.to_file(kml_path, driver="KML")

    # Export to Shapefile
    shapefile_path = f"{geofile_dir}/cross_walks.shp"
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")
                
@app.local_entrypoint()
def main(model: str, mode: str):
    run_inference.remote(model_path=model, mode=mode)