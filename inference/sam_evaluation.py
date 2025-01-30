import modal

import os
from pathlib import Path
import time
import json

import cv2
import numpy as np
from shapely.geometry import Polygon
from pycocotools import mask as mask_util

# Use the same app configuration as sam_inference.py
app = modal.App(name="crossing-distance-sam2-evaluation")

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
        "shapely",
        "wandb"
    )
    .run_commands(f"git clone https://git@github.com/DSC-Qian/sam2.git")
    .run_commands("pip install -e sam2/.")
    .run_commands("pip install -e 'sam2/.[dev]'")
    .run_commands("cd 'sam2/checkpoints'; ./download_ckpts.sh")
)

weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")
outputs_volume = modal.Volume.from_name("eval-results", create_if_missing=True)

def mask_to_polygons(mask):
    """Convert binary mask to multiple polygons"""
    import cv2
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Need at least 3 points for a polygon
            polygon = [(point[0][0], point[0][1]) for point in contour]
            poly = Polygon(polygon)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
    
    if len(polygons) == 0:
        return None
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)

def draw_comparison(image, gt_polygons, pred_polygons):
    """Draw ground truth and prediction overlays handling multiple polygons"""
    import cv2
    import numpy as np
    from shapely.geometry import MultiPolygon
    
    overlay = np.array(image.copy())
    
    # Convert single polygons to MultiPolygon for consistent handling
    if gt_polygons is not None:
        if not isinstance(gt_polygons, MultiPolygon):
            gt_polygons = MultiPolygon([gt_polygons])
        # Draw ground truth in green
        for polygon in gt_polygons.geoms:
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(overlay, [coords], True, (0, 255, 0), 2)
    
    if pred_polygons is not None:
        if not isinstance(pred_polygons, MultiPolygon):
            pred_polygons = MultiPolygon([pred_polygons])
        # Draw predictions in red
        for polygon in pred_polygons.geoms:
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(overlay, [coords], True, (255, 0, 0), 2)
    
    return overlay

def calculate_iou(poly1, poly2):
    """Calculate Intersection over Union between two polygons or multipolygons"""
    if poly1 is None or poly2 is None:
        return 0.0
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0

def decode_rle_to_mask(segmentation, height, width):
    """Convert RLE segmentation to binary mask"""
    import numpy as np
    from pycocotools import mask as mask_util
    
    if isinstance(segmentation, dict):
        # RLE format
        mask = mask_util.decode(segmentation)
    else:
        # Polygon format
        rles = mask_util.frPyObjects(segmentation, height, width)
        mask = mask_util.decode(rles)
    
    return mask.astype(np.uint8)

@app.function(mounts=[modal.Mount.from_local_dir("../data/images/Satellite-Curb-Segmentation-12/test", remote_path="/inputs")],
              volumes={"/weights": weights_volume, 
                      "/outputs": outputs_volume}, 
              image=infer_image, 
              gpu="A100", 
              secrets=[modal.Secret.from_name("wandb-secret")],
              timeout=36000)
def evaluate_model(model_path: str):
    """
    Evaluates SAM2 model on annotated images.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    import wandb
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from shapely.geometry import Polygon
    from pycocotools import mask as mask_util


    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    
    # Set up CUDA optimizations
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    wandb.init(project="pedestrian-crossing-distance", 
               name=f"eval_{model_path}", 
               job_type="evaluation")

    # Build SAM model
    checkpoint = f"../../weights/{model_path}/checkpoints/checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    eval_metrics = {
        'image_iou': [],
        'inference_times': [],
        'images': []
    }

    output_base = Path("/outputs") / model_path
    vis_dir = output_base / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_path in Path("/inputs").glob("*.jpg"):
        img_name = img_path.stem
        annotation_path = img_path.parent / f"{img_name}.json"
        
        print(f"Processing {img_name}")
        
        # Load image and ground truth
        image = Image.open(img_path).convert("RGB")
        with open(annotation_path) as f:
            data = json.load(f)
            
        # Get image dimensions
        height = data['image']['height']
        width = data['image']['width']
        
        # Combine all annotations into one mask
        gt_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in data['annotations']:
            mask = decode_rle_to_mask(ann['segmentation'], height, width)
            gt_mask = cv2.bitwise_or(gt_mask, mask)
            
        # Convert ground truth mask to polygons
        gt_polygons = mask_to_polygons(gt_mask)
        
        # Time inference
        start_time = time.time()
        resized_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        result = mask_generator.generate(np.array(resized_image))
        inference_time = time.time() - start_time

        pred_mask = np.zeros((height, width), dtype=np.uint8)
        for r in result:  # result is the output from SAM
            pred_mask = cv2.bitwise_or(pred_mask, r['segmentation'].astype(np.uint8))
        pred_polygons = mask_to_polygons(pred_mask)

        # Calculate IoU
        iou = calculate_iou(gt_polygons, pred_polygons)

        # Draw comparison
        overlay_img = draw_comparison(image, gt_polygons, pred_polygons)
        
        # Save visualization
        output_path = vis_dir / f"{img_name}_eval.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        
        # Log to wandb
        eval_metrics['image_iou'].append(iou)
        eval_metrics['inference_times'].append(inference_time)
        eval_metrics['images'].append(
            wandb.Image(
                overlay_img,
                caption=f"IoU: {iou:.3f}, Time: {inference_time:.3f}s"
            )
        )

    # Log summary metrics
    wandb.log({
        'mean_iou': np.mean(eval_metrics['image_iou']),
        'std_iou': np.std(eval_metrics['image_iou']),
        'mean_inference_time': np.mean(eval_metrics['inference_times']),
        'evaluation_samples': eval_metrics['images']
    })

    # Create IoU distribution plot
    plt.figure(figsize=(10, 5))
    plt.hist(eval_metrics['image_iou'], bins=20)
    plt.title('Distribution of IoU Scores')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    wandb.log({'iou_distribution': wandb.Image(plt)})
    plt.close()

    # Save metrics to JSON
    metrics_path = output_base / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_path': model_path,  # Include model identifier in metrics
            'mean_iou': float(np.mean(eval_metrics['image_iou'])),
            'std_iou': float(np.std(eval_metrics['image_iou'])),
            'mean_inference_time': float(np.mean(eval_metrics['inference_times'])),
            'individual_results': [{
                'image_name': img_path.stem,
                'iou': float(iou),
                'inference_time': float(time)
            } for img_path, iou, time in zip(
                Path("/inputs").glob("*.jpg"),
                eval_metrics['image_iou'], 
                eval_metrics['inference_times']
            )]
        }, f, indent=2)

    wandb.finish()

@app.local_entrypoint()
def main(model_path: str = "train_1"):
    evaluate_model.remote(model_path=model_path)