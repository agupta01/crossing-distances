import modal
from dotenv import dotenv_values

from inference.utils import (
    PRECISION,
    RADIUS,
    app,
    coords_from_distance,
    create_logger,
    get_crosswalk_id,
)

config = dotenv_values("./.env")

sam_image = (
    modal.Image.from_registry("gboeing/osmnx:latest")
    .pip_install("segment-geospatial", "timm==1.0.9")
    .env({"TINI_SUBREAPER": "1"})
)

committer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pillow")
    .env({"TINI_SUBREAPER": "1"})
)

dataset_volume = modal.Volume.from_name(
    f"crosswalk-data-{config["CITY_CODE"]}", create_if_missing=True
)

redrive_dict = modal.Dict.from_name(
    f"crosswalk-data-{config["CITY_CODE"]}-redrive", create_if_missing=True
)


@app.function(image=committer_image, timeout=86400)
def task_dispatcher(lats: list[float], longs: list[float], batch_size=1000):
    import math
    from itertools import batched

    logger = create_logger()
    logger.info(f"Received {len(lats)} coordinates.")

    lats_batcher = batched(lats, batch_size)
    longs_batcher = batched(longs, batch_size)
    num_batches = math.ceil(len(lats) / batch_size)

    save_calls: list[modal.functions.FunctionCall] = []
    for i, (lat_batch, long_batch) in enumerate(zip(lats_batcher, longs_batcher)):
        logger.info(f"[BATCH {i}/{num_batches}] contains {len(lat_batch)} coordinates.")
        ids_and_raw_images = list(
            get_image_for_crosswalk.map(lat_batch, long_batch, return_exceptions=True)
        )
        logger.info("Images pulled. Starting save task...")
        save_calls.append(commit_images_to_volume.spawn(ids_and_raw_images))

    # wait for all save calls to finish
    modal.functions.gather(*save_calls)


@app.function(image=committer_image, volumes={"/data": dataset_volume}, timeout=3600)
def commit_images_to_volume(ids_and_raw_images):
    logger = create_logger()

    # convert images to JPEG files with filenames as ids
    saved_files = []
    for j, data in enumerate(ids_and_raw_images):
        if isinstance(data, Exception):
            logger.warning(f"Unhandled Exception found: {data}. Logging to redrive.")
            redrive_dict[
                f"{modal.current_input_id()}_{modal.current_function_call_id()}_{j}"
            ] = data
            continue
        crosswalk_id, image_data = data
        if isinstance(image_data, Exception):
            logger.warning(
                f"Exception found: {image_data} for coordinates {crosswalk_id}. Logging to redrive."
            )
            redrive_dict[crosswalk_id] = image_data
            continue
        filename = f"/data/crosswalk_{crosswalk_id}.jpeg"
        image_data.convert("RGB").save(filename, "JPEG", optimize=True, keep_rgb=True)
        saved_files.append(filename)
        logger.info(f"Saved file {j} of {len(ids_and_raw_images)} in batch.")
    # commit volume
    dataset_volume.commit()
    logger.info(f"Saved {len(saved_files)} to volume.")


@app.function(image=sam_image)
def get_image_for_crosswalk(lat: float, long: float):
    import math

    from samgeo import tms_to_geotiff

    logger = create_logger()
    logger.info(f"Executing function for coordinate ({lat}, {long})")

    radius = RADIUS  # meters

    try:
        crosswalk_id = get_crosswalk_id(lat, long)

        # Build bounding box based on radius
        diag_radius = math.sqrt(2) * radius
        top_left = coords_from_distance(lat, long, diag_radius, 315)
        bottom_right = coords_from_distance(lat, long, diag_radius, 135)
        bounding_box = [
            bottom_right.long,
            bottom_right.lat,
            top_left.long,
            top_left.lat,
        ]

        image_data = tms_to_geotiff(
            output="./scratch.tif",
            bbox=bounding_box,
            crs="EPSG:3857",
            zoom=22,
            source="Satellite",
            overwrite=True,
            quiet=True,
            return_image=True,
        )
    except Exception as e:
        return ((lat, long), e)
    else:
        return crosswalk_id, image_data


@app.function(image=committer_image, timeout=86400)
def remote(city_code: str, sample: int = 1000, batch_size: int = 1000):
    import modal
    import pandas as pd

    # same thing as main() just gets data from modal volume
    outputs_vol = modal.Volume.lookup("outputs", environment_name=city_code)
    # TODO: get coords file from volume
    coords_file = None

    task_dispatcher.call()


@app.local_entrypoint()
def main(mode: str, input: str, sample: int = 1000, batch_size: int = 1000):
    import pandas as pd

    df = pd.read_csv(input)
    coords = (
        df.sample(n=sample if sample > 0 else len(df))
        .apply(lambda c: (round(c.y, PRECISION), round(c.x, PRECISION)), axis=1)
        .tolist()
    )
    # TODO: filter coordinates that have existing images in the volume

    if mode.lower() == "remote":
        task_dispatcher.remote(*zip(*coords), batch_size=batch_size)
    else:
        task_dispatcher.local(*zip(*coords), batch_size=batch_size)
