import json
import os
from logging import Logger

import modal

from inference.utils import (
    PRECISION,
    RADIUS,
    app,
    committer_image,
    coords_from_distance,
    create_logger,
    dataset_volume,
    filter_coordinates,
    get_crosswalk_id,
    get_from_volume,
    redrive_dict,
    sam_image,
)


@app.function(
    image=committer_image,
    volumes={"/data": dataset_volume},
    timeout=3600,
    concurrency_limit=10,  # need to leave container quota open for the get_image_for_crosswalk function
)
def get_crosswalks_batch(
    lat_batch: tuple[float, ...], long_batch: tuple[float, ...], logger: Logger
):
    try:
        logger.info(f"Batch contains {len(lat_batch)} coordinates.")
        ids_and_raw_images = list(
            get_image_for_crosswalk.map(
                lat_batch, long_batch, return_exceptions=True, order_outputs=True
            )
        )
        logger.info("Images pulled. Starting save task...")
        saved_files = commit_images_to_volume(ids_and_raw_images, logger)
        logger.info("Saved images to volume.")
        create_decoder(saved_files, lat_batch, long_batch)
        logger.info("Created decoder from this batch.")
        return saved_files
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error("Logging unsaved coordinates to redrive.")
        for i, (lat, long) in enumerate(zip(lat_batch, long_batch)):
            redrive_dict[
                f"{modal.current_input_id()}_{modal.current_function_call_id()}_{i}"
            ] = (lat, long, e)


def create_decoder(
    saved_files: list[str], lat_batch: tuple[float, ...], long_batch: tuple[float, ...]
):
    """Saves a decoder file to map coordinates to volume filenames. Assumes the saved_files are in order."""
    decoder_filename = f"/data/decoder_{modal.current_function_call_id()}.json"
    decoder = {
        f"{lat},{long}": saved_files[i]
        for i, (lat, long) in enumerate(zip(lat_batch, long_batch))
    }
    with open(decoder_filename, "x") as f:
        json.dump(decoder, f)

    dataset_volume.commit()


@app.function(image=committer_image, volumes={"/data": dataset_volume}, timeout=86400)
def task_dispatcher(
    lats: list[float], longs: list[float], logger: Logger, batch_size: int = 1000
):
    """DEPRECATED: Use get_crosswalks_batch instead."""
    from itertools import batched

    logger.info(f"Received {len(lats)} coordinates.")

    lats_batcher = batched(lats, batch_size)
    longs_batcher = batched(longs, batch_size)

    save_calls: list[modal.functions.FunctionCall] = []
    for i, (lat_batch, long_batch) in enumerate(zip(lats_batcher, longs_batcher)):
        get_crosswalks_batch.local(lat_batch, long_batch, logger=logger)

    # wait for all save calls to finish
    modal.functions.gather(*save_calls)


def commit_images_to_volume(ids_and_raw_images, logger: Logger) -> list[str]:
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
    return saved_files


@app.function(image=sam_image)
def get_image_for_crosswalk(lat: float, long: float):
    import math

    from samgeo import tms_to_geotiff

    logger = create_logger()
    logger.debug(f"Executing function for coordinate ({lat}, {long})")

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
def main(city_code: str | None = None, sample: int = 1000, batch_size: int = 1000):
    """Same thing as local_main() just gets data from modal volume"""
    import math
    from itertools import batched

    import pandas as pd

    logger = create_logger()

    if not city_code:
        if os.getenv("MODAL_ENVIRONMENT"):
            city_code = os.environ["MODAL_ENVIRONMENT"]
        else:
            raise ValueError(
                "City code not provided. Attempted to get from MODAL_ENVIRONMENT environment variable but none was found."
            )

    df = pd.read_csv(
        get_from_volume(
            "scratch", "intersection_coordinates.csv", environment_name=city_code
        )
    )
    coords = filter_coordinates(df, city_code, sample, logger)
    lats, longs = zip(*coords)

    lats_batcher = batched(lats, batch_size)
    longs_batcher = batched(longs, batch_size)
    num_batches = math.ceil(len(lats) / batch_size)
    logger.info(
        f"[MAP BEGIN] Split into {num_batches} batches containing {batch_size} coordinates each."
    )

    map_results = list(
        get_crosswalks_batch.map(
            lats_batcher,
            longs_batcher,
            kwargs={"logger": logger},
            return_exceptions=True,
        )
    )

    # Log out the number of batches that succeeded and the number that failed
    logger.info(
        f"[MAP END] Successfully processed {len([r for r in map_results if not isinstance(r, Exception)])} batches."
    )
    if any(isinstance(r, Exception) for r in map_results):
        logger.warning(
            f"[MAP END] Failed to process {len([r for r in map_results if isinstance(r, Exception)])} batches."
        )


@app.local_entrypoint()
def local_main(mode: str, input: str, sample: int = 1000, batch_size: int = 1000):
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
