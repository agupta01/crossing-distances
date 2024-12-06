import modal

app = modal.App("crossing-distance-inference")
input_volume = modal.Volume.from_name("inputs", create_if_missing=True)
output_volume = modal.Volume.from_name("outputs", create_if_missing=True)


@app.function(
    volumes={"/inputs": input_volume, "/outputs": output_volume},
    mounts=[modal.Mount.from_local_dir("./", remote_path="/src")],
)
def pipeline():
    from datetime import datetime

    import papermill as pm
    from crosswalk_image_ingest import remote as crosswalk_image_ingest_main
    from dotenv import dotenv_values

    config = dotenv_values("/src/.env")

    execution_start_time = datetime.now()
    # Get raw data and prep it (orange + blue)
    pm.execute_notebook(
        input_path="/src/osm_ingest.ipynb",
        output_path=f'/outputs/osm_ingest-{execution_start_time.strftime("%Y-%m-%d %H:%M:%S")}.ipynb',
    )

    # Get images based on coordinates file (purple)
    fc = crosswalk_image_ingest_main.spawn(
        city_code=config["CITY_CODE"], sample=int(config["NUM_CROSSWALKS"])
    )
    modal.functions.gather(fc)

    # Produce mask shapefiles from images (red + yellow)
    pm.execute_notebook(
        input_path="/src/sam_inference.ipynb",
        output_path=f"/outputs/sam_inference-{execution_start_time.strftime("%Y-%m-%d %H:%M:%S")}.ipynb",
    )

    # Compute corrected crosswalks using masks
    pm.execute_notebook(
        input_path="/src/grow_cut.ipynb",
        output_path=f"/outputs/grow_cut-{execution_start_time.strftime("%Y-%m-%d %H:%M:%S")}.ipynb",
    )


@app.local_entrypoint()
def main(mode: str):
    if mode == "remote":
        pipeline.remote()
    else:
        pipeline.local()
