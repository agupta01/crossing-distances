import argparse
import os
import tempfile
import subprocess
from contextlib import contextmanager

import modal
from modal.functions import FunctionCall
from tqdm.auto import tqdm

from inference.utils import (
    create_logger,
    app,
    filter_coordinates,
    fuzzy_search_optimized,
    get_from_volume,
    main_scratch,
)
from inference.utils import osmnx_image
from inference.utils import APP_NAME


def read_envs_file(envs_path: str):
    import pandas as pd

    return pd.read_csv(envs_path, delimiter="\t")


def deploy(envs_path: str, functions_file: str, subset: str | None = None):
    envs = read_envs_file(envs_path)
    if subset:
        envs = envs[envs["Code"].str.lower().isin(subset.split(","))]
    for _, env in envs.iterrows():
        subprocess.run(
            [
                "modal",
                "deploy",
                "--env",
                env["Code"].lower(),
                functions_file,
            ]
        )


@contextmanager
def download_scratch_to_temp_dir(filepath, env, conda_env="ox"):
    """
    Create a temporary directory, download files from the scratch volume into it,
    and ensure the directory is deleted when the context exits.

    This function runs the shell command:
        modal volume get scratch {filepath} {temp_dir}/
    where {temp_dir} is the temporary folder created.

    :param filepath: The path of the file on the scratch volume to download.
    :param env: The environment to download the file from.
    :param conda_env: The conda environment to run the command in. If none, defaults to the current environment.
    :yield: The path to the temporary directory containing the downloaded files.
    """
    # Create a temporary directory; it is automatically cleaned up when closed.
    with tempfile.TemporaryDirectory() as temp_dir:
        if conda_env:
            if not conda_env:
                raise EnvironmentError(
                    "CONDA_DEFAULT_ENV is not set. Make sure you are running inside a conda environment."
                )
            # Construct the command using "conda run" to make sure Modal CLI is available.
            command = f"conda run -n {conda_env} modal volume get -e {env} scratch {filepath} {temp_dir}/"
        else:
            command = f"modal volume get -e {env} scratch {filepath} {temp_dir}/"

        # Execute the command in a shell, and raise an exception if it fails.
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Yield the temporary directory to the caller.
        yield temp_dir


def get_file_from_all_envs(envs_path: str, filename: str, destination_dir: str):
    envs = read_envs_file(envs_path)
    for _, env in tqdm(envs.iterrows(), total=len(envs)):
        env_code = env["Code"].lower()
        subprocess.run(
            f"uv run modal volume get -e {env_code} scratch {filename} {destination_dir}/"
        )


def verify(env_name: str, logger: Logger):
    """Verifies that an environment contains all all requested images."""
    import pandas as pd

    # Check intersection list in scratch volume
    crosswalks_df = pd.read_csv(
        get_from_volume(
            "scratch", "intersection_coordinates.csv", environment_name=env_name
        )
    )
    crosswalks_vol = modal.Volume.lookup(
        f"crosswalk-data-{env_name}", environment_name=env_name
    )
    filenames = list(map(lambda x: x.path, crosswalks_vol.listdir("/")))
    missing_coords: pd.DataFrame = filter_coordinates(
        crosswalks_df, env_name, -1, logger, return_df=True
    )
    if len(missing_coords) != 0:
        missing_coords["best_match"] = missing_coords["filename"].apply(
            lambda x: fuzzy_search_optimized(x, filenames)
        )
        missing_coords["error"] = missing_coords.apply(
            lambda row: sum(
                1 if row["filename"][i] != row["best_match"][i] else 0
                for i in range(len(row["filename"]))
            ),
            axis=1,
        )
        missing_coords.to_csv(
            f"/scratch/missing_intersections_{env_name}.csv", index=False
        )
        logger.error(
            f"[MISSING INTERSECTIONS] {len(missing_coords)} intersections missing in {env_name}. Missing coordinates and closest matches are saved at /scratch/missing_intersections_{env_name}.csv"
        )


def run(envs_path: str, function: str, subset: str | None = None):
    subset = list(map(lambda x: x.lower(), subset.split(",")))
    envs = read_envs_file(envs_path)
    if subset:
        envs = envs[envs["Code"].str.lower().isin(subset)]
    calls = []
    os.environ["MODAL_ENVIRONMENT"] = envs["Code"].iloc[0].lower()
    logger = create_logger()
    logger.info(f"Running function {function} in environments {envs['Code'].tolist()}")
    for i, env in envs.iterrows():
        env_code = env["Code"].lower()
        calls.append(
            run_single_city(
                env_code, function, logger, place=f"{env['Title']}, {env['State']}"
            )
        )

    modal.functions.gather(*calls)

    # Verify that all images were processed
    if function == "main":
        for _, env in envs.iterrows():
            verify(env["Code"].lower(), logger)


def run_single_city(
    env_name: str, function: str, logger: Logger, place: str
) -> FunctionCall:
    logger.info(f"Running function {function} in environment {env_name}")
    os.environ["MODAL_ENVIRONMENT"] = env_name
    f = modal.Function.lookup(APP_NAME, function, environment_name=env_name)
    if function == "main":
        call = f.spawn(city_code=env_name, sample=-1, batch_size=1000)
    elif function == "osm_ingest":
        call = f.spawn(place=place)
    else:
        raise ValueError(f"Function {function} not recognized")
    return call


@app.function(
    image=osmnx_image,
    volumes={"/scratch": main_scratch},
    concurrency_limit=1,
    timeout=86400,
)
def modal_run(env_name: str):
    logger = create_logger()
    call = run_single_city(env_name, "main", logger, place="")
    modal.functions.gather(call)
    verify(env_name, logger)


def main():
    parser = argparse.ArgumentParser(description="CLI utilities for PCD project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy functions in file to all environments"
    )
    deploy_parser.add_argument(
        "--envs-path",
        type=str,
        required=True,
        help='Path to environments file. Should contain a column "Code" with the environment names.',
    )
    deploy_parser.add_argument(
        "--functions-file",
        type=str,
        required=True,
        help="Path to functions Python file.",
    )
    deploy_parser.add_argument(
        "--subset",
        type=str,
        required=False,
        help="Subset of environments to run the function in. Comma separated list of environment names. If not specified, runs in all environments.",
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run a function in all or selected environments"
    )
    run_parser.add_argument(
        "--envs-path",
        type=str,
        required=True,
        help='Path to environments file. Should contain a column "Code" with the environment names.',
    )
    run_parser.add_argument(
        "--function", type=str, required=True, help="Function to run."
    )
    run_parser.add_argument(
        "--subset",
        type=str,
        required=False,
        help="Subset of environments to run the function in. Comma separated list of environment names. If not specified, runs in all environments.",
    )

    # Get file command
    get_parser = subparsers.add_parser("get", help="Get a file from all environments")
    get_parser.add_argument(
        "--envs-path",
        type=str,
        required=True,
        help='Path to environments file. Should contain a column "Code" with the environment names.',
    )
    get_parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Filename to get from all environments.",
    )
    get_parser.add_argument(
        "--destination-dir",
        type=str,
        required=True,
        help="Destination directory to save the file.",
    )

    args = parser.parse_args()

    if args.command == "deploy":
        deploy(args.envs_path, args.functions_file, args.subset)
    elif args.command == "run":
        run(args.envs_path, args.function, args.subset)
    elif args.command == "get":
        get_file_from_all_envs(args.envs_path, args.filename, args.destination_dir)


if __name__ == "__main__":
    main()
