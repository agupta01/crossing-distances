import os
from pathlib import Path

import modal

app = modal.App("crossing-distances-sam2-fine-tuning")
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "python3-opencv", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python==4.10.0.84",
        "pycocotools~=2.0.8",
        "matplotlib~=3.9.2",
        "wandb",
        force_build = True
    )
    .run_commands(f"git clone https://git@github.com/DSC-Qian/sam2.git")
    .run_commands("pip install -e sam2/.")
    .run_commands("pip install -e 'sam2/.[dev]'")
    .run_commands("cd 'sam2/checkpoints'; ./download_ckpts.sh")
)

weights_volume = modal.Volume.from_name("sam2-weights", create_if_missing=True, environment_name="sam_test")


@app.function(
    mounts=[
        modal.Mount.from_local_dir(
            "../data/configs", remote_path="../sam2/sam2/configs"
        ),
        modal.Mount.from_local_dir("../data/images", remote_path="../sam2/images"),
    ],
    volumes={"/weights": weights_volume},
    image=train_image,
    gpu="H100",
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=36000,
)
def run_training(config_path: str):
    import os
    import shutil
    import subprocess
    from pathlib import Path

    # Set up W&B environment
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = "pedestrian-crossing-distance"
    os.environ["WANDB_ENTITY"] = "tonyqian99-new-york-university"

    # Run the training command
    command = [
        "python",
        "../sam2/training/train.py",
        "-c",
        config_path,
        "--use-cluster",
        "0",
        "--num-gpus",
        "1",
    ]
    subprocess.run(command, check=True)

    weights_dir = Path(
        "../sam2/sam2_logs/modal_results"
    )  # Replace with the actual weights directory
    volume_base_dir = Path("/weights")

    if weights_dir.exists():
        existing_dirs = [
            d.name
            for d in volume_base_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("train_")
            and d.name.split("_")[1].isdigit()
        ]
        existing_nums = [int(d.split("_")[1]) for d in existing_dirs]
        next_num = max(existing_nums, default=0) + 1

        new_dir = volume_base_dir / f"train_{next_num}"
        new_dir.mkdir()

        # Copy the weights to the new folder
        for weight_file in weights_dir.iterdir():
            target_path = new_dir / weight_file.name
            if weight_file.is_file():
                shutil.copy(weight_file, target_path)
                print(f"Copied {weight_file} to {target_path}")
            elif weight_file.is_dir():
                shutil.copytree(weight_file, target_path)
                print(f"Copied directory {weight_file} to {target_path}")

        print(f"Saved training outputs to: {new_dir}")
    else:
        print(f"Weights directory {weights_dir} does not exist.")


@app.local_entrypoint()
def main():
    config_path = "configs/train.yaml"
    run_training.remote(config_path)
