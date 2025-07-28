from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO


def get_available_gpus() -> list[int]:
    """Return a list of GPU indices visible to the current process.

    When launched under Slurm, CUDA_VISIBLE_DEVICES is set to the GPUs
    allocated for the job (e.g. "0,1,3,4"). Ultralytics expects a list of
    integer indices. If the env-var is missing we fall back to
    torch.cuda.device_count().
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [int(x) for x in cvd.split(",") if x.strip()]
    return list(range(torch.cuda.device_count()))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    gpus = get_available_gpus()
    if not gpus:
        raise RuntimeError("No CUDA devices available – cannot train model.")
    print(f"Using GPUs: {gpus}")

    # -------------------------
    # Weights & Biases logging
    # -------------------------
    # Ultralytics automatically logs to wandb if the package is installed and
    # WANDB_MODE is not set to "disabled". Setting WANDB_PROJECT (and optional
    # WANDB_NAME/ENTITY) here ensures the run is grouped correctly.
    os.environ.setdefault("WANDB_PROJECT", "yolo_bdd")
    os.environ.setdefault("WANDB_NAME", "yolo_bdd_train")

    # Initialize model (downloads weights if necessary)
    model = YOLO("yolov9e.pt")

    # Start training
    model.train(
        data="bdd_ultra.yaml",
        epochs=10,
        imgsz=1280,
        batch=32,
        device=gpus,
        project="runs",          # local directory for artifacts
        name="yolo_bdd",         # run name inside WandB & runs/
        deterministic=True,       # reproducibility
        workers=8,                # dataloader workers
    )


if __name__ == "__main__":
    main() 