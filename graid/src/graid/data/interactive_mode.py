"""
Interactive Mode for HuggingFace Dataset Generation

This module provides interactive command-line interfaces for:
- Single model selection
- WBF multi-model selection with validation
- Configuration parameter setting
"""

import logging
from typing import Any, Optional

import typer

from graid.data.config_support import (
    ConfigurationError,
    DatasetGenerationConfig,
    ModelConfig,
    WBFConfig,
)
from graid.data.generate_dataset import validate_model_config
from graid.data.generate_db import list_available_models
from graid.utilities.coco import coco_labels
from graid.utilities.common import get_default_device

logger = logging.getLogger(__name__)


def get_dataset_choice() -> str:
    """Interactive dataset selection."""
    typer.secho("📊 Step 1: Choose a dataset", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    datasets = {
        "1": ("bdd", "BDD100K - Berkeley DeepDrive autonomous driving dataset"),
        "2": ("nuimage", "NuImages - Large-scale autonomous driving dataset"),
        "3": ("waymo", "Waymo Open Dataset - Self-driving car dataset"),
    }

    for key, (name, desc) in datasets.items():
        typer.echo(
            f"  {key}. {typer.style(name.upper(), fg=typer.colors.GREEN)} - {desc}"
        )

    typer.echo()
    while True:
        choice = typer.prompt("Select dataset (1-3)")
        if choice in datasets:
            dataset_name = datasets[choice][0]
            typer.secho(f"✓ Selected: {dataset_name.upper()}", fg=typer.colors.GREEN)
            typer.echo()
            return dataset_name
        typer.secho("Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED)


def get_split_choice() -> str:
    """Interactive split selection."""
    typer.secho("🔄 Step 2: Choose data split", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    splits = {
        "1": ("train", "Training set - typically largest portion of data"),
        "2": ("val", "Validation set - used for model evaluation"),
        "3": ("test", "Test set - used for final evaluation"),
    }

    for key, (name, desc) in splits.items():
        typer.echo(
            f"  {key}. {typer.style(name.upper(), fg=typer.colors.GREEN)} - {desc}"
        )

    typer.echo()
    while True:
        choice = typer.prompt("Select split (1-3)")
        if choice in splits:
            split_name = splits[choice][0]
            typer.secho(f"✓ Selected: {split_name.upper()}", fg=typer.colors.GREEN)
            typer.echo()
            return split_name
        typer.secho("Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED)


def get_model_backend_choice() -> str:
    """Interactive model backend selection."""
    available_models = list_available_models()
    backends = list(available_models.keys())

    typer.echo("Available backends:")
    for i, backend in enumerate(backends, 1):
        typer.echo(f"  {i}. {typer.style(backend.upper(), fg=typer.colors.GREEN)}")

    typer.echo()
    while True:
        try:
            backend_choice = int(typer.prompt("Select backend (number)")) - 1
            if 0 <= backend_choice < len(backends):
                backend = backends[backend_choice]
                typer.secho(
                    f"✓ Selected backend: {backend.upper()}", fg=typer.colors.GREEN
                )
                return backend
        except ValueError:
            pass
        typer.secho("Invalid choice. Please enter a valid number.", fg=typer.colors.RED)


def get_model_name_choice(backend: str) -> str:
    """Interactive model name selection for a given backend."""
    available_models = list_available_models()
    models = available_models[backend]

    typer.echo(f"Available {backend.upper()} models:")
    for i, model in enumerate(models, 1):
        typer.echo(f"  {i}. {typer.style(model, fg=typer.colors.GREEN)}")

    typer.echo()
    while True:
        try:
            model_choice = int(typer.prompt("Select model (number)")) - 1
            if 0 <= model_choice < len(models):
                model_name = models[model_choice]
                typer.secho(f"✓ Selected model: {model_name}", fg=typer.colors.GREEN)
                return model_name
        except ValueError:
            pass
        typer.secho("Invalid choice. Please enter a valid number.", fg=typer.colors.RED)


def get_custom_config_choice(backend: str) -> Optional[dict[str, Any]]:
    """Interactive custom configuration setup."""
    typer.echo()
    use_custom = typer.confirm(
        "Do you want to use custom configuration?", default=False
    )

    if not use_custom:
        return None

    typer.echo()
    typer.secho("🛠️ Custom Model Configuration", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    custom_config = {}

    if backend == "detectron":
        typer.echo("Detectron2 Configuration:")
        config_file = typer.prompt(
            "Config file path (e.g., 'COCO-Detection/retinanet_R_50_FPN_3x.yaml')"
        )
        weights_file = typer.prompt("Weights file path (e.g., 'path/to/model.pth')")
        custom_config = {"config": config_file, "weights": weights_file}

    elif backend == "mmdetection":
        typer.echo("MMDetection Configuration:")
        config_file = typer.prompt(
            "Config file path (e.g., 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')"
        )
        checkpoint = typer.prompt("Checkpoint file path or URL")
        custom_config = {"config": config_file, "checkpoint": checkpoint}

    elif backend == "ultralytics":
        typer.echo("Ultralytics Configuration:")
        model_file = typer.prompt("Model file path (e.g., 'yolov8x.pt')")
        custom_config = {"model_file": model_file}

    return custom_config


def get_confidence_threshold() -> float:
    """Interactive confidence threshold selection."""
    typer.echo("🎯 Confidence threshold filters out low-confidence detections.")
    typer.echo("• Lower values (0.1-0.3): More detections, some false positives")
    typer.echo("• Higher values (0.5-0.8): Fewer detections, higher precision")
    typer.echo()

    while True:
        try:
            conf = float(typer.prompt("Enter confidence threshold", default="0.2"))
            if 0.0 <= conf <= 1.0:
                typer.secho(f"✓ Confidence threshold: {conf}", fg=typer.colors.GREEN)
                return conf
            typer.secho(
                "Please enter a value between 0.0 and 1.0.", fg=typer.colors.RED
            )
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)


def validate_model_interactive(model_config: ModelConfig) -> bool:
    """Validate a model configuration interactively."""
    typer.echo(
        f"🔍 Validating model: {model_config.backend} - {model_config.model_name}"
    )

    try:
        # Import the enhanced validation function
        from graid.data.generate_dataset import validate_model_config

        # Validate the model
        is_valid, error_msg = validate_model_config(
            backend=model_config.backend,
            model_name=model_config.model_name,
            config=model_config.custom_config,
            device=model_config.device,
        )

        if is_valid:
            typer.secho("✅ Model validation successful!", fg=typer.colors.GREEN)
            return True
        else:
            typer.secho(f"❌ Model validation failed: {error_msg}", fg=typer.colors.RED)

            # Ask user what to do
            typer.echo()
            typer.echo("Options:")
            typer.echo("  1. Continue anyway (not recommended)")
            typer.echo("  2. Choose a different model")
            typer.echo("  3. Cancel generation")

            while True:
                choice = typer.prompt("Select option (1-3)")
                if choice == "1":
                    typer.secho(
                        "⚠️ Continuing with potentially invalid model",
                        fg=typer.colors.YELLOW,
                    )
                    return True
                elif choice == "2":
                    return False
                elif choice == "3":
                    typer.secho("Generation cancelled", fg=typer.colors.RED)
                    raise typer.Exit(1)
                else:
                    typer.secho(
                        "Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED
                    )

    except Exception as e:
        typer.secho(f"❌ Validation error: {str(e)}", fg=typer.colors.RED)

        # Ask user what to do
        typer.echo()
        typer.echo("Options:")
        typer.echo("  1. Continue anyway (not recommended)")
        typer.echo("  2. Choose a different model")
        typer.echo("  3. Cancel generation")

        while True:
            choice = typer.prompt("Select option (1-3)")
            if choice == "1":
                typer.secho(
                    "⚠️ Continuing with potentially invalid model",
                    fg=typer.colors.YELLOW,
                )
                return True
            elif choice == "2":
                return False
            elif choice == "3":
                typer.secho("Generation cancelled", fg=typer.colors.RED)
                raise typer.Exit(1)
            else:
                typer.secho(
                    "Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED
                )


def get_single_model_config() -> ModelConfig:
    """Interactive single model configuration."""
    typer.secho("🧠 Single Model Configuration", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    backend = get_model_backend_choice()
    model_name = get_model_name_choice(backend)
    custom_config = get_custom_config_choice(backend)

    typer.echo()
    confidence_threshold = get_confidence_threshold()

    # Create model config
    model_config = ModelConfig(
        backend=backend,
        model_name=model_name,
        custom_config=custom_config,
        confidence_threshold=confidence_threshold,
        device=None,  # Will use default device
    )

    # Validate the model
    typer.echo()
    if not validate_model_interactive(model_config):
        typer.secho(
            "Model validation failed. Please check your configuration.",
            fg=typer.colors.RED,
        )
        if not typer.confirm("Do you want to continue anyway?", default=False):
            raise typer.Abort()

    return model_config


def get_wbf_models_config() -> list[ModelConfig]:
    """Interactive WBF multi-model configuration."""
    typer.secho("🔥 WBF Multi-Model Configuration", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    typer.echo("WBF (Weighted Boxes Fusion) combines predictions from multiple models.")
    typer.echo("You need at least 2 models for WBF to work.")
    typer.echo()

    models = []

    while True:
        typer.secho(
            f"🔧 Adding Model #{len(models) + 1}", fg=typer.colors.CYAN, bold=True
        )
        typer.echo()

        backend = get_model_backend_choice()
        model_name = get_model_name_choice(backend)
        custom_config = get_custom_config_choice(backend)

        typer.echo()
        confidence_threshold = get_confidence_threshold()

        # Create model config
        model_config = ModelConfig(
            backend=backend,
            model_name=model_name,
            custom_config=custom_config,
            confidence_threshold=confidence_threshold,
            device=None,  # Will use default device
        )

        # Validate the model
        typer.echo()
        if validate_model_interactive(model_config):
            models.append(model_config)
            typer.secho(
                f"✅ Model {len(models)} added successfully!", fg=typer.colors.GREEN
            )
        else:
            typer.secho("❌ Model validation failed.", fg=typer.colors.RED)
            if typer.confirm("Do you want to add this model anyway?", default=False):
                models.append(model_config)
                typer.secho(
                    f"⚠️ Model {len(models)} added with warnings.",
                    fg=typer.colors.YELLOW,
                )

        typer.echo()

        # Check if we have enough models
        if len(models) >= 2:
            if not typer.confirm("Do you want to add another model?", default=False):
                break
        else:
            typer.echo("You need at least 2 models for WBF. Adding another model...")
            typer.echo()

    return models


def get_wbf_config(num_models: int) -> WBFConfig:
    """Interactive WBF configuration."""
    typer.secho("⚖️ WBF Configuration", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    # IoU threshold
    typer.echo("IoU threshold for box matching (0.0-1.0):")
    typer.echo("• Lower values: More aggressive fusion")
    typer.echo("• Higher values: More conservative fusion")
    while True:
        try:
            iou_threshold = float(typer.prompt("IoU threshold", default="0.55"))
            if 0.0 <= iou_threshold <= 1.0:
                break
            typer.secho(
                "Please enter a value between 0.0 and 1.0.", fg=typer.colors.RED
            )
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)

    # Skip box threshold
    typer.echo()
    typer.echo("Skip box threshold (0.0-1.0):")
    typer.echo("• Boxes with scores below this threshold will be ignored")
    while True:
        try:
            skip_threshold = float(typer.prompt("Skip box threshold", default="0.0"))
            if 0.0 <= skip_threshold <= 1.0:
                break
            typer.secho(
                "Please enter a value between 0.0 and 1.0.", fg=typer.colors.RED
            )
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)

    # Model weights
    typer.echo()
    use_weights = typer.confirm(
        f"Do you want to specify custom weights for the {num_models} models?",
        default=False,
    )

    model_weights = None
    if use_weights:
        typer.echo("Enter weights for each model (positive numbers):")
        model_weights = []
        for i in range(num_models):
            while True:
                try:
                    weight = float(
                        typer.prompt(f"Weight for model {i+1}", default="1.0")
                    )
                    if weight > 0:
                        model_weights.append(weight)
                        break
                    typer.secho("Weight must be positive.", fg=typer.colors.RED)
                except ValueError:
                    typer.secho("Please enter a valid number.", fg=typer.colors.RED)

    return WBFConfig(
        iou_threshold=iou_threshold,
        skip_box_threshold=skip_threshold,
        model_weights=model_weights,
    )


def get_generation_settings() -> dict[str, Any]:
    """Interactive generation settings."""
    typer.secho("⚙️ Generation Settings", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    # Batch size
    while True:
        try:
            batch_size = int(typer.prompt("Batch size", default="1"))
            if batch_size > 0:
                break
            typer.secho("Batch size must be positive.", fg=typer.colors.RED)
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)

    # Save path
    save_path = typer.prompt("Save path (optional, press Enter to skip)", default="")
    if not save_path:
        save_path = None

    # HuggingFace Hub settings
    typer.echo()
    upload_to_hub = typer.confirm("Upload to HuggingFace Hub?", default=False)

    hub_repo_id = None
    hub_private = False
    if upload_to_hub:
        hub_repo_id = typer.prompt("Hub repository ID (e.g., 'username/dataset-name')")
        hub_private = typer.confirm("Make repository private?", default=False)

    return {
        "batch_size": batch_size,
        "save_path": save_path,
        "upload_to_hub": upload_to_hub,
        "hub_repo_id": hub_repo_id,
        "hub_private": hub_private,
    }


def get_allowable_set_choice() -> Optional[list[str]]:
    """Interactive allowable set selection."""
    typer.secho(
        "🎯 Step: Configure object filtering (optional)",
        fg=typer.colors.BLUE,
        bold=True,
    )
    typer.echo()

    typer.echo(
        "Allowable set filters detections to only include specified COCO objects."
    )
    typer.echo(
        "This is useful when you know your images are biased toward certain object types."
    )
    typer.echo("If you leave this empty, all detected objects will be included.")
    typer.echo()

    # Get valid COCO objects
    valid_coco_objects = set(coco_labels.values())
    # Remove undefined as it's not a real COCO class
    valid_coco_objects.discard("undefined")
    valid_objects_sorted = sorted(valid_coco_objects)

    use_allowable_set = typer.confirm(
        "Do you want to filter detections to specific object types?", default=False
    )

    if not use_allowable_set:
        typer.secho(
            "✓ No filtering - all detected objects will be included",
            fg=typer.colors.GREEN,
        )
        return None

    typer.echo()
    typer.echo("📝 Choose how to specify the allowable objects:")
    typer.echo("  1. Select from common autonomous driving objects")
    typer.echo("  2. Select from all COCO objects")
    typer.echo("  3. Enter objects manually")
    typer.echo()

    while True:
        choice = typer.prompt("Select option (1-3)")

        if choice == "1":
            return get_common_av_objects()
        elif choice == "2":
            return get_all_coco_objects_interactive(valid_objects_sorted)
        elif choice == "3":
            return get_manual_objects_input(valid_objects_sorted)
        else:
            typer.secho("Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED)


def get_common_av_objects() -> list[str]:
    """Get common autonomous vehicle objects."""
    typer.echo()
    typer.secho("🚗 Common Autonomous Vehicle Objects", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    av_objects = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "train",
        "truck",
        "traffic light",
        "stop sign",
        "fire hydrant",
        "parking meter",
        "bench",
    ]

    typer.echo("Available objects:")
    for i, obj in enumerate(av_objects, 1):
        typer.echo(f"  {i:2d}. {obj}")

    typer.echo()
    typer.echo(
        "Enter the numbers of objects to include (comma-separated, e.g., '1,2,3-5,7'):"
    )
    typer.echo("Use ranges like '3-5' for objects 3, 4, and 5")
    typer.echo("Or enter 'all' to include all objects")

    while True:
        selection = typer.prompt("Selection").strip()

        if selection.lower() == "all":
            selected_objects = av_objects.copy()
            break

        try:
            selected_objects = []
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    for i in range(start, end + 1):
                        if 1 <= i <= len(av_objects):
                            selected_objects.append(av_objects[i - 1])
                else:
                    i = int(part)
                    if 1 <= i <= len(av_objects):
                        selected_objects.append(av_objects[i - 1])

            # Remove duplicates while preserving order
            selected_objects = list(dict.fromkeys(selected_objects))

            if not selected_objects:
                typer.secho(
                    "No valid objects selected. Please try again.", fg=typer.colors.RED
                )
                continue

            break

        except ValueError:
            typer.secho(
                "Invalid input format. Please use numbers, ranges, or 'all'.",
                fg=typer.colors.RED,
            )

    typer.echo()
    typer.secho(f"✓ Selected {len(selected_objects)} objects:", fg=typer.colors.GREEN)
    for obj in selected_objects:
        typer.echo(f"  • {obj}")

    return selected_objects


def get_all_coco_objects_interactive(valid_objects: list[str]) -> list[str]:
    """Interactive selection from all COCO objects."""
    typer.echo()
    typer.secho("📋 All COCO Objects", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    typer.echo(f"Total COCO objects: {len(valid_objects)}")
    typer.echo("This is a long list - consider using manual entry instead.")
    typer.echo()

    # Show objects in groups of 10
    for i in range(0, len(valid_objects), 10):
        group = valid_objects[i : i + 10]
        typer.echo(f"Objects {i+1}-{min(i+10, len(valid_objects))}:")
        for j, obj in enumerate(group, i + 1):
            typer.echo(f"  {j:2d}. {obj}")
        typer.echo()

    typer.echo(
        "Enter the numbers of objects to include (comma-separated, e.g., '1,2,3-5,7'):"
    )
    typer.echo("Use ranges like '3-5' for objects 3, 4, and 5")

    while True:
        selection = typer.prompt("Selection").strip()

        try:
            selected_objects = []
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    for i in range(start, end + 1):
                        if 1 <= i <= len(valid_objects):
                            selected_objects.append(valid_objects[i - 1])
                else:
                    i = int(part)
                    if 1 <= i <= len(valid_objects):
                        selected_objects.append(valid_objects[i - 1])

            # Remove duplicates while preserving order
            selected_objects = list(dict.fromkeys(selected_objects))

            if not selected_objects:
                typer.secho(
                    "No valid objects selected. Please try again.", fg=typer.colors.RED
                )
                continue

            break

        except ValueError:
            typer.secho(
                "Invalid input format. Please use numbers and ranges.",
                fg=typer.colors.RED,
            )

    typer.echo()
    typer.secho(f"✓ Selected {len(selected_objects)} objects:", fg=typer.colors.GREEN)
    for obj in selected_objects:
        typer.echo(f"  • {obj}")

    return selected_objects


def get_manual_objects_input(valid_objects: list[str]) -> list[str]:
    """Manual object input with validation."""
    typer.echo()
    typer.secho("✏️ Manual Object Entry", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    typer.echo("Enter object names separated by commas (e.g., 'person, car, truck'):")
    typer.echo(
        "Valid COCO object names include: person, car, truck, bus, bicycle, etc."
    )
    typer.echo()

    while True:
        input_str = typer.prompt("Objects").strip()

        if not input_str:
            typer.secho("Please enter at least one object.", fg=typer.colors.RED)
            continue

        # Parse and validate objects
        objects = [obj.strip() for obj in input_str.split(",")]
        objects = [obj for obj in objects if obj]  # Remove empty strings

        valid_objects_set = set(valid_objects)
        invalid_objects = [obj for obj in objects if obj not in valid_objects_set]

        if invalid_objects:
            typer.secho(f"Invalid objects: {invalid_objects}", fg=typer.colors.RED)
            typer.echo("Please check spelling and use valid COCO object names.")
            typer.echo(
                "Common objects: person, car, truck, bus, bicycle, motorcycle, airplane, boat, train"
            )
            continue

        # Remove duplicates while preserving order
        objects = list(dict.fromkeys(objects))

        typer.echo()
        typer.secho(f"✓ Selected {len(objects)} objects:", fg=typer.colors.GREEN)
        for obj in objects:
            typer.echo(f"  • {obj}")

        return objects


def create_interactive_config() -> DatasetGenerationConfig:
    """Create a complete dataset generation configuration interactively."""
    typer.secho(
        "🚀 GRAID HuggingFace Dataset Generation", fg=typer.colors.CYAN, bold=True
    )
    typer.echo()
    typer.echo(
        "This interactive wizard will help you create a dataset generation configuration."
    )
    typer.echo()

    # Step 1: Dataset and split
    dataset_name = get_dataset_choice()
    split = get_split_choice()

    # Step 2: Model selection mode
    typer.secho(
        "🧠 Step 3: Choose model configuration", fg=typer.colors.BLUE, bold=True
    )
    typer.echo()

    mode_options = {
        "1": ("single", "Single Model - Use one model for predictions"),
        "2": (
            "wbf",
            "WBF Multi-Model - Use multiple models with Weighted Boxes Fusion",
        ),
        "3": ("ground_truth", "Ground Truth - Use original dataset annotations"),
    }

    for key, (mode, desc) in mode_options.items():
        typer.echo(
            f"  {key}. {typer.style(mode.upper(), fg=typer.colors.GREEN)} - {desc}"
        )

    typer.echo()
    while True:
        choice = typer.prompt("Select mode (1-3)")
        if choice in mode_options:
            mode = mode_options[choice][0]
            typer.secho(f"✓ Selected: {mode.upper()}", fg=typer.colors.GREEN)
            break
        typer.secho("Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED)

    # Step 3: Model configuration
    typer.echo()
    models = []
    use_wbf = False
    wbf_config = None

    if mode == "single":
        models = [get_single_model_config()]
    elif mode == "wbf":
        models = get_wbf_models_config()
        use_wbf = True
        typer.echo()
        wbf_config = get_wbf_config(len(models))
    # For ground_truth, models remains empty

    # Step 4: Generation settings
    typer.echo()
    settings = get_generation_settings()

    # Step 5: Allowable set selection
    allowable_set = get_allowable_set_choice()

    # Create configuration
    try:
        config = DatasetGenerationConfig(
            dataset_name=dataset_name,
            split=split,
            models=models,
            use_wbf=use_wbf,
            wbf_config=wbf_config,
            confidence_threshold=0.2,  # Default, can be overridden per model
            batch_size=settings["batch_size"],
            device=None,  # Will use default device
            save_path=settings["save_path"],
            upload_to_hub=settings["upload_to_hub"],
            hub_repo_id=settings["hub_repo_id"],
            hub_private=settings["hub_private"],
            allowable_set=allowable_set,
        )

        typer.echo()
        typer.secho(
            "✅ Configuration created successfully!", fg=typer.colors.GREEN, bold=True
        )
        return config

    except ConfigurationError as e:
        typer.secho(f"❌ Configuration error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"❌ Unexpected error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
