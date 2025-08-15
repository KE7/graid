"""
GRAID (Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence) - Main CLI Interface

An interactive command-line tool for generating object detection databases 
using various models and datasets.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import typer

from graid.data.config_support import load_config_from_file

# Suppress common warnings for better user experience
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*TorchScript.*functional optimizers.*deprecated.*"
)

# Suppress mmengine info messages
logging.getLogger("mmengine").setLevel(logging.WARNING)


# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def _configure_logging():
    # Simple logic: GRAID_DEBUG_VERBOSE controls console debug, file always gets debug
    debug_verbose = bool(os.getenv("GRAID_DEBUG_VERBOSE"))
    console_level = logging.DEBUG if debug_verbose else logging.INFO
    file_level = logging.DEBUG  # Always debug to file
    root_level = logging.DEBUG   # Root logger must be permissive for debug messages
    
    # Configure root logger once with both console and file handlers
    logger = logging.getLogger()
    if logger.handlers:
        # If already configured, update levels
        logger.setLevel(root_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(console_level)
            elif isinstance(handler, logging.FileHandler):
                handler.setLevel(file_level)
        return
    
    logger.setLevel(root_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler with timestamp
    log_dir = os.getenv("GRAID_LOG_DIR", "logs")
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    # Generate timestamped log filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"graid_{timestamp}.log"
    
    fh = logging.FileHandler(Path(log_dir) / log_filename)
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Quiet noisy libraries a bit
    logging.getLogger("mmengine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


_configure_logging()


app = typer.Typer(
    name="graid",
    help="GRAID: Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence",
    add_completion=False,
)


def print_welcome():
    """Print welcome message and project info."""
    typer.echo()
    typer.secho("🤖 Welcome to GRAID!", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        "   Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence"
    )
    typer.echo()
    typer.echo("GRAID provides three main capabilities:")
    typer.echo()
    typer.secho("📁 Database Generation (generate):",
                fg=typer.colors.BLUE, bold=True)
    typer.echo("• Multiple datasets: BDD100K, NuImages, Waymo")
    typer.echo("• Various model backends: Detectron, MMDetection, Ultralytics")
    typer.echo("• Ground truth data or custom model predictions")
    typer.echo()
    typer.secho(
        "🤗 HuggingFace Dataset Generation (generate-dataset):",
        fg=typer.colors.BLUE,
        bold=True,
    )
    typer.echo("• Generate HuggingFace datasets with VQA pairs")
    typer.echo("• Support for WBF multi-model ensembles")
    typer.echo("• Allowable set filtering for COCO objects")
    typer.echo("• Interactive mode with config file support")
    typer.echo()
    typer.secho("🧠 VLM Evaluation (eval-vlms):",
                fg=typer.colors.BLUE, bold=True)
    typer.echo("• Evaluate Vision Language Models: GPT, Gemini, Llama")
    typer.echo("• Multiple evaluation metrics: LLMJudge, ExactMatch, Contains")
    typer.echo(
        "• Various prompting strategies: ZeroShot, CoT, SetOfMark, Constrained Decoding"
    )
    typer.echo()


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
    typer.secho("💡 Custom Dataset Support:", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        "   GRAID supports any PyTorch-compatible dataset. Only images are required for VQA."
    )
    typer.echo("   Annotations are optional (only needed for mAP/mAR evaluation).")
    typer.echo()

    while True:
        choice = typer.prompt("Select dataset (1-3)")
        if choice in datasets:
            dataset_name = datasets[choice][0]
            typer.secho(
                f"✓ Selected: {dataset_name.upper()}", fg=typer.colors.GREEN)
            typer.echo()
            return dataset_name
        typer.secho("Invalid choice. Please enter 1, 2, or 3.",
                    fg=typer.colors.RED)


def get_split_choice() -> str:
    """Interactive split selection."""
    typer.secho("🔄 Step 2: Choose data split", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    splits = {
        "1": ("train", "Training set - typically largest portion of data"),
        "2": ("val", "Validation set - used for model evaluation"),
    }

    for key, (name, desc) in splits.items():
        typer.echo(
            f"  {key}. {typer.style(name.upper(), fg=typer.colors.GREEN)} - {desc}"
        )

    typer.echo()
    while True:
        choice = typer.prompt("Select split (1-2)")
        if choice in splits:
            split_name = splits[choice][0]
            typer.secho(
                f"✓ Selected: {split_name.upper()}", fg=typer.colors.GREEN)
            typer.echo()
            return split_name
        typer.secho("Invalid choice. Please enter 1 or 2.", fg=typer.colors.RED)


def get_model_choice() -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """Interactive model selection with custom model support."""
    typer.secho("🧠 Step 3: Choose model type", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    typer.echo("  1. Ground Truth - Use original dataset annotations (fastest)")
    typer.echo(
        "  2. Pre-configured Models - Choose from built-in model configurations")
    typer.echo(
        "  3. Custom Model - Bring your own Detectron/MMDetection/Ultralytics model"
    )
    typer.echo()

    while True:
        choice = typer.prompt("Select option (1-3)")

        if choice == "1":
            typer.secho("✓ Selected: Ground Truth", fg=typer.colors.GREEN)
            typer.echo()
            return None, None, None

        elif choice == "2":
            return get_preconfigured_model()

        elif choice == "3":
            return get_custom_model()

        typer.secho("Invalid choice. Please enter 1, 2, or 3.",
                    fg=typer.colors.RED)


def get_preconfigured_model() -> tuple[str, str, None]:
    """Interactive pre-configured model selection."""
    typer.echo()
    typer.secho("🔧 Pre-configured Models", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    # Local import to avoid heavy dependencies
    from graid.data.generate_db import list_available_models
    available_models = list_available_models()

    backends = list(available_models.keys())
    typer.echo("Available backends:")
    for i, backend in enumerate(backends, 1):
        typer.echo(
            f"  {i}. {typer.style(backend.upper(), fg=typer.colors.GREEN)}")

    typer.echo()
    while True:
        try:
            backend_choice = int(typer.prompt("Select backend (number)")) - 1
            if 0 <= backend_choice < len(backends):
                backend = backends[backend_choice]
                break
        except ValueError:
            pass
        typer.secho("Invalid choice. Please enter a valid number.",
                    fg=typer.colors.RED)

    typer.echo()
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
                break
        except ValueError:
            pass
        typer.secho("Invalid choice. Please enter a valid number.",
                    fg=typer.colors.RED)

    typer.secho(
        f"✓ Selected: {backend.upper()} - {model_name}", fg=typer.colors.GREEN)
    typer.echo()
    return backend, model_name, None


def get_custom_model() -> tuple[str, str, dict]:
    """Interactive custom model configuration."""
    typer.echo()
    typer.secho("🛠️ Custom Model Configuration",
                fg=typer.colors.BLUE, bold=True)
    typer.echo()

    typer.echo("Supported backends for custom models:")
    typer.echo("  1. Detectron2 - Facebook's object detection framework")
    typer.echo("  2. MMDetection - OpenMMLab's detection toolbox")
    typer.echo("  3. Ultralytics - YOLO and RT-DETR models")
    typer.echo()

    while True:
        choice = typer.prompt("Select backend (1-3)")
        if choice == "1":
            backend = "detectron"
            break
        elif choice == "2":
            backend = "mmdetection"
            break
        elif choice == "3":
            backend = "ultralytics"
            break
        typer.secho("Invalid choice. Please enter 1, 2, or 3.",
                    fg=typer.colors.RED)

    typer.echo()
    custom_config = {}

    if backend == "detectron":
        typer.echo("Detectron2 Configuration:")
        typer.echo("You need to provide paths to configuration and weights files.")
        typer.echo()

        config_file = typer.prompt(
            "Config file path (e.g., 'COCO-Detection/retinanet_R_50_FPN_3x.yaml')"
        )
        weights_file = typer.prompt(
            "Weights file path (e.g., 'path/to/model.pth')")

        custom_config = {"config": config_file, "weights": weights_file}

    elif backend == "mmdetection":
        typer.echo("MMDetection Configuration:")
        typer.echo(
            "You need to provide paths to configuration and checkpoint files.")
        typer.echo()

        config_file = typer.prompt(
            "Config file path (e.g., 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')"
        )
        checkpoint = typer.prompt("Checkpoint file path or URL")

        custom_config = {"config": config_file, "checkpoint": checkpoint}

    elif backend == "ultralytics":
        typer.echo("Ultralytics Configuration:")
        typer.echo("You need to provide the path to a custom trained model file.")
        typer.echo()

        model_path = typer.prompt(
            "Model file path (e.g., 'path/to/custom_model.pt')")

        custom_config = {"model_path": model_path}

    # Generate a custom model name
    model_name = f"custom_{Path(custom_config.get('config', custom_config.get('model_path', 'model'))).stem}"

    typer.secho(
        f"✓ Custom model configured: {backend.upper()}", fg=typer.colors.GREEN)
    typer.echo()
    return backend, model_name, custom_config


def get_confidence_threshold() -> float:
    """Interactive confidence threshold selection."""
    typer.secho("🎯 Step 4: Set confidence threshold",
                fg=typer.colors.BLUE, bold=True)
    typer.echo()
    typer.echo("Confidence threshold filters out low-confidence detections.")
    typer.echo("• Lower values (0.1-0.3): More detections, some false positives")
    typer.echo("• Higher values (0.5-0.8): Fewer detections, higher precision")
    typer.echo()

    while True:
        try:
            conf = float(typer.prompt(
                "Enter confidence threshold", default="0.2"))
            if 0.0 <= conf <= 1.0:
                typer.secho(
                    f"✓ Confidence threshold: {conf}", fg=typer.colors.GREEN)
                typer.echo()
                return conf
            typer.secho(
                "Please enter a value between 0.0 and 1.0.", fg=typer.colors.RED
            )
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)


@app.command()
def generate(
    dataset: Optional[str] = typer.Option(
        None, help="Dataset name (bdd, nuimage, waymo)"
    ),
    split: Optional[str] = typer.Option(None, help="Data split (train, val)"),
    backend: Optional[str] = typer.Option(None, help="Model backend"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    conf: Optional[float] = typer.Option(None, help="Confidence threshold"),
    config: Optional[str] = typer.Option(None, help="Custom model config file"),
    checkpoint: Optional[str] = typer.Option(
        None, help="Custom model checkpoint/weights"
    ),
    interactive: bool = typer.Option(True, help="Use interactive mode"),
):
    """
    Generate object detection database.

    Run without arguments for interactive mode, or specify all parameters for batch mode.
    """

    if interactive and not all([dataset, split]):
        print_welcome()

        # Interactive mode
        if not dataset:
            dataset = get_dataset_choice()
        if not split:
            split = get_split_choice()

        backend_choice, model_choice, custom_config = get_model_choice()
        if backend_choice:
            backend = backend_choice
            model = model_choice

        if not conf:
            conf = get_confidence_threshold()

    else:
        # Batch mode - validate required parameters
        if not dataset or not split:
            typer.secho(
                "Error: dataset and split are required in non-interactive mode",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        # Local import for dataset validation
        from graid.data.generate_db import DATASET_TRANSFORMS
        if dataset not in DATASET_TRANSFORMS:
            typer.secho(
                f"Error: Invalid dataset '{dataset}'. Choose from: {list(DATASET_TRANSFORMS.keys())}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        if split not in ["train", "val"]:
            typer.secho(
                "Error: Invalid split. Choose 'train' or 'val'", fg=typer.colors.RED
            )
            raise typer.Exit(1)

        if conf is None:
            conf = 0.2

        custom_config = None
        if config and checkpoint:
            if backend == "detectron":
                custom_config = {"config": config, "weights": checkpoint}
            elif backend == "mmdetection":
                custom_config = {"config": config, "checkpoint": checkpoint}

    # Handle custom model configuration
    if "custom_config" in locals() and custom_config:
        # Custom model configuration is handled directly by create_model
        pass

    # Start generation
    typer.secho("🚀 Starting database generation...",
                fg=typer.colors.BLUE, bold=True)
    typer.echo()
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Split: {split}")
    if backend and model:
        typer.echo(f"Model: {backend} - {model}")
    else:
        typer.echo("Model: Ground Truth")
    typer.echo(f"Confidence: {conf}")
    typer.echo()

    try:
        from graid.data.generate_db import generate_db
        db_name = generate_db(
            dataset_name=dataset,
            split=split,
            conf=conf,
            backend=backend,
            model_name=model,
        )

        typer.echo()
        typer.secho(
            "✅ Database generation completed successfully!",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo(f"Database created: {db_name}")

    except Exception as e:
        typer.echo()
        typer.secho(
            f"❌ Error during generation: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)


@app.command("generate-dataset")
def generate_dataset_cmd(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        help="Dataset name (bdd, nuimage, waymo) - supports custom PyTorch datasets",
    ),
    split: Optional[str] = typer.Option(
        None, help="Data split (train, val, test)"),
    allowable_set: Optional[str] = typer.Option(
        None, help="Comma-separated list of allowed COCO objects"
    ),
    save_path: Optional[str] = typer.Option(
        None, help="Path to save the generated dataset"
    ),
    upload_to_hub: bool = typer.Option(
        False, help="Upload dataset to HuggingFace Hub"),
    hub_repo_id: Optional[str] = typer.Option(
        None, help="HuggingFace Hub repository ID"
    ),
    hub_private: bool = typer.Option(
        False, help="Make HuggingFace Hub repository private"
    ),
    interactive: bool = typer.Option(True, help="Use interactive mode"),
    list_valid_objects: bool = typer.Option(
        False, "--list-objects", help="List valid COCO objects and exit"
    ),
    list_questions: bool = typer.Option(
        False, "--list-questions", help="List available questions and exit"
    ),
    interactive_questions: bool = typer.Option(
        False, "--interactive-questions", help="Use interactive question selection"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-j", help="DataLoader workers for parallel image loading"
    ),
    qa_workers: int = typer.Option(
        4, "--qa-workers", help="Parallel threads for QA generation"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force restart from scratch, ignore existing checkpoints"
    ),
):
    """
    Generate HuggingFace datasets for object detection question-answering.

    Supports built-in datasets (BDD100K, NuImages, Waymo) and custom PyTorch datasets
    with COCO-style annotations. Use interactive mode or config files for easy setup.
    """

    # Handle special flags
    if list_valid_objects:
        typer.echo("Valid COCO objects:")
        # Local import to avoid heavy dependencies
        from graid.utilities.coco import coco_labels
        valid_objects = list(coco_labels.values())
        # Remove undefined as it's not a real COCO class
        if "undefined" in valid_objects:
            valid_objects.remove("undefined")
        valid_objects.sort()
        for i, obj in enumerate(valid_objects, 1):
            typer.echo(f"  {i:2d}. {obj}")
        typer.echo(f"\nTotal: {len(valid_objects)} objects")
        return

    if list_questions:
        typer.secho("📋 Available Questions:", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        # Local import to avoid heavy dependencies
        from graid.data.generate_dataset import list_available_questions
        questions = list_available_questions()
        for i, (name, info) in enumerate(questions.items(), 1):
            typer.secho(f"{i:2d}. {name}", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"    {info['question']}")
            if info["parameters"]:
                typer.echo("    Parameters:")
                for param_name, param_info in info["parameters"].items():
                    typer.echo(
                        f"      • {param_name}: {param_info['description']} (default: {param_info['default']})"
                    )
            typer.echo()
        return

    print_welcome()

    try:
        if config_file:
            # Load configuration from file
            typer.secho(
                "📄 Loading configuration from file...", fg=typer.colors.BLUE, bold=True
            )
            config = load_config_from_file(config_file)
            # Override CLI arguments if provided (CLI takes precedence over config file)
            if force:
                config.force = force
            if save_path:
                config.save_path = save_path
            if upload_to_hub:
                config.upload_to_hub = upload_to_hub
            if hub_repo_id:
                config.hub_repo_id = hub_repo_id
            if hub_private:
                config.hub_private = hub_private
            if dataset:
                config.dataset_name = dataset
            if split:
                config.split = split
            if num_workers != 4:  # Only override if not default
                config.num_workers = num_workers
            if qa_workers != 4:  # Only override if not default
                config.qa_workers = qa_workers
            if allowable_set:
                # Parse allowable_set from CLI
                allowable_set_list = [obj.strip() for obj in allowable_set.split(",")]
                # Validate COCO objects
                from graid.utilities.coco import validate_coco_objects
                is_valid, error_msg = validate_coco_objects(allowable_set_list)
                if not is_valid:
                    typer.secho(f"❌ {error_msg}", fg=typer.colors.RED)
                    raise typer.Exit(1)
                config.allowable_set = allowable_set_list
            typer.secho(
                f"✓ Configuration loaded from: {config_file}", fg=typer.colors.GREEN
            )
        elif interactive:
            # Interactive mode
            typer.secho("🎮 Interactive Mode", fg=typer.colors.BLUE, bold=True)
            typer.echo(
                "Let's configure your HuggingFace dataset generation step by step."
            )
            typer.echo()
            # Local import to avoid heavy dependencies
            from graid.data.config_support import DatasetGenerationConfig
            # For now, create a basic config - would need to implement interactive config creation
            typer.secho(
                "❌ Interactive configuration is not yet implemented. Please use --config.",
                fg=typer.colors.RED,
            )
            typer.echo("Use 'graid generate-dataset --help' for more information.")
            raise typer.Exit(1)
        else:
            # Command line parameters mode
            typer.secho("⚙️  Command Line Mode",
                        fg=typer.colors.BLUE, bold=True)

            # Parse allowable_set if provided
            allowable_set_list = None
            if allowable_set:
                allowable_set_list = [obj.strip()
                                      for obj in allowable_set.split(",")]
                # Validate COCO objects
                from graid.utilities.coco import validate_coco_objects

                is_valid, error_msg = validate_coco_objects(allowable_set_list)
                if not is_valid:
                    typer.secho(f"❌ {error_msg}", fg=typer.colors.RED)
                    raise typer.Exit(1)

            # For now, require interactive mode or config file
            typer.secho(
                "❌ Command line mode is not yet implemented. Please use --interactive or --config.",
                fg=typer.colors.RED,
            )
            typer.echo(
                "Use 'graid generate-dataset --help' for more information.")
            raise typer.Exit(1)

        # Generate the dataset
        typer.echo()
        typer.secho(
            "🚀 Starting dataset generation...", fg=typer.colors.BLUE, bold=True
        )

        # Handle interactive question selection
        question_configs = None
        if interactive_questions:
            from graid.data.generate_dataset import interactive_question_selection
            question_configs = interactive_question_selection()
            if not question_configs:
                typer.secho("No questions selected. Exiting.",
                            fg=typer.colors.YELLOW)
                return

        # Create models from configuration
        models = config.create_models()

        # Lazy import heavy modules only when needed
        from graid.data.generate_dataset import generate_dataset
        
        # Generate the dataset (support multi-split in a single final DatasetDict)
        from datasets import DatasetDict as _HF_DatasetDict

        def _normalize_splits(split_value):
            # Accept list or special combined tokens
            if isinstance(split_value, (list, tuple)):
                return list(split_value)
            value = str(split_value).lower()
            if value in {"train+val", "both", "all", "trainval"}:
                return ["train", "val"]
            return [str(split_value)]

        requested_splits = _normalize_splits(config.split)

        if len(requested_splits) == 1:
            dataset_dict = generate_dataset(
                dataset_name=config.dataset_name,
                split=requested_splits[0],
                models=models,
                use_wbf=config.use_wbf,
                wbf_config=config.wbf_config.to_dict() if config.wbf_config else None,
                conf_threshold=config.confidence_threshold,
                batch_size=config.batch_size,
                device=config.device,
                allowable_set=config.allowable_set,
                question_configs=question_configs or config.question_configs,
                num_workers=num_workers or config.num_workers,
                qa_workers=qa_workers or config.qa_workers,
                save_steps=config.save_steps,
                save_path=config.save_path,
                upload_to_hub=config.upload_to_hub,
                hub_repo_id=config.hub_repo_id,
                hub_private=config.hub_private,
                num_samples=config.num_samples,
                use_original_filenames=config.use_original_filenames,
                filename_prefix=config.filename_prefix,
                force=config.force,
            )
        else:
            # Build each split without saving/pushing; combine and then save/push once
            combined = _HF_DatasetDict()
            for split_name in requested_splits:
                partial = generate_dataset(
                    dataset_name=config.dataset_name,
                    split=split_name,
                    models=models,
                    use_wbf=config.use_wbf,
                    wbf_config=config.wbf_config.to_dict() if config.wbf_config else None,
                    conf_threshold=config.confidence_threshold,
                    batch_size=config.batch_size,
                    device=config.device,
                    allowable_set=config.allowable_set,
                    question_configs=question_configs or config.question_configs,
                    num_workers=num_workers or config.num_workers,
                    qa_workers=qa_workers or config.qa_workers,
                    save_steps=config.save_steps,
                    save_path=config.save_path,
                    upload_to_hub=False,
                    hub_repo_id=None,
                    hub_private=config.hub_private,
                    num_samples=config.num_samples,
                    use_original_filenames=config.use_original_filenames,
                    filename_prefix=config.filename_prefix,
                    force=config.force,
                )
                # Copy the split into combined
                combined[split_name] = partial[split_name]

            # Save combined if requested
            import os as _os
            dry_run = bool(_os.getenv("GRAID_DRY_RUN"))
            # NOTE: Skipping combined.save_to_disk() because individual splits are already 
            # saved efficiently in split directories with images and metadata.parquet
            # if config.save_path and not dry_run:
            #     combined.save_to_disk(config.save_path)
            # Push combined if requested: upload split folders (images + metadata) via large-folder upload
            if config.upload_to_hub and not dry_run:
                if not config.hub_repo_id:
                    raise ValueError("hub_repo_id is required when upload_to_hub=True")

                from huggingface_hub import HfApi as _HfApi
                _api = _HfApi()

                if not config.save_path:
                    raise ValueError("save_path is required to upload folders to the Hub")

                _base_dataset_dir = Path(config.save_path)
                typer.echo("Uploading dataset folder (with split subfolders) to the Hub using upload_large_folder...")
                # Upload the entire dataset directory so train/ and val/ are preserved in repo
                _api.upload_large_folder(
                    repo_id=config.hub_repo_id,
                    repo_type="dataset",
                    folder_path=str(_base_dataset_dir),
                )
                typer.echo("✓ Upload completed")

            dataset_dict = combined

        # Success message
        typer.echo()
        typer.secho(
            "✅ Dataset generation completed successfully!",
            fg=typer.colors.GREEN,
            bold=True,
        )

        # Show summary
        if len(requested_splits) == 1:
            split_dataset = dataset_dict[requested_splits[0]]
            typer.echo(f"📊 Generated {len(split_dataset)} question-answer pairs")
        else:
            counts = ", ".join(f"{s}={len(dataset_dict[s])}" for s in requested_splits)
            typer.echo(f"📊 Generated per-split counts: {counts}")

        if config.save_path:
            typer.echo(f"💾 Saved to: {config.save_path}")

        if config.upload_to_hub:
            typer.echo(f"🤗 Uploaded to HuggingFace Hub: {config.hub_repo_id}")

    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        typer.secho(f"❌ Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("eval-vlms")
def eval_vlms(
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to SQLite database"
    ),
    vlm: str = typer.Option("Llama", help="VLM type to use"),
    model: Optional[str] = typer.Option(
        None, help="Specific model name (required for some VLMs)"
    ),
    metric: str = typer.Option("LLMJudge", help="Evaluation metric"),
    prompt: str = typer.Option("ZeroShotPrompt", help="Prompt type"),
    sample_size: int = typer.Option(
        100, "--sample-size", "-n", help="Sample size per table"
    ),
    region: str = typer.Option("us-central1", help="Cloud region"),
    gpu_id: int = typer.Option(7, "--gpu-id", help="GPU ID"),
    batch: bool = typer.Option(False, help="Use batch processing"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Custom output directory"
    ),
    list_vlms: bool = typer.Option(
        False, "--list-vlms", help="List available VLM types"
    ),
    list_metrics: bool = typer.Option(
        False, "--list-metrics", help="List available metrics"
    ),
    list_prompts: bool = typer.Option(
        False, "--list-prompts", help="List available prompts"
    ),
    interactive: bool = typer.Option(True, help="Use interactive mode"),
):
    """
    Evaluate Vision Language Models using SQLite databases.

    Run without arguments for interactive mode, or specify parameters for batch mode.
    """

    # Handle information commands
    if list_vlms:
        typer.secho("🤖 Available VLM Types:", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        # Local import to avoid heavy dependencies
        from graid.evaluator.eval_vlms import VLM_CONFIGS
        for vlm_type, config in VLM_CONFIGS.items():
            typer.secho(f"{vlm_type}:", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  {config['description']}")
            if config["requires_model_selection"]:
                typer.echo(f"  Available models: {', '.join(config['models'])}")
            typer.echo()
        return

    if list_metrics:
        typer.secho("📊 Available Metrics:", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        # Local import to avoid heavy dependencies
        from graid.evaluator.eval_vlms import METRIC_CONFIGS
        for metric_type, config in METRIC_CONFIGS.items():
            typer.secho(f"{metric_type}:", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  {config['description']}")
        typer.echo()
        return

    if list_prompts:
        typer.secho("💬 Available Prompts:", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        # Local import to avoid heavy dependencies
        from graid.evaluator.eval_vlms import PROMPT_CONFIGS
        for prompt_type, config in PROMPT_CONFIGS.items():
            typer.secho(f"{prompt_type}:", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  {config['description']}")
        typer.echo()
        return

    # Interactive mode for database selection
    if interactive and not db_path:
        typer.secho("🔍 VLM Evaluation", fg=typer.colors.CYAN, bold=True)
        typer.echo()
        typer.echo(
            "This tool evaluates Vision Language Models using SQLite databases")
        typer.echo("containing questions and answers about images.")
        typer.echo()

        db_path = typer.prompt("Enter path to SQLite database")

    # Validate required arguments
    if not db_path:
        typer.secho("Error: --db-path is required", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Check if model name is required
    # Local import to avoid heavy dependencies
    from graid.evaluator.eval_vlms import VLM_CONFIGS
    vlm_config = VLM_CONFIGS.get(vlm)
    if not vlm_config:
        typer.secho(
            f"Error: Unknown VLM type '{vlm}'. Use --list-vlms to see available options.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if vlm_config["requires_model_selection"] and not model:
        typer.secho(
            f"Error: Model selection required for {vlm}.", fg=typer.colors.RED)
        typer.echo(f"Available models: {', '.join(vlm_config['models'])}")
        typer.echo("Use --model to specify a model.")
        raise typer.Exit(1)

    # Start evaluation
    from graid.evaluator.eval_vlms import evaluate_vlm, METRIC_CONFIGS, PROMPT_CONFIGS, VLM_CONFIGS
    typer.secho("🚀 Starting VLM evaluation...", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    typer.echo(f"Database: {db_path}")
    typer.echo(f"VLM: {vlm}" + (f" ({model})" if model else ""))
    typer.echo(f"Metric: {metric}")
    typer.echo(f"Prompt: {prompt}")
    typer.echo(f"Sample Size: {sample_size}")
    typer.echo()

    try:
        accuracy = evaluate_vlm(
            db_path=db_path,
            vlm_type=vlm,
            model_name=model,
            metric=metric,
            prompt=prompt,
            sample_size=sample_size,
            region=region,
            gpu_id=gpu_id,
            use_batch=batch,
            output_dir=output_dir,
        )

        typer.echo()
        typer.secho(
            "✅ VLM evaluation completed successfully!",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo(f"Final accuracy: {accuracy:.4f}")

    except Exception as e:
        typer.echo()
        typer.secho(
            f"❌ Error during evaluation: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)


@app.command()
def list_models():
    """List all available pre-configured models."""
    typer.secho("📋 Available Models", fg=typer.colors.BLUE, bold=True)
    typer.echo()

    # Local import to avoid heavy dependencies
    from graid.data.generate_db import list_available_models
    models = list_available_models()
    for backend, model_list in models.items():
        typer.secho(f"{backend.upper()}:", fg=typer.colors.GREEN, bold=True)
        for model in model_list:
            typer.echo(f"  • {model}")
        typer.echo()


@app.command()
def list_questions():
    """List available questions with their parameters."""
    typer.secho("📋 Available Questions:", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    # Local import to avoid heavy dependencies
    from graid.data.generate_dataset import list_available_questions
    questions = list_available_questions()
    for i, (name, info) in enumerate(questions.items(), 1):
        typer.secho(f"{i:2d}. {name}", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"    {info['question']}")
        if info["parameters"]:
            typer.echo("    Parameters:")
            for param_name, param_info in info["parameters"].items():
                typer.echo(
                    f"      • {param_name}: {param_info['description']} (default: {param_info['default']})"
                )
        typer.echo()

    typer.secho("💡 Usage:", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        "Use --interactive-questions flag with generate-dataset for interactive selection"
    )
    typer.echo("Or configure questions in a config file")


@app.command()
def info():
    """Show information about GRAID and supported datasets/models."""
    print_welcome()

    typer.secho("📊 Supported Datasets:", fg=typer.colors.BLUE, bold=True)
    # Local import to avoid heavy dependencies
    from graid.data.generate_db import DATASET_TRANSFORMS
    for dataset in DATASET_TRANSFORMS.keys():
        typer.echo(f"  • {dataset.upper()}")
    typer.echo()

    typer.secho("🧠 Supported Model Backends:", fg=typer.colors.BLUE, bold=True)
    for backend in ["detectron", "mmdetection", "ultralytics"]:
        typer.echo(f"  • {backend.upper()}")
    typer.echo()

    typer.secho("🛠️ Custom Model Support:", fg=typer.colors.BLUE, bold=True)
    typer.echo("  • Detectron2: Provide config.yaml and weights file")
    typer.echo("  • MMDetection: Provide config.py and checkpoint file")
    typer.echo()


if __name__ == "__main__":
    app()
