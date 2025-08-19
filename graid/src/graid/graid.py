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
from typing import Optional, List, Dict

import typer



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
    
    # Create a custom filter to only show GRAID logs on console
    class GraidLogFilter(logging.Filter):
        def filter(self, record):
            # Only show logs from graid modules (and a few important system messages)
            return (record.name.startswith('graid.') or 
                   record.name == 'graid' or
                   record.levelno >= logging.WARNING)  # Always show warnings/errors from any source
    
    # Console handler with GRAID-only filter
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    ch.addFilter(GraidLogFilter())  # Only show GRAID logs on console
    logger.addHandler(ch)
    
    # File handler with timestamp
    log_dir = os.getenv("GRAID_LOG_DIR", "logs")
    # Create log directory with proper error handling
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Log to stderr if we can't create log directory
        print(f"Warning: Permission denied creating log directory: {log_dir}", file=sys.stderr)
        log_dir = "/tmp"  # Fallback to /tmp
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        except Exception as fallback_e:
            print(f"Warning: Could not create fallback log directory: {fallback_e}", file=sys.stderr)
            log_dir = None
    except OSError as e:
        print(f"Warning: OS error creating log directory {log_dir}: {e}", file=sys.stderr)
        log_dir = None
    except Exception as e:
        print(f"Warning: Unexpected error creating log directory {log_dir}: {e}", file=sys.stderr)
        log_dir = None
    
    # Generate timestamped log filename and create file handler
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"graid_{timestamp}.log"
    
    # Only create file handler if we have a valid log directory
    if log_dir is not None:
        try:
            fh = logging.FileHandler(Path(log_dir) / log_filename)
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Warning: Failed to create log file handler: {e}", file=sys.stderr)
            print("Logging will only go to console", file=sys.stderr)
    else:
        print("Warning: No log directory available, logging only to console", file=sys.stderr)
    # Quiet noisy libraries more aggressively
    logging.getLogger("mmengine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.Image").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


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
    typer.echo("GRAID provides two main capabilities:")
    typer.echo()
    typer.secho("🗄️ Dataset Generation (generate-dataset):",
                fg=typer.colors.BLUE, bold=True)
    typer.echo("• Multiple datasets: BDD100K, NuImages, Waymo + Custom datasets")
    typer.echo("• Multi-backend support: Detectron2, MMDetection, Ultralytics")
    typer.echo("• Ensemble models with Weighted Box Fusion (WBF)")
    typer.echo("• Ground truth data or custom model predictions")
    typer.echo("• Unlabeled image support (models generate detections)")
    typer.echo("• Standard formats with COCO-style annotations")
    typer.echo("• Interactive configuration and batch processing")
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

    from graid.data.generate_db import DATASET_TRANSFORMS
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
        except ValueError as e:
            typer.secho(f"Invalid input: Expected a number", fg=typer.colors.RED)
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
        except ValueError as e:
            typer.secho(f"Invalid input: Expected a number", fg=typer.colors.RED)
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

    from graid.data.generate_db import (
        DATASET_TRANSFORMS,
        MODEL_CONFIGS,
        generate_db,
    )

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

        # Note: Custom model configuration support would need to be implemented
        # in the model creation logic. For now, custom models are not supported 
        # in non-interactive mode.

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

    except KeyboardInterrupt:
        typer.echo("\n⏹️  Generation cancelled by user")
        raise typer.Exit(130)  # Standard exit code for SIGINT
    except PermissionError as e:
        typer.echo()
        typer.secho(f"❌ Permission Error: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check file/directory permissions and try again.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.echo()
        typer.secho(f"❌ File Not Found: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo()
        typer.secho(f"❌ Invalid Value: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check your input parameters and try again.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo()
        typer.secho(f"❌ Unexpected Error: {e}", fg=typer.colors.RED, bold=True)
        if os.getenv("GRAID_DEBUG_VERBOSE"):
            import traceback
            typer.echo("Detailed traceback:")
            typer.echo(traceback.format_exc())
        else:
            typer.secho("Use GRAID_DEBUG_VERBOSE=1 for detailed error information.", fg=typer.colors.CYAN)
        raise typer.Exit(1)


def _handle_list_commands(
    list_valid_objects: bool,
    list_questions: bool
) -> bool:
    """Handle --list-objects and --list-questions commands."""
    if list_valid_objects:
        from graid.utilities.coco import coco_labels
        typer.secho("📋 Valid COCO Objects", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        
        valid_objects = list(coco_labels.values())
        # Remove undefined as it's not a real COCO class
        if "undefined" in valid_objects:
            valid_objects.remove("undefined")
        valid_objects.sort()
        
        for i, obj in enumerate(valid_objects, 1):
            typer.echo(f"  {i:2d}. {obj}")
        
        typer.echo()
        typer.echo(f"Total: {len(valid_objects)} objects")
        return True
    
    if list_questions:
        from graid.data.generate_dataset import list_available_questions
        typer.secho("📋 Available Questions", fg=typer.colors.BLUE, bold=True)
        typer.echo()
        
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
        
        typer.echo(f"Total: {len(questions)} question types")
        return True
    
    return False


def _handle_interactive_questions(interactive_questions: bool) -> Optional[List[Dict]]:
    """Handle interactive question selection."""
    if interactive_questions:
        from graid.data.generate_dataset import interactive_question_selection
        return interactive_question_selection()
    return None


def _load_and_validate_config(
    config_file: Optional[str],
    **cli_args
):
    """Load configuration from file and apply CLI overrides."""
    from graid.cli_helpers import ConfigurationManager
    
    if config_file:
        config = ConfigurationManager.load_from_file(config_file)
        config = ConfigurationManager.apply_cli_overrides(config, **cli_args)
    else:
        # Create config from CLI args only
        if not cli_args.get('dataset') or not cli_args.get('split'):
            raise ValueError("Either --config file or both --dataset and --split are required")
        
        # Local import for config creation
        from graid.data.config_support import DatasetGenerationConfig
        config = DatasetGenerationConfig(
            dataset_name=cli_args['dataset'],
            split=cli_args['split'],
            models=[],
            use_wbf=False,
            confidence_threshold=0.0,
            batch_size=32,
            device=None,
            allowable_set=cli_args.get('allowable_set', []).split(',') if cli_args.get('allowable_set') else None,
            num_workers=cli_args.get('num_workers', 4),
            qa_workers=cli_args.get('qa_workers', 4),
            save_path=cli_args.get('save_path'),
            upload_to_hub=cli_args.get('upload_to_hub', False),
            hub_repo_id=cli_args.get('hub_repo_id'),
            hub_private=cli_args.get('hub_private', False),
            num_samples=None,
            use_original_filenames=True,
            filename_prefix='img',
            force=cli_args.get('force', False),
            question_configs=[{'name': 'HowMany', 'params': {}}]  # Default question
        )
    
    # Final validation
    ConfigurationManager.validate_final_config(config)
    return config


def _process_dataset_generation(config, question_configs: Optional[List] = None):
    """Process the actual dataset generation."""
    from graid.cli_helpers import DatasetProcessor, ArgumentValidator
    
    # Override question configs if provided interactively
    if question_configs:
        config.question_configs = question_configs
    
    # Determine processing strategy
    splits = ArgumentValidator.validate_split_format(config.split)
    
    if len(splits) == 1:
        return DatasetProcessor.process_single_split(config)
    else:
        return DatasetProcessor.process_multiple_splits(config)


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
    save_path: str = typer.Option(
        "./graid-datasets", help="Path to save the generated dataset"
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
    
    from graid.cli_helpers import (
        ValidationError, ConfigurationError, ProcessingError, ErrorHandler
    )
    
    try:
        # Handle list commands first
        if _handle_list_commands(list_valid_objects, list_questions):
            return
        
        print_welcome()
        
        # Handle interactive question selection
        question_configs = _handle_interactive_questions(interactive_questions)
        
        # Load and validate configuration
        typer.echo("📄 Loading and validating configuration...")
        
        # Only include CLI args that were explicitly provided (not defaults)
        cli_args = {}
        
        # String/Optional args (None means not provided)
        if dataset is not None:
            cli_args['dataset'] = dataset
        if split is not None:
            cli_args['split'] = split
        if allowable_set is not None:
            cli_args['allowable_set'] = allowable_set
        if hub_repo_id is not None:
            cli_args['hub_repo_id'] = hub_repo_id
            
        # Boolean args - need to check if explicitly set vs default
        # For typer, we need to detect if these were provided by user
        # Since typer doesn't provide an easy way, we'll use a different approach
        import sys
        cli_flags = sys.argv[1:]  # Get CLI arguments
        
        if '--upload-to-hub' in cli_flags:
            cli_args['upload_to_hub'] = upload_to_hub
        if '--hub-private' in cli_flags:
            cli_args['hub_private'] = hub_private
        if '--force' in cli_flags:
            cli_args['force'] = force
            
        # For numeric args, check if they differ from defaults
        if '--num-workers' in cli_flags or '-j' in cli_flags:
            cli_args['num_workers'] = num_workers
        if '--qa-workers' in cli_flags:
            cli_args['qa_workers'] = qa_workers
        
        # For save_path, only override if explicitly provided
        if '--save-path' in cli_flags:
            cli_args['save_path'] = save_path
        
        config = _load_and_validate_config(config_file, **cli_args)
        typer.secho("✓ Configuration validated successfully", fg=typer.colors.GREEN)
        
        # Display configuration summary
        typer.echo()
        typer.secho("📋 Configuration Summary", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"Dataset: {config.dataset_name}")
        typer.echo(f"Split: {config.split}")
        typer.echo(f"Models: {len(getattr(config, 'models', [])) if getattr(config, 'models', None) else 0} (using ground truth if 0)")
        typer.echo(f"Batch size: {getattr(config, 'batch_size', 32)}")
        typer.echo(f"Workers: {config.num_workers} (loading), {config.qa_workers} (QA)")
        typer.echo(f"Save path: {config.save_path}")
        if config.allowable_set:
            typer.echo(f"COCO filter: {', '.join(config.allowable_set[:3])}{'...' if len(config.allowable_set) > 3 else ''}")
        typer.echo(f"Upload to Hub: {'Yes' if config.upload_to_hub else 'No'}")
        if config.upload_to_hub:
            typer.echo(f"Hub repo: {config.hub_repo_id}")
        typer.echo()
        
        # Start dataset generation
        typer.secho("🚀 Starting dataset generation...", fg=typer.colors.BLUE, bold=True)
        dataset_dict = _process_dataset_generation(config, question_configs)
        
        # Success reporting
        typer.echo()
        typer.secho("✅ Dataset generation completed successfully!", fg=typer.colors.GREEN, bold=True)
        
        total_pairs = sum(len(dataset) for dataset in dataset_dict.values())
        typer.echo(f"📊 Generated {total_pairs} question-answer pairs")
        
        if len(dataset_dict) > 1:
            counts = ", ".join(f"{s}={len(dataset_dict[s])}" for s in dataset_dict.keys())
            typer.echo(f"📊 Per-split counts: {counts}")

        if config.save_path:
            typer.echo(f"💾 Saved to: {config.save_path}")

        if config.upload_to_hub:
            typer.echo(f"🤗 Uploaded to HuggingFace Hub: {config.hub_repo_id}")
    
    except ValidationError as e:
        ErrorHandler.handle_validation_error(e)
    except ConfigurationError as e:
        ErrorHandler.handle_configuration_error(e)
    except ProcessingError as e:
        ErrorHandler.handle_processing_error(e)
    except Exception as e:
        ErrorHandler.handle_unexpected_error(e)





def _load_configuration(config_file, interactive, interactive_questions, **cli_args):
    """Load configuration from file or interactive mode with CLI overrides."""
    from graid.cli import ConfigurationManager
    
    if config_file:
        # Load from file and apply CLI overrides
        config = ConfigurationManager.load_from_file(config_file)
        config = ConfigurationManager.apply_cli_overrides(config, **cli_args)
        
        typer.secho("✓ Configuration loaded from:", fg=typer.colors.GREEN)
        typer.echo(f"  {config_file}")
        typer.echo()
    else:
        # Interactive configuration
        config = ConfigurationManager.create_interactive_config(
            interactive_questions=interactive_questions, **cli_args
        )
    
    return config


def _validate_configuration(config):
    """Validate final configuration."""
    from graid.cli import ConfigurationManager
    ConfigurationManager.validate_configuration(config)


def _report_success(dataset_dict, config):
    """Report successful completion with summary."""
    from graid.cli.validators import ArgumentValidator
    
    requested_splits = ArgumentValidator.parse_and_validate_split(config.split)
    
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
        from graid.evaluator.eval_vlms import evaluate_vlm
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

    except KeyboardInterrupt:
        typer.echo("\n⏹️  Evaluation cancelled by user")
        raise typer.Exit(130)  # Standard exit code for SIGINT
    except FileNotFoundError as e:
        typer.echo()
        typer.secho(f"❌ File Not Found: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check that the database file exists and try again.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except PermissionError as e:
        typer.echo()
        typer.secho(f"❌ Permission Error: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check file permissions and try again.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo()
        typer.secho(f"❌ Invalid Parameter: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check your evaluation parameters and try again.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except ImportError as e:
        typer.echo()
        typer.secho(f"❌ Import Error: {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Check that VLM dependencies are installed.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo()
        typer.secho(f"❌ Unexpected Error during evaluation: {e}", fg=typer.colors.RED, bold=True)
        if os.getenv("GRAID_DEBUG_VERBOSE"):
            import traceback
            typer.echo("Detailed traceback:")
            typer.echo(traceback.format_exc())
        else:
            typer.secho("Use GRAID_DEBUG_VERBOSE=1 for detailed error information.", fg=typer.colors.CYAN)
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

    from graid.data.generate_db import DATASET_TRANSFORMS, MODEL_CONFIGS
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
