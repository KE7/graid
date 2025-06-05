"""
GRAID (Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence) - Main CLI Interface

An interactive command-line tool for generating object detection databases 
using various models and datasets.
"""

import typer
from typing import Optional, Dict, List
from pathlib import Path
import sys
import os

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from graid.data.generate_db import (
    generate_db, 
    list_available_models, 
    MODEL_CONFIGS,
    DATASET_TRANSFORMS
)

app = typer.Typer(
    name="graid",
    help="GRAID: Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence",
    add_completion=False,
)

def print_welcome():
    """Print welcome message and project info."""
    typer.echo()
    typer.secho("🤖 Welcome to GRAID!", fg=typer.colors.CYAN, bold=True)
    typer.echo("   Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence")
    typer.echo()
    typer.echo("This tool helps you generate object detection databases using:")
    typer.echo("• Multiple datasets: BDD100K, NuImages, Waymo")
    typer.echo("• Various model backends: Detectron, MMDetection, Ultralytics")
    typer.echo("• Ground truth data or custom model predictions")
    typer.echo()

def get_dataset_choice() -> str:
    """Interactive dataset selection."""
    typer.secho("📊 Step 1: Choose a dataset", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    
    datasets = {
        "1": ("bdd", "BDD100K - Berkeley DeepDrive autonomous driving dataset"),
        "2": ("nuimage", "NuImages - Large-scale autonomous driving dataset"), 
        "3": ("waymo", "Waymo Open Dataset - Self-driving car dataset")
    }
    
    for key, (name, desc) in datasets.items():
        typer.echo(f"  {key}. {typer.style(name.upper(), fg=typer.colors.GREEN)} - {desc}")
    
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
        "2": ("val", "Validation set - used for model evaluation")
    }
    
    for key, (name, desc) in splits.items():
        typer.echo(f"  {key}. {typer.style(name.upper(), fg=typer.colors.GREEN)} - {desc}")
    
    typer.echo()
    while True:
        choice = typer.prompt("Select split (1-2)")
        if choice in splits:
            split_name = splits[choice][0]
            typer.secho(f"✓ Selected: {split_name.upper()}", fg=typer.colors.GREEN)
            typer.echo()
            return split_name
        typer.secho("Invalid choice. Please enter 1 or 2.", fg=typer.colors.RED)

def get_model_choice() -> tuple[Optional[str], Optional[str], Optional[Dict]]:
    """Interactive model selection with custom model support."""
    typer.secho("🧠 Step 3: Choose model type", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    
    typer.echo("  1. Ground Truth - Use original dataset annotations (fastest)")
    typer.echo("  2. Pre-configured Models - Choose from built-in model configurations")
    typer.echo("  3. Custom Model - Bring your own Detectron/MMDetection model")
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
            
        typer.secho("Invalid choice. Please enter 1, 2, or 3.", fg=typer.colors.RED)

def get_preconfigured_model() -> tuple[str, str, None]:
    """Interactive pre-configured model selection."""
    typer.echo()
    typer.secho("🔧 Pre-configured Models", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    
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
                break
        except ValueError:
            pass
        typer.secho("Invalid choice. Please enter a valid number.", fg=typer.colors.RED)
    
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
        typer.secho("Invalid choice. Please enter a valid number.", fg=typer.colors.RED)
    
    typer.secho(f"✓ Selected: {backend.upper()} - {model_name}", fg=typer.colors.GREEN)
    typer.echo()
    return backend, model_name, None

def get_custom_model() -> tuple[str, str, Dict]:
    """Interactive custom model configuration."""
    typer.echo()
    typer.secho("🛠️ Custom Model Configuration", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    
    typer.echo("Supported backends for custom models:")
    typer.echo("  1. Detectron2 - Facebook's object detection framework")
    typer.echo("  2. MMDetection - OpenMMLab's detection toolbox")
    typer.echo()
    
    while True:
        choice = typer.prompt("Select backend (1-2)")
        if choice == "1":
            backend = "detectron"
            break
        elif choice == "2":
            backend = "mmdetection"
            break
        typer.secho("Invalid choice. Please enter 1 or 2.", fg=typer.colors.RED)
    
    typer.echo()
    custom_config = {}
    
    if backend == "detectron":
        typer.echo("Detectron2 Configuration:")
        typer.echo("You need to provide paths to configuration and weights files.")
        typer.echo()
        
        config_file = typer.prompt("Config file path (e.g., 'COCO-Detection/retinanet_R_50_FPN_3x.yaml')")
        weights_file = typer.prompt("Weights file path (e.g., 'path/to/model.pth')")
        
        custom_config = {
            "config": config_file,
            "weights": weights_file
        }
        
    elif backend == "mmdetection":
        typer.echo("MMDetection Configuration:")
        typer.echo("You need to provide paths to configuration and checkpoint files.")
        typer.echo()
        
        config_file = typer.prompt("Config file path (e.g., 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')")
        checkpoint = typer.prompt("Checkpoint file path or URL")
        
        custom_config = {
            "config": config_file,
            "checkpoint": checkpoint
        }
    
    # Generate a custom model name
    model_name = f"custom_{Path(custom_config.get('config', 'model')).stem}"
    
    typer.secho(f"✓ Custom model configured: {backend.upper()}", fg=typer.colors.GREEN)
    typer.echo()
    return backend, model_name, custom_config

def get_confidence_threshold() -> float:
    """Interactive confidence threshold selection."""
    typer.secho("🎯 Step 4: Set confidence threshold", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    typer.echo("Confidence threshold filters out low-confidence detections.")
    typer.echo("• Lower values (0.1-0.3): More detections, some false positives")
    typer.echo("• Higher values (0.5-0.8): Fewer detections, higher precision")
    typer.echo()
    
    while True:
        try:
            conf = float(typer.prompt("Enter confidence threshold", default="0.2"))
            if 0.0 <= conf <= 1.0:
                typer.secho(f"✓ Confidence threshold: {conf}", fg=typer.colors.GREEN)
                typer.echo()
                return conf
            typer.secho("Please enter a value between 0.0 and 1.0.", fg=typer.colors.RED)
        except ValueError:
            typer.secho("Please enter a valid number.", fg=typer.colors.RED)

@app.command()
def generate(
    dataset: Optional[str] = typer.Option(None, help="Dataset name (bdd, nuimage, waymo)"),
    split: Optional[str] = typer.Option(None, help="Data split (train, val)"),
    backend: Optional[str] = typer.Option(None, help="Model backend"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    conf: Optional[float] = typer.Option(None, help="Confidence threshold"),
    config: Optional[str] = typer.Option(None, help="Custom model config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Custom model checkpoint/weights"),
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
            typer.secho("Error: dataset and split are required in non-interactive mode", fg=typer.colors.RED)
            raise typer.Exit(1)
        
        if dataset not in DATASET_TRANSFORMS:
            typer.secho(f"Error: Invalid dataset '{dataset}'. Choose from: {list(DATASET_TRANSFORMS.keys())}", fg=typer.colors.RED)
            raise typer.Exit(1)
        
        if split not in ["train", "val"]:
            typer.secho("Error: Invalid split. Choose 'train' or 'val'", fg=typer.colors.RED)
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
    if 'custom_config' in locals() and custom_config:
        # Add custom model to MODEL_CONFIGS temporarily
        if backend not in MODEL_CONFIGS:
            MODEL_CONFIGS[backend] = {}
        MODEL_CONFIGS[backend][model] = custom_config
    
    # Start generation
    typer.secho("🚀 Starting database generation...", fg=typer.colors.BLUE, bold=True)
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
        db_name = generate_db(
            dataset_name=dataset,
            split=split,
            conf=conf,
            backend=backend,
            model_name=model,
        )
        
        typer.echo()
        typer.secho("✅ Database generation completed successfully!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"Database created: {db_name}")
        
    except Exception as e:
        typer.echo()
        typer.secho(f"❌ Error during generation: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)

@app.command()
def list_models():
    """List all available pre-configured models."""
    typer.secho("📋 Available Models", fg=typer.colors.BLUE, bold=True)
    typer.echo()
    
    models = list_available_models()
    for backend, model_list in models.items():
        typer.secho(f"{backend.upper()}:", fg=typer.colors.GREEN, bold=True)
        for model in model_list:
            typer.echo(f"  • {model}")
        typer.echo()

@app.command()
def info():
    """Show information about GRAID and supported datasets/models."""
    print_welcome()
    
    typer.secho("📊 Supported Datasets:", fg=typer.colors.BLUE, bold=True)
    for dataset in DATASET_TRANSFORMS.keys():
        typer.echo(f"  • {dataset.upper()}")
    typer.echo()
    
    typer.secho("🧠 Supported Model Backends:", fg=typer.colors.BLUE, bold=True)
    for backend in MODEL_CONFIGS.keys():
        typer.echo(f"  • {backend.upper()}")
    typer.echo()
    
    typer.secho("🛠️ Custom Model Support:", fg=typer.colors.BLUE, bold=True)
    typer.echo("  • Detectron2: Provide config.yaml and weights file")
    typer.echo("  • MMDetection: Provide config.py and checkpoint file")
    typer.echo()

if __name__ == "__main__":
    app() 