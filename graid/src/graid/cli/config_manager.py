"""
Configuration management for CLI commands.

Handles loading configuration from files and merging with CLI arguments.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import typer

from graid.data.config_support import load_config_from_file, DatasetGenerationConfig
from graid.cli.exceptions import ConfigurationError


class ConfigurationManager:
    """Handles loading and merging configuration from files and CLI."""
    
    @staticmethod
    def load_from_file(config_file: str) -> DatasetGenerationConfig:
        """Load configuration from JSON file."""
        try:
            typer.secho(
                "📄 Loading configuration from file...", fg=typer.colors.BLUE, bold=True
            )
            return load_config_from_file(config_file)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    @staticmethod
    def apply_cli_overrides(config: DatasetGenerationConfig, **cli_args) -> DatasetGenerationConfig:
        """Apply CLI arguments that override config file values."""
        # Remove None values to avoid overriding config with None
        cli_args = {k: v for k, v in cli_args.items() if v is not None}
        
        # Map CLI arguments to config attributes
        if 'force' in cli_args and cli_args['force']:
            config.force = True
        if 'save_path' in cli_args:
            config.save_path = cli_args['save_path']
        if 'upload_to_hub' in cli_args and cli_args['upload_to_hub']:
            config.upload_to_hub = True
        if 'hub_repo_id' in cli_args:
            config.hub_repo_id = cli_args['hub_repo_id']
        if 'hub_private' in cli_args and cli_args['hub_private']:
            config.hub_private = True
        if 'dataset' in cli_args:
            config.dataset_name = cli_args['dataset']
        if 'split' in cli_args:
            config.split = cli_args['split']
        if 'num_workers' in cli_args:
            config.num_workers = cli_args['num_workers']
        if 'qa_workers' in cli_args:
            config.qa_workers = cli_args['qa_workers']
        if 'allowable_set' in cli_args:
            # Parse comma-separated allowable set
            allowable_str = cli_args['allowable_set']
            if allowable_str:
                config.allowable_set = [obj.strip() for obj in allowable_str.split(',') if obj.strip()]
            else:
                config.allowable_set = None
        
        return config
    
    @staticmethod
    def create_interactive_config(**cli_args) -> DatasetGenerationConfig:
        """Create configuration through interactive prompts."""
        from graid.graid import (
            get_dataset_name, get_split, get_model_selection,
            get_confidence_threshold, interactive_question_selection
        )
        
        # Interactive configuration gathering
        typer.secho("🛠️ Interactive Configuration", fg=typer.colors.BLUE, bold=True)
        typer.echo("Let's configure your dataset generation step by step.")
        typer.echo()
        
        dataset_name = get_dataset_name()
        split = get_split()
        backend_name, model_name, custom_config = get_model_selection()
        confidence_threshold = get_confidence_threshold()
        
        # Get question selection
        if cli_args.get('interactive_questions'):
            # Local import to avoid heavy dependencies
            from graid.data.generate_dataset import interactive_question_selection
            question_configs = interactive_question_selection()
        else:
            # Use default question set
            question_configs = [
                {"name": "HowMany", "params": {}},
                {"name": "IsObjectCentered", "params": {"buffer_ratio": 0.05}},
            ]
        
        # Create configuration object
        config = DatasetGenerationConfig(
            dataset_name=dataset_name,
            split=split,
            models=[model_name] if model_name else [],
            confidence_threshold=confidence_threshold,
            question_configs=question_configs,
            save_path=cli_args.get('save_path'),
            upload_to_hub=cli_args.get('upload_to_hub', False),
            hub_repo_id=cli_args.get('hub_repo_id'),
            hub_private=cli_args.get('hub_private', False),
            num_workers=cli_args.get('num_workers', 4),
            qa_workers=cli_args.get('qa_workers', 4),
            force=cli_args.get('force', False),
        )
        
        # Handle allowable set if provided via CLI
        if cli_args.get('allowable_set'):
            allowable_str = cli_args['allowable_set']
            config.allowable_set = [obj.strip() for obj in allowable_str.split(',') if obj.strip()]
        
        return config
    
    @staticmethod
    def validate_configuration(config: DatasetGenerationConfig):
        """Validate final configuration for consistency."""
        # Import validators locally
        from graid.cli.validators import ArgumentValidator
        
        validator = ArgumentValidator()
        
        # Validate core parameters
        validator.require_valid_dataset(config.dataset_name)
        validator.require_valid_split(config.split)
        
        if config.allowable_set:
            validator.require_valid_coco_objects(config.allowable_set)
        
        # Validate upload configuration
        if config.upload_to_hub and not config.hub_repo_id:
            raise ConfigurationError("hub_repo_id is required when upload_to_hub=True")
        
        # Validate save path
        if not config.save_path:
            raise ConfigurationError("save_path is required")
        
        # Validate question configuration
        if not config.question_configs:
            raise ConfigurationError("At least one question type must be configured")
        
        typer.secho("✓ Configuration validated successfully", fg=typer.colors.GREEN)
        typer.echo(f"  Dataset: {config.dataset_name}")
        typer.echo(f"  Split: {config.split}")
        typer.echo(f"  Questions: {len(config.question_configs)} types")
        typer.echo(f"  Save path: {config.save_path}")
        if config.upload_to_hub:
            typer.echo(f"  Hub upload: {config.hub_repo_id}")
        typer.echo()
