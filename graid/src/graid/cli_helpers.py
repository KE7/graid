"""
CLI Helper Classes for GRAID

Provides centralized configuration management, validation, and error handling
for all GRAID CLI commands.
"""

import logging
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import typer

logger = logging.getLogger(__name__)


class GraidError(Exception):
    """Base exception for GRAID CLI errors."""

    pass


class ValidationError(GraidError):
    """Configuration or argument validation error."""

    pass


class DatasetValidationError(ValidationError):
    """Dataset name validation error."""

    pass


class COCOValidationError(ValidationError):
    """COCO object validation error."""

    pass


class ConfigurationError(GraidError):
    """Configuration loading or processing error."""

    pass


class ProcessingError(GraidError):
    """Dataset generation or processing error."""

    pass


class ArgumentValidator:
    """Centralized validation logic for CLI arguments."""

    @staticmethod
    def validate_dataset_name(dataset_name: str) -> str:
        """
        Validate dataset name against supported options.

        Args:
            dataset_name: Dataset name to validate

        Returns:
            Validated dataset name

        Raises:
            DatasetValidationError: If dataset name is not supported
        """
        # Local import to avoid heavy dependencies
        from graid.data.loaders import DatasetLoaderFactory

        supported_datasets = DatasetLoaderFactory.get_supported_datasets()
        if dataset_name not in supported_datasets:
            raise DatasetValidationError(
                f"Invalid dataset: '{dataset_name}'. Supported: {supported_datasets}"
            )
        return dataset_name

    @staticmethod
    def validate_split_format(split_value: str) -> List[str]:
        """
        Parse and validate split specification.

        Args:
            split_value: Split value (e.g., "train", "train+val", "train,val")

        Returns:
            List of validated split names

        Raises:
            ValidationError: If split format is invalid
        """
        if not split_value:
            raise ValidationError("Split value cannot be empty")

        # Handle different split formats
        if "+" in split_value:
            splits = split_value.split("+")
        elif "," in split_value:
            splits = split_value.split(",")
        else:
            splits = [split_value]

        # Clean and validate each split
        valid_splits = ["train", "val", "validation", "test"]
        cleaned_splits = []

        for split in splits:
            split = split.strip()
            if not split:
                continue
            if split not in valid_splits:
                raise ValidationError(
                    f"Invalid split: '{split}'. Valid splits: {valid_splits}"
                )
            cleaned_splits.append(split)

        if not cleaned_splits:
            raise ValidationError("No valid splits found")

        return cleaned_splits

    @staticmethod
    def validate_coco_objects(allowable_set: List[str]) -> List[str]:
        """
        Validate COCO object class names.

        Args:
            allowable_set: List of COCO object class names

        Returns:
            Validated list of COCO object names

        Raises:
            COCOValidationError: If any object names are invalid
        """
        if not allowable_set:
            return allowable_set

        # Local import for COCO validation
        from graid.utilities.coco import validate_coco_objects

        is_valid, error_msg = validate_coco_objects(allowable_set)
        if not is_valid:
            raise COCOValidationError(error_msg)

        return allowable_set

    @staticmethod
    def validate_path(path: Optional[str], must_exist: bool = False) -> Optional[Path]:
        """
        Validate and convert path string to Path object.

        Args:
            path: Path string to validate
            must_exist: Whether the path must already exist

        Returns:
            Validated Path object or None

        Raises:
            ValidationError: If path validation fails
        """
        if not path:
            return None

        try:
            path_obj = Path(path)

            if must_exist and not path_obj.exists():
                raise ValidationError(f"Path does not exist: {path}")

            return path_obj

        except Exception as e:
            raise ValidationError(f"Invalid path '{path}': {e}")

    @staticmethod
    def validate_hub_config(upload_to_hub: bool, hub_repo_id: Optional[str]) -> None:
        """
        Validate HuggingFace Hub configuration.

        Args:
            upload_to_hub: Whether uploading to hub is requested
            hub_repo_id: Hub repository ID

        Raises:
            ValidationError: If hub configuration is invalid
        """
        if upload_to_hub and not hub_repo_id:
            raise ValidationError("hub_repo_id is required when upload_to_hub=True")

        if hub_repo_id and "/" not in hub_repo_id:
            raise ValidationError(
                "hub_repo_id must be in format 'username/repo-name' or 'org/repo-name'"
            )


class ConfigurationManager:
    """Handles loading and merging configuration from files and CLI."""

    @staticmethod
    def load_from_file(config_file: str):
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to configuration file

        Returns:
            DatasetGenerationConfig object

        Raises:
            ConfigurationError: If config loading fails
        """
        try:
            # Local import for config support
            from graid.data.config_support import load_config_from_file

            config_path = Path(config_file)
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_file}")

            return load_config_from_file(config_file)

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration from {config_file}: {e}"
            )

    @staticmethod
    def apply_cli_overrides(config, **cli_args):
        """
        Apply CLI arguments that override config file values.

        Args:
            config: Configuration object to modify
            **cli_args: CLI arguments to apply

        Returns:
            Modified configuration object
        """
        # Override config values with CLI arguments (only if CLI arg is not None)
        override_fields = [
            "save_path",
            "upload_to_hub",
            "hub_repo_id",
            "hub_private",
            "dataset",
            "split",
            "num_workers",
            "qa_workers",
            "allowable_set",
        ]

        for field in override_fields:
            cli_value = cli_args.get(field)
            if cli_value is not None:
                # Map CLI field names to config attribute names
                config_attr = field
                if field == "dataset":
                    config_attr = "dataset_name"

                setattr(config, config_attr, cli_value)

        return config

    @staticmethod
    def validate_final_config(config) -> None:
        """
        Validate final configuration for consistency.

        Args:
            config: Configuration object to validate

        Raises:
            ValidationError: If configuration is invalid
        """
        validator = ArgumentValidator()

        # Validate required fields
        if not config.dataset_name:
            raise ValidationError("Dataset name is required")

        if not config.split:
            raise ValidationError("Split is required")

        # Validate individual fields
        validator.validate_dataset_name(config.dataset_name)
        validator.validate_split_format(config.split)

        if config.allowable_set:
            validator.validate_coco_objects(config.allowable_set)

        validator.validate_hub_config(config.upload_to_hub, config.hub_repo_id)

        # Validate numeric fields
        if config.batch_size <= 0:
            raise ValidationError("batch_size must be positive")

        if config.num_workers < 0:
            raise ValidationError("num_workers must be non-negative")

        if config.qa_workers <= 0:
            raise ValidationError("qa_workers must be positive")


class ErrorHandler:
    """Standardized error handling for CLI commands."""

    @staticmethod
    def handle_validation_error(error: ValidationError) -> None:
        """
        Handle validation errors with appropriate user messaging.

        Args:
            error: The validation error to handle
        """
        typer.secho(f"❌ Validation Error: {error}", fg=typer.colors.RED)
        typer.echo("Use --help for usage information.")
        raise typer.Exit(1)

    @staticmethod
    def handle_configuration_error(error: ConfigurationError) -> None:
        """
        Handle configuration errors.

        Args:
            error: The configuration error to handle
        """
        typer.secho(f"❌ Configuration Error: {error}", fg=typer.colors.RED)
        typer.echo("Check your configuration file and try again.")
        raise typer.Exit(1)

    @staticmethod
    def handle_processing_error(error: ProcessingError) -> None:
        """
        Handle processing errors with debugging information.

        Args:
            error: The processing error to handle
        """
        typer.secho(f"❌ Processing Error: {error}", fg=typer.colors.RED)

        # Show traceback in debug mode
        if os.getenv("GRAID_DEBUG_VERBOSE"):
            import traceback

            typer.echo("\nDetailed traceback:")
            typer.echo(traceback.format_exc())
        else:
            typer.echo("Use GRAID_DEBUG_VERBOSE=1 for detailed error information.")

        raise typer.Exit(1)

    @staticmethod
    def handle_unexpected_error(error: Exception) -> None:
        """
        Handle unexpected errors.

        Args:
            error: The unexpected error to handle
        """
        typer.secho(f"❌ Unexpected Error: {error}", fg=typer.colors.RED)
        typer.echo(
            "This is likely a bug. Please report it with the following information:"
        )

        import traceback

        typer.echo("\nFull traceback:")
        typer.echo(traceback.format_exc())

        raise typer.Exit(1)


class DatasetProcessor:
    """Handles the actual dataset generation workflow."""

    @staticmethod
    def process_single_split(config) -> Any:
        """
        Process single split dataset generation.

        Args:
            config: Configuration object

        Returns:
            Generated DatasetDict

        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Local import for dataset generation
            from graid.data.generate_dataset import generate_dataset

            # Normalize split
            splits = ArgumentValidator.validate_split_format(config.split)
            if len(splits) != 1:
                raise ProcessingError(
                    "Single split processing called with multiple splits"
                )

            result = generate_dataset(
                dataset_name=config.dataset_name,
                split=splits[0],
                models=getattr(config, "models", []),
                use_wbf=getattr(config, "use_wbf", False),
                wbf_config=getattr(config, "wbf_config", None),
                conf_threshold=getattr(config, "confidence_threshold", 0.0),
                batch_size=getattr(config, "batch_size", 32),
                device=getattr(config, "device", None),
                allowable_set=getattr(config, "allowable_set", None),
                question_configs=getattr(config, "question_configs", []),
                num_workers=getattr(config, "num_workers", 4),
                qa_workers=getattr(config, "qa_workers", 4),
                save_path=getattr(config, "save_path", "./graid-datasets"),
                upload_to_hub=getattr(config, "upload_to_hub", False),
                hub_repo_id=getattr(config, "hub_repo_id", None),
                hub_private=getattr(config, "hub_private", False),
                num_samples=getattr(config, "num_samples", None),
                use_original_filenames=getattr(config, "use_original_filenames", True),
                filename_prefix=getattr(config, "filename_prefix", "img"),
            )

            return result

        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(f"Single split processing failed: {e}")

    @staticmethod
    def process_multiple_splits(config) -> Any:
        """
        Process multi-split dataset generation with combined upload.

        Args:
            config: Configuration object

        Returns:
            Combined DatasetDict

        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Local imports
            from datasets import DatasetDict

            splits = ArgumentValidator.validate_split_format(config.split)
            if len(splits) <= 1:
                raise ProcessingError(
                    "Multiple split processing called with single split"
                )

            combined_dict = DatasetDict()

            # Aggregate statistics across all splits
            aggregated_question_counts = {}
            aggregated_detailed_stats = {}

            # Process each split separately
            for split_name in splits:
                logger.info(f"Processing split: {split_name}")

                # Create temporary config for this split
                config_dict = config.to_dict()
                config_dict["split"] = split_name
                config_dict["upload_to_hub"] = False  # Don't upload individual splits

                split_config = config.__class__(**config_dict)

                split_result, split_stats = DatasetProcessor.process_single_split(
                    split_config
                )

                # Add to combined dict
                for key, dataset in split_result.items():
                    combined_dict[key] = dataset

                # Aggregate question statistics
                if split_stats:
                    for qtype, count in split_stats.get("question_counts", {}).items():
                        aggregated_question_counts[qtype] = (
                            aggregated_question_counts.get(qtype, 0) + count
                        )

                    # Aggregate detailed stats (timings, etc.)
                    for qtype, stats in split_stats.get("detailed_stats", {}).items():
                        if qtype not in aggregated_detailed_stats:
                            aggregated_detailed_stats[qtype] = {
                                "is_applicable_time": (0.0, 0),
                                "is_applicable_true_count": 0,
                                "apply_time": (0.0, 0),
                                "apply_empty_results": 0,
                                "total_qa_generated": 0,
                                "question_text": stats.get("question_text", qtype),
                            }

                        # Aggregate timing data
                        agg_stats = aggregated_detailed_stats[qtype]

                        # Add is_applicable times
                        is_app_time, is_app_count = agg_stats["is_applicable_time"]
                        split_is_app_time, split_is_app_count = stats.get(
                            "is_applicable_time", (0.0, 0)
                        )
                        agg_stats["is_applicable_time"] = (
                            is_app_time + split_is_app_time,
                            is_app_count + split_is_app_count,
                        )

                        # Add apply times
                        apply_time, apply_count = agg_stats["apply_time"]
                        split_apply_time, split_apply_count = stats.get(
                            "apply_time", (0.0, 0)
                        )
                        agg_stats["apply_time"] = (
                            apply_time + split_apply_time,
                            apply_count + split_apply_count,
                        )

                        # Add other counters
                        agg_stats["is_applicable_true_count"] += stats.get(
                            "is_applicable_true_count", 0
                        )
                        agg_stats["apply_empty_results"] += stats.get(
                            "apply_empty_results", 0
                        )
                        agg_stats["total_qa_generated"] += stats.get(
                            "total_qa_generated", 0
                        )

            # Prepare aggregated statistics for README
            question_stats = (
                {
                    "question_counts": aggregated_question_counts,
                    "detailed_stats": aggregated_detailed_stats,
                }
                if aggregated_question_counts
                else None
            )

            # Log aggregated profiling statistics for multi-split processing
            if question_stats and "detailed_stats" in question_stats:
                from graid.utils.profiling import log_profiling_statistics

                log_profiling_statistics(
                    question_stats,
                    "Multi-Split Aggregated Question Processing Statistics",
                )
                logger.info("Notes: Aggregated across all processed splits")

            # Persist README with profiling information for later uploads
            try:
                save_dir = Path(getattr(config, "save_path", "./graid-datasets"))
                safe_mkdir(save_dir, description="save directory")

                # If we have stats, generate and persist README and stats JSON
                if question_stats:
                    readme_content = DatasetProcessor._create_dataset_readme(
                        combined_dict, config, question_stats
                    )

                    readme_disk_path = save_dir / "README.md"
                    readme_disk_path.write_text(readme_content, encoding="utf-8")
                    logger.info(
                        f"📝 Saved README with profiling to disk: {readme_disk_path}"
                    )

                    # Persist raw stats for future regeneration if needed
                    stats_disk_path = save_dir / "graid_profiling.json"
                    with stats_disk_path.open("w", encoding="utf-8") as f:
                        json.dump(question_stats, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"💾 Saved profiling statistics JSON: {stats_disk_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to persist README/stats for reuse: {e}")

            # Handle combined upload if requested
            if config.upload_to_hub and config.hub_repo_id:
                DatasetProcessor.handle_hub_upload(
                    combined_dict, config, question_stats
                )

                # Clean up image files after successful multi-split upload
                # DatasetProcessor._cleanup_multi_split_images(config.save_path)

            return combined_dict

        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(f"Multiple split processing failed: {e}")

    @staticmethod
    def _cleanup_multi_split_images(save_path: str) -> None:
        """Clean up temporary image files after multi-split processing."""
        try:
            import shutil
            from pathlib import Path

            save_path_obj = Path(save_path)

            # Clean up all split image directories
            for split_dir in save_path_obj.iterdir():
                if split_dir.is_dir():
                    images_dir = split_dir / "images"
                    if images_dir.exists():
                        logger.debug(f"Cleaning up images directory: {images_dir}")
                        shutil.rmtree(images_dir)
                        logger.debug(f"✅ Cleaned up {images_dir}")

            logger.info("✅ Multi-split image cleanup completed successfully")

        except Exception as e:
            logger.warning(f"Failed to cleanup multi-split image files: {e}")

    @staticmethod
    def handle_hub_upload(dataset_dict: Any, config, question_stats=None) -> None:
        """
        Handle HuggingFace Hub upload workflow with comprehensive README.

        Args:
            dataset_dict: DatasetDict to upload
            config: Configuration object
            question_stats: Optional aggregated question statistics

        Raises:
            ProcessingError: If upload fails
        """
        try:
            # Local import for Hub utilities
            from pathlib import Path

            from huggingface_hub import create_repo, upload_file

            logger.info(f"Uploading to HuggingFace Hub: {config.hub_repo_id}")

            # Create repository
            create_repo(
                config.hub_repo_id,
                repo_type="dataset",
                private=config.hub_private,
                exist_ok=True,
            )

            # Determine README source: prefer freshly generated, else persisted
            persisted_readme_path = None
            try:
                save_dir = Path(getattr(config, "save_path", "./graid-datasets"))
                persisted_readme_candidate = save_dir / "README.md"
                if persisted_readme_candidate.exists():
                    persisted_readme_path = persisted_readme_candidate
            except Exception:
                persisted_readme_path = None

            readme_path_to_upload = None

            if question_stats:
                # Generate fresh README and persist it for future runs
                readme_content = DatasetProcessor._create_dataset_readme(
                    dataset_dict, config, question_stats
                )
                try:
                    save_dir = Path(getattr(config, "save_path", "./graid-datasets"))
                    safe_mkdir(save_dir, description="save directory")
                    disk_readme = save_dir / "README.md"
                    disk_readme.write_text(readme_content, encoding="utf-8")
                    readme_path_to_upload = disk_readme

                    # Persist stats JSON as well
                    stats_disk_path = save_dir / "graid_profiling.json"
                    with stats_disk_path.open("w", encoding="utf-8") as f:
                        json.dump(question_stats, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"📝 Generated and saved README + profiling stats to {save_dir}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to persist README/stats to disk, using temp file: {e}"
                    )
                    # Fallback: temp README in CWD
                    tmp_path = Path("README.md")
                    tmp_path.write_text(readme_content, encoding="utf-8")
                    readme_path_to_upload = tmp_path
            else:
                # No fresh stats - try to regenerate from persisted stats JSON
                if not persisted_readme_path:
                    try:
                        save_dir = Path(getattr(config, "save_path", "./graid-datasets"))
                        stats_disk_path = save_dir / "graid_profiling.json"
                        if stats_disk_path.exists():
                            with stats_disk_path.open("r", encoding="utf-8") as f:
                                persisted_stats = json.load(f)
                            readme_content = DatasetProcessor._create_dataset_readme(
                                dataset_dict, config, persisted_stats
                            )
                            disk_readme = save_dir / "README.md"
                            disk_readme.write_text(readme_content, encoding="utf-8")
                            persisted_readme_path = disk_readme
                            logger.info(
                                f"📝 Reconstructed README from persisted stats: {disk_readme}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to reconstruct README from persisted stats: {e}"
                        )

                # Use persisted README if available
                if persisted_readme_path:
                    readme_path_to_upload = persisted_readme_path

            # Upload README if we have a path ready
            if readme_path_to_upload and readme_path_to_upload.exists():
                upload_file(
                    path_or_fileobj=str(readme_path_to_upload),
                    path_in_repo="README.md",
                    repo_id=config.hub_repo_id,
                    repo_type="dataset",
                    commit_message="Add comprehensive README with GRAID statistics",
                )
                logger.info("📄 README uploaded successfully")

            # Push dataset with large shard size to avoid 10k file limit
            dataset_dict.push_to_hub(
                repo_id=config.hub_repo_id,
                private=config.hub_private,
                # embed_external_files=False,
                commit_message=f"Upload {config.dataset_name} dataset",
                max_shard_size="5GB",  # Large shards to minimize file count
            )

            logger.info(f"Successfully uploaded to Hub: {config.hub_repo_id}")

            # Note: Using CC BY-NC 4.0 license - no custom license file upload needed
            # Commercial licensing information is included in the README

        except Exception as e:
            raise ProcessingError(f"Hub upload failed: {e}")

    @staticmethod
    def _create_dataset_readme(dataset_dict, config, question_stats):
        """
        Generate comprehensive README content for HuggingFace Hub.

        Args:
            dataset_dict: DatasetDict with the generated dataset
            config: Configuration object
            question_stats: Dictionary with aggregated statistics

        Returns:
            str: Complete README content in markdown format
        """
        from datetime import datetime

        # Calculate total QA pairs across all splits
        total_qa_pairs = sum(len(dataset_dict[split]) for split in dataset_dict.keys())

        # Dataset-specific configuration
        dataset_configs = {
            "bdd": {
                "full_name": "BDD100K",
                "description": "Berkeley DeepDrive autonomous driving dataset",
                "license": "bsd-3-clause",
                "tags": ["autonomous-driving", "bdd100k"],
                "source_info": "BDD100K (Berkeley DeepDrive)",
                "citation": """@INPROCEEDINGS{9156329,
    author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning}, 
    year={2020},
    volume={},
    number={},
    pages={2633-2642},
    keywords={Task analysis;Visualization;Roads;Image segmentation;Meteorology;Training;Benchmark testing},
    doi={10.1109/CVPR42600.2020.00271}
}""",
                "license_text": "This dataset is derived from the BDD100K dataset. Please refer to the [BDD100K license terms](https://github.com/bdd100k/bdd100k) for usage restrictions.",
            },
            "waymo": {
                "full_name": "Waymo Open Dataset",
                "description": "Waymo autonomous driving dataset",
                "license": "other",
                "tags": ["autonomous-driving", "waymo"],
                "source_info": "Waymo Open Dataset",
                "citation": """@inproceedings{waymo,
    title={Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={2446--2454},
    year={2020}
}""",
                "license_text": "This dataset is derived from the Waymo Open Dataset. Please refer to the [Waymo Open Dataset license terms](https://waymo.com/open/terms/) for usage restrictions.",
            },
            "nuimage": {
                "full_name": "NuImages",
                "description": "Large-scale autonomous driving dataset from nuTonomy",
                "license": "other",
                "tags": ["autonomous-driving", "nuimages"],
                "source_info": "NuImages Dataset",
                "citation": """@InProceedings{Caesar_2020_CVPR,
    author = {Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}""",
                "license_text": "This dataset is derived from the nuImages dataset. Please refer to the [nuImages license terms](https://www.nuscenes.org/terms-of-use) for usage restrictions.",
            },
            "custom": {
                "full_name": "Custom Dataset",
                "description": "User-provided custom dataset",
                "license": "other",
                "tags": ["custom-dataset"],
                "source_info": "Custom user dataset",
                "citation": """@dataset{graid_custom,
    title={GRAID Generated Question-Answer Dataset},
    author={GRAID Framework},
    year={2025},
    note={Generated using GRAID framework with custom dataset}
}""",
                "license_text": "Please refer to your original dataset's license terms for usage restrictions.",
            },
        }

        # Get dataset config, default to custom if not found
        dataset_name = config.dataset_name.lower()
        dataset_config = dataset_configs.get(dataset_name, dataset_configs["custom"])

        readme_content = f"""---
pretty_name: "GRAID {dataset_config['full_name']} Question-Answer Dataset"
language:
- en
license: "cc-by-nc-4.0"
task_categories:
- visual-question-answering
- object-detection
tags:
- visual-reasoning
- spatial-reasoning
- object-detection
- computer-vision"""

        # Add dataset-specific tags
        for tag in dataset_config["tags"]:
            readme_content += f"\n- {tag}"

        readme_content += f"""
---

# GRAID {dataset_config['full_name']} Question-Answer Dataset

## Overview

This dataset was generated using **GRAID** (**G**enerating **R**easoning questions from **A**nalysis of **I**mages via **D**iscriminative artificial intelligence), a framework for creating spatial reasoning datasets from object detection annotations.

**GRAID** transforms raw object detection data into structured question-answer pairs that test various aspects of object localization, visual reasoning, spatial reasoning, and object relationship comprehension.

## Dataset Details

"""

        # Add basic dataset information
        readme_content += f"""- **Total QA Pairs**: {total_qa_pairs:,}
- **Source Dataset**: {dataset_config['source_info']}
- **Generation Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Image Format**: Embedded in parquet files (no separate image files)
- **Question Types**: {len(question_stats.get('question_counts', {})) if question_stats else 'Multiple'} different reasoning patterns

## Dataset Splits

"""

        # Add split information
        for split_name in sorted(dataset_dict.keys()):
            split_size = len(dataset_dict[split_name])
            percentage = (split_size / total_qa_pairs) * 100
            readme_content += (
                f"- **{split_name}**: {split_size:,} ({percentage:.2f}%)\n"
            )

        readme_content += "\n## Question Type Distribution\n\n"

        # Add question statistics across all splits
        if question_stats and "question_counts" in question_stats:
            total_questions = sum(question_stats["question_counts"].values())
            sorted_counts = sorted(
                question_stats["question_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for qtype, count in sorted_counts:
                percentage = (count / total_questions) * 100
                # Get full question text if available
                question_text = qtype
                if (
                    "detailed_stats" in question_stats
                    and qtype in question_stats["detailed_stats"]
                ):
                    question_text = question_stats["detailed_stats"][qtype].get(
                        "question_text", qtype
                    )

                readme_content += (
                    f"- **{question_text}**: {count:,} ({percentage:.2f}%)\n"
                )

        # Add performance profiling information if available
        if question_stats and "detailed_stats" in question_stats:
            from graid.utils.profiling import (
                format_profiling_notes,
                format_profiling_table,
            )

            readme_content += "\n## Performance Analysis\n\n"
            readme_content += "### Question Processing Efficiency\n\n"
            readme_content += format_profiling_table(question_stats, "markdown")
            readme_content += format_profiling_notes("markdown")

        # Add usage information
        readme_content += f"""
## Usage

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("{config.hub_repo_id}")

# Access individual splits"""

        for split_name in sorted(dataset_dict.keys()):
            readme_content += f'\n{split_name}_data = dataset["{split_name}"]'

        readme_content += f"""

# Example of accessing a sample
sample = dataset["train"][0]  # or "val"
print(f"Question: {{sample['question']}}")
print(f"Answer: {{sample['answer']}}")  
print(f"Question Type: {{sample['question_type']}}")

# The image is embedded as a PIL Image object
image = sample["image"]
image.show()  # Display the image
```

## Dataset Schema

- **image**: PIL Image object (embedded, no separate files)
- **annotations**: COCO-style bounding box annotations  
- **question**: Generated question text
- **answer**: Corresponding answer text
- **reasoning**: Additional reasoning information (if applicable)
- **question_type**: Type of question (e.g., "HowMany", "LeftOf", "Quadrants")
- **source_id**: Original image identifier from {dataset_config['source_info']}

## License

"""

        readme_content += f"""This generated dataset is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, which permits free use for non-commercial purposes including academic research and education.

**Commercial Use Policy**: Commercial entities (including startups and companies) that wish to use this dataset for commercial purposes must obtain a paid license from **MESH**. The CC BY-NC license prohibits commercial use without explicit permission.

To request a commercial license, please contact **Karim Elmaaroufi**.

**Original Source Compliance**: The original source datasets and their licenses still apply to the underlying images and annotations. You must comply with both the CC BY-NC terms and the source dataset terms:

{dataset_config['license_text']}

## Citation

If you use this dataset in your research, please cite both the original dataset and the GRAID framework:

```bibtex
@dataset{{graid_{dataset_name},
    title={{GRAID {dataset_config['full_name']} Question-Answer Dataset}},
    author={{GRAID Framework}},
    year={{2025}},
    note={{Generated using GRAID: Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence}}
}}

{dataset_config['citation']}
```

## Contact

For questions about this dataset or the GRAID framework, please open an issue in the repository.
"""

        return readme_content


def safe_mkdir(path: Union[str, Path], description: str = "directory") -> Path:
    """
    Safely create directory with proper error handling.

    Args:
        path: Directory path to create
        description: Description for error messages

    Returns:
        Created Path object

    Raises:
        ProcessingError: If directory creation fails
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except PermissionError:
        raise ProcessingError(f"Permission denied creating {description}: {path}")
    except OSError as e:
        raise ProcessingError(f"Failed to create {description} '{path}': {e}")
    except Exception as e:
        raise ProcessingError(f"Unexpected error creating {description} '{path}': {e}")
