"""
Configuration File Support for HuggingFace Dataset Generation

This module provides configuration file support for specifying models and WBF settings
without using CLI arguments. It supports JSON configuration files with validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from graid.data.generate_db import create_model
from graid.utilities.coco import coco_labels
from graid.utilities.common import get_default_device

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class ModelConfig:
    """Configuration for a single model."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        custom_config: Optional[dict[str, Any]] = None,
        confidence_threshold: float = 0.2,
        device: Optional[str] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.custom_config = custom_config
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate the model configuration."""
        # Check if backend is supported
        supported_backends = ["detectron", "mmdetection", "ultralytics"]
        if self.backend not in supported_backends:
            raise ConfigurationError(
                f"Unsupported backend: {self.backend}. Supported: {supported_backends}"
            )

        # For detectron and mmdetection, custom config is required
        if self.backend in ["detectron", "mmdetection"] and self.custom_config is None:
            raise ConfigurationError(
                f"Custom config is required for {self.backend} backend. "
                f"Use create_model() with custom_config parameter."
            )

        # Validate custom config structure
        if self.custom_config:
            self._validate_custom_config()

    def _validate_custom_config(self):
        """Validate custom configuration structure."""
        if self.custom_config is None:
            return

        if self.backend == "detectron":
            if (
                "config" not in self.custom_config
                or "weights" not in self.custom_config
            ):
                raise ConfigurationError(
                    "Detectron custom config must have 'config' and 'weights' keys"
                )
        elif self.backend == "mmdetection":
            if (
                "config" not in self.custom_config
                or "checkpoint" not in self.custom_config
            ):
                raise ConfigurationError(
                    "MMDetection custom config must have 'config' and 'checkpoint' keys"
                )
        elif self.backend == "ultralytics":
            if (
                isinstance(self.custom_config, dict)
                and "model_file" not in self.custom_config
            ):
                raise ConfigurationError(
                    "Ultralytics custom config must have 'model_file' key"
                )

    def create_model(self):
        """Create a model instance from this configuration."""
        device = self.device or get_default_device()

        return create_model(
            backend=self.backend,
            model_name=self.model_name,
            device=device,
            threshold=self.confidence_threshold,
            custom_config=self.custom_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "custom_config": self.custom_config,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
        }


class WBFConfig:
    """Configuration for Weighted Boxes Fusion."""

    def __init__(
        self,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.0,
        model_weights: Optional[list[float]] = None,
    ):
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.model_weights = model_weights

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate WBF configuration."""
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfigurationError(
                f"iou_threshold must be between 0.0 and 1.0, got {self.iou_threshold}"
            )

        if not 0.0 <= self.skip_box_threshold <= 1.0:
            raise ConfigurationError(
                f"skip_box_threshold must be between 0.0 and 1.0, got {self.skip_box_threshold}"
            )

        if self.model_weights is not None:
            if not all(w > 0 for w in self.model_weights):
                raise ConfigurationError("All model weights must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "iou_threshold": self.iou_threshold,
            "skip_box_threshold": self.skip_box_threshold,
            "model_weights": self.model_weights,
        }


class DatasetGenerationConfig:
    """Complete configuration for dataset generation."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        models: list[ModelConfig],
        use_wbf: bool = False,
        wbf_config: Optional[WBFConfig] = None,
        confidence_threshold: float = 0.2,
        batch_size: int = 1,
        device: Optional[str] = None,
        allowable_set: Optional[list[str]] = None,
        question_configs: Optional[list[dict[str, Any]]] = None,
        num_workers: int = 4,
        qa_workers: int = 4,
        save_path: Optional[str] = None,
        upload_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False,
        num_samples: Optional[int] = None,
        use_original_filenames: bool = True,
        filename_prefix: str = "img",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.models = models
        self.use_wbf = use_wbf
        self.wbf_config = wbf_config
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.device = device
        self.allowable_set = allowable_set
        self.question_configs = question_configs
        self.num_workers = num_workers
        self.qa_workers = qa_workers
        self.save_path = save_path
        self.upload_to_hub = upload_to_hub
        self.hub_repo_id = hub_repo_id
        self.hub_private = hub_private
        self.num_samples = num_samples
        self.use_original_filenames = use_original_filenames
        self.filename_prefix = filename_prefix

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate the complete configuration."""
        # Validate dataset name
        supported_datasets = ["bdd", "nuimage", "waymo"]
        if self.dataset_name not in supported_datasets:
            raise ConfigurationError(f"Unsupported dataset: {self.dataset_name}")

        # Validate split - support individual splits and combined splits like "train+val"
        valid_individual_splits = ["train", "val", "test"]
        if "+" in self.split:
            # Handle combined splits like "train+val"
            split_parts = [s.strip() for s in self.split.split("+")]
            for part in split_parts:
                if part not in valid_individual_splits:
                    raise ConfigurationError(
                        f"Invalid split part: {part}. Valid splits: {valid_individual_splits}"
                    )
        else:
            # Handle individual splits
            if self.split not in valid_individual_splits:
                raise ConfigurationError(
                    f"Invalid split: {self.split}. Valid splits: {valid_individual_splits}"
                )

        # Validate models
        if not self.models:
            logger.warning("No models specified, will use ground truth")

        # Validate WBF configuration
        if self.use_wbf:
            if len(self.models) < 2:
                raise ConfigurationError("WBF requires at least 2 models")

            if self.wbf_config is None:
                self.wbf_config = WBFConfig()

            if self.wbf_config.model_weights is not None:
                if len(self.wbf_config.model_weights) != len(self.models):
                    raise ConfigurationError(
                        f"Number of model weights ({len(self.wbf_config.model_weights)}) "
                        f"must match number of models ({len(self.models)})"
                    )

        # Validate Hub configuration
        if self.upload_to_hub and not self.hub_repo_id:
            raise ConfigurationError("hub_repo_id is required when upload_to_hub=True")

        # Validate allowable_set
        if self.allowable_set is not None:
            valid_coco_objects = set(coco_labels.values())
            # Remove undefined as it's not a real COCO class
            valid_coco_objects.discard("undefined")

            invalid_objects = []
            for obj in self.allowable_set:
                if obj not in valid_coco_objects:
                    invalid_objects.append(obj)

            if invalid_objects:
                raise ConfigurationError(
                    f"Invalid COCO objects in allowable_set: {invalid_objects}. "
                    f"Valid objects: {sorted(valid_coco_objects)}"
                )

    def create_models(self):
        """Create model instances from the configuration."""
        return [model.create_model() for model in self.models]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "models": [model.to_dict() for model in self.models],
            "use_wbf": self.use_wbf,
            "wbf_config": self.wbf_config.to_dict() if self.wbf_config else None,
            "confidence_threshold": self.confidence_threshold,
            "batch_size": self.batch_size,
            "device": self.device,
            "allowable_set": self.allowable_set,
            "save_path": self.save_path,
            "upload_to_hub": self.upload_to_hub,
            "hub_repo_id": self.hub_repo_id,
            "hub_private": self.hub_private,
            "question_configs": self.question_configs,
            "num_workers": self.num_workers,
            "qa_workers": self.qa_workers,
            "num_samples": self.num_samples,
            "use_original_filenames": self.use_original_filenames,
            "filename_prefix": self.filename_prefix,
        }


def load_config_from_file(config_path: Union[str, Path]) -> DatasetGenerationConfig:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        DatasetGenerationConfig instance

    Raises:
        ConfigurationError: If the file doesn't exist or is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")

    return load_config_from_dict(config_data)


def load_config_from_dict(config_data: dict[str, Any]) -> DatasetGenerationConfig:
    """
    Load configuration from dictionary.

    Args:
        config_data: Configuration dictionary

    Returns:
        DatasetGenerationConfig instance

    Raises:
        ConfigurationError: If the configuration is invalid
    """
    try:
        # Parse model configurations
        models = []
        for model_data in config_data.get("models", []):
            models.append(
                ModelConfig(
                    backend=model_data["backend"],
                    model_name=model_data["model_name"],
                    custom_config=model_data.get("custom_config"),
                    confidence_threshold=model_data.get("confidence_threshold", 0.2),
                    device=model_data.get("device"),
                )
            )

        # Parse WBF configuration
        wbf_config = None
        if config_data.get("wbf_config"):
            wbf_data = config_data["wbf_config"]
            wbf_config = WBFConfig(
                iou_threshold=wbf_data.get("iou_threshold", 0.55),
                skip_box_threshold=wbf_data.get("skip_box_threshold", 0.0),
                model_weights=wbf_data.get("model_weights"),
            )

        # Create main configuration
        return DatasetGenerationConfig(
            dataset_name=config_data["dataset_name"],
            split=config_data["split"],
            models=models,
            use_wbf=config_data.get("use_wbf", False),
            wbf_config=wbf_config,
            confidence_threshold=config_data.get("confidence_threshold", 0.2),
            batch_size=config_data.get("batch_size", 1),
            device=config_data.get("device"),
            allowable_set=config_data.get("allowable_set"),
            question_configs=config_data.get("question_configs"),
            num_workers=config_data.get("num_workers", 4),
            qa_workers=config_data.get("qa_workers", 4),
            save_path=config_data.get("save_path"),
            upload_to_hub=config_data.get("upload_to_hub", False),
            hub_repo_id=config_data.get("hub_repo_id"),
            hub_private=config_data.get("hub_private", False),
            num_samples=config_data.get("num_samples"),
            use_original_filenames=config_data.get("use_original_filenames", True),
            filename_prefix=config_data.get("filename_prefix", "img"),
        )

    except KeyError as e:
        raise ConfigurationError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error parsing configuration: {e}")


def create_example_config() -> dict[str, Any]:
    """Create an example configuration dictionary."""
    return {
        "dataset_name": "bdd",
        "split": "val",
        "models": [
            {
                "backend": "ultralytics",
                "model_name": "yolov8x",
                "confidence_threshold": 0.3,
            },
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "confidence_threshold": 0.2,
            },
            {
                "backend": "mmdetection",
                "model_name": "co_detr",
                "confidence_threshold": 0.25,
            },
        ],
        "use_wbf": True,
        "wbf_config": {
            "iou_threshold": 0.55,
            "skip_box_threshold": 0.0,
            "model_weights": [1.0, 1.0, 1.0],
        },
        "confidence_threshold": 0.2,
        "batch_size": 4,
        "allowable_set": ["person", "car", "truck", "bus", "bicycle", "motorcycle"],
        "save_path": "./my_dataset",
        "upload_to_hub": False,
        "hub_repo_id": "username/my-dataset",
        "hub_private": True,
    }


def save_example_config(output_path: Union[str, Path]):
    """Save an example configuration file."""
    config = create_example_config()
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Example configuration saved to {output_path}")


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        config = load_config_from_file(config_path)
        return True, None
    except ConfigurationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"
