import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from graid.data.generate_db import DATASET_TRANSFORMS, create_model
from graid.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from graid.models.Detectron import Detectron_obj
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import RT_DETR, Yolo
from graid.models.WBF import WBF
from graid.questions.ObjectDetectionQ import (
    ALL_QUESTIONS,
    AreMore,
    HowMany,
    IsObjectCentered,
    LargestAppearance,
    LeastAppearance,
    LeftMost,
    LeftMostWidthVsHeight,
    LeftOf,
    MostAppearance,
    MostClusteredObjects,
    Quadrants,
    RightMost,
    RightMostWidthVsHeight,
    RightOf,
    WhichMore,
    WidthVsHeight,
)
from graid.utilities.coco import validate_coco_objects
from graid.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

logger = logging.getLogger(__name__)


def bdd_transform(i, l):
    return yolo_bdd_transform(i, l, new_shape=(768, 1280))


def nuimage_transform(i, l):
    return yolo_nuscene_transform(i, l, new_shape=(896, 1600))


def waymo_transform(i, l):
    return yolo_waymo_transform(i, l, (1280, 1920))


DATASET_TRANSFORMS = {
    "bdd": bdd_transform,
    "nuimage": nuimage_transform,
    "waymo": waymo_transform,
}

# GRAID supports any model from the supported backends
# Users can provide custom configurations for detectron and mmdetection
# or use any available model file for ultralytics


class HuggingFaceDatasetBuilder:
    """Builder class for generating HuggingFace datasets from object detection models."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        models: Optional[list[Any]] = None,
        model_configs: Optional[list[dict[str, Any]]] = None,
        use_wbf: bool = False,
        wbf_config: Optional[dict[str, Any]] = None,
        conf_threshold: float = 0.2,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        allowable_set: Optional[list[str]] = None,
        selected_questions: Optional[list[str]] = None,
        question_configs: Optional[list[dict[str, Any]]] = None,
        custom_transforms: Optional[dict[str, Any]] = None,
    ):
        """Initialize the HuggingFace dataset builder."""
        self.dataset_name = dataset_name
        self.split = split
        self.models = models or []
        self.model_configs = model_configs or []
        self.use_wbf = use_wbf
        self.wbf_config = wbf_config or {}
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()

        # Validate and set allowable_set
        if allowable_set is not None:
            is_valid, error_msg = validate_coco_objects(allowable_set)
            if not is_valid:
                raise ValueError(f"Invalid allowable_set: {error_msg}")
        self.allowable_set = allowable_set

        # Initialize wbf_ensemble to None
        self.wbf_ensemble = None

        # Handle custom transforms
        if custom_transforms:
            self.transform = self._create_custom_transform(custom_transforms)
        else:
            if dataset_name not in DATASET_TRANSFORMS:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            self.transform = DATASET_TRANSFORMS[dataset_name]

        # Handle question configuration
        if question_configs is not None:
            self.questions = self._create_questions_from_config(question_configs)
        elif selected_questions is not None:
            # Map question names to actual question objects
            available_questions = {q.__class__.__name__: q for q in ALL_QUESTIONS}
            self.questions = []
            for question_name in selected_questions:
                if question_name in available_questions:
                    self.questions.append(available_questions[question_name])
                else:
                    logger.warning(f"Unknown question type: {question_name}")

            if not self.questions:
                raise ValueError("No valid questions selected")
        else:
            self.questions = ALL_QUESTIONS

        # Initialize dataset loader
        self._init_dataset_loader()

        # Prepare model ensemble if using WBF
        if self.use_wbf and self.models:
            self._prepare_wbf_ensemble()

    def _create_custom_transform(self, custom_transforms: dict[str, Any]) -> Any:
        """Create a custom transform function from configuration."""
        transform_type = custom_transforms.get("type", "yolo")
        new_shape = custom_transforms.get("new_shape", (640, 640))

        if transform_type == "yolo_bdd":

            def custom_transform(i, l):
                return yolo_bdd_transform(i, l, new_shape=new_shape)

        elif transform_type == "yolo_nuscene":

            def custom_transform(i, l):
                return yolo_nuscene_transform(i, l, new_shape=new_shape)

        elif transform_type == "yolo_waymo":

            def custom_transform(i, l):
                return yolo_waymo_transform(i, l, new_shape=new_shape)

        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

        return custom_transform

    def _create_questions_from_config(
        self, question_configs: list[dict[str, Any]]
    ) -> list[Any]:
        """Create question objects from configuration."""
        questions = []

        for config in question_configs:
            question_name = config.get("name")
            question_params = config.get("params", {})

            if question_name == "IsObjectCentered":
                questions.append(IsObjectCentered())
            elif question_name == "WidthVsHeight":
                threshold = question_params.get("threshold", 0.30)
                questions.append(WidthVsHeight(threshold=threshold))
            elif question_name == "LargestAppearance":
                threshold = question_params.get("threshold", 0.3)
                questions.append(LargestAppearance(threshold=threshold))
            elif question_name == "MostAppearance":
                questions.append(MostAppearance())
            elif question_name == "LeastAppearance":
                questions.append(LeastAppearance())
            elif question_name == "LeftOf":
                questions.append(LeftOf())
            elif question_name == "RightOf":
                questions.append(RightOf())
            elif question_name == "LeftMost":
                questions.append(LeftMost())
            elif question_name == "RightMost":
                questions.append(RightMost())
            elif question_name == "HowMany":
                questions.append(HowMany())
            elif question_name == "MostClusteredObjects":
                threshold = question_params.get("threshold", 100)
                questions.append(MostClusteredObjects(threshold=threshold))
            elif question_name == "WhichMore":
                questions.append(WhichMore())
            elif question_name == "AreMore":
                questions.append(AreMore())
            elif question_name == "Quadrants":
                N = question_params.get("N", 2)
                M = question_params.get("M", 2)
                questions.append(Quadrants(N, M))
            elif question_name == "LeftMostWidthVsHeight":
                threshold = question_params.get("threshold", 0.3)
                questions.append(LeftMostWidthVsHeight(threshold=threshold))
            elif question_name == "RightMostWidthVsHeight":
                threshold = question_params.get("threshold", 0.3)
                questions.append(RightMostWidthVsHeight(threshold=threshold))
            else:
                logger.warning(f"Unknown question type: {question_name}")

        if not questions:
            raise ValueError("No valid questions configured")

        return questions

    def _init_dataset_loader(self):
        """Initialize the appropriate dataset loader."""
        try:
            if self.dataset_name == "bdd":
                self.dataset_loader = Bdd100kDataset(
                    split=self.split, transform=self.transform
                )  # type: ignore
            elif self.dataset_name == "nuimage":
                self.dataset_loader = NuImagesDataset(
                    split=self.split, size="all", transform=self.transform
                )  # type: ignore
            elif self.dataset_name == "waymo":
                split_name = "validation" if self.split == "val" else self.split + "ing"
                self.dataset_loader = WaymoDataset(
                    split=split_name, transform=self.transform
                )  # type: ignore
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        except Exception as e:
            logger.error(f"Failed to initialize dataset loader: {e}")
            raise

    def _prepare_wbf_ensemble(self):
        """Prepare WBF ensemble from individual models."""
        if not self.models:
            return

        # Import WBF here to avoid circular imports
        from graid.models.Detectron import Detectron_obj
        from graid.models.MMDetection import MMdetection_obj
        from graid.models.Ultralytics import RT_DETR, Yolo
        from graid.models.WBF import WBF

        # Group models by backend
        detectron_models = []
        mmdet_models = []
        ultralytics_models = []

        for model in self.models:
            if isinstance(model, Detectron_obj):
                detectron_models.append(model)
            elif isinstance(model, MMdetection_obj):
                mmdet_models.append(model)
            elif isinstance(model, (Yolo, RT_DETR)):
                ultralytics_models.append(model)

        # Create WBF ensemble
        self.wbf_ensemble = WBF(
            detectron2_models=detectron_models if detectron_models else None,
            mmdet_models=mmdet_models if mmdet_models else None,
            ultralytics_models=ultralytics_models if ultralytics_models else None,
            **self.wbf_config,
        )

    def _convert_image_to_pil(
        self, image: Union[torch.Tensor, np.ndarray]
    ) -> Image.Image:
        """Convert tensor or numpy array to PIL Image."""
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            if image.dim() == 3:  # (C, H, W)
                image = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 4:  # (B, C, H, W)
                image = image[0].permute(1, 2, 0).cpu().numpy()

        # Ensure proper data type and range
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return Image.fromarray(image)

    def _create_metadata(self) -> dict[str, Any]:
        """Create metadata dictionary for the dataset."""
        metadata = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "confidence_threshold": self.conf_threshold,
            "batch_size": self.batch_size,
            "use_wbf": self.use_wbf,
            "questions": [str(q.__class__.__name__) for q in self.questions],
            "models": [],
        }

        # Only include device info when not using WBF (single device usage)
        if not self.use_wbf:
            metadata["device"] = str(self.device)
        else:
            metadata["device_info"] = "Multiple devices may be used in WBF ensemble"

        # Add model information
        if self.models:
            for i, model in enumerate(self.models):
                model_info = {
                    "backend": model.__class__.__module__.split(".")[-1],
                    "model_name": getattr(
                        model, "model_name", str(model.__class__.__name__)
                    ),
                    "config": (
                        self.model_configs[i] if i < len(self.model_configs) else None
                    ),
                }
                metadata["models"].append(model_info)
        else:
            metadata["models"] = [{"type": "ground_truth"}]

        return metadata

    def build(self) -> DatasetDict:
        """Build the HuggingFace dataset."""
        logger.info(
            f"Building HuggingFace dataset for {self.dataset_name} {self.split}"
        )

        # For now, create a simple placeholder dataset
        # This will be expanded with full functionality
        results = []

        # Process a small subset to demonstrate structure
        data_loader = DataLoader(
            self.dataset_loader,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=1,
        )

        for base_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            if base_idx >= 10:  # Limit to 10 batches for demonstration
                break

            # Handle different dataset return formats
            if isinstance(batch[0], tuple):
                # Tuple format (BDD dataset)
                batch_images = torch.stack([sample[0] for sample in batch])
                ground_truth_labels = [sample[1] for sample in batch]
            else:
                # Dictionary format (NuImages/Waymo datasets)
                batch_images = torch.stack([sample["image"] for sample in batch])
                ground_truth_labels = [sample["labels"] for sample in batch]

            # Get predictions from model(s)
            if self.use_wbf and hasattr(self, "wbf_ensemble"):
                batch_images = batch_images.to(self.device)
                labels = self.wbf_ensemble.identify_for_image_batch(batch_images)
            elif self.models:
                batch_images = batch_images.to(self.device)
                # Use first model if multiple models without WBF
                model = self.models[0]
                labels = model.identify_for_image_batch(batch_images)
            else:
                # Use ground truth
                labels = ground_truth_labels

            # Process each image in the batch
            for j, (image_tensor, detections) in enumerate(zip(batch_images, labels)):
                # Convert to PIL Image
                pil_image = self._convert_image_to_pil(image_tensor)

                # Filter detections by confidence threshold
                if detections:
                    detections = [
                        d for d in detections if d.score >= self.conf_threshold
                    ]

                # Filter detections by allowable set if specified
                if detections and self.allowable_set:
                    filtered_detections = []
                    for detection in detections:
                        if detection.label in self.allowable_set:
                            filtered_detections.append(detection)
                        else:
                            logger.debug(
                                f"Filtered out detection of class '{detection.label}' (not in allowable set)"
                            )
                    detections = filtered_detections

                # Extract bounding boxes
                bboxes = []
                if detections:
                    for detection in detections:
                        bbox = detection.as_xyxy().squeeze().tolist()
                        bboxes.append(
                            {
                                "bbox": bbox,
                                "label": detection.label,
                                "score": float(detection.score),
                                "class_id": int(detection.cls),
                            }
                        )

                # Generate questions and answers
                for question in self.questions:
                    if detections and question.is_applicable(pil_image, detections):
                        qa_pairs = question.apply(pil_image, detections)

                        for question_text, answer_text in qa_pairs:
                            results.append(
                                {
                                    "image": pil_image,
                                    "question": question_text,
                                    "answer": answer_text,
                                    "bboxes": bboxes,
                                    "image_id": f"{base_idx + j}",
                                    "question_type": str(question.__class__.__name__),
                                    "num_detections": (
                                        len(detections) if detections else 0
                                    ),
                                }
                            )

        if not results:
            logger.warning("No question-answer pairs generated!")
            # Create a minimal example
            results = [
                {
                    "image": Image.new("RGB", (224, 224)),
                    "question": "How many objects are there?",
                    "answer": "0",
                    "bboxes": [],
                    "image_id": "0",
                    "question_type": "HowMany",
                    "num_detections": 0,
                }
            ]

        # Create HuggingFace dataset
        dataset = Dataset.from_list(results)

        # Add metadata info
        metadata = self._create_metadata()
        dataset.info.description = (
            f"Object detection QA dataset for {self.dataset_name}"
        )
        dataset.info.features = dataset.features
        # Store metadata in the dataset info
        dataset.info.version = metadata

        # Create DatasetDict
        dataset_dict = DatasetDict({self.split: dataset})

        logger.info(f"Generated {len(dataset)} question-answer pairs")
        return dataset_dict


def generate_dataset(
    dataset_name: str,
    split: str,
    models: Optional[list[Any]] = None,
    model_configs: Optional[list[dict[str, Any]]] = None,
    use_wbf: bool = False,
    wbf_config: Optional[dict[str, Any]] = None,
    conf_threshold: float = 0.2,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    allowable_set: Optional[list[str]] = None,
    selected_questions: Optional[list[str]] = None,
    question_configs: Optional[list[dict[str, Any]]] = None,
    custom_transforms: Optional[dict[str, Any]] = None,
    save_path: Optional[str] = None,
    upload_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
) -> DatasetDict:
    """Generate a HuggingFace dataset for object detection question-answering."""

    # Create dataset builder
    builder = HuggingFaceDatasetBuilder(
        dataset_name=dataset_name,
        split=split,
        models=models,
        model_configs=model_configs,
        use_wbf=use_wbf,
        wbf_config=wbf_config,
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        device=device,
        allowable_set=allowable_set,
        selected_questions=selected_questions,
        question_configs=question_configs,
        custom_transforms=custom_transforms,
    )

    # Build the dataset
    dataset_dict = builder.build()

    # Save locally if requested
    if save_path:
        dataset_dict.save_to_disk(save_path)
        logger.info(f"Dataset saved to {save_path}")

    # Upload to HuggingFace Hub if requested
    if upload_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when upload_to_hub=True")

        dataset_dict.push_to_hub(
            repo_id=hub_repo_id,
            private=hub_private,
            commit_message=f"Upload {dataset_name} {split} dataset",
        )
        logger.info(f"Dataset uploaded to HuggingFace Hub: {hub_repo_id}")

    return dataset_dict


def validate_model_config(
    backend: str,
    model_name: str,
    config: Optional[dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate that a model configuration can be loaded and used.

    Args:
        backend: Model backend (detectron, mmdetection, ultralytics)
        model_name: Name of the model
        config: Optional custom configuration
        device: Device to test on

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Set device
        if device is None:
            device = get_default_device()

        logger.info(f"Validating {backend} model: {model_name}")

        # Create and test the model
        model = create_model(backend, model_name, device, 0.2)

        # Basic validation - check if model can be moved to device
        model.to(device)

        # Test with a dummy input to ensure model is functional
        if hasattr(model, "identify_for_image_batch"):
            try:
                # Create a dummy batch of images (batch_size=1, channels=3, height=224, width=224)
                dummy_images = torch.rand(1, 3, 224, 224, device=device)

                # Test inference
                _ = model.identify_for_image_batch(dummy_images)
                logger.info(f"✓ {backend} model {model_name} validated successfully")
                return True, None

            except Exception as inference_error:
                error_msg = f"Model inference test failed: {str(inference_error)}"
                logger.error(error_msg)
                return False, error_msg
        else:
            # If no identify_for_image_batch method, assume basic validation passed
            logger.info(f"✓ {backend} model {model_name} basic validation passed")
            return True, None

    except ImportError as e:
        error_msg = f"Import error for {backend}: {str(e)}. Make sure the required dependencies are installed."
        logger.error(error_msg)
        return False, error_msg
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}. Check the model path or download the model."
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Model validation failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def validate_models_batch(
    model_configs: list[dict[str, Any]],
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, tuple[bool, Optional[str]]]:
    """
    Validate multiple model configurations in batch.

    Args:
        model_configs: List of model configuration dictionaries
        device: Device to test on

    Returns:
        Dictionary mapping model identifiers to (is_valid, error_message) tuples
    """
    results = {}

    for i, config in enumerate(model_configs):
        model_id = f"{config['backend']}_{config['model_name']}_{i}"

        try:
            is_valid, error_msg = validate_model_config(
                backend=config["backend"],
                model_name=config["model_name"],
                config=config.get("custom_config"),
                device=device,
            )
            results[model_id] = (is_valid, error_msg)

        except Exception as e:
            results[model_id] = (False, f"Validation error: {str(e)}")

    return results


def validate_wbf_compatibility(
    model_configs: list[dict[str, Any]],
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate that models are compatible for WBF ensemble.

    Args:
        model_configs: List of model configuration dictionaries
        device: Device to test on

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(model_configs) < 2:
        return False, "WBF requires at least 2 models"

    # Validate individual models first
    validation_results = validate_models_batch(model_configs, device)

    failed_models = []
    for model_id, (is_valid, error_msg) in validation_results.items():
        if not is_valid:
            failed_models.append(f"{model_id}: {error_msg}")

    if failed_models:
        return False, f"Some models failed validation: {'; '.join(failed_models)}"

    # Check backend compatibility
    supported_backends = {"detectron", "mmdetection", "ultralytics"}
    model_backends = set(config["backend"] for config in model_configs)

    unsupported_backends = model_backends - supported_backends
    if unsupported_backends:
        return False, f"Unsupported backends for WBF: {unsupported_backends}"

    # Test that models can be grouped properly
    try:
        # Create temporary models to test grouping
        models = []
        for config in model_configs:
            model = create_model(
                config["backend"],
                config["model_name"],
                device,
                config.get("confidence_threshold", 0.2),
            )
            models.append(model)

        # Test WBF ensemble creation
        detectron_models = [m for m in models if isinstance(m, Detectron_obj)]
        mmdet_models = [m for m in models if isinstance(m, MMdetection_obj)]
        ultralytics_models = [m for m in models if isinstance(m, (Yolo, RT_DETR))]

        # Create WBF ensemble
        wbf_ensemble = WBF(
            detectron2_models=detectron_models if detectron_models else None,
            mmdet_models=mmdet_models if mmdet_models else None,
            ultralytics_models=ultralytics_models if ultralytics_models else None,
        )

        # Test with dummy input
        dummy_images = torch.rand(1, 3, 224, 224, device=device)
        _ = wbf_ensemble.identify_for_image_batch(dummy_images)

        logger.info("✓ WBF ensemble validation passed")
        return True, None

    except Exception as e:
        error_msg = f"WBF ensemble validation failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load model configuration from JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def list_available_models() -> dict[str, list[str]]:
    """List supported backends and example models."""
    return {
        "detectron": [
            "Custom models via config file - provide config and weights paths"
        ],
        "mmdetection": [
            "Custom models via config file - provide config and checkpoint paths"
        ],
        "ultralytics": [
            "yolov8x.pt",
            "yolov10x.pt",
            "yolo11x.pt",
            "rtdetr-x.pt",
            "Any YOLOv8/YOLOv10/YOLOv11/RT-DETR model file or custom trained model",
        ],
    }


def list_available_questions() -> dict[str, dict[str, Any]]:
    """List available question types, their descriptions, and parameters."""
    question_info = {}

    for q in ALL_QUESTIONS:
        question_name = q.__class__.__name__
        question_text = getattr(q, "question", str(q.__class__.__name__))

        # Determine parameters for each question type
        params = {}
        if question_name == "WidthVsHeight":
            params = {
                "threshold": {
                    "type": "float",
                    "default": 0.30,
                    "description": "Threshold for width vs height comparison",
                }
            }
        elif question_name == "LargestAppearance":
            params = {
                "threshold": {
                    "type": "float",
                    "default": 0.3,
                    "description": "Threshold for largest appearance comparison",
                }
            }
        elif question_name == "MostClusteredObjects":
            params = {
                "threshold": {
                    "type": "int",
                    "default": 100,
                    "description": "Distance threshold for clustering",
                }
            }
        elif question_name == "Quadrants":
            params = {
                "N": {
                    "type": "int",
                    "default": 2,
                    "description": "Number of rows in grid",
                },
                "M": {
                    "type": "int",
                    "default": 2,
                    "description": "Number of columns in grid",
                },
            }
        elif question_name == "LeftMostWidthVsHeight":
            params = {
                "threshold": {
                    "type": "float",
                    "default": 0.3,
                    "description": "Threshold for width vs height comparison",
                }
            }
        elif question_name == "RightMostWidthVsHeight":
            params = {
                "threshold": {
                    "type": "float",
                    "default": 0.3,
                    "description": "Threshold for width vs height comparison",
                }
            }

        question_info[question_name] = {"question": question_text, "parameters": params}

    return question_info


def interactive_question_selection() -> list[dict[str, Any]]:
    """Interactive question selection with parameter configuration."""
    print("\n📋 Question Selection")
    print("=" * 50)

    available_questions = list_available_questions()
    question_configs = []

    print("Available questions:")
    question_names = list(available_questions.keys())
    for i, name in enumerate(question_names, 1):
        info = available_questions[name]
        print(f"  {i}. {name}")
        print(f"     {info['question']}")
        if info["parameters"]:
            params_str = ", ".join(
                f"{k}={v['default']}" for k, v in info["parameters"].items()
            )
            print(f"     Parameters: {params_str}")
        print()

    print("Enter question numbers (comma-separated) or 'all' for all questions:")

    while True:
        try:
            selection = input("Selection: ").strip()

            if selection.lower() == "all":
                # Add all questions with default parameters
                for name, info in available_questions.items():
                    params = {}
                    for param_name, param_info in info["parameters"].items():
                        params[param_name] = param_info["default"]
                    question_configs.append({"name": name, "params": params})
                break

            # Parse comma-separated numbers
            selected_indices = []
            for part in selection.split(","):
                part = part.strip()
                if part:
                    idx = int(part) - 1
                    if 0 <= idx < len(question_names):
                        selected_indices.append(idx)
                    else:
                        print(f"Invalid selection: {part}")
                        continue

            if not selected_indices:
                print("No valid selections made. Please try again.")
                continue

            # Configure selected questions
            for idx in selected_indices:
                name = question_names[idx]
                info = available_questions[name]
                params = {}

                print(f"\n⚙️  Configuring {name}")
                print(f"Question: {info['question']}")

                # Configure parameters
                for param_name, param_info in info["parameters"].items():
                    while True:
                        try:
                            default_val = param_info["default"]
                            param_type = param_info["type"]
                            description = param_info["description"]

                            user_input = input(
                                f"{param_name} ({description}, default: {default_val}): "
                            ).strip()

                            if not user_input:
                                # Use default
                                params[param_name] = default_val
                                break

                            if param_type == "int":
                                params[param_name] = int(user_input)
                            elif param_type == "float":
                                params[param_name] = float(user_input)
                            else:
                                params[param_name] = user_input
                            break
                        except ValueError:
                            print(
                                f"Invalid input for {param_name}. Expected {param_type}."
                            )

                question_configs.append({"name": name, "params": params})

            break

        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return []

    return question_configs
