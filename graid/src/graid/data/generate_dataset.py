"""
GRAID HuggingFace Dataset Generation

This module provides comprehensive functionality for generating HuggingFace datasets 
from object detection data, supporting multiple model backends, ensemble methods,
and flexible question-answer generation patterns.

Key Features:
    - Multi-backend support: Detectron2, MMDetection, Ultralytics
    - Weighted Box Fusion (WBF) ensemble methods
    - Parallel question-answer generation
    - COCO-style annotations with embedded PIL images
    - Unlabeled image support (model-generated detections)
    - Robust checkpointing and crash recovery
    - HuggingFace Hub integration

Classes:
    HuggingFaceDatasetBuilder: Main dataset generation engine
    QABatchProcessor: Abstract strategy for QA processing
    SequentialQAProcessor: Sequential QA generation strategy
    ParallelQAProcessor: Parallel QA generation with ThreadPoolExecutor
    QAProcessorFactory: Factory for creating QA processing strategies

Functions:
    generate_dataset: High-level API for dataset generation
    list_available_questions: Query available question types
    interactive_question_selection: Interactive question configuration
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from graid.utilities.common import get_default_device

logger = logging.getLogger(__name__)


class QABatchProcessor(ABC):
    """
    Abstract strategy for processing question-answer generation in batches.

    This class defines the interface for different QA processing strategies,
    allowing flexible switching between sequential and parallel processing
    approaches based on performance requirements and resource constraints.

    The strategy pattern enables:
        - Sequential processing for memory-limited environments
        - Parallel processing for high-throughput scenarios
        - Easy extension with new processing strategies
    """

    @abstractmethod
    def process_batch(
        self, batch_data: List[Tuple[Image.Image, List[Any], str, int, int]]
    ) -> List[Any]:
        """
        Process a batch of image data and generate question-answer pairs.

        This method takes prepared batch data and applies question generation
        algorithms to produce structured QA pairs with optional timing information.

        Args:
            batch_data: List of tuples containing:
                - pil_image (PIL.Image.Image): Processed image
                - detections (List[Detection]): Object detection results
                - source_id (str): Unique identifier for the image
                - base_image_index (int): Starting index for this batch
                - j (int): Position within the batch

        Returns:
            List of QA results where each element is either:
                - List[Dict[str, Any]]: QA pairs (when profiling disabled)
                - Tuple[List[Dict[str, Any]], Dict[str, tuple[float, int]]]:
                  QA pairs with timing data (when profiling enabled)

        Raises:
            NotImplementedError: If called on abstract base class
        """
        pass


class SequentialQAProcessor(QABatchProcessor):
    """
    Sequential question-answer processing strategy.

    This implementation processes images one by one in a single thread,
    providing predictable memory usage and easier debugging at the cost
    of processing speed. Ideal for:
        - Memory-constrained environments
        - Debugging and development
        - Small batch sizes
        - Systems with limited CPU cores

    Attributes:
        qa_generator: Reference to the dataset builder instance
        profile_questions: Whether to collect timing statistics
    """

    def __init__(self, qa_generator, profile_questions: bool):
        """
        Initialize sequential QA processor.

        Args:
            qa_generator: The HuggingFaceDatasetBuilder instance that contains
                the question generation logic and configuration
            profile_questions: Whether to enable timing profiling for
                performance analysis
        """
        self.qa_generator = qa_generator
        self.profile_questions = profile_questions
        logger.debug(
            "✓ Initialized SequentialQAProcessor with profiling=%s", profile_questions
        )

    def process_batch(
        self, batch_data: List[Tuple[Image.Image, List[Any], str, int, int]]
    ) -> List[Any]:
        """
        Process QA generation sequentially for all images in the batch.

        Args:
            batch_data: List of prepared image data tuples

        Returns:
            List of QA results maintaining input order
        """
        logger.debug("🔄 Processing batch of %d images sequentially", len(batch_data))
        results = []

        for i, args in enumerate(batch_data):
            pil_image, detections, source_id, base_image_index, j = args
            image_index = base_image_index + j

            # Set current example for filename inference
            self.qa_generator._current_example = {"name": source_id}

            try:
                ret = self.qa_generator._qa_for_image(
                    pil_image, detections, source_id, image_index
                )
                results.append(ret)
                logger.debug(
                    "✓ Processed image %d/%d: %s", i + 1, len(batch_data), source_id
                )
            except Exception as e:
                logger.error("❌ Failed to process image %s: %s", source_id, e)
                # Add empty result to maintain order
                empty_result = ([], {}) if self.profile_questions else []
                results.append(empty_result)

        logger.debug(
            "✅ Sequential batch processing completed: %d results", len(results)
        )
        return results


class ParallelQAProcessor(QABatchProcessor):
    """
    Parallel question-answer processing strategy using ThreadPoolExecutor.

    This implementation processes multiple images concurrently using a thread pool,
    providing significant speedup for I/O-bound question generation tasks.
    Uses ThreadPoolExecutor.map() to maintain result ordering. Ideal for:
        - High-throughput scenarios
        - Systems with multiple CPU cores
        - I/O-bound question generation
        - Large batch processing

    Note:
        Maintains strict ordering through executor.map() to ensure
        QA results correspond to input images correctly.

    Attributes:
        qa_generator: Reference to the dataset builder instance
        qa_workers: Number of parallel worker threads
        profile_questions: Whether to collect timing statistics
    """

    def __init__(self, qa_generator, qa_workers: int, profile_questions: bool):
        """
        Initialize parallel QA processor.

        Args:
            qa_generator: The HuggingFaceDatasetBuilder instance containing
                the thread-safe question generation logic
            qa_workers: Number of parallel worker threads to spawn.
                Recommended: 2-4x CPU cores for I/O-bound tasks
            profile_questions: Whether to enable timing profiling for
                performance analysis
        """
        self.qa_generator = qa_generator
        self.qa_workers = qa_workers
        self.profile_questions = profile_questions
        logger.debug(
            "✓ Initialized ParallelQAProcessor with %d workers, profiling=%s",
            qa_workers,
            profile_questions,
        )

    def process_batch(
        self, batch_data: List[Tuple[Image.Image, List[Any], str, int, int]]
    ) -> List[Any]:
        """
        Process QA generation in parallel with strict order preservation.

        Uses ThreadPoolExecutor.map() which maintains the order of results
        corresponding to the input batch_data order, ensuring QA pairs
        match their source images correctly.

        Args:
            batch_data: List of prepared image data tuples

        Returns:
            List of QA results in the same order as input batch_data
        """
        logger.debug(
            "🚀 Processing batch of %d images with %d parallel workers",
            len(batch_data),
            self.qa_workers,
        )

        with ThreadPoolExecutor(max_workers=self.qa_workers) as executor:
            results = list(
                executor.map(self.qa_generator._qa_for_image_threadsafe, batch_data)
            )

        logger.debug("✅ Parallel batch processing completed: %d results", len(results))
        return results


class QAProcessorFactory:
    """
    Factory for creating appropriate QA processing strategies.

    This factory implements the Strategy pattern by selecting the optimal
    QA processing approach based on configuration parameters. The selection
    logic considers performance requirements, resource constraints, and
    system capabilities.

    Strategy Selection Rules:
        - qa_workers = 1: Sequential processing (safe, predictable)
        - qa_workers > 1: Parallel processing (high throughput)
    """

    @staticmethod
    def create(
        qa_workers: int, qa_generator, profile_questions: bool
    ) -> QABatchProcessor:
        """
        Create the appropriate QA processing strategy based on configuration.

        Automatically selects between sequential and parallel processing
        strategies based on the number of workers requested. This enables
        transparent optimization without changing client code.

        Args:
            qa_workers: Number of QA worker threads to use:
                - 1: Creates SequentialQAProcessor for single-threaded processing
                - >1: Creates ParallelQAProcessor with specified worker count
            qa_generator: The HuggingFaceDatasetBuilder instance that provides
                the question generation logic and configuration
            profile_questions: Whether to enable performance profiling and
                timing collection for analysis

        Returns:
            QABatchProcessor: Configured strategy instance ready for processing

        Example:
            >>> # Single-threaded for debugging
            >>> processor = QAProcessorFactory.create(1, builder, True)
            >>>
            >>> # Multi-threaded for production
            >>> processor = QAProcessorFactory.create(8, builder, False)
        """
        if qa_workers > 1:
            logger.info("🚀 Creating ParallelQAProcessor with %d workers", qa_workers)
            return ParallelQAProcessor(qa_generator, qa_workers, profile_questions)
        else:
            logger.info(
                "🔄 Creating SequentialQAProcessor for single-threaded processing"
            )
            return SequentialQAProcessor(qa_generator, profile_questions)


class HuggingFaceDatasetBuilder:
    """
    Advanced HuggingFace dataset builder for object detection question-answering.

    This class orchestrates the complete pipeline for generating high-quality VQA datasets
    from object detection data. It supports multiple detection backends, ensemble methods,
    parallel processing, and produces datasets compatible with modern vision-language models.

    Key Capabilities:
        🎯 Multi-Backend Support: Detectron2, MMDetection, Ultralytics models
        🔗 Ensemble Methods: Weighted Box Fusion (WBF) for improved accuracy
        🚀 Parallel Processing: Configurable worker threads for QA generation
        📊 COCO Compatibility: Standard annotations with category strings
        🖼️ PIL Integration: Embedded images ready for VLM workflows
        📁 Flexible Storage: Original or generated filenames
        🔄 Crash Recovery: Robust checkpointing for long-running jobs
        🌐 Hub Integration: Direct upload to HuggingFace Hub

    Architecture:
        The builder uses the Strategy pattern for QA processing, Factory pattern
        for dataset loading, and incremental dataset construction to handle
        large-scale data generation efficiently.

    Workflow:
        1. Initialize models and configure processing parameters
        2. Load and transform source dataset (BDD100K, NuImages, Waymo, Custom)
        3. Apply object detection (ensemble via WBF or single model)
        4. Generate question-answer pairs using parallel/sequential strategies
        5. Build incremental HuggingFace datasets with embedded PIL images
        6. Optional: Upload to HuggingFace Hub with metadata

    Performance Optimizations:
        - Batch processing with configurable sizes
        - Parallel QA generation with ThreadPoolExecutor
        - Incremental dataset building to manage memory
        - Optional checkpointing for crash recovery
        - Confidence thresholds for quality control

    Example:
        >>> builder = HuggingFaceDatasetBuilder(
        ...     dataset_name="bdd",
        ...     split="val",
        ...     models=[yolo_model, detectron_model],
        ...     use_wbf=True,
        ...     qa_workers=8,
        ...     num_samples=1000
        ... )
        >>> dataset_dict = builder.build()
        >>> print(f"Generated {len(dataset_dict['val'])} QA pairs")
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        models: Optional[List[Any]] = None,
        use_wbf: bool = False,
        wbf_config: Optional[Dict[str, Any]] = None,
        conf_threshold: float = 0.2,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        allowable_set: Optional[List[str]] = None,
        question_configs: Optional[List[Dict[str, Any]]] = None,
        num_workers: int = 4,
        qa_workers: int = 4,
        num_samples: Optional[int] = None,
        save_steps: int = 50,
        use_original_filenames: bool = True,
        filename_prefix: str = "img",
        force: bool = False,
        save_path: str = "./graid-datasets",
    ):
        """
        Initialize the HuggingFace dataset builder.

        Args:
            dataset_name: Name of the dataset ("bdd", "nuimage", "waymo")
            split: Dataset split ("train", "val", "test")
            models: List of model objects for inference (optional)
            use_wbf: Whether to use Weighted Box Fusion ensemble
            wbf_config: Configuration for WBF ensemble (optional)
            conf_threshold: Confidence threshold for filtering detections
            batch_size: Batch size for processing
            device: Device to use for inference (optional)
            allowable_set: List of allowed object classes (optional)
            question_configs: List of question configuration dictionaries (optional)
            num_workers: Number of data loading workers
            qa_workers: Number of QA generation workers
            num_samples: Maximum number of samples to process (0 or None = process all)
            save_steps: Save checkpoint every N batches for crash recovery
            save_path: Path to save dataset (required)
            use_original_filenames: Whether to keep original filenames
            filename_prefix: Prefix for generated filenames if not using originals
            force: Force restart from scratch, ignoring existing checkpoints
        """
        self.dataset_name = dataset_name
        self.split = split
        self.models = models or []
        self.use_wbf = use_wbf
        self.wbf_config = wbf_config or {}
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()
        self.allowable_set = allowable_set
        self.num_workers = num_workers
        self.qa_workers = qa_workers
        self.num_samples = num_samples
        self.save_steps = save_steps
        self.save_path = Path(save_path)
        self.use_original_filenames = use_original_filenames
        self.filename_prefix = filename_prefix
        self.force = force

        # Question profiling (timings)
        self.profile_questions: bool = bool(os.getenv("GRAID_PROFILE_QUESTIONS"))
        self.question_timings: Dict[str, Tuple[float, int]] = {}
        self.question_counts: Dict[str, int] = {}

        # Checkpointing support
        self.checkpoint_dir = self.save_path / "checkpoints"
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.split}.json"

        # Validate allowable_set
        if allowable_set is not None:
            from graid.utilities.coco import validate_coco_objects

            is_valid, error_msg = validate_coco_objects(allowable_set)
            if not is_valid:
                raise ValueError(f"Invalid allowable_set: {error_msg}")

        # Initialize dataset transforms
        self.transform = self._get_dataset_transform()

        # Initialize questions
        self.questions = self._initialize_questions(question_configs)

        # Initialize dataset loader
        self._init_dataset_loader()

        # Note: No longer creating image directories - using embedded images in parquet

        # Prepare WBF ensemble if needed
        self.wbf_ensemble = None
        if self.use_wbf and self.models:
            self._prepare_wbf_ensemble()

    def _get_dataset_transform(self):
        """Get the appropriate transform for the dataset."""
        from graid.utilities.common import (
            yolo_bdd_transform,
            yolo_nuscene_transform,
            yolo_waymo_transform,
        )

        if self.dataset_name == "bdd":
            return lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
        elif self.dataset_name == "nuimage":
            return lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600))
        elif self.dataset_name == "waymo":
            return lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _initialize_questions(
        self, question_configs: Optional[List[Dict[str, Any]]]
    ) -> List[Any]:
        """Initialize question objects from configuration."""
        if question_configs is None:
            # Use all available questions
            from graid.questions.ObjectDetectionQ import ALL_QUESTION_CLASSES

            return list(ALL_QUESTION_CLASSES.values())

        questions = []
        from graid.questions.ObjectDetectionQ import (
            IsObjectCentered,
            WidthVsHeight,
            LargestAppearance,
            RankLargestK,
            MostAppearance,
            LeastAppearance,
            LeftOf,
            RightOf,
            LeftMost,
            RightMost,
            HowMany,
            MostClusteredObjects,
            WhichMore,
            AreMore,
            Quadrants,
            LeftMostWidthVsHeight,
            RightMostWidthVsHeight,
            ObjectsInRow,
            ObjectsInLine,
            MoreThanThresholdHowMany,
            LessThanThresholdHowMany,
            MultiChoiceHowMany,
        )

        # Map question names to classes
        question_class_map = {
            "IsObjectCentered": IsObjectCentered,
            "WidthVsHeight": WidthVsHeight,
            "LargestAppearance": LargestAppearance,
            "RankLargestK": RankLargestK,
            "MostAppearance": MostAppearance,
            "LeastAppearance": LeastAppearance,
            "LeftOf": LeftOf,
            "RightOf": RightOf,
            "LeftMost": LeftMost,
            "RightMost": RightMost,
            "HowMany": HowMany,
            "MostClusteredObjects": MostClusteredObjects,
            "WhichMore": WhichMore,
            "AreMore": AreMore,
            "Quadrants": Quadrants,
            "LeftMostWidthVsHeight": LeftMostWidthVsHeight,
            "RightMostWidthVsHeight": RightMostWidthVsHeight,
            "ObjectsInRow": ObjectsInRow,
            "ObjectsInLine": ObjectsInLine,
            "MoreThanThresholdHowMany": MoreThanThresholdHowMany,
            "LessThanThresholdHowMany": LessThanThresholdHowMany,
            "MultiChoiceHowMany": MultiChoiceHowMany,
        }

        for config in question_configs:
            question_name = config.get("name")
            question_params = config.get("params", {})

            if question_name not in question_class_map:
                logger.warning(f"Unknown question type: {question_name}")
                continue

            question_class = question_class_map[question_name]

            # Handle questions that require parameters
            if question_params:
                try:
                    question_instance = question_class(**question_params)
                except Exception as e:
                    logger.error(
                        f"Failed to initialize {question_name} with params {question_params}: {e}"
                    )
                    # Fall back to default initialization
                    question_instance = question_class()
            else:
                question_instance = question_class()

            questions.append(question_instance)

        if not questions:
            raise ValueError("No valid questions configured")

        return questions

    def _init_dataset_loader(self):
        """Initialize the appropriate dataset loader using the common factory."""
        from graid.data.loaders import DatasetLoaderFactory

        try:
            self.dataset_loader = DatasetLoaderFactory.create(
                dataset_name=self.dataset_name,
                split=self.split,
                transform=self.transform,
            )
        except Exception as e:
            logger.error(f"Failed to initialize dataset loader: {e}")
            raise

    def _prepare_wbf_ensemble(self):
        """Prepare WBF ensemble from individual models."""
        # Import WBF classes locally
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

    def _infer_source_name(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract source filename from dataset example."""
        if isinstance(example, dict) and "name" in example:
            return example["name"]
        return None

    def _generate_filename(self, index: int, source_name: Optional[str]) -> str:
        """Generate filename based on configuration."""
        if self.use_original_filenames and source_name:
            return Path(source_name).name
        return f"{self.filename_prefix}{index:06d}.jpg"

    def _convert_image_to_pil(
        self, image: Union[torch.Tensor, np.ndarray]
    ) -> Image.Image:
        """Convert tensor or numpy array to PIL Image."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # (C, H, W)
                image = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 4:  # (B, C, H, W)
                image = image[0].permute(1, 2, 0).cpu().numpy()

        # Ensure proper data type and range
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return Image.fromarray(image)

    def _build_coco_annotations(
        self, detections: List[Any], image_width: int, image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Build COCO-style annotations from detections.

        Args:
            detections: List of detection objects
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of COCO annotation dictionaries
        """
        annotations = []

        for detection in detections:
            # Get bounding box in XYWH format
            xywh = detection.as_xywh()[0]
            x, y, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])

            # Build COCO annotation
            annotation = {
                "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
                "category_id": 1,  # Default category ID
                "category": detection.label,  # Add category string
                "iscrowd": 0,
                "area": float(w * h),
                "score": float(detection.score) if hasattr(detection, "score") else 1.0,
            }
            annotations.append(annotation)

        return annotations

    def _qa_for_image(
        self,
        pil_image: Image.Image,
        detections: List[Any],
        source_id: str,
        image_index: int,
    ) -> Union[
        List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, tuple[float, int]]]
    ]:
        """Generate question-answer pairs for a single image with embedded image bytes."""
        qa_pairs = []
        local_timings: Dict[str, tuple[float, int]] = (
            {} if self.profile_questions else {}
        )

        # Ensure image is in RGB format for consistency
        rgb_img = (
            pil_image if pil_image.mode in ("RGB", "L") else pil_image.convert("RGB")
        )

        # SOLUTION: Embed image bytes directly instead of saving separate files
        # This solves HuggingFace 10k file limit by storing images in parquet
        # No compression - preserve original format or store as uncompressed PNG
        import io
        
        # Try to preserve original format from source_id extension
        _, ext = os.path.splitext(source_id)
        original_format = ext.upper().lstrip('.') if ext else 'PNG'
        
        # Map common extensions to PIL formats
        format_map = {'JPG': 'JPEG', 'JPEG': 'JPEG', 'PNG': 'PNG', 'BMP': 'BMP', 'TIFF': 'TIFF'}
        pil_format = format_map.get(original_format, 'PNG')  # Default to PNG if unknown
        
        buffer = io.BytesIO()
        if pil_format == 'JPEG':
            # For JPEG, save without additional compression (quality=100)
            rgb_img.save(buffer, format=pil_format, quality=100, optimize=False)
        elif pil_format == 'PNG':
            # For PNG, save without compression
            rgb_img.save(buffer, format=pil_format, compress_level=0, optimize=False)
        else:
            # For other formats, save as-is
            rgb_img.save(buffer, format=pil_format)
        
        image_bytes = buffer.getvalue()
        
        # Store image as bytes with original format info
        image_reference = {"bytes": image_bytes, "path": None}

        # Generate COCO annotations
        annotations = self._build_coco_annotations(
            detections, pil_image.width, pil_image.height
        )

        # Generate questions and answers
        for question in self.questions:
            if detections and question.is_applicable(pil_image, detections):
                t0 = time.perf_counter() if self.profile_questions else None
                try:
                    qa_results = question.apply(pil_image, detections)
                    if self.profile_questions and t0 is not None:
                        dt = time.perf_counter() - t0
                        qname = question.__class__.__name__
                        t_total, t_cnt = local_timings.get(qname, (0.0, 0))
                        local_timings[qname] = (t_total + dt, t_cnt + 1)

                    for qa_item in qa_results:
                        if not isinstance(qa_item, (tuple, list)) or len(qa_item) != 2:
                            logger.warning(
                                f"{question.__class__.__name__}.apply() returned malformed item: {qa_item!r}"
                            )
                            continue

                        question_text, answer_text = qa_item

                        # Build the final QA pair with embedded image bytes
                        qa_pair = {
                            "image": image_reference,  # Embedded bytes dict format
                            "annotations": annotations,
                            "question": question_text,
                            "answer": answer_text,
                            "reasoning": None,
                            "question_type": question.__class__.__name__,
                            "source_id": source_id,
                        }

                        # Add source_filename if using generated filenames for reference
                        if not self.use_original_filenames:
                            source_name = (
                                self._infer_source_name({"name": source_id})
                                if hasattr(self, "_current_example")
                                else None
                            )
                            if source_name:
                                qa_pair["source_filename"] = source_name

                        qa_pairs.append(qa_pair)

                except Exception as e:
                    logger.warning(
                        f"Question {question.__class__.__name__} failed on image {source_id}: {e}"
                    )
                    continue

        if self.profile_questions:
            return (qa_pairs, local_timings)
        return qa_pairs

    def _qa_for_image_threadsafe(
        self, batch_args: tuple
    ) -> Union[
        List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, tuple[float, int]]]
    ]:
        """Thread-safe wrapper for _qa_for_image using source_id for uniqueness."""
        pil_image, detections, source_id, base_image_index, batch_j = batch_args

        # Use source_id + batch_j for unique identification (no magic numbers)
        unique_image_key = f"{source_id}_{batch_j}"

        try:
            return self._qa_for_image(
                pil_image, detections, source_id, base_image_index + batch_j
            )
        except Exception as e:
            logger.error(f"Error in threaded QA generation for {unique_image_key}: {e}")
            # Return appropriate empty result based on profiling mode
            return ([], {}) if self.profile_questions else []

    def _save_checkpoint(
        self, batch_idx: int, results: List[Dict[str, Any]], processed_images: int
    ):
        """Save checkpoint to resume from crash."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "batch_idx": batch_idx,
            "processed_images": processed_images,
            "num_results": len(results),
            "dataset_name": self.dataset_name,
            "split": self.split,
            "timestamp": time.time(),
        }

        # Save checkpoint metadata
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save results so far
        results_file = self.checkpoint_dir / f"results_{self.split}.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        logger.info(
            f"Checkpoint saved at batch {batch_idx} ({processed_images} images processed)"
        )

    def _load_checkpoint(self) -> tuple[int, List[Dict[str, Any]], int]:
        """Load checkpoint to resume from crash. Returns (start_batch_idx, results, processed_images)."""
        if not self.checkpoint_file.exists():
            return 0, [], 0

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            results_file = self.checkpoint_dir / f"results_{self.split}.json"
            if not results_file.exists():
                logger.warning(
                    "Checkpoint metadata found but results file missing. Starting from scratch."
                )
                return 0, [], 0

            with open(results_file, "r") as f:
                results = json.load(f)

            start_batch = checkpoint_data["batch_idx"] + 1  # Resume from next batch
            processed_images = checkpoint_data["processed_images"]

            from datasets import Dataset

            checkpoint_dataset = Dataset.from_list(results)

            logger.info(
                f"Resuming from checkpoint: batch {start_batch}, {processed_images} images processed, {len(results)} QA pairs"
            )
            return start_batch, [checkpoint_dataset], processed_images

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            return 0, [], 0

    def _cleanup_checkpoint(self):
        """Clean up checkpoint files after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            results_file = self.checkpoint_dir / f"results_{self.split}.json"
            if results_file.exists():
                results_file.unlink()
            # Remove checkpoint dir if empty
            if self.checkpoint_dir.exists() and not any(self.checkpoint_dir.iterdir()):
                self.checkpoint_dir.rmdir()
            logger.debug("Checkpoint files cleaned up")
        except Exception as e:
            logger.debug(f"Failed to cleanup checkpoint files: {e}")

    def _cleanup_images(self):
        """Clean up image files after successful dataset creation to avoid duplicate storage."""
        if not self.save_path:
            return

        images_dir = self.save_path / self.split / "images"
        if images_dir.exists():
            import shutil

            logger.info(
                f"🧹 Cleaning up image files in {images_dir} (images are embedded in Parquet)"
            )
            shutil.rmtree(images_dir)
            logger.debug(f"✅ Removed images directory: {images_dir}")

            # Remove split directory if it's now empty
            split_dir = self.save_path / self.split
            if split_dir.exists() and not any(split_dir.iterdir()):
                split_dir.rmdir()
                logger.debug(f"✅ Removed empty split directory: {split_dir}")

    def _create_data_loader(self) -> DataLoader:
        """Create and configure the PyTorch DataLoader."""
        return DataLoader(
            self.dataset_loader,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=self.num_workers,
            prefetch_factor=1,
            persistent_workers=False,
        )

    def _initialize_processing_state(self) -> tuple[int, List, int]:
        """Initialize or resume processing state from checkpoints."""
        force_restart = bool(os.getenv("GRAID_FORCE_RESTART")) or self.force
        if force_restart:
            logger.info(
                "Force restart requested - removing existing checkpoints and starting from scratch"
            )
            self._cleanup_checkpoint()
            return 0, [], 0
        else:
            return self._load_checkpoint()

    def _should_skip_batch(self, batch_idx: int, start_batch_idx: int) -> bool:
        """Check if batch should be skipped (for checkpoint resume)."""
        return batch_idx < start_batch_idx

    def _should_stop_early(self, batch_idx: int, processed_images: int) -> bool:
        """Check if processing should stop early due to limits."""
        # Check max_batches environment variable
        try:
            max_batches_env = os.getenv("GRAID_MAX_BATCHES")
            max_batches = int(max_batches_env) if max_batches_env else None
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(
                    f"Stopping early after {batch_idx} batches (GRAID_MAX_BATCHES={max_batches})"
                )
                return True
        except Exception:
            pass

        # Check num_samples limit
        if (
            self.num_samples is not None
            and self.num_samples > 0
            and processed_images >= int(self.num_samples)
        ):
            logger.info(
                f"Reached num_samples={self.num_samples}. Stopping further processing."
            )
            return True

        return False

    def _calculate_total_batches(self, data_loader: DataLoader) -> Optional[int]:
        """Calculate total number of batches considering early stopping."""
        total_batches = len(data_loader)

        # Adjust for num_samples limit
        if self.num_samples is not None and self.num_samples > 0:
            max_batches_for_samples = (
                self.num_samples + self.batch_size - 1
            ) // self.batch_size
            total_batches = min(total_batches, max_batches_for_samples)

        # Adjust for GRAID_MAX_BATCHES environment variable
        try:
            max_batches_env = os.getenv("GRAID_MAX_BATCHES")
            if max_batches_env:
                max_batches = int(max_batches_env)
                total_batches = min(total_batches, max_batches)
        except Exception:
            pass

        return total_batches

    def _get_batch_predictions(
        self, batch: List[Any]
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Extract images and predictions from batch data."""
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
        if self.use_wbf and self.wbf_ensemble is not None:
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

        return batch_images, labels

    def _prepare_batch_data(
        self,
        batch_idx: int,
        batch: List[Any],
        batch_images: torch.Tensor,
        labels: List[Any],
    ) -> List[Tuple[Image.Image, List[Any], str, int, int]]:
        """Prepare batch data for QA processing."""
        batch_data = []

        # Prepare data for processing (parallel or sequential)
        base_image_index = batch_idx * self.batch_size
        
        for j, (image_tensor, detections) in enumerate(zip(batch_images, labels)):
            # Convert to PIL Image
            pil_image = self._convert_image_to_pil(image_tensor)

            # Filter detections by confidence threshold
            if detections:
                detections = [d for d in detections if d.score >= self.conf_threshold]

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

            # Extract source_id from batch sample
            if isinstance(batch[j], dict) and "name" in batch[j]:
                source_id = batch[j]["name"]
            else:
                source_id = f"{self.dataset_name}_{batch_idx}_{j}"

            # Store current example for filename inference
            self._current_example = (
                batch[j] if isinstance(batch[j], dict) else {"name": source_id}
            )

            # Add this image to batch data for processing
            batch_data.append((pil_image, detections, source_id, base_image_index, j))

        return batch_data

    def _process_qa_results(
        self, batch_results_raw: List[Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, int]]]:
        """Process raw QA results and extract timings."""
        batch_results: List[Dict[str, Any]] = []
        batch_timings: Dict[str, Tuple[float, int]] = (
            {} if self.profile_questions else {}
        )

        # Process results and collect timings
        for ret in batch_results_raw:
            if self.profile_questions and isinstance(ret, tuple) and len(ret) == 2:
                qa_pairs, local_timings = ret
                if isinstance(qa_pairs, list):
                    batch_results.extend(qa_pairs)
                if isinstance(local_timings, dict):
                    for k, (t, n) in local_timings.items():
                        T, N = batch_timings.get(k, (0.0, 0))
                        batch_timings[k] = (T + t, N + n)
            elif isinstance(ret, list):
                batch_results.extend(ret)
            else:
                logger.warning(
                    f"Unexpected return type from QA processing: {type(ret)}"
                )

        return batch_results, batch_timings

    def _create_batch_dataset(
        self, batch_results: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Create a Dataset from batch results with deferred image casting."""
        from datasets import Dataset

        if not batch_results:
            return None

        try:
            logger.debug(f"Creating batch dataset from {len(batch_results)} results...")
            batch_dataset = Dataset.from_list(batch_results)
            logger.debug(f"✓ Created batch dataset with {len(batch_dataset)} rows")
            # Note: We deliberately do NOT cast image column here - defer until the very end
            return batch_dataset
        except Exception as e:
            logger.error(f"❌ Failed to create batch dataset: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _update_progress_tracking(
        self,
        batch_results: List[Dict[str, Any]],
        batch_timings: Dict[str, Tuple[float, int]],
    ):
        """Update question counts and timings tracking."""
        # Update per-question counts
        for item in batch_results:
            try:
                qtype = item.get("question_type")
                if qtype:
                    self.question_counts[qtype] = self.question_counts.get(qtype, 0) + 1
            except Exception:
                pass

        # Merge batch timings into builder-level aggregation
        if self.profile_questions and batch_timings:
            for k, (t, n) in batch_timings.items():
                T, N = self.question_timings.get(k, (0.0, 0))
                self.question_timings[k] = (T + t, N + n)

    def _log_progress(self, batch_idx: int, processed_images: int, total_qa_pairs: int):
        """Log progress every 10 batches."""
        if batch_idx % 10 == 0:
            logger.info(
                f"Processed {processed_images} images, generated {total_qa_pairs} QA pairs"
            )

    def _create_final_dataset(self, batch_datasets: List) -> Any:
        """Combine batch datasets into final DatasetDict with metadata."""
        from datasets import (
            Dataset,
            DatasetDict,
            Image as HFImage,
            concatenate_datasets,
        )

        if not batch_datasets:
            logger.warning("No batch datasets created - no QA pairs generated")
            # Create empty dataset with proper schema
            empty_data = {
                "image": [],
                "annotations": [],
                "question": [],
                "answer": [],
                "question_type": [],
                "source_id": [],
            }
            dataset = Dataset.from_dict(empty_data)
            dataset = dataset.cast_column("image", HFImage())
        else:
            # Concatenate all batch datasets
            try:
                logger.info(f"Concatenating {len(batch_datasets)} batch datasets...")
                dataset = concatenate_datasets(batch_datasets)
                logger.debug(f"Final concatenated dataset: {len(dataset)} rows")

                # Cast image column from paths to HFImage at the very end (memory optimization)
                logger.debug(
                    "🎯 Converting image paths to HFImage format at the end..."
                )
                dataset = dataset.cast_column("image", HFImage())

            except Exception as e:
                logger.error(f"Failed to concatenate batch datasets: {e}")
                raise

        # Add metadata
        metadata = self._create_metadata()
        dataset.info.description = (
            f"Object detection QA dataset for {self.dataset_name}"
        )
        dataset.info.features = dataset.features
        # dataset.info.version = "1.0.0"
        dataset.info.config_name = json.dumps(metadata)

        # Create DatasetDict
        dataset_dict = DatasetDict({self.split: dataset})

        logger.info(f"Generated {len(dataset)} question-answer pairs")

        # Clean up checkpoint files on successful completion
        self._cleanup_checkpoint()

        # Log profiling information
        if self.profile_questions and self.question_timings:
            items = [
                (k, t / max(n, 1), n) for k, (t, n) in self.question_timings.items()
            ]
            items.sort(key=lambda x: x[1], reverse=True)
            top = ", ".join(
                [f"{k}: avg {avg:.4f}s over {n}" for k, avg, n in items[:5]]
            )
            logger.info(f"[PROFILE] Top slow questions (avg): {top}")

        # Log per-question counts
        if self.question_counts:
            pairs = sorted( # by question type, most frequent first
                self.question_counts.items(), key=lambda kv: kv[1], reverse=True
            )
            summary = ", ".join([f"{k}={v}" for k, v in pairs])
            logger.info(f"Per-question counts: {summary}")

        return dataset_dict

    def build(self):
        """
        Build the HuggingFace dataset using clean architecture with extracted methods.

        This method orchestrates the complete dataset generation pipeline:
        1. Setup data loaders and processing strategies
        2. Initialize or resume from checkpoints
        3. Process batches with progress tracking
        4. Generate QA pairs using configured strategy
        5. Build incremental datasets and combine
        6. Return final DatasetDict with metadata

        Returns:
            DatasetDict containing the generated VQA dataset
        """
        logger.info(
            "🚀 Building HuggingFace dataset for %s/%s", self.dataset_name, self.split
        )

        # Setup phase
        logger.debug("📋 Initializing data loader and processing components")
        data_loader = self._create_data_loader()
        start_batch_idx, batch_datasets, processed_images = (
            self._initialize_processing_state()
        )
        qa_processor = QAProcessorFactory.create(
            self.qa_workers, self, self.profile_questions
        )

        # Calculate total batches for accurate progress bar
        total_batches = self._calculate_total_batches(data_loader)
        logger.info(
            "📊 Processing %d total batches (%d images per batch)",
            total_batches,
            self.batch_size,
        )

        # Skip already processed batches if resuming
        if start_batch_idx > 0:
            logger.info(
                "⏭️  Resuming from checkpoint: skipping first %d batches",
                start_batch_idx,
            )

        # Processing phase with accurate progress bar
        logger.debug(
            "🔄 Starting batch processing with %s strategy",
            "parallel" if self.qa_workers > 1 else "sequential",
        )
        progress_bar = tqdm(
            enumerate(data_loader), desc="Processing batches", total=total_batches
        )

        for batch_idx, batch in progress_bar:
            # Skip and continue logic
            if self._should_skip_batch(batch_idx, start_batch_idx):
                continue
            if self._should_stop_early(batch_idx, processed_images):
                break

            # Get predictions and prepare batch data
            batch_images, labels = self._get_batch_predictions(batch)
            batch_data = self._prepare_batch_data(
                batch_idx, batch, batch_images, labels
            )

            # Process QA using strategy pattern
            batch_results_raw = qa_processor.process_batch(batch_data)

            # Process results and update tracking
            batch_results, batch_timings = self._process_qa_results(batch_results_raw)
            self._update_progress_tracking(batch_results, batch_timings)

            # Create batch dataset and add to collection
            batch_dataset = self._create_batch_dataset(batch_results)
            if batch_dataset:
                batch_datasets.append(batch_dataset)

            # Update progress
            processed_images += len(batch)
            total_qa_pairs = sum(len(ds) for ds in batch_datasets)
            self._log_progress(batch_idx, processed_images, total_qa_pairs)

            # Update progress bar description
            progress_bar.set_description(
                f"Processing batches ({processed_images} images, {total_qa_pairs} QA pairs)"
            )

        # Close progress bar
        progress_bar.close()

        # Finalization phase
        logger.info("🔧 Finalizing dataset construction and adding metadata")
        final_dataset = self._create_final_dataset(batch_datasets)

        # Success summary
        total_qa_pairs = sum(len(ds) for ds in batch_datasets) if batch_datasets else 0
        logger.info("✅ Dataset generation completed successfully!")
        logger.info(
            "📊 Generated %d QA pairs from %d processed images",
            total_qa_pairs,
            processed_images,
        )

        return final_dataset

    def _create_metadata(self) -> Dict[str, Any]:
        """Create metadata dictionary for the dataset."""
        metadata = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "confidence_threshold": self.conf_threshold,
            "batch_size": self.batch_size,
            "use_wbf": self.use_wbf,
            "questions": [str(q.__class__.__name__) for q in self.questions],
            "use_original_filenames": self.use_original_filenames,
            "filename_prefix": self.filename_prefix,
            "models": [],
        }

        # Add device info
        if not self.use_wbf:
            metadata["device"] = str(self.device)
        else:
            metadata["device_info"] = "Multiple devices may be used in WBF ensemble"

        # Add model information
        if self.models:
            for model in self.models:
                model_info = {
                    "backend": model.__class__.__module__.split(".")[-1],
                    "model_name": getattr(
                        model, "model_name", str(model.__class__.__name__)
                    ),
                }
                metadata["models"].append(model_info)
        else:
            metadata["models"] = [{"type": "ground_truth"}]

        return metadata


def generate_dataset(
    dataset_name: str,
    split: str,
    models: Optional[List[Any]] = None,
    use_wbf: bool = False,
    wbf_config: Optional[Dict[str, Any]] = None,
    conf_threshold: float = 0.2,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    allowable_set: Optional[List[str]] = None,
    question_configs: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 4,
    qa_workers: int = 4,
    save_steps: int = 50,
    save_path: str = "./graid-datasets",
    upload_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
    num_samples: Optional[int] = None,
    use_original_filenames: bool = True,
    filename_prefix: str = "img",
    force: bool = False,
):
    """
    Generate comprehensive HuggingFace datasets for object detection question-answering.

    This is the primary API function for creating VQA datasets from object detection data.
    It supports multiple detection backends, ensemble methods, parallel processing, and
    produces datasets ready for modern vision-language model training and evaluation.

    The function orchestrates the complete pipeline:
        1. Dataset loading and preprocessing
        2. Object detection (model-based or ground truth)
        3. Question-answer generation with configurable parallelism
        4. HuggingFace dataset construction with embedded PIL images
        5. Optional local saving and Hub upload

    Key Features:
        🎯 Multi-Backend Support: Detectron2, MMDetection, Ultralytics
        🔗 Ensemble Methods: Weighted Box Fusion for improved accuracy
        🚀 Parallel Processing: Configurable QA generation workers
        📊 Quality Control: Confidence thresholds and object filtering
        🖼️ Modern Format: PIL images ready for VLM workflows
        🌐 Hub Integration: Direct upload with metadata

    Args:
        dataset_name (str): Source dataset identifier. Supported values:
            - "bdd": BDD100K autonomous driving dataset
            - "nuimage": NuImages large-scale dataset
            - "waymo": Waymo Open Dataset
            - "custom": User-provided PyTorch dataset

        split (str): Dataset split to process. Common values:
            - "train": Training split
            - "val" or "validation": Validation split
            - "test": Test split

        models (Optional[List[Any]]): Object detection models for inference.
            If None, uses ground truth annotations from the dataset.
            Supports models from Detectron2, MMDetection, and Ultralytics.

        use_wbf (bool): Whether to use Weighted Box Fusion ensemble method
            to combine predictions from multiple models. Improves accuracy
            when multiple models are provided. Default: False

        wbf_config (Optional[Dict[str, Any]]): Configuration for WBF ensemble:
            - iou_threshold: IoU threshold for box fusion
            - model_weights: List of weights for each model
            - confidence_threshold: Minimum confidence for fusion

        conf_threshold (float): Minimum confidence score for accepting detections.
            Lower values include more detections (potentially noisy), higher values
            are more conservative. Range: 0.0-1.0. Default: 0.2

        batch_size (int): Number of images to process in each batch.
            Larger batches improve GPU utilization but require more memory.
            Default: 1 (safe for most systems)

        device (Optional[Union[str, torch.device]]): Device for model inference.
            If None, automatically detects best available device (CUDA/CPU).
            Examples: "cuda:0", "cpu", torch.device("cuda")

        allowable_set (Optional[List[str]]): Filter to include only specific
            object classes. Must be valid COCO category names. If None,
            includes all detected objects. Example: ["person", "car", "bicycle"]

        question_configs (Optional[List[Dict[str, Any]]]): Configuration for
            question generation. Each dict contains:
            - name: Question type (e.g., "HowMany", "LeftOf", "Quadrants")
            - params: Question-specific parameters
            If None, uses default question set.

        num_workers (int): Number of parallel workers for data loading.
            Should typically match CPU core count. Default: 4

        qa_workers (int): Number of parallel workers for QA generation.
            - 1: Sequential processing (debugging, memory-limited)
            - >1: Parallel processing (production, high-throughput)
            Recommended: 2-4x CPU cores. Default: 4

        save_steps (int): Save checkpoint every N batches for crash recovery.
            Larger values save less frequently but reduce I/O overhead.
            Default: 50

        save_path (str): Local directory to save the generated dataset.
            Creates standard HuggingFace dataset structure with Parquet files.
            Default: "./graid-datasets"

        upload_to_hub (bool): Whether to upload the dataset to HuggingFace Hub
            for sharing and distribution. Requires hub_repo_id. Default: False

        hub_repo_id (Optional[str]): HuggingFace Hub repository identifier
            in format "username/dataset-name". Required if upload_to_hub=True.

        hub_private (bool): Whether to make the Hub repository private.
            Public repositories are discoverable by the community. Default: False

        num_samples (Optional[int]): Maximum number of images to process.
            - None or 0: Process entire dataset
            - >0: Limit processing to specified number
            Useful for testing and quick iterations.

        use_original_filenames (bool): Whether to preserve original image filenames
            from the source dataset. If False, generates sequential names using
            filename_prefix. Default: True

        filename_prefix (str): Prefix for generated filenames when
            use_original_filenames=False. Example: "img" → "img000001.jpg"
            Default: "img"

        force (bool): Whether to force restart from scratch, ignoring any
            existing checkpoints from previous runs. Default: False

    Returns:
        DatasetDict: HuggingFace dataset dictionary containing the generated
            VQA dataset. Keys correspond to the processed split(s). Each dataset
            contains:
            - image: PIL Image objects ready for VLM workflows
            - annotations: COCO-style bounding box annotations
            - question: Generated question text
            - answer: Corresponding answer text
            - question_type: Type of question (e.g., "HowMany", "LeftOf")
            - source_id: Original image identifier

    Raises:
        ValueError: If dataset_name is not supported, configuration is invalid,
            or required parameters are missing
        RuntimeError: If model loading fails, inference fails, or dataset
            construction encounters errors
        FileNotFoundError: If specified paths don't exist
        PermissionError: If unable to write to save_path or access Hub

    Examples:
        Basic usage with ground truth:
        >>> dataset = generate_dataset(
        ...     dataset_name="bdd",
        ...     split="val",
        ...     num_samples=100
        ... )
        >>> print(f"Generated {len(dataset['val'])} QA pairs")

        Multi-model ensemble with WBF:
        >>> from graid.models import YoloModel, DetectronModel
        >>> models = [YoloModel("yolov8x.pt"), DetectronModel("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")]
        >>> dataset = generate_dataset(
        ...     dataset_name="bdd",
        ...     split="train",
        ...     models=models,
        ...     use_wbf=True,
        ...     wbf_config={"iou_threshold": 0.6, "model_weights": [1.0, 1.2]},
        ...     qa_workers=8,
        ...     allowable_set=["person", "car", "bicycle"],
        ...     save_path="./datasets/bdd_vqa",
        ...     upload_to_hub=True,
        ...     hub_repo_id="myuser/bdd-reasoning-dataset"
        ... )

        Custom question configuration:
        >>> questions = [
        ...     {"name": "HowMany", "params": {}},
        ...     {"name": "Quadrants", "params": {"N": 3, "M": 3}},
        ...     {"name": "LeftOf", "params": {}}
        ... ]
        >>> dataset = generate_dataset(
        ...     dataset_name="nuimage",
        ...     split="val",
        ...     question_configs=questions,
        ...     qa_workers=4
        ... )
    """
    # Create dataset builder
    builder = HuggingFaceDatasetBuilder(
        dataset_name=dataset_name,
        split=split,
        models=models,
        use_wbf=use_wbf,
        wbf_config=wbf_config,
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        device=device,
        allowable_set=allowable_set,
        question_configs=question_configs,
        num_workers=num_workers,
        qa_workers=qa_workers,
        num_samples=num_samples,
        save_steps=save_steps,
        save_path=save_path,
        use_original_filenames=use_original_filenames,
        filename_prefix=filename_prefix,
        force=force,
    )

    # Build the dataset
    dataset_dict = builder.build()

    # Save locally if requested
    if save_path:
        save_path_obj = Path(save_path)
        data_dir = save_path_obj / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for split_name, dataset in dataset_dict.items():
            parquet_file = data_dir / f"{split_name}-00000-of-00001.parquet"
            dataset.to_parquet(str(parquet_file))
            logger.info(f"Dataset {split_name} split saved to {parquet_file}")

    # Upload to HuggingFace Hub if requested
    if upload_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when upload_to_hub=True")

        # Import Hub utilities locally
        from huggingface_hub import create_repo, upload_large_folder

        logger.info(f"Uploading to HuggingFace Hub: {hub_repo_id}")

        # Create repository
        create_repo(
            hub_repo_id, repo_type="dataset", private=hub_private, exist_ok=True
        )

        # Upload images and directory structure using upload_large_folder
        # if save_path:
        #     logger.info(
        #         f"Uploading dataset files from {save_path} to Hub repository..."
        #     )
        #     try:
        #         upload_large_folder(
        #             repo_id=hub_repo_id,
        #             repo_type="dataset",
        #             folder_path=str(save_path),
        #         )
        #         logger.info("Image and directory upload completed successfully")
        #     except Exception as e:
        #         logger.error(f"Failed to upload files to Hub: {e}")
        #         raise

        # Push dataset (images already cast to HFImage in builder.build())

        # Push dataset with proper settings
        dataset_dict.push_to_hub(
            repo_id=hub_repo_id,
            private=hub_private,
            # embed_external_files=False,  # Critical: no byte duplication
            commit_message=f"Upload {dataset_name} {split} dataset",
            max_shard_size="5GB",
        )
        logger.info(f"Dataset pushed to HuggingFace Hub: {hub_repo_id}")

    # Clean up temporary image files only if we uploaded to hub
    # In multi-split scenarios, cleanup is deferred until all splits are processed
    # if upload_to_hub and hasattr(builder, "_cleanup_images"):
    #     try:
    #         builder._cleanup_images()
    #         logger.debug(
    #             "✅ Cleaned up temporary image files after successful Hub upload"
    #         )
    #     except Exception as e:
    #         logger.warning(f"Failed to cleanup temporary image files: {e}")

    return dataset_dict


# Compatibility functions for existing code
def list_available_questions() -> Dict[str, Dict[str, Any]]:
    """
    List all available question types with their descriptions and parameters.

    This function provides a comprehensive catalog of question generation strategies
    available in the GRAID system. Each question type implements specific reasoning
    patterns for visual question answering based on object detection results.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping question names to their metadata:
            - "question": Human-readable description of the question type
            - "parameters": Dict of configurable parameters (currently empty,
              reserved for future parameter introspection)

    Example:
        >>> questions = list_available_questions()
        >>> for name, info in questions.items():
        ...     print(f"{name}: {info['question']}")
        HowMany: How many objects of type X are in the image?
        LeftOf: Which objects are to the left of object X?
        ...
    """
    # Local import to avoid heavy dependencies
    from graid.questions.ObjectDetectionQ import ALL_QUESTION_CLASSES

    question_info = {}

    for question_name, question_class in ALL_QUESTION_CLASSES.items():
        try:
            # Create a temporary instance to get the question text
            temp_instance = question_class()
            question_text = getattr(temp_instance, "question", question_name)
        except Exception:
            question_text = question_name

        # For now, return basic info - can be extended later
        question_info[question_name] = {
            "question": question_text,
            "parameters": {},  # Would need to be populated based on inspection
        }

    return question_info


def interactive_question_selection() -> List[Dict[str, Any]]:
    """
    Interactive terminal interface for selecting and configuring question types.

    This function provides a user-friendly command-line interface for selecting
    which question generation strategies to use in dataset creation. Users can
    choose from all available question types or select specific subsets.

    The interface displays:
        - Numbered list of all available question types
        - Description of each question type
        - Parameter configuration options (future enhancement)

    User Input Options:
        - Specific numbers (comma-separated): Select individual questions
        - "all": Select all available question types with default parameters

    Returns:
        List[Dict[str, Any]]: List of question configuration dictionaries, each containing:
            - "name": Question type name (e.g., "HowMany", "LeftOf")
            - "params": Parameter dictionary (currently empty, default parameters)

    Raises:
        KeyboardInterrupt: If user cancels the selection process

    Example:
        >>> configs = interactive_question_selection()
        📋 Question Selection
        ========================
        Available questions:
          1. HowMany
             How many objects of type X are in the image?
        ...
        Selection: 1,3,5
        >>> print(configs)
        [{"name": "HowMany", "params": {}}, {"name": "LeftOf", "params": {}}, ...]
    """
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
        print()

    print("Enter question numbers (comma-separated) or 'all' for all questions:")

    while True:
        try:
            selection = input("Selection: ").strip()

            if selection.lower() == "all":
                # Add all questions with default parameters
                for name in available_questions.keys():
                    question_configs.append({"name": name, "params": {}})
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
                question_configs.append({"name": name, "params": {}})

            break

        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            raise KeyboardInterrupt()

    return question_configs


def create_webdataset_archive(dataset_path: str, output_path: str, max_tar_size_mb: int = 1000):
    """
    ALTERNATIVE SOLUTION: Convert existing dataset to WebDataset format (TAR archives).
    
    This function creates TAR archives from an existing GRAID dataset to solve the
    HuggingFace 10k file limit issue. Creates multiple TAR files if needed to stay
    under size limits.
    
    Args:
        dataset_path: Path to existing dataset directory
        output_path: Path where TAR files will be created
        max_tar_size_mb: Maximum size per TAR file in MB
        
    Returns:
        List of created TAR file paths
    """
    import tarfile
    import json
    from pathlib import Path
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing parquet to get QA pairs
    from datasets import load_dataset
    
    tar_files = []
    current_size = 0
    tar_index = 0
    current_tar = None
    
    logger.info(f"Converting {dataset_path} to WebDataset format...")
    
    # Process each split
    for split in ['train', 'val']:
        parquet_file = dataset_path / "data" / f"{split}-00000-of-00001.parquet"
        if not parquet_file.exists():
            continue
            
        dataset = load_dataset('parquet', data_files=str(parquet_file))
        
        for i, sample in enumerate(dataset[split]):
            # Create new TAR if needed
            if current_tar is None or current_size > max_tar_size_mb * 1024 * 1024:
                if current_tar:
                    current_tar.close()
                tar_path = output_path / f"{split}_{tar_index:04d}.tar"
                current_tar = tarfile.open(tar_path, 'w')
                tar_files.append(str(tar_path))
                current_size = 0
                tar_index += 1
                logger.info(f"Creating TAR archive: {tar_path}")
            
            # Add image to TAR
            image_path = sample['image']['path']
            full_image_path = dataset_path / image_path
            if full_image_path.exists():
                current_tar.add(full_image_path, arcname=f"{i:08d}.jpg")
                current_size += full_image_path.stat().st_size
                
                # Add metadata JSON
                metadata = {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'question_type': sample['question_type'],
                    'source_id': sample['source_id'],
                    'annotations': sample['annotations']
                }
                
                # Create temp JSON file and add to TAR
                temp_json = f"/tmp/meta_{i}.json"
                with open(temp_json, 'w') as f:
                    json.dump(metadata, f)
                current_tar.add(temp_json, arcname=f"{i:08d}.json")
                Path(temp_json).unlink()  # cleanup temp file
    
    if current_tar:
        current_tar.close()
    
    logger.info(f"Created {len(tar_files)} WebDataset TAR files")
    return tar_files
