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
    - Memory-efficient dataset generation
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

# Example:
    # GRAID_PROFILE_QUESTIONS=1 GRAID_DEBUG_VERBOSE=1 graid generate-dataset --config bdd_train_gt_dataset_config.json --no-interactive
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union
import threading

import numpy as np
import torch
from graid.utilities.common import get_default_device
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


from graid.questions.ObjectDetectionQ import Question, DEPTH_QUESTIONS


logger = logging.getLogger(__name__)

# Global thread-local storage to hold per-thread resources like models
thread_local_storage = threading.local()


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
        self, batch_data: list[tuple[Image.Image, list[Any], str, int, int]]
    ) -> list[Any]:
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
                - List[dict[str, Any]]: QA pairs (when profiling disabled)
                - Tuple[List[dict[str, Any]], dict[str, tuple[float, int]]]:
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
        self, batch_data: list[tuple[Image.Image, list[Any], str, int, int]]
    ) -> list[Any]:
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
                results.append([])

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
        # Persist a single executor across batches so worker threads keep
        # their thread-local state (e.g., per-thread SAM predictors).
        self._executor = ThreadPoolExecutor(max_workers=self.qa_workers)

    def shutdown(self):
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def process_batch(
        self, batch_data: list[tuple[Image.Image, list[Any], str, int, int]]
    ) -> list[Any]:
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

        results = list(
            self._executor.map(self.qa_generator._qa_for_image_threadsafe, batch_data)
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
        - Memory-efficient generator-based processing
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
        models: Optional[list[Any]] = None,
        use_wbf: bool = False,
        wbf_config: Optional[dict[str, Any]] = None,
        conf_threshold: float = 0.2,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        allowable_set: Optional[list[str]] = None,
        question_configs: Optional[list[dict[str, Any]]] = None,
        num_workers: int = 4,
        qa_workers: int = 4,
        num_samples: Optional[int] = None,
        use_original_filenames: bool = True,
        filename_prefix: str = "img",
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
            save_path: Path to save dataset (required)
            use_original_filenames: Whether to keep original filenames
            filename_prefix: Prefix for generated filenames if not using originals
        """
        self.dataset_name = dataset_name
        self.split = split
        self.models = models or []
        self.use_wbf = use_wbf
        self.wbf_config = wbf_config or {}
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()
        logger.debug("✓ Initialized device: %s", self.device)
        self.allowable_set = allowable_set
        self.num_workers = num_workers
        self.qa_workers = qa_workers
        self.num_samples = num_samples
        self.save_path = Path(save_path)
        self.use_original_filenames = use_original_filenames
        self.filename_prefix = filename_prefix

        # Question profiling (timings)
        self.profile_questions: bool = bool(os.getenv("GRAID_PROFILE_QUESTIONS"))
        self.question_counts: dict[str, int] = {}

        # Enhanced profiling for is_applicable vs apply efficiency
        self.question_detailed_stats: dict[str, dict[str, Any]] = {}

        # Validate allowable_set
        if allowable_set is not None:
            from graid.utilities.coco import validate_coco_objects

            is_valid, error_msg = validate_coco_objects(allowable_set)
            if not is_valid:
                raise ValueError(f"Invalid allowable_set: {error_msg}")

        self.transform = self._get_dataset_transform()

        self.questions = self._initialize_questions(question_configs)

        self._init_dataset_loader()

        self.depth_model = None
        if self._has_depth_questions(self.questions):
            from graid.models.DepthPro import DepthPro
            self.depth_model = DepthPro(device=self.device)

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
        self, question_configs: Optional[list[dict[str, Any]]]
    ) -> list[Question]:
        """Initialize question objects from configurations."""
        from graid.questions.ObjectDetectionQ import ALL_QUESTION_CLASSES

        if question_configs is None:
            # If no specific questions are configured, use all available ones.
            # We instantiate them with default parameters.
            return [cls() for cls in ALL_QUESTION_CLASSES.values()]

        questions = []
        for config in question_configs:
            question_name = config["name"]
            params = config.get("params", {})
            question_class = ALL_QUESTION_CLASSES.get(question_name)

            if question_class:
                try:
                    questions.append(question_class(**params))
                except Exception as e:
                    logger.error(
                        f"Error initializing question '{question_name}' with params {params}: {e}"
                    )
            else:
                logger.warning(f"Unknown question type: {question_name}")
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

    def _has_depth_questions(self, questions: list[Question]) -> bool:
        for q in questions:
            if q.__class__ in DEPTH_QUESTIONS:
                logger.debug(f"Question {q.__class__.__name__} is a depth question")
                return True
        return False

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

    def _infer_source_name(self, example: dict[str, Any]) -> Optional[str]:
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
            if image.max() > 1.0:
                # Values already in [0, 255] range, just convert to uint8
                image = image.astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return Image.fromarray(image)

    def _build_coco_annotations(
        self, detections: list[Any], image_width: int, image_height: int
    ) -> list[dict[str, Any]]:
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
                "category_id": int(
                    detection.cls
                ),  # Use actual category ID from detection
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
        detections: list[Any],
        source_id: str,
        image_index: int,
    ) -> list[dict[str, Any]]:
        """Generate question-answer pairs for a single image with embedded image bytes."""
        qa_pairs = []

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
        original_format = ext.upper().lstrip(".") if ext else "PNG"

        # Map common extensions to PIL formats
        format_map = {
            "JPG": "JPEG",
            "JPEG": "JPEG",
            "PNG": "PNG",
            "BMP": "BMP",
            "TIFF": "TIFF",
        }
        pil_format = format_map.get(original_format, "PNG")  # Default to PNG if unknown

        buffer = io.BytesIO()
        if pil_format == "JPEG":
            # For JPEG, save without additional compression (quality=100)
            rgb_img.save(buffer, format=pil_format, quality=100, optimize=False)
        elif pil_format == "PNG":
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

        # Normalize detections for consistent tensor shapes before question processing
        if detections:
            from graid.interfaces.ObjectDetectionI import ObjectDetectionUtils

            normalized_data = ObjectDetectionUtils.normalize_detections(detections)
            detections = normalized_data["detections"]

        # Generate questions and answers with enhanced profiling
        # Shared context cache that reusable-heavy questions (e.g. depth) can populate / reuse
        cache: dict[str, Any] = {}

        # Lazily initialize and cache a SAM predictor for each thread.
        if self._has_depth_questions(self.questions) and not hasattr(thread_local_storage, "sam_predictor"):
            from graid.utilities.sam_utils import SAMPredictor
            logger.debug("Initializing SAM predictor for thread %s...", threading.get_ident())
            thread_local_storage.sam_predictor = SAMPredictor(device=self.device)
        if self._has_depth_questions(self.questions):
            cache["sam_predictor"] = thread_local_storage.sam_predictor
            cache["depth_model"] = self.depth_model

        for question in self.questions:
            qname = question.__class__.__name__

            # Initialize detailed stats for this question if not exists
            if self.profile_questions and qname not in self.question_detailed_stats:
                self.question_detailed_stats[qname] = {
                    "is_applicable_time": (0.0, 0),
                    "is_applicable_true_count": 0,
                    "apply_time": (0.0, 0),
                    "apply_empty_results": 0,
                    "total_qa_generated": 0,
                    "question_text": getattr(
                        question, "question", qname
                    ),  # Full question string
                }

            # Profile is_applicable timing
            if detections:
                is_applicable_start = (
                    time.perf_counter() if self.profile_questions else None
                )
                is_applicable_result = question.is_applicable(rgb_img, detections)

                if self.profile_questions and is_applicable_start is not None:
                    is_applicable_time = time.perf_counter() - is_applicable_start
                    current_time, current_count = self.question_detailed_stats[qname][
                        "is_applicable_time"
                    ]
                    self.question_detailed_stats[qname]["is_applicable_time"] = (
                        current_time + is_applicable_time,
                        current_count + 1,
                    )

                if is_applicable_result:
                    if self.profile_questions:
                        self.question_detailed_stats[qname][
                            "is_applicable_true_count"
                        ] += 1

                    # Profile apply timing
                    apply_start = (
                        time.perf_counter() if self.profile_questions else None
                    )
                    try:
                        # Always use apply_with_cache: the base class provides a
                        # transparent default wrapper, and specialised questions
                        # override it for cache-aware behaviour.
                        qa_results = question.apply_with_cache(
                            rgb_img, detections, cache
                        )

                        if self.profile_questions and apply_start is not None:
                            apply_time = time.perf_counter() - apply_start
                            current_time, current_count = self.question_detailed_stats[
                                qname
                            ]["apply_time"]
                            self.question_detailed_stats[qname]["apply_time"] = (
                                current_time + apply_time,
                                current_count + 1,
                            )

                        # Check if apply returned empty results despite is_applicable=True
                        if not qa_results and self.profile_questions:
                            self.question_detailed_stats[qname][
                                "apply_empty_results"
                            ] += 1

                        for qa_item in qa_results:
                            if (
                                not isinstance(qa_item, (tuple, list))
                                or len(qa_item) != 2
                            ):
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

                            # Track successful QA generation
                            if self.profile_questions:
                                self.question_detailed_stats[qname][
                                    "total_qa_generated"
                                ] += 1

                    except Exception as e:
                        # Temporary verbose trace to locate indexing bugs
                        logger.exception(
                            "Question %s failed on image %s",
                            question.__class__.__name__,
                            source_id,
                        )
                        continue

        # Clear heavy per-image cache items after all questions
        for k in ["sam_masks", "depth_map"]:
            try:
                if k in cache:
                    del cache[k]
            except Exception:
                pass
        return qa_pairs

    def _qa_for_image_threadsafe(self, batch_args: tuple) -> list[dict[str, Any]]:
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
            # Return empty result
            return []

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
        self, batch: list[Any]
    ) -> tuple[torch.Tensor, list[Any]]:
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
        batch: list[Any],
        batch_images: torch.Tensor,
        labels: list[Any],
    ) -> list[tuple[Image.Image, list[Any], str, int, int]]:
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

    def _process_qa_results(self, batch_results_raw: list[Any]) -> list[dict[str, Any]]:
        """Process raw QA results."""
        batch_results: list[dict[str, Any]] = []

        # Process results
        for ret in batch_results_raw:
            if isinstance(ret, list):
                batch_results.extend(ret)
            else:
                logger.warning(
                    f"Unexpected return type from QA processing: {type(ret)}"
                )

        return batch_results

    def _update_progress_tracking(
        self,
        batch_results: list[dict[str, Any]],
    ):
        """Update question counts tracking."""
        # Update per-question counts
        for item in batch_results:
            try:
                qtype = item.get("question_type")
                if qtype:
                    self.question_counts[qtype] = self.question_counts.get(qtype, 0) + 1
            except Exception:
                pass

    def _log_progress(self, batch_idx: int, processed_images: int, total_qa_pairs: int):
        """Log progress every 10 batches."""
        if batch_idx % 10 == 0:
            logger.info(
                f"Processed {processed_images} images, generated {total_qa_pairs} QA pairs"
            )

    def build(self):
        """
        Build the HuggingFace dataset using memory-efficient generator approach.

        This method creates datasets using Dataset.from_generator to maintain bounded
        memory usage while preserving parallel QA processing. Key improvements:
        1. Generator-based processing eliminates memory accumulation
        2. Parallel QA workers still utilized for performance
        3. Bounded memory via writer_batch_size parameter
        4. Embedded images preserved (solving 10k file limit)

        Returns:
            DatasetDict containing the generated VQA dataset
        """
        logger.info(
            "🚀 Building HuggingFace dataset for %s/%s ",
            self.dataset_name,
            self.split,
        )

        # Import Dataset locally to avoid import issues
        from datasets import Dataset, DatasetDict
        from datasets import Image as HFImage

        # Create dataset using memory-efficient generator
        logger.info("🔧 Creating dataset using a generator...")

        dataset = Dataset.from_generator(
            self._qa_data_generator,
            # Let HuggingFace infer features from the first examples
        )

        # Cast image column to HFImage format
        logger.debug("🎯 Converting image bytes to HFImage format...")
        dataset = dataset.cast_column("image", HFImage())

        # Add metadata
        metadata = self._create_metadata()
        dataset.info.description = (
            f"Object detection QA dataset for {self.dataset_name}"
        )
        dataset.info.features = dataset.features
        dataset.info.config_name = json.dumps(metadata)

        # Create DatasetDict
        dataset_dict = DatasetDict({self.split: dataset})

        logger.info(f"✅ Generated {len(dataset)} question-answer pairs")

        # Log profiling information - using detailed stats only

        # Log per-question counts
        if self.question_counts:
            pairs = sorted(  # by question type, most frequent first
                self.question_counts.items(), key=lambda kv: kv[1], reverse=True
            )
            summary = ", ".join([f"{k}={v}" for k, v in pairs])
            logger.info(f"Per-question counts: {summary}")

        # Log detailed profiling statistics
        if self.profile_questions and self.question_detailed_stats:
            from graid.utils.profiling import log_profiling_statistics

            question_stats = {"detailed_stats": self.question_detailed_stats}
            log_profiling_statistics(
                question_stats, "Detailed Question Processing Statistics"
            )

        return dataset_dict

    def _qa_data_generator(self):
        """
        Memory-efficient generator that yields individual QA pairs with parallel processing.

        This generator maintains bounded memory usage by yielding QA pairs one at a time
        instead of accumulating them in memory. Parallel QA processing is preserved
        within each batch for optimal performance.

        Yields:
            Dict[str, Any]: Individual QA pair with embedded image bytes and unique ID
        """
        logger.debug("📋 Initializing data loader and processing components")
        data_loader = self._create_data_loader()
        qa_processor = QAProcessorFactory.create(
            self.qa_workers, self, self.profile_questions
        )

        # Calculate total batches for progress tracking
        total_batches = self._calculate_total_batches(data_loader)
        processed_images = 0
        total_qa_pairs = 0
        unique_id_counter = 0  # Counter for unique IDs

        logger.info(
            "📊 Processing %d total batches (%d images per batch) with generator",
            total_batches,
            self.batch_size,
        )

        logger.debug(
            "🔄 Starting batch processing with %s strategy",
            "parallel" if self.qa_workers > 1 else "sequential",
        )

        # Create progress bar for batch processing
        progress_bar = tqdm(
            enumerate(data_loader), desc="Generating QA pairs", total=total_batches
        )

        for batch_idx, batch in progress_bar:
            # Early stopping logic
            if self._should_stop_early(batch_idx, processed_images):
                logger.info(f"Early stopping at batch {batch_idx}")
                break

            # Get predictions and prepare batch data (same as before)
            batch_images, labels = self._get_batch_predictions(batch)
            batch_data = self._prepare_batch_data(
                batch_idx, batch, batch_images, labels
            )

            # Process QA using parallel/sequential strategy (unchanged)
            batch_results_raw = qa_processor.process_batch(batch_data)

            # Process results and update tracking
            batch_results = self._process_qa_results(batch_results_raw)
            self._update_progress_tracking(batch_results)

            # Yield individual QA pairs instead of accumulating
            for qa_pair in batch_results:
                # Add unique ID to each QA pair
                qa_pair["id"] = unique_id_counter
                unique_id_counter += 1
                yield qa_pair
                total_qa_pairs += 1

            # Update progress tracking
            processed_images += len(batch)
            self._log_progress(batch_idx, processed_images, total_qa_pairs)

            # Update progress bar description
            progress_bar.set_description(
                f"Generated {total_qa_pairs} QA pairs from {processed_images} images"
            )

        # Close progress bar
        progress_bar.close()

        # Ensure parallel executor is shut down cleanly
        try:
            if hasattr(qa_processor, "shutdown"):
                qa_processor.shutdown()
        except Exception:
            pass

        logger.info(
            f"🎯 Generator completed: {total_qa_pairs} QA pairs from {processed_images} images"
        )

    def _get_features_schema(self):
        """
        Define the dataset features schema for Dataset.from_generator.

        Returns:
            datasets.Features: Schema definition for the generated dataset
        """
        from datasets import Features, Sequence, Value

        return Features(
            {
                "id": Value("int64"),  # Unique identifier for each QA pair
                "image": {
                    "bytes": Value("binary"),
                    "path": Value("string"),
                },  # Image dict with embedded bytes
                "annotations": Sequence(
                    {
                        "bbox": Sequence(Value("float32"), length=4),
                        "category_id": Value("int32"),
                        "category": Value("string"),
                        "iscrowd": Value("int32"),
                        "area": Value("float32"),
                        "score": Value("float32"),
                    }
                ),
                "question": Value("string"),
                "answer": Value("string"),
                "question_type": Value("string"),
                "source_id": Value("string"),
            }
        )

    def _create_metadata(self) -> dict[str, Any]:
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

        # Add license metadata for downstream consumers
        metadata["license"] = "cc-by-nc-4.0"
        metadata["commercial_license_required"] = True
        metadata["license_contact"] = "Karim Elmaaroufi for commercial licensing"

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
    models: Optional[list[Any]] = None,
    use_wbf: bool = False,
    wbf_config: Optional[dict[str, Any]] = None,
    conf_threshold: float = 0.2,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    allowable_set: Optional[list[str]] = None,
    question_configs: Optional[list[dict[str, Any]]] = None,
    num_workers: int = 4,
    qa_workers: int = 4,
    save_path: str = "./graid-datasets",
    upload_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
    num_samples: Optional[int] = None,
    use_original_filenames: bool = True,
    filename_prefix: str = "img",
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

        wbf_config (Optional[dict[str, Any]]): Configuration for WBF ensemble:
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

        question_configs (Optional[list[dict[str, Any]]]): Configuration for
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



    Returns:
        DatasetDict: HuggingFace dataset dictionary containing the generated
            VQA dataset. Keys correspond to the processed split(s). Each dataset
            contains:
            - id: Unique identifier for each QA pair (row number)
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
        save_path=save_path,
        use_original_filenames=use_original_filenames,
        filename_prefix=filename_prefix,
    )

    # Build the dataset
    dataset_dict = builder.build()

    # Save locally if requested
    # if save_path:
    #     save_path_obj = Path(save_path)
    #     data_dir = save_path_obj / "data"
    #     data_dir.mkdir(parents=True, exist_ok=True)

    #     for split_name, dataset in dataset_dict.items():
    #         parquet_file = data_dir / f"{split_name}-00000-of-00001.parquet"
    #         dataset.to_parquet(str(parquet_file))
    #         logger.info(f"Dataset {split_name} split saved to {parquet_file}")

    # Upload to HuggingFace Hub if requested
    if upload_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when upload_to_hub=True")

        # Import Hub utilities locally
        from huggingface_hub import create_repo, upload_file

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
            commit_message=f"Upload {dataset_name} {split} dataset",
            max_shard_size="5GB",
        )
        logger.info(f"Dataset pushed to HuggingFace Hub: {hub_repo_id}")

        # Note: Using CC BY-NC 4.0 license - no custom license file upload needed
        # Commercial licensing information is included in the README

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

    # Collect statistics from builder
    stats = None
    if builder.profile_questions and hasattr(builder, "question_detailed_stats"):
        stats = {
            "question_counts": builder.question_counts,
            "detailed_stats": builder.question_detailed_stats,
        }

    return dataset_dict, stats


# Compatibility functions for existing code
def list_available_questions() -> dict[str, dict[str, Any]]:
    """
    List all available question types with their descriptions and parameters.

    This function provides a comprehensive catalog of question generation strategies
    available in the GRAID system. Each question type implements specific reasoning
    patterns for visual question answering based on object detection results.

    Returns:
        dict[str, dict[str, Any]]: Dictionary mapping question names to their metadata:
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


def interactive_question_selection() -> list[dict[str, Any]]:
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
        list[dict[str, Any]]: List of question configuration dictionaries, each containing:
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


def create_webdataset_archive(
    dataset_path: str, output_path: str, max_tar_size_mb: int = 1000
) -> list[str]:
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
    import json
    import tarfile
    from pathlib import Path

    dataset_path_obj = Path(dataset_path)
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Load existing parquet to get QA pairs
    from datasets import load_dataset

    tar_files = []
    current_size = 0
    tar_index = 0
    current_tar = None

    logger.info(f"Converting {dataset_path} to WebDataset format...")

    # Process each split
    for split in ["train", "val"]:
        parquet_file = dataset_path_obj / "data" / f"{split}-00000-of-00001.parquet"
        if not parquet_file.exists():
            continue

        dataset = load_dataset("parquet", data_files=str(parquet_file))

        for i, sample in enumerate(dataset[split]):
            # Create new TAR if needed
            if current_tar is None or current_size > max_tar_size_mb * 1024 * 1024:
                if current_tar:
                    current_tar.close()
                tar_path = output_path_obj / f"{split}_{tar_index:04d}.tar"
                current_tar = tarfile.open(tar_path, "w")
                tar_files.append(str(tar_path))
                current_size = 0
                tar_index += 1
                logger.info(f"Creating TAR archive: {tar_path}")

            # Add image to TAR
            image_path = sample["image"]["path"]
            full_image_path = dataset_path_obj / image_path
            if full_image_path.exists():
                current_tar.add(full_image_path, arcname=f"{i:08d}.jpg")
                current_size += full_image_path.stat().st_size

                # Add metadata JSON
                metadata = {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "question_type": sample["question_type"],
                    "source_id": sample["source_id"],
                    "annotations": sample["annotations"],
                }

                # Create temp JSON file and add to TAR
                temp_json = f"/tmp/meta_{i}.json"
                with open(temp_json, "w") as f:
                    json.dump(metadata, f)
                current_tar.add(temp_json, arcname=f"{i:08d}.json")
                Path(temp_json).unlink()  # cleanup temp file

    if current_tar:
        current_tar.close()

    logger.info(f"Created {len(tar_files)} WebDataset TAR files")
    return tar_files
