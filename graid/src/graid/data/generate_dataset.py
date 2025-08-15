"""
GRAID HuggingFace Dataset Generation

Complete rewrite for generating HuggingFace datasets with proper COCO bbox format,
path-based Image columns, and simplified architecture.
"""

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from graid.utilities.common import get_default_device

logger = logging.getLogger(__name__)


class HuggingFaceDatasetBuilder:
    """
    Complete rewrite of the dataset builder for generating HuggingFace datasets.
    
    Features:
    - Proper COCO bbox format with category strings
    - Path-based Image columns (no byte duplication)
    - Clean directory structure: {split}/images/ for images
    - Support for original filenames vs generated filenames
    - Simplified architecture without complex checkpointing
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
        save_path: Optional[str] = None,
        use_original_filenames: bool = True,
        filename_prefix: str = "img",
        force: bool = False,
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
            save_path: Path to save dataset (optional)
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
        self.save_path = Path(save_path) if save_path else Path("./graid_dataset")
        self.use_original_filenames = use_original_filenames
        self.filename_prefix = filename_prefix
        self.force = force
        
        # Question profiling (timings)
        self.profile_questions: bool = bool(os.getenv("GRAID_PROFILE_QUESTIONS"))
        self._question_timings: Dict[str, tuple[float, int]] = {}
        self._question_counts: Dict[str, int] = {}
        
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

        # Create directory structure
        self.images_dir = self.save_path / self.split / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def _initialize_questions(self, question_configs: Optional[List[Dict[str, Any]]]) -> List[Any]:
        """Initialize question objects from configuration."""
        if question_configs is None:
            # Use all available questions
            from graid.questions.ObjectDetectionQ import ALL_QUESTION_CLASSES
            return list(ALL_QUESTION_CLASSES.values())
        
        questions = []
        from graid.questions.ObjectDetectionQ import (
            IsObjectCentered, WidthVsHeight, LargestAppearance, RankLargestK,
            MostAppearance, LeastAppearance, LeftOf, RightOf, LeftMost, RightMost,
            HowMany, MostClusteredObjects, WhichMore, AreMore, Quadrants,
            LeftMostWidthVsHeight, RightMostWidthVsHeight, ObjectsInRow, ObjectsInLine,
            MoreThanThresholdHowMany, LessThanThresholdHowMany, MultiChoiceHowMany
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
                    logger.error(f"Failed to initialize {question_name} with params {question_params}: {e}")
                    # Fall back to default initialization
                    question_instance = question_class()
            else:
                question_instance = question_class()
            
            questions.append(question_instance)

        if not questions:
            raise ValueError("No valid questions configured")

        return questions

    def _init_dataset_loader(self):
        """Initialize the appropriate dataset loader."""
        from graid.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
        
        try:
            if self.dataset_name == "bdd":
                pkl_root = Path("data") / f"bdd_{self.split}"
                rebuild_needed = not (pkl_root / "0.pkl").exists()
                self.dataset_loader = Bdd100kDataset(
                    split=self.split,  # type: ignore
                    transform=self.transform,
                    use_time_filtered=False,
                    rebuild=rebuild_needed,
                )
            elif self.dataset_name == "nuimage":
                self.dataset_loader = NuImagesDataset(
                    split=self.split,  # type: ignore
                    size="all",
                    transform=self.transform
                )
            elif self.dataset_name == "waymo":
                split_name = "validation" if self.split == "val" else self.split + "ing"
                self.dataset_loader = WaymoDataset(
                    split=split_name,  # type: ignore
                    transform=self.transform
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
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
    
    def _convert_image_to_pil(self, image: Union[torch.Tensor, np.ndarray]) -> Image.Image:
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
        self, 
        detections: List[Any], 
        image_width: int, 
        image_height: int
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
                "score": float(detection.score) if hasattr(detection, 'score') else 1.0,
            }
            annotations.append(annotation)
        
        return annotations
    
    def _qa_for_image(
        self, 
        pil_image: Image.Image, 
        detections: List[Any], 
        source_id: str, 
        image_index: int
    ) -> Union[List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, tuple[float, int]]]]:
        """Generate question-answer pairs for a single image."""
        qa_pairs = []
        local_timings: Dict[str, tuple[float, int]] = {} if self.profile_questions else {}
        
        # Generate filename and save image
        source_name = self._infer_source_name({"name": source_id}) if hasattr(self, '_current_example') else None
        filename = self._generate_filename(image_index, source_name)
        image_path = self.images_dir / filename
        
        # Save image if it doesn't exist
        if not image_path.exists():
            try:
                rgb_img = pil_image if pil_image.mode in ("RGB", "L") else pil_image.convert("RGB")
                rgb_img.save(image_path, format="JPEG", quality=95, optimize=True)
                except Exception as e:
                logger.error(f"Failed to save image to '{image_path}': {e}")
                return []
        
        # Generate COCO annotations
        annotations = self._build_coco_annotations(
            detections, 
            pil_image.width, 
            pil_image.height
        )
        
        # Generate relative path for HuggingFace dataset
        relative_image_path = f"{self.split}/images/{filename}"
        
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
                        
                        # Build the final QA pair
                        qa_pair = {
                            "image": relative_image_path,
                            "annotations": annotations,
                        "question": question_text,
                        "answer": answer_text,
                            "question_type": question.__class__.__name__,
                        "source_id": source_id,
                        }
                        
                        # Add source_filename if using generated filenames
                        if not self.use_original_filenames and source_name:
                            qa_pair["source_filename"] = source_name
                        
                        qa_pairs.append(qa_pair)
                        
                except Exception as e:
                    logger.warning(f"Question {question.__class__.__name__} failed on image {source_id}: {e}")
                    continue
        
        if self.profile_questions:
            return (qa_pairs, local_timings)
        return qa_pairs

    def _qa_for_image_threadsafe(self, batch_args: tuple) -> Union[List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, tuple[float, int]]]]:
        """Thread-safe wrapper for _qa_for_image with unique image indexing."""
        pil_image, detections, source_id, base_image_index, batch_j = batch_args
        
        # Create thread-safe unique image index
        thread_id = threading.get_ident()
        unique_image_index = base_image_index + (thread_id % 1000000) * 10000 + batch_j
        
        try:
            return self._qa_for_image(pil_image, detections, source_id, unique_image_index)
        except Exception as e:
            logger.error(f"Error in threaded QA generation for {source_id}: {e}")
            # Return empty results that match expected format
            if self.profile_questions:
                return ([], {})
        else:
                return []
    
    def _save_checkpoint(self, batch_idx: int, results: List[Dict[str, Any]], processed_images: int):
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
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save results so far
        results_file = self.checkpoint_dir / f"results_{self.split}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        logger.info(f"Checkpoint saved at batch {batch_idx} ({processed_images} images processed)")
    
    def _load_checkpoint(self) -> tuple[int, List[Dict[str, Any]], int]:
        """Load checkpoint to resume from crash. Returns (start_batch_idx, results, processed_images)."""
        if not self.checkpoint_file.exists():
            return 0, [], 0
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            results_file = self.checkpoint_dir / f"results_{self.split}.json"
            if not results_file.exists():
                logger.warning("Checkpoint metadata found but results file missing. Starting from scratch.")
                return 0, [], 0
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            start_batch = checkpoint_data["batch_idx"] + 1  # Resume from next batch
            processed_images = checkpoint_data["processed_images"]
            
            logger.info(f"Resuming from checkpoint: batch {start_batch}, {processed_images} images processed, {len(results)} QA pairs")
            return start_batch, results, processed_images
            
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
    
    def build(self):
        """Build the HuggingFace dataset."""
        from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage
        
        logger.info(f"Building HuggingFace dataset for {self.dataset_name} {self.split}")
        
        # Create data loader
        data_loader = DataLoader(
            self.dataset_loader,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=self.num_workers,
            prefetch_factor=1,
            persistent_workers=False,
        )

        # Load checkpoint if available (unless force restart)
        force_restart = bool(os.getenv("GRAID_FORCE_RESTART")) or self.force
        if force_restart:
            logger.info("Force restart requested - removing existing checkpoints and starting from scratch")
            self._cleanup_checkpoint()  # Remove existing checkpoints first
            start_batch_idx, results, processed_images = 0, [], 0
        else:
            start_batch_idx, results, processed_images = self._load_checkpoint()
        
        # Skip already processed batches if resuming
        if start_batch_idx > 0:
            logger.info(f"Skipping first {start_batch_idx} batches (already processed)")
        
        # Track starting state for this run  
        results_at_start = len(results)
        
        # Check for early stopping via environment variable
        max_batches = None
        try:
            max_batches_env = os.getenv("GRAID_MAX_BATCHES")
                max_batches = int(max_batches_env) if max_batches_env else None
            except Exception:
            pass
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            # Skip batches that were already processed (resuming from checkpoint)
            if batch_idx < start_batch_idx:
                continue
                
            # Early stopping for testing
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(f"Stopping early after {batch_idx} batches (GRAID_MAX_BATCHES={max_batches})")
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

            # Process each image in the batch
            batch_results: List[Dict[str, Any]] = []
            batch_timings: Dict[str, tuple[float, int]] = {} if self.profile_questions else {}
            
            # Prepare batch data for parallel processing
            batch_data = []
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
                            logger.debug(f"Filtered out detection of class '{detection.label}' (not in allowable set)")
                    detections = filtered_detections

                # Extract source_id from batch sample
                if isinstance(batch[j], dict) and "name" in batch[j]:
                    source_id = batch[j]["name"]
                else:
                    source_id = f"{self.dataset_name}_{batch_idx}_{j}"
                
                # Store current example for filename inference
                self._current_example = batch[j] if isinstance(batch[j], dict) else {"name": source_id}
                
                # Prepare data for processing (parallel or sequential)
                base_image_index = batch_idx * self.batch_size
                batch_data.append((pil_image, detections, source_id, base_image_index, j))
            
            # Process batch data (parallel or sequential)
            if self.qa_workers > 1 and len(batch_data) > 1:
                logger.debug(f"Processing batch with {self.qa_workers} workers")
                # Parallel processing with order preservation
                with ThreadPoolExecutor(max_workers=self.qa_workers) as executor:
                    batch_results_raw = list(executor.map(self._qa_for_image_threadsafe, batch_data))
            else:
                logger.debug("Processing batch sequentially")
                # Sequential processing 
                batch_results_raw = []
                for args in batch_data:
                    pil_image, detections, source_id, base_image_index, j = args
                    image_index = base_image_index + j
                    self._current_example = batch[j] if isinstance(batch[j], dict) else {"name": source_id}
                    try:
                        ret = self._qa_for_image(pil_image, detections, source_id, image_index)
                        batch_results_raw.append(ret)
                    except Exception as e:
                        logger.error(f"Error processing image {source_id}: {e}")
                        # Add empty result to maintain order
                    if self.profile_questions:
                            batch_results_raw.append(([], {}))
                    else:
                            batch_results_raw.append([])
            
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
                    logger.warning(f"Unexpected return type from QA processing: {type(ret)}")

            # Add batch results to main results
            results.extend(batch_results)
            processed_images += len(batch)
            
            # Update per-question counts
            for item in batch_results:
                try:
                    qtype = item.get("question_type")
                        if qtype:
                            self._question_counts[qtype] = self._question_counts.get(qtype, 0) + 1
                except Exception:
                    pass
            
            # Merge batch timings into builder-level aggregation
            if self.profile_questions and batch_timings:
                for k, (t, n) in batch_timings.items():
                    T, N = self._question_timings.get(k, (0.0, 0))
                    self._question_timings[k] = (T + t, N + n)

            # Periodic progress log
            if batch_idx % 10 == 0:
                logger.info(f"Processed {processed_images} images, generated {len(results)} QA pairs")
            
            # Save checkpoint every save_steps batches
            if self.save_steps > 0 and (batch_idx + 1) % self.save_steps == 0:
                self._save_checkpoint(batch_idx, results, processed_images)
            
            # Early stop on num_samples (0 or None means process all)
            if self.num_samples is not None and self.num_samples > 0 and processed_images >= int(self.num_samples):
                logger.info(f"Reached num_samples={self.num_samples}. Stopping further processing.")
                break

        # Create final dataset
        if not results:
            logger.warning("No question-answer pairs generated!")
            raise RuntimeError("Dataset generation failed - no QA pairs were generated")
        
        # Debug: Check the structure of the first few results
        logger.debug(f"Total results: {len(results)}")
        if results:
            logger.debug(f"First result keys: {list(results[0].keys())}")
            logger.debug(f"First result annotations type: {type(results[0].get('annotations', None))}")
            if results[0].get('annotations'):
                ann = results[0]['annotations'][0] if results[0]['annotations'] else None
                if ann:
                    logger.debug(f"First annotation keys: {list(ann.keys())}")
                    logger.debug(f"First annotation bbox: {ann.get('bbox')} (type: {type(ann.get('bbox'))})")
        
        # Validate results structure
        for i, result in enumerate(results[:5]):  # Check first 5 results
            if not isinstance(result, dict):
                logger.error(f"Result {i} is not a dict: {type(result)}")
                continue
            
            required_keys = ["image", "annotations", "question", "answer", "question_type", "source_id"]
            for key in required_keys:
                if key not in result:
                    logger.error(f"Result {i} missing key: {key}")
                    
            # Validate annotations structure
            annotations = result.get('annotations', [])
            if not isinstance(annotations, list):
                logger.error(f"Result {i} annotations is not a list: {type(annotations)}")
            else:
                for j, ann in enumerate(annotations):
                    if not isinstance(ann, dict):
                        logger.error(f"Result {i} annotation {j} is not a dict: {type(ann)}")
                    else:
                        bbox = ann.get('bbox')
                        if bbox is not None and not isinstance(bbox, list):
                            logger.error(f"Result {i} annotation {j} bbox is not a list: {type(bbox)}")
        
        # Simplified approach - let HuggingFace infer the features automatically
        try:
            # First create without explicit features to let HF infer
                dataset = Dataset.from_list(results)
            # Then cast the image column to HFImage with decode=False
            dataset = dataset.cast_column("image", HFImage(decode=False))
            except Exception as e:
                logger.error(f"Failed to create dataset from results: {e}")
                raise

        # Add metadata
        metadata = self._create_metadata()
        dataset.info.description = f"Object detection QA dataset for {self.dataset_name}"
        dataset.info.features = dataset.features
        dataset.info.version = "1.0.0"
        dataset.info.config_name = json.dumps(metadata)

        # Create DatasetDict
        dataset_dict = DatasetDict({self.split: dataset})

        logger.info(f"Generated {len(dataset)} question-answer pairs")

        # Clean up checkpoint files on successful completion
        self._cleanup_checkpoint()
        
        # Log profiling information
        if self.profile_questions and self._question_timings:
            items = [(k, t / max(n, 1), n) for k, (t, n) in self._question_timings.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            top = ", ".join([f"{k}: avg {avg:.4f}s over {n}" for k, avg, n in items[:5]])
            logger.info(f"[PROFILE] Top slow questions (avg): {top}")

        # Log per-question counts
        if self._question_counts:
                pairs = sorted(self._question_counts.items(), key=lambda kv: kv[1], reverse=True)
                summary = ", ".join([f"{k}={v}" for k, v in pairs])
                logger.info(f"Per-question counts: {summary}")

        return dataset_dict

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
                    "model_name": getattr(model, "model_name", str(model.__class__.__name__)),
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
    save_path: Optional[str] = None,
    upload_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
    num_samples: Optional[int] = None,
    use_original_filenames: bool = True,
    filename_prefix: str = "img",
    force: bool = False,
):
    """
    Generate a HuggingFace dataset for object detection question-answering.
    
    Args:
        dataset_name: Name of the dataset ("bdd", "nuimage", "waymo")
        split: Dataset split ("train", "val", "test")
        models: List of model objects for inference (optional, uses ground truth if None)
        use_wbf: Whether to use Weighted Box Fusion ensemble (default: False)
        wbf_config: Configuration for WBF ensemble (optional)
        conf_threshold: Confidence threshold for filtering detections (default: 0.2)
        batch_size: Batch size for processing (default: 1)
        device: Device to use for inference (optional, auto-detected if None)
        allowable_set: List of allowed object classes (optional, uses all if None)
        question_configs: List of question configuration dictionaries (optional)
        num_workers: Number of data loading workers (default: 4)
        qa_workers: Number of QA generation workers (default: 4)
        save_steps: Save checkpoint every N batches for crash recovery (default: 50)
        save_path: Path to save dataset (optional)
        upload_to_hub: Whether to upload to HuggingFace Hub (default: False)
        hub_repo_id: HuggingFace Hub repository ID (required if upload_to_hub=True)
        hub_private: Whether to make Hub repository private (default: False)
        num_samples: Maximum number of samples to process (0 or None = process all)
        use_original_filenames: Whether to keep original filenames (default: True)
        filename_prefix: Prefix for generated filenames if not using originals (default: "img")
        force: Force restart from scratch, ignoring existing checkpoints (default: False)
        
    Returns:
        DatasetDict: Generated HuggingFace dataset
    """
    from datasets import DatasetDict

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
        create_repo(hub_repo_id, repo_type="dataset", private=hub_private, exist_ok=True)
        
        # Upload images and directory structure using upload_large_folder
        if save_path:
            logger.info(f"Uploading dataset files from {save_path} to Hub repository...")
            try:
                upload_large_folder(
                        repo_id=hub_repo_id,
                        repo_type="dataset",
                    folder_path=str(save_path),
                    )
                logger.info("Image and directory upload completed successfully")
                except Exception as e:
                logger.error(f"Failed to upload files to Hub: {e}")
                    raise
        
        # Cast image column and push dataset
        try:
            from datasets import Image as HFImage
            for split_name in dataset_dict.keys():
                dataset_dict[split_name] = dataset_dict[split_name].cast_column("image", HFImage(decode=False))
        except Exception as e:
            logger.warning(f"Failed to cast image column before push_to_hub: {e}")

        # Push dataset with proper settings
        dataset_dict.push_to_hub(
            repo_id=hub_repo_id,
            private=hub_private,
            embed_external_files=False,  # Critical: no byte duplication
            commit_message=f"Upload {dataset_name} {split} dataset",
            max_shard_size="100MB",
        )
        logger.info(f"Dataset pushed to HuggingFace Hub: {hub_repo_id}")

    return dataset_dict


# Compatibility functions for existing code
def list_available_questions() -> Dict[str, Dict[str, Any]]:
    """List available question types, their descriptions, and parameters."""
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
            "parameters": {}  # Would need to be populated based on inspection
        }
    
    return question_info


def interactive_question_selection() -> List[Dict[str, Any]]:
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