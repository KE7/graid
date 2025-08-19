"""
Unit tests for configuration support module.

This module tests the configuration classes and functions for dataset generation
including ModelConfig, WBFConfig, and DatasetGenerationConfig.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from graid.data.config_support import (
    ConfigurationError,
    DatasetGenerationConfig,
    ModelConfig,
    WBFConfig,
    create_example_config,
    load_config_from_dict,
    load_config_from_file,
    save_example_config,
    validate_config_file,
)


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig class."""

    def test_model_config_creation_valid(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            backend="detectron",
            model_name="faster_rcnn_R_50_FPN_3x",
            confidence_threshold=0.5,
            device="cpu",
            custom_config={
                "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            }
        )
        self.assertEqual(config.backend, "detectron")
        self.assertEqual(config.model_name, "faster_rcnn_R_50_FPN_3x")
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.device, "cpu")

    def test_model_config_invalid_backend(self):
        """Test creating ModelConfig with invalid backend."""
        with self.assertRaises(ConfigurationError) as context:
            ModelConfig(
                backend="invalid_backend",
                model_name="some_model",
            )
        self.assertIn("Unsupported backend", str(context.exception))

    def test_model_config_invalid_model_name(self):
        """Test creating ModelConfig with invalid model name."""
        with self.assertRaises(ConfigurationError) as context:
            ModelConfig(
                backend="detectron",
                model_name="invalid_model",
            )
        self.assertIn(
            "Custom config is required for detectron backend", str(context.exception))

    def test_model_config_custom_config_detectron(self):
        """Test ModelConfig with custom Detectron config."""
        custom_config = {
            "config": "path/to/config.yaml",
            "weights": "path/to/weights.pth"
        }
        config = ModelConfig(
            backend="detectron",
            model_name="custom_model",
            custom_config=custom_config
        )
        self.assertEqual(config.custom_config, custom_config)

    def test_model_config_custom_config_invalid_detectron(self):
        """Test ModelConfig with invalid custom Detectron config."""
        custom_config = {"config": "path/to/config.yaml"}  # Missing weights
        with self.assertRaises(ConfigurationError) as context:
            ModelConfig(
                backend="detectron",
                model_name="custom_model",
                custom_config=custom_config
            )
        self.assertIn("must have 'config' and 'weights' keys",
                      str(context.exception))

    def test_model_config_custom_config_mmdetection(self):
        """Test ModelConfig with custom MMDetection config."""
        custom_config = {
            "config": "path/to/config.py",
            "checkpoint": "path/to/checkpoint.pth"
        }
        config = ModelConfig(
            backend="mmdetection",
            model_name="custom_model",
            custom_config=custom_config
        )
        self.assertEqual(config.custom_config, custom_config)

    def test_model_config_custom_config_invalid_mmdetection(self):
        """Test ModelConfig with invalid custom MMDetection config."""
        custom_config = {"config": "path/to/config.py"}  # Missing checkpoint
        with self.assertRaises(ConfigurationError) as context:
            ModelConfig(
                backend="mmdetection",
                model_name="custom_model",
                custom_config=custom_config
            )
        self.assertIn("must have 'config' and 'checkpoint' keys",
                      str(context.exception))

    def test_model_config_to_dict(self):
        """Test ModelConfig to_dict method."""
        custom_config = {
            "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        }
        config = ModelConfig(
            backend="detectron",
            model_name="faster_rcnn_R_50_FPN_3x",
            confidence_threshold=0.7,
            device="cuda:0",
            custom_config=custom_config
        )
        result = config.to_dict()
        expected = {
            "backend": "detectron",
            "model_name": "faster_rcnn_R_50_FPN_3x",
            "custom_config": custom_config,
            "confidence_threshold": 0.7,
            "device": "cuda:0"
        }
        self.assertEqual(result, expected)


class TestWBFConfig(unittest.TestCase):
    """Test cases for WBFConfig class."""

    def test_wbf_config_creation_default(self):
        """Test creating WBFConfig with default values."""
        config = WBFConfig()
        self.assertEqual(config.iou_threshold, 0.55)
        self.assertEqual(config.skip_box_threshold, 0.0)
        self.assertIsNone(config.model_weights)

    def test_wbf_config_creation_custom(self):
        """Test creating WBFConfig with custom values."""
        config = WBFConfig(
            iou_threshold=0.6,
            skip_box_threshold=0.1,
            model_weights=[1.0, 2.0, 0.5]
        )
        self.assertEqual(config.iou_threshold, 0.6)
        self.assertEqual(config.skip_box_threshold, 0.1)
        self.assertEqual(config.model_weights, [1.0, 2.0, 0.5])

    def test_wbf_config_invalid_iou_threshold(self):
        """Test WBFConfig with invalid iou_threshold."""
        with self.assertRaises(ConfigurationError) as context:
            WBFConfig(iou_threshold=1.5)
        self.assertIn("iou_threshold must be between 0.0 and 1.0",
                      str(context.exception))

        with self.assertRaises(ConfigurationError) as context:
            WBFConfig(iou_threshold=-0.1)
        self.assertIn("iou_threshold must be between 0.0 and 1.0",
                      str(context.exception))

    def test_wbf_config_invalid_skip_box_threshold(self):
        """Test WBFConfig with invalid skip_box_threshold."""
        with self.assertRaises(ConfigurationError) as context:
            WBFConfig(skip_box_threshold=1.5)
        self.assertIn(
            "skip_box_threshold must be between 0.0 and 1.0", str(context.exception))

    def test_wbf_config_invalid_model_weights(self):
        """Test WBFConfig with invalid model_weights."""
        with self.assertRaises(ConfigurationError) as context:
            WBFConfig(model_weights=[1.0, -0.5, 2.0])
        self.assertIn("All model weights must be positive",
                      str(context.exception))

    def test_wbf_config_to_dict(self):
        """Test WBFConfig to_dict method."""
        config = WBFConfig(
            iou_threshold=0.7,
            skip_box_threshold=0.05,
            model_weights=[1.0, 1.5]
        )
        result = config.to_dict()
        expected = {
            "iou_threshold": 0.7,
            "skip_box_threshold": 0.05,
            "model_weights": [1.0, 1.5]
        }
        self.assertEqual(result, expected)


class TestDatasetGenerationConfig(unittest.TestCase):
    """Test cases for DatasetGenerationConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_config: ModelConfig = self._create_model_config()

    def _create_model_config(self) -> ModelConfig:
        """Create a model config for testing."""
        return ModelConfig(
            backend="detectron",
            model_name="faster_rcnn_R_50_FPN_3x",
            custom_config={
                "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            }
        )

    def test_dataset_generation_config_creation_valid(self):
        """Test creating a valid DatasetGenerationConfig."""
        config = DatasetGenerationConfig(
            dataset_name="bdd",
            split="val",
            models=[self.model_config],
            confidence_threshold=0.5,
            batch_size=2,
            device="cpu"
        )
        self.assertEqual(config.dataset_name, "bdd")
        self.assertEqual(config.split, "val")
        self.assertEqual(len(config.models), 1)
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.batch_size, 2)

    def test_dataset_generation_config_invalid_dataset(self):
        """Test creating DatasetGenerationConfig with invalid dataset."""
        with self.assertRaises(ConfigurationError) as context:
            DatasetGenerationConfig(
                dataset_name="invalid_dataset",
                split="val",
                models=[self.model_config]
            )
        self.assertIn("Unsupported dataset", str(context.exception))

    def test_dataset_generation_config_invalid_split(self):
        """Test creating DatasetGenerationConfig with invalid split."""
        with self.assertRaises(ConfigurationError) as context:
            DatasetGenerationConfig(
                dataset_name="bdd",
                split="invalid_split",
                models=[self.model_config]
            )
        self.assertIn("Invalid split", str(context.exception))

    def test_dataset_generation_config_wbf_insufficient_models(self):
        """Test WBF configuration with insufficient models."""
        with self.assertRaises(ConfigurationError) as context:
            DatasetGenerationConfig(
                dataset_name="bdd",
                split="val",
                models=[self.model_config],
                use_wbf=True
            )
        self.assertIn("WBF requires at least 2 models", str(context.exception))

    def test_dataset_generation_config_wbf_weight_mismatch(self):
        """Test WBF configuration with weight count mismatch."""
        model_config_2 = ModelConfig(
            backend="detectron",
            model_name="retinanet_R_101_FPN_3x",
            custom_config={
                "config": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                "weights": "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
            }
        )
        # Only 1 weight for 2 models
        wbf_config = WBFConfig(model_weights=[1.0])

        with self.assertRaises(ConfigurationError) as context:
            DatasetGenerationConfig(
                dataset_name="bdd",
                split="val",
                models=[self.model_config, model_config_2],
                use_wbf=True,
                wbf_config=wbf_config
            )
        self.assertIn("Number of model weights", str(context.exception))

    def test_dataset_generation_config_allowable_set_valid(self):
        """Test DatasetGenerationConfig with valid allowable set."""
        config = DatasetGenerationConfig(
            dataset_name="bdd",
            split="val",
            models=[self.model_config],
            allowable_set=["person", "car", "truck"]
        )
        self.assertEqual(config.allowable_set, ["person", "car", "truck"])

    def test_dataset_generation_config_allowable_set_invalid(self):
        """Test DatasetGenerationConfig with invalid allowable set."""
        with self.assertRaises(ConfigurationError) as context:
            DatasetGenerationConfig(
                dataset_name="bdd",
                split="val",
                models=[self.model_config],
                allowable_set=["person", "invalid_object"]
            )
        self.assertIn("Invalid COCO objects in allowable_set",
                      str(context.exception))

    def test_dataset_generation_config_to_dict(self):
        """Test DatasetGenerationConfig to_dict method."""
        config = DatasetGenerationConfig(
            dataset_name="bdd",
            split="val",
            models=[self.model_config],
            confidence_threshold=0.6,
            allowable_set=["person", "car"]
        )
        result = config.to_dict()

        self.assertEqual(result["dataset_name"], "bdd")
        self.assertEqual(result["split"], "val")
        self.assertEqual(len(result["models"]), 1)
        self.assertEqual(result["confidence_threshold"], 0.6)
        self.assertEqual(result["allowable_set"], ["person", "car"])


class TestConfigurationFunctions(unittest.TestCase):
    """Test cases for configuration utility functions."""

    def test_create_example_config(self):
        """Test creating example configuration."""
        example = create_example_config()

        self.assertIn("dataset_name", example)
        self.assertIn("split", example)
        self.assertIn("models", example)
        self.assertIn("allowable_set", example)
        self.assertIsInstance(example["models"], list)
        self.assertGreater(len(example["models"]), 0)

    def test_load_config_from_dict_valid(self):
        """Test loading configuration from valid dictionary."""
        config_dict = {
            "dataset_name": "bdd",
            "split": "val",
            "models": [
                {
                    "backend": "detectron",
                    "model_name": "faster_rcnn_R_50_FPN_3x",
                    "confidence_threshold": 0.5,
                    "custom_config": {
                        "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                        "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                    }
                }
            ],
            "allowable_set": ["person", "car"]
        }

        config = load_config_from_dict(config_dict)
        self.assertEqual(config.dataset_name, "bdd")
        self.assertEqual(config.split, "val")
        self.assertEqual(len(config.models), 1)
        self.assertEqual(config.allowable_set, ["person", "car"])

    def test_load_config_from_file_valid(self):
        """Test loading configuration from valid file."""
        config_dict = {
            "dataset_name": "bdd",
            "split": "val",
            "models": [
                {
                    "backend": "detectron",
                    "model_name": "faster_rcnn_R_50_FPN_3x",
                    "custom_config": {
                        "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                        "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config_from_file(temp_path)
            self.assertEqual(config.dataset_name, "bdd")
            self.assertEqual(config.split, "val")
            self.assertEqual(len(config.models), 1)
        finally:
            Path(temp_path).unlink()

    def test_load_config_from_file_invalid_json(self):
        """Test loading configuration from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with self.assertRaises(ConfigurationError) as context:
                load_config_from_file(temp_path)
            self.assertIn("Invalid JSON", str(context.exception))
        finally:
            Path(temp_path).unlink()

    def test_load_config_from_file_nonexistent(self):
        """Test loading configuration from non-existent file."""
        with self.assertRaises(ConfigurationError):
            load_config_from_file("nonexistent_file.json")

    def test_save_example_config(self):
        """Test saving example configuration to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_example_config(temp_path)

            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                loaded_config = json.load(f)

            self.assertIn("dataset_name", loaded_config)
            self.assertIn("models", loaded_config)
            self.assertIsInstance(loaded_config["models"], list)
        finally:
            Path(temp_path).unlink()

    def test_validate_config_file_valid(self):
        """Test validating a valid configuration file."""
        config_dict = {
            "dataset_name": "bdd",
            "split": "val",
            "models": [
                {
                    "backend": "detectron",
                    "model_name": "faster_rcnn_R_50_FPN_3x",
                    "custom_config": {
                        "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                        "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            is_valid, error_msg = validate_config_file(temp_path)
            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
        finally:
            Path(temp_path).unlink()

    def test_validate_config_file_invalid(self):
        """Test validating an invalid configuration file."""
        config_dict = {
            "dataset_name": "invalid_dataset",
            "split": "val",
            "models": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            is_valid, error_msg = validate_config_file(temp_path)
            self.assertFalse(is_valid)
            self.assertIsNotNone(error_msg)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
