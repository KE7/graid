"""
Unit tests for dataset generation functionality.

This module tests the core dataset generation functionality including
the HuggingFaceDatasetBuilder class and generate_dataset function.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import numpy as np

from graid.data.generate_dataset import (
    HuggingFaceDatasetBuilder,
    generate_dataset,
    validate_model_config,
    validate_models_batch,
    validate_wbf_compatibility,
    list_available_models,
)


class TestHuggingFaceDatasetBuilder(unittest.TestCase):
    """Test cases for HuggingFaceDatasetBuilder class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        builder = HuggingFaceDatasetBuilder(
            dataset_name="bdd",
            split="val",
            models=[],
            conf_threshold=0.5,
            batch_size=1,
            device="cpu"
        )
        self.assertEqual(builder.dataset_name, "bdd")
        self.assertEqual(builder.split, "val")
        self.assertEqual(builder.conf_threshold, 0.5)
        self.assertEqual(builder.batch_size, 1)

    def test_init_invalid_dataset(self):
        """Test initialization with invalid dataset name."""
        with self.assertRaises(ValueError) as context:
            HuggingFaceDatasetBuilder(
                dataset_name="invalid_dataset",
                split="val",
                models=[]
            )
        self.assertIn("Unsupported dataset", str(context.exception))

    def test_init_with_allowable_set_valid(self):
        """Test initialization with valid allowable set."""
        builder = HuggingFaceDatasetBuilder(
            dataset_name="bdd",
            split="val",
            models=[],
            allowable_set=["person", "car", "truck"]
        )
        self.assertEqual(builder.allowable_set, ["person", "car", "truck"])

    @patch('graid.data.generate_dataset.validate_coco_objects')
    def test_init_with_allowable_set_invalid(self, mock_validate):
        """Test initialization with invalid allowable set."""
        mock_validate.return_value = (False, "Invalid objects")

        with self.assertRaises(ValueError) as context:
            HuggingFaceDatasetBuilder(
                dataset_name="bdd",
                split="val",
                models=[],
                allowable_set=["person", "invalid_object"]
            )
        self.assertIn("Invalid allowable_set", str(context.exception))

    def test_convert_image_to_pil_tensor(self):
        """Test converting tensor to PIL image."""
        builder = HuggingFaceDatasetBuilder(
            dataset_name="bdd",
            split="val",
            models=[]
        )

        # Create a dummy tensor (3, 224, 224) with values 0-1
        tensor = torch.rand(3, 224, 224)

        pil_image = builder._convert_image_to_pil(tensor)

        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.size, (224, 224))

    def test_convert_image_to_pil_numpy(self):
        """Test converting numpy array to PIL image."""
        builder = HuggingFaceDatasetBuilder(
            dataset_name="bdd",
            split="val",
            models=[]
        )

        # Create a dummy numpy array (224, 224, 3) with values 0-255
        numpy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        pil_image = builder._convert_image_to_pil(numpy_array)

        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.size, (224, 224))

    def test_create_metadata(self):
        """Test metadata creation."""
        builder = HuggingFaceDatasetBuilder(
            dataset_name="bdd",
            split="val",
            models=[],
            conf_threshold=0.3,
            batch_size=2,
            device="cpu"
        )

        metadata = builder._create_metadata()

        self.assertEqual(metadata["dataset_name"], "bdd")
        self.assertEqual(metadata["split"], "val")
        self.assertEqual(metadata["confidence_threshold"], 0.3)
        self.assertEqual(metadata["batch_size"], 2)
        self.assertEqual(metadata["device"], "cpu")
        self.assertIn("questions", metadata)
        self.assertIn("models", metadata)


class TestGenerateDataset(unittest.TestCase):
    """Test cases for generate_dataset function."""

    @patch('graid.data.generate_dataset.HuggingFaceDatasetBuilder')
    def test_generate_dataset_basic(self, mock_builder_class):
        """Test basic dataset generation."""
        mock_builder = Mock()
        mock_dataset = Mock()
        mock_builder.build.return_value = mock_dataset
        mock_builder_class.return_value = mock_builder

        result = generate_dataset(
            dataset_name="bdd",
            split="val",
            models=[],
            conf_threshold=0.5
        )

        # Verify builder was created with correct parameters
        mock_builder_class.assert_called_once()
        call_args = mock_builder_class.call_args
        self.assertEqual(call_args[1]["dataset_name"], "bdd")
        self.assertEqual(call_args[1]["split"], "val")
        self.assertEqual(call_args[1]["conf_threshold"], 0.5)

        # Verify build was called
        mock_builder.build.assert_called_once()
        self.assertEqual(result, mock_dataset)

    @patch('graid.data.generate_dataset.HuggingFaceDatasetBuilder')
    def test_generate_dataset_with_allowable_set(self, mock_builder_class):
        """Test dataset generation with allowable set."""
        mock_builder = Mock()
        mock_dataset = Mock()
        mock_builder.build.return_value = mock_dataset
        mock_builder_class.return_value = mock_builder

        allowable_set = ["person", "car", "truck"]
        result = generate_dataset(
            dataset_name="bdd",
            split="val",
            models=[],
            allowable_set=allowable_set
        )

        # Verify builder was created with allowable_set
        call_args = mock_builder_class.call_args
        self.assertEqual(call_args[1]["allowable_set"], allowable_set)


class TestValidationFunctions(unittest.TestCase):
    """Test cases for model validation functions."""

    def test_validate_model_config_valid_detectron(self):
        """Test model validation with valid Detectron model."""
        with patch('graid.data.generate_dataset.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            is_valid, error_msg = validate_model_config(
                backend="detectron",
                model_name="faster_rcnn_R_50_FPN_3x",
                device="cpu"
            )

            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
            mock_create.assert_called_once()

    def test_validate_model_config_invalid_backend(self):
        """Test model validation with invalid backend."""
        is_valid, error_msg = validate_model_config(
            backend="invalid_backend",
            model_name="some_model",
            device="cpu"
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        self.assertIn("Unsupported backend", error_msg)

    def test_validate_model_config_invalid_model_name(self):
        """Test model validation with invalid model name."""
        is_valid, error_msg = validate_model_config(
            backend="detectron",
            model_name="invalid_model",
            device="cpu"
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        self.assertIn("Detectron backend requires custom_config", error_msg)

    def test_validate_models_batch_valid(self):
        """Test batch validation with valid models."""
        model_configs = [
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "config": None
            },
            {
                "backend": "detectron",
                "model_name": "retinanet_R_101_FPN_3x",
                "config": None
            }
        ]

        with patch('graid.data.generate_dataset.validate_model_config') as mock_validate:
            mock_validate.return_value = (True, None)

            results = validate_models_batch(model_configs, device="cpu")

            self.assertEqual(len(results), 2)
            for key, (is_valid, error_msg) in results.items():
                self.assertTrue(is_valid)
                self.assertIsNone(error_msg)

    def test_validate_models_batch_mixed_results(self):
        """Test batch validation with mixed results."""
        model_configs = [
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "config": None
            },
            {
                "backend": "invalid_backend",
                "model_name": "some_model",
                "config": None
            }
        ]

        def mock_validate_side_effect(backend, model_name, config=None, device=None):
            if backend == "detectron":
                return (True, None)
            else:
                return (False, "Invalid backend")

        with patch('graid.data.generate_dataset.validate_model_config') as mock_validate:
            mock_validate.side_effect = mock_validate_side_effect

            results = validate_models_batch(model_configs, device="cpu")

            self.assertEqual(len(results), 2)
            # Check that we have both valid and invalid results
            valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
            invalid_count = sum(
                1 for is_valid, _ in results.values() if not is_valid)
            self.assertEqual(valid_count, 1)
            self.assertEqual(invalid_count, 1)

    def test_validate_wbf_compatibility_valid(self):
        """Test WBF compatibility validation with valid models."""
        model_configs = [
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "custom_config": {
                    "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                    "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                }
            },
            {
                "backend": "detectron",
                "model_name": "retinanet_R_101_FPN_3x",
                "custom_config": {
                    "config": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                    "weights": "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
                }
            }
        ]

        with patch('graid.data.generate_dataset.validate_models_batch') as mock_validate:
            with patch('graid.data.generate_dataset.create_model') as mock_create_model:
                with patch('graid.data.generate_dataset.WBF') as mock_wbf:
                    # Mock all the components
                    mock_validate.return_value = {
                        "model_0": (True, None),
                        "model_1": (True, None)
                    }

                    mock_model1 = Mock()
                    mock_model2 = Mock()
                    mock_create_model.side_effect = [mock_model1, mock_model2]

                    mock_wbf_instance = Mock()
                    mock_wbf.return_value = mock_wbf_instance
                    mock_wbf_instance.identify_for_image_batch.return_value = []

                    is_valid, error_msg = validate_wbf_compatibility(
                        model_configs, device="cpu")

                    self.assertTrue(is_valid)
                    self.assertIsNone(error_msg)

    def test_validate_wbf_compatibility_insufficient_models(self):
        """Test WBF compatibility validation with insufficient models."""
        model_configs = [
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "config": None
            }
        ]

        is_valid, error_msg = validate_wbf_compatibility(
            model_configs, device="cpu")

        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        self.assertIn("at least 2 models", error_msg)

    def test_validate_wbf_compatibility_invalid_models(self):
        """Test WBF compatibility validation with invalid models."""
        model_configs = [
            {
                "backend": "detectron",
                "model_name": "faster_rcnn_R_50_FPN_3x",
                "config": None
            },
            {
                "backend": "invalid_backend",
                "model_name": "some_model",
                "config": None
            }
        ]

        with patch('graid.data.generate_dataset.validate_models_batch') as mock_validate:
            mock_validate.return_value = {
                "model_0": (True, None),
                "model_1": (False, "Invalid backend")
            }

            is_valid, error_msg = validate_wbf_compatibility(
                model_configs, device="cpu")

            self.assertFalse(is_valid)
            self.assertIsNotNone(error_msg)
            self.assertIn("Some models failed validation", error_msg)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()

        self.assertIsInstance(models, dict)
        self.assertIn("detectron", models)
        self.assertIn("mmdetection", models)
        self.assertIn("ultralytics", models)

        # Check that each backend has a list of models
        for backend, model_list in models.items():
            self.assertIsInstance(model_list, list)
            self.assertGreater(len(model_list), 0)


if __name__ == "__main__":
    unittest.main()
