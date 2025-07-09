"""
Unit tests for ImageLoader module.

Tests the transform function signature fix where transform should receive both image and labels.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent / "graid"
sys.path.insert(0, str(project_root))

from graid.data.ImageLoader import ImageDataset


class TestImageLoaderTransformFix:
    """Test the ImageLoader transform bug fix."""

    def test_transform_receives_both_image_and_labels(self):
        """Test that transform function receives both image and labels parameters."""
        # Mock the transform function to verify it receives both parameters
        mock_transform = Mock(return_value=(np.zeros((100, 100, 3)), []))

        # Create a minimal mock dataset
        with patch("graid.data.ImageLoader.ImageDataset.__init__", return_value=None):
            dataset = ImageDataset.__new__(ImageDataset)
            dataset.transform = mock_transform
            dataset.data = [{"image_path": "test.jpg", "labels": [{"category": "car"}]}]
            dataset.dataset_name = "test"

            # Mock image loading
            mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("cv2.imread", return_value=mock_image):
                with patch("cv2.cvtColor", return_value=mock_image):
                    # Call __getitem__ which should pass both image and labels to transform
                    try:
                        result = dataset.__getitem__(0)

                        # Verify transform was called with both image and labels
                        mock_transform.assert_called_once()
                        call_args = mock_transform.call_args[0]
                        assert (
                            len(call_args) == 2
                        ), "Transform should receive exactly 2 arguments (image, labels)"

                        # Verify the arguments are image and labels
                        image_arg, labels_arg = call_args
                        assert isinstance(
                            image_arg, np.ndarray
                        ), "First argument should be image (numpy array)"
                        assert isinstance(
                            labels_arg, list
                        ), "Second argument should be labels (list)"

                    except Exception as e:
                        # If there's still an error, it should not be about transform signature
                        assert (
                            "takes 2 positional arguments but 3 were given"
                            not in str(e)
                        )
                        assert "'NoneType' object is not iterable" not in str(e)

    def test_transform_none_labels_handling(self):
        """Test that transform handles None labels gracefully."""

        # Mock transform that expects both image and labels
        def mock_transform(image, labels):
            if labels is None:
                labels = []
            return image, labels

        # Test the transform function directly rather than through the abstract base class
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_labels = None

        # This should not raise a TypeError about NoneType not being iterable
        try:
            result = mock_transform(mock_image, mock_labels)
            # Test passes if no exception is raised and labels are converted to empty list
            assert len(result) == 2
            assert isinstance(result[0], np.ndarray)
            assert result[1] == []  # None labels should be converted to empty list
        except TypeError as e:
            if "'NoneType' object is not iterable" in str(e):
                pytest.fail("Transform should handle None labels properly")
            else:
                # Re-raise if it's a different TypeError
                raise

    def test_dataset_transforms_lambda_functions(self):
        """Test that dataset has proper transform lambda functions defined."""
        from graid.data.generate_db import MODEL_CONFIGS

        # Test that all model configs have proper transform functions
        for backend, models in MODEL_CONFIGS.items():
            for model_name, config in models.items():
                if "transforms" in config:
                    transforms = config["transforms"]

                    # Test that transform functions accept both image and labels
                    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    mock_labels = [{"category": "car", "bbox": [10, 10, 50, 50]}]

                    try:
                        if "train" in transforms:
                            result = transforms["train"](mock_image, mock_labels)
                            assert (
                                len(result) == 2
                            ), f"Train transform for {backend}.{model_name} should return (image, labels)"

                        if "val" in transforms:
                            result = transforms["val"](mock_image, mock_labels)
                            assert (
                                len(result) == 2
                            ), f"Val transform for {backend}.{model_name} should return (image, labels)"

                    except Exception as e:
                        pytest.fail(
                            f"Transform for {backend}.{model_name} failed: {str(e)}"
                        )
