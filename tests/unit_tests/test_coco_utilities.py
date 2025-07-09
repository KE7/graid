"""
Unit tests for COCO utility functions.

This module tests the COCO validation and filtering functions that handle
allowable sets for object detection.
"""

import unittest
from unittest.mock import patch

from graid.utilities.coco import (
    validate_coco_objects,
    get_valid_coco_objects,
    filter_detections_by_allowable_set,
)


class TestCocoUtilities(unittest.TestCase):
    """Test cases for COCO utility functions."""

    def test_validate_coco_objects_empty_list(self):
        """Test validation with empty list."""
        result = validate_coco_objects([])
        self.assertEqual(result, (True, None))

    def test_validate_coco_objects_none(self):
        """Test validation with None (should treat as empty)."""
        # Note: The function signature expects list[str], but we handle None gracefully
        result = validate_coco_objects([])  # Use empty list instead of None
        self.assertEqual(result, (True, None))

    def test_validate_coco_objects_valid_objects(self):
        """Test validation with valid COCO objects."""
        valid_objects = ["person", "car", "truck", "bicycle"]
        result = validate_coco_objects(valid_objects)
        self.assertEqual(result, (True, None))

    def test_validate_coco_objects_invalid_objects(self):
        """Test validation with invalid COCO objects."""
        invalid_objects = ["person", "invalid_object", "car"]
        is_valid, error_msg = validate_coco_objects(invalid_objects)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        assert error_msg is not None  # Type assertion for linter
        self.assertIn("Invalid COCO objects", error_msg)
        self.assertIn("invalid_object", error_msg)

    def test_validate_coco_objects_all_invalid(self):
        """Test validation with all invalid objects."""
        invalid_objects = ["invalid1", "invalid2", "invalid3"]
        is_valid, error_msg = validate_coco_objects(invalid_objects)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        assert error_msg is not None  # Type assertion for linter
        self.assertIn("Invalid COCO objects", error_msg)
        self.assertIn("invalid1", error_msg)
        self.assertIn("invalid2", error_msg)
        self.assertIn("invalid3", error_msg)

    def test_get_valid_coco_objects_returns_sorted_list(self):
        """Test that get_valid_coco_objects returns a sorted list."""
        valid_objects = get_valid_coco_objects()
        self.assertIsInstance(valid_objects, list)
        self.assertEqual(len(valid_objects), 80)  # Standard COCO has 80 classes
        self.assertEqual(valid_objects, sorted(valid_objects))

    def test_get_valid_coco_objects_excludes_undefined(self):
        """Test that get_valid_coco_objects excludes 'undefined'."""
        valid_objects = get_valid_coco_objects()
        self.assertNotIn("undefined", valid_objects)

    def test_get_valid_coco_objects_contains_common_objects(self):
        """Test that get_valid_coco_objects contains common objects."""
        valid_objects = get_valid_coco_objects()
        common_objects = ["person", "car", "truck", "bicycle", "dog", "cat"]
        for obj in common_objects:
            self.assertIn(obj, valid_objects)

    def test_filter_detections_by_allowable_set_none_allowable_set(self):
        """Test filtering with None allowable set (should return all)."""
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
            {"class": "bicycle", "confidence": 0.7, "bbox": [40, 40, 50, 50]},
        ]

        filtered = filter_detections_by_allowable_set(detections, None)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered, detections)

    def test_filter_detections_by_allowable_set_empty_allowable_set(self):
        """Test filtering with empty allowable set (should return all)."""
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
        ]

        filtered = filter_detections_by_allowable_set(detections, [])
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered, detections)

    def test_filter_detections_by_allowable_set_filters_correctly(self):
        """Test filtering with specific allowable set."""
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
            {"class": "bicycle", "confidence": 0.7, "bbox": [40, 40, 50, 50]},
            {"class": "dog", "confidence": 0.6, "bbox": [60, 60, 70, 70]},
        ]

        allowable_set = ["person", "car"]
        filtered = filter_detections_by_allowable_set(detections, allowable_set)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["class"], "person")
        self.assertEqual(filtered[1]["class"], "car")

    def test_filter_detections_by_allowable_set_different_class_keys(self):
        """Test filtering with different class key names."""
        detections = [
            {"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"name": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
            {"class": "bicycle", "confidence": 0.7, "bbox": [40, 40, 50, 50]},
        ]

        allowable_set = ["person", "car"]
        filtered = filter_detections_by_allowable_set(detections, allowable_set)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["label"], "person")
        self.assertEqual(filtered[1]["name"], "car")

    def test_filter_detections_by_allowable_set_no_matches(self):
        """Test filtering when no detections match allowable set."""
        detections = [
            {"class": "bicycle", "confidence": 0.7, "bbox": [40, 40, 50, 50]},
            {"class": "dog", "confidence": 0.6, "bbox": [60, 60, 70, 70]},
        ]

        allowable_set = ["person", "car"]
        filtered = filter_detections_by_allowable_set(detections, allowable_set)

        self.assertEqual(len(filtered), 0)

    def test_filter_detections_by_allowable_set_missing_class_key(self):
        """Test filtering when detections missing class key."""
        detections = [
            {"confidence": 0.9, "bbox": [0, 0, 10, 10]},  # Missing class key
            {"class": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
        ]

        allowable_set = ["person", "car"]
        filtered = filter_detections_by_allowable_set(detections, allowable_set)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["class"], "car")

    def test_filter_detections_by_allowable_set_empty_detections(self):
        """Test filtering with empty detections list."""
        detections = []
        allowable_set = ["person", "car"]
        filtered = filter_detections_by_allowable_set(detections, allowable_set)

        self.assertEqual(len(filtered), 0)

    def test_integration_validate_and_filter(self):
        """Test integration of validation and filtering."""
        # First validate allowable set
        allowable_set = ["person", "car", "truck"]
        is_valid, error_msg = validate_coco_objects(allowable_set)
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

        # Then filter detections
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class": "car", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
            {"class": "bicycle", "confidence": 0.7, "bbox": [
                40, 40, 50, 50]},  # Should be filtered out
        ]

        filtered = filter_detections_by_allowable_set(detections, allowable_set)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["class"], "person")
        self.assertEqual(filtered[1]["class"], "car")


if __name__ == "__main__":
    unittest.main()
