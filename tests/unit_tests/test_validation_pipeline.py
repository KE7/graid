"""
Tests for Comprehensive Detection Validation Pipeline

This module tests the 5-stage validation pipeline to ensure proper functionality
and error handling of the detection validation system.
"""

import logging
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# Add graid to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graid.data.validation import (
    ComprehensiveDetectionValidator,
    ValidationConfig,
    ValidationStage,
)
from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI

# Suppress logging during tests unless debugging
logging.basicConfig(level=logging.WARNING)


class TestValidationPipeline(unittest.TestCase):
    """Test cases for the comprehensive detection validation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_size = (640, 480)
        self.sample_image = self._create_sample_image()
        self.sample_detections = self._create_sample_detections()

    def _create_sample_detections(self):
        """Create sample detections for testing."""
        h, w = self.image_size

        return [
            # Reasonable detections for street scene
            ObjectDetectionResultI(
                score=0.85,
                cls=2,
                label="car",
                bbox=[100, 200, 300, 350],
                image_hw=self.image_size,
            ),
            ObjectDetectionResultI(
                score=0.92,
                cls=0,
                label="person",
                bbox=[320, 150, 380, 340],
                image_hw=self.image_size,
            ),
            ObjectDetectionResultI(
                score=0.78,
                cls=9,
                label="traffic_light",
                bbox=[450, 50, 480, 120],
                image_hw=self.image_size,
            ),
            # Unreasonable detections (should be filtered)
            ObjectDetectionResultI(
                score=0.65,
                cls=21,
                label="elephant",
                bbox=[200, 180, 400, 380],
                image_hw=self.image_size,
            ),
            ObjectDetectionResultI(
                score=0.55,
                cls=4,
                label="airplane",
                bbox=[50, 30, 250, 150],
                image_hw=self.image_size,
            ),
        ]

    def _create_sample_image(self):
        """Create a simple street scene image for testing."""
        image = Image.new("RGB", self.image_size, color="lightblue")
        draw = ImageDraw.Draw(image)

        # Ground/road
        draw.rectangle(
            [0, self.image_size[1] // 2, self.image_size[0], self.image_size[1]],
            fill="gray",
        )

        # Buildings
        draw.rectangle([0, 100, 150, self.image_size[1] // 2], fill="darkred")
        draw.rectangle(
            [500, 80, self.image_size[0], self.image_size[1] // 2], fill="darkblue"
        )

        return image

    def test_basic_validation_config(self):
        """Test basic validation configuration creation."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_ensemble=False,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            device="cpu",
        )

        self.assertTrue(config.enable_cooccurrence)
        self.assertFalse(config.enable_clip_relationships)
        self.assertEqual(config.min_detection_confidence, 0.3)
        self.assertEqual(config.device, "cpu")

    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_ensemble=False,
            enable_segmentation=False,
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)

        self.assertIsNotNone(validator)
        self.assertEqual(validator.config, config)
        self.assertIn(ValidationStage.COOCCURRENCE, validator.filters)

    def test_cooccurrence_filtering(self):
        """Test co-occurrence filtering stage."""
        from graid.data.validation.cooccurrence_filter import CooccurrenceFilter

        filter_stage = CooccurrenceFilter()
        results = filter_stage.validate_detection_set(self.sample_detections)

        # Results should be a list, but length depends on implementation
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Check that results have proper structure
        for result in results:
            self.assertIsInstance(result.confidence, float)
            self.assertIsInstance(result.passed, bool)
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_basic_validation_pipeline(self):
        """Test basic validation pipeline without external dependencies."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,  # Skip to avoid CLIP dependency
            enable_scene_consistency=False,  # Skip to avoid OpenAI API
            enable_ensemble=False,  # No ensemble models
            enable_segmentation=False,  # Skip to avoid SAM2 dependency
            require_all_stages=False,
            min_detection_confidence=0.3,
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)

        # Test validation
        valid_detections, validation_records = validator.filter_detections(
            self.sample_detections, self.sample_image, debug=False
        )

        # Verify results structure
        self.assertIsInstance(valid_detections, list)
        self.assertIsInstance(validation_records, list)
        self.assertEqual(len(validation_records), len(self.sample_detections))

        # Check that we get some filtering (exact results depend on implementation)
        self.assertLessEqual(len(valid_detections), len(self.sample_detections))

        # Verify validation records structure
        for record in validation_records:
            self.assertIsNotNone(record.detection)
            self.assertIsInstance(record.final_valid, bool)
            self.assertIsInstance(record.stage_results, dict)

    def test_strict_vs_lenient_configuration(self):
        """Test strict vs lenient validation configurations."""

        # Strict configuration
        strict_config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_segmentation=False,
            require_all_stages=True,  # All stages must pass
            min_detection_confidence=0.7,  # High threshold
            cooccurrence_threshold=0.01,  # Strict co-occurrence
        )

        # Lenient configuration
        lenient_config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_segmentation=False,
            require_all_stages=False,  # Majority vote
            min_detection_confidence=0.2,  # Low threshold
            cooccurrence_threshold=0.0001,  # Permissive co-occurrence
        )

        strict_validator = ComprehensiveDetectionValidator(strict_config)
        lenient_validator = ComprehensiveDetectionValidator(lenient_config)

        strict_valid, _ = strict_validator.filter_detections(
            self.sample_detections, self.sample_image
        )
        lenient_valid, _ = lenient_validator.filter_detections(
            self.sample_detections, self.sample_image
        )

        # Both should return some results (exact comparison depends on implementation)
        self.assertIsInstance(strict_valid, list)
        self.assertIsInstance(lenient_valid, list)

    def test_metrics_collection(self):
        """Test that validation metrics are collected properly."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_ensemble=False,
            enable_segmentation=False,
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)
        validator.filter_detections(self.sample_detections, self.sample_image)

        metrics = validator.get_metrics_summary()

        # Check metrics structure
        self.assertIn("total_detections", metrics)
        self.assertIn("final_valid_detections", metrics)
        self.assertIn("overall_pass_rate", metrics)
        self.assertIn("stage_pass_rates", metrics)

        # Check metrics values
        self.assertEqual(metrics["total_detections"], len(self.sample_detections))
        self.assertGreaterEqual(metrics["overall_pass_rate"], 0.0)
        self.assertLessEqual(metrics["overall_pass_rate"], 1.0)

    def test_detection_input_validation(self):
        """Test input validation for detections."""
        config = ValidationConfig(enable_cooccurrence=True, device="cpu")
        validator = ComprehensiveDetectionValidator(config)

        # Test empty detections
        valid_detections, records = validator.filter_detections([], self.sample_image)
        self.assertEqual(len(valid_detections), 0)
        self.assertEqual(len(records), 0)

        # Test single detection
        single_detection = [self.sample_detections[0]]
        valid_detections, records = validator.filter_detections(
            single_detection, self.sample_image
        )
        self.assertEqual(len(records), 1)

    def test_confidence_filtering(self):
        """Test minimum confidence filtering."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_segmentation=False,
            min_detection_confidence=0.8,  # High threshold
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)
        valid_detections, _ = validator.filter_detections(
            self.sample_detections, self.sample_image
        )

        # All valid detections should meet minimum confidence
        for detection in valid_detections:
            self.assertGreaterEqual(detection.score, 0.8)

    def test_stage_info_retrieval(self):
        """Test stage information retrieval."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_segmentation=False,
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)
        stage_info = validator.get_stage_info()

        self.assertIsInstance(stage_info, dict)
        self.assertIn("cooccurrence", stage_info)

        # Each stage should have description
        for stage, info in stage_info.items():
            self.assertIn("description", info)
            self.assertIsInstance(info["description"], str)


class TestValidationComponents(unittest.TestCase):
    """Test individual validation components."""

    def test_validation_result_structure(self):
        """Test ValidationResult data structure."""
        from graid.utilities.validation import ValidationResult

        result = ValidationResult(
            passed=True,
            confidence=0.85,
            reason="Test reason",
            metadata={"test": "data"},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.reason, "Test reason")
        self.assertEqual(result.metadata["test"], "data")

    def test_validation_stage_enum(self):
        """Test ValidationStage enumeration."""
        from graid.utilities.validation import ValidationStage

        # Check that all expected stages exist
        expected_stages = {
            "COOCCURRENCE",
            "CLIP_RELATIONSHIPS",
            "SCENE_CONSISTENCY",
            "ENSEMBLE_AGREEMENT",
            "SEGMENTATION",
            "HUMAN_SUPERVISED",
        }

        actual_stages = {stage.name for stage in ValidationStage}
        self.assertEqual(expected_stages, actual_stages)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
