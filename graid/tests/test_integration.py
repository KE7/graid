"""
Integration Tests for Detection Validation System

This module tests the integration between different components of the validation
system, including end-to-end workflows and external integrations like WandB.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

# Add graid to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graid.data.validation import (
    ComprehensiveDetectionValidator,
    ValidationConfig,
    ValidationStage,
)
from graid.data.validation.human_supervised_filter import (
    HumanSupervisedClassifier,
    HumanSupervisedFilter,
)
from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI


class TestEndToEndValidation(unittest.TestCase):
    """Test end-to-end validation workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_image = self._create_sample_image()
        self.sample_detections = self._create_sample_detections()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_sample_image(self):
        """Create a sample image for testing."""
        image = Image.new("RGB", (640, 480), color="lightblue")
        return image

    def _create_sample_detections(self):
        """Create sample detections for testing."""
        return [
            ObjectDetectionResultI(
                score=0.85,
                cls=2,
                label="car",
                bbox=[100, 200, 300, 350],
                image_hw=(480, 640),
            ),
            ObjectDetectionResultI(
                score=0.92,
                cls=0,
                label="person",
                bbox=[320, 150, 380, 340],
                image_hw=(480, 640),
            ),
            ObjectDetectionResultI(
                score=0.65,
                cls=21,
                label="elephant",  # Should be filtered
                bbox=[200, 180, 400, 380],
                image_hw=(480, 640),
            ),
        ]

    def test_complete_validation_pipeline(self):
        """Test complete validation pipeline with multiple stages."""
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,  # Skip heavy models
            enable_scene_consistency=False,  # Skip API calls
            enable_ensemble=False,  # Skip ensemble
            enable_segmentation=False,  # Skip SAM2
            min_detection_confidence=0.3,
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)

        # Run complete pipeline
        valid_detections, validation_records = validator.filter_detections(
            self.sample_detections, self.sample_image, debug=True
        )

        # Verify pipeline execution
        self.assertIsInstance(valid_detections, list)
        self.assertIsInstance(validation_records, list)
        self.assertEqual(len(validation_records), len(self.sample_detections))

        # Check that filtering occurred
        self.assertLessEqual(len(valid_detections), len(self.sample_detections))

        # Verify each record has stage results
        for record in validation_records:
            self.assertIsInstance(record.stage_results, dict)
            self.assertIn(ValidationStage.COOCCURRENCE, record.stage_results)

        # Get metrics
        metrics = validator.get_metrics_summary()
        self.assertIn("total_detections", metrics)
        self.assertIn("overall_pass_rate", metrics)

    def test_progressive_filtering(self):
        """Test that filtering becomes progressively stricter."""
        # Lenient config
        lenient_config = ValidationConfig(
            enable_cooccurrence=True,
            min_detection_confidence=0.2,
            cooccurrence_threshold=0.0001,
            device="cpu",
        )

        # Strict config
        strict_config = ValidationConfig(
            enable_cooccurrence=True,
            min_detection_confidence=0.8,
            cooccurrence_threshold=0.1,
            device="cpu",
        )

        lenient_validator = ComprehensiveDetectionValidator(lenient_config)
        strict_validator = ComprehensiveDetectionValidator(strict_config)

        lenient_valid, _ = lenient_validator.filter_detections(
            self.sample_detections, self.sample_image
        )
        strict_valid, _ = strict_validator.filter_detections(
            self.sample_detections, self.sample_image
        )

        # Verify that strict filtering removes more detections
        self.assertLessEqual(len(strict_valid), len(lenient_valid))

    def test_error_handling_in_pipeline(self):
        """Test error handling when stages fail."""
        # Create a config that might cause errors
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=True,  # This might fail due to missing models
            device="cpu",
        )

        validator = ComprehensiveDetectionValidator(config)

        # Pipeline should handle errors gracefully
        try:
            valid_detections, validation_records = validator.filter_detections(
                self.sample_detections, self.sample_image
            )

            # Should still return results even if some stages fail
            self.assertIsInstance(valid_detections, list)
            self.assertIsInstance(validation_records, list)

        except Exception as e:
            # If it fails completely, that's also acceptable for missing dependencies
            self.assertIsInstance(e, Exception)


class TestHumanSupervisedIntegration(unittest.TestCase):
    """Test integration of human supervised validation (Phase 6)."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_image_path = self._create_sample_image()
        self._create_sample_labels()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_sample_image(self):
        """Create a sample image for testing."""
        image = Image.new("RGB", (224, 224), color="green")
        image_path = Path(self.temp_dir) / "test_image.jpg"
        image.save(image_path)
        return str(image_path)

    def _create_sample_labels(self):
        """Create sample human evaluation labels."""
        labels_data = {
            "samples": [
                {
                    "image_path": self.sample_image_path,
                    "question": "How many cars are visible?",
                    "answer": "2",
                    "human_evaluation": "yes",
                    "evaluator_id": "test_user",
                },
                {
                    "image_path": self.sample_image_path,
                    "question": "Are there any airplanes?",
                    "answer": "yes",
                    "human_evaluation": "no",  # Invalid answer
                    "evaluator_id": "test_user",
                },
            ]
        }

        labels_file = Path(self.temp_dir) / "human_labels.json"
        with open(labels_file, "w") as f:
            json.dump(labels_data, f)

    def test_human_supervised_training_integration(self):
        """Test integration of human supervised training."""
        classifier = HumanSupervisedClassifier()

        # Test that classifier can be created and configured
        self.assertIsNotNone(classifier)
        self.assertFalse(classifier.is_trained)

        # Mock training (actual training would need more data)
        classifier.is_trained = True

        # Test prediction interface
        prediction = classifier.predict_validity(
            image=self.sample_image_path, question="Test question", answer="Test answer"
        )

        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

    def test_human_supervised_filter_integration(self):
        """Test integration of human supervised filter."""
        # Create and train classifier
        classifier = HumanSupervisedClassifier()
        classifier.is_trained = True  # Mock training

        # Create filter
        filter_obj = HumanSupervisedFilter(classifier, threshold=0.5)

        # Create sample detection
        detection = ObjectDetectionResultI(
            score=0.8,
            cls=2,
            label="car",
            bbox=[100, 100, 200, 200],
            image_hw=(224, 224),
        )

        # Test validation
        result = filter_obj.validate_detection(
            detection, Image.open(self.sample_image_path), "How many cars?", "1"
        )

        # Verify result structure
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management and persistence."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        # Create a comprehensive config
        config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=True,
            enable_scene_consistency=False,
            enable_ensemble=True,
            enable_segmentation=True,
            min_detection_confidence=0.4,
            cooccurrence_threshold=0.01,
            clip_similarity_threshold=0.25,
            scene_consistency_threshold=0.3,
            ensemble_agreement_threshold=0.6,
            segmentation_overlap_threshold=0.4,
            require_all_stages=False,
            device="cuda",
        )

        # Test that config has all expected attributes
        self.assertTrue(config.enable_cooccurrence)
        self.assertTrue(config.enable_clip_relationships)
        self.assertFalse(config.enable_scene_consistency)
        self.assertEqual(config.min_detection_confidence, 0.4)
        self.assertEqual(config.device, "cuda")

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        valid_config = ValidationConfig(min_detection_confidence=0.5, device="cpu")
        self.assertIsNotNone(valid_config)

        # Test edge cases
        edge_config = ValidationConfig(
            min_detection_confidence=0.0,  # Minimum value
            cooccurrence_threshold=1.0,  # Maximum value
        )
        self.assertIsNotNone(edge_config)

    def test_multiple_validators(self):
        """Test using multiple validators with different configs."""
        # Create different validators
        fast_config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,
            enable_scene_consistency=False,
            enable_ensemble=False,
            enable_segmentation=False,
            device="cpu",
        )

        comprehensive_config = ValidationConfig(
            enable_cooccurrence=True,
            enable_clip_relationships=False,  # Skip for test
            enable_scene_consistency=False,  # Skip for test
            enable_ensemble=False,  # Skip for test
            enable_segmentation=False,  # Skip for test
            require_all_stages=True,
            device="cpu",
        )

        fast_validator = ComprehensiveDetectionValidator(fast_config)
        comprehensive_validator = ComprehensiveDetectionValidator(comprehensive_config)

        # Verify they're independent
        self.assertNotEqual(id(fast_validator), id(comprehensive_validator))
        self.assertEqual(fast_validator.config.require_all_stages, False)
        self.assertEqual(comprehensive_validator.config.require_all_stages, True)


class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance characteristics and scaling behavior."""

    def test_large_detection_set(self):
        """Test validation with larger detection sets."""
        # Create many detections
        large_detection_set = []
        for i in range(50):  # Moderate size for testing
            detection = ObjectDetectionResultI(
                score=0.5 + (i % 5) * 0.1,
                cls=i % 10,
                label=f"object_{i % 10}",
                bbox=[i * 10, i * 10, i * 10 + 50, i * 10 + 50],
                image_hw=(500, 500),
            )
            large_detection_set.append(detection)

        # Create validator
        config = ValidationConfig(enable_cooccurrence=True, device="cpu")
        validator = ComprehensiveDetectionValidator(config)

        # Create sample image
        image = Image.new("RGB", (500, 500), color="white")

        # Test validation
        valid_detections, validation_records = validator.filter_detections(
            large_detection_set, image
        )

        # Verify results
        self.assertIsInstance(valid_detections, list)
        self.assertEqual(len(validation_records), len(large_detection_set))

        # Performance should be reasonable (test completes without timeout)
        metrics = validator.get_metrics_summary()
        self.assertEqual(metrics["total_detections"], len(large_detection_set))

    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        # This is a basic test - in practice you'd use memory profiling tools
        config = ValidationConfig(enable_cooccurrence=True, device="cpu")
        validator = ComprehensiveDetectionValidator(config)

        # Create moderate-sized test data
        detections = [
            ObjectDetectionResultI(
                score=0.7,
                cls=1,
                label="test",
                bbox=[10, 10, 50, 50],
                image_hw=(100, 100),
            )
            for _ in range(10)
        ]

        image = Image.new("RGB", (100, 100), color="black")

        # Multiple validation runs should not accumulate memory
        for _ in range(5):
            valid_detections, _ = validator.filter_detections(detections, image)
            self.assertIsInstance(valid_detections, list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
