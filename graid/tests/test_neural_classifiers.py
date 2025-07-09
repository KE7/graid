"""
Tests for Neural Classifiers and Phase 6 Multimodal Validation

This module tests the PyTorch-based neural network classifiers used in Phase 6
of the validation pipeline, including multimodal models combining vision and text.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np

# Add graid to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graid.data.validation.neural_classifiers import MultimodalValidationClassifier
from graid.data.validation.human_supervised_filter import (
    HumanSupervisedClassifier,
    HumanSupervisionDataLoader,
    HumanEvaluationSample
)


class TestMultimodalValidationClassifier(unittest.TestCase):
    """Test cases for the multimodal validation classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_image_path = self._create_sample_image()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_image(self):
        """Create a sample image for testing."""
        image = Image.new('RGB', (224, 224), color='blue')
        image_path = Path(self.temp_dir) / "test_image.jpg"
        image.save(image_path)
        return str(image_path)
    
    def test_classifier_initialization(self):
        """Test classifier initialization with default parameters."""
        classifier = MultimodalValidationClassifier(
            vit_model_name="google/vit-base-patch16-224",
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.device, "cpu")
        self.assertFalse(classifier.is_trained)
    
    def test_classifier_initialization_custom_params(self):
        """Test classifier initialization with custom parameters."""
        classifier = MultimodalValidationClassifier(
            hidden_dims=[128, 64],
            dropout_rate=0.5,
            learning_rate=0.001,
            n_epochs=10,
            batch_size=8,
            device="cpu"
        )
        
        self.assertEqual(classifier.hidden_dims, [128, 64])
        self.assertEqual(classifier.dropout_rate, 0.5)
        self.assertEqual(classifier.learning_rate, 0.001)
        self.assertEqual(classifier.n_epochs, 10)
        self.assertEqual(classifier.batch_size, 8)
    
    def test_training_interface(self):
        """Test training interface without actual training."""
        classifier = MultimodalValidationClassifier(device="cpu")
        
        # Mock training data
        image_paths = [self.sample_image_path] * 4
        questions = ["How many cars?", "What color is the car?", "Is there a person?", "Any traffic lights?"]
        answers = ["2", "blue", "yes", "no"]
        labels = [1, 1, 0, 1]  # 1 = valid, 0 = invalid
        
        # Test training call
        result = classifier.fit(
            image_paths=image_paths,
            questions=questions,
            answers=answers,
            labels=labels,
            validation_split=0.0
        )
        
        # Check training completed
        self.assertTrue(classifier.is_trained)
        self.assertIn('final_train_accuracy', result)
        self.assertIn('train_history', result)
    
    def test_prediction_interface(self):
        """Test prediction interface."""
        classifier = MultimodalValidationClassifier(device="cpu")
        
        # Train first
        image_paths = [self.sample_image_path] * 2
        questions = ["How many cars?", "What color?"]
        answers = ["2", "blue"]
        labels = [1, 0]
        
        classifier.fit(image_paths, questions, answers, labels)
        
        # Test prediction
        prediction = classifier.predict_single(
            image_path=self.sample_image_path,
            question="How many cars are there?",
            answer="3"
        )
        
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        classifier = MultimodalValidationClassifier(device="cpu")
        
        with self.assertRaises(ValueError):
            classifier.predict_single(
                image_path=self.sample_image_path,
                question="Test question",
                answer="Test answer"
            )
    
    def test_model_save_load_interface(self):
        """Test model save/load interface."""
        classifier = MultimodalValidationClassifier(device="cpu")
        
        # Train first
        image_paths = [self.sample_image_path] * 2
        questions = ["Question 1", "Question 2"]
        answers = ["Answer 1", "Answer 2"]
        labels = [1, 0]
        
        classifier.fit(image_paths, questions, answers, labels)
        
        # Test save
        model_path = Path(self.temp_dir) / "test_model.pth"
        classifier.save_model(model_path)
        
        # Test load
        new_classifier = MultimodalValidationClassifier(device="cpu")
        new_classifier.load_model(model_path)
        
        self.assertTrue(new_classifier.is_trained)
    
    def test_save_without_training(self):
        """Test that save fails without training."""
        classifier = MultimodalValidationClassifier(device="cpu")
        
        with self.assertRaises(ValueError):
            classifier.save_model("dummy_path.pth")


class TestHumanSupervisedClassifier(unittest.TestCase):
    """Test cases for the human supervised classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_image_path = self._create_sample_image()
        self._create_sample_labels_file()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_image(self):
        """Create a sample image for testing."""
        image = Image.new('RGB', (224, 224), color='red')
        image_path = Path(self.temp_dir) / "sample.jpg"
        image.save(image_path)
        return str(image_path)
    
    def _create_sample_labels_file(self):
        """Create a sample labels JSON file."""
        import json
        
        labels_data = {
            "samples": [
                {
                    "image_path": self.sample_image_path,
                    "question": "How many objects are visible?",
                    "answer": "3",
                    "human_evaluation": "yes",
                    "evaluator_id": "test_evaluator",
                    "confidence": 0.9
                },
                {
                    "image_path": self.sample_image_path,
                    "question": "Is there a plane in the image?",
                    "answer": "yes",
                    "human_evaluation": "no",  # This should be invalid
                    "evaluator_id": "test_evaluator",
                    "confidence": 0.8
                }
            ]
        }
        
        labels_file = Path(self.temp_dir) / "test_labels.json"
        with open(labels_file, 'w') as f:
            json.dump(labels_data, f)
        
        return str(labels_file)
    
    def test_classifier_initialization(self):
        """Test human supervised classifier initialization."""
        classifier = HumanSupervisedClassifier()
        
        self.assertIsNotNone(classifier)
        self.assertFalse(classifier.is_trained)
    
    def test_classifier_with_model_config(self):
        """Test classifier initialization with model config."""
        model_config = {
            "learning_rate": 0.001,
            "batch_size": 32
        }
        
        classifier = HumanSupervisedClassifier(
            model_config=model_config
        )
        
        self.assertEqual(classifier.model_config["learning_rate"], 0.001)
        self.assertEqual(classifier.model_config["batch_size"], 32)
    
    def test_data_loading_and_training(self):
        """Test loading data and training the classifier."""
        classifier = HumanSupervisedClassifier()
        
        # Mock the training process with minimal data
        try:
            results = classifier.load_and_train(
                labels_dir=self.temp_dir,
                test_size=0.0,  # No test split for small dataset
                val_size=0.0,   # No validation split
                min_samples_per_class=1,  # Reduce requirement
                image_base_path=None
            )
            
            self.assertTrue(classifier.is_trained)
            self.assertIn('training_results', results)
            
        except ValueError as e:
            # Expected for insufficient data
            self.assertIn("samples", str(e).lower())
    
    def test_prediction_interface(self):
        """Test prediction interface."""
        classifier = HumanSupervisedClassifier()
        
        # Mock training
        classifier.is_trained = True
        
        # Test prediction
        prediction = classifier.predict_validity(
            image=self.sample_image_path,
            question="Test question",
            answer="Test answer"
        )
        
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    def test_model_persistence(self):
        """Test model save/load functionality."""
        classifier = HumanSupervisedClassifier()
        classifier.is_trained = True  # Mock training
        
        # Test save
        model_path = Path(self.temp_dir) / "human_model.pkl"
        classifier.save_model(model_path)
        self.assertTrue(model_path.exists())
        
        # Test load
        new_classifier = HumanSupervisedClassifier()
        new_classifier.load_model(model_path)
        self.assertTrue(new_classifier.is_trained)


class TestHumanSupervisionDataLoader(unittest.TestCase):
    """Test cases for the human supervision data loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_data_files()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data_files(self):
        """Create test data files in different formats."""
        import json
        
        # New format
        new_format_data = {
            "samples": [
                {
                    "image_path": "image1.jpg",
                    "question": "Question 1",
                    "answer": "Answer 1",
                    "human_evaluation": "yes",
                    "evaluator_id": "evaluator1"
                }
            ]
        }
        
        new_format_file = Path(self.temp_dir) / "new_format.json"
        with open(new_format_file, 'w') as f:
            json.dump(new_format_data, f)
        
        # Legacy format
        legacy_format_data = {
            "evaluations": {
                "eval_1": {
                    "image_path": "image2.jpg",
                    "question": "Question 2",
                    "answer": "Answer 2",
                    "human_evaluation": "no",
                    "evaluator": "evaluator2"
                }
            }
        }
        
        legacy_format_file = Path(self.temp_dir) / "legacy_format.json"
        with open(legacy_format_file, 'w') as f:
            json.dump(legacy_format_data, f)
    
    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        loader = HumanSupervisionDataLoader(self.temp_dir)
        self.assertEqual(loader.labels_dir, Path(self.temp_dir))
    
    def test_load_manual_evaluations(self):
        """Test loading manual evaluations from JSON files."""
        loader = HumanSupervisionDataLoader(self.temp_dir)
        samples = loader.load_manual_evaluations()
        
        # Should load from both new and legacy format files
        self.assertGreater(len(samples), 0)
        
        # Check sample structure
        for sample in samples:
            self.assertIsInstance(sample, HumanEvaluationSample)
            self.assertIsInstance(sample.image_path, str)
            self.assertIsInstance(sample.question, str)
            self.assertIsInstance(sample.answer, str)
            self.assertIn(sample.human_evaluation, ["yes", "no"])
    
    def test_load_specific_datasets(self):
        """Test loading specific datasets."""
        loader = HumanSupervisionDataLoader(self.temp_dir)
        
        # Test loading specific dataset
        samples = loader.load_manual_evaluations(datasets=["new_format"])
        
        # Should only load from files containing "new_format"
        self.assertGreater(len(samples), 0)
    
    def test_load_from_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        loader = HumanSupervisionDataLoader("/nonexistent/path")
        samples = loader.load_manual_evaluations()
        
        # Should return empty list
        self.assertEqual(len(samples), 0)


class TestHumanEvaluationSample(unittest.TestCase):
    """Test cases for the HumanEvaluationSample data structure."""
    
    def test_sample_creation(self):
        """Test creating human evaluation samples."""
        sample = HumanEvaluationSample(
            image_path="test.jpg",
            question="Test question",
            answer="Test answer",
            human_evaluation="yes",
            evaluator_id="test_evaluator",
            confidence=0.9,
            metadata={"source": "test"}
        )
        
        self.assertEqual(sample.image_path, "test.jpg")
        self.assertEqual(sample.question, "Test question")
        self.assertEqual(sample.answer, "Test answer")
        self.assertEqual(sample.human_evaluation, "yes")
        self.assertEqual(sample.evaluator_id, "test_evaluator")
        self.assertEqual(sample.confidence, 0.9)
        self.assertEqual(sample.metadata["source"], "test")
    
    def test_sample_with_minimal_data(self):
        """Test creating samples with minimal required data."""
        sample = HumanEvaluationSample(
            image_path="minimal.jpg",
            question="Minimal question",
            answer="Minimal answer",
            human_evaluation="no"
        )
        
        self.assertEqual(sample.image_path, "minimal.jpg")
        self.assertEqual(sample.human_evaluation, "no")
        self.assertIsNone(sample.evaluator_id)
        self.assertIsNone(sample.confidence)
        self.assertIsNone(sample.metadata)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 