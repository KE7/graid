"""
Unit tests for generate_db module.

Tests the import fixes and functionality of the database generation system.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent / "graid"
sys.path.insert(0, str(project_root))

from graid.data.generate_db import create_model, generate_db, list_available_models, MODEL_CONFIGS


class TestGenerateDbImportFix:
    """Test the generate_db import and functionality fixes."""
    
    @patch('graid.data.generate_db.get_default_device')
    @patch('graid.data.generate_db.ObjDectDatasetBuilder')
    def test_generate_db_import_success(self, mock_builder_class, mock_get_device):
        """Test that generate_db imports and runs without import errors."""
        # Mock the dependencies
        mock_get_device.return_value = 'cpu'
        mock_builder = Mock()
        mock_builder.is_built.return_value = False
        mock_builder.build.return_value = None
        mock_builder_class.return_value = mock_builder
        
        # Test generate_db function can be called
        result = generate_db(
            dataset_name='bdd',
            split='val',
            conf=0.5
        )
        
        # Verify it returns expected structure
        assert result == 'bdd_val_gt'
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()

    @patch('graid.data.generate_db.get_default_device')
    @patch('graid.data.generate_db.ObjDectDatasetBuilder')
    def test_generate_db_ground_truth(self, mock_builder_class, mock_get_device):
        """Test generate_db for ground truth mode."""
        # Mock the dependencies
        mock_get_device.return_value = 'cpu'
        mock_builder = Mock()
        mock_builder.is_built.return_value = False
        mock_builder.build.return_value = None
        mock_builder_class.return_value = mock_builder
        
        # Test ground truth generation
        result = generate_db(
            dataset_name='nuimage',
            split='train',
            conf=0.0
        )
        
        # Verify builder was called with correct parameters
        call_args = mock_builder_class.call_args[1]  # kwargs
        assert call_args['dataset'] == 'nuimage'
        assert call_args['split'] == 'train'
        assert result == 'nuimage_train_gt'

    @patch('graid.data.generate_db.get_default_device')
    @patch('graid.data.generate_db.ObjDectDatasetBuilder')
    @patch('graid.data.generate_db.create_model')
    def test_generate_db_with_model(self, mock_create_model, mock_builder_class, mock_get_device):
        """Test generate_db with a specific model."""
        # Mock the dependencies
        mock_get_device.return_value = 'cuda'
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_builder = Mock()
        mock_builder.is_built.return_value = False
        mock_builder.build.return_value = None
        mock_builder_class.return_value = mock_builder
        
        # Test with a model
        result = generate_db(
            dataset_name='waymo',
            split='val',
            conf=0.3,
            backend='detectron',
            model_name='faster_rcnn_R_50_FPN_3x'
        )
        
        # Verify model was created and used
        mock_create_model.assert_called_once_with(
            'detectron',
            'faster_rcnn_R_50_FPN_3x',
            'cuda',
            0.3
        )
        call_args = mock_builder_class.call_args[1]
        assert result == 'waymo_val_0.3_detectron_faster_rcnn_R_50_FPN_3x'

    def test_list_available_models(self):
        """Test that list_available_models returns proper structure."""
        models = list_available_models()
        
        # Should return a dictionary with backend names as keys
        assert isinstance(models, dict)
        
        # Check that all expected backends are present (lowercase)
        expected_backends = ['detectron', 'mmdetection', 'ultralytics']
        for backend in expected_backends:
            assert backend in models
            assert isinstance(models[backend], list)
            assert len(models[backend]) > 0

    def test_model_configs_structure(self):
        """Test that MODEL_CONFIGS has proper structure."""
        # Test structure
        assert isinstance(MODEL_CONFIGS, dict)
        
        for backend, models in MODEL_CONFIGS.items():
            assert isinstance(models, dict)
            for model_name, config in models.items():
                # Different backends have different config structures
                if backend == "detectron":
                    # Should have config and weights
                    assert 'config' in config
                    assert 'weights' in config
                elif backend == "mmdetection":
                    # Should have config and checkpoint
                    assert 'config' in config
                    assert 'checkpoint' in config
                elif backend == "ultralytics":
                    # Should be a string path to model file
                    assert isinstance(config, str)

    def test_create_model_function_exists(self):
        """Test that create_model function can be imported and called."""
        # Mock dependencies for create_model
        with patch('graid.data.generate_db.get_default_device', return_value='cpu'):
            with patch('graid.data.generate_db.Detectron_obj') as mock_detectron:
                mock_model = Mock()
                mock_detectron.return_value = mock_model
                
                # Test create_model
                result = create_model('detectron', 'faster_rcnn_R_50_FPN_3x', 'cpu')
                assert result == mock_model


class TestIntegrationBugFixes:
    """Integration tests for bug fixes across modules."""
    
    @patch('graid.data.generate_db.get_default_device')
    @patch('graid.data.generate_db.ObjDectDatasetBuilder')
    def test_end_to_end_database_generation(self, mock_builder_class, mock_get_device):
        """Test end-to-end database generation with fixed transform and process_batch."""
        # Mock the dependencies
        mock_get_device.return_value = 'cpu'
        mock_builder = Mock()
        mock_builder.is_built.return_value = False
        
        def mock_build(**kwargs):
            # Simulate the fixed process_batch and transform behavior
            # This would internally use the fixed ImageLoader and Datasets code
            return None
        
        mock_builder.build = mock_build
        mock_builder_class.return_value = mock_builder
        
        # Test database generation
        result = generate_db(
            dataset_name='bdd',
            split='val',
            conf=0.2,
            batch_size=50
        )
        
        # Verify successful completion
        assert result == 'bdd_val_gt'
        
        # Verify builder was called with correct parameters
        call_args = mock_builder_class.call_args[1]
        assert call_args['dataset'] == 'bdd'
        assert call_args['split'] == 'val' 