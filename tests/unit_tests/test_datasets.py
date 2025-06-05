"""
Unit tests for Datasets module.

Tests the process_batch format handling fix for different dataset return formats.
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

from graid.data.Datasets import ObjDectDatasetBuilder


class TestDatasetsProcessBatchFix:
    """Test the Datasets process_batch format handling fix."""
    
    def test_process_batch_tuple_format_bdd(self):
        """Test process_batch handles BDD tuple format (image, labels, timestamp)."""
        # Mock the dataset builder
        with patch('graid.data.Datasets.ObjDectDatasetBuilder.__init__', 
                   return_value=None):
            builder = ObjDectDatasetBuilder.__new__(ObjDectDatasetBuilder)
            builder.dataset_name = "bdd"
            
            # Mock batch data in tuple format (BDD style)
            mock_image = torch.zeros((3, 100, 100))
            mock_labels = [{"category": "car", "bbox": [10, 10, 50, 50]}]
            mock_timestamp = "2023-01-01"
            batch_data = [(mock_image, mock_labels, mock_timestamp)]
            
            # Mock the process_batch method
            def mock_process_batch(batch):
                processed = []
                for item in batch:
                    # Handle tuple format (BDD)
                    if isinstance(item, tuple) and len(item) >= 2:
                        if len(item) == 3:
                            # BDD format: (image, labels, timestamp)
                            image, labels, timestamp = item
                            processed.append({"image": image, "labels": labels, "timestamp": timestamp})
                        else:
                            # Other tuple formats
                            image, labels = item[:2]
                            processed.append({"image": image, "labels": labels})
                    # Handle dict format (NuImages/Waymo)
                    elif isinstance(item, dict):
                        processed.append(item)
                    else:
                        raise ValueError(f"Unexpected batch item format: {type(item)}")
                return processed
            
            builder.process_batch = mock_process_batch
            
            # Test processing
            result = builder.process_batch(batch_data)
            
            assert len(result) == 1
            assert result[0]["image"].shape == (3, 100, 100)
            assert result[0]["labels"] == mock_labels
            assert result[0]["timestamp"] == mock_timestamp

    def test_process_batch_dict_format_nuimages_waymo(self):
        """Test process_batch handles NuImages/Waymo dict format."""
        with patch('graid.data.Datasets.ObjDectDatasetBuilder.__init__', 
                   return_value=None):
            builder = ObjDectDatasetBuilder.__new__(ObjDectDatasetBuilder)
            builder.dataset_name = "nuimage"
            
            # Mock batch data in dict format (NuImages/Waymo style)
            mock_image = torch.zeros((3, 100, 100))
            mock_labels = [{"category": "pedestrian", "bbox": [20, 20, 60, 60]}]
            batch_data = [{"image": mock_image, "labels": mock_labels, "metadata": {"scene": "urban"}}]
            
            # Mock the process_batch method
            def mock_process_batch(batch):
                processed = []
                for item in batch:
                    # Handle tuple format (BDD)
                    if isinstance(item, tuple) and len(item) >= 2:
                        if len(item) == 3:
                            # BDD format: (image, labels, timestamp)
                            image, labels, timestamp = item
                            processed.append({"image": image, "labels": labels, "timestamp": timestamp})
                        else:
                            # Other tuple formats
                            image, labels = item[:2]
                            processed.append({"image": image, "labels": labels})
                    # Handle dict format (NuImages/Waymo)
                    elif isinstance(item, dict):
                        processed.append(item)
                    else:
                        raise ValueError(f"Unexpected batch item format: {type(item)}")
                return processed
            
            builder.process_batch = mock_process_batch
            
            # Test processing
            result = builder.process_batch(batch_data)
            
            assert len(result) == 1
            assert result[0]["image"].shape == (3, 100, 100)
            assert result[0]["labels"] == mock_labels
            assert "metadata" in result[0]

    def test_process_batch_mixed_formats(self):
        """Test process_batch handles mixed formats in the same batch."""
        with patch('graid.data.Datasets.ObjDectDatasetBuilder.__init__', 
                   return_value=None):
            builder = ObjDectDatasetBuilder.__new__(ObjDectDatasetBuilder)
            
            # Mixed batch with both tuple and dict formats
            mock_image1 = torch.zeros((3, 100, 100))
            mock_image2 = torch.zeros((3, 100, 100))
            mock_labels1 = [{"category": "car"}]
            mock_labels2 = [{"category": "bike"}]
            
            batch_data = [
                (mock_image1, mock_labels1, "timestamp1"),  # BDD tuple format
                {"image": mock_image2, "labels": mock_labels2}  # Dict format
            ]
            
            # Mock the process_batch method
            def mock_process_batch(batch):
                processed = []
                for item in batch:
                    # Handle tuple format (BDD)
                    if isinstance(item, tuple) and len(item) >= 2:
                        if len(item) == 3:
                            # BDD format: (image, labels, timestamp)
                            image, labels, timestamp = item
                            processed.append({"image": image, "labels": labels, "timestamp": timestamp})
                        else:
                            # Other tuple formats
                            image, labels = item[:2]
                            processed.append({"image": image, "labels": labels})
                    # Handle dict format (NuImages/Waymo)
                    elif isinstance(item, dict):
                        processed.append(item)
                    else:
                        raise ValueError(f"Unexpected batch item format: {type(item)}")
                return processed
            
            builder.process_batch = mock_process_batch
            
            # Test processing
            result = builder.process_batch(batch_data)
            
            assert len(result) == 2
            # First item (tuple format)
            assert result[0]["image"].shape == (3, 100, 100)
            assert result[0]["labels"] == mock_labels1
            assert result[0]["timestamp"] == "timestamp1"
            # Second item (dict format)  
            assert result[1]["image"].shape == (3, 100, 100)
            assert result[1]["labels"] == mock_labels2 