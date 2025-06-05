# Unit Tests

This directory contains **automated unit tests** that can be run without human intervention.

## Running the Tests

```bash
# Run all unit tests
python -m pytest tests/unit_tests/ -v

# Run specific test file
python -m pytest tests/unit_tests/test_imageloader.py -v
python -m pytest tests/unit_tests/test_datasets.py -v  
python -m pytest tests/unit_tests/test_generate_db.py -v
```

## Test Files

- **`test_imageloader.py`** - Tests for ImageLoader module transform bug fixes
- **`test_datasets.py`** - Tests for Datasets module process_batch format handling  
- **`test_generate_db.py`** - Tests for generate_db module import fixes and functionality

## Purpose

These tests validate bug fixes and core functionality without requiring:
- Real model weights
- Large datasets
- GPU resources
- Human interaction

All dependencies are mocked to ensure fast, reliable automated testing. 