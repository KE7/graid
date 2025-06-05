# Test Organization

This directory has been reorganized to separate different types of tests based on their requirements and execution characteristics.

## Directory Structure

```
tests/
├── unit_tests/          # Automated tests (no human intervention)
│   ├── test_imageloader.py
│   ├── test_datasets.py
│   ├── test_generate_db.py
│   └── README.md
├── manual_tests/        # Integration tests (require human setup)  
│   ├── test_detectron_obj.py
│   ├── test_dino_obj.py
│   ├── test_mmdetection_obj.py
│   ├── test_mmdetection_seg.py
│   ├── test_ultralytics_obj.py
│   ├── test_ultralytics_seg.py
│   ├── test_questions.py
│   └── README.md
└── README.md           # This file
```

## Quick Start

### Automated Unit Tests
```bash
# Run all automated tests
python -m pytest tests/unit_tests/ -v

# Fast, reliable, no setup required
# All dependencies mocked
# Perfect for CI/CD
```

### Manual Integration Tests  
```bash
# Run individual tests with proper setup
python -m pytest tests/manual_tests/test_detectron_obj.py -v -s

# Requires real models, datasets, GPU
# Human verification needed
# Run when validating end-to-end functionality
```

## Test Types

| Type | Location | Purpose | Requirements |
|------|----------|---------|--------------|
| **Unit Tests** | `unit_tests/` | Validate bug fixes, core logic | None (mocked) |
| **Manual Tests** | `manual_tests/` | End-to-end validation | Models, data, GPU |

## When to Use Each

**Use Unit Tests:**
- During development 
- In CI/CD pipelines
- To validate bug fixes
- For quick feedback loops

**Use Manual Tests:**
- Before releases
- When adding new models
- For performance validation
- When debugging real-world issues

## Contributing

- Add new **unit tests** for any bug fixes or core functionality changes
- Update **manual tests** when adding new models or major features
- Keep unit tests fast and dependency-free
- Document any special setup requirements for manual tests 