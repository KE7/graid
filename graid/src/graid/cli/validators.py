"""
Centralized validation logic for CLI arguments.

Eliminates duplicate validation code and provides consistent error messages.
"""

from typing import List, Union

from graid.cli.exceptions import (
    DatasetValidationError, COCOValidationError, SplitValidationError
)


class ArgumentValidator:
    """Centralized validation logic for CLI arguments."""
    
    # Supported datasets (can be extended)
    SUPPORTED_DATASETS = ["bdd", "nuimage", "waymo"]
    
    # Valid split names
    VALID_SPLITS = ["train", "val", "test", "train+val", "both", "all", "trainval"]
    
    @classmethod
    def require_valid_dataset(cls, dataset_name: str):
        """Validate dataset name or raise DatasetValidationError."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise DatasetValidationError(
                f"Invalid dataset: {dataset_name}. Supported datasets: {', '.join(cls.SUPPORTED_DATASETS)}"
            )
    
    @classmethod
    def require_valid_split(cls, split_value: Union[str, List[str]]):
        """Validate split specification or raise SplitValidationError."""
        if isinstance(split_value, (list, tuple)):
            splits = list(split_value)
        else:
            splits = [str(split_value)]
        
        for split in splits:
            split_lower = split.lower()
            # Allow individual splits or combined formats
            if split_lower not in cls.VALID_SPLITS and split_lower not in ["train", "val", "test"]:
                raise SplitValidationError(
                    f"Invalid split: {split}. Valid splits: {', '.join(cls.VALID_SPLITS)}"
                )
    
    @classmethod
    def require_valid_coco_objects(cls, objects: List[str]):
        """Validate COCO objects or raise COCOValidationError."""
        # Local import to avoid heavy dependencies
        from graid.utilities.coco import validate_coco_objects
        
        is_valid, error_msg = validate_coco_objects(objects)
        if not is_valid:
            raise COCOValidationError(f"Invalid COCO objects: {error_msg}")
    
    @classmethod
    def parse_and_validate_split(cls, split_value: str) -> List[str]:
        """Parse and validate split specification, returning normalized list."""
        cls.require_valid_split(split_value)
        
        # Normalize split specification  
        if isinstance(split_value, (list, tuple)):
            return list(split_value)
        
        value = str(split_value).lower()
        if value in {"train+val", "both", "all", "trainval"}:
            return ["train", "val"]
        
        return [str(split_value)]
    
    @classmethod
    def validate_numeric_range(cls, value: float, min_val: float, max_val: float, name: str):
        """Validate numeric value is within specified range."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    @classmethod
    def validate_positive_int(cls, value: int, name: str):
        """Validate integer is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @classmethod
    def require_non_empty_string(cls, value: str, name: str):
        """Validate string is not empty."""
        if not value or not value.strip():
            raise ValueError(f"{name} cannot be empty")
