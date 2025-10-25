"""
Common Dataset Loader Factory for GRAID

Provides a centralized way to create dataset loaders across the entire codebase.
Eliminates duplicate dataset initialization logic scattered throughout different modules.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetLoaderCreator(ABC):
    """Abstract base class for dataset loader creators."""

    @staticmethod
    @abstractmethod
    def create(split: str, transform: Any, **kwargs) -> Any:
        """Create a dataset loader instance."""
        pass


class BddLoaderCreator(DatasetLoaderCreator):
    """Creator for BDD100K dataset loaders."""

    @staticmethod
    def create(split: str, transform: Any, **kwargs) -> Any:
        """Create BDD100K dataset loader."""
        from graid.data.ImageLoader import Bdd100kDataset

        pkl_root = Path("data") / f"bdd_{split}"
        rebuild_needed = not (pkl_root / "0.pkl").exists()

        # Allow override of rebuild behavior
        rebuild = kwargs.get("rebuild", rebuild_needed)
        use_time_filtered = kwargs.get("use_time_filtered", False)

        return Bdd100kDataset(
            split=split,
            transform=transform,
            use_time_filtered=use_time_filtered,
            rebuild=rebuild,
        )


class NuImagesLoaderCreator(DatasetLoaderCreator):
    """Creator for NuImages dataset loaders."""

    @staticmethod
    def create(split: str, transform: Any, **kwargs) -> Any:
        """Create NuImages dataset loader."""
        from graid.data.ImageLoader import NuImagesDataset

        # Allow override of size parameter
        size = kwargs.get("size", "all")

        return NuImagesDataset(
            split=split, 
            size=size, 
            transform=transform,
            # rebuild=True,
            use_time_filtered=False,
        )


class WaymoLoaderCreator(DatasetLoaderCreator):
    """Creator for Waymo dataset loaders."""

    @staticmethod
    def create(split: str, transform: Any, **kwargs) -> Any:
        """Create Waymo dataset loader."""
        from graid.data.ImageLoader import WaymoDataset

        # Convert split name for Waymo's naming convention
        split_name = "validation" if split == "val" else split + "ing"

        return WaymoDataset(
            split=split_name, 
            transform=transform,
            # keep reading from *_interesting
            use_interesting_path=True,
            # never drop night frames at load time
            filter_working_hours=False,
            # maintain backward-compatible arg
            use_time_filtered=True,
        ) 


class DatasetLoaderFactory:
    """
    Centralized factory for creating dataset loaders.

    This factory can be used throughout the GRAID codebase to eliminate
    duplicate dataset initialization logic.

    Example usage:
        transform = get_some_transform()
        loader = DatasetLoaderFactory.create("bdd", "train", transform)
    """

    # Registry of available dataset creators
    _CREATORS: dict[str, DatasetLoaderCreator] = {
        "bdd": BddLoaderCreator,
        "nuimage": NuImagesLoaderCreator,
        "waymo": WaymoLoaderCreator,
    }

    @classmethod
    def create(cls, dataset_name: str, split: str, transform: Any, **kwargs) -> Any:
        """
        Create a dataset loader for the specified dataset.

        Args:
            dataset_name: Name of the dataset ("bdd", "nuimage", "waymo")
            split: Dataset split ("train", "val", "test")
            transform: Transform function to apply to images
            **kwargs: Additional arguments passed to the specific creator

        Returns:
            Dataset loader instance

        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in cls._CREATORS:
            available = list(cls._CREATORS.keys())
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Available: {available}"
            )

        creator = cls._CREATORS[dataset_name]
        return creator.create(split, transform, **kwargs)

    @classmethod
    def register_creator(cls, dataset_name: str, creator: DatasetLoaderCreator):
        """
        Register a new dataset creator.

        Args:
            dataset_name: Name to register the dataset under
            creator: Creator class implementing DatasetLoaderCreator interface
        """
        cls._CREATORS[dataset_name] = creator

    @classmethod
    def get_supported_datasets(cls) -> list[str]:
        """Get list of supported dataset names."""
        return list(cls._CREATORS.keys())


# Convenience function for backward compatibility
def create_dataset_loader(
    dataset_name: str, split: str, transform: Any, **kwargs
) -> Any:
    """
    Convenience function to create dataset loaders.

    This is a simple wrapper around DatasetLoaderFactory.create() for
    easier migration from existing code.
    """
    return DatasetLoaderFactory.create(dataset_name, split, transform, **kwargs)
