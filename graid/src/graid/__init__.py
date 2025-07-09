"""
GRAID: Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence
"""

import logging
import os
import warnings

# Suppress common warnings for better user experience
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*TorchScript.*functional optimizers.*deprecated.*"
)
warnings.filterwarnings("ignore", message=".*Adafactor is already registered.*")

# Suppress mmengine info messages
logging.getLogger("mmengine").setLevel(logging.ERROR)

# Set environment variable to reduce mmengine verbosity
os.environ["MMENGINE_LOGGING_LEVEL"] = "ERROR"


def __getattr__(name):
    """Lazy import for the CLI app to avoid loading it on every import."""
    if name == "app":
        from graid.graid import app

        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["app"]
