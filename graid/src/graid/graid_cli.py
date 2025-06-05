#!/usr/bin/env python3
"""
GRAID CLI Entry Point

Run this script to launch the GRAID interactive interface.
"""

import sys
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        # Add the src path to make imports work
        src_path = Path(__file__).parent
        sys.path.insert(0, str(src_path))
        
        from graid import app
        app()
    except ImportError as e:
        print(f"Error importing GRAID modules: {e}")
        print("Make sure you're in the project root directory and all dependencies are installed.")
        print("Try: poetry install")
        sys.exit(1) 