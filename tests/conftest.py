"""
Test configuration to make the project modules importable in tests.

Adds the repository root to sys.path so that modules like `drone_soc`
and `hjb` can be imported directly when running pytest from the repo root.
"""

import os
import sys
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Suppress Matplotlib's animation "deleted without rendering" warning in tests.
warnings.filterwarnings(
    "ignore",
    message=r"Animation was deleted without rendering anything.*",
    category=UserWarning,
)
