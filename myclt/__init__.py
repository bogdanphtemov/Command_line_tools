"""CLT - Command Line Tools Package."""

import os
import sys

# Add the parent directory to Python path so imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Command-line tools for system cleaning, project automation, and machine learning"
