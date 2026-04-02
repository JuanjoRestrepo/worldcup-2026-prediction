"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add the project root to sys.path so that 'src' imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
