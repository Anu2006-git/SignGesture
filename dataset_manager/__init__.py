"""
dataset_manager package
=======================
Provides dataset scanning, validation, statistics, and dashboard
for the Sign Language Detection project.
"""

from .scanner   import DatasetScanner
from .validator import ImageValidator
from .stats     import DatasetStats
from .dashboard import DatasetDashboard
from .reporter  import DatasetReporter

__all__ = [
    "DatasetScanner",
    "ImageValidator",
    "DatasetStats",
    "DatasetDashboard",
    "DatasetReporter",
]
