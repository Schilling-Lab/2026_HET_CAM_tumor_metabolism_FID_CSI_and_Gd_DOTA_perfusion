"""
Multi-dimensional MRI data masking and overlay system.

This module provides tools for creating, applying, and managing masks across different
MRI datasets with automatic coordinate alignment and cross-dataset application.

Core Components:
- MaskApplicator: Apply masks across datasets with coordinate alignment
- CoordinateAligner: Handle spatial alignment between different datasets
- MaskContainer: Interactive mask creation with identical coordinates
"""

from .alignment import CoordinateAligner
from .applicator import MaskApplicator
from .interactive import MaskContainer, MaskCreator

__all__ = [
    "MaskApplicator",
    "CoordinateAligner",
    "MaskContainer",
    "MaskCreator",
]
