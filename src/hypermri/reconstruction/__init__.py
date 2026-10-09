"""
Reconstruction module for hypermri.

This module provides the reconstruction strategy system with a registry-based
approach that replaces the massive if-elif chain pattern.
"""

from .registry import (
    ReconstructionRegistry,
    get_reconstruction_registry,
    get_reconstruction_strategy,
    register_reconstruction_strategy,
)

__all__ = [
    "ReconstructionRegistry",
    "get_reconstruction_registry",
    "register_reconstruction_strategy",
    "get_reconstruction_strategy",
]
