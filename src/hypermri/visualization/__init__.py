"""
Unified visualization system for hypermri.

This module provides a registry-based visualization system that consolidates
the various plotting approaches found throughout the codebase.

The system follows the registry pattern established in Phase 1 of the refactoring,
providing consistent interfaces and configuration-driven plotter selection.
"""

from .adapters import (
    LegacyPlotterAdapter,
    ModernPlotterAdapter,
    UmpireMGEPlotterAdapter,
)
from .interfaces import BasePlotter, InteractivePlotter, PlotterProtocol, StaticPlotter
from .registry import (
    PlotterRegistry,
    PlotterRegistryError,
    get_plotter,
    get_plotter_for_category,
    get_plotter_registry,
)

__all__ = [
    # Interfaces
    "BasePlotter",
    "InteractivePlotter",
    "StaticPlotter",
    "PlotterProtocol",
    # Registry
    "PlotterRegistry",
    "PlotterRegistryError",
    "get_plotter_registry",
    "get_plotter",
    "get_plotter_for_category",
    # Adapters
    "LegacyPlotterAdapter",
    "UmpireMGEPlotterAdapter",
    "ModernPlotterAdapter",
]
