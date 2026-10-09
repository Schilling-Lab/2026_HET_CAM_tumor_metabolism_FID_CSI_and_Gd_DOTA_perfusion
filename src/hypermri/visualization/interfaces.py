"""
Unified plotting interfaces for hypermri visualization system.

This module provides a consolidated plotting interface that unifies the various
plotter abstractions found throughout the codebase, following the registry pattern
established in Phase 1 of the refactoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import display


@runtime_checkable
class PlotterProtocol(Protocol):
    """Protocol defining the interface for plotters."""

    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """Create a plot of the provided data."""
        ...

    def setup_interactive(self) -> None:
        """Set up interactive components if supported."""
        ...


class BasePlotter(ABC):
    """
    Abstract base class for all plotters in the hypermri system.

    This class consolidates the functionality from the various plotter base classes
    and provides a unified interface following the registry pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plotter with optional configuration.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for plotter-specific settings
        """
        self.config = config or {}
        self._denoised_cache = {}
        self._last_original_data_id = None
        self.state = []
        self.widgets = {}
        self.plot_handles = None
        self.fig = None
        self.ax = None

    @abstractmethod
    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """
        Create a plot of the provided data.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot handles or figure reference
        """

    def setup_figure(self) -> None:
        """
        Set up matplotlib figure and axes.

        Creates figure in non-interactive mode to prevent premature display.
        Subclasses can override for custom figure setup.
        """
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=self.config.get("figsize", (8, 6)))

    def setup_interactive(self) -> None:
        """
        Set up interactive components.

        Default implementation does nothing. Subclasses should override
        to add interactive widgets and controls.
        """

    def update(self) -> None:
        """
        Update the plot display.

        Default implementation does nothing. Subclasses should override
        to define how the plot updates when parameters change.
        """

    def show(self) -> None:
        """
        Display the plot.

        Default implementation shows the matplotlib figure.
        """
        if self.fig is not None:
            display(self.fig)
            plt.show()

    def clear_cache(self) -> None:
        """Clear any cached data."""
        self._denoised_cache.clear()
        self._last_original_data_id = None

    def get_cache_key(self, data: xr.DataArray, **kwargs) -> str:
        """
        Generate a cache key for the given data and parameters.

        Parameters
        ----------
        data : xarray.DataArray
            The data array
        **kwargs
            Additional parameters

        Returns
        -------
        str
            Cache key string
        """
        data_id = id(data)
        param_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
        return f"{data_id}_{param_str}"


class InteractivePlotter(BasePlotter):
    """
    Base class for interactive plotters with widget support.

    This class extends BasePlotter to provide common interactive functionality
    that was previously scattered across PlotterABC and the utils/plot module.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.setup_figure()
        self.setup_interactive()

    def setup_interactive(self) -> None:
        """Set up interactive widgets. Must be implemented by subclasses."""

    def create_widgets(self) -> Dict[str, Any]:
        """
        Create and return interactive widgets.

        Returns
        -------
        dict
            Dictionary of widget name to widget object mappings
        """
        return {}


class StaticPlotter(BasePlotter):
    """
    Base class for static (non-interactive) plotters.

    This class provides a simplified interface for plotters that don't
    require interactive components.
    """

    def show(self) -> None:
        """Display the static plot."""
        if self.fig is not None:
            plt.show()
