"""
Adapter classes for backward compatibility with existing plotters.

This module provides adapters to make existing plotters work with the new
unified visualization system while maintaining backward compatibility.
"""

import logging
from typing import Any, Dict, Optional

import xarray as xr

from .interfaces import BasePlotter

logger = logging.getLogger(__name__)


class LegacyPlotterAdapter(BasePlotter):
    """
    Adapter for legacy plotters that don't follow the unified interface.

    This adapter wraps existing plotter classes to make them compatible
    with the new registry-based visualization system.
    """

    def __init__(
        self,
        legacy_plotter_class,
        config: Optional[Dict[str, Any]] = None,
        constructor_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Parameters
        ----------
        legacy_plotter_class : type
            The legacy plotter class to wrap
        config : dict, optional
            Configuration for the adapter
        constructor_args : dict, optional
            Arguments to pass to the legacy plotter constructor
        """
        super().__init__(config)
        self.legacy_plotter_class = legacy_plotter_class
        self.constructor_args = constructor_args or {}
        self._legacy_instance = None

    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """
        Create a plot using the legacy plotter.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            The legacy plotter instance
        """
        try:
            # Merge constructor args with any data-specific parameters
            init_args = {**self.constructor_args}

            # Some legacy plotters expect the data object directly
            if "data" not in init_args and hasattr(data, "values"):
                init_args["data"] = data

            # Try to create the legacy plotter instance
            self._legacy_instance = self.legacy_plotter_class(**init_args)

            return self._legacy_instance

        except Exception as e:
            logger.error(f"Failed to create legacy plotter: {e}")
            return None

    def setup_interactive(self) -> None:
        """Set up interactive components using the legacy plotter."""
        if self._legacy_instance and hasattr(
            self._legacy_instance, "setup_interactive"
        ):
            try:
                self._legacy_instance.setup_interactive()
            except Exception as e:
                logger.warning(f"Failed to setup interactive components: {e}")

    def show(self) -> None:
        """Display the plot using the legacy plotter."""
        if self._legacy_instance and hasattr(self._legacy_instance, "show"):
            try:
                self._legacy_instance.show()
            except Exception as e:
                logger.warning(f"Failed to show plot: {e}")


class ModernPlotterAdapter(BasePlotter):
    """
    Adapter for modern plotters that mostly follow the new interface.

    This adapter provides a bridge between the registry system and modern
    plotters that are already close to the unified interface but may need
    minor adjustments for full compatibility.
    """

    def __init__(
        self,
        legacy_class=None,
        config: Optional[Dict[str, Any]] = None,
        constructor_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the modern plotter adapter.

        Parameters
        ----------
        legacy_class : type, optional
            The plotter class to wrap (passed from registry)
        config : dict, optional
            Configuration for the adapter
        constructor_args : dict, optional
            Arguments to pass to the plotter constructor
        """
        super().__init__(config)
        self.plotter_class = legacy_class
        self.constructor_args = constructor_args or {}
        self._instance = None

    def plot(self, data: xr.DataArray, data_object=None, **kwargs) -> Any:
        """
        Create a plot using the modern plotter.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot
        data_object : object, optional
            The data object (sequence instance) that contains the plotter
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            The plotter instance or plot handles
        """
        try:
            # Handle different initialization patterns
            if data_object is not None:
                # Modern plotters typically take the data object as constructor arg
                self._instance = self.plotter_class(data_object)
            else:
                # Try direct instantiation
                init_args = self.constructor_args
                self._instance = self.plotter_class(**init_args)

            # Call the plot method with appropriate parameters
            if hasattr(self._instance, "plot"):
                # Combine data with config options and kwargs
                plot_params = {**self.config.get("options", {}), **kwargs}
                return self._instance.plot(
                    input_array_list=[data], plot_params=plot_params
                )
            else:
                logger.warning(f"Plotter {self.plotter_class} has no plot method")
                return self._instance

        except Exception as e:
            logger.error(f"Failed to create modern plotter: {e}")
            return None

    def setup_interactive(self) -> None:
        """Set up interactive components using the modern plotter."""
        if self._instance and hasattr(self._instance, "setup_interactive"):
            try:
                self._instance.setup_interactive()
            except Exception as e:
                logger.warning(f"Failed to setup interactive components: {e}")

    def show(self) -> None:
        """Display the plot using the modern plotter."""
        if self._instance and hasattr(self._instance, "show"):
            try:
                self._instance.show()
            except Exception as e:
                logger.warning(f"Failed to show plot: {e}")


class UmpireMGEPlotterAdapter(BasePlotter):
    """
    Specialized adapter for umpireMGE plotters.

    The umpireMGE plotters have a specific interface that expects
    uMGE objects and has specialized initialization.
    """

    def __init__(self, plotter_class, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the umpireMGE adapter.

        Parameters
        ----------
        plotter_class : type
            The umpireMGE plotter class
        config : dict, optional
            Configuration for the adapter
        """
        super().__init__(config)
        self.plotter_class = plotter_class
        self._instance = None

    def plot(
        self, data: xr.DataArray, uMGE=None, anatref=None, parameters=None, **kwargs
    ) -> Any:
        """
        Create an umpireMGE plot.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot (may not be used directly)
        uMGE : object
            The umpireMGE object containing the data and methods
        anatref : array-like, optional
            Anatomical reference data
        parameters : dict, optional
            Plotting parameters
        **kwargs
            Additional parameters

        Returns
        -------
        Any
            The umpireMGE plotter instance
        """
        if uMGE is None:
            logger.error("umpireMGE plotters require a uMGE object")
            return None

        try:
            # Create the umpireMGE plotter with expected arguments
            init_kwargs = {}
            if anatref is not None:
                init_kwargs["anatref"] = anatref
            if parameters is not None:
                init_kwargs["parameters"] = parameters

            # Update with any config parameters
            init_kwargs.update(kwargs)

            self._instance = self.plotter_class(uMGE, **init_kwargs)
            return self._instance

        except Exception as e:
            logger.error(f"Failed to create umpireMGE plotter: {e}")
            return None

    def setup_interactive(self) -> None:
        """umpireMGE plotters handle this in their constructor."""

    def show(self) -> None:
        """umpireMGE plotters handle this in their constructor."""


class ModernPlotterAdapter(BasePlotter):
    """
    Adapter for modern plotters that mostly follow the unified interface
    but may have constructor compatibility issues.
    """

    def __init__(self, plotter_class, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the modern plotter adapter.

        Parameters
        ----------
        plotter_class : type
            The modern plotter class
        config : dict, optional
            Configuration for the adapter
        """
        super().__init__(config)
        self.plotter_class = plotter_class
        self._instance = None

    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """
        Create a plot using the modern plotter.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            The modern plotter instance
        """
        try:
            # Try to create the plotter instance
            # First try with data and config, then fallback to just data
            try:
                self._instance = self.plotter_class(data, config=self.config, **kwargs)
            except TypeError:
                # Fallback for plotters that don't accept config
                self._instance = self.plotter_class(data, **kwargs)

            return self._instance

        except Exception as e:
            logger.error(f"Failed to create modern plotter: {e}")
            return None

    def setup_interactive(self) -> None:
        """Set up interactive components."""
        if self._instance and hasattr(self._instance, "setup_interactive"):
            try:
                self._instance.setup_interactive()
            except Exception as e:
                logger.warning(f"Failed to setup interactive components: {e}")

    def show(self) -> None:
        """Display the plot."""
        if self._instance and hasattr(self._instance, "show"):
            try:
                self._instance.show()
            except Exception as e:
                logger.warning(f"Failed to show plot: {e}")
        elif hasattr(self._instance, "plot"):
            # Some plotters just have a plot method
            try:
                self._instance.plot()
            except Exception as e:
                logger.warning(f"Failed to plot: {e}")
