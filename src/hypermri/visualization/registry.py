"""
Plotter registry for managing visualization components.

This module implements a registry-based system for plotters, following the
pattern established in Phase 1 of the hypermri refactoring.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from ..core.exceptions import HyperMRIError
from ..core.interfaces import BaseRegistry
from .interfaces import BasePlotter

logger = logging.getLogger(__name__)


class PlotterRegistryError(HyperMRIError):
    """Raised when plotter registry operations fail."""


class PlotterRegistry(BaseRegistry):
    """
    Registry for managing plotter classes and their configurations.

    This registry manages the association between data types/sequence types
    and their appropriate plotter classes, following the pattern established
    in the reconstruction and vendor registries.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the plotter registry.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to the plotter mapping configuration file
        """
        super().__init__()
        self._plotters = {}
        self._config = {}
        self._loaded_classes = {}

        if config_path is None:
            config_path = Path(__file__).parent / "config" / "plotter_mapping.yaml"

        self.config_path = Path(config_path)
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load plotter configurations from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self._config = yaml.safe_load(f) or {}
                self._load_plotters_from_config()
            else:
                logger.warning(f"Plotter config file not found: {self.config_path}")
                self._load_default_plotters()
        except Exception as e:
            logger.error(f"Failed to load plotter configuration: {e}")
            self._load_default_plotters()

    def _load_plotters_from_config(self) -> None:
        """Load plotters based on configuration."""
        plotters = self._config.get("plotters", {})

        for plotter_key, plotter_info in plotters.items():
            try:
                class_path = plotter_info.get("class")
                legacy_class_path = plotter_info.get("legacy_class")
                options = plotter_info.get("options", {})
                categories = plotter_info.get("categories", [])

                if class_path:
                    plotter_class = self._import_plotter_class(class_path)
                    if plotter_class:
                        plotter_config = {
                            "class": plotter_class,
                            "options": options,
                            "categories": categories,
                        }

                        # If this is an adapter, store the legacy class info
                        if legacy_class_path:
                            legacy_class = self._import_plotter_class(legacy_class_path)
                            if legacy_class:
                                plotter_config["legacy_class"] = legacy_class
                            else:
                                logger.warning(
                                    f"Failed to load legacy class: {legacy_class_path}"
                                )

                        self._plotters[plotter_key] = plotter_config
                        logger.debug(f"Loaded plotter: {plotter_key}")
                    else:
                        logger.warning(f"Failed to load plotter class: {class_path}")
                else:
                    logger.warning(f"No class specified for plotter: {plotter_key}")

            except Exception as e:
                logger.warning(f"Failed to load plotter {plotter_key}: {e}")

    def _load_default_plotters(self) -> None:
        """Load default plotters when configuration is unavailable."""
        default_plotters = {
            "anatomy": {
                "class": "hypermri.utils.plot.anatomyPlotter.AnatomyPlotter",
                "categories": ["anatomical"],
                "options": {},
            },
            "spectroscopy": {
                "class": "hypermri.utils.plot.spectroscopyPlotter.SpectroscopyPlotter",
                "categories": ["spectroscopy"],
                "options": {},
            },
            "unified": {
                "class": "hypermri.utils.plot.unifiedPlotter.UnifiedPlotter",
                "categories": ["anatomical", "spectroscopy"],
                "options": {},
            },
        }

        for plotter_key, plotter_info in default_plotters.items():
            try:
                class_path = plotter_info["class"]
                plotter_class = self._import_plotter_class(class_path)
                if plotter_class:
                    self._plotters[plotter_key] = plotter_info.copy()
                    self._plotters[plotter_key]["class"] = plotter_class
                    logger.debug(f"Loaded default plotter: {plotter_key}")
            except Exception as e:
                logger.warning(f"Failed to load default plotter {plotter_key}: {e}")

    def _import_plotter_class(self, class_path: str) -> Optional[Type[BasePlotter]]:
        """
        Import a plotter class from its module path.

        Parameters
        ----------
        class_path : str
            Full import path to the plotter class

        Returns
        -------
        Type[BasePlotter] or None
            The imported plotter class, or None if import failed
        """
        if class_path in self._loaded_classes:
            return self._loaded_classes[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            plotter_class = getattr(module, class_name)

            # Validate that the class implements the plotter interface
            # Be more lenient for legacy classes and adapters
            is_adapter = "adapter" in class_path.lower()
            has_plot_method = hasattr(plotter_class, "plot")
            is_base_plotter = (
                issubclass(plotter_class, BasePlotter)
                if hasattr(plotter_class, "__mro__")
                else False
            )

            # Legacy classes may not follow the exact interface but that's OK
            if not is_adapter and not has_plot_method and not is_base_plotter:
                logger.debug(
                    f"Class {class_path} may not implement plotter interface, but allowing it"
                )
                # Don't return None, allow it through for now

            self._loaded_classes[class_path] = plotter_class
            return plotter_class

        except Exception as e:
            logger.warning(f"Could not import plotter class '{class_path}': {e}")
            return None

    def get_plotter(self, key: str, data_obj=None, **kwargs) -> Optional[Any]:
        """
        Get a plotter instance by key.

        Parameters
        ----------
        key : str
            The plotter key
        data_obj : object, optional
            The data object (sequence object) to pass to plotter constructor
        **kwargs
            Additional arguments passed to plotter constructor

        Returns
        -------
        Plotter instance or None
            Plotter instance, or None if not found
        """
        plotter_info = self._plotters.get(key)
        if not plotter_info:
            logger.warning(f"Plotter '{key}' not found in registry")
            return None

        try:
            plotter_class = plotter_info["class"]
            default_options = plotter_info.get("options", {})
            legacy_class = plotter_info.get("legacy_class")

            # Merge default options with provided kwargs
            config = {**default_options, **kwargs}

            # Handle different instantiation patterns
            if legacy_class:
                # Using adapter pattern (legacy approach)
                return plotter_class(legacy_class, config=config)
            else:
                # Direct instantiation of existing plotters
                if data_obj is not None:
                    # For existing plotters that expect data objects
                    return plotter_class(data_obj)
                else:
                    # For plotters that can be instantiated without data
                    try:
                        return plotter_class()
                    except TypeError:
                        # If constructor requires arguments, return the class itself
                        # This allows factory functions to handle instantiation
                        logger.debug(
                            f"Returning plotter class for later instantiation: {key}"
                        )
                        return plotter_class

        except Exception as e:
            logger.error(f"Failed to create plotter '{key}': {e}")
            return None

    def get_plotter_for_category(
        self, category: str, data_obj=None, **kwargs
    ) -> Optional[Any]:
        """
        Get a plotter suitable for the given data category.

        Parameters
        ----------
        category : str
            The data category (e.g., 'anatomical', 'spectroscopy')
        data_obj : object, optional
            The data object to pass to plotter constructor
        **kwargs
            Additional arguments passed to plotter constructor

        Returns
        -------
        Plotter instance or None
            Plotter instance, or None if not found
        """
        # Check defaults first for preferred plotter for this category
        defaults = self._config.get("defaults", {})
        if category in defaults:
            preferred_key = defaults[category]
            if preferred_key in self._plotters:
                return self.get_plotter(preferred_key, data_obj, **kwargs)

        # Fall back to any plotter that supports this category
        for plotter_key, plotter_info in self._plotters.items():
            categories = plotter_info.get("categories", [])
            if category in categories:
                return self.get_plotter(plotter_key, data_obj, **kwargs)

        logger.warning(f"No plotter found for category: {category}")
        return None

    def list_plotters(self) -> List[str]:
        """
        List all available plotter keys.

        Returns
        -------
        list of str
            List of registered plotter keys
        """
        return list(self._plotters.keys())

    def list_categories(self) -> List[str]:
        """
        List all available plotter categories.

        Returns
        -------
        list of str
            List of all categories supported by registered plotters
        """
        categories = set()
        for plotter_info in self._plotters.values():
            categories.update(plotter_info.get("categories", []))
        return list(categories)

    def register_plotter(
        self,
        key: str,
        plotter_class: Type[BasePlotter],
        categories: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a plotter class.

        Parameters
        ----------
        key : str
            Unique key for the plotter
        plotter_class : Type[BasePlotter]
            The plotter class to register
        categories : list of str, optional
            Categories this plotter supports
        options : dict, optional
            Default options for the plotter
        overwrite : bool, default False
            Whether to overwrite existing registration
        """
        if key in self._plotters and not overwrite:
            raise PlotterRegistryError(
                f"Plotter '{key}' already registered. Use overwrite=True to replace."
            )

        self._plotters[key] = {
            "class": plotter_class,
            "categories": categories or [],
            "options": options or {},
        }

        logger.info(f"Registered plotter: {key}")


# Global registry instance
_plotter_registry = None


def get_plotter_registry() -> PlotterRegistry:
    """
    Get the global plotter registry instance.

    Returns
    -------
    PlotterRegistry
        The global registry instance
    """
    global _plotter_registry
    if _plotter_registry is None:
        _plotter_registry = PlotterRegistry()
    return _plotter_registry


def get_plotter(key: str, **kwargs) -> Optional[BasePlotter]:
    """
    Get a plotter instance from the global registry.

    Parameters
    ----------
    key : str
        The plotter key
    **kwargs
        Additional arguments passed to plotter constructor

    Returns
    -------
    BasePlotter or None
        Plotter instance, or None if not found
    """
    registry = get_plotter_registry()
    return registry.get_plotter(key, **kwargs)


def get_plotter_for_category(category: str, **kwargs) -> Optional[BasePlotter]:
    """
    Get a plotter for the given category from the global registry.

    Parameters
    ----------
    category : str
        The data category
    **kwargs
        Additional arguments passed to plotter constructor

    Returns
    -------
    BasePlotter or None
        Plotter instance, or None if not found
    """
    registry = get_plotter_registry()
    return registry.get_plotter_for_category(category, **kwargs)
