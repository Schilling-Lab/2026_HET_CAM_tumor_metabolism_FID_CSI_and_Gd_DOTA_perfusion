"""
Core interfaces and protocols for the hypermri package.

This module defines the abstract base classes and protocols that establish
the contracts for various components in the hypermri architecture.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import xarray as xr


class VendorHandlerProtocol(Protocol):
    """Protocol defining the interface for vendor-specific handlers."""

    @property
    def id(self) -> str:
        """Unique identifier for the experiment."""
        ...

    @property
    def type(self) -> str:
        """Type/sequence name of the experiment."""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Get experiment parameters."""
        ...

    def load_data(self) -> Optional[xr.DataArray]:
        """Load raw data."""
        ...


class ReconstructionStrategyProtocol(Protocol):
    """Protocol defining the interface for reconstruction strategies."""

    def reconstruct(
        self, handler: VendorHandlerProtocol, **kwargs
    ) -> Optional[xr.DataArray]:
        """Perform reconstruction using the provided handler."""
        ...


class PlotterProtocol(Protocol):
    """Protocol defining the interface for plotters."""

    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """Create a plot of the provided data."""
        ...


class RegistryProtocol(Protocol):
    """Protocol for registry implementations."""

    def register(self, key: str, value: Any) -> None:
        """Register a value with the given key."""
        ...

    def get(self, key: str) -> Optional[Any]:
        """Get a registered value by key."""
        ...

    def list_registered(self) -> list:
        """List all registered keys."""
        ...


class BaseVendorHandler(ABC):
    """Abstract base class for vendor-specific handlers."""

    def __init__(self, source: Union[str, Path], load_data: bool = False):
        """Initialise a vendor handler with a canonicalised source path.

        Parameters
        ----------
        source
            Filesystem location of the vendor measurement. Relative paths are
            resolved against the current working directory.
        load_data
            If ``True``, subclasses are expected to load and cache the raw data
            during initialisation.
        """
        self.source = Path(source)
        self.load_data = load_data
        self._parameters = None
        self._data = None

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the experiment."""

    @property
    @abstractmethod
    def type(self) -> str:
        """Type/sequence name of the experiment."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get experiment parameters."""

    @abstractmethod
    def load_data(self) -> Optional[xr.DataArray]:
        """Load raw data."""

    @classmethod
    @abstractmethod
    def is_valid_scan_dir(cls, path: Union[str, Path]) -> bool:
        """Check if the path contains valid scan data for this vendor."""


class BaseReconstructionStrategy(ABC):
    """Abstract base class for reconstruction strategies."""

    @abstractmethod
    def reconstruct(
        self, handler: VendorHandlerProtocol, **kwargs
    ) -> Optional[xr.DataArray]:
        """Perform reconstruction using the provided handler."""

    def validate_options(self, **kwargs) -> Dict[str, Any]:
        """Validate and process reconstruction options."""
        return kwargs


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Attach configuration options to the plotter instance.

        Parameters
        ----------
        config
            Optional mapping that stores default keyword arguments for concrete
            plotting implementations.
        """
        self.config = config or {}

    @abstractmethod
    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """Create a plot of the provided data."""


class BaseRegistry(ABC):
    """Abstract base class for registries."""

    def __init__(self):
        """Initialise an empty registry dictionary."""
        self._registry = {}

    def register(self, key: str, value: Any, overwrite: bool = False) -> None:
        """Register a value under ``key``.

        Parameters
        ----------
        key
            Canonical registry key.
        value
            Object stored under ``key``.
        overwrite
            When ``False`` (default) the method raises ``KeyError`` if the key
            already exists. Setting it to ``True`` replaces the previous value.

        Raises
        ------
        KeyError
            If ``overwrite`` is ``False`` and ``key`` already exists.
        """
        if key in self._registry and not overwrite:
            raise KeyError(
                f"Key '{key}' already registered. Use overwrite=True to replace."
            )
        self._registry[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Return the object registered under ``key``.

        Parameters
        ----------
        key
            Registry key to look up.

        Returns
        -------
        Any or None
            Registered object or ``None`` if the key is missing.
        """
        return self._registry.get(key)

    def list_registered(self) -> list:
        """List all registered keys.

        Returns
        -------
        list[str]
            Snapshot of keys currently stored in the registry.
        """
        return list(self._registry.keys())

    def unregister(self, key: str) -> None:
        """Remove ``key`` from the registry if present.

        Parameters
        ----------
        key
            Registry key to delete.
        """
        if key in self._registry:
            del self._registry[key]

    @abstractmethod
    def _load_configurations(self) -> None:
        """Load configurations from external sources."""


class SequenceProtocol(Protocol):
    """Protocol defining the interface for sequence classes."""

    @property
    def primary_data(self) -> Optional[xr.DataArray]:
        """Get the primary data array."""
        ...

    def reconstruct(self, reco_type: str, **kwargs) -> Optional[xr.DataArray]:
        """Perform reconstruction."""
        ...


class AnalysisProtocol(Protocol):
    """Protocol for analysis components."""

    def analyze(self, data: xr.DataArray, **kwargs) -> Any:
        """Perform analysis on the provided data."""
        ...
