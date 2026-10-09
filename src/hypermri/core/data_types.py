"""
Standardized data types and containers for the hypermri package.

This module defines data structures used throughout the hypermri ecosystem
to ensure consistent data handling and type safety.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import xarray as xr


class VendorType(Enum):
    """Supported MRI vendors."""

    BRUKER = "bruker"
    SIEMENS = "siemens"
    PHILIPS = "philips"
    GE = "ge"


class SequenceCategory(Enum):
    """Categories of MRI sequences."""

    ANATOMICAL = "anatomical"
    SPECTROSCOPY = "spectroscopy"
    DIFFUSION = "diffusion"
    PERFUSION = "perfusion"
    FUNCTIONAL = "functional"


class ReconstructionType(Enum):
    """Types of reconstruction."""

    RAW = "raw"
    SEQ2D = "seq2d"
    KSPACE = "kspace"
    CUSTOM = "custom"


class DataDomain(Enum):
    """Data domain types."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    KSPACE = "kspace"
    FREQUENCY = "frequency"
    MIXED = "mixed"


@dataclass
class ExperimentMetadata:
    """Metadata for MRI experiments."""

    id: str
    sequence_name: str
    vendor: VendorType
    category: SequenceCategory
    timestamp: Optional[str] = None
    subject_id: Optional[str] = None
    study_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionOptions:
    """Options for reconstruction strategies."""

    reco_type: ReconstructionType
    options: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the stored option for ``key``.

        Parameters
        ----------
        key
            Option name.
        default
            Value returned when the option is not present.

        Returns
        -------
        Any
            Stored value or ``default`` if the option is missing.
        """
        return self.options.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``.

        Parameters
        ----------
        key
            Option name.
        value
            Option value to store.
        """
        self.options[key] = value

    def update(self, **kwargs) -> None:
        """Update the option dictionary with keyword arguments."""
        self.options.update(kwargs)


@dataclass
class DataArrayInfo:
    """Information about data arrays."""

    shape: Tuple[int, ...]
    dtype: str
    dimensions: List[str]
    coordinates: Dict[str, Any]
    attributes: Dict[str, Any]
    domain: DataDomain

    @classmethod
    def from_data_array(cls, data_array: xr.DataArray) -> "DataArrayInfo":
        """Create :class:`DataArrayInfo` metadata from an ``xarray`` object.

        Parameters
        ----------
        data_array
            ``xarray.DataArray`` instance to summarise.

        Returns
        -------
        DataArrayInfo
            Populated dataclass instance mirroring the array's structure and
            attributes.
        """
        return cls(
            shape=data_array.shape,
            dtype=str(data_array.dtype),
            dimensions=list(data_array.dims),
            coordinates=dict(data_array.coords),
            attributes=dict(data_array.attrs),
            domain=DataDomain(data_array.attrs.get("domain", "spatial")),
        )


@dataclass
class PlotConfiguration:
    """Configuration for plotting."""

    plot_type: str
    figure_size: Tuple[float, float] = (10, 8)
    colormap: str = "viridis"
    title: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    style_options: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = True

    def merge(self, other: "PlotConfiguration") -> "PlotConfiguration":
        """Return a configuration merged with ``other``.

        Parameters
        ----------
        other
            Configuration whose values supersede the current instance.

        Returns
        -------
        PlotConfiguration
            New instance that combines labels, limits, and options from both
            configurations.
        """
        merged_labels = {**self.labels, **other.labels}
        merged_limits = {**self.limits, **other.limits}
        merged_style = {**self.style_options, **other.style_options}

        return PlotConfiguration(
            plot_type=other.plot_type or self.plot_type,
            figure_size=other.figure_size or self.figure_size,
            colormap=other.colormap or self.colormap,
            title=other.title or self.title,
            labels=merged_labels,
            limits=merged_limits,
            style_options=merged_style,
            interactive=(
                other.interactive if other.interactive is not None else self.interactive
            ),
        )


@dataclass
class RegistryEntry:
    """Entry in a registry."""

    key: str
    value: Any
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_filter(self, filters: Dict[str, Any]) -> bool:
        """Return ``True`` when ``filters`` match the entry metadata.

        Parameters
        ----------
        filters
            Mapping of metadata keys to required values. The special key
            ``"category"`` is matched against the ``category`` attribute.

        Returns
        -------
        bool
            ``True`` if all provided filters match; ``False`` otherwise.
        """
        for filter_key, filter_value in filters.items():
            if filter_key == "category" and self.category != filter_value:
                return False
            elif (
                filter_key in self.metadata
                and self.metadata[filter_key] != filter_value
            ):
                return False
        return True


class ValidationResult:
    """Result of data validation."""

    def __init__(
        self,
        valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """Create a validation result object.

        Parameters
        ----------
        valid
            Overall validity flag. Defaults to ``True``.
        errors
            Optional list of validation error messages.
        warnings
            Optional list of validation warnings.
        """
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str) -> None:
        """Append an error message and mark the result invalid."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Append a warning message without changing validity."""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combine this result with ``other``.

        Parameters
        ----------
        other
            Another :class:`ValidationResult` instance.

        Returns
        -------
        ValidationResult
            New instance whose validity is the logical ``AND`` of both inputs
            and whose messages are concatenated.
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )

    def __bool__(self) -> bool:
        """Return the boolean validity flag."""
        return self.valid


@dataclass
class AnalysisResult:
    """Result of analysis operations."""

    result_type: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the analysis result to a dictionary."""
        return {
            "result_type": self.result_type,
            "data": self.data,
            "metadata": self.metadata,
            "parameters": self.parameters,
        }


class PathManager:
    """Utility class for managing file paths."""

    def __init__(self, base_path: Union[str, Path]):
        """Store the normalised base path used for relative operations.

        Parameters
        ----------
        base_path
            Reference directory from which relative paths are computed.
            Relative inputs are resolved against the current working directory.
        """
        self.base_path = Path(base_path).resolve()

    def get_relative_path(self, path: Union[str, Path]) -> Path:
        """Return ``path`` relative to ``base_path`` when possible.

        Parameters
        ----------
        path
            Filesystem path to normalise.

        Returns
        -------
        pathlib.Path
            Relative path when ``path`` resides within ``base_path``;
            otherwise the absolute path is returned unchanged.
        """
        path = Path(path).resolve()
        try:
            return path.relative_to(self.base_path)
        except ValueError:
            return path

    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """Create ``path`` if required and return the resulting directory.

        Parameters
        ----------
        path
            Directory path, interpreted relative to ``base_path`` unless it is
            already absolute.

        Returns
        -------
        pathlib.Path
            Absolute directory path after creation.
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.base_path / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_files(self, pattern: str = "*", recursive: bool = False) -> List[Path]:
        """List files below ``base_path`` that match ``pattern``.

        Parameters
        ----------
        pattern
            Glob pattern applied to entries under ``base_path``.
        recursive
            When ``True`` the search descends into sub-directories via
            :py:meth:`Path.rglob`.

        Returns
        -------
        list[pathlib.Path]
            Collection of matching file paths.
        """
        if recursive:
            return list(self.base_path.rglob(pattern))
        else:
            return list(self.base_path.glob(pattern))
