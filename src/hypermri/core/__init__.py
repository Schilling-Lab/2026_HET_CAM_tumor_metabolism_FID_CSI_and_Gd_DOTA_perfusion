"""
Core module for hypermri package.

This module provides the fundamental abstractions, interfaces, and data types
used throughout the hypermri ecosystem.
"""

from .data_types import (
    AnalysisResult,
    DataArrayInfo,
    DataDomain,
    ExperimentMetadata,
    PathManager,
    PlotConfiguration,
    ReconstructionOptions,
    ReconstructionType,
    RegistryEntry,
    SequenceCategory,
    ValidationResult,
    VendorType,
)
from .exceptions import (
    AnalysisError,
    CompatibilityError,
    ConfigurationError,
    DataLoadingError,
    HyperMRIError,
    MissingDependencyError,
    PlottingError,
    ReconstructionError,
    RegistryError,
    SequenceError,
    ValidationError,
    VendorError,
)
from .experiment import MRIExperiment
from .factory import (
    create_experiment,
    load_external_experiment,
)
from .interfaces import (
    AnalysisProtocol,
    BasePlotter,
    BaseReconstructionStrategy,
    BaseRegistry,
    BaseVendorHandler,
    PlotterProtocol,
    ReconstructionStrategyProtocol,
    RegistryProtocol,
    SequenceProtocol,
    VendorHandlerProtocol,
)

__all__ = [
    # Interfaces
    "VendorHandlerProtocol",
    "ReconstructionStrategyProtocol",
    "PlotterProtocol",
    "RegistryProtocol",
    "SequenceProtocol",
    "AnalysisProtocol",
    "BaseVendorHandler",
    "BaseReconstructionStrategy",
    "BasePlotter",
    "BaseRegistry",
    # Data Types
    "VendorType",
    "SequenceCategory",
    "ReconstructionType",
    "DataDomain",
    "ExperimentMetadata",
    "ReconstructionOptions",
    "DataArrayInfo",
    "PlotConfiguration",
    "RegistryEntry",
    "ValidationResult",
    "AnalysisResult",
    "PathManager",
    # Exceptions
    "HyperMRIError",
    "DataLoadingError",
    "ReconstructionError",
    "SequenceError",
    "VendorError",
    "RegistryError",
    "ConfigurationError",
    "ValidationError",
    "AnalysisError",
    "PlottingError",
    "CompatibilityError",
    "MissingDependencyError",
    # Experiment classes
    "MRIExperiment",
    "create_experiment",
    "load_external_experiment",
]
