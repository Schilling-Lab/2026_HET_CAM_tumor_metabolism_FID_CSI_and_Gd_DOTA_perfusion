"""
Custom exception classes for the hypermri package.

This module defines a hierarchy of exceptions that provide clear error
handling throughout the hypermri ecosystem.
"""


class HyperMRIError(Exception):
    """Base exception for all hypermri-related errors."""


class DataLoadingError(HyperMRIError):
    """Raised when data loading fails."""

    def __init__(self, message: str, path: str = None, vendor: str = None):
        """Initialise the error with optional context metadata.

        Parameters
        ----------
        message
            Human readable explanation of the failure.
        path
            Filesystem location of the dataset that triggered the error.
        vendor
            Vendor identifier associated with the failing dataset.
        """
        super().__init__(message)
        self.path = path
        self.vendor = vendor


class ReconstructionError(HyperMRIError):
    """Raised when reconstruction fails."""

    def __init__(self, message: str, strategy: str = None, sequence: str = None):
        """Capture strategy metadata when reconstruction cannot proceed.

        Parameters
        ----------
        message
            Description of the reconstruction failure.
        strategy
            Fully qualified strategy identifier, if known.
        sequence
            Sequence name associated with the failing reconstruction.
        """
        super().__init__(message)
        self.strategy = strategy
        self.sequence = sequence


class SequenceError(HyperMRIError):
    """Raised when sequence-related operations fail."""

    def __init__(self, message: str, sequence_name: str = None):
        """Store contextual information about the affected sequence.

        Parameters
        ----------
        message
            Description of the sequence issue.
        sequence_name
            Name of the sequence involved in the failure.
        """
        super().__init__(message)
        self.sequence_name = sequence_name


class VendorError(HyperMRIError):
    """Raised when vendor-specific operations fail."""

    def __init__(self, message: str, vendor: str = None):
        """Record the vendor responsible for the error.

        Parameters
        ----------
        message
            Description of the vendor-specific failure.
        vendor
            Vendor identifier (for example ``"bruker"``).
        """
        super().__init__(message)
        self.vendor = vendor


class RegistryError(HyperMRIError):
    """Raised when registry operations fail."""

    def __init__(self, message: str, registry_type: str = None, key: str = None):
        """Describe errors that occur when interacting with registries.

        Parameters
        ----------
        message
            Error description.
        registry_type
            Human readable name of the registry that triggered the error.
        key
            Registry key involved in the failing operation.
        """
        super().__init__(message)
        self.registry_type = registry_type
        self.key = key


class ConfigurationError(HyperMRIError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_type: str = None, config_path: str = None):
        """Capture configuration metadata for debugging.

        Parameters
        ----------
        message
            Error message detailing the configuration issue.
        config_type
            Logical grouping of the configuration file or section.
        config_path
            Filesystem path to the configuration resource.
        """
        super().__init__(message)
        self.config_type = config_type
        self.config_path = config_path


class ValidationError(HyperMRIError):
    """Raised when data validation fails."""

    def __init__(
        self, message: str, data_type: str = None, validation_rule: str = None
    ):
        """Include data type and rule information for failed validations.

        Parameters
        ----------
        message
            Description of the validation failure.
        data_type
            Optional label describing the validated data kind.
        validation_rule
            Rule identifier or short description of the violated constraint.
        """
        super().__init__(message)
        self.data_type = data_type
        self.validation_rule = validation_rule


class AnalysisError(HyperMRIError):
    """Raised when analysis operations fail."""

    def __init__(self, message: str, analysis_type: str = None):
        """Wrap errors that occur during analytical pipelines.

        Parameters
        ----------
        message
            Description of the analysis failure.
        analysis_type
            Category or identifier for the failing analysis routine.
        """
        super().__init__(message)
        self.analysis_type = analysis_type


class PlottingError(HyperMRIError):
    """Raised when plotting operations fail."""

    def __init__(self, message: str, plot_type: str = None):
        """Record the plot type that triggered the failure.

        Parameters
        ----------
        message
            Description of the plotting problem.
        plot_type
            Type of plot that failed to render.
        """
        super().__init__(message)
        self.plot_type = plot_type


class CompatibilityError(HyperMRIError):
    """Raised when compatibility issues are detected."""

    def __init__(self, message: str, old_version: str = None, new_version: str = None):
        """Explain incompatibilities between versions.

        Parameters
        ----------
        message
            Description of the compatibility issue.
        old_version
            Version the asset was produced with or previously supported.
        new_version
            Version that triggered the compatibility warning.
        """
        super().__init__(message)
        self.old_version = old_version
        self.new_version = new_version


class MissingDependencyError(HyperMRIError):
    """Raised when required dependencies are missing."""

    def __init__(self, message: str, dependency: str = None, feature: str = None):
        """Highlight missing third-party requirements.

        Parameters
        ----------
        message
            Description of the missing dependency.
        dependency
            Name of the package or module that is required.
        feature
            Feature or component that cannot be used without the dependency.
        """
        super().__init__(message)
        self.dependency = dependency
        self.feature = feature
