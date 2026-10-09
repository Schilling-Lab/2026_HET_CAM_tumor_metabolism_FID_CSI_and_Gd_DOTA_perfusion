from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hypermri")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

# Import core classes
try:
    from .core import (
        MRIExperiment,
        create_experiment,
        load_external_experiment,
    )
except ImportError as e:
    print(f"Critical error: Could not import hypermri core components: {e}")
    raise

# Import other components with optional fallbacks
try:
    from .brukerexp import BrukerHandler
except ImportError:
    BrukerHandler = None

try:
    from .agilentexp import AgilentHandler
except ImportError:
    AgilentHandler = None

try:
    from .study import Study
except ImportError:
    Study = None

try:
    from .mixins import (
        ParameterAccessMixin,
        StandardizedDataArrayMixin,
        StandardizedMRIMixin,
        StandardizedPathMixin,
    )
except ImportError:
    StandardizedMRIMixin = object  # Fallback base class
    StandardizedDataArrayMixin = object
    StandardizedPathMixin = object
    ParameterAccessMixin = object

try:
    from .compatibility import (
        check_data_array_consistency,
        migrate_object_attributes,
    )
except ImportError:

    def check_data_array_consistency(*args, **kwargs):
        pass

    def migrate_object_attributes(*args, **kwargs):
        pass


# Make the new implementation the default
__all__ = [
    "__version__",
    "MRIExperiment",
    "create_experiment",
    "load_external_experiment",
]

# Add optional components to __all__ if they exist
if BrukerHandler is not None:
    __all__.append("BrukerHandler")
if "AgilentHandler" in locals() and AgilentHandler is not None:
    __all__.append("AgilentHandler")
if Study is not None:
    __all__.append("Study")
if StandardizedMRIMixin != object:
    __all__.extend(
        [
            "StandardizedMRIMixin",
            "StandardizedDataArrayMixin",
            "StandardizedPathMixin",
            "ParameterAccessMixin",
        ]
    )
    __all__.extend(["check_data_array_consistency", "migrate_object_attributes"])

# Optional import of visualization system (Phase 4 - Visualization Unification)
try:
    from .visualization.plotter_factory import (
        create_anatomy_plotter,
        create_plotter,
        create_quantitative_plotter,
        create_spectroscopy_plotter,
        create_unified_plotter,
        list_available_plotters,
    )
    from .visualization.registry import get_plotter_registry

    __all__.extend(
        [
            "create_plotter",
            "create_anatomy_plotter",
            "create_spectroscopy_plotter",
            "create_quantitative_plotter",
            "create_unified_plotter",
            "list_available_plotters",
            "get_plotter_registry",
        ]
    )
except ImportError:
    # Visualization system is optional
    pass
