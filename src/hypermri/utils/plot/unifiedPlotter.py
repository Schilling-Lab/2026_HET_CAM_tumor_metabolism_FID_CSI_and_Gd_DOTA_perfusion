import traceback
import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from ipywidgets import (
    HTML,
    Checkbox,
    Dropdown,
    FloatSlider,
    FloatText,
    HBox,
    IntSlider,
    Layout,
    ToggleButton,
    ToggleButtons,
    VBox,
    fixed,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hypermri.analysis.denoising.algorithms import denoise_pca
from hypermri.analysis.spectroscopy.core import multi_dim_linebroadening
from hypermri.utils.coordinates import get_xyz_extent_for_coordinate
from hypermri.utils.plot.plotter import BasePlotter
from hypermri.utils.plot.utils_plot import common_cmaps
from hypermri.utils.system.logging import debug_log, init_default_logger

logger = init_default_logger(__name__)


DEBUG_LOG = True  # Enable debug logging for widget debugging
DEBUG_FILE = "/tmp/unified_plotter_debug.log"  # Use widget-specific debug file


def debug_log_unified_plotter(message, file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    debug_log(message=message, file_name=file_name, write_to_file=write_to_file)


class UnifiedPlotData:
    """Standardize plot data representation across MRI data types.

    Attributes
    ----------
    spatial_data : xr.DataArray or None
        Spatial data used for image display.
    spectral_data : xr.DataArray or None
        Spectral data used for spectrum plots.
    raw_spectral_data : xr.DataArray or None
        Original measured spectroscopy data when available.
    fitted_spectra : dict
        Mapping of metabolite names to fitted spectra.
    metadata : dict
        Plot metadata such as labels and units.
    data_type : str or None
        Data classification, for example ``"anatomy"`` or ``"spectroscopy_fitted"``.
    source_type : str or None
        Source indicator, ``"dataarray"`` or ``"dataset"``.
    metabolite_names : list
        Clean metabolite names for legend display.
    is_overlay : bool
        ``True`` when the data should be plotted as an overlay.
    widget_prefix : str
        Prefix used for widget names (for example ``"ds0"``).
    overlay_interactions : dict
        Cross-modal interaction settings used by the plotting framework.
    """

    def __init__(self):
        self.spatial_data = None
        self.spectral_data = None
        self.raw_spectral_data = None
        self.fitted_spectra = {}
        self.metadata = {}
        # 'anatomy', 'spectroscopy_measured', 'spectroscopy_fitted', 'parameter_map'
        self.data_type = None
        self.source_type = None  # 'dataarray', 'dataset'
        self.metabolite_names = []
        self.is_overlay = False
        self.widget_prefix = ""
        self.overlay_interactions = {}  # For create_mask and other interactive overlays

    def has_spatial_data(self):
        """Check if spatial data is available for image display."""
        return self.spatial_data is not None

    def has_spectral_data(self):
        """Check if spectral data is available for spectrum display."""
        return self.spectral_data is not None or bool(self.fitted_spectra)

    def has_fitted_data(self):
        """Check if this contains fitted/reconstructed data."""
        return self.data_type == "spectroscopy_fitted" and bool(self.fitted_spectra)

    def is_anatomical(self):
        """Check if this is anatomical data."""
        result = self.data_type in [
            "anatomy",
            "parameter_map",
            "multi_variable_dataset",
        ]
        return result

    def is_spectroscopic(self):
        """Check if this is spectroscopic data (measured or fitted)."""
        return self.data_type in ["spectroscopy_measured", "spectroscopy_fitted"]

    def is_pure_spectroscopy(self):
        """Check if this is pure spectroscopy data (no meaningful spatial component)."""
        return self.is_spectroscopic() and not self.has_spatial_data()

    def is_spatial_spectroscopy(self):
        """Check if this is spatial spectroscopy data (has meaningful spatial component)."""
        return self.is_spectroscopic() and self.has_spatial_data()

    def get_metabolite_spectrum(self, metabolite_name):
        """Get spectrum for a specific metabolite."""
        return self.fitted_spectra.get(metabolite_name, None)


class UnifiedPlotter(BasePlotter):
    """Unified plotting system supporting anatomical and spectroscopic data with overlays.

    This class provides a consistent plotting interface that can handle:
    - Pure anatomical data visualization
    - Pure spectroscopic data visualization
    - Mixed anatomical + spectroscopic overlays
    - Interactive features like create_mask integration
    - Cross-modal interactions (anatomy clicks → spectra, spectrum clicks → anatomy)
    """

    def __init__(
        self,
        data_obj,
        preprocessor=None,
        cache_manager=None,
        data_selector=None,
        widget_factory=None,
    ):
        """Initialize the unified plotter.

        Parameters
        ----------
        data_obj : object
            Data provider that supplies anatomical or spectroscopy content.
        preprocessor : DataPreprocessor, optional
            Preprocessing helper used to transform data before rendering. A default
            instance is created when omitted.
        cache_manager : CacheManager, optional
            Cache manager used to reuse intermediate results. A default instance is
            used when omitted.
        data_selector : DataSelector, optional
            Selector responsible for spatial dimension handling. A default instance is
            created when omitted.
        widget_factory : WidgetFactory, optional
            Factory that builds the interactive widgets. A default instance is created
            when omitted.
        """
        super().__init__()
        self.data_obj = data_obj
        self._click_positions = {}  # Unified click position storage
        self._plot_data_list = []  # Store processed plot data
        self.update_function = None  # Will store the update function reference
        self._updating_from_click = False  # Flag to prevent feedback loops

        # Initialize helper components with sensible defaults for backward compatibility
        from hypermri.utils.plot.cache_manager import CacheManager
        from hypermri.utils.plot.data_preprocessor import DataPreprocessor
        from hypermri.utils.plot.data_selector import DataSelector
        from hypermri.utils.plot.widget_factory import WidgetFactory

        self.preprocessor = preprocessor or DataPreprocessor(data_obj)
        self.cache_manager = cache_manager or CacheManager(max_size=10)
        self.data_selector = data_selector or DataSelector()
        self.widget_factory = widget_factory or WidgetFactory(
            processed_plot_params=None
        )

        # Interactive overlays support (for create_mask etc.)
        self.interactive_overlays = {}
        self.overlay_callbacks = {}

        # Track active perspective for 3D data
        self._active_perspective_dim = (
            None  # Track which dimension is being actively sliced
        )

    def _set_active_perspective(self, dim_name):
        """Update active perspective across plotter components."""
        self._active_perspective_dim = dim_name
        if hasattr(self, "data_selector") and self.data_selector:
            try:
                self.data_selector.set_active_perspective(dim_name)
            except Exception:
                # Best effort: keep legacy behaviour if selector signature differs
                pass

    def plot(
        self, input_array_list: list = None, plot_params: dict = None, **slider_defaults
    ):
        """Main plotting method for the UnifiedPlotter.

        This method determines the type of plot to generate based on the input data.
        Supports anatomical data, spectroscopic data, and mixed overlays.

        Parameters
        ----------
        input_array_list : list of (xr.DataArray or xr.Dataset), optional
            Data objects to plot. When omitted, no plot is generated.
        plot_params : dict, optional
            Plot customization dictionary.
        **slider_defaults
            Additional keyword arguments used to initialize widget state.
        """
        # Enhanced input validation
        try:
            self.processed_plot_params = self._process_plot_params(plot_params)
        except Exception as e:
            warnings.warn(f"Error processing plot parameters: {e}. Using defaults.")
            self.processed_plot_params = self._process_plot_params(None)

        if not input_array_list:
            print("No data provided to plot.")
            return

        # Validate input types
        if not isinstance(input_array_list, (list, xr.DataArray, xr.Dataset)):
            # Check if it's a scan-like object
            if not (
                hasattr(input_array_list, "reco_data")
                or hasattr(input_array_list, "data_array")
            ):
                raise TypeError(
                    "input_array_list must be a list, xarray.DataArray, xarray.Dataset, "
                    "or a scan object with reco_data/data_array."
                )

        # Try unified plotting
        try:
            self._plot_unified(input_array_list, **slider_defaults)
        except Exception as e:
            warnings.warn(f"Unified plotting failed: {e}.")
            traceback.print_exc()

    def _plot_unified(self, input_array_list, **slider_defaults):
        """New unified plotting method that handles all input types consistently.

        Parameters
        ----------
        input_array_list : list
            xarray objects to plot.
        **slider_defaults
            Default widget values provided by the caller.
        """
        debug_log_unified_plotter(
            f"_plot_unified called with {len(input_array_list)} items"
        )
        try:
            # Convert all inputs to standardized UnifiedPlotData objects
            self._plot_data_list = self._prepare_plot_data_list(
                input_array_list, slider_defaults
            )

            if not self._plot_data_list:
                print("No valid data to plot.")
                return

            # Create widgets based on the plot data
            controls_box, self.widgets = self._create_unified_widgets(
                self._plot_data_list, self.processed_plot_params, slider_defaults
            )

            # Set up the unified update function
            def update_plot(**kwargs):
                self._render_unified_plot(self._plot_data_list, kwargs)

            self.update_function = update_plot

            # Connect interactive widgets
            interactive_widgets = {
                k: v for k, v in self.widgets.items() if hasattr(v, "value")
            }
            output = widgets.interactive_output(update_plot, interactive_widgets)
            display(VBox([controls_box, output]))
            # Show initial plot
            update_plot(
                **{k: v.value for k, v in self.widgets.items() if hasattr(v, "value")}
            )

        except Exception as e:
            warnings.warn(f"Error in unified plotting: {e}.")
            traceback.print_exc()

    def _prepare_plot_data_list(self, input_array_list, kwargs):
        """Convert any input combination to standardized UnifiedPlotData list.

        Parameters
        ----------
        input_array_list : list
            Collection of xarray objects representing the requested plots.
        kwargs : dict
            Widget parameters and settings.

        Returns
        -------
        list of UnifiedPlotData
            Standardized plot data objects.
        """
        debug_log_unified_plotter(f"_prepare_plot_data_list called")
        if not isinstance(input_array_list, list):
            input_array_list = [input_array_list]

        plot_data_list = []

        for i, input_item in enumerate(input_array_list):
            is_overlay = i > 0  # First item is base, rest are overlays
            widget_prefix = f"ds{i}"

            # If the input item is a scan object (like StackedScanObject or MRIExperiment),
            # extract its primary data array.
            if not isinstance(input_item, (xr.DataArray, xr.Dataset)):
                if (
                    hasattr(input_item, "reco_data")
                    and input_item.reco_data is not None
                ):
                    input_item = input_item.reco_data
                elif (
                    hasattr(input_item, "data_array")
                    and input_item.data_array is not None
                ):
                    input_item = input_item.data_array

            if isinstance(input_item, xr.DataArray):
                plot_data = self._prepare_dataarray_plot_data(
                    input_item, kwargs, is_overlay, widget_prefix
                )
            elif isinstance(input_item, xr.Dataset):
                plot_data = self._prepare_dataset_plot_data(
                    input_item,
                    kwargs,
                    is_overlay,
                    widget_prefix,
                    context={"previous_plot_data": plot_data_list},
                )
            else:
                raise TypeError(f"Unsupported input type: {type(input_item)}")

            plot_data_list.append(plot_data)

        return plot_data_list

    def _prepare_dataarray_plot_data(
        self, data_array, kwargs, is_overlay, widget_prefix
    ):
        """Process xr.DataArray into UnifiedPlotData format.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array to standardize.
        kwargs : dict
            Widget parameters controlling preprocessing.
        is_overlay : bool
            ``True`` when the data should be treated as an overlay.
        widget_prefix : str
            Prefix used for widget identifiers.

        Returns
        -------
        UnifiedPlotData
            Standardized plot data object.
        """
        debug_log_unified_plotter(f"_prepare_dataarray_plot_data called")
        plot_data = UnifiedPlotData()
        plot_data.source_type = "dataarray"
        plot_data.is_overlay = is_overlay
        plot_data.widget_prefix = widget_prefix

        # Determine data type based on attributes and dimensions
        is_spectroscopic = data_array.attrs.get("is_spectroscopic", False)
        has_acquisition_dim = "acquisition_idx" in data_array.dims

        if is_spectroscopic or has_acquisition_dim:
            plot_data.data_type = "spectroscopy_measured"
            # Apply preprocessing (denoising, line broadening, domain
            # transforms)
            processed_data = self._apply_preprocessing(
                data_array, kwargs, widget_prefix
            )

            # Check if data has meaningful spatial dimensions
            spatial_dims = self._get_spatial_dims(processed_data)

            has_spatial_dims = (
                len(spatial_dims) >= 2
            )  # Need at least 2D for spatial display
            debug_log_unified_plotter(
                f"spatial_dims={spatial_dims}/has_spatial_dims={has_spatial_dims}"
            )

            # For spectroscopy data, check if spatial dimensions are meaningful
            # (i.e., have size > 1). Single-voxel spectroscopy should be treated as pure spectral data
            meaningful_spatial = False
            if has_spatial_dims:
                spatial_sizes = [
                    processed_data.sizes.get(dim, 1) for dim in spatial_dims[:2]
                ]
                meaningful_spatial = any(size > 1 for size in spatial_sizes)

                # Additional check: for single-voxel NSPECT, all spatial dims typically have size 1
                is_single_voxel = all(size == 1 for size in spatial_sizes)
                if is_single_voxel:
                    meaningful_spatial = False

            if meaningful_spatial:
                plot_data.spatial_data = processed_data
            else:
                # Pure spectroscopy data - no spatial component
                plot_data.spatial_data = None
            plot_data.spectral_data = processed_data
        else:
            plot_data.data_type = "anatomy"
            # For anatomical data, apply basic processing
            processed_data = self._apply_preprocessing(
                data_array, kwargs, widget_prefix
            )
            plot_data.spatial_data = processed_data

        plot_data.raw_spectral_data = data_array  # Keep original

        # Extract metadata
        plot_data.metadata = {
            "domain": data_array.attrs.get("domain", "spatial"),
            "signal_domain": data_array.attrs.get("signal_domain", "temporal"),
            "is_spectroscopic": is_spectroscopic,
        }

        return plot_data

    def _prepare_dataset_plot_data(
        self, dataset, kwargs, is_overlay, widget_prefix, context=None
    ):
        """Process xr.Dataset (fitted data or parameter maps) into UnifiedPlotData format.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing fitted data or parameter maps.
        kwargs : dict
            Widget parameters controlling preprocessing.
        is_overlay : bool
            ``True`` when the dataset should be treated as an overlay.
        widget_prefix : str
            Prefix used for widget identifiers.
        context : dict, optional
            Contextual information (e.g. previous plot data).

        Returns
        -------
        UnifiedPlotData
            Standardized plot data object.
        """
        plot_data = UnifiedPlotData()
        plot_data.source_type = "dataset"
        plot_data.is_overlay = is_overlay
        plot_data.widget_prefix = widget_prefix

        # Check if this is fitted spectroscopic data
        if hasattr(dataset, "hp"):
            model = dataset.attrs.get("model_name", None)

            # Define models that should be treated as multi-variable datasets
            multi_var_models = [
                "t1_inversion_recovery",
                "monoexponential_adc",
                "patlak",
                "two_compartment_model",
                "tofts_model",
                "b1_sinusoidal",
            ]

            if model == "FID":
                plot_data.data_type = "spectroscopy_fitted"

                # Extract clean metabolite names
                if "metabolite" in dataset.coords:
                    plot_data.metabolite_names = [
                        str(m) for m in dataset.coords["metabolite"].values
                    ]

                # Use fit_amps for spatial display (magnitude) - but only if spatial dimensions are meaningful
                fit_amps_data = np.abs(dataset["fit_amps"])

                # Apply same spatial validation logic as for DataArrays
                spatial_dims = self._get_spatial_dims(fit_amps_data)
                has_spatial_dims = len(spatial_dims) >= 2

                # Check if spatial dimensions are meaningful (size > 1)
                meaningful_spatial = False
                if has_spatial_dims:
                    spatial_sizes = [
                        fit_amps_data.sizes.get(dim, 1) for dim in spatial_dims[:2]
                    ]
                    meaningful_spatial = any(size > 1 for size in spatial_sizes)

                    # Additional check: for single-voxel data, all spatial dims typically have size 1
                    is_single_voxel = all(size == 1 for size in spatial_sizes)
                    if is_single_voxel:
                        meaningful_spatial = False

                if meaningful_spatial:
                    plot_data.spatial_data = fit_amps_data
                else:
                    # Pure spectroscopy fitted data - no spatial component
                    plot_data.spatial_data = None

                # Reconstruct fitted spectra early for consistent preprocessing
                has_spatial_data = any(
                    pd.has_spatial_data() for pd in getattr(self, "_plot_data_list", [])
                )
                should_reconstruct = self._has_click_position() or not has_spatial_data

                if should_reconstruct:
                    reconstructed_data = (
                        self._reconstruct_fitted_spectra_for_preprocessing(
                            dataset, kwargs, widget_prefix
                        )
                    )
                    if reconstructed_data is not None:
                        # Apply the same preprocessing as measured data
                        plot_data.spectral_data = self._apply_preprocessing(
                            reconstructed_data, kwargs, widget_prefix
                        )
                        plot_data.raw_spectral_data = reconstructed_data

                # Store dataset for spectrum reconstruction
                plot_data.fitted_spectra = {"_dataset": dataset}

                # Support "Reconstructed" (FIDs/Spectra) images as overlay if requested
                if context and "previous_plot_data" in context:
                    for prev_pd in context["previous_plot_data"]:
                        candidate_data = (
                            prev_pd.spatial_data
                            if prev_pd.spatial_data is not None
                            else prev_pd.raw_spectral_data
                        )
                        if candidate_data is not None:
                            kwarg, coord = (
                                self._find_dynamic_coordinate_for_reconstruction(
                                    model, candidate_data, dataset
                                )
                            )
                            if kwarg and coord is not None:
                                try:
                                    reconstructed = dataset.hp.reconstruct(
                                        **{kwarg: coord}
                                    )
                                    plot_data.dataset_variables = {
                                        "Reconstructed": reconstructed
                                    }
                                    plot_data.dataset_variable_names = ["Reconstructed"]
                                    plot_data.data_type = "multi_variable_dataset"  # Switch type to enable variable selection
                                    break
                                except Exception:
                                    pass

            elif model in multi_var_models:
                plot_data.data_type = "multi_variable_dataset"

                # Store all data variables for selection
                data_vars = list(dataset.data_vars.keys())
                plot_data.dataset_variables = {var: dataset[var] for var in data_vars}
                plot_data.dataset_variable_names = data_vars

                # Determine special settings for models
                if model == "tofts_model":
                    if "s0_map" in dataset or "s0" in dataset:
                        plot_data.reco_space = "signal"
                    else:
                        plot_data.reco_space = "concentration"

                # Try to reconstruct image series dynamically
                if context and "previous_plot_data" in context:
                    for prev_pd in context["previous_plot_data"]:
                        candidate_data = (
                            prev_pd.spatial_data
                            if prev_pd.spatial_data is not None
                            else prev_pd.raw_spectral_data
                        )
                        if candidate_data is not None:
                            kwarg, coord = (
                                self._find_dynamic_coordinate_for_reconstruction(
                                    model, candidate_data, dataset
                                )
                            )
                            if kwarg and coord is not None:
                                try:
                                    # Prepare reconstruction parameters
                                    reco_params = {kwarg: coord}
                                    if hasattr(plot_data, "reco_space"):
                                        reco_params["reco_space"] = plot_data.reco_space

                                    reconstructed = dataset.hp.reconstruct(
                                        **reco_params
                                    )
                                    plot_data.dataset_variables["Reconstructed"] = (
                                        reconstructed
                                    )
                                    if (
                                        "Reconstructed"
                                        not in plot_data.dataset_variable_names
                                    ):
                                        plot_data.dataset_variable_names.append(
                                            "Reconstructed"
                                        )
                                    debug_log_unified_plotter(
                                        f"✅ Successfully reconstructed {model} image series using {kwarg}"
                                    )
                                    break
                                except Exception as e:
                                    debug_log_unified_plotter(
                                        f"Reconstruction failed for {model}: {e}"
                                    )

                # Get currently selected variable from widgets (default to first)
                selected_var = kwargs.get(
                    f"{widget_prefix}_data_variable", data_vars[0]
                )
                if selected_var not in plot_data.dataset_variable_names:
                    selected_var = data_vars[0]

                # Use selected variable for spatial display
                if (
                    selected_var == "Reconstructed"
                    and "Reconstructed" in plot_data.dataset_variables
                ):
                    plot_data.spatial_data = plot_data.dataset_variables[
                        "Reconstructed"
                    ]
                else:
                    plot_data.spatial_data = dataset[selected_var]

                plot_data.current_data_variable = selected_var
                plot_data.fitted_spectra = {"_dataset": dataset}

            # For any other models or datasets without hp accessor, fall through to else block
        else:
            # This is a parameter map dataset - check if it has multiple data variables
            data_vars = list(dataset.data_vars.keys())

            if len(data_vars) > 1:
                # Multi-variable dataset - set up for variable selection
                plot_data.data_type = "multi_variable_dataset"

                # Store all data variables for selection
                plot_data.dataset_variables = {var: dataset[var] for var in data_vars}
                plot_data.dataset_variable_names = data_vars

                # Get currently selected variable from widgets (default to first)
                selected_var = kwargs.get(
                    f"{widget_prefix}_data_variable", data_vars[0]
                )
                if selected_var not in data_vars:
                    selected_var = data_vars[0]

                # Use selected variable for spatial display
                plot_data.spatial_data = dataset[selected_var]
                plot_data.current_data_variable = selected_var

            else:
                # Single-variable parameter map - use existing logic
                plot_data.data_type = "parameter_map"
                first_var = next(iter(dataset.data_vars.values()))
                plot_data.spatial_data = first_var

        # Extract metadata
        plot_data.metadata = dataset.attrs.copy()

        return plot_data

    def _apply_preprocessing(self, data_array, kwargs, widget_prefix):
        """Apply preprocessing using the shared DataPreprocessor helper.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data requiring preprocessing.
        kwargs : dict
            Widget state controlling preprocessing steps.
        widget_prefix : str
            Prefix used to namespace widget parameters.

        Returns
        -------
        xr.DataArray
            Preprocessed data ready for plotting.

        Notes
        -----
        Delegates operations such as dimension standardization, denoising,
        domain transforms, and channel combination to the
        :class:`DataPreprocessor` component.
        """
        # Check if a preprocessor helper is available
        if hasattr(self, "preprocessor") and self.preprocessor is not None:
            # Delegate all preprocessing to the DataPreprocessor component
            processed_data = self.preprocessor.apply_preprocessing(
                data_array, kwargs, widget_prefix
            )
            return processed_data

        # Legacy fallback for instances without an explicit preprocessor
        # This preserves backward compatibility
        processed_data = self._apply_preprocessing_legacy(
            data_array, kwargs, widget_prefix
        )
        return processed_data

    # ==================== DataSelector Delegation Methods ====================
    # These helpers delegate spatial operations to the DataSelector component for
    # consistent dimension handling across all plotters.

    def _get_all_spatial_dims_using_selector(self, data_array):
        """Get all spatial dimensions using DataSelector when available.

        Parameters
        ----------
        data_array : xr.DataArray
            Data array containing spatial dimensions.

        Returns
        -------
        list
            All spatial dimension names.
        """
        if hasattr(self, "data_selector") and self.data_selector:
            return self.data_selector.get_all_spatial_dims(data_array)
        # Fallback to parent implementation
        return self._get_all_spatial_dims(data_array)

    def _get_spatial_dims_using_selector(
        self, data_array, kwargs=None, widget_prefix=""
    ):
        """Get active spatial dimensions using DataSelector.

        Parameters
        ----------
        data_array : xr.DataArray
            Data array with spatial dimensions.
        kwargs : dict, optional
            Widget values used for perspective detection.
        widget_prefix : str, optional
            Prefix applied to widget names for namespacing.

        Returns
        -------
        list
            Active spatial dimensions, typically two for 2D displays.
        """
        if hasattr(self, "data_selector") and self.data_selector:
            try:
                return self.data_selector.get_spatial_dims(
                    data_array, kwargs or {}, widget_prefix
                )
            except Exception as e:
                # Log warning and fallback
                logger.debug(
                    f"DataSelector.get_spatial_dims failed: {e}, falling back to parent"
                )

        # Fallback to parent implementation
        return self._get_spatial_dims(data_array, kwargs, widget_prefix)

    def _detect_active_perspective_using_selector(self, all_spatial_dims, kwargs=None):
        """Detect the active perspective using DataSelector.

        Parameters
        ----------
        all_spatial_dims : list
            All spatial dimension names available in the dataset.
        kwargs : dict, optional
            Widget values used to determine the active perspective.

        Returns
        -------
        list
            Active perspective dimension names.
        """
        if hasattr(self, "data_selector") and self.data_selector:
            try:
                return self.data_selector.detect_active_perspective(
                    all_spatial_dims, kwargs or {}
                )
            except Exception as e:
                logger.debug(
                    f"DataSelector.detect_active_perspective failed: {e}, falling back to parent"
                )

        # Fallback to parent implementation
        return self._detect_active_perspective(all_spatial_dims, kwargs)

    def _apply_spatial_slicing_using_selector(
        self, spatial_data, kwargs, widget_prefix=""
    ):
        """Apply spatial slicing using DataSelector when available.

        Parameters
        ----------
        spatial_data : xr.DataArray
            Data array to slice.
        kwargs : dict
            Widget values controlling the slice.
        widget_prefix : str, optional
            Prefix applied to widget names for namespacing.

        Returns
        -------
        xr.DataArray
            Sliced data produced by the DataSelector helper or fallback logic.
        """
        if hasattr(self, "data_selector") and self.data_selector:
            try:
                # Get spatial dims for the selector using helper method
                spatial_dims = self._get_spatial_dims_using_selector(
                    spatial_data, kwargs, widget_prefix
                )

                return self.data_selector.apply_spatial_slicing(
                    spatial_data, spatial_dims, kwargs, widget_prefix
                )
            except Exception as e:
                logger.debug(
                    f"DataSelector.apply_spatial_slicing failed: {e}, falling back to parent"
                )

        # Fallback to parent implementation
        return self._apply_spatial_slicing(spatial_data, kwargs, widget_prefix)

    def _apply_preprocessing_legacy(self, data_array, kwargs, widget_prefix):
        """Legacy preprocessing implementation for backward compatibility.

        Notes
        -----
        Retained for cases where no :class:`DataPreprocessor` helper is
        available. New code should use :meth:`_apply_preprocessing`.
        """
        # Make a copy to avoid modifying original
        processed_data = data_array.copy(deep=True)

        # Standardize dimension ordering for consistent overlay alignment
        processed_data = self._standardize_dimension_order(processed_data)

        # Apply denoising if requested
        denoise_components = kwargs.get(f"{widget_prefix}_denoise_components", 0)
        if denoise_components > 0:
            processed_data = denoise_pca(
                processed_data, n_components=denoise_components
            )

        # Transform to k-space if requested
        show_image_space = kwargs.get(f"{widget_prefix}_show_image_space", True)
        if not show_image_space:  # If show_image_space is False, show k-space
            processed_data = self._apply_kspace_transform(processed_data)

        # Apply line broadening if requested (only for spectroscopic data)
        lb_value = kwargs.get(f"{widget_prefix}_linebroadening", 0.0)
        if (
            lb_value > 0
            and hasattr(self.data_obj, "sampling_time_axis_s")
            and hasattr(data_array, "dims")
            and "acquisition_idx" in data_array.dims
        ):
            dwell_time_s = (
                self.data_obj.sampling_time_axis_s[2]
                - self.data_obj.sampling_time_axis_s[1]
            )
            signal_domain = (
                "spectral"
                if not processed_data.attrs.get("signal_domain", "temporal")
                == "temporal"
                else "temporal"
            )
            axis = processed_data.dims.index("acquisition_idx")
            processed_data.values = multi_dim_linebroadening(
                processed_data.values,
                lb_hz=lb_value,
                dwell_time_s=dwell_time_s,
                signal_domain=signal_domain,
                axis=axis,
            )

        # Apply domain transformation (temporal/spectral) if needed
        is_fid = processed_data.attrs.get("signal_domain", "temporal") == "temporal"
        show_temporal = kwargs.get(f"{widget_prefix}_view_toggle", False)
        npoints = kwargs.get(
            f"{widget_prefix}_npoints",
            data_array.sizes.get(
                "acquisition_idx", len(data_array.attrs.get("sampling_time_axis_s", []))
            ),
        )

        # Average the data along the repetition dimension if requested
        average_reps = kwargs.get(f"{widget_prefix}_average_repetitions", False)
        if average_reps and "repetition_idx" in processed_data.dims:
            processed_data = processed_data.mean(dim="repetition_idx")

        processed_data = self._get_plot_signal(
            raw_signal=processed_data,
            is_fid=is_fid,
            show_temporal_view=show_temporal,
            npoints_to_plot=npoints,
            dimension=(
                processed_data.dims.index("acquisition_idx")
                if "acquisition_idx" in processed_data.dims
                else 0
            ),
        )

        # Apply specialized preprocessing
        processed_data = self._apply_specialized_preprocessing(
            processed_data, kwargs, widget_prefix
        )

        return processed_data

    def _standardize_dimension_order(self, data_array):
        """Standardize dimension ordering and spatial orientation for consistent overlay alignment.

        This function reorders dimensions to a standard pattern and ensures consistent
        spatial orientation so that datasets with different dimension orders and orientations
        (like reco_data vs data_array) are aligned consistently when overlaid.

        Parameters
        ----------
        data_array : xr.DataArray or None
            Input data with potentially non-standard dimension ordering.

        Returns
        -------
        xr.DataArray or None
            Data with standardized dimension ordering and orientation.
        """
        if data_array is None:
            return None

        # Define the preferred dimension order for anatomical data
        preferred_order = [
            "read_idx",
            "phase_idx",
            "slice_idx",  # Spatial dims first
            "channel_idx",
            "inversion_idx",
            "repetition_idx",  # Common non-spatial dims
            "echo_idx",
            "frame_idx",
            "acquisition_idx",  # Other temporal/acquisition dims
        ]

        # Get current dimensions
        current_dims = list(data_array.dims)

        # Create new dimension order: preferred dims first, then remaining dims
        new_dim_order = []
        remaining_dims = current_dims.copy()

        # Add preferred dimensions in order (if they exist)
        for dim in preferred_order:
            if dim in current_dims:
                new_dim_order.append(dim)
                remaining_dims.remove(dim)

        # Add any remaining dimensions at the end
        new_dim_order.extend(remaining_dims)

        # Only transpose if order actually changed
        if new_dim_order != current_dims:
            logger.debug(
                f"Standardizing dimension order: {current_dims} -> {new_dim_order}"
            )
            data_array = data_array.transpose(*new_dim_order)

        # Apply spatial orientation standardization
        data_array = self._standardize_spatial_orientation(data_array)

        return data_array

    def _standardize_spatial_orientation(self, data_array):
        """Standardize spatial orientation for consistent display.

        Ensures that:
        - read_idx always points from left→right (if present)
        - If no read_idx, slice_idx points up→down

        Parameters
        ----------
        data_array : xr.DataArray or None
            Data array with standardized dimension ordering.

        Returns
        -------
        xr.DataArray or None
            Data array with standardized spatial orientation.
        """
        if data_array is None:
            return None

        # Check if we need to apply orientation standardization
        has_read_idx = "read_idx" in data_array.dims
        has_slice_idx = "slice_idx" in data_array.dims

        if not (has_read_idx or has_slice_idx):
            # No spatial dimensions to standardize
            return data_array

        # Apply orientation rules
        if has_read_idx:
            # Rule 1: read_idx should go left→right
            # Check if read_idx needs flipping using coordinate values
            try:
                if hasattr(data_array, "coords") and "read_idx" in data_array.coords:
                    read_coords = data_array.coords["read_idx"]
                    if len(read_coords) > 1:
                        # Extract scalar values for comparison
                        first_val = float(read_coords.values[0])
                        last_val = float(read_coords.values[-1])
                        # If coordinates are decreasing, flip the read dimension
                        if first_val > last_val:
                            logger.debug(
                                f"Flipping read_idx dimension for left→right orientation: {first_val} > {last_val}"
                            )
                            data_array = data_array.isel(read_idx=slice(None, None, -1))
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(
                    f"Could not determine read_idx orientation from coordinates: {e}"
                )
                # Skip orientation adjustment if coordinates are not numeric or accessible

        elif has_slice_idx and not has_read_idx:
            # Rule 2: If no read_idx, slice_idx should go up→down
            # Similar coordinate-based logic for slice_idx
            try:
                if hasattr(data_array, "coords") and "slice_idx" in data_array.coords:
                    slice_coords = data_array.coords["slice_idx"]
                    if len(slice_coords) > 1:
                        # Extract scalar values for comparison
                        first_val = float(slice_coords.values[0])
                        last_val = float(slice_coords.values[-1])
                        # If coordinates are increasing, flip for up→down orientation
                        if first_val < last_val:
                            logger.debug(
                                f"Flipping slice_idx dimension for up→down orientation: {first_val} < {last_val}"
                            )
                            data_array = data_array.isel(
                                slice_idx=slice(None, None, -1)
                            )
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(
                    f"Could not determine slice_idx orientation from coordinates: {e}"
                )
                # Skip orientation adjustment if coordinates are not numeric or accessible

        return data_array

    def _get_plot_signal(
        self, raw_signal, is_fid, show_temporal_view, npoints_to_plot, dimension=0
    ):
        """Process spectroscopic signal by performing FFT or slicing.

        This is adapted from SpectroscopyPlotter._get_plot_signal
        """
        # Enhanced input validation
        if not isinstance(raw_signal, xr.DataArray):
            raise TypeError("Input 'raw_signal' must be an xarray.DataArray")

        if not isinstance(npoints_to_plot, int) or npoints_to_plot <= 0:
            raise ValueError("npoints_to_plot must be a positive integer")

        if dimension < 0 or dimension >= raw_signal.ndim:
            raise ValueError(f"dimension must be between 0 and {raw_signal.ndim - 1}")

        try:
            # Select the correct number of points along the acquisition
            # dimension
            dim_name = raw_signal.dims[dimension]
            current_size = raw_signal.sizes[dim_name]
            actual_npoints = min(npoints_to_plot, current_size)

            # Transform domain if necessary (temporal <-> spectral)
            if show_temporal_view and is_fid:
                slicer = {dim_name: slice(0, actual_npoints)}
                processed_signal = raw_signal.isel(**slicer)
            elif show_temporal_view and not is_fid:
                # Data is spectral, but we want to see temporal (FID)
                slicer = {dim_name: slice(0, current_size)}
                processed_signal = raw_signal.isel(**slicer)
                processed_signal = xr.apply_ufunc(
                    lambda x: np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1),
                    processed_signal,
                    input_core_dims=[[dim_name]],
                    output_core_dims=[[dim_name]],
                    keep_attrs=True,
                )
                slicer = {dim_name: slice(0, npoints_to_plot)}
                processed_signal = processed_signal.isel(**slicer)
                processed_signal.attrs["signal_domain"] = "temporal"
            elif not show_temporal_view and is_fid:
                # Data is temporal (FID), but we want to see spectral
                slicer = {dim_name: slice(0, npoints_to_plot)}
                processed_signal = raw_signal.isel(**slicer)
                processed_signal = xr.apply_ufunc(
                    lambda x: np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1),
                    processed_signal,
                    input_core_dims=[[dim_name]],
                    output_core_dims=[[dim_name]],
                    keep_attrs=True,
                )
                processed_signal.attrs["signal_domain"] = "spectral"
            elif not show_temporal_view and not is_fid:
                # Data is spectral, keep spectral
                slicer = {dim_name: slice(0, current_size)}
                processed_signal = raw_signal.isel(**slicer)
                # Apply spec -> FID -> spec transformation to get correct
                # npoints
                processed_signal = xr.apply_ufunc(
                    lambda x: np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1),
                    processed_signal,
                    input_core_dims=[[dim_name]],
                    output_core_dims=[[dim_name]],
                    keep_attrs=True,
                )
                slicer = {dim_name: slice(0, npoints_to_plot)}
                processed_signal = processed_signal.isel(**slicer)
                processed_signal = xr.apply_ufunc(
                    lambda x: np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1),
                    processed_signal,
                    input_core_dims=[[dim_name]],
                    output_core_dims=[[dim_name]],
                    keep_attrs=True,
                )
                processed_signal.attrs["signal_domain"] = "spectral"

            return processed_signal

        except Exception as e:
            warnings.warn(f"Error processing signal: {e}. Returning original signal.")
            return raw_signal

    def _get_plot_image(self, data_array, is_spatial, show_spatial_view):
        """Handle transformation between image space and k-space for anatomical data.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data to transform.
        is_spatial : bool
            ``True`` when the data represents the spatial domain.
        show_spatial_view : bool
            ``True`` to display image space, ``False`` for k-space.

        Returns
        -------
        xr.DataArray
            Transformed data array.
        """
        # If no transformation needed, return as-is
        if is_spatial and show_spatial_view:
            return data_array  # Already spatial, want spatial
        elif not is_spatial and not show_spatial_view:
            return data_array  # Already k-space, want k-space

        # Apply FFT transformation between image and k-space
        try:
            spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
            transform_dims = [
                dim
                for dim in spatial_dims
                if dim in data_array.dims and data_array.sizes[dim] > 1
            ]

            if not transform_dims:
                return data_array

            # Apply FFT along spatial dimensions
            axes = [data_array.get_axis_num(dim) for dim in transform_dims]

            if is_spatial and not show_spatial_view:
                # Transform from image space to k-space (forward FFT)
                transformed_data = np.fft.fftshift(
                    np.fft.fftn(
                        np.fft.ifftshift(data_array.values, axes=axes), axes=axes
                    ),
                    axes=axes,
                )
            else:
                # Transform from k-space to image space (inverse FFT)
                transformed_data = np.fft.fftshift(
                    np.fft.ifftn(
                        np.fft.ifftshift(data_array.values, axes=axes), axes=axes
                    ),
                    axes=axes,
                )

            # Create new DataArray with transformed data
            result = data_array.copy(data=transformed_data)
            # Update domain attribute
            result.attrs["domain"] = "spatial" if show_spatial_view else "kspace"
            return result

        except Exception as e:
            warnings.warn(
                f"k-space/image transformation failed: {e}. Using original data."
            )
            return data_array

    # ========================================================================
    # Cache Methods shared across plotters
    # ========================================================================

    def _get_cached_preprocessing(self, plot_data, kwargs):
        """Get preprocessed data from cache or compute and cache it.

        Uses CacheManager for unified cache management with automatic eviction.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data object containing the original data reference.
        kwargs : dict
            Current widget values.

        Returns
        -------
        UnifiedPlotData or None
            Cached preprocessing result when available, otherwise ``None``.
        """
        if not hasattr(plot_data, "raw_data") or plot_data.raw_data is None:
            return None

        # Generate cache key using CacheManager
        raw_data = plot_data.raw_data
        if hasattr(raw_data, "values"):
            # Use the underlying numpy buffer ID so cache entries change when data mutates
            data_id = id(raw_data.values)
        else:
            data_id = id(raw_data)
        cache_key = self.cache_manager.generate_cache_key(
            plot_data.widget_prefix, kwargs, data_id
        )

        # Check cache via CacheManager
        cached_result = self.cache_manager.get(cache_key, category="preprocessing")
        if cached_result is not None:
            return cached_result

        # Cache miss - compute and cache
        # Create fresh copy and reapply preprocessing
        fresh_plot_data = self._prepare_dataarray_plot_data(
            plot_data.raw_data,
            kwargs,
            plot_data.is_overlay,
            plot_data.widget_prefix,
        )
        fresh_plot_data.widget_prefix = plot_data.widget_prefix
        fresh_plot_data.is_overlay = plot_data.is_overlay

        # Cache the result via CacheManager (handles eviction automatically)
        self.cache_manager.set(cache_key, fresh_plot_data, category="preprocessing")

        return fresh_plot_data

    def _get_cached_minmax(self, spatial_data, spec_part_val=None):
        """Get cached min/max values or compute and cache them.

        Uses CacheManager for unified cache management.

        Parameters
        ----------
        spatial_data : xr.DataArray or np.ndarray
            Spatial data array used to compute the range.
        spec_part_val : str, optional
            Data transformation identifier such as ``"Magnitude"`` or ``"Phase"``.

        Returns
        -------
        tuple of float
            Minimum and maximum values after caching.
        """
        data_id = id(spatial_data)
        data_shape = spatial_data.shape if hasattr(spatial_data, "shape") else ()

        # Generate cache key and check cache
        cache_key = self.cache_manager.generate_minmax_key(
            data_id, spec_part_val, data_shape
        )
        cached_result = self.cache_manager.get(cache_key, category="minmax")
        if cached_result is not None:
            return cached_result

        # Cache miss - compute min/max
        try:
            # Apply transformation if specified before computing min/max
            # Always apply transformation if spec_part_val is provided (even for "Magnitude")
            # to ensure we compute min/max on the visualized data (e.g. abs(complex))
            if spec_part_val:
                from hypermri.utils.mathematics.calculations import part_func

                # Extract values if xarray
                input_values = (
                    spatial_data.values
                    if hasattr(spatial_data, "values")
                    else spatial_data
                )

                # Check if transformation was already applied (e.g. in preprocessing)
                if (
                    hasattr(spatial_data, "attrs")
                    and spatial_data.attrs.get("spec_part_applied") == spec_part_val
                ):
                    transformed_data = input_values
                else:
                    transformed_data = part_func(
                        input_data=input_values, signal_part=spec_part_val
                    )

                min_val, max_val = np.nanmin(transformed_data), np.nanmax(
                    transformed_data
                )
            else:
                min_val, max_val = np.nanmin(spatial_data), np.nanmax(spatial_data)
        except Exception as e:
            # Fallback to basic min/max if transformation fails
            import warnings

            warnings.warn(
                f"MinMax computation failed for {spec_part_val}: {e}, using basic min/max"
            )
            min_val, max_val = np.nanmin(spatial_data), np.nanmax(spatial_data)

        result = (min_val, max_val)

        # Cache the result via CacheManager
        self.cache_manager.set(cache_key, result, category="minmax")

        return min_val, max_val

    def _get_cached_transform(self, data, spec_part_val):
        """Get cached transformed data or compute and cache it.

        Uses CacheManager for unified cache management.

        Parameters
        ----------
        data : xarray.DataArray or np.ndarray
            Input data to transform.
        spec_part_val : str
            Transformation type such as ``"Magnitude"``, ``"Phase"``, ``"Real"``, or ``"Imaginary"``.

        Returns
        -------
        xarray.DataArray or np.ndarray
            Transformed data array, matching the input container type.
        """
        # Determine data type and extract values
        if hasattr(data, "values"):
            data_id = id(data.values)
            input_values = data.values
            data_shape = data.shape
            is_xarray = True
        else:
            data_id = id(data)
            input_values = data
            data_shape = data.shape if hasattr(data, "shape") else ()
            is_xarray = False

        # Generate cache key and check cache
        cache_key = self.cache_manager.generate_transform_key(
            data_id, spec_part_val, data_shape
        )
        cached_result = self.cache_manager.get(cache_key, category="transform")
        if cached_result is not None:
            # Verify cached result shape matches current data shape
            if hasattr(cached_result, "shape") and cached_result.shape != data_shape:
                self.cache_manager.invalidate(cache_key, category="transform")
            else:
                # Shape matches - safe to return cached result
                if is_xarray and hasattr(data, "dims"):
                    return data.copy(data=cached_result)
                else:
                    return cached_result

        # Cache miss or shape mismatch - compute transformation

        # Check if transformation was already applied (e.g. in preprocessing)
        if (
            is_xarray
            and hasattr(data, "attrs")
            and data.attrs.get("spec_part_applied") == spec_part_val
        ):
            return data

        from hypermri.utils.mathematics.calculations import part_func

        try:
            transformed_values = part_func(
                input_data=input_values, signal_part=spec_part_val
            )
        except Exception as e:
            # If transformation fails, return original data for graceful degradation
            import warnings

            warnings.warn(
                f"Transform failed for {spec_part_val}: {e}, returning original data"
            )
            if is_xarray:
                return data
            else:
                return input_values

        # Cache the transformed values via CacheManager
        self.cache_manager.set(cache_key, transformed_values, category="transform")

        # Return appropriate format
        if is_xarray and hasattr(data, "dims"):
            # Preserve xarray structure with transformed data
            try:
                return data.copy(data=transformed_values)
            except ValueError as e:
                # Shape mismatch during copy - clear this cache entry and return uncached result
                self.cache_manager.invalidate(cache_key, category="transform")
                import warnings

                warnings.warn(f"Cache shape mismatch for {spec_part_val}: {e}")
                return data.copy(data=transformed_values)
        else:
            return transformed_values

    def clear_caches(self):
        """Clear all caches to free memory or force recomputation.

        Delegates to CacheManager for unified cache management.
        """
        self.cache_manager.clear()

    def get_cache_stats(self):
        """Return cache performance statistics.

        Returns
        -------
        dict
            Cache statistics including hits, misses, evictions, and sizes.
        """
        return self.cache_manager.get_stats()

    def invalidate_data_caches(self, data_obj):
        """Invalidate cache entries related to a specific data object.

        Delegates to CacheManager for unified cache invalidation.

        Parameters
        ----------
        data_obj : object
            Data object whose cached results should be invalidated.
        """
        if data_obj is None:
            return

        data_id = (
            id(data_obj) if not hasattr(data_obj, "values") else id(data_obj.values)
        )
        self.cache_manager.invalidate_by_data_id(data_id)

    def _check_data_changes_and_invalidate(self):
        """Check if data object has changed and invalidate relevant caches.

        Note: This method is designed to be overridden in child classes
        to provide data-specific invalidation logic.
        """
        # Default implementation - can be overridden by child classes
        if hasattr(self.data_obj, "data_array"):
            self.invalidate_data_caches(self.data_obj.data_array)
        if hasattr(self.data_obj, "reco_array"):
            self.invalidate_data_caches(self.data_obj.reco_array)

    def print_cache_performance(self):
        """Print cache performance statistics for debugging and optimization monitoring."""
        # Use CacheManager for new implementation (if available)
        if hasattr(self, "cache_manager"):
            self.cache_manager.print_stats()
        else:
            # Legacy fallback for old instances
            stats = self.get_cache_stats()
            print(f"Cache Stats: {stats}")

    # ========================================================================
    # End of cache helper methods
    # ========================================================================

    def add_interactive_overlay(self, overlay_name, overlay_data, callback=None):
        """Add an interactive overlay (like masks from create_mask).

        Parameters
        ----------
        overlay_name : str
            Name of the overlay.
        overlay_data : dict
            Overlay data and metadata.
        callback : callable, optional
            Callback invoked when overlay interactions occur.
        """
        self.interactive_overlays[overlay_name] = overlay_data
        if callback:
            self.overlay_callbacks[overlay_name] = callback

        # Trigger plot update if we have an active update function
        if self.update_function:
            try:
                current_kwargs = {
                    k: v.value for k, v in self.widgets.items() if hasattr(v, "value")
                }
                self.update_function(**current_kwargs)
            except Exception as e:
                warnings.warn(f"Failed to update plot after adding overlay: {e}")

    def remove_interactive_overlay(self, overlay_name):
        """Remove an interactive overlay.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to remove.
        """
        self.interactive_overlays.pop(overlay_name, None)
        self.overlay_callbacks.pop(overlay_name, None)

        # Trigger plot update if we have an active update function
        if self.update_function:
            try:
                current_kwargs = {
                    k: v.value for k, v in self.widgets.items() if hasattr(v, "value")
                }
                self.update_function(**current_kwargs)
            except Exception as e:
                warnings.warn(f"Failed to update plot after removing overlay: {e}")

    def _create_unified_widgets(
        self, plot_data_list, processed_plot_params, slider_defaults
    ):
        """Create widgets for the unified plotting interface.

        This method consolidates layout logic from child plotters to create a
        consistent widget interface for all plot types.

        Parameters
        ----------
        plot_data_list : list of UnifiedPlotData
            List of plot data objects for which to create controls.
        processed_plot_params : dict
            Dictionary of processed plot parameters.
        slider_defaults : dict
            Default values for sliders and other widgets.

        Returns
        -------
        controls_box : ipywidgets.VBox
            The main container widget holding all control widgets.
        widgets_dict : dict
            A dictionary mapping widget names to their ipywidgets objects.

        """
        self.widgets = {}

        # Determine global dimensions from all data - only non-spatial
        # dimensions get sliders
        all_dims = set()
        max_sizes = {}
        has_spatial_data = any(pd.has_spatial_data() for pd in plot_data_list)

        for plot_data in plot_data_list:
            if plot_data.spatial_data is not None:
                # Get all spatial dimensions that could be sliced through
                all_spatial_dims = self._get_all_spatial_dims(plot_data.spatial_data)

                # For 3D data with standard dimensions, create sliders for all
                # spatial dims
                if len(all_spatial_dims) >= 3 and all(
                    dim in all_spatial_dims
                    for dim in ["read_idx", "phase_idx", "slice_idx"]
                ):
                    # 3D data: create sliders for all spatial dimensions
                    displayed_dims = set()  # None are excluded from sliders in 3D case
                else:
                    # 2D or non-standard data: use the original logic
                    dims_and_sizes = [
                        (dim, plot_data.spatial_data.sizes[dim])
                        for dim in plot_data.spatial_data.dims
                        if dim not in ["metabolite", "acquisition_idx"]
                    ]
                    dims_and_sizes.sort(key=lambda x: x[1], reverse=True)

                    displayed_dims = set()
                    if len(dims_and_sizes) >= 2:
                        displayed_dims = {dims_and_sizes[0][0], dims_and_sizes[1][0]}
                    elif len(dims_and_sizes) == 1:
                        displayed_dims = {dims_and_sizes[0][0]}

                # Add dimensions that should get sliders
                for dim in plot_data.spatial_data.dims:
                    if (
                        dim not in ["metabolite", "acquisition_idx"]
                        and dim not in displayed_dims
                        and plot_data.spatial_data.sizes[dim] > 1
                    ):
                        all_dims.add(dim)
                        current_max = max_sizes.get(dim, 0)
                        max_sizes[dim] = max(
                            current_max, plot_data.spatial_data.sizes[dim]
                        )

        # Create global dimension sliders
        slider_list = []
        if has_spatial_data:
            # Initialize with default perspective
            if self._active_perspective_dim is None:
                self._set_active_perspective("slice_idx")  # Default to axial view

            self._get_perspective_name(self._active_perspective_dim)
            # self.widgets["plane_label"] = Label(
            #     f"View: {perspective_name} (slicing through {self._active_perspective_dim})"
            # )

        for dim_name in sorted(list(all_dims)):
            if dim_name in max_sizes and max_sizes[dim_name] > 1:
                max_val = max_sizes[dim_name] - 1
                default_val = slider_defaults.get(dim_name, max_val // 2)
                slider = IntSlider(
                    min=0,
                    max=max_val,
                    step=1,
                    value=max(0, min(int(default_val), max_val)),
                    description=f"{dim_name}:",
                    continuous_update=True,
                    layout=Layout(max_width="300px", width="auto"),
                )
                self.widgets[dim_name] = slider
                slider_list.append(slider)

                # Add observer to track which slider was last moved for perspective detection
                # Note: This observer only updates the perspective, it doesn't
                # interfere with plotting
                if dim_name in ["read_idx", "phase_idx", "slice_idx"]:
                    slider.observe(
                        self._create_perspective_observer(dim_name), names="value"
                    )
            else:
                self.widgets[dim_name] = fixed(0)

        # Set up basic observers for child classes (if they need specialized behavior)
        if hasattr(self, "_setup_basic_observers"):
            self._setup_basic_observers()

        # Ensure all spatial sliders exist (important for spectroscopy data)
        self._ensure_spatial_sliders_exist_unified(plot_data_list, slider_defaults)

        # IMPORTANT: Update the widgets dictionary with any newly created sliders
        # This ensures they get connected to the interactive_output for plot updates
        # This is the critical fix from spectroscopyPlotter that was needed for
        # the repetition slider to work properly

        # Create per-dataset controls with enhanced layout
        data_array_controls = []
        for i, plot_data in enumerate(plot_data_list):
            widget_prefix = plot_data.widget_prefix

            dataset_widgets = self._create_dataset_specific_widgets(
                plot_data, processed_plot_params, widget_prefix, slider_defaults
            )

            # Add widgets to main dictionary with prefix
            for w_name, w_obj in dataset_widgets.items():
                self.widgets[f"{widget_prefix}_{w_name}"] = w_obj

            # Create control box for this dataset with improved labeling
            if plot_data.is_spectroscopic():
                data_type_name = "Spectroscopy"
            elif plot_data.is_anatomical():
                data_type_name = "Anatomy"
            else:
                # Handle case where data_type might be None
                if plot_data.data_type is not None:
                    data_type_name = plot_data.data_type.replace("_", " ").title()
                else:
                    data_type_name = "Data"

            if i == 0:
                title = f"<b>Controls for: {data_type_name} Data {i}</b>"
            else:
                title = f"<b>Controls for: Overlay Data {i}</b>"

            data_type_title = HTML(value=title)
            control_list = list(dataset_widgets.values())
            data_array_box = VBox(
                [data_type_title, *control_list],
                layout=Layout(
                    border="solid 1px grey",
                    padding="5px",
                    margin="5px",
                    min_width="350px",
                ),
            )
            data_array_controls.append(data_array_box)

        # Create overlay controls if multiple datasets
        overlay_controls = []
        if len(plot_data_list) > 1:
            self.widgets["overlay_alpha"] = FloatSlider(
                value=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Overlay Alpha:",
                continuous_update=True,
            )
            self.widgets["overlay_alpha"].layout.max_width = "300px"
            self.widgets["show_base"] = Checkbox(value=True, description="Show Base")
            self.widgets["subtract_overlay"] = Checkbox(
                value=False,
                description="Subtract (Overlay - Base)",
                tooltip="Show difference between overlay and base data",
            )
            self.widgets["percent_difference"] = Checkbox(
                value=False,
                description="% Difference",
                tooltip="Show percent difference: 100 * (Overlay - Base) / Base",
            )
            overlay_controls = [
                VBox(
                    [
                        self.widgets["overlay_alpha"],
                        self.widgets["show_base"],
                        self.widgets["subtract_overlay"],
                        self.widgets["percent_difference"],
                    ]
                )
            ]

        # Consolidated layout arrangement from both spectroscopy and anatomy plotters
        layout_rows = []

        # Add plane label if spatial data exists
        if has_spatial_data and "plane_label" in self.widgets:
            layout_rows.append(HBox([self.widgets["plane_label"]]))

        # Create combined row for global controls
        main_controls_row = []

        # Global dimension controls (includes all spatial dimensions for comprehensive control)
        global_sliders = []
        all_potential_dims = [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "acquisition_idx",  # Important for spectroscopy data
            "repetition_idx",  # Critical for repetition slider functionality
            "echo_idx",  # For multi-echo data
            "channel_idx",  # For multi-channel data
            "inversion_idx",  # For inversion recovery sequences
            "rfpower_idx",  # For B1 mapping RF power dimension
        ]

        for dim_name in all_potential_dims:
            if dim_name in self.widgets and hasattr(self.widgets[dim_name], "value"):
                global_sliders.append(self.widgets[dim_name])

        # Add spectral inset controls to global controls if they exist
        spectral_inset_controls = []
        if "show_spec_inset" in self.widgets:
            spectral_inset_controls.append(self.widgets["show_spec_inset"])
        if "spec_inset_min_Hz" in self.widgets and "spec_inset_max_Hz" in self.widgets:
            spectral_inset_controls.append(
                HBox(
                    [
                        self.widgets["spec_inset_min_Hz"],
                        self.widgets["spec_inset_max_Hz"],
                    ]
                )
            )

        if global_sliders:

            sliders_title = HTML(value="<b>Global Dimension Controls</b>")

            # Add coordinate system toggle for spatial plots
            slider_box_contents = [sliders_title]
            if has_spatial_data:
                coord_system_toggle = ToggleButtons(
                    options=["Auto", "XYZ", "RPS"],
                    value="Auto",
                    description="Coord System:",
                    disable=True,
                    button_style="",
                    tooltips=[
                        "Automatic coordinate detection",
                        "Scanner coordinates (X, Y, Z)",
                        "Acquisition coordinates (Read, Phase, Slice)",
                    ],
                )
                self.widgets["coord_system"] = coord_system_toggle
                slider_box_contents.append(coord_system_toggle)
            else:
                # Add fixed widget for non-spatial plots
                self.widgets["coord_system"] = fixed("Auto")

            sliders_box = VBox(
                slider_box_contents
                + global_sliders
                + overlay_controls
                + spectral_inset_controls,
                layout=Layout(
                    border="solid 1px grey",
                    padding="5px",
                    margin="5px",
                    min_width="300px",
                ),
            )
            # Add overlay controls at the bottom if they exist
            # if overlay_controls:
            # sliders_box.append(HBox(overlay_controls))

            main_controls_row.append(sliders_box)

        # Group dataset controls by prefix for better organization
        dataset_controls = {}
        main_controls = []

        for widget_name, widget in self.widgets.items():
            # Skip individual widgets that are already grouped in spatial_options_row
            is_sub_widget = False
            for suffix in ["interpolate_image", "sync_y_range"]:
                if widget_name.endswith(f"_{suffix}"):
                    is_sub_widget = True
                    break
            if is_sub_widget:
                continue

            if (
                "_" in widget_name
                and not widget_name.startswith("plane_")
                and widget_name not in all_potential_dims
                and widget_name
                not in [
                    "overlay_alpha",
                    "show_base",
                    "subtract_overlay",
                    "percent_difference",
                    "show_spec_inset",
                    "spec_inset_min_Hz",
                    "spec_inset_max_Hz",
                    "coord_system",
                ]
            ):
                prefix = widget_name.split("_")[0]
                if prefix not in dataset_controls:
                    dataset_controls[prefix] = []
                dataset_controls[prefix].append(widget)
            elif (
                "_" not in widget_name
                and widget_name not in all_potential_dims
                and widget_name
                not in ["plane_label", "overlay_alpha", "show_base", "coord_system"]
            ):
                main_controls.append(widget)

        # Create controls box for each dataset
        for prefix, controls in dataset_controls.items():
            if controls:
                if prefix in ["ds0", ""]:
                    # Determine data type for the title
                    if plot_data_list and plot_data_list[0].is_spectroscopic():
                        title = "<b>Controls for: Spectroscopy Data 0</b>"
                    elif plot_data_list and plot_data_list[0].is_anatomical():
                        title = "<b>Controls for: Anatomy Data 0</b>"
                    else:
                        title = "<b>Controls for: Data 0</b>"
                else:
                    dataset_num = (
                        prefix.replace("ds", "") if prefix.startswith("ds") else prefix
                    )
                    title = f"<b>Controls for: Overlay Data {dataset_num}</b>"

                controls_title = HTML(value=title)

                # Add main controls (spec_part, etc.) to first dataset box
                if prefix in ["ds0", ""] and main_controls:
                    controls = main_controls + controls

                controls_box_widget = VBox(
                    [controls_title] + controls,
                    layout=Layout(
                        border="solid 1px grey",
                        padding="5px",
                        margin="5px",
                        min_width="350px",
                    ),
                )
                main_controls_row.append(controls_box_widget)

        # If no dataset controls but have main controls, create a default box
        if not dataset_controls and main_controls:
            controls_title = HTML(value="<b>Controls for: Data 0</b>")
            controls_box_widget = VBox(
                [controls_title] + main_controls,
                layout=Layout(
                    border="solid 1px grey",
                    padding="5px",
                    margin="5px",
                    min_width="300px",
                ),
            )
            main_controls_row.append(controls_box_widget)

        if main_controls_row:
            layout_rows.append(HBox(main_controls_row))

        # Create final layout
        final_controls_box = VBox(layout_rows)
        return final_controls_box, self.widgets

    def _create_perspective_observer(self, dim_name):
        """Create an observer that tracks which slider was last moved for perspective detection."""

        def observer(change):
            if change["new"] != change["old"]:
                # Update the active perspective dimension
                self._set_active_perspective(dim_name)

                # Update plane label if it exists
                if "plane_label" in self.widgets:
                    perspective_name = self._get_perspective_name(dim_name)
                    self.widgets["plane_label"].value = (
                        f"View: {perspective_name} (slicing through {dim_name})"
                    )

        return observer

    def _ensure_spatial_sliders_exist_unified(self, plot_data_list, slider_defaults):
        """Ensure sliders exist for all spatial and spectral dimensions.

        This is a consolidated version from both spectroscopyPlotter and anatomyPlotter
        that ensures all necessary sliders are created for proper interactive functionality.
        """
        # Include all dimensions that should have sliders
        spatial_dims = [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "acquisition_idx",  # Important for spectroscopy data
            "repetition_idx",  # Critical fix - this was missing and caused the repetition slider issue
            "channel_idx",  # For multi-channel data
            "inversion_idx",  # For inversion recovery sequences
        ]

        for i, plot_data in enumerate(plot_data_list):
            # For spectroscopy data, check both spatial_data and spectral_data for spatial dimensions
            # For anatomy data, check spatial_data
            data_to_check = (
                plot_data.spatial_data
                if plot_data.spatial_data is not None
                else plot_data.spectral_data
            )

            if data_to_check is not None:
                for dim_name in spatial_dims:
                    if dim_name in data_to_check.dims:
                        size = data_to_check.sizes[dim_name]
                        exists_in_widgets = dim_name in self.widgets

                        if size > 1 and not exists_in_widgets:
                            # Create missing spatial slider
                            max_val = size - 1
                            default_val = slider_defaults.get(dim_name, max_val // 2)

                            slider = IntSlider(
                                min=0,
                                max=max_val,
                                step=1,
                                value=max(0, min(int(default_val), max_val)),
                                description=f"{dim_name}:",
                                continuous_update=True,
                                layout=Layout(max_width="300px", width="auto"),
                            )

                            self.widgets[dim_name] = slider

                            # Add observer for spatial dimensions if in child class
                            if hasattr(
                                self, "_create_slider_click_sync_observer"
                            ) and dim_name in ["read_idx", "phase_idx", "slice_idx"]:
                                slider.observe(
                                    self._create_slider_click_sync_observer(dim_name),
                                    names="value",
                                )

    def _sync_sliders_to_click_position(self, plot_dim_y, y_idx, plot_dim_x, x_idx):
        """Synchronize slider widgets to clicked position."""
        logger.debug(f"Syncing sliders - {plot_dim_y}={y_idx}, {plot_dim_x}={x_idx}")

        # Update sliders for the clicked dimensions
        for dim_name, idx_value in [(plot_dim_y, y_idx), (plot_dim_x, x_idx)]:
            if dim_name in self.widgets:
                widget = self.widgets[dim_name]
                if hasattr(widget, "value") and hasattr(widget, "max"):
                    # Ensure the value is within bounds
                    old_value = widget.value
                    new_value = max(0, min(idx_value, widget.max))

                    if old_value != new_value:
                        logger.debug(
                            f"Updating {dim_name} slider: {old_value} -> {new_value}"
                        )
                        widget.value = new_value
                    else:
                        logger.debug(
                            f"{dim_name} slider already at correct value: {new_value}"
                        )
                else:
                    logger.debug(f"Widget {dim_name} doesn't have value/max attributes")
            else:
                logger.debug(f"Widget {dim_name} not found in self.widgets")

    def _get_perspective_name(self, active_dim):
        """Get the perspective name based on which dimension is being actively sliced."""
        if active_dim == "slice_idx":
            return "Slice"
        elif active_dim == "read_idx":
            return "Read"
        elif active_dim == "phase_idx":
            return "Phase"
        else:
            return f"Custom ({active_dim})"

    def _create_dataset_specific_widgets(
        self, plot_data, processed_plot_params, widget_prefix, slider_defaults
    ):
        """Create widgets specific to a single dataset.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the dataset.
        processed_plot_params : dict
            Plotting parameters used to determine default widget values.
        widget_prefix : str
            Prefix applied to widget names for namespacing.
        slider_defaults : dict
            Default widget values supplied by the caller.

        Returns
        -------
        dict
            Mapping of widget names to widget instances.
        """
        widgets_dict = {}

        # Data variable selector for multi-variable datasets
        if plot_data.data_type == "multi_variable_dataset" and hasattr(
            plot_data, "dataset_variable_names"
        ):
            var_options = plot_data.dataset_variable_names
            current_var = getattr(plot_data, "current_data_variable", var_options[0])
            data_var_widget = Dropdown(
                options=var_options, value=current_var, description="Variable:"
            )

            # Create observer to handle variable changes
            def on_data_variable_change(change):
                """Handle data variable selection changes."""
                new_var = change["new"]
                if (
                    hasattr(plot_data, "dataset_variables")
                    and new_var in plot_data.dataset_variables
                ):
                    plot_data.spatial_data = plot_data.dataset_variables[new_var]
                    plot_data.current_data_variable = new_var
                    # Trigger plot update if update function is available
                    if hasattr(self, "update_function") and self.update_function:
                        # Call the update function to refresh the plot
                        try:
                            self.update_function()
                        except Exception as e:
                            import warnings

                            warnings.warn(
                                f"Failed to update plot after variable change: {e}"
                            )

            data_var_widget.observe(on_data_variable_change, names="value")
            widgets_dict["data_variable"] = data_var_widget

        # Data variable selector for datasets with multiple variables (legacy support)
        elif plot_data.source_type == "dataset":
            dataset = plot_data.fitted_spectra.get("_dataset")
            if dataset and len(dataset.data_vars) > 1:
                var_options = list(dataset.data_vars.keys())
                widgets_dict["data_var"] = Dropdown(
                    options=var_options, value=var_options[0], description="Variable:"
                )

        # Metabolite selector for fitted data
        if plot_data.has_fitted_data() and plot_data.metabolite_names:
            widgets_dict["metabolite"] = Dropdown(
                options=plot_data.metabolite_names,
                value=plot_data.metabolite_names[0],
                description="Metabolite:",
            )

        # Colormap and display controls for spatial data
        if plot_data.has_spatial_data():
            valid_cmaps = [c for c in common_cmaps if c in plt.colormaps()]
            if plot_data.is_overlay:
                widgets_dict["cmap"] = Dropdown(
                    options=valid_cmaps,
                    value=processed_plot_params.get("overlay_colormap", "magma"),
                    description="Overlay Colormap:",
                )
            else:
                widgets_dict["cmap"] = Dropdown(
                    options=valid_cmaps,
                    value=processed_plot_params.get("colormap", "bone"),
                    description="Colormap:",
                )
            widgets_dict["range_mode"] = Dropdown(
                options=["Percentage", "Min/Max"],
                value=processed_plot_params.get("range_mode", "Percentage"),
                description="Range Mode:",
                layout=Layout(width="auto"),
            )
            widgets_dict["vmin"] = FloatSlider(
                value=processed_plot_params.get("vmin_perc", 0.0),
                min=0,
                max=100,
                step=0.1,
                description="Vmin: [%]",
                continuous_update=False,
            )
            widgets_dict["vmax"] = FloatSlider(
                value=processed_plot_params.get("vmax_perc", 100.0),
                min=0,
                max=100,
                step=0.1,
                description="Vmax: [%]",
                continuous_update=False,
            )
            widgets_dict["vmin"].style.handle_color = "lightblue"
            widgets_dict["vmax"].style.handle_color = "lightgreen"

        # For 3D spatial data, we'll automatically detect perspective based on which dimension is being sliced
        # No widget needed - the perspective will be determined by widget
        # interactions

        # Spectroscopic controls (for spectroscopic data)
        if plot_data.is_spectroscopic():
            spec_unit_button = ToggleButton(
                value=True,
                description="ppm/Hz",
                tooltip="Toggle between ppm (default) and Hz frequency units",
                layout=Layout(width="70px", height="25px"),
            )
            view_toggle_button = ToggleButton(
                value=False,
                description="Show FID",
                tooltip="Toggle between spectrum (default) and FID view",
                layout=Layout(width="70px", height="25px"),
            )

            # Add individual buttons to the dictionary to be registered as widgets
            widgets_dict["spec_unit"] = spec_unit_button
            widgets_dict["view_toggle"] = view_toggle_button

            # Add an HBox containing the buttons for layout purposes
            # This will be displayed in the UI, while the individual buttons are used for logic

            # Additional controls for measured data
            if plot_data.data_type == "spectroscopy_measured":
                # Create npoints widget for both spatial and purely spectroscopic data
                # Fix: Use proper None check instead of boolean evaluation
                data_for_npoints = None
                if plot_data.spatial_data is not None:
                    data_for_npoints = plot_data.spatial_data
                elif plot_data.spectral_data is not None:
                    data_for_npoints = plot_data.spectral_data

                if data_for_npoints is not None:
                    max_acquisition_points = data_for_npoints.sizes.get(
                        "acquisition_idx", 1
                    )
                    # Allow npoints to extend beyond original for zeropadding
                    max_zeropad = 2048
                    extended_max = max_acquisition_points + max_zeropad

                    npoints_slider = IntSlider(
                        value=max_acquisition_points,
                        max=extended_max,
                        min=1,
                        description="NPoints:",
                        tooltip=f"Number of points (original: {max_acquisition_points}, values > {max_acquisition_points} add zeropadding)",
                    )
                    # Store original max for zeropadding calculation in preprocessing
                    npoints_slider._original_max = max_acquisition_points

                    def npoints_observer(change):
                        """Observer to sync acquisition_idx max with npoints."""
                        new_npoints = change["new"]
                        if "acquisition_idx" in self.widgets:
                            acq_idx_slider = self.widgets["acquisition_idx"]
                            # Update acquisition_idx max to match npoints
                            acq_idx_slider.max = new_npoints - 1
                            # If current index is outside the new range of points
                            if acq_idx_slider.value >= new_npoints:
                                # Set index to the last valid point (npoints - 1)
                                acq_idx_slider.value = new_npoints - 1

                    npoints_slider.observe(npoints_observer, names="value")
                    widgets_dict["npoints"] = npoints_slider
                widgets_dict["denoise_components"] = IntSlider(
                    value=slider_defaults.get(f"{widget_prefix}_denoise_components", 0),
                    min=0,
                    max=20,
                    description="Denoise Comp:",
                )
                # Only show line broadening for spectroscopic data
                if plot_data.is_spectroscopic():
                    widgets_dict["linebroadening"] = FloatText(
                        value=slider_defaults.get(
                            f"{widget_prefix}_linebroadening", 0.0
                        ),
                        description="LB (Hz):",
                    )

        # Anatomical controls (for anatomical data)
        if plot_data.is_anatomical():
            debug_log_unified_plotter(
                f"🎛️ Creating anatomical controls - data_type: {plot_data.data_type}"
            )
            # widgets_dict["show_image_space"] = ToggleButton(
            #     value=True,
            #     description="Image/k-space",
            #     tooltip="Switch between Image and k-space view",
            #     icon="image",
            # )

            # Denoising for anatomical data
            widgets_dict["denoise_components"] = IntSlider(
                value=slider_defaults.get(f"{widget_prefix}_denoise_components", 0),
                min=0,
                max=20,
                description="Denoise Comp:",
            )
            # Secondary plot dimension selector for anatomical data
            if plot_data.spatial_data is not None:
                debug_log_unified_plotter(
                    f"🔍 spatial_data found, dims: {list(plot_data.spatial_data.dims)}"
                )
                debug_log_unified_plotter(
                    f"🔍 spatial_data sizes: {dict(plot_data.spatial_data.sizes)}"
                )
                all_dims = list(plot_data.spatial_data.dims)

                # Get all spatial dimensions (including ones used for slicing)
                all_spatial_dims = self._get_all_spatial_dims(plot_data.spatial_data)
                debug_log_unified_plotter(f"🔍 all_spatial_dims: {all_spatial_dims}")

                # Get non-spatial dimensions that could be used for secondary plot
                # Exclude all spatial dimensions and acquisition_idx
                secondary_plot_options = [
                    d
                    for d in all_dims
                    if d not in all_spatial_dims
                    and d not in ["acquisition_idx"]
                    and plot_data.spatial_data.sizes[d] > 1
                ]
                debug_log_unified_plotter(
                    f"🔍 secondary_plot_options: {secondary_plot_options}"
                )

                if secondary_plot_options:
                    debug_log_unified_plotter(
                        f"✅ Creating secondary_plot_dim widget with options: {secondary_plot_options}"
                    )
                    widgets_dict["secondary_plot_dim"] = Dropdown(
                        options=secondary_plot_options,
                        value=(
                            secondary_plot_options[0]
                            if secondary_plot_options
                            else None
                        ),
                        description="2nd Plot Dim:",
                        disabled=not secondary_plot_options,
                    )
                else:
                    debug_log_unified_plotter(
                        f"❌ No secondary_plot_options found - secondary plot widget NOT created"
                    )
            else:
                debug_log_unified_plotter(
                    f"❌ plot_data.spatial_data is None - cannot create secondary plot widget"
                )

        # Repetition averaging
        has_repetitions = False
        if (
            plot_data.spatial_data is not None
            and "repetition_idx" in plot_data.spatial_data.dims
            and plot_data.spatial_data.sizes["repetition_idx"] > 1
        ):
            has_repetitions = True
        elif (
            plot_data.spectral_data is not None
            and "repetition_idx" in plot_data.spectral_data.dims
            and plot_data.spectral_data.sizes["repetition_idx"] > 1
        ):
            has_repetitions = True

        if has_repetitions:
            widgets_dict["average_repetitions"] = Checkbox(
                value=False,
                description="Average Reps",
                disabled=not has_repetitions,
            )

        # Add reconstruction interpolation controls when fitted data is available
        fitted_dataset = None
        if hasattr(plot_data, "fitted_spectra") and plot_data.fitted_spectra:
            fitted_dataset = plot_data.fitted_spectra.get("_dataset")

        if isinstance(fitted_dataset, xr.Dataset):
            axis_info = self._get_reconstruction_axis_info(fitted_dataset)
            if axis_info:
                default_factor = slider_defaults.get(
                    f"{widget_prefix}_reco_interp_factor",
                    slider_defaults.get("reco_interp_factor", 1),
                )
                max_factor = slider_defaults.get(
                    f"{widget_prefix}_reco_interp_factor_max",
                    slider_defaults.get("reco_interp_factor_max", 8),
                )
                try:
                    default_factor = int(default_factor)
                except (TypeError, ValueError):
                    default_factor = 1
                try:
                    max_factor = int(max_factor)
                except (TypeError, ValueError):
                    max_factor = 8
                max_factor = max(1, max_factor)
                default_factor = min(max_factor, max(1, default_factor))

                widgets_dict["reco_interp_factor"] = IntSlider(
                    value=default_factor,
                    min=1,
                    max=max_factor,
                    step=1,
                    description=axis_info["slider_description"],
                    tooltip=axis_info["tooltip"],
                    layout=Layout(width="220px"),
                    continuous_update=True,
                )

            # Add Space selection for Tofts
            model_name = fitted_dataset.attrs.get("model_name")
            if model_name == "tofts_model":
                default_space = getattr(plot_data, "reco_space", "concentration")
                # Capitalize first letter for display
                default_space = (
                    default_space.capitalize() if default_space else "Concentration"
                )

                widgets_dict["reco_space"] = Dropdown(
                    options=["Concentration", "Signal"],
                    value=default_space,
                    description="Reco Space:",
                    tooltip="Choose whether to reconstruct in concentration (mM) or signal space (a.u.)",
                    layout=Layout(width="220px"),
                )

        # Add specialized widgets from child classes
        specialized_widgets = self._get_specialized_widgets(
            plot_data, processed_plot_params, widget_prefix, slider_defaults
        )
        widgets_dict.update(specialized_widgets)

        for w in widgets_dict.values():
            try:
                w.style.font_size = processed_plot_params.get("widget_fontsize", "10px")
            except:
                pass

        return widgets_dict

    def _render_unified_plot(self, plot_data_list, kwargs):
        """Unified rendering function for all plot types.

        This function orchestrates the entire plotting process for each update.
        It clears the previous plot, re-processes data based on current widget
        settings, renders the spatial and spectral components for each dataset
        (including overlays), and finalizes the plot with colorbars and legends.

        Parameters
        ----------
        plot_data_list : list of UnifiedPlotData
            A list of standardized plot data objects to be rendered.
        kwargs : dict
            A dictionary containing the current values from all interactive widgets,
            used to configure the plot appearance and data processing.

        """
        # Check if figure exists, create if not
        if not hasattr(self.update_function, "fig") or not plt.fignum_exists(
            self.update_function.fig.number
        ):
            self._setup_unified_figure(plot_data_list)

        # Reset base spatial dimensions for consistent overlay alignment
        self._base_spatial_dims = None

        # Clear existing plots
        if hasattr(self.update_function, "ax_img") and self.update_function.ax_img:
            self.update_function.ax_img.cla()
            # Clear stored crosshair lines since they are removed by cla()
            if hasattr(self.update_function, "crosshair_lines"):
                self.update_function.crosshair_lines = []
        if hasattr(self.update_function, "ax_spec") and self.update_function.ax_spec:
            if (
                hasattr(self.update_function, "_spec_inset_ax")
                and getattr(self.update_function, "_spec_inset_ax", None) is not None
            ):
                try:
                    self.update_function._spec_inset_ax.remove()
                except ValueError:
                    pass
                finally:
                    self.update_function._spec_inset_ax = None
            if hasattr(self.update_function, "_spec_inset_artists"):
                for artist in getattr(self.update_function, "_spec_inset_artists", []):
                    try:
                        artist.remove()
                    except ValueError:
                        continue
                self.update_function._spec_inset_artists = []
            self.update_function.ax_spec.cla()
            # Clear stored spectral click lines
            if hasattr(self.update_function, "_spec_click_lines"):
                self.update_function._spec_click_lines = []
        if hasattr(self.update_function, "cax") and self.update_function.cax:
            self.update_function.cax.cla()

        # Reset overlay image storage for dual colorbars
        if hasattr(self.update_function, "overlay_ims"):
            self.update_function.overlay_ims = []

        # REPROCESS plot data with current widget values to ensure updates are
        # applied
        processed_plot_data_list = []
        for i, plot_data in enumerate(plot_data_list):
            # Create a fresh copy and reapply preprocessing with current kwargs
            if (
                plot_data.data_type == "spectroscopy_measured"
                and plot_data.raw_spectral_data is not None
            ):
                fresh_plot_data = self._prepare_dataarray_plot_data(
                    plot_data.raw_spectral_data,
                    kwargs,
                    plot_data.is_overlay,
                    plot_data.widget_prefix,
                )
                fresh_plot_data.widget_prefix = plot_data.widget_prefix
                fresh_plot_data.is_overlay = plot_data.is_overlay
                processed_plot_data_list.append(fresh_plot_data)
            elif (
                plot_data.data_type == "spectroscopy_fitted"
                and plot_data.raw_spectral_data is not None
            ):
                # Apply same preprocessing to fitted data
                fresh_plot_data = self._apply_preprocessing_to_fitted_data(
                    plot_data, kwargs
                )
                processed_plot_data_list.append(fresh_plot_data)
            else:
                # For other types (e.g. multi_variable_dataset), allow reprocessing
                # to handle variable selection changes
                fresh_plot_data = self._reprocess_plot_data(plot_data, kwargs)
                processed_plot_data_list.append(fresh_plot_data)

        # Process each plot data object
        for i, plot_data in enumerate(processed_plot_data_list):
            is_base = i == 0

            # Skip base data if "Show Base" checkbox is unchecked (for multiple datasets)
            if (
                is_base
                and len(processed_plot_data_list) > 1
                and not kwargs.get("show_base", True)
            ):
                continue

            # Render this plot data
            self._render_plot_data(plot_data, is_base, kwargs)

        # Handle interactive overlays (like masks from create_mask)
        self._render_interactive_overlays(kwargs)

        # Update crosshair if click position exists
        if self._has_click_position():
            self._update_spatial_crosshair(self._get_click_position())

        # Set up colorbars and legends
        self._finalize_unified_plot(processed_plot_data_list, kwargs)

        # Update plane label if spatial data exists
        if hasattr(self.update_function, "_current_plot_dims") and self.widgets.get(
            "plane_label"
        ):
            plot_dims = self.update_function._current_plot_dims
            if plot_dims and self._has_click_position():
                click_pos = self._get_click_position()
                if plot_dims[0] in click_pos and plot_dims[1] in click_pos:
                    self.widgets["plane_label"].value = (
                        f"{plot_dims[0]}={click_pos[plot_dims[0]]}, {plot_dims[1]}={click_pos[plot_dims[1]]}"
                    )

        # Draw the canvas
        if hasattr(self.update_function, "fig"):
            self.update_function.fig.canvas.draw_idle()

    def _setup_unified_figure(self, plot_data_list):
        """Set up figure layout for unified plotting."""

        has_spatial = any(pd.has_spatial_data() for pd in plot_data_list)
        has_spectral = any(pd.has_spectral_data() for pd in plot_data_list)

        # Determine layout: spatial only, spectral only, or both
        if has_spatial and has_spectral:
            ncols = 2
            figsize = self.processed_plot_params.get("figsize", (16, 6))
        elif has_spatial:
            ncols = 1
            figsize = self.processed_plot_params.get("figsize", (8, 6))
        else:  # spectral only
            ncols = 1
            figsize = self.processed_plot_params.get("figsize", (10, 6))

        fig, axes = plt.subplots(
            1, ncols, figsize=figsize, squeeze=False, constrained_layout=False
        )

        # Use manual subplot adjustments for stable positioning
        if ncols == 2:
            # Two subplots: spatial and spectral
            plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95, wspace=0.3)
        else:
            # Single subplot
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
        self.update_function.fig = fig

        if has_spatial:
            self.update_function.ax_img = axes[0, 0]
            if has_spectral:
                self.update_function.ax_spec = axes[0, 1]
                # Skip aspect ratio enforcement to prevent layout shifts
                # The manual subplot adjustments provide stable positioning
            else:
                self.update_function.ax_spec = None
        else:
            self.update_function.ax_img = None
            self.update_function.ax_spec = axes[0, 0]

        # Initialize mask-related attributes for create_mask compatibility
        self.update_function.current_selector = None
        self.update_function.current_plot_dims = (0, 1)  # Default to read, phase
        self.update_function.current_origin = "lower"
        self.update_function.current_state = [
            None,
            None,
        ]  # [slicing_dim, clicked_indices]

        # Set up colorbars
        if self.update_function.ax_img:
            divider = make_axes_locatable(self.update_function.ax_img)

            # Create colorbars based on number of datasets
            if len(plot_data_list) > 1:
                # Left colorbar for base data, right for overlay
                self.update_function.cax_base = divider.append_axes(
                    "left", size="5%", pad=0.05
                )
                self.update_function.cax_overlay = divider.append_axes(
                    "right", size="5%", pad=0.05
                )
            else:
                # Single colorbar on the right
                self.update_function.cax = divider.append_axes(
                    "right", size="5%", pad=0.05
                )

        # Connect click events
        if self.update_function.ax_img:
            self.update_function.fig.canvas.mpl_connect(
                "button_press_event", self._onclick_spatial
            )
        if self.update_function.ax_spec:
            self.update_function.fig.canvas.mpl_connect(
                "button_press_event", self._onclick_spectral
            )

        # Initialize inset tracking helpers
        self.update_function._spec_inset_ax = None
        self.update_function._spec_inset_artists = []

    def _reprocess_plot_data(self, plot_data, kwargs):
        """Reprocess plot data with current widget values.

        This ensures that preprocessing settings (like channel combination, denoising)
        are applied correctly even for complex dataset types.
        """
        # Create a shallow copy of UnifiedPlotData to avoid mutating the original persistent state
        import copy

        fresh_plot_data = copy.copy(plot_data)

        # Handle multi-variable dataset variable selection
        if fresh_plot_data.data_type == "multi_variable_dataset":
            selected_var = kwargs.get(f"{fresh_plot_data.widget_prefix}_data_variable")

            # Always re-apply preprocessing from the raw dataset variable
            if (
                selected_var
                and hasattr(fresh_plot_data, "dataset_variables")
                and selected_var in fresh_plot_data.dataset_variables
            ):
                # Start from the raw data array
                raw_data = fresh_plot_data.dataset_variables[selected_var]

                # Apply preprocessing (important for RSS, denoising, etc.)
                fresh_plot_data.spatial_data = self._apply_preprocessing(
                    raw_data, kwargs, fresh_plot_data.widget_prefix
                )
                fresh_plot_data.current_data_variable = selected_var

        return fresh_plot_data

    def _render_plot_data(self, plot_data, is_base, kwargs):
        """Render a single UnifiedPlotData object."""
        if plot_data.has_spatial_data() and self.update_function.ax_img:
            self._render_spatial_component(plot_data, is_base, kwargs)
        debug_log_unified_plotter(
            f"---------\nplot_data.has_spectral_data()={plot_data.has_spectral_data()}\nplot_data.has_spatial_data()={plot_data.has_spatial_data()}\nself.update_function.ax_spec={self.update_function.ax_spec}\n---------"
        )

        # Render spectral component if we have spectral data OR if we have anatomical data that can show temporal signals
        if (
            plot_data.has_spectral_data() or plot_data.is_anatomical()
        ) and self.update_function.ax_spec:
            self._render_spectral_component(plot_data, is_base, kwargs)

    def _render_spectral_component(self, plot_data, is_base, kwargs):
        """Render the spectral component of plot data."""
        debug_log_unified_plotter(
            f"_render_spectral_component called - data_type: {plot_data.data_type}, is_base: {is_base}"
        )

        # Handle spectroscopic data rendering logic
        if plot_data.is_spectroscopic():
            # For pure spectroscopy data (like single-voxel NSPECT), no click position needed
            if plot_data.is_pure_spectroscopy():
                debug_log_unified_plotter(
                    "Rendering pure spectroscopy data - no click position required"
                )
                # Proceed directly to spectrum rendering

            # For spatial spectroscopy data, check if click position is needed
            elif plot_data.is_spatial_spectroscopy():
                requires_click = not self._has_click_position() and any(
                    pd.has_spatial_data() for pd in self._plot_data_list
                )

                if requires_click:
                    # Auto-initialize click position to center of spatial data
                    spatial_data = plot_data.spatial_data
                    if spatial_data is not None:
                        # Set click position to center of spatial dimensions
                        spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
                        default_click_pos = {}
                        for dim_name in spatial_dims:
                            if dim_name in spatial_data.dims:
                                size = spatial_data.sizes[dim_name]
                                center_idx = size // 2
                                default_click_pos[dim_name] = center_idx

                        if default_click_pos:
                            self._set_click_position(**default_click_pos)

                    # If still no click position, show the message
                    if not self._has_click_position():
                        if is_base:
                            self.update_function.ax_spec.set_title(
                                "Click spatial plot for Spectrum/FID"
                            )
                        return

        # Render spectrum based on data type
        if plot_data.data_type == "spectroscopy_fitted":
            self._render_fitted_spectrum(plot_data, is_base, kwargs)
        elif plot_data.is_spectroscopic():
            self._render_measured_spectrum(plot_data, is_base, kwargs)
        else:
            # For anatomical data, call specialized rendering directly
            logger.debug("Calling specialized spectral rendering for anatomical data")
            self._render_specialized_spectral(plot_data, is_base, kwargs)

    # def _render_measured_spectrum(self, plot_data, is_base, kwargs):
    # Defined in child class spectroscopyPlotter

    # def _render_fitted_spectrum(self, plot_data, is_base, kwargs):
    # Defined in child class SpectroscopyPlotter

    # def _render_measured_spectrum(self, plot_data, is_base, kwargs):
    #     """Render spectrum from measured data (DataArray)."""
    #     # Only require click position if spectroscopic data has spatial
    #     # component
    #     if (
    #         not self._has_click_position()
    #         and plot_data.is_spectroscopic()
    #         and plot_data.has_spatial_data()
    #     ):
    #         return

    #     # Select voxel signal if spectroscopic and has spatial data
    #     if plot_data.is_spectroscopic() and plot_data.has_spatial_data():
    #         signal_1d = self._select_voxel_signal_unified(
    #             plot_data, kwargs, self._get_click_position()
    #         )
    #     elif plot_data.is_spectroscopic():
    #         # For purely spectral data, use the entire spectrum
    #         signal_1d = plot_data.spectral_data
    #     else:
    #         # For anatomical data, show line profile or similar
    #         signal_1d = self._get_anatomical_profile(plot_data, kwargs)

    #     if signal_1d is None:
    #         return

    #     import datetime

    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #     debug_file = "/tmp/spectroscopy_plot_debug.log"
    #     with open(debug_file, "a") as f:
    #         f.flush()

    #     # Plot 1D signal
    #     self._plot_1d_signal_unified(plot_data, signal_1d, is_base, kwargs)

    #     # Apply specialized spectral rendering
    #     self._render_specialized_spectral(plot_data, is_base, kwargs)

    def _render_interactive_overlays(self, kwargs):
        """Render interactive overlays like masks from create_mask."""
        if not self.interactive_overlays or not hasattr(self.update_function, "ax_img"):
            return

    def _finalize_unified_plot(self, plot_data_list, kwargs):
        """Add colorbars, legends, and final styling."""
        # Handle colorbars based on number of datasets
        if len(plot_data_list) > 1:
            self._create_dual_colorbars(plot_data_list, kwargs)
        else:
            self._create_single_colorbar(plot_data_list, kwargs)

        # Add legend for spectral plots
        if hasattr(self.update_function, "ax_spec") and self.update_function.ax_spec:
            handles, labels = self.update_function.ax_spec.get_legend_handles_labels()
            if handles:
                self.update_function.ax_spec.legend(
                    loc=self.processed_plot_params.get("legend_loc", "upper right"),
                    fontsize=self.processed_plot_params.get("legend_fontsize", 8),
                )
            elif self.update_function.ax_spec.legend_:
                self.update_function.ax_spec.legend_.remove()

    def _create_single_colorbar(self, plot_data_list, kwargs):
        """Create a single colorbar for single dataset plots."""
        if hasattr(self.update_function, "last_im") and hasattr(
            self.update_function, "cax"
        ):
            cbar = plt.colorbar(
                self.update_function.last_im,
                cax=self.update_function.cax,
            )
            cbar.ax.tick_params(labelsize=8)

    def _create_dual_colorbars(self, plot_data_list, kwargs):
        """Create separate colorbars for base and overlay data."""
        # Clear any existing colorbars
        if hasattr(self.update_function, "cax_base") and self.update_function.cax_base:
            self.update_function.cax_base.cla()
        if (
            hasattr(self.update_function, "cax_overlay")
            and self.update_function.cax_overlay
        ):
            self.update_function.cax_overlay.cla()

        # Base data colorbar (left side)
        if (
            hasattr(self.update_function, "last_im")
            and self.update_function.last_im
            and hasattr(self.update_function, "cax_base")
            and self.update_function.cax_base
        ):
            cbar_base = plt.colorbar(
                self.update_function.last_im,
                cax=self.update_function.cax_base,
            )
            cbar_base.ax.tick_params(
                labelsize=self.processed_plot_params.get("cbar_label_fontsize", 8)
            )

            # Get label with unit information for base data
            base_label = self._get_colorbar_label(plot_data_list[0])
            cbar_base.ax.yaxis.set_ticks_position("left")
            cbar_base.ax.yaxis.set_label_position("left")
            cbar_base.set_label(
                base_label,
                fontsize=self.processed_plot_params.get("cbar_label_fontsize", 8),
            )

        # Overlay data colorbar (right side)
        if (
            hasattr(self.update_function, "overlay_ims")
            and self.update_function.overlay_ims
            and hasattr(self.update_function, "cax_overlay")
            and self.update_function.cax_overlay
        ):
            # Use the last overlay image
            overlay_im = self.update_function.overlay_ims[-1]
            cbar_overlay = plt.colorbar(
                overlay_im,
                cax=self.update_function.cax_overlay,
            )
            cbar_overlay.ax.tick_params(
                labelsize=self.processed_plot_params.get("cbar_label_fontsize", 8)
            )

            # Get label with unit information for overlay data
            overlay_label = self._get_colorbar_label(plot_data_list[1])
            cbar_overlay.set_label(
                overlay_label,
                fontsize=self.processed_plot_params.get("cbar_label_fontsize", 8),
            )

    def _get_colorbar_label(self, plot_data, selected_var=None):
        """Get colorbar label with unit information from data attributes."""
        # Start with data type
        base_label = f"{plot_data.data_type.replace('_', ' ').title()}"

        # Try to get unit information from different sources
        unit = None

        if plot_data.source_type == "dataset":
            try:
                # For fitted data, check the dataset variables for unit
                # attributes
                dataset = plot_data.fitted_spectra.get("_dataset")
                if dataset:
                    # Check common variables for unit attributes
                    for var_name in [
                        "fit_amps",
                        "fit_freqs",
                        "fit_t2s",
                        "fit_fwhm_Hz",
                    ]:
                        if var_name in dataset.data_vars:
                            data_var = dataset[var_name]
                            if hasattr(data_var, "attrs") and "unit" in data_var.attrs:
                                unit = data_var.attrs["unit"]
                                break

                    # Also check dataset-level attributes
                    if not unit and hasattr(dataset, "attrs"):
                        if "unit" in dataset.attrs:
                            unit = dataset.attrs["unit"]
            except BaseException:
                pass
        elif plot_data.source_type == "dataarray":
            # For measured data, check the spatial_data for unit attributes
            try:
                if plot_data.spatial_data and "unit" in plot_data.spatial_data.attrs:
                    unit = plot_data.spatial_data.attrs["unit"]
            except BaseException:
                pass

        # Construct final label
        if unit:
            return f"{base_label} ({unit})"
        else:
            return f"{base_label} (a.u.)"

    def _apply_spatial_slicing(self, spatial_data, kwargs, widget_prefix=""):
        """Apply slicing to spatial data based on current slider values."""
        # Identify spatial dimensions (2D dimensions for display)
        spatial_dims = self._get_spatial_dims(spatial_data, kwargs, widget_prefix)

        # Store spatial dimensions for mask creation compatibility
        if hasattr(self, "update_function") and spatial_dims:
            # Store the first two spatial dimensions as the plot dimensions
            self.update_function.current_plot_dims = tuple(spatial_dims[:2])

        # Get all spatial dimensions for proper slicing logic
        all_spatial_dims = self._get_all_spatial_dims(spatial_data)

        # Store slicing dimension for mask creation compatibility
        if hasattr(self, "update_function") and spatial_dims:
            # Determine slicing dimension (typically the third spatial dimension)
            slicing_dim = None
            if len(all_spatial_dims) >= 3:
                # Find the dimension that's not being displayed (the slicing dimension)
                displayed_dims = set(spatial_dims[:2])
                for dim in all_spatial_dims:
                    if dim not in displayed_dims:
                        slicing_dim = dim
                        break

            # Store slicing dimension in current_state
            self.update_function.current_state[0] = slicing_dim

        non_spatial_selectors = {}

        # Create selectors for ALL dimensions that should be sliced
        for key, value in kwargs.items():
            if key in spatial_data.dims and key not in [
                "acquisition_idx",
                "metabolite",
            ]:
                # For spatial dimensions that are not being displayed, we need
                # to slice them
                if key in all_spatial_dims and key not in spatial_dims:
                    non_spatial_selectors[key] = value
                    # Removed debug message to avoid duplicates during normal operation
                # For truly non-spatial dimensions, also slice them
                elif key not in all_spatial_dims:
                    non_spatial_selectors[key] = value

        # Handle metabolite selection for datasets
        widget_prefix = getattr(spatial_data, "_widget_prefix", "ds0")
        metabolite_key = f"{widget_prefix}_metabolite"
        if metabolite_key in kwargs and "metabolite" in spatial_data.dims:
            selected_metabolite = kwargs[metabolite_key]
            non_spatial_selectors["metabolite"] = selected_metabolite

        # Default to first spectral point for spatial display
        if "acquisition_idx" in spatial_data.dims:
            non_spatial_selectors["acquisition_idx"] = kwargs.get("acquisition_idx", 0)

        # Apply selections, keeping spatial dimensions intact
        data_slice = spatial_data

        # Separate sel (label-based) and isel (index-based) selections
        sel_selectors = {}
        isel_selectors = {}

        for key, value in non_spatial_selectors.items():
            if key in data_slice.dims:
                if key == "metabolite":
                    sel_selectors[key] = value
                else:
                    isel_selectors[key] = value

        # Apply label-based selections first
        if sel_selectors:
            try:
                data_slice = data_slice.sel(**sel_selectors)
            except Exception as e:
                warnings.warn(f"Error in sel selection: {e}")

        # Apply index-based selections
        if isel_selectors:
            # Filter to only include dimensions that actually exist AND have valid indices
            # This prevents IndexError when overlaying arrays with different dimension sizes
            # (e.g., 2-channel data vs RSS-combined 1-channel data)
            valid_isel_selectors = {
                k: v
                for k, v in isel_selectors.items()
                if k in data_slice.dims and v < data_slice.sizes[k]
            }
            if valid_isel_selectors:
                data_slice = data_slice.isel(**valid_isel_selectors)

        # Squeeze singleton dimensions that aren't spatial or acquisition_idx
        # This is crucial for multi-channel data where channel_idx=1 can
        # interfere with perspective switching
        singleton_dims_to_squeeze = []
        for dim in data_slice.dims:
            if (
                data_slice.sizes[dim] == 1
                and dim not in spatial_dims
                and dim != "acquisition_idx"
            ):
                singleton_dims_to_squeeze.append(dim)

        if singleton_dims_to_squeeze:
            data_slice = data_slice.squeeze(singleton_dims_to_squeeze)

        return data_slice

    def _get_all_spatial_dims(self, data_array):
        """Get all potential spatial dimensions from a data array."""
        # Find all dimensions that could be spatial (excluding non-spatial
        # dimensions)
        non_spatial_dims = [
            "metabolite",
            "acquisition_idx",
            "channel_idx",
            "repetition_idx",
            "echo_idx",
            "inversion_idx",
            "rfpower_idx",  # B1 mapping RF power dimension
        ]
        all_spatial = [
            dim
            for dim in data_array.dims
            if dim not in non_spatial_dims and data_array.sizes[dim] > 1
        ]

        # Try to order by common spatial dimension names first, then by size
        known_spatial_order = ["read_idx", "phase_idx", "slice_idx", "x", "y", "z"]
        ordered_dims = []

        # Add known spatial dims in order
        for known_dim in known_spatial_order:
            if known_dim in all_spatial:
                ordered_dims.append(known_dim)

        # Add any remaining dims sorted by size (largest first)
        remaining_dims = [dim for dim in all_spatial if dim not in ordered_dims]
        remaining_dims.sort(key=lambda x: data_array.sizes[x], reverse=True)
        ordered_dims.extend(remaining_dims)

        return ordered_dims

    def _get_spatial_dims(self, data_array, kwargs=None, widget_prefix=""):
        """Get the spatial dimensions from a data array with automatic perspective detection."""
        all_spatial = self._get_all_spatial_dims(data_array)

        # For 3D data with standard dimensions, use automatic perspective
        # detection
        if (
            len(all_spatial) >= 3
            and kwargs
            and all(
                dim in all_spatial for dim in ["read_idx", "phase_idx", "slice_idx"]
            )
        ):
            return self._detect_active_perspective(all_spatial, kwargs)

        # Test call:
        active_perspective_dims = self._detect_active_perspective(all_spatial, kwargs)
        return [dim for dim in active_perspective_dims if dim in all_spatial][:2]

    def _detect_active_perspective(self, all_spatial_dims, kwargs):
        """Automatically detect which perspective to use based on widget interactions.

        The idea is to show the two dimensions that are NOT being actively sliced.
        For example, if the user is moving the slice_idx slider, show read_idx vs phase_idx.
        """
        if len(all_spatial_dims) < 3:
            # Not enough dimensions for perspective switching
            return (
                all_spatial_dims[:2] if len(all_spatial_dims) >= 2 else all_spatial_dims
            )

        # If we have exactly 3 spatial dims and standard names, use active
        # perspective detection
        if len(all_spatial_dims) == 3 and all(
            dim in all_spatial_dims for dim in ["read_idx", "phase_idx", "slice_idx"]
        ):
            # Use tracked active perspective dimension if available
            if (
                self._active_perspective_dim is not None
                and self._active_perspective_dim in all_spatial_dims
            ):
                # Show the two dimensions that are NOT the active slicing dimension
                # First dimension = Y axis (up→down), Second dimension = X axis (left→right)
                if self._active_perspective_dim == "slice_idx":
                    # Axial view: shows read_idx vs phase_idx
                    # Want: phase_idx up→down (Y), read_idx left→right (X)
                    return ["phase_idx", "read_idx"]
                elif self._active_perspective_dim == "read_idx":
                    # Sagittal view: shows phase_idx vs slice_idx
                    # Want: slice_idx up→down (Y), phase_idx left→right (X)
                    return ["slice_idx", "phase_idx"]
                elif self._active_perspective_dim == "phase_idx":
                    # Coronal view: shows read_idx vs slice_idx
                    # Want: slice_idx up→down (Y), read_idx left→right (X)
                    return ["slice_idx", "read_idx"]

            # Default to axial view if no active perspective is tracked yet
            # Axial: phase_idx up→down (Y), read_idx left→right (X)
            return ["phase_idx", "read_idx"]

        # For non-standard dimensions, fall back to size-based selection
        # Handle case where kwargs might be None
        if kwargs is None:
            kwargs = {}

        dims_and_sizes = [
            (dim, kwargs.get(f"_data_sizes", {}).get(dim, 1))
            for dim in all_spatial_dims
        ]
        dims_and_sizes.sort(key=lambda x: x[1], reverse=True)
        return (
            [dims_and_sizes[0][0], dims_and_sizes[1][0]]
            if len(dims_and_sizes) >= 2
            else all_spatial_dims[:2]
        )

    def _create_slider_click_sync_observer(self, dim_name):
        """Create observer that syncs slider changes while respecting click updates.

        Parameters
        ----------
        dim_name : str
            Name of the spatial dimension controlled by the slider.

        Returns
        -------
        callable
            Observer function that handles slider updates.
        """

        def observer(change):
            if change["new"] != change["old"]:
                if dim_name in ["read_idx", "phase_idx", "slice_idx"]:
                    if not self._updating_from_click:
                        # Manual slider change - allow perspective switching

                        # Step 1: Update perspective
                        try:
                            new_perspective = dim_name
                        except:
                            new_perspective = "slice_idx"  # Fallback to slice_idx
                        old_perspective = getattr(
                            self, "_active_perspective_dim", "slice_idx"
                        )

                        if new_perspective != old_perspective:

                            # Update active perspective
                            self._set_active_perspective(new_perspective)

                            # Update plane label if it exists
                            if "plane_label" in self.widgets:
                                perspective_name = self._get_perspective_name(
                                    new_perspective
                                )
                                new_label = f"View: {perspective_name} (slicing through {new_perspective})"
                                self.widgets["plane_label"].value = new_label

                        # Update the click position for this dimension so the crosshair follows
                        # and the secondary plot stays updated
                        self._set_click_position(**{dim_name: change["new"]})
                    else:
                        # Click-induced slider change - no perspective switching
                        pass

                    # Trigger plot update for both manual and click-induced changes
                    if (
                        hasattr(self, "update_function")
                        and self.update_function
                        and hasattr(self, "widgets")
                    ):
                        current_kwargs = {
                            k: v.value
                            for k, v in self.widgets.items()
                            if hasattr(v, "value")
                        }
                        perspective_type = (
                            "manual"
                            if not self._updating_from_click
                            else "click-induced"
                        )
                        logger.debug(
                            f": Triggering plot update - {perspective_type} change, perspective: {getattr(self, '_active_perspective_dim', 'slice_idx')}"
                        )
                        self.update_function(**current_kwargs)
                        logger.debug(" Plot update completed")

        return observer

    def _setup_basic_observers(self):
        """Set up observers that change perspective for manual slider moves but not click-induced moves."""
        spatial_dims = ["read_idx", "phase_idx", "slice_idx"]

        # Set initial perspective - changes with manual slider moves, not clicks
        self._set_active_perspective(
            "slice_idx"
        )  # Start with slice perspective (read_idx vs phase_idx)

        # Initialize default click position based on current slider values
        # This ensures we always have a valid voxel selected for crosshairs and secondary plots
        default_click_pos = {}
        for dim_name in spatial_dims:
            if dim_name in self.widgets and hasattr(self.widgets[dim_name], "value"):
                default_click_pos[dim_name] = self.widgets[dim_name].value

        # Always set initial click position from sliders if not already set
        if default_click_pos:
            self._set_click_position(**default_click_pos)
            logger.debug(
                f" Initialized click position from sliders: {default_click_pos}"
            )

        for dim_name in spatial_dims:
            if dim_name in self.widgets and hasattr(self.widgets[dim_name], "value"):
                slider = self.widgets[dim_name]

                # Clear all existing observers
                if (
                    hasattr(slider, "_trait_notifiers")
                    and "value" in slider._trait_notifiers
                ):
                    slider.unobserve_all("value")

                # Add observer that changes perspective for manual moves, not clicks
                slider.observe(
                    self._create_slider_click_sync_observer(dim_name), names="value"
                )
                logger.debug(f" Added observer for {dim_name} slider")

    def _calculate_coordinate_extent(
        self,
        spatial_data,
        spatial_dims,
        data_shape,
        coord_system="Auto",
        data_obj=None,
        kwargs=None,
    ):
        """Calculate proper coordinate extent from xarray data.

        Parameters
        ----------
        spatial_data : xr.DataArray
            Data array containing coordinate information.
        spatial_dims : list
            Spatial dimension names ``[y_dim, x_dim]``.
        data_shape : tuple
            Shape of the 2D data to be plotted.
        coord_system : str, optional
            Coordinate system to use: "Auto", "XYZ", or "RPS". Defaults to "Auto".
        data_obj : object, optional
            Data object (anatomy_obj or spectroscopy_obj) for XYZ extent calculation.
        kwargs : dict, optional
            Widget state containing slider values for determining current slice.

        Returns
        -------
        tuple
            ``(extent, origin, xlabel, ylabel)`` describing the spatial layout.
        """
        ny, nx = data_shape[-2:]

        # Default pixel-based extent
        extent = [-0.5, nx - 0.5, ny - 0.5, -0.5]
        origin = "upper"

        # Add perspective context to labels if available
        perspective_info = ""
        if self._active_perspective_dim is not None:
            perspective_name = self._get_perspective_name(self._active_perspective_dim)
            perspective_info = f" ({perspective_name})"

        xlabel = (
            f"{spatial_dims[1]} [idx]{perspective_info}"
            if len(spatial_dims) > 1
            else f"x [idx]{perspective_info}"
        )
        ylabel = (
            f"{spatial_dims[0]} [idx]{perspective_info}"
            if len(spatial_dims) > 0
            else f"y [idx]{perspective_info}"
        )

        if len(spatial_dims) < 2:
            logger.debug("Less than 2 spatial dims, using pixel extent")
            return extent, origin, xlabel, ylabel

        plot_dim_y, plot_dim_x = spatial_dims[0], spatial_dims[1]

        # Try XYZ coordinate system if requested
        if coord_system == "XYZ" and data_obj is not None and kwargs is not None:
            try:
                # Determine which object coordinate (read_idx, phase_idx, slice_idx) corresponds to the slicing dimension
                # Look for the slicing dimension in kwargs
                object_coord_candidates = ["read_idx", "phase_idx", "slice_idx"]
                object_coord_name = None
                object_index = None

                for coord_name in object_coord_candidates:
                    if coord_name in spatial_data.dims and coord_name in kwargs:
                        object_coord_name = coord_name
                        object_index = kwargs[coord_name]
                        break

                if object_coord_name is not None and object_index is not None:
                    logger.debug(
                        f"Attempting XYZ extent calculation for {object_coord_name}[{object_index}]"
                    )

                    # Call get_xyz_extent_for_coordinate
                    xyz_result = get_xyz_extent_for_coordinate(
                        input_data=spatial_data,
                        object_coord_name=object_coord_name,
                        object_index=object_index,
                        scanner_axes=("x", "y", "z"),
                    )

                    if xyz_result and xyz_result.get("imshow_extent") is not None:
                        extent = xyz_result["imshow_extent"]
                        origin = xyz_result.get("imshow_origin", "lower")

                        # Get plane axes info for labels
                        plane_axes = xyz_result.get("plane_axes")
                        if plane_axes:
                            horiz_axis = plane_axes.get("horizontal", "x")
                            vert_axis = plane_axes.get("vertical", "y")

                            # Get units from coordinates if available
                            horiz_unit = "mm"
                            vert_unit = "mm"
                            if horiz_axis in spatial_data.coords:
                                horiz_unit = spatial_data.coords[horiz_axis].attrs.get(
                                    "unit", "mm"
                                )
                            if vert_axis in spatial_data.coords:
                                vert_unit = spatial_data.coords[vert_axis].attrs.get(
                                    "unit", "mm"
                                )

                            xlabel = (
                                f"{horiz_axis} [{horiz_unit}] (XYZ){perspective_info}"
                            )
                            ylabel = (
                                f"{vert_axis} [{vert_unit}] (XYZ){perspective_info}"
                            )

                            logger.debug(f"Using XYZ extent: {extent}")
                            logger.debug(f"XYZ labels: {xlabel}, {ylabel}")
                            return extent, origin, xlabel, ylabel
                        else:
                            logger.debug(
                                "XYZ extent calculation succeeded but no plane_axes info"
                            )
                    else:
                        logger.debug("XYZ extent calculation returned no imshow_extent")
                else:
                    logger.debug(
                        f"Could not determine object coordinate for XYZ calculation from dims: {spatial_data.dims}"
                    )

            except Exception as e:
                logger.debug(
                    f"XYZ extent calculation failed: {e}, falling back to standard method"
                )

        # Standard coordinate extent calculation (Auto or RPS mode, or XYZ fallback)
        try:
            # Check if coordinates exist and are accessible
            if plot_dim_x in spatial_data.coords and plot_dim_y in spatial_data.coords:
                full_coords_x = spatial_data.coords[plot_dim_x]
                full_coords_y = spatial_data.coords[plot_dim_y]

                # Handle 1D coordinates (most common case)
                if full_coords_x.ndim == 1 and full_coords_y.ndim == 1:
                    x_vals, y_vals = full_coords_x.values, full_coords_y.values

                    # Check for valid numeric values
                    if (
                        np.all(np.isfinite(x_vals))
                        and np.all(np.isfinite(y_vals))
                        and len(x_vals) > 0
                        and len(y_vals) > 0
                    ):
                        # Calculate spacing
                        dx = abs(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1
                        dy = abs(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 1

                        # Create extent (ensure finite values)
                        x_min, x_max = float(x_vals.min()), float(x_vals.max())
                        y_min, y_max = float(y_vals.min()), float(y_vals.max())

                        if all(np.isfinite([x_min, x_max, y_min, y_max, dx, dy])):
                            extent = [
                                x_min - dx / 2.0,
                                x_max + dx / 2.0,
                                y_min - dy / 2.0,
                                y_max + dy / 2.0,
                            ]
                            origin = "lower"

                            # Create axis labels with units and perspective
                            # info
                            x_unit = full_coords_x.attrs.get("unit", "val")
                            y_unit = full_coords_y.attrs.get("unit", "val")

                            # Add perspective context to labels if available
                            perspective_info = ""
                            if self._active_perspective_dim is not None:
                                perspective_name = self._get_perspective_name(
                                    self._active_perspective_dim
                                )
                                perspective_info = f" ({perspective_name})"

                            # Add coordinate system label
                            coord_suffix = (
                                f" ({coord_system})" if coord_system == "RPS" else ""
                            )
                            xlabel = f"{plot_dim_x} [{x_unit}]{coord_suffix}{perspective_info}"
                            ylabel = f"{plot_dim_y} [{y_unit}]{coord_suffix}{perspective_info}"

                            logger.debug(
                                f"Using coordinate extent - x: {x_min:.2f} to {x_max:.2f} {x_unit}"
                            )
                            logger.debug(
                                f"Using coordinate extent - y: {y_min:.2f} to {y_max:.2f} {y_unit}"
                            )
                        else:
                            logger.debug(
                                "Non-finite coordinate values, using pixel extent"
                            )
                    else:
                        logger.debug(
                            "Invalid coordinate values (NaN/Inf), using pixel extent"
                        )

                elif full_coords_x.ndim > 1 or full_coords_y.ndim > 1:
                    # Handle multidimensional coordinates (more complex case)
                    x_min, x_max = np.nanmin(full_coords_x.values), np.nanmax(
                        full_coords_x.values
                    )
                    y_min, y_max = np.nanmin(full_coords_y.values), np.nanmax(
                        full_coords_y.values
                    )

                    # Check for valid values
                    if all(np.isfinite([x_min, x_max, y_min, y_max])):
                        extent = [x_min, x_max, y_min, y_max]
                        origin = "lower"

                        # Create axis labels with units and perspective info
                        x_unit = full_coords_x.attrs.get("unit", "val")
                        y_unit = full_coords_y.attrs.get("unit", "val")

                        # Add perspective context to labels if available
                        perspective_info = ""
                        if self._active_perspective_dim is not None:
                            perspective_name = self._get_perspective_name(
                                self._active_perspective_dim
                            )
                            perspective_info = f" ({perspective_name})"

                        # Add coordinate system label
                        coord_suffix = (
                            f" ({coord_system})" if coord_system == "RPS" else ""
                        )
                        xlabel = (
                            f"{plot_dim_x} [{x_unit}]{coord_suffix}{perspective_info}"
                        )
                        ylabel = (
                            f"{plot_dim_y} [{y_unit}]{coord_suffix}{perspective_info}"
                        )

                    else:
                        logger.debug(
                            "Non-finite multidim coordinate values, using pixel extent"
                        )
            else:
                logger.debug(
                    f"Coordinates {plot_dim_x}, {plot_dim_y} not found in data, using pixel extent"
                )

        except Exception as e:
            logger.debug(
                f"Failed to calculate coordinate extent: {e}, using pixel extent"
            )

        return extent, origin, xlabel, ylabel

    def _get_anatomical_profile(self, plot_data, kwargs):
        """Extract 1D profile from anatomical data at click position."""
        if not self._has_click_position():
            return None

        click_pos = self._get_click_position()
        spatial_data = plot_data.spatial_data

        if spatial_data is None:
            return None

        # Get spatial dimensions
        spatial_dims = self._get_spatial_dims(spatial_data)
        if len(spatial_dims) < 2:
            return None

        # Extract line profile (horizontal line through clicked point)
        try:
            y_dim, x_dim = spatial_dims[0], spatial_dims[1]
            if y_dim in click_pos:
                y_idx = click_pos[y_dim]
                # Create selector for other dimensions
                selector = {}
                for dim in spatial_data.dims:
                    if dim == y_dim:
                        selector[dim] = y_idx
                    elif dim not in [x_dim]:
                        # For other dimensions, use current slider values or
                        # take first
                        if dim in kwargs:
                            idx = kwargs[dim]
                            # Validate index against dimension size
                            if dim in spatial_data.dims:
                                size = spatial_data.sizes[dim]
                                if idx >= size:
                                    # Index out of bounds (e.g., broadcasting or singleton dimension)
                                    # Fallback to 0
                                    idx = 0
                            selector[dim] = idx
                        else:
                            selector[dim] = 0

                # Extract line profile
                profile = spatial_data.isel(**selector)
                return profile

        except Exception as e:
            warnings.warn(f"Failed to extract anatomical profile: {e}")

        return None

    def _get_anatomical_temporal_signal(self, plot_data, kwargs, clicked_indices):
        """Extract temporal/multi-dimensional signal from anatomical data at click position.

        This function is designed for anatomical data that has temporal dimensions
        (like repetition_idx or other non-spatial dimensions) and extracts the signal
        time course at the clicked voxel.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data containing spatial information.
        kwargs : dict
            Widget parameters including ``secondary_plot_dim`` selection.
        clicked_indices : dict
            Spatial indices associated with the current click position.

        Returns
        -------
        tuple
            ``(signal_1d, axis_data, xlabel)`` on success, otherwise ``(None, None, None)``.
        """
        debug_log_unified_plotter(
            f"🔍 _get_anatomical_temporal_signal called - data_type: {plot_data.data_type}"
        )
        debug_log_unified_plotter(f"🔍 clicked_indices: {clicked_indices}")
        debug_log_unified_plotter(
            f"🔍 plot_data.spatial_data is None: {plot_data.spatial_data is None}"
        )

        if not clicked_indices or plot_data.spatial_data is None:
            debug_log_unified_plotter(
                "❌ Early return: no clicked_indices or spatial_data is None"
            )
            return None, None, None

        # For multi-variable datasets, ensure we're using the currently selected variable
        if plot_data.data_type == "multi_variable_dataset":
            debug_log_unified_plotter("🔍 Processing multi-variable dataset")
            selected_var = kwargs.get(f"{plot_data.widget_prefix}_data_variable")
            debug_log_unified_plotter(f"🔍 selected_var from kwargs: {selected_var}")
            debug_log_unified_plotter(
                f"🔍 plot_data.widget_prefix: {plot_data.widget_prefix}"
            )
            debug_log_unified_plotter(
                f"🔍 hasattr dataset_variables: {hasattr(plot_data, 'dataset_variables')}"
            )

            if (
                selected_var
                and hasattr(plot_data, "dataset_variables")
                and selected_var in plot_data.dataset_variables
            ):
                spatial_data = plot_data.dataset_variables[selected_var]
                debug_log_unified_plotter(
                    f"✅ Using selected variable '{selected_var}' for spatial_data"
                )
                debug_log_unified_plotter(
                    f"🔍 Selected variable data dims: {spatial_data.dims}"
                )
                debug_log_unified_plotter(
                    f"🔍 Selected variable data shape: {spatial_data.shape}"
                )
            else:
                spatial_data = plot_data.spatial_data
                debug_log_unified_plotter(f"⚠️ Fallback to plot_data.spatial_data")
        else:
            spatial_data = plot_data.spatial_data
            debug_log_unified_plotter(
                f"🔍 Using regular spatial_data for data_type: {plot_data.data_type}"
            )

        # Get the secondary plot dimension from widget
        secondary_dim = kwargs.get(
            f"{plot_data.widget_prefix}_secondary_plot_dim", None
        )
        debug_log_unified_plotter(
            f"🔍 secondary_plot_dim widget key: {plot_data.widget_prefix}_secondary_plot_dim"
        )
        debug_log_unified_plotter(f"🔍 secondary_dim from kwargs: {secondary_dim}")
        debug_log_unified_plotter(f"🔍 spatial_data.dims: {spatial_data.dims}")

        def _collapse_to_secondary_dimension(data_array):
            """Ensure only the secondary dimension remains with a length > 1."""
            if not isinstance(data_array, xr.DataArray):
                return data_array

            if secondary_dim not in data_array.dims:
                debug_log_unified_plotter(
                    f"🧭 collapse skipped - '{secondary_dim}' not present in {data_array.dims}"
                )
                return data_array

            extra_dims = [dim for dim in data_array.dims if dim != secondary_dim]
            if not extra_dims:
                return data_array

            selection = {}
            for dim in extra_dims:
                dim_size = data_array.sizes[dim]
                if dim_size <= 1:
                    continue

                idx_value = None
                prefixed_key = f"{plot_data.widget_prefix}_{dim}"
                if prefixed_key in kwargs:
                    idx_value = kwargs[prefixed_key]
                elif dim in kwargs:
                    idx_value = kwargs[dim]
                elif dim in clicked_indices:
                    idx_value = clicked_indices[dim]

                if idx_value is None:
                    debug_log_unified_plotter(
                        f"⚠️ No selector for dim '{dim}', defaulting to index 0"
                    )
                    idx_value = 0

                try:
                    idx_int = int(idx_value)
                except (TypeError, ValueError):
                    debug_log_unified_plotter(
                        f"⚠️ Unable to cast selector '{idx_value}' for dim '{dim}', using 0"
                    )
                    idx_int = 0

                idx_int = max(0, min(idx_int, dim_size - 1))
                selection[dim] = idx_int

            if selection:
                debug_log_unified_plotter(
                    f"🧭 Applying extra-dimension selection: {selection}"
                )
                data_array = data_array.isel(**selection)

            singleton_dims = [
                dim
                for dim in data_array.dims
                if dim != secondary_dim and data_array.sizes[dim] == 1
            ]

            if singleton_dims:
                debug_log_unified_plotter(
                    f"🧭 Squeezing singleton dims (excluding secondary): {singleton_dims}"
                )
                data_array = data_array.squeeze(dim=singleton_dims, drop=True)

            return data_array

        if not secondary_dim or secondary_dim not in spatial_data.dims:
            debug_log_unified_plotter(
                f"❌ Secondary dimension '{secondary_dim}' not valid"
            )
            debug_log_unified_plotter(
                f"🔍 Available secondary widget kwargs: {[k for k in kwargs.keys() if 'secondary' in k]}"
            )
            return None, None, None

        # Get all non-spatial dimensions
        spatial_dims = self._get_spatial_dims(spatial_data)
        debug_log_unified_plotter(f"🔍 spatial_dims detected: {spatial_dims}")

        non_spatial_dims = [
            dim
            for dim in spatial_data.dims
            if dim not in spatial_dims and spatial_data.sizes[dim] > 1
        ]
        debug_log_unified_plotter(f"🔍 non_spatial_dims available: {non_spatial_dims}")

        if secondary_dim not in non_spatial_dims:
            debug_log_unified_plotter(
                f"❌ secondary_dim '{secondary_dim}' not in non_spatial_dims: {non_spatial_dims}"
            )
            return None, None, None

        try:
            debug_log_unified_plotter(
                f"✅ Attempting temporal signal extraction for dim: {secondary_dim}"
            )

            # Create selector for clicked position and other dimensions
            selector = {}

            # Add clicked spatial position
            selector.update(clicked_indices)
            debug_log_unified_plotter(
                f"🔍 Added clicked spatial position to selector: {clicked_indices}"
            )

            # For other non-spatial dimensions (except the one we want to plot),
            # use current slider values or default to first index
            for dim in non_spatial_dims:
                if dim != secondary_dim:
                    if dim in kwargs:
                        selector[dim] = kwargs[dim]
                        debug_log_unified_plotter(
                            f"🔍 Added {dim}={kwargs[dim]} from kwargs to selector"
                        )
                    else:
                        selector[dim] = 0
                        debug_log_unified_plotter(
                            f"🔍 Added {dim}=0 (default) to selector"
                        )

            debug_log_unified_plotter(f"🔍 Final selector for isel: {selector}")
            signal_data = spatial_data.isel(**selector)
            debug_log_unified_plotter(
                f"✅ Signal data extracted, dims: {signal_data.dims}, shape: {signal_data.shape}"
            )

            # Handle repetition averaging if requested
            has_repetitions = "repetition_idx" in signal_data.dims
            average_reps = kwargs.get(
                f"{plot_data.widget_prefix}_average_repetitions", False
            )
            debug_log_unified_plotter(
                f"🔍 has_repetitions: {has_repetitions}, average_reps: {average_reps}"
            )

            if has_repetitions and average_reps and secondary_dim != "repetition_idx":
                debug_log_unified_plotter("📊 Processing with repetition averaging")
                # Calculate mean and std across repetitions
                signal_mean = signal_data.mean(dim="repetition_idx")
                signal_std = signal_data.std(dim="repetition_idx")

                signal_mean = _collapse_to_secondary_dimension(signal_mean)
                signal_std = _collapse_to_secondary_dimension(signal_std)

                # Create axis data for the secondary dimension
                # Prefer 'time' coordinate over 'repetition_idx' for temporal plots
                if secondary_dim == "repetition_idx" and "time" in signal_mean.coords:
                    debug_log_unified_plotter(
                        "✨ Using 'time' coordinate for averaged signal"
                    )
                    axis_data = signal_mean.coords["time"].values
                    axis_unit = signal_mean.coords["time"].attrs.get("unit", "s")
                    xlabel = f"Time [{axis_unit}] (mean ± std over repetitions)"
                elif secondary_dim in signal_mean.coords:
                    axis_data = signal_mean.coords[secondary_dim].values
                    axis_unit = signal_mean.coords[secondary_dim].attrs.get(
                        "unit", "idx"
                    )
                    xlabel = (
                        f"{secondary_dim} [{axis_unit}] (mean ± std over repetitions)"
                    )
                else:
                    axis_data = np.arange(signal_mean.sizes[secondary_dim])
                    axis_unit = "idx"
                    xlabel = (
                        f"{secondary_dim} [{axis_unit}] (mean ± std over repetitions)"
                    )
                debug_log_unified_plotter(
                    f"✅ Returning averaged signal: mean_shape={signal_mean.shape}, std_shape={signal_std.shape}"
                )
                debug_log_unified_plotter(
                    f"✅ axis_data: {axis_data}, xlabel: {xlabel}"
                )

                return (signal_mean, signal_std), axis_data, xlabel

            else:
                debug_log_unified_plotter("📊 Processing without repetition averaging")
                # No repetition averaging - return the signal along secondary
                # dimension
                signal_data = _collapse_to_secondary_dimension(signal_data)

                # Prefer 'time' coordinate over 'repetition_idx' for temporal plots
                if secondary_dim == "repetition_idx" and "time" in signal_data.coords:
                    debug_log_unified_plotter(
                        "✨ Using 'time' coordinate instead of repetition_idx"
                    )
                    axis_data = signal_data.coords["time"].values
                    axis_unit = signal_data.coords["time"].attrs.get("unit", "s")
                    # Use more descriptive label when using time coordinate
                    xlabel = f"Time [{axis_unit}]"
                elif secondary_dim in signal_data.coords:
                    axis_data = signal_data.coords[secondary_dim].values
                    axis_unit = signal_data.coords[secondary_dim].attrs.get(
                        "unit", "idx"
                    )
                    xlabel = f"{secondary_dim} [{axis_unit}]"
                else:
                    axis_data = np.arange(signal_data.sizes[secondary_dim])
                    axis_unit = "idx"
                    xlabel = f"{secondary_dim} [{axis_unit}]"
                debug_log_unified_plotter(
                    f"✅ Returning regular signal: signal_shape={signal_data.shape}"
                )
                debug_log_unified_plotter(
                    f"✅ axis_data: {axis_data}, xlabel: {xlabel}"
                )

                return signal_data, axis_data, xlabel

        except Exception as e:
            debug_log_unified_plotter(
                f"💥 Exception in temporal signal extraction: {e}"
            )
            import traceback

            debug_log_unified_plotter(f"💥 Traceback: {traceback.format_exc()}")
            import warnings

            warnings.warn(f"Failed to extract anatomical temporal signal: {e}")

        debug_log_unified_plotter(
            "❌ Returning None - temporal signal extraction failed"
        )
        return None, None, None

    def _plot_anatomical_temporal_signal(
        self, plot_data, signal_data, axis_data, xlabel, is_base, kwargs
    ):
        """Plot temporal signal from anatomical data.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the anatomical dataset.
        signal_data : xr.DataArray or tuple
            Signal to plot, optionally provided as ``(mean, std)``.
        axis_data : array-like
            Values shown on the x-axis.
        xlabel : str
            X-axis label.
        is_base : bool
            ``True`` when visualizing the primary dataset.
        kwargs : dict
            Widget parameters controlling plot appearance.
        """
        if signal_data is None or not hasattr(self.update_function, "ax_spec"):
            return

        # Define part function for data transformation
        spec_part_val = kwargs.get(f"{plot_data.widget_prefix}_spec_part", "Magnitude")
        part_func_map = {
            "Magnitude": np.abs,
            "Real": np.real,
            "Imaginary": np.imag,
            "Phase": np.angle,
        }
        part_func = part_func_map.get(spec_part_val, np.abs)

        # Check if transformation was already applied (e.g. in preprocessing)
        already_applied = False
        if isinstance(signal_data, tuple):
            if hasattr(signal_data[0], "attrs"):
                already_applied = (
                    signal_data[0].attrs.get("spec_part_applied") == spec_part_val
                )
        elif hasattr(signal_data, "attrs"):
            already_applied = (
                signal_data.attrs.get("spec_part_applied") == spec_part_val
            )

        # Handle mean ± std case
        if isinstance(signal_data, tuple) and len(signal_data) == 2:
            signal_mean, signal_std = signal_data

            if already_applied:
                mean_values = signal_mean.values
                std_values = signal_std.values
            else:
                mean_values = part_func(signal_mean.values)
                std_values = part_func(signal_std.values)

            # Plot styling
            color = "blue" if is_base else "red"
            alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
            label = "Temporal Signal" if is_base else "Overlay Signal"

            # Plot mean line
            self.update_function.ax_spec.plot(
                axis_data,
                mean_values,
                label=label,
                color=color,
                alpha=alpha,
                linewidth=self.processed_plot_params.get("linewidth", 2.0),
            )

            # Plot error bars
            self.update_function.ax_spec.fill_between(
                axis_data,
                mean_values - std_values,
                mean_values + std_values,
                alpha=alpha * 0.3,
                color=color,
                label=f"{label} ± std",
            )

        else:
            # Regular signal plotting
            if already_applied:
                signal_values = signal_data.values
            else:
                signal_values = part_func(signal_data.values)

            # Plot styling
            color = "blue" if is_base else "red"
            alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
            label = "Temporal Signal" if is_base else "Overlay Signal"

            self.update_function.ax_spec.plot(
                axis_data,
                signal_values,
                label=label,
                color=color,
                alpha=alpha,
                linewidth=self.processed_plot_params.get("linewidth", 2.0),
            )

        # Set labels
        self.update_function.ax_spec.set_xlabel(xlabel)
        self.update_function.ax_spec.set_ylabel(f"{spec_part_val} Intensity")

        # Apply synced y-axis limits if requested
        if (
            hasattr(self.update_function, "_synced_ylim")
            and self.update_function._synced_ylim is not None
        ):
            self.update_function.ax_spec.set_ylim(self.update_function._synced_ylim)

    def _onclick_spatial(self, event):
        """Handle click events on the spatial plot."""
        if (
            not hasattr(self.update_function, "ax_img")
            or event.inaxes != self.update_function.ax_img
        ):
            logger.debug("Click event rejected - not on spatial plot")
            return

        # Calculate click indices based on current plot
        if hasattr(self.update_function, "_current_plot_dims"):
            plot_dims = self.update_function._current_plot_dims
            data_shape = getattr(self.update_function, "_current_data_shape", None)
            extent = getattr(self.update_function, "_current_extent", None)
            origin = getattr(self.update_function, "_current_origin", "upper")

            if not all([plot_dims, data_shape, extent]):
                logger.debug("Missing plot metadata - cannot calculate click position")
                return

            plot_dim_y, plot_dim_x = plot_dims

            ny, nx = data_shape

            # Check for valid click coordinates
            if (
                event.xdata is None
                or event.ydata is None
                or not np.isfinite(event.xdata)
                or not np.isfinite(event.ydata)
            ):
                logger.debug("Invalid click coordinates")
                return

            # Check for valid extent values
            if not all(np.isfinite(extent)):
                logger.debug("Invalid extent values, cannot calculate click position")
                return

            # Calculate click indices
            # Convert from extent coordinates to array indices
            x_range = extent[1] - extent[0]
            y_range = extent[3] - extent[2]

            if x_range == 0 or y_range == 0:
                logger.debug("Zero extent range, cannot calculate click position")
                return

            # Map click coordinates to array indices
            # extent is [x_min, x_max, y_min, y_max]
            # Array indices: x goes from 0 to nx-1, y goes from 0 to ny-1

            # For x: map from extent[0] -> 0 to extent[1] -> nx-1
            x_idx = int(np.round((event.xdata - extent[0]) / x_range * (nx - 1)))

            # For y: depends on origin
            if origin == "upper":
                # In 'upper' origin: extent[2] corresponds to array index ny-1,
                # extent[3] to index 0
                y_idx = int(np.round((extent[3] - event.ydata) / y_range * (ny - 1)))
            else:
                # In 'lower' origin: extent[2] corresponds to array index 0,
                # extent[3] to index ny-1
                y_idx = int(np.round((event.ydata - extent[2]) / y_range * (ny - 1)))

            x_idx = max(0, min(x_idx, nx - 1))
            y_idx = max(0, min(y_idx, ny - 1))
            # Update click position
            self._set_click_position(**{plot_dim_y: y_idx, plot_dim_x: x_idx})

            # Sync sliders to clicked position
            self._sync_sliders_to_click_position(plot_dim_y, y_idx, plot_dim_x, x_idx)

            # Trigger plot update
            if self.update_function and self.widgets:
                current_kwargs = {
                    k: v.value for k, v in self.widgets.items() if hasattr(v, "value")
                }
                self.update_function(**current_kwargs)
            else:
                logger.debug(
                    "Cannot trigger plot update - missing update_function or widgets"
                )
        else:
            logger.debug("Missing _current_plot_dims - cannot calculate click position")

    def _onclick_spectral(self, event):
        """Handle click events on the spectral plot."""
        if (
            not hasattr(self.update_function, "ax_spec")
            or event.inaxes != self.update_function.ax_spec
        ):
            return

        # Add vertical line at clicked x position
        if event.xdata is not None:
            # Remove any existing spectral click lines
            if hasattr(self.update_function, "_spec_click_lines"):
                for line in self.update_function._spec_click_lines:
                    try:
                        line.remove()
                    except ValueError:
                        pass  # Line already removed

            # Add new vertical line at click position
            line = self.update_function.ax_spec.axvline(
                x=event.xdata, color="black", linestyle="-", alpha=0.8, linewidth=1.0
            )

            # Store the line for future removal
            self.update_function._spec_click_lines = [line]

            # Refresh the canvas to show the line
            self.update_function.fig.canvas.draw_idle()

        # Handle spectral axis clicking (e.g., for acquisition_idx updates)
        if (
            hasattr(self.update_function, "_spec_axis_data")
            and hasattr(self.update_function, "_show_temporal")
            and event.xdata is not None
        ):
            axis_data = self.update_function._spec_axis_data
            self.update_function._show_temporal

            # Find the closest point in the axis data
            try:
                closest_idx = np.argmin(np.abs(axis_data - event.xdata))

                # Update the acquisition_idx widget if it exists
                if "acquisition_idx" in self.widgets and hasattr(
                    self.widgets["acquisition_idx"], "value"
                ):
                    self.widgets["acquisition_idx"].value
                    max_value = self.widgets["acquisition_idx"].max
                    new_value = min(closest_idx, max_value)

                    self.widgets["acquisition_idx"].value = new_value

                    # Trigger plot update
                    if self.update_function and self.widgets:
                        current_kwargs = {
                            k: v.value
                            for k, v in self.widgets.items()
                            if hasattr(v, "value")
                        }
                        self.update_function(**current_kwargs)

            except Exception as e:
                logger.debug(f"Error processing spectral click: {e}")

    def _set_click_position(self, **coords):
        """Set the unified click position.

        Parameters
        ----------
        **coords
            Coordinate values such as ``read_idx=5`` or ``phase_idx=10``.
        """
        # Update the unified click position dictionary
        self._click_positions.update(coords)

    def clear_click_position(self):
        """Reset the stored click position so the next render uses widget defaults."""
        self._click_positions.clear()

    def _get_click_position(self):
        """Get the unified click position.

        Returns
        -------
        dict
            Copy of the current click position coordinates.
        """
        return self._click_positions.copy()

    def _has_click_position(self):
        """Check if a click position has been set.

        Returns
        -------
        bool
            ``True`` if a click position is available.
        """
        return bool(self._click_positions)

    def _process_plot_params(self, plot_params):
        """Process and validate plot parameters.

        Parameters
        ----------
        plot_params : dict or None
            Raw plot parameters.

        Returns
        -------
        dict
            Processed plot parameters with defaults applied.
        """
        if plot_params is None:
            plot_params = {}

        # Default parameters
        defaults = {
            "figsize": (16, 6),
            "colormap": "bone",
            "overlay_colormap": "magma",
            "linecolor": "blue",
            "overlay_linecolor": "red",
            "linewidth": 1.0,
            "vmin_perc": 0.0,
            "vmax_perc": 100.0,
            "default_spec_part": "Magnitude",
            "legend_loc": "upper right",
            "legend_fontsize": 8,
            "cbar_label_fontsize": 8,
            "xlabel": None,
            "ylabel": None,
            "show_legend": True,
            "widget_fontsize": 10,
            "range_mode": "Percentage",
            "combine_channels": False,
            "interpolate_image": True,
        }
        if type(defaults["widget_fontsize"]) is int:
            defaults["widget_fontsize"] = f"{defaults['widget_fontsize']}px"

        # Merge with provided parameters
        processed = defaults.copy()
        processed.update(plot_params)

        return processed

    def _vmin_vmax_range_check(
        self,
        range_mode,
        data_to_plot_spatial,
        image_source_changed,
        min_sig,
        max_sig,
        plot_data={},
    ):

        spec_part_key = f"{plot_data.widget_prefix}_spec_part"
        spec_part_widget = self.widgets.get(spec_part_key, "Magnitude")
        if hasattr(spec_part_widget, "value"):  # Widget object
            spec_part_val = spec_part_widget.value
        else:
            spec_part_val = (
                spec_part_widget if isinstance(spec_part_widget, str) else "Magnitude"
            )

        # part_func = {
        #     "Magnitude": np.abs,
        #     "Real": np.real,
        #     "Imaginary": np.imag,
        #     "Phase": np.angle,
        # }.get(spec_part_val, np.abs)

        # If data part is magnitude and range mode is Min/Max,
        # recalculate min_sig and max_sig from magnitude values

        data_min_abs = np.nanmin(data_to_plot_spatial)
        data_max_abs = np.nanmax(data_to_plot_spatial)
        vmin_slider = self.widgets.get(f"{plot_data.widget_prefix}_vmin")
        vmax_slider = self.widgets.get(f"{plot_data.widget_prefix}_vmax")

        spec_part_val = self.widgets.get(
            f"{plot_data.widget_prefix}_spec_part", "Magnitude"
        )

        # Check if we need to reconfigure sliders due to mode switch
        if range_mode == "Min/Max" and "Abs" not in vmin_slider.description:
            # Switching TO Min/Max mode: reconfigure sliders but preserve user
            # intent

            # Save current user values to preserve their intent
            current_vmin = vmin_slider.value
            current_vmax = vmax_slider.value

            if "%" in str(vmin_slider.description):
                # Coming from percentage mode - convert percentages to absolute
                # values
                vmin_percent = current_vmin / 100.0
                vmax_percent = current_vmax / 100.0
                desired_vmin = min_sig + vmin_percent * (max_sig - min_sig)
                desired_vmax = min_sig + vmax_percent * (max_sig - min_sig)
            else:
                # Already in absolute mode but need to update ranges
                desired_vmin = max(min_sig, min(current_vmin, max_sig))
                desired_vmax = max(min_sig, min(current_vmax, max_sig))

            # Update descriptions
            vmin_slider.description = "Vmin (Abs)"
            vmax_slider.description = "Vmax (Abs)"
            step = (max_sig - min_sig) / 200.0
            if step <= 0:
                step = 0.01

            # Only update ranges, not values, to avoid jumping
            if vmin_slider.min != min_sig or vmin_slider.max != max_sig:
                # Temporarily disable observe to prevent triggering during
                # setup
                # FIX: Update vmax first to establish correct upper bound before vmin update
                # triggers recursive checks. This prevents "Max == Min" collapse.
                with vmax_slider.hold_trait_notifications():
                    vmax_slider.min = min_sig
                    vmax_slider.max = max_sig
                    vmax_slider.step = step
                    vmax_slider.value = desired_vmax

                with vmin_slider.hold_trait_notifications():
                    vmin_slider.min = min_sig
                    vmin_slider.max = max_sig
                    vmin_slider.step = step
                    vmin_slider.value = desired_vmin

            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value
        elif range_mode == "Min/Max" and image_source_changed:
            # Image changed but already in Min/Max mode - just update ranges if
            # needed
            step = (max_sig - min_sig) / 200.0
            if step <= 0:
                step = 0.01

            # Clamp current values to new range
            current_vmin = max(min_sig, min(vmin_slider.value, max_sig))
            current_vmax = max(min_sig, min(vmax_slider.value, max_sig))

            # Only update if ranges actually changed
            if vmin_slider.min != min_sig or vmin_slider.max != max_sig:
                with vmax_slider.hold_trait_notifications():
                    vmax_slider.min = min_sig
                    vmax_slider.max = max_sig
                    vmax_slider.step = step
                    vmax_slider.value = current_vmax

                with vmin_slider.hold_trait_notifications():
                    vmin_slider.min = min_sig
                    vmin_slider.max = max_sig
                    vmin_slider.step = step
                    vmin_slider.value = current_vmin

            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value
        elif range_mode == "Percentage" and "%" not in vmin_slider.description:
            # Switching TO Percentage mode: reconfigure sliders
            vmin_slider.description = "Vmin: [%]"
            vmax_slider.description = "Vmax: [%]"
            vmin_slider.min, vmax_slider.min = 0.0, 0.0
            vmin_slider.max, vmax_slider.max = 100.0, 100.0
            vmin_slider.step, vmax_slider.step = 0.1, 0.1
            vmin_slider.value = 0.0  # Reset to 0%
            vmax_slider.value = 100.0  # Reset to 100%

            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value

        # Calculate final vmin_val and vmax_val for imshow
        if range_mode == "Min/Max":
            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value
        else:  # Percentage mode
            vmax_for_perc = data_max_abs if data_max_abs > 0 else 1.0
            vmin_val = data_min_abs + vmin_slider.value / 100.0 * vmax_for_perc
            vmax_val = (vmax_slider.value / 100.0) * vmax_for_perc

        return vmin_val, vmax_val

    def _get_clean_metabolite_label(self, metabolite_coord, is_overlay=False):
        """Convert xarray metabolite coordinate to clean string label."""
        if hasattr(metabolite_coord, "values"):
            metab_name = str(metabolite_coord.values)
        else:
            metab_name = str(metabolite_coord)

        # Remove any numpy array formatting artifacts
        metab_name = metab_name.replace("'", "").strip()

        # Add overlay indicator if needed
        suffix = " (overlay)" if is_overlay else ""
        return f"{metab_name}{suffix}"

    def _apply_preprocessing_to_fitted_data(self, plot_data, kwargs):
        """Apply preprocessing to fitted data using current widget values.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data object with fitted spectroscopy results.
        kwargs : dict
            Current widget values.

        Returns
        -------
        UnifiedPlotData
            Updated plot data containing reprocessed spectral data.
        """
        # Always reconstruct with current click position and widget values
        dataset = plot_data.fitted_spectra.get("_dataset")
        if dataset is not None:
            # Reconstruct with current parameters (including updated click
            # position)
            raw_data = self._reconstruct_fitted_spectra_for_preprocessing(
                dataset, kwargs, plot_data.widget_prefix
            )
            if raw_data is not None:
                # Apply the same preprocessing as measured data
                processed_data = self._apply_preprocessing(
                    raw_data, kwargs, plot_data.widget_prefix
                )
                # Update both raw and processed data
                plot_data.raw_spectral_data = raw_data
            else:
                # Fallback to existing data if reconstruction fails
                if plot_data.raw_spectral_data is not None:
                    processed_data = self._apply_preprocessing(
                        plot_data.raw_spectral_data, kwargs, plot_data.widget_prefix
                    )
                else:
                    return plot_data
        else:
            # No dataset available, try to use existing raw data
            if plot_data.raw_spectral_data is not None:
                processed_data = self._apply_preprocessing(
                    plot_data.raw_spectral_data, kwargs, plot_data.widget_prefix
                )
            else:
                return plot_data

        # Update the plot data with reprocessed spectral data
        fresh_plot_data = UnifiedPlotData()
        fresh_plot_data.source_type = plot_data.source_type
        fresh_plot_data.data_type = plot_data.data_type
        fresh_plot_data.is_overlay = plot_data.is_overlay
        fresh_plot_data.widget_prefix = plot_data.widget_prefix
        fresh_plot_data.spatial_data = (
            plot_data.spatial_data
        )  # Keep spatial data unchanged
        fresh_plot_data.spectral_data = processed_data  # Use reprocessed spectral data
        fresh_plot_data.raw_spectral_data = (
            plot_data.raw_spectral_data
        )  # Keep raw version
        fresh_plot_data.fitted_spectra = (
            plot_data.fitted_spectra
        )  # Keep dataset reference
        fresh_plot_data.metadata = plot_data.metadata
        fresh_plot_data.metabolite_names = plot_data.metabolite_names

        return fresh_plot_data

    def _reconstruct_fitted_spectra_for_preprocessing(
        self, dataset, kwargs, widget_prefix
    ):
        """This is overwritten in spectroscopyPlotter, Method Resolution Order MRO
        ensures the correct method is called based on the instance type.
        """
        return None

    # Template method pattern for specialization
    def _get_specialized_widgets(
        self, plot_data, processed_plot_params, widget_prefix, slider_defaults
    ):
        """Get specialized widgets common to both spectroscopy and anatomy plotters.

        This consolidates the common widget creation logic from both child classes.
        Child classes can still override this method to add additional specialized widgets.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the dataset.
        processed_plot_params : dict
            Plot parameters after default handling.
        widget_prefix : str
            Prefix applied to widget names.
        slider_defaults : dict
            Default slider values supplied by the caller.

        Returns
        -------
        dict
            Mapping of widget names to widget instances.
        """
        widgets_dict = {}

        # Use Dropdown for spec_part selection (common to both plotters)
        widgets_dict["spec_part"] = Dropdown(
            options=["Magnitude", "Real", "Imaginary", "Phase"],
            value=processed_plot_params.get("default_spec_part", "Magnitude"),
            description="Data Part:",
            layout=Layout(width="200px"),
        )

        # Add image/k-space toggle for spatial data (common to both plotters)
        if plot_data.has_spatial_data():
            widgets_dict["show_image_space"] = ToggleButton(
                value=True,
                description="Image/k-space",
                tooltip="Switch between Image and k-space view",
                icon="image",
                layout=Layout(width="125px", height="25px"),
            )

            # Add interpolation controls for spatial data
            interpolate_widget = Checkbox(
                value=processed_plot_params.get("interpolate_image", True),
                description="Interpolate Image",
                tooltip="Interpolate image",
                disabled=False,
                layout=Layout(width="auto"),
            )
            widgets_dict["interpolate_image"] = interpolate_widget

            # Add sync y-range checkbox
            sync_y_widget = Checkbox(
                value=False,
                description="Sync Y-Range",
                tooltip="Use spatial vmin/vmax for secondary plot y-axis",
                layout=Layout(width="auto"),
            )
            widgets_dict["sync_y_range"] = sync_y_widget

            # Group them in an HBox for layout
            widgets_dict["spatial_options_row"] = HBox(
                [interpolate_widget, sync_y_widget],
                layout=Layout(display="flex", flex_flow="row wrap"),
            )

            # Add combine channels for multichannel data (critical functionality)
            has_channels = False
            if (
                plot_data.spatial_data is not None
                and "channel_idx" in plot_data.spatial_data.dims
                and plot_data.spatial_data.sizes["channel_idx"] > 1
            ):
                has_channels = True
            elif plot_data.data_type == "multi_variable_dataset" and hasattr(
                plot_data, "dataset_variables"
            ):
                # Check if any variable in the dataset has channels
                for var_data in plot_data.dataset_variables.values():
                    if (
                        "channel_idx" in var_data.dims
                        and var_data.sizes["channel_idx"] > 1
                    ):
                        has_channels = True
                        break

            if has_channels:
                widgets_dict["combine_channels"] = Checkbox(
                    value=processed_plot_params.get("combine_channels", False),
                    description="Combine Channels (RSS)",
                    tooltip="Combine channels using root-sum-of-squares",
                )

        return widgets_dict

    @staticmethod
    def _densify_axis_values(axis_values, factor):
        """Generate an interpolated axis by inserting additional samples per interval."""
        axis_array = np.asarray(axis_values)
        if factor <= 1 or axis_array.size < 2:
            return axis_array.astype(axis_array.dtype, copy=True)

        segments = []
        for start, stop in zip(axis_array[:-1], axis_array[1:]):
            segment = np.linspace(
                start, stop, factor + 1, endpoint=False, dtype=axis_array.dtype
            )
            segments.append(segment)
        segments.append(np.asarray([axis_array[-1]], dtype=axis_array.dtype))
        return np.concatenate(segments)

    def _find_dynamic_coordinate_for_reconstruction(
        self, model_name, candidate_data, dataset
    ):
        """Find the coordinate in candidate_data that corresponds to the dynamic axis of the model.

        This uses heuristics to match the model's required reconstruction parameter
        (e.g., inversion_times_ms) with available coordinates in the reference data.

        Parameters
        ----------
        model_name : str
            Name of the fitting model.
        candidate_data : xr.DataArray
            Reference (measured) data containing the dynamic dimension.
        dataset : xr.Dataset
            Fit result dataset (maps).

        Returns
        -------
        tuple
            (kwarg_name, coord_array) or (None, None) if not found.
        """
        # Mapping of model to its primary reconstruction keyword argument
        MODEL_DYNAMIC_KWARG = {
            "t1_inversion_recovery": "inversion_times_ms",
            "monoexponential_adc": "b_values",
            "b1_sinusoidal": "refpower_W",
            "patlak": "time_points_s",
            "tofts_model": "time_points_s",
            "two_compartment_model": "time_points_s",
            "FID": "sampling_time_axis_s",
        }

        # Mapping of model to coordinate name hints in measured data
        MODEL_COORD_HINTS = {
            "t1_inversion_recovery": [
                "inversion_idx",
                "inversion_times",
                "TI",
                "inversion_times_ms",
            ],
            "monoexponential_adc": ["b_value", "b_values", "b_idx"],
            "b1_sinusoidal": ["rfpower_idx", "rf_power", "power", "refpower_W"],
            "patlak": ["time", "acquisition_time_s", "repetition_idx", "time_points_s"],
            "tofts_model": [
                "time",
                "acquisition_time_s",
                "repetition_idx",
                "time_points_s",
            ],
            "two_compartment_model": [
                "time",
                "acquisition_time_s",
                "repetition_idx",
                "time_points_s",
            ],
            "FID": ["acquisition_idx", "time", "t", "sampling_time_axis_s"],
        }

        kwarg_name = MODEL_DYNAMIC_KWARG.get(model_name)
        if not kwarg_name:
            return None, None

        hints = MODEL_COORD_HINTS.get(model_name, [])

        # 1. Try to find a coordinate that matches the standard hints
        for hint in hints:
            if hint in candidate_data.coords:
                return kwarg_name, candidate_data.coords[hint]

        # 2. Heuristic: find dimensions in candidate_data NOT in dataset maps
        # (excluding known static dimensions like channel_idx)
        dataset_dims = set(dataset.dims)
        extra_dims = [
            d
            for d in candidate_data.dims
            if d not in dataset_dims
            and d not in ["channel_idx", "acquisition_idx"]
            and candidate_data.sizes[d] > 1
        ]

        if len(extra_dims) == 1:
            dim_name = extra_dims[0]
            return kwarg_name, candidate_data.coords[dim_name]

        # 3. Last resort: if acquisition_idx is in candidate_data and fits the size, use it
        if (
            "acquisition_idx" in candidate_data.dims
            and candidate_data.sizes["acquisition_idx"] > 1
        ):
            return kwarg_name, candidate_data.coords["acquisition_idx"]

        return None, None

    def _get_reconstruction_axis_info(self, dataset):
        """Return metadata describing the reconstruction axis for interpolation widgets."""
        if not isinstance(dataset, xr.Dataset):
            return None

        model_name = dataset.attrs.get("model_name", "")
        if model_name == "t1_inversion_recovery":
            axis = dataset.attrs.get("inversion_times_ms")
            if axis is None:
                return None
            unit = None
            if hasattr(axis, "attrs"):
                unit = axis.attrs.get("unit")
            unit_str = f" ({unit})" if unit else ""
            return {
                "axis_attr": "inversion_times_ms",
                "axis_label": f"Inversion Time{unit_str}",
                "slider_description": "Interp Factor:",
                "tooltip": "Increase to evaluate fitted curves on a denser inversion-time grid.",
            }
        elif model_name in ["tofts_model", "patlak", "two_compartment_model"]:
            axis = dataset.attrs.get("time_points_s")
            if axis is None:
                return None
            return {
                "axis_attr": "time_points_s",
                "axis_label": "Time (s)",
                "slider_description": "Interp Factor:",
                "tooltip": "Increase to evaluate fitted curves on a denser time grid.",
            }
        return None

    def _get_reconstruction_axis_override(
        self, dataset, selector, widget_prefix, kwargs
    ):
        """
        Build keyword overrides for hp.reconstruct when interpolation or space override is requested.

        Parameters
        ----------
        dataset : xr.Dataset
            Fit dataset providing reconstruction metadata.
        selector : dict
            Mapping of dimension names to indices identifying the voxel.
        widget_prefix : str
            Prefix used for widget names.
        kwargs : dict
            Current widget values.

        Returns
        -------
        dict or None
            Dictionary with keys ``kwargs`` (mapping of reconstruction keyword
            arguments), ``axis_da`` (interpolated axis as DataArray) and
            ``axis_label`` (string). Returns ``None`` if no override is needed.
        """
        if not isinstance(dataset, xr.Dataset):
            return None

        # --- Handle Interpolation ---
        factor_key = f"{widget_prefix}_reco_interp_factor"
        factor = kwargs.get(factor_key, kwargs.get("reco_interp_factor", 1))
        try:
            factor = int(factor)
        except (TypeError, ValueError):
            factor = 1

        # --- Handle Space Override (Tofts) ---
        space_key = f"{widget_prefix}_reco_space"
        space_val = kwargs.get(space_key, kwargs.get("reco_space"))

        reco_kwargs = {}
        if space_val:
            reco_kwargs["reco_space"] = str(space_val).lower()

        if factor <= 1:
            # If no interpolation, check if we still need a space override
            if reco_kwargs:
                return {"kwargs": reco_kwargs, "axis_da": None, "axis_label": None}
            return None

        axis_info = self._get_reconstruction_axis_info(dataset)
        if not axis_info:
            if reco_kwargs:
                return {"kwargs": reco_kwargs, "axis_da": None, "axis_label": None}
            return None

        axis_attr_name = axis_info["axis_attr"]
        axis_source = dataset.attrs.get(axis_attr_name)
        if axis_source is None:
            if reco_kwargs:
                return {"kwargs": reco_kwargs, "axis_da": None, "axis_label": None}
            return None

        if isinstance(axis_source, xr.DataArray):
            axis_da = axis_source
            selector = selector or {}
            axis_selector = {
                dim: selector[dim]
                for dim in axis_da.dims
                if dim in selector and dim in axis_da.dims
            }
            try:
                axis_da = axis_da.isel(axis_selector)
            except Exception:
                pass
            axis_da = axis_da.squeeze()
            axis_attrs = dict(getattr(axis_da, "attrs", {}))
            axis_values = axis_da.values
            axis_dim_name = axis_da.dims[-1] if axis_da.dims else "dynamic_idx"
        else:
            axis_values = np.asarray(axis_source)
            axis_attrs = {}
            axis_dim_name = (
                "inversion_idx"
                if axis_attr_name == "inversion_times_ms"
                else "dynamic_idx"
            )

        axis_values = np.asarray(axis_values).astype(float)
        if axis_values.ndim == 0:
            axis_values = axis_values[np.newaxis]

        interpolated_values = self._densify_axis_values(axis_values, factor)
        interpolated_da = xr.DataArray(
            interpolated_values,
            dims=(axis_dim_name,),
            coords={axis_dim_name: interpolated_values},
            attrs=axis_attrs,
        )

        # Add interpolated axis to kwargs
        reco_kwargs[axis_attr_name] = interpolated_da

        return {
            "kwargs": reco_kwargs,
            "axis_da": interpolated_da,
            "axis_label": axis_info["axis_label"],
        }

    def _apply_specialized_preprocessing(self, data_array, kwargs, widget_prefix):
        """Override in child classes to add specialized preprocessing.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array.
        kwargs : dict
            Widget parameters controlling preprocessing.
        widget_prefix : str
            Prefix applied to widget names.

        Returns
        -------
        xr.DataArray
            Processed data array.
        """
        return data_array

    def _render_specialized_spectral(self, plot_data, is_base, kwargs):
        """Override in child classes to add specialized spectral rendering.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data object.
        is_base : bool
            ``True`` when rendering the primary dataset.
        kwargs : dict
            Widget parameters controlling the rendering.
        """

    def _combine_channels_rss(self, data_array):
        """Combine channels using root-sum-of-squares.

        .. deprecated::
            This method is deprecated. Use :func:`hypermri.utils.mri.channels.combine_channels` instead.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array with a ``channel_idx`` dimension.

        Returns
        -------
        xr.DataArray
            Data array with channels combined via RSS.
        """
        warnings.warn(
            "UnifiedPlotter._combine_channels_rss() is deprecated. "
            "Use hypermri.utils.mri.channels.combine_channels() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Delegate to the new utility
        from hypermri.utils.mri.channels import combine_channels

        return combine_channels(data_array, method="rss")

    def _update_spatial_crosshair(self, click_pos):
        """Update the red cross position on the spatial plot."""
        if (
            not hasattr(self.update_function, "ax_img")
            or not self.update_function.ax_img
        ):
            return

        # Get current plot dimensions and extent
        if not hasattr(self.update_function, "_current_plot_dims"):
            return

        plot_dims = self.update_function._current_plot_dims
        data_shape = getattr(self.update_function, "_current_data_shape", None)
        extent = getattr(self.update_function, "_current_extent", None)
        origin = getattr(self.update_function, "_current_origin", "upper")

        if not all([plot_dims, data_shape, extent]):
            return

        plot_dim_y, plot_dim_x = plot_dims
        ny, nx = data_shape

        # Get indices for the current plotting dimensions
        # If we don't have the exact dimensions, try to find them or use reasonable defaults
        if plot_dim_x not in click_pos or plot_dim_y not in click_pos:
            # Debug what dimensions are available vs what we need
            logger.debug(
                f"Crosshair dimensions mismatch - need {plot_dim_x}, {plot_dim_y}, have {list(click_pos.keys())}"
            )

            # Try to use the click position if it has any matching dimensions
            # or use center of data as fallback
            if plot_dim_x in click_pos:
                x_idx = click_pos[plot_dim_x]
            else:
                x_idx = nx // 2  # Use center as fallback

            if plot_dim_y in click_pos:
                y_idx = click_pos[plot_dim_y]
            else:
                y_idx = ny // 2  # Use center as fallback

            logger.debug(
                f"Using fallback crosshair position: {plot_dim_x}={x_idx}, {plot_dim_y}={y_idx}"
            )
        else:
            x_idx = click_pos[plot_dim_x]
            y_idx = click_pos[plot_dim_y]

        # Convert indices back to extent coordinates
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]

        # Map array indices to extent coordinates
        x_coord = extent[0] + (x_idx / (nx - 1)) * x_range

        if origin == "upper":
            y_coord = extent[3] - (y_idx / (ny - 1)) * y_range
        else:
            y_coord = extent[2] + (y_idx / (ny - 1)) * y_range

        # Remove existing crosshair lines
        if hasattr(self.update_function, "crosshair_lines"):
            for line in self.update_function.crosshair_lines:
                try:
                    line.remove()
                except (ValueError, NotImplementedError):
                    pass
            self.update_function.crosshair_lines = []
        else:
            self.update_function.crosshair_lines = []

        # Remove legacy marker if it exists
        if (
            hasattr(self.update_function, "click_marker")
            and self.update_function.click_marker is not None
        ):
            try:
                self.update_function.click_marker.remove()
            except ValueError:
                pass
            self.update_function.click_marker = None

        # Draw new crosshair lines
        # Use slightly transparent red lines
        v_line = self.update_function.ax_img.axvline(
            x=x_coord, color="white", linestyle="--", alpha=0.7, linewidth=0.5
        )
        h_line = self.update_function.ax_img.axhline(
            y=y_coord, color="white", linestyle="--", alpha=0.7, linewidth=0.5
        )

        self.update_function.crosshair_lines = [v_line, h_line]

        # Update the plot
        if hasattr(self.update_function, "fig") and self.update_function.fig:
            self.update_function.fig.canvas.draw_idle()

    def _compute_overlay_difference(self, overlay_data, base_data, percent=False):
        """Compute difference between overlay and base data.

        Parameters
        ----------
        overlay_data : np.ndarray or xr.DataArray
            Overlay data array.
        base_data : np.ndarray or xr.DataArray
            Base data array.
        percent : bool
            If True, compute percent difference: 100 * (overlay - base) / base
            If False, compute simple difference: overlay - base

        Returns
        -------
        np.ndarray or xr.DataArray
            Difference data (same type as input).
        """
        if base_data is None:
            return overlay_data

        try:
            # Extract values if xarray
            overlay_vals = (
                overlay_data.values if hasattr(overlay_data, "values") else overlay_data
            )
            base_vals = base_data.values if hasattr(base_data, "values") else base_data

            # Ensure same shape
            if overlay_vals.shape != base_vals.shape:
                logger.warning(
                    f"Shape mismatch in difference: overlay {overlay_vals.shape} vs base {base_vals.shape}"
                )
                return overlay_data

            # Compute difference
            if percent:
                # Avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    diff = 100.0 * (overlay_vals - base_vals) / np.abs(base_vals)
                    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                diff = overlay_vals - base_vals

            # Return in same format as input
            if hasattr(overlay_data, "values"):
                return overlay_data.copy(data=diff)
            else:
                return diff

        except Exception as e:
            logger.warning(f"Failed to compute difference: {e}")
            return overlay_data

    def _apply_kspace_transform(self, data_array):
        """Apply k-space/image space transformation.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array.

        Returns
        -------
        xr.DataArray
            Data transformed between image and k-space domains.
        """
        try:
            spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
            transform_dims = [
                dim
                for dim in spatial_dims
                if dim in data_array.dims and data_array.sizes[dim] > 1
            ]

            if not transform_dims:
                return data_array

            # Apply FFT along spatial dimensions
            axes = [data_array.get_axis_num(dim) for dim in transform_dims]
            transformed_data = np.fft.fftshift(
                np.fft.fftn(np.fft.ifftshift(data_array.values, axes=axes), axes=axes),
                axes=axes,
            )

            return data_array.copy(data=transformed_data)

        except Exception as e:
            warnings.warn(f"k-space transformation failed: {e}. Using original data.")
            return data_array

    def _select_voxel_signal_unified(self, plot_data, kwargs, clicked_indices):
        """Select voxel signal from plot data."""
        data_array = plot_data.spectral_data
        if data_array is None:
            return None

        isel_selector = {}
        sel_selector = {}

        # Add clicked position
        if clicked_indices:
            isel_selector.update(clicked_indices)

        # Add other dimension selections (exclude acquisition_idx as that's
        # what we want to plot)
        for key, value in kwargs.items():
            if key in data_array.dims and key not in ["acquisition_idx"]:
                if key == "metabolite":
                    sel_selector[key] = value
                else:
                    isel_selector[key] = value

        # Filter valid selectors with bounds checking
        # Prevents IndexError when overlaying arrays with different dimension sizes
        isel_selector = {
            k: v
            for k, v in isel_selector.items()
            if k in data_array.dims and v < data_array.sizes[k]
        }
        sel_selector = {k: v for k, v in sel_selector.items() if k in data_array.dims}

        # Apply selections
        selected_data = data_array
        if sel_selector:
            selected_data = selected_data.sel(**sel_selector)
        if isel_selector:
            selected_data = selected_data.isel(**isel_selector)

        return selected_data

    # def _plot_1d_signal_unified(self, plot_data, signal_1d, is_base, kwargs):
    #     """Plot 1D signal for both measured and fitted data."""
    #     # Get display parameters
    #     show_temporal = kwargs.get(f"{plot_data.widget_prefix}_view_toggle", False)
    #     spec_unit = (
    #         "ppm" if kwargs.get(f"{plot_data.widget_prefix}_spec_unit", True) else "Hz"
    #     )
    #     # Get npoints and validate range - find the acquisition dimension length
    #     if "acquisition_idx" in signal_1d.dims:
    #         max_npoints = signal_1d.sizes["acquisition_idx"]
    #     else:
    #         # Fallback to total length if no acquisition_idx
    #         max_npoints = len(signal_1d)

    #     requested_npoints = kwargs.get(
    #         f"{plot_data.widget_prefix}_npoints", max_npoints
    #     )
    #     # Clamp to maximum available
    #     npoints = min(requested_npoints, max_npoints)

    #     # Warn if requested npoints exceeds available data
    #     if requested_npoints > max_npoints:
    #         warnings.warn(
    #             f"Requested npoints ({requested_npoints}) exceeds signal length ({max_npoints}). "
    #             f"Using maximum available points: {max_npoints}"
    #         )

    #     # Use the already processed signal (domain transformation was applied
    #     # during preprocessing)
    #     # Slice to npoints along the acquisition dimension
    #     if "acquisition_idx" in signal_1d.dims:
    #         signal_to_plot = signal_1d.isel(acquisition_idx=slice(0, npoints))
    #     else:
    #         signal_to_plot = signal_1d[:npoints]  # Fallback for 1D data

    #     # Get axis data based on the user's desired view (show_temporal)
    #     # The signal domain is already transformed during preprocessing
    #     signal_domain = signal_to_plot.attrs.get("signal_domain", "temporal")

    #     # Check if we need to display different axis than what the signal was transformed to
    #     # The logic should be: if user wants temporal view, show time axis; else show frequency
    #     if show_temporal:
    #         if hasattr(self.data_obj, "sampling_time_axis_s"):
    #             axis_data = self.data_obj.sampling_time_axis_s[:npoints]
    #         else:
    #             axis_data = calc_sampling_time_axis(self.data_obj)[:npoints]
    #         xlabel = "Time (s)"
    #     else:
    #         axis_data = get_freq_axis(
    #             self.data_obj, spec_unit, getattr(self.data_obj, "cut_off", 0), npoints
    #         )
    #         xlabel = f"Frequency ({spec_unit})"

    #     # Apply data transformation
    #     spec_part_val = kwargs.get(f"{plot_data.widget_prefix}_spec_part", "Magnitude")

    #     # part_func = {
    #     #     "Magnitude": np.abs,
    #     #     "Real": np.real,
    #     #     "Imaginary": np.imag,
    #     #     "Phase": np.angle,
    #     # }.get(spec_part_val, np.abs)

    #     # Ensure signal_to_plot is 1D for plotting
    #     signal_values = signal_to_plot.values

    #     # Find the acquisition dimension index
    #     acq_dim_idx = None
    #     if "acquisition_idx" in signal_to_plot.dims:
    #         acq_dim_idx = signal_to_plot.dims.index("acquisition_idx")

    #     # Squeeze out singleton dimensions, but preserve the acquisition
    #     # dimension
    #     if signal_values.ndim > 1:
    #         # Get all singleton dimensions except the acquisition dimension
    #         singleton_dims = []
    #         for i in range(signal_values.ndim):
    #             if signal_values.shape[i] == 1 and i != acq_dim_idx:
    #                 singleton_dims.append(i)

    #         if singleton_dims:
    #             # Squeeze out singleton dimensions
    #             signal_values = np.squeeze(signal_values, axis=tuple(singleton_dims))
    #         elif acq_dim_idx is not None:
    #             # If no singleton dims but we have acquisition dim, move it to
    #             # the end and take first slice of others
    #             signal_values = np.moveaxis(signal_values, acq_dim_idx, -1)
    #             # Take first slice of all non-acquisition dimensions
    #             while signal_values.ndim > 1:
    #                 signal_values = signal_values[0]
    #         else:
    #             # Fallback: just squeeze everything
    #             signal_values = signal_values.squeeze()

    #     # Check if data part transformation was already applied in preprocessing
    #     already_applied = signal_to_plot.attrs.get("spec_part_applied", None)
    #     if already_applied == spec_part_val:
    #         # Transformation already applied, use as-is
    #         plot_data_values = signal_values
    #     else:
    #         # Apply transformation now
    #         plot_data_values = part_func(
    #             input_data=signal_values, signal_part=spec_part_val
    #         )

    #     # Ensure we have the right length
    #     min_len = min(len(axis_data), len(plot_data_values))
    #     axis_data = axis_data[:min_len]
    #     plot_data_values = plot_data_values[:min_len]

    #     # Store axis data for click handling (only for base plot)
    #     if is_base:
    #         self.update_function._spec_axis_data = axis_data
    #         self.update_function._show_temporal = show_temporal

    #     # Plot styling
    #     label = "Signal" if is_base else "Overlay"
    #     color = (
    #         self.processed_plot_params.get("linecolor", "blue")
    #         if is_base
    #         else self.processed_plot_params.get("overlay_linecolor", "red")
    #     )
    #     alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)

    #     self.update_function.ax_spec.plot(
    #         axis_data,
    #         plot_data_values,
    #         label=label,
    #         color=color,
    #         alpha=alpha,
    #         linewidth=self.processed_plot_params.get("linewidth", 1.0),
    #     )

    #     # Add gray reference line at y=0 for spectral data (when not showing temporal view)
    #     if not show_temporal and is_base:
    #         self.update_function.ax_spec.axhline(
    #             y=0, color="gray", linestyle="-", alpha=0.5, linewidth=0.8
    #         )

    #     # Set labels
    #     self.update_function.ax_spec.set_xlabel(xlabel)
    #     self.update_function.ax_spec.set_ylabel(f"{spec_part_val} Intensity")

    #     if spec_unit == "ppm" and signal_domain == "spectral":
    #         self.update_function.ax_spec.invert_xaxis()
    #     print(f"-------->---------->-------->")
