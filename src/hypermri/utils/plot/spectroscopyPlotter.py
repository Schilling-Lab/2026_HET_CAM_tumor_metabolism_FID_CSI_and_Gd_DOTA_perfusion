import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import (
    HTML,
    Checkbox,
    Dropdown,
    FloatText,
    HBox,
    IntSlider,
    Layout,
    Output,
    VBox,
)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from hypermri.analysis.spectroscopy.core import freq_Hz_to_ppm, get_freq_axis
from hypermri.utils.mathematics.calculations import calc_sampling_time_axis, part_func
from hypermri.utils.plot.unifiedPlotter import UnifiedPlotter
from hypermri.utils.system.logging import debug_log, init_default_logger

# Import fit_maps to register the hp accessor on xarray datasets
try:
    from hypermri.analysis.fitting import fit_maps  # noqa: F401
except ImportError:
    # Handle case where fit_maps might not be available
    pass

logger = init_default_logger(__name__)

DEBUG_FILE = "/tmp/spectroscopy_plot_debug.log"
DEBUG_LOG = True


def debug_log_spectroscopy_plot(
    message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG
):
    """Debug logging system for SpectroscopyPlotter similar to create_mask_simplified."""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


class SpectroscopyPlotter(UnifiedPlotter):
    """Interactive MR spectroscopy plotter with fitting integration.

    The plotter supports single-voxel spectroscopy (SVS), chemical shift imaging
    (CSI/MRSI), fitted spectra overlays, metabolite decomposition, and interactive
    voxel inspection with synchronized spatial views.
    """

    def __init__(
        self,
        spectroscopy_obj,
        preprocessor=None,
        cache_manager=None,
        data_selector=None,
        widget_factory=None,
    ):
        """Initialize the spectroscopy plotter.

        Parameters
        ----------
        spectroscopy_obj : BaseSpectroscopy
            Spectroscopy data object containing measured or fitted spectra.
        preprocessor : DataPreprocessor, optional
            Preprocessing helper used to transform data before rendering. A
            default instance is created when omitted.
        cache_manager : CacheManager, optional
            Cache manager used to reuse intermediate results. A default instance is
            used when omitted.
        data_selector : DataSelector, optional
            Selector responsible for spatial dimension handling. A default instance is
            created when omitted.
        widget_factory : WidgetFactory, optional
            Factory that builds the interactive widgets. A default instance is created
            when omitted.

        Notes
        -----
        Caching, preprocessing, and widget infrastructure is delegated to the
        :class:`UnifiedPlotter` base class.
        """
        # Initialize parent class infrastructure
        super().__init__(
            spectroscopy_obj, preprocessor, cache_manager, data_selector, widget_factory
        )
        self.spectroscopy_obj = spectroscopy_obj
        self.is_fit_plot = False
        # _click_positions and _plot_data_list now in parent

    def _get_cached_reconstruction(self, dataset, kwargs, widget_prefix):
        """Retrieve cached fitted spectra reconstruction or rebuild it.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing fitted spectroscopy data.
        kwargs : dict
            Widget state including selectors.
        widget_prefix : str
            Prefix used to namespace widget parameters.

        Returns
        -------
        object or None
            Reconstructed spectrum data when successful, otherwise ``None``.
        """
        if dataset is None:
            return None

        # Generate cache key
        data_id = id(dataset)

        # Include relevant widget values in cache key
        isel_selector = kwargs.get("isel_selector", {})
        sel_selector = kwargs.get("sel_selector", {})
        cache_params = {
            **kwargs,
            "isel_selector": (
                str(sorted(isel_selector.items())) if isel_selector else ""
            ),
            "sel_selector": str(sorted(sel_selector.items())) if sel_selector else "",
        }

        cache_key = self.cache_manager.generate_cache_key(
            widget_prefix, cache_params, data_id
        )

        # Check cache
        cached_result = self.cache_manager.get(cache_key, category="fitted_spectra")
        if cached_result is not None:
            return cached_result

        # Cache miss - reconstruct
        result = self._reconstruct_single_voxel_spectrum(
            dataset, isel_selector, sel_selector, kwargs, widget_prefix
        )

        # Cache result
        if result is not None:
            self.cache_manager.set(cache_key, result, category="fitted_spectra")

        return result

    # ========================================================================
    # End of Spectroscopy-specific Cache Methods
    # ========================================================================

    def plot(
        self,
        input_array_list: list = None,
        plot_params: dict = None,
        return_handles: bool = False,
        return_plot_params_info: bool = False,
        **slider_defaults,
    ):
        """Visualize spectroscopy data interactively.

        Parameters
        ----------
        input_array_list : list of (xr.DataArray or xr.Dataset), optional
            Spectroscopy datasets to display. When omitted, data is pulled from
            ``self.spectroscopy_obj``.
        plot_params : dict, optional
            Plot customization dictionary, for example ``colormap``, ``figsize``,
            ``vmin_perc``, ``vmax_perc``, ``show_fit``, or ``show_residual``.
        return_handles : bool, optional
            When ``True``, return plotting handles instead of rendering directly.
        return_plot_params_info : bool, optional
            When ``True``, prints information about all available plot_params and
            slider_defaults and returns None (does not create the plot).
            When True, input_array_list can be None.
        **slider_defaults
            Default widget values including options such as ``overlay_data``,
            ``input_data``, ``show_fit_results``, ``overlay_display_mode``,
            ``linebroadening``, and ``npoints``.

        Returns
        -------
        tuple or None
            ``(fig, ax_img, ax_spec, widgets, update_func, controls_box)`` when
            ``return_handles`` is ``True``; otherwise ``None``.
            When ``return_plot_params_info`` is ``True``, returns ``None``.

        Raises
        ------
        AttributeError
            Raised when no input data is provided and the spectroscopy object
            lacks usable data arrays.
        TypeError
            Raised when ``input_array_list`` is neither a list nor an xarray object.

        Notes
        -----
        The method automatically detects measured versus fitted data, integrates
        with the ``review_fit`` workflow, and relies on the caching provided by
        :class:`UnifiedPlotter` for performance.

        Examples
        --------
        >>> plotter.plot()
        >>> plotter.plot(overlay_data=fitted_dataset, show_fit_results=True)
        >>> plotter.plot(plot_params={'colormap': 'jet', 'figsize': (20, 6)})
        >>> plotter.plot(return_plot_params_info=True)  # Print available parameters
        """

        # Handle parameter information request
        if return_plot_params_info:
            params_info = {
                "optional_plot_params": {
                    "figsize": "Figure size as tuple (width, height). Default: (16, 6)",
                    "colormap": "Colormap for spatial visualization. Default: 'bone'",
                    "overlay_colormap": "Colormap for overlay data. Default: 'magma'",
                    "linecolor": "Line color for primary spectrum. Default: 'blue'",
                    "overlay_linecolor": "Line color for overlay spectrum. Default: 'red'",
                    "linewidth": "Line width for spectral plots. Default: 1.0",
                    "vmin_perc": "Minimum percentile for colormap scaling (0-100). Default: 0.0",
                    "vmax_perc": "Maximum percentile for colormap scaling (0-100). Default: 100.0",
                    "default_spec_part": "Default spectral component to display ('Magnitude', 'Real', 'Imaginary', 'Phase'). Default: 'Magnitude'",
                    "legend_loc": "Legend location. Default: 'upper right'",
                    "legend_fontsize": "Legend font size. Default: 8",
                    "cbar_label_fontsize": "Colorbar label font size. Default: 8",
                    "xlabel": "Custom x-axis label. Default: None (auto-generated)",
                    "ylabel": "Custom y-axis label. Default: None (auto-generated)",
                    "show_legend": "Whether to show legend. Default: True",
                    "widget_fontsize": "Font size for widgets (e.g., '10px'). Default: '10px'",
                },
                "optional_slider_defaults": {
                    "overlay_data": "xr.Dataset containing fitted spectroscopy data to overlay. Used in review_fit workflow.",
                    "input_data": "xr.DataArray containing raw spectroscopy data. Alternative to input_array_list.",
                    "show_fit_results": "bool - Display fit parameters as text overlay (requires fitted data). Default: False",
                    "overlay_display_mode": "bool - Toggle between individual metabolite peaks (False) and sum view (True). Default: False",
                    "show_spec_inset": "bool - Display zoomed inset view for selected frequency band. Default: False",
                    "spec_inset_min_Hz": "float - Minimum frequency [Hz] for spectral inset. Default: 250.0",
                    "spec_inset_max_Hz": "float - Maximum frequency [Hz] for spectral inset. Default: 500.0",
                    "linebroadening": "float - Line broadening factor for spectral processing (Hz). Default: varies by data",
                    "npoints": "int - Number of points for spectral interpolation. Default: varies by data",
                },
                "return_values": {
                    "return_handles=False": "None - Displays plot inline in Jupyter notebook",
                    "return_handles=True": "Tuple of (fig, axes, widgets, update_func, controls_box) for custom integration",
                },
                "data_requirements": {
                    "input_array_list": "List of xr.DataArray (measured spectra) or xr.Dataset (fitted spectra). "
                    "Can contain both measured and fitted data for comparison.",
                    "fitted_data_format": "xr.Dataset must contain 'fit_amps' and 'fit_freqs' data variables for fitted spectra. "
                    "Optional: 'fit_linewidths', 'fit_phases' for detailed visualization.",
                },
                "special_workflows": {
                    "review_fit": "Pass raw data via input_data= and fitted results via overlay_data= to enable fit review mode with residuals.",
                    "multi_voxel": "For CSI data with spatial dimensions, interactive voxel selection is automatically enabled.",
                },
            }

            # Print parameter information in formatted style
            print("=" * 80)
            print("Parameter Information for SpectroscopyPlotter.plot()")
            print("=" * 80)

            print(f"\n{'Optional Parameters in plot_params:':-<80}")
            for param, description in params_info["optional_plot_params"].items():
                print(f"  • {param}")
                print(f"      {description}")

            print(f"\n{'Optional Parameters in **slider_defaults:':-<80}")
            for param, description in params_info["optional_slider_defaults"].items():
                print(f"  • {param}")
                print(f"      {description}")

            print(f"\n{'Return Values:':-<80}")
            for case, description in params_info["return_values"].items():
                print(f"  • {case}")
                print(f"      {description}")

            print(f"\n{'Data Requirements:':-<80}")
            for param, description in params_info["data_requirements"].items():
                print(f"  • {param}")
                print(f"      {description}")

            print(f"\n{'Special Workflows:':-<80}")
            for workflow, description in params_info["special_workflows"].items():
                print(f"  • {workflow}")
                print(f"      {description}")

            print("=" * 80)

            return None

        # Check if hp accessor is available on xarray Datasets
        test_ds = xr.Dataset()
        if not hasattr(test_ds, "hp"):
            debug_log_spectroscopy_plot(
                "ERROR: hp accessor not registered! Check imports. (fit_maps)"
            )

        debug_log_spectroscopy_plot("=== SpectroscopyPlotter.plot called ===")

        # ENHANCED: Check for review_fit style call with overlay_data (fitted data)
        overlay_data = slider_defaults.pop("overlay_data", None)
        input_data = slider_defaults.pop("input_data", None)

        # Enhanced input validation with review_fit support
        if input_array_list is None:
            if input_data is not None:
                # review_fit style call - input_data is the raw spectroscopy data
                input_array_list = [input_data]
            elif (
                hasattr(self.spectroscopy_obj, "reco_data")
                and self.spectroscopy_obj.reco_data is not None
            ):
                input_array_list = [self.spectroscopy_obj.reco_data]
            elif (
                hasattr(self.spectroscopy_obj, "data_array")
                and self.spectroscopy_obj.data_array is not None
            ):
                input_array_list = [self.spectroscopy_obj.data_array]
            else:
                raise AttributeError(
                    "Input data not provided and neither self.spectroscopy_obj.reco_array "
                    "nor self.spectroscopy_obj.data_array found."
                )

        # Add overlay_data to input_array_list if provided (for review_fit)
        if overlay_data is not None:
            if not isinstance(input_array_list, list):
                input_array_list = [input_array_list]
            input_array_list.append(overlay_data)

        try:
            self.processed_plot_params = self._process_plot_params(plot_params)
        except Exception as e:
            warnings.warn(f"Error processing plot parameters: {e}. Using defaults.")
            self.processed_plot_params = self._process_plot_params(None)

        if not input_array_list:
            debug_log_spectroscopy_plot("ERROR: No data provided to plot")
            return

        # Validate input types and check hp accessor availability
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

        # Convert to list if needed
        if not isinstance(input_array_list, list):
            input_array_list = [input_array_list]

        # ENHANCED: Better fit plot detection for review_fit scenarios
        self.is_fit_plot = False

        # Check if any input contains fitted data
        for i, data_item in enumerate(input_array_list):
            if isinstance(data_item, xr.Dataset):
                if (
                    "fit_amps" in data_item.data_vars
                    and "fit_freqs" in data_item.data_vars
                ):
                    self.is_fit_plot = True
                    break

        # Special handling for review_fit: overlay_data contains fitted results
        if overlay_data is not None and isinstance(overlay_data, xr.Dataset):
            if (
                "fit_amps" in overlay_data.data_vars
                and "fit_freqs" in overlay_data.data_vars
            ):
                self.is_fit_plot = True

        # Handle return_handles for review_fit integration
        if return_handles:
            return self._plot_with_handles(
                input_array_list, plot_params=plot_params, **slider_defaults
            )

        try:
            self._plot_unified(input_array_list, **slider_defaults)
        except Exception as e:
            debug_log_spectroscopy_plot(f"Unified plotting failed: {e}")
            warnings.warn(f"Unified plotting failed: {e}.")

    def _plot_with_handles(self, input_array_list, plot_params=None, **slider_defaults):
        """Plot with handles returned for review_fit integration.

        This method provides the 6-element tuple expected by review_fit in base_spectroscopy.py.
        It follows the same pattern as AnatomyPlotter._plot_with_handles but returns the
        complete 6-element tuple including widgets_dict.

        Parameters
        ----------
        input_array_list : list
            Collection of spectroscopy inputs to render.
        plot_params : dict, optional
            Plot customization parameters forwarded from the caller.
        **slider_defaults
            Default widget values used when building the UI.
        """

        try:
            # Initialize processed plot params first
            try:
                self.processed_plot_params = self._process_plot_params(plot_params)
            except Exception as e:
                warnings.warn(f"Error processing plot parameters: {e}. Using defaults.")
                self.processed_plot_params = self._process_plot_params(None)

            # Convert inputs to standardized UnifiedPlotData objects
            self._plot_data_list = self._prepare_plot_data_list(
                input_array_list, slider_defaults
            )

            if not self._plot_data_list:
                raise ValueError("No valid data to plot.")

            # Create widgets
            controls_box, self.widgets = self._create_unified_widgets(
                self._plot_data_list, self.processed_plot_params, slider_defaults
            )

            # Create output area
            from IPython.display import clear_output

            output_area = Output()

            # Set up the update function with enhanced state tracking
            def update_plot(**kwargs):
                with output_area:
                    clear_output(wait=True)
                    try:
                        self._render_unified_plot(self._plot_data_list, kwargs)
                    except Exception as e:
                        debug_log_spectroscopy_plot(f"Plot rendering failed: {e}")
                        raise

            self.update_function = update_plot

            # CRITICAL FIX: Connect widgets to update function via manual observers
            # This is the missing piece that makes widget changes trigger plot updates in review_fit
            interactive_widgets = {
                k: v for k, v in self.widgets.items() if hasattr(v, "value")
            }

            # Manually set up observers on each widget to call update_plot
            def create_widget_observer():
                def widget_change_handler(*args, **kwargs):
                    # Get current values from all widgets and call update_plot
                    current_values = {
                        k: w.value for k, w in interactive_widgets.items()
                    }
                    update_plot(**current_values)

                return widget_change_handler

            # Connect each widget to the observer
            widget_observer = create_widget_observer()
            for widget_name, widget in interactive_widgets.items():
                widget.observe(widget_observer, names="value")

            debug_log_spectroscopy_plot("Widget observers connected to update function")

            # Enhanced update plot handle with review_fit compatibility
            update_plot.ax_img = None  # Will be set during rendering
            update_plot.ax_spec = None  # Will be set during rendering
            update_plot.fig = None  # Will be set during rendering
            update_plot.current_plot_dims = None  # For lasso selection
            update_plot.current_extent = None  # For coordinate mapping
            update_plot.current_state = None  # For voxel selection state

            # Interactive output wrapper with enhanced state management
            def run_interactive_update(**kwargs):

                # Store current overlay masks for review_fit lasso functionality
                if hasattr(self, "interactive_overlays"):
                    update_plot.overlay_masks_data = self.interactive_overlays

                # Get current widget values and store as current_state
                current_widget_values = {}
                for name, widget in self.widgets.items():
                    if hasattr(widget, "value"):
                        current_widget_values[name] = widget.value

                # Store current selector state for review_fit
                update_plot.current_selector = current_widget_values

                # Create current_state tuple compatible with review_fit expectations
                # Format: (plot_data, voxel_indices)
                voxel_indices = {}
                for key, value in current_widget_values.items():
                    if key in [
                        "read_idx",
                        "phase_idx",
                        "slice_idx",
                        "repetition_idx",
                        "channel_idx",
                    ]:
                        voxel_indices[key] = value

                update_plot.current_state = (
                    self._plot_data_list[0] if self._plot_data_list else None,
                    voxel_indices,
                )

                # Call the actual update
                update_plot(**kwargs)

            # Initial widget values
            initial_kwargs = {
                name: (w.value if hasattr(w, "value") else w)
                for name, w in self.widgets.items()
                if hasattr(w, "value")
            }

            # Set up figure for review_fit compatibility
            self._setup_unified_figure(self._plot_data_list)

            # Store figure reference in update_plot for review_fit
            if hasattr(self, "update_function") and hasattr(
                self.update_function, "fig"
            ):
                update_plot.fig = self.update_function.fig

            # Create the 6-element tuple expected by review_fit
            result_tuple = (
                controls_box,  # plot_controls
                output_area,  # plot_output_area
                run_interactive_update,  # run_interactive_update
                initial_kwargs,  # initial_kwargs
                update_plot,  # update_plot_handle
                self.widgets,  # widgets_dict (the missing 6th element!)
            )

            return result_tuple

        except Exception as e:
            debug_log_spectroscopy_plot(f"_plot_with_handles failed: {e}")
            import traceback

            debug_log_spectroscopy_plot(f"Traceback: {traceback.format_exc()}")
            raise

    def _ensure_review_fit_widgets(self, widgets_dict, plot_data_list):
        """Ensure widgets have the names expected by review_fit.

        The review_fit function expects specific widget names that may differ
        from the default UnifiedPlotter widget names. This method creates
        aliases or renames widgets to match review_fit expectations.
        """

        # Check if we have fitted data (Datasets with metabolites)
        has_fitted_data = any(
            isinstance(pd.fitted_spectra.get("_dataset"), xr.Dataset)
            and "metabolite" in pd.fitted_spectra.get("_dataset", xr.Dataset()).coords
            for pd in plot_data_list
            if hasattr(pd, "fitted_spectra") and pd.fitted_spectra
        )

        if not has_fitted_data:
            return widgets_dict

        # Find the fitted data and extract metabolites
        metabolites = []
        for pd in plot_data_list:
            if hasattr(pd, "fitted_spectra") and pd.fitted_spectra:
                dataset = pd.fitted_spectra.get("_dataset")
                if isinstance(dataset, xr.Dataset) and "metabolite" in dataset.coords:
                    metabolites = [str(m) for m in dataset.coords["metabolite"].values]
                    break

        if not metabolites:
            return widgets_dict

        # Create/rename the overlay_selected_metabolites widget expected by review_fit
        # Check if a metabolite widget already exists under a different name
        metabolite_widget = None
        for widget_name, widget in widgets_dict.items():
            if (
                isinstance(widget, Dropdown)
                and hasattr(widget, "options")
                and widget.options
                and all(
                    str(opt) in metabolites or str(opt) == metabolites[0]
                    for opt in widget.options
                )
            ):
                metabolite_widget = widget
                break

        if metabolite_widget is None:
            # Create new metabolite selection widget with compact layout
            compact_layout = Layout(width="auto", max_width="200px")
            metabolite_widget = Dropdown(
                options=metabolites,
                value=metabolites[0],
                description="Overlay Mets:",
                disabled=False,
                layout=compact_layout,
            )

        # Add/update the widget with the name expected by review_fit
        widgets_dict["overlay_selected_metabolites"] = metabolite_widget

        return widgets_dict

    def _ensure_spatial_sliders_exist(self, plot_data_list, slider_defaults):
        """Ensure sliders exist for all spatial and spectral dimensions in spectroscopic data."""

        # Include all dimensions that should have sliders for spectroscopy data
        spatial_dims = [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "acquisition_idx",
            "repetition_idx",  # Added missing repetition_idx
            "channel_idx",  # Added channel_idx for completeness
            "inversion_idx",  # Added inversion_idx for completeness
        ]

        for i, plot_data in enumerate(plot_data_list):
            # For spectroscopy data, check both spatial_data and spectral_data for spatial dimensions
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
                        else:
                            pass
            else:
                pass

    def _apply_preprocessing(self, data_array, kwargs, widget_prefix):
        """Apply preprocessing using DataPreprocessor (inherited from UnifiedPlotter).

        Preprocessing delegation is now handled in UnifiedPlotter via DataPreprocessor,
        which includes:
        - Dimension standardization
        - Denoising
        - K-space transforms
        - Line broadening (spectroscopy-specific)
        - Repetition averaging
        - Domain transforms
        - Channel combination (spec_part and channel logic now in DataPreprocessor)
        - Data part transformations (Magnitude/Phase/Real/Imaginary)

        This eliminates duplication while preserving all spectroscopy functionality.
        """
        # Delegate to parent (UnifiedPlotter) which handles DataPreprocessor delegation
        return super()._apply_preprocessing(data_array, kwargs, widget_prefix)

    def _render_spatial_component(self, plot_data, is_base, kwargs):
        """Render the spatial component for CSI/MRSI data.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data containing spatial and spectral information.
        is_base : bool
            ``True`` when rendering the primary layer rather than an overlay.
        kwargs : dict
            Widget values and rendering parameters.

        Notes
        -----
        Supports metabolite-specific spatial maps, click-driven voxel selection,
        and reuses cached transforms for improved performance.
        """
        # Identify spatial dimensions (2D dimensions for display)
        spatial_dims = []
        non_spatial_selectors = {}

        # Get the current data to plot (may be updated by widget changes)
        spatial_data = plot_data.spatial_data

        # For Dataset plotting, apply variable selection first
        if plot_data.source_type == "dataset":
            dataset = plot_data.fitted_spectra.get("_dataset")
            if dataset:
                data_var_key = f"{plot_data.widget_prefix}_data_var"
                selected_var = kwargs.get(data_var_key, "fit_amps")

                if selected_var in dataset.data_vars:
                    # Use the selected variable for spatial display
                    spatial_data = np.abs(
                        dataset[selected_var]
                    )  # Take magnitude for display

        # Use automatic perspective detection based on widget interactions
        spatial_dims = self._get_spatial_dims_using_selector(
            spatial_data, kwargs, plot_data.widget_prefix
        )

        # Create selectors for non-spatial dimensions
        for key, value in kwargs.items():
            if (
                key in spatial_data.dims
                and key not in spatial_dims
                and key not in ["acquisition_idx", "metabolite"]
            ):
                non_spatial_selectors[key] = value

        # Handle metabolite selection for datasets
        metabolite_key = f"{plot_data.widget_prefix}_metabolite"
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
            except Exception:
                pass

        # Apply index-based selections
        if isel_selectors:
            # Filter to only include dimensions that actually exist AND have valid indices
            # Prevents IndexError when overlaying arrays with different dimension sizes
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

        # Apply data part transformation using cached transform (inherited from UnifiedPlotter)
        spec_part_val = kwargs.get(f"{plot_data.widget_prefix}_spec_part", "Magnitude")

        plot_data_values = self._get_cached_transform(data_slice, spec_part_val)

        # Use DataSelector for 2D preparation to avoid manual dimension juggling
        if hasattr(self, "data_selector") and self.data_selector:
            try:
                # For overlays, use base spatial dims for alignment
                if plot_data.is_overlay and hasattr(self, "_base_spatial_dims"):
                    base_dims = self._base_spatial_dims
                else:
                    base_dims = None
                    if not plot_data.is_overlay:
                        self._base_spatial_dims = spatial_dims

                # Prepare 2D data with correct orientation (delegates complex logic)
                plot_data_values = self.data_selector.prepare_2d_spatial_data(
                    plot_data_values,
                    spatial_dims=spatial_dims,
                    is_overlay=plot_data.is_overlay,
                    base_spatial_dims=base_dims,
                )
            except Exception as e:
                logger.debug(
                    f"DataSelector.prepare_2d_spatial_data failed: {e}, using fallback"
                )
                # Fallback: simple dimension reduction
                while plot_data_values.ndim > 2:
                    plot_data_values = plot_data_values.isel(
                        {plot_data_values.dims[0]: 0}
                    )
        else:
            # Fallback for instances without DataSelector
            if plot_data_values.ndim > 2:
                while plot_data_values.ndim > 2:
                    plot_data_values = plot_data_values.isel(
                        {plot_data_values.dims[0]: 0}
                    )
            elif plot_data_values.ndim == 1:
                plot_data_values = plot_data_values.values.reshape(1, -1)
            elif plot_data_values.ndim == 0:
                plot_data_values = np.array([[plot_data_values.item()]])

        # Convert to numpy if needed
        if hasattr(plot_data_values, "values"):
            plot_data_values = plot_data_values.values

        # Handle subtraction if requested
        subtract_mode = kwargs.get("subtract_overlay", False)
        percent_mode = kwargs.get("percent_difference", False)

        if not is_base and (subtract_mode or percent_mode):
            # Retrieve stored base data
            if hasattr(self, "_cached_base_spatial_data"):
                plot_data_values = self._compute_overlay_difference(
                    plot_data_values,
                    self._cached_base_spatial_data,
                    percent=percent_mode,
                )
        elif is_base:
            # Store base data for potential subtraction
            self._cached_base_spatial_data = plot_data_values.copy()

        # Determine colormap and alpha
        cmap = kwargs.get(
            f"{plot_data.widget_prefix}_cmap", self.processed_plot_params["colormap"]
        )

        # When subtracting, use full opacity and different colormap default
        if not is_base and (subtract_mode or percent_mode):
            alpha = 1.0
            if percent_mode:
                cmap = kwargs.get(f"{plot_data.widget_prefix}_cmap", "RdBu_r")
        else:
            alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)

        # Calculate intensity range using cached min/max (inherited from UnifiedPlotter)
        min_sig, max_sig = self._get_cached_minmax(spatial_data, spec_part_val)

        interpolation = kwargs.get(
            f"{plot_data.widget_prefix}_interpolate_image",
            False,
        )
        if interpolation:
            interpolation_str = "spline36"
        else:
            interpolation_str = "nearest"

        vmin, vmax = self._vmin_vmax_range_check(
            range_mode=kwargs.get(
                f"{plot_data.widget_prefix}_range_mode", "Percentage"
            ),
            data_to_plot_spatial=plot_data_values,
            min_sig=min_sig,
            max_sig=max_sig,
            image_source_changed=True,
            plot_data=plot_data,
        )

        # Calculate extent for proper click handling and tick labels
        ny, nx = plot_data_values.shape[-2:]
        # Use automatic perspective detection provided by the DataSelector helper
        spatial_dims = self._get_spatial_dims_using_selector(
            spatial_data, kwargs, plot_data.widget_prefix
        )
        coord_system = kwargs.get("coord_system", "Auto")
        extent, origin, xlabel, ylabel = self._calculate_coordinate_extent(
            spatial_data,
            spatial_dims,
            plot_data_values.shape,
            coord_system=coord_system,
            data_obj=self.spectroscopy_obj,
            kwargs=kwargs,
        )

        # Plot the image
        im = self.update_function.ax_img.imshow(
            plot_data_values,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation_str,
            extent=extent,
            origin=origin,
        )

        # Store vmin/vmax for secondary plot sync if requested
        sync_enabled = kwargs.get(f"{plot_data.widget_prefix}_sync_y_range", False)
        if sync_enabled:
            self.update_function._synced_ylim = (vmin, vmax)
        elif is_base:
            # Only base data can UNSET the sync if it was set by base
            # This is a bit simplified but usually what's desired
            if hasattr(self.update_function, "_synced_ylim"):
                self.update_function._synced_ylim = None

        # Store click handling information for DataArray plots
        if is_base:
            self.update_function.last_im = im
            # Set up click handling info
            if len(spatial_dims) >= 2:
                self.update_function._current_plot_dims = (
                    spatial_dims[0],
                    spatial_dims[1],
                )
                self.update_function._current_data_shape = (ny, nx)
                self.update_function._current_extent = extent
                self.update_function._current_origin = origin
            else:
                print(
                    f"DEBUG: Not enough spatial dims for click handling: {spatial_dims}"
                )

            # Set axis labels
            self.update_function.ax_img.set_xlabel(xlabel)
            self.update_function.ax_img.set_ylabel(ylabel)
        else:
            # Store overlay image for dual colorbar
            if not hasattr(self.update_function, "overlay_ims"):
                self.update_function.overlay_ims = []
            self.update_function.overlay_ims.append(im)

    def _render_fitted_spectrum(self, plot_data, is_base, kwargs):
        """Render spectrum from fitted data (Dataset)."""

        # Check display mode: True = sum only, False = individual peaks
        # Check for prefixed version first (e.g., ds1_overlay_display_mode for overlay data)
        overlay_mode_key = f"{plot_data.widget_prefix}_overlay_display_mode"
        show_sum_only = kwargs.get(
            overlay_mode_key, kwargs.get("overlay_display_mode", False)
        )

        # Get the fitted dataset for individual metabolite plotting
        dataset = plot_data.fitted_spectra.get("_dataset")
        if dataset is None:
            debug_log_spectroscopy_plot("[ERROR]: No dataset found in fitted_spectra")
            debug_log_spectroscopy_plot(
                f"[DEBUG]: fitted_spectra contents: {plot_data.fitted_spectra}"
            )
            return

        # Validate dataset structure for fitted spectrum rendering
        required_vars = ["fit_amps", "fit_freqs"]
        missing_vars = [var for var in required_vars if var not in dataset.data_vars]
        if missing_vars:
            debug_log_spectroscopy_plot(
                f"[ERROR]: Missing required variables in dataset: {missing_vars}"
            )
            debug_log_spectroscopy_plot(
                f"[DEBUG]: Available data_vars: {list(dataset.data_vars.keys())}"
            )
            debug_log_spectroscopy_plot(
                f"[DEBUG]: Available coords: {list(dataset.coords.keys())}"
            )
            return

        debug_log_spectroscopy_plot(
            f"Rendering fitted spectrum, is_base={is_base}, show_sum_only={show_sum_only}"
        )

        # For individual mode, always use the reconstruction path to plot metabolites separately
        if not show_sum_only:
            try:
                self._plot_reconstructed_spectrum(
                    self.update_function.ax_spec,
                    dataset,
                    kwargs,
                    self.processed_plot_params,
                    is_overlay=not is_base,
                    widget_prefix=plot_data.widget_prefix,
                )

            except Exception as e:
                debug_log_spectroscopy_plot(
                    f"[ERROR]: _plot_reconstructed_spectrum failed in individual mode: {e}"
                )
                debug_log_spectroscopy_plot(f"[DEBUG]: Dataset shape: {dataset.dims}")
                debug_log_spectroscopy_plot(
                    f"[DEBUG]: fit_amps shape: {dataset['fit_amps'].shape if 'fit_amps' in dataset else 'N/A'}"
                )
                debug_log_spectroscopy_plot(
                    f"[DEBUG]: kwargs keys: {list(kwargs.keys())}"
                )
                raise
        else:
            # For sum mode, always use reconstruction path to respect individual/sum toggle properly
            try:
                self._plot_reconstructed_spectrum(
                    self.update_function.ax_spec,
                    dataset,
                    kwargs,
                    self.processed_plot_params,
                    is_overlay=not is_base,
                    widget_prefix=plot_data.widget_prefix,
                )
            except Exception as e:
                debug_log_spectroscopy_plot(
                    f"[ERROR]: _plot_reconstructed_spectrum failed in sum mode: {e}"
                )
                debug_log_spectroscopy_plot(f"[DEBUG]: Dataset shape: {dataset.dims}")
                debug_log_spectroscopy_plot(
                    f"[DEBUG]: fit_amps shape: {dataset['fit_amps'].shape if 'fit_amps' in dataset else 'N/A'}"
                )
                debug_log_spectroscopy_plot(
                    f"[DEBUG]: kwargs keys: {list(kwargs.keys())}"
                )
                raise

        # Add fit results text if enabled
        self._add_fit_results_text(kwargs, is_base, plot_data.widget_prefix)

    def _render_measured_spectrum(self, plot_data, is_base, kwargs):
        """Render spectrum from measured data (DataArray)."""
        # Only require click position if data has spatial component
        if not self._has_click_position() and plot_data.has_spatial_data():
            # with open("/tmp/spectroscopy_plot_debug.log", "a") as f:
            #     f.write(
            #         "DEBUG: No click position set for spatial data - showing message on axis\n"
            #     )
            if hasattr(self, "update_function") and hasattr(
                self.update_function, "ax_spec"
            ):
                self.update_function.ax_spec.text(
                    0.5,
                    0.5,
                    "Click spatial plot for Spectrum/FID",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.update_function.ax_spec.transAxes,
                )
            return

        # Select voxel signal if has spatial data, otherwise use entire
        # spectrum
        if plot_data.has_spatial_data():
            # with open("/tmp/spectroscopy_plot_debug.log", "a") as f:
            # f.write("DEBUG: Selecting voxel signal for spatial data\n")
            signal_1d = self._select_voxel_signal_unified(
                plot_data, kwargs, self._get_click_position()
            )
        else:
            # with open("/tmp/spectroscopy_plot_debug.log", "a") as f:
            # f.write("DEBUG: Using spectral data directly for non-spatial data\n")
            # For purely spectral data, apply dimension selections
            signal_1d = plot_data.spectral_data

            # Apply dimension selectors for non-spatial spectroscopic data
            selectors = {}
            for key, value in kwargs.items():
                if (
                    key in signal_1d.dims
                    and key
                    not in ["acquisition_idx"]  # Don't select acquisition dimension
                    and key
                    != f"{plot_data.widget_prefix}_spec_part"  # Skip widget controls
                    and not key.startswith(f"{plot_data.widget_prefix}_")
                    or key in ["repetition_idx", "channel_idx", "inversion_idx"]
                ):  # Include dimension selectors
                    selectors[key] = value

            # Apply selectors if any exist
            if selectors:
                # Filter to only include dimensions that actually exist AND have valid indices
                # Prevents IndexError when overlaying arrays with different dimension sizes
                valid_selectors = {
                    k: v
                    for k, v in selectors.items()
                    if k in signal_1d.dims and v < signal_1d.sizes[k]
                }
                if valid_selectors:
                    signal_1d = signal_1d.isel(**valid_selectors)

        if signal_1d is None:
            # with open("/tmp/spectroscopy_plot_debug.log", "a") as f:
            # f.write("DEBUG: signal_1d is None, returning without plotting\n")
            return

        # Plot 1D signal using existing logic
        self._plot_1d_signal_unified(plot_data, signal_1d, is_base, kwargs)

    def _reconstruct_fitted_spectra_for_preprocessing(
        self, dataset, kwargs, widget_prefix
    ):
        """Reconstruct fitted spectra early for consistent preprocessing.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset produced by ``fit_data_pseudo_inv``.
        kwargs : dict
            Widget parameters governing reconstruction.
        widget_prefix : str
            Prefix used for widget names.

        Returns
        -------
        xr.DataArray or None
            Reconstructed total spectrum suitable for preprocessing.
        """
        try:
            # Get selectors for reconstruction
            isel_selector, sel_selector = self._get_reconstruction_selectors(
                dataset, kwargs, False, widget_prefix
            )

            if isel_selector is None:
                # If no click position, use center voxel as default
                spatial_dims = [
                    dim
                    for dim in dataset.dims
                    if dim in ["read_idx", "phase_idx", "slice_idx"]
                ]
                isel_selector = {}
                for dim in spatial_dims:
                    if dim in dataset.dims:
                        isel_selector[dim] = dataset.sizes[dim] // 2

            # Always reconstruct total spectrum for preprocessing
            # The individual/sum logic will be handled during rendering

            # Reconstruct single voxel spectrum using cache (spectroscopy-specific)
            total_fid = self._get_cached_reconstruction(dataset, kwargs, widget_prefix)

            if total_fid is None:
                return None

            # Handle the case where total_fid might be a DataArray or have multiple dimensions
            if hasattr(total_fid, "values"):
                # It's a DataArray, extract values and flatten if needed
                total_fid_values = total_fid.values
            else:
                # It's already a numpy array
                total_fid_values = total_fid

            # Create DataArray with proper attributes for preprocessing
            # Reuse metabolite coordinates from original dataset
            metabolites = dataset.coords.get("metabolite", [])

            # Reuse acquisition_idx coordinates from original dataset if available
            if "acquisition_idx" in dataset.coords:
                acquisition_coords = dataset.coords["acquisition_idx"]
            else:
                # Fallback to range if no acquisition_idx coordinate exists
                try:
                    acquisition_coords = dataset.attrs["sampling_time_axis_s"]
                except:
                    acquisition_coords = np.arange(total_fid_values.shape[-1])

            reconstructed_data = xr.DataArray(
                total_fid_values,
                dims=["metabolite", "acquisition_idx"],
                coords={
                    "metabolite": metabolites,
                    "acquisition_idx": acquisition_coords,
                },
                attrs={
                    "signal_domain": "temporal",
                    "is_spectroscopic": True,
                    **dataset.attrs,
                },
            )
            return reconstructed_data

        except Exception as e:
            debug_log_spectroscopy_plot(
                f"ERROR in _reconstruct_fitted_spectra_for_preprocessing: {e}"
            )
            debug_log_spectroscopy_plot(f"Exception type: {type(e)}")
            import traceback

            debug_log_spectroscopy_plot(f"Full traceback: {traceback.format_exc()}")
            return None

    def _reconstruct_single_voxel_spectrum(
        self, dataset, isel_selector, sel_selector, kwargs, widget_prefix
    ):
        """Reconstruct total spectrum from a fit dataset.

        This method uses the `hp.reconstruct` accessor on the provided dataset
        to generate the FID for each metabolite at a specific voxel, then sums
        them to produce the total FID.

        Parameters
        ----------
        dataset : xr.Dataset
            The input xarray Dataset containing the fit results. It must have
            the `hp` accessor registered and contain variables like `fit_amps`,
            `fit_freqs`, etc.
        isel_selector : dict
            Dictionary of integer-based selectors for spatial and other
            dimensions (e.g., `{'read_idx': 10, 'phase_idx': 10}`).
        sel_selector : dict
            Dictionary of label-based selectors (e.g., for metabolites).
        kwargs : dict
            Dictionary of widget parameters, used to get `npoints`.
        widget_prefix : str
            The prefix for widgets associated with this plot data, used to
            access the correct `npoints` widget from `kwargs`.

        Returns
        -------
        np.ndarray or None
            The reconstructed total FID as a complex numpy array. Returns
            `None` if reconstruction fails.
        """
        try:
            # Get number of points for reconstruction (fall back to full FID length)
            default_points = self.spectroscopy_obj.n_acquisition - getattr(
                self.spectroscopy_obj, "cut_off", 0
            )
            requested_npoints = kwargs.get(f"{widget_prefix}_npoints", default_points)

            # Prepare dataset copy with sampling time axis/metadata adjustments
            dataset_copy = dataset.copy()

            # Ensure voxel coordinates cover all loop dimensions so the
            # reconstructed FID collapses to a single spectrum. When the
            # UI has not yet provided selectors (e.g. first render), fall
            # back to the first index along each remaining dimension such as
            # repetition_idx or channel_idx.
            full_isel_selector = dict(isel_selector or {})
            for dim in dataset_copy.dims:
                if dim in ("metabolite", "acquisition_idx"):
                    continue
                if dim not in full_isel_selector:
                    full_isel_selector[dim] = 0

            axis_override = self._get_reconstruction_axis_override(
                dataset_copy, full_isel_selector, widget_prefix, kwargs
            )

            if axis_override and "axis_da" in axis_override:
                axis_da = axis_override["axis_da"]
                axis_dim = axis_da.dims[-1] if axis_da.dims else "inversion_idx"
                effective_npoints = axis_da.sizes.get(axis_dim, len(axis_da))
                sampling_time_axis_s = None
            else:
                sampling_axis_full = dataset.attrs.get("sampling_time_axis_s")
                if hasattr(sampling_axis_full, "values"):
                    sampling_axis_full = sampling_axis_full.values
                if sampling_axis_full is not None:
                    sampling_axis_full = np.asarray(sampling_axis_full)
                    available_points = sampling_axis_full.shape[0]
                else:
                    sampling_axis_full = calc_sampling_time_axis(
                        self.spectroscopy_obj, npoints=default_points
                    )
                    available_points = sampling_axis_full.shape[0]

                if requested_npoints is None or requested_npoints <= 0:
                    effective_npoints = available_points
                else:
                    effective_npoints = int(min(available_points, requested_npoints))

                sampling_time_axis_s = calc_sampling_time_axis(
                    dwell_time_s=dataset.dwell_time_s, npoints=10 * effective_npoints
                )

                sampling_time_axis_s = sampling_axis_full[:effective_npoints]
                dataset_copy.attrs["sampling_time_axis_s"] = sampling_time_axis_s
                dataset_copy.attrs["nsamplepoints"] = effective_npoints
                if effective_npoints > 1:
                    dataset_copy.attrs.setdefault(
                        "dwell_time_s",
                        sampling_time_axis_s[1] - sampling_time_axis_s[0],
                    )

            metabolites = dataset.coords.get("metabolite", [])
            total_fid = np.zeros(
                (len(metabolites), effective_npoints), dtype=np.complex128
            )
            reconstruct_kwargs = axis_override["kwargs"] if axis_override else {}

            # Reconstruct spectrum for each metabolite and sum
            for m, metab in enumerate(metabolites):
                try:

                    if hasattr(dataset_copy, "hp"):

                        metab_fid = (
                            dataset_copy.hp.reconstruct(
                                voxel_coords=full_isel_selector, **reconstruct_kwargs
                            )
                            .sel(metabolite=metab)
                            .squeeze()
                        )
                        # If any loop dimensions remain (e.g. repetition_idx)
                        # after squeezing, take the first entry to obtain a
                        # 1D spectrum for overlay.
                        for extra_dim in list(metab_fid.dims):
                            if extra_dim not in ("acquisition_idx",):
                                metab_fid = metab_fid.isel({extra_dim: 0}).squeeze()

                        if "acquisition_idx" in metab_fid.dims:
                            metab_fid = metab_fid.isel(
                                acquisition_idx=slice(0, effective_npoints)
                            )

                        metab_values = np.asarray(metab_fid).reshape(-1)
                        if metab_values.size > effective_npoints:
                            metab_values = metab_values[:effective_npoints]
                        elif metab_values.size < effective_npoints:
                            padded = np.zeros(
                                effective_npoints, dtype=metab_values.dtype
                            )
                            padded[: metab_values.size] = metab_values
                            metab_values = padded

                        total_fid[m, :] = metab_values
                    else:
                        debug_log_spectroscopy_plot(
                            "hp accessor missing on dataset_copy – attempting manual reconstruction"
                        )
                except (ValueError, IndexError) as e:
                    debug_log_spectroscopy_plot(
                        f"ValueError/IndexError for {metab} at {isel_selector}: {e}"
                    )
                    print(
                        f"Warning: Could not reconstruct {metab} at {isel_selector}: {e}"
                    )
                    continue
                except AttributeError as e:
                    debug_log_spectroscopy_plot(f"AttributeError for {metab}: {e}")
                    print(f"Warning: AttributeError for {metab}: {e}")
                    continue
            return (
                total_fid if isinstance(total_fid, np.ndarray) else np.array(total_fid)
            )

        except Exception as e:
            debug_log_spectroscopy_plot(f"ERROR in _reconstruct_total_spectrum: {e}")
            debug_log_spectroscopy_plot(f"Exception type: {type(e)}")
            import traceback

            debug_log_spectroscopy_plot(f"Full traceback: {traceback.format_exc()}")
            print(f"Warning: Failed to reconstruct total spectrum: {e}")
            return None

    def _collapse_to_acquisition_axis(self, data_array, npoints):
        """Ensure reconstructed signals are 1D along acquisition_idx."""

        if data_array is None:
            return None

        if isinstance(data_array, xr.DataArray):
            collapsed = data_array
            for dim in list(collapsed.dims):
                if dim != "acquisition_idx":
                    try:
                        collapsed = collapsed.isel({dim: 0}).squeeze(drop=True)
                    except Exception:
                        collapsed = collapsed.isel({dim: 0}).squeeze()

            if "acquisition_idx" not in collapsed.dims:
                values = np.asarray(collapsed).reshape(-1)
                collapsed = xr.DataArray(
                    values,
                    dims=["acquisition_idx"],
                    coords={"acquisition_idx": np.arange(values.size)},
                )
        else:
            values = np.asarray(data_array).reshape(-1)
            collapsed = xr.DataArray(
                values,
                dims=["acquisition_idx"],
                coords={"acquisition_idx": np.arange(values.size)},
            )

        if "acquisition_idx" in collapsed.dims and npoints is not None:
            collapsed = collapsed.isel(
                acquisition_idx=slice(
                    0, min(npoints, collapsed.sizes["acquisition_idx"])
                )
            )

        return collapsed

    def _get_reconstruction_selectors(
        self, fit_dataset, kwargs, is_overlay, widget_prefix
    ):
        """
        Get the appropriate selectors for spectrum reconstruction based on context.

        Parameters
        ----------
        fit_dataset : xr.Dataset
            Dataset containing fitted spectroscopy results.
        kwargs : dict
            Widget parameters influencing selection.
        is_overlay : bool
            ``True`` when reconstructing an overlay dataset.
        widget_prefix : str
            Prefix used for widget names.

        Returns
        -------
        tuple
            ``(isel_selector, sel_selector)`` when a valid position exists,
            otherwise ``(None, None)``.
        """

        if is_overlay:
            spatial_dims = [
                d
                for d in fit_dataset.dims
                if d in ["read_idx", "phase_idx", "slice_idx"]
            ]

            if not self._has_click_position():  # Use parent class method
                # Check if this is pure spectroscopy data (no meaningful spatial component)
                # Check if spatial dimensions have meaningful size (> 1)
                is_pure_spectroscopy = True
                if spatial_dims:
                    spatial_sizes = [
                        fit_dataset.sizes.get(dim, 1) for dim in spatial_dims
                    ]
                    is_pure_spectroscopy = all(size <= 1 for size in spatial_sizes)

                # Initialize selectors - use slider values from kwargs if available
                isel_selector = {}
                for dim in spatial_dims:
                    if dim in fit_dataset.dims:
                        # Use slider value from kwargs if available, otherwise default to 0
                        isel_selector[dim] = kwargs.get(dim, 0)
            else:
                isel_selector = (
                    self._get_click_position().copy()
                )  # Use parent class method

            # Update spatial selectors with current slider values from kwargs
            # This ensures slice_idx and other spatial sliders update the fitted spectrum
            for dim in spatial_dims:
                if dim in kwargs and dim in fit_dataset.dims:
                    isel_selector[dim] = kwargs[dim]

            # Handle other (non-spatial) dimensions
            other_dims = [
                d
                for d in fit_dataset.dims
                if d
                not in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "metabolite",
                    "acquisition_idx",
                ]
            ]
            for dim in other_dims:
                if dim in kwargs:
                    isel_selector[dim] = kwargs[dim]

            sel_selector = {}
            metabolite_key = f"{widget_prefix}_metabolite"
            if metabolite_key in kwargs and "metabolite" in fit_dataset.dims:
                sel_selector["metabolite"] = kwargs[metabolite_key]
        else:  # Standalone fit plot
            spatial_dims = [
                dim
                for dim in fit_dataset.dims
                if dim in ["read_idx", "phase_idx", "slice_idx"]
            ]

            if not self._has_click_position():
                # Check if this is pure spectroscopy data (no meaningful spatial component)
                # Check if spatial dimensions have meaningful size (> 1)
                is_pure_spectroscopy = True
                if spatial_dims:
                    spatial_sizes = [
                        fit_dataset.sizes.get(dim, 1) for dim in spatial_dims
                    ]
                    is_pure_spectroscopy = all(size <= 1 for size in spatial_sizes)

                # Initialize selectors - use slider values from kwargs if available
                isel_selector = {}
                # Set defaults for all non-metabolite, non-acquisition dimensions
                for dim in fit_dataset.dims:
                    if dim not in ["metabolite", "acquisition_idx"]:
                        # Use slider value from kwargs if available, otherwise default to 0
                        isel_selector[dim] = kwargs.get(dim, 0)
            else:
                isel_selector = self._get_click_position()

            # Update spatial selectors with current slider values from kwargs
            # This ensures slice_idx and other spatial sliders update the fitted spectrum
            for dim in spatial_dims:
                if dim in kwargs and dim in fit_dataset.dims:
                    isel_selector[dim] = kwargs[dim]

            sel_selector = {}
            for key, value in kwargs.items():
                if key == "data_var":
                    continue
                if key in fit_dataset.dims:
                    if fit_dataset.coords[key].dtype == object or key == "metabolite":
                        sel_selector[key] = value
                    else:
                        if key not in isel_selector:
                            isel_selector[key] = value

        # Filter selectors with bounds checking
        # Prevents IndexError when overlaying arrays with different dimension sizes
        isel_selector = {
            k: v
            for k, v in isel_selector.items()
            if k in fit_dataset.dims and v < fit_dataset.sizes[k]
        }
        sel_selector = {k: v for k, v in sel_selector.items() if k in fit_dataset.dims}

        return isel_selector, sel_selector

    def _get_reconstruction_plot_params(self, kwargs, is_overlay, widget_prefix):
        """
        Extract plot parameters for spectrum reconstruction.

        Parameters
        ----------
        kwargs : dict
            Widget parameters influencing reconstruction.
        is_overlay : bool
            ``True`` when reconstructing an overlay dataset.
        widget_prefix : str
            Prefix used for widget names.

        Returns
        -------
        dict
            Dictionary containing all relevant plot parameters.
        """
        spec_part_key = f"{widget_prefix}_spec_part" if is_overlay else "spec_part"
        spec_part_val = kwargs.get(spec_part_key, "Magnitude")
        part_func = {
            "Magnitude": np.abs,
            "Real": np.real,
            "Imaginary": np.imag,
            "Phase": np.angle,
        }.get(spec_part_val, np.abs)

        spec_unit_key = f"{widget_prefix}_spec_unit" if is_overlay else "spec_unit"
        spec_unit = "ppm" if kwargs.get(spec_unit_key, True) else "Hz"

        npoints_key = f"{widget_prefix}_npoints" if is_overlay else "npoints"
        cut_off = getattr(self.spectroscopy_obj, "cut_off", 0)
        n_acq = getattr(self.spectroscopy_obj, "n_acquisition", None)
        if n_acq is None:
            base_signal = getattr(self.spectroscopy_obj, "reco_data", None)
            if (
                isinstance(base_signal, xr.DataArray)
                and "acquisition_idx" in base_signal.dims
            ):
                n_acq = base_signal.sizes["acquisition_idx"]
        default_npoints = None
        if n_acq is not None:
            default_npoints = max(0, n_acq - cut_off)
        if not default_npoints:
            time_axis = getattr(self.spectroscopy_obj, "sampling_time_axis_s", None)
            if isinstance(time_axis, (list, tuple, np.ndarray)):
                default_npoints = len(time_axis)
            else:
                default_npoints = kwargs.get(npoints_key, 0) or 0
        npoints = kwargs.get(npoints_key, default_npoints)

        view_toggle_key = (
            f"{widget_prefix}_view_toggle" if is_overlay else "view_toggle"
        )
        show_temporal = kwargs.get(view_toggle_key, False)

        return {
            "spec_part_val": spec_part_val,
            "part_func": part_func,
            "spec_unit": spec_unit,
            "npoints": npoints,
            "show_temporal": show_temporal,
        }

    def _reconstruct_metabolite_spectra(
        self,
        fit_dataset,
        isel_selector,
        sel_selector,
        plot_params,
        ax,
        is_overlay,
        kwargs,
        processed_plot_params,
        widget_prefix="",
    ):
        """
        Reconstruct and plot individual metabolite spectra.

        Parameters
        ----------
        fit_dataset : xr.Dataset
            Dataset containing fitted spectroscopy results.
        isel_selector : dict
            Index selectors for positional dimensions.
        sel_selector : dict
            Coordinate selectors for labelled dimensions.
        plot_params : dict
            Plot parameters derived from widget state.
        ax : matplotlib.axes.Axes
            Axes used for plotting.
        is_overlay : bool
            ``True`` when plotting as an overlay.
        kwargs : dict
            Widget parameters forwarded for reconstruction.
        processed_plot_params : dict
            Normalized plotting parameters with defaults applied.

        Returns
        -------
        tuple
            ``(fit_dataset_copy, total_fid)`` containing the dataset copy and
            reconstructed total FID.
        """

        # Prepare dataset copy with sampling time axis metadata
        fit_dataset_copy = fit_dataset.copy()

        if "sampling_time_axis_s" in fit_dataset.attrs:
            sampling_time_axis_s = np.asarray(
                fit_dataset.attrs["sampling_time_axis_s"]
            )[: plot_params["npoints"]]
        else:
            sampling_time_axis_s = calc_sampling_time_axis(
                self.spectroscopy_obj, npoints=plot_params["npoints"]
            )
        fit_dataset_copy.attrs.setdefault("sampling_time_axis_s", sampling_time_axis_s)

        # Ensure voxel selector covers all loop dimensions so each reconstruction
        # collapses to a single spectrum.
        full_isel_selector = dict(isel_selector or {})
        for dim in fit_dataset_copy.dims:
            if dim in ("metabolite", "acquisition_idx"):
                continue
            full_isel_selector.setdefault(dim, 0)

        # Initialise accumulator lazily once we have a first valid metabolite
        total_fid = None
        selected_metab = sel_selector.get("metabolite", None)
        metabolites = fit_dataset.coords.get("metabolite", [])

        # Check display mode: True = sum only, False = individual peaks
        # Check for prefixed version first (e.g., ds1_overlay_display_mode for overlay data)
        overlay_mode_key = (
            f"{widget_prefix}_overlay_display_mode"
            if widget_prefix
            else "overlay_display_mode"
        )
        show_sum_only = kwargs.get(
            overlay_mode_key, kwargs.get("overlay_display_mode", False)
        )

        # Set up colors
        if is_overlay:
            colors = processed_plot_params.get("overlay_linecolormap", None)
            if colors is None:
                linecolors = plt.cm.viridis(np.linspace(0, 1, len(metabolites)))
            else:
                linecolors = [colors(i) for i in np.linspace(0, 1, len(metabolites))]
        else:
            linecolors = plt.cm.viridis(np.linspace(0, 1, len(metabolites)))

        if len(metabolites) < 8:
            linecolors = colors = [
                "#0C5DA5",
                "#00B945",
                "#FF9500",
                "#FF2C00",
                "#845B97",
                "#474747",
                "#9e9e9e",
            ]

        # Plot each metabolite (only if not showing sum only)
        if not show_sum_only:
            # Check if subtraction mode is active
            subtract_mode = kwargs.get("subtract_overlay", False)
            percent_mode = kwargs.get("percent_difference", False)

            for m, metab in enumerate(metabolites):
                # In subtraction/percent mode with overlay, only plot the selected metabolite
                if is_overlay and (subtract_mode or percent_mode):
                    if selected_metab is not None and metab != selected_metab:
                        # Skip non-selected metabolites in subtraction mode
                        # But still accumulate to total_fid for sum calculation
                        try:
                            if hasattr(fit_dataset_copy, "hp"):
                                metab_fid = (
                                    fit_dataset_copy.hp.reconstruct(
                                        voxel_coords=full_isel_selector
                                    )
                                    .sel(metabolite=metab)
                                    .squeeze()
                                )
                                metab_1d = self._collapse_to_acquisition_axis(
                                    metab_fid, plot_params["npoints"]
                                )
                                if metab_1d is not None:
                                    total_fid = (
                                        metab_1d.copy(deep=True)
                                        if total_fid is None
                                        else total_fid + metab_1d
                                    )
                        except:
                            pass
                        continue

                try:

                    if hasattr(fit_dataset_copy, "hp"):
                        debug_log_spectroscopy_plot(
                            f"Calling hp.reconstruct for {metab} with isel_selector: {full_isel_selector}"
                        )
                        metab_fid = (
                            fit_dataset_copy.hp.reconstruct(
                                voxel_coords=full_isel_selector
                            )
                            .sel(metabolite=metab)
                            .squeeze()
                        )
                    else:
                        debug_log_spectroscopy_plot(
                            f"ERROR: fit_dataset_copy missing hp accessor for {metab}"
                        )
                        raise AttributeError(
                            f"fit_dataset_copy object has no attribute 'hp'"
                        )
                except (ValueError, IndexError) as e:
                    debug_log_spectroscopy_plot(
                        f"ValueError/IndexError for {metab}: {e}"
                    )
                    continue
                except AttributeError as e:
                    debug_log_spectroscopy_plot(f"AttributeError for {metab}: {e}")
                    continue

                metab_1d = self._collapse_to_acquisition_axis(
                    metab_fid, plot_params["npoints"]
                )
                if metab_1d is None:
                    continue

                total_fid = (
                    metab_1d.copy(deep=True)
                    if total_fid is None
                    else total_fid + metab_1d
                )

                # Determine plot data and axis
                if plot_params["show_temporal"]:
                    plot_data = metab_1d.values
                    x_axis = sampling_time_axis_s[: len(plot_data)]
                else:
                    plot_data = np.fft.fftshift(np.fft.fft(metab_1d.values))
                    cut_off = getattr(self.spectroscopy_obj, "cut_off", 0)
                    x_axis = get_freq_axis(
                        self.spectroscopy_obj,
                        plot_params["spec_unit"],
                        cut_off,
                        len(plot_data),
                    )
                    if x_axis is None:
                        x_axis = np.arange(len(plot_data))

                # Plot styling
                linewidth = 3 if metab == selected_metab else 2
                alpha = (
                    kwargs.get("overlay_alpha", 1)
                    if is_overlay
                    else (1.0 if metab == selected_metab else 1)
                )
                label = self._get_clean_metabolite_label(metab, is_overlay)
                color = linecolors[m] if is_overlay else None

                plot_values = plot_params["part_func"](plot_data)

                # Handle subtraction for individual metabolites
                subtract_mode = kwargs.get("subtract_overlay", False)
                percent_mode = kwargs.get("percent_difference", False)

                if is_overlay and (subtract_mode or percent_mode):
                    # For individual metabolites, check if we have cached base data
                    # Priority: 1) metabolite-specific cache, 2) total measured signal cache
                    base_values = None
                    if (
                        hasattr(self, "_cached_base_metabolite_spectra")
                        and metab in self._cached_base_metabolite_spectra
                    ):
                        # Use cached base metabolite if available (fitted vs fitted comparison)
                        base_values = self._cached_base_metabolite_spectra[metab]
                    elif hasattr(self, "_cached_base_spectrum_1d"):
                        # Use total measured signal (fitted vs measured comparison)
                        base_values = self._cached_base_spectrum_1d
                    elif hasattr(self, "_cached_base_spectrum"):
                        # Use total fitted signal as fallback
                        base_values = self._cached_base_spectrum

                    if base_values is not None and len(plot_values) == len(base_values):
                        # For individual metabolites: compute Base - Selected Metabolite
                        # This shows what remains after removing the metabolite from the base signal
                        # Reverse the order compared to sum view (swap arguments)
                        plot_values = self._compute_overlay_difference(
                            base_values,  # Base first
                            plot_values,  # Metabolite second
                            percent=percent_mode,
                        )
                        if percent_mode:
                            label = f"Base - {self._get_clean_metabolite_label(metab, is_overlay)} (% Diff)"
                        else:
                            label = f"Base - {self._get_clean_metabolite_label(metab, is_overlay)}"
                elif not is_overlay:
                    # Store base metabolite spectrum for potential subtraction
                    if not hasattr(self, "_cached_base_metabolite_spectra"):
                        self._cached_base_metabolite_spectra = {}
                    self._cached_base_metabolite_spectra[metab] = plot_values.copy()

                ax.plot(
                    x_axis,
                    plot_values,
                    label=label,
                    linewidth=linewidth,
                    alpha=alpha,
                    color=color,
                )
                if len(x_axis) > 0:
                    ax.set_xlim(x_axis[0], x_axis[-1])
        else:
            # When showing sum only, still need to calculate total_fid
            for m, metab in enumerate(metabolites):
                try:
                    if hasattr(fit_dataset_copy, "hp"):
                        metab_fid = (
                            fit_dataset_copy.hp.reconstruct(
                                voxel_coords=full_isel_selector
                            )
                            .sel(metabolite=metab)
                            .squeeze()
                        )
                        metab_1d = self._collapse_to_acquisition_axis(
                            metab_fid, plot_params["npoints"]
                        )
                        if metab_1d is None:
                            continue
                        total_fid = (
                            metab_1d.copy(deep=True)
                            if total_fid is None
                            else total_fid + metab_1d
                        )
                    else:
                        raise AttributeError(
                            f"fit_dataset_copy object has no attribute 'hp'"
                        )
                except (ValueError, IndexError, AttributeError) as e:
                    print(
                        f"Could not reconstruct for {metab} at {isel_selector}. Error: {e}"
                    )
                    continue
        if total_fid is None:
            zeros = np.zeros(len(sampling_time_axis_s), dtype=np.complex128)
            total_fid = xr.DataArray(
                zeros,
                dims=["acquisition_idx"],
                coords={"acquisition_idx": np.arange(len(zeros))},
            )
        return fit_dataset_copy, total_fid

    def _plot_total_spectrum(
        self,
        ax,
        total_fid,
        plot_params,
        processed_plot_params,
        is_overlay,
        isel_selector,
        sel_selector,
        fit_dataset,
        widget_prefix="",
        **kwargs,
    ):
        """
        Plot the total spectrum/FID and optional raw data overlay.
        """

        # Get sampling time axis for total spectrum
        if "sampling_time_axis_s" in fit_dataset.attrs:
            sampling_time_axis_s = np.asarray(
                fit_dataset.attrs["sampling_time_axis_s"]
            )[: plot_params["npoints"]]
        else:
            try:
                sampling_time_axis_s = calc_sampling_time_axis(
                    self.spectroscopy_obj, npoints=plot_params["npoints"]
                )
            except:
                sampling_time_axis_s = np.arange(plot_params["npoints"])

        # Plot total spectrum
        if plot_params["show_temporal"]:
            total_plot_data = (
                total_fid.values if isinstance(total_fid, xr.DataArray) else total_fid
            )
            x_axis = sampling_time_axis_s[: len(total_plot_data)]
            ax.set_xlabel("Time (s)")
        else:
            fid_values = (
                total_fid.values if isinstance(total_fid, xr.DataArray) else total_fid
            )
            total_plot_data = np.fft.fftshift(np.fft.fft(fid_values))
            cut_off = getattr(self.spectroscopy_obj, "cut_off", 0)
            x_axis = get_freq_axis(
                self.spectroscopy_obj,
                plot_params["spec_unit"],
                cut_off,
                len(total_plot_data),
            )
            if x_axis is None:
                x_axis = np.arange(len(total_plot_data))
            ax.set_xlabel(
                processed_plot_params["xlabel"]
                or f"Frequency ({plot_params['spec_unit']})"
            )

        # Check for prefixed version first (e.g., ds1_overlay_display_mode for overlay data)
        overlay_mode_key = (
            f"{widget_prefix}_overlay_display_mode"
            if widget_prefix
            else "overlay_display_mode"
        )
        show_sum_only = kwargs.get(
            overlay_mode_key, kwargs.get("overlay_display_mode", False)
        )
        sum_label = "Sum (fitted)" if is_overlay else "Sum"
        sum_color = "r" if is_overlay else "k"
        sum_linestyle = ":" if is_overlay else "--"
        # If only showing sum, make line solid for visibility
        if show_sum_only:
            sum_linestyle = "-"
        display_values = plot_params["part_func"](total_plot_data)

        # Handle subtraction if requested
        subtract_mode = kwargs.get("subtract_overlay", False)
        percent_mode = kwargs.get("percent_difference", False)

        if is_overlay and (subtract_mode or percent_mode):
            # Retrieve stored base spectrum - check both caches
            # (fitted base uses _cached_base_spectrum, measured base uses _cached_base_spectrum_1d)
            base_spectrum = None
            if hasattr(self, "_cached_base_spectrum"):
                base_spectrum = self._cached_base_spectrum
            elif hasattr(self, "_cached_base_spectrum_1d"):
                base_spectrum = self._cached_base_spectrum_1d

            if base_spectrum is not None:
                display_values = self._compute_overlay_difference(
                    display_values, base_spectrum, percent=percent_mode
                )
                sum_label = (
                    "% Difference" if percent_mode else "Difference (Overlay - Base)"
                )
                sum_color = "purple"
                sum_linestyle = "-"
        elif not is_overlay:
            # Store base spectrum for potential subtraction
            self._cached_base_spectrum = display_values.copy()

        ax.plot(
            x_axis,
            display_values,
            label=sum_label,
            color=sum_color,
            linestyle=sum_linestyle,
        )

        if not plot_params["show_temporal"]:
            self._maybe_render_spectral_inset(
                is_base=not is_overlay,
                show_temporal=False,
                spec_unit=plot_params["spec_unit"],
                x_axis=x_axis,
                plot_values=display_values,
                color=sum_color,
                alpha=1.0,
                linewidth=processed_plot_params.get("linewidth", 1.0),
                kwargs=kwargs,
                widget_prefix=widget_prefix,
            )

        # Plot raw data if available and not overlay
        if not is_overlay and self.raw_data_array is not None:
            raw_sel = {
                k: v for k, v in sel_selector.items() if k in self.raw_data_array.dims
            }
            # Add bounds checking for raw data isel selectors
            # Prevents IndexError when overlaying arrays with different dimension sizes
            raw_isel = {
                k: v
                for k, v in isel_selector.items()
                if k in self.raw_data_array.dims and v < self.raw_data_array.sizes[k]
            }
            if "metabolite" in raw_sel:
                raw_sel.pop("metabolite")
            raw_spec = self.raw_data_array.sel(**raw_sel).isel(**raw_isel)
            raw_spec_proc = np.fft.fftshift(np.fft.fft(raw_spec.values))
            cut_off = getattr(self.spectroscopy_obj, "cut_off", 0)
            x_axis = get_freq_axis(
                self.spectroscopy_obj,
                plot_params["spec_unit"],
                cut_off,
                len(raw_spec_proc),
            )
            if x_axis is None:
                x_axis = np.arange(len(raw_spec_proc))
            raw_display = plot_params["part_func"](raw_spec_proc)
            ax.plot(
                x_axis,
                raw_display,
                label="Raw Data",
                color="gray",
                alpha=0.7,
            )

            if not plot_params["show_temporal"]:
                self._maybe_render_spectral_inset(
                    is_base=False,
                    show_temporal=False,
                    spec_unit=plot_params["spec_unit"],
                    x_axis=x_axis,
                    plot_values=raw_display,
                    color="gray",
                    alpha=0.7,
                    linewidth=processed_plot_params.get("linewidth", 1.0),
                    kwargs=kwargs,
                    widget_prefix=widget_prefix,
                )

        # Set labels and legend
        ax.set_ylabel(
            processed_plot_params["ylabel"]
            or f"{plot_params['spec_part_val']} Intensity"
        )
        if processed_plot_params["show_legend"]:
            ax.legend(
                loc=processed_plot_params["legend_loc"],
                fontsize=processed_plot_params["legend_fontsize"],
            )
        if plot_params["spec_unit"] == "ppm":
            ax.invert_xaxis()

        if len(x_axis) > 0:
            ax.set_xlim(x_axis[0], x_axis[-1])

    def _plot_reconstructed_spectrum(
        self,
        ax,
        fit_dataset,
        kwargs,
        processed_plot_params,
        is_overlay=False,
        widget_prefix=None,
    ):
        """Plot the reconstructed spectrum from a fit dataset."""
        # Get selectors and plot parameters
        try:
            isel_selector, sel_selector = self._get_reconstruction_selectors(
                fit_dataset, kwargs, is_overlay, widget_prefix
            )
        except Exception:
            debug_log_spectroscopy_plot(
                "[Warning]: _get_reconstruction_selectors failed\n"
            )
            raise

        debug_log_spectroscopy_plot(
            f"[DEBUG]: isel_selector: {isel_selector}, sel_selector: {sel_selector}"
        )

        if isel_selector is None:
            debug_log_spectroscopy_plot(
                f"[Warning]: No valid position selected: isel_selector is {isel_selector}"
            )
            return

        try:
            plot_params = self._get_reconstruction_plot_params(
                kwargs, is_overlay, widget_prefix
            )
        except Exception:
            debug_log_spectroscopy_plot(
                "[Warning]: _get_reconstruction_plot_params failed\n"
            )
            raise

        # Check display mode: True = sum only, False = individual peaks
        # kwargs.get("overlay_display_mode", False)

        # Prepare dataset and reconstruct spectra
        try:
            fit_dataset_copy, total_fid = self._reconstruct_metabolite_spectra(
                fit_dataset,
                isel_selector,
                sel_selector,
                plot_params,
                ax,
                is_overlay,
                kwargs,
                processed_plot_params,
                widget_prefix=widget_prefix,
            )
        except Exception:
            debug_log_spectroscopy_plot(
                "[Warning]: _reconstruct_metabolite_spectra failed\n"
            )
            raise

        # Plot total spectrum
        try:
            self._plot_total_spectrum(
                ax,
                total_fid,
                plot_params,
                processed_plot_params,
                is_overlay,
                isel_selector,
                sel_selector,
                fit_dataset,
                widget_prefix=widget_prefix,
                **kwargs,
            )
        except Exception:
            raise

    def _plot_1d_signal_unified(self, plot_data, signal_1d, is_base, kwargs):
        """Plot 1D spectrum with optional fitted data overlay.

        Renders the spectral domain visualization with support for:
        - Measured spectra (raw or preprocessed)
        - Fitted spectra with metabolite decomposition
        - Individual metabolite peaks vs total fit
        - Residuals display
        - Spectral inset (zoom) views

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data containing spectral information.
        signal_1d : xr.DataArray
            Spectral data to visualize.
        is_base : bool
            ``True`` when plotting the primary spectrum rather than an overlay.
        kwargs : dict
            Widget values including ``show_fit_results``, ``overlay_display_mode``,
            ``show_spec_inset``, and ``linebroadening``.

        Notes
        -----
        Automatically detects fitted data, supports both temporal and frequency
        domain display, and integrates with the ``review_fit`` workflow.
        """
        # Allow review_fit to refresh init-frequency state before drawing
        if hasattr(self.update_function, "_init_freq_update_hook"):
            try:
                self.update_function._init_freq_update_hook()
            except Exception as hook_err:
                debug_log_spectroscopy_plot(f"Init frequency hook failed: {hook_err}")

        # Get display parameters
        show_temporal = kwargs.get(f"{plot_data.widget_prefix}_view_toggle", False)
        spec_unit = (
            "ppm" if kwargs.get(f"{plot_data.widget_prefix}_spec_unit", True) else "Hz"
        )

        # Use the actual size of the preprocessed data (which includes zeropadding if applied)
        # Don't slice based on widget npoints - preprocessing already handled that
        if "acquisition_idx" in signal_1d.dims:
            actual_npoints = signal_1d.sizes["acquisition_idx"]
        else:
            actual_npoints = len(signal_1d)

        signal_to_plot = signal_1d

        # Get axis data based on the user's desired view (show_temporal)
        # The signal domain is already transformed during preprocessing
        signal_domain = signal_to_plot.attrs.get("signal_domain", "temporal")

        # Check if we need to display different axis than what the signal was transformed to
        # The logic should be: if user wants temporal view, show time axis; else show frequency
        if show_temporal:
            # For temporal view, we need to extend or create the time axis to match actual data length
            if hasattr(self.data_obj, "sampling_time_axis_s"):
                orig_time_axis = self.data_obj.sampling_time_axis_s
                if len(orig_time_axis) >= actual_npoints:
                    x_axis = orig_time_axis[:actual_npoints]
                else:
                    # Extend time axis if zeropadding increased length
                    dt = (
                        orig_time_axis[1] - orig_time_axis[0]
                        if len(orig_time_axis) > 1
                        else 1.0
                    )
                    x_axis = np.arange(actual_npoints) * dt
            else:
                x_axis = calc_sampling_time_axis(self.data_obj, npoints=actual_npoints)
            xlabel = "Time (s)"
        else:
            # For frequency view, use actual data length to calculate frequency axis
            x_axis = get_freq_axis(
                self.data_obj,
                spec_unit,
                getattr(self.data_obj, "cut_off", 0),
                actual_npoints,
            )

            # Fallback if preferred unit failed (e.g. ppm unavailable)
            if x_axis is None or len(x_axis) == 0:
                if spec_unit == "ppm":
                    # Try Hz
                    x_axis = get_freq_axis(
                        self.data_obj,
                        "Hz",
                        getattr(self.data_obj, "cut_off", 0),
                        actual_npoints,
                    )
                    xlabel = "Frequency (Hz)"
                    spec_unit = "Hz"

                # If still failed, use indices
                if x_axis is None or len(x_axis) == 0:
                    x_axis = np.arange(actual_npoints)
                    xlabel = "Index"
            else:
                xlabel = f"Frequency ({spec_unit})"

        # Ensure signal_to_plot is 1D for plotting
        signal_values = signal_to_plot.values

        # Find the acquisition dimension index
        acq_dim_idx = None
        if "acquisition_idx" in signal_to_plot.dims:
            acq_dim_idx = signal_to_plot.dims.index("acquisition_idx")

        # Squeeze out singleton dimensions, but preserve the acquisition
        # dimension
        if signal_values.ndim > 1:
            # Get all singleton dimensions except the acquisition dimension
            singleton_dims = []
            for i in range(signal_values.ndim):
                if signal_values.shape[i] == 1 and i != acq_dim_idx:
                    singleton_dims.append(i)

            if singleton_dims:
                # Squeeze out singleton dimensions
                signal_values = np.squeeze(signal_values, axis=tuple(singleton_dims))
            elif acq_dim_idx is not None:
                # If no singleton dims but we have acquisition dim, move it to
                # the end and take first slice of others
                signal_values = np.moveaxis(signal_values, acq_dim_idx, -1)
                # Take first slice of all non-acquisition dimensions
                while signal_values.ndim > 1:
                    signal_values = signal_values[0]
            else:
                # Fallback: just squeeze everything
                signal_values = signal_values.squeeze()

        # Check if data part transformation was already applied in preprocessing
        already_applied = signal_to_plot.attrs.get("spec_part_applied", None)

        # Apply data transformation
        spec_part_val = kwargs.get(f"{plot_data.widget_prefix}_spec_part", "Magnitude")
        if already_applied == spec_part_val:
            # Transformation already applied, use as-is
            plot_values = signal_values
        else:
            # Apply transformation now
            plot_values = part_func(signal_values, spec_part_val)

        # Ensure we have the right length
        min_len = min(len(x_axis), len(plot_values))
        x_axis = x_axis[:min_len]
        plot_values = plot_values[:min_len]

        # Store axis data for click handling (only for base plot)
        if is_base:
            self.update_function._spec_axis_data = x_axis
            self.update_function._show_temporal = show_temporal

        # Handle subtraction if requested
        subtract_mode = kwargs.get("subtract_overlay", False)
        percent_mode = kwargs.get("percent_difference", False)

        if not is_base and (subtract_mode or percent_mode):
            if hasattr(self, "_cached_base_spectrum_1d"):
                plot_values = self._compute_overlay_difference(
                    plot_values, self._cached_base_spectrum_1d, percent=percent_mode
                )
        elif is_base:
            # Store base spectrum for potential subtraction
            self._cached_base_spectrum_1d = plot_values.copy()

        # Plot styling (define variables before conditional block)
        color = (
            self.processed_plot_params.get("linecolor", "blue")
            if is_base
            else self.processed_plot_params.get("overlay_linecolor", "red")
        )

        if not is_base and (subtract_mode or percent_mode):
            alpha = 1.0
            label = "% Difference" if percent_mode else "Difference (Overlay - Base)"
        else:
            alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
            label = "Signal" if is_base else "Overlay"

        linewidth = self.processed_plot_params.get("linewidth", 1.0)

        # Check if we should skip plotting the base signal (when in subtraction mode)
        # We still process it above to cache it, but don't display it
        should_plot = True
        if is_base and (subtract_mode or percent_mode):
            should_plot = False

        if should_plot:
            self.update_function.ax_spec.plot(
                x_axis,
                plot_values,
                label=label,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

        # Add gray reference line at y=0 for spectral data (when not showing temporal view)
        if not show_temporal and is_base:
            self.update_function.ax_spec.axhline(
                y=0, color="gray", linestyle="-", alpha=0.5, linewidth=0.8
            )

        # Add vertical line indicator for currently selected frequency/timepoint
        if is_base:
            # Get the current acquisition index
            acquisition_idx = kwargs.get("acquisition_idx", 0)
            if acquisition_idx < len(x_axis):
                current_freq_time = x_axis[acquisition_idx]

                # Remove any existing frequency indicator line
                if (
                    hasattr(self.update_function, "_freq_indicator_line")
                    and self.update_function._freq_indicator_line
                ):
                    try:
                        self.update_function._freq_indicator_line.remove()
                    except (NotImplementedError, ValueError):
                        # Artist might already be removed or not attached
                        pass

                # Add new vertical line at current frequency/time
                self.update_function._freq_indicator_line = self.update_function.ax_spec.axvline(
                    x=current_freq_time,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.6,
                    label=f'Selected: {current_freq_time:.3f} {spec_unit if not show_temporal else "s"}',
                )

        # Draw per-metabolite init-frequency indicator (spectral view only)
        if is_base and not show_temporal:
            init_value = None
            if hasattr(self.update_function, "_init_freq_value"):
                init_info = getattr(self.update_function, "_init_freq_value", {}) or {}
                if spec_unit == "ppm":
                    init_value = init_info.get("ppm")
                else:
                    init_value = init_info.get("Hz")

            # Remove any existing init-frequency marker before adding a new one
            if hasattr(self.update_function, "_init_freq_marker"):
                existing_marker = getattr(self.update_function, "_init_freq_marker")
                if existing_marker is not None:
                    try:
                        existing_marker.remove()
                    except (ValueError, NotImplementedError):
                        # Matplotlib already removed the artist; ignore
                        pass
                    finally:
                        setattr(self.update_function, "_init_freq_marker", None)

            # Remove any existing shaded frequency band
            if hasattr(self.update_function, "_init_freq_band_patch"):
                existing_patch = getattr(self.update_function, "_init_freq_band_patch")
                if existing_patch is not None:
                    try:
                        existing_patch.remove()
                    except (ValueError, NotImplementedError):
                        pass
                    finally:
                        setattr(self.update_function, "_init_freq_band_patch", None)

            if init_value is not None and np.isfinite(init_value):
                try:
                    marker = self.update_function.ax_spec.axvline(
                        x=float(init_value),
                        color="lightblue",
                        linestyle="-",
                        linewidth=0.7,
                        alpha=0.9,
                    )
                    setattr(self.update_function, "_init_freq_marker", marker)
                except Exception as marker_err:
                    debug_log_spectroscopy_plot(
                        f"Init frequency marker failed: {marker_err}"
                    )

            # Draw shaded frequency band if available
            band_limits = None
            if hasattr(self.update_function, "_init_freq_band"):
                band_data = getattr(self.update_function, "_init_freq_band", {}) or {}
                unit_key = "ppm" if spec_unit == "ppm" else "Hz"
                band_limits = band_data.get(unit_key)

            if (
                band_limits
                and isinstance(band_limits, (tuple, list))
                and len(band_limits) == 2
            ):
                band_min, band_max = band_limits
                if (
                    band_min is not None
                    and band_max is not None
                    and np.isfinite(band_min)
                    and np.isfinite(band_max)
                    and band_min != band_max
                ):
                    span_min, span_max = sorted((float(band_min), float(band_max)))
                    try:
                        patch = self.update_function.ax_spec.axvspan(
                            span_min,
                            span_max,
                            color="lightblue",
                            alpha=0.25,
                            linewidth=0,
                            zorder=0,
                            label=f"Init freq: ({init_value:.2f} ± {np.abs(band_max-band_min):.2f} {spec_unit})",
                        )
                        setattr(self.update_function, "_init_freq_band_patch", patch)
                    except Exception as band_err:
                        debug_log_spectroscopy_plot(
                            f"Init frequency band failed: {band_err}"
                        )

        # ENHANCED: Add fit results display if enabled and fitted data is available
        self._add_fit_results_text(kwargs, is_base, plot_data.widget_prefix)

        # Set labels
        self.update_function.ax_spec.set_xlabel(xlabel)
        self.update_function.ax_spec.set_ylabel(f"{spec_part_val} Intensity")

        # Apply synced y-axis limits if requested
        if (
            hasattr(self.update_function, "_synced_ylim")
            and self.update_function._synced_ylim is not None
        ):
            self.update_function.ax_spec.set_ylim(self.update_function._synced_ylim)

        # Use 'auto' aspect ratio to prevent layout shifts from dynamic calculations
        self.update_function.ax_spec.set_aspect("auto")

        if len(x_axis) > 0:
            self.update_function.ax_spec.set_xlim(x_axis[0], x_axis[-1])

        self._maybe_render_spectral_inset(
            is_base=is_base,
            show_temporal=show_temporal,
            spec_unit=spec_unit,
            x_axis=x_axis,
            plot_values=plot_values,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            kwargs=kwargs,
            widget_prefix=plot_data.widget_prefix,
        )

        if spec_unit == "ppm" and signal_domain == "spectral":
            self.update_function.ax_spec.invert_xaxis()

    def _add_fit_results_text(self, kwargs, is_base, widget_prefix):
        """Add fit results text display to the spectrum plot if enabled and fitted data is available."""
        # Check if fit results display is enabled
        # Check for prefixed version first (e.g., ds1_show_fit_results for overlay data)
        fit_results_key = (
            f"{widget_prefix}_show_fit_results" if widget_prefix else "show_fit_results"
        )
        show_fit_results = kwargs.get(
            fit_results_key, kwargs.get("show_fit_results", False)
        )
        if not show_fit_results:
            # Remove existing fit results text if toggle is disabled
            if (
                hasattr(self.update_function, "_fit_results_texts")
                and self.update_function._fit_results_texts
            ):
                for text_obj in self.update_function._fit_results_texts:
                    try:
                        text_obj.remove()
                    except (NotImplementedError, ValueError):
                        pass
                self.update_function._fit_results_texts = []
            return

        # Find the plot_data that has fitted_spectra matching this widget_prefix
        fitted_data = None
        for pd in self._plot_data_list:
            # Match based on widget_prefix
            if pd.widget_prefix == widget_prefix:
                if (
                    hasattr(pd, "fitted_spectra")
                    and pd.fitted_spectra
                    and "_dataset" in pd.fitted_spectra
                ):
                    fitted_data = pd.fitted_spectra["_dataset"]
                    break

        if fitted_data is None:
            # If not found by prefix, try any fitted data (backward compatibility)
            for pd in self._plot_data_list:
                if (
                    hasattr(pd, "fitted_spectra")
                    and pd.fitted_spectra
                    and "_dataset" in pd.fitted_spectra
                ):
                    fitted_data = pd.fitted_spectra["_dataset"]
                    break

        if fitted_data is None:
            return

        # Get current voxel selection for fit results
        voxel_selectors = {}
        for key in [
            "channel_idx",
            "read_idx",
            "phase_idx",
            "slice_idx",
            "repetition_idx",
        ]:
            if key in kwargs and key in fitted_data.dims:
                voxel_selectors[key] = kwargs[key]

        # Remove existing fit results text
        if (
            hasattr(self.update_function, "_fit_results_texts")
            and self.update_function._fit_results_texts
        ):
            for text_obj in self.update_function._fit_results_texts:
                try:
                    text_obj.remove()
                except (NotImplementedError, ValueError):
                    pass

        # Create new fit results text
        self.update_function._fit_results_texts = []

        try:
            # Get fit results for current voxel
            if "metabolite" in fitted_data.dims:
                metabolites = fitted_data.coords["metabolite"].values

                # Build text content
                fit_text_lines = ["Fit Results:"]

                # Determine display mode and unit
                show_temporal = kwargs.get(f"{widget_prefix}_view_toggle", False)
                spec_unit = (
                    "ppm" if kwargs.get(f"{widget_prefix}_spec_unit", True) else "Hz"
                )

                for i, metab in enumerate(metabolites):
                    try:
                        # Get fit parameters for this metabolite and voxel
                        # Use .sel() for metabolite (string coordinate) and .isel() for spatial indices
                        spatial_selectors = {
                            k: v
                            for k, v in voxel_selectors.items()
                            if k in fitted_data.dims
                        }

                        # Extract amplitude with scientific notation formatting
                        if "fit_amps" in fitted_data.data_vars:
                            if spatial_selectors:
                                amp = complex(
                                    fitted_data["fit_amps"]
                                    .isel(spatial_selectors)
                                    .sel(metabolite=metab)
                                    .values
                                )
                            else:
                                amp = complex(
                                    fitted_data["fit_amps"].sel(metabolite=metab).values
                                )
                            amp_mag = abs(amp)
                            # Format as XX.X*eY
                            amp_str = f"{amp_mag:.1e}"
                        else:
                            amp_str = "0.0e0"

                        # Extract frequency in appropriate unit
                        if "fit_freqs" in fitted_data.data_vars:
                            if spatial_selectors:
                                freq_hz = float(
                                    fitted_data["fit_freqs"]
                                    .isel(spatial_selectors)
                                    .sel(metabolite=metab)
                                    .values
                                )
                            else:
                                freq_hz = float(
                                    fitted_data["fit_freqs"]
                                    .sel(metabolite=metab)
                                    .values
                                )
                            if spec_unit == "ppm":
                                # Convert Hz to ppm using the reference frequency
                                freq_ppm = freq_Hz_to_ppm(
                                    freq_Hz=freq_hz,
                                    data_obj=self.spectroscopy_obj,
                                )
                                freq_str = f"{freq_ppm:.2f} ppm"

                            else:
                                freq_str = f"{freq_hz:.1f} Hz"
                        else:
                            freq_str = f"0.0 {spec_unit}"

                        # Extract T2 or calculate FWHM based on temporal view
                        if show_temporal:
                            # Show T2 when in temporal (FID) view
                            if "fit_t2s" in fitted_data.data_vars:
                                if spatial_selectors:
                                    t2_s = float(
                                        fitted_data["fit_t2s"]
                                        .isel(spatial_selectors)
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                else:
                                    t2_s = float(
                                        fitted_data["fit_t2s"]
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                t2_ms = t2_s * 1000  # Convert to ms
                                t2_fwhm_str = f"T2={t2_ms:.1f}ms"
                            else:
                                t2_fwhm_str = "T2=0.0ms"
                        else:
                            # Show FWHM when in spectral view
                            if "fit_fwhm_Hz" in fitted_data.data_vars:
                                if spatial_selectors:
                                    fwhm_hz = float(
                                        fitted_data["fit_fwhm_Hz"]
                                        .isel(spatial_selectors)
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                else:
                                    fwhm_hz = float(
                                        fitted_data["fit_fwhm_Hz"]
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                if spec_unit == "ppm":
                                    fwhm_ppm = freq_Hz_to_ppm(
                                        freq_Hz=fwhm_hz,
                                        data_obj=self.spectroscopy_obj,
                                        ppm_centered_at_0=True,
                                    )
                                    t2_fwhm_str = f"FWHM={fwhm_ppm:.3f} ppm"
                                else:
                                    t2_fwhm_str = f"FWHM={fwhm_hz:.1f} Hz"
                            elif "fit_t2s" in fitted_data.data_vars:
                                # Calculate FWHM from T2: FWHM = 1/(π*T2)
                                if spatial_selectors:
                                    t2_s = float(
                                        fitted_data["fit_t2s"]
                                        .isel(spatial_selectors)
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                else:
                                    t2_s = float(
                                        fitted_data["fit_t2s"]
                                        .sel(metabolite=metab)
                                        .values
                                    )
                                fwhm_hz = 1.0 / (np.pi * t2_s)
                                if spec_unit == "ppm":
                                    fwhm_ppm = freq_Hz_to_ppm(
                                        freq_Hz=fwhm_hz,
                                        data_obj=self.spectroscopy_obj,
                                        ppm_centered_at_0=True,
                                    )
                                    t2_fwhm_str = f"FWHM={fwhm_ppm:.3f} ppm"
                                else:
                                    t2_fwhm_str = f"FWHM={fwhm_hz:.1f} Hz"
                            else:
                                t2_fwhm_str = "FWHM=0.0 Hz"

                        # Format text line with improved formatting
                        fit_text_lines.append(
                            f"{metab}: {freq_str}, {t2_fwhm_str}, $\\|A\\|$={amp_str}"
                        )

                    except Exception as e:
                        # Skip problematic metabolites
                        debug_log_spectroscopy_plot(
                            f"_add_fit_results: Error processing metabolite {metab}: {e}"
                        )
                        continue

                # Add text to plot (upper right corner)
                if len(fit_text_lines) > 1:  # Only if we have actual results
                    fit_text = "\n".join(fit_text_lines)

                    text_obj = self.update_function.ax_spec.text(
                        0.98,
                        0.8,  # Upper right corner
                        fit_text,
                        transform=self.update_function.ax_spec.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8
                        ),
                        fontsize=9,
                        fontfamily="monospace",
                    )

                    self.update_function._fit_results_texts.append(text_obj)

        except Exception as e:
            # Silently fail if we can't display fit results
            debug_log_spectroscopy_plot(f"Error in _add_fit_results_text: {e}")

    def _maybe_render_init_frequency_inset_marker(
        self,
        *,
        spec_unit: str,
        init_value: float,
        kwargs: dict,
    ) -> None:
        """Draw the init-frequency marker inside the inset if present."""

        inset_ax = getattr(self.update_function, "_spec_inset_ax", None)
        if inset_ax is None or inset_ax.figure is None:
            return

        inset_enabled = bool(kwargs.get("show_spec_inset", False))
        if not inset_enabled:
            return

        inset_min_hz = kwargs.get("spec_inset_min_Hz", 250.0)
        inset_max_hz = kwargs.get("spec_inset_max_Hz", 500.0)
        inset_min = self._convert_frequency_to_axis_unit(inset_min_hz, spec_unit)
        inset_max = self._convert_frequency_to_axis_unit(inset_max_hz, spec_unit)

        if not np.isfinite(inset_min) or not np.isfinite(inset_max):
            return

        low, high = sorted((inset_min, inset_max))
        if not (low <= init_value <= high):
            return

        try:
            line = inset_ax.axvline(
                x=init_value,
                color="lightblue",
                linestyle="-",
                linewidth=0.6,
                alpha=0.9,
            )
        except Exception as err:
            debug_log_spectroscopy_plot(f"Init freq inset marker failed: {err}")
        else:
            self.update_function._spec_inset_artists.append(line)

    def _maybe_render_init_frequency_inset_band(
        self,
        *,
        spec_unit: str,
        span_min: float,
        span_max: float,
        kwargs: dict,
    ) -> None:
        """Add shaded frequency range to inset when available."""

        inset_ax = getattr(self.update_function, "_spec_inset_ax", None)
        if inset_ax is None or inset_ax.figure is None:
            return

        inset_enabled = bool(kwargs.get("show_spec_inset", False))
        if not inset_enabled:
            return

        inset_min_hz = kwargs.get("spec_inset_min_Hz", 250.0)
        inset_max_hz = kwargs.get("spec_inset_max_Hz", 500.0)
        inset_min = self._convert_frequency_to_axis_unit(inset_min_hz, spec_unit)
        inset_max = self._convert_frequency_to_axis_unit(inset_max_hz, spec_unit)

        if not np.isfinite(inset_min) or not np.isfinite(inset_max):
            return

        low, high = sorted((inset_min, inset_max))
        band_low, band_high = sorted((span_min, span_max))

        if band_high < low or band_low > high:
            return

        clipped_low = max(band_low, low)
        clipped_high = min(band_high, high)

        try:
            patch = inset_ax.axvspan(
                clipped_low,
                clipped_high,
                color="lightblue",
                alpha=0.25,
                linewidth=0,
                zorder=0,
            )
        except Exception as err:
            debug_log_spectroscopy_plot(f"Init freq inset band failed: {err}")
        else:
            self.update_function._spec_inset_artists.append(patch)

    def _convert_frequency_to_axis_unit(self, freq_hz: float, spec_unit: str) -> float:
        """Convert frequency from Hz to the current spectral axis unit."""

        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return np.nan

        if spec_unit.lower() == "ppm":
            try:
                converted = freq_Hz_to_ppm(
                    freq_Hz=freq_hz, data_obj=self.spectroscopy_obj
                )
                arr = np.asarray(converted, dtype=float).ravel()
                if arr.size == 0 or not np.isfinite(arr[0]):
                    return np.nan
                return float(arr[0])
            except Exception as err:
                debug_log_spectroscopy_plot(
                    f"Unable to convert {freq_hz} Hz to ppm: {err}"
                )
                return np.nan

        return freq_hz

    def _maybe_render_spectral_inset(
        self,
        *,
        is_base: bool,
        show_temporal: bool,
        spec_unit: str,
        x_axis: np.ndarray,
        plot_values: np.ndarray,
        color: str,
        alpha: float,
        linewidth: float,
        kwargs: dict,
        widget_prefix: str = "",
    ) -> None:
        """Render or update a zoomed frequency inset if the user requested it."""

        if show_temporal or not hasattr(self.update_function, "ax_spec"):
            return

        # Check for global (unprefixed) version first, then fall back to prefixed
        # This allows spectral inset to be a global control
        inset_requested = bool(kwargs.get("show_spec_inset", False))
        if not inset_requested and widget_prefix:
            # Fall back to prefixed version
            inset_key = f"{widget_prefix}_show_spec_inset"
            inset_requested = bool(kwargs.get(inset_key, False))

        if is_base:
            if not inset_requested:
                return

            # Check for global versions first, then fall back to prefixed
            inset_min_hz = kwargs.get("spec_inset_min_Hz")
            inset_max_hz = kwargs.get("spec_inset_max_Hz")

            if inset_min_hz is None and widget_prefix:
                min_key = f"{widget_prefix}_spec_inset_min_Hz"
                inset_min_hz = kwargs.get(min_key, 250.0)
            elif inset_min_hz is None:
                inset_min_hz = 250.0

            if inset_max_hz is None and widget_prefix:
                max_key = f"{widget_prefix}_spec_inset_max_Hz"
                inset_max_hz = kwargs.get(max_key, 500.0)
            elif inset_max_hz is None:
                inset_max_hz = 500.0
            inset_min = self._convert_frequency_to_axis_unit(inset_min_hz, spec_unit)
            inset_max = self._convert_frequency_to_axis_unit(inset_max_hz, spec_unit)

            if not np.isfinite(inset_min) or not np.isfinite(inset_max):
                return

            low, high = sorted((inset_min, inset_max))
            mask = (x_axis >= low) & (x_axis <= high)

            if not mask.any():
                return

            x_inset = x_axis[mask]
            y_inset = plot_values[mask]

            if (
                hasattr(self.update_function, "_spec_inset_ax")
                and self.update_function._spec_inset_ax is not None
            ):
                try:
                    self.update_function._spec_inset_ax.remove()
                except:
                    pass

            try:
                inset_ax = zoomed_inset_axes(
                    self.update_function.ax_spec, zoom=2.5, loc="upper right"
                )
                self.update_function._spec_inset_ax = inset_ax

                inset_ax.plot(
                    x_inset, y_inset, color=color, linewidth=linewidth, alpha=alpha
                )
                inset_ax.set_xlim(low, high)
                inset_ax.set_ylim(np.nanmin(y_inset) * 0.95, np.nanmax(y_inset) * 1.05)
                if spec_unit == "ppm":
                    inset_ax.invert_xaxis()

                inset_ax.tick_params(axis="both", which="major", labelsize=6)
                inset_ax.grid(True, alpha=0.3)

                mark_inset(
                    self.update_function.ax_spec,
                    inset_ax,
                    loc1=2,
                    loc2=4,
                    fc="none",
                    ec="0.5",
                )
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to create spectral inset: {e}")
        else:
            if (
                hasattr(self.update_function, "_spec_inset_ax")
                and self.update_function._spec_inset_ax is not None
            ):
                try:
                    self.update_function._spec_inset_ax.plot(
                        x_axis,
                        plot_values,
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
                except:
                    pass

    def _get_specialized_widgets(
        self, plot_data, processed_plot_params, widget_prefix, slider_defaults
    ):
        """Create spectroscopy-specific widgets using WidgetFactory pattern.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data containing spectroscopy information.
        processed_plot_params : dict
            Normalized plotting parameters controlling widget defaults.
        widget_prefix : str
            Prefix used to namespace widget names.
        slider_defaults : dict
            Default slider values supplied by the caller.

        Returns
        -------
        dict
            Mapping of widget names to widget instances.
        """
        from ipywidgets import (
            Checkbox,
            Layout,
            ToggleButton,
        )

        # Get parent widgets first (includes spec_part dropdown, show_image_space toggle, etc.)
        widgets_dict = super()._get_specialized_widgets(
            plot_data, processed_plot_params, widget_prefix, slider_defaults
        )

        # Check if we have fitted data for fit-related widgets
        has_fitted_data = (
            hasattr(plot_data, "fitted_spectra")
            and plot_data.fitted_spectra
            and "_dataset" in plot_data.fitted_spectra
        )

        # Custom layout for smaller widgets
        compact_layout = Layout(width="auto", max_width="400px")
        button_layout = Layout(width="120px", height="30px")

        # Fit results display toggle (only if fitted data available)
        if has_fitted_data:
            widgets_dict["show_fit_results"] = Checkbox(
                value=slider_defaults.get("show_fit_results", False),
                description="Show Fit Results",
                disabled=False,
                tooltip="Display fit parameters (freq, T2/FWHM, amplitude) as text overlay",
                layout=compact_layout,
            )

            # Individual/Sum toggle for fitted spectra
            sum_individual_toggle = ToggleButton(
                value=slider_defaults.get("overlay_display_mode", False),
                description="Individual",
                disabled=False,
                tooltip="Toggle between individual metabolite peaks and sum view",
                layout=button_layout,
                button_style="danger",
            )

            # Add observer to change button color and description based on state
            def on_sum_toggle_change(change):
                if change["new"]:
                    sum_individual_toggle.description = "Sum View"
                    sum_individual_toggle.button_style = "success"
                else:
                    sum_individual_toggle.description = "Individual"
                    sum_individual_toggle.button_style = "danger"

            sum_individual_toggle.observe(on_sum_toggle_change, names="value")
            widgets_dict["overlay_display_mode"] = sum_individual_toggle

        return widgets_dict

    def _create_unified_widgets(
        self, plot_data_list, processed_plot_params, slider_defaults
    ):
        """Extend widget creation with spectroscopy-specific controls.

        Parameters
        ----------
        plot_data_list : list
            Plot data objects under construction.
        processed_plot_params : dict
            Normalized plotting parameters that guide default widget values.
        slider_defaults : dict
            Default slider configuration provided by the caller.

        Returns
        -------
        tuple
            Base controls box and widget mapping augmented with spectroscopy widgets.
        """
        # Call parent method - it handles everything including calling _get_specialized_widgets
        controls_box, widgets = super()._create_unified_widgets(
            plot_data_list, processed_plot_params, slider_defaults
        )

        # Add spectral inset controls to self.widgets AFTER calling parent
        compact_layout = Layout(width="auto", max_width="400px")

        self.widgets["show_spec_inset"] = Checkbox(
            value=slider_defaults.get("show_spec_inset", False),
            description="Show Spectral Inset",
            disabled=False,
            tooltip="Toggle zoomed inset view for a selected frequency band",
            layout=compact_layout,
        )

        self.widgets["spec_inset_min_Hz"] = FloatText(
            value=float(slider_defaults.get("spec_inset_min_Hz", 250.0)),
            description="Min [Hz]",
            step=5.0,
            layout=Layout(width="140px"),
        )

        self.widgets["spec_inset_max_Hz"] = FloatText(
            value=float(slider_defaults.get("spec_inset_max_Hz", 500.0)),
            description="Max [Hz]",
            step=5.0,
            layout=Layout(width="140px"),
        )

        # Find and update the Global Dimension Controls VBox to include spectral inset widgets
        if isinstance(controls_box, VBox) and len(controls_box.children) > 0:
            # Look for the HBox containing the sliders_box
            for i, child in enumerate(controls_box.children):
                if isinstance(child, HBox):
                    for j, box in enumerate(child.children):
                        if isinstance(box, VBox) and len(box.children) > 0:
                            # Check if this is the Global Dimension Controls box
                            if (
                                isinstance(box.children[0], HTML)
                                and "Global Dimension Controls" in box.children[0].value
                            ):
                                # Found it! Add spectral inset controls to this VBox
                                current_children = list(box.children)
                                current_children.append(self.widgets["show_spec_inset"])
                                current_children.append(
                                    HBox(
                                        [
                                            self.widgets["spec_inset_min_Hz"],
                                            self.widgets["spec_inset_max_Hz"],
                                        ]
                                    )
                                )
                                box.children = current_children
                                break

        # Ensure review_fit specific widgets are added
        self.widgets = self._ensure_review_fit_widgets(self.widgets, plot_data_list)

        return controls_box, self.widgets

    def _get_clean_metabolite_label(self, metabolite_coord, is_overlay=False):
        """Convert xarray metabolite coordinate to clean string label."""
        if hasattr(metabolite_coord, "values"):
            metab_name = str(metabolite_coord.values)
        else:
            metab_name = str(metabolite_coord)

        # Remove any numpy array formatting artifacts
        metab_name = metab_name.replace("'", "").strip()

        # Add overlay indicator if needed
        suffix = " " if is_overlay else ""
        return f"{metab_name}{suffix}"

    def update_metabolite_options(self, new_metabolites: list[str]) -> None:
        """
        Update the metabolite selection dropdown with a new list of metabolite names.

        This provides a public API for external components (like base_spectroscopy)
        to refresh the metabolite dropdown when the metabolite list changes.

        Parameters
        ----------
        new_metabolites : list[str]
            Updated list of active metabolite names
        """
        dropdown = self.widgets.get("overlay_selected_metabolites")

        if dropdown and hasattr(dropdown, "options"):
            current_value = dropdown.value

            # Update the options with the new list
            dropdown.options = new_metabolites

            # Preserve current selection if metabolite still exists
            if current_value in new_metabolites:
                dropdown.value = current_value
            # Otherwise, select the first available metabolite
            elif new_metabolites:
                dropdown.value = new_metabolites[0]

    def _has_spatial_dimensions(self, dataset) -> bool:
        """
        Check if a dataset has spatial dimensions (read_idx, phase_idx, slice_idx).

        Returns True if the dataset contains any spatial dimensions that would
        require spatial_data handling. Returns False for purely spectroscopic
        datasets that only have repetition_idx, channel_idx, metabolite dimensions.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to check for spatial dimensions

        Returns
        -------
        bool
            True if dataset has spatial dimensions, False otherwise
        """
        spatial_dims = {"read_idx", "phase_idx", "slice_idx"}
        dataset_dims = set(dataset.dims)
        return bool(spatial_dims.intersection(dataset_dims))

    def update_dataset_with_new_metabolites(self, fit_session) -> None:
        """
        Update the plotter's dataset structure to include new metabolites from FitSession.

        This method ensures the visualization dataset stays synchronized with
        metabolite additions/removals by using FitSession's build_fit_results() method.

        Parameters
        ----------
        fit_session : FitSession
            The fit session containing staged metabolite changes
        """
        try:
            # Use FitSession to build updated results dataset with new metabolites
            updated_results = fit_session.build_fit_results()

            # Update the plot data list to use the new dataset structure
            for plot_data in self._plot_data_list:
                if hasattr(plot_data, "fitted_spectra") and plot_data.fitted_spectra:
                    if "_dataset" in plot_data.fitted_spectra:
                        # Replace the dataset with the updated one
                        plot_data.fitted_spectra["_dataset"] = updated_results
                        debug_log_spectroscopy_plot(
                            f"Updated dataset structure with {len(updated_results.metabolite)} metabolites"
                        )

                        # Update spatial_data reference only if dataset has spatial dimensions
                        if plot_data.source_type == "dataset" and hasattr(
                            plot_data, "spatial_data"
                        ):

                            # Update spatial_data appropriately based on dataset dimensions
                            if self._has_spatial_dimensions(updated_results):
                                # For spatial datasets, use fit_amps as spatial data
                                data_var_name = "fit_amps"  # Default spatial variable
                                if data_var_name in updated_results.data_vars:
                                    plot_data.spatial_data = updated_results[
                                        data_var_name
                                    ]
                                    debug_log_spectroscopy_plot(
                                        f"Updated spatial_data reference for spatial dataset plotting"
                                    )
                            else:
                                # For non-spatial spectroscopic data, update spatial_data only if it was previously set
                                # This maintains interactive functionality while avoiding spatial interpretation
                                if plot_data.spatial_data is not None:
                                    data_var_name = (
                                        "fit_amps"  # Use fit_amps for consistency
                                    )
                                    if data_var_name in updated_results.data_vars:
                                        plot_data.spatial_data = updated_results[
                                            data_var_name
                                        ]
                                        debug_log_spectroscopy_plot(
                                            f"Updated data reference for non-spatial spectroscopic dataset"
                                        )
                                else:
                                    debug_log_spectroscopy_plot(
                                        "No spatial_data reference to update for non-spatial spectroscopic dataset"
                                    )

        except Exception as e:
            debug_log_spectroscopy_plot(f"Error updating dataset structure: {e}")
            import traceback

            debug_log_spectroscopy_plot(f"Traceback: {traceback.format_exc()}")

    def sync_with_fit_session(self, fit_session) -> None:
        """
        Synchronize plotter state with FitSession changes.

        This is the main synchronization method that should be called whenever
        metabolites are added or removed through the FitSession.

        Parameters
        ----------
        fit_session : FitSession
            The fit session to synchronize with
        """
        try:
            # Get updated metabolite list
            active_metabolites = fit_session.get_active_metabolite_names()

            # Update dropdown options
            self.update_metabolite_options(active_metabolites)

            # Update dataset structure for visualization
            self.update_dataset_with_new_metabolites(fit_session)

            debug_log_spectroscopy_plot(
                f"Synchronized plotter with FitSession: {len(active_metabolites)} metabolites"
            )

        except Exception as e:
            debug_log_spectroscopy_plot(f"Error synchronizing with FitSession: {e}")

    def _onclick_spatial(self, event):
        """Override spatial click handling to update only displayed dimension sliders.

        This ensures that when clicking into a 2D view (e.g., read_idx vs phase_idx),
        only the 2 displayed dimensions' sliders update, leaving the third (slicing)
        dimension stable. This prevents unwanted slice jumping during voxel selection.
        """
        # Check if a LassoSelector is currently active for mask creation
        # If so, skip slider updates to prevent slice jumping during mask drawing
        if hasattr(self.update_function, "active_mask_info"):
            active_mask_info = self.update_function.active_mask_info
            selector_tool = active_mask_info.get("selector_tool")
            # If a selector tool exists and mask creation is active, skip slider updates
            if selector_tool is not None and active_mask_info.get("name") is not None:
                logger.debug(
                    "Mask creation mode detected - skipping slider updates to prevent slice jumping"
                )
                return

        print(f"Spatial click detected at {event.xdata}, {event.ydata}")
        # Set flag BEFORE any slider updates to prevent perspective changes
        self._updating_from_click = True

        try:
            # Call parent method to handle click position updating
            super()._onclick_spatial(event)
        finally:
            # Reset flag after parent method completes
            self._updating_from_click = False

        # Update only the currently displayed dimension sliders based on the click position
        # (Don't update the slicing dimension to prevent unwanted slice jumps)
        if (
            hasattr(self.update_function, "_current_plot_dims")
            and self._has_click_position()
        ):
            click_pos = self._get_click_position()
            logger.debug(f"Click position: {click_pos}")

            # Get the current plot dimensions (the two dimensions being displayed)
            plot_dims = self.update_function._current_plot_dims
            if plot_dims and len(plot_dims) == 2:
                plot_dim_y, plot_dim_x = plot_dims
                displayed_dims = [plot_dim_x, plot_dim_y]

                # Flag is already set above, so additional slider updates won't trigger perspective changes
                self._updating_from_click = True

                try:
                    # Update ONLY the displayed dimension sliders to the clicked voxel position
                    # Do NOT update the slicing dimension to prevent slice jumping
                    for dim_name in displayed_dims:
                        if dim_name in click_pos and dim_name in self.widgets:
                            widget = self.widgets[dim_name]
                            if hasattr(widget, "value") and hasattr(widget, "max"):
                                old_val = widget.value
                                new_val = max(
                                    0, min(int(click_pos[dim_name]), widget.max)
                                )
                                logger.debug(
                                    f"Updating {dim_name} slider: {old_val} -> {new_val}"
                                )
                                widget.value = new_val
                finally:
                    # Reset flag
                    self._updating_from_click = False

                logger.debug(
                    f"Updated sliders for displayed dimensions: {displayed_dims}"
                )
            else:
                logger.debug(
                    "Could not determine current plot dimensions for selective slider update"
                )
