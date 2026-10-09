import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import clear_output, display
from ipywidgets import VBox
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import fit_maps to register the hp accessor on xarray datasets
# Import fit_maps to register the hp accessor on xarray datasets
try:
    from hypermri.analysis.fitting import fit_maps  # noqa: F401
except ImportError:
    # Handle case where fit_maps might not be available
    pass


from hypermri.utils.mathematics.calculations import part_func
from hypermri.utils.plot.cache_manager import CacheManager
from hypermri.utils.plot.data_preprocessor import DataPreprocessor
from hypermri.utils.plot.data_selector import DataSelector
from hypermri.utils.plot.unifiedPlotter import UnifiedPlotter
from hypermri.utils.plot.widget_factory import WidgetFactory
from hypermri.utils.system.logging import debug_log, init_default_logger

logger = init_default_logger(
    __name__,
    default_mode="critical",
    fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s ",
)

DEBUG_FILE = "/tmp/anatomy_plotter_debug.log"
DEBUG_LOG = True


def debug_log_anatomy_plot(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for AnatomyPlotter."""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


class AnatomyPlotter(UnifiedPlotter):
    """Modern anatomy plotter using the unified plotting framework.

    This class extends UnifiedPlotter to provide anatomy-specific functionality
    while supporting spectroscopic overlays and interactive features like create_mask.
    """

    def __init__(
        self,
        anatomy_obj,
        preprocessor=None,
        cache_manager=None,
        data_selector=None,
        widget_factory=None,
    ):
        """Initialize the anatomy plotter.

        Parameters
        ----------
        anatomy_obj : BaseAnatomy or compatible
            Anatomical data provider containing arrays such as `data_array` or
            `reco_array`.
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
        super().__init__(anatomy_obj)
        self.anatomy_obj = anatomy_obj

        self.preprocessor = preprocessor or DataPreprocessor(anatomy_obj)
        self.cache_manager = cache_manager or CacheManager(max_size=10)
        self.data_selector = data_selector or DataSelector()
        self.widget_factory = widget_factory or WidgetFactory(
            processed_plot_params=None
        )

    def _check_data_changes_and_invalidate(self):
        """Check if anatomy object data has changed and invalidate relevant caches.

        Override of parent method to handle anatomy-specific data objects.
        """
        if hasattr(self.anatomy_obj, "data_array"):
            self.invalidate_data_caches(self.anatomy_obj.data_array)
        if hasattr(self.anatomy_obj, "reco_array"):
            self.invalidate_data_caches(self.anatomy_obj.reco_array)

    def plot(
        self,
        input_array_list: list = None,
        plot_params: dict = None,
        overlay_masks: dict = None,
        return_handles: bool = False,
        return_plot_params_info: bool = False,
        **slider_defaults,
    ):
        """Display anatomical data with optional overlays.

        Parameters
        ----------
        input_array_list : list of (xr.DataArray or xr.Dataset), optional
            Data arrays or datasets to display. When omitted, uses
            `self.anatomy_obj.reco_array` or `self.anatomy_obj.data_array` if
            available.
        plot_params : dict, optional
            Plot customization dictionary, for example color maps or figure size.
        overlay_masks : dict, optional
            Mapping of overlay names to overlay definitions with keys such as
            ``buffer``, ``color``, and ``alpha``.
        return_handles : bool, optional
            When ``True``, return plotting handles instead of rendering directly.
        return_plot_params_info : bool, optional
            When ``True``, prints information about all available plot_params and
            slider_defaults and returns None (does not create the plot).
            When True, input_array_list can be None.
        **slider_defaults
            Default slider values forwarded to the widget layer.

        Returns
        -------
        tuple or None
            Plotting handles when ``return_handles`` is ``True``; otherwise ``None``.
            When ``return_plot_params_info`` is ``True``, returns ``None``.
        """

        # Handle parameter information request
        if return_plot_params_info:
            params_info = {
                "optional_plot_params": {
                    "figsize": "Figure size as tuple (width, height). Default: (16, 5) for multiple arrays, (12, 5) for single",
                    "colormap": "Colormap for anatomical visualization. Default: 'bone'",
                    "overlay_colormap": "Colormap for overlay data. Default: 'magma'",
                    "linecolor": "Line color for primary signal plots. Default: 'blue'",
                    "overlay_linecolor": "Line color for overlay signal plots. Default: 'red'",
                    "linewidth": "Line width for signal plots. Default: 1.0",
                    "vmin_perc": "Minimum percentile for colormap scaling (0-100). Default: 0.0",
                    "vmax_perc": "Maximum percentile for colormap scaling (0-100). Default: 100.0",
                    "default_spec_part": "Default data component to display ('Magnitude', 'Real', 'Imaginary', 'Phase'). Default: 'Magnitude'",
                    "legend_loc": "Legend location. Default: 'upper right'",
                    "legend_fontsize": "Legend font size. Default: 8",
                    "cbar_label_fontsize": "Colorbar label font size. Default: 8",
                    "xlabel": "Custom x-axis label. Default: None (auto-generated)",
                    "ylabel": "Custom y-axis label. Default: None (auto-generated)",
                    "show_legend": "Whether to show legend. Default: True",
                    "widget_fontsize": "Font size for widgets (e.g., '10px'). Default: '10px'",
                    "combine_channels": "Combine multi-channel data using root-sum-of-squares. Default: False",
                },
                "optional_overlay_masks": {
                    "mask_name": {
                        "data": "xr.DataArray - Binary or continuous mask data with matching spatial dimensions",
                        "color": "str - Color for mask overlay (e.g., 'red', 'blue', '#FF00FF'). Default: 'red'",
                        "alpha": "float - Transparency of mask overlay (0-1). Default: 0.3",
                        "buffer": "int - Buffer size in pixels for mask edges. Default: 0",
                    }
                },
                "optional_slider_defaults": {
                    "show_image_space": "bool - Toggle between image space (True) and k-space (False). Default: True",
                    "interpolate_image": "bool - Apply interpolation to anatomical images. Default: False",
                    "combine_channels": "bool - Combine multi-channel data using RSS. Default: False",
                    "spec_part": "str - Data component to display ('Magnitude', 'Real', 'Imaginary', 'Phase'). Default: 'Magnitude'",
                },
                "return_values": {
                    "return_handles=False": "None - Displays plot inline in Jupyter notebook",
                    "return_handles=True": "Tuple of (fig, axes, widgets, update_func, controls_box) for custom integration",
                },
                "data_requirements": {
                    "input_array_list": "List of xr.DataArray containing anatomical image data. "
                    "Must have spatial dimensions (e.g., 'x_idx', 'y_idx') and optionally 'slice_idx'.",
                    "overlay_masks": "Dictionary mapping mask names to mask definitions. Mask data must have compatible spatial dimensions.",
                },
                "special_workflows": {
                    "multi_slice": "For 3D anatomical data, interactive slice navigation is automatically enabled.",
                    "multi_channel": "For multi-channel data (e.g., from phased array coils), channel combination or individual viewing is supported.",
                    "temporal_dynamics": "For data with 'repetition_idx' or time dimensions, interactive temporal plotting is enabled with clickable voxels.",
                    "mask_integration": "Use add_interactive_overlay() or overlay_masks parameter to overlay segmentation masks or ROIs.",
                },
            }

            # Print parameter information in formatted style
            print("=" * 80)
            print("Parameter Information for AnatomyPlotter.plot()")
            print("=" * 80)

            print(f"\n{'Optional Parameters in plot_params:':-<80}")
            for param, description in params_info["optional_plot_params"].items():
                print(f"  • {param}")
                print(f"      {description}")

            print(f"\n{'Overlay Masks Structure (overlay_masks parameter):':-<80}")
            print(
                "  Dictionary with mask names as keys and mask definitions as values:"
            )
            for param, description in params_info["optional_overlay_masks"][
                "mask_name"
            ].items():
                print(f"    • {param}")
                print(f"        {description}")
            print("\n  Example:")
            print("    overlay_masks = {")
            print(
                "        'roi_mask': {'data': mask_array, 'color': 'red', 'alpha': 0.5},"
            )
            print(
                "        'tumor': {'data': tumor_mask, 'color': 'yellow', 'alpha': 0.3}"
            )
            print("    }")

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

        # Performance optimization: Check for data changes and invalidate caches if needed
        self._check_data_changes_and_invalidate()

        # Enhanced input validation - consistent with spectroscopyPlotter
        if input_array_list is None:
            # Determine input data from anatomy object
            if (
                hasattr(self.anatomy_obj, "reco_array")
                and self.anatomy_obj.reco_array is not None
            ):
                input_array_list = [self.anatomy_obj.reco_array]
            elif (
                hasattr(self.anatomy_obj, "data_array")
                and self.anatomy_obj.data_array is not None
            ):
                input_array_list = [self.anatomy_obj.data_array]
            else:
                raise AttributeError(
                    "Input data not provided and neither self.anatomy_obj.data_array "
                    "nor self.anatomy_obj.reco_array found."
                )

        # Validate input types - consistent with spectroscopyPlotter
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

        # Store overlay masks for interactive rendering
        if overlay_masks:
            for mask_name, mask_data in overlay_masks.items():
                self.add_interactive_overlay(mask_name, mask_data)

        # Set anatomy-specific plot parameters
        if plot_params is None:
            plot_params = {}
        anatomy_defaults = {
            "colormap": "bone",
            "figsize": (16, 5) if len(input_array_list) > 1 else (12, 5),
            "default_spec_part": "Magnitude",
            "vmin_perc": 0.0,
            "vmax_perc": 80.0,
            "range_mode": "Min/Max",
            "combine_channels": True,
            "interpolate_image": True,
        }
        for key, value in anatomy_defaults.items():
            if key not in plot_params:
                plot_params[key] = value

        if return_handles:
            # For create_mask integration, we need to return handles
            return self._plot_with_handles(
                input_array_list, plot_params, **slider_defaults
            )
        else:
            # Normal plotting
            return super().plot(input_array_list, plot_params, **slider_defaults)

    def _plot_with_handles(self, input_array_list, plot_params, **slider_defaults):
        """Render anatomical data and return widget handles.

        Parameters
        ----------
        input_array_list : list of (xr.DataArray or xr.Dataset)
            Inputs prepared for plotting.
        plot_params : dict
            Plot customization parameters.
        **slider_defaults
            Default slider values forwarded to the widget layer.

        Returns
        -------
        tuple
            ``(controls_box, output_area, run_interactive_update, initial_kwargs,
            update_plot, widgets_dict)`` used by the mask creation workflow.
        """
        # Process plot parameters
        self.processed_plot_params = self._process_plot_params(plot_params)

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
        output_area = widgets.Output()

        # Set up the update function
        def update_plot(**kwargs):
            with output_area:
                clear_output(wait=True)
                self._render_unified_plot(self._plot_data_list, kwargs)

        self.update_function = update_plot

        # CRITICAL FIX: Connect widgets to update function via manual observers
        # This is the missing piece that makes widget changes trigger plot updates in create_mask
        interactive_widgets = {
            k: v for k, v in self.widgets.items() if hasattr(v, "value")
        }

        # Manually set up observers on each widget to call update_plot
        def create_widget_observer():
            def widget_change_handler(*args, **kwargs):
                # Get current values from all widgets and call update_plot
                current_values = {k: w.value for k, w in interactive_widgets.items()}
                update_plot(**current_values)

            return widget_change_handler

        # Connect each widget to the observer
        widget_observer = create_widget_observer()
        for widget_name, widget in interactive_widgets.items():
            widget.observe(widget_observer, names="value")

        # Interactive output wrapper
        def run_interactive_update(**kwargs):
            # Store current overlay masks in the update function for
            # create_mask
            if hasattr(self, "interactive_overlays"):
                update_plot.overlay_masks_data = self.interactive_overlays

            current_widget_values = {}
            for name, widget in self.widgets.items():
                if hasattr(widget, "value"):
                    current_widget_values[name] = widget.value

            # Update current_selector with actual current values
            update_plot.current_selector = current_widget_values

            update_plot(**kwargs)

        # Initial values
        initial_kwargs = {
            name: (w.value if hasattr(w, "value") else w)
            for name, w in self.widgets.items()
            if hasattr(w, "value")
        }

        # Set up figure for create_mask compatibility
        self._setup_unified_figure(self._plot_data_list)

        # Return the expected 6-element tuple (FIXED: added widgets_dict as 6th element)
        return (
            controls_box,  # plot_controls
            output_area,  # plot_output_area
            run_interactive_update,  # run_interactive_update
            initial_kwargs,  # initial_kwargs
            update_plot,  # update_plot_handle
            self.widgets,  # widgets_dict (ADDED: the missing 6th element!)
        )

    def _prepare_dataarray_plot_data(
        self, data_array, kwargs, is_overlay, widget_prefix
    ):
        """Prepare plot data for anatomical arrays.

        Parameters
        ----------
        data_array : xr.DataArray
            Input array containing anatomical information.
        kwargs : dict
            Widget state used during preprocessing.
        is_overlay : bool
            Whether the plot represents an overlay.
        widget_prefix : str
            Prefix applied to widget identifiers.

        Returns
        -------
        UnifiedPlotData
            Populated plot data object tailored for anatomy rendering.
        """
        logger.debug(f"called for {widget_prefix}, is_overlay: {is_overlay}")
        # Call parent method first
        plot_data = super()._prepare_dataarray_plot_data(
            data_array, kwargs, is_overlay, widget_prefix
        )

        # Force anatomical data type to prevent spectroscopic widgets for ALL
        # data (base and overlay)
        original_data_type = plot_data.data_type
        plot_data.data_type = "anatomy"

        # Store the original data for later reprocessing with anatomy widgets
        plot_data.raw_data = data_array

        # If the original data was spectroscopic but we're forcing it to anatomical,
        # ensure spectral_data is preserved for temporal signal extraction
        if (
            original_data_type == "spectroscopy_measured"
            and plot_data.spectral_data is None
        ):
            # Apply preprocessing to get the spectral data for temporal
            # extraction
            processed_data = self._apply_preprocessing(
                data_array, kwargs, widget_prefix
            )
            plot_data.spectral_data = processed_data
            # Ensure spatial_data is also set correctly
            if plot_data.spatial_data is None:
                plot_data.spatial_data = processed_data

        return plot_data

    def _render_unified_plot(self, plot_data_list, kwargs):
        """Override main rendering to intercept and process anatomy widgets with caching."""

        # REPROCESS plot data with current widget values for anatomical data
        processed_plot_data_list = []
        for i, plot_data in enumerate(plot_data_list):
            prefix = plot_data.widget_prefix

            # Check for ANY preprocessing widget that requires data reprocessing
            combine_channels = kwargs.get(f"{prefix}_combine_channels", False)
            denoise_components = kwargs.get(f"{prefix}_denoise_components", 0)
            average_reps = kwargs.get(f"{prefix}_average_repetitions", False)

            # If any preprocessing widgets are active, force data reprocessing
            needs_reprocessing = (
                combine_channels or denoise_components > 0 or average_reps
            )

            if (
                needs_reprocessing
                and hasattr(plot_data, "raw_data")
                and plot_data.raw_data is not None
            ):
                # Performance optimization: Try to get from cache first
                cached_plot_data = self._get_cached_preprocessing(plot_data, kwargs)
                if cached_plot_data is not None:
                    processed_plot_data_list.append(cached_plot_data)
                else:
                    # Fallback: Create fresh copy if caching failed
                    fresh_plot_data = self._prepare_dataarray_plot_data(
                        plot_data.raw_data,
                        kwargs,
                        plot_data.is_overlay,
                        plot_data.widget_prefix,
                    )
                    fresh_plot_data.widget_prefix = plot_data.widget_prefix
                    fresh_plot_data.is_overlay = plot_data.is_overlay
                    processed_plot_data_list.append(fresh_plot_data)
            else:
                processed_plot_data_list.append(plot_data)

        # Call parent method with potentially updated data
        super()._render_unified_plot(processed_plot_data_list, kwargs)

    def _render_specialized_spatial(self, plot_data, is_base, kwargs):
        """Render spatial view with anatomy-specific preprocessing.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the spatial content to render.
        is_base : bool
            Whether the plot represents the primary dataset.
        kwargs : dict
            Current widget state used to determine preprocessing behaviour.
        """

        # Very obvious debug output that can't be missed

        prefix = plot_data.widget_prefix
        combine_key = f"{prefix}_combine_channels"

        [k for k in kwargs.keys() if prefix in k]

        # Check if any of our anatomy widgets have changed
        combine_val = kwargs.get(combine_key, False)

        if combine_val:
            # Force reprocessing of the data with current settings
            if (
                hasattr(plot_data, "raw_spectral_data")
                and plot_data.raw_spectral_data is not None
            ):
                raw_data = plot_data.raw_spectral_data
            else:
                # Fallback to spatial_data or any available data
                raw_data = plot_data.spatial_data or plot_data.spectral_data

            if raw_data is not None:
                # Apply our custom preprocessing
                processed_data = self._apply_preprocessing(raw_data, kwargs, prefix)
                # Update the plot_data with reprocessed data
                plot_data.spatial_data = processed_data

        # Call parent method with potentially updated data
        super()._render_specialized_spatial(plot_data, is_base, kwargs)

    def _get_specialized_widgets(
        self, plot_data, processed_plot_params, widget_prefix, slider_defaults
    ):
        """Create anatomy-specific widgets.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the spatial and spectral content.
        processed_plot_params : dict
            Normalized plotting parameters controlling widget defaults.
        widget_prefix : str
            Prefix used to namespace widget names.
        slider_defaults : dict
            Default slider values provided by the caller.

        Returns
        -------
        dict
            Mapping of widget names to widget instances.
        """
        # Get parent widgets first (includes spec_part, show_image_space, interpolate_image, sync_y_range, etc.)
        widgets_dict = super()._get_specialized_widgets(
            plot_data, processed_plot_params, widget_prefix, slider_defaults
        )

        return widgets_dict

    def _apply_preprocessing(self, data_array, kwargs, widget_prefix):
        """Apply preprocessing through the shared pipeline.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data requiring preprocessing.
        kwargs : dict
            Widget state controlling the preprocessing choices.
        widget_prefix : str
            Prefix identifying the widget namespace.

        Returns
        -------
        xr.DataArray
            Preprocessed data produced by the parent implementation.
        """
        # Delegate to parent (UnifiedPlotter) which handles DataPreprocessor delegation
        return super()._apply_preprocessing(data_array, kwargs, widget_prefix)

    def _handle_specialized_overlays(self, kwargs):
        """Handle anatomy-specific overlays like create_mask integration."""
        # Render interactive overlays like masks from create_mask
        self._render_interactive_overlays(kwargs)

    def _render_interactive_overlays(self, kwargs):
        """Render interactive overlays like masks from create_mask with proper coordinate alignment."""
        if not self.interactive_overlays or not hasattr(self.update_function, "ax_img"):
            return

        ax_img = self.update_function.ax_img

        # Clear previous mask overlays
        if hasattr(self.update_function, "mask_artists"):
            for artist_list in self.update_function.mask_artists.values():
                for artist in artist_list:
                    if artist.axes:
                        artist.remove()
        self.update_function.mask_artists = {}

        # Get current plot state for overlay positioning
        if not hasattr(self.update_function, "_current_extent"):
            return

        extent = self.update_function._current_extent
        origin = self.update_function._current_origin

        # Enable debug mode for coordinate alignment debugging
        self._debug_mask_alignment = False

        # Render each interactive overlay
        for overlay_name, overlay_data in self.interactive_overlays.items():
            if "buffer" not in overlay_data or overlay_data["buffer"] is None:
                continue

            mask_buffer = overlay_data["buffer"]
            color = overlay_data.get("color", "red")
            alpha = overlay_data.get("alpha", 0.4)

            try:
                # Apply current slicing to mask buffer with coordinate alignment
                mask_slice = self._apply_mask_slicing(mask_buffer, kwargs)

                if mask_slice is not None and np.any(mask_slice):
                    # Create RGBA overlay
                    from matplotlib.colors import to_rgba

                    rgba_color = to_rgba(color, alpha=alpha)
                    mask_rgba = np.zeros((*mask_slice.shape, 4), dtype=float)
                    mask_rgba[mask_slice] = rgba_color

                    # Overlay the mask
                    mask_artist = ax_img.imshow(
                        mask_rgba,
                        extent=extent,
                        origin=origin,
                        interpolation="nearest",
                        zorder=10,  # Above base images
                    )

                    # Store artist for cleanup
                    if not hasattr(self.update_function, "mask_artists"):
                        self.update_function.mask_artists = {}
                    if overlay_name not in self.update_function.mask_artists:
                        self.update_function.mask_artists[overlay_name] = []
                    self.update_function.mask_artists[overlay_name].append(mask_artist)
                else:
                    pass

            except Exception as e:
                warnings.warn(f"Failed to render overlay '{overlay_name}': {e}")

    def _setup_unified_figure(self, plot_data_list):
        """Set up figure layout for anatomy plotting with temporal signal display.

        Override to ensure we always have both spatial and spectral subplots available
        for anatomical data, even if no spectroscopic data is detected initially.
        """
        has_spatial = any(pd.has_spatial_data() for pd in plot_data_list)
        has_anatomical_with_temporal = any(
            pd.is_anatomical()
            and pd.spatial_data is not None
            and any(
                pd.spatial_data.sizes.get(dim, 1) > 1
                for dim in pd.spatial_data.dims
                if dim not in self._get_spatial_dims_using_selector(pd.spatial_data)
            )
            for pd in plot_data_list
        )

        # For anatomical data with temporal dimensions, always create both
        # subplots
        if has_spatial and has_anatomical_with_temporal:
            ncols = 2
            figsize = self.processed_plot_params.get("figsize", (16, 6))
        else:
            # Fallback to parent behavior
            return super()._setup_unified_figure(plot_data_list)

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
        self.update_function.ax_img = axes[0, 0]
        self.update_function.ax_spec = axes[0, 1]

        # Initialize mask-related attributes for create_mask compatibility
        self.update_function.current_selector = None
        self.update_function.current_plot_dims = (0, 1)  # Default to read, phase
        self.update_function.current_origin = "lower"
        self.update_function.current_state = [
            None,
            None,
        ]  # [slicing_dim, clicked_indices]

        # Set up colorbars for spatial plot
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

        # Connect click events (inherited from parent UnifiedPlotter)
        if self.update_function.ax_img:
            self.update_function.fig.canvas.mpl_connect(
                "button_press_event", self._onclick_spatial
            )
        if self.update_function.ax_spec:
            self.update_function.fig.canvas.mpl_connect(
                "button_press_event", self._onclick_spectral
            )

    def _update_secondary_plot_title(self, plot_data, is_base, kwargs, click_pos):
        """Update the secondary plot title with the currently selected voxel's value.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Data describing the currently selected dataset.
        is_base : bool
            Whether this is the primary (base) dataset.
        kwargs : dict
            Current widget values.
        click_pos : dict
            The currently selected voxel's coordinates.
        """
        try:
            if (
                not hasattr(self.update_function, "ax_spec")
                or self.update_function.ax_spec is None
            ):
                return

            # Get 2D slice used for display to extract the exact value shown
            spatial_slice = self._apply_spatial_slicing(
                plot_data.spatial_data, kwargs, plot_data.widget_prefix
            )

            if hasattr(self.update_function, "current_plot_dims"):
                plot_dim_y, plot_dim_x = self.update_function.current_plot_dims

                if plot_dim_x in click_pos and plot_dim_y in click_pos:
                    x_idx = int(click_pos[plot_dim_x])
                    y_idx = int(click_pos[plot_dim_y])

                    # Extract scalar value if indices are within bounds
                    if (
                        plot_dim_x in spatial_slice.dims
                        and plot_dim_y in spatial_slice.dims
                        and x_idx < spatial_slice.sizes[plot_dim_x]
                        and y_idx < spatial_slice.sizes[plot_dim_y]
                    ):

                        val = spatial_slice.isel(
                            {plot_dim_x: x_idx, plot_dim_y: y_idx}
                        ).values
                        if hasattr(val, "item"):
                            val = val.item()

                        # Apply data part transformation (Magnitude, Real, etc.)
                        spec_part_widget = kwargs.get(
                            f"{plot_data.widget_prefix}_spec_part", "Magnitude"
                        )
                        spec_part_val = (
                            spec_part_widget
                            if isinstance(spec_part_widget, str)
                            else "Magnitude"
                        )

                        # Use part_func from hypermri.utils.mathematics.calculations
                        # Check if transformation was already applied
                        already_applied = plot_data.spatial_data.attrs.get(
                            "spec_part_applied"
                        )
                        if already_applied == spec_part_val:
                            pass  # Already applied
                        else:
                            val = part_func(input_data=val, signal_part=spec_part_val)

                        # Format value
                        if isinstance(val, (int, float, np.number)):
                            if abs(val) < 0.01 and val != 0:
                                val_str = f"{val:.2e}"
                            else:
                                val_str = f"{val:.2f}"
                        else:
                            val_str = str(val)

                        # Get unit
                        unit = plot_data.spatial_data.attrs.get("unit", "")
                        if unit:
                            val_str += f" {unit}"

                        # Determine label name
                        name = "Base" if is_base else "Overlay"
                        if hasattr(plot_data, "current_data_variable"):
                            name = plot_data.current_data_variable

                        title_part = f"{name}: {val_str}"

                        # Update title
                        current_title = self.update_function.ax_spec.get_title()
                        if is_base:
                            self.update_function.ax_spec.set_title(title_part)
                        elif current_title:
                            self.update_function.ax_spec.set_title(
                                f"{current_title} | {title_part}"
                            )
                        else:
                            self.update_function.ax_spec.set_title(title_part)

        except Exception as e:
            logger.debug(f"Failed to update title with voxel value: {e}")

    def _render_specialized_spectral(self, plot_data, is_base, kwargs):
        """Simplified spectral rendering: show temporal plot only after click."""
        logger.debug(
            f"🚀 _render_specialized_spectral called - is_base: {is_base}, is_anatomical: {plot_data.is_anatomical()}, data_type: {plot_data.data_type}"
        )

        # Check if we have a spectral axis available
        if (
            not hasattr(self.update_function, "ax_spec")
            or self.update_function.ax_spec is None
        ):
            logger.debug("  No spectral axis available")
            return

        # For anatomical data, show temporal signal analysis when clicked
        if plot_data.is_anatomical():
            logger.debug(
                f"🔍 TEMPORAL SIGNAL DEBUG: plot_data.is_anatomical() = True, data_type = {plot_data.data_type}"
            )
            if self._has_click_position():
                logger.debug(
                    "🔍 TEMPORAL SIGNAL DEBUG: Has click position, showing temporal signal"
                )
                click_pos = self._get_click_position()
                logger.debug(f"🔍 TEMPORAL SIGNAL DEBUG: Click position = {click_pos}")

                # Update title with voxel value
                self._update_secondary_plot_title(plot_data, is_base, kwargs, click_pos)

                # PRIORITY 1: Check if this is fitted data that needs reconstruction
                # This should be checked FIRST before trying temporal signal extraction
                dataset = None
                if hasattr(plot_data, "fitted_spectra") and plot_data.fitted_spectra:
                    dataset = plot_data.fitted_spectra.get("_dataset")

                # Check if we have a dataset with model_name (indicates fitted data)
                if dataset is not None and hasattr(dataset, "attrs"):
                    model_name = dataset.attrs.get("model_name", None)
                    # List of models that should use reconstruction
                    reconstruction_models = [
                        "t1_inversion_recovery",
                        "monoexponential_adc",
                        "patlak",
                        "two_compartment_model",
                        "tofts_model",
                    ]

                    if model_name in reconstruction_models:
                        debug_log_anatomy_plot(
                            f"🔍 FITTED DATA DEBUG: Detected {model_name} - attempting reconstruction"
                        )

                        # Try to reconstruct fitted curve
                        fitted_curve, axis_data, xlabel = (
                            self._reconstruct_fitted_curve(plot_data, kwargs, click_pos)
                        )

                        if fitted_curve is not None:
                            debug_log_anatomy_plot(
                                "🔍 FITTED DATA DEBUG: Successfully reconstructed fitted curve"
                            )
                            # Plot the reconstructed fitted curve
                            self._plot_anatomical_temporal_signal(
                                plot_data,
                                fitted_curve,
                                axis_data,
                                xlabel,
                                is_base,
                                kwargs,
                            )
                            curve_type = (
                                "Fitted curve" if is_base else "Overlay fitted curve"
                            )
                            debug_log_anatomy_plot(
                                f"🔍 FITTED DATA DEBUG: {curve_type} displayed for clicked voxel"
                            )
                            return  # Success - don't try other methods
                        else:
                            debug_log_anatomy_plot(
                                "🔍 FITTED DATA DEBUG: Reconstruction returned None"
                            )

                # PRIORITY 2: Try to get temporal signal (for non-fitted or if reconstruction failed)
                (
                    temporal_signal,
                    axis_data,
                    xlabel,
                ) = self._get_anatomical_temporal_signal(plot_data, kwargs, click_pos)
                logger.debug(
                    f"🔍 TEMPORAL SIGNAL DEBUG: _get_anatomical_temporal_signal returned:"
                )
                logger.debug(
                    f"🔍 TEMPORAL SIGNAL DEBUG:   temporal_signal = {temporal_signal}"
                )
                logger.debug(f"🔍 TEMPORAL SIGNAL DEBUG:   axis_data = {axis_data}")
                logger.debug(f"🔍 TEMPORAL SIGNAL DEBUG:   xlabel = {xlabel}")

                if temporal_signal is not None:
                    logger.debug(
                        "🔍 TEMPORAL SIGNAL DEBUG: temporal_signal is not None - plotting"
                    )
                    # Plot temporal signal curve
                    self._plot_anatomical_temporal_signal(
                        plot_data, temporal_signal, axis_data, xlabel, is_base, kwargs
                    )
                    signal_type = (
                        "Temporal plot" if is_base else "Overlay temporal plot"
                    )
                    logger.debug(
                        f"🔍 TEMPORAL SIGNAL DEBUG: {signal_type} displayed for clicked voxel"
                    )
                else:
                    logger.debug(
                        "🔍 TEMPORAL SIGNAL DEBUG: temporal_signal is None - falling back to line profile"
                    )
                    # PRIORITY 3: Fallback to line profile
                    profile_data = self._get_anatomical_profile(plot_data, kwargs)
                    if profile_data is not None:
                        self._plot_anatomical_profile(profile_data, is_base, kwargs)
                        profile_type = (
                            "Profile plot" if is_base else "Overlay profile plot"
                        )
                        logger.debug(": {profile_type} displayed for clicked voxel")
                    elif (
                        is_base
                    ):  # Only show message for base data to avoid duplication
                        # Show message if no data available
                        self.update_function.ax_spec.text(
                            0.5,
                            0.5,
                            "No temporal data available\nfor this voxel",
                            transform=self.update_function.ax_spec.transAxes,
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
            elif is_base:  # Only show instruction for base data to avoid duplication
                # No click position - show instruction
                self.update_function.ax_spec.text(
                    0.5,
                    0.5,
                    "Click on the anatomical image\nto see temporal signal",
                    transform=self.update_function.ax_spec.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7
                    ),
                )
                logger.debug("Showing instruction - user needs to click")

    def _reconstruct_fitted_curve(self, plot_data, kwargs, click_pos):
        """Reconstruct a fitted curve for temporal display.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data containing the fitted dataset.
        kwargs : dict
            Widget state parameters governing reconstruction options.
        click_pos : dict
            Clicked voxel coordinates.

        Returns
        -------
        tuple
            ``(reconstructed_curve, axis_data, xlabel)`` when reconstruction
            succeeds, otherwise ``(None, None, None)``.
        """
        if not hasattr(plot_data, "fitted_spectra") or not plot_data.fitted_spectra:
            return None, None, None

        dataset = plot_data.fitted_spectra.get("_dataset")
        if dataset is None or not hasattr(dataset, "hp"):
            return None, None, None

        model = dataset.attrs.get("model_name", None)
        if model is None:
            return None, None, None

        try:
            # Determine dynamic dimension to avoid selecting it
            dynamic_dim = None
            if dataset is not None and hasattr(dataset, "attrs"):
                dynamic_dim = dataset.attrs.get("dynamic_dim", None)

            # Build voxel selector from click position
            voxel_coords = {}
            if dataset is not None:
                dataset_dims = dict(dataset.sizes)
            else:
                dataset_dims = {}

            def _normalize_index(dim_name, idx_value):
                """Cast selector to valid integer within dataset bounds."""
                try:
                    idx_int = int(idx_value)
                except (TypeError, ValueError):
                    idx_int = 0
                dim_size = dataset_dims.get(dim_name)
                if dim_size is not None and dim_size > 0:
                    idx_int = max(0, min(idx_int, dim_size - 1))
                return idx_int

            # 1) Use click position for displayed spatial axes
            for dim, idx in click_pos.items():
                if dim in dataset_dims and dim != dynamic_dim:
                    voxel_coords[dim] = _normalize_index(dim, idx)

            # 2) Fall back to current widget/slider values for remaining dims
            for dim in dataset_dims:
                if dim == dynamic_dim or dim in voxel_coords:
                    continue
                if dataset_dims[dim] <= 1:
                    continue  # Nothing to select

                widget_key = f"{plot_data.widget_prefix}_{dim}"
                if widget_key in kwargs:
                    selector_val = kwargs[widget_key]
                else:
                    selector_val = kwargs.get(dim, 0)

                voxel_coords[dim] = _normalize_index(dim, selector_val)

            axis_override = self._get_reconstruction_axis_override(
                dataset, voxel_coords, plot_data.widget_prefix, kwargs
            )

            # Prepare reconstruction kwargs, ensuring we include unprefixed versions of
            # critical parameters that the hp accessor expects
            reconstruct_kwargs = dict(kwargs)

            # Map prefixed keys to unprefixed ones for the accessor
            prefix = plot_data.widget_prefix
            for key in ["reco_space", "tr_s", "fa_deg", "r1", "t10_ms", "s0"]:
                prefixed_key = f"{prefix}_{key}"
                if prefixed_key in kwargs:
                    reconstruct_kwargs[key] = kwargs[prefixed_key]

            if axis_override:
                reconstruct_kwargs.update(axis_override["kwargs"])

            # Pass reconstruction space hint if available
            if hasattr(plot_data, "reco_space"):
                reconstruct_kwargs.setdefault("reco_space", plot_data.reco_space)

            # Reconstruct using the hp accessor
            reconstructed = dataset.hp.reconstruct(
                voxel_coords=voxel_coords, **reconstruct_kwargs
            )

            # Get the appropriate axis based on model type
            if model == "t1_inversion_recovery":
                debug_log_anatomy_plot(
                    "🔍 FITTED DATA DEBUG: Model is T1 inversion recovery"
                )
                # For T1: inversion times
                axis_coord = dataset.attrs.get("inversion_times_ms")
                axis_data = None
                if axis_override and axis_override.get("axis_da") is not None:
                    axis_da = axis_override["axis_da"]
                    axis_data = axis_da.values
                elif axis_coord is not None:
                    debug_log_anatomy_plot(
                        f"🔍 FITTED DATA DEBUG: inversion_times_ms = {axis_coord}"
                    )
                    if hasattr(axis_coord, "isel"):
                        debug_log_anatomy_plot(
                            "🔍 FITTED DATA DEBUG: axis_coord has 'isel' - selecting voxel-specific data"
                        )
                        # Select inversion times for this voxel if multi-dimensional
                        axis_selector = {
                            k: v
                            for k, v in voxel_coords.items()
                            if k in axis_coord.dims
                        }
                        debug_log_anatomy_plot(
                            f"🔍 FITTED DATA DEBUG: axis_selector = {axis_selector}, voxel_coords.items() = {voxel_coords.items()}"
                        )
                        if axis_selector:
                            axis_data = axis_coord.isel(axis_selector).values
                        else:
                            axis_data = axis_coord.values
                        debug_log_anatomy_plot(f"axis_data after isel: {axis_data}")
                    else:
                        axis_data = np.asarray(axis_coord)
                if axis_data is None:
                    return None, None, None
                xlabel = (
                    axis_override.get("axis_label", "Inversion Time (ms)")
                    if axis_override
                    else "Inversion Time (ms)"
                )

            elif model == "monoexponential_adc":
                # For ADC: b-values
                axis_data = dataset.attrs.get("b_values")
                if axis_data is None:
                    return None, None, None
                axis_data = np.asarray(axis_data)
                xlabel = "b-value (s/mm²)"

            elif model == "patlak":
                # For Patlak: time points
                axis_coord = dataset.attrs.get("time_points_s")
                if axis_coord is not None:
                    if hasattr(axis_coord, "values"):
                        axis_data = axis_coord.values
                    else:
                        axis_data = np.asarray(axis_coord)
                    xlabel = "Time (s)"
                else:
                    return None, None, None

            elif model == "two_compartment_model":
                # For 2CM: time points
                axis_coord = dataset.attrs.get("time_points_s")
                if axis_coord is not None:
                    if hasattr(axis_coord, "values"):
                        axis_data = axis_coord.values
                    else:
                        axis_data = np.asarray(axis_coord)
                    xlabel = "Time (s)"
                else:
                    return None, None, None

            elif model == "tofts_model":
                # For Tofts: time points
                axis_coord = dataset.attrs.get("time_points_s")
                if axis_override and axis_override.get("axis_da") is not None:
                    axis_data = axis_override["axis_da"].values
                elif axis_coord is not None:
                    if hasattr(axis_coord, "values"):
                        axis_data = axis_coord.values
                    else:
                        axis_data = np.asarray(axis_coord)
                else:
                    return None, None, None
                xlabel = "Time (s)"
            else:
                return None, None, None

            # Extract the reconstructed curve data
            if hasattr(reconstructed, "values"):
                curve_data = reconstructed.values
            else:
                curve_data = np.asarray(reconstructed)

            # Ensure 1D
            curve_data = curve_data.squeeze()

            # Ensure axis_data is 1D and matches curve_data length
            debug_log_anatomy_plot(f"🔍 FITTED DATA DEBUG: axis_data = {axis_data}")
            debug_log_anatomy_plot(f"🔍 FITTED DATA DEBUG: curve_data = {curve_data}")

            # Create DataArray for consistent handling
            axis_dim_name = "dynamic_idx"
            axis_values = np.asarray(axis_data).reshape(-1)
            if axis_override and axis_override.get("axis_da") is not None:
                axis_da = axis_override["axis_da"]
                if axis_da.dims:
                    axis_dim_name = axis_da.dims[-1]
                axis_values = axis_da.values

            if curve_data.ndim == 1:
                reconstructed_da = xr.DataArray(
                    curve_data,
                    dims=[axis_dim_name],
                    coords={axis_dim_name: axis_values},
                )
            else:
                # For multi-model results, we preserve all dimensions
                # but ensure the dynamic dimension is named and coordinated correctly
                if isinstance(reconstructed, xr.DataArray):
                    reconstructed_da = reconstructed
                    # Check if the dynamic dimension in reconstructed matches what we expect
                    # if not, we might need to rename or reassign
                    current_dynamic_dim = reconstructed.dims[-1]
                    if current_dynamic_dim != axis_dim_name:
                        reconstructed_da = reconstructed_da.rename(
                            {current_dynamic_dim: axis_dim_name}
                        )
                    reconstructed_da = reconstructed_da.assign_coords(
                        {axis_dim_name: axis_values}
                    )
                else:
                    reconstructed_da = xr.DataArray(
                        curve_data,
                        dims=[f"dim_{i}" for i in range(curve_data.ndim - 1)]
                        + [axis_dim_name],
                        coords={axis_dim_name: axis_values},
                    )

            return reconstructed_da, axis_data, xlabel

        except Exception as e:
            logger.debug(f"Failed to reconstruct fitted curve: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None, None, None

    def _plot_anatomical_profile(self, profile_data, is_base, kwargs):
        """Plot anatomical line profile or similar 1D data."""
        if not hasattr(self.update_function, "ax_spec"):
            return

        ax_spec = self.update_function.ax_spec

        # Get display parameters (now using Dropdown)
        # Handle both PlotData objects and DataArrays
        if hasattr(profile_data, "widget_prefix"):
            widget_key = f"{profile_data.widget_prefix}_spec_part"
        else:
            # For DataArray profiles, use a default key
            widget_key = "spec_part"

        spec_part_widget = kwargs.get(widget_key, "Magnitude")
        spec_part_val = (
            spec_part_widget if isinstance(spec_part_widget, str) else "Magnitude"
        )

        # Apply transformation and plot (cached)
        plot_values = self._get_cached_transform(profile_data, spec_part_val)
        x_axis = np.arange(len(plot_values))

        # Handle subtraction if requested
        subtract_mode = kwargs.get("subtract_overlay", False)
        percent_mode = kwargs.get("percent_difference", False)

        if not is_base and (subtract_mode or percent_mode):
            if hasattr(self, "_cached_base_profile"):
                plot_values = self._compute_overlay_difference(
                    plot_values, self._cached_base_profile, percent=percent_mode
                )
        elif is_base:
            # Store base profile for potential subtraction
            self._cached_base_profile = plot_values.copy()

        # Check if we should skip plotting the base signal (when in subtraction mode)
        # We still process it above to cache it, but don't display it
        should_plot = True
        if is_base and (subtract_mode or percent_mode):
            should_plot = False

        if should_plot:
            # Plot styling
            color = "blue" if is_base else "red"
            if not is_base and (subtract_mode or percent_mode):
                alpha = 1.0
                label = (
                    "% Difference" if percent_mode else "Difference (Overlay - Base)"
                )
            else:
                alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
                label = "Profile" if is_base else "Overlay Profile"

            ax_spec.plot(x_axis, plot_values, label=label, color=color, alpha=alpha)
        ax_spec.set_xlabel("Position")
        ax_spec.set_ylabel(f"{spec_part_val} Intensity")
        ax_spec.set_xlim(np.min(x_axis), np.max(x_axis))
        # Use 'auto' aspect ratio to prevent layout shifts from dynamic calculations
        ax_spec.set_aspect("auto")

        # Add thin gray horizontal line at y=0
        ax_spec.axhline(y=0, color="gray", linewidth=0.5, alpha=0.7, zorder=0)

        ax_spec.legend()

    def _apply_mask_slicing(self, mask_buffer, kwargs):
        """Apply current slider values to mask buffer for display with coordinate alignment."""
        try:
            # ENHANCED: Get current slice selectors more accurately
            selectors = {}

            # For masks, we want to slice on ALL dimensions that are not the current plot dimensions
            # Get current plot dimensions (these are the 2D spatial dims being displayed)
            current_plot_dims = getattr(
                self.update_function, "current_plot_dims", ("phase_idx", "read_idx")
            )

            # Apply slicing for all non-spatial dimensions (like slice_idx, repetition_idx, etc.)
            for dim in mask_buffer.dims:
                if dim not in current_plot_dims and dim in kwargs:
                    idx = kwargs[dim]
                    # Ensure index is valid
                    if idx < mask_buffer.sizes[dim]:
                        selectors[dim] = idx
                    else:
                        # Use first slice as fallback
                        selectors[dim] = 0

            # Apply slicing
            if selectors:
                mask_slice = mask_buffer.isel(**selectors)
            else:
                mask_slice = mask_buffer

            # Ensure we have 2D data for display
            while mask_slice.ndim > 2:
                # Remove extra dimensions by selecting first element
                extra_dim = mask_slice.dims[-1]
                mask_slice = mask_slice.isel({extra_dim: 0})

            # Convert to boolean numpy array
            mask_2d = mask_slice.values.astype(bool)

            # COORDINATE ALIGNMENT FIX:
            # Ensure the mask is oriented correctly for display
            buffer_dims = mask_slice.dims

            # Check if we need to transpose for display alignment
            if (
                len(current_plot_dims) == 2
                and len(buffer_dims) == 2
                and current_plot_dims != buffer_dims
                and current_plot_dims == buffer_dims[::-1]
            ):

                # Transpose mask to match plot coordinate order
                mask_2d = mask_2d.T

            return mask_2d

        except Exception as e:
            import traceback

            print(f"[MASK_SLICE] ERROR in _apply_mask_slicing: {e}")
            print(f"[MASK_SLICE] Traceback: {traceback.format_exc()}")
            return None

    def create_mask_integration(self):
        """Create integration point for create_mask functionality.

        This method provides the interface that create_mask expects,
        making it easy to integrate the mask creation workflow.
        """
        # This would be called by create_mask to get the plotting handles
        # The create_mask function would then add its own UI elements
        # and manage the mask creation workflow

    def add_spectroscopic_overlay(
        self, spec_data, overlay_name="spectroscopy", overlay_params=None
    ):
        """Add spectroscopic data as an overlay to anatomical images.

        Parameters
        ----------
        spec_data : xr.DataArray or xr.Dataset
            Spectroscopic data to overlay.
        overlay_name : str, optional
            Identifier used for the overlay. The default is ``"spectroscopy"``.
        overlay_params : dict, optional
            Display parameters such as ``alpha`` or ``colormap``.

        Returns
        -------
        UnifiedPlotData
            Plot data object representing the overlay.
        """
        if overlay_params is None:
            overlay_params = {}

        # Convert spectroscopic data to standardized format
        if isinstance(spec_data, xr.DataArray):
            overlay_plot_data = self._prepare_dataarray_plot_data(
                spec_data, {}, True, f"overlay_{overlay_name}"
            )
        elif isinstance(spec_data, xr.Dataset):
            overlay_plot_data = self._prepare_dataset_plot_data(
                spec_data, {}, True, f"overlay_{overlay_name}"
            )
        else:
            raise TypeError("spec_data must be xarray.DataArray or xarray.Dataset")

        # Add to plot data list if it exists
        if hasattr(self, "_plot_data_list"):
            self._plot_data_list.append(overlay_plot_data)

        # Set up overlay parameters
        overlay_data = {
            "plot_data": overlay_plot_data,
            "alpha": overlay_params.get("alpha", 0.5),
            "colormap": overlay_params.get("colormap", "magma"),
            "visible": overlay_params.get("visible", True),
        }

        # Add as interactive overlay
        self.add_interactive_overlay(overlay_name, overlay_data)

        return overlay_plot_data

    def remove_spectroscopic_overlay(self, overlay_name):
        """Remove a spectroscopic overlay.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to remove.
        """
        # Remove from interactive overlays
        self.remove_interactive_overlay(overlay_name)

        # Remove from plot data list if it exists
        if hasattr(self, "_plot_data_list"):
            self._plot_data_list = [
                pd
                for pd in self._plot_data_list
                if not pd.widget_prefix.endswith(overlay_name)
            ]

    def _prepare_mixed_plot_data(self, anatomy_data, spec_data_list=None):
        """Prepare mixed anatomical and spectroscopic data for unified plotting.

        Parameters
        ----------
        anatomy_data : xr.DataArray
            Primary anatomical dataset.
        spec_data_list : list, optional
            Spectroscopic datasets to overlay.

        Returns
        -------
        list
            ``UnifiedPlotData`` objects ready for rendering.
        """
        plot_data_list = []

        # Add anatomical data first (as base)
        anatomy_plot_data = self._prepare_dataarray_plot_data(
            anatomy_data, {}, False, "anatomy"
        )
        plot_data_list.append(anatomy_plot_data)

        # Add spectroscopic overlays
        if spec_data_list:
            for i, spec_data in enumerate(spec_data_list):
                if isinstance(spec_data, xr.DataArray):
                    spec_plot_data = self._prepare_dataarray_plot_data(
                        spec_data, {}, True, f"spec_{i}"
                    )
                elif isinstance(spec_data, xr.Dataset):
                    spec_plot_data = self._prepare_dataset_plot_data(
                        spec_data, {}, True, f"spec_{i}"
                    )
                else:
                    continue

                plot_data_list.append(spec_plot_data)

        return plot_data_list

    def _clear_click_position(self):
        """Clear the click position so user needs to click again to see temporal plot."""
        self._click_positions.clear()

    def _translate_crosshair_coordinates(
        self, current_pos, old_perspective, new_perspective
    ):
        """Translate crosshair coordinates when the view changes.

        Parameters
        ----------
        current_pos : dict
            Current coordinate mapping.
        old_perspective : str
            Previous slicing dimension.
        new_perspective : str
            Target slicing dimension.

        Returns
        -------
        dict
            Coordinate mapping aligned with the new perspective.
        """
        if old_perspective == new_perspective:
            return current_pos

        logger.debug(" === COORDINATE TRANSLATION ===")
        print(f"Translating crosshair from {old_perspective} to {new_perspective}")
        logger.debug(" Input position: {current_pos}")

        # Start with the current position - anatomical coordinates stay the
        # same
        translated_pos = current_pos.copy()

        # Ensure all spatial dimensions exist and are within bounds
        spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
        for dim_name in spatial_dims:
            if dim_name in self.widgets and hasattr(self.widgets[dim_name], "max"):
                max_val = self.widgets[dim_name].max
                if dim_name in translated_pos:
                    old_val = translated_pos[dim_name]
                    # Clamp to valid range
                    translated_pos[dim_name] = max(0, min(int(old_val), max_val))
                    if translated_pos[dim_name] != old_val:
                        logger.debug(
                            f"Clamped {dim_name}: {old_val} -> {translated_pos[dim_name]} (max: {max_val})"
                        )
                else:
                    # If dimension is missing, use current slider value
                    current_slider_val = self.widgets[dim_name].value
                    translated_pos[dim_name] = current_slider_val
                    logger.debug(" Added missing {dim_name}: {current_slider_val}")

        # IMPORTANT: For perspective change, we need to ensure crosshair coordinates
        # are reasonable for the NEW viewing dimensions, not just clamped to slider bounds
        # The slider bounds may be larger than the actual data dimensions in
        # the new view
        logger.debug(": Pre-validation translated position: {translated_pos}")

        # We'll validate during rendering when we have access to the actual
        # data dimensions
        logger.debug(
            " Translated position (will validate during render): {translated_pos}"
        )
        return translated_pos

    def _onclick_spatial(self, event):
        """Simplified spatial click handling: update only displayed dimension sliders and show temporal plot."""

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
            # The temporal plot will automatically update via the normal plot refresh

    def _onclick_spectral(self, event):
        """Override spectral click handling to update temporal dimension slider."""
        if (
            not hasattr(self.update_function, "ax_spec")
            or event.inaxes != self.update_function.ax_spec
        ):
            return

        # Handle temporal axis clicking for anatomical data
        if (
            hasattr(self.update_function, "_temporal_axis_data")
            and event.xdata is not None
            and self._plot_data_list
        ):
            axis_data = self.update_function._temporal_axis_data

            # Find the closest point in the temporal axis
            try:
                closest_idx = np.argmin(np.abs(axis_data - event.xdata))

                # Find the secondary plot dimension widget for the base
                # anatomical data
                base_plot_data = self._plot_data_list[
                    0
                ]  # First plot data is usually the base anatomy
                secondary_dim_widget_name = (
                    f"{base_plot_data.widget_prefix}_secondary_plot_dim"
                )

                if secondary_dim_widget_name in self.widgets:
                    secondary_dim_name = self.widgets[secondary_dim_widget_name].value

                    # Update the corresponding dimension slider
                    if secondary_dim_name in self.widgets:
                        widget = self.widgets[secondary_dim_name]
                        if hasattr(widget, "value") and hasattr(widget, "max"):
                            new_val = max(0, min(int(closest_idx), widget.max))
                            # Set flag to prevent feedback loop
                            self._updating_from_click = True
                            try:
                                logger.debug(
                                    f"Updating {secondary_dim_name} slider from spectral click: {widget.value} -> {new_val}"
                                )
                                widget.value = new_val
                            finally:
                                self._updating_from_click = False

                            # Trigger plot update
                            if self.update_function and self.widgets:
                                current_kwargs = {
                                    k: v.value
                                    for k, v in self.widgets.items()
                                    if hasattr(v, "value")
                                }
                                self.update_function(**current_kwargs)

            except (IndexError, ValueError) as e:
                logger.debug(
                    f"Error updating temporal dimension from spectral click: {e}"
                )

        # Fall back to parent spectral click handling for other cases
        else:
            super()._onclick_spectral(event)

    def _plot_anatomical_temporal_signal(
        self, plot_data, signal_data, axis_data, xlabel, is_base, kwargs
    ):
        """Override temporal signal plotting to handle range adjustment and index highlighting."""
        if signal_data is None or not hasattr(self.update_function, "ax_spec"):
            return

        # Handle mean ± std case
        if isinstance(signal_data, tuple) and len(signal_data) == 2:
            signal_mean, signal_std = signal_data

            # Apply data part transformation and squeeze to 1D
            spec_part_widget = kwargs.get(
                f"{plot_data.widget_prefix}_spec_part", "Magnitude"
            )
            spec_part_val = (
                spec_part_widget if isinstance(spec_part_widget, str) else "Magnitude"
            )

            # Properly squeeze multi-dimensional arrays to 1D
            mean_values = part_func(
                input_data=signal_mean.values, signal_part=spec_part_val
            ).squeeze()
            std_values = part_func(
                input_data=signal_std.values, signal_part=spec_part_val
            ).squeeze()

            # Ensure 1D
            if mean_values.ndim > 1:
                mean_values = mean_values.flatten()
            if std_values.ndim > 1:
                std_values = std_values.flatten()

            # Handle subtraction if requested
            subtract_mode = kwargs.get("subtract_overlay", False)
            percent_mode = kwargs.get("percent_difference", False)

            if not is_base and (subtract_mode or percent_mode):
                if hasattr(self, "_cached_base_temporal_signal"):
                    mean_values = self._compute_overlay_difference(
                        mean_values,
                        self._cached_base_temporal_signal,
                        percent=percent_mode,
                    )
                    # For difference, std is combined std
                    if hasattr(self, "_cached_base_temporal_std"):
                        std_values = np.sqrt(
                            std_values**2 + self._cached_base_temporal_std**2
                        )
            elif is_base:
                # Store base signal for potential subtraction
                self._cached_base_temporal_signal = mean_values.copy()
                self._cached_base_temporal_std = std_values.copy()

            # Check if we should skip plotting the base signal (when in subtraction mode)
            # We still process it above to cache it, but don't display it
            should_plot = True
            if is_base and (subtract_mode or percent_mode):
                should_plot = False

            if should_plot:
                # Plot styling
                color = "blue" if is_base else "red"
                if not is_base and (subtract_mode or percent_mode):
                    alpha = 1.0
                    label = (
                        "% Difference"
                        if percent_mode
                        else "Difference (Overlay - Base)"
                    )
                else:
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

            # Store axis and signal data for position indicators (only for base
            # plot)
            if is_base:
                self.update_function._temporal_axis_data = axis_data
                self.update_function._temporal_signal_data = mean_values

        else:
            # Regular signal plotting
            # Apply data part transformation
            spec_part_widget = kwargs.get(
                f"{plot_data.widget_prefix}_spec_part", "Magnitude"
            )
            spec_part_val = (
                spec_part_widget if isinstance(spec_part_widget, str) else "Magnitude"
            )

            # Properly squeeze multi-dimensional arrays to 1D
            signal_values = part_func(
                input_data=signal_data.values, signal_part=spec_part_val
            ).squeeze()

            # Ensure 1D - if still multi-dimensional, check if it's multi-model
            is_multi_model = False
            if signal_values.ndim > 1:
                if (
                    signal_values.ndim == 2
                    and (
                        signal_values.shape[0] == len(axis_data)
                        or signal_values.shape[1] == len(axis_data)
                    )
                    and hasattr(signal_data, "coords")
                    and "model_type" in signal_data.coords
                ):
                    is_multi_model = True
                else:
                    logger.debug(
                        f" Signal still multi-dimensional ({signal_values.shape}), flattening..."
                    )
                    signal_values = signal_values.flatten()
            else:
                # If it is 1D, we still might have a single-entry model_type coordinate
                if (
                    hasattr(signal_data, "coords")
                    and "model_type" in signal_data.coords
                ):
                    # If there's only one model, we can treat it as multi-model to get the label
                    is_multi_model = True

            logger.debug(f" Final signal shape for plotting: {signal_values.shape}")
            logger.debug(f" Axis data shape: {axis_data.shape}")

            # Handle subtraction if requested
            subtract_mode = kwargs.get("subtract_overlay", False)
            percent_mode = kwargs.get("percent_difference", False)

            if not is_base and (subtract_mode or percent_mode):
                if hasattr(self, "_cached_base_temporal_signal"):
                    # Subtraction only supported for single-model for now
                    if not is_multi_model:
                        signal_values = self._compute_overlay_difference(
                            signal_values,
                            self._cached_base_temporal_signal,
                            percent=percent_mode,
                        )
            elif is_base:
                # Store base signal for potential subtraction
                if not is_multi_model:
                    self._cached_base_temporal_signal = signal_values.copy()

            # Check if we should skip plotting the base signal (when in subtraction mode)
            # We still process it above to cache it, but don't display it
            should_plot = True
            if is_base and (subtract_mode or percent_mode):
                should_plot = False

            if should_plot:
                # Plot styling
                color = "blue" if is_base else "red"
                if not is_base and (subtract_mode or percent_mode):
                    alpha = 1.0
                    label_prefix = (
                        "% Difference"
                        if percent_mode
                        else "Difference (Overlay - Base)"
                    )
                else:
                    alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
                    label_prefix = "Temporal Signal" if is_base else "Overlay Signal"

                if is_multi_model:
                    model_names = signal_data.coords["model_type"].values
                    if signal_values.ndim == 1:
                        # Only one model actually returned, or was already 1D
                        self.update_function.ax_spec.plot(
                            axis_data,
                            signal_values,
                            label=f"{label_prefix} ({model_names[0]})",
                            color=color,
                            alpha=alpha,
                            linewidth=self.processed_plot_params.get("linewidth", 2.0),
                        )
                    else:
                        for i, m_name in enumerate(model_names):
                            # Extract the curve for this model
                            if signal_values.shape[0] == len(model_names):
                                curve = signal_values[i, :]
                            else:
                                curve = signal_values[:, i]

                            # Use different line styles for multiple models if it's the same color
                            linestyles = ["-", "--", ":", "-."]
                            self.update_function.ax_spec.plot(
                                axis_data,
                                curve,
                                label=f"{label_prefix} ({m_name})",
                                color=color,
                                alpha=alpha,
                                linestyle=linestyles[i % len(linestyles)],
                                linewidth=self.processed_plot_params.get(
                                    "linewidth", 2.0
                                ),
                            )
                else:
                    self.update_function.ax_spec.plot(
                        axis_data,
                        signal_values,
                        label=label_prefix,
                        color=color,
                        alpha=alpha,
                        linewidth=self.processed_plot_params.get("linewidth", 2.0),
                    )

        # Store axis and signal data for position indicators (only for base
        # plot)
        if is_base:
            self.update_function._temporal_axis_data = axis_data
            if isinstance(signal_data, tuple):
                self.update_function._temporal_signal_data = mean_values
            else:
                self.update_function._temporal_signal_data = signal_values

        # Set labels
        self.update_function.ax_spec.set_xlabel(xlabel)
        self.update_function.ax_spec.set_ylabel(f"{spec_part_val} Intensity")

        # Apply synced y-axis limits if requested (overrides automatic adjustments below)
        sync_active = False
        if (
            hasattr(self.update_function, "_synced_ylim")
            and self.update_function._synced_ylim is not None
        ):
            self.update_function.ax_spec.set_ylim(self.update_function._synced_ylim)
            sync_active = True

        # Add legend
        if self.processed_plot_params.get("show_legend", True):
            self.update_function.ax_spec.legend()

        # Add range adjustment for Min/Max mode
        if (
            is_base
            and not sync_active
            and hasattr(self.update_function, "ax_spec")
            and self.update_function.ax_spec
        ):
            # Get range mode
            range_mode = kwargs.get(f"{plot_data.widget_prefix}_range_mode", "Min/Max")

            # Adjust y-axis range for Min/Max mode
            if range_mode == "Min/Max":
                try:
                    # Get current y-axis limits
                    ymin, ymax = self.update_function.ax_spec.get_ylim()
                    y_range = ymax - ymin

                    # Expand by ±10%
                    expanded_ymin = ymin - 0.1 * y_range
                    expanded_ymax = ymax + 0.1 * y_range

                    self.update_function.ax_spec.set_ylim(expanded_ymin, expanded_ymax)
                except Exception as e:
                    warnings.warn(f"Failed to adjust y-axis range: {e}")

        # Add index highlighting and reference lines (always run if is_base)
        if (
            is_base
            and hasattr(self.update_function, "ax_spec")
            and self.update_function.ax_spec
        ):
            # Add thin gray horizontal line at y=0
            self.update_function.ax_spec.axhline(
                y=0, color="gray", linewidth=0.5, alpha=0.7, zorder=0
            )

            # Add highlighting for selected index in secondary dimension
            self._highlight_selected_index(kwargs, axis_data)

    def _highlight_selected_index(self, kwargs, axis_data):
        """Highlight the currently selected index in the secondary data dimension."""
        try:
            if (
                not hasattr(self.update_function, "ax_spec")
                or not self.update_function.ax_spec
            ):
                return

            # Find the base plot data to get secondary_plot_dim widget
            base_plot_data = None
            if hasattr(self, "_plot_data_list") and self._plot_data_list:
                base_plot_data = self._plot_data_list[0]

            if not base_plot_data:
                return

            # Get secondary plot dimension
            secondary_dim_widget_name = (
                f"{base_plot_data.widget_prefix}_secondary_plot_dim"
            )
            secondary_dim_name = kwargs.get(secondary_dim_widget_name)

            if secondary_dim_name and secondary_dim_name in kwargs:
                selected_index = kwargs[secondary_dim_name]

                # Check if we have the right axis data length
                if selected_index < len(axis_data):
                    # Add vertical line at selected index
                    x_position = axis_data[selected_index]

                    # Remove previous highlight if it exists
                    if hasattr(self.update_function, "index_highlight_line"):
                        try:
                            self.update_function.index_highlight_line.remove()
                        except BaseException:
                            pass

                    # Add new highlight line
                    self.update_function.index_highlight_line = (
                        self.update_function.ax_spec.axvline(
                            x_position,
                            color="black",
                            linestyle="-",
                            alpha=0.7,
                            linewidth=0.5,
                            label=f"Selected {secondary_dim_name}={selected_index}",
                            zorder=0,
                        )
                    )

                    # Update legend if it exists
                    if self.processed_plot_params.get("show_legend", True):
                        self.update_function.ax_spec.legend()

        except Exception as e:
            # Don't let highlighting errors break the plot
            warnings.warn(f"Failed to highlight selected index: {e}")

    def plot_with_spectroscopic_overlay(
        self, anatomy_data, spec_data_list, plot_params=None, **kwargs
    ):
        """Main method for plotting anatomy with spectroscopic overlays.

        This provides the key functionality that makes AnatomyPlotter
        comparable to spectroscopyPlotter in terms of cross-modal capabilities.

        Parameters
        ----------
        anatomy_data : xr.DataArray
            Primary anatomical dataset to display.
        spec_data_list : list
            Spectroscopic datasets to overlay.
        plot_params : dict, optional
            Plot parameter overrides.
        **kwargs
            Additional keyword arguments forwarded to the widget layer.
        """
        # Prepare mixed data
        mixed_plot_data = self._prepare_mixed_plot_data(anatomy_data, spec_data_list)

        # Use parent plotting functionality
        self._plot_data_list = mixed_plot_data

        # Set up widgets and rendering
        if plot_params is None:
            plot_params = self.processed_plot_params
        else:
            self.processed_plot_params = self._process_plot_params(plot_params)

        # Create widgets for mixed data
        controls_box, self.widgets = self._create_unified_widgets(
            mixed_plot_data, self.processed_plot_params, kwargs
        )

        # Set up update function
        def update_plot(**update_kwargs):
            self._render_unified_plot(mixed_plot_data, update_kwargs)

        self.update_function = update_plot
        output = widgets.interactive_output(update_plot, self.widgets)
        display(VBox([controls_box, output]))

        # Show initial plot
        update_plot(
            **{k: v.value for k, v in self.widgets.items() if hasattr(v, "value")}
        )

    def _render_spatial_component(self, plot_data, is_base, kwargs):
        """Render the spatial component of the plot data.

        Parameters
        ----------
        plot_data : UnifiedPlotData
            Plot data describing the spatial content to display.
        is_base : bool
            Whether the plot represents the primary dataset.
        kwargs : dict
            Widget state controlling the rendered view.
        """
        # Get spatial data to plot
        spatial_data = plot_data.spatial_data
        if spatial_data is None:
            return

        # Apply data slicing based on current slider values (using DataSelector helper)
        sliced_data = self._apply_spatial_slicing_using_selector(
            spatial_data, kwargs, plot_data.widget_prefix
        )

        # Transform to k-space if requested
        show_image_space = kwargs.get(
            f"{plot_data.widget_prefix}_show_image_space", True
        )
        if not show_image_space:  # If show_image_space is False, show k-space
            sliced_data = self._apply_kspace_transform(sliced_data)

        # Apply data part transformation
        spec_part_val = kwargs.get(f"{plot_data.widget_prefix}_spec_part", "Magnitude")

        # Performance optimization: Use cached transformation
        plot_data_values = self._get_cached_transform(sliced_data, spec_part_val)

        # Get spatial dimensions for correct ordering
        if (
            plot_data.is_overlay
            and hasattr(self, "_base_spatial_dims")
            and self._base_spatial_dims
        ):
            spatial_dims = self._base_spatial_dims
            base_dims = self._base_spatial_dims
        else:
            spatial_dims = self._get_spatial_dims_using_selector(
                spatial_data, kwargs, plot_data.widget_prefix
            )
            base_dims = None
            # Store base spatial dims for consistent overlay alignment
            if not plot_data.is_overlay:
                self._base_spatial_dims = spatial_dims

        # Use DataSelector to prepare 2D data with correct orientation
        plot_data_values = self.data_selector.prepare_2d_spatial_data(
            plot_data_values,
            spatial_dims=spatial_dims,
            is_overlay=plot_data.is_overlay,
            base_spatial_dims=base_dims,
        )

        # Convert to numpy array for imshow
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
            f"{plot_data.widget_prefix}_cmap",
            self.processed_plot_params.get("colormap", "bone"),
        )

        # When subtracting, use full opacity and different colormap default
        if not is_base and (subtract_mode or percent_mode):
            alpha = 1.0
            if percent_mode:
                cmap = kwargs.get(f"{plot_data.widget_prefix}_cmap", "RdBu_r")
        else:
            alpha = 1.0 if is_base else kwargs.get("overlay_alpha", 0.5)
        interpolation = kwargs.get(
            f"{plot_data.widget_prefix}_interpolate_image",
            False,
        )

        if interpolation:
            interpolation_str = "spline36"
        else:
            interpolation_str = "nearest"

        # Performance optimization: Use cached min/max values
        min_sig, max_sig = self._get_cached_minmax(spatial_data, spec_part_val)

        vmin, vmax = self._vmin_vmax_range_check(
            range_mode=kwargs.get(f"{plot_data.widget_prefix}_range_mode", "Min/Max"),
            data_to_plot_spatial=plot_data_values,
            min_sig=min_sig,
            max_sig=max_sig,
            image_source_changed=True,
            plot_data=plot_data,
        )

        # Calculate extent for proper click handling and tick labels
        ny, nx = plot_data_values.shape[-2:]
        coord_system = kwargs.get("coord_system", "Auto")
        extent, origin, xlabel, ylabel = self._calculate_coordinate_extent(
            spatial_data,
            spatial_dims,
            plot_data_values.shape,
            coord_system=coord_system,
            data_obj=self.anatomy_obj,
            kwargs=kwargs,
        )

        # Plot the image
        im = self.update_function.ax_img.imshow(
            plot_data_values,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation=interpolation_str,
            origin=origin,
        )

        # Store vmin/vmax for secondary plot sync if requested
        sync_enabled = kwargs.get(f"{plot_data.widget_prefix}_sync_y_range", False)
        if sync_enabled:
            self.update_function._synced_ylim = (vmin, vmax)
        elif is_base:
            if hasattr(self.update_function, "_synced_ylim"):
                self.update_function._synced_ylim = None

        # Store references for click handling and colorbars
        if is_base:
            self.update_function.last_im = im
            # Store click handling info
            if len(spatial_dims) >= 2:
                self.update_function._current_plot_dims = (
                    spatial_dims[0],
                    spatial_dims[1],
                )
                self.update_function._current_data_shape = (ny, nx)
                self.update_function._current_extent = extent
                self.update_function._current_origin = origin

            # Set axis labels
            self.update_function.ax_img.set_xlabel(xlabel)
            self.update_function.ax_img.set_ylabel(ylabel)
        else:
            # Store overlay image for dual colorbar
            if not hasattr(self.update_function, "overlay_ims"):
                self.update_function.overlay_ims = []
            self.update_function.overlay_ims.append(im)
