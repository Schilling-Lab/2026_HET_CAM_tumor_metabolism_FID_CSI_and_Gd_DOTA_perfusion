"""Interactive mask creation with simplified coordinate handling."""

from typing import Any, Dict, Optional

import ipywidgets as widgets
import numpy as np
import xarray as xr
from IPython.display import clear_output, display

from ...utils.system.logging import (
    debug_log,
)
from .lasso_handler import LassoHandler
from .mask_container import MaskContainer
from .visualization_sync import VisualizationSync

DEBUG_LOG = True
DEBUG_FILE = "/tmp/create_mask_debug.log"


def debug_log_mask_creator(message, enabled=DEBUG_LOG):
    """Log mask creation debug messages to /tmp/create_mask_debug.log with timestamp."""
    debug_log(
        message=message,
        file_name=DEBUG_FILE,
        write_to_file=enabled,
    )


class MaskCreator:
    """
    Interactive mask creation with identical coordinate systems.

    Replaces the complex create_mask method with a clean, modular approach
    that eliminates coordinate transforms by using identical coordinates.
    """

    def __init__(
        self,
        data_source,
    ):
        """
        Initialize mask creator with preprocessing support.

        Parameters
        ----------
        data_source : MRIExperiment-like object
            Data source object with MRI data
        """
        self.data_source = data_source

        # Core components
        self.lasso_handler: Optional[LassoHandler] = None
        self.viz_sync = VisualizationSync()

        # Mask storage - identical coordinates to source data
        self.masks: Dict[str, MaskContainer] = {}

        # UI state
        self.active_mask_name: Optional[str] = None
        self.ui_widgets: Dict[str, Any] = {}
        self.plot_handles: Dict[str, Any] = {}
        self._layout_mode: str = "horizontal"

        # ENHANCED: Preprocessing tracking for accurate volume calculations
        self._preprocessing_applied: bool = False
        self._original_plot_params: Dict = {}
        self._original_slider_defaults: Dict = {}

    def create_interactive_session(
        self,
        input_data: Optional[xr.DataArray] = None,
        plot_params: Optional[Dict] = None,
        layout_mode: str = "horizontal",
        **slider_defaults,
    ) -> None:
        """
        Launch interactive masking session with preprocessing support.

        Parameters
        ----------
        input_data : xr.DataArray, optional
            Data to display for masking. If provided, this should be preprocessed data
            that matches exactly what the user sees on screen.
        plot_params : dict, optional
            Parameters passed to the plot method. Used to apply same preprocessing
            as the display when input_data is None.
        layout_mode : str, optional
            Layout arrangement for controls; accepts "horizontal" (default) or "vertical".
        **slider_defaults : dict
            Default slider positions passed to plot method.
        """
        try:
            # ENHANCED: Get input data with preprocessing support
            data_array = self._get_preprocessed_data(
                input_data, plot_params, slider_defaults
            )
            if data_array is None:
                raise ValueError("Could not obtain input data for masking")

            # Store preprocessing context for volume calculations
            self._preprocessing_applied = input_data is not None or (
                plot_params and self._has_preprocessing_params(plot_params)
            )
            self._original_plot_params = plot_params or {}
            self._original_slider_defaults = slider_defaults

            debug_log_mask_creator(
                message=f"🔧 Session setup - preprocessing applied: {self._preprocessing_applied}"
            )
            debug_log_mask_creator(
                message=f"📊 Final data for masking - shape: {data_array.shape}, dims: {data_array.dims}"
            )

            # Normalize layout preference for UI assembly
            layout_mode = (
                str(layout_mode).strip().lower()
                if layout_mode is not None
                else "horizontal"
            )
            if layout_mode not in {"horizontal", "vertical"}:
                layout_mode = "horizontal"
            self._layout_mode = layout_mode

            # Initialize lasso handler with preprocessed data
            self.lasso_handler = LassoHandler(data_array)

            # Set up plotting system with preprocessed data
            self._setup_plot_system(data_array, plot_params, slider_defaults)

            # Create UI
            self._create_mask_ui()

            # Display everything
            self._display_session()

        except Exception:
            raise

    def _get_input_data(
        self, input_data: Optional[xr.DataArray]
    ) -> Optional[xr.DataArray]:
        """Get input data from various sources, preferring preprocessed data for consistency."""
        # CRITICAL FIX: Prefer explicitly provided input_data (likely preprocessed)
        # This ensures the mask creator works with the same data the user sees on screen
        if input_data is not None:
            debug_log_mask_creator(
                message=f"✅ Using provided input_data - shape: {input_data.shape}, dims: {input_data.dims}"
            )
            return input_data

        debug_log_mask_creator(
            message="⚠️  No input_data provided, falling back to raw data sources"
        )
        debug_log_mask_creator(
            message="⚠️  This may cause coordinate mismatches with preprocessed display data"
        )

        # Try multiple data sources for backward compatibility
        # NOTE: These are raw data sources and may not match what's displayed to user
        candidates = ["seq2d_rps_xarray", "data_array", "reco_array"]

        for candidate in candidates:
            if hasattr(self.data_source, candidate):
                data = getattr(self.data_source, candidate)
                if data is not None:
                    debug_log_mask_creator(
                        message=f"📊 Using fallback data source '{candidate}' - shape: {data.shape}, dims: {data.dims}"
                    )
                    return data

        debug_log_mask_creator(message="❌ No valid data sources found")
        return None

    def _get_preprocessed_data(
        self,
        input_data: Optional[xr.DataArray],
        plot_params: Optional[Dict],
        slider_defaults: Dict,
    ) -> Optional[xr.DataArray]:
        """
        Get preprocessed data that matches what the user sees on screen.

        This is critical for ensuring masks are created on the same data
        as what's displayed, especially for multi-dimensional data.
        """
        # If input_data is provided, assume it's already preprocessed and use it directly
        if input_data is not None:
            debug_log_mask_creator(
                message=f"✅ Using provided preprocessed data - shape: {input_data.shape}"
            )
            return input_data

        # If no input_data provided, try to get and preprocess raw data
        raw_data = self._get_input_data(None)  # Get from fallback sources
        if raw_data is None:
            return None

        # Apply preprocessing if parameters suggest it's needed
        if plot_params and self._has_preprocessing_params(plot_params):
            debug_log_mask_creator(
                message="🔧 Applying preprocessing to match display data"
            )
            return self._apply_mask_preprocessing(
                raw_data, plot_params, slider_defaults
            )
        else:
            debug_log_mask_creator(
                message="📊 Using raw data (no preprocessing params detected)"
            )
            return raw_data

    def _has_preprocessing_params(self, plot_params: Dict) -> bool:
        """Check if plot_params contains preprocessing directives."""
        if not plot_params:
            return False

        # Check for channel combination or other preprocessing flags
        preprocessing_keys = [
            "combine_channels",
            "_combine_channels",
            "average_reps",
            "_average_reps",
            "spec_part",
            "_spec_part",
        ]

        return any(key in plot_params for key in preprocessing_keys)

    def _apply_mask_preprocessing(
        self, data_array: xr.DataArray, plot_params: Dict, slider_defaults: Dict
    ) -> xr.DataArray:
        """
        Apply preprocessing to match what the plotter displays.

        This replicates the key preprocessing steps from AnatomyPlotter
        to ensure coordinate consistency.
        """
        try:
            processed_data = data_array.copy()

            # Apply channel combination if requested
            combine_channels = plot_params.get(
                "combine_channels", False
            ) or plot_params.get("_combine_channels", False)
            if (
                combine_channels
                and "channel_idx" in processed_data.dims
                and processed_data.sizes["channel_idx"] > 1
            ):

                debug_log_mask_creator(
                    message=f"🔧 Applying channel RSS combination - before: {processed_data.shape}"
                )
                processed_data = self._combine_channels_rss(processed_data)
                debug_log_mask_creator(
                    message=f"🔧 After channel combination: {processed_data.shape}"
                )

            # Apply repetition averaging if requested
            average_reps = plot_params.get("average_reps", False) or plot_params.get(
                "_average_reps", False
            )
            if (
                average_reps
                and "repetition_idx" in processed_data.dims
                and processed_data.sizes["repetition_idx"] > 1
            ):

                debug_log_mask_creator(
                    message=f"🔧 Applying repetition averaging - before: {processed_data.shape}"
                )
                processed_data = processed_data.mean(dim="repetition_idx")
                debug_log_mask_creator(
                    message=f"🔧 After repetition averaging: {processed_data.shape}"
                )

            # Apply other transformations as needed
            spec_part = plot_params.get("spec_part") or plot_params.get("_spec_part")
            if spec_part and isinstance(spec_part, str) and spec_part != "magnitude":
                debug_log_mask_creator(
                    message=f"🔧 Applying spec_part transformation: {spec_part}"
                )
                processed_data = self._apply_spec_part_transform(
                    processed_data, spec_part
                )

            debug_log_mask_creator(
                message=f"✅ Preprocessing complete - final shape: {processed_data.shape}"
            )
            return processed_data

        except Exception as e:
            debug_log_mask_creator(
                message=f"❌ Preprocessing failed: {e}, using original data"
            )
            return data_array

    def _combine_channels_rss(self, data_array: xr.DataArray) -> xr.DataArray:
        """Combine channels using root-sum-of-squares.

        Notes
        -----
        This method now delegates to the general-purpose channel combination utility.
        """
        if "channel_idx" not in data_array.dims:
            return data_array

        try:
            # Use the general-purpose channel combination utility
            from hypermri.utils.mri.channels import combine_channels

            return combine_channels(data_array, method="rss")

        except Exception as e:
            debug_log_mask_creator(message=f"❌ Channel RSS failed: {e}")
            return data_array

    def _apply_spec_part_transform(
        self, data_array: xr.DataArray, spec_part: str
    ) -> xr.DataArray:
        """Apply spectral part transformation."""
        try:
            if spec_part == "real":
                return data_array.real
            elif spec_part == "imag":
                return data_array.imag
            elif spec_part == "magnitude":
                return np.abs(data_array)
            elif spec_part == "phase":
                return np.angle(data_array)
            else:
                return data_array
        except Exception:
            return data_array

    def _setup_plot_system(
        self, data_array: xr.DataArray, plot_params: Dict, slider_defaults: Dict
    ):
        """Set up the plotting system for masking."""

        # Set default slice to middle slice if not provided
        if "slice_idx" not in slider_defaults and "slice_idx" in data_array.dims:
            max_slice = data_array.sizes["slice_idx"] - 1
            middle_slice = max_slice // 2
            slider_defaults["slice_idx"] = middle_slice

        # Call the plot method with return_handles=True
        try:
            plot_result = self.data_source.plot(
                plot_params=plot_params, return_handles=True, **slider_defaults
            )

            # Unpack plot handles (now expects 6 elements after widget observer fix)
            (
                plot_controls,
                plot_output_area,
                run_interactive_update,
                initial_kwargs,
                update_plot_handle,
                widgets_dict,
            ) = plot_result

            # Store handles
            self.plot_handles = {
                "controls": plot_controls,
                "output_area": plot_output_area,
                "run_update": run_interactive_update,
                "initial_kwargs": initial_kwargs,
                "update_plot": update_plot_handle,
                "widgets_dict": widgets_dict,
            }

            # --- NEU: Widget-Observer für alle relevanten Dimensionen ---
            plotter_widgets = getattr(self.data_source.plotter, "widgets", {})
            for dim in self.lasso_handler.data_array.dims:
                if dim in plotter_widgets:
                    plotter_widgets[dim].observe(
                        lambda change, d=dim: self._update_lasso_plot_state(),
                        names="value",
                    )
        except Exception:
            raise

    def _create_mask_ui(self):
        """Create the mask creation UI widgets."""

        # ENHANCED: Build dropdown options from existing masks
        dropdown_options = ["Create New..."]

        # Add existing masks to dropdown
        existing_masks = list(self.masks.keys())
        if existing_masks:
            # Sort masks, but keep 'Create New...' at the end
            sorted_masks = sorted(existing_masks)
            dropdown_options = sorted_masks + ["Create New..."]

        # Mask selection dropdown
        self.ui_widgets["mask_selector"] = widgets.Dropdown(
            options=dropdown_options,
            description="Active Mask:",
            style={"description_width": "initial"},
        )

        # New mask name input
        self.ui_widgets["new_mask_name"] = widgets.Text(
            placeholder="Enter New Mask Name", description="New Name:", disabled=False
        )

        # Color picker
        self.ui_widgets["color_picker"] = widgets.ColorPicker(
            concise=False, description="Color:", value="red"
        )

        # Alpha slider for mask transparency
        self.ui_widgets["alpha_slider"] = widgets.FloatSlider(
            value=0.4,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Alpha:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            layout=widgets.Layout(width="200px"),
        )

        # Volume display widget (positioned next to buttons)
        self.ui_widgets["volume_display"] = widgets.HTML(
            value="<b>Volume: 0.0 mL</b>",
            description="",
            layout=widgets.Layout(
                width="120px",
                margin="5px",
                padding="5px",
                border="1px solid #ccc",
                border_radius="3px",
            ),
        )

        # Buttons
        self.ui_widgets["activate_btn"] = widgets.Button(
            description="Activate / Create", button_style="info"
        )

        self.ui_widgets["finalize_btn"] = widgets.Button(
            description="Finalize Active Mask", button_style="success", disabled=True
        )

        self.ui_widgets["clear_lasso_btn"] = widgets.Button(
            description="Clear Last Lasso", button_style="warning", disabled=True
        )

        self.ui_widgets["clear_mask_btn"] = widgets.Button(
            description="Clear Active Mask", button_style="danger", disabled=True
        )

        self.ui_widgets["done_btn"] = widgets.Button(
            description="Done", button_style="primary"
        )

        # Status output
        self.ui_widgets["status_output"] = widgets.Output(
            layout=widgets.Layout(height="120px", overflow_y="auto")
        )

        # Connect callbacks
        self._connect_ui_callbacks()

    def _connect_ui_callbacks(self):
        """Connect UI widget callbacks."""

        # Dropdown change
        self.ui_widgets["mask_selector"].observe(
            self._on_dropdown_change, names="value"
        )

        # Alpha slider change
        self.ui_widgets["alpha_slider"].observe(self._on_alpha_change, names="value")

        # Button clicks
        self.ui_widgets["activate_btn"].on_click(self._on_activate_click)
        self.ui_widgets["finalize_btn"].on_click(self._on_finalize_click)
        self.ui_widgets["clear_lasso_btn"].on_click(self._on_clear_lasso_click)
        self.ui_widgets["clear_mask_btn"].on_click(self._on_clear_mask_click)
        self.ui_widgets["done_btn"].on_click(self._on_done_click)

    def _on_dropdown_change(self, change):
        """Handle mask selection change."""
        if change["new"] == "Create New...":
            self.ui_widgets["new_mask_name"].layout.display = "flex"
            self.ui_widgets["new_mask_name"].disabled = False
            # Reset alpha slider to default when creating new mask
            self.ui_widgets["alpha_slider"].value = 0.4
            self._update_volume_display()  # Reset volume display
        else:
            self.ui_widgets["new_mask_name"].layout.display = "none"
            self.ui_widgets["new_mask_name"].disabled = True

            # Update color picker and alpha slider to match selected mask
            if change["new"] in self.masks:
                mask_container = self.masks[change["new"]]
                self.ui_widgets["color_picker"].value = mask_container.color
                self.ui_widgets["alpha_slider"].value = mask_container.alpha
                # Set active mask and update overlay and volume
                self.active_mask_name = change["new"]
                self._trigger_plot_update()
                self._update_volume_display()  # Update volume when switching masks

    def _on_alpha_change(self, change):
        """Handle alpha slider change."""
        new_alpha = change["new"]

        # Update active mask alpha if there is one
        if self.active_mask_name and self.active_mask_name in self.masks:
            mask_container = self.masks[self.active_mask_name]
            mask_container.alpha = new_alpha

            # Trigger plot update to show the alpha change
            self._trigger_plot_update()
        else:
            pass

    def _on_activate_click(self, b):
        """Handle activate/create button click."""
        with self.ui_widgets["status_output"]:
            clear_output(wait=True)

            try:
                mask_name = self._get_mask_name_from_ui()
                if not mask_name:
                    return

                is_new = mask_name not in self.masks
                color = self.ui_widgets["color_picker"].value
                alpha = self.ui_widgets["alpha_slider"].value

                if is_new:
                    self._create_new_mask(mask_name, color, alpha)
                else:
                    self._activate_existing_mask(mask_name, color, alpha)

                self._update_ui_state()
                self._trigger_plot_update()
                self._update_volume_display()  # Update volume when mask is activated

            except Exception as e:
                print(f"Error in activate: {e}")

    def _get_mask_name_from_ui(self) -> Optional[str]:
        """Get mask name from UI, with validation."""
        if self.ui_widgets["mask_selector"].value == "Create New...":
            mask_name = self.ui_widgets["new_mask_name"].value.strip()
            if not mask_name:
                print("Please enter a name for the new mask.")
                return None
            if mask_name in self.masks:
                print(
                    f"Mask '{mask_name}' already exists. Select it from dropdown to edit."
                )
                return None
            return mask_name
        else:
            mask_name = self.ui_widgets["mask_selector"].value
            if mask_name not in self.masks:
                print(f"Error: Selected mask '{mask_name}' not found.")
                return None
            return mask_name

    def _create_new_mask(self, mask_name: str, color: str, alpha: float = 0.4):
        """Create a new mask container with enhanced slice tracking."""
        print(f"Creating new mask: '{mask_name}' (alpha: {alpha})")

        # Get template data for identical coordinates
        template_data = self.lasso_handler.data_array

        # Use the original MaskContainer with specified alpha
        mask_container = MaskContainer(
            name=mask_name,
            template_data=template_data,
            color=color,
            alpha=alpha,
        )

        # Store mask
        self.masks[mask_name] = mask_container
        self.active_mask_name = mask_name

        # Update dropdown options
        current_options = list(self.ui_widgets["mask_selector"].options)
        if mask_name not in current_options:
            current_options.append(mask_name)
            self.ui_widgets["mask_selector"].options = tuple(
                sorted([o for o in current_options if o != "Create New..."])
                + ["Create New..."]
            )
        self.ui_widgets["mask_selector"].value = mask_name

        # Set up lasso handler
        self._setup_lasso_for_active_mask()

        # Show only the active mask as overlay
        self.viz_sync.sync_mask_to_plotter(
            mask_container, self.data_source.plotter, show_only_active=True
        )

        print(f"✅ Created mask '{mask_name}' with identical coordinates")

    def _activate_existing_mask(self, mask_name: str, color: str, alpha: float = None):
        """Activate existing mask."""
        print(
            f"Activating mask: '{mask_name}' (alpha: {alpha if alpha is not None else 'unchanged'})"
        )

        mask_container = self.masks[mask_name]
        mask_container.color = color  # Update color if changed

        # Update alpha if provided
        if alpha is not None:
            mask_container.alpha = alpha

        self.active_mask_name = mask_name

        # Set up lasso handler
        self._setup_lasso_for_active_mask()

        # Show only the active mask as overlay
        self.viz_sync.sync_mask_to_plotter(
            mask_container, self.data_source.plotter, show_only_active=True
        )

        print(f"✅ Activated mask '{mask_name}'")

    def _setup_lasso_for_active_mask(self):
        """Set up lasso selector for the active mask."""
        if not self.active_mask_name or not self.lasso_handler:
            return

        mask_container = self.masks[self.active_mask_name]

        # Set active mask in lasso handler
        self.lasso_handler.set_active_mask(mask_container)

        # Set up volume update callback
        self.lasso_handler.set_volume_update_callback(self._update_volume_display)

        # Get plot axes
        update_plot = self.plot_handles.get("update_plot")
        if (
            not update_plot
            or not hasattr(update_plot, "ax_img")
            or not update_plot.ax_img
        ):
            print("Plot axes not ready. Triggering plot...")
            self.plot_handles["run_update"](**self.plot_handles["initial_kwargs"])

            if not hasattr(update_plot, "ax_img") or not update_plot.ax_img:
                print("ERROR: Cannot get plot axes")
                return

        # Set plot state for coordinate mapping
        self._update_lasso_plot_state()

        # Create lasso selector
        ax_img = update_plot.ax_img
        selector = self.lasso_handler.create_lasso_selector(
            ax_img, enhanced_visibility=True
        )

        # Store selector reference
        mask_container._lasso_selector = selector

        # Force canvas update
        if hasattr(ax_img.figure, "canvas"):
            ax_img.figure.canvas.draw()

    def _update_lasso_plot_state(self):
        """Update lasso handler with current plot state with enhanced slice capture."""
        if not self.lasso_handler:
            return

        # Get current plot state from update handle
        update_plot = self.plot_handles.get("update_plot")
        if not update_plot:
            return

        # Extract plot dimensions and slice selector
        plot_dims = getattr(update_plot, "current_plot_dims", None)
        current_selector = getattr(update_plot, "current_selector", None)

        # CRITICAL FIX: Get the plot extent for accurate coordinate mapping
        plot_extent = getattr(update_plot, "_current_extent", None)

        # ENHANCED: Build the most accurate slice selector possible
        final_selector = {}

        # Method 1: Start with widget values (most reliable for current state)
        if hasattr(self.data_source, "plotter") and hasattr(
            self.data_source.plotter, "widgets"
        ):
            if self.data_source.plotter.widgets:
                for widget_name, widget in self.data_source.plotter.widgets.items():
                    if (
                        hasattr(widget, "value")
                        and widget_name in self.lasso_handler.data_array.dims
                    ):
                        final_selector[widget_name] = widget.value

        # Method 2: Fill gaps with plot handle selector
        if current_selector:
            for key, value in current_selector.items():
                if (
                    key in self.lasso_handler.data_array.dims
                    and key not in final_selector
                ):
                    final_selector[key] = value

        # Method 3: Ensure critical dimensions exist
        data_dims = set(self.lasso_handler.data_array.dims)

        # Critical check for slice_idx
        if "slice_idx" not in final_selector and "slice_idx" in data_dims:
            # Try multiple fallback sources for slice_idx
            slice_sources = []

            # From initial kwargs
            if "slice_idx" in self.plot_handles.get("initial_kwargs", {}):
                slice_sources.append(
                    ("initial_kwargs", self.plot_handles["initial_kwargs"]["slice_idx"])
                )

            # From data array middle
            middle_slice = self.lasso_handler.data_array.sizes["slice_idx"] // 2
            slice_sources.append(("middle_slice", middle_slice))

            # Use the first available source
            if slice_sources:
                source, value = slice_sources[0]
                final_selector["slice_idx"] = value

        if plot_dims and final_selector:
            # Filter out spatial plot dimensions but keep all slicing dimensions
            spatial_dims = set(plot_dims)  # Current 2D plot dimensions

            # Keep all dimensions that are not the current spatial plot dimensions
            filtered_selector = {
                key: value
                for key, value in final_selector.items()
                if key in data_dims and key not in spatial_dims
            }

            # CRITICAL VALIDATION: Ensure slice_idx is preserved
            if "slice_idx" in final_selector and "slice_idx" not in filtered_selector:
                filtered_selector["slice_idx"] = final_selector["slice_idx"]

            # Final validation
            filtered_selector.get("slice_idx", "MISSING")

            # Update lasso handler with the robust selector
            self.lasso_handler.set_plot_state(plot_dims, filtered_selector, plot_extent)

        else:
            debug_log_mask_creator(
                message=f"❌ Cannot update lasso plot state - plot_dims={plot_dims}, selector keys={list(final_selector.keys()) if final_selector else 'NONE'}",
            )

    def _on_finalize_click(self, b):
        """Handle finalize button click."""
        with self.ui_widgets["status_output"]:
            clear_output(wait=True)

            if not self.active_mask_name or self.active_mask_name not in self.masks:
                print("No active mask to finalize.")
                return

            try:
                mask_container = self.masks[self.active_mask_name]

                # Calculate volume before finalizing
                try:
                    input_data = self._get_input_data(None)
                    if input_data is not None and mask_container.buffer is not None:
                        mask_data = mask_container.buffer.values
                        volume_ml = self._calculate_mask_volume(mask_data, input_data)

                        # Add volume to mask attributes
                        mask_container.buffer.attrs["volume_ml"] = float(volume_ml)
                        mask_container.buffer.attrs["volume_mm3"] = float(
                            volume_ml * 1000
                        )

                        # Also add pixel count for reference
                        pixel_count = int(np.sum(mask_data))
                        mask_container.buffer.attrs["pixel_count"] = pixel_count

                        print(f"📊 Volume: {volume_ml:.3f} mL ({pixel_count} pixels)")

                    else:
                        debug_log_mask_creator(
                            message=f"⚠️ Could not calculate volume for mask '{self.active_mask_name}' - input_data or mask buffer is None",
                        )

                except Exception as e:
                    debug_log_mask_creator(
                        message=f"❌ Failed to calculate volume for finalization: {e}",
                    )

                # Now finalize the mask (this will add finalized=True and timestamp)
                mask_container.finalize()

                print(f"✅ Mask '{self.active_mask_name}' finalized")
                self._update_ui_state()

            except Exception as e:
                print(f"Error finalizing mask: {e}")

    def _on_clear_lasso_click(self, b):
        """Handle clear lasso button click."""
        with self.ui_widgets["status_output"]:
            clear_output(wait=True)

            if not self.active_mask_name or not self.lasso_handler:
                print("No active mask to clear lasso for.")
                return

            try:
                # Remove visual patches
                mask_container = self.masks[self.active_mask_name]
                self.viz_sync.remove_last_visual_patches(mask_container, patch_count=2)

                # Undo last lasso from buffer
                success = self.lasso_handler.undo_last_lasso()

                if success:
                    print(f"✅ Cleared last lasso for '{self.active_mask_name}'")
                    self._trigger_plot_update()
                    self._update_volume_display()  # Update volume when lasso is cleared
                else:
                    print("No lasso to clear or undo failed")

                self._update_ui_state()

            except Exception as e:
                print(f"Error clearing lasso: {e}")

    def _on_clear_mask_click(self, b):
        """Handle clear mask button click."""
        with self.ui_widgets["status_output"]:
            clear_output(wait=True)

            if not self.active_mask_name or self.active_mask_name not in self.masks:
                print("No active mask to clear.")
                return

            try:
                mask_container = self.masks[self.active_mask_name]

                # Clear buffer and visual patches
                mask_container.clear_buffer()
                self.viz_sync.clear_all_visual_patches(mask_container)

                print(f"✅ Cleared mask '{self.active_mask_name}'")
                self._trigger_plot_update()
                self._update_ui_state()
                self._update_volume_display()  # Update volume when mask is cleared

            except Exception as e:
                print(f"Error clearing mask: {e}")

    def _on_done_click(self, b):
        """Handle done button click."""
        with self.ui_widgets["status_output"]:
            clear_output(wait=True)
            print("🏁 Masking session complete")

            # Disable all UI elements
            for widget_name, widget in self.ui_widgets.items():
                if hasattr(widget, "disabled") and widget_name != "status_output":
                    widget.disabled = True

            # Calculate and add volume attributes for all masks before finalizing
            try:
                input_data = self._get_input_data(None)
                if input_data is not None:
                    for mask_name, mask in self.masks.items():
                        try:
                            # Only calculate volume if not already present
                            if "volume_ml" not in mask.buffer.attrs:
                                mask_data = mask.buffer.values
                                if np.any(mask_data):  # Only if mask has data
                                    volume_ml = self._calculate_mask_volume(
                                        mask_data, input_data
                                    )
                                    pixel_count = int(np.sum(mask_data))

                                    # Add volume attributes
                                    mask.buffer.attrs["volume_ml"] = float(volume_ml)
                                    mask.buffer.attrs["volume_mm3"] = float(
                                        volume_ml * 1000
                                    )
                                    mask.buffer.attrs["pixel_count"] = pixel_count
                                else:
                                    # Empty mask
                                    mask.buffer.attrs["volume_ml"] = 0.0
                                    mask.buffer.attrs["volume_mm3"] = 0.0
                                    mask.buffer.attrs["pixel_count"] = 0

                        except Exception as e:
                            debug_log_mask_creator(
                                message=f"❌ Failed to add volume to mask '{mask_name}': {e}",
                            )

            except Exception as e:
                debug_log_mask_creator(
                    message=f"❌ Failed to calculate volumes during session completion: {e}",
                )

            # Finalize all masks (this will add finalized=True and timestamp)
            for mask in self.masks.values():
                if not mask.finalized:
                    mask.finalize()

            # Clear all visual patches (lasso outlines)
            for mask_container in self.masks.values():
                self.viz_sync.clear_all_visual_patches(mask_container)
                if hasattr(mask_container, "_lasso_selector"):
                    mask_container._lasso_selector.set_active(False)

            # CRITICAL: Ensure all finalized masks remain visible after session ends
            finalized_masks = {
                name: mask for name, mask in self.masks.items() if mask.finalized
            }

            if finalized_masks:
                print(f"✅ Finalized masks: {list(finalized_masks.keys())}")
                for name, mask in finalized_masks.items():
                    pixels = mask.get_total_pixels()
                    # Show volume if available in attributes
                    volume_info = ""
                    if "volume_ml" in mask.buffer.attrs:
                        volume_ml = mask.buffer.attrs["volume_ml"]
                        volume_info = f", {volume_ml:.3f} mL"
                    print(f"   {name}: {pixels} pixels{volume_info}")

                # Sync all finalized masks to plotter for persistent display
                self.viz_sync.sync_all_finalized_masks(
                    self.masks, self.data_source.plotter
                )
                print("✅ Finalized masks will remain visible when switching slices")

            else:
                # Clear plotter overlays if no finalized masks
                if hasattr(self.data_source, "plotter"):
                    self.viz_sync.clear_plotter_overlays(self.data_source.plotter)
                print("⚠️  No masks were finalized")

            # Clear active mask since session is done
            self.active_mask_name = None

            # --- NEW: synchronize finalized masks to the data_source attributes for
            # backward compatibility so users can access anatomy.masks directly ---
            try:
                # finalized_masks is name -> MaskContainer
                finalized_buffers = {
                    name: mask.buffer for name, mask in finalized_masks.items()
                }

                # store the MaskContainer objects separately
                try:
                    self.data_source.mask_containers = self.masks.copy()
                except Exception:
                    self.data_source.mask_containers = finalized_masks

                # Create an xarray.Dataset from the finalized DataArray buffers
                try:
                    if finalized_buffers:
                        masks_ds = xr.Dataset(finalized_buffers)
                        masks_ds.attrs["creation_method"] = "create_mask_simplified"
                        masks_ds.attrs["total_masks"] = len(finalized_buffers)
                        masks_ds.attrs["mask_names"] = list(finalized_buffers.keys())
                    else:
                        masks_ds = xr.Dataset()

                    # store Dataset on data_source for downstream usage
                    try:
                        self.data_source.masks = masks_ds
                    except Exception:
                        setattr(self.data_source, "masks", masks_ds)

                    print(
                        "✅ Finalized masks synchronized to data_source.masks (Dataset) and data_source.mask_containers"
                    )
                except Exception as e:
                    print(f"Warning: failed to create masks Dataset: {e}")
                    # fallback to raw buffers dict
                    try:
                        self.data_source.masks = finalized_buffers
                    except Exception:
                        setattr(self.data_source, "masks", finalized_buffers)

            except Exception as e:
                print(f"Warning: failed to sync finalized masks to data_source: {e}")

    def _update_ui_state(self):
        """Update UI button states based on current state."""
        has_active_mask = (
            self.active_mask_name is not None and self.active_mask_name in self.masks
        )

        if has_active_mask:
            mask = self.masks[self.active_mask_name]
            can_finalize = not mask.finalized

            self.ui_widgets["finalize_btn"].disabled = not can_finalize
            self.ui_widgets["clear_lasso_btn"].disabled = False
            self.ui_widgets["clear_mask_btn"].disabled = False
        else:
            self.ui_widgets["finalize_btn"].disabled = True
            self.ui_widgets["clear_lasso_btn"].disabled = True
            self.ui_widgets["clear_mask_btn"].disabled = True

    def _trigger_plot_update(self):
        """Trigger plot update to show ONLY the aktive Maske als Overlay (nicht alle)."""
        try:
            # Onlz show active mask
            if self.active_mask_name and self.active_mask_name in self.masks:
                mask_container = self.masks[self.active_mask_name]
                # Only show active mask, remove all other overlays
                self.viz_sync.sync_mask_to_plotter(
                    mask_container,
                    self.data_source.plotter,
                    show_only_active=True,  # Only show active mask
                )
            else:
                # Wenn there is no active mask, show all finalized masks
                finalized_count = sum(
                    1 for mask in self.masks.values() if mask.finalized
                )
                if finalized_count > 0:
                    self.viz_sync.sync_all_finalized_masks(
                        self.masks, self.data_source.plotter
                    )

            # Update plot state for lasso handler
            self._update_lasso_plot_state()

            # Get current widget values to preserve slice position
            current_kwargs = self.plot_handles.get("initial_kwargs", {}).copy()
            try:
                if hasattr(self.data_source, "plotter") and hasattr(
                    self.data_source.plotter, "widgets"
                ):
                    if self.data_source.plotter.widgets:
                        for name, widget in self.data_source.plotter.widgets.items():
                            if hasattr(widget, "value"):
                                current_kwargs[name] = widget.value
                        current_kwargs.get("slice_idx", "NOT_FOUND")

            except Exception:
                debug_log_mask_creator(
                    message="❌ Failed to get current widget values, using initial kwargs",
                )

            # Enable debugging
            if hasattr(self.data_source, "plotter"):
                self.data_source.plotter._debug_mask_alignment = True

            # Trigger plot update
            run_update = self.plot_handles.get("run_update")
            if run_update:
                run_update(**current_kwargs)

            # Log final overlay count for debugging
            overlay_count = len(
                getattr(self.data_source.plotter, "interactive_overlays", {})
            )

        except Exception as e:
            debug_log_mask_creator(
                message=f"❌ Exception in _trigger_plot_update: {e}",
            )

    def _display_session(self):
        """Display the complete masking session."""

        # Create layout
        row1 = widgets.HBox(
            [self.ui_widgets["mask_selector"], self.ui_widgets["new_mask_name"]]
        )

        row2 = widgets.HBox(
            [
                self.ui_widgets["color_picker"],
            ]
        )

        row2_1 = widgets.HBox(
            [
                self.ui_widgets["alpha_slider"],
                self.ui_widgets["activate_btn"],
            ]
        )

        row3 = widgets.HBox(
            [
                self.ui_widgets["finalize_btn"],
                self.ui_widgets["clear_lasso_btn"],
                self.ui_widgets["clear_mask_btn"],
            ]
        )
        row3_1 = widgets.HBox(
            [
                self.ui_widgets["volume_display"],
            ]
        )  # Add volume display next to buttons])

        row4 = widgets.HBox([self.ui_widgets["done_btn"]])

        mask_ui = widgets.VBox(
            [row1, row2, row2_1, row3, row3_1, row4, self.ui_widgets["status_output"]]
        )

        layout_mode = getattr(self, "_layout_mode", "horizontal")

        # Configure layouts so controls and mask pane align appropriately
        plot_controls = self.plot_handles.get("controls")
        if hasattr(plot_controls, "layout"):
            if layout_mode == "horizontal":
                plot_controls.layout = widgets.Layout(
                    flex="1 1 45%",
                    min_width="320px",
                    margin="5px",
                )
            else:
                plot_controls.layout = widgets.Layout(
                    width="100%",
                    margin="5px",
                )

        mask_layout_kwargs = {
            "border": "solid 1px grey",
            "padding": "5px",
            "margin": "5px",
        }
        if layout_mode == "horizontal":
            mask_layout_kwargs.update({"flex": "1 1 55%", "min_width": "360px"})
        else:
            mask_layout_kwargs.update({"width": "100%"})
        mask_ui.layout = widgets.Layout(**mask_layout_kwargs)

        output_area = self.plot_handles.get("output_area")
        if hasattr(output_area, "layout"):
            output_area.layout = widgets.Layout(width="100%")

        controls_children = [
            child for child in (plot_controls, mask_ui) if child is not None
        ]

        if layout_mode == "horizontal":
            controls_container = widgets.HBox(
                controls_children,
                layout=widgets.Layout(align_items="flex-start", width="100%"),
            )
        else:
            controls_container = widgets.VBox(
                controls_children,
                layout=widgets.Layout(width="100%"),
            )

        full_children = [controls_container]
        if output_area is not None:
            full_children.append(output_area)

        full_layout = widgets.VBox(
            full_children,
            layout=widgets.Layout(width="100%"),
        )

        display(full_layout)

        # Trigger initial plot and volume display update
        self._trigger_plot_update()
        self._update_volume_display()

    def get_finalized_masks(self) -> Dict[str, xr.DataArray]:
        """Get all finalized masks."""
        print(f"self.masks={self.masks}")
        return {
            name: mask.buffer for name, mask in self.masks.items() if mask.finalized
        }

    def _calculate_mask_volume(self, mask_data, input_data):
        """
        Calculate physical volume of active mask pixels using correct dimensional data.

        CRITICAL FIX: This method now uses the same preprocessed data that was used
        for mask creation, ensuring accurate volume calculations.

        Parameters
        ----------
        mask_data : np.ndarray
            Boolean array of active mask pixels
        input_data : xr.DataArray
            DataArray with coordinate information (should match mask coordinate system)

        Returns
        -------
        float
            Volume in milliliters
        """

        if mask_data is None:
            return 0.0

        if not np.any(mask_data):
            return 0.0

        try:
            # Step 1: Count active mask pixels
            active_pixels = np.sum(mask_data)

            debug_log_mask_creator(
                message=f"🔍 Volume calculation - active pixels: {active_pixels}"
            )
            debug_log_mask_creator(
                message=f"📊 Input data for volume calc - shape: {input_data.shape}, dims: {input_data.dims}"
            )

            # CRITICAL FIX: Use input_data (preprocessed) for all spatial calculations
            # This ensures volume calculation matches the coordinate system of the mask

            # Step 2: Extract slice thickness
            slice_thickness = self._get_slice_thickness(input_data)

            debug_log_mask_creator(message=f"📏 Slice thickness: {slice_thickness} mm")

            # Step 3: Extract pixel spacing from the preprocessed data coordinates
            read_spacing, phase_spacing = self._get_pixel_spacing(input_data)

            debug_log_mask_creator(
                message=f"📏 Pixel spacing - read: {read_spacing} mm, phase: {phase_spacing} mm"
            )

            # Step 4: Calculate voxel volume using preprocessed data dimensions
            voxel_volume_mm3 = read_spacing * phase_spacing * slice_thickness

            # Step 5: Calculate total volume
            total_volume_mm3 = active_pixels * voxel_volume_mm3
            total_volume_ml = total_volume_mm3 / 1000.0  # Convert mm³ to mL

            debug_log_mask_creator(message=f"📊 Volume calculation complete:")
            debug_log_mask_creator(message=f"   Active pixels: {active_pixels}")
            debug_log_mask_creator(
                message=f"   Voxel volume: {voxel_volume_mm3:.4f} mm³"
            )
            debug_log_mask_creator(message=f"   Total volume: {total_volume_ml:.4f} mL")

            return total_volume_ml

        except Exception as e:
            debug_log_mask_creator(
                message=f"❌ Volume calculation failed with exception: {e}"
            )
            return 0.0

    def _get_slice_thickness(self, input_data: xr.DataArray) -> float:
        """Extract slice thickness from data source or input data."""
        slice_thickness = None

        # Try data_source.slice_thickness_mm first (the correct location)
        if hasattr(self.data_source, "slice_thickness_mm"):
            slice_thickness = getattr(self.data_source, "slice_thickness_mm")
            debug_log_mask_creator(
                message=f"📏 Slice thickness from data_source: {slice_thickness}"
            )

        # Fallback to input_data attrs
        if slice_thickness is None:
            slice_thickness = input_data.attrs.get("slice_thickness_mm", None)
            debug_log_mask_creator(
                message=f"📏 Slice thickness from input_data attrs: {slice_thickness}"
            )

        # Final fallback
        if slice_thickness is None:
            slice_thickness = 1.0
            debug_log_mask_creator(message="⚠️  Using fallback slice thickness: 1.0 mm")

        return slice_thickness

    def _get_pixel_spacing(self, input_data: xr.DataArray) -> tuple[float, float]:
        """Extract pixel spacing from preprocessed data coordinates."""
        read_spacing = None
        phase_spacing = None

        # ENHANCED: Try to get spacing from preprocessed data coordinates
        if "read_idx" in input_data.coords and "phase_idx" in input_data.coords:
            read_coords = input_data.coords["read_idx"]
            phase_coords = input_data.coords["phase_idx"]

            try:
                # Handle different coordinate array structures
                if read_coords.ndim >= 2:
                    # For coordinates with shape (read, phase, slice)
                    read_diff = float(read_coords[1, 0, 0] - read_coords[0, 0, 0])
                    phase_diff = float(phase_coords[0, 1, 0] - phase_coords[0, 0, 0])
                elif read_coords.ndim == 1:
                    # For 1D coordinate arrays
                    if len(read_coords) > 1 and len(phase_coords) > 1:
                        read_diff = float(read_coords[1] - read_coords[0])
                        phase_diff = float(phase_coords[1] - phase_coords[0])
                    else:
                        read_diff = phase_diff = None
                else:
                    read_diff = phase_diff = None

                if read_diff is not None and phase_diff is not None:
                    read_spacing = abs(read_diff)
                    phase_spacing = abs(phase_diff)
                    debug_log_mask_creator(
                        message=f"📏 Pixel spacing from coordinates - read: {read_spacing}, phase: {phase_spacing}"
                    )

            except Exception as e:
                debug_log_mask_creator(
                    message=f"⚠️  Coordinate-based spacing calculation failed: {e}"
                )

        # Fallback to FOV/matrix from data_source if coordinates failed
        if read_spacing is None or phase_spacing is None:
            debug_log_mask_creator(message="📏 Falling back to FOV/matrix calculation")

            fov_mm = getattr(self.data_source, "fov_mm", None)
            matrix = getattr(self.data_source, "matrix", None)

            if fov_mm is not None and matrix is not None:
                try:
                    if len(fov_mm) >= 2 and len(matrix) >= 2:
                        read_spacing = fov_mm[0] / matrix[0]
                        phase_spacing = fov_mm[1] / matrix[1]
                        debug_log_mask_creator(
                            message=f"📏 Pixel spacing from FOV/matrix - read: {read_spacing}, phase: {phase_spacing}"
                        )
                except Exception as e:
                    debug_log_mask_creator(
                        message=f"⚠️  FOV/matrix calculation failed: {e}"
                    )

            # Last resort fallback
            if read_spacing is None or phase_spacing is None:
                read_spacing = phase_spacing = 1.0
                debug_log_mask_creator(
                    message="⚠️  Using fallback pixel spacing: 1.0 mm"
                )

        return read_spacing, phase_spacing

    def _update_volume_display(self):
        """Update the volume display widget with current active mask volume."""

        try:
            # Check if volume display widget exists
            if "volume_display" not in self.ui_widgets:
                return

            if not self.active_mask_name:
                self.ui_widgets["volume_display"].value = "<b>Volume: 0.0 mL</b>"
                return

            if self.active_mask_name not in self.masks:
                self.ui_widgets["volume_display"].value = "<b>Volume: 0.0 mL</b>"
                return

            # Get active mask data
            active_mask = self.masks[self.active_mask_name]

            # Get mask data - try multiple sources
            mask_data = None

            # Try buffer first (DataArray)
            if hasattr(active_mask, "buffer") and active_mask.buffer is not None:
                try:
                    mask_data = active_mask.buffer.values
                except Exception:
                    debug_log_mask_creator(
                        message=f"❌ buffer attrs: {active_mask.buffer.attrs if hasattr(active_mask.buffer, 'attrs') else 'N/A'}",
                    )

            # Try data attribute
            if mask_data is None and hasattr(active_mask, "data"):
                try:
                    mask_data = getattr(active_mask, "data")
                    if mask_data is not None:

                        debug_log_mask_creator(
                            message=f"✅ mask_data shape: {mask_data.shape}, dtype: {mask_data.dtype}, True pixels: {np.sum(mask_data)}",
                        )
                except Exception as e:
                    debug_log_mask_creator(
                        message=f"❌ Error accessing data attribute: {e}",
                    )

            # Try calling get_current_mask_data method if available
            if mask_data is None and hasattr(active_mask, "get_current_mask_data"):
                try:
                    mask_data = active_mask.get_current_mask_data()
                    if mask_data is not None:

                        debug_log_mask_creator(
                            message=f"✅ mask_data shape: {mask_data.shape}, dtype: {mask_data.dtype}, True pixels: {np.sum(mask_data)}",
                        )
                except Exception as e:
                    debug_log_mask_creator(
                        message=f"❌ Error calling get_current_mask_data(): {e}",
                    )

            if mask_data is None:
                self.ui_widgets["volume_display"].value = "<b>Volume: ? mL</b>"
                return

            # Get input data for coordinate information - FIXED: pass None parameter
            input_data = self._get_input_data(None)
            if input_data is None:
                self.ui_widgets["volume_display"].value = "<b>Volume: ? mL</b>"
                return

            # Calculate volume - this will do detailed logging
            volume_ml = self._calculate_mask_volume(mask_data, input_data)

            # Update display with appropriate precision
            if volume_ml >= 10:
                volume_str = f"{volume_ml:.1f}"
            elif volume_ml >= 1:
                volume_str = f"{volume_ml:.2f}"
            elif volume_ml >= 0.1:
                volume_str = f"{volume_ml:.3f}"
            else:
                volume_str = f"{volume_ml:.4f}"

            display_text = f"<b>Volume: {volume_str} mL</b>"
            self.ui_widgets["volume_display"].value = display_text

        except Exception:
            self.ui_widgets["volume_display"].value = "<b>Volume: ? mL</b>"
