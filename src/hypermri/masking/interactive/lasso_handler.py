"""Direct lasso selection to mask mapping without coordinate transforms."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from ...utils.system.logging import debug_log
from .mask_container import MaskContainer

DEBUG_LOG = True
DEBUG_FILE = "/tmp/lasso_handler_debug.log"


def debug_log_mask_application(message: str):
    """Helper function for consistent debug logging in mask application."""
    if DEBUG_LOG:
        debug_log(
            message=message,
            file_name=DEBUG_FILE,
            write_to_file=DEBUG_LOG,
        )


class LassoHandler:
    """
    Handle lasso selection with direct mapping to data coordinates.

    Key principle: No coordinate transforms - direct mapping from lasso vertices
    to data array indices using the same coordinate system.
    """

    def __init__(self, data_array: xr.DataArray):
        """
        Initialize lasso handler.

        Parameters
        ----------
        data_array : xr.DataArray
            Source data array that masks will be created for
        """
        self.data_array = data_array

        # Store reference to active mask container
        self.active_mask: Optional[MaskContainer] = None

        # Track current plot state
        self.current_plot_dims: Optional[Tuple[str, str]] = None
        self.current_slice_selector: Optional[Dict[str, Any]] = None

        # Store plot extent information for coordinate mapping
        self.current_plot_extent: Optional[Tuple[float, float, float, float]] = None

        # Store last lasso data for undo functionality
        self.last_lasso_data: Optional[np.ndarray] = None
        self.last_lasso_selector: Optional[Dict[str, Any]] = None

        # Callback for volume updates
        self.volume_update_callback: Optional[callable] = None

    def set_active_mask(self, mask_container: MaskContainer):
        """Set the active mask container for lasso operations."""
        self.active_mask = mask_container

    def set_volume_update_callback(self, callback: callable):
        """
        Set callback function to be called when mask volume changes.

        Parameters
        ----------
        callback : callable
            Function to call when volume should be updated
        """
        self.volume_update_callback = callback

    def set_plot_state(
        self,
        plot_dims: Tuple[str, str],
        slice_selector: Dict[str, Any],
        plot_extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Set current plot state for lasso mapping.

        Parameters
        ----------
        plot_dims : Tuple[str, str]
            Current plot dimensions (y_dim, x_dim) e.g., ('phase_idx', 'read_idx')
        slice_selector : Dict[str, Any]
            Current slice selector e.g., {'slice_idx': 5, 'acquisition_idx': 0}
        plot_extent : Tuple[float, float, float, float], optional
            Plot extent (xmin, xmax, ymin, ymax) for coordinate mapping
        """
        self.current_plot_dims = plot_dims
        self.current_slice_selector = slice_selector
        self.current_plot_extent = plot_extent

    def create_lasso_selector(
        self, ax, enhanced_visibility: bool = True
    ) -> LassoSelector:
        """
        Create lasso selector that maps directly to data indices.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes to attach selector to
        enhanced_visibility : bool, default True
            Use enhanced line properties for visibility

        Returns
        -------
        LassoSelector
            Configured lasso selector
        """

        def onselect(verts):
            """Handle lasso selection - direct mapping to mask."""
            result = self._handle_lasso_selection(verts)

            # Add visual patches if successful
            if result and result.get("success", False):
                try:
                    import numpy as np

                    from .visualization_sync import VisualizationSync

                    # Create visualization sync if not available
                    if not hasattr(self, "_viz_sync"):
                        self._viz_sync = VisualizationSync()

                    # Add visual patches for the lasso
                    verts_array = np.array(verts)
                    patches = self._viz_sync.add_lasso_visual_patches(
                        ax, verts_array, self.active_mask
                    )

                    # Force canvas update
                    self._viz_sync.force_canvas_refresh(ax)

                except Exception as e:
                    debug_log_mask_application(
                        message=f"ERROR in onselect visualization: {e}",
                    )

        # Create selector with enhanced visibility
        props = (
            dict(color="red", linestyle="-", linewidth=4, alpha=1.0)
            if enhanced_visibility
            else dict(color="red")
        )

        selector = LassoSelector(
            ax=ax,
            onselect=onselect,
            useblit=False,  # Better visibility
            button=[1],  # Left mouse button
            props=props,
        )

        return selector

    def _handle_lasso_selection(self, verts):
        """
        Handle lasso selection with enhanced slice tracking.

        Parameters
        ----------
        verts : list
            Lasso vertices from matplotlib
        """
        if not self.active_mask:
            debug_log_mask_application(
                message="ERROR: No active mask set for lasso selection",
            )
            return

        if not self.current_plot_dims or not self.current_slice_selector:
            debug_log_mask_application(
                message="ERROR: Plot state not fully set for lasso selection",
            )
            return

        try:
            # CRITICAL: Ensure slice_idx is captured correctly
            if "slice_idx" in self.current_slice_selector:
                self.current_slice_selector["slice_idx"]
            else:
                debug_log_mask_application(
                    message="ERROR: No slice_idx in current slice selector",
                )

                # Get slice from data array middle as emergency fallback
                if "slice_idx" in self.data_array.dims:
                    middle_slice = self.data_array.sizes["slice_idx"] // 2
                    self.current_slice_selector["slice_idx"] = middle_slice

            # Create 2D mask directly in data array coordinate space
            lasso_mask_2d = self._create_mask_from_lasso_vertices(verts)

            if lasso_mask_2d is None:

                debug_log_mask_application(
                    message="ERROR: Failed to create mask from lasso vertices",
                )
                return

            # Apply directly to mask container - no coordinate transforms!
            success, applied_mask = self.active_mask.apply_lasso_selection(
                lasso_mask_2d, self.current_slice_selector
            )

            if success:
                # Store aligned mask for undo functionality
                self.last_lasso_selector = self.current_slice_selector.copy()
                if applied_mask is not None:
                    self.last_lasso_data = applied_mask.copy()
                else:
                    self.last_lasso_data = None

                # VALIDATION: Verify mask was stored correctly
                if "slice_idx" in self.current_slice_selector:
                    stored_slice = self.current_slice_selector["slice_idx"]
                    try:
                        validation_slice = self.active_mask.buffer.isel(
                            slice_idx=stored_slice
                        )
                        np.sum(validation_slice.values)
                        np.sum(self.active_mask.buffer.values)
                    except Exception as e:
                        debug_log_mask_application(
                            message=f"ERROR in validation: {e}",
                        )

                # NEW: Trigger volume update callback
                if self.volume_update_callback:
                    try:
                        self.volume_update_callback()
                    except Exception as e:
                        debug_log_mask_application(
                            message=f"ERROR in volume update callback: {e}",
                        )

                # Return both the mask data and the vertices for visualization
                return {
                    "lasso_mask_2d": lasso_mask_2d,
                    "vertices": verts,
                    "success": True,
                }
            else:
                return {"success": False}

        except Exception as e:
            debug_log_mask_application(
                message=f"ERROR in _handle_lasso_selection: {e}",
            )

    def _convert_vertices_to_coordinates(self, verts) -> Optional[np.ndarray]:
        """
        Convert matplotlib lasso vertices to data coordinate space.

        This creates a direct coordinate mapping instead of boolean mask approach.

        Parameters
        ----------
        verts : list
            Lasso vertices in plot coordinate space

        Returns
        -------
        np.ndarray or None
            Array of coordinate points that define the polygon
        """
        try:
            if not self.current_plot_dims:
                return None

            plot_dim_y, plot_dim_x = self.current_plot_dims

            # Get data array shape for bounds checking
            y_size = self.data_array.sizes[plot_dim_y]
            x_size = self.data_array.sizes[plot_dim_x]

            # Convert vertices to numpy array
            verts_array = np.array(verts)

            # COORDINATE MAPPING: Convert from plot space to data indices
            if self.current_plot_extent:
                # Use plot extent for accurate mapping
                xmin, xmax, ymin, ymax = self.current_plot_extent

                # Map vertices from plot extent to array indices
                x_indices = ((verts_array[:, 0] - xmin) / (xmax - xmin)) * (x_size - 1)
                y_indices = ((verts_array[:, 1] - ymin) / (ymax - ymin)) * (y_size - 1)

            else:
                # Fallback: assume vertices are already in index space or scale appropriately
                x_range = verts_array[:, 0].max() - verts_array[:, 0].min()
                y_range = verts_array[:, 1].max() - verts_array[:, 1].min()

                # Check if vertices are in reasonable index range
                if (
                    0 <= verts_array[:, 0].min()
                    and verts_array[:, 0].max() < x_size
                    and 0 <= verts_array[:, 1].min()
                    and verts_array[:, 1].max() < y_size
                    and x_range > 1
                    and y_range > 1
                ):
                    # Vertices are already in index space
                    x_indices = verts_array[:, 0]
                    y_indices = verts_array[:, 1]
                else:
                    # Scale from vertex range to array range
                    x_min, x_max = verts_array[:, 0].min(), verts_array[:, 0].max()
                    y_min, y_max = verts_array[:, 1].min(), verts_array[:, 1].max()

                    if x_range > 0 and y_range > 0:
                        x_indices = ((verts_array[:, 0] - x_min) / x_range) * (
                            x_size - 1
                        )
                        y_indices = ((verts_array[:, 1] - y_min) / y_range) * (
                            y_size - 1
                        )
                    else:
                        # Degenerate case
                        x_indices = np.full(len(verts_array), x_size // 2)
                        y_indices = np.full(len(verts_array), y_size // 2)

            # Clamp to valid ranges
            x_indices = np.clip(x_indices, 0, x_size - 1)
            y_indices = np.clip(y_indices, 0, y_size - 1)

            # Create coordinate array (x, y pairs)
            coordinate_points = np.column_stack([x_indices, y_indices])

            return coordinate_points

        except Exception as e:
            debug_log_mask_application(
                message=f"ERROR in _convert_vertices_to_coordinates: {e}",
            )
            return None

    def _create_mask_from_lasso_vertices(self, verts) -> Optional[np.ndarray]:
        """
        Create 2D mask directly from lasso vertices with improved coordinate mapping.

        Parameters
        ----------
        verts : list
            Lasso vertices in plot coordinate space

        Returns
        -------
        np.ndarray or None
            2D boolean mask aligned with plot dimensions (not buffer dimensions)
        """
        try:
            if not self.current_plot_dims:
                return None

            plot_dim_y, plot_dim_x = self.current_plot_dims

            # Get the 2D slice shape from data array IN PLOT DIMENSION ORDER
            # This is critical - we create the mask in plot coordinate order
            slice_shape_plot_order = (
                self.data_array.sizes[plot_dim_y],  # Y dimension first (rows)
                self.data_array.sizes[plot_dim_x],  # X dimension second (cols)
            )

            # Convert vertices to numpy array
            verts_array = np.array(verts)

            # IMPROVED COORDINATE MAPPING:
            # Map lasso vertices (in plot space) to array indices (in plot dimension order)

            if self.current_plot_extent:
                # Use plot extent for accurate physical coordinate mapping
                xmin, xmax, ymin, ymax = self.current_plot_extent

                # Map vertices from plot extent to array indices
                # Note: vertices are (x,y) but we need to map to (cols, rows)
                x_indices = ((verts_array[:, 0] - xmin) / (xmax - xmin)) * (
                    slice_shape_plot_order[1] - 1
                )
                y_indices = ((verts_array[:, 1] - ymin) / (ymax - ymin)) * (
                    slice_shape_plot_order[0] - 1
                )

                # Clamp to valid index range
                x_indices = np.clip(x_indices, 0, slice_shape_plot_order[1] - 1)
                y_indices = np.clip(y_indices, 0, slice_shape_plot_order[0] - 1)

            else:
                # Fallback coordinate mapping strategies
                x_range = verts_array[:, 0].max() - verts_array[:, 0].min()
                y_range = verts_array[:, 1].max() - verts_array[:, 1].min()

                # Strategy 1: Check if vertices are already in index space
                verts_in_index_space = (
                    0 <= verts_array[:, 0].min()
                    and verts_array[:, 0].max() < slice_shape_plot_order[1]
                    and 0 <= verts_array[:, 1].min()
                    and verts_array[:, 1].max() < slice_shape_plot_order[0]
                    and x_range > 1
                    and y_range > 1  # Reasonable selection size
                )

                if verts_in_index_space:
                    # Vertices are already in array index space - use directly
                    x_indices = verts_array[:, 0]
                    y_indices = verts_array[:, 1]

                else:
                    # Strategy 2: Linear scaling from vertex range to array range

                    # Check if vertex range makes sense for scaling
                    if x_range > 0 and y_range > 0:
                        # Scale vertex range to full array range
                        x_min, x_max = verts_array[:, 0].min(), verts_array[:, 0].max()
                        y_min, y_max = verts_array[:, 1].min(), verts_array[:, 1].max()

                        x_indices = ((verts_array[:, 0] - x_min) / x_range) * (
                            slice_shape_plot_order[1] - 1
                        )
                        y_indices = ((verts_array[:, 1] - y_min) / y_range) * (
                            slice_shape_plot_order[0] - 1
                        )

                    else:
                        # Strategy 3: Center point mapping for degenerate cases
                        x_indices = np.full(
                            len(verts_array), slice_shape_plot_order[1] // 2
                        )
                        y_indices = np.full(
                            len(verts_array), slice_shape_plot_order[0] // 2
                        )

            # Create coordinate grid for path testing IN PLOT DIMENSION ORDER
            y_coords = np.arange(slice_shape_plot_order[0])  # Row indices (plot_dim_y)
            x_coords = np.arange(slice_shape_plot_order[1])  # Col indices (plot_dim_x)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

            # Create path and test containment
            mapped_verts = np.column_stack([x_indices, y_indices])
            path = Path(mapped_verts)
            points = np.column_stack([xx.ravel(), yy.ravel()])
            mask_1d = path.contains_points(points)

            # Reshape to 2D mask in PLOT DIMENSION ORDER
            mask_2d_plot_order = mask_1d.reshape(slice_shape_plot_order)

            # Count selected pixels and validate
            selected_pixels = np.sum(mask_2d_plot_order)
            total_pixels = mask_2d_plot_order.size
            (selected_pixels / total_pixels) * 100

            # Return mask in PLOT DIMENSION ORDER
            # The MaskContainer will handle any necessary transposition for buffer alignment
            return mask_2d_plot_order

        except Exception as e:
            debug_log_mask_application(
                message=f"ERROR in _create_mask_from_lasso_vertices: {e}",
            )
            return None

    def undo_last_lasso(self) -> bool:
        """
        Undo the last lasso selection.

        Returns
        -------
        bool
            True if undo successful
        """
        if (
            not self.active_mask
            or not self.last_lasso_data
            or not self.last_lasso_selector
        ):
            return False

        try:
            # Use the original boolean buffer approach
            success = self.active_mask.remove_last_lasso_region(
                self.last_lasso_data, self.last_lasso_selector
            )

            if success:
                # Clear undo data after successful undo
                self.last_lasso_data = None
                self.last_lasso_selector = None

            return success

        except Exception as e:
            debug_log_mask_application(
                message=f"ERROR in undo_last_lasso: {e}",
            )
            return False

    def get_current_mask_slice(self) -> Optional[np.ndarray]:
        """
        Get current mask slice for the active plot state.

        Returns
        -------
        np.ndarray or None
            2D mask for current slice
        """
        if not self.active_mask or not self.current_slice_selector:
            return None

        return self.active_mask.get_mask_for_slice(self.current_slice_selector)

    def validate_plot_coordinate_alignment(self) -> bool:
        """
        Validate that plot coordinates align with data coordinates.

        Returns
        -------
        bool
            True if alignment is correct
        """
        if not self.current_plot_dims:
            return False

        plot_dim_y, plot_dim_x = self.current_plot_dims

        # Check that plot dimensions exist in data
        if (
            plot_dim_y not in self.data_array.dims
            or plot_dim_x not in self.data_array.dims
        ):
            debug_log_mask_application(
                message=f"ERROR: Plot dims {self.current_plot_dims} not in data dims {self.data_array.dims}",
            )
            return False

        # Check that we have reasonable slice selector
        if not self.current_slice_selector:
            debug_log_mask_application(
                message="ERROR: No slice selector set",
            )
            return False

        # Validate slice selector dimensions exist
        for dim in self.current_slice_selector:
            if dim not in self.data_array.dims:
                debug_log_mask_application(
                    message=f"ERROR: Slice dim {dim} not in data dims {self.data_array.dims}",
                )
                return False

        return True
