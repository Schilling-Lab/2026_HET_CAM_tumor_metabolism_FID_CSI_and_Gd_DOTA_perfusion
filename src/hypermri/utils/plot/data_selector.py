"""Data selection module for unified plotting system.

This module handles all voxel selection and array slicing logic,
consolidating dimension selection code from multiple plotters.
"""

from typing import Dict, List, Optional, Tuple

import xarray as xr


class DataSelector:
    """Handles voxel selection and array slicing operations.

    This class provides methods for:
    - Identifying spatial dimensions
    - Slicing data arrays based on widget values
    - Selecting voxel signals from spatial data
    - Detecting active viewing perspectives
    """

    def __init__(self):
        """Initialize the DataSelector."""
        self._active_perspective_dim = None

    def get_all_spatial_dims(self, data_array: xr.DataArray) -> List[str]:
        """Get all potential spatial dimensions from data array.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array

        Returns
        -------
        List[str]
            List of spatial dimension names found in the data
        """
        potential_spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
        return [dim for dim in potential_spatial_dims if dim in data_array.dims]

    def get_spatial_dims(
        self,
        data_array: xr.DataArray,
        kwargs: Optional[Dict] = None,
        widget_prefix: str = "",
    ) -> List[str]:
        """Get the spatial dimensions to display (2D for plotting).

        This method determines which 2 dimensions should be displayed
        based on data size and active perspective.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array
        kwargs : Optional[Dict], optional
            Widget parameters for perspective detection
        widget_prefix : str, optional
            Widget prefix

        Returns
        -------
        List[str]
            List of 2 spatial dimension names to display
        """
        all_spatial_dims = self.get_all_spatial_dims(data_array)

        if len(all_spatial_dims) < 2:
            return all_spatial_dims

        # If we have 3D data, detect active perspective
        if len(all_spatial_dims) >= 3:
            active_perspective = self.detect_active_perspective(
                data_array, kwargs or {}, all_spatial_dims
            )

            if active_perspective:
                ordered_dims = self._ordered_dims_for_perspective(
                    active_perspective, all_spatial_dims
                )
                if ordered_dims:
                    return ordered_dims
                # Fallback: return the two dimensions that are NOT being sliced
                return [d for d in all_spatial_dims if d != active_perspective]

        # Fallback: sort by size and return the two largest dimensions
        dims_and_sizes = [(dim, data_array.sizes[dim]) for dim in all_spatial_dims]
        dims_and_sizes.sort(key=lambda x: x[1], reverse=True)

        return [dims_and_sizes[0][0], dims_and_sizes[1][0]]

    def detect_active_perspective(
        self, data_array: xr.DataArray, kwargs: Dict, spatial_dims: List[str]
    ) -> Optional[str]:
        """Detect which spatial dimension is being actively sliced.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array
        kwargs : Dict
            Widget parameters
        spatial_dims : List[str]
            List of spatial dimensions

        Returns
        -------
        Optional[str]
            Name of the dimension being sliced, or None
        """
        # Check if we have a stored active perspective
        if (
            self._active_perspective_dim
            and self._active_perspective_dim in spatial_dims
        ):
            return self._active_perspective_dim

        # Default to slice_idx for axial view
        if "slice_idx" in spatial_dims:
            self._active_perspective_dim = "slice_idx"
            return "slice_idx"

        # Fallback to first spatial dimension
        if spatial_dims:
            self._active_perspective_dim = spatial_dims[0]
            return spatial_dims[0]

        return None

    def set_active_perspective(self, perspective_dim: Optional[str]) -> None:
        """Set the active perspective dimension.

        Parameters
        ----------
        perspective_dim : Optional[str]
            Dimension name for the active perspective. Use ``None`` to clear.
        """
        self._active_perspective_dim = perspective_dim

    def _ordered_dims_for_perspective(
        self, perspective_dim: str, available_dims: List[str]
    ) -> Optional[List[str]]:
        """Return ordered spatial dimensions matching the requested perspective.

        Parameters
        ----------
        perspective_dim : str
            Dimension currently being sliced.
        available_dims : List[str]
            Spatial dimensions present in the data.

        Returns
        -------
        Optional[List[str]]
            Ordered list of spatial dimensions [y_dim, x_dim] or ``None`` if unavailable.
        """
        perspective_map = {
            "slice_idx": ("phase_idx", "read_idx"),
            "read_idx": ("slice_idx", "phase_idx"),
            "phase_idx": ("slice_idx", "read_idx"),
        }

        target_dims = perspective_map.get(perspective_dim)
        if not target_dims:
            return None

        ordered_dims = [dim for dim in target_dims if dim in available_dims]
        if len(ordered_dims) == 2:
            return ordered_dims

        return None

    def get_perspective_name(self, dim_name: str) -> str:
        """Get human-readable perspective name from dimension name.

        Parameters
        ----------
        dim_name : str
            Dimension name (e.g., 'slice_idx')

        Returns
        -------
        str
            Human-readable name (e.g., 'Slice')
        """
        perspective_names = {
            "slice_idx": "Slice",
            "read_idx": "Read",
            "phase_idx": "Phase",
        }
        return perspective_names.get(dim_name, f"Custom ({dim_name})")

    def apply_spatial_slicing(
        self, data_array: xr.DataArray, kwargs: Dict, widget_prefix: str = ""
    ) -> xr.DataArray:
        """Apply slicing to select a 2D spatial plane from data.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array
        kwargs : Dict
            Widget parameters with dimension indices
        widget_prefix : str, optional
            Widget prefix

        Returns
        -------
        xr.DataArray
            Sliced 2D data array
        """
        # Get spatial dimensions that will be displayed
        spatial_dims = self.get_spatial_dims(data_array, kwargs, widget_prefix)

        # Create selectors for non-spatial dimensions
        selectors = {}
        for dim in data_array.dims:
            # Skip dimensions that will be displayed
            if dim in spatial_dims:
                continue

            # Skip special dimensions
            if dim in ["acquisition_idx", "metabolite"]:
                continue

            # Check if we have a widget value for this dimension
            if dim in kwargs:
                selectors[dim] = kwargs[dim]

        # Apply selections
        if selectors:
            # Filter to only include dimensions that actually exist AND have valid indices
            # Prevents IndexError when overlaying arrays with different dimension sizes
            valid_selectors = {
                k: v
                for k, v in selectors.items()
                if k in data_array.dims and v < data_array.sizes[k]
            }
            if valid_selectors:
                data_array = data_array.isel(**valid_selectors)

        return data_array

    def select_voxel_signal(
        self,
        data_array: xr.DataArray,
        voxel_coords: Dict[str, int],
        kwargs: Optional[Dict] = None,
    ) -> Optional[xr.DataArray]:
        """Extract 1D signal from a specific voxel in spatial data.

        Parameters
        ----------
        data_array : xr.DataArray
            Input spatial data array
        voxel_coords : Dict[str, int]
            Dictionary mapping dimension names to indices
        kwargs : Optional[Dict], optional
            Additional widget parameters

        Returns
        -------
        Optional[xr.DataArray]
            1D signal array, or None if selection fails
        """
        if data_array is None:
            return None

        try:
            # Create selector for spatial dimensions
            spatial_selector = {}
            for dim, idx in voxel_coords.items():
                if dim in data_array.dims:
                    # Ensure index is within bounds
                    max_idx = data_array.sizes[dim] - 1
                    spatial_selector[dim] = max(0, min(int(idx), max_idx))

            # Apply spatial selection
            if spatial_selector:
                signal = data_array.isel(**spatial_selector)
            else:
                signal = data_array

            # Apply additional dimension selections from kwargs if provided
            if kwargs is not None:
                additional_selectors = {}
                for dim in signal.dims:
                    # Skip acquisition_idx (this is our signal dimension)
                    if dim == "acquisition_idx":
                        continue

                    # Check if we have a selector for this dimension
                    if dim in kwargs:
                        additional_selectors[dim] = kwargs[dim]

                if additional_selectors:
                    # Filter to existing dimensions with bounds checking
                    # Prevents IndexError when overlaying arrays with different dimension sizes
                    valid_selectors = {
                        k: v
                        for k, v in additional_selectors.items()
                        if k in signal.dims and v < signal.sizes[k]
                    }
                    if valid_selectors:
                        signal = signal.isel(**valid_selectors)

            return signal

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to select voxel signal: {e}")
            return None

    def get_click_coordinates(
        self,
        event_xdata: float,
        event_ydata: float,
        extent: Tuple[float, float, float, float],
        data_shape: Tuple[int, int],
        plot_dims: Tuple[str, str],
        origin: str = "lower",
    ) -> Dict[str, int]:
        """Convert matplotlib click coordinates to array indices.

        Parameters
        ----------
        event_xdata : float
            X coordinate from click event
        event_ydata : float
            Y coordinate from click event
        extent : Tuple[float, float, float, float]
            Matplotlib extent [xmin, xmax, ymin, ymax]
        data_shape : Tuple[int, int]
            Shape of the data array (ny, nx)
        plot_dims : Tuple[str, str]
            Names of the plotted dimensions (dim_y, dim_x)
        origin : str, optional
            Matplotlib origin ('lower' or 'upper'), by default 'lower'

        Returns
        -------
        Dict[str, int]
            Dictionary mapping dimension names to array indices
        """
        ny, nx = data_shape
        dim_y, dim_x = plot_dims

        # Convert real coordinates to array indices
        x_idx = int((event_xdata - extent[0]) / (extent[1] - extent[0]) * nx)

        if origin == "upper":
            # In 'upper' origin: y coordinate 0 is at top
            y_idx = int((event_ydata - extent[2]) / (extent[3] - extent[2]) * ny)
        else:
            # In 'lower' origin: y coordinate 0 is at bottom
            y_idx = int((event_ydata - extent[2]) / (extent[3] - extent[2]) * ny)

        # Clamp to valid range
        x_idx = max(0, min(x_idx, nx - 1))
        y_idx = max(0, min(y_idx, ny - 1))

        return {dim_x: x_idx, dim_y: y_idx}

    def apply_dimension_selectors(
        self,
        data_array: xr.DataArray,
        kwargs: Dict,
        exclude_dims: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """Apply dimension selectors from kwargs to data array.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array
        kwargs : Dict
            Widget parameters containing dimension indices
        exclude_dims : Optional[List[str]], optional
            Dimensions to exclude from selection

        Returns
        -------
        xr.DataArray
            Data array with selections applied
        """
        if exclude_dims is None:
            exclude_dims = []

        # Separate isel (index) and sel (label) selectors
        isel_selectors = {}
        sel_selectors = {}

        for key, value in kwargs.items():
            # Check if this is a dimension selector
            if key not in data_array.dims:
                continue

            # Skip excluded dimensions
            if key in exclude_dims:
                continue

            # Determine if this is label-based or index-based
            if key == "metabolite":
                sel_selectors[key] = value
            else:
                isel_selectors[key] = value

        # Apply label-based selections first
        if sel_selectors:
            try:
                data_array = data_array.sel(**sel_selectors)
            except Exception:
                pass

        # Apply index-based selections
        if isel_selectors:
            # Filter to only include dimensions that actually exist AND have valid indices
            # Prevents IndexError when overlaying arrays with different dimension sizes
            valid_isel_selectors = {
                k: v
                for k, v in isel_selectors.items()
                if k in data_array.dims and v < data_array.sizes[k]
            }
            if valid_isel_selectors:
                data_array = data_array.isel(**valid_isel_selectors)

        return data_array

    def squeeze_singleton_dimensions(
        self, data_array: xr.DataArray, keep_dims: Optional[List[str]] = None
    ) -> xr.DataArray:
        """Squeeze singleton dimensions from data array.

        Parameters
        ----------
        data_array : xr.DataArray
            Input data array
        keep_dims : Optional[List[str]], optional
            Dimensions to keep even if singleton

        Returns
        -------
        xr.DataArray
            Data array with singleton dimensions removed
        """
        if keep_dims is None:
            keep_dims = []

        singleton_dims_to_squeeze = []
        for dim in data_array.dims:
            if data_array.sizes[dim] == 1 and dim not in keep_dims:
                singleton_dims_to_squeeze.append(dim)

        if singleton_dims_to_squeeze:
            data_array = data_array.squeeze(singleton_dims_to_squeeze)

        return data_array

    def prepare_2d_spatial_data(
        self,
        data_array: xr.DataArray,
        spatial_dims: List[str],
        is_overlay: bool = False,
        base_spatial_dims: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """Prepare data for 2D spatial plotting by ensuring correct dimension order.

        This method handles the complex logic of:
        1. Determining which spatial dimensions to use
        2. Reordering dimensions so spatial dims are last
        3. Squeezing out non-spatial singleton dimensions
        4. Ensuring correct orientation for imshow

        Parameters
        ----------
        data_array : xr.DataArray
            Input data (can be >2D)
        spatial_dims : List[str]
            List of spatial dimension names
        is_overlay : bool, optional
            Whether this is overlay data (uses base_spatial_dims if provided)
        base_spatial_dims : Optional[List[str]], optional
            Spatial dimensions from base image for overlay alignment

        Returns
        -------
        xr.DataArray
            2D data array ready for plotting with correct dimension order
        """
        # Use base spatial dims for overlays if available
        if is_overlay and base_spatial_dims:
            target_spatial_dims = base_spatial_dims
        else:
            target_spatial_dims = spatial_dims

        # If already 2D with correct dims, return as-is
        if data_array.ndim == 2:
            if list(data_array.dims) == target_spatial_dims[:2]:
                return data_array
            # Transpose to correct order if needed
            if all(dim in data_array.dims for dim in target_spatial_dims[:2]):
                return data_array.transpose(
                    target_spatial_dims[0], target_spatial_dims[1]
                )

        # Handle edge cases
        if data_array.ndim == 0:
            return xr.DataArray([[data_array.item()]], dims=["y", "x"])
        elif data_array.ndim == 1:
            return data_array.expand_dims("y", axis=0)

        # For >2D data, reorder to put spatial dims last, then squeeze
        if data_array.ndim > 2 and len(target_spatial_dims) >= 2:
            current_dims = list(data_array.dims)

            try:
                # Find positions of target spatial dimensions
                target_dims = target_spatial_dims[:2]
                if all(dim in current_dims for dim in target_dims):
                    # Separate spatial and non-spatial dimensions
                    spatial_positions = [current_dims.index(dim) for dim in target_dims]
                    non_spatial_positions = [
                        i
                        for i in range(len(current_dims))
                        if i not in spatial_positions
                    ]

                    # Create new dimension order: non-spatial first, then spatial
                    new_order = non_spatial_positions + spatial_positions
                    data_array = data_array.transpose(
                        [current_dims[i] for i in new_order]
                    )

                    # Squeeze out non-spatial singleton dimensions
                    while data_array.ndim > 2:
                        # Take first slice of first non-spatial dimension
                        data_array = data_array.isel({data_array.dims[0]: 0})

                    return data_array
            except Exception:
                pass

        # Fallback: keep squeezing until 2D
        while data_array.ndim > 2:
            data_array = data_array.isel({data_array.dims[0]: 0})

        return data_array
