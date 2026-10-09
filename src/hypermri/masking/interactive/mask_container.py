"""Mask container with identical coordinates to source data."""

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ...utils.system.logging import debug_log

DEBUG_LOG = True
DEBUG_FILE = "/tmp/mask_container.log"


def debug_log_mask_application(message, file_nanme=DEBUG_FILE, enabled=DEBUG_LOG):
    """Log mask application debug messages to /tmp/mask_application_debug.log with timestamp."""
    debug_log(
        message=message,
        file_name=file_nanme,
        write_to_file=enabled,
    )


class MaskContainer:
    """
    Mask container with identical coordinates to source data - no transforms needed.

    The key principle: mask buffer uses EXACT same coordinates and dimensions
    as the source data, eliminating need for coordinate transforms.
    """

    def __init__(
        self,
        name: str,
        template_data: xr.DataArray,
        color: str = "red",
        alpha: float = 0.4,
    ):
        """
        Create mask container with identical coordinates to template data.

        Parameters
        ----------
        name : str
            Name of the mask
        template_data : xr.DataArray
            Template data array to copy coordinates from
        color : str, default 'red'
            Color for visualization
        alpha : float, default 0.4
            Alpha transparency for visualization
        """
        self.name = name

        # Critical: Use EXACT same coordinates and dimensions as template
        self.buffer = xr.DataArray(
            np.zeros(template_data.shape, dtype=bool),
            coords=template_data.coords.copy(),  # Identical coordinates!
            dims=template_data.dims,  # Identical dimensions!
            name=f"{name}_mask",
        )

        # Store metadata in buffer attributes
        self.buffer.attrs.update(
            {
                "color": color,
                "alpha": alpha,
                "finalized": False,
                "creation_time": datetime.datetime.now().isoformat(),
            }
        )

        # Convenience access to metadata
        self.metadata = self.buffer.attrs

        # Track visual patches for cleanup
        self.visual_patches: List[Any] = []

    @property
    def finalized(self) -> bool:
        """Check if mask is finalized."""
        return self.buffer.attrs.get("finalized", False)

    @property
    def color(self) -> str:
        """Get mask color."""
        return self.buffer.attrs.get("color", "red")

    @color.setter
    def color(self, value: str):
        """Set mask color."""
        self.buffer.attrs["color"] = value

    @property
    def alpha(self) -> float:
        """Get mask alpha."""
        return self.buffer.attrs.get("alpha", 0.4)

    @alpha.setter
    def alpha(self, value: float):
        """Set mask alpha."""
        self.buffer.attrs["alpha"] = value

    def apply_lasso_selection(
        self, lasso_mask_2d: np.ndarray, slice_selector: Dict[str, Any]
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Apply lasso selection directly to mask buffer with fixed coordinate alignment.

        Parameters
        ----------
        lasso_mask_2d : np.ndarray
            2D boolean mask from lasso selection in plot coordinate order
        slice_selector : Dict[str, Any]
            Slice selector dict (e.g., {'slice_idx': 5})
        Returns
        -------
        tuple(bool, Optional[np.ndarray])
            Success flag and the aligned mask that was applied (for undo support)
        """
        try:
            # ENHANCED DEBUGGING: Comprehensive slice selector validation
            # Validate slice selector
            for dim, value in slice_selector.items():
                if dim in self.buffer.dims:
                    max_index = self.buffer.sizes[dim] - 1
                    if value > max_index:
                        debug_log_mask_application(
                            message=f"❌ INVALID: {dim}={value} > max={max_index}",
                        )
                        return False
                    else:
                        pass

                else:
                    debug_log_mask_application(
                        message=f"❌ MISSING: {dim} not in buffer dims {self.buffer.dims}",
                    )

            # Get the exact same slice from buffer as from source data
            buffer_slice = self.buffer.isel(**slice_selector)

            # Align mask orientation to buffer slice
            final_mask_2d = self._align_mask_with_buffer_slice(
                lasso_mask_2d, buffer_slice
            )

            # BEFORE applying mask - log current buffer state
            np.sum(self.buffer.values)

            # Apply the aligned mask to the buffer
            if buffer_slice.ndim == 2:
                # Simple 2D case - direct assignment
                np.sum(buffer_slice.data)
                buffer_slice.data |= final_mask_2d
                np.sum(buffer_slice.data)
            else:
                # Multi-dimensional case - need to handle extra dimensions properly
                # Create slice for spatial dimensions only
                spatial_slice = tuple(
                    slice(None) if i < 2 else 0 for i in range(buffer_slice.ndim)
                )

                # Get spatial view of buffer data
                spatial_view = buffer_slice.data[spatial_slice]

                # Ensure shapes match for spatial view
                if spatial_view.shape != final_mask_2d.shape:
                    error_msg = f"Spatial view shape mismatch: {spatial_view.shape} vs {final_mask_2d.shape}"
                    debug_log_mask_application(
                        message=f"❌ {error_msg}",
                    )
                    return False

                # Apply mask to spatial view
                np.sum(spatial_view)
                spatial_view |= final_mask_2d
                np.sum(spatial_view)

            # Count pixels for logging
            np.sum(final_mask_2d)

            # AFTER applying mask - validate total buffer change
            np.sum(self.buffer.values)

            # FINAL VALIDATION: Check that the specified slice now has the expected data
            if "slice_idx" in slice_selector:
                validation_slice = self.buffer.isel(
                    slice_idx=slice_selector["slice_idx"]
                )
                np.sum(validation_slice.values)

            return True, final_mask_2d.copy()

        except Exception:
            return False, None

    def remove_last_lasso_region(
        self, last_region_mask: np.ndarray, slice_selector: Dict[str, Any]
    ) -> bool:
        """
        Remove last lasso region from buffer using XOR operation.

        Parameters
        ----------
        last_region_mask : np.ndarray
            2D boolean mask of the last region to remove
        slice_selector : Dict[str, Any]
            Slice selector dict for the region location

        Returns
        -------
        bool
            True if removal successful
        """
        try:
            buffer_slice = self.buffer.isel(**slice_selector)

            # Align mask orientation to buffer slice
            aligned_mask = self._align_mask_with_buffer_slice(
                last_region_mask, buffer_slice
            )

            # XOR operation to remove the region (toggle the last addition)
            buffer_slice.data ^= aligned_mask

            np.sum(last_region_mask)

            return True

        except Exception as e:
            debug_log_mask_application(
                message=f"❌ Error removing last lasso region for {self.name}: {e}",
            )
            return False

    def _align_mask_with_buffer_slice(
        self, mask_2d: np.ndarray, buffer_slice: xr.DataArray
    ) -> np.ndarray:
        """
        Align a 2D mask with the orientation of a buffer slice.

        Parameters
        ----------
        mask_2d : np.ndarray
            Mask produced in plot coordinate order.
        buffer_slice : xr.DataArray
            Slice of the mask buffer obtained via isel.

        Returns
        -------
        np.ndarray
            Mask aligned to the buffer slice orientation.
        """
        buffer_shape = buffer_slice.shape
        if len(buffer_shape) > 2:
            buffer_shape = buffer_shape[:2]

        if mask_2d.shape == buffer_shape:
            return mask_2d

        if len(mask_2d.shape) == 2 and mask_2d.shape == buffer_shape[::-1]:
            transposed = mask_2d.T
            if np.sum(mask_2d) != np.sum(transposed):
                raise ValueError("Mask transpose altered pixel count.")
            return transposed

        raise ValueError(
            f"Incompatible mask shape {mask_2d.shape} for buffer slice {buffer_slice.shape}"
        )

    def clear_buffer(self):
        """Clear all mask data from buffer."""
        try:
            self.buffer.data.fill(False)
            self.buffer.attrs["finalized"] = False
            self.visual_patches.clear()

        except Exception as e:
            debug_log_mask_application(
                message=f"❌ Error clearing buffer for {self.name}: {e}",
            )

    def finalize(self):
        """Mark mask as finalized."""
        self.buffer.attrs["finalized"] = True
        self.buffer.attrs["finalized_timestamp"] = datetime.datetime.now().isoformat()

        np.sum(self.buffer.values)

    def get_mask_for_slice(
        self, slice_selector: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Get 2D mask for specific slice.

        Parameters
        ----------
        slice_selector : Dict[str, Any]
            Slice selector dict

        Returns
        -------
        np.ndarray or None
            2D boolean mask for the slice
        """
        try:
            return self.buffer.isel(**slice_selector).values
        except Exception:
            return None

    def get_total_pixels(self) -> int:
        """Get total number of masked pixels."""
        return int(np.sum(self.buffer.values))

    def add_visual_patch(self, patch):
        """Add visual patch for cleanup tracking."""
        self.visual_patches.append(patch)

    def clear_visual_patches(self):
        """Clear all visual patches."""
        for patch in self.visual_patches:
            try:
                if hasattr(patch, "remove"):
                    patch.remove()
            except:
                pass  # Ignore removal errors
        self.visual_patches.clear()

    def validate_coordinates(self, template_data: xr.DataArray) -> bool:
        """
        Validate that mask coordinates match template data coordinates.

        Parameters
        ----------
        template_data : xr.DataArray
            Template to compare against

        Returns
        -------
        bool
            True if coordinates match exactly
        """
        try:
            # Check dimensions match
            if self.buffer.dims != template_data.dims:
                return False

            # Check shapes match
            if self.buffer.shape != template_data.shape:
                return False

            # Check coordinate keys match
            buffer_coord_keys = set(self.buffer.coords.keys())
            template_coord_keys = set(template_data.coords.keys())

            if buffer_coord_keys != template_coord_keys:
                return False
            return True

        except Exception as e:
            debug_log_mask_application(
                message=f"❌ Error validating coordinates for {self.name}: {e}",
            )
            return False

    def __repr__(self) -> str:
        """String representation of mask container."""
        total_pixels = self.get_total_pixels()
        finalized = self.finalized
        return (
            f"MaskContainer(name='{self.name}', pixels={total_pixels}, "
            f"finalized={finalized}, shape={self.buffer.shape})"
        )


class CoordinateBasedMaskContainer:
    """
    Coordinate-based mask storage that preserves exact slice locations.

    Instead of using a boolean buffer that can lose slice information,
    this stores mask regions as coordinate-based polygons that maintain
    their exact 3D position in the coordinate system.
    """

    def __init__(
        self,
        name: str,
        template_data: xr.DataArray,
        color: str = "red",
        alpha: float = 0.4,
    ):
        """
        Create coordinate-based mask container.

        Parameters
        ----------
        name : str
            Name of the mask
        template_data : xr.DataArray
            Template data array to get coordinate system from
        color : str, default 'red'
            Color for visualization
        alpha : float, default 0.4
            Alpha transparency for visualization
        """
        self.name = name

        # Store coordinate system reference
        self.coord_system = {
            "dims": template_data.dims,
            "coords": {dim: template_data.coords[dim] for dim in template_data.dims},
            "shape": template_data.shape,
        }

        # Store mask regions as coordinate-based polygons
        # Each region: {'slice_coords': {...}, 'polygon_coords': [...], 'pixels': int}
        self.regions: List[Dict[str, Any]] = []

        # Metadata
        self.color = color
        self.alpha = alpha
        self.finalized = False
        self.creation_time = datetime.datetime.now().isoformat()

        # Track visual patches for cleanup
        self.visual_patches: List[Any] = []

    def add_region(
        self,
        slice_coords: Dict[str, Any],
        polygon_coords: np.ndarray,
        plot_dims: Tuple[str, str],
    ) -> bool:
        """
        Add a mask region using coordinate-based storage.

        Parameters
        ----------
        slice_coords : Dict[str, Any]
            Exact slice coordinates where mask was drawn (e.g., {'slice_idx': 5})
        polygon_coords : np.ndarray
            Polygon coordinates in data coordinate space
        plot_dims : Tuple[str, str]
            Plot dimensions used (y_dim, x_dim)

        Returns
        -------
        bool
            True if region added successfully
        """
        try:
            # Create mask region record with exact coordinates
            region = {
                "slice_coords": slice_coords.copy(),
                "polygon_coords": polygon_coords.copy(),
                "plot_dims": plot_dims,
                "pixels": len(polygon_coords),
                "timestamp": datetime.datetime.now().isoformat(),
            }

            self.regions.append(region)

            return True

        except Exception:
            return False

    def get_mask_for_slice(self, slice_coords: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get mask for specific slice coordinates.

        Parameters
        ----------
        slice_coords : Dict[str, Any]
            Slice coordinates to get mask for

        Returns
        -------
        np.ndarray or None
            2D boolean mask for the slice, or None if no mask on this slice
        """
        try:
            # Find regions that match the requested slice coordinates
            matching_regions = []
            for region in self.regions:
                region_slice = region["slice_coords"]
                # Check if slice coordinates match
                if all(
                    region_slice.get(k) == v
                    for k, v in slice_coords.items()
                    if k in region_slice
                ):
                    matching_regions.append(region)

            if not matching_regions:
                return None

            # Get the plot dimensions from the first region (should be consistent)
            plot_dims = matching_regions[0]["plot_dims"]
            plot_dim_y, plot_dim_x = plot_dims

            # Get slice shape from coordinate system
            slice_shape = (
                self.coord_system["coords"][plot_dim_y].size,
                self.coord_system["coords"][plot_dim_x].size,
            )

            # Create combined mask for all regions on this slice
            combined_mask = np.zeros(slice_shape, dtype=bool)

            for region in matching_regions:
                # Convert stored coordinates to mask
                polygon_coords = region["polygon_coords"]
                if len(polygon_coords) > 0:
                    # Create boolean mask from polygon coordinates
                    from matplotlib.path import Path

                    # Create coordinate grid
                    y_coords = np.arange(slice_shape[0])
                    x_coords = np.arange(slice_shape[1])
                    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

                    # Test points against polygon
                    path = Path(polygon_coords)
                    points = np.column_stack([xx.ravel(), yy.ravel()])
                    mask_1d = path.contains_points(points)
                    region_mask = mask_1d.reshape(slice_shape)

                    # Add to combined mask
                    combined_mask |= region_mask

            return combined_mask

        except Exception:
            return None

    def get_total_pixels(self) -> int:
        """Get total number of pixels across all regions."""
        return sum(region["pixels"] for region in self.regions)

    def get_slice_distribution(self) -> Dict[str, int]:
        """
        Get distribution of mask pixels across slices.

        Returns
        -------
        Dict[str, int]
            Mapping of slice coordinates to pixel counts
        """
        distribution = {}
        for region in self.regions:
            slice_key = str(region["slice_coords"])
            if slice_key not in distribution:
                distribution[slice_key] = 0
            distribution[slice_key] += region["pixels"]
        return distribution

    def finalize(self):
        """Mark mask as finalized."""
        self.finalized = True

    def clear_buffer(self):
        """Clear all mask regions."""
        self.regions.clear()
        self.finalized = False

    @property
    def buffer(self) -> xr.DataArray:
        """
        Create traditional buffer view for compatibility.

        This converts coordinate-based regions back to a boolean buffer
        when needed for compatibility with existing systems.
        """
        try:
            # Create empty buffer with same structure as original
            buffer = xr.DataArray(
                np.zeros(self.coord_system["shape"], dtype=bool),
                coords=self.coord_system["coords"],
                dims=self.coord_system["dims"],
                name=f"{self.name}_buffer",
            )

            # Fill buffer with all regions
            for region in self.regions:
                slice_coords = region["slice_coords"]
                try:
                    # Get the 2D mask for this slice
                    mask_2d = self.get_mask_for_slice(slice_coords)
                    if mask_2d is not None:
                        # Apply to buffer
                        buffer_slice = buffer.isel(**slice_coords)
                        buffer_slice.data |= mask_2d
                except Exception as e:
                    debug_log_mask_application(
                        message=f"❌ Error applying region to buffer for {self.name}: {e}",
                    )

            # Add metadata
            buffer.attrs.update(
                {
                    "color": self.color,
                    "alpha": self.alpha,
                    "finalized": self.finalized,
                    "creation_time": self.creation_time,
                }
            )

            return buffer

        except Exception as e:
            debug_log_mask_application(
                message=f"❌ Error creating buffer for {self.name}: {e}",
            )
            # Fallback: return empty buffer
            return xr.DataArray(
                np.zeros(self.coord_system["shape"], dtype=bool),
                coords=self.coord_system["coords"],
                dims=self.coord_system["dims"],
                name=f"{self.name}_empty_buffer",
            )
