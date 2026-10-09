"""
Mask application system for cross-dataset masking with coordinate alignment.

This module handles the application of masks created on one dataset to other datasets,
with automatic coordinate system alignment and interpolation as needed.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from hypermri.utils.coordinates.grids import get_rps_grid, get_xyz_grid
from hypermri.utils.system.logging import debug_log, init_default_logger

logger = init_default_logger(__name__)
DEBUG_LOG = True
DEBUG_FILE = "/tmp/mask_application_debug.log"


def debug_log_mask_application(message, enabled=DEBUG_LOG):
    """Log mask application debug messages to /tmp/mask_application_debug.log with timestamp."""
    debug_log(
        message=message,
        file_name=DEBUG_FILE,
        write_to_file=enabled,
    )


class MaskApplicator:
    """Apply masks across datasets with automatic coordinate alignment.

    This class handles the application of 3D masks created on one MRI dataset
    to other datasets, automatically handling coordinate system differences
    and spatial alignment through interpolation.

    The applicator leverages the existing coordinate infrastructure in
    hypermri.utils.coordinates to handle complex coordinate transformations
    including tilted/rotated slices.
    """

    def __init__(self, source_experiment, mask_data: xr.DataArray):
        """Initialize the mask applicator.

        Parameters
        ----------
        source_experiment : MRIExperiment or BaseAnatomy
            The experiment object where the mask was originally created
        mask_data : xr.DataArray
            The 3D mask data with dual coordinate support (from create_mask())
        """
        self.source_experiment = source_experiment
        self.mask_data = mask_data
        self.source_xyz_grid = None
        self._cache = {}  # Cache for expensive coordinate calculations

        # Extract source coordinate information
        self._initialize_source_coordinates()

    def _initialize_source_coordinates(self):
        """Initialize coordinate information from the source experiment."""
        try:
            # Try to get existing XYZ coordinates from mask data
            if all(coord in self.mask_data.coords for coord in ["x", "y", "z"]):
                raw = {
                    "x": self.mask_data.coords["x"].values,
                    "y": self.mask_data.coords["y"].values,
                    "z": self.mask_data.coords["z"].values,
                }
                self.source_xyz_grid = self._normalize_xyz_grid(raw)
            else:
                # Generate XYZ coordinates using existing infrastructure
                raw = self._get_xyz_coordinates(self.source_experiment)
                if raw is not None:
                    self.source_xyz_grid = self._normalize_xyz_grid(raw)
                else:
                    debug_log_mask_application(
                        "ERROR: Failed to generate XYZ coordinates"
                    )
                    self.source_xyz_grid = None

            # Verification: compare stored coordinates vs freshly computed grids
            if self.source_xyz_grid is not None:
                try:
                    fresh_xyz = get_xyz_grid(self.source_experiment)
                    fresh_xyz_dict = {
                        "x": np.asarray(fresh_xyz[0]),
                        "y": np.asarray(fresh_xyz[1]),
                        "z": np.asarray(fresh_xyz[2]),
                    }
                    diffs_xyz = {
                        axis: (
                            float(
                                np.min(self.source_xyz_grid[axis])
                                - np.min(fresh_xyz_dict[axis])
                            ),
                            float(
                                np.max(self.source_xyz_grid[axis])
                                - np.max(fresh_xyz_dict[axis])
                            ),
                        )
                        for axis in ("x", "y", "z")
                    }

                    # Also compare against RPS grid ranges to detect mislabeled coordinates
                    rps_grid = get_rps_grid(self.source_experiment)
                    rps_dict = {
                        "x": np.asarray(rps_grid[0]),
                        "y": np.asarray(rps_grid[1]),
                        "z": np.asarray(rps_grid[2]),
                    }
                    diffs_rps = {
                        axis: (
                            float(
                                np.min(self.source_xyz_grid[axis])
                                - np.min(rps_dict[axis])
                            ),
                            float(
                                np.max(self.source_xyz_grid[axis])
                                - np.max(rps_dict[axis])
                            ),
                        )
                        for axis in ("x", "y", "z")
                    }

                    logger.debug(
                        "MASK DEBUG → source XYZ min/max delta vs computed xyz: %s",
                        diffs_xyz,
                    )
                    logger.debug(
                        "MASK DEBUG → source XYZ min/max delta vs RPS grid: %s",
                        diffs_rps,
                    )
                except Exception as compare_err:
                    logger.debug(
                        "MASK DEBUG → coordinate comparison skipped (%s)", compare_err
                    )

        except Exception as e:
            debug_log_mask_application(f"ERROR initializing source coordinates: {e}")
            logger.warning(f"Could not initialize source coordinates: {e}")
            self.source_xyz_grid = None

    def _normalize_xyz_grid(self, grid):
        """Normalize various xyz-grid representations into a dict with numpy arrays.

        Accepts:
        - dict with keys 'x','y','z' (values can be 1D or full-grid arrays)
        - xarray.DataArray or xarray.Dataset coords (with .coords)
        - tuple/list of arrays (x,y,z)
        - numpy arrays
        Returns dict {'x': arr, 'y': arr, 'z': arr} or raises ValueError.
        """
        if grid is None:
            raise ValueError("No coordinate grid provided")

        # If already a dict with x,y,z
        if isinstance(grid, dict):
            try:
                return {
                    "x": np.asarray(grid["x"]),
                    "y": np.asarray(grid["y"]),
                    "z": np.asarray(grid["z"]),
                }
            except Exception as e:
                raise ValueError(f"Invalid dict grid format: {e}")

        # If it's an xarray-like object with coords
        # If it's an xarray-like object with coords
        try:
            if isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
                if all(k in grid.coords for k in ["x", "y", "z"]):
                    return {
                        "x": np.asarray(grid.coords["x"].values),
                        "y": np.asarray(grid.coords["y"].values),
                        "z": np.asarray(grid.coords["z"].values),
                    }
        except Exception:
            pass

        # If it's a tuple/list like (x,y,z)
        if isinstance(grid, (list, tuple)) and len(grid) == 3:
            try:
                return {
                    "x": np.asarray(grid[0]),
                    "y": np.asarray(grid[1]),
                    "z": np.asarray(grid[2]),
                }
            except Exception as e:
                raise ValueError(f"Invalid tuple/list grid format: {e}")

        # If it's a numpy array with shape matching (3, ...) or (...,3)
        try:
            arr = np.asarray(grid)
            if arr.ndim >= 1:
                # Not enough info to split into x,y,z
                raise ValueError(
                    "Ambiguous numpy array grid - expected dict or (x,y,z) tuple"
                )
        except Exception:
            pass

        raise ValueError("Unsupported xyz grid format")

    def _get_xyz_coordinates(self, experiment):
        """Get XYZ coordinates for an experiment or DataArray using existing infrastructure.

        Parameters
        ----------
        experiment : MRIExperiment or xr.DataArray
            Either an experiment object or a DataArray directly

        Returns
        -------
        dict or None
            Dictionary with 'x', 'y', 'z' keys containing coordinate arrays
        """
        # Case 1: If it's already a DataArray, check for coordinates
        if isinstance(experiment, xr.DataArray):
            if all(coord in experiment.coords for coord in ["x", "y", "z"]):
                return {
                    "x": experiment.coords["x"].values,
                    "y": experiment.coords["y"].values,
                    "z": experiment.coords["z"].values,
                }
            else:
                logger.warning(
                    "DataArray provided but missing x,y,z coordinates - cannot extract XYZ grid"
                )
                return None

        # Case 2: Try experiment-level method
        try:
            return get_xyz_grid(experiment)
        except Exception:
            pass

            # Case 3: Extract coordinates directly from DataArray if available
            data_array = None
            if hasattr(experiment, "data_array") and experiment.data_array is not None:
                data_array = experiment.data_array
            elif hasattr(experiment, "reco_data") and experiment.reco_data is not None:
                data_array = experiment.reco_data
            elif hasattr(experiment, "reco_data") and experiment.reco_data is not None:
                data_array = experiment.reco_data

            if (
                data_array is not None
                and "x" in data_array.coords
                and "y" in data_array.coords
                and "z" in data_array.coords
            ):
                logger.info("Using existing XYZ coordinates from data array")
                return {
                    "x": data_array.coords["x"].values,
                    "y": data_array.coords["y"].values,
                    "z": data_array.coords["z"].values,
                }

            # Case 3: Generate coordinates using method if available
            elif hasattr(experiment, "method"):
                if data_array is not None:
                    return get_xyz_grid(data_array, experiment.method)
                else:
                    # Try to use the experiment object directly
                    return get_xyz_grid(data_obj=experiment)
            else:
                logger.warning(
                    "Cannot extract XYZ coordinates: no coordinates in DataArray and no .method attribute"
                )
                return None

        except Exception as e:
            logger.error(f"Error generating XYZ coordinates: {e}")
            return None

    def apply_to_dataset(
        self,
        target_experiment,
        alignment_method: str = "scanner_coords",
        interpolation_method: str = "nearest",
        mask_name: Optional[str] = None,
        preserve_dims: Optional[list] = None,
        select_dims: Optional[dict] = None,
    ) -> xr.DataArray:
        """Apply mask to target dataset with coordinate alignment.

        Parameters
        ----------
        target_experiment : MRIExperiment or BaseAnatomy
            Target experiment to apply mask to
        alignment_method : str, default "scanner_coords"
            Method for alignment: "scanner_coords", "object_coords", or "simple"
        interpolation_method : str, default "nearest"
            Interpolation method: "nearest", "linear", or "cubic"
        mask_name : str, optional
            Name to assign to the applied mask
        preserve_dims : str or list of str, optional
            Non-spatial dimension name(s) to preserve in the output mask.
            Can be a single string or list of strings.
        select_dims : dict, optional
            Dictionary mapping dimension names to indices for selection before masking

        Returns
        -------
        xr.DataArray
            Mask data aligned to target dataset coordinates
        """
        cache_key = f"{id(target_experiment)}_{alignment_method}_{interpolation_method}_{preserve_dims}_{select_dims}"
        if cache_key in self._cache:
            logger.debug("Using cached mask application result")
            return self._cache[cache_key]

        try:
            if alignment_method == "scanner_coords":
                result = self._apply_with_scanner_coords(
                    target_experiment,
                    interpolation_method,
                    preserve_dims=preserve_dims,
                    select_dims=select_dims,
                )
            elif alignment_method == "object_coords":
                result = self._apply_with_object_coords(
                    target_experiment, interpolation_method
                )
            elif alignment_method == "simple":
                result = self._apply_simple_alignment(target_experiment)
            else:
                raise ValueError(f"Unknown alignment method: {alignment_method}")

            # Add metadata
            if mask_name:
                result.name = mask_name
            result.attrs.update(
                {
                    "source_experiment": str(self.source_experiment),
                    "alignment_method": alignment_method,
                    "interpolation_method": interpolation_method,
                    "applied_from_mask": (
                        str(self.mask_data.name)
                        if self.mask_data.name
                        else "unnamed_mask"
                    ),
                }
            )

            # Cache the result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error applying mask to target dataset: {e}")
            raise

    def _apply_with_scanner_coords(
        self,
        target_experiment,
        interpolation_method: str,
        preserve_dims: Optional[list] = None,
        select_dims: Optional[dict] = None,
    ) -> xr.DataArray:
        """Apply mask using scanner coordinate alignment."""

        logger.debug("=" * 60)
        logger.debug("=== MaskApplicator: Scanner coordinate alignment START ===")
        logger.debug("=" * 60)

        # Normalize preserve_dims to list
        if preserve_dims is not None:
            if isinstance(preserve_dims, str):
                if preserve_dims.strip():  # Not empty
                    preserve_dims = [preserve_dims]
                else:
                    preserve_dims = None
            elif not isinstance(preserve_dims, (list, tuple)):
                raise TypeError(
                    f"preserve_dims must be a string or list of strings, got {type(preserve_dims).__name__}"
                )

        if self.source_xyz_grid is None:
            debug_log_mask_application("ERROR: Source XYZ coordinates not available")
            raise ValueError(
                "Source XYZ coordinates not available for scanner coordinate alignment"
            )

        # Log source coordinate information
        logger.debug("SOURCE (mask) coordinate info:")
        logger.debug(f"  Source experiment orientation:")
        if hasattr(self.source_experiment, "read_orientation"):
            logger.debug(
                f"    read_orientation: {self.source_experiment.read_orientation}"
            )
            logger.debug(
                f"    slice_orientation: {self.source_experiment.slice_orientation}"
            )
            logger.debug(f"    patient_pos: {self.source_experiment.patient_pos}")
        logger.debug(f"  Source XYZ ranges:")
        logger.debug(
            f"    x: [{np.min(self.source_xyz_grid['x']):.2f}, {np.max(self.source_xyz_grid['x']):.2f}] mm"
        )
        logger.debug(
            f"    y: [{np.min(self.source_xyz_grid['y']):.2f}, {np.max(self.source_xyz_grid['y']):.2f}] mm"
        )
        logger.debug(
            f"    z: [{np.min(self.source_xyz_grid['z']):.2f}, {np.max(self.source_xyz_grid['z']):.2f}] mm"
        )
        logger.debug(
            f"  Mask data shape: {self.mask_data.shape}, dims: {self.mask_data.dims}"
        )

        # Get target XYZ coordinates - prefer direct coords on target data objects
        target_xyz_grid = None
        target_data_candidate = None

        # If target_experiment is already a DataArray, use it directly
        if isinstance(target_experiment, xr.DataArray):
            target_data_candidate = target_experiment
        # Otherwise, extract from experiment object
        elif (
            hasattr(target_experiment, "reco_data")
            and target_experiment.reco_data is not None
        ):
            target_data_candidate = target_experiment.reco_data
        elif (
            hasattr(target_experiment, "data_array")
            and target_experiment.data_array is not None
        ):
            target_data_candidate = target_experiment.data_array
        elif (
            hasattr(target_experiment, "reco_data")
            and target_experiment.reco_data is not None
        ):
            target_data_candidate = target_experiment.reco_data

        if target_data_candidate is not None and all(
            k in target_data_candidate.coords for k in ["x", "y", "z"]
        ):
            target_xyz_grid = {
                "x": np.asarray(target_data_candidate.coords["x"].values),
                "y": np.asarray(target_data_candidate.coords["y"].values),
                "z": np.asarray(target_data_candidate.coords["z"].values),
            }
        else:
            # Fallback to existing helper (may return dict or None)
            target_xyz_grid = self._get_xyz_coordinates(target_experiment)

        if target_xyz_grid is None:
            debug_log_mask_application(
                "ERROR: Could not generate target XYZ coordinates"
            )
            raise ValueError("Could not generate XYZ coordinates for target experiment")

        # Log target coordinate information
        logger.debug("\nTARGET coordinate info:")
        logger.debug(f"  Target experiment orientation:")
        if hasattr(target_experiment, "read_orientation"):
            logger.debug(f"    read_orientation: {target_experiment.read_orientation}")
            logger.debug(
                f"    slice_orientation: {target_experiment.slice_orientation}"
            )
            logger.debug(f"    patient_pos: {target_experiment.patient_pos}")
            if hasattr(target_experiment, "read_offset_mm"):
                logger.debug(
                    f"    Offsets: read={target_experiment.read_offset_mm:.2f}, "
                    f"phase={target_experiment.phase_offset_mm:.2f}, "
                    f"slice={target_experiment.slice_offset_mm:.2f} mm"
                )
        logger.debug(f"  Target XYZ ranges:")
        logger.debug(
            f"    x: [{np.min(target_xyz_grid['x']):.2f}, {np.max(target_xyz_grid['x']):.2f}] mm"
        )
        logger.debug(
            f"    y: [{np.min(target_xyz_grid['y']):.2f}, {np.max(target_xyz_grid['y']):.2f}] mm"
        )
        logger.debug(
            f"    z: [{np.min(target_xyz_grid['z']):.2f}, {np.max(target_xyz_grid['z']):.2f}] mm"
        )

        # Check coordinate space overlap
        logger.debug("\nCoordinate space OVERLAP analysis:")
        for axis in ["x", "y", "z"]:
            src_min, src_max = np.min(self.source_xyz_grid[axis]), np.max(
                self.source_xyz_grid[axis]
            )
            tgt_min, tgt_max = np.min(target_xyz_grid[axis]), np.max(
                target_xyz_grid[axis]
            )
            overlap_min = max(src_min, tgt_min)
            overlap_max = min(src_max, tgt_max)
            if overlap_max > overlap_min:
                overlap_pct = 100 * (overlap_max - overlap_min) / (src_max - src_min)
                logger.debug(
                    f"  {axis}: OVERLAP [{overlap_min:.2f}, {overlap_max:.2f}] mm ({overlap_pct:.1f}% of source)"
                )
            else:
                logger.debug(
                    f"  {axis}: NO OVERLAP (source: [{src_min:.2f}, {src_max:.2f}], target: [{tgt_min:.2f}, {tgt_max:.2f}])"
                )

        # Get target data array for shape information
        target_data = self._get_target_data_array(target_experiment)

        # Store original target data for dimension preservation
        original_target_data = target_data

        # Apply dimension selection if requested
        if select_dims is not None:
            logger.info(f"Selecting dimensions: {select_dims}")
            target_data = target_data.isel(**select_dims)

        # Extract spatial dimensions only (those with x,y,z coordinates)
        if hasattr(target_data, "coords") and "x" in target_data.coords:
            x_coord = target_data.coords["x"]
            spatial_dims = list(x_coord.dims)

        else:
            spatial_dims = list(target_data.dims[-3:])  # Assume last 3 dims are spatial

        # Log spatial dimension ordering for debugging
        logger.debug("MASK DEBUG → mask dims=%s", self.mask_data.dims)
        logger.debug("MASK DEBUG → target spatial dims=%s", spatial_dims)

        # Get spatial shapes for coordinate flattening
        source_spatial_dims = [
            dim for dim in self.mask_data.dims if dim in spatial_dims
        ]
        logger.debug("MASK DEBUG → source spatial dims=%s", source_spatial_dims)

        source_spatial_shape = tuple(
            self.mask_data.sizes[dim] for dim in source_spatial_dims
        )
        target_spatial_shape = tuple(target_data.sizes[dim] for dim in spatial_dims)

        logger.debug(f"\nSpatial dimensions:")
        logger.debug(f"  Detected spatial dims: {spatial_dims}")
        logger.debug(f"  Source spatial dims: {source_spatial_dims}")
        logger.debug(f"  Source spatial shape: {source_spatial_shape}")
        logger.debug(f"  Target spatial shape: {target_spatial_shape}")

        # Handle different interpolation methods
        if interpolation_method == "nearest":
            interp_method = "nearest"
        elif interpolation_method == "linear":
            interp_method = "linear"
        elif interpolation_method == "cubic":
            interp_method = "cubic"
        else:
            logger.warning(
                f"Unknown interpolation method {interpolation_method}, using nearest"
            )
            interp_method = "nearest"

        def _flatten_coords_safe(coord_x, coord_y, coord_z, expected_shape):
            """Safely flatten coordinates ensuring consistent size with expected shape."""
            # Convert to numpy
            cx = np.asarray(coord_x)
            cy = np.asarray(coord_y)
            cz = np.asarray(coord_z)

            expected_size = np.prod(expected_shape)

            # Case 1: Full coordinate grids matching expected shape
            if (
                cx.size == expected_size
                and cy.size == expected_size
                and cz.size == expected_size
            ):
                return cx.ravel(), cy.ravel(), cz.ravel()

            # Case 2: Coordinates need to be extracted to match spatial dimensions only
            if cx.ndim > 1 and len(expected_shape) <= cx.ndim:
                try:
                    spatial_slices = tuple([slice(None)] * len(expected_shape))

                    if len(spatial_slices) <= cx.ndim:
                        cx_spatial = cx[spatial_slices]
                        cy_spatial = cy[spatial_slices]
                        cz_spatial = cz[spatial_slices]

                        if (
                            cx_spatial.size == expected_size
                            and cy_spatial.size == expected_size
                            and cz_spatial.size == expected_size
                        ):
                            debug_log_mask_application(
                                "Using spatial subset of coordinates (case 2)"
                            )
                            return (
                                cx_spatial.ravel(),
                                cy_spatial.ravel(),
                                cz_spatial.ravel(),
                            )
                except Exception as e:
                    debug_log_mask_application(f"Failed case 2: {e}")

            # Case 3: 1D coordinate vectors - create meshgrid
            if cx.ndim == 1 and cy.ndim == 1 and cz.ndim == 1:
                try:
                    if len(expected_shape) == 3:
                        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
                        if gx.size == expected_size:
                            return gx.ravel(), gy.ravel(), gz.ravel()
                except Exception as e:
                    debug_log_mask_application(f"Failed case 3: {e}")

            # Case 4: Try to reshape coordinates to match expected shape
            try:
                if (
                    cx.size >= expected_size
                    and cy.size >= expected_size
                    and cz.size >= expected_size
                ):
                    cx_reshaped = cx.ravel()[:expected_size]
                    cy_reshaped = cy.ravel()[:expected_size]
                    cz_reshaped = cz.ravel()[:expected_size]

                    return cx_reshaped, cy_reshaped, cz_reshaped
            except Exception as e:
                debug_log_mask_application(f"Failed case 4: {e}")

            error_msg = (
                f"Cannot flatten coordinates to match expected shape. "
                f"Coord shapes: x={cx.shape}, y={cy.shape}, z={cz.shape}, "
                f"Expected size: {expected_size}, Expected shape: {expected_shape}"
            )
            debug_log_mask_application(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        try:
            # Use spatial shapes for coordinate flattening
            logger.debug(f"\nFlattening coordinates...")
            logger.debug(f"  Interpolation method: {interp_method}")

            source_x_flat, source_y_flat, source_z_flat = _flatten_coords_safe(
                self.source_xyz_grid["x"],
                self.source_xyz_grid["y"],
                self.source_xyz_grid["z"],
                source_spatial_shape,
            )
            logger.debug(f"  Source flattened: {source_x_flat.shape[0]} points")

            target_x_flat, target_y_flat, target_z_flat = _flatten_coords_safe(
                target_xyz_grid["x"],
                target_xyz_grid["y"],
                target_xyz_grid["z"],
                target_spatial_shape,
            )
            logger.debug(f"  Target flattened: {target_x_flat.shape[0]} points")

            logger.debug(
                "MASK DEBUG → mask shape=%s, target spatial shape=%s",
                tuple(self.mask_data.sizes[d] for d in self.mask_data.dims),
                target_spatial_shape,
            )

            if len(source_spatial_dims) == len(self.mask_data.dims):
                # Mask is purely spatial
                source_values = self.mask_data.values.ravel()
            else:
                # Mask has extra (non-spatial) dimensions. Prefer to reduce those dimensions
                # by logical OR (any) so that any True across repetitions/channels/etc is
                # preserved. This avoids choosing a single index (often 0) that may be empty.
                extra_dims = [
                    d for d in self.mask_data.dims if d not in source_spatial_dims
                ]

                try:
                    # Collapse extras using any(); result keeps only spatial dims
                    spatial_mask = self.mask_data.any(dim=extra_dims)
                    source_values = spatial_mask.values.ravel()
                except Exception as e:
                    # If any() fails for some reason, fall back to previous isel strategy
                    debug_log_mask_application(
                        f"Failed to reduce extras with any(): {e} - falling back to isel first-index strategy"
                    )
                    slicing_dict = {}
                    for dim in self.mask_data.dims:
                        if dim not in source_spatial_dims:
                            # Prefer selecting the slice/index that best matches the target
                            try:
                                if (
                                    dim in ("slice_idx", "z")
                                    and target_xyz_grid is not None
                                    and "z" in target_xyz_grid
                                ):
                                    tz = np.asarray(target_xyz_grid["z"]).ravel()
                                    target_z_val = (
                                        float(tz.mean())
                                        if tz.size > 1
                                        else float(tz[0])
                                    )

                                    src_z = None
                                    if (
                                        "z" in self.source_xyz_grid
                                        and self.source_xyz_grid["z"] is not None
                                    ):
                                        src_z = np.asarray(
                                            self.source_xyz_grid["z"]
                                        ).ravel()
                                    elif dim in self.mask_data.coords:
                                        src_z = np.asarray(
                                            self.mask_data.coords[dim].values
                                        ).ravel()

                                    if src_z is not None and src_z.size > 0:
                                        nearest = int(
                                            np.argmin(np.abs(src_z - target_z_val))
                                        )
                                        slicing_dict[dim] = nearest
                                        continue
                            except Exception as e2:
                                debug_log_mask_application(
                                    f"Failed to choose nearest index for dim '{dim}': {e2}"
                                )

                            slicing_dict[dim] = 0

                    if slicing_dict:
                        spatial_mask = self.mask_data.isel(**slicing_dict)
                        source_values = spatial_mask.values.ravel()
                    else:
                        source_values = self.mask_data.values.ravel()

            logger.debug(
                "MASK DEBUG → first source coords=([%.3f, %.3f, %.3f]), mask dims order=%s",
                source_x_flat[0],
                source_y_flat[0],
                source_z_flat[0],
                self.mask_data.dims,
            )

            # Verify consistency
            expected_source_size = np.prod(source_spatial_shape)

            if source_values.size != expected_source_size:
                error_msg = f"Source values size mismatch: got {source_values.size}, expected {expected_source_size}"
                debug_log_mask_application(f"ERROR: {error_msg}")
                raise ValueError(error_msg)

            if source_x_flat.size != source_values.size:
                error_msg = f"Source coordinate/values size mismatch: coords {source_x_flat.size}, values {source_values.size}"
                debug_log_mask_application(f"ERROR: {error_msg}")
                raise ValueError(error_msg)

            # Create source points array for griddata
            source_points = np.stack(
                [source_x_flat, source_y_flat, source_z_flat], axis=-1
            )

            # Create target coordinate grid for interpolation
            target_points = np.stack(
                [target_x_flat, target_y_flat, target_z_flat], axis=-1
            )

            # Verify target consistency
            expected_target_size = np.prod(target_spatial_shape)
            if target_points.shape[0] != expected_target_size:
                error_msg = f"Target points size mismatch: got {target_points.shape[0]}, expected {expected_target_size}"
                debug_log_mask_application(f"ERROR: {error_msg}")
                raise ValueError(error_msg)

            logger.debug(f"\nPerforming griddata interpolation...")
            logger.debug(f"  Source points: {source_points.shape}")
            logger.debug(
                f"  Source values (mask): {source_values.shape}, nonzero: {np.count_nonzero(source_values)}"
            )
            logger.debug(f"  Target points: {target_points.shape}")

            interpolated_mask = griddata(
                points=source_points,
                values=source_values,
                xi=target_points,
                method=interp_method,
                fill_value=0,
            )

            logger.debug(
                f"  Interpolated values: min={interpolated_mask.min():.3f}, max={interpolated_mask.max():.3f}"
            )

            # Reshape to target spatial shape
            interpolated_mask = interpolated_mask.reshape(target_spatial_shape)

            # Convert to boolean mask
            interpolated_mask = interpolated_mask > 0.5
            logger.debug(
                f"  Boolean mask: shape={interpolated_mask.shape}, nonzero count={np.count_nonzero(interpolated_mask)}"
            )
            debug_log_mask_application(
                f"Boolean mask nonzero count: {np.count_nonzero(interpolated_mask)}"
            )

            # Create output DataArray with only spatial coordinates and dimensions
            spatial_coords = {dim: target_data.coords[dim] for dim in spatial_dims}

            result = xr.DataArray(
                interpolated_mask.astype(bool),
                coords=spatial_coords,
                dims=spatial_dims,
                name=(
                    f"applied_mask_{self.mask_data.name}"
                    if self.mask_data.name
                    else "applied_mask"
                ),
            )

            # Preserve additional dimensions if requested
            if preserve_dims is not None and len(preserve_dims) > 0:
                logger.info(f"Preserving dimensions: {preserve_dims}")

                # Validate that requested dims exist in original data
                available_dims = [
                    d for d in preserve_dims if d in original_target_data.dims
                ]
                if len(available_dims) < len(preserve_dims):
                    missing = set(preserve_dims) - set(available_dims)
                    logger.warning(
                        f"Dimensions {missing} not found in target data, skipping"
                    )

                if available_dims:
                    # Broadcast spatial mask to include preserved dimensions
                    # Get the preserved dimension coords from original data
                    preserved_coords = {
                        dim: original_target_data.coords[dim] for dim in available_dims
                    }

                    # Combine with spatial coords
                    all_coords = {**spatial_coords, **preserved_coords}
                    all_dims = list(spatial_dims) + available_dims

                    # Broadcast the mask values to the new shape
                    # Create new shape with preserved dimensions
                    new_shape = tuple(
                        original_target_data.sizes[dim] for dim in all_dims
                    )

                    # Broadcast the spatial mask to all preserved dimension indices
                    # Add new axes for the preserved dimensions
                    mask_with_newaxes = interpolated_mask.astype(bool)
                    for _ in range(len(available_dims)):
                        mask_with_newaxes = mask_with_newaxes[..., np.newaxis]

                    broadcasted_mask = np.broadcast_to(mask_with_newaxes, new_shape)

                    result = xr.DataArray(
                        broadcasted_mask,
                        coords=all_coords,
                        dims=all_dims,
                        name=result.name,
                    )
                    result.attrs.update(result.attrs)  # Preserve metadata

            logger.debug("=" * 60)
            logger.debug("=== MaskApplicator: Scanner coordinate alignment SUCCESS ===")
            logger.debug("=" * 60)
            return result

        except Exception as e:
            debug_log_mask_application(f"FATAL ERROR: {e}")
            debug_log_mask_application("=== Scanner coordinate alignment FAILED ===")
            logger.error(f"Error in scanner coordinate interpolation: {e}")
            raise

    def _apply_with_object_coords(
        self, target_experiment, interpolation_method: str
    ) -> xr.DataArray:
        """Apply mask using object coordinate alignment (read_idx, phase_idx, slice_idx)."""
        target_data = self._get_target_data_array(target_experiment)

        # Simple approach: match dimensions that exist in both
        common_dims = set(self.mask_data.dims) & set(target_data.dims)

        if not common_dims:
            raise ValueError(
                "No common dimensions found between source mask and target data"
            )

        # Align along common dimensions
        aligned_mask = self.mask_data.interp_like(
            target_data, method=interpolation_method
        )

        # Fill any NaN values with False
        aligned_mask = aligned_mask.fillna(False).astype(bool)

        return aligned_mask

    def _apply_simple_alignment(self, target_experiment) -> xr.DataArray:
        """Apply mask using simple shape-based alignment."""
        target_data = self._get_target_data_array(target_experiment)

        # Simple resize/crop to match target shape
        mask_array = self.mask_data.values
        target_shape = target_data.shape

        # Simple center-based alignment
        result_array = np.zeros(target_shape, dtype=bool)

        # Calculate offsets for center alignment
        offsets = []
        slices = []
        for i, (mask_size, target_size) in enumerate(
            zip(mask_array.shape, target_shape)
        ):
            if mask_size <= target_size:
                # Center the smaller mask in the larger target
                offset = (target_size - mask_size) // 2
                offsets.append(offset)
                slices.append(slice(offset, offset + mask_size))
            else:
                # Crop the larger mask to fit the smaller target
                offset = (mask_size - target_size) // 2
                offsets.append(0)
                slices.append(slice(0, target_size))
                # Also need to crop the mask
                mask_array = mask_array[
                    tuple(
                        slice(offset, offset + target_size) if j == i else slice(None)
                        for j in range(len(mask_array.shape))
                    )
                ]

        # Copy mask data to aligned position
        result_array[tuple(slices)] = mask_array.astype(bool)

        # Create output DataArray
        result = xr.DataArray(
            result_array,
            coords=target_data.coords,
            dims=target_data.dims,
            name=(
                f"aligned_mask_{self.mask_data.name}"
                if self.mask_data.name
                else "aligned_mask"
            ),
        )

        return result

    def _get_target_data_array(self, target_experiment):
        """Get the main data array from target experiment or return DataArray directly.

        Parameters
        ----------
        target_experiment : MRIExperiment or xr.DataArray
            Either an experiment object with data_array/reco_data attributes,
            or an xr.DataArray directly

        Returns
        -------
        xr.DataArray
            The target data array
        """
        # If it's already a DataArray, return it directly
        if isinstance(target_experiment, xr.DataArray):
            return target_experiment

        # Otherwise, extract from experiment object
        if (
            hasattr(target_experiment, "data_array")
            and target_experiment.data_array is not None
        ):
            return target_experiment.data_array
        elif (
            hasattr(target_experiment, "reco_data")
            and target_experiment.reco_data is not None
        ):
            return target_experiment.reco_data
        elif (
            hasattr(target_experiment, "reco_data")
            and target_experiment.reco_data is not None
        ):
            return target_experiment.reco_data
        else:
            raise ValueError("No suitable data array found in target experiment")

    def batch_apply(
        self, target_experiments: List, **apply_kwargs
    ) -> Dict[str, xr.DataArray]:
        """Apply mask to multiple datasets.

        Parameters
        ----------
        target_experiments : list
            List of experiment objects or xr.DataArrays to apply mask to
        **apply_kwargs
            Additional arguments passed to apply_to_dataset

        Returns
        -------
        dict
            Dictionary mapping experiment identifiers to applied masks
        """
        results = {}

        for i, target_exp in enumerate(target_experiments):
            try:
                # Generate appropriate identifier
                if isinstance(target_exp, xr.DataArray):
                    # Use DataArray name if available, otherwise use index
                    exp_id = target_exp.name if target_exp.name else f"data_array_{i}"
                else:
                    # Try to get name from experiment object
                    exp_id = getattr(target_exp, "name", f"experiment_{i}")

                logger.info(f"Applying mask to: {exp_id}")

                result = self.apply_to_dataset(target_exp, **apply_kwargs)
                results[exp_id] = result

            except Exception as e:
                logger.error(f"Failed to apply mask to target {i}: {e}")
                results[f"target_{i}_failed"] = None

        return results

    def get_alignment_quality_report(self, target_experiment) -> Dict:
        """Analyze spatial alignment quality between source and target datasets.

        Parameters
        ----------
        target_experiment
            Target experiment to analyze alignment with

        Returns
        -------
        dict
            Report containing alignment quality metrics
        """
        report = {
            "alignment_possible": False,
            "coordinate_overlap": None,
            "resolution_match": None,
            "orientation_match": None,
            "recommended_method": "simple",
        }

        try:
            # Check if scanner coordinate alignment is possible
            target_xyz = self._get_xyz_coordinates(target_experiment)

            if self.source_xyz_grid and target_xyz:
                report["alignment_possible"] = True
                report["recommended_method"] = "scanner_coords"

                # Calculate coordinate overlap
                overlap = self._calculate_coordinate_overlap(
                    self.source_xyz_grid, target_xyz
                )
                report["coordinate_overlap"] = overlap

                # Check resolution matching
                source_res = self._estimate_resolution(self.source_xyz_grid)
                target_res = self._estimate_resolution(target_xyz)
                report["resolution_match"] = np.allclose(
                    source_res, target_res, rtol=0.1
                )

            else:
                # Try object coordinate alignment
                target_data = self._get_target_data_array(target_experiment)
                common_dims = set(self.mask_data.dims) & set(target_data.dims)

                if common_dims:
                    report["alignment_possible"] = True
                    report["recommended_method"] = "object_coords"

        except Exception as e:
            logger.warning(f"Error in alignment quality analysis: {e}")

        return report

    def _calculate_coordinate_overlap(self, coord1, coord2) -> float:
        """Calculate the overlap percentage between two coordinate systems."""
        try:
            overlaps = []
            for dim in ["x", "y", "z"]:
                if dim in coord1 and dim in coord2:
                    min1, max1 = coord1[dim].min(), coord1[dim].max()
                    min2, max2 = coord2[dim].min(), coord2[dim].max()

                    overlap_min = max(min1, min2)
                    overlap_max = min(max1, max2)

                    if overlap_max > overlap_min:
                        overlap = (overlap_max - overlap_min) / max(
                            max1 - min1, max2 - min2
                        )
                        overlaps.append(overlap)
                    else:
                        overlaps.append(0.0)

            return np.mean(overlaps) if overlaps else 0.0

        except Exception:
            return 0.0

    def _estimate_resolution(self, xyz_grid) -> np.ndarray:
        """Estimate voxel resolution from coordinate grid."""
        try:
            resolutions = []
            for dim in ["x", "y", "z"]:
                if dim in xyz_grid:
                    coords = xyz_grid[dim]
                    if coords.size > 1:
                        res = np.mean(np.diff(coords))
                        resolutions.append(abs(res))

            return np.array(resolutions)

        except Exception:
            return np.array([1.0, 1.0, 1.0])  # Default 1mm resolution
