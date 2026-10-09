"""
Mixins for standardizing variable access patterns across the HyperMRI codebase.

This module provides base classes and mixins to ensure consistent handling of:
- Data arrays (standardized naming)
- Path objects (always pathlib.Path)
- Parameter access (unified interface)
"""

import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Set, Union

import xarray as xr


class StandardizedDataArrayMixin:
    """
    Mixin providing standardized data array naming and access patterns with lazy loading.

    This mixin establishes the following standard data array attributes:
    - data_array: Scanner reconstructed data (e.g., Bruker seq2d)
    - reco_data: Raw-reconstructed data (reconstructed from raw k-space)

    All data arrays are lazy-loaded to reduce memory usage during initial object creation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize private attributes for lazy loading
        self._data_array: Optional[xr.DataArray] = (
            None  # Scanner reconstructed (e.g., seq2d)
        )
        # Raw-reconstructed data
        self._reco_data: Optional[xr.DataArray] = None

        # Lazy loading flags
        self._data_array_loaded = False
        self._reco_data_loaded = False

        # Private attributes for backward compatibility
        self._legacy_seq2d: Optional[xr.DataArray] = None
        self._legacy_reco_array: Optional[xr.DataArray] = None

    @property
    def data_array(self) -> Optional[xr.DataArray]:
        """Lazy loading property for scanner reconstructed data (e.g., seq2d)."""
        if not self._data_array_loaded and self._data_array is None:
            self._data_array = self._load_data_array()
            self._data_array_loaded = True
        return self._data_array

    @data_array.setter
    def data_array(self, value: Optional[xr.DataArray]):
        """Setter for data_array."""
        self._data_array = value
        self._data_array_loaded = True

    @property
    def reco_data(self) -> Optional[xr.DataArray]:
        """Lazy loading property for raw-reconstructed data."""
        if not self._reco_data_loaded and self._reco_data is None:
            self._reco_data = self._load_reco_data()
            self._reco_data_loaded = True
        return self._reco_data

    @reco_data.setter
    def reco_data(self, value: Optional[xr.DataArray]):
        """Setter for reco_data."""
        self._reco_data = value
        self._reco_data_loaded = True

    def _load_data_array(self) -> Optional[xr.DataArray]:
        """Override in subclasses to implement lazy loading of scanner data."""
        return None

    def _load_reco_data(self) -> Optional[xr.DataArray]:
        """Override in subclasses to implement lazy loading of reconstructed data."""
        return None

    @property
    def primary_data(self) -> Optional[xr.DataArray]:
        """
        Returns the primary data array for this object.
        Priority: reco_data > data_array
        """
        # Check reco_data first (raw-reconstructed data preferred)
        if hasattr(self, "reco_data") and self.reco_data is not None:
            return self.reco_data
        # Then check data_array (scanner reconstructed)
        elif hasattr(self, "data_array") and self.data_array is not None:
            return self.data_array
        else:
            return None

    def set_primary_data(self, data: xr.DataArray, data_type: str = "reco_data"):
        """
        Set the primary data array with proper type designation.

        Parameters
        ----------
        data : xr.DataArray
            The data array to store
        data_type : str
            Type of data: "reco_data" (raw-reconstructed), "scanner_data" (seq2d), or "legacy"
        """
        # Ensure unit are set to "a.u." if not already present
        if "unit" not in data.attrs:
            data.attrs["unit"] = "a.u."

        if (
            data_type == "reco_data" or data_type == "reconstructed"
        ):  # Support old name for compatibility
            self.reco_data = data
        elif data_type == "scanner_data" or data_type == "seq2d":
            self.data_array = data
        elif data_type == "kspace":
            raise ValueError(
                "kspace_array is no longer supported for permanent storage. Use during reconstruction only."
            )
        elif data_type == "legacy":
            self._legacy_seq2d = data
        else:
            raise ValueError(
                f"Unknown data_type: {data_type}. Use 'reco_data', 'scanner_data', or 'legacy'"
            )

    # Backward compatibility properties
    # @property
    # def seq2d(self) -> Optional[xr.DataArray]:
    #     """Legacy property for backward compatibility. Use data_array instead."""
    #     warnings.warn(
    #         "seq2d is deprecated. Use data_array instead.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )

    #     # Handle xarray DataArray boolean evaluation properly - avoid using
    #     # 'or' with DataArrays
    #     if self._legacy_seq2d is not None:
    #         return self._legacy_seq2d
    #     else:
    #         # Safely return data_array, handling potential None or empty xarray
    #         # cases
    #         try:
    #             if hasattr(self, "data_array") and self.data_array is not None:
    #                 return self.data_array
    #             else:
    #                 return None
    #         except ValueError:
    #             # In case of xarray boolean evaluation issues
    #             return getattr(self, "data_array", None)

    # @seq2d.setter
    # def seq2d(self, value: xr.DataArray):
    #     """Legacy setter for backward compatibility."""
    #     warnings.warn(
    #         "seq2d is deprecated. Use data_array instead.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )

    #     # Handle None values gracefully
    #     if value is None:
    #         self._legacy_seq2d = None
    #         return

    #     # Ensure unit are set to "a.u." if not already present
    #     if "unit" not in value.attrs:
    #         value.attrs["unit"] = "a.u."

    #     self._legacy_seq2d = value
    #     if self.data_array is None:
    #         self.data_array = value

    @property
    def reco_array(self) -> Optional[xr.DataArray]:
        """Legacy property for backward compatibility. Use reco_data instead."""
        warnings.warn(
            "reco_array is deprecated. Use reco_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Handle xarray DataArray boolean evaluation properly - avoid using
        # 'or' with DataArrays
        if self._legacy_reco_array is not None:
            return self._legacy_reco_array
        else:
            # Safely return reco_data, handling potential None or empty xarray
            # cases
            try:
                if hasattr(self, "reco_data") and self.reco_data is not None:
                    return self.reco_data
                else:
                    return None
            except ValueError:
                # In case of xarray boolean evaluation issues
                return getattr(self, "reco_data", None)

    @reco_array.setter
    def reco_array(self, value: xr.DataArray):
        """Legacy setter for backward compatibility."""
        warnings.warn(
            "reco_array is deprecated. Use reco_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Ensure unit are set to "a.u." if not already present
        if "unit" not in value.attrs:
            value.attrs["unit"] = "a.u."

        self._legacy_reco_array = value
        if self.reco_data is None:
            self.reco_data = value


class StandardizedPathMixin:
    """
    Mixin providing standardized path handling using pathlib.Path objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        """Returns the path as a pathlib.Path object."""
        return self._path

    @path.setter
    def path(self, value: Union[str, Path, None]):
        """Sets the path, converting to pathlib.Path if needed."""
        if value is None:
            self._path = None
        elif isinstance(value, (str, Path)):
            self._path = Path(value).resolve()
        else:
            raise TypeError(f"Path must be str, Path, or None, got {type(value)}")


class ParameterAccessMixin:
    """
    Mixin providing unified parameter access interface.
    """

    def get_param(
        self, key: str, default: Any = None, paramfile: str = "method"
    ) -> Any:
        """
        Unified parameter access across all classes.

        Parameters
        ----------
        key : str
            Parameter key to retrieve
        default : Any
            Default value if key not found
        paramfile : str
            Parameter file to search in ("method", "acqp", "visu", "reco")

        Returns
        -------
        Any
            Parameter value or default
        """
        if hasattr(self, "params") and key in getattr(self, "params", {}):
            return self.params.get(key, default)

        if hasattr(self, paramfile):
            param_dict = getattr(self, paramfile, {})
            if isinstance(param_dict, dict):
                return param_dict.get(key, default)

        # Fallback to direct attribute access
        if hasattr(self, key):
            return getattr(self, key, default)

        return default

    def set_param(self, key: str, value: Any, paramfile: str = "method"):
        """
        Set a parameter value in the specified parameter file.

        Parameters
        ----------
        key : str
            Parameter key to set
        value : Any
            Parameter value
        paramfile : str
            Parameter file to modify
        """
        if not hasattr(self, paramfile):
            setattr(self, paramfile, {})

        param_dict = getattr(self, paramfile)
        if isinstance(param_dict, dict):
            param_dict[key] = value
        else:
            # If paramfile is not a dict, create a new dict
            setattr(self, paramfile, {key: value})

        if hasattr(self, "params"):
            self.params[key] = value


class UniformDataPropertiesMixin:
    """
    Mixin providing uniform data access properties across all vendor handlers.

    This mixin standardizes access to:
    - Spatial coordinate grids (RPS and XYZ systems)
    - Frequency axes for spectroscopy data

    Properties are implemented with proper error handling and will return None
    when required data is not available.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize caches for expensive computations
        self._rps_axes_mm = None
        self._rps_extent_mm = None
        self._rps_grid_mm = None
        self._xyz_axes_mm = None
        self._xyz_extent_mm = None
        self._xyz_grid_mm = None

    @property
    def rps_extent_mm(self):
        """Get RPS (read, phase, slice) coordinate extents in mm."""
        try:
            if self._rps_extent_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._rps_extent_mm = grids.get_rps_extent(data_obj=self)
            return self._rps_extent_mm
        except Exception:
            return None

    @property
    def rps_axes_mm(self):
        """Get RPS (read, phase, slice) coordinate axes in mm."""
        try:
            if self._rps_axes_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._rps_axes_mm = grids.get_rps_axes(data_obj=self)
            return self._rps_axes_mm
        except Exception:
            return None

    @property
    def rps_grid_mm(self):
        """Get RPS (read, phase, slice) coordinate grid in mm."""
        try:
            if self._rps_grid_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._rps_grid_mm = grids.get_rps_grid(data_obj=self)
            return self._rps_grid_mm
        except Exception:
            return None

    @property
    def xyz_extent_mm(self):
        """Get XYZ coordinate extents in mm."""
        try:
            if self._xyz_extent_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._xyz_extent_mm = grids.get_xyz_extent(data_obj=self)
            return self._xyz_extent_mm
        except Exception:
            return None

    @property
    def xyz_axes_mm(self):
        """Get XYZ coordinate axes in mm."""
        try:
            if self._xyz_axes_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._xyz_axes_mm = grids.get_xyz_axes(data_obj=self)
            return self._xyz_axes_mm
        except Exception:
            return None

    @property
    def xyz_grid_mm(self):
        """Get XYZ coordinate grid in mm."""
        try:
            if self._xyz_grid_mm is None:
                import hypermri.utils.coordinates.grids as grids

                self._xyz_grid_mm = grids.get_xyz_grid(data_obj=self)
            return self._xyz_grid_mm
        except Exception:
            return None

    @property
    def ppm(self):
        """Get frequency axis in ppm for spectroscopy data."""
        try:
            from hypermri.analysis.spectroscopy.core import get_freq_axis

            return get_freq_axis(data_obj=self, unit="ppm")
        except Exception:
            return None

    @property
    def Hz(self):
        """Get frequency axis in Hz for spectroscopy data."""
        try:
            from hypermri.analysis.spectroscopy.core import get_freq_axis

            return get_freq_axis(data_obj=self, unit="Hz")
        except Exception:
            return None


class StandardizedMRIMixin(
    StandardizedDataArrayMixin,
    StandardizedPathMixin,
    ParameterAccessMixin,
    UniformDataPropertiesMixin,
):
    """
    Combined mixin providing all standardized patterns for MRI classes.
    """

    # Canonical parameter definitions shared across vendors. Keys grouped by logical
    # category to keep the schema human-readable and maintainable.
    PARAMETER_SCHEMA: ClassVar[Dict[str, Iterable[str]]] = {
        "core": (
            "experiment_number",
            "timestamp",
            "vendor",
            "nucleus",
            "method_type",
            "is_spectroscopic",
            "cut_off",
            "git_commit_hash",
        ),
        "subject": (
            "subject_name",
            "subject_id",
            "subject_position",
            "subject_type",
            "patient_position",
        ),
        "geometry": (
            "spatial_dimension",
            "n_slice_packs",
            "n_slices",
            "slice_thickness_mm",
            "slice_gap_mm",
            "slice_orientation",
            "slice_order",
            "slice_offset_mm",
            "slice_offset_mm_array",
            "read_orientation",
            "read_offset_mm",
            "phase_offset_mm",
            "phase_0_offset_mm",
            "phase_1_offset_mm",
            "position_matrix",
            "fov_mm",
            "matrix",
            "encoding_matrix",
            "resolution_mm",
            "measurement_time",
            "center_ppm",
        ),
        "acquisition": (
            "n_receivers",
            "n_repetitions",
            "n_averages",
            "n_echoes",
            "n_acquisition",
            "n_voxels",
            "n_inversion_time",
            "inversion_time_ms",
            "rare_factor",
            "scan_duration_s",
            "bandwidth_hz",
            "bandwidth_ppm",
            "echo_time_ms",
            "repetition_time_ms",
            "frame_time_s",
            "dwell_time_s",
            "sampling_time_axis_s",
            "fat_water_separation",
            "b_values",
            "sequence_name",
            "echo_spacing_ms",
            "effective_te_ms",
        ),
        "hardware": (
            "coil_receive",
            "coil_transmit",
        ),
        "rf": (
            "rf_pulse_params",
            "rfpulse_duration_ms",
            "rfpulse_bandwidth_hz",
            "rfpulse_is_calculated",
            "rfpulse_sharpness",
            "rfpulse_bwfac_hzms",
            "rfpulse_sint",
            "rfpulse_pint",
            "rfpulse_type",
            "rfpulse_rpfac_percent",
            "rfpulse_peakpow_w",
            "rfpulse_shape",
            "rfpulse_refpow_w",
            "flipangle_deg",
            "ref_pulse_params",
            "refpulse_duration_ms",
            "refpulse_bandwidth_hz",
            "refpulse_flipangle_deg",
            "refpulse_is_calculated",
            "refpulse_sharpness",
            "refpulse_bwfac_hzms",
            "refpulse_sint",
            "refpulse_pint",
            "refpulse_type",
            "refpulse_rpfac_percent",
            "refpulse_peakpow_w",
            "refpulse_shape",
            "refpulse_refpow_w",
        ),
        "reconstruction": (
            "recon_names",
            "acquisition_matrix",
        ),
    }

    OPTIONAL_PARAMETERS: ClassVar[Set[str]] = {
        "resolution_mm",
        "sampling_time_axis_s",
        "fat_water_separation",
        "b_values",
        "rf_pulse_params",
        "ref_pulse_params",
        "rfpulse_duration_ms",
        "rfpulse_bandwidth_hz",
        "rfpulse_is_calculated",
        "rfpulse_sharpness",
        "rfpulse_bwfac_hzms",
        "rfpulse_sint",
        "rfpulse_pint",
        "rfpulse_type",
        "rfpulse_rpfac_percent",
        "rfpulse_peakpow_w",
        "rfpulse_shape",
        "rfpulse_refpow_w",
        "flipangle_deg",
        "refpulse_duration_ms",
        "refpulse_bandwidth_hz",
        "refpulse_flipangle_deg",
        "refpulse_is_calculated",
        "refpulse_sharpness",
        "refpulse_bwfac_hzms",
        "refpulse_sint",
        "refpulse_pint",
        "refpulse_type",
        "refpulse_rpfac_percent",
        "refpulse_peakpow_w",
        "refpulse_shape",
        "refpulse_refpow_w",
        "coil_receive",
        "coil_transmit",
        "recon_names",
        "acquisition_matrix",
        "echo_spacing_ms",
        "effective_te_ms",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize common attributes with standard types
        self.params: dict = {}

        # Ensure all standardized parameter attributes exist with a default of None.
        # This makes downstream vendor handlers simpler because they can rely on the
        # base class providing placeholders for every canonical field.
        for key in self.parameter_keys():
            setattr(self, key, None)

    @classmethod
    def parameter_keys(cls) -> Set[str]:
        keys: Set[str] = set()
        for group_keys in cls.PARAMETER_SCHEMA.values():
            keys.update(group_keys)
        return keys

    @classmethod
    def required_parameter_keys(cls) -> Set[str]:
        return cls.parameter_keys() - cls.OPTIONAL_PARAMETERS

    def set_parameters(
        self, parameters: Mapping[str, Any], *, allow_extra: bool = True
    ) -> None:
        """Store standardized parameters after validating against the schema."""
        provided_keys = set(parameters.keys())
        missing = self.required_parameter_keys() - provided_keys
        if missing:
            raise ValueError(
                f"Missing required parameters: {', '.join(sorted(missing))}"
            )

        allowed_keys = self.parameter_keys()
        extra = provided_keys - allowed_keys
        if extra and not allow_extra:
            raise ValueError(f"Unexpected parameters: {', '.join(sorted(extra))}")
        elif extra:
            warnings.warn(
                (
                    "Received additional parameters not in the canonical schema: "
                    + ", ".join(sorted(extra))
                ),
                UserWarning,
                stacklevel=2,
            )

        self.params = dict(parameters)

        # Keep attribute access in sync with the standardized parameter mapping so
        # handlers can safely reference attributes without checking `params`.
        for key, value in self.params.items():
            setattr(self, key, value)

    def get_parameters(self) -> Dict[str, Any]:
        """Return the standardized parameter mapping for metadata consumers."""
        return dict(self.params)

    def __getattr__(self, name: str):
        """Fallback attribute access to standardized parameters."""
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def get_primary_data_info(self) -> dict:
        """
        Returns information about available data arrays.

        Returns
        -------
        dict
            Dictionary with data array information
        """
        info = {
            "has_data_array": self.data_array is not None,
            "has_kspace_array": False,  # kspace_array no longer stored permanently
            "has_legacy_data": (
                self._legacy_seq2d is not None or self._legacy_reco_array is not None
            ),
            "primary_data_shape": None,
            "primary_data_dims": None,
        }

        primary = self.primary_data
        if primary is not None:
            info["primary_data_shape"] = primary.shape
            info["primary_data_dims"] = primary.dims

        return info
