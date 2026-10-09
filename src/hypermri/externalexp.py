"""Generic handler for externally supplied MRI data.

This module provides a minimal vendor handler that can ingest arbitrary data
files (e.g., raw ``.dat`` dumps) together with user-specified acquisition
parameters and expose them via the standardized :class:`MRIExperiment`
interface. The resulting handler focuses on two goals:

* Read data saved as native complex numbers or as interleaved real/imag parts.
* Construct an :class:`xarray.DataArray` compatible with downstream utilities
  such as :func:`fit_data_pseudo_inv` by honouring user supplied dimensions,
  coordinates, and metadata like matrix size, FOV, sampling duration, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from hypermri.mixins import StandardizedMRIMixin
from hypermri.utils.system.logging import LOG_MODES, init_default_logger
from hypermri.vendorhandler import VendorHandler

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class ExternalDataHandler(VendorHandler, StandardizedMRIMixin):
    """Vendor handler for externally provided datasets.

    Parameters
    ----------
    source : str or path-like
        Path to the binary file containing the data. If ``data`` is supplied
        directly, the source is only used for bookkeeping.
    load_data : bool, optional
        If ``True`` (default), the data file is read immediately.
    log_mode : str, optional
        Logging verbosity. Uses the same mapping as other handlers.
    **kwargs : dict
        Additional handler options. The most relevant ones are:

        ``matrix``
            Tuple describing the target shape. Required when reading from file.
        ``dims``
            Iterable of dimension names matching ``matrix``.
        ``coords``
            Mapping of coordinates per dimension.
        ``data``
            Pre-loaded :class:`numpy.ndarray` to bypass file IO.
        ``dtype``
            Data type inside the file. Defaults to ``np.complex64``.
        ``interleaved``
            If ``True``, treat the file as ``real, imag, real, imag`` floats.
        ``interleaved_dtype``
            Data type for the interleaved representation (default ``np.float32``).
        ``fov_mm`` / ``sampling_duration_s`` / ``dwell_time_s`` / ``bandwidth_hz``
            Common acquisition parameters stored in the resulting ``DataArray``.
        ``n_acquisition`` / ``center_ppm`` / ``cut_off``
            Required parameters for frequency axis calculation. These must be provided
            to enable spectroscopic data analysis.
        ``data_attrs`` / ``metadata``
            Extra attributes added to the ``DataArray`` or handler metadata.
        ``data_domain``
            Domain descriptor (e.g., ``"frequency"``) stored in ``data.attrs``.
        ``data_role``
            Controls whether the array should be stored as ``reco_data`` or
            ``scanner_data`` on the handler (defaults to ``"reco_data"``).
        ``sequence_name`` and ``experiment_id``
            Cosmetic identifiers exposed through the :class:`MRIExperiment` API.
    """

    _DEFAULT_DIM_NAMES: Tuple[str, ...] = (
        "acquisition_idx",
        "channel_idx",
        "read_idx",
        "phase_idx",
        "partition_idx",
        "component_idx",
    )

    def __init__(
        self,
        source: Union[str, Path, None],
        load_data: bool = True,
        log_mode: str = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        print(f"init of external handler was called with {source}")

        logger.setLevel(LOG_MODES.get(log_mode, LOG_MODES["error"]))
        self.log_mode = log_mode

        self.path = Path(source).resolve() if source else None

        # Handler options -------------------------------------------------
        self._raw_array = self._pop_option(kwargs, "data")
        self._matrix = self._normalise_shape(self._pop_option(kwargs, "matrix"))
        self._dims = self._normalise_dims(self._pop_option(kwargs, "dims"))
        self._coords = self._pop_option(kwargs, "coords")
        self._data_domain = self._pop_option(kwargs, "data_domain", default="unknown")
        self._data_role = self._pop_option(kwargs, "data_role", default="reco_data")
        self._data_attrs = self._pop_option(kwargs, "data_attrs", default={})
        self._metadata = self._pop_option(kwargs, "metadata", default={})
        self._fov_mm = self._pop_option(kwargs, "fov_mm")
        self._sampling_duration_s = self._pop_option(kwargs, "sampling_duration_s")
        self._dwell_time_s = self._pop_option(kwargs, "dwell_time_s")
        self._bandwidth_hz = self._pop_option(kwargs, "bandwidth_hz")
        self._bandwidth_ppm = self._pop_option(kwargs, "bandwidth_ppm")
        self._n_acquisition = self._pop_option(kwargs, "n_acquisition")
        self._center_ppm = self._pop_option(kwargs, "center_ppm")
        self._cut_off = self._pop_option(kwargs, "cut_off")
        self._data_scaling = self._pop_option(kwargs, "data_scaling", default=1.0)
        self._sequence_name = self._pop_option(
            kwargs, "sequence_name", default="sequence_name"
        )
        self._experiment_id = str(
            self._pop_option(kwargs, "experiment_id")
            or (self.path.stem if self.path else "external")
        )
        self._vendor = self._pop_option(kwargs, "vendor", default="external")

        self._interleaved = bool(self._pop_option(kwargs, "interleaved", default=False))
        self._dtype = np.dtype(self._pop_option(kwargs, "dtype", default=np.complex64))
        self._interleaved_dtype = np.dtype(
            self._pop_option(kwargs, "interleaved_dtype", default=np.float32)
        )

        if kwargs:
            # Surface any unsupported options early to make debugging easier.
            raise TypeError(
                "Unsupported handler options provided: " + ", ".join(sorted(kwargs))
            )

        # Standardized attributes expected by downstream code ---------------
        self.ExpNum = self._experiment_id
        self.params = self._metadata.copy()

        # Optional geometry defaults
        self.fov_mm = self._fov_mm
        self.sampling_duration_s = self._sampling_duration_s
        self.dwell_time_s = self._dwell_time_s
        self.bandwidth_hz = self._bandwidth_hz
        self.n_acquisition = self._n_acquisition
        self.center_ppm = self._center_ppm
        self.cut_off = self._cut_off

        # Validate required parameters for frequency axis calculation
        self._validate_freq_axis_parameters()

        if load_data:
            self.load_data(source)

    # ------------------------------------------------------------------
    # VendorHandler interface
    # ------------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.ExpNum

    @property
    def freq_axis_ppm(self) -> Optional[np.ndarray]:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "_freq_axis_ppm") and self._freq_axis_ppm is not None:
            return self._freq_axis_ppm

        from hypermri.analysis.spectroscopy.core import get_freq_axis

        self._freq_axis_ppm = get_freq_axis(data_obj=self, unit="ppm")
        return self._freq_axis_ppm

    @property
    def freq_axis_hz(self) -> Optional[np.ndarray]:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "_freq_axis_hz") and self._freq_axis_hz is not None:
            return self._freq_axis_hz

        from hypermri.analysis.spectroscopy.core import get_freq_axis

        self._freq_axis_hz = get_freq_axis(data_obj=self, unit="Hz")
        return self._freq_axis_hz

    @property
    def dwell_times_s(self) -> float:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "dwell_time_s") and self.dwell_time_s is not None:
            return self.dwell_time_s

        from hypermri.utils.time.temporal import get_dwell_time_s

        return get_dwell_time_s(data_obj=self)

    @property
    def type(self) -> str:
        return self.sequence_name

    @staticmethod
    def is_valid_scan_dir(path: Union[str, Path]) -> bool:  # type: ignore[override]
        """External data should only be selected explicitly via ``force_vendor``."""

        return False

    def get_parameters(self, source) -> dict:  # type: ignore[override]
        return {
            "matrix": self._matrix,
            "fov_mm": self._fov_mm,
            "sampling_duration_s": self._sampling_duration_s,
            "dwell_time_s": self._dwell_time_s,
            "bandwidth_hz": self._bandwidth_hz,
            "n_acquisition": self._n_acquisition,
            "center_ppm": self._center_ppm,
            "cut_off": self._cut_off,
            "data_domain": self._data_domain,
            **self._metadata,
        }

    def load_data(self, source) -> xr.DataArray:  # type: ignore[override]
        data = self._load_raw_array(source)
        data_array = self._wrap_data_array(data)

        data_type = self._data_role or "reco_data"
        self.set_primary_data(data_array, data_type=data_type)

        return data_array

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_freq_axis_parameters(self) -> None:
        """Validate that all parameters needed for frequency axis calculation are present."""
        required_params = {
            "bandwidth_hz": self._bandwidth_hz,
            "dwell_time_s": self._dwell_time_s,
            "n_acquisition": self._n_acquisition,
            "center_ppm": self._center_ppm,
            "cut_off": self._cut_off,
        }

        missing = [name for name, value in required_params.items() if value is None]
        if missing:
            raise ValueError(
                f"Required parameters for frequency axis calculation are missing: {missing}. "
                f"Please provide these parameters when creating the ExternalDataHandler."
            )

    def _load_raw_array(self, source: Union[str, Path, None]) -> np.ndarray:
        if self._raw_array is not None:
            array = np.asarray(self._raw_array)
            if self._matrix and array.size != int(np.prod(self._matrix)):
                raise ValueError("Provided data does not match requested matrix size.")
            if self._matrix:
                array = array.reshape(self._matrix)
            return array

        if source is None:
            raise ValueError("No source provided for external data.")

        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"External data file not found: {file_path}")

        expected_size = int(np.prod(self._matrix)) if self._matrix else None

        if self._interleaved:
            raw = np.fromfile(file_path, dtype=self._interleaved_dtype)
            if raw.size % 2 != 0:
                raise ValueError(
                    "Interleaved data must contain an even number of elements."
                )
            if expected_size and raw.size != expected_size * 2:
                raise ValueError(
                    f"Expected {expected_size * 2} interleaved values, "
                    f"found {raw.size} in {file_path}."
                )
            array = raw[0::2] + 1j * raw[1::2]
        else:
            array = np.fromfile(file_path, dtype=self._dtype)
            if expected_size and array.size != expected_size:
                raise ValueError(
                    f"Expected {expected_size} values, found {array.size} in {file_path}."
                )

        if expected_size:
            array = array.reshape(self._matrix)

        if self._data_scaling and self._data_scaling != 1.0:
            array = array * self._data_scaling

        return array

    def _wrap_data_array(self, data: np.ndarray) -> xr.DataArray:
        dims = self._dims or self._infer_dims(data.ndim)
        if len(dims) != data.ndim:
            raise ValueError(
                f"Number of dimensions ({len(dims)}) does not match data.ndim ({data.ndim})."
            )

        coords = self._coords or self._default_coords(dims, data.shape)
        data_array = xr.DataArray(data, dims=dims, coords=coords)

        attrs: Dict[str, Any] = {
            "unit": "a.u.",
            "domain": self._data_domain,
            "matrix": data.shape,
            "source_path": str(self.path) if self.path else None,
        }
        if self._fov_mm is not None:
            attrs["fov_mm"] = tuple(self._fov_mm)
        if self._sampling_duration_s is not None:
            attrs["sampling_duration_s"] = float(self._sampling_duration_s)
        if self._dwell_time_s is not None:
            attrs["dwell_time_s"] = float(self._dwell_time_s)
        if self._bandwidth_hz is not None:
            attrs["bandwidth_hz"] = float(self._bandwidth_hz)

        attrs.update(self._data_attrs)
        data_array.attrs.update({k: v for k, v in attrs.items() if v is not None})

        return data_array

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pop_option(options: Dict[str, Any], key: str, default: Any = None) -> Any:
        if key in options:
            return options.pop(key)
        return default

    @staticmethod
    def _normalise_shape(matrix: Optional[Sequence[int]]) -> Tuple[int, ...]:
        if matrix is None:
            return tuple()
        return tuple(int(dim) for dim in matrix)

    def _normalise_dims(self, dims: Optional[Iterable[str]]) -> Tuple[str, ...]:
        if dims is None:
            return tuple()
        normalised = tuple(str(dim) for dim in dims)
        return normalised

    def _infer_dims(self, ndim: int) -> Tuple[str, ...]:
        if ndim <= len(self._DEFAULT_DIM_NAMES):
            return self._DEFAULT_DIM_NAMES[:ndim]
        # Fallback for higher-dimensional data
        return tuple(f"dim_{idx}" for idx in range(ndim))

    @staticmethod
    def _default_coords(dims: Sequence[str], shape: Sequence[int]) -> Dict[str, Any]:
        return {dim: np.arange(size) for dim, size in zip(dims, shape)}
