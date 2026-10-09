"""
Factory module for creating MRI experiments.

This module provides factory functions for creating MRI experiments
with automatic vendor detection and appropriate class selection.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Type, Union

import numpy as np

from ..vendors import detect_vendor, get_vendor_registry
from .experiment import MRIExperiment

try:
    from ..sequences import get_sequence_class
except ImportError:
    # Fallback if sequences module not fully set up yet
    def get_sequence_class(method_name, vendor=None):
        return None


def _resolve_sequence_class(method_name: str, vendor: Optional[str] = None):
    """Return the registered sequence class, re-importing if required."""

    global get_sequence_class

    # If the placeholder from the ImportError branch is still active, attempt
    # a late import of the real helper now that optional deps may be available.
    if getattr(get_sequence_class, "__module__", __name__) == __name__:
        try:
            from ..sequences import get_sequence_class as real_get_sequence_class
        except ImportError:
            # Fallback to using the registry directly if sequences module import fails
            try:
                from ..sequences.registry import get_sequence_registry

                registry = get_sequence_registry()
                return registry.get_sequence_class(method_name, vendor)
            except ImportError:
                return None
        else:
            get_sequence_class = real_get_sequence_class

    return get_sequence_class(method_name, vendor)


def create_experiment(
    path: Union[str, Path],
    load_data: bool = False,
    sequence_class: Optional[Type] = None,
    force_vendor: Optional[str] = None,
    handler_options: Optional[Dict[str, Any]] = None,
) -> MRIExperiment:
    """Construct an :class:`MRIExperiment` tailored to the dataset.

    Parameters
    ----------
    path
        Filesystem location of the experiment. Relative paths are resolved
        against the current working directory.
    load_data
        If ``True``, the vendor handler loads raw data during construction.
    sequence_class
        Optional sequence subclass that wraps the base experiment instead of
        using smart loading.
    force_vendor
        Optional vendor string that bypasses automatic vendor detection.
    handler_options
        Additional keyword arguments passed to the vendor handler factory.

    Returns
    -------
    MRIExperiment
        Experiment instance wrapping the detected or forced vendor handler.
    """
    # Create basic experiment
    experiment = MRIExperiment(
        path,
        load_data,
        force_vendor=force_vendor,
        handler_options=handler_options,
    )

    # If specific sequence class requested, wrap it
    if sequence_class:
        return sequence_class(experiment)

    # Try smart loading based on sequence type
    vendor = force_vendor or detect_vendor(path)
    if vendor:
        smart_class = _resolve_sequence_class(experiment.type, vendor)
        if smart_class:
            try:
                return smart_class(experiment)
            except Exception as e:
                warnings.warn(
                    f"Failed to create smart sequence class {smart_class.__name__}: {e}. "
                    f"Falling back to base MRIExperiment."
                )

    return experiment


def load_external_experiment(
    source: Union[str, Path],
    *,
    matrix: Optional[Sequence[int]] = None,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    fov_mm: Optional[Sequence[float]] = None,
    sampling_duration_s: Optional[float] = None,
    dwell_time_s: Optional[float] = None,
    bandwidth_hz: Optional[float] = None,
    bandwidth_ppm: Optional[float] = 1,  # default to 1 ppm --- IGNORE ---
    data: Optional[np.ndarray] = None,
    data_domain: str = "unknown",
    data_role: str = "reco_data",
    data_attrs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dtype: Union[str, np.dtype] = np.complex64,
    interleaved: bool = False,
    interleaved_dtype: Union[str, np.dtype] = np.float32,
    data_scaling: float = 1.0,
    sequence_name: str = "external",
    experiment_id: Optional[str] = None,
    load_data: bool = True,
    log_mode: str = "error",
    wrap_sequence: bool = False,
    sequence_class: Optional[Type] = None,
    n_acquisition: Optional[int] = None,
    center_ppm: Optional[float] = 0,
    cut_off: Optional[int] = 0,
) -> MRIExperiment:
    """Create an :class:`MRIExperiment` from externally supplied data arrays.

    Parameters
    ----------
    source
        Path identifying the dataset on disk. Can point to a directory that will
        host supplementary artefacts generated later.
    matrix, dims, coords
        Geometry definition forwarded to :class:`ExternalDataHandler` describing
        the target :class:`xarray.DataArray` layout.
    fov_mm, sampling_duration_s, dwell_time_s, bandwidth_hz
        Acquisition metadata used to populate coordinate attributes.
    data
        Optional ``numpy`` array containing the primary data payload. If
        ``None``, the handler expects to load the data lazily.
    data_domain, data_role, data_attrs
        Metadata that describes what the provided array represents and how it is
        positioned in subsequent pipelines.
    metadata
        Arbitrary experiment metadata stored alongside the array.
    dtype, interleaved, interleaved_dtype
        Data type configuration for the stored payload. ``interleaved`` flags
        interleaved real/imaginary encodings.
    data_scaling
        Multiplicative factor applied to the provided ``data`` before exposure.
    sequence_name, experiment_id
        Logical identifiers used to populate :class:`ExperimentMetadata`.
    load_data
        When ``True``, the external handler converts the provided ``data`` into
        the canonical ``xarray`` representation immediately.
    log_mode
        Logging verbosity hint forwarded to the experiment.
    wrap_sequence
        When ``True``, attempt to wrap the resulting experiment in a registered
        sequence matching ``sequence_name``.
    sequence_class
        Explicit sequence class to use for wrapping. Overrides ``wrap_sequence``
        lookups when provided.

    Returns
    -------
    MRIExperiment
        Experiment instance backed by an ``external`` vendor handler that
        packages the supplied data.
    """

    registry = get_vendor_registry()

    # Ensure the external vendor handler is registered even if the YAML
    # configuration was loaded before the external handler existed.
    if registry.get_vendor_info("external") is None:
        from ..externalexp import ExternalDataHandler

        registry.register_vendor(
            "external",
            ExternalDataHandler,
            detection_patterns=[],
            priority=-100,
        )

    normalised_matrix: Optional[Sequence[int]] = (
        tuple(int(dim) for dim in matrix) if matrix is not None else None
    )
    normalised_dims: Optional[Sequence[str]] = (
        tuple(str(dim) for dim in dims) if dims is not None else None
    )

    prepared_data: Optional[np.ndarray] = None
    if data is not None:
        prepared_data = np.asarray(data)

        if normalised_matrix is None:
            normalised_matrix = prepared_data.shape
        else:
            expected_size = int(np.prod(normalised_matrix))
            if prepared_data.size != expected_size:
                raise ValueError(
                    "Provided data does not match the requested matrix size: "
                    f"matrix={normalised_matrix}, data.shape={prepared_data.shape}"
                )
            if prepared_data.shape != tuple(normalised_matrix):
                prepared_data = prepared_data.reshape(normalised_matrix)

    normalised_coords: Optional[Mapping[str, Any]]
    if coords is None:
        normalised_coords = None
    elif isinstance(coords, Mapping):
        normalised_coords = coords
    else:
        if normalised_dims is None:
            raise ValueError(
                "When coords are provided as a sequence, dims must also be supplied."
            )
        coord_sequence = list(coords)
        if len(coord_sequence) != len(normalised_dims):
            raise ValueError(
                "Number of coord entries does not match dims: "
                f"dims={normalised_dims}, coords={len(coord_sequence)}"
            )
        normalised_coords = dict(zip(normalised_dims, coord_sequence))

    handler_options: Dict[str, Any] = {
        "matrix": normalised_matrix,
        "dims": normalised_dims,
        "coords": normalised_coords,
        "fov_mm": fov_mm,
        "sampling_duration_s": sampling_duration_s,
        "dwell_time_s": dwell_time_s,
        "bandwidth_hz": bandwidth_hz,
        "data": prepared_data,
        "data_domain": data_domain,
        "data_role": data_role,
        "data_attrs": data_attrs or {},
        "metadata": metadata or {},
        "dtype": dtype,
        "interleaved": interleaved,
        "interleaved_dtype": interleaved_dtype,
        "data_scaling": data_scaling,
        "sequence_name": sequence_name,
        "experiment_id": experiment_id,
        "n_acquisition": n_acquisition,
        "center_ppm": center_ppm,
        "cut_off": cut_off,
        "bandwidth_ppm": bandwidth_ppm,
    }
    if bandwidth_hz is None and dwell_time_s:
        handler_options["bandwidth_hz"] = 1.0 / dwell_time_s
    if dwell_time_s is None and bandwidth_hz:
        handler_options["dwell_time_s"] = 1.0 / bandwidth_hz
    if n_acquisition is None and sampling_duration_s and dwell_time_s:
        handler_options["n_acquisition"] = int(
            round(sampling_duration_s / dwell_time_s)
        )

    handler_options = {k: v for k, v in handler_options.items() if v is not None}

    experiment = MRIExperiment(
        source,
        load_data=load_data,
        log_mode=log_mode,
        force_vendor="external",
        handler_options=handler_options,
    )

    handler = experiment.handler
    primary_payload = None
    primary_role = "reco_data"
    if handler is not None:
        handler_primary = getattr(handler, "primary_data", None)
        if handler_primary is not None:
            handler_role = getattr(handler, "_data_role", None) or "reco_data"
            primary_payload = handler_primary
            primary_role = handler_role
            try:
                experiment.set_primary_data(handler_primary, data_type=handler_role)
            except ValueError:
                if handler_role in {"scanner_data", "seq2d"}:
                    experiment.data_array = handler_primary
                else:
                    experiment.reco_data = handler_primary

    if sequence_class:
        sequence_obj = sequence_class(experiment)
        if primary_payload is not None:
            try:
                sequence_obj.set_primary_data(primary_payload, data_type=primary_role)
            except ValueError:
                if primary_role in {"scanner_data", "seq2d"}:
                    sequence_obj.data_array = primary_payload
                else:
                    sequence_obj.reco_data = primary_payload
        return sequence_obj

    if wrap_sequence:
        smart_class = None
        experiment_type = experiment.type

        if experiment_type and experiment_type.lower() != "unknown":
            smart_class = _resolve_sequence_class(experiment_type, vendor="external")
            if smart_class is None:
                smart_class = _resolve_sequence_class(experiment_type)

        if smart_class:
            try:
                sequence_obj = smart_class(experiment)
                if primary_payload is not None:
                    try:
                        sequence_obj.set_primary_data(
                            primary_payload, data_type=primary_role
                        )
                    except ValueError:
                        if primary_role in {"scanner_data", "seq2d"}:
                            sequence_obj.data_array = primary_payload
                        else:
                            sequence_obj.reco_data = primary_payload
                return sequence_obj
            except Exception as exc:
                warnings.warn(
                    "Failed to wrap external experiment with sequence "
                    f"{smart_class.__name__}: {exc}. Falling back to MRIExperiment."
                )
        else:
            warnings.warn(
                "Requested wrap_sequence for external experiment but no matching "
                "sequence class was found. Returning base MRIExperiment."
            )

    return experiment
