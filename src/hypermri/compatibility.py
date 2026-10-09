"""
Backward compatibility utilities for HyperMRI.

This module provides utilities to help migrate from the old variable naming
conventions to the new standardized ones while maintaining backward compatibility.
"""

import warnings
from collections.abc import MutableMapping
from typing import Any, Optional

import xarray as xr


def deprecation_warning(old_name: str, new_name: str, stacklevel: int = 3):
    """
    Issue a standardized deprecation warning.

    Parameters
    ----------
    old_name : str
        The deprecated attribute/method name
    new_name : str
        The recommended replacement
    stacklevel : int
        Stack level for the warning (default 3 to point to caller's caller)
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def get_data_array_with_fallback(
    obj, prefer_legacy: bool = False
) -> Optional[xr.DataArray]:
    """
    Get data array from object with fallback to legacy attributes.

    Parameters
    ----------
    obj : object
        Object to search for data arrays
    prefer_legacy : bool
        If True, prefer legacy attributes over standardized ones

    Returns
    -------
    xr.DataArray or None
        First available data array found
    """
    if prefer_legacy:
        # Check legacy attributes first
        for attr in ["seq2d", "reco_array", "reco_data"]:
            if hasattr(obj, attr):
                data = getattr(obj, attr)
                if data is not None and (not hasattr(data, "__len__") or len(data) > 0):
                    return data

        # Then check standardized attributes
        for attr in ["data_array", "kspace_array"]:
            if hasattr(obj, attr):
                data = getattr(obj, attr)
                if data is not None and (not hasattr(data, "__len__") or len(data) > 0):
                    return data
    else:
        # Check standardized attributes first
        for attr in ["data_array", "kspace_array"]:
            if hasattr(obj, attr):
                data = getattr(obj, attr)
                if data is not None and (not hasattr(data, "__len__") or len(data) > 0):
                    return data

        # Then check legacy attributes
        for attr in ["seq2d", "reco_array", "reco_data"]:
            if hasattr(obj, attr):
                data = getattr(obj, attr)
                if data is not None and (not hasattr(data, "__len__") or len(data) > 0):
                    return data

    return None


class LegacyAttributeProxy:
    """
    Proxy class that provides access to legacy attributes with deprecation warnings.
    """

    def __init__(self, target_obj, legacy_mappings: dict):
        """
        Parameters
        ----------
        target_obj : object
            The object that contains the actual attributes
        legacy_mappings : dict
            Mapping from legacy attribute names to new attribute names
        """
        self._target = target_obj
        self._mappings = legacy_mappings

    def __getattr__(self, name: str) -> Any:
        if name in self._mappings:
            new_name = self._mappings[name]
            deprecation_warning(name, new_name)
            return getattr(self._target, new_name)

        # Fall back to target object
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            # Internal attributes
            super().__setattr__(name, value)
            return

        if name in self._mappings:
            new_name = self._mappings[name]
            deprecation_warning(name, new_name)
            setattr(self._target, new_name, value)
        else:
            setattr(self._target, name, value)


# Common legacy mappings
COMMON_DATA_ARRAY_MAPPINGS = {
    "seq2d": "data_array",
    "reco_array": "data_array",  # Note: this is a simplification
    "reco_data": "data_array",
    "kspace_xarray": "kspace_array",
}

COMMON_PATH_MAPPINGS = {
    # Most path attributes should just use 'path'
}

LEGACY_FREQUENCY_KEY_MAP = {
    "freq_axis_hz": ("freq_range_hz", "freq_range_Hz", "freq_range_HZ"),
    "freq_axis_ppm": ("freq_range_ppm", "freq_range_PPM"),
}


def _normalize_frequency_unit(unit: str) -> str:
    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"hz", "ppm"}:
        raise ValueError(
            f"Unsupported frequency unit '{unit}'. Expected one of ['Hz', 'ppm']."
        )
    return unit_norm


def _resolve_frequency_keys(unit: str) -> tuple[str, tuple[str, ...]]:
    normalized_unit = _normalize_frequency_unit(unit)
    new_key = f"freq_axis_{normalized_unit}"
    legacy_keys = LEGACY_FREQUENCY_KEY_MAP.get(new_key, ())
    return new_key, legacy_keys


def _find_existing_key_case_insensitive(
    attrs: MutableMapping, candidate_keys: tuple[str, ...]
) -> Optional[str]:
    """Return the actual key present in attrs that matches any candidate (case-insensitive)."""
    if not candidate_keys:
        return None

    lower_key_map = {str(key).lower(): key for key in attrs.keys()}

    for candidate in candidate_keys:
        if candidate in attrs:
            return candidate
        candidate_lower = str(candidate).lower()
        if candidate_lower in lower_key_map:
            return lower_key_map[candidate_lower]

    return None


def resolve_frequency_axis(
    attrs: MutableMapping | None,
    unit: str,
    *,
    create_legacy_aliases: bool = True,
) -> Any:
    """
    Retrieve a frequency axis from attrs, falling back to legacy keys if needed.

    Parameters
    ----------
    attrs : MutableMapping or None
        Attribute mapping to inspect (e.g., xarray object's attrs).
    unit : {'Hz', 'ppm'}
        Frequency unit to resolve.
    create_legacy_aliases : bool, optional
        If True, mirror the resolved axis back onto the legacy key names.

    Returns
    -------
    Any
        The resolved frequency axis (often a NumPy array or list), or None if unavailable.
    """
    if attrs is None or not isinstance(attrs, MutableMapping):
        return None

    new_key, legacy_keys = _resolve_frequency_keys(unit)
    axis = attrs.get(new_key, None)

    if axis is None:
        legacy_key = _find_existing_key_case_insensitive(attrs, legacy_keys)
        if legacy_key is not None:
            axis = attrs[legacy_key]
            if axis is not None:
                try:
                    deprecation_warning(legacy_key, new_key)
                except Exception:
                    pass
                attrs[new_key] = axis
    if axis is not None and create_legacy_aliases:
        for legacy_key in legacy_keys:
            attrs.setdefault(legacy_key, axis)

    return axis


def assign_frequency_axis(
    attrs: MutableMapping | None,
    freq_axis: Any,
    unit: str,
    *,
    mirror_legacy_keys: bool = True,
) -> None:
    """
    Assign a frequency axis to attrs, optionally mirroring legacy key names.

    Parameters
    ----------
    attrs : MutableMapping or None
        Attribute mapping to update.
    freq_axis : Any
        Frequency axis to assign.
    unit : {'Hz', 'ppm'}
        Frequency unit associated with the axis.
    mirror_legacy_keys : bool, optional
        If True, also store the axis under legacy key names.
    """
    if attrs is None or not isinstance(attrs, MutableMapping) or freq_axis is None:
        return

    new_key, legacy_keys = _resolve_frequency_keys(unit)
    attrs[new_key] = freq_axis

    if mirror_legacy_keys:
        for legacy_key in legacy_keys:
            attrs[legacy_key] = freq_axis


def migrate_object_attributes(obj, mappings: dict = None):
    """
    Migrate an object's attributes from legacy names to standardized names.

    Parameters
    ----------
    obj : object
        Object to migrate
    mappings : dict, optional
        Custom attribute mappings. If None, uses common mappings.
    """
    if mappings is None:
        mappings = COMMON_DATA_ARRAY_MAPPINGS

    for old_attr, new_attr in mappings.items():
        if hasattr(obj, old_attr) and not hasattr(obj, new_attr):
            # Copy legacy attribute to new name
            legacy_value = getattr(obj, old_attr)
            if legacy_value is not None:
                setattr(obj, new_attr, legacy_value)
                # Optionally remove the old attribute or leave it for compatibility
                # delattr(obj, old_attr)


def check_data_array_consistency(obj) -> dict:
    """
    Check consistency of data arrays in an object and return a report.

    Parameters
    ----------
    obj : object
        Object to check

    Returns
    -------
    dict
        Dictionary with consistency report
    """
    report = {
        "has_standardized_data_array": hasattr(obj, "data_array")
        and obj.data_array is not None,
        "has_kspace_array": False,  # kspace_array no longer stored permanently
        "legacy_attributes": {},
        "inconsistencies": [],
        "recommendations": [],
    }

    # Check legacy attributes
    legacy_attrs = ["seq2d", "reco_array", "reco_data"]
    for attr in legacy_attrs:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            report["legacy_attributes"][attr] = {
                "exists": True,
                "is_none": value is None,
                "is_empty": (
                    hasattr(value, "__len__") and len(value) == 0
                    if value is not None
                    else True
                ),
            }

    # Check for inconsistencies
    if report["has_standardized_data_array"]:
        for attr, info in report["legacy_attributes"].items():
            if info["exists"] and not info["is_none"] and not info["is_empty"]:
                # Both standardized and legacy data exist
                legacy_data = getattr(obj, attr)
                if hasattr(legacy_data, "shape") and hasattr(obj.data_array, "shape"):
                    if legacy_data.shape != obj.data_array.shape:
                        report["inconsistencies"].append(
                            f"Shape mismatch: {attr} {legacy_data.shape} vs data_array {obj.data_array.shape}"
                        )

    # Generate recommendations
    if not report["has_standardized_data_array"] and report["legacy_attributes"]:
        available_legacy = [
            attr
            for attr, info in report["legacy_attributes"].items()
            if info["exists"] and not info["is_none"] and not info["is_empty"]
        ]
        if available_legacy:
            report["recommendations"].append(
                f"Migrate data from legacy attribute {available_legacy[0]} to data_array"
            )

    if report["inconsistencies"]:
        report["recommendations"].append(
            "Resolve data array inconsistencies by choosing primary data source"
        )

    return report
