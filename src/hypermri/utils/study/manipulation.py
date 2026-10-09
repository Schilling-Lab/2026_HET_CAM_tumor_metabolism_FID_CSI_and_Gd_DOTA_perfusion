"""Utility functions for working with Study objects."""

import warnings
from bisect import bisect_left
from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr


def _extract_sort_value(scan, sort_by):
    """Return the value used for sorting for a single scan."""
    if sort_by is None:
        return None

    if callable(sort_by):
        return sort_by(scan)

    if isinstance(sort_by, str):
        if hasattr(scan, sort_by):
            return getattr(scan, sort_by)
        if hasattr(scan, "reco_data") and sort_by in scan.reco_data.coords:
            coord_val = scan.reco_data.coords[sort_by]
            if hasattr(coord_val, "values"):
                values = coord_val.values
                if hasattr(values, "__len__") and len(values) > 0:
                    return float(values.flat[0])
                return float(values)
            return coord_val
        if hasattr(scan, "reco_data") and sort_by in scan.reco_data.attrs:
            return scan.reco_data.attrs[sort_by]
        if sort_by == "rfpower" and hasattr(scan, "rfpulse_peakpow_W"):
            return scan.rfpulse_peakpow_W

        raise ValueError(
            f"sort_by='{sort_by}' not found as attribute, coordinate, or attribute "
            f"in scan object. Available scan attributes: "
            f"{[a for a in dir(scan) if not a.startswith('_')]}"
        )

    raise TypeError("sort_by must be None, a string, or a callable.")


def _prepare_stack_inputs(study, scan_indices, sort_by):
    """Collect scan objects and metadata prior to stacking."""
    if scan_indices is None:
        scan_indices = list(study.dataframe.index)
    else:
        scan_indices = list(scan_indices)

    scan_objects = []
    for idx in scan_indices:
        scan_objects.append(study[idx])

    if not scan_objects:
        raise ValueError("No scans selected for stacking.")

    scan_types = [type(scan).__name__ for scan in scan_objects]
    unique_scan_types = set(scan_types)

    if len(unique_scan_types) > 1:
        warnings.warn(
            f"Mixed scan types detected: {unique_scan_types}. "
            f"Returning stacked data as scan object of type '{scan_types[0]}'. "
            f"Some attributes may not be meaningful across different scan types.",
            UserWarning,
        )

    scan_type_class = type(scan_objects[0])
    scan_type_name = scan_types[0]

    if sort_by is not None:
        sort_values = [_extract_sort_value(scan, sort_by) for scan in scan_objects]
    else:
        sort_values = list(scan_indices)

    if sort_by is not None:
        order = np.argsort(sort_values)
        scan_objects = [scan_objects[i] for i in order]
        sort_values = [sort_values[i] for i in order]
        scan_indices_sorted = [scan_indices[i] for i in order]
    else:
        scan_indices_sorted = scan_indices

    return (
        scan_objects,
        sort_values,
        scan_indices_sorted,
        scan_type_class,
        scan_type_name,
    )


def _normalize_insertions(insertions):
    """Normalize insertion specifications to a list of dictionaries."""
    if insertions is None:
        return []

    if isinstance(insertions, dict):
        insertions_list = [insertions]
    elif isinstance(insertions, Sequence) and not isinstance(insertions, (str, bytes)):
        insertions_list = list(insertions)
    else:
        raise TypeError("insertions must be a dict, a sequence of dicts, or None.")

    normalized = []
    for idx, item in enumerate(insertions_list):
        if not isinstance(item, dict):
            raise TypeError(
                f"Insertion specification at position {idx} must be a dict."
            )
        if "scan" not in item:
            raise ValueError(
                f"Insertion specification at position {idx} is missing the required 'scan' key."
            )
        normalized.append(dict(item))

    return normalized


def _resolve_scan_reference(study, reference):
    """Return a scan object given either a reference or an already loaded scan."""
    if hasattr(reference, "reco_data"):
        return reference
    return study[reference]


def _normalize_insert_index(index, current_length):
    """Clamp list.insert-style indices to valid positions."""
    if index < 0:
        return max(0, current_length + index)
    return min(index, current_length)


def _stack_scan_objects(
    scan_objects,
    sort_values,
    scan_indices_sorted,
    scan_type_class,
    scan_type_name,
    stack_dim_name,
    sort_by,
    match_orientation_to,
    average_repetitions,
    return_scan_object,
):
    """Stack the provided scan objects into a single DataArray or scan object."""
    if not scan_objects:
        raise ValueError("No scan objects were provided for stacking.")

    first_scan = scan_objects[0]
    first_reco = first_scan.reco_data

    rotate_k = 0
    has_spatial_dims = "x" in first_reco.dims and "y" in first_reco.dims

    if match_orientation_to is not None and has_spatial_dims:
        if hasattr(match_orientation_to, "reco_data"):
            ref_data = match_orientation_to.reco_data
        else:
            ref_data = match_orientation_to

        ref_patient_pos = ref_data.attrs.get("patient_pos", None)
        ref_slice_orient = ref_data.attrs.get("slice_orientation", None)
        ref_read_orient = ref_data.attrs.get("read_orientation", None)

        curr_patient_pos = first_scan.patient_position
        curr_slice_orient = first_scan.slice_orientation
        curr_read_orient = first_scan.read_orientation

        if (
            ref_patient_pos != curr_patient_pos
            or ref_slice_orient != curr_slice_orient
            or ref_read_orient != curr_read_orient
        ):

            if ref_patient_pos == "Head_Supine" and curr_patient_pos == "Head_Prone":
                rotate_k = 2
            elif ref_patient_pos == "Head_Prone" and curr_patient_pos == "Head_Supine":
                rotate_k = 2

            print(f"Orientation mismatch detected:")
            print(
                f"  Reference: {ref_patient_pos}, {ref_slice_orient}, {ref_read_orient}"
            )
            print(
                f"  Current:   {curr_patient_pos}, {curr_slice_orient}, {curr_read_orient}"
            )
            print(f"  Applying {rotate_k}x90° rotation")

    data_list = []
    for scan in scan_objects:
        data = scan.reco_data

        if rotate_k != 0 and has_spatial_dims:
            data = data.copy()
            x_axis = list(data.dims).index("x")
            y_axis = list(data.dims).index("y")
            rotated_values = np.rot90(data.values, k=rotate_k, axes=(x_axis, y_axis))
            data.values = rotated_values

            for coord_name in data.coords:
                if coord_name in ["x", "y", "read_idx", "phase_idx"]:
                    coord_data = data.coords[coord_name]
                    if "x" in coord_data.dims and "y" in coord_data.dims:
                        x_coord_axis = list(coord_data.dims).index("x")
                        y_coord_axis = list(coord_data.dims).index("y")
                        rotated_coord = np.rot90(
                            coord_data.values,
                            k=rotate_k,
                            axes=(x_coord_axis, y_coord_axis),
                        )
                        data.coords[coord_name] = (coord_data.dims, rotated_coord)

        data_list.append(data)

    common_attrs = {}
    if data_list:
        first_attrs = dict(data_list[0].attrs)

        for key, value in first_attrs.items():
            is_common = True
            for data in data_list[1:]:
                if key not in data.attrs:
                    is_common = False
                    break
                other_value = data.attrs[key]
                try:
                    if isinstance(value, np.ndarray) or isinstance(
                        other_value, np.ndarray
                    ):
                        if not np.array_equal(value, other_value):
                            is_common = False
                            break
                    else:
                        if value != other_value:
                            is_common = False
                            break
                except (ValueError, TypeError):
                    is_common = False
                    break

            if is_common:
                common_attrs[key] = value

    common_scan_attrs = {}
    if scan_objects and return_scan_object:
        first_scan = scan_objects[0]
        candidate_attrs = [
            "matrix",
            "fov_mm",
            "slice_thickness_mm",
            "n_slices",
            "n_receivers",
            "patient_position",
            "slice_orientation",
            "read_orientation",
            "rfpulse_peakpow_W",
            "acquisition_time_s",
            "te_ms",
            "tr_ms",
            "flip_angle_deg",
            "bandwidth_hz",
            "echo_time_ms",
            "repetition_time_ms",
            "n_repetitions",
            "n_averages",
            "read_offset_mm",
            "phase_offset_mm",
            "slice_offset_mm",
            "read_direction",
            "phase_direction",
            "slice_direction",
        ]

        for attr_name in candidate_attrs:
            if hasattr(first_scan, attr_name):
                try:
                    first_value = getattr(first_scan, attr_name)
                    is_common = True

                    for scan in scan_objects[1:]:
                        if not hasattr(scan, attr_name):
                            is_common = False
                            break
                        other_value = getattr(scan, attr_name)

                        try:
                            if isinstance(first_value, np.ndarray) or isinstance(
                                other_value, np.ndarray
                            ):
                                if not np.array_equal(first_value, other_value):
                                    is_common = False
                                    break
                            else:
                                if first_value != other_value:
                                    is_common = False
                                    break
                        except (ValueError, TypeError):
                            is_common = False
                            break

                    if is_common:
                        common_scan_attrs[attr_name] = first_value
                except Exception:
                    pass

    stacked_data = xr.concat(data_list, dim=stack_dim_name, combine_attrs="override")
    stacked_data.attrs.update(common_attrs)
    stacked_data = stacked_data.assign_coords({stack_dim_name: sort_values})

    stacked_data.attrs["scan_indices"] = scan_indices_sorted
    stacked_data.attrs["sorted_by"] = str(sort_by) if sort_by is not None else "None"
    stacked_data.attrs["rotation_applied"] = rotate_k
    stacked_data.attrs["stack_dim_name"] = stack_dim_name

    if hasattr(first_scan, "n_slices"):
        stacked_data.attrs["n_slices"] = first_scan.n_slices
    if hasattr(first_scan, "n_receivers"):
        stacked_data.attrs["n_channels"] = first_scan.n_receivers
    if hasattr(first_scan, "matrix") and len(first_scan.matrix) >= 2:
        stacked_data.attrs["matrix"] = list(first_scan.matrix[:2])

    if average_repetitions and "repetition_idx" in stacked_data.dims:
        stacked_data = stacked_data.mean(dim="repetition_idx")
        stacked_data.attrs["repetitions_averaged"] = True

    if return_scan_object:
        try:

            class StackedScanObject(scan_type_class):
                """
                Stacked scan object that inherits from the original scan type.

                This class is a true subclass of the original scan type (e.g., FLASH, NSPECT)
                and provides all the same methods and properties, but with stacked data.
                """

                def __init__(self, reco_data, common_scan_attrs, first_scan):
                    if hasattr(first_scan, "handler"):
                        self.handler = first_scan.handler
                    if hasattr(first_scan, "path"):
                        self._path = first_scan.path

                    self._reco_data = reco_data
                    self._reco_data_loaded = True

                    self._data_array = None
                    self._data_array_loaded = False

                    self._common_scan_attrs = common_scan_attrs

                    for attr_name, attr_value in common_scan_attrs.items():
                        try:
                            setattr(self, attr_name, attr_value)
                        except AttributeError:
                            pass

                    try:
                        from hypermri.visualization.plotter_factory import (
                            create_plotter,
                        )

                        is_spectroscopy = False
                        for base_class in scan_type_class.__mro__:
                            class_name = base_class.__name__
                            if "Spectroscopy" in class_name or class_name == "NSPECT":
                                is_spectroscopy = True
                                break

                        if is_spectroscopy:
                            self.plotter = create_plotter(
                                sequence_obj=self,
                                category="spectroscopy",
                                plotter_type="spectroscopy",
                            )
                        else:
                            self.plotter = create_plotter(
                                sequence_obj=self,
                                category="anatomical",
                                plotter_type="anatomy",
                            )
                    except Exception:
                        try:
                            if is_spectroscopy:
                                from hypermri.utils.plot.spectroscopyPlotter import (
                                    SpectroscopyPlotter,
                                )

                                self.plotter = SpectroscopyPlotter(self)
                            else:
                                from hypermri.utils.plot.anatomyPlotter import (
                                    AnatomyPlotter,
                                )

                                self.plotter = AnatomyPlotter(self)
                        except Exception:
                            self.plotter = None

                    self._is_stacked = True
                    self._stack_dim_name = stack_dim_name

                @property
                def reco_data(self):
                    """Return the stacked reconstructed data."""
                    return self._reco_data

                @reco_data.setter
                def reco_data(self, value):
                    """Set the reconstructed data."""
                    self._reco_data = value

                def __getattr__(self, name):
                    """Delegate attribute access to common_scan_attrs or handler."""
                    if (
                        "_common_scan_attrs" in self.__dict__
                        and name in self._common_scan_attrs
                    ):
                        return self._common_scan_attrs[name]

                    if (
                        hasattr(self, "handler")
                        and self.handler
                        and hasattr(self.handler, name)
                    ):
                        return getattr(self.handler, name)

                    if hasattr(self, "handler") and self.handler:
                        handler_params = getattr(self.handler, "params", None)
                        if handler_params and name in handler_params:
                            return handler_params[name]

                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )

                def __repr__(self):
                    return (
                        f"Stacked{scan_type_name}(shape={self._reco_data.shape}, "
                        f"dims={self._reco_data.dims}, n_stacked={len(scan_objects)})"
                    )

            stacked_scan = StackedScanObject(
                stacked_data, common_scan_attrs, first_scan
            )
            return stacked_scan

        except Exception as e:
            warnings.warn(
                f"Failed to create scan object: {e}. Returning xarray DataArray instead.",
                UserWarning,
            )
            return stacked_data
    else:
        return stacked_data


def stack_scans_from_study(
    study: object,
    scan_indices: list | None = None,
    stack_dim_name: str = "stack_idx",
    sort_by: Callable | str | None = None,
    match_orientation_to=None,
    average_repetitions: bool = False,
    return_scan_object: bool = True,
):
    """
    Stack multiple scans from a Study object into a single xarray DataArray.

    This is a general-purpose function that works for any kind of data (B1, spectroscopy, etc.).
    It can match the orientation of the stacked data to a reference scan.

    Parameters
    ----------
    study : hypermri.Study
        Study object containing the scans to stack
    scan_indices : list of int or str, or None
        List of scan indices (int) or scan names (str) to include in the stacked dataset.
        If None, all scans from the study will be stacked.
    stack_dim_name : str, optional
        Name for the new stacking dimension. Default is 'stack_idx'.
        Examples: 'rfpower_idx', 'slice_idx', 'repetition_idx', etc.
    sort_by : str or callable, optional
        How to sort scans before stacking. Options:

        - str: Any string that matches either a scan attribute or a coordinate/index in reco_data
          (e.g., 'rfpower', 'n_repetitions', 'acquisition_time_s', etc.)
        - callable: Custom function that takes a scan object and returns a sortable value
        - None: Keep original order (default)

    match_orientation_to : data object or xr.DataArray, optional
        Reference data object to match orientation. If provided, the function will
        automatically determine the rotation needed to match the reference orientation.
        Can be a scan object or an xarray DataArray. Default is None.
    average_repetitions : bool, optional
        Whether to average over repetitions dimension. Default is False.
    return_scan_object : bool, optional
        If True (default), returns a scan object (e.g., FLASH, RARE) with the stacked data.
        If False, returns only the xarray DataArray. Default is True.

    Returns
    -------
    scan object or xr.DataArray
        If return_scan_object=True: A scan object (e.g., FLASH, RARE) containing the stacked data
        with all common attributes preserved. The type matches the input scans.
        If return_scan_object=False: xr.DataArray with stacked data.
        The stacked data has a new first dimension named according to stack_dim_name.
        Coordinates and attributes are preserved from the first scan.

    Examples
    --------
    # B1 mapping example
    >>> from hypermri.study_utils import stack_scans_from_study
    >>> b1_data = stack_scans_from_study(
    ...     study=scans,
    ...     scan_indices=list(range(42, 56)),
    ...     stack_dim_name='rfpower_idx',
    ...     sort_by='rfpower',
    ...     match_orientation_to=reference_scan
    ... )

    # Stack all scans from a study
    >>> all_stacked = stack_scans_from_study(
    ...     study=scans,
    ...     scan_indices=None,
    ...     stack_dim_name='scan_idx'
    ... )

    # Spectroscopy example with custom sorting
    >>> spec_data = stack_scans_from_study(
    ...     study=scans,
    ...     scan_indices=['scan1', 'scan2', 'scan3'],
    ...     stack_dim_name='block_idx',
    ...     match_orientation_to=reference_data.reco_data
    ... )

    """
    (
        scan_objects,
        sort_values,
        scan_indices_sorted,
        scan_type_class,
        scan_type_name,
    ) = _prepare_stack_inputs(
        study=study,
        scan_indices=scan_indices,
        sort_by=sort_by,
    )

    return _stack_scan_objects(
        scan_objects=scan_objects,
        sort_values=sort_values,
        scan_indices_sorted=scan_indices_sorted,
        scan_type_class=scan_type_class,
        scan_type_name=scan_type_name,
        stack_dim_name=stack_dim_name,
        sort_by=sort_by,
        match_orientation_to=match_orientation_to,
        average_repetitions=average_repetitions,
        return_scan_object=return_scan_object,
    )


def stack_scans_from_study_with_insertions(
    study,
    scan_indices,
    insertions,
    stack_dim_name="stack_idx",
    sort_by=None,
    match_orientation_to=None,
    average_repetitions=False,
    return_scan_object=True,
):
    """
    Enhanced stacking utility that allows inserting additional acquisitions at custom positions.

    Parameters
    ----------
    study : hypermri.Study
        Study object containing the scans to stack.
    scan_indices : list of int or str, or None
        Base scans to include before applying insertions. Behaves like ``stack_scans_from_study``.
    insertions : dict or sequence of dict, required
        Additional acquisitions to inject. Each dict must contain:

        - ``scan``: Study index/name or an already loaded scan object.

        Optional keys:

        - ``position``: Zero-based insertion index (negative values count from the end).
          When omitted, the function will try to auto-detect the correct slot using ``value``.
          If neither is provided, the acquisition is appended.
        - ``value``: Coordinate value for the stacking dimension. Also used to auto-determine
          the insertion index when ``position`` is missing. When ``sort_by`` is set, the value
          is optional because it can be derived from the scan attributes automatically.
        - ``scan_id``: Label stored in ``attrs['scan_indices']`` and, when ``sort_by`` is None,
          used as the stacking coordinate value. Defaults to the provided ``scan`` reference or
          ``'inserted_{n}'`` for standalone scan objects.

    Other parameters mirror ``stack_scans_from_study``.

    Returns
    -------
    scan object or xr.DataArray
        Same as ``stack_scans_from_study``.
    """
    normalized_insertions = _normalize_insertions(insertions)
    if not normalized_insertions:
        return stack_scans_from_study(
            study=study,
            scan_indices=scan_indices,
            stack_dim_name=stack_dim_name,
            sort_by=sort_by,
            match_orientation_to=match_orientation_to,
            average_repetitions=average_repetitions,
            return_scan_object=return_scan_object,
        )

    (
        scan_objects,
        sort_values,
        scan_indices_sorted,
        scan_type_class,
        scan_type_name,
    ) = _prepare_stack_inputs(
        study=study,
        scan_indices=scan_indices,
        sort_by=sort_by,
    )

    scan_objects = list(scan_objects)
    sort_values = list(sort_values)
    scan_indices_sorted = list(scan_indices_sorted)

    mixed_type_warned = False

    for idx, spec in enumerate(normalized_insertions):
        scan_obj = _resolve_scan_reference(study, spec["scan"])

        if not isinstance(scan_obj, scan_type_class) and not mixed_type_warned:
            warnings.warn(
                "Inserted scans do not share the same type as the base stack. "
                f"Result will still inherit from '{scan_type_name}', but some attributes "
                "may not be meaningful across types.",
                UserWarning,
            )
            mixed_type_warned = True

        scan_id = spec.get("scan_id")
        if scan_id is None:
            reference = spec["scan"]
            if isinstance(reference, (int, str)):
                scan_id = reference
            else:
                scan_id = f"inserted_{idx}"

        explicit_value = spec.get("value")
        derived_value = None
        if explicit_value is None and sort_by is not None:
            derived_value = _extract_sort_value(scan_obj, sort_by)
            print(f"Derived insertion value for scan_id '{scan_id}': {derived_value}")

        auto_value = explicit_value if explicit_value is not None else derived_value

        if explicit_value is not None:
            stack_coord_value = explicit_value
        elif derived_value is not None:
            stack_coord_value = derived_value
        else:
            stack_coord_value = scan_id
        position = spec.get("position")
        if position is not None:
            if not isinstance(position, int):
                raise TypeError(
                    f"position must be an integer for insertion with scan_id '{scan_id}'."
                )
            position = _normalize_insert_index(position, len(scan_objects))
        elif auto_value is not None:
            try:
                position = bisect_left(sort_values, auto_value)
            except TypeError as exc:
                raise TypeError(
                    "Automatic insertion failed because provided 'value' is not comparable "
                    "to existing stack coordinates. Specify 'position' explicitly."
                ) from exc
        else:
            position = len(scan_objects)

        scan_objects.insert(position, scan_obj)
        sort_values.insert(position, stack_coord_value)
        scan_indices_sorted.insert(position, scan_id)

    return _stack_scan_objects(
        scan_objects=scan_objects,
        sort_values=sort_values,
        scan_indices_sorted=scan_indices_sorted,
        scan_type_class=scan_type_class,
        scan_type_name=scan_type_name,
        stack_dim_name=stack_dim_name,
        sort_by=sort_by,
        match_orientation_to=match_orientation_to,
        average_repetitions=average_repetitions,
        return_scan_object=return_scan_object,
    )
