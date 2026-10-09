"""Mask and ROI utilities for anatomical analysis.

These helpers originate from the legacy ``utils_general`` module and are now part
of the structured anatomical analysis package.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping

import numpy as np

__all__ = [
    "mask_reshaper",
    "calc_overlap",
    "calc_coverage",
    "convert_mask_to_diff_fov",
    "apply_mask",
]


def mask_reshaper(
    mask: np.ndarray | None = None,
    input_axes: Iterable[int] | None = None,
    match_standard_orient: bool = False,
):
    """Reshape a mask to six dimensions expected by the spectroscopy pipeline."""

    if mask is None:
        warnings.warn("No mask defined!", RuntimeWarning, stacklevel=2)
        return False

    mask_array = np.asarray(mask)
    if input_axes is None:
        input_axes = [1, 2, 3]

    reshaped_axes: list[object] = [np.newaxis] * 6
    for axis_index, axis_position in enumerate(input_axes):
        if axis_position < 0 or axis_position >= len(reshaped_axes):
            raise ValueError("input_axes contains invalid axis index")
        reshaped_axes[axis_position] = slice(None)

    reshaped_mask = mask_array[tuple(reshaped_axes)]

    if match_standard_orient:
        reshaped_mask = np.rot90(reshaped_mask, k=-1, axes=(2, 3))

    return reshaped_mask


def calc_overlap(high_res_image: np.ndarray, low_res_image: np.ndarray) -> Mapping:
    """Return overlap percentages between two 2D images with different grids."""

    high_res_image = np.squeeze(np.asarray(high_res_image))
    low_res_image = np.squeeze(np.asarray(low_res_image))

    if high_res_image.ndim != 2 or low_res_image.ndim != 2:
        raise AssertionError("Both images must be 2D after squeezing singletons")

    high_res_shape = high_res_image.shape
    low_res_shape = low_res_image.shape

    overlap_info: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}

    high_res_voxel_height = 1.0 / high_res_shape[0]
    high_res_voxel_width = 1.0 / high_res_shape[1]
    low_res_voxel_height = 1.0 / low_res_shape[0]
    low_res_voxel_width = 1.0 / low_res_shape[1]

    for y_high in range(high_res_shape[0]):
        for x_high in range(high_res_shape[1]):
            high_res_bounds = {
                "left": x_high * high_res_voxel_width,
                "right": (x_high + 1) * high_res_voxel_width,
                "top": y_high * high_res_voxel_height,
                "bottom": (y_high + 1) * high_res_voxel_height,
            }

            overlaps: list[tuple[tuple[int, int], float]] = []

            for y_low in range(low_res_shape[0]):
                for x_low in range(low_res_shape[1]):
                    low_res_bounds = {
                        "left": x_low * low_res_voxel_width,
                        "right": (x_low + 1) * low_res_voxel_width,
                        "top": y_low * low_res_voxel_height,
                        "bottom": (y_low + 1) * low_res_voxel_height,
                    }

                    overlap_width = min(
                        high_res_bounds["right"], low_res_bounds["right"]
                    ) - max(high_res_bounds["left"], low_res_bounds["left"])
                    overlap_height = min(
                        high_res_bounds["bottom"], low_res_bounds["bottom"]
                    ) - max(high_res_bounds["top"], low_res_bounds["top"])

                    if overlap_width > 0 and overlap_height > 0:
                        overlap_area = overlap_width * overlap_height
                        high_res_voxel_area = (
                            high_res_voxel_width * high_res_voxel_height
                        )
                        overlap_percentage = np.round(
                            (overlap_area / high_res_voxel_area) * 100, 0
                        )
                        overlaps.append(((x_low, y_low), overlap_percentage))

            overlap_info[(x_high, y_high)] = overlaps

    return overlap_info


def _ensure_mask_is_6d(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        raise ValueError("mask must be provided")

    mask_array = np.asarray(mask)
    if mask_array.ndim < 1:
        raise ValueError("mask must have at least one dimension")

    if mask_array.ndim < 6:
        mask_array = np.expand_dims(mask_array, axis=0)
        while mask_array.ndim < 6:
            mask_array = np.expand_dims(mask_array, axis=-1)

    return mask_array


def calc_coverage(
    input_data: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    return_nans: bool = False,
    masks_drawn_on: str = "axial",
):
    """Compute mask coverage for a low-resolution volume against a high-res mask."""

    if input_data is None or mask is None:
        raise ValueError("Both input_data and mask must be provided")

    data_array = np.asarray(input_data)
    mask_array = _ensure_mask_is_6d(mask)

    if data_array.ndim < 6:
        raise ValueError("input_data must be at least 6D")

    if data_array.ndim == 7:
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_array = np.repeat(mask_array, data_array.shape[6], axis=-1)

    for dim_index in (0, 4, 5):
        if (
            data_array.shape[dim_index] > mask_array.shape[dim_index]
            and mask_array.shape[dim_index] == 1
        ):
            mask_array = np.repeat(
                mask_array, data_array.shape[dim_index], axis=dim_index
            )

    low_res_shape = data_array.shape[1:4]
    high_res_shape = mask_array.shape[1:4]

    output_shape = tuple(
        data_dim if mask_dim != 1 else 1
        for data_dim, mask_dim in zip(data_array.shape, mask_array.shape)
    )
    coverage = np.zeros(output_shape, dtype=float)

    depth_scale = high_res_shape[0] / (low_res_shape[0] if low_res_shape[0] != 1 else 1)
    row_scale = high_res_shape[1] / (low_res_shape[1] if low_res_shape[1] != 1 else 1)
    col_scale = high_res_shape[2] / (low_res_shape[2] if low_res_shape[2] != 1 else 1)

    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            for k in range(output_shape[3]):
                start_depth = int(i * depth_scale) if high_res_shape[0] > 1 else 0
                end_depth = (
                    min(int((i + 1) * depth_scale), mask_array.shape[1])
                    if high_res_shape[0] > 1
                    else 1
                )

                start_row = int(j * row_scale) if high_res_shape[1] > 1 else 0
                end_row = (
                    min(int((j + 1) * row_scale), mask_array.shape[2])
                    if high_res_shape[1] > 1
                    else 1
                )

                start_col = int(k * col_scale) if high_res_shape[2] > 1 else 0
                end_col = (
                    min(int((k + 1) * col_scale), mask_array.shape[3])
                    if high_res_shape[2] > 1
                    else 1
                )

                nelems = max(
                    (end_depth - start_depth)
                    * (end_row - start_row)
                    * (end_col - start_col),
                    1,
                )

                coverage[:, i, j, k, :, :] = (
                    np.nansum(
                        mask_array[
                            :,
                            start_depth:end_depth,
                            start_row:end_row,
                            start_col:end_col,
                            :,
                            :,
                        ],
                        axis=(1, 2, 3),
                    )
                    / nelems
                )

    if return_nans:
        coverage[coverage == 0] = np.nan

    return coverage


def convert_mask_to_diff_fov(
    mask_a: np.ndarray | None = None,
    fov_a: Iterable[float] | None = None,
    fov_b: Iterable[float] | None = None,
    matrix_a: Iterable[int] | None = None,
    matrix_b: Iterable[int] | None = None,
):
    """Resample ``mask_a`` to the field-of-view defined by ``fov_b``."""

    if (
        mask_a is None
        or fov_a is None
        or fov_b is None
        or matrix_a is None
        or matrix_b is None
    ):
        raise ValueError(
            "mask_a, fov_a, fov_b, matrix_a, and matrix_b must be provided"
        )

    mask_a = np.asarray(mask_a)
    res_a = [f / m for f, m in zip(fov_a, matrix_a)]
    matrix_new = [f / r for f, r in zip(fov_b, res_a)]

    mask_b = np.zeros(
        (
            mask_a.shape[0],
            mask_a.shape[1],
            int(np.round(matrix_new[1])),
            int(np.round(matrix_new[2])),
            mask_a.shape[4],
            mask_a.shape[5],
        )
    )

    diff_matrix = [b - a for b, a in zip(mask_b.shape, mask_a.shape)]

    if diff_matrix[2] > 0:
        xb_start = diff_matrix[2] // 2 + 1
        xb_end = diff_matrix[2] // 2 + mask_a.shape[2]
        xa_start = 0
        xa_end = -1
    elif diff_matrix[2] < 0:
        xa_start = abs(diff_matrix[2] // 2 - 1)
        xa_end = diff_matrix[2] // 2 + mask_a.shape[2]
        xb_start = 0
        xb_end = -1
    else:
        xa_start = xb_start = 0
        xa_end = xb_end = -1

    if diff_matrix[3] > 0:
        yb_start = diff_matrix[3] // 2 + 1
        yb_end = diff_matrix[3] // 2 + mask_a.shape[3]
        ya_start = 0
        ya_end = -1
    elif diff_matrix[3] < 0:
        ya_start = abs(diff_matrix[3] // 2 - 1)
        ya_end = diff_matrix[3] // 2 + mask_a.shape[3]
        yb_start = 0
        yb_end = -1
    else:
        ya_start = yb_start = 0
        ya_end = yb_end = mask_a.shape[3]

    mask_b[:, :, xb_start:xb_end, yb_start:yb_end, :, :] = mask_a[
        :, :, xa_start:xa_end, ya_start:ya_end, :, :
    ]

    return mask_b


def apply_mask(
    input_data: np.ndarray | None = None,
    mask_dict: Mapping[str, np.ndarray] | None = None,
    mask_key: str | None = None,
    mask: np.ndarray | None = None,
    return_nans: bool = False,
    avg_all_space_dims: bool = False,
    masks_drawn_on: str = "axial",
    mask_slice_ind: int | None = None,
    bitwise: bool = False,
    **kwargs,
):
    """Apply a mask to spectroscopy data and optionally average spatial dims."""

    if input_data is None:
        return False

    selected_mask = mask
    if selected_mask is None and mask_dict is not None and mask_key is not None:
        selected_mask = mask_dict.get(mask_key)
        if selected_mask is None:
            warnings.warn(
                "Cannot extract mask from mask_dict", RuntimeWarning, stacklevel=2
            )
            return False

    if selected_mask is None:
        warnings.warn("No mask provided", RuntimeWarning, stacklevel=2)
        return False

    mask_copy = _ensure_mask_is_6d(selected_mask)

    if mask_slice_ind is None:
        warnings.warn("No mask_slice_ind was given!", RuntimeWarning, stacklevel=2)
    else:
        if masks_drawn_on == "axial":
            mask_copy = mask_copy[:, [mask_slice_ind], :, :, :, :]
        elif masks_drawn_on == "coronal":
            mask_copy = mask_copy[:, :, :, [mask_slice_ind], :, :]
        else:
            mask_copy = mask_copy[:, :, [mask_slice_ind], :, :, :]

    data_array = np.asarray(input_data)
    if data_array.ndim < 6:
        from ...utils.image.dimensions import get_indices_for_strings
        from ..spectroscopy.core import make_NDspec_6Dspec

        if "provided_dims" in kwargs:
            provided_dims = get_indices_for_strings(kwargs["provided_dims"])
            data_array = make_NDspec_6Dspec(
                input_data=data_array, provided_dims=provided_dims
            )
        else:
            data_array = make_NDspec_6Dspec(input_data=data_array)

    coverage = calc_coverage(
        input_data=data_array, mask=mask_copy, return_nans=return_nans
    )

    if bitwise:
        threshold = kwargs.get("bitwise_lower_threshold", 0.5)
        coverage[coverage >= threshold] = 1
        if return_nans:
            coverage[coverage < threshold] = np.nan
        else:
            coverage[coverage < threshold] = 0.0

    masked_data = data_array * coverage

    if avg_all_space_dims:
        masked_data = np.squeeze(np.nanmean(masked_data, axis=(1, 2, 3), keepdims=True))

    return masked_data
