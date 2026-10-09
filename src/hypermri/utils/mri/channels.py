"""Channel combination utilities for multi-channel MRI data.

This module provides general-purpose channel combination methods that work
with xarray DataArrays and properly handle dimension management.
"""

from typing import Literal, Optional, Union

import numpy as np
import xarray as xr


def combine_channels(
    data_array: xr.DataArray,
    method: Literal[
        "rss", "sum", "mean", "phase_aligned", "weighted_mean", "adaptive"
    ] = "rss",
    adjust_phase: bool = True,
    raw_fid_data: Optional[np.ndarray] = None,
    phase_diff_deg: Optional[float] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    reference_channel: int = 0,
    auto_reference: bool = True,
    signal_domain: Optional[Literal["spatial", "k-space"]] = None,
) -> xr.DataArray:
    """Combine multiple receiver channels into a single channel.

    This is the main API for channel combination. It automatically handles
    dimension management, ensuring the output has a channel_idx dimension
    with size 1.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data with channel_idx dimension.
    method : {"rss", "sum", "mean", "phase_aligned", "weighted_mean", "adaptive"}
        Combination method:
            - "rss": Root-sum-of-squares (default, recommended for general use)
            - "sum": Simple sum across channels
            - "mean": Average across channels
            - "phase_aligned": Phase-aligned combination for 2-channel spectroscopy
            - "weighted_mean": Weighted average across channels. Requires 'weights'.
            - "adaptive": Phase-aligned combination based on a reference channel using least-squares alignment.
    adjust_phase : bool, optional
        For phase_aligned and adaptive methods: whether to apply phase correction.
        Default is True.
    raw_fid_data : np.ndarray, optional
        For phase_aligned method only: raw FID data for phase calculation.
        Required when method="phase_aligned".
    phase_diff_deg: float, optional
        For phase_aligned method only: the phase difference in degree between the two channels.
    weights : np.ndarray | xr.DataArray, optional
        For weighted_mean method only: weights for each point.
        Must be broadcastable to data_array.
    reference_channel : int, optional
        For adaptive method: index of the channel to use as phase reference.
        Default is 0. Used only if auto_reference is False.
    auto_reference : bool, optional
        For adaptive method: if True (default), automatically selects the channel with
        the highest mean signal magnitude as the reference channel, overriding
        reference_channel.
    signal_domain : {"spatial", "k-space"}, optional
        The domain in which to perform the combination. If None, uses the
        'signal_domain' attribute of the data_array. If the requested domain
        differs from the current domain, FFT/IFFT is performed.

    Returns
    -------
    xr.DataArray
        Data with channels combined. The channel_idx dimension has size 1.

    Raises
    ------
    ValueError
        If channel_idx dimension is missing, or if invalid parameters are provided.
    """
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")

    n_channels = data_array.sizes["channel_idx"]
    if n_channels == 1:
        # Single channel - already has channel_idx with size 1
        return data_array

    # Dispatch to appropriate method
    if method == "rss":
        return _combine_channels_rss(data_array)
    elif method == "sum":
        return _combine_channels_sum(data_array)
    elif method == "mean":
        return _combine_channels_mean(data_array)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError(
                "weights are required for weighted_mean channel combination"
            )
        return _combine_channels_weighted_mean(data_array, weights)
    elif method == "phase_aligned":
        if raw_fid_data is None:
            raise ValueError(
                "raw_fid_data is required for phase_aligned channel combination"
            )
        if n_channels != 2:
            raise ValueError(
                f"phase_aligned method requires exactly 2 channels, got {n_channels}"
            )
        return _combine_channels_phase_aligned(
            data_array,
            raw_fid_data=raw_fid_data,
            adjust_phase=adjust_phase,
            phase_diff_deg=phase_diff_deg,
        )
    elif method == "adaptive":
        return _combine_channels_adaptive(
            data_array,
            reference_channel=reference_channel,
            auto_reference=auto_reference,
            signal_domain=signal_domain,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: rss, sum, mean, phase_aligned, weighted_mean, adaptive"
        )


def _combine_channels_rss(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels using root-sum-of-squares (RSS)."""
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")

    if data_array.sizes["channel_idx"] == 1:
        return data_array

    # Get channel axis index
    channel_axis = data_array.dims.index("channel_idx")

    # Calculate RSS: sqrt(sum(|channel|^2))
    if np.iscomplexobj(data_array.values):
        rss_values = np.sqrt(
            np.sum(np.abs(data_array.values) ** 2, axis=channel_axis, keepdims=True)
        )
    else:
        rss_values = np.sqrt(
            np.sum(data_array.values**2, axis=channel_axis, keepdims=True)
        )

    return _create_result_array(data_array, rss_values, "rss")


def _combine_channels_sum(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels by summing."""
    result_values = data_array.sum(dim="channel_idx", keepdims=True).values
    return _create_result_array(data_array, result_values, "sum")


def _combine_channels_mean(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels by averaging."""
    result_values = data_array.mean(dim="channel_idx", keepdims=True).values
    return _create_result_array(data_array, result_values, "mean")


def _combine_channels_weighted_mean(
    data_array: xr.DataArray, weights: Union[np.ndarray, xr.DataArray]
) -> xr.DataArray:
    """Combine channels by weighted averaging."""
    numerator = (data_array * weights).sum(dim="channel_idx", keepdims=True)
    ones = xr.ones_like(data_array)
    broadcasted_weights = ones * weights
    denominator = broadcasted_weights.sum(dim="channel_idx", keepdims=True)
    result_values = (numerator / denominator).values
    return _create_result_array(data_array, result_values, "weighted_mean")


def _combine_channels_phase_aligned(
    data_array: xr.DataArray,
    raw_fid_data: np.ndarray,
    adjust_phase: bool = True,
    combination_method: Literal["mean", "sum"] = "mean",
    phase_diff_deg: Optional[float] = None,
) -> xr.DataArray:
    """Combine 2 channels with phase alignment for spectroscopy data."""
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")
    if "acquisition_idx" not in data_array.dims:
        raise ValueError(
            "Input data must have 'acquisition_idx' dimension for phase-aligned combination"
        )

    n_channels = data_array.sizes["channel_idx"]
    if n_channels != 2:
        raise ValueError(
            f"Phase-aligned combination requires exactly 2 channels, got {n_channels}"
        )

    acquisition_idx_length = data_array.sizes["acquisition_idx"]

    if phase_diff_deg is None:
        phase_diff_rad = np.angle(raw_fid_data[acquisition_idx_length]) - np.angle(
            raw_fid_data[0]
        )
        phase_diff_deg = phase_diff_rad * 180 / np.pi

    if not adjust_phase:
        phase_diff_deg = 0.0

    ch1_data = data_array.isel(channel_idx=0)
    ch2_data = data_array.isel(channel_idx=1)
    ch2_corrected = ch2_data * np.exp(-1j * np.deg2rad(phase_diff_deg))

    if combination_method == "mean":
        combined_values = ((ch1_data + ch2_corrected) / 2).values
    else:
        combined_values = (ch1_data + ch2_corrected).values

    # Add back the channel dimension for consistency in _create_result_array
    combined_values = np.expand_dims(
        combined_values, axis=data_array.dims.index("channel_idx")
    )

    result = _create_result_array(data_array, combined_values, "phase_aligned")
    result.attrs["phase_correction_deg"] = float(phase_diff_deg)
    return result


def _combine_channels_adaptive(
    data_array: xr.DataArray,
    reference_channel: int = 0,
    auto_reference: bool = True,
    signal_domain: Optional[Literal["spatial", "k-space"]] = None,
) -> xr.DataArray:
    """Adaptive channel combination using least-squares phase alignment.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data with channel_idx dimension.
    reference_channel : int
        Index of the channel to use as phase reference.
    auto_reference : bool
        If True, automatically select the reference channel with the highest mean magnitude.
    signal_domain : {"spatial", "k-space"}, optional
        The domain in which to perform the combination.
    """
    original_domain = data_array.attrs.get("signal_domain", "spatial")
    target_domain = signal_domain or original_domain

    working_data = data_array.copy()

    # Domain transformation if necessary
    if target_domain == "k-space" and original_domain == "spatial":
        working_data = _transform_spatial_to_kspace(working_data)
    elif target_domain == "spatial" and original_domain == "k-space":
        working_data = _transform_kspace_to_spatial(working_data)

    n_channels = working_data.sizes["channel_idx"]

    # Auto-select reference channel if requested
    if auto_reference:
        # Calculate mean magnitude for each channel across all other dimensions
        # Use abs() first, then mean().
        # xarray's mean() over all dims except channel_idx.
        dims_to_reduce = [d for d in working_data.dims if d != "channel_idx"]
        mean_magnitudes = np.abs(working_data).mean(dim=dims_to_reduce)
        reference_channel = int(mean_magnitudes.argmax(dim="channel_idx"))

    reference_scan = working_data.isel(channel_idx=reference_channel)

    # We'll work on the numpy values for efficiency
    vals = working_data.values
    ref_vals = reference_scan.values

    channel_axis = working_data.get_axis_num("channel_idx")

    # Reshape to (other_dims, n_channels)
    # Move channel axis to the end
    vals_transposed = np.moveaxis(vals, channel_axis, -1)
    original_shape = vals_transposed.shape

    # Flatten all other dimensions
    vals_flat = vals_transposed.reshape(-1, n_channels)
    ref_vals_flat = ref_vals.reshape(-1, 1)

    phased_vals_flat = np.zeros_like(vals_flat, dtype=complex)

    for c in range(n_channels):
        ck_flat = vals_flat[:, c : c + 1]
        # Least squares to find phase factor: ck * pf = ref
        # pf = (ck.H @ ck)^-1 @ ck.H @ ref
        pf, _, _, _ = np.linalg.lstsq(ck_flat, ref_vals_flat, rcond=None)

        # Normalize to get only phase rotation
        if np.abs(pf) > 0:
            pf_normed = pf / np.abs(pf)
        else:
            pf_normed = 1.0

        phased_vals_flat[:, c : c + 1] = ck_flat * pf_normed

    # Sum across channels
    combined_flat = np.sum(phased_vals_flat, axis=-1, keepdims=True)

    # Reshape back to original dimensions (with channel_idx size 1)
    new_shape = list(original_shape)
    new_shape[-1] = 1
    combined_vals_transposed = combined_flat.reshape(new_shape)

    # Move channel axis back to its original position
    combined_vals = np.moveaxis(combined_vals_transposed, -1, channel_axis)

    # Create result DataArray
    result = _create_result_array(working_data, combined_vals, "adaptive")
    result.attrs["reference_channel"] = reference_channel

    # Transform back to original domain if requested
    if target_domain != original_domain:
        if original_domain == "spatial":
            result = _transform_kspace_to_spatial(result)
        else:
            result = _transform_spatial_to_kspace(result)

    return result


def _transform_spatial_to_kspace(data: xr.DataArray) -> xr.DataArray:
    """Transform DataArray from spatial to k-space domain."""
    res = data.copy()
    axes_to_transform = []
    if "read_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("read_idx"))
    if "phase_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("phase_idx"))

    if not axes_to_transform:
        return res

    vals = res.values
    for ax in axes_to_transform:
        vals = np.fft.ifftshift(
            np.fft.ifft(np.fft.ifftshift(vals, axes=ax), axis=ax), axes=ax
        )

    res.values = vals
    res.attrs["signal_domain"] = "k-space"
    return res


def _transform_kspace_to_spatial(data: xr.DataArray) -> xr.DataArray:
    """Transform DataArray from k-space to spatial domain."""
    res = data.copy()
    axes_to_transform = []
    if "read_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("read_idx"))
    if "phase_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("phase_idx"))

    if not axes_to_transform:
        return res

    vals = res.values
    for ax in axes_to_transform:
        vals = np.fft.fftshift(
            np.fft.fft(np.fft.fftshift(vals, axes=ax), axis=ax), axes=ax
        )

    res.values = vals
    res.attrs["signal_domain"] = "spatial"
    return res


def _create_result_array(
    original_array: xr.DataArray, result_values: np.ndarray, method_name: str
) -> xr.DataArray:
    """Helper to create a result DataArray with consistent metadata."""
    # Create new coordinates, keeping channel_idx with size 1
    new_coords = {}
    for dim in original_array.dims:
        if dim in original_array.coords:
            coord = original_array.coords[dim]
            if dim == "channel_idx":
                new_coords[dim] = [0]
            elif "channel_idx" in getattr(coord, "dims", []):
                new_coords[dim] = coord.isel(channel_idx=0)
            else:
                new_coords[dim] = coord

    result = xr.DataArray(
        result_values,
        dims=original_array.dims,
        coords=new_coords,
        attrs=original_array.attrs.copy(),
    )
    result.attrs["channels_combined"] = method_name
    result.attrs["n_channels_combined"] = original_array.sizes["channel_idx"]
    return result
