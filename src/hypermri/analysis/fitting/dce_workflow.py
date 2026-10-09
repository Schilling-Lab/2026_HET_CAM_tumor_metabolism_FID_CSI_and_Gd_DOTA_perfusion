import warnings

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.fit_maps import (
    fit_quantitative_maps,
    signal_to_concentration,
)
from hypermri.utils.mri.channels import combine_channels


def run_dce_pipeline(
    dce_data: xr.DataArray,
    t1_data: xr.DataArray,
    aif_signal: np.ndarray = None,
    aif_signal_concentration: np.ndarray = None,
    aif_times: np.ndarray = None,
    aif_voxel: dict = None,
    tr_s: float = None,
    fa_deg: float = None,
    r1: float = 3.5,
    fit_params_t1: dict = None,
    fit_params_tofts: dict = None,
    use_multiprocessing: bool = True,
    **kwargs,
):
    """
    Run a full DCE-MRI analysis pipeline:
    1. Fit T1 map (per channel if multi-channel).
    2. Combine channels (if multi-channel) using T1 amplitude weights.
    3. Calculate S0 map from pre-contrast DCE frames.
    4. Extract AIF from voxel (if aif_voxel provided) and convert to concentration.
    5. Perform Tofts fitting (1-comp, 2-comp, or both).

    Parameters
    ----------
    dce_data : xr.DataArray
        DCE time-series data. Expected dims: (spatial..., repetition_idx, [channel_idx]).
    t1_data : xr.DataArray
        T1 recovery data. Expected dims: (spatial..., inversion_idx, [channel_idx]).
    aif_signal_concentration : np.ndarray, optional
        Arterial Input Function concentration values.
        If None, aif_voxel must be provided.
    aif_times : np.ndarray, optional
        Time points for AIF in seconds. Defaults to dce_data.frame_time_s if available.
    aif_voxel : dict, optional
        Coordinates for AIF extraction, e.g. {'read_idx': 21, 'phase_idx': 29, 'slice_idx': 8}.
        Is needed, even if aif_signal is passed to select the matching T1 voxel.
    tr_s : float, optional
        Repetition Time in seconds.
    fa_deg : float, optional
        Flip Angle in degrees.
    r1 : float, optional
        Contrast agent relaxivity (default 3.5).
    fit_params_t1 : dict, optional
        Parameters for T1 fitting. Supported keys:
            - 't1_values_s': np.ndarray, T1 grid search values in seconds.
            - 'fit_baseline': bool, if True fits baseline offset.
    fit_params_tofts : dict, optional
        Parameters for Tofts fitting. Supported keys:
            - 'model_mode': "1comp", "2comp", or "both" (default).
            - 'n_precontrast': int, number of frames for S0 calculation (default 5).
            - 'fit_start_index': int, first repetition index to include.
            - 'fit_end_index': int, index after the last repetition to include.
            - 'fit_start_time_s': float, start time in seconds for the window.
            - 'fit_end_time_s': float, end time in seconds for the window.
            - 'fit_indices': list, specific repetition indices to include.
            - 'fit_times_s': list, specific time points in seconds to include.
    use_multiprocessing : bool, optional
        Enable parallel processing (default True).
    **kwargs : additional key word arguments
        fitted_t1: xr.DataSet containing the already fitted T1 data.
            Generate T1 data from inversion recovery by using
            # from hypermri.analysis.fitting.fit_maps import fit_quantitative_maps
            # t1_fit_ms = fit_quantitative_maps(input_data=t1.reco_data,
            #                                   fit_type='t1',
            #                                   fit_params={},)
        aif_signal: xr.DataArray or np.ndarray
            The raw arterial input funciton, not yet converted to concentration.

    """
    if fit_params_t1 is None:
        fit_params_t1 = {}
    if fit_params_tofts is None:
        fit_params_tofts = {}

    # Set defaults for Tofts params if not provided
    n_precontrast = fit_params_tofts.get("n_precontrast", 5)
    model_mode = fit_params_tofts.get("model_mode", "both")
    fit_start_index = fit_params_tofts.get("fit_start_index", 0)

    # --- 0. Data Trimming ---
    # If a start index > 0 is requested, we trim the data completely before any processing.
    # This ensures that these points are truly excluded from S0 calculation, AIF, and convolution.
    if fit_start_index > 0:
        print(f"Trimming data: excluding first {fit_start_index} frames as requested.")
        dce_data = dce_data.isel(repetition_idx=slice(fit_start_index, None))

        if "aif_signal" in kwargs:
            aif_signal_kw = kwargs.get("aif_signal")
            if isinstance(aif_signal_kw, (np.ndarray, xr.DataArray)):
                kwargs["aif_signal"] = aif_signal_kw[fit_start_index:]

        if aif_times is not None:
            aif_times = aif_times[fit_start_index:]

        # Adjust indices in the fit_params to reflect the new data start
        if (
            "fit_end_index" in fit_params_tofts
            and fit_params_tofts["fit_end_index"] is not None
        ):
            fit_params_tofts["fit_end_index"] -= fit_start_index

        # Reset start index since the data is now already trimmed
        fit_params_tofts["fit_start_index"] = 0

    if aif_voxel is None:
        raise ValueError("aif_voxel must be provided!.")

    if "aif_signal" in kwargs:
        aif_signal = kwargs.get("aif_signal", None)

    # --- 1. Parameter Extraction ---
    if tr_s is None:
        tr_ms = dce_data.attrs.get("repetition_time_ms")
        if tr_ms:
            tr_s = tr_ms / 1000.0

    if fa_deg is None:
        fa_deg = dce_data.attrs.get("flipangle_deg")

    if tr_s is None or fa_deg is None:
        raise ValueError(
            "TR (seconds) and Flip Angle (degrees) must be provided or present in dce_data.attrs."
        )

    if aif_times is None:
        if hasattr(dce_data, "frame_time_s"):
            aif_times = np.array(dce_data.frame_time_s)
        elif "time" in dce_data.coords:
            aif_times = dce_data.coords["time"].values
        else:
            aif_times = np.arange(dce_data.sizes.get("repetition_idx", 0)) * tr_s

    if "fitted_t1" in kwargs:
        t1_data = kwargs["fitted_t1"]

    # --- 1.1 Align T1 data to DCE grid ---
    # We want to keep the full DCE grid. T1 data might be a subset (different slices).
    # We identify overlapping slices to know where we can fit.
    from hypermri.utils.plot.utils_plot import get_overlapping_slices

    dce_indices, t1_indices = get_overlapping_slices(
        data_obj_1=dce_data, data_obj_2=t1_data
    )

    print(f"Aligning T1 data (slices {t1_indices}) to DCE data (slices {dce_indices})")
    print(f"Full DCE shape: {dce_data.sizes}")

    # --- 2. T1 Fitting & Channel Combination ---
    print("Step 1/3: Fitting T1 Map (on valid slices)...")

    # Fit T1 only on the overlapping subset to save time
    t1_data_subset = t1_data.isel(slice_idx=t1_indices)

    if "fitted_t1" in kwargs:
        print(f"Step 1/3: No need to fit since T1 fit results were passed!")
        t1_results_subset = kwargs["fitted_t1"].isel(slice_idx=t1_indices)
    else:
        fit_params_t1["use_multiprocessing"] = use_multiprocessing
        t1_results_subset = fit_quantitative_maps(
            input_data=t1_data_subset, fit_type="t1", fit_params=fit_params_t1
        )

    # Combine channels for T1 subset
    t1_map_subset = t1_results_subset["t1"]
    weights_subset = None

    if "channel_idx" in t1_results_subset.dims:
        print(
            "  Multi-channel T1 data detected. Combining using weighted mean (weights=Amplitude)..."
        )
        t1_amplitude = np.abs(t1_results_subset["A"])
        t1_map_subset = combine_channels(
            data_array=t1_results_subset["t1"],
            method="weighted_mean",
            weights=t1_amplitude,
        ).squeeze()
        weights_subset = t1_amplitude

    # --- 3. Create Full Aligned Maps ---
    # Create full T1 map aligned with DCE grid, initialized with NaNs
    # Use dce_data spatial structure
    template_spatial = dce_data.isel(repetition_idx=0, drop=True)
    if "channel_idx" in template_spatial.dims:
        template_spatial = template_spatial.isel(channel_idx=0, drop=True)

    # Force float64 to avoid overflow warnings during assignment/calculation
    t1_map = xr.full_like(template_spatial, np.nan, dtype=np.float64)

    # Assign subset to correct slices
    # We need to use integer indexing for assignment to ensure robust alignment
    # assuming dce_indices map to t1_indices
    # t1_map is (read, phase, slice) usually

    # Get spatial dims
    slice_dim = "slice_idx"
    if slice_dim in t1_map.dims:
        # Create a selector for assignment
        # We assign the values from t1_map_subset to t1_map at positions dce_indices

        # 1. Transpose subset to match target dims
        # Only consider dims present in both
        common_dims = [d for d in t1_map.dims if d in t1_map_subset.dims]
        if len(common_dims) < len(t1_map_subset.dims):
            warnings.warn(
                f"t1_map_subset has dims {t1_map_subset.dims} but target has {t1_map.dims}. Some dims might be lost or mismatched."
            )

        # Transpose subset to match the order in target
        # Use ellipsis to include any remaining dimensions
        t1_subset_aligned = t1_map_subset.transpose(*common_dims, ...)

        t1_subset_vals = t1_subset_aligned.values

        # Check shapes
        # Best approach: Broadcast t1_subset_vals to the shape of the target slice block.
        target_slice_block = t1_map.isel({slice_dim: dce_indices})

        if t1_subset_vals.shape != target_slice_block.shape:
            try:
                t1_subset_vals = t1_subset_vals.reshape(target_slice_block.shape)
            except ValueError:
                # If reshape fails, maybe dimensions are permuted? We already transposed.
                # Maybe sizes differ?
                warnings.warn(
                    f"Shape mismatch in T1 alignment: subset {t1_subset_vals.shape} vs target block {target_slice_block.shape}. Attempting flexible assignment."
                )

        # Assign
        # We need robust slicing for arbitrary dimensions
        full_vals = t1_map.values.copy()
        slice_axis = t1_map.dims.index(slice_dim)
        slicer = [slice(None)] * t1_map.ndim
        slicer[slice_axis] = dce_indices

        # Use the possibly reshaped/broadcasted values
        full_vals[tuple(slicer)] = t1_subset_vals

        t1_map.values = full_vals

    weights = None
    # Handle weights expansion if needed for DCE combination
    if weights_subset is not None:
        # Create weights template with same spatial dims as dce_data
        # but same channel count as weights_subset to ensure they can be aligned
        template_w = dce_data.isel(repetition_idx=0, drop=True)
        if "channel_idx" in weights_subset.dims:
            n_ch_subset = weights_subset.sizes["channel_idx"]
            if template_w.sizes.get("channel_idx", 0) != n_ch_subset:
                # Expand or replace channel_idx in template to match subset
                if "channel_idx" in template_w.dims:
                    template_w = template_w.isel(channel_idx=0, drop=True)
                template_w = template_w.expand_dims(
                    channel_idx=weights_subset.channel_idx
                )

        # Force float64
        weights = xr.full_like(template_w, np.nan, dtype=np.float64)

        # 1. Transpose subset to match target dims
        common_dims_w = [d for d in weights.dims if d in weights_subset.dims]
        weights_subset_aligned = weights_subset.transpose(*common_dims_w, ...)

        weights_subset_vals = weights_subset_aligned.values

        # 2. Check shapes against target block
        slice_axis_w = weights.dims.index(slice_dim)
        target_block_w = weights.isel({slice_dim: dce_indices})

        if weights_subset_vals.shape != target_block_w.shape:
            try:
                weights_subset_vals = weights_subset_vals.reshape(target_block_w.shape)
            except ValueError:
                warnings.warn(
                    f"Shape mismatch in Weights alignment: subset {weights_subset_vals.shape} vs target block {target_block_w.shape}. Attempting flexible assignment."
                )

        full_vals_w = weights.values.copy()
        slicer_w = [slice(None)] * weights.ndim
        slicer_w[slice_axis_w] = dce_indices

        full_vals_w[tuple(slicer_w)] = weights_subset_vals
        weights.values = full_vals_w

    # --- 4. DCE Combination (Full Grid) ---
    dce_combined = dce_data
    if "channel_idx" in dce_data.dims and dce_data.sizes["channel_idx"] > 1:
        print("Step 2/3: Combining DCE channels...")
        if weights is not None:
            try:
                # Combine using weights where available (overlap), RSS elsewhere?
                # Actually combine_channels with NaN weights results in NaNs, which is correct for weighting
                dce_combined = combine_channels(
                    data_array=dce_data, method="weighted_mean", weights=weights
                ).squeeze()
            except Exception as e:
                warnings.warn(f"Weighted combination failed: {e}. Falling back to RSS.")
                dce_combined = combine_channels(dce_data, method="rss").squeeze()
        else:
            dce_combined = combine_channels(dce_data, method="rss").squeeze()

    # --- 5. AIF Extraction ---
    # Determine slice index for AIF
    aif_voxel.get("slice_idx")

    # Check if AIF voxel is in the valid T1 region
    # t1_map has NaNs outside valid region
    try:
        t1_vox = t1_map.isel(aif_voxel).values.item()
    except Exception as e:
        # Might fail if index out of bounds
        raise ValueError(f"AIF voxel {aif_voxel} is out of bounds for T1 map: {e}")

    if np.isnan(t1_vox):
        raise ValueError(
            f"AIF voxel {aif_voxel} is in a slice/region with no T1 data overlap. "
            f"Please select an AIF voxel within the overlapping slices: {dce_indices} (DCE indices)"
        )

    if aif_signal_concentration is None:
        if aif_voxel is None and aif_signal is None:
            raise ValueError(
                "Either aif_signal or aif_signal_concentration must be provided."
            )

        if aif_signal is None:
            print(f"Extracting AIF from voxel: {aif_voxel}")
            # Exctract AIF raw values:
            aif_vox_da = dce_combined.isel(aif_voxel).squeeze()

            # We need S0 for this voxel
            s0_vox = (
                aif_vox_da.isel(repetition_idx=slice(0, n_precontrast))
                .mean()
                .values.item()
            )
            aif_signal = aif_vox_da.values
        else:
            if isinstance(aif_signal, xr.DataArray):
                # Get S0 from aif_signal mean:
                s0_vox = (
                    aif_signal.isel(repetition_idx=slice(0, n_precontrast))
                    .mean()
                    .values.item()
                )
                aif_signal = aif_signal.values

            elif isinstance(aif_signal, np.ndarray):
                aif_signal = np.squeeze(aif_signal)
                # Get S0 from aif_signal mean:
                s0_vox = np.mean(aif_signal[0:n_precontrast])
            else:
                raise Exception("aif_signal has to be xr.DataArray or np.ndarray")

        try:
            aif_signal_concentration = signal_to_concentration(
                S_t=aif_signal,
                S_0=s0_vox,
                T10_s=t1_vox / 1000.0,
                TR_s=tr_s,
                flip_angle_deg=fa_deg,
                r1=r1,
            )
        except Exception as e:
            raise ValueError(
                f"Could not perform signal to concentration conversion: {e}"
            )
    else:
        # If aif_signal is provided as xarray, ensure it's combined and converted if needed
        if isinstance(aif_signal_concentration, xr.DataArray):
            if "channel_idx" in aif_signal_concentration.dims:
                if weights is not None:
                    # Try to combine using the same weights as DCE data
                    try:
                        # Extract the weights for the AIF voxel if we can find it
                        if aif_voxel:
                            w_vox = weights.sel(aif_voxel)
                        else:
                            w_vox = None

                        if w_vox is not None:
                            aif_signal_concentration = combine_channels(
                                aif_signal_concentration,
                                method="weighted_mean",
                                weights=w_vox,
                            )
                        else:
                            aif_signal_concentration = combine_channels(
                                aif_signal_concentration, method="rss"
                            )
                    except:
                        aif_signal_concentration = combine_channels(
                            aif_signal_concentration, method="rss"
                        )
                else:
                    aif_signal_concentration = combine_channels(
                        aif_signal_concentration, method="rss"
                    )

        elif isinstance(aif_signal_concentration, np.ndarray):
            aif_signal_concentration = np.squeeze(aif_signal_concentration)

        # Ensure it's a 1D numpy array for the interpolator
        aif_signal_concentration = np.asarray(aif_signal_concentration).squeeze()
        if aif_signal_concentration.ndim > 1:
            warnings.warn(
                f"aif_signal has {aif_signal_concentration.ndim} dimensions. Squeezing to 1D."
            )
            aif_signal_concentration = (
                aif_signal_concentration.flatten()
            )  # Last resort to ensure 1D

    # --- 5. Tofts Fitting ---
    print("Step 3/3: Fitting Tofts Model...")
    tofts_params = {
        "aif_signal": aif_signal_concentration,
        "aif_times": aif_times,
        "t1_map": t1_map,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "r1": r1,
        "use_multiprocessing": use_multiprocessing,
    }
    # Ensure defaults or provided values in fit_params_tofts are used
    tofts_params.update(
        {
            "n_precontrast": n_precontrast,
            "model_mode": model_mode,
        }
    )
    tofts_params.update(fit_params_tofts)

    dce_results = fit_quantitative_maps(
        input_data=dce_combined, fit_type="tofts", fit_params=tofts_params
    )

    # --- 6. Merge & Finalize ---
    print("Finalizing results...")

    # --- Compute Concentration Map for full volume ---
    # 1. Compute S0 map for full volume
    if n_precontrast > 0:
        s0_map = dce_combined.isel(repetition_idx=slice(0, n_precontrast)).mean(
            dim="repetition_idx"
        )
    else:
        # Fallback if no precontrast frames (unlikely for DCE)
        s0_map = dce_combined.isel(repetition_idx=0)  # rough estimate

    # 2. Compute Concentration Map
    print("Computing Concentration Map...")
    try:
        # Align S0 and T1 maps with the full temporal structure of DCE for vectorized calculation
        # This handles the broadcasting required for numpy-based signal_to_concentration
        s0_map_broadcast = s0_map.broadcast_like(dce_combined)
        t1_map_broadcast = t1_map.broadcast_like(dce_combined)

        concentration_map = signal_to_concentration(
            S_t=dce_combined,
            S_0=s0_map_broadcast,
            T10_s=t1_map_broadcast / 1000.0,
            TR_s=tr_s,
            flip_angle_deg=fa_deg,
            r1=r1,
        )

        # Ensure it has a name
        concentration_map.name = "concentration_map"

    except Exception as e:
        warnings.warn(f"Failed to compute concentration map: {e}")
        concentration_map = None

    final_ds = dce_results

    # Add maps
    final_ds["t1_map"] = t1_map
    final_ds["s0_map"] = s0_map

    if concentration_map is not None:
        # Mask out excluded points at the start if fit_start_index was provided
        # This ensures they don't show up in concentration-based plots
        if n_precontrast > 0:
            # Note: signal_to_concentration returns a DataArray with repetition_idx
            if "repetition_idx" in concentration_map.dims:
                # We use .values for assignment to handle potential broadcast/read-only issues
                # but we must be careful with dimension order.
                # Safer to use isel-based masking if possible, or just slice the values.
                try:
                    # repetition_idx is usually the last dimension in these arrays
                    # but let's be safe and find it
                    rep_axis = concentration_map.dims.index("repetition_idx")
                    slicer = [slice(None)] * concentration_map.ndim
                    slicer[rep_axis] = slice(0, n_precontrast)
                    concentration_map.values[tuple(slicer)] = np.nan
                except Exception as e:
                    warnings.warn(f"Failed to mask concentration map: {e}")

        final_ds["concentration_map"] = concentration_map

    print("Pipeline complete.")
    return final_ds
