import traceback

import numpy as np
import xarray as xr
from IPython.display import display
from ipywidgets import (
    HTML,
    Button,
    Checkbox,
    FloatProgress,
    FloatText,
    HBox,
    IntRangeSlider,
    Label,
    Layout,
    Text,
    ToggleButton,
    VBox,
)

from ..analysis.spectroscopy.core import freq_Hz_to_ppm, freq_ppm_to_Hz, get_freq_axis
from ..compatibility import resolve_frequency_axis
from ..core.experiment import MRIExperiment
from ..utils.system.logging import LOG_MODES, debug_log, init_default_logger
from ..utils.time.temporal import calc_sampling_time_axis

try:
    from matplotlib import colormaps

    get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
except ImportError:
    from matplotlib.cm import get_cmap  # Matplotlib <3.10

from matplotlib.colors import to_hex as mplto_hex

# initialize logger
logger = init_default_logger(__name__)
logger.setLevel(LOG_MODES["Warning"])

DEBUG_LOG = True  # Enable debug logging for widget debugging
DEBUG_FILE = "/tmp/base_spectroscopy_debug.log"  # Use widget-specific debug file


def debug_log_base_spec(message, file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    debug_log(message=message, file_name=file_name, write_to_file=write_to_file)


def debug_log_fit_state(stage, voxel_coords, metabolite_names, amps, freqs, t2s):
    """Log metabolite fit parameters for debugging during refits."""

    if not DEBUG_LOG:
        return

    try:
        voxel_descriptor = ", ".join(
            f"{dim}={int(coord)}" for dim, coord in sorted(voxel_coords.items())
        )
    except Exception:
        voxel_descriptor = str(voxel_coords)

    lines = [f"------------\n[{stage}] voxel({voxel_descriptor})"]
    for metab_name, amp, freq, t2 in zip(metabolite_names, amps, freqs, t2s):
        try:
            amp_repr = complex(amp)
        except Exception:
            amp_repr = amp

        try:
            freq_repr = float(freq)
        except Exception:
            freq_repr = freq

        try:
            t2_repr = float(t2)
        except Exception:
            t2_repr = t2

        lines.append(
            f"  {metab_name}: amp={amp_repr}, freq_Hz={freq_repr}, t2_s={t2_repr} \n------------\n"
        )

    debug_log_base_spec("\n".join(lines))


def is_single_point_dataset(data_obj):
    """Check if a dataset represents single-point spectroscopy data.

    A single-point dataset has spatial dimensions (read_idx, phase_idx, slice_idx)
    and other relevant dimensions (repetition_idx, channel_idx) with size <= 1.

    Parameters
    ----------
    data_obj : xr.Dataset or xr.DataArray
        The data object to check

    Returns
    -------
    bool
        True if this is a single-point dataset, False otherwise
    """
    if data_obj is None:
        return False

    # Check spatial dimensions
    spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
    other_dims = ["repetition_idx", "channel_idx"]

    # Get all relevant dimensions that exist in the data
    relevant_dims = spatial_dims + other_dims
    existing_dims = [dim for dim in relevant_dims if dim in data_obj.dims]

    if not existing_dims:
        # If no relevant dimensions exist, assume it's single-point
        return True

    # Check if all existing relevant dimensions have size <= 1
    for dim in existing_dims:
        if data_obj.sizes[dim] > 1:
            return False

    return True


def get_valid_voxel_coords(data_obj, widget_coords):
    """Extract valid voxel coordinates that exist in both the data and widget state.

    Parameters
    ----------
    data_obj : xr.Dataset or xr.DataArray
        The data object to check dimensions against
    widget_coords : dict
        Dictionary of coordinate values from widgets

    Returns
    -------
    dict
        Dictionary containing only coordinates that exist in the data
    """
    if data_obj is None or widget_coords is None:
        return {}

    valid_coords = {}
    for dim_name, coord_value in widget_coords.items():
        if dim_name in data_obj.dims and data_obj.sizes[dim_name] > 1:
            # Only include coordinates for dimensions that exist and have meaningful size
            valid_coords[dim_name] = coord_value
        elif dim_name in data_obj.dims and data_obj.sizes[dim_name] == 1:
            # For size-1 dimensions, always use index 0
            valid_coords[dim_name] = 0

    return valid_coords


def safe_isel_data(data_obj, coords, fallback_to_first=True):
    """Safely select data using integer-based indexing with fallback options.

    Parameters
    ----------
    data_obj : xr.Dataset or xr.DataArray
        The data object to select from
    coords : dict
        Dictionary of coordinates for selection
    fallback_to_first : bool, default True
        If True, fallback to first index for missing/invalid coordinates

    Returns
    -------
    xr.Dataset or xr.DataArray or None
        Selected data, or None if selection fails
    """
    if data_obj is None:
        return None

    try:
        # Get valid coordinates
        valid_coords = get_valid_voxel_coords(data_obj, coords)

        if not valid_coords:
            if fallback_to_first:
                # For single-point data, just return the data as-is (it's already a single point)
                if is_single_point_dataset(data_obj):
                    return data_obj
                # For multi-dimensional data, select first indices
                fallback_coords = {}
                for dim in data_obj.dims:
                    if dim not in ["metabolite", "acquisition_idx"]:
                        fallback_coords[dim] = 0
                if fallback_coords:
                    return data_obj.isel(fallback_coords)
            return data_obj

        return data_obj.isel(valid_coords)

    except (KeyError, IndexError, ValueError) as e:
        debug_log_base_spec(
            f"Warning: safe_isel_data failed with {e}, returning original data"
        )
        return data_obj


class FitSession:
    """Manages the state of fit parameters for an interactive spectroscopy fitting session.

    This class implements a staging/transactional pattern to efficiently handle
    dynamic metabolite addition/removal without expensive xarray operations until necessary.
    """

    def __init__(
        self,
        initial_fit_params: xr.Dataset,
        initial_fit_results: xr.Dataset,
        live_fit_params: xr.Dataset | None = None,
        spectroscopy_context=None,
    ):
        """Initialize FitSession with original fit parameters and results.

        Parameters
        ----------
        initial_fit_params : xr.Dataset
            Original fit parameters dataset
        initial_fit_results : xr.Dataset
            Original fit results dataset
        """
        # The pristine, original datasets - treat as immutable during session
        self._base_params = initial_fit_params.copy()
        self._base_results = initial_fit_results.copy()

        # Optional reference to the mutable dataset that widgets update directly
        self._live_params = live_fit_params

        # Track in-session overrides so widget edits survive table rebuilds
        self._overrides: dict[str, dict[str, float | bool]] = {}

        # Staging area for changes
        self.added_metabolites = []  # List of dicts with metabolite parameters
        self.removed_metabolites = set()  # Set of metabolite names to remove
        self.committed_metabolites = (
            []
        )  # List of dicts for metabolites committed to results
        self.committed_removals = set()  # Set of committed removals

        # Cache for metabolite names to avoid repeated computation
        self._cached_active_names = None
        self._cache_dirty = True

        # Optional spectroscopy object or axes for ppm↦Hz conversion when adding metabolites
        self._spectroscopy_context = spectroscopy_context
        self._hz_axis = None
        self._ppm_axis = None

        if spectroscopy_context is not None:
            try:
                self._hz_axis = get_freq_axis(data_obj=spectroscopy_context, unit="Hz")
            except Exception as hz_err:
                debug_log_base_spec(
                    f"Warning: Could not precompute Hz axis from spectroscopy context: {hz_err}"
                )
            try:
                self._ppm_axis = get_freq_axis(
                    data_obj=spectroscopy_context, unit="ppm"
                )
            except Exception as ppm_err:
                debug_log_base_spec(
                    f"Warning: Could not precompute ppm axis from spectroscopy context: {ppm_err}"
                )

    def add_metabolite(
        self,
        name: str,
        freq_ppm: float = 0.0,
        freq_hz: float = 0.0,
        freq_axis_hz: float = 70.0,
        initial_t2_s: float = 0.02,
        min_t2_s: float = 0.01,
        max_t2_s: float = 0.05,
        start_with_zero_amplitude: bool = True,
    ):
        """Stage a new metabolite for addition.

        Parameters
        ----------
        name : str
            Name of the new metabolite
        freq_ppm : float, optional
            Frequency in ppm, by default 0.0
        freq_hz : float, optional
            Frequency in Hz (takes precedence over ppm), by default 0.0
        freq_axis_hz : float, optional
            Frequency search range in Hz, by default 70.0
        initial_t2_s : float, optional
            Initial T2* value in seconds, by default 0.05
        min_t2_s : float, optional
            Minimum T2* bound in seconds, by default 0.01
        max_t2_s : float, optional
            Maximum T2* bound in seconds, by default 0.07
        start_with_zero_amplitude : bool, optional
            Whether to initialize with zero amplitude, by default True
        """
        # Validate that the name is not already in use
        current_names = self.get_active_metabolite_names()
        if name in current_names:
            raise ValueError(f"Metabolite '{name}' already exists.")

        # Convert frequency if needed (use spectroscopy context or cached axes)
        if freq_hz == 0.0 and freq_ppm != 0.0:
            converted_freq = self._convert_ppm_to_hz(freq_ppm)
            if converted_freq is None:
                raise ValueError(
                    "Cannot convert ppm frequency to Hz automatically. "
                    "Please provide a frequency in Hz."
                )
            freq_hz = converted_freq

        metabolite_params = {
            "name": name,
            "initial_freqs_Hz": freq_hz,
            "metabs_freq_ranges": freq_axis_hz,
            "initial_t2s_s": initial_t2_s,
            "min_t2s_s": min_t2_s,
            "max_t2s_s": max_t2_s,
            "initial_amps": 0.0 if start_with_zero_amplitude else 1.0,
            "skip_fitting": False,
        }

        self.added_metabolites.append(metabolite_params)
        self._cache_dirty = True

        debug_log_base_spec(f"Added metabolite '{name}' with freq={freq_hz:.1f} Hz")

    def _convert_ppm_to_hz(self, freq_ppm: float) -> float | None:
        """Convert ppm value to Hz using available spectroscopy context."""

        if self._spectroscopy_context is not None:
            try:
                converted = freq_ppm_to_Hz(
                    freq_ppm=freq_ppm,
                    data_obj=self._spectroscopy_context,
                )
                if converted is not None:
                    return float(converted)
            except Exception as conversion_err:
                debug_log_base_spec(
                    f"Warning: ppm→Hz conversion via spectroscopy context failed: {conversion_err}"
                )

        if self._hz_axis is not None and self._ppm_axis is not None:
            try:
                converted = freq_ppm_to_Hz(
                    freq_ppm=freq_ppm,
                    hz_axis=self._hz_axis,
                    ppm_axis=self._ppm_axis,
                )
                if converted is not None:
                    return float(converted)
            except Exception as conversion_err:
                debug_log_base_spec(
                    f"Warning: ppm→Hz conversion via cached axes failed: {conversion_err}"
                )

        return None

    def set_live_params(self, live_fit_params: xr.Dataset | None) -> None:
        """Update the reference dataset that reflects widget edits in-place."""

        self._live_params = live_fit_params

    def update_override(self, metab_name: str, param_key: str, value) -> None:
        """Store an override value for a metabolite parameter.

        Parameters
        ----------
        metab_name : str
            Name of metabolite to update
        param_key : str
            Parameter key matching dataset variable (e.g. "initial_freqs_Hz")
        value : Any
            The value to store for this metabolite/parameter pair
        """
        debug_log_base_spec(f"Override: {metab_name}[{param_key}] = {value}")

        overrides_for_metab = self._overrides.setdefault(metab_name, {})
        overrides_for_metab[param_key] = value

        # Keep staged and committed metabolites in sync so new entries also honor overrides
        for metab in self.added_metabolites + self.committed_metabolites:
            if metab["name"] == metab_name:
                metab[param_key] = value
                break

    def remove_metabolite(self, name: str):
        """Stage a metabolite for removal.

        Parameters
        ----------
        name : str
            Name of metabolite to remove
        """
        # If it's a newly added metabolite, just remove it from the staging list
        original_length = len(self.added_metabolites)
        self.added_metabolites = [
            m for m in self.added_metabolites if m["name"] != name
        ]

        # If it was a base or committed metabolite, add it to the removal set
        if len(self.added_metabolites) == original_length:
            # Check if it's in base params OR committed metabolites
            is_base = name in self._base_params.metabolite.values
            is_committed = any(m["name"] == name for m in self.committed_metabolites)

            if is_base or is_committed:
                self.removed_metabolites.add(name)
                debug_log_base_spec(f"Staged removal of existing metabolite '{name}'")
            else:
                debug_log_base_spec(f"Warning: Metabolite '{name}' not found")
        else:
            debug_log_base_spec(f"Removed newly added metabolite '{name}'")

        self._cache_dirty = True

    def get_active_metabolite_names(self) -> list[str]:
        """Get the list of metabolite names reflecting staged changes.

        Returns
        -------
        list[str]
            List of active metabolite names
        """
        if not self._cache_dirty and self._cached_active_names is not None:
            return self._cached_active_names

        # Preserve original coordinate order from xarray Dataset
        all_base_names = list(self._base_results.metabolite.values)
        active_base_names = [
            name for name in all_base_names if name not in self.removed_metabolites
        ]
        added_names = [m["name"] for m in self.added_metabolites]
        debug_log_base_spec(
            f"Base names: {all_base_names}, Active base names: {active_base_names}, Added names: {added_names}"
        )

        self._cached_active_names = active_base_names + added_names
        self._cache_dirty = False
        debug_log_base_spec(
            f"Active metabolites: {self._cached_active_names}, Base names: {all_base_names}, Added: {added_names}, Removed: {self.removed_metabolites}"
        )

        return self._cached_active_names

    def get_metabolite_params(self, metab_name: str) -> dict:
        """Get parameters for a specific metabolite.

        Parameters
        ----------
        metab_name : str
            Name of metabolite

        Returns
        -------
        dict
            Dictionary of parameters for the metabolite
        """
        # Check if it's a newly added or committed metabolite
        # Combine lists to search both
        for metab in self.added_metabolites + self.committed_metabolites:
            if metab["name"] == metab_name:
                params = metab.copy()
                overrides = self._overrides.get(metab_name)
                if overrides:
                    params.update(overrides)
                return params

        # Otherwise get from datasets (prefer live params if available)
        dataset = None
        if (
            self._live_params is not None
            and "metabolite" in self._live_params.coords
            and metab_name in self._live_params.metabolite.values
        ):
            dataset = self._live_params
        elif metab_name in self._base_params.metabolite.values:
            dataset = self._base_params

        if dataset is not None and "metabolite" in dataset.coords:
            # Extract parameters for this metabolite from the base dataset
            # Use center voxel as reference
            center_coords = self._get_center_voxel_coords(dataset)

            params = {}
            for var_name in [
                "initial_freqs_Hz",
                "metabs_freq_ranges",
                "initial_t2s_s",
                "min_t2s_s",
                "max_t2s_s",
            ]:
                if var_name in dataset:
                    data = dataset[var_name].sel(metabolite=metab_name)
                    if center_coords:
                        data = data.isel(center_coords)
                    params[var_name] = float(data.values)

            params["name"] = metab_name
            if "skip_fitting" in dataset:
                skip_data = dataset["skip_fitting"].sel(metabolite=metab_name)
                if center_coords:
                    skip_data = skip_data.isel(center_coords)
                params["skip_fitting"] = bool(skip_data.values)
            else:
                params["skip_fitting"] = False

            # Apply any widget overrides on top of the dataset values
            overrides = self._overrides.get(metab_name)
            if overrides:
                params.update(overrides)
            return params

        return {}

    def _get_center_voxel_coords(self, dataset: xr.Dataset | None = None) -> dict:
        """Get coordinates for center voxel."""
        coords = {}
        ds = dataset if dataset is not None else self._base_params
        for dim_name in [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "repetition_idx",
            "channel_idx",
        ]:
            if dim_name in ds.dims:
                dim_size = ds.sizes[dim_name]
                coords[dim_name] = dim_size // 2
        return coords

    def build_fit_params(self) -> xr.Dataset:
        """Construct a new Dataset reflecting all staged changes.

        This is the expensive operation, called only before fitting.

        Returns
        -------
        xr.Dataset
            Updated fit parameters dataset
        """
        debug_log_base_spec(
            f"Building fit params with {len(self.added_metabolites)} additions, {len(self.removed_metabolites)} removals"
        )

        # Start with the live parameters (reflecting widget edits) or fall back to base
        base_params = (
            self._live_params if self._live_params is not None else self._base_params
        )
        params = base_params

        # Determine all metabolites to remove
        all_removals = self.removed_metabolites.union(self.committed_removals)
        if all_removals:
            # Only remove if they exist in the params dataset
            existing_removals = [
                m for m in all_removals if m in params.metabolite.values
            ]
            if existing_removals:
                params = params.drop_sel(metabolite=existing_removals)

        # Determine all metabolites to add
        all_additions = self.committed_metabolites + self.added_metabolites
        if not all_additions:
            params = params.copy()
            params.attrs = params.attrs.copy()
            params.attrs["metabs"] = self.get_active_metabolite_names()
            return params

        # Construct a Dataset for the new metabolites
        # We temporarily set added_metabolites to all_additions for the helper
        # to avoid modifying the helper signature
        original_added = self.added_metabolites
        self.added_metabolites = all_additions
        try:
            new_met_ds = self._create_dataset_for_added(params)
        finally:
            self.added_metabolites = original_added

        # Concatenate and return
        result = xr.concat([params, new_met_ds], dim="metabolite")

        # Update attributes
        result.attrs = params.attrs.copy()
        result.attrs["metabs"] = self.get_active_metabolite_names()

        return result

    def build_fit_results(self) -> xr.Dataset:
        """Construct updated fit results dataset with new metabolites.

        Returns
        -------
        xr.Dataset
            Updated fit results dataset with zero-initialized values for new metabolites
        """
        if not self.added_metabolites:
            # Apply removals only
            if self.removed_metabolites:
                return self._base_results.drop_sel(
                    metabolite=list(self.removed_metabolites)
                )
            return self._base_results

        # Create results structure for new metabolites with zero/default values
        new_names = [m["name"] for m in self.added_metabolites]

        # Get reference structure from existing results
        base_results = self._base_results
        if self.removed_metabolites:
            base_results = base_results.drop_sel(
                metabolite=list(self.removed_metabolites)
            )

        # Create zero-filled results for new metabolites
        coords = {"metabolite": new_names}
        for coord_name, coord_val in base_results.coords.items():
            if "metabolite" not in coord_val.dims:
                coords[coord_name] = coord_val

        data_vars = {}
        for var_name in base_results.data_vars:
            debug_log_base_spec(
                f"Processing variable '{var_name}' / {base_results.data_vars} for new metabolites"
            )
            if "metabolite" not in base_results[var_name].dims:
                continue

            ref_array = base_results[var_name]
            new_shape = tuple(
                len(new_names) if dim == "metabolite" else size
                for dim, size in ref_array.sizes.items()
            )
            new_dims = ref_array.dims
            metab_axis = new_dims.index("metabolite")

            if ref_array.dtype == bool:
                broadcast_data = np.zeros(new_shape, dtype=ref_array.dtype)
            else:
                broadcast_data = np.zeros(new_shape, dtype=ref_array.dtype)

            for idx, metab in enumerate(self.added_metabolites):
                if var_name == "fit_freqs":
                    val = float(metab.get("initial_freqs_Hz", 0.0))
                elif var_name == "fit_t2s":
                    val = float(metab.get("initial_t2s_s", 0.015))
                elif var_name == "fit_fwhm_Hz":
                    t2_val = float(metab.get("initial_t2s_s", 0.015))
                    val = 1.0 / (np.pi * t2_val) if t2_val > 0 else 0.0
                elif var_name == "fit_amps":
                    val = float(metab.get("initial_amps", 0.0))
                elif ref_array.dtype == bool:
                    val = False
                else:
                    val = 0.0

                slices = [slice(None)] * len(new_dims)
                slices[metab_axis] = idx
                broadcast_data[tuple(slices)] = val

            data_vars[var_name] = xr.DataArray(
                broadcast_data,
                dims=new_dims,
                coords={
                    dim: coords.get(dim, ref_array.coords[dim]) for dim in new_dims
                },
            )

        new_results_ds = xr.Dataset(data_vars, coords=coords)

        # Concatenate with base results
        return xr.concat([base_results, new_results_ds], dim="metabolite")

    def _create_dataset_for_added(self, reference_ds: xr.Dataset) -> xr.Dataset:
        """Helper to build a dataset for newly added metabolites."""
        new_names = [m["name"] for m in self.added_metabolites]
        coords = {"metabolite": new_names}

        # Inherit non-metabolite coordinates from reference dataset
        for coord_name, coord_val in reference_ds.coords.items():
            if "metabolite" not in coord_val.dims:
                coords[coord_name] = coord_val

        data_vars = {}
        for var_name in reference_ds.data_vars:
            ref_array = reference_ds[var_name]
            if "metabolite" not in ref_array.dims:
                continue  # Skip data_vars that don't depend on metabolites

            # Create array of values for the new metabolites
            default_value = False if ref_array.dtype == bool else np.nan
            values = [m.get(var_name, default_value) for m in self.added_metabolites]

            # Build the new DataArray, broadcasting values across other dimensions
            new_shape = tuple(
                len(new_names) if dim == "metabolite" else size
                for dim, size in ref_array.sizes.items()
            )
            new_dims = ref_array.dims

            # Broadcast the metabolite-specific values across all other dimensions
            broadcast_data = np.zeros(new_shape, dtype=ref_array.dtype)

            # Fill in the values - this assumes metabolite is the last dimension
            metab_axis = new_dims.index("metabolite")
            for i, val in enumerate(values):
                if not np.isnan(val):
                    slices = [slice(None)] * len(new_dims)
                    slices[metab_axis] = i
                    broadcast_data[tuple(slices)] = val

            data_vars[var_name] = xr.DataArray(
                broadcast_data,
                dims=new_dims,
                coords={
                    dim: coords.get(dim, reference_ds.coords[dim]) for dim in new_dims
                },
            )

        return xr.Dataset(data_vars, coords=coords)

    def commit_results(self, fit_results_dataset: xr.Dataset):
        """Commit fit results and clear staging area.

        This method implements the proper state transition after a successful fit:
        1. Updates the base results with the new fitted data
        2. Clears the staging area (added/removed metabolites) but tracks them as committed
        3. Invalidates the cache

        This prevents metabolite duplication on subsequent refits while ensuring
        that build_fit_params() continues to include them.

        Parameters
        ----------
        fit_results_dataset : xr.Dataset
            The updated fit results dataset to commit as the new base
        """
        self._base_results = fit_results_dataset.copy()

        # Move staging to committed
        self.committed_metabolites.extend(self.added_metabolites)
        self.added_metabolites.clear()

        self.committed_removals.update(self.removed_metabolites)
        self.removed_metabolites.clear()

        # Do not clear overrides as they might apply to committed metabolites
        # self._overrides.clear()
        self._cache_dirty = True

        debug_log_base_spec(
            f"Committed fit results with {len(fit_results_dataset.metabolite)} metabolites. "
            f"Staging area moved to committed."
        )


class BaseSpectroscopy(MRIExperiment):
    """ "Base class for most spectroscopy classes.

    This class provides some basic functionality which can be used for most
    spectroscopy sequences. It is encouraged to inherit this class for custom
    spectroscopy sequence classes.

    Methods
    --------
    summary() : pd.Dataframe.Styler
        Prints a summary of all frequency related parameters.

    single_spec_linebroadening(...)
        Linebroadens a given spectrum by LB Herz and returns it.
        See method docs for more info.

    single_spec_find_peaks(...)
        Finds peaks of a given spec using scipy.signal.find_peaks().
        See method docs for more info.

    single_spec_plot(...):
        Plots a single given spectrum. See method docs for more info.
    """

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs
        kwargs.pop("load_data", False)
        kwargs.pop("log_mode", "error")
        kwargs.pop("reco_options", {})

        # CRITICAL FIX: Don't call super().__init__() with new path!
        # Instead, copy all attributes from the loaded MRIExperiment object

        # Copy all essential attributes from the loaded MRIExperiment
        self.path = mri_exp_obj.path
        self.handler = mri_exp_obj.handler  # This preserves the loaded data!
        self.reconstructions = getattr(mri_exp_obj, "reconstructions", {})

        # Copy data arrays if they exist
        if (
            hasattr(mri_exp_obj, "primary_data")
            and mri_exp_obj.primary_data is not None
        ):
            self.data_array = mri_exp_obj.primary_data
        elif hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array
        else:
            self.data_array = None

        # Copy reco_data if it exists (from raw reconstructions)
        if hasattr(mri_exp_obj, "reco_data") and mri_exp_obj.reco_data is not None:
            self.reco_data = mri_exp_obj.reco_data

        # Copy other important attributes from MRIExperiment if they exist and are not None
        # Avoid setting reco_array to None as it triggers property setter validation
        if hasattr(mri_exp_obj, "reco_array") and mri_exp_obj.reco_array is not None:
            self.reco_array = mri_exp_obj.reco_array

        logger.debug(
            f"BaseSpectroscopy: Initialized with handler type {type(self.handler)}"
        )
        logger.debug(
            f"BaseSpectroscopy: Handler has fid: {hasattr(self.handler, 'fid')}"
        )
        if hasattr(self.handler, "fid") and self.handler.fid is not None:
            logger.debug(f"BaseSpectroscopy: Fid shape: {self.handler.fid.shape}")

        # Initialize mixins manually since we're not calling super().__init__()
        # Initialize standardized path handling
        super(
            MRIExperiment, self
        ).__init__()  # This calls StandardizedMRIMixin.__init__()
        # Set path again to ensure mixin is properly initialized
        self.path = mri_exp_obj.path

        # Keep standardized metadata in sync with the handler that still owns the data
        if hasattr(self.handler, "get_parameters"):
            try:
                handler_params = self.handler.get_parameters()
                if handler_params:
                    self.set_parameters(handler_params, allow_extra=True)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(
                    f"BaseSpectroscopy: parameter synchronisation skipped: {exc}"
                )

        # Initialize plotter using registry system
        try:
            from ..visualization.plotter_factory import create_plotter

            self.plotter = create_plotter(
                sequence_obj=self, category="spectroscopy", plotter_type="spectroscopy"
            )
        except ImportError:
            # Fallback to direct import if registry system not available
            from ..utils.plot.spectroscopyPlotter import SpectroscopyPlotter

            self.plotter = SpectroscopyPlotter(self)

        # Initialize standardized data arrays (inherited from StandardizedMRIMixin via MRIExperiment)
        # self.data_array is already initialized via mixin (kspace_array no
        # longer stored)

        # self.reco_rawdata()

    def _load_data_array(self):
        """Lazy loading implementation for scanner reconstructed data (seq2d)."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="seq2d")
            except Exception as e:
                logger.warning(f"Failed to lazy load data_array: {e}")
        return None

    def _load_reco_data(self):
        """Lazy loading implementation for raw reconstructed data with on-demand FID loading."""
        if hasattr(self, "handler") and self.handler is not None:
            # Check if FID data is available, if not try to load it on-demand
            if not hasattr(self.handler, "fid") or self.handler.fid is None:
                logger.info(
                    f"{self.__class__.__name__}: FID data not available, attempting on-demand loading..."
                )
                try:
                    # Load FID data on-demand
                    self.handler.load_data(source=None)
                    logger.info(
                        f"{self.__class__.__name__}: On-demand FID loading successful"
                    )
                except Exception as e:
                    logger.warning(
                        f"{self.__class__.__name__}: On-demand FID loading failed: {e}"
                    )
                    return None

            # Now try reconstruction if FID data is available
            if hasattr(self.handler, "fid") and self.handler.fid is not None:
                try:
                    return self.reconstruct(reco_type="raw")
                except Exception as e:
                    logger.warning(f"Failed to lazy load reco_data: {e}")
            else:
                logger.warning(
                    f"{self.__class__.__name__}: No FID data available for reconstruction"
                )
        return None

    @property
    def sampling_time_axis(self):
        # Prefer pre-calculated axis from handler if available (e.g. for MGE/EPSI)
        handler_axis = getattr(self.handler, "sampling_time_axis_s", None)
        if handler_axis is not None:
            return handler_axis

        return calc_sampling_time_axis(
            data_obj=self, npoints=self.n_acquisition - self.cut_off
        )

    @property
    def time_axis(self):
        return calc_sampling_time_axis(data_obj=self, start_with_0=True)

    @property
    def ppm(self):
        return get_freq_axis(
            data_obj=self, unit="ppm", cut_off=self.cut_off, npoints=None
        )

    @property
    def Hz(self):
        return get_freq_axis(
            data_obj=self, unit="Hz", cut_off=self.cut_off, npoints=None
        )

    def review_fit(
        self,
        input_data: xr.DataArray,
        fit_results: xr.Dataset,
        fit_params: xr.Dataset,
        plot_params: dict = None,
        **slider_defaults,
    ):
        """Interactively review, edit, and refit spectroscopy results.

        This UI-centric helper wires the current reconstruction into the
        spectroscopy plotter, exposes per-metabolite parameter widgets, and adds
        a metabolite management pane. Users can add entirely new metabolites,
        toggle fitting, and trigger voxel refits while seeing immediate feedback
        in the plot display.

        Key behaviours managed here include

        * synchronising the :class:`~hypermri.sequences.base_spectroscopy.FitSession`
          with widget edits so that dynamically added metabolites participate in
          downstream refits;
        * copying the current ``acquisition_idx``-derived frequency into the **Add
          New Metabolite** controls without perturbing the frequencies of the
          already-selected metabolite;
        * automatically selecting a newly-added metabolite in the overlay widget
          so subsequent refits target the staged entry by default;
        * seeding all per-voxel widgets (including staged metabolites) with their
          stored initial parameters whenever the user navigates to a new voxel,
          ensuring the init values captured during creation persist across the
          volume.

        The function mutates neither ``fit_results`` nor ``fit_params`` directly;
        instead it mirrors edits into a copy that lives inside the
        :class:`FitSession` and keeps the plotter in sync. Downstream callers can
        retrieve the modified datasets from the session once the review is
        complete.
        """
        # TEST: Immediate debug log to verify debug_log_base_spec works
        debug_log_base_spec(
            message="review_fit function started - testing debug logging",
        )

        fit_params_new = fit_params.copy()

        # Allow callers to control review layout without leaking the key into the plotter
        layout_mode = "horizontal"
        if plot_params is not None:
            layout_mode = plot_params.get("review_layout", layout_mode)
            if "review_layout" in plot_params:
                plot_params = {
                    key: value
                    for key, value in plot_params.items()
                    if key != "review_layout"
                }
        layout_mode = (
            str(layout_mode).strip().lower()
            if layout_mode is not None
            else "horizontal"
        )
        if layout_mode not in {"horizontal", "vertical"}:
            layout_mode = "horizontal"

        # Convert legacy T2* range format to min/max format for backward compatibility
        if "metabs_t2_ranges" in fit_params_new and "min_t2s_s" not in fit_params_new:
            logger.info("Converting legacy T2* range format to min/max format")
            try:
                initial_t2s = fit_params_new.get("initial_t2s_s")
                t2_ranges = fit_params_new.get("metabs_t2_ranges")

                if initial_t2s is not None and t2_ranges is not None:
                    # Calculate min/max from initial ± range/2
                    min_t2s = initial_t2s - t2_ranges / 2.0
                    max_t2s = initial_t2s + t2_ranges / 2.0

                    # Ensure min values are positive (T2* cannot be negative)
                    min_t2s = np.maximum(min_t2s, 0.001)  # Minimum T2* of 1ms

                    fit_params_new["min_t2s_s"] = min_t2s
                    fit_params_new["max_t2s_s"] = max_t2s

                    logger.info("Successfully converted T2* ranges to min/max format")
                else:
                    logger.warning(
                        "Could not find initial_t2s_s or metabs_t2_ranges for conversion"
                    )
            except Exception as e:
                logger.error(f"Error converting T2* range format: {e}")

        if not hasattr(self, "fit_quality_da"):
            template = fit_results["fit_amps"]
            fit_quality_da = xr.DataArray(
                np.ones(template.shape, dtype=bool),
                coords=template.coords,
                dims=template.dims,
                name="fit_quality",
            )
        else:
            fit_quality_da = self.fit_quality_da.copy()

        # Get metabolites list
        metabs = fit_params.attrs.get("metabs", [])
        if not metabs and "metabolite" in fit_results.dims:
            metabs = list(fit_results.coords["metabolite"].values)

        # Initialize FitSession for dynamic metabolite management
        fit_session = FitSession(
            fit_params_new,
            fit_results,
            live_fit_params=fit_params_new,
            spectroscopy_context=self,
        )

        # --- Create Metabolite Management UI ---
        metabolite_mgmt_title = HTML(value="<b>Metabolite Management</b>")

        # Add metabolite controls
        add_metab_name = Text(
            value="",
            placeholder="New metabolite name",
            description="Name:",
            layout=Layout(width="150px"),
        )
        add_metab_freq_ppm = FloatText(
            value=0.0,
            step=0.1,
            description="Freq [ppm]:",
            layout=Layout(width="150px"),
        )
        add_metab_freq_hz = FloatText(
            value=0.0,
            step=1.0,
            description="Freq [Hz]:",
            layout=Layout(width="150px"),
        )
        add_metab_range = FloatText(
            value=70.0,
            step=5.0,
            description="Range [Hz]:",
            layout=Layout(width="150px"),
        )
        add_metab_t2_init = FloatText(
            value=0.05,
            step=0.005,
            description="Init T2* [s]:",
            layout=Layout(width="150px"),
        )
        add_metab_t2_min = FloatText(
            value=0.01,
            step=0.005,
            description="Min T2* [s]:",
            layout=Layout(width="150px"),
        )
        add_metab_t2_max = FloatText(
            value=0.07,
            step=0.005,
            description="Max T2* [s]:",
            layout=Layout(width="150px"),
        )
        add_zero_amp_toggle = ToggleButton(
            value=True,
            description="Start with zero amplitude",
            button_style="info",
            tooltip="If checked, new metabolite starts with zero amplitude (not fitted initially)",
            layout=Layout(width="180px"),
        )
        add_metab_button = Button(
            description="Add Metabolite",
            button_style="primary",
            layout=Layout(width="150px"),
        )

        # Current metabolites display and removal
        current_metabs_title = HTML(value="<b>Current Metabolites</b>")
        metab_list_area = VBox(
            [],
            layout=Layout(
                min_height="100px", border="1px solid lightgray", padding="5px"
            ),
        )

        def update_metabolite_list():
            """Update the display of current metabolites with remove buttons."""
            active_metabs = fit_session.get_active_metabolite_names()
            metab_widgets = []

            for metab_name in active_metabs:
                # Check if this is an original or added metabolite
                is_added = any(
                    m["name"] == metab_name for m in fit_session.added_metabolites
                )
                style = "warning" if is_added else "info"
                label_text = f"{metab_name} {'(added)' if is_added else '(original)'}"

                metab_label = HTML(value=label_text, layout=Layout(width="150px"))
                remove_button = Button(
                    description="Remove",
                    button_style="danger",
                    layout=Layout(width="80px"),
                )

                # Closure to capture metabolite name
                def make_remove_handler(name):
                    def remove_handler(b):
                        try:
                            # Check if this is a newly added metabolite or original
                            is_added = any(
                                m["name"] == name for m in fit_session.added_metabolites
                            )

                            if is_added:
                                # For newly added metabolites, remove globally since they don't exist in original data
                                fit_session.remove_metabolite(name)
                                debug_log_base_spec(
                                    f"Removed newly added metabolite: {name} (global removal)"
                                )
                            else:
                                # For original metabolites, disable only at current voxel
                                # Get current voxel coordinates from plot widgets
                                current_coords = {}
                                for dim_name in [
                                    "read_idx",
                                    "phase_idx",
                                    "slice_idx",
                                    "repetition_idx",
                                    "channel_idx",
                                ]:
                                    if dim_name in widgets_dict and hasattr(
                                        widgets_dict[dim_name], "value"
                                    ):
                                        current_coords[dim_name] = widgets_dict[
                                            dim_name
                                        ].value

                                # Filter to only include dimensions that exist in fit_params_new
                                param_coords = {
                                    k: v
                                    for k, v in current_coords.items()
                                    if k in fit_params_new.dims
                                }

                                if param_coords:
                                    # Set amplitude to zero for this metabolite at current voxel
                                    if (
                                        "fit_amps" in fit_results
                                        and name in fit_results.metabolite.values
                                    ):
                                        try:
                                            fit_results["fit_amps"].sel(
                                                metabolite=name
                                            ).isel(param_coords).values[()] = 0.0
                                            debug_log_base_spec(
                                                f"Set amplitude to zero for '{name}' at voxel {param_coords}"
                                            )
                                        except Exception as amp_e:
                                            debug_log_base_spec(
                                                f"Warning: Could not set amplitude to zero for {name}: {amp_e}"
                                            )

                                    # Set very tight constraints so it doesn't get fitted at this voxel
                                    if "metabs_freq_ranges" in fit_params_new:
                                        try:
                                            # Find metabolite index
                                            metab_idx = list(
                                                fit_session.get_active_metabolite_names()
                                            ).index(name)
                                            fit_params_new["metabs_freq_ranges"].isel(
                                                param_coords
                                            ).values[metab_idx] = 0.001
                                            debug_log_base_spec(
                                                f"Set tight frequency constraints for '{name}' at voxel {param_coords}"
                                            )
                                        except Exception as range_e:
                                            debug_log_base_spec(
                                                f"Warning: Could not set tight constraints for {name}: {range_e}"
                                            )

                                    debug_log_base_spec(
                                        f"Disabled metabolite: {name} at voxel {param_coords} (voxel-specific)"
                                    )
                                else:
                                    debug_log_base_spec(
                                        f"Warning: Could not get current voxel coordinates for {name}"
                                    )

                            update_metabolite_list()
                            update_parameter_table()

                            # CRITICAL FIX: Synchronize plotter with FitSession changes
                            # This replaces the old update_metabolite_options call and adds dataset synchronization
                            self.plotter.sync_with_fit_session(fit_session)

                            # Manually trigger a plot update to reflect the changes
                            try:
                                current_kwargs = {
                                    k: v.value
                                    for k, v in widgets_dict.items()
                                    if hasattr(v, "value")
                                }
                                run_interactive_update(**current_kwargs)
                                update_init_freq_indicator(trigger_plot_update=False)
                            except Exception as update_e:
                                debug_log_base_spec(
                                    f"Failed to trigger plot update after removing metabolite: {update_e}"
                                )

                            # Update global metabs variable for backward compatibility
                            nonlocal metabs
                            active_metabs = fit_session.get_active_metabolite_names()
                            metabs = active_metabs  # Use the same list for consistency

                            # Re-attach observers after widgets are recreated
                            try:
                                attach_widget_observers()
                            except Exception as obs_e:
                                debug_log_base_spec(
                                    f"Warning: Could not attach observers after removing {name}: {obs_e}"
                                )

                        except Exception as e:
                            debug_log_base_spec(
                                f"Error removing metabolite {name}: {e}"
                            )

                    return remove_handler

                remove_button.on_click(make_remove_handler(metab_name))

                metab_row = HBox([metab_label, remove_button])
                metab_widgets.append(metab_row)

            metab_list_area.children = metab_widgets

        def add_metabolite_handler(b):
            """Handle adding a new metabolite."""
            try:
                existing_metabs = fit_session.get_active_metabolite_names()
                init_freq_widgets_before = metab_param_widgets.get("init_freq", {})
                saved_initial_freqs: dict[str, float] = {}

                for metab_name in existing_metabs:
                    try:
                        widget = init_freq_widgets_before.get(metab_name)
                        if widget is not None and hasattr(widget, "value"):
                            saved_initial_freqs[metab_name] = float(widget.value)
                        else:
                            params = fit_session.get_metabolite_params(metab_name)
                            if params and "initial_freqs_Hz" in params:
                                saved_initial_freqs[metab_name] = float(
                                    params["initial_freqs_Hz"]
                                )
                    except Exception as snapshot_err:
                        debug_log_base_spec(
                            message=(
                                "Warning: Could not snapshot initial frequency for "
                                f"'{metab_name}': {snapshot_err}"
                            ),
                        )

                name = add_metab_name.value.strip()
                if not name:
                    debug_log_base_spec("Error: Metabolite name cannot be empty")
                    return

                # Use Hz frequency if provided, otherwise convert from ppm
                freq_hz = (
                    add_metab_freq_hz.value if add_metab_freq_hz.value != 0.0 else 0.0
                )
                freq_ppm = add_metab_freq_ppm.value if freq_hz == 0.0 else 0.0

                fit_session.add_metabolite(
                    name=name,
                    freq_ppm=freq_ppm,
                    freq_hz=freq_hz,
                    freq_axis_hz=add_metab_range.value,
                    initial_t2_s=add_metab_t2_init.value,
                    min_t2_s=add_metab_t2_min.value,
                    max_t2_s=add_metab_t2_max.value,
                    start_with_zero_amplitude=add_zero_amp_toggle.value,
                )

                # Clear form
                add_metab_name.value = ""
                add_metab_freq_ppm.value = 0.0
                add_metab_freq_hz.value = 0.0

                # Update UI
                update_metabolite_list()
                update_parameter_table()

                def _restore_initial_freqs(freq_snapshot: dict[str, float]) -> None:
                    if not freq_snapshot:
                        return

                    try:
                        current_coords = {
                            dim_name: widget.value
                            for dim_name, widget in widgets_dict.items()
                            if hasattr(widget, "value")
                        }
                        selector_coords = {
                            k: v
                            for k, v in current_coords.items()
                            if k in fit_params_new.dims and k != "metabolite"
                        }
                    except Exception as coord_err:
                        selector_coords = {}
                        debug_log_base_spec(
                            message=(
                                "Warning: Could not build selector while restoring initial "
                                f"frequencies: {coord_err}"
                            ),
                        )

                    restore_widgets = metab_param_widgets.get("init_freq", {})

                    for metab_name, freq_val in freq_snapshot.items():
                        if metab_name == name:
                            continue  # Skip the newly added metabolite

                        widget = restore_widgets.get(metab_name)
                        if widget is None:
                            continue

                        try:
                            if hasattr(widget, "unobserve_all"):
                                widget.unobserve_all()
                            widget.value = float(freq_val)

                            if "initial_freqs_Hz" in fit_params_new:
                                try:
                                    target_da = fit_params_new["initial_freqs_Hz"]
                                    if selector_coords:
                                        target_da = target_da.isel(selector_coords)
                                    target_da.loc[dict(metabolite=metab_name)] = float(
                                        freq_val
                                    )
                                except Exception as live_restore_err:
                                    debug_log_base_spec(
                                        message=(
                                            "Warning: Could not restore live initial_freqs_Hz for "
                                            f"'{metab_name}': {live_restore_err}"
                                        ),
                                    )

                            if (
                                hasattr(fit_session, "_base_params")
                                and "initial_freqs_Hz" in fit_session._base_params
                            ):
                                try:
                                    base_da = fit_session._base_params[
                                        "initial_freqs_Hz"
                                    ]
                                    if selector_coords:
                                        base_da = base_da.isel(selector_coords)
                                    base_da.loc[dict(metabolite=metab_name)] = float(
                                        freq_val
                                    )
                                except Exception as base_restore_err:
                                    debug_log_base_spec(
                                        message=(
                                            "Warning: Could not restore base initial_freqs_Hz for "
                                            f"'{metab_name}': {base_restore_err}"
                                        ),
                                    )

                            fit_session.update_override(
                                metab_name=metab_name,
                                param_key="initial_freqs_Hz",
                                value=float(freq_val),
                            )
                        except Exception as restore_err:
                            debug_log_base_spec(
                                message=(
                                    "Warning: Could not restore initial frequency for "
                                    f"'{metab_name}': {restore_err}"
                                ),
                            )

                _restore_initial_freqs(saved_initial_freqs)

                # CRITICAL FIX: Synchronize plotter with FitSession changes
                # This replaces the old update_metabolite_options call and adds dataset synchronization
                self.plotter.sync_with_fit_session(fit_session)

                # Automatically focus the newly added metabolite in the selection widget
                metabolite_selection_widget = widgets_dict.get(
                    "overlay_selected_metabolites"
                )
                if metabolite_selection_widget is not None and hasattr(
                    metabolite_selection_widget, "value"
                ):
                    try:
                        current_value = metabolite_selection_widget.value
                        if isinstance(current_value, (tuple, list, set)):
                            metabolite_selection_widget.value = (name,)
                        else:
                            metabolite_selection_widget.value = name
                        debug_log_base_spec(
                            message=(
                                "Auto-selected newly added metabolite in selection widget: "
                                f"{name}"
                            ),
                        )
                    except Exception as selection_err:
                        debug_log_base_spec(
                            message=(
                                "Warning: Could not auto-select new metabolite '"
                                f"{name}': {selection_err}"
                            ),
                        )

                # Manually trigger a plot update to reflect the changes
                try:
                    current_kwargs = {
                        k: v.value
                        for k, v in widgets_dict.items()
                        if hasattr(v, "value")
                    }
                    run_interactive_update(**current_kwargs)
                    update_init_freq_indicator(
                        trigger_plot_update=False, metab_hint=name
                    )
                except Exception as update_e:
                    debug_log_base_spec(
                        f"Failed to trigger plot update after adding metabolite: {update_e}"
                    )

                # Update global metabs variable for backward compatibility
                nonlocal metabs
                active_metabs = fit_session.get_active_metabolite_names()
                metabs = active_metabs  # Use the same list for consistency

                # Re-attach observers after widgets are recreated
                try:
                    attach_widget_observers()
                except Exception as obs_e:
                    debug_log_base_spec(
                        f"Warning: Could not attach observers after adding {name}: {obs_e}"
                    )

                debug_log_base_spec(f"Added metabolite: {name}")

            except Exception as e:
                debug_log_base_spec(f"Error adding metabolite: {e}")

        add_metab_button.on_click(add_metabolite_handler)

        # Build metabolite management panel
        add_metab_form = VBox(
            [
                HTML(value="<b>Add New Metabolite</b>"),
                add_metab_name,
                add_metab_freq_ppm,
                add_metab_freq_hz,
                add_metab_range,
                add_metab_t2_init,
                add_metab_t2_min,
                add_metab_t2_max,
            ]
        )

        metabolite_management_panel = HBox(
            [
                VBox([metabolite_mgmt_title, add_metab_form, add_metab_button]),
                # VBox(
                #     [
                #         # current_metabs_title,
                #         # metab_list_area,
                #         # add_zero_amp_toggle,
                #         add_metab_button,
                #     ]
                # ),
            ],
            layout=Layout(border="solid 1px lightgrey", margin="5px"),
        )

        # Initialize the metabolite list display
        update_metabolite_list()

        # --- Create Review Controls ---
        review_title = HTML(value="<b>Fit Review Controls</b>")
        mark_good_button = Button(
            description="Mark Fit as Good",
            button_style="success",
            layout=Layout(width="auto"),
        )
        mark_bad_button = Button(
            description="Mark Fit as Bad",
            button_style="danger",
            layout=Layout(width="auto"),
        )
        finish_button = Button(
            description="Finish Review",
            button_style="info",
            layout=Layout(width="auto"),
        )
        refit_button = Button(
            description="Refit Voxel",
            button_style="warning",
            layout=Layout(width="auto"),
        )

        # Legacy controls (kept for backward compatibility, will be removed in the future)
        fit_range_hz_float = FloatText(
            value=20,
            description="Global Peak Search [Hz]",
            layout=Layout(width="200px"),
        )
        t2_min_float = FloatText(
            value=0.01,
            step=0.005,
            description="Global T2* Min [s]",
            layout=Layout(width="200px"),
        )
        t2_max_float = FloatText(
            value=0.070,
            step=0.005,
            description="Global T2* Max [s]",
            layout=Layout(width="200px"),
        )

        # Add SSE display checkbox and label
        sse_label = Label(value="SSE: N/A", layout=Layout(width="200px"))

        # --- Create Dynamic Per-Metabolite Parameter Table ---
        metab_params_title = HTML(value="<b>Per-Metabolite Parameters</b>")

        # Initialize parameter widgets dictionary - now dynamic
        metab_param_widgets = {}
        metab_params_table = VBox(
            [metab_params_title],
            layout=Layout(border="solid 1px lightgrey", padding="10px", margin="5px"),
        )

        # Global init-frequency offset controls (useful for B0 inhomogeneities)
        freq_offset_input = FloatText(
            value=0.0,
            step=1.0,
            description="Offset [Hz]:",
            layout=Layout(width="140px"),
        )
        apply_freq_offset_button = Button(
            description="Apply Offset",
            button_style="warning",
            tooltip="Add this offset to all Init Freq [Hz] values for the current voxel",
            layout=Layout(width="120px"),
        )

        def update_parameter_table():
            """Update the parameter table with current metabolites from fit_session."""
            active_metabs = fit_session.get_active_metabolite_names()
            # Enable/disable offset application based on availability of metabolites
            apply_freq_offset_button.disabled = not bool(active_metabs)

            if not active_metabs:
                # No metabolites case
                metab_params_table.children = [
                    metab_params_title,
                    HBox(
                        [freq_offset_input, apply_freq_offset_button],
                        layout=Layout(margin="0 0 5px 0"),
                    ),
                    HTML(value="<i>No metabolites in current session</i>"),
                ]
                return

            previous_values: dict[str, dict[str, float | bool]] = {}
            if metab_param_widgets:
                for key in (
                    "freq_range",
                    "init_t2",
                    "min_t2",
                    "max_t2",
                    "enable_fitting",
                ):
                    for metab_name, widget in metab_param_widgets.get(key, {}).items():
                        prev_entry = previous_values.setdefault(metab_name, {})
                        prev_entry[key] = widget.value

            # Create table header
            header_row = [HTML(value="<b>Parameter</b>", layout=Layout(width="100px"))]
            for metab in active_metabs:
                header_row.append(
                    HTML(value=f"<b>{metab}</b>", layout=Layout(width="75px"))
                )

            # Initialize widget dictionaries for current metabolites
            init_freq_widgets = {}
            freq_range_widgets = {}
            init_t2_widgets = {}
            min_t2_widgets = {}
            max_t2_widgets = {}
            enable_fitting_widgets = {}

            # Get initial voxel coordinates for parameter extraction
            def get_initial_voxel_coords():
                """Get coordinates for center voxel or first available voxel for initialization"""
                init_coords = {}
                for dim_name in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                ]:
                    if dim_name in fit_params_new.dims:
                        dim_size = fit_params_new.sizes[dim_name]
                        center_idx = dim_size // 2
                        init_coords[dim_name] = center_idx
                return init_coords

            init_voxel_coords = get_initial_voxel_coords()

            # Create widgets for each metabolite
            for metab_idx, metab in enumerate(active_metabs):
                # Get parameters for this metabolite from fit_session
                metab_params = fit_session.get_metabolite_params(metab)

                # Extract values with fallbacks
                try:
                    init_freq_val = metab_params.get("initial_freqs_Hz", 0.0)
                    freq_range_val = previous_values.get(metab, {}).get(
                        "freq_range",
                        metab_params.get("metabs_freq_ranges", 70.0),
                    )
                    init_t2_val = previous_values.get(metab, {}).get(
                        "init_t2",
                        metab_params.get("initial_t2s_s", 0.05) * 1000.0,
                    )
                    min_t2_val = previous_values.get(metab, {}).get(
                        "min_t2",
                        metab_params.get("min_t2s_s", 0.01) * 1000.0,
                    )
                    max_t2_val = previous_values.get(metab, {}).get(
                        "max_t2",
                        metab_params.get("max_t2s_s", 0.07) * 1000.0,
                    )
                except Exception as e:
                    debug_log_base_spec(
                        f"Warning: Could not extract parameters for {metab}: {e}"
                    )
                    init_freq_val = 0.0
                    freq_range_val = 70.0
                    init_t2_val = 50.0
                    min_t2_val = 10.0
                    max_t2_val = 70.0

                # Create widgets with actual or fallback values
                init_freq_widgets[metab] = FloatText(
                    value=float(init_freq_val),
                    step=1.0,
                    description="",
                    layout=Layout(width="70px"),
                )
                freq_range_widgets[metab] = FloatText(
                    value=float(freq_range_val),
                    step=1.0,
                    description="",
                    layout=Layout(width="70px"),
                )
                init_t2_widgets[metab] = FloatText(
                    value=float(init_t2_val),
                    step=1.0,
                    description="",
                    layout=Layout(width="70px"),
                )
                min_t2_widgets[metab] = FloatText(
                    value=float(min_t2_val),
                    step=1.0,
                    description="",
                    layout=Layout(width="70px"),
                )
                max_t2_widgets[metab] = FloatText(
                    value=float(max_t2_val),
                    step=1.0,
                    description="",
                    layout=Layout(width="70px"),
                )

                # Create enable fitting checkbox
                enable_fitting_val = previous_values.get(metab, {}).get(
                    "enable_fitting",
                    not metab_params.get("skip_fitting", False),
                )
                enable_fitting_widgets[metab] = Checkbox(
                    value=enable_fitting_val,
                    description="",
                    layout=Layout(width="70px"),
                )

            # Store widgets in organized dictionary
            metab_param_widgets.clear()
            metab_param_widgets.update(
                {
                    "init_freq": init_freq_widgets,
                    "freq_range": freq_range_widgets,
                    "init_t2": init_t2_widgets,
                    "min_t2": min_t2_widgets,
                    "max_t2": max_t2_widgets,
                    "enable_fitting": enable_fitting_widgets,
                }
            )

            # Create table rows
            init_freq_row = [HTML(value="Init Freq [Hz]", layout=Layout(width="100px"))]
            for metab in active_metabs:
                init_freq_row.append(init_freq_widgets[metab])

            freq_range_row = [
                HTML(value="Range Freq [Hz]", layout=Layout(width="100px"))
            ]
            for metab in active_metabs:
                freq_range_row.append(freq_range_widgets[metab])

            init_t2_row = [HTML(value="Init T2* [ms]", layout=Layout(width="100px"))]
            for metab in active_metabs:
                init_t2_row.append(init_t2_widgets[metab])

            min_t2_row = [HTML(value="Min T2* [ms]", layout=Layout(width="100px"))]
            for metab in active_metabs:
                min_t2_row.append(min_t2_widgets[metab])

            max_t2_row = [HTML(value="Max T2* [ms]", layout=Layout(width="100px"))]
            for metab in active_metabs:
                max_t2_row.append(max_t2_widgets[metab])

            # Row 6: Enable Fitting
            enable_fitting_row = [
                HTML(value="Enable Fitting", layout=Layout(width="100px"))
            ]
            for metab in active_metabs:
                enable_fitting_row.append(enable_fitting_widgets[metab])

            # Update the table content
            metab_params_table.children = [
                metab_params_title,
                HBox(
                    [freq_offset_input, apply_freq_offset_button],
                    layout=Layout(margin="0 0 5px 0"),
                ),
                HBox(header_row),
                HBox(init_freq_row),
                HBox(freq_range_row),
                HBox(init_t2_row),
                HBox(min_t2_row),
                HBox(max_t2_row),
                HBox(enable_fitting_row),
            ]

            # Populate the global metab_param_widgets dictionary for observer attachment
            metab_param_widgets["init_freq"] = init_freq_widgets
            metab_param_widgets["freq_range"] = freq_range_widgets
            metab_param_widgets["init_t2"] = init_t2_widgets
            metab_param_widgets["min_t2"] = min_t2_widgets
            metab_param_widgets["max_t2"] = max_t2_widgets
            metab_param_widgets["enable_fitting"] = enable_fitting_widgets

        # Initialize the parameter table
        update_parameter_table()

        def attach_widget_observers():
            """Attach observers to the current set of metabolite parameter widgets."""
            active_metabs = fit_session.get_active_metabolite_names()
            if not active_metabs or not metab_param_widgets:
                return

            for metab in active_metabs:
                # Only attach observers if widgets exist for this metabolite
                if (
                    "init_freq" in metab_param_widgets
                    and metab in metab_param_widgets["init_freq"]
                ):
                    metab_param_widgets["init_freq"][metab].observe(
                        create_widget_observer(metab, "initial_frequency"),
                        names="value",
                    )
                if (
                    "freq_range" in metab_param_widgets
                    and metab in metab_param_widgets["freq_range"]
                ):
                    metab_param_widgets["freq_range"][metab].observe(
                        create_widget_observer(metab, "frequency_range"), names="value"
                    )
                if (
                    "init_t2" in metab_param_widgets
                    and metab in metab_param_widgets["init_t2"]
                ):
                    metab_param_widgets["init_t2"][metab].observe(
                        create_widget_observer(metab, "initial_t2"), names="value"
                    )
                if (
                    "min_t2" in metab_param_widgets
                    and metab in metab_param_widgets["min_t2"]
                ):
                    metab_param_widgets["min_t2"][metab].observe(
                        create_widget_observer(metab, "min_t2"), names="value"
                    )
                if (
                    "max_t2" in metab_param_widgets
                    and metab in metab_param_widgets["max_t2"]
                ):
                    metab_param_widgets["max_t2"][metab].observe(
                        create_widget_observer(metab, "max_t2"), names="value"
                    )
                if (
                    "enable_fitting" in metab_param_widgets
                    and metab in metab_param_widgets["enable_fitting"]
                ):
                    metab_param_widgets["enable_fitting"][metab].observe(
                        create_widget_observer(metab, "enable_fitting"), names="value"
                    )

        def create_widget_observer(metab, widget_type):
            """Create observer function for per-metabolite widget"""

            def on_widget_change(change):
                # Define update_fit_params_from_widgets inline to avoid scoping issues
                try:
                    debug_log_base_spec(
                        message=f"Widget '{widget_type}' for '{metab}' changed to {change}",
                    )
                    if not metabs:
                        debug_log_base_spec(
                            message="No metabolites available in fit_session; skipping widget update",
                        )
                        return

                    # Get current voxel coordinates from plot widgets
                    current_coords = {}
                    for dim_name in [
                        "read_idx",
                        "phase_idx",
                        "slice_idx",
                        "repetition_idx",
                        "channel_idx",
                    ]:
                        if dim_name in widgets_dict and hasattr(
                            widgets_dict[dim_name], "value"
                        ):
                            current_coords[dim_name] = widgets_dict[dim_name].value

                    # Filter to only include dimensions that exist in fit_params_new
                    param_coords = {
                        k: v
                        for k, v in current_coords.items()
                        if k in fit_params_new.dims
                    }
                    debug_log_base_spec(
                        f"on_widget_change:\nmetab_param_widgets={metab_param_widgets}"
                    )

                    if metab_param_widgets:
                        selector = param_coords if param_coords else {}
                        metab_lookup = {str(m): m for m in metabs}

                        def _apply_metabolite_updates(
                            target_dataset, var_name, values_map, context_label
                        ):
                            """Assign metabolite values for datasets that already hold the metabolite."""

                            if target_dataset is None or var_name not in target_dataset:
                                return set()

                            target_da = target_dataset[var_name]
                            local_selector = {
                                dim_name: sel_val
                                for dim_name, sel_val in selector.items()
                                if dim_name in target_da.dims
                            }

                            try:
                                target_view = (
                                    target_da.isel(local_selector)
                                    if local_selector
                                    else target_da
                                )
                            except Exception as select_err:
                                debug_log_base_spec(
                                    f"Warning: Unable to select {context_label}.{var_name} with {local_selector}: {select_err}"
                                )
                                return set()

                            if "metabolite" not in target_view.dims:
                                return set()

                            assigned_keys = set()
                            cast_type = getattr(target_view.dtype, "type", None)
                            normalized_map = {str(k): v for k, v in values_map.items()}

                            for metab_coord in target_view.coords["metabolite"].values:
                                norm_key = str(metab_coord)
                                if norm_key not in normalized_map:
                                    continue

                                raw_value = normalized_map[norm_key]
                                cast_value = (
                                    cast_type(raw_value)
                                    if cast_type is not None
                                    else raw_value
                                )

                                try:
                                    target_view.loc[{"metabolite": metab_coord}] = (
                                        cast_value
                                    )
                                except Exception as assign_err:
                                    debug_log_base_spec(
                                        f"Warning: Could not assign {context_label}.{var_name}[{metab_coord}] = {raw_value}: {assign_err}"
                                    )
                                else:
                                    assigned_keys.add(norm_key)

                            return assigned_keys

                        if widget_type == "enable_fitting":
                            skip_values = np.array(
                                [
                                    not metab_param_widgets["enable_fitting"][m].value
                                    for m in metabs
                                ],
                                dtype=bool,
                            )

                            skip_map = {
                                str(metab_name): bool(skip_val)
                                for metab_name, skip_val in zip(metabs, skip_values)
                            }

                            updated_skips = _apply_metabolite_updates(
                                fit_params_new,
                                "skip_fitting",
                                skip_map,
                                "fit_params_new",
                            )

                            if (
                                hasattr(fit_session, "_base_params")
                                and "skip_fitting" in fit_session._base_params
                            ):
                                _apply_metabolite_updates(
                                    fit_session._base_params,
                                    "skip_fitting",
                                    skip_map,
                                    "fit_session._base_params",
                                )

                            missing_skip = [
                                metab_lookup.get(name, name)
                                for name in skip_map
                                if name not in updated_skips
                            ]
                            if missing_skip:
                                debug_log_base_spec(
                                    message=(
                                        "skip_fitting update skipped for metabolites "
                                        f"{missing_skip} (not present in target dataset coordinates)"
                                    )
                                )

                            for metab_name, skip_flag in zip(metabs, skip_values):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="skip_fitting",
                                    value=bool(skip_flag),
                                )
                            return

                        # Update initial frequencies - use .isel() for integer indexing
                        if (
                            "initial_freqs_Hz" in fit_params_new
                            and "init_freq" in metab_param_widgets
                        ):
                            new_freqs = np.array(
                                [
                                    float(metab_param_widgets["init_freq"][m].value)
                                    for m in metabs
                                ]
                            )
                            debug_log_base_spec(
                                f"on_widget_change: new_freqs={new_freqs}"
                            )

                            freq_map = {
                                str(metab_name): freq_val
                                for metab_name, freq_val in zip(metabs, new_freqs)
                            }

                            updated_freqs = _apply_metabolite_updates(
                                fit_params_new,
                                "initial_freqs_Hz",
                                freq_map,
                                "fit_params_new",
                            )

                            if (
                                hasattr(fit_session, "_base_params")
                                and "initial_freqs_Hz" in fit_session._base_params
                            ):
                                _apply_metabolite_updates(
                                    fit_session._base_params,
                                    "initial_freqs_Hz",
                                    freq_map,
                                    "fit_session._base_params",
                                )

                            missing_freqs = [
                                metab_lookup.get(name, name)
                                for name in freq_map
                                if name not in updated_freqs
                            ]
                            if missing_freqs:
                                debug_log_base_spec(
                                    message=(
                                        "initial_freqs_Hz update skipped for metabolites "
                                        f"{missing_freqs} (not present in target dataset coordinates)"
                                    )
                                )

                            debug_log_base_spec(
                                f"metabs={metabs}, new_freqs={new_freqs}"
                            )
                            for metab_name, freq_val in zip(metabs, new_freqs):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="initial_freqs_Hz",
                                    value=float(freq_val),
                                )

                        # Update frequency ranges
                        if (
                            "metabs_freq_ranges" in fit_params_new
                            and "freq_range" in metab_param_widgets
                        ):
                            new_freq_ranges = np.array(
                                [
                                    float(metab_param_widgets["freq_range"][m].value)
                                    for m in metabs
                                ]
                            )
                            debug_log_base_spec(
                                f"on_widget_change: new_freq_ranges={new_freq_ranges}"
                            )

                            freq_range_map = {
                                str(metab_name): range_val
                                for metab_name, range_val in zip(
                                    metabs, new_freq_ranges
                                )
                            }

                            updated_ranges = _apply_metabolite_updates(
                                fit_params_new,
                                "metabs_freq_ranges",
                                freq_range_map,
                                "fit_params_new",
                            )

                            missing_ranges = [
                                metab_lookup.get(name, name)
                                for name in freq_range_map
                                if name not in updated_ranges
                            ]
                            if missing_ranges:
                                debug_log_base_spec(
                                    message=(
                                        "metabs_freq_ranges update skipped for metabolites "
                                        f"{missing_ranges} (not present in target dataset coordinates)"
                                    )
                                )

                            for metab_name, range_val in zip(metabs, new_freq_ranges):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="metabs_freq_ranges",
                                    value=float(range_val),
                                )

                            if widget_type == "frequency_range":
                                selected_widget = widgets_dict.get(
                                    "overlay_selected_metabolites"
                                )
                                selected_metab = None
                                if selected_widget is not None:
                                    selected_value = (
                                        selected_widget.value
                                        if hasattr(selected_widget, "value")
                                        else selected_widget
                                    )
                                    if isinstance(selected_value, (list, tuple, set)):
                                        selected_value = (
                                            list(selected_value)[0]
                                            if selected_value
                                            else None
                                        )
                                    elif isinstance(selected_value, np.ndarray):
                                        selected_value = (
                                            selected_value.tolist()
                                            if selected_value.size > 0
                                            else []
                                        )
                                        selected_value = (
                                            selected_value[0]
                                            if selected_value
                                            else None
                                        )
                                    selected_metab = selected_value

                                trigger_update = (
                                    selected_metab == metab or selected_metab is None
                                )
                                update_init_freq_indicator(
                                    trigger_plot_update=trigger_update,
                                    metab_hint=metab,
                                )

                        # Update initial T2* values
                        if (
                            "initial_t2s_s" in fit_params_new
                            and "init_t2" in metab_param_widgets
                        ):
                            new_t2s_ms = np.array(
                                [
                                    metab_param_widgets["init_t2"][m].value
                                    for m in metabs
                                ]
                            )
                            new_t2s = new_t2s_ms / 1000.0
                            debug_log_base_spec(f"on_widget_change: new_t2s={new_t2s}")

                            t2_map = {
                                str(metab_name): float(t2_val)
                                for metab_name, t2_val in zip(metabs, new_t2s)
                            }

                            updated_t2s = _apply_metabolite_updates(
                                fit_params_new,
                                "initial_t2s_s",
                                t2_map,
                                "fit_params_new",
                            )

                            missing_t2s = [
                                metab_lookup.get(name, name)
                                for name in t2_map
                                if name not in updated_t2s
                            ]
                            if missing_t2s:
                                debug_log_base_spec(
                                    message=(
                                        "initial_t2s_s update skipped for metabolites "
                                        f"{missing_t2s} (not present in target dataset coordinates)"
                                    )
                                )

                            for metab_name, t2_val in zip(metabs, new_t2s):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="initial_t2s_s",
                                    value=float(t2_val),
                                )

                        # Update T2* min/max bounds
                        if (
                            "min_t2s_s" in fit_params_new
                            and "min_t2" in metab_param_widgets
                        ):
                            new_min_t2s_ms = np.array(
                                [metab_param_widgets["min_t2"][m].value for m in metabs]
                            )
                            new_min_t2s = new_min_t2s_ms / 1000.0
                            debug_log_base_spec(
                                f"on_widget_change: new_min_t2s={new_min_t2s}"
                            )

                            min_map = {
                                str(metab_name): float(min_val)
                                for metab_name, min_val in zip(metabs, new_min_t2s)
                            }

                            updated_min = _apply_metabolite_updates(
                                fit_params_new,
                                "min_t2s_s",
                                min_map,
                                "fit_params_new",
                            )

                            missing_min = [
                                metab_lookup.get(name, name)
                                for name in min_map
                                if name not in updated_min
                            ]
                            if missing_min:
                                debug_log_base_spec(
                                    message=(
                                        "min_t2s_s update skipped for metabolites "
                                        f"{missing_min} (not present in target dataset coordinates)"
                                    )
                                )

                            for metab_name, min_val in zip(metabs, new_min_t2s):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="min_t2s_s",
                                    value=float(min_val),
                                )

                        if (
                            "max_t2s_s" in fit_params_new
                            and "max_t2" in metab_param_widgets
                        ):
                            new_max_t2s_ms = np.array(
                                [metab_param_widgets["max_t2"][m].value for m in metabs]
                            )
                            new_max_t2s = new_max_t2s_ms / 1000.0
                            debug_log_base_spec(
                                f"on_widget_change: new_max_t2s={new_max_t2s}"
                            )

                            max_map = {
                                str(metab_name): float(max_val)
                                for metab_name, max_val in zip(metabs, new_max_t2s)
                            }

                            updated_max = _apply_metabolite_updates(
                                fit_params_new,
                                "max_t2s_s",
                                max_map,
                                "fit_params_new",
                            )

                            missing_max = [
                                metab_lookup.get(name, name)
                                for name in max_map
                                if name not in updated_max
                            ]
                            if missing_max:
                                debug_log_base_spec(
                                    message=(
                                        "max_t2s_s update skipped for metabolites "
                                        f"{missing_max} (not present in target dataset coordinates)"
                                    )
                                )

                            for metab_name, max_val in zip(metabs, new_max_t2s):
                                fit_session.update_override(
                                    metab_name=metab_name,
                                    param_key="max_t2s_s",
                                    value=float(max_val),
                                )

                        debug_log_base_spec(
                            message=f"Updated fit_params_new for voxel {param_coords} via widget observer",
                        )

                        if widget_type == "initial_frequency":
                            selected_widget = widgets_dict.get(
                                "overlay_selected_metabolites"
                            )
                            selected_metab = None
                            if selected_widget is not None:
                                selected_value = (
                                    selected_widget.value
                                    if hasattr(selected_widget, "value")
                                    else selected_widget
                                )
                                if isinstance(selected_value, (list, tuple, set)):
                                    selected_value = (
                                        list(selected_value)[0]
                                        if selected_value
                                        else None
                                    )
                                elif isinstance(selected_value, np.ndarray):
                                    selected_value = (
                                        selected_value.tolist()
                                        if selected_value.size > 0
                                        else []
                                    )
                                    selected_value = (
                                        selected_value[0] if selected_value else None
                                    )
                                selected_metab = selected_value

                            trigger_update = (
                                selected_metab == metab or selected_metab is None
                            )
                            update_init_freq_indicator(
                                trigger_plot_update=trigger_update,
                                metab_hint=metab,
                            )

                except Exception as e:
                    logger.warning(
                        f"Widget observer could not update fit_params_new: {e}"
                    )

            return on_widget_change

        # Attach observers to current widgets
        attach_widget_observers()

        # Use the fit_session metabolites for backward compatibility with existing code
        metabs = fit_session.get_active_metabolite_names()

        # --- Masking Controls ---
        masking_title = HTML(value="<b>Masking Controls</b>")
        draw_mask_button = ToggleButton(
            value=False,
            description="Draw Mask",
            button_style="",
            icon="pencil",
            disabled=False,
        )
        refit_masked_button = Button(
            description="Refit Masked",
            button_style="primary",
            layout=Layout(width="auto"),
            disabled=True,
        )
        clear_mask_button = Button(
            description="Clear Mask",
            button_style="info",
            layout=Layout(width="auto"),
            disabled=True,
        )
        fitting_progress = FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            description="",
            bar_style="info",
            orientation="horizontal",
            layout=Layout(width="95%"),
        )
        fitting_info_label = Label(value="")  # New label for additional info

        # Create repetition range slider if repetition_idx dimension exists
        repetition_range_slider = None
        if "repetition_idx" in input_data.dims:
            num_repetitions = input_data.sizes["repetition_idx"]
            if num_repetitions > 1:
                repetition_range_slider = IntRangeSlider(
                    value=[0, num_repetitions - 1],
                    min=0,
                    max=num_repetitions - 1,
                    step=1,
                    description="Reps:",
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    layout=Layout(width="95%"),
                )

        # Build masking management panel widgets list
        masking_widgets = [
            masking_title,
            draw_mask_button,
            refit_masked_button,
            clear_mask_button,
        ]

        # Add repetition range slider if it exists
        if repetition_range_slider is not None:
            masking_widgets.append(repetition_range_slider)

        # Add progress and info widgets
        masking_widgets.extend(
            [
                fitting_progress,
                fitting_info_label,
            ]
        )

        masking_management_panel = VBox(masking_widgets)

        # --- Call Plot ---
        # Ensure overlay plot starts from the current widget defaults rather than an
        # old click position from a previous session. Otherwise the measured and
        # fitted spectra appear misaligned until the user clicks again.
        if hasattr(self.plotter, "clear_click_position"):
            self.plotter.clear_click_position()

        (
            plot_controls,
            plot_output_area,
            run_interactive_update,
            initial_kwargs,
            update_plot_handle,
            widgets_dict,
        ) = self.plot(
            plot_params=plot_params,
            input_data=input_data,
            overlay_data=fit_results,
            return_handles=True,
            **slider_defaults,
        )

        if not hasattr(update_plot_handle, "_init_freq_value") or not isinstance(
            getattr(update_plot_handle, "_init_freq_value"), dict
        ):
            update_plot_handle._init_freq_value = {"Hz": None, "ppm": None}
        else:
            update_plot_handle._init_freq_value.setdefault("Hz", None)
            update_plot_handle._init_freq_value.setdefault("ppm", None)

        debug_log_base_spec(f"_init_freq_value={update_plot_handle._init_freq_value}")

        if not hasattr(update_plot_handle, "_init_freq_marker"):
            update_plot_handle._init_freq_marker = None

        if not hasattr(update_plot_handle, "_init_freq_band") or not isinstance(
            getattr(update_plot_handle, "_init_freq_band"), dict
        ):
            update_plot_handle._init_freq_band = {
                "Hz": (None, None),
                "ppm": (None, None),
            }
        else:
            update_plot_handle._init_freq_band.setdefault("Hz", (None, None))
            update_plot_handle._init_freq_band.setdefault("ppm", (None, None))

        if not hasattr(update_plot_handle, "_init_freq_band_patch"):
            update_plot_handle._init_freq_band_patch = None

        _init_freq_state = {"updating": False}

        def update_init_freq_indicator(
            trigger_plot_update: bool = False, metab_hint: str | None = None
        ):
            """Store current init frequency (Hz/ppm) for the secondary plot indicator."""

            debug_log_base_spec(
                f"update_init_freq_indicator called with trigger_plot_update={trigger_plot_update}, metab_hint={metab_hint}"
            )

            if _init_freq_state["updating"]:
                trigger_plot_update = False

            freq_hz = None
            freq_ppm = None
            freq_range_hz = None
            freq_band_hz = (None, None)
            freq_band_ppm = (None, None)

            try:
                init_widgets = metab_param_widgets.get("init_freq", {})
                debug_log_base_spec(f"metab_param_widgets:\n")
                for m in metab_param_widgets.keys():
                    debug_log_base_spec(f"  {m}: {metab_param_widgets[m]}")
                if init_widgets:
                    selected_metab_widget = widgets_dict.get(
                        "overlay_selected_metabolites"
                    )
                    selected_metab = None
                    if selected_metab_widget is not None:
                        raw_value = (
                            selected_metab_widget.value
                            if hasattr(selected_metab_widget, "value")
                            else selected_metab_widget
                        )
                        if isinstance(raw_value, (list, tuple, set)):
                            raw_list = list(raw_value)
                            selected_metab = raw_list[0] if raw_list else None
                        elif isinstance(raw_value, np.ndarray):
                            raw_list = raw_value.tolist()
                            selected_metab = raw_list[0] if raw_list else None
                        else:
                            selected_metab = raw_value

                    target_metab = None
                    if selected_metab in init_widgets:
                        target_metab = selected_metab
                    elif metab_hint in init_widgets:
                        target_metab = metab_hint
                    elif init_widgets:
                        target_metab = next(iter(init_widgets))

                    if target_metab in init_widgets:
                        metab_widget = init_widgets[target_metab]
                        if hasattr(metab_widget, "value"):
                            try:
                                freq_hz = float(metab_widget.value)
                            except (TypeError, ValueError):
                                freq_hz = None
                            else:
                                if np.isnan(freq_hz):
                                    freq_hz = None

                if freq_hz is not None:
                    try:
                        ppm_val = freq_Hz_to_ppm(freq_Hz=freq_hz, data_obj=self)
                        if isinstance(ppm_val, (list, tuple, set)):
                            ppm_val = list(ppm_val)
                        if isinstance(ppm_val, np.ndarray):
                            ppm_array = np.asarray(ppm_val, dtype=float).ravel()
                            if ppm_array.size > 0 and np.isfinite(ppm_array[0]):
                                freq_ppm = float(ppm_array[0])
                        elif ppm_val is not None and np.isfinite(ppm_val):
                            freq_ppm = float(ppm_val)
                    except Exception as conv_err:
                        debug_log_base_spec(
                            message=f"Failed to convert init freq to ppm: {conv_err}",
                        )

                # Determine frequency range shading information if available
                try:
                    range_widgets = metab_param_widgets.get("freq_range", {})
                    if range_widgets and target_metab in range_widgets:
                        raw_range = range_widgets[target_metab].value
                        freq_range_hz = float(raw_range)
                        if (
                            np.isfinite(freq_range_hz)
                            and freq_range_hz > 0
                            and freq_hz is not None
                        ):
                            half_span = float(freq_range_hz) / 2.0
                            lower_hz = freq_hz - half_span
                            upper_hz = freq_hz + half_span
                            freq_band_hz = (lower_hz, upper_hz)

                            try:
                                lower_ppm_raw = freq_Hz_to_ppm(
                                    freq_Hz=lower_hz, data_obj=self
                                )
                                upper_ppm_raw = freq_Hz_to_ppm(
                                    freq_Hz=upper_hz, data_obj=self
                                )

                                lower_array = (
                                    np.asarray(lower_ppm_raw, dtype=float).ravel()
                                    if lower_ppm_raw is not None
                                    else np.array([])
                                )
                                upper_array = (
                                    np.asarray(upper_ppm_raw, dtype=float).ravel()
                                    if upper_ppm_raw is not None
                                    else np.array([])
                                )

                                if (
                                    lower_array.size > 0
                                    and upper_array.size > 0
                                    and np.isfinite(lower_array[0])
                                    and np.isfinite(upper_array[0])
                                ):
                                    freq_band_ppm = (
                                        float(lower_array[0]),
                                        float(upper_array[0]),
                                    )
                            except Exception as ppm_err:
                                debug_log_base_spec(
                                    message=f"Failed to convert freq range to ppm: {ppm_err}",
                                )
                except Exception as range_err:
                    debug_log_base_spec(
                        message=f"Failed to derive init freq range: {range_err}",
                    )
            except Exception as helper_err:
                debug_log_base_spec(
                    message=f"Failed to prepare init frequency indicator: {helper_err}",
                )
                freq_hz = None
                freq_ppm = None

            init_value = getattr(update_plot_handle, "_init_freq_value", None)
            if not isinstance(init_value, dict):
                init_value = {"Hz": None, "ppm": None}

            init_value["Hz"] = freq_hz
            init_value["ppm"] = freq_ppm
            update_plot_handle._init_freq_value = init_value
            debug_log_base_spec(f"--> init_value={init_value}")

            init_band = getattr(update_plot_handle, "_init_freq_band", {})
            init_band["Hz"] = freq_band_hz
            init_band["ppm"] = freq_band_ppm
            update_plot_handle._init_freq_band = init_band
            debug_log_base_spec(f"--> init_band={init_band}")

            if trigger_plot_update and not _init_freq_state["updating"]:
                try:
                    _init_freq_state["updating"] = True
                    current_kwargs = {
                        name: widget.value
                        for name, widget in widgets_dict.items()
                        if hasattr(widget, "value")
                    }
                    run_interactive_update(**current_kwargs)
                except Exception as update_err:
                    debug_log_base_spec(
                        message=(
                            "Failed to refresh plot after init frequency change: "
                            f"{update_err}"
                        ),
                    )
                finally:
                    _init_freq_state["updating"] = False

        update_plot_handle._init_freq_update_hook = update_init_freq_indicator

        def apply_init_freq_offset(_):
            """Apply a common offset to all Init Freq [Hz] widgets for current voxel."""

            try:
                offset_val = float(freq_offset_input.value)
            except (TypeError, ValueError):
                debug_log_base_spec(
                    "Frequency offset is not a valid number; skipping apply"
                )
                return

            if not np.isfinite(offset_val):
                debug_log_base_spec("Frequency offset is not finite; skipping apply")
                return

            init_widgets = metab_param_widgets.get("init_freq", {})
            if not init_widgets:
                debug_log_base_spec(
                    "No init_freq widgets available; skipping offset application"
                )
                return

            debug_log_base_spec(
                f"Applying offset of {offset_val} Hz to {len(init_widgets)} metabolites"
            )

            for metab_name, widget in init_widgets.items():
                try:
                    current_val = float(widget.value)
                except (TypeError, ValueError):
                    debug_log_base_spec(
                        f"Skipping offset for {metab_name}: non-numeric value"
                    )
                    continue

                new_val = current_val + offset_val
                try:
                    widget.value = new_val
                except Exception as widget_err:
                    debug_log_base_spec(
                        f"Could not apply offset for {metab_name}: {widget_err}"
                    )

            update_init_freq_indicator(trigger_plot_update=True)

        apply_freq_offset_button.on_click(apply_init_freq_offset)

        # Get the overlay_display_mode widget from the plot widgets
        # Check for unprefixed version first (old behavior), then prefixed (new behavior)
        overlay_mode_widget = widgets_dict.get("overlay_display_mode")
        if overlay_mode_widget is None:
            # Try with ds1 prefix (overlay data is typically the second dataset)
            overlay_mode_widget = widgets_dict.get("ds1_overlay_display_mode")
        # === BIDIRECTIONAL BINDING AND MODE-SWITCH FUNCTIONALITY ===

        def update_metab_param_widgets(update_ranges=True):
            """Update metabolite parameter widgets with current voxel values

            Parameters
            ----------
            update_ranges : bool, default True
                If True, update range widgets from fit_params_new.
                If False, preserve user inputs for range widgets and only update fitted values.
            """
            debug_log_base_spec(
                message=f"update_metab_param_widgets called with update_ranges={update_ranges}",
            )
            metabs = fit_session.get_active_metabolite_names()
            if not metabs:
                debug_log_base_spec(
                    message="No metabolites found, returning",
                )
                return

            try:
                # Get current voxel coordinates from plot widgets
                current_coords = {}
                for dim_name in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                    "acquisition_idx",
                ]:
                    if dim_name in widgets_dict and hasattr(
                        widgets_dict[dim_name], "value"
                    ):
                        current_coords[dim_name] = widgets_dict[dim_name].value

                # Create separate coordinate dictionaries for fit_results and fit_params_new
                results_coords = {
                    k: v for k, v in current_coords.items() if k in fit_results.dims
                }
                params_coords = {
                    k: v for k, v in current_coords.items() if k in fit_params_new.dims
                }

                # Identify the metabolite selected in the overlay widget (if any)
                selected_metabolite = None
                metabolite_widget = widgets_dict.get("overlay_selected_metabolites")
                if metabolite_widget is not None:
                    selected_metabolite = (
                        metabolite_widget.value
                        if hasattr(metabolite_widget, "value")
                        else metabolite_widget
                    )
                    if (
                        isinstance(selected_metabolite, (list, tuple))
                        and selected_metabolite
                    ):
                        selected_metabolite = selected_metabolite[0]

                # Determine the frequency chosen in the secondary plot via acquisition index
                freq_from_click = None
                freq_from_click_ppm = None
                acquisition_widget = widgets_dict.get("acquisition_idx")
                acq_value = None
                if acquisition_widget is not None:
                    acq_value = (
                        acquisition_widget.value
                        if hasattr(acquisition_widget, "value")
                        else acquisition_widget
                    )

                try:
                    acq_index = int(acq_value) if acq_value is not None else None
                except (TypeError, ValueError):
                    acq_index = None

                def _get_axis_from_sources(axis_name):
                    for source in (fit_params_new, fit_params, input_data):
                        if source is None:
                            continue
                        attrs = getattr(source, "attrs", None)
                        if attrs and axis_name in attrs:
                            return attrs[axis_name]
                    return None

                if acq_index is not None and acq_index >= 0:
                    hz_axis = _get_axis_from_sources("freq_axis_hz")
                    if hz_axis is not None:
                        try:
                            hz_array = np.asarray(hz_axis, dtype=float).ravel()
                            if acq_index < hz_array.size:
                                freq_from_click = float(hz_array[acq_index])
                        except Exception as e:
                            debug_log_base_spec(
                                message=f"Failed to derive Hz frequency from acquisition_idx: {e}",
                            )

                    ppm_axis = _get_axis_from_sources("freq_axis_ppm")
                    if ppm_axis is not None:
                        try:
                            ppm_array = np.asarray(ppm_axis, dtype=float).ravel()
                            if acq_index < ppm_array.size:
                                freq_from_click_ppm = float(ppm_array[acq_index])
                        except Exception as e:
                            debug_log_base_spec(
                                message=f"Failed to derive ppm frequency from acquisition_idx: {e}",
                            )

                if freq_from_click is not None:
                    try:
                        add_metab_freq_hz.value = freq_from_click
                        debug_log_base_spec(
                            message=(
                                "Set 'Add New Metabolite' Hz frequency from "
                                f"acquisition_idx {acq_index}: {freq_from_click}"
                            ),
                        )
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Could not update Add New Metabolite Hz field: {e}",
                        )

                if freq_from_click_ppm is not None:
                    try:
                        add_metab_freq_ppm.value = freq_from_click_ppm
                        debug_log_base_spec(
                            message=(
                                "Set 'Add New Metabolite' ppm frequency from "
                                f"acquisition_idx {acq_index}: {freq_from_click_ppm}"
                            ),
                        )
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Could not update Add New Metabolite ppm field: {e}",
                        )

                if results_coords:
                    # Update initial frequency and T2* widgets with CURRENT fitted values
                    for i, metab in enumerate(metabs):
                        debug_log_base_spec(
                            message=f"Updating widgets for metabolite '{metab}' at voxel {results_coords}",
                        )
                        try:
                            is_added_metab = any(
                                added["name"] == metab
                                for added in fit_session.added_metabolites
                            )

                            current_fitted_freq = None
                            current_fitted_t2_s = None

                            # Get current fit results (includes updated values from refitting)
                            current_fit_results = fit_session.build_fit_results()

                            if (
                                "metabolite" in current_fit_results.coords
                                and metab in current_fit_results.metabolite.values
                            ):
                                current_fitted_freq = float(
                                    current_fit_results["fit_freqs"]
                                    .isel(results_coords)
                                    .sel(metabolite=metab)
                                    .values
                                )
                                current_fitted_t2_s = float(
                                    current_fit_results["fit_t2s"]
                                    .isel(results_coords)
                                    .sel(metabolite=metab)
                                    .values
                                )

                            freq_nan = current_fitted_freq is not None and np.isnan(
                                current_fitted_freq
                            )
                            t2_nan = current_fitted_t2_s is not None and np.isnan(
                                current_fitted_t2_s
                            )
                            if freq_nan or t2_nan:
                                metab_params = fit_session.get_metabolite_params(metab)
                                current_fitted_freq = float(
                                    metab_params.get("initial_freqs_Hz", 0.0)
                                )
                                current_fitted_t2_s = float(
                                    metab_params.get("initial_t2s_s", 0.05)
                                )

                            if current_fitted_freq is None:
                                metab_params = fit_session.get_metabolite_params(metab)
                                current_fitted_freq = float(
                                    metab_params.get("initial_freqs_Hz", 0.0)
                                )
                                current_fitted_t2_s = float(
                                    metab_params.get("initial_t2s_s", 0.05)
                                )

                            old_freq = metab_param_widgets["init_freq"][metab].value
                            # Temporarily disable widget observers to prevent interference
                            metab_param_widgets["init_freq"][metab].unobserve_all()
                            target_freq = current_fitted_freq

                            metab_param_widgets["init_freq"][metab].value = target_freq
                            # Re-enable observers
                            metab_param_widgets["init_freq"][metab].observe(
                                create_widget_observer(metab, "initial_frequency"),
                                names="value",
                            )
                            debug_log_base_spec(
                                message=(
                                    f"Updated {metab} frequency: {old_freq} -> {target_freq}"
                                ),
                            )

                            current_fitted_t2_ms = current_fitted_t2_s * 1000.0
                            old_t2 = metab_param_widgets["init_t2"][metab].value
                            # Temporarily disable widget observers to prevent interference
                            metab_param_widgets["init_t2"][metab].unobserve_all()
                            metab_param_widgets["init_t2"][
                                metab
                            ].value = current_fitted_t2_ms
                            # Re-enable observers
                            metab_param_widgets["init_t2"][metab].observe(
                                create_widget_observer(metab, "initial_t2"),
                                names="value",
                            )
                            debug_log_base_spec(
                                message=(
                                    f"Updated {metab} T2*: {old_t2} ms -> {current_fitted_t2_ms} ms"
                                ),
                            )

                            # Force UI refresh by triggering layout recalculation
                            try:
                                metab_param_widgets["init_freq"][
                                    metab
                                ].layout.visibility = "hidden"
                                metab_param_widgets["init_freq"][
                                    metab
                                ].layout.visibility = "visible"
                                metab_param_widgets["init_t2"][
                                    metab
                                ].layout.visibility = "hidden"
                                metab_param_widgets["init_t2"][
                                    metab
                                ].layout.visibility = "visible"
                            except Exception as ui_e:
                                debug_log_base_spec(
                                    message=f"Could not force UI refresh: {ui_e}",
                                )

                            # Update enable fitting checkbox to reflect stored skip state
                            if (
                                "skip_fitting" in fit_params_new
                                and "enable_fitting" in metab_param_widgets
                            ):
                                try:
                                    skip_entry = (
                                        fit_params_new["skip_fitting"]
                                        .isel(params_coords)
                                        .sel(metabolite=metab)
                                    )
                                    metab_param_widgets["enable_fitting"][
                                        metab
                                    ].value = not bool(skip_entry.values)
                                except Exception as skip_update_err:
                                    debug_log_base_spec(
                                        message=f"Warning: Could not sync enable_fitting for {metab}: {skip_update_err}",
                                    )

                        except (KeyError, IndexError, ValueError) as e:
                            # Fallback: try to get from fit_params_new if fit_results access fails
                            debug_log_base_spec(
                                message=f"Warning: Could not get fitted values for {metab} from fit_results: {e}",
                            )
                else:
                    debug_log_base_spec(
                        message="Warning: No valid coordinates found for accessing fit_results, trying to set repetition_idx as current voxel coordinates",
                    )
                    results_coords["repetition_idx"] = 0
                    params_coords["repetition_idx"] = 0

                # Only update range widgets if requested (to preserve user inputs)
                if update_ranges and params_coords:
                    # Update frequency range widgets (these come from fit_params_new)
                    if "metabs_freq_ranges" in fit_params_new:
                        freq_ranges = (
                            fit_params_new["metabs_freq_ranges"]
                            .isel(params_coords)
                            .values
                        )
                        if freq_ranges.ndim == 0:
                            freq_ranges = [float(freq_ranges)]
                        elif freq_ranges.ndim == 1:
                            freq_ranges = freq_ranges.tolist()
                        else:
                            freq_ranges = freq_ranges.ravel().tolist()

                        for i, metab in enumerate(metabs):
                            if i < len(freq_ranges):
                                metab_param_widgets["freq_range"][metab].value = float(
                                    freq_ranges[i]
                                )

                    # Update min T2* widgets (these come from fit_params_new)
                    if "min_t2s_s" in fit_params_new:
                        min_t2s = fit_params_new["min_t2s_s"].isel(params_coords).values
                        if min_t2s.ndim == 0:
                            min_t2s = [float(min_t2s) * 1000.0]
                        elif min_t2s.ndim == 1:
                            min_t2s = (min_t2s * 1000.0).tolist()
                        else:
                            min_t2s = (min_t2s * 1000.0).ravel().tolist()

                        for i, metab in enumerate(metabs):
                            if i < len(min_t2s):
                                metab_param_widgets["min_t2"][metab].value = float(
                                    min_t2s[i]
                                )

                    # Update max T2* widgets (these come from fit_params_new)
                    if "max_t2s_s" in fit_params_new:
                        max_t2s = fit_params_new["max_t2s_s"].isel(params_coords).values
                        if max_t2s.ndim == 0:
                            max_t2s = [float(max_t2s) * 1000.0]
                        elif max_t2s.ndim == 1:
                            max_t2s = (max_t2s * 1000.0).tolist()
                        else:
                            max_t2s = (max_t2s * 1000.0).ravel().tolist()

                        for i, metab in enumerate(metabs):
                            if i < len(max_t2s):
                                metab_param_widgets["max_t2"][metab].value = float(
                                    max_t2s[i]
                                )

                update_init_freq_indicator(
                    trigger_plot_update=False, metab_hint=selected_metabolite
                )

            except Exception as e:
                debug_log_base_spec(
                    message=f"Error: Could not update metabolite parameter widgets: {e}"
                )

        def update_widget_enable_state():
            """Enable/disable metabolite widgets based on Individual/Sum mode and selected metabolite"""
            if not metabs:
                return

            try:
                # Get current mode (False = Individual, True = Sum)
                overlay_mode_widget = widgets_dict.get("overlay_display_mode")
                if overlay_mode_widget is None:
                    overlay_mode_widget = widgets_dict.get("ds1_overlay_display_mode")
                if overlay_mode_widget:
                    is_sum_mode = (
                        overlay_mode_widget.value
                        if hasattr(overlay_mode_widget, "value")
                        else overlay_mode_widget
                    )
                else:
                    is_sum_mode = True  # Default to Sum mode if widget not found

                if is_sum_mode:
                    # Sum mode: enable all widgets
                    for metab in metabs:
                        metab_param_widgets["init_freq"][metab].disabled = False
                        metab_param_widgets["freq_range"][metab].disabled = False
                        metab_param_widgets["init_t2"][metab].disabled = False
                        metab_param_widgets["min_t2"][metab].disabled = False
                        metab_param_widgets["max_t2"][metab].disabled = False
                else:
                    # Individual mode: only enable selected metabolite's widgets
                    # Get selected metabolite
                    selected_metab_widget = widgets_dict.get(
                        "overlay_selected_metabolites"
                    )
                    if selected_metab_widget:
                        selected_metab = (
                            selected_metab_widget.value
                            if hasattr(selected_metab_widget, "value")
                            else selected_metab_widget
                        )

                        for metab in metabs:
                            is_selected = metab == selected_metab
                            metab_param_widgets["init_freq"][
                                metab
                            ].disabled = not is_selected
                            metab_param_widgets["freq_range"][
                                metab
                            ].disabled = not is_selected
                            metab_param_widgets["init_t2"][
                                metab
                            ].disabled = not is_selected
                            metab_param_widgets["min_t2"][
                                metab
                            ].disabled = not is_selected
                            metab_param_widgets["max_t2"][
                                metab
                            ].disabled = not is_selected

                        debug_log_base_spec(
                            message=f"Individual mode: only {selected_metab} widgets enabled",
                        )

            except Exception as e:
                debug_log_base_spec(
                    message=f"Error updating widget enable state: {e}",
                )

        def on_mode_switch(change):
            """Handle overlay display mode changes (Sum/Individual)"""
            try:
                is_sum_mode = (
                    change["new"]
                    if hasattr(change, "__getitem__")
                    else change.get("new", False)
                )
                mode_name = "Sum Mode" if is_sum_mode else "Individual Mode"

                debug_log_base_spec(
                    message=f"Switched to {mode_name}",
                )

                # Update metabolite parameter widgets to reflect current voxel
                update_metab_param_widgets(update_ranges=False)

                # Update widget enable/disable state based on mode
                update_widget_enable_state()

            except Exception as e:
                logger.warning(f"Error in mode switch handler: {e}")

        def on_metabolite_selection_change(change):
            """Handle changes to selected metabolite"""
            debug_log_base_spec(
                message=f"Selected metabolite changed to: {change['new']}",
            )
            # Update widget enable state when metabolite selection changes
            update_widget_enable_state()
            new_value = change.get("new") if isinstance(change, dict) else None
            metab_hint = None
            if isinstance(new_value, (list, tuple, set)):
                new_list = list(new_value)
                metab_hint = new_list[0] if new_list else None
            elif isinstance(new_value, np.ndarray):
                new_list = new_value.tolist()
                metab_hint = new_list[0] if new_list else None
            else:
                metab_hint = new_value

            update_init_freq_indicator(trigger_plot_update=True, metab_hint=metab_hint)

        # Add observer to overlay display mode widget for mode switching
        if overlay_mode_widget:
            overlay_mode_widget.observe(on_mode_switch, names="value")

        # Add observer to metabolite selection widget
        metabolite_selection_widget = widgets_dict.get("overlay_selected_metabolites")
        if metabolite_selection_widget:
            metabolite_selection_widget.observe(
                on_metabolite_selection_change, names="value"
            )

        # Initial update of parameter widgets and enable state
        debug_log_base_spec(
            message="About to call initial update_metab_param_widgets",
        )
        update_metab_param_widgets()
        debug_log_base_spec(
            message="Initial update_metab_param_widgets completed",
        )
        update_widget_enable_state()
        selected_metabolite = None
        overlay_widget = widgets_dict.get("overlay_selected_metabolites")
        if overlay_widget is not None:
            raw_value = (
                overlay_widget.value
                if hasattr(overlay_widget, "value")
                else overlay_widget
            )
            if isinstance(raw_value, (list, tuple, set)):
                raw_list = list(raw_value)
                selected_metabolite = raw_list[0] if raw_list else None
            elif isinstance(raw_value, np.ndarray):
                raw_list = raw_value.tolist()
                selected_metabolite = raw_list[0] if raw_list else None
            else:
                selected_metabolite = raw_value

        update_init_freq_indicator(
            trigger_plot_update=False, metab_hint=selected_metabolite
        )

        # --- NEW: Beobachter für Voxel-Index-Widgets ---
        # Wenn der Anwender in einen Voxel klickt, ändern sich die Index-Widget-Werte.
        # Wir registrieren hier Observer an allen bekannten Index-Widgets, damit
        # update_metab_param_widgets bei jeder Änderung ausgeführt wird.
        try:
            voxel_index_names = [
                "read_idx",
                "phase_idx",
                "slice_idx",
                "repetition_idx",
                "channel_idx",
                "acquisition_idx",
            ]

            def _make_voxel_observer():
                def _observer(change):
                    try:
                        debug_log_base_spec(
                            message=f"Voxel widget changed: {change}",
                        )
                        update_metab_param_widgets(update_ranges=False)
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Error in voxel observer: {e}",
                        )

                return _observer

            for vname in voxel_index_names:
                w = widgets_dict.get(vname)
                if w is None:
                    continue
                # Nur hinzufügen, wenn das Widget observe unterstützt
                if hasattr(w, "observe"):
                    try:
                        w.observe(_make_voxel_observer(), names="value")
                        debug_log_base_spec(
                            message=f"Attached voxel observer to widget '{vname}'",
                        )
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Could not attach observer to '{vname}': {e}",
                        )
        except Exception as e:
            debug_log_base_spec(
                message=f"Error while attaching voxel observers: {e}",
            )

        # Create the review controls with overlay mode widget and SSE display
        review_box_layout_kwargs = {
            "border": "solid 1px grey",
            "padding": "5px",
            "margin": "5px",
        }
        if layout_mode == "horizontal":
            review_box_layout_kwargs.update(
                {
                    "flex": "1 1 auto",
                    "min_width": "360px",
                }
            )
        else:
            review_box_layout_kwargs.update(
                {
                    "width": "100%",
                }
            )

        review_controls_box = VBox(
            [
                review_title,
                HBox(
                    [
                        mark_good_button,
                        mark_bad_button,
                        refit_button,
                        finish_button,
                        overlay_mode_widget,
                        sse_label,
                    ]
                ),
                # HBox([fit_range_hz_float, t2_min_float, t2_max_float, sse_label]),
                HBox(
                    [
                        metab_params_table,  # Dynamic metabolite parameters table
                        metabolite_management_panel,
                        masking_management_panel,
                    ]
                ),  # NEW: Metabolite management functionality
            ],
            layout=Layout(**review_box_layout_kwargs),
        )

        # Let the plotter controls expand beside the review pane instead of stacking
        if hasattr(plot_controls, "layout"):
            if layout_mode == "horizontal":
                plot_controls.layout = Layout(
                    flex="1 1 auto",
                    min_width="320px",
                    margin="5px",
                )
            else:
                plot_controls.layout = Layout(width="100%", margin="5px")

        if hasattr(plot_output_area, "layout"):
            plot_output_area.layout = Layout(width="100%")

        plot_handles = {
            "controls": plot_controls,
            "output_area": plot_output_area,
            "run_update": run_interactive_update,
            "init_kwargs": initial_kwargs,
            "update_plot": update_plot_handle,
            "widgets_dict": widgets_dict,
        }

        mask_state = {
            "lasso_selector": None,
            "lasso_patch": None,
            "masked_indices": None,
        }

        def calculate_sse():
            """Calculate SSE between measured and fitted spectrum for current voxel"""
            try:
                # Get current voxel coordinates from widgets
                widget_values = plot_handles["widgets_dict"]
                voxel_coords = {}

                for dim_name in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                ]:
                    if dim_name in widget_values and hasattr(
                        widget_values[dim_name], "value"
                    ):
                        voxel_coords[dim_name] = widget_values[dim_name].value

                # Get measured spectrum for current voxel
                measured_spectrum = input_data.isel(voxel_coords).values.squeeze()

                debug_log_base_spec(
                    message=f"Measured spectrum shape: {measured_spectrum.shape}, dtype: {measured_spectrum.dtype}",
                    file_name="/tmp/sse_debug.log",
                )

                # Get fit parameters for current voxel
                sampling_time_axis_s = fit_params.attrs["sampling_time_axis_s"]

                ####################################
                total_fid = np.zeros_like(measured_spectrum)
                debug_log_base_spec(
                    message=f"Number of metabolites: {len(fit_results.coords['metabolite'].values)}",
                    file_name="/tmp/sse_debug.log",
                )
                for m, metab in enumerate(fit_results.coords["metabolite"].values):
                    try:

                        metab_fid = (
                            fit_results.hp.reconstruct(voxel_coords=voxel_coords)
                            .sel(metabolite=metab)
                            .squeeze()
                        )
                    except (ValueError, IndexError) as e:
                        print(
                            f"Could not reconstruct for {metab} at {voxel_coords}. Error: {e}"
                        )
                        debug_log_base_spec(
                            message=f"Could not reconstruct for {metab} at {voxel_coords}. Error: {e}",
                            file_name="/tmp/sse_debug.log",
                        )
                        continue

                    total_fid += metab_fid

                # Determine plot data and axis
                matching_keys = [
                    key for key in widget_values if key.endswith("_view_toggle")
                ]
                has_view_toggle = len(matching_keys) > 0

                if has_view_toggle:
                    if matching_keys[0]:
                        fitted_temporal = total_fid
                        x_axis = sampling_time_axis_s
                    else:
                        fitted_temporal = np.fft.fftshift(np.fft.fft(total_fid))
                        x_axis = get_freq_axis(
                            self.spectroscopy_obj,
                            widget_values["spec_unit"],
                            self.spectroscopy_obj.cut_off,
                            len(fitted_temporal),
                        )
                else:
                    fitted_temporal = total_fid  # Default to time domain if no toggle

                if fit_params.attrs.get("signal_domain", "spectral") == "spectral":
                    measured_temporal = np.fft.ifft(np.fft.ifftshift(measured_spectrum))
                else:
                    measured_temporal = measured_spectrum

                ######################################################
                # Convert measured to time domain if needed

                # Calculate SSE (same as in fitting functions)
                from ..analysis.fitting.spectroscopic_fitting import _estimate_diff

                sse = _estimate_diff(y=measured_temporal, y_hat=fitted_temporal)

                # Format to 3 significant figures
                if sse > 0:
                    # Convert to scientific notation with 3 significant figures
                    exponent = int(np.floor(np.log10(abs(sse))))
                    mantissa = sse / (10**exponent)
                    sse_formatted = f"{mantissa:.2f}e{exponent:+03d}"
                else:
                    sse_formatted = "0.00e+00"

                return sse_formatted

            except Exception as e:
                debug_log_base_spec(
                    message=f"Error during SSE calculation: {str(e)}\nTraceback: {traceback.format_exc()}",
                    file_name="/tmp/sse_debug.log",
                )
                return f"Error: {str(e)[:20]}"

        def update_sse_display(change=None):
            """Update SSE display when checkbox is checked or voxel changes"""
            sse_value = calculate_sse()
            sse_label.value = f"SSE: {sse_value}"

        # Connect to the main plot update function
        orig_update_plot = plot_handles["update_plot"]

        class _UpdatePlotWrapper:
            """Proxy around the original update function that preserves attributes.

            The plotting stack mutates attributes such as ``ax_img`` and
            ``current_plot_dims`` directly on the callable returned by the plotter.
            When ``review_fit`` adds bookkeeping (SSE updates, widget sync, …) we wrap
            that callable.  Without a proxy the attribute access performed by the
            masking tools fails, because the wrapper function does not expose the
            state set on the original update function.  The proxy forwards attribute
            access (and writes) to the wrapped callable while still allowing us to
            inject post-update callbacks.
            """

            __slots__ = ("_base_func", "_callbacks")

            def __init__(self, base_func, callbacks):
                object.__setattr__(self, "_base_func", base_func)
                object.__setattr__(self, "_callbacks", callbacks or [])

            def __call__(self, *args, **kwargs):
                debug_log_base_spec(
                    message="update_plot_with_sse proxy invoked",
                )
                result = self._base_func(*args, **kwargs)
                for callback in self._callbacks:
                    try:
                        callback()
                    except Exception as cb_err:
                        debug_log_base_spec(
                            message=(
                                "update_plot_with_sse callback failed: "
                                f"{cb_err}\n" + traceback.format_exc()
                            ),
                        )
                return result

            def __getattr__(self, name):
                return getattr(self._base_func, name)

            def __setattr__(self, name, value):
                if name in {"_base_func", "_callbacks"}:
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._base_func, name, value)

        plot_handles["update_plot"] = _UpdatePlotWrapper(
            base_func=orig_update_plot,
            callbacks=[
                update_sse_display,
                lambda: update_metab_param_widgets(update_ranges=False),
            ],
        )

        if layout_mode == "horizontal":
            controls_container = HBox(
                [plot_handles["controls"], review_controls_box],
                layout=Layout(align_items="flex-start", width="100%"),
            )
        else:
            controls_container = VBox(
                [plot_handles["controls"], review_controls_box],
                layout=Layout(width="100%"),
            )

        full_layout = VBox(
            [controls_container, plot_handles["output_area"]],
            layout=Layout(width="100%"),
        )

        display(full_layout)

        def on_lasso_select(verts):
            """Handle lasso selection for masking with comprehensive validation."""
            from matplotlib.patches import Polygon
            from matplotlib.path import Path

            try:
                # Input validation
                if not verts or len(verts) < 3:
                    debug_log_base_spec(
                        message="Lasso select aborted: not enough vertices",
                        file_name="/tmp/masking_debug.log",
                    )
                    draw_mask_button.value = False
                    return

                update_plot_ref = plot_handles["update_plot"]

                # Validate plot handles and axes
                ax_img = getattr(update_plot_ref, "ax_img", None)
                if ax_img is None:
                    debug_log_base_spec(
                        message="Mask abort: update_plot lacks ax_img",
                        file_name="/tmp/masking_debug.log",
                    )
                    draw_mask_button.value = False
                    return

                # Validate plot dimensions and extent, supporting legacy attribute names
                plot_dims = getattr(update_plot_ref, "current_plot_dims", None)

                # Some plotters expose integer indices (e.g. (0, 1)); prefer
                # the explicit labels stored in ``_current_plot_dims`` when present.
                labeled_plot_dims = getattr(update_plot_ref, "_current_plot_dims", None)
                if plot_dims and any(not isinstance(dim, str) for dim in plot_dims):
                    plot_dims = labeled_plot_dims or plot_dims

                if not plot_dims:
                    plot_dims = labeled_plot_dims

                debug_log_base_spec(
                    message=f"Mask toggle using plot dims {plot_dims}",
                    file_name="/tmp/masking_debug.log",
                )

                if not plot_dims or len(plot_dims) < 2:
                    debug_log_base_spec(
                        message=(
                            "Mask abort: could not determine plot dimensions "
                            f"(found: {plot_dims})"
                        ),
                        file_name="/tmp/masking_debug.log",
                    )
                    draw_mask_button.value = False
                    return

                extent = getattr(update_plot_ref, "current_extent", None)
                if extent is None:
                    extent = getattr(update_plot_ref, "_current_extent", None)

                if extent is None:
                    debug_log_base_spec(
                        message="Mask abort: no extent on update_plot",
                        file_name="/tmp/masking_debug.log",
                    )
                    draw_mask_button.value = False
                    return

                # Remove existing mask patch
                if mask_state["lasso_patch"]:
                    try:
                        mask_state["lasso_patch"].remove()
                    except Exception:
                        raise

                # Create new mask patch
                mask_state["lasso_patch"] = ax_img.add_patch(
                    Polygon(
                        verts,
                        closed=True,
                        fill=True,
                        edgecolor="yellow",
                        facecolor="yellow",
                        alpha=0.5,
                        zorder=15,
                    )
                )

                # Coordinate conversion with validation
                lasso_path = Path(verts)
                plot_dim_y, plot_dim_x = plot_dims[:2]

                # Validate dimensions exist in data
                if (
                    plot_dim_y not in input_data.dims
                    or plot_dim_x not in input_data.dims
                ):
                    debug_log_base_spec(
                        message=(
                            "Mask abort: plot dims not in data dims "
                            f"({plot_dim_y}/{plot_dim_x} vs {input_data.dims})"
                        ),
                        file_name="/tmp/masking_debug.log",
                    )
                    _clear_mask()
                    return

                ny, nx = input_data.sizes[plot_dim_y], input_data.sizes[plot_dim_x]

                # Validate extent format
                if not extent or len(extent) != 4:
                    debug_log_base_spec(
                        message=f"Mask abort: invalid extent format {extent}",
                        file_name="/tmp/masking_debug.log",
                    )
                    _clear_mask()
                    return

                # Validate that extent values are numeric
                try:
                    extent = [float(x) for x in extent]
                except (TypeError, ValueError):
                    print("Error: Non-numeric values in plot extent.")
                    debug_log_base_spec(
                        message=f"Mask abort: non-numeric extent {extent}",
                        file_name="/tmp/masking_debug.log",
                    )
                    _clear_mask()
                    return

                # Create coordinate grids with validation
                if extent[1] <= extent[0] or extent[3] <= extent[2]:
                    print("Error: Invalid extent range (min >= max).")
                    debug_log_base_spec(
                        message=f"Mask abort: invalid extent ordering {extent}",
                        file_name="/tmp/masking_debug.log",
                    )
                    _clear_mask()
                    return

                x_coords = np.linspace(extent[0], extent[1], nx)
                y_coords = np.linspace(extent[2], extent[3], ny)
                xv, yv = np.meshgrid(x_coords, y_coords)
                points_grid = np.vstack((xv.ravel(), yv.ravel())).T

                # Create mask with validation
                mask_2d = lasso_path.contains_points(points_grid).reshape((ny, nx))
                masked_y, masked_x = np.where(mask_2d)

                # Validate coordinate indices are within bounds
                valid_indices = []
                for y, x in zip(masked_y, masked_x):
                    if 0 <= y < ny and 0 <= x < nx:
                        valid_indices.append({plot_dim_y: int(y), plot_dim_x: int(x)})
                    else:
                        debug_log_base_spec(
                            message=f"Warning: Skipping out-of-bounds coordinate: y={y}, x={x} (bounds: y<{ny}, x<{nx})",
                            file_name="/tmp/masking_debug.log",
                        )

                mask_state["masked_indices"] = valid_indices

                debug_log_base_spec(
                    message=(
                        "Mask selection completed: "
                        f"{len(valid_indices)} voxels on dims {plot_dim_y}/{plot_dim_x}"
                    ),
                    file_name="/tmp/masking_debug.log",
                )

                fitting_info_label.value = f"Sel. {len(valid_indices)} voxels for refit"

                if mask_state["masked_indices"]:
                    refit_masked_button.disabled = False
                    clear_mask_button.disabled = False
                else:
                    _clear_mask()

            except Exception as e:
                debug_log_base_spec(
                    message=f"Error in lasso selection: {str(e)}",
                    file_name="/tmp/masking_debug.log",
                )
                _clear_mask()
                return

            finally:
                draw_mask_button.value = False
                try:
                    # Use the matplotlib figure we found earlier instead of plot_handles
                    import matplotlib.pyplot as plt

                    fig = plt.gcf()
                    if fig:
                        fig.canvas.draw_idle()
                except Exception as e:
                    debug_log_base_spec(
                        message=f"Warning: Could not refresh plot canvas: {e}",
                    )

        def on_draw_mask_toggle(change):
            """Handle mask drawing toggle with error handling and validation."""
            import sys

            try:
                update_plot_ref = plot_handles.get("update_plot")

                debug_log_base_spec(
                    message=(
                        "Mask toggle triggered "
                        f"(new={change.get('new') if isinstance(change, dict) else change})"
                    ),
                    file_name="/tmp/masking_debug.log",
                )

                ax_img = None

                # Method 1: Try to get the current figure
                try:
                    import matplotlib.pyplot as plt

                    fig = None

                    if update_plot_ref is not None and hasattr(update_plot_ref, "fig"):
                        fig = update_plot_ref.fig

                    if fig is None:
                        fig = plt.gcf()  # Get current figure
                    sys.stdout.flush()

                    if fig and hasattr(fig, "axes") and fig.axes:
                        sys.stdout.flush()

                        # Look for axes with images (spatial plots)
                        for i, ax in enumerate(fig.axes):
                            has_images = hasattr(ax, "images") and len(ax.images) > 0
                            has_collections = (
                                hasattr(ax, "collections") and len(ax.collections) > 0
                            )

                            # Prioritize axes with images (spatial plots)
                            if has_images:
                                ax_img = ax
                                break

                        # If no axes with images, look for axes with collections (alternative plots)
                        if not ax_img:
                            for i, ax in enumerate(fig.axes):
                                if (
                                    hasattr(ax, "collections")
                                    and len(ax.collections) > 0
                                ):
                                    ax_img = ax
                                    break

                        # Last resort: use first available axes
                        if not ax_img and fig.axes:
                            ax_img = fig.axes[0]

                except Exception as e:
                    debug_log_base_spec(
                        message=f"ERROR: Failed to get matplotlib figure: {str(e)}",
                        file_name="/tmp/masking_debug.log",
                    )

                # Final validation
                if not ax_img:
                    debug_log_base_spec(
                        message="ERROR: No suitable axes found for masking - may need spatial plot",
                        file_name="/tmp/masking_debug.log",
                    )
                    draw_mask_button.value = False
                    return

                # Validate change parameter
                if not isinstance(change, dict) or "new" not in change:
                    debug_log_base_spec(
                        message="Invalid change parameter in mask toggle",
                        file_name="/tmp/masking_debug.log",
                    )
                    return

                if change["new"]:
                    # Activating lasso selector
                    try:

                        if not mask_state["lasso_selector"]:
                            from matplotlib.widgets import LassoSelector

                            mask_state["lasso_selector"] = LassoSelector(
                                ax_img, on_lasso_select, useblit=True
                            )
                            debug_log_base_spec(
                                message="Created new LassoSelector",
                                file_name="/tmp/masking_debug.log",
                            )

                        else:
                            pass

                        mask_state["lasso_selector"].set_active(True)
                        debug_log_base_spec(
                            message="Activated LassoSelector",
                            file_name="/tmp/masking_debug.log",
                        )

                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Error activating lasso selector: {str(e)}",
                        )
                        draw_mask_button.value = False

                else:
                    # Deactivating lasso selector
                    try:
                        if mask_state["lasso_selector"]:
                            mask_state["lasso_selector"].set_active(False)
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Error deactivating lasso selector: {str(e)}",
                        )

            except Exception as e:
                debug_log_base_spec(
                    message=f"Unexpected error in mask toggle: {str(e)}",
                )
                draw_mask_button.value = False

        def _clear_mask(b=None):
            """Clear mask with comprehensive error handling."""
            try:
                # Safely remove mask patch
                if mask_state.get("lasso_patch"):
                    try:
                        mask_state["lasso_patch"].remove()
                    except Exception as e:
                        debug_log_base_spec(
                            message=f"Warning: Could not remove mask patch: {str(e)}",
                        )
                        # Continue with clearing state even if patch removal fails

                # Reset mask state safely
                try:
                    mask_state.update({"lasso_patch": None, "masked_indices": None})

                    # Disable related buttons
                    if refit_masked_button:
                        refit_masked_button.disabled = True
                    if clear_mask_button:
                        clear_mask_button.disabled = True

                except Exception as e:
                    debug_log_base_spec(
                        message=f"Warning: Error updating mask state: {str(e)}",
                    )

                # Safely refresh plot
                try:
                    if (
                        hasattr(plot_handles["update_plot"], "fig")
                        and plot_handles["update_plot"].fig is not None
                    ):
                        plot_handles["update_plot"].fig.canvas.draw_idle()
                except Exception as e:
                    debug_log_base_spec(
                        message=f"Warning: Could not refresh plot after mask clearing: {str(e)}",
                    )

                print("Mask cleared.")

            except Exception as e:
                debug_log_base_spec(
                    message=f"Error in mask clearing: {str(e)}",
                )
                print(f"Error clearing mask: {str(e)}")

        def _mark_fit_quality(is_good):
            """Mark fit quality with comprehensive validation and error handling."""
            try:
                # Validate input parameter
                if not isinstance(is_good, (bool, int)):
                    return

                # Validate plot handles and widgets exist
                if "widgets_dict" not in plot_handles:
                    return

                widget_values = plot_handles["widgets_dict"]
                if not widget_values:
                    return

                # Validate fit quality data array exists
                if fit_quality_da is None:
                    return

                # Extract voxel coordinates with validation
                voxel_indices = {}
                for dim_name in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                ]:
                    if dim_name in widget_values and dim_name in fit_quality_da.dims:
                        widget = widget_values[dim_name]
                        if hasattr(widget, "value"):
                            # Validate widget value is within valid range
                            widget_value = widget.value
                            max_index = fit_quality_da.sizes[dim_name] - 1
                            if not (0 <= widget_value <= max_index):
                                print(
                                    f"Warning: Widget value {widget_value} for {dim_name} out of range [0, {max_index}]"
                                )
                                continue
                            voxel_indices[dim_name] = int(widget_value)
                        else:
                            debug_log_base_spec(
                                message=f"Widget for {dim_name} has no 'value' attribute",
                            )

                if not voxel_indices:
                    return

                # Validate coordinates exist in fit quality array
                try:
                    # Test indexing before actual assignment
                    _ = fit_quality_da.loc[voxel_indices]
                except (KeyError, IndexError) as e:
                    print(f"Error: Invalid voxel coordinates {voxel_indices}: {str(e)}")
                    return

                # Mark fit quality
                try:
                    fit_quality_da.loc[voxel_indices] = bool(is_good)
                except Exception as e:
                    debug_log_base_spec(
                        message=f"Error assigning fit quality: {str(e)}",
                    )

            except Exception as e:
                debug_log_base_spec(
                    message=f"Unexpected error in _mark_fit_quality: {str(e)}",
                )

        def _refit_logic(list_of_indices):
            """Refits spectral data for selected voxels based on interactive input.
            This function serves as the core logic for interactively refitting
            spectroscopic data, typically triggered by a user action like a mouse click
            on a spectral plot. It reads the current state from various UI widgets
            (e.g., sliders, dropdowns) to perform a targeted refit of one or more
            voxels.
            The refitting process can operate in two distinct modes determined by the
            'overlay_display_mode' widget:
            1.  **Individual Mode**: Only the user-selected metabolite's parameters are
                allowed to vary significantly during the fit. The initial frequency for
                this metabolite is set to the new position clicked by the user. All
                other metabolites are effectively "frozen" by setting their fitting
                bounds to be extremely tight around their previously fitted values.
            2.  **Sum Mode**: All metabolites are refit. The user-selected metabolite's
                frequency is initialized to the new clicked position, while all other
                metabolites start the fit from their last fitted values. All
                metabolites are allowed to vary within the user-defined fit range.
            The function performs the following steps:
            - Gathers parameters from UI widgets (e.g., selected metabolite, clicked
              frequency, fit width).
            - Implements fallback logic to robustly identify the metabolite selection
              widget if it's not found with the expected key.
            - For each voxel in the input list:
                - Extracts the relevant data slice and initial parameters.
                - Sets the initial guess for the refit, using current fitted values as
                  a starting point to ensure stability.
                - Updates the selected metabolite's frequency to the new user-defined
                  value.
                - Sets up fitting bounds according to the selected mode (Individual or
                  Sum).
                - Calls the `_fit_single_spectrum_pseudo_inv` function to perform the
                  actual fitting.
                - Updates the main `fit_results` xarray.Dataset in-place with the new
                  amplitudes, frequencies, and T2* values.
            - After refitting all specified voxels, it updates the plotter's internal
              cached dataset to ensure the visualization reflects the changes.
            - Triggers a plot refresh and updates the Sum of Squared Errors (SSE)
              display.
            Parameters
            ----------
            list_of_indices : list of dict
                A list containing the coordinates of the voxels to be refit. Each
                element of the list is a dictionary where keys are dimension names
                (e.g., 'x', 'y') and values are the integer indices for that dimension.
                Example: `[{'x': 10, 'y': 15}, {'x': 11, 'y': 15}]`.
            Notes
            -----
            This is an inner function that relies on variables from its enclosing
            scope, including `plot_handles`, `fit_params_new`, `input_data`,
            `fit_results`, `debug_log`, and various widget objects. It is not
            designed to be called directly in isolation. The function has significant
            side effects, as it modifies the `fit_results` dataset in-place and
            interacts with a plotting backend.
            """

            widget_values = plot_handles["widgets_dict"]

            def _extract_scalar(value):
                arr = np.asarray(value)
                return arr.item() if arr.ndim == 0 else arr.squeeze()

            # CRITICAL UPDATE: Use FitSession to build current fit parameters including any added metabolites
            current_fit_params = fit_session.build_fit_params()
            current_fit_results = fit_session.build_fit_results()

            global_params = current_fit_params.attrs.copy()
            from ..analysis.fitting.spectroscopic_fitting import (
                _fit_single_spectrum_pseudo_inv,
            )

            debug_log_base_spec(
                f"Using updated fit params with {len(fit_session.get_active_metabolite_names())} metabolites for refit"
            )
            metabs = current_fit_params.coords["metabolite"].values.tolist()
            debug_log_base_spec(f"metab = {metabs}")

            # Get common parameters once (these don't depend on voxel)
            # Check if we should use per-metabolite parameters
            use_per_metab_params = metabs and len(metabs) > 0

            if use_per_metab_params:
                # Use per-metabolite parameters from the table widgets
                pass
            else:
                # Fall back to global parameters
                fit_width_hz = fit_range_hz_float.value

            # Check if overlay display mode is set to Sum or Individual Peaks
            apply_sum_widget = widget_values.get("overlay_display_mode")
            if apply_sum_widget is None:
                # Fallback for prefixed overlay widget used when overlay_data is the second dataset
                apply_sum_widget = widget_values.get("ds1_overlay_display_mode")

            if apply_sum_widget is None:
                print("Error: No overlay display mode selected.")
                return
            else:
                # Handle both widget objects and plain values
                apply_sum = (
                    apply_sum_widget.value
                    if hasattr(apply_sum_widget, "value")
                    else apply_sum_widget
                )

            # Print mode to user for immediate feedback
            mode_text = (
                "SUM MODE (all metabolites will be refit)"
                if apply_sum
                else "INDIVIDUAL MODE (only selected metabolite will be refit)"
            )

            # Get the acquisition index / Frequency (clicked position)
            acquisition_idx = widget_values.get("acquisition_idx", None)
            if acquisition_idx is None:
                return
            else:
                # Handle both widget objects and plain values
                acquisition_idx = (
                    acquisition_idx.value
                    if hasattr(acquisition_idx, "value")
                    else acquisition_idx
                )

            # Get the selected metabolite that should be refitted
            overlay_selected_metabolites = widget_values.get(
                "overlay_selected_metabolites", None
            )
            if overlay_selected_metabolites is None:
                # Smart fallback logic to find the metabolite selection widget
                debug_log_base_spec(
                    message="Metabolite widget 'overlay_selected_metabolites' not found. Starting fallback search.",
                )
                metabolite_widget_key = None
                best_candidate_key = None

                for w_name, w_obj in widget_values.items():
                    if hasattr(w_obj, "options") and hasattr(w_obj, "value"):
                        # Priority 1: Widget name contains "metabolite"
                        if "metabolite" in w_name.lower():
                            metabolite_widget_key = w_name
                            debug_log_base_spec(
                                message=f"FALLBACK SUCCESS (Priority 1): Found widget '{w_name}' by name.",
                            )
                            break

                        # Priority 2: Widget options look like metabolite names
                        options = getattr(w_obj, "options", [])
                        if len(options) > 0 and isinstance(options[0], str):
                            # List of common non-metabolite options in dropdowns
                            non_metabolite_opts = {
                                "Magnitude",
                                "Real",
                                "Imaginary",
                                "Phase",
                                "Percentage",
                                "Min/Max",
                                "fit_amps",
                                "fit_freqs",
                                "fit_t2s",
                                "fit_r2_freq",
                                "fit_r2_t2",
                                "fit_fwhm_Hz",
                                "bone",
                                "gray",
                                "plasma",
                                "inferno",
                                "magma",
                                "cividis",
                                "viridis",
                                "jet",
                                "hsv",
                                "cool",
                                "seismic",
                                "twilight",
                                "tab10",
                                "prism",
                            }
                            # If any option is not in the common list, it's likely a metabolite list
                            if any(opt not in non_metabolite_opts for opt in options):
                                best_candidate_key = w_name
                                debug_log_base_spec(
                                    message=f"FALLBACK CANDIDATE (Priority 2): Found widget '{w_name}' by its options.",
                                )

                # Use the best match found
                final_widget_key = metabolite_widget_key or best_candidate_key

                if final_widget_key:
                    widget = widget_values[final_widget_key]
                    overlay_selected_metabolites = (
                        widget.value if hasattr(widget, "value") else widget
                    )
                    debug_log_base_spec(
                        message=f"FALLBACK: Using widget '{final_widget_key}'. Selected metabolite: {overlay_selected_metabolites}",
                    )

                if overlay_selected_metabolites is None:
                    print("Error: No metabolite selected for refit.")
                    return
            else:
                # Handle both widget objects and plain values
                overlay_selected_metabolites = (
                    overlay_selected_metabolites.value
                    if hasattr(overlay_selected_metabolites, "value")
                    else overlay_selected_metabolites
                )

            # Get current metabolite list from FitSession (includes newly added metabolites)
            metabs = fit_session.get_active_metabolite_names()
            added_metabolite_names = [m["name"] for m in fit_session.added_metabolites]

            # Convert to metabolite index using the current metabolite list
            overlay_selected_metabolites_ind = metabs.index(
                overlay_selected_metabolites
            )

            # Get the NEW frequency from clicked position -> Will be set as initial freq for refit
            # further down we will update only the selected metabolite's freq
            freq_axis_hz = resolve_frequency_axis(
                getattr(fit_params, "attrs", None), "Hz"
            )
            if freq_axis_hz is None and "freq_axis_hz" in fit_params:
                freq_axis_hz = fit_params["freq_axis_hz"].values
            if freq_axis_hz is None:
                raise KeyError(
                    "Could not determine frequency axis (Hz) from fit parameters."
                )
            new_freq = freq_axis_hz[acquisition_idx]
            fitting_progress.max = len(list_of_indices)
            num_voxels = len(list_of_indices)
            for i, voxel_indices in enumerate(list_of_indices):
                fitting_progress.value = i + 1
                fitting_info_label.value = (
                    f"Refitting voxel {i+1}/{len(list_of_indices)}"
                )

                # Update progress bar color from rainbow colormap
                if num_voxels > 1:
                    progress_fraction = i / (num_voxels - 1)
                else:
                    progress_fraction = 0.5  # Mid-point color for a single voxel

                rainbow = get_cmap("rainbow")
                color_rgba = rainbow(progress_fraction)
                fitting_progress.style.bar_color = mplto_hex(color_rgba)

                # if not voxel_indices:

                full_selector = voxel_indices.copy()

                # Get fresh widget values to ensure we have the most current coordinates
                for dim, widget in widget_values.items():
                    if (
                        dim in input_data.dims
                        and dim not in full_selector
                        and hasattr(widget, "value")
                    ):
                        full_selector[dim] = widget.value

                full_selector.pop("acquisition_idx", None)  # Remove if present
                data_slice = input_data.isel(full_selector).values
                param_selector = {
                    k: v
                    for k, v in full_selector.items()
                    if k in current_fit_params.dims
                }

                # Build parameter arrays more efficiently using list comprehensions
                initial_freqs_Hz_slice = np.array(
                    [metab_param_widgets["init_freq"][metab].value for metab in metabs]
                )
                initial_t2s_ms_slice = np.array(
                    [metab_param_widgets["init_t2"][metab].value for metab in metabs]
                )
                initial_t2s_s_slice = initial_t2s_ms_slice / 1000.0
                min_t2s_ms_slice = np.array(
                    [metab_param_widgets["min_t2"][metab].value for metab in metabs]
                )
                min_t2s_slice = min_t2s_ms_slice / 1000.0
                max_t2s_ms_slice = np.array(
                    [metab_param_widgets["max_t2"][metab].value for metab in metabs]
                )
                max_t2s_slice = max_t2s_ms_slice / 1000.0
                metabs_t2_ranges_slice = max_t2s_slice - min_t2s_slice
                metabs_freq_ranges_slice = np.array(
                    [metab_param_widgets["freq_range"][metab].value for metab in metabs]
                )

                results_selector = {
                    k: v
                    for k, v in full_selector.items()
                    if k in current_fit_results.dims
                }
                target_selector = results_selector if results_selector else {}

                previous_result_cache: dict[str, dict[str, float | complex]] = {}
                pre_fit_amps: list[complex | float] = []
                pre_fit_freqs: list[float] = []
                pre_fit_t2s: list[float] = []

                existing_metab_values = {
                    str(m): m for m in current_fit_results.metabolite.values
                }

                for metab_name in metabs:
                    metab_key = str(metab_name)
                    cache_entry: dict[str, float | complex] = {}

                    if metab_key in existing_metab_values:
                        try:
                            amp_val = (
                                current_fit_results["fit_amps"]
                                .isel(**target_selector)
                                .sel(metabolite=existing_metab_values[metab_key])
                                .values
                            )
                            freq_val = (
                                current_fit_results["fit_freqs"]
                                .isel(**target_selector)
                                .sel(metabolite=existing_metab_values[metab_key])
                                .values
                            )
                            t2_val = (
                                current_fit_results["fit_t2s"]
                                .isel(**target_selector)
                                .sel(metabolite=existing_metab_values[metab_key])
                                .values
                            )
                            fwhm_val = (
                                current_fit_results["fit_fwhm_Hz"]
                                .isel(**target_selector)
                                .sel(metabolite=existing_metab_values[metab_key])
                                .values
                            )
                        except Exception as cache_err:
                            debug_log_base_spec(
                                message=(
                                    f"Warning: Could not cache existing fit results for {metab_key}: {cache_err}"
                                )
                            )
                            amp_scalar = complex(np.nan, np.nan)
                            freq_scalar = float("nan")
                            t2_scalar = float("nan")
                            fwhm_scalar = float("nan")
                        else:
                            amp_scalar = _extract_scalar(amp_val)
                            freq_scalar = float(_extract_scalar(freq_val))
                            t2_scalar = float(_extract_scalar(t2_val))
                            fwhm_scalar = float(_extract_scalar(fwhm_val))

                        cache_entry["amp"] = amp_scalar
                        cache_entry["freq"] = freq_scalar
                        cache_entry["t2"] = t2_scalar
                        cache_entry["fwhm"] = fwhm_scalar
                    else:
                        debug_log_base_spec(
                            message=(
                                f"Warning: Metabolite {metab_key} not found in current_fit_results"
                            )
                        )
                        amp_scalar = complex(np.nan, np.nan)
                        freq_scalar = float("nan")
                        t2_scalar = float("nan")
                        cache_entry["amp"] = amp_scalar
                        cache_entry["freq"] = freq_scalar
                        cache_entry["t2"] = t2_scalar
                        cache_entry["fwhm"] = float("nan")

                    previous_result_cache[metab_key] = cache_entry
                    pre_fit_amps.append(cache_entry.get("amp", complex(np.nan, np.nan)))
                    pre_fit_freqs.append(cache_entry.get("freq", float("nan")))
                    pre_fit_t2s.append(cache_entry.get("t2", float("nan")))

                debug_log_fit_state(
                    stage="before_fit",
                    voxel_coords=full_selector,
                    metabolite_names=metabs,
                    amps=pre_fit_amps,
                    freqs=pre_fit_freqs,
                    t2s=pre_fit_t2s,
                )

                # Log old frequency before updating to new cursor position
                old_freq = initial_freqs_Hz_slice[overlay_selected_metabolites_ind]

                should_update_selected_freq = (
                    overlay_selected_metabolites in added_metabolite_names
                )

                if should_update_selected_freq:
                    # Now update ONLY the selected metabolite's frequency to cursor position
                    initial_freqs_Hz_slice[overlay_selected_metabolites_ind] = new_freq

                    if not apply_sum:
                        debug_log_base_spec(
                            message=(
                                f"Voxel {i+1}: INDIVIDUAL MODE - Updated selected "
                                f"new metabolite freq: {old_freq:.2f} -> {new_freq:.2f} Hz"
                            ),
                        )
                    else:
                        debug_log_base_spec(
                            message=(
                                f"Voxel {i+1}: SUM MODE - Updated selected new metabolite freq: "
                                f"{old_freq:.2f} -> {new_freq:.2f} Hz"
                            ),
                        )
                else:
                    debug_log_base_spec(
                        message=(
                            f"Voxel {i+1}: Skipping acquisition_idx retargeting for existing "
                            f"metabolite '{overlay_selected_metabolites}'"
                        ),
                    )

                # Handle zero amplitude metabolites and skipped fitting
                # Get enable fitting status for each metabolite
                enable_fitting_status = []
                if "enable_fitting" in metab_param_widgets:
                    enable_fitting_status = [
                        metab_param_widgets["enable_fitting"][metab].value
                        for metab in fit_session.get_active_metabolite_names()
                    ]
                else:
                    # Default to enabled for all metabolites if widgets not available
                    enable_fitting_status = [True] * len(initial_freqs_Hz_slice)

                skip_mask = np.array(
                    [not status for status in enable_fitting_status], dtype=bool
                )

                selected_metab_name = overlay_selected_metabolites
                locked_params: dict[str, bool] = {}
                for metab_idx, metab_name in enumerate(metabs):
                    if skip_mask[metab_idx]:
                        continue  # honour disabled metabolites fully
                    if not apply_sum and metab_name != selected_metab_name:
                        locked_params[str(metab_name)] = True
                    # In sum mode the user-configured skip mask is respected as-is

                skip_lookup = {
                    metab_name: bool(skip_flag)
                    for metab_name, skip_flag in zip(metabs, skip_mask)
                }

                all_metabs_skipped = skip_mask.size > 0 and bool(np.all(skip_mask))
                if all_metabs_skipped:
                    debug_log_base_spec(
                        message=(
                            "All metabolites marked as 'Enable Fitting = False'; "
                            "skipping optimisation for current voxel"
                        )
                    )

                # Calculate bounds using either per-metabolite or global parameters
                if use_per_metab_params:
                    debug_log_base_spec(f"using per-metabolite fitting bounds")
                    # Use per-metabolite frequency ranges
                    freq_lower_bound = np.array(
                        [
                            initial_freqs_Hz_slice[metab_idx]
                            - metab_param_widgets["freq_range"][metab].value / 2.0
                            for metab_idx, metab in enumerate(metabs)
                        ]
                    )
                    freq_upper_bound = np.array(
                        [
                            initial_freqs_Hz_slice[metab_idx]
                            + metab_param_widgets["freq_range"][metab].value / 2.0
                            for metab_idx, metab in enumerate(metabs)
                        ]
                    )

                    # Use per-metabolite T2* min/max bounds
                    t2_min_vals = np.array(
                        [
                            max(
                                0.001,
                                metab_param_widgets["min_t2"][metab].value / 1000.0,
                            )
                            for metab in metabs
                        ]
                    )
                    t2_max_vals = np.array(
                        [
                            metab_param_widgets["max_t2"][metab].value / 1000.0
                            for metab in metabs
                        ]
                    )
                else:
                    # Fall back to global bounds
                    freq_lower_bound = initial_freqs_Hz_slice - fit_width_hz / 2.0
                    freq_upper_bound = initial_freqs_Hz_slice + fit_width_hz / 2.0
                    t2_min_vals = np.full_like(initial_t2s_s_slice, t2_min_float.value)
                    t2_max_vals = np.full_like(initial_t2s_s_slice, t2_max_float.value)

                # Apply fitting constraints based on mode
                if not apply_sum:
                    # Individual mode: tighten constraints for non-selected metabolites
                    freq_tolerance = 0.001
                    t2_tolerance = 0.00001

                    for metab_idx in range(len(freq_lower_bound)):
                        if metab_idx != overlay_selected_metabolites_ind:
                            freq_lower_bound[metab_idx] = (
                                initial_freqs_Hz_slice[metab_idx] - freq_tolerance
                            )
                            freq_upper_bound[metab_idx] = (
                                initial_freqs_Hz_slice[metab_idx] + freq_tolerance
                            )
                            t2_min_vals[metab_idx] = max(
                                0.001, initial_t2s_s_slice[metab_idx] - t2_tolerance
                            )
                            t2_max_vals[metab_idx] = (
                                initial_t2s_s_slice[metab_idx] + t2_tolerance
                            )
                else:
                    # Sum mode: all metabolites can vary freely
                    debug_log_base_spec(
                        f"Voxel {i+1}: SUM MODE - All metabolites can vary freely from their current fitted positions"
                    )
                    for metab_idx, metab_name in enumerate(
                        fit_results.metabolite.values
                    ):
                        debug_log_base_spec(
                            f"Voxel {i+1}: --> {metab_name}: fit freqs in fit_results: "
                            f"{fit_results['fit_freqs'].isel(**full_selector).sel(metabolite=metab_name).squeeze().values}"
                        )
                debug_log_base_spec(
                    message=f"""initial_freqs_Hz_slice: {initial_freqs_Hz_slice}
                                    initial_t2s_s_slice: {initial_t2s_s_slice}\n
                                    metabs_freq_ranges_slice: {metabs_freq_ranges_slice}\n
                                    metabs_t2_ranges_slice: {metabs_t2_ranges_slice}\n
                                    freq_lower_bound: {freq_lower_bound}\n
                                    freq_upper_bound: {freq_upper_bound}\n
                                    t2_min_vals: {t2_min_vals}\n
                                    t2_max_vals: {t2_max_vals}\n
                                    """
                )

                if all_metabs_skipped:
                    new_amps = np.zeros_like(initial_freqs_Hz_slice, dtype=complex)
                    new_freqs = initial_freqs_Hz_slice.copy()
                    new_t2s = initial_t2s_s_slice.copy()
                    combined_r2_freq = None
                    combined_r2_t2 = None
                else:
                    (
                        new_amps,
                        new_freqs,
                        new_t2s,
                        combined_r2_freq,
                        combined_r2_t2,
                        low_snr_refitted,
                    ) = _fit_single_spectrum_pseudo_inv(
                        data_slice=data_slice.squeeze(),
                        initial_freqs_Hz=initial_freqs_Hz_slice,
                        initial_t2s_s=initial_t2s_s_slice,
                        metabs_freq_ranges=metabs_freq_ranges_slice,
                        metabs_t2_ranges=metabs_t2_ranges_slice,
                        min_t2_s=t2_min_vals,
                        max_t2_s=t2_max_vals,
                        freq_range_lower_bound_Hz=freq_lower_bound,
                        freq_range_upper_bound_Hz=freq_upper_bound,
                        skip_fitting_mask=skip_mask,
                        **global_params,
                    )

                debug_log_base_spec(
                    f"current_fit_results.metabolite.values={current_fit_results.metabolite.values}"
                )
                # Store the new fit results in both current_fit_results and original fit_results
                post_fit_amps: list[complex | float] = []
                post_fit_freqs: list[float] = []
                post_fit_t2s: list[float] = []

                for metab_idx, metab_name in enumerate(metabs):
                    metab_key = str(metab_name)
                    cached = previous_result_cache.get(metab_key, {})
                    skip_current_metab = bool(
                        skip_lookup.get(metab_name, False)
                        or skip_lookup.get(metab_key, False)
                    )

                    if skip_current_metab:
                        amp_value = 0.0 + 0.0j
                        freq_value = cached.get(
                            "freq", float(initial_freqs_Hz_slice[metab_idx])
                        )
                        t2_value = cached.get(
                            "t2", float(initial_t2s_s_slice[metab_idx])
                        )
                        fwhm_value = cached.get("fwhm", np.nan)
                    else:
                        amp_value = new_amps[metab_idx]
                        freq_value = new_freqs[metab_idx]
                        t2_value = new_t2s[metab_idx]
                        fwhm_value = cached.get("fwhm", np.nan)

                        if metab_key in locked_params:
                            # Individual mode: preserve previous fit for non-selected metabolites
                            amp_value = cached.get("amp", amp_value)
                            freq_value = cached.get("freq", freq_value)
                            t2_value = cached.get("t2", t2_value)
                        else:
                            if np.isfinite(t2_value) and t2_value != 0:
                                fwhm_value = 1 / (np.pi * t2_value)
                            else:
                                fwhm_value = cached.get("fwhm", np.nan)

                    amp_log = _extract_scalar(amp_value)
                    freq_log = _extract_scalar(freq_value)
                    t2_log = _extract_scalar(t2_value)

                    post_fit_amps.append(amp_log)
                    post_fit_freqs.append(freq_log)
                    post_fit_t2s.append(t2_log)

                    debug_log_base_spec(
                        f"Voxel {i+1}: Updating metabolite: {metab_key}, idx {metab_idx}: Amp: {amp_value}, Freq: {freq_value} Hz, T2*: {t2_value} s (skip={skip_current_metab})"
                    )

                    metab_coord = existing_metab_values.get(metab_key, metab_name)

                    try:
                        current_fit_results["fit_amps"].isel(**full_selector).sel(
                            metabolite=metab_coord
                        )[:] = amp_value
                        current_fit_results["fit_freqs"].isel(**full_selector).sel(
                            metabolite=metab_coord
                        )[:] = freq_value
                        current_fit_results["fit_t2s"].isel(**full_selector).sel(
                            metabolite=metab_coord
                        )[:] = t2_value
                        current_fit_results["fit_fwhm_Hz"].isel(**full_selector).sel(
                            metabolite=metab_coord
                        )[:] = fwhm_value
                    except Exception as update_err:
                        debug_log_base_spec(
                            message=(
                                f"Warning: Could not update current_fit_results for {metab_coord}: {update_err}"
                            )
                        )

                    if metab_coord in fit_results.metabolite.values:
                        try:
                            fit_results["fit_amps"].isel(**full_selector).sel(
                                metabolite=metab_coord
                            )[:] = amp_value
                            fit_results["fit_freqs"].isel(**full_selector).sel(
                                metabolite=metab_coord
                            )[:] = freq_value
                            fit_results["fit_t2s"].isel(**full_selector).sel(
                                metabolite=metab_coord
                            )[:] = t2_value
                            fit_results["fit_fwhm_Hz"].isel(**full_selector).sel(
                                metabolite=metab_coord
                            )[:] = fwhm_value
                        except Exception as legacy_update_err:
                            debug_log_base_spec(
                                message=(
                                    f"Warning: Could not update legacy fit_results for {metab_coord}: {legacy_update_err}"
                                )
                            )

                debug_log_fit_state(
                    stage="after_fit",
                    voxel_coords=full_selector,
                    metabolite_names=metabs,
                    amps=post_fit_amps,
                    freqs=post_fit_freqs,
                    t2s=post_fit_t2s,
                )

                # Update the fit_session's base results to persist changes
                # CRITICAL FIX: Use commit_results() to properly clear staging area
                # This prevents metabolite duplication on subsequent refits
                fit_session.commit_results(current_fit_results)

            fitting_progress.style.bar_color = "green"
            fitting_info_label.value = "Refitting complete."

            # CRITICAL FIX: Update the plotter's cached data to reflect fitting changes
            # Use the plotter's sync method to update cached datasets with fitted results
            try:
                self.plotter.sync_with_fit_session(fit_session)
                debug_log_base_spec(
                    "Successfully synchronized plotter with fitted results"
                )
            except Exception as e:
                debug_log_base_spec(f"Error synchronizing plotter after fitting: {e}")

            # Update metabolite parameter widgets to reflect new fitted values
            update_metab_param_widgets(update_ranges=False)

            plot_handles["run_update"](
                **{k: v.value for k, v in widget_values.items() if hasattr(v, "value")}
            )

            # Update SSE display after refit
            update_sse_display()

            # Update widget enable/disable state
            update_widget_enable_state()

        def _initiate_refit(b):
            if b == refit_button:
                # ENHANCED: Force a small delay to ensure widget values are fully updated after click
                import time

                time.sleep(0.1)  # 100ms delay to ensure UI state synchronization

                # Get current voxel coordinates directly from widget values instead of relying on current_state
                current_voxel_coords = {}
                widget_values = plot_handles["widgets_dict"]

                # Extract spatial coordinates from current widget values
                for dim_name in [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                ]:
                    if dim_name in widget_values:
                        widget = widget_values[dim_name]
                        if hasattr(widget, "value"):
                            current_voxel_coords[dim_name] = widget.value

                if not current_voxel_coords:
                    if "repetition_idx" in input_data.dims:
                        current_voxel_coords["repetition_idx"] = 0
                    else:
                        return None

                # Check if we need to iterate over repetition range
                voxels_to_refit = []
                if (
                    repetition_range_slider is not None
                    and "repetition_idx" in current_voxel_coords
                ):
                    rep_min, rep_max = repetition_range_slider.value
                    # Create a voxel coordinate dict for each repetition in range
                    for rep_idx in range(rep_min, rep_max + 1):
                        voxel_coords = current_voxel_coords.copy()
                        voxel_coords["repetition_idx"] = rep_idx
                        voxels_to_refit.append(voxel_coords)
                else:
                    voxels_to_refit = [current_voxel_coords]

                _refit_logic(voxels_to_refit)
            elif b == refit_masked_button:
                # Comprehensive validation for masked refit
                try:
                    # Validate mask state exists
                    if not mask_state or "masked_indices" not in mask_state:
                        debug_log_base_spec(
                            message="Masked refit aborted: mask_state missing indices",
                            file_name="/tmp/masking_debug.log",
                        )

                        return

                    # Validate mask indices exist
                    if not mask_state["masked_indices"]:
                        debug_log_base_spec(
                            message="Masked refit aborted: no voxels stored in mask",
                            file_name="/tmp/masking_debug.log",
                        )
                        return

                    # Validate mask indices are still valid
                    valid_masked_indices = []
                    for idx, voxel_coords in enumerate(mask_state["masked_indices"]):
                        if not isinstance(voxel_coords, dict):
                            continue

                        # Validate coordinate values are within data bounds
                        valid_coords = True
                        for dim_name, coord_value in voxel_coords.items():
                            if dim_name not in input_data.dims:
                                valid_coords = False
                                break

                            if not isinstance(coord_value, (int, np.integer)):
                                valid_coords = False
                                break

                            max_index = input_data.sizes[dim_name] - 1
                            if not (0 <= coord_value <= max_index):
                                valid_coords = False
                                break

                        if valid_coords:
                            valid_masked_indices.append(voxel_coords)
                        else:
                            debug_log_base_spec(
                                message=(
                                    "Skipping invalid voxel from mask: "
                                    f"{voxel_coords}"
                                ),
                                file_name="/tmp/masking_debug.log",
                            )

                    if not valid_masked_indices:
                        debug_log_base_spec(
                            message="Masked refit aborted: no valid voxels after validation",
                            file_name="/tmp/masking_debug.log",
                        )
                        return

                    # Update mask state with only valid indices
                    if len(valid_masked_indices) != len(mask_state["masked_indices"]):
                        mask_state["masked_indices"] = valid_masked_indices

                    # Expand masked indices with repetition range if applicable
                    voxels_to_refit = []
                    if (
                        repetition_range_slider is not None
                        and "repetition_idx" in input_data.dims
                    ):
                        rep_min, rep_max = repetition_range_slider.value
                        # For each masked voxel, create entries for each repetition in range
                        for voxel_coords in valid_masked_indices:
                            for rep_idx in range(rep_min, rep_max + 1):
                                expanded_coords = voxel_coords.copy()
                                expanded_coords["repetition_idx"] = rep_idx
                                voxels_to_refit.append(expanded_coords)

                        debug_log_base_spec(
                            message=(
                                f"Masked refit proceeding with {len(valid_masked_indices)} voxels "
                                f"across {rep_max - rep_min + 1} repetitions "
                                f"(total: {len(voxels_to_refit)} fits)"
                            ),
                            file_name="/tmp/masking_debug.log",
                        )
                    else:
                        voxels_to_refit = valid_masked_indices
                        debug_log_base_spec(
                            message=(
                                "Masked refit proceeding with "
                                f"{len(valid_masked_indices)} voxels"
                            ),
                            file_name="/tmp/masking_debug.log",
                        )

                    _refit_logic(voxels_to_refit)

                except Exception as e:
                    debug_log_base_spec(
                        message=f"Error in masked refit: {str(e)}",
                        file_name="/tmp/masking_debug.log",
                    )

        def _finish_review(b):
            try:
                final_fit_params = fit_session.build_fit_params().copy()
                final_fit_results = fit_session.build_fit_results().copy()
                final_fit_results.attrs = final_fit_results.attrs.copy()
                final_fit_results.attrs["metabs"] = (
                    fit_session.get_active_metabolite_names()
                )

                self.fit_quality_review = fit_quality_da.copy()
                self.fit_params_review = final_fit_params
                self.fit_results_review = final_fit_results
                print("Review finished. Results stored in object attributes.")
            except Exception as finish_err:
                debug_log_base_spec(
                    message=f"Error while finalizing review: {finish_err}"
                )
                print(f"Error finalizing review: {finish_err}")

        # --- Attach Callbacks ---
        mark_good_button.on_click(lambda b: _mark_fit_quality(True))
        mark_bad_button.on_click(lambda b: _mark_fit_quality(False))
        finish_button.on_click(_finish_review)
        refit_button.on_click(_initiate_refit)
        refit_masked_button.on_click(_initiate_refit)
        clear_mask_button.on_click(_clear_mask)
        draw_mask_button.observe(on_draw_mask_toggle, names="value")

        run_interactive_update(**initial_kwargs)

        # FINAL TEST: Log completion
        debug_log_base_spec(
            message="review_fit function completed - debug logging test finished",
        )

    def filter_signal(
        self,
        input_data=None,
        filter_type="None",
        decay_constant=0.01,
        alpha=0.54,
        center_position=None,
        signal_end=None,
        apply_flipangle_correction=False,
        flipangle_deg=None,
        filter_axis=0,
        **kwargs,
    ):
        """
        Apply filtering to raw FID data before k-space reconstruction.

        This function should be applied to raw data before calling reconstruct() for optimal results.
        The filtering is applied to the 1D FID data, which is then sorted into k-space during reconstruction.

        Parameters
        ----------
        input_data : numpy.ndarray, optional
            The raw FID data array. If not provided, uses data from handler.
        filter_type : str, optional
            Type of filter to apply. Options are "None", "Exponential", "Hamming", "Hann".
            Default is "None".
        decay_constant : float, optional
            Decay constant for the exponential filter. Default is 0.01.
        alpha : float, optional
            Alpha parameter for the Hamming filter. Default is 0.54.
        center_position : int, optional
            Center position for Hamming and Hann filters. If None, uses middle of signal.
        signal_end : int, optional
            End index of the significant signal segment. Used if filter_type is "None".
        apply_flipangle_correction : bool, optional
            Whether to apply flipangle correction for hyperpolarized experiments. Default is False.
        flipangle_deg : float, optional
            Flip angle in degrees for correction. Required if apply_flipangle_correction is True.
        filter_axis : int, optional
            Axis along which to apply the filter. Default is 0 (acquisition/time dimension).

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            If preview_reconstruction is False: Returns filtered raw FID data.
            If preview_reconstruction is True: Returns reconstructed spatial-spectroscopic data for preview.

        Notes
        -----
        For typical workflow:
        1. Use plot_signal_decay() to determine optimal filter parameters
        2. Call filter_signal() with selected parameters on raw data
        3. Use the filtered data with reconstruct() for final results
        """
        # Get raw data from handler if not provided
        if input_data is None:
            if (
                hasattr(self.handler, "rawdatajob0")
                and self.handler.rawdatajob0 is not None
            ):
                input_data = self.handler.rawdatajob0.copy()
                debug_log_base_spec(message="Using rawdatajob0 from handler")
            elif hasattr(self.handler, "fid") and self.handler.fid is not None:
                input_data = self.handler.fid.copy()
                debug_log_base_spec(message="Using fid from handler")
            else:
                raise ValueError(
                    "No raw data available for filtering. Ensure CSI data is loaded properly."
                )

        # Reshape input_data to proper format for processing
        # Raw data should be 1D, reshape to (n_acquisition, n_spatial_points)
        if input_data.ndim == 1:
            n_total_points = len(input_data)
            n_acquisition = self.n_acquisition
            n_spatial_points = n_total_points // n_acquisition
            input_data = np.reshape(input_data, (n_spatial_points, n_acquisition)).T
        elif input_data.ndim == 2:
            n_acquisition = input_data.shape[0]
            n_spatial_points = input_data.shape[1]
        else:
            raise ValueError(f"Unexpected input data shape: {input_data.shape}")

        input_data_filtered = input_data.copy()

        # Apply flipangle correction if requested (for hyperpolarized experiments)
        if apply_flipangle_correction:
            if flipangle_deg is None:
                flipangle_deg = getattr(self, "flipangle_deg", 90.0)

            from hypermri.utils.physics import hyperpolarized_Mz_flipangle

            # Create pulse array for the number of acquisitions
            n_pulses = input_data_filtered.shape[filter_axis]
            pulses = np.arange(n_pulses)
            p = hyperpolarized_Mz_flipangle(
                M0_au=1,
                pulses=pulses,
                flipangle_deg=flipangle_deg,
            )

            # Reshape correction factor for broadcasting
            if filter_axis == 0:
                correction = p[:, np.newaxis]
            else:
                correction = p[np.newaxis, :]

            input_data_filtered = input_data_filtered / correction

        # Apply the selected filter type
        N = input_data_filtered.shape[filter_axis]
        valid_filter_types = ["None", "Exponential", "Hann", "Hamming"]
        if filter_type not in valid_filter_types:
            debug_log_base_spec(
                message=f"filter_type is {filter_type}, must be in {valid_filter_types}",
            )
            return input_data_filtered.flatten()

        if filter_type == "Exponential":
            decay = np.exp(-decay_constant * np.arange(N))
            if filter_axis == 0:
                input_data_filtered = input_data_filtered * decay[:, np.newaxis]
            else:
                input_data_filtered = input_data_filtered * decay[np.newaxis, :]
        elif filter_type in ["Hamming", "Hann"]:
            n_array = np.arange(N)
            if center_position is None:
                center_position = N // 2

            n_shifted = n_array - center_position + N // 2
            n_shifted = np.clip(n_shifted, 0, N - 1)

            if filter_type == "Hamming":
                beta = 1 - alpha
                decay = alpha - beta * np.cos(2 * np.pi * n_shifted / (N - 1))
            elif filter_type == "Hann":
                decay = 0.5 * (1 - np.cos(2 * np.pi * n_shifted / (N - 1)))

            if filter_axis == 0:
                input_data_filtered = input_data_filtered * decay[:, np.newaxis]
            else:
                input_data_filtered = input_data_filtered * decay[np.newaxis, :]
        else:  # filter_type == "None"
            # Simple truncation filter - zero out data after signal_end
            if signal_end is not None and signal_end < N:
                if filter_axis == 0:
                    input_data_filtered[signal_end:, :] = 0 + 1j * 0
                else:
                    input_data_filtered[:, signal_end:] = 0 + 1j * 0

        # Flatten back to 1D for use in reconstruction
        filtered_data_1d = input_data_filtered.T.flatten()
        return filtered_data_1d

    def plot(self, *args, **kwargs):
        """
        Forwards the plotting call to the dedicated SpectroscopyPlotter instance.
        """
        return self.plotter.plot(*args, **kwargs)
