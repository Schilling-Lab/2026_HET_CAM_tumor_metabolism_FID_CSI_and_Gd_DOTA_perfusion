import cmcrameri  # Needs to be imported once to register the colormaps with Matplotlib
import numpy as np
import xarray as xr

from hypermri.masking import MaskApplicator, MaskCreator
from hypermri.utils.system.logging import debug_log, init_default_logger

from ..core import MRIExperiment

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


DEBUG_LOG = True  # Debug message flag
DEBUG_FILE = "/tmp/create_mask_debug.log"


def debug_log_create_mask(message, file_name=DEBUG_FILE, enabled=DEBUG_LOG):
    """Log create_mask debug messages to /tmp/create_mask_debug.log with timestamp."""
    debug_log(message=message, file_name=file_name, write_to_file=enabled)


class BaseAnatomy(MRIExperiment):
    """Experimental base class for anatomy classes."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract MRIExperiment-specific kwargs and filter out others
        requested_load_data = kwargs.pop("load_data", False)
        log_mode = kwargs.pop("log_mode", "error")
        # Remove any sequence-specific kwargs that MRIExperiment doesn't understand
        kwargs.pop("reco_options", None)

        # Avoid re-triggering heavy data loading when study-level minimal/lazy modes pass strings
        load_data = (
            requested_load_data if isinstance(requested_load_data, bool) else False
        )

        # Call the parent (MRIExperiment) constructor with the path from the loaded object.
        # This ensures MRIExperiment's internal setup is done, but the data is already
        # loaded.
        super().__init__(str(mri_exp_obj.path), load_data=load_data, log_mode=log_mode)

        # Preserve the original handler if it has data loaded
        if hasattr(mri_exp_obj, "handler") and hasattr(mri_exp_obj.handler, "fid"):
            logger.debug("BaseAnatomy: preserving loaded handler from mri_exp_obj")
            self.handler = mri_exp_obj.handler
        elif hasattr(mri_exp_obj, "handler"):
            logger.debug("BaseAnatomy: handler exists but no fid, ensuring data load")
            self.handler = mri_exp_obj.handler
            # Ensure fid data is loaded if the handler exists but fid is missing
            if not hasattr(self.handler, "fid") or self.handler.fid is None:
                try:
                    self.handler.load_data(source=str(mri_exp_obj.path))
                    logger.debug(
                        f"BaseAnatomy: loaded data, handler has fid: {hasattr(self.handler, 'fid')}"
                    )
                except Exception as e:
                    logger.warning(f"BaseAnatomy: Failed to load data: {e}")

        # Copy data from the loaded MRIExperiment object using standardized access
        if (
            hasattr(mri_exp_obj, "primary_data")
            and mri_exp_obj.primary_data is not None
        ):
            self.data_array = mri_exp_obj.primary_data
        elif hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array

        # Mirror handler parameters so metadata (e.g. center_ppm) stays available on this instance
        if hasattr(self.handler, "get_parameters"):
            try:
                handler_params = self.handler.get_parameters()
                if handler_params:
                    self.set_parameters(handler_params, allow_extra=True)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(f"BaseAnatomy: parameter synchronisation skipped: {exc}")

        # Initialize plotter using registry system
        try:
            from ..visualization.plotter_factory import create_plotter

            self.plotter = create_plotter(
                sequence_obj=self, category="anatomical", plotter_type="anatomy"
            )
        except ImportError:
            # Fallback to direct import if registry system not available
            from ..utils.plot.anatomyPlotter import AnatomyPlotter

            self.plotter = AnatomyPlotter(self)

        # try:
        #     self.extent = get_extent(data_obj=self)
        # except BaseException:
        #     self.extent = None

        # Initialize standardized data arrays (inherited from StandardizedMRIMixin via MRIExperiment)
        # self.data_array is already initialized via mixin (kspace_array no
        # longer stored)

        self.clear_lasso_patch_ref = []

    def _load_data_array(self):
        """Lazy loading implementation for scanner reconstructed data (seq2d)."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="seq2d")
            except Exception as e:
                logger.warning(f"Failed to lazy load data_array: {e}")
        return None

    def _load_reco_data(self):
        """Lazy loading implementation for raw reconstructed data."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="raw")
            except Exception as e:
                logger.warning(f"Failed to lazy load reco_data: {e}")
        return None

    def _force_line_properties(self, selector, color="red", linewidth=4, alpha=1.0):
        """Aggressively enforce line properties on LassoSelector for visibility."""
        debug_log_create_mask(
            f"[LINE] Forcing line properties: color={color}, width={linewidth}, alpha={alpha}"
        )

        if hasattr(selector, "_selection_artist") and selector._selection_artist:
            selector._selection_artist.set_color(color)
            selector._selection_artist.set_linestyle("-")
            selector._selection_artist.set_linewidth(linewidth)
            selector._selection_artist.set_alpha(alpha)
            debug_log_create_mask(f"[LINE] Set _selection_artist properties")

        if hasattr(selector, "artists"):
            for i, artist in enumerate(selector.artists):
                if hasattr(artist, "set_color"):
                    artist.set_color(color)
                    artist.set_linestyle("-")
                    artist.set_linewidth(linewidth)
                    artist.set_alpha(alpha)
                    debug_log_create_mask(f"[LINE] Set artist {i} properties")

        return True

    def get_masks_as_dataset(self):
        """
        Convert the current masks to an xarray Dataset for better organization.

        Returns
        -------
        xarray.Dataset
            Dataset containing all finalized masks as DataArrays with metadata.
        """
        # Prefer finalized masks from the simplified mask creator
        finalized = self.get_finalized_masks()

        if not finalized:
            print("No masks available. Use create_mask() first.")
            return None

        # Create Dataset from finalized masks only
        mask_dict = {}
        for mask_name, mask_data in finalized.items():
            # mask_data can be either an xr.DataArray or a MaskContainer
            if hasattr(mask_data, "values") and isinstance(mask_data, xr.DataArray):
                dataarr = mask_data
            else:
                # Try to handle MaskContainer objects (which expose .buffer)
                try:
                    dataarr = getattr(mask_data, "buffer", None)
                except Exception:
                    dataarr = None

            if isinstance(dataarr, xr.DataArray) and dataarr.attrs.get(
                "finalized", False
            ):
                # Clean the mask name for valid variable name
                clean_name = mask_name.replace(" ", "_").replace("-", "_")
                mask_dict[clean_name] = dataarr

        if mask_dict:
            masks_dataset = xr.Dataset(mask_dict)
            # Add global attributes
            masks_dataset.attrs["creation_method"] = "create_mask"
            masks_dataset.attrs["total_masks"] = len(mask_dict)
            masks_dataset.attrs["mask_names"] = list(mask_dict.keys())
            return masks_dataset
        else:
            print("No finalized masks found.")
            return None

    @property
    def masks_dataset(self):
        """
        Property to access masks as xarray Dataset.

        Returns
        -------
        xarray.Dataset or dict
            If finalized masks exist, returns them as a Dataset.
            Otherwise returns the raw masks dictionary.
        """
        dataset = self.get_masks_as_dataset()
        if dataset is not None:
            return dataset
        else:
            # Fallback to internal masks dictionary if no finalized masks
            return getattr(self, "masks", {})

    def plot(self, *args, **kwargs):
        """
        Forwards the plotting call to the dedicated AnatomyPlotter instance.
        """
        return self.plotter.plot(*args, **kwargs)

    def apply_mask_to_datasets(
        self,
        mask_name,
        target_experiments,
        alignment_method="scanner_coords",
        return_dict=False,
        preserve_dims=None,
        select_dims=None,
        **kwargs,
    ):
        """Apply a finalized mask to other datasets with coordinate alignment.


        Parameters
        ----------
        mask_name : str
            Name of the mask to apply (must exist in self.masks)
        target_experiments : list
            List of MRIExperiment objects or xr.DataArrays to apply the mask to.
            Can be experiment objects (e.g., [dce]) or DataArrays directly (e.g., [dce.reco_data])
        alignment_method : str, default "scanner_coords"
            Method for coordinate alignment: "scanner_coords", "object_coords", or "simple"
        return_dict : bool, default False
            If False and only one experiment is provided, return the DataArray directly.
            If True, always return a dict mapping experiment IDs to DataArrays.
        preserve_dims : str or list of str, optional
            Non-spatial dimension name(s) to preserve in the output mask.
            Can be a single string (e.g., 'repetition_idx') or a list of strings
            (e.g., ['repetition_idx', 'channel_idx']). The mask will be broadcast
            to include these dimensions from the target data. Default is None (only spatial dims).
        select_dims : dict, optional
            Dictionary mapping dimension names to indices for selecting specific
            slices before masking (e.g., {'repetition_idx': 0, 'channel_idx': 0}).
            Useful for applying mask to a specific repetition/channel. Default is None.
        **kwargs
            Additional arguments passed to MaskApplicator.apply_to_dataset()

        Returns
        -------
        xr.DataArray or dict
            If single experiment and return_dict=False: applied mask DataArray
            Otherwise: dictionary mapping experiment identifiers to applied mask DataArrays
        """
        # Type checking with clear error message
        # Check if single object was passed instead of list
        if not isinstance(target_experiments, (list, tuple)):
            # Allow single DataArray or experiment object - wrap in list
            target_experiments = [target_experiments]

        # Robust mask selection: prefer self.masks if available, else get_finalized_masks()
        masks = None
        if hasattr(self, "masks") and self.masks:
            masks = self.masks
        elif hasattr(self, "get_finalized_masks"):
            masks = self.get_finalized_masks()
        if not masks:
            raise RuntimeError(
                "No masks available. Neither self.masks nor get_finalized_masks() returned any masks. "
                "Please create and finalize a mask first."
            )
        if mask_name not in masks:
            raise ValueError(
                f"Mask '{mask_name}' not found. Available masks: {list(masks.keys())}"
            )

        # Create mask applicator
        applicator = MaskApplicator(self, masks[mask_name])

        # Apply to all target datasets
        results = applicator.batch_apply(
            target_experiments,
            alignment_method=alignment_method,
            mask_name=f"{mask_name}_applied",
            preserve_dims=preserve_dims,
            select_dims=select_dims,
            **kwargs,
        )

        # Store results for later reference
        if not hasattr(self, "cross_dataset_masks"):
            self.cross_dataset_masks = {}
        self.cross_dataset_masks[mask_name] = results

        # Simplify return value for single experiment
        if len(target_experiments) == 1 and not return_dict:
            # Return the DataArray directly instead of dict
            return list(results.values())[0]

        return results

    def extract_masked_signal(
        self,
        mask_name,
        target_experiments,
        aggregation="mean",
        spatial_dims=None,
        aggregate_extra_dims=None,
        alignment_method="scanner_coords",
        return_dict=False,
        preserve_dims=None,
        select_dims=None,
        **kwargs,
    ):
        """Extract signal from masked regions with optional aggregation.

        This is a high-level convenience function that:
        1. Applies a mask to target datasets (calls apply_mask_to_datasets)
        2. Applies the boolean mask to the actual signal data
        3. Optionally aggregates over spatial dimensions

        Useful for extracting ROI timecourses, computing statistics within masks, etc.

        Parameters
        ----------
        mask_name : str
            Name of the mask to apply (must exist in self.masks)
        target_experiments : list
            List of MRIExperiment objects or xr.DataArrays to extract signal from.
            Can be experiment objects (e.g., [dce]) or DataArrays directly (e.g., [dce.reco_data])
        aggregation : str or None, default 'mean'
            How to aggregate over spatial dimensions:
            - 'mean': Mean signal within mask
            - 'sum': Sum of signal within mask
            - 'median': Median signal within mask
            - 'count': Number of voxels in mask
            - None: Return full masked data (NaN outside mask, no aggregation)
        spatial_dims : list of str, optional
            Spatial dimension names to aggregate over.
            Default: ['read_idx', 'phase_idx', 'slice_idx']
        aggregate_extra_dims : str or list of str, optional
            Non-spatial dimension name(s) to aggregate over in addition to spatial dims
            (e.g., 'channel_idx' or ['channel_idx', 'acquisition_idx']).
            These dimensions will be averaged using the same aggregation method.
            Useful for combining channels or acquisitions while preserving other dimensions
            like repetition_idx. Default: None (no extra aggregation).
        alignment_method : str, default "scanner_coords"
            Method for coordinate alignment: "scanner_coords", "object_coords", or "simple"
        return_dict : bool, default False
            If False and only one experiment is provided, return the DataArray directly.
            If True, always return a dict mapping experiment IDs to DataArrays.
        preserve_dims : str or list of str, optional
            Non-spatial dimension name(s) to preserve (e.g., 'repetition_idx').
            The mask will be broadcast to include these dimensions.
        select_dims : dict, optional
            Dictionary mapping dimension names to indices for selecting specific
            slices before masking (e.g., {'repetition_idx': 0, 'channel_idx': 0}).
        **kwargs
            Additional arguments passed to MaskApplicator.apply_to_dataset()

        Returns
        -------
        xr.DataArray or dict
            If aggregation is None: Masked data with NaN outside mask
            If aggregation is specified: Aggregated statistics
            If single experiment and return_dict=False: DataArray directly
            Otherwise: dict mapping experiment IDs to results

        Examples
        --------
        Extract mean AIF signal timecourse (average over channels):
        >>> aif_signal = anat_ref.extract_masked_signal(
        ...     mask_name='AIF',
        ...     target_experiments=[dce],
        ...     aggregation='mean',
        ...     preserve_dims='repetition_idx',
        ...     aggregate_extra_dims=['channel_idx', 'acquisition_idx']
        ... )
        # Result shape: (182,) - one value per repetition

        Extract signal per channel and repetition:
        >>> aif_signal = anat_ref.extract_masked_signal(
        ...     mask_name='AIF',
        ...     target_experiments=[dce],
        ...     aggregation='mean',
        ...     preserve_dims=['repetition_idx', 'channel_idx'],
        ...     aggregate_extra_dims='acquisition_idx'
        ... )
        # Result shape: (182, 8) - one value per repetition and channel

        Get full masked data with NaNs outside mask:
        >>> masked_data = anat_ref.extract_masked_signal(
        ...     mask_name='tumor',
        ...     target_experiments=[adc_map],
        ...     aggregation=None
        ... )

        Count voxels in mask:
        >>> voxel_count = anat_ref.extract_masked_signal(
        ...     mask_name='tumor',
        ...     target_experiments=[dce],
        ...     aggregation='count'
        ... )

        Select specific channel before masking:
        >>> aif_signal = anat_ref.extract_masked_signal(
        ...     mask_name='AIF',
        ...     target_experiments=[dce],
        ...     preserve_dims='repetition_idx',
        ...     select_dims={'channel_idx': 0, 'acquisition_idx': 0}
        ... )
        # Result shape: (182,) - only channel 0, acquisition 0
        """
        # Default spatial dimensions
        if spatial_dims is None:
            spatial_dims = ["read_idx", "phase_idx", "slice_idx"]

        # Normalize aggregate_extra_dims to list
        if aggregate_extra_dims is not None:
            if isinstance(aggregate_extra_dims, str):
                if aggregate_extra_dims.strip():
                    aggregate_extra_dims = [aggregate_extra_dims]
                else:
                    aggregate_extra_dims = None
            elif not isinstance(aggregate_extra_dims, (list, tuple)):
                raise TypeError(
                    f"aggregate_extra_dims must be a string or list of strings, got {type(aggregate_extra_dims).__name__}"
                )

        # Check for conflicts between preserve_dims and aggregate_extra_dims
        if aggregate_extra_dims and preserve_dims:
            preserve_list = (
                [preserve_dims] if isinstance(preserve_dims, str) else preserve_dims
            )
            overlap = set(preserve_list) & set(aggregate_extra_dims)
            if overlap:
                raise ValueError(
                    f"Dimensions {overlap} specified in both preserve_dims and aggregate_extra_dims. "
                    f"A dimension cannot be both preserved and aggregated."
                )

        # Validate aggregation method
        valid_aggregations = ["mean", "sum", "median", "count", None]
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got '{aggregation}'"
            )

        # Get boolean mask (always as dict internally for consistency)
        masks = self.apply_mask_to_datasets(
            mask_name=mask_name,
            target_experiments=target_experiments,
            alignment_method=alignment_method,
            return_dict=True,  # Always get dict internally
            preserve_dims=preserve_dims,
            select_dims=select_dims,
            **kwargs,
        )

        # Extract signal for each experiment
        results = {}
        for exp_id, mask in masks.items():
            if mask is None:
                logger.warning(f"Mask application failed for {exp_id}, skipping")
                results[exp_id] = None
                continue

            # Find corresponding experiment or DataArray
            target_exp = None
            for i, exp in enumerate(target_experiments):
                # Generate appropriate identifier for comparison
                if isinstance(exp, xr.DataArray):
                    current_id = exp.name if exp.name else f"data_array_{i}"
                else:
                    current_id = getattr(exp, "name", f"experiment_{i}")

                if current_id == exp_id:
                    target_exp = exp
                    break

            if target_exp is None:
                logger.warning(f"Could not find target for {exp_id}")
                results[exp_id] = None
                continue

            # Get data from experiment or DataArray
            if isinstance(target_exp, xr.DataArray):
                # It's already a DataArray - use it directly
                data = target_exp
            elif hasattr(target_exp, "reco_data") and target_exp.reco_data is not None:
                data = target_exp.reco_data
            elif (
                hasattr(target_exp, "data_array") and target_exp.data_array is not None
            ):
                data = target_exp.data_array
            else:
                logger.warning(f"No data found for {exp_id}")
                results[exp_id] = None
                continue

            # Apply mask based on aggregation type
            if aggregation == "count":
                # Count voxels in mask
                available_spatial_dims = [d for d in spatial_dims if d in mask.dims]
                result = mask.sum(dim=available_spatial_dims)
            else:
                # Apply boolean mask to data
                masked_data = data.where(mask)

                if aggregation is None:
                    # Return full masked data
                    result = masked_data
                elif aggregation == "mean":
                    # Mean over spatial dimensions
                    available_spatial_dims = [
                        d for d in spatial_dims if d in masked_data.dims
                    ]
                    result = masked_data.mean(dim=available_spatial_dims)
                elif aggregation == "sum":
                    # Sum over spatial dimensions
                    available_spatial_dims = [
                        d for d in spatial_dims if d in masked_data.dims
                    ]
                    result = masked_data.sum(dim=available_spatial_dims)
                elif aggregation == "median":
                    # Median over spatial dimensions
                    available_spatial_dims = [
                        d for d in spatial_dims if d in masked_data.dims
                    ]
                    result = masked_data.median(dim=available_spatial_dims)

            # Apply extra dimension aggregation if requested
            if aggregate_extra_dims and aggregation is not None:
                available_extra_dims = [
                    d for d in aggregate_extra_dims if d in result.dims
                ]
                if available_extra_dims:
                    logger.info(
                        f"Aggregating over extra dimensions: {available_extra_dims}"
                    )
                    if aggregation == "mean":
                        result = result.mean(dim=available_extra_dims)
                    elif aggregation == "sum":
                        result = result.sum(dim=available_extra_dims)
                    elif aggregation == "median":
                        result = result.median(dim=available_extra_dims)
                    elif aggregation == "count":
                        # For count, just keep the result (already summed over spatial)
                        pass

            results[exp_id] = result

        # Simplify return value for single experiment
        if len(target_experiments) == 1 and not return_dict:
            return list(results.values())[0]

        return results

    def _apply_masks_to_datasets(self, masks_dict, target_datasets):
        """Internal helper to apply multiple masks to target datasets."""
        all_results = {}

        for mask_name, mask_data in masks_dict.items():
            try:
                # Create applicator for this mask
                applicator = MaskApplicator(self, mask_data)

                # Apply to all target datasets
                results = applicator.batch_apply(
                    target_datasets, mask_name=f"{mask_name}_applied"
                )
                all_results[mask_name] = results

                print(f"  Applied mask '{mask_name}' to {len(results)} datasets")

            except Exception as e:
                print(f"  Failed to apply mask '{mask_name}': {e}")
                all_results[mask_name] = None

        # Store all results
        if not hasattr(self, "cross_dataset_masks"):
            self.cross_dataset_masks = {}
        self.cross_dataset_masks.update(all_results)

        print(
            f"Cross-dataset mask application complete. Results stored in self.cross_dataset_masks"
        )

        return all_results

    def get_mask_alignment_report(self, mask_name, target_experiment):
        """Get alignment quality report for applying a mask to a target dataset.

        Parameters
        ----------
        mask_name : str
            Name of the mask to analyze
        target_experiment
            Target experiment to analyze alignment with

        Returns
        -------
        dict
            Alignment quality report and recommendations
        """
        if mask_name not in self.masks:
            raise ValueError(
                f"Mask '{mask_name}' not found. Available masks: {list(self.masks.keys())}"
            )

        # Create mask applicator and get alignment report
        applicator = MaskApplicator(self, self.masks[mask_name])
        return applicator.get_alignment_quality_report(target_experiment)

    def create_mask(
        self,
        input_data=None,
        plot_params=None,
        **slider_defaults,
    ):
        """Create masks interactively using a robust interface with preprocessing support.

        This method launches an interactive widget for drawing masks on the data.
        CRITICAL ENHANCEMENT: Now ensures masks are created on the same preprocessed
        data that the user sees on screen, fixing issues with multi-dimensional data.

        Parameters
        ----------
        input_data : xr.DataArray, optional
            Data to display for masking. If not provided, it is auto-detected
            and preprocessed to match current plotter state.
        plot_params : dict, optional
            Parameters to pass to the underlying plot method. If not provided,
            current plotter widget states will be extracted.
        **slider_defaults : dict, optional
            Default slider positions (e.g., `slice_idx=10`) to pass to the
            plot method.

        Returns
        -------
        None
            This function displays interactive widgets in a Jupyter environment.
            Masks can be retrieved after creation using `self.get_finalized_masks()`.

        Notes
        -----
        Key Features:
        - Masks use the same coordinate system as the displayed data, preventing alignment issues.
        - Supports complex multi-dimensional data with channel combination and rep averaging.
        - High-visibility lasso tool for clear mask drawing.
        - Session persistence: previously created masks are restored and visible.
        - Accurate volume calculations on preprocessed data.
        - Debugging information is logged to `/tmp/create_mask_debug.log`.

        Usage
        -----
        # Start the interactive masking session
        anatomy.create_mask()

        # After drawing and finalizing masks in the widget, retrieve them:
        finalized_masks = anatomy.get_finalized_masks()
        """

        # Allow callers to control the mask UI layout without leaking the key to the plotter
        layout_mode = "horizontal"
        if plot_params is not None:
            layout_mode = plot_params.get("mask_layout", layout_mode)
            if "mask_layout" in plot_params:
                plot_params = {
                    key: value
                    for key, value in plot_params.items()
                    if key != "mask_layout"
                }
        layout_mode = (
            str(layout_mode).strip().lower()
            if layout_mode is not None
            else "horizontal"
        )
        if layout_mode not in {"horizontal", "vertical"}:
            layout_mode = "horizontal"

        # CRITICAL ENHANCEMENT: Extract current plotter state for preprocessing consistency
        enhanced_plot_params, enhanced_input_data = (
            self._prepare_mask_data_with_preprocessing(
                input_data, plot_params, **slider_defaults
            )
        )

        # ENHANCED: Reuse existing mask creator if available to preserve masks
        if hasattr(self, "_mask_creator") and self._mask_creator:
            print("🔄 Reusing existing mask creator to preserve previous masks")
            mask_creator = self._mask_creator
            existing_mask_count = len(mask_creator.masks)
            print(
                f"📋 Found {existing_mask_count} existing masks: {list(mask_creator.masks.keys())}"
            )
        else:
            print("🆕 Creating new mask creator")
            # Create mask creator with debug logging
            mask_creator = MaskCreator(
                data_source=self,
            )
            # Store reference for accessing results
            self._mask_creator = mask_creator

        # ENHANCED: Restore finalized masks to plotter for immediate visibility
        self._restore_finalized_masks_to_plotter()

        # Launch interactive session with preprocessed data and parameters
        try:
            debug_log_create_mask(
                "🚀 Launching mask creator with preprocessing support"
            )
            debug_log_create_mask(f"📊 Enhanced plot params: {enhanced_plot_params}")
            if enhanced_input_data is not None:
                debug_log_create_mask(
                    f"📊 Enhanced input data shape: {enhanced_input_data.shape}"
                )
                debug_log_create_mask(
                    f"📊 Enhanced input data dims: {enhanced_input_data.dims}"
                )

            mask_creator.create_interactive_session(
                input_data=enhanced_input_data,
                plot_params=enhanced_plot_params,
                layout_mode=layout_mode,
                **slider_defaults,
            )

            # Show existing masks info
            finalized_masks = self.get_finalized_masks()
            if finalized_masks:
                print(f"📋 Available masks: {list(finalized_masks.keys())}")
                print("💡 Select existing masks from dropdown to edit")

        except Exception as e:
            logger.error(f"Failed to launch enhanced masking: {e}")
            print(f"❌ Error launching enhanced masking: {e}")
            print("💡 Check debug log for detailed error information")
            raise

    def _prepare_mask_data_with_preprocessing(
        self, input_data, plot_params, **slider_defaults
    ):
        """
        Extract current plotter state and prepare data for consistent masking.

        CRITICAL METHOD: This ensures masks are created on the same preprocessed
        data that the user sees, fixing multi-dimensional data issues.

        Parameters
        ----------
        input_data : xr.DataArray, optional
            Data provided by user
        plot_params : dict, optional
            Plot parameters provided by user
        **slider_defaults : dict
            Slider defaults provided by user

        Returns
        -------
        tuple
            (enhanced_plot_params, enhanced_input_data)
        """
        debug_log_create_mask("🔧 Preparing mask data with preprocessing consistency")

        # If user provided input_data, assume it's preprocessed and use as-is
        if input_data is not None:
            debug_log_create_mask("✅ Using user-provided input_data")
            return plot_params or {}, input_data

        # Extract current plotter widget states for preprocessing consistency
        enhanced_plot_params = self._extract_current_plotter_state(plot_params)
        debug_log_create_mask(f"📊 Extracted plotter state: {enhanced_plot_params}")

        # Get and preprocess data to match current display
        enhanced_input_data = self._get_preprocessed_display_data(
            enhanced_plot_params, **slider_defaults
        )

        if enhanced_input_data is not None:
            debug_log_create_mask(
                f"✅ Preprocessed data ready - shape: {enhanced_input_data.shape}"
            )
            debug_log_create_mask(
                f"📊 Preprocessed data dims: {enhanced_input_data.dims}"
            )
        else:
            debug_log_create_mask(
                "⚠️  No preprocessed data available, mask creator will use fallbacks"
            )

        return enhanced_plot_params, enhanced_input_data

    def _extract_current_plotter_state(self, user_plot_params):
        """Extract current widget states from plotter if available."""
        enhanced_params = user_plot_params.copy() if user_plot_params else {}

        # Try to extract current widget states from the plotter
        if hasattr(self, "plotter") and hasattr(self.plotter, "widgets"):
            try:
                widgets = self.plotter.widgets
                if widgets:
                    debug_log_create_mask(
                        "🎛️  Extracting current widget states from plotter"
                    )

                    # Extract common preprocessing widget states
                    widget_keys = [
                        "combine_channels",
                        "_combine_channels",
                        "average_reps",
                        "_average_reps",
                        "spec_part",
                        "_spec_part",
                    ]

                    for key in widget_keys:
                        if key in widgets:
                            widget = widgets[key]
                            if hasattr(widget, "value"):
                                enhanced_params[key] = widget.value
                                debug_log_create_mask(f"   📝 {key}: {widget.value}")
                else:
                    debug_log_create_mask("⚠️  No plotter widgets available")
            except Exception as e:
                debug_log_create_mask(f"⚠️  Failed to extract plotter state: {e}")
        else:
            debug_log_create_mask("⚠️  No plotter available for state extraction")

        return enhanced_params

    def _get_preprocessed_display_data(self, plot_params, **slider_defaults):
        """Get data that matches what the plotter displays."""
        try:
            # Use the plotter's preprocessing pipeline if available
            if hasattr(self, "plotter"):
                debug_log_create_mask("🔧 Using plotter's preprocessing pipeline")

                # Get raw data
                raw_data = self._get_raw_data_for_preprocessing()
                if raw_data is None:
                    return None

                # Apply the same preprocessing as the plotter
                if hasattr(self.plotter, "_apply_preprocessing"):
                    processed_data = self.plotter._apply_preprocessing(
                        raw_data, plot_params, ""  # Empty widget prefix
                    )
                    debug_log_create_mask(
                        f"✅ Preprocessing applied - shape: {processed_data.shape}"
                    )
                    return processed_data
                else:
                    debug_log_create_mask(
                        "⚠️  Plotter has no _apply_preprocessing method"
                    )
                    return raw_data
            else:
                debug_log_create_mask("⚠️  No plotter available for preprocessing")
                return None

        except Exception as e:
            debug_log_create_mask(f"❌ Preprocessing failed: {e}")
            return None

    def _get_raw_data_for_preprocessing(self):
        """Get raw data from the same sources the plotter uses."""
        # Try the same data sources as the plotter in order of preference
        candidates = [
            "reco_data",
            "data_array",
            "seq2d_rps_xarray",
            "reco_array",
        ]

        for candidate in candidates:
            if hasattr(self, candidate):
                data = getattr(self, candidate)
                if data is not None:
                    debug_log_create_mask(
                        f"📊 Using raw data from '{candidate}' for preprocessing"
                    )
                    return data

        debug_log_create_mask("❌ No raw data sources found")
        return None

    def _restore_finalized_masks_to_plotter(self):
        """
        Restore all finalized masks to plotter for persistent visibility.

        This ensures that previously created masks remain visible when
        starting a new masking session or switching slices.
        """
        try:
            if not hasattr(self, "_mask_creator") or not self._mask_creator:
                return

            # Get all finalized masks
            finalized_masks = {}
            for name, mask_container in self._mask_creator.masks.items():
                if mask_container.finalized:
                    finalized_masks[name] = mask_container

            if not finalized_masks:
                return

            # Sync all finalized masks to plotter for persistent display
            if hasattr(self, "plotter") and self.plotter:
                self._mask_creator.viz_sync.sync_all_finalized_masks(
                    self._mask_creator.masks, self.plotter
                )
                print(f"🔄 Restored {len(finalized_masks)} finalized masks to display")
                for name in finalized_masks:
                    pixels = finalized_masks[name].get_total_pixels()
                    print(f"   📍 {name}: {pixels} pixels")

        except Exception as e:
            logger.warning(f"Failed to restore finalized masks to plotter: {e}")

    def ensure_mask_visibility(self):
        """
        Ensure all finalized masks remain visible in the current plotting session.

        Call this method after switching slices or if masks disappear to restore visibility.
        """
        try:
            finalized_masks = self.get_finalized_masks()

            if not finalized_masks:
                print("📭 No finalized masks to restore")
                return

            if hasattr(self, "_mask_creator") and self._mask_creator:
                # Use the mask creator's visualization sync
                self._mask_creator.viz_sync.sync_all_finalized_masks(
                    self._mask_creator.masks, self.plotter
                )

                # Enable debug mode on plotter for better mask alignment
                if hasattr(self.plotter, "_debug_mask_alignment"):
                    self.plotter._debug_mask_alignment = True

                print(f"🔄 Restored visibility for {len(finalized_masks)} masks:")
                for name, mask_data in finalized_masks.items():
                    if hasattr(mask_data, "values"):
                        pixels = int(np.sum(mask_data.values))
                    else:
                        pixels = "unknown"
                    print(f"   📍 {name}: {pixels} pixels")

            else:
                print(
                    "⚠️  No mask creator available - use create_mask_simplified() first"
                )

        except Exception as e:
            logger.warning(f"Failed to ensure mask visibility: {e}")
            print(f"❌ Error restoring mask visibility: {e}")
            print("💡 Try calling create_mask_simplified() again to restore masks")

    def debug_mask_slice_distribution(self):
        """
        Debug method to check which slices contain mask data.

        This helps diagnose slice indexing issues where masks appear on wrong slices.
        """
        try:
            finalized_masks = self.get_finalized_masks()

            if not finalized_masks:
                print("📭 No finalized masks to analyze")
                return

            print("🔍 MASK SLICE DISTRIBUTION ANALYSIS")
            print("=" * 50)

            for mask_name, mask_data in finalized_masks.items():
                print(f"\n📍 Mask: '{mask_name}'")
                print(f"   Shape: {mask_data.shape}")
                print(f"   Dims: {mask_data.dims}")

                # Check if this is a 3D mask with slice dimension
                if "slice_idx" in mask_data.dims:
                    print("   Slice-by-slice analysis:")

                    total_slices = mask_data.sizes["slice_idx"]
                    slice_with_data = []

                    for slice_i in range(total_slices):
                        slice_mask = mask_data.isel(slice_idx=slice_i)
                        pixels_in_slice = int(np.sum(slice_mask.values))

                        if pixels_in_slice > 0:
                            slice_with_data.append((slice_i, pixels_in_slice))
                            print(f"     Slice {slice_i}: {pixels_in_slice} pixels ✓")

                    if not slice_with_data:
                        print("     ⚠️  No slices contain mask data!")
                    else:
                        total_pixels = sum(pixels for _, pixels in slice_with_data)
                        print(
                            f"   Total: {total_pixels} pixels across {len(slice_with_data)} slices"
                        )
                        print(f"   Slices with data: {[s for s, _ in slice_with_data]}")
                else:
                    # 2D mask or no slice dimension
                    total_pixels = int(np.sum(mask_data.values))
                    print(f"   Total pixels: {total_pixels}")
                    print("   (No slice dimension - 2D mask)")

            print(f"\n🔍 Analysis complete - check if masks are on expected slices")

        except Exception as e:
            logger.warning(f"Failed to analyze mask slice distribution: {e}")
            print(f"❌ Error analyzing masks: {e}")

    def debug_current_mask_state(self):
        """
        Debug current masking state including widget values and slice selectors.

        Call this to understand what slice the system thinks it's on and why
        masks might appear on wrong slices.
        """
        try:
            print("🔍 CURRENT MASK STATE ANALYSIS")
            print("=" * 50)

            # Check widget values
            if hasattr(self, "plotter") and hasattr(self.plotter, "widgets"):
                if self.plotter.widgets:
                    print("\n📱 Current Widget Values:")
                    for name, widget in self.plotter.widgets.items():
                        if hasattr(widget, "value"):
                            print(f"   {name}: {widget.value}")
                else:
                    print("\n⚠️  No widgets found in plotter")

            # Check plot handle state
            if hasattr(self, "_mask_creator") and self._mask_creator:
                update_plot = self._mask_creator.plot_handles.get("update_plot")
                if update_plot:
                    print(f"\n🎯 Plot Handle State:")
                    print(
                        f"   current_plot_dims: {getattr(update_plot, 'current_plot_dims', 'NOT_SET')}"
                    )
                    print(
                        f"   current_selector: {getattr(update_plot, 'current_selector', 'NOT_SET')}"
                    )
                    print(
                        f"   _current_extent: {getattr(update_plot, '_current_extent', 'NOT_SET')}"
                    )

            # Check lasso handler state
            if (
                hasattr(self, "_mask_creator")
                and self._mask_creator
                and self._mask_creator.lasso_handler
            ):
                lasso = self._mask_creator.lasso_handler
                print(f"\n🎯 Lasso Handler State:")
                print(f"   current_plot_dims: {lasso.current_plot_dims}")
                print(f"   current_slice_selector: {lasso.current_slice_selector}")
                print(f"   current_plot_extent: {lasso.current_plot_extent}")
                print(
                    f"   active_mask: {lasso.active_mask.name if lasso.active_mask else 'NONE'}"
                )

            # Check active mask info
            if hasattr(self, "_mask_creator") and self._mask_creator:
                print(f"\n📋 Active Mask Info:")
                print(f"   active_mask_name: {self._mask_creator.active_mask_name}")

                if self._mask_creator.active_mask_name:
                    active_mask = self._mask_creator.masks.get(
                        self._mask_creator.active_mask_name
                    )
                    if active_mask:
                        print(f"   mask buffer shape: {active_mask.buffer.shape}")
                        print(f"   mask buffer dims: {active_mask.buffer.dims}")
                        print(f"   total pixels: {active_mask.get_total_pixels()}")

            print(f"\n💡 Use this info to debug slice indexing issues")
            print(f"💡 Check if widget values match where you expect masks to appear")

        except Exception as e:
            logger.warning(f"Failed to debug current mask state: {e}")
            print(f"❌ Error debugging state: {e}")

    def trace_mask_slice_pipeline(self, mask_name: str):
        """
        Comprehensive tracing of mask slice pipeline from storage to display.

        This traces exactly how a mask gets from the 3D buffer to the 2D display
        and identifies where slice indexing goes wrong.

        Parameters
        ----------
        mask_name : str
            Name of mask to trace
        """
        try:
            print("🔍 MASK SLICE PIPELINE TRACE")
            print("=" * 60)

            # Get mask data
            finalized_masks = self.get_finalized_masks()
            if mask_name not in finalized_masks:
                print(f"❌ Mask '{mask_name}' not found in finalized masks")
                return

            mask_data = finalized_masks[mask_name]
            print(f"\n📋 STEP 1: Mask Storage Analysis")
            print(f"   Mask name: {mask_name}")
            print(f"   Buffer shape: {mask_data.shape}")
            print(f"   Buffer dims: {mask_data.dims}")
            print(f"   Coordinate keys: {list(mask_data.coords.keys())}")

            # Analyze slice distribution in buffer
            if "slice_idx" in mask_data.dims:
                print(f"\n🔍 STEP 2: Buffer Slice Distribution")
                slices_with_data = []
                for i in range(mask_data.sizes["slice_idx"]):
                    slice_data = mask_data.isel(slice_idx=i)
                    pixels = int(np.sum(slice_data.values))
                    if pixels > 0:
                        slices_with_data.append((i, pixels))
                        print(f"   Buffer slice {i}: {pixels} pixels ✓")

                print(f"   Total slices with data: {slices_with_data}")

            # Get current widget/plot state
            print(f"\n🎯 STEP 3: Current Display State")
            current_slice_widget = None
            if hasattr(self, "plotter") and hasattr(self.plotter, "widgets"):
                if "slice_idx" in self.plotter.widgets:
                    current_slice_widget = self.plotter.widgets["slice_idx"].value
                    print(f"   Current widget slice_idx: {current_slice_widget}")

            # Simulate the display pipeline
            print(f"\n🖥️  STEP 4: Display Pipeline Simulation")

            if hasattr(self, "plotter") and current_slice_widget is not None:
                # Simulate _apply_mask_slicing
                display_selectors = {"slice_idx": current_slice_widget}
                print(f"   Display selectors: {display_selectors}")

                try:
                    # Apply the same slicing as the display system
                    display_slice = mask_data.isel(**display_selectors)
                    display_pixels = int(np.sum(display_slice.values))
                    print(f"   Display slice shape: {display_slice.shape}")
                    print(f"   Display slice pixels: {display_pixels}")

                    if display_pixels > 0:
                        print(
                            f"   ✅ Mask should be visible on slice {current_slice_widget}"
                        )
                    else:
                        print(f"   ❌ No mask data on slice {current_slice_widget}")

                        # Find which slice actually has the data
                        for stored_slice, stored_pixels in slices_with_data:
                            if stored_pixels > 0:
                                print(
                                    f"   💡 Mask data is actually on slice {stored_slice} ({stored_pixels} pixels)"
                                )
                                print(
                                    f"   🐛 BUG: Widget shows slice {current_slice_widget}, but mask is on slice {stored_slice}"
                                )
                                break

                except Exception as e:
                    print(f"   ❌ Display simulation failed: {e}")

            print(f"\n🎯 STEP 5: Diagnosis")
            if "slice_idx" in mask_data.dims and current_slice_widget is not None:
                # Check if there's a mismatch
                expected_slices = [s for s, _ in slices_with_data]
                if expected_slices and current_slice_widget not in expected_slices:
                    print(f"   🐛 SLICE INDEXING BUG DETECTED!")
                    print(f"   📍 Current display slice: {current_slice_widget}")
                    print(f"   📍 Slices with mask data: {expected_slices}")
                    print(f"   💡 The mask was stored on wrong slice during creation")
                else:
                    print(f"   ✅ Slice indexing appears correct")

            print(f"\n=" * 60)

        except Exception as e:
            logger.warning(f"Failed to trace mask slice pipeline: {e}")
            print(f"❌ Error tracing pipeline: {e}")
            import traceback

            traceback.print_exc()

    def get_finalized_masks(self):
        """
        Get finalized masks from the simplified mask creator.

        Returns
        -------
        dict
            Dictionary of finalized masks {name: xr.DataArray}
            Each mask has identical coordinates to the source data.
        """
        # Prefer masks from the new MaskCreator (these are MaskContainer objects)
        if hasattr(self, "_mask_creator") and self._mask_creator:
            # mask_containers: name -> MaskContainer
            mask_containers = getattr(self._mask_creator, "masks", {}).copy()

            # Preserve original MaskContainer objects for advanced usage
            try:
                self.mask_containers = mask_containers
            except Exception:
                self.mask_containers = {}

            # Build Dataset from finalized mask buffers (MaskContainer.buffer -> xr.DataArray)
            mask_dict = {}
            for name, container in mask_containers.items():
                try:
                    buf = getattr(container, "buffer", None)
                    if isinstance(buf, xr.DataArray) and buf.attrs.get(
                        "finalized", False
                    ):
                        clean_name = name.replace(" ", "_").replace("-", "_")
                        mask_dict[clean_name] = buf
                except Exception:
                    continue

            if mask_dict:
                masks_ds = xr.Dataset(mask_dict)
                masks_ds.attrs["creation_method"] = "create_mask_simplified"
                masks_ds.attrs["total_masks"] = len(mask_dict)
                masks_ds.attrs["mask_names"] = list(mask_dict.keys())

                # Store Dataset on instance for easy access
                try:
                    self.masks = masks_ds
                except Exception:
                    pass

                return masks_ds
            else:
                # No finalized masks
                return xr.Dataset()

            # Fallback to check for masks from original create_mask (could already be a Dataset)
            existing = getattr(self, "masks", None)
            if isinstance(existing, dict):
                # try to convert dict of DataArrays to Dataset
                try:
                    ds = xr.Dataset(existing)
                    self.masks = ds
                    return ds
                except Exception:
                    return xr.Dataset()
            elif isinstance(existing, xr.Dataset):
                return existing
            else:
                return xr.Dataset()

    def get_mask_creator(self):
        """
        Get the mask creator instance for advanced operations.

        Returns
        -------
        MaskCreator or None
            The mask creator instance if available
        """
        return getattr(self, "_mask_creator", None)

    def create_overlay_with(self, other_experiments, blend_mode="alpha"):
        """Create multi-dataset overlay using enhanced AnatomyPlotter.

        Parameters
        ----------
        other_experiments : list
            List of other MRI experiments to overlay
        blend_mode : str, default "alpha"
            Blending mode for overlays

        Returns
        -------
        dict
            Plot handles for the multi-dataset overlay viewer
        """
        # This will be implemented when we enhance the AnatomyPlotter in Phase 2
        print(f"Multi-dataset overlay with {len(other_experiments)} datasets")
        print("Note: Enhanced overlay functionality will be available in Phase 2")

        # For now, return the standard plot
        return self.plot()
