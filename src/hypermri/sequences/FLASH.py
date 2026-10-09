"""FLASH gradient-echo sequence wrapper for anatomical imaging."""

from hypermri.core.experiment import MRIExperiment

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

DB_MSG = False  # Change this to False to hide detailed error messages


class FLASH(BaseAnatomy):
    """Fast low-angle shot gradient-echo sequence for anatomical imaging."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Copy handler and data from the parent MRIExperiment object
        if hasattr(mri_exp_obj, "handler"):
            self.handler = mri_exp_obj.handler

        # Copy any existing reconstructed data if available
        if hasattr(mri_exp_obj, "_data_array") and mri_exp_obj._data_array is not None:
            self._data_array = mri_exp_obj._data_array
            self._data_array_loaded = True
        # kspace_array no longer stored permanently
        if hasattr(mri_exp_obj, "_reco_data") and mri_exp_obj._reco_data is not None:
            self._reco_data = mri_exp_obj._reco_data
            self._reco_data_loaded = True

        # Only perform seq2d reconstruction during initialization
        # Raw reconstruction will be lazy-loaded when needed
        if not minimal_loading and not self._data_array_loaded:
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except Exception as e:
                logger.error(f"FLASH seq2d reconstruction failed: {e}")
                # Don't raise - allow partial initialization

        # Skip raw reconstruction for minimal loading - will be done lazily
        if not minimal_loading:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except Exception as e:
                logger.error(f"FLASH raw reconstruction failed: {e}")
                # Don't raise - allow partial initialization
