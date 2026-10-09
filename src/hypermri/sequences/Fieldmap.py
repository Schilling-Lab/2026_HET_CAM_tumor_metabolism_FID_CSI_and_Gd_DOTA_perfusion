"""Field mapping sequence wrapper providing reconvenience helpers."""

from hypermri.core.experiment import MRIExperiment

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

DB_MSG = False  # Change this to False to hide detailed error messages


class Fieldmap(BaseAnatomy):
    """Gradient-echo acquisition tailored for B0 field mapping."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Copy handler and data from the parent MRIExperiment object
        if hasattr(mri_exp_obj, "handler"):
            self.handler = mri_exp_obj.handler

        # Copy any existing reconstructed data
        if hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array
        # kspace_array no longer stored permanently

        # Perform reconstructions
        if not minimal_loading and not self._data_array_loaded:
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except Exception as e:
                logger.error(f"Fieldmap seq2d reconstruction failed: {e}")
                # Don't raise - allow partial initialization

        # Raw data processing:
        if not minimal_loading:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except Exception as e:
                logger.error(f"Fieldmap raw reconstruction failed: {e}")
                # Don't raise - allow partial initialization
