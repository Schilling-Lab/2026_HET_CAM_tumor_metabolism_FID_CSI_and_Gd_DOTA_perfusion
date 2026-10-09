import numpy as np

from hypermri.core.experiment import MRIExperiment

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__)


class RARE(BaseAnatomy):
    """Rapid acquisition with relaxation enhancement for fast spin-echo anatomy."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Handler copying is now handled by BaseAnatomy
        # Just verify the handler is available
        if hasattr(self, "handler") and self.handler:
            logger.debug(
                f"RARE: handler available, has fid: {hasattr(self.handler, 'fid')}"
            )
        else:
            logger.warning(
                "RARE: No handler available after BaseAnatomy initialization"
            )

        # Copy any existing reconstructed data
        if hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array
        # kspace_array no longer stored permanently

        if not minimal_loading:
            # Perform reconstructions
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except Exception as e:
                logger.error(f"RARE seq2d reconstruction failed: {e}")
                # Don't raise - allow partial initialization

            # Raw data processing:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except Exception as e:
                logger.error(f"RARE raw reconstruction failed: {e}")
                # Don't raise - allow partial initialization

    # ---------------------------------------------
    # This needs to brought to recon_strategy.py
    # ---------------------------------------------
    def postprocess_fid_rare(self):
        # process rawdata:
        self.fid_pp = self.postprocess_fid_file(input_data=self.fid)

        fid_process = self.fid_pp.copy()

        fid_process = np.reshape(
            fid_process, (-1, self.rare_factor, self.n_receivers, self.matrix[0])
        )

        order_ind = self.method["PVM_EncSteps1"]
        order = np.argsort(order_ind)

        if self.spat_dim == "<3D>":
            n_slices = self.enc_matrix[-1]
        else:
            n_slices = self.n_slices

        fid_reordered = []
        if self.spat_dim == "<3D>":
            order_ind2 = self.method["PVM_EncSteps2"]
            order2 = np.argsort(order_ind2)

            fid_process = np.reshape(
                fid_process,
                (
                    self.enc_matrix[2],
                    self.enc_matrix[1],
                    self.n_receivers,
                    self.matrix[0],
                ),
            )

            fid_process = fid_process[order2, ...]
            fid_process = np.transpose(fid_process, (1, 0, 2, 3))
            fid_process = fid_process[order, ...]
            fid_process = np.transpose(fid_process, (1, 0, 2, 3))
            fid_reordered = np.reshape(
                fid_process,
                (
                    -1,
                    n_slices,
                    self.enc_matrix[1],
                    self.n_receivers,
                    self.enc_matrix[0],
                ),
            )
            # reorder to: repetitions x slices x phase x channels x read
            fid_reordered = np.transpose(fid_reordered, (0, 1, 3, 2, 4))
            self.fid_pp = fid_reordered

        else:
            for k in range(n_slices):
                for r in range(self.n_receivers):
                    # select slice
                    temp = fid_process[k::n_slices, :, r, :]
                    # reshape selected slice:
                    temp = np.reshape(temp, (-1, self.matrix[0]))
                    temp = temp[order, ...]
                    # sort selected slice:
                    fid_reordered.append(temp)

            # order: slice x phase x read:
            fid_reordered = np.array(fid_reordered)

            fid_reordered = np.reshape(
                fid_reordered,
                (
                    -1,
                    self.n_slices,
                    self.n_receivers,
                    self.enc_matrix[1],
                    self.enc_matrix[0],
                ),
            )
            # reorder to: repetitions x slices x phase x channels x read
            fid_reordered = np.transpose(fid_reordered, (0, 1, 3, 2, 4))
            self.fid_pp = fid_reordered
