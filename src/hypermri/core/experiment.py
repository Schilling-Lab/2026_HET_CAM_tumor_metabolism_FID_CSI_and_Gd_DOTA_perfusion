"""
Core experiment class for the hypermri package.

This module provides the new MRIExperiment class that uses registry-based
strategy selection instead of the massive if-elif chain.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

from ..mixins import StandardizedMRIMixin
from ..reconstruction import get_reconstruction_strategy
from ..utils.system.logging import debug_log, init_default_logger
from ..vendors import get_vendor_handler
from .data_types import (
    ExperimentMetadata,
    VendorType,
)
from .exceptions import ReconstructionError
from .interfaces import VendorHandlerProtocol

# For paper_summary() method only
try:
    pass

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

logger = init_default_logger(__name__)

# Global configuration
DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


DEBUG_FILE = "/tmp/experiment_debug.log"
DEBUG_LOG = True


def debug_log_experiment(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


class MRIExperiment(StandardizedMRIMixin):
    """
    Modern MRI experiment wrapper with registry-based vendor delegation.

    This class provides a standardized interface for MRI experiments
    regardless of vendor (Bruker, Siemens, etc.) using a clean
    registry-based architecture.
    """

    def __init__(
        self,
        path: Union[str, Path],
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: Optional[str] = None,
        handler_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialise a registry-backed experiment wrapper.

        Parameters
        ----------
        path
            Root directory that stores the vendor-specific measurement files.
        load_data
            If ``True``, the vendor handler immediately loads the raw data
            payload when instantiated. Defers loading otherwise.
        log_mode
            Logging verbosity forwarded to vendor handlers and reconstruction
            strategies. Typical values are ``"error"`` (default) or ``"debug"``.
        force_vendor
            Optional vendor name that bypasses automatic vendor detection. The
            value is normalised to lowercase before being stored.
        handler_options
            Extra keyword arguments forwarded to the vendor handler factory to
            customise loading behaviour.
        """
        super().__init__()

        # Store log_mode for passing to reconstruction functions
        self.log_mode = log_mode
        self._forced_vendor = force_vendor.lower() if force_vendor else None
        self._handler_options = handler_options or {}

        # Use standardized path handling
        self.path = path

        # Get vendor handler using the new registry system
        # Note: log_mode is accepted for backward compatibility but not currently used in registry system
        self.handler = self._get_vendor_handler(
            str(self.path),
            load_data,
            log_mode,
            self._forced_vendor,
            self._handler_options,
        )

        # Store log_mode on handler if it exists
        if self.handler and hasattr(self.handler, "__dict__"):
            self.handler.log_mode = log_mode
        # Note: Unlike the strict version, we allow handler to be None for backward compatibility
        # This allows Study class to filter out invalid paths later via scan.handler is not None

        # Initialize metadata
        self._metadata = None
        self.reconstructions = {}

    @property
    def id(self) -> str:
        """Return the experiment identifier supplied by the vendor handler.

        Returns
        -------
        str
            Identifier emitted by :py:attr:`~hypermri.core.interfaces.VendorHandlerProtocol.id`.
            ``"unknown"`` is returned if no handler was initialised.
        """
        if not self.handler:
            return "unknown"
        return self.handler.id

    @property
    def type(self) -> str:
        """Return the sequence name associated with the experiment.

        Returns
        -------
        str
            Sequence identifier reported by the vendor handler. ``"unknown"``
            is returned when the experiment lacks a handler.
        """
        if not self.handler:
            return "unknown"
        return self.handler.type

    @property
    def metadata(self) -> ExperimentMetadata:
        """Generate and cache experiment metadata.

        Returns
        -------
        ExperimentMetadata
            Dataclass describing vendor, sequence, and parameter snapshot for
            the experiment.
        """
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata

    def reconstruct(self, reco_type: str, **kwargs) -> Optional[xr.DataArray]:
        """Run a registered reconstruction strategy and store the result.

        Parameters
        ----------
        reco_type
            Name of the reconstruction workflow to execute, such as ``"raw"``
            or ``"seq2d"``.
        **kwargs
            Keyword arguments that override the strategy's default options.

        Returns
        -------
        xarray.DataArray or None
            Reconstructed dataset, or ``None`` if the strategy explicitly
            returns nothing.

        Raises
        ------
        ReconstructionError
            If no handler is available, if the registry cannot provide a
            strategy, or if the strategy itself raises an exception.
        """
        if not self.handler:
            raise ReconstructionError("No handler available for this experiment")

        try:
            # Get strategy and options using the new registry system
            strategy, default_options = get_reconstruction_strategy(
                vendor=self._get_vendor_name(), sequence=self.type, reco_type=reco_type
            )

            if not strategy:
                raise ReconstructionError(
                    f"{self.id}: No reconstruction strategy found for vendor "
                    f"'{self._get_vendor_name()}', sequence '{self.type}', "
                    f"reconstruction type '{reco_type}'",
                    strategy=f"{self._get_vendor_name()}.{self.type}.{reco_type}",
                    sequence=self.type,
                )

            # Merge default options with provided kwargs
            final_options = {**default_options, **kwargs}

            logger.debug(
                f"{self.id} ({self.type}): Using strategy: {strategy.__class__.__name__}"
            )
            logger.debug(f"Reconstruction options: {final_options}")

            # Perform reconstruction
            reconstructed_data = strategy.reconstruct(self.handler, **final_options)

            # Validate reconstruction result
            if reconstructed_data is None:
                logger.warning(
                    f"Reconstruction failed for {reco_type} - strategy returned None"
                )
                return None

            # Store reconstruction using standardized data handling
            self._store_reconstruction(reco_type, reconstructed_data)

            return reconstructed_data

        except Exception as e:
            raise ReconstructionError(
                f"Reconstruction failed: {e}",
                strategy=f"{self._get_vendor_name()}.{self.type}.{reco_type}",
                sequence=self.type,
            )

    def load_reconstruction(self, recon_num: int = 1, set_as_primary: bool = True):
        """
        Load a specific pdata reconstruction and optionally set it as the primary data.

        This method delegates to the vendor-specific handler to load different
        reconstructions (e.g., pdata/1, pdata/2, etc.) after the scan object has
        been initialized.

        Parameters
        ----------
        recon_num : int, optional
            The pdata reconstruction number to load (e.g., 1 for pdata/1, 2 for pdata/2).
            Default is 1.
        set_as_primary : bool, optional
            If True, sets the loaded reconstruction as the primary data (accessible via
            scan.data_array). If False, only returns the data without modifying the scan's
            primary data. Default is True.

        Returns
        -------
        xr.DataArray or None
            The loaded reconstruction as an xarray DataArray, or None if loading failed.

        Examples
        --------
        >>> # Load a study and get a scan
        >>> study = Study('/path/to/study')
        >>> scan = study[5]
        >>>
        >>> # Load alternative reconstruction (pdata/2) and set as primary
        >>> scan.load_reconstruction(recon_num=2)
        >>> scan.data_array  # Now contains pdata/2 data
        >>>
        >>> # Load reconstruction without changing primary data (for comparison)
        >>> pdata3 = scan.load_reconstruction(recon_num=3, set_as_primary=False)
        >>> # Compare: scan.data_array still has pdata/2, pdata3 has pdata/3
        """
        if hasattr(self.handler, "load_reconstruction"):
            # Load from handler (this sets the handler's _data_array)
            result = self.handler.load_reconstruction(
                recon_num=recon_num,
                set_as_primary=False,  # Don't set on handler, we'll set on self instead
            )

            # Also set as primary on the MRIExperiment/sequence object if requested
            if set_as_primary and result is not None:
                self.set_primary_data(result, "scanner_data")

            return result
        else:
            raise NotImplementedError(
                f"load_reconstruction is not implemented for {type(self.handler).__name__}"
            )

    def get_available_reconstructions(self) -> Dict[str, Any]:
        """Describe the reconstruction strategies registered for this experiment.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of reconstruction names to a dictionary containing the
            strategy class name and the default options.
        """
        from ..reconstruction import get_reconstruction_registry

        registry = get_reconstruction_registry()
        vendor_name = self._get_vendor_name()

        available = {}
        for reco_type in ["raw", "seq2d", "kspace"]:
            strategy, options = registry.get_strategy(vendor_name, self.type, reco_type)
            if strategy:
                available[reco_type] = {
                    "strategy_class": strategy.__class__.__name__,
                    "default_options": options,
                }

        return available

    def _get_vendor_handler(
        self,
        path: str,
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: Optional[str] = None,
        handler_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[VendorHandlerProtocol]:
        """Instantiate a vendor handler resolved via the registry.

        Parameters
        ----------
        path
            Filesystem path that contains the scan data.
        load_data
            If ``True``, the handler eagerly loads the raw data payload.
        log_mode
            Logging level hint forwarded to handler creation.
        force_vendor
            Optional vendor override. When ``None`` the detection heuristic is
            executed.
        handler_options
            Additional keyword arguments forwarded to the handler factory.

        Returns
        -------
        VendorHandlerProtocol or None
            Concrete handler instance if detection and construction succeed,
            otherwise ``None``.
        """
        return get_vendor_handler(
            source=path,
            load_data=load_data,
            log_mode=log_mode,
            force_vendor=force_vendor,
            handler_options=handler_options,
        )

    def _get_vendor_name(self) -> str:
        """Infer the vendor identifier from the handler class name.

        Returns
        -------
        str
            Lowercase vendor key (for example ``"bruker"`` or ``"siemens"``).
        """
        class_name = self.handler.__class__.__name__
        if class_name == "BrukerHandler":
            return "bruker"
        elif class_name == "SiemensHandler":
            return "siemens"
        else:
            # Extract vendor name from class name
            return class_name.replace("Handler", "").lower()

    def _create_metadata(self) -> ExperimentMetadata:
        """Assemble :class:`ExperimentMetadata` from handler information.

        Returns
        -------
        ExperimentMetadata
            Metadata populated with handler parameters and inferred sequence
            category.
        """
        vendor_name = self._get_vendor_name()

        # Determine vendor type
        try:
            vendor_type = VendorType(vendor_name)
        except ValueError:
            vendor_type = VendorType.BRUKER  # Default fallback

        # Get parameters from handler
        try:
            parameters = self.handler.get_parameters()
        except Exception as e:
            logger.warning(f"Could not get parameters from handler: {e}")
            parameters = {}

        return ExperimentMetadata(
            id=self.id,
            sequence_name=self.type,
            vendor=vendor_type,
            category=self._guess_sequence_category(),
            parameters=parameters,
        )

    def _guess_sequence_category(self):
        """Best-effort categorisation of the sequence label.

        Returns
        -------
        SequenceCategory
            ``SequenceCategory.SPECTROSCOPY`` if the sequence resembles a
            spectroscopy method, otherwise ``SequenceCategory.ANATOMICAL``.
        """
        from ..core.data_types import SequenceCategory

        sequence_lower = self.type.lower()

        # Spectroscopy sequences
        if any(
            seq in sequence_lower
            for seq in ["csi", "press", "steam", "laser", "nspect", "spec"]
        ):
            return SequenceCategory.SPECTROSCOPY

        # Default to anatomical
        return SequenceCategory.ANATOMICAL

    def _store_reconstruction(
        self, reco_type: str, reconstructed_data: xr.DataArray
    ) -> None:
        """Store reconstructed data in the standard experiment registry.

        Parameters
        ----------
        reco_type
            Name of the reconstruction that produced ``reconstructed_data``.
        reconstructed_data
            Resulting :class:`xarray.DataArray` emitted by the reconstruction
            strategy.

        Raises
        ------
        ValueError
            If ``reco_type`` is ``"raw"`` and ``reconstructed_data`` encodes
            k-space rather than image space data, which is no longer supported
            for persistent storage.
        """
        if reco_type == "seq2d":
            self.set_primary_data(reconstructed_data, "scanner_data")
        elif reco_type == "raw":
            # Check if it's k-space data
            if (
                hasattr(reconstructed_data, "attrs")
                and reconstructed_data.attrs.get("domain") == "kspace"
            ):
                raise ValueError(
                    "kspace_array is no longer supported for permanent storage. Use during reconstruction only."
                )
            else:
                self.set_primary_data(reconstructed_data, "reco_data")

        # Always store in reconstructions dict for reference
        self.reconstructions[reco_type] = reconstructed_data

    def __getattr__(self, name):
        """Delegate attribute access to the vendor handler when possible.

        Parameters
        ----------
        name
            Attribute name requested by Python's attribute lookup.

        Returns
        -------
        Any
            Attribute resolved from the vendor handler.

        Raises
        ------
        AttributeError
            If the attribute is not provided by either the experiment or the
            vendor handler.
        """
        if self.handler and hasattr(self.handler, name):
            return getattr(self.handler, name)

        handler_params = getattr(self.handler, "params", None) if self.handler else None
        if handler_params and name in handler_params:
            return handler_params[name]

        if name in self.params:
            return self.params[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __lt__(self, other):
        """Compare experiments by deferring to handler ordering.

        Parameters
        ----------
        other
            Another :class:`MRIExperiment` instance.

        Returns
        -------
        bool
            ``True`` when both experiments have handlers and the current handler
            is strictly less than ``other.handler``.
        """
        if not self.handler or not other.handler:
            return False
        debug_log_experiment(message=f"Comparing {self.handler} < {other.handler}")
        return self.handler < other.handler

    def __gt__(self, other):
        """Check if this experiment sorts higher than another.

        Parameters
        ----------
        other
            Another :class:`MRIExperiment` instance.

        Returns
        -------
        bool
            ``True`` when both experiments provide handlers and the current
            handler compares greater than ``other.handler``.
        """
        if not self.handler or not other.handler:
            return False
        return self.handler > other.handler

    @property
    def display_name(self) -> str:
        """Return a human-readable experiment description.

        Returns
        -------
        str
            String representation of the underlying handler, or a fallback that
            marks the experiment invalid when no handler is available.
        """
        if not self.handler:
            return f"Invalid experiment: {self.path}"
        return str(self.handler)

    def paper_summary(self):
        """Generate a paper-ready experiment summary (requires IPython)."""
        if not HAS_IPYTHON:
            print("IPython is required for paper_summary functionality")
            return

        # This would be a direct copy with minimal changes
