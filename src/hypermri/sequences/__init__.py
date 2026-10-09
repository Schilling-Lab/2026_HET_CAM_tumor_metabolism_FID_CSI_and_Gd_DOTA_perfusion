"""
Sequences module for hypermri.

This module provides sequence class management with smart loading
and registry capabilities.
"""

# Import all sequence classes
from .B1Calibration import B1Calibration

# Import base classes
from .base_anatomy import BaseAnatomy
from .base_spectroscopy import BaseSpectroscopy
from .BSSFP import BSSFP
from .CPMG import CPMG
from .CSI import CSI
from .DTIEPI import DTIEPI
from .EPI import EPI
from .FAIR_RARE import FAIR_RARE

# from .FEXI import FEXI
from .Fieldmap import Fieldmap
from .FLASH import FLASH
from .GeoffSlcSpec import GeoffSlcSpec
from .gNSPECT_MS import gNSPECT_MS
from .IVRecovery import IVRecovery
from .MGE import MGE
from .NSPECT import NSPECT
from .PRESS import PRESS
from .RARE import RARE
from .RAREVTR import RAREVTR
from .registry import (
    SequenceRegistry,
    get_sequence_class,
    get_sequence_registry,
    register_sequence,
)
from .semiLASER import semiLASER
from .Singlepulse import Singlepulse
from .STEAM import STEAM
from .umpireFLASH import umpireFLASH
from .umpireMGE.umpireMGE import umpireMGE

__all__ = [
    "SequenceRegistry",
    "get_sequence_registry",
    "register_sequence",
    "get_sequence_class",
    # Base classes
    "BaseAnatomy",
    "BaseSpectroscopy",
    # Sequence classes
    "B1Calibration",
    "BSSFP",
    "CPMG",
    "CSI",
    "DTIEPI",
    "EPI",
    "FAIR_RARE",
    "FEXI",
    "Fieldmap",
    "FLASH",
    "GeoffSlcSpec",
    "gNSPECT_MS",
    "IVRecovery",
    "MGE",
    "NSPECT",
    "PRESS",
    "RARE",
    "RAREVTR",
    "semiLASER",
    "Singlepulse",
    "STEAM",
    "umpireFLASH",
    "umpireMGE",
]
