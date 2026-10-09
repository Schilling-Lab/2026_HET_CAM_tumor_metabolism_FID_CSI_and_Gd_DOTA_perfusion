"""
Vendor module for hypermri.

This module provides vendor handler management with automatic detection
and registration capabilities.
"""

from .registry import (
    VendorRegistry,
    detect_vendor,
    get_vendor_handler,
    get_vendor_registry,
    register_vendor,
)

__all__ = [
    "VendorRegistry",
    "get_vendor_registry",
    "register_vendor",
    "get_vendor_handler",
    "detect_vendor",
]
