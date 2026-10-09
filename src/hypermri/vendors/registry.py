"""
Registry for vendor handlers.

This module provides a registry system for managing vendor-specific handlers
and their detection patterns.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__)

import yaml

from ..core.exceptions import ConfigurationError
from ..core.interfaces import BaseRegistry, VendorHandlerProtocol


class VendorRegistry(BaseRegistry):
    """Registry for vendor handlers."""

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.config_path = config_path
        self._load_configurations()

    def register_vendor(
        self,
        vendor_name: str,
        handler_class: Type[VendorHandlerProtocol],
        detection_patterns: List[str],
        priority: int = 0,
    ) -> None:
        """
        Register a vendor handler.

        Parameters
        ----------
        vendor_name: str
            Name of the vendor (e.g., 'bruker', 'siemens')
        handler_class: Type[VendorHandlerProtocol]
            Handler class implementing VendorHandlerProtocol
        detection_patterns: list
            List of patterns for automatic vendor detection
        priority: int
            Priority for detection (higher = checked first)

        """
        value = {
            "handler_class": handler_class,
            "detection_patterns": detection_patterns,
            "priority": priority,
        }
        self.register(vendor_name.lower(), value, overwrite=True)

    def get_vendor_handler(
        self,
        source: Union[str, Path],
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: Optional[str] = None,
        handler_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[VendorHandlerProtocol]:
        """
        Get vendor handler instance with automatic detection and option preparation.

        Parameters
        ----------
        source: str, Path
            Path to the data.
        load_data: bool
            Whether to load data immediately. Defaults to False
        log_mode: str
            Logging verbosity forwarded to handler
        force_vendor: bool
            Optional vendor override to bypass detection
        handler_options: dict
            Extra keyword arguments for handler

        Returns
        -------
        Vendor handler instance or None

        """
        try:
            # Use registry-based vendor detection
            vendor_name = force_vendor or self.detect_vendor(source)
            if not vendor_name:
                return None

            # Prepare handler options
            final_options = self._prepare_handler_options(handler_options, log_mode)

            # Get handler from registry entry
            entry = self.get(vendor_name.lower())
            if entry:
                handler_class = entry["handler_class"]
                return handler_class(
                    source=source, load_data=load_data, **final_options
                )
            return None

        except Exception as e:
            # Import here to avoid circular dependencies
            import traceback

            logger.error(f"Error creating vendor handler: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None

    def detect_vendor(self, path: Union[str, Path]) -> Optional[str]:
        """
        Automatically detect vendor from path.

        Parameters
        ----------
        path: Path
            Path to analyze for vendor

        Returns
        -------
        Vendor name: str
            vendor name or None if not detected

        """
        path = Path(path)

        # Get all vendors sorted by priority (descending)
        vendors = []
        for vendor_name in self.list_registered():
            entry = self.get(vendor_name)
            vendors.append((vendor_name, entry))

        vendors.sort(key=lambda x: x[1]["priority"], reverse=True)

        # First pass: ask each handler whether the path is a valid scan dir
        valid_matches = []
        for vendor_name, entry in vendors:
            handler_class = entry["handler_class"]

            if hasattr(handler_class, "is_valid_scan_dir"):
                try:
                    if handler_class.is_valid_scan_dir(path):
                        valid_matches.append((vendor_name, entry))
                except Exception:
                    continue

        if valid_matches:
            # Entries are already priority-sorted; prefer the first hit
            return valid_matches[0][0]

        # Second pass: fallback to pattern matching heuristics
        for vendor_name, entry in vendors:
            for pattern in entry["detection_patterns"]:
                if self._matches_pattern(path, pattern):
                    return vendor_name

        return None

    def list_vendors(self) -> List[str]:
        """List all registered vendor names."""
        return self.list_registered()

    def get_vendor_info(self, vendor_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered vendor."""
        return self.get(vendor_name.lower())

    def _matches_pattern(self, path: Path, pattern: str) -> bool:
        """Check if path matches detection pattern."""
        if pattern.startswith("extension:"):
            # File extension pattern
            ext = pattern.replace("extension:", "")
            return path.suffix.lower() == ext.lower()
        elif pattern.startswith("file:"):
            # Specific file pattern
            filename = pattern.replace("file:", "")
            return (
                (path / filename).exists() if path.is_dir() else path.name == filename
            )
        elif pattern.startswith("dir:"):
            # Directory pattern
            dirname = pattern.replace("dir:", "")
            return path.is_dir() and path.name == dirname
        elif pattern.startswith("contains:"):
            # Directory contains file pattern
            filename = pattern.replace("contains:", "")
            return path.is_dir() and any(p.name == filename for p in path.iterdir())
        else:
            # Direct string matching
            return pattern.lower() in str(path).lower()

    def _prepare_handler_options(
        self, handler_options: Optional[Dict[str, Any]], log_mode: str
    ) -> Dict[str, Any]:
        """Merge default registry kwargs with user overrides.

        Parameters
        ----------
        handler_options
            Optional mapping of user-provided handler keyword arguments.
        log_mode
            Logging verbosity hint propagated to the handler.

        Returns
        -------
        dict[str, Any]
            Combined handler keyword arguments.

        """
        options: Dict[str, Any] = {"log_mode": log_mode}
        if handler_options:
            options.update(handler_options)
        return options

    def _load_configurations(self) -> None:
        """Load vendor configurations from YAML file."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                self._load_from_config(config)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load vendor config from {self.config_path}: {e}",
                    config_type="vendor",
                    config_path=str(self.config_path),
                )
        else:
            # Load default configurations
            self._load_default_configurations()

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        """Load vendors from configuration dictionary."""
        for vendor_name, vendor_config in config.get("vendors", {}).items():
            try:
                handler_class = self._import_handler_class(
                    vendor_config["handler_class"]
                )
                detection_patterns = vendor_config.get("detection_patterns", [])
                priority = vendor_config.get("priority", 0)

                self.register_vendor(
                    vendor_name=vendor_name,
                    handler_class=handler_class,
                    detection_patterns=detection_patterns,
                    priority=priority,
                )
            except Exception as e:
                print(f"Warning: Failed to load vendor {vendor_name}: {e}")

    def _import_handler_class(self, class_path: str) -> Type[VendorHandlerProtocol]:
        """Import handler class from string path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = import_module(module_path)
            handler_class = getattr(module, class_name)
            return handler_class
        except Exception as e:
            raise ImportError(f"Could not import handler class '{class_path}': {e}")

    def _load_default_configurations(self) -> None:
        """Load default vendor configurations."""
        try:
            # Import current handlers to maintain compatibility
            from hypermri.brukerexp import BrukerHandler

            # Register Bruker
            self.register_vendor(
                vendor_name="bruker",
                handler_class=BrukerHandler,
                detection_patterns=[
                    "file:fid",
                    "file:rawdata.job0",
                    "file:2dseq",
                    "contains:method",
                    "contains:acqp",
                ],
                priority=100,
            )

            try:
                from hypermri.agilentexp import AgilentHandler

                self.register_vendor(
                    vendor_name="agilent",
                    handler_class=AgilentHandler,
                    detection_patterns=["file:fid", "file:procpar"],
                    priority=40,
                )
            except ImportError:
                pass

            try:
                from hypermri.siemensexp import SiemensHandler

                # Register Siemens
                self.register_vendor(
                    vendor_name="siemens",
                    handler_class=SiemensHandler,
                    detection_patterns=["extension:.dat", "file:.twix"],
                    priority=50,
                )
            except ImportError:
                pass

            try:
                from hypermri.externalexp import ExternalDataHandler

                # Register external data handler (requires explicit selection)
                self.register_vendor(
                    vendor_name="external",
                    handler_class=ExternalDataHandler,
                    detection_patterns=[],
                    priority=-100,
                )
            except ImportError:
                pass

        except ImportError as e:
            print(f"Warning: Could not load default vendor handlers: {e}")


# Global registry instance
_vendor_registry = None


def get_vendor_registry() -> VendorRegistry:
    """Get the global vendor registry instance."""
    global _vendor_registry
    if _vendor_registry is None:
        # Look for config file in vendors/config/
        config_path = Path(__file__).parent / "config" / "vendor_mapping.yaml"
        _vendor_registry = VendorRegistry(config_path)
    return _vendor_registry


def register_vendor(
    vendor_name: str,
    handler_class: Type[VendorHandlerProtocol],
    detection_patterns: List[str],
    priority: int = 0,
) -> None:
    """Convenience function to register a vendor handler."""
    registry = get_vendor_registry()
    registry.register_vendor(vendor_name, handler_class, detection_patterns, priority)


def get_vendor_handler(
    source: Union[str, Path],
    load_data: bool = False,
    log_mode: str = "error",
    force_vendor: Optional[str] = None,
    handler_options: Optional[Dict[str, Any]] = None,
) -> Optional[VendorHandlerProtocol]:
    """Convenience function to get a vendor handler with automatic detection."""
    registry = get_vendor_registry()
    return registry.get_vendor_handler(
        source=source,
        load_data=load_data,
        log_mode=log_mode,
        force_vendor=force_vendor,
        handler_options=handler_options,
    )


def detect_vendor(path: Union[str, Path]) -> Optional[str]:
    """Convenience function to detect vendor from path."""
    registry = get_vendor_registry()
    return registry.detect_vendor(path)
