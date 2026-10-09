"""
Registry for sequence classes.

This module provides a registry system for managing sequence classes
and their smart loading configurations.
"""

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from ..core import (
    BaseRegistry,
    ConfigurationError,
    SequenceCategory,
    SequenceProtocol,
)


class SequenceRegistry(BaseRegistry):
    """Registry for sequence classes."""

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.config_path = config_path
        self._load_configurations()

    def register_sequence(
        self,
        sequence_name: str,
        sequence_class: Type[SequenceProtocol],
        category: SequenceCategory,
        method_patterns: List[str],
        vendor: Optional[str] = None,
    ) -> None:
        """
        Register a sequence class.

        Parameters
            sequence_name: Name of the sequence (e.g., 'flash', 'csi')
            sequence_class: Sequence class implementing SequenceProtocol
            category: Sequence category (anatomical, spectroscopy, etc.)
            method_patterns: List of method name patterns that match this sequence
            vendor: Specific vendor (None for all vendors)
        """
        key = sequence_name.lower()
        if vendor:
            key = f"{vendor.lower()}.{key}"

        value = {
            "sequence_class": sequence_class,
            "category": category,
            "method_patterns": method_patterns,
            "vendor": vendor,
        }
        self.register(key, value, overwrite=True)

    def get_sequence_class(
        self, method_name: str, vendor: Optional[str] = None
    ) -> Optional[Type[SequenceProtocol]]:
        """
        Get sequence class based on method name.

        Parameters
            method_name: Method name from the scan
            vendor: Vendor name (for vendor-specific matching)

        Returns
            Sequence class or None if not found
        """
        if method_name is None:
            return None

        method_lower = method_name.lower()

        # Try vendor-specific match first
        if vendor:
            for key in self.list_registered():
                if key.startswith(f"{vendor.lower()}."):
                    entry = self.get(key)
                    if self._matches_method(method_lower, entry["method_patterns"]):
                        return entry["sequence_class"]

        # Try general matches
        for key in self.list_registered():
            if "." not in key:  # General (non-vendor-specific) entries
                entry = self.get(key)
                if self._matches_method(method_lower, entry["method_patterns"]):
                    return entry["sequence_class"]

        return None

    def list_sequences(
        self, category: Optional[SequenceCategory] = None, vendor: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        List registered sequences, optionally filtered.

        Parameters
            category: Filter by sequence category
            vendor: Filter by vendor

        Returns
            Dictionary of sequence information
        """
        sequences = {}
        for key in self.list_registered():
            entry = self.get(key)

            # Apply filters
            if category and entry["category"] != category:
                continue
            if vendor and entry.get("vendor") != vendor:
                continue

            sequences[key] = entry

        return sequences

    def get_sequence_info(
        self, sequence_name: str, vendor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about a registered sequence."""
        key = sequence_name.lower()
        if vendor:
            key = f"{vendor.lower()}.{key}"
        return self.get(key)

    def _matches_method(self, method_name: str, patterns: List[str]) -> bool:
        """Check if method name matches any of the patterns."""
        method_lower = method_name.lower()

        for pattern in patterns:
            pattern_lower = pattern.lower()

            if pattern_lower.startswith("exact:"):
                # Exact match
                exact_pattern = pattern_lower.replace("exact:", "")
                if method_lower == exact_pattern:
                    return True
            elif pattern_lower.startswith("startswith:"):
                # Starts with pattern
                start_pattern = pattern_lower.replace("startswith:", "")
                if method_lower.startswith(start_pattern):
                    return True
            elif pattern_lower.startswith("contains:"):
                # Contains pattern
                contains_pattern = pattern_lower.replace("contains:", "")
                if contains_pattern in method_lower:
                    return True
            else:
                # Default: contains matching
                if pattern_lower in method_lower:
                    return True

        return False

    def _load_configurations(self) -> None:
        """Load sequence configurations from file."""
        if self.config_path and self.config_path.exists():
            try:
                # Support both YAML and JSON (for backward compatibility)
                if self.config_path.suffix == ".json":
                    with open(self.config_path, "r") as f:
                        config = json.load(f)
                else:
                    with open(self.config_path, "r") as f:
                        config = yaml.safe_load(f)

                self._load_from_config(config)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load sequence config from {self.config_path}: {e}",
                    config_type="sequence",
                    config_path=str(self.config_path),
                )
        else:
            # Load default configurations
            self._load_default_configurations()

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        """Load sequences from configuration dictionary."""
        # Handle both old JSON format and new YAML format

        if "sequences" in config:
            # New YAML format
            sequences_config = config["sequences"]
        else:
            # Old JSON format (smart loading config)
            sequences_config = config

        for sequence_name, sequence_config in sequences_config.items():
            try:
                # Handle different config formats
                if isinstance(sequence_config, str):
                    # Simple string format: just class path
                    class_path = sequence_config
                    method_patterns = [sequence_name]  # Default pattern
                    category = SequenceCategory.ANATOMICAL  # Default
                    vendor = None
                elif isinstance(sequence_config, dict):
                    # Full config format
                    class_path = sequence_config.get(
                        "class", sequence_config.get("Class")
                    )
                    method_patterns = sequence_config.get(
                        "method_patterns", [sequence_name]
                    )
                    category_str = sequence_config.get("category", "anatomical")
                    category = SequenceCategory(category_str.lower())
                    vendor = sequence_config.get("vendor")
                else:
                    continue

                if class_path:
                    sequence_class = self._import_sequence_class(class_path)
                    self.register_sequence(
                        sequence_name=sequence_name,
                        sequence_class=sequence_class,
                        category=category,
                        method_patterns=method_patterns,
                        vendor=vendor,
                    )
            except Exception as e:
                print(f"Warning: Failed to load sequence {sequence_name}: {e}")

    def _import_sequence_class(self, class_path: str) -> Type[SequenceProtocol]:
        """Import sequence class from string path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = import_module(module_path)
            sequence_class = getattr(module, class_name)
            return sequence_class
        except Exception as e:
            raise ImportError(f"Could not import sequence class '{class_path}': {e}")

    def _load_default_configurations(self) -> None:
        """Load default sequence configurations."""
        try:
            # Load from existing smart loading config if available
            smart_config_path = (
                Path(__file__).parent.parent / "config" / "SmartLoadMeasurements.json"
            )
            if smart_config_path.exists():
                with open(smart_config_path, "r") as f:
                    config = json.load(f)
                self._load_from_config(config)
            else:
                # Hardcoded defaults for initial setup
                self._load_hardcoded_defaults()

        except Exception as e:
            print(f"Warning: Could not load default sequence configurations: {e}")
            self._load_hardcoded_defaults()

    def _load_hardcoded_defaults(self) -> None:
        """Load hardcoded default sequence mappings."""
        # Basic mappings to get started
        default_mappings = [
            (
                "flash",
                "hypermri.sequences.FLASH.FLASH",
                SequenceCategory.ANATOMICAL,
                ["flash"],
            ),
            (
                "csi",
                "hypermri.sequences.CSI.CSI",
                SequenceCategory.SPECTROSCOPY,
                ["csi"],
            ),
            (
                "press",
                "hypermri.sequences.PRESS.PRESS",
                SequenceCategory.SPECTROSCOPY,
                ["press"],
            ),
            (
                "rare",
                "hypermri.sequences.RARE.RARE",
                SequenceCategory.ANATOMICAL,
                ["rare"],
            ),
            ("epi", "hypermri.sequences.EPI.EPI", SequenceCategory.ANATOMICAL, ["epi"]),
        ]

        for seq_name, class_path, category, patterns in default_mappings:
            try:
                seq_class = self._import_sequence_class(class_path)
                self.register_sequence(seq_name, seq_class, category, patterns)
            except ImportError:
                # Skip if import fails (sequence may not exist yet)
                pass


# Global registry instance
_sequence_registry = None


def get_sequence_registry() -> SequenceRegistry:
    """Get the global sequence registry instance."""
    global _sequence_registry
    if _sequence_registry is None:
        # Look for config file in sequences/config/
        config_path = Path(__file__).parent / "config" / "sequence_mapping.yaml"
        _sequence_registry = SequenceRegistry(config_path)
    return _sequence_registry


def register_sequence(
    sequence_name: str,
    sequence_class: Type[SequenceProtocol],
    category: SequenceCategory,
    method_patterns: List[str],
    vendor: Optional[str] = None,
) -> None:
    """Convenience function to register a sequence class."""
    registry = get_sequence_registry()
    registry.register_sequence(
        sequence_name, sequence_class, category, method_patterns, vendor
    )


def get_sequence_class(
    method_name: str, vendor: Optional[str] = None
) -> Optional[Type[SequenceProtocol]]:
    """Convenience function to get a sequence class."""
    registry = get_sequence_registry()
    return registry.get_sequence_class(method_name, vendor)
