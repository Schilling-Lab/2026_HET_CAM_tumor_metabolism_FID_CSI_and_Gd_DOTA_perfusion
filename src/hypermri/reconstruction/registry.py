"""
Registry for reconstruction strategies.

This module provides a registry system that replaces the massive if-elif chain
in MRIExperiment.get_reco_strategy() with a configuration-driven approach.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import yaml

from ..core.exceptions import ConfigurationError
from ..core.interfaces import BaseRegistry, ReconstructionStrategyProtocol
from .aliases import resolve_sequence_alias


class ReconstructionRegistry(BaseRegistry):
    """Registry for reconstruction strategies."""

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.config_path = config_path
        self._strategy_cache = {}
        self._load_configurations()

    def register_strategy(
        self,
        vendor: str,
        sequence: str,
        reco_type: str,
        strategy_class: Type[ReconstructionStrategyProtocol],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a reconstruction strategy.

        Parameters
        ----------
        vendor: str
            Vendor name (e.g., 'bruker', 'siemens')
        sequence: str
            Sequence name (e.g., 'flash', 'csi')
        reco_type: str
            Reconstruction type (e.g., 'raw', 'seq2d')
        strategy_class: Type[ReconstructionStrategyProtocol]
            Strategy class implementing ReconstructionStrategyProtocol
        options: dict
            Default reconstruction options

        """
        key = self._make_key(vendor, sequence, reco_type)
        value = {"strategy_class": strategy_class, "options": options or {}}
        self.register(key, value, overwrite=True)

    def get_strategy(
        self, vendor: str, sequence: str, reco_type: str
    ) -> Tuple[Optional[ReconstructionStrategyProtocol], Dict[str, Any]]:
        """
        Get reconstruction strategy and options.

        Parameters
        ----------
        vendor: str
            Vendor name
        sequence: str
            Sequence name (will be resolved through aliases if needed)
        reco_type: str
            Reconstruction type

        Returns
        -------
            Tuple of (strategy_instance, options_dict)

        """
        # First, resolve any sequence aliases
        resolved_sequence = self._resolve_sequence_alias(sequence)

        key = self._make_key(vendor, resolved_sequence, reco_type)

        # Try exact match first
        entry = self.get(key)
        if entry:
            strategy = self._get_strategy_instance(entry["strategy_class"])
            return strategy, entry["options"].copy()

        # Try fallback patterns with resolved sequence
        fallback_key = self._find_fallback(vendor, resolved_sequence, reco_type)
        if fallback_key:
            entry = self.get(fallback_key)
            if entry:
                strategy = self._get_strategy_instance(entry["strategy_class"])
                return strategy, entry["options"].copy()

        return None, {}

    def list_strategies(self, vendor: Optional[str] = None) -> Dict[str, Any]:
        """List all registered strategies, optionally filtered by vendor."""
        strategies = {}
        for key in self.list_registered():
            if vendor is None or key.startswith(f"{vendor}."):
                strategies[key] = self.get(key)
        return strategies

    def _make_key(self, vendor: str, sequence: str, reco_type: str) -> str:
        """Create registry key from vendor, sequence, and reco_type."""
        return f"{vendor.lower()}.{sequence.lower()}.{reco_type.lower()}"

    def _get_strategy_instance(
        self, strategy_class: Type[ReconstructionStrategyProtocol]
    ) -> ReconstructionStrategyProtocol:
        """Get or create strategy instance with caching."""
        class_name = f"{strategy_class.__module__}.{strategy_class.__name__}"

        if class_name not in self._strategy_cache:
            self._strategy_cache[class_name] = strategy_class()

        return self._strategy_cache[class_name]

    def _find_fallback(
        self, vendor: str, sequence: str, reco_type: str
    ) -> Optional[str]:
        """Find fallback strategy using pattern matching."""
        vendor_lower = vendor.lower()
        sequence_lower = sequence.lower()
        reco_type_lower = reco_type.lower()

        # Sequence-based fallback patterns
        sequence_fallbacks = []

        # Check if sequence contains known patterns for additional fallbacks
        if "csi" in sequence_lower:
            sequence_fallbacks.append(f"{vendor_lower}.csi.{reco_type_lower}")
        elif "flash" in sequence_lower:
            sequence_fallbacks.append(f"{vendor_lower}.flash.{reco_type_lower}")
        elif "press" in sequence_lower:
            sequence_fallbacks.append(f"{vendor_lower}.press.{reco_type_lower}")
        elif "rare" in sequence_lower:
            sequence_fallbacks.append(f"{vendor_lower}.rare.{reco_type_lower}")
        elif "epi" in sequence_lower:
            sequence_fallbacks.append(f"{vendor_lower}.epi.{reco_type_lower}")

        # Common fallback patterns
        fallback_patterns = sequence_fallbacks + [
            # Try default strategy for vendor and reco_type
            f"{vendor_lower}.default.{reco_type_lower}",
            # Try base strategy class patterns
            f"{vendor_lower}.base.{reco_type_lower}",
        ]

        for pattern in fallback_patterns:
            if self.get(pattern):
                return pattern

        return None

    def _load_configurations(self) -> None:
        """Load strategy configurations from YAML file."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                self._load_from_config(config)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load reconstruction config from {self.config_path}: {e}",
                    config_type="reconstruction",
                    config_path=str(self.config_path),
                )
        else:
            # Load default configurations
            self._load_default_configurations()

    def _resolve_sequence_alias(self, sequence: str) -> str:
        """Resolve a sequence name using aliases."""
        return resolve_sequence_alias(sequence)

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        """Load strategies from configuration dictionary."""
        for vendor_name, vendor_config in config.get("vendors", {}).items():
            for sequence_name, sequence_config in vendor_config.get(
                "sequences", {}
            ).items():
                for reco_type, reco_config in sequence_config.items():
                    try:
                        strategy_class = self._import_strategy_class(
                            reco_config["strategy"]
                        )
                        options = reco_config.get("options", {})

                        self.register_strategy(
                            vendor=vendor_name,
                            sequence=sequence_name,
                            reco_type=reco_type,
                            strategy_class=strategy_class,
                            options=options,
                        )
                    except Exception as e:
                        # Log warning but continue loading other strategies
                        print(
                            f"Warning: Failed to load strategy {vendor_name}.{sequence_name}.{reco_type}: {e}"
                        )

    def _import_strategy_class(
        self, class_path: str
    ) -> Type[ReconstructionStrategyProtocol]:
        """Import strategy class from string path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = import_module(module_path)
            strategy_class = getattr(module, class_name)
            return strategy_class
        except Exception as e:
            raise ImportError(f"Could not import strategy class '{class_path}': {e}")

    def _load_default_configurations(self) -> None:
        """Load minimal default strategy configurations when config file is missing."""
        try:
            # Import only the essential strategies for basic functionality
            from hypermri.recon_strategy import (
                BrukerRawRecoStrategy,
                Brukerseq2dRecoStrategy,
            )

            # Register only minimal fallback strategies
            # These are essential for basic operation when config file is unavailable
            minimal_strategies = [
                # Default raw strategy for any Bruker sequence
                ("bruker", "default", "raw", BrukerRawRecoStrategy),
                # Default seq2d strategy for any Bruker sequence
                ("bruker", "default", "seq2d", Brukerseq2dRecoStrategy),
            ]

            for vendor, sequence, reco_type, strategy_class in minimal_strategies:
                self.register_strategy(vendor, sequence, reco_type, strategy_class, {})

        except ImportError as e:
            print(f"Warning: Could not load minimal default strategies: {e}")
            print("Reconstruction registry will rely entirely on config file.")


# Global registry instance
_reconstruction_registry = None


def get_reconstruction_registry() -> ReconstructionRegistry:
    """Get the global reconstruction registry instance."""
    global _reconstruction_registry
    if _reconstruction_registry is None:
        # Look for config file in reconstruction/config/
        config_path = Path(__file__).parent / "config" / "strategy_mapping.yaml"
        _reconstruction_registry = ReconstructionRegistry(config_path)
    return _reconstruction_registry


def register_reconstruction_strategy(
    vendor: str,
    sequence: str,
    reco_type: str,
    strategy_class: Type[ReconstructionStrategyProtocol],
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a reconstruction strategy."""
    registry = get_reconstruction_registry()
    registry.register_strategy(vendor, sequence, reco_type, strategy_class, options)


def get_reconstruction_strategy(
    vendor: str, sequence: str, reco_type: str
) -> Tuple[Optional[ReconstructionStrategyProtocol], Dict[str, Any]]:
    """Gget a reconstruction strategy."""
    registry = get_reconstruction_registry()
    return registry.get_strategy(vendor, sequence, reco_type)
