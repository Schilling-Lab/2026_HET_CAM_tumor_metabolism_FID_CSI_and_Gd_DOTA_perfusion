"""
Utilities for resolving sequence aliases to canonical names.

This module centralizes loading and resolving of sequence alias mappings that
are stored in ``reconstruction/config/sequence_aliases.yaml``. It provides a
lightweight, cached API that other parts of the codebase can reuse instead of
duplicating YAML parsing logic.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

_ALIAS_CACHE: Optional[Dict[str, str]] = None
_ALIAS_PATH = Path(__file__).parent / "config" / "sequence_aliases.yaml"


def _load_aliases() -> Dict[str, str]:
    """Load alias mapping from YAML (sequence_aliases.yaml) once and cache the lowercase dict."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE

    try:
        with open(_ALIAS_PATH, "r") as alias_file:
            alias_config = yaml.safe_load(alias_file) or {}
        aliases = alias_config.get("aliases", {})
        _ALIAS_CACHE = {key.lower(): value.lower() for key, value in aliases.items()}
    except FileNotFoundError:
        _ALIAS_CACHE = {}
    except Exception as exc:
        print(f"Warning: Failed to load sequence aliases from {_ALIAS_PATH}: {exc}")
        _ALIAS_CACHE = {}

    return _ALIAS_CACHE


def resolve_sequence_alias(sequence: Optional[str]) -> Optional[str]:
    """
    Resolve ``sequence`` name to its canonical alias if configured.

    Parameters
    ----------
    sequence
        Raw sequence name (e.g., ``\"<Bruker:MGE>\"``). ``None`` returns ``None``.

    Returns
    -------
    str or None
        Canonical sequence name (lowercase) or the normalized input if no alias
        exists.
    """
    if sequence is None:
        return None

    sequence_lower = sequence.lower()
    aliases = _load_aliases()
    return aliases.get(sequence_lower, sequence_lower)


def get_sequence_aliases() -> Dict[str, str]:
    """
    Return the cached alias dictionary. Mainly useful for diagnostics/tests.
    """
    return _load_aliases().copy()
