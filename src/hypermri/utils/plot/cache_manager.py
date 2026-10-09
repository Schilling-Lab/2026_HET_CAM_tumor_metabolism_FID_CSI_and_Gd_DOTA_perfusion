"""Cache management module for unified plotting system.

This module provides a unified caching strategy that consolidates the three
separate cache systems previously used in AnatomyPlotter.
"""

import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class CacheManager:
    """Unified cache manager for plotting data.

    This class implements an LRU (Least Recently Used) cache with separate
    categories for different types of cached data. It replaces the three
    separate cache dictionaries in AnatomyPlotter.

    Categories:
    - 'preprocessing': Cached preprocessed data arrays
    - 'minmax': Cached min/max values for colormapping
    - 'transform': Cached data transformations (magnitude/phase/etc.)
    """

    def __init__(self, max_size: int = 10):
        """Initialize the CacheManager.

        Parameters
        ----------
        max_size : int, optional
            Maximum number of entries per category before eviction, by default 10
        """
        self.max_size = max_size
        self._caches: Dict[str, OrderedDict] = {
            "preprocessing": OrderedDict(),
            "minmax": OrderedDict(),
            "transform": OrderedDict(),
        }
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: Tuple, category: str = "preprocessing") -> Optional[Any]:
        """Get a value from the cache.

        Parameters
        ----------
        key : Tuple
            Cache key (should be hashable tuple)
        category : str, optional
            Cache category, by default 'preprocessing'

        Returns
        -------
        Optional[Any]
            Cached value if found, None otherwise
        """
        if category not in self._caches:
            raise ValueError(f"Invalid cache category: {category}")

        cache = self._caches[category]

        if key in cache:
            self._stats["hits"] += 1
            # Move to end (most recently used)
            cache.move_to_end(key)
            return cache[key]
        else:
            self._stats["misses"] += 1
            return None

    def set(self, key: Tuple, value: Any, category: str = "preprocessing") -> None:
        """Set a value in the cache.

        Parameters
        ----------
        key : Tuple
            Cache key (should be hashable tuple)
        value : Any
            Value to cache
        category : str, optional
            Cache category, by default 'preprocessing'
        """
        if category not in self._caches:
            raise ValueError(f"Invalid cache category: {category}")

        cache = self._caches[category]

        # If key already exists, update it and move to end
        if key in cache:
            cache[key] = value
            cache.move_to_end(key)
        else:
            # Check if we need to evict
            if len(cache) >= self.max_size:
                # Remove oldest entry (first item)
                cache.popitem(last=False)
                self._stats["evictions"] += 1

            # Add new entry
            cache[key] = value

    def invalidate(
        self, key: Optional[Tuple] = None, category: Optional[str] = None
    ) -> None:
        """Invalidate cache entries.

        Parameters
        ----------
        key : Optional[Tuple], optional
            Specific key to invalidate. If None, clears all entries in category
        category : Optional[str], optional
            Category to invalidate. If None, invalidates all categories
        """
        if category is None:
            # Invalidate all categories
            for cat in self._caches:
                if key is None:
                    self._caches[cat].clear()
                elif key in self._caches[cat]:
                    del self._caches[cat][key]
        else:
            # Invalidate specific category
            if category not in self._caches:
                warnings.warn(f"Invalid cache category: {category}")
                return

            cache = self._caches[category]
            if key is None:
                cache.clear()
            elif key in cache:
                del cache[key]

    def invalidate_by_data_id(self, data_id: int) -> None:
        """Invalidate all cache entries related to a specific data object.

        Parameters
        ----------
        data_id : int
            ID of the data object (from id(data_obj))
        """
        # Remove entries from minmax cache
        minmax_cache = self._caches["minmax"]
        keys_to_remove = [k for k in minmax_cache.keys() if k[0] == data_id]
        for key in keys_to_remove:
            del minmax_cache[key]

        # Remove entries from transform cache
        transform_cache = self._caches["transform"]
        keys_to_remove = [k for k in transform_cache.keys() if k[0] == data_id]
        for key in keys_to_remove:
            del transform_cache[key]

        # Remove entries from preprocessing cache
        preprocessing_cache = self._caches["preprocessing"]
        keys_to_remove = [
            k for k in preprocessing_cache.keys() if len(k) > 0 and k[0] == data_id
        ]
        for key in keys_to_remove:
            del preprocessing_cache[key]

    def clear(self, category: Optional[str] = None) -> None:
        """Clear all caches or a specific category.

        Parameters
        ----------
        category : Optional[str], optional
            Category to clear. If None, clears all categories and resets stats
        """
        if category is None:
            for cache in self._caches.values():
                cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        else:
            if category in self._caches:
                self._caches[category].clear()
            else:
                warnings.warn(f"Invalid cache category: {category}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing cache statistics including hits, misses,
            evictions, sizes, and hit rate
        """
        stats = self._stats.copy()

        # Add cache sizes
        stats["preprocessing_cache_size"] = len(self._caches["preprocessing"])
        stats["minmax_cache_size"] = len(self._caches["minmax"])
        stats["transform_cache_size"] = len(self._caches["transform"])
        stats["total_cache_entries"] = sum(
            len(cache) for cache in self._caches.values()
        )

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0

        return stats

    def print_stats(self) -> None:
        """Print cache performance statistics in a formatted way."""
        stats = self.get_stats()

        print("=== Cache Performance Statistics ===")
        print(f"Total cache entries: {stats['total_cache_entries']}")
        print(f"  - Preprocessing cache: {stats['preprocessing_cache_size']}")
        print(f"  - Min/Max cache: {stats['minmax_cache_size']}")
        print(f"  - Transform cache: {stats['transform_cache_size']}")
        print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        print(f"Cache hits: {stats['hits']}")
        print(f"Cache misses: {stats['misses']}")
        print(f"Cache evictions: {stats['evictions']}")
        print("====================================")

        # Performance assessment
        if stats["hit_rate"] > 0.7:
            print("✓ Excellent cache performance!")
        elif stats["hit_rate"] > 0.5:
            print("✓ Good cache performance")
        elif stats["hit_rate"] > 0.3:
            print("⚠ Moderate cache performance - consider increasing cache size")
        else:
            print("⚠ Low cache performance - data may be changing frequently")

    def generate_cache_key(
        self, widget_prefix: str, kwargs: dict, data_id: Optional[int] = None
    ) -> Tuple:
        """Generate a cache key for preprocessing based on widget states.

        Parameters
        ----------
        widget_prefix : str
            Prefix for widget names
        kwargs : dict
            Current widget values
        data_id : Optional[int], optional
            Data identifier for more specific caching

        Returns
        -------
        Tuple
            Cache key tuple
        """
        key_components = [
            widget_prefix,
            kwargs.get(f"{widget_prefix}_combine_channels", False),
            kwargs.get(f"{widget_prefix}_denoise_components", 0),
            kwargs.get(f"{widget_prefix}_average_repetitions", False),
            kwargs.get(f"{widget_prefix}_spec_part", "Magnitude"),
        ]

        if data_id is not None:
            key_components.insert(0, data_id)

        return tuple(key_components)

    def generate_minmax_key(
        self, data_id: int, spec_part_val: Optional[str], data_shape: Tuple
    ) -> Tuple:
        """Generate a cache key for min/max values.

        Parameters
        ----------
        data_id : int
            ID of the data array
        spec_part_val : Optional[str]
            Data transformation type (e.g., 'Magnitude', 'Phase')
        data_shape : Tuple
            Shape of the data array

        Returns
        -------
        Tuple
            Cache key tuple
        """
        return (data_id, spec_part_val, data_shape)

    def generate_transform_key(
        self, data_id: int, spec_part_val: str, data_shape: Tuple
    ) -> Tuple:
        """Generate a cache key for transformed data.

        Parameters
        ----------
        data_id : int
            ID of the data array
        spec_part_val : str
            Transformation type ('Magnitude', 'Phase', 'Real', 'Imaginary')
        data_shape : Tuple
            Shape of the data array

        Returns
        -------
        Tuple
            Cache key tuple
        """
        return (data_id, spec_part_val, data_shape)
