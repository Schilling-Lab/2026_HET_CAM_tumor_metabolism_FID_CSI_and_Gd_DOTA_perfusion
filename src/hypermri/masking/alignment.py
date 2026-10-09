"""
Coordinate alignment utilities for cross-dataset masking and overlay.

This module provides tools for aligning different MRI datasets spatially,
leveraging the existing coordinate transformation infrastructure in hypermri.
"""

from typing import Dict

import numpy as np

from hypermri.utils.coordinates.grids import get_xyz_grid
from hypermri.utils.system.logging import init_default_logger

logger = init_default_logger(__name__)


class CoordinateAligner:
    """Leverage existing coordinate infrastructure for dataset alignment.

    This class provides utilities for analyzing and aligning coordinate systems
    between different MRI datasets, building on the existing coordinate
    transformation infrastructure in hypermri.utils.coordinates.
    """

    def __init__(self):
        """Initialize the coordinate aligner."""
        self._alignment_cache = {}

    def align_datasets(
        self, source_exp, target_exp, method: str = "scanner_coords"
    ) -> Dict:
        """Align datasets using scanner or object coordinates.

        Parameters
        ----------
        source_exp : MRIExperiment or BaseAnatomy
            Source dataset (where mask was created)
        target_exp : MRIExperiment or BaseAnatomy
            Target dataset (where mask will be applied)
        method : str, default "scanner_coords"
            Alignment method: "scanner_coords" or "object_coords"

        Returns
        -------
        dict
            Alignment information including transformation parameters
        """
        cache_key = f"{id(source_exp)}_{id(target_exp)}_{method}"
        if cache_key in self._alignment_cache:
            logger.debug("Using cached alignment result")
            return self._alignment_cache[cache_key]

        try:
            if method == "scanner_coords":
                result = self._align_scanner_coordinates(source_exp, target_exp)
            elif method == "object_coords":
                result = self._align_object_coordinates(source_exp, target_exp)
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            # Cache the result
            self._alignment_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error aligning datasets: {e}")
            raise

    def _align_scanner_coordinates(self, source_exp, target_exp) -> Dict:
        """Align datasets using XYZ scanner coordinates."""
        # Get XYZ coordinates for both datasets
        source_xyz = self._get_xyz_coordinates(source_exp)
        target_xyz = self._get_xyz_coordinates(target_exp)

        if source_xyz is None or target_xyz is None:
            raise ValueError("Could not generate XYZ coordinates for alignment")

        # Calculate alignment parameters
        alignment_info = {
            "method": "scanner_coords",
            "source_xyz_range": {},
            "target_xyz_range": {},
            "overlap_region": {},
            "scale_factors": {},
            "offset_vectors": {},
            "alignment_quality": None,
        }

        # Calculate coordinate ranges and overlaps
        for dim in ["x", "y", "z"]:
            if dim in source_xyz and dim in target_xyz:
                source_range = (source_xyz[dim].min(), source_xyz[dim].max())
                target_range = (target_xyz[dim].min(), target_xyz[dim].max())

                alignment_info["source_xyz_range"][dim] = source_range
                alignment_info["target_xyz_range"][dim] = target_range

                # Calculate overlap region
                overlap_min = max(source_range[0], target_range[0])
                overlap_max = min(source_range[1], target_range[1])

                if overlap_max > overlap_min:
                    alignment_info["overlap_region"][dim] = (overlap_min, overlap_max)
                    overlap_fraction = (overlap_max - overlap_min) / max(
                        source_range[1] - source_range[0],
                        target_range[1] - target_range[0],
                    )
                else:
                    alignment_info["overlap_region"][dim] = None

                # Calculate scale and offset
                source_span = source_range[1] - source_range[0]
                target_span = target_range[1] - target_range[0]

                scale_factor = target_span / source_span if source_span > 0 else 1.0
                offset = target_range[0] - source_range[0]

                alignment_info["scale_factors"][dim] = scale_factor
                alignment_info["offset_vectors"][dim] = offset

        # Calculate overall alignment quality
        overlap_scores = []
        for dim in ["x", "y", "z"]:
            if alignment_info["overlap_region"].get(dim) is not None:
                source_span = (
                    alignment_info["source_xyz_range"][dim][1]
                    - alignment_info["source_xyz_range"][dim][0]
                )
                target_span = (
                    alignment_info["target_xyz_range"][dim][1]
                    - alignment_info["target_xyz_range"][dim][0]
                )
                overlap_span = (
                    alignment_info["overlap_region"][dim][1]
                    - alignment_info["overlap_region"][dim][0]
                )

                overlap_score = overlap_span / max(source_span, target_span)
                overlap_scores.append(overlap_score)
            else:
                overlap_scores.append(0.0)

        alignment_info["alignment_quality"] = np.mean(overlap_scores)

        return alignment_info

    def _align_object_coordinates(self, source_exp, target_exp) -> Dict:
        """Align datasets using object coordinates (read_idx, phase_idx, slice_idx)."""
        source_data = self._get_data_array(source_exp)
        target_data = self._get_data_array(target_exp)

        if source_data is None or target_data is None:
            raise ValueError(
                "Could not access data arrays for object coordinate alignment"
            )

        alignment_info = {
            "method": "object_coords",
            "source_shape": source_data.shape,
            "target_shape": target_data.shape,
            "source_dims": list(source_data.dims),
            "target_dims": list(target_data.dims),
            "common_dims": list(set(source_data.dims) & set(target_data.dims)),
            "dimension_mapping": {},
            "alignment_quality": None,
        }

        # Create dimension mapping
        common_dims = alignment_info["common_dims"]

        for dim in common_dims:
            source_size = source_data.sizes.get(dim, 0)
            target_size = target_data.sizes.get(dim, 0)

            alignment_info["dimension_mapping"][dim] = {
                "source_size": source_size,
                "target_size": target_size,
                "size_ratio": target_size / source_size if source_size > 0 else 1.0,
            }

        # Calculate alignment quality based on dimension compatibility
        if common_dims:
            size_ratios = [
                alignment_info["dimension_mapping"][dim]["size_ratio"]
                for dim in common_dims
            ]
            # Quality is higher when size ratios are closer to 1.0
            quality_scores = [1.0 / max(ratio, 1.0 / ratio) for ratio in size_ratios]
            alignment_info["alignment_quality"] = np.mean(quality_scores)
        else:
            alignment_info["alignment_quality"] = 0.0

        return alignment_info

    def _get_xyz_coordinates(self, experiment):
        """Get XYZ coordinates for an experiment using existing infrastructure."""
        try:
            # Use existing coordinate generation functions
            if hasattr(experiment, "get_xyz_grid"):
                return experiment.get_xyz_grid()
            elif (
                hasattr(experiment, "data_array") and experiment.data_array is not None
            ):
                # Use the coordinate utilities directly
                return get_xyz_grid(experiment.data_array, experiment.method)
            elif hasattr(experiment, "reco_data") and experiment.reco_data is not None:
                return get_xyz_grid(experiment.reco_data, experiment.method)
            else:
                logger.warning("No suitable data array found for coordinate generation")
                return None
        except Exception as e:
            logger.error(f"Error generating XYZ coordinates: {e}")
            return None

    def _get_data_array(self, experiment):
        """Get the main data array from an experiment."""
        if hasattr(experiment, "data_array") and experiment.data_array is not None:
            return experiment.data_array
        elif hasattr(experiment, "reco_data") and experiment.reco_data is not None:
            return experiment.reco_data
        else:
            return None

    def check_alignment_quality(self, source_exp, target_exp) -> Dict:
        """Analyze spatial alignment between datasets.

        Parameters
        ----------
        source_exp : MRIExperiment or BaseAnatomy
            Source dataset
        target_exp : MRIExperiment or BaseAnatomy
            Target dataset

        Returns
        -------
        dict
            Comprehensive alignment quality analysis
        """
        analysis = {
            "scanner_coords_available": False,
            "object_coords_available": False,
            "recommended_method": "simple",
            "alignment_feasibility": "poor",
            "scanner_alignment": None,
            "object_alignment": None,
            "warnings": [],
        }

        try:
            # Check scanner coordinate alignment
            try:
                scanner_alignment = self._align_scanner_coordinates(
                    source_exp, target_exp
                )
                analysis["scanner_coords_available"] = True
                analysis["scanner_alignment"] = scanner_alignment

                if scanner_alignment["alignment_quality"] > 0.7:
                    analysis["recommended_method"] = "scanner_coords"
                    analysis["alignment_feasibility"] = "excellent"
                elif scanner_alignment["alignment_quality"] > 0.4:
                    analysis["recommended_method"] = "scanner_coords"
                    analysis["alignment_feasibility"] = "good"

            except Exception as e:
                analysis["warnings"].append(f"Scanner coordinate alignment failed: {e}")

            # Check object coordinate alignment
            try:
                object_alignment = self._align_object_coordinates(
                    source_exp, target_exp
                )
                analysis["object_coords_available"] = True
                analysis["object_alignment"] = object_alignment

                if (
                    analysis["recommended_method"] == "simple"
                    and object_alignment["alignment_quality"] > 0.5
                ):
                    analysis["recommended_method"] = "object_coords"
                    analysis["alignment_feasibility"] = "fair"

            except Exception as e:
                analysis["warnings"].append(f"Object coordinate alignment failed: {e}")

            # Final assessment
            if (
                not analysis["scanner_coords_available"]
                and not analysis["object_coords_available"]
            ):
                analysis["alignment_feasibility"] = "poor"
                analysis["warnings"].append("No coordinate alignment methods available")

        except Exception as e:
            logger.error(f"Error in alignment quality check: {e}")
            analysis["warnings"].append(f"Alignment analysis failed: {e}")

        return analysis

    def suggest_alignment_strategy(self, source_exp, target_exp) -> Dict:
        """Suggest the best alignment strategy for two datasets.

        Parameters
        ----------
        source_exp : MRIExperiment or BaseAnatomy
            Source dataset
        target_exp : MRIExperiment or BaseAnatomy
            Target dataset

        Returns
        -------
        dict
            Recommended alignment strategy and parameters
        """
        quality_analysis = self.check_alignment_quality(source_exp, target_exp)

        strategy = {
            "recommended_method": quality_analysis["recommended_method"],
            "confidence": quality_analysis["alignment_feasibility"],
            "parameters": {},
            "warnings": quality_analysis["warnings"],
            "fallback_methods": [],
        }

        # Add method-specific parameters
        if strategy["recommended_method"] == "scanner_coords":
            if quality_analysis["scanner_alignment"]:
                strategy["parameters"] = {
                    "interpolation_method": "linear",
                    "alignment_quality": quality_analysis["scanner_alignment"][
                        "alignment_quality"
                    ],
                }
        elif strategy["recommended_method"] == "object_coords":
            if quality_analysis["object_alignment"]:
                strategy["parameters"] = {
                    "interpolation_method": "nearest",
                    "common_dims": quality_analysis["object_alignment"]["common_dims"],
                }
        else:
            strategy["parameters"] = {
                "interpolation_method": "nearest",
                "note": "Simple shape-based alignment only",
            }

        # Add fallback methods
        if quality_analysis["scanner_coords_available"]:
            strategy["fallback_methods"].append("scanner_coords")
        if quality_analysis["object_coords_available"]:
            strategy["fallback_methods"].append("object_coords")
        if "simple" not in strategy["fallback_methods"]:
            strategy["fallback_methods"].append("simple")

        return strategy
