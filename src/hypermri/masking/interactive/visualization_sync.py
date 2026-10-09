"""Simple synchronization between masks and plot overlays."""

from typing import Any, Dict, List

import numpy as np
from matplotlib.patches import Polygon

from .mask_container import MaskContainer


class VisualizationSync:
    """
    Simple synchronization between mask data and plotting system overlays.

    Key principle: Since mask coordinates are identical to data coordinates,
    sync is straightforward without transforms.
    """

    def __init__(self):
        """
        Initialize visualization sync.
        """
        # Track active overlays for cleanup
        self.active_overlays: Dict[str, Dict[str, Any]] = {}

    def sync_mask_to_plotter(
        self, mask_container: MaskContainer, plotter, show_only_active: bool = True
    ):
        """
        Sync mask to plotter overlays - same coordinates, no transforms.

        Parameters
        ----------
        mask_container : MaskContainer
            Mask to sync
        plotter : AnatomyPlotter
            Plotter instance to sync to
        show_only_active : bool, default True
            If True, clear other overlays and show only this mask.
            If False, add this mask to existing overlays.
        """
        try:
            if not hasattr(plotter, "interactive_overlays"):
                return False

            # Clear existing overlays only if showing only active
            if show_only_active:
                len(plotter.interactive_overlays)
                plotter.interactive_overlays.clear()

            # Since coordinates are identical, direct assignment
            overlay_data = {
                "buffer": mask_container.buffer,
                "color": mask_container.color,
                "alpha": mask_container.alpha,
                "finalized": mask_container.finalized,
            }

            # Direct assignment to plotter overlays
            plotter.interactive_overlays[mask_container.name] = overlay_data
            self.active_overlays[mask_container.name] = overlay_data.copy()

            mask_container.get_total_pixels()

            return True

        except Exception:
            return False

    def sync_all_finalized_masks(self, masks_dict: Dict[str, MaskContainer], plotter):
        """
        Sync all finalized masks to plotter for persistent display.

        This ensures finalized masks remain visible when switching slices,
        even when no mask is actively being edited.

        Parameters
        ----------
        masks_dict : Dict[str, MaskContainer]
            Dictionary of all masks
        plotter : AnatomyPlotter
            Plotter instance to sync to
        """
        try:
            if not hasattr(plotter, "interactive_overlays"):
                return False

            # CORRECTED: Don't clear existing overlays - preserve active masks being drawn
            # Instead, selectively update/add only finalized masks
            len(plotter.interactive_overlays)

            synced_count = 0
            for name, mask_container in masks_dict.items():
                if mask_container.finalized:
                    # Sync finalized mask without clearing others
                    success = self.sync_mask_to_plotter(
                        mask_container, plotter, show_only_active=False
                    )
                    if success:
                        synced_count += 1

            len(plotter.interactive_overlays)
            return synced_count > 0

        except Exception:
            return False

    def clear_plotter_overlays(self, plotter):
        """
        Clear all overlays from plotter.

        Parameters
        ----------
        plotter : AnatomyPlotter
            Plotter to clear overlays from
        """
        try:
            if hasattr(plotter, "interactive_overlays"):
                len(plotter.interactive_overlays)
                plotter.interactive_overlays.clear()
                self.active_overlays.clear()
                return True
            else:
                return False

        except Exception:
            return False

    def add_lasso_visual_patches(
        self, ax, verts_array: np.ndarray, mask_container: MaskContainer
    ) -> List[Any]:
        """
        Add visual patches for lasso selection with dual patch system.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes
        verts_array : np.ndarray
            Lasso vertices
        mask_container : MaskContainer
            Mask container for styling

        Returns
        -------
        List[Any]
            List of created patches for cleanup
        """
        try:
            patches = []

            # 1. Filled patch with mask color and transparency
            filled_patch = Polygon(
                verts_array,
                closed=True,
                facecolor=mask_container.color,
                alpha=mask_container.alpha,
                edgecolor="none",
                zorder=9,  # Below outline patch
            )
            ax.add_patch(filled_patch)
            patches.append(filled_patch)

            # 2. Outline patch with yellow dashed style for visual feedback
            outline_patch = Polygon(
                verts_array,
                closed=True,
                facecolor="none",
                edgecolor="yellow",
                linewidth=1,
                linestyle="--",
                zorder=10,  # Above filled patch
            )
            ax.add_patch(outline_patch)
            patches.append(outline_patch)

            # Store patches in mask container for cleanup
            mask_container.visual_patches.extend(patches)

            return patches

        except Exception:
            return []

    def remove_last_visual_patches(
        self, mask_container: MaskContainer, patch_count: int = 2
    ) -> bool:
        """
        Remove last N visual patches from mask container.

        Parameters
        ----------
        mask_container : MaskContainer
            Mask container to remove patches from
        patch_count : int, default 2
            Number of patches to remove (filled + outline = 2)

        Returns
        -------
        bool
            True if patches were removed
        """
        try:
            patches = mask_container.visual_patches

            if len(patches) < patch_count:
                return False

            # Remove last N patches
            patches_to_remove = patches[-patch_count:]
            removed_count = 0

            for patch in patches_to_remove:
                try:
                    if hasattr(patch, "remove") and patch.axes:
                        patch.remove()
                        removed_count += 1
                except:
                    pass  # Ignore removal errors

            # Remove from list
            mask_container.visual_patches = patches[:-patch_count]

            return removed_count > 0

        except Exception:
            return False

    def clear_all_visual_patches(self, mask_container: MaskContainer) -> int:
        """
        Clear all visual patches from mask container.

        Parameters
        ----------
        mask_container : MaskContainer
            Mask container to clear patches from

        Returns
        -------
        int
            Number of patches cleared
        """
        try:
            patches = mask_container.visual_patches[:]  # Copy list
            cleared_count = 0

            for patch in patches:
                try:
                    if hasattr(patch, "remove") and patch.axes:
                        patch.remove()
                        cleared_count += 1
                except:
                    pass  # Ignore removal errors

            mask_container.visual_patches.clear()

            return cleared_count

        except Exception:
            return 0

    def force_canvas_refresh(self, ax):
        """
        Force canvas refresh for immediate visual updates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes to refresh
        """
        try:
            if hasattr(ax, "figure") and hasattr(ax.figure, "canvas"):
                ax.figure.canvas.draw_idle()
                return True
            else:
                return False

        except Exception:
            return False

    def setup_enhanced_lasso_visibility(self, selector):
        """
        Apply enhanced visibility properties to lasso selector.

        Parameters
        ----------
        selector : LassoSelector
            Lasso selector to enhance

        Returns
        -------
        bool
            True if enhancement applied
        """
        try:
            # Set enhanced line properties
            enhanced_props = {
                "color": "red",
                "linestyle": "-",
                "linewidth": 4,
                "alpha": 1.0,
            }

            # Apply to selection artist if available
            if hasattr(selector, "_selection_artist") and selector._selection_artist:
                for prop, value in enhanced_props.items():
                    if hasattr(selector._selection_artist, f"set_{prop}"):
                        getattr(selector._selection_artist, f"set_{prop}")(value)

            # Apply to any other artists
            if hasattr(selector, "artists"):
                for artist in selector.artists:
                    for prop, value in enhanced_props.items():
                        if hasattr(artist, f"set_{prop}"):
                            getattr(artist, f"set_{prop}")(value)

            return True

        except Exception:
            return False

    def validate_overlay_sync(self, mask_container: MaskContainer, plotter) -> bool:
        """
        Validate that mask is properly synced to plotter.

        Parameters
        ----------
        mask_container : MaskContainer
            Mask to validate
        plotter : AnatomyPlotter
            Plotter to validate against

        Returns
        -------
        bool
            True if sync is valid
        """
        try:
            if not hasattr(plotter, "interactive_overlays"):
                return False

            if mask_container.name not in plotter.interactive_overlays:
                return False

            overlay = plotter.interactive_overlays[mask_container.name]

            # Validate overlay has required keys
            required_keys = ["buffer", "color", "alpha", "finalized"]
            missing_keys = [k for k in required_keys if k not in overlay]

            if missing_keys:
                return False

            # Validate buffer is the same object (identical coordinates guaranteed)
            if overlay["buffer"] is not mask_container.buffer:
                return False

            return True

        except Exception:
            return False
