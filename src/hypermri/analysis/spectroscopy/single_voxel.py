"""Utilities for plotting and manipulating single-voxel spectroscopy data."""

# This file contains utility functions for single voxel spectroscopy methods such as PRESS, STEAM, semi-lASER, ISIS and their
# multi-voxel implementations mv-PRESS, mv-semi-LASER

import numpy as np
from matplotlib.patches import Rectangle

from ...utils.system.logging import LOG_MODES, init_default_logger

logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


def define_extent_PRESS(img_instance):
    """
    Calculate the extent of a anatomical image from the method file
    To be used on data of type: axial, coronal or sagittal

    Returns
    -------
    extent, np.array
    e.g.:
    """
    if img_instance.method["PVM_Fov"].shape == (2,):
        PVM_Fov = img_instance.method["PVM_Fov"]
        img_instance.method["PVM_SPackArrSliceOffset"]
        read_offset = img_instance.method["PVM_SPackArrReadOffset"]
        phase_offset = img_instance.method["PVM_SPackArrPhase1Offset"]
        read_orient = img_instance.method["PVM_SPackArrReadOrient"]
        if img_instance.method["PVM_SPackArrSliceOrient"] == "coronal":
            if read_orient == "H_F":
                # coronal is xz plane
                # z is read, x is phase and y (into the plane) is slice
                # FOV(0) is in z dimension (up down on image in pv)
                # FOV(1) is in x dimension (left right on image in pv)
                return [
                    # [1] is x direction, i.e. phase
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    # [0] is y direction,i.e. read offset in coronal
                    PVM_Fov[0] / 2 + read_offset,
                ]
            elif read_orient == "L_R":
                # x is read, z is phase and y (into the plane) is slice
                # FOV(0) is in x dimension (left right on image in pv)
                # FOV(1) is in z dimension (up down on image in pv)
                # return array should be flipped, i.e. in case we had (-5,5,-3,3) before
                # we now go from (-3,3,-5,5)
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            else:
                raise KeyError("This read orientation is not known for coronal")
        elif img_instance.method["PVM_SPackArrSliceOrient"] == "axial":
            if read_orient == "L_R":
                # axial is xy plane
                # y is phase
                # x is read
                # z is slice
                # FOV[0] is x dimension, left right on image
                # FOV[1] is y dimension, into the image
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            elif read_orient == "A_P":
                # phase encoding flipped
                # axial is xy plane
                # y is read
                # x is phase
                # z is slice
                # FOV[0] is y dimension, left right on image
                # FOV[1] is x dimension, into the image
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                raise KeyError("This read orientation is not know for axial")
        elif img_instance.method["PVM_SPackArrSliceOrient"] == "sagittal":
            # sagittal is yz plane
            if read_orient == "H_F":
                # %FIXME THIS IS NOT RIGHT YET; NEEDS TO BE TESTED IN PHANTOM
                # phase is in y direction
                # read in z direction
                # FOV[0] is in z direction
                # FOV[1] in y direction
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]

    elif img_instance.method["PVM_Fov"].shape == (3,):
        # TODO adapt this in case it does not work properly for 3D FLASH
        # acqusitions
        PVM_Fov = img_instance.method["PVM_Fov"]
        img_instance.method["PVM_SPackArrSliceOffset"]
        read_offset = img_instance.method["PVM_SPackArrReadOffset"]
        phase_offset = img_instance.method["PVM_SPackArrPhase1Offset"]
        read_orient = img_instance.method["PVM_SPackArrReadOrient"]
        if img_instance.method["PVM_SPackArrSliceOrient"] == "coronal":
            if read_orient == "H_F":
                # coronal is xz plane
                # z is read, x is phase and y (into the plane) is slice
                # FOV(0) is in z dimension (up down on image in pv)
                # FOV(1) is in x dimension (left right on image in pv)

                return [
                    # [1] is x direction, i.e. phase
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    # [0] is y direction,i.e. read offset in coronal
                    PVM_Fov[0] / 2 + read_offset,
                ]
            elif read_orient == "L_R":
                # x is read, z is phase and y (into the plane) is slice
                # FOV(0) is in x dimension (left right on image in pv)
                # FOV(1) is in z dimension (up down on image in pv)
                # return array should be flipped, i.e. in case we had (-5,5,-3,3) before
                # we now go from (-3,3,-5,5)
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            else:
                raise KeyError("This read orientation is not known for coronal")
        elif img_instance.method["PVM_SPackArrSliceOrient"] == "axial":
            if read_orient == "L_R":
                # axial is xy plane
                # y is phase
                # x is read
                # z is slice
                # FOV[0] is x dimension, left right on image
                # FOV[1] is y dimension, into the image
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            elif read_orient == "A_P":
                # phase encoding flipped
                # axial is xy plane
                # y is read
                # x is phase
                # z is slice
                # FOV[0] is y dimension, left right on image
                # FOV[1] is x dimension, into the image
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                raise KeyError("This read orientation is not know for axial")
        elif img_instance.method["PVM_SPackArrSliceOrient"] == "sagittal":
            # sagittal is yz plane
            if read_orient == "H_F":
                # %FIXME THIS IS NOT RIGHT YET; NEEDS TO BE TESTED IN PHANTOM
                # phase is in y direction
                # read in z direction
                # FOV[0] is in z direction
                # FOV[1] in y direction
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
        else:
            pass
    else:
        return None


def plot_voxel_on_anat(
    sv_spec_instance,
    img_instance,
    axis,
    voxel_num=0,
    vmin=False,
    vmax=False,
    vox_color=False,
    plot_number=False,
):
    """
    Plots voxel positions on anatomical data
    Parameters
    ----------
    img_instance: Bruker_Exp instance of type image
    voxel_num: Number of voxel to be plotted, if this is a single-voxel method leave this at 0
    axis: axis to be plotted into, i.e. "ax" from fig,ax=plt.subplots(1)
    vmin, vmax: int, for windowing of plot, e.g. 0.1,25
    vox_color: color of voxel to be plotted
    Returns
    -------
    img_array:

    """

    # meta data needed:
    vox_pos = sv_spec_instance.method["PVM_VoxArrPosition"]
    sv_spec_instance.method["PVM_VoxArrSize"]

    sv_spec_instance.method["PVM_VoxArrSize"]
    pat_pos = img_instance.acqp["ACQ_patient_pos"]
    # slice positions
    if img_instance.method["PVM_SpatDimEnum"] == "<3D>":
        slice_positions = img_instance.method["PVM_SliceOffset"]
    else:
        slice_positions = img_instance.method["PVM_SliceOffset"]
    if pat_pos == "Head_Prone":
        if img_instance.method["PVM_SPackArrSliceOrient"] == "axial":
            slice_positions = np.flip(slice_positions, axis=0)
        elif img_instance.method["PVM_SPackArrSliceOrient"] == "coronal":
            slice_positions = np.flip(slice_positions, axis=0)
    if pat_pos == "Head_Supine":
        # if img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
        #    slice_positions=np.flip(slice_positions,axis=0)
        # else:
        pass
    img_data = np.squeeze(
        img_instance.seq2d
    )  # remove axis of length 0 (i.e. if we have no repetitions)
    img_instance.method
    # FIXME possibly have to call the Orient_2dseq function here in the future
    if img_instance.method["PVM_SPackArrSliceOrient"] == "coronal":
        slice_for_vox = np.argmin(abs(slice_positions - (vox_pos[voxel_num, 1])))
        img_type = "Cor"
        ax_x = "x"
        ax_y = "z"
    elif img_instance.method["PVM_SPackArrSliceOrient"] == "axial":
        slice_for_vox = np.argmin(abs(slice_positions - (vox_pos[voxel_num, 2])))
        img_type = "Ax"
        ax_x = "x"
        ax_y = "y"
    elif img_instance.method["PVM_SPackArrSliceOrient"] == "sagittal":
        slice_for_vox = np.argmin(abs(slice_positions - (vox_pos[voxel_num, 0])))
        img_type = "Sag"
        ax_x = "y"
        ax_y = "z"
    else:
        pass
    from ...analysis.anatomical import Define_Extent_press

    imshow_params = {
        "X": (img_data[:, :, slice_for_vox].T),
        "extent": Define_Extent_press(img_instance),
        "cmap": "gray",
    }

    if vmin == 0:
        vmin = 0.001
    if vmin and vmax:
        imshow_params["vmin"] = vmin
        imshow_params["vmax"] = vmax
    else:
        pass
    from ...analysis.anatomical import Get_Voxel_patch_params

    patch_params = Get_Voxel_patch_params(sv_spec_instance, img_instance, voxel_num)
    slice_title = (
        str(img_type)
        + "["
        + str(slice_for_vox + 1)
        + "/"
        + str(img_data.shape[2])
        + "]"
    )

    patch_params["edgecolor"] = "w"
    if not vox_color:
        vox_color = "r"
    patch_params["facecolor"] = vox_color
    if plot_number == True:
        xy = patch_params["xy"]
        width = patch_params["width"]
        height = patch_params["height"]
        center_x = xy[0] + width / 2
        center_y = xy[1] + height / 2

        # Add the label at the center of the rectangle
        axis.text(
            center_x,
            center_y,
            voxel_num,
            ha="center",
            va="center",
            fontsize=11,
            color="w",
        )

    patch_params["alpha"] = 0.8
    # plotting
    axis.imshow(**imshow_params)
    axis.add_patch(Rectangle(**patch_params))
    axis.set_xlabel(str(ax_x) + "[mm]")
    axis.set_ylabel(str(ax_y) + "[mm]")
    axis.set_title(slice_title)


def show_voxel_overlap(centers, sizes):
    """
    Computes the overlap of voxels in a multi-voxel sequence like mv-press or mv-sLASER and shows it as a pandas dataframe.
    Written using ChatGPT 4o.

    Parameters
    ----------
    centers: np.array, voxel locations, i.e. mrs.method['PVM_VoxArrPosition']
    sizes: np.array,  method['PVM_VoxArrSize']

    Returns
    -------
    overlap_x_data
    overlap_y_data
    overlap_z_data
    total_overlap_data
    styled_df_x
    styled_df_y
    styled_df_z
    styled_df_total
    """

    def compute_bounds(center, size):
        """Calculate the minimum and maximum bounds of a voxel."""
        half_size = size / 2.0
        min_bound = center - half_size
        max_bound = center + half_size
        return min_bound, max_bound

    def overlap_amount(min1, max1, min2, max2):
        """Calculate the overlap between two intervals."""
        if max1 > min2 and min1 < max2:  # Overlap condition
            return min(max1, max2) - max(min1, min2)
        return 0  # No overlap

    def find_overlaps(centers, sizes):
        """Find overlaps between voxels given their centers and sizes."""
        n = len(centers)
        overlap_x_data = np.zeros((n, n))
        overlap_y_data = np.zeros((n, n))
        overlap_z_data = np.zeros((n, n))
        total_overlap_data = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Get bounds for voxel i
                min_i, max_i = compute_bounds(centers[i], sizes[i])
                # Get bounds for voxel j
                min_j, max_j = compute_bounds(centers[j], sizes[j])

                # Check overlaps in each dimension
                overlap_x = overlap_amount(min_i[0], max_i[0], min_j[0], max_j[0])
                overlap_y = overlap_amount(min_i[1], max_i[1], min_j[1], max_j[1])
                overlap_z = overlap_amount(min_i[2], max_i[2], min_j[2], max_j[2])

                # Fill the matrices symmetrically
                overlap_x_data[i, j] = round(overlap_x, 1)
                overlap_x_data[j, i] = round(overlap_x, 1)
                overlap_y_data[i, j] = round(overlap_y, 1)
                overlap_y_data[j, i] = round(overlap_y, 1)
                overlap_z_data[i, j] = round(overlap_z, 1)
                overlap_z_data[j, i] = round(overlap_z, 1)

                # Calculate the average percentage overlap across dimensions
                size_i = np.array(sizes[i])
                size_j = np.array(sizes[j])
                avg_size = (size_i + size_j) / 2.0

                overlap_x_percentage = (
                    (overlap_x / avg_size[0]) * 100 if avg_size[0] > 0 else 0
                )
                overlap_y_percentage = (
                    (overlap_y / avg_size[1]) * 100 if avg_size[1] > 0 else 0
                )
                overlap_z_percentage = (
                    (overlap_z / avg_size[2]) * 100 if avg_size[2] > 0 else 0
                )

                total_overlap_percentage = (
                    overlap_x_percentage + overlap_y_percentage + overlap_z_percentage
                ) / 3
                total_overlap_data[i, j] = round(total_overlap_percentage, 1)
                total_overlap_data[j, i] = round(total_overlap_percentage, 1)

        return (
            overlap_x_data,
            overlap_y_data,
            overlap_z_data,
            total_overlap_data,
            avg_size,
        )

    def visualize_overlaps(
        overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size
    ):
        """Visualize the overlap matrices using color-coded DataFrames."""
        import pandas as pd
        from IPython.display import display

        n = overlap_x_data.shape[0]
        index_labels = [f"voxel {i}" for i in range(n)]

        # Create DataFrames for each axis and the total overlap
        df_x = pd.DataFrame(overlap_x_data, columns=index_labels, index=index_labels)
        df_y = pd.DataFrame(overlap_y_data, columns=index_labels, index=index_labels)
        df_z = pd.DataFrame(overlap_z_data, columns=index_labels, index=index_labels)
        df_total = pd.DataFrame(
            total_overlap_data, columns=index_labels, index=index_labels
        )

        # Define a color map for the overlap ranges
        def color_map(val):
            if val > 50:
                color = "red"
            elif val > 0:
                color = "orange"
            else:
                color = "green"
            return f"background-color: {color}"

        # Style the DataFrames

        styled_df_x = (
            df_x.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in X Axis")
        )
        styled_df_y = (
            df_y.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in Y Axis")
        )
        styled_df_z = (
            df_z.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in Z Axis")
        )
        styled_df_total = (
            df_total.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Total Overlap Percentage")
        )

        # Display DataFrames (for Jupyter Notebooks)
        display(styled_df_x)
        display(styled_df_y)
        display(styled_df_z)
        display(styled_df_total)

        return styled_df_x, styled_df_y, styled_df_z, styled_df_total

    # Step 1: Calculate overlaps
    overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size = (
        find_overlaps(centers, sizes)
    )

    # Step 2: Visualize overlaps
    styled_df_x, styled_df_y, styled_df_z, styled_df_total = visualize_overlaps(
        overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size
    )
    return (
        overlap_x_data,
        overlap_y_data,
        overlap_z_data,
        total_overlap_data,
        styled_df_x,
        styled_df_y,
        styled_df_z,
        styled_df_total,
    )
