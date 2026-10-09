##########################################################################

# Plot boundaries of spectroscopy slice in an anatomical image -
# compatible with spec and CSI slice

##########################################################################


# =========================================================================================================================================================
# Import of necessary packages
# =========================================================================================================================================================

import glob
import os

import imageio
import ipywidgets as widgets
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm

from ...analysis.anatomical.core import add_scalebar

# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================


def get_extent(mri_experiment, specimen="rodent"):
    read_offset = mri_experiment.method["PVM_SPackArrReadOffset"]
    phase_offset = mri_experiment.method["PVM_SPackArrPhase1Offset"]

    mm_read = mri_experiment.method["PVM_Fov"][0]
    mm_phase = mri_experiment.method["PVM_Fov"][1]

    # mm_read_pixel_size = mri_experiment.method['PVM_SpatResol'][0]
    # mm_phase_pixel_size = mri_experiment.method['PVM_SpatResol'][1]

    if mri_experiment.method["PVM_SPackArrSliceOrient"] == "axial":
        if specimen == "rodent":
            # Conventional image display: turn by 90 degrees counterclockwise
            extent = [
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
            ]
        elif specimen == "other":
            # Conventional image display: turn by 90 degrees counterclockwise
            extent = [
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
            ]

    elif mri_experiment.method["PVM_SPackArrSliceOrient"] == "coronal":
        if specimen == "rodent":
            # Conventional image display: turn by 90 degrees clockwise, mirror
            # 3rd axis
            extent = [
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
            ]
        elif specimen == "other":
            # Conventional image display: turn by 90 degrees clockwise, mirror
            # 3rd axis
            extent = [
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
            ]

    elif mri_experiment.method["PVM_SPackArrSliceOrient"] == "sagittal":
        if specimen == "rodent":
            # Conventional image display: turn by 90 degrees clockwise, mirror
            # 2nd axis, 3rd axis
            extent = [
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
            ]
        elif specimen == "other":
            # Conventional image display: turn by 90 degrees clockwise, mirror
            # 2nd axis, 3rd axis
            extent = [
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
            ]

    return extent


def add_colorbar(
    image=None,
    boundaries=None,
    title=None,
    mappable=None,
    color="k",
    fontsize=12,
    box=[0.075, -0.05, 0.2, 0.05],
    shrink=1.0,
    aspect=4,
    pad=-0.075,
    tickpad=0.5,
    titlepad=None,
    xtext=None,
    ytext=0.15,
    strformat="%.2f",
    orientation="horizontal",
    side=False,
):
    """
    Add colorbar to an image,
    provide 'image' or 'boundaries' (dictionary with vmin and vmax).

    Parameters
    ----------
    image: 2D-array, optional
        Input image.
    boundaries: dictionary, optional
        Dictionary that contains vmin and vmax limits.
    title: string, optional
        Colorbar title.
    mappable: plt.mappable, optional
        Mappable of the image to add the colorbar.
    color: string, optional
        Edge and text color.
    fontsize: int, optional
        Text font size.
    box: list, optional
        Subaxis location [start_x, start_y, width, height].
    shrink: float, optional
        Colorbar shrink.
    aspect: float, optional
        Colorbar aspect ratio: horizontal/vertical
    pad, tickpad, titlepad: float, optional
        Pad value for the colorbar, ticks, title.
    xtext, ytext: float, optional
        Position of the tick labels.
    strformat: string, optional
        Tick label number format.
    orientation: string, optional
        Colorbar orientation: 'horizontal' or 'vertical'
    side: bool, optional
        Add limit tick labels on the colorbar side.
    """

    if not boundaries:
        assert not isinstance(image, type(None))
        boundaries = dict(vmin=np.percentile(image, 0), vmax=np.percentile(image, 100))

    cbax = plt.gca().inset_axes(box)

    if side:
        cbar = plt.colorbar(
            cax=cbax,
            shrink=shrink,
            aspect=aspect,
            pad=pad,
            format=strformat,
            orientation=orientation,
            mappable=mappable,
        )

        # cbar = plt.colorbar(ticks=[], cax=cbax, shrink=shrink, aspect=aspect,
        # pad=pad, format=strformat, orientation=orientation, mappable=mappable)
        # cbar.ax.tick_params(labelsize=fontsize, colors='none', direction='out', pad=tickpad)
        # if xtext is None:
        # xtext=[-0.3,1.05]
        # cbar.ax.annotate(strformat %boundaries['vmin'], xy=(0, 0), c=color, xycoords='axes fraction', xytext=(xtext[0], ytext),  textcoords='axes fraction')
        # cbar.ax.annotate(strformat %boundaries['vmax'], xy=(0, 0), c=color, xycoords='axes fraction', xytext=(xtext[1], ytext),  textcoords='axes fraction')

    else:
        cbar = plt.colorbar(
            ticks=[boundaries["vmin"], boundaries["vmax"]],
            cax=cbax,
            shrink=shrink,
            aspect=aspect,
            pad=pad,
            format=strformat,
            orientation=orientation,
            mappable=mappable,
        )
        cbar.ax.tick_params(
            labelsize=fontsize, colors=color, direction="out", pad=tickpad
        )

    if title:
        if titlepad:
            cbar.set_label(title, labelpad=titlepad, color=color, fontsize=fontsize)
        else:
            cbar.ax.set_title(title, color=color, fontsize=fontsize)

    cbar.outline.set_color(color)


def add_patch(
    image,
    patchcenter=None,
    patchbox=None,
    ax=None,
    sl=None,
    color="r",
    lw=2,
    fc="none",
    out=False,
):
    """
    Add rectangular patch to an image.
    If no other variables are passed, adds a patch of the size of the image.

    Parameters
    ----------
    image: 2D-array
        Input image.
    patchcenter: (2,) tuple, optional
        Center of the rectangle.
    patchbox: tuple of ints, optional
        Rectangle side dimensions in pixels: (dy, dx).
    ax: plt.axis, optional
        Axis of the image to add the patch.
    sl: slice, optional
        Slice/ROI of the patch (i.e. np.s_[:,...]).
    color: string, optional
        Patch color.
    lw: int, optional
        Patch line width.
    fc: string, optional
        Patch face color.
    out: bool, optional
        If true returns numpy slice.

    Returns
    -------
    sl : slice
        If out is True returns the slice.
    """

    sh = image.shape

    if isinstance(patchbox, int):
        patchbox = [patchbox, patchbox]

    if isinstance(sl, type(None)):
        if patchbox and patchcenter:
            sl = np.s_[
                patchcenter[0] - patchbox[0] // 2 : patchcenter[0] + patchbox[0] // 2,
                patchcenter[1] - patchbox[1] // 2 : patchcenter[1] + patchbox[1] // 2,
            ]
        elif patchbox[0]:
            sl = np.s_[
                sh[0] // 2 - patchbox[0] // 2 : sh[0] // 2 + patchbox[0] // 2,
                sh[1] // 2 - patchbox[1] // 2 : sh[1] // 2 + patchbox[1] // 2,
            ]
        else:
            sl = np.s_[0 : sh[0], 0 : sh[1]]

    if ax is None:
        ax = plt.gca()
    ax.add_patch(
        patches.Rectangle(
            (sl[1].start, sl[0].start),
            sl[1].stop - sl[1].start,
            sl[0].stop - sl[0].start,
            linewidth=lw,
            edgecolor=color,
            facecolor=fc,
        )
    )
    if out:
        return sl


def plane_x(n, r0, y, z):
    return -(n[1] * (y - r0[1]) + n[2] * (z - r0[2])) / n[0] + r0[0]


def plane_y(n, r0, x, z):
    return -(n[0] * (x - r0[0]) + n[2] * (z - r0[2])) / n[1] + r0[1]


def plane_z(n, r0, x, y):
    return -(n[0] * (x - r0[0]) + n[1] * (y - r0[1])) / n[2] + r0[2]


def get_slice_boundaries_xarray(
    spec_BrukerExp,
    anatomical_BrukerExp,
    anatomical_slice_name,
    anatomical_slice_number,
):
    anat_seq2d_xyz_xarray = anatomical_BrukerExp.seq2d_xyz_xarray
    anat_slice_offset_mm = anat_seq2d_xyz_xarray[anatomical_slice_name].attrs[
        "axis_mm"
    ][anatomical_slice_number]

    spec_slice_thickness_mm = spec_BrukerExp.slice_thickness_mm
    spec_slice_offset_mm = spec_BrukerExp.slice_offset_mm

    n = spec_BrukerExp.acquisition_matrix[2]
    t = spec_slice_offset_mm * n

    spec_slice_boundary_1_center = t - spec_slice_thickness_mm / 2 * n
    spec_slice_boundary_2_center = t + spec_slice_thickness_mm / 2 * n

    # Calculate two points of both slice boundaries in xy, yz and zx plane

    ##########################################################################
    slice_boundary_1_x = np.array([-50, 50])
    slice_boundary_2_x = np.array([-50, 50])

    slice_boundary_1_y = plane_y(
        n,
        spec_slice_boundary_1_center,
        slice_boundary_1_x,
        np.repeat(anat_slice_offset_mm, 2),
    )
    slice_boundary_2_y = plane_y(
        n,
        spec_slice_boundary_2_center,
        slice_boundary_2_x,
        np.repeat(anat_slice_offset_mm, 2),
    )

    if np.any(slice_boundary_1_y == np.inf):
        slice_boundary_1_y = np.array([-50, 50])
        slice_boundary_2_y = np.array([-50, 50])

        slice_boundary_1_x = plane_x(
            n,
            spec_slice_boundary_1_center,
            slice_boundary_1_y,
            np.repeat(anat_slice_offset_mm, 2),
        )
        slice_boundary_2_x = plane_x(
            n,
            spec_slice_boundary_2_center,
            slice_boundary_2_y,
            np.repeat(anat_slice_offset_mm, 2),
        )

    dims_xy = [
        "x",
        "y",
    ]
    coords_xy = {
        "x": np.arange(2),
        "y": np.arange(2),
    }

    spec_slice_boundary_1_xy = xr.DataArray(
        np.vstack([slice_boundary_1_x, slice_boundary_1_y]),
        dims=dims_xy,
        coords=coords_xy,
    )
    spec_slice_boundary_2_xy = xr.DataArray(
        np.vstack([slice_boundary_2_x, slice_boundary_2_y]),
        dims=dims_xy,
        coords=coords_xy,
    )

    ##########################################################################
    slice_boundary_1_y = np.array([-50, 50])
    slice_boundary_2_y = np.array([-50, 50])

    slice_boundary_1_z = plane_z(
        n,
        spec_slice_boundary_1_center,
        np.repeat(anat_slice_offset_mm, 2),
        slice_boundary_1_y,
    )
    slice_boundary_2_z = plane_z(
        n,
        spec_slice_boundary_2_center,
        np.repeat(anat_slice_offset_mm, 2),
        slice_boundary_2_y,
    )

    if np.any(slice_boundary_1_z == np.inf):
        slice_boundary_1_z = np.array([-50, 50])
        slice_boundary_2_z = np.array([-50, 50])

        slice_boundary_1_y = plane_y(
            n,
            spec_slice_boundary_1_center,
            np.repeat(anat_slice_offset_mm, 2),
            slice_boundary_1_z,
        )
        slice_boundary_2_y = plane_y(
            n,
            spec_slice_boundary_2_center,
            np.repeat(anat_slice_offset_mm, 2),
            slice_boundary_2_z,
        )

    dims_yz = [
        "y",
        "z",
    ]
    coords_yz = {
        "y": np.arange(2),
        "z": np.arange(2),
    }

    spec_slice_boundary_1_yz = xr.DataArray(
        np.vstack([slice_boundary_1_y, slice_boundary_1_z]),
        dims=dims_yz,
        coords=coords_yz,
    )
    spec_slice_boundary_2_yz = xr.DataArray(
        np.vstack([slice_boundary_2_y, slice_boundary_2_z]),
        dims=dims_yz,
        coords=coords_yz,
    )

    ##########################################################################
    slice_boundary_1_z = np.array([-50, 50])
    slice_boundary_2_z = np.array([-50, 50])

    slice_boundary_1_x = plane_x(
        n,
        spec_slice_boundary_1_center,
        np.repeat(anat_slice_offset_mm, 2),
        slice_boundary_1_z,
    )
    slice_boundary_2_x = plane_x(
        n,
        spec_slice_boundary_2_center,
        np.repeat(anat_slice_offset_mm, 2),
        slice_boundary_2_z,
    )

    if np.any(slice_boundary_1_x == np.inf):
        slice_boundary_1_x = np.array([-50, 50])
        slice_boundary_2_x = np.array([-50, 50])

        slice_boundary_1_z = plane_z(
            n,
            spec_slice_boundary_1_center,
            slice_boundary_1_x,
            np.repeat(anat_slice_offset_mm, 2),
        )
        slice_boundary_2_z = plane_z(
            n,
            spec_slice_boundary_2_center,
            slice_boundary_2_x,
            np.repeat(anat_slice_offset_mm, 2),
        )

    dims_zx = [
        "z",
        "x",
    ]
    coords_zx = {
        "z": np.arange(2),
        "x": np.arange(2),
    }

    spec_slice_boundary_1_zx = xr.DataArray(
        np.vstack([slice_boundary_1_z, slice_boundary_1_x]),
        dims=dims_zx,
        coords=coords_zx,
    )
    spec_slice_boundary_2_zx = xr.DataArray(
        np.vstack([slice_boundary_2_z, slice_boundary_2_x]),
        dims=dims_zx,
        coords=coords_zx,
    )

    ##########################################################################
    return {
        "spec_slice_boundary_1_xy": spec_slice_boundary_1_xy,
        "spec_slice_boundary_2_xy": spec_slice_boundary_2_xy,
        "spec_slice_boundary_1_yz": spec_slice_boundary_1_yz,
        "spec_slice_boundary_2_yz": spec_slice_boundary_2_yz,
        "spec_slice_boundary_1_zx": spec_slice_boundary_1_zx,
        "spec_slice_boundary_2_zx": spec_slice_boundary_2_zx,
    }


def plot_image_stack(
    spec_BrukerExp,
    anatomical_BrukerExp,
    plot_axes_names=["y", "x"],
    slice_axis_name="z",
    undisplayed_axes_dict={},
    animal_ID="",
    cell_line=None,
    initial_slice_number=0,
    plotsavepath="",
    plotname="",
    saveplot=False,
    plot_params={},
):
    # Load anatomical images
    anatomical_seq2d_xyz_xarray = anatomical_BrukerExp.seq2d_xyz_xarray

    # Load plot parameters
    figsize = plot_params.get("figsize", (4, 4))
    plot_all_in_one = plot_params.get("plot_all_in_one", False)

    vmin_anat = plot_params.get("vmin_anat", 0)
    vmax_anat = plot_params.get("vmax_anat", 120)

    cmap_anat = plot_params.get("cmap_anat", "bone")

    line_label = plot_params.get("line_label", "spectroscopy slice boundaries")
    linecolor = plot_params.get("linecolor", "dodgerblue")
    linewidth = plot_params.get("linewidth", 1)
    linestyle = plot_params.get("linestyle", "dashed")

    showtitle = plot_params.get("showtitle", True)
    showticks = plot_params.get("showticks", True)
    showtime = plot_params.get("showtime", False)
    showcolorbar = plot_params.get("showcolorbar", False)
    showscale_bar = plot_params.get("showscale_bar", True)
    showspectroscopy_slice = plot_params.get("showspectroscopy_slice", True)
    showinset = plot_params.get("showinset", True)
    showtime_inimage = plot_params.get("showtime_inimage", False)

    patchcenterlist = plot_params.get("patchcenterlist", [(0, 0)])
    patchboxlist = plot_params.get("patchboxlist", [(10, 10)])
    colorlist = plot_params.get("colorlist", ["r"])

    leg_distance_to_plot = plot_params.get("leg_distance_to_plot", 1.2)
    flip_rows_columns = plot_params.get("flip_rows_columns", False)

    slice_number = anatomical_seq2d_xyz_xarray[slice_axis_name].attrs["matrix"]
    slice_range = plot_params.get("slice_range", range(slice_number))

    y_axis_name = plot_axes_names[0]
    x_axis_name = plot_axes_names[1]
    anatomical_seq2d_xyz_xarray.get_axis_num(y_axis_name)
    anatomical_seq2d_xyz_xarray.get_axis_num(x_axis_name)

    y_extent_mm = anatomical_seq2d_xyz_xarray[y_axis_name].attrs["extent_mm"]
    x_extent_mm = anatomical_seq2d_xyz_xarray[x_axis_name].attrs["extent_mm"]
    extent_mm = [x_extent_mm[0], x_extent_mm[1], y_extent_mm[0], y_extent_mm[1]]

    if plot_all_in_one:
        from ..plot.utils_plot import subplot_auto_size

        grid_rows, grid_cols = subplot_auto_size(num_subplots=slice_number)
        if flip_rows_columns:
            fig, ax = plt.subplots(grid_cols, grid_rows, figsize=figsize)
        else:
            fig, ax = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        ax = ax.flatten()

    else:
        fig, ax = plt.subplots(figsize=figsize)

    def plot_image_stack_single(slice):
        ax.clear()
        try:
            cb_ax.clear()
            cb_ax.set_visible(False)
        except BaseException:
            pass

        subplot = ax.imshow(
            anatomical_seq2d_xyz_xarray.isel(
                {**undisplayed_axes_dict, **{slice_axis_name: slice_number - 1 - slice}}
            )
            .transpose(y_axis_name, x_axis_name)
            .to_numpy(),
            cmap=cmap_anat,
            vmin=vmin_anat,
            vmax=vmax_anat,
            extent=extent_mm,
            origin="lower",
        )

        if showspectroscopy_slice:
            try:
                slice_boundary_1_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_1_" + y_axis_name + x_axis_name]
                slice_boundary_2_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_2_" + y_axis_name + x_axis_name]
                slice_boundary_1_y = slice_boundary_1_xarray[0]
                slice_boundary_1_x = slice_boundary_1_xarray[1]
                slice_boundary_2_y = slice_boundary_2_xarray[0]
                slice_boundary_2_x = slice_boundary_2_xarray[1]

            except BaseException:
                slice_boundary_1_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_1_" + x_axis_name + y_axis_name]
                slice_boundary_2_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_2_" + x_axis_name + y_axis_name]
                slice_boundary_1_y = slice_boundary_1_xarray[1]
                slice_boundary_1_x = slice_boundary_1_xarray[0]
                slice_boundary_2_y = slice_boundary_2_xarray[1]
                slice_boundary_2_x = slice_boundary_2_xarray[0]

            slice_boundary_1 = ax.plot(
                slice_boundary_1_x,
                slice_boundary_1_y,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                label=line_label,
            )
            slice_boundary_2 = ax.plot(
                slice_boundary_2_x,
                slice_boundary_2_y,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )

        # if anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        # ax.scatter(t[0], t[1], color=linecolor)
        # if anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        # ax.scatter(t[0], t[2], color=linecolor)
        # if anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        # ax.scatter(t[2], t[1], color=linecolor)

        ax.set_ylim(
            extent_mm[2],
            extent_mm[3],
        )
        ax.set_xlim(
            extent_mm[0],
            extent_mm[1],
        )

        if showticks:
            ax.set_ylabel(y_axis_name + " [mm]")
            ax.set_xlabel(x_axis_name + " [mm]")
        else:
            ax.axis("off")

        if showtime_inimage:
            frame_time_s = anatomical_BrukerExp.frame_time_s[slice]
            text_x = extent_mm[0] + 1 / 10 * (extent_mm[1] - extent_mm[0])
            text_y = extent_mm[3] - 1 / 10 * (extent_mm[3] - extent_mm[2])
            plt.text(
                text_x,
                text_y,
                "t = {1:02d} min {2:02d} s".format(
                    int(frame_time_s // 3600),
                    int((frame_time_s % 3600) // 60),
                    int(frame_time_s % 60),
                    int(((frame_time_s % 60) * 1000) % 1000),
                ),
                color="white",
                bbox=dict(fill=False, edgecolor=None, linewidth=0),
                fontsize=10,
            )

        if showcolorbar:
            divider = make_axes_locatable(ax)
            cb_ax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(subplot, cax=cb_ax)
            cbar.set_label(r"signal [a. u.]")
            cbar.set_ticks(ticks=[vmin_anat, (vmax_anat - vmin_anat) // 2, vmax_anat])

        # add_colorbar(
        # image=anatomical_seq2d_xyz_xarray.isel({slice_axis_name: slice}).transpose(y_axis_name, x_axis_name).to_numpy(),
        # # image=None,
        # # boundaries=None,
        # boundaries={'vmin': vmin, 'vmax': vmax},
        # title=r'signal [a. u.]',
        # mappable=subplot,
        # color='k',
        # fontsize=12,
        # box=[0.3, -0.1, 0.2, 0.05],
        # shrink=1.0,
        # aspect=4,
        # pad=-0.075,
        # tickpad=0.5,
        # titlepad=0.075,
        # xtext=None,
        # ytext=0.15,
        # strformat='%d',
        # orientation='horizontal',
        # side=False
        # )

        # add_colorbar(
        # image=anatomical_seq2d_xyz_xarray.isel({slice_axis_name: slice}).transpose(y_axis_name, x_axis_name).to_numpy(),
        # boundaries=None,
        # # boundaries={'vmin': min_signal, 'vmax': max_signal},
        # title=r'signal [a. u.]',
        # mappable=subplot,
        # color='k',
        # fontsize=12,
        # box=[1.01, 0.0, 0.05, 1.0],
        # shrink=1.0,
        # aspect=4,
        # pad=0.0,
        # tickpad=0.5,
        # titlepad=5,
        # xtext=None,
        # ytext=0.3,
        # strformat='%d',
        # orientation='vertical',
        # side=True
        # )

        # plot_colorbar(fig, ax, anatomical_seq2d_xyz_xarray.isel({slice_axis_name: slice}).transpose(y_axis_name, x_axis_name).to_numpy(), label=r'signal [a. u.]')

        if showscale_bar:
            add_scalebar(
                px=1e-1,
                unit="cm",
                fixed_value=1,
                color="w",
                box_alpha=0.0,
                box_color="w",
                location="lower right",
                frameon=True,
                pad=0.075,
                length_fraction=None,
                border_pad=0.075,
                fontsize=12,
            )
        else:
            pass

        if showinset:
            insetslicelist = []

            for i, patchcenter in enumerate(patchcenterlist):
                insetslicelist.append(
                    add_patch(
                        image=anatomical_seq2d_xyz_xarray.isel(
                            {
                                **undisplayed_axes_dict,
                                **{slice_axis_name: slice_number - 1 - slice},
                            }
                        )
                        .transpose(y_axis_name, x_axis_name)
                        .to_numpy(),
                        patchcenter=patchcenter,
                        patchbox=patchboxlist[i],
                        ax=None,
                        sl=None,
                        color=colorlist[i],
                        lw=1,
                        fc="none",
                        out=True,
                    )
                )

            image_shape = (
                anatomical_seq2d_xyz_xarray.isel(
                    {
                        **undisplayed_axes_dict,
                        **{slice_axis_name: slice_number - 1 - slice},
                    }
                )
                .transpose(y_axis_name, x_axis_name)
                .to_numpy()
                .shape
            )

            axin = ax.inset_axes(
                [
                    1.05,
                    0.0,
                    (0.6 * image_shape[0] * patchboxlist[0][1] / patchboxlist[0][0])
                    / image_shape[1],
                    0.6,
                ]
            )

            inset = axin.imshow(
                anatomical_seq2d_xyz_xarray.isel(
                    {
                        **undisplayed_axes_dict,
                        **{slice_axis_name: slice_number - 1 - slice},
                    }
                )
                .transpose(y_axis_name, x_axis_name)
                .to_numpy(),
                vmin=vmin_anat,
                vmax=vmax_anat,
                cmap=cmap_anat,
                origin="lower",
                extent=extent_mm,
            )

            axin.set_xlim([insetslicelist[0][1].start, insetslicelist[0][1].stop])
            axin.set_ylim([insetslicelist[0][0].start, insetslicelist[0][0].stop])

            axin.set_xticks([])
            axin.set_yticks([])

            axin.spines["bottom"].set_color(colorlist[0])
            axin.spines["top"].set_color(colorlist[0])
            axin.spines["left"].set_color(colorlist[0])
            axin.spines["right"].set_color(colorlist[0])

            add_scalebar(
                px=1,
                ax=axin,
                unit="mm",
                fixed_value=2,
                color="w",
                box_alpha=0.0,
                box_color="w",
                location="lower right",
                frameon=True,
                pad=0.075,
                length_fraction=None,
                border_pad=0.075,
                fontsize=12,
            )

        handles, labels = ax.get_legend_handles_labels()
        if showtime:
            frame_time_s = anatomical_BrukerExp.frame_time_s[slice]
            leg = ax.legend(
                loc="center left",
                bbox_to_anchor=(0, leg_distance_to_plot),
                handles=handles,
                title=animal_ID
                + "\n"
                + cell_line
                + "\n"
                + "t = {1:02d} min {2:02d} s".format(
                    int(frame_time_s // 3600),
                    int((frame_time_s % 3600) // 60),
                    int(frame_time_s % 60),
                    int(((frame_time_s % 60) * 1000) % 1000),
                ),
            )
            leg.get_frame().set_edgecolor("black")

        elif cell_line:
            leg = ax.legend(
                loc="center left",
                bbox_to_anchor=(0, leg_distance_to_plot),
                handles=handles,
                title=animal_ID + "\n" + cell_line,
            )
            leg.get_frame().set_edgecolor("black")

        else:
            pass
        plt.tight_layout()

    if plot_all_in_one:
        for s, slice in enumerate(slice_range):
            subplot = ax[s].imshow(
                anatomical_seq2d_xyz_xarray.isel(
                    {
                        **undisplayed_axes_dict,
                        **{slice_axis_name: slice_number - 1 - slice},
                    }
                )
                .transpose(y_axis_name, x_axis_name)
                .to_numpy(),
                cmap=cmap_anat,
                vmin=vmin_anat,
                vmax=vmax_anat,
                extent=extent_mm,
            )

            try:
                slice_boundary_1_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_1_" + y_axis_name + x_axis_name]
                slice_boundary_2_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_2_" + y_axis_name + x_axis_name]
                slice_boundary_1_y = slice_boundary_1_xarray[0]
                slice_boundary_1_x = slice_boundary_1_xarray[1]
                slice_boundary_2_y = slice_boundary_2_xarray[0]
                slice_boundary_2_x = slice_boundary_2_xarray[1]

            except BaseException:
                slice_boundary_1_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_1_" + x_axis_name + y_axis_name]
                slice_boundary_2_xarray = get_slice_boundaries_xarray(
                    spec_BrukerExp, anatomical_BrukerExp, slice_axis_name, slice
                )["spec_slice_boundary_2_" + x_axis_name + y_axis_name]
                slice_boundary_1_y = slice_boundary_1_xarray[1]
                slice_boundary_1_x = slice_boundary_1_xarray[0]
                slice_boundary_2_y = slice_boundary_2_xarray[1]
                slice_boundary_2_x = slice_boundary_2_xarray[0]

            if showspectroscopy_slice:
                slice_boundary_1 = ax[s].plot(
                    slice_boundary_1_x,
                    slice_boundary_1_y,
                    color=linecolor,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=line_label,
                )
                slice_boundary_2 = ax[s].plot(
                    slice_boundary_2_x,
                    slice_boundary_2_y,
                    color=linecolor,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )

            ax[s].set_ylim(
                extent_mm[2],
                extent_mm[3],
            )
            ax[s].set_xlim(
                extent_mm[0],
                extent_mm[1],
            )

            if showscale_bar:
                add_scalebar(
                    px=1e-1,
                    unit="cm",
                    fixed_value=1,
                    color="w",
                    box_alpha=0.0,
                    box_color="w",
                    location="lower right",
                    frameon=True,
                    pad=0.075,
                    length_fraction=None,
                    border_pad=0.075,
                    fontsize=12,
                )
            else:
                pass

            if not showticks:
                if showspectroscopy_slice:
                    ax[s].axis("off")
                else:
                    if s in np.arange(
                        slice_number // 2
                        - spec_BrukerExp.slice_thickness_mm
                        // anatomical_BrukerExp.slice_thickness_mm
                        // 2,
                        slice_number // 2
                        + spec_BrukerExp.slice_thickness_mm
                        // anatomical_BrukerExp.slice_thickness_mm
                        // 2
                        + 1,
                    ):
                        ax[s].set_xticks([])
                        ax[s].set_yticks([])

                        ax[s].spines["bottom"].set_color(linecolor)
                        ax[s].spines["top"].set_color(linecolor)
                        ax[s].spines["left"].set_color(linecolor)
                        ax[s].spines["right"].set_color(linecolor)

                        ax[s].spines["bottom"].set_linewidth(2)
                        ax[s].spines["top"].set_linewidth(2)
                        ax[s].spines["left"].set_linewidth(2)
                        ax[s].spines["right"].set_linewidth(2)
                    else:
                        ax[s].axis("off")

            if showtitle:
                ax[s].set_title(slice)

        # Turn off unused subplots
        for i in range(slice_number, len(ax)):
            ax[i].set_axis_off()

        plt.tight_layout()

        if saveplot:
            plt.savefig(
                os.path.join(
                    plotsavepath,
                    plotname + "_" + anatomical_BrukerExp.slice_orientation,
                )
            )

    elif saveplot:
        for slice in tqdm(slice_range):
            ax.clear()
            plot_image_stack_single(slice)
            plt.savefig(
                os.path.join(
                    plotsavepath,
                    plotname
                    + anatomical_BrukerExp.slice_orientation
                    + "_slice_{:03d}".format(slice),
                )
            )

    else:

        @widgets.interact(slice=(0, (slice_number - 1), 1))
        def plot_image_stack_with_slice_slider(slice=initial_slice_number):
            ax.clear()
            plot_image_stack_single(slice)


def plot_image_intensity_over_time(
    anatomical_BrukerExp,
    undisplayed_axes_dict={},
    displayed_axes_names=["y", "x"],
    time_of_anestetic_administration=None,
    animal_ID="",
    anesthetic_administration="i.v. anesthesia",
    plotsavepath="",
    plotname="",
    saveplot=False,
    plot_params={},
):
    # Load anatomical images
    anatomical_seq2d_xyz_xarray = anatomical_BrukerExp.seq2d_xyz_xarray
    y_axis_name = displayed_axes_names[0]
    x_axis_name = displayed_axes_names[1]
    y_axis_number = anatomical_seq2d_xyz_xarray.get_axis_num(y_axis_name)
    x_axis_number = anatomical_seq2d_xyz_xarray.get_axis_num(x_axis_name)

    # Load plot parameters
    figsize = plot_params.get("figsize", (4, 4))

    patchcenter = plot_params.get("patchcenter", None)
    patchbox = plot_params.get("patchbox", None)

    fig, ax = plt.subplots(figsize=figsize)

    x = anatomical_seq2d_xyz_xarray["repetition_idx"].attrs["axis_s"] / 60
    if patchcenter:
        insetslice_mm = add_patch(
            image=anatomical_seq2d_xyz_xarray.isel(undisplayed_axes_dict)
            .transpose(y_axis_name, x_axis_name, "repetition_idx")
            .to_numpy(),
            patchcenter=patchcenter,
            patchbox=patchbox,
            ax=None,
            sl=None,
            color="none",
            lw=1,
            fc="none",
            out=True,
        )

        insetslice_idx = np.s_[
            np.argmin(
                np.abs(
                    anatomical_seq2d_xyz_xarray[y_axis_name].attrs["axis_mm"]
                    - insetslice_mm[0].start
                )
            ) : np.argmin(
                np.abs(
                    anatomical_seq2d_xyz_xarray[y_axis_name].attrs["axis_mm"]
                    - insetslice_mm[0].stop
                )
            ),
            np.argmin(
                np.abs(
                    anatomical_seq2d_xyz_xarray[x_axis_name].attrs["axis_mm"]
                    - insetslice_mm[1].start
                )
            ) : np.argmin(
                np.abs(
                    anatomical_seq2d_xyz_xarray[x_axis_name].attrs["axis_mm"]
                    - insetslice_mm[1].stop
                )
            ),
            :,
        ]

        y = (
            anatomical_seq2d_xyz_xarray.isel(undisplayed_axes_dict)
            .transpose(y_axis_name, x_axis_name, "repetition_idx")
            .to_numpy()[insetslice_idx]
            .sum((y_axis_number, x_axis_number))[::-1]
        )
        ylabel = "integrated heart ROI signal [a.u.]"

    else:
        y = (
            anatomical_seq2d_xyz_xarray.isel(undisplayed_axes_dict)
            .transpose(y_axis_name, x_axis_name, "repetition_idx")
            .sum(displayed_axes_names)[::-1]
        )
        ylabel = "integrated image signal [a.u.]"

    subplot_1 = ax.plot(
        x,
        y,
        color="blue",
        linestyle="solid",
        linewidth=0.5,
    )
    if time_of_anestetic_administration:
        vline_1 = ax.axvline(
            time_of_anestetic_administration,
            color="magenta",
            linestyle="dashed",
            linewidth=1,
            label="anesthetic\n administration",
        )

    ax.set_xlabel("time [min]")
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        loc="upper right",
        handles=handles,
        title=animal_ID + "\n" + anesthetic_administration,
    )
    leg.get_frame().set_edgecolor("black")

    if saveplot:
        plt.savefig(
            os.path.join(plotsavepath, animal_ID + "_" + plotname + "_" + "iis")
        )

    return x, y


def make_gif(frame_folder, gif_name, sample_ID, duration=50):
    # frames = [
    # Image.open(image)
    # for image in sorted(glob.glob(f'{frame_folder}/*.png'), key=os.path.getmtime)
    # ]
    # frame_one = frames[0]
    # # frame_one.save(gif_name + '.gif', format='GIF', append_images=frames, save_all=True, duration=250, loop=0)
    # frame_one.save(
    # gif_name + '.gif',
    # format='GIF',
    # append_images=frames,
    # save_all=True,
    # duration=duration,
    # loop=0,
    # )

    with imageio.get_writer(
        gif_name + ".gif", mode="I", duration=duration * 1e-3, loop=0,
    ) as writer:
        for filename in sorted(glob.glob(f"{frame_folder}/{sample_ID}*.png")):
            image = imageio.imread(filename)
            writer.append_data(image)


def plot_slice_orientation(
    spec_BrukerExp, anatomical_BrukerExp, plotsavepath, plotname, figsize=None
):
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    n = spec_BrukerExp.acquisition_matrix[2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.quiver(
        0,
        0,
        0,
        ex[0],
        ex[1],
        ex[2],
        color="red",
        label="x",
    )
    ax.quiver(
        0,
        0,
        0,
        ey[0],
        ey[1],
        ey[2],
        color="green",
        label="y",
    )
    ax.quiver(
        0,
        0,
        0,
        ez[0],
        ez[1],
        ez[2],
        color="blue",
        label="z",
    )
    ax.quiver(
        0,
        0,
        0,
        n[0],
        n[2],
        n[1],
        color="black",
        label="slice normal",
    )

    # spec sample of slice orientation
    xx, yy = np.meshgrid(
        np.linspace(-0.5, 0.5, 1000), np.linspace(-0.5, 0.5, 1000), indexing="xy"
    )

    zz = plane_z(n, np.zeros(3), xx, yy)
    # print(zz)

    if np.any(zz == np.inf):
        yy, zz = np.meshgrid(
            np.linspace(-0.5, 0.5, 1000), np.linspace(-0.5, 0.5, 1000), indexing="xy"
        )
        xx = plane_x(n, np.zeros(3), yy, zz)
        # print(xx)

        if np.any(xx == np.inf):
            zz, xx = np.meshgrid(
                np.linspace(-0.5, 0.5, 1000),
                np.linspace(-0.5, 0.5, 1000),
                indexing="xy",
            )
            yy = plane_y(n, np.zeros(3), xx, zz)
            # print(yy)

    ax.plot_surface(xx, yy, np.where(np.abs(zz) > 1, np.nan, zz), alpha=0.5)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.dist = 13

    leg = ax.legend(framealpha=1)
    # leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plotsavepath,
            plotname,
        )
    )
