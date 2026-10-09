"""Anatomical post-processing utilities for grid interpolation and overlays."""

import matplotlib.pyplot as plt
import numpy as np

from ...analysis.spectroscopy import make_NDspec_6Dspec
from ...utils.system.logging import init_default_logger

# Handle numpy.distutils compatibility issue with mpl_interactions
try:
    from mpl_interactions import image_segmenter

    MPL_INTERACTIONS_AVAILABLE = True
except ImportError as e:
    if "numpy.distutils" in str(e):
        logger = init_default_logger(__name__)
        logger.warning(
            "mpl_interactions not available due to numpy.distutils incompatibility. "
            "Interactive segmentation features will be disabled."
        )
        image_segmenter = None
        MPL_INTERACTIONS_AVAILABLE = False
    else:
        raise e

from matplotlib_scalebar.scalebar import ScaleBar

logger = init_default_logger(__name__)


def bruker2complex(arr):
    """Turns bruker semi-complex arrays (2xfloat) into numpy.complex (1xcomplex)"""
    assert arr.shape[-1] == 2, "Last dimension must be complex dimension."
    return arr[..., 0] + 1j * arr[..., 1]


def interpolate_array(
    input_data,
    data_object,
    interp_size=None,
    interp_method="linear",
    use_multiprocessing=False,
    dtype=None,
    number_of_cpu_cores=None,
):
    """ "
    This function interpolates the input data onto a new grid in the same range as
    input_data but points in each dimension are described by interp_size. Uses scipy's RegularGridInterpolator
    method. Multithreading can be activated.

    Parameters
    ----------
    input data: np.ndarray
        Expected to have the order Echoes, Read, phase, slice, repetitions, channels
    data_object: MRIExperiment
        Object containing info about the coordinates
    interp_size: tuple
        Describes the interpolated data size.
        Has to be 6D: number of echhoes, points in  Read, phase, slice; number of repetitions, channels
    interp_method: str
        interpolation method that is used. Supported are "linear", "nearest",
        # if scipy==1.10.0: "slinear", "cubic", "quintic" and "pchip".
    use_multiprocessing: bool
        toggles multiprocessing
    dtype:
        should give you the option to change the dtype of the to be interpolated array to reduce the size
    number_of_cpu_cores: int
        Number of CPU cores to if multi processing is desired

    """

    import time

    from scipy.interpolate import interpn

    from ...utils.coordinates.grids import define_xyz_grid

    if use_multiprocessing:
        try:
            from joblib import Parallel, cpu_count, delayed
            from tqdm.auto import tqdm

            # try to set the number of usable cpu cores to the amount of
            # available cores
            if number_of_cpu_cores is None:
                # int-divide by 2 because hyperthreading cores dont count/help:
                number_of_cpu_cores = cpu_count() // 2
            else:
                pass

        except BaseException:
            use_multiprocessing = False

    if interp_size is None:
        logger.error("You have to enter an interpolation size")
        interp_size = input_data.shape

    # check method:
    allowed_interp_methods = ["linear", "nearest", "splinef2d", "cubic"]
    if interp_method not in allowed_interp_methods:
        logger.critical(
            "uknown interpolation method: " + interp_method + ", using linear instead."
        )
        interp_method = "linear"

    if dtype is None:
        dtype = input_data.dtype

    # duration:
    t = np.arange(0, input_data.shape[4], 1)
    t_num = input_data.shape[4]

    # cant loop over repetitions if this also should
    # be interpolated ...
    if len(t) == interp_size[4]:
        pass
    else:
        use_multiprocessing = False
    # get extend in x-y-z:
    x, y, z = define_xyz_grid(data_object)
    # number of echoes:
    e_num = input_data.shape[0]
    # range of echoes:
    e = np.linspace(1, e_num, input_data.shape[0])
    # number of channels:
    c_num = input_data.shape[5]
    # range of channels:
    c = np.linspace(1, c_num, input_data.shape[5])

    # points to interpolate onto:
    # echoes:
    eres = np.linspace(1, e[-1], interp_size[0])
    # points in space:
    xres = np.linspace(x[0], x[-1], interp_size[1])
    yres = np.linspace(y[0], y[-1], interp_size[2])
    zres = np.linspace(z[0], z[-1], interp_size[3])
    # repetitions:
    tres = np.linspace(t[0], t[-1], interp_size[4])

    # channels:
    if e_num == 1:
        eres = np.linspace(1, e_num, input_data.shape[0])
    else:
        eres = np.linspace(1, e[-1], interp_size[0])

    # channels:
    if c_num == 1:
        cres = np.linspace(1, c_num, input_data.shape[5])
    else:
        cres = c

    # get the start time
    st = time.time()

    if use_multiprocessing:
        # generate grids (to interpolate onto)
        egres, xgres, ygres, zgres, cgres = np.meshgrid(
            eres, xres, yres, zres, cres, indexing="ij"
        )
        # init empty array:
        interpolated_data = np.zeros(
            (len(eres), len(xres), len(yres), len(zres), len(t), len(cres)),
            dtype=dtype,
        )

        # index list (time range)
        index_list = list(np.ndindex(len(t)))

        # create progress bar
        index_list = tqdm(index_list, desc="interpolation progress", leave=True)

        # define interpolationm functionm
        def interpolate_image(it):
            # the conditions speed up the interpolation quite a bit
            # if there are less dimensions (1 echo e.g.)

            # if more than 1 echo and more than 1 channel:
            if e_num > 1 and c_num > 1:
                interpolated_data_timepoint = np.squeeze(
                    interpn(
                        points=(e, x, y, z, c),
                        values=np.squeeze(input_data[:, :, :, :, it, :]),
                        xi=(egres, xgres, ygres, zgres, cgres),
                        method=interp_method,
                    )
                )
            # if more than 1 echo and 1 channel:
            elif e_num > 1 and c_num == 0:
                interpolated_data_timepoint = np.squeeze(
                    interpn(
                        points=(e, x, y, z),
                        values=np.squeeze(input_data[:, :, :, :, it, 0]),
                        xi=(egres, xgres, ygres, zgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                interpolated_data_timepoint = np.squeeze(
                    interpn(
                        points=(x, y, z, c),
                        values=np.squeeze(input_data[0, :, :, :, it, :]),
                        xi=(xgres, ygres, zgres, cgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                interpolated_data_timepoint = np.squeeze(
                    interpn(
                        points=(x, y, z),
                        values=np.squeeze(input_data[0, :, :, :, it, 0]),
                        xi=(xgres, ygres, zgres),
                        method=interp_method,
                    )
                )

            return interpolated_data_timepoint

        # interpolate multiple timesteps in parallel:
        interpolated_data_tuple = Parallel(n_jobs=number_of_cpu_cores)(
            delayed(interpolate_image)(it) for it in index_list
        )

        # if more than 1 echo and more than 1 channel:
        for it in range(len(t)):
            if e_num > 1 and c_num > 1:
                interpolated_data[:, :, :, :, it, :] = interpolated_data_tuple[it]
            # if more than 1 echo and 1 channel:
            elif e_num > 1 and c_num == 0:
                interpolated_data[:, :, :, :, it, 0] = interpolated_data_tuple[it]
            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                interpolated_data[0, :, :, :, it, :] = interpolated_data_tuple[it]
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                interpolated_data[0, :, :, :, it, 0] = interpolated_data_tuple[it]

    elif use_multiprocessing == False and (t_num > 1):
        # generate grids (to interpolate onto)
        egres, xgres, ygres, zgres, tgres, cgres = np.meshgrid(
            eres, xres, yres, zres, tres, cres, indexing="ij"
        )

        # init empty array:
        interpolated_data = np.zeros(
            (
                len(eres),
                len(xres),
                len(yres),
                len(zres),
                len(tres),
                len(cres),
            ),
            dtype=dtype,
        )
        # if more than 1 echo and more than 1 channel:
        if e_num > 1 and c_num > 1:
            interpolated_data = np.squeeze(
                interpn(
                    points=(e, x, y, z, t, c),
                    values=input_data[:, :, :, :, :, :],
                    xi=(egres, xgres, ygres, zgres, tgres, cgres),
                    method=interp_method,
                )
            )
        # if more than 1 echo and 1 channel:
        elif e_num > 1 and c_num == 0:
            interpolated_data[:, :, :, :, :, 0] = np.squeeze(
                interpn(
                    points=(e, x, y, z, t),
                    values=input_data[:, :, :, :, :, 0],
                    xi=(egres, xgres, ygres, zgres, tgres),
                    method=interp_method,
                )
            )

        # if 1 echo and more than 1 channel:
        elif e_num == 1 and c_num > 1:
            interpolated_data[0, :, :, :, :, :] = np.squeeze(
                interpn(
                    points=(x, y, z, t, c),
                    values=np.squeeze(input_data[0, :, :, :, :, :]),
                    xi=(xgres, ygres, zgres, tgres, cgres),
                    method=interp_method,
                )
            )
        # if 1 echo and 1 channel:
        elif e_num == 1 and c_num == 1:
            interpolated_data[0, :, :, :, :, 0] = np.squeeze(
                interpn(
                    points=(x, y, z, t),
                    values=input_data[0, :, :, :, :, 0],
                    xi=(xgres, ygres, zgres, tgres),
                    method=interp_method,
                )
            )

    elif use_multiprocessing == False and (t_num == 1):
        # generate grids (to interpolate onto)
        egres, xgres, ygres, zgres, cgres = np.meshgrid(
            eres, xres, yres, zres, cres, indexing="ij"
        )
        # init empty array:
        interpolated_data = np.zeros(
            (len(eres), len(xres), len(yres), len(zres), len(tres), len(cres)),
            dtype=dtype,
        )
        # if more than 1 echo and more than 1 channel:
        if e_num > 1 and c_num > 1:
            interpolated_data[:, :, :, :, 0, :] = np.squeeze(
                interpn(
                    points=(e, x, y, z, c),
                    values=input_data[:, :, :, :, 0, :],
                    xi=(egres, xgres, ygres, zgres, cgres),
                    method=interp_method,
                )
            )

        # if more than 1 echo and  1 channel:
        elif e_num > 1 and c_num == 1:
            interpolated_data[:, :, :, :, 0, 0] = np.squeeze(
                interpn(
                    points=(
                        e,
                        x,
                        y,
                        z,
                    ),
                    values=input_data[:, :, :, :, 0, 0],
                    xi=(egres, xgres, ygres, zgres),
                    method=interp_method,
                )
            )
        # if 1 echo and more than 1 channel:
        elif e_num == 1 and c_num > 1:
            for it in range(len(t)):
                interpolated_data[0, :, :, :, 0, :] = np.squeeze(
                    interpn(
                        points=(x, y, z, c),
                        values=np.squeeze(input_data[0, :, :, :, 0, :]),
                        xi=(xgres, ygres, zgres, cgres),
                        method=interp_method,
                    )
                )
        # if 1 echo and 1 channel:
        elif e_num == 1 and c_num == 1:
            for it in range(len(t)):
                interpolated_data[0, :, :, :, 0, 0] = np.squeeze(
                    interpn(
                        points=(x, y, z),
                        values=input_data[0, :, :, :, 0, 0],
                        xi=(xgres, ygres, zgres),
                        method=interp_method,
                    )
                )

    # get the execution time
    et = time.time()
    elapsed_time = et - st
    # print excution time:
    logger.debug("Execution time:", elapsed_time, "seconds")

    return interpolated_data


def plot_voxel_grids(image1=None, image2=None, plot_params={}):
    """
    Plots two images side by side, each with an overlaid grid to highlight their respective voxel (pixel) sizes.
    The function supports rotation and custom color mapping for the visualization of each image. If only one image is provided,
    it will be displayed with its grid. The function allows customization of the slice index, line color, rotation,
    and colormap for each image through the `plot_params` dictionary.

    Parameters
    ----------
    image1 : ndarray, optional
        A 3D or higher-dimensional array representing the first image. The function expects the image to be in a specific
        format where the relevant slice is indexed as `image1[0, slice_ind_image1, :, :, 0, 0]`. If `None`, the function
        will not display this image.
    image2 : ndarray, optional
        A 3D or higher-dimensional array representing the second image. Similar to `image1`, it is expected to be in a
        format where the relevant slice is indexed as `image2[0, slice_ind_image2, :, :, 0, 0]`. If `None`, only `image1`
        will be displayed if it is provided.
    plot_params : dict, optional
        A dictionary containing parameters for plotting. Supported parameters include:
        - `slice_ind_image1` (int): The slice index for `image1`. Defaults to 0.
        - `slice_ind_image2` (int): The slice index for `image2`. Defaults to 0.
        - `line_color_image1` (str): Line color for the grid overlaid on `image1`. Defaults to "yellow".
        - `line_color_image2` (str): Line color for the grid overlaid on `image2`. Defaults to "white".
        - `rot_image` (int): Number of times the images should be rotated by 90 degrees. Defaults to 1.
        - `cmap` (str): The colormap used for both images. Defaults to "jet".

    Returns
    -------
    None
        The function does not return a value but displays a matplotlib plot with the images and their grids.

    Raises
    ------
    ValueError
        If `image1` is not provided, a critical log message is displayed indicating that `image1` must be passed.

    Notes
    -----
    The function is designed to visually compare two images of potentially different resolutions by overlaying a grid on each.
    This can be particularly useful for examining the alignment of voxel data in medical imaging or comparing model predictions
    against ground truth in machine learning tasks.
    Use hypermri.analysis.anatomical.masks.calc_overlap to calculate the overlap on a voxel by voxel basis.

    Examples
    --------
    >>> plot_voxel_grids(image1=my_image_array, plot_params={'slice_ind_image1': 10, 'linecolor_image1': 'blue'})
    This will display `my_image_array` with a slice index of 10 and a blue grid overlaid.
    """
    figsize = plot_params.get("figsize", (12, 6))
    slice_ind_image1 = plot_params.get("slice_ind_image1", 0)
    slice_ind_image2 = plot_params.get("slice_ind_image2", 0)
    linecolor_image1 = plot_params.get("linecolor_image1", "yellow")
    linecolor_image2 = plot_params.get("linecolor_image2", "white")
    linewidth_image1 = plot_params.get("linewidth_image1", 0.5)
    linewidth_image2 = plot_params.get("linewidth_image2", 0.5)
    linestyle_image1 = plot_params.get("linestyle_image1", "-")
    linestyle_image2 = plot_params.get("linestyle_image2", "-")
    colorbar_image1 = plot_params.get("colorbar_image1", False)
    colorbar_image2 = plot_params.get("colorbar_image2", False)
    colorbar_label_image1 = plot_params.get("colorbar_label_image1", "signal [a.u]")
    colorbar_label_image2 = plot_params.get("colorbar_label_image2", "signal [a.u]")
    vmin_image1 = plot_params.get("vmin_image1", None)
    vmax_image1 = plot_params.get("vmax_image1", None)
    vmin_image2 = plot_params.get("vmin_image2", None)
    vmax_image2 = plot_params.get("vmax_image2", None)
    rot_image = plot_params.get("rot_image", 1)
    cmap_image1 = plot_params.get("cmap_image1", "jet")
    cmap_image2 = plot_params.get("cmap_image2", "jet")
    title_image1 = plot_params.get("title_image1", "")
    title_image2 = plot_params.get("title_image2", "")
    savepath = plot_params.get("savepath", None)

    orientation_image1 = plot_params.get("orientation_image1", "axial")
    orientation_image2 = plot_params.get("orientation_image2", "coronal")

    match orientation_image1:
        case "axial":
            orientation_image1_ind = 1
        case "sagittal":
            orientation_image1_ind = 2
        case "coronal":
            orientation_image1_ind = 3
        case _:  # axial
            orientation_image1_ind = 1

    match orientation_image2:
        case "axial":
            orientation_image2_ind = 1
        case "sagittal":
            orientation_image2_ind = 2
        case "coronal":
            orientation_image2_ind = 3
        case _:  # axial
            orientation_image2_ind = 1

    extent_image1 = plot_params.get("extent_image1", None)
    extent_image2 = plot_params.get("extent_image2", None)

    if extent_image1 is not None and extent_image2 is None:
        extent_image2 = extent_image1

    if extent_image2 is not None and extent_image1 is None:
        extent_image1 = extent_image2

    if extent_image1 is None and extent_image2 is None:
        extent_image1 = [0, 1, 0, 1]
        extent_image2 = [0, 1, 0, 1]

    startx_image1 = extent_image1[0]
    endx_image1 = extent_image1[1]
    starty_image1 = extent_image1[2]
    endy_image1 = extent_image1[3]

    startx_image2 = extent_image2[0]
    endx_image2 = extent_image2[1]
    starty_image2 = extent_image2[2]
    endy_image2 = extent_image2[3]

    provided_dims = plot_params.get("provided_dims", ["x", "y"])

    if image1 is not None and image2 is not None:
        if np.ndim(image1) < 6:

            if np.ndim(image1) == len(provided_dims):
                image1 = make_NDspec_6Dspec(
                    input_data=image1, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image1 = make_NDspec_6Dspec(input_data=image1)
        if np.ndim(image2) < 6:

            if np.ndim(image2) == len(provided_dims):
                image2 = make_NDspec_6Dspec(
                    input_data=image2, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image2 = make_NDspec_6Dspec(input_data=image2)
        print(f"image1.shape={image1.shape}")
        print(f"image2.shape={image2.shape}")
        image1 = np.squeeze(
            np.take(image1, axis=orientation_image1_ind, indices=slice_ind_image1)
        )
        image2 = np.squeeze(
            np.take(image2, axis=orientation_image2_ind, indices=slice_ind_image2)
        )
        print(f"image1.shape={image1.shape}")
        print(f"image2.shape={image2.shape}")
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Overlay the low-resolution image
        im2 = ax[0].imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )
        ax[0].set_title(title_image1)

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[0].set_xlim(startx_image1, endx_image1)
        ax[0].set_ylim(starty_image1, endy_image1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # # Display the high-resolution image
        im1 = ax[1].imshow(
            np.rot90(image2, rot_image),
            extent=extent_image2,
            alpha=1,
            cmap=cmap_image2,
            vmin=vmin_image2,
            vmax=vmax_image2,
        )

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[1].set_xlim(startx_image2, endx_image2)
        ax[1].set_ylim(starty_image2, endy_image2)
        ax[1].set_title(title_image2)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
        plt.show()

        if colorbar_image2:
            fig.colorbar(im2, ax=ax[0], label=colorbar_label_image2, shrink=0.8)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax[1], label=colorbar_label_image1, shrink=0.8)

        plt.tight_layout()

    elif image1 is not None and image2 is None:
        if np.ndim(image1) < 6:

            if np.ndim(image1) == len(provided_dims):
                image1 = make_NDspec_6Dspec(
                    input_data=image1, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image1 = make_NDspec_6Dspec(input_data=image1)

        image1 = np.take(image1, axis=orientation_image1_ind, indices=slice_ind_image1)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Overlay the low-resolution image
        im1 = ax.imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )

        # Add grid
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax.axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax.axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Set the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Set the limits and labels
        ax.set_xlim(startx_image1, endx_image1)
        ax.set_ylim(starty_image1, endy_image1)
        ax.set_title(title_image1)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax, label=colorbar_label_image1, shrink=0.8)
        plt.show()
        plt.tight_layout()
    else:
        logger.critical("image1 has to be passed!")
    if savepath is not None:
        try:
            # plt.savefig(savepath + ".svg", format="svg", dpi=600, transparent=True)
            plt.savefig(savepath + ".png", format="png", dpi=600, transparent=True)
        except BaseException:
            raise Warning(f"image could not be save as {savepath}.png/.svg")


def add_scalebar(
    px,
    ax=None,
    unit="mm",
    fixed_value=None,
    color="k",
    box_alpha=0.0,
    box_color="w",
    location="lower right",
    frameon=False,
    pad=0.075,
    length_fraction=None,
    border_pad=0.075,
    fontsize=12,
):
    """
    Add scalebar to image, given the pixel size 'px'.
    """

    if not ax:
        ax = plt.gca()
    ax.add_artist(
        ScaleBar(
            px,
            unit=unit,
            fixed_value=fixed_value,
            color=color,
            box_alpha=box_alpha,
            box_color=box_color,
            location=location,
            frameon=frameon,
            pad=pad,
            length_fraction=length_fraction,
            border_pad=border_pad,
            font_properties={"size": fontsize},
        )
    )


def calc_k_space_trajectory_from_measurement(data_obj=None, db=False):
    """Estimate the slice-selection trajectory from phase-encoded FIDs.

    Parameters
    ----------
    data_obj : MRIExperiment
        Bruker dataset proxy that exposes ``fid``, ``method``, and geometry
        attributes used in the calculation.
    db : bool, optional
        When ``True`` intermediate plots are generated for debugging purposes.

    Returns
    -------
    numpy.ndarray
        Estimated trajectory in reciprocal metres.
    """
    fid = data_obj.fid

    # length of trajetory:
    traj_length = data_obj.matrix[0] * data_obj.matrix[1]

    # number of slices
    nslices = data_obj.n_slices

    # number of repetitions:
    nr = data_obj.n_repetitions

    # number of repetitions:
    nchans = data_obj.n_receivers

    # reshape trajectory
    fid = np.reshape(fid, (nr, nslices, nchans, traj_length))

    if db is True:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        for r in range(nr):
            for c in range(nchans):
                ax[0].plot(np.real(fid[r, 0, c, :]), label=f"Rep {r}, Chan {c}")
                ax[1].plot(np.real(fid[r, 1, c, :]), label=f"Rep {r}, Chan {c}")
            ax[0].plot(
                np.mean(np.real(fid[r, 0, :, :]), axis=-2),
                color="k",
                label=f"Rep {r}, Mean",
            )
            ax[1].plot(
                np.mean(np.real(fid[r, 1, :, :]), axis=-2),
                color="k",
                label=f"Rep {r}, Mean",
            )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title("Slice 0")
        ax[1].set_title("Slice 1")

    # average along the repetitions:
    fid = np.squeeze(np.mean(fid, axis=(0, -2)))

    # define slice distance:
    dx = data_obj.slice_gap_mm * 1e-3  # in metres

    traj = (np.unwrap(np.angle(fid[0, :])) - np.unwrap(np.angle(fid[1, :]))) / (
        2 * np.pi * dx
    )
    if db is True:
        ax[2].plot(traj)
        ax[2].set_title("Trajectory")
        plt.show()

    return traj


def Define_Extent_press(img_instance):
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
