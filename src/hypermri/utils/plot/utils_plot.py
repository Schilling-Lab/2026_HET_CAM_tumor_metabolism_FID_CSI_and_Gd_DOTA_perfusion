"""
Most of the utils functions are used for plotting inside jupyter-notebooks.

Available functions:

1. plot_colorbar

2. plot_image_series

3. plot_4d
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar



def add_scalebar(
    px,
    ax=None,
    units="mm",
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
            units=units,
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


def add_patch(
    image,
    patchcenter=None,
    patchbox=None,
    ax=None,
    sl=None,
    color='r',
    lw=2,
    fc='none',
    out=False,
):
    '''
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
    '''

    sh = image.shape

    if type(patchbox) == int:
        patchbox = [patchbox, patchbox]

    if type(sl) == type(None):
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
    if out == True:
        return sl


def extent_to_slice(shape, extent, patch, origin="lower"):
    """
    Convert a patch in data coordinates into numpy slice indices.

    Parameters
    ----------
    shape : tuple
        (height, width) of the image
    extent : [xmin, xmax, ymin, ymax]
        Coordinate extent of the image
    patch : [pxmin, pxmax, pymin, pymax]
        Patch bounds in same coordinate system
    origin : str
        "lower" or "upper" (matplotlib default) 

    Returns
    -------
    (rowslice, colslice) : tuple of slice objects
    """

    h, w = shape
    xmin, xmax, ymin, ymax = extent
    pxmin, pxmax, pymin, pymax = patch

    # Coordinate → index
    def x_to_col(x):
        return (x - xmin) / (xmax - xmin) * w

    if origin == "lower":
        def y_to_row(y):
            return (y - ymin) / (ymax - ymin) * h
    else:  # origin == "upper"
        def y_to_row(y):
            return (ymax - y) / (ymax - ymin) * h

    c0, c1 = x_to_col(pxmin), x_to_col(pxmax)
    r0, r1 = y_to_row(pymin), y_to_row(pymax)

    # Convert to integers
    c0, c1 = int(np.floor(c0)), int(np.ceil(c1))
    r0, r1 = int(np.floor(r0)), int(np.ceil(r1))

    # Clip to valid bounds
    c0, c1 = np.clip([c0, c1], 0, w)
    r0, r1 = np.clip([r0, r1], 0, h)

    return np.s_[r0:r1+1, c0:c1+1]


def plot_colorbar(figure, axis, data, **kwargs):
    """Appends colorbar to axis and scales it according to min and max of data.

    Requires the following imports:

        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

    Which cmap the colorbar picks is determined in the following order:
        1. In case the user specifies a cmap with 'cmap=...', that cmap is used.
        2. We try to extract the used cmap from the parent axis object.
        3. Fallback to default colormap, which is 'viridis'.
    """
    # sort out arrangement of colorbar and plot
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # normalize internal data form 0 to 1
    norm = plt.Normalize(np.nanmin(data), np.nanmax(data))

    # try to get corresponding cmap from axis in case none was specified
    cmap = kwargs.get("cmap", False)
    if not cmap:
        try:
            cmap = axis.get_children()[0].get_cmap()
        except AttributeError:
            print("Failed to retrieve cmap from axis.get_children()[0]")
            cmap = "viridis"

    # plot the colorbar
    figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        **kwargs,
    )

    return cax


def plot_image_series(
    arrays,
    label_list,
    nrows=1,
    cmap="plasma",
    plot_func=None,
    normalize=False,
    **subplot_kwrags,
):
    """Plots series of images into subplots, optionally into multiple rows"""

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [""] * (target_len - len(some_list))

    ncols = len(arrays) // nrows

    if len(label_list) != nrows * ncols:
        label_list = pad_or_truncate(label_list, nrows * ncols)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, **subplot_kwrags)

    if normalize:
        global_min, global_max = np.min(arrays), np.max(arrays)

    for ax, arr, label in zip(axs.flat, arrays, label_list):
        if plot_func:
            plot_func(ax, arr)
            plot_colorbar(fig, ax, arr)
        else:
            if normalize:
                ax.imshow(arr, cmap=cmap, vmin=global_min, vmax=global_max)
                plot_colorbar(fig, ax, [global_min, global_max])
            else:
                ax.imshow(arr, cmap=cmap)
                plot_colorbar(fig, ax, arr)
        ax.axis("off")
        ax.set_title(label)

    fig.tight_layout()


def plot_3d(data, **scatter_kwargs):
    """3D colorcoded matrix plot.

    https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data < 1e5
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    p = ax.scatter(
        x,
        y,
        z,
        c=data.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.5,
        marker="o",
        linewidth=0,
        **scatter_kwargs,
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = fig.colorbar(p, shrink=0.5, label="Frequency [Hz]")
    cbar.solids.set(alpha=1)
    fig.tight_layout()
    return ax


def subplot_auto_size(num_subplots=0):
    """
    Automatically caclculates the number of rows and columns for a number of subplots so that the outlie is as close as
    possible to a square.
    Parameters
    ----------
    num_subplots: int
        Number of subplots

    Returns
    -------
    Number of Rows, Number of Colums

    Examples
    --------
    >>> grid_rows, grid_cols = subplot_auto_size(num_subplots = 10)
    >>> fig, axs = plt.subplots(grid_rows, grid_cols)
    """
    # Calculate subplot grid dimensions
    if num_subplots == 0:
        return 0, 0
    grid_size = int(np.ceil(np.sqrt(num_subplots)))
    grid_rows = (
        grid_size - 1 if grid_size * (grid_size - 1) >= num_subplots else grid_size
    )
    grid_cols = grid_size
    return grid_rows, grid_cols


def remove_empty_subplots(fig, ax):
    if isinstance(ax, plt.Axes):
        return fig, ax

    ax = np.array(ax)  # Ensure ax is a numpy array

    if ax.ndim == 1:
        ax = ax.reshape(1, -1)  # Convert 1D to 2D array

    for i in range(ax.shape[0] - 1, -1, -1):
        for j in range(ax.shape[1] - 1, -1, -1):
            if (
                not ax[i, j].get_lines()
                and not ax[i, j].get_images()
                and not ax[i, j].collections
            ):
                fig.delaxes(ax[i, j])

    # Remove empty rows and columns
    ax = ax[~np.all(np.equal(ax, None), axis=1)]
    ax = ax[:, ~np.all(np.equal(ax, None), axis=0)]

    if fig.get_axes():
        fig.tight_layout()

    return fig, ax.flatten() if ax.size > 1 else ax[0, 0]


def format_y_tick_labels(ax, width=15, font_family="monospace"):
    """Format y-axis tick labels using a consistent width."""

    ticks = ax.get_yticks()
    formatted_labels = []
    for tick in ticks:
        formatted = f"{tick:.1e}" if "e" in f"{tick:.1e}" else f"{tick:>{width}.4g}"
        formatted_labels.append(formatted)

    ax.set_yticklabels(formatted_labels, family=font_family)


def get_overlapping_slices(data_obj_1=None, data_obj_2=None, tol_mm=0.1):
    """
    Get the overlapping slices of 2 data_objects. Both have to have the same orientation.
    No check for orientation so use carefully

    Parameters
    ----------
    data_obj_1: MRIExperiment or xr.DataArray
        First MRIExperiment object or xr.DataArray that contains information about the slice positions.
    data_obj_2: MRIExperiment or xr.DataArray
        Second MRIExperiment or xr.DataArrayobject that contains information about the slice positions.
    tol_mm: float, int
        Tolerance in difference between the slices in mm.

    Returns
    -------
    indice lists containing the indices of the overlapping slices:

    Example
    -------
    >>> from hypermri.utils.plot.utils_plot import get_overlapping_slices
    >>> slices = get_overlapping_slices(data_obj_1=dce,
    >>>                                 data_obj_2=t1)
    >>> fig, ax = plt.subplots(3,4)
    >>> ax=ax.flatten()
    >>> for i, s in enumerate(slices[0]):
    >>>     ax[i].imshow(np.squeeze(dce.data_array.isel(slice_idx=s, repetition_idx=10)))
    >>> fig, ax = plt.subplots(3,4)
    >>> ax=ax.flatten()
    >>> for i, s in enumerate(slices[1]):
    >>>     ax[i].imshow(np.squeeze(t1.data_array.isel(slice_idx=s, inversion_idx=0)))

    """
    import xarray as xr

    if not (isinstance(tol_mm, float) or isinstance(tol_mm, int)):
        raise ValueError(f"diff_mm has to be float or integer!")
    tol_mm = np.abs(tol_mm)
    # --- 1. Some type checking to extract axis_mm ----
    if isinstance(data_obj_1, xr.DataArray):
        data_obj_1_slice_coords = data_obj_1.slice_idx.attrs["axis_mm"]
    elif isinstance(data_obj_1, xr.Dataset):
        data_obj_1_slice_coords = data_obj_1.slice_idx.attrs["axis_mm"]
    else:
        if hasattr(data_obj_1, "data_array"):
            data_obj_1_slice_coords = data_obj_1.data_array.slice_idx.attrs["axis_mm"]
        elif hasattr(data_obj_1, "reco_data"):
            data_obj_1_slice_coords = data_obj_1.data_array.slice_idx.attrs["axis_mm"]
        else:
            print("Could not exctract axis_mm attribute from passed data_obj_1")

    if isinstance(data_obj_2, xr.DataArray):
        data_obj_2_slice_coords = data_obj_2.slice_idx.attrs["axis_mm"]
    elif isinstance(data_obj_2, xr.Dataset):
        data_obj_2_slice_coords = data_obj_2.slice_idx.attrs["axis_mm"]
    else:
        if hasattr(data_obj_2, "data_array"):
            data_obj_2_slice_coords = data_obj_2.data_array.slice_idx.attrs["axis_mm"]
        elif hasattr(data_obj_2, "reco_data"):
            data_obj_2_slice_coords = data_obj_2.data_array.slice_idx.attrs["axis_mm"]
        else:
            print("Could not exctract axis_mm attribute from passed data_obj_2")

    # --- 2. Compare the axis_mm to find overlapping slices ---
    slice_map = {}  # dce_index -> t1_index
    for i_1, c_1 in enumerate(data_obj_1_slice_coords):
        diffs = np.abs(data_obj_2_slice_coords - c_1)
        i_2 = np.argmin(diffs)
        if diffs[i_2] < tol_mm:
            # slice_map keys are dce index, slice_map values are t1 index
            slice_map[i_1] = int(i_2)

    # --- 3. Extract the indices to keep, ensuring they stay in the same relative order ---
    data_obj_1_indices = sorted(slice_map.keys())
    data_obj_2_indices = [slice_map[idx] for idx in data_obj_1_indices]
    return data_obj_1_indices, data_obj_2_indices


def create_colormap(
    hex_colors,
    name: str = "custom_cmap",
    steps_for_interpolation: int = 256,
    discrete: bool = False,
    register: bool = True,
    reverse: bool = False,
    show: bool = False,
    show_labels: bool = True,
    nan_color: str = "black",
):
    """
    Create and optionally preview a Matplotlib colormap from a list of hex colors.

    Handles Matplotlib <3.6 and >=3.6. If a colormap with the same `name`
    is already registered, registration is skipped (to avoid ValueError).

    Parameters
    ----------
    hex_colors : list of str
        Hex color strings, e.g. ['#00429d', '#96ffea', '#ffffff'].
    name : str, optional
        Colormap name to (try to) register under.
    steps_for_interpolation : int, optional
        Steps for continuous colormap interpolation (default 256).
    discrete : bool, optional
        If True use ListedColormap (discrete); else LinearSegmentedColormap.
    register : bool, optional
        If True, attempt to register with Matplotlib (default True).
    reverse : bool, optional
        If True, reverse the color order before building the cmap.
    show : bool, optional
        If True, show a small preview of the colormap.
    show_labels : bool, optional
        If True, annotate min/max under the preview.
    nan_color : str or tuple, optional
        Color for NaN values; set to None to skip (default 'black').

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The created colormap.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    if reverse:
        hex_colors = list(reversed(hex_colors))

    # Build cmap
    if discrete:
        cmap = ListedColormap(hex_colors, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(
            name, hex_colors, N=steps_for_interpolation
        )

    if nan_color is not None:
        cmap.set_bad(color=nan_color)

    # Try to register safely across Matplotlib versions
    if register:
        import matplotlib as mpl

        registry = mpl.colormaps
        try:
            # Newer Matplotlib supports override_builtin
            registry.register(cmap, name=name, override_builtin=True)
        except TypeError:
            # Older Matplotlib: no override_builtin kw
            if name in registry:
                # Name taken: skip to avoid ValueError
                pass
            else:
                registry.register(cmap, name=name)
        except ValueError:
            # Name already registered (or builtin) -> skip
            # You can choose a new `name` or set register=False if you want to reuse the object
            pass

    if show:
        fig, ax = plt.subplots(figsize=(6, 1))
        gradient = [range(cmap.N)] if not discrete else [range(len(hex_colors))]
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_axis_off()
        if show_labels:
            ax.text(0, 1.5, "min", fontsize=8, ha="left", va="bottom", color="gray")
            ax.text(
                (cmap.N if not discrete else len(hex_colors)) - 1,
                1.5,
                "max",
                fontsize=8,
                ha="right",
                va="bottom",
                color="gray",
            )
        plt.title(name, fontsize=9)
        plt.show()

    return cmap


common_cmaps = [
    "bone",  # Good general-purpose grayscale
    "gray",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "viridis",
    "jet",
    "hsv",
    "cool",
    "seismic",
    "twilight",
    "tab10",
    "prism",
    "gist_rainbow",
    "nipy_spectral",  # Perceptually uniform rainbow
    "cmc.lipari",  # Crameri's Lipari colormap
    "cmc.navia",  # Crameri's Navia colormap
]
