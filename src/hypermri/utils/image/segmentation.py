import pathlib

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import __version__ as mpl_version
from matplotlib import get_backend
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS, to_rgba_array
from matplotlib.path import Path
from matplotlib.pyplot import ioff
from matplotlib.widgets import LassoSelector

# Handle numpy.distutils compatibility issue with mpl_interactions
try:
    from mpl_interactions import figure, panhandler, zoom_factory

    MPL_INTERACTIONS_AVAILABLE = True
except ImportError as e:
    if "numpy.distutils" in str(e):
        # Create dummy objects to prevent import errors
        panhandler = None
        zoom_factory = None
        figure = None
        MPL_INTERACTIONS_AVAILABLE = False
        import warnings

        warnings.warn(
            "mpl_interactions not available due to numpy.distutils incompatibility. "
            "3D interactive segmentation features will be disabled.",
            ImportWarning,
            stacklevel=2,
        )
    else:
        raise e


class image_segmenter_3d:
    """
    Manually segment 3D image volume slice by slice with the lasso selector.
    (2D works as well, note however that there is already a native 2D segmenter
    from the mpl_interactions library)
    """

    def __init__(
        self,
        img3d,
        nclasses=1,
        slicedim=None,
        class_names=None,
        mask=None,
        mask_colors=None,
        mask_alpha=0.75,
        lineprops=None,
        props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        figsize=(7, 7),
        **kwargs,
    ):
        """
        Create an image segmenter. Any ``kwargs`` will be passed through to the ``imshow``
        call that displays *img*.

        Modification of the mpl_interactions.image_segmenter() which works for a single
        2D image.

        Parameters
        ----------
        img3d : array_like
            A 3D array of shape NxAxB with N 2D image slices of size AxB.
            See 'slicedim' argument below, in case N is not the first dimension.
        nclasses : int, optional
            Number of ROIs/classes you want to draw. Default is one.
        class_names : list, optional
            List of ROI names. Number of names must match 'nclasses'.
        slicedim : int, optional
            Specifies the dimension/axis of the slices. This is helpful in case
            the slices are not stored in the first dimension. E.g.:
                NxAxB --> slicedim=0 (default)
                AxNxB --> slicedim=1
                AxBxN --> slicedim=2
        mask : arraylike, optional
            If you want to pre-seed the mask
        mask_colors : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be totally transparent
        mask_alpha : float, default .75
            The alpha values to use for selected regions. This will always override the alpha values
            in mask_colors if any were passed
        lineprops : dict, default: None
            DEPRECATED - use props instead.
            lineprops passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        props : dict, default: None
            props passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        lasso_mousebutton : str, or int, default: "left"
            The mouse button to use for drawing the selecting lasso.
        pan_mousebutton : str, or int, default: "middle"
            The button to use for `~mpl_interactions.generic.panhandler`. One of 'left', 'middle' or
            'right', or 1, 2, 3 respectively.
        ax : `matplotlib.axes.Axes`, optional
            The axis on which to plot. If *None* a new figure will be created.
        figsize : (float, float), optional
            passed to plt.figure. Ignored if *ax* is given.
        **kwargs
            All other kwargs will passed to the imshow command for the image
        """
        # Check if interactive functionality is available
        if not MPL_INTERACTIONS_AVAILABLE:
            raise ImportError(
                "Interactive 3D segmentation functionality is not available. "
                "This is likely due to a numpy.distutils compatibility issue with mpl_interactions. "
                "Please use alternative segmentation methods or downgrade numpy to <2.0."
            )

        self.slice_dim = slicedim
        self.imshow_kwargs = kwargs
        # ensure mask colors is iterable and the same length as the number of classes
        # choose colors from default color cycle?

        self.mask_alpha = mask_alpha

        if mask_colors is None:
            # this will break if there are more than 10 classes
            if nclasses <= 10:
                self.mask_colors = to_rgba_array(list(TABLEAU_COLORS)[:nclasses])
            else:
                # up to 949 classes. Hopefully that is always enough....
                self.mask_colors = to_rgba_array(list(XKCD_COLORS)[:nclasses])
        else:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            # should probably check the shape here
        self.mask_colors[:, -1] = self.mask_alpha

        self._img3d = np.asanyarray(img3d)
        # in case of a 2D array, convert to 3D
        if np.ndim(self._img3d) == 2:
            self._img3d = self._img3d[np.newaxis, ...]
        # rearrange axis to ensure slicedim is first if necessary
        if self.slice_dim:
            self._img3d = np.moveaxis(self._img3d, self.slice_dim, 0)

        self.nslices = self._img3d.shape[0]
        self.current_slice = 0

        if mask is None:
            self.mask = np.zeros(self._img3d.shape)
            """See :doc:`/examples/image-segmentation`."""
        else:
            self.mask = mask

        self._overlay = np.zeros((*self._img3d.shape, 4))

        self.nclasses = nclasses
        self.class_names = class_names
        if self.class_names is not None:
            assert self.nclasses == len(
                self.class_names
            ), f"nclasses != len(class_names) -> [{nclasses} != {len(class_names)}]"

        for i in range(nclasses + 1):
            idx = self.mask == i
            if i == 0:
                self._overlay[idx] = [0, 0, 0, 0]
            else:
                self._overlay[idx] = self.mask_colors[i - 1]

        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with ioff():
                self.fig = figure(figsize=figsize)
                self.ax = self.fig.gca()

        self.displayed = self.ax.imshow(
            self._curr_image, animated=True, **self.imshow_kwargs
        )
        self._mask = self.ax.imshow(self._curr_overlay, animated=True)

        self._setup_widgets()

        default_props = {"color": "black", "linewidth": 1, "alpha": 0.8}
        if (props is None) and (lineprops is None):
            props = default_props
        elif (lineprops is not None) and (mpl_version >= "3.7"):
            print("*lineprops* is deprecated - please use props")
            props = {"color": "black", "linewidth": 1, "alpha": 0.8}

        useblit = False if "ipympl" in get_backend().lower() else True
        button_dict = {"left": 1, "middle": 2, "right": 3}
        if isinstance(pan_mousebutton, str):
            pan_mousebutton = button_dict[pan_mousebutton.lower()]
        if isinstance(lasso_mousebutton, str):
            lasso_mousebutton = button_dict[lasso_mousebutton.lower()]

        if mpl_version < "3.7":
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                lineprops=props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        else:
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                props=props,
                useblit=useblit,
            )
        self.lasso.set_visible(True)

        pix_x = np.arange(self._img3d.shape[1])
        pix_y = np.arange(self._img3d.shape[2])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.ph = panhandler(self.fig, button=pan_mousebutton)
        self.disconnect_zoom = zoom_factory(self.ax)
        self.current_class = 1
        self.erasing = False

    @property
    def _curr_image(self):
        return self._img3d[self.current_slice]

    @property
    def _curr_mask(self):
        return self.mask[self.current_slice]

    @property
    def _curr_overlay(self):
        return self._overlay[self.current_slice]

    def _setup_widgets(self):
        """Create widgets for slice navigation, erasing mode, roi class..."""
        # roi-class selector
        class_options = list(range(1, self.nclasses + 1))

        if self.class_names is not None:
            class_options = [(self.class_names[i - 1], i) for i in class_options]

        self.wgt_class_selector = widgets.Dropdown(
            options=class_options, description="ROI: "
        )
        # button that acts as color indicator, same height as roi-class
        # selector
        self.wgt_color_indicator = widgets.Button(
            disabled=False,
            tooltip="Color of current ROI.",
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="30px"
            ),
        )
        # button for copy&pasting ROIs
        self.wgt_copy_button = widgets.Button(
            description="",
            disabled=False,
            tooltip="Copy all ROIs from this slice.",
            icon="copy",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="40px"
            ),
        )
        self.wgt_paste_button = widgets.Button(
            description="",
            disabled=True,
            tooltip="Paste all ROIs into this slice.",
            icon="paste",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="40px"
            ),
        )
        # button that acts as color indicator, same height as roi-class
        # selector
        self.wgt_color_indicator = widgets.Button(
            disabled=False,
            tooltip="Color of current ROI.",
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="30px"
            ),
        )
        # erasing button
        self.wgt_erasing_button = widgets.ToggleButton(
            value=False,
            description="ERASING",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Click to start erasing from mask.",
            icon="eraser",  # (FontAwesome names without the `fa-` prefix)
        )
        # slice slider
        self.wgt_slice_slider = widgets.IntSlider(
            value=0, min=0, max=self.nslices - 1, description="Slice: "
        )
        # widget container
        ui_editing = widgets.HBox(
            [
                self.wgt_erasing_button,
                self.wgt_copy_button,
                self.wgt_paste_button,
            ],
            layout=widgets.Layout(display="flex", justify_content="center"),
        )
        ui_selection = widgets.HBox(
            [
                self.wgt_class_selector,
                self.wgt_color_indicator,
            ],
            layout=widgets.Layout(display="flex", justify_content="center"),
        )
        self.ui = widgets.HBox(
            [self.wgt_slice_slider, ui_editing, ui_selection],
            layout=widgets.Layout(display="flex", justify_content="space-between"),
        )
        # connect buttons and other widgets with functions
        self.wgt_copy_button.on_click(self._copy_mask_from_slice)
        self.wgt_paste_button.on_click(self._paste_mask_to_slice)

        self.sliders = widgets.interactive_output(
            self._draw_img,
            {
                "slice_nr": self.wgt_slice_slider,
                "erasing_mode": self.wgt_erasing_button,
                "class_nr": self.wgt_class_selector,
            },
        )

    def _copy_mask_from_slice(self, btn):
        """Callback function for copy button."""
        self._clipboard = np.ma.copy(self._curr_mask), np.ma.copy(self._curr_overlay)
        self.wgt_copy_button.button_style = "info"
        self.wgt_paste_button.disabled = False

    def _paste_mask_to_slice(self, btn):
        """Callback function for paste button."""
        cp_mask, cp_overlay = self._clipboard

        self.mask[self.current_slice] = cp_mask
        self._overlay[self.current_slice] = cp_overlay

        self.displayed.set_data(self._curr_image)
        self._mask.set_data(self._curr_overlay)
        self.fig.canvas.draw_idle()

    def _draw_img(self, **kwargs):
        """Callback functions for (non-button) widgets."""
        self.erasing = kwargs.get("erasing_mode")
        self.current_slice = kwargs.get("slice_nr")
        self.current_class = kwargs.get("class_nr")

        self.wgt_erasing_button.button_style = "danger" if self.erasing else ""

        # convert rgba() [0-1] to rgb [0-255] for ipywidget button color
        r, g, b, _ = self.mask_colors[self.current_class - 1] * 255
        bcolor = f"rgb({int(r)}, {int(g)}, {int(b)})"
        self.wgt_color_indicator.style.button_color = bcolor

        self.displayed.set_data(self._curr_image)
        self._mask.set_data(self._curr_overlay)
        self.fig.canvas.draw_idle()

    def _onselect(self, verts):
        """Callback function for Lasso Selector."""
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(
            self._curr_mask.shape
        )
        if self.erasing:
            self._curr_mask[self.indices] = 0
            self._curr_overlay[self.indices] = [0, 0, 0, 0]
        else:
            self._curr_mask[self.indices] = self.current_class
            self._curr_overlay[self.indices] = self.mask_colors[self.current_class - 1]

        self._mask.set_data(self._curr_overlay)
        self.fig.canvas.draw_idle()

    def _ipython_display_(self):
        display(self.ui, self.sliders, self.fig.canvas)

    def get_mask(self, key=None):
        """Get mask of a certain class only. Key can be an int or from self.class_names."""
        if key is None:
            return self.mask
        elif isinstance(key, int) and 1 <= key <= self.nclasses:
            return self.mask == key
        elif key in self.class_names:
            return self.mask == self.class_names.index(key) + 1
        else:
            raise AssertionError(f"'{key}' is not a valid class key.")

    def save_mask(self, fpath, key=None):
        """Save numpy array of mask to given filename/path.

        fpath : str or path
           File path the masked is saved to. In case fpath is only a filename,
           mask is saved to './fpath.npy'.
        key : str or int, optional
            By default the complete mask, containting all classes/rois is stored.
            With key you can explicitly pick a single class/roi to save.
        """
        # obtain array to be saved
        out = self.get_mask(key=key)

        # do all the file path handling
        save_path = pathlib.Path(fpath)

        if not save_path.parent.is_dir():
            raise AssertionError(f"Directory {save_path.parent} does not exist")

        if not save_path.name.endswith(".npy"):
            save_path = save_path.parent / (save_path.name + ".npy")

        # Ask for confirmation, then store and give a nice little success msg.
        if input(f"Saving to '{save_path}'.\nPress enter to continue..."):
            print("Aborted!")
        else:
            np.save(save_path, out)
            nbytes = out.nbytes + 128  # array size + 128 bytes filesize
            if nbytes < 1e4:
                sz = f"{nbytes * 1e-3:.2f} KB"
            elif nbytes < 1e7:
                sz = f"{nbytes * 1e-6:.2f} MB"
            else:
                sz = f"{nbytes * 1e-9:.2f} GB"

            print(f"Mask saved! ({sz})")

        if self.slice_dim:
            print(
                "WARNING: {self.slice_dim} != None not implemented for saving yet. -> Slice Dim at dim 0."
            )


class image_segmenter_3d_overlay(image_segmenter_3d):
    """
    Manually segment 3D image volume slice by slice with the lasso selector.
    Additionally you can overlay a second image.
    (2D works as well, note however that there is already a native 2D segmenter
    from the mpl_interactions library -> mpl_interactions.image_segmenter())
    """

    def __init__(
        self,
        img3d,
        img3d_overlay,
        nclasses=1,
        class_names=None,
        slicedim=None,
        mask=None,
        mask_colors=None,
        mask_alpha=0.75,
        lineprops=None,
        props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        figsize=(10, 10),
        params=None,
        params_overlay=None,
    ):
        """
        Create an image segmenter with an overlayed image. Any ``kwargs`` will
        be passed through to the ``imshow`` call that displays *img*.

        Modification of the mpl_interactions.image_segmenter().

        Parameters
        ----------
        img3d : array_like
            A 3D array of shape NxAxB with N 2D image slices of size AxB.
            See 'slicedim' argument below, in case N is not the first dimension.
        img3d_overlay : array_like
            A 3D array of shape NxCxD with N 2D image slices of size CxD.
            Must have the same number of slices as img3d.
        nclasses : int, optional
            Number of ROIs/classes you want to draw. Default is one.
        class_names : list, optional
            List of ROI names. Number of names must match 'nclasses'.
        slicedim : int, optional
            Specifies the dimension/axis of the slices. This is helpful in case
            the slices are not stored in the first dimension. E.g.:
                NxAxB --> slicedim=0 (default)
                AxNxB --> slicedim=1
                AxBxN --> slicedim=2
        mask : arraylike, optional
            If you want to pre-seed the mask
        mask_colors : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be totally transparent
        mask_alpha : float, default .75
            The alpha values to use for selected regions. This will always override the alpha values
            in mask_colors if any were passed
        lineprops : dict, default: None
            DEPRECATED - use props instead.
            lineprops passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        props : dict, default: None
            props passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        lasso_mousebutton : str, or int, default: "left"
            The mouse button to use for drawing the selecting lasso.
        pan_mousebutton : str, or int, default: "middle"
            The button to use for `~mpl_interactions.generic.panhandler`. One of 'left', 'middle' or
            'right', or 1, 2, 3 respectively.
        ax : `matplotlib.axes.Axes`, optional
            The axis on which to plot. If *None* a new figure will be created.
        figsize : (float, float), optional
            passed to plt.figure. Ignored if *ax* is given.
        **kwargs
            All other kwargs will passed to the imshow command for the image
        """
        self.slice_dim = slicedim
        self.imshow_kwargs = params if params is not None else dict()
        self.imshow_kwargs_overlay = (
            params_overlay if params_overlay is not None else dict()
        )
        # ensure mask colors is iterable and the same length as the number of classes
        # choose colors from default color cycle?

        self.mask_alpha = mask_alpha

        if mask_colors is None:
            # this will break if there are more than 10 classes
            if nclasses <= 10:
                self.mask_colors = to_rgba_array(list(TABLEAU_COLORS)[:nclasses])
            else:
                # up to 949 classes. Hopefully that is always enough....
                self.mask_colors = to_rgba_array(list(XKCD_COLORS)[:nclasses])
        else:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            # should probably check the shape here
        self.mask_colors[:, -1] = self.mask_alpha

        self._img3d = np.asanyarray(img3d)
        self._img3d_overlay = np.asanyarray(img3d_overlay)

        # in case of a 2D array, convert to 3D
        if np.ndim(self._img3d) == 2:
            self._img3d = self._img3d[np.newaxis, ...]
            self._img3d_overlay = self._img3d_overlay[np.newaxis, ...]
        # rearrange axis to ensure slicedim is first if necessary
        if self.slice_dim:
            self._img3d = np.moveaxis(self._img3d, self.slice_dim, 0)
            self._img3d_overlay = np.moveaxis(self._img3d_overlay, self.slice_dim, 0)

        # The images do not have to be of equal shape, this can be handled by
        # the user via the image extent keyword argument and the built-in
        # imshow() interpolation. However, both images must have an equal amount
        # of slices.
        na, nb = self._img3d.shape[0], self._img3d_overlay.shape[0]
        assert na == nb, f"Images must have an equal amount of slices. {na} != {nb}."

        self.nslices = self._img3d.shape[0]
        self.current_slice = 0

        if mask is None:
            self.mask = np.zeros(self._img3d_overlay.shape)
            """See :doc:`/examples/image-segmentation`."""
        else:
            self.mask = mask

        # mask overlay, not to be confused with the img3d_overlay
        self._overlay = np.zeros((*self._img3d_overlay.shape, 4))

        self.nclasses = nclasses
        self.class_names = class_names
        if self.class_names is not None:
            assert self.nclasses == len(
                self.class_names
            ), f"nclasses != len(class_names) -> [{nclasses} != {len(class_names)}]"

        for i in range(nclasses + 1):
            idx = self.mask == i
            if i == 0:
                self._overlay[idx] = [0, 0, 0, 0]
            else:
                self._overlay[idx] = self.mask_colors[i - 1]

        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with ioff():
                self.fig = figure(figsize=figsize)
                self.ax = self.fig.gca()

        i3s = self._img3d_overlay.shape
        extent_overlay = [-0.5, i3s[2] - 0.5, -0.5, i3s[1] - 0.5]
        # assuming the same in plane FOV:
        # could theoretically be scaled in case of a different FOV
        # overlay: image with AxB matrix size and FaxFb FOV[mm]
        # image: image with CxD matrix aize and FcxFd FOV[mm]
        #
        # --> a_pix_per_mm = A / Fa
        # --> length_a = A
        # --> lenght_c = a_pix_per_mm * Fc
        # --> diff = length_c - length_a
        # --> extent = [- diff / 2 - 0.5, length_c + diff / 2 - 0.5, ...]
        extent = extent_overlay

        # extent_min
        self.displayed = self.ax.imshow(
            self._curr_image, animated=True, extent=extent, **self.imshow_kwargs
        )
        self.displayed_overlay = self.ax.imshow(
            self._curr_image_overlay,
            animated=True,
            extent=extent_overlay,
            **self.imshow_kwargs_overlay,
        )
        self._mask = self.ax.imshow(
            self._curr_overlay, animated=True, extent=extent_overlay, origin="lower"
        )

        # self.ax.set_xlim(extent[0], extent[1])
        # self.ax.set_ylim(extent[2], extent[3])

        self._setup_widgets()

        default_props = {"color": "black", "linewidth": 1, "alpha": 0.8}
        if (props is None) and (lineprops is None):
            props = default_props
        elif (lineprops is not None) and (mpl_version >= "3.7"):
            print("*lineprops* is deprecated - please use props")
            props = {"color": "black", "linewidth": 1, "alpha": 0.8}

        useblit = False if "ipympl" in get_backend().lower() else True
        button_dict = {"left": 1, "middle": 2, "right": 3}
        if isinstance(pan_mousebutton, str):
            pan_mousebutton = button_dict[pan_mousebutton.lower()]
        if isinstance(lasso_mousebutton, str):
            lasso_mousebutton = button_dict[lasso_mousebutton.lower()]

        if mpl_version < "3.7":
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                lineprops=props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        else:
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                props=props,
                useblit=useblit,
            )
        self.lasso.set_visible(True)

        pix_x = np.arange(self._img3d_overlay.shape[1])
        pix_y = np.arange(self._img3d_overlay.shape[2])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.ph = panhandler(self.fig, button=pan_mousebutton)
        self.disconnect_zoom = zoom_factory(self.ax)
        self.current_class = 1
        self.erasing = False

    @property
    def _curr_image_overlay(self):
        return self._img3d_overlay[self.current_slice]

    def _setup_widgets(self):
        """Create widgets for slice navigation, erasing mode, roi class..."""
        # roi-class selector
        class_options = list(range(1, self.nclasses + 1))

        if self.class_names is not None:
            class_options = [(self.class_names[i - 1], i) for i in class_options]

        self.wgt_class_selector = widgets.Dropdown(
            options=class_options, description="ROI: "
        )
        # button for copy&pasting ROIs
        self.wgt_copy_button = widgets.Button(
            description="",
            disabled=False,
            tooltip="Copy all ROIs from this slice.",
            icon="copy",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="40px"
            ),
        )
        self.wgt_paste_button = widgets.Button(
            description="",
            disabled=True,
            tooltip="Paste all ROIs into this slice.",
            icon="paste",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="40px"
            ),
        )
        # button that acts as color indicator, same height as roi-class
        # selector
        self.wgt_color_indicator = widgets.Button(
            disabled=False,
            tooltip="Color of current ROI.",
            layout=widgets.Layout(
                height=self.wgt_class_selector.layout.height, width="30px"
            ),
        )
        # erasing button
        self.wgt_erasing_button = widgets.ToggleButton(
            value=False,
            description="ERASING",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Click to start erasing from mask.",
            icon="eraser",  # (FontAwesome names without the `fa-` prefix)
        )
        # slice slider
        self.wgt_slice_slider = widgets.IntSlider(
            value=0, min=0, max=self.nslices - 1, description="Slice: "
        )
        # opacity slider for overlay
        self.wgt_opacity_slider = widgets.FloatSlider(
            value=0.2, min=0, max=1, description="Opacity: "
        )
        # widget container
        ui_editing = widgets.HBox(
            [
                self.wgt_erasing_button,
                self.wgt_copy_button,
                self.wgt_paste_button,
            ],
            layout=widgets.Layout(display="flex", justify_content="center"),
        )
        ui_selection = widgets.HBox(
            [
                self.wgt_class_selector,
                self.wgt_color_indicator,
            ],
            layout=widgets.Layout(display="flex", justify_content="center"),
        )
        self.ui = widgets.HBox(
            [
                self.wgt_slice_slider,
                self.wgt_opacity_slider,
                ui_editing,
                ui_selection,
            ],
            layout=widgets.Layout(display="flex", justify_content="space-between"),
        )
        # connect buttons and other widgets with functions
        self.wgt_copy_button.on_click(self._copy_mask_from_slice)
        self.wgt_paste_button.on_click(self._paste_mask_to_slice)

        self.sliders = widgets.interactive_output(
            self._draw_img,
            {
                "slice_nr": self.wgt_slice_slider,
                "erasing_mode": self.wgt_erasing_button,
                "class_nr": self.wgt_class_selector,
                "opacity": self.wgt_opacity_slider,
            },
        )

    def _draw_img(self, **kwargs):
        """Callback functions for (non-button) widgets."""
        self.erasing = kwargs.get("erasing_mode")
        self.current_slice = kwargs.get("slice_nr")
        self.current_class = kwargs.get("class_nr")
        opacity = kwargs.get("opacity")

        # indicate wether in erasing mode or not
        self.wgt_erasing_button.button_style = "danger" if self.erasing else ""

        # convert rgba() [0-1] to rgb [0-255] for ipywidget button color
        r, g, b, _ = self.mask_colors[self.current_class - 1] * 255
        bcolor = f"rgb({int(r)}, {int(g)}, {int(b)})"
        self.wgt_color_indicator.style.button_color = bcolor

        # update plot-image data
        self.displayed.set_data(self._curr_image)
        self.displayed_overlay.set_data(self._curr_image_overlay)
        self.displayed_overlay.set_alpha(opacity)
        self._mask.set_data(self._curr_overlay)

        self.fig.canvas.draw_idle()
