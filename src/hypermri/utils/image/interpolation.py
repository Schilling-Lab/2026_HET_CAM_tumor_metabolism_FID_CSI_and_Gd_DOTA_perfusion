import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import block_reduce


def interpolate_binary_mask(mask, new_shape, method="nearest", **kwargs):
    """Interpolate a binary mask to a new shape.

    Parameters
    ----------
    mask : ndarray
        The binary mask to be interpolated.
    new_shape : tuple
        The new shape (dimensions) for the interpolated mask.
    method : str, optional
        The interpolation method. Default is 'nearest'.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to RegularGridInterpolator.

    Returns
    -------
    ndarray
        The interpolated binary mask.

    Notes
    -----
    This function interpolates a binary mask to a new shape using RegularGridInterpolator
    from scipy.interpolate. The mask is assumed to contain binary values (0 or 1), and
    the interpolation is performed based on the specified method.
    """
    assert (
        np.min(mask) >= 0 and np.max(mask) <= 1
    ), "Not a binary mask. Values outside [0,1]."

    n, m, k = mask.shape
    n_new, m_new, k_new = new_shape

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n),
        np.arange(m),
        np.arange(k),
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1), n_new),
        np.linspace(0, (m - 1), m_new),
        np.linspace(0, (k - 1), k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator(
        (x, y, z), mask.astype(float), method=method, **kwargs
    )

    # Interpolate values on the new grid
    interpolated_mask = interpolator((x_new, y_new, z_new))

    # Round the values to 0 or 1
    interpolated_mask = np.round(interpolated_mask).astype(int)

    return interpolated_mask


def interpolate_3d_array(
    arr, new_dimensions, voxel_size=None, return_grid=False, **kwargs
):
    """Interpolate a(n unmasked!) 3D array onto a new grid using regular grid interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        The input 3D array
    new_dimensions : tuple
        A tuple specifying the new dimensions (n_new, m_new, k_new) of the
        interpolated array.
    voxel_size : numpy.ndarray, optional
        The voxel size of the original array. If not provided, it defaults to
        [1.0, 1.0, 1.0].
    return_grid : bool, optional
        If True, the function also returns the coordinates of the new grid.
        Defaults to False.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the RegularGridInterpolator.
        You can e.g. specify the interpolation type with method='cubic'.

    Returns
    -------
    numpy.ndarray or tuple
        - If return_grid is False (default), returns the interpolated masked
          array on the new grid.
        - If return_grid is True, returns a tuple containing the interpolated
          masked array and the coordinates of the new grid.

    Raises
    ------
    AssertionError
        If the input array is not a np.ndarray

    """
    assert isinstance(arr, np.ndarray), "We need a default numpy.ndarray mate."

    if voxel_size is None:
        voxel_size = np.array((1.0, 1.0, 1.0))

    n, m, k = arr.shape

    n_new, m_new, k_new = new_dimensions

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n) * voxel_size[0],
        np.arange(m) * voxel_size[1],
        np.arange(k) * voxel_size[2],
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1) * voxel_size[0], n_new),
        np.linspace(0, (m - 1) * voxel_size[1], m_new),
        np.linspace(0, (k - 1) * voxel_size[2], k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator((x, y, z), arr, **kwargs)

    # Interpolate the pH values on the new grid
    array_interpolated = interpolator((x_new, y_new, z_new))

    return array_interpolated


def interpolate_masked_3d_array(
    masked_array, new_dimensions, voxel_size=None, return_grid=False, **kwargs
):
    """
    Interpolate a masked 3D array onto a new grid using regular grid interpolation.

    Parameters
    ----------
    masked_array : numpy.ma.MaskedArray
        The input 3D array with missing values masked.
    new_dimensions : tuple
        A tuple specifying the new dimensions (n_new, m_new, k_new) of the
        interpolated array.
    voxel_size : numpy.ndarray, optional
        The voxel size of the original array. If not provided, it defaults to
        [1.0, 1.0, 1.0].
    return_grid : bool, optional
        If True, the function also returns the coordinates of the new grid.
        Defaults to False.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the RegularGridInterpolator.
        You can e.g. specify the interpolation type with method='cubic'.

    Returns
    -------
    numpy.ma.MaskedArray or tuple
        - If return_grid is False (default), returns the interpolated masked
          array on the new grid.
        - If return_grid is True, returns a tuple containing the interpolated
          masked array and the coordinates of the new grid.

    Raises
    ------
    AssertionError
        If the input array is not a masked array.
    """
    assert np.ma.is_masked(masked_array), "We need a masked array mate."

    if voxel_size is None:
        voxel_size = np.array((1.0, 1.0, 1.0))

    n, m, k = masked_array.shape

    n_new, m_new, k_new = new_dimensions

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n) * voxel_size[0],
        np.arange(m) * voxel_size[1],
        np.arange(k) * voxel_size[2],
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1) * voxel_size[0], n_new),
        np.linspace(0, (m - 1) * voxel_size[1], m_new),
        np.linspace(0, (k - 1) * voxel_size[2], k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator((x, y, z), masked_array, **kwargs)

    # Interpolate the pH values on the new grid
    array_interpolated = interpolator((x_new, y_new, z_new))

    # # Interpolate mask:
    # mask_interpolated = interpolate_binary_mask(masked_array.mask, new_dimensions, method='linear')

    # # Apply the interpolated mask to the interpolated data
    # masked_array_interpolated = np.ma.masked_array(
    #     array_interpolated, mask_interpolated
    # )
    # Create a separate interpolator for the mask
    mask_interpolator = RegularGridInterpolator(
        (x, y, z), masked_array.mask, bounds_error=False, fill_value=False, **kwargs
    )

    # Interpolate the mask values on the new grid
    mask_interpolated = mask_interpolator((x_new, y_new, z_new))

    # Apply the interpolated mask to the interpolated data
    masked_array_interpolated = np.ma.masked_array(
        array_interpolated, mask_interpolated
    )

    if return_grid:
        return masked_array_interpolated, (x_new, y_new, z_new)
    return masked_array_interpolated


def downsample_masked_array(masked_array, downsampling_factor):
    """
    Downsamples a masked array using scikit-image's block_reduce function while preserving the mask information.

    Parameters
        masked_array (numpy.ma.masked_array): The original 3D masked array.
        downsampling_factor (tuple): The downsampling factor for each dimension.

    Returns
        numpy.ma.masked_array: The downsampled masked array.

    Notes
        This function uses `skimage.measure.block_reduce` to downsample the array
        by applying a block-wise reduction operation. It avoids interpolation
        artifacts and is well-suited for downsampling with fixed downsampling
        factors. (And more robust to noise!) Unlike `scipy.ndimage.zoom`,
        `block_reduce` only supports downsampling and does not perform
        interpolation or upsampling.
    """
    # Downsample the array
    downsampled_array = block_reduce(masked_array.data, downsampling_factor, np.mean)
    downsampled_mask = block_reduce(masked_array.mask, downsampling_factor, np.any)

    # Create a new masked array with the downsampled data and mask
    downsampled_masked_array = np.ma.masked_array(
        downsampled_array, mask=downsampled_mask
    )

    return downsampled_masked_array


def img_interp(
    input_data=None,
    interp_factor=2,
    interp_size=None,
    cmap=None,
    overlay=True,
    interp_method="bilinear",
    threshold=0,
):
    """
    Interpolates a 2D input image array by a given factor or to a specific size.

    This function interpolates a 2D image array using various interpolation methods supported by
    `matplotlib`, optionally applies a colormap, and masks values below a certain threshold if
    overlay is enabled.

    Parameters
    ----------
    input_data : numpy.ndarray, optional
        Input 2D image array to be interpolated.
    interp_factor : float, optional
        Scaling factor for the interpolation. Ignored if `interp_size` is provided.
    interp_size : tuple of int, optional
        Desired output dimensions of the image as (height, width). If provided, this
        overrides `interp_factor`.
    cmap : str, optional
        Name of the colormap to apply to the image. If None, no colormap is applied.
    overlay : bool, optional
        If True, applies a mask where values below `threshold` are set to NaN. Default is True.
    interp_method : str, optional
        Interpolation method to use. Available methods include 'nearest', 'bilinear' (default),
        'bicubic', 'spline16', 'spline36', and others supported by `matplotlib`. Each offers
        different trade-offs between quality and performance.
    threshold : float, optional
        Threshold value for masking in overlay mode. Values below this threshold are set to NaN.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        The interpolated and potentially masked 2D image array.

    Raises
    ------
    ValueError
        If `interp_method` is not supported, or if the input `input_data` is
        not a 2D array.
    TypeError
        If `interp_factor` is not a float or `interp_size` is not a tuple of integers.

    Examples
    --------
    >>> test_image = np.random.rand(20, 20)
    >>> result1 = img_interp(test_image, interp_factor=2, cmap='gray', interp_method='bilinear')
    >>> result2 = img_interp(test_image, interp_size=(40, 40), cmap='gray', interp_method='lanczos')

    Notes
    -----
    - The function uses matplotlib's `imshow` and internal `_resample` functions for image resizing.
    - If both `interp_factor` and `interp_size` are provided, `interp_size` takes precedence.
    """

    # Input validation
    if input_data is None:
        raise ValueError("Input image array `input_data` must be provided.")

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
        if input_data.ndim != 2:
            raise ValueError("`input_data` must be a 2D numpy array.")

    # Define a set of valid interpolation methods
    valid_interp_methods = {
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
    }

    # Check if interp_method is valid
    if interp_method not in valid_interp_methods:
        raise ValueError(
            f"Invalid interpolation method '{interp_method}'. Valid methods are: {valid_interp_methods}."
        )

    if interp_factor is not None and not isinstance(interp_factor, (float, int)):
        raise TypeError("`interp_factor` must be a float or int.")

    if interp_size is not None and not (
        isinstance(interp_size, tuple) and len(interp_size) == 2
    ):
        raise TypeError(
            "`interp_size` must be a tuple of two integers representing the output size."
        )

    # Skip interpolation if interpolation factor is 1 and no interp_size is
    # provided
    if interp_factor == 1 and interp_size is None:
        if overlay:
            input_data[input_data <= threshold] = np.nan
        return input_data

    # Import necessary libraries
    from matplotlib.image import _resample
    from matplotlib.transforms import Affine2D

    if interp_factor:
        # Define new output dimensions
        out_dimensions = (
            int(input_data.shape[0] * interp_factor),
            int(input_data.shape[1] * interp_factor),
        )

        input_data_max = np.max(input_data)
        input_data_min = np.min(input_data)

        # Normalize image (required for _resample)
        input_data = (input_data - input_data_min) / (input_data_max - input_data_min)

        # Create axis object and apply transformation
        _, axs = plt.subplots(1)
        transform = Affine2D().scale(interp_factor, interp_factor)
        img = axs.imshow(input_data, interpolation=interp_method, cmap=cmap)

        # Perform resampling
        interpolated = _resample(img, input_data, out_dimensions, transform=transform)

        # Clean up
        axs.clear()
        plt.close()

        # Rescale to original range
        interpolated = (
            interpolated * (input_data_max - input_data_min)
        ) + input_data_min

    else:
        # Use interp_size for interpolation
        out_dimensions = interp_size
        interp_factor_0 = interp_size[0] / input_data.shape[0]
        interp_factor_1 = interp_size[1] / input_data.shape[1]

        if interp_factor_0 != interp_factor_1:
            logging.warning(
                f"Desired interp_size {interp_size} is not a uniform scaling of input_data size {input_data.shape}."
            )

        input_data_max = np.max(input_data)
        input_data = input_data / input_data_max  # Normalize

        _, axs = plt.subplots(1)
        transform = Affine2D().scale(interp_factor_0, interp_factor_1)
        img = axs.imshow(input_data, interpolation=interp_method, cmap=cmap)

        interpolated = _resample(img, input_data, out_dimensions, transform=transform)

        axs.clear()
        plt.close()

        # Rescale
        interpolated = interpolated * input_data_max

    # Apply masking if overlay is enabled
    if overlay:
        interpolated[interpolated <= threshold] = np.nan

    return interpolated
