##########################################################################

# Statistical analysis of datasets

##########################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np


def bruker_generate_gradient_file(
    grad_array: np.ndarray = None,
    savepath: str = None,
    savename: str = None,
    sampling_dt: float = 10e-6,
    gradient_duration: float = 10 * 1e-3,
    save_as_gp_file: bool = False,
    db: bool = False,
):
    """
    Generate a gradient file that can be read by ParaVision.
    copy the generated file to: `/opt/PV7.0.0/prog/curdir/nmrsu/ParaVision/exp/lists/gp/` to make it available

    Parameters
    ----------
    grad_array: array
        1D Gradient shape array. will be normalized to be [-1, 1]
    savepath: str
        Path where to store the generated gradient file
    savename: str
        Name of the generated gradient shape file. file-type will be "number-of-points"
    sampling_dt: float
        Scanner gradient sampling rate. Min @ Bruker 7T = 8.6us
    gradient_duration: float
        Total duration of the generated gradient file. This will determine the number of points
    save_as_gp_file: bool
        If True, the file will be saved as a gp file, which is a text file with the gradient shape values
        If False, the file will be saved as a number-of-points file, which is a text file with the gradient shape values
    db: bool
        If True, a plot will be generated showing the gradient shape before and after interpolation

    Returns
    -------
    None

    Notes
    -----
    The gradient shape will be interpolated to match the desired gradient duration. This is done to ensure that the gradient shape is sampled at the correct rate.

    """
    from scipy.interpolate import interp1d

    if grad_array is None:
        warnings.warn("Please provide grad_array")
    if np.ndim(grad_array) != 1:
        warnings.warn(f"grad_array has to be 1D but is {np.ndim(grad_array)}")

    if sampling_dt < 8.6 * 1e-6:
        warnings.warn(
            f"Gradient sampling time of {sampling_dt}, is too small, setting it to 8.6 us"
        )
        sampling_dt = 8.6

    # get number of points:
    npoints = len(grad_array)
    # calculate the total duration
    total_duration = npoints * sampling_dt
    # reduce number of points if resulting sampling rate would be too high:
    if total_duration >= gradient_duration:
        # define number of points that the gradient file should have to match
        # the desired sampling rate:
        desired_npoints = int(np.floor(gradient_duration / sampling_dt))

        # Generate 2 time axis, one that was provided, one that the gradient
        # shape will be interpolated onto:
        desired_time_axis = np.linspace(0, 1, desired_npoints)
        current_time_axis = np.linspace(0, 1, npoints)

        # Create interpolation function:
        interpolation_func = interp1d(current_time_axis, grad_array, kind="linear")

        # Save of gradient array:
        grad_array_old = grad_array.copy()

        # Interpolate data
        grad_array = interpolation_func(desired_time_axis)
        print(
            f"npoints before := {len(current_time_axis)}, npoints after={len(desired_time_axis)}"
        )

        # plot before and after interpolation:
        if db:
            fig, ax = plt.subplots(1, 3)
            ax[0].plot(current_time_axis, grad_array_old)
            ax[1].plot(desired_time_axis, grad_array)
            ax[2].plot(current_time_axis, grad_array_old)
            ax[2].plot(desired_time_axis, grad_array)
    else:
        desired_npoints = npoints

    def normalize_grad(input_grad_array):
        input_grad_array = input_grad_array / np.max(input_grad_array)
        return input_grad_array

    def format_number_E_minus_01(num):
        # Format the number to move the decimal point one digit to the left
        adjusted_number = num * 10
        return f"{adjusted_number:.6f}E-01"

    normalized_grad_array = normalize_grad(input_grad_array=grad_array)
    formatted_numbers = [format_number_E_minus_01(num) for num in normalized_grad_array]

    # Importatnt to not put an offset here and keep this part
    header = f"""##TITLE=
##JCAMP-DX= 5.00 Bruker JCAMP library
##DATA TYPE= Shape Data
##ORIGIN= Bruker BioSpin GmbH
##OWNER= <demo>
##DATE= 2008/09/08
##TIME= 13:20:52
##$SHAPE_PARAMETERS= Type: Sinus ; Number of Cycles 1.0 ; Phase Angle [deg] 0.0
##MINX= 0.000000E00
##MAXX= 1.000000E02
##MINY= 0.000000E00
##MAXY= 1.000000E00
##$SHAPE_EXMODE= Gradient
##$SHAPE_TOTROT= 9.000000E01
##$SHAPE_TYPE= None
##$SHAPE_USER_DEF=
##$SHAPE_REPHFAC=
##$SHAPE_BWFAC= 0.000000E00
##$SHAPE_BWFAC50=
##$SHAPE_INTEGFAC= 6.3656674E-01
##$SHAPE_MODE= 0
##NPOINTS= {desired_npoints}
##XYDATA= (X++(Y..Y))
"""
    if save_as_gp_file is False:
        footer = "##END="
        filename = f"{savename}.{desired_npoints}"
        savename = os.path.join(savepath, filename)
        # Write to file:
        with open(savename, "w") as file:
            file.write(header)
            for number in formatted_numbers:
                file.write(number + "\n")
            file.write(footer)
        file.close()
    else:
        filename = f"{savename}.gp"
        savename = os.path.join(savepath, filename)
        # Write to file:
        with open(savename, "w") as file:
            for number in formatted_numbers:
                file.write(number + "\n")
        file.close()


def bruker_generate_rfpulse_file(
    rf_array: np.ndarray = None,
    savepath: str = None,
    savename: str = None,
    sampling_dt: float = 10e-6,
    rf_duration: float = 10 * 1e-3,
    shape_mode: int = 0,
    save_as: str = "exc",
    db: bool = False,
):
    """
    Generates a rf file that can be read by ParaVision
    copy the generated file to:
    `/opt/PV7.0.0/prog/curdir/nmrsu/ParaVision/exp/lists/wave/`

    Parameters
    ----------
    grad_array: array
        1D rf pulse shape array. will be normalized to be [-1, 1] (amp) and [0, 360] (phase)
    savepath: str
        Path where to store the generated rfpulse file
    savename: str
        Name of the generated rfpulse file. file-type will be "number-of-points"
    sampling_dt: float
        Scanner rf sampling rate. Min @ Bruker 7T = 8.6us
    rf_duration: float
        Total duration of the generated rf file. This will determine the number of points
    shape_mode: integer
        Set to 0 for excitation.
    save_as: string
        Can be "exc", "ref". If something else, the file will be saved as a number-of-points file,
        which is a text file with the rf shape values
    db: bool
        If True, a plot will be generated showing the rf shape before and after interpolation

    Returns
    -------
    None


    """
    from scipy.interpolate import interp1d

    if rf_array is None:
        warnings.warn("Please provide rf_array")
    if np.ndim(rf_array) != 1:
        warnings.warn(f"rf_array has to be 1D but is {np.ndim(rf_array)}")

    if sampling_dt < 8.6 * 1e-6:
        warnings.warn(
            f"rf sampling time of {sampling_dt}, is too small, setting it to 8.6 us"
        )
        sampling_dt = 8.6

    # Normalize to max=100:
    rf_array = rf_array / np.max(np.abs(rf_array)) * 100.0

    # get number of points:
    npoints = len(rf_array)
    # calculate the total duration
    total_duration = npoints * sampling_dt
    # reduce number of points if resulting sampling rate would be too high:
    if total_duration >= rf_duration:
        # define number of points that the rf file should have to match the
        # desired sampling rate:
        desired_npoints = int(np.floor(rf_duration / sampling_dt))

        # Generate 2 time axis, one that was provided, one that the rf shape
        # will be interpolated onto:
        desired_time_axis = np.linspace(0, 1, desired_npoints)
        current_time_axis = np.linspace(0, 1, npoints)

        # Create interpolation function:
        interpolation_func = interp1d(current_time_axis, rf_array, kind="linear")

        # Save of rf array:
        rf_array.copy()

        # Interpolate data
        rf_array = interpolation_func(desired_time_axis)
        print(
            f"npoints before := {len(current_time_axis)}, npoints after={len(desired_time_axis)}"
        )

    else:
        desired_npoints = npoints

    def normalize_rf(input_rf_array):
        input_rf_array = input_rf_array / np.max(input_rf_array)
        return input_rf_array

    def format_number_E_minus_01(num):
        # Format the number to move the decimal point one digit to the left
        adjusted_number = num * 10
        return f"{adjusted_number:.6f}E-01"

    # plot before and after interpolation:
    if db:
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(np.abs(rf_array))
        ax[1].plot(np.real(rf_array))
        ax[1].plot(np.imag(rf_array))
        ax[2].plot(np.angle(rf_array) / np.pi * 180.0)

    normalized_rf_array = normalize_rf(input_rf_array=rf_array)
    phase_rf_array_deg = np.angle(normalized_rf_array) / np.pi * 180.0
    formatted_numbers = [
        f"{format_number_E_minus_01(np.abs(num))}, {format_number_E_minus_01(angle)}"
        for num, angle in zip(normalized_rf_array, phase_rf_array_deg)
    ]
    amps = [float(item.split(",")[0].strip()) for item in formatted_numbers]
    phases = [float(item.split(",")[1].strip()) for item in formatted_numbers]

    # Important to not put an offset here and keep this part
    header = f"""##TITLE=/opt/PV7.0.0/prog/curdir/nmrsu/ParaVision/exp/lists/wave/{savename}
##JCAMP-DX= 5.00 Bruker JCAMP library
##DATA TYPE= Shape Data
##ORIGIN= Bruker BioSpin GmbH
##OWNER= <luca>
##DATE= 2025/01/10
##TIME= 13:20:52
##MINX= {format_number_E_minus_01(np.min(amps))}
##MAXX= {format_number_E_minus_01(np.max(amps))}
##MINY= {format_number_E_minus_01(np.min(phases))}
##MAXY= {format_number_E_minus_01(np.max(phases))}
##$SHAPE_EXMODE= Excitation
##$SHAPE_TOTROT= 90.000000e+00
##$SHAPE_BWFAC= 6.20000e+00
##$SHAPE_INTEGFAC= 1.731923e-01
##$SHAPE_REPHFAC=50
##$SHAPE_TYPE=conventional
##$SHAPE_MODE= {shape_mode}
##NPOINTS= {desired_npoints}
##XYPOINTS= (XY..XY)
"""
    if save_as in ["exc", "ref"]:
        footer = "##END="
        filename = f"{savename}.{save_as}"
        savename = os.path.join(savepath, filename)
        savename = os.path.join(savepath, filename)
        # Write to file:
        with open(savename, "w") as file:
            file.write(header)
            for number in formatted_numbers:
                file.write(number + "\n")
            file.write(footer)
        file.close()
    else:
        footer = "##END="
        filename = f"{savename}.{desired_npoints}"
        savename = os.path.join(savepath, filename)
        # Write to file:
        with open(savename, "w") as file:
            file.write(header)
            for number in formatted_numbers:
                file.write(number + "\n")
            file.write(footer)
        file.close()


def generate_csi_sampling_pattern(size_x, size_y, pattern_type="spiral", **kwargs):
    """
    Generate a sampling pattern for chemical shift imaging (CSI) experiments similar to PV sequence lucaCSI4.

    Parameters
    ----------
    size_x : int
        The width of the output array.
    size_y : int
        The height of the output array.
    pattern_type : str
        The type of sampling pattern to generate. Can be either 'spiral' or 'centric'.
    **kwargs
        Additional keyword arguments to pass to the sampling pattern generation function (spatial_resolution_mm in case of pattern_type = "centric").

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (size_y, size_x) containing the sampling pattern.

    Examples
    --------
    >>> import hypermri.utils.utils_sequences as utseq
    >>> spiral_pattern = utseq.generate_csi_sampling_pattern(size_x=16, size_y=16, pattern_type="spiral")
    >>> centric_pattern = utseq.generate_csi_sampling_pattern(size_x=16, size_y=16, pattern_type="centric")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax=ax.flatten()
    >>> ax[0].imshow(spiral_pattern)
    >>> ax[1].imshow(centric_pattern)
    """

    def generate_centric_sampling(size_x, size_y, spatial_resolution_mm):
        """
        Generate an equidistant sampling pattern based on distance to k-space center.

        This function creates a 2D numpy array representing a sampling pattern
        where points are ordered based on their distance to the k-space center,
        taking into account the spatial resolution.

        Parameters
        ----------
        size_x : int
            The width of the output array.
        size_y : int
            The height of the output array.
        spatial_resolution_mm : tuple
            A tuple of two floats representing the spatial resolution in mm
            for x and y dimensions.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (size_y, size_x) containing the sampling
            pattern. The array is filled with integers from 0 to
            (size_x * size_y - 1), representing the order of sampling.

        Notes
        -----
        This function implements a sampling pattern similar to a spiral but
        orders points based on their distance to the k-space center, calculated
        in fractions of k_bwx and k_bwy.

        Examples
        --------
        >>> pattern = generate_equidistant_sampling(64, 64, (1.0, 1.0))
        >>> plt.imshow(pattern, cmap='viridis')
        >>> plt.colorbar(label='Sampling Order')
        >>> plt.title('Equidistant Sampling Pattern')
        >>> plt.show()
        """
        if spatial_resolution_mm is None:
            spatial_resolution_mm = np.zeros(2)
            spatial_resolution_mm[0] = 1.0
            spatial_resolution_mm[1] = 1.0

        # Calculate k-space "bandwidth"
        k_bwx = 1 / spatial_resolution_mm[0]
        k_bwy = 1 / spatial_resolution_mm[1]

        # Create k-space coordinate matrices
        kx = np.linspace(-1 + 1 / size_x, 1 - 1 / size_x, size_x)
        ky = np.linspace(-1 + 1 / size_y, 1 - 1 / size_y, size_y)
        kx_mat, ky_mat = np.meshgrid(kx, ky)

        # Calculate actual k-space values
        kgrad_matx = kx_mat * k_bwx / 2
        kgrad_maty = ky_mat * k_bwy / 2

        # Calculate distance to k-space center
        k_mat_dist = np.sqrt(kgrad_matx**2 + kgrad_maty**2)

        # Flatten the arrays
        k_mat_dist_1d = k_mat_dist.flatten()
        kx_1d = kx_mat.flatten()
        ky_1d = ky_mat.flatten()
        kgrad_matx_1d = kgrad_matx.flatten()
        kgrad_maty_1d = kgrad_maty.flatten()

        # Sort based on distance to k-space center
        sorted_indices = np.argsort(k_mat_dist_1d)
        k_mat_dist_1d = k_mat_dist_1d[sorted_indices]
        kx_1d = kx_1d[sorted_indices]
        ky_1d = ky_1d[sorted_indices]
        kgrad_matx_1d = kgrad_matx_1d[sorted_indices]
        kgrad_maty_1d = kgrad_maty_1d[sorted_indices]

        # Create the sampling pattern
        pattern = np.zeros((size_y, size_x), dtype=int)
        for i, (x, y) in enumerate(zip(kx_1d, ky_1d)):
            x_idx = np.argmin(np.abs(kx - x))
            y_idx = np.argmin(np.abs(ky - y))
            pattern[y_idx, x_idx] = i

        return pattern

    def generate_spiral_sampling(size_x, size_y):
        """
        Generate a center-out spiral sampling pattern.

        This function creates a 2D numpy array representing a center-out spiral
        sampling pattern. The spiral starts from the center of the array and
        moves outward. It handles square arrays.

        Parameters
        ----------
        size_x : int
            The width of the output array.
        size_y : int
            The height of the output array.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (size_y, size_x) containing the spiral
            sampling pattern. The array is filled with integers from 0 to
            (size_x * size_y - 1), representing the order of sampling.

        Notes
        -----
        This function adapts the logic from a provided C function to create
        a spiral sampling pattern. It handles square arrays.

        Examples
        --------
        >>> pattern = generate_spiral_sampling_adapted(5, 5)
        >>> print(pattern)
        [[20 19 18 17 16]
        [21  6  5  4 15]
        [22  7  0  3 14]
        [23  8  1  2 13]
        [24  9 10 11 12]]
        """
        if size_x != size_y:
            raise ValueError(
                f"Size must be rectangular but is size_x={size_x} x size_y={size_y}"
            )
        total_points = size_x * size_y

        # Create arrays to store the spiral order
        order_spiral_x = np.zeros(total_points, dtype=int)
        order_spiral_y = np.zeros(total_points, dtype=int)

        if size_x == size_y:
            # Square matrix case
            arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
            arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
            arr_y = np.flipud(arr_y)

            offset = 0
            cols = size_y - 1
            rows = size_x - 1
            counter = 0

            while offset < (size_x - 1):
                # Top row
                for col in range(offset, cols + 1):
                    order_spiral_x[counter] = arr_x[offset, col]
                    order_spiral_y[counter] = arr_y[offset, col]
                    counter += 1

                # Right column
                for row in range(offset + 1, rows + 1):
                    order_spiral_x[counter] = arr_x[row, cols]
                    order_spiral_y[counter] = arr_y[row, cols]
                    counter += 1

                # Bottom row
                for col in range(cols - 1, offset - 1, -1):
                    order_spiral_x[counter] = arr_x[rows, col]
                    order_spiral_y[counter] = arr_y[rows, col]
                    counter += 1

                # Left column
                for row in range(rows - 1, offset, -1):
                    order_spiral_x[counter] = arr_x[row, offset]
                    order_spiral_y[counter] = arr_y[row, offset]
                    counter += 1

                offset += 1
                rows -= 1
                cols -= 1

        else:
            # Rectangular matrix case
            y_bigger_than_x = size_y > size_x
            leftover = abs(size_y - size_x)
            np.ceil(leftover / 2)
            half_leftover_floor = np.floor(leftover / 2)

            if y_bigger_than_x:
                arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
                arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
                arr_y = np.flipud(arr_y)

                for i in range(size_x):
                    for j in range(half_leftover_floor, size_x + half_leftover_floor):
                        arr_x[i, j] = i
                        arr_y[i, j] = j
            else:
                arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
                arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
                arr_y = np.flipud(arr_y)

                for i in range(half_leftover_floor, size_y + half_leftover_floor):
                    for j in range(size_y):
                        arr_x[i, j] = i
                        arr_y[i, j] = j

            # Now implement the spiral ordering for the rectangular case
            # (This part would need to be implemented similar to the square case,
            # but accounting for the rectangular shape)

        # Create the final pattern array
        pattern = np.full((size_y, size_x), -1, dtype=int)
        for i in range(total_points):
            x, y = order_spiral_x[i], order_spiral_y[i]
            if pattern[y, x] == -1:
                pattern[y, x] = i

        # Fill any remaining -1 values
        remaining = np.where(pattern == -1)
        pattern[remaining] = range(np.sum(pattern >= 0), total_points)
        pattern = np.abs(pattern - np.max(pattern))

        return pattern

    if pattern_type == "spiral":
        return generate_spiral_sampling(size_x, size_y)
    elif pattern_type == "centric":
        if kwargs is None:
            kwargs = {}
            kwargs["spatial_resolution_mm"] = (1.0, 1.0)
        elif "spatial_resolution_mm" not in kwargs:
            kwargs["spatial_resolution_mm"] = (1.0, 1.0)
        return generate_centric_sampling(size_x, size_y, **kwargs)


def create_sample_image(
    nx: int = 16,
    ny: int = 16,
    shape: str = "square",
    size_object_ratio: float = 0.5,
    dtype: str = "complex",
):
    """
    Create a sample image with specified shape and size.

    Parameters
    ----------
    nx, ny : int
        Dimensions of the image.
    shape : str
        Shape of the sample ('square' or 'circle' or '2 squares').
    size_object_ratio : float
        Size of the sample as a fraction of the image size (0 to 1).
    dtype : str
        Data type of the image array.

    Returns
    -------
    sample_image : ndarray
        2D array representing the sample image.
    """
    sample_image = np.zeros((nx, ny), dtype=dtype)
    center_x, center_y = nx // 2, ny // 2
    radius = int(min(nx, ny) * size_object_ratio / 2)

    if shape == "square":
        x_start, x_end = center_x - radius, center_x + radius
        y_start, y_end = center_y - radius, center_y + radius
        sample_image[x_start:x_end, y_start:y_end] = 1 + 0j

    if shape == "2 squares":
        x_start_1, x_end_1 = center_x + radius // 2, center_x + 3 * radius // 2
        y_start_1, y_end_1 = center_y + radius // 2, center_y + 3 * radius // 2
        sample_image[x_start_1:x_end_1, y_start_1:y_end_1] = 1 + 0j

        x_start_2, x_end_2 = center_x - 3 * radius // 2, center_x - radius // 2
        y_start_2, y_end_2 = center_y - 3 * radius // 2, center_y - radius // 2
        sample_image[x_start_2:x_end_2, y_start_2:y_end_2] = 1 + 0j
        print(x_start_1, x_end_1, y_start_1, y_end_1)
        print(x_start_2, x_end_2, y_start_2, y_end_2)

    elif shape == "circle":
        y, x = np.ogrid[-center_x : nx - center_x, -center_y : ny - center_y]
        mask = x * x + y * y <= radius * radius
        sample_image[mask] = 1 + 0j

    return sample_image


def interactive_psf_simulation(plot_params={}):
    """
    All-in-one function for interactive PSF simulation based on sampling pattern and hyperpolarized signal decay.

    This function creates an interactive plot that updates based on user-defined parameters.
    It displays the signal decay, sampling pattern, PSF, sample image, and convolved image.
    """
    from IPython.display import clear_output, display
    from ipywidgets import (
        Dropdown,
        FloatSlider,
        HBox,
        IntSlider,
        VBox,
        interactive,
    )

    clear_output(wait=True)

    figsize = plot_params.get("figsize", (10, 4))

    def update_plot(
        nx=16,
        ny=16,
        pattern_type="spiral",
        M0_au=1e5,
        TR_s=0.05,
        T1_s=30,
        flipangle_deg=5,
        sample_shape="square",
        sample_size=0.5,
        noise_level=0,
    ):
        def generate_psf(sampling_pattern, Mz, flipangle_deg):
            """Generate the PSF based on the sampling pattern and hyperpolarized signal decay."""
            psf = np.zeros_like(sampling_pattern, dtype=float)
            for i in range(len(Mz)):
                psf[sampling_pattern == i] = np.sin(np.deg2rad(flipangle_deg)) * Mz[i]
                # psf[sampling_pattern == i] = Mz[i]
            return psf

        from ..physics import (
            hyperpolarized_Mz_flipangle,
            hyperpolarized_Mz_flipangle_T1,
            hyperpolarized_Mz_T1,
        )

        # Clear any previous output
        clear_output(wait=True)

        # Create the figure and axes
        fig, axs = plt.subplots(2, 3, figsize=figsize)

        # Generate sampling pattern
        sampling_pattern = generate_csi_sampling_pattern(nx, ny, pattern_type)

        # Calculate Mz
        n_excitations = nx * ny
        Mz, Mxy, sampling_time = hyperpolarized_Mz_flipangle_T1(
            M0_au=M0_au,
            n_excitations=n_excitations,
            TR_s=TR_s,
            T1_s=T1_s,
            flipangle_deg=flipangle_deg,
            plot=False,
            interactive=False,
        )
        Mz_T1, Mxy_flipangle = hyperpolarized_Mz_T1(
            M0_au=M0_au, t_s=sampling_time, T1_s=T1_s
        )
        Mz_flipangle, Mxy_flipangle = hyperpolarized_Mz_flipangle(
            M0_au=M0_au, n_excitations=n_excitations, flipangle_deg=flipangle_deg
        )

        # Generate PSF
        psf = generate_psf(sampling_pattern, Mz, flipangle_deg)
        psf = psf / psf.shape[0] / psf.shape[1]

        # Create sample image
        sample_image = create_sample_image(nx, ny, sample_shape, sample_size)
        noise = np.random.normal(
            0, noise_level, sample_image.shape
        ) + 1j * np.random.normal(0, noise_level, sample_image.shape)
        sample_image += noise
        sample_image = sample_image / np.sum(np.abs(sample_image))

        # Calculate PSF FT and convolved image
        psf_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sample_image)))
        convolved_image = np.abs(
            np.fft.ifftshift(
                np.fft.ifft2(
                    np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sample_image)))
                    * np.rot90(psf)
                )
            )
        )
        convolved_image = convolved_image / np.sum(np.abs(convolved_image))

        # Set figure title
        fig.suptitle(
            f"PSF Simulation(nx={nx}, ny={ny}, pattern={pattern_type}, shape={sample_shape}, size={sample_size: .2f})"
        )

        # Plot signal decay
        axs[0, 0].clear()
        axs[0, 0].plot(sampling_time, Mz, label="Mz (Total)")
        axs[0, 0].plot(
            sampling_time, Mz_flipangle, "--", alpha=0.5, label="Mz (Flip Angle)"
        )
        axs[0, 0].plot(sampling_time, Mz_T1, "--", alpha=0.5, label="Mz (T1)")
        axs[0, 0].set_title("Signal Decay")
        axs[0, 0].set_xlabel("Excitation")
        axs[0, 0].set_ylabel("Mz")
        axs[0, 0].axhline(0, color="r", linestyle="--")
        axs[0, 0].legend()
        ax2 = axs[0, 0].twiny()  # Create a twin axis for pulse numbers

        # Plot sampling pattern
        axs[0, 1].clear()
        im1 = axs[0, 1].imshow(sampling_pattern, cmap="viridis")
        axs[0, 1].set_title("Sampling Pattern")
        fig.colorbar(im1, ax=axs[0, 1])

        # Plot PSF FT magnitude
        axs[0, 2].clear()
        im2 = axs[0, 2].imshow(np.log(np.abs(psf_ft)), cmap="viridis")
        axs[0, 2].set_title("PSF FT Magnitude (log scale)")
        fig.colorbar(im2, ax=axs[0, 2])

        # Plot sample image
        axs[1, 0].clear()
        im5 = axs[1, 0].imshow(np.abs(sample_image), cmap="gray")
        axs[1, 0].set_title("Sample Image")
        fig.colorbar(im5, ax=axs[1, 0])

        # Plot convolved image
        axs[1, 1].clear()
        im3 = axs[1, 1].imshow(convolved_image, cmap="viridis")
        axs[1, 1].set_title("Convolved Image")
        fig.colorbar(im3, ax=axs[1, 1])

        # Plot difference between convolved and sample image
        axs[1, 2].clear()
        im4 = axs[1, 2].imshow(
            100
            * (np.abs(convolved_image) - np.abs(sample_image))
            / np.abs(convolved_image),
            cmap="seismic",
        )
        axs[1, 2].set_title("convolved - sample")
        fig.colorbar(im4, ax=axs[1, 2])
        # Set up the secondary x-axis for pulse numbers
        ax2.set_xlim(axs[0, 0].get_xlim())
        ax2.set_xticks(np.arange(0, n_excitations, max(1, n_excitations // 10)) * TR_s)
        ax2.set_xticklabels(np.arange(0, n_excitations, max(1, n_excitations // 10)))

        # Adjust layout and display the figure
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    # Create widgets for the first column
    column1_widgets = [
        IntSlider(min=2, max=64, step=1, value=16, description="nx:"),
        IntSlider(min=2, max=64, step=1, value=16, description="ny:"),
        Dropdown(
            options=["spiral", "centric"], value="centric", description="Pattern:"
        ),
        FloatSlider(min=1e4, max=1e6, step=1e4, value=1e5, description="M0:"),
        FloatSlider(min=0.01, max=0.50, step=0.01, value=0.05, description="TR (s):"),
    ]

    # Create widgets for the second column
    column2_widgets = [
        FloatSlider(min=1, max=100, step=1, value=30, description="T1 (s):"),
        FloatSlider(min=1, max=90, step=1, value=5, description="Flip Angle (°):"),
        Dropdown(
            options=["square", "circle", "2 squares"],
            value="square",
            description="Shape:",
        ),
        FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5, description="Size:"),
        FloatSlider(min=0, max=1, step=0.1, value=0, description="Noise:"),
    ]

    # Create the two-column layout
    column1 = VBox(column1_widgets)
    column2 = VBox(column2_widgets)
    layout = HBox([column1, column2])

    # Create an interactive widget using your existing widgets
    interactive_plot = interactive(
        update_plot,
        nx=column1_widgets[0],
        ny=column1_widgets[1],
        pattern_type=column1_widgets[2],
        M0_au=column1_widgets[3],
        TR_s=column1_widgets[4],
        T1_s=column2_widgets[0],
        flipangle_deg=column2_widgets[1],
        sample_shape=column2_widgets[2],
        sample_size=column2_widgets[3],
        noise_level=column2_widgets[4],
    )

    # Create a VBox that contains your layout and the output of the
    # interactive plot
    final_layout = VBox([layout, interactive_plot.children[-1]])

    # Display the final layout
    display(final_layout)
