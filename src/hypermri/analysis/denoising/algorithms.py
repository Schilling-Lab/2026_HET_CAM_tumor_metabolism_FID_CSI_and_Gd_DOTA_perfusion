"""
Denoising algorithms used across HyperMRI analysis workflows.

References
----------
.. [Veraart2016a] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
       Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
       theory. Neuroimage 142:394-406.
       doi: 10.1016/j.neuroimage.2016.08.016
.. [Veraart2016b] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
       mapping using random matrix theory. Magnetic Resonance in Medicine.
       doi: 10.1002/mrm.26059.
.. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
       Diffusion Weighted Image Denoising Using Overcomplete Local
       PCA. PLoS ONE 8(9): e73021.
       https://doi.org/10.1371/journal.pone.0073021
.. [Henriques2021] Henriques RN, Ianuş A, Novello L, Jovicich J, Jespersen SN, Shemesh N.
       Efficient PCA denoising of diffusion MRI data corrupted by spatially correlated
       noise.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dipy.denoise.localpca import _pca_classifier
from scipy.linalg import eigh, svd

from ...utils.system.logging import init_default_logger

logger = init_default_logger(__name__)
COORDINATE_PRECISION = np.float16  # Or np.float16, np.float64, etc.


# All code between the two lines below is copied from:
# References:
#   Henriques, R. N., et al. (2023). "Efficient PCA denoising of spatially
#   correlated redundant MRI data." Imaging Neuroscience 1: 1-26.
#   DOI: 10.1162/imag_a_00049
# Code available at:
#   git@github.com:RafaelNH/PCAdenoising.git

# --------------------------------------------------------------------------------
# ------------------------------PCAdenoising--------------------------------------


def mp_distribution(x, var, y):
    """
    Sample the Marchenko–Pastur probability distribution.

    Parameters
    ----------
    x : np.ndarray
        Values of random variable to sample the probability distribution
    var : float
        Variance of the random variable
    y : float
        Parameter associated to the matrix X that produces the distributions.
        This X is a M x N random matrix which columns entries are identical
        distributed random variables with mean 0 and given variance, y is given
        by N/M.

    """
    xpos = var * (1 + np.sqrt(y)) ** 2
    xneg = var * (1 - np.sqrt(y)) ** 2

    p = np.zeros(x.shape)
    xdis = np.logical_and(x < xpos, x > xneg)
    p[xdis] = np.sqrt((xpos - x[xdis]) * (x[xdis] - xneg)) / (
        2 * np.pi * var * y * x[xdis]
    )
    if np.sum(p) == 0:
        return p
    else:
        return p / np.sum(p)


def _gpca_classifier(L, varest):
    """
    Classifies which PCA eigenvalues are related to noise and estimates the noise variance.

    Parameters
    ----------
    L : array (n,)
        Array containing the PCA eigenvalues in ascending order.
    nvoxels : int
        Number of voxels used to compute L
    varest : float
        Variance prior estimate (if you don't have this you cannot use this method)

    Returns
    -------
    var : float
        Estimation of the noise variance
    ncomps : int
        Number of eigenvalues related to noise

    Notes
    -----
    This is based on the TPCA algorithm described in [1]_.

    References
    ----------
    .. [1] Henriques RN, Ianuş A, Novello L, Jovicich J, Jespersen SN, Shemesh N.
    Efficient PCA denoising of diffusion MRI data corrupted by spatially correlated
    noise.
    """
    var = np.mean(L)
    c = L.size - 1
    while var > varest:
        var = np.mean(L[:c])
        c = c - 1
    ncomps = c + 1
    return var, ncomps


def _pca_classifier_slow(L, nvoxels):
    """
    Classifies which PCA eigenvalues are related to noise and estimates the noise variance.

    Parameters
    ----------
    L : array (n,)
        Array containing the PCA eigenvalues in ascending order.
    nvoxels : int
        Number of voxels used to compute L

    Returns
    -------
    var : float
        Estimation of the noise variance
    ncomps : int
        Number of eigenvalues related to noise

    Notes
    -----
    This is based on the algorithm described in [1]_.

    References
    ----------
    .. [1] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
           mapping using random matrix theory. Magnetic Resonance in Medicine.
           doi: 10.1002/mrm.26059.

    """
    # To not make any assumptions about the expectation of the
    # number of signal compartments, let's initialize the code
    # assessing all components as being noise (this is the case of
    # background voxels)
    n = L.size
    c = n

    # Iterations will stop if the next number of signal components
    # was already assessed in previous iterations. This includes the
    # case c[i] = c[i+1], and also avoids infinite loops, e.g. cases
    # that c[i] = a leads to c[i+1] = b which then leads to c[i+2] = a
    # (i.e. creating a infinite loop of assessing a and b number of
    # components)
    # Given the above, let create a vector to tarked which number of
    # components were assessed.
    c_considered = np.zeros(n)

    # start loop
    # (note c-1 is used instead of c, because c is the number of components,
    # which indexes goes from 0 to c-1)
    while c_considered[c - 1] == 0:
        # Compute optimal number of bins for the selected noise components

        nbin = np.arange(5, n)  # maximum number of bins is set to number
        # of eigenvalues, i.e. case of a number of bins larger than
        # a number of egienvalues is not considered.
        # 5 bins are considered as the minumun reasonable
        # number of bins.

        # procedure implemented according to Veraart et al., 2016 [1]
        rang = L[c - 1] - L[0]
        h = rang / nbin
        nb_vall = np.zeros(nbin.size)
        for ii, ni in enumerate(nbin):
            ro, bins = np.histogram(L[:c], bins=ni, range=(L[0], L[c - 1]))
            ro_mean = np.mean(ro)
            ro_var = np.var(ro)
            nb_vall[ii] = (2 * ro_mean - ro_var) / (h[ii] ** 2)
        nb = nbin[np.argmin(nb_vall)]

        # compute final histogram to be fitted
        ro, bins = np.histogram(L[:c], bins=nb, range=(L[0], L[c - 1]))

        # initial variance estimate
        var_in = L[0] / ((1 - np.sqrt(c / nvoxels)) ** 2)

        # iteratively refine variance estimate that produces
        # a theoritical distribution that fits the measured histogram
        # (for that let consider a range between half of the initial std
        # estimate and four times larger that the initial std estimate)
        sigma_grid = np.linspace(np.sqrt(var_in) / 2, 4 * np.sqrt(var_in), num=500)
        sigma_eval = np.zeros(sigma_grid.shape)
        for std_i, std_val in enumerate(sigma_grid):
            dist = c * mp_distribution(
                bins[:-1] + rang / (2 * nb), std_val**2, c / nvoxels
            )
            notzero = dist > 0
            if np.sum(dist) == 0:
                sigma_eval[std_i] = np.inf
            else:
                sigma_eval[std_i] = np.sum(
                    (ro[notzero] - dist[notzero]) ** 2 / dist[notzero]
                )

        indmin = np.flatnonzero(sigma_eval == sigma_eval.min())
        var_out = sigma_grid[indmin[-1]] ** 2

        # use refined variance estimate for component classification
        tau_factor = 1 + np.sqrt(c / nvoxels)
        tau = tau_factor**2 * var_out

        # update number of noise components
        c_considered[c - 1] = 1
        c = np.sum(L < tau)

    ncomps = c
    var = np.mean(L[:c])
    return var, ncomps


def genpca(
    arr: np.ndarray = None,
    patch_radius: int = 2,
    mppca_slow: bool = False,
    sigma=None,
    mask=None,
    pca_method: str = "eig",
    tau_factor=None,
    return_sigma=False,
    out_dtype=None,
    db_mode=False,
):
    r"""
    General function to perform PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : np.ndarray
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 2 (denoise in blocks of 5x5x5 voxels).
    sigma : float or 3D array (optional)
        Standard deviation of the noise estimated from the data. If no sigma
        is given, this will be estimated based on random matrix theory
        [Veraart2016a]_, [Veraart2016b]_
    mppca_slow : boolean (optional)
        If true and noise sigma estimate is provided it will run the slow
        version of MPPCA denoising
    mask : 3D boolean array (optional)
        A mask with voxels that are true inside the brain and false outside of
        it. The function denoises within the true part and returns zeros
        outside of those voxels.
    pca_method : 'eig' or 'svd' (optional)
        Use either eigenvalue decomposition (eig) or singular value
        decomposition (svd) for principal component analysis. The default
        method is 'eig' which is faster. However, occasionally 'svd' might be
        more accurate.
    tau_factor : float (optional)
        If this variable is not None, thresholding of PCA eigenvalues is done
        by nulling out eigenvalues that are smaller than:

                .. math ::

                        \tau = (\tau_{factor} \sigma)^2

        \tau_{factor} can be set to a predefined values (e.g. \tau_{factor} =
        2.3 [Manjon2013]_). In case that the variable is not defined (i.e. in case that \tau_{factor} is set to None),
        tau_factor is calculated according MP-PCA distribution in case that noise variance
        is not give. If noise variance is given,  GPCA algorithm is used.
        Default: None.
    return_sigma : bool (optional)
        If true, the Standard deviation of the noise will be returned.
        Default: False.
    out_dtype : str or dtype (optional)
        The dtype for the output array. Default: output has the same dtype as
        the input.

    Returns
    -------
    denoised_arr : np.ndarray
        This is the denoised 4D-array of the same size as that of the input data,
        clipped to non-negative values

    """
    if mask is None:
        # If mask is not specified, use the whole volume
        mask = np.ones_like(arr, dtype=bool)[..., 0]

    if out_dtype is None:
        out_dtype = arr.dtype

    # We retain float64 precision, iff the input is in this precision:
    if np.iscomplexobj(arr):
        calc_dtype = np.complex128 if arr.dtype == np.complex128 else np.complex64
    else:
        calc_dtype = np.float64 if arr.dtype == np.float64 else np.float32

    float_calc_dtype = (
        np.float64 if calc_dtype in [np.complex128, np.float64] else np.float32
    )

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.", arr.shape)

    if pca_method.lower() == "svd":
        is_svd = True
    elif pca_method.lower() == "eig":
        is_svd = False
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    if is_svd:
        svd_args = {"full_matrices": False}
    else:
        svd_args = {}

    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius).astype(int)

    # Adjust patch radius for singleton dimensions
    for i in range(3):
        if arr.shape[i] == 1:
            patch_radius[i] = 0

    patch_size = 2 * patch_radius + 1
    m = np.prod(patch_size)

    if np.prod(patch_size) < arr.shape[-1]:
        e_s = "You asked for PCA denoising with a "
        e_s += "patch_radius of {0} ".format(patch_radius)
        e_s += "with total patch size of {0}".format(np.prod(patch_size))
        e_s += "for data with {0} directions. ".format(arr.shape[-1])
        e_s += "This would result in an ill-conditioned PCA matrix. "
        e_s += "Please increase the patch_radius."
        raise ValueError(e_s)

    if isinstance(sigma, np.ndarray):
        if not sigma.shape == arr.shape[:-1]:
            e_s = "You provided a sigma array with a shape"
            e_s += "{0} for data with".format(sigma.shape)
            e_s += "shape {0}. Please provide a sigma array".format(arr.shape)
            e_s += " that matches the spatial dimensions of the data."
            raise ValueError(e_s)
    elif isinstance(sigma, (int, float)):
        sigma = sigma * np.ones(arr.shape[:-1])

    dim = arr.shape[-1]

    theta = np.zeros(arr.shape, dtype=float_calc_dtype)
    thetax = np.zeros(arr.shape, dtype=calc_dtype)
    ncall = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)

    thetavar = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)
    if return_sigma is True:
        out_sigma = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)

    print(f"tau_factor: {tau_factor}")
    # Get center voxel indices for debug printing
    center_i = (arr.shape[0] - 2 * patch_radius[0]) // 2 + patch_radius[0]
    center_j = (arr.shape[1] - 2 * patch_radius[1]) // 2 + patch_radius[1]
    center_k = (arr.shape[2] - 2 * patch_radius[2]) // 2 + patch_radius[2]

    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius[2], arr.shape[2] - patch_radius[2]):
        for j in range(patch_radius[1], arr.shape[1] - patch_radius[1]):
            for i in range(patch_radius[0], arr.shape[0] - patch_radius[0]):
                if not mask[i, j, k]:
                    continue
                ix1 = i - patch_radius[0]
                ix2 = i + patch_radius[0] + 1
                jx1 = j - patch_radius[1]
                jx2 = j + patch_radius[1] + 1
                kx1 = k - patch_radius[2]
                kx2 = k + patch_radius[2] + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(np.prod(patch_size), dim)
                M = np.mean(X, axis=0)
                X = X - M

                if is_svd:
                    # Scale data to avoid overflow during squared singular value calculation
                    X_scaled = X / np.sqrt(X.shape[0])
                    U, S, Vt = svd(X_scaled, **svd_args)[:3]
                    d = S[::-1] ** 2
                    W = Vt[::-1].T
                else:
                    # Scale data to avoid overflow during matrix multiplication
                    X_scaled = X / np.sqrt(X.shape[0])
                    C = np.transpose(X_scaled).dot(X_scaled)
                    [d, W] = eigh(C)

                if sigma is None:
                    if mppca_slow:
                        this_var, ncomps = _pca_classifier_slow(d, m)
                    else:
                        this_var, ncomps = _pca_classifier(d, m)
                else:
                    this_var = sigma[i, j, k] ** 2
                    if tau_factor is None:
                        this_var, ncomps = _gpca_classifier(d, this_var)
                    else:
                        tau = tau_factor**2 * this_var
                        ncomps = np.sum(d < tau)
                        if return_sigma is True:
                            this_var = np.mean(d[:ncomps])
                if db_mode and i == center_i and j == center_j and k == center_k:
                    print(f"Voxel ({i}, {j}, {k})")
                    print(f"this_var: {this_var:.4f}")

                    if tau_factor is None:
                        print(f"    Method: GPCA (variance comparison)")
                    else:
                        print(f"    Method: TPCA (pre-defined tau_factor)")
                        print(f"    Threshold (tau): {tau:.4f}")

                # if db_mode:
                # print(f"keeping: {ncomps} components at voxel ({i}, {j}, {k})")
                W[:, :ncomps] = 0
                Xest = X.dot(W).dot(W.T) + M
                Xest = Xest.reshape(patch_size[0], patch_size[1], patch_size[2], dim)
                this_theta = 1.0 / (1.0 + dim - ncomps)
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta
                ncall[ix1:ix2, jx1:jx2, kx1:kx2] += (dim - ncomps) * this_theta
                thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta

                if return_sigma is True:
                    out_sigma[ix1:ix2, jx1:jx2, kx1:kx2] += (
                        np.sqrt(this_var) * this_theta
                    )

    denoised_arr = thetax / theta
    denoised_arr[mask == 0] = 0
    ncall = ncall / thetavar

    if return_sigma is True:
        out_sigma = out_sigma / thetavar
        out_sigma[mask == 0] = 0
        return denoised_arr.astype(out_dtype), out_sigma, ncall
    else:
        return denoised_arr.astype(out_dtype)


def median_filt(sigma, patch_radius=2):
    r"""
    Inspect the actual sigma used by denosing algorithms if sigma is selected from the median of sliding windows.

    Parameters
    ----------
    sigma : float or 3D array (optional)
        Standard deviation of the noise estimated from the data. If no sigma
        is given, this will be estimated based on random matrix theory
        [Veraart2016a]_, [Veraart2016b]_
    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 2 (denoise in blocks of 5x5x5 voxels)

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values
    """

    sigma_final = np.zeros(sigma.shape)
    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius[2], sigma.shape[2] - patch_radius[2]):
        for j in range(patch_radius[1], sigma.shape[1] - patch_radius[1]):
            for i in range(patch_radius[0], sigma.shape[0] - patch_radius[0]):
                ix1 = i - patch_radius[0]
                ix2 = i + patch_radius[0] + 1
                jx1 = j - patch_radius[1]
                jx2 = j + patch_radius[1] + 1
                kx1 = k - patch_radius[2]
                kx2 = k + patch_radius[2] + 1

                this_var = np.median(sigma[ix1:ix2, jx1:jx2, kx1:kx2])
                sigma_final[i, j, k] = this_var

    return sigma_final


# ------------------------------PCAdenoising for xarrays------------------
# --------------------------------------------------------------------------------


def denoise_pca(
    input_xarray: xr.DataArray,
    components: Union[int, list, slice] = None,
    pca_method: str = "eig",
    plot: bool = False,
    db_mode: bool = False,
    spatial_dims_present: list = None,
    redundant_dims_present: list = None,
    noise_threshold: float = None,
    noise_estimation_method: str = "auto",
    treat_as_redundant: list = None,
) -> xr.DataArray:
    """
    Perform PCA-based denoising on a complex-valued xarray.DataArray.

    This function automatically identifies spatial and redundant dimensions,
    reshapes the data into a 2D matrix (pixels x repetitions), applies
    PCA denoising, and then reconstructs the data back into its original
    xarray structure, preserving all original coordinates and dimension orders.

    Parameters
    ----------
    input_xarray: xr.DataArray
        The input data with spatial and redundant dimensions.
    components: int, list, or slice, optional
        The principal components to keep for reconstruction.
        - If int: Keep the top N components (0 to N-1).
        - If list: Keep specific component indices (e.g., [0, 2, 4]).
        - If slice: Keep a range of components (e.g., slice(3, 5)).
        - If None: Automatically determine the number of components using TPCA threshold.
        Defaults to None.
    pca_method: str, optional
        The method for PCA ('eig' or 'svd'). eig' is faster, 'svd' may be more stable. Defaults to "eig".
    plot: bool, optional
        If True, displays the scree plot of eigenvalues. Defaults to False.
    db_mode: bool, optional
        If True, prints debug information. Defaults to False.
    spatial_dims_present: list, optional
        List of spatial dimensions to consider. If None, defaults to ["read_idx", "phase_idx", "slice_idx"].
        Defaults to None.
    redundant_dims_present: list, optional
        List of redundant dimensions for PCA. If None, it is automatically determined from the input data.
        Defaults to None.
    noise_threshold: float, optional
        Manual noise variance threshold for component selection. If None, noise is estimated automatically.
        Defaults to None.
    noise_estimation_method: str, optional
        Method for estimating noise variance. Defaults to 'auto'.
        Options:
        - 'auto' (corner-based with fallback),
        - 'corner' (corners only), 'last_reps' (last 10% of repetitions),
        - 'manual' (use provided noise_threshold).

    treat_as_redundant: list, optional
        List of dimension names to force-treat as redundant dimensions rather than spatial. Useful for
        spectral data where 'slice_idx' represents multiple acquisitions.
        Defaults to None.

    Returns
    -------
    pca_denoised array: xr.DataArray
        A new xarray.DataArray containing the denoised data.

    """
    # --- 1. Identify Spatial and Redundant Dimensions ---
    if db_mode:
        print("--- Step 1: Identifying Dimensions ---")

    SPATIAL_DIMS = ["read_idx", "phase_idx", "slice_idx"]

    # Handle treat_as_redundant parameter for spectral data
    if treat_as_redundant is None:
        treat_as_redundant = []

    if spatial_dims_present is None:
        # Exclude dimensions that should be treated as redundant
        spatial_dims_present = [
            dim
            for dim in SPATIAL_DIMS
            if dim in input_xarray.dims and dim not in treat_as_redundant
        ]

    if redundant_dims_present is None:
        # Include all non-spatial dims plus those forced to be redundant
        redundant_dims_present = [
            dim
            for dim in input_xarray.dims
            if dim not in spatial_dims_present or dim in treat_as_redundant
        ]

    # Validate dimensions
    if not spatial_dims_present:
        # For spectral-only data, we need at least acquisition_idx as "spatial" dimension
        if "acquisition_idx" in input_xarray.dims:
            spatial_dims_present = ["acquisition_idx"]
            redundant_dims_present = [
                dim for dim in input_xarray.dims if dim != "acquisition_idx"
            ]
            if db_mode:
                print(
                    "Note: Using 'acquisition_idx' as spatial dimension for spectral data."
                )
        else:
            raise ValueError(
                "No spatial or pixel dimensions found. "
                "For spectral data, 'acquisition_idx' is required. "
                "Available dimensions: " + str(list(input_xarray.dims))
            )

    if not redundant_dims_present:
        raise ValueError(
            "No redundant dimensions for PCA found. "
            "PCA denoising requires multiple measurements (e.g., repetitions, slices, echoes). "
            "Available dimensions: " + str(list(input_xarray.dims))
        )

    if db_mode:
        print(f"Spatial dimensions identified: {spatial_dims_present}")
        print(f"Redundant dimensions identified: {redundant_dims_present}")
        if treat_as_redundant:
            print(f"Dimensions forced to be redundant: {treat_as_redundant}")
        print()

    # --- 2. Reshape Data for PCA ---
    if db_mode:
        print("--- Step 2: Reshaping Data for PCA ---")

    original_dims = input_xarray.dims
    transposed_xarray = input_xarray.transpose(
        *redundant_dims_present, *spatial_dims_present
    )
    n_pixels = np.prod([transposed_xarray.sizes[dim] for dim in spatial_dims_present])
    n_repetitions = np.prod(
        [transposed_xarray.sizes[dim] for dim in redundant_dims_present]
    )
    X = transposed_xarray.values.reshape(n_repetitions, n_pixels).T

    # Check for NaNs or Infs
    if not np.all(np.isfinite(X)):
        msg = "Input data contains NaNs or Infs. Replacing with 0 for PCA denoising."
        if db_mode:
            print(f"Warning: {msg}")
        logger.warning(msg)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if db_mode:
        print(f"Shape of reshaped matrix X: {X.shape}\n")

    # --- 3. Center the Data ---
    mean_of_each_repetition = X.mean(axis=0)
    X_centered = X - mean_of_each_repetition

    # --- 4. Perform Eigendecomposition ---
    if db_mode:
        print(f"--- Step 4: Performing Eigendecomposition using '{pca_method}'---")
    M, N = X_centered.shape

    if pca_method.lower() == "eig":
        # Scale data to avoid overflow during matrix multiplication
        # covariance = (1/M) * (X.T @ X) = (X/sqrt(M)).T @ (X/sqrt(M))
        X_scaled = X_centered / np.sqrt(M)
        covariance_matrix = X_scaled.conj().T @ X_scaled
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    elif pca_method.lower() == "svd":
        # Scale data to avoid overflow during squared singular value calculation
        X_scaled = X_centered / np.sqrt(M)
        U, s, Vt = svd(X_scaled, full_matrices=False)
        eigenvalues = s**2
        eigenvectors = Vt.conj().T
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    # Ensure eigenvalues is a numpy array (not xarray DataArray)
    if hasattr(eigenvalues, "values"):
        eigenvalues = eigenvalues.values

    if db_mode:
        print("Eigendecomposition complete.\n")

    # --- 5. Determine Components to Keep ---
    if db_mode:
        print("--- Step 5: Determining Components to Keep ---")

    # Indices of components to keep
    components_to_keep_indices = None

    if components is None:
        # Convert noise_threshold to scalar if it's an xarray or numpy array
        if noise_threshold is not None:
            if hasattr(noise_threshold, "values"):
                # xarray DataArray
                noise_threshold = float(noise_threshold.values)
            elif isinstance(noise_threshold, np.ndarray):
                # numpy array - take mean or first element
                noise_threshold = float(np.mean(noise_threshold))
            else:
                noise_threshold = float(noise_threshold)

        # Estimate noise variance based on method
        if noise_estimation_method == "manual":
            if noise_threshold is None:
                raise ValueError(
                    "noise_estimation_method='manual' requires noise_threshold to be provided."
                )
            noise_variance_prior = noise_threshold
            if db_mode:
                print(f"Using manual noise threshold: {noise_variance_prior:.4f}")

        elif noise_estimation_method == "corner":
            if db_mode:
                print("Estimating noise from corner pixels...")
            try:
                corner_noise = input_xarray.isel(
                    read_idx=[0, -1], phase_idx=[0, -1]
                ).values
                noise_variance_prior = np.var(corner_noise)
                if db_mode:
                    print(f"Corner-based noise variance: {noise_variance_prior:.4f}")
            except Exception as e:
                raise ValueError(
                    f"Corner-based noise estimation failed: {e}. "
                    "Try noise_estimation_method='last_reps' or provide noise_threshold."
                )

        elif noise_estimation_method == "last_reps":
            if db_mode:
                print("Estimating noise from last 10% of repetitions...")
            noise_variance_prior = np.var(X[:, -int(N * 0.1) :])
            if db_mode:
                print(f"Last-reps noise variance: {noise_variance_prior:.4f}")

        elif noise_estimation_method == "auto":
            if db_mode:
                print(
                    "Using automatic noise estimation (corner-based with fallback)..."
                )
            try:
                corner_noise = input_xarray.isel(
                    read_idx=[0, -1], phase_idx=[0, -1]
                ).values
                noise_variance_prior = np.var(corner_noise)
                if db_mode:
                    print(f"Corner-based noise variance: {noise_variance_prior:.4f}")
            except Exception:
                if db_mode:
                    print(
                        "Warning: Could not estimate noise from corners. Using last 10% of reps."
                    )
                noise_variance_prior = np.var(X[:, -int(N * 0.1) :])
                if db_mode:
                    print(f"Last-reps noise variance: {noise_variance_prior:.4f}")
        else:
            raise ValueError(
                f"Unknown noise_estimation_method: {noise_estimation_method}. "
                "Valid options: 'auto', 'corner', 'last_reps', 'manual'"
            )

        # Apply manual threshold override if provided
        if noise_threshold is not None and noise_estimation_method != "manual":
            if db_mode:
                print(
                    f"Overriding estimated noise ({noise_variance_prior:.4f}) "
                    f"with manual threshold ({noise_threshold:.4f})"
                )
            noise_variance_prior = noise_threshold

        # Calculate TPCA threshold
        gamma = n_pixels / n_repetitions
        lambda_t_tpca = (1 + np.sqrt(gamma)) ** 2 * noise_variance_prior
        n_components_to_keep = max(1, np.sum(eigenvalues > lambda_t_tpca))
        components_to_keep_indices = np.arange(n_components_to_keep)

        if db_mode:
            print(f"Noise variance estimate: {noise_variance_prior:.4f}")
            print(f"Gamma (pixels/repetitions): {gamma:.4f}")
            print(f"TPCA Threshold (lambda_t): {lambda_t_tpca:.4f}")
            print(f"Keeping top {n_components_to_keep} components.\n")

    elif isinstance(components, int):
        if db_mode:
            print(f"Keeping top {components} components (user-specified).")
        components_to_keep_indices = np.arange(components)
    elif isinstance(components, list):
        if db_mode:
            print(f"Keeping specific components: {components} (user-specified).")
        components_to_keep_indices = np.array(components)
    elif isinstance(components, slice):
        if db_mode:
            print(f"Keeping range of components: {components} (user-specified).")
        components_to_keep_indices = np.arange(n_repetitions)[components]
    else:
        raise ValueError(
            "components must be an int, list, slice, or None. "
            f"Got {type(components)}: {components}"
        )

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(eigenvalues.real, "o-")
        ax[0].set_title("Scree Plot: Eigenvalues")
        ax[0].set_xlabel("Principal Component Number")
        ax[0].set_ylabel("Eigenvalue (Variance Explained)")
        if components is None:
            ax[0].axhline(
                y=lambda_t_tpca,
                color="r",
                linestyle="--",
                label=f"TPCA Threshold ({lambda_t_tpca:.2f})",
            )

        # Highlight kept components
        for idx in components_to_keep_indices:
            if idx < len(eigenvalues):
                ax[0].plot(idx, eigenvalues[idx].real, "go")

        ax[0].legend()
        ax[1].plot(eigenvalues.real, "o-")
        ax[1].set_yscale("log")
        ax[1].set_title("Scree Plot (Log Scale)")
        ax[1].set_xlabel("Principal Component Number")
        ax[1].set_ylabel("Log(Eigenvalue)")
        if components is None:
            ax[1].axhline(y=lambda_t_tpca, color="r", linestyle="--")
        plt.tight_layout()
        plt.show()

    # --- 6. Denoise and Reconstruct Data ---
    if db_mode:
        print("--- Step 6: Denoising and Reconstructing ---")
    projected_data = X_centered @ eigenvectors
    projected_data_filtered = np.zeros_like(projected_data)

    # Keep only the selected components
    valid_indices = components_to_keep_indices[
        components_to_keep_indices < n_repetitions
    ]
    projected_data_filtered[:, valid_indices] = projected_data[:, valid_indices]

    X_denoised_centered = projected_data_filtered @ eigenvectors.conj().T
    X_denoised = X_denoised_centered + mean_of_each_repetition

    # --- 7. Reshape Data Back to Original Xarray Format ---
    if db_mode:
        print("--- Step 7: Reshaping Data to Original Format ---")

    denoised_multidim_array = X_denoised.T.reshape(transposed_xarray.shape)
    denoised_transposed_xarray = xr.DataArray(
        data=denoised_multidim_array,
        coords=transposed_xarray.coords,
        dims=transposed_xarray.dims,
    )
    input_xarray_dn = denoised_transposed_xarray.transpose(*original_dims)
    input_xarray_dn.attrs = input_xarray.attrs
    input_xarray_dn.name = input_xarray.name

    if db_mode:
        print("\n--- Denoising Complete ---")
        print(f"Shape of final denoised xarray: {input_xarray_dn.shape}\n")

    return input_xarray_dn


# ------------------------Threshold-PCAdenoising for xarrays--------------
# --------------------------------------------------------------------------------


def _calc_tau_factor(
    pr: list = [0, 0, 0],
    data_xarray: xr.DataArray = None,
    redundant_dims: list = None,
    db_mode: bool = False,
):
    """
    Calculate the tau factor for TPCA denoising.

    Parameters
    ----------
    pr: list, optional
        Patch radius in each dimension. Defaults to [4, 4, 0].
    data_xarray: xr.DataArray, optional
        The xarray.DataArray containing the data. Used to determine the number of redundant dimensions.
        Defaults to None.
    redundant_dims: list, optional
        Redundant dimensions for PCA. If None, it is determined from data_xarray.
    db_mode: bool, optional
        If True, prints debug information.

    Returns
    -------
    tau_factor: float
        The calculated tau factor.

    """
    n = 1
    for dim_name in redundant_dims:
        n *= data_xarray.sizes[dim_name]
    m = (2 * pr[0] + 1) * (2 * pr[1] + 1) * (2 * pr[2] + 1)
    tau_factor = 1 + np.sqrt(n / m)
    if db_mode:
        print(
            f"Calculated tau_factor: {tau_factor} for patch_radius: {pr} and n: {n}, m: {m}"
        )
    return tau_factor


def tpca_denoise(
    input_xarray: xr.DataArray,
    patch_radius=4,
    mppca_slow=False,
    sigma=None,
    pca_method="eig",
    tau_factor=None,
    out_dtype=None,
    db_mode: bool = False,
    spatial_dims_present: list = None,
    redundant_dims_present: list = None,
    apply_median_filter: bool = True,
    return_sigma=False,
) -> xr.DataArray:
    """
    Perform local PCA-based denoising on a complex-valued xarray.DataArray,
    with automated dimension handling and optional median filtering of the noise map.

    Parameters
    ----------
    input_xarray: xr.DataArray
        Input data.
    patch_radius: int or list
        The radius of the local patch.
    mppca_slow: bool
        f True, use the slower but more robust MPPCA variant.
    sigma: float or array
        Noise standard deviation. Must be provided.
    pca_method: str
        'eig' or 'svd'.
    tau_factor: float
        Thresholding factor for component selection.
    return_sigma: bool
        If True, also returns the estimated sigma map.
    out_dtype:
        Dtype for the output array.
    db_mode: bool
        If True, prints debug information.
    spatial_dims_present: list
        A list of spatial dimension names.
    redundant_dims_present: list
        A list of redundant dimension names.
    apply_median_filter: bool
        If True, applies a median filter to the provided sigma map.
    median_filter_patch_radius: list
        Patch radius for the median filter.

    Returns
    -------
    denoised data: xr.DataArray
        The denoised data in an xarray.DataArray.

    """
    # --- 1. Identify Spatial and Redundant Dimensions ---
    if db_mode:
        print("--- Step 1: Identifying Dimensions ---")

    SPATIAL_DIMS = ["read_idx", "phase_idx", "slice_idx"]
    if spatial_dims_present is None:
        spatial_dims_present = [dim for dim in SPATIAL_DIMS if dim in input_xarray.dims]
    if redundant_dims_present is None:
        redundant_dims_present = [
            dim for dim in input_xarray.dims if dim not in spatial_dims_present
        ]

    if not spatial_dims_present:
        raise ValueError("No spatial dimensions found in the xarray.")
    if not redundant_dims_present:
        raise ValueError("No redundant dimensions for PCA found.")

    if db_mode:
        print(f"Spatial dimensions identified: {spatial_dims_present}")
        print(f"Redundant dimensions identified: {redundant_dims_present}\n")

    # --- 2. Reshape Data for Local PCA ---
    if db_mode:
        print("--- Step 2: Reshaping Data for Local PCA ---")

    original_dims = input_xarray.dims

    # Preserve multi-dimensional coordinates before stacking
    coords_to_preserve = {}
    coords_to_drop = []
    for d in redundant_dims_present:
        if d in input_xarray.coords and input_xarray.coords[d].ndim > 1:
            coords_to_preserve[d] = input_xarray.coords[d]
            coords_to_drop.append(d)

    temp_input_xarray = input_xarray.drop_vars(coords_to_drop, errors="ignore")

    # Create a view with a single redundant dimension
    temp_stacked_xarray = temp_input_xarray.stack(redundant=redundant_dims_present)

    # Create a 4D numpy array from the xarray for the local PCA logic
    arr = temp_stacked_xarray.transpose(*spatial_dims_present, "redundant").values

    # Pad spatial dimensions to 3 if necessary, as genpca expects 3 spatial
    # dims
    spatial_shape = arr.shape[:-1]
    if len(spatial_shape) < 3:
        padding = (1,) * (3 - len(spatial_shape))
        padded_shape = spatial_shape + padding + (arr.shape[-1],)
        arr = arr.reshape(padded_shape)

    if db_mode:
        print(f"Shape of array passed to local PCA: {arr.shape}\n")

    # --- 3. Estimate Noise Map if not provided ---
    if sigma is None:
        if db_mode:
            print("--- Step 3: Estimating Noise Map ---")

        if "repetition_idx" not in redundant_dims_present:
            raise ValueError(
                "Cannot estimate sigma automatically without 'repetition_idx' dimension. "
                "Please provide a sigma map manually."
            )

        if db_mode:
            print(
                "  Estimating noise from variation along 'repetition_idx' "
                "(other redundant dims fixed at index 0)."
            )

        selector = {dim: 0 for dim in redundant_dims_present if dim != "repetition_idx"}
        b0_equivalent_data = input_xarray.isel(selector)
        sigma_xarray = b0_equivalent_data.std(dim="repetition_idx")
        sigma = sigma_xarray.transpose(*spatial_dims_present).values
        if db_mode:
            # fig, ax = plt.subplots(1)
            # ax.plot(np.squeeze(np.mean(sigma, axis=(0, 1))), label="Sigma mean")

            print("  Sigma map estimated successfully.")

    # --- 4. Apply Median Filter to Sigma Map if Requested ---
    if apply_median_filter and sigma is not None:
        if db_mode:
            print("--- Step 4: Applying Median Filter to Sigma Map ---")

        sigma_padded = sigma
        if sigma.ndim < 3:
            sigma_padding = (1,) * (3 - sigma.ndim)
            sigma_padded_shape = sigma.shape + sigma_padding
            sigma_padded = sigma.reshape(sigma_padded_shape)

        sigma = median_filt(sigma_padded, patch_radius=patch_radius)
        if db_mode:
            pass
            # ax.plot(np.squeeze(np.mean(sigma, axis=(0, 1))))

        # Ensure sigma has the correct final shape by removing any added padding
        # if sigma.ndim > len(spatial_shape):
        #     sigma = sigma.reshape(spatial_shape)

    # --- 5. Run Local PCA Denoising ---
    if db_mode:
        print("--- Step 5: Running Local PCA Denoising ---")
    mask = np.ones(arr.shape[:-1], dtype=bool)

    if out_dtype is None:
        out_dtype = arr.dtype

    # Determine calculation dtype, preserving complex types
    if np.iscomplexobj(arr):
        calc_dtype = np.complex128 if arr.dtype == np.complex128 else np.complex64
    else:
        calc_dtype = np.float64 if arr.dtype == np.float64 else np.float32

    float_calc_dtype = (
        np.float64 if calc_dtype in [np.complex128, np.float64] else np.float32
    )

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.", arr.shape)

    is_svd = pca_method.lower() == "svd"
    svd_args = {"full_matrices": False} if is_svd else {}

    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    patch_radius = np.asarray(patch_radius).astype(int)

    for i in range(3):
        if arr.shape[i] == 1:
            patch_radius[i] = 0

    patch_size = 2 * patch_radius + 1
    m = np.prod(patch_size)

    if tau_factor is None:
        tau_factor = _calc_tau_factor(
            data_xarray=input_xarray,
            pr=patch_radius,
            redundant_dims=redundant_dims_present,
            db_mode=db_mode,
        )

    if m < arr.shape[-1]:
        raise ValueError(
            f"Patch size {m} is smaller than the number of redundant measurements. {arr.shape[-1]}"
        )

    if db_mode:
        print(f"tau_factor: {tau_factor}")
        fig, ax = plt.subplots(1)
        im = ax.imshow(np.squeeze(sigma[:, :, -1]))
        plt.colorbar(im, ax=ax, orientation="horizontal")

    if isinstance(sigma, (int, float)):
        sigma = sigma * np.ones(arr.shape[:-1])

    dim = arr.shape[-1]
    theta = np.zeros(arr.shape, dtype=float_calc_dtype)
    thetax = np.zeros(arr.shape, dtype=calc_dtype)
    ncall = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)
    thetavar = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)
    if return_sigma is True:
        out_sigma = np.zeros(arr.shape[:-1], dtype=float_calc_dtype)

    # Get center voxel indices for debug printing
    center_i = (arr.shape[0] - 2 * patch_radius[0]) // 2 + patch_radius[0]
    center_j = (arr.shape[1] - 2 * patch_radius[1]) // 2 + patch_radius[1]
    center_k = (arr.shape[2] - 2 * patch_radius[2]) // 2 + patch_radius[2]

    for k in range(patch_radius[2], arr.shape[2] - patch_radius[2]):
        for j in range(patch_radius[1], arr.shape[1] - patch_radius[1]):
            for i in range(patch_radius[0], arr.shape[0] - patch_radius[0]):
                if not mask[i, j, k]:
                    continue
                ix1, ix2 = i - patch_radius[0], i + patch_radius[0] + 1
                jx1, jx2 = j - patch_radius[1], j + patch_radius[1] + 1
                kx1, kx2 = k - patch_radius[2], k + patch_radius[2] + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(m, dim)
                M = np.mean(X, axis=0)
                X = X - M

                if is_svd:
                    # Scale data to avoid overflow during squared singular value calculation
                    X_scaled = X / np.sqrt(X.shape[0])
                    _, S, Vt = svd(X_scaled, **svd_args)
                    d = S[::-1] ** 2
                    W = Vt[::-1].T
                else:
                    # Scale data to avoid overflow during matrix multiplication
                    X_scaled = X / np.sqrt(X.shape[0])
                    C = np.transpose(X_scaled).dot(X_scaled)
                    d, W = eigh(C)

                if sigma is None:
                    this_var, ncomps = (
                        _pca_classifier_slow(d, m)
                        if mppca_slow
                        else _pca_classifier(d, m)
                    )
                else:
                    this_var = sigma[i, j, k] ** 2
                    if tau_factor is None:
                        this_var, ncomps = _gpca_classifier(d, this_var)
                    else:
                        tau = tau_factor**2 * this_var
                        ncomps = np.sum(d < tau)

                if db_mode and i == center_i and j == center_j and k == center_k:
                    print(f"this_var: {this_var:.4f}")
                    print(f"    [Debug for center voxel ({i},{j},{k})]")
                    print(f"    Estimated noise variance(this_var): {this_var: .4f}")
                    if sigma is None:
                        print(f"    Method: MPPCA (random matrix theory)")
                    else:
                        if tau_factor is None:
                            print(f"    Method: GPCA (variance comparison)")
                        else:
                            print(f"    Method: TPCA (pre-defined tau_factor)")
                            print(f"    Threshold (tau): {tau:.4f}")
                    print(f"    Eigenvalues classified as noise (ncomps): {ncomps}")
                    print(f"    Eigenvalues classified as signal: {dim - ncomps}")

                # if db_mode:
                # print(f"keeping: {ncomps} components at voxel ({i}, {j}, {k})")
                W[:, :ncomps] = 0
                Xest = X.dot(W).dot(W.T) + M
                Xest = Xest.reshape(patch_size[0], patch_size[1], patch_size[2], dim)
                this_theta = 1.0 / (1.0 + dim - ncomps)
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta
                ncall[ix1:ix2, jx1:jx2, kx1:kx2] += (dim - ncomps) * this_theta
                thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                if return_sigma is True:
                    out_sigma[ix1:ix2, jx1:jx2, kx1:kx2] += (
                        np.sqrt(this_var) * this_theta
                    )

    with np.errstate(divide="ignore", invalid="ignore"):
        denoised_arr = thetax / theta
    denoised_arr[mask == 0] = 0
    denoised_arr = np.nan_to_num(denoised_arr)

    # --- 5. Reshape Data Back to Original Xarray Format ---
    if db_mode:
        print("--- Step 5: Reshaping Data to Original Format ---")

    # Remove padding dimensions
    denoised_arr = denoised_arr.reshape(spatial_shape + (arr.shape[-1],))

    # Create a new xarray with the denoised data
    denoised_stacked_xarray = xr.DataArray(
        data=denoised_arr,
        coords=temp_stacked_xarray.transpose(*spatial_dims_present, "redundant").coords,
        dims=temp_stacked_xarray.transpose(*spatial_dims_present, "redundant").dims,
    )

    # Unstack the redundant dimension and transpose back to original order
    input_xarray_dn = denoised_stacked_xarray.unstack("redundant").transpose(
        *original_dims
    )
    input_xarray_dn.attrs = input_xarray.attrs
    input_xarray_dn.name = input_xarray.name

    # Restore multi-dimensional coordinates
    if coords_to_preserve:
        input_xarray_dn = input_xarray_dn.assign_coords(coords_to_preserve)

    ncall = ncall / thetavar

    if return_sigma is True:
        out_sigma = out_sigma / thetavar
        out_sigma[mask == 0] = 0

    if db_mode:
        print("\n--- Denoising Complete ---")
        print(f"Shape of final denoised xarray: {input_xarray_dn.shape}\n")

    return input_xarray_dn.astype(out_dtype)
