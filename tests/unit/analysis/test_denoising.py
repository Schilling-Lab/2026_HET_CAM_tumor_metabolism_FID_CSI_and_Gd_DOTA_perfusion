import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.denoising.algorithms import denoise_pca


def test_denoise_pca_overflow_protection():
    """
    Test that denoise_pca handles large values that would normally overflow
    during covariance matrix calculation if not scaled properly.
    """
    n_pixels = 1000000
    n_reps = 5

    # Values that overflow float32 if squared and summed over 1e6 pixels
    # (1e20)^2 * 1e6 = 1e46 (> 3.4e38)
    data = np.random.normal(loc=1e20, scale=1e19, size=(n_reps, n_pixels)).astype(
        np.float32
    )

    da = xr.DataArray(
        data.reshape(n_reps, 1000, 1000),
        dims=["repetition_idx", "read_idx", "phase_idx"],
        name="test_data",
    )

    # This should succeed without overflow thanks to the scaling fix
    try:
        dn = denoise_pca(da, components=2)
        assert dn is not None
        assert dn.shape == da.shape
    except Exception as e:
        pytest.fail(f"denoise_pca failed with overflow error even with fix: {e}")


def test_denoise_pca_svd_overflow_protection():
    """
    Test that denoise_pca with pca_method='svd' handles large values.
    """
    n_pixels = 1000000
    n_reps = 5
    data = np.random.normal(loc=1e20, scale=1e19, size=(n_reps, n_pixels)).astype(
        np.float32
    )
    da = xr.DataArray(
        data.reshape(n_reps, 1000, 1000),
        dims=["repetition_idx", "read_idx", "phase_idx"],
    )

    try:
        dn = denoise_pca(da, components=2, pca_method="svd")
        assert dn is not None
        assert dn.shape == da.shape
    except Exception as e:
        pytest.fail(f"denoise_pca (SVD) failed with overflow error: {e}")


def test_denoise_pca_basic():
    """Basic functional test for denoise_pca."""
    n_reps = 10
    nx, ny = 32, 32

    # Create signal + noise
    t = np.linspace(0, 1, n_reps)
    signal = np.exp(-t[:, None, None] * 5) * np.ones((n_reps, nx, ny))
    noise = np.random.normal(0, 0.1, (n_reps, nx, ny))
    data = signal + noise

    da = xr.DataArray(
        data, dims=["repetition_idx", "read_idx", "phase_idx"], name="test_data"
    )

    dn = denoise_pca(da, components=1)
    assert dn is not None
    assert dn.shape == da.shape

    # Denoised data should be closer to signal than original data
    orig_error = np.mean((data - signal) ** 2)
    dn_error = np.mean((dn.values - signal) ** 2)
    assert dn_error < orig_error
