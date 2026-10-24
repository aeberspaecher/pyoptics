#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the chirp-z transform and the propagators onto free output windows.

All lengths in units of the wavelength (wl = 1). References:

- direct evaluation of the DFT sums and the FFT based routines of pyoptics
- tilted plane wave (exact for the angular spectrum)
- exact paraxial Gaussian beam (Fresnel integral)
- far field of a Gaussian (Fourier integral)
- Airy pattern
"""

import numpy as np
import pytest

from pyoptics.analytical import airy_pattern
from pyoptics.fft import FT, czt, inv_FT, zoom_FT, zoom_inv_FT
from pyoptics.imaging import get_psf
from pyoptics.propagators import (far_field, fraunhofer, fraunhofer_scaled,
                                  fresnel_rescaling, fresnel_scaled,
                                  rayleigh_sommerfeld_I_TF,
                                  shifted_angular_spectrum)
from pyoptics.utils import frequencies, grid1d, pad


WL = 1.0
K = 2*np.pi/WL


def _centered(N, dx):
    return dx*(np.arange(N) - N//2)


def _random(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j*rng.normal(size=shape)


# chirp-z transform
# -----------------

@pytest.mark.parametrize("N, M", [(16, 16), (37, 50), (50, 7), (1, 5), (8, 1)])
def test_czt_equals_direct_sum(N, M):
    x = _random(N)
    # |w| != 1 makes the chirps grow exponentially, keep it close to 1:
    w, a = 0.9995*np.exp(-0.123j), 1.01*np.exp(0.4j)
    n, m = np.arange(N), np.arange(M)
    X_direct = np.sum(x[None, :]*a**(-n[None, :])*w**(n[None, :]*m[:, None]), axis=1)

    assert np.allclose(czt(x, M, w, a), X_direct, rtol=1E-10, atol=1E-10*np.abs(X_direct).max())


@pytest.mark.parametrize("N", [8, 13, 64])
def test_czt_defaults_give_fft(N):
    x = _random(N)

    assert np.allclose(czt(x), np.fft.fft(x), atol=1E-11)


def test_czt_along_axis():
    x = _random((5, 12, 3))
    X = czt(x, 20, np.exp(-0.2j), np.exp(0.1j), axis=1)

    assert X.shape == (5, 20, 3)
    assert np.allclose(X[2, :, 1], czt(x[2, :, 1], 20, np.exp(-0.2j), np.exp(0.1j)))


def test_czt_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    if not hasattr(signal, "czt"):
        pytest.skip("scipy.signal.czt needs SciPy >= 1.8")
    x = _random(40)
    w, a = np.exp(-0.05j), np.exp(0.3j)

    assert np.allclose(czt(x, 64, w, a), signal.czt(x, 64, w, a))


# zoomed transforms
# -----------------

@pytest.mark.parametrize("shape", [(8, 8), (6, 10), (7, 9)])
def test_zoom_FT_equals_FT_on_centered_grid(shape):
    f = _random(shape)
    x, y = _centered(shape[1], 0.3), _centered(shape[0], 0.4)
    kx, ky = frequencies(x), frequencies(y)

    assert np.allclose(zoom_FT(f, x, y, kx, ky), FT(f))
    assert np.allclose(zoom_inv_FT(FT(f), kx, ky, x, y), f)


def test_zoom_FT_equals_direct_sum_on_any_grid():
    f = _random((9, 11))
    x, y = 2.0 + 0.3*np.arange(11), -1.0 + 0.25*np.arange(9)  # not centered
    kx_out, ky_out = np.linspace(-1.0, 3.0, 17), np.linspace(0.5, 0.9, 5)
    X, Y = np.meshgrid(x, y)
    F_direct = np.array([[np.sum(f*np.exp(-1j*(kx*X + ky*Y))) for kx in kx_out]
                         for ky in ky_out])

    assert np.allclose(zoom_FT(f, x, y, kx_out, ky_out), F_direct)


def test_zoom_FT_equals_padded_FFT():
    # a frequency grid 4x finer than 2*pi/L is the spectrum of the 4x padded field
    N, dx, factor = 16, 0.5, 4
    f = _random((N, N))
    x = _centered(N, dx)
    f_pad, x_pad, y_pad = pad(f, x, x, factor=factor)
    kx = frequencies(x_pad)

    assert np.allclose(zoom_FT(f, x, x, kx, kx), FT(f_pad))


def test_zoom_inv_FT_interpolates():
    # a band-limited field is reproduced between the samples
    N, dx = 32, 0.5
    x = _centered(N, dx)
    kx = frequencies(x)
    field = lambda X, Y: np.exp(1j*(kx[N//2 + 3]*X - kx[N//2 - 5]*Y))
    X, Y = np.meshgrid(x, x)
    x_fine = np.linspace(-1.3, 2.1, 23)
    X_f, Y_f = np.meshgrid(x_fine, x_fine)

    assert np.allclose(zoom_inv_FT(FT(field(X, Y)), kx, kx, x_fine, x_fine), field(X_f, Y_f))


def test_zoom_needs_equidistant_coordinates():
    f = _random((4, 4))
    x = np.array([0.0, 1.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        zoom_FT(f, x, x, x, x)


# shifted angular spectrum
# ------------------------

def test_shifted_angular_spectrum_tilted_plane_wave():
    # a periodic plane wave on the grid frequencies is exact for any window
    L, N = 32.0, 64
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    kx, ky = frequencies(x)[N//2 + 7], frequencies(x)[N//2 - 3]
    kz = np.sqrt(K**2 - kx**2 - ky**2)
    z = 50.0
    x_out, y_out = np.linspace(100.0, 103.0, 31), np.linspace(-40.0, -38.0, 11)
    X_o, Y_o = np.meshgrid(x_out, y_out)

    U, _, _ = shifted_angular_spectrum(np.exp(1j*(kx*X + ky*Y)), x, x, z, WL, x_out, y_out,
                                       band_limit=False, pad_factor=1)

    assert np.allclose(U, np.exp(1j*(kx*X_o + ky*Y_o + kz*z)))


@pytest.fixture(scope="module")
def tilted_gaussian():
    """Tilted Gaussian (10 deg), propagated by z = 200 to an off-axis window,
    with a reference from a large padded angular spectrum computation.
    """

    L, N, z, theta = 64.0, 128, 200.0, np.deg2rad(10)
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    f = np.exp(-(X**2 + Y**2)/5.0**2)*np.exp(1j*K*np.sin(theta)*X)

    N_ref = 1024
    f_ref, x_ref, _ = pad(f, x, x, shape=(N_ref, N_ref))
    U_ref, _, _ = rayleigh_sommerfeld_I_TF(f_ref, x_ref, x_ref, z, WL)
    i0 = np.argmin(np.abs(x_ref - z*np.tan(theta)))
    rows, cols = slice(N_ref//2 - 64, N_ref//2 + 64), slice(i0 - 64, i0 + 64)

    return f, x, z, x_ref[cols], x_ref[rows], U_ref[rows, cols]


def test_shifted_angular_spectrum_off_axis_window(tilted_gaussian):
    f, x, z, x_out, y_out, U_ref = tilted_gaussian
    U, x_o, y_o = shifted_angular_spectrum(f, x, x, z, WL, x_out, y_out)

    assert U.shape == (len(y_out), len(x_out))
    assert x_o is x_out
    assert np.abs(U - U_ref).max() < 1E-6*np.abs(U_ref).max()


def test_shifted_angular_spectrum_rescaled_window(tilted_gaussian):
    # a window with dx_out = dx/3 agrees with the window on the input sampling;
    # the smaller window has a narrower band limit, which costs some accuracy
    f, x, z, x_out, y_out, _ = tilted_gaussian
    U, _, _ = shifted_angular_spectrum(f, x, x, z, WL, x_out, y_out)
    x_fine, y_fine = np.linspace(x_out[40], x_out[80], 121), np.linspace(y_out[40], y_out[80], 121)
    U_fine, _, _ = shifted_angular_spectrum(f, x, x, z, WL, x_fine, y_fine)

    assert np.abs(U_fine[::3, ::3] - U[40:81, 40:81]).max() < 1E-4*np.abs(U).max()


def test_shifted_angular_spectrum_uncentered_input():
    # shifting the input grid by a shifts the field
    L, N, z, a = 32.0, 64, 30.0, 5.0
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    f = np.exp(-(X**2 + Y**2)/3.0**2)
    x_out = np.linspace(-4.0, 4.0, 33)

    U, _, _ = shifted_angular_spectrum(f, x, x, z, WL, x_out, x_out)
    U_shifted, _, _ = shifted_angular_spectrum(f, x + a, x, z, WL, x_out + a, x_out)

    assert np.allclose(U_shifted, U, atol=1E-10)


def test_shifted_angular_spectrum_warns_for_too_coarse_spectrum():
    L, N = 16.0, 32
    x = grid1d(-L/2, L/2, N)

    with pytest.warns(UserWarning, match="pad_factor"):
        shifted_angular_spectrum(np.ones((N, N)), x, x, 4000.0, WL, x + 600.0, x)


def test_band_limit_suppresses_replicas():
    # without padding and band limit, light reenters the window
    L, N, z = 32.0, 64, 100.0
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    f = (np.abs(X) < 4)*(np.abs(Y) < 4)*1.0
    f_ref, x_ref, _ = pad(f, x, x, factor=16)
    U_ref, _, _ = rayleigh_sommerfeld_I_TF(f_ref, x_ref, x_ref, z, WL, band_limit=True)
    U_ref = U_ref[8*N - N//2:8*N + N//2, 8*N - N//2:8*N + N//2]

    U, _, _ = shifted_angular_spectrum(f, x, x, z, WL, x, x)
    U_aliased, _, _ = shifted_angular_spectrum(f, x, x, z, WL, x, x, band_limit=False,
                                               pad_factor=1)

    err = np.abs(U - U_ref).max()/np.abs(U_ref).max()
    err_aliased = np.abs(U_aliased - U_ref).max()/np.abs(U_ref).max()
    assert err < 0.05
    assert err_aliased > 4*err


# Fresnel, Fraunhofer, far field
# ------------------------------

def _paraxial_gaussian(X, Y, z, w0):
    q0 = -1j*K*w0**2/2
    q = q0 + z

    return q0/q*np.exp(1j*K*z)*np.exp(1j*K*(X**2 + Y**2)/(2*q))


def test_fresnel_scaled_gaussian_off_axis_fine_window():
    L, N, w0, z = 40.0, 128, 3.0, 60.0
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    x_out, y_out = np.linspace(2.0, 9.0, 71), np.linspace(-6.0, -1.0, 41)
    X_o, Y_o = np.meshgrid(x_out, y_out)

    U, _, _ = fresnel_scaled(_paraxial_gaussian(X, Y, 0, w0), x, x, z, WL, x_out, y_out)

    assert np.allclose(U, _paraxial_gaussian(X_o, Y_o, z, w0), atol=1E-10)


@pytest.mark.parametrize("propagator, scaled", [(fresnel_rescaling, fresnel_scaled),
                                                (fraunhofer, fraunhofer_scaled)])
def test_scaled_equals_fixed_grid_versions(propagator, scaled):
    L, N, z = 40.0, 64, 300.0
    x = grid1d(-L/2, L/2, N)
    f = _random((N, N))

    U, x_new, y_new = propagator(f, x, x, z, WL)
    U_scaled, _, _ = scaled(f, x, x, z, WL, x_new, y_new)

    assert np.allclose(U_scaled, U)


def test_fresnel_scaled_warns_for_undersampled_chirp():
    L, N = 40.0, 64
    x = grid1d(-L/2, L/2, N)
    f = np.ones((N, N))

    with pytest.warns(UserWarning, match="chirp is undersampled"):
        fresnel_scaled(f, x, x, 5.0, WL, x, x)
    with pytest.warns(UserWarning, match="replicas"):
        fraunhofer_scaled(f, x, x, 10.0, WL, 10*x, 10*x)


def test_far_field_of_gaussian():
    # f(s) = k*sigma_z/(2*pi*i)*pi*w0**2*exp(-k**2*w0**2*sigma_perp**2/4)
    L, N, w0, n = 40.0, 128, 2.0, 1.5
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    s_x, s_y = np.linspace(-0.3, 0.5, 41), np.linspace(0.0, 0.2, 11)
    S_X, S_Y = np.meshgrid(s_x, s_y)
    k = n*K

    f, _, _ = far_field(np.exp(-(X**2 + Y**2)/w0**2), x, x, WL, s_x, s_y, n=n)
    f_exact = (k*np.sqrt(1 - S_X**2 - S_Y**2)/(2j*np.pi)*np.pi*w0**2
               *np.exp(-k**2*w0**2*(S_X**2 + S_Y**2)/4))

    assert np.allclose(f, f_exact, atol=1E-10)


def test_far_field_agrees_with_fresnel_at_small_angles():
    # U(r*s) = f(s)*exp(i*k*r)/r, paraxially the Fresnel integral at z = r*sigma_z
    L, N, r = 16.0, 64, 4E4  # Fresnel number of the source ~ 1E-4
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    field = np.exp(-((X - 1.0)**2 + Y**2)/1.5**2)*(1 + 0.5j*Y)
    sigma = np.linspace(-0.01, 0.01, 5)
    f, _, _ = far_field(field, x, x, WL, sigma, sigma)

    S_X, S_Y = np.meshgrid(sigma, sigma)
    z = r*np.sqrt(1 - S_X**2 - S_Y**2)
    U = np.array([[fresnel_scaled(field, x, x, z[i, j], WL, r*sigma[j:j + 1], r*sigma[i:i + 1])[0][0, 0]
                   for j in range(len(sigma))] for i in range(len(sigma))])

    assert np.allclose(U, f*np.exp(1j*K*r)/r, rtol=0, atol=2E-3*np.abs(U).max())


# zoomed PSF
# ----------

NA = 0.5


def test_zoomed_psf_equals_psf_on_grid():
    L, N = 40*WL, 128
    x = grid1d(-L/2, L/2, N)
    phase = lambda P_x, P_y: 0.7*P_x*(P_x**2 + P_y**2)  # some coma

    psf = get_psf(x, x, NA, K, phase=phase)
    psf_zoom = get_psf(x, x, NA, K, phase=phase, x_out=x[40:90], y_out=x[50:70])

    assert np.allclose(psf_zoom, psf[50:70, 40:90])


def test_zoomed_psf_is_airy_pattern():
    # coarse grid (pupil radius 50 samples), fine output over the Airy disk
    L, N = 100*WL, 256
    x = grid1d(-L/2, L/2, N)
    x_out = np.linspace(-2.0, 2.0, 201)

    psf = get_psf(x, x, NA, K, x_out=x_out, y_out=np.array([0.0]))
    I = np.abs(psf[0])**2

    assert np.isclose(I[100], 1.0)
    assert np.allclose(I, airy_pattern(np.abs(x_out), NA, WL), atol=5E-3)


def test_zoomed_psf_energy_normalization():
    L, N = 40*WL, 128
    x = grid1d(-L/2, L/2, N)
    psf = get_psf(x, x, NA, K, normalization="energy")
    psf_zoom = get_psf(x, x, NA, K, normalization="energy", x_out=x[60:70], y_out=x[60:70])

    assert np.allclose(psf_zoom, psf[60:70, 60:70])
    with pytest.raises(ValueError):
        get_psf(x, x, NA, K, x_out=x)
