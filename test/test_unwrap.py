#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test 2D phase unwrapping against wrapped analytic surfaces (Gaussian
bumps, Zernike wavefronts), vortices and noisy wavefronts.
"""

import numpy as np
import pytest

from pyoptics.bases import FringeZernikes
from pyoptics.masks import circular_mask
from pyoptics.processing import CameraNoiser
from pyoptics.unwrap import (residues, reliability, synthetic_wavelength, unwrap,
                             unwrap_1d, unwrap_ls, unwrap_quality_guided,
                             unwrap_two_wavelength, unwrap_with_reference, wrap)
from pyoptics.utils import grid2d


N = 128


@pytest.fixture(scope="module")
def grid():
    return grid2d(-1, 1, -1, 1, N)


def gaussian_bump(X, Y):
    # peak 40 rad plus a tilt: about 12 wraps, but < π per pixel
    return 40*np.exp(-((X - 0.1)**2 + (Y + 0.2)**2)/0.2) + 12*X - 5*Y


def zernike_wavefront(x, y):
    Z = FringeZernikes(x, y, R_norm=0.9)
    # defocus, astigmatism, coma, spherical, in radians
    phi = Z([4, 5, 7, 9], [20.0, -12.0, 8.0, 6.0])
    return phi, Z.mask.astype(bool)


def assert_equal_up_to_piston(u, phi, valid, atol=1E-8):
    d = (u - phi)[valid]
    assert np.all(np.isfinite(d))
    assert np.ptp(d) < atol


def test_wrap_range_and_congruence():
    phi = np.linspace(-50, 50, 10001)
    w = wrap(phi)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    k = (phi - w)/(2*np.pi)
    assert np.allclose(k, np.rint(k), atol=1E-9)
    assert wrap(np.pi) == pytest.approx(-np.pi)
    assert np.isnan(wrap(np.nan))


def test_unwrap_1d():
    t = np.linspace(0, 1, 500)
    phi = 60*t**2 - 10*t
    assert np.allclose(unwrap_1d(wrap(phi)) - phi, (wrap(phi) - phi)[0])
    assert np.allclose(unwrap(wrap(phi)), unwrap_1d(wrap(phi)))


@pytest.mark.parametrize("func, kwargs", [
    (unwrap_ls, {"method": "dct"}),
    (unwrap_ls, {"method": "pcg"}),
    (unwrap_ls, {"weights": np.full((N, N), 0.5)}),
    (unwrap_quality_guided, {}),
])
def test_gaussian_bump_unwraps_exactly(grid, func, kwargs):
    x, y, X, Y = grid
    phi = gaussian_bump(X, Y)
    psi = wrap(phi)
    u = func(psi, **kwargs)
    assert_equal_up_to_piston(u, phi, np.ones_like(phi, dtype=bool))
    # the piston is a multiple of 2π: the result is congruent to the data
    assert np.allclose(np.cos(u - psi), 1.0, atol=1E-12)
    assert -np.pi < np.mean(u) <= np.pi


@pytest.mark.parametrize("func", [unwrap_ls, unwrap_quality_guided])
def test_zernike_wavefront_in_pupil(grid, func):
    x, y, X, Y = grid
    phi, pupil = zernike_wavefront(x, y)
    psi = wrap(phi)
    psi[~pupil] = np.random.default_rng(0).uniform(-np.pi, np.pi, np.sum(~pupil))
    u = func(psi, mask=pupil)
    assert_equal_up_to_piston(u, phi, pupil)
    assert np.all(np.isnan(u[~pupil]))


def test_nan_is_mask(grid):
    x, y, X, Y = grid
    phi, pupil = zernike_wavefront(x, y)
    psi = np.where(pupil, wrap(phi), np.nan)
    for u in (unwrap_ls(psi), unwrap_quality_guided(psi)):
        assert_equal_up_to_piston(u, phi, pupil)
        assert np.all(np.isnan(u[~pupil]))


def test_pcg_info_and_dct_rejects_weights(grid):
    x, y, X, Y = grid
    psi = wrap(gaussian_bump(X, Y))
    _, info = unwrap_ls(psi, method="pcg", return_info=True)
    assert info["residual"] < 1E-10
    assert info["iterations"] < 10  # the DCT preconditioner is exact here
    with pytest.raises(ValueError):
        unwrap_ls(psi, weights=np.ones_like(psi), method="dct")


def test_disconnected_regions_get_own_piston(grid):
    x, y, X, Y = grid
    phi = gaussian_bump(X, Y)
    mask = np.abs(X) > 0.2
    u = unwrap_quality_guided(wrap(phi), mask=mask)
    for region in (X < -0.2, X > 0.2):
        assert_equal_up_to_piston(u, phi, region)


@pytest.mark.parametrize("charge", [1, -1, 2, -3])
def test_residues_of_vortex(grid, charge):
    x, y, X, Y = grid
    x0, y0 = 0.3*(x[1] - x[0]), -0.4*(y[1] - y[0])  # core between pixels
    field = ((X - x0) + 1j*np.sign(charge)*(Y - y0))**abs(charge)*np.exp(-X**2 - Y**2)
    r = residues(np.angle(field))
    assert r.shape == (N - 1, N - 1)
    assert r.sum() == charge
    # the smallest loop around a higher order core has steps of π and more,
    # so a charge of 2 or 3 shows up as several residues next to the core
    c = N//2 - 1
    assert r[c - 3:c + 4, c - 3:c + 4].sum() == charge
    if abs(charge) > 1:
        return
    assert np.count_nonzero(r) == 1
    i, j = np.argwhere(r)[0]
    # the loop around [i, j] .. [i+1, j+1] contains the core:
    assert y[i] < y0 < y[i + 1] and x[j] < x0 < x[j + 1]


def test_smooth_phase_has_no_residues(grid):
    x, y, X, Y = grid
    assert not np.any(residues(wrap(gaussian_bump(X, Y))))


def test_reliability(grid):
    x, y, X, Y = grid
    R = reliability(wrap(5*X**2))  # constant second difference
    assert np.all(R[0, :] == 0) and np.all(R[:, -1] == 0)
    inner = R[1:-1, 1:-1]
    assert np.allclose(inner, inner[0, 0])
    d2 = 2*5*(x[1] - x[0])**2  # H = D1 = D2, V = 0
    assert inner[0, 0] == pytest.approx(1/(np.sqrt(3)*d2))


def test_congruent_ls_matches_data_and_truth(grid):
    x, y, X, Y = grid
    phi = gaussian_bump(X, Y)
    rng = np.random.default_rng(1)
    psi = wrap(phi + 0.3*rng.standard_normal(phi.shape))
    u = unwrap_ls(psi, congruent=True)
    assert np.allclose(wrap(u - psi), 0.0, atol=1E-9)


def noisy_wavefront(X, Y, photons, seed=0):
    """Wrapped phase of a field with camera noise on real and imaginary
    parts (as in a phase shifting measurement), and its amplitude."""

    phi = gaussian_bump(X, Y)
    amp = np.exp(-(X**2 + Y**2)/1.5)
    E = np.sqrt(photons)*amp*np.exp(1j*phi)
    noiser = CameraNoiser("gaussian", sigma=1.0, rng=seed)
    E_noisy = noiser(E)
    return phi, np.angle(E_noisy), np.abs(E_noisy), amp


@pytest.mark.parametrize("func, use_weights", [
    (unwrap_ls, False), (unwrap_ls, True), (unwrap_quality_guided, True),
    (unwrap_quality_guided, False)])
def test_noisy_wavefront(grid, func, use_weights):
    x, y, X, Y = grid
    phi, psi, A, amp = noisy_wavefront(X, Y, photons=200.0)
    kwargs = {}
    if use_weights:
        kwargs = {"weights": A} if func is unwrap_ls else {"quality": A}
    u = func(psi, **kwargs)
    d = u - phi
    d -= np.median(d)
    # the phase noise is about sigma/|E| = 1/(sqrt(200)*amp) <= 0.14 rad; no
    # pixel may be off by 2π, and the error is of the order of the noise
    noise = 1/(np.sqrt(200.0)*amp)
    assert np.sqrt(np.mean(d**2)) < 2*np.sqrt(np.mean(noise**2))
    assert np.max(np.abs(d)) < np.pi


def test_unwrap_with_reference(grid):
    x, y, X, Y = grid
    phi = gaussian_bump(X, Y)
    rng = np.random.default_rng(0)
    # any reference within ±π (here up to ±0.9π off) gives the exact phase
    reference = phi + 0.9*np.pi*(2*rng.random(phi.shape) - 1)
    u = unwrap_with_reference(wrap(phi), reference)
    assert np.allclose(u, phi, atol=1E-10)
    assert np.allclose(unwrap(wrap(phi), method="reference", reference=reference), phi,
                       atol=1E-10)
    # a reference off by more than π picks the neighbouring multiple of 2π
    u = unwrap_with_reference(wrap(phi), phi + 1.1*np.pi)
    assert np.allclose(u, phi + 2*np.pi, atol=1E-10)
    mask = X**2 + Y**2 < 0.5
    u = unwrap_with_reference(wrap(phi), reference, mask=mask)
    assert np.all(np.isnan(u[~mask])) and np.allclose(u[mask], phi[mask], atol=1E-10)
    # works on 1D data, too (no path, so no dimension restriction)
    assert np.allclose(unwrap(wrap(phi[0]), method="reference", reference=phi[0]), phi[0])


def test_unwrap_with_reference_across_a_step(grid):
    # a step of 7 wraps cannot be unwrapped by path following, but is no
    # problem for a reference that knows it to ±π
    x, y, X, Y = grid
    phi = 3*X + np.where(X > 0.2, 7*2*np.pi + 1.0, 0.0)
    assert not np.allclose(unwrap_quality_guided(wrap(phi)) - phi, (unwrap_quality_guided(wrap(phi)) - phi)[0, 0])
    u = unwrap_with_reference(wrap(phi), phi + 2.5)
    assert np.allclose(u, phi, atol=1E-10)


def test_synthetic_wavelength():
    assert np.isclose(synthetic_wavelength(0.633, 0.532), 0.633*0.532/0.101)
    assert synthetic_wavelength(0.532, 0.633) == synthetic_wavelength(0.633, 0.532)


@pytest.mark.parametrize("wl1, wl2", [(0.633, 0.785), (0.785, 0.633), (0.633, 0.532)])
def test_two_wavelength_opd(grid, wl1, wl2):
    x, y, X, Y = grid
    Lambda = synthetic_wavelength(wl1, wl2)
    # a step of several waves plus a bump, within ±Λ/2
    opd = 0.4*Lambda*np.where(X > 0.1, 1.0, -0.4)*np.exp(-Y**2) + 0.05*np.exp(-X**2/0.1)
    assert np.max(np.abs(opd)) < Lambda/2 and np.ptp(opd) > 2*max(wl1, wl2)
    psi1, psi2 = wrap(2*np.pi*opd/wl1), wrap(2*np.pi*opd/wl2)
    coarse = unwrap_two_wavelength(psi1, psi2, wl1, wl2, refine=False)
    assert np.allclose(coarse, opd, atol=1E-10)
    assert np.allclose(unwrap_two_wavelength(psi1, psi2, wl1, wl2), opd, atol=1E-10)

    # phase noise: the coarse OPD has Λ/λ times the noise, the refined one
    # only that of the single wavelengths (averaged)
    rng = np.random.default_rng(1)
    sigma = 0.02
    n1, n2 = sigma*rng.standard_normal(opd.shape), sigma*rng.standard_normal(opd.shape)
    coarse = unwrap_two_wavelength(psi1 + n1, psi2 + n2, wl1, wl2, refine=False)
    fine = unwrap_two_wavelength(psi1 + n1, psi2 + n2, wl1, wl2)
    noise_single = sigma*max(wl1, wl2)/(2*np.pi)
    assert np.std(coarse - opd) > 3*noise_single
    assert np.std(fine - opd) < noise_single

    mask = X > 0
    o = unwrap_two_wavelength(psi1, psi2, wl1, wl2, mask=mask)
    assert np.all(np.isnan(o[~mask]))


def test_two_wavelength_range_and_errors(grid):
    x, y, X, Y = grid
    wl1, wl2 = 0.633, 0.785
    Lambda = synthetic_wavelength(wl1, wl2)
    opd = np.full(X.shape, 0.6*Lambda)  # beyond Λ/2: aliased by one Λ
    o = unwrap_two_wavelength(wrap(2*np.pi*opd/wl1), wrap(2*np.pi*opd/wl2), wl1, wl2,
                              refine=False)
    assert np.allclose(o, opd - Lambda)
    with pytest.raises(ValueError):
        unwrap_two_wavelength(opd, opd, wl1, wl1)
