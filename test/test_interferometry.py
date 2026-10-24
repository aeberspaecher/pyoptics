#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the phase-shifting algorithms and the Fourier-transform method of
pyoptics.interferometry against known phases.
"""

import warnings

import numpy as np
import pytest

from pyoptics.interferometry import (find_carrier, fourier_fringe_analysis,
                                     interfere, interferogram, phase_shifts,
                                     psi_4_step, psi_carre, psi_hariharan,
                                     psi_n_step, shift_error_sensitivity,
                                     zernike_phase)
from pyoptics.processing import CameraNoiser
from pyoptics.unwrap import unwrap_ls, wrap
from pyoptics.utils import grid2d


COEFFS = [0.0, 0.2, -0.1, 1.5, 0.3, -0.2, 0.5, 0.0, 0.4]  # waves


@pytest.fixture(scope="module")
def grid():
    return grid2d(-1, 1, -1, 1, 128)


@pytest.fixture(scope="module")
def zernike(grid):
    x, y, X, Y = grid
    return zernike_phase(x, y, COEFFS, 1.0)


def _max_error(phase, phase_true, mask):
    return np.max(np.abs(wrap(phase - phase_true))[mask])


def test_zernike_phase_is_2pi_W(grid):
    x, y, X, Y = grid
    phase, mask = zernike_phase(x, y, [0.0, 0.0, 0.0, 0.5], 1.0)
    rho2 = X**2 + Y**2
    assert np.allclose(phase[mask], 2*np.pi*0.5*(2*rho2[mask] - 1))
    assert np.all(phase[~mask] == 0)

    phase2, _ = zernike_phase(x, y, [0.0, 0.0, 0.0, 0.5], 1.0, passes=2)
    assert np.allclose(phase2, 2*phase)


def test_interferogram_matches_two_fields(grid, zernike):
    """|exp(i phi) + exp(i delta)|**2 = 2*(1 + cos(phi - delta))."""

    phase, mask = zernike
    shifts = phase_shifts(4)
    I_fields = interfere(np.exp(1j*phase), 1.0, shifts)
    I_model = interferogram(phase, shift=shifts, I0=2.0)
    assert I_fields.shape == phase.shape + (4,)
    assert np.allclose(I_fields, I_model)

    # the algorithms recover arg(E_test*conj(E_ref)):
    E_ref = 0.5*np.exp(0.7j)
    rec, gamma = psi_4_step(interfere(np.exp(1j*phase), E_ref, shifts))
    assert _max_error(rec, phase - 0.7, mask) < 1E-12
    assert np.allclose(gamma, 2*0.5/(1 + 0.25))


def test_phase_shifts():
    assert np.allclose(phase_shifts(4), [0, np.pi/2, np.pi, 3*np.pi/2])
    assert np.allclose(phase_shifts(5, np.pi/2, -np.pi), np.pi/2*np.arange(-2, 3))
    assert np.allclose(phase_shifts(3, 1.0, linear_error=0.1), [0, 1.1, 2.2])
    assert np.allclose(phase_shifts(3, np.pi, quadratic_error=0.1),
                       [0, 1.1*np.pi, 2*np.pi + 0.4*np.pi])


@pytest.mark.parametrize("algorithm, N, step, centered", [
    ("n-step", 3, 2*np.pi/3, False),
    ("n-step", 7, 2*np.pi/7, False),
    ("n-step-irregular", 5, None, False),
    ("4-step", 4, np.pi/2, False),
    ("carre", 4, 1.3, True),
    ("carre", 4, np.pi/2, True),
    ("hariharan", 5, np.pi/2, True),
])
def test_psi_recovers_zernike_phase(zernike, algorithm, N, step, centered):
    phase, mask = zernike
    if algorithm == "n-step-irregular":
        shifts = np.array([0.0, 0.9, 2.1, 2.9, 4.4])
    else:
        shifts = phase_shifts(N, step, -(N - 1)*step/2 if centered else 0.0)
    gamma = 0.6
    frames = interferogram(phase, visibility=gamma, shift=shifts, I0=3.0)

    if algorithm.startswith("n-step"):
        rec, mod = psi_n_step(frames, shifts)
    else:
        func = {"4-step": psi_4_step, "carre": psi_carre,
                "hariharan": psi_hariharan}[algorithm]
        rec, mod = func(frames)

    assert _max_error(rec, phase, mask) < 1E-10
    assert np.allclose(mod[mask], gamma)

    # after unwrapping, the phase itself (up to piston) is recovered:
    unwrapped = unwrap_ls(np.where(mask, rec, np.nan), mask=mask)
    diff = (unwrapped - phase)[mask]
    assert np.max(np.abs(diff - np.mean(diff))) < 1E-8


def test_n_step_defaults_and_bias(zernike):
    phase, mask = zernike
    frames = interferogram(phase, visibility=0.5, shift=phase_shifts(6), I0=2.0)
    rec, mod, bias = psi_n_step(frames, return_bias=True)
    assert _max_error(rec, phase, mask) < 1E-12
    assert np.allclose(bias, 2.0)
    assert np.allclose(mod, 0.5)

    with pytest.raises(ValueError):
        psi_n_step(frames[..., :2])
    with pytest.raises(ValueError):
        psi_n_step(frames[..., :3], [0.0, 2*np.pi, 4*np.pi])  # rank 1


def test_four_step_equals_least_squares(zernike):
    phase, mask = zernike
    frames = interferogram(phase, visibility=0.8, shift=phase_shifts(4))
    frames = CameraNoiser("gaussian", sigma=0.02, rng=1)(frames)
    p4, m4 = psi_4_step(frames)
    pn, mn = psi_n_step(frames, phase_shifts(4))
    assert np.allclose(p4, pn)
    assert np.allclose(m4, mn)


def test_wrong_number_of_frames(zernike):
    phase, _ = zernike
    frames = interferogram(phase, shift=phase_shifts(5))
    for func in (psi_4_step, psi_carre):
        with pytest.raises(ValueError):
            func(frames)
    with pytest.raises(ValueError):
        psi_hariharan(frames[..., :4])


@pytest.mark.parametrize("alpha", [0.8, 1.3, np.pi/2, 2.0])
def test_carre_with_unknown_step(zernike, alpha):
    phase, mask = zernike
    frames = interferogram(phase, visibility=0.7, I0=1.5,
                           shift=phase_shifts(4, alpha, -1.5*alpha))
    rec, mod, alpha_est = psi_carre(frames, return_step=True)
    assert alpha_est == pytest.approx(alpha, rel=1E-10)
    assert _max_error(rec, phase, mask) < 1E-10
    assert np.allclose(mod[mask], 0.7)


def test_hariharan_step_estimate(zernike):
    phase, _ = zernike
    step = np.pi/2*1.07
    frames = interferogram(phase, shift=phase_shifts(5, step, -2*step))
    alpha = psi_hariharan(frames, return_step=True)[2]
    assert alpha == pytest.approx(step, rel=1E-8)


def test_hariharan_insensitive_to_linear_shift_error():
    """The phase error of the four-bucket algorithm grows linearly with the
    calibration error, Hariharan's only quadratically."""

    eps = np.array([0.01, 0.02, 0.04])
    err_4 = shift_error_sensitivity("4-step", eps)
    err_h = shift_error_sensitivity("hariharan", eps)

    assert np.allclose(err_4/eps, err_4[0]/eps[0], rtol=0.02)
    assert np.allclose(err_h/eps**2, err_h[0]/eps[0]**2, rtol=0.02)
    assert np.all(err_h < err_4/20)
    assert shift_error_sensitivity("4-step", [0.0])[0] < 1E-12
    assert shift_error_sensitivity("hariharan", [0.0])[0] < 1E-12


def test_carre_immune_to_linear_shift_error():
    err = shift_error_sensitivity("carre", [-0.1, 0.05, 0.2])
    assert np.all(err < 1E-10)


def test_n_step_error_sensitivity_callable():
    eps = [0.03]
    by_name = shift_error_sensitivity("n-step", eps, N=4)
    by_func = shift_error_sensitivity(psi_4_step, eps, N=4, step=np.pi/2)
    assert by_name == pytest.approx(by_func)


def test_psi_with_noise(zernike):
    phase, mask = zernike
    photons = 1E4
    noiser = CameraNoiser("poisson", rng=0)
    frames = interferogram(phase, visibility=0.9, shift=phase_shifts(4),
                           I0=photons, noiser=noiser)
    rec, mod = psi_4_step(frames)
    err = wrap(rec - phase)[mask]
    # shot noise: sigma_phi = 1/(gamma*sqrt(N*I0/2)) for N buckets
    expected = 1/(0.9*np.sqrt(4*photons/2))
    assert np.std(err) == pytest.approx(expected, rel=0.15)


def test_takeda_periodic_phase():
    """A phase that is periodic on the grid with a carrier on the grid is
    recovered almost exactly."""

    x, y, X, Y = grid2d(-1, 1, -1, 1, 256)
    L = 2.0
    phase = 0.8*np.sin(2*np.pi*2*X/L)*np.cos(2*np.pi*Y/L)
    carrier = (2*np.pi*20/L, 2*np.pi*4/L)
    I = interferogram(phase, x, y, visibility=0.6, carrier=carrier, I0=2.0)

    rec, mod = fourier_fringe_analysis(I, x, y, carrier=carrier)
    assert np.max(np.abs(wrap(rec - phase))) < 1E-3
    assert np.max(np.abs(mod - 0.6)) < 1E-3

    rec_c, _ = fourier_fringe_analysis(I, x, y, carrier=carrier,
                                       remove_carrier=False)
    assert np.max(np.abs(wrap(rec_c - phase - carrier[0]*X - carrier[1]*Y))) < 1E-3


def test_takeda_zernike_phase():
    x, y, X, Y = grid2d(-1, 1, -1, 1, 256)
    phase, _ = zernike_phase(x, y, [0, 0, 0, 1.0, 0.3], 1.5)
    carrier = (2*np.pi*20/2, 0.0)
    I = interferogram(phase, x, y, visibility=0.8, carrier=carrier)

    rec, mod = fourier_fringe_analysis(I, x, y, carrier=carrier)
    inner = (np.abs(X) < 0.7) & (np.abs(Y) < 0.7)
    assert np.max(np.abs(wrap(rec - phase))[inner]) < 0.06
    assert np.median(mod[inner]) == pytest.approx(0.8, abs=0.01)

    unwrapped = unwrap_ls(rec)
    diff = (unwrapped - phase)[inner]
    assert np.std(diff) < 0.02


def test_find_carrier():
    x, y, X, Y = grid2d(-1, 1, -1, 1, 256)
    carrier = (2*np.pi*20/2, 2*np.pi*3.3/2)
    phase, _ = zernike_phase(x, y, [0, 0, 0, 1.0], 1.5)
    I = interferogram(phase, x, y, carrier=carrier)
    found = find_carrier(I, x, y)
    assert np.allclose(found, carrier, atol=2*np.pi/2)  # within 1 sample

    rec, _ = fourier_fringe_analysis(I, x, y)
    inner = (np.abs(X) < 0.7) & (np.abs(Y) < 0.7)
    # a residual tilt may remain; it vanishes after removing a plane fit
    unwrapped = unwrap_ls(rec)
    diff = unwrapped - phase
    A = np.stack([np.ones(inner.sum()), X[inner], Y[inner]], axis=1)
    coeffs = np.linalg.lstsq(A, diff[inner], rcond=None)[0]
    resid = diff[inner] - A @ coeffs
    assert np.std(resid) < 0.03


def test_takeda_mask():
    x, y, X, Y = grid2d(-1, 1, -1, 1, 256)
    phase, mask = zernike_phase(x, y, [0, 0, 0, 0.5, 0.2], 0.9)
    carrier = (2*np.pi*24/2, 2*np.pi*8/2)
    I = interferogram(phase, x, y, carrier=carrier)*mask

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no false overlap warning at the edge
        rec, mod = fourier_fringe_analysis(I, x, y, carrier=carrier, mask=mask)
    assert np.all(rec[~mask] == 0) and np.all(mod[~mask] == 0)
    inner = np.hypot(X, Y) < 0.6
    assert np.max(np.abs(wrap(rec - phase))[inner]) < 0.1


def test_takeda_warnings():
    x, y, X, Y = grid2d(-1, 1, -1, 1, 128)
    phase, _ = zernike_phase(x, y, [0, 0, 0, 3.0], 1.0)

    # small carrier: orders overlap
    carrier = (2*np.pi*6/2, 0.0)
    I = interferogram(phase, x, y, carrier=carrier)
    with pytest.warns(UserWarning, match="sideband power"):
        fourier_fringe_analysis(I, x, y, carrier=carrier)

    # carrier near Nyquist
    carrier = (2*np.pi*55/2, 0.0)
    I = interferogram(np.zeros_like(X), x, y, carrier=carrier)
    with pytest.warns(UserWarning, match="Nyquist"):
        fourier_fringe_analysis(I, x, y, carrier=carrier)

    with pytest.raises(ValueError):
        fourier_fringe_analysis(I, x, y, carrier=(0.0, 0.0))
