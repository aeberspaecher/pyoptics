#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the propagators against analytical results.

All lengths are given in units of the wavelength (wl = 1). The analytical
references are written out in this file so that the tests do not depend on
other parts of pyoptics being right:

- exact paraxial Gaussian beam (Fresnel regime, complex field)
- tilted plane wave (exact for transfer function propagators)
- square aperture via Fresnel integrals (Fresnel regime)
- Airy pattern of a circular aperture (Fraunhofer regime and lens focus)
- on-axis intensity behind a circular aperture (Rayleigh-Sommerfeld, near
  field), see pyoptics.analytical.mahajan_on_axis_intensity
"""

import numpy as np
import pytest
from scipy.special import fresnel, j1

from pyoptics.analytical import mahajan_on_axis_intensity
from pyoptics.beams import gauss_hermite, gauss_laguerre
from pyoptics.propagators import (fraunhofer, fresnel_IR, fresnel_TF,
                                  fresnel_rescaling, rayleigh_sommerfeld_I_DI,
                                  rayleigh_sommerfeld_I_IR,
                                  rayleigh_sommerfeld_I_TF)
from pyoptics.utils import ensure_meshgrid, grid2d


WL = 1.0
K = 2*np.pi/WL


def _grid(L, N):
    """Square periodic grid of side length L with x = 0 at index N//2."""

    return grid2d(-L/2, +L/2, -L/2, +L/2, N, assume_periodicity=True)


def _max_rel_err(actual, reference):
    return np.max(np.abs(actual - reference))/np.max(np.abs(reference))


def _gaussian_beam(X, Y, z, w0):
    """Fundamental Gaussian beam for exp(+ikz) time convention, normalized to
    1 at the waist center. Exact solution of the paraxial wave equation.
    """

    z_r = np.pi*w0**2/WL
    q = z - 1j*z_r

    return -1j*z_r/q*np.exp(1j*K*z)*np.exp(1j*K*(X**2 + Y**2)/(2*q))


def _circular_aperture(X, Y, a, supersampling=8):
    """Circular aperture of radius a. Edge pixels carry their area fraction to
    reduce the staircase error of a hard edge.
    """

    dx = X[0, 1] - X[0, 0]
    offsets = ((np.arange(supersampling) + 0.5)/supersampling - 0.5)*dx
    aperture = np.zeros(X.shape)
    for ox in offsets:
        for oy in offsets:
            aperture += ((X + ox)**2 + (Y + oy)**2 <= a**2)

    return aperture/supersampling**2


def _airy_intensity(R, a, z):
    """Fraunhofer intensity of a unit amplitude circular aperture of radius a
    at distance z (including the absolute scale).
    """

    v = K*a*R/z
    v = np.where(v == 0.0, 1E-30, v)

    return (np.pi*a**2/(WL*z))**2*(2*j1(v)/v)**2


# (propagator, number of samples, tolerance on max. relative field error)
# The impulse response methods are sampled critically (N = L**2/(wl*z)), the
# direct integration method needs dx < wl.
GAUSSIAN_CASES = [
    (fresnel_TF, 128, 1E-10),  # exact up to round-off
    (fresnel_IR, 128, 5E-3),
    (rayleigh_sommerfeld_I_TF, 128, 2E-3),  # paraxial vs. non-paraxial
    (rayleigh_sommerfeld_I_IR, 128, 2E-2),
    (rayleigh_sommerfeld_I_DI, 256, 2E-3),
]


@pytest.mark.parametrize("propagator, N, tol", GAUSSIAN_CASES,
                         ids=[c[0].__name__ for c in GAUSSIAN_CASES])
def test_gaussian_beam(propagator, N, tol):
    """Propagate a Gaussian beam by one Rayleigh length and compare the
    complex field to the analytical beam.
    """

    w0 = 10.0
    z_r = np.pi*w0**2/WL
    x, y, X, Y = _grid(200.0, N)

    field, x_new, y_new = propagator(_gaussian_beam(X, Y, 0.0, w0), x, y, z_r, WL)

    assert np.allclose(x_new, x) and np.allclose(y_new, y)
    assert _max_rel_err(field, _gaussian_beam(X, Y, z_r, w0)) < tol


@pytest.mark.parametrize("beam, indices", [(gauss_laguerre, (0, 0)),
                                           (gauss_laguerre, (3, 3)),
                                           (gauss_laguerre, (1, -2)),
                                           (gauss_hermite, (0, 0)),
                                           (gauss_hermite, (2, 1))])
@pytest.mark.parametrize("z_over_z_r", [0.5, 1.0, 2.0])
def test_higher_order_beams_fresnel(beam, indices, z_over_z_r):
    """Laguerre- and Hermite-Gaussian beams from pyoptics.beams must agree
    with their Fresnel propagated waist fields (complex field incl. Gouy phase
    and wavefront curvature).
    """

    w0 = 10.0
    z_r = np.pi*w0**2/WL
    z = z_over_z_r*z_r
    x, y, X, Y = _grid(300.0, 256)

    field_0 = beam(*indices, x, y, 0.0, w0, z_r, WL)
    field_z = beam(*indices, x, y, z, w0, z_r, WL)
    propagated, _, _ = fresnel_TF(field_0, x, y, z, WL)

    assert _max_rel_err(propagated, field_z) < 1E-8


@pytest.mark.parametrize("propagator", [rayleigh_sommerfeld_I_TF, fresnel_TF])
def test_tilted_plane_wave(propagator):
    """A plane wave with a spatial frequency on the FFT grid is an eigenmode
    of the transfer function propagators: only its phase changes, by k_z*z
    (Rayleigh-Sommerfeld) or by the paraxial approximation of it (Fresnel).
    """

    L, N, z = 64.0, 128, 37.3
    x, y, X, Y = _grid(L, N)
    f_x, f_y = 20/L, -9/L  # integer multiples of the grid frequency 1/L

    field = np.exp(2j*np.pi*(f_x*X + f_y*Y))
    propagated, _, _ = propagator(field, x, y, z, WL)

    if propagator is rayleigh_sommerfeld_I_TF:
        phase = z*np.sqrt(K**2 - (2*np.pi*f_x)**2 - (2*np.pi*f_y)**2)
    else:
        phase = K*z - np.pi*WL*z*(f_x**2 + f_y**2)

    assert _max_rel_err(propagated, field*np.exp(1j*phase)) < 1E-10


@pytest.mark.parametrize("propagator", [fresnel_TF, rayleigh_sommerfeld_I_TF])
def test_transfer_function_round_trip(propagator):
    """Propagating forth and back must return the input field for propagating
    waves, and the transfer function must conserve energy.
    """

    # dx = wl keeps all grid frequencies below 1/wl, i.e. there are no
    # evanescent waves (which would be damped and could not be undone):
    x, y, X, Y = _grid(128.0, 128)
    rng = np.random.default_rng(0)
    field = (_circular_aperture(X, Y, 40.0)
             *np.exp(1j*rng.uniform(0, 2*np.pi, X.shape)))
    forth, _, _ = propagator(field, x, y, 50.0, WL)
    back, _, _ = propagator(forth, x, y, -50.0, WL)

    assert np.isclose(np.sum(np.abs(forth)**2), np.sum(np.abs(field)**2), rtol=1E-10)
    assert _max_rel_err(back, field) < 1E-10


def _fresnel_slit(x, a, z):
    """1d Fresnel integral over a slit [-a, a], up to a factor sqrt(wl*z/2)."""

    s = np.sqrt(2/(WL*z))
    S_1, C_1 = fresnel(s*(a - x))
    S_2, C_2 = fresnel(s*(-a - x))

    return (C_1 - C_2) + 1j*(S_1 - S_2)


@pytest.mark.parametrize("propagator", [fresnel_TF, fresnel_IR])
@pytest.mark.parametrize("fresnel_number", [0.5, 1.0, 2.0])
def test_square_aperture_fresnel_integrals(propagator, fresnel_number):
    """Diffraction at a square aperture compared to the closed form solution
    in terms of Fresnel integrals.
    """

    a = 16.0  # half width of the aperture
    z = a**2/(WL*fresnel_number)
    L = 256.0
    N = int(round(L**2/(WL*z)))  # critical sampling
    x, y, X, Y = _grid(L, N)

    # aperture edges lie on the grid; weight the edge samples by 1/2:
    edge = lambda t: np.where(np.isclose(np.abs(t), a), 0.5, 1.0*(np.abs(t) < a))
    field = np.outer(edge(y), edge(x)).astype(complex)

    propagated, _, _ = propagator(field, x, y, z, WL)
    reference = np.exp(1j*K*z)/2j*np.outer(_fresnel_slit(y, a, z),
                                           _fresnel_slit(x, a, z))

    I, I_ref = np.abs(propagated)**2, np.abs(reference)**2
    window = (np.abs(X) < 2*a) & (np.abs(Y) < 2*a)
    assert np.max(np.abs(I - I_ref)[window]) < 0.03*np.max(I_ref)


@pytest.mark.parametrize("propagator", [fraunhofer, fresnel_rescaling])
def test_airy_pattern_far_field(propagator):
    """Far field of a circular aperture is the Airy pattern, including the
    absolute scale (pi*a**2/(wl*z))**2 of the central maximum.
    """

    a, z = 10.0, 1E5  # Fresnel number 1E-3
    x, y, X, Y = _grid(400.0, 512)

    propagated, x_new, y_new = propagator(_circular_aperture(X, Y, a), x, y, z, WL)
    X_new, Y_new = ensure_meshgrid(x_new, y_new)
    I_ref = _airy_intensity(np.hypot(X_new, Y_new), a, z)

    assert np.isclose(np.max(np.abs(propagated)**2), np.max(I_ref), rtol=5E-3)
    assert _max_rel_err(np.abs(propagated)**2, I_ref) < 5E-3


@pytest.mark.parametrize("propagator, tol", [(fresnel_IR, 1E-2),
                                             (fresnel_TF, 3E-2),
                                             (rayleigh_sommerfeld_I_TF, 3E-2)])
def test_airy_pattern_lens_focus(propagator, tol):
    """A circular aperture with a thin lens phase, propagated to the focal
    plane, yields the Airy pattern.
    """

    a, f = 50.0, 2500.0  # NA = 0.02, first zero at 0.61*wl/NA = 30.5
    x, y, X, Y = _grid(512.0, 512)

    field = _circular_aperture(X, Y, a)*np.exp(-1j*K*(X**2 + Y**2)/(2*f))
    propagated, _, _ = propagator(field, x, y, f, WL)
    I_ref = _airy_intensity(np.hypot(X, Y), a, f)

    assert _max_rel_err(np.abs(propagated)**2, I_ref) < tol


@pytest.mark.parametrize("propagator", [rayleigh_sommerfeld_I_TF,
                                        rayleigh_sommerfeld_I_IR,
                                        rayleigh_sommerfeld_I_DI])
def test_on_axis_intensity_circular_aperture(propagator):
    """Near field on-axis intensity behind a circular aperture (radius 10 wl)
    compared to the exact Rayleigh-Sommerfeld result. The intensity oscillates
    between 0 and 4 in this range; the Fresnel approximation fails here.
    """

    a = 10.0
    x, y, X, Y = _grid(128.0, 512)  # dx = wl/4
    center = len(x)//2
    field = _circular_aperture(X, Y, a).astype(complex)

    z_vals = np.array([15.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0])
    I = np.array([np.abs(propagator(field, x, y, z, WL)[0][center, center])**2
                  for z in z_vals])
    I_ref = mahajan_on_axis_intensity(a, 0.0, 1.0, WL, z_vals)

    assert np.max(np.abs(I - I_ref)) < 0.1
