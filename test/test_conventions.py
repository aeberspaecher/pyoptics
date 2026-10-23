#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Pin the phase convention used throughout pyoptics.

Fields vary as exp(i(k.r - omega*t)): the time dependence exp(-i*omega*t) is
implicit, and propagation in +z adds the phase +k*z. Absorbing media have
n + i*kappa with kappa > 0. All lengths in units of the wavelength.
"""

import numpy as np
import pytest

from pyoptics.beams import gauss_hermite, gauss_laguerre, plane_wave, spherical_wave
from pyoptics.fft import FT
from pyoptics.interfaces import T_p, T_s, refracted_angle, r_s
from pyoptics.masks import thin_lens
from pyoptics.processing import Shifter
from pyoptics.propagators import (fraunhofer, fresnel_TF,
                                  fresnel_rescaling, rayleigh_sommerfeld_I_TF)
from pyoptics.thin_films import r_t_coeffs
from pyoptics.TIE import tie
from pyoptics.utils import grid2d, k_z


WL = 1.0
K = 2*np.pi/WL


def _grid(L, N):
    return grid2d(-L/2, +L/2, -L/2, +L/2, N, assume_periodicity=True)


def test_FT_kernel():
    """FT() maps the plane wave exp(+i k_x x) to the frequency +k_x."""

    L, N = 32.0, 64
    x, y, X, Y = _grid(L, N)
    m = 5
    spectrum = FT(np.exp(2j*np.pi*m/L*X))

    i, j = np.unravel_index(np.argmax(np.abs(spectrum)), spectrum.shape)
    assert i == N//2 and j == N//2 + m


@pytest.mark.parametrize("propagator", [rayleigh_sommerfeld_I_TF, fresnel_TF])
def test_plane_wave_gains_plus_k_dz(propagator):
    L, N, dz = 32.0, 64, 0.3
    x, y, X, Y = _grid(L, N)
    field = np.ones([N, N], dtype=complex)

    propagated, _, _ = propagator(field, x, y, dz, WL)
    center = propagated[N//2, N//2]

    assert np.isclose(np.angle(center), K*dz)


def test_plane_wave_definition():
    x = y = np.linspace(-1, 1, 5)
    E = plane_wave(1.0, [0, 0, 1], x, y, 0.25, WL)

    assert np.allclose(E, np.exp(1j*K*0.25))


def test_evanescent_waves_decay():
    k_x = np.array([2*K])
    assert np.imag(k_z(K, k_x, 0.0)) > 0

    L, N, dz = 8.0, 64, 0.5
    x, y, X, Y = _grid(L, N)
    f_x = 16/L  # 2/WL: evanescent
    field = np.exp(2j*np.pi*f_x*X)
    propagated, _, _ = rayleigh_sommerfeld_I_TF(field, x, y, dz, WL)

    decay = np.exp(-dz*np.sqrt((2*np.pi*f_x)**2 - K**2))
    assert np.allclose(np.abs(propagated), decay)


def test_absorbing_medium_decays():
    """n + i*kappa with kappa > 0 attenuates a wave travelling in +z."""

    L, N, dz, kappa = 16.0, 32, 2.0, 0.05
    x, y, X, Y = _grid(L, N)
    field = np.ones([N, N], dtype=complex)

    propagated, _, _ = rayleigh_sommerfeld_I_TF(field, x, y, dz, WL, n=1.5 + 1j*kappa)

    assert np.allclose(propagated, np.exp(1j*K*(1.5 + 1j*kappa)*dz))


def test_thin_film_phase_and_absorption():
    """A layer in a medium of the same absorbing index n + i*kappa transmits
    exp(i k_0 (n + i kappa) d).
    """

    n, kappa, d = 1.5, 0.01, 2.3
    r, t = r_t_coeffs(0.0, [n], [kappa], [d], n, kappa, n, kappa, "s", WL)

    assert np.isclose(r, 0.0)
    assert np.isclose(t, np.exp(1j*K*(n + 1j*kappa)*d))


def test_spherical_wave_diverges_with_plus_ikr():
    x = y = np.array([0.0])
    z = 7.3
    U = spherical_wave(x, y, z, WL, 0.0, 0.0, 0.0)

    assert np.isclose(U, np.exp(1j*K*z)/z)


def test_thin_lens_phase():
    x = y = np.linspace(-2, 2, 9)
    f = 50.0
    X, Y = np.meshgrid(x, y)

    assert np.allclose(thin_lens(x, y, f, WL), np.exp(-1j*K*(X**2 + Y**2)/(2*f)))


def test_converging_lens_focuses_forward():
    """A plane wave behind thin_lens() is focused at +f (and not at -f)."""

    L, N, a, f = 128.0, 256, 20.0, 400.0
    x, y, X, Y = _grid(L, N)
    field = (X**2 + Y**2 <= a**2)*thin_lens(x, y, f, WL)

    I_forward = np.abs(fresnel_TF(field, x, y, f, WL)[0][N//2, N//2])**2
    I_backward = np.abs(fresnel_TF(field, x, y, -f, WL)[0][N//2, N//2])**2

    assert I_forward > 100*I_backward


@pytest.mark.parametrize("beam, indices, N_mode", [(gauss_hermite, (0, 0), 0),
                                                   (gauss_hermite, (2, 2), 4),
                                                   (gauss_laguerre, (1, 0), 2)])
def test_beam_on_axis_phase(beam, indices, N_mode):
    """On axis, a beam carries the phase k z - (N + 1) arctan(z/z_r)."""

    w0 = 5.0
    z_r = np.pi*w0**2/WL
    z = 0.37*z_r
    x = y = np.array([0.0])

    U0 = beam(*indices, x, y, 0.0, w0, z_r, WL)
    Uz = beam(*indices, x, y, z, w0, z_r, WL)
    dphase = np.angle(Uz/U0)
    expected = K*z - (N_mode + 1)*np.arctan(z/z_r)

    assert np.isclose(np.angle(np.exp(1j*(dphase - expected))).item(), 0.0)


def test_beam_wavefront_diverges_behind_waist():
    """Behind the waist (z > 0), the phase grows as +k r**2/(2 R) with R > 0;
    in front of it (z < 0), the wavefront converges.
    """

    w0 = 5.0
    z_r = np.pi*w0**2/WL
    x = np.array([0.0, 2.0])
    y = np.array([0.0])

    for z, sign in [(z_r, +1), (-z_r, -1)]:
        U = gauss_hermite(0, 0, x, y, z, w0, z_r, WL)
        dphase = np.angle(U[0, 1]/U[0, 0])
        R = z*(1 + (z_r/z)**2)
        assert np.isclose(dphase, K*2.0**2/(2*R))
        assert np.sign(dphase) == sign


def test_laguerre_gauss_helical_phase():
    """LG modes with l > 0 have the phase +l*phi."""

    l = 1
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    U = gauss_laguerre(0, l, x, y, 0.0, 3.0, np.pi*9.0/WL, WL)

    # U[0, 0]: (x, y) = (1, 0), phi = 0; U[1, 1]: (x, y) = (0, 1), phi = pi/2
    assert np.isclose(np.exp(1j*np.angle(U[1, 1]/U[0, 0])), np.exp(1j*l*np.pi/2))


@pytest.mark.parametrize("propagator", [fraunhofer, fresnel_rescaling])
def test_tilted_plane_wave_goes_to_plus_x(propagator):
    """A plane wave exp(+i k_x x) travels towards +x: the far field peaks at
    x = wl*z*k_x/(2 pi).
    """

    L, N, z = 64.0, 128, 1E4
    x, y, X, Y = _grid(L, N)
    f_x = 6/L
    field = np.exp(-(X**2 + Y**2)/(2*8.0**2))*np.exp(2j*np.pi*f_x*X)

    U, x_new, y_new = propagator(field, x, y, z, WL)
    i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)

    assert np.isclose(y_new[i], 0.0)
    assert np.isclose(x_new[j], WL*z*f_x)


def test_TIR_evanescent_wave_decays():
    n_1, n_2, phi = 1.5, 1.0, 1.2
    phi_2 = refracted_angle(phi, n_1, n_2)

    assert np.imag(n_2*np.cos(phi_2)) > 0
    assert np.isclose(T_s(phi, n_1, n_2), 0.0)
    assert np.isclose(T_p(phi, n_1, n_2), 0.0)
    # exp(-i omega t): r_s = (a - i b)/(a + i b) with a, b > 0
    assert -np.pi < np.angle(r_s(phi, n_1, n_2)) < 0


def test_absorbing_substrate_branch():
    n_2 = 0.2 + 3j
    phi_2 = refracted_angle(0.5, 1.0, n_2)

    assert np.imag(n_2*np.cos(phi_2)) > 0


def test_shifter_shifts_to_plus_x():
    L, N = 32.0, 64
    x, y, X, Y = _grid(L, N)
    img = np.exp(-(X**2 + Y**2)/2.0)
    shifted = Shifter(3*L/N, 0.0, x, y).process_single_image(img)

    i, j = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert i == N//2 and j == N//2 + 3


def test_tie_converging_wave_gains_intensity():
    """k dI/dz = -div(I grad(phase)): a converging wave (phase maximal on
    axis, like the lens phase -k r**2/(2f)) increases the on-axis intensity.
    """

    L, N, a, s = 64.0, 128, 3.0, 5.0
    x, y, X, Y = _grid(L, N)
    I = np.exp(-(X**2 + Y**2)/(2*8.0**2))
    phase = a*np.exp(-(X**2 + Y**2)/(2*s**2))  # ~ a - a r**2/(2 s**2)

    dI_dz = tie(phase, I, x, y, WL)

    # on axis, -div(I grad(phase))/k = -I(0)*laplace(phase)(0)/k = 2a/(k s**2)
    assert np.isclose(dI_dz[N//2, N//2], 2*a/(K*s**2), rtol=1E-6)


def test_wpm_tilted_plane_wave():
    """WPM in a homogeneous medium: exp(i k_x x) -> exp(i (k_x x + k_z dz))."""

    wpm = pytest.importorskip("pyoptics.wpm")

    L, N, dz = 16.0, 32, 2.0
    x, y, X, Y = _grid(L, N)
    f_x = 3/L
    field = np.exp(2j*np.pi*f_x*X)

    propagated = wpm.wpm_propagate(lambda X, Y, z: np.ones_like(X), field, x, y, 0.0, dz, WL)
    kz = np.sqrt(K**2 - (2*np.pi*f_x)**2)

    assert np.allclose(propagated, field*np.exp(1j*kz*dz))
