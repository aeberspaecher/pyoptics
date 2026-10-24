#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the vectorial propagators.

All lengths are in units of the wavelength (wl = 1). References:

- tilted plane wave: E_z = -k_x/k_z*E_x, propagation phase exp(i*k_z*z)
- paraxial Gaussian beam: first order longitudinal field E_z = (i/k) dE_x/dx
- div(E) = 0 and E perpendicular to k for every plane wave
- Richards-Wolf integrals (scipy.integrate.quad) for the focus of an
  aplanatic high NA lens
"""

import functools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from pyoptics.fft import FT, inv_FT
from pyoptics.propagators import (aplanatic_rotation, fraunhofer, fresnel_TF,
                                  longitudinal_component, propagate_padded,
                                  propagate_vectorial, rayleigh_sommerfeld_I_TF,
                                  vectorial_fresnel, vectorial_mansuripur,
                                  vectorial_rayleigh_sommerfeld_TF)
from pyoptics.utils import freq_grid, grid1d


WL = 1.0
K = 2*np.pi/WL


def _grid(L, N):
    x = grid1d(-L/2, L/2, N)

    return x, x, *np.meshgrid(x, x)


def _stack(E_x, E_y):
    return np.stack([E_x, E_y], axis=-1)


def _divergence_in_k_space(E, x, y, n=1.0):
    """Return max |k.E(k)| and max |k||E(k)| of a vector field E, leaving
    out the near grazing waves, for which longitudinal_component() bounds
    1/k_z."""

    k = K*n
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    K_z = np.sqrt((k**2 - K_x**2 - K_y**2).astype(complex))
    E_k = [FT(E[:, :, i]) for i in range(3)]
    div = K_x*E_k[0] + K_y*E_k[1] + K_z*E_k[2]
    dk = 2*np.pi/(len(x)*(x[1] - x[0]))
    div[np.abs(K_z) < np.sqrt(k*dk)] = 0.0

    return np.abs(div).max(), k*max(np.abs(E_k[i]).max() for i in range(3))


@pytest.mark.parametrize("n", [1.0, 1.5])
def test_longitudinal_component_tilted_plane_wave(n):
    x, y, X, Y = _grid(20.0, 64)
    k_x, k_y = 2*np.pi*7/20, -2*np.pi*4/20  # on the frequency grid
    k = K*n
    k_z = np.sqrt(k**2 - k_x**2 - k_y**2)
    E_x = 0.3*np.exp(1j*(k_x*X + k_y*Y))
    E_y = (0.5 - 0.2j)*np.exp(1j*(k_x*X + k_y*Y))

    E_z = longitudinal_component(E_x, E_y, x, y, WL, n)

    assert np.allclose(E_z, -(k_x*E_x + k_y*E_y)/k_z)


def test_longitudinal_component_bounds_grazing_waves():
    """A plane wave sampled exactly at |k_perp| = k (k_z = 0) gets a finite
    E_z of the order of sqrt(k/dk)*E_x, a slightly steeper one is
    evanescent."""

    L, N = 20.0, 64
    x, y, X, Y = _grid(L, N)
    k_x = K  # 20 periods in the window: on the grid
    E_x = np.exp(1j*k_x*X)

    E_z = longitudinal_component(E_x, 0*E_x, x, y, WL)

    dk = 2*np.pi/L
    assert np.all(np.isfinite(E_z))
    assert np.allclose(np.abs(E_z), np.sqrt(K/dk))


def test_vectorial_RS_tilted_plane_wave():
    x, y, X, Y = _grid(20.0, 64)
    k_x = 2*np.pi*8/20
    k_z = np.sqrt(K**2 - k_x**2)
    E_x = np.exp(1j*k_x*X)
    z = 3.7

    E, x_out, y_out = vectorial_rayleigh_sommerfeld_TF(_stack(E_x, 0*E_x), x, y, z, WL)

    assert E.shape == (64, 64, 3)
    assert x_out is x and y_out is y
    assert np.allclose(E[:, :, 0], E_x*np.exp(1j*k_z*z))
    assert np.allclose(E[:, :, 1], 0)
    assert np.allclose(E[:, :, 2], -k_x/k_z*E_x*np.exp(1j*k_z*z))


def test_vectorial_RS_keeps_transverse_field_at_z0_and_ignores_given_E_z():
    x, y, X, Y = _grid(16.0, 64)
    E_x = np.exp(-(X**2 + Y**2)/2**2)
    E_y = 0.5j*np.exp(-((X - 1)**2 + Y**2)/1.5**2)

    E2, _, _ = vectorial_rayleigh_sommerfeld_TF(_stack(E_x, E_y), x, y, 0.0, WL)
    E3, _, _ = vectorial_rayleigh_sommerfeld_TF(np.stack([E_x, E_y, 7 + 0*E_x], axis=-1),
                                                x, y, 0.0, WL)

    assert np.allclose(E2[:, :, 0], E_x) and np.allclose(E2[:, :, 1], E_y)
    assert np.allclose(E2, E3)


@pytest.mark.parametrize("z", [0.0, 2.0, 10.0])
def test_vectorial_RS_is_divergence_free(z):
    """A tightly focused field (waist of one wavelength) has a strong E_z; it
    must be such that k.E = 0 for all propagating waves."""

    x, y, X, Y = _grid(24.0, 128)
    E_x = np.exp(-(X**2 + Y**2)/1.0**2)
    E_y = np.exp(-((X - 0.5)**2 + Y**2)/0.8**2)

    E, _, _ = vectorial_rayleigh_sommerfeld_TF(_stack(E_x, E_y), x, y, z, WL)
    div, scale = _divergence_in_k_space(E, x, y)

    assert div < 1E-10*scale
    if z == 0:
        assert np.abs(E[:, :, 2]).max() > 0.2*np.abs(E[:, :, 0]).max()


def test_vectorial_RS_transverse_components_equal_scalar_propagation():
    x, y, X, Y = _grid(24.0, 128)
    E_x = np.exp(-(X**2 + Y**2)/1.5**2)
    E_y = 1j*X*E_x
    z = 5.0

    E, _, _ = vectorial_rayleigh_sommerfeld_TF(_stack(E_x, E_y), x, y, z, WL, n=1.3)
    E_x_scalar = rayleigh_sommerfeld_I_TF(E_x, x, y, z, WL, n=1.3)[0]
    E_y_scalar = rayleigh_sommerfeld_I_TF(E_y, x, y, z, WL, n=1.3)[0]

    assert np.allclose(E[:, :, 0], E_x_scalar)
    assert np.allclose(E[:, :, 1], E_y_scalar)


def test_longitudinal_field_of_paraxial_gaussian():
    """For an x polarized Gaussian waist, E_z = (i/k) dE_x/dx to first order
    in 1/(k*w_0)."""

    x, y, X, Y = _grid(40.0, 256)
    w_0 = 4.0
    E_x = np.exp(-(X**2 + Y**2)/w_0**2)

    E, _, _ = vectorial_rayleigh_sommerfeld_TF(_stack(E_x, 0*E_x), x, y, 0.0, WL)
    E_z_paraxial = 1j/K*(-2*X/w_0**2)*E_x

    err = np.abs(E[:, :, 2] - E_z_paraxial).max()/np.abs(E_z_paraxial).max()
    assert err < 10/(K*w_0)**2  # next order


def test_propagate_vectorial_with_fresnel_close_to_exact_for_paraxial_beam():
    x, y, X, Y = _grid(80.0, 256)
    w_0 = 6.0
    E_x = np.exp(-(X**2 + Y**2)/w_0**2)
    field = _stack(E_x, 0.5*E_x)
    z = 50.0

    E_fresnel, _, _ = propagate_vectorial(fresnel_TF, field, x, y, z, WL)
    E_exact, _, _ = vectorial_rayleigh_sommerfeld_TF(field, x, y, z, WL)

    assert np.abs(E_fresnel - E_exact).max() < 1E-2*np.abs(E_exact).max()
    assert np.abs(E_fresnel[:, :, 2]).max() > 0.01*np.abs(E_fresnel[:, :, 0]).max()


def test_propagate_vectorial_fraunhofer_far_field_is_transverse():
    x, y, X, Y = _grid(20.0, 128)
    E_x = np.where(X**2 + Y**2 < 2.0**2, 1.0, 0.0)
    E_y = 0.3*E_x
    z = 1E4

    E, x_ff, y_ff = propagate_vectorial(fraunhofer, _stack(E_x, E_y), x, y, z, WL)
    X_ff, Y_ff = np.meshgrid(x_ff, y_ff)
    r_dot_E = X_ff*E[:, :, 0] + Y_ff*E[:, :, 1] + z*E[:, :, 2]

    assert np.allclose(E[:, :, 0], fraunhofer(E_x, x, y, z, WL)[0])
    assert np.abs(r_dot_E).max() < 1E-10*z*np.abs(E).max()


def test_propagate_padded_accepts_vector_fields():
    x, y, X, Y = _grid(16.0, 64)
    E_x = np.where((np.abs(X) < 3) & (np.abs(Y) < 3), 1.0, 0.0)
    field = _stack(E_x, 1j*E_x)
    z = 20.0

    E_pad, x_out, _ = propagate_padded(vectorial_rayleigh_sommerfeld_TF, field, x, y, z, WL,
                                       pad_factor=3)
    E_x_pad = propagate_padded(rayleigh_sommerfeld_I_TF, E_x, x, y, z, WL, pad_factor=3)[0]

    assert E_pad.shape == (64, 64, 3) and x_out is x
    assert np.allclose(E_pad[:, :, 0], E_x_pad)
    assert np.allclose(E_pad[:, :, 1], 1j*E_x_pad)
    # wrap-around makes the unpadded result differ:
    E_cyclic = vectorial_rayleigh_sommerfeld_TF(field, x, y, z, WL)[0]
    assert np.abs(E_cyclic - E_pad).max() > 0.1*np.abs(E_pad).max()

    E_pv, _, _ = propagate_padded(functools.partial(propagate_vectorial, fresnel_TF),
                                  field, x, y, z, WL, pad_factor=3)
    assert E_pv.shape == (64, 64, 3)


def test_vectorial_propagators_reject_scalar_fields():
    x, y, X, Y = _grid(8.0, 32)
    with pytest.raises(ValueError):
        vectorial_rayleigh_sommerfeld_TF(X, x, y, 1.0, WL)
    with pytest.raises(ValueError):
        vectorial_mansuripur(np.stack([X, X, X], axis=-1), x, y, 1.0, WL)


# aplanatic rotation and Mansuripur's method:

def test_aplanatic_rotation_is_orthonormal_and_transverse():
    rng = np.random.default_rng(1)
    r, phi = np.sqrt(rng.uniform(0, 1, 50)), rng.uniform(0, 2*np.pi, 50)
    s_x, s_y = r*np.cos(phi), r*np.sin(phi)
    s_z = np.sqrt(1 - s_x**2 - s_y**2)

    R = aplanatic_rotation(s_x, s_y)
    gram = np.einsum("ian,ibn->abn", R, R)
    s_dot_R = s_x*R[0] + s_y*R[1] + s_z*R[2]

    assert R.shape == (3, 2, 50)
    assert np.allclose(gram, np.eye(2)[:, :, None])
    assert np.allclose(s_dot_R, 0)
    assert np.allclose(aplanatic_rotation(0.0, 0.0), [[1, 0], [0, 1], [0, 0]])


def test_aplanatic_rotation_keeps_s_polarization():
    """In the meridional plane x-z an y polarized ray is s polarized and
    stays y polarized; the x polarized (p) one tilts with the ray."""

    theta = 0.8
    R = aplanatic_rotation(np.sin(theta), 0.0)

    assert np.allclose(R[:, 1], [0, 1, 0])
    assert np.allclose(R[:, 0], [np.cos(theta), 0, -np.sin(theta)])


def _richards_wolf_x_polarized(x, y, z, NA, pupil):
    """Focal field of an x polarized pupil pupil(sin(theta)) of an aplanatic
    lens, up to a constant factor (Richards and Wolf 1959)."""

    k = K
    rho, phi = np.hypot(x, y), np.arctan2(y, x)

    def integral(f):
        g = lambda t: f(t)*np.sqrt(np.cos(t))*np.sin(t)*pupil(np.sin(t))*np.exp(1j*k*z*np.cos(t))
        args = dict(a=0, b=np.arcsin(NA), limit=200)
        return quad(lambda t: g(t).real, **args)[0] + 1j*quad(lambda t: g(t).imag, **args)[0]

    I0 = integral(lambda t: (1 + np.cos(t))*jv(0, k*rho*np.sin(t)))
    I1 = integral(lambda t: np.sin(t)*jv(1, k*rho*np.sin(t)))
    I2 = integral(lambda t: (1 - np.cos(t))*jv(2, k*rho*np.sin(t)))

    return np.array([I0 + I2*np.cos(2*phi), I2*np.sin(2*phi), -2j*I1*np.cos(phi)])


def test_mansuripur_focus_equals_richards_wolf():
    """A smooth (Gaussian) pupil up to NA 0.98 avoids the sampling error of
    the pupil edge, so the agreement is limited by the quadrature only."""

    L, N = 40.0, 256
    x, y, X, Y = _grid(L, N)
    NA, s_0 = 0.98, 0.4
    pupil = lambda s: np.exp(-s**2/s_0**2)
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    s = np.hypot(K_x, K_y)/K
    P = np.where(s < NA, pupil(s), 0.0)
    offsets = [(0, 0), (0, 2), (2, 0), (3, 3), (1, 4), (-2, 5), (6, 1), (-3, -4)]
    idx = [(N//2 + i, N//2 + j) for i, j in offsets]

    scales = []
    for z in (0.0, 0.7):
        E, _, _ = vectorial_mansuripur(_stack(inv_FT(P), 0*P), x, y, z, WL)
        num = np.array([E[i, j] for i, j in idx])
        ref = np.array([_richards_wolf_x_polarized(X[i, j], Y[i, j], z, NA, pupil)
                        for i, j in idx])
        scale = np.vdot(ref.ravel(), num.ravel())/np.vdot(ref.ravel(), ref.ravel())
        assert np.abs(num - scale*ref).max() < 1E-4*np.abs(num).max()
        scales.append(scale)

    # same constant in all planes, i.e. no spurious z dependent phase:
    assert np.isclose(scales[0], scales[1], rtol=1E-5)


def test_mansuripur_conserves_power_and_is_transverse():
    x, y, X, Y = _grid(30.0, 128)
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    s2 = (K_x**2 + K_y**2)/K**2
    P = np.where(s2 < 0.9**2, 1.0, 0.0)
    field = _stack(inv_FT(P), inv_FT(0.5j*P))

    E, _, _ = vectorial_mansuripur(field, x, y, 1.3, WL)

    # power through the plane: sum over plane waves of |E(k)|**2*k_z/k
    s_z = np.sqrt(np.clip(1 - s2, 0, None))
    power_out = sum(np.sum(np.abs(FT(E[:, :, i]))**2*s_z) for i in range(3))
    power_in = sum(np.sum(np.abs(FT(field[:, :, i]))**2) for i in range(2))
    assert np.isclose(power_out, power_in)

    div, scale = _divergence_in_k_space(E, x, y)
    assert div < 1E-10*scale


def test_mansuripur_paraxial_limit_is_scalar_propagation():
    x, y, X, Y = _grid(200.0, 256)
    w_0 = 15.0
    E_x = np.exp(-(X**2 + Y**2)/w_0**2)
    z = 40.0

    E, _, _ = vectorial_mansuripur(_stack(E_x, 0*E_x), x, y, z, WL)
    E_scalar = rayleigh_sommerfeld_I_TF(E_x, x, y, z, WL)[0]

    assert np.abs(E[:, :, 0] - E_scalar).max() < 1E-3
    assert np.abs(E[:, :, 1]).max() < 1E-4
    assert np.abs(E[:, :, 2]).max() < 0.05


def test_vectorial_fresnel_is_deprecated_alias():
    x, y, X, Y = _grid(8.0, 32)
    field = _stack(np.exp(-X**2 - Y**2), 0*X)

    with pytest.warns(DeprecationWarning):
        E_old, _, _ = vectorial_fresnel(field, x, y, 0.5, WL)

    assert np.allclose(E_old, vectorial_mansuripur(field, x, y, 0.5, WL)[0])
