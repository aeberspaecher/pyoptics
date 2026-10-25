#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the propagation onto tilted planes (pyoptics.tilted).

All lengths are in units of the wavelength (wl = 1). References:

- the angular spectrum propagator rayleigh_sommerfeld_I_TF() for parallel
  planes (theta = 0, and slice by slice for tilted planes)
- plane waves exp(i*k.r), evaluated at the 3D points of the tilted plane
- the paraxial Gaussian beam of abcd.gaussian_field() (with the Gouy phase)
  at the 3D points of the tilted plane
- the power flux, which is the same on every plane
"""

import numpy as np
import pytest

from pyoptics import tilted
from pyoptics.abcd import gaussian_field
from pyoptics.geometry import R_x, R_y, R_z
from pyoptics.propagators import (propagate_padded, rayleigh_sommerfeld_I_TF,
                                  vectorial_rayleigh_sommerfeld_TF)
from pyoptics.tilted import (power_flux, rotate_angular_spectrum, rotation_from_angles,
                             tilted_beam_widths, tilted_plane, vectorial_tilted_plane)
from pyoptics.utils import frequencies, grid1d


WL = 1.0
K = 2*np.pi/WL


def _grid(L, N):
    x = grid1d(-L/2, L/2, N)
    return x, x.copy()


def _gaussian(x, y, w0, x0=0.0):
    z_R = np.pi*w0**2/WL
    return gaussian_field(x, y, 1j*z_R, WL, x0=x0)


def _gaussian_3d(P, w0):
    """Paraxial Gaussian beam with its waist at z = 0, at the points P = (x, y, z)."""

    z_R = np.pi*w0**2/WL
    q = P[2] + 1j*z_R
    return gaussian_field(P[0], P[1], q, WL)*np.exp(1j*K*P[2])*np.exp(-1j*np.arctan(P[2]/z_R))


def _plane_points(r0, R, x_t, y_t):
    X, Y = np.meshgrid(x_t, y_t)
    return np.asarray(r0)[:, None, None] + R[:, 0, None, None]*X + R[:, 1, None, None]*Y


def _rel_err(a, b):
    return np.max(np.abs(a - b))/np.max(np.abs(b))


@pytest.fixture(scope="module")
def beam():
    x, y = _grid(96.0, 128)
    return _gaussian(x, y, 5.0), x, y


@pytest.mark.parametrize("method", ["exact", "rotation"])
def test_untilted_equals_angular_spectrum(beam, method):
    u, x, y = beam
    u_t, x_t, y_t, c = tilted_plane(u, x, y, 30.0, WL, method=method, pad=1)
    reference = rayleigh_sommerfeld_I_TF(u, x, y, 30.0, WL)[0]
    assert _rel_err(u_t, reference) < 1E-10
    assert np.allclose(c, 0, atol=1E-12)
    assert np.array_equal(x_t, x) and np.array_equal(y_t, y)

    # with padding, the padded angular spectrum method:
    u_t = tilted_plane(u, x, y, 30.0, WL, method=method, pad=2)[0]
    reference = propagate_padded(rayleigh_sommerfeld_I_TF, u, x, y, 30.0, WL, pad_factor=2)[0]
    assert _rel_err(u_t, reference) < 1E-10


@pytest.mark.parametrize("R", [R_y(0.4), R_x(-0.6), R_z(0.3) @ R_y(0.5) @ R_x(0.2)])
def test_plane_wave(R):
    # an infinite plane wave (no padding) on the grid frequencies:
    x, y = _grid(32.0, 32)
    dk = 2*np.pi/32.0
    k_in = np.array([5*dk, -3*dk, 0.0])
    k_in[2] = np.sqrt(K**2 - k_in[0]**2 - k_in[1]**2)
    X, Y = np.meshgrid(x, y)
    u = np.exp(1j*(k_in[0]*X + k_in[1]*Y))
    r0 = np.array([1.0, -2.0, 25.0])

    u_t, x_t, y_t, c = tilted_plane(u, x, y, r0, WL, R=R, method="exact", pad=1,
                                    remove_carrier=False)
    reference = np.exp(1j*np.einsum("i,imn->mn", k_in, _plane_points(r0, R, x_t, y_t)))
    assert _rel_err(u_t, reference) < 1E-10
    assert np.allclose(c, 0)

    # the carrier is the rotated wave vector, and the rest is a constant phase:
    u_t, _, _, c = tilted_plane(u, x, y, r0, WL, R=R, method="exact", pad=1)
    assert np.allclose(c, np.dot(R.T, k_in)[:2])
    assert np.allclose(u_t, np.exp(1j*np.dot(k_in, r0)))


@pytest.mark.parametrize("method", ["exact", "rotation"])
@pytest.mark.parametrize("theta_deg", [20, 45, 60])
def test_gaussian_beam_on_tilted_plane(method, theta_deg):
    w0, z0 = 6.0, 90.0
    x, y = _grid(96.0, 128)
    u = _gaussian(x, y, w0)
    R = R_y(np.radians(theta_deg))
    x_out = 1.5*x   # the footprint stretches by 1/cos(theta)

    u_t, x_t, y_t, c = tilted_plane(u, x, y, z0, WL, R=R, method=method, x_out=x_out,
                                    remove_carrier=False)
    reference = _gaussian_3d(_plane_points([0, 0, z0], R, x_t, y_t), w0)
    # paraxial reference: deviations of order (lambda/(pi*w0))**2
    assert _rel_err(u_t, reference) < 2E-3


@pytest.mark.parametrize("theta_deg", [30, 60])
def test_rotation_converges_to_exact(theta_deg):
    w0, z0 = 5.0, 90.0
    x, y = _grid(96.0, 128)
    u = _gaussian(x, y, w0)
    R = R_y(np.radians(theta_deg))
    x_out = grid1d(-80, 80, 160)
    exact = tilted_plane(u, x, y, z0, WL, R=R, method="exact", x_out=x_out)[0]

    errors = [_rel_err(tilted_plane(u, x, y, z0, WL, R=R, method="rotation", kind="linear",
                                    pad=p, x_out=x_out)[0], exact)
              for p in (1, 2, 4)]
    assert errors[0] < 5E-3
    assert errors[1] < errors[0]/2 and errors[2] < errors[1]/2

    cubic = tilted_plane(u, x, y, z0, WL, R=R, method="rotation", kind="cubic", x_out=x_out)[0]
    assert _rel_err(cubic, exact) < 1E-5


def test_jacobian_is_needed():
    """Rotate the spectrum of a beam onto a tilted plane through its waist:
    the field at the origin must not change. Without the Jacobian it drops
    by cos(theta)."""

    theta = np.radians(60)
    x, y = _grid(96.0, 128)
    u = _gaussian(x, y, 5.0)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
    kx, ky = frequencies(x), frequencies(y)
    common = dict(kx_out=kx - K*np.sin(theta), ky_out=ky)   # centered on the carrier
    centre = (len(y)//2, len(x)//2)
    inv = lambda F: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))

    u_with = inv(rotate_angular_spectrum(F, kx, ky, K, R_y(theta), **common))
    u_without = inv(rotate_angular_spectrum(F, kx, ky, K, R_y(theta), jacobian=False, **common))
    assert np.isclose(u_with[centre], u[centre], rtol=1E-4)
    assert np.isclose(abs(u_without[centre]/u[centre]), np.cos(theta), rtol=1E-2)


def test_rotate_angular_spectrum_identity():
    kx = np.linspace(-3, 3, 31)
    ky = np.linspace(-2, 2, 21)
    KX, KY = np.meshgrid(kx, ky)
    A = np.exp(-(KX**2 + KY**2))*np.exp(1j*KX)
    assert np.allclose(rotate_angular_spectrum(A, kx, ky, K, np.eye(3)), A)


def test_equals_slice_and_pick():
    """Method C: propagate to the parallel plane through each column of the
    tilted plane and evaluate the field there by trigonometric interpolation."""

    w0, z0, theta = 3.0, 20.0, np.radians(35)
    x, y = _grid(48.0, 64)
    u = _gaussian(x, y, w0)
    R = R_y(theta)
    u_t, x_t, y_t, _ = tilted_plane(u, x, y, z0, WL, R=R, method="exact", pad=1,
                                    remove_carrier=False)

    kx = frequencies(x)
    reference = np.empty_like(u_t)
    for p, xh in enumerate(x_t):
        x_p, z_p = R[0, 0]*xh, z0 + R[2, 0]*xh
        slice_ = rayleigh_sommerfeld_I_TF(u, x, y, z_p, WL)[0]
        F_x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(slice_, axes=1), axis=1), axes=1)
        reference[:, p] = np.dot(F_x, np.exp(1j*kx*x_p))/len(x)
    assert _rel_err(u_t, reference) < 1E-9


@pytest.mark.parametrize("method", ["exact", "rotation"])
def test_round_trip(method):
    w0, z0 = 4.0, 40.0
    x, y = _grid(64.0, 128)
    u = _gaussian(x, y, w0, x0=3.0)
    R = rotation_from_angles(0.3, 0.5)
    x_t = grid1d(-40, 40, 160)
    u_t, x_t, y_t, c = tilted_plane(u, x, y, z0, WL, R=R, method=method, x_out=x_t, y_out=x_t)
    back, x_b, y_b, c_b = tilted_plane(u_t, x_t, y_t, -z0, WL, R=R, reverse=True,
                                       method=method, k_carrier_in=c, x_out=x, y_out=y,
                                       remove_carrier=False)
    assert _rel_err(back, u) < 1E-4


def test_reverse_to_parallel_plane():
    """Tilted plane, then on to a plane parallel to the source: the same as
    the direct angular spectrum propagation."""

    w0, z0, d = 4.0, 40.0, 30.0
    x, y = _grid(64.0, 128)
    u = _gaussian(x, y, w0)
    R = R_y(0.5)
    x_t = grid1d(-50, 50, 200)
    u_t, x_t, y_t, c = tilted_plane(u, x, y, z0, WL, R=R, x_out=x_t, y_out=x_t)
    u_p = tilted_plane(u_t, x_t, y_t, d, WL, R=R, reverse=True, k_carrier_in=c, x_out=x,
                       y_out=y, remove_carrier=False)[0]
    reference = propagate_padded(rayleigh_sommerfeld_I_TF, u, x, y, z0 + d, WL)[0]
    assert _rel_err(u_p, reference) < 1E-4


@pytest.mark.parametrize("theta_deg", [0, 30, 60])
def test_power_flux_conserved(theta_deg):
    w0, z0 = 8.0, 100.0
    x, y = _grid(96.0, 128)
    u = _gaussian(x, y, w0)
    R = R_y(np.radians(theta_deg))
    x_out = grid1d(-100, 100, 256)
    u_t, x_t, y_t, c = tilted_plane(u, x, y, z0, WL, R=R, x_out=x_out, method="rotation")

    P_source = power_flux(u, x, y, WL)
    assert np.isclose(power_flux(u_t, x_t, y_t, WL, k_carrier=c), P_source, rtol=1E-5)
    # the integral of |u|² grows by 1/cos(theta) (paraxial, nearly collimated beam):
    dA = (x_t[1] - x_t[0])*(y_t[1] - y_t[0])
    assert np.isclose(np.sum(np.abs(u_t)**2)*dA, P_source/np.cos(np.radians(theta_deg)),
                      rtol=1E-2)


def test_general_rotation_methods_agree():
    x, y = _grid(48.0, 64)
    u = _gaussian(x, y, 4.0, x0=2.0)
    R = R_z(0.3) @ R_y(0.5) @ R_x(0.2)
    exact = tilted_plane(u, x, y, 30.0, WL, R=R, method="exact")[0]
    rotation = tilted_plane(u, x, y, 30.0, WL, R=R, method="rotation", pad=4)[0]
    assert _rel_err(rotation, exact) < 1E-5


def test_finufft_matches_direct_sum(monkeypatch):
    pytest.importorskip("finufft")
    x, y = _grid(32.0, 32)
    u = _gaussian(x, y, 3.0)
    R = R_z(0.2) @ R_y(0.4)
    with_nufft = tilted_plane(u, x, y, 20.0, WL, R=R, method="exact")[0]
    monkeypatch.setattr(tilted, "finufft", None)
    direct = tilted_plane(u, x, y, 20.0, WL, R=R, method="exact")[0]
    assert _rel_err(with_nufft, direct) < 1E-9


def test_arbitrary_output_points():
    x, y = _grid(48.0, 64)
    u = _gaussian(x, y, 4.0)
    R = R_y(0.4)
    x_out = np.array([-7.0, -1.3, 0.0, 2.2, 9.1])
    y_out = np.array([-2.0, 0.5])
    u_t, x_t, y_t, _ = tilted_plane(u, x, y, 25.0, WL, R=R, x_out=x_out, y_out=y_out,
                                    method="exact", remove_carrier=False)
    reference = _gaussian_3d(_plane_points([0, 0, 25.0], R, x_t, y_t), 4.0)
    assert u_t.shape == (2, 5)
    assert _rel_err(u_t, reference) < 2E-3
    with pytest.raises(ValueError):
        tilted_plane(u, x, y, 25.0, WL, R=R, x_out=x_out, y_out=y_out, method="rotation")


def test_shifted_output_window():
    x, y = _grid(64.0, 128)
    u = _gaussian(x, y, 4.0)
    R = R_y(0.4)
    x_out = grid1d(-20, 40, 100)   # not centered
    y_out = grid1d(-15, 15, 50)
    exact = tilted_plane(u, x, y, 30.0, WL, R=R, x_out=x_out, y_out=y_out, method="exact")[0]
    rotation = tilted_plane(u, x, y, 30.0, WL, R=R, x_out=x_out, y_out=y_out,
                            method="rotation")[0]
    assert _rel_err(rotation, exact) < 1E-4


def test_vectorial_untilted_equals_vectorial_angular_spectrum():
    x, y = _grid(48.0, 64)
    g = _gaussian(x, y, 2.0)
    field = np.stack([g, 0.5j*g], axis=-1)
    E_t = vectorial_tilted_plane(field, x, y, 10.0, WL, pad=1, method="exact")[0]
    reference = vectorial_rayleigh_sommerfeld_TF(field, x, y, 10.0, WL)[0]
    assert np.max(np.abs(E_t - reference)) < 1E-8*np.max(np.abs(reference))


@pytest.mark.parametrize("method", ["exact", "rotation"])
def test_vectorial_plane_wave(method):
    x, y = _grid(32.0, 32)
    dk = 2*np.pi/32.0
    k_in = np.array([4*dk, 2*dk, 0.0])
    k_in[2] = np.sqrt(K**2 - k_in[0]**2 - k_in[1]**2)
    E0 = np.cross(k_in, [0.0, 1.0, 0.0])
    E0 /= np.linalg.norm(E0)          # transverse polarization, E0.k = 0
    X, Y = np.meshgrid(x, y)
    phase = np.exp(1j*(k_in[0]*X + k_in[1]*Y))
    field = np.stack([E0[0]*phase, E0[1]*phase], axis=-1)

    R = R_y(0.5)
    E_t, x_t, y_t, c = vectorial_tilted_plane(field, x, y, 20.0, WL, R=R, pad=1, method="exact")
    expected = np.dot(R.T, E0)*np.exp(1j*k_in[2]*20.0)
    assert np.allclose(E_t, expected[None, None, :], atol=1E-10)
    # a transverse wave in the tilted frame: Ê.k^ = 0
    assert np.isclose(np.dot(np.dot(R.T, E0), np.dot(R.T, k_in)), 0, atol=1E-12)


def test_vectorial_gaussian_methods_agree():
    x, y = _grid(64.0, 128)
    g = _gaussian(x, y, 3.0)
    field = np.stack([g, np.zeros_like(g)], axis=-1)
    R = R_y(np.radians(50))
    x_out = 1.5*x
    exact = vectorial_tilted_plane(field, x, y, 40.0, WL, R=R, method="exact", x_out=x_out)[0]
    rotation = vectorial_tilted_plane(field, x, y, 40.0, WL, R=R, method="rotation",
                                      x_out=x_out)[0]
    assert np.max(np.abs(rotation - exact)) < 1E-5*np.max(np.abs(exact))
    # x polarized beam, plane tilted about y: Ê_x^ = cos(theta)*E_x, Ê_n = sin(theta)*E_x
    centre = (len(x)//2, len(x)//2)
    assert np.isclose(exact[centre][2]/exact[centre][0], np.tan(np.radians(50)), rtol=2E-2)


def test_beam_widths_on_inclined_plane():
    w0, theta = 8.0, np.radians(45)
    x, y = _grid(128.0, 128)
    u = _gaussian(x, y, w0)
    x_out = grid1d(-100, 100, 200)
    widths, u_t, x_t, y_t = tilted_beam_widths(u, x, y, 100.0, WL, R=R_y(theta), x_out=x_out)
    w = w0*np.sqrt(1 + (100.0/(np.pi*w0**2))**2)
    assert np.isclose(widths.d_r if widths.ellipticity > 0.87 else max(widths.d_x, widths.d_y),
                      2*w/np.cos(theta), rtol=2E-2)
    assert np.isclose(min(widths.d_x, widths.d_y), 2*w, rtol=2E-2)


def test_errors():
    x, y = _grid(32.0, 32)
    u = _gaussian(x, y, 3.0)
    with pytest.raises(ValueError, match="crosses"):
        tilted_plane(u, x, y, 5.0, WL, R=R_y(np.radians(60)))
    with pytest.raises(ValueError):
        tilted_plane(u, x, y, 20.0, WL, R=2*np.eye(3))
    with pytest.raises(ValueError):
        tilted_plane(u, x, y, 20.0, WL, R=np.eye(3), angles=(0.1, 0.2))
    with pytest.raises(ValueError):
        tilted_plane(u, x, y, 20.0, WL, method="nonsense")
    with pytest.raises(ValueError):
        tilted_plane(u, x + 1.0, y, 20.0, WL)
    with pytest.raises(ValueError):
        vectorial_tilted_plane(u, x, y, 20.0, WL)


def test_angles_and_backwards_plane():
    x, y = _grid(48.0, 64)
    u = _gaussian(x, y, 4.0)
    a = tilted_plane(u, x, y, 25.0, WL, angles=(0.2, 0.3), method="exact")[0]
    b = tilted_plane(u, x, y, 25.0, WL, R=R_x(0.2) @ R_y(0.3), method="exact")[0]
    assert np.allclose(a, b)
    # a plane behind the source: backward propagation of the forward waves
    back = tilted_plane(u, x, y, -25.0, WL, R=R_y(0.3), method="exact", remove_carrier=False)
    reference = _gaussian_3d(_plane_points([0, 0, -25.0], R_y(0.3), x, y), 4.0)
    assert _rel_err(back[0], reference) < 3E-3
