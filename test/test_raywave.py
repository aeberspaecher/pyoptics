#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.raywave (conversion between rays and wave fields)."""

import numpy as np
import pytest

from pyoptics.analytic_phase import QuadraticPhaseField
from pyoptics.propagators import rayleigh_sommerfeld_I_TF
from pyoptics.raytrace import System, Surface, IdealLens
from pyoptics.raywave import (straight_rays, ray_irradiance, rays_to_field, GaussianBeamlets,
                              field_to_rays, local_spectrum_rays, wigner)
from pyoptics.utils import grid1d


WL = 0.5e-3
K = 2*np.pi/WL


def _ray_grid(N=21, half=1.0):
    u = np.linspace(-half, half, N)
    U, V = np.meshgrid(u, u)
    return U, V


def _spherical_rays(F, N=21, half=1.0):
    """Rays in z = 0 converging to the point (0, 0, F)."""
    U, V = _ray_grid(N, half)
    r = np.sqrt(U**2 + V**2 + F**2)
    D = np.stack([-U, -V, np.full_like(U, F)], -1)/r[..., None]
    P = np.stack([U, V, 0*U], -1)
    return P, D, -(r - F)


def test_rays_to_field_tilted_plane_wave():
    U, V = _ray_grid()
    theta = 0.05
    D = np.zeros(U.shape + (3,))
    D[..., 0], D[..., 2] = np.sin(theta), np.cos(theta)
    P = np.stack([U, V, 0*U], -1)
    x = np.linspace(-0.9, 0.9, 101)
    u = rays_to_field(x, x, P, D, U*np.sin(theta), WL, launch=P[..., :2], irradiance=4.0)
    X, _ = np.meshgrid(x, x)
    assert np.allclose(u, 2*np.exp(1j*K*np.sin(theta)*X), atol=1e-10)


@pytest.mark.parametrize("z", [5.0, 15.0])
def test_rays_to_field_spherical_wave_and_gouy_phase(z):
    F = 10.0
    P, D, S = _spherical_rays(F)
    Pz, Sz = straight_rays(P, D, S, z)
    x = np.linspace(-0.4, 0.4, 81)
    u = rays_to_field(x, x, Pz, D, Sz, WL, launch=P[..., :2])
    c = 40
    # amplitude transport: the beam shrinks by |F - z|/F
    assert np.isclose(abs(u[c, c]), F/abs(F - z), rtol=1e-6)
    # on axis the phase is k*z, minus pi behind the focus (two caustics)
    expected = 0.0 if z < F else -np.pi
    assert np.isclose(np.angle(u[c, c]*np.exp(-1j*K*z)*np.exp(-1j*expected)), 0.0, atol=1e-6)
    # off axis: the spherical wavefront about the focus
    rho = np.sqrt(x**2 + (F - z)**2)
    sign = np.sign(F - z)
    ph = np.angle(u[c]*np.conj(u[c, c]))
    assert np.allclose(ph[20:61], np.angle(np.exp(-1j*sign*K*(rho - abs(F - z))))[20:61],
                       atol=1e-3)


def test_rays_to_field_counts_shared_edges_once():
    # grid points of the field on the ray grid lines and vertices
    U, V = _ray_grid(11)
    P = np.stack([U, V, 0*U], -1)
    D = np.zeros(U.shape + (3,))
    D[..., 2] = 1
    x = np.linspace(-1, 1, 41)
    u = rays_to_field(x, x, P, D, 0*U, WL, launch=P[..., :2])
    inner = np.abs(u)[1:-1, 1:-1]
    assert np.allclose(inner, 1.0)


def test_rays_to_field_power_conservation_through_lens():
    lens = System([Surface(np.inf, 0.0, semi_diameter=5.0, label="stop"),
                   IdealLens(100.0, thickness=60.0)], 0, WL)
    rho, th = np.linspace(0, 1, 40), np.linspace(0, 2*np.pi, 81)
    R, T = np.meshgrid(rho, th)
    px, py = R*np.cos(T), R*np.sin(T)
    px[-1], py[-1] = px[0], py[0]
    b = lens.trace_pupil(0.0, px.ravel(), py.ravel())
    shape = px.shape
    P = b.points[-1].reshape(shape + (3,))
    x = grid1d(-2.5, 2.5, 256)
    u = rays_to_field(x, x, P, b.directions[-1].reshape(shape + (3,)), b.opl[-1].reshape(shape),
                      WL, launch=np.stack([5*px, 5*py], -1))
    power = np.sum(np.abs(u)**2)*(x[1] - x[0])**2
    assert np.isclose(power, np.pi*5.0**2*np.cos(np.pi/80), rtol=3e-3)  # polygon of 80 sides
    # irradiance (100/40)**2
    assert np.isclose(abs(u[128, 128])**2, 6.25, rtol=1e-3)


def test_ray_irradiance_and_straight_rays():
    P = np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.6, 0.1, 0.0]])
    D = np.array([[0, 0, 1.0], [0.6, 0, 0.8], [0, 0, 1.0]])
    Q, S = straight_rays(P, D, np.zeros(3), 1.0, n=1.5)
    assert np.allclose(Q[1], [0.85, 0.1, 1.0])
    assert np.allclose(S, [1.5, 1.875, 1.5])
    E = ray_irradiance(P, [1.0, 2.0, 1.0], [0, 0.5, 1.0], [0, 0.5])
    assert np.allclose(E, [[12.0, 4.0]])


def _gaussian(N=512, L=2.0, w0=0.25, f=20.0, tilt=0.01):
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    u = np.exp(-(X**2 + Y**2)/w0**2)*np.exp(-1j*K*(X**2 + Y**2)/(2*f))*np.exp(1j*K*tilt*X)
    return u, x, X, Y


@pytest.mark.parametrize("z", [10.0, 20.0, 24.0])
def test_beamlets_from_field_through_focus(z):
    u0, x, X, Y = _gaussian()
    B = GaussianBeamlets.from_field(u0, x, x, WL, step=4)
    ref = rayleigh_sommerfeld_I_TF(u0, x, x, z, WL)[0]
    c = len(x)//2
    ub = B.field(X[c], Y[c], z)
    assert np.max(np.abs(ub - ref[c]))/np.max(np.abs(ref)) < 0.015


def test_beamlets_from_quadratic_phase_field():
    u0, x, X, Y = _gaussian(tilt=0.0)
    q = QuadraticPhaseField.from_field(u0, x, x, WL, R=-20.0)
    B1 = GaussianBeamlets.from_field(q, None, None, None, step=4)
    B2 = GaussianBeamlets.from_field(u0, x, x, WL, step=4)
    c = len(x)//2
    assert np.allclose(B1.field(X[c], Y[c], 19.0), B2.field(X[c], Y[c], 19.0), atol=1e-6)


def test_beamlets_from_rays_focus():
    # converging spherical wave: the beamlets give the Airy focus of the aperture
    F = 50.0
    U, V = _ray_grid(61, 2.0)
    r = np.sqrt(U**2 + V**2 + F**2)
    D = np.stack([-U, -V, np.full_like(U, F)], -1)/r[..., None]
    P = np.stack([U, V, 0*U], -1)
    S = -(r - F)
    inside = U**2 + V**2 <= 4.0
    P[~inside] = np.nan
    B = GaussianBeamlets.from_rays(P, D, S, WL, launch=P[..., :2], overlap=1.5)
    axis = B.field(np.zeros(1), np.zeros(1), F)
    # power in focus: |u(0)|**2 = (pi a**2/(lambda F))**2 for a uniform disk of radius a
    a_eff = np.sqrt(np.sum(inside)*(U[0, 1] - U[0, 0])**2/np.pi)
    assert np.isclose(abs(axis[0])**2, (np.pi*a_eff**2/(WL*F))**2, rtol=0.05)


def test_field_to_rays_gaussian_beam():
    u0, x, X, Y = _gaussian()
    R = field_to_rays(u0, x, x, WL, step=8, z=1.0)
    assert np.isclose(R.power.sum(), np.sum(np.abs(u0)**2)*(x[1] - x[0])**2, rtol=5e-3)
    assert np.allclose(R.points[:, 2], 1.0)
    # converging wavefront with tilt: direction cosine 0.01 - x/f
    assert np.allclose(R.directions[:, 0], 0.01 - R.points[:, 0]/20.0, atol=2e-5)
    assert np.allclose(np.linalg.norm(R.directions, axis=1), 1.0)
    # round trip: rays back to a field
    Rq = field_to_rays(QuadraticPhaseField.from_field(u0, x, x, WL, R=-20.0), step=8)
    assert np.allclose(Rq.directions, R.directions, atol=1e-6)


def test_field_to_rays_traces_through_system():
    # a collimated Gaussian beam focused by an ideal lens: the rays meet at the focus
    x = grid1d(-2, 2, 128)
    X, Y = np.meshgrid(x, x)
    u = np.exp(-(X**2 + Y**2)/1.0)
    rays = field_to_rays(u, x, x, WL, z=-1.0, step=4)
    lens = System([IdealLens(80.0, thickness=80.0, semi_diameter=5.0)], 0, WL)
    b = lens.trace(rays)
    assert np.max(np.abs(b.points[-1][:, :2])) < 1e-9


def test_local_spectrum_rays_crossing_waves():
    x = grid1d(-1, 1, 256)
    X, _ = np.meshgrid(x, x)
    u = np.exp(1j*K*0.02*X) + 0.5*np.exp(-1j*K*0.03*X)
    R = local_spectrum_rays(u, x, x, WL, window=32, threshold=1e-4)
    sx = R.directions[:, 0]
    p1, p2 = R.power[sx > 0].sum(), R.power[sx < 0].sum()
    assert np.isclose(p1/p2, 4.0, rtol=0.05)
    assert np.isclose(np.average(sx[sx > 0], weights=R.power[sx > 0]), 0.02, atol=1e-3)
    assert np.isclose(np.average(sx[sx < 0], weights=R.power[sx < 0]), -0.03, atol=1e-3)
    assert np.isclose(R.power.sum(), np.sum(np.abs(u)**2)*(x[1] - x[0])**2)
    assert np.all(np.isnan(R.opl))
    Rj = local_spectrum_rays(u, x, x, WL, window=32, threshold=1e-4, jitter=1)
    assert np.isclose(np.average(Rj.directions[:, 0], weights=Rj.power),
                      np.average(sx, weights=R.power), atol=1e-4)
    assert np.allclose(np.linalg.norm(Rj.directions, axis=1), 1.0)


def test_wigner_marginals_and_ray_line():
    x = grid1d(-1, 1, 256)
    u = np.exp(-x**2/0.1**2)*np.exp(1j*K*(0.005*x - x**2/(2*40.0)))
    k, W = wigner(u, x)
    dk = k[1] - k[0]
    assert np.allclose(np.sum(W, 0)*dk/(2*np.pi), np.abs(u)**2, atol=1e-10)
    # the Wigner function of a chirped Gaussian sits on the ray line k = dS/dx
    for j in (100, 128, 150):
        assert abs(k[np.argmax(W[:, j])] - K*(0.005 - x[j]/40.0)) < 1.5*dk
