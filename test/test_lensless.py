#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the lensless imaging tools against analytical results: the on-axis
Fresnel intensity of a pinhole, the delta correlation of MURAs, exact coded
aperture decoding and its photon noise, the shift and scaling of a diffuser
PSF, and deconvolution of a noise-free picture.
"""

import numpy as np
import pytest

from pyoptics import lensless as ll
from pyoptics.propagators import fresnel_scaled
from pyoptics.utils import grid2d
from pyoptics.wavefront_shaping import random_phase_screen


def test_rayleigh_and_petzval_pinholes():
    wl, v = 0.5, 4e4
    assert ll.rayleigh_pinhole_diameter(wl, v) == pytest.approx(1.9*np.sqrt(wl*v))
    # object at u: 1/z = 1/u + 1/v
    u = 1e5
    z = 1/(1/u + 1/v)
    assert ll.rayleigh_pinhole_diameter(wl, v, u) == pytest.approx(1.9*np.sqrt(wl*z))
    # the quadrature blur is smallest at sqrt(2.44 wl z)
    d = np.linspace(50, 400, 20001)
    for obj in (np.inf, u):
        blur, geo, diff = ll.pinhole_blur(d, wl, v, obj)
        zz = v if np.isinf(obj) else z
        assert d[np.argmin(blur)] == pytest.approx(np.sqrt(2.44*wl*zz), rel=1e-3)
        assert np.allclose(blur, np.hypot(geo, diff))


def test_pinhole_on_axis_intensity():
    wl, v = 0.5, 2e4
    s = np.sqrt(wl*v)
    # first Fresnel zone: four times the incident intensity; two zones: dark
    assert ll.pinhole_on_axis_intensity(2*s, wl, v) == pytest.approx(4.0)
    assert ll.pinhole_on_axis_intensity(2*np.sqrt(2)*s, wl, v) == pytest.approx(0.0, abs=1e-12)
    # against the Fresnel integral of a plane wave through a circular hole
    for q in (0.8, 1.5, 2.0, 2.6):
        d = q*s
        n = 256
        x = (np.arange(n) - n/2 + 0.5)*(1.05*d/n)
        X, Y = np.meshgrid(x, x)
        hole = (X**2 + Y**2 <= (d/2)**2).astype(complex)
        U, _, _ = fresnel_scaled(hole, x, x, v, wl, np.array([0.0]), np.array([0.0]))
        assert np.abs(U[0, 0])**2 == pytest.approx(ll.pinhole_on_axis_intensity(d, wl, v), abs=0.03)


def test_mura_orders_and_errors():
    assert ll.mura_orders(62) == [5, 13, 17, 29, 37, 41, 53, 61]
    with pytest.raises(ValueError):
        ll.mura(7)
    with pytest.raises(ValueError):
        ll.mura(25)


@pytest.mark.parametrize("p", [5, 13, 29, 61])
def test_mura_delta_correlation(p):
    A = ll.mura(p)
    G = ll.mura_decoder(A)
    n_open = (p*p - 1)//2
    assert A.sum() == n_open and G.sum() == 1 and A[0, 0] == 0
    corr = np.real(np.fft.ifft2(np.fft.fft2(A)*np.conj(np.fft.fft2(G))))
    expected = np.zeros((p, p))
    expected[0, 0] = n_open
    assert np.allclose(corr, expected, atol=1e-9)


def test_coded_aperture_roundtrip_and_background():
    p = 29
    A = ll.mura(p)
    G = ll.mura_decoder(A)
    rng = np.random.default_rng(0)
    scene = rng.random((p, p))
    picture = ll.coded_aperture_image(scene, A)
    # one source element casts one shifted copy of the mask
    delta = np.zeros((p, p))
    delta[3, 7] = 1.0
    assert np.allclose(ll.coded_aperture_image(delta, A), np.roll(A, (3, 7), axis=(0, 1)))
    assert np.allclose(ll.coded_aperture_decode(picture, G), scene)
    B = 5.0
    n_open = (p*p - 1)/2
    assert np.allclose(ll.coded_aperture_decode(picture + B, G), scene + B/n_open)


def test_coded_aperture_snr_monte_carlo():
    p = 13
    A = ll.mura(p)
    G = ll.mura_decoder(A)
    n_open = (p*p - 1)/2
    scene = np.zeros((p, p))
    scene[2, 5] = 20.0
    scene[8, 1] = 5.0
    B = 2.0
    picture = ll.coded_aperture_image(scene, A) + B
    rng = np.random.default_rng(1)
    est = np.array([ll.coded_aperture_decode(rng.poisson(picture), G) for _ in range(4000)])
    for (i, j) in ((2, 5), (8, 1), (0, 0)):
        expected = ll.coded_aperture_snr(scene[i, j], scene.sum(), n_open, B, p*p)
        signal = est[:, i, j].mean() - B/n_open
        assert signal/est[:, i, j].std() == pytest.approx(expected, rel=0.06, abs=0.1)
    # a lone bright point gains sqrt(n_open) over one hole
    assert ll.coded_aperture_snr(100.0, 100.0, n_open)/ll.pinhole_snr(100.0) == pytest.approx(np.sqrt(n_open))


def test_caustic_distance_scaling():
    f = ll.caustic_distance(90.0, 5.0, 0.55)
    assert f == pytest.approx(2*np.pi*90**2/(np.sqrt(12)*5*0.55))
    assert ll.caustic_distance(180.0, 5.0, 0.55) == pytest.approx(4*f)
    assert ll.caustic_distance(90.0, 10.0, 0.55) == pytest.approx(f/2)


@pytest.fixture(scope="module")
def diffuser():
    wl, n, dx, z = 0.55, 512, 4.0, 4000.0
    x, y, X, Y = grid2d(-n*dx/2, n*dx/2, -n*dx/2, n*dx/2, n)
    phase = random_phase_screen((n, n), 90.0, 5.0, dx=dx, rng=3)
    T = (np.hypot(X, Y) <= 600.0)*np.exp(1j*phase)
    return x, y, T, wl, z, dx


def test_diffuser_psf_shift_and_scale(diffuser):
    x, y, T, wl, z, dx = diffuser
    u = 40000.0
    p0 = ll.point_source_psf(T, x, y, z, wl, u)
    assert p0.sum() == pytest.approx(1.0)
    # lateral shift -s z/u
    s = 1600.0
    p_shift = ll.point_source_psf(T, x, y, z, wl, u, source_x=s)
    C = np.real(np.fft.ifft2(np.fft.fft2(p_shift)*np.conj(np.fft.fft2(p0))))
    iy, ix = np.unravel_index(np.argmax(C), C.shape)
    n = len(x)
    assert ((iy + n//2) % n - n//2, (ix + n//2) % n - n//2) == (0, round(-s*z/u/dx))
    # depth: the pattern scales with (1 + z/u)
    u2 = 2*u
    p2 = ll.point_source_psf(T, x, y, z, wl, u2)
    factors = np.linspace(0.92, 1.0, 33)
    corr = [np.corrcoef(ll.scale_psf(p0, f).ravel(), p2.ravel())[0, 1] for f in factors]
    expected = ll.psf_magnification(z, u2)/ll.psf_magnification(z, u)
    assert factors[np.argmax(corr)] == pytest.approx(expected, abs=0.004)
    assert max(corr) > 0.95
    assert ll.psf_magnification(z, np.inf) == 1.0


def test_scale_psf_preserves_sum_and_centre():
    n = 128
    j, i = np.meshgrid(np.arange(n) - n//2, np.arange(n) - n//2)
    blob = lambda x0, w: np.exp(-((j - x0)**2 + i**2)/(2*w**2))
    psf = blob(0.0, 3.0) + blob(20.0, 3.0)
    s = ll.scale_psf(psf, 1.5, order=3)
    assert s.sum() == pytest.approx(psf.sum())
    # the blob at the centre stays, the one at 20 moves to 30, both widen by 1.5
    expected = blob(0.0, 4.5) + blob(30.0, 4.5)
    expected *= psf.sum()/expected.sum()
    assert np.allclose(s, expected, atol=2e-3*s.max())


def test_deconvolution_noise_free():
    rng = np.random.default_rng(5)
    n = 128
    psf = np.zeros((n, n))
    idx = rng.integers(n//2 - 20, n//2 + 20, (40, 2))
    psf[idx[:, 0], idx[:, 1]] = rng.random(40)
    psf /= psf.sum()
    scene = np.zeros((n, n))
    scene[50:70, 40:60] = 1.0
    scene[60:90, 75:80] = 0.5
    picture = ll.convolve_crop(scene, psf)
    # Wiener with a tiny regularization inverts a well conditioned PSF
    x_w = ll.wiener_deconvolve(picture, psf, 1e-8)
    assert np.allclose(x_w, scene, atol=1e-3)
    # TV-ADMM on a cropped sensor: piecewise constant scene, PSF support of 41
    sensor = (96, 96)
    b = ll.convolve_crop(scene, psf, sensor)
    assert b.shape == sensor
    x_a = ll.admm_tv_deconvolve(b, psf, tau=1e-6, n_iter=600)
    err = np.linalg.norm(x_a - scene)/np.linalg.norm(scene)
    assert err < 0.05
    assert x_a.min() > -1e-3
    assert ll.total_variation(scene) == pytest.approx(2*(20 + 20) + 2*(30 + 5)*0.5)
