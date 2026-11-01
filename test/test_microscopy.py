#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.microscopy."""

import numpy as np
import pytest

from pyoptics import microscopy as mic
from pyoptics.imaging import image as coherent_image
from pyoptics.utils import grid2d


WL = 0.5
N = 128
DX = 0.05


@pytest.fixture
def grid():
    L = N*DX
    x, y, X, Y = grid2d(-L/2, L/2, -L/2, L/2, N)
    return x, y, X, Y


def test_source_points_disk_and_annulus(grid):
    x, y, X, Y = grid
    k_x, k_y, w = mic.source_points(x, y, WL, 0.5)
    k0 = 2*np.pi/WL
    assert np.all(np.hypot(k_x, k_y) <= 0.5*k0 + 1e-9)
    assert np.isclose(w.sum(), 1)
    # area of the disk in grid cells
    dk = 2*np.pi/(N*DX)
    assert abs(len(w) - np.pi*(0.5*k0/dk)**2)/len(w) < 0.1
    # symmetric
    assert np.isclose(np.sum(w*k_x), 0) and np.isclose(np.sum(w*k_y), 0)
    k_x, k_y, w = mic.source_points(x, y, WL, 0.6, NA_inner=0.5)
    rho = np.hypot(k_x, k_y)/k0
    assert np.all((rho >= 0.5 - 1e-9) & (rho <= 0.6 + 1e-9))
    k_x, k_y, w = mic.source_points(x, y, WL, 0.0)
    assert len(w) == 1 and k_x[0] == 0 and k_y[0] == 0


def test_clear_object_gives_unit_intensity(grid):
    x, y, X, Y = grid
    src = mic.source_points(x, y, WL, 0.4, step=2)
    I = mic.abbe_image(np.ones((N, N)), x, y, WL, 0.8, source=src)
    assert np.allclose(I, 1)


def test_coherent_abbe_matches_imaging(grid):
    x, y, X, Y = grid
    obj = np.exp(1j*np.sin(2*np.pi*X/0.9))*(np.abs(Y) < 1.0)
    I = mic.abbe_image(obj, x, y, WL, 0.6)
    I_ref = np.abs(coherent_image(obj, x, y, 0.6, 2*np.pi/WL))**2
    assert np.allclose(I, I_ref, atol=1e-10)


def test_grating_cutoff(grid):
    x, y, X, Y = grid
    NA = 0.5
    # period 0.8 um < wl/NA = 1 um: coherent on-axis light cannot resolve it ...
    p = N*DX/8
    obj = 1 + 0.5*np.cos(2*np.pi*X/p)
    I_coh = mic.abbe_image(obj, x, y, WL, NA)
    assert np.ptp(I_coh) < 1e-10
    # ... oblique illumination (sigma = 1) can, down to wl/(2 NA) = 0.5 um
    assert mic.grating_cutoff(WL, NA, NA) < p < mic.grating_cutoff(WL, NA)
    src = mic.source_points(x, y, WL, NA)
    I_inc = mic.abbe_image(obj, x, y, WL, NA, source=src)
    assert np.ptp(I_inc) > 0.05


def test_weak_object_transfer(grid):
    x, y, X, Y = grid
    NA = 0.6
    src = mic.source_points(x, y, WL, 0.3)
    H_a, H_phi = mic.weak_object_transfer(x, y, WL, NA, source=src)
    # in focus, symmetric: no phase transfer
    assert np.max(np.abs(H_phi)) < 1e-12
    assert np.isclose(H_a[N//2, N//2], 2)
    # defocus: weak phase object image matches the linear prediction
    P = mic.pupil(x, y, WL, NA, defocus=0.6)
    H_a, H_phi = mic.weak_object_transfer(x, y, WL, NA, source=src, P=P)
    assert np.max(np.abs(H_phi)) > 0.1
    eps = 1e-3
    phi = eps*(np.cos(2*np.pi*X/(N*DX/5)) + np.sin(2*np.pi*(X + 2*Y)/(N*DX/3)))
    I = mic.abbe_image(np.exp(1j*phi), x, y, WL, NA, source=src, P=P)
    from pyoptics.fft import FT, inv_FT
    I_lin = 1 + np.real(inv_FT(H_phi*FT(phi)))
    assert np.max(np.abs(I - I_lin)) < 5*eps**2


def test_phase_contrast_weak_object(grid):
    x, y, X, Y = grid
    NA = 0.8
    # coherent axial light and a small central phase plate: the +/-1 orders miss it
    eps = 0.01
    phi = eps*np.cos(2*np.pi*X/(N*DX/8))
    for sign in (-1, 1):
        ring = mic.phase_ring(x, y, WL, 0.0, 0.05, phase=sign*np.pi/2, transmission=0.25)
        P = mic.pupil(x, y, WL, NA, mask=ring)
        I = mic.abbe_image(np.exp(1j*phi), x, y, WL, NA, P=P)
        # I = T + 2 sqrt(T) sin(phase) phi
        assert np.allclose(I, 0.25 + 2*0.5*sign*phi, atol=3*eps**2)


def test_dic_uniform_and_gradient(grid):
    x, y, X, Y = grid
    for beta in (0.0, np.pi/2, np.pi):
        t = mic.dic_object(np.ones((N, N)), x, y, 0.1, beta)
        assert np.allclose(np.abs(t)**2, np.sin(beta/2)**2)
    # small shear: I ~ sin^2(b/2) + sin(b)/2 * s * dphi/dx
    eps, s, beta = 0.05, 0.05, np.pi/2
    q = 2*np.pi/(N*DX/2)
    phi = eps*np.sin(q*X)
    t = mic.dic_object(np.exp(1j*phi), x, y, s, beta)
    I = np.abs(t)**2
    I_lin = 0.5 + 0.5*s*eps*q*np.cos(q*X)
    assert np.max(np.abs(I - I_lin)) < 0.05*np.ptp(I_lin)


def test_psf_3d_focus_and_axis(grid):
    x, y, X, Y = grid
    NA = 0.3
    z = np.linspace(-6, 6, 25)
    h = mic.psf_3d(x, y, z, WL, NA)
    assert h.shape == (len(z), N, N)
    assert np.isclose(np.abs(h[12, N//2, N//2])**2, 1)
    # paraxial on-axis intensity sinc^2(u/4), u = k0 NA^2 z
    u = 2*np.pi/WL*NA**2*z
    I_axis = np.abs(h[:, N//2, N//2])**2
    assert np.allclose(I_axis, np.sinc(u/(4*np.pi))**2, atol=0.03)
    # energy per plane conserved
    E = np.sum(np.abs(h)**2, axis=(1, 2))
    assert np.allclose(E, E[0])
    h_a = mic.psf_3d(x, y, 0.0, WL, NA, apodization="aplanatic")
    assert h_a.shape == (N, N)


def test_confocal_psf(grid):
    x, y, X, Y = grid
    NA = 0.8
    z = np.linspace(-2, 2, 9)
    h = mic.psf_3d(x, y, z, WL, NA)
    point = mic.confocal_psf(h, h, x, y, None)
    assert np.allclose(point, np.abs(h)**4)
    wide = mic.plane_response(np.abs(h)**2)
    assert np.allclose(wide, 1)
    conf = mic.plane_response(mic.confocal_psf(h, h, x, y, mic.airy_unit(WL, NA)))
    assert conf[4] == 1 and conf[0] < 0.5 and np.allclose(conf, conf[::-1])
    # a pinhole as large as the window approaches wide field
    big = mic.plane_response(mic.confocal_psf(h, h, x, y, 4*N*DX))
    assert np.allclose(big, 1, atol=1e-6)


def test_mirror_response():
    NA = 0.2
    z = np.linspace(-20, 20, 101)
    I = mic.mirror_response(z, WL, NA)
    assert np.isclose(I[50], 1)
    u = 2*np.pi/WL*z*NA**2
    assert np.allclose(I, np.sinc(u/(2*np.pi))**2, atol=0.01)


def test_circle_overlap():
    assert np.isclose(mic.circle_overlap(0, 1), 1)
    assert np.isclose(mic.circle_overlap(2, 1), 0)
    assert np.isclose(mic.circle_overlap(5, 1), 0)
    # d = R: (2 pi/3 - sqrt(3)/2)/pi
    assert np.isclose(mic.circle_overlap(1, 1), (2*np.pi/3 - np.sqrt(3)/2)/np.pi)


def test_fpm_forward_and_reconstruct():
    wl, NA = 0.5, 0.1
    N, dx, m = 128, 0.25, 32           # 32 um field, camera pitch 1 um <= wl/(4 NA)
    L = N*dx
    x, y, X, Y = grid2d(-L/2, L/2, -L/2, L/2, N)
    k_x, k_y, NA_ill = mic.led_array(5, 4.0, 80.0, wl)
    images, x_lr, y_lr = mic.fpm_images(np.ones((N, N)), x, y, wl, NA, k_x, k_y, m)
    assert images.shape == (25, m, m)
    bright = NA_ill < NA*0.95
    assert np.allclose(images[bright], 1)
    assert np.allclose(images[NA_ill > NA*1.05], 0, atol=1e-20)
    # a smooth test object: amplitude and phase
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    a = 1 - 0.3*gaussian_filter(rng.random((N, N)), 2, mode="wrap")*4
    ph = 2*gaussian_filter(rng.random((N, N)), 2, mode="wrap")*4
    obj = a*np.exp(1j*ph)
    images, _, _ = mic.fpm_images(obj, x, y, wl, NA, k_x, k_y, m)
    rec, P, err = mic.fpm_reconstruct(images, x, y, wl, NA, k_x, k_y, n_iter=30,
                                      update_pupil=False)
    assert err[-1] < err[0]/10
    # compare within the synthetic aperture (remove the global phase)
    from pyoptics.fft import FT, inv_FT
    from pyoptics.utils import freq_grid
    K_x, K_y = freq_grid(x, y)
    k0 = 2*np.pi/wl
    synth = np.hypot(K_x, K_y) <= k0*(NA + 0.9*NA_ill.max())*0.7
    ref = inv_FT(FT(obj)*synth)
    rec_f = inv_FT(FT(rec)*synth)
    c = np.vdot(rec_f, ref)/abs(np.vdot(rec_f, ref))
    rel = np.linalg.norm(rec_f*c - ref)/np.linalg.norm(ref)
    assert rel < 0.05


def test_fpm_pupil_recovery():
    from scipy.ndimage import gaussian_filter
    from pyoptics.bases import fringe_zernike
    from pyoptics.utils import freq_grid
    wl, NA = 0.5, 0.1
    N, dx, m = 128, 0.25, 32
    L = N*dx
    x, y, X, Y = grid2d(-L/2, L/2, -L/2, L/2, N)
    k_x, k_y, NA_ill = mic.led_array(7, 4.0, 80.0, wl)
    rng = np.random.default_rng(1)
    a = 1 - 0.5*(gaussian_filter(rng.random((N, N)), 2, mode="wrap") > 0.5)
    ph = 6*gaussian_filter(rng.random((N, N)), 3, mode="wrap")
    obj = gaussian_filter(a, 0.5, mode="wrap")*np.exp(1j*ph)
    P0, x_lr, y_lr = mic.fpm_pupil(x, y, m, wl, NA)
    K_x, K_y = freq_grid(x_lr, y_lr)
    k0 = 2*np.pi/wl
    rx, ry = K_x/(k0*NA), K_y/(k0*NA)
    W = 0.1*fringe_zernike(4, rx, ry)
    P_true = P0*np.exp(2j*np.pi*W)
    images, _, _ = mic.fpm_images(obj, x, y, wl, NA, k_x, k_y, m, P=P_true)
    rec, P, err = mic.fpm_reconstruct(images, x, y, wl, NA, k_x, k_y, n_iter=60)
    # compare without piston and tilt (a pupil tilt is an object shift), inside
    # 80 % of the radius (the rim is seen by few frequencies)
    inside = np.hypot(rx, ry) < 0.8
    # compare without piston and tilt (a pupil tilt is an object shift)
    d = np.unwrap(np.angle(P*np.conj(P_true)))[inside]
    A = np.stack([np.ones(inside.sum()), rx[inside], ry[inside]], axis=1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    rms_W = np.std(W[inside])
    assert np.std(d - A @ c)/(2*np.pi) < 0.75*rms_W
    # the data are fitted better than with the ideal pupil
    _, _, err_fixed = mic.fpm_reconstruct(images, x, y, wl, NA, k_x, k_y, n_iter=60,
                                          update_pupil=False)
    assert err[-1] < err_fixed[-1]/2
    with pytest.raises(ValueError):
        mic.fpm_reconstruct(images, x, y, wl, NA, k_x, k_y, n_iter=1, method="newton")
