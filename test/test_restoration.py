#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the restoration module
(notebook 160, image restoration and PSF deconvolution)."""

import numpy as np
import pytest
from scipy.signal import convolve2d

from pyoptics import raytrace as rt
from pyoptics import restoration as rs


def _gauss(p, s):
    yy, xx = np.indices((p, p)) - p//2
    g = np.exp(-(xx**2 + yy**2)/(2*s**2))
    return g/g.sum()


def test_invariant_blur_is_linear_convolution():
    rng = np.random.default_rng(0)
    psf = rng.random((7, 7))
    psf /= psf.sum()
    x = rng.random((40, 50))
    op = rs.SpaceVariantBlur.invariant(psf, x.shape)
    assert np.allclose(op.forward(x), convolve2d(x, psf, mode="same"), atol=1e-12)
    # with a margin, the image is the crop of the full convolution of the larger scene
    op = rs.SpaceVariantBlur.invariant(psf, (30, 40), margin=5)
    assert np.allclose(op.forward(x), convolve2d(x, psf, mode="same")[5:-5, 5:-5], atol=1e-12)


@pytest.mark.parametrize("rank", [None, 2])
def test_space_variant_adjoint(rank):
    rng = np.random.default_rng(1)
    psfs = rng.random((2, 3, 9, 9))
    psfs /= psfs.sum(axis=(-1, -2), keepdims=True)
    op = rs.SpaceVariantBlur.from_grid(psfs, [0, 39], [0, 20, 49], (40, 50), rank=rank, margin=4)
    x, y = rng.random(op.scene_shape), rng.random(op.shape)
    assert np.isclose(np.vdot(op.forward(x), y), np.vdot(x, op.adjoint(y)), rtol=1e-10)
    assert op.norm() <= 1.0 + 1e-6


def test_space_variant_interpolates_between_nodes():
    # a delta scene at a node is blurred by that node's PSF, halfway in between by the mean
    psfs = np.array([[_gauss(9, 1.0), _gauss(9, 2.0)]])
    op = rs.SpaceVariantBlur.from_grid(psfs, [20], [10, 30], (41, 41))
    for col, psf in ((10, psfs[0, 0]), (30, psfs[0, 1]), (20, psfs[0].mean(axis=0))):
        x = np.zeros((41, 41))
        x[20, col] = 1.0
        assert np.allclose(op.forward(x)[16:25, col - 4:col + 5], psf, atol=1e-12)
    # full rank SVD model equals the interpolation model
    full = rs.SpaceVariantBlur.from_grid(psfs, [20], [10, 30], (41, 41), rank=2)
    x = np.random.default_rng(2).random((41, 41))
    assert np.allclose(full.forward(x), op.forward(x), atol=1e-12)


def test_hat_weights_partition_of_unity():
    W = rs.hat_weights([0, 10, 40], 50)
    assert np.allclose(W.sum(axis=0), 1.0)
    assert W[1, 10] == 1.0 and np.isclose(W[1, 25], 0.5)


def test_wiener_inverts_noiseless_blur():
    x = rs.dead_leaves((96, 96), r_min=2, r_max=30, rng=3)
    psf = _gauss(11, 1.0)
    op = rs.SpaceVariantBlur.invariant(psf, (64, 64), margin=16)
    y = op.forward(x)
    xr = rs.wiener(y, psf, 1e-8)
    assert rs.psnr(xr, x[16:-16, 16:-16], border=12) > 35
    assert rs.psnr(xr, x[16:-16, 16:-16], border=12) > rs.psnr(y, x[16:-16, 16:-16], border=12) + 10


def test_richardson_lucy_improves_and_stays_positive():
    rng = np.random.default_rng(4)
    x = rs.dead_leaves((96, 96), r_min=2, r_max=30, rng=5)
    psf = _gauss(11, 1.5)
    op = rs.SpaceVariantBlur.invariant(psf, (64, 64), margin=16)
    y = rng.poisson(op.forward(x)*2000)/2000
    xr = rs.richardson_lucy(y, op, 30)
    assert np.all(xr >= 0)
    ref = x[16:-16, 16:-16]
    assert rs.psnr(xr[16:-16, 16:-16], ref, border=8) > rs.psnr(y, ref, border=8) + 1.5


def test_tv_and_cross_channel():
    rng = np.random.default_rng(6)
    g = rs.dead_leaves((80, 80), r_min=3, r_max=30, rng=7)
    psf = _gauss(9, 1.5)
    op = rs.SpaceVariantBlur.invariant(psf, (80, 80))
    y = op.forward(0.7*g) + 0.01*rng.standard_normal((80, 80))
    plain = rs.deconvolve_tv(y, op, 2e-3, n_iter=150)
    cross = rs.deconvolve_tv(y, op, 1e-3, n_iter=150, reference=g, cross=5e-3)
    e_plain = rs.psnr(plain, 0.7*g, border=8)
    assert e_plain > rs.psnr(y, 0.7*g, border=8) + 1
    # a reference with the same edges helps
    assert rs.psnr(cross, 0.7*g, border=8) > e_plain + 1


def test_masked_data_term():
    # with half of the pixels missing, the masked solvers still restore
    x = rs.dead_leaves((80, 80), r_min=3, r_max=30, rng=8)
    psf = _gauss(7, 1.0)
    op = rs.SpaceVariantBlur.invariant(psf, (80, 80))
    m = (np.indices((80, 80)).sum(axis=0) % 2).astype(float)
    y = op.forward(x)*m
    for xr in (rs.richardson_lucy(y + 0.01, op, 40, background=0.01, mask=m),
               rs.deconvolve_tv(y, op, 1e-4, n_iter=300, mask=m)):
        assert rs.psnr(xr, x, border=8) > 22


def test_haar_and_unrolled_hqs():
    x = np.random.default_rng(9).random((32, 48))
    c = rs.haar2(x, 3)
    assert np.isclose(np.linalg.norm(c), np.linalg.norm(x))
    assert np.allclose(rs.inverse_haar2(c, 3), x)
    img = rs.dead_leaves((96, 96), r_min=2, r_max=30, rng=10)
    psf = _gauss(9, 1.2)
    y = rs.SpaceVariantBlur.invariant(psf, (64, 64), margin=16).forward(img)
    xr = rs.unrolled_hqs(y, psf, [(-3, -4)]*4)
    assert xr.shape == y.shape
    assert rs.psnr(xr, img[16:-16, 16:-16], border=10) > rs.psnr(y, img[16:-16, 16:-16], border=10) + 5


def test_blended_with_one_node_is_the_solver():
    rng = np.random.default_rng(11)
    y = rng.random((40, 40))
    psf = _gauss(7, 1.0)
    f = lambda w, p: rs.wiener(w, p, 1e-2)
    out = rs.blended_deconvolve(y, psf[None, None], [20], [20], f)
    assert np.allclose(out, f(y, psf))


def test_cfa_and_channel_psfs():
    lam = np.arange(400, 701, 10)
    S = rs.cfa_sensitivity(lam)
    assert S.shape == (3, len(lam))
    assert lam[np.argmax(S[2])] < lam[np.argmax(S[1])] < lam[np.argmax(S[0])]
    psfs = np.random.default_rng(12).random((2, len(lam), 5, 5))
    C = rs.channel_psfs(psfs, lam)
    assert C.shape == (2, 3, 5, 5)


def test_psf_samples_of_a_paraxial_lens():
    # an ideal lens: the PSF is centred, symmetric, and has no lateral colour
    s = rt.System([rt.IdealLens(5.0, thickness=5.0, semi_diameter=1.0)], 0, 550e-6,
                  field_angle=np.radians(10.0))
    p, shifts = rs.psf_samples(s, [(0.0, 0.0), (0.0, 0.5)], [500e-6, 600e-6], 2e-3, n_pix=15, N=32)
    assert np.allclose(p.sum(axis=(-1, -2)), 1.0)
    assert np.allclose(shifts, 0.0, atol=1e-9)
    c = 7
    assert np.argmax(p[0, 0]) == c*15 + c
    assert np.allclose(p[0, 0], p[0, 0][::-1, ::-1], atol=1e-6)
    f = rs.field_for_image_point(s, 0.2, 0.3)
    assert np.allclose(rs.chief_ray_point(s, f), (0.2, 0.3), atol=1e-8)
    # a finite object: the image is inverted
    s.object_distance = 20.0
    s.max_field = 4.0
    f = rs.field_for_image_point(s, 0.2, 0.3)
    assert f[0] < 0 and f[1] < 0
    assert np.allclose(rs.chief_ray_point(s, f), (0.2, 0.3), atol=1e-8)


def test_dead_leaves_and_star():
    d = rs.dead_leaves((64, 64), r_min=2, rng=13)
    assert d.shape == (64, 64) and 0.1 <= d.min() and d.max() <= 0.9
    st = rs.siemens_star(64, spokes=16)
    assert np.isclose(st.mean(), 0.5, atol=0.01)
