import numpy as np
import pytest
from scipy import ndimage

from pyoptics import lightfield as lf
from pyoptics import photography as ph


def _smooth(shape, seed=0, sigma=2.0):
    r = np.random.default_rng(seed).random(shape)
    return ndimage.gaussian_filter(r, sigma, mode="wrap")


def _shifted_views(img, d, n=5):
    """Views of a plane with disparity d: view (i, j) sees the plane at
    column c0 - (j - cen) d and row r0 - (i - cen) d (periodic shifts)."""
    cen = (n - 1)/2
    V = np.zeros((n, n) + img.shape)
    for i in range(n):
        for j in range(n):
            V[i, j] = ndimage.shift(img, (-(i - cen)*d, -(j - cen)*d), order=5, mode="wrap")
    return V


def _noise_texture():
    g = _smooth((256, 256), 3, 2.0) - _smooth((256, 256), 3, 6.0)
    g = 0.5 + 0.5*np.tanh(2*g/g.std())

    def tex(a, b):
        return ndimage.map_coordinates(g, [b/0.001, a/0.001], order=1, mode="wrap"), 1.0
    return tex


def test_camera_matched():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 7, 10)
    v = 50.0*1000.0/950.0
    assert c.image_distance == pytest.approx(v)
    assert c.aperture == pytest.approx(12.5)
    assert c.microlens_focal_length/c.pitch == pytest.approx(v/12.5)
    assert lf.pixel_pitch(c) == pytest.approx(0.1*(1 + c.microlens_focal_length/v)/7)


def test_disparity():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 7, 10)
    v = c.image_distance
    assert lf.disparity(c, 1000.0) == pytest.approx(0.0, abs=1e-12)
    s = np.array([300.0, 600.0, 2000.0, 1e5])
    d = lf.disparity(c, s)
    # linear in 1/s
    assert lf.view_step(c) == pytest.approx(12.5*(1 + c.microlens_focal_length/v)/7)
    k = c.image_distance*lf.view_step(c)/c.pitch
    assert np.allclose(d, k*(1/s - 1/1000.0))
    assert np.allclose(lf.depth_from_disparity(c, d), s)
    assert lf.disparity(c, np.inf) == pytest.approx(-k/1000.0)


def test_views_roundtrip():
    raw = np.random.default_rng(0).random((35, 42))
    V = lf.subaperture_views(raw, 7)
    assert V.shape == (7, 7, 5, 6)
    assert V[2, 3, 1, 4] == raw[1*7 + 2, 4*7 + 3]
    assert np.array_equal(lf.raw_from_views(V), raw)
    white = np.ones_like(raw)*2
    assert np.allclose(lf.subaperture_views(raw, 7, white), V/2)


@pytest.mark.parametrize("d", [-0.8, 0.35, 1.2])
def test_refocus_recovers_plane(d):
    img = _smooth((64, 64))
    V = _shifted_views(img, d)
    out = lf.refocus(V, d)
    sl = (slice(10, -10), slice(10, -10))
    assert np.abs(out - img)[sl].max() < 1e-3*np.ptp(img)
    err0 = np.abs(lf.refocus(V, 0.0) - img)[sl].max()
    assert err0 > 10*np.abs(out - img)[sl].max()


def test_view_shift():
    img = _smooth((64, 64), 1)
    b = ndimage.shift(img, (0.4, -1.3), order=5, mode="wrap")
    dy, dx = lf.view_shift(img, b)
    assert dy == pytest.approx(0.4, abs=0.01)
    assert dx == pytest.approx(-1.3, abs=0.01)


def test_disparity_map_synthetic():
    img = _smooth((48, 48), 2, 1.5)
    V = _shifted_views(img, 0.42)
    d, cost = lf.disparity_map(V, np.arange(-1, 1.01, 0.1), patch=7)
    assert cost.shape == (21, 48, 48)
    assert np.median(d[8:-8, 8:-8]) == pytest.approx(0.42, abs=0.03)


def test_render_white_and_in_focus():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 9, 6)
    w = lf.render_raw(c, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=32,
                      rng=np.random.default_rng(0))
    # matched: the microlens image is a disk inscribed in the cell
    assert w.mean() == pytest.approx(np.pi/4, abs=0.02)
    assert w.max() == pytest.approx(1.0)
    assert w[4::9, 4::9].min() == pytest.approx(1.0)
    assert w[0::9, 0::9].max() == pytest.approx(0.0)
    # without the neighbours nothing changes when the f-numbers are matched
    w0 = lf.render_raw(c, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=32,
                       rng=np.random.default_rng(0), neighbours=0)
    assert w0.sum() == pytest.approx(w.sum(), rel=1e-2)
    # a faster main lens overlaps the microlens images
    c2 = c._replace(f_number=2.8, aperture=50/2.8)
    w2 = lf.render_raw(c2, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=32,
                       rng=np.random.default_rng(0))
    assert w2.max() > 1.5


def test_render_disparity_matches_theory():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 7, 40)
    tex = _noise_texture()
    white = lf.render_raw(c, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=8,
                          rng=np.random.default_rng(1))
    for s in (600.0, 3000.0):
        raw = lf.render_raw(c, [lf.Layer(s, tex)], samples=8, rng=np.random.default_rng(1))
        V = lf.subaperture_views(raw, 7, white)
        dx = lf.view_shift(V[3, 1], V[3, 5])[1]/-4
        dy = lf.view_shift(V[1, 3], V[5, 3])[0]/-4
        assert dx == pytest.approx(lf.disparity(c, s), abs=0.03)
        assert dy == pytest.approx(lf.disparity(c, s), abs=0.03)


def test_render_occlusion():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 5, 8)
    front = lf.Layer(500.0, lambda a, b: (0.2, 1.0))
    back = lf.Layer(2000.0, lambda a, b: (1.0, 1.0))
    raw = lf.render_raw(c, [back, front], samples=4, rng=np.random.default_rng(0))
    white = lf.render_raw(c, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=4,
                          rng=np.random.default_rng(0))
    V = lf.subaperture_views(raw, 5, white)
    lit = lf.subaperture_views(white, 5) > 0
    assert np.allclose(V[lit], 0.2)


def test_refocused_mtf():
    f = np.linspace(0, 1, 101)
    assert np.allclose(lf.refocused_mtf(f, 0.0), np.abs(np.sinc(f)))
    m = lf.refocused_mtf(f, 0.5)
    assert np.allclose(m, np.abs(np.sinc(f)*np.sinc(0.5*f)))
    w = np.ones(7)
    m2 = lf.refocused_mtf(f, 0.5, 0.0, w)
    assert m2[0] == pytest.approx(1.0)
    assert np.all(m2 <= m + 1e-12)
    assert lf.mtf50(f, lf.refocused_mtf(f, 0.0)) == pytest.approx(0.6034, abs=2e-3)
    assert np.isnan(lf.mtf50(f, np.ones_like(f)))


def test_microlens_images_shift():
    lens = ph.double_gauss(2.8)
    p = lens.paraxial()
    L = lens.image_z - p.xp_z
    f_m = 0.28
    xs, ys, hx, hy = lf.microlens_images(lens, 0.0, 0.1, f_m, n_rays=3000,
                                         rng=np.random.default_rng(0))
    assert abs(ys.mean()) < 2e-3 and abs(xs.mean()) < 2e-3
    # the image of the exit pupil: diameter f_m/N = 0.1 mm
    assert 4*xs.std() == pytest.approx(0.1, rel=0.05)
    xs, ys, hx, hy = lf.microlens_images(lens, np.radians(10.0), 0.1, f_m, n_rays=3000,
                                         rng=np.random.default_rng(0))
    assert ys.mean() == pytest.approx(f_m*hy.mean()/L, rel=0.05)


def test_view_weights():
    c = lf.plenoptic_camera(50.0, 4.0, 1000.0, 0.1, 7, 6)
    w = lf.render_raw(c, [lf.Layer(1000.0, lambda a, b: (1.0, 1.0))], samples=32,
                      rng=np.random.default_rng(0))
    vw = lf.view_weights(w, 7)
    assert vw.shape == (7, 7)
    assert vw[1:6, 1:6].min() == 1
    assert vw[0, 0] == 0 and vw[0, 1] == 0
    assert vw.sum() == 29
    assert lf.aperture_weights(lf.subaperture_views(w, 7)).sum() == 45
