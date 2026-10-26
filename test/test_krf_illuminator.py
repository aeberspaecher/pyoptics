#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the KrF illuminator: axicon surfaces, the REMA objective of US
5,982,558 (raytrace.zeiss_krf_rema()), the zoom-axicon of US 5,675,401
(raytrace.zeiss_zoom_axicon()) and the whole chain
(litho.illuminator.KrFIlluminator) against the Caldwell lens.

References: the ray angle sines and first order data printed in the REMA
patent, Snell's law for the axicons, the sigma range of the zoom-axicon
patent and the telecentricity the REMA patent asks for (a few mrad).
"""

import numpy as np
import pytest

from pyoptics.litho.illuminator import KrFIlluminator
from pyoptics.raytrace import (Axicon, Rays, Surface, System, caldwell_krf_lens,
                               zeiss_krf_rema, zeiss_zoom_axicon)


def _plate_pair(gap, slope=0.5):
    """Glass, concave cone, air gap, convex cone, glass, plane exit."""

    n = 1.5
    return System([Surface(np.inf, 5.0, n_after=n, semi_diameter=50.0),
                   Axicon(slope, gap, n_after=1.0),
                   Axicon(slope, 5.0, n_after=n),
                   Surface(np.inf, 10.0, n_after=1.0)], 0, 500e-6, object_distance=10.0)


def _collimated(y, z0=-10.0):
    y = np.atleast_1d(y)
    P = np.stack([0*y, y, np.full(len(y), z0)], axis=-1)
    d = np.tile([0.0, 0.0, 1.0], (len(y), 1))
    return Rays(P, d, np.zeros(len(y)))


def test_axicon_pair_in_contact_is_a_plate():
    s = _plate_pair(0.0)
    b = s.trace(_collimated([0.5, 3.0, -7.0]), clip=False)
    assert np.allclose(b.directions[-1], [0, 0, 1], atol=1e-12)
    assert np.allclose(b.points[-1][:, 1], [0.5, 3.0, -7.0], atol=1e-9)


def test_axicon_pair_shifts_rays_outwards():
    gap, slope, n = 20.0, 0.5, 1.5
    s = _plate_pair(gap, slope)
    y = np.array([2.0, 5.0, -4.0])
    b = s.trace(_collimated(y), clip=False)
    # Snell at the cone: incidence atan(slope) from glass to air
    i = np.arctan(slope)
    t = np.arcsin(n*np.sin(i))
    shift = gap*np.tan(t - i)
    assert np.allclose(b.directions[-1], [0, 0, 1], atol=1e-12)
    # the second cone sits further out (sag), the shift is that of the gap
    # plus the cone geometry; the ray leaves parallel and further out:
    r_out = np.abs(b.points[-1][:, 1])
    assert np.all(r_out > np.abs(y) + 0.9*shift)
    assert np.all(r_out < np.abs(y) + 1.4*shift)
    # all rays move by the same amount (a pure radial shift):
    assert np.ptp(r_out - np.abs(y)) < 1e-9


def test_axicon_intersection_lies_on_the_cone():
    a = Axicon(-0.3)
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.normal(size=(50, 2)), np.full(50, -5.0)])
    d = np.column_stack([0.1*rng.normal(size=(50, 2)), np.ones(50)])
    d /= np.linalg.norm(d, axis=1)[:, None]
    t = a.intersect(P, d)
    Q = P + t[:, None]*d
    assert np.allclose(Q[:, 2], a.sag(Q[:, 0]**2 + Q[:, 1]**2), atol=1e-12)
    assert np.all(np.isfinite(t))


@pytest.fixture(scope="module")
def rema():
    return zeiss_krf_rema()


def test_rema_first_order_data(rema):
    p = rema.paraxial()
    assert p.magnification == pytest.approx(-4.457, abs=2e-3)
    # object to reticle 1200 mm, as in the text of the patent:
    assert rema.image_z + rema.object_distance == pytest.approx(1200.0, abs=0.01)
    assert rema.surfaces[-1].thickness == 94.55
    assert abs(p.image_distance - 94.55) < 1.0


# sines of the angles of incidence in air printed in Table 1 (RD, O1, H1, U1)
_TABLE_SINES = {4: (-.43, -.19, .28, .65), 11: (.62, .67, .05, -.56),
                16: (-.66, -.62, .03, .69), 23: (.52, .70, .21, -.34),
                25: (.17, .69, .47, .27), 28: (.11, .69, .54, .41),
                30: (-.25, -.59, -.33, -.07), 33: (-.13, -.13, .00, .14)}


def _sines(system, h=-9.56):
    """Sines of the angles of incidence on the air side of each surface of
    the patent's marginal, upper coma, chief and lower coma rays."""

    s = np.array([.6, .6, 0, -.6])
    y = np.array([0, h, h, h])
    P = np.stack([0*y, y, np.full(4, -system.object_distance)], axis=-1)
    b = system.trace(Rays(P, np.stack([0*s, s, np.sqrt(1 - s**2)], axis=-1), np.zeros(4)),
                     clip=False)
    n = system.n
    out = {}
    for i, surf in enumerate(system.surfaces):
        Q = b.points[i + 1] - [0, 0, system.vertex_z[i]]
        d = b.directions[i] if n[i] < 1.2 else b.directions[i + 1]
        c = np.cross(d, surf.normal(Q))
        out[int(surf.label)] = np.sign(c[:, 0])*np.linalg.norm(c, axis=1)
    out[33] = b.directions[-1][:, 1]
    return out


def test_rema_reproduces_the_ray_sines_of_the_patent(rema):
    sines = _sines(rema)
    for number, printed in _TABLE_SINES.items():
        assert np.allclose(sines[number], printed, atol=0.016), number


def test_rema_misprint_shows_in_the_sines():
    sines = _sines(zeiss_krf_rema(thickness_19=36.53))
    err = max(np.abs(sines[k] - v).max() for k, v in _TABLE_SINES.items())
    assert err > 0.04


def test_rema_aperture(rema):
    # stop from the 0.6 marginal ray; image side NA about 0.6/4.457
    s = 0.6
    P = np.array([[0.0, 0.0, -rema.object_distance]])
    b = rema.trace(Rays(P, np.array([[0, 0.99*s, np.sqrt(1 - (0.99*s)**2)]]), np.zeros(1)))
    assert b.valid[0]
    b = rema.trace(Rays(P, np.array([[0, 1.01*s, np.sqrt(1 - (1.01*s)**2)]]), np.zeros(1)))
    assert not b.valid[0]
    b = rema.trace(Rays(P, np.array([[0, 0.999*s, np.sqrt(1 - (0.999*s)**2)]]), np.zeros(1)))
    assert abs(b.directions[-1][0, 1]) == pytest.approx(0.6/4.457, abs=0.004)


def _pupil_scale(system):
    """Height in plane 30 per direction sine of a ray from the source centre."""

    u = 1e-4
    rays = Rays(np.array([[0, 0, -system.object_distance]]), np.array([[0, u, np.sqrt(1 - u*u)]]),
                np.zeros(1))
    return system.trace(rays, clip=False).points[-1][0, 1]/u


def test_zoom_axicon_zoom_range():
    small = abs(_pupil_scale(zeiss_zoom_axicon(0.0)))
    large = abs(_pupil_scale(zeiss_zoom_axicon(1.0)))
    assert 2.7 < large/small < 3.3
    # the fused silica adaptation keeps the first order data of the i-line
    # design (each element keeps its power):
    for zoom in (0.0, 1.0):
        adapted = _pupil_scale(zeiss_zoom_axicon(zoom))
        original = _pupil_scale(zeiss_zoom_axicon(zoom, original=True, wavelength=365e-6))
        assert adapted == pytest.approx(original, rel=0.03)


def test_zoom_axicon_length_is_fixed():
    lengths = [zeiss_zoom_axicon(z, g).image_z for z, g in [(0, 0), (1, 0), (0.3, 50), (1, 80)]]
    assert np.allclose(lengths, lengths[0])
    with pytest.raises(ValueError):
        zeiss_zoom_axicon(0.0, 200.0)
    with pytest.raises(ValueError):
        zeiss_zoom_axicon(1.2)


@pytest.fixture(scope="module")
def caldwell():
    return caldwell_krf_lens()


def test_sigma_range_of_the_zoom():
    lo = KrFIlluminator(zoom=0.0, n_rays=30000).sigma()[1]
    hi = KrFIlluminator(zoom=1.0, n_rays=30000).sigma()[1]
    assert lo == pytest.approx(0.32, abs=0.02)
    assert hi == pytest.approx(0.83, abs=0.02)


def test_for_sigma_conventional_and_annular():
    il = KrFIlluminator(n_rays=30000).for_sigma(0.8)
    assert il.axicon_gap == 0.0
    assert il.sigma()[1] == pytest.approx(0.8, abs=0.01)
    il.for_sigma(0.8, 0.5)
    assert il.axicon_gap > 10
    assert np.allclose(il.sigma(), [0.5, 0.8], atol=0.015)
    with pytest.raises(ValueError):
        il.for_sigma(0.2)


def test_rod_makes_the_fill_symmetric_and_the_field_uniform():
    il = KrFIlluminator(zoom=0.6, n_rays=60000)
    # the diffuser fills the rod entrance, its overfill is lost:
    assert 0.75 < il.rod_exit_rays()[2] < 0.95
    assert KrFIlluminator(zoom=0.6, n_rays=2000, diffuser_angle=0).rod_exit_rays()[2] == 1.0
    fill = il.pupil_fill((3.0, -2.0), window=2.0, n_samples=21)
    # as much light from the left as from the right, from above as from below:
    for s in (fill.s_x, fill.s_y):
        assert abs(np.sum(fill.weights*(s > 1e-9)) - np.sum(fill.weights*(s < -1e-9))) < 0.05
    centroid = [np.sum(fill.weights*fill.s_x), np.sum(fill.weights*fill.s_y)]
    assert np.allclose(centroid, 0, atol=0.02)
    _, non_uniformity = il.uniformity(n_bins=4)
    assert non_uniformity < 0.08


def test_illumination_matches_caldwell_telecentricity(caldwell):
    il = KrFIlluminator(zoom=1.0, n_rays=150000)
    for field in [(0.0, 0.0), (8.0, 0.0), (7.0, 7.0)]:
        # statistical error of the centroid about 1 mrad:
        mismatch = il.telecentricity(caldwell, field, window=1.2)
        assert np.all(np.abs(mismatch) < 5e-3), field
    assert il.pupil_fill((0.0, 0.0), window=2.0).sigma_max < 0.9
