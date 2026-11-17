#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of nodal (nodal aberration theory) against the ray tracer."""

import numpy as np
import pytest

from pyoptics import nodal as nd
from pyoptics import off_axis as oa
from pyoptics import raytrace as rt
from pyoptics import telescopes as tl


DEG = np.radians(1)
TH = 0.25*DEG


@pytest.fixture(scope="module")
def rc():
    return tl.two_mirror_telescope(1000.0, 3000.0, 8000.0, 300.0, "rc", wavelength=550e-6,
                                   field_angle=TH)


@pytest.fixture(scope="module")
def field():
    F = oa.rectangular_field((0, 0), (TH, TH), (3, 3))
    return F, nd.complex_field(F, TH)


def _misaligned(rc, decenter=(0.0, 0.0), tilt=(0.0, 0.0, 0.0), surface=1):
    s = rc.copy()
    s.surfaces[surface].decenter, s.surfaces[surface].tilt = decenter, tilt
    return s


def test_aligned(rc, field):
    F, H = field
    c = nd.nodal_coefficients(rc)
    assert abs(c.W131) < 1e-9 and abs(c.W040) < 1e-9
    assert c.A131 == 0 and c.A222 == 0 and c.B222 == 0
    assert np.allclose(nd.astigmatic_nodes(c), 0)
    Z = oa.field_zernikes(rc, F, 9, N=32)
    P = nd.nat_zernikes(c, H)
    assert np.allclose(Z[:, 4:8], P[:, 4:8], atol=0.01)
    assert np.allclose(Z[:, 3], P[:, 3], atol=0.03)


@pytest.mark.parametrize("decenter, tilt", [((0.0, 1.0), (0.0, 0.0, 0.0)),
                                            ((0.5, -0.3), (0.01*DEG, -0.02*DEG, 0.0)),
                                            ((0.0, 0.0), (0.0, 0.03*DEG, 0.0))])
def test_misaligned_secondary(rc, field, decenter, tilt):
    """Constant coma and binodal astigmatism, within a few per cent."""
    F, H = field
    s = _misaligned(rc, decenter, tilt)
    Z = oa.field_zernikes(s, F, 9, N=32)
    c = nd.nodal_coefficients(s)
    P = nd.nat_zernikes(c, H)
    scale = np.max(np.abs(Z[:, 4:8]))
    assert np.max(np.abs(Z[:, 4:8] - P[:, 4:8])) < 0.04*scale
    # focus: with the tilt of the beam against the image plane
    P4 = P[:, 3] + nd.boresight_focus(s, F)
    assert np.max(np.abs(Z[:, 3] - P4)) < 0.01*np.max(np.abs(Z[:, 3]))
    fit = nd.fit_field(H, Z)
    assert abs(fit.A131 - c.A131) < 0.03*abs(c.A131)
    assert abs(fit.A222 - c.A222) < 0.05*abs(c.A222) + 1e-3
    assert np.allclose(np.sort_complex(nd.astigmatic_nodes(fit)),
                       np.sort_complex(nd.astigmatic_nodes(c)), atol=3e-3)


def test_coma_free_pivot(rc, field):
    F, H = field
    z_p = nd.coma_free_pivot(rc, 1)
    a = 0.02*DEG
    s = _misaligned(rc, (0.0, z_p*a), (a, 0.0, 0.0))
    Z = oa.field_zernikes(s, F, 9, N=32)
    tilted = oa.field_zernikes(_misaligned(rc, tilt=(a, 0.0, 0.0)), F, 9, N=32)
    aligned = oa.field_zernikes(rc, F, 9, N=32)
    coma = np.abs((Z - aligned)[:, 6] + 1j*(Z - aligned)[:, 7])
    assert np.max(coma) < 0.02*np.max(np.abs(tilted[:, 6] + 1j*tilted[:, 7]))
    # the astigmatic nodes stay split
    n = nd.astigmatic_nodes(nd.nodal_coefficients(s))
    assert abs(n[1] - n[0]) > 0.05
    # x tilts: the same pivot
    assert np.isclose(nd.coma_free_pivot(rc, 1, axis=1), z_p, rtol=1e-6)


def test_element_in_front_of_the_stop():
    """The optical axis ray: displacements in front of the stop."""
    s = oa.cook_tma(efl=600.0, f_number=12.0, field_angle=DEG, prescription=(-2/3, 0.2, -1/3, 4/9))
    F = oa.rectangular_field((0, 0), (DEG, DEG), (3, 3))
    H = nd.complex_field(F, DEG)
    Z0 = oa.field_zernikes(s, F, 9, N=32)
    for dec, tilt in [((0.0, 0.02), (0.0, 0.0, 0.0)), ((0.0, 0.0), (1e-4, 0.0, 0.0))]:
        m = s.copy()
        m.surfaces[0].decenter, m.surfaces[0].tilt = dec, tilt
        dZ = oa.field_zernikes(m, F, 9, N=32) - Z0
        P = nd.nat_zernikes(nd.nodal_coefficients(m), H)
        assert np.max(np.abs(dZ[:, 4:8] - P[:, 4:8])) < 0.1*np.max(np.abs(dZ[:, 4:8]))


def test_aperture_offset():
    """The linear astigmatism of an off-axis paraboloid from the parent's coma."""
    s = oa.off_axis_parabola(500.0, 150.0, 100.0, field_angle=DEG)
    F = oa.rectangular_field((0, 0), (DEG, DEG), (3, 3))
    H = nd.complex_field(F, DEG)
    Z = oa.field_zernikes(s, F, 9, N=32)
    P = nd.nat_zernikes(nd.nodal_coefficients(s), H, 9, pupil_offset=3j)
    assert np.max(np.abs(Z[:, 3:8] - P[:, 3:8])) < 0.05*np.max(np.abs(Z[:, 3:8]))
    # no aperture offset in the closed form and in the sampled form: the same
    c = nd.nodal_coefficients(_misaligned(tl.two_mirror_telescope(
        1000.0, 3000.0, 8000.0, 300.0, "rc", field_angle=TH), (0.0, 1.0)))
    Hs = np.array([0.3 - 0.2j, -1.0 + 0.5j])
    assert np.allclose(nd.nat_zernikes(c, Hs)[:, 3:9],
                       nd.nat_zernikes(c, Hs, 9, pupil_offset=1e-12)[:, 3:9], atol=1e-6)


@pytest.mark.parametrize("surface, j", [(0, 5), (1, 5), (1, 8), (0, 8)])
def test_freeform_terms(rc, field, surface, j):
    """Zernike departures: constant at the stop, field-linear astigmatism
    from coma away from it."""
    F, H = field
    s = rc.copy()
    S = s.surfaces[surface]
    r = oa.footprints(rc, F)[surface].radius
    s.surfaces[surface] = rt.ZernikeSurface(S.R, S.thickness, zernike={j: 2e-4}, norm_radius=r,
                                            conic=S.conic, mirror=True,
                                            semi_diameter=S.semi_diameter)
    dZ = oa.field_zernikes(s, F, 9, N=32) - oa.field_zernikes(rc, F, 9, N=32)
    P = nd.freeform_field_zernikes(s, surface, H, 9)
    assert np.max(np.abs(dZ[:, 3:9] - P[:, 3:9])) < 0.02*np.max(np.abs(dZ[:, 3:9]))
    if surface == 0:  # at the stop: constant over the field
        assert np.ptp(P[:, 4]) < 1e-9 and np.ptp(P[:, 7]) < 1e-9


def test_errors(rc):
    with pytest.raises(ValueError):
        nd.nodal_coefficients(tl.two_mirror_telescope(1000.0, 3000.0, 8000.0, 300.0,
                                                       field_angle=0.0))
    with pytest.raises(TypeError):
        nd.freeform_field_zernikes(rc, 1, [0j])
    s = rc.copy()
    s.surfaces.insert(1, rt.CoordinateBreak(tilt=(0.001, 0, 0)))
    with pytest.raises(NotImplementedError):
        nd.nodal_coefficients(s)
