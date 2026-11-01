#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.fresnel (Fresnel lenses, facets and kinoforms)."""

import numpy as np
import pytest

from pyoptics import fresnel as fr
from pyoptics.fresnel import FresnelSurface, FACET, RISER, SHADOWED, TIR
from pyoptics.raytrace import Surface, System


WL = 550e-6
N_PMMA = float(fr.pmma(550.0))
F, A, T = 120.0, 50.0, 3.0


@pytest.fixture(scope="module")
def parent():
    """Aberration free parent for a collimated beam through a PMMA plate,
    flat side first, grooves on the exit side (a solar concentrator)."""
    r = np.linspace(0, A, 400)
    s, rms = fr.surface_from_slopes(r, fr.focusing_slopes(r, F, N_PMMA, 1.0), n_aspheric=6)
    assert rms < 1e-4
    return s


def concentrator(parent, **kwargs):
    kwargs.setdefault("pitch", 0.5)
    fs = FresnelSurface.from_surface(parent, thickness=F, n_after=1.0, semi_diameter=A, **kwargs)
    return System([Surface(np.inf, T, material=fr.pmma, semi_diameter=A), fs], 0, WL)


def test_pmma_index():
    assert fr.pmma(587.56) == pytest.approx(1.4906, abs=2e-4)
    assert fr.pmma(486.1) > fr.pmma(656.3)


def test_zone_radii_pitch_and_depth():
    s = Surface(-60.0)
    r = fr.zone_radii(s, 10.0, pitch=0.3)
    assert r[0] == 0 and r[-1] == 10.0
    assert np.allclose(np.diff(r)[:-1], 0.3)
    r = fr.zone_radii(s, 10.0, depth=0.05)
    z = np.abs(s.sag(r[1:-1]**2))
    assert np.allclose(z, 0.05*np.arange(1, len(r) - 1), rtol=0, atol=1e-10)
    assert np.allclose(fr.zone_radii(s, 10.0, n_zones=4), [0, 2.5, 5, 7.5, 10])
    with pytest.raises(ValueError):
        fr.zone_radii(s, 10.0, pitch=0.3, depth=0.1)


def test_profile_collapses_sag():
    fs = FresnelSurface(-60.0, pitch=0.5, radius=10.0)
    g = fs._geometry()
    r = np.linspace(0, 10, 5001)
    z = fs.profile(r)
    # every facet starts on the base plane and carries the slopes of the parent:
    assert np.allclose(fs.profile(g["r"][:-1] + 1e-9), 0, atol=1e-8)
    assert z.max() <= 1e-12 and z.min() >= g["h"].min() - 1e-12
    k = 7
    rr = np.linspace(g["r"][k] + 0.01, g["r"][k + 1] - 0.01, 9)
    assert np.allclose(np.gradient(fs.profile(rr), rr),
                       np.gradient(Surface(-60.0).sag(rr**2), rr), rtol=1e-6)
    # the relief is much thinner than the parent:
    assert np.ptp(z) < 0.11*abs(Surface(-60.0).sag(100.0))


def test_round_trip_parent(parent):
    fs = FresnelSurface.from_surface(Surface(-40.0, 5.0, n_after=1.5, conic=-1.2,
                                             aspheric=(1e-6,), semi_diameter=8.0), pitch=0.4)
    back = fs.parent_surface()
    assert (back.R, back.thickness, back.n_after, back.conic, back.aspheric,
            back.semi_diameter) == (-40.0, 5.0, 1.5, -1.2, (1e-6,), 8.0)
    assert type(back) is Surface


def test_surface_from_slopes_recovers_asphere():
    s0 = Surface(-80.0, aspheric=(2e-6, -3e-10))
    r = np.linspace(0, 20, 200)
    slope = s0._slope_over_r(r**2)*r
    s, rms = fr.surface_from_slopes(r, slope, n_aspheric=2)
    assert s.R == pytest.approx(-80.0, rel=1e-8)
    assert np.allclose(s.aspheric, s0.aspheric, rtol=1e-6)
    assert rms < 1e-12


def test_ideal_fresnel_focuses_perfectly(parent):
    sys_ = concentrator(parent, model="ideal")
    assert sys_.paraxial().efl == pytest.approx(F, rel=2e-3)
    assert sys_.spot_diagram(0.0, N=41).rms < 0.01
    # the same parent as a thick plano-convex lens is far from perfect:
    thick = System([Surface(np.inf, T, material=fr.pmma, semi_diameter=A),
                    parent.__class__(parent.R, F, n_after=1.0, aspheric=parent.aspheric,
                                     semi_diameter=A)], 0, WL)
    assert thick.spot_diagram(0.0, N=41).rms > 20*sys_.spot_diagram(0.0, N=41).rms
    assert np.all(np.isnan(sys_.seidel().S[1]))


def test_facets_refract_with_the_parent_normals(parent):
    sys_ = concentrator(parent)
    fs = sys_.surfaces[1]
    r = np.linspace(0.3, 49.7, 50)
    P = np.stack([0*r, r, np.full_like(r, -1.0)], -1)
    d = np.tile([0.0, 0.0, 1.0], (len(r), 1))
    t = fs.intersect(P, d)
    Q = P + t[:, None]*d
    assert np.allclose(Q[:, 2], fs.profile(r))
    assert np.allclose(fs.normal(Q), Surface.normal(fs, Q), atol=1e-12)
    # facets close to the ideal model; finite groove depth costs a little:
    rms = sys_.spot_diagram(0.0, N=41).rms
    assert 0.005 < rms < 0.2


def test_draft_loss_equals_riser_footprints(parent):
    draft = np.radians(3.0)
    sys_ = concentrator(parent, draft=draft)
    fs = sys_.surfaces[1]
    g = fs._geometry()
    L = fr.facet_losses(sys_, 1, n_rays=40000)
    footprint = np.sum(g["e"]**2 - g["r"][1:-1]**2)/A**2
    assert L.fractions["riser"] == pytest.approx(footprint, rel=0.05)
    assert L.fractions["facet"] + L.fractions["riser"] == pytest.approx(1.0)
    assert np.all(L.r[L.code == RISER] >= g["r"][1] - 1e-9)
    # lost in the trace (stray="lose"), traced on with stray="refract":
    b = sys_.trace_pupil(0.0, L.px, L.py)
    assert np.mean(~b.valid) == pytest.approx(L.fractions["riser"], abs=1e-3)
    # in the glass, the rays meet the drafted risers at 87 deg: total internal
    # reflection, so stray="refract" loses them as well
    stray = concentrator(parent, draft=draft, stray="refract").trace_pupil(0.0, L.px, L.py)
    assert np.mean(~stray.valid) == pytest.approx(np.mean(~b.valid))


def test_grooves_facing_the_source_shadow(parent):
    """Grooves on the entrance side: after the facets, the rays in the glass
    hit the next riser (shadowing); on the exit side they do not."""
    r = np.linspace(0, A, 400)
    # facet in air -> glass, aiming (inside the glass) at the focus behind a 3 mm plate:
    slope = fr.focusing_slopes(r, F/N_PMMA, 1.0, N_PMMA)
    p, _ = fr.surface_from_slopes(r, slope, n_aspheric=6)
    fs = FresnelSurface.from_surface(p, pitch=0.5, thickness=T, material=fr.pmma,
                                     semi_diameter=A)
    # (the stop on the relief: no ray aiming needed for a collimated beam)
    sys_in = System([fs, Surface(np.inf, F/N_PMMA, n_after=1.0, semi_diameter=A)], 0, WL,
                    ray_aiming=False)
    L_in = fr.facet_losses(sys_in, 0, n_rays=5000)
    L_out = fr.facet_losses(concentrator(parent), 1, n_rays=5000)
    assert L_in.fractions["shadowed"] > 0.01
    assert L_out.fractions["shadowed"] == 0
    # the shadowed rays are the outer ones (steep facets):
    assert np.median(L_in.r[L_in.code == SHADOWED]) > 0.5*A


def test_vertical_riser_hit_at_oblique_incidence():
    fs = FresnelSurface(-30.0, n_zones=10, radius=10.0, n_after=1.5)
    g = fs._geometry()
    k = 5
    r0, h = g["r"][k + 1], g["h"][k]
    # a ray crossing the riser wall half way up, coming from outside:
    P = np.array([[0.0, r0 + 1.0, h/2 - 1.0]])
    d = np.array([[0.0, -1.0, 1.0]])/np.sqrt(2)
    code, Q = fs.classify(P, d, 1.0, 1.5)
    assert code[0] == RISER
    assert np.hypot(Q[0, 0], Q[0, 1]) == pytest.approx(r0)
    # lost by default, refracted at the (radial) riser normal with stray="refract":
    assert np.all(np.isnan(fs.deflect(Q, d, 1.0, 1.5)[0]))
    fs.stray = "refract"
    d2 = fs.deflect(Q, d, 1.0, 1.5)[0][0]
    sin_t = abs(d2[2])  # the wall is vertical: z is the tangential component
    assert sin_t == pytest.approx(abs(d[0, 2])/1.5)


def test_tir_at_steep_facet():
    fs = FresnelSurface(-5.0, n_zones=3, radius=4.5, n_after=1.0)
    P = np.array([[0.0, 4.0, -1.0]])
    d = np.array([[0.0, 0.0, 1.0]])
    code, _ = fs.classify(P, d, 1.5, 1.0)
    assert code[0] == TIR


def test_opl_steps_at_zone_edges(parent):
    """The facets model keeps the path difference of the removed material."""
    sys_ = concentrator(parent, pitch=2.0)
    fs = sys_.surfaces[1]
    g = fs._geometry()
    k = 10
    r0 = g["r"][k + 1]
    py = np.array([r0 - 1e-4, r0 + 1e-4])/A
    b = sys_.trace(sys_.rays(0.0, 0*py, py), last=1)
    # optical path to a plane just behind the relief:
    z_plane = T + 0.1
    P, d, opl = b.points[-1], b.directions[-1], b.opl[-1]
    opl = opl + (z_plane - P[:, 2])/d[:, 2]
    # inner ray: |h| less glass, |h|/cos more air
    step = -(N_PMMA - 1/d[0, 2])*abs(g["h"][k])
    assert opl[0] - opl[1] == pytest.approx(step, rel=2e-3)
    # the ideal model has no step:
    ideal = concentrator(parent, pitch=2.0, model="ideal")
    b = ideal.trace(ideal.rays(0.0, 0*py, py), last=1)
    P, d, opl = b.points[-1], b.directions[-1], b.opl[-1]
    opl = opl + (z_plane - P[:, 2])/d[:, 2]
    assert abs(opl[0] - opl[1]) < 1e-6


def test_flat_facets_blur_by_the_pitch(parent):
    curved = concentrator(parent, pitch=1.0).spot_diagram(0.0, N=61).rms
    flat = concentrator(parent, pitch=1.0, facets="flat").spot_diagram(0.0, N=61).rms
    assert flat > 2*curved
    assert flat < 1.0  # about the pitch


def test_draw_and_kernel_fallback(parent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys_ = concentrator(parent, pitch=2.0)
    ax = sys_.draw(n_rays=5)
    assert len(ax.lines) > 3
    plt.close("all")
    y, z = sys_.surfaces[1].section(A)
    assert len(y) > 2*sys_.surfaces[1].n_zones


def test_kinoform():
    lam0 = 550e-6
    assert fr.kinoform_depth(lam0, N_PMMA) == pytest.approx(lam0/(N_PMMA - 1))
    assert fr.kinoform_efficiency(1, lam0, lam0, N_PMMA, N_PMMA) == pytest.approx(1.0)
    eta = fr.kinoform_efficiency(np.arange(-3, 5), 450e-6, lam0, fr.pmma(450), N_PMMA)
    assert 0.7 < eta[4] < 1 and eta.sum() <= 1 + 1e-12
    assert fr.diffractive_abbe_number() == pytest.approx(-3.45, abs=0.01)


def test_kinoform_depth_zones_are_one_wave():
    s = Surface(-60.0)
    depth = fr.kinoform_depth(550e-6, N_PMMA)
    fs = FresnelSurface.from_surface(s, depth=depth, radius=3.0)
    g = fs._geometry()
    assert np.allclose(np.abs(g["h"][:-1]), depth, rtol=1e-8)
    # zone radii of a diffractive lens: r_k**2 ~ 2 k lambda f
    f = 60.0/(N_PMMA - 1)
    k = np.arange(1, 6)
    assert np.allclose(g["r"][1:6]**2, 2*k*550e-6*f, rtol=2e-3)
