#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for the schematic drawings in pyoptics.sketch.

References
----------
E. Hecht, "Optics", 5th ed., Pearson (2017), section 5.2 (thin lenses).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyoptics import sketch


def test_paraxial_ray_focus():
    # a ray parallel to the axis crosses it in the back focal point, in both directions
    for direction in (1, -1):
        z, y = sketch.paraxial_ray([(0.0, 2.0)], -direction*1.0, 0.5, 0.0, direction*2.0)
        assert np.allclose(z, [-direction*1.0, 0.0, direction*2.0])
        assert abs(y[-1]) < 1e-12


def test_paraxial_ray_imaging():
    # thin lens equation: object at 3f is imaged at 1.5f with magnification -1/2
    f = 1.0
    for u in (-0.2, 0.0, 0.3):
        z, y = sketch.paraxial_ray([(0.0, f)], -3*f, 0.4, u, 1.5*f)
        assert np.isclose(y[-1], -0.2)
    # 4f system: point in the front focal plane is imaged inverted, unit magnification
    lenses = [(1.0, 1.0), (3.0, 1.0)]
    for u in (-0.3, 0.1):
        z, y = sketch.paraxial_ray(lenses, 0.0, 0.25, u, 4.0)
        assert len(z) == 4 and np.isclose(y[-1], -0.25)
    # lenses outside the traced range are ignored
    z, y = sketch.paraxial_ray(lenses, 0.0, 0.25, 0.0, 2.0)
    assert len(z) == 3


@pytest.mark.parametrize("draw", [
    sketch.koehler_microscope,
    lambda ax: sketch.koehler_microscope(ax, rays="imaging"),
    sketch.abbe_4f,
    lambda ax: sketch.abbe_4f(ax, period_orders=3, blocked=(-1, 1)),
    sketch.phase_contrast_microscope,
    sketch.dic_microscope,
    sketch.confocal_microscope,
    sketch.fpm_microscope,
    sketch.twyman_green,
    sketch.fizeau_interferometer,
    sketch.shack_hartmann_sensor,
    sketch.phase_diversity_setup,
    sketch.holography_setup,
    lambda ax: sketch.holography_setup(ax, microscope=True),
    sketch.fold_mirror,
    sketch.krf_illuminator,
    sketch.spatial_filter,
    lambda ax: sketch.spatial_filter(ax, pinhole=False),
    sketch.axicon_bessel,
    lambda ax: sketch.axicon_bessel(ax, obstacle=False),
    sketch.cubic_phase_airy,
    sketch.young_extended_source,
    sketch.vcz_setup,
    sketch.stellar_interferometer,
    sketch.michelson_interferometer,
    sketch.hbt_interferometer,
    sketch.vortex_interferometer,
    lambda ax: sketch.vortex_interferometer(ax, reference="spherical"),
    sketch.mode_converter,
    lambda ax: sketch.mode_converter(ax, images=(np.eye(8), np.ones((8, 8)))),
    sketch.wavefront_shaping_setup,
    lambda ax: sketch.wavefront_shaping_setup(ax, medium="slab"),
    lambda ax: sketch.wavefront_shaping_setup(ax, medium="fibre", feedback=False),
    lambda ax: sketch.wavefront_shaping_setup(ax, reference=True),
    sketch.optical_tweezers,
    sketch.ashkin_rays,
    lambda ax: sketch.ashkin_rays(ax, offset=(-0.6, 0.0)),
    lambda ax: sketch.ashkin_rays(ax, offset=(0.0, 0.6)),
    sketch.momentum_balance,
    sketch.lensless_camera,
    lambda ax: sketch.lensless_camera(ax, kind="coded"),
    lambda ax: sketch.lensless_camera(ax, kind="diffuser"),
    sketch.pulse_through_lens,
    lambda ax: sketch.pulse_through_lens(ax, achromat=True),
    sketch.pulse_front_tilt,
    lambda ax: sketch.pulse_front_tilt(ax, element="grating"),
    sketch.oct_setup,
    lambda ax: sketch.oct_setup(ax, kind="time"),
    sketch.twinkling_star,
    sketch.adaptive_optics_system,
    sketch.rainbow_observer,
    lambda ax: sketch.rainbow_observer(ax, sun_elevation=10.0),
    sketch.glory_aircraft,
    sketch.foucault_tester,
    lambda ax: sketch.foucault_tester(ax, test="wire"),
    lambda ax: sketch.foucault_tester(ax, test="ronchi"),
    sketch.schlieren_system,
    lambda ax: sketch.schlieren_system(ax, method="shadowgraph"),
    sketch.grin_rod_imaging,
    sketch.road_mirage,
    sketch.sim_microscope,
    sketch.sted_microscope,
    lambda ax: sketch.sted_microscope(ax, images=(np.ones((8, 8)), np.eye(8), np.eye(8))),
    sketch.approximation_map,
    sketch.sign_conventions,
    sketch.diffraction_geometry,
    sketch.eikonal_wavefronts,
    sketch.phase_space_portraits,
    sketch.liouville_lens,
    sketch.burning_glass,
    lambda ax: sketch.burning_glass(ax, lens="ideal"),
    sketch.etendue_diffuser,
    sketch.light_pipe_images,
    sketch.cpc_edge_rays,
    lambda ax: sketch.cpc_edge_rays(ax, ray_angle=25.0),
    lambda ax: sketch.cpc_edge_rays(ax, truncation=2.0),
    lambda ax: sketch.cpc_edge_rays(ax, acceptance=30.0, n=1.5),
    sketch.cpc_skew_ray,
    sketch.cpc_secondary,
    sketch.koehler_critical,
    lambda ax: sketch.koehler_critical(ax, mode="critical"),
    sketch.fly_eye_homogenizer,
    lambda ax: sketch.fly_eye_homogenizer(ax, imaging=False, tilt=0.1),
    sketch.speckle_formation,
    lambda ax: sketch.speckle_diversity(ax, "polarization"),
    lambda ax: sketch.speckle_diversity(ax, "wavelength"),
    lambda ax: sketch.speckle_diversity(ax, "angle"),
    sketch.laser_projector,
    sketch.led_above_plane,
    lambda ax: sketch.led_above_plane(ax, n_leds=3, spacing=1.0, m=3.0),
    sketch.photometric_distance,
    lambda ax: sketch.led_collimator(ax, "lens"),
    lambda ax: sketch.led_collimator(ax, "reflector"),
    sketch.led_collimator,
    sketch.beam_shaping_ray_map,
    sketch.flattop_beam_shaper,
    sketch.freeform_projection,
    lambda ax: sketch.freeform_projection(ax, image=np.ones((8, 8))),
    sketch.bokeh_formation,
    sketch.spherical_aberration_bokeh,
    sketch.cats_eye_vignetting,
    sketch.rippled_asphere,
    sketch.apodized_lens,
    sketch.iris_diffraction,
    sketch.iris_spikes,
    lambda ax: sketch.iris_spikes(ax, 8),
    sketch.rounded_iris,
    sketch.iris_stop_down,
    sketch.sunstar_scene,
    sketch.ghost_reflections,
    sketch.ar_coating_layers,
    sketch.ghost_focus,
    sketch.flare_scene,
    sketch.pixel_sampling,
    sketch.bayer_cfa,
    sketch.olpf_plates,
    sketch.microlens_pixels,
    sketch.depth_of_field,
    sketch.defocus_and_diffraction,
    sketch.scheimpflug_hinge,
    sketch.dof_wedge,
    sketch.plenoptic_camera,
    sketch.subaperture_parallax,
    sketch.light_field_refocus,
    sketch.plenoptic_tradeoff,
    sketch.coronagraph_layout,
    lambda ax: sketch.coronagraph_layout(ax, mask="vortex", dm=True),
    lambda ax: sketch.telescope_layout(ax, "newtonian"),
    sketch.telescope_layout,
    lambda ax: sketch.telescope_layout(ax, "schmidt"),
])
def test_sketches_draw(draw):
    fig, ax = plt.subplots()
    draw(ax)
    assert len(ax.lines) + len(ax.patches) > 5
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_fold_unfolds_light_pipe():
    # a ray in a light pipe of half width h stays inside and is mirrored at the walls
    h = 0.3
    y = np.linspace(-3, 3, 601)
    yf = sketch._fold(y, h)
    assert np.all(np.abs(yf) <= h + 1e-12)
    assert np.allclose(sketch._fold(np.array([0.1, 0.5, 0.7]), h), [0.1, 0.1, -0.1])


def test_cpc_wall_and_edge_rays():
    # the drawn CPC has a = a'/sin(theta), L = (a + a')/tan(theta), and the
    # edge rays reflected by the upper wall all meet the lower exit rim
    th = np.radians(20)
    z, y, a, L = sketch._cpc_wall(th, 1.0, n=4000)
    assert np.isclose(a, 1/np.sin(th)) and np.isclose(L, (a + 1)/np.tan(th))
    assert np.allclose([z[0], y[0], z[-1], y[-1]], [0, a, L, 1.0])
    up = np.c_[z, y]
    lo = np.c_[z, -y]
    d = np.array([np.cos(th), np.sin(th)])
    for y0 in np.linspace(0.2, 0.95, 5)*a:
        V, k = sketch._mirror_trace([0.0, y0], d, [up, lo], -1e-9, L)
        assert k == 1 and np.allclose(V[-1], [L, -1.0], atol=2e-3)
    # beyond the acceptance angle the rays turn back
    d = np.array([np.cos(1.3*th), np.sin(1.3*th)])
    V, k = sketch._mirror_trace([0.0, 0.0], d, [up, lo], -1e-9, L)
    assert V[-1, 0] < 1e-6


def test_hyperbolic_lens_is_stigmatic():
    a, f = 1.0, 1.0
    z_v = sketch._hyperbolic_lens(a, f)[2]
    for y0 in (0.1, 0.5, 0.95):
        V, e = sketch._lens_ray(y0, 0.0, a, f)
        y_f = V[-1, 1] + (z_v + f - V[-1, 0])/e[0]*e[1]
        assert abs(y_f) < 1e-9


def test_clip_x():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    piece = sketch._clip_x(square, 0.0, 0.5)
    assert np.isclose(np.ptp(piece[:, 0]), 0.5) and np.isclose(np.ptp(piece[:, 1]), 2.0)
    assert len(sketch._clip_x(square, 2.0, 3.0)) == 0


def test_speckle_diversity_rejects_unknown_method():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        sketch.speckle_diversity(ax, "time")
    plt.close(fig)


def test_led_collimator_rejects_unknown_kind():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        sketch.led_collimator(ax, "tir")
    plt.close(fig)


def test_bokeh_formation_blur_disks():
    fig, ax = plt.subplots()
    sketch.bokeh_formation(ax)
    # two blur disks (background and foreground), the subject is sharp
    bars = [p for p in ax.patches if type(p).__name__ == "Rectangle" and p.get_alpha() == 0.85]
    assert len(bars) == 2
    plt.close(fig)
