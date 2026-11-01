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
])
def test_sketches_draw(draw):
    fig, ax = plt.subplots()
    draw(ax)
    assert len(ax.lines) + len(ax.patches) > 5
    assert ax.get_aspect() == 1.0
    plt.close(fig)
