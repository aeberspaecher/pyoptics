#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the water immersion ArF lens of US 7,187,503 B2 (raytrace.
zeiss_arfi_lens()) and the ray aiming its high NA needs.

References: the first order data of the patent (magnification 1/8, image
distance), its NA and semi diameters (rim rays of the full field), a
diffraction limited wavefront, and the polarization ray trace and tilts
that work on the Caldwell lens.
"""

import numpy as np
import pytest

from pyoptics.litho import lens
from pyoptics.raytrace import SIO2_ARF, zeiss_arfi_lens


@pytest.fixture(scope="module")
def arfi():
    return zeiss_arfi_lens()


def test_first_order_data(arfi):
    p = arfi.paraxial()
    assert p.magnification == pytest.approx(-0.125, abs=1e-5)
    # the table's image distance is 20 nm behind the paraxial image:
    assert p.image_distance == pytest.approx(1.000052369, abs=1e-4)
    assert arfi.image_z + arfi.object_distance == pytest.approx(960.16, abs=0.01)


def test_na_and_semi_diameters(arfi):
    assert lens.lens_NA(arfi, 24) == pytest.approx(1.35, abs=0.005)
    # the rim rays of the full field reach the semi diameters of the table
    # (enlarged by the 1% margin) but none goes beyond:
    sd = np.array([s.semi_diameter for s in arfi.surfaces])
    margin = np.where(np.arange(len(sd)) == arfi.stop_index, 1.0, 1.01)
    phi = np.linspace(0, 2*np.pi, 24, endpoint=False)
    bundle = arfi.trace_pupil((0.0, 56.0), np.cos(phi), np.sin(phi), clip=False)
    h = np.hypot(bundle.points[1:-1, :, 0], bundle.points[1:-1, :, 1])
    assert np.all(bundle.valid)
    reach = np.max(h, axis=1)/(sd/margin)
    assert np.all(reach < 1.001)
    assert np.all(reach[1:] > 0.999)  # (surface 1: dummy plane)


def test_rim_rays_hit_the_stop(arfi):
    # the paraxial guess of the rim rays is far outside the aspheres; the
    # continuation brings them to the stop:
    _, PX, PY, inside = arfi._pupil_grid(24)
    bundle = arfi.trace_pupil((0.0, 56.0), PX[inside], PY[inside])
    assert np.all(bundle.valid)
    stop = bundle.points[arfi.stop_index + 1]
    a_s = arfi.surfaces[arfi.stop_index].semi_diameter
    assert np.allclose(stop[:, 0], a_s*PX[inside], atol=1e-8)
    assert np.allclose(stop[:, 1], a_s*PY[inside], atol=1e-8)


def test_no_rays_on_folded_paths(arfi):
    # Newton's method from the paraxial guess can find rays that hit the
    # stop target on a folded path far outside the clear apertures (here in
    # the middle of the pupil on a 41 x 41 grid); those are aimed again:
    _, PX, PY, inside = arfi._pupil_grid(41)
    bundle = arfi.trace_pupil(0.0, PX[inside], PY[inside])
    assert np.all(bundle.valid)


def test_diffraction_limited(arfi):
    assert arfi.rms_opd(0.0, N=24) < 0.025
    assert arfi.rms_opd((0.0, 56.0), N=24) < 0.035
    ab = lens.lens_aberrations(arfi, (0.0, 7.0), n_terms=36, N=24)
    assert abs(ab.distortion[1]) < 100.0  # nm
    # not telecentric in the image space: 40 mrad (in water) at the rim
    assert ab.telecentricity[1] == pytest.approx(0.040, abs=0.003)


def test_refocus_follows_the_axial_colour(arfi):
    other = zeiss_arfi_lens(193.369e-6)
    dz = other.surfaces[-1].thickness - arfi.surfaces[-1].thickness
    assert dz == pytest.approx(0.3e-3, abs=0.05e-3)  # 0.3 um per pm
    assert other.rms_opd(0.0, N=24) < 0.03
    assert zeiss_arfi_lens(193.369e-6, refocus=False).rms_opd(0.0, N=24) > 0.3


def test_decentre_gives_coma(arfi):
    d = arfi.copy()
    for i in (40, 41):
        d.surfaces[i].decenter = (0.0, 0.005)
    z0 = lens.lens_aberrations(arfi, (0.0, 0.0), n_terms=16, N=16).zernikes
    z1 = lens.lens_aberrations(d, (0.0, 0.0), n_terms=16, N=16).zernikes
    assert abs(z0[8]) < 1e-6
    assert abs(z1[8]) > 0.1
    assert abs(z1[7]) < 0.01*abs(z1[8])


def test_polarization_with_coatings(arfi):
    bare = lens.lens_jones_pupil(arfi, N=11)
    # 43 bare surfaces of about 4% reflection each:
    assert bare.transmission == pytest.approx(0.96**43, rel=0.3)
    coat = lens.quarter_wave_coating(SIO2_ARF, arfi.wavelength, 1.43)
    coated = lens.lens_jones_pupil(arfi, N=11, coating=coat)
    assert coated.transmission > 2*bare.transmission
    # uncoated: radial diattenuator, p (radial) transmitted better at the rim
    rim = np.argmax(bare.P_y)
    assert abs(bare.J[rim, 1, 1]) > abs(bare.J[rim, 0, 0])
