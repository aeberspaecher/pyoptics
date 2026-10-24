#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the image processors Scaler, CameraNoiser and Centerer and the
helpers utils.center_of_mass() and utils.add_camera_noise().
"""

import numpy as np
import pytest

from pyoptics.processing import CameraNoiser, Centerer, Scaler
from pyoptics.utils import add_camera_noise, center_of_mass, grid1d


def _spot(x, y, x0, y0, w):
    X, Y = np.meshgrid(x, y)
    return np.exp(-((X - x0)**2 + (Y - y0)**2)/w**2)


@pytest.fixture
def grid():
    return grid1d(-5, 5, 64), grid1d(-4, 4, 48)


# Scaler

def test_scaler_peak_intensity_of_field(grid):
    x, y = grid
    E = 3*_spot(x, y, 0, 0, 1)*np.exp(1j*_spot(x, y, 1, 0, 2))

    scaled = Scaler(100.0)(E)
    assert np.isclose(np.max(np.abs(scaled)**2), 100.0)
    # the phase is untouched
    assert np.allclose(np.angle(scaled), np.angle(E))

    scaled = Scaler(2.0, quantity="field")(E)
    assert np.isclose(np.max(np.abs(scaled)), 2.0)


def test_scaler_intensity_input(grid):
    x, y = grid
    I = 7*_spot(x, y, 0, 0, 1)**2

    assert np.isclose(np.max(Scaler(5.0)(I)), 5.0)
    # field strength 2 corresponds to intensity 4
    assert np.isclose(np.max(Scaler(2.0, quantity="field")(I)), 4.0)
    assert np.isclose(np.mean(Scaler(0.3, measure="mean")(I)), 0.3)


def test_scaler_total_power(grid):
    x, y = grid
    dA = (x[1] - x[0])*(y[1] - y[0])
    E = (_spot(x, y, 0, 0, 1) + 0j)

    scaled = Scaler(2.0, measure="total", pixel_area=dA)(E)
    assert np.isclose(np.sum(np.abs(scaled)**2)*dA, 2.0)

    # a field norm of sqrt(P) is the same as a power P
    scaled_f = Scaler(np.sqrt(2.0), measure="total", quantity="field",
                      pixel_area=dA)(E)
    assert np.allclose(scaled, scaled_f)


def test_scaler_real_field(grid):
    """A real valued amplitude is only treated as a field if asked to."""

    x, y = grid
    A = 2*_spot(x, y, 0, 0, 1)

    as_field = Scaler(9.0, image_type="field")(A)
    assert np.isclose(np.max(as_field**2), 9.0)
    as_intensity = Scaler(9.0)(A)
    assert np.isclose(np.max(as_intensity), 9.0)


def test_scaler_stack(grid):
    x, y = grid
    stack = np.dstack([_spot(x, y, 0, 0, 1), 4*_spot(x, y, 0, 0, 1)])

    individual = Scaler(1.0)(stack)
    assert np.allclose(np.max(individual, axis=(0, 1)), 1.0)

    common = Scaler(1.0, per_image=False)(stack)
    assert np.isclose(np.max(common), 1.0)
    assert np.allclose(common[:, :, 1], 4*common[:, :, 0])


def test_scaler_invalid():
    with pytest.raises(ValueError):
        Scaler(1.0, measure="median")
    with pytest.raises(ValueError):
        Scaler(1.0, quantity="amplitude")
    with pytest.raises(ValueError):
        Scaler(1.0, image_type="photons")
    with pytest.raises(ValueError):
        Scaler(1.0)(np.zeros((4, 4)))


# CameraNoiser

def test_gaussian_noise_statistics():
    img = np.full((200, 200), 10.0)

    noisy = CameraNoiser("gaussian", sigma=0.5, rng=1)(img)
    assert np.isclose(np.mean(noisy), 10.0, atol=0.01)
    assert np.isclose(np.std(noisy), 0.5, rtol=0.02)

    noisy = CameraNoiser("gaussian", sigma=0.1, relative=True, rng=1)(img)
    assert np.isclose(np.std(noisy), 1.0, rtol=0.02)

    noisy = CameraNoiser("gaussian", sigma=0.5, rng=1)(img + 0j)
    assert np.iscomplexobj(noisy)
    assert np.isclose(np.std(noisy.imag), 0.5, rtol=0.02)


def test_poisson_noise_statistics():
    """Shot noise: variance equals mean, counts are integers."""

    for n in (0.5, 20.0, 1000.0):
        noisy = CameraNoiser("poisson", rng=2)(np.full((200, 200), n))
        assert np.allclose(noisy, np.round(noisy))
        assert np.isclose(np.mean(noisy), n, rtol=0.02)
        assert np.isclose(np.var(noisy), n, rtol=0.05)


def test_poisson_noise_of_field():
    """Fields are converted to photon numbers |E|**2."""

    E = np.full((200, 200), 10.0*np.exp(0.3j))
    noisy = CameraNoiser("poisson", rng=3)(E)
    assert np.isrealobj(noisy)
    assert np.isclose(np.mean(noisy), 100.0, rtol=0.01)


def test_camera_photon_transfer():
    """Mean and temporal variance of the EMVA 1288 linear model."""

    noiser = CameraNoiser(quantum_efficiency=0.6, dark_current=50.0,
                          exposure_time=0.1, read_noise=3.0, gain=0.5,
                          offset=20.0, rng=4)
    for photons in (0.0, 100.0, 2000.0):
        img = np.full((300, 300), photons)
        noisy = noiser(img)
        assert np.isclose(np.mean(noisy), noiser.mean_signal(photons), rtol=0.01)
        assert np.isclose(np.std(noisy), noiser.temporal_noise(photons),
                          rtol=0.03)


def test_camera_saturation_and_quantization():
    noiser = CameraNoiser(full_well=1000.0, gain=0.1, bit_depth=6, rng=5)
    img = np.array([[0.0, 1e4], [200.0, 5e3]])
    noisy = noiser(img)

    assert np.allclose(noisy, np.round(noisy))
    assert np.all((noisy >= 0) & (noisy <= 63))
    # 1000 electrons * 0.1 DN/e = 100 DN would exceed the 6 bit range
    assert noisy[0, 1] == 63 and noisy[1, 1] == 63

    noiser = CameraNoiser(full_well=1000.0, gain=0.01, rng=5)
    assert noiser(img)[0, 1] == 10.0


def test_camera_fixed_pattern_noise():
    """PRNU and DSNU stay the same from image to image."""

    noiser = CameraNoiser(prnu=0.05, dsnu=2.0, rng=6)
    stack = np.full((100, 100, 40), 400.0)
    noisy = noiser(stack)
    assert noisy.shape == stack.shape

    # averaging removes the temporal noise but not the fixed pattern
    prnu, dsnu = noiser.fixed_pattern((100, 100))
    expected = 400*(1 + prnu) + dsnu
    assert np.std(np.mean(noisy, axis=2) - expected) < 0.25*np.std(expected - 400)
    assert np.isclose(np.std(expected - 400), np.sqrt(20**2 + 2**2), rtol=0.05)

    # a second call reuses the pattern
    assert np.std(np.mean(noiser(stack), axis=2) - expected) < 4.0


def test_camera_noise_reproducible():
    img = np.full((16, 16), 50.0)
    a = CameraNoiser(read_noise=2.0, rng=7)(img)
    b = CameraNoiser(read_noise=2.0, rng=7)(img)
    assert np.array_equal(a, b)


def test_camera_integer_input_stack():
    """Integer input stacks give float output (no truncation)."""

    stack = np.full((8, 8, 3), 100, dtype=int)
    noisy = CameraNoiser("gaussian", sigma=0.3, rng=8)(stack)
    assert noisy.dtype.kind == "f"
    assert not np.allclose(noisy, np.round(noisy))


def test_camera_invalid():
    with pytest.raises(ValueError):
        CameraNoiser("salt_and_pepper")
    with pytest.raises(ValueError):
        CameraNoiser("poisson")(-np.ones((4, 4)))


def test_add_camera_noise():
    img = np.full((50, 50), 30.0)
    assert np.array_equal(add_camera_noise(img, "poisson", rng=9),
                          CameraNoiser("poisson", rng=9)(img))


# center of mass and Centerer

def test_center_of_mass(grid):
    x, y = grid
    assert np.allclose(center_of_mass(_spot(x, y, 1.2, -0.7, 0.5), x, y),
                       (1.2, -0.7))

    # a constant background pulls the centroid towards the origin; a
    # threshold removes it
    img = _spot(x, y, 1.2, -0.7, 0.5) + 0.05
    assert abs(center_of_mass(img, x, y)[0]) < 1.0
    assert np.allclose(center_of_mass(img, x, y, threshold=0.1), (1.2, -0.7),
                       atol=1e-2)

    with pytest.raises(ValueError):
        center_of_mass(np.zeros((48, 64)), x, y)


def test_centerer_single_image(grid):
    x, y = grid
    img = _spot(x, y, 1.3, -0.9, 0.6)

    centered = Centerer(x, y)(img)
    assert np.isrealobj(centered)
    assert np.allclose(center_of_mass(centered, x, y), (0, 0), atol=1e-6)
    assert np.allclose(centered, _spot(x, y, 0, 0, 0.6), atol=1e-6)

    centered = Centerer(x, y, x0=-1.0, y0=0.5)(img)
    assert np.allclose(center_of_mass(centered, x, y), (-1.0, 0.5), atol=1e-6)


def test_centerer_field(grid):
    x, y = grid
    E = _spot(x, y, 1.3, -0.9, 0.6)*np.exp(0.4j)

    centered = Centerer(x, y)(E)
    assert np.iscomplexobj(centered)
    assert np.allclose(centered, _spot(x, y, 0, 0, 0.6)*np.exp(0.4j), atol=1e-6)


def test_centerer_stack(grid):
    x, y = grid
    positions = [(1.0, 0.5), (-0.8, 0.2), (0.3, -1.1)]
    stack = np.dstack([_spot(x, y, px, py, 0.6) for px, py in positions])

    centerer = Centerer(x, y)
    assert np.allclose(centerer.get_shifts(stack), -np.array(positions),
                       atol=1e-6)
    centered = centerer(stack)
    for i in range(3):
        assert np.allclose(center_of_mass(centered[:, :, i], x, y), (0, 0),
                           atol=1e-6)

    # common shift: relative positions are kept, the mean position is centered
    common = Centerer(x, y, per_image=False)
    mean_pos = np.mean(positions, axis=0)
    assert np.allclose(common.get_shifts(stack), -mean_pos, atol=1e-6)
    centered = common(stack)
    for i, (px, py) in enumerate(positions):
        assert np.allclose(center_of_mass(centered[:, :, i], x, y),
                           (px - mean_pos[0], py - mean_pos[1]), atol=1e-6)


def test_centerer_noisy_with_threshold(grid):
    x, y = grid
    rng = np.random.default_rng(10)
    img = 100*_spot(x, y, 1.0, 0.8, 0.5) + 5 + rng.standard_normal((48, 64))

    plain = center_of_mass(Centerer(x, y)(img), x, y, threshold=0.2)
    thresholded = Centerer(x, y, threshold=0.2, iterations=2)(img)
    assert np.hypot(*plain) > 0.3
    assert np.allclose(center_of_mass(thresholded, x, y, threshold=0.2), (0, 0),
                       atol=0.02)
