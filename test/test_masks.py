import numpy as np
import pytest

from pyoptics.masks import rectangular_mask, rectangluar_mask, siemens_star


def test_rectangular_mask():
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-1, 1, 5)
    mask, mx, my = rectangular_mask(x, y, 1.0, 0.5, include_1d_output=True)
    assert mask.shape == (5, 9)
    assert mx.sum() == 5 and my.sum() == 3
    assert mask.sum() == 15


def test_misspelled_alias_warns():
    x = np.linspace(-1, 1, 5)
    with pytest.warns(DeprecationWarning):
        old = rectangluar_mask(x, x, 0.5, 0.5)
    assert np.array_equal(old, rectangular_mask(x, x, 0.5, 0.5))


def test_siemens_star():
    x = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x, x)
    star = siemens_star(X, Y, 8, 0.8)
    assert star.shape == (201, 201)
    assert star[100, 150] == 1 and star[100, 199] == 0   # on a bright spoke; outside the disk
    assert abs(star.sum()/(np.pi*0.8**2/0.01**2) - 0.5) < 0.02   # half of the disk is bright
