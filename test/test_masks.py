import numpy as np
import pytest

from pyoptics.masks import rectangular_mask, rectangluar_mask


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
