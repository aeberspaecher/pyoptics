#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

import pytest

from pyoptics.bases import FringeZernikes, NumericallyOrthogonalized
from pyoptics.masks import circular_mask
from pyoptics.utils import scalar_product


def _select_from_indexed(data, index, selected):
    selected = list(selected)
    selected_items = [item for item, ind in zip(data, index) if ind in selected]

    return selected_items


@pytest.fixture
def Z():
    x = np.linspace(-2, +2, 200)
    y = x
    basis = FringeZernikes(x, y, R_norm=1.5, x0=0.1, y0=-0.2)

    return basis


def test_fit_gridded(Z):
    """Test LSTSQ fitting of basis functions.
    """

    coeffs = [1, -1, 2, -2, 3, -3]
    inds = [1, 4, 5, 27, 36, 66]

    sampled = Z(inds, coeffs)

    inds_fitted = indices=np.arange(1,100+1)
    coeffs_fitted = Z.coeffs_from_field(sampled, inds_fitted)
    assert np.all(np.isclose(_select_from_indexed(coeffs_fitted, inds_fitted, inds), coeffs))


def test_fit_scattered(Z):
    """Test fit to scattered data.
    """

    coeffs = [1, -1, 0.5, -0.5]
    inds = [3, 9, 13, 31]

    # generate 300 data points in [-2.5, +2.5]
    x = 5*np.random.rand(300) - 2.5
    y = 5*np.random.rand(300) - 2.5

    inds_fitted = indices=np.arange(1,100+1)
    data = Z(inds, coeffs, x, y)
    coeffs_fitted = Z.coeffs_from_scattered(x, y, data, inds_fitted)
    assert np.all(np.isclose(_select_from_indexed(coeffs_fitted, inds_fitted, inds), coeffs))
    print (np.isclose(_select_from_indexed(coeffs_fitted, inds_fitted, inds), coeffs))


def test_numerically_orthogonalized():
    """Test numerical orthogonalization of basis sets.
    """

    x = np.linspace(-0.7, +0.7, 200)
    y = x
    inds = np.arange(1, 36+1)
    new_mask = circular_mask(x, y, 0.7)
    Z_orth = NumericallyOrthogonalized(x, y, FringeZernikes, inds, new_mask)

    scalar_products = np.zeros([len(inds), len(inds)])
    for i, ind_i in enumerate(inds):
        for j, ind_j in enumerate(inds):
            Zi = Z_orth([ind_i], [1.0])
            Zj = Z_orth([ind_j], [1.0])
            scalar_products[i, j] = scalar_product(Zi, Zj, x, y)
    np.fill_diagonal(scalar_products, 0.0)
    assert np.all(np.isclose(scalar_products.ravel(), 0.0))


if __name__ == '__main__':
    test_fit_gridded(Z())
    test_fit_scattered(Z())
    test_numerically_orthogonalized()
