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


# rotated and shifted bases, tensor product bases:

from pyoptics.bases import (ChebyshevPolynomials, LegendrePolynomials,
                            TensorProductBasis, _poly_single_index_to_n_m,
                            local_coordinates)


def _n_m_to_index(n, m):
    return next(j for j in range(200) if _poly_single_index_to_n_m(j) == (n, m))


GRID = np.linspace(-2, 2, 121)


def test_local_coordinates():
    u, v = local_coordinates(1.0 + 0.5, 2.0, x0=0.5, angle=np.pi/2)
    assert np.allclose((u, v), (2.0, -1.0))


@pytest.mark.parametrize("j_from, j_to, angle", [(2, 3, np.pi/2), (5, 6, np.pi/4), (7, 8, np.pi/2)])
def test_rotated_zernikes(j_from, j_to, angle):
    """Rotating x tilt, 0° astigmatism and x coma gives y tilt, 45° astigmatism and y coma."""

    Z = FringeZernikes(GRID, GRID, R_norm=1.5, x0=0.1, y0=-0.2)
    Z_rot = FringeZernikes(GRID, GRID, R_norm=1.5, x0=0.1, y0=-0.2, angle=angle)

    assert np.allclose(Z_rot.eval_single(j_from), Z.eval_single(j_to), atol=1E-12)


@pytest.mark.parametrize("basis", [
    lambda: FringeZernikes(GRID, GRID, R_norm=1.2, x0=0.3, y0=-0.1, angle=0.4),
    lambda: LegendrePolynomials(GRID, GRID, a=1.2, b=0.7, x0=0.3, y0=-0.1, angle=0.4),
    lambda: ChebyshevPolynomials(GRID, GRID, a=1.2, b=0.7, x0=0.3, y0=-0.1),
])
def test_scattered_evaluation_matches_grid(basis):
    B = basis()
    X, Y = np.meshgrid(GRID, GRID)
    for j in (1, 4, 7, 12):
        on_grid = B.eval_single(j)
        scattered = B.eval_single_scattered(X, Y, j)
        inside = B.mask > 0
        assert np.allclose(on_grid[inside], scattered[inside])


def test_rotated_legendre_by_quarter_turn():
    """L_n(u/a) L_m(v/b) rotated by pi/2 equals (-1)^m L_m(x/b) L_n(y/a)."""

    L_rot = LegendrePolynomials(GRID, GRID, a=1.0, b=1.5, angle=np.pi/2)
    L = LegendrePolynomials(GRID, GRID, a=1.5, b=1.0)

    assert L_rot.mask_x is None and L.mask_x is not None
    for n, m in [(0, 0), (1, 0), (2, 1), (1, 3)]:
        assert np.allclose(L_rot.eval_single(_n_m_to_index(n, m)),
                           (-1)**m*L.eval_single(_n_m_to_index(m, n)), atol=1E-12)


def test_rotated_rectangle_mask():
    L = LegendrePolynomials(GRID, GRID, a=1.0, b=0.2, angle=np.pi/4)
    X, Y = np.meshgrid(GRID, GRID)

    assert L.mask[np.argmin(np.abs(GRID - 0.6)), np.argmin(np.abs(GRID - 0.6))] == 1  # on the diagonal
    assert L.mask[np.argmin(np.abs(GRID + 0.6)), np.argmin(np.abs(GRID - 0.6))] == 0  # on the other one
    assert np.isclose(L.mask.sum()*(GRID[1] - GRID[0])**2, 4*1.0*0.2, rtol=0.05)


def test_fit_rotated_legendre():
    L = LegendrePolynomials(GRID, GRID, a=1.3, b=0.9, x0=0.2, angle=0.3)
    inds, coeffs = [0, 2, 5, 9, 13], [0.5, -1.0, 2.0, 0.3, -0.7]
    field = L(inds, coeffs)

    fitted = L.coeffs_from_field(field, np.arange(15))
    assert np.allclose(fitted[inds], coeffs)

    rng = np.random.default_rng(3)
    x, y = rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300)
    fitted = L.coeffs_from_scattered(x, y, L(inds, coeffs, x, y), np.arange(15))
    assert np.allclose(fitted[inds], coeffs)


def test_tensor_product_basis_is_abstract():
    with pytest.raises(TypeError):
        TensorProductBasis(GRID, GRID)

    class Monomials(TensorProductBasis):
        def eval_1d(self, t, n):
            return t**n

    M = Monomials(GRID, GRID)
    j = _n_m_to_index(2, 1)
    X, Y = np.meshgrid(GRID, GRID)
    assert np.allclose(M.eval_single(j), M.mask*X**2*Y)
