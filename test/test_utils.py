#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test helpers in pyoptics.utils.
"""

import numpy as np
import pytest

from pyoptics.utils import Ez, grid1d


def test_Ez_tilted_plane_wave():
    """For a plane wave polarized in the x-z plane, div(E) = 0 gives
    E_z = -k_x/k_z E_x.
    """

    wl, n = 0.5, 1.3
    x = grid1d(-10, 10, 64)
    X, Y = np.meshgrid(x, x)
    k_x = 2*np.pi*12/20  # on the frequency grid
    k = 2*np.pi*n/wl
    k_z = np.sqrt(k**2 - k_x**2)

    E_x = np.exp(1j*k_x*X)
    E_y = np.zeros_like(E_x)

    assert np.allclose(Ez(x, x, E_x, E_y, wl, n), -k_x/k_z*E_x)


# padding:

from pyoptics.utils import pad, unpad
from pyoptics.fft import FT


@pytest.mark.parametrize("N, N_pad", [(64, 128), (64, 129), (63, 128), (63, 101)])
def test_pad_keeps_grid_centered(N, N_pad):
    dx = 0.1
    x = dx*(np.arange(N) - N//2)
    y = dx*(np.arange(N + 2) - (N + 2)//2)
    field = np.random.default_rng(0).standard_normal((N + 2, N))

    field_pad, x_pad, y_pad = pad(field, x, y, shape=(N_pad + 2, N_pad))

    assert field_pad.shape == (N_pad + 2, N_pad)
    assert np.isclose(x_pad[N_pad//2], 0.0) and np.isclose(y_pad[(N_pad + 2)//2], 0.0)
    assert np.allclose(np.diff(x_pad), dx)
    # every sample keeps its coordinate:
    i0 = np.argmin(np.abs(x_pad - x[0]))
    j0 = np.argmin(np.abs(y_pad - y[0]))
    assert np.allclose(field_pad[j0:j0 + N + 2, i0:i0 + N], field)
    assert np.sum(field_pad) == pytest.approx(np.sum(field))

    field_back, x_back, y_back = unpad(field_pad, x_pad, y_pad, field.shape)
    assert np.array_equal(field_back, field)
    assert np.allclose(x_back, x) and np.allclose(y_back, y)


def test_pad_interpolates_spectrum():
    """Zero padding a centered field samples its spectrum more finely: every
    second sample of the padded spectrum is the original one."""

    x = grid1d(-5, 5, 32)
    X, Y = np.meshgrid(x, x)
    field = np.exp(-(X - 0.3)**2 - (Y + 0.2)**2/2)
    field_pad, _, _ = pad(field, x, x, factor=2)

    assert np.allclose(FT(field_pad)[::2, ::2], FT(field))


def test_pad_options():
    x = grid1d(-1, 1, 10)
    stack = np.ones((10, 10, 3), dtype=complex)
    stack_pad, _, _ = pad(stack, x, x, factor=1.5, value=2.0)

    assert stack_pad.shape == (15, 15, 3) and stack_pad.dtype == complex
    assert stack_pad[0, 0, 0] == 2.0 and stack_pad[7, 7, 2] == 1.0

    with pytest.raises(ValueError):
        pad(stack, x, x)
    with pytest.raises(ValueError):
        pad(stack, x, x, shape=(5, 5))
