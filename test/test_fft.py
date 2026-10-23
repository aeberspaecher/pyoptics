#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.fft against the convention exp(i(k.r - omega*t)): the
forward transform has the kernel exp(-i k.r) on centered grids.
"""

import numpy as np
import pytest

from pyoptics.fft import (FT, FT_unitary, inv_FT, inv_FT_unitary, inv_rFT,
                          inv_rFT_unitary, rFT, rFT_unitary)
from pyoptics.utils import frequencies, grid1d


SHAPES = [(8, 8), (6, 10), (7, 9), (6, 5)]


def _centered(N, dx):
    return dx*(np.arange(N) - N//2)


def _field(shape, real=False):
    rng = np.random.default_rng(1)
    f = rng.normal(size=shape)
    return f if real else f + 1j*rng.normal(size=shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_FT_is_sum_with_kernel_exp_minus_ikx(shape):
    Ny, Nx = shape
    x, y = _centered(Nx, 0.4), _centered(Ny, 0.3)
    k_x, k_y = frequencies(x), frequencies(y)
    f = _field(shape)

    reference = np.einsum("jl,nl,mj->mn", f, np.exp(-1j*np.outer(k_x, x)),
                          np.exp(-1j*np.outer(k_y, y)))

    assert np.allclose(FT(f), reference)
    assert np.allclose(inv_FT(reference), f)


@pytest.mark.parametrize("N", [8, 9])
def test_zero_frequency_and_origin_at_N_half(N):
    x = _centered(N, 0.5)
    assert frequencies(x)[N//2] == 0.0

    delta = np.zeros([N, N])
    delta[N//2, N//2] = 1.0
    assert np.allclose(FT(delta), 1.0)  # no phase ramp: the origin is at N//2


def test_grid1d_centered_for_even_N():
    assert grid1d(-2.0, 2.0, 16)[8] == 0.0


def test_shifted_delta_gets_phase_minus_ikx0():
    N, dx = 16, 0.5
    x = _centered(N, dx)
    f = np.zeros([N, N])
    f[N//2, N//2 + 3] = 1.0  # delta at x0 = 3*dx

    spectrum = FT(f)
    assert np.allclose(spectrum[N//2, :], np.exp(-1j*frequencies(x)*3*dx))


@pytest.mark.parametrize("shape", SHAPES)
def test_unitary_transforms(shape):
    f = _field(shape)

    assert np.isclose(np.sum(np.abs(FT_unitary(f))**2), np.sum(np.abs(f)**2))
    assert np.allclose(inv_FT_unitary(FT_unitary(f)), f)


@pytest.mark.parametrize("shape", SHAPES)
def test_real_transforms(shape):
    Ny, Nx = shape
    f = _field(shape, real=True)

    # rFT() is the k_x >= 0 half of FT() (for even Nx plus the Nyquist column):
    assert rFT(f).shape == (Ny, Nx//2 + 1)
    assert np.allclose(rFT(f)[:, :Nx - Nx//2], FT(f)[:, Nx//2:])
    assert np.allclose(inv_rFT(rFT(f), shape), f)
    assert np.allclose(inv_rFT_unitary(rFT_unitary(f), shape), f)
    if Nx % 2 == 0:
        assert np.allclose(inv_rFT(rFT(f)), f)
