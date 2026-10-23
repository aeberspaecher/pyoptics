#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test helpers in pyoptics.utils.
"""

import numpy as np

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
