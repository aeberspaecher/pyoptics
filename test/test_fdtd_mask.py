#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the rigorous FDTD mask model (litho.FDTDMask) and what it
needs from pyoptics.fdtd: metals with Re(eps) < 0 at one wavelength and
the diffraction orders of a 2D-periodic cell in 3D."""

import numpy as np
import pytest

from pyoptics import fdtd
from pyoptics.litho import (FDTDMask, Illumination, LithoSimulation, MaskLayout, Projector,
                            RigorousMask, hopkins)

WL, NA, M = 193.0, 1.35, 4.0
CR, MOSI = 0.84 + 1.65j, 2.343 + 0.586j
needs_numba = pytest.mark.skipif(not fdtd.numba_available, reason="needs numba (slow with NumPy)")


def test_metal_from_index():
    for n, k in ((0.84, 1.65), (0.2, 3.0), (0.5, 0.0), (2.343, 0.586)):
        med = fdtd.Medium.from_index(n, k, WL)
        assert med.eps >= 1
        assert abs(med.complex_eps(WL) - complex(n, k)**2) < 1e-12


def test_orders_3d_against_2d():
    """A grating in a 3D cell one cell thick along y gives the orders of the
    same grating in 2D (E along the invariant axis)."""

    P, h, wls = 1.5, 0.025, [1.0, 1.3]
    out = []
    for ndim in (2, 3):
        if ndim == 2:
            st = [(fdtd.Rect((0, 0.06), (0.75, 0.12)), fdtd.Medium.from_index(1.5)),
                  (fdtd.Slab(-3, 0.0, axis=1), fdtd.Medium.from_index(1.5))]
            src = fdtd.LineSource(1.0, fdtd.GaussianPulse(1.0, 0.2), axis=1)
            grid, bnd = fdtd.Grid((P, 4.0), h), (fdtd.Bloch(0.8), "pml")
        else:
            st = [(fdtd.Box((0, 0, 0.06), (0.75, 10, 0.12)), fdtd.Medium.from_index(1.5)),
                  (fdtd.Slab(-3, 0.0, axis=2), fdtd.Medium.from_index(1.5))]
            src = fdtd.PlaneSource(1.0, fdtd.GaussianPulse(1.0, 0.2), axis=2, component="Ey")
            grid, bnd = fdtd.Grid((P, h, 4.0), h), (fdtd.Bloch(0.8), "periodic", "pml")
        mon = fdtd.DiffractionMonitor(-1.0, wls, axis=ndim - 1, max_order=3)
        fdtd.Simulation(grid, st, [src], [mon], boundaries=bnd, dtype=np.complex128).run(time=40)
        out.append(mon.orders())
    o2, o3 = out
    assert o3.components == ["Ex", "Ey"] and list(o3.n) == [0]
    # 3D powers are through the whole cell, 2D powers per unit length in y
    assert np.allclose(o3.power[:, 0, :]/h, o2.power, rtol=0, atol=2e-4*np.abs(o2.power).max())
    assert np.allclose(o3.amplitude[:, 1, 0, :], o2.amplitude, atol=2e-3)
    assert np.allclose(np.abs(o3.amplitude[:, 0]), 0, atol=1e-6)
    assert np.allclose(o3.ka[:, 0, :], o2.ka, atol=1e-2)     # the time steps differ


@needs_numba
@pytest.mark.parametrize("nk, t", [(CR, 70.0), (MOSI, 68.0)])
def test_lines_against_rcwa(nk, t):
    """Chrome and MoSi lines: the propagating orders at normal and oblique
    (conical) incidence agree with RCWA to 1 % of the clear field."""

    fm, rig = FDTDMask([(nk, t)]), RigorousMask([(nk, t)], n_orders=40)
    lay = MaskLayout.lines_spaces(130.0, 65.0)
    m = np.arange(-2, 3)
    s_x, s_y = [0.0, 0.5], [0.0, 0.3]
    Jf = fm.order_jones(lay, m, [0], s_x, s_y, NA, WL, M)
    Jr = rig.order_jones(lay, m, [0], s_x, s_y, NA, WL, M)
    assert Jf.shape == Jr.shape == (2, 2, 2, 1, 5)
    assert np.max(np.abs(Jf - Jr)) < 1e-2
    assert fm.runs == 4
    # cached
    fm.order_jones(lay, m, [0], s_x, s_y, NA, WL, M)
    assert fm.runs == 4


@needs_numba
def test_2d_layout_and_hopkins():
    """Lines described as a 2D layout give the orders of the 1D layout; a
    contact array is symmetric under exchanging x and y with the
    polarization; Hopkins images of lines match those of RCWA."""

    fm = FDTDMask([(CR, 70.0)], spacing=WL/24, incidence="normal")
    m = np.arange(-2, 3)
    J1 = fm.order_jones(MaskLayout.lines_spaces(130.0, 65.0), m, [0], [0.3], [0.2], NA, WL, M)
    J2 = fm.order_jones(MaskLayout(130.0, 20.0, [(0.0, 0.0, 65.0, 20.02)]), m, [0], [0.3], [0.2],
                        NA, WL, M)
    assert np.allclose(J1, J2, atol=1e-3)
    holes = MaskLayout.contact_holes(150.0, 75.0)
    J = fm.order_jones(holes, np.arange(-1, 2), np.arange(-1, 2), [0.0], [0.0], NA, WL, M)[0]
    assert np.allclose(J[0, 0], J[1, 1].T, atol=2e-3)
    assert np.abs(J[0, 0, 1, 1]) > 0.1 and np.max(np.abs(J[0, 1])) < 0.1*np.abs(J[0, 0, 1, 1])

    lay = MaskLayout.lines_spaces(130.0, 65.0)
    ill = Illumination.annular(0.9, 0.6, n_samples=9, polarization="TE")
    proj = Projector(NA, WL, n=1.437)
    x = np.linspace(-65, 65, 41)
    I_f = hopkins.TCC(LithoSimulation(lay, FDTDMask([(CR, 70.0)], incidence="normal"), ill, proj)).image(x)[0]
    I_r = hopkins.TCC(LithoSimulation(lay, RigorousMask([(CR, 70.0)]), ill, proj)).image(x)[0]
    assert np.max(np.abs(I_f - I_r)) < 1e-2


def test_cells_put_edges_on_nodes():
    from pyoptics.litho.fdtd_mask import _cells
    L, edges = 400.0, [-100.0, 100.0]
    n = _cells(L, edges, 193/40)
    assert round(L/(193/40)) <= n <= 1.25*round(L/(193/40))
    f = ((np.array(edges) + L/2)/(L/n)) % 1
    assert np.all(np.minimum(f, 1 - f) < 0.02)
    assert _cells(L, [], 5.0) == 80
