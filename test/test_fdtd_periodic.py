#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of periodic FDTD: Bloch boundaries (oblique incidence on a slab
against thin_films), diffraction efficiencies of a lamellar grating against
RCWA, band structures of a 1D photonic crystal (analytic) and of a square
lattice of rods (band gap), and the stability of dispersive media at the
default time step."""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyoptics import fdtd
from pyoptics.litho.rcwa import Layer, rcwa
from pyoptics.thin_films import r_t_coeffs


def _two_runs(grid, structure, mode, bnd, wls, src_y=2.0, mon=(1.5, -1.5), time=40):
    out = []
    for st in ([], structure):
        src = fdtd.LineSource(src_y, fdtd.GaussianPulse(1.0, 0.15), axis=1)
        top = fdtd.DiffractionMonitor(mon[0], wls)
        bot = fdtd.DiffractionMonitor(mon[1], wls)
        sim = fdtd.Simulation(grid, st, [src], [top, bot], mode=mode, boundaries=bnd)
        sim.run(time=time)
        out.append((top, bot))
    (rt, rb), (top, bot) = out
    P = rb.orders().order(0)
    R = -top.orders(rt).power/P[:, None]
    T = bot.orders().power/P[:, None]
    return top.orders().m, R, T


@pytest.mark.parametrize("mode,pol", [("Ez", "s"), ("Hz", "p")])
def test_oblique_slab_against_thin_films(mode, pol):
    """A Bloch wave number k launches plane waves at sin(theta) = k/k0, an
    angle that changes across the pulse; R and T of a slab follow the
    Fresnel/Airy result at each wavelength."""

    n, d = 2.0, 0.4
    k = 2*np.pi*np.sin(np.radians(30))     # 30 degrees at wavelength 1
    wls = np.array([0.9, 1.0, 1.2])
    grid = fdtd.Grid((0.25, 6.0), 0.00625)
    slab = [(fdtd.Slab(-d, 0.0, axis=1), fdtd.Medium.from_index(n))]
    m, R, T = _two_runs(grid, slab, mode, (fdtd.Bloch(k), "pml"), wls)
    i0 = int(np.nonzero(m == 0)[0][0])
    for j, wl in enumerate(wls):
        theta = np.arcsin(k*wl/(2*np.pi))
        r, t = r_t_coeffs(theta, [n], [0], [d], 1.0, 0, 1.0, 0, pol, wl)
        # second-order discretization error, 1e-3 at 1/80 of the wavelength
        assert abs(R[j, i0] - abs(r)**2) < 1.5e-3
        assert abs(T[j, i0] - abs(t)**2) < 1.5e-3
    # only order 0 exists for this period
    assert np.allclose(np.delete(R, i0, axis=1), 0, atol=1e-6)


@pytest.mark.parametrize("mode,theta", [("Ez", 0.0), ("Hz", 0.0), ("Ez", 20.0), ("Hz", 20.0)])
def test_lamellar_grating_against_rcwa(mode, theta):
    P, depth, fill, ng = 1.5, 0.5, 0.5, 1.5
    k = 2*np.pi*np.sin(np.radians(theta))
    wls = np.array([0.9, 1.1])
    grid = fdtd.Grid((P, 6.0), 0.02)
    glass = fdtd.Medium.from_index(ng)
    st = [(fdtd.Slab(-10, -depth, axis=1), glass), (fdtd.Rect((0, -depth/2), (fill*P, depth)), glass)]
    bnd = (fdtd.Bloch(k) if k else "periodic", "pml")
    m, R, T = _two_runs(grid, st, mode, bnd, wls)
    jones = (0, 1) if mode == "Ez" else (1, 0)     # Ez: E along the grooves (TE)
    for j, wl in enumerate(wls):
        res = rcwa(P, [Layer(depth, 1.0, [(0.0, fill*P, ng)])], wl, k_x=k*wl/(2*np.pi), n_in=1.0,
                   n_out=ng, n_orders=25)
        Rr, Tr = res.efficiencies(jones)
        for mm in range(-3, 4):
            i = int(np.nonzero(m == mm)[0][0])
            ir = int(np.nonzero(res.m == mm)[0][0])
            assert abs(R[j, i] - Rr[0, ir]) < 1e-3
            assert abs(T[j, i] - Tr[0, ir]) < 3e-3
        assert abs(R[j].sum() + T[j].sum() - 1) < 4e-3


def test_bands_1d_photonic_crystal():
    """Bloch modes of a quarter-wave-like stack against the analytic
    dispersion relation cos(k L) = cos(a) cos(b) - (n1/n2 + n2/n1)/2 sin(a) sin(b)."""

    n1, n2, d1, d2 = 1.0, 3.0, 0.5, 0.5

    def disp(w, k):
        a, b = n1*w*d1, n2*w*d2
        return np.cos(a)*np.cos(b) - 0.5*(n1/n2 + n2/n1)*np.sin(a)*np.sin(b) - np.cos(k)

    grid = fdtd.Grid(1.0, 0.005)
    st = [(fdtd.Slab(-0.25, 0.25), fdtd.Medium.from_index(n2))]
    for k in (0.4*np.pi, np.pi):
        r = fdtd.bloch_modes(grid, st, k, band=(1/0.9, 1/0.1), time=250)
        ws = np.linspace(2*np.pi*0.1, 2*np.pi*0.9, 4000)
        v = disp(ws, k)
        exact = np.array([brentq(disp, ws[i], ws[i + 1], args=(k,))
                          for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]])/(2*np.pi)
        assert len(r) == len(exact)
        assert np.allclose(r.frequency, exact, rtol=1e-3)


def test_rod_lattice_band_gap():
    """Square lattice of rods (eps = 8.9, r = 0.2 a), Ez polarization: the
    complete gap between bands 1 and 2 (Joannopoulos et al., Photonic
    Crystals, ch. 5: 31.4 % gap-midgap ratio, upper edge 0.44 at X)."""

    grid = fdtd.Grid((1.0, 1.0), 1/32)
    rods = [(fdtd.Circle((0, 0), 0.2), fdtd.Medium(8.9))]
    X = fdtd.bloch_modes(grid, rods, (np.pi, 0), band=(1/0.6, 1/0.2), mode="Ez")
    M = fdtd.bloch_modes(grid, rods, (np.pi, np.pi), band=(1/0.6, 1/0.2), mode="Ez")
    lower = M.frequency[0]         # top of band 1 (at M)
    upper = X.frequency[1]         # bottom of band 2 (at X)
    assert abs(upper - 0.443) < 0.005
    ratio = (upper - lower)/(0.5*(upper + lower))
    assert abs(ratio - 0.314) < 0.01


def test_drude_default_time_step_is_stable():
    """A Drude metal with eps_inf = 1 needs a smaller time step than vacuum;
    the default includes it."""

    metal = fdtd.Medium(1.0, poles=[fdtd.Drude(2*np.pi*np.sqrt(7), 0.45)])
    grid = fdtd.Grid((3, 2), 1/20)
    sim = fdtd.Simulation(grid, [(fdtd.Circle((0, 0), 0.5), metal)],
                          [fdtd.PointSource((-1, 0), fdtd.GaussianPulse(1.0))],
                          boundaries=(fdtd.PML(10), "periodic"), subpixel=1)
    vac = grid.time_step(0.99)
    assert sim.dt < vac
    om = fdtd.simulation.stability_omega(metal)
    assert np.isclose((sim.dt/grid.time_step(1.0))**2 + (om*sim.dt/2)**2, 0.99**2)
    sim.run(steps=3000)
    assert np.isfinite(sim.fields["Ez"]).all() and np.abs(sim.fields["Ez"]).max() < 1e-2
