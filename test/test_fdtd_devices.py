#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the FDTD device tools: the Lattice shape, the guided band of a
photonic-crystal line defect, the Q of a point-defect cavity, and the
complex order amplitudes of the diffraction monitor (the phase library of
metasurfaces) against thin_films."""

import numpy as np

from pyoptics import fdtd
from pyoptics.thin_films import r_t_coeffs


ROD = fdtd.Medium(8.9)


def test_lattice_shape():
    x, y = np.meshgrid(np.linspace(-3, 3, 301), np.linspace(-3, 3, 301), indexing="ij")
    disc = fdtd.Circle((0, 0), 0.2)
    for lat in (fdtd.Lattice(disc, (1, 0), (0, 1), (5, 4), remove=[(0, 0.5)]),
                fdtd.Lattice(disc, (1, 0), (0.5, np.sqrt(3)/2), (5, 5), center=(0.3, -0.2),
                             remove=[(0, 0)], shift={(1, 0): (0.15, 0.0), (-1, 1): (0, -0.1)})):
        ref = np.zeros(x.shape, dtype=bool)
        for p in lat.sites():
            ref |= fdtd.Circle(p, 0.2).contains(x, y)
        assert np.array_equal(lat.contains(x, y), ref)
    assert len(fdtd.Lattice(disc, (1, 0), (0, 1), (5, 4), remove=[(0, 0.5)]).sites()) == 19
    # unbounded along a2
    stripe = fdtd.Lattice(disc, (1, 0), (0, 1), (3, None))
    assert stripe.contains(np.array(0.0), np.array(40.0)) and \
        not stripe.contains(np.array(2.0), np.array(0.0))


def test_line_defect_guided_band():
    """A missing row in the square lattice of rods (Ez) guides a band that
    crosses the gap (0.323 to 0.442): from the dipoles in the defect row of
    a supercell, the strongest mode is the guided one."""

    N = 7
    grid = fdtd.Grid((1.0, float(N)), 1/20)
    rods = [(fdtd.Lattice(fdtd.Circle((0, 0), 0.2), (1, 0), (0, 1), (1, N), remove=[(0, 0)]), ROD)]
    f = []
    for kx in (0.2*np.pi, 0.5*np.pi):
        r = fdtd.bloch_modes(grid, rods, (kx, 0.0), band=(1/0.44, 1/0.31),
                             region=((-0.5, 0.5), (-0.3, 0.3)))
        f.append(r.frequency[np.argmax(r.amplitude)])
    assert 0.323 < f[0] < f[1] < 0.442
    assert abs(f[1] - 0.380) < 0.003


def test_point_defect_cavity_q():
    """A missing rod: a mode inside the gap whose Q grows by a constant
    factor with every ring of rods around it."""

    Q, f = [], []
    for rings in (2, 3):
        n = 2*rings + 1
        st = [(fdtd.Lattice(fdtd.Circle((0, 0), 0.2), (1, 0), (0, 1), (n, n), remove=[(0, 0)]), ROD)]
        pulse = fdtd.GaussianPulse(1/0.39, bandwidth=0.15)
        probe = fdtd.Probe((0.13, 0.07))
        sim = fdtd.Simulation(fdtd.Grid((n + 3.0, n + 3.0), 1/20), st,
                              [fdtd.PointSource((0.1, 0.05), pulse)], [probe])
        sim.run(time=pulse.end + 100)
        sel = probe.times > pulse.end
        r = fdtd.resonances(probe.values[sel], sim.dt, t0=probe.times[sel][0], band=(1/0.44, 1/0.33),
                            min_q=5)
        assert len(r) == 1
        Q.append(r.Q[0])
        f.append(r.frequency[0])
    assert abs(f[1] - 0.3945) < 1e-3
    assert 100 < Q[0] < 140 and 5 < Q[1]/Q[0] < 8


def test_order_amplitude_against_thin_films():
    """The complex amplitude of order 0 behind a slab, divided by that of a
    run without it, is the Airy transmission coefficient times the phase
    of the free path the slab replaces."""

    n, d = 2.0, 0.4
    wls = np.array([0.9, 1.0, 1.2])
    out = []
    for st in ([], [(fdtd.Slab(-d, 0.0, axis=1), fdtd.Medium.from_index(n))]):
        src = fdtd.LineSource(-1.5, fdtd.GaussianPulse(1.0, 0.15), axis=1)
        top = fdtd.DiffractionMonitor(1.0, wls)
        sim = fdtd.Simulation(fdtd.Grid((0.1, 5.0), 0.005), st, [src], [top],
                              boundaries=("periodic", "pml"))
        sim.run(time=40)
        out.append(top.orders().coefficient(0))
    for wl, ratio in zip(wls, out[1]/out[0]):
        t = r_t_coeffs(0, [n], [0], [d], 1.0, 0, 1.0, 0, "s", wl)[1]
        assert abs(ratio - t*np.exp(-2j*np.pi*d/wl)) < 2e-3
