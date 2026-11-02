#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the FDTD waveguide tools: the mode solver against the analytic
slab waveguide, mode sources and monitors (power, phase, backward
leakage in all four directions), coupling between parallel waveguides
against their supermodes, waveguide shapes, and the resonance analysis
against damped sinusoids and the Fabry-Perot modes of a dielectric slab."""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyoptics import fdtd


def _slab_neff(n1, n2, d, wl, mode):
    """Fundamental mode of a symmetric slab; Ez: E along the slab plane
    (TE of the slab), Hz: TM."""

    k0 = 2*np.pi/wl
    r = (n1/n2)**2 if mode == "Hz" else 1.0

    def f(ne):
        kap = k0*np.sqrt(n1**2 - ne**2)
        gam = k0*np.sqrt(ne**2 - n2**2)
        return np.tan(kap*d/2) - r*gam/kap
    lo = np.sqrt(max(n2**2, n1**2 - (np.pi/(k0*d))**2)) + 1e-12
    return brentq(f, lo, n1 - 1e-12)


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_slab_modes_converge(mode):
    n1, n2, d, wl = 3.48, 1.44, 0.25, 1.55
    ref = _slab_neff(n1, n2, d, wl, mode)
    errs = []
    for N in (10, 20, 40):
        h = d/N
        y = (np.arange(-int(2/h), int(2/h)) + 0.5)*h      # slab faces on cell faces
        eps = np.where(np.abs(y) < d/2, n1**2, n2**2)
        yl = np.concatenate([y - h/2, [y[-1] + h/2]])
        el = np.where(np.isclose(np.abs(yl), d/2), (n1**2 + n2**2)/2,
                      np.where(np.abs(yl) < d/2, n1**2, n2**2))
        m = fdtd.waveguide_modes(eps, h, wl, mode, eps_long=el if mode == "Hz" else None)[0]
        errs.append(abs(m.n_eff - ref))
        # unit power and the field relations
        assert np.isclose(0.5*np.sum(m.E*m.H)*h*(-1 if mode == "Ez" else 1), 1.0)
    assert errs[-1] < 1e-3
    assert 3.5 < errs[1]/errs[2] < 4.5        # second order


def test_higher_modes_and_cutoff():
    h = 0.01
    y = np.arange(-300, 300)*h
    eps = np.where(np.abs(y) < 0.5, 2.0**2, 1.5**2)
    ms = fdtd.waveguide_modes(eps, h, 1.0, "Ez", num_modes=10)
    # V = pi d/wl NA: number of guided TE modes = ceil(V/(pi/2))
    V = np.pi*1.0/1.0*np.sqrt(4 - 2.25)
    assert len(ms) == int(np.ceil(V/(np.pi/2)))
    assert np.all(np.diff([m.n_eff for m in ms]) < 0)
    # mode k has k nodes, alternating parity
    for k, m in enumerate(ms):
        core = m.E[np.abs(y) < 0.6]
        assert np.sum(np.diff(np.sign(core[np.abs(core) > 1e-3*np.abs(core).max()])) != 0) == k


@pytest.mark.parametrize("mode,axis,direction", [("Ez", 0, 1), ("Hz", 0, 1), ("Hz", 0, -1),
                                                 ("Ez", 1, -1), ("Hz", 1, 1)])
def test_mode_source_and_monitor(mode, axis, direction):
    h = 0.025
    wl = 1.55
    size = (5.0, 2.5) if axis == 0 else (2.5, 5.0)
    rect = (20, 0.4) if axis == 0 else (0.4, 20)
    wls = np.linspace(1.45, 1.65, 5)
    d = direction
    ext = (-1.0, 1.0)
    src = fdtd.ModeSource(-1.6*d, fdtd.GaussianPulse(wl, 0.08), axis=axis, direction=d,
                          extent=ext)
    behind = fdtd.ModeMonitor(-1.9*d, wls, axis=axis, extent=ext)
    m1 = fdtd.ModeMonitor(-1.2*d, wls, axis=axis, extent=ext, every=4)
    m2 = fdtd.ModeMonitor(1.6*d, wls, axis=axis, extent=ext)
    flux = fdtd.Flux(1.6*d, wls, axis=axis)
    sim = fdtd.Simulation(fdtd.Grid(size, h), [(fdtd.Rect((0, 0), rect),
                                                fdtd.Medium.from_index(3.0))],
                          [src], [behind, m1, m2, flux], mode=mode,
                          background=fdtd.Medium.from_index(1.45))
    sim.run(time=80)
    fwd = 0 if d > 0 else 1
    a1, a2 = m1.amplitudes()[fwd], m2.amplitudes()[fwd]
    P = np.abs(a2)**2
    # lossless straight waveguide: the same power at both monitors, nothing
    # travelling backwards anywhere
    assert np.allclose(np.abs(a1)**2, P, rtol=1e-4)
    assert np.max(np.abs(m2.amplitudes()[1 - fwd])**2) < 1e-6*P.max()
    assert np.max(np.abs(behind.amplitudes()[fwd])**2) < 1e-6*P.max()
    assert np.max(np.abs(behind.amplitudes()[1 - fwd])**2) < 1e-3*P.max()
    # the phase advances by beta*L exactly (the modes are exact on the grid)
    beta = np.array([m.beta for m in m1.modes])
    L = 2.8
    assert np.allclose(np.angle(a2/a1*np.exp(-1j*beta*L)), 0, atol=1e-4)
    # Flux reads cos(beta h/2) of the mode power (interpolated H)
    assert np.allclose(d*flux.flux(), P*np.cos(beta*h/2), rtol=2e-3)


def test_parallel_waveguides_supermodes():
    """Light launched into one of two coupled waveguides is a superposition
    of their even and odd supermodes, which beat with beta_e - beta_o."""

    h = 0.025
    w, gap = 0.4, 0.12
    core = fdtd.Medium.from_index(2.85)
    clad = fdtd.Medium.from_index(1.44)
    yc = (w + gap)/2
    guides = [(fdtd.Rect((0, -yc), (20, w)), core), (fdtd.Rect((0, yc), (20, w)), core)]
    win = (-1.2, 1.2)
    sup = fdtd.section_modes(guides, 1.55, win, h, "Hz", clad, num_modes=2)
    single = fdtd.section_modes(guides[:1], 1.55, (-1.5, 1.5), h, "Hz", clad)[0]
    assert sup[1].n_eff < single.n_eff < sup[0].n_eff
    x0 = -3.0
    src = fdtd.ModeSource(x0, fdtd.ContinuousWave(1.55, ramp=10), extent=(-yc - 0.6, 0.0))
    dft = fdtd.DFTField([1.55], components=["Hz"], start=60.0)
    sim = fdtd.Simulation(fdtd.Grid((7.0, 2.6), h), guides, [src], [dft], mode="Hz",
                          background=clad)
    sim.run(time=90)
    Hz = dft.field("Hz")[0]
    x, y = dft.coords("Hz")
    j = np.round((sup[0].coords - y[0])/h).astype(int)
    assert np.allclose(y[j], sup[0].coords)
    # projections with the weight 1/eps (E_t of the modes): orthogonal
    c = [Hz[:, j] @ m.E for m in sup]
    sel = (x > x0 + 0.3) & (x < 2.5)
    ratio = c[0][sel]/c[1][sel]
    # constant magnitudes, relative phase growing with the beat
    assert np.ptp(np.abs(ratio))/np.mean(np.abs(ratio)) < 0.05
    slope = np.polyfit(x[sel], np.unwrap(np.angle(ratio)), 1)[0]
    assert np.isclose(slope, sup[0].beta - sup[1].beta, rtol=5e-3)


def test_waveguide_shapes():
    x, y = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121), indexing="ij")
    ring = fdtd.Ring((0, 0), 2.0, 0.4)
    assert np.array_equal(ring.contains(x, y), fdtd.Annulus((0, 0), 1.8, 2.2).contains(x, y))
    bend = fdtd.Bend((0, 0), 2.0, 0.4, (0, np.pi/2))
    inside = bend.contains(x, y)
    assert np.all(inside <= ring.contains(x, y))
    assert not np.any(inside & ((x < 0) | (y < 0)))
    assert bend.contains(np.array(2.0*np.cos(0.7)), np.array(2.0*np.sin(0.7)))
    # a straight path is a rectangle
    p = fdtd.Path([(-2, 0.3), (2, 0.3)], 0.4)
    r = fdtd.Rect((0, 0.3), (4, 0.4))
    off = ~np.isclose(np.abs(y - 0.3), 0.2) & ~np.isclose(np.abs(x), 2)
    assert np.array_equal(p.contains(x, y)[off], r.contains(x, y)[off])
    # S-bend: ends at the offset, width measured across the centre line
    pts = fdtd.sbend_points((0, 0), 4.0, 1.0)
    assert np.allclose(pts[0], (0, 0)) and np.allclose(pts[-1], (4, 1))
    sb = fdtd.SBend((0, 0), 4.0, 1.0, 0.3)
    assert sb.contains(np.array(2.0), np.array(0.5)) and not sb.contains(np.array(2.0), np.array(0.8))


def test_resonances_damped_sinusoids():
    dt = 0.01
    t = 50 + np.arange(20000)*dt
    w1, w2 = 2*np.pi/1.55, 2*np.pi/1.53
    f = 3*np.exp(-0.002*t)*np.cos(w1*t + 0.3) + np.exp(-0.0005*t)*np.cos(w2*t - 1)
    r = fdtd.resonances(f, dt, t0=t[0])
    assert len(r) == 2
    # sorted by frequency
    assert np.allclose(r.wavelength, [1.55, 1.53], rtol=1e-9)
    assert np.allclose(r.Q, [w1/0.004, w2/0.001], rtol=1e-6)
    assert np.allclose(r.amplitude, [3, 1], rtol=1e-6)
    assert np.allclose(r.phase, [0.3, -1], atol=1e-6)


def test_slab_cavity_q():
    """A dielectric slab in vacuum: resonances at 2 n L/m with
    Q = pi m/(2 |ln r|), r = (n - 1)/(n + 1)."""

    n, L = 3.0, 1.0
    g = fdtd.Grid(6.0, 0.005)
    probe = fdtd.Probe(0.13)
    pulse = fdtd.GaussianPulse(1.0, 0.25)
    sim = fdtd.Simulation(g, [(fdtd.Slab(-L/2, L/2), fdtd.Medium.from_index(n))],
                          [fdtd.PointSource(-0.21, pulse)], [probe])
    sim.run(time=pulse.end + 60)
    sel = probe.times > pulse.end
    r = fdtd.resonances(probe.values[sel], sim.dt, t0=probe.times[sel][0], band=(0.7, 1.6),
                        min_q=5)
    m = np.round(2*n*L/r.wavelength)
    assert set(m) == {4, 5, 6, 7, 8}
    assert np.allclose(r.wavelength, 2*n*L/m, rtol=2e-3)
    assert np.allclose(r.Q, np.pi*m/(2*np.log((n + 1)/(n - 1))), rtol=1e-2)
