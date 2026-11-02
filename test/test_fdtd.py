#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the FDTD solver against exact results: thin film stacks
(thin_films), the dielectric cylinder (analytical.cylinder_field), PML
reflection against a large reference domain, exact energy conservation in a
PEC box, TFSF leakage, and storage round trips."""

import numpy as np
import pytest

from pyoptics import fdtd
from pyoptics.analytical import cylinder_field
from pyoptics.thin_films import r_t_coeffs


def _stack_R(ns, ts, n_sub, wls):
    return np.array([abs(r_t_coeffs(0, ns, [0]*len(ns), ts, 1, 0, n_sub, 0, "s", w)[0])**2
                     for w in wls])


def test_grid_coordinates_and_index():
    g = fdtd.Grid((2.0, 1.0), 0.1)
    assert g.shape == (20, 10)
    assert np.allclose(g.extent, (-1, 1, -0.5, 0.5))
    x, y = g.coords("Hy", "Ez")
    assert np.isclose(x[0], -0.95) and np.isclose(y[0], -0.5)
    assert g.index((0.0, 0.0), "Ez") == (10, 5)
    assert g.index((0.0, 0.0), "Hz", "Hz") == (10, 5)   # node at (0.05, 0.05)
    assert np.isclose(g.time_step(1.0), 0.1/np.sqrt(2))


def test_shapes():
    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41), indexing="ij")
    ring = fdtd.Circle((0, 0), 1.5) - fdtd.Circle((0, 0), 1.0)
    assert np.array_equal(ring.contains(x, y), fdtd.Annulus((0, 0), 1.0, 1.5).contains(x, y))
    tri = fdtd.Polygon([(0, 0), (1, 0), (0, 1)])
    assert tri.contains(np.array(0.2), np.array(0.2)) and not tri.contains(np.array(0.8), np.array(0.8))
    sq = fdtd.Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    off_edge = ~np.isclose(np.abs(x), 1) & ~np.isclose(np.abs(y), 1)
    assert np.array_equal(sq.contains(x, y)[off_edge], fdtd.Rect((0, 0), (2, 2)).contains(x, y)[off_edge])


def test_medium_from_index():
    m = fdtd.Medium.from_index(1.5, 0.1, wavelength=0.8)
    assert np.isclose(m.index(0.8), 1.5 + 0.1j)


@pytest.mark.parametrize("ppw,tol", [(20, 0.12), (40, 0.03)])
def test_bragg_mirror_1d(ppw, tol):
    """Broadband R of a quarter-wave stack on glass from one pulse; the
    error falls with the square of the cell size."""

    n1, n2, N = 2.3, 1.45, 5
    d1, d2 = 0.25/n1, 0.25/n2
    grid = fdtd.Grid(20.0, 1/n1/ppw)
    structure, x = [], 0.0
    for _ in range(N):
        structure.append((fdtd.Slab(x, x + d1), fdtd.Medium(n1**2)))
        structure.append((fdtd.Slab(x + d1, x + d1 + d2), fdtd.Medium(n2**2)))
        x += d1 + d2
    structure.append((fdtd.Slab(x, 100), fdtd.Medium(1.52**2)))
    wls = np.linspace(0.7, 1.5, 41)
    pulse = fdtd.GaussianPulse(1.0, bandwidth=0.35)

    def run(st):
        fr, ft = fdtd.Flux(-3, wls), fdtd.Flux(x + 2, wls)
        sim = fdtd.Simulation(grid, st, [fdtd.PointSource(-6, pulse)], [fr, ft], boundaries=fdtd.PML(40))
        sim.run(time=60)
        return fr.flux(), ft.flux()

    inc, _ = run([])
    refl, trans = run(structure)
    R, T = 1 - refl/inc, trans/inc
    assert np.abs(R - _stack_R([n1, n2]*N, [d1, d2]*N, 1.52, wls)).max() < tol
    assert np.abs(R + T - 1).max() < 1e-4


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_slab_2d_periodic(mode):
    """A slab at normal incidence in 2D with periodic boundaries along y
    (line source), both polarizations."""

    n, d, dx = 2.0, 0.3, 1/60
    wls = np.linspace(0.8, 1.3, 6)
    grid = fdtd.Grid((10, 4*dx), dx)
    pulse = fdtd.GaussianPulse(1.0, bandwidth=0.25)

    def run(st):
        fr, ft = fdtd.Flux(-1.5, wls), fdtd.Flux(1.5, wls)
        sim = fdtd.Simulation(grid, st, [fdtd.LineSource(-3.0, pulse)], [fr, ft], mode=mode,
                              boundaries=(fdtd.PML(20), "periodic"))
        sim.run(time=40)
        return fr.flux(), ft.flux()

    inc, _ = run([])
    refl, trans = run([(fdtd.Slab(0, d), fdtd.Medium(n**2))])
    R, T = 1 - refl/inc, trans/inc
    assert np.abs(R - _stack_R([n], [d], 1.0, wls)).max() < 0.01
    assert np.abs(R + T - 1).max() < 1e-6


def _cylinder_run(ppw, structure, mode="Ez"):
    grid = fdtd.Grid((6.0, 6.0), 1/1.5/ppw)
    dft = fdtd.DFTField(1.0, components=[mode], region=((-1.9, 1.9), (-1.9, 1.9)))
    sim = fdtd.Simulation(grid, structure, [fdtd.TFSF((-2, 2, -2, 2), fdtd.GaussianPulse(1.0, 0.15))],
                          [dft], mode=mode, boundaries=fdtd.PML(12))
    sim.run(time=45)
    return sim, dft


def test_cylinder_vs_exact_series():
    """Ez scattering by a dielectric cylinder (TFSF plane wave) against the
    exact series of analytical.cylinder_field (its TE case)."""

    R, n = 0.8, 1.5
    _, d0 = _cylinder_run(20, [])
    _, d1 = _cylinder_run(20, [(fdtd.Circle((0, 0), R), fdtd.Medium(n**2))])
    x, y = d0.coords("Ez")
    X, Y = np.meshgrid(x, y, indexing="ij")
    ref = d0.field("Ez")[0][np.argmin(np.abs(x)), np.argmin(np.abs(y))]
    exact = cylinder_field(Y, X, R, n, 1.0)    # cylinder_field propagates along its z = our x
    err = np.abs(d1.field("Ez")[0]/ref - exact)
    assert err.max() < 0.08
    assert np.sqrt(np.mean(err**2)) < 0.025
    # the incident field has the pyoptics phase convention exp(+ikx)
    F0 = d0.field("Ez")[0][:, np.argmin(np.abs(y))]/ref
    assert np.abs(F0 - np.exp(2j*np.pi*x)).max() < 0.02


def _cylinder_series_H(x, z, radius, n_cyl, wl, M=25):
    """Exact total field of a dielectric cylinder for H along its axis (the
    Hz mode): continuous u and du/dr/eps at the surface."""

    from scipy.special import jv, jvp, hankel1, h1vp
    k = 2*np.pi/wl
    kr, phi, xr = k*np.hypot(x, z), np.arctan2(x, z), k*radius
    u = np.zeros(np.shape(kr), complex)
    for m in range(-M, M + 1):
        J, Jp, H, Hp = jv(m, xr), jvp(m, xr), hankel1(m, xr), h1vp(m, xr)
        Ji, Jip = jv(m, n_cyl*xr), jvp(m, n_cyl*xr)
        a = (Jip*J - n_cyl*Ji*Jp)/(n_cyl*Ji*Hp - Jip*H)
        c = (J + a*H)/Ji
        u += np.where(kr < xr, c*jv(m, n_cyl*kr), jv(m, kr) + a*hankel1(m, kr))*1j**m*np.exp(1j*m*phi)
    return u


def test_cylinder_hz_vs_exact_series():
    R, n = 0.8, 1.5
    _, d0 = _cylinder_run(20, [], "Hz")
    _, d1 = _cylinder_run(20, [(fdtd.Circle((0, 0), R), fdtd.Medium(n**2))], "Hz")
    x, y = d0.coords("Hz")
    i, j = np.argmin(np.abs(x)), np.argmin(np.abs(y))
    ref = d0.field("Hz")[0][i, j]*np.exp(-2j*np.pi*x[i])   # incident wave referred to x = 0
    X, Y = np.meshgrid(x, y, indexing="ij")
    err = np.abs(d1.field("Hz")[0]/ref - _cylinder_series_H(Y, X, R, n, 1.0))
    assert np.sqrt(np.mean(err**2)) < 0.04


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_tfsf_no_leakage(mode):
    pulse = fdtd.GaussianPulse(1.0, 0.3)
    sim = fdtd.Simulation(fdtd.Grid((6.0, 6.0), 1/15), [], [fdtd.TFSF((-2, 2, -2, 2), pulse)],
                          mode=mode, boundaries=fdtd.PML(12))
    sim.run(time=pulse.delay + 2.0)   # the pulse is inside the box
    c = "Ez" if mode == "Ez" else "Ey"
    F = np.abs(sim.fields[c])
    x, y = sim.coords(c)
    box = (np.abs(x)[:, None] < 1.9) & (np.abs(y)[None, :] < 1.9)
    outside_box = (np.abs(x)[:, None] > 2.1) | (np.abs(y)[None, :] > 2.1)
    inside, outside = F[box].max(), F[outside_box].max()
    assert inside > 0.1 and outside < 1e-12*inside


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_pml_reflection(mode):
    """Probe signal next to a 12 cell PML against a domain so large that no
    reflection arrives in time: below -80 dB."""

    pulse = fdtd.GaussianPulse(1.0, bandwidth=0.3)

    def run(L):
        p = fdtd.Probe((1.5, 1.0))
        sim = fdtd.Simulation(fdtd.Grid((L, L), 1/20), [], [fdtd.PointSource((0, 0), pulse)], [p],
                              mode=mode, boundaries=fdtd.PML(12))
        sim.run(time=14)
        return p.values

    ref, v = run(30.0), run(4 + 24/20)
    assert 20*np.log10(np.abs(v - ref).max()/np.abs(ref).max()) < -80


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_energy_conservation_pec_box(mode):
    pulse = fdtd.GaussianPulse(1.0, bandwidth=0.3)
    sim = fdtd.Simulation(fdtd.Grid((3, 2.5), 1/20), [(fdtd.Circle((0.3, 0.2), 0.5), fdtd.Medium(2.25))],
                          [fdtd.PointSource((-0.5, 0.1), pulse)], mode=mode, boundaries="pec")
    sim.run(time=2*pulse.end)
    sim.track_energy = True
    W = []
    for _ in range(1000):
        sim.step()
        W.append(sim.energy())
    W = np.array(W)
    assert W[0] > 0 and np.ptp(W)/W.mean() < 1e-12


def test_conductor_absorbs():
    """A lossy slab: R + T < 1 and R, T from the conductivity model match
    thin films with the complex index at the centre wavelength."""

    wl = 1.0
    m = fdtd.Medium.from_index(1.8, 0.05, wavelength=wl)
    grid = fdtd.Grid(16.0, 1/80)
    pulse = fdtd.GaussianPulse(wl, bandwidth=0.2)

    def run(st):
        fr, ft = fdtd.Flux(-2, [wl]), fdtd.Flux(3, [wl])
        sim = fdtd.Simulation(grid, st, [fdtd.PointSource(-4, pulse)], [fr, ft], boundaries=fdtd.PML(40))
        sim.run(time=50)
        return fr.flux(), ft.flux()

    inc, _ = run([])
    refl, trans = run([(fdtd.Slab(0, 1.0), m)])
    R, T = (1 - refl/inc)[0], (trans/inc)[0]
    r, t = r_t_coeffs(0, [1.8], [0.05], [1.0], 1, 0, 1, 0, "s", wl)
    assert abs(R - abs(r)**2) < 0.01 and abs(T - abs(t)**2) < 0.01


def test_snapshots_and_storage(tmp_path):
    snap = fdtd.Snapshots(every=5, region=((-1, 1), (-1, 1)), stride=2, keep_last=4)
    dft = fdtd.DFTField([1.0, 1.2], components=["Ez"])
    probe = fdtd.Probe((0.5, 0.0))
    sim = fdtd.Simulation(fdtd.Grid((3, 3), 1/10), [], [fdtd.PointSource((0, 0), fdtd.GaussianPulse(1.0))],
                          [snap, dft, probe])
    sim.run(steps=40)
    st = snap.stack()
    assert len(st) == 4 and st.frames.dtype == np.float16
    assert st.frames.shape[1:] == (len(st.coords[0]), len(st.coords[1])) == (11, 11)
    assert np.allclose(st.times, sim.dt*np.arange(25, 45, 5))
    st.save(tmp_path/"s.npz")
    st2 = fdtd.SnapshotStack.load(tmp_path/"s.npz")
    assert np.array_equal(st2.frames, st.frames) and st2.component == "Ez"
    d = dft.data()
    d.save(tmp_path/"d.npz")
    d2 = fdtd.DFTFieldData.load(tmp_path/"d.npz")
    assert np.allclose(d2["Ez"], d["Ez"]) and np.allclose(d2.coords["Ez"][1], d.coords["Ez"][1])
    assert len(probe.times) == 40


def test_hdf5_streaming(tmp_path):
    pytest.importorskip("h5py")
    snap = fdtd.Snapshots(every=2, path=str(tmp_path/"run.h5"), dtype=np.float32)
    mem = fdtd.Snapshots(every=2, dtype=np.float32)
    sim = fdtd.Simulation(fdtd.Grid((2, 2), 1/10), [], [fdtd.PointSource((0, 0), fdtd.GaussianPulse(1.0))],
                          [snap, mem], mode="Hz", boundaries="pec")
    sim.run(steps=20)
    a, b = snap.stack(), mem.stack()
    assert a.component == "Hz" and np.array_equal(a.frames, b.frames) and np.allclose(a.times, b.times)
    b.save(tmp_path/"b.h5")
    assert np.array_equal(fdtd.SnapshotStack.load(tmp_path/"b.h5").frames, b.frames)


def test_float32_matches_float64():
    vals = []
    for dtype in (np.float64, np.float32):
        p = fdtd.Probe((0.7, 0.3))
        sim = fdtd.Simulation(fdtd.Grid((3, 3), 1/15), [(fdtd.Circle((0, 0), 0.5), fdtd.Medium(4.0))],
                              [fdtd.PointSource((-0.8, 0), fdtd.GaussianPulse(1.0))], [p], dtype=dtype)
        sim.run(time=8)
        vals.append(p.values)
    assert np.abs(vals[0] - vals[1]).max() < 1e-4*np.abs(vals[0]).max()


def test_plots():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from pyoptics.fdtd import plot as fplot
    snap = fdtd.Snapshots(every=10)
    dft = fdtd.DFTField(1.0, components=["Ez"], region=((-1, 1), (-1, 1)))
    sim = fdtd.Simulation(fdtd.Grid((3, 3), 1/10), [(fdtd.Circle((0, 0), 0.4), fdtd.Medium(2.25))],
                          [fdtd.TFSF((-1, 1, -1, 1), fdtd.GaussianPulse(1.0))],
                          [snap, dft, fdtd.Flux(1.2, [1.0]), fdtd.Probe((0.5, 0.5))])
    sim.run(steps=50)
    fig, ax = plt.subplots(1, 3)
    fplot.plot_structure(sim, ax[0])
    fplot.plot_field(sim, ax=ax[1])
    fplot.plot_field(dft, ax=ax[2], sim=sim, part="intensity", log=True)
    anim = fplot.animate(snap.stack(), sim=sim)
    anim._func(1)
    anim._draw_was_started = True   # silence the "deleted without rendering" warning
    anim = fplot.animate(dft, frames=4)
    anim._func(2)
    anim._draw_was_started = True
    sim1 = fdtd.Simulation(fdtd.Grid(4.0, 0.05), [], [fdtd.PointSource(0, fdtd.GaussianPulse(1.0))],
                           [fdtd.Snapshots(every=2)])
    sim1.run(steps=30)
    fplot.xt_diagram(sim1.monitors[0].stack(), sim=sim1)
    fplot.plot_structure(sim1)
    plt.close("all")


# ---------------------------------------------------------------- milestone 2a

DRUDE = fdtd.Medium(1.0, poles=[fdtd.Drude(2*np.pi*np.sqrt(7), 0.45)])     # eps(1) = -6 + 0.5i


def test_pole_susceptibilities():
    w = 2*np.pi/np.array([0.5, 1.0, 2.0])
    d = fdtd.Drude(10.0, 0.5)
    assert np.allclose(d.chi(w), -100/(w**2 + 0.5j*w))
    lz = fdtd.Lorentz(2.0, 8.0, 0.3)
    assert np.allclose(lz.chi(w), 2*64/(64 - w**2 - 0.3j*w))
    assert np.allclose(fdtd.Debye(3.0, 0.2).chi(w), 3/(1 - 0.2j*w))
    m = fdtd.Medium(2.0, sigma=0.1, poles=[d, lz])
    assert np.allclose(m.complex_eps(1.0), 2 + 0.1j/(2*np.pi) + d.chi(2*np.pi) + lz.chi(2*np.pi))
    assert np.imag(DRUDE.index(1.0)) > 0
    assert np.isclose(fdtd.omega_from_eV(1.0), 2*np.pi/1.23984, rtol=1e-4)   # 1 eV <-> 1.2398 µm


def test_fit_medium():
    wl = np.linspace(0.4, 1.0, 30)
    target = DRUDE.index(wl)
    m = fdtd.Medium.fit(wl, target, n_lorentz=0, drude=True)
    assert m.fit_error < 1e-3*np.max(np.abs(target)**2)
    glass = 1.5 + 0.01/(wl - 0.2) + 0.002j/(wl - 0.25)
    g = fdtd.Medium.fit(wl, glass, n_lorentz=2)
    assert np.abs(g.index(wl) - glass).max() < 2e-3


@pytest.mark.parametrize("medium,d,tol", [(DRUDE, 0.05, 2e-3),
                                         (fdtd.Medium(1.8, poles=[fdtd.Lorentz(0.6, 2*np.pi*1.6, 0.8)]), 0.6, 5e-3),
                                         (fdtd.Medium(2.0, poles=[fdtd.Debye(1.5, 0.1)]), 0.5, 0.03)])  # 1st order
def test_dispersive_slab_1d(medium, d, tol):
    """Broadband R and T of Drude, Lorentz and Debye slabs against thin films
    with the complex index of each wavelength."""

    wls = np.linspace(0.7, 1.4, 15)
    grid = fdtd.Grid(16.0, 1/80)
    pulse = fdtd.GaussianPulse(1.0, 0.3)

    def run(st):
        fr, ft = fdtd.Flux(-2, wls), fdtd.Flux(2, wls)
        fdtd.Simulation(grid, st, [fdtd.PointSource(-4, pulse)], [fr, ft],
                        boundaries=fdtd.PML(40)).run(time=60)
        return fr.flux(), ft.flux()

    inc, _ = run([])
    refl, trans = run([(fdtd.Slab(0, d), medium)])
    n = medium.index(wls)
    exact = [r_t_coeffs(0, [n[i].real], [n[i].imag], [d], 1, 0, 1, 0, "s", w) for i, w in enumerate(wls)]
    assert np.abs(1 - refl/inc - [abs(r)**2 for r, _ in exact]).max() < tol
    assert np.abs(trans/inc - [abs(t)**2 for _, t in exact]).max() < tol


def _cylinder_series(x, z, radius, n_cyl, wl, polarization, M=30):
    """Exact total field of a cylinder (E or H along its axis) for a unit
    plane wave along z; n_cyl may be complex (metals)."""

    from scipy.special import jv, jvp, hankel1, h1vp
    k = 2*np.pi/wl
    kr, phi, xr = k*np.hypot(x, z), np.arctan2(x, z), k*radius
    u = np.zeros(np.shape(kr), complex)
    for m in range(-M, M + 1):
        J, Jp, H, Hp = jv(m, xr), jvp(m, xr), hankel1(m, xr), h1vp(m, xr)
        Ji, Jip = jv(m, n_cyl*xr), jvp(m, n_cyl*xr)
        if polarization == "E":
            a = (n_cyl*Jip*J - Ji*Jp)/(Ji*Hp - n_cyl*Jip*H)
        else:
            a = (Jip*J - n_cyl*Ji*Jp)/(n_cyl*Ji*Hp - Jip*H)
        c = (J + a*H)/Ji
        u += np.where(kr < xr, c*jv(m, n_cyl*kr), jv(m, kr) + a*hankel1(m, kr))*1j**m*np.exp(1j*m*phi)
    return u


def test_metal_cylinder_ez():
    """A Drude cylinder (eps = -6 + 0.5i) in the Ez mode against the exact
    series with the complex index."""

    def run(st):
        dft = fdtd.DFTField(1.0, components=["Ez"], region=((-1.4, 1.4), (-1.4, 1.4)))
        fdtd.Simulation(fdtd.Grid((5, 5), 0.025), st, [fdtd.TFSF((-1.5, 1.5, -1.5, 1.5), fdtd.GaussianPulse(1.0, 0.15))],
                        [dft], boundaries=fdtd.PML(12)).run(time=40)
        return dft

    d0, d1 = run([]), run([(fdtd.Circle((0, 0), 0.4), DRUDE)])
    x, y = d1.coords("Ez")
    F = d1.field("Ez")[0]/d0.field("Ez")[0][np.argmin(np.abs(x)), np.argmin(np.abs(y))]
    X, Y = np.meshgrid(x, y, indexing="ij")
    exact = _cylinder_series(Y, X, 0.4, complex(DRUDE.index(1.0)), 1.0, "E")
    assert np.sqrt(np.mean(np.abs(F - exact)**2)) < 0.006


def test_anisotropic_averaging_hz_converges():
    """Hz mode, glass cylinder: anisotropic averaging beats the arithmetic
    mean and converges with second order."""

    def err(ppw, averaging):
        def run(st):
            dft = fdtd.DFTField(1.0, components=["Hz"], region=((-1.9, 1.9), (-1.9, 1.9)))
            fdtd.Simulation(fdtd.Grid((6.0, 6.0), 1/1.5/ppw), st,
                            [fdtd.TFSF((-2, 2, -2, 2), fdtd.GaussianPulse(1.0, 0.15))], [dft], mode="Hz",
                            boundaries=fdtd.PML(12), averaging=averaging).run(time=45)
            return dft
        d0, d1 = run([]), run([(fdtd.Circle((0, 0), 0.8), fdtd.Medium(2.25))])
        x, y = d0.coords("Hz")
        i, j = np.argmin(np.abs(x)), np.argmin(np.abs(y))
        F = d1.field("Hz")[0]/d0.field("Hz")[0][i, j]*np.exp(2j*np.pi*x[i])
        X, Y = np.meshgrid(x, y, indexing="ij")
        return np.sqrt(np.mean(np.abs(F - _cylinder_series(Y, X, 0.8, 1.5, 1.0, "H"))**2))

    e10, e20 = err(10, "anisotropic"), err(20, "anisotropic")
    assert e20 < 0.025 and e10/e20 > 3.5
    assert err(20, "arithmetic") > 1.3*e20


numba_only = pytest.mark.skipif(not fdtd.numba_available, reason="needs numba")


@numba_only
@pytest.mark.parametrize("ndim,mode,boundaries", [(1, "Ez", "pml"), (2, "Ez", "pml"), (2, "Hz", "pml"),
                                                  (2, "Ez", "pec"), (2, "Hz", "periodic"),
                                                  (2, "Ez", "bloch"), (2, "Hz", "bloch")])
def test_numba_matches_numpy(ndim, mode, boundaries):
    glass = fdtd.Medium(2.0, sigma=0.1, poles=[fdtd.Lorentz(0.5, 20.0, 1.0)])
    out = []
    for backend in ("numpy", "numba"):
        if ndim == 1:
            grid = fdtd.Grid(8.0, 1/30)
            st = [(fdtd.Slab(0, 0.5), DRUDE), (fdtd.Slab(1, 2), glass)]
            src = [fdtd.PointSource(-2, fdtd.GaussianPulse(1.0))]
            bnd = fdtd.PML(10)
        else:
            grid = fdtd.Grid((5, 4), 1/20)
            st = [(fdtd.Circle((0.5, 0), 0.6), DRUDE), (fdtd.Rect((-1, 0.5), (1, 1)), glass)]
            if boundaries in ("periodic", "bloch"):
                src = [fdtd.LineSource(-2, fdtd.GaussianPulse(1.0))]
                bnd = (fdtd.PML(10), "periodic" if boundaries == "periodic" else fdtd.Bloch(1.3))
            else:
                src = [fdtd.PointSource((-1.5, -1), fdtd.GaussianPulse(1.0)),
                       fdtd.TFSF((-2, 2, -1.5, 1.5), fdtd.GaussianPulse(1.0))]
                bnd = fdtd.PML(10) if boundaries == "pml" else "pec"
        sim = fdtd.Simulation(grid, st, src, mode=mode, boundaries=bnd, backend=backend)
        sim.run(steps=200)
        out.append(sim)
    a, b = out
    assert b._backend == "numba"
    scale = max(np.abs(a.fields[c]).max() for c in a.components)
    assert scale > 0
    for c in a.components:
        assert np.allclose(a.fields[c], b.fields[c], rtol=0, atol=1e-12*scale)
