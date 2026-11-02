#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the 3D FDTD: the Yee core (energy in a PEC box, NumPy against
numba), 3D shapes and sub-cell averaging, the TFSF plane wave, the vector
Mie series and the dielectric sphere against it (V8), flux planes,
snapshots on planes and in HDF5, and the plots of sections."""

import numpy as np
import pytest

from pyoptics import fdtd
from pyoptics import analytical as an


def _small(backend, boundaries, medium=None):
    grid = fdtd.Grid((2.0, 1.6, 1.8), 0.1)
    medium = medium or fdtd.Medium(2.25)
    sources = [fdtd.PointSource((0.05, 0.1, -0.15), fdtd.GaussianPulse(0.8), component=c)
               for c in ("Ex", "Ey", "Ez")]
    return fdtd.Simulation(grid, [(fdtd.Slab(-0.3, 0.2, axis=2), medium)], sources,
                           [fdtd.Probe((0.3, -0.2, 0.4), "Hy")], boundaries=boundaries,
                           backend=backend, dtype=np.float64)


def test_shapes_3d():
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, 41)]*3, indexing="ij")
    assert np.array_equal(fdtd.Sphere((0.1, 0, 0), 0.5).contains(x, y, z),
                          (x - 0.1)**2 + y**2 + z**2 < 0.25)
    assert np.array_equal(fdtd.Box((0, 0, 0.2), (1.0, 0.6, 0.4)).contains(x, y, z),
                          (np.abs(x) <= 0.5) & (np.abs(y) <= 0.3) & (np.abs(z - 0.2) <= 0.2))
    cyl = fdtd.Cylinder((0, 0.2, 0), 0.3, 1.0, axis=1)
    assert np.array_equal(cyl.contains(x, y, z),
                          (x**2 + z**2 < 0.09) & (y >= -0.3) & (y < 0.7))
    prism = fdtd.Prism(fdtd.Rect((0, 0), (0.4, 0.2)), -0.5, 0.5)
    assert np.array_equal(prism.contains(x, y, z),
                          fdtd.Box((0, 0, 0), (0.4, 0.2, 1.0)).contains(x, y, z) & (z < 0.5))


def test_grid_and_averaging_3d():
    grid = fdtd.Grid((2.0, 2.0, 2.0), 0.1)
    assert grid.shape == (20, 20, 20)
    assert np.allclose(grid.coords("Hx", "3D")[2], grid.axis(2, 0.5))
    sim = fdtd.Simulation(grid, [(fdtd.Sphere((0.03, -0.02, 0.01), 0.55), fdtd.Medium(4.0))])
    assert sim.mode == "3D" and sim.dtype == np.float32
    eps, sigma, frac = sim.rasterize("Ez")
    # filling fractions add up to the volume of the sphere
    assert abs(frac[1].sum()*0.1**3/(4/3*np.pi*0.55**3) - 1) < 2e-3
    # only cells crossed by the surface are mixed; normal E sees the
    # harmonic, tangential E the arithmetic mean
    exf = sim.rasterize("Ex")[0]
    cx = sim.coords("Ex")
    i, j, k = np.argmin(np.abs(cx[0] - 0.55)), np.argmin(np.abs(cx[1])), np.argmin(np.abs(cx[2]))
    e_n = exf[i - 1:i + 2, j, k]
    eps_y = sim.rasterize("Ey")[0]
    assert np.all(e_n >= 1) and np.all(e_n <= 4)
    assert np.all((eps >= 1) & (eps <= 4)) and np.all(eps_y >= 1)


def test_energy_pec_box_3d():
    sim = _small("numpy", "pec")
    sim.track_energy = True
    sim.run(time=8.0)      # after the pulse
    w0 = sim.energy()
    sim.run(steps=200)
    assert w0 > 0 and abs(sim.energy()/w0 - 1) < 1e-6


@pytest.mark.skipif(not fdtd.numba_available, reason="needs numba")
@pytest.mark.parametrize("boundaries, medium", [
    ("pec", None),
    (fdtd.PML(4), fdtd.Medium(2.25, sigma=0.3)),
    (("periodic", fdtd.Bloch(1.0), fdtd.PML(4)), None),
    (fdtd.PML(4), fdtd.Medium(1.0, poles=[fdtd.Drude(3.0, 0.3), fdtd.Lorentz(1.0, 2.0, 0.1)])),
])
def test_numba_matches_numpy_3d(boundaries, medium):
    runs = [_small(b, boundaries, medium).run(steps=120) for b in ("numpy", "numba")]
    scale = max(np.max(np.abs(runs[0].fields[c])) for c in runs[0].components)
    for c in runs[0].components:
        assert np.max(np.abs(runs[0].fields[c] - runs[1].fields[c])) < 1e-12*scale


@pytest.mark.parametrize("polarization", ["x", "y"])
def test_tfsf_3d(polarization):
    """Without a scatterer the plane wave stays inside the box; inside, it
    is the incident wave of the auxiliary grid."""

    grid = fdtd.Grid((1.6, 1.6, 1.6), 0.05)
    src = fdtd.TFSF((-0.4, 0.4)*3, fdtd.GaussianPulse(1.0), polarization=polarization)
    sim = fdtd.Simulation(grid, [], [src], [], boundaries=fdtd.PML(6))
    c = "E" + polarization
    leak = peak = 0.0
    for _ in range(10):
        sim.run(time=0.5)
        for comp in sim.components:
            X = np.meshgrid(*sim.coords(comp), indexing="ij")
            out = np.max([np.abs(v) for v in X], axis=0) > 0.45
            leak = max(leak, np.max(np.abs(sim.fields[comp][out])))
        peak = max(peak, np.max(np.abs(sim.fields[c])))
        z = sim.coords(c)[2]
        k = np.argmin(np.abs(z - 0.1))
        i, j = sim.grid.index((0.0, 0.0, z[k]), c, "3D")[:2]
        assert abs(sim.fields[c][i, j, k] - src.incident(z[k])) < 1e-5
    assert peak > 0.4 and leak < 1e-5*peak


def test_mie_series():
    # Bohren & Huffman, appendix A: x = 2 pi 0.525/0.6328, m = 1.55
    q_ext, q_sca = an.mie_efficiencies(0.525, 1.55, 0.6328)
    assert abs(q_ext - 3.10543) < 1e-5 and abs(q_sca - q_ext) < 1e-12
    # boundary conditions: tangential E and normal eps E continuous
    rng = np.random.default_rng(0)
    th, ph = rng.uniform(0, np.pi, 30), rng.uniform(0, 2*np.pi, 30)
    nrm = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    for m in (1.5, 2.0 + 0.3j):
        Eo = an.mie_field(*(nrm*0.4*(1 + 1e-9)), 0.4, m, 1.0)
        Ei = an.mie_field(*(nrm*0.4*(1 - 1e-9)), 0.4, m, 1.0)
        En_o, En_i = np.sum(Eo*nrm, 0), np.sum(Ei*nrm, 0)
        assert np.max(np.abs((Eo - En_o*nrm) - (Ei - En_i*nrm))) < 1e-6
        assert np.max(np.abs(En_o - m**2*En_i)) < 1e-6
    # far from a small sphere: the incident wave
    E = an.mie_field(0.0, 0.3, -40.0, 0.05, 1.5, 1.0)
    assert np.allclose(E, [np.exp(-2j*np.pi*40), 0, 0], atol=1e-3)
    # Rayleigh limit: uniform inner field 3/(m**2 + 2)
    E = an.mie_field(0.0, 0.0, 0.0, 0.01, 2.0, 1.0)
    assert abs(E[0] - 0.5) < 3e-3


@pytest.mark.skipif(not fdtd.numba_available, reason="needs numba (slow with NumPy)")
def test_sphere_against_mie():
    """V8: a dielectric sphere (n = 1.5, diameter 0.6 wavelengths) in a
    plane wave: the field on the xz plane, away from the surface, within
    2 % of the Mie series, and the scattering cross section from the flux
    through a box around it within 2 %."""

    R, n, h = 0.3, 1.5, 1/30
    wls = np.array([0.8, 1.0])
    L = 1.6 + 24*h
    src = fdtd.TFSF((-0.5, 0.5)*3, fdtd.GaussianPulse(0.9, 0.25), wavelengths=wls)
    dft = fdtd.DFTField(wls, components=["Ex", "Ey", "Ez"], region=((-0.45, 0.45), 0.0, (-0.45, 0.45)))
    box = [fdtd.Flux(s*0.65, wls, axis=a, extent=((-0.65, 0.65), (-0.65, 0.65)))
           for a in range(3) for s in (-1, 1)]
    sim = fdtd.Simulation(fdtd.Grid((L, L, L), h), [(fdtd.Sphere((0, 0, 0), R), fdtd.Medium.from_index(n))],
                          [src], [dft] + box, boundaries=fdtd.PML(12))
    sim.run(time=30)
    z0 = sim.grid.axis(2)[src.box_indices[4]]
    A = src.incident_phasor(z0)*np.exp(-2j*np.pi/wls*z0)    # incident amplitude at z = 0
    for k, wl in enumerate(wls):
        num = den = 0.0
        for c in ("Ex", "Ey", "Ez"):
            x, y, z = np.meshgrid(*dft.coords(c), indexing="ij")
            M = an.mie_field(x, y, z, R, n, wl)["xyz".index(c[1])]
            away = np.abs(np.sqrt(x**2 + y**2 + z**2) - R) > 2*h
            num += np.sum(np.abs(dft.field(c)[k]/A[k] - M)[away]**2)
            den += np.sum(np.abs(M[away])**2)
        assert np.sqrt(num/den) < 0.02
    P = sum(s*m.flux() for m, s in zip(box, (-1, 1)*3))
    q_sca = P/(0.5*np.abs(A)**2)/(np.pi*R**2)
    q_mie = np.array([an.mie_efficiencies(R, n, wl)[1] for wl in wls])
    assert np.all(np.abs(q_sca/q_mie - 1) < 0.02)


def test_snapshots_and_dft_planes_3d(tmp_path):
    pytest.importorskip("h5py")
    grid = fdtd.Grid((1.2, 1.2, 1.2), 0.1)
    snap = fdtd.Snapshots("Ex", every=5, region=(None, 0.0, None), path=str(tmp_path/"xz.h5"))
    vol = fdtd.Snapshots("Ez", every=10, dtype=np.float32)
    dft = fdtd.DFTField(1.0, components=["Ex"], region=(None, None, 0.2))
    sim = fdtd.Simulation(grid, [], [fdtd.PlaneSource(-0.2, fdtd.GaussianPulse(1.0), axis=2)],
                          [snap, vol, dft], boundaries=("periodic", "periodic", fdtd.PML(3)))
    sim.run(steps=40)
    st = snap.stack()
    assert st.frames.shape == (8, 12, 1, 12) and len(st.coords) == 3
    assert vol.stack().frames.shape == (4, 12, 12, 12)
    assert dft.field("Ex").shape == (1, 12, 12, 1)
    # a plane wave: uniform across the plane
    F = dft.field("Ex")[0, :, :, 0]
    assert np.allclose(F, F[0, 0], rtol=1e-4)


def test_plots_3d():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyoptics.fdtd import plot as fplot
    grid = fdtd.Grid((1.2, 1.0, 1.4), 0.1)
    dft = fdtd.DFTField(1.0, components=["Ex"], region=(None, 0.0, None))
    sim = fdtd.Simulation(grid, [(fdtd.Sphere((0, 0, 0), 0.3), fdtd.Medium(4.0))],
                          [fdtd.TFSF((-0.3, 0.3, -0.3, 0.3, -0.4, 0.4), fdtd.GaussianPulse(1.0))],
                          [dft, fdtd.Flux(0.5, [1.0], axis=2)], boundaries=fdtd.PML(2))
    sim.run(steps=5)
    ax = fplot.plot_structure(sim, plt.figure().gca())
    assert ax.get_images()[0].get_array().shape == (14, 12)      # z vertical, x horizontal
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "z"
    ax = fplot.plot_structure(sim, plt.figure().gca(), plane=("z", 0.0))
    assert ax.get_images()[0].get_array().shape == (10, 12) and ax.get_ylabel() == "y"
    ax = fplot.plot_field(dft, sim=sim, ax=plt.figure().gca(), part="intensity")
    assert ax.get_images()[0].get_array().shape == (14, 12)
    ax = fplot.plot_field(sim, "Hy", ax=plt.figure().gca(), plane=("x", 0.0))
    assert ax.get_xlabel() == "y"
    plt.close("all")


def test_colocated():
    """colocated() averages half-node components onto the integer nodes
    they share: a linear field comes out exact."""

    grid = fdtd.Grid((1.0, 0.8, 0.6), 0.1)
    dft = fdtd.DFTField(1.0, components=["Ex", "Ey", "Ez"])
    sim = fdtd.Simulation(grid, [], [], [dft], boundaries="pec")
    sim.setup()
    for c in ("Ex", "Ey", "Ez"):
        X = np.meshgrid(*dft.coords(c), indexing="ij")
        dft._acc[c][0] = X[0] + 2*X[1] - X[2]
    crd, F = dft.colocated()
    X = np.meshgrid(*crd, indexing="ij")
    assert F.shape[0] == 3 and all(len(c) > 3 for c in crd)
    assert np.allclose(F, (X[0] + 2*X[1] - X[2])[None])
    assert np.allclose(np.diff(crd[0]), 0.1) and np.min(np.abs(crd[0])) < 1e-9
