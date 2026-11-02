#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the 2D FDFD solver: plane-wave injection, scattering by a
dielectric cylinder and the resonances of a microdisk against the exact
series, a slab under oblique incidence with Bloch boundaries, the far-field
transform, and the analytical results used for them."""

import numpy as np
import pytest

from pyoptics import analytical as an
from pyoptics import fdfd, fdtd


@pytest.mark.parametrize("mode", ["Ez", "Hz"])
def test_plane_wave_injection_is_exact(mode):
    sim = fdfd.Simulation(fdtd.Grid((3.0, 3.0), 0.03), mode=mode)
    for d in ((0, 1), (1, 0.4)):
        sol = sim.solve(1.0, fdfd.PlaneWave(d, ((-0.9, 0.9), (-0.9, 0.9))))
        x, y = sol.coords[mode]
        X, Y = np.meshgrid(x, y, indexing="ij")
        out = (np.abs(X) > 1.0) | (np.abs(Y) > 1.0)
        inside = (np.abs(X) < 0.85) & (np.abs(Y) < 0.85)
        assert np.abs(sol.main[out]).max() < 1e-10
        assert np.abs(np.abs(sol.main[inside]) - 1).max() < 1e-10


def test_cylinder_coefficients_h_polarization():
    # the H series against the one written out in test_fdtd.py
    from test_fdtd import _cylinder_series_H
    x, z = np.meshgrid(np.linspace(-1.5, 1.5, 31), np.linspace(-1.5, 1.5, 29), indexing="ij")
    u1 = an.cylinder_field(x, z, 0.8, 1.5, 1.0, polarization="H")
    u2 = _cylinder_series_H(x, z, 0.8, 1.5, 1.0, M=30)
    assert np.abs(u1 - u2).max() < 1e-7


@pytest.mark.parametrize("mode, pol, tol", [("Ez", "E", 0.06), ("Hz", "H", 0.09)])
def test_cylinder_scattering_against_series(mode, pol, tol):
    R, n, wl = 0.6, 1.8, 1.0
    errs = []
    for h in (0.04, 0.02):
        sim = fdfd.Simulation(fdtd.Grid((4.0, 4.0), h), [(fdtd.Circle((0, 0), R), fdtd.Medium.from_index(n))],
                              mode=mode)
        sol = sim.solve(wl, fdfd.PlaneWave((0, 1), ((-1.2, 1.2), (-1.2, 1.2))))
        x, y = sol.coords[mode]
        X, Y = np.meshgrid(x, y, indexing="ij")
        exact = an.cylinder_field(X, Y, R, n, wl, polarization=pol)   # incident along +y
        sel = (np.abs(X) < 1.1) & (np.abs(Y) < 1.1) & (np.abs(np.hypot(X, Y) - R) > 2*h)
        errs.append(np.abs(sol.main - exact)[sel].max())
    assert errs[1] < tol
    assert errs[0]/errs[1] > 3.0          # second order


def test_disk_resonance_solves_the_matching_condition():
    from scipy.special import jv, jvp, hankel1, h1vp
    for pol, g in (("E", 3.3), ("H", 1/3.3)):
        k = an.disk_resonance(6, 1.0, 3.3, 2.4 if pol == "E" else 2.1, polarization=pol)
        x = k*1.0
        f = g*jvp(6, 3.3*x)*hankel1(6, x) - jv(6, 3.3*x)*h1vp(6, x)
        assert abs(f)/abs(jv(6, 3.3*x)*h1vp(6, x)) < 1e-10
        assert k.imag < 0


@pytest.mark.parametrize("mode, pol, q_tol", [("Ez", "E", 0.02), ("Hz", "H", 0.06)])
def test_microdisk_resonances_against_exact(mode, pol, q_tol):
    m, R, n = 6, 1.0, 3.3
    k = an.disk_resonance(m, R, n, 2.4 if pol == "E" else 2.1, polarization=pol)
    wl0, Q0 = 2*np.pi/k.real, k.real/(-2*k.imag)
    sim = fdfd.Simulation(fdtd.Grid((4.0, 4.0), 0.02), [(fdtd.Circle((0, 0), R), fdtd.Medium.from_index(n))],
                          mode=mode)
    modes = sim.resonances(wl0, n=2)
    for r in modes:   # the degenerate pair cos/sin(m phi)
        assert abs(r.wavelength/wl0 - 1) < 1e-3
        assert abs(r.Q/Q0 - 1) < q_tol
        mm, c = fdfd.circle_harmonics(r.field, 0.95*R)
        assert abs(mm[np.argmax(np.abs(c))]) == m


def test_slab_reflection_with_bloch_boundaries():
    # an oblique plane wave on a dielectric slab, periodic along x
    wl, n, d, theta = 1.0, 1.6, 0.35, np.radians(30)
    k0 = 2*np.pi/wl
    kx = k0*np.sin(theta)
    L = 0.6
    grid = fdtd.Grid((L, 4.0), 0.01)
    for mode in ("Ez", "Hz"):
        sim = fdfd.Simulation(grid, [(fdtd.Slab(0.0, d, axis=1), fdtd.Medium.from_index(n))], mode=mode,
                              boundaries=(fdtd.Bloch(kx), fdtd.PML(20)))
        sol = sim.solve(wl, fdfd.PlaneWave((np.sin(theta), np.cos(theta)), ((-1, 1), (-0.5, 1.2))))
        x, y = sol.coords[mode]
        j = np.argmin(np.abs(y + 0.9))         # below the box: the reflected wave only
        r_num = np.abs(sol.main[:, j]).mean()
        ky1, ky2 = k0*np.cos(theta), np.sqrt((n*k0)**2 - kx**2)
        if mode == "Ez":
            r12 = (ky1 - ky2)/(ky1 + ky2)
        else:
            r12 = (ky1*n**2 - ky2)/(ky1*n**2 + ky2)
        e = np.exp(2j*ky2*d)
        r = r12*(1 - e)/(1 - r12**2*e)
        assert abs(r_num - abs(r)) < 5e-3


def test_farfield_of_a_line_source_is_isotropic():
    sim = fdfd.Simulation(fdtd.Grid((4.0, 4.0), 0.02))
    ff = fdfd.farfield(sim.solve(1.0, fdfd.PointSource((0.3, 0.1))), 1.2)
    assert np.abs(ff.intensity - 1).max() < 5e-3
    assert abs(ff.directionality()) < 1e-3
    # a mirror on the right sends the light to the left
    sim = fdfd.Simulation(fdtd.Grid((4.0, 4.0), 0.02), [(fdtd.Rect((0.8, 0), (0.1, 1.0)), fdtd.Medium(12.0))])
    ff = fdfd.farfield(sim.solve(1.0, fdfd.PointSource((0, 0))), 1.3)
    assert ff.directionality() < -0.2


def test_limacon_shape():
    lim = fdtd.Limacon((0.2, -0.1), 1.0, 0.43)
    x, y = np.meshgrid(np.linspace(-1.5, 1.8, 661), np.linspace(-1.6, 1.4, 601), indexing="ij")
    area = lim.contains(x, y).sum()*(x[1, 0] - x[0, 0])*(y[0, 1] - y[0, 0])
    assert abs(area/lim.area() - 1) < 5e-3
    assert lim.contains(0.2 + 1.42, -0.1) and not lim.contains(0.2 + 1.44, -0.1)
    assert lim.contains(0.2 - 0.56, -0.1) and not lim.contains(0.2 - 0.58, -0.1)
    with pytest.raises(ValueError):
        fdtd.Limacon((0, 0), 1.0, 1.0)


def test_mode_tracing_keeps_the_mode():
    R, n = 1.0, 3.3
    grid = fdtd.Grid((4.0, 4.0), 0.03)
    k = an.disk_resonance(6, R, n, 2.4)
    sim = fdfd.Simulation(grid, [(fdtd.Limacon((0, 0), R, 0.05), fdtd.Medium.from_index(n))])
    prev = sim.resonance(2*np.pi/k.real)
    for eps in (0.1, 0.15):
        sim = fdfd.Simulation(grid, [(fdtd.Limacon((0, 0), R, eps), fdtd.Medium.from_index(n))])
        r = sim.resonance(like=prev)
        assert r.overlap(prev) > 0.9
        prev = r
