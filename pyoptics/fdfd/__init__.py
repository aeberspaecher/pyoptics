#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Finite-difference frequency-domain (FDFD) solver for Maxwell's equations
in 2D.

The FDFD method solves the time-harmonic Maxwell equations on the Yee grid
of the FDTD (`pyoptics.fdtd`), one frequency at a time, as a sparse linear
system: a driven problem (a source at a given wavelength) is one sparse LU
solve, and the resonances of an open structure (quasi-normal modes with
complex frequencies, their Q and fields) are the eigenvalues of the same
operator near a guess. That makes it the natural tool for resonators with
high Q, where an FDTD run would have to ring down for a long time, and for
media given by (n + ik) at a single wavelength.

Building blocks
---------------
- The grid, shapes and media of `pyoptics.fdtd` (`Grid`, `Circle`,
  `Polygon`, `Limacon`, ... and `Medium`; the permittivity is taken at the
  wavelength, so dispersive and lossy media need no fit), drawn with
  sub-cell averaging of the permittivity (anisotropic for the in-plane E of
  the Hz polarization).
- Boundaries per axis: `fdtd.PML` (a stretched-coordinate PML,
  s = kappa + i sigma/omega, graded polynomially), "pec", "periodic" and
  `fdtd.Bloch(k)`.
- Sources: `PointSource` (a line current along z) and `PlaneWave` (a plane
  wave injected on the boundary of a total-field box: total field inside,
  scattered field outside).
- `Simulation.solve()` returns the fields as `fdtd.DFTFieldData` (plots
  with `pyoptics.fdtd.plot`), `Simulation.resonances()` the quasi-normal
  modes near a wavelength.
- `farfield()`: the far-field pattern from an expansion of the field on a
  circle around the structure into outgoing cylindrical waves, and the
  directionality of the emission; `circle_harmonics()` for the azimuthal
  numbers of modes.

Polarization and conventions
----------------------------
2D, invariant along z, with the two polarizations of `pyoptics.fdtd`:
``mode="Ez"`` (E along z; fields Ez, Hx, Hy) and ``mode="Hz"`` (H along
z; Ex, Ey, Hz). Units are those of the FDTD (c = 1, the angular frequency
of a vacuum wavelength wl is 2 pi/wl); the time dependence is
exp(-i omega t), so outgoing waves go as exp(+i k r) and resonances have
Im(omega) < 0, Q = Re(omega)/(-2 Im(omega)), as in
`pyoptics.analytical.cylinder_field` and `disk_resonance`.

Example
-------
>>> from pyoptics import fdtd, fdfd
>>> grid = fdtd.Grid((5.0, 5.0), 0.025)
>>> sim = fdfd.Simulation(grid, [(fdtd.Circle((0, 0), 1.0), fdtd.Medium.from_index(3.3))])
>>> modes = sim.resonances(1.55, n=2)          # the WGM pair near 1.55
>>> modes[0].wavelength, modes[0].Q            # doctest: +SKIP
"""

from .solver import Simulation, PointSource, PlaneWave, Resonance
from .radiation import farfield, FarField, circle_harmonics

__all__ = ["Simulation", "PointSource", "PlaneWave", "Resonance", "farfield", "FarField", "circle_harmonics"]
