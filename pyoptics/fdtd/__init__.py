#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Finite-difference time-domain (FDTD) solver for Maxwell's equations in
1D and 2D.

The FDTD method (Yee 1966) samples E and H on staggered grids in space and
time and leapfrogs them through the curl equations. It is rigorous for any
structure that can be drawn on the grid: waveguides, resonators, gratings,
scatterers of any size, metals. A single run with a short pulse gives the
response over a whole band.

Building blocks
---------------
- `Grid`: domain size, cell size and time step (normalized units: c = 1,
  lengths and wavelengths in the same unit, times in that unit over c).
- Shapes (`Slab`, `Rect`, `Circle`, `Annulus`, `Polygon`, unions and
  differences) and `Medium` (permittivity, conductivity and `Drude`,
  `Lorentz` and `Debye` poles; `Medium.fit()` to tabulated n + ik), drawn
  on the grid with sub-cell averaging of the permittivity (anisotropic for
  E in the plane of the Hz mode).
- Boundaries: `PML` (convolutional PML), "pec" walls, "periodic".
- Sources: `PointSource`, `LineSource` (soft currents), `TFSF` (plane wave
  in a total-field / scattered-field box), with the time profiles
  `GaussianPulse` and `ContinuousWave`.
- Monitors: `Probe`, `Snapshots`, `DFTField`, `Flux`.
- `Simulation` puts them together; `pyoptics.fdtd.plot` draws structures,
  fields and animations.

Polarization: Ez and Hz
-----------------------
2D simulations are invariant along z and split into two independent
polarizations, named here after the field component along z:

- ``mode="Ez"``: E along z (out of the plane), fields Ez, Hx, Hy,
- ``mode="Hz"``: H along z, fields Ex, Ey, Hz.

The literature names them inconsistently. In the FDTD literature (Taflove
and Hagness) Ez is "TM" (H transverse to z) and Hz "TE". For waveguides,
gratings and scattering, the names refer to the plane of incidence
instead: E perpendicular to it (here: E along z) is "TE" or s, so there Ez
is "TE" and Hz "TM". `pyoptics.litho.rcwa` and
`pyoptics.analytical.cylinder_field` use the latter: their TE case is the
Ez mode here.

Example
-------
>>> from pyoptics import fdtd
>>> grid = fdtd.Grid((8.0, 8.0), 1/30)
>>> sim = fdtd.Simulation(grid, structure=[(fdtd.Circle((0, 0), 1.0), fdtd.Medium.from_index(1.5))],
...                       sources=[fdtd.TFSF((-2, 2, -2, 2), fdtd.GaussianPulse(1.0))])
>>> dft = fdtd.DFTField(1.0, components=["Ez"])
>>> sim.monitors.append(dft)
>>> sim.run(time=40)  # doctest: +SKIP

Speed: the NumPy kernels are the reference; with numba installed
(``pip install pyoptics[fast]``) compiled kernels are used automatically
(``Simulation(backend=...)``), with identical results and 10 to 20 times
faster on multi-core machines.

Planned (proposals/fdtd.md): mode sources and S-parameters, Bloch
boundaries, 3D.
"""

from .grid import Grid
from .geometry import (Shape, Union, Difference, Everywhere, Slab, Rect, Circle, Annulus,
                       Polygon)
from .media import Medium, VACUUM, Drude, Lorentz, Debye, omega_from_eV
from .boundaries import PML, PEC, PERIODIC
from .sources import GaussianPulse, ContinuousWave, PointSource, LineSource, TFSF
from .monitors import (Probe, Snapshots, SnapshotStack, DFTField, DFTFieldData, Flux)
from .simulation import Simulation
from ._kernels_numba import available as numba_available

__all__ = ["Grid", "Shape", "Union", "Difference", "Everywhere", "Slab", "Rect", "Circle",
           "Annulus", "Polygon", "Medium", "VACUUM", "Drude", "Lorentz", "Debye", "omega_from_eV", "PML", "PEC", "PERIODIC", "GaussianPulse",
           "ContinuousWave", "PointSource", "LineSource", "TFSF", "Probe", "Snapshots",
           "SnapshotStack", "DFTField", "DFTFieldData", "Flux", "Simulation", "numba_available"]
