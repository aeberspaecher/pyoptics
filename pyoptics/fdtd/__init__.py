#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Finite-difference time-domain (FDTD) solver for Maxwell's equations in
1D, 2D and 3D.

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
  differences; waveguides `Path`, `SBend`, `Bend`, `Ring`; `Lattice`; in
  3D `Sphere`, `Box`, `Cylinder`, `Prism`) and `Medium` (permittivity, conductivity and `Drude`,
  `Lorentz` and `Debye` poles; `Medium.fit()` to tabulated n + ik), drawn
  on the grid with sub-cell averaging of the permittivity (anisotropic for
  E in the plane of the Hz mode).
- Boundaries: `PML` (convolutional PML), "pec" walls, "periodic",
  `Bloch(k)` (Bloch-periodic, complex fields: oblique incidence on
  periodic structures, band structures).
- Sources: `PointSource`, `LineSource` (soft currents), `TFSF` (plane wave
  in a total-field / scattered-field box), with the time profiles
  `GaussianPulse` and `ContinuousWave`.
- Monitors: `Probe`, `Snapshots`, `DFTField`, `Flux`.
- Waveguides (`pyoptics.fdtd.modes`): a mode solver that is exact on the
  Yee grid (`waveguide_modes`, `find_modes`, `section_modes`),
  `ModeSource` (one-sided TFSF injection of a mode), `ModeMonitor`
  (forward and backward mode amplitudes per wavelength) and
  `s_parameters`.
- `resonances` (`pyoptics.fdtd.analysis`): frequencies and Q of ring-down
  signals by the matrix pencil method.
- Periodic structures (`pyoptics.fdtd.periodic`): `DiffractionMonitor`
  (power and complex amplitude of the diffraction orders of gratings and
  unit cells, e.g. the phase library of a metasurface) and `bloch_modes`
  (frequencies of Bloch modes, for band diagrams of photonic crystals and
  of defects in supercells); `Lattice` places a shape on a lattice with
  removed or shifted sites (photonic crystals with waveguides and
  cavities).
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

3D
--
A Grid with three sizes makes a 3D simulation with all six components
(mode "3D"). The defaults suit the larger grids: float32 fields, the numba
kernels, CPML auxiliary fields only inside the layers, sub-cell averaging
only in the cells a boundary crosses. 3D shapes are `Sphere`, `Box`,
`Cylinder` and `Prism` (any 2D shape extruded); `PlaneSource` (a 3D
LineSource) and `TFSF` (a plane wave along +z, polarized along x or y)
excite the fields; monitor regions take a number for a single node layer
(planes); `Flux` works on planes; plots show sections (``plane=("y",
0.0)``). `pyoptics.analytical.mie_field` is the exact reference for a
sphere.

Planned (proposals/fdtd.md): the litho mask model.
"""

from .grid import Grid
from .geometry import (Shape, Union, Difference, Everywhere, Slab, Rect, Circle, Annulus,
                       Polygon, Path, SBend, Bend, Ring, Lattice, Sphere, Box, Prism, Cylinder,
                       sbend_points)
from .media import Medium, VACUUM, Drude, Lorentz, Debye, omega_from_eV
from .boundaries import PML, PEC, PERIODIC, Bloch
from .sources import GaussianPulse, ContinuousWave, PointSource, LineSource, PlaneSource, TFSF
from .monitors import (Probe, Snapshots, SnapshotStack, DFTField, DFTFieldData, Flux)
from .modes import (Mode, waveguide_modes, find_modes, section_modes, ModeSource, ModeMonitor,
                    s_parameters)
from .analysis import resonances, Resonances
from .periodic import DiffractionMonitor, DiffractionOrders, bloch_modes
from .simulation import Simulation
from ._kernels_numba import available as numba_available

__all__ = ["Grid", "Shape", "Union", "Difference", "Everywhere", "Slab", "Rect", "Circle",
           "Annulus", "Polygon", "Path", "SBend", "Bend", "Ring", "Lattice", "Sphere", "Box", "Prism",
           "Cylinder", "sbend_points", "Medium",
           "VACUUM", "Drude", "Lorentz", "Debye", "omega_from_eV", "PML", "PEC", "PERIODIC", "Bloch",
           "GaussianPulse", "ContinuousWave", "PointSource", "LineSource",
           "PlaneSource", "TFSF", "Probe",
           "Snapshots", "SnapshotStack", "DFTField", "DFTFieldData", "Flux", "Mode",
           "waveguide_modes", "find_modes", "section_modes", "ModeSource", "ModeMonitor",
           "s_parameters", "resonances", "Resonances", "DiffractionMonitor",
           "DiffractionOrders", "bloch_modes", "Simulation", "numba_available"]
