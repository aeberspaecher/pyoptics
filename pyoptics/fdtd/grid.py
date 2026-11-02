#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""The Yee grid: sizes, staggered sample positions and the time step.

Units are normalized: the vacuum speed of light, permittivity and
permeability are 1, lengths are in any unit (the same as the wavelengths)
and times are in that unit divided by c. The frequency of a vacuum
wavelength wl is f = 1/wl, its angular frequency 2*pi/wl.

Arrays are indexed [ix] in 1D, [ix, iy] in 2D and [ix, iy, iz] in 3D (x
first). The grid has shape[a] cells along axis a, with origin + i*spacing
(i = 0..shape[a] - 1) for the integer nodes and origin + (i + 1/2)*spacing
for the half nodes.
The outer walls lie at origin and origin + shape*spacing.
"""

import numpy as np


# sampling offsets (in cells, per axis) of each field component on the Yee
# grid, per polarization mode; the first entry of each mode is the electric
# field component(s) along the "main" field.
OFFSETS = {
    1: {"Ez": {"Ez": (0.0,), "Hy": (0.5,)}},
    2: {"Ez": {"Ez": (0.0, 0.0), "Hx": (0.0, 0.5), "Hy": (0.5, 0.0)},
        "Hz": {"Ex": (0.5, 0.0), "Ey": (0.0, 0.5), "Hz": (0.5, 0.5)}},
    # 3D: the 2D offsets extended (E along z and H in the plane sit half a
    # cell up along z)
    3: {"3D": {"Ex": (0.5, 0.0, 0.0), "Ey": (0.0, 0.5, 0.0), "Ez": (0.0, 0.0, 0.5),
               "Hx": (0.0, 0.5, 0.5), "Hy": (0.5, 0.0, 0.5), "Hz": (0.5, 0.5, 0.0)}},
}

E_COMPONENTS = {(1, "Ez"): ("Ez",), (2, "Ez"): ("Ez",), (2, "Hz"): ("Ex", "Ey"),
                (3, "3D"): ("Ex", "Ey", "Ez")}
H_COMPONENTS = {(1, "Ez"): ("Hy",), (2, "Ez"): ("Hx", "Hy"), (2, "Hz"): ("Hz",),
                (3, "3D"): ("Hx", "Hy", "Hz")}
AXES = "xyz"


def mode_key(ndim, mode):
    """The key of OFFSETS[ndim] for a polarization mode: the mode in 2D,
    "Ez" in 1D, "3D" in 3D."""

    return mode if ndim == 2 else ("Ez" if ndim == 1 else "3D")


class Grid(object):
    """Regular Yee grid in 1D, 2D or 3D.

    Parameters
    ----------
    size : number or tuple
        Extent of the computational domain along each axis, including the
        absorbing layers. A number gives a 1D grid.
    spacing : number or tuple
        Cell size, the same for all axes if a number. The number of cells is
        round(size/spacing).
    center : number or tuple, optional
        Centre of the domain, 0 by default.
    """

    def __init__(self, size, spacing, center=None):
        size = np.atleast_1d(np.asarray(size, dtype=float))
        ndim = len(size)
        if ndim not in (1, 2, 3):
            raise ValueError("grids have 1, 2 or 3 dimensions")
        spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,)).copy()
        center = np.zeros(ndim) if center is None else \
            np.broadcast_to(np.asarray(center, dtype=float), (ndim,)).copy()
        self.ndim = ndim
        self.shape = tuple(int(round(s/d)) for s, d in zip(size, spacing))
        self.spacing = spacing
        self.size = np.array(self.shape)*spacing
        self.origin = center - self.size/2

    def __repr__(self):
        return "Grid(shape=%s, spacing=%s, origin=%s)" % (self.shape, tuple(self.spacing),
                                                          tuple(self.origin))

    def axis(self, a, offset=0.0):
        """Coordinates of the nodes along axis a, shifted by offset cells."""

        return self.origin[a] + (np.arange(self.shape[a]) + offset)*self.spacing[a]

    def coords(self, component, mode="Ez"):
        """1D coordinate arrays (one per axis) of a field component."""

        offsets = OFFSETS[self.ndim][mode_key(self.ndim, mode)][component]
        return tuple(self.axis(a, o) for a, o in enumerate(offsets))

    def index(self, position, component, mode="Ez"):
        """Index tuple of the node of a component closest to a position."""

        position = np.atleast_1d(np.asarray(position, dtype=float))
        offsets = OFFSETS[self.ndim][mode_key(self.ndim, mode)][component]
        idx = []
        for a, o in enumerate(offsets):
            i = int(np.floor((position[a] - self.origin[a])/self.spacing[a] - o + 0.5 + 1e-9))
            idx.append(min(max(i, 0), self.shape[a] - 1))
        return tuple(idx)

    def time_step(self, courant=0.99):
        """Time step as a fraction courant of the stability limit
        1/sqrt(sum(1/spacing**2))."""

        return courant/np.sqrt(np.sum(1.0/self.spacing**2))

    @property
    def extent(self):
        """(xmin, xmax[, ymin, ymax[, zmin, zmax]]) of the domain."""

        return tuple(v for a in range(self.ndim)
                     for v in (self.origin[a], self.origin[a] + self.size[a]))
