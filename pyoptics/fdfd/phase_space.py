#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Husimi functions of FDFD modes in the Birkhoff coordinates of the ray
picture.

`husimi_birkhoff()` samples the field along z (``mode="Ez"``) and its
derivative along the outward normal on the boundary of a cavity drawn with a
`fdtd.PolarShape` (`Limacon`, `Stadium`, ...) and passes them to
`pyoptics.billiard.boundary_husimi()`. For E along z both are continuous
across the boundary (the tangential E and H), so they can be taken from the
grid without knowing on which side of the staircase a sample lies; the
derivative is a central difference across the boundary.

References
----------
M. Hentschel, H. Schomerus and R. Schubert, "Husimi functions at dielectric
interfaces: Inside-outside duality for optical systems and beyond",
Europhys. Lett. 62, 636-642 (2003).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..billiard import Billiard, boundary_husimi


def _interpolator(solution):
    x, y = solution.coords[solution.sim.components[0]]
    u = solution.main
    re = RegularGridInterpolator((x, y), u.real, method="cubic")
    im = RegularGridInterpolator((x, y), u.imag, method="cubic")
    return lambda P: re(P) + 1j*im(P)


def boundary_field(solution, shape, num=1024, delta=None):
    """Field along z and its outward normal derivative on the boundary of a
    shape.

    Parameters
    ----------
    solution : Solution
        Field of a ``mode="Ez"`` simulation (e.g. `Resonance.field`).
    shape : fdtd.PolarShape or billiard.Billiard
        Boundary.
    num : int, optional
        Equidistant points along the boundary.
    delta : number, optional
        Step of the central difference along the normal (default: the grid
        cell).

    Returns
    -------
    s : array (num,)
        Arc lengths.
    psi, dpsi : complex arrays (num,)
    billiard : Billiard
        The boundary (perimeter, Birkhoff coordinates).
    """

    if solution.sim.mode != "Ez":
        raise ValueError("husimi_birkhoff needs a mode='Ez' solution (normal derivative of Hz jumps)")
    bil = shape if isinstance(shape, Billiard) else Billiard(shape)
    x, y = solution.coords[solution.sim.components[0]]
    delta = abs(x[1] - x[0]) if delta is None else delta
    f = _interpolator(solution)
    s, P, _, N = bil.boundary(num)
    up, dn = f(P + delta*N), f(P - delta*N)
    psi = f(P)
    dpsi = (up - dn)/(2*delta)
    return s, psi, dpsi, bil


def husimi_birkhoff(solution, shape, n, side="inside", num=1024, q=None, p=None, sigma=None, delta=None):
    """Husimi functions of an FDFD field on the boundary of a cavity, in
    Birkhoff coordinates (arc length q, p = sin(chi)).

    Parameters
    ----------
    solution : Solution
        ``mode="Ez"`` field, e.g. `Resonance.field`.
    shape : fdtd.PolarShape or billiard.Billiard
        The cavity boundary.
    n : number
        Refractive index of the cavity (inside); the outside is taken as
        air.
    side : "inside" or "outside", optional
        Where the coherent states live: inside, p is sin(chi) in the cavity
        (index n) and the result is (incident, reflected); outside, p is
        sin(chi) in air and the result is (emerging, incoming).
    num : int, optional
        Points along the boundary.
    q, p, sigma : optional
        As `billiard.boundary_husimi()`.
    delta : number, optional
        Normal step of the derivative (default: the grid cell).

    Returns
    -------
    q, p : arrays
    H_1, H_2 : arrays (len(p), len(q))
        inside: incident and reflected; outside: emerging and incoming.
    """

    s, psi, dpsi, bil = boundary_field(solution, shape, num, delta)
    k = float(np.real(solution.omega))
    n_side = {"inside": n, "outside": 1.0}[side]
    q, p, H_out, H_in = boundary_husimi(s, psi, dpsi, bil.length, n_side, k, q=q, p=p, sigma=sigma)
    return q, p, H_out, H_in
