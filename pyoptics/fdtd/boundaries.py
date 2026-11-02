#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Boundary conditions: convolutional PML, PEC walls, periodicity and
Bloch periodicity.

The convolutional perfectly matched layer (CPML, Roden and Gedney 2000)
stretches the coordinate across the absorbing layer with
s(d) = kappa(d) + sigma(d)/(alpha(d) - i*omega), graded polynomially from
the inner edge (d = 0) to the outer wall (d = 1). Each spatial derivative
inside the layer becomes dF/dx/kappa + psi, with the recursive convolution
psi <- b*psi + c*dF/dx. The layer is matched for any medium that extends
into it, and its outer wall is a PEC.
"""

import numpy as np


class PML(object):
    """Absorbing layer of the given number of cells on both ends of an axis.

    Parameters
    ----------
    cells : int
        Thickness in cells.
    order : number
        Polynomial grading order m of sigma and kappa.
    reflection : number
        Theoretical reflection R0 at normal incidence; sets
        sigma_max = -(m + 1)*ln(R0)/(2*thickness).
    kappa_max : number
        Maximum real coordinate stretch (>= 1); > 1 helps with evanescent
        and grazing waves.
    alpha_max : number
        Complex frequency shift at the inner edge (in units of 1/time,
        decreasing linearly to 0 at the wall); absorbs low frequencies less
        and so avoids late-time drift for long runs.
    """

    def __init__(self, cells=12, order=3, reflection=1e-8, kappa_max=1.0, alpha_max=0.0):
        self.cells = int(cells)
        self.order = order
        self.reflection = reflection
        self.kappa_max = kappa_max
        self.alpha_max = alpha_max

    def __repr__(self):
        return "PML(cells=%d)" % self.cells

    def coefficients(self, n_cells, spacing, dt, offset):
        """b, c and 1/kappa for the left and the right layer along an axis
        of n_cells cells, at nodes shifted by offset cells. Each is an array
        of length cells; the left layer covers indices 0..cells - 1, the
        right one n_cells - cells..n_cells - 1."""

        n = self.cells
        thickness = n*spacing
        sigma_max = -(self.order + 1)*np.log(self.reflection)/(2*thickness)
        out = []
        for idx in (np.arange(n), np.arange(n_cells - n, n_cells)):
            p = idx + offset   # position in cells
            d = np.clip(np.maximum(n - p, p - (n_cells - n))/n, 0.0, 1.0)
            sigma = sigma_max*d**self.order
            kappa = 1 + (self.kappa_max - 1)*d**self.order
            alpha = self.alpha_max*(1 - d)*(d > 0)
            b = np.exp(-(sigma/kappa + alpha)*dt)
            denom = sigma*kappa + kappa**2*alpha
            c = np.where(denom > 0, sigma*(b - 1)/np.where(denom > 0, denom, 1), 0.0)
            out.append((b, c, 1/kappa))
        return out


PEC = "pec"
PERIODIC = "periodic"


class Bloch(object):
    """Bloch-periodic boundary along an axis: F(x + L) = F(x) exp(i k L)
    for all fields, with L the length of the domain along the axis.

    Parameters
    ----------
    k : float
        Bloch wave number along the axis (rad per length unit). k = 0 is
        plain periodicity; otherwise the fields are complex (the
        simulation switches to complex dtype).

    Uses: oblique incidence on gratings (k = n k0 sin(theta) of the
    incident wave at one wavelength; with a pulse, the angle varies across
    the band at fixed k), band structures of photonic crystals (one run per
    k on the unit cell).
    """

    def __init__(self, k=0.0):
        self.k = float(k)

    def __eq__(self, other):
        return isinstance(other, Bloch) and other.k == self.k

    def __hash__(self):
        return hash(("bloch", self.k))

    def __repr__(self):
        return "Bloch(k=%g)" % self.k


def parse(boundaries, ndim):
    """Normalize a boundary spec to one entry (PML, "pec", "periodic" or
    Bloch; Bloch(0) becomes "periodic") per axis. A single entry is used for all axes; "pml" means PML()."""

    if isinstance(boundaries, (str, PML, Bloch)):
        boundaries = (boundaries,)*ndim
    boundaries = tuple(boundaries)
    if len(boundaries) != ndim:
        raise ValueError("need one boundary per axis")
    out = []
    for b in boundaries:
        if isinstance(b, Bloch):
            out.append(PERIODIC if b.k == 0 else b)
        elif isinstance(b, PML):
            out.append(b)
        elif b == "pml":
            out.append(PML())
        elif b in (PEC, PERIODIC):
            out.append(b)
        else:
            raise ValueError("unknown boundary %r" % (b,))
    return tuple(out)
