#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wafer stack: a resist layer on a thin film stack, for images inside the
resist.

The image-forming plane waves come from the immersion medium (refractive
index n of the projector) and hit a stack ::

    immersion medium | layers above (top coat) | resist | layers below
    (BARC, ...) | substrate (e.g. silicon)

whose layers are parallel to the x-y plane. Each plane wave is decomposed
into its TE (s) and TM (p) parts with respect to its own plane of
incidence. With the coefficients of pyoptics.thin_films (ratios of
tangential electric fields), the field in the resist at the depth z below
the resist top is the sum of the downward and the upward (reflected) wave, ::

    E_tan(z) = A*(exp(i*k_z*z) + r_b*exp(i*k_z*(2*d - z))),
    A = t_top/(1 - r_top*r_b*exp(2*i*k_z*d)),

with the transmission t_top of the layers above into the resist, the
reflection r_top of those layers seen from the resist, the reflection r_b of
the layers below seen from the resist, the resist thickness d and the
(complex, for an absorbing resist) z component k_z of the wave vector in the
resist. The TE field stays perpendicular to the plane of incidence; the TM
field has the tangential part E_tan along the transverse wave vector and the
z component E_z = -k_perp/k_z*E_tan for the downward and +k_perp/k_z*E_tan
for the upward wave (transversality). The swing of the intensity with z are
the standing waves in the resist.

In the scalar model of the projector, the TE coefficients are used for both
field components and there is no E_z.

Convention: fields vary as exp(i(k.r - omega*t)), absorbing media have
n + i*k with k > 0.
"""

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt

from ..thin_films import _r_t


def _complex_n(layers):
    return np.array([n + 1j*k for n, k, t in layers], dtype=complex)


def _thicknesses(layers):
    return np.array([t for n, k, t in layers], dtype=float)


class WaferStack(object):
    """Resist on a substrate with optional layers above and below.

    Parameters
    ----------
    resist : tuple (n, k, thickness)
    below : sequence of (n, k, thickness), optional
        Layers below the resist, from the resist downwards (e.g. a BARC).
    substrate : tuple (n, k), optional
        Default: silicon at 193 nm, about 0.88 + 2.78i.
    above : sequence of (n, k, thickness), optional
        Layers between the immersion medium and the resist, from the top
        (e.g. a top coat).
    name : str, optional
    """

    def __init__(self, resist, below=(), substrate=(0.883, 2.778), above=(), name=None):
        self.resist = tuple(resist)
        self.below, self.above = list(below), list(above)
        self.substrate = substrate
        self.name = name or "wafer stack"

    @property
    def resist_n(self):
        return self.resist[0] + 1j*self.resist[1]

    @property
    def thickness(self):
        return self.resist[2]

    def coefficients(self, k_perp, k0, n_medium, pol):
        """t_top, r_top, r_b and k_z in the resist for plane waves with the
        transverse wave numbers k_perp (array) coming from the medium
        n_medium, for pol "TE" or "TM"."""

        k_perp = np.asarray(k_perp, dtype=float)
        n_r = self.resist_n
        n_top = np.concatenate([[n_medium], _complex_n(self.above), [n_r]])
        n_up = n_top[::-1]
        n_bottom = np.concatenate([[n_r], _complex_n(self.below),
                                   [self.substrate[0] + 1j*self.substrate[1]]])
        t_above = _thicknesses(self.above)
        wl = 2*np.pi/k0

        t_top = _r_t(k_perp/(k0*n_medium), n_top, t_above, pol, wl)[1]
        r_top = _r_t(k_perp/(k0*n_r), n_up, t_above[::-1], pol, wl)[0]
        r_b = _r_t(k_perp/(k0*n_r), n_bottom, _thicknesses(self.below), pol, wl)[0]
        k_z = k0*complex_sqrt(n_r**2 - (k_perp/k0)**2)

        return t_top, r_top, r_b, k_z

    def factors(self, k_perp, k0, n_medium, depth, pol):
        """Tangential field factors of the downward and upward waves at the
        depth (below the resist top) relative to the incident tangential
        field at the top of the stack, and k_z."""

        t_top, r_top, r_b, k_z = self.coefficients(k_perp, k0, n_medium, pol)
        d = self.thickness
        A = t_top/(1 - r_top*r_b*np.exp(2j*k_z*d))
        down = A*np.exp(1j*k_z*depth)
        up = A*r_b*np.exp(1j*k_z*(2*d - depth))

        return down, up, k_z

    def transmit(self, E, K_x, K_y, k0, n_medium, depth, vectorial=True):
        """Field in the resist at the depth for plane waves with the fields E
        (shape (3,) + K_x.shape, at the top of the stack in the medium) and
        transverse wave vectors (K_x, K_y)."""

        k_perp = np.hypot(K_x, K_y)
        if not vectorial:
            down, up, _ = self.factors(k_perp, k0, n_medium, depth, "TE")
            out = np.zeros_like(E)
            out[:2] = E[:2]*(down + up)
            return out

        phi = np.arctan2(K_y, K_x)
        c, s = np.cos(phi), np.sin(phi)
        a_TE = -s*E[0] + c*E[1]
        a_TM = c*E[0] + s*E[1]  # tangential part of the TM field
        d_TE, u_TE, _ = self.factors(k_perp, k0, n_medium, depth, "TE")
        d_TM, u_TM, k_z = self.factors(k_perp, k0, n_medium, depth, "TM")
        TE = a_TE*(d_TE + u_TE)
        TM = a_TM*(d_TM + u_TM)
        E_z = -k_perp/k_z*a_TM*(d_TM - u_TM)

        return np.array([-s*TE + c*TM, c*TE + s*TM, E_z])
