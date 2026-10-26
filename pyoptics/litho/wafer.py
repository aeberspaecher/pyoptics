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

Swing curves: for a plane wave with the transverse wave number
k_perp = k_0*na (na = n_medium*sin(theta) in the immersion medium, so na
runs up to the NA of the projector), WaferStack.reflectance() gives the
reflectance of the whole stack, substrate_reflectance() the reflectance
``|r_b|**2`` of the layers below as seen from the resist (it drives the
standing waves and the swing) and absorbed() the fraction of the incident
power absorbed in the resist, ::

    A = k_0*Im(n_r**2)*int_0^d |E(z)|**2 dz/(n_medium*cos(theta))

for a unit incident field, integrated analytically. swing_curve() evaluates
them through resist thickness, optimize_barc() scans the thickness of a
bottom anti-reflective coating for the lowest substrate reflectance.

Convention: fields vary as exp(i(k.r - omega*t)), absorbing media have
n + i*k with k > 0.
"""

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt

from ..polarization import TE_TM_basis
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

    def with_(self, thickness=None, **kwargs):
        """Copy with some parts exchanged, e.g. stack.with_(thickness=300.0)
        for another resist thickness or stack.with_(below=[...])."""

        p = dict(resist=self.resist, below=self.below, substrate=self.substrate, above=self.above,
                 name=self.name)
        p.update(kwargs)
        if thickness is not None:
            p["resist"] = tuple(p["resist"][:2]) + (thickness,)

        return WaferStack(**p)

    @property
    def resist_n(self):
        return self.resist[0] + 1j*self.resist[1]

    @property
    def thickness(self):
        return self.resist[2]

    def coefficients(self, k_perp, k0, n_medium, pol):
        """Fresnel coefficients of the stack seen from the resist.

        Parameters
        ----------
        k_perp : array
            Transverse wave numbers of the plane waves.
        k0 : number
            Vacuum wave number.
        n_medium : number
            Refractive index of the medium above the stack.
        pol : {"TE", "TM"}

        Returns
        -------
        t_top : array
            Transmission from the medium through the layers above into the resist.
        r_top : array
            Reflection at the layers above, seen from the resist.
        r_b : array
            Reflection of the layers below the resist (BARC, substrate).
        k_z : array
            Axial wave number in the resist.
        """

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
        """Tangential field factors of the downward and upward waves at a depth.

        Relative to the incident tangential field at the top of the stack,
        including the multiple reflections in the resist.

        Parameters
        ----------
        k_perp : array
            Transverse wave numbers.
        k0 : number
            Vacuum wave number.
        n_medium : number
            Refractive index of the medium above the stack.
        depth : number
            Depth below the resist top.
        pol : {"TE", "TM"}

        Returns
        -------
        down, up : arrays
            Factors of the downward and upward travelling waves.
        k_z : array
            Axial wave number in the resist.
        """

        t_top, r_top, r_b, k_z = self.coefficients(k_perp, k0, n_medium, pol)
        d = self.thickness
        A = t_top/(1 - r_top*r_b*np.exp(2j*k_z*d))
        down = A*np.exp(1j*k_z*depth)
        up = A*r_b*np.exp(1j*k_z*(2*d - depth))

        return down, up, k_z

    def transmit(self, E, K_x, K_y, k0, n_medium, depth, vectorial=True):
        """Field in the resist at a depth for a set of plane waves.

        Parameters
        ----------
        E : array
            Fields of the plane waves at the top of the stack in the medium,
            shape (3,) + K_x.shape.
        K_x, K_y : arrays
            Transverse wave vectors.
        k0 : number
            Vacuum wave number.
        n_medium : number
            Refractive index of the medium above the stack.
        depth : number
            Depth below the resist top.
        vectorial : bool, optional
            False: scalar model (both components with the TE factors, no E_z).

        Returns
        -------
        E_resist : array
            Field components (E_x, E_y, E_z) in the resist, same shape as E.
        """

        k_perp = np.hypot(K_x, K_y)
        if not vectorial:
            down, up, _ = self.factors(k_perp, k0, n_medium, depth, "TE")
            out = np.zeros_like(E)
            out[:2] = E[:2]*(down + up)
            return out

        e_TE, e_TM = TE_TM_basis(K_x, K_y)
        a_TE = e_TE[0]*E[0] + e_TE[1]*E[1]
        a_TM = e_TM[0]*E[0] + e_TM[1]*E[1]  # tangential part of the TM field
        d_TE, u_TE, _ = self.factors(k_perp, k0, n_medium, depth, "TE")
        d_TM, u_TM, k_z = self.factors(k_perp, k0, n_medium, depth, "TM")
        TE = a_TE*(d_TE + u_TE)
        TM = a_TM*(d_TM + u_TM)
        E_z = -k_perp/k_z*a_TM*(d_TM - u_TM)

        return np.array([e_TE[0]*TE + e_TM[0]*TM, e_TE[1]*TE + e_TM[1]*TM, E_z])

    # plane wave quantities for swing curves:

    def _media(self, n_medium):
        n = np.concatenate([[n_medium], _complex_n(self.above), [self.resist_n], _complex_n(self.below),
                            [self.substrate[0] + 1j*self.substrate[1]]])
        t = np.concatenate([_thicknesses(self.above), [self.thickness], _thicknesses(self.below)])

        return n, t

    def reflectance(self, na, wavelength, n_medium=1.0, pol="unpolarized"):
        """Reflectance of the whole stack for plane waves from the medium.

        Parameters
        ----------
        na : number or array
            Transverse wave numbers in units of k_0 (n_medium*sin(theta)).
        wavelength : number
            Vacuum wavelength.
        n_medium : number, optional
            Refractive index of the medium above the stack.
        pol : {"TE", "TM", "unpolarized"}, optional
            "unpolarized" is the mean of TE and TM.

        Returns
        -------
        R : number or array
        """

        if pol == "unpolarized":
            return np.mean([self.reflectance(na, wavelength, n_medium, p) for p in ("TE", "TM")], axis=0)
        n, t = self._media(n_medium)
        r = _r_t(np.asarray(na, dtype=float)/n_medium, n, t, pol, wavelength)[0]

        return np.abs(r)**2

    def substrate_reflectance(self, na, wavelength, n_medium=1.0, pol="unpolarized"):
        """Reflectance ``|r_b|**2`` of the layers below the resist, seen from the resist.

        Parameters
        ----------
        na : number or array
            Transverse wave numbers in units of k_0.
        wavelength : number
            Vacuum wavelength.
        n_medium : number, optional
            Refractive index of the medium above the stack.
        pol : {"TE", "TM", "unpolarized"}, optional

        Returns
        -------
        R : number or array
        """

        if pol == "unpolarized":
            return np.mean([self.substrate_reflectance(na, wavelength, n_medium, p)
                            for p in ("TE", "TM")], axis=0)
        k0 = 2*np.pi/wavelength

        return np.abs(self.coefficients(k0*np.asarray(na, dtype=float), k0, n_medium, pol)[2])**2

    def absorbed(self, na, wavelength, n_medium=1.0, pol="unpolarized"):
        """Fraction of the incident power absorbed in the resist.

        Parameters
        ----------
        na : number or array
            Transverse wave numbers in units of k_0.
        wavelength : number
            Vacuum wavelength.
        n_medium : number, optional
            Refractive index of the medium above the stack.
        pol : {"TE", "TM", "unpolarized"}, optional

        Returns
        -------
        A : number or array
        """

        if pol == "unpolarized":
            return np.mean([self.absorbed(na, wavelength, n_medium, p) for p in ("TE", "TM")], axis=0)
        k0 = 2*np.pi/wavelength
        k_perp = k0*np.asarray(na, dtype=float)
        cos_m = np.sqrt(1 - (np.asarray(na, dtype=float)/n_medium)**2)
        d = self.thickness
        t_top, r_top, r_b, k_z = self.coefficients(k_perp, k0, n_medium, pol)
        # tangential field a*exp(i*k_z*z) + b*exp(-i*k_z*z) for a unit incident field
        # (its tangential part is cos(theta) for TM):
        a = t_top/(1 - r_top*r_b*np.exp(2j*k_z*d))*(cos_m if pol == "TM" else 1.0)
        b = a*r_b*np.exp(2j*k_z*d)
        beta, kappa = k_z.real, k_z.imag

        def integral(c):  # int_0^d exp(c*z) dz
            c = np.asarray(c, dtype=complex)
            small = np.abs(c*d) < 1e-8
            c_safe = np.where(small, 1.0, c)
            return np.where(small, d*(1 + c*d/2), np.expm1(c_safe*d)/c_safe)

        down_up = np.abs(a)**2*integral(-2*kappa).real + np.abs(b)**2*integral(2*kappa).real
        cross = 2*(a*b.conj()*integral(2j*beta)).real
        E2 = down_up + cross
        if pol == "TM":  # E_z = -k_perp/k_z*(a*exp(i*k_z*z) - b*exp(-i*k_z*z))
            E2 = E2 + np.abs(k_perp/k_z)**2*(down_up - cross)

        return k0*(self.resist_n**2).imag*E2/(n_medium*cos_m)


def _average(values, weights):
    values = np.asarray(values)
    if weights is None:
        return values.mean(axis=-1)
    w = np.asarray(weights, dtype=float)

    return (values*w).sum(axis=-1)/w.sum()


class SwingCurve(object):
    """Result of swing_curve(): thickness, reflectance (of the whole
    stack), absorbed (fraction of the incident power absorbed in the resist)
    and substrate_reflectance (seen from the resist), averaged over the
    given angles; swing() gives the swing ratio (max - min)/mean of a
    curve."""

    def __init__(self, thickness, reflectance, absorbed, substrate_reflectance):
        self.thickness = thickness
        self.reflectance, self.absorbed = reflectance, absorbed
        self.substrate_reflectance = substrate_reflectance

    @staticmethod
    def swing(values):
        values = np.asarray(values)

        return (values.max() - values.min())/values.mean()


def swing_curve(wafer, thicknesses, wavelength, n_medium=1.0, na=0.0, weights=None,
                pol="unpolarized"):
    """Reflectance and absorbed power of a wafer stack through resist
    thickness.

    Parameters
    ----------
    wafer : WaferStack
    thicknesses : array
        Resist thicknesses.
    wavelength : number
    n_medium : number, optional
        Refractive index of the medium above the stack (immersion liquid).
    na : number or array, optional
        Transverse wave numbers k_perp/k_0 of the plane waves (0: normal
        incidence); the results are averaged over them with the weights.
    weights : array, optional
    pol : "TE", "TM" or "unpolarized", optional

    Returns
    -------
    SwingCurve
    """

    na = np.atleast_1d(np.asarray(na, dtype=float))
    R, A, R_b = [], [], []
    for d in thicknesses:
        w = wafer.with_(thickness=d)
        R.append(w.reflectance(na, wavelength, n_medium, pol))
        A.append(w.absorbed(na, wavelength, n_medium, pol))
        R_b.append(w.substrate_reflectance(na, wavelength, n_medium, pol))

    return SwingCurve(np.asarray(thicknesses, dtype=float), _average(R, weights), _average(A, weights),
                      _average(R_b, weights))


def optimize_barc(wafer, barc, thicknesses, wavelength, n_medium=1.0, na=0.0, weights=None,
                  pol="unpolarized"):
    """Thickness of a bottom anti-reflective coating with the lowest
    substrate reflectance seen from the resist, averaged over the plane
    waves k_0*na with the weights.

    Parameters
    ----------
    wafer : WaferStack
        The BARC goes directly below the resist, above the layers already
        in wafer.below.
    barc : tuple (n, k)
    thicknesses : array
        BARC thicknesses to scan.
    wavelength, n_medium, na, weights, pol :
        See swing_curve().

    Returns
    -------
    best : number
        Thickness with the lowest reflectance (scanned value, refined by a
        parabola through its neighbours).
    reflectance : array
        Averaged substrate reflectance for all thicknesses.
    """

    na = np.atleast_1d(np.asarray(na, dtype=float))
    R = np.array([_average(wafer.with_(below=[(barc[0], barc[1], t)] + list(wafer.below))
                           .substrate_reflectance(na, wavelength, n_medium, pol), weights)
                  for t in thicknesses])
    i = int(np.argmin(R))
    best = float(thicknesses[i])
    if 0 < i < len(R) - 1:
        t3, R3 = np.asarray(thicknesses[i - 1:i + 2], dtype=float), R[i - 1:i + 2]
        c = np.polyfit(t3, R3, 2)
        if c[0] > 0:
            best = float(np.clip(-c[1]/(2*c[0]), t3[0], t3[2]))

    return best, R
