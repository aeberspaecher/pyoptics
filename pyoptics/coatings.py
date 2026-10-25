#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film coatings on the surfaces of a ray traced system.

A `Coating` is a stack of `Layer` objects (material and thickness) on a
surface of `pyoptics.raytrace`. The layers are listed in the order the light
meets them: from the medium in front of the surface (in the sequence of the
system) to the medium behind it, or for a mirror to its `substrate`. Every
ray sees the stack as a plane stack in the tangent plane of the surface at
its hit point (the layers are thin compared to the radii of curvature); its
angle of incidence is taken against the local surface normal, and the layer
thicknesses are measured along that normal. `pyoptics.thin_films` gives the
amplitude coefficients.

Thickness variations
--------------------
Coating thicknesses vary across a surface (e.g. thinner towards the rim of a
strongly curved lens). A layer's `profile` is the relative thickness
t(x, y)/t_0 as a function of the local surface coordinates x, y (the vertex
plane of the surface in its own, possibly decentred and tilted frame):

- `RadialProfile`: 1 + a_2 rho**2 + a_4 rho**4 + ...,
- `PolynomialProfile`: 1 + sum c_ij x_n**i y_n**j,
- `ZernikeProfile`: 1 + sum c_j Z_j(x_n, y_n) (Fringe Zernikes),

with rho, x_n, y_n normalized by r_norm. Any function f(x, y) works as a
profile too.

Convention: exp(i(k.r - omega*t)) as in `pyoptics.thin_films`; absorbing
media have n + i*k with k > 0. r and t are the ratios of the tangential
electric field components (Zangwill), see `Coating.amplitudes()`.
"""

import numpy as np

from .bases import fringe_zernike_sum
from .materials import complex_index
from .thin_films import _r_t, _impedances, _tilted_Z


# thickness profiles:
class RadialProfile(object):
    """Relative thickness 1 + a_2 rho**2 + a_4 rho**4 + ..., rho = r/r_norm.

    Parameters
    ----------
    coefficients : sequence
        (a_2, a_4, ...) of the even powers of rho.
    r_norm : number, optional
        Normalization radius (e.g. the semi diameter), in system units.
    """

    def __init__(self, coefficients, r_norm=1.0):
        self.coefficients, self.r_norm = tuple(coefficients), r_norm

    def __repr__(self):
        return "RadialProfile({}, r_norm={})".format(self.coefficients, self.r_norm)

    def __call__(self, x, y):
        rho2 = (np.asarray(x)**2 + np.asarray(y)**2)/self.r_norm**2
        f = np.ones(np.shape(rho2))
        for j, a in enumerate(self.coefficients):
            f = f + a*rho2**(j + 1)

        return f


class PolynomialProfile(object):
    """Relative thickness 1 + sum c_ij (x/r_norm)**i (y/r_norm)**j.

    Parameters
    ----------
    coefficients : dict
        {(i, j): c_ij}, e.g. {(1, 0): 0.02} for a linear wedge along x.
    r_norm : number, optional
    """

    def __init__(self, coefficients, r_norm=1.0):
        self.coefficients, self.r_norm = dict(coefficients), r_norm

    def __repr__(self):
        return "PolynomialProfile({}, r_norm={})".format(self.coefficients, self.r_norm)

    def __call__(self, x, y):
        x_n, y_n = np.asarray(x)/self.r_norm, np.asarray(y)/self.r_norm
        f = np.ones(np.broadcast(x_n, y_n).shape)
        for (i, j), c in self.coefficients.items():
            f = f + c*x_n**i*y_n**j

        return f


class ZernikeProfile(object):
    """Relative thickness 1 + sum c_j Z_j(x/r_norm, y/r_norm) with the
    Fringe Zernikes of `pyoptics.bases` (Z_1 = 1 is a uniform change).

    Parameters
    ----------
    coefficients : dict
        {j: c_j}.
    r_norm : number, optional
    """

    def __init__(self, coefficients, r_norm=1.0):
        self.coefficients, self.r_norm = dict(coefficients), r_norm

    def __repr__(self):
        return "ZernikeProfile({}, r_norm={})".format(self.coefficients, self.r_norm)

    def __call__(self, x, y):
        return 1 + fringe_zernike_sum(self.coefficients, np.asarray(x)/self.r_norm,
                                      np.asarray(y)/self.r_norm)


# layers and stacks:
class Layer(object):
    """A homogeneous thin film.

    Parameters
    ----------
    material : number, callable, (n_d, V_d) or materials.Material
        Refractive index n + i*k (k > 0 absorbing); anisotropic materials
        enter with their mean index.
    thickness : number
        Nominal (physical) thickness in system units.
    profile : callable, optional
        Relative thickness f(x, y) on the local surface coordinates, e.g.
        RadialProfile; None: uniform.
    label : str, optional
    """

    def __init__(self, material, thickness, profile=None, label=None):
        self.material, self.thickness, self.profile, self.label = material, thickness, profile, label

    def __repr__(self):
        return "Layer({!r}, {}, profile={!r})".format(self.material, self.thickness, self.profile)

    @classmethod
    def quarter_wave(cls, material, wavelength, wavelength_nm=None, profile=None, label=None):
        """Layer of optical thickness wavelength/4 at normal incidence
        (wavelength in system units, the material evaluated at
        wavelength_nm)."""

        n = complex_index(material, wavelength_nm).real

        return cls(material, wavelength/(4*n), profile, label)

    def index(self, wavelength_nm=None):
        return complex_index(self.material, wavelength_nm)

    def thickness_at(self, x=None, y=None):
        """Thicknesses at the local surface points (x, y)."""

        if self.profile is None or x is None:
            return self.thickness

        return self.thickness*self.profile(x, y)


class Coating(object):
    """Thin film stack on a surface.

    Parameters
    ----------
    layers : sequence of Layer or (material, thickness)
        From the medium in front of the surface (in the order of the system)
        towards the medium behind it (or the substrate of a mirror).
    substrate : number, callable, (n_d, V_d) or materials.Material, optional
        Substrate of a reflecting surface (e.g. a metal, n + i*k). Ignored
        for refracting surfaces, whose substrate is the medium behind the
        surface.
    label : str, optional
    """

    def __init__(self, layers=(), substrate=None, label=None):
        self.layers = [l if isinstance(l, Layer) else Layer(*l) for l in layers]
        self.substrate, self.label = substrate, label

    def __repr__(self):
        return "Coating({!r}, substrate={!r})".format(self.layers, self.substrate)

    def reversed(self):
        """The same stack seen from the other side."""

        return Coating(self.layers[::-1], self.substrate, self.label)

    def _stack(self, n1, n2, wavelength_nm, x, y):
        n2 = complex_index(self.substrate, wavelength_nm) if n2 is None else n2
        n = np.array([n1] + [l.index(wavelength_nm) for l in self.layers] + [n2], dtype=complex)
        t = [l.thickness_at(x, y) for l in self.layers]

        return n, t

    def amplitudes(self, cos_i, n1, n2, wavelength, wavelength_nm=None, x=None, y=None):
        """Amplitude coefficients of the stack for the angles of incidence
        with the cosines cos_i (array) in the medium n1.

        Parameters
        ----------
        cos_i : array
        n1, n2 : numbers
            Indices in front of and behind the stack; n2=None: the substrate.
        wavelength : number
            Vacuum wavelength in the unit of the layer thicknesses.
        wavelength_nm : number, optional
            Wavelength for the dispersion of the materials.
        x, y : arrays, optional
            Local surface coordinates of the rays for the thickness profiles.

        Returns
        -------
        r_s, r_p, t_s, t_p : arrays
            Ratios of the tangential electric field components (the
            convention of thin_films.r_t_coeffs()).
        """

        cos_i = np.asarray(cos_i, dtype=float)
        sin_i = np.sqrt(np.clip(1 - cos_i**2, 0, None))
        n, t = self._stack(n1, n2, wavelength_nm, x, y)
        r_s, t_s = _r_t(sin_i, n, t, "TE", wavelength)
        r_p, t_p = _r_t(sin_i, n, t, "TM", wavelength)

        return r_s, r_p, t_s, t_p

    def power(self, angle, n1, n2=None, wavelength=None, wavelength_nm=None, x=None, y=None):
        """Reflectance and transmittance of the stack (e.g. to check a
        design over the angle of incidence).

        Parameters as for amplitudes(), but with the angles of incidence (in
        radians) in the medium n1 (real).

        Returns
        -------
        dict
            "R_s", "R_p", "T_s", "T_p" and the absorptances "A_s", "A_p".
        """

        angle = np.asarray(angle, dtype=float)
        r_s, r_p, t_s, t_p = self.amplitudes(np.cos(angle), n1, n2, wavelength, wavelength_nm, x, y)
        n = np.array([n1, complex_index(self.substrate, wavelength_nm) if n2 is None else n2],
                     dtype=complex)
        cos = np.sqrt(1 - (n[0]*np.sin(angle)/n.reshape(-1, *(1,)*angle.ndim))**2 + 0j)
        out = {}
        for pol, r, t in (("s", r_s, t_s), ("p", r_p, t_p)):
            Z = _tilted_Z(_impedances(n).reshape(cos.shape[:1] + (1,)*angle.ndim)*np.ones_like(cos),
                          cos, pol)
            out["R_" + pol] = np.abs(r)**2
            out["T_" + pol] = np.abs(t)**2*np.real(1/Z[-1])/np.real(1/Z[0])
            out["A_" + pol] = 1 - out["R_" + pol] - out["T_" + pol]

        return out
