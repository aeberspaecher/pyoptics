#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Illumination sources for lithographic imaging.

A source is a set of mutually incoherent points (s_x, s_y) in the sigma
plane (the pupil of the illuminator, normalized to the NA of the projection
lens: the illuminating plane wave of a point has the image-side transverse
wave vector k_0*NA*(s_x, s_y)) with weights that add up to 1, plus a
polarization.

Common shapes are sampled on a square grid of n_samples points across the
unit circle:

- coherent: one point,
- conventional: disk of radius sigma,
- annular: sigma_in <= ``|s|`` <= sigma_out,
- dipole: two poles on the x (or y) axis,
- quadrupole: four poles on the x and y axes (also called C-quad or cross
  quadrupole),
- quasar: four poles on the diagonals (45°, 135°, ...),
- multipole: any number of poles at given azimuths,
- from_map: any (pixelated, freeform) intensity map.

Poles are annular sectors between sigma_in and sigma_out with an opening
angle ("sector", the default), or disks that fill the ring between sigma_in
and sigma_out ("circle").

Polarization is one of

- "x", "y", "unpolarized",
- "TE" (= "azimuthal", "tangential": perpendicular to the radius vector of
  each source point) or "TM" (= "radial"); on the axis they fall back to y and
  x, see polarization.TE_TM_basis(),
- a Jones vector [E_x, E_y] or a Stokes vector [S_0, S_1, S_2, S_3],
- a callable pol(s_x, s_y) returning one of the above for each point.

A degree of polarization dop < 1 mixes the polarization with unpolarized
light: the fraction dop of the intensity is in the given state, the rest is
unpolarized. For "TE" the intensity in the preferred state (IPS) is then
(1 + dop)/2.
"""

import numpy as np

from ..vectorial_imaging import _polarization_states


def _angle_dist(phi, phi_0):
    """Angular distance in degrees, in [0, 180]."""

    return np.abs((phi - phi_0 + 180) % 360 - 180)


def _grid(n_samples):
    s = np.linspace(-1, 1, n_samples)
    S_x, S_y = np.meshgrid(s, s)

    return S_x.ravel(), S_y.ravel()


class Illumination(object):
    """Partially coherent, polarized illumination.

    Parameters
    ----------
    s_x, s_y : arrays
        Source points in the sigma plane.
    weights : array, optional
        Relative intensities; normalized to a sum of 1. Default: equal.
    polarization : see module docstring
    dop : number, optional
        Degree of polarization in [0, 1].
    name : str, optional
        Label, e.g. for plots.
    """

    def __init__(self, s_x, s_y, weights=None, polarization="unpolarized", dop=1.0,
                 name=None):

        self.s_x = np.atleast_1d(np.asarray(s_x, dtype=float))
        self.s_y = np.atleast_1d(np.asarray(s_y, dtype=float))
        if self.s_x.shape != self.s_y.shape:
            raise ValueError("s_x and s_y must have the same shape")
        if len(self.s_x) == 0:
            raise ValueError("The source has no points; increase n_samples")
        w = np.ones(self.s_x.shape) if weights is None else np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self.weights = w/w.sum()

        if not 0 <= dop <= 1:
            raise ValueError("dop must be in [0, 1]")
        self.polarization = polarization
        self.dop = dop
        self.name = name

    # shapes:

    @classmethod
    def coherent(cls, s_x=0.0, s_y=0.0, **kwargs):
        """One source point at (s_x, s_y)."""

        kwargs.setdefault("name", "coherent")

        return cls([s_x], [s_y], **kwargs)

    @classmethod
    def conventional(cls, sigma, n_samples=31, **kwargs):
        """Disk of radius sigma."""

        S_x, S_y = _grid(n_samples)
        inside = np.hypot(S_x, S_y) <= sigma + 1E-12
        kwargs.setdefault("name", "conventional σ={:g}".format(sigma))

        return cls(S_x[inside], S_y[inside], **kwargs)

    @classmethod
    def annular(cls, sigma_out, sigma_in, n_samples=31, **kwargs):
        """Ring sigma_in <= ``|s|`` <= sigma_out."""

        S_x, S_y = _grid(n_samples)
        r = np.hypot(S_x, S_y)
        inside = (r <= sigma_out + 1E-12) & (r >= sigma_in - 1E-12)
        kwargs.setdefault("name", "annular {:g}–{:g}".format(sigma_in, sigma_out))

        return cls(S_x[inside], S_y[inside], **kwargs)

    @classmethod
    def multipole(cls, angles, sigma_out, sigma_in, opening_angle=35.0, pole="sector",
                  n_samples=31, **kwargs):
        """Poles at the azimuths angles (degrees, counterclockwise from the x
        axis).

        Parameters
        ----------
        angles : sequence of numbers
            Azimuths of the pole centers in degrees.
        sigma_out, sigma_in : numbers
            Outer and inner radius of the poles.
        opening_angle : number, optional
            Opening angle of "sector" poles in degrees.
        pole : "sector" or "circle", optional
            Annular sectors, or disks of radius (sigma_out - sigma_in)/2
            centered at the radius (sigma_out + sigma_in)/2.
        n_samples : int, optional
            Grid points across the unit circle.
        """

        S_x, S_y = _grid(n_samples)
        r, phi = np.hypot(S_x, S_y), np.degrees(np.arctan2(S_y, S_x))
        if pole == "sector":
            in_ring = (r <= sigma_out + 1E-12) & (r >= sigma_in - 1E-12)
            inside = in_ring & np.any([_angle_dist(phi, a) <= opening_angle/2 + 1E-9
                                       for a in angles], axis=0)
        elif pole == "circle":
            c, rad = (sigma_out + sigma_in)/2, (sigma_out - sigma_in)/2
            inside = np.any([np.hypot(S_x - c*np.cos(np.radians(a)), S_y - c*np.sin(np.radians(a)))
                             <= rad + 1E-12 for a in angles], axis=0)
        else:
            raise ValueError("pole must be 'sector' or 'circle'")
        kwargs.setdefault("name", "multipole")

        return cls(S_x[inside], S_y[inside], **kwargs)

    @classmethod
    def dipole(cls, sigma_out, sigma_in, opening_angle=35.0, axis="x", pole="sector",
               n_samples=31, **kwargs):
        """Two poles on the x axis (axis="x"), the y axis ("y") or at the
        azimuth axis (a number in degrees) and opposite."""

        a = {"x": 0.0, "y": 90.0}.get(axis, axis)
        kwargs.setdefault("name", "dipole {} {:g}–{:g}".format(axis, sigma_in, sigma_out))

        return cls.multipole((a, a + 180), sigma_out, sigma_in, opening_angle, pole, n_samples,
                             **kwargs)

    @classmethod
    def dipole_for_pitch(cls, pitch, wavelength, NA, width=0.2, opening_angle=35.0, axis="x",
                         **kwargs):
        """Dipole adapted to the pitch: poles of the radial width centered at
        s_c = wavelength/(2*pitch*NA), where orders 0 and -1 pass the pupil
        symmetrically (two-beam imaging); s_c is limited to 1 - width/2."""

        s_c = min(wavelength/(2*pitch*NA), 1 - width/2)
        kwargs.setdefault("name", "dipole {} at σ={:.2f}".format(axis, s_c))

        return cls.dipole(s_c + width/2, s_c - width/2, opening_angle, axis, **kwargs)

    @classmethod
    def quadrupole(cls, sigma_out, sigma_in, opening_angle=30.0, rotation=0.0, pole="sector",
                   n_samples=31, **kwargs):
        """Four poles on the x and y axes (rotated by rotation degrees)."""

        kwargs.setdefault("name", "quadrupole {:g}–{:g}".format(sigma_in, sigma_out))

        return cls.multipole([rotation + a for a in (0, 90, 180, 270)], sigma_out, sigma_in,
                             opening_angle, pole, n_samples, **kwargs)

    @classmethod
    def quasar(cls, sigma_out, sigma_in, opening_angle=30.0, pole="sector", n_samples=31,
               **kwargs):
        """Four poles on the diagonals (a quadrupole rotated by 45°)."""

        kwargs.setdefault("name", "quasar {:g}–{:g}".format(sigma_in, sigma_out))

        return cls.quadrupole(sigma_out, sigma_in, opening_angle, 45.0, pole, n_samples, **kwargs)

    @classmethod
    def from_map(cls, S_x, S_y, intensity, threshold=0.0, **kwargs):
        """Freeform (pixelated) source from an intensity map on the points
        S_x, S_y (e.g. a meshgrid); points with intensity <= threshold or
        outside the unit circle are dropped."""

        S_x, S_y, intensity = (np.ravel(a) for a in (S_x, S_y, intensity))
        keep = (intensity > threshold) & (np.hypot(S_x, S_y) <= 1 + 1E-12)
        kwargs.setdefault("name", "freeform")

        return cls(S_x[keep], S_y[keep], intensity[keep], **kwargs)

    # polarization:

    def with_polarization(self, polarization, dop=1.0):
        """Copy of this source with another polarization."""

        return Illumination(self.s_x, self.s_y, self.weights, polarization, dop, self.name)

    def polarization_states(self, s_x, s_y):
        """Mutually incoherent polarization states of the source point (s_x,
        s_y): a list of (weight, Jones vector), weights adding up to 1."""

        states = _polarization_states(self.polarization, s_x, s_y)
        if self.dop < 1 and not (isinstance(self.polarization, str)
                                 and self.polarization.lower() == "unpolarized"):
            unpol = 0.5*(1 - self.dop)
            states = ([(self.dop*w, J) for w, J in states]
                      + [(unpol, np.array([1.0, 0.0])), (unpol, np.array([0.0, 1.0]))])

        return [(w, np.asarray(J, dtype=complex)) for w, J in states if w > 1E-15]

    # access:

    def __len__(self):
        return len(self.s_x)

    def __iter__(self):
        """Yield (s_x, s_y, weight, polarization states) for all points."""

        for s_x, s_y, w in zip(self.s_x, self.s_y, self.weights):
            yield s_x, s_y, w, self.polarization_states(s_x, s_y)

    @property
    def sigma_max(self):
        """Largest ``|s|`` of the source."""

        return float(np.max(np.hypot(self.s_x, self.s_y)))

    def points(self):
        """Return s_x, s_y, weights (as vectorial_imaging.source_points())."""

        return self.s_x, self.s_y, self.weights

    def __repr__(self):
        return "Illumination({}, {} points, polarization={!r}, dop={:g})".format(
            self.name, len(self), self.polarization, self.dop)
