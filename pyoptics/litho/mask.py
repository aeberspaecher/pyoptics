#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Masks for lithographic imaging: layouts (geometry) and mask stacks
(materials and the model that turns them into transmissions).

Layout
------
A MaskLayout is one period (pitch_x, pitch_y) of a periodic mask that holds
rectangular features on a background (the field). A pitch of None makes the
mask invariant along that direction: lines along y have pitch_y = None. With
tone="clear" the field is clear and the features are absorber (lines on a
clear field mask), with tone="dark" the field is absorber and the features
are clear (contact holes, trenches). Isolated features are periodic features
with a pitch large enough that the neighbours do not interact (the coherence
length of the illumination, a few lambda/NA).

The diffraction orders are the Fourier coefficients of the transmission on
one period. They are computed analytically from the rectangles (products of
sinc functions), so there is no pixelation of the mask.

Mask stacks and approximations
------------------------------
A mask stack gives the transmissions of the clear and absorber regions as
Jones matrices for an illuminating plane wave (source point (s_x, s_y)):

- ThinMask: Kirchhoff (thin mask) approximation with given complex
  transmissions, e.g. binary (chrome on glass, t = 0) or attenuated phase
  shift masks (t = sqrt(T)*exp(i*phase)); independent of angle and
  polarization.
- FilmMask: the absorber is a stack of thin films (n + i*k, thickness) on
  the mask blank. Absorber and clear regions are taken as laterally infinite
  film stacks (a thin mask whose local transmissions are those of the films):
  their transmissions t_TE and t_TM are computed with pyoptics.thin_films for
  the angle of incidence of each source point at the mask, n*sin(theta) =
  NA*|s|/M (M: reduction ratio), and for TE and TM with respect to its plane
  of incidence. Phase and amplitude of the absorber, e.g. of an attenuated
  PSM, follow from the material data instead of being set by hand.
- BoundaryLayer (optional for both): thick mask (mask 3D) effects at the
  edges of the absorber are approximated by strips of width w centered on
  every edge, with a complex transmission that depends on whether the
  electric field is parallel (TE) or perpendicular (TM) to the edge
  (Tirapu-Azpiroz and Yablonovitch's boundary layer model). Its parameters
  come from a rigorous simulation or a calibration. At the corners of
  rectangles, the strips along y are taken to cover the corner squares.

Transmissions are relative to the clear area: the clear region has the
transmission 1 (FilmMask normalizes to the clear region unless
normalize=False), so that images can be normalized to the open frame.
"""

import numpy as np

from ..polarization import TE_TM_basis
from ..thin_films import _r_t


# layouts:

def _rect_coeffs(m, P, center, size):
    """1D Fourier coefficients of a rect of width size at center on the
    period P; P None: invariant direction (only order 0, coefficient 1)."""

    m = np.asarray(m)
    if P is None:
        return np.ones(m.shape, dtype=complex)

    return size/P*np.sinc(m*size/P)*np.exp(-2j*np.pi*m*center/P)


class MaskLayout(object):
    """Geometry of one period of a mask.

    Parameters
    ----------
    pitch_x, pitch_y : numbers or None
        Periods; None for an invariant direction (at most one).
    features : sequence of (x0, y0, w_x, w_y)
        Centers and sizes of rectangles (must not overlap). Sizes along an
        invariant direction are ignored.
    tone : "clear" or "dark"
        Clear field (features are absorber) or dark field (features are
        clear).
    name : str, optional
    """

    def __init__(self, pitch_x, pitch_y=None, features=(), tone="clear", name=None):

        if pitch_x is None and pitch_y is None:
            raise ValueError("At least one direction must be periodic")
        if tone not in ("clear", "dark"):
            raise ValueError("tone must be 'clear' or 'dark'")
        self.pitch_x, self.pitch_y = pitch_x, pitch_y
        self.features = [tuple(float(v) for v in f) for f in features]
        self.tone = tone
        self.name = name

    # common layouts:

    @classmethod
    def lines_spaces(cls, pitch, cd, orientation="y", tone="clear", name=None):
        """Lines and spaces: lines of width cd (absorber for tone="clear",
        clear trenches for tone="dark") along y (orientation="y", pitch along
        x) or along x."""

        name = name or "L/S {:g}/{:g}".format(cd, pitch - cd)
        if orientation == "y":
            return cls(pitch, None, [(0.0, 0.0, cd, 0.0)], tone, name)
        if orientation == "x":
            return cls(None, pitch, [(0.0, 0.0, 0.0, cd)], tone, name)
        raise ValueError("orientation must be 'x' or 'y'")

    @classmethod
    def iso_line(cls, cd, pitch=None, orientation="y", tone="clear", name=None):
        """Isolated line (or trench, tone="dark"); the default pitch is
        20*cd."""

        return cls.lines_spaces(20*cd if pitch is None else pitch, cd, orientation, tone,
                                name or "iso line {:g}".format(cd))

    @classmethod
    def contact_holes(cls, pitch, cd, pitch_y=None, cd_y=None, tone="dark", name=None):
        """Array of rectangular contact holes cd x cd_y on the pitches pitch
        x pitch_y (defaults: square holes on a square grid). Dark field:
        clear holes in absorber."""

        pitch_y = pitch if pitch_y is None else pitch_y
        cd_y = cd if cd_y is None else cd_y
        name = name or "contacts {:g} @ {:g}".format(cd, pitch)

        return cls(pitch, pitch_y, [(0.0, 0.0, cd, cd_y)], tone, name)

    @classmethod
    def iso_contact(cls, cd, pitch=None, cd_y=None, tone="dark", name=None):
        """Isolated contact hole; the default pitch is 10*cd."""

        return cls.contact_holes(10*cd if pitch is None else pitch, cd, None, cd_y, tone,
                                 name or "iso contact {:g}".format(cd))

    @classmethod
    def staggered_contacts(cls, pitch_x, pitch_y, cd, cd_y=None, tone="dark", name=None):
        """Staggered (brick) contact array: rows at the distance pitch_y,
        every other row shifted by pitch_x/2."""

        cd_y = cd if cd_y is None else cd_y

        return cls(pitch_x, 2*pitch_y, [(0.0, -pitch_y/2, cd, cd_y), (pitch_x/2, pitch_y/2, cd, cd_y)],
                   tone, name or "staggered contacts {:g}".format(cd))

    # properties:

    @property
    def is_1d(self):
        return self.pitch_x is None or self.pitch_y is None

    @property
    def feature_is_bright(self):
        """True if the features transmit (dark field mask)."""

        return self.tone == "dark"

    def scaled(self, factor_cd=1.0, bias=0.0):
        """Copy with all feature sizes multiplied by factor_cd and enlarged
        by bias (e.g. a mask CD error for MEEF)."""

        feats = [(x0, y0, w*factor_cd + bias, h*factor_cd + bias) for x0, y0, w, h in self.features]

        return MaskLayout(self.pitch_x, self.pitch_y, feats, self.tone, self.name)

    # diffraction:

    def max_orders(self, NA, wavelength, sigma_max=1.0):
        """Largest order indices (m, n) that can pass a pupil of radius NA
        for source points up to sigma_max."""

        f_max = NA*(1 + sigma_max)/wavelength
        M = 0 if self.pitch_x is None else int(np.floor(self.pitch_x*f_max)) + 1
        N = 0 if self.pitch_y is None else int(np.floor(self.pitch_y*f_max)) + 1

        return M, N

    def region_coefficients(self, m, n, boundary_width=0.0):
        """Fourier coefficients of the regions of the layout.

        Parameters
        ----------
        m, n : arrays of ints
            Order indices along x and y (1D each).
        boundary_width : number, optional
            Width of the boundary layer strips.

        Returns
        -------
        c : dict of arrays of shape (len(n), len(m))
            "field", "feature" and, for boundary_width > 0, "edge_y" (strips
            along edges parallel to y) and "edge_x" (edges parallel to x).
            The coefficients of all regions add up to the delta of order 0.
        """

        m, n = np.atleast_1d(m), np.atleast_1d(n)
        Px, Py, w = self.pitch_x, self.pitch_y, boundary_width
        shape = (len(n), len(m))

        def rect(x0, y0, a, b):
            return np.outer(_rect_coeffs(n, Py, y0, b), _rect_coeffs(m, Px, x0, a))

        c = {"feature": np.zeros(shape, dtype=complex)}
        outer = np.zeros(shape, dtype=complex)
        if w > 0:
            c["edge_y"] = np.zeros(shape, dtype=complex)
            c["edge_x"] = np.zeros(shape, dtype=complex)
        for x0, y0, a, b in self.features:
            if w <= 0:
                c["feature"] += rect(x0, y0, a, b)
                outer += rect(x0, y0, a, b)
                continue
            if (Px is not None and a <= w) or (Py is not None and b <= w):
                raise ValueError("Feature smaller than the boundary layer width")
            c["feature"] += rect(x0, y0, a - w, b - w)
            outer += rect(x0, y0, a + w, b + w)
            if Px is not None:  # edges parallel to y at x0 -+ a/2, including the corners
                for sgn in (-1, 1):
                    c["edge_y"] += rect(x0 + sgn*a/2, y0, w, b + w)
            if Py is not None:  # edges parallel to x, between the corners
                for sgn in (-1, 1):
                    c["edge_x"] += rect(x0, y0 + sgn*b/2, a - w, w)

        delta = np.outer(n == 0, m == 0).astype(complex)
        c["field"] = delta - outer

        return c

    def transmission(self, x, y=None, t_clear=1.0, t_absorber=0.0):
        """Sample the (scalar, thin mask) transmission on the grid x, y
        (for plots; y=None gives a cut at y = 0)."""

        X, Y = np.meshgrid(np.atleast_1d(x), np.atleast_1d(0.0 if y is None else y))
        inside = np.zeros(X.shape, dtype=bool)
        for x0, y0, a, b in self.features:
            in_x = True if self.pitch_x is None else (
                np.abs((X - x0 + self.pitch_x/2) % self.pitch_x - self.pitch_x/2) < a/2)
            in_y = True if self.pitch_y is None else (
                np.abs((Y - y0 + self.pitch_y/2) % self.pitch_y - self.pitch_y/2) < b/2)
            inside |= in_x & in_y
        t_feat, t_field = (t_clear, t_absorber) if self.tone == "dark" else (t_absorber, t_clear)
        t = np.where(inside, t_feat, t_field)

        return t[0] if y is None else t

    def __repr__(self):
        return "MaskLayout({}, pitch=({}, {}), tone={})".format(self.name, self.pitch_x,
                                                                 self.pitch_y, self.tone)


# mask stacks:

class BoundaryLayer(object):
    """Boundary layer at the absorber edges.

    Parameters
    ----------
    width : number
        Width of the strip centered on each edge (wafer scale).
    t_TE, t_TM : complex numbers
        Transmission of the strip for the electric field parallel (TE) and
        perpendicular (TM) to the edge.
    """

    def __init__(self, width, t_TE, t_TM):
        self.width, self.t_TE, self.t_TM = width, complex(t_TE), complex(t_TM)

    def jones(self):
        """Jones matrices of the strips along y and along x."""

        return np.diag([self.t_TM, self.t_TE]), np.diag([self.t_TE, self.t_TM])


class ThinMask(object):
    """Kirchhoff (thin mask) model with given transmissions.

    Parameters
    ----------
    t_absorber : complex, optional
        Amplitude transmission of the absorber (default 0: binary mask).
    t_clear : complex, optional
    boundary_layer : BoundaryLayer, optional
    name : str, optional
    """

    def __init__(self, t_absorber=0.0, t_clear=1.0, boundary_layer=None, name=None):
        self.t_absorber, self.t_clear = complex(t_absorber), complex(t_clear)
        self.boundary_layer = boundary_layer
        self.name = name or "thin mask t={:.3g}".format(self.t_absorber)

    @classmethod
    def binary(cls, **kwargs):
        """Binary (chrome on glass) mask."""

        kwargs.setdefault("name", "binary")

        return cls(0.0, **kwargs)

    @classmethod
    def att_psm(cls, transmission=0.06, phase=180.0, **kwargs):
        """Attenuated phase shift mask with the intensity transmission and
        the phase (degrees) of the absorber."""

        kwargs.setdefault("name", "att. PSM {:g} %".format(100*transmission))

        return cls(np.sqrt(transmission)*np.exp(1j*np.radians(phase)), **kwargs)

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices (J_clear, J_absorber) for the source point."""

        return self.t_clear*np.eye(2), self.t_absorber*np.eye(2)


class FilmMask(object):
    """Mask whose absorber is a thin film stack on the mask blank.

    Light comes from the blank (substrate_n), passes the absorber layers in
    the given order and leaves into the exit medium (air). The clear region
    is the bare blank; its transmission is taken at the same plane as the
    absorber's (bottom of the absorber), i.e. through exit medium of the same
    thickness.

    Parameters
    ----------
    n, k, t : sequences
        Refractive indices n + i*k and thicknesses of the absorber layers,
        from the blank outwards (mask scale thicknesses).
    substrate_n, substrate_k : numbers, optional
        Mask blank (fused silica at 193 nm: about 1.563).
    exit_n : number, optional
        Medium between mask and projection lens.
    normalize : bool, optional
        Divide by the clear transmission (per TE/TM), so that the clear
        region transmits 1.
    boundary_layer : BoundaryLayer, optional
    name : str, optional
    """

    def __init__(self, n, k, t, substrate_n=1.563, substrate_k=0.0, exit_n=1.0, normalize=True,
                 boundary_layer=None, name=None):
        self.n, self.k, self.t = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (n, k, t))
        if not len(self.n) == len(self.k) == len(self.t):
            raise ValueError("n, k and t must have the same length")
        self.substrate = substrate_n + 1j*substrate_k
        self.exit_n = exit_n
        self.normalize = normalize
        self.boundary_layer = boundary_layer
        self.name = name or "film mask"

    def transmissions(self, sin_theta, wavelength):
        """Transmissions of absorber and clear area for the angle of
        incidence theta in the blank.

        Returns
        -------
        t_abs, t_clear : arrays of shape (2,) + np.shape(sin_theta)
            [t_TE, t_TM] each (ratios of tangential electric fields).
        """

        sin_theta = np.asarray(sin_theta, dtype=float)
        n_abs = np.concatenate([[self.substrate], self.n + 1j*self.k, [self.exit_n]])
        n_clear = np.array([self.substrate, self.exit_n, self.exit_n], dtype=complex)
        t_abs = np.array([_r_t(sin_theta, n_abs, self.t, pol, wavelength)[1] for pol in ("TE", "TM")])
        t_clear = np.array([_r_t(sin_theta, n_clear, [self.t.sum()], pol, wavelength)[1]
                            for pol in ("TE", "TM")])

        return t_abs, t_clear

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices (J_clear, J_absorber) for the source point."""

        sin_theta = NA*np.hypot(s_x, s_y)/(magnification*np.real(self.substrate))
        t_abs, t_clear = self.transmissions(sin_theta, wavelength)
        if self.normalize:
            t_abs, t_clear = t_abs/t_clear, np.ones(2)
        e_TE, e_TM = TE_TM_basis(s_x, s_y)
        B = np.array([e_TE, e_TM]).T  # columns: TE and TM Jones vectors

        def to_xy(t):
            return B @ np.diag(t) @ B.T

        return to_xy(t_clear), to_xy(t_abs)
