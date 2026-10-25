#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""ABCD (ray transfer) matrices for paraxial ray optics and Gaussian beams.

Rays
----
A paraxial ray in a meridional plane is described by its height y and its
*reduced* angle u = n*theta, where theta is the angle to the optical axis and
n the refractive index of the medium the ray is in. An optical system maps
the ray vector (y, u) in its input plane onto the output plane by a real
2x2 matrix ::

    [y_out]   [A  B] [y_in]
    [u_out] = [C  D] [u_in].

With reduced angles, every lossless system has det M = AD - BC = 1, and a
flat interface is the identity. In air, u is simply the angle. Lengths
inside a medium of index n enter as reduced distances d/n.

Matrices are written for the light travelling from left to right (+z).
Matrices of consecutive elements multiply from the right to the left: a
system made of M_1 followed by M_2 has M = M_2 @ M_1 (see `compose()`).
Mirrors are unfolded, i.e. light continues in +z behind them.

Sign conventions
----------------

- Radii of curvature R > 0 if the centre of curvature lies to the right of
  the surface (in +z). A biconvex lens thus has R1 > 0 and R2 < 0.
- A converging thin lens has f > 0.
- Mirrors: R > 0 for a concave (focusing) mirror, which acts like a thin
  lens with f = R/2.
- Ray angles are positive for rays going upwards (+y) while travelling in
  +z.

Gaussian beams
--------------
A fundamental Gaussian beam is described by its complex beam parameter q,
with ::

    1/q = 1/R - i*lambda_0/(pi*n*w**2),   q = z + i*z_R at distance z
                                           behind the waist,

where w is the 1/e**2 intensity radius, R the radius of curvature of the
wavefront, lambda_0 the vacuum wavelength and z_R = pi*n*w_0**2/lambda_0 the
Rayleigh length. This is the q of most textbooks (Siegman, Kogelnik & Li).
The ABCD law holds for the reduced parameter q/n::

    q_out/n_out = (A*q_in/n_in + B)/(C*q_in/n_in + D).

All functions of this module take and return the true q (not q/n) and the
refractive index separately.

In the exp(i(kz - omega*t)) convention of pyoptics, the transverse field of
the beam is proportional to exp(+i*k*r**2/(2*R))*exp(-r**2/w**2) (see
`gaussian_field()`), which is exp(i*k*r**2/(2*conj(q))). The Gouy phase psi
enters as exp(-i*psi), as in `pyoptics.beams`. Wavelengths are always vacuum
wavelengths.

Contents
--------

- element matrices: `free_space()`, `thin_lens()`, `refraction()`,
  `mirror()`, `thick_lens()`, `grin()`, `compose()`
- `OpticalSystem`: builds a system element by element and traces rays and
  Gaussian beams through it (also inside the elements), computes cardinal
  points and images and draws a schematic layout
- Gaussian beams: `q_from_waist()`, `q_from_w_R()`, `beam_radius()`,
  `curvature_radius()`, `waist()`, `transform_q()`, `gouy_phase()`,
  `gaussian_field()`, `coupling_efficiency()`
- resonators: `stability_parameter()`, `is_stable()`, `eigenmode()`,
  `round_trip_gouy_phase()`, `g_parameters()`
- imaging: `cardinal_points()`, `image()`
- wave optics: `collins()`, the Collins integral (Fresnel diffraction
  through any ABCD system)
"""

from collections import namedtuple
from math import pi

import numpy as np

from .fft import FT, inv_FT
from .utils import ensure_meshgrid, freq_grid, get_dx, wavenumber


# element matrices:
def free_space(d, n=1.0):
    """Propagation over a distance d in a homogeneous medium of index n.

    Returns [[1, d/n], [0, 1]] (reduced distance d/n).
    """

    return np.array([[1.0, d/n], [0.0, 1.0]])


def thin_lens(f):
    """Thin lens of focal length f (f > 0 converging, in air).

    Returns [[1, 0], [-1/f, 1]]. f = np.inf is the identity.
    """

    return np.array([[1.0, 0.0], [-1.0/f, 1.0]])


def refraction(R, n1, n2):
    """Refraction at a spherical interface from index n1 to index n2.

    Parameters
    ----------
    R : number
        Radius of curvature, R > 0 if the centre of curvature lies behind the
        interface (in +z). np.inf for a flat interface.
    n1, n2 : numbers
        Refractive indices before and behind the interface.

    Returns
    -------
    M : array
        [[1, 0], [-(n2 - n1)/R, 1]]. A flat interface is the identity in
        reduced angles (Snell's law n1*theta1 = n2*theta2).
    """

    return np.array([[1.0, 0.0], [-(n2 - n1)/R, 1.0]])


def mirror(R, theta=0.0, plane="tangential", n=1.0):
    """Spherical mirror (unfolded), R > 0 for a concave (focusing) mirror.

    Parameters
    ----------
    R : number
        Radius of curvature. np.inf for a plane mirror.
    theta : number, optional
        Angle of incidence in radians. Under oblique incidence, the effective
        radius is R*cos(theta) in the tangential plane (the plane of
        incidence) and R/cos(theta) in the sagittal plane.
    plane : {"tangential", "sagittal"}, optional
    n : number, optional
        Refractive index of the medium in front of the mirror.

    Returns
    -------
    M : array
        [[1, 0], [-2*n/R_eff, 1]], a thin lens with focal length R_eff/2 in
        air.
    """

    if plane in ("tangential", "t"):
        R_eff = R*np.cos(theta)
    elif plane in ("sagittal", "s"):
        R_eff = R/np.cos(theta)
    else:
        raise ValueError("plane must be 'tangential' or 'sagittal'")

    return np.array([[1.0, 0.0], [-2.0*n/R_eff, 1.0]])


def grin(d, n0, g):
    """Section of length d of a graded index medium n(r) = n0*(1 - g**2*r**2/2).

    Rays oscillate with the period 2*pi/g (pitch). For g -> 0 this becomes
    `free_space(d, n0)`.

    Returns
    -------
    M : array
        [[cos(g*d), sin(g*d)/(n0*g)], [-n0*g*sin(g*d), cos(g*d)]].
    """

    if g == 0:
        return free_space(d, n0)

    c, s = np.cos(g*d), np.sin(g*d)

    return np.array([[c, s/(n0*g)], [-n0*g*s, c]])


def compose(*matrices):
    """System matrix of elements given in the order the light passes them.

    compose(M1, M2, M3) = M3 @ M2 @ M1.
    """

    M = np.eye(2)
    for M_i in matrices:
        M = np.asarray(M_i) @ M

    return M


def thick_lens(R1, R2, d, n_lens, n=1.0):
    """Thick lens with radii R1, R2, centre thickness d and index n_lens in
    a medium of index n (on both sides).

    For a biconvex lens R1 > 0 and R2 < 0. The matrix maps the vertex of the
    first surface onto the vertex of the second surface.
    """

    return compose(refraction(R1, n, n_lens), free_space(d, n_lens), refraction(R2, n_lens, n))


# Gaussian beams:
def q_from_waist(w0, z, wavelength, n=1.0):
    """Complex beam parameter q = z + i*z_R at distance z behind a waist of
    radius w0 (z < 0: in front of the waist), z_R = pi*n*w0**2/wavelength.
    """

    return z + 1j*pi*n*w0**2/wavelength


def q_from_w_R(w, R, wavelength, n=1.0):
    """Complex beam parameter from beam radius w and wavefront radius R
    (R = np.inf for a flat wavefront, R > 0 for a diverging beam).
    """

    return 1.0/(1.0/R - 1j*wavelength/(pi*n*np.asarray(w)**2))


def beam_radius(q, wavelength, n=1.0):
    """1/e**2 intensity radius w = sqrt(-wavelength/(pi*n*Im(1/q)))."""

    return np.sqrt(-wavelength/(pi*n*np.imag(1.0/np.asarray(q))))


def curvature_radius(q):
    """Wavefront radius of curvature R = 1/Re(1/q), np.inf at a waist.

    R > 0 for a diverging beam (behind the waist).
    """

    with np.errstate(divide="ignore"):
        return 1.0/np.real(1.0/np.asarray(q))


def rayleigh_length(q):
    """Rayleigh length z_R = Im(q)."""

    return np.imag(q)


def waist(q, wavelength, n=1.0):
    """Waist radius w0 and the waist position relative to the plane of q.

    Returns
    -------
    w0 : number
    z_waist : number
        Position of the waist relative to the current plane, -Re(q): positive
        if the beam is converging towards a waist that lies ahead (in +z).
    """

    q = np.asarray(q)

    return np.sqrt(np.imag(q)*wavelength/(pi*n)), -np.real(q)


def transform_q(M, q, n_in=1.0, n_out=1.0):
    """ABCD law: q_out/n_out = (A*q/n_in + B)/(C*q/n_in + D)."""

    (A, B), (C, D) = M
    q_r = np.asarray(q)/n_in

    return n_out*(A*q_r + B)/(C*q_r + D)


def gouy_phase(M, q, n_in=1.0):
    """Gouy phase psi accumulated by a fundamental Gaussian beam in a system.

    psi = -arg(A + B*n_in/q), wrapped to (-pi, pi]. The field is multiplied
    by ``exp(-i*psi)/|A + B*n_in/q|`` (rotationally symmetric beam in a
    rotationally symmetric system). In one transverse dimension (or for each
    axis of an astigmatic system) the phase is psi/2; a Hermite-Gaussian
    HG_lm acquires (l + 1/2)*psi_x + (m + 1/2)*psi_y.

    For free space behind a waist, psi = arctan(z/z_R), consistent with
    `pyoptics.beams`. To follow psi beyond +-pi, sample the system densely
    and unwrap, as `OpticalSystem.trace_beam()` does.
    """

    (A, B), _ = M

    return -np.angle(A + B*n_in/np.asarray(q))


def gaussian_field(x, y, q, wavelength, n=1.0, q_y=None, x0=0.0, y0=0.0):
    """Sample a fundamental Gaussian beam with beam parameter q.

    Parameters
    ----------
    x, y : arrays
    q : complex
        Beam parameter (in x, and in y unless q_y is given).
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.
    q_y : complex, optional
        Beam parameter in y for simple astigmatic beams.
    x0, y0 : numbers, optional
        Centre of the beam.

    Returns
    -------
    E : array
        sqrt(2/(pi*w_x*w_y))*exp(i*k*(x**2/(2*R_x) + y**2/(2*R_y)))
        ``*exp(-x**2/w_x**2 - y**2/w_y**2)``, normalized to unit power. The
        Gouy phase is not included (see `gouy_phase()`).
    """

    if q_y is None:
        q_y = q
    X, Y = ensure_meshgrid(x, y)
    k = wavenumber(wavelength, n)
    w_x, w_y = beam_radius(q, wavelength, n), beam_radius(q_y, wavelength, n)
    phase = 1j*k/2*((X - x0)**2/np.conj(q) + (Y - y0)**2/np.conj(q_y))

    return np.sqrt(2/(pi*w_x*w_y))*np.exp(phase)


def coupling_efficiency(q1, q2, wavelength, n=1.0):
    """Power coupling efficiency of two coaxial fundamental Gaussian beams
    (rotationally symmetric) given in the same plane.

    eta = ``|<E1|E2>|**2`` = 4/((w1/w2 + w2/w1)**2
                            + (pi*n*w1*w2/wavelength)**2*(1/R1 - 1/R2)**2).

    This is the mode matching efficiency into a fibre or a resonator mode.
    In one transverse dimension the efficiency is sqrt(eta).
    """

    w1, w2 = beam_radius(q1, wavelength, n), beam_radius(q2, wavelength, n)
    dc = np.real(1.0/np.asarray(q1)) - np.real(1.0/np.asarray(q2))

    return 4.0/((w1/w2 + w2/w1)**2 + (pi*n*w1*w2/wavelength)**2*dc**2)


# resonators:
def stability_parameter(M):
    """m = (A + D)/2 of a round trip matrix; a resonator is stable for ``|m|`` < 1."""

    return (M[0][0] + M[1][1])/2


def is_stable(M):
    """True if the round trip matrix M describes a stable resonator."""

    return abs(stability_parameter(M)) < 1


def eigenmode(M, n=1.0):
    """Self-reproducing beam parameter of a round trip matrix M.

    Solves q = (A*q + B)/(C*q + D) for a confined beam (Im q > 0)::

        n/q = (D - A)/(2*B) - i*sqrt(1 - m**2)/|B|,   m = (A + D)/2.

    Combine with `beam_radius()` (with the wavelength) to get the mode size.
    Raises ValueError for an unstable resonator (``|m|`` >= 1).

    Parameters
    ----------
    M : array
        Round trip matrix starting and ending in the reference plane.
    n : number, optional
        Refractive index in the reference plane.
    """

    (A, B), (C, D) = M
    m = (A + D)/2
    if abs(m) >= 1:
        raise ValueError(f"Unstable resonator, (A + D)/2 = {m}: no Gaussian eigenmode.")

    inv_q = (D - A)/(2*B) - 1j*np.sqrt(1 - m**2)/abs(B)

    return n/inv_q


def round_trip_gouy_phase(M):
    """Gouy phase of the eigenmode per round trip (rotationally symmetric
    resonator; per transverse axis it is half of that): arccos((A + D)/2)
    for B > 0 and 2*pi - arccos((A + D)/2) for B < 0, i.e. in [0, 2*pi). It sets the spacing of the transverse mode frequencies,
    Delta nu = (Delta psi/(2*pi))*FSR per unit of mode order.
    """

    (A, B), _ = M
    psi = np.arccos(stability_parameter(M))

    return psi if B >= 0 else 2*pi - psi


def g_parameters(L, R1, R2):
    """Resonator g parameters g_i = 1 - L/R_i of a two mirror resonator of
    length L. Stable for 0 < g1*g2 < 1.
    """

    return 1.0 - L/R1, 1.0 - L/R2


# imaging:
CardinalPoints = namedtuple("CardinalPoints", ["f", "f_prime", "F", "F_prime", "H", "H_prime",
                                               "N", "N_prime", "ffl", "bfl"])
CardinalPoints.__doc__ = """Cardinal points of an optical system.

f, f_prime : front and rear effective focal length (f = n_in/P,
    f_prime = n_out/P with the power P = -C)
F, F_prime, H, H_prime, N, N_prime : z positions of the focal, principal and
    nodal points, measured from the input plane (z = 0); the output plane is
    at z = length
ffl, bfl : front and back focal lengths, i.e. distances of F in front of the
    input plane and of F_prime behind the output plane
"""


def cardinal_points(M, length=0.0, n_in=1.0, n_out=1.0):
    """Cardinal points of a system with matrix M from the input plane (z = 0)
    to the output plane (z = length). Raises ValueError for an afocal system
    (C = 0).
    """

    (A, B), (C, D) = M
    if C == 0:
        raise ValueError("Afocal system (C = 0) has no focal points.")

    f, f_prime = -n_in/C, -n_out/C
    F, H = n_in*D/C, n_in*(D - 1)/C
    F_prime, H_prime = length - n_out*A/C, length + n_out*(1 - A)/C

    return CardinalPoints(f, f_prime, F, F_prime, H, H_prime, F + f_prime, F_prime - f,
                          -F, F_prime - length)


def image(M, s, n_in=1.0, n_out=1.0):
    """Image of an object at distance s in front of the input plane.

    Returns
    -------
    s_prime : number
        Distance of the image behind the output plane (negative: virtual
        image in front of it). np.inf if the image lies at infinity.
    m : number
        Lateral magnification.
    """

    (A, B), (C, D) = M
    s_r = s/n_in
    denominator = C*s_r + D
    with np.errstate(divide="ignore"):
        s_prime_r = -(A*s_r + B)/denominator
    m = A + C*s_prime_r if np.isfinite(s_prime_r) else np.inf

    return n_out*s_prime_r, m


# systems:
class _Element(namedtuple("_Element", ["kind", "z", "length", "n", "matrix_at", "label", "params"])):
    """Element of an OpticalSystem; matrix_at(s) maps the entrance of the
    element onto s in [0, length] behind it."""

    @property
    def matrix(self):
        return self.matrix_at(self.length)


BeamTrace = namedtuple("BeamTrace", ["z", "q", "w", "R", "gouy", "n"])
BeamTrace.__doc__ = """Gaussian beam traced through an OpticalSystem: beam
parameter q, radius w, wavefront radius R, accumulated Gouy phase (unwrapped)
and refractive index n at the positions z."""


class OpticalSystem:
    """Paraxial optical system built element by element along the z axis.

    The builder methods return the system itself and can be chained:

    >>> system = OpticalSystem().free_space(100).thin_lens(50).free_space(100)
    >>> system.matrix   # 2f-2f imaging: [[-1, 0], [-0.02, -1]]

    Rays and beams can be evaluated at any z, also inside thick elements.
    Positions z < 0 lie in the input medium in front of the system and
    z > length in the output medium behind it. Thin elements act at their z
    position; quantities evaluated exactly there refer to the plane just
    behind the element.

    Parameters
    ----------
    n : number, optional
        Refractive index of the input medium.
    """

    def __init__(self, n=1.0):
        self.n_in = n
        self.elements = []

    # bookkeeping:
    @property
    def length(self):
        """Axial length of the system."""

        if not self.elements:
            return 0.0
        last = self.elements[-1]

        return last.z + last.length

    @property
    def n_out(self):
        """Refractive index behind the last element."""

        return self.elements[-1].n if self.elements else self.n_in

    def _add(self, kind, length, n, matrix_at, label, **params):
        self.elements.append(_Element(kind, self.length, length, n, matrix_at, label, params))

        return self

    # builders:
    def free_space(self, d, n=None, label=None):
        """Homogeneous medium of length d and index n (default: current
        medium). A change of n implies a flat interface."""

        n = self.n_out if n is None else n

        return self._add("free", d, n, lambda s: free_space(s, n), label)

    def thin_lens(self, f, label=None):
        """Thin lens of focal length f."""

        M = thin_lens(f)

        return self._add("lens", 0.0, self.n_out, lambda s: M, label, f=f)

    def refraction(self, R, n2, label=None):
        """Spherical (R) or flat (R = np.inf) interface into index n2."""

        M = refraction(R, self.n_out, n2)

        return self._add("interface", 0.0, n2, lambda s: M, label, R=R)

    def mirror(self, R, theta=0.0, plane="tangential", label=None):
        """Spherical mirror (unfolded), R > 0 concave."""

        M = mirror(R, theta, plane, self.n_out)

        return self._add("mirror", 0.0, self.n_out, lambda s: M, label, R=R, theta=theta)

    def thick_lens(self, R1, R2, d, n_lens, label=None):
        """Thick lens (two refracting surfaces and the glass in between),
        surrounded by the current medium."""

        n = self.n_out
        self.refraction(R1, n_lens, label=label)
        self.free_space(d, n_lens, label="glass")
        self.refraction(R2, n, label=label)

        return self

    def grin(self, d, n0, g, label=None):
        """Graded index section n(r) = n0*(1 - g**2*r**2/2) of length d."""

        return self._add("grin", d, n0, lambda s: grin(s, n0, g), label, g=g)

    def element(self, M, length=0.0, n=None, label=None):
        """Any element with the (full) matrix M. Inside an element of
        nonzero length, rays and beams are only available at its ends."""

        M = np.asarray(M, dtype=float)
        n = self.n_out if n is None else n

        return self._add("matrix", length, n, lambda s: M if s >= length else np.eye(2), label)

    def append(self, other):
        """Append all elements of another OpticalSystem."""

        for e in other.elements:
            self._add(e.kind, e.length, e.n, e.matrix_at, e.label, **e.params)

        return self

    # evaluation:
    @property
    def matrix(self):
        """System matrix from z = 0 to z = length."""

        return compose(*(e.matrix for e in self.elements))

    def n_at(self, z):
        """Refractive index at the position z."""

        n = self.n_in
        for e in self.elements:
            if z < e.z or (e.length > 0 and z == e.z):
                break
            n = e.n

        return n

    def matrix_at(self, z):
        """Matrix from z = 0 to the position z (just behind thin elements at z)."""

        if z < 0:
            return free_space(z, self.n_in)

        M = np.eye(2)
        for e in self.elements:
            if z < e.z:
                break
            M = e.matrix_at(min(z - e.z, e.length)) @ M
        if z > self.length:
            M = free_space(z - self.length, self.n_out) @ M

        return M

    def trace(self, rays, z):
        """Trace rays through the system.

        Parameters
        ----------
        rays : array
            Ray(s) (y, theta) in the plane z = 0 (or (2, N) for N rays),
            theta being the true angle in the input medium.
        z : array
            Positions at which to evaluate the rays.

        Returns
        -------
        y, theta : arrays
            Height and true angle, shape (len(z), N) (or (len(z),) for a
            single ray).
        """

        rays = np.asarray(rays, dtype=float)
        single = rays.ndim == 1
        rays = rays.reshape(2, -1)
        reduced = np.vstack([rays[0], self.n_in*rays[1]])
        z = np.atleast_1d(z)
        y = np.empty((len(z), rays.shape[1]))
        theta = np.empty_like(y)
        for i, z_i in enumerate(z):
            y[i], u = self.matrix_at(z_i) @ reduced
            theta[i] = u/self.n_at(z_i)

        if single:
            return y[:, 0], theta[:, 0]

        return y, theta

    def trace_beam(self, q, wavelength, z):
        """Trace a Gaussian beam with parameter q in the plane z = 0.

        Returns
        -------
        BeamTrace
            q, w, R, n and the Gouy phase at the positions z (relative to
            z = 0, unwrapped along z, so z should be sampled densely enough
            for the phase to change by less than pi between samples).
        """

        z = np.atleast_1d(z)
        n = np.array([self.n_at(z_i) for z_i in z])
        Ms = [self.matrix_at(z_i) for z_i in z]
        q_z = np.array([transform_q(M, q, self.n_in, n_i) for M, n_i in zip(Ms, n)])
        psi = np.unwrap([gouy_phase(M, q, self.n_in) for M in Ms])
        # the reference for the unwrapping is z = 0:
        psi -= 2*pi*np.round(psi[np.argmin(np.abs(z))]/(2*pi))

        return BeamTrace(z, q_z, beam_radius(q_z, wavelength, n), curvature_radius(q_z), psi, n)

    def cardinal_points(self):
        """Cardinal points, see `cardinal_points()`."""

        return cardinal_points(self.matrix, self.length, self.n_in, self.n_out)

    def image(self, s):
        """Image of an object at distance s in front of the system, see
        `image()`."""

        return image(self.matrix, s, self.n_in, self.n_out)

    def draw(self, ax=None, height=1.0, color="0.3"):
        """Draw a schematic layout of the system into a matplotlib axis.

        Lenses are drawn as double arrows (pointing outwards for converging,
        inwards for diverging lenses), mirrors as thick lines, interfaces as
        arcs and media with n != 1 as shaded areas. height is the half height
        of the symbols in data units.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        def arrow(z, y0, y1):
            ax.annotate("", xy=(z, y1), xytext=(z, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=0, shrinkB=0))

        for e in self.elements:
            if e.length > 0 and (e.n != 1 or e.kind == "grin"):
                shade = "C0" if e.kind == "grin" else "C9"
                ax.axvspan(e.z, e.z + e.length, color=shade, alpha=0.12 + 0.05*(e.n - 1), lw=0)
            if e.kind == "lens":
                if e.params["f"] > 0:
                    arrow(e.z, 0, height); arrow(e.z, 0, -height)
                else:
                    arrow(e.z, height, 0); arrow(e.z, -height, 0)
                    ax.plot([e.z, e.z], [-height, height], color=color, lw=1.5)
            elif e.kind in ("mirror", "interface"):
                # curvature exaggerated for visibility:
                R = e.params["R"]
                yy = np.linspace(-height, height, 50)
                sag = np.sign(R)*0.015*max(self.length, 1e-12)*(yy/height)**2 if np.isfinite(R) else 0*yy
                if e.kind == "mirror":
                    ax.plot(e.z - sag, yy, color=color, lw=3)
                else:
                    ax.plot(e.z + sag, yy, color=color, lw=1.2)
            elif e.kind == "matrix":
                ax.add_patch(plt.Rectangle((e.z, -height), max(e.length, 1e-12), 2*height,
                                           fill=False, ec=color, ls="--"))
            if e.label and e.label != "glass":
                ax.text(e.z + e.length/2, 1.08*height, e.label, ha="center", va="bottom",
                        fontsize=9, color=color)
        ax.axhline(0, color="0.6", lw=0.6, ls="-.")

        return ax


# wave optics:
def _magnify(field, x, y, A_x, A_y, C_x, C_y, wavelength):
    """Imaging case B = 0 of the Collins integral (interpolation)."""

    from scipy.interpolate import RegularGridInterpolator

    x_l, y_l = np.asarray(x), np.asarray(y)
    if x_l.ndim == 2:
        x_l, y_l = x_l[0], y_l[:, 0]
    X, Y = np.meshgrid(x_l, y_l)
    points = np.stack([Y.ravel()/A_y, X.ravel()/A_x], axis=-1)
    interpolate = lambda f: RegularGridInterpolator((y_l, x_l), f, bounds_error=False, fill_value=0.0)(points)
    E = (interpolate(field.real) + 1j*interpolate(field.imag)).reshape(field.shape)
    k = wavenumber(wavelength)

    return E/(np.sqrt(A_x + 0j)*np.sqrt(A_y + 0j))*np.exp(1j*k/2*(C_x/A_x*X**2 + C_y/A_y*Y**2))


def collins(field, x, y, M, wavelength, M_y=None):
    """Propagate a field through a paraxial ABCD system (Collins integral).

    The generalized Fresnel integral of S. A. Collins (JOSA 60, 1168 (1970)), ::

        E_out(x) = (i*lambda*B)**(-1/2) * integral E_in(x')
                   * exp(i*pi/(lambda*B)*(A*x'**2 - 2*x'*x + D*x**2)) dx'

    per transverse axis (with the vacuum wavelength and reduced B), is
    evaluated by splitting M into a thin lens, free space B and another thin
    lens; the free space part uses the Fresnel transfer function. The output
    is sampled on the input grid, so the grid has to hold the output field
    and resolve both lens chirps (wrap-around is not prevented). For B = 0
    (imaging planes) the field is scaled by interpolation. The constant phase
    exp(i*k*L) of the axial optical path is omitted.

    Parameters
    ----------
    field : array
    x, y : arrays
        Centered coordinates (linear or 2d).
    M : array
        ABCD matrix (in x, and in y unless M_y is given) with det M = 1.
    wavelength : number
        Vacuum wavelength.
    M_y : array, optional
        Matrix in y for systems with cylindrical elements or tilted mirrors.

    Returns
    -------
    E : array
        Field in the output plane on the grid x, y.
    """

    M_y = M if M_y is None else M_y
    (A_x, B_x), (C_x, D_x) = M
    (A_y, B_y), (C_y, D_y) = M_y
    dx, dy = get_dx(x), get_dx(y)

    # B small compared with the scale on which the grid resolves diffraction:
    is_imaging = [abs(B)*wavelength < 1E-9*d**2 for B, d in ((B_x, dx), (B_y, dy))]
    if all(is_imaging):
        return _magnify(field, x, y, A_x, A_y, C_x, C_y, wavelength)
    if any(is_imaging):
        raise NotImplementedError("collins() needs B = 0 in both or in none of the axes.")

    X, Y = ensure_meshgrid(x, y)
    k = wavenumber(wavelength)
    F_x, F_y = freq_grid(x, y, wavenumbers=False, normal_order=True)

    # M = lens((D - 1)/B) @ free(B) @ lens((A - 1)/B); a lens [[1, 0], [c, 1]]
    # has the transmission exp(i*k*c*r**2/2):
    E = field*np.exp(1j*k/2*((A_x - 1)/B_x*X**2 + (A_y - 1)/B_y*Y**2))
    E = inv_FT(FT(E)*np.exp(-1j*pi*wavelength*(B_x*F_x**2 + B_y*F_y**2)))
    E = E*np.exp(1j*k/2*((D_x - 1)/B_x*X**2 + (D_y - 1)/B_y*Y**2))

    return E
