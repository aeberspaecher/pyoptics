#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Stellar coronagraphs: telescope pupils, apodizers, focal plane masks and
Lyot stops for imaging faint planets next to bright stars (notebook 96).

A coronagraph takes the light of an on-axis star out of the image while it
lets an off-axis planet through. The classical layout has four planes:

1. the entrance pupil (the telescope aperture, optionally apodized),
2. a focal plane with a mask, where the star is focused,
3. a Lyot plane, an image of the pupil, with a Lyot stop,
4. the final focal plane with the camera.

All planes are scalar and Fraunhofer: the pupil coordinates x, y are in
units of the pupil diameter D, the focal plane coordinates u, v in units of
lambda/D (lambda*F in the focal plane of an f/F beam). The field in a focal
plane is the matrix Fourier transform (MFT) of the pupil field,

    E(u, v) = sum_xy E(x, y) exp(-2 pi i (u x + v y)) dx dy,

which samples it on any grid of u, v (Soummer et al. 2007). A plane wave
exp(2 pi i (alpha x + beta y)) from a source at (alpha, beta) lambda/D is
imaged at u = alpha, v = beta.

Masks:

- Lyot coronagraph: an occulting disk of radius a lambda/D (or a phase
  disk, transmission t). The field in the Lyot plane follows from Babinet's
  principle, E_L = E - (1 - t) F^-1[M F[E]], with the finely sampled
  transform over the disk M alone (Soummer et al. 2007), which is exact.
- Apodized pupil Lyot coronagraph (APLC): the pupil is apodized with the
  circular prolate spheroidal function of the mask size. It is the
  eigenfunction of "transform, keep the disk of radius a, transform back,
  keep the pupil" with the largest eigenvalue Lambda, so behind the mask the
  field in the pupil is exactly (1 - Lambda) times the apodizer
  (Aime, Soummer and Ferrari 2002; Soummer et al. 2003).
- Vortex coronagraph: the focal plane phase exp(i l theta) of charge l
  (Mawet et al. 2005; Foo et al. 2005). For even l and a clear circular
  pupil the field in the Lyot plane is zero inside the pupil, and an
  aberration Z_n^m passes only if n + |m| >= |l|. The focal field is
  singular at the centre and extends to large u, so its transform is
  evaluated with nested grids, fine near the centre and coarse far out,
  joined by smooth windows that add up to one (after Krist et al. 2011).

Functions:

- pupil_grid(): cell centred pupil coordinates.
- telescope_pupil(): circular aperture with a central obscuration and
  spider vanes, antialiased by supersampling.
- mft(): matrix Fourier transform between a pupil and a focal plane.
- tilt(): plane wave of an off-axis source.
- airy_envelope(): the ring averaged Airy pattern far from the core,
  4/(pi^4 u^3).
- sonine_apodizer(), sonine_psf(): the apodization (1 - r^2/R^2)^nu and
  its analytic PSF.
- prolate_apodizer(): the circular prolate spheroidal apodizer of a mask
  of radius a, with its eigenvalue.
- occulter_lyot_field(), occulter_lyot_analytic(): Lyot plane field
  behind an occulting disk, numerically and for a clear circular pupil
  (a Hankel integral; J_0(pi a) at the centre).
- vortex_lyot_field(), vortex_lyot_analytic(): Lyot plane field behind a
  vortex mask, numerically and for a clear circular pupil.
- vortex_blocks(): the Zernike selection rule of a vortex.
- Coronagraph: a complete coronagraph, its Lyot plane field and image.
- radial_profile(): azimuthal mean (or maximum) of an image in rings.
- aberration_screen(): a random wavefront (or amplitude) error with a
  given power spectral density.
- dm_influence(): Gaussian influence functions of a deformable mirror.
- efc_jacobian(), efc_step(): electric field conjugation, the linear
  dark hole controller of Give'on et al. (2007).

References
----------
B. Lyot, "The study of the solar corona and prominences without eclipses",
Mon. Not. R. Astron. Soc. 99, 580-594 (1939).

C. Aime, R. Soummer and A. Ferrari, "Total coronagraphic extinction of
rectangular apertures using linear prolate apodizations", Astron.
Astrophys. 389, 334-344 (2002).

R. Soummer, C. Aime and P. E. Falloon, "Stellar coronagraphy with prolate
apodized circular apertures", Astron. Astrophys. 397, 1161-1172 (2003).

R. Soummer, L. Pueyo, A. Sivaramakrishnan and R. J. Vanderbei, "Fast
computation of Lyot-style coronagraph propagation", Opt. Express 15,
15935-15951 (2007).

D. Mawet, P. Riaud, O. Absil and J. Surdej, "Annular groove phase mask
coronagraph", Astrophys. J. 633, 1191-1200 (2005).

G. Foo, D. M. Palacios and G. A. Swartzlander, "Optical vortex
coronagraph", Opt. Lett. 30, 3308-3310 (2005).

J. Krist, J. Trauger, E. Serabyn and D. Mawet, "Numerical modeling of the
vortex coronagraph" (internal note, JPL, 2011); see also J. E. Krist et
al., Proc. SPIE 8151, 81510E (2011).

G. Ruane, D. Mawet, B. Mennesson, J. Jewell and S. Shaklan, "Vortex
coronagraphs for the Habitable Exoplanet Imaging Mission concept:
theoretical performance and telescope requirements", J. Astron. Telesc.
Instrum. Syst. 4, 015004 (2018).

A. Give'on, R. Belikov, S. Shaklan and J. Kasdin, "Closed loop, DM
diversity-based, wavefront correction algorithm for high contrast
imaging systems", Opt. Express 15, 12338-12343 (2007).

W. B. Traub and B. R. Oppenheimer, "Direct imaging of exoplanets", in
"Exoplanets", ed. S. Seager, University of Arizona Press (2010), 111-156.
"""

from collections import namedtuple

import numpy as np
from scipy.special import gamma, jv, j0, roots_legendre


__all__ = ["pupil_grid", "telescope_pupil", "mft", "tilt", "airy_envelope", "sonine_apodizer",
           "sonine_psf", "Prolate", "prolate_apodizer", "occulter_lyot_field", "occulter_lyot_analytic",
           "VORTEX_SCALES",
           "vortex_lyot_field", "vortex_lyot_analytic", "vortex_blocks", "Coronagraph",
           "radial_profile", "aberration_screen", "dm_influence", "efc_jacobian", "efc_step"]


def pupil_grid(n, extent=1.0):
    """Cell centred coordinates of n samples over extent (in units of D).

    Parameters
    ----------
    n : int
        Number of samples.
    extent : float, optional
        Width of the grid in units of the pupil diameter (default 1).

    Returns
    -------
    x : array
        Coordinates (k - n/2 + 1/2)*extent/n, symmetric about 0.
    """

    return (np.arange(n) - n/2 + 0.5)*extent/n


def telescope_pupil(x, y, D=1.0, obscuration=0.0, n_spiders=0, spider_width=0.0, spider_angle=0.0,
                    oversample=4):
    """Telescope aperture: a disk with a central obscuration and spider vanes.

    Parameters
    ----------
    x, y : arrays
        Linear, equidistant coordinates.
    D : float, optional
        Outer diameter.
    obscuration : float, optional
        Diameter of the central obscuration as a fraction of D.
    n_spiders : int, optional
        Number of straight vanes from the centre to the rim, equally spaced.
    spider_width : float, optional
        Width of each vane (same units as x).
    spider_angle : float, optional
        Direction of the first vane in radians from the x axis.
    oversample : int, optional
        Sub-samples per pixel and direction for the antialiased edges.

    Returns
    -------
    P : array
        Transmission of shape (len(y), len(x)), the fraction of each pixel
        that is open.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    s = (np.arange(oversample) + 0.5)/oversample - 0.5
    P = np.zeros((len(y), len(x)))
    for a in s:
        X, Y = np.meshgrid(x + a*dx, y)
        for b in s:
            Yb = Y + b*dy
            r = np.hypot(X, Yb)
            open_ = (r <= D/2) & (r >= obscuration*D/2)
            for k in range(n_spiders):
                phi = spider_angle + 2*np.pi*k/n_spiders
                along = X*np.cos(phi) + Yb*np.sin(phi)
                across = -X*np.sin(phi) + Yb*np.cos(phi)
                open_ &= ~((along > 0) & (np.abs(across) < spider_width/2))
            P += open_
    return P/oversample**2


def mft(E, x, y, u, v, inverse=False):
    """Matrix Fourier transform between a pupil and a focal plane.

    Computes ::

        F(v, u) = sum_jl E[j, l] exp(-+2 pi i (u x_l + v y_j)) dx dy

    (sign + for inverse=True). With x, y in units of D, u, v are in units of
    lambda/D, and the other way round for the inverse transform. The last
    two axes of E are transformed, so a stack of fields is transformed at
    once.

    Parameters
    ----------
    E : array
        Field(s) of shape (..., len(y), len(x)).
    x, y : arrays
        Linear, equidistant input coordinates.
    u, v : arrays
        Output coordinates (any spacing).
    inverse : bool, optional
        Use the sign + in the exponent.

    Returns
    -------
    F : array
        Transform of shape (..., len(v), len(u)).
    """

    s = 1.0 if inverse else -1.0
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = (x[-1] - x[0])/(len(x) - 1)
    dy = (y[-1] - y[0])/(len(y) - 1)
    Ax = np.exp(s*2j*np.pi*np.outer(u, x))
    Ay = np.exp(s*2j*np.pi*np.outer(v, y))
    return (Ay @ E @ Ax.T)*(dx*dy)


def tilt(x, y, alpha, beta=0.0):
    """Plane wave of a source at (alpha, beta) lambda/D off the axis.

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates in units of D.
    alpha, beta : float
        Source position in units of lambda/D.

    Returns
    -------
    E : array
        exp(2 pi i (alpha x + beta y)) of shape (len(y), len(x)); its image
        lies at u = alpha, v = beta.
    """

    X, Y = np.meshgrid(x, y)
    return np.exp(2j*np.pi*(alpha*X + beta*Y))


def airy_envelope(u):
    """Ring averaged Airy pattern far from the core.

    With <J_1(z)^2> = 1/(pi z) the Airy pattern [2 J_1(pi u)/(pi u)]^2
    averages to 4/(pi^4 u^3) over a ring.

    Parameters
    ----------
    u : array
        Distance from the centre in lambda/D.

    Returns
    -------
    I : array
        Mean intensity relative to the peak.
    """

    u = np.asarray(u, float)
    return 4/(np.pi**4*u**3)


def sonine_apodizer(x, y, nu, D=1.0):
    """Sonine apodization (1 - r^2/R^2)^nu of a circular pupil.

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates.
    nu : float
        Exponent (0: clear pupil).
    D : float, optional
        Pupil diameter, R = D/2.

    Returns
    -------
    A : array
        Amplitude transmission of shape (len(y), len(x)), zero outside the
        pupil (for nu = 0: one everywhere, the pupil function sets the
        edge).
    """

    X, Y = np.meshgrid(x, y)
    q = 1 - (X**2 + Y**2)/(D/2)**2
    if nu == 0:
        return np.ones_like(q)
    return np.where(q > 0, np.abs(q)**nu, 0.0)


def sonine_psf(u, nu):
    """Analytic PSF of the Sonine apodized pupil of diameter D.

    The Hankel transform of (1 - r^2)^nu on the unit disk is
    2^nu Gamma(nu + 1) J_{nu+1}(k)/k^(nu+1), so with k = pi u the PSF is ::

        I(u)/I(0) = [Gamma(nu + 2) 2^(nu+1) J_{nu+1}(pi u)/(pi u)^(nu+1)]^2

    and its rings fall as u^-(2 nu + 3). The peak relative to the clear
    pupil is 1/(nu + 1)^2 and the transmitted power 1/(2 nu + 1).

    Parameters
    ----------
    u : array
        Distance from the centre in lambda/D.
    nu : float
        Exponent of the apodization.

    Returns
    -------
    I : array
        Intensity relative to its own peak.
    """

    k = np.pi*np.asarray(u, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = gamma(nu + 2)*2**(nu + 1)*jv(nu + 1, k)/k**(nu + 1)
    a = np.where(k == 0, 1.0, a)
    return a**2


Prolate = namedtuple("Prolate", ["eigenvalue", "r", "values", "mask_radius"])
Prolate.__doc__ = """Circular prolate spheroidal function of a pupil and a focal disk.

Attributes
----------
eigenvalue : float
    Largest eigenvalue Lambda: the fraction of the light of the apodized
    pupil inside the focal disk.
r : array
    Quadrature nodes in units of D (0 to 1/2).
values : array
    The function at the nodes, normalized to 1 at its maximum.
mask_radius : float
    Radius of the focal disk in lambda/D.
"""


def _hankel_kernel(r, a, n_rho):
    """K(r_i, r_j) with Gauss-Legendre weights; symmetric form."""
    rho, w_rho = roots_legendre(n_rho)
    rho = 0.5*a*(rho + 1)
    w_rho = 0.5*a*w_rho
    B = j0(2*np.pi*np.outer(r, rho))
    return 4*np.pi**2*(B*(w_rho*rho)) @ B.T


def prolate_apodizer(a, x=None, y=None, n=120):
    """Circular prolate spheroidal apodizer for an occulting disk of radius a.

    Finds the radial eigenfunction with the largest eigenvalue of ::

        Lambda A(r) = 4 pi^2 int_0^{1/2} K(r, r') A(r') r' dr',
        K(r, r') = int_0^a J_0(2 pi rho r) J_0(2 pi rho r') rho drho,

    i.e. "Fourier transform, keep the disk of radius a lambda/D, transform
    back, keep the pupil of diameter 1". Lambda is the fraction of the light
    of the apodized pupil inside the disk; behind the disk the field in the
    pupil is (1 - Lambda) times the apodizer. The integral is a Nystrom
    discretization with Gauss-Legendre nodes, and the eigenfunction is
    interpolated with the same kernel.

    Parameters
    ----------
    a : float
        Radius of the focal disk in lambda/D.
    x, y : arrays, optional
        Pupil coordinates in units of D. If given, the apodizer is
        returned on this grid as well.
    n : int, optional
        Number of quadrature nodes.

    Returns
    -------
    prolate : Prolate
        Eigenvalue and the radial function.
    A : array
        The apodizer on the grid x, y (zero outside the pupil), only if x
        and y are given.
    """

    t, w = roots_legendre(n)
    r = 0.25*(t + 1)
    w = 0.25*w
    n_rho = max(n, int(8*a) + 40)
    K = _hankel_kernel(r, a, n_rho)
    sw = np.sqrt(w*r)
    S = sw[:, None]*K*sw[None, :]
    lam, vec = np.linalg.eigh(S)
    Lam = lam[-1]
    f = vec[:, -1]/sw
    f /= f[np.argmax(np.abs(f))]
    pro = Prolate(Lam, r, f, a)
    if x is None or y is None:
        return pro
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)
    inside = R <= 0.5
    rr = R[inside]
    # interpolate with the kernel: A(r) = (1/Lambda) sum_j K(r, r_j) w_j r_j A(r_j)
    rho, w_rho = roots_legendre(n_rho)
    rho = 0.5*a*(rho + 1)
    w_rho = 0.5*a*w_rho
    coef = (j0(2*np.pi*np.outer(rho, r)) @ (w*r*f))*w_rho*rho
    vals = np.empty(rr.shape)
    chunk = 20000
    for k in range(0, len(rr), chunk):
        vals[k:k + chunk] = 4*np.pi**2*j0(2*np.pi*np.outer(rr[k:k + chunk], rho)) @ coef/Lam
    A = np.zeros(R.shape)
    A[inside] = vals
    return pro, A


def _focal_grid(R, d):
    return (np.arange(-int(round(R/d)), int(round(R/d))) + 0.5)*d


def occulter_lyot_field(E, x, y, a, transmission=0.0, samples=None):
    """Field in the Lyot plane behind an occulting disk (Babinet's principle).

    E_L = E - (1 - t) F^-1[M F[E]] with the disk M of radius a lambda/D:
    only the light that falls on the disk is transformed, on a fine grid
    over the disk, which is exact up to the sampling of the disk.

    Parameters
    ----------
    E : array
        Pupil field of shape (len(y), len(x)) (or a stack of fields).
    x, y : arrays
        Pupil coordinates in units of D.
    a : float
        Radius of the disk in lambda/D.
    transmission : complex, optional
        Amplitude transmission t of the disk: 0 (default) for an opaque
        Lyot mask, -1 for a pi phase disk (Roddier and Roddier 1997).
    samples : int, optional
        Samples across the disk diameter (default: 32 per lambda/D, at
        least 128).

    Returns
    -------
    E_L : array
        Field in the Lyot plane (on the pupil grid), before the Lyot stop.
    """

    if samples is None:
        samples = max(128, int(np.ceil(64*a)))
    u = pupil_grid(samples, 2*a)
    M = telescope_pupil(u, u, D=2*a, oversample=8)
    F = mft(E, x, y, u, u)*M
    return E - (1 - transmission)*mft(F, u, u, x, y, inverse=True)


def occulter_lyot_analytic(r, a, transmission=0.0):
    """Lyot plane field of a clear circular pupil behind an occulting disk.

    The pupil of diameter 1 has the focal field J_1(pi rho)/(2 rho); the
    disk takes out the part with rho < a, whose transform back is ::

        pi int_0^a J_0(2 pi rho r) J_1(pi rho) drho,

    so E_L(r) = P(r) - (1 - t) times this integral (Gauss-Legendre
    quadrature). At the centre of the pupil the integral is
    1 - J_0(pi a), and E_L(0) = J_0(pi a) for an opaque disk.

    Parameters
    ----------
    r : array
        Radius in the Lyot plane in units of D.
    a : float
        Radius of the disk in lambda/D.
    transmission : complex, optional
        Amplitude transmission of the disk.

    Returns
    -------
    E : array
        Field (real for real t), radially symmetric.
    """

    r = np.asarray(r, float)
    t, w = roots_legendre(max(64, int(16*a) + 32))
    rho = 0.5*a*(t + 1)
    w = 0.5*a*w
    blocked = np.pi*(j0(2*np.pi*np.multiply.outer(r, rho))*jv(1, np.pi*rho)) @ w
    P = np.where(r < 0.5, 1.0, np.where(r == 0.5, 0.5, 0.0))
    return P - (1 - transmission)*blocked


VORTEX_SCALES = ((0.25, 1/512), (2.0, 1/64), (16.0, 1/8), (128.0, 1/4))
"""Default nested focal grids (half width, step) of vortex_lyot_field(), in lambda/D."""


def _smooth_step(r, R, f=0.5):
    """1 for r < f R, 0 for r > R, raised cosine in between."""
    t = np.clip((r - f*R)/((1 - f)*R), 0, 1)
    return 0.5*(1 + np.cos(np.pi*t))


def vortex_lyot_field(E, x, y, charge, x_out=None, y_out=None, scales=VORTEX_SCALES, offset=(0.0, 0.0)):
    """Field in the Lyot plane behind a vortex phase mask exp(i l theta).

    The focal field is computed and transformed back on nested grids
    (half widths R_k, steps d_k), with smooth windows w_k = s(r/R_k) -
    s(r/R_{k-1}) (s a raised cosine step from 1 at R/2 to 0 at R) that add
    up to one inside the largest grid. Each windowed integrand is smooth,
    so the midpoint sums converge fast; the singular centre lies on the
    corner of four cells of the finest grid. Light beyond the largest grid
    is lost, which blurs the field near the pupil edge by about 1/R_max.

    Parameters
    ----------
    E : array
        Pupil field of shape (len(y), len(x)) (or a stack of fields).
    x, y : arrays
        Pupil coordinates in units of D.
    charge : int
        Topological charge l of the vortex.
    x_out, y_out : arrays, optional
        Lyot plane coordinates (default: x, y).
    scales : sequence of (R, d), optional
        Half widths and steps of the focal grids in lambda/D, from the
        finest to the coarsest. R/d must be an integer.
    offset : (float, float), optional
        Position of the vortex centre in lambda/D (a misaligned mask).

    Returns
    -------
    E_L : array
        Field in the Lyot plane, before the Lyot stop.
    """

    if x_out is None:
        x_out, y_out = x, y
    out = 0.0
    R_prev = None
    for R, d in scales:
        u = _focal_grid(R, d)
        U, V = np.meshgrid(u, u)
        r = np.hypot(U, V)
        w = _smooth_step(r, R)
        if R_prev is not None:
            w = w - _smooth_step(r, R_prev)
        phase = np.exp(1j*charge*np.arctan2(V - offset[1], U - offset[0]))
        F = mft(E, x, y, u, u)*(phase*w)
        out = out + mft(F, u, u, x_out, y_out, inverse=True)
        R_prev = R
    return out


def vortex_lyot_analytic(r, charge, D=1.0):
    """Lyot plane field of a clear circular pupil behind a vortex (radial part).

    For a uniform pupil of radius R = D/2 and even charge l the Lyot plane
    field is zero inside the pupil and, outside it (Mawet et al. 2005;
    Jenkins 2008), for l = 2 and 4 ::

        E_2(r) = -(R/r)^2,    E_4(r) = 2 (R/r)^2 - 3 (R/r)^4,

    times exp(i l theta). Charge 0 returns the pupil itself.

    Parameters
    ----------
    r : array
        Radius in the Lyot plane.
    charge : int
        0, 2 or 4 (also -2, -4: the same radial part).
    D : float, optional
        Pupil diameter.

    Returns
    -------
    E : array
        Radial part of the field (multiply by exp(i l theta)).
    """

    r = np.asarray(r, float)
    R = D/2
    q = np.where(r > R, R/np.maximum(r, 1e-300), 0.0)
    l = abs(int(charge))
    if l == 0:
        return (r <= R).astype(float)
    if l == 2:
        return -q**2
    if l == 4:
        return 2*q**2 - 3*q**4
    raise ValueError("only charges 0, 2 and 4 are implemented")


def vortex_blocks(charge, n, m):
    """Zernike selection rule of an ideal vortex (clear circular pupil).

    A pupil field R_n^m(r) exp(i m theta) has the focal field
    J_{n+1}(2 pi R rho)/rho exp(i m phi). The vortex adds exp(i l phi), and
    the transform of order |m + l| back into the pupil is a
    Weber-Schafheitlin integral that vanishes inside the pupil when
    |m + l| - (n + 1) is odd and positive. For a real Zernike term (both
    signs of m) and even l this means it is blocked if n + |m| < |l|.

    Parameters
    ----------
    charge : int
        Even charge l of the vortex.
    n, m : int
        Radial and azimuthal Zernike orders.

    Returns
    -------
    blocked : bool
        True if the term does not reach the inside of the Lyot stop.
    """

    return n + abs(m) < abs(charge)


class Coronagraph:
    """A coronagraph: pupil, apodizer, focal plane mask and Lyot stop.

    Parameters
    ----------
    x, y : arrays
        Pupil (and Lyot plane) coordinates in units of D.
    pupil : array
        Entrance pupil transmission.
    mask : str, optional
        "none" (the telescope alone), "lyot" (an occulting disk of radius
        mask_radius, transmission t) or "vortex" (charge l).
    mask_radius : float, optional
        Radius of the occulting disk in lambda/D.
    charge : int, optional
        Charge of the vortex.
    lyot_stop : array, optional
        Transmission of the Lyot stop (default: all open).
    apodizer : array, optional
        Amplitude apodization of the pupil (default 1).
    transmission : complex, optional
        Transmission of the occulting disk (0: opaque).
    scales : sequence, optional
        Nested focal grids of the vortex (see vortex_lyot_field()).

    Attributes
    ----------
    peak : float
        Peak intensity of an off-axis source far from the mask, i.e. of the
        PSF through apodizer and Lyot stop without the mask; images are
        normalized by it (contrast).
    throughput : float
        Fraction of the light of a source far from the mask that passes the
        apodizer and the Lyot stop.
    """

    def __init__(self, x, y, pupil, mask="none", mask_radius=None, charge=None, lyot_stop=None,
                 apodizer=None, transmission=0.0, scales=VORTEX_SCALES):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.pupil = np.asarray(pupil)
        self.mask = mask
        self.mask_radius = mask_radius
        self.charge = charge
        self.lyot_stop = np.ones_like(self.pupil, dtype=float) if lyot_stop is None else lyot_stop
        self.apodizer = 1.0 if apodizer is None else apodizer
        self.transmission = transmission
        self.scales = scales
        if mask not in ("none", "lyot", "vortex"):
            raise ValueError("mask must be 'none', 'lyot' or 'vortex'")
        dA = (self.x[1] - self.x[0])*(self.y[1] - self.y[0])
        through = self.pupil*self.apodizer*self.lyot_stop
        self.peak = abs(np.sum(through)*dA)**2
        self.throughput = np.sum(np.abs(through)**2)/np.sum(np.abs(self.pupil)**2)

    def pupil_field(self, E=None):
        """Field after the apodizer.

        Parameters
        ----------
        E : array, optional
            Incoming field multiplying the pupil (default 1).

        Returns
        -------
        Ep : array
            Pupil times apodizer times E.
        """

        P = self.pupil*self.apodizer
        return P if E is None else P*E

    def lyot_field(self, E=None):
        """Field in the Lyot plane before the Lyot stop.

        Parameters
        ----------
        E : array, optional
            Incoming field (wavefront, tilt, amplitude errors) multiplying
            the pupil; a stack of fields is propagated at once.

        Returns
        -------
        E_L : array
            Lyot plane field.
        """

        Ep = self.pupil_field(E)
        if self.mask == "none":
            return Ep
        if self.mask == "lyot":
            return occulter_lyot_field(Ep, self.x, self.y, self.mask_radius, self.transmission)
        return vortex_lyot_field(Ep, self.x, self.y, self.charge, scales=self.scales)

    def image_field(self, u, v, E=None):
        """Complex field in the final focal plane, in units of sqrt(peak).

        Parameters
        ----------
        u, v : arrays
            Final focal plane coordinates in lambda/D.
        E : array, optional
            Incoming field (see lyot_field()).

        Returns
        -------
        F : array
            Field normalized so that abs(F)**2 is the contrast.
        """

        return mft(self.lyot_field(E)*self.lyot_stop, self.x, self.y, u, v)/np.sqrt(self.peak)

    def image(self, u, v, E=None):
        """Image in the final focal plane relative to the off-axis peak.

        Parameters
        ----------
        u, v : arrays
            Final focal plane coordinates in lambda/D.
        E : array, optional
            Incoming field (see lyot_field()).

        Returns
        -------
        I : array
            Intensity divided by `peak` (the contrast).
        """

        return np.abs(self.image_field(u, v, E))**2


def radial_profile(I, u, v, edges, stat="mean", mask=None):
    """Azimuthal mean or maximum of an image in rings.

    Parameters
    ----------
    I : array
        Image of shape (len(v), len(u)).
    u, v : arrays
        Image coordinates.
    edges : array
        Ring edges in the units of u.
    stat : str, optional
        "mean" (default) or "max".
    mask : array of bool, optional
        Only pixels where mask is True are used (e.g. half a dark hole).

    Returns
    -------
    centres : array
        Ring centres.
    p : array
        Mean or maximum in each ring (nan for empty rings).
    """

    edges = np.asarray(edges, float)
    U, V = np.meshgrid(u, v)
    r = np.hypot(U, V)
    if mask is None:
        mask = np.ones(I.shape, bool)
    out = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        sel = mask & (r >= edges[k]) & (r < edges[k + 1])
        if np.any(sel):
            out[k] = np.mean(I[sel]) if stat == "mean" else np.max(I[sel])
    return 0.5*(edges[1:] + edges[:-1]), out


def aberration_screen(x, y, K, exponent=3.0, f0=1.0, rng=None):
    """Random, periodic wavefront error with a power law spectrum.

    The screen has the two sided power spectral density ::

        S(f) = K (f0^2 + f^2)^(-exponent/2)

    in units of x^2 times its own unit squared (radian^2 D^2 for a phase
    in radians and x in units of D). Each Fourier component of the grid
    gets a complex Gaussian amplitude, and the real part is kept, so that
    the variance is the sum of S df^2 over the grid. Far from the core, the
    first order speckle halo of a pupil of area A with this phase error is
    S(u)/A relative to the peak.

    Parameters
    ----------
    x, y : arrays
        Equidistant coordinates (the screen is periodic over the grid).
    K : float
        Amplitude of the spectrum.
    exponent : float, optional
        Power law exponent (3 for polished optics, a common choice).
    f0 : float, optional
        Frequency below which the spectrum flattens.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    phi : array
        Screen of shape (len(y), len(x)).
    """

    rng = np.random.default_rng() if rng is None else rng
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    fx, fy = np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    S = K*(f0**2 + FX**2 + FY**2)**(-exponent/2)
    c = (rng.standard_normal((ny, nx)) + 1j*rng.standard_normal((ny, nx)))*np.sqrt(S/(nx*dx*ny*dy))
    return np.real(np.fft.ifft2(c))*nx*ny


def dm_influence(x, y, n_act, coupling=0.15, D=1.0, margin=0.5):
    """Gaussian influence functions of a square deformable mirror.

    n_act actuators span the pupil diameter D; neighbours couple by
    `coupling` (the influence of an actuator at the position of the next).

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates.
    n_act : int
        Actuators across D.
    coupling : float, optional
        Inter-actuator coupling, 0 < coupling < 1.
    D : float, optional
        Pupil diameter.
    margin : float, optional
        Actuators whose centre lies within D/2 + margin*pitch are kept.

    Returns
    -------
    H : array
        Influence functions of shape (n_kept, len(y), len(x)), 1 at their
        centre.
    centres : array
        Actuator positions of shape (n_kept, 2).
    """

    p = D/n_act
    c = (np.arange(n_act) - (n_act - 1)/2)*p
    CX, CY = np.meshgrid(c, c)
    keep = np.hypot(CX, CY) <= D/2 + margin*p
    centres = np.column_stack([CX[keep], CY[keep]])
    sigma2 = -p**2/(2*np.log(coupling))
    gx = np.exp(-(x[None, :] - centres[:, :1])**2/(2*sigma2))
    gy = np.exp(-(y[None, :] - centres[:, 1:])**2/(2*sigma2))
    H = gy[:, :, None]*gx[:, None, :]
    return H, centres


def efc_jacobian(coronagraph, E0, modes, u, v, select, batch=64):
    """Jacobian of the dark hole field with respect to pupil phase modes.

    For a phase exp(i sum_k c_k h_k) the image field changes by
    dE/dc_k = C[i h_k E0] (the coronagraph C is linear in the field).

    Parameters
    ----------
    coronagraph : Coronagraph
        The coronagraph.
    E0 : array
        Current incoming field (aberrations and DM shape included).
    modes : array
        Phase modes h_k (radians per unit command), shape (n, ny, nx).
    u, v : arrays
        Final focal plane coordinates.
    select : array of bool
        Dark hole pixels, shape (len(v), len(u)).
    batch : int, optional
        Modes propagated at once.

    Returns
    -------
    G : array
        Complex Jacobian of shape (n_pixels, n_modes), in units of
        sqrt(contrast) per unit command.
    """

    cols = []
    for k in range(0, len(modes), batch):
        F = coronagraph.image_field(u, v, 1j*modes[k:k + batch]*E0)
        cols.append(F[:, select].T)
    return np.concatenate(cols, axis=1)


def efc_step(G, E, regularization=1e-3):
    """Electric field conjugation: commands that cancel the dark hole field.

    Minimizes ||E + G c||^2 + alpha ||c||^2 over real commands c, with
    alpha = regularization times the largest eigenvalue of Re(G^H G)
    (Tikhonov regularization).

    Parameters
    ----------
    G : array
        Complex Jacobian (n_pixels, n_modes).
    E : array
        Complex field in the dark hole pixels.
    regularization : float, optional
        Relative Tikhonov parameter.

    Returns
    -------
    c : array
        Command increment (real).
    """

    Gr = np.vstack([G.real, G.imag])
    Er = np.concatenate([E.real, E.imag])
    M = Gr.T @ Gr
    alpha = regularization*np.linalg.eigvalsh(M)[-1]
    return -np.linalg.solve(M + alpha*np.eye(len(M)), Gr.T @ Er)
