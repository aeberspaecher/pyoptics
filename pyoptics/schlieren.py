#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Schlieren, shadowgraph and the knife-edge tests of Foucault and Ronchi.

A schlieren system and the Foucault test are the same optics: a
converging wave forms an image of a small source (the focus of a lens or
mirror of focal length f, or the centre of curvature of a concave mirror
under test), a filter in this focal plane removes part of the light, and
a camera focused on the pupil (the test section or the mirror) sees the
filtered field. Wave optically the focal field is the Fourier transform of
the pupil field, and the camera sees the inverse transform of the filtered
focal field, a 4f system with a pupil filter as in `microscopy` (notebooks
52 and 53) with the roles of object and pupil exchanged.

Conventions: the pupil field is ``A exp(i k W)`` with the wavefront
(optical path) error W relative to the reference sphere centred on the
focus. With the pyoptics Fourier convention a ray from the pupil point
(x, y) reaches the focal plane at the transverse aberration ::

    (xi, eta) = f*(dW/dx, dW/dy)

(no inversion), so a tilt W = alpha*x moves the focus to xi = f*alpha. An
observation plane delta behind the focus (further from the pupil) adds
``W = -delta*r**2/(2 f**2)``, `defocus_wavefront()`. The camera image is
returned on the pupil grid, upright, as seen from the focus.

Fourier plane filters, transmissions on the focal plane grid (callables
mask(XI, ETA)):

- knife_edge(): blocks the half plane xi < position (Foucault 1859,
  Toepler 1864), 1/2 on the edge itself.
- slit(), wire(): a slit lets only the light near the focus through, an
  opaque wire or strip removes it (dark field schlieren, the wire test).
- ronchi_ruling(): a binary amplitude grating (Ronchi 1923).

Images:

- focal_coordinates(): the focal plane coordinates xi = f k_x/k of the
  pupil grid.
- filtered_image(): the coherent image field of a point source.
- extended_source_image(): the incoherent image of an extended source
  (a slit or disk sampled by source points); every source point shifts the
  focal pattern, i.e. the filter the other way.
- transverse_aberration(), geometric_image(): the geometric optics limit,
  a pupil point is as bright as the filter where its ray hits it.
- knife_edge_contrast(): Toepler's contrast ``Delta I/I = f eps/a`` for a
  source image cut down to the height a by the knife, saturated at +-1.
- knife_edge_transfer(): the linear knife-edge image of a weak wavefront
  error for any spatial frequency: Toepler's slope response for coarse
  structure, the phase itself (a single sideband filter) for structure
  finer than lambda*f/a, twice that for a point source.

Test objects:

- defocus_wavefront(): knife (and filter) along the axis.
- conic_wavefront_at_center(), zonal_offset(): a conic mirror tested at
  its centre of curvature, lowest order ``W = -K r**4/(4 R**3)``, and the
  axial offset of the centre of curvature of a zone, ``-K r**2/R`` with a
  fixed source and ``-K r**2/(2 R)`` with a source moving with the knife
  (K = -1 for a paraboloid).
- air_index(): the refractive index of air from temperature and pressure
  by the Gladstone-Dale relation ``n - 1 = K_GD rho`` and the ideal gas
  law.
- shadowgraph_contrast(): the shadowgraph of a weak phase object in the
  geometric limit, ``Delta I/I = -L laplace(W)`` at the distance L.

References
----------
L. Foucault, "Mémoire sur la construction des télescopes en verre argenté",
Ann. Obs. Imp. Paris 5, 197-237 (1859).

A. Toepler, "Beobachtungen nach einer neuen optischen Methode", Max Cohen
und Sohn, Bonn (1864).

V. Ronchi, "Forty years of history of a grating interferometer", Appl. Opt.
3, 437-451 (1964).

J. Texereau, "How to Make a Telescope", 2nd ed., Willmann-Bell (1984).

D. Malacara (ed.), "Optical Shop Testing", 3rd ed., Wiley (2007),
chapters 8 (Foucault, wire and Ronchi tests) and 9.

G. S. Settles, "Schlieren and Shadowgraph Techniques", Springer (2001).

W. Merzkirch, "Flow Visualization", 2nd ed., Academic Press (1987).
"""

import numpy as np

from .fft import FT, inv_FT
from .utils import freq_grid, frequencies, get_dx


__all__ = ["focal_coordinates", "knife_edge", "slit", "wire", "ronchi_ruling",
           "filtered_image", "extended_source_image", "transverse_aberration",
           "geometric_image", "knife_edge_contrast", "knife_edge_transfer", "defocus_wavefront",
           "conic_wavefront_at_center", "zonal_offset", "air_index", "shadowgraph_contrast"]


R_GAS = 8.314462618        # J/(mol K)
M_AIR = 0.028964           # kg/mol, dry air
K_GD_AIR = 2.26E-4         # m**3/kg, Gladstone-Dale constant of air in the visible (about 550 nm)


def focal_coordinates(x, y, wavelength, f):
    """Focal plane coordinates of a pupil grid.

    Parameters
    ----------
    x, y : 1d arrays
        Equidistant pupil coordinates (centered, see `pyoptics.fft`).
    wavelength : number
    f : number
        Focal length (or radius of the converging wave), same unit as x.

    Returns
    -------
    xi, eta : 1d arrays
        ``f*k_x/k`` and ``f*k_y/k``, the positions in the focal plane that
        belong to the samples of FT() of the pupil field.
    """

    scale = wavelength*f/(2*np.pi)
    return (scale*frequencies(x, wavenumbers=True, normal_order=True),
            scale*frequencies(y, wavenumbers=True, normal_order=True))


def _rotate(XI, ETA, position, angle):
    # coordinate across the edge, measured from the edge
    return XI*np.cos(angle) + ETA*np.sin(angle) - position


def knife_edge(XI, ETA, position=0.0, angle=0.0):
    """A knife edge: blocks the half plane on the side of smaller u,
    ``u = xi cos(angle) + eta sin(angle)``.

    Parameters
    ----------
    XI, ETA : arrays
        Focal plane coordinates.
    position : number, optional
        Position of the edge along u. Defaults to 0 (through the focus).
    angle : number, optional
        Direction of the edge normal (radians); the blocked side is
        u < position. 0 is a vertical edge coming in from the left (-xi),
        pi/2 a horizontal edge coming from below. Defaults to 0.

    Returns
    -------
    t : array
        Transmission 0 or 1, and 1/2 for samples on the edge.
    """

    u = _rotate(XI, ETA, position, angle)
    return np.where(u > 0, 1.0, np.where(u < 0, 0.0, 0.5))


def slit(XI, ETA, width, position=0.0, angle=0.0):
    """A slit of the given width: transmits abs(u - position) < width/2,
    u as in `knife_edge()` (angle 0: a vertical slit).

    Parameters
    ----------
    XI, ETA : arrays
    width : number
    position : number, optional
    angle : number, optional

    Returns
    -------
    t : array
        Transmission 0 or 1 (1/2 on the edges).
    """

    u = np.abs(_rotate(XI, ETA, position, angle))
    return np.where(u < width/2, 1.0, np.where(u > width/2, 0.0, 0.5))


def wire(XI, ETA, width, position=0.0, angle=0.0):
    """An opaque wire (strip) of the given width, the complement of
    `slit()`.

    Parameters
    ----------
    XI, ETA : arrays
    width : number
    position : number, optional
    angle : number, optional

    Returns
    -------
    t : array
    """

    return 1.0 - slit(XI, ETA, width, position, angle)


def ronchi_ruling(XI, ETA, period, position=0.0, angle=0.0, duty=0.5):
    """A Ronchi ruling: opaque and clear bars of the period, bars parallel
    to the edge direction of `knife_edge()` (angle 0: vertical lines).

    Parameters
    ----------
    XI, ETA : arrays
    period : number
    position : number, optional
        Centre of a clear bar. Defaults to 0.
    angle : number, optional
    duty : number, optional
        Clear fraction of the period. Defaults to 1/2.

    Returns
    -------
    t : array
        Transmission 0 or 1.
    """

    u = _rotate(XI, ETA, position, angle)/period
    frac = u - np.floor(u + 0.5)          # in [-1/2, 1/2), 0 at bar centres
    return (np.abs(frac) < duty/2).astype(float)


def _mask_array(mask, XI, ETA):
    return mask(XI, ETA) if callable(mask) else np.asarray(mask)


def filtered_image(field, x, y, wavelength, f, mask):
    """Coherent image of a pupil field through a focal plane filter.

    Parameters
    ----------
    field : 2d array
        Pupil field, shape (len(y), len(x)), e.g. ``A*exp(1j*k*W)``.
    x, y : 1d arrays
        Pupil coordinates.
    wavelength, f : numbers
    mask : callable or 2d array
        Filter transmission mask(XI, ETA) on the focal plane grid of
        `focal_coordinates()`, or the sampled array.

    Returns
    -------
    image : 2d array
        Complex field in the image of the pupil (unit magnification,
        upright); ``abs(image)**2`` is the camera image.
    """

    xi, eta = focal_coordinates(x, y, wavelength, f)
    XI, ETA = np.meshgrid(xi, eta)
    return inv_FT(FT(field)*_mask_array(mask, XI, ETA))


def extended_source_image(field, x, y, wavelength, f, mask, offsets, weights=None):
    """Incoherent image of an extended source through a focal plane filter.

    A source point displaced so that its image in the focal plane moves by
    (s_xi, s_eta) shifts the whole focal pattern; the filter then acts as
    if it were shifted by (-s_xi, -s_eta). The images of all source points
    add in intensity.

    Parameters
    ----------
    field : 2d array
        Pupil field for the central source point.
    x, y : 1d arrays
    wavelength, f : numbers
    mask : callable
        mask(XI, ETA), see `filtered_image()`.
    offsets : sequence of (s_xi, s_eta) or 1d array
        Displacements of the source image in the focal plane. A 1d array
        holds displacements along xi only (a slit source parallel to a
        vertical edge).
    weights : 1d array, optional
        Weights of the source points. Defaults to equal weights. They are
        normalized to a sum of 1.

    Returns
    -------
    image : 2d array
        Intensity.
    """

    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim == 1:
        offsets = np.c_[offsets, np.zeros_like(offsets)]
    w = np.ones(len(offsets)) if weights is None else np.asarray(weights, dtype=float)
    w = w/w.sum()
    xi, eta = focal_coordinates(x, y, wavelength, f)
    XI, ETA = np.meshgrid(xi, eta)
    spectrum = FT(field)
    image = np.zeros(np.shape(field))
    for (s_xi, s_eta), wi in zip(offsets, w):
        image += wi*np.abs(inv_FT(spectrum*mask(XI + s_xi, ETA + s_eta)))**2
    return image


def transverse_aberration(W, x, y, f):
    """Transverse ray aberration in the focal plane, ``f*grad(W)``.

    Parameters
    ----------
    W : 2d array
        Wavefront error (optical path), shape (len(y), len(x)).
    x, y : 1d arrays
    f : number

    Returns
    -------
    eps_xi, eps_eta : 2d arrays
        Where the ray from each pupil point hits the focal plane.
    """

    dW_dy, dW_dx = np.gradient(W, get_dx(y), get_dx(x))
    return f*dW_dx, f*dW_dy


def geometric_image(W, x, y, f, mask, offsets=None, weights=None):
    """Geometric optics image through a focal plane filter: every pupil
    point is as bright as the filter transmission where its ray (from each
    source point) hits it.

    Parameters
    ----------
    W : 2d array
        Wavefront error.
    x, y : 1d arrays
    f : number
    mask : callable
        mask(XI, ETA).
    offsets, weights : optional
        Source points, see `extended_source_image()`. Defaults to a point
        source.

    Returns
    -------
    image : 2d array
        Intensity relative to the unfiltered image (no pupil mask applied).
    """

    eps_xi, eps_eta = transverse_aberration(W, x, y, f)
    if offsets is None:
        return mask(eps_xi, eps_eta).astype(float)
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim == 1:
        offsets = np.c_[offsets, np.zeros_like(offsets)]
    w = np.ones(len(offsets)) if weights is None else np.asarray(weights, dtype=float)
    w = w/w.sum()
    image = np.zeros(np.shape(W))
    for (s_xi, s_eta), wi in zip(offsets, w):
        image += wi*mask(eps_xi + s_xi, eps_eta + s_eta)
    return image


def knife_edge_contrast(eps, f, a):
    """Toepler's schlieren contrast of a knife edge.

    The source image of uniform brightness is cut by the knife down to the
    unobstructed height a. A ray deflected by the angle eps (across the
    edge, towards the open side positive) moves the source image by f*eps,
    and the image brightness changes by ::

        Delta I/I = f eps/a,

    saturating at -1 (source image fully cut off) and +1 (source image
    fully passing; for a knife at the middle of a source image of width 2a).

    Parameters
    ----------
    eps : number or array
        Deflection angle (radians).
    f : number
        Focal length of the schlieren head (the lens or mirror forming the
        source image at the knife).
    a : number
        Unobstructed height of the source image.

    Returns
    -------
    contrast : number or array
    """

    return np.clip(f*np.asarray(eps)/a, -1.0, 1.0)


def knife_edge_transfer(W, x, y, wavelength, f, a=0.0, angle=0.0):
    """Linear image of a weak wavefront error with a knife edge cutting the
    source image in the middle.

    A ripple ``W = w cos(q u)`` across the edge (u as in `knife_edge()`)
    diffracts into the orders at ``xi_1 = +-f q/k``. With a uniform slit
    source image of the half width a, much larger than the diffraction
    spot, the knife passes the +1 order for the fraction min(xi_1/a, 1) of
    the source points and the -1 order for none, and ::

        Delta I/I_b = -k w min(xi_1/a, 1) sin(q u):

    for xi_1 < a the slope response ``f/a dW/du`` of `knife_edge_contrast()`,
    for xi_1 > a the phase k w of the ripple. A point source (a = 0) halves
    the zero order at the edge and doubles the response to ``2 k w`` at all
    frequencies. In Fourier space the response is
    ``i k sgn(q) min(f abs(q)/(k a), 1)`` (``2 i k sgn(q)`` for a = 0).

    Parameters
    ----------
    W : 2d array
        Wavefront error, ``abs(k W) << 1`` (any constant and tilt only shift
        the result; the pupil edge is ignored).
    x, y : 1d arrays
    wavelength, f : numbers
    a : number, optional
        Half width of the source image across the knife; 0 (default) for a
        point source.
    angle : number, optional
        Edge normal, see `knife_edge()`.

    Returns
    -------
    contrast : 2d array
        ``Delta I/I_b``, relative to the background with a perfect wavefront.
    """

    k = 2*np.pi/wavelength
    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    q = K_X*np.cos(angle) + K_Y*np.sin(angle)
    if a == 0:
        T = 2j*k*np.sign(q)
    else:
        T = 1j*k*np.sign(q)*np.minimum(f*np.abs(q)/(k*a), 1.0)
    return np.real(inv_FT(FT(W)*T))


def defocus_wavefront(X, Y, f, delta):
    """Wavefront error of a focal plane displaced by delta along the axis.

    Parameters
    ----------
    X, Y : arrays
        Pupil coordinates.
    f : number
        Distance pupil to focus.
    delta : number
        Displacement of the observation (knife) plane, positive away from
        the pupil (outside focus).

    Returns
    -------
    W : array
        ``-delta*(X**2 + Y**2)/(2 f**2)``: a ray from the pupil height r
        crosses the displaced plane at ``-r*delta/f``.
    """

    return -delta*(X**2 + Y**2)/(2*f**2)


def conic_wavefront_at_center(X, Y, R, K):
    """Wavefront error of a conic mirror (radius R, conic constant K) tested
    at its centre of curvature, in lowest order.

    The sag of the conic differs from the sphere by ``-K r**4/(8 R**3)``;
    on reflection the error doubles. A paraboloid (K = -1) gives
    ``r**4/(4 R**3)``: its outer zones focus further away from the mirror.

    Parameters
    ----------
    X, Y : arrays
    R : number
    K : number
        Conic constant (0 sphere, -1 paraboloid).

    Returns
    -------
    W : array
    """

    return -K*(X**2 + Y**2)**2/(4*R**3)


def zonal_offset(r, R, K, moving_source=False):
    """Axial offset of the centre of curvature of the zone r of a conic
    mirror from the paraxial one (the knife reading of the Foucault zonal
    test).

    Parameters
    ----------
    r : number or array
        Zone radius.
    R : number
    K : number
        Conic constant.
    moving_source : bool, optional
        True for a tester whose source moves with the knife (the beam then
        refocuses twice as fast). Defaults to False (fixed source).

    Returns
    -------
    delta : number or array
        ``-K r**2/R`` (fixed source) or ``-K r**2/(2 R)`` (moving source).
    """

    return -K*np.asarray(r)**2/(R*(2 if moving_source else 1))


def air_index(T, p=101325.0, K=K_GD_AIR):
    """Refractive index of dry air by the Gladstone-Dale relation.

    Parameters
    ----------
    T : number or array
        Temperature in K.
    p : number or array, optional
        Pressure in Pa. Defaults to 101325 Pa.
    K : number, optional
        Gladstone-Dale constant in m**3/kg. Defaults to 2.26E-4 (visible).

    Returns
    -------
    n : number or array
        ``1 + K*p*M/(R T)``; 1.000277 at 15 °C.
    """

    return 1.0 + K*np.asarray(p)*M_AIR/(R_GAS*np.asarray(T))


def shadowgraph_contrast(W, x, y, L):
    """Shadowgraph of a weak phase object in the geometric limit.

    Rays leave the object with the slopes grad(W); after the distance L the
    irradiance is redistributed as ``I/I_0 = 1/det(1 + L hess(W))``, to first
    order ``1 - L laplace(W)`` (the transport of intensity equation).

    Parameters
    ----------
    W : 2d array
        Optical path imprinted by the object.
    x, y : 1d arrays
    L : number
        Distance from the object (or the defocus of the camera).

    Returns
    -------
    contrast : 2d array
        ``Delta I/I_0 = -L laplace(W)``.
    """

    dx, dy = get_dx(x), get_dx(y)
    lap = (np.gradient(np.gradient(W, dx, axis=1), dx, axis=1)
           + np.gradient(np.gradient(W, dy, axis=0), dy, axis=0))
    return -L*lap
