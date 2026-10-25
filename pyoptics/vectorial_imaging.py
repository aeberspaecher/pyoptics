#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Vectorial, partially coherent imaging (Abbe's method), e.g. for
lithography.

A thin mask is illuminated by a partially coherent source. Each source point
is a plane wave that is mutually incoherent with all other source points; its
direction is given by the partial coherence coordinates (s_x, s_y) (sigma in
lithography): the transverse wave vector of the illuminating plane wave in
image space is k_0*NA*(s_x, s_y). The mask diffracts it into plane waves
(diffraction orders) with the transverse wave vectors k_0*NA*s + k_mask, which
the projection lens passes if ``|k_perp|`` <= k_0*NA. All quantities refer to the
image side: pitches and NA are image-side values (for a reduction lens the
mask dimensions divided by the reduction ratio).

Polarization: the illumination has a Jones vector [E_x, E_y] at the mask,
which a thin (Kirchhoff) mask keeps. In the vectorial model each diffraction
order is then turned into a plane wave in the image space medium (refractive
index n) by the aplanatic lens (propagators.aplanatic_rotation()): its TE (s)
component stays perpendicular to the plane of incidence, its TM (p) component
tilts with the wave and gets a z component. Interfering TM waves are not
parallel, which reduces the image contrast at high NA. An optional
obliquity (radiometric) factor 1/sqrt(cos(theta)) conserves the energy
flowing through the lens, as in propagators.vectorial_mansuripur(). In the
scalar model the Jones vector is kept as it is.

The image is the energy density ``|E_x|**2`` + ``|E_y|**2`` + ``|E_z|**2``, summed over
the source points (weighted) and over the mutually incoherent polarization
states of each source point.

Polarization of the illumination can be given as

- a Jones vector, e.g. polarization.jones.J_V,
- a Stokes vector (shape (4,)), e.g. polarization.mueller.S_unpol (it is
  decomposed into two incoherent Jones vectors, see
  polarization.jones_from_stokes()),
- one of the strings "x", "y", "unpolarized", "TE" (= "azimuthal", or
  tangential: perpendicular to the radius vector of the source point) and
  "TM" (= "radial"); "TE"/"TM" refer to the plane of incidence of the
  illumination, on the axis they fall back to y and x,
- a callable pol(s_x, s_y) returning one of the above for each source point.

For lines along y (pitch along x) and illumination in the x-z plane (e.g. an
x dipole), "y" is TE (s) polarization for all diffraction orders, and "x" is
TM (p).

Convention: fields vary as exp(i(k.r - omega*t)), a defocus z > 0 (image
plane behind the focal plane) multiplies each plane wave by
exp(+i*k_z*z).
"""

import numpy as np

from .fft import FT, inv_FT
from .propagators import aplanatic_rotation
from .polarization import jones_from_stokes, TE_TM_basis
from .utils import freq_grid


# masks and sources:

def lines_spaces_orders(pitch, linewidth, max_order, line=0.0, space=1.0):
    """Fourier coefficients of a lines and spaces mask.

    One period consists of a line of width linewidth (centered at x = 0) with
    the complex amplitude transmission line and a space with the
    transmission space: t(x) = sum_m c_m exp(2*pi*i*m*x/pitch).

    Parameters
    ----------
    pitch, linewidth : numbers
    max_order : int
        Orders -max_order ... max_order are returned.
    line, space : complex numbers, optional
        Amplitude transmissions; the defaults describe a binary (chrome on
        glass) mask. An attenuated phase shift mask with 6 % intensity
        transmission has line = -sqrt(0.06).

    Returns
    -------
    m : array of ints
    c_m : array
    """

    m = np.arange(-max_order, max_order + 1)
    duty = linewidth/pitch
    c_m = (line - space)*duty*np.sinc(m*duty)
    c_m[m == 0] += space

    return m, c_m.astype(complex)


def _angle_dist(phi, phi_0):
    """Angular distance in degrees, in [0, 180]."""

    return np.abs((phi - phi_0 + 180) % 360 - 180)


def _sigma_grid(n_samples):
    """Square grid of n_samples x n_samples points on [-1, 1]**2 in the
    sigma plane, flattened."""

    s = np.linspace(-1, 1, n_samples)
    S_x, S_y = np.meshgrid(s, s)

    return S_x.ravel(), S_y.ravel()


def source_points(shape="conventional", sigma=0.5, sigma_in=0.0, opening_angle=None,
                  n_samples=21, axis="x"):
    """Sample an illumination source on a square grid in the sigma plane.

    Parameters
    ----------
    shape : str
        "coherent" (one point at sigma), "conventional" (disk of radius
        sigma), "annular" (sigma_in <= ``|s|`` <= sigma), "dipole" (two poles on
        the axis, annular sector with the opening angle) or "quasar" (four
        poles on the diagonals).
    sigma : number or tuple
        Outer radius of the source; for "coherent" the source point
        (s_x, s_y) or a number s_x.
    sigma_in : number, optional
        Inner radius for "annular", "dipole", "quasar".
    opening_angle : number, optional
        Opening angle of the poles of "dipole" and "quasar" in degrees
        (default 35°).
    n_samples : int, optional
        Number of grid points across the diameter of the unit circle.
    axis : "x" or "y", optional
        Orientation of the dipole.

    Returns
    -------
    s_x, s_y, weights : arrays
        Source points and their weights (summing to 1).
    """

    if shape == "coherent":
        s_x, s_y = (sigma, 0.0) if np.isscalar(sigma) else sigma
        return np.array([s_x], dtype=float), np.array([s_y], dtype=float), np.array([1.0])

    S_x, S_y = _sigma_grid(n_samples)
    r, phi = np.hypot(S_x, S_y), np.degrees(np.arctan2(S_y, S_x))
    half_opening = (35.0 if opening_angle is None else opening_angle)/2

    if shape == "conventional":
        inside = r <= sigma
    elif shape == "annular":
        inside = (r <= sigma) & (r >= sigma_in)
    elif shape == "dipole":
        phi_0 = 0.0 if axis == "x" else 90.0
        inside = ((r <= sigma) & (r >= sigma_in)
                  & ((_angle_dist(phi, phi_0) <= half_opening)
                     | (_angle_dist(phi, phi_0 + 180) <= half_opening)))
    elif shape == "quasar":
        inside = ((r <= sigma) & (r >= sigma_in)
                  & np.any([_angle_dist(phi, a) <= half_opening for a in (45, 135, 225, 315)], axis=0))
    else:
        raise ValueError("Unknown source shape {}".format(shape))

    if not np.any(inside):
        raise ValueError("No source point inside the source; increase n_samples")

    w = np.ones(np.count_nonzero(inside))

    return S_x[inside], S_y[inside], w/w.sum()


def _polarization_states(polarization, s_x, s_y):
    """Return a list of (weight, Jones vector) for one source point."""

    if callable(polarization):
        return _polarization_states(polarization(s_x, s_y), s_x, s_y)

    if isinstance(polarization, str):
        name = polarization.lower()
        if name == "x":
            return [(1.0, np.array([1.0, 0.0]))]
        if name == "y":
            return [(1.0, np.array([0.0, 1.0]))]
        if name == "unpolarized":
            return [(0.5, np.array([1.0, 0.0])), (0.5, np.array([0.0, 1.0]))]
        if name in ("te", "azimuthal", "tangential", "tm", "radial"):
            e_TE, e_TM = TE_TM_basis(s_x, s_y)
            return [(1.0, e_TE if name in ("te", "azimuthal", "tangential") else e_TM)]
        raise ValueError("Unknown polarization {}".format(polarization))

    p = np.asarray(polarization)
    if p.shape == (4,):
        w, J = jones_from_stokes(p/p[0])
        return [(w[i], J[i]) for i in range(2) if w[i] > 1E-12]
    if p.shape == (2,):
        return [(1.0, p/np.linalg.norm(p))]

    raise ValueError("Polarization must be a string, a Jones vector, a Stokes "
                     "vector or a callable")


def _plane_wave_fields(Sigma_x, Sigma_y, jones, vectorial, obliquity):
    """Field vectors [E_x, E_y, E_z] of plane waves with the directions
    (Sigma_x, Sigma_y) (in the image space medium) for the pupil Jones vector
    jones; shape (3,) + Sigma_x.shape."""

    jones = np.asarray(jones, dtype=complex)
    if not vectorial:
        E = np.zeros((3,) + np.shape(Sigma_x), dtype=complex)
        E[0], E[1] = jones[0], jones[1]
        return E

    R = aplanatic_rotation(Sigma_x, Sigma_y)
    E = R[:, 0]*jones[0] + R[:, 1]*jones[1]
    if obliquity:
        cos_theta = np.sqrt(np.clip(1 - Sigma_x**2 - Sigma_y**2, 1E-12, None))
        E = E/np.sqrt(cos_theta)

    return E


def _pupil_phase(phase, P_x, P_y):
    if phase is None:
        return 1.0
    return np.exp(1j*phase(P_x, P_y))


# imaging:

def image_1d(x, pitch, orders, NA, wavelength, source=None, polarization="unpolarized",
             n=1.0, defocus=0.0, vectorial=True, obliquity=True, phase=None,
             return_components=False):
    """Aerial image of a periodic one-dimensional mask (lines along y).

    The diffraction orders of the mask are summed up directly, so the pitch
    is arbitrary and no FFT grid is involved.

    Parameters
    ----------
    x : array
        Positions (image side) at which the image is computed.
    pitch : number
        Image-side pitch.
    orders : tuple (m, c_m)
        Diffraction orders and their amplitudes, e.g. from
        lines_spaces_orders().
    NA : number
        Image-side numerical aperture (NA <= n).
    wavelength : number
        Vacuum wavelength.
    source : tuple (s_x, s_y, weights), optional
        Source points from source_points(); defaults to coherent, normal
        illumination.
    polarization : see module docstring
    n : number, optional
        Refractive index of the image space (immersion liquid).
    defocus : number, optional
        Distance of the image plane from the focal plane.
    vectorial : bool, optional
        If False, use the scalar model (no rotation of the polarization).
    obliquity : bool, optional
        Use the radiometric factor 1/sqrt(cos(theta)) (vectorial model only).
    phase : callable, optional
        Pupil phase phase(P_x, P_y) in radians on normalized pupil
        coordinates (unit disk), e.g. a Zernike aberration as in imaging.
    return_components : bool, optional
        Also return the intensities of the three field components.

    Returns
    -------
    I : array
        Image intensity ``|E|**2`` at x.
    I_xyz : array, optional
        Shape (3,) + x.shape: ``|E_x|**2``, ``|E_y|**2``, ``|E_z|**2``.
    """

    if NA > n:
        raise ValueError("NA = {} exceeds the refractive index n = {}".format(NA, n))
    if source is None:
        source = source_points("coherent", (0.0, 0.0))

    k0 = 2*np.pi/wavelength
    k = k0*n
    x = np.asarray(x, dtype=float)
    m, c_m = map(np.asarray, orders)
    I_xyz = np.zeros((3,) + x.shape)

    for s_x, s_y, w in zip(*source):
        k_x = k0*NA*s_x + 2*np.pi*m/pitch
        k_y = k0*NA*s_y*np.ones_like(k_x)
        passed = k_x**2 + k_y**2 <= (k0*NA)**2*(1 + 1E-12)
        if not np.any(passed):
            continue
        k_x, k_y, c = k_x[passed], k_y[passed], c_m[passed]
        Sigma_x, Sigma_y = k_x/k, k_y/k
        k_z = k*np.sqrt(1 - Sigma_x**2 - Sigma_y**2)
        amplitude = c*np.exp(1j*k_z*defocus)*_pupil_phase(phase, k_x/(k0*NA), k_y/(k0*NA))
        waves = np.exp(1j*np.outer(x.ravel(), k_x))  # (N_x, N_orders)

        for weight, jones in _polarization_states(polarization, s_x, s_y):
            E_orders = _plane_wave_fields(Sigma_x, Sigma_y, jones, vectorial, obliquity)
            E = waves @ (E_orders*amplitude).T  # (N_x, 3)
            I_xyz += w*weight*(np.abs(E.T)**2).reshape(I_xyz.shape)

    I = I_xyz.sum(axis=0)

    return (I, I_xyz) if return_components else I


def image_2d(mask, x, y, NA, wavelength, source=None, polarization="unpolarized",
             n=1.0, defocus=0.0, vectorial=True, obliquity=True, phase=None,
             return_components=False):
    """Aerial image of a two-dimensional mask with Abbe's method.

    The mask (complex amplitude transmission, image-side coordinates) is
    taken as periodic in the computational window. For each source point the
    mask spectrum is filtered by the pupil, shifted by the illumination
    direction; the three field components are transformed back separately.

    Parameters
    ----------
    mask : array
        Amplitude transmission sampled on x, y.
    x, y : arrays
        Linear, centered coordinates (see pyoptics.fft).
    NA, wavelength, source, polarization, n, defocus, vectorial, obliquity,
    phase, return_components
        As for image_1d().

    Returns
    -------
    I : array
        Image intensity, shape (Ny, Nx).
    I_xyz : array, optional
        Shape (3, Ny, Nx).
    """

    if NA > n:
        raise ValueError("NA = {} exceeds the refractive index n = {}".format(NA, n))
    if source is None:
        source = source_points("coherent", (0.0, 0.0))

    k0 = 2*np.pi/wavelength
    k = k0*n
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    spectrum = FT(mask)
    I_xyz = np.zeros((3,) + np.shape(mask))

    for s_x, s_y, w in zip(*source):
        k_x, k_y = K_x + k0*NA*s_x, K_y + k0*NA*s_y
        passed = k_x**2 + k_y**2 <= (k0*NA)**2*(1 + 1E-12)
        Sigma_x, Sigma_y = np.where(passed, k_x/k, 0.0), np.where(passed, k_y/k, 0.0)
        k_z = k*np.sqrt(1 - Sigma_x**2 - Sigma_y**2)
        amplitude = np.where(passed, spectrum*np.exp(1j*k_z*defocus), 0.0)
        amplitude = amplitude*_pupil_phase(phase, k_x/(k0*NA), k_y/(k0*NA))

        for weight, jones in _polarization_states(polarization, s_x, s_y):
            E_k = _plane_wave_fields(Sigma_x, Sigma_y, jones, vectorial, obliquity)*amplitude
            for i in range(3):
                if vectorial or i < 2:
                    I_xyz[i] += w*weight*np.abs(inv_FT(E_k[i]))**2

    I = I_xyz.sum(axis=0)

    return (I, I_xyz) if return_components else I


# image metrics:

def contrast(I):
    """Michelson contrast (I_max - I_min)/(I_max + I_min) of an image."""

    I_max, I_min = np.max(I), np.min(I)

    return (I_max - I_min)/(I_max + I_min)


def nils(x, I, x_edge, linewidth):
    """Normalized image log slope NILS = ``linewidth*|d ln(I)/dx|`` at x_edge.

    Parameters
    ----------
    x, I : arrays
        Sampled image (x increasing, finely sampled).
    x_edge : number
        Position of the nominal edge of the feature.
    linewidth : number
        Nominal width of the feature.

    Returns
    -------
    NILS : number
    """

    log_slope = np.gradient(np.log(I), x)

    return linewidth*np.abs(np.interp(x_edge, x, log_slope))


def two_beam_contrast(theta, polarization):
    """Contrast of the interference of two plane waves of equal amplitude
    that travel at the angles +theta and -theta to the z axis.

    TE (s) polarized waves are parallel and give contrast 1, TM (p) polarized
    waves enclose the angle 2*theta and give ``|cos(2*theta)|``. Unpolarized light
    is the incoherent mean of both, I = 2 + (1 + cos(2*theta))*cos(Delta),
    with the contrast (1 + cos(2*theta))/2 = cos(theta)**2.

    Parameters
    ----------
    theta : number or array
        Half the angle between the waves, in the medium [rad].
    polarization : "TE", "TM" or "unpolarized"

    Returns
    -------
    contrast : number or array
    """

    c = np.cos(2*np.asarray(theta))
    if polarization == "TE":
        return np.ones_like(c)
    if polarization == "TM":
        return np.abs(c)
    if polarization == "unpolarized":
        return (1 + c)/2
    raise ValueError("polarization must be TE, TM or unpolarized")
