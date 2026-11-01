#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Microscopy: partially coherent bright field imaging, phase contrast, DIC,
confocal microscopy and Fourier ptychography.

All functions describe a scalar microscope with a thin object (transmission
function t(x, y), complex) in object space: lengths are object space lengths,
the magnification is left out. The objective is a band limited (Debye)
system with the numerical aperture NA = n*sin(alpha) in the immersion medium
n, so the pupil passes the transverse wave vectors ``|k_perp|`` <= k0*NA
(k0 = 2*pi/wavelength). Its pupil is sampled on the centered frequency grid
of x, y (utils.freq_grid(), as in imaging and fft.FT()).

Bright field imaging after Abbe and Köhler:

- source_points(): the condenser aperture (disk, annulus, off-axis point or
  any list of points) as plane wave directions on the frequency grid, with
  weights. sigma = NA_condenser/NA is the partial coherence factor.
- pupil(): the objective pupil with defocus (exact k_z in the medium),
  aberrations and masks (e.g. a phase ring).
- abbe_image(): the partially coherent image as the incoherent sum of the
  coherent images for all source points (Abbe's method).
- weak_object_transfer(): amplitude and phase transfer functions of weak
  objects for a given condenser and pupil (Hopkins' transmission cross
  coefficient with one argument at zero).
- grating_cutoff(): the smallest period of a grating that is still imaged,
  wavelength/(NA + NA_condenser).

Phase contrast methods:

- phase_ring(): Zernike's phase plate as a pupil mask, an annulus that shifts
  the phase of the direct light and attenuates it; with the annular condenser
  of source_points() it gives positive or negative phase contrast with its
  halo and shade-off.
- dic_object(): Nomarski differential interference contrast, the effective
  object of two copies sheared by a Wollaston prism, with a bias, between
  crossed polarizers (for a compensated condenser prism); image it with
  abbe_image().

Confocal microscopy:

- psf_3d(): the 3D amplitude PSF from the pupil by angular spectrum
  propagation, uniform or aplanatic (sine condition) pupil.
- airy_unit(): the Airy disk diameter 1.22*wavelength/NA used for pinholes.
- confocal_psf(): the fluorescence confocal PSF, the product of the
  excitation PSF and the detection PSF convolved with the pinhole.
- plane_response(): the signal of a thin fluorescent plane vs. defocus, the
  measure of optical sectioning.
- mirror_response(): the axial response of a reflection confocal microscope
  to a plane mirror (point pinhole).

Fourier ptychography:

- led_array(): illumination wave vectors of an LED matrix below the sample.
- fpm_images(): the low resolution images of a thin object under each LED.
- fpm_reconstruct(): the object spectrum stitched together in k-space by
  alternating projections, with embedded pupil recovery (EPRY).
- circle_overlap(): the overlap of two pupil circles in k-space (FPM needs
  roughly 40 to 60 % between neighbouring LEDs).

Conventions: fields vary as exp(i(k.r - omega*t)); the object phase is
phi = k0*OPD with OPD = integral (n_s - n_m) dz, so optically denser objects
have positive phase. A tilted plane wave exp(i*(k_sx*x + k_sy*y)) shifts the
object spectrum by (k_sx, k_sy). abbe_image() samples the source on the
frequency grid, so these shifts are exact integer shifts; the grid must
resolve k0*(NA + NA_condenser) (dx <= wavelength/(2*(NA + NA_condenser))),
and wavelength/(4*NA) to sample the image intensity.

References
----------
E. Abbe, "Beiträge zur Theorie des Mikroskops und der mikroskopischen
Wahrnehmung", Arch. Mikrosk. Anat. 9, 413-468 (1873).

H. H. Hopkins, "On the diffraction theory of optical images", Proc. R. Soc.
Lond. A 217, 408-432 (1953).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), chapters 8.6 (phase contrast) and 10.5 (partially coherent
imaging).

F. Zernike, "Phase contrast, a new method for the microscopic observation of
transparent objects", Physica 9, 686-698 (1942).

G. Nomarski, "Microinterféromètre différentiel à ondes polarisées", J. Phys.
Radium 16, 9S-13S (1955).

C. Preza, D. L. Snyder and J.-A. Conchello, "Theoretical development and
experimental evaluation of imaging models for differential-interference-
contrast microscopy", J. Opt. Soc. Am. A 16, 2185-2199 (1999).

N. Streibl, "Three-dimensional imaging by a microscope", J. Opt. Soc. Am. A
2, 121-127 (1985).

T. Wilson and C. Sheppard, "Theory and Practice of Scanning Optical
Microscopy", Academic Press (1984).

J. B. Pawley (ed.), "Handbook of Biological Confocal Microscopy", 3rd ed.,
Springer (2006).

G. Zheng, R. Horstmeyer and C. Yang, "Wide-field, high-resolution Fourier
ptychographic microscopy", Nat. Photonics 7, 739-745 (2013).

X. Ou, G. Zheng and C. Yang, "Embedded pupil function recovery for Fourier
ptychographic microscopy", Opt. Express 22, 4960-4972 (2014).

L. Tian, X. Li, K. Ramchandran and L. Waller, "Multiplexed coded
illumination for Fourier ptychography with an LED array microscope",
Biomed. Opt. Express 5, 2376-2389 (2014).

J. Mertz, "Introduction to Optical Microscopy", 2nd ed., Cambridge
University Press (2019).
"""

import warnings

import numpy as np
from numpy.fft import fftshift, ifftshift

from .fft import FT, inv_FT, fft2, ifft2
from .utils import freq_grid, frequencies, get_dx


__all__ = ["source_points", "pupil", "abbe_image", "weak_object_transfer",
           "grating_cutoff", "phase_ring", "dic_object", "psf_3d", "airy_unit",
           "confocal_psf", "plane_response", "mirror_response", "led_array",
           "fpm_images", "fpm_reconstruct", "circle_overlap"]


_AXES = (-2, -1)


def _FT2(a):
    """Centered 2D FFT over the last two axes (batched FT())."""
    return fftshift(fft2(ifftshift(a, axes=_AXES), axes=_AXES), axes=_AXES)


def _inv_FT2(a):
    """Centered inverse 2D FFT over the last two axes (batched inv_FT())."""
    return fftshift(ifft2(ifftshift(a, axes=_AXES), axes=_AXES), axes=_AXES)


def _dk(x, y):
    """Spacings of the frequency grid of x, y (rad per length unit)."""
    return 2*np.pi/(len(x)*get_dx(x)), 2*np.pi/(len(y)*get_dx(y))


def _shift(field, x, y, s_x, s_y):
    """field(x - s_x, y - s_y) by a Fourier shift (periodic, any s)."""
    K_x, K_y = freq_grid(x, y)
    return inv_FT(FT(field)*np.exp(-1j*(K_x*s_x + K_y*s_y)))


def source_points(x, y, wavelength, NA_outer, NA_inner=0.0, center=(0.0, 0.0), step=1,
                  points=None, weights=None):
    """Condenser aperture sampled as plane waves on the frequency grid.

    In Köhler illumination every point of the condenser aperture sends a
    plane wave through the object. The points are taken from the frequency
    grid of x, y, so that the tilts are exact shifts of the object spectrum
    in abbe_image().

    Parameters
    ----------
    x, y : 1d arrays
        Object coordinates.
    wavelength : number
        Vacuum wavelength.
    NA_outer : number
        Outer numerical aperture of the condenser (sigma*NA). 0 gives a single
        plane wave (coherent illumination).
    NA_inner : number, optional
        Inner numerical aperture for annular illumination (phase contrast,
        dark field). Defaults to 0 (disk).
    center : tuple of numbers, optional
        Center (NA_x, NA_y) of the aperture, e.g. for oblique illumination
        with a small off-axis disk. Defaults to (0, 0).
    step : int, optional
        Take every step-th grid point in k_x and k_y (coarser source sampling,
        faster abbe_image()). Defaults to 1.
    points : array of shape (M, 2), optional
        Explicit illumination directions (NA_x, NA_y) instead of an aperture;
        rounded to the grid.
    weights : array of shape (M,), optional
        Intensities of the explicit points. Defaults to equal weights.

    Returns
    -------
    k_x, k_y : 1d arrays
        Transverse wave vectors of the plane waves (on the grid).
    w : 1d array
        Weights (intensities), normalized to sum 1.
    """

    k0 = 2*np.pi/wavelength
    dkx, dky = _dk(x, y)
    if points is not None:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        k_x = np.round(points[:, 0]*k0/dkx)*dkx
        k_y = np.round(points[:, 1]*k0/dky)*dky
        w = np.ones(len(k_x)) if weights is None else np.asarray(weights, dtype=float)
        return k_x, k_y, w/np.sum(w)

    c_x, c_y = np.round(center[0]*k0/dkx), np.round(center[1]*k0/dky)
    n_x = int(np.ceil(NA_outer*k0/dkx)) + 1
    n_y = int(np.ceil(NA_outer*k0/dky)) + 1
    i_x = np.arange(-(n_x//step)*step, n_x + 1, step)
    i_y = np.arange(-(n_y//step)*step, n_y + 1, step)
    I_x, I_y = np.meshgrid(i_x, i_y)
    rho = np.hypot(I_x*dkx, I_y*dky)
    # half a sample of tolerance keeps the aperture symmetric and its area right
    tol = 1e-9*k0
    keep = (rho <= NA_outer*k0 + tol) & (rho >= NA_inner*k0 - tol)
    if NA_outer > 0 and NA_inner > 0 and not np.any(keep):
        raise ValueError("The annulus contains no grid point; use a larger window or a "
                         "wider annulus")
    if not np.any(keep):
        keep = (I_x == 0) & (I_y == 0)
    k_x = (I_x[keep] + c_x)*dkx
    k_y = (I_y[keep] + c_y)*dky
    w = np.ones(len(k_x))

    return k_x, k_y, w/np.sum(w)


def pupil(x, y, wavelength, NA, n=1.0, defocus=0.0, phase=None, mask=None):
    """Objective pupil on the centered frequency grid of x, y.

    Parameters
    ----------
    x, y : 1d arrays
        Object coordinates.
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture n*sin(alpha), at most n.
    n : number, optional
        Refractive index of the immersion (and object) medium. Defaults to 1.
    defocus : number, optional
        Distance of the observed plane from the object along z, in the medium
        (positive: focus behind the object). The pupil gets the exact phase
        exp(i*(k_z - n*k0)*defocus). Defaults to 0.
    phase : array or callable, optional
        Pupil aberration in radians, as array on the frequency grid or as
        callable phase(P_x, P_y) of the normalized pupil coordinates
        (k_x, k_y)/(k0*NA), e.g. a Zernike polynomial.
    mask : array or callable, optional
        Complex pupil mask (apodization, phase ring, ...), array or callable
        mask(P_x, P_y); multiplied with the circular pupil.

    Returns
    -------
    P : 2d array
        Complex pupil function.
    """

    if NA > n:
        raise ValueError("NA = n*sin(alpha) cannot exceed the index n of the medium")
    k0 = 2*np.pi/wavelength
    K_x, K_y = freq_grid(x, y)
    K_r2 = K_x**2 + K_y**2
    P = (K_r2 <= (k0*NA)**2*(1 + 1e-12)).astype(complex)
    P_x, P_y = K_x/(k0*NA), K_y/(k0*NA)
    if defocus:
        k_z = np.sqrt(np.clip((n*k0)**2 - K_r2, 0, None))
        P *= np.exp(1j*(k_z - n*k0)*defocus)
    for q, as_phase in ((phase, True), (mask, False)):
        if q is None:
            continue
        q = q(P_x, P_y) if callable(q) else np.asarray(q)
        P *= np.exp(1j*q) if as_phase else q

    return P


def _check_grid(x, y, wavelength, NA, NA_c):
    k0 = 2*np.pi/wavelength
    if k0*(NA + NA_c) > np.pi/max(get_dx(x), get_dx(y))*(1 + 1e-9):
        warnings.warn("Grid too coarse: the object spectrum up to k0*(NA + NA_condenser) wraps "
                      "around; use dx <= wavelength/(2*(NA + NA_condenser))")


def abbe_image(obj, x, y, wavelength, NA, source=None, P=None, n=1.0, defocus=0.0, chunk=16):
    """Partially coherent image of a thin object (Abbe's method).

    Each source point illuminates the object with a plane wave
    exp(i*(k_sx*x + k_sy*y)), which shifts the object spectrum; the
    objective passes the part inside its pupil, and the coherent images of
    all source points add up in intensity::

        I(r) = sum_s w_s |inv_FT{ P(k) * T(k - k_s) }(r)|**2.

    Parameters
    ----------
    obj : 2d array
        Complex transmission t(x, y) of the object.
    x, y : 1d arrays
        Object coordinates (equidistant, as from utils.grid2d()).
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture of the objective.
    source : tuple (k_x, k_y, w), optional
        Source points from source_points(). Defaults to a single on-axis
        plane wave (coherent illumination).
    P : 2d array, optional
        Pupil function from pupil() (with defocus, aberrations, phase ring,
        ...). Defaults to the clear pupil of NA in the medium n with the given
        defocus.
    n : number, optional
        Index of the medium (for the default pupil). Defaults to 1.
    defocus : number, optional
        Defocus of the default pupil (see pupil()). Defaults to 0.
    chunk : int, optional
        Number of source points imaged per batch of FFTs. Defaults to 16.

    Returns
    -------
    I : 2d array
        Image intensity, normalized such that a clear object (t = 1) whose
        direct light passes the pupil gives I = 1.
    """

    if source is None:
        source = (np.zeros(1), np.zeros(1), np.ones(1))
    k_x, k_y, w = (np.atleast_1d(np.asarray(v, dtype=float)) for v in source)
    if P is None:
        P = pupil(x, y, wavelength, NA, n=n, defocus=defocus)
    _check_grid(x, y, wavelength, NA, np.max(np.hypot(k_x, k_y))*wavelength/(2*np.pi))

    T = FT(np.asarray(obj, dtype=complex))
    dkx, dky = _dk(x, y)
    s_x = np.round(k_x/dkx).astype(int)
    s_y = np.round(k_y/dky).astype(int)
    I = np.zeros(T.shape)
    for start in range(0, len(w), chunk):
        sl = slice(start, start + chunk)
        stack = np.stack([np.roll(T, (sy, sx), axis=(0, 1)) for sx, sy in zip(s_x[sl], s_y[sl])])
        E = _inv_FT2(stack*P)
        I += np.tensordot(w[sl], np.abs(E)**2, axes=1)

    return I


def weak_object_transfer(x, y, wavelength, NA, source=None, P=None, n=1.0, defocus=0.0):
    """Amplitude and phase transfer functions for weak objects.

    For a weak object t = 1 + a + i*phi (``|a|, |phi|`` << 1) the image is linear
    in a and phi::

        I = 1 + inv_FT{ H_a * FT(a) + H_phi * FT(phi) }

    with C(f) = sum_s w_s conj(P(k_s)) P(k_s + f) / sum_s w_s |P(k_s)|**2,
    H_a(f) = C(f) + conj(C(-f)) and H_phi(f) = i*(C(f) - conj(C(-f))).
    In focus with a symmetric condenser and pupil H_phi vanishes: bright
    field shows no phase. Defocus, oblique illumination, a phase ring or a
    DIC prism make H_phi nonzero.

    Parameters
    ----------
    x, y, wavelength, NA, source, P, n, defocus
        As in abbe_image().

    Returns
    -------
    H_a, H_phi : 2d arrays
        Transfer functions on the centered frequency grid of x, y (H_a(0) = 2
        for a clear pupil). Both are real for real symmetric systems; in
        general they are Hermitian (H(-f) = conj(H(f))).
    """

    if source is None:
        source = (np.zeros(1), np.zeros(1), np.ones(1))
    k_x, k_y, w = (np.atleast_1d(np.asarray(v, dtype=float)) for v in source)
    if P is None:
        P = pupil(x, y, wavelength, NA, n=n, defocus=defocus)
    dkx, dky = _dk(x, y)
    N_y, N_x = P.shape
    s_x = np.round(k_x/dkx).astype(int)
    s_y = np.round(k_y/dky).astype(int)
    C = np.zeros(P.shape, dtype=complex)
    norm = 0.0
    for sx, sy, ws in zip(s_x, s_y, w):
        P_s = P[N_y//2 + sy, N_x//2 + sx]
        C += ws*np.conj(P_s)*np.roll(P, (-sy, -sx), axis=(0, 1))
        norm += ws*abs(P_s)**2
    if norm == 0:
        raise ValueError("No direct light passes the pupil (dark field): the weak object "
                         "transfer functions are zero")
    C /= norm
    C_m = np.conj(_flip(C))

    return C + C_m, 1j*(C - C_m)


def _flip(A):
    """A(-f) on the centered frequency grid (even sizes: index 0 maps to itself)."""
    B = A[::-1, ::-1]
    shift = [1 if s % 2 == 0 else 0 for s in A.shape]
    return np.roll(B, shift, axis=(0, 1))


def grating_cutoff(wavelength, NA, NA_condenser=0.0):
    """Smallest grating period that the microscope still images.

    Abbe: at least two diffraction orders must pass the pupil. With oblique
    illumination at NA_condenser the zero and first order fit in for periods
    down to wavelength/(NA + NA_condenser); for on-axis coherent
    illumination the limit is wavelength/NA.

    Parameters
    ----------
    wavelength : number
        Vacuum wavelength.
    NA : number
        Objective numerical aperture.
    NA_condenser : number, optional
        Largest illumination NA. Defaults to 0.

    Returns
    -------
    period : number
    """

    return wavelength/(NA + NA_condenser)


def phase_ring(x, y, wavelength, NA_inner, NA_outer, phase=-np.pi/2, transmission=0.25):
    """Zernike phase plate as a pupil mask.

    The ring sits where the image of the condenser annulus falls in the
    objective pupil, so the direct (undiffracted) light passes it while most
    of the diffracted light misses it. It shifts the phase of the direct
    light by `phase` and attenuates its intensity to `transmission`. For a
    weak phase object t = 1 + i*phi whose diffracted light misses the ring,
    I = T - 2*sqrt(T)*sin(-phase)*phi: with phase = -pi/2 (the ring advances
    the direct light by a quarter wave) optically denser structures appear
    dark (positive phase contrast); +pi/2 gives negative (bright) contrast.

    Parameters
    ----------
    x, y : 1d arrays
        Object coordinates.
    wavelength : number
        Vacuum wavelength.
    NA_inner, NA_outer : numbers
        Inner and outer edge of the ring in the pupil, as numerical apertures.
    phase : number, optional
        Phase added to the light that passes the ring (radians). Defaults to
        -pi/2 (positive phase contrast).
    transmission : number, optional
        Intensity transmission of the ring. Defaults to 0.25.

    Returns
    -------
    mask : 2d array
        Complex pupil mask, 1 outside the ring; pass it to pupil(mask=...).
    """

    k0 = 2*np.pi/wavelength
    K_x, K_y = freq_grid(x, y)
    rho = np.hypot(K_x, K_y)/k0
    tol = 1e-9
    ring = (rho >= NA_inner - tol) & (rho <= NA_outer + tol)
    mask = np.ones(K_x.shape, dtype=complex)
    mask[ring] = np.sqrt(transmission)*np.exp(1j*phase)

    return mask


def dic_object(obj, x, y, shear, bias, angle=0.0):
    """Effective object of Nomarski differential interference contrast.

    A Wollaston (Nomarski) prism splits the light into two orthogonal
    polarizations that pass the object shifted by the shear s against each
    other; a second prism recombines them, and the analyzer (crossed with
    the polarizer) makes them interfere with the bias phase beta between
    them. With a matching condenser prism every source point sees the same
    bias, and the image is the partially coherent image of the effective
    object::

        t_DIC(r) = (t(r + s/2)*exp(i*beta/2) - t(r - s/2)*exp(-i*beta/2))/2.

    A clear field appears with the intensity sin(beta/2)**2; for a phase
    object and small shear, I ~ sin(beta/2)**2 + sin(beta)/2 * s.grad(phi):
    the image shows the phase gradient along the shear (relief look), and
    beta = 0 (extinction) shows only its square.

    Parameters
    ----------
    obj : 2d array
        Complex object transmission t(x, y).
    x, y : 1d arrays
        Object coordinates.
    shear : number
        Shear distance s (object space), typically below the resolution.
    bias : number
        Bias retardation beta in radians.
    angle : number, optional
        Direction of the shear against the x axis, radians. Defaults to 0.

    Returns
    -------
    t_dic : 2d array
        Effective complex object for abbe_image().
    """

    s_x, s_y = shear*np.cos(angle), shear*np.sin(angle)
    t1 = _shift(obj, x, y, -s_x/2, -s_y/2)    # t(r + s/2)
    t2 = _shift(obj, x, y, s_x/2, s_y/2)      # t(r - s/2)

    return (t1*np.exp(0.5j*bias) - t2*np.exp(-0.5j*bias))/2


def airy_unit(wavelength, NA):
    """Diameter of the Airy disk (first dark ring), 1.22*wavelength/NA.

    Confocal pinholes are given in Airy units (AU), as back projected into
    object space.

    Parameters
    ----------
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture.

    Returns
    -------
    d : number
        1 AU in object space.
    """

    return 1.21967*wavelength/NA


def psf_3d(x, y, z, wavelength, NA, n=1.0, apodization=None):
    """Scalar 3D amplitude point spread function of an objective.

    The focal field is the angular spectrum of the pupil, propagated to each
    plane z with the exact k_z in the medium (Debye integral)::

        h(x, y, z) = inv_FT{ P(k) * A(k) * exp(i*(k_z - n*k0)*z) }.

    Parameters
    ----------
    x, y : 1d arrays
        Lateral coordinates (object space).
    z : number or 1d array
        Axial positions relative to focus, in the medium.
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture n*sin(alpha).
    n : number, optional
        Immersion index. Defaults to 1.
    apodization : {None, "aplanatic"}, optional
        None: uniform pupil in k_x, k_y (the paraxial limit). "aplanatic": an
        objective obeying the sine condition, A = (n*k0/k_z)**(1/2) (the
        plane wave amplitudes sqrt(cos(theta)) of Richards and Wolf times the
        Jacobian 1/cos(theta) of the angular spectrum integral). Defaults to
        None.

    Returns
    -------
    h : array
        Complex amplitude PSF, shape (len(z), Ny, Nx) (or (Ny, Nx) for a
        scalar z), normalized such that ``|h|**2`` = 1 in the focus.
    """

    scalar = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))
    k0 = 2*np.pi/wavelength
    P = pupil(x, y, wavelength, NA, n=n)
    K_x, K_y = freq_grid(x, y)
    k_z = np.sqrt(np.clip((n*k0)**2 - K_x**2 - K_y**2, 0, None))
    if apodization == "aplanatic":
        P = P*np.sqrt(n*k0/np.where(k_z > 0, k_z, n*k0))
    elif apodization is not None:
        raise ValueError("apodization must be None or 'aplanatic'")
    h = _inv_FT2(P[None]*np.exp(1j*(k_z - n*k0)[None]*z[:, None, None]))
    h *= P.size/np.sum(np.abs(P))

    return h[0] if scalar else h


def confocal_psf(h_exc, h_det, x, y, pinhole):
    """Intensity PSF of a fluorescence confocal microscope.

    A fluorophore at r is excited with ``|h_exc(r)|**2``; its light, imaged
    back onto the pinhole D, is detected with the efficiency
    ``(|h_det|**2 * D)(r)`` (convolution, lateral only). The fluorescence is
    incoherent, so the confocal PSF is the product::

        PSF(r) = |h_exc(r)|**2 * ( |h_det|**2 (x) D )(r).

    For a point pinhole it is ``|h_exc|**2 |h_det|**2`` (about 1.4 times
    narrower than the wide field PSF), for a large pinhole it approaches
    ``|h_exc|**2`` (wide field resolution, no sectioning).

    Parameters
    ----------
    h_exc : array
        Excitation amplitude PSF, (Ny, Nx) or (Nz, Ny, Nx) (psf_3d()).
    h_det : array
        Detection amplitude PSF at the emission wavelength, same shape.
    x, y : 1d arrays
        Lateral coordinates.
    pinhole : number or 2d array or None
        Pinhole diameter in object space (e.g. in units of airy_unit()), or
        a 2d detector mask (Ny, Nx), or None or 0 for a point pinhole.

    Returns
    -------
    psf : array
        Confocal intensity PSF, shape of h_exc. Normalized such that its
        peak equals 1 for a point pinhole (``|h|**2`` = 1 in focus); a pinhole
        of area A collects up to A/(dx*dy) times more.
    """

    I_exc = np.abs(h_exc)**2
    I_det = np.abs(h_det)**2
    if pinhole is None or (np.ndim(pinhole) == 0 and pinhole == 0):
        return I_exc*I_det
    if np.ndim(pinhole) == 0:
        X, Y = np.meshgrid(x, y)
        D = (np.hypot(X - x[len(x)//2], Y - y[len(y)//2]) <= pinhole/2).astype(float)
        if not D.any():
            D[len(y)//2, len(x)//2] = 1.0
    else:
        D = np.asarray(pinhole, dtype=float)
    # D is centred on the grid centre, as the PSFs: convolution via the centered FT
    detected = np.real(_inv_FT2(_FT2(I_det)*FT(D)))

    return I_exc*detected


def plane_response(psf_stack):
    """Signal of a thin, uniform fluorescent plane vs. its axial position.

    The integral of the intensity PSF over each plane. A wide field
    microscope collects the same light from every plane (no optical
    sectioning); a confocal one falls off as 1/z**2 far from focus.

    Parameters
    ----------
    psf_stack : array
        Intensity PSF, shape (Nz, Ny, Nx).

    Returns
    -------
    s : 1d array
        Normalized to its maximum.
    """

    s = np.sum(psf_stack, axis=(-2, -1))
    return s/np.max(s)


def mirror_response(z, wavelength, NA, n=1.0, N_r=2000):
    """Axial response of a reflection confocal microscope to a plane mirror.

    A mirror at the distance z from focus returns the light with the defocus
    2z; with a point pinhole the detected amplitude is the pupil average of
    exp(2i*(k_z - n*k0)*z) (uniform pupil in k_x, k_y). In the paraxial
    limit the intensity is sinc(u/(2*pi))**2 with u = k0*z*NA**2/n
    (numpy's sinc(x) = sin(pi*x)/(pi*x)), the first zero at
    z = n*wavelength/NA**2.

    Parameters
    ----------
    z : number or array
        Mirror position relative to focus, in the medium.
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture.
    n : number, optional
        Immersion index. Defaults to 1.
    N_r : int, optional
        Number of radial samples of the pupil. Defaults to 2000.

    Returns
    -------
    I : array
        Detected intensity, 1 at focus.
    """

    k0 = 2*np.pi/wavelength
    z = np.asarray(z, dtype=float)
    # uniform in k_perp**2: midpoint rule in s = (k_perp/(k0*NA))**2
    s = (np.arange(N_r) + 0.5)/N_r
    k_z = np.sqrt((n*k0)**2 - s*(k0*NA)**2)
    amp = np.mean(np.exp(2j*(k_z - n*k0)*z[..., None]), axis=-1)

    return np.abs(amp)**2


def led_array(n_side, pitch, distance, wavelength, center=(0.0, 0.0), n_x=None):
    """Illumination wave vectors of an LED matrix below the sample.

    Parameters
    ----------
    n_side : int
        Number of LEDs per row (n_side x n_side array, centred under the
        sample for odd n_side).
    pitch : number
        LED spacing.
    distance : number
        Distance between the LED plane and the sample.
    wavelength : number
        Vacuum wavelength.
    center : tuple of numbers, optional
        Lateral position of the field point (x, y) relative to the array
        axis; LEDs at (X, Y) illuminate it from the direction
        (X - x, Y - y, distance). Defaults to (0, 0).
    n_x : int, optional
        Number of LEDs per column if different from n_side.

    Returns
    -------
    k_x, k_y : 1d arrays
        Transverse wave vectors k0*sin(theta) of the LEDs, row by row.
    NA_ill : 1d array
        Illumination NA (sine of the angle to the axis) of each LED.
    """

    n_x = n_side if n_x is None else n_x
    k0 = 2*np.pi/wavelength
    i = (np.arange(n_side) - (n_side - 1)/2)*pitch
    j = (np.arange(n_x) - (n_x - 1)/2)*pitch
    X, Y = np.meshgrid(j, i)
    d_x, d_y = X.ravel() - center[0], Y.ravel() - center[1]
    r = np.sqrt(d_x**2 + d_y**2 + distance**2)
    s_x, s_y = d_x/r, d_y/r

    return k0*s_x, k0*s_y, np.hypot(s_x, s_y)


def _fpm_windows(x, y, m, k_x, k_y):
    """Integer centres of the m x m spectrum windows for the LED wave vectors."""
    dkx, dky = _dk(x, y)
    N_y, N_x = len(y), len(x)
    # the tilt k_j moves the object spectrum by +k_j: the pupil sees T around -k_j
    c_x = N_x//2 - np.round(np.asarray(k_x)/dkx).astype(int)
    c_y = N_y//2 - np.round(np.asarray(k_y)/dky).astype(int)
    if (np.any(c_x - m//2 < 0) or np.any(c_x + m - m//2 > N_x)
            or np.any(c_y - m//2 < 0) or np.any(c_y + m - m//2 > N_y)):
        raise ValueError("The high resolution grid is too coarse for the largest illumination "
                         "angle plus NA: reduce dx")
    return c_x, c_y


def _low_res_pupil(x, y, m, wavelength, NA, n=1.0):
    """Pupil on the m x m low resolution frequency grid (the central crop)."""
    N_y, N_x = len(y), len(x)
    x_lr = (np.arange(m) - m//2)*get_dx(x)*N_x/m
    y_lr = (np.arange(m) - m//2)*get_dx(y)*N_y/m
    return pupil(x_lr, y_lr, wavelength, NA, n=n), x_lr, y_lr


def fpm_images(obj, x, y, wavelength, NA, k_x, k_y, m, P=None):
    """Low resolution images of a thin object under LED illumination.

    The LED with the transverse wave vector k_j shifts the object spectrum by
    k_j; the objective cuts out the disk of radius k0*NA, and the camera
    samples the image with the pitch dx*N/m (m x m pixels)::

        I_j = |inv_FT{ P(k) * T(k - k_j) }|**2  on the coarse grid.

    LED directions are rounded to the frequency grid of the object.

    Parameters
    ----------
    obj : 2d array
        Complex object transmission on the high resolution grid x, y (N x N).
    x, y : 1d arrays
        High resolution coordinates.
    wavelength : number
        Vacuum wavelength.
    NA : number
        Objective numerical aperture.
    k_x, k_y : 1d arrays
        LED wave vectors (from led_array()).
    m : int
        Number of camera pixels per side; the camera pitch is dx*N/m, which
        should fulfil dx*N/m <= wavelength/(4*NA) for unaliased images.
    P : 2d array, optional
        Pupil on the m x m low resolution frequency grid (e.g. aberrated).
        Defaults to the clear pupil.

    Returns
    -------
    images : 3d array
        Intensities, shape (len(k_x), m, m); 1 for a clear object in bright
        field.
    x_lr, y_lr : 1d arrays
        Camera pixel coordinates.
    """

    N_y, N_x = len(y), len(x)
    P_lr, x_lr, y_lr = _low_res_pupil(x, y, m, wavelength, NA)
    if P is not None:
        P_lr = P
    T = FT(np.asarray(obj, dtype=complex))
    c_x, c_y = _fpm_windows(x, y, m, k_x, k_y)
    scale = m*m/(N_x*N_y)
    stack = np.stack([T[cy - m//2:cy - m//2 + m, cx - m//2:cx - m//2 + m]
                      for cx, cy in zip(c_x, c_y)])
    images = np.abs(_inv_FT2(stack*P_lr)*scale)**2

    return images, x_lr, y_lr


def fpm_reconstruct(images, x, y, wavelength, NA, k_x, k_y, n_iter=20, P=None,
                    update_pupil=True, alpha=1.0, beta=1.0, order=None, start=None,
                    callback=None):
    """Fourier ptychographic reconstruction with embedded pupil recovery.

    The high resolution object spectrum is stitched together from the low
    resolution images by alternating projections (Zheng et al. 2013): for
    each LED, the part of the spectrum under the shifted pupil is
    propagated to the camera, its amplitude replaced by the measured one,
    and the change fed back into the spectrum (and the pupil, EPRY after Ou
    et al. 2014)::

        dPsi = FT(sqrt(I_j)*psi/|psi|) - Psi
        T_j += alpha*conj(P)/max|P|**2 * dPsi
        P   += beta*conj(T_j)/max|T_j|**2 * dPsi.

    Parameters
    ----------
    images : 3d array
        Measured intensities (L, m, m) as from fpm_images().
    x, y : 1d arrays
        High resolution coordinates of the reconstruction (N x N), the same
        grid that relates the LED angles to spectrum shifts.
    wavelength : number
        Vacuum wavelength.
    NA : number
        Objective numerical aperture.
    k_x, k_y : 1d arrays
        LED wave vectors (L), rounded to the frequency grid of x, y.
    n_iter : int, optional
        Number of passes over all LEDs. Defaults to 20.
    P : 2d array, optional
        Initial pupil on the m x m grid. Defaults to the clear pupil.
    update_pupil : bool, optional
        Recover the pupil (aberrations) along with the object. Defaults to
        True.
    alpha, beta : numbers, optional
        Step sizes of the object and pupil updates. Defaults to 1.
    order : array of int, optional
        Order in which the LEDs are processed. Defaults to increasing
        illumination angle (bright field first), which converges best.
    start : 2d array, optional
        Initial high resolution object. Defaults to the up-sampled square
        root of the image with the smallest illumination angle (amplitude
        only).
    callback : callable, optional
        Called as callback(iteration, obj, P) after every pass.

    Returns
    -------
    obj : 2d array
        Complex high resolution object (N x N).
    P : 2d array
        Recovered pupil (m x m).
    errors : 1d array
        Normalized amplitude error sum (|psi| - sqrt(I))**2 / sum I per pass.
    """

    images = np.asarray(images, dtype=float)
    L, m, _ = images.shape
    N_y, N_x = len(y), len(x)
    P_support, _, _ = _low_res_pupil(x, y, m, wavelength, NA)
    support = np.abs(P_support) > 0
    P = P_support.copy() if P is None else np.asarray(P, dtype=complex).copy()
    c_x, c_y = _fpm_windows(x, y, m, k_x, k_y)
    scale = m*m/(N_x*N_y)
    amp = np.sqrt(np.clip(images, 0, None))
    if order is None:
        order = np.argsort(np.hypot(k_x, k_y), kind="stable")

    if start is None:
        j0 = order[0]
        T = np.zeros((N_y, N_x), dtype=complex)
        T[N_y//2 - m//2:N_y//2 - m//2 + m, N_x//2 - m//2:N_x//2 - m//2 + m] = FT(amp[j0])/scale
        # the low resolution image of LED j0 belongs to the spectrum centred at -k_j0
        T = np.roll(T, (c_y[j0] - N_y//2, c_x[j0] - N_x//2), axis=(0, 1))
        T = FT(np.abs(inv_FT(T)))
    else:
        T = FT(np.asarray(start, dtype=complex))

    errors = np.zeros(n_iter)
    total = np.sum(images)
    for it in range(n_iter):
        err = 0.0
        for j in order:
            sl = (slice(c_y[j] - m//2, c_y[j] - m//2 + m), slice(c_x[j] - m//2, c_x[j] - m//2 + m))
            T_j = T[sl]
            Psi = T_j*P
            psi = inv_FT(Psi)*scale
            mag = np.abs(psi)
            err += np.sum((mag - amp[j])**2)
            psi_new = amp[j]*np.exp(1j*np.angle(psi))
            dPsi = FT(psi_new)/scale - Psi
            T[sl] = T_j + alpha*np.conj(P)/np.max(np.abs(P))**2*dPsi
            if update_pupil:
                P = P + beta*np.conj(T_j)/np.max(np.abs(T_j))**2*dPsi
                P[~support] = 0.0
        errors[it] = err/total
        if callback is not None:
            callback(it, inv_FT(T), P)

    return inv_FT(T), P, errors


def circle_overlap(d, R):
    """Overlap area of two circles of radius R at distance d, relative to
    the area of one circle.

    In Fourier ptychography the pupils of neighbouring LEDs must overlap in
    k-space for the phase retrieval to converge: about 40 % at least, 60 %
    is a common design value.

    Parameters
    ----------
    d : number or array
        Distance of the centres (e.g. the difference of the LED wave vectors).
    R : number
        Radius (k0*NA).

    Returns
    -------
    f : number or array
        Fraction between 0 and 1.
    """

    q = np.clip(np.asarray(d, dtype=float)/(2*R), 0, 1)
    return 2/np.pi*(np.arccos(q) - q*np.sqrt(1 - q**2))
