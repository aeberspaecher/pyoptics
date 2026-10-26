#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Simple imaging: coherent imaging, point spread function (PSF), optical
transfer function (OTF), modulation transfer function (MTF) and Strehl ratio.

All functions here describe a scalar imaging system in the paraxial picture.
The pupil is sampled on the frequency grid that belongs to the image plane
grid x, y (utils.freq_grid() with normal_order=True). The pupil edge lies at
``|k_perp|`` = k*NA. Normalized pupil coordinates are (P_x, P_y) = (k_x, k_y)/(k*NA),
so that the pupil is the unit disk P_x**2 + P_y**2 <= 1; Fringe Zernikes with
R_norm = 1 evaluated on P_x, P_y thus describe the usual pupil aberrations.

Pupil masks may be given as arrays sampled on the frequency grid or as
callables mask(P_x, P_y); the mask may be real-valued (apodization) or
complex. Phases (in radians, i.e. 2*pi*W for a wavefront error W in waves)
may be given in the same two ways. With the convention exp(i(k.r - omega*t)),
a positive phase corresponds to a wavefront that runs ahead.

Sampling: the PSF is band limited to ``|k_perp|`` <= k*NA, the intensity PSF and
hence the OTF to 2*k*NA. The grid spacing should thus fulfil
dx <= wavelength/(4*NA) to sample the OTF without aliasing. The pupil is
sampled with spacing 2*pi/L (L: size of the image plane window), i.e. with
about L*NA/wavelength samples across its radius; the larger the window, the
better the pupil (and with it the PSF) is resolved.

get_psf() can evaluate the PSF on a separate output grid x_out, y_out (e.g. a
fine grid around the core or an off-axis window) by a chirp-z transform of
the pupil, without zero padding: x, y then only set the pupil sampling.
"""

import warnings

import numpy as np
from numpy import pi
from math import sqrt
from numpy.lib.scimath import sqrt as cmpl_sqrt
from .fft import FT, inv_FT, zoom_inv_FT
from .utils import freq_grid, frequencies


def image(obj, x, y, NA, k):
    """Coherent image of an object through a 4f system.

    The spectrum of the object is truncated at the cut-off k*NA (no
    aberrations, no obliquity factor).

    Parameters
    ----------
    obj : 2d array
        Object field in the focal plane (may be complex).
    x, y : 1d arrays
        Coordinates of the object.
    NA : number
        Numerical aperture.
    k : number
        Wavenumber.

    Returns
    -------
    image : 2d array
        Complex image field.
    """

    # FT() returns a centered spectrum, so the frequency grid needs to be in
    # normal (centered) order as well:
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)

    obj_spectrum = FT(obj)

    # NA truncation:
    mask = np.ones(np.shape(obj))
    mask[K_x**2 + K_y**2 > k**2*NA**2] = 0.0

    #obj_spectrum *= mask
    obj_spectrum[K_x**2 + K_y**2 > k**2*NA**2] = 0.0

    #obj_spectrum *= cmpl_sqrt(1 - K_x**2/k**2 - K_y**2/k**2)  # is this the obliquity factor rewritten using the sine condition?
    # TODO: cosine factor?

    image = inv_FT(obj_spectrum)

    return image


def _sample_on_pupil(quantity, P_x, P_y):
    """Sample an array or a callable quantity(P_x, P_y) on the pupil grid."""

    if quantity is None:
        return None
    if callable(quantity):
        return quantity(P_x, P_y)

    quantity = np.asarray(quantity)
    if quantity.shape != P_x.shape:
        raise ValueError("Pupil arrays need the shape {} of the frequency grid, got {}"
                         .format(P_x.shape, quantity.shape))

    return quantity


def _check_sampling(x, y, NA, k, band_limit_factor):
    """Warn if band_limit_factor*k*NA exceeds the Nyquist frequency."""

    dx, dy = x[1] - x[0], y[1] - y[0]
    if band_limit_factor*k*NA > np.pi/max(dx, dy):
        warnings.warn("Grid too coarse: band limit {}*k*NA exceeds the Nyquist "
                      "wavenumber pi/dx; reduce dx below wavelength/({}*NA)"
                      .format(band_limit_factor, 2*band_limit_factor))


def pupil_coordinates(x, y, NA, k):
    """Normalized pupil coordinates on the frequency grid of x, y.

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Image-side numerical aperture.
    k : number
        Wavenumber (2*pi/wavelength).

    Returns
    -------
    P_x, P_y : arrays
        (k_x, k_y)/(k*NA) in normal (centered) order, shape (Ny, Nx). The
        pupil is the unit disk.
    """

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=True)

    return K_X/(k*NA), K_Y/(k*NA)


def pupil_function(x, y, NA, k, pupil_mask=None, phase=None):
    """Sampled pupil function P = mask*exp(i*phase) inside the unit disk.

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Image-side numerical aperture.
    k : number
        Wavenumber.
    pupil_mask : array or callable, optional
        Amplitude transmission of the pupil (e.g. a central obscuration or an
        apodization), sampled on the frequency grid or a callable
        pupil_mask(P_x, P_y). It is always multiplied with the circular pupil
        of radius k*NA. Defaults to None (clear circular pupil).
    phase : array or callable, optional
        Pupil phase in radians, as array or callable phase(P_x, P_y). Defaults
        to None (no aberrations).

    Returns
    -------
    P : array
        Complex pupil function on the centered frequency grid, shape (Ny, Nx).
    """

    P_x, P_y = pupil_coordinates(x, y, NA, k)
    P = (P_x**2 + P_y**2 <= 1.0).astype(complex)

    mask = _sample_on_pupil(pupil_mask, P_x, P_y)
    if mask is not None:
        P *= mask

    phi = _sample_on_pupil(phase, P_x, P_y)
    if phi is not None:
        P *= np.exp(1j*phi)

    return P


def get_psf(x, y, NA, k, pupil_mask=None, phase=None, normalization="peak",
            x_out=None, y_out=None):
    """Amplitude point spread function of an imaging system.

    The PSF is the (inverse) Fourier transform of the pupil function, located
    at the center x = y = 0 of the grid (for a centered grid, see fft).

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Image-side numerical aperture.
    k : number
        Wavenumber.
    pupil_mask, phase : arrays or callables, optional
        See pupil_function().
    normalization : {"peak", "energy", None}, optional
        "peak" (default): scale such that the aberration-free PSF of the same
        pupil mask has the amplitude 1 at its center, so that
        ``|psf[Ny//2, Nx//2]|**2`` is the Strehl ratio. "energy": scale such that
        ``sum(|psf|**2)*dx*dy`` = 1 on the grid x, y (with x_out, y_out the
        same scale factor is used, so the PSF energy on the full plane is
        1). None: plain inverse DFT of the pupil.
    x_out, y_out : arrays, optional
        Linear, equidistant output coordinates. If given, the PSF is
        evaluated there by a chirp-z transform (fft.zoom_inv_FT()) of the
        pupil sampled on the frequency grid of x, y: any center, extent and
        sampling, e.g. a fine grid over the Airy disk without padding. The
        PSF is periodic with the window size of x, y. Defaults to None (PSF
        on x, y).

    Returns
    -------
    psf : array
        Complex amplitude PSF, shape (Ny, Nx) or (len(y_out), len(x_out)).
        The intensity PSF is ``|psf|**2``.
    """

    _check_sampling(x, y, NA, k, 1)
    P = pupil_function(x, y, NA, k, pupil_mask, phase)
    if (x_out is None) != (y_out is None):
        raise ValueError("Give both x_out and y_out or none of them")
    if x_out is None:
        psf = inv_FT(P)
    else:
        psf = zoom_inv_FT(P, frequencies(x), frequencies(y), x_out, y_out)

    if normalization == "peak":
        # the aberration-free amplitude at the center is sum(|P|)/(Nx*Ny):
        psf *= np.size(P)/np.sum(np.abs(P))
    elif normalization == "energy":
        # Parseval: sum(|inv_FT(P)|**2) = sum(|P|**2)/(Nx*Ny)
        dx, dy = x[1] - x[0], y[1] - y[0]
        psf /= np.sqrt(np.sum(np.abs(P)**2)/np.size(P)*dx*dy)
    elif normalization is not None:
        raise ValueError("normalization must be one of 'peak', 'energy' or None")

    return psf


def get_otf(x, y, NA, k, pupil_mask=None, phase=None):
    """Optical transfer function of an incoherent imaging system.

    The OTF is the Fourier transform of the intensity PSF, i.e. the
    normalized autocorrelation of the pupil function.

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Image-side numerical aperture.
    k : number
        Wavenumber.
    pupil_mask, phase : arrays or callables, optional
        See pupil_function().

    Returns
    -------
    otf : array
        Complex OTF normalized to OTF(0) = 1 on the centered frequency grid,
        i.e. at the frequencies utils.freq_grid(x, y, wavenumbers=False,
        normal_order=True). It vanishes beyond the cut-off frequency
        2*NA/wavelength.
    """

    _check_sampling(x, y, NA, k, 2)
    psf = get_psf(x, y, NA, k, pupil_mask, phase, normalization=None)
    otf = FT(np.abs(psf)**2)
    Ny, Nx = np.shape(otf)

    return otf/otf[Ny//2, Nx//2]


def get_mtf(x, y, NA, k, pupil_mask=None, phase=None):
    """Modulation transfer function ``|OTF|``.

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Image-side numerical aperture.
    k : number
        Wavenumber.
    pupil_mask, phase : arrays or callables, optional
        See pupil_function().

    Returns
    -------
    mtf : array
        On the frequency grid of `get_otf()`.
    """

    return np.abs(get_otf(x, y, NA, k, pupil_mask, phase))


def rms_phase(x, y, NA, k, pupil_mask=None, phase=None):
    """Root mean square of the pupil phase with the piston removed.

    The mean and the variance are weighted with the pupil amplitude ``|P|``.

    Parameters
    ----------
    Same as pupil_function().

    Returns
    -------
    sigma : number
        RMS phase error in radians (multiply by wavelength/(2*pi) for the RMS
        wavefront error).
    """

    if phase is None:
        return 0.0

    P_x, P_y = pupil_coordinates(x, y, NA, k)
    weights = np.abs(pupil_function(x, y, NA, k, pupil_mask))
    phi = np.broadcast_to(_sample_on_pupil(phase, P_x, P_y), weights.shape)

    mean = np.sum(weights*phi)/np.sum(weights)
    variance = np.sum(weights*(phi - mean)**2)/np.sum(weights)

    return np.sqrt(variance)


def strehl_ratio(x, y, NA, k, pupil_mask=None, phase=None):
    """Strehl ratio of an aberrated pupil.

    Computed as the intensity at the Gaussian image point (the grid center)
    relative to the aberration-free system with the same pupil mask,
    ``|sum(P)|**2/sum(|P|)**2``. This is Born & Wolf's definition; with tilt in
    the phase the PSF maximum moves away from the center and can be larger.

    Parameters
    ----------
    Same as pupil_function().

    Returns
    -------
    S : number
        Strehl ratio in [0, 1].
    """

    P = pupil_function(x, y, NA, k, pupil_mask, phase)

    return np.abs(np.sum(P))**2/np.sum(np.abs(P))**2


def marechal_strehl(sigma, approximation="exponential"):
    """Approximate Strehl ratio from the RMS phase error sigma (radians).

    Parameters
    ----------
    sigma : number or array
        RMS phase error with piston removed, e.g. from rms_phase(), or
        2*pi*W_rms/wavelength for a wavefront error W_rms.
    approximation : {"exponential", "marechal", "quadratic"}, optional
        "exponential" (default): exp(-sigma**2) (Mahajan), good up to
        sigma of about 1 rad for many aberrations. "marechal":
        (1 - sigma**2/2)**2. "quadratic": 1 - sigma**2, the leading order
        valid for small sigma.

    Returns
    -------
    S : number or array
    """

    sigma = np.asarray(sigma)
    if approximation == "exponential":
        return np.exp(-sigma**2)
    elif approximation == "marechal":
        return (1 - sigma**2/2)**2
    elif approximation == "quadratic":
        return 1 - sigma**2
    else:
        raise ValueError("approximation must be one of 'exponential', 'marechal', 'quadratic'")
