#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Super-resolution fluorescence microscopy: structured illumination (SIM),
stimulated emission depletion (STED) and single molecule localization
(PALM/STORM).

All functions work in object space with the conventions of
pyoptics.microscopy and pyoptics.imaging: lateral coordinates x, y on a
centered grid, wave vectors in rad per length unit on the centered
frequency grid (utils.freq_grid(x, y), normal order), and OTFs normalized
to 1 at zero frequency as imaging.get_otf() returns them.

Structured illumination (Gustafsson 2000; Heintzmann and Cremer 1999):

- sim_pattern(): the sinusoidal excitation 1 + m*cos(k_p.r + phase).
- sim_images(): raw wide field images of a fluorescent sample under the
  patterns for all angles and phases.
- sim_bands(): band separation; the three components of the spectrum of
  each angle, OTF(k)*S(k - j*k_p) for j = -1, 0, 1.
- shift_spectrum(): moves a band to its true place in k-space by
  multiplying the image with exp(-i*k.r) (any shift, not only grid steps).
- sim_reconstruct(): generalized Wiener filter over all shifted bands with
  an apodization out to the extended cutoff k_c + k_p.
- wiener_deconvolution(): the wide field counterpart, for a fair comparison.
- otf_support(): the region of k-space a set of shifted OTFs covers.

STED (Hell and Wichmann 1994):

- vortex_focus(): the vectorial (Richards-Wolf) focus of a uniformly filled
  aplanatic objective with a helical phase exp(i*l*phi) and any Jones
  vector, via propagators.vectorial_mansuripur(). For l = 1 the doughnut has
  a true zero only for circular polarization whose spin is parallel to the
  orbital angular momentum (Jones vector (1, i)/sqrt(2) for l = +1).
- sted_psf(): the effective PSF, the excitation PSF times the probability
  exp(-ln(2)*I_STED/I_s) that a fluorophore survives the depletion.
- sted_fwhm(): the square root law d = lambda/(2*NA*sqrt(1 + I/I_s)).

Single molecule localization (Thompson et al. 2002; Mortensen et al.
2010):

- gaussian_spot(): expected photon counts of a pixelated Gaussian spot on a
  background.
- localize(): estimates the position of one emitter: centroid, least
  squares or Poisson maximum likelihood fit.
- localization_precision(): Mortensen's formulas for the least squares and
  the maximum likelihood precision with pixels and background; without
  both it reduces to sigma/sqrt(N).

Conventions: the saturation intensity I_s is the depletion intensity that
halves the fluorescence (exp(-ln 2) = 1/2); the saturation factor
zeta = I_max/I_s refers to the peak intensity of the depletion beam.
Background b is given in photons per pixel.

References
----------
M. G. L. Gustafsson, "Surpassing the lateral resolution limit by a factor
of two using structured illumination microscopy", J. Microsc. 198, 82-87
(2000).

R. Heintzmann and C. Cremer, "Laterally modulated excitation microscopy:
improvement of resolution by using a diffraction grating", Proc. SPIE 3568,
185-196 (1999).

S. W. Hell and J. Wichmann, "Breaking the diffraction resolution limit by
stimulated emission: stimulated-emission-depletion fluorescence
microscopy", Opt. Lett. 19, 780-782 (1994).

B. Harke, J. Keller, C. K. Ullal, V. Westphal, A. Schönle and S. W. Hell,
"Resolution scaling in STED microscopy", Opt. Express 16, 4154-4162 (2008).

R. E. Thompson, D. R. Larson and W. W. Webb, "Precise nanometer
localization analysis for individual fluorescent probes", Biophys. J. 82,
2775-2783 (2002).

K. I. Mortensen, L. S. Churchman, J. A. Spudich and H. Flyvbjerg,
"Optimized localization analysis for single-molecule tracking and
super-resolution microscopy", Nat. Methods 7, 377-381 (2010).
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize
from scipy.special import erf

from .fft import FT, inv_FT
from .utils import freq_grid


__all__ = ["sim_pattern", "sim_images", "sim_bands", "shift_spectrum", "sim_reconstruct",
           "wiener_deconvolution", "otf_support", "ideal_otf", "vortex_focus", "sted_psf",
           "sted_fwhm", "gaussian_spot", "localize", "localization_precision"]


# ---------------------------------------------------------------------------
# structured illumination

def _direction(angle):
    return np.cos(angle), np.sin(angle)


def sim_pattern(x, y, k_p, angle, phase, modulation=1.0):
    """Sinusoidal SIM excitation 1 + m*cos(k_p*(x*cos(angle) + y*sin(angle)) + phase).

    Two coherent beams in the pupil at +-k_p/2 (s polarized, equal
    intensity) give m = 1.

    Parameters
    ----------
    x, y : 1d arrays
        Lateral coordinates.
    k_p : number
        Pattern wave number 2*pi/period (rad per length unit).
    angle : number
        Direction of the pattern's wave vector (rad).
    phase : number
        Phase of the pattern (rad).
    modulation : number, optional
        Modulation depth m, 0 <= m <= 1. Defaults to 1.

    Returns
    -------
    I : array
        Excitation intensity, shape (Ny, Nx), mean 1.
    """

    X, Y = np.meshgrid(x, y)
    c, s = _direction(angle)
    return 1.0 + modulation*np.cos(k_p*(c*X + s*Y) + phase)


def _convolve(a, otf):
    return np.real(inv_FT(FT(a)*otf))


def sim_images(sample, x, y, otf, k_p, angles, phases, modulation=1.0):
    """Raw SIM images: the sample times each pattern, imaged incoherently.

    Parameters
    ----------
    sample : array
        Fluorophore density, shape (Ny, Nx).
    x, y : 1d arrays
        Lateral coordinates.
    otf : array
        Detection OTF on the centered frequency grid (imaging.get_otf()).
    k_p : number
        Pattern wave number.
    angles, phases : sequences
        Pattern directions and phases (rad).
    modulation : number, optional
        Modulation depth. Defaults to 1.

    Returns
    -------
    images : array
        Shape (len(angles), len(phases), Ny, Nx).
    """

    return np.array([[_convolve(sample*sim_pattern(x, y, k_p, a, p, modulation), otf)
                      for p in phases] for a in angles])


def sim_bands(images, phases, modulation=1.0):
    """Separates the three bands of SIM images by their phase dependence.

    For each angle the spectrum of the raw image with pattern phase phi_n is::

        D_n(k) = OTF(k)*[S(k) + m/2*exp(i*phi_n)*S(k - k_p) + m/2*exp(-i*phi_n)*S(k + k_p)],

    a linear system for the three bands, solved (in the least squares sense
    for more than three phases).

    Parameters
    ----------
    images : array
        Raw images, shape (n_angles, n_phases, Ny, Nx) (sim_images()).
    phases : sequence
        The pattern phases (at least three distinct values modulo 2*pi).
    modulation : number, optional
        Modulation depth m. Defaults to 1.

    Returns
    -------
    bands : array
        Complex spectra of shape (n_angles, 3, Ny, Nx), ordered j = -1, 0,
        +1: bands[:, j+1] = OTF(k)*S(k - j*k_p) on the centered frequency
        grid.
    """

    images = np.asarray(images, dtype=float)
    phases = np.asarray(phases, dtype=float)
    orders = np.array([-1, 0, 1])
    weight = np.where(orders == 0, 1.0, modulation/2)
    M = weight[None, :]*np.exp(1j*orders[None, :]*phases[:, None])
    M_inv = np.linalg.pinv(M)
    spectra = np.array([[FT(img) for img in per_angle] for per_angle in images])
    return np.einsum("jn,anyx->ajyx", M_inv, spectra)


def shift_spectrum(spectrum, x, y, k_x, k_y):
    """Spectrum shifted by -k: returns F(k + k_shift) for the spectrum F(k).

    The shift is done by multiplying the field in real space with
    exp(-i*(k_x*x + k_y*y)), so k need not lie on the frequency grid. The
    field must decay towards the window edges (it is treated as periodic).

    Parameters
    ----------
    spectrum : array
        Spectrum on the centered frequency grid of x, y.
    x, y : 1d arrays
        Lateral coordinates.
    k_x, k_y : numbers
        The shift (rad per length unit).

    Returns
    -------
    shifted : array
        F(k_x' + k_x, k_y' + k_y) on the same grid.
    """

    X, Y = np.meshgrid(x, y)
    return FT(inv_FT(spectrum)*np.exp(-1j*(k_x*X + k_y*Y)))


def ideal_otf(x, y, k_c):
    """Incoherent OTF of a diffraction limited circular pupil with cutoff k_c,
    (2/pi)*(arccos(q) - q*sqrt(1 - q**2)), q = ``|k|``/k_c.

    Parameters
    ----------
    x, y : 1d arrays
        Lateral coordinates (only their frequency grid is used).
    k_c : number
        Cutoff wave number (2*k0*NA for a microscope).

    Returns
    -------
    otf : array
        On the centered frequency grid, 1 at k = 0, zero beyond k_c.
    """

    K_x, K_y = freq_grid(x, y)
    q = np.clip(np.hypot(K_x, K_y)/k_c, 0, 1)
    return 2/np.pi*(np.arccos(q) - q*np.sqrt(1 - q**2))


def _cutoff(otf, x, y, threshold=1e-6):
    K_x, K_y = freq_grid(x, y)
    return np.max(np.hypot(K_x, K_y)[np.abs(otf) > threshold])


def _shifted_otfs(otf, x, y, k_p, angles):
    out = []
    for a in angles:
        c, s = _direction(a)
        out.append([shift_spectrum(otf, x, y, j*k_p*c, j*k_p*s) for j in (-1, 0, 1)])
    return np.array(out)


def otf_support(otf, x, y, k_p, angles, threshold=1e-3):
    """Region of k-space covered by the OTF and its copies shifted by
    +-k_p in each pattern direction.

    Parameters
    ----------
    otf : array
        Detection OTF on the centered frequency grid.
    x, y : 1d arrays
        Lateral coordinates.
    k_p : number
        Pattern wave number (0 for wide field).
    angles : sequence
        Pattern directions (rad).
    threshold : number, optional
        Relative level of ``|OTF|`` that counts as transmitted. Defaults to 1e-3.

    Returns
    -------
    support : bool array
        Shape (Ny, Nx).
    """

    H = np.abs(_shifted_otfs(otf, x, y, k_p, angles))
    return np.any(H > threshold, axis=(0, 1))


def _apodization(apodization, x, y, k_max):
    if apodization is None:
        return ideal_otf(x, y, k_max)
    if apodization is False:
        return 1.0
    return np.asarray(apodization)


def sim_reconstruct(images, x, y, otf, k_p, angles, phases, modulation=1.0, wiener=0.05,
                    apodization=None):
    """SIM reconstruction by a generalized Wiener filter.

    The bands of each angle (sim_bands()) are moved to their true place,
    B_aj(k + j*k_p) = OTF(k + j*k_p)*S(k), and combined as::

        S_est(k) = A(k)*sum OTF*(k + j*k_p) B_aj(k + j*k_p) / (sum |OTF(k + j*k_p)|**2 + w**2)

    over all angles a and orders j. The apodization A tapers the result
    towards the extended cutoff k_c + k_p (k_c: cutoff of the OTF) and
    suppresses ringing.

    Parameters
    ----------
    images : array
        Raw images (n_angles, n_phases, Ny, Nx).
    x, y : 1d arrays
        Lateral coordinates.
    otf : array
        Detection OTF on the centered frequency grid.
    k_p : number
        Pattern wave number.
    angles, phases : sequences
        Pattern directions and phases (rad), as used for the images.
    modulation : number, optional
        Modulation depth. Defaults to 1.
    wiener : number, optional
        Wiener parameter w (relative to OTF(0) = 1). Defaults to 0.05.
    apodization : None, False or array, optional
        None (default): ideal_otf() with the cutoff k_c + k_p. False: no
        apodization. An array: the apodization on the frequency grid.

    Returns
    -------
    image : array
        Reconstructed image, shape (Ny, Nx).
    spectrum : array
        Its spectrum (before the inverse FT).
    """

    bands = sim_bands(images, phases, modulation)
    H = _shifted_otfs(otf, x, y, k_p, angles)
    num = np.zeros(otf.shape, dtype=complex)
    den = np.zeros(otf.shape)
    for a, angle in enumerate(angles):
        c, s = _direction(angle)
        for i, j in enumerate((-1, 0, 1)):
            B = shift_spectrum(bands[a, i], x, y, j*k_p*c, j*k_p*s)
            num += np.conj(H[a, i])*B
            den += np.abs(H[a, i])**2
    k_max = _cutoff(otf, x, y) + k_p
    spectrum = _apodization(apodization, x, y, k_max)*num/(den + wiener**2)
    return np.real(inv_FT(spectrum)), spectrum


def wiener_deconvolution(image, otf, x, y, wiener=0.05, apodization=None):
    """Wiener deconvolution of a wide field image, the same filter as
    sim_reconstruct() with only the unshifted band.

    Parameters
    ----------
    image : array
        Wide field image (Ny, Nx).
    otf : array
        Its OTF on the centered frequency grid.
    x, y : 1d arrays
        Lateral coordinates.
    wiener : number, optional
        Wiener parameter. Defaults to 0.05.
    apodization : None, False or array, optional
        None (default): ideal_otf() with the cutoff of the OTF. False: none.

    Returns
    -------
    image : array
        Deconvolved image.
    """

    A = _apodization(apodization, x, y, _cutoff(otf, x, y))
    return np.real(inv_FT(A*np.conj(otf)*FT(image)/(np.abs(otf)**2 + wiener**2)))


# ---------------------------------------------------------------------------
# STED

def vortex_focus(x, y, wavelength, NA, n=1.0, charge=0, jones=(1, 1j), z=0.0):
    """Vectorial focus of an aplanatic objective with a uniformly filled
    pupil, a helical phase exp(i*charge*phi) and the Jones vector jones.

    The pupil field is turned into the focal field with
    propagators.vectorial_mansuripur() (Richards-Wolf theory).

    For charge = +1 and the circular polarization (1, i)/sqrt(2), whose spin
    is parallel to the orbital angular momentum, all three field components
    vanish on the axis: the transverse components carry exp(i*phi) and
    exp(3i*phi), the longitudinal one exp(2i*phi). With the opposite spin the
    longitudinal component has no azimuthal dependence and fills the
    centre of the doughnut.

    Parameters
    ----------
    x, y : 1d arrays
        Lateral coordinates of the focal plane (centered grid); the spacing
        must resolve the medium wavelength, dx <= wavelength/(2*n).
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture, NA < n.
    n : number, optional
        Refractive index of the immersion medium. Defaults to 1.
    charge : int, optional
        Topological charge of the helical phase plate. Defaults to 0.
    jones : sequence of 2 complex, optional
        Jones vector of the pupil field (normalized here). Defaults to
        (1, 1j).
    z : number, optional
        Distance from the focal plane. Defaults to 0.

    Returns
    -------
    E : array
        Complex field (Ny, Nx, 3) with components E_x, E_y, E_z.
    """

    from .propagators import vectorial_mansuripur

    k = 2*np.pi*n/wavelength
    K_x, K_y = freq_grid(x, y)
    P = (np.hypot(K_x, K_y) < k*NA/n)*np.exp(1j*charge*np.arctan2(K_y, K_x))
    jones = np.asarray(jones, dtype=complex)
    jones = jones/np.linalg.norm(jones)
    field = np.stack([inv_FT(jones[0]*P), inv_FT(jones[1]*P)], -1)
    return vectorial_mansuripur(field, x, y, z, wavelength, n=n)[0]


def sted_psf(I_exc, I_sted, zeta, I_det=None):
    """Effective STED PSF.

    A fluorophore excited with I_exc survives the depletion pulse with the
    probability exp(-ln(2)*I_STED/I_s); the effective PSF is::

        h_eff = I_exc*exp(-ln(2)*zeta*I_sted/max(I_sted)) [*I_det],

    with the saturation factor zeta = max(I_STED)/I_s.

    Parameters
    ----------
    I_exc, I_sted : arrays
        Excitation and depletion intensities on the same grid.
    zeta : number
        Saturation factor (peak depletion intensity over I_s).
    I_det : array, optional
        Detection efficiency (e.g. the pinhole convolved detection PSF).

    Returns
    -------
    h : array
        Effective PSF, unnormalized.
    """

    h = I_exc*np.exp(-np.log(2)*zeta*I_sted/np.max(I_sted))
    return h if I_det is None else h*I_det


def sted_fwhm(wavelength, NA, zeta, d0=None):
    """The STED square root law d = lambda/(2*NA*sqrt(1 + zeta)).

    Parameters
    ----------
    wavelength : number
        Wavelength.
    NA : number
        Numerical aperture.
    zeta : number or array
        Saturation factor I_max/I_s.
    d0 : number, optional
        Diffraction limited FWHM to use instead of wavelength/(2*NA).

    Returns
    -------
    d : number or array
        Lateral FWHM.
    """

    d0 = wavelength/(2*NA) if d0 is None else d0
    return d0/np.sqrt(1 + np.asarray(zeta, dtype=float))


# ---------------------------------------------------------------------------
# single molecule localization

def gaussian_spot(x, y, x0, y0, sigma, n_photons, background=0.0):
    """Expected photon counts of a Gaussian spot integrated over square pixels.

    Parameters
    ----------
    x, y : 1d arrays
        Pixel centres (equidistant).
    x0, y0 : numbers
        Emitter position.
    sigma : number
        Standard deviation of the PSF (per axis).
    n_photons : number
        Total number of photons of the spot (on an infinite detector).
    background : number, optional
        Background photons per pixel. Defaults to 0.

    Returns
    -------
    counts : array
        Expected counts per pixel, shape (len(y), len(x)).
    """

    a = x[1] - x[0]
    s = np.sqrt(2)*sigma
    ex = 0.5*(erf((x + a/2 - x0)/s) - erf((x - a/2 - x0)/s))
    ey = 0.5*(erf((y + a/2 - y0)/s) - erf((y - a/2 - y0)/s))
    return n_photons*np.outer(ey, ex) + background


def localize(image, x, y, sigma, method="mle"):
    """Position of a single emitter in a small image.

    Parameters
    ----------
    image : array
        Photon counts, shape (len(y), len(x)).
    x, y : 1d arrays
        Pixel centres.
    sigma : number
        PSF standard deviation used by the fits.
    method : {"mle", "lsq", "centroid"}, optional
        Poisson maximum likelihood fit (default), unweighted least squares
        fit of gaussian_spot() with position, photon number and background
        as parameters, or the intensity centroid.

    Returns
    -------
    x0, y0 : numbers
        Estimated position.
    """

    image = np.asarray(image, dtype=float)
    total = image.sum()
    xc = np.sum(image.sum(axis=0)*x)/total
    yc = np.sum(image.sum(axis=1)*y)/total
    if method == "centroid":
        return xc, yc
    b0 = max(np.percentile(image, 20), 0.1)
    p0 = np.array([xc, yc, max(total - b0*image.size, 1.0), b0])

    def model(p):
        return gaussian_spot(x, y, p[0], p[1], sigma, p[2], p[3])

    if method == "lsq":
        res = least_squares(lambda p: (model(p) - image).ravel(), p0,
                            bounds=([x[0], y[0], 0, 0], [x[-1], y[-1], np.inf, np.inf]))
        return res.x[0], res.x[1]
    if method != "mle":
        raise ValueError("method must be 'mle', 'lsq' or 'centroid'.")

    # fit in scaled parameters (pixels, initial photon number and background)
    a = x[1] - x[0]
    scale = np.array([a, a, p0[2], p0[3]])

    def nll(q):
        mu = np.maximum(model(q*scale), 1e-12)
        return np.sum(mu - image*np.log(mu))

    lo, hi = np.array([x[0], y[0], 1e-3, 1e-6]), np.array([x[-1], y[-1], np.inf, np.inf])
    res = minimize(nll, p0/scale, method="L-BFGS-B", bounds=list(zip(lo/scale, hi/scale)))
    res.x = res.x*scale
    return res.x[0], res.x[1]


def localization_precision(sigma, n_photons, pixel=0.0, background=0.0, method="mle"):
    """Precision (standard deviation per axis) of localizing a Gaussian spot
    after Mortensen et al. (2010).

    With sigma_a**2 = sigma**2 + pixel**2/12 and the background b (photons
    per pixel),

        least squares:  var = sigma_a**2/N*(16/9 + 8*pi*sigma_a**2*b/(N*a**2)),
        maximum likelihood:  var = sigma_a**2/N/(1 + int_0^1 ln(t)/(1 + N*a**2*t/(2*pi*sigma_a**2*b)) dt).

    Without background the MLE reaches sigma_a/sqrt(N), and sigma/sqrt(N)
    for vanishing pixels.

    Parameters
    ----------
    sigma : number
        PSF standard deviation.
    n_photons : number or array
        Detected photons N.
    pixel : number, optional
        Pixel size a. Defaults to 0.
    background : number, optional
        Background photons per pixel. Defaults to 0.
    method : {"mle", "lsq"}, optional
        Defaults to "mle".

    Returns
    -------
    s : number or array
        Standard deviation per axis.
    """

    N = np.asarray(n_photons, dtype=float)
    sa2 = sigma**2 + pixel**2/12
    if method == "lsq":
        if background > 0 and pixel <= 0:
            raise ValueError("A background needs a pixel size.")
        bg = 8*np.pi*sa2*background/(N*pixel**2) if background > 0 else 0.0
        return np.sqrt(sa2/N*(16/9 + bg))
    if method != "mle":
        raise ValueError("method must be 'mle' or 'lsq'.")
    if background <= 0:
        return np.sqrt(sa2/N)
    if pixel <= 0:
        raise ValueError("A background needs a pixel size.")

    def factor(n):
        tau = 2*np.pi*sa2*background/(n*pixel**2)
        return 1 + quad(lambda t: np.log(t)/(1 + t/tau), 0, 1)[0]

    F = np.vectorize(factor)(N)
    return np.sqrt(sa2/N/F)
