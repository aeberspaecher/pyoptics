#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Speckle: statistics of laser light scattered by rough surfaces and the
ways to reduce its contrast.

A rough surface (height errors larger than the wavelength) gives every
point of the illuminated area a random phase. An imaging system of
numerical aperture NA sums the contributions within one resolution cell,
and the random sum is a circular complex Gaussian field: fully developed
speckle, with an exponential intensity distribution, contrast
C = std(I)/mean(I) = 1 and a grain about lambda/NA across (Goodman).

The contrast drops only when the detector adds intensities of patterns that
are, at least partly, uncorrelated. For M patterns of equal mean intensity
with the field correlation coefficients mu_ij,

    C**2 = sum_ij abs(mu_ij)**2 / M**2 = sum lambda_n**2/(sum lambda_n)**2,

with the eigenvalues lambda_n of the matrix mu (`summed_contrast()`); M
independent patterns give C = 1/sqrt(M) and a gamma distribution of the
intensity (`speckle_pdf()`). The patterns can come from

- polarization: a depolarizing surface makes two independent patterns in
  orthogonal polarizations, C = sqrt((1 + P**2)/2) for a degree of
  polarization P (`polarization_contrast()`),
- wavelength: a change of the wavenumber by dk changes the phase of a path
  L by dk L; paths with an rms spread sigma_L (2 sigma_h for a surface of
  rms roughness sigma_h in reflection) decorrelate the pattern as
  abs(mu)**2 = exp(-(dk sigma_L)**2) (`wavelength_correlation()`), and a
  Gaussian spectrum of rms width sigma_lambda gives
  C = (1 + (4 pi sigma_L sigma_lambda/lambda**2)**2)**(-1/4)
  (`spectral_contrast()`),
- angle: illumination from a range of directions (a moving diffuser, many
  laser modes) changes the speckle of the observer; with the illumination
  filling NA_ill and the observer NA_obs, the average of infinitely many
  independent illuminations has
  C**2 = int int abs(h(a))**2 abs(h(b))**2 abs(J(a - b))**2 da db/(int abs(h)**2)**2
  for the amplitude point spread function h of the observer and the
  mutual intensity J of the illumination (`angular_diversity_contrast()`),
  sqrt(1 - 16/(3 pi**2)) NA_obs/NA_ill = 0.678 NA_obs/NA_ill for
  NA_ill >> NA_obs with circular pupils, and M independent illuminations
  (diffuser positions) leave C**2 = C_inf**2 + (1 + C_inf**2)/M
  (`moving_diffuser_contrast()`).

The simulation helpers build rough surfaces with a Gaussian height
distribution and correlation (`rough_surface()`), image fields through a
circular pupil (`coherent_image()`), make random illumination fields of a
given NA (`diffuser_field()`) and measure contrast and the normalized
intensity autocorrelation (`speckle_contrast()`,
`intensity_autocorrelation()`), whose theoretical value for a circular
pupil is 1 + abs(2 J1(v)/v)**2 (`speckle_autocorrelation()`).

Lengths are in any unit (the notebook uses um). Fields are sampled on
square grids of spacing dx with the origin at the centre.

References
----------
J. W. Goodman, "Speckle Phenomena in Optics: Theory and Applications",
2nd edition, SPIE Press (2020).

J. W. Goodman, "Some fundamental properties of speckle", J. Opt. Soc. Am.
66, 1145-1150 (1976).

J. I. Trisnadi, "Speckle contrast reduction in laser projection displays",
Proc. SPIE 4657, 131-137 (2002).

S. Roelandt, Y. Meuret, G. Craggs, G. Verschaffelt, P. Janssens and
H. Thienpont, "Standardized speckle measurement method matched to human
speckle perception in laser projection systems", Opt. Express 20,
8770-8783 (2012).
"""

import numpy as np
from scipy.special import gammaln, j1, xlogy

from . import fft


__all__ = ["rough_surface", "frequency_grid", "coherent_image", "diffuser_field",
           "speckle_contrast", "speckle_pdf", "intensity_autocorrelation",
           "speckle_autocorrelation", "summed_contrast", "polarization_contrast",
           "wavelength_correlation", "spectral_contrast", "angular_diversity_contrast",
           "moving_diffuser_contrast", "ANGULAR_LIMIT"]


ANGULAR_LIMIT = np.sqrt(1 - 16/(3*np.pi**2))   # C_inf NA_ill/NA_obs for NA_ill >> NA_obs


def frequency_grid(n, dx):
    """Spatial frequencies (fx, fy) of an n x n grid of spacing dx, in the
    unshifted FFT order, as 2D arrays."""

    f = np.fft.fftfreq(n, dx)
    return np.meshgrid(f, f)


def rough_surface(n, dx, sigma_h, correlation_length, rng=None):
    """Random height map with a Gaussian height distribution of rms sigma_h
    and a Gaussian autocorrelation exp(-r**2/correlation_length**2).

    White Gaussian noise is filtered in the Fourier domain and scaled to the
    exact rms value.

    Parameters
    ----------
    n : int
        Grid size (n x n samples).
    dx : float
        Sample spacing.
    sigma_h : float
        Rms height.
    correlation_length : float
        1/e length of the height autocorrelation. Zero gives uncorrelated
        heights from sample to sample.
    rng : numpy Generator or int, optional

    Returns
    -------
    h : 2d array
    """

    rng = np.random.default_rng(rng)
    noise = rng.standard_normal((n, n))
    if correlation_length > 0:
        fx, fy = frequency_grid(n, dx)
        # Autocorrelation exp(-r^2/l^2) <-> power spectrum exp(-pi^2 l^2 f^2),
        # so the amplitude filter is exp(-pi^2 l^2 f^2/2).
        filt = np.exp(-0.5*np.pi**2*correlation_length**2*(fx**2 + fy**2))
        noise = np.real(fft.ifft2(fft.fft2(noise)*filt))
    noise -= noise.mean()
    return sigma_h*noise/noise.std()


def coherent_image(field, dx, NA, wavelength, center=(0.0, 0.0)):
    """Coherent image of a field through a circular pupil (magnification
    one, aberration free): low pass filter with the cut-off NA/lambda.

    Parameters
    ----------
    field : 2d array
        Complex field just behind the object.
    dx : float
        Sample spacing.
    NA : float
        Numerical aperture of the imaging system on the object side.
    wavelength : float
    center : (float, float), optional
        Centre of the pupil in direction cosines (an off-axis observer).

    Returns
    -------
    U : 2d complex array
    """

    n = field.shape[0]
    fx, fy = frequency_grid(n, dx)
    pupil = (fx - center[0]/wavelength)**2 + (fy - center[1]/wavelength)**2 <= (NA/wavelength)**2
    return fft.ifft2(fft.fft2(field)*pupil)


def diffuser_field(n, dx, NA, wavelength, rng=None):
    """Random illumination field whose plane waves fill the numerical
    aperture NA uniformly with random phases: the field a diffuser makes
    behind a condenser, a fully developed speckle field of grain
    lambda/NA, normalized to mean intensity one.

    Parameters
    ----------
    n : int
    dx : float
    NA : float
        Numerical aperture of the illumination. NA = 0 gives a plane wave.
    wavelength : float
    rng : numpy Generator or int, optional

    Returns
    -------
    u : 2d complex array
    """

    if NA == 0:
        return np.ones((n, n), complex)
    rng = np.random.default_rng(rng)
    fx, fy = frequency_grid(n, dx)
    disk = fx**2 + fy**2 <= (NA/wavelength)**2
    spectrum = disk*np.exp(2j*np.pi*rng.random((n, n)))
    u = fft.ifft2(spectrum)
    return u/np.sqrt(np.mean(np.abs(u)**2))


def speckle_contrast(I, axis=None):
    """Speckle contrast std(I)/mean(I) over the given axes (all by
    default)."""

    I = np.asarray(I, float)
    return np.std(I, axis=axis)/np.mean(I, axis=axis)


def speckle_pdf(I, mean=1.0, M=1):
    """Probability density of the intensity of the sum of M independent
    fully developed speckle patterns of equal mean (a gamma density of
    shape M); M = 1 is the exponential density exp(-I/mean)/mean.

    Parameters
    ----------
    I : array
        Intensities (>= 0).
    mean : float, optional
        Mean intensity of the sum.
    M : float, optional
        Number of independent patterns (any positive number: 1/C**2 for
        partly correlated sums).
    """

    I = np.asarray(I, float)
    with np.errstate(divide="ignore"):
        logp = M*np.log(M/mean) + xlogy(M - 1, I) - M*I/mean - gammaln(M)
    return np.exp(logp)


def intensity_autocorrelation(I, dx):
    """Normalized intensity autocorrelation <I(x) I(x + r)>/<I>**2 of a
    periodic pattern, azimuthally averaged.

    Parameters
    ----------
    I : 2d array
    dx : float

    Returns
    -------
    r : 1d array
        Radii, in steps of dx.
    g : 1d array
        Autocorrelation, 1 + abs(mu)**2 for fully developed speckle.
    """

    n = I.shape[0]
    F = fft.fft2(I)
    ac = np.real(fft.ifft2(np.abs(F)**2))/I.size/np.mean(I)**2
    ac = np.fft.fftshift(ac)
    c = np.arange(n) - n//2
    X, Y = np.meshgrid(c, c)
    R = np.rint(np.hypot(X, Y)).astype(int)
    counts = np.bincount(R.ravel())
    g = np.bincount(R.ravel(), ac.ravel())/np.maximum(counts, 1)
    r_max = n//2
    return np.arange(r_max)*dx, g[:r_max]


def speckle_autocorrelation(r, NA, wavelength):
    """Theoretical normalized intensity autocorrelation of fully developed
    speckle imaged through a circular pupil, 1 + abs(2 J1(v)/v)**2 with
    v = 2 pi NA r/lambda. Its first zero (the grain radius) lies at
    r = 0.61 lambda/NA."""

    v = 2*np.pi*NA*np.asarray(r, float)/wavelength
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(v == 0, 1.0, 2*j1(v)/np.where(v == 0, 1.0, v))
    return 1 + mu**2


def summed_contrast(mu):
    """Contrast of the sum of M speckle patterns of equal mean intensity
    with the complex field correlation coefficients mu_ij,
    sqrt(sum abs(mu_ij)**2)/M; equal to sqrt(sum lambda**2)/sum lambda with
    the eigenvalues lambda of mu (Goodman).

    Parameters
    ----------
    mu : (M, M) array
        Hermitian matrix with ones on the diagonal. Only abs(mu)**2 matters,
        so the intensity correlation coefficients can be passed through
        sqrt().

    Returns
    -------
    C : float
    """

    mu = np.asarray(mu)
    M = mu.shape[0]
    return np.sqrt(np.sum(np.abs(mu)**2))/M


def polarization_contrast(P):
    """Contrast of the sum of the two polarization components of fully
    developed speckle with the degree of polarization P,
    sqrt((1 + P**2)/2): 1 for polarized, 1/sqrt(2) for depolarized
    light."""

    P = np.asarray(P, float)
    return np.sqrt((1 + P**2)/2)


def wavelength_correlation(delta_lambda, wavelength, path_spread):
    """Intensity correlation coefficient abs(mu)**2 of two speckle patterns
    at wavelengths delta_lambda apart, exp(-(dk sigma_L)**2) with
    dk = 2 pi delta_lambda/lambda**2, for Gaussian distributed paths of rms
    spread sigma_L (2 sigma_h for a rough surface in reflection at normal
    incidence, deep roughness)."""

    dk = 2*np.pi*np.asarray(delta_lambda, float)/wavelength**2
    return np.exp(-(dk*path_spread)**2)


def spectral_contrast(sigma_lambda, wavelength, path_spread, wavelengths=None, spectrum=None):
    """Speckle contrast for a source of finite spectral width.

    For a Gaussian spectrum of rms width sigma_lambda the contrast is
    (1 + (4 pi sigma_L sigma_lambda/lambda**2)**2)**(-1/4). If a sampled
    spectrum is given instead (wavelengths and powers), the double sum
    C**2 = sum_ij S_i S_j abs(mu_ij)**2/(sum S)**2 is evaluated with
    `wavelength_correlation()`, which also covers lines of a multi-line
    laser.

    Parameters
    ----------
    sigma_lambda : float
        Rms spectral width (ignored if a sampled spectrum is given).
    wavelength : float
        Centre wavelength.
    path_spread : float
        Rms spread sigma_L of the path lengths.
    wavelengths, spectrum : 1d arrays, optional
        Sampled spectrum.

    Returns
    -------
    C : float or array
    """

    if wavelengths is None:
        x = 4*np.pi*path_spread*np.asarray(sigma_lambda, float)/wavelength**2
        return (1 + x**2)**-0.25
    lam = np.asarray(wavelengths, float)
    S = np.asarray(spectrum, float)
    rho = wavelength_correlation(lam[:, None] - lam[None, :], wavelength, path_spread)
    return np.sqrt(S @ rho @ S)/np.sum(S)


def _airy_amplitude(r, NA, wavelength):
    v = 2*np.pi*NA*r/wavelength
    safe = np.where(v == 0, 1.0, v)
    return np.where(v == 0, 1.0, 2*j1(safe)/safe)


def angular_diversity_contrast(NA_ill, NA_obs, wavelength, n=1024, dx=None):
    """Speckle contrast left after averaging over all illuminations from a
    diffuser that fills the numerical aperture NA_ill, seen through an
    observer of numerical aperture NA_obs (an object with deep roughness
    and a correlation length below the resolution).

    C**2 = int int abs(h(a))**2 abs(h(b))**2 abs(J(a - b))**2 da db/(int abs(h)**2)**2,
    with the Airy amplitude h of the observer and the mutual intensity
    J = 2 J1(v)/v, v = 2 pi NA_ill r/lambda, of the illumination, evaluated
    on a grid with FFTs. It tends to 1 for NA_ill -> 0. For
    NA_ill >> NA_obs it tends to sqrt(kappa) NA_obs/NA_ill, where
    kappa = 1 - 16/(3 pi**2) = 0.460 is the integral of the squared OTF of
    the observer over the area of its pupil (`ANGULAR_LIMIT`): the
    coherence area lambda**2/(pi NA_ill**2) of the illumination over the
    effective area (int abs(h)**2)**2/int abs(h)**4 of the observer's
    point spread function.

    Parameters
    ----------
    NA_ill, NA_obs : float
    wavelength : float
    n : int, optional
        Grid size. The default 1024 is good to about 1 % up to
        NA_ill/NA_obs = 10 (the tails of the observer's point spread
        function wrap around for larger ratios).
    dx : float, optional
        Grid spacing (default: a tenth of the smaller grain).

    Returns
    -------
    C : float
    """

    if NA_ill == 0:
        return 1.0
    if dx is None:
        dx = 0.1*wavelength/max(NA_ill, NA_obs)
    c = (np.arange(n) - n//2)*dx
    X, Y = np.meshgrid(c, c)
    R = np.hypot(X, Y)
    p = _airy_amplitude(R, NA_obs, wavelength)**2           # abs(h)^2
    J2 = _airy_amplitude(R, NA_ill, wavelength)**2          # abs(J)^2
    P = fft.fft2(np.fft.ifftshift(p))
    auto = np.real(fft.ifft2(np.abs(P)**2))                 # autocorrelation of abs(h)^2
    C2 = np.sum(auto*np.fft.ifftshift(J2))/np.sum(p)**2
    return float(np.sqrt(C2))


def moving_diffuser_contrast(M, C_inf):
    """Contrast after averaging M independent diffuser positions,
    sqrt(C_inf**2 + (1 + C_inf**2)/M), for the limit C_inf of
    `angular_diversity_contrast()`.

    One position (M = 1) gives sqrt(1 + 2 C_inf**2) > 1: the speckled
    illumination modulates the speckle of the object (product speckle).
    """

    M = np.asarray(M, float)
    return np.sqrt(C_inf**2 + (1 + C_inf**2)/M)
