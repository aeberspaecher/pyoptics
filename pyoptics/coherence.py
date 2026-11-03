#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Spatial and temporal coherence, stellar interferometry and intensity
correlations.

Conventions: fields vary as exp(-i*omega*t) (see the conventions page). The
complex degree of (temporal) coherence of a stationary field u(t) is ::

    gamma(tau) = <conj(u(t))*u(t + tau)>/<|u|**2>
               = int S(nu)*exp(-2*pi*i*nu*tau) dnu/int S(nu) dnu

(Wiener-Khinchin), and the complex degree of spatial coherence of two points
x1, x2 is mu_12 = <u(x1)*conj(u(x2))>/sqrt(<|u(x1)|**2><|u(x2)|**2>).

Spatial coherence:

- mutual_intensity_vcz(): the van Cittert-Zernike theorem, mu as the
  normalized Fourier transform of the intensity of an incoherent source,
  evaluated numerically with chirp-z transforms (fft.zoom_FT()) for any
  source.
- uniform_disk_visibility(), limb_darkened_disk_visibility(),
  binary_visibility(): the analytic mu of a uniform disk (2*J1(v)/v), of a
  disk with linear limb darkening and of a binary star against the baseline.
- uniform_disk_first_zero(): the baseline 1.22*wavelength/theta at which the
  fringes of a uniform disk vanish (Michelson's criterion).
- incoherent_source_realization(): a random field sqrt(I)*exp(i*phi) (or a
  circular Gaussian field) whose ensemble average is a spatially incoherent
  source with intensity I.

Temporal coherence:

- degree_of_coherence(): gamma(tau) of a sampled spectrum (Wiener-Khinchin).
- line_spectrum(), line_coherence(), coherence_time_analytic(): the
  spectrum, |gamma| and the coherence time of Gaussian, Lorentzian and
  rectangular lines.
- coherence_time(): Mandel's coherence time int |gamma|**2 dtau of a sampled
  gamma.
- fit_fringes(): visibility and phase of a fringe pattern of known period
  by a linear least squares fit.

Intensity interferometry (Hanbury Brown and Twiss):

- chaotic_light(), correlated_chaotic_pair(): time series of thermal
  (chaotic) light with a given line shape, two of them with a given mu.
- g1(), g2(): first and second order correlation functions of time series
  (circular, via FFTs). For chaotic light the Siegert relation
  g2 = 1 + |g1|**2 holds.
- intensity_correlation(): <dI1*dI2>/(<I1><I2>), which equals |mu_12|**2
  for chaotic light.

References
----------
P. H. van Cittert, "Die wahrscheinliche Schwingungsverteilung in einer von
einer Lichtquelle direkt oder mittels einer Linse beleuchteten Ebene",
Physica 1, 201-210 (1934).

F. Zernike, "The concept of degree of coherence and its application to
optical problems", Physica 5, 785-795 (1938).

A. A. Michelson and F. G. Pease, "Measurement of the diameter of alpha
Orionis with the interferometer", Astrophys. J. 53, 249-259 (1921).

R. Hanbury Brown and R. Q. Twiss, "A test of a new type of stellar
interferometer on Sirius", Nature 178, 1046-1048 (1956).

R. Hanbury Brown, J. Davis and L. R. Allen, "The angular diameters of 32
stars", Mon. Not. R. Astron. Soc. 167, 121-136 (1974).

L. Mandel and E. Wolf, Optical Coherence and Quantum Optics, Cambridge
University Press (1995), chapters 4 and 9.
"""

import numpy as np
from scipy.special import j1, spherical_jn

from . import fft


SHAPES = ("gaussian", "lorentzian", "rectangular")


def _check_shape(shape):
    if shape not in SHAPES:
        raise ValueError("shape must be one of {}, not {!r}".format(SHAPES, shape))


def _jinc2(v):
    """2*J1(v)/v with the limit 1 at v = 0."""
    v = np.asarray(v, dtype=float)
    small = np.abs(v) < 1e-8
    v_safe = np.where(small, 1.0, v)

    return np.where(small, 1.0 - v**2/8, 2*j1(v_safe)/v_safe)


def uniform_disk_visibility(baseline, angular_diameter, wavelength):
    """Complex degree of coherence of a uniform disk, 2*J1(v)/v.

    By the van Cittert-Zernike theorem, a uniformly bright, incoherent disk
    of angular diameter theta gives mu = 2*J1(v)/v with
    v = pi*theta*B/wavelength for two points a baseline B apart. mu is real;
    it is negative between the first and second zero (the fringes shift by
    half a period).

    Parameters
    ----------
    baseline : float or array
        Distance between the two points (same unit as the wavelength).
    angular_diameter : float
        Angular diameter of the disk (rad).
    wavelength : float
        Wavelength.

    Returns
    -------
    mu : float or array
        Degree of coherence (real, between -0.132 and 1).
    """

    return _jinc2(np.pi*angular_diameter*np.asarray(baseline, dtype=float)/wavelength)


def limb_darkened_disk_visibility(baseline, angular_diameter, wavelength, u):
    """Degree of coherence of a disk with linear limb darkening.

    The brightness falls from the centre to the limb as
    I(r) = 1 - u*(1 - sqrt(1 - (r/R)**2)). The visibility is (Hanbury Brown
    et al. 1974) ::

        mu = ((1 - u)*J1(v)/v + u*(sin(v) - v*cos(v))/v**3)/((1 - u)/2 + u/3)

    with v = pi*theta*B/wavelength. u = 0 is the uniform disk. A darkened
    limb makes the star look smaller: its first zero lies at a longer
    baseline than that of a uniform disk of the same diameter.

    Parameters
    ----------
    baseline : float or array
        Distance between the two points.
    angular_diameter : float
        Angular diameter of the limb (rad).
    wavelength : float
        Wavelength.
    u : float
        Linear limb darkening coefficient, 0 <= u <= 1.

    Returns
    -------
    mu : float or array
        Degree of coherence (real).
    """

    v = np.pi*angular_diameter*np.asarray(baseline, dtype=float)/wavelength
    small = np.abs(v) < 1e-6
    v_safe = np.where(small, 1.0, v)
    uniform = 0.5*_jinc2(v)                                      # J1(v)/v
    darkened = np.where(small, 1/3 - v**2/30, spherical_jn(1, v_safe)/v_safe)

    return ((1 - u)*uniform + u*darkened)/((1 - u)/2 + u/3)


def binary_visibility(baseline, separation, brightness_ratio, wavelength):
    """Complex degree of coherence of two unresolved stars.

    For two point sources with the intensities 1 and r, a projected angular
    separation s along the baseline, and the brighter one on the axis ::

        mu = (1 + r*exp(-2*pi*i*s*B/wavelength))/(1 + r)

    |mu| oscillates between 1 and |1 - r|/(1 + r) with the period
    wavelength/s in the baseline.

    Parameters
    ----------
    baseline : float or array
        Baseline B.
    separation : float
        Angular separation projected onto the baseline (rad).
    brightness_ratio : float
        Intensity of the companion relative to the primary.
    wavelength : float
        Wavelength.

    Returns
    -------
    mu : complex or array
        Complex degree of coherence.
    """

    B = np.asarray(baseline, dtype=float)
    r = brightness_ratio

    return (1 + r*np.exp(-2j*np.pi*separation*B/wavelength))/(1 + r)


def uniform_disk_first_zero(angular_diameter, wavelength):
    """Baseline of the first zero of 2*J1(v)/v, about 1.22*wavelength/theta.

    Parameters
    ----------
    angular_diameter : float
        Angular diameter (rad).
    wavelength : float
        Wavelength.

    Returns
    -------
    B0 : float
        Baseline (same unit as the wavelength).
    """

    return 3.8317059702075125/np.pi*wavelength/angular_diameter


def mutual_intensity_vcz(intensity, x, y, wavelength, dx_out, dy_out, z=None):
    """Degree of coherence of an incoherent source by van Cittert-Zernike.

    For a planar, spatially incoherent, quasi-monochromatic source of
    intensity I(xi, eta) the degree of coherence of two points a distance
    z away, separated by (dx, dy), is (in the Fresnel approximation) ::

        mu(dx, dy) = exp(i*psi)*sum I*exp(-i*k*(dx*xi + dy*eta)/z)/sum I

    with the phase psi = k*(r1**2 - r2**2)/(2*z) that depends on the
    positions of both points. This function returns the sum alone, the
    reduced degree of coherence, with fft.zoom_FT(). For a source at
    infinity (a star) pass angular coordinates as x, y and z=None; dx_out,
    dy_out are then the baselines.

    Parameters
    ----------
    intensity : array
        Source intensity of shape (len(y), len(x)).
    x, y : arrays
        Linear, equidistant source coordinates (or angles in rad if z is
        None).
    wavelength : float
        Wavelength.
    dx_out, dy_out : arrays
        Linear, equidistant separations (or baselines) at which mu is
        computed.
    z : float, optional
        Distance between the source and the observation plane. None for a
        source at infinity.

    Returns
    -------
    mu : array
        Complex degree of coherence of shape (len(dy_out), len(dx_out)),
        1 at zero separation.
    """

    k = 2*np.pi/wavelength
    scale = k if z is None else k/z
    kx = scale*np.asarray(dx_out, dtype=float)
    ky = scale*np.asarray(dy_out, dtype=float)
    mu = fft.zoom_FT(np.asarray(intensity, dtype=complex), x, y, kx, ky)

    return mu/np.sum(intensity)


def incoherent_source_realization(intensity, rng=None, gaussian=False):
    """One realization of a spatially incoherent source.

    Returns sqrt(I)*exp(i*phi) with independent phases phi, uniform in
    [0, 2*pi), in each sample. With gaussian=True the amplitude is a
    circular complex Gaussian random number of variance I instead (thermal
    light, each sample the sum of many independent emitters). In both cases
    <u(r1)*conj(u(r2))> = I(r1)*delta_12. Averaging the intensities that
    many realizations produce after propagation gives the partially coherent
    result.

    Parameters
    ----------
    intensity : array
        Source intensity.
    rng : numpy.random.Generator, optional
        Random number generator.
    gaussian : bool, optional
        Circular Gaussian amplitudes instead of a random phase only.

    Returns
    -------
    u : array
        Complex field of the shape of intensity.
    """

    rng = np.random.default_rng() if rng is None else rng
    I = np.asarray(intensity, dtype=float)
    if gaussian:
        noise = (rng.standard_normal(I.shape) + 1j*rng.standard_normal(I.shape))/np.sqrt(2)
        return np.sqrt(I)*noise

    return np.sqrt(I)*np.exp(2j*np.pi*rng.random(I.shape))


def degree_of_coherence(nu, spectrum, tau):
    """Complex degree of coherence of a spectrum (Wiener-Khinchin).

    gamma(tau) = sum S(nu)*exp(-2*pi*i*nu*tau)/sum S(nu), evaluated
    directly. For a spectrum centered at nu_0, gamma is a carrier
    exp(-2*pi*i*nu_0*tau) under the envelope |gamma|; pass nu - nu_0 to get
    the envelope alone (the fringe visibility of a Michelson interferometer
    at the optical path difference c*tau).

    Parameters
    ----------
    nu : array
        Frequencies (equidistant or not; the sum uses the samples as weights
        of equal width, so resample non-equidistant grids first).
    spectrum : array
        Power spectral density at nu.
    tau : float or array
        Delays.

    Returns
    -------
    gamma : complex or array
        Complex degree of coherence at tau.
    """

    nu = np.asarray(nu, dtype=float)
    S = np.asarray(spectrum, dtype=float)
    tau = np.asarray(tau, dtype=float)
    phase = np.exp(-2j*np.pi*np.multiply.outer(tau, nu))

    return phase @ S/np.sum(S)


def line_coherence(tau, delta_nu, shape="gaussian"):
    """Envelope |gamma(tau)| of a spectral line of FWHM delta_nu.

    - gaussian: exp(-(pi*delta_nu*tau)**2/(4*ln 2))
    - lorentzian: exp(-pi*delta_nu*|tau|)
    - rectangular: |sinc(delta_nu*tau)|

    Parameters
    ----------
    tau : float or array
        Delay.
    delta_nu : float
        Full width at half maximum of the line (for the rectangle its full
        width).
    shape : str, optional
        "gaussian", "lorentzian" or "rectangular".

    Returns
    -------
    gamma : float or array
        |gamma(tau)|.
    """

    _check_shape(shape)
    tau = np.asarray(tau, dtype=float)
    if shape == "gaussian":
        return np.exp(-(np.pi*delta_nu*tau)**2/(4*np.log(2)))
    if shape == "lorentzian":
        return np.exp(-np.pi*delta_nu*np.abs(tau))

    return np.abs(np.sinc(delta_nu*tau))


def coherence_time_analytic(delta_nu, shape="gaussian"):
    """Mandel's coherence time int |gamma|**2 dtau of a spectral line.

    gaussian: sqrt(2*ln 2/pi)/delta_nu = 0.664/delta_nu, lorentzian:
    1/(pi*delta_nu) = 0.318/delta_nu, rectangular: 1/delta_nu. The coherence
    length is c times the coherence time; with delta_nu = c*delta_lambda/
    lambda**2 it is of the order of lambda**2/delta_lambda.

    Parameters
    ----------
    delta_nu : float
        FWHM of the line (full width of the rectangle).
    shape : str, optional
        "gaussian", "lorentzian" or "rectangular".

    Returns
    -------
    tau_c : float
        Coherence time.
    """

    _check_shape(shape)
    factor = {"gaussian": np.sqrt(2*np.log(2)/np.pi), "lorentzian": 1/np.pi,
              "rectangular": 1.0}[shape]

    return factor/delta_nu


def coherence_time(tau, gamma):
    """Mandel's coherence time int |gamma|**2 dtau of a sampled gamma.

    The integral runs over the samples given (trapezoidal rule). Pass delays
    from -T to T, or delays >= 0 and double the result (|gamma| is even).

    Parameters
    ----------
    tau : array
        Delays.
    gamma : array
        Degree of coherence at tau.

    Returns
    -------
    tau_c : float
        Coherence time.
    """

    g = np.abs(gamma)**2
    return np.sum(0.5*(g[1:] + g[:-1])*np.diff(tau))


def fit_fringes(x, intensity, period, weights=None):
    """Visibility and phase of fringes of known period.

    Fits I = a + b*cos(2*pi*x/period) + c*sin(2*pi*x/period) by linear
    least squares and returns V = sqrt(b**2 + c**2)/a and the fringe phase
    phi with I = a*(1 + V*cos(2*pi*x/period - phi)). Unlike
    (I_max - I_min)/(I_max + I_min), the fit is insensitive to noise and
    gives the phase, i.e. the sign of a real degree of coherence.

    Parameters
    ----------
    x : array
        Positions (or optical path differences).
    intensity : array
        Fringe intensities at x.
    period : float
        Fringe period.
    weights : array, optional
        Weights of the samples, e.g. an envelope.

    Returns
    -------
    V : float
        Visibility.
    phi : float
        Phase of the fringes (rad).
    """

    x = np.ravel(x)
    I = np.ravel(intensity)
    w = np.ones_like(I) if weights is None else np.sqrt(np.ravel(weights))
    arg = 2*np.pi*x/period
    A = np.stack([np.ones_like(x), np.cos(arg), np.sin(arg)], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(A*w[:, None], I*w, rcond=None)

    return np.hypot(b, c)/a, np.arctan2(c, b)


def line_spectrum(nu, delta_nu, shape="gaussian"):
    """Line shape of FWHM delta_nu centred at nu = 0, with the peak 1.

    Parameters
    ----------
    nu : float or array
        Frequency offset from the line centre.
    delta_nu : float
        FWHM of the line (full width of the rectangle).
    shape : str, optional
        "gaussian", "lorentzian" or "rectangular".

    Returns
    -------
    S : float or array
        Spectral density, 1 at the centre.
    """

    _check_shape(shape)
    f = np.asarray(nu, dtype=float)
    if shape == "gaussian":
        return np.exp(-4*np.log(2)*(f/delta_nu)**2)
    if shape == "lorentzian":
        return 1/(1 + (2*f/delta_nu)**2)

    return (np.abs(f) <= delta_nu/2).astype(float)


def chaotic_light(n, dt, delta_nu, shape="gaussian", rng=None, size=None):
    """Complex envelope of chaotic (thermal) light.

    Filters complex white Gaussian noise to the line shape (centred at zero
    frequency, i.e. the slowly varying envelope of the optical field) and
    normalizes it to a mean intensity of 1. The field is a circular complex
    Gaussian process: its intensity is exponentially distributed and
    g2(tau) = 1 + |g1(tau)|**2. The series is periodic (circular filter).

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float
        Time step.
    delta_nu : float
        FWHM of the line.
    shape : str, optional
        "gaussian", "lorentzian" or "rectangular".
    rng : numpy.random.Generator, optional
        Random number generator.
    size : int, optional
        Number of independent series (rows of the result).

    Returns
    -------
    u : array
        Complex field of shape (n,) or (size, n) with <|u|**2> = 1.
    """

    rng = np.random.default_rng() if rng is None else rng
    shape_out = (n,) if size is None else (size, n)
    noise = rng.standard_normal(shape_out) + 1j*rng.standard_normal(shape_out)
    f = np.fft.fftfreq(n, dt)
    filt = np.sqrt(line_spectrum(f, delta_nu, shape))
    u = np.fft.ifft(np.fft.fft(noise, axis=-1)*filt, axis=-1)

    return u/np.sqrt(np.mean(np.abs(u)**2, axis=-1, keepdims=True))


def correlated_chaotic_pair(n, dt, delta_nu, mu, shape="gaussian", rng=None):
    """Fields of chaotic light at two points with the degree of coherence mu.

    u2 = conj(mu)*u1 + sqrt(1 - |mu|**2)*w with u1, w independent series from
    chaotic_light(), so that <u1*conj(u2)> = mu and both have unit mean
    intensity. This is the light of a star seen by two telescopes whose
    baseline gives mu (without delay between them).

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float
        Time step.
    delta_nu : float
        FWHM of the line.
    mu : complex
        Degree of spatial coherence, |mu| <= 1.
    shape : str, optional
        Line shape, see chaotic_light().
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    u1, u2 : arrays
        Complex fields of length n.
    """

    rng = np.random.default_rng() if rng is None else rng
    u1, w = chaotic_light(n, dt, delta_nu, shape, rng, size=2)

    return u1, np.conj(mu)*u1 + np.sqrt(1 - abs(mu)**2)*w


def _circular_correlation(a, b, max_lag):
    """<conj(a(t))*b(t + lag)> for lag = 0..max_lag (circular)."""
    n = a.shape[-1]
    c = np.fft.ifft(np.conj(np.fft.fft(a, axis=-1))*np.fft.fft(b, axis=-1), axis=-1)/n

    return c[..., :max_lag + 1]


def g1(u, max_lag):
    """First order correlation <conj(u(t))*u(t + tau)>/<|u|**2>.

    Circular estimate over the series (or averaged over the rows of a 2d
    array of independent series).

    Parameters
    ----------
    u : array
        Complex field samples, shape (n,) or (m, n).
    max_lag : int
        Largest delay in samples.

    Returns
    -------
    g : array
        g1 for the delays 0..max_lag samples.
    """

    u = np.atleast_2d(u)
    c = _circular_correlation(u, u, max_lag).mean(axis=0)

    return c/c[0].real


def g2(intensity, max_lag):
    """Second order correlation <I(t)*I(t + tau)>/<I>**2.

    Circular estimate over the series (or averaged over the rows).

    Parameters
    ----------
    intensity : array
        Intensity samples, shape (n,) or (m, n).
    max_lag : int
        Largest delay in samples.

    Returns
    -------
    g : array
        g2 for the delays 0..max_lag samples.
    """

    I = np.atleast_2d(np.asarray(intensity, dtype=float))
    c = _circular_correlation(I, I, max_lag).real.mean(axis=0)

    return c/np.mean(I)**2


def intensity_correlation(I1, I2, axis=None):
    """Normalized correlation of intensity fluctuations.

    c = <(I1 - <I1>)*(I2 - <I2>)>/(<I1>*<I2>). For chaotic light at two
    points, c = |mu_12|**2 (if the detectors resolve the coherence time;
    slower detectors reduce c by about tau_c/T_detector).

    Parameters
    ----------
    I1, I2 : arrays
        Intensities at the two points.
    axis : int, optional
        Axis to average over (all by default).

    Returns
    -------
    c : float or array
        Normalized intensity correlation.
    """

    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    m1, m2 = I1.mean(axis=axis, keepdims=True), I2.mean(axis=axis, keepdims=True)
    c = np.mean((I1 - m1)*(I2 - m2), axis=axis, keepdims=True)/(m1*m2)

    return np.squeeze(c, axis=axis) if axis is not None else c.item()
