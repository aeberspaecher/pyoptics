#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical coherence tomography (OCT): time-domain and spectral-domain
A-scans of layered samples.

A Michelson interferometer splits light of a broadband source into a
reference arm (a mirror at the distance z_R) and a sample arm. The sample
reflects with the wavenumber dependent amplitude r_S(k), which holds the
depths of all its reflectors as phases exp(2ikz). With the field convention
of `thin_films` (exp(i(kz - omega t)), a path z adds exp(+ikz)), the
detector sees::

    I(k) = S(k) abs(r_R exp(2ikz_R) + r_S(k) exp(2ikz_S))**2

for the power spectrum S(k) of the source (the constant losses of the beam
splitter are left out). Expanded, it is the sum of the "DC" term
S(abs(r_R)**2 + abs(r_S)**2), which holds the autocorrelation of the sample
reflectors, and the cross term 2 S Re(r_R* r_S exp(2ik(z_S - z_R))), which
holds their depths relative to the reference.

- Time-domain OCT (TD-OCT) scans z_R and records the integral of I(k) over
  k: the fringes of the cross term are visible only within the coherence
  length of the source, and their envelope is the A-scan
  (`time_domain_signal()`, `demodulate()`).
- Spectral-domain OCT (SD-OCT) keeps z_R fixed and records I(k) with a
  spectrometer (`spectrometer()`); the A-scan is its Fourier transform from
  k to z (`a_scan()`, depth axis `depth_axis()`). The Fourier transform of a
  real signal is symmetric: every reflector appears at +z and at its mirror
  image -z, and the DC term sits at z = 0. Two spectra with the reference
  shifted by a quarter wave give a complex interferogram without mirror
  images (`full_range_interferogram()`).

Lengths are in any unit (the notebook uses um), k = 2 pi/lambda is the
vacuum wavenumber and z an optical path (geometric depth times group index
inside a sample). Depths are single pass: a reflector at the depth z has
the round trip delay 2z.

Analytic results:

- `axial_resolution()`: the A-scan of a mirror with a Gaussian spectrum is
  the Gaussian coherence function, FWHM 2 ln2/pi lambda_0**2/Delta_lambda.
- `max_depth()`: a spectrometer with pixels delta_k apart sees the depth
  range pi/(2 delta_k) = lambda_0**2/(4 delta_lambda).
- `sensitivity_rolloff()`: the finite pixel width and spectral resolution
  of the spectrometer wash out fast fringes, the signal of deep reflectors
  falls off as sinc(z delta_k/pi) exp(-z**2 w**2/(4 ln2)) (Yun et al.,
  Hu et al.).
- `dispersion_broadening()`: unbalanced group velocity dispersion phi_2 =
  d**2 phi/dk**2 widens the Gaussian coherence peak by the factor
  sqrt(1 + (phi_2 w**2/(8 ln2))**2) (Hitzenberger et al.).
- `sd_td_snr_ratio()`: for the same exposure and shot noise, SD-OCT has a
  signal-to-noise ratio higher by the number of depths it measures in
  parallel, L/int abs(gamma)**2 dz (Leitgeb et al., de Boer et al., Choma
  et al.).

`stack_reflection()` computes r_S(k) of a layered sample with
`thin_films.r_t_coeffs()`, including dispersion, absorption and multiple
reflections. `compensate_dispersion()` and `optimize_dispersion()` undo the
quadratic and cubic phase of an unbalanced interferometer numerically, by
maximizing the sharpness of the A-scan (Wojtkowski et al.).

References
----------
D. Huang, E. A. Swanson, C. P. Lin, J. S. Schuman, W. G. Stinson, W. Chang,
M. R. Hee, T. Flotte, K. Gregory, C. A. Puliafito and J. G. Fujimoto,
"Optical coherence tomography", Science 254, 1178-1181 (1991).

A. F. Fercher, W. Drexler, C. K. Hitzenberger and T. Lasser, "Optical
coherence tomography - principles and applications", Rep. Prog. Phys. 66,
239-303 (2003).

J. A. Izatt and M. A. Choma, "Theory of optical coherence tomography", in
W. Drexler and J. G. Fujimoto (eds.), "Optical Coherence Tomography",
Springer (2008).

R. Leitgeb, C. K. Hitzenberger and A. F. Fercher, "Performance of Fourier
domain vs. time domain optical coherence tomography", Opt. Express 11,
889-894 (2003).

S. H. Yun, G. J. Tearney, B. E. Bouma, B. H. Park and J. F. de Boer,
"High-speed spectral-domain optical coherence tomography at 1.3 um
wavelength", Opt. Express 11, 3598-3604 (2003).

Z. Hu, Y. Pan and A. M. Rollins, "Analytical model of spectrometer-based
two-beam spectral interferometry", Appl. Opt. 46, 8499-8505 (2007).

M. Wojtkowski, V. J. Srinivasan, T. H. Ko, J. G. Fujimoto, A. Kowalczyk
and J. S. Duker, "Ultrahigh-resolution, high-speed, Fourier domain optical
coherence tomography and methods for dispersion compensation", Opt.
Express 12, 2404-2422 (2004).

C. K. Hitzenberger, A. Baumgartner, W. Drexler and A. F. Fercher,
"Dispersion effects in partial coherence interferometry: implications for
intraocular ranging", J. Biomed. Opt. 4, 144-151 (1999).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import hilbert

from .thin_films import r_t_coeffs


__all__ = ["axial_resolution", "bandwidth_k", "gaussian_spectrum", "coherence_envelope",
           "max_depth", "depth_axis", "spectral_interferogram", "spectrometer", "a_scan",
           "sensitivity_rolloff", "time_domain_signal", "demodulate", "stack_reflection",
           "single_reflection_amplitudes", "analytic_interferogram",
           "full_range_interferogram", "round_trip_phase", "dispersion_coefficients",
           "dispersion_broadening", "compensate_dispersion", "sharpness",
           "optimize_dispersion", "sd_td_snr_ratio", "fwhm"]


LN2 = np.log(2.0)


def axial_resolution(center_wavelength, bandwidth, n=1.0):
    """Axial resolution (FWHM of the A-scan of a mirror) for a Gaussian
    spectrum, 2 ln2/pi lambda_0**2/Delta_lambda, divided by the group index
    n of the sample.

    Parameters
    ----------
    center_wavelength : number or array
        Centre wavelength lambda_0.
    bandwidth : number or array
        FWHM Delta_lambda of the power spectrum (same unit).
    n : number, optional
        Group index of the medium (1: optical path).

    Returns
    -------
    number or array
        The depth resolution, in the unit of the wavelength.
    """

    return 2*LN2/np.pi*np.asarray(center_wavelength)**2/np.asarray(bandwidth)/n


def bandwidth_k(center_wavelength, bandwidth):
    """FWHM in wavenumber of a spectrum with the FWHM bandwidth in
    wavelength, 2 pi Delta_lambda/lambda_0**2 (to first order).

    Parameters
    ----------
    center_wavelength, bandwidth : number

    Returns
    -------
    number
    """

    return 2*np.pi*bandwidth/center_wavelength**2


def gaussian_spectrum(k, k0, width):
    """Gaussian power spectrum exp(-4 ln2 (k - k0)**2/width**2), peak 1.

    Parameters
    ----------
    k : array
        Wavenumbers.
    k0 : number
        Centre wavenumber.
    width : number
        FWHM in k.

    Returns
    -------
    array
    """

    return np.exp(-4*LN2*(np.asarray(k) - k0)**2/width**2)


def coherence_envelope(z, width):
    """Envelope of the A-scan of a mirror for a Gaussian spectrum of FWHM
    `width` in k, exp(-z**2 width**2/(4 ln2)), as a function of the
    (single pass) path difference z. Its FWHM is `axial_resolution()`.

    Parameters
    ----------
    z : array
    width : number

    Returns
    -------
    array
    """

    return np.exp(-np.asarray(z)**2*width**2/(4*LN2))


def max_depth(k):
    """Maximum depth pi/(2 delta_k) of a spectrometer with the equidistant
    wavenumbers k (lambda_0**2/(4 delta_lambda)).

    Parameters
    ----------
    k : array

    Returns
    -------
    number
    """

    return np.pi/(2*abs(k[1] - k[0]))


def depth_axis(k, n_fft=None):
    """Depths of the FFT bins of an interferogram sampled at the equidistant
    wavenumbers k, in FFT order (0, positive, negative depths).

    Parameters
    ----------
    k : array
    n_fft : int, optional
        FFT length (zero padding), default len(k).

    Returns
    -------
    array
    """

    n_fft = len(k) if n_fft is None else n_fft
    return np.fft.fftfreq(n_fft, d=abs(k[1] - k[0])/np.pi)


def spectral_interferogram(k, spectrum, r_sample, z_sample=0.0, r_ref=1.0, z_ref=0.0):
    """Spectrum at the detector of a Michelson interferometer,
    S abs(r_R exp(2ikz_R) + r_S exp(2ikz_S))**2.

    Parameters
    ----------
    k : array
        Wavenumbers.
    spectrum : array
        Power spectrum S(k) of the source.
    r_sample : number or array
        Reflection coefficient of the sample, r_S(k) (referred to its top
        surface).
    z_sample : number, optional
        Path to the top surface of the sample.
    r_ref : number or complex, optional
        Reflection coefficient of the reference mirror (a phase factor
        exp(i phi) shifts the reference by a fraction of a wave at all k).
    z_ref : number, optional
        Path to the reference mirror.

    Returns
    -------
    array
    """

    k = np.asarray(k)
    E_R = r_ref*np.exp(2j*k*z_ref)
    E_S = r_sample*np.exp(2j*k*z_sample)
    return np.asarray(spectrum)*np.abs(E_R + E_S)**2


def spectrometer(k, signal, n_pixels, resolution=0.0):
    """Record a finely sampled spectrum with a spectrometer of n_pixels
    pixels, linear in k: blur with a Gaussian of FWHM `resolution` (the
    image of the slit and the optics), then average over each pixel.

    Parameters
    ----------
    k : array
        Equidistant fine wavenumbers; len(k) must be a multiple of
        n_pixels.
    signal : array
        Spectrum on k (last axis).
    n_pixels : int
    resolution : number, optional
        FWHM of the spectral resolution in k (0: limited by the pixels).

    Returns
    -------
    k_pixels, counts : arrays
        Centre wavenumbers of the pixels and the recorded spectrum.
    """

    k = np.asarray(k)
    signal = np.asarray(signal, dtype=float)
    m = len(k)//n_pixels
    if m*n_pixels != len(k):
        raise ValueError("len(k) must be a multiple of n_pixels.")
    if resolution > 0:
        sigma = resolution/(2*np.sqrt(2*LN2))/abs(k[1] - k[0])
        signal = gaussian_filter1d(signal, sigma, axis=-1, mode="nearest")
    counts = signal.reshape(signal.shape[:-1] + (n_pixels, m)).mean(axis=-1)
    return k.reshape(n_pixels, m).mean(axis=-1), counts


def a_scan(interferogram, k, background=None, window=None, norm=None, n_fft=None):
    """A-scan of a spectral interferogram: the Fourier transform from k to
    the depth z with the kernel exp(-2ikz).

    Parameters
    ----------
    interferogram : array
        Spectrum on the equidistant wavenumbers k (last axis), real or
        complex (`analytic_interferogram()`, `full_range_interferogram()`).
    k : array
    background : array or number, optional
        Subtracted before the transform (e.g. the reference arm spectrum);
        default: the mean of the interferogram, which removes the DC peak at
        z = 0 only partly if S(k) is not flat. Pass 0 to keep the DC term.
    window : array, optional
        Apodization over k (e.g. np.hanning(len(k))).
    norm : number, optional
        The transform is divided by it, default the sum of the window. Pass
        sum(spectrum*window) to get the reflection amplitude abs(r_R r_S) as
        the peak height of a single reflector.
    n_fft : int, optional
        FFT length (zero padding for finer depth sampling).

    Returns
    -------
    z, A : arrays
        Depths from -z_max to z_max (fftshifted) and the complex A-scan.
    """

    x = np.asarray(interferogram)
    if background is None:
        x = x - x.mean(axis=-1, keepdims=True)
    else:
        x = x - background
    w = np.ones(len(k)) if window is None else np.asarray(window)
    n_fft = len(k) if n_fft is None else n_fft
    z = depth_axis(k, n_fft)
    if k[1] < k[0]:
        raise ValueError("k must increase.")
    A = np.fft.fft(x*w, n=n_fft, axis=-1)*np.exp(-2j*k[0]*z)
    A = A/(np.sum(w) if norm is None else norm)
    return np.fft.fftshift(z), np.fft.fftshift(A, axes=-1)


def sensitivity_rolloff(z, pixel_width, resolution=0.0):
    """Depth dependent fringe amplitude of an SD-OCT spectrometer, relative
    to z = 0: sinc(z delta_k/pi) exp(-z**2 w**2/(4 ln2)) for pixels of the
    width delta_k (box) and a Gaussian spectral resolution of FWHM w.

    In decibels of the signal power, 20 log10 of the result. At the maximum
    depth pi/(2 delta_k) the sinc alone gives 2/pi (-3.9 dB).

    Parameters
    ----------
    z : array
        Depth (single pass optical path).
    pixel_width : number
        Width (= spacing) of the pixels in k.
    resolution : number, optional
        FWHM of the spectral resolution in k.

    Returns
    -------
    array
    """

    z = np.asarray(z)
    return np.sinc(z*pixel_width/np.pi)*np.exp(-z**2*resolution**2/(4*LN2))


def time_domain_signal(k, spectrum, r_sample, z_ref, z_sample=0.0, r_ref=1.0, chunk=2048):
    """TD-OCT detector signal: the interferogram integrated over k, for each
    reference position, divided by the integral of S(k) (a mirror sample
    gives fringes of amplitude 2 abs(r_R r_S) on the background
    abs(r_R)**2 + abs(r_S)**2).

    Parameters
    ----------
    k : array
        Equidistant wavenumbers that sample the spectrum.
    spectrum : array
        S(k).
    r_sample : number or array
        r_S(k).
    z_ref : array
        Positions of the reference mirror.
    z_sample : number, optional
    r_ref : number, optional
    chunk : int, optional
        Number of reference positions computed at once.

    Returns
    -------
    array
        The signal at z_ref.
    """

    k = np.asarray(k)
    S = np.asarray(spectrum)*np.ones(len(k))
    E_S = np.asarray(r_sample)*np.exp(2j*k*z_sample)*np.ones(len(k))
    z_ref = np.asarray(z_ref, dtype=float)
    dc = np.sum(S*(np.abs(r_ref)**2 + np.abs(E_S)**2))
    out = np.empty(z_ref.shape)
    flat = z_ref.ravel()
    res = out.reshape(-1)
    for i in range(0, len(flat), chunk):
        zr = flat[i:i + chunk]
        cross = np.exp(-2j*np.outer(zr, k)) @ (S*E_S)
        res[i:i + chunk] = dc + 2*np.real(np.conj(r_ref)*cross)
    return out/np.sum(S)


def demodulate(signal):
    """Complex envelope of TD-OCT fringes, half the analytic signal of the
    signal minus its mean, so that a mirror gives the peak abs(r_R r_S) as in
    `a_scan()`.

    Parameters
    ----------
    signal : array

    Returns
    -------
    array
        Complex; its modulus is the A-scan.
    """

    s = np.asarray(signal, dtype=float)
    return hilbert(s - s.mean())/2


def _n_at(n, k):
    """Refractive index (number or callable of the vacuum wavelength in the
    length unit of k) at the wavenumbers k."""

    return n(2*np.pi/k) if callable(n) else n*np.ones(np.shape(k))


def stack_reflection(k, n_layers, thicknesses, n_embedding=1.0, n_substrate=1.0):
    """Reflection coefficient r_S(k) of a layer stack at normal incidence,
    with `thin_films.r_t_coeffs()`: all multiple reflections, dispersion and
    absorption included.

    Parameters
    ----------
    k : array
        Vacuum wavenumbers.
    n_layers : sequence
        Complex refractive index n + i kappa of each layer (from the
        embedding side): a number or a callable of the vacuum wavelength
        (in the length unit of k), e.g. lambda wl: glass.n(1000*wl) for um.
    thicknesses : sequence
        Geometric thicknesses of the layers.
    n_embedding, n_substrate : number or callable, optional

    Returns
    -------
    array
        Complex r_S(k), referred to the top surface of the stack.
    """

    k = np.atleast_1d(np.asarray(k, dtype=float))
    n_all = [np.asarray(_n_at(n, k), dtype=complex) for n in n_layers]
    n_emb = np.asarray(_n_at(n_embedding, k), dtype=complex)
    n_sub = np.asarray(_n_at(n_substrate, k), dtype=complex)
    r = np.empty(len(k), dtype=complex)
    t = list(thicknesses)
    for i, kk in enumerate(k):
        n = np.array([nn[i] for nn in n_all])
        if len(n) == 0:  # bare interface: r_t_coeffs needs one layer
            n, t = np.array([n_emb[i]]), [0.0]
        r[i], _ = r_t_coeffs(0.0, n.real, n.imag, t, n_emb[i].real, n_emb[i].imag,
                             n_sub[i].real, n_sub[i].imag, "TE", 2*np.pi/kk)
    return r


def single_reflection_amplitudes(n_layers, thicknesses, n_embedding=1.0, n_substrate=1.0,
                                 group_index=None):
    """Depths and amplitudes of the interfaces of a non-absorbing stack as
    an OCT sees them without multiple reflections: the Fresnel coefficient
    of each interface times the round trip transmission
    prod (1 - r_i**2) of the interfaces above it.

    Parameters
    ----------
    n_layers : sequence of numbers
        Real refractive indices of the layers (at the centre wavelength).
    thicknesses : sequence
    n_embedding, n_substrate : number, optional
    group_index : sequence, optional
        Group indices of the layers for the optical depths (default the
        phase indices).

    Returns
    -------
    z, r : arrays
        Optical depths (sum of n_g t above each interface) and signed
        amplitudes of the len(n_layers) + 1 interfaces.
    """

    n = np.concatenate(([n_embedding], np.asarray(n_layers, float), [n_substrate]))
    ng = np.asarray(n_layers, float) if group_index is None else np.asarray(group_index, float)
    r_if = (n[:-1] - n[1:])/(n[:-1] + n[1:])
    trans = np.concatenate(([1.0], np.cumprod(1 - r_if[:-1]**2)))
    z = np.concatenate(([0.0], np.cumsum(ng*np.asarray(thicknesses, float))))
    return z, r_if*trans


def analytic_interferogram(interferogram, background=None):
    """The cross term S r_R* r_S exp(2ik(z_S - z_R)) of a real
    interferogram with all reflectors at positive depth: half the analytic
    signal (Hilbert transform over k) of the interferogram minus its
    background. Its A-scan has no mirror images, and a phase correction
    (`compensate_dispersion()`) acts on it like on the field.

    Parameters
    ----------
    interferogram : array
        Real spectrum (last axis over k, increasing).
    background : array or number, optional
        Default: the mean.

    Returns
    -------
    array
    """

    x = np.asarray(interferogram, dtype=float)
    x = x - (x.mean(axis=-1, keepdims=True) if background is None else background)
    return hilbert(x, axis=-1)/2


def full_range_interferogram(I_0, I_90, background=None):
    """Complex interferogram from two spectra taken with the reference
    shifted by a quarter wave (r_R and r_R exp(i pi/2)): (I_0 - B)/2 +
    i (I_90 - B)/2 = S r_R* r_S exp(...), whose A-scan has no mirror images
    (complex conjugate resolved or "full range" OCT).

    Parameters
    ----------
    I_0, I_90 : arrays
    background : array or number, optional
        The DC term (default: the mean of the two spectra).

    Returns
    -------
    array
    """

    I_0, I_90 = np.asarray(I_0, dtype=float), np.asarray(I_90, dtype=float)
    if background is None:
        background = (I_0 + I_90).mean(axis=-1, keepdims=True)/2
    return ((I_0 - background) + 1j*(I_90 - background))/2


def round_trip_phase(k, n, thickness):
    """Round trip phase 2 k n(k) L through a medium of the thickness L.

    Parameters
    ----------
    k : array
    n : number, array or callable
        Refractive index (callable of the vacuum wavelength in the unit
        of k).
    thickness : number

    Returns
    -------
    array
    """

    return 2*np.asarray(k)*_n_at(n, k)*thickness


def dispersion_coefficients(k, phase, k0, order=3):
    """Taylor coefficients phi_j = d**j phi/dk**j at k0 (j = 0 .. order) of
    a phase, from a polynomial fit.

    phi_1 is the round trip group delay (twice the group optical path),
    phi_2 the group delay dispersion that broadens the A-scan.

    Parameters
    ----------
    k, phase : arrays
    k0 : number
    order : int, optional

    Returns
    -------
    array
        phi_0 ... phi_order.
    """

    from math import factorial

    c = np.polynomial.polynomial.polyfit(np.asarray(k) - k0, phase, order)
    return np.array([c[j]*factorial(j) for j in range(order + 1)])


def dispersion_broadening(phi_2, width):
    """Factor by which a quadratic spectral phase (1/2) phi_2 (k - k0)**2
    widens the A-scan peak of a Gaussian spectrum of FWHM `width` in k,
    sqrt(1 + (phi_2 width**2/(8 ln2))**2); the peak drops by its square root.

    Parameters
    ----------
    phi_2 : number or array
        Unbalanced d**2 phi/dk**2 of the round trip phase.
    width : number

    Returns
    -------
    number or array
    """

    return np.sqrt(1 + (np.asarray(phi_2)*width**2/(8*LN2))**2)


def compensate_dispersion(interferogram, k, k0, coefficients):
    """Multiply a complex interferogram by exp(-i sum_j phi_j (k - k0)**j/j!)
    for j = 2, 3, ...: numerical dispersion compensation.

    Parameters
    ----------
    interferogram : array
        Complex (`analytic_interferogram()`), last axis over k.
    k : array
    k0 : number
    coefficients : sequence
        phi_2, phi_3, ... (as returned by `dispersion_coefficients()` from
        index 2 on).

    Returns
    -------
    array
    """

    from math import factorial

    q = np.asarray(k) - k0
    phase = sum(c*q**(j + 2)/factorial(j + 2) for j, c in enumerate(coefficients))
    return np.asarray(interferogram)*np.exp(-1j*phase)


def sharpness(a_scan_values):
    """Sharpness sum abs(A)**4/(sum abs(A)**2)**2 of an A-scan: larger for fewer,
    narrower peaks.

    Parameters
    ----------
    a_scan_values : array

    Returns
    -------
    number
    """

    p = np.abs(a_scan_values)**2
    return np.sum(p**2)/np.sum(p)**2


def optimize_dispersion(interferogram, k, k0, width, order=3, window=None, n_fft=None,
                        start=None):
    """Find the dispersion coefficients phi_2 (and phi_3, ...) that make the
    A-scan of a complex interferogram sharpest (Nelder-Mead on
    `sharpness()`).

    Parameters
    ----------
    interferogram : array
        Complex interferogram (`analytic_interferogram()`), a 1D spectrum or
        several A-scans (2D, sharpness summed).
    k : array
    k0 : number
    width : number
        FWHM of the spectrum in k; it scales the coefficients for the
        optimizer.
    order : int, optional
        Highest order of the phase polynomial (>= 2).
    window : array, optional
    n_fft : int, optional
    start : sequence, optional
        Initial phi_2, phi_3, ... (default 0).

    Returns
    -------
    array
        phi_2 ... phi_order.
    """

    from math import factorial

    q = (np.asarray(k) - k0)/width
    scales = np.array([width**j for j in range(2, order + 1)])
    x0 = np.zeros(order - 1) if start is None else np.asarray(start, float)*scales
    w = np.ones(len(k)) if window is None else window
    X = np.atleast_2d(interferogram)*w

    def cost(x):
        phase = sum(c*q**(j + 2)/factorial(j + 2) for j, c in enumerate(x))
        A = np.fft.fft(X*np.exp(-1j*phase), n=n_fft, axis=-1)
        return -sum(sharpness(a) for a in A)

    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 4000,
                            "initial_simplex": np.vstack([x0] + [x0 + 2.0*e for e in np.eye(order - 1)])})
    return res.x/scales


def sd_td_snr_ratio(depth_range, resolution):
    """Shot noise limited SNR of SD-OCT over TD-OCT for the same source
    power and A-scan time: L/int abs(gamma(z))**2 dz for a Gaussian coherence
    function of FWHM `resolution` scanned over the depth range L,
    L/(sqrt(pi/(8 ln2)) l_c), about the number of resolvable depths.

    Parameters
    ----------
    depth_range : number
        Depth L scanned by the TD reference mirror.
    resolution : number
        FWHM l_c of abs(gamma).

    Returns
    -------
    number
    """

    return depth_range/(np.sqrt(np.pi/(8*LN2))*resolution)


def fwhm(x, y):
    """Full width at half maximum of a single peak, with linear
    interpolation of the half maximum crossings.

    Parameters
    ----------
    x, y : arrays

    Returns
    -------
    number
    """

    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    h = y[i]/2
    left = i
    while left > 0 and y[left] > h:
        left -= 1
    right = i
    while right < len(y) - 1 and y[right] > h:
        right += 1
    xl = np.interp(h, [y[left], y[left + 1]], [x[left], x[left + 1]])
    xr = np.interp(h, [y[right], y[right - 1]], [x[right], x[right - 1]])
    return xr - xl
