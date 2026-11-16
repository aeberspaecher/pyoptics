#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Coherence scanning interferometry (CSI), also called white-light
interferometry or scanning white-light interferometry: surface heights
from the coherence envelope and the phase of broadband fringes.

A Mirau or Michelson objective splits the light of a broadband source
(an LED or a halogen lamp) into a reference mirror inside the objective
and the surface. The objective is scanned along its axis; every camera
pixel records a correlogram, fringes that appear only where the optical
paths of reference and surface agree within the coherence length. The
position of the envelope peak is the height of the surface at that pixel,
free of the 2 pi ambiguity of single-wavelength interferometry; the phase
of the fringes refines it to a fraction of a nanometre. It is the same
interference as in time-domain OCT (`oct`, notebook 82, optical coherence
tomography), applied to a surface instead of a volume, with an objective
of high numerical aperture instead of a nearly collimated beam.

Signal
------
With the scan position z of the objective (the height at which the
reference path equals the surface path), a surface of height h and the
wavenumber k = 2 pi/lambda, a plane wave in the pupil at the angle theta
gives the fringe term 2 Re(r_R* r_S exp(2 i k cos(theta) (z - h))): a
lower surface means a longer path and, in the convention of pyoptics
(and of `thin_films`, whose r_S holds the deeper reflections of a stack
as exp(+2 i k n d)), a larger phase. The
correlogram sums it over the source spectrum S(k) and over the
illuminated pupil (`correlogram()`, uniformly filled up to NA,
`pupil_quadrature()`) ::

    I(z) = sum_k sum_theta S(k) w(theta) [abs(r_R)**2 + abs(r_S)**2
                        + 2 Re(r_R* r_S exp(2 i k cos(theta) (z - h)))].

The envelope peaks at z = h for any NA. The fringes, however, have the
period lambda/(2 <cos(theta)>), longer than lambda/2 by the obliquity
factor <cos(theta)> = 2 (1 - cos(alpha)**3)/(3 sin(alpha)**2) for a
uniformly filled pupil of NA sin(alpha) (`obliquity_factor()`). A phase
evaluated with the nominal wavelength must be scaled with it.

Evaluation
----------
`envelope_peak()` finds the peak of the envelope (Hilbert transform, as
`oct.demodulate()`) by its centroid or a parabola. Frequency domain
analysis (FDA, de Groot and Deck) Fourier transforms the correlogram:
its phase at the spatial frequency K = 2 k cos(theta) is K h minus the
phase of r_S. A straight line fitted to the phase gives the coherence
height (slope) and, at the mean frequency K_0, the phase height, which is
unambiguous once the coherence height picks its fringe order (`fda()`).
FDA needs no knowledge of the NA or the spectrum.

The phase of r_S breaks both: a metal reflects with a phase that differs
from that of glass, which offsets the phase height by -arg(r_S)/(2 k)
(tens of nanometres between materials), and a transparent film makes the phase of
r_S depend on k, which tilts the FDA phase and shifts the coherence
height (the thin film effect). `film_reflectance()` gives r_S of layered
surfaces from `thin_films` for both effects; fitting the measured spectral
phase and amplitude against it gives the film thickness.

Lateral effects
---------------
`profile_scan()` simulates the correlograms of a line profile h(x) with
scalar imaging: each plane wave of the illumination is reflected (thin
element approximation, phase -2 k_z h(x)), its angular spectrum
propagated to the focal plane z of the objective and cut at the NA, and
interfered with the reference. It shows the batwing artefact at steps
lower than the coherence length, where the light diffracted by the edge
interferes with that from both levels and the envelope evaluation
overshoots, and the lateral resolution of the objective.

Lengths are in any unit (the notebook uses um); k is the vacuum
wavenumber.

References
----------
P. de Groot, "Coherence scanning interferometry", in R. Leach (ed.),
"Optical Measurement of Surface Topography", Springer (2011), 187-208.

P. de Groot and L. Deck, "Surface profiling by analysis of white-light
interferograms in the spatial frequency domain", J. Mod. Opt. 42, 389-401
(1995).

P. de Groot, X. Colonna de Lega, J. Kramer and M. Turzhitsky,
"Determination of fringe order in white-light interference microscopy",
Appl. Opt. 41, 4571-4578 (2002).

C. J. R. Sheppard and K. G. Larkin, "Effect of numerical aperture on
interference fringe spacing", Appl. Opt. 34, 4731-4734 (1995).

A. Harasaki and J. C. Wyant, "Fringe modulation skewing effect in
white-light vertical scanning interferometry", Appl. Opt. 39, 2101-2106
(2000).

S.-W. Kim and G.-H. Kim, "Thickness-profile measurement of transparent
thin-film layers by white-light scanning interferometry", Appl. Opt. 38,
5968-5973 (1999).
"""

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from scipy.signal import hilbert

from .thin_films import _r_t
from .oct import gaussian_spectrum, bandwidth_k


def led_spectrum(k, center_wavelength, fwhm_wavelength):
    """Gaussian spectrum in k of an LED (peak 1).

    Parameters
    ----------
    k : array
        Wavenumbers 2 pi/lambda.
    center_wavelength, fwhm_wavelength : numbers

    Returns
    -------
    array
    """

    return gaussian_spectrum(k, 2*np.pi/center_wavelength,
                             bandwidth_k(center_wavelength, fwhm_wavelength))


def white_led_spectrum(k, blue=0.455, blue_fwhm=0.022, phosphor=0.56, phosphor_fwhm=0.12,
                       blue_weight=0.6):
    """Spectrum of a phosphor-converted white LED: a narrow blue line and
    a broad yellow phosphor band (Gaussians in wavelength, per unit k).

    Parameters
    ----------
    k : array
        Wavenumbers 2 pi/lambda (um^-1 for the default wavelengths in um).
    blue, blue_fwhm, phosphor, phosphor_fwhm : numbers, optional
        Peaks and widths of the two bands.
    blue_weight : number, optional
        Peak of the blue band relative to the phosphor band.

    Returns
    -------
    array
        Peak normalized to 1.
    """

    wl = 2*np.pi/np.asarray(k, dtype=float)
    s = (blue_weight*np.exp(-4*np.log(2)*(wl - blue)**2/blue_fwhm**2)
         + np.exp(-4*np.log(2)*(wl - phosphor)**2/phosphor_fwhm**2))*wl**2  # d lambda/dk
    return s/s.max()


def coherence_length(center_wavelength, fwhm_wavelength):
    """FWHM of the envelope of a correlogram for a Gaussian spectrum, in
    scan distance: half the round trip coherence length,
    2 ln2/pi lambda**2/Delta_lambda (`oct.axial_resolution()`).

    Parameters
    ----------
    center_wavelength, fwhm_wavelength : numbers

    Returns
    -------
    number
    """

    return 2*np.log(2)/np.pi*center_wavelength**2/fwhm_wavelength


def pupil_quadrature(NA, n=16):
    """Nodes cos(theta) and weights for a uniformly filled pupil up to NA
    (uniform in sin(theta)**2, i.e. in the area of the pupil).

    Parameters
    ----------
    NA : number
        Numerical aperture (0 gives a single node cos(theta) = 1).
    n : int, optional
        Number of Gauss-Legendre nodes.

    Returns
    -------
    cos_theta, weights : arrays
        Weights sum to 1.
    """

    if NA <= 0:
        return np.array([1.0]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    s2 = 0.5*(x + 1)*NA**2
    return np.sqrt(1 - s2), w/np.sum(w)


def obliquity_factor(NA):
    """Mean cos(theta) of a uniformly filled pupil,
    2 (1 - cos(alpha)**3)/(3 sin(alpha)**2): the fringe period of a
    correlogram is lambda/(2 obliquity_factor(NA)).

    Parameters
    ----------
    NA : number or array

    Returns
    -------
    number or array
    """

    NA = np.asarray(NA, dtype=float)
    c = np.sqrt(1 - NA**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2*(1 - c**3)/(3*NA**2)
    return np.where(NA > 1e-6, f, 1 - NA**2/4)


def film_reflectance(layers, substrate, ambient=1.0):
    """Reflection coefficients of a layered surface for `correlogram()`.

    Parameters
    ----------
    layers : sequence of (index, thickness)
        From the top down; index complex (n + i k) or a callable of the
        wavelength.
    substrate, ambient : number or callable of the wavelength

    Returns
    -------
    [r_s, r_p] : list of two callables
        r(k, cos_theta) for the vacuum wavenumber k and an array of the
        cosines of the angles in the ambient; r_p is the ratio of the
        tangential fields (r_p = r_s at normal incidence). Passing the list
        to `correlogram()` averages the two polarizations (unpolarized
        light).
    """

    def idx(n, wl):
        return complex(n(wl) if callable(n) else n)

    def make(pol):
        def r(k, cos_theta):
            wl = 2*np.pi/k
            n_all = np.array([idx(ambient, wl)] + [idx(n, wl) for n, _ in layers]
                             + [idx(substrate, wl)])
            sin0 = np.sqrt(np.clip(1 - np.asarray(cos_theta, dtype=float)**2, 0, None))
            return _r_t(sin0, n_all, [float(d) for _, d in layers], pol, wl)[0]
        return r

    return [make("s"), make("p")]


def _reflectances(reflectance):
    if reflectance is None:
        return [lambda k, c: np.ones(np.shape(c), dtype=complex)]
    if callable(reflectance):
        return [reflectance]
    if np.isscalar(reflectance):
        return [lambda k, c, r=complex(reflectance): np.full(np.shape(c), r)]
    return list(reflectance)


def correlogram(z, k, spectrum, height=0.0, NA=0.0, reflectance=None, r_ref=1.0, n_pupil=16):
    """Correlograms of surfaces at the heights h for the scan positions z.

    Parameters
    ----------
    z : 1d array
        Scan positions (the height at which the paths are equal).
    k : 1d array
        Vacuum wavenumbers of the source.
    spectrum : 1d array
        Power spectrum S(k).
    height : number or array, optional
        Surface heights (any shape).
    NA : number, optional
        Numerical aperture of the illumination (uniformly filled pupil).
    reflectance : None, number, callable or list of callables, optional
        Reflection coefficient r_S of the surface: a constant, or
        r(k, cos_theta) (`film_reflectance()`); a list is averaged
        incoherently (polarizations). None is 1.
    r_ref : number, optional
        Reflection coefficient of the reference.
    n_pupil : int, optional
        Nodes of `pupil_quadrature()`.

    Returns
    -------
    I : array, shape height.shape + (len(z),)
        Normalized so that the DC level is (abs(r_R)**2 + abs(r_S)**2)
        averaged over the spectrum.
    """

    z = np.asarray(z, dtype=float)
    h = np.asarray(height, dtype=float)[..., None]
    cos_t, w_t = pupil_quadrature(NA, n_pupil)
    refl = _reflectances(reflectance)
    S = np.asarray(spectrum, dtype=float)
    out = np.zeros(h.shape[:-1] + z.shape)
    for kk, s in zip(np.asarray(k, dtype=float), S):
        if s == 0:
            continue
        for r_fn in refl:
            r = r_fn(kk, cos_t)
            dc = np.sum(w_t*(abs(r_ref)**2 + np.abs(r)**2))
            out += s*dc/len(refl)
            for c, w, rr in zip(cos_t, w_t, r):
                out += s*w/len(refl)*2*np.real(np.conj(r_ref)*rr*np.exp(2j*kk*c*(z - h)))
    return out/np.sum(S)


def envelope(signal):
    """Envelope of a correlogram along its last axis: the modulus of the
    analytic signal of the signal minus its mean.

    Parameters
    ----------
    signal : array

    Returns
    -------
    array
    """

    s = np.asarray(signal, dtype=float)
    return np.abs(hilbert(s - s.mean(axis=-1, keepdims=True), axis=-1))


def envelope_peak(z, signal, method="centroid", threshold=0.3):
    """Height from the peak of the envelope of correlograms (last axis).

    Parameters
    ----------
    z : 1d array
        Scan positions, equidistant.
    signal : array, shape (..., len(z))
    method : {"centroid", "parabola", "max"}, optional
        Centroid of the squared envelope above threshold*max, a least
        squares parabola through the logarithm of the envelope above
        threshold*max around the maximum (a Gaussian fit, exact for a
        Gaussian envelope), or the sample at the maximum.
    threshold : number, optional
        For "centroid" and "parabola".

    Returns
    -------
    array
        Heights, shape signal.shape[:-1].
    """

    z = np.asarray(z, dtype=float)
    env = envelope(signal)
    i = np.argmax(env, axis=-1)
    if method == "max":
        return z[i]
    if method == "parabola":
        # weighted least squares fit of a + b u + c u**2 to log(env), u = z - z_max,
        # over the samples above threshold*max (weights env**2)
        u = z - z[i][..., None]
        w = np.where(env >= threshold*env.max(axis=-1, keepdims=True), env**2, 0.0)
        g = np.log(np.clip(env, 1e-300, None))
        M = np.stack([np.stack([np.sum(w*u**(a + b), axis=-1) for b in range(3)], axis=-1)
                      for a in range(3)], axis=-2)
        r = np.stack([np.sum(w*g*u**a, axis=-1) for a in range(3)], axis=-1)
        c = np.linalg.solve(M, r[..., None])[..., 0]
        off = np.where(c[..., 2] < 0, -c[..., 1]/(2*np.where(c[..., 2] < 0, c[..., 2], -1.0)), 0.0)
        return z[i] + off
    if method == "centroid":
        e2 = env**2
        mask = env >= threshold*env.max(axis=-1, keepdims=True)
        w = np.where(mask, e2, 0.0)
        return np.sum(w*z, axis=-1)/np.sum(w, axis=-1)
    raise ValueError("method must be 'centroid', 'parabola' or 'max'")


def fourier_signal(z, signal, K):
    """Fourier transform of correlograms at the spatial frequencies K,
    F(K) = sum_z (I(z) - mean) exp(i K z), so that a surface at the height
    h has the phase K h - arg(r_S) (the positive frequency half of the
    fringe term r_S* exp(-i K (z - h)) + c.c.).

    Parameters
    ----------
    z : 1d array
    signal : array, shape (..., len(z))
    K : 1d array
        Spatial frequencies (2 k cos(theta)).

    Returns
    -------
    complex array, shape (..., len(K))
    """

    s = np.asarray(signal, dtype=float)
    s = s - s.mean(axis=-1, keepdims=True)
    return s @ np.exp(1j*np.outer(np.asarray(z, dtype=float), K))


def fda(z, signal, K_range, n_K=64, power_threshold=0.1, phase_gap=0.0):
    """Frequency domain analysis of correlograms (de Groot and Deck).

    Parameters
    ----------
    z : 1d array
        Scan positions.
    signal : array, shape (..., len(z))
    K_range : (number, number)
        Band of spatial frequencies 2 k cos(theta) to analyse, around the
        fringe frequency (e.g. 2 k_min cos(alpha) to 2 k_max).
    n_K : int, optional
        Frequencies in the band.
    power_threshold : number, optional
        Frequencies with less than this fraction of the peak power are
        left out of the fit.
    phase_gap : number, optional
        Phase of r_S/r_R (the "phase gap" of de Groot et al.: the phase on
        reflection of the surface material and of the instrument), added
        to the phase before the fringe order is chosen; pi for glass (r_S
        = -0.2) against a reference with r_R > 0. Without it the phase
        height is offset by -phase_gap/K0 and the fringe order decision is
        biased.

    Returns
    -------
    h_coherence, h_phase, K0 : arrays
        The coherence height (slope of the phase against K), the phase
        height with its fringe order taken from the coherence height, and
        the power weighted mean frequency K0.
    """

    z = np.asarray(z, dtype=float)
    zc = 0.5*(z[0] + z[-1])
    K = np.linspace(K_range[0], K_range[1], n_K)
    F = fourier_signal(z - zc, signal, K)
    P = np.abs(F)**2
    w = np.where(P >= power_threshold*P.max(axis=-1, keepdims=True), P, 0.0)
    phi = np.unwrap(np.angle(F*np.exp(1j*phase_gap)), axis=-1)
    W = np.sum(w, axis=-1)
    K0 = np.sum(w*K, axis=-1)/W
    dK = K - K0[..., None]
    phi0 = np.sum(w*phi, axis=-1)/W
    slope = np.sum(w*dK*(phi - phi0[..., None]), axis=-1)/np.sum(w*dK**2, axis=-1)
    h_coh = slope
    # phase at K0, with its 2 pi order chosen to agree with the coherence height
    resid = phi0 - K0*h_coh
    resid = np.mod(resid + np.pi, 2*np.pi) - np.pi
    h_phase = h_coh + resid/K0
    return h_coh + zc, h_phase + zc, K0


def profile_scan(x, height, z, k, spectrum, NA, NA_illumination=None, n_illumination=9,
                 reflectance=1.0):
    """Correlograms of a line profile h(x) with scalar imaging.

    Each plane wave of the illumination (transverse wavenumber q up to
    k NA_illumination) is reflected by the surface in the thin element
    approximation, r exp(i q x) exp(-2 i k_z h(x)), its angular spectrum is
    propagated to the focal plane z of the objective, cut at the NA and
    imaged; the reference is a mirror at the height z. The intensities of
    all wavenumbers and illumination angles add. The profile is periodic
    in x (pad it if necessary).

    Parameters
    ----------
    x : 1d array
        Equidistant positions.
    height : 1d array
        Surface heights h(x).
    z : 1d array
        Scan positions.
    k, spectrum : 1d arrays
        Wavenumbers and power spectrum.
    NA : number
        Numerical aperture of the objective (imaging).
    NA_illumination : number, optional
        Of the illumination (defaults to NA).
    n_illumination : int, optional
        Illumination angles, uniform in sin(theta) (a 1D pupil).
    reflectance : number or array, optional
        Reflection coefficient r of the surface (may vary along x).

    Returns
    -------
    I : array, shape (len(z), len(x))
    """

    x = np.asarray(x, dtype=float)
    h = np.asarray(height, dtype=float)
    z = np.asarray(z, dtype=float)
    NA_ill = NA if NA_illumination is None else NA_illumination
    dx = x[1] - x[0]
    kx = 2*np.pi*np.fft.fftfreq(len(x), dx)
    s_ill = np.linspace(-NA_ill, NA_ill, n_illumination) if n_illumination > 1 else np.zeros(1)
    r = np.asarray(reflectance, dtype=complex)*np.ones_like(x)
    I = np.zeros((len(z), len(x)))
    S = np.asarray(spectrum, dtype=float)
    for kk, s in zip(np.asarray(k, dtype=float), S):
        if s == 0:
            continue
        kz = complex_sqrt(kk**2 - kx**2)
        pupil = np.abs(kx) <= kk*NA
        prop = np.exp(1j*np.outer(z, kz))*pupil
        for sj in s_ill:
            q = kk*sj
            q = kx[np.argmin(np.abs(kx - q))]  # on the grid: periodic illumination
            kz_in = np.sqrt(kk**2 - q**2)
            u0 = r*np.exp(1j*q*x)*np.exp(-2j*kz_in*h)
            img = np.fft.ifft(np.fft.fft(u0)[None, :]*prop, axis=-1)
            ref = np.exp(1j*q*x)[None, :]*np.exp(-1j*kz_in*z)[:, None]
            I += s/len(s_ill)*np.abs(ref + img)**2
    return I/np.sum(S)
