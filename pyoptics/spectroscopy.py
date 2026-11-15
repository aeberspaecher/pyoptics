#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Spectroscopy: line shapes, absorption, instrument functions, Fourier
transform spectrometers, Fabry-Perot interferometers and birefringent
filters (notebooks 110 to 114).

The functions do not care about units as long as they are used
consistently; the notebooks use wavenumbers nu in cm^-1 with path
differences in cm (Fourier transform spectroscopy), and wavelengths in nm
with thicknesses in nm (filters).

Line shapes (all normalized to unit area, `coherence.line_spectrum()` has
the same shapes with the peak 1) ::

    Lorentz:  L(nu) = (gamma/2pi) / ((nu - nu0)**2 + (gamma/2)**2),  FWHM gamma,
    Gauss:    G(nu) = sqrt(4 ln2/pi)/w exp(-4 ln2 (nu - nu0)**2/w**2),  FWHM w,
    Voigt:    their convolution, Re w(z) of the Faddeeva function.

The Doppler FWHM of a gas of molecular mass m at the temperature T is
nu0 sqrt(8 k T ln2/(m c**2)).

A Lorentz oscillator gives the complex index near a resonance ::

    n**2 = eps_inf + f nu_p**2/(nu0**2 - nu**2 - i gamma nu),

with absorption and anomalous dispersion linked by the Kramers-Kronig
relation ::

    n(nu) - n_inf = (2/pi) P int_0^inf nu' kappa(nu')/(nu'**2 - nu**2) dnu'.

Beer-Lambert: transmittance T = exp(-tau), optical depth tau = sigma N L,
absorbance A = -log10 T. The equivalent width W = int (1 - T) dnu of a
line grows as tau_0 (weak), as sqrt(ln tau_0) (saturated core) and as
sqrt(tau_0) (Lorentzian wings): the curve of growth.

Fourier transform spectroscopy. A Michelson with the optical path
difference x records ::

    I(x) = int S(nu) (1 + cos(2 pi nu x)) dnu,

and the spectrum is the cosine transform of the modulated part. A scan to
x = L multiplies it by the window A(x/L), so the measured spectrum is the
true one convolved with the instrument line shape (ILS), the transform of
A. Boxcar: 2L sinc(2 nu L), FWHM 1.207/(2L); see `apodization()` for the
other windows. A finite aperture of solid angle Omega shifts and widens
a line at nu by nu Omega/(4 pi) and nu Omega/(2 pi); the resolving power
R = nu/dnu allows Omega = 2 pi/R (Jacquinot).

Fabry-Perot of mirror reflectance R (intensity), absorptance A and
round trip phase delta = 4 pi n d cos(theta) nu ::

    T = (1 - R - A)**2/(1 - R)**2 / (1 + F sin**2(delta/2)),
    F = 4R/(1 - R)**2,  reflective finesse N_R = pi sqrt(R)/(1 - R),

with the free spectral range 1/(2 n d) in wavenumbers and the resolving
power m N. Flatness errors of rms lambda/M limit the finesse to about
M/2 (`effective_finesse()` adds the inverse squares).

Lyot filter: stages of thickness d, 2d, ..., 2**(N - 1) d of birefringence
Delta n between parallel polarizers at 45 degrees to the optic axes ::

    T = prod_k cos**2(pi Delta n 2**k d/lambda),

FSR lambda**2/(Delta n d) set by the thinnest plate, FWHM 0.886
lambda**2/(2**N Delta n d) (for N >= 4) set by the thickest.

References
----------
P. R. Griffiths and J. A. de Haseth, "Fourier Transform Infrared
Spectrometry", 2nd ed., Wiley (2007).

R. H. Norton and R. Beer, "New apodizing functions for Fourier
spectrometry", J. Opt. Soc. Am. 66, 259-264 (1976); erratum 67, 419 (1977).

L. Mertz, "Auxiliary computation for Fourier spectrometry", Infrared
Phys. 7, 17-23 (1967).

P. Jacquinot, "The luminosity of spectrometers with prisms, gratings, or
Fabry-Perot etalons", J. Opt. Soc. Am. 44, 761-765 (1954).

J. M. Vaughan, "The Fabry-Perot Interferometer", Adam Hilger (1989).

B. Lyot, "Le filtre monochromatique polarisant et ses applications en
physique solaire", Ann. Astrophys. 7, 31-79 (1944).

J. J. Olivero and R. L. Longbothum, "Empirical fits to the Voigt line
width: a brief review", J. Quant. Spectrosc. Radiat. Transfer 17, 233-236
(1977).

K. Ohta and H. Ishida, "Comparison among several numerical integration
methods for Kramers-Kronig transformation", Appl. Spectrosc. 42, 952-957
(1988).
"""

import numpy as np
from numpy import pi
from scipy.signal import fftconvolve
from scipy.special import wofz

from .radiometry import H, C, K_B

AMU = 1.66053906660e-27  # kg


# line shapes:
def lorentzian(nu, nu0, fwhm):
    """Lorentzian of unit area and the FWHM fwhm centred at nu0."""

    nu = np.asarray(nu, dtype=float)

    return fwhm/(2*pi)/((nu - nu0)**2 + (fwhm/2)**2)


def gaussian(nu, nu0, fwhm):
    """Gaussian of unit area and the FWHM fwhm centred at nu0."""

    nu = np.asarray(nu, dtype=float)
    a = 4*np.log(2)/fwhm**2

    return np.sqrt(a/pi)*np.exp(-a*(nu - nu0)**2)


def voigt(nu, nu0, fwhm_gauss, fwhm_lorentz):
    """Voigt profile (Gaussian convolved with Lorentzian) of unit area.

    Parameters
    ----------
    nu : array
    nu0 : number
        Line centre.
    fwhm_gauss, fwhm_lorentz : numbers
        FWHM of the Gaussian and the Lorentzian parts (either may be 0).

    Returns
    -------
    V : array
    """

    nu = np.asarray(nu, dtype=float)
    if fwhm_gauss == 0:
        return lorentzian(nu, nu0, fwhm_lorentz)
    sigma = fwhm_gauss/(2*np.sqrt(2*np.log(2)))
    z = ((nu - nu0) + 0.5j*fwhm_lorentz)/(sigma*np.sqrt(2))

    return wofz(z).real/(sigma*np.sqrt(2*pi))


def voigt_fwhm(fwhm_gauss, fwhm_lorentz):
    """FWHM of a Voigt profile (Olivero and Longbothum, 0.02 %)."""

    fl, fg = fwhm_lorentz, fwhm_gauss

    return 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)


def doppler_fwhm(nu0, T, mass_amu):
    """Doppler FWHM nu0 sqrt(8 k T ln2/(m c^2)) of a line at nu0 (any unit).

    Parameters
    ----------
    nu0 : number or array
        Line centre (frequency, wavenumber or wavelength).
    T : number
        Temperature in K.
    mass_amu : number
        Molecular mass in atomic mass units.

    Returns
    -------
    fwhm : float or array
        In the unit of nu0.
    """

    return nu0*np.sqrt(8*K_B*T*np.log(2)/(mass_amu*AMU*C**2))


def lorentz_index(nu, nu0, strength, gamma, eps_inf=1.0):
    """Complex refractive index n + i kappa of a Lorentz oscillator.

    n**2 = eps_inf + strength nu0**2/(nu0**2 - nu**2 - i gamma nu), the
    pyoptics convention exp(-i omega t) (kappa > 0 absorbs).

    Parameters
    ----------
    nu : array
        Frequency or wavenumber.
    nu0 : number
        Resonance.
    strength : number
        Oscillator strength (the step of eps across the resonance).
    gamma : number
        Damping (FWHM of the absorption of a weak line).
    eps_inf : number, optional
        Background permittivity.

    Returns
    -------
    n : complex array
    """

    nu = np.asarray(nu, dtype=float)
    eps = eps_inf + strength*nu0**2/(nu0**2 - nu**2 - 1j*gamma*nu)

    return np.sqrt(eps)


def kramers_kronig(nu, kappa):
    """Real index minus its value at infinity from the extinction kappa.

    n(nu) - n_inf = (2/pi) P int_0^inf nu' kappa(nu')/(nu'^2 - nu^2) dnu',
    evaluated with Maclaurin's method (Ohta and Ishida): on a uniform grid
    the sum runs over the points of the other parity, which avoids the
    pole. kappa must be negligible outside the grid.

    Parameters
    ----------
    nu : array
        Uniform grid (> 0).
    kappa : array
        Extinction coefficient on the grid.

    Returns
    -------
    dn : array
    """

    nu = np.asarray(nu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    h = nu[1] - nu[0]
    idx = np.arange(len(nu))
    dn = np.empty(len(nu))
    for i in range(len(nu)):
        j = idx[(idx % 2) != (i % 2)]
        dn[i] = (4*h/pi)*np.sum(nu[j]*kappa[j]/(nu[j]**2 - nu[i]**2))

    return dn


# absorption:
def transmittance(optical_depth):
    """Beer-Lambert transmittance exp(-tau)."""

    return np.exp(-np.asarray(optical_depth, dtype=float))


def absorbance(T):
    """Decadic absorbance -log10(T)."""

    return -np.log10(np.asarray(T, dtype=float))


def equivalent_width(nu, T):
    """Equivalent width int (1 - T) dnu of an absorption line or band."""

    return np.trapezoid(1 - np.asarray(T, dtype=float), np.asarray(nu, dtype=float))


def linear_molecule_band(nu0, B_lower, B_upper, T, J_max=80, even_only=False,
                         strength=1.0):
    """Line list of a rovibrational band of a linear molecule (rigid rotor,
    a parallel band with P and R branches, no Q branch).

    Lower state energies B'' J(J + 1), line positions
    R(J): nu0 + B'(J + 1)(J + 2) - B'' J(J + 1),
    P(J): nu0 + B'(J - 1) J - B'' J(J + 1),
    intensities proportional to the Hönl-London factors (J + 1 for R(J),
    J for P(J)) times the Boltzmann population exp(-E''/kT) (the
    degeneracy 2J + 1 cancels against the Hönl-London normalization), and
    a common factor such that all lines sum to `strength`. even_only keeps
    the even J of the lower state (CO2: the nuclear spin statistics of
    two O-16 nuclei leave every second line out).

    Parameters
    ----------
    nu0 : number
        Band origin (cm^-1).
    B_lower, B_upper : numbers
        Rotational constants (cm^-1).
    T : number
        Temperature (K).
    J_max : int, optional
    even_only : bool, optional
    strength : number, optional
        Sum of the line strengths.

    Returns
    -------
    positions, strengths : arrays
        Line centres (cm^-1) and strengths, P lines first, then R.
    branch : array of str
        "P(J)" and "R(J)" labels.
    """

    hc_k = H*C*100/K_B  # cm K (second radiation constant)
    J = np.arange(J_max + 1)
    if even_only:
        J = J[J % 2 == 0]
    E = B_lower*J*(J + 1)
    pop = np.exp(-E*hc_k/T)
    nu_R = nu0 + B_upper*(J + 1)*(J + 2) - E
    S_R = (J + 1)*pop
    Jp = J[J >= 1]
    nu_P = nu0 + B_upper*(Jp - 1)*Jp - B_lower*Jp*(Jp + 1)
    S_P = Jp*np.exp(-B_lower*Jp*(Jp + 1)*hc_k/T)
    positions = np.concatenate([nu_P, nu_R])
    S = np.concatenate([S_P, S_R])
    labels = np.array(["P({})".format(j) for j in Jp] + ["R({})".format(j) for j in J])

    return positions, strength*S/S.sum(), labels


def line_list_spectrum(nu, positions, strengths, fwhm_gauss=0.0, fwhm_lorentz=0.0):
    """Sum of Voigt lines: sum_i S_i V(nu - nu_i) (e.g. a cross section)."""

    nu = np.asarray(nu, dtype=float)
    out = np.zeros_like(nu)
    for p, s in zip(positions, strengths):
        out += s*voigt(nu, p, fwhm_gauss, fwhm_lorentz)

    return out


# instrument functions:
def slit_function(x, width_entrance, width_exit=0.0):
    """Instrument function of a dispersive spectrometer with perfect optics:
    the image of the entrance slit (width_entrance) convolved with the exit
    slit or pixel (width_exit), a trapezoid (a triangle for equal widths),
    of unit area. x and the widths in the same (spectral) unit."""

    x = np.asarray(x, dtype=float)
    a, b = max(width_entrance, width_exit), min(width_entrance, width_exit)
    if b == 0:
        return (np.abs(x) <= a/2)/a
    y = np.clip(((a + b)/2 - np.abs(x))/b, 0.0, 1.0)

    return y/a


def convolve(x, y, kernel):
    """Convolve samples y on a uniform grid x with kernel(dx), a function of
    the offset of unit area (e.g. lambda dx: slit_function(dx, w)).

    The ends are padded with the edge values.

    Returns
    -------
    y_conv : array
    """

    x = np.asarray(x, dtype=float)
    h = x[1] - x[0]
    n = len(x)
    dx = (np.arange(2*n - 1) - (n - 1))*h
    k = kernel(dx)*h
    yp = np.concatenate([np.full(n - 1, y[0]), y, np.full(n - 1, y[-1])])

    return fftconvolve(yp, k, mode="same")[n - 1:2*n - 1]


# Fourier transform spectroscopy:
_NB = {"norton_beer_weak": (0.384093, -0.087577, 0.703484, 0.0),
       "norton_beer_medium": (0.152442, -0.136176, 0.983734, 0.0),
       "norton_beer_strong": (0.045335, 0.0, 0.554883, 0.399782)}

APODIZATIONS = ("boxcar", "triangle", "happ_genzel", "norton_beer_weak", "norton_beer_medium",
                "norton_beer_strong", "blackman_harris")


def apodization(x, L, kind="boxcar"):
    """Apodization window A(x) on the path differences |x| <= L (zero
    beyond).

    kind : "boxcar", "triangle", "happ_genzel" (0.54 + 0.46 cos(pi x/L)),
        "norton_beer_weak", "norton_beer_medium", "norton_beer_strong"
        (sum_i C_i (1 - (x/L)^2)^i), "blackman_harris" (3 term).
    """

    u = np.abs(np.asarray(x, dtype=float))/L
    inside = u <= 1
    if kind == "boxcar":
        A = np.ones_like(u)
    elif kind == "triangle":
        A = 1 - u
    elif kind == "happ_genzel":
        A = 0.54 + 0.46*np.cos(pi*u)
    elif kind in _NB:
        q = 1 - u**2
        A = sum(c*q**p for p, c in zip((0, 1, 2, 4), _NB[kind]))
    elif kind == "blackman_harris":
        A = 0.42323 + 0.49755*np.cos(pi*u) + 0.07922*np.cos(2*pi*u)
    else:
        raise ValueError("Unknown apodization {}.".format(kind))

    return np.where(inside, A, 0.0)


def instrument_line_shape(nu, L, kind="boxcar", n=4001):
    """Instrument line shape of a Fourier transform spectrometer with the
    maximum path difference L: the cosine transform of the window,
    normalized to unit area.

    Parameters
    ----------
    nu : array
        Offsets from the line (reciprocal unit of L).
    L : number
    kind : str, optional
        See `apodization()`.
    n : int, optional
        Samples of the window.

    Returns
    -------
    ils : array
    """

    x = np.linspace(0, L, n)
    A = apodization(x, L, kind)
    nu = np.asarray(nu, dtype=float)
    ils = 2*np.trapezoid(A[None, :]*np.cos(2*pi*np.outer(nu.ravel(), x)), x, axis=1)

    # unit area, as A(0) = 1 for all windows
    return ils.reshape(nu.shape)


def fwhm(x, y):
    """FWHM of a single peak sampled on x (linear interpolation)."""

    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    half = y[i]/2
    j = i
    while j > 0 and y[j] > half:
        j -= 1
    k = i
    while k < len(y) - 1 and y[k] > half:
        k += 1
    xl = x[j] + (half - y[j])*(x[j + 1] - x[j])/(y[j + 1] - y[j])
    xr = x[k - 1] + (half - y[k - 1])*(x[k] - x[k - 1])/(y[k] - y[k - 1])

    return xr - xl


def interferogram(x, nu, spectrum, phase=None):
    """Modulated part of a two-beam interferogram, sum_nu S cos(2 pi nu x + phi).

    Parameters
    ----------
    x : array
        Optical path differences.
    nu : array
        Wavenumbers of the spectrum samples (uniform grid, the sum is
        weighted with the grid step).
    spectrum : array
        Spectral density S(nu).
    phase : array, optional
        Spectral phase phi(nu) (beam splitter dispersion, offset of the
        zero path difference).

    Returns
    -------
    I : array
    """

    x = np.asarray(x, dtype=float)
    nu = np.asarray(nu, dtype=float)
    dnu = nu[1] - nu[0]
    ph = 0.0 if phase is None else np.asarray(phase)[None, :]
    out = np.empty(x.shape)
    for s in range(0, x.size, 2048):
        xx = x.ravel()[s:s + 2048]
        out.ravel()[s:s + 2048] = np.sum(spectrum[None, :]*np.cos(2*pi*np.outer(xx, nu) + ph),
                                         axis=1)*dnu

    return out


def fts_spectrum(x, I, kind="boxcar", n_fft=None):
    """Spectrum from a symmetric (double sided) interferogram.

    Parameters
    ----------
    x : array
        Uniform path differences from -L to L (x = 0 included).
    I : array
        Modulated interferogram.
    kind : str, optional
        Apodization (`apodization()` with L = max |x|).
    n_fft : int, optional
        FFT length (zero filling).

    Returns
    -------
    nu, S : arrays
        Wavenumbers from 0 to the Nyquist limit 1/(2 dx) and the spectrum
        (the real part; its scale matches `interferogram()`'s input
        S(nu) for a symmetric interferogram).
    """

    x = np.asarray(x, dtype=float)
    h = x[1] - x[0]
    L = np.max(np.abs(x))
    y = np.asarray(I, dtype=float)*apodization(x, L, kind)
    n_fft = len(x) if n_fft is None else int(n_fft)
    i0 = int(np.argmin(np.abs(x)))
    buf = np.zeros(n_fft)
    m = len(x)
    # put x = 0 first (wrap the negative path differences to the end):
    buf[:m - i0] = y[i0:]
    buf[n_fft - i0:] = y[:i0]
    S = np.fft.rfft(buf).real*h
    nu = np.fft.rfftfreq(n_fft, h)

    return nu, 2*S


def mertz_spectrum(x, I, L_phase, kind="norton_beer_medium", n_fft=None):
    """Spectrum of a one-sided interferogram with Mertz phase correction.

    The interferogram runs from -L_phase to L (L >> L_phase). The phase
    phi(nu) is computed from the short double sided part |x| <= L_phase
    (triangle apodized), the whole interferogram is weighted with Mertz's
    ramp (linear from 0 at -L_phase to 1 at +L_phase, so the doubly
    measured part counts once) and the apodization, transformed, and the
    complex spectrum multiplied by exp(-i phi).

    Parameters
    ----------
    x : array
        Uniform path differences, x = 0 included.
    I : array
    L_phase : number
        Half length of the double sided part.
    kind : str, optional
        Apodization of the full scan (L = max x).
    n_fft : int, optional

    Returns
    -------
    nu, S, phase : arrays
        Phase corrected spectrum and the measured phase.
    """

    x = np.asarray(x, dtype=float)
    I = np.asarray(I, dtype=float)
    h = x[1] - x[0]
    L = x.max()
    n_fft = len(x) if n_fft is None else int(n_fft)
    n_fft = max(n_fft, 2*len(x))
    i0 = int(np.argmin(np.abs(x)))

    def wrap(y):
        buf = np.zeros(n_fft)
        buf[:len(x) - i0] = y[i0:]
        buf[n_fft - i0:] = y[:i0]
        return buf

    short = np.abs(x) <= L_phase
    S_short = np.fft.rfft(wrap(np.where(short, I*apodization(x, L_phase, "triangle"), 0.0)))
    phase = np.unwrap(np.angle(S_short))
    ramp = np.clip((x + L_phase)/(2*L_phase), 0.0, 1.0)
    S_full = np.fft.rfft(wrap(I*ramp*apodization(x, L, kind)))
    S = (S_full*np.exp(-1j*phase)).real*h
    nu = np.fft.rfftfreq(n_fft, h)

    return nu, 4*S, phase


def jacquinot_solid_angle(resolving_power):
    """Largest solid angle 2 pi/R of a Fourier transform (or Fabry-Perot)
    spectrometer of the resolving power R."""

    return 2*pi/resolving_power


# Fabry-Perot:
def airy_transmission(delta, R, A=0.0):
    """Fabry-Perot transmission at the round trip phase delta.

    Parameters
    ----------
    delta : array
        4 pi n d cos(theta) nu (+ mirror phases).
    R, A : numbers
        Reflectance and absorptance of each mirror.

    Returns
    -------
    T : array
    """

    F = 4*R/(1 - R)**2

    return ((1 - R - A)/(1 - R))**2/(1 + F*np.sin(np.asarray(delta)/2)**2)


def reflective_finesse(R):
    """Finesse pi sqrt(R)/(1 - R) of ideal mirrors of reflectance R."""

    return pi*np.sqrt(R)/(1 - R)


def effective_finesse(*finesses):
    """Combined finesse 1/sqrt(sum 1/N_i^2) (reflective, flatness,
    aperture)."""

    return 1/np.sqrt(sum(1/np.asarray(f, dtype=float)**2 for f in finesses))


def free_spectral_range(thickness, n=1.0, theta=0.0):
    """Free spectral range 1/(2 n d cos(theta)) in wavenumbers (reciprocal
    unit of thickness)."""

    return 1/(2*n*thickness*np.cos(theta))


# birefringent filters:
def lyot_transmission(wavelength, thickness, birefringence, n_stages):
    """Lyot filter: prod_k cos^2(pi Delta n 2^k d/lambda), k = 0 .. N - 1.

    Parameters
    ----------
    wavelength : array
        Same unit as thickness.
    thickness : number
        Thinnest plate d.
    birefringence : number or callable
        Delta n, or a function Delta n(wavelength).
    n_stages : int

    Returns
    -------
    T : array
        Transmission for unpolarized light times 2 (the first polarizer is
        not counted).
    """

    lam = np.asarray(wavelength, dtype=float)
    dn = birefringence(lam) if callable(birefringence) else birefringence
    T = np.ones_like(lam)
    for k in range(n_stages):
        T = T*np.cos(pi*dn*thickness*2**k/lam)**2

    return T


# resolving powers:
def prism_resolving_power(base, dn_dlambda):
    """Resolving power b |dn/dlambda| of a prism of base b used full."""

    return base*np.abs(dn_dlambda)


def grating_resolving_power(order, n_grooves):
    """Resolving power |m| N of a grating."""

    return np.abs(order)*n_grooves
