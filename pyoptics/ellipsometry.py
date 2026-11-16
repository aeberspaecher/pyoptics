#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ellipsometry: Psi and Delta of layered samples, dispersion models,
ellipsometers and model fits.

An ellipsometer measures how a sample changes the polarization of light
reflected under an oblique angle. With the reflection coefficients r_p and
r_s of the sample it measures the ratio ::

    rho = r_p/r_s = tan(Psi) exp(i Delta).

Psi and Delta are ratios, so the intensity of the source and the
transmission of the optics drop out, and Delta is a phase difference that
resolves changes in a film far below a nanometre.

Conventions
-----------
Ellipsometry has its own sign conventions, the "Nebraska conventions" of
Muller (1969) used by Azzam and Bashara, Tompkins and Fujiwara: r_p with
the reference directions of Born & Wolf (r_p = -r_s at normal incidence)
and a time dependence exp(+i omega t). pyoptics uses exp(-i omega t) and
`thin_films` gives r_p as a ratio of the tangential fields (r_p = r_s at
normal incidence). This module converts (`rho_from_coefficients()`) ::

    rho = conj(-r_p,tangential/r_s),

so Psi and Delta come out as in the literature and in instrument software:
a bare silicon wafer has Delta close to 180 degrees, a native oxide lowers
it, and Delta lies in [0, 360) degrees. Angles of Psi, Delta and of the
incidence are in degrees throughout, as is usual in ellipsometry; lengths
(wavelengths, thicknesses) in any common unit, the notebooks use um.
Refractive indices are complex, n + i k with k > 0 (pyoptics convention);
`pseudo_dielectric()` returns epsilon_1 + i epsilon_2 with epsilon_2 > 0.

Samples
-------
A sample is an ambient medium, a list of layers (index, thickness) from
the ambient side down and a substrate (`rho()`, `psi_delta()`). An index
is a number or a callable of the wavelength, e.g. `cauchy()`,
`sellmeier()`, `silicon_index()`, `sio2_index()`, or a function built from
the dielectric models: `tauc_lorentz()` (amorphous semiconductors and
dielectrics, Jellison and Modine), `lorentz()` and `drude()`, combined with
`index_from_eps()`; `bruggeman()` mixes two media into an effective medium
(roughness layers, porous films).

Instruments
-----------
The sample has the normalized Mueller matrix (`sample_mueller()`) ::

    [[1, -N, 0, 0], [-N, 1, 0, 0], [0, 0, C, S], [0, 0, -S, C]],
    N = cos(2 Psi), C = sin(2 Psi) cos(Delta), S = sin(2 Psi) sin(Delta).

`ellipsometer_signal()` simulates the detector signal of a
polarizer-(compensator)-sample-(compensator)-analyzer ellipsometer for any
angles of its elements, e.g. a rotating analyzer or a rotating
compensator. The signal is linear in (1, N, C, S), so `fit_ncs()` recovers
them by linear least squares from any set of measured intensities. A
rotating analyzer ellipsometer cannot see S: it measures cos(Delta) only,
which loses the sign of Delta and is insensitive near Delta = 0 and 180
degrees; a rotating compensator measures S as well. `psi_delta_from_ncs()`
converts back and reports the degree of polarization sqrt(N**2 + C**2 +
S**2), below 1 for depolarizing samples or a finite bandwidth.

Model fits
----------
`fit_psi_delta()` fits the parameters of any model of Psi and Delta to
measured spectra by Levenberg-Marquardt and returns the parameters, their
standard errors and correlations and the mean squared error ::

    MSE = 1/(2 n - m) sum [((Psi_mod - Psi)/sigma_Psi)**2
                           + ((Delta_mod - Delta)/sigma_Delta)**2]

of Jellison (n wavelengths and angles, m parameters), which is about 1 for
a model that fits within the noise.

References
----------
R. M. A. Azzam and N. M. Bashara, "Ellipsometry and Polarized Light",
North-Holland (1977).

H. Fujiwara, "Spectroscopic Ellipsometry: Principles and Applications",
Wiley (2007).

H. G. Tompkins and E. A. Irene (eds.), "Handbook of Ellipsometry",
William Andrew (2005).

R. H. Muller, "Definitions and conventions in ellipsometry", Surf. Sci.
16, 14-33 (1969).

G. E. Jellison and F. A. Modine, "Parameterization of the optical
functions of amorphous materials in the interband region", Appl. Phys.
Lett. 69, 371-373 (1996); erratum 69, 2137 (1996).

G. E. Jellison, "Data analysis for spectroscopic ellipsometry", Thin Solid
Films 234, 416-422 (1993).

D. A. G. Bruggeman, "Berechnung verschiedener physikalischer Konstanten
von heterogenen Substanzen", Ann. Phys. 416, 636-664 (1935).

I. H. Malitson, "Interspecimen comparison of the refractive index of fused
silica", J. Opt. Soc. Am. 55, 1205-1209 (1965).

M. A. Green, "Self-consistent optical parameters of intrinsic silicon at
300 K including temperature coefficients", Sol. Energy Mater. Sol. Cells
92, 1305-1310 (2008).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .thin_films import _r_t
from .polarization.mueller import M_linear_polarizer, M_retarder


HC_EV_UM = 1.2398419843320026  # photon energy (eV) times wavelength (um)


# --------------------------------------------------------------------------
# conventions and conversions
# --------------------------------------------------------------------------

def photon_energy(wavelength):
    """Photon energy in eV of a vacuum wavelength in um.

    Parameters
    ----------
    wavelength : number or array
        Vacuum wavelength (um).

    Returns
    -------
    number or array
        E = 1.23984 eV um/wavelength.
    """

    return HC_EV_UM/np.asarray(wavelength, dtype=float)


def rho_from_coefficients(r_p, r_s):
    """Ellipsometric ratio rho in the Nebraska convention from the
    reflection coefficients of `thin_films` (r_p as a ratio of tangential
    fields, time dependence exp(-i omega t)).

    Parameters
    ----------
    r_p, r_s : complex numbers or arrays

    Returns
    -------
    rho : complex
        conj(-r_p/r_s) = tan(Psi) exp(i Delta).
    """

    return np.conj(-np.asarray(r_p)/np.asarray(r_s))


def psi_delta_from_rho(rho):
    """Psi and Delta (degrees) of rho = tan(Psi) exp(i Delta).

    Parameters
    ----------
    rho : complex number or array

    Returns
    -------
    psi, delta : arrays
        Psi in [0, 90], Delta in [0, 360) degrees.
    """

    rho = np.asarray(rho)
    return np.degrees(np.arctan(np.abs(rho))), np.mod(np.degrees(np.angle(rho)), 360.0)


def rho_from_psi_delta(psi, delta):
    """rho = tan(Psi) exp(i Delta) from Psi and Delta in degrees.

    Parameters
    ----------
    psi, delta : numbers or arrays (degrees)

    Returns
    -------
    rho : complex array
    """

    return np.tan(np.radians(psi))*np.exp(1j*np.radians(delta))


def pseudo_dielectric(rho, angle, n_ambient=1.0):
    """Pseudo-dielectric function of a measurement: the dielectric function
    of the bulk substrate that would give rho without any layers (exact
    inversion of the two-phase model) ::

        <eps> = eps_a sin(phi)**2 (1 + tan(phi)**2 ((1 - rho)/(1 + rho))**2)

    Parameters
    ----------
    rho : complex number or array
        Ellipsometric ratio (Nebraska convention, as from `rho()`).
    angle : number or array
        Angle of incidence (degrees).
    n_ambient : number, optional
        Index of the ambient medium.

    Returns
    -------
    eps : complex array
        <eps_1> + i <eps_2> (pyoptics convention, <eps_2> > 0 for
        absorption); equals the substrate's eps for a bare substrate.
    """

    phi = np.radians(angle)
    q = (1 - np.conj(rho))/(1 + np.conj(rho))
    return n_ambient**2*np.sin(phi)**2*(1 + np.tan(phi)**2*q**2)


def brewster_angle(n, n_ambient=1.0):
    """Brewster angle (degrees) of a transparent substrate, where Psi = 0
    and Delta jumps from 180 to 0 degrees.

    Parameters
    ----------
    n : number
        Index of the substrate.
    n_ambient : number, optional

    Returns
    -------
    number
    """

    return np.degrees(np.arctan(np.real(n)/np.real(n_ambient)))


# --------------------------------------------------------------------------
# optical constants and dispersion models
# --------------------------------------------------------------------------

def index_from_eps(eps):
    """Complex refractive index n + i k from the dielectric function, with
    k >= 0.

    Parameters
    ----------
    eps : complex number or array

    Returns
    -------
    complex array
    """

    N = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(N.imag < 0, -N, N)


def cauchy(wavelength, A, B=0.0, C=0.0, k=0.0):
    """Cauchy dispersion n = A + B/wavelength**2 + C/wavelength**4 of a
    transparent film (wavelength in um, B in um**2, C in um**4), with an
    optional constant extinction k.

    Parameters
    ----------
    wavelength : number or array (um)
    A, B, C : numbers
    k : number, optional

    Returns
    -------
    complex array
    """

    w2 = np.asarray(wavelength, dtype=float)**2
    return A + B/w2 + C/w2**2 + 1j*k


def sellmeier(wavelength, B, C):
    """Sellmeier dispersion n**2 = 1 + sum B_j wavelength**2/(wavelength**2
    - C_j) (wavelength in um, C_j in um**2).

    Parameters
    ----------
    wavelength : number or array (um)
    B, C : sequences of numbers

    Returns
    -------
    array
    """

    w2 = np.asarray(wavelength, dtype=float)**2
    n2 = 1.0 + sum(b*w2/(w2 - c) for b, c in zip(B, C))
    return np.sqrt(n2) + 0j


def sio2_index(wavelength):
    """Index of fused silica (Malitson 1965), also used for thermal SiO2.

    Parameters
    ----------
    wavelength : number or array (um, 0.21 to 3.7)

    Returns
    -------
    complex array
    """

    return sellmeier(wavelength, (0.6961663, 0.4079426, 0.8974794),
                     (0.0684043**2, 0.1162414**2, 9.896161**2))


# approximate optical constants of crystalline silicon at 300 K, rounded
# from Green (2008); interpolated in photon energy
_SI_WL = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00])
_SI_N = np.array([5.57, 4.67, 4.29, 4.08, 3.94, 3.85, 3.78, 3.68, 3.62, 3.57])
_SI_K = np.array([0.387, 0.135, 0.0705, 0.041, 0.025, 0.017, 0.0105, 0.0055, 0.0023, 0.0004])
_SI_SPLINE_N = CubicSpline(HC_EV_UM/_SI_WL[::-1], _SI_N[::-1])
_SI_SPLINE_LOGK = CubicSpline(HC_EV_UM/_SI_WL[::-1], np.log(_SI_K[::-1]))


def silicon_index(wavelength):
    """Complex index of crystalline silicon at 300 K between 0.40 and 1.00
    um, interpolated in photon energy between rounded values of Green
    (2008). Good to about 1 % in n; for teaching, not as reference data.

    Parameters
    ----------
    wavelength : number or array (um)

    Returns
    -------
    complex array
    """

    w = np.asarray(wavelength, dtype=float)
    if np.any(w < 0.3999) or np.any(w > 1.0001):
        raise ValueError("silicon_index() covers 0.40 to 1.00 um")
    E = HC_EV_UM/w
    return _SI_SPLINE_N(E) + 1j*np.exp(_SI_SPLINE_LOGK(E))


def tauc_lorentz(E, A, E0, C, Eg, eps_inf=1.0):
    """Tauc-Lorentz dielectric function of Jellison and Modine (1996) for
    amorphous semiconductors and dielectrics: a Lorentz oscillator times
    the Tauc joint density of states, zero below the gap, ::

        eps_2 = A E0 C (E - Eg)**2/((E**2 - E0**2)**2 + C**2 E**2)/E  (E > Eg),

    and eps_1 from the Kramers-Kronig integral in closed form.

    Parameters
    ----------
    E : number or array
        Photon energy (eV), see `photon_energy()`.
    A : number
        Amplitude (eV).
    E0 : number
        Peak transition energy (eV).
    C : number
        Broadening (eV), C < 2 E0.
    Eg : number
        Band gap (eV).
    eps_inf : number, optional
        Constant contribution to eps_1.

    Returns
    -------
    complex array
        eps_1 + i eps_2.
    """

    E = np.asarray(E, dtype=float)
    a_ln = (Eg**2 - E0**2)*E**2 + Eg**2*C**2 - E0**2*(E0**2 + 3*Eg**2)
    a_atan = (E**2 - E0**2)*(E0**2 + Eg**2) + Eg**2*C**2
    alpha = np.sqrt(4*E0**2 - C**2)
    gamma = np.sqrt(E0**2 - C**2/2)
    zeta4 = (E**2 - gamma**2)**2 + alpha**2*C**2/4
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gap = np.log(np.abs(E - Eg)/(E + Eg))
        log_gap2 = np.log(np.abs(E - Eg)*(E + Eg)/np.sqrt((E0**2 - Eg**2)**2 + Eg**2*C**2))
        eps1 = (eps_inf
                + A*C*a_ln/(2*np.pi*zeta4*alpha*E0)
                * np.log((E0**2 + Eg**2 + alpha*Eg)/(E0**2 + Eg**2 - alpha*Eg))
                - A*a_atan/(np.pi*zeta4*E0)
                * (np.pi - np.arctan((2*Eg + alpha)/C) + np.arctan((alpha - 2*Eg)/C))
                + 2*A*E0/(np.pi*zeta4*alpha)*Eg*(E**2 - gamma**2)
                * (np.pi + 2*np.arctan(2*(gamma**2 - Eg**2)/(alpha*C)))
                - A*E0*C*(E**2 + Eg**2)/(np.pi*zeta4*E)*log_gap
                + 2*A*E0*C*Eg/(np.pi*zeta4)*log_gap2)
        eps2 = np.where(E > Eg, A*E0*C*(E - Eg)**2/((E**2 - E0**2)**2 + C**2*E**2)/E, 0.0)
    return eps1 + 1j*eps2


def lorentz(E, A, E0, C, eps_inf=1.0):
    """Lorentz oscillator eps = eps_inf + A E0 C/(E0**2 - E**2 - i C E)
    (pyoptics sign of the damping, eps_2 > 0); A is the peak of eps_2.

    Parameters
    ----------
    E : number or array
        Photon energy (eV).
    A, E0, C : numbers
        Amplitude, centre energy (eV) and broadening (eV).
    eps_inf : number, optional

    Returns
    -------
    complex array
    """

    E = np.asarray(E, dtype=float)
    return eps_inf + A*E0*C/(E0**2 - E**2 - 1j*C*E)


def drude(E, E_p, gamma, eps_inf=1.0):
    """Drude free carrier dielectric function eps = eps_inf -
    E_p**2/(E**2 + i gamma E) for metals and transparent conductors.

    Parameters
    ----------
    E : number or array
        Photon energy (eV).
    E_p : number
        Plasma energy (eV).
    gamma : number
        Damping (eV).
    eps_inf : number, optional

    Returns
    -------
    complex array
    """

    E = np.asarray(E, dtype=float)
    return eps_inf - E_p**2/(E**2 + 1j*gamma*E)


def bruggeman(eps_a, eps_b, fraction_b):
    """Bruggeman effective medium of two components: the root of
    f_a (eps_a - eps)/(eps_a + 2 eps) + f_b (eps_b - eps)/(eps_b + 2 eps) = 0
    with Im(eps) >= 0. Models surface roughness (the layer mixed with 50 %
    void) and porous or mixed films.

    Parameters
    ----------
    eps_a, eps_b : complex numbers or arrays
        Dielectric functions of the components.
    fraction_b : number or array
        Volume fraction of component b (0 to 1).

    Returns
    -------
    complex array
    """

    ea, eb = np.asarray(eps_a, dtype=complex), np.asarray(eps_b, dtype=complex)
    fb = np.asarray(fraction_b, dtype=float)
    fa = 1 - fb
    # 2 eps**2 - b eps - ea eb = 0 with b = (2 fa - fb) ea + (2 fb - fa) eb
    b = (2*fa - fb)*ea + (2*fb - fa)*eb
    root = np.sqrt(b**2 + 8*ea*eb)
    e1, e2 = (b + root)/4, (b - root)/4
    # the physical root lies between the components; for lossless media the
    # positive one, otherwise the one with Im(eps) >= 0
    pick = np.where(np.abs(e1.imag) < 1e-14*np.abs(e1), e1.real > 0, e1.imag >= e2.imag)
    return np.where(pick, e1, e2)


def _index(n, wavelength):
    return np.asarray(n(wavelength) if callable(n) else n, dtype=complex)


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------

def reflection_coefficients(wavelength, angle, layers, substrate, ambient=1.0):
    """Reflection coefficients r_p (tangential fields, as `thin_films`) and
    r_s of a layered sample.

    Parameters
    ----------
    wavelength : number or array
        Vacuum wavelength(s).
    angle : number or array
        Angle(s) of incidence in the ambient (degrees), broadcast against
        wavelength.
    layers : sequence of (index, thickness)
        From the ambient side down; index a number or a callable of the
        wavelength. May be empty.
    substrate, ambient : number or callable

    Returns
    -------
    r_p, r_s : complex arrays of the broadcast shape
    """

    wl, ang = np.broadcast_arrays(np.asarray(wavelength, dtype=float),
                                  np.asarray(angle, dtype=float))
    r_p = np.zeros(wl.shape, dtype=complex)
    r_s = np.zeros(wl.shape, dtype=complex)
    t = [float(d) for _, d in layers]
    for w in np.unique(wl):
        sel = wl == w
        n_all = np.array([_index(ambient, w)] + [_index(n, w) for n, _ in layers]
                         + [_index(substrate, w)], dtype=complex).ravel()
        sin0 = np.sin(np.radians(ang[sel]))
        r_p[sel] = _r_t(sin0, n_all, t, "p", w)[0]
        r_s[sel] = _r_t(sin0, n_all, t, "s", w)[0]
    return r_p, r_s


def rho(wavelength, angle, layers, substrate, ambient=1.0):
    """Ellipsometric ratio rho = tan(Psi) exp(i Delta) of a layered sample
    (Nebraska convention).

    Parameters
    ----------
    wavelength, angle, layers, substrate, ambient
        As in `reflection_coefficients()` (angle in degrees).

    Returns
    -------
    complex array
    """

    return rho_from_coefficients(*reflection_coefficients(wavelength, angle, layers, substrate,
                                                          ambient))


def psi_delta(wavelength, angle, layers, substrate, ambient=1.0):
    """Psi and Delta (degrees) of a layered sample.

    Parameters
    ----------
    wavelength, angle, layers, substrate, ambient
        As in `reflection_coefficients()` (angle in degrees).

    Returns
    -------
    psi, delta : arrays (degrees, Delta in [0, 360))
    """

    return psi_delta_from_rho(rho(wavelength, angle, layers, substrate, ambient))


def film_period(wavelength, n_film, angle, n_ambient=1.0):
    """Thickness period of a transparent film: Psi and Delta repeat when the
    thickness grows by wavelength/(2 sqrt(n_film**2 - n_ambient**2
    sin(angle)**2)).

    Parameters
    ----------
    wavelength : number or array
    n_film : number
    angle : number (degrees)
    n_ambient : number, optional

    Returns
    -------
    number or array
    """

    s = n_ambient*np.sin(np.radians(angle))
    return np.asarray(wavelength)/(2*np.sqrt(np.real(n_film)**2 - s**2))


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------

def ncs_from_psi_delta(psi, delta):
    """N, C, S of an isotropic, non-depolarizing sample.

    Parameters
    ----------
    psi, delta : numbers or arrays (degrees)

    Returns
    -------
    N, C, S : arrays
        cos(2 Psi), sin(2 Psi) cos(Delta), sin(2 Psi) sin(Delta).
    """

    p, d = np.radians(psi), np.radians(delta)
    return np.cos(2*p), np.sin(2*p)*np.cos(d), np.sin(2*p)*np.sin(d)


def psi_delta_from_ncs(N, C, S=None):
    """Psi, Delta and the degree of polarization from N, C, S.

    Parameters
    ----------
    N, C : numbers or arrays
    S : number or array, optional
        If None (rotating analyzer: S not measured), Delta is taken from
        cos(Delta) = C/sin(2 Psi) in [0, 180] degrees, with sin(2 Psi) =
        sqrt(1 - N**2) for a non-depolarizing sample.

    Returns
    -------
    psi, delta, P : arrays
        Psi (degrees), Delta in [0, 360) (degrees) and the degree of
        polarization sqrt(N**2 + C**2 + S**2) (NaN without S).
    """

    N, C = np.asarray(N, dtype=float), np.asarray(C, dtype=float)
    if S is None:
        s2p = np.sqrt(np.clip(1 - N**2, 0, None))
        psi = np.degrees(0.5*np.arctan2(s2p, N))
        delta = np.degrees(np.arccos(np.clip(C/np.where(s2p > 0, s2p, np.nan), -1, 1)))
        return psi, delta, np.full(N.shape, np.nan)
    S = np.asarray(S, dtype=float)
    psi = np.degrees(0.5*np.arctan2(np.hypot(C, S), N))
    delta = np.mod(np.degrees(np.arctan2(S, C)), 360.0)
    return psi, delta, np.sqrt(N**2 + C**2 + S**2)


def sample_mueller(N, C, S):
    """Normalized Mueller matrix of an isotropic sample in the p-s basis
    (x along p, y along s).

    Parameters
    ----------
    N, C, S : numbers

    Returns
    -------
    (4, 4) array
    """

    return np.array([[1.0, -N, 0, 0], [-N, 1.0, 0, 0], [0, 0, C, S], [0, 0, -S, C]])


_BASIS = (np.diag([1.0, 1.0, 0, 0]), sample_mueller(1.0, 0, 0) - np.diag([1.0, 1.0, 0, 0]),
          np.diag([0, 0, 1.0, 1.0]), sample_mueller(0, 0, 1.0) - np.diag([1.0, 1.0, 0, 0]))


def _chain(polarizer, analyzer, compensator, retardance, side):
    """Row vector a and column vector s with I = a M_sample s for one set of
    element angles (radians)."""

    s = M_linear_polarizer(polarizer) @ np.array([1.0, 0, 0, 0])
    a = np.array([1.0, 0, 0, 0]) @ M_linear_polarizer(analyzer)
    if compensator is not None:
        M_c = M_retarder(compensator, retardance)
        if side == "input":
            s = M_c @ s
        else:
            a = a @ M_c
    return a, s


def _design(polarizer, analyzer, compensator, retardance, side):
    C0 = 0.0 if compensator is None else compensator
    P, A, Cc = np.broadcast_arrays(np.radians(polarizer), np.radians(analyzer), np.radians(C0))
    Cc = None if compensator is None else Cc
    X = np.zeros(P.shape + (4,))
    for idx in np.ndindex(P.shape):
        a, s = _chain(P[idx], A[idx], None if Cc is None else Cc[idx], np.radians(retardance), side)
        X[idx] = [a @ B @ s for B in _BASIS]
    return X


def ellipsometer_signal(N, C, S, polarizer, analyzer, compensator=None, retardance=90.0,
                        side="input", intensity=1.0):
    """Detector signal of a P(C)S(C)A ellipsometer.

    Light passes a linear polarizer, an optional compensator (a linear
    retarder) on the input or output side, the sample and a linear
    analyzer. Element angles are measured from the plane of incidence (p).
    Any of the angles may be an array (e.g. a rotating analyzer or
    compensator); they are broadcast together.

    Parameters
    ----------
    N, C, S : numbers
        Of the sample (`ncs_from_psi_delta()`); for a depolarizing sample
        N**2 + C**2 + S**2 < 1.
    polarizer, analyzer : numbers or arrays
        Transmission axes (degrees).
    compensator : number or array, optional
        Fast axis of the compensator (degrees); None for none.
    retardance : number, optional
        Of the compensator (degrees).
    side : {"input", "output"}, optional
        Compensator between polarizer and sample, or sample and analyzer.
    intensity : number, optional
        Source intensity.

    Returns
    -------
    array
        Intensity at the detector.
    """

    X = _design(polarizer, analyzer, compensator, retardance, side)
    return intensity*(X @ np.array([1.0, N, C, S]))


def fit_ncs(signal, polarizer, analyzer, compensator=None, retardance=90.0, side="input"):
    """N, C, S from measured ellipsometer signals by linear least squares.

    Parameters
    ----------
    signal : array
        Measured intensities.
    polarizer, analyzer, compensator, retardance, side
        The element angles of each measurement, as in
        `ellipsometer_signal()`.

    Returns
    -------
    N, C, S : numbers
        S is None if the configuration is blind to it (e.g. a rotating
        analyzer without compensator).
    """

    X = _design(polarizer, analyzer, compensator, retardance, side).reshape(-1, 4)
    y = np.asarray(signal, dtype=float).ravel()
    keep = np.linalg.norm(X, axis=0) > 1e-12*np.linalg.norm(X)
    coef = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    full = np.full(4, np.nan)
    full[keep] = coef
    N, C, S = full[1:]/full[0]
    return N, C, (None if np.isnan(S) else S)


def rotating_analyzer_coefficients(psi, delta, polarizer=45.0):
    """Normalized Fourier coefficients of a rotating analyzer ellipsometer,
    I(A) proportional to 1 + alpha cos(2A) + beta sin(2A) ::

        alpha = (tan(Psi)**2 - tan(P)**2)/(tan(Psi)**2 + tan(P)**2),
        beta = 2 tan(Psi) cos(Delta) tan(P)/(tan(Psi)**2 + tan(P)**2).

    Parameters
    ----------
    psi, delta : numbers or arrays (degrees)
    polarizer : number, optional
        Polarizer angle P (degrees).

    Returns
    -------
    alpha, beta : arrays
    """

    tp, tP = np.tan(np.radians(psi)), np.tan(np.radians(polarizer))
    den = tp**2 + tP**2
    return (tp**2 - tP**2)/den, 2*tp*np.cos(np.radians(delta))*tP/den


# --------------------------------------------------------------------------
# model fits
# --------------------------------------------------------------------------

class EllipsometryFit(object):
    """Result of `fit_psi_delta()`.

    Attributes
    ----------
    params : array
        Best fit parameters.
    std : array
        Standard errors, sqrt(diag((J^T J)^-1) MSE).
    correlation : (m, m) array
        Correlation matrix of the parameters.
    mse : number
        Jellison's mean squared error (about 1 for a fit within the noise).
    psi, delta : arrays
        Model Psi and Delta at the best fit.
    nfev : int
        Number of model evaluations.
    """

    def __init__(self, params, std, correlation, mse, psi, delta, nfev):
        self.params, self.std, self.correlation = params, std, correlation
        self.mse, self.psi, self.delta, self.nfev = mse, psi, delta, nfev

    def __repr__(self):
        p = ", ".join("{:.5g} +- {:.2g}".format(a, b) for a, b in zip(self.params, self.std))
        return "EllipsometryFit([{}], MSE = {:.3g})".format(p, self.mse)


def _wrap180(d):
    return np.mod(d + 180.0, 360.0) - 180.0


def fit_psi_delta(model, p0, psi, delta, sigma_psi=0.05, sigma_delta=0.1, bounds=None,
                  x_scale="jac", **kwargs):
    """Fit model parameters to measured Psi and Delta.

    Parameters
    ----------
    model : callable
        model(params) -> (psi, delta) in degrees, arrays of the shape of
        the data.
    p0 : sequence of numbers
        Start values.
    psi, delta : arrays
        Measured Psi and Delta (degrees). Delta residuals are wrapped to
        [-180, 180).
    sigma_psi, sigma_delta : numbers or arrays, optional
        Standard deviations of the measurements (degrees).
    bounds : (lower, upper), optional
        As for scipy.optimize.least_squares.
    x_scale : optional
        As for scipy.optimize.least_squares.
    **kwargs
        Passed to scipy.optimize.least_squares.

    Returns
    -------
    EllipsometryFit
    """

    psi, delta = np.asarray(psi, dtype=float), np.asarray(delta, dtype=float)

    def residuals(p):
        pm, dm = model(p)
        return np.concatenate([np.ravel((pm - psi)/sigma_psi),
                               np.ravel(_wrap180(dm - delta)/sigma_delta)])

    kw = dict(x_scale=x_scale, method="trf" if bounds is not None else "lm")
    if bounds is not None:
        kw["bounds"] = bounds
    kw.update(kwargs)
    res = least_squares(residuals, np.asarray(p0, dtype=float), **kw)
    m = len(res.x)
    n_res = res.fun.size
    mse = np.sum(res.fun**2)/max(n_res - m, 1)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    std = np.sqrt(np.clip(np.diag(cov), 0, None)*mse)
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov/np.outer(d, d)
    pm, dm = model(res.x)
    return EllipsometryFit(res.x, std, corr, mse, pm, dm, res.nfev)
