#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Nonlinear optics: second-order (chi(2)) frequency conversion with
birefringent and quasi-phase matching, parametric amplification and
oscillators, and third-order (chi(3)) effects: the optical Kerr effect,
self-focusing and the Kerr lens. See
notebooks 145 (second harmonic generation and phase matching),
146 (parametric amplification and OPOs) and
147 (Kerr effect and self-focusing); pulses in fibres are in
`pyoptics.splitstep` and notebook 121 (SPM and solitons in fibres).

SI units (m, s, W, V/m, pm/V only in the `CRYSTALS` table) unless a
function says otherwise; wavelengths are vacuum wavelengths in nm where
the name says so (`wavelength_nm`), as in `pyoptics.crystals`. Fields
follow the pyoptics convention exp(i(kz - omega t)).

Coupled amplitude equations
---------------------------
Three waves omega_1 + omega_2 = omega_3 in a medium with the effective
nonlinear coefficient d_eff (Boyd's convention E = A exp(i(kz - omega t))
+ c.c., which is the pyoptics sign) are written here with amplitudes a_j
normalized to the intensity, abs(a_j)**2 = I_j = 2 n_j eps0 c abs(A_j)**2::

    da_1/dz = i omega_1 K a_3 conj(a_2) exp(-i dk z)
    da_2/dz = i omega_2 K a_3 conj(a_1) exp(-i dk z)
    da_3/dz = i omega_3 K a_1 a_2 exp(+i dk z),

K = d_eff sqrt(2/(n_1 n_2 n_3 eps0 c**3)) and dk = k_1 + k_2 - k_3
(`three_wave()`). The photon fluxes I_j/omega_j change in step (the
Manley–Rowe relations), the total intensity is conserved. Second harmonic
generation is the degenerate case (`shg()`)::

    da_1/dz = i kappa a_2 conj(a_1) exp(-i dk z),  da_2/dz = i kappa a_1**2 exp(+i dk z),

with kappa = omega_1 d_eff sqrt(2/(n_1**2 n_2 eps0 c**3)) and
dk = 2 k_1 - k_2. Phase matched, the second harmonic grows as
I_2 = I tanh**2(kappa sqrt(I) z) (Armstrong et al.); without depletion the
efficiency is ::

    eta = 2 omega**2 d_eff**2 L**2 I/(n_1**2 n_2 eps0 c**3) sinc**2(dk L/2)

(`shg_efficiency()`), with sinc(x) = sin(x)/x. The second harmonic grows
only over the coherence length pi/abs(dk) (`coherence_length()`) before
it flows back.

Phase matching
--------------
Birefringent phase matching uses the different indices of the ordinary
and extraordinary waves (`indices()`, `extraordinary_index()`, or the
plane waves of `pyoptics.crystals` in `uniaxial_waves()`, which also
give the walk-off). The polarizations are written as three letters for
the waves 1, 2 and 3 (lowest frequency first): in a negative uniaxial
crystal type I is "ooe" (both input waves ordinary), type II "oee" or
"eoe". `phase_matching_angle()` finds the angle theta to the optic axis
with dk = 0, `d_eff()` the effective coefficient of BBO and KDP at theta
and the azimuth phi. Quasi-phase matching flips the sign of d_33 every
half period Lambda = 2 pi/abs(dk) (`qpm_period()`, with the thermal
expansion of LiNbO3), with d_eff = 2/(m pi) sin(m pi D) d_33 for the
order m and the duty cycle D. `three_wave()` and `shg()` accept d_eff as a
function of z for real domain patterns.

The tolerances follow from dk against a parameter x (angle, wavelength,
temperature): the efficiency drops to one half where abs(dk) L/2 = 1.392
(`acceptance_bandwidth()`), which for dk linear in x is
FWHM = 5.566/(L abs(d dk/dx)).

Focused beams: `boyd_kleinman_h()` is the focusing function h(sigma, B,
xi) of Boyd and Kleinman (1968), with xi = L/b (b = 2 z_R the confocal
parameter), the normalized phase mismatch sigma = b dk/2 and the walk-off
parameter B = rho sqrt(L k_1)/2; the second harmonic power of a focused
Gaussian beam is ::

    P_2 = 16 pi**2 d_eff**2 L P_1**2 h/(eps0 c lambda_1**3 n_1 n_2)

(`shg_power_focused()`), with the optimum h = 1.068 at xi = 2.84 without
walk-off.

Parametric amplification
------------------------
A strong pump I_p amplifies a weak signal with the gain coefficient
Gamma = sqrt(omega_s omega_i K**2 I_p) (`parametric_gain_coefficient()`)
and g = sqrt(Gamma**2 - (dk/2)**2): the signal power gain is
1 + (Gamma/g)**2 sinh**2(g L) (`opa_gain()`). In an optical parametric
oscillator the signal (singly resonant, SRO) or signal and idler (doubly
resonant, DRO) are fed back; the plane wave threshold is
cosh(Gamma L) = (1 + r_s r_i)/(r_s + r_i) for the amplitude feedback r_s,
r_i per round trip (`opo_threshold()`; SRO: r_i = 0). The pump depletion
of an SRO with a lossless resonated signal is sin**2(x), where x solves
x**2/sin**2(x) = N = I_p/I_th (`sro_pump_depletion()`): 100 % at
N = pi**2/4. `qpm_signal()` gives the tuning curve of a quasi-phase
matched OPO.

Pulses: `shg_pulse()` solves the coupled equations of second harmonic
generation for pulses with group velocity mismatch and dispersion, with
`pyoptics.splitstep.rk4ip()`.

Third order
-----------
The optical Kerr effect n = n_0 + n_2 I (`N2` for common materials) gives
a beam the nonlinear phase k_0 n_2 I L (`kerr_phase()`) and a Gaussian
beam a lens of focal length pi w**4/(8 n_2 L P) (`kerr_lens_focal()`). Above
the critical power P_cr = 3.77 lambda**2/(8 pi n_0 n_2) (`critical_power()`,
for a Gaussian beam) self-focusing beats diffraction and a collimated beam
collapses after Marburger's distance (`self_focusing_distance()`);
`pyoptics.splitstep.beam_propagate()` simulates it. In a laser, the Kerr
lens in the gain crystal and an aperture form a fast saturable loss:
`kerr_lens_mode()` finds the self-consistent mode of a resonator with a
Kerr lens, and `kerr_lens_loss()` turns the loss at the aperture into a
function q(P) for `pyoptics.lasers.master_equation()` (Kerr-lens
mode-locking).

References
----------
R. W. Boyd, "Nonlinear Optics", 4th ed., Academic Press (2020),
chapters 2, 4 and 7.

B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019), chapter 22.

J. A. Armstrong, N. Bloembergen, J. Ducuing and P. S. Pershan,
"Interactions between light waves in a nonlinear dielectric", Phys. Rev.
127, 1918-1939 (1962).

G. D. Boyd and D. A. Kleinman, "Parametric interaction of focused Gaussian
light beams", J. Appl. Phys. 39, 3597-3639 (1968).

M. M. Fejer, G. A. Magel, D. H. Jundt and R. L. Byer, "Quasi-phase-matched
second harmonic generation: tuning and tolerances", IEEE J. Quantum
Electron. 28, 2631-2654 (1992).

O. Gayer, Z. Sacks, E. Galun and A. Arie, "Temperature and wavelength
dependent refractive index equations for MgO-doped congruent and
stoichiometric LiNbO3", Appl. Phys. B 91, 343-348 (2008).

D. Eimerl, L. Davis, S. Velsko, E. K. Graham and A. Zalkin, "Optical,
mechanical, and thermal properties of barium borate", J. Appl. Phys. 62,
1968-1983 (1987).

D. N. Nikogosyan, "Nonlinear Optical Crystals: A Complete Survey",
Springer (2005).

J. H. Marburger, "Self-focusing: theory", Prog. Quantum Electron. 4,
35-110 (1975).

A. Couairon and A. Mysyrowicz, "Femtosecond filamentation in transparent
media", Phys. Rep. 441, 47-189 (2007).

R. L. Byer, "Optical parametric oscillators", in "Quantum Electronics: A
Treatise", Academic Press (1975).

V. Magni, G. Cerullo and S. De Silvestri, "ABCD matrix analysis of
propagation of Gaussian beams through Kerr media", Opt. Commun. 96,
348-355 (1993).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from . import abcd
from . import crystals
from . import splitstep
from .electrooptics import linbo3_indices


__all__ = ["C0", "EPS0", "CRYSTALS", "N2", "bbo_indices", "kdp_indices", "mgo_ln_ne", "indices",
           "extraordinary_index", "uniaxial_waves", "wave_number", "delta_k",
           "phase_matching_angle", "d_eff", "qpm_d_eff", "qpm_period", "coherence_length",
           "acceptance_bandwidth", "coupling_K", "shg_kappa", "shg_efficiency", "shg_depleted",
           "shg", "three_wave", "boyd_kleinman_h", "boyd_kleinman_optimum", "shg_power_focused",
           "parametric_gain_coefficient", "opa_gain", "opo_threshold", "sro_pump_depletion",
           "qpm_signal", "shg_pulse", "kerr_phase", "kerr_lens_focal", "critical_power",
           "self_focusing_distance", "kerr_lens_mode", "kerr_lens_loss"]


C0 = 2.99792458E8          # m/s
EPS0 = 8.8541878128E-12    # F/m


# --- crystals -------------------------------------------------------------------

def bbo_indices(wavelength_nm, T=None):
    """Ordinary and extraordinary indices of beta-barium borate (BBO, negative
    uniaxial), Sellmeier fit of Eimerl et al. (1987), 0.2 to 2.6 µm (T is
    ignored).

    Parameters
    ----------
    wavelength_nm : number or array
    T : number, optional
        Not used.

    Returns
    -------
    n_o, n_e : arrays
        1.6545 and 1.5392 at 1064 nm.
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1E-3)**2
    n_o = np.sqrt(2.7359 + 0.01878/(l2 - 0.01822) - 0.01354*l2)
    n_e = np.sqrt(2.3753 + 0.01224/(l2 - 0.01667) - 0.01516*l2)
    return n_o, n_e


def kdp_indices(wavelength_nm, T=None):
    """Ordinary and extraordinary indices of potassium dihydrogen phosphate
    (KDP, negative uniaxial), Sellmeier fit of Zernike (1964), 0.2 to 1.5 µm
    (T is ignored).

    Parameters
    ----------
    wavelength_nm : number or array
    T : number, optional
        Not used.

    Returns
    -------
    n_o, n_e : arrays
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1E-3)**2
    n_o = np.sqrt(2.259276 + 0.01008956/(l2 - 0.012942625) + 13.00522*l2/(l2 - 400))
    n_e = np.sqrt(2.132668 + 0.008637494/(l2 - 0.012281043) + 3.2279924*l2/(l2 - 400))
    return n_o, n_e


def mgo_ln_ne(wavelength_nm, T=24.5):
    """Extraordinary index of 5 mol% MgO-doped congruent LiNbO3 against
    wavelength and temperature (Gayer et al. 2008, 0.5 to 4 µm, 20 to
    200 °C), the crystal of most PPLN devices.

    Parameters
    ----------
    wavelength_nm : number or array
    T : number or array, optional
        Temperature in °C.

    Returns
    -------
    n_e : array
        2.1482 at 1064 nm and 24.5 °C.
    """

    lam = np.asarray(wavelength_nm, dtype=float)*1E-3
    T = 24.5 if T is None else T
    f = (np.asarray(T, dtype=float) - 24.5)*(np.asarray(T, dtype=float) + 570.82)
    a1, a2, a3, a4, a5, a6 = 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32E-2
    b1, b2, b3, b4 = 2.860E-6, 4.700E-8, 6.113E-8, 1.516E-4
    n2 = (a1 + b1*f + (a2 + b2*f)/(lam**2 - (a3 + b3*f)**2) + (a4 + b4*f)/(lam**2 - a5**2)
          - a6*lam**2)
    return np.sqrt(n2)


def _mgo_ln_indices(wavelength_nm, T=None):
    # ordinary index of congruent LiNbO3 (Zelmon) shifted to MgO:LN is not needed for QPM (eee)
    n_o = linbo3_indices(wavelength_nm)[0]
    return n_o, mgo_ln_ne(wavelength_nm, T)


def _ln_indices(wavelength_nm, T=None):
    return linbo3_indices(wavelength_nm)


#: Nonlinear crystals: index functions (wavelength in nm, temperature in °C),
#: d coefficients in pm/V (Nikogosyan; d33 of MgO:LN for 1064 nm SHG) and
#: whether the crystal is used birefringently or quasi-phase matched.
CRYSTALS = {
    "BBO": {"indices": bbo_indices, "d": {"d22": 2.2, "d31": 0.04}, "class": "3m",
            "use": "birefringent"},
    "KDP": {"indices": kdp_indices, "d": {"d36": 0.39}, "class": "-42m", "use": "birefringent"},
    "MgO:LN": {"indices": _mgo_ln_indices, "d": {"d33": 25.0, "d31": 4.4}, "class": "3m",
               "use": "quasi-phase matched", "expansion": (1.54E-5, 5.3E-9)},
    "LN": {"indices": _ln_indices, "d": {"d33": 25.2, "d31": 4.6}, "class": "3m",
           "use": "quasi-phase matched", "expansion": (1.54E-5, 5.3E-9)},
}

#: Kerr coefficients n2 (m²/W) near 800 to 1064 nm (Boyd, table 4.1.2; Couairon and
#: Mysyrowicz; Major et al.); orders of magnitude, they depend on wavelength and pulse length.
N2 = {"air": 3.0E-23, "fused silica": 2.7E-20, "BK7": 3.4E-20, "water": 4.1E-20,
      "sapphire": 3.0E-20, "YAG": 6.2E-20, "CS2": 3.0E-18}


def indices(crystal, wavelength_nm, T=None):
    """Ordinary and extraordinary indices of a crystal in `CRYSTALS`.

    Parameters
    ----------
    crystal : str
    wavelength_nm : number or array
    T : number, optional
        Temperature in °C (where the data have it).

    Returns
    -------
    n_o, n_e : arrays
    """

    return CRYSTALS[crystal]["indices"](wavelength_nm, T)


def extraordinary_index(n_o, n_e, theta):
    """Index of the extraordinary wave whose wave normal makes the angle theta
    with the optic axis, 1/sqrt(cos**2/n_o**2 + sin**2/n_e**2)."""
    return 1/np.sqrt(np.cos(theta)**2/n_o**2 + np.sin(theta)**2/n_e**2)


def uniaxial_waves(n_o, n_e, theta):
    """Ordinary and extraordinary wave for the wave normal at theta to the
    optic axis, from the index ellipsoid of `pyoptics.crystals`.

    Parameters
    ----------
    n_o, n_e : numbers or arrays
    theta : number or array
        Radians.

    Returns
    -------
    n_ord, n_ext : arrays
        Indices of the two waves.
    rho : array
        Walk-off angle of the extraordinary wave (radians).
    """

    n_o, n_e, theta = np.broadcast_arrays(np.asarray(n_o, dtype=float), np.asarray(n_e, dtype=float),
                                          np.asarray(theta, dtype=float))
    bad = ~(np.isfinite(n_o) & np.isfinite(n_e) & np.isfinite(theta))
    n_o, n_e, theta = (np.where(bad, 1.5, x) for x in (n_o, n_e, theta))
    eps = crystals.uniaxial_eps(n_o, n_e, axis=(0.0, 0.0, 1.0))
    s = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)
    n, D, E, S = crystals.wave_indices(eps, s)
    # the ordinary wave has D along y (perpendicular to the plane of axis and wave normal)
    is_o = np.abs(D[..., :, 1]) > np.abs(D[..., :, 0]) + np.abs(D[..., :, 2]) - 1E-12
    first_o = is_o[..., 0] & ~is_o[..., 1]
    n = n.real
    n_ord = np.where(first_o, n[..., 0], n[..., 1])
    n_ext = np.where(first_o, n[..., 1], n[..., 0])
    rho_all = crystals.walkoff(eps, s)
    rho = np.where(first_o, rho_all[..., 1], rho_all[..., 0])
    nan = lambda x: np.where(bad, np.nan, x)
    return nan(n_ord), nan(n_ext), nan(rho)


def wave_number(n, wavelength_nm):
    """k = 2 pi n/lambda in 1/m."""
    return 2*np.pi*n/(np.asarray(wavelength_nm, dtype=float)*1E-9)


def _index(crystal, wl, pol, theta, T):
    n_o, n_e = indices(crystal, wl, T)
    if pol == "o":
        return n_o
    if theta is None:
        return n_e
    return extraordinary_index(n_o, n_e, theta)


def _wavelengths(wl1, wl2):
    wl1 = np.asarray(wl1, dtype=float)
    wl2 = wl1 if wl2 is None else np.asarray(wl2, dtype=float)
    wl3 = 1/(1/wl1 + 1/wl2)
    return wl1, wl2, wl3


def delta_k(crystal, wl1, wl2=None, theta=None, pol=None, T=None, period=None, order=1):
    """Phase mismatch dk = k_1 + k_2 - k_3 (+ 2 pi m/Lambda for quasi-phase
    matching) of the three wave process omega_1 + omega_2 = omega_3.

    Parameters
    ----------
    crystal : str
        Key of `CRYSTALS`.
    wl1, wl2 : numbers or arrays
        Vacuum wavelengths (nm) of the two lower frequency waves; wl2 =
        wl1 (default) for second harmonic generation.
    theta : number or array, optional
        Angle of the wave normal to the optic axis (radians); None means
        perpendicular to the axis with the extraordinary index n_e
        (quasi-phase matching, "eee").
    pol : str, optional
        Polarizations of waves 1, 2 and 3, "o" or "e"; default "ooe" with
        theta, "eee" without.
    T : number, optional
        Temperature in °C.
    period : number, optional
        Poling period Lambda (m), at 25 °C; the thermal expansion of the
        crystal (if listed) is applied at T.
    order : int, optional
        QPM order m.

    Returns
    -------
    dk : array
        1/m.
    """

    wl1, wl2, wl3 = _wavelengths(wl1, wl2)
    if pol is None:
        pol = "eee" if theta is None else "ooe"
    ns = [_index(crystal, wl, p, theta, T) for wl, p in zip((wl1, wl2, wl3), pol)]
    dk = wave_number(ns[0], wl1) + wave_number(ns[1], wl2) - wave_number(ns[2], wl3)
    if period is not None:
        dk = dk + 2*np.pi*order/_expanded(crystal, period, T)
    return dk


def _expanded(crystal, period, T):
    coef = CRYSTALS[crystal].get("expansion")
    if coef is None or T is None:
        return period
    dT = np.asarray(T, dtype=float) - 25.0
    return period*(1 + coef[0]*dT + coef[1]*dT**2)


def phase_matching_angle(crystal, wl1, wl2=None, pol="ooe", T=None):
    """Angle theta to the optic axis at which dk = 0 (birefringent phase
    matching).

    Parameters
    ----------
    crystal : str
    wl1, wl2 : numbers
        Wavelengths (nm) of the two lower frequency waves (wl2 = wl1 for
        second harmonic generation).
    pol : str, optional
        "ooe" (type I in a negative crystal), "oee" or "eoe" (type II).
    T : number, optional

    Returns
    -------
    theta : number
        Radians, nan if no angle phase matches.
    """

    f = lambda th: float(delta_k(crystal, wl1, wl2, th, pol, T))
    grid = np.linspace(0.0, np.pi/2, 721)
    vals = np.array([f(th) for th in grid])
    sign = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign) == 0:
        return np.nan
    i = sign[0]
    return brentq(f, grid[i], grid[i + 1], xtol=1E-13)


def d_eff(crystal, theta, phi=np.pi/2, pol="ooe"):
    """Effective nonlinear coefficient (pm/V) of a birefringently phase
    matched process in BBO (point group 3m) or KDP (-42m), Nikogosyan's
    formulas.

    Parameters
    ----------
    crystal : str
        "BBO" or "KDP".
    theta, phi : numbers
        Polar angle to the optic axis and azimuth (radians); phi = 90° for
        BBO type I and phi = 45° for KDP maximize d_eff.
    pol : str, optional
        "ooe" (type I) or "oee"/"eoe" (type II).

    Returns
    -------
    d : number
        pm/V (absolute value).
    """

    d = CRYSTALS[crystal]["d"]
    type_I = pol.count("o") == 2
    if crystal == "BBO":
        if type_I:
            val = d["d31"]*np.sin(theta) - d["d22"]*np.cos(theta)*np.sin(3*phi)
        else:
            val = d["d22"]*np.cos(theta)**2*np.cos(3*phi)
    elif crystal == "KDP":
        if type_I:
            val = -d["d36"]*np.sin(theta)*np.sin(2*phi)
        else:
            val = d["d36"]*np.sin(2*theta)*np.cos(2*phi)
    else:
        raise ValueError("d_eff() knows BBO and KDP; use qpm_d_eff() for quasi-phase matching")
    return np.abs(val)


def qpm_d_eff(d33, order=1, duty=0.5):
    """Effective coefficient 2/(m pi) sin(m pi D) d33 of quasi-phase matching
    of order m with the duty cycle D (2/pi d33 for first order and D = 1/2)."""
    return 2/(order*np.pi)*np.abs(np.sin(order*np.pi*duty))*d33


def qpm_period(crystal, wl1, wl2=None, T=None, order=1, pol="eee"):
    """Poling period Lambda = 2 pi m/abs(dk) (m) that quasi-phase matches the
    process at the temperature T (°C), referred to 25 °C (the period is
    written at room temperature and expands when heated).

    Parameters
    ----------
    crystal : str
    wl1, wl2 : numbers or arrays
        Wavelengths (nm); wl2 = wl1 for SHG.
    T : number, optional
    order : int, optional
    pol : str, optional

    Returns
    -------
    period : array
        m.
    """

    dk = delta_k(crystal, wl1, wl2, None, pol, T)
    Lam_T = 2*np.pi*order/np.abs(dk)
    coef = CRYSTALS[crystal].get("expansion")
    if coef is None or T is None:
        return Lam_T
    dT = np.asarray(T, dtype=float) - 25.0
    return Lam_T/(1 + coef[0]*dT + coef[1]*dT**2)


def coherence_length(dk):
    """Coherence length pi/abs(dk): the length over which a mismatched
    harmonic grows before it converts back."""
    return np.pi/np.abs(dk)


def acceptance_bandwidth(dk_of_x, x0, length, span=None):
    """Full width at half maximum of sinc**2(dk(x) L/2) around the phase
    matched x0, found numerically (it works for critical and noncritical
    phase matching alike).

    Parameters
    ----------
    dk_of_x : callable
        dk (1/m) against the parameter x (an angle, a wavelength, a
        temperature).
    x0 : number
        Phase matched value.
    length : number
        Crystal length (m).
    span : number, optional
        Scale of the search: the band edges are bracketed in steps of
        span/100 (default: from the slope and curvature of dk at x0).

    Returns
    -------
    fwhm : number
        In the units of x.
    lo, hi : numbers
        Edges of the band.
    """

    f = lambda x: np.sinc(dk_of_x(x)*length/(2*np.pi))**2 - 0.5
    if span is None:
        h = 1E-6*max(abs(x0), 1.0)
        slope = abs(dk_of_x(x0 + h) - dk_of_x(x0 - h))/(2*h)
        curv = abs(dk_of_x(x0 + h) + dk_of_x(x0 - h) - 2*dk_of_x(x0))/h**2
        span = min(5.566/(length*slope) if slope > 0 else np.inf,
                   np.sqrt(2*5.566/(length*curv)) if curv > 0 else np.inf)*3
    edges = []
    step = span/100
    for direction in (-1, 1):
        a, fa = x0, f(x0)
        for _ in range(100000):
            b = a + direction*step
            fb = f(b)
            if fb < 0:
                break
            a, fa = b, fb
        else:
            raise RuntimeError("acceptance_bandwidth: no half maximum found")
        edges.append(brentq(f, min(a, b), max(a, b)))
    lo, hi = min(edges), max(edges)
    return hi - lo, lo, hi


# --- coupled amplitude equations -----------------------------------------------------

def coupling_K(d, n1, n2, n3):
    """Coupling constant K = d sqrt(2/(n1 n2 n3 eps0 c**3)) of the three wave
    equations in intensity units (d in m/V; K omega_j is in 1/(m sqrt(W/m²)))."""
    return d*np.sqrt(2/(n1*n2*n3*EPS0*C0**3))


def shg_kappa(d, wavelength, n1, n2):
    """kappa = omega_1 d sqrt(2/(n1**2 n2 eps0 c**3)) of second harmonic
    generation (d in m/V, wavelength the fundamental in m)."""
    omega = 2*np.pi*C0/wavelength
    return omega*d*np.sqrt(2/(n1**2*n2*EPS0*C0**3))


def shg_efficiency(I, length, d, wavelength, n1, n2, dk=0.0):
    """Undepleted plane wave conversion efficiency
    2 omega**2 d**2 L**2 I/(n1**2 n2 eps0 c**3) sinc**2(dk L/2).

    Parameters
    ----------
    I : number or array
        Fundamental intensity (W/m²).
    length : number
        m.
    d : number
        d_eff (m/V).
    wavelength : number
        Fundamental (m).
    n1, n2 : numbers
        Indices at the fundamental and the harmonic.
    dk : number or array, optional

    Returns
    -------
    eta : array
    """

    k = shg_kappa(d, wavelength, n1, n2)
    return k**2*np.asarray(I)*length**2*np.sinc(np.asarray(dk)*length/(2*np.pi))**2


def shg_depleted(I, length, d, wavelength, n1, n2):
    """Phase matched conversion efficiency with pump depletion,
    tanh**2(kappa sqrt(I) L) (Armstrong et al.).

    Parameters
    ----------
    I : number or array
        Fundamental intensity (W/m²).
    length : number
    d : number
        m/V.
    wavelength : number
        m.
    n1, n2 : numbers

    Returns
    -------
    eta : array
    """

    return np.tanh(shg_kappa(d, wavelength, n1, n2)*np.sqrt(np.asarray(I))*length)**2


def _d_function(d):
    return d if callable(d) else (lambda z: d)


def three_wave(z, a0, wavelengths, n, d, dk=0.0, rtol=1E-9, atol=1E-12, max_step=np.inf):
    """Plane wave coupled amplitude equations of omega_1 + omega_2 = omega_3
    (sum and difference frequency generation, parametric amplification).

    Parameters
    ----------
    z : array
        Positions (m) at which to return the amplitudes, starting at the
        input.
    a0 : sequence of 3 complex
        Input amplitudes (abs(a)**2 = intensity in W/m²).
    wavelengths : sequence of 3 numbers
        Vacuum wavelengths (m) of waves 1, 2, 3 (1/l1 + 1/l2 = 1/l3).
    n : sequence of 3 numbers
        Indices.
    d : number or callable
        d_eff (m/V), or d(z) for a poled crystal.
    dk : number, optional
        k1 + k2 - k3 (1/m).
    rtol, atol, max_step : optional
        Passed to solve_ivp (max_step must resolve a poling period).

    Returns
    -------
    a : complex array (len(z), 3)
    """

    z = np.asarray(z, dtype=float)
    om = 2*np.pi*C0/np.asarray(wavelengths, dtype=float)
    K = coupling_K(1.0, *n)
    d_of = _d_function(d)
    scale = np.sqrt(max(np.sum(np.abs(np.asarray(a0))**2), 1E-300))

    def rhs(zz, y):
        a = (y[:3] + 1j*y[3:])*scale
        c = 1j*K*d_of(zz)
        ph = np.exp(1j*dk*zz)
        da = np.array([c*om[0]*a[2]*np.conj(a[1])*np.conj(ph),
                       c*om[1]*a[2]*np.conj(a[0])*np.conj(ph),
                       c*om[2]*a[0]*a[1]*ph])/scale
        return np.concatenate([da.real, da.imag])

    y0 = np.asarray(a0, dtype=complex)/scale
    sol = solve_ivp(rhs, (z[0], z[-1]), np.concatenate([y0.real, y0.imag]), t_eval=z, rtol=rtol,
                    atol=atol, max_step=max_step, method="DOP853")
    return ((sol.y[:3] + 1j*sol.y[3:])*scale).T


def shg(z, I0, wavelength, n1, n2, d, dk=0.0, a2_0=0.0, rtol=1E-9, atol=1E-12, max_step=np.inf):
    """Plane wave second harmonic generation with depletion.

    Parameters
    ----------
    z : array
        Positions (m), starting at the input.
    I0 : number
        Fundamental intensity (W/m²); its amplitude is taken real.
    wavelength : number
        Fundamental (m).
    n1, n2 : numbers
    d : number or callable
        d_eff (m/V) or d(z).
    dk : number, optional
        2 k1 - k2.
    a2_0 : complex, optional
        Input harmonic amplitude (sqrt(W/m²)).
    rtol, atol, max_step : optional
        Passed to solve_ivp.

    Returns
    -------
    a1, a2 : complex arrays
        Amplitudes along z (abs()**2 = intensities).
    """

    z = np.asarray(z, dtype=float)
    kap = shg_kappa(1.0, wavelength, n1, n2)
    d_of = _d_function(d)
    s = np.sqrt(I0)

    def rhs(zz, y):
        a1, a2 = (y[0] + 1j*y[2])*s, (y[1] + 1j*y[3])*s
        c = 1j*kap*d_of(zz)
        ph = np.exp(1j*dk*zz)
        d1 = c*a2*np.conj(a1)*np.conj(ph)/s
        d2 = c*a1**2*ph/s
        return [d1.real, d2.real, d1.imag, d2.imag]

    y0 = [1.0, np.real(a2_0)/s, 0.0, np.imag(a2_0)/s]
    sol = solve_ivp(rhs, (z[0], z[-1]), y0, t_eval=z, rtol=rtol, atol=atol, max_step=max_step,
                    method="DOP853")
    return (sol.y[0] + 1j*sol.y[2])*s, (sol.y[1] + 1j*sol.y[3])*s


# --- focused beams ----------------------------------------------------------------------

def boyd_kleinman_h(xi, sigma, B=0.0, n_points=600):
    """Boyd–Kleinman focusing function of second harmonic generation with
    the focus in the centre of the crystal and no absorption,

        h = 1/(4 xi) int int exp(i sigma (tau - tau') - B**2 (tau - tau')**2/xi)
                             /((1 + i tau)(1 - i tau')) dtau dtau'

    over -xi..xi.

    Parameters
    ----------
    xi : number
        L/b, b = 2 z_R the confocal parameter.
    sigma : number or array
        Normalized phase mismatch b dk/2.
    B : number, optional
        Walk-off parameter rho sqrt(L k_1)/2.
    n_points : int, optional
        Gauss–Legendre points.

    Returns
    -------
    h : array
        xi for xi << 1 (near field), 1.068 at the optimum xi = 2.84,
        sigma = 0.57 without walk-off.
    """

    x, w = np.polynomial.legendre.leggauss(n_points)
    tau, w = xi*x, xi*w
    g = 1/(1 + 1j*tau)
    sig = np.atleast_1d(np.asarray(sigma, dtype=float))
    out = np.empty(sig.shape)
    if B == 0:
        for i, s in enumerate(sig):
            out[i] = np.abs(np.sum(w*np.exp(1j*s*tau)*g))**2/(4*xi)
    else:
        dt = tau[:, None] - tau[None, :]
        G = (w*g)[:, None]*np.conj(w*g)[None, :]*np.exp(-B**2*dt**2/xi)
        for i, s in enumerate(sig):
            out[i] = np.real(np.sum(G*np.exp(1j*s*dt)))/(4*xi)
    return out.reshape(np.shape(sigma)) if np.ndim(sigma) else out[0]


def boyd_kleinman_optimum(xi, B=0.0, n_points=600):
    """Largest h over sigma at the focusing xi.

    Parameters
    ----------
    xi : number
    B : number, optional
    n_points : int, optional
        Gauss–Legendre points of `boyd_kleinman_h()`.

    Returns
    -------
    h, sigma : numbers
    """

    r = minimize_scalar(lambda s: -boyd_kleinman_h(xi, s, B, n_points), bounds=(-4.0, 4.0), method="bounded",
                        options={"xatol": 1E-6})
    return -r.fun, r.x


def shg_power_focused(P1, length, d, wavelength, n1, n2, h):
    """Second harmonic power 16 pi**2 d**2 L P1**2 h/(eps0 c lambda**3 n1 n2)
    of a focused Gaussian beam (Boyd and Kleinman; SI units, d in m/V).

    Parameters
    ----------
    P1 : number or array
        Fundamental power (W).
    length : number
        m.
    d : number
        m/V.
    wavelength : number
        Fundamental (m).
    n1, n2 : numbers
    h : number
        Focusing function.

    Returns
    -------
    P2 : array
    """

    return 16*np.pi**2*d**2*length*np.asarray(P1)**2*h/(EPS0*C0*wavelength**3*n1*n2)


# --- parametric amplification and oscillation -------------------------------------------

def parametric_gain_coefficient(I_p, d, wavelengths, n):
    """Gain coefficient Gamma = sqrt(omega_s omega_i) K sqrt(I_p) =
    sqrt(8 pi**2 d**2 I_p/(n_s n_i n_p eps0 c lambda_s lambda_i)) (1/m).

    Parameters
    ----------
    I_p : number or array
        Pump intensity (W/m²).
    d : number
        m/V.
    wavelengths : sequence (lambda_s, lambda_i, lambda_p)
        m.
    n : sequence (n_s, n_i, n_p)

    Returns
    -------
    Gamma : array
    """

    om = 2*np.pi*C0/np.asarray(wavelengths[:2], dtype=float)
    return np.sqrt(om[0]*om[1])*coupling_K(d, *n)*np.sqrt(np.asarray(I_p))


def opa_gain(Gamma, length, dk=0.0):
    """Signal power gain 1 + (Gamma/g)**2 sinh**2(g L), g = sqrt(Gamma**2 -
    (dk/2)**2) (an imaginary g turns sinh into sin) of an undepleted
    parametric amplifier without input idler.

    Parameters
    ----------
    Gamma : number or array
    length : number
    dk : number or array, optional

    Returns
    -------
    G : array
    """

    g2 = np.asarray(Gamma, dtype=float)**2 - (np.asarray(dk, dtype=float)/2)**2
    g = np.sqrt(g2.astype(complex))
    gl = g*length
    ratio = np.where(np.abs(gl) > 1E-12, np.sinh(gl)/np.where(np.abs(g) > 0, g, 1), length)
    return 1 + np.real(np.asarray(Gamma)**2*ratio**2)


def opo_threshold(length, d, wavelengths, n, r_s, r_i=0.0):
    """Plane wave pump threshold intensity of an OPO: cosh(Gamma L) =
    (1 + r_s r_i)/(r_s + r_i) for the amplitude feedback r_s and r_i of
    signal and idler per round trip (r_i = 0: singly resonant; for small
    losses Gamma**2 L**2 = 2 (1 - r_s) and, doubly resonant, (1 - r_s)(1 - r_i)).

    Parameters
    ----------
    length : number
        m.
    d : number
        m/V.
    wavelengths : sequence (lambda_s, lambda_i, lambda_p)
    n : sequence (n_s, n_i, n_p)
    r_s, r_i : numbers
        Round trip amplitude feedback (sqrt of the power fraction kept).

    Returns
    -------
    I_th : number
        W/m².
    """

    gl = np.arccosh((1 + r_s*r_i)/(r_s + r_i))
    Gamma1 = parametric_gain_coefficient(1.0, d, wavelengths, n)
    return (gl/(length*Gamma1))**2


def sro_pump_depletion(N):
    """Pump depletion sin**2(x) of a singly resonant OPO with a lossless
    (high Q) signal, plane waves, x**2/sin**2(x) = N the pump in units of
    the threshold (0 below threshold, 1 at N = pi**2/4, falling again above).

    Parameters
    ----------
    N : number or array

    Returns
    -------
    depletion : array
    """

    N = np.atleast_1d(np.asarray(N, dtype=float))
    out = np.zeros_like(N)
    for i, v in enumerate(N):
        if v <= 1:
            continue
        x = brentq(lambda x: x**2/np.sin(x)**2 - v, 1E-9, np.pi - 1E-12)
        out[i] = np.sin(x)**2
    return out if out.size > 1 else out[0]


def qpm_signal(crystal, wl_pump, period, T=None, order=1, max_idler=5000.0):
    """Signal wavelength (nm, below twice the pump wavelength) of a
    quasi-phase matched parametric process with the poling period
    `period` (m) at T (°C); nan if no signal phase matches.

    Parameters
    ----------
    crystal : str
    wl_pump : number
        nm.
    period : number
        m (at 25 °C).
    T : number, optional
    order : int, optional
    max_idler : number, optional
        Longest idler wavelength searched (nm; the index data end near
        5 µm).

    Returns
    -------
    wl_signal : number
    """

    def f(wl_s):
        wl_i = 1/(1/wl_pump - 1/wl_s)
        return float(delta_k(crystal, wl_s, wl_i, None, "eee", T, period, order))

    wl_lo = 1/(1/wl_pump - 1/max_idler)
    grid = np.linspace(wl_lo, wl_pump*2.0, 600)
    with np.errstate(invalid="ignore"):
        vals = np.array([f(x) for x in grid])
    ok = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
    idx = np.nonzero(ok & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return brentq(f, grid[i], grid[i + 1], xtol=1E-9)


def shg_pulse(t, A1, length, kappa, gvm=0.0, beta2=(0.0, 0.0), dk=0.0, A2=None, steps=400,
              n_save=None):
    """Second harmonic generation of pulses with group velocity mismatch
    and dispersion, by `pyoptics.splitstep.rk4ip()`.

    The envelopes (abs()**2 an intensity) are in the frame of the
    fundamental pulse; the harmonic walks off by gvm = 1/v_2 - 1/v_1 per
    unit length::

        dA1/dz = -i beta2_1/2 d²A1/dt² + i kappa A2 conj(A1)
        dA2/dz = -gvm dA2/dt - i beta2_2/2 d²A2/dt² - i dk A2 + i kappa A1**2,

    (the harmonic written as A2 exp(-i dk z)).

    Parameters
    ----------
    t : array
        Equally spaced times.
    A1 : complex array
        Fundamental envelope.
    length : number
    kappa : number
        `shg_kappa()`.
    gvm : number, optional
        Group velocity mismatch (units of t per unit length).
    beta2 : tuple, optional
        Group velocity dispersion of fundamental and harmonic.
    dk : number, optional
        2 k1 - k2 at the carriers.
    A2 : complex array, optional
        Input harmonic.
    steps : int, optional
    n_save : int, optional

    Returns
    -------
    A1, A2 : complex arrays
        Output envelopes (with n_save also z and the histories, as
        `pyoptics.splitstep.rk4ip()`).
    """

    Om = splitstep.angular_frequency(t)
    D1 = splitstep.dispersion_operator(Om, (beta2[0],))
    D2 = splitstep.dispersion_operator(Om, (beta2[1],)) + 1j*gvm*Om - 1j*dk
    A2 = np.zeros_like(A1, dtype=complex) if A2 is None else A2

    def nl(z, F):
        a1, a2 = F
        return [1j*kappa*a2*np.conj(a1), 1j*kappa*a1**2]

    res = splitstep.rk4ip(t, [np.asarray(A1, dtype=complex), np.asarray(A2, dtype=complex)],
                          [D1, D2], nl, length, steps=steps, n_save=n_save)
    if n_save is None:
        return res[0], res[1]
    return res[0][0], res[0][1], res[1], res[2]


# --- third order: Kerr effect --------------------------------------------------------------

def kerr_phase(I, n2, length, wavelength):
    """Nonlinear phase k0 n2 I L (rad) of the optical Kerr effect."""
    return 2*np.pi/wavelength*n2*np.asarray(I)*length


def kerr_lens_focal(P, w, n2, length):
    """Focal length pi w**4/(8 n2 L P) of the Kerr lens that a Gaussian beam
    of power P and radius w (1/e² intensity) makes in a thin medium of
    length L (parabolic approximation near the axis)."""
    return np.pi*w**4/(8*n2*length*np.asarray(P, dtype=float))


def critical_power(wavelength, n0, n2):
    """Critical power 3.77 lambda**2/(8 pi n0 n2) of self-focusing of a
    Gaussian beam (for the Townes profile the factor is 3.72; 2.5 MW in
    fused silica and 3 GW in air at 800 nm)."""
    return 3.77*wavelength**2/(8*np.pi*n0*n2)


def self_focusing_distance(P, P_cr, w0, wavelength, n0=1.0, R=np.inf):
    """Marburger's collapse distance of a Gaussian beam above the critical
    power,

        z_c = 0.367 z_R/sqrt((sqrt(P/P_cr) - 0.852)**2 - 0.0219),

    with the Rayleigh length z_R = k w0**2/2, w0 the 1/e² radius of the
    intensity and k = n0 k0 (Couairon and Mysyrowicz); with a focusing lens of focal length R in front,
    1/z = 1/z_c + 1/R.

    Parameters
    ----------
    P, P_cr : numbers or arrays
    w0 : number
    wavelength : number
    n0 : number, optional
    R : number, optional
        Focal length of a lens before the medium (inf: collimated).

    Returns
    -------
    z : array
        nan below the power where the formula applies.
    """

    k = 2*np.pi*n0/wavelength
    p = np.sqrt(np.asarray(P, dtype=float)/P_cr)
    arg = (p - 0.852)**2 - 0.0219
    with np.errstate(invalid="ignore", divide="ignore"):
        zc = np.where((arg > 0) & (p > 0.852), 0.367*k*w0**2/2/np.sqrt(np.abs(arg)), np.nan)
    if np.isinf(R):
        return zc
    return 1/(1/zc + 1/R)


def kerr_lens_mode(cavity, P, n2, length, wavelength, n=1.0, iterations=200, damping=0.5, tol=1E-10):
    """Self-consistent Gaussian mode of a resonator with a thin Kerr lens.

    Parameters
    ----------
    cavity : callable
        cavity(f_kerr) -> (M_rt, M_out): the round trip ABCD matrix starting
        and ending at the Kerr medium, with the Kerr lens of focal length
        f_kerr included where it acts, and the matrix from the Kerr medium
        to the plane of interest (an aperture).
    P : number
        Intracavity peak power (W).
    n2 : number
        m²/W.
    length : number
        Length of the Kerr medium (m).
    wavelength : number
        m.
    n : number, optional
        Index at the reference plane of M_rt.
    iterations, damping, tol : optional
        Fixed point iteration on the spot size in the Kerr medium.

    Returns
    -------
    w_kerr, w_out : numbers
        Beam radii in the Kerr medium and at the output plane (nan if the
        resonator is unstable at this power).
    """

    w = None
    for _ in range(iterations):
        f = np.inf if (w is None or P == 0) else kerr_lens_focal(P, w, n2, length)
        M_rt, M_out = cavity(f)
        if not abcd.is_stable(M_rt):
            return np.nan, np.nan
        q = abcd.eigenmode(M_rt, n=n)
        w_new = abcd.beam_radius(q, wavelength, n)
        if w is not None and abs(w_new - w) < tol*w:
            w = w_new
            break
        w = w_new if w is None else damping*w_new + (1 - damping)*w
    M_rt, M_out = cavity(np.inf if P == 0 else kerr_lens_focal(P, w, n2, length))
    q = abcd.eigenmode(M_rt, n=n)
    w_out = abcd.beam_radius(abcd.transform_q(M_out, q, n, n), wavelength, n)
    return w, w_out


def kerr_lens_loss(cavity, aperture, n2, length, wavelength, P_max, n_powers=200, n=1.0):
    """Effective fast saturable loss of Kerr-lens mode-locking for
    `pyoptics.lasers.master_equation()`.

    The self-consistent mode (`kerr_lens_mode()`) is found on a grid of
    powers; a hard circular aperture of radius a passes the fraction
    1 - exp(-2 a**2/w**2) of a Gaussian beam of radius w. The returned
    function q(P) is the amplitude loss per round trip,
    -ln(1 - exp(-2 a**2/w(P)**2))/2, interpolated in P.

    Parameters
    ----------
    cavity : callable
        As in `kerr_lens_mode()`.
    aperture : number
        Radius (m).
    n2, length, wavelength : numbers
    P_max : number
        Largest power (W) of the grid; q is constant beyond it.
    n_powers : int, optional
    n : number, optional

    Returns
    -------
    q : callable
        q(P) for P in W (arrays allowed).
    table : dict
        "P", "w_kerr", "w_aperture" and "loss" on the grid.
    """

    Ps = np.linspace(0.0, P_max, n_powers)
    wk, wa = np.empty(n_powers), np.empty(n_powers)
    for i, P in enumerate(Ps):
        wk[i], wa[i] = kerr_lens_mode(cavity, P, n2, length, wavelength, n=n)
    T = 1 - np.exp(-2*aperture**2/wa**2)
    loss = -0.5*np.log(T)
    good = np.isfinite(loss)
    Pg, lg = Ps[good], loss[good]

    def q(P):
        return np.interp(P, Pg, lg)

    return q, {"P": Ps, "w_kerr": wk, "w_aperture": wa, "loss": loss}
