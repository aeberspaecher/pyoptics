#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Lasers: gain, threshold, modes, pulses and semiconductor laser diodes.

SI units throughout (m, s, W, J, Hz) unless a function says otherwise; the
semiconductor laser transmitter `DiodeLaser.nrz()` returns its field in the
units of `pyoptics.fibrecomm` (ps, mW). Fields follow the pyoptics convention
exp(i(kz - omega t)): an instantaneous frequency shift is -d phi/dt of the
envelope phase phi.

Gain media
----------
The Einstein coefficients of a transition with the spontaneous lifetime
tau_rad = 1/A21 (einstein_coefficients()) and its normalized line shape g(nu)
(lorentzian(), gaussian_line()) give the emission cross section ::

    sigma(nu) = lambda**2/(8 pi n**2 tau_rad) g(nu)

(cross_section(), peak_cross_section()). An inversion density Delta N gives
the gain coefficient g = sigma Delta N (1/m). Light of the intensity I
depletes it: a homogeneously broadened line saturates as
g0/(1 + I/I_sat), an inhomogeneous one as g0/sqrt(1 + I/I_sat), with
I_sat = h nu/(sigma tau) (saturation_intensity(), saturated_gain()). A
saturated amplifier has the gain of amplifier_gain() (continuous wave,
through the Lambert W function) or frantz_nodvik() (short pulses).
GAIN_MEDIA lists cross sections, lifetimes and line widths of common
lasers.

Level schemes
-------------
three_level_inversion() and four_level_inversion() are the steady states of
the populations under a pump rate W_p and a stimulated rate W_s. A three
level laser (ruby) must pump more than half of its atoms before it has any
gain, a four level laser (Nd:YAG) has gain from the first pumped atom on.

The time dependent laser is described by the rate equations of the
inversion N and the photon number Phi (rate_equations()), normalized to the
threshold inversion N_th and to the photon number N_th tau_c/tau at which
the pump is twice the threshold::

    dn/dt = (r - n - n p)/tau
    dp/dt = ((n - q(t)) p + beta n)/tau_c

with the pump r = R_p tau/N_th relative to threshold, the upper level
lifetime tau, the photon lifetime tau_c of the high Q cavity, the cavity
loss q(t) relative to it (an active Q switch) and the fraction beta of the
spontaneous emission that goes into the mode. A saturable absorber
(passive Q switching) adds a loss q_a a(t) whose absorbers bleach and
recover, da/dt = (1 - a)/tau_a - s a p/tau. Above threshold the steady
state is n = 1, p = r - 1; small deviations ring at the relaxation
oscillation frequency sqrt((r - 1)/(tau tau_c)) and decay at the rate
r/(2 tau) (relaxation_oscillation()). q_switch() gives the energy
extraction efficiency, peak power and length of a Q switched pulse from the
initial inversion (Wagner and Birnbaum).

Resonators
----------
threshold_gain(), photon_lifetime() and round-trip losses; the output of a
laser with a homogeneous gain medium against the output coupling
transmission T, in the low loss limit (output_power(),
optimum_output_coupling()) and exact for any mirror reflectivities
(rigrod_output()); slope efficiency and threshold pump power of an end
pumped laser (slope_efficiency(), threshold_pump_power()).
free_spectral_range(), mode_frequencies() and transverse_mode_frequencies()
(from the round-trip Gouy phase of `pyoptics.abcd`), cavity_linewidth()
and the Schawlow-Townes linewidth (schawlow_townes()), with Henry's
(1 + alpha**2) for semiconductor lasers.

Mode locking
------------
mode_locked_field() adds the longitudinal modes with locked or random
phases. master_equation() iterates Haus's master equation of mode locking
round trip by round trip: saturated gain with a parabolic gain filter,
loss, an active amplitude modulator, a fast saturable loss, group delay
dispersion and self-phase modulation. The saturable loss may be given as a
function of the instantaneous power, which is the hook for an effective
Kerr lens (Kerr-lens mode locking). active_ml_width() is the
Kuizenga-Siegman pulse length of active mode locking.

Laser diodes
------------
DiodeLaser holds the single-mode rate equations of a semiconductor laser
for the carrier number N and the photon number S in the mode::

    dN/dt = I/q - V R(N/V) - G S
    dS/dt = (G - 1/tau_p) S + R_sp
    dphi/dt = -(alpha/2) Gamma v_g a (N/V - n_th)

with the recombination R(n) = A n + B n**2 + C n**3, the modal gain
G = Gamma v_g a (n - n_tr)/(1 + eps s) per second (s = Gamma S/V the photon
density), the photon lifetime tau_p = 1/(v_g (alpha_i + alpha_m)), the
spontaneous emission into the mode R_sp = beta_sp B n**2 V, the linewidth
enhancement factor alpha and the threshold density n_th. Defaults are the
1.55 um InGaAsP DFB laser of Agrawal's table 3.1. Methods give the
threshold current, the L-I curve, the small-signal modulation response and
relaxation oscillation frequency (from the Jacobian of the rate equations),
the relative intensity noise and the linewidth (from the Langevin forces),
time domain simulations with or without noise, an NRZ transmitter in the
units of `pyoptics.fibrecomm`, and the spectrum of a multimode Fabry-Perot
laser against a DFB laser.

References
----------
A. E. Siegman, "Lasers", University Science Books (1986).

O. Svelto, "Principles of Lasers", 5th ed., Springer (2010).

W. Koechner, "Solid-State Laser Engineering", 6th ed., Springer (2006).

B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019), chapters 15 to 18.

W. W. Rigrod, "Saturation effects in high-gain lasers", J. Appl. Phys. 36,
2487-2490 (1965).

W. G. Wagner and B. A. Lengyel, "Evolution of the giant pulse in a laser",
J. Appl. Phys. 34, 2040-2046 (1963).

H. A. Haus, "Mode-locking of lasers", IEEE J. Sel. Top. Quantum Electron. 6,
1173-1185 (2000).

D. J. Kuizenga and A. E. Siegman, "FM and AM mode locking of the
homogeneous laser", IEEE J. Quantum Electron. 6, 694-708 (1970).

C. H. Henry, "Theory of the linewidth of semiconductor lasers", IEEE J.
Quantum Electron. 18, 259-264 (1982).

G. P. Agrawal, "Fiber-Optic Communication Systems", 4th ed., Wiley (2010),
chapter 3.

L. A. Coldren, S. W. Corzine and M. L. Mashanovitch, "Diode Lasers and
Photonic Integrated Circuits", 2nd ed., Wiley (2012), chapters 2, 5 and 6.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw, wrightomega

from . import abcd


__all__ = ["H", "C0", "Q_E", "GAIN_MEDIA", "photon_energy", "lorentzian", "gaussian_line",
           "einstein_coefficients", "cross_section", "peak_cross_section",
           "saturation_intensity", "saturated_gain", "amplifier_gain", "frantz_nodvik",
           "three_level_inversion", "four_level_inversion", "rate_equations",
           "relaxation_oscillation", "q_switch", "threshold_gain", "round_trip_loss",
           "photon_lifetime", "output_power", "optimum_output_coupling", "rigrod_output",
           "slope_efficiency", "threshold_pump_power", "free_spectral_range",
           "mode_frequencies", "transverse_mode_frequencies", "cavity_linewidth",
           "schawlow_townes", "mode_locked_field", "time_bandwidth_product",
           "active_ml_width", "master_equation", "DiodeLaser"]


H = 6.62607015E-34      # J s
C0 = 2.99792458E8       # m/s
Q_E = 1.602176634E-19   # C


def photon_energy(wavelength):
    """h c/lambda (J) for the vacuum wavelength (m)."""
    return H*C0/np.asarray(wavelength, dtype=float)


# Common gain media: wavelength (m), stimulated emission cross section (m²),
# upper level lifetime (s), refractive index, gain line width FWHM (Hz) and
# broadening; ions/m³ at a typical doping; pump wavelength (m).
GAIN_MEDIA = {
    "Nd:YAG": dict(wavelength=1064.2E-9, sigma=2.8E-23, tau=230E-6, n=1.82, dnu=120E9,
                   broadening="homogeneous", N=1.38E26, pump=808E-9, levels=4),
    "Nd:YVO4": dict(wavelength=1064.3E-9, sigma=15.6E-23, tau=98E-6, n=2.17, dnu=260E9,
                    broadening="homogeneous", N=1.25E26, pump=808E-9, levels=4),
    "Ruby": dict(wavelength=694.3E-9, sigma=2.5E-24, tau=3.0E-3, n=1.76, dnu=330E9,
                 broadening="homogeneous", N=1.6E25, pump=545E-9, levels=3),
    "Ti:sapphire": dict(wavelength=800E-9, sigma=3.0E-23, tau=3.2E-6, n=1.76, dnu=100E12,
                        broadening="homogeneous", N=3.3E25, pump=532E-9, levels=4),
    "Yb:YAG": dict(wavelength=1030E-9, sigma=2.0E-24, tau=0.95E-3, n=1.82, dnu=2.6E12,
                   broadening="homogeneous", N=1.38E27, pump=940E-9, levels=3),
    "Er:glass": dict(wavelength=1535E-9, sigma=0.8E-24, tau=8E-3, n=1.53, dnu=4E12,
                     broadening="inhomogeneous", N=1E26, pump=980E-9, levels=3),
    "HeNe": dict(wavelength=632.8E-9, sigma=3E-17, tau=0.7E-6, n=1.0, dnu=1.5E9,
                 broadening="inhomogeneous", N=None, pump=None, levels=4),
}


# --- line shapes and Einstein coefficients ---------------------------------

def lorentzian(nu, nu0, dnu):
    """Normalized Lorentzian line shape (1/Hz) of FWHM dnu, integral 1."""
    return (dnu/(2*np.pi))/((np.asarray(nu) - nu0)**2 + (dnu/2)**2)


def gaussian_line(nu, nu0, dnu):
    """Normalized Gaussian (Doppler) line shape (1/Hz) of FWHM dnu."""
    s = dnu/(2*np.sqrt(2*np.log(2)))
    return np.exp(-(np.asarray(nu) - nu0)**2/(2*s**2))/(np.sqrt(2*np.pi)*s)


def einstein_coefficients(A21, wavelength, g1=1, g2=1, n=1.0):
    """Einstein coefficients of a transition.

    Parameters
    ----------
    A21 : number
        Spontaneous emission rate (1/s), 1/tau_rad.
    wavelength : number
        Vacuum wavelength (m).
    g1, g2 : numbers, optional
        Degeneracies of the lower and upper level.
    n : number, optional
        Refractive index of the host.

    Returns
    -------
    dict
        A21, B21 and B12 (m³/(J s²), for the spectral energy density per unit
        frequency): B21 = A21 lambda**3/(8 pi h n**3), g1 B12 = g2 B21.
    """

    B21 = A21*wavelength**3/(8*np.pi*H*n**3)
    return {"A21": A21, "B21": B21, "B12": g2/g1*B21}


def cross_section(nu, nu0, wavelength, tau_rad, dnu, shape="lorentzian", n=1.0):
    """Emission cross section (m²) sigma(nu) = lambda**2/(8 pi n**2 tau_rad) g(nu).

    Parameters
    ----------
    nu : number or array
        Frequency (Hz).
    nu0 : number
        Line centre (Hz).
    wavelength : number
        Vacuum wavelength of the line centre (m).
    tau_rad : number
        Radiative lifetime (s).
    dnu : number
        Line width (FWHM, Hz).
    shape : str, optional
        "lorentzian" (homogeneous) or "gaussian" (inhomogeneous).
    n : number, optional
        Refractive index.

    Returns
    -------
    array
    """

    g = lorentzian(nu, nu0, dnu) if shape == "lorentzian" else gaussian_line(nu, nu0, dnu)
    return wavelength**2/(8*np.pi*n**2*tau_rad)*g


def peak_cross_section(wavelength, tau_rad, dnu, shape="lorentzian", n=1.0):
    """Cross section (m²) at the line centre: g(nu0) = 2/(pi dnu) for a
    Lorentzian, 2 sqrt(ln 2/pi)/dnu for a Gaussian."""
    g0 = 2/(np.pi*dnu) if shape == "lorentzian" else 2*np.sqrt(np.log(2)/np.pi)/dnu
    return wavelength**2/(8*np.pi*n**2*tau_rad)*g0


# --- gain and saturation ---------------------------------------------------

def saturation_intensity(wavelength, sigma, tau, levels=4):
    """Saturation intensity (W/m²) h nu/(sigma tau) of a four level medium,
    h nu/(2 sigma tau) of a three level medium (equal degeneracies)."""
    return photon_energy(wavelength)/(sigma*tau*(2 if levels == 3 else 1))


def saturated_gain(g0, I, I_sat, broadening="homogeneous"):
    """Gain coefficient at the intensity I: g0/(1 + I/I_sat) (homogeneous
    line) or g0/sqrt(1 + I/I_sat) (inhomogeneous line, at its centre)."""
    x = 1 + np.asarray(I, dtype=float)/I_sat
    return g0/(x if broadening == "homogeneous" else np.sqrt(x))


def amplifier_gain(G0, I_in, I_sat):
    """Gain of a homogeneously saturated continuous wave amplifier.

    Integrates dI/dz = g0 I/(1 + I/I_sat): ln(G/G0) + (G - 1) I_in/I_sat = 0,
    solved with the Lambert W function, G = W(x G0 exp(x))/x with
    x = I_in/I_sat.

    Parameters
    ----------
    G0 : number
        Small signal gain exp(g0 L).
    I_in : number or array
        Input intensity (or power, with I_sat as a power).
    I_sat : number
        Saturation intensity.

    Returns
    -------
    array
        Gain I_out/I_in.
    """

    x = np.asarray(I_in, dtype=float)/I_sat
    with np.errstate(divide="ignore", invalid="ignore"):
        # W(x G0 exp(x)) = omega(ln x + ln G0 + x), Wright's omega, without overflow
        G = wrightomega(np.log(x) + np.log(G0) + x).real/x
    return np.where(x > 0, G, G0)


def frantz_nodvik(E_in, E_sat, G0):
    """Output fluence of a pulse much shorter than the upper level lifetime,
    E_out = E_sat ln(1 + G0 (exp(E_in/E_sat) - 1)) (Frantz and Nodvik);
    E_sat = h nu/sigma for a four level medium."""
    return E_sat*np.log1p(G0*np.expm1(np.asarray(E_in, dtype=float)/E_sat))


def three_level_inversion(W_p, tau, N=1.0, W_s=0.0):
    """Steady state inversion N2 - N1 of a three level laser with equal
    degeneracies and a fast decay from the pump band:
    N (W_p tau - 1)/(1 + W_p tau + 2 W_s tau). Positive (gain) only for
    W_p tau > 1, i.e. more than half the atoms pumped."""
    W_p = np.asarray(W_p, dtype=float)
    return N*(W_p*tau - 1)/(1 + W_p*tau + 2*W_s*tau)


def four_level_inversion(W_p, tau, N=1.0, W_s=0.0):
    """Steady state inversion of a four level laser with an empty lower
    level and pumping from the ground state: N W_p tau/(1 + W_p tau + W_s tau)."""
    W_p = np.asarray(W_p, dtype=float)
    return N*W_p*tau/(1 + W_p*tau + W_s*tau)


# --- rate equations --------------------------------------------------------

def _as_function(x):
    if callable(x):
        return x
    return lambda t: x


def rate_equations(t, r, tau, tau_c, beta=1E-8, n0=0.0, p0=None, loss=1.0, absorber=None,
                   rtol=1E-8, atol=1E-10, max_step=None):
    """Integrate the normalized laser rate equations (see the module docstring).

    The photon number is integrated as ln p, which spans the many decades of
    the build-up of a Q switched pulse without trouble.

    Parameters
    ----------
    t : array
        Output times (s), increasing, starting at the initial time.
    r : number or callable
        Pump rate relative to threshold, r(t).
    tau : number
        Upper level lifetime (s).
    tau_c : number
        Photon lifetime of the high Q cavity (s).
    beta : number, optional
        Fraction of the spontaneous emission into the lasing mode.
    n0 : number, optional
        Initial inversion relative to threshold.
    p0 : number, optional
        Initial photon number (default: the spontaneous level beta n0).
    loss : number or callable, optional
        Cavity loss relative to the high Q cavity, q(t) >= 1.
    absorber : tuple, optional
        (q_a, tau_a, s): a saturable absorber with the unbleached loss q_a
        (relative to the high Q loss), recovery time tau_a and saturation
        ratio s = (sigma_a/sigma) (A/A_a) (absorber against gain cross
        sections and mode areas). Starts unbleached.
    rtol, atol : numbers, optional
        Tolerances of the integrator.
    max_step : number, optional
        Largest time step (s); default a tenth of the smallest of tau and
        the output sampling.

    Returns
    -------
    n, p : arrays
        Inversion relative to threshold and photon number in units of
        N_th tau_c/tau at the times t (the steady state above threshold is
        n = 1, p = r - 1); with an absorber also the unbleached fraction a.
    """

    t = np.asarray(t, dtype=float)
    r_f, q_f = _as_function(r), _as_function(loss)
    p0 = max(beta*max(n0, 1E-3), 1E-30) if p0 is None else max(p0, 1E-30)
    y0 = [n0, np.log(p0)]
    if absorber is not None:
        q_a, tau_a, s = absorber
        y0.append(1.0)

    def rhs(tt, y):
        n, lp = y[0], y[1]
        p = np.exp(min(lp, 700.0))
        dn = (r_f(tt) - n - n*p)/tau
        q = q_f(tt)
        out = [dn, 0.0]
        if absorber is not None:
            a = y[2]
            q = q + q_a*a
            out.append((1 - a)/tau_a - s*a*p/tau)
        out[1] = ((n - q) + beta*max(n, 0.0)/p)/tau_c
        return out

    if max_step is None:
        dt = np.min(np.diff(t)) if len(t) > 1 else tau
        max_step = min(tau, dt)/2
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="LSODA", rtol=rtol, atol=atol,
                    max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    n, p = sol.y[0], np.exp(sol.y[1])
    if absorber is not None:
        return n, p, sol.y[2]
    return n, p


def relaxation_oscillation(r, tau, tau_c):
    """Relaxation oscillation of a laser pumped r times above threshold.

    Returns
    -------
    f_R : number
        Frequency (Hz), sqrt((r - 1)/(tau tau_c) - (r/(2 tau))**2)/(2 pi).
    gamma : number
        Amplitude decay rate (1/s), r/(2 tau).
    """

    r = np.asarray(r, dtype=float)
    gamma = r/(2*tau)
    w2 = (r - 1)/(tau*tau_c) - gamma**2
    return np.sqrt(np.maximum(w2, 0.0))/(2*np.pi), gamma


def q_switch(n_i, tau_c):
    """Giant pulse of an ideal Q switched four level laser (switching much
    faster than the build-up, pumping negligible during the pulse).

    Parameters
    ----------
    n_i : number or array
        Initial inversion relative to the threshold of the high Q cavity.
    tau_c : number
        Photon lifetime of the high Q cavity (s).

    Returns
    -------
    dict
        n_f: final inversion, the root below 1 of n_i - n_f = ln(n_i/n_f);
        efficiency: (n_i - n_f)/n_i, the fraction of the inversion
        extracted; p_peak: peak photon number in units of N_th,
        n_i - 1 - ln(n_i); width: the pulse energy over the peak power,
        tau_c (n_i - n_f)/(n_i - 1 - ln(n_i)) (s).
    """

    n_i = np.asarray(n_i, dtype=float)
    n_f = -np.real(lambertw(-n_i*np.exp(-n_i)))
    p_peak = n_i - 1 - np.log(n_i)
    return {"n_f": n_f, "efficiency": (n_i - n_f)/n_i, "p_peak": p_peak,
            "width": tau_c*(n_i - n_f)/p_peak}


# --- resonators ------------------------------------------------------------

def round_trip_loss(R1, R2, alpha_i=0.0, L=0.0):
    """Logarithmic round trip loss -ln(R1 R2) + 2 alpha_i L."""
    return -np.log(np.asarray(R1)*R2) + 2*alpha_i*L


def threshold_gain(R1, R2, L, alpha_i=0.0):
    """Threshold gain coefficient (1/m) alpha_i - ln(R1 R2)/(2 L) of a gain
    medium of length L that fills a linear cavity."""
    return alpha_i - np.log(np.asarray(R1)*R2)/(2*L)


def photon_lifetime(L, R1, R2, alpha_i=0.0, n=1.0, l_medium=None):
    """Photon lifetime tau_c = 2 L_opt/(c delta) (s) of a linear cavity with the
    optical length L_opt = n L and the round trip loss delta (internal loss
    alpha_i over the length l_medium, default L)."""
    l_medium = L if l_medium is None else l_medium
    return 2*n*L/(C0*round_trip_loss(R1, R2, alpha_i, l_medium))


def output_power(T, g0l2, delta_i, P_sat):
    """Output power of a laser with a homogeneous gain medium, in the limit
    of small losses (standing wave, Koechner eq. 3.70)::

        P_out = (T P_sat/2) (g0l2/(T + delta_i) - 1)

    Parameters
    ----------
    T : number or array
        Transmission of the output coupler.
    g0l2 : number
        Round trip small signal gain 2 g0 l.
    delta_i : number
        Round trip internal loss.
    P_sat : number
        Saturation power I_sat A of the mode (W).

    Returns
    -------
    array
        W, zero below threshold.
    """

    T = np.asarray(T, dtype=float)
    return np.maximum(T*P_sat/2*(g0l2/(T + delta_i) - 1), 0.0)


def optimum_output_coupling(g0l2, delta_i, P_sat=None):
    """Output coupling sqrt(g0l2 delta_i) - delta_i that maximizes
    output_power(); with P_sat also the maximum power
    P_sat (sqrt(g0l2) - sqrt(delta_i))**2/2."""
    T = np.sqrt(g0l2*delta_i) - delta_i
    if P_sat is None:
        return T
    return T, P_sat*(np.sqrt(g0l2) - np.sqrt(delta_i))**2/2


def rigrod_output(g0, l, R1, R2, I_sat):
    """Output intensity through mirror 2 of a laser with a homogeneous gain
    medium of length l filling the cavity, any gain and reflectivities, no
    internal loss (Rigrod)::

        I_out = I_sat (1 - R2) sqrt(R1) (g0 l + ln sqrt(R1 R2))
                /((sqrt(R1) + sqrt(R2))(1 - sqrt(R1 R2)))

    Returns
    -------
    array
        Intensity (units of I_sat), zero below threshold.
    """

    R2 = np.asarray(R2, dtype=float)
    s = np.sqrt(R1*R2)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = I_sat*(1 - R2)*np.sqrt(R1)*(g0*l + np.log(s))/((np.sqrt(R1) + np.sqrt(R2))*(1 - s))
    return np.maximum(np.nan_to_num(I), 0.0)


def slope_efficiency(T, delta_i, wavelength_pump, wavelength_laser, eta_pump=1.0,
                     eta_mode=1.0):
    """Slope efficiency dP_out/dP_pump = eta_pump eta_mode (lambda_p/lambda_l)
    T/(T + delta_i): absorbed pump fraction, mode overlap, quantum defect and
    output coupling efficiency."""
    T = np.asarray(T, dtype=float)
    return eta_pump*eta_mode*wavelength_pump/wavelength_laser*T/(T + delta_i)


def threshold_pump_power(delta, sigma, tau, wavelength_pump, area, eta_pump=1.0):
    """Absorbed pump power (W) at threshold of a four level laser,
    delta h nu_p A/(2 sigma tau eta_pump), with the round trip loss delta and
    the effective mode area A (pi (w_l**2 + w_p**2)/2 for Gaussian laser and
    pump beams of radii w_l and w_p)."""
    return delta*photon_energy(wavelength_pump)*area/(2*sigma*tau*eta_pump)


# --- modes and linewidth ---------------------------------------------------

def free_spectral_range(L, n=1.0, ring=False):
    """Mode spacing (Hz) c/(2 n L) of a linear cavity, c/(n L) of a ring."""
    return C0/((1 if ring else 2)*n*np.asarray(L, dtype=float))


def mode_frequencies(L, nu_min, nu_max, n=1.0, ring=False):
    """Frequencies q FSR of the longitudinal modes in [nu_min, nu_max]
    (Hz) and their indices q."""
    fsr = free_spectral_range(L, n, ring)
    q = np.arange(np.ceil(nu_min/fsr), np.floor(nu_max/fsr) + 1)
    return q*fsr, q.astype(int)


def transverse_mode_frequencies(M, L_round_trip, q, max_order=3, n=1.0):
    """Frequencies (Hz) of the Hermite-Gaussian modes TEM_mn of a stable
    cavity, nu = (c/(n L_rt)) (q + (m + n + 1) psi/(2 pi)) with the round
    trip Gouy phase psi of the round trip matrix M
    (`pyoptics.abcd.round_trip_gouy_phase()`).

    Parameters
    ----------
    M : 2x2 array
        Round trip ray matrix.
    L_round_trip : number
        Geometrical round trip length (m), twice the length of a linear
        cavity.
    q : int or array
        Longitudinal mode numbers.
    max_order : int, optional
        Largest transverse order m + n.
    n : number, optional
        Refractive index.

    Returns
    -------
    nu : array, shape (len(q), max_order + 1)
        Frequency of each longitudinal mode and transverse order m + n.
    """

    psi = abcd.round_trip_gouy_phase(M)
    fsr = C0/(n*L_round_trip)
    q = np.atleast_1d(q)[:, None]
    order = np.arange(max_order + 1)[None, :]
    return fsr*(q + (order + 1)*psi/(2*np.pi))


def cavity_linewidth(tau_c):
    """FWHM (Hz) 1/(2 pi tau_c) of a passive cavity mode."""
    return 1/(2*np.pi*np.asarray(tau_c, dtype=float))


def schawlow_townes(nu, dnu_c, P_out, n_sp=1.0, alpha=0.0):
    """Quantum limited linewidth (FWHM, Hz) of a laser well above threshold,
    pi h nu dnu_c**2 n_sp (1 + alpha**2)/P_out (Schawlow and Townes in the
    modern form with the passive cavity linewidth dnu_c, the population
    inversion factor n_sp = N2/(N2 - N1) and Henry's factor for a linewidth
    enhancement factor alpha)."""
    return np.pi*H*nu*dnu_c**2*n_sp*(1 + alpha**2)/np.asarray(P_out, dtype=float)


# --- mode locking ----------------------------------------------------------

def mode_locked_field(t, n_modes, fsr, phases="locked", amplitudes=None, rng=None):
    """Envelope of the sum of n_modes longitudinal modes,
    sum_k a_k exp(-i (2 pi k fsr t - phi_k)), k centred on zero.

    Parameters
    ----------
    t : array
        Time (s).
    n_modes : int
    fsr : number
        Mode spacing (Hz).
    phases : str or array, optional
        "locked" (all zero), "random" (uniform) or an array of n_modes phases.
    amplitudes : array, optional
        Mode amplitudes (default all 1).
    rng : seed or Generator, optional

    Returns
    -------
    complex array
        Field; abs()**2 is the power in units of one mode.
    """

    k = np.arange(n_modes) - (n_modes - 1)/2
    if isinstance(phases, str):
        if phases == "locked":
            phases = np.zeros(n_modes)
        else:
            phases = np.random.default_rng(rng).uniform(0, 2*np.pi, n_modes)
    a = np.ones(n_modes) if amplitudes is None else np.asarray(amplitudes)
    t = np.asarray(t, dtype=float)
    return np.sum(a[:, None]*np.exp(-1j*(2*np.pi*k[:, None]*fsr*t[None, :] - np.asarray(phases)[:, None])),
                  axis=0)


def time_bandwidth_product(shape="gaussian"):
    """FWHM duration times FWHM bandwidth of a transform limited pulse:
    0.441 (Gaussian), 0.315 (sech²), 0.142 (Lorentzian... single sided
    exponential: 0.110)."""
    return {"gaussian": 2*np.log(2)/np.pi, "sech2": 0.3148, "exponential": 0.1103,
            "lorentzian": 0.1420}[shape]


def active_ml_width(g, M, f_m, dnu_g):
    """FWHM pulse length (s) of active AM mode locking (Kuizenga and Siegman,
    in the master equation form)::

        tau = 2 sqrt(ln 2) (2 g/(M omega_m**2 Omega_g**2))**(1/4)
            = 0.446 (g/M)**(1/4)/sqrt(f_m dnu_g)

    with the saturated round trip amplitude gain g, the modulation depth M
    (round trip amplitude loss M (1 - cos omega_m t)), the modulation
    frequency f_m and the gain bandwidth dnu_g (FWHM, Omega_g = pi dnu_g)."""
    w_m, W_g = 2*np.pi*f_m, np.pi*dnu_g
    return 2*np.sqrt(np.log(2))*(2*g/(M*w_m**2*W_g**2))**0.25


def master_equation(t, A0, round_trips, g0, E_sat, dnu_g, loss, modulation=None,
                    saturable_loss=None, gdd=0.0, spm=0.0, record_every=1):
    """Haus's master equation of mode locking, iterated round trip by round
    trip on a periodic time window (split step).

    Per round trip the envelope (pyoptics convention, A~(Omega) = int A
    exp(i Omega t) dt) is multiplied by::

        exp(g/(1 + (Omega/Omega_g)**2) - l + i gdd Omega**2/2)      (spectrum)
        exp(-M (1 - cos(omega_m t)) - q(P) + i spm P)                (time)

    with the gain saturated by the pulse energy, g = g0/(1 + E/E_sat), the
    gain half width Omega_g = pi dnu_g, the amplitude losses l (linear),
    M (modulator), q(P) (fast saturable loss of the instantaneous power
    P = abs(A)**2) and the self-phase modulation coefficient spm (1/W, the
    pyoptics sign: Kerr phase +k0 n2 I L). All gains and losses are for the
    field amplitude per round trip.

    Parameters
    ----------
    t : array
        Equally spaced time window, one modulation period or a window
        around the pulse.
    A0 : complex array
        Initial field (e.g. noise or a weak pulse).
    round_trips : int
    g0 : number
        Small signal amplitude gain per round trip.
    E_sat : number
        Saturation energy of the gain (in the units of sum(abs(A)**2) dt).
    dnu_g : number
        Gain bandwidth (FWHM) in the inverse units of t.
    loss : number
        Linear amplitude loss l per round trip.
    modulation : tuple, optional
        (M, f_m): active amplitude modulator with depth M at frequency f_m.
    saturable_loss : tuple or callable, optional
        (q0, P_sat): fast saturable absorber, q = q0/(1 + P/P_sat); or a
        function q(P) of the instantaneous power, e.g. the effective loss of
        a Kerr lens and an aperture (Kerr-lens mode locking).
    gdd : number, optional
        Round trip group delay dispersion (units of t squared).
    spm : number, optional
        Self-phase modulation per unit power.
    record_every : int, optional
        Keep every n-th round trip.

    Returns
    -------
    A : complex array
        Field after the last round trip.
    history : dict
        "energy", "gain" per kept round trip and "fields", the kept fields
        (array of shape (kept, len(t))).
    """

    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    Om = 2*np.pi*np.fft.fftfreq(len(t), dt)
    W_g = np.pi*dnu_g
    filt = 1/(1 + (Om/W_g)**2)
    disp = np.exp(1j*gdd*Om**2/2)
    tmod = 0.0
    if modulation is not None:
        M, f_m = modulation
        tc = t - t[len(t)//2]
        tmod = M*(1 - np.cos(2*np.pi*f_m*tc))
    if saturable_loss is None:
        q_of = None
    elif callable(saturable_loss):
        q_of = saturable_loss
    else:
        q0, P_sat = saturable_loss
        q_of = lambda P: q0/(1 + P/P_sat)
    A = np.asarray(A0, dtype=complex).copy()
    hist = {"energy": [], "gain": [], "fields": []}
    for k in range(round_trips):
        E = np.sum(np.abs(A)**2)*dt
        g = g0/(1 + E/E_sat)
        A = np.fft.ifft(np.fft.fft(A)*np.exp(g*filt - loss)*disp)
        P = np.abs(A)**2
        expo = -tmod + 1j*spm*P
        if q_of is not None:
            expo = expo - q_of(P)
        A = A*np.exp(expo)
        if k % record_every == 0 or k == round_trips - 1:
            hist["energy"].append(E)
            hist["gain"].append(g)
            hist["fields"].append(A.copy())
    hist = {key: np.array(val) for key, val in hist.items()}
    return A, hist


# --- semiconductor laser diodes ----------------------------------------------

@dataclass
class DiodeLaser:
    """Single-mode rate equations of a semiconductor laser (see the module
    docstring). SI units; the defaults are the 1.55 um InGaAsP DFB laser of
    Agrawal, "Fiber-Optic Communication Systems", table 3.1, with a
    spontaneous emission factor that gives n_sp of about 2 at threshold.

    Parameters
    ----------
    wavelength : number
        Vacuum wavelength (m).
    length, width, thickness : numbers
        Dimensions of the active region (m).
    confinement : number
        Confinement factor Gamma.
    v_g : number
        Group velocity (m/s).
    alpha_m, alpha_i : numbers
        Mirror (or grating) and internal loss (1/m).
    a : number
        Differential gain (m²).
    n_tr : number
        Transparency carrier density (1/m³).
    eps : number
        Gain compression factor (m³).
    A, B, C : numbers
        Recombination coefficients (1/s, m³/s, m⁶/s).
    beta_sp : number
        Fraction of the radiative recombination into the mode.
    alpha : number
        Linewidth enhancement factor.
    front_fraction : number
        Fraction of the outcoupled power leaving through the front facet.
    """

    wavelength: float = 1.55E-6
    length: float = 250E-6
    width: float = 2E-6
    thickness: float = 0.2E-6
    confinement: float = 0.3
    v_g: float = 7.5E7
    alpha_m: float = 4500.0
    alpha_i: float = 4000.0
    a: float = 2.5E-20
    n_tr: float = 1E24
    eps: float = 2.5E-23
    A: float = 1E8
    B: float = 1E-16
    C: float = 3E-41
    beta_sp: float = 3E-5
    alpha: float = 5.0
    front_fraction: float = 0.5

    # --- derived quantities ---
    @property
    def volume(self):
        """Volume of the active region (m³)."""
        return self.length*self.width*self.thickness

    @property
    def tau_p(self):
        """Photon lifetime 1/(v_g (alpha_i + alpha_m)) (s)."""
        return 1/(self.v_g*(self.alpha_i + self.alpha_m))

    @property
    def n_th(self):
        """Threshold carrier density (1/m³), without gain compression."""
        return self.n_tr + 1/(self.confinement*self.v_g*self.a*self.tau_p)

    @property
    def photon_energy(self):
        """h nu (J)."""
        return H*C0/self.wavelength

    def recombination(self, n):
        """Recombination rate density A n + B n**2 + C n**3 (1/(m³ s))."""
        return self.A*n + self.B*n**2 + self.C*n**3

    def carrier_lifetime(self, n=None):
        """Carrier lifetime n/R(n) (s), at threshold by default."""
        n = self.n_th if n is None else n
        return n/self.recombination(n)

    @property
    def i_th(self):
        """Threshold current q V R(n_th) (A), neglecting the spontaneous
        emission into the mode."""
        return Q_E*self.volume*self.recombination(self.n_th)

    def gain_rate(self, N, S):
        """Modal gain G (1/s) at the carrier and photon numbers N and S."""
        n = N/self.volume
        s = S*self.confinement/self.volume
        return self.confinement*self.v_g*self.a*(n - self.n_tr)/(1 + self.eps*s)

    def r_sp(self, N):
        """Spontaneous emission into the mode (photons/s), beta_sp B n**2 V."""
        n = N/self.volume
        return self.beta_sp*self.B*n**2*self.volume

    def power(self, S):
        """Front facet power (W) h nu v_g alpha_m f S."""
        return self.photon_energy*self.v_g*self.alpha_m*self.front_fraction*np.asarray(S)

    def photons(self, P):
        """Photon number in the mode for the front facet power P (W)."""
        return np.asarray(P)/(self.photon_energy*self.v_g*self.alpha_m*self.front_fraction)

    def slope_efficiency(self):
        """Front facet slope efficiency dP/dI (W/A) well above threshold,
        f (h nu/q) alpha_m/(alpha_m + alpha_i) (internal efficiency 1)."""
        return self.front_fraction*self.photon_energy/Q_E*self.alpha_m/(self.alpha_m + self.alpha_i)

    def rhs(self, N, S, current):
        """Time derivatives (dN/dt, dS/dt) of the rate equations."""
        G = self.gain_rate(N, S)
        dN = current/Q_E - self.volume*self.recombination(N/self.volume) - G*S
        dS = (G - 1/self.tau_p)*S + self.r_sp(N)
        return dN, dS

    @property
    def kappa(self):
        """Adiabatic chirp coefficient (1/(W s)) of the Koch-Linke formula
        delta nu = (alpha/(4 pi)) (d ln P/dt + kappa P) for the front facet
        power P: kappa = eps Gamma/(V tau_p h nu v_g alpha_m f), from the gain
        compression."""
        return self.eps*self.confinement/(self.volume*self.tau_p*self.photon_energy*self.v_g*self.alpha_m
                                          *self.front_fraction)

    def chirp(self, N):
        """Instantaneous frequency shift (Hz) (alpha/(4 pi)) Gamma v_g a (n - n_th)
        of the carrier number N (-d phi/dt/(2 pi) of the envelope)."""
        return self.alpha/(4*np.pi)*self.confinement*self.v_g*self.a*(np.asarray(N)/self.volume - self.n_th)

    # --- steady state ---
    def steady_state(self, current):
        """Steady state carrier and photon numbers (N, S) at the current (A).

        Solves dS/dt = 0 for S(N) and then dN/dt = 0 for N by bisection.
        """

        def s_of(N):
            # (G(N, S) - 1/tau_p) S + R_sp = 0, with G compressed by S: solve the
            # quadratic in S of (g_l - (1 + e S)/tau_p) S + R_sp (1 + e S) = 0
            n = N/self.volume
            g_l = self.confinement*self.v_g*self.a*(n - self.n_tr)
            e = self.eps*self.confinement/self.volume
            rsp = self.r_sp(N)
            # -e/tau_p S² + (g_l - 1/tau_p + e rsp) S + rsp = 0
            a2, a1, a0 = -e/self.tau_p, g_l - 1/self.tau_p + e*rsp, rsp
            disc = a1**2 - 4*a2*a0
            return (-a1 - np.sqrt(disc))/(2*a2)

        def f(N):
            S = s_of(N)
            return self.rhs(N, S, current)[0]

        if current <= 0:
            return 0.0, 0.0
        N_lo = 0.0
        N_hi = 3*self.n_th*self.volume
        N = brentq(f, N_lo, N_hi, xtol=1E-12*self.n_th*self.volume, rtol=1E-13)
        return N, s_of(N)

    def li_curve(self, currents):
        """Front facet power (W) and carrier density (1/m³) against current (A)."""
        out = [self.steady_state(i) for i in np.atleast_1d(currents)]
        N, S = np.array(out).T
        return self.power(S), N/self.volume

    # --- small signal: modulation response, RIN, linewidth ---
    def jacobian(self, current):
        """Jacobian d(dN/dt, dS/dt)/d(N, S) at the steady state of the current."""
        N, S = self.steady_state(current)
        hN, hS = 1E-6*N, 1E-6*max(S, 1.0)
        J = np.empty((2, 2))
        for j, (dN, dS) in enumerate(((hN, 0), (0, hS))):
            fp = self.rhs(N + dN, S + dS, current)
            fm = self.rhs(N - dN, S - dS, current)
            J[0, j] = (fp[0] - fm[0])/(2*(dN + dS))
            J[1, j] = (fp[1] - fm[1])/(2*(dN + dS))
        return J, N, S

    def relaxation(self, current):
        """Relaxation oscillation frequency f_R (Hz) and damping rate gamma
        (1/s) at the current: the eigenvalues of the Jacobian are
        -gamma/2 +- i 2 pi f_R."""
        J = self.jacobian(current)[0]
        lam = np.linalg.eigvals(J)
        return np.abs(lam.imag).max()/(2*np.pi), -2*lam.real.mean()

    def modulation_response(self, f, current, normalize=True):
        """Small-signal intensity modulation response delta S/delta I against
        the modulation frequency f (Hz), from the linearized rate equations
        (complex, 1 at f = 0 if normalized)."""
        J = self.jacobian(current)[0]
        f = np.atleast_1d(np.asarray(f, dtype=float))
        b = np.array([1/Q_E, 0.0])
        Hs = np.array([np.linalg.solve(-2j*np.pi*ff*np.eye(2) - J, b)[1] for ff in f])
        if normalize:
            Hs = Hs/np.linalg.solve(-J, b)[1]
        return Hs

    def bandwidth_3db(self, current):
        """Frequency (Hz) where abs(modulation_response())**2 falls to 1/2."""
        f_R, _ = self.relaxation(current)
        f = np.linspace(1E6, 10*max(f_R, 1E8), 4001)
        h = np.abs(self.modulation_response(f, current))**2
        i = np.nonzero(h >= 0.5)[0][-1]
        return np.interp(0.5, [h[i + 1], h[i]], [f[i + 1], f[i]])

    def diffusion(self, N, S):
        """Langevin diffusion coefficients (D_NN, D_SS, D_NS, D_phiphi): the
        correlations <F_i(t) F_j(t')> = D_ij delta(t - t') of the noise of the
        carrier number, photon number and phase (Agrawal sec. 3.5.4 with the
        carrier shot noise of recombination and stimulated emission)."""
        rsp = self.r_sp(N)
        rec = self.volume*self.recombination(N/self.volume)
        D_SS = 2*rsp*S
        D_NN = 2*(rsp*S + rec)
        D_NS = -2*rsp*S
        D_pp = rsp/(2*S)
        return D_NN, D_SS, D_NS, D_pp

    def rin(self, f, current, shot_noise=False):
        """One-sided relative intensity noise (1/Hz) of the photon number at the
        frequencies f (Hz), from the linearized rate equations driven by
        the Langevin forces. With shot_noise, the shot noise floor 2 h nu/P
        of the output is added. 10 log10() gives dB/Hz."""
        J, N, S = self.jacobian(current)
        D_NN, D_SS, D_NS, _ = self.diffusion(N, S)
        D = np.array([[D_NN, D_NS], [D_NS, D_SS]])
        out = []
        for ff in np.atleast_1d(np.asarray(f, dtype=float)):
            Hm = np.linalg.inv(-2j*np.pi*ff*np.eye(2) - J)
            out.append(np.real(Hm @ D @ Hm.conj().T)[1, 1])
        rin = 2*np.array(out)/S**2
        if shot_noise:
            rin = rin + 2*self.photon_energy/self.power(S)
        return rin

    def linewidth(self, current):
        """Linewidth (FWHM, Hz) R_sp (1 + alpha**2)/(4 pi S) at the current
        (Henry)."""
        N, S = self.steady_state(current)
        return self.r_sp(N)*(1 + self.alpha**2)/(4*np.pi*S)

    # --- time domain ---
    def simulate(self, t, current, N0=None, S0=None, noise=False, rng=None, realizations=None):
        """Integrate the rate equations with a fixed step RK4 (and Langevin
        noise, added per step, if requested).

        Parameters
        ----------
        t : array
            Equally spaced times (s); a step of 0.1 to 0.5 ps is fine.
        current : number, array or callable
            Drive current (A), one value per time or a function of t.
        N0, S0 : numbers, optional
            Initial state (default the steady state at the first current).
        noise : bool, optional
            Add the Langevin forces of diffusion().
        rng : seed or Generator, optional
        realizations : int, optional
            Number of independent noisy runs integrated at once (vectorized).

        Returns
        -------
        N, S, phi : arrays
            Carrier and photon numbers and the envelope phase (rad), shaped
            (len(t),) or (realizations, len(t)).
        """

        t = np.asarray(t, dtype=float)
        dt = t[1] - t[0]
        if callable(current):
            I = np.asarray(current(t), dtype=float)
        else:
            I = np.broadcast_to(np.asarray(current, dtype=float), t.shape).astype(float)
        if N0 is None or S0 is None:
            N0s, S0s = self.steady_state(I[0])
            N0 = N0s if N0 is None else N0
            S0 = S0s if S0 is None else S0
        shape = () if realizations is None else (realizations,)
        N = np.full(shape, float(N0))
        S = np.full(shape, float(S0))
        phi = np.zeros(shape)
        rng = np.random.default_rng(rng)
        out_N = np.empty((len(t),) + shape)
        out_S = np.empty_like(out_N)
        out_p = np.empty_like(out_N)
        k_phi = -self.alpha/2*self.confinement*self.v_g*self.a/self.volume
        n_th_N = self.n_th*self.volume
        for k in range(len(t)):
            out_N[k], out_S[k], out_p[k] = N, S, phi
            if k == len(t) - 1:
                break
            Ik, Ik1 = I[k], I[k + 1]
            Im = 0.5*(Ik + Ik1)
            k1 = self.rhs(N, S, Ik)
            k2 = self.rhs(N + 0.5*dt*k1[0], S + 0.5*dt*k1[1], Im)
            k3 = self.rhs(N + 0.5*dt*k2[0], S + 0.5*dt*k2[1], Im)
            k4 = self.rhs(N + dt*k3[0], S + dt*k3[1], Ik1)
            Nn = N + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            Sn = S + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            phi = phi + dt*k_phi*(0.5*(N + Nn) - n_th_N)
            if noise:
                D_NN, D_SS, D_NS, D_pp = self.diffusion(N, np.maximum(S, 1.0))
                z1 = rng.standard_normal(shape)
                z2 = rng.standard_normal(shape)
                z3 = rng.standard_normal(shape)
                # correlated (F_N, F_S) by a 2x2 Cholesky factor
                l11 = np.sqrt(D_NN)
                l21 = D_NS/l11
                l22 = np.sqrt(np.maximum(D_SS - l21**2, 0.0))
                Nn = Nn + np.sqrt(dt)*l11*z1
                Sn = Sn + np.sqrt(dt)*(l21*z1 + l22*z2)
                phi = phi + np.sqrt(dt*D_pp)*z3
            N, S = Nn, np.maximum(Sn, 0.0)
        tr = (lambda x: x) if realizations is None else (lambda x: np.moveaxis(x, 0, -1))
        return tr(out_N), tr(out_S), tr(out_p)

    def nrz(self, bits, bit_rate, i_bias, i_mod, samples_per_bit=32, rise_time=None,
            dt_max=0.25E-12, noise=False, rng=None):
        """Field of a directly modulated laser driven by NRZ bits, in the units
        and conventions of `pyoptics.fibrecomm.nrz()`.

        The current switches between i_bias (zeros) and i_bias + i_mod (ones)
        with Gaussian edges of the given 10-90 % rise time; the rate equations
        run over the pattern twice (periodic signal) and the second pass is
        returned.

        Parameters
        ----------
        bits : array of 0 and 1
        bit_rate : number
            Gb/s.
        i_bias, i_mod : numbers
            Current of a zero and the modulation current (A).
        samples_per_bit : int, optional
        rise_time : number, optional
            10-90 % rise time of the current (ps), default 0.2 bit.
        dt_max : number, optional
            Largest integration step (s).
        noise : bool, optional
            Include the Langevin noise.
        rng : seed or Generator, optional

        Returns
        -------
        t : array
            Time (ps).
        A : complex array
            Field envelope, abs(A)**2 the power (mW), phase with the transient
            and adiabatic chirp of the laser.
        current : array
            Drive current (A).
        """

        from .fibrecomm import _gaussian_filter_periodic

        bits = np.asarray(bits)
        T_bit = 1E3/bit_rate
        n = len(bits)*samples_per_bit
        t = np.arange(n)*T_bit/samples_per_bit
        dt_s = T_bit/samples_per_bit*1E-12
        sub = max(int(np.ceil(dt_s/dt_max)), 1)
        rise_time = 0.2*T_bit if rise_time is None else rise_time
        I = np.repeat(np.where(bits > 0, i_bias + i_mod, i_bias), samples_per_bit).astype(float)
        I = _gaussian_filter_periodic(I, rise_time/2.563/(T_bit/samples_per_bit))
        # fine grid over two periods
        I2 = np.tile(I, 2)
        tf = np.arange(2*n*sub + 1)*dt_s/sub
        If = np.interp(tf, np.arange(2*n + 1)*dt_s, np.append(I2, I2[0]))
        N, S, phi = self.simulate(tf, If, noise=noise, rng=rng)
        sel = slice(n*sub, 2*n*sub, sub)
        P = self.power(S[sel])*1E3
        ph = phi[sel] - phi[sel][0]
        # the mean frequency offset (adiabatic chirp of the average power) is a
        # carrier shift, not removed: the phase stays as integrated
        return t, np.sqrt(np.maximum(P, 0.0))*np.exp(1j*ph), I

    # --- multimode Fabry-Perot spectrum ---
    def mode_spectrum(self, current, n_modes=41, mode_spacing=None, gain_fwhm=8E12,
                      selection=None):
        """Steady state powers of the longitudinal modes of a Fabry-Perot (or
        DFB) laser with a parabolic gain spectrum, all modes sharing the
        carriers (multimode rate equations without gain compression).

        Parameters
        ----------
        current : number
            A.
        n_modes : int, optional
            Number of modes, centred on the gain peak.
        mode_spacing : number, optional
            Hz, default c/(2 n_g L) with n_g = c/v_g.
        gain_fwhm : number, optional
            FWHM (Hz) of the parabolic gain, G_m = G(N) (1 - (2 nu_m/gain_fwhm)**2)
            (zero gain at half the FWHM, a Lorentzian approximated near its peak).
        selection : tuple, optional
            (m, delta): mode index (from the centre) with a loss lower by
            the factor 1 - delta, e.g. a DFB grating; all others have the
            loss of the Fabry-Perot cavity.

        Returns
        -------
        nu : array
            Mode frequency offsets (Hz).
        P : array
            Front facet power of each mode (W).
        """

        if mode_spacing is None:
            mode_spacing = self.v_g/(2*self.length)
        m = np.arange(n_modes) - n_modes//2
        nu = m*mode_spacing
        shape = 1 - (2*nu/gain_fwhm)**2
        inv_tau = np.full(n_modes, 1/self.tau_p)
        if selection is not None:
            idx, delta = selection
            inv_tau[np.nonzero(m == idx)[0]] *= 1 - delta
        G0 = self.confinement*self.v_g*self.a
        # parametrize by the gap delta = 1/tau_best - G_best of the best mode
        # (log scale: near and above threshold delta becomes tiny)
        ratio = inv_tau/np.maximum(shape, 1E-9)
        b = np.argmin(ratio)

        def carriers(u):
            return (self.n_tr + (inv_tau[b] - np.exp(u))/(G0*shape[b]))*self.volume

        def modes(N):
            G = G0*(N/self.volume - self.n_tr)*shape
            return self.r_sp(N)/(inv_tau - G), G

        def f(u):
            N = carriers(u)
            S, G = modes(N)
            S[b] = self.r_sp(N)/np.exp(u)
            return current/Q_E - self.volume*self.recombination(N/self.volume) - np.sum(G*S)

        u_hi = np.log(inv_tau[b] + G0*shape[b]*self.n_tr*(1 - 1E-3))
        u = brentq(f, np.log(1E-14*inv_tau[b]), u_hi, xtol=1E-12)
        N = carriers(u)
        S, _ = modes(N)
        S[b] = self.r_sp(N)/np.exp(u)
        return nu, self.photon_energy*self.v_g*self.alpha_m*self.front_fraction*S
