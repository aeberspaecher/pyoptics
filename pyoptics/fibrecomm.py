#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""A basic intensity modulated fibre link: bit patterns, pulse propagation
through a dispersive fibre, receiver noise, eye diagrams, the bit error rate
and the budgets of a link.

Units as in `pyoptics.fibres`: time in ps, length in km, wavelength in nm
where the dispersion parameter D (ps/(nm km)) is involved, beta_2 in ps²/km,
power in mW (or dBm), currents in mA, bit rates in Gb/s (one bit lasts
1000/B ps).

Signals
-------
prbs() is a pseudo-random bit sequence from a linear feedback shift
register (PRBS7 = x**7 + x**6 + 1 by default, period 127, all patterns of 7
bits except 0000000). nrz() turns bits into the optical field envelope of a
non-return-to-zero on-off keyed laser: power levels P1 and P0 =
P1/extinction, with finite rise time (a Gaussian filter of the given 10-90 %
rise time on the power) and, optionally, the transient chirp of a directly
modulated laser, d phi/dt = (alpha_H/2) d ln P/dt (Koch and Linke), the
frequency excursion that makes such lasers broaden pulses faster at
1550 nm.

Propagation
-----------
The field envelope A(t) in the frame moving with the group velocity obeys
the linear dispersive propagation of Agrawal's convention (fields
exp(i(beta z - omega t)), as everywhere in pyoptics) ::

    A~(z, Omega) = A~(0, Omega) exp(i beta_2 Omega**2 z/2 + i beta_3 Omega**3 z/6 - alpha z/2),

with A~(Omega) = int A(t) exp(i Omega t) dt. propagate() applies it with
FFTs (periodic in time: a bit pattern should be a whole period of the PRBS).
gaussian_pulse() is the chirped Gaussian exp(-(1 + iC) t**2/(2 T0**2)) and
gaussian_width() the analytic rms width after a length z,
T1/T0 = sqrt((1 + C beta_2 z/T0**2)**2 + (beta_2 z/T0**2)**2).

Receiver
--------
The receiver functions here wrap those of `pyoptics.detectors` (SI units)
in the units of this module; responsivity from quantum efficiency,
bandwidth, NEP and photon counting are found there.
A photodiode with responsivity R (A/W) and, for an APD, the gain M and
excess noise factor F(M) = k_A M + (1 - k_A)(2 - 1/M) (McIntyre). The noise
variances in a bandwidth Delta f are ::

    shot:    sigma_s**2 = 2 q M**2 F(M) (R P + I_d) Delta f
    thermal: sigma_T**2 = 4 k_B T F_n Delta f/R_L

receiver_noise() returns both; detect() adds them to a detected waveform.
The intensity noise of the laser adds a third term that grows with the
power, rin_noise(): sigma_RIN**2 = RIN (R P)**2 Delta f for a relative
intensity noise RIN (1/Hz, from `pyoptics.lasers.DiodeLaser.rin()`); it
caps the Q factor at about 1/(2 sqrt(RIN Delta f)) however strong the
signal.
With the means I1, I0 and standard deviations sigma1, sigma0 of the
currents of ones and zeros at the decision time, the Q factor
Q = (I1 - I0)/(sigma1 + sigma0) gives the bit error rate of Gaussian noise
with the optimal threshold, BER = erfc(Q/sqrt(2))/2 (ber()); Q = 6 is
1E-9, Q = 7 about 1E-12. sensitivity() is the average received power
needed for a given Q, ::

    P_rec = (Q/R) (q F(M) M Delta f Q + sigma_T/M),

for an ideal extinction ratio (Agrawal, eq. 4.5.10), and optimum_apd_gain()
the gain that minimizes it.

Eye diagram and budgets
-----------------------
eye_diagram() folds a waveform into traces two bits long; eye_opening()
measures the vertical opening at the sampling time. power_budget() and
rise_time_budget() add up a link (losses in dB, rise times in quadrature,
rise time of a Gaussian channel 0.44/(bandwidth)). dispersion_penalty() is the
Bellcore estimate of the power penalty of chromatic dispersion with a
broad source, -5 log10(1 - (4 B L abs(D) sigma_lambda)**2) dB.

References
----------
G. P. Agrawal, "Fiber-Optic Communication Systems", 4th ed., Wiley
(2010), chapters 2 to 5.

G. Keiser, "Optical Fiber Communications", 5th ed., McGraw-Hill (2015),
chapters 6, 7 and 8.

R. J. McIntyre, "Multiplication noise in uniform avalanche diodes", IEEE
Trans. Electron Devices 13, 164-168 (1966).

T. L. Koch and R. A. Linke, "Effect of nonlinear gain reduction on
semiconductor laser wavelength chirping", Appl. Phys. Lett. 48, 613-615
(1986).
"""

import numpy as np

from .fibres import beta2_from_D
from . import detectors as _det
from .detectors import Q_E, K_B, q_factor, ber, q_from_ber


__all__ = ["Q_E", "K_B", "prbs", "nrz", "gaussian_pulse", "gaussian_width", "beta2_from_D",
           "propagate", "photocurrent", "excess_noise", "receiver_noise", "rin_noise", "detect",
           "noise_bandwidth", "q_factor",
           "ber", "q_from_ber", "sensitivity", "optimum_apd_gain", "eye_diagram", "eye_opening",
           "dbm", "mw", "power_budget", "rise_time_budget", "dispersion_penalty"]


_C_NM_PER_PS = 2.99792458E5


def prbs(order=7, n=None, seed=None):
    """Pseudo-random bit sequence of a maximum length shift register.

    Parameters
    ----------
    order : {7, 9, 11, 15, 23, 31}, optional
        Register length; the sequence repeats after 2**order - 1 bits.
    n : int, optional
        Number of bits (default: one period).
    seed : int, optional
        Nonzero start state (default all ones).

    Returns
    -------
    bits : int array of 0 and 1
    """

    taps = {7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14), 23: (23, 18), 31: (31, 28)}[order]
    if n is None:
        n = 2**order - 1
    state = (seed or (2**order - 1)) & (2**order - 1)
    out = np.empty(n, dtype=int)
    for i in range(n):
        bit = ((state >> (taps[0] - 1)) ^ (state >> (taps[1] - 1))) & 1
        out[i] = state >> (order - 1) & 1
        state = ((state << 1) | bit) & (2**order - 1)
    return out


def _gaussian_filter_periodic(x, sigma_samples):
    """Periodic Gaussian smoothing (sigma in samples) via FFT."""
    if sigma_samples <= 0:
        return x
    f = np.fft.fftfreq(len(x))
    return np.real(np.fft.ifft(np.fft.fft(x)*np.exp(-2*(np.pi*f*sigma_samples)**2)))


def nrz(bits, bit_rate, samples_per_bit=32, p1=1.0, extinction=np.inf, rise_time=None,
        alpha_h=0.0):
    """Field envelope of an NRZ on-off keyed transmitter.

    Parameters
    ----------
    bits : array of 0 and 1
    bit_rate : number
        Gb/s.
    samples_per_bit : int, optional
    p1 : number, optional
        Power of a one (mW).
    extinction : number, optional
        Extinction ratio P1/P0 (linear, inf: P0 = 0).
    rise_time : number, optional
        10-90 % rise time of the power (ps), default 0.3 bit.
    alpha_h : number, optional
        Linewidth enhancement factor of a directly modulated laser
        (transient chirp, typically 3 to 6; 0 for an external modulator).

    Returns
    -------
    t : array
        Time (ps), periodic with the pattern.
    A : complex array
        Field envelope, abs(A)**2 is the power (mW).
    """

    bits = np.asarray(bits)
    T_bit = 1E3/bit_rate
    n = len(bits)*samples_per_bit
    t = np.arange(n)*T_bit/samples_per_bit
    p0 = 0.0 if np.isinf(extinction) else p1/extinction
    P = np.repeat(np.where(bits > 0, p1, p0), samples_per_bit).astype(float)
    if rise_time is None:
        rise_time = 0.3*T_bit
    # Gaussian step response: 10-90 % = 2.563 sigma
    P = np.maximum(_gaussian_filter_periodic(P, rise_time/2.563/(T_bit/samples_per_bit)), 0.0)
    phase = 0.0
    if alpha_h:
        # d phi/dt = (alpha_H/2) d ln P/dt, phase = (alpha_H/2) ln P (exp(-i omega t): an
        # instantaneous frequency omega0 - d phi/dt, blue shifted on the rising edge)
        phase = -alpha_h/2*np.log(np.maximum(P, 1E-6*p1))
    return t, np.sqrt(P)*np.exp(1j*phase)


def gaussian_pulse(t, T0, C=0.0, p0=1.0):
    """Chirped Gaussian field sqrt(p0) exp(-(1 + iC) t**2/(2 T0**2))
    (T0 the 1/e half width of the power; FWHM = 1.665 T0)."""
    return np.sqrt(p0)*np.exp(-(1 + 1j*C)*np.asarray(t)**2/(2*T0**2))


def gaussian_width(z, T0, beta2, C=0.0, beta3=0.0):
    """Half width T1 (1/e of the power, ps) of a chirped Gaussian pulse after
    z km: T0 sqrt((1 + C beta2 z/T0**2)**2 + (beta2 z/T0**2)**2) (for beta3 =
    0; with beta3 the rms based width of Agrawal's eq. 2.4.17 times
    sqrt(2))."""
    z = np.asarray(z, dtype=float)
    a = (1 + C*beta2*z/T0**2)**2 + (beta2*z/T0**2)**2
    b = (1 + C**2)**2/2*(beta3*z/(4*T0**3))**2
    return T0*np.sqrt(a + b)


def propagate(t, A, length_km, beta2=0.0, beta3=0.0, alpha_db=0.0):
    """Linear propagation of a field envelope through a fibre.

    Parameters
    ----------
    t : array
        Uniform time samples (ps), the window is periodic.
    A : complex array
        Field envelope (sqrt(mW)).
    length_km : number
    beta2, beta3 : numbers, optional
        ps²/km and ps³/km.
    alpha_db : number, optional
        Loss in dB/km.

    Returns
    -------
    A : complex array
        Envelope in the frame moving with the group velocity.
    """

    dt = t[1] - t[0]
    W = 2*np.pi*np.fft.fftfreq(len(t), dt)
    H = np.exp(1j*beta2*W**2*length_km/2 + 1j*beta3*W**3*length_km/6)
    gain = 10**(-alpha_db*length_km/20)
    # numpy's ifft has exp(+i W t): the spectrum of exp(i W t) envelopes, see the module docstring
    return gain*np.fft.fft(np.fft.ifft(A)*H)


def photocurrent(P, responsivity=1.0, gain=1.0):
    """Mean photocurrent (mA) M R P for the power P (mW)."""
    return gain*responsivity*np.asarray(P)


def excess_noise(M, k_A=0.7):
    """McIntyre's excess noise factor k_A M + (1 - k_A)(2 - 1/M) (k_A: about
    0.02 for Si, 0.5 to 0.7 for InGaAs APDs), `pyoptics.detectors.excess_noise()`."""
    return _det.excess_noise(M, k_A)


def receiver_noise(P, bandwidth_ghz, responsivity=1.0, gain=1.0, k_A=0.7, dark_na=0.0,
                   load_ohm=50.0, noise_figure_db=3.0, temperature=300.0):
    """Shot and thermal noise standard deviations (mA) of a PIN or APD
    receiver, `pyoptics.detectors.noise_currents()` in the units of this
    module.

    Parameters
    ----------
    P : number or array
        Optical power (mW).
    bandwidth_ghz : number
        Electrical noise bandwidth (GHz).
    responsivity : number, optional
        A/W.
    gain : number, optional
        APD gain M (1: PIN, no excess noise).
    k_A : number, optional
        Ionization ratio of the APD.
    dark_na : number, optional
        Dark current (nA, primary).
    load_ohm, noise_figure_db, temperature : numbers, optional
        Load resistor, amplifier noise figure and temperature.

    Returns
    -------
    sigma_shot, sigma_thermal : arrays (mA)
    """

    s_shot, s_th = _det.noise_currents(np.asarray(P, dtype=float)*1E-3, bandwidth_ghz*1E9,
                                       responsivity, gain, k_A, dark_na*1E-9, load_ohm,
                                       noise_figure_db, temperature)
    return s_shot*1E3, s_th*1E3


def rin_noise(P, bandwidth_ghz, rin_db_hz, responsivity=1.0, gain=1.0):
    """Standard deviation (mA) of the photocurrent from the relative intensity
    noise of the transmitter, M R P sqrt(RIN Delta f), with RIN in dB/Hz
    (flat over the noise bandwidth Delta f), `pyoptics.detectors.rin_noise()`."""
    return _det.rin_noise(P, bandwidth_ghz*1E9, rin_db_hz, responsivity, gain)


def detect(t, A, bandwidth_ghz, responsivity=1.0, gain=1.0, k_A=0.7, load_ohm=50.0,
           noise_figure_db=3.0, rng=None, noise=True, rin_db_hz=None):
    """Photocurrent of a field envelope behind a receiver with a Gaussian
    electrical filter of the given 3 dB bandwidth, with signal dependent
    Gaussian shot noise and thermal noise, and optionally the intensity
    noise of the transmitter (rin_db_hz, see rin_noise()).

    Returns
    -------
    I : array
        Current (mA).
    """

    P = np.abs(A)**2
    dt = t[1] - t[0]
    # Gaussian filter: |H(f)|**2 = exp(-f**2/(2 s_f**2)), 3 dB at f_3dB = s_f sqrt(2 ln 2)
    s_f = bandwidth_ghz*1E-3/np.sqrt(2*np.log(2))   # 1/ps
    I = photocurrent(P, responsivity, gain)
    Iw = np.real(np.fft.ifft(np.fft.fft(I)*np.exp(-0.25*(np.fft.fftfreq(len(t), dt)/s_f)**2)))
    if not noise:
        return Iw
    rng = np.random.default_rng(rng)
    # noise bandwidth of the Gaussian filter: int_0^inf |H|**2 df = sqrt(pi/2) s_f = 1.064 f_3dB
    nbw = noise_bandwidth(bandwidth_ghz)
    s_shot, s_th = receiver_noise(np.maximum(Iw, 0)/(responsivity*gain), nbw, responsivity,
                                  gain, k_A, 0.0, load_ohm, noise_figure_db)
    var = s_shot**2 + s_th**2
    if rin_db_hz is not None:
        var = var + rin_noise(np.maximum(Iw, 0)/(responsivity*gain), nbw, rin_db_hz, responsivity, gain)**2
    return Iw + rng.standard_normal(len(t))*np.sqrt(var)


def noise_bandwidth(bandwidth_ghz):
    """Noise bandwidth (GHz) of the Gaussian receiver filter of detect() with
    the 3 dB bandwidth bandwidth_ghz: sqrt(pi/(4 ln 2)) f_3dB = 1.064 f_3dB."""
    return np.sqrt(np.pi/(4*np.log(2)))*bandwidth_ghz


def sensitivity(Q, bandwidth_ghz, responsivity=1.0, gain=1.0, k_A=0.7, load_ohm=50.0,
                noise_figure_db=3.0, temperature=300.0):
    """Average received power (mW) for the Q factor Q (ideal extinction,
    no dark current), Agrawal's eq. 4.5.10, `pyoptics.detectors.sensitivity()`."""
    return _det.sensitivity(Q, bandwidth_ghz*1E9, responsivity, gain, k_A, load_ohm,
                            noise_figure_db, temperature)*1E3


def optimum_apd_gain(Q, bandwidth_ghz, responsivity=1.0, k_A=0.7, **kwargs):
    """APD gain with the lowest sensitivity (numerical minimum over M in
    [1, 1000]); returns (M_opt, P_rec in mW)."""
    M, P = _det.optimum_apd_gain(Q, bandwidth_ghz*1E9, responsivity, k_A, **kwargs)
    return M, P*1E3


def eye_diagram(t, signal, bit_rate, offset=0.0):
    """Fold a periodic waveform into traces of two bit periods.

    Parameters
    ----------
    t : array (ps)
    signal : array
    bit_rate : number (Gb/s)
    offset : number, optional
        Time (ps) shifted to the left edge of the eye (e.g. a delay).

    Returns
    -------
    tau : array
        Time within the trace (ps), from 0 to 2 bit periods.
    traces : 2d array (n_traces, len(tau))
    """

    T_bit = 1E3/bit_rate
    dt = t[1] - t[0]
    spb = int(round(T_bit/dt))
    shift = int(round(offset/dt))
    s = np.roll(np.asarray(signal), -shift)
    n_bits = len(s)//spb
    s = np.concatenate([s, s[:spb]])
    traces = np.array([s[i*spb:(i + 2)*spb + 1] for i in range(n_bits - 1)])
    return np.arange(2*spb + 1)*dt, traces


def eye_opening(traces, bits=None, sample=None, threshold=None):
    """Vertical eye opening at the sampling index: the lowest one minus the
    highest zero (negative if the eye is closed).

    Parameters
    ----------
    traces : 2d array
        From eye_diagram().
    bits : array, optional
        The transmitted bit of each trace at the sampling time. Without it
        the traces are split at threshold (default: mean).
    sample : int, optional
        Column of the decision (default: the centre).
    """

    if sample is None:
        sample = traces.shape[1]//2
    v = traces[:, sample]
    if bits is None:
        thr = v.mean() if threshold is None else threshold
        ones, zeros = v[v > thr], v[v <= thr]
    else:
        bits = np.asarray(bits)[:len(v)].astype(bool)
        ones, zeros = v[bits], v[~bits]
    return ones.min() - zeros.max()


def dbm(P_mw):
    """Power in dBm."""
    return 10*np.log10(np.asarray(P_mw))


def mw(P_dbm):
    """Power in mW from dBm."""
    return 10**(np.asarray(P_dbm)/10)


def power_budget(launch_dbm, sensitivity_dbm, length_km, loss_db_km, splices=0, splice_db=0.1,
                 connectors=2, connector_db=0.5, penalties_db=0.0):
    """Power budget of a link.

    Returns
    -------
    dict with the received power, the losses and the margin (dB, dBm).
    """

    fibre = length_km*loss_db_km
    total = fibre + splices*splice_db + connectors*connector_db + penalties_db
    received = launch_dbm - total
    return {"launch_dbm": launch_dbm, "fibre_db": fibre, "splices_db": splices*splice_db,
            "connectors_db": connectors*connector_db, "penalties_db": penalties_db,
            "received_dbm": received, "sensitivity_dbm": sensitivity_dbm,
            "margin_db": received - sensitivity_dbm}


def rise_time_budget(*rise_times):
    """System rise time sqrt(sum t_i**2) (ps) of transmitter, fibre and
    receiver rise times. NRZ needs it below 70 % of a bit period."""
    return np.sqrt(np.sum(np.asarray(rise_times, dtype=float)**2))


def dispersion_penalty(bit_rate, length_km, D, sigma_lambda_nm):
    """Bellcore power penalty (dB) of chromatic dispersion with a source of
    rms spectral width sigma_lambda: -5 log10(1 - (4 B L abs(D) sigma)**2),
    inf beyond B L abs(D) sigma = 1/4. B in Gb/s."""
    x = 4*bit_rate*1E-3*length_km*np.abs(D)*sigma_lambda_nm
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x < 1, -5*np.log10(1 - x**2), np.inf)[()]
