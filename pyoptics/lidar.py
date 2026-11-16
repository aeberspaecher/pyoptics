#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""LiDAR: link budgets of time-of-flight and FMCW sensors, background
sunlight, detectors, eye safety, coherent detection and beam scanners.

All quantities are in SI units (m, s, W, J, A, Hz) unless a name says
otherwise (``wavelength_nm``, ``..._km``).

Link budget
-----------
A LiDAR sends a pulse (or a chirp) of power P_t and receives the light
scattered back by a target at the range R. For a Lambertian target of
reflectivity rho that is larger than the laser spot, the received power is
independent of the beam divergence, ::

    P_r = P_t * rho*cos(theta)/pi * A_rx/R**2 * T_atm**2 * eta_rx,

(`lambertian_return()`), with the receiver aperture A_rx = pi*D**2/4, the
one-way atmospheric transmission T_atm = exp(-alpha*R) and the receiver
efficiency eta_rx. Fog and haze are modelled with Kim's empirical
extinction against the visibility (`extinction_coefficient()`).

The Sun lights the same target. The receiver sees the part of it inside
its field of view (full angle theta_fov) and the passband Delta_lambda of
its filter, ::

    P_bg = E_sun(lambda)*Delta_lambda * rho/pi * A_rx*Omega_fov * eta_rx,

independent of the range (`background_power()`); `solar_irradiance()` is a
simple model of the direct solar spectrum on the ground.

Detectors
---------
A linear mode avalanche photodiode (APD) with gain M, excess noise factor
F(M) = k_A*M + (1 - k_A)*(2 - 1/M) and an amplifier with the input noise
current density i_n has the peak signal to noise ratio ::

    SNR = M*R_d*P_s / sqrt(2*q*M**2*F*(R_d*(P_s + P_bg) + I_d)*B + i_n**2*B)

(`apd_snr()`). A threshold detector fires on a pulse with the probability
``P_d = erfc((T - SNR)/sqrt(2))/2`` (threshold T in units of the noise) and
on noise alone with Rice's rate of up-crossings, ``B/sqrt(3)*exp(-T**2/2)``
(`detection_probability()`, `false_alarm_rate()`).

A single photon avalanche diode (SPAD) counts photons into time bins over
many pulses. With a dead time longer than the pulse period it registers
at most the first photon of each period, which piles up the histogram
towards early bins under bright background light. Coates' formula gives
the expected histogram for the mean numbers of photons per bin
(`coates_histogram()`) and inverts it (`coates_correction()`).

Eye safety
----------
`ael_class1()` gives the accessible emission limit of class 1 (IEC
60825-1:2014, Table 3) for a point source (C6 = 1, T2 = 10 s) between 700
and 1400 nm (retinal thermal hazard; correction factors C4 and C7) and
between 1500 and 1800 nm (cornea and lens), as an energy for an exposure
of the given duration. The accessible emission is the part of the beam
through the measurement aperture (`measurement_aperture()`,
`aperture_fraction()`). `class1_pulse_energy()` applies the three rules
for pulse trains (single pulse, average power, and the reduction
C5 = N**-0.25 for N pulses within T2 in the retinal range) and returns the
largest accessible pulse energy. This is a teaching model with the
simplifications listed there; it does not replace the standard.

FMCW LiDAR and coherent detection
---------------------------------
A frequency modulated continuous wave (FMCW) LiDAR sweeps the frequency
of a narrow line laser linearly by the bandwidth B in the chirp time T
(slope kappa = B/T) and mixes the echo with the outgoing light (the local
oscillator, LO). An echo delayed by tau = 2R/c and Doppler shifted by
f_D = 2v/lambda (v > 0: approaching) beats at ::

    f_up = kappa*tau - f_D,   f_down = kappa*tau + f_D

for an up and a down chirp (`fmcw_beat_frequencies()`), which separates
range and velocity (`fmcw_range_velocity()`); the range resolution is
c/(2B) (`range_resolution()`). `fmcw_beat_signal()` simulates the beat
with laser phase noise (Lorentzian line of width Delta_nu) and a nonlinear
chirp; the coherent part of the beat power falls as exp(-2*pi*Delta_nu*tau)
(`coherent_fraction()`).

The coherent receiver mixes the signal with a strong LO on a balanced
pair of photodiodes. Its signal is the mean square heterodyne current
2*R_d**2*P_s*P_LO and its noise the shot noise of the LO, thermal noise
and the residual intensity noise of the LO, so that ::

    SNR = 2*R_d**2*P_s*P_LO / (2*q*R_d*(P_LO + P_s)*B + i_n**2*B + RIN terms)

(`coherent_snr()`), which approaches the shot noise limit eta*P_s/(h*nu*B)
for a strong LO: one photoelectron per unit SNR, independent of the
thermal noise. The same receiver serves coherent optical communication.

Beam scanning
-------------
`resolvable_spots()` is the number of resolvable spots theta*D/(a*lambda)
of a scanner of full optical angle theta and aperture D. `galvo_xy()`
traces a beam through two galvanometer mirrors exactly (the scan line bows
because the second mirror sees a moving beam); `polygon_scan()` reflects a
beam on a rotating polygon in 2D; `lissajous()` gives the path of a
two-axis resonant MEMS mirror. `opa_array_factor()` is the far field of
an optical phased array of N emitters with the pitch d (grating lobes at
sin(theta) = sin(theta_0) + m*lambda/d), with phase quantization and
phase errors; `grating_emission_angle()` the wavelength steering of a
grating emitter, and `risley_direction()` the exact direction behind two
rotating wedge prisms.

References
----------
- P. F. McManamon, *LiDAR Technologies and Systems*, SPIE Press (2019).
- IEC 60825-1:2014, Safety of laser products, Part 1.
- P. B. Coates, "The correction for photon 'pile-up' in the measurement of
  radiative lifetimes", J. Phys. E 1, 878 (1968).
- I. I. Kim, B. McArthur and E. Korevaar, "Comparison of laser beam
  propagation at 785 nm and 1550 nm in fog and haze for optical wireless
  communications", Proc. SPIE 4214, 26 (2001).
- S. O. Rice, "Mathematical analysis of random noise", Bell Syst. Tech. J.
  24, 46 (1945).
- P. J. Winzer and W. R. Leeb, "Coherent lidar at low signal powers: basic
  considerations on optical heterodyning", J. Mod. Opt. 45, 1549 (1998).
- C. V. Poulton et al., "Long-range LiDAR and free-space data
  communication with high-performance optical phased arrays", IEEE J. Sel.
  Top. Quantum Electron. 25, 7700108 (2019).
- G. F. Marshall and G. E. Stutz (eds.), *Handbook of Optical and Laser
  Scanning*, CRC Press (2011).
"""

import numpy as np
from scipy.special import erfc

from .radiometry import planck_radiance, SOLAR_CONSTANT, blackbody_radiance
from .detectors import excess_noise


__all__ = ["C", "Q_E", "K_B", "H", "lambertian_return", "extinction_coefficient",
           "solar_irradiance", "background_power", "photons", "excess_noise", "apd_snr",
           "detection_probability", "false_alarm_rate", "coates_histogram", "coates_correction",
           "max_unambiguous_range", "c4", "c7", "ael_class1", "measurement_aperture",
           "aperture_fraction", "class1_pulse_energy", "range_resolution",
           "fmcw_beat_frequencies", "fmcw_range_velocity", "fmcw_beat_signal",
           "coherent_fraction", "coherent_snr", "resolvable_spots", "galvo_xy", "polygon_scan",
           "lissajous", "opa_array_factor", "grating_emission_angle", "risley_direction"]


C = 299792458.0         # m/s
Q_E = 1.602176634e-19   # C
K_B = 1.380649e-23      # J/K
H = 6.62607015e-34      # J s

R_SUN = 6.957e8         # m
AU = 1.495978707e11     # m
T_SUN = 5772.0          # K, effective temperature


# --------------------------------------------------------------------------
# link budget
# --------------------------------------------------------------------------

def lambertian_return(P_t, R, reflectivity, aperture_diameter, incidence=0.0,
                      transmission=1.0, efficiency=1.0, target_fraction=1.0):
    """Power scattered back into the receiver by a Lambertian target.

    Parameters
    ----------
    P_t : number or array
        Transmitted power (or energy: the result is then an energy).
    R : number or array
        Range (m).
    reflectivity : number or array
        Diffuse reflectivity rho of the target.
    aperture_diameter : number
        Diameter of the receiver aperture (m).
    incidence : number, optional
        Angle between the beam and the normal of the target (rad).
    transmission : number or array, optional
        One-way transmission of the atmosphere; it is squared.
    efficiency : number, optional
        Transmission of the receiver optics and filter.
    target_fraction : number or array, optional
        Fraction of the laser spot that falls onto the target (1 for an
        extended target).

    Returns
    -------
    P_r : number or array
        ``P_t*rho*cos(theta)/pi*A_rx/R**2*T**2*eta*f``.
    """

    A = np.pi*aperture_diameter**2/4
    R = np.asarray(R, dtype=float)
    return (np.asarray(P_t, dtype=float)*np.asarray(reflectivity)*np.cos(incidence)/np.pi*A/R**2
            * np.asarray(transmission)**2*efficiency*np.asarray(target_fraction))


def extinction_coefficient(visibility_km, wavelength_nm):
    """Extinction coefficient of haze and fog from the visibility (Kim et
    al. 2001).

    ``alpha = 3.912/V*(lambda/550 nm)**(-q)`` with q = 1.6 (V > 50 km),
    1.3 (6 to 50 km), 0.16 V + 0.34 (1 to 6 km), V - 0.5 (0.5 to 1 km)
    and 0 (fog, V < 0.5 km: no wavelength dependence).

    Parameters
    ----------
    visibility_km : number or array
        Meteorological visibility (km), where the contrast falls to 2 %.
    wavelength_nm : number or array

    Returns
    -------
    alpha : number or array
        Extinction coefficient (1/km); the one-way transmission over the
        range R (km) is exp(-alpha*R).
    """

    V = np.asarray(visibility_km, dtype=float)
    q = np.select([V > 50, V > 6, V > 1, V > 0.5], [1.6, 1.3, 0.16*V + 0.34, V - 0.5], 0.0)
    return 3.912/V*(np.asarray(wavelength_nm, dtype=float)/550.0)**(-q)


def solar_irradiance(wavelength_nm, airmass=1.5, aerosol_optical_depth=0.1):
    """Direct spectral irradiance of the Sun on the ground (simple model).

    A black body of the Sun's effective temperature (5772 K), scaled to the
    solar constant, attenuated by Rayleigh scattering (optical depth
    0.0088*lambda_um**-4.05) and aerosols (aod*(lambda/550 nm)**-1.3)
    along the airmass. Molecular absorption bands (H2O at 940, 1130, 1380
    and 1870 nm, O2 at 760 nm) are not included: real LiDARs at 940 nm
    profit from the water band, and the model overestimates the
    background there.

    Parameters
    ----------
    wavelength_nm : number or array
    airmass : number, optional
        1/cos(zenith angle); 1.5 is the AM1.5 standard.
    aerosol_optical_depth : number, optional
        At 550 nm.

    Returns
    -------
    E : number or array
        W m^-2 nm^-1 on a surface facing the Sun.
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    # black body radiance scaled to the solar constant, times the solid angle of the Sun
    omega_sun = np.pi*(R_SUN/AU)**2
    L = planck_radiance(lam, T_SUN)
    scale = SOLAR_CONSTANT/(blackbody_radiance(T_SUN)*omega_sun)
    E0 = L*omega_sun*scale
    lam_um = lam/1000.0
    tau = 0.0088*lam_um**-4.05 + aerosol_optical_depth*(lam/550.0)**-1.3
    return E0*np.exp(-airmass*tau)


def background_power(irradiance, reflectivity, aperture_diameter, fov, bandwidth_nm,
                     efficiency=1.0, sun_incidence=0.0):
    """Sunlight scattered by a Lambertian target into the receiver.

    Parameters
    ----------
    irradiance : number or array
        Spectral irradiance on a surface facing the Sun (W m^-2 nm^-1).
    reflectivity : number or array
    aperture_diameter : number
        Receiver aperture (m).
    fov : number or array
        Full angle of the (circular) field of view of one detector (rad).
    bandwidth_nm : number or array
        Passband of the optical filter.
    efficiency : number, optional
    sun_incidence : number, optional
        Angle between the Sun and the normal of the target (rad).

    Returns
    -------
    P_bg : number or array
        Background power (W), independent of the range as long as the
        target fills the field of view.
    """

    A = np.pi*aperture_diameter**2/4
    omega = 2*np.pi*(1 - np.cos(np.asarray(fov, dtype=float)/2))
    E = np.asarray(irradiance)*np.cos(sun_incidence)*np.asarray(bandwidth_nm)
    return E*np.asarray(reflectivity)/np.pi*A*omega*efficiency


def photons(energy, wavelength_nm):
    """Number of photons in an energy (J), or photon rate in a power (W).

    Parameters
    ----------
    energy : number or array
    wavelength_nm : number or array

    Returns
    -------
    number or array
    """

    return np.asarray(energy, dtype=float)*np.asarray(wavelength_nm, dtype=float)*1e-9/(H*C)


# --------------------------------------------------------------------------
# detectors (excess_noise() is imported from pyoptics.detectors)
# --------------------------------------------------------------------------

def apd_snr(P_s, bandwidth, responsivity, gain=1.0, k_A=0.02, P_bg=0.0, dark_current=0.0,
            input_noise=2e-12):
    """Peak signal to noise ratio of a pulse on a PIN or linear mode APD
    receiver.

    Parameters
    ----------
    P_s : number or array
        Peak signal power (W).
    bandwidth : number
        Noise bandwidth of the receiver (Hz).
    responsivity : number
        Primary responsivity R_d (A/W) at unit gain.
    gain : number or array, optional
        APD gain M.
    k_A : number, optional
        Ionization ratio.
    P_bg : number or array, optional
        Background power (W).
    dark_current : number, optional
        Primary (multiplied) dark current (A).
    input_noise : number, optional
        Input referred current noise density of the amplifier (A/sqrt(Hz)).

    Returns
    -------
    SNR : number or array
        Peak signal current over the rms noise current.
    """

    M = np.asarray(gain, dtype=float)
    F = excess_noise(M, k_A)
    P_s = np.asarray(P_s, dtype=float)
    I_p = responsivity*(P_s + P_bg) + dark_current
    var = 2*Q_E*M**2*F*I_p*bandwidth + input_noise**2*bandwidth
    return M*responsivity*P_s/np.sqrt(var)


def detection_probability(snr, threshold):
    """Probability that a pulse of peak SNR crosses a threshold in Gaussian
    noise.

    Parameters
    ----------
    snr : number or array
        Peak signal over the rms noise.
    threshold : number or array
        Threshold in units of the rms noise.

    Returns
    -------
    P_d : number or array
        ``erfc((T - SNR)/sqrt(2))/2``.
    """

    return 0.5*erfc((np.asarray(threshold) - np.asarray(snr))/np.sqrt(2))


def false_alarm_rate(threshold, bandwidth):
    """Rate of noise crossings of a threshold (Rice), for Gaussian noise
    with a flat spectrum up to the bandwidth B.

    Parameters
    ----------
    threshold : number or array
        Threshold in units of the rms noise.
    bandwidth : number
        Hz.

    Returns
    -------
    rate : number or array
        Up-crossings per second, ``B/sqrt(3)*exp(-T**2/2)``.
    """

    return bandwidth/np.sqrt(3)*np.exp(-np.asarray(threshold, dtype=float)**2/2)


def coates_histogram(mean_counts):
    """Expected first-photon histogram of a SPAD with a dead time longer
    than the measurement period (Coates 1968).

    Parameters
    ----------
    mean_counts : array
        Mean number of detectable photons (signal and background, times
        the detection efficiency) in each time bin of one period, along
        the last axis.

    Returns
    -------
    p : array
        Probability per period that the (first) detection falls into each
        bin, ``(1 - exp(-n_i))*exp(-sum_{j<i} n_j)``; multiply by the
        number of periods for the expected counts.
    """

    n = np.asarray(mean_counts, dtype=float)
    before = np.cumsum(n, axis=-1) - n
    return -np.expm1(-n)*np.exp(-before)


def coates_correction(counts, n_periods):
    """Mean photon numbers per bin from a piled-up SPAD histogram (the
    inverse of `coates_histogram()`).

    Parameters
    ----------
    counts : array
        Counts per bin, accumulated over n_periods, along the last axis.
    n_periods : int
        Number of laser periods (cycles).

    Returns
    -------
    n : array
        Estimated mean number of photons per bin and period,
        ``-ln(1 - h_i/(N - sum_{j<i} h_j))``.
    """

    h = np.asarray(counts, dtype=float)
    remaining = n_periods - (np.cumsum(h, axis=-1) - h)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(remaining > 0, h/remaining, np.nan)
    return -np.log1p(-np.minimum(ratio, 1.0))


def max_unambiguous_range(prf):
    """Largest range c/(2 PRF) that returns before the next pulse.

    Parameters
    ----------
    prf : number or array
        Pulse repetition frequency (Hz).

    Returns
    -------
    number or array
        Range (m).
    """

    return C/(2*np.asarray(prf, dtype=float))


# --------------------------------------------------------------------------
# eye safety
# --------------------------------------------------------------------------

def c4(wavelength_nm):
    """Correction factor C4 of IEC 60825-1: 10**(0.002*(lambda - 700)) from
    700 to 1050 nm, 5 from 1050 to 1400 nm (1 below 700 nm).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    number or array
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    return np.where(lam < 700, 1.0, np.where(lam < 1050, 10**(0.002*(lam - 700)), 5.0))


def c7(wavelength_nm):
    """Correction factor C7 of IEC 60825-1:2014: 1 from 700 to 1150 nm,
    10**(0.018*(lambda - 1150)) to 1200 nm, 8 + 10**(0.04*(lambda - 1250))
    from 1200 to 1400 nm (water absorption in front of the retina).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    number or array
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    return np.select([lam < 1150, lam < 1200], [1.0, 10**(0.018*(lam - 1150))],
                     8 + 10**(0.04*(lam - 1250)))


def _check_band(lam):
    lam = np.asarray(lam, dtype=float)
    ok = ((lam >= 700) & (lam <= 1400)) | ((lam >= 1500) & (lam <= 1800))
    if not np.all(ok):
        raise ValueError("ael_class1 covers 700 to 1400 nm and 1500 to 1800 nm only")


def ael_class1(wavelength_nm, duration, T2=10.0):
    """Accessible emission limit of class 1 for one exposure (IEC
    60825-1:2014, Table 3, point source C6 = 1).

    Retinal thermal limits from 700 to 1400 nm: ::

        700-1050 nm: 7.7e-8 C4 J         (1e-11 s to 5e-6 s)
                     7e-4 t**0.75 C4 J  (5e-6 s to T2)
                     7e-4 T2**-0.25 C4 W (T2 and longer)
        1050-1400 nm: 7.7e-7 C7 J, 3.5e-3 t**0.75 C7 J (from 1.3e-5 s),
                      3.5e-3 T2**-0.25 C7 W,

    (the 2007 edition allowed 2e-7 C4 J up to 18e-6 s, and 2e-6 C7 J up
    to 5e-5 s, for short pulses),

    and for 1500 to 1800 nm (cornea and lens) 8e-3 J up to 0.35 s,
    1.8e-2 t**0.75 J to 10 s and 1e-2 W beyond.

    Parameters
    ----------
    wavelength_nm : number or array
    duration : number or array
        Exposure duration t (s), at least 1e-11 s.
    T2 : number, optional
        10 s for a point source (alpha <= 1.5 mrad).

    Returns
    -------
    E : number or array
        Energy (J) through the measurement aperture for the exposure t
        (the power limit times t beyond T2).
    """

    _check_band(wavelength_nm)
    lam = np.asarray(wavelength_nm, dtype=float)
    t = np.asarray(duration, dtype=float)
    lam, t = np.broadcast_arrays(lam, t)
    C4 = c4(lam)
    C7 = c7(lam)
    near = lam <= 1050
    E_short = np.where(near, 7.7e-8*C4, 7.7e-7*C7)
    k = np.where(near, 7e-4*C4, 3.5e-3*C7)
    t_break = np.where(near, 5e-6, 1.3e-5)
    E_ret = np.where(t < t_break, E_short,
                     np.where(t < T2, k*t**0.75, k*T2**-0.25*t))
    E_ir = np.where(t < 0.35, 8e-3, np.where(t < 10, 1.8e-2*t**0.75, 1e-2*t))
    E = np.where(lam <= 1400, E_ret, E_ir)
    return E[()] if E.ndim == 0 else E


def measurement_aperture(wavelength_nm, duration):
    """Diameter of the measurement aperture for class 1 (condition 3, at
    the closest point of access): 7 mm from 400 to 1400 nm; for 1400 to
    4000 nm 1 mm up to 0.35 s, 1.5 t**(3/8) mm to 10 s and 3.5 mm beyond.

    Parameters
    ----------
    wavelength_nm : number or array
    duration : number or array
        Exposure duration (s).

    Returns
    -------
    D : number or array
        Diameter (m).
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    t = np.asarray(duration, dtype=float)
    D_ir = np.where(t < 0.35, 1.0, np.where(t < 10, 1.5*np.maximum(t, 1e-12)**0.375, 3.5))
    D = np.where(lam <= 1400, 7.0, D_ir)*1e-3
    return D[()] if D.ndim == 0 else D


def aperture_fraction(beam_diameter, aperture_diameter):
    """Fraction of a centred Gaussian beam through a circular aperture.

    Parameters
    ----------
    beam_diameter : number or array
        1/e**2 diameter of the beam.
    aperture_diameter : number or array

    Returns
    -------
    number or array
        ``1 - exp(-2*a**2/w**2)`` with the radii a and w.
    """

    w = np.asarray(beam_diameter, dtype=float)/2
    a = np.asarray(aperture_diameter, dtype=float)/2
    return -np.expm1(-2*a**2/w**2)


def class1_pulse_energy(wavelength_nm, pulse_duration, prf, time_base=100.0, T2=10.0,
                        n_pulses=None, return_rules=False):
    """Largest accessible pulse energy of a class 1 pulse train.

    The three rules of IEC 60825-1 for repetitive pulses: (1) every single
    pulse below the AEL for its duration; (2) the average power over the
    time base below the AEL for the time base; (3) between 400 and 1400
    nm, every pulse below the single pulse AEL times C5 = N**-0.25, N the
    number of pulses within T2.

    Simplifications: a point source (C6 = 1), no grouping of pulses within
    the time T_i, and C5 for all pulse durations (the 2014 edition relaxes
    C5 for small sources, so this is conservative). For a scanned beam,
    pass the number of pulses that reach the eye within T2 as n_pulses
    and the exposure time as time_base.

    Parameters
    ----------
    wavelength_nm : number
    pulse_duration : number
        s.
    prf : number
        Pulse repetition frequency (Hz).
    time_base : number, optional
        Exposure time for rule 2 (100 s for class 1 at wavelengths above
        400 nm).
    T2 : number, optional
    n_pulses : number, optional
        Pulses within T2 for rule 3. Defaults to prf*min(T2, time_base).
    return_rules : bool, optional
        Also return the limits of the three rules.

    Returns
    -------
    E : number
        Largest pulse energy (J) through the measurement aperture.
    rules : dict
        Only with return_rules: {"single", "average", "train"}.
    """

    E1 = ael_class1(wavelength_nm, pulse_duration, T2)
    E2 = ael_class1(wavelength_nm, time_base, T2)/(prf*time_base)
    if n_pulses is None:
        n_pulses = prf*min(T2, time_base)
    if wavelength_nm <= 1400:
        E3 = E1*max(n_pulses, 1.0)**-0.25
    else:
        E3 = np.inf
    E = min(E1, E2, E3)
    if return_rules:
        return E, {"single": E1, "average": E2, "train": E3}
    return E


# --------------------------------------------------------------------------
# FMCW
# --------------------------------------------------------------------------

def range_resolution(bandwidth):
    """Range resolution c/(2B) of a chirp (or a pulse) of bandwidth B.

    Parameters
    ----------
    bandwidth : number or array
        Hz.

    Returns
    -------
    number or array
        m.
    """

    return C/(2*np.asarray(bandwidth, dtype=float))


def fmcw_beat_frequencies(R, bandwidth, chirp_time, velocity=0.0, wavelength=1550e-9):
    """Beat frequencies of a triangular FMCW chirp.

    Parameters
    ----------
    R : number or array
        Range (m).
    bandwidth, chirp_time : numbers
        Sweep B (Hz) and duration T (s) of each ramp.
    velocity : number or array, optional
        Radial velocity (m/s), positive for an approaching target.
    wavelength : number, optional
        m.

    Returns
    -------
    f_up, f_down : numbers or arrays
        ``kappa*tau - f_D`` and ``kappa*tau + f_D`` (Hz), with
        kappa = B/T, tau = 2R/c and f_D = 2v/lambda. Negative values mean
        that the Doppler shift exceeds the range frequency (a real
        receiver sees their magnitude).
    """

    f_R = bandwidth/chirp_time*2*np.asarray(R, dtype=float)/C
    f_D = 2*np.asarray(velocity, dtype=float)/wavelength
    return f_R - f_D, f_R + f_D


def fmcw_range_velocity(f_up, f_down, bandwidth, chirp_time, wavelength=1550e-9):
    """Range and radial velocity from the up and down chirp beat
    frequencies (the inverse of `fmcw_beat_frequencies()`).

    Parameters
    ----------
    f_up, f_down : numbers or arrays
        Hz (signed).
    bandwidth, chirp_time : numbers
    wavelength : number, optional

    Returns
    -------
    R, v : numbers or arrays
        Range (m) and velocity (m/s, positive approaching).
    """

    kappa = bandwidth/chirp_time
    f_up = np.asarray(f_up, dtype=float)
    f_down = np.asarray(f_down, dtype=float)
    return C*(f_up + f_down)/(4*kappa), wavelength*(f_down - f_up)/4


def fmcw_beat_signal(t, R, bandwidth, chirp_time, velocity=0.0, wavelength=1550e-9,
                     linewidth=0.0, nonlinearity=None, amplitude=1.0, rng=None,
                     complex_output=True, down=False):
    """Beat signal of one FMCW ramp, with laser phase noise and a nonlinear
    chirp.

    The laser's optical phase is ::

        Phi(t) = 2 pi (s*kappa t**2/2 + s*int eps dt) + phi_n(t),

    with s = +1 (up) or -1 (down chirp), eps(t) the deviation of the
    instantaneous frequency from the linear ramp and phi_n a Wiener
    process with the variance 2 pi Delta_nu per second (a Lorentzian line
    of full width Delta_nu). The beat of the LO with the echo delayed by
    tau = 2R/c and shifted by f_D = 2v/lambda is ::

        b(t) = A exp(i (Phi(t) - Phi(t - tau) - 2 pi f_D t)).

    Parameters
    ----------
    t : array
        Equally spaced sample times (s) within the ramp, starting at or
        after tau.
    R, bandwidth, chirp_time : numbers
    velocity, wavelength : numbers, optional
    linewidth : number, optional
        Full width at half maximum of the laser line (Hz).
    nonlinearity : callable, optional
        eps(t), the frequency deviation (Hz) as a function of time.
        Integrated numerically on a grid covering t - tau.
    amplitude : number, optional
    rng : numpy Generator or seed, optional
    complex_output : bool, optional
        Return the complex (I/Q) beat; otherwise its real part.
    down : bool, optional
        A down chirp.

    Returns
    -------
    b : array
    """

    t = np.asarray(t, dtype=float)
    tau = 2*R/C
    kappa = bandwidth/chirp_time*(-1 if down else 1)
    dt = t[1] - t[0]
    # deterministic part
    phase = 2*np.pi*(kappa*tau*t - kappa*tau**2/2) - 2*np.pi*2*velocity/wavelength*t
    if nonlinearity is not None or linewidth > 0:
        n_lag = int(np.ceil(tau/dt)) + 1
        grid = t[0] - n_lag*dt + dt*np.arange(len(t) + n_lag)
        extra = np.zeros(len(grid))
        if nonlinearity is not None:
            eps = np.asarray(nonlinearity(grid), dtype=float)*(-1 if down else 1)
            extra += 2*np.pi*np.concatenate([[0.0], np.cumsum(0.5*(eps[1:] + eps[:-1])*dt)])
        if linewidth > 0:
            rng = np.random.default_rng(rng)
            steps = rng.standard_normal(len(grid) - 1)*np.sqrt(2*np.pi*linewidth*dt)
            extra += np.concatenate([[0.0], np.cumsum(steps)])
        phase = phase + np.interp(t, grid, extra) - np.interp(t - tau, grid, extra)
    b = amplitude*np.exp(1j*phase)
    return b if complex_output else b.real


def coherent_fraction(tau, linewidth):
    """Fraction of the beat power in the coherent peak for a laser with a
    Lorentzian line, ``exp(-2 pi Delta_nu tau)``; the rest spreads over a
    Lorentzian pedestal of width 2 Delta_nu.

    Parameters
    ----------
    tau : number or array
        Delay between LO and echo (s).
    linewidth : number
        FWHM (Hz); the coherence time is 1/(pi Delta_nu).

    Returns
    -------
    number or array
    """

    return np.exp(-2*np.pi*linewidth*np.asarray(tau, dtype=float))


def coherent_snr(P_s, P_lo, bandwidth, responsivity, input_noise=10e-12, rin_db=-150.0,
                 balanced=True, cmrr_db=25.0):
    """Electrical SNR of a heterodyne (coherent) receiver.

    Signal: the mean square beat current ``2 R_d**2 P_s P_LO``. Noise in
    the bandwidth B: shot noise ``2 q R_d (P_LO + P_s) B``, amplifier
    noise ``i_n**2 B`` and the relative intensity noise of the LO,
    ``(R_d P_LO)**2 10**(RIN/10) B``, which a balanced receiver suppresses
    by its common mode rejection ratio.

    Parameters
    ----------
    P_s, P_lo : numbers or arrays
        Signal and LO power (W) on the receiver (sum over a balanced pair).
    bandwidth : number
        Hz.
    responsivity : number
        A/W.
    input_noise : number, optional
        A/sqrt(Hz).
    rin_db : number, optional
        dB/Hz.
    balanced : bool, optional
    cmrr_db : number, optional
        Common mode rejection of the balanced pair (power, dB).

    Returns
    -------
    SNR : number or array
        Signal power over noise power. For a strong LO it tends to
        R_d P_s/(q B) = eta P_s/(h nu B).
    """

    P_s = np.asarray(P_s, dtype=float)
    P_lo = np.asarray(P_lo, dtype=float)
    sig = 2*responsivity**2*P_s*P_lo
    shot = 2*Q_E*responsivity*(P_lo + P_s)*bandwidth
    rin = (responsivity*P_lo)**2*10**(rin_db/10)*bandwidth
    if balanced:
        rin = rin*10**(-cmrr_db/10)
    return sig/(shot + input_noise**2*bandwidth + rin)


# --------------------------------------------------------------------------
# beam scanning
# --------------------------------------------------------------------------

def resolvable_spots(scan_angle, aperture, wavelength, factor=1.0):
    """Number of resolvable spots ``theta*D/(a*lambda)`` of a scanner.

    Parameters
    ----------
    scan_angle : number or array
        Full optical scan angle (rad).
    aperture : number or array
        Beam (mirror) diameter.
    wavelength : number
        Same unit as the aperture.
    factor : number, optional
        a: 1 for a uniform beam (Rayleigh), 1.27 for a Gaussian beam of
        1/e**2 diameter D (4/pi), more with truncation.

    Returns
    -------
    number or array
    """

    return np.asarray(scan_angle)*np.asarray(aperture)/(factor*wavelength)


def _rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K @ K


def _reflect(d, n):
    return d - 2*np.sum(d*n, axis=-1, keepdims=True)*n


def galvo_xy(theta_x, theta_y, separation, distance):
    """Spot position of a two-mirror galvanometer scanner on a flat field.

    The beam comes along +x and hits the first (x) mirror at the origin,
    which turns it into +y; its axis is z. The second (y) mirror at
    (0, separation, 0) turns it into +z; its axis is x. The field is the
    plane z = distance. Mechanical angles theta_x and theta_y rotate the
    mirrors from their 45° rest positions; each deflects the beam by
    twice its angle.

    Parameters
    ----------
    theta_x, theta_y : numbers or arrays
        Mechanical angles (rad), broadcast against each other.
    separation : number
        Distance between the mirrors along the beam.
    distance : number
        Distance of the field plane from the second mirror.

    Returns
    -------
    x, y : arrays
        Spot coordinates in the field plane. Lines of constant theta_y are
        bowed (the pincushion-like distortion of XY scanners): the x
        mirror lies before the y mirror, so the effective lever of x
        changes with theta_y.
    """

    tx, ty = np.broadcast_arrays(np.asarray(theta_x, dtype=float),
                                 np.asarray(theta_y, dtype=float))
    d0 = np.array([1.0, 0.0, 0.0])
    n1_0 = np.array([-1.0, 1.0, 0.0])/np.sqrt(2)
    n2_0 = np.array([0.0, -1.0, 1.0])/np.sqrt(2)
    x_out = np.empty(tx.shape)
    y_out = np.empty(tx.shape)
    for idx in np.ndindex(tx.shape):
        n1 = _rot([0, 0, 1], -tx[idx]) @ n1_0
        d1 = _reflect(d0, n1)
        n2 = _rot([1, 0, 0], ty[idx]) @ n2_0
        p2 = np.array([0.0, separation, 0.0])
        s = np.dot(p2, n2)/np.dot(d1, n2)          # hit the second mirror plane
        h = s*d1
        d2 = _reflect(d1, n2)
        s2 = (distance - h[2])/d2[2]
        q = h + s2*d2
        x_out[idx], y_out[idx] = q[0], q[1] - separation
    return x_out, y_out


def polygon_scan(rotation, n_facets, radius, offset=0.0):
    """Reflect a beam on a rotating regular polygon mirror (2D).

    The polygon (inscribed radius ``radius``) turns about the origin; the
    beam comes in along -x on the line y = offset. Each facet scans the
    beam over 4 pi/N (twice the facet angle) before the next facet takes
    over.

    Parameters
    ----------
    rotation : number or array
        Rotation angle of the polygon (rad).
    n_facets : int
    radius : number
        Distance of the facets from the axis.
    offset : number, optional
        Height of the incoming beam.

    Returns
    -------
    angle : array
        Direction of the reflected beam, measured from +x (rad), NaN where
        the beam hits a corner.
    point : array
        Hit points, shape (..., 2).
    """

    rot = np.asarray(rotation, dtype=float)
    step = 2*np.pi/n_facets
    d = np.array([-1.0, 0.0])
    angle = np.full(rot.shape, np.nan)
    point = np.full(rot.shape + (2,), np.nan)
    half = radius*np.tan(np.pi/n_facets)
    for idx in np.ndindex(rot.shape):
        best = None
        for k in range(n_facets):
            a = rot[idx] + k*step
            n = np.array([np.cos(a), np.sin(a)])
            if np.dot(d, n) >= 0:
                continue
            p0 = np.array([1e6*radius, offset])
            s = (radius - np.dot(p0, n))/np.dot(d, n)
            p = p0 + s*d
            u = np.dot(p, [-n[1], n[0]])
            if abs(u) <= half and (best is None or s < best[0]):
                best = (s, p, n)
        if best is None:
            continue
        s, p, n = best
        r = d - 2*np.dot(d, n)*n
        angle[idx] = np.arctan2(r[1], r[0])
        point[idx] = p
    return angle, point


def lissajous(t, f_x, f_y, amplitude_x=1.0, amplitude_y=1.0, phase=np.pi/2):
    """Path of a two-axis resonant scanner, x = A_x sin(2 pi f_x t),
    y = A_y sin(2 pi f_y t + phase).

    Parameters
    ----------
    t : array
        s.
    f_x, f_y : numbers
        Resonance frequencies (Hz).
    amplitude_x, amplitude_y : numbers, optional
    phase : number, optional

    Returns
    -------
    x, y : arrays
    """

    t = np.asarray(t, dtype=float)
    return (amplitude_x*np.sin(2*np.pi*f_x*t),
            amplitude_y*np.sin(2*np.pi*f_y*t + phase))


def opa_array_factor(sin_theta, n_elements, pitch, wavelength, steer=0.0, phase_bits=None,
                     phase_error=0.0, element_width=None, rng=None):
    """Far field intensity of a 1D optical phased array.

    Element j at x_j = j*d radiates with the phase -2 pi x_j sin(theta_0)/
    lambda (steering to sin(theta_0) = steer), optionally quantized to
    2**bits levels and disturbed by Gaussian phase errors.

    Parameters
    ----------
    sin_theta : array
        Directions (sine of the angle).
    n_elements : int
    pitch : number
        Element spacing d.
    wavelength : number
    steer : number, optional
        sin(theta_0).
    phase_bits : int, optional
        Quantize the phases to 2**bits levels.
    phase_error : number, optional
        rms phase error (rad).
    element_width : number, optional
        Width of a uniform emitter; multiplies the array factor by the
        element factor sinc**2(w sin(theta)/lambda).
    rng : numpy Generator or seed, optional

    Returns
    -------
    I : array
        Intensity, normalized so that an ideal array without element
        factor gives 1 at sin(theta) = steer.
    """

    s = np.asarray(sin_theta, dtype=float)
    x = pitch*np.arange(n_elements)
    phi = -2*np.pi*x*steer/wavelength
    if phase_bits is not None:
        q = 2*np.pi/2**phase_bits
        phi = q*np.round(phi/q)
    if phase_error:
        rng = np.random.default_rng(rng)
        phi = phi + phase_error*rng.standard_normal(n_elements)
    E = np.exp(1j*(2*np.pi/wavelength*np.multiply.outer(s, x) + phi)).sum(axis=-1)/n_elements
    I = np.abs(E)**2
    if element_width is not None:
        I = I*np.sinc(element_width*s/wavelength)**2
    return I


def grating_emission_angle(wavelength, n_eff, period, order=-1, n_clad=1.0):
    """Emission angle of a waveguide grating antenna,
    ``n_clad sin(theta) = n_eff + m lambda/Lambda``.

    Parameters
    ----------
    wavelength : number or array
    n_eff : number, array or callable
        Effective index of the guided mode; a callable is evaluated at the
        wavelength (dispersion).
    period : number
    order : int, optional
    n_clad : number, optional

    Returns
    -------
    theta : number or array
        rad; NaN where no order radiates.
    """

    lam = np.asarray(wavelength, dtype=float)
    n = n_eff(lam) if callable(n_eff) else n_eff
    s = (n + order*lam/period)/n_clad
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(s) <= 1, np.arcsin(np.clip(s, -1, 1)), np.nan)


def _refract(d, nrm, n1, n2):
    """Vector Snell's law; nrm points against the incoming ray's side."""
    c = -np.dot(nrm, d)
    if c < 0:
        nrm, c = -nrm, -c
    r = n1/n2
    k = 1 - r**2*(1 - c**2)
    if k < 0:
        return None
    return r*d + (r*c - np.sqrt(k))*nrm


def risley_direction(phi1, phi2, wedge, n):
    """Direction of a beam behind a Risley prism pair (exact).

    Each prism has a flat face normal to z and a face tilted by the wedge
    angle; its thin end points to the azimuth phi_k (rad). The first is
    passed flat face first, the second tilted face first (the tilted
    faces face each other). The beam comes in along +z.

    Parameters
    ----------
    phi1, phi2 : numbers or arrays
        Rotation angles of the two prisms (broadcast).
    wedge : number
        Wedge angle (rad).
    n : number
        Index of the prisms.

    Returns
    -------
    d : array
        Unit direction vectors, shape (..., 3); NaN for total internal
        reflection. To first order each prism deviates the beam by
        (n - 1) wedge towards its thick end (azimuth phi + pi), and the
        pair by the vector sum: from 0 (opposed) to 2 (n - 1) wedge
        (aligned).
    """

    p1, p2 = np.broadcast_arrays(np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float))
    out = np.full(p1.shape + (3,), np.nan)
    z = np.array([0.0, 0.0, 1.0])
    for idx in np.ndindex(p1.shape):
        d = z.copy()
        ok = True
        for k, phi in enumerate((p1[idx], p2[idx])):
            sgn = 1.0 if k == 0 else -1.0
            tilt = np.array([sgn*np.sin(wedge)*np.cos(phi), sgn*np.sin(wedge)*np.sin(phi),
                             np.cos(wedge)])
            faces = (z, tilt) if k == 0 else (tilt, z)
            for j, nrm in enumerate(faces):
                d = _refract(d, nrm, 1.0 if j == 0 else n, n if j == 0 else 1.0)
                if d is None:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out[idx] = d/np.linalg.norm(d)
    return out
