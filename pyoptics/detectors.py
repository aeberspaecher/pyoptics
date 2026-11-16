#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Photodetectors and photon statistics: where the responsivity of a
photodiode comes from, how fast and how quiet it is, photon counting with
single-photon avalanche diodes, and the statistics of the photons
themselves.

SI units throughout (W, A, Hz, m, s), except where a name says otherwise
(`wavelength_nm`, `_ev`), and the specific detectivity D*, which is quoted
in the customary Jones (cm Hz^(1/2)/W).

Responsivity and absorption
---------------------------
A photon of energy h nu = hc/lambda absorbed in the depleted region of a
photodiode yields one electron of charge q with the probability eta (the
quantum efficiency), so the responsivity is ::

    R = eta q lambda/(h c) = eta lambda[nm]/1239.84 A/W.

Photons with an energy below the band gap E_g are not absorbed: the cutoff
wavelength is lambda_c = hc/E_g = 1239.84 nm/E_g[eV] (`cutoff_wavelength()`).
`bandgap()` gives E_g(T) from Varshni's formula for Si and
In0.53Ga0.47As and from Hansen's formula for Hg(1-x)Cd(x)Te, whose cutoff
is set by the composition x. Above the gap, the absorption coefficient
alpha (`absorption_coefficient()`) grows slowly for the indirect gap of Si
(tabulated values of Green at 300 K) and steeply for the direct gaps
(alpha = A sqrt(E - E_g) for InGaAs, the Kane region model of Chu et al.
for HgCdTe, both with an exponential Urbach tail below the gap). The
absorption depth 1/alpha decides how thick the absorber must be; the
internal quantum efficiency of a layer of thickness d below a dead layer
d_0 and a surface of reflectance R_s is ::

    eta = (1 - R_s) exp(-alpha d_0) (1 - exp(-alpha d))

(`quantum_efficiency_layer()`).

Bandwidth
---------
The junction capacitance C = eps A/w (`junction_capacitance()`) and the
load R_L limit the bandwidth to f_RC = 1/(2 pi R_L C), the transit of the
carriers through the depletion width w at the saturation velocity v_s to
f_tr = 0.44 v_s/w. A thin depletion layer is fast to cross but has a large
capacitance and absorbs little; `photodiode_bandwidth()` combines both,
1/f**2 = 1/f_RC**2 + 1/f_tr**2.

Noise, NEP and D*
-----------------
The noise currents in a bandwidth B are ::

    shot:     i_s**2 = 2 q M**2 F(M) (R P + I_d) B
    thermal:  i_T**2 = 4 k_B T F_n B/R_L

with the gain M of an avalanche photodiode and McIntyre's excess noise
factor F(M) = k_A M + (1 - k_A)(2 - 1/M) (`excess_noise()`,
`noise_currents()`). The noise equivalent power is the input power that
gives a signal equal to the noise in 1 Hz, NEP = i_n/R in W/Hz^(1/2)
(`nep()`), and the specific detectivity normalizes it to the detector area,
D* = sqrt(A)/NEP (`specific_detectivity()`). An infrared detector that sees
only the photons of its warm surroundings is background limited (BLIP),
D*_BLIP = (lambda/hc) sqrt(eta/(2 Phi_B)) for a photodiode with the
background photon flux density Phi_B (`blip_detectivity()`), and a cold
shield that narrows its field of view raises it. A cooled photodiode with
a zero-bias resistance-area product R_0 A is Johnson limited at
D* = R sqrt(R_0 A/(4 k_B T)) (`r0a_detectivity()`).
`q_factor()`, `ber()`, `sensitivity()` and `optimum_apd_gain()` turn the
noise into the bit error rate of a digital receiver (the functions of
`pyoptics.fibrecomm` wrap them in its units), and
`quantum_limit_photons()` is the number of photons per bit an ideal photon
counter needs.

Photon counting
---------------
A single-photon avalanche diode (SPAD) biased above breakdown answers to a
photon with a macroscopic avalanche, then needs the dead time tau to
recover. With Poisson light of mean photon number n per gate, the
probability of a click is 1 - exp(-eta n) (`click_probability()`). At the
true rate r the measured rate is r/(1 + r tau) for a non-paralyzable
detector and r exp(-r tau) for a paralyzable one (`dead_time_rate()`,
inverted by `true_rate()`); `spad_detections()` applies the dead time to
simulated arrival times. The dead time regularizes the clicks and so makes
the counts sub-Poissonian (Fano factor 1/(1 + r tau)**2,
`dead_time_fano()`), an artefact that must not be mistaken for
nonclassical light.

Photon statistics
-----------------
Counts n in a time window have the mean <n>, the variance (Delta n)**2,
Mandel's Q = (Delta n)**2/<n> - 1 and the normalized second moment
g2 = <n(n - 1)>/<n>**2 = 1 + Q/<n> (`mandel_q()`, `g2()`). Coherent light
is Poissonian (Q = 0, g2 = 1, `poisson_pmf()`); a single mode of thermal
light is Bose-Einstein distributed (Q = <n>, g2 = 2), M modes follow the
negative binomial distribution (g2 = 1 + 1/M, `thermal_pmf()`); a number
state N (Fock state) has no fluctuations at all (Q = -1, g2 = 1 - 1/N,
`fock_pmf()`). Counting chaotic light in windows of length T with the
coherence time tau_c gives Q = <n> tau_c/T for T >> tau_c
(`mandel_q_chaotic()`, simulated with `chaotic_intensity()` and
`photon_counts()`). Losses of transmission eta thin the counts binomially
(`thin()`): Poisson stays Poisson and Q becomes eta Q, so sub-Poissonian
light needs efficient detectors.

The beam splitter with photons
------------------------------
A lossless beam splitter with transmittance T and reflectance R = 1 - T
turns the input creation operators into
a1+ -> t b1+ + r b2+, a2+ -> r b1+ + t b2+ with t = sqrt(T) and
r = i sqrt(R). `beam_splitter_fock()` returns the output photon number
distribution for number states with n1 and n2 photons at the inputs. With vacuum at the
second input every state is split binomially, so the cross correlation of
the outputs, <n3 n4>/(<n3><n4>), is the g2 of the input (Hanbury Brown and
Twiss): 1 for a laser, 2 for thermal light, 0 for a single photon
(`cross_g2()`). Two identical single photons, one at each input of a
50:50 beam splitter, always leave together, because the amplitudes of both
transmitted and both reflected cancel (t**2 + r**2 = 0): the Hong-Ou-Mandel
effect. Delaying one photon by tau makes them distinguishable; for
photons with Gaussian spectra of rms angular width sigma_omega the
coincidence rate relative to distinguishable photons is ::

    C(tau)/C(inf) = 1 - 2 T R V exp(-sigma_omega**2 tau**2)/(T**2 + R**2),

with the overlap V of all other degrees of freedom (`hom_coincidence()`).
Two classical fields of random relative phase reach at most a dip of 1/2.

References
----------
B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019), chapters 13 and 19.

A. Rogalski, "Infrared and Terahertz Detectors", 3rd ed., CRC Press (2019).

M. A. Green, "Self-consistent optical parameters of intrinsic silicon at
300 K including temperature coefficients", Sol. Energy Mater. Sol. Cells
92, 1305-1310 (2008).

G. L. Hansen, J. L. Schmit and T. N. Casselman, "Energy gap versus alloy
composition and temperature in Hg(1-x)Cd(x)Te", J. Appl. Phys. 53,
7099-7101 (1982).

J. Chu, B. Li, K. Liu and D. Tang, "Empirical rule of intrinsic absorption
spectroscopy in Hg(1-x)Cd(x)Te", J. Appl. Phys. 75, 1234-1235 (1994).

Y. P. Varshni, "Temperature dependence of the energy gap in
semiconductors", Physica 34, 149-154 (1967).

R. Loudon, "The Quantum Theory of Light", 3rd ed., Oxford University Press
(2000), chapters 5 and 6.

C. K. Hong, Z. Y. Ou and L. Mandel, "Measurement of subpicosecond time
intervals between two photons by interference", Phys. Rev. Lett. 59,
2044-2046 (1987).

R. J. McIntyre, "Multiplication noise in uniform avalanche diodes", IEEE
Trans. Electron Devices 13, 164-168 (1966).
"""

import numpy as np
from scipy.special import erf, erfc, erfcinv, gammaln, lambertw
from scipy.integrate import quad


__all__ = ["Q_E", "K_B", "H", "C", "EPS0", "MATERIALS", "photon_energy_ev", "responsivity",
           "quantum_efficiency", "cutoff_wavelength", "bandgap", "hgcdte_bandgap",
           "absorption_coefficient", "absorption_depth", "quantum_efficiency_layer",
           "junction_capacitance", "rc_bandwidth", "transit_bandwidth", "photodiode_bandwidth",
           "excess_noise", "shot_noise", "thermal_noise", "noise_currents", "rin_noise",
           "nep", "specific_detectivity", "nep_from_detectivity", "background_photon_flux",
           "blip_detectivity", "r0a_detectivity",
           "q_factor", "ber", "q_from_ber", "sensitivity", "optimum_apd_gain",
           "quantum_limit_photons",
           "click_probability", "dead_time_rate", "true_rate", "dead_time_fano",
           "poisson_arrivals", "spad_detections",
           "poisson_pmf", "thermal_pmf", "fock_pmf", "pmf_moments", "mandel_q", "g2",
           "thin", "chaotic_intensity", "photon_counts", "mandel_q_chaotic",
           "beam_splitter_fock", "cross_g2", "hom_coincidence", "hom_dip_fwhm"]


Q_E = 1.602176634E-19     # C
K_B = 1.380649E-23        # J/K
H = 6.62607015E-34        # J s
C = 299792458.0           # m/s
EPS0 = 8.8541878128E-12   # F/m
_HC_EV_NM = H*C/Q_E*1E9   # 1239.84 eV nm


# Absorption coefficient of intrinsic silicon at 300 K (1/cm), Green (2008).
_SI_TABLE = np.array([
    (400, 9.52E4), (450, 2.55E4), (500, 1.11E4), (550, 6.39E3), (600, 4.14E3),
    (650, 2.81E3), (700, 1.90E3), (750, 1.30E3), (800, 8.50E2), (850, 5.35E2),
    (900, 3.06E2), (950, 1.57E2), (1000, 6.40E1), (1050, 1.63E1), (1100, 3.50E0),
    (1150, 2.20E-1), (1200, 2.20E-2)])

#: Detector materials: Varshni parameters (E_g(0) in eV, alpha in eV/K,
#: beta in K), relative permittivity, saturation velocity (m/s) and the
#: absorption model ("table" for Si, "direct" with the coefficient A in
#: 1/(cm eV^(1/2)) for InGaAs, "kane" for HgCdTe, whose gap follows Hansen's
#: formula with the composition x instead).
MATERIALS = {
    "Si": {"varshni": (1.170, 4.73E-4, 636.0), "eps_r": 11.7, "v_sat": 1.0E5, "model": "table"},
    "InGaAs": {"varshni": (0.800, 4.5E-4, 327.0), "eps_r": 13.9, "v_sat": 6.0E4,
               "model": "direct", "A": 2.7E4},
    "HgCdTe": {"varshni": None, "eps_r": None, "v_sat": 1.0E5, "model": "kane"},
}


def photon_energy_ev(wavelength_nm):
    """Photon energy hc/lambda (eV) for the vacuum wavelength (nm)."""
    return _HC_EV_NM/np.asarray(wavelength_nm, dtype=float)


def responsivity(wavelength_nm, quantum_efficiency=1.0):
    """Responsivity eta q lambda/(hc) (A/W) of a detector with the quantum
    efficiency eta at the wavelength (nm)."""
    return np.asarray(quantum_efficiency)*np.asarray(wavelength_nm, dtype=float)/_HC_EV_NM


def quantum_efficiency(responsivity, wavelength_nm):
    """Quantum efficiency R hc/(q lambda) of a responsivity R (A/W) at the
    wavelength (nm)."""
    return np.asarray(responsivity)*_HC_EV_NM/np.asarray(wavelength_nm, dtype=float)


def cutoff_wavelength(bandgap_ev):
    """Cutoff wavelength hc/E_g (nm) of the band gap (eV)."""
    return _HC_EV_NM/np.asarray(bandgap_ev, dtype=float)


def hgcdte_bandgap(x, temperature=77.0):
    """Band gap (eV) of Hg(1-x)Cd(x)Te after Hansen, Schmit and Casselman:
    -0.302 + 1.93 x - 0.81 x**2 + 0.832 x**3 + 5.35E-4 T (1 - 2x)."""
    x = np.asarray(x, dtype=float)
    return -0.302 + 1.93*x - 0.81*x**2 + 0.832*x**3 + 5.35E-4*temperature*(1 - 2*x)


def bandgap(material, temperature=300.0, x=None):
    """Band gap (eV) of a detector material.

    Parameters
    ----------
    material : {"Si", "InGaAs", "HgCdTe"}
        In0.53Ga0.47As lattice matched to InP for "InGaAs".
    temperature : number or array, optional
        K.
    x : number, optional
        Cd fraction of HgCdTe (required for it).

    Returns
    -------
    E_g : number or array
        eV; Varshni's E_g(0) - alpha T**2/(T + beta), or Hansen's formula
        for HgCdTe.
    """

    if material == "HgCdTe":
        if x is None:
            raise ValueError("HgCdTe needs the composition x")
        return hgcdte_bandgap(x, temperature)
    E0, a, b = MATERIALS[material]["varshni"]
    T = np.asarray(temperature, dtype=float)
    return E0 - a*T**2/(T + b)


def _urbach(E, Eg, alpha_at_gap, urbach_ev):
    return alpha_at_gap*np.exp((E - Eg)/urbach_ev)


def absorption_coefficient(material, wavelength_nm, temperature=300.0, x=None, urbach_ev=None):
    """Absorption coefficient (1/m) of a detector material.

    Parameters
    ----------
    material : {"Si", "InGaAs", "HgCdTe"}
        Si: Green's table at 300 K (400 to 1200 nm, interpolated
        logarithmically, zero beyond 1200 nm, the temperature is ignored).
        InGaAs: A sqrt(E - E_g), joined smoothly at E_g + E_U/2 to an Urbach
        tail. HgCdTe: Chu's Kane region rule
        alpha_g exp(sqrt(beta (E - E_g))) with an Urbach tail below E_g.
    wavelength_nm : number or array
        Vacuum wavelength (nm).
    temperature : number, optional
        K (InGaAs, HgCdTe).
    x : number, optional
        Cd fraction of HgCdTe.
    urbach_ev : number, optional
        Urbach energy E_U of the tail below the gap (eV), a rough default
        of 0.01 eV for InGaAs and 0.005 eV for HgCdTe.

    Returns
    -------
    alpha : array
        1/m.
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    if material == "Si":
        lt, at = _SI_TABLE[:, 0], np.log(_SI_TABLE[:, 1])
        a = np.exp(np.interp(lam, lt, at, left=np.nan, right=-np.inf))
        a = np.where(lam < lt[0], np.exp(at[0] + (at[1] - at[0])/(lt[1] - lt[0])*(lam - lt[0])), a)
        return a*100
    E = photon_energy_ev(lam)
    Eg = bandgap(material, temperature, x)
    if urbach_ev is None:
        urbach_ev = 0.005 if material == "HgCdTe" else 0.01
    if material == "InGaAs":
        A = MATERIALS["InGaAs"]["A"]
        E1 = Eg + urbach_ev/2
        above = A*np.sqrt(np.maximum(E - Eg, 0))
        tail = A*np.sqrt(urbach_ev/2)*np.exp((E - E1)/urbach_ev)
        return np.where(E >= E1, above, tail)*100
    if material == "HgCdTe":
        T = temperature
        ag = -65 + 1.88*T + (8694 - 10.31*T)*x
        beta = -1 + 0.083*T + (21 - 0.13*T)*x
        above = ag*np.exp(np.sqrt(beta*np.maximum(E - Eg, 0)))
        return np.where(E >= Eg, above, _urbach(E, Eg, ag, urbach_ev))*100
    raise ValueError("unknown material " + repr(material))


def absorption_depth(material, wavelength_nm, temperature=300.0, x=None):
    """Absorption depth 1/alpha (m), see `absorption_coefficient()`."""
    with np.errstate(divide="ignore"):
        return 1/absorption_coefficient(material, wavelength_nm, temperature, x)


def quantum_efficiency_layer(alpha, thickness, reflectance=0.0, dead_layer=0.0):
    """External quantum efficiency (1 - R) exp(-alpha d_0)(1 - exp(-alpha d))
    of an absorbing layer of thickness d (m) below a dead layer d_0 (m),
    for the absorption coefficient alpha (1/m) and the surface reflectance R
    (single pass, every carrier generated in the layer collected)."""
    a = np.asarray(alpha, dtype=float)
    return (1 - np.asarray(reflectance))*np.exp(-a*dead_layer)*-np.expm1(-a*np.asarray(thickness))


def junction_capacitance(area, depletion_width, eps_r=11.7):
    """Capacitance eps_0 eps_r A/w (F) of a depleted junction of area A (m²)
    and width w (m)."""
    return EPS0*eps_r*np.asarray(area)/np.asarray(depletion_width)


def rc_bandwidth(capacitance, load_ohm=50.0):
    """3 dB bandwidth 1/(2 pi R_L C) (Hz)."""
    return 1/(2*np.pi*np.asarray(load_ohm)*np.asarray(capacitance))


def transit_bandwidth(depletion_width, v_sat=1.0E5):
    """3 dB bandwidth 0.44 v_s/w (Hz) of the carrier transit through the
    depletion width w (m) at the saturation velocity v_s (m/s)."""
    return 0.44*v_sat/np.asarray(depletion_width)


def photodiode_bandwidth(area, depletion_width, load_ohm=50.0, eps_r=11.7, v_sat=1.0E5,
                         parasitic=0.0):
    """3 dB bandwidth (Hz) of a PIN photodiode limited by its RC time and the
    transit time, 1/f**2 = 1/f_RC**2 + 1/f_tr**2.

    Parameters
    ----------
    area : number or array
        Junction area (m²).
    depletion_width : number or array
        Depletion (intrinsic layer) width (m).
    load_ohm : number, optional
        Load resistance.
    eps_r : number, optional
        Relative permittivity of the semiconductor.
    v_sat : number, optional
        Saturation velocity of the slower carriers (m/s).
    parasitic : number, optional
        Package and amplifier capacitance added to the junction (F).

    Returns
    -------
    f : array
        Hz.
    """

    f_rc = rc_bandwidth(junction_capacitance(area, depletion_width, eps_r) + parasitic, load_ohm)
    f_tr = transit_bandwidth(depletion_width, v_sat)
    return 1/np.sqrt(1/f_rc**2 + 1/f_tr**2)


def excess_noise(M, k_A):
    """McIntyre's excess noise factor k_A M + (1 - k_A)(2 - 1/M).

    Parameters
    ----------
    M : number or array
        Avalanche gain (1 for a PIN diode).
    k_A : number
        Ratio of the ionization coefficients (about 0.02 for Si, 0.5 to 0.7
        for InGaAs).

    Returns
    -------
    F : number or array
    """

    M = np.asarray(M, dtype=float)
    return k_A*M + (1 - k_A)*(2 - 1/M)


def shot_noise(current, bandwidth, gain=1.0, k_A=0.7):
    """Shot noise current sqrt(2 q M**2 F(M) I B) (A rms) of the primary
    (unmultiplied) current I (A) in the bandwidth B (Hz), with the excess
    noise of an APD of gain M (F = 1 for M = 1)."""
    F = np.where(np.asarray(gain) == 1, 1.0, excess_noise(gain, k_A))
    return np.sqrt(2*Q_E*np.asarray(gain)**2*F*np.asarray(current, dtype=float)*bandwidth)


def thermal_noise(bandwidth, load_ohm=50.0, temperature=300.0, noise_figure_db=0.0):
    """Thermal (Johnson) noise current sqrt(4 k_B T F_n B/R_L) (A rms) of the
    load R_L in the bandwidth B (Hz), with the noise figure F_n of the
    amplifier (dB)."""
    return np.sqrt(4*K_B*temperature*10**(noise_figure_db/10)*np.asarray(bandwidth)/load_ohm)


def noise_currents(P, bandwidth, responsivity=1.0, gain=1.0, k_A=0.7, dark_current=0.0,
                   load_ohm=50.0, noise_figure_db=3.0, temperature=300.0):
    """Shot and thermal noise currents (A rms) of a PIN or APD receiver.

    Parameters
    ----------
    P : number or array
        Optical power (W).
    bandwidth : number
        Electrical noise bandwidth (Hz).
    responsivity : number, optional
        A/W (primary, before avalanche gain).
    gain : number, optional
        APD gain M (1: PIN, no excess noise).
    k_A : number, optional
        Ionization ratio of the APD.
    dark_current : number, optional
        Primary dark current (A).
    load_ohm, noise_figure_db, temperature : numbers, optional
        Load resistor, amplifier noise figure (dB) and temperature (K).

    Returns
    -------
    sigma_shot, sigma_thermal : arrays
        A rms; sigma_thermal has the shape of sigma_shot.
    """

    I = responsivity*np.asarray(P, dtype=float) + dark_current
    s_shot = shot_noise(I, bandwidth, gain, k_A)
    s_th = thermal_noise(bandwidth, load_ohm, temperature, noise_figure_db)
    return s_shot, s_th + 0*s_shot


def rin_noise(P, bandwidth, rin_db_hz, responsivity=1.0, gain=1.0):
    """Current noise M R P sqrt(RIN B) (A rms) from the relative intensity
    noise of the source, RIN in dB/Hz, flat over the bandwidth B (Hz), for
    the power P (W)."""
    rin = 10**(np.asarray(rin_db_hz, dtype=float)/10)
    return gain*responsivity*np.asarray(P, dtype=float)*np.sqrt(rin*bandwidth)


def nep(responsivity, dark_current=0.0, shunt_ohm=np.inf, temperature=300.0, noise_density=0.0):
    """Noise equivalent power (W/Hz^(1/2)) of a photodiode limited by the
    shot noise of its dark current, the Johnson noise of its shunt
    resistance and an amplifier input noise density.

    Parameters
    ----------
    responsivity : number or array
        A/W.
    dark_current : number, optional
        A.
    shunt_ohm : number, optional
        Shunt (zero-bias) resistance (Ohm).
    temperature : number, optional
        K.
    noise_density : number, optional
        Input noise current density of the amplifier (A/Hz^(1/2)).

    Returns
    -------
    NEP : array
        sqrt(2 q I_d + 4 k_B T/R_sh + i_n**2)/R.
    """

    i_n2 = 2*Q_E*dark_current + 4*K_B*temperature/shunt_ohm + noise_density**2
    return np.sqrt(i_n2)/np.asarray(responsivity, dtype=float)


def specific_detectivity(nep, area):
    """Specific detectivity D* = sqrt(A)/NEP in Jones (cm Hz^(1/2)/W), for
    the NEP in W/Hz^(1/2) and the area A in m²."""
    return np.sqrt(np.asarray(area)*1E4)/np.asarray(nep)


def nep_from_detectivity(d_star, area):
    """NEP (W/Hz^(1/2)) of a detector of area A (m²) with D* in Jones."""
    return np.sqrt(np.asarray(area)*1E4)/np.asarray(d_star)


def background_photon_flux(cutoff_nm, temperature=300.0, half_angle=np.pi/2, cuton_nm=None):
    """Photon flux density (photons/(s m²)) of a black body background seen
    through a cone of the given half angle, sin²(half_angle) times the
    hemispherical flux, between cuton_nm (default cutoff_nm/20) and the
    cutoff wavelength (nm)."""
    from .radiometry import planck_radiance
    lo = cutoff_nm/20 if cuton_nm is None else cuton_nm
    L, _ = quad(lambda l: planck_radiance(l, temperature, photons=True), lo, cutoff_nm, limit=200)
    return np.pi*np.sin(half_angle)**2*L


def blip_detectivity(cutoff_nm, temperature=300.0, half_angle=np.pi/2, quantum_efficiency=1.0,
                     photoconductive=False):
    """Background limited specific detectivity (Jones) at the cutoff
    wavelength.

    Parameters
    ----------
    cutoff_nm : number or array
        Cutoff wavelength (nm); the detector responds to all shorter ones.
    temperature : number, optional
        Background temperature (K).
    half_angle : number, optional
        Half angle of the field of view (rad) set by the cold shield.
    quantum_efficiency : number, optional
    photoconductive : bool, optional
        Generation-recombination noise of a photoconductor, sqrt(2) worse
        than a photodiode.

    Returns
    -------
    D_star : array
        (lambda_c/(hc)) sqrt(eta/(2 Phi_B)) in cm Hz^(1/2)/W (Phi_B with
        4 instead of 2 for a photoconductor).
    """

    lc = np.atleast_1d(np.asarray(cutoff_nm, dtype=float))
    phi = np.array([background_photon_flux(l, temperature, half_angle) for l in lc])
    k = 4 if photoconductive else 2
    d = lc*1E-9/(H*C)*np.sqrt(quantum_efficiency/(k*phi))*100
    return d if np.ndim(cutoff_nm) else d[0]


def r0a_detectivity(responsivity, r0a, temperature=77.0):
    """Johnson noise limited D* = R sqrt(R_0 A/(4 k_B T)) (Jones) of a
    photodiode at zero bias, with R in A/W and the resistance-area product
    R_0 A in Ohm cm²."""
    return np.asarray(responsivity)*np.sqrt(np.asarray(r0a)/(4*K_B*temperature))


def q_factor(I1, I0, sigma1, sigma0):
    """Q = (I1 - I0)/(sigma1 + sigma0)."""
    return (np.asarray(I1) - I0)/(np.asarray(sigma1) + sigma0)


def ber(Q):
    """Bit error rate erfc(Q/sqrt(2))/2 of Gaussian noise at the optimal
    threshold."""
    return 0.5*erfc(np.asarray(Q)/np.sqrt(2))


def q_from_ber(ber_value):
    """Q that gives the bit error rate."""
    return np.sqrt(2)*erfcinv(2*np.asarray(ber_value))


def sensitivity(Q, bandwidth, responsivity=1.0, gain=1.0, k_A=0.7, load_ohm=50.0,
                noise_figure_db=3.0, temperature=300.0):
    """Average received power (W) for the Q factor Q with on-off keying
    (ideal extinction, no dark current), (Q/R)(q F M B Q + sigma_T/M),
    Agrawal's eq. 4.5.10, for the noise bandwidth B (Hz)."""
    F = 1.0 if np.all(np.asarray(gain) == 1) else excess_noise(gain, k_A)
    s_T = thermal_noise(bandwidth, load_ohm, temperature, noise_figure_db)
    return np.asarray(Q)/responsivity*(Q_E*F*np.asarray(gain)*bandwidth*Q + s_T/gain)


def optimum_apd_gain(Q, bandwidth, responsivity=1.0, k_A=0.7, **kwargs):
    """APD gain with the lowest `sensitivity()` (numerical minimum over M in
    [1, 1000]); returns (M_opt, P_rec in W)."""
    M = np.logspace(0, 3, 3001)
    P = sensitivity(Q, bandwidth, responsivity, M, k_A, **kwargs)
    i = np.argmin(P)
    return M[i], P[i]


def quantum_limit_photons(ber_value=1E-9):
    """Average number of photons per bit, -ln(2 BER)/2, that an ideal photon
    counter needs for on-off keying with equally likely ones and zeros: an
    error only when a one brings no photon, exp(-N1)/2 = BER (10 photons
    per bit for 1E-9)."""
    return -np.log(2*np.asarray(ber_value))/2


def click_probability(mean_photons, efficiency=1.0, dark=0.0):
    """Probability 1 - exp(-(eta n + n_d)) that a SPAD clicks in a gate with
    Poisson light of mean photon number n, detection efficiency eta and
    the mean number of dark counts n_d per gate."""
    return -np.expm1(-(efficiency*np.asarray(mean_photons, dtype=float) + dark))


def dead_time_rate(rate, dead_time, paralyzable=False):
    """Measured count rate r/(1 + r tau) (non-paralyzable) or r exp(-r tau)
    (paralyzable) of Poisson events at the true rate r with the dead time
    tau."""
    r = np.asarray(rate, dtype=float)
    return r*np.exp(-r*dead_time) if paralyzable else r/(1 + r*dead_time)


def true_rate(measured, dead_time, paralyzable=False):
    """Invert `dead_time_rate()`: m/(1 - m tau), or for a paralyzable
    detector the lower branch -W(-m tau)/tau (Lambert W; m tau <= 1/e)."""
    m = np.asarray(measured, dtype=float)
    if paralyzable:
        return np.real(-lambertw(-m*dead_time, 0))/dead_time
    return m/(1 - m*dead_time)


def dead_time_fano(rate, dead_time):
    """Fano factor 1/(1 + r tau)**2 of the counts of a non-paralyzable
    detector in windows much longer than the dead time, for Poisson events
    at the true rate r."""
    return 1/(1 + np.asarray(rate, dtype=float)*dead_time)**2


def poisson_arrivals(rate, duration, rng=None):
    """Sorted arrival times (s) of a Poisson process of the given rate (1/s)
    in [0, duration)."""
    rng = np.random.default_rng(rng)
    n = rng.poisson(rate*duration)
    return np.sort(rng.uniform(0, duration, n))


def spad_detections(arrivals, dead_time, paralyzable=False):
    """Times at which a detector with the dead time tau registers the sorted
    arrival times.

    Parameters
    ----------
    arrivals : array
        Sorted photon arrival times (s), already thinned by the efficiency.
    dead_time : number
        s.
    paralyzable : bool, optional
        If True, every arrival (detected or not) restarts the dead time;
        otherwise only detections do (active quenching).

    Returns
    -------
    detections : array
    """

    t = np.asarray(arrivals, dtype=float)
    if len(t) == 0:
        return t
    if paralyzable:
        keep = np.concatenate([[True], np.diff(t) > dead_time])
        return t[keep]
    out = []
    ready = -np.inf
    for ti in t:
        if ti >= ready:
            out.append(ti)
            ready = ti + dead_time
    return np.array(out)


def poisson_pmf(n, mean):
    """Poisson distribution exp(-<n>) <n>**n/n! (coherent light)."""
    n = np.asarray(n, dtype=float)
    mean = np.asarray(mean, dtype=float)
    with np.errstate(divide="ignore"):
        return np.exp(n*np.log(mean) - mean - gammaln(n + 1))


def thermal_pmf(n, mean, modes=1):
    """Photon number distribution of thermal light with M modes (negative
    binomial); for M = 1 the Bose-Einstein distribution
    <n>**n/(1 + <n>)**(n + 1). Variance <n> + <n>**2/M."""
    n = np.asarray(n, dtype=float)
    p = mean/(mean + modes)
    return np.exp(gammaln(n + modes) - gammaln(n + 1) - gammaln(modes) + n*np.log(p)
                  + modes*np.log1p(-p))


def fock_pmf(n, N):
    """Photon number distribution of the number state with N photons: 1 at n = N."""
    return (np.asarray(n) == N).astype(float)


def pmf_moments(n, p):
    """Mean, variance, Mandel's Q and g2 of a photon number distribution
    p(n).

    Parameters
    ----------
    n : array
        Photon numbers.
    p : array
        Probabilities (normalized along the last axis).

    Returns
    -------
    mean, variance, Q, g2 : arrays
    """

    n = np.asarray(n, dtype=float)
    m = np.sum(n*p, axis=-1)
    m2 = np.sum(n**2*p, axis=-1)
    var = m2 - m**2
    return m, var, var/m - 1, (m2 - m)/m**2


def mandel_q(counts, axis=None):
    """Mandel's Q = var(n)/<n> - 1 of counts (0 for Poisson, < 0
    sub-Poissonian, > 0 super-Poissonian)."""
    c = np.asarray(counts, dtype=float)
    return np.var(c, axis=axis)/np.mean(c, axis=axis) - 1


def g2(counts, axis=None):
    """Normalized second factorial moment <n(n - 1)>/<n>**2 of counts in a
    window short against the coherence time (g2(0))."""
    c = np.asarray(counts, dtype=float)
    return np.mean(c*(c - 1), axis=axis)/np.mean(c, axis=axis)**2


def thin(counts, efficiency, rng=None):
    """Binomial thinning: the counts that survive a loss or a detector of
    the given efficiency (each photon independently)."""
    rng = np.random.default_rng(rng)
    return rng.binomial(np.asarray(counts, dtype=np.int64), efficiency)


def chaotic_intensity(n, dt, coherence_time, rng=None):
    """Intensity of chaotic (thermal, polarized) light with a Gaussian
    spectrum, mean 1, sampled at n points spaced by dt (periodic).

    The field is a circular complex Gaussian process with
    abs(g1(tau)) = exp(-pi tau**2/(2 tau_c**2)), so that the coherence time
    tau_c = int abs(g1)**2 dtau (Mandel's definition).

    Parameters
    ----------
    n : int
    dt : number
        s.
    coherence_time : number
        s.
    rng : seed or Generator, optional

    Returns
    -------
    I : array
        Exponentially distributed at each time, with <I> = 1.
    """

    rng = np.random.default_rng(rng)
    w = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    f = np.fft.fftfreq(n, dt)
    # kernel exp(-t**2/(2 s**2)) with s = tau_c/sqrt(2 pi): spectrum exp(-2 pi**2 s**2 f**2)
    s = coherence_time/np.sqrt(2*np.pi)
    E = np.fft.ifft(np.fft.fft(w)*np.exp(-2*np.pi**2*s**2*f**2))
    I = np.abs(E)**2
    return I/np.mean(I)


def photon_counts(intensity, dt, rate, window, rng=None):
    """Photon counts in consecutive windows of a fluctuating intensity
    (doubly stochastic Poisson, Mandel's semiclassical photodetection).

    Parameters
    ----------
    intensity : array
        Normalized intensity I(t) with mean 1, sampled every dt.
    dt : number
        s.
    rate : number
        Mean detected photon rate (1/s).
    window : number
        Counting time T (s), a multiple of dt.

    Returns
    -------
    counts : array
        Poisson with the mean rate int_T I dt in each window.
    """

    rng = np.random.default_rng(rng)
    k = max(int(round(window/dt)), 1)
    I = np.asarray(intensity, dtype=float)
    m = len(I)//k
    W = I[:m*k].reshape(m, k).sum(axis=1)*dt*rate
    return rng.poisson(W)


def mandel_q_chaotic(mean_counts, window, coherence_time):
    """Mandel's Q of chaotic light with the Gaussian spectrum of
    `chaotic_intensity()` counted in windows T,
    <n> (1/T**2) int int abs(g1(t - t'))**2 dt dt':
    <n>[(tau_c/T) erf(sqrt(pi) T/tau_c) - tau_c**2/(pi T**2)(1 - exp(-pi T**2/tau_c**2))].
    Tends to <n> (Bose-Einstein) for T << tau_c and to <n> tau_c/T for
    T >> tau_c."""
    r = np.asarray(window, dtype=float)/coherence_time
    return np.asarray(mean_counts)*(erf(np.sqrt(np.pi)*r)/r - (-np.expm1(-np.pi*r**2))/(np.pi*r**2))


def beam_splitter_fock(n1, n2, transmittance=0.5, amplitudes=False):
    """Output of a lossless beam splitter for the number states with n1 and n2 photons at
    its inputs.

    With t = sqrt(T) and r = i sqrt(1 - T), the input creation operators
    become a1+ -> t b1+ + r b2+ and a2+ -> r b1+ + t b2+.

    Parameters
    ----------
    n1, n2 : int
        Photons at inputs 1 and 2.
    transmittance : number, optional
        T.
    amplitudes : bool, optional
        Return the complex amplitudes instead of the probabilities.

    Returns
    -------
    p : array
        p[k]: probability (or amplitude) of k photons at output 1 and
        n1 + n2 - k at output 2, k = 0 ... n1 + n2.
    """

    t = np.sqrt(transmittance)
    r = 1j*np.sqrt(1 - transmittance)
    N = n1 + n2
    c = np.zeros(N + 1, dtype=complex)
    # (t x + r y)**n1 (r x + t y)**n2, x = b1+, y = b2+
    from math import comb, factorial, sqrt
    for j in range(n1 + 1):
        a = comb(n1, j)*t**j*r**(n1 - j)
        for l in range(n2 + 1):
            c[j + l] += a*comb(n2, l)*r**l*t**(n2 - l)
    k = np.arange(N + 1)
    norm = np.array([sqrt(factorial(int(kk))*factorial(N - int(kk))) for kk in k])
    amp = c*norm/sqrt(factorial(n1)*factorial(n2))
    return amp if amplitudes else np.abs(amp)**2


def cross_g2(n3, n4):
    """Normalized cross correlation <n3 n4>/(<n3><n4>) of the counts at two
    outputs (Hanbury Brown and Twiss)."""
    n3 = np.asarray(n3, dtype=float)
    n4 = np.asarray(n4, dtype=float)
    return np.mean(n3*n4)/(np.mean(n3)*np.mean(n4))


def hom_coincidence(delay, bandwidth_hz, transmittance=0.5, overlap=1.0, classical=False):
    """Coincidence rate of the Hong-Ou-Mandel experiment relative to
    distinguishable photons (1 at large delays).

    Parameters
    ----------
    delay : number or array
        Relative delay tau of the photons (s).
    bandwidth_hz : number
        FWHM of the photons' power spectrum (Hz), Gaussian; sigma_omega =
        2 pi FWHM/(2 sqrt(2 ln 2)).
    transmittance : number, optional
        T of the beam splitter.
    overlap : number, optional
        Overlap V of the other degrees of freedom (polarization, spatial
        mode, spectral purity), 1 for identical photons.
    classical : bool, optional
        Two classical fields of equal, constant intensity and random
        relative phase instead of single photons:
        1 - 2 T R V exp(-sigma_omega**2 tau**2), at most a dip of 1/2.

    Returns
    -------
    C : array
        1 - 2 T R V exp(-sigma_omega**2 tau**2)/(T**2 + R**2) for photons.
    """

    T = transmittance
    R = 1 - T
    s = 2*np.pi*bandwidth_hz/(2*np.sqrt(2*np.log(2)))
    e = overlap*np.exp(-(s*np.asarray(delay, dtype=float))**2)
    if classical:
        return 1 - 2*T*R*e
    return 1 - 2*T*R*e/(T**2 + R**2)


def hom_dip_fwhm(bandwidth_hz):
    """FWHM (s) of the Hong-Ou-Mandel dip of photons with a Gaussian spectrum
    of the given FWHM (Hz): 2 sqrt(ln 2)/sigma_omega = 4 ln 2/(pi sqrt(2)
    FWHM) = 0.624/FWHM."""
    s = 2*np.pi*bandwidth_hz/(2*np.sqrt(2*np.log(2)))
    return 2*np.sqrt(np.log(2))/s
