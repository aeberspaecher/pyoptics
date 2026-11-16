#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Electro-optic modulators: the Pockels and Kerr effects, Pockels cells,
phase modulators and the Mach-Zehnder intensity modulator with its bias,
extinction, chirp and bandwidth (notebook 139, electro-optic modulators;
the modulator of notebook 120, fibre link).

The linear electro-optic (Pockels) effect changes the impermeability
tensor eta = eps^-1 by Delta eta_i = sum_k r_ik E_k in the contracted
notation (i = 1..6 for xx, yy, zz, yz, xz, xy). `pockels_eps()` applies it
to any tensor; `r_3m()` (LiNbO3, LiTaO3) and `r_42m()` (KDP, ADP) build
the r matrices of the two classical point groups, with constants in
`LINBO3` and `KDP` (pm/V). The quadratic (Kerr) effect of isotropic
media is Delta n = lambda K E**2 along the field (`kerr_retardance()`).

Pockels cells. In a longitudinal KDP cell (field along the optic axis z)
the principal axes turn by 45° and the retardance is
2 pi n_o**3 r_63 V/lambda, independent of the length: the half-wave
voltage V_pi = lambda/(2 n_o**3 r_63) (`half_wave_voltage()`). A transverse
LiNbO3 phase modulator with the field along z changes the phase of light
polarized along z by pi n_e**3 r_33 Gamma V L/(lambda g) (electrode gap
g, overlap Gamma).

Phase modulation by beta sin(Omega t) puts the power J_n(beta)**2 into
the sidebands n Omega (`phase_modulation_sidebands()`).

Mach-Zehnder modulator. Two arms with the drive phases
phi_k = pi V_k/V_pi (plus a bias) recombine to ::

    E_out = E_in (sqrt(a) exp(i phi_1) + sqrt(1 - a) exp(i phi_2))*...,

`mzm_field()`; with an ideal splitter (a = 1/2) the power is
cos**2((phi_1 - phi_2)/2) and the phase (phi_1 + phi_2)/2. Push-pull
drive (V_2 = -V_1) gives pure amplitude modulation, single-arm drive a
chirp with the small signal parameter alpha = (V_1 + V_2)/(V_1 - V_2)
(Koyama and Iga): the instantaneous frequency follows
d phi/dt = (alpha/2) d ln P/dt. Unequal Y branches (power fraction a in
arm 1 at splitter and combiner) limit the extinction ratio to
1/(2a - 1)**2 (`mzm_extinction()`). `mzm_nrz()` is a complete NRZ transmitter: bits,
a drive with finite rise time, bias and swing, ready for
`fibrecomm.propagate()`.

Bandwidth. A travelling-wave electrode of length L with the microwave
index n_m, the optical group index n_g and the microwave loss alpha(f)
modulates with the response ::

    H(f) = (1 - exp(-alpha L + i u))/(alpha L - i u),  u = 2 pi f L (n_m - n_g)/c

(`travelling_wave_response()`); without loss the electrical 3 dB
bandwidth (abs(H)**2 = 1/2) is f L = 1.39 c/(pi abs(n_m - n_g)).

Units: voltages in V, fields in V/m, lengths and wavelengths in m unless
a name says otherwise, r in m/V (the constants are given in pm/V and
converted by the functions), time in ps and bit rates in Gb/s in
`mzm_nrz()` as in `fibrecomm`.

References
----------
A. Yariv and P. Yeh, "Photonics", 6th ed., Oxford University Press
(2007), chapter 9.

B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019), chapter 21.

E. L. Wooten et al., "A review of lithium niobate modulators for
fiber-optic communications systems", IEEE J. Sel. Top. Quantum Electron.
6, 69-82 (2000).

F. Koyama and K. Iga, "Frequency chirping in external modulators", J.
Lightwave Technol. 6, 87-93 (1988).

D. E. Zelmon, D. L. Small and D. Jundt, "Infrared corrected Sellmeier
coefficients for congruently grown lithium niobate and 5 mol.%
magnesium oxide-doped lithium niobate", J. Opt. Soc. Am. B 14, 3319-3322
(1997).
"""

import numpy as np
from scipy.special import jv

from .fibrecomm import _gaussian_filter_periodic


__all__ = ["LINBO3", "KDP", "KDP_DEUTERATED", "NITROBENZENE_KERR", "linbo3_indices", "r_3m",
           "r_42m", "pockels_eps", "half_wave_voltage", "transverse_half_wave_voltage",
           "kerr_retardance", "phase_modulation_sidebands", "mzm_field", "mzm_power",
           "mzm_extinction", "mzm_chirp", "mzm_nrz", "travelling_wave_response",
           "bandwidth_3db"]


C0 = 299792458.0

# electro-optic constants in pm/V and indices (LiNbO3 at 1550 nm, KDP at 633 nm)
LINBO3 = {"r13": 8.6, "r22": 3.4, "r33": 30.8, "r51": 28.0, "n_o": 2.211, "n_e": 2.138,
          "n_microwave": 4.2, "n_group": 2.18}
KDP = {"r41": 8.77, "r63": 10.3, "n_o": 1.507, "n_e": 1.467}
KDP_DEUTERATED = {"r41": 8.8, "r63": 26.4, "n_o": 1.502, "n_e": 1.462}
NITROBENZENE_KERR = 2.4e-12        # m/V**2


def linbo3_indices(wavelength_nm):
    """Ordinary and extraordinary indices of congruent LiNbO3 at room
    temperature (Zelmon, Small and Jundt 1997, 0.4 to 5 µm).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    n_o, n_e : arrays
        2.211 and 2.138 at 1550 nm.
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2
    n_o = np.sqrt(1 + 2.6734*l2/(l2 - 0.01764) + 1.2290*l2/(l2 - 0.05914) + 12.614*l2/(l2 - 474.6))
    n_e = np.sqrt(1 + 2.9804*l2/(l2 - 0.02047) + 0.5981*l2/(l2 - 0.0666) + 8.9543*l2/(l2 - 416.08))

    return n_o, n_e


def r_3m(r13, r22, r33, r51):
    """Electro-optic matrix (6 x 3, contracted) of point group 3m
    (LiNbO3, LiTaO3), optic axis z, mirror plane y-z.

    Parameters
    ----------
    r13, r22, r33, r51 : numbers

    Returns
    -------
    r : (6, 3) array
    """

    return np.array([[0, -r22, r13],
                     [0, r22, r13],
                     [0, 0, r33],
                     [0, r51, 0],
                     [r51, 0, 0],
                     [-r22, 0, 0]], dtype=float)


def r_42m(r41, r63):
    """Electro-optic matrix of point group -42m (KDP, ADP), optic axis z.

    Parameters
    ----------
    r41, r63 : numbers

    Returns
    -------
    r : (6, 3) array
    """

    r = np.zeros((6, 3))
    r[3, 0] = r[4, 1] = r41
    r[5, 2] = r63

    return r


_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def pockels_eps(eps, r, E_field):
    """Dielectric tensor in an electric field (linear electro-optic effect).

    Parameters
    ----------
    eps : (3, 3) array
        Tensor without field, in the crystal frame of r.
    r : (6, 3) array
        Electro-optic matrix in m/V.
    E_field : (..., 3) array
        Field in V/m.

    Returns
    -------
    eps_E : (..., 3, 3) array
        (eps^-1 + Delta eta)^-1, Delta eta_i = sum_k r_ik E_k.
    """

    E_field = np.asarray(E_field, dtype=float)
    d = E_field @ np.asarray(r).T                          # (..., 6)
    deta = np.zeros(E_field.shape[:-1] + (3, 3))
    for i, (a, b) in enumerate(_VOIGT):
        deta[..., a, b] = d[..., i]
        deta[..., b, a] = d[..., i]

    return np.linalg.inv(np.linalg.inv(np.asarray(eps, dtype=complex)) + deta)


def half_wave_voltage(wavelength, n_o, r63):
    """Half-wave voltage lambda/(2 n_o**3 r_63) of a longitudinal Pockels
    cell (KDP type, field along the optic axis).

    Parameters
    ----------
    wavelength : number
        m.
    n_o : number
    r63 : number
        m/V.

    Returns
    -------
    V_pi : number
        V.
    """

    return wavelength/(2*n_o**3*r63)


def transverse_half_wave_voltage(wavelength, n, r, length, gap, overlap=1.0):
    """Half-wave voltage lambda g/(n**3 r Gamma L) of a transverse phase
    modulator (phase shift pi).

    Parameters
    ----------
    wavelength : number
    n : number
        Index for the polarization used (n_e with r33 in LiNbO3).
    r : number
        m/V.
    length, gap : numbers
        Electrode length and gap (m).
    overlap : number, optional
        Overlap Gamma of the optical mode and the field (0.5 to 0.7 in Ti
        diffused waveguides).

    Returns
    -------
    V_pi : number
    """

    return wavelength*gap/(n**3*r*overlap*length)


def kerr_retardance(field, length, K=NITROBENZENE_KERR):
    """Retardance 2 pi K L E**2 of a Kerr cell (Delta n = lambda K E**2).

    Parameters
    ----------
    field : number or array
        V/m.
    length : number
        m.
    K : number, optional
        Kerr constant (m/V**2), nitrobenzene by default.

    Returns
    -------
    delta : array
        Radians.
    """

    return 2*np.pi*K*length*np.asarray(field)**2


def phase_modulation_sidebands(beta, n_max=10):
    """Powers J_n(beta)**2 of the carrier and sidebands of a sinusoidal
    phase modulation exp(i beta sin(Omega t)).

    Parameters
    ----------
    beta : number
        Modulation index (radians).
    n_max : int, optional

    Returns
    -------
    n, P : arrays
        Orders -n_max..n_max and their power fractions.
    """

    n = np.arange(-n_max, n_max + 1)

    return n, jv(n, beta)**2


def mzm_field(V1, V2, V_pi, bias=0.0, split=0.5, combine=None):
    """Output field of a Mach-Zehnder modulator for unit input.

    E = sqrt(c) sqrt(a) exp(i(phi_1 + bias/2)) + sqrt(1 - c) sqrt(1 - a)
    exp(i(phi_2 - bias/2)), phi_k = pi V_k/V_pi.

    Parameters
    ----------
    V1, V2 : numbers or arrays
        Drive voltages of the two arms (V).
    V_pi : number
        Voltage for a phase shift pi in one arm (the switching voltage of
        a single-drive modulator; push-pull needs V_pi/2 in each arm).
    bias : number, optional
        Static phase difference between the arms (radians); 0 is the peak
        of the transfer function, pi/2 quadrature.
    split : number, optional
        Power fraction a of the input splitter into arm 1.
    combine : number, optional
        Power fraction of arm 1 at the output combiner (default: split).

    Returns
    -------
    E : complex array
    """

    a = split
    c = split if combine is None else combine
    p1 = np.pi*np.asarray(V1)/V_pi + bias/2
    p2 = np.pi*np.asarray(V2)/V_pi - bias/2

    return np.sqrt(c*a)*np.exp(1j*p1) + np.sqrt((1 - c)*(1 - a))*np.exp(1j*p2)


def mzm_power(V, V_pi, bias=np.pi/2, push_pull=True, split=0.5):
    """Transfer function abs(E)**2 for the drive V (push-pull: +V/2 and
    -V/2 on the arms, single drive: V on arm 1).

    Parameters
    ----------
    V : number or array
    V_pi : number
        Total voltage swing between peak and null (for push-pull the arm
        voltages are +-V/2, so the switching voltage is V_pi either way).
    bias : number, optional
    push_pull : bool, optional
    split : number, optional

    Returns
    -------
    P : array
    """

    V = np.asarray(V, dtype=float)
    if push_pull:
        return np.abs(mzm_field(V/2, -V/2, V_pi, bias, split))**2
    return np.abs(mzm_field(V, 0.0*V, V_pi, bias, split))**2


def mzm_extinction(split):
    """Largest extinction ratio (linear) of an MZM whose splitter and
    combiner send the power fraction a into arm 1: the null field is
    a - (1 - a), so ER = 1/(2a - 1)**2.

    Parameters
    ----------
    split : number or array

    Returns
    -------
    ER : array
    """

    a = np.asarray(split, dtype=float)
    return (1.0/np.abs(2*a - 1))**2


def mzm_chirp(V1_swing, V2_swing):
    """Small signal chirp parameter alpha = (V1 + V2)/(V1 - V2) of an MZM
    biased at quadrature, for the drive swings of the two arms (push-pull
    V2 = -V1: 0; single drive V2 = 0: 1; the sign of alpha flips with the
    bias slope).

    Parameters
    ----------
    V1_swing, V2_swing : numbers

    Returns
    -------
    alpha : number
    """

    return (V1_swing + V2_swing)/(V1_swing - V2_swing)


def mzm_nrz(bits, bit_rate, V_pi=4.0, swing=None, alpha=0.0, split=0.5, samples_per_bit=32,
            rise_time=None, p_in=2.0, insertion_loss_db=0.0, bias=None):
    """Field envelope of an NRZ transmitter with a Mach-Zehnder modulator.

    The drive is the bit pattern with a Gaussian rise time, swinging
    between -swing/2 and +swing/2 around the bias point; the arms get
    (1 + alpha) V/2 and -(1 - alpha) V/2 (alpha = 0 push-pull, alpha = +-1
    single drive), so that the small signal chirp is alpha.

    Parameters
    ----------
    bits : array of 0 and 1
    bit_rate : number
        Gb/s.
    V_pi : number, optional
        Switching voltage (V).
    swing : number, optional
        Peak to peak drive (default V_pi: from null to peak).
    alpha : number, optional
        Chirp parameter; its sign is that of the frequency shift on rising
        edges (positive: blue shifted rising edges, like a directly
        modulated laser).
    split : number, optional
        Splitting ratio (0.5 ideal; the finite extinction).
    samples_per_bit : int, optional
    rise_time : number, optional
        10-90 % rise time of the drive (ps), default 0.3 bit.
    p_in : number, optional
        Laser power (mW).
    insertion_loss_db : number, optional
    bias : number, optional
        Bias phase (radians), default quadrature pi/2 below the peak.

    Returns
    -------
    t : array
        Time (ps).
    A : complex array
        Field envelope, abs(A)**2 in mW, phase convention of
        `fibrecomm.nrz()`.
    """

    bits = np.asarray(bits)
    T_bit = 1E3/bit_rate
    if swing is None:
        swing = V_pi
    if rise_time is None:
        rise_time = 0.3*T_bit
    if bias is None:
        bias = np.pi/2
    n = len(bits)*samples_per_bit
    t = np.arange(n)*T_bit/samples_per_bit
    v = np.repeat(np.where(bits > 0, 0.5, -0.5), samples_per_bit).astype(float)
    v = _gaussian_filter_periodic(v, rise_time/2.563/(T_bit/samples_per_bit))*swing
    # the transfer is cos**2((phi1 - phi2)/2) with phi1 - phi2 = pi v/V_pi + bias - pi:
    # v = +swing/2 at the peak for swing = V_pi
    V1 = (1 + alpha)*v/2
    V2 = -(1 - alpha)*v/2
    E = mzm_field(V1, V2, V_pi, bias=bias - np.pi, split=split)
    # remove the static phase, keep the modulation; exp(-i omega t): a phase growing
    # with time is a red shift, so the chirp sign is flipped to make alpha > 0 blue
    E = np.conj(E)*np.exp(1j*np.angle(np.mean(E)))
    loss = 10**(-insertion_loss_db/10)

    return t, np.sqrt(p_in*loss)*E


def travelling_wave_response(f_ghz, length, n_m=LINBO3["n_microwave"], n_g=LINBO3["n_group"],
                             alpha0_db=0.0):
    """Modulation response of a travelling-wave electrode.

    Parameters
    ----------
    f_ghz : number or array
        Modulation frequency (GHz).
    length : number
        Electrode length (m).
    n_m, n_g : numbers, optional
        Microwave index of the electrode and optical group index.
    alpha0_db : number, optional
        Microwave loss in dB/(cm sqrt(GHz)) (skin effect, about 0.2 to 0.6
        for gold electrodes).

    Returns
    -------
    H : complex array
        Normalized to 1 at f = 0; abs(H)**2 is the electrical response.
    """

    f = np.asarray(f_ghz, dtype=float)
    u = 2*np.pi*f*1e9*length*(n_m - n_g)/C0
    aL = alpha0_db*np.sqrt(f)*length*100/(20*np.log10(np.e))   # field attenuation * L
    x = aL - 1j*u
    small = np.abs(x) < 1e-9
    x_safe = np.where(small, 1.0, x)

    return np.where(small, 1.0, (1 - np.exp(-x_safe))/x_safe)


def bandwidth_3db(length, n_m=LINBO3["n_microwave"], n_g=LINBO3["n_group"], alpha0_db=0.0,
                  f_max=500.0):
    """Electrical 3 dB bandwidth (abs(H)**2 = 1/2) of a travelling-wave
    modulator (GHz).

    Parameters
    ----------
    length : number
        m.
    n_m, n_g : numbers, optional
    alpha0_db : number, optional
        dB/(cm sqrt(GHz)).
    f_max : number, optional
        Upper end of the search (GHz).

    Returns
    -------
    f3 : number
    """

    f = np.linspace(0.0, f_max, 200001)
    H2 = np.abs(travelling_wave_response(f, length, n_m, n_g, alpha0_db))**2
    i = np.argmax(H2 < 0.5)
    if i == 0:
        return np.inf

    return float(np.interp(0.5, [H2[i], H2[i - 1]], [f[i], f[i - 1]]))
