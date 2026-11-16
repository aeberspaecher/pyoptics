#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Acousto-optics: diffraction of light by sound, Raman-Nath against
Bragg diffraction, modulators (AOM), deflectors (AOD) and tunable filters
(AOTF) (notebook 141, acousto-optics; the AOTF of
notebook 122, imaging spectroscopy).

A sound wave of frequency f and speed v writes a moving index grating of
period Lambda = v/f and amplitude Delta n = sqrt(M2 I_a/2) (acoustic
intensity I_a, figure of merit M2 = n**6 p**2/(rho v**3)). Light crossing
a transducer of length L is described by two numbers: the peak phase
modulation (Raman-Nath parameter) nu = 2 pi Delta n L/lambda and the
Klein-Cook parameter Q = 2 pi lambda L/(n Lambda**2), the ratio of the
length to the length over which a diffracted order walks off the grating
phase. In the paraxial coupled wave (Raman-Nath) equations the order m
grows as ::

    dE_m/dzeta = (nu/2)(E_{m-1} - E_{m+1}) - i (Q/2) m (m - alpha) E_m,

zeta = z/L, for light incident at alpha times the Bragg angle
theta_B = lambda/(2 n Lambda) (inside the medium, alpha = 1: Bragg
matched for the order +1). For Q << 1 the orders are J_m(nu)**2
(Raman-Nath diffraction, many orders); for Q >> 1 only the Bragg order
survives, with the efficiency sin**2(nu/2) at alpha = 1 (`coupled_orders()`,
`bragg_efficiency()` with a phase mismatch).

The order +1 is shifted up by f (the moving grating: a Doppler shift).
An AOM switches in the time the sound needs to cross the beam,
0.64 D/v for the 10-90 % rise with a Gaussian beam of 1/e**2 diameter D
(`aom_rise_time()`). An AOD deflects by lambda f/v (outside the crystal);
over the bandwidth Delta f it resolves N = tau Delta f spots, tau = D/v
the access time (`aod_spots()`). An AOTF uses the anisotropic Bragg
diffraction in a birefringent crystal (TeO2): the order changes its
polarization, and only a narrow band is matched. In Chang's
non-collinear (parallel tangent) geometry with the light at theta_i to
the optic axis the matched acoustic frequency is ::

    f = (v Delta n/lambda) sqrt(sin**4 theta_i + sin**2 2 theta_i)

and the passband FWHM 0.9 lambda**2/(Delta n L sin**2 theta_i)
(`aotf_frequency()`, `aotf_transmission()`).

Units: SI (m, s, Hz, W) unless a name says otherwise.

References
----------
W. R. Klein and B. D. Cook, "Unified approach to ultrasonic light
diffraction", IEEE Trans. Sonics Ultrason. 14, 123-134 (1967).

A. Korpel, "Acousto-Optics", 2nd ed., Marcel Dekker (1997).

A. Yariv and P. Yeh, "Optical Waves in Crystals", Wiley (1984), chapters
9 and 10.

B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019), chapter 20.

I. C. Chang, "Noncollinear acousto-optic filter with large angular
aperture", Appl. Phys. Lett. 25, 370-372 (1974).

N. Uchida, "Optical properties of single-crystal paratellurite (TeO2)",
Phys. Rev. B 4, 3736-3745 (1971).
"""

import numpy as np
from scipy.linalg import expm


__all__ = ["MATERIALS", "teo2_indices", "klein_cook_q", "raman_nath_parameter",
           "delta_n_from_power", "coupled_orders", "bragg_efficiency", "bragg_angle",
           "aom_rise_time", "aod_spots", "aod_deflection", "aotf_frequency", "aotf_passband",
           "aotf_transmission"]


# speed (m/s), index (at 633 nm, 10.6 µm for Ge) and figure of merit M2 (s^3/kg)
MATERIALS = {
    "fused silica": {"v": 5960.0, "n": 1.457, "M2": 1.51e-15},
    "TeO2 longitudinal": {"v": 4200.0, "n": 2.26, "M2": 34.5e-15},
    "TeO2 shear": {"v": 617.0, "n": 2.26, "M2": 793e-15},
    "PbMoO4": {"v": 3630.0, "n": 2.39, "M2": 36.3e-15},
    "Ge (10.6 µm)": {"v": 5500.0, "n": 4.0, "M2": 180e-15},
}


def teo2_indices(wavelength_nm):
    """Ordinary and extraordinary indices of paratellurite (TeO2), the
    Sellmeier fit of Uchida (1971).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    n_o, n_e : arrays
        2.260 and 2.412 at 633 nm.
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2
    n_o = np.sqrt(1 + 2.584*l2/(l2 - 0.1342**2) + 1.157*l2/(l2 - 0.2638**2))
    n_e = np.sqrt(1 + 2.823*l2/(l2 - 0.1342**2) + 1.542*l2/(l2 - 0.2631**2))

    return n_o, n_e


def klein_cook_q(wavelength, n, period, length):
    """Klein-Cook parameter Q = 2 pi lambda L/(n Lambda**2).

    Parameters
    ----------
    wavelength : number or array
        Vacuum wavelength.
    n : number
    period : number or array
        Acoustic wavelength v/f.
    length : number
        Interaction length.

    Returns
    -------
    Q : array
    """

    return 2*np.pi*np.asarray(wavelength)*length/(n*np.asarray(period)**2)


def raman_nath_parameter(delta_n, length, wavelength):
    """nu = 2 pi Delta n L/lambda, the peak phase modulation.

    Parameters
    ----------
    delta_n, length, wavelength : numbers or arrays

    Returns
    -------
    nu : array
    """

    return 2*np.pi*np.asarray(delta_n)*length/np.asarray(wavelength)


def delta_n_from_power(power, M2, length, height):
    """Index amplitude sqrt(M2 I_a/2) for the acoustic power P spread over
    the transducer area L x H.

    Parameters
    ----------
    power : number or array
        W.
    M2 : number
        s**3/kg.
    length, height : numbers
        m.

    Returns
    -------
    delta_n : array
    """

    return np.sqrt(M2*np.asarray(power)/(length*height)/2)


def bragg_angle(wavelength, frequency, v, n=1.0):
    """Bragg angle arcsin(lambda f/(2 n v)) (inside a medium of index n;
    n = 1 for the angle in air).

    Parameters
    ----------
    wavelength, frequency, v : numbers or arrays
    n : number, optional

    Returns
    -------
    theta_B : array
        Radians.
    """

    return np.arcsin(np.asarray(wavelength)*np.asarray(frequency)/(2*n*v))


def coupled_orders(nu, Q, alpha=1.0, n_orders=12):
    """Amplitudes of the diffraction orders behind a sound column
    (Raman-Nath coupled wave equations, solved exactly by a matrix
    exponential).

    Parameters
    ----------
    nu : number
        Raman-Nath parameter 2 pi Delta n L/lambda.
    Q : number
        Klein-Cook parameter.
    alpha : number, optional
        Angle of incidence in units of the Bragg angle (1: matched for
        order +1, -1 for order -1, 0 normal incidence).
    n_orders : int, optional
        Orders -n_orders .. n_orders are kept.

    Returns
    -------
    m : array
        Orders.
    E : complex array
        Amplitudes; abs(E)**2 sums to 1.
    """

    m = np.arange(-n_orders, n_orders + 1)
    M = np.diag(-0.5j*Q*m*(m - alpha)).astype(complex)
    idx = np.arange(len(m) - 1)
    M[idx + 1, idx] += nu/2          # dE_m/dz += (nu/2) E_{m-1}
    M[idx, idx + 1] -= nu/2          # dE_m/dz -= (nu/2) E_{m+1}
    E0 = np.zeros(len(m), dtype=complex)
    E0[n_orders] = 1.0

    return m, expm(M) @ E0


def bragg_efficiency(nu, mismatch=0.0):
    """Two-wave Bragg efficiency with the phase mismatch Delta k L/2:
    (nu/2)**2 sin**2(s)/s**2, s = sqrt((nu/2)**2 + mismatch**2).

    Parameters
    ----------
    nu : number or array
    mismatch : number or array, optional
        Delta k L/2 (radians).

    Returns
    -------
    eta : array
    """

    a = np.asarray(nu)/2
    s = np.sqrt(a**2 + np.asarray(mismatch)**2)

    return a**2*np.sinc(s/np.pi)**2


def aom_rise_time(beam_diameter, v):
    """10-90 % rise time 0.64 D/v of an AOM for a Gaussian beam of 1/e**2
    diameter D.

    Parameters
    ----------
    beam_diameter, v : numbers

    Returns
    -------
    t_r : number
        s.
    """

    return 0.64*beam_diameter/v


def aod_deflection(wavelength, frequency, v):
    """Angle lambda f/v between the orders 0 and 1 outside the crystal
    (radians, small angles).

    Parameters
    ----------
    wavelength, frequency, v : numbers or arrays

    Returns
    -------
    angle : array
    """

    return np.asarray(wavelength)*np.asarray(frequency)/v


def aod_spots(aperture, v, bandwidth):
    """Number of resolvable spots tau Delta f of a deflector (uniformly lit
    aperture D, access time tau = D/v).

    Parameters
    ----------
    aperture, v, bandwidth : numbers

    Returns
    -------
    N : number
    """

    return aperture/v*bandwidth


def aotf_frequency(wavelength, delta_n, v, theta_i):
    """Acoustic frequency that tunes a non-collinear AOTF to `wavelength`:
    (v Delta n/lambda) sqrt(sin**4 theta_i + sin**2 2 theta_i).

    Parameters
    ----------
    wavelength : number or array
    delta_n : number or array
        n_e - n_o at the wavelength.
    v : number
        Speed of the shear wave.
    theta_i : number
        Angle of the light to the optic axis (radians).

    Returns
    -------
    f : array
        Hz.
    """

    s = np.sin(theta_i)

    return v*np.asarray(delta_n)/np.asarray(wavelength)*np.sqrt(s**4 + np.sin(2*theta_i)**2)


def _b(wavelength, delta_n_func, h=1e-3):
    lam = np.asarray(wavelength, dtype=float)
    dn = delta_n_func(lam)
    ddn = (delta_n_func(lam*(1 + h)) - delta_n_func(lam*(1 - h)))/(2*h*lam)
    return 2*np.pi*(dn - lam*ddn)


def aotf_passband(wavelength, length, theta_i, delta_n_func):
    """Passband FWHM 1.8 pi lambda**2/(b L sin**2 theta_i) of an AOTF (Chang),
    b = 2 pi (Delta n - lambda dDelta n/dlambda).

    Parameters
    ----------
    wavelength : number or array
    length : number
        Interaction length (same unit as the wavelength).
    theta_i : number
    delta_n_func : callable
        Delta n(wavelength).

    Returns
    -------
    fwhm : array
    """

    lam = np.asarray(wavelength, dtype=float)

    return 1.8*np.pi*lam**2/(_b(lam, delta_n_func)*length*np.sin(theta_i)**2)


def aotf_transmission(wavelength, wavelength0, length, theta_i, delta_n_func, nu=np.pi):
    """Transmission of an AOTF tuned to wavelength0 (two-wave Bragg
    diffraction with the mismatch Delta k = b sin**2 theta_i (1/lambda0 -
    1/lambda), which reproduces Chang's passband).

    Parameters
    ----------
    wavelength : array
    wavelength0 : number
        Tuned wavelength.
    length : number
        Interaction length (same unit).
    theta_i : number
    delta_n_func : callable
    nu : number, optional
        Coupling (pi: 100 % at the peak; it changes slowly with the
        wavelength at constant drive power, ignored here).

    Returns
    -------
    T : array
        Fraction of the light in the right polarization diffracted into
        the passband (a polarizer in front halves it for unpolarized
        light). The FWHM is within 2 % of `aotf_passband()` for a weak
        drive and about 10 % narrower at nu = pi (100 % at the peak).
    """

    lam = np.asarray(wavelength, dtype=float)
    b = _b(wavelength0, delta_n_func)
    dk = b*np.sin(theta_i)**2*(1/wavelength0 - 1/lam)

    return bragg_efficiency(nu, dk*length/2)
