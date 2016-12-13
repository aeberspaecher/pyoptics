#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for Müller calculus: Müller matrices of common elements optical
elements and Stokes vectors of common polarization states.
"""

import numpy as np
from scipy.linalg import inv

from pyoptics.utils import sin_cos
from pyoptics.interfaces import refracted_angle


# common Stokes vectors:
S_LHP = np.array([1., 1, 0, 0])  # linear horizontal
S_LVP = np.array([1., -1, 0, 0])  # linear vertical
S_Lp45P = np.array([1., 0, 1, 0])  # linear +45°
S_Lm45P = np.array([1., 0, -1, 0])  # linear -45°
S_LCP = np.array([1., 0, 0, -1])  # left-hand circular
S_RCP = np.array([1., 0, 0, +1])  # right-hand circular
S_unpol = np.array([1., 0, 0, 0])  # unpolarized


def stokes_from_complex_E(E):
    """Compute Stokes vector for polarization state described by complex
    electric field components.

    Parameters
    ----------
    E : array
        Complex field amplitude, E[0] is interpreteted as x-component; E[1] as
        y-component.

    Returns
    -------
    S : array
        Stokes vector.
    """

    E_x, E_y = E[0], E[1]

    S = np.array( [
                    E_x*np.conj(E_x) + E_y*np.conj(E_y),
                    E_x*np.conj(E_x) - E_y*np.conj(E_y),
                    E_x*np.conj(E_y) + E_y*np.conj(E_x),
                    1j*(E_x*np.conj(E_y) - E_y*np.conj(E_x))
                  ]
                )

    return S


def stokes_DOP(S):
    """Compute degree of polarization for given Stokes vector.

    Parameters
    ----------
    S : arrray

    Returns
    -------
    DOP : number
    """

    return np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)/S[0]


def M_linear_polarizer(phi):
    """Linear polarizer with axis of polarization at angle phi.

    Parameters
    ----------
    phi : number
        Angle of axis

    Returns
    -------
    M : array
        Müller matrix
    """

    s, c = np.sin(2*phi), np.cos(2*phi)

    M = np.array( [ [1., c, s, 0],
                    [c, c**2, s*c, 0],
                    [s, s*c, s**2, 0],
                    [0, 0, 0, 0]
                  ]
                )/2.

    return M


def M_retarder(phi, delta):
    """Müller matrix for retarder with fast axis angle phi and retardance delta.

    Parameters
    ----------
    phi : number
        Angle of fast axis.
    delta : number
        Phase retardance [rad].

    Returns
    -------
    M : array
        Müller matrix
    """

    c_d = np.cos(delta)
    s_d = np.sin(delta)
    c_p = np.cos(2*phi)
    s_p = np.sin(2*phi)

    M = np.array( [ [1., 0, 0, 0],
                    [0, c_p**2 + c_d*s_p**2, c_d*s_d - c_p*c_d*s_p, s_p*s_d],
                    [0, c_p*s_p - c_p*c_d*s_d, c_d*c_p**2 + s_p**2, -c_p*s_d],
                    [0, -s_p*s_d, c_p*s_d, c_d]
                  ]
                )

    return M


def M_attenuator(trans):
    """Müller matrix for attenuator with transmission trans.

    Parameters
    ----------
    trans : number
        Transmission

    Returns
    -------
    M : array
        Müller matrix
    """

    return trans*np.eye(4)


def M_reflection_dielectric(r_s, r_p, *args):
    """Müller matrix for reflection from a dieletric film stack.

    Parameters
    ----------
    r_s, r_p : complex
        Reflection coefficients for s- and p-polarization.

    Returns
    -------
    M : array
        Müller matrix for reflection in a coordinate system in which
        p-polarization is x-axis and s-polarization is y-axis.

    Note
    ----
    The function accepts additional (unused) arguments which allows to call it
    with a signature as used in transmission_dielectric().
    """

    M = np.array( [ [np.abs(r_p)**2 + np.abs(r_s)**2, np.abs(r_p)**2 - np.abs(r_s)**2, 0, 0],
                    [np.abs(r_p)**2 - np.abs(r_s)**2, np.abs(r_p)**2 + np.abs(r_s)**2, 0, 0],
                    [0, 0, 2*np.real(np.conj(r_p)*r_s), 2*np.imag(np.conj(r_p)*r_s)],
                    [0, 0, -2*np.imag(np.conj(r_p)*r_s), 2*np.imag(np.conj(r_p)*r_s)]
                  ]
                )

    return M


def M_transmission_dielectric(t_s, t_p, theta_in, n_embedding, n_substrate, pol):
    """Müller matrix for transmission from a dieletric film stack.

    Parameters
    ----------
    t_s, t_p : complex
        Transmission coefficients for s- and p-polarization.
    theta_in : number
        Incident angle (in rad).
    n_embedding, n_substrate : number
        Refractive index of embedding medium and substrate.
    pol : string
        "TE" or "TM" for corresponding polarization.

    Returns
    -------
    M : array
        Müller matrix for transmission in a coordinate system in which
        p-polarization is x-axis and s-polarization is y-axis.
    """

    theta_out = refracted_angle(theta_in, n_embedding, n_substrate)
    if (pol == "TE") or (pol == "s"):
        eta_m = n_embedding*np.cos(theta_in)  # embedding medium prefactor
        eta_s = n_substrate*np.cos(theta_out)  # substrate prefactor
    else:
        eta_m = n_embedding/np.cos(theta_in)
        eta_s = n_substrate/np.cos(theta_out)

    M = eta_s/(2*eta_m) * np.array( [ [np.abs(t_p)**2 + np.abs(t_s)**2, np.abs(t_p)**2 - np.abs(t_s)**2, 0, 0],
                                      [np.abs(t_p)**2 - np.abs(t_s)**2, np.abs(t_p)**2 + np.abs(t_s)**2, 0, 0],
                                      [0, 0, 2*np.real(np.conj(t_p)*t_s), 2*np.imag(np.conj(t_p)*t_s)],
                                      [0, 0, -2*np.imag(np.conj(t_p)*t_s), 2*np.imag(np.conj(t_p)*t_s)]
                                    ]
                                  )

    return M


def M_reflection_metal(theta, n_metal, n_embedding=1.0):
    """Müller matrix for reflection from a metal surface.

    Parameters
    ----------
    theta : number
        Angle of incidence (in rad).
    n_metal : complex
        (Complex) refractive index of metal.

    Returns
    -------
    M : array
        Müller matrix for reflection in a coordinate system in which
        p-polarization is x-axis and s-polarization is y-axis.

    Note
    ----
    Source: Christoph Keller's book "Astrophysical Spectropolarimetry", p. 330.
    """

    i = theta
    r = refracted_angle(theta, n_embedding, n_metal)

    s_fac = (n_embedding*np.cos(i) - n_metal*np.cos(r))/(n_embedding*np.cos(i) + n_metal*np.cos(r))
    rho_s = np.abs(s_fac)
    phi_s = np.angle(s_fac)

    p_fac = np.tan(i - r)/np.tan(i + r)
    rho_p = np.abs(p_fac)
    phi_p = np.angle(rho_p)

    delta = phi_s - phi_p
    s, c = sin_cos(delta)

    M = 1./2 * np.array( [ [rho_s**2 + rho_p**2, rho_s**2 - rho_p**2, 0, 0],
                           [rho_s**2 - rho_p**2, rho_s**2 + rho_p**2, 0, 0],
                           [0, 0, 2*rho_s*rho_p*c, 2*rho_s*rho_p*s],
                           [0, 0, -2*rho_s*rho_p*s, 2*rho_s*rho_p*c]
                         ]
                       )

    return M


def M_from_jones(J):
    """Construct Müller matrix from Jones matrix.

    Parameters
    ----------
    J : array
        Jones matrix

    Returns
    -------
    M : array
        Müller matrix
    """

    A = np.array([ [1., 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0] ])
    A_inv = inv(A)
    M = reduce(np.dot, [A, np.kron(J, np.conj(J.T)), A_inv])

    return M


def retardance_from_AOI(theta, phi, delta_0, n_o, n_e):
    """Retardance of uniaxial crystal hit by ray under an angle.

    Parameters
    ----------
    theta : number
        Angle of incidence
    phi : number
        Angle between plane of incidence and optic axis of crystal
    delta_0 : number
        Retardance at normal incidence
    n_o, n_e : numbers
        Refractive indices as seen by ordinary and extraordinary rays.

    Returns
    -------
    delta : number
        Retardance for oblique incidence

    Note
    ----
    The formula is only valid for sin(theta)**2 << n_o**2, n_e**2.
    """

    delta = delta_0*(1. + np.sin(theta)**2/(2*n_o)*(np.sin(phi)**2/n_e - np.cos(phi)**2/n_o))

    return delta

