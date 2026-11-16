#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Nematic liquid crystal cells: the director profile in a field, the
Fréedericksz transition, variable retarders, twisted nematic cells and
their viewing angle (notebook 140, liquid crystals; the tunable filters
of notebook 122, imaging spectroscopy).

Director and elastic energy. In a cell of thickness d between plates
parallel to x-y the director n = (cos theta cos phi, cos theta sin phi,
sin theta) depends on z only (tilt theta from the plates, twist phi).
The Frank energy per area is ::

    F = int 1/2 (K11 cos^2 theta + K33 sin^2 theta) theta'^2
            + 1/2 (K22 cos^2 theta + K33 sin^2 theta) cos^2 theta phi'^2 dz
        - 1/2 eps0 V^2 / int dz/eps_zz(theta),

eps_zz = eps_perp + Delta eps sin^2 theta, the last term the electrostatic
energy at a fixed voltage V (the field is not uniform: D_z is). With
strong anchoring (theta and phi fixed at both plates) `director_profile()`
minimizes the discretized F. For a planar cell without pretilt nothing
happens below the splay Fréedericksz threshold
V_th = pi sqrt(K11/(eps0 Delta eps)) (`threshold_voltage()`), independent
of d.

Optics. A slice with the director n is the uniaxial tensor
n_o**2 + (n_e**2 - n_o**2) n n^T (`crystals.uniaxial_eps()`); a cell is
a stack of slices for `crystals.stack_jones()` (`cell_layers()`).
At normal incidence the retardance is
2 pi/lambda int (n_eff(theta) - n_o) dz with
1/n_eff**2 = sin**2 theta/n_o**2 + cos**2 theta/n_e**2 (`retardance()`),
which falls from 2 pi Delta n d/lambda at 0 V towards 0: a variable
retarder, and the phase of a phase-only spatial light modulator for light
polarized along the rubbing.

A 90° twisted nematic cell guides the polarization when the Mauguin
parameter u = 2 d Delta n/lambda is large; between parallel polarizers
it transmits (Gooch and Tarry) ::

    T = sin**2((pi/2) sqrt(1 + u**2))/(1 + u**2),

zero at u = sqrt(3), sqrt(15), ... (`gooch_tarry()`; crossed polarizers:
1 - T, the normally white mode of a TN display).

Response: the decay after switching off takes about
gamma1 d**2/(pi**2 K11), switching on gamma1 d**2/(pi**2 K11
((V/V_th)**2 - 1)) (`response_times()`).

Units: SI for the material constants (K in N, gamma1 in Pa s), volts;
the profile uses z/d from 0 to 1, so lengths for the optics can be in any
unit consistent with the wavelength.

References
----------
P. G. de Gennes and J. Prost, "The Physics of Liquid Crystals", 2nd ed.,
Oxford University Press (1993).

P. Yeh and C. Gu, "Optics of Liquid Crystal Displays", 2nd ed., Wiley
(2010).

D.-K. Yang and S.-T. Wu, "Fundamentals of Liquid Crystal Devices", 2nd
ed., Wiley (2015).

C. H. Gooch and H. A. Tarry, "The optical properties of twisted nematic
liquid crystal structures with twist angles <= 90 degrees", J. Phys. D 8,
1575-1584 (1975).

J. Li and S.-T. Wu, "Extended Cauchy equations for the refractive indices
of liquid crystals", J. Appl. Phys. 95, 896-901 (2004).
"""

import numpy as np
from scipy.optimize import minimize

from .crystals import uniaxial_eps, director


__all__ = ["E7", "EPS0", "e7_indices", "threshold_voltage", "director_profile", "retardance",
           "cell_layers", "gooch_tarry", "mauguin_parameter", "response_times",
           "voltage_for_retardance"]


EPS0 = 8.8541878128e-12
# E7 at room temperature (Merck): elastic constants in N, viscosity in Pa s
E7 = {"K11": 11.1e-12, "K22": 6.5e-12, "K33": 17.1e-12, "eps_par": 19.0, "eps_perp": 5.2,
      "gamma1": 0.23}


def e7_indices(wavelength_nm):
    """Ordinary and extraordinary indices of E7 at 25 °C, three-term
    Cauchy fit of Li and Wu (2004).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    n_o, n_e : arrays
        1.523 and 1.739 at 589 nm.
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2
    n_e = 1.6933 + 0.0078/l2 + 0.0028/l2**2
    n_o = 1.4994 + 0.0070/l2 + 0.0004/l2**2

    return n_o, n_e


def threshold_voltage(K11=E7["K11"], eps_par=E7["eps_par"], eps_perp=E7["eps_perp"]):
    """Splay Fréedericksz threshold pi sqrt(K11/(eps0 Delta eps)) (V).

    Parameters
    ----------
    K11 : number, optional
        N.
    eps_par, eps_perp : numbers, optional

    Returns
    -------
    V_th : number
    """

    return np.pi*np.sqrt(K11/(EPS0*(eps_par - eps_perp)))


def _energy(x, N, th0, th1, ph0, ph1, k11, k22, k33, ep, de, u, twist):
    h = 1.0/N
    th = np.concatenate([[th0], x[:N - 1], [th1]])
    if twist:
        ph = np.concatenate([[ph0], x[N - 1:], [ph1]])
    else:
        ph = np.zeros(N + 1)
    dth, dph = np.diff(th), np.diff(ph)
    m = 0.5*(th[1:] + th[:-1])
    s, c = np.sin(m), np.cos(m)
    A = k11*c**2 + k33*s**2
    dA = (k33 - k11)*np.sin(2*m)
    B = (k22*c**2 + k33*s**2)*c**2
    dB = -4*k22*c**3*s + 2*k33*s*c*(c**2 - s**2)
    eps = ep + de*s**2
    deps = de*np.sin(2*m)
    I = np.sum(h/eps)
    F = np.sum(0.5*A*dth**2/h + 0.5*B*dph**2/h) - 0.5*u/I
    # gradient with respect to the node values
    g_seg = 0.25*dA*dth**2/h + 0.25*dB*dph**2/h            # d/dm, half of it to each node
    g_seg_el = 0.5*u/I**2*(-h*deps/eps**2)                 # dW/dm
    gth = np.zeros(N + 1)
    gth[:-1] += g_seg + 0.5*g_seg_el - A*dth/h
    gth[1:] += g_seg + 0.5*g_seg_el + A*dth/h
    g = gth[1:-1]
    if twist:
        gph = np.zeros(N + 1)
        gph[:-1] -= B*dph/h
        gph[1:] += B*dph/h
        g = np.concatenate([g, gph[1:-1]])
    return F, g


def director_profile(V, n_nodes=41, pretilt=np.radians(2.0), twist=0.0, K11=E7["K11"],
                     K22=E7["K22"], K33=E7["K33"], eps_par=E7["eps_par"], eps_perp=E7["eps_perp"],
                     initial=None):
    """Equilibrium director of a nematic cell at the voltage V.

    Parameters
    ----------
    V : number
        RMS voltage across the cell (V).
    n_nodes : int, optional
        Nodes from z = 0 to z = d.
    pretilt : number, optional
        Tilt at both plates (radians); the rubbing directions are
        antiparallel (no splay at 0 V).
    twist : number, optional
        Twist from the first to the second plate (radians), pi/2 for a TN
        cell; the director at z = 0 is along x.
    K11, K22, K33 : numbers, optional
        Elastic constants (N).
    eps_par, eps_perp : numbers, optional
        Low frequency dielectric constants (positive anisotropy).
    initial : (tilt, twist) arrays, optional
        Start of the minimization (e.g. the profile at a nearby voltage).

    Returns
    -------
    z : array
        z/d.
    tilt, twist : arrays
        Radians, at the nodes.
    """

    N = n_nodes - 1
    z = np.linspace(0.0, 1.0, n_nodes)
    de = eps_par - eps_perp
    u = EPS0*V**2/K11
    if initial is None:
        th_init = pretilt + (np.pi/2 - pretilt)*0.9*np.sin(np.pi*z)*min(1.0, (V/threshold_voltage(K11, eps_par, eps_perp))**2/4 + 0.05)
        ph_init = twist*z
    else:
        th_init, ph_init = initial
    use_twist = abs(twist) > 0
    x0 = th_init[1:-1]
    if use_twist:
        x0 = np.concatenate([x0, ph_init[1:-1]])
    args = (N, pretilt, pretilt, 0.0, twist, 1.0, K22/K11, K33/K11, eps_perp, de, u, use_twist)
    bounds = [(-np.pi/2, np.pi/2)]*(N - 1) + ([(None, None)]*(N - 1) if use_twist else [])
    res = minimize(_energy, x0, args=args, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
    th = np.concatenate([[pretilt], res.x[:N - 1], [pretilt]])
    ph = np.concatenate([[0.0], res.x[N - 1:], [twist]]) if use_twist else np.zeros(n_nodes)

    return z, th, ph


def _midpoints(a):
    a = np.asarray(a, dtype=float)
    return 0.5*(a[1:] + a[:-1])


def retardance(tilt, thickness, wavelength, n_o, n_e):
    """Retardance at normal incidence of a cell with the tilt profile
    (nodes equally spaced from z = 0 to d; twist ignored).

    Parameters
    ----------
    tilt : array
        Radians, at the nodes.
    thickness, wavelength : numbers
        Same unit.
    n_o, n_e : numbers

    Returns
    -------
    delta : number
        2 pi/lambda int (n_eff - n_o) dz (radians).
    """

    m = _midpoints(tilt)
    n_eff = n_o*n_e/np.sqrt(n_o**2*np.cos(m)**2 + n_e**2*np.sin(m)**2)

    return 2*np.pi/wavelength*np.mean(n_eff - n_o)*thickness


def cell_layers(tilt, twist, thickness, n_o, n_e, azimuth=0.0):
    """Slices of a liquid crystal cell for `crystals.stack_jones()`.

    Parameters
    ----------
    tilt, twist : arrays
        Director at the nodes (radians).
    thickness : number
    n_o, n_e : numbers
    azimuth : number, optional
        Rotation of the cell about z (radians): the director at z = 0 then
        points along this azimuth.

    Returns
    -------
    layers : list of (eps, thickness)
    """

    m_t, m_p = _midpoints(tilt), _midpoints(twist) + azimuth
    eps = uniaxial_eps(n_o, n_e, director(m_t, m_p))
    h = thickness/len(m_t)

    return [(e, h) for e in eps]


def gooch_tarry(u):
    """Transmission of a 90° twisted nematic cell between parallel
    polarizers (along the director at the entrance):
    sin**2((pi/2) sqrt(1 + u**2))/(1 + u**2), u = 2 d Delta n/lambda.

    Parameters
    ----------
    u : number or array

    Returns
    -------
    T : array
    """

    u = np.asarray(u, dtype=float)

    return np.sin(np.pi/2*np.sqrt(1 + u**2))**2/(1 + u**2)


def mauguin_parameter(thickness, delta_n, wavelength):
    """u = 2 d Delta n/lambda (the cell guides the polarization for u
    much larger than 1).

    Parameters
    ----------
    thickness, delta_n, wavelength : numbers

    Returns
    -------
    u : number
    """

    return 2*thickness*delta_n/wavelength


def response_times(thickness, V, gamma1=E7["gamma1"], K11=E7["K11"], eps_par=E7["eps_par"],
                   eps_perp=E7["eps_perp"]):
    """Rise and decay times of the small-tilt director (seconds).

    Parameters
    ----------
    thickness : number
        m.
    V : number
        Switching voltage (V).
    gamma1 : number, optional
        Rotational viscosity (Pa s).
    K11, eps_par, eps_perp : numbers, optional

    Returns
    -------
    tau_on, tau_off : numbers
        gamma1 d**2/(pi**2 K11 ((V/V_th)**2 - 1)) and gamma1 d**2/(pi**2 K11).
    """

    tau_off = gamma1*thickness**2/(np.pi**2*K11)
    r = (V/threshold_voltage(K11, eps_par, eps_perp))**2 - 1

    return (tau_off/r if r > 0 else np.inf), tau_off


def voltage_for_retardance(target, voltages, retardances):
    """Voltage at which a cell has the retardance `target`, interpolated in
    a computed curve (retardance falling with the voltage).

    Parameters
    ----------
    target : number or array
    voltages, retardances : arrays
        The curve, voltages ascending.

    Returns
    -------
    V : array
        nan where the target is outside the curve.
    """

    v = np.asarray(voltages, dtype=float)
    d = np.asarray(retardances, dtype=float)
    order = np.argsort(d)
    t = np.asarray(target, dtype=float)
    out = np.interp(t, d[order], v[order])

    return np.where((t < d.min()) | (t > d.max()), np.nan, out)
