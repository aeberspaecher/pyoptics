#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.

Convention: fields vary as exp(i(k.r - omega*t)), as in Zangwill. Absorbing
layers therefore have the complex refractive index n + i*k with k > 0, and a
layer of thickness t adds the phase +k_0*(n + i*k)*cos(theta)*t. McLeod's
characteristic matrices (exp(+i*omega*t), n - i*k) enter as their complex
conjugates.
"""

# Literature:
# Zangwill, Modern Electrodynamics, CUP, 2013
# Brooker, Modern Classical Optics, OUP, 2002
# McLeod, Thin Film Optical Filters, Institute of Physics Publishing, 2001

from copy import copy
from itertools import product

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from numpy import sin

from .utils import Z_0 as Z_vac, wavenumber, freq_grid


def _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k):
    """Extend n and k arrays by embedding medium and substrate values.
    """

    n_extended = np.asarray([embedding_n] + list(n) + [substrate_n])
    k_extended = np.asarray([embedding_k] + list(k) + [substrate_k])

    return n_extended, k_extended


def _impedances(complex_n):
    """Return impedances for stack defined by (complex) refractive indices.
    """

    return Z_vac/complex_n


def _cos_theta(theta_0, complex_n):
    """Compute cosines of complex propagation angles in all media of a given
    stack.

    Parameters
    ----------
    theta_0 : double
        Angle in embedding medium.
    complex_n : array
        Refractive indices for whole stack.

    Returns
    -------
    angles : array
        Array cos(theta_i).
    """

    # cos(theta) = sqrt(1 - sin(theta)); sin(theta) is found from Snell's law:
    cos_theta = complex_sqrt(1.0-(complex_n[0]/complex_n*sin(theta_0))**2)

    # NOTE: n_i sin(theta_i) = n_j sin(theta_j) for any j, i! thus, we can use
    # any layer (here: the embedding medium) to compute angles in any other
    # layer.

    return cos_theta


def _tilted_Z(Z_in, cos_theta, pol):
    """Compute 'tilted impedances' for a stack.

    Parameters
    ----------
    Z_in, cos_theta : arrays
        (Complex) impedances of layer stack and cosines of propagation angles.
    pol : string
        One of {'s','TE'; 'p', 'TM'}.

    Returns
    -------
    Z_tilted : array
        Polarization dependent tilted impedances.
    """

    # NOTE: source is McLeod's Thin Film Optical Filters, 3rd edition. McLeod
    # uses admittances, whereas our code uses impedances (cmp. Zangwil).
    # Admittance is the reciprocal value of impedance.
    # Using admittance y: H = Y*E
    # Using impedance: E = Z*H = 1/Y*H

    Z = copy(Z_in)
    # compute 'tilted' impedance from McLeod's tilted admittances:
    if pol in ("s", "TE"):
        Z /= cos_theta  # McLeod 3rd ed, eq. (8.5)
    elif pol in ("p", "TM"):
        Z *= cos_theta  # McLeod 3rd ed, eq. (8.6)
    else:
        raise ValueError("'pol' must be one of 's'/'TE', 'p'/'TM'")

    return Z


def r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
               substrate_k, pol, wavelength, phi0=None):
    """Reflection and transmission coefficients for a thin film stack.

    Parameters
    ----------
    angle : double
        Incident angle in embedding medium.
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film
        stack, n + i*k (k > 0 for absorption).
    t : arrays
        Thicknesses of the layers.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization. Also accepted: "TE" or "TM".

    Returns
    -------
    r, t : double
        Reflection and transmission amplitude of the stack.
    """

    # TODO: make checks more verbose and raise sensible exceptions
    assert len(n) >= 1
    assert len(k) >= 1
    assert len(t) == len(n) == len(k)

    n, k = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)  # extend n, k arrays by embedding medium and substrate
    r, t = _r_t(np.sin(angle), n + 1j*k, t, pol, wavelength, phi0)

    return r[()], t[()]


def _r_t(sin_theta_0, complex_n, t, pol, wavelength, phi0=None):
    """Reflection and transmission coefficients of a stack for an array of
    incidence angles given by sin(theta_0) in the embedding medium (may be
    complex or > 1 for evanescent incidence).

    complex_n holds the refractive indices of embedding medium, layers and
    substrate; t the thicknesses of the layers only (may be empty). r and t
    have the shape of sin_theta_0.
    """

    sin_theta_0 = np.asarray(sin_theta_0)
    Z = _impedances(complex_n)  # complex impedance for all media

    # cos(theta_j) in all media from Snell's law n_j sin(theta_j) = n_0 sin(theta_0);
    # first axis: media, remaining axes: those of sin_theta_0
    n_col = complex_n.reshape((-1,) + (1,)*sin_theta_0.ndim)
    cos_theta = complex_sqrt(1.0 - (complex_n[0]*sin_theta_0/n_col)**2)
    # grazing incidence (cos(theta) = 0) makes the tilted impedances singular;
    # a tiny cos(theta) gives the limiting values instead:
    cos_theta = np.where(cos_theta == 0, 1E-12, cos_theta)
    Z = _tilted_Z(Z.reshape(n_col.shape)*np.ones_like(cos_theta), cos_theta, pol)  # McLeod's 'tilted admittances'

    if phi0 is None:  # if no additional phases are given, assume zero additional phases
        phi0 = np.zeros(len(complex_n))
    else:
        phi0 = np.array([0.] + list(phi0) + [0.])  # additional zeros for substrate and embedding medium

    # multiply Zangwill's transfer matrices of all layers, element by element
    # for all angles. Layers with evanescent waves or absorption make the
    # entries grow like exp(|Im(phi_j)|); to avoid overflows, each matrix is
    # divided by that factor and the product is kept normalized, with the
    # logarithm of the total scale in log_scale (r does not depend on it):
    A, B = np.ones_like(cos_theta[0]), np.zeros_like(cos_theta[0])
    C, D = np.zeros_like(cos_theta[0]), np.ones_like(cos_theta[0])
    log_scale = np.zeros(np.shape(cos_theta[0]))
    for j in range(1, len(complex_n) - 1):
        phi_j = t[j-1]*wavenumber(wavelength, complex_n[j])*cos_theta[j] + phi0[j]  # 'phase' accumulated in layer j

        g = np.abs(np.imag(phi_j))
        exp_plus, exp_minus = np.exp(1j*phi_j - g), np.exp(-1j*phi_j - g)
        cos_j, sin_j = (exp_plus + exp_minus)/2, (exp_plus - exp_minus)/2j  # cos(phi_j), sin(phi_j) times exp(-g)

        a, b = cos_j, -1j*Z[j]*sin_j
        c, d = -1j/Z[j]*sin_j, cos_j
        A, B, C, D = A*a + B*c, A*b + B*d, C*a + D*c, C*b + D*d

        norm = np.maximum.reduce([np.abs(A), np.abs(B*np.abs(Z[0])**-1), np.abs(C*np.abs(Z[0])), np.abs(D)])
        A, B, C, D = A/norm, B/norm, C/norm, D/norm
        log_scale += g + np.log(norm)

    # solve Zangwill's eq. (17.92) for r and t:
    denominator = A*Z[-1] + B + Z[0]*(C*Z[-1] + D)
    r = (A*Z[-1] + B - Z[0]*(C*Z[-1] + D))/denominator
    t = 2*Z[-1]/denominator*np.exp(-log_scale)

    return r, t


def stack_transfer_functions(x, y, n, k, t, embedding_n, embedding_k, substrate_n,
                             substrate_k, pol, wavelength, normal_order=True):
    """Reflection and transmission of a layer stack as transfer functions for
    the angular spectrum of a field.

    A field in a plane z = const in the embedding medium is a superposition of
    plane waves exp(i*(k_x*x + k_y*y + k_z*z)), whose amplitudes FT() yields
    (see pyoptics.fft). Each plane wave hits the stack (layers parallel to the
    x-y plane, light incident from the embedding medium towards +z) under the
    angle given by sin(theta_0) = sqrt(k_x**2 + k_y**2)/(k_0*n_0) and is
    reflected and transmitted with r(theta_0) and t(theta_0) of
    r_t_coeffs(). Thus

        E_reflected = inv_FT(r*FT(E_incident)),
        E_transmitted = inv_FT(t*FT(E_incident)),

    with E_incident and E_reflected taken at the top of the stack (the
    interface embedding medium/first layer) and E_transmitted at its bottom
    (last layer/substrate). Propagate them with the propagators to other
    planes. Note that reflected plane waves run towards -z: with the
    convention exp(i(k.r - omega*t)), propagating the reflected field by a
    distance d away from the stack means using exp(+i*k_z*d) as for the
    incident field, i.e. the usual propagators applied to E_reflected.

    This is a scalar model: the field is one tangential component of the
    electric field, and all plane waves are treated as either s (TE) or p
    (TM) polarized. This is exact for fields that only vary along x (k_y = 0)
    with the electric field along y (pol="TE") or the magnetic field along y
    (pol="TM"), and a good approximation for small angular spectra.
    Plane waves with sin(theta_0) > 1 are evanescent in the embedding medium;
    they are included with the complex angle that the formulas give (e.g.
    frustrated total internal reflection).

    Parameters
    ----------
    x, y : arrays
        Linear coordinates of the field's grid.
    n, k, t : arrays
        Refractive indices n + i*k and thicknesses of the layers (top to
        bottom). Empty arrays describe a single interface.
    embedding_n, embedding_k, substrate_n, substrate_k : numbers
        Refractive index of the embedding medium (incidence side) and the
        substrate.
    pol : string
        "s"/"TE" or "p"/"TM".
    wavelength : number
        Vacuum wavelength.
    normal_order : boolean, optional
        If True (default), the arrays are centered as FT() expects; if False,
        they are in the standard order of numpy.fft.fft2().

    Returns
    -------
    r, t : arrays
        Complex transfer functions of shape (Ny, Nx), ratios of the
        tangential electric field components as in r_t_coeffs().
    """

    assert len(t) == len(n) == len(k)

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=normal_order)
    complex_n = (np.asarray([embedding_n] + list(n) + [substrate_n])
                 + 1j*np.asarray([embedding_k] + list(k) + [substrate_k]))
    # r and t only depend on |k_perp|: evaluate the stack once per distinct
    # value (about 8 times fewer than grid points) and map back to the grid
    k_perp_sq, grid_index = np.unique(K_X**2 + K_Y**2, return_inverse=True)
    sin_theta_0 = np.sqrt(k_perp_sq)/wavenumber(wavelength, complex_n[0])
    r, t = _r_t(sin_theta_0, complex_n, t, pol, wavelength)

    return r[grid_index].reshape(K_X.shape), t[grid_index].reshape(K_X.shape)


def R_T_A_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                 substrate_k, pol, wavelength, incoherent=None, N_phases=None):
    """Same as r_t_coeffs, but for intensity reflectance and transmittance.
    Also return 'absorbtance' such that R + T + A = 1.

    Parameters
    ----------
    angle : double
        Incident angle in embedding medium.
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film stack.
    t : arrays
        Thicknesses of the layers.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization. Also accepted: "TE" or "TM".
    wavelength : number
        Wavelength of incident light.
    incoherent : array, boolean, optional
        Element i is True if the i-th layer is "incoherent" (i.e. such that
        phase information is lost). Defaults to None which is interpreted as full
        coherence in all layers.
    N_phases : int, optional
        Use N_phases phase offsets for incoherent layers in averaging. Only
        used if at least one incoherent layer is present.

    Note
    ----
    A = 0 can only be expected for losless media under the condition that the
    embeddig medium is equal to the substrate.
    """

    if (incoherent is not None) and N_phases is None:
        raise ValueError("N_phases must be a number")

    # generate ensemble of phases to add in each layer. if a layer is coherent,
    # zero phase is added beyond the phase acquired by propagation through the
    # layer. if the layer is incoherent, linearly distributed values in [0, 2pi)
    # are added for each realization of the ensemble.
    if incoherent is not None:
        add_phases = [ np.linspace(0, 2*np.pi, N_phases, endpoint=False) if is_incoherent else [0.0]
                       for is_incoherent in incoherent
                     ]
    else:
        add_phases = [[0.0] for _ in n]

    # incoherence means that intensities (not amplitudes) add up, so average
    # |r|**2 and |t|**2 over all realizations of the phase ensemble:
    abs_r_sq, abs_t_sq = [], []
    for phi0 in product(*add_phases):
        curr_r, curr_t = r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                                    substrate_k, pol, wavelength, phi0)
        abs_r_sq.append(np.abs(curr_r)**2)
        abs_t_sq.append(np.abs(curr_t)**2)

    # r and t are ratios of the tangential field components. the z-component
    # of the Poynting vector is proportional to |E_tan|**2*Re(1/Z_tilted),
    # which holds for both s and p polarization:
    n_extended, k_extended = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)
    complex_n = n_extended + 1j*k_extended
    Z = _impedances(complex_n)  # complex impedance for all media
    cos_theta = _cos_theta(angle, complex_n)
    Z = _tilted_Z(Z, cos_theta, pol)

    R = np.mean(abs_r_sq)
    T = np.mean(abs_t_sq)*np.real(1/Z[-1])/np.real(1/Z[0])
    A = 1.0 - R - T

    return R, T, A
