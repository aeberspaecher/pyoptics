#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Analytical solutions for testing.

References
----------
M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999).

J. W. Goodman, "Introduction to Fourier Optics", 4th ed., W. H. Freeman (2017).

V. N. Mahajan, "Optical Imaging and Aberrations, Part II: Wave Diffraction
Optics", 2nd ed., SPIE Press (2011).
"""

import numpy as np
from scipy.special import j0, j1


def mahajan_on_axis_intensity(a, epsilon, I0, wavelength, z, R=None):
    """On-axis intensity for light diffracted by an annular aperture.

    Parameters
    ----------
    a, epsilon : numbers
        Outer and inner anulus radii defined by a and epsilon*a respecticvely.
    I0 :
    wavelength : number
    z : number
        Axial distance from apeture.
    R : number, optional
        Radius defining curvature 1/R of wavefront in aperture. Defaults to
        None, which is interpreted as the infite curvature of a collimated
        beam. R is the distance of geometric focus from the center of the
        aperture.

    Returns
    -------
    I_z : number
        On axis intensity at distance z from aperture.

    Notes
    -----
    Mahajan's results are obtained using scalar Rayleigh-Sommerfeld
    diffraction theory.

    TODO: Mahajan paper citation
    """

    s1_prime = np.sqrt(z**2 + epsilon**2*a**2)
    s2_prime = np.sqrt(z**2 + a**2)

    cos_alpha1 = z/s1_prime
    cos_alpha2 = z/s2_prime

    if R is None:  # collimated beam
        #I_z = 4*I0*np.sin(np.pi*a**2*(1.-epsilon**2)/(2.*wavelength*z))**2  # far field expression
        # TODO: test switching to far-field expression - advantages?
        s1_prime = np.sqrt(a**2 + z**2)
        s2_prime = np.sqrt((epsilon*a)**2 + z**2)
        I_z = I0*(cos_alpha1**2 + cos_alpha2**2 - 2*cos_alpha1*cos_alpha2*np.cos(2*np.pi*(s1_prime - s2_prime)/wavelength))
    else:  # focussed beam
        A0 = np.sqrt(I0*R**2)
        s1 = np.sqrt(R**2 + epsilon**2*a**2)
        s2 = np.sqrt(R**2 + a**2)
        d1 = s1_prime - s1
        d2 = s2_prime - s2
        I_z = A0**2*((cos_alpha1/d1)**2 + (cos_alpha2/d2)**2
                     - 2*cos_alpha1*cos_alpha2/(d1*d2)*np.cos(2*np.pi*(d1-d2)/wavelength)
                    )

    return I_z



def airy_pattern(r, NA, wavelength):
    """Intensity PSF of an aberration-free imaging system with a circular
    pupil (Airy pattern) in scalar paraxial theory.

    Parameters
    ----------
    r : number or array
        Distance from the optical axis in the image plane.
    NA : number
        Image-side numerical aperture.
    wavelength : number
        Wavelength in the image space medium (vacuum wavelength if NA already
        includes the refractive index).

    Returns
    -------
    I : number or array
        (2*J_1(v)/v)**2 with v = 2*pi*NA*r/wavelength, normalized to I(0) = 1.
        The first zero lies at r = 0.61*wavelength/NA.
    """

    v = 2*np.pi*NA*np.asarray(r, dtype=float)/wavelength
    v_safe = np.where(v == 0, 1.0, v)
    I = np.where(v == 0, 1.0, (2*j1(v_safe)/v_safe)**2)

    return I


def airy_encircled_energy(r, NA, wavelength):
    """Fraction of the energy of the Airy pattern inside a circle.

    Rayleigh's formula 1 - J_0(v)**2 - J_1(v)**2, v = 2*pi*NA*r/wavelength.

    Parameters
    ----------
    r : number or array
        Radius of the circle.
    NA : number
        Numerical aperture.
    wavelength : number
        Wavelength.

    Returns
    -------
    fraction : number or array
    """

    v = 2*np.pi*NA*np.asarray(r, dtype=float)/wavelength

    return 1.0 - j0(v)**2 - j1(v)**2


def circular_otf(f, NA, wavelength):
    """Diffraction limited optical transfer function (OTF) of an incoherent
    imaging system with a circular pupil.

    Parameters
    ----------
    f : number or array
        Spatial frequency (in 1/length, i.e. cycles per unit length, not
        angular wavenumber).
    NA : number
        Image-side numerical aperture.
    wavelength : number
        Wavelength.

    Returns
    -------
    OTF : number or array
        2/pi*(arccos(s) - s*sqrt(1 - s**2)) with s = ``|f|/f_c`` and the cut-off
        frequency f_c = 2*NA/wavelength; zero for s > 1. The OTF is real and
        non-negative, so it is identical to the MTF.
    """

    s = np.abs(np.asarray(f, dtype=float))/(2*NA/wavelength)
    s_clipped = np.clip(s, 0.0, 1.0)
    otf = 2/np.pi*(np.arccos(s_clipped) - s_clipped*np.sqrt(1 - s_clipped**2))

    return np.where(s <= 1.0, otf, 0.0)


def _num_orders(size_parameter, m):
    """Number of multipole orders for the series of a sphere or cylinder
    (Wiscombe's criterion with the larger of the outer and inner size
    parameter)."""

    x = max(size_parameter, abs(m)*size_parameter)

    return int(x + 4.05*x**(1/3) + 10)


def cylinder_coefficients(radius, n_cyl, wl, n_medium=1.0, num_orders=None):
    """Scattering coefficients of a circular dielectric cylinder for the scalar
    Helmholtz equation (TE polarization: E parallel to the cylinder axis).

    The total field is written as

    - outside: u = sum_m i**m B_m (J_m(k r) + a_m H_m(k r)) exp(i m phi),
    - inside:  u = sum_m i**m B_m c_m J_m(m_rel k r) exp(i m phi),

    with the Hankel function H_m of the first kind (time dependence
    exp(-i omega t)), k = 2 pi n_medium/wl and m_rel = n_cyl/n_medium. B_m
    describes the incident field (B_m = 1 for a plane wave along phi = 0).
    For E parallel to the axis, u = E_axis and its radial derivative are
    continuous at r = radius, so this is the exact solution of Maxwell's
    equations [1]_.

    Parameters
    ----------
    radius : number
        Cylinder radius.
    n_cyl : number
        Refractive index of the cylinder (may be complex).
    wl : number
        Vacuum wavelength.
    n_medium : number, optional
        Refractive index of the surrounding medium.
    num_orders : int, optional
        Highest order m. Defaults to Wiscombe's criterion.

    Returns
    -------
    m : array
        Orders -num_orders..num_orders.
    a_m, c_m : arrays
        Coefficients of the scattered and the inner field.

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
       by Small Particles", Wiley (1983), chapter 8.
    """

    from scipy.special import jv, jvp, hankel1, h1vp

    k = 2*np.pi*n_medium/wl
    m_rel = n_cyl/n_medium
    x = k*radius
    if num_orders is None:
        num_orders = _num_orders(x, m_rel)
    m = np.arange(-num_orders, num_orders + 1)
    J, Jp = jv(m, x), jvp(m, x)
    H, Hp = hankel1(m, x), h1vp(m, x)
    Ji, Jip = jv(m, m_rel*x), jvp(m, m_rel*x)
    den = Ji*Hp - m_rel*Jip*H
    a_m = (m_rel*Jip*J - Ji*Jp)/den
    c_m = 2j/(np.pi*x)/den   # Wronskian J H' - J' H = 2i/(pi x)

    return m, a_m, c_m


def cylinder_field(x, z, radius, n_cyl, wl, n_medium=1.0, k_x=None, amplitudes=None,
                   num_orders=None):
    """Exact total field of a circular dielectric cylinder (axis along y,
    centred at x = z = 0) illuminated by plane waves travelling towards +z, TE
    polarization (scalar Helmholtz equation).

    Parameters
    ----------
    x, z : arrays
        Coordinates (broadcastable against each other).
    radius, n_cyl, wl, n_medium, num_orders :
        See cylinder_coefficients().
    k_x : array, optional
        Transverse wavenumbers of the incident plane waves. Defaults to a
        single plane wave along z.
    amplitudes : array, optional
        Complex amplitudes of the plane waves exp(i (k_x x + k_z z)), with the
        phase referred to the cylinder centre. The incident field is
        sum(amplitudes*exp(i (k_x x + k_z z))), e.g. the angular spectrum of a
        beam. Defaults to 1.

    Returns
    -------
    u : array
        Total field (incident plus scattered outside, transmitted inside).
    """

    from scipy.special import jv, hankel1

    k = 2*np.pi*n_medium/wl
    m_rel = n_cyl/n_medium
    m, a_m, c_m = cylinder_coefficients(radius, n_cyl, wl, n_medium, num_orders)
    k_x = np.zeros(1) if k_x is None else np.asarray(k_x, dtype=float)
    amplitudes = np.ones(len(k_x)) if amplitudes is None else np.asarray(amplitudes)
    k_z = np.sqrt(np.clip(k**2 - k_x**2, 0, None))
    alpha = np.arctan2(k_x, k_z)   # plane wave directions (from the z axis)
    B_m = np.exp(-1j*np.outer(m, alpha)) @ amplitudes

    x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
    u = np.empty(x.shape, dtype=complex)
    r = np.hypot(x, z)
    phi = np.arctan2(x, z)
    inside = r < radius
    out = ~inside
    # outside: incident field (directly) plus the scattered field (series)
    x_o, z_o, kr_o, phi_o = x[out], z[out], k*r[out], phi[out]
    u_o = np.zeros(x_o.shape, dtype=complex)
    for k_xj, k_zj, A_j in zip(k_x, k_z, amplitudes):
        u_o += A_j*np.exp(1j*(k_xj*x_o + k_zj*z_o))
    kr_i, phi_i = m_rel*k*r[inside], phi[inside]
    u_i = np.zeros(kr_i.shape, dtype=complex)
    M = len(m)//2   # m = -M..M; orders +-m share the radial functions
    H = None
    for m_k in range(M + 1):
        B_p, B_n = B_m[M + m_k], B_m[M - m_k]
        ang_o = B_p*np.exp(1j*m_k*phi_o) + (B_n*np.exp(-1j*m_k*phi_o) if m_k else 0)
        ang_i = B_p*np.exp(1j*m_k*phi_i) + (B_n*np.exp(-1j*m_k*phi_i) if m_k else 0)
        if m_k < 2:
            H_prev, H = H, hankel1(m_k, kr_o)
        else:   # upward recurrence, stable for H_m (and much faster than hankel1)
            H_prev, H = H, 2*(m_k - 1)/kr_o*H - H_prev
        u_o += 1j**m_k*a_m[M + m_k]*H*ang_o
        u_i += 1j**m_k*c_m[M + m_k]*jv(m_k, kr_i)*ang_i
    u[out] = u_o
    u[inside] = u_i

    return u


def sphere_coefficients(radius, n_sph, wl, n_medium=1.0, num_orders=None):
    """Scattering coefficients of a dielectric sphere for the scalar Helmholtz
    equation (a scalar analogue of Mie theory).

    The total field of an axially symmetric incident field is written as

    - outside: u = sum_l i**l (2l + 1) B_l (j_l(k r) + a_l h_l(k r)) P_l(cos theta),
    - inside:  u = sum_l i**l (2l + 1) B_l c_l j_l(m_rel k r) P_l(cos theta),

    with spherical Bessel and Hankel functions (first kind, time dependence
    exp(-i omega t)) and Legendre polynomials P_l; B_l = 1 for a plane wave
    along +z. u and its radial derivative are continuous at r = radius.
    This is the "acoustic" (soft contrast) sphere of [1]_ and the scalar
    counterpart of the Lorenz-Mie solution [2]_: the electric field of the
    full vectorial problem additionally depends on the polarization.

    Parameters
    ----------
    radius : number
        Sphere radius.
    n_sph : number
        Refractive index of the sphere (may be complex).
    wl : number
        Vacuum wavelength.
    n_medium : number, optional
        Refractive index of the surrounding medium.
    num_orders : int, optional
        Highest order l. Defaults to Wiscombe's criterion.

    Returns
    -------
    l : array
        Orders 0..num_orders.
    a_l, c_l : arrays
        Coefficients of the scattered and the inner field.

    References
    ----------
    .. [1] P. M. Morse and K. U. Ingard, "Theoretical Acoustics", McGraw-Hill
       (1968).
    .. [2] C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
       by Small Particles", Wiley (1983), chapter 4.
    """

    from scipy.special import spherical_jn, spherical_yn

    k = 2*np.pi*n_medium/wl
    m_rel = n_sph/n_medium
    x = k*radius
    if num_orders is None:
        num_orders = _num_orders(x, m_rel)
    l = np.arange(num_orders + 1)
    j, jp = spherical_jn(l, x), spherical_jn(l, x, derivative=True)
    h = j + 1j*spherical_yn(l, x)
    hp = jp + 1j*spherical_yn(l, x, derivative=True)
    ji, jip = spherical_jn(l, m_rel*x), spherical_jn(l, m_rel*x, derivative=True)
    den = ji*hp - m_rel*jip*h
    a_l = (m_rel*jip*j - ji*jp)/den
    c_l = 1j/x**2/den   # Wronskian j h' - j' h = i/x**2

    return l, a_l, c_l


def sphere_field(rho, z, radius, n_sph, wl, n_medium=1.0, k_rho=None, amplitudes=None,
                 num_orders=None):
    """Exact solution of the scalar Helmholtz equation for a dielectric sphere
    (centred at the origin) illuminated by an axially symmetric field
    travelling towards +z.

    Parameters
    ----------
    rho, z : arrays
        Distance from the z axis and axial position (broadcastable).
    radius, n_sph, wl, n_medium, num_orders :
        See sphere_coefficients().
    k_rho : array, optional
        Radial wavenumbers of the incident field. Defaults to a single plane
        wave along z.
    amplitudes : array, optional
        Weights of the incident field
        sum(amplitudes*J_0(k_rho rho)*exp(i k_z z)) (a sum of cones of plane
        waves, e.g. the discretized Hankel transform of a beam, phases
        referred to z = 0). Defaults to 1.

    Returns
    -------
    u : array
        Total field (incident plus scattered outside, transmitted inside).

    Notes
    -----
    The averaged response to the plane waves on a cone of half angle alpha
    follows from the addition theorem of the Legendre polynomials,
    B_l = sum(amplitudes*P_l(cos alpha)) (the simplest case of the generalized
    Lorenz-Mie theory, G. Gouesbet and G. Grehan, "Generalized Lorenz-Mie
    Theories", Springer (2011)).
    """

    from scipy.special import spherical_jn, spherical_yn, eval_legendre, j0

    k = 2*np.pi*n_medium/wl
    m_rel = n_sph/n_medium
    l, a_l, c_l = sphere_coefficients(radius, n_sph, wl, n_medium, num_orders)
    k_rho = np.zeros(1) if k_rho is None else np.asarray(k_rho, dtype=float)
    amplitudes = np.ones(len(k_rho)) if amplitudes is None else np.asarray(amplitudes)
    k_z = np.sqrt(np.clip(k**2 - k_rho**2, 0, None))
    B_l = np.array([np.sum(amplitudes*eval_legendre(l_k, k_z/k)) for l_k in l])

    rho, z = np.broadcast_arrays(np.abs(np.asarray(rho, dtype=float)), np.asarray(z, dtype=float))
    u = np.empty(rho.shape, dtype=complex)
    r = np.hypot(rho, z)
    cos_theta = np.where(r > 0, z/np.where(r > 0, r, 1.0), 1.0)
    inside = r < radius
    out = ~inside
    # outside: incident field (directly) plus the scattered field (series)
    rho_o, z_o = rho[out], z[out]
    u_o = np.zeros(rho_o.shape, dtype=complex)
    for k_rho_j, k_zj, A_j in zip(k_rho, k_z, amplitudes):
        u_o += A_j*j0(k_rho_j*rho_o)*np.exp(1j*k_zj*z_o)
    kr_o, kr_i = k*r[out], m_rel*k*r[inside]
    u_i = np.zeros(kr_i.shape, dtype=complex)
    P_prev, P = np.zeros_like(cos_theta), np.ones_like(cos_theta)   # P_{-1}, P_0
    for l_k, a_k, c_k, B_k in zip(l, a_l, c_l, B_l):
        w = 1j**l_k*(2*l_k + 1)*B_k*P
        h_o = spherical_jn(l_k, kr_o) + 1j*spherical_yn(l_k, kr_o)
        u_o += w[out]*a_k*h_o
        u_i += w[inside]*c_k*spherical_jn(l_k, kr_i)
        P_prev, P = P, ((2*l_k + 1)*cos_theta*P - l_k*P_prev)/(l_k + 1)
    u[out] = u_o
    u[inside] = u_i

    return u
