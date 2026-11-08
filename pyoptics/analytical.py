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

C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light by Small
Particles", Wiley (1983) (Mie series, cylinders).

W. J. Wiscombe, "Improved Mie scattering algorithms", Appl. Opt. 19,
1505-1509 (1980) (number of terms of the Mie series).
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


def cylinder_coefficients(radius, n_cyl, wl, n_medium=1.0, num_orders=None, polarization="E"):
    """Scattering coefficients of a circular dielectric cylinder for the scalar
    Helmholtz equation (polarization "E": E parallel to the cylinder axis, the
    default; "H": H parallel to it).

    The total field is written as

    - outside: u = sum_m i**m B_m (J_m(k r) + a_m H_m(k r)) exp(i m phi),
    - inside:  u = sum_m i**m B_m c_m J_m(m_rel k r) exp(i m phi),

    with the Hankel function H_m of the first kind (time dependence
    exp(-i omega t)), k = 2 pi n_medium/wl and m_rel = n_cyl/n_medium. B_m
    describes the incident field (B_m = 1 for a plane wave along phi = 0).
    For E parallel to the axis, u = E_axis and its radial derivative are
    continuous at r = radius; for H parallel to the axis, u = H_axis and its
    radial derivative divided by n**2. Either way this is the exact solution
    of Maxwell's equations [1]_.

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
    polarization : {"E", "H"}, optional
        Field along the cylinder axis.

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
    if polarization not in ("E", "H"):
        raise ValueError("polarization must be 'E' or 'H'")
    g = m_rel if polarization == "E" else 1/m_rel   # factor of the inner radial derivative
    den = Ji*Hp - g*Jip*H
    a_m = (g*Jip*J - Ji*Jp)/den
    c_m = 2j/(np.pi*x)/den   # Wronskian J H' - J' H = 2i/(pi x)

    return m, a_m, c_m


def cylinder_field(x, z, radius, n_cyl, wl, n_medium=1.0, k_x=None, amplitudes=None,
                   num_orders=None, polarization="E"):
    """Exact total field of a circular dielectric cylinder (axis along y,
    centred at x = z = 0) illuminated by plane waves travelling towards +z:
    the field component along the axis, E (TE polarization, the default) or
    H (TM).

    Parameters
    ----------
    x, z : arrays
        Coordinates (broadcastable against each other).
    radius, n_cyl, wl, n_medium, num_orders, polarization :
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
    m, a_m, c_m = cylinder_coefficients(radius, n_cyl, wl, n_medium, num_orders, polarization)
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


def disk_resonance(m, radius, n_disk, wavelength, n_medium=1.0, polarization="E"):
    """Exact resonance (quasi-normal mode) of a dielectric disk or cylinder
    with azimuthal number m, near a guess of its wavelength.

    The field along the axis is u = J_m(n_disk k r) exp(i m phi) inside and
    proportional to H_m(n_medium k r) exp(i m phi) outside (outgoing, time
    dependence exp(-i omega t)); matching u and its radial derivative (E
    along the axis) or the radial derivative over n**2 (H along the axis)
    at r = radius gives a complex vacuum wavenumber k with Im k < 0: the
    mode decays in time. High m and low radial order are the whispering
    gallery modes.

    Parameters
    ----------
    m : int
        Azimuthal number (abs(m); the modes +-m are degenerate).
    radius : number
    n_disk, n_medium : numbers
        Refractive indices inside and outside.
    wavelength : number or complex
        Guess of the vacuum wavelength: real, the root next to the zero
        crossing on the real axis nearest to it (whispering gallery modes);
        complex (2 pi/k of an estimate, e.g. a numerical resonance), the
        root Newton's method converges to from there (also strongly
        leaking modes).
    polarization : {"E", "H"}, optional
        Field along the axis.

    Returns
    -------
    k : complex
        Vacuum wavenumber of the resonance: wavelength 2 pi/Re(k),
        Q = Re(k)/(-2 Im(k)).

    References
    ----------
    .. [1] H. Cao and J. Wiersig, "Dielectric microcavities: Model systems for
       wave chaos and non-Hermitian physics", Rev. Mod. Phys. 87, 61-111
       (2015), section II.
    """

    from scipy.special import jv, jvp, hankel1, h1vp
    from scipy.optimize import newton

    if polarization not in ("E", "H"):
        raise ValueError("polarization must be 'E' or 'H'")
    m = abs(int(m))
    m_rel = n_disk/n_medium
    g = m_rel if polarization == "E" else 1/m_rel

    def f(x):   # x = n_medium k radius; the logarithmic derivatives match
        return g*jvp(m, m_rel*x)/jv(m, m_rel*x) - h1vp(m, x)/hankel1(m, x)

    if np.iscomplexobj(wavelength):   # a complex guess: Newton from there
        x0 = 2*np.pi*n_medium*radius/complex(wavelength)
        x = newton(f, x0, x1=x0*(1 + 1e-6) - 1e-6j, tol=1e-14, maxiter=200)
        return complex(x)/(n_medium*radius)
    # start from the zero crossing of Re f on the real axis nearest to the
    # guess (not a pole of the Bessel ratio), then Newton in the complex plane
    x_g = 2*np.pi*n_medium*radius/wavelength
    xs = np.linspace(0.8*x_g, 1.2*x_g, 4001)
    v = np.real(f(xs + 0j))
    i = np.nonzero((np.sign(v[:-1]) != np.sign(v[1:])) & (np.abs(v[:-1]) + np.abs(v[1:]) < 5))[0]
    if len(i) == 0:
        raise RuntimeError("no resonance near the given wavelength")
    i = i[np.argmin(np.abs(xs[i] - x_g))]
    x0 = xs[i] - v[i]*(xs[i + 1] - xs[i])/(v[i + 1] - v[i]) + 0j
    x = newton(f, x0, x1=x0*(1 + 1e-6) - 1e-6j, tol=1e-14, maxiter=200)

    return complex(x)/(n_medium*radius)


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


def mie_coefficients(radius, n_sph, wl, n_medium=1.0, num_orders=None):
    """Lorenz-Mie coefficients of a homogeneous sphere (vector theory, non-magnetic).

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
        Highest order n. Defaults to Wiscombe's criterion.

    Returns
    -------
    n : array
        Orders 1..num_orders.
    a_n, b_n : arrays
        Coefficients of the scattered field (electric and magnetic
        multipoles).
    c_n, d_n : arrays
        Coefficients of the inner field.

    Notes
    -----
    Bohren and Huffman [1]_, eq. (4.52) and (4.53), time dependence
    exp(-i omega t).

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
       by Small Particles", Wiley (1983), chapter 4.
    """

    from scipy.special import spherical_jn, spherical_yn

    k = 2*np.pi*n_medium/wl
    m = n_sph/n_medium
    x = k*radius
    if num_orders is None:
        num_orders = _num_orders(x, m)
    n = np.arange(1, num_orders + 1)
    j, jp = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    h = j + 1j*spherical_yn(n, x)
    hp = jp + 1j*spherical_yn(n, x, derivative=True)
    ji, jip = spherical_jn(n, m*x), spherical_jn(n, m*x, derivative=True)
    xj, xh = j + x*jp, h + x*hp              # [x j_n(x)]', [x h_n(x)]'
    mxj = ji + m*x*jip                        # [mx j_n(mx)]'
    den_e = m**2*ji*xh - h*mxj
    den_m = ji*xh - h*mxj
    a_n = (m**2*ji*xj - j*mxj)/den_e
    b_n = (ji*xj - j*mxj)/den_m
    c_n = (j*xh - h*xj)/den_m
    d_n = m*(j*xh - h*xj)/den_e

    return n, a_n, b_n, c_n, d_n


def mie_efficiencies(radius, n_sph, wl, n_medium=1.0, num_orders=None):
    """Extinction and scattering efficiencies (cross sections over the
    geometric one, pi radius**2) of a sphere from mie_coefficients()."""

    n, a, b, _, _ = mie_coefficients(radius, n_sph, wl, n_medium, num_orders)
    x = 2*np.pi*n_medium/wl*radius
    q_ext = 2/x**2*np.sum((2*n + 1)*np.real(a + b))
    q_sca = 2/x**2*np.sum((2*n + 1)*(np.abs(a)**2 + np.abs(b)**2))

    return q_ext, q_sca


def mie_amplitudes(theta, radius, n_sph, wl, n_medium=1.0, num_orders=None, chunk=None):
    """Far field scattering amplitudes S1, S2 of a sphere (Lorenz-Mie).

    The scattered far field is ::

        E_par  = exp(i k r)/(-i k r) S2(theta) E_par_inc,
        E_perp = exp(i k r)/(-i k r) S1(theta) E_perp_inc,

    with the components parallel and perpendicular to the scattering plane,
    so that the differential cross sections are ``|S1|**2/k**2`` (incident
    light polarized perpendicular to the scattering plane) and
    ``|S2|**2/k**2`` (parallel to it); unpolarized light gives their mean.

    Parameters
    ----------
    theta : array
        Scattering angles (radians, 0: forward).
    radius, n_sph, wl, n_medium :
        See mie_coefficients().
    num_orders : int, optional
        Highest order. Default: Wiscombe's criterion x + 4.05 x**(1/3) + 2
        for the size parameter x outside the sphere (the scattered field
        needs no more orders; the inner field of mie_field() needs the
        larger m x).
    chunk : int, optional
        Number of angles summed at once (memory: chunk times num_orders
        numbers). Default: about 4 million numbers.

    Returns
    -------
    S1, S2 : complex arrays (shape of theta)

    Notes
    -----
    Bohren and Huffman [1]_, eq. (4.74), with the angular functions pi_n
    and tau_n from their upward recurrence.

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
       by Small Particles", Wiley (1983), chapter 4.
    """

    if num_orders is None:
        x = 2*np.pi*n_medium/wl*radius
        num_orders = int(x + 4.05*x**(1/3) + 2)
    n, a_n, b_n, _, _ = mie_coefficients(radius, n_sph, wl, n_medium, num_orders)
    theta = np.asarray(theta, dtype=float)
    mu = np.cos(theta).ravel()
    w = (2*n + 1)/(n*(n + 1.0))
    wa, wb = w*a_n, w*b_n
    if chunk is None:
        chunk = max(1, 4_000_000//len(n))
    S1 = np.empty(mu.shape, dtype=complex)
    S2 = np.empty(mu.shape, dtype=complex)
    for start in range(0, len(mu), chunk):
        m = mu[start:start + chunk]
        pi_n = np.empty((len(n), len(m)))
        pi_prev, pi_cur = np.zeros(len(m)), np.ones(len(m))     # pi_0, pi_1
        for j in range(len(n)):
            pi_n[j] = pi_cur
            nn = j + 1
            pi_prev, pi_cur = pi_cur, ((2*nn + 1)*m*pi_cur - (nn + 1)*pi_prev)/nn
        pi_m1 = np.vstack([np.zeros(len(m)), pi_n[:-1]])
        tau_n = n[:, None]*m*pi_n - (n[:, None] + 1)*pi_m1
        S1[start:start + chunk] = wa @ pi_n + wb @ tau_n
        S2[start:start + chunk] = wa @ tau_n + wb @ pi_n

    return S1.reshape(theta.shape), S2.reshape(theta.shape)


def mie_field(x, y, z, radius, n_sph, wl, n_medium=1.0, num_orders=None):
    """Electric field of the Lorenz-Mie solution: a homogeneous sphere
    (centred at the origin) illuminated by a plane wave travelling towards
    +z, polarized along x, E_inc = exp(i k z) (time dependence
    exp(-i omega t), k = 2 pi n_medium/wl).

    The vector counterpart of sphere_field(): exact solution of Maxwell's
    equations, which depends on the polarization.

    Parameters
    ----------
    x, y, z : arrays
        Coordinates (broadcastable against each other).
    radius, n_sph, wl, n_medium, num_orders :
        See mie_coefficients().

    Returns
    -------
    E : complex array (3, ...)
        Ex, Ey, Ez of the total field (incident plus scattered outside,
        transmitted inside).

    Notes
    -----
    Sums the vector spherical harmonics M_o1n and N_e1n of Bohren and
    Huffman [1]_ (eq. 4.37, 4.45, 4.50).

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
       by Small Particles", Wiley (1983), chapter 4.
    """

    from scipy.special import spherical_jn, spherical_yn

    k = 2*np.pi*n_medium/wl
    m = n_sph/n_medium
    n_k, a_n, b_n, c_n, d_n = mie_coefficients(radius, n_sph, wl, n_medium, num_orders)
    x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z)))
    r = np.sqrt(x**2 + y**2 + z**2)
    rs = np.where(r > 0, r, 1e-12*wl)
    cos_t = np.clip(z/rs, -1, 1)
    sin_t = np.sqrt(1 - cos_t**2)
    phi = np.arctan2(y, x)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    inside = r < radius
    rho = np.where(inside, m*k*rs, k*rs)
    E_r = np.zeros(r.shape, dtype=complex)
    E_t = np.zeros(r.shape, dtype=complex)
    E_p = np.zeros(r.shape, dtype=complex)
    pi_prev, pi_n = np.zeros(r.shape), np.ones(r.shape)       # pi_0, pi_1
    j_prev = spherical_jn(0, rho)
    y_prev = spherical_yn(0, np.where(inside, 1.0, rho))
    for n, a, b, c, d in zip(n_k, a_n, b_n, c_n, d_n):
        tau_n = n*cos_t*pi_n - (n + 1)*pi_prev
        j = spherical_jn(n, rho)
        # outside: the scattered field (h_n), inside: the inner field (j_n)
        if np.any(~inside):
            yy = spherical_yn(n, np.where(inside, 1.0, rho))
            h = j + 1j*yy
            dh = (j_prev + 1j*y_prev) - n*h/rho      # [rho h_n]'/rho
            y_prev = yy
        else:
            h = dh = 0
        dj = j_prev - n*j/rho                        # [rho j_n]'/rho
        En = 1j**n*(2*n + 1)/(n*(n + 1))
        # radial functions times coefficients: M (z_n) and N ([rho z_n]'/rho, z_n/rho)
        zM = np.where(inside, c*j, -b*h)
        zN = np.where(inside, -1j*d*dj, 1j*a*dh)
        zR = np.where(inside, -1j*d*j/rho, 1j*a*h/rho)
        E_r += En*n*(n + 1)*sin_t*pi_n*zR
        E_t += En*(pi_n*zM + tau_n*zN)
        E_p += En*(-tau_n*zM - pi_n*zN)
        j_prev = j
        pi_prev, pi_n = pi_n, ((2*n + 1)*cos_t*pi_n - (n + 1)*pi_prev)/n
    E_r *= cos_p
    E_t *= cos_p
    E_p *= sin_p
    E = np.array([sin_t*cos_p*E_r + cos_t*cos_p*E_t - sin_p*E_p,
                  sin_t*sin_p*E_r + cos_t*sin_p*E_t + cos_p*E_p,
                  cos_t*E_r - sin_t*E_t])
    E[0] += np.where(inside, 0, np.exp(1j*k*z))

    return E
