#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical forces on a dielectric sphere in a focused laser beam (optical
tweezers), in three models.

Conventions: fields vary as exp(-i*omega*t), the beam travels towards +z and
its geometric focus is at the origin; r0 is the position of the centre of the
sphere. Forces are given as dimensionless trapping efficiencies
Q = F*c/(n_medium*P), the fraction of the momentum flux n_medium*P/c of the
beam (power P) that is transferred to the sphere. For a laser power P (W),
F = Q*n_medium*P/c (N).

Ray optics (Ashkin's model, valid for spheres much larger than the
wavelength):

- ashkin_efficiencies(): closed form scattering and gradient efficiencies
  Q_s, Q_g of a single ray that hits a sphere, summed over all internal
  reflections, for s or p polarization.
- traced_efficiencies(): the same by following the ray through 0, 1, 2, ...
  internal reflections and adding up the momentum of every ray that leaves
  (a check of the closed form).
- ray_optics_force(): Q of a fan of rays from the pupil of an aplanatic
  objective focused into the medium, each with the power and polarization it
  carries.

Wave optics (any sphere size):

- focus_plane_waves(): the Richards-Wolf (Debye) focus of an aplanatic
  objective as a quadrature of plane waves over the pupil, with the
  polarization rotated by the lens and an apodization sqrt(cos theta), for
  a Gaussian beam that fills the pupil to a given degree.
- focus_field(): E and H of such a superposition at any points;
  beam_power(): its power.
- MieTrap: the exact (Lorenz-Mie) force on a sphere in such a focus. Each
  plane wave is scattered as in Mie theory (analytical.mie_amplitudes());
  the force is the momentum that the scattered far field takes away from the
  incident far field, i.e. the Maxwell stress tensor integrated over a
  sphere at infinity (a simple form of the generalized Lorenz-Mie theory).
- dipole_force(): the force on a small sphere (Rayleigh regime) from its
  polarizability with radiation reaction: gradient force
  Re(alpha)/4*grad(abs(E)**2) plus the scattering and spin curl forces.
- polarizability(): Clausius-Mossotti with the radiative correction.
- radiation_pressure_efficiency(): Q_pr = Q_ext - g Q_sca of a sphere in a
  plane wave (Mie theory), the limit of a weakly focused beam.

Helpers: equilibrium(), stiffness() (trap position and spring constants from
any force function), brownian_trajectory() and lorentzian_psd() (Brownian
motion of a trapped bead and its power spectrum, for the calibration of a
trap by its corner frequency).

References
----------
A. Ashkin, J. M. Dziedzic, J. E. Bjorkholm and S. Chu, "Observation of a
single-beam gradient force optical trap for dielectric particles", Opt.
Lett. 11, 288-290 (1986).

A. Ashkin, "Forces of a single-beam gradient laser trap on a dielectric
sphere in the ray optics regime", Biophys. J. 61, 569-582 (1992).

B. Richards and E. Wolf, "Electromagnetic diffraction in optical systems II.
Structure of the image field in an aplanatic system", Proc. R. Soc. Lond. A
253, 358-379 (1959).

C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light by
Small Particles", Wiley (1983), chapter 4.

G. Gouesbet and G. Grehan, "Generalized Lorenz-Mie Theories", 2nd ed.,
Springer (2017).

P. C. Chaumet and M. Nieto-Vesperinas, "Time-averaged total force on a
dipolar sphere in an electromagnetic field", Opt. Lett. 25, 1065-1067
(2000).

T. A. Nieminen, V. L. Y. Loke, A. B. Stilgoe, G. Knöner, A. M. Brańczyk,
N. R. Heckenberg and H. Rubinsztein-Dunlop, "Optical tweezers computational
toolbox", J. Opt. A 9, S196-S203 (2007).

K. Berg-Sørensen and H. Flyvbjerg, "Power spectrum analysis for optical
tweezers", Rev. Sci. Instrum. 75, 594-612 (2004).
"""

import numpy as np

from .analytical import mie_amplitudes

try:
    import numba as _nb
except ImportError:  # pragma: no cover - depends on the installation
    _nb = None

# MieTrap sums the scattered far field with a compiled kernel if numba is
# installed (pip install pyoptics[fast]), about ten times faster
use_numba = True


__all__ = ["ashkin_efficiencies", "traced_efficiencies", "ray_optics_force",
           "focus_plane_waves", "focus_field", "beam_power", "MieTrap", "polarizability",
           "radiation_pressure_efficiency",
           "dipole_force", "equilibrium", "stiffness", "brownian_trajectory",
           "lorentzian_psd", "C_LIGHT", "K_B"]


C_LIGHT = 299792458.0
K_B = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _fresnel(theta_i, n_rel):
    """Power reflectance and transmittance of an interface (s and p)."""

    theta_i = np.asarray(theta_i, dtype=float)
    ci = np.cos(theta_i)
    st = np.sin(theta_i)/n_rel
    tir = st >= 1.0
    ct = np.sqrt(np.clip(1 - st**2, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (ci - n_rel*ct)/(ci + n_rel*ct)
        rp = (n_rel*ci - ct)/(n_rel*ci + ct)
    Rs = np.where(tir, 1.0, rs**2)
    Rp = np.where(tir, 1.0, rp**2)

    return Rs, Rp, np.arcsin(np.clip(st, -1, 1))


def ashkin_efficiencies(theta_i, n_rel, polarization="s"):
    """Scattering and gradient efficiency of one ray hitting a sphere.

    Parameters
    ----------
    theta_i : array
        Angle of incidence on the sphere (radians, 0 to pi/2).
    n_rel : number
        Index of the sphere over that of the medium (real).
    polarization : {"s", "p", "circular"}, optional
        Polarization relative to the plane of incidence; "circular" averages
        s and p.

    Returns
    -------
    Q_s : array
        Efficiency along the ray (pushing the sphere forward).
    Q_g : array
        Efficiency perpendicular to the ray, in the plane of incidence;
        positive values pull the sphere towards the ray (towards higher
        intensity), negative ones push it away.

    Notes
    -----
    Ashkin [1]_, eqs. (1) and (2), with the reflectance R, transmittance
    T = 1 - R and refraction angle r of the first surface:

        Q_s = 1 + R cos 2θ - T² (cos(2θ - 2r) + R cos 2θ)/(1 + R² + 2R cos 2r),
        Q_g = -(R sin 2θ - T² (sin(2θ - 2r) + R sin 2θ)/(1 + R² + 2R cos 2r)).

    Ashkin's Q_g points away from the ray; here its sign is flipped so that
    a positive value attracts.

    References
    ----------
    .. [1] A. Ashkin, "Forces of a single-beam gradient laser trap on a
       dielectric sphere in the ray optics regime", Biophys. J. 61, 569-582
       (1992).
    """

    if polarization == "circular":
        qs1, qg1 = ashkin_efficiencies(theta_i, n_rel, "s")
        qs2, qg2 = ashkin_efficiencies(theta_i, n_rel, "p")
        return (qs1 + qs2)/2, (qg1 + qg2)/2
    theta_i = np.asarray(theta_i, dtype=float)
    Rs, Rp, r = _fresnel(theta_i, n_rel)
    R = Rs if polarization == "s" else Rp
    T = 1 - R
    den = 1 + R**2 + 2*R*np.cos(2*r)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_s = 1 + R*np.cos(2*theta_i) - T**2*(np.cos(2*theta_i - 2*r) + R*np.cos(2*theta_i))/den
        q_g = R*np.sin(2*theta_i) - T**2*(np.sin(2*theta_i - 2*r) + R*np.sin(2*theta_i))/den
    # total internal reflection (n_rel < 1): the ray is simply reflected
    tir = R >= 1.0
    q_s = np.where(tir, 1 + np.cos(2*theta_i), q_s)
    q_g = np.where(tir, np.sin(2*theta_i), q_g)

    return q_s, -q_g


def traced_efficiencies(theta_i, n_rel, polarization="s", num_reflections=60):
    """Q_s and Q_g of one ray by explicitly following its internal
    reflections inside the sphere (a check of ashkin_efficiencies()).

    The ray travels along +x in the plane of incidence and hits a unit circle
    at the angle of incidence theta_i above its centre. At every hit the
    power splits by the Fresnel coefficients; the force is the momentum flux
    of the incident ray minus that of all rays that leave the sphere.

    Parameters
    ----------
    theta_i : number
        Angle of incidence (radians).
    n_rel : number
        Index of the sphere over that of the medium.
    polarization : {"s", "p"}, optional
    num_reflections : int, optional
        Number of internal reflections followed.

    Returns
    -------
    Q_s, Q_g : floats
        As ashkin_efficiencies().
    """

    Rs, Rp, r = _fresnel(theta_i, n_rel)
    R = float(Rs if polarization == "s" else Rp)
    d_in = np.array([1.0, 0.0])
    p = np.array([-np.cos(theta_i), np.sin(theta_i)])      # hit point (outward normal)
    F = d_in.copy()
    F -= R*(d_in - 2*np.dot(d_in, p)*p)                     # reflected at the entrance
    power = 1 - R
    # refracted direction: rotate the inward normal -p by r towards d_in
    t = -p*np.cos(r) + (d_in - np.dot(d_in, p)*p)/np.sin(theta_i)*np.sin(r) if theta_i > 0 else d_in
    pos = p
    for _ in range(num_reflections + 1):
        pos = pos - 2*np.dot(pos, t)*t                      # next hit on the unit circle
        pos /= np.linalg.norm(pos)                          # (reflections amplify round-off)
        n_out = pos
        tang = t - np.dot(t, n_out)*n_out
        tang_norm = np.linalg.norm(tang)
        # leaving ray: refracted back into the medium at the angle theta_i
        d_out = n_out*np.cos(theta_i) + (tang/tang_norm*np.sin(theta_i) if tang_norm > 0 else 0)
        F -= power*(1 - R)*d_out
        power *= R
        t = t - 2*np.dot(t, n_out)*n_out                    # internally reflected
        t /= np.linalg.norm(t)

    return F[0], F[1]                                       # y: from the centre towards the ray


def _pupil_amplitude(sin_theta, sin_max, filling):
    """Field in the pupil of a Gaussian beam with waist filling*pupil radius
    (filling None: uniform)."""

    if filling is None:
        return np.ones_like(sin_theta)

    return np.exp(-(sin_theta/(filling*sin_max))**2)


def _lens_polarization(theta, phi, polarization):
    """Field directions behind an aplanatic lens for an input polarization."""

    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    e_th = np.stack([ct*cp, ct*sp, -st], axis=-1)
    e_ph = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    if polarization == "x":
        e = cp[..., None]*e_th - sp[..., None]*e_ph
    elif polarization == "y":
        e = sp[..., None]*e_th + cp[..., None]*e_ph
    elif polarization in ("circular", "right", "left"):
        s = -1 if polarization == "left" else 1
        e = ((cp + 1j*s*sp)[..., None]*e_th + (-sp + 1j*s*cp)[..., None]*e_ph)/np.sqrt(2)
    elif polarization == "radial":
        e = e_th.astype(complex)
    elif polarization == "azimuthal":
        e = e_ph.astype(complex)
    else:
        raise ValueError("unknown polarization '{}'".format(polarization))

    return e.astype(complex)


def focus_plane_waves(NA, n_medium=1.33, num_theta=24, num_phi=64, filling=1.0, polarization="x",
                      theta_range=None):
    """The focus of an aplanatic objective as a quadrature of plane waves.

    The field near the focus is the Richards-Wolf (Debye) integral ::

        E(r) = sum_j weights_j * fields_j * exp(i k directions_j . r),

    a Gauss-Legendre quadrature in cos(theta) over (cos(theta_max), 1) and
    a uniform one in the azimuth phi. The amplitude of each plane wave is
    that of the pupil (a Gaussian beam of waist filling*pupil radius, with
    the sine condition rho = f sin(theta)) times the aplanatic factor
    sqrt(cos(theta)); the polarization is rotated by the lens (meridional
    component along e_theta, sagittal along e_phi).

    Parameters
    ----------
    NA : number
        Numerical aperture n_medium*sin(theta_max) (< n_medium).
    n_medium : number, optional
        Refractive index of the medium around the focus.
    num_theta, num_phi : int, optional
        Number of quadrature nodes in cos(theta) and phi.
    filling : number or None, optional
        Gaussian beam radius (1/e² intensity) over the pupil radius;
        None for uniform illumination.
    polarization : {"x", "y", "circular", "left", "radial", "azimuthal"}, optional
        Polarization in the pupil ("circular" is right-handed).
    theta_range : tuple, optional
        (theta_min, theta_max) to override (0, arcsin(NA/n_medium)), e.g.
        for an annular pupil.

    Returns
    -------
    directions : array (N, 3)
        Unit wave vectors.
    fields : complex array (N, 3)
        Unit polarization vectors (orthogonal to the directions).
    amplitudes : array (N,)
        Pupil amplitude times sqrt(cos theta).
    solid_angles : array (N,)
        Quadrature weights (solid angle per plane wave).

    Notes
    -----
    The weights of the superposition are amplitudes*solid_angles. The beam
    power is n_medium/(2 Z0)*(2 pi/k)**2*sum(abs(amplitudes)**2*solid_angles)
    for E in V/m (from the outgoing far field of the focus).
    """

    th_max = np.arcsin(NA/n_medium)
    th_min = 0.0
    if theta_range is not None:
        th_min, th_max = theta_range
    u, wu = np.polynomial.legendre.leggauss(num_theta)
    c0, c1 = np.cos(th_max), np.cos(th_min)
    cos_t = (c1 - c0)/2*u + (c1 + c0)/2
    w_t = (c1 - c0)/2*wu
    phi = (np.arange(num_phi) + 0.5)*2*np.pi/num_phi
    CT, PHI = np.meshgrid(cos_t, phi, indexing="ij")
    W = np.repeat(w_t[:, None], num_phi, axis=1)*2*np.pi/num_phi
    TH = np.arccos(CT)
    ST = np.sin(TH)
    directions = np.stack([ST*np.cos(PHI), ST*np.sin(PHI), CT], axis=-1).reshape(-1, 3)
    fields = _lens_polarization(TH, PHI, polarization).reshape(-1, 3)
    amplitudes = (_pupil_amplitude(ST, np.sin(np.arcsin(NA/n_medium)), filling)*np.sqrt(CT)).ravel()

    return directions, fields, amplitudes, W.ravel()


def focus_field(points, wavelength, plane_waves, n_medium=1.33, gradient=False):
    """Electric and magnetic field of a plane wave superposition.

    Parameters
    ----------
    points : array (..., 3)
        Positions (same unit as the wavelength).
    wavelength : number
        Vacuum wavelength.
    plane_waves : tuple
        (directions, fields, amplitudes, solid_angles) from
        focus_plane_waves().
    n_medium : number, optional
    gradient : bool, optional
        Also return the derivatives dE_i/dx_l.

    Returns
    -------
    E : complex array (..., 3)
    H : complex array (..., 3)
        Z0*H, i.e. in the units of E (n_medium times d x E for each plane
        wave).
    dE : complex array (..., 3, 3), only if gradient
        dE[..., i, l] = dE_i/dx_l.
    """

    d, e, a, w = plane_waves
    k = 2*np.pi*n_medium/wavelength
    pts = np.asarray(points, dtype=float)
    shape = pts.shape[:-1]
    pts = pts.reshape(-1, 3)
    ph = np.exp(1j*k*pts @ d.T)*(a*w)                   # (M, N)
    E = ph @ e
    h = n_medium*np.cross(d, e)
    H = ph @ h
    if not gradient:
        return E.reshape(shape + (3,)), H.reshape(shape + (3,))
    dE = 1j*k*np.einsum("mj,ji,jl->mil", ph, e, d)

    return E.reshape(shape + (3,)), H.reshape(shape + (3,)), dE.reshape(shape + (3, 3))


def beam_power(plane_waves, wavelength, n_medium=1.33):
    """Power (W) of a plane wave superposition from focus_plane_waves() for
    E in V/m (lengths in m), from its outgoing far field.

    Parameters
    ----------
    plane_waves : tuple
        From focus_plane_waves().
    wavelength : number
        Vacuum wavelength (m).
    n_medium : number, optional

    Returns
    -------
    P : float
    """

    _, _, a, w = plane_waves
    k = 2*np.pi*n_medium/wavelength
    Z0 = 1/(_EPS0*C_LIGHT)

    return n_medium/(2*Z0)*(2*np.pi/k)**2*np.sum(np.abs(a)**2*w)


def polarizability(radius, n_sph, wavelength, n_medium=1.33, radiative=True):
    """Polarizability alpha (p = alpha*E, SI, C m²/V) of a small sphere.

    Clausius-Mossotti alpha_0 = 4 pi eps0 n_m² a³ (m² - 1)/(m² + 2), with
    the radiative correction alpha = alpha_0/(1 - i k³ alpha_0/(6 pi eps0
    n_m²)) that makes the dipole scatter (and so lose momentum) consistently.

    Parameters
    ----------
    radius : number
        Sphere radius (m).
    n_sph : number
        Refractive index of the sphere.
    wavelength : number
        Vacuum wavelength (m).
    n_medium : number, optional
    radiative : bool, optional
        Apply the radiative correction.

    Returns
    -------
    alpha : complex
    """

    m = n_sph/n_medium
    a0 = 4*np.pi*_EPS0*n_medium**2*radius**3*(m**2 - 1)/(m**2 + 2)
    if not radiative:
        return complex(a0)
    k = 2*np.pi*n_medium/wavelength

    return a0/(1 - 1j*k**3*a0/(6*np.pi*_EPS0*n_medium**2))


def radiation_pressure_efficiency(radius, n_sph, wavelength, n_medium=1.33):
    """Radiation pressure efficiency of a sphere in a plane wave.

    Q_pr = Q_ext - g Q_sca with the asymmetry parameter g = <cos Θ> of the
    scattered light (Bohren and Huffman, eq. 4.62 and 4.79): a sphere in
    a wide beam of intensity I feels the force Q_pr pi a² n_medium I/c.

    Parameters
    ----------
    radius, n_sph, wavelength, n_medium :
        See analytical.mie_coefficients().

    Returns
    -------
    Q_pr : float
    """

    from .analytical import mie_coefficients

    n, a, b, _, _ = mie_coefficients(radius, n_sph, wavelength, n_medium)
    x = 2*np.pi*n_medium/wavelength*radius
    q_ext = 2/x**2*np.sum((2*n + 1)*np.real(a + b))
    a1, b1 = np.r_[a[1:], 0], np.r_[b[1:], 0]
    g_q_sca = 4/x**2*np.sum(n*(n + 2)/(n + 1)*np.real(a*np.conj(a1) + b*np.conj(b1))
                            + (2*n + 1)/(n*(n + 1))*np.real(a*np.conj(b)))

    return q_ext - g_q_sca


def dipole_force(r0, radius, n_sph, wavelength, plane_waves, n_medium=1.33, parts=False):
    """Force on a small (Rayleigh) sphere at r0 as a trapping efficiency.

    F_l = 1/2 Re(sum_i alpha E_i conj(dE_i/dx_l)) for the dipole p = alpha E
    induced by the focused field; with Re(alpha) this is the gradient force
    Re(alpha)/4 grad(abs(E)**2), with Im(alpha) the scattering force (along
    the local Poynting vector and the curl of the spin density).

    Parameters
    ----------
    r0 : array (..., 3)
        Positions of the sphere's centre (m).
    radius, n_sph : numbers
        Sphere radius (m) and index.
    wavelength : number
        Vacuum wavelength (m).
    plane_waves : tuple
        From focus_plane_waves().
    n_medium : number, optional
    parts : bool, optional
        Also return the gradient and the non-conservative (Im alpha) parts.

    Returns
    -------
    Q : array (..., 3)
        F*c/(n_medium*P).
    Q_grad, Q_scat : arrays (..., 3), only if parts
    """

    alpha = polarizability(radius, n_sph, wavelength, n_medium)
    E, _, dE = focus_field(r0, wavelength, plane_waves, n_medium, gradient=True)
    s = np.einsum("...i,...il->...l", E, np.conj(dE))
    P = beam_power(plane_waves, wavelength, n_medium)
    unit = C_LIGHT/(n_medium*P)
    Q_grad = 0.5*np.real(alpha)*np.real(s)*unit
    Q_scat = -0.5*np.imag(alpha)*np.imag(s)*unit
    if parts:
        return Q_grad + Q_scat, Q_grad, Q_scat

    return Q_grad + Q_scat


class MieTrap:
    """Exact force on a homogeneous sphere in a focused beam.

    The focus is the plane wave superposition of focus_plane_waves(). Every
    plane wave j (direction d_j, field e_j, weight w_j) is scattered by the
    sphere at r0 into the far field ::

        E_s = exp(i k r)/(-i k r) * w_j exp(i k (d_j - u).r0)
              * [S2(Θ) (e_j.e_par) e_theta + S1(Θ) (e_j.e_phi) e_phi]

    in the direction u, with the Mie amplitudes S1, S2 of the scattering
    angle Θ between d_j and u (Bohren and Huffman's convention). The incident
    focus leaves the focal region as the outgoing wave
    -i (2 pi/k) a(u) e(u) exp(i k r)/r within the cone of the objective. Far
    away, the time averaged Maxwell stress tensor is a momentum flux
    n_m abs(E)²/(2 Z) along u; the incoming part of the beam is the same with
    or without the sphere, so the force is the outgoing momentum flux lost:

        Q = -int (2 Re(E_inc* . E_s) + abs(E_s)²) u dΩ / int abs(E_inc)² dΩ.

    The far field is sampled on the pupil grid inside the cone (where the
    incident wave is) and on a Gauss-Legendre grid elsewhere.

    Parameters
    ----------
    radius : number
        Sphere radius (same unit as the wavelength).
    n_sph : number
        Refractive index of the sphere (real or complex).
    wavelength : number
        Vacuum wavelength.
    NA : number
        Numerical aperture of the objective (in the medium).
    n_medium : number, optional
    filling : number or None, optional
        See focus_plane_waves().
    polarization : str, optional
        See focus_plane_waves().
    num_theta, num_phi : int, optional
        Pupil quadrature. Defaults scale with k*(radius + max_offset).
    max_offset : number, optional
        Largest distance of the sphere from the focus the grids are made
        for. Defaults to max(radius, 2 wavelengths).
    num_theta_out, num_phi_out : int, optional
        Far field grid outside the cone.

    Attributes
    ----------
    plane_waves : tuple
        The focus, from focus_plane_waves().
    directions_far, solid_angles_far : arrays (M, 3), (M,)
        Far field directions u and quadrature weights (the first
        num_plane_waves of them are the pupil directions).
    incident_far_field : complex array (M, 3)
        Far field amplitude F of the focused beam (E = F exp(i k r)/r).
    """

    def __init__(self, radius, n_sph, wavelength, NA, n_medium=1.33, filling=1.0,
                 polarization="x", num_theta=None, num_phi=None, max_offset=None,
                 num_theta_out=None, num_phi_out=None):
        self.radius, self.n_sph, self.wavelength = radius, n_sph, wavelength
        self.NA, self.n_medium = NA, n_medium
        self.k = k = 2*np.pi*n_medium/wavelength
        if max_offset is None:
            max_offset = max(radius, 2*wavelength/n_medium)
        kR = k*(radius + max_offset)
        th_max = np.arcsin(NA/n_medium)
        if num_theta is None:
            num_theta = int(np.ceil(0.6*kR*th_max)) + 12
        if num_phi is None:
            num_phi = 2*int(np.ceil(0.6*kR*np.sin(th_max))) + 24
        if num_theta_out is None:
            num_theta_out = int(np.ceil(0.6*kR*(np.pi - th_max))) + 16
        if num_phi_out is None:
            num_phi_out = 2*int(np.ceil(0.6*kR)) + 24
        self.plane_waves = focus_plane_waves(NA, n_medium, num_theta, num_phi, filling, polarization)
        d, e, a, w = self.plane_waves
        self._d, self._e, self._w = d, e, a*w
        # far field grid: the pupil grid (inside the cone) and the rest of the sphere
        u, wu = np.polynomial.legendre.leggauss(num_theta_out)
        c0, c1 = -1.0, np.cos(th_max)
        ct = (c1 - c0)/2*u + (c1 + c0)/2
        phi = (np.arange(num_phi_out) + 0.5 + 0.1234)*2*np.pi/num_phi_out
        CT, PHI = np.meshgrid(ct, phi, indexing="ij")
        ST = np.sqrt(1 - CT**2)
        u_out = np.stack([ST*np.cos(PHI), ST*np.sin(PHI), CT], axis=-1).reshape(-1, 3)
        w_out = (np.repeat(((c1 - c0)/2*wu)[:, None], num_phi_out, axis=1)*2*np.pi/num_phi_out).ravel()
        self.directions_far = np.vstack([d, u_out])
        self.solid_angles_far = np.concatenate([w, w_out])
        F_inc = np.zeros((len(self.directions_far), 3), dtype=complex)
        F_inc[:len(d)] = -1j*(2*np.pi/k)*a[:, None]*e
        self.incident_far_field = F_inc
        self._norm = np.sum(self.solid_angles_far*np.sum(np.abs(F_inc)**2, axis=1))
        # Mie amplitudes tabulated against the scattering angle
        num_tab = int(60*(k*radius + 10))
        self._theta_tab = np.linspace(0, np.pi, num_tab)
        self._S1, self._S2 = mie_amplitudes(self._theta_tab, radius, n_sph, wavelength, n_medium)
        self.num_plane_waves, self.num_far = len(d), len(self.directions_far)

    def _amplitudes(self, mu):
        th = np.arccos(np.clip(mu, -1, 1))
        S1 = np.interp(th, self._theta_tab, self._S1.real) + 1j*np.interp(th, self._theta_tab, self._S1.imag)
        S2 = np.interp(th, self._theta_tab, self._S2.real) + 1j*np.interp(th, self._theta_tab, self._S2.imag)
        return S1, S2

    def scattered_far_field(self, r0, chunk=None):
        """Far field amplitude F (E_s = F exp(i k r)/r) of the scattered
        wave on the far field grid, for the sphere at r0."""

        d, e, wj = self._d, self._e, self._w
        k = self.k
        r0 = np.asarray(r0, dtype=float)
        if use_numba and _kernel is not None:
            src = wj*np.exp(1j*k*d @ r0)
            ph_u = np.exp(-1j*k*self.directions_far @ r0)/(-1j*k)
            return _kernel(self.directions_far, d, e, src, ph_u, self._theta_tab[1], self._S1, self._S2)
        if chunk is None:
            chunk = max(1, 3_000_000//len(d))
        out = np.empty((len(self.directions_far), 3), dtype=complex)
        src = wj*np.exp(1j*k*d @ r0)                          # incident amplitude at the centre
        for s in range(0, len(self.directions_far), chunk):
            u = self.directions_far[s:s + chunk]                          # (M, 3)
            mu = u @ d.T                                       # (M, N)
            S1, S2 = self._amplitudes(mu)
            # with e_par = (u - mu d)/sin, e_theta = (mu u - d)/sin, e_j.d = 0 and
            # c = (u.e_j)/sin²: (e_j.e_par) e_theta = c (mu u - d) and
            # (e_j.e_phi) e_phi = e_j - c (u - mu d). Exactly forward (sin = 0)
            # the field is S(0) e_j (c = 0).
            sin2 = 1 - mu**2
            c = np.where(sin2 > 1e-20, (u @ e.T)/np.where(sin2 > 1e-20, sin2, 1), 0)
            phase = np.exp(-1j*k*u @ r0)[:, None]*src[None, :]
            A1, A2 = S1*phase, S2*phase
            term_u = np.sum(c*(A2*mu - A1), axis=1)[:, None]*u
            term_d = (c*(A1*mu - A2)) @ d
            term_e = A1 @ e
            out[s:s + chunk] = (term_u + term_d + term_e)/(-1j*k)
        return out

    def force(self, r0):
        """Trapping efficiency Q (3,) for the sphere centred at r0 (or an
        array (..., 3) of positions)."""

        r0 = np.asarray(r0, dtype=float)
        if r0.ndim > 1:
            return np.array([self.force(p) for p in r0.reshape(-1, 3)]).reshape(r0.shape)
        Fs = self.scattered_far_field(r0)
        flux = 2*np.real(np.sum(np.conj(self.incident_far_field)*Fs, axis=1)) + np.sum(np.abs(Fs)**2, axis=1)

        return -np.sum((self.solid_angles_far*flux)[:, None]*self.directions_far, axis=0)/self._norm

    def force_N(self, r0, power):
        """Force in newtons for the beam power (W)."""

        return self.force(r0)*self.n_medium*power/C_LIGHT


def _far_field_kernel(u, d, e, src, ph_u, dth, S1t, S2t):
    """Compiled version of the plane wave sum in MieTrap.scattered_far_field()."""

    M, N = u.shape[0], d.shape[0]
    out = np.zeros((M, 3), dtype=np.complex128)
    n_tab = S1t.shape[0]
    for m in _nb.prange(M):
        u0, u1, u2 = u[m, 0], u[m, 1], u[m, 2]
        acc_u = 0j
        a0 = 0j
        a1 = 0j
        a2 = 0j
        for j in range(N):
            mu = u0*d[j, 0] + u1*d[j, 1] + u2*d[j, 2]
            mu = min(1.0, max(-1.0, mu))
            x = np.arccos(mu)/dth
            i = min(int(x), n_tab - 2)
            f = x - i
            S1 = S1t[i]*(1 - f) + S1t[i + 1]*f
            S2 = S2t[i]*(1 - f) + S2t[i + 1]*f
            A = src[j]
            A1, A2 = A*S1, A*S2
            sin2 = 1.0 - mu*mu
            if sin2 > 1e-20:
                c = (u0*e[j, 0] + u1*e[j, 1] + u2*e[j, 2])/sin2
                acc_u += c*(A2*mu - A1)
                cd = c*(A1*mu - A2)
                a0 += cd*d[j, 0]
                a1 += cd*d[j, 1]
                a2 += cd*d[j, 2]
            a0 += A1*e[j, 0]
            a1 += A1*e[j, 1]
            a2 += A1*e[j, 2]
        out[m, 0] = (acc_u*u0 + a0)*ph_u[m]
        out[m, 1] = (acc_u*u1 + a1)*ph_u[m]
        out[m, 2] = (acc_u*u2 + a2)*ph_u[m]
    return out


_kernel = None if _nb is None else _nb.njit(parallel=True, cache=True)(_far_field_kernel)


def ray_optics_force(r0, radius, n_sph, NA, n_medium=1.33, filling=1.0, polarization="x",
                     num_theta=200, num_phi=180, per_ray=False):
    """Ashkin's ray optics force on a sphere at r0 in the focus of an
    aplanatic objective.

    Rays from the pupil (power per ray from the Gaussian pupil
    illumination, as in focus_plane_waves()) converge towards the
    geometric focus at the origin. A ray that hits the sphere is split by
    the Fresnel coefficients for its s and p components relative to its
    plane of incidence; ashkin_efficiencies() gives the force along the ray
    and towards it.

    Parameters
    ----------
    r0 : array (..., 3)
        Positions of the sphere's centre (same unit as radius).
    radius : number
    n_sph : number
        Index of the sphere.
    NA, n_medium, filling, polarization :
        See focus_plane_waves().
    num_theta, num_phi : int, optional
        Number of rays in theta (midpoint rule in theta) and phi.
    per_ray : bool, optional
        For a single position, also return the contribution of every ray.

    Returns
    -------
    Q : array (..., 3)
        Trapping efficiency.
    theta, Q_rays : arrays (num_theta*num_phi,), (num_theta*num_phi, 3), only if per_ray
        Pupil angle and force contribution of each ray (zero for rays that
        miss the sphere).
    """

    r0 = np.asarray(r0, dtype=float)
    th_max = np.arcsin(NA/n_medium)
    th = (np.arange(num_theta) + 0.5)*th_max/num_theta
    ph = (np.arange(num_phi) + 0.5)*2*np.pi/num_phi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    ST, CT = np.sin(TH), np.cos(TH)
    d = np.stack([ST*np.cos(PH), ST*np.sin(PH), CT], axis=-1).reshape(-1, 3)
    pol = _lens_polarization(TH, PH, polarization).reshape(-1, 3)
    w = (_pupil_amplitude(ST, np.sin(th_max), filling)**2*CT*ST).ravel()
    w /= w.sum()
    n_rel = n_sph/n_medium
    shape = r0.shape[:-1]
    out = np.zeros((int(np.prod(shape)), 3))
    for i, c in enumerate(r0.reshape(-1, 3)):
        dc = d @ c
        disc = dc**2 - (c @ c - radius**2)
        hit = disc > 0
        t = dc[hit] - np.sqrt(disc[hit])
        dh = d[hit]
        n_hat = (t[:, None]*dh - c)/radius
        cos_i = np.clip(-np.sum(n_hat*dh, axis=1), 0, 1)
        th_i = np.arccos(cos_i)
        perp = n_hat + cos_i[:, None]*dh                      # n - (n.d) d
        pn = np.linalg.norm(perp, axis=1)
        e_g = perp/np.where(pn > 0, pn, 1)[:, None]
        s_hat = np.cross(dh, n_hat)
        sn = np.linalg.norm(s_hat, axis=1)
        s_hat = s_hat/np.where(sn > 0, sn, 1)[:, None]
        f_s = np.where(sn > 0, np.abs(np.sum(pol[hit]*s_hat, axis=1))**2, 0.5)
        qs_s, qg_s = ashkin_efficiencies(th_i, n_rel, "s")
        qs_p, qg_p = ashkin_efficiencies(th_i, n_rel, "p")
        q_s = f_s*qs_s + (1 - f_s)*qs_p
        q_g = f_s*qg_s + (1 - f_s)*qg_p
        # e_g points from the centre towards the ray: the force on the sphere
        # is q_g along it (towards the ray for q_g > 0)
        q_rays = np.zeros((len(d), 3))
        q_rays[hit] = w[hit][:, None]*(q_s[:, None]*dh + q_g[:, None]*e_g)
        out[i] = q_rays.sum(axis=0)
    if per_ray:
        return out.reshape(shape + (3,)), TH.ravel(), q_rays

    return out.reshape(shape + (3,))


def equilibrium(force_z, z_lo, z_hi, num=41):
    """Axial equilibrium: the stable zero crossing (from + to -) of
    force_z(z) in (z_lo, z_hi), or None if there is none.

    Parameters
    ----------
    force_z : callable
        Axial force (or efficiency) against the position z on the axis.
    z_lo, z_hi : numbers
        Search interval.
    num : int, optional
        Number of samples used to bracket the root.

    Returns
    -------
    z_eq : float or None
    """

    from scipy.optimize import brentq

    zs = np.linspace(z_lo, z_hi, num)
    f = np.array([force_z(z) for z in zs])
    i = np.nonzero((f[:-1] > 0) & (f[1:] <= 0))[0]
    if len(i) == 0:
        return None
    i = i[0]

    return brentq(force_z, zs[i], zs[i + 1], xtol=1e-6*(z_hi - z_lo))


def stiffness(force, r_eq, step):
    """Spring constants -dF_l/dx_l (l = x, y, z) at r_eq by central
    differences.

    Parameters
    ----------
    force : callable
        Force (3,) against the position (3,).
    r_eq : array (3,)
        Equilibrium position.
    step : number
        Finite difference step.

    Returns
    -------
    kappa : array (3,)
    """

    r_eq = np.asarray(r_eq, dtype=float)
    kappa = np.empty(3)
    for l in range(3):
        dr = np.zeros(3)
        dr[l] = step
        kappa[l] = -(force(r_eq + dr)[l] - force(r_eq - dr)[l])/(2*step)

    return kappa


def brownian_trajectory(force, x0, gamma, temperature, dt, num_steps, rng=None):
    """Overdamped Langevin dynamics of a bead in a force field.

    x_{n+1} = x_n + force(x_n) dt/gamma + sqrt(2 k_B T dt/gamma) xi_n with
    standard normal xi_n (Euler-Maruyama).

    Parameters
    ----------
    force : callable
        Force (N) against the position array (m), vectorized over the last
        axis of shape (..., dim).
    x0 : array (dim,) or (num_beads, dim)
        Start position(s) (m).
    gamma : number
        Friction coefficient 6 pi eta a (kg/s).
    temperature : number
        Temperature (K).
    dt : number
        Time step (s), well below gamma/kappa.
    num_steps : int
    rng : numpy Generator, optional

    Returns
    -------
    x : array (num_steps + 1, ...)
        Trajectory.
    """

    rng = np.random.default_rng() if rng is None else rng
    x = np.empty((num_steps + 1,) + np.shape(x0))
    x[0] = x0
    sig = np.sqrt(2*K_B*temperature*dt/gamma)
    for n in range(num_steps):
        x[n + 1] = x[n] + force(x[n])*dt/gamma + sig*rng.standard_normal(np.shape(x0))

    return x


def lorentzian_psd(f, f_c, gamma, temperature):
    """One-sided power spectral density (m²/Hz) of the position of a trapped
    bead, k_B T/(pi² gamma (f_c² + f²)), with the corner frequency
    f_c = kappa/(2 pi gamma).

    Parameters
    ----------
    f : array
        Frequencies (Hz).
    f_c : number
        Corner frequency (Hz).
    gamma : number
        Friction coefficient (kg/s).
    temperature : number
        Temperature (K).

    Returns
    -------
    psd : array
    """

    return K_B*temperature/(np.pi**2*gamma*(f_c**2 + np.asarray(f)**2))
