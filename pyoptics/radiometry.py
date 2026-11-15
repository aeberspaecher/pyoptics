#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Radiometry and photometry: how much light, in watts, photons and lumens.

Radiometry measures light as power. A source sends the flux Phi (W) into
space; per solid angle that is the intensity I = dPhi/dOmega (W/sr), per
area of a surface the irradiance E = dPhi/dA (W/m^2, the exitance M for
light leaving a surface), and per projected area and solid angle the
radiance ::

    L = d^2 Phi/(dA cos(theta) dOmega)      (W m^-2 sr^-1).

Radiance is what an eye or a camera pixel measures of an extended
source, and it is the quantity that optics transports: along a ray in a
lossless medium L/n^2 stays constant (the brightness theorem, see
`illumination`). A Lambertian surface has the same radiance in all
directions; it emits the exitance M = pi L and the intensity
I(theta) = L A cos(theta).

Every quantity has a spectral version (per nm here), a photon version
(divide by h c/lambda) and a photometric version, weighted with the
spectral luminous efficiency of the eye: ::

    Phi_v = K_m int V(lambda) Phi_e,lambda d lambda,

with K_m = 683 lm/W for daylight (photopic) vision and K_m' = 1700 lm/W
with V'(lambda) for night (scotopic) vision. The photometric units are
the lumen (lm) for flux, the candela (cd = lm/sr) for intensity, the lux
(lx = lm/m^2) for illuminance and cd/m^2 for luminance. The SI defines
them through the candela: 683 lm/W exactly at 540 THz (555.17 nm in vacuum).

The functions here are

- black bodies: `planck_radiance()` (absolute spectral radiance in
  W m^-2 sr^-1 nm^-1 or photons), `blackbody_radiance()` (sigma T^4/pi),
  `wien_peak()`;
- photometry: `V_photopic()`, `V_scotopic()`, `luminous_flux()`,
  `luminous_efficacy()` (lm per W of radiation) and
  `blackbody_efficacy()`;
- photons: `photon_energy()`, `photon_flux()`;
- geometry: `solid_angle_cone()`, `projected_solid_angle_cone()`, the
  view factors between Lambertian surfaces `view_factor_point_disk()` and
  `view_factor_parallel_disks()`;
- the integrating sphere: `sphere_gain()` (the closed form) and
  `sphere_trace()` (Monte Carlo through the bounces);
- images: `working_f_number()`, `camera_irradiance()` (the camera
  equation), the exposure meter relations `exposure_value()` and
  `metered_exposure()`, and `photons_per_pixel()`.

V(lambda) and V'(lambda) are analytic fits (sums of two-sided Gaussians,
the photopic one is the CIE 1931 ybar of `colour.cie_xyz()`, after Wyman,
Sloan and Shirley). The CIE tabulates both functions at 1 nm (CIE
018:2019, ISO 23539/CIE S 010); the fits follow the tables to within
0.007 (photopic) and 0.006 (scotopic) of the peak and their integrals to
0.1 %, which is enough for efficacies to about 1 %. Use the tables for
calibration work.

Units: wavelengths in nm, lengths in metres unless stated, angles in
radians.

References
----------
R. W. Boyd, "Radiometry and the Detection of Optical Radiation", Wiley
(1983).

W. R. McCluney, "Introduction to Radiometry and Photometry", 2nd ed.,
Artech House (2014).

J. M. Palmer and B. G. Grant, "The Art of Radiometry", SPIE Press (2010).

CIE 018:2019, "The Basis of Physical Photometry", 3rd ed.

C. Wyman, P.-P. Sloan and P. Shirley, "Simple analytic approximations to
the CIE XYZ color matching functions", JCGT 2(2), 1-11 (2013).

M. W. Finkel, "View factors", in J. R. Howell, M. P. Mengüç, K. Daun and
R. Siegel, "Thermal Radiation Heat Transfer", 7th ed., CRC Press (2021),
appendix C (configuration factor catalogue).

ISO 2720:1974, "General purpose photographic exposure meters" (the meter
constants K and C).
"""

import numpy as np

from . import colour


__all__ = ["H", "C", "K_B", "SIGMA", "K_M", "K_M_SCOTOPIC", "SOLAR_CONSTANT",
           "planck_radiance", "blackbody_radiance", "wien_peak", "V_photopic", "V_scotopic",
           "luminous_flux", "luminous_efficacy", "blackbody_efficacy", "photon_energy",
           "photon_flux", "solid_angle_cone", "projected_solid_angle_cone",
           "view_factor_point_disk", "view_factor_parallel_disks", "sphere_gain", "SphereResult",
           "sphere_trace", "working_f_number", "camera_irradiance", "exposure_value",
           "metered_exposure", "photons_per_pixel"]


H = 6.62607015e-34  # J s
C = 299792458.0  # m/s
K_B = 1.380649e-23  # J/K
SIGMA = 5.670374419e-8  # W m^-2 K^-4, Stefan-Boltzmann
K_M = 683.002  # lm/W, peak photopic efficacy (683 lm/W exactly at 540 THz)
K_M_SCOTOPIC = 1700.06  # lm/W, peak scotopic efficacy
SOLAR_CONSTANT = 1361.0  # W/m^2, total solar irradiance at 1 AU


def _trapz(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


# black bodies:
def planck_radiance(wavelength_nm, T, photons=False):
    """Spectral radiance of a black body.

    Parameters
    ----------
    wavelength_nm : array
        Vacuum wavelengths in nm.
    T : number or array
        Temperature in K (broadcast against the wavelengths).
    photons : bool, optional
        Return the photon radiance (photons s^-1 m^-2 sr^-1 nm^-1) instead
        of the power.

    Returns
    -------
    L : array
        W m^-2 sr^-1 nm^-1 (or photons s^-1 m^-2 sr^-1 nm^-1).
    """

    lam = np.asarray(wavelength_nm, float)*1e-9
    T = np.asarray(T, float)
    with np.errstate(over="ignore"):
        x = H*C/(lam*K_B*T)
        L = 2*H*C**2/lam**5/np.expm1(x)*1e-9
    if photons:
        L = L/(H*C/lam)

    return L


def blackbody_radiance(T):
    """Total radiance sigma T^4/pi of a black body (W m^-2 sr^-1); its
    exitance is pi times this, sigma T^4 (Stefan-Boltzmann).

    Parameters
    ----------
    T : number or array
        Temperature in K.

    Returns
    -------
    L : number or array
    """

    return SIGMA*np.asarray(T, float)**4/np.pi


def wien_peak(T):
    """Wavelength (nm) at which the spectral radiance per wavelength of a
    black body peaks, b/T with b = 2.897771955e-3 m K.

    Parameters
    ----------
    T : number or array
        Temperature in K.

    Returns
    -------
    wavelength_nm : number or array
    """

    return 2.897771955e6/np.asarray(T, float)


# photometry:
def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)

    return np.exp(-0.5*(lam - mu)**2/s**2)


def V_photopic(wavelength_nm):
    """Spectral luminous efficiency V(lambda) of daylight vision (peak 1 near
    555 nm): the CIE 1931 ybar, which is V(lambda) by definition, from the
    analytic fit of `colour.cie_xyz()`.

    Parameters
    ----------
    wavelength_nm : array

    Returns
    -------
    V : array
    """

    return colour.cie_xyz(wavelength_nm)[1]


def V_scotopic(wavelength_nm):
    """Spectral luminous efficiency V'(lambda) of night (rod) vision, peak 1
    near 507 nm: a fit of two two-sided Gaussians to the CIE 1951 table
    (within 0.006 of it from 380 to 780 nm).

    Parameters
    ----------
    wavelength_nm : array

    Returns
    -------
    V : array
    """

    lam = np.asarray(wavelength_nm, float)

    return 0.770*_g(lam, 512.15, 31.52, 32.95) + 0.370*_g(lam, 457.1, 21.71, 54.26)


def luminous_flux(wavelength_nm, spectrum, scotopic=False, axis=-1):
    """Photometric quantity of a spectral radiometric one,
    K_m int V(lambda) S(lambda) d lambda: lumens from W/nm, lux from
    W m^-2 nm^-1, cd/m^2 from W m^-2 sr^-1 nm^-1.

    Parameters
    ----------
    wavelength_nm : 1d array
        Wavelengths in nm (the integration grid).
    spectrum : array
        Spectral quantity per nm along the given axis.
    scotopic : bool, optional
        Use night vision, K_m' V'(lambda), instead of K_m V(lambda).
    axis : int, optional

    Returns
    -------
    array
    """

    lam = np.asarray(wavelength_nm, float)
    S = np.moveaxis(np.asarray(spectrum, float), axis, -1)
    if scotopic:
        w = K_M_SCOTOPIC*V_scotopic(lam)
    else:
        w = K_M*V_photopic(lam)

    return _trapz(w*S, lam)


def luminous_efficacy(wavelength_nm, spectrum, scotopic=False, axis=-1):
    """Luminous efficacy of radiation (lm/W): luminous over radiant flux of
    a spectrum (683 lm/W at most, for light at 555 nm).

    Parameters
    ----------
    wavelength_nm : 1d array
    spectrum : array
        Spectral power (any units per nm) along the given axis.
    scotopic : bool, optional
    axis : int, optional

    Returns
    -------
    array
    """

    lam = np.asarray(wavelength_nm, float)
    S = np.moveaxis(np.asarray(spectrum, float), axis, -1)

    return luminous_flux(lam, S, scotopic) / _trapz(S, lam)


def blackbody_efficacy(T, scotopic=False, n=4001):
    """Luminous efficacy of black body radiation (lm/W of the total
    radiation, sigma T^4): about 15 lm/W for a tungsten lamp at 2850 K, at
    most 95 lm/W near 6600 K.

    Parameters
    ----------
    T : number or array
        Temperature in K.
    scotopic : bool, optional
    n : int, optional
        Points of the integration grid from 360 to 830 nm.

    Returns
    -------
    number or array
    """

    T = np.asarray(T, float)
    lam = np.linspace(360.0, 830.0, n)
    L = planck_radiance(lam, T[..., None])

    return luminous_flux(lam, L, scotopic)/blackbody_radiance(T)


# photons:
def photon_energy(wavelength_nm):
    """Energy h c/lambda of a photon in J.

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    number or array
    """

    return H*C/(np.asarray(wavelength_nm, float)*1e-9)


def photon_flux(power, wavelength_nm):
    """Photons per second in a power (W) at a wavelength; divide by the
    Avogadro number 6.022e23 for mol/s (PPFD is in umol m^-2 s^-1).

    Parameters
    ----------
    power : number or array
        In W (or W/m^2 for a photon irradiance).
    wavelength_nm : number or array

    Returns
    -------
    number or array
    """

    return np.asarray(power, float)/photon_energy(wavelength_nm)


# geometry:
def solid_angle_cone(half_angle):
    """Solid angle 2 pi (1 - cos theta) of a cone (sr).

    Parameters
    ----------
    half_angle : number or array
        In radians.

    Returns
    -------
    number or array
    """

    return 2*np.pi*(1 - np.cos(half_angle))


def projected_solid_angle_cone(half_angle):
    """Projected solid angle int cos(theta) dOmega = pi sin^2(theta) of a
    cone about the normal of a surface; a Lambertian surface of radiance L
    sends pi L sin^2(theta) per area into the cone (pi L into the
    hemisphere).

    Parameters
    ----------
    half_angle : number or array
        In radians.

    Returns
    -------
    number or array
    """

    return np.pi*np.sin(half_angle)**2


def view_factor_point_disk(radius, distance):
    """View factor from a small surface element to a parallel coaxial disk
    facing it, a^2/(a^2 + d^2): the fraction of the element's Lambertian
    emission that reaches the disk.

    Parameters
    ----------
    radius : number or array
        Radius a of the disk.
    distance : number or array
        Distance d between the element and the disk.

    Returns
    -------
    number or array
    """

    a2 = np.asarray(radius, float)**2

    return a2/(a2 + np.asarray(distance, float)**2)


def view_factor_parallel_disks(r1, r2, distance):
    """View factor F_12 from a disk of radius r1 to a parallel coaxial disk
    of radius r2 at the given distance: the fraction of the Lambertian
    emission of disk 1 that lands on disk 2, ::

        F_12 = (X - sqrt(X^2 - 4 (r2/r1)^2))/2,  X = 1 + (1 + R2^2)/R1^2,

    with R_i = r_i/distance. Reciprocity: A_1 F_12 = A_2 F_21.

    Parameters
    ----------
    r1, r2 : number or array
    distance : number or array

    Returns
    -------
    number or array
    """

    R1 = np.asarray(r1, float)/distance
    R2 = np.asarray(r2, float)/distance
    X = 1 + (1 + R2**2)/R1**2

    return 0.5*(X - np.sqrt(X**2 - 4*(R2/R1)**2))


# integrating sphere:
def sphere_gain(reflectance, port_fraction):
    """Sphere multiplier rho/(1 - rho (1 - f)) of an integrating sphere: the
    radiance of its wall is L = Phi/(pi A_s) times this, for the flux Phi
    that first hits the wall, the wall area A_s (ports included), the wall
    reflectance rho and the port fraction f (the ports' share of A_s).

    Parameters
    ----------
    reflectance : number or array
    port_fraction : number or array

    Returns
    -------
    number or array
    """

    rho = np.asarray(reflectance, float)

    return rho/(1 - rho*(1 - np.asarray(port_fraction, float)))


def _sphere_hit(P, d, radius):
    b = np.sum(P*d, axis=1)
    c = np.sum(P*P, axis=1) - radius**2
    t = -b + np.sqrt(np.clip(b*b - c, 0, None))
    Q = P + t[:, None]*d

    # back onto the sphere (no drift of the rounding errors over many bounces)
    return radius*Q/np.linalg.norm(Q, axis=1)[:, None]


def _lambert_about(normal, rng):
    """Lambertian directions about the unit normals (N, 3)."""

    n = len(normal)
    u = rng.random((n, 2))
    r = np.sqrt(u[:, 0])
    phi = 2*np.pi*u[:, 1]
    a, b, c = r*np.cos(phi), r*np.sin(phi), np.sqrt(np.clip(1 - u[:, 0], 0, None))
    # orthonormal frame (t1, t2, normal)
    helper = np.where(np.abs(normal[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(normal, t1)

    return a[:, None]*t1 + b[:, None]*t2 + c[:, None]*normal


class SphereResult(object):
    """Result of `sphere_trace()`.

    Attributes
    ----------
    port_power : (n_ports,) array
        Power that left through each port (first-pass escapes included).
    absorbed : float
        Power absorbed by the wall.
    lost : float
        Power still in the sphere after the last bounce.
    hits : list of ((N, 3) array, (N,) array)
        Wall hit points and the power that arrives there, for each bounce
        (index 0: the first hit), if requested.
    """

    def __init__(self, port_power, absorbed, lost, hits):
        self.port_power = port_power
        self.absorbed = absorbed
        self.lost = lost
        self.hits = hits


def sphere_trace(points, directions, radius, reflectance, ports=(), power=1.0, max_bounces=200,
                 keep_hits=0, rng=None):
    """Monte Carlo trace of light through an integrating sphere.

    Each ray goes from its point to the wall; inside a port it leaves the
    sphere, elsewhere it is reflected into Lambertian directions with its
    power multiplied by the reflectance (weights, no Russian roulette), up
    to max_bounces.

    Parameters
    ----------
    points : (N, 3) array
        Start points inside the sphere (centre at the origin).
    directions : (N, 3) array
        Unit directions.
    radius : number
    reflectance : number
        Diffuse reflectance of the wall.
    ports : sequence of (axis, half_angle)
        Circular ports as caps of the sphere: unit vector of the centre and
        the half angle seen from the centre of the sphere. The port fraction
        of a cap is (1 - cos(half_angle))/2.
    power : number or (N,) array, optional
        Power per ray.
    max_bounces : int, optional
    keep_hits : int, optional
        Keep the wall hits of the first keep_hits bounces.
    rng : None, int or numpy.random.Generator, optional

    Returns
    -------
    SphereResult
    """

    rng = np.random.default_rng(rng)
    P = np.asarray(points, float)
    D = np.asarray(directions, float)
    w = np.broadcast_to(np.asarray(power, float), (len(P),)).copy()
    axes = [np.asarray(a, float)/np.linalg.norm(a) for a, _ in ports]
    cosines = [np.cos(h) for _, h in ports]
    port_power = np.zeros(len(axes))
    absorbed = 0.0
    hits = []
    for bounce in range(max_bounces + 1):
        if len(P) == 0:
            break
        Q = _sphere_hit(P, D, radius)
        u = Q/radius
        out = np.zeros(len(Q), bool)
        for k, (a, c) in enumerate(zip(axes, cosines)):
            inside = (u @ a >= c) & ~out
            port_power[k] += w[inside].sum()
            out |= inside
        if bounce < keep_hits:
            hits.append((Q[~out], w[~out].copy()))
        Q, w, u = Q[~out], w[~out], u[~out]
        absorbed += (1 - reflectance)*w.sum()
        w = w*reflectance
        if bounce == max_bounces:
            break
        P = Q
        D = _lambert_about(-u, rng)
    lost = w.sum() if len(P) else 0.0

    return SphereResult(port_power, absorbed, lost, hits)


# images:
def working_f_number(f_number, magnification=0.0, pupil_magnification=1.0):
    """Working F-number N (1 + |m|/m_p) of a lens imaging at the
    magnification m (0 at infinity, -1 at 1:1), with the pupil magnification
    m_p (exit over entrance pupil diameter).

    Parameters
    ----------
    f_number : number or array
        F-number for an object at infinity, f/D.
    magnification : number or array, optional
    pupil_magnification : number or array, optional

    Returns
    -------
    number or array
    """

    return np.asarray(f_number, float)*(1 + np.abs(magnification)/pupil_magnification)


def camera_irradiance(radiance, f_number, magnification=0.0, pupil_magnification=1.0,
                      transmittance=1.0, field_angle=0.0, n_object=1.0, n_image=1.0):
    """Irradiance in the image of an extended Lambertian object (the camera
    equation), ::

        E' = pi T L (n'/n)^2 cos^4(theta) / (4 N_w^2),

    with the working F-number N_w (`working_f_number()`). On the axis this
    is the brightness theorem E' = pi T L' sin^2(theta'), with the image
    space NA sin(theta') = 1/(2 N_w) of an aplanatic lens and L' = (n'/n)^2
    L; cos^4 is the law for a lens without distortion, pupil aberration and
    vignetting.

    Parameters
    ----------
    radiance : number or array
        Object radiance L (W m^-2 sr^-1, or cd/m^2 for an illuminance in lx).
    f_number : number or array
        F-number for an object at infinity.
    magnification : number or array, optional
    pupil_magnification : number or array, optional
    transmittance : number or array, optional
        Transmittance T of the lens.
    field_angle : number or array, optional
        Field angle theta in radians (object space).
    n_object, n_image : numbers, optional
        Refractive indices of object and image space.

    Returns
    -------
    number or array
    """

    Nw = working_f_number(f_number, magnification, pupil_magnification)

    return (np.pi*transmittance*np.asarray(radiance, float)*(n_image/n_object)**2
            * np.cos(field_angle)**4/(4*Nw**2))


def exposure_value(f_number, exposure_time):
    """Exposure value EV = log2(N^2/t) (EV 15 is f/16 at 1/125 s).

    Parameters
    ----------
    f_number : number or array
    exposure_time : number or array
        In s.

    Returns
    -------
    number or array
    """

    return np.log2(np.asarray(f_number, float)**2/np.asarray(exposure_time, float))


def metered_exposure(luminance, iso, f_number, K=12.5):
    """Exposure time (s) that a reflected light meter sets for a scene of the
    given mean luminance (cd/m^2) at the ISO speed and F-number:
    N^2/t = L S/K, with the meter calibration constant K (ISO 2720,
    12.5 cd s m^-2 for most makers, 14 for some).

    Parameters
    ----------
    luminance : number or array
        In cd/m^2.
    iso : number or array
        Arithmetic ISO speed S.
    f_number : number or array
    K : number, optional

    Returns
    -------
    number or array
    """

    return np.asarray(f_number, float)**2*K/(np.asarray(luminance, float)*iso)


def photons_per_pixel(irradiance, wavelength_nm, pixel_pitch, exposure_time, fill=1.0):
    """Photons that reach a square pixel: E A t/(h c/lambda).

    Parameters
    ----------
    irradiance : number or array
        In W/m^2 (monochromatic at wavelength_nm).
    wavelength_nm : number or array
    pixel_pitch : number
        In m.
    exposure_time : number or array
        In s.
    fill : number, optional
        Fill factor (the share of the pixel area that collects light).

    Returns
    -------
    number or array
    """

    return photon_flux(np.asarray(irradiance, float)*fill*pixel_pitch**2*exposure_time,
                       wavelength_nm)
