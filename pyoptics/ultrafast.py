#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ultrashort pulses through glass, lenses, prisms and gratings.

A femtosecond pulse has a broad spectrum, and the optics it passes treat each
frequency differently. Three effects of first and second order in the
frequency matter for lenses and dispersive elements (notebook 81):

- Group delay dispersion (GDD): glass delays the frequencies by different
  amounts, a pulse is chirped and stretched. `group_index()`, `gvd()`,
  `tod()` from any dispersion formula n(wavelength_nm) (such as a
  `glasses.Glass`), `material_phase()` for the exact spectral phase of a
  slab and `gaussian_broadening()` for the closed form of a Gaussian pulse.
- Pulse-front delay in lenses (Bor's effect): the pulse travels with the
  group velocity, the phase fronts with the phase velocity. Behind a lens
  of normal dispersion the pulse front lags behind the phase front most on
  the axis, where the glass is thickest. `bor_delay()` is Bor's formula
  from the chromatic derivative of the power, `ray_delays()` sums phase and
  group delays (and the GDD) along rays traced through a `raytrace.System`.
- Pulse-front tilt from angular dispersion: behind a prism or a grating the
  pulse front is tilted against the phase fronts by tan(gamma) =
  lambda d(theta)/d(lambda) (`tilt_angle()`); `prism_ray_delays()` sums the
  group delays of parallel rays through a prism, `prism_dispersion()` and
  `grating_dispersion()` give d(theta)/d(lambda) in closed form.

Focal fields of pulses are sums of monochromatic focuses: `focal_spectrum()`
traces a lens at many wavelengths and evaluates the focal field of each
from the traced optical paths (a Debye type integral over the exit
surface), and `to_time()` sums the spectrum into a field E(r, t).
`beamlet_focus_1d()` is the exact paraxial focus of a 1D Gaussian beamlet,
the building block of simultaneous spatial and temporal focusing.

Units: lengths in mm, times in fs, vacuum wavelengths in nm, angular
frequencies in rad/fs; GVD in fs^2/mm, GDD in fs^2, TOD in fs^3. Fields
use the pyoptics convention exp(i(k z - omega t)), a spectrum E(omega)
gives E(t) = sum E(omega) exp(-i (omega - omega_0) t) d(omega)/(2 pi)
(the envelope relative to the carrier omega_0).

References
----------
Z. Bor, "Distortion of femtosecond laser pulses in lenses and lens
systems", J. Mod. Opt. 35, 1907-1918 (1988).

Z. Bor, "Distortion of femtosecond laser pulses in lenses", Opt. Lett. 14,
119-121 (1989).

J. Hebling, "Derivation of the pulse front tilt caused by angular
dispersion", Opt. Quantum Electron. 28, 1759-1763 (1996).

J.-C. Diels and W. Rudolph, "Ultrashort Laser Pulse Phenomena", 2nd ed.,
Academic Press (2006), chapters 1 and 2.

I. H. Malitson, "Interspecimen comparison of the refractive index of fused
silica", J. Opt. Soc. Am. 55, 1205-1209 (1965).

I. H. Malitson, "A redetermination of some optical properties of calcium
fluoride", Appl. Opt. 2, 1103-1107 (1963).
"""

from collections import namedtuple

import numpy as np
from scipy.special import j0

from .glasses import Glass

C_NM_PER_FS = 299.792458                  # speed of light in nm/fs
C_MM_PER_FS = C_NM_PER_FS*1e-6            # in mm/fs
NM_PER_MM = 1e6

#: Fused silica after Malitson (1965), 0.21 to 3.71 um.
FUSED_SILICA = Glass("fused silica", "Malitson", "sellmeier",
                     [0.0, 0.6961663, 0.0684043, 0.4079426, 0.1162414, 0.8974794, 9.896161],
                     range_um=(0.21, 3.71))
#: Calcium fluoride after Malitson (1963), 0.23 to 9.7 um.
CAF2 = Glass("CaF2", "Malitson", "sellmeier",
             [0.0, 0.5675888, 0.050263605, 0.4710914, 0.1003909, 3.8484723, 34.649040],
             range_um=(0.23, 9.7))


def omega_from_wavelength(wavelength_nm):
    """Angular frequency (rad/fs) of a vacuum wavelength.

    Parameters
    ----------
    wavelength_nm : number or array
        Vacuum wavelength in nm.

    Returns
    -------
    omega : number or array
        2 pi c / lambda in rad/fs.
    """

    return 2*np.pi*C_NM_PER_FS/np.asarray(wavelength_nm, float)


def wavelength_from_omega(omega):
    """Vacuum wavelength (nm) of an angular frequency.

    Parameters
    ----------
    omega : number or array
        Angular frequency in rad/fs.

    Returns
    -------
    wavelength_nm : number or array
    """

    return 2*np.pi*C_NM_PER_FS/np.asarray(omega, float)


# five point stencils (step h): derivatives 1 to 3
_STENCILS = {1: (np.array([1, -8, 0, 8, -1])/12.0, 1),
             2: (np.array([-1, 16, -30, 16, -1])/12.0, 2),
             3: (np.array([-1, 2, 0, -2, 1])/2.0, 3)}


def dn_dlambda(n, wavelength_nm, order=1, step_nm=None):
    """Derivative of the refractive index with respect to the wavelength.

    Parameters
    ----------
    n : callable
        n(wavelength_nm), e.g. a `glasses.Glass`.
    wavelength_nm : number or array
    order : {1, 2, 3}, optional
        Order of the derivative.
    step_nm : number, optional
        Step of the five point finite difference (default lambda/500).

    Returns
    -------
    d^order n / d lambda^order in 1/nm^order.
    """

    lam = np.asarray(wavelength_nm, float)
    h = lam/500.0 if step_nm is None else step_nm
    weights, p = _STENCILS[order]
    vals = [np.asarray(n(lam + o*h), float) for o in (-2, -1, 0, 1, 2)]

    return sum(w*v for w, v in zip(weights, vals))/h**p


def group_index(n, wavelength_nm, step_nm=None):
    """Group index n_g = n - lambda dn/dlambda.

    Parameters
    ----------
    n : callable
        n(wavelength_nm).
    wavelength_nm : number or array
    step_nm : number, optional
        See `dn_dlambda()`.

    Returns
    -------
    n_g : number or array
    """

    lam = np.asarray(wavelength_nm, float)

    return np.asarray(n(lam), float) - lam*dn_dlambda(n, lam, 1, step_nm)


def gvd(n, wavelength_nm, step_nm=None):
    """Group velocity dispersion d^2 k / d omega^2 = lambda^3/(2 pi c^2)
    d^2 n / d lambda^2.

    Parameters
    ----------
    n : callable
        n(wavelength_nm).
    wavelength_nm : number or array
    step_nm : number, optional
        See `dn_dlambda()`.

    Returns
    -------
    GVD in fs^2/mm (the GDD of 1 mm of the material).
    """

    lam = np.asarray(wavelength_nm, float)

    return lam**3/(2*np.pi*C_NM_PER_FS**2)*dn_dlambda(n, lam, 2, step_nm)*NM_PER_MM


def tod(n, wavelength_nm, step_nm=None):
    """Third order dispersion d^3 k / d omega^3 = -lambda^4/(4 pi^2 c^3)
    (3 n'' + lambda n''').

    Parameters
    ----------
    n : callable
        n(wavelength_nm).
    wavelength_nm : number or array
    step_nm : number, optional
        See `dn_dlambda()`.

    Returns
    -------
    TOD in fs^3/mm.
    """

    lam = np.asarray(wavelength_nm, float)
    d2 = dn_dlambda(n, lam, 2, step_nm)
    d3 = dn_dlambda(n, lam, 3, step_nm)

    return -lam**4/(4*np.pi**2*C_NM_PER_FS**3)*(3*d2 + lam*d3)*NM_PER_MM


def material_phase(n, omega, length_mm, omega0=None):
    """Spectral phase omega n(omega) L / c of a slab, optionally without its
    constant and linear parts at omega0 (the phase and group delay).

    Parameters
    ----------
    n : callable
        n(wavelength_nm).
    omega : array
        Angular frequencies in rad/fs.
    length_mm : number
        Thickness of the slab.
    omega0 : number, optional
        If given, the Taylor terms of order 0 and 1 about omega0 are removed,
        leaving the phase that changes the shape of the pulse.

    Returns
    -------
    phase : array (rad)
    """

    omega = np.asarray(omega, float)
    phase = omega*np.asarray(n(wavelength_from_omega(omega)), float)*length_mm/C_MM_PER_FS
    if omega0 is not None:
        lam0 = float(wavelength_from_omega(omega0))
        n0 = float(n(lam0))
        ng0 = float(group_index(n, lam0))
        phase = phase - omega0*n0*length_mm/C_MM_PER_FS - (omega - omega0)*ng0*length_mm/C_MM_PER_FS

    return phase


def gaussian_broadening(tau_fwhm, gdd):
    """FWHM of an initially unchirped Gaussian pulse after the GDD:
    tau sqrt(1 + (4 ln 2 GDD/tau^2)^2).

    Parameters
    ----------
    tau_fwhm : number or array
        Input intensity FWHM in fs.
    gdd : number or array
        Group delay dispersion in fs^2.

    Returns
    -------
    tau_out : number or array (fs)
    """

    tau = np.asarray(tau_fwhm, float)

    return tau*np.sqrt(1 + (4*np.log(2)*np.asarray(gdd, float)/tau**2)**2)


def optimal_input_duration(gdd):
    """Input FWHM sqrt(4 ln 2 abs(GDD)) of a Gaussian pulse that comes out
    shortest after the GDD (it comes out sqrt(2) times longer).

    Parameters
    ----------
    gdd : number or array
        Group delay dispersion in fs^2.

    Returns
    -------
    tau_fwhm : number or array (fs)
    """

    return np.sqrt(4*np.log(2)*np.abs(np.asarray(gdd, float)))


def gaussian_spectrum(omega, omega0, tau_fwhm, gdd=0.0):
    """Spectral amplitude of a Gaussian pulse (intensity FWHM tau_fwhm when
    unchirped), optionally with a quadratic spectral phase GDD/2 (omega -
    omega0)^2.

    Normalized such that the unchirped pulse has the peak envelope 1 in
    `to_time()`.

    Parameters
    ----------
    omega : array
        Angular frequencies (rad/fs).
    omega0 : number
        Carrier frequency.
    tau_fwhm : number
        Transform-limited intensity FWHM in fs.
    gdd : number, optional
        Group delay dispersion in fs^2.

    Returns
    -------
    E(omega) : complex array
    """

    a = 2*np.log(2)/tau_fwhm**2          # E(t) = exp(-a t^2)
    w = np.asarray(omega, float) - omega0

    return np.sqrt(np.pi/a)*np.exp(-w**2/(4*a) + 0.5j*gdd*w**2)


def pulse_grid(wavelength_nm, t_window, n):
    """Matching time and frequency grids for FFT-free pulse synthesis.

    Parameters
    ----------
    wavelength_nm : number
        Centre wavelength.
    t_window : number
        Width of the time window (fs); the frequency spacing is 2 pi/t_window.
    n : int
        Number of samples of both grids.

    Returns
    -------
    t : array (fs), centred on 0
    omega : array (rad/fs), centred on omega_0
    """

    omega0 = float(omega_from_wavelength(wavelength_nm))
    dt = t_window/n
    t = (np.arange(n) - n//2)*dt
    dw = 2*np.pi/t_window
    omega = omega0 + (np.arange(n) - n//2)*dw

    return t, omega


def to_time(E_omega, omega, t, omega0=None, axis=0):
    """Envelope E(t) = sum E(omega) exp(-i (omega - omega0) t) d(omega)/(2 pi).

    A direct sum (no FFT), so the grids are free; the frequency spacing must
    be uniform.

    Parameters
    ----------
    E_omega : array
        Spectrum, frequencies along axis.
    omega : array
        Uniformly spaced angular frequencies (rad/fs).
    t : array
        Times (fs).
    omega0 : number, optional
        Carrier (default: the centre of omega).
    axis : int, optional
        Frequency axis of E_omega; replaced by the time axis in the result.

    Returns
    -------
    E(t) : complex array
    """

    omega = np.asarray(omega, float)
    w0 = omega[len(omega)//2] if omega0 is None else omega0
    dw = omega[1] - omega[0]
    K = np.exp(-1j*np.outer(np.asarray(t, float), omega - w0))*dw/(2*np.pi)
    E = np.moveaxis(np.asarray(E_omega), axis, 0)
    out = np.tensordot(K, E, axes=(1, 0))

    return np.moveaxis(out, 0, axis)


def fwhm(x, y):
    """Full width at half maximum between the outermost half maximum
    crossings (linear interpolation).

    Parameters
    ----------
    x : array
        Uniform or increasing coordinates.
    y : array
        Non-negative profile.

    Returns
    -------
    width : number
    """

    x, y = np.asarray(x, float), np.asarray(y, float)
    half = y.max()/2
    above = np.nonzero(y >= half)[0]
    i0, i1 = above[0], above[-1]
    if i0 == 0 or i1 == len(y) - 1:
        return np.nan
    xl = np.interp(half, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    xr = np.interp(half, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])

    return xr - xl


# ---------------------------------------------------------------- lenses

def bor_delay(r, wavelength_nm, dpower_dlambda):
    """Bor's pulse-front delay of a lens: the pulse at the height r leads the
    pulse on the axis by -r^2/(2c) lambda dP/dlambda.

    For a thin singlet of index n and focal length f, dP/dlambda =
    (dn/dlambda)/((n - 1) f) (`thin_lens_power_derivative()`); for any
    system, the derivative of its paraxial power. An achromat (dP/dlambda =
    0) has no pulse-front delay to this order.

    Parameters
    ----------
    r : number or array
        Ray height in the entrance pupil (mm).
    wavelength_nm : number
    dpower_dlambda : number
        Derivative of the power 1/f in 1/(mm nm).

    Returns
    -------
    delay : number or array (fs)
        Lag of the pulse front on the axis behind the pulse front at r
        (positive for lenses of normal dispersion).
    """

    r = np.asarray(r, float)

    return -r**2/(2*C_MM_PER_FS)*wavelength_nm*dpower_dlambda


def thin_lens_power_derivative(f, n, wavelength_nm, step_nm=None):
    """dP/dlambda = (dn/dlambda)/((n - 1) f) of a thin singlet.

    Parameters
    ----------
    f : number
        Focal length (mm) at wavelength_nm.
    n : callable
        n(wavelength_nm) of the glass.
    wavelength_nm : number
    step_nm : number, optional
        See `dn_dlambda()`.

    Returns
    -------
    dP/dlambda in 1/(mm nm).
    """

    return dn_dlambda(n, wavelength_nm, 1, step_nm)/((float(n(wavelength_nm)) - 1)*f)


def system_power_derivative(system, step_nm=0.5):
    """Derivative of the paraxial power of a `raytrace.System` with respect
    to the wavelength (central difference).

    Parameters
    ----------
    system : raytrace.System
    step_nm : number, optional

    Returns
    -------
    dP/dlambda in 1/(unit nm), unit the length unit of the system.
    """

    from .raytrace import NM_PER_UNIT
    scale = NM_PER_UNIT[system.units]
    lam = system.wavelength_nm
    p = [system.with_wavelength((lam + s*step_nm)/scale).power() for s in (-1, 1)]

    return (p[1] - p[0])/(2*step_nm)


def bor_axis_spectrum(omega, spectrum, delay, dgdd=0.0, omega0=None, n_s=2001):
    """Spectrum on the axis of the focus of a lens with Bor's pulse-front
    delay (model of Bor and Horvath).

    Equal areas of a uniformly lit pupil contribute equally to the axial
    focal field, and the delay grows with the area, s = (r/a)^2 from 0 to 1.
    The field on the axis is therefore the input pulse averaged over delays
    spread uniformly from 0 to `delay` (a box of that width in time), times
    omega from the Huygens factor:

        E(omega) = omega E_in(omega) int_0^1 exp(i (Omega delay s + Omega^2
        dgdd s/2)) ds,    Omega = omega - omega0,

    where dgdd is the change of the GDD from the axis to the rim (the lens
    is thinner at the rim). The sign convention of `bor_delay()`: the rim
    leads, so the pulse arrives between -delay and 0 after the axial ray.

    Parameters
    ----------
    omega : array
        Angular frequencies (rad/fs).
    spectrum : array
        Spectrum E_in(omega) that the axial ray alone would deliver to the
        focus (the input spectrum times the phase of the axial glass path
        and of any compensation of it; with a perfect compensator the
        unchirped input spectrum).
    delay : number
        Bor delay at the rim (fs).
    dgdd : number, optional
        GDD at the rim minus GDD on the axis (fs^2).
    omega0 : number, optional
        Carrier (default: the centre of omega).
    n_s : int, optional
        Samples of the integral over s.

    Returns
    -------
    E(omega) : complex array
    """

    omega = np.asarray(omega, float)
    w0 = omega[len(omega)//2] if omega0 is None else omega0
    W = (omega - w0)[None, :]
    s = np.linspace(0.0, 1.0, n_s)[:, None]
    avg = np.trapezoid(np.exp(-1j*W*delay*s + 0.5j*W**2*dgdd*s), s[:, 0], axis=0)

    return omega*np.asarray(spectrum)*avg


RayDelays = namedtuple("RayDelays", "height phase group gdd glass")
RayDelays.__doc__ = """Delays along traced rays (`ray_delays()`).

Attributes
----------
height : array
    Ray heights in the entrance pupil (length unit of the system).
phase : array
    Phase delay OPL/c to the image plane (fs), from the reference of the
    traced optical path (for an object at infinity the plane of the first
    vertex), the same reference as in `focal_spectrum()`.
group : array
    Group delay, sum of n_g L/c over the media (fs).
gdd : array
    Group delay dispersion, sum of GVD L (fs^2).
glass : array
    Geometric path length in media of index above 1.0001.
"""


def ray_delays(system, px, field=0.0, step_nm=None):
    """Phase delay, group delay and GDD along rays of a `raytrace.System`.

    The rays are traced at the wavelength of the system; group index and GVD
    of each medium follow from the indices of the system at neighbouring
    wavelengths, so any material of a `raytrace.Surface` works. By Fermat's
    principle the change of the ray paths with the wavelength does not
    change the group delay to first order, so the sum of n_g L/c along the
    rays of the centre wavelength is the group delay.

    Parameters
    ----------
    system : raytrace.System
        Units of length "mm" give delays in fs (other units are converted).
    px : array
        Normalized meridional pupil coordinates (the rays are traced
        through the pupil points (0, px), heights along y).
    field : number, optional
        Field angle or height.
    step_nm : number, optional
        Wavelength step of the derivatives (default lambda/500).

    Returns
    -------
    RayDelays
    """

    from .raytrace import NM_PER_UNIT
    scale = NM_PER_UNIT[system.units]
    lam = system.wavelength_nm
    h = lam/500.0 if step_nm is None else step_nm
    ns = np.array([system.with_wavelength((lam + o*h)/scale).n for o in (-2, -1, 0, 1, 2)])
    n0 = ns[2]
    dn1 = np.tensordot(_STENCILS[1][0], ns, axes=(0, 0))/h
    dn2 = np.tensordot(_STENCILS[2][0], ns, axes=(0, 0))/h**2
    n_g = n0 - lam*dn1
    gvd_mm = lam**3/(2*np.pi*C_NM_PER_FS**2)*dn2*NM_PER_MM        # fs^2/mm
    b = system.trace_pupil(field, px=0.0, py=np.asarray(px, float))
    L = np.linalg.norm(np.diff(b.points, axis=0), axis=-1)*scale/NM_PER_MM   # (S + 1, N) in mm
    c = C_MM_PER_FS
    glass = np.sum(L*(n0[:, None] > 1.0001), axis=0)*NM_PER_MM/scale

    # the traced OPL starts with an offset (zero on the plane of the first vertex for
    # an object at infinity); keep that reference for both delays
    L0 = b.opl[0]*scale/NM_PER_MM/n0[0]

    return RayDelays(b.points[0][:, 1], (n0[0]*L0 + np.sum(n0[:, None]*L, axis=0))/c,
                     (n_g[0]*L0 + np.sum(n_g[:, None]*L, axis=0))/c,
                     gvd_mm[0]*L0 + np.sum(gvd_mm[:, None]*L, axis=0), glass)


def focal_spectrum(system, wavelengths_nm, r, z=0.0, n_rho=400, amplitude=None):
    """Monochromatic focal fields of a rotationally symmetric lens at many
    wavelengths, from traced rays.

    For each wavelength, meridional rays from the entrance pupil (heights
    rho) are traced to the last surface. Every exit point P carries the
    phase k OPL(P); the field at a point X near the axial image point F of
    the system (at its own wavelength) follows from a Kirchhoff integral
    over the exit surface, with the distance from P to X expanded to first
    order in the radius r of X (a Debye type integral):

        E(r, z) = (k/i) int A(rho) exp(ik W) J0(k rho_P r/D) rho drho / D,

    W = OPL(P) + D, D the distance from P to the point F + z and rho_P the
    height of P.
    Amplitudes are those of the entrance pupil (no apodization), which is
    fine at moderate NA. The absolute phase is kept, so a sum over the
    wavelengths gives the pulse in the focus with its group delays and
    dispersion.

    Parameters
    ----------
    system : raytrace.System
        Object at infinity, image plane at the reference focus; the axial
        point of the image plane at the wavelength of the system is F.
    wavelengths_nm : array
        Vacuum wavelengths.
    r : array
        Radii in the focal region (length unit of the system).
    z : number or array, optional
        Axial offsets from F.
    n_rho : int, optional
        Number of rays (midpoint rule over the pupil radius).
    amplitude : callable, optional
        A(rho_normalized) of the incident beam (default uniform).

    Returns
    -------
    E : complex array, shape (len(wavelengths_nm), len(z), len(r))
    """

    from .raytrace import NM_PER_UNIT
    scale = NM_PER_UNIT[system.units]
    r = np.atleast_1d(np.asarray(r, float))
    zs = np.atleast_1d(np.asarray(z, float))
    s = (np.arange(n_rho) + 0.5)/n_rho
    ep_r = system._pupils()[1]
    rho = s*ep_r
    A = np.ones_like(s) if amplitude is None else np.asarray(amplitude(s), float)
    F = system.trace_pupil(0.0, 0.0, 0.0).points[-1][0]
    n_img = system.n[-1]
    E = np.zeros((len(wavelengths_nm), len(zs), len(r)), complex)
    for i, lam in enumerate(np.asarray(wavelengths_nm, float)):
        si = system.with_wavelength(lam/scale)
        b = si.trace_pupil(0.0, 0.0, s)
        P, opl = b.points[-2], b.opl[-2]
        ok = np.isfinite(opl)
        k = 2*np.pi/si.wavelength
        rho_p = np.hypot(P[ok, 0], P[ok, 1])
        for j, dz in enumerate(zs):
            D = np.linalg.norm(F + np.array([0.0, 0.0, dz]) - P[ok], axis=-1)
            W = opl[ok] + n_img*D
            kern = j0(k*n_img*np.outer(r, rho_p/D))
            E[i, j] = (k*n_img/1j)*kern @ (A[ok]*np.exp(1j*k*W)*rho[ok]/D)*(ep_r/n_rho)

    return E


# ------------------------------------------------------ angular dispersion

def tilt_angle(wavelength_nm, dtheta_dlambda):
    """Pulse-front tilt angle gamma from tan(gamma) = lambda dtheta/dlambda
    (angular dispersion in vacuum or air).

    Parameters
    ----------
    wavelength_nm : number
    dtheta_dlambda : number or array
        Angular dispersion in rad/nm.

    Returns
    -------
    gamma : number or array (rad)
    """

    return np.arctan(wavelength_nm*np.asarray(dtheta_dlambda, float))


def prism_angles(apex, n, theta_in):
    """Internal and exit angles of a ray through a prism (to the face normals).

    Parameters
    ----------
    apex : number
        Apex angle (rad).
    n : number or array
        Index of the prism (in air).
    theta_in : number
        Angle of incidence at the first face (rad).

    Returns
    -------
    theta_r1, theta_r2, theta_out : arrays (rad)
    """

    t1 = np.arcsin(np.sin(theta_in)/np.asarray(n, float))
    t2 = apex - t1

    return t1, t2, np.arcsin(n*np.sin(t2))


def minimum_deviation_incidence(apex, n):
    """Angle of incidence arcsin(n sin(apex/2)) of minimum deviation.

    Parameters
    ----------
    apex : number
        Apex angle (rad).
    n : number
        Index of the prism.

    Returns
    -------
    theta_in : number (rad)
    """

    return np.arcsin(n*np.sin(apex/2))


def prism_dispersion(apex, n, wavelength_nm, theta_in):
    """Angular dispersion dtheta_out/dlambda = (dn/dlambda) sin(apex)/(cos
    theta_r1 cos theta_out) of a prism at a fixed angle of incidence.

    Parameters
    ----------
    apex : number
        Apex angle (rad).
    n : callable
        n(wavelength_nm) of the prism.
    wavelength_nm : number
    theta_in : number
        Angle of incidence (rad).

    Returns
    -------
    dtheta/dlambda in rad/nm.
    """

    t1, _, t_out = prism_angles(apex, float(n(wavelength_nm)), theta_in)

    return dn_dlambda(n, wavelength_nm)*np.sin(apex)/(np.cos(t1)*np.cos(t_out))


def grating_angle(period_nm, order, wavelength_nm, theta_in=0.0):
    """Diffraction angle from sin(theta_m) = sin(theta_in) + m lambda/d.

    Parameters
    ----------
    period_nm : number
    order : int
    wavelength_nm : number or array
    theta_in : number, optional
        Angle of incidence (rad).

    Returns
    -------
    theta_m : number or array (rad)
    """

    return np.arcsin(np.sin(theta_in) + order*np.asarray(wavelength_nm, float)/period_nm)


def grating_dispersion(period_nm, order, wavelength_nm, theta_in=0.0):
    """Angular dispersion m/(d cos theta_m) of a grating.

    Parameters
    ----------
    period_nm : number
    order : int
    wavelength_nm : number
    theta_in : number, optional

    Returns
    -------
    dtheta/dlambda in rad/nm.
    """

    return order/(period_nm*np.cos(grating_angle(period_nm, order, wavelength_nm, theta_in)))


def _refract(d, normal, n1, n2):
    """Refract unit directions d (N, 2) at a plane with unit normal (2,)
    pointing against d."""

    cos_i = -d @ normal
    mu = n1/n2
    cos_t = np.sqrt(1 - mu**2*(1 - cos_i**2))

    return mu*d + (mu*cos_i - cos_t)[:, None]*normal[None, :]


PrismDelays = namedtuple("PrismDelays", "x phase group glass direction")
PrismDelays.__doc__ = """Delays of parallel rays through a prism (`prism_ray_delays()`).

Attributes
----------
x : array
    Transverse position in the output plane, normal to the central exit
    direction (mm, towards the apex positive).
phase, group : arrays
    Phase and group delays from the input plane to the output plane (fs).
glass : array
    Path length in the prism (mm).
direction : (2,) array
    Central exit direction (z, y).
"""


def prism_ray_delays(apex, n, wavelength_nm, theta_in, heights, hit=5.0, distance=20.0):
    """Trace a parallel beam through a prism in air (2D) and sum the delays.

    The prism has its apex at the origin and its base below (y < 0); its
    first face is met by the central ray at the distance `hit` from the
    apex. The rays start on a plane normal to the beam and end on a plane
    normal to the central exit direction, `distance` behind the second face.

    Parameters
    ----------
    apex : number
        Apex angle (rad).
    n : callable
        n(wavelength_nm).
    wavelength_nm : number
    theta_in : number
        Angle of incidence at the first face (rad).
    heights : array
        Offsets of the rays across the input beam (mm, towards the apex
        positive).
    hit : number, optional
        Distance (mm) of the central hit point from the apex along face 1.
    distance : number, optional
        Distance (mm) of the output plane behind the exit point of the
        central ray.

    Returns
    -------
    PrismDelays
    """

    s = np.asarray(heights, float)
    a2 = apex/2
    n1 = np.array([-np.cos(a2), np.sin(a2)])             # outward normals (z, y)
    n2 = np.array([np.cos(a2), np.sin(a2)])
    t1 = np.array([-np.sin(a2), -np.cos(a2)])            # face 1, from the apex down
    ang = -a2 + theta_in                                 # direction of the incident beam
    d_in = np.array([np.cos(ang), np.sin(ang)])
    perp_in = np.array([-d_in[1], d_in[0]])
    Q0 = hit*t1
    start = Q0 - 10.0*d_in + s[:, None]*perp_in[None, :]
    nn = float(n(wavelength_nm))
    ng = float(group_index(n, wavelength_nm))
    d = np.repeat(d_in[None, :], len(s), axis=0)
    t = -(start @ n1)/(d @ n1)
    P1 = start + t[:, None]*d
    d_g = _refract(d, n1, 1.0, nn)
    t_g = -(P1 @ n2)/(d_g @ n2)
    P2 = P1 + t_g[:, None]*d_g
    d_out = _refract(d_g, -n2, nn, 1.0)
    i0 = np.argmin(np.abs(s))
    dc = d_out[i0]
    O = P2[i0] + distance*dc
    t_o = ((O - P2) @ dc)/(d_out @ dc)
    P3 = P2 + t_o[:, None]*d_out
    x = (P3 - O) @ np.array([-dc[1], dc[0]])
    air = t + t_o
    c = C_MM_PER_FS

    return PrismDelays(x, (air + nn*t_g)/c, (air + ng*t_g)/c, t_g, dc)


def dispersed_beam_xt(x, t, omega, spectrum, theta, w):
    """Envelope E(x, t) of a beam with angular dispersion, in the plane normal
    to its central direction: each frequency is a plane wave at the angle
    theta(omega) - theta(omega_0) with the Gaussian profile exp(-x^2/w^2).

    Parameters
    ----------
    x : array
        Transverse positions (mm).
    t : array
        Times (fs).
    omega : array
        Uniform angular frequencies (rad/fs); the centre one is omega_0.
    spectrum : array
        E(omega).
    theta : array
        Propagation angles theta(omega) (rad).
    w : number
        Beam radius (mm).

    Returns
    -------
    E : complex array, shape (len(t), len(x))
    """

    omega = np.asarray(omega, float)
    th0 = theta[len(omega)//2]
    kx = omega/C_MM_PER_FS*np.sin(np.asarray(theta) - th0)
    E_wx = spectrum[:, None]*np.exp(1j*np.outer(kx, x))*np.exp(-(np.asarray(x)/w)**2)[None, :]

    return to_time(E_wx, omega, t, axis=0)


# --------------------------------------------------- space-time focusing

def beamlet_focus_1d(x, z, wavelength_nm, x_c, w, f):
    """Exact paraxial field of a 1D Gaussian beamlet exp(-(x - x_c)^2/w^2)
    focused by an ideal cylindrical lens of focal length f, observed at the
    distance f + z behind the lens (Collins integral).

    The carrier phase k (f + z) is included, so that sums over wavelengths
    keep the group delay of the propagation.

    Parameters
    ----------
    x : array
        Transverse positions (mm).
    z : number
        Distance from the focal plane (mm).
    wavelength_nm : number
    x_c : number
        Centre of the beamlet on the lens (mm).
    w : number
        Beamlet radius (field 1/e) on the lens (mm).
    f : number
        Focal length (mm).

    Returns
    -------
    E : complex array
    """

    k = 2*np.pi/(wavelength_nm/NM_PER_MM)
    d = f + z
    A, B, D = 1 - d/f, d, 1.0
    x = np.asarray(x, float)
    a = 1/w**2 - 1j*k*A/(2*B)
    b = 2*x_c/w**2 - 1j*k*x/B
    c = -x_c**2/w**2 + 1j*k*D*x**2/(2*B)

    return np.sqrt(k/(2j*np.pi*B))*np.sqrt(np.pi/a)*np.exp(b**2/(4*a) + c + 1j*k*d)


def sstf_axis_pulse(z, tau_fwhm, spatial_chirp, w, f, wavelength_nm):
    """Duration of the pulse on the axis near the focus of a spatially
    chirped beam (simultaneous spatial and temporal focusing, 1D).

    Each frequency omega_0 + Omega is a beamlet of radius w centred at
    x_c = spatial_chirp * Omega on the lens; all beamlets cross in the
    focus, where the full bandwidth is present. At the distance z from the
    focus, a beamlet is displaced by -x_c z/f and travels at the angle -x_c/f,
    so the axis sees a narrower spectrum (Gaussian beamlets of radius
    w(z)) with the quadratic spectral phase of the tilt and the curvature:

        tau_TL(z) = tau sqrt(1 + (spatial_chirp Omega_b z/(f w(z)))^2),
        GDD(z) = -k spatial_chirp^2 z z_R^2/(f^2 (z^2 + z_R^2)),

    Omega_b the 1/e half width of the field spectrum, w0 = lambda f/(pi w)
    and z_R = pi w0^2/lambda the waist and Rayleigh range of a beamlet. The
    duration follows from `gaussian_broadening()` with these two.

    Parameters
    ----------
    z : number or array
        Distance from the focus (mm).
    tau_fwhm : number
        Transform-limited intensity FWHM of the pulse (fs).
    spatial_chirp : number
        dx_c/dOmega on the lens (mm fs/rad).
    w : number
        Beamlet radius (field 1/e) on the lens (mm).
    f : number
        Focal length (mm).
    wavelength_nm : number

    Returns
    -------
    tau : array
        FWHM on the axis (fs).
    tau_tl : array
        Transform limit of the local spectrum (fs).
    gdd : array
        Local GDD (fs^2).
    """

    z = np.asarray(z, float)
    lam = wavelength_nm/NM_PER_MM
    k = 2*np.pi/lam
    w0 = lam*f/(np.pi*w)
    z_r = np.pi*w0**2/lam
    wz = w0*np.sqrt(1 + (z/z_r)**2)
    omega_b = np.sqrt(8*np.log(2))/tau_fwhm
    tau_tl = tau_fwhm*np.sqrt(1 + (spatial_chirp*omega_b*z/(f*wz))**2)
    gdd = -k*spatial_chirp**2*z*z_r**2/(f**2*(z**2 + z_r**2))

    return gaussian_broadening(tau_tl, gdd), tau_tl, gdd
