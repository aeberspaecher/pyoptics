#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical fibres: index profiles, rays, modes, dispersion, attenuation and
fibre Bragg gratings.

Units
-----
The practical units of fibre optics: the cross section (core radius, radial
coordinates) and the wavelength in µm, fibre lengths in km, times in ps,
the dispersion parameter D in ps/(nm km), group velocity dispersion beta_2
in ps²/km, attenuation in dB/km. Fibre Bragg gratings are short, so their
length and positions are in µm as well (a 10 mm grating is 1E4 µm); their
group delays are in ps and their dispersion in ps/nm.

Index profiles
--------------
A `Fibre` has the core radius a, a core and a cladding index (numbers, or
functions of the wavelength for dispersive glasses) and a profile shape
f(rho) of rho = r/a with ::

    n(r)**2 = n_clad**2 + f(r/a) (n_core**2 - n_clad**2),

f = 1 on the axis and 0 in the cladding. step_index(), graded_index()
(power law f = 1 - rho**alpha, alpha = 2 is parabolic, alpha -> inf the
step), triangular() and w_profile() (core, depressed ring, cladding: f < 0
in the ring) build the usual fibres. ge_doped() gives the core and
cladding glasses from `materials.germania_silica_index()` (Fleming's
Sellmeier data), so the material dispersion, and the dispersion of the
profile (Delta depends on the wavelength), are included. With n(r) the same
shape at all wavelengths, Delta(lambda) = (n_core**2 - n_clad**2)/(2 n_core**2)
carries all of the profile dispersion.

Rays
----
In the step-index core a ray is straight between reflections at r = a. Its
invariants are beta_bar = n_core cos(theta_z) (theta_z the angle to the
axis) and l_bar = n_core sin(theta_z) r_min/a, with r_min the closest
approach of its projection to the axis (l_bar = 0 for meridional rays).
After Snyder and Love the ray is bound for n_clad <= beta_bar, tunnelling
for n_clad**2 - l_bar**2 <= beta_bar**2 < n_clad**2 (totally reflected at
the boundary, but light leaks out through the curved interface), and
refracting otherwise (ray_class()). A ray's transit time through a length L
of the core is L n_core/(c cos(theta_z)) whether meridional or skew. The
spread between the axial ray and the steepest bound ray is
L n_core/c (n_core/n_clad - 1) ~ L n_core Delta/c (step_delay_spread()).

For graded profiles Fibre.index_field() returns a `grin.IndexField` that
`grin.trace()` follows; the ray parameter is t with dt = ds/n and the transit
time of a ray (without material dispersion) is its optical path over c.

Modes
-----
scalar_modes() solves the scalar wave equation of the weakly guiding fibre
for a field u(r) cos(l phi), ::

    u'' + u'/r - l**2/r**2 u + k**2 n(r)**2 u = beta**2 u,

for any profile, by finite volumes on the cells r_j = (j + 1/2) h (no
point at r = 0, cell averaged n**2 at the steps): multiplied by r, the
operator is symmetric and tridiagonal, and the guided modes (beta > k
n_clad) come from scipy's tridiagonal eigensolver. For the step index the
result is the LP mode of `modes.lp_modes()`, second order in h. The
modes of the infinite parabolic profile n**2 = n_core**2 (1 - 2 Delta
(r/a)**2) have the analytic eigenvalues
beta**2 = k**2 n_core**2 - 2 k n_core sqrt(2 Delta)/a (2 p + l + 1)
(parabolic_betas()): the modes with the same principal number q = 2p + l + 1
form degenerate, equally spaced groups.

vector_modes() solves the exact eigenvalue equations of the step-index
fibre for the TE_0m, TM_0m, HE_nu,m and EH_nu,m modes (Snyder and Love,
table 12-3). As Delta -> 0, HE_11 becomes LP_01 and TE_01, TM_01 and HE_21
become LP_11 (generally HE_l+1,m, EH_l-1,m and TE/TM form LP_lm).

Mode field diameters: petermann_mfd() (Petermann II, 2 w with
w**2 = 2 int u**2 r dr/int (du/dr)**2 r dr, the definition of ITU-T G.650 that
predicts splice losses and waveguide dispersion); Marcuse's Gaussian fit is
`modes.marcuse_mode_field_radius()`. Bend loss of LP01: bend_loss()
(Marcuse 1976).

Dispersion
----------
group_index() and dispersion() differentiate the effective index of a mode
numerically with respect to the wavelength, with the full material
dispersion of the profile, ::

    N_g = n_eff - lambda dn_eff/dlambda,   D = -(lambda/c) d**2 n_eff/dlambda**2,

and material_dispersion() does the same for a glass. The group delay of a
length L is tau = L N_g/c. waveguide_dispersion() gives Gloge's classic
split D_wg = -(N_clad Delta/(c lambda)) V d**2(Vb)/dV**2 of the step-index
fibre. wkb_group_delay() is the WKB group delay of the modes of a power law
profile (Gloge and Marcatili, Olshansky and Keck), in terms of the
normalized depth delta = (k**2 n_core**2 - beta**2)/(2 Delta k**2 n_core**2)
of a mode in the profile (0 on the axis, 1 at cutoff), ::

    tau/L = N_core/c [1 + Delta (alpha - 2 - eps)/(alpha + 2) delta
                     + Delta**2 (3 alpha - 2 - 2 eps)/(2 (alpha + 2)) delta**2],

with the profile dispersion eps = -2 n_core lambda/(N_core Delta) dDelta/dlambda;
optimal_alpha() is the profile that minimizes the rms spread,
alpha_opt = 2 + eps - Delta (4 + eps)(3 + eps)/(5 + 2 eps). For
uniformly excited modes the mode number m/M is uniform in [0, 1] and
delta = (m/M)**(alpha/(alpha + 2)).

pmd_dgd() draws the differential group delay of fibres made of randomly
oriented birefringent sections (polarization mode dispersion), whose mean
grows as the square root of the length.

Attenuation
-----------
attenuation() is the usual model of a silica fibre in dB/km: Rayleigh
scattering A/lambda**4, the infrared absorption tail
7.81E11 exp(-48.48/lambda) of the Si-O vibrations (Pinnow et al.), the
ultraviolet Urbach tail of GeO2-doped silica
154.2 x/(46.6 x + 60) 1E-4 exp(4.63/lambda) (Keiser), and OH overtone
peaks at 1383, 1240 and 950 nm (Gaussian bands of adjustable height; 0 for
"zero water peak" fibres).

Fibre Bragg gratings
--------------------
A grating n_eff(z) + dn_dc(z) + dn_ac(z) cos(2 pi z/Lambda + phi(z)) couples
the forward and backward core mode. With the detuning
delta = 2 pi n_eff/lambda - pi/Lambda, sigma = 2 pi dn_dc/lambda and
kappa = pi dn_ac/lambda the coupled mode equations for the forward and
backward amplitudes R and S read (Erdogan 1997) ::

    dR/dz =  i (delta + sigma) R + i kappa S,
    dS/dz = -i (delta + sigma) S - i conj(kappa) R.

fbg_uniform() is the analytic solution of a uniform grating,
r = i conj(kappa) sinh(gamma L)/(gamma cosh(gamma L) - i sigma_hat sinh(gamma L)),
gamma**2 = kappa**2 - sigma_hat**2, with the peak reflectivity tanh**2(kappa L)
at the Bragg wavelength lambda_B = 2 (n_eff + dn_dc) Lambda.
fbg_transfer_matrix() multiplies the 2x2 transfer matrices of short uniform
sections (the piecewise uniform method of Yamada and Sakuda): apodization
(dn_ac(z)), chirp (Lambda(z)), dn_dc(z) and discrete phase shifts.
fbg_delay() gives the group delay tau = d(arg r)/d omega and the
dispersion d tau/d lambda of the reflection; bragg_shift() the shift of
the Bragg wavelength with strain and temperature.

Sign conventions: fields vary as exp(i(beta z - omega t)), as everywhere in
pyoptics.

References
----------
A. W. Snyder and J. D. Love, "Optical Waveguide Theory", Chapman and Hall
(1983), chapters 2, 12 and 14.

D. Gloge and E. A. J. Marcatili, "Multimode theory of graded-core fibers",
Bell Syst. Tech. J. 52, 1563-1578 (1973).

R. Olshansky and D. B. Keck, "Pulse broadening in graded-index optical
fibers", Appl. Opt. 15, 483-491 (1976).

D. Gloge, "Dispersion in weakly guiding fibers", Appl. Opt. 10, 2442-2445
(1971).

K. Petermann, "Constraints for fundamental-mode spot size for broadband
dispersion-compensated single-mode fibres", Electron. Lett. 19, 712-714
(1983).

D. Marcuse, "Curvature loss formula for optical fibers", J. Opt. Soc. Am.
66, 216-220 (1976).

D. A. Pinnow, T. C. Rich, F. W. Ostermayer and M. DiDomenico,
"Fundamental optical attenuation limits in the liquid and glassy state with
application to fiber optical waveguide materials", Appl. Phys. Lett. 22,
527-529 (1973).

G. Keiser, "Optical Fiber Communications", 5th ed., McGraw-Hill (2015),
chapters 2 and 3.

T. Erdogan, "Fiber grating spectra", J. Lightwave Technol. 15, 1277-1294
(1997).

M. Yamada and K. Sakuda, "Analysis of almost-periodic distributed feedback
slab waveguides via a fundamental matrix approach", Appl. Opt. 26,
3474-3478 (1987).

A. Othonos, "Fiber Bragg gratings", Rev. Sci. Instrum. 68, 4309-4341
(1997).

J. P. Gordon and H. Kogelnik, "PMD fundamentals: polarization mode
dispersion in optical fibers", PNAS 97, 4541-4550 (2000).
"""

from collections import namedtuple

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import jv, kv, kve

from . import grin
from .materials import germania_silica_index
from .modes import lp_modes, numerical_aperture

_trapz = getattr(np, "trapezoid", None) or np.trapz


__all__ = ["C_UM_PER_PS", "Fibre", "step_index", "graded_index", "triangular", "w_profile",
           "ge_doped", "smf28", "acceptance_angle", "ray_class", "step_ray", "step_ray_delay",
           "step_delay_spread", "ScalarMode", "scalar_modes", "scalar_mode", "parabolic_betas",
           "VectorMode", "vector_modes", "petermann_mfd", "bend_loss", "effective_index",
           "group_index", "dispersion", "dispersion_slope", "material_dispersion",
           "waveguide_dispersion", "vb_derivatives", "beta2_from_D", "D_from_beta2",
           "wkb_group_delay", "wkb_delay_spread", "optimal_alpha", "pmd_dgd", "attenuation",
           "attenuation_components", "bragg_wavelength", "fbg_uniform", "fbg_transfer_matrix",
           "fbg_delay", "bragg_shift"]


C_UM_PER_PS = 299.792458   # speed of light in µm/ps
_PS_PER_KM = 1E9/C_UM_PER_PS   # ps per km at n = 1
_D_FACTOR = -1E12/2.99792458E8   # -lambda n''/c in ps/(nm km) for lambda in µm and n'' in 1/µm²


def _index_function(value):
    """A number or a function of the wavelength (µm) as a function."""
    if callable(value):
        return value
    v = float(value)
    return lambda wavelength: v + 0*np.asarray(wavelength, dtype=float)


class Fibre(object):
    """A circularly symmetric fibre.

    Parameters
    ----------
    a : number
        Core radius in µm.
    shape : callable
        f(rho), rho = r/a: 1 on the axis, 0 in the cladding, negative in a
        depressed region.
    n_core, n_clad : number or callable
        Indices on the axis and in the cladding, numbers or functions of the
        vacuum wavelength in µm.
    name : str, optional
    step_at : sequence of numbers, optional
        rho of the discontinuities of the shape (the mode solver averages
        n**2 over the cells that contain one).
    """

    def __init__(self, a, shape, n_core, n_clad, name="", step_at=()):
        self.a = float(a)
        self.shape = shape
        self._n_core, self._n_clad = _index_function(n_core), _index_function(n_clad)
        self.name = name
        self.step_at = tuple(step_at)

    def __repr__(self):
        return f"Fibre({self.name or 'custom'}, a = {self.a} µm)"

    def n_core(self, wavelength):
        """Index on the axis at the vacuum wavelength (µm)."""
        return self._n_core(wavelength)

    def n_clad(self, wavelength):
        """Index of the cladding."""
        return self._n_clad(wavelength)

    def n2(self, r, wavelength):
        """n(r)**2 at the radii r (µm)."""
        n1, n2 = self.n_core(wavelength), self.n_clad(wavelength)
        f = self.shape(np.abs(np.asarray(r, dtype=float))/self.a)
        return n2**2 + f*(n1**2 - n2**2)

    def n(self, r, wavelength):
        """Refractive index n(r)."""
        return np.sqrt(self.n2(r, wavelength))

    def delta(self, wavelength):
        """Relative index difference (n_core**2 - n_clad**2)/(2 n_core**2)."""
        n1, n2 = self.n_core(wavelength), self.n_clad(wavelength)
        return (n1**2 - n2**2)/(2*n1**2)

    def NA(self, wavelength):
        """Numerical aperture sqrt(n_core**2 - n_clad**2) (of the axis)."""
        return numerical_aperture(self.n_core(wavelength), self.n_clad(wavelength))

    def V(self, wavelength):
        """V number 2 pi a NA/lambda."""
        return 2*np.pi*self.a/np.asarray(wavelength, dtype=float)*self.NA(wavelength)

    def index_field(self, wavelength):
        """The fibre as a `grin.IndexField` at one wavelength, with the axis
        along the last coordinate (positions (..., 2) for (x, z) or
        (..., 3) for (x, y, z), in µm)."""
        n1, n2 = float(self.n_core(wavelength)), float(self.n_clad(wavelength))
        shape, a = self.shape, self.a

        def n2_of(r):
            rho = np.sqrt(np.sum(np.asarray(r)[..., :-1]**2, axis=-1))/a
            return n2**2 + shape(rho)*(n1**2 - n2**2)

        field = grin.IndexField(n=lambda r: np.sqrt(n2_of(r)), h=1E-4*a)
        field.n2 = n2_of
        return field

    def modes(self, wavelength, **kwargs):
        """All guided scalar (LP) modes, see `scalar_modes()`."""
        return scalar_modes(self, wavelength, **kwargs)


def _step_shape(rho):
    return (np.asarray(rho) < 1).astype(float)


def step_index(a, n_core, n_clad, name="step index"):
    """Step-index fibre (f = 1 for rho < 1)."""
    return Fibre(a, _step_shape, n_core, n_clad, name, step_at=(1.0,))


def graded_index(a, alpha, n_core, n_clad, name=None):
    """Power-law graded-index fibre, f = 1 - rho**alpha in the core.

    alpha = 2 is the parabolic profile, alpha = 1 triangular, large alpha
    approaches the step.
    """

    alpha = float(alpha)

    def shape(rho):
        rho = np.asarray(rho, dtype=float)
        return np.where(rho < 1, 1 - rho**alpha, 0.0)

    return Fibre(a, shape, n_core, n_clad, name or f"graded index, alpha = {alpha:g}",
                 step_at=(1.0,) if alpha > 20 else ())


def triangular(a, n_core, n_clad, name="triangular"):
    """Triangular core f = 1 - rho, the classic dispersion-shifted profile."""
    return graded_index(a, 1.0, n_core, n_clad, name)


def w_profile(a, ring_outer, ring_depth, n_core, n_clad, name="W profile"):
    """Step core, a depressed ring and the cladding (a "W" profile).

    Parameters
    ----------
    a : number
        Core radius (µm).
    ring_outer : number
        Outer radius of the depressed ring in units of a (> 1).
    ring_depth : number
        f in the ring, as a fraction of the core step (0.5: the ring lies
        half as far below the cladding as the core lies above it).
    n_core, n_clad : number or callable
    """

    ring_outer, ring_depth = float(ring_outer), float(ring_depth)

    def shape(rho):
        rho = np.asarray(rho, dtype=float)
        return np.where(rho < 1, 1.0, np.where(rho < ring_outer, -ring_depth, 0.0))

    return Fibre(a, shape, n_core, n_clad, name, step_at=(1.0, ring_outer))


def ge_doped(x_core, x_clad=0.0):
    """Core and cladding glasses of GeO2-doped silica.

    Parameters
    ----------
    x_core, x_clad : numbers
        Mole fractions of GeO2 (0.035 for 3.5 mol-%).

    Returns
    -------
    n_core, n_clad : callables
        n(wavelength in µm) from `materials.germania_silica_index()`.
    """

    return (lambda wl: germania_silica_index(1E3*np.asarray(wl, dtype=float), x_core),
            lambda wl: germania_silica_index(1E3*np.asarray(wl, dtype=float), x_clad))


def smf28(x_core=0.032, a=4.4):
    """A standard single-mode fibre like ITU-T G.652 (Corning SMF-28) as a
    step-index model: core radius 4.4 µm, GeO2-doped core (3.2 mol-%,
    Delta about 0.33 %) in a pure silica cladding. The parameters are chosen
    to match the data sheet: theoretical cutoff 1.35 µm (cabled cutoff below
    1.26 µm), zero dispersion at 1.30 µm, D = 17.6 ps/(nm km) at 1550 nm,
    mode field diameters 9.3 µm at 1310 nm and 10.5 µm at 1550 nm (Petermann
    II). The real fibre has a nominal core radius of 4.1 µm and a profile
    that is not quite a step."""
    return step_index(a, *ge_doped(x_core), name="SMF-28-like")


# ---------------------------------------------------------------- rays

def acceptance_angle(NA, n0=1.0):
    """Half angle of the acceptance cone in the medium n0 (radians),
    arcsin(NA/n0) (pi/2 if NA >= n0)."""
    return np.arcsin(np.clip(np.asarray(NA, dtype=float)/n0, 0, 1))


def ray_class(beta_bar, l_bar, n_core, n_clad):
    """Classify step-index rays by their invariants (Snyder and Love).

    Parameters
    ----------
    beta_bar : number or array
        n_core cos(theta_z).
    l_bar : number or array
        n_core sin(theta_z) r_min/a.
    n_core, n_clad : numbers

    Returns
    -------
    kind : int array
        0: bound, 1: tunnelling, 2: refracting.
    """

    b2 = np.asarray(beta_bar, dtype=float)**2
    l2 = np.asarray(l_bar, dtype=float)**2
    return np.where(b2 >= n_clad**2, 0, np.where(b2 + l2 >= n_clad**2, 1, 2))


def step_ray(a, position, direction, length):
    """A ray in a step-index core, straight between specular reflections at
    the cylinder x**2 + y**2 = a**2 (axis z).

    Parameters
    ----------
    a : number
        Core radius.
    position : array (3,)
        Start (x, y, z) inside the core.
    direction : array (3,)
        Direction (normalized here), with a positive z component.
    length : number
        Length along z to follow.

    Returns
    -------
    points : array (n, 3)
        The start, the reflection points and the end point at z0 + length.
    """

    p = np.asarray(position, dtype=float).copy()
    d = np.asarray(direction, dtype=float)
    d = d/np.linalg.norm(d)
    z_end = p[2] + length
    points = [p.copy()]
    dt2 = d[0]**2 + d[1]**2
    while True:
        if dt2 < 1E-30:
            break
        # p + s d on the cylinder: |p_t + s d_t|**2 = a**2, s > 0
        b = p[0]*d[0] + p[1]*d[1]
        c = p[0]**2 + p[1]**2 - a**2
        s = (-b + np.sqrt(max(b**2 - dt2*c, 0.0)))/dt2
        if p[2] + s*d[2] >= z_end:
            break
        p = p + s*d
        nrm = np.array([p[0], p[1], 0.0])/a
        d = d - 2*np.dot(d, nrm)*nrm
        points.append(p.copy())
    s = (z_end - p[2])/d[2]
    points.append(p + s*d)
    return np.array(points)


def step_ray_delay(theta_z, n_core, length_km=1.0):
    """Transit time (ps) of a ray at the angle theta_z to the axis through
    length_km of a step-index core (no material dispersion):
    L n_core/(c cos(theta_z)), the same for meridional and skew rays."""
    return length_km*_PS_PER_KM*n_core/np.cos(theta_z)


def step_delay_spread(n_core, n_clad, length_km=1.0):
    """Transit time difference (ps) between the steepest bound ray and the
    axial ray of a step-index fibre, L n_core/c (n_core/n_clad - 1)."""
    return length_km*_PS_PER_KM*n_core*(n_core/n_clad - 1)


# ---------------------------------------------------------------- modes

class ScalarMode(namedtuple("ScalarMode", ["l", "m", "beta", "n_eff", "wavelength", "r", "u"])):
    """A scalar (LP) mode u(r) cos(l phi) of a fibre.

    l, m: azimuthal and radial index; beta: propagation constant (1/µm);
    n_eff = beta/k; wavelength (µm); r, u: radial samples, u normalized to
    int u**2 r dr = 1 and positive near the axis (or at its first maximum).
    """

    def b(self, fibre):
        """Normalized propagation constant (n_eff**2 - n_clad**2)/(n_core**2 - n_clad**2)."""
        n1, n2 = fibre.n_core(self.wavelength), fibre.n_clad(self.wavelength)
        return (self.n_eff**2 - n2**2)/(n1**2 - n2**2)

    def radial(self, r):
        """u interpolated at the radii r (0 beyond the window)."""
        return np.interp(np.abs(r), self.r, self.u, right=0.0)

    def field(self, x, y, orientation="cos"):
        """Field on a grid (x, y in µm, linear or meshed), cos(l phi),
        sin(l phi) or exp(+-i l phi) ("cos", "sin", "+", "-"), normalized
        to unit power on the grid."""
        X, Y = np.meshgrid(x, y) if np.ndim(x) == 1 else (x, y)
        phi = np.arctan2(Y, X)
        az = {"cos": np.cos(self.l*phi), "sin": np.sin(self.l*phi),
              "+": np.exp(1j*self.l*phi), "-": np.exp(-1j*self.l*phi)}[orientation]
        f = self.radial(np.hypot(X, Y))*az
        dx = abs(X[0, 1] - X[0, 0])
        dy = abs(Y[1, 0] - Y[0, 0])
        return f/np.sqrt(np.sum(np.abs(f)**2)*dx*dy)


def _cell_n2(fibre, wavelength, r_edges, sub=16):
    """n**2 averaged (with weight r) over the cells between r_edges."""
    t = (np.arange(sub) + 0.5)/sub
    lo, hi = r_edges[:-1], r_edges[1:]
    rs = lo[:, None] + t[None, :]*(hi - lo)[:, None]
    return np.sum(fibre.n2(rs, wavelength)*rs, axis=1)/np.sum(rs, axis=1)


def _radial_operator(fibre, wavelength, r_max, n_points):
    h = r_max/n_points
    r = (np.arange(n_points) + 0.5)*h
    edges = np.arange(n_points + 1)*h
    n2 = _cell_n2(fibre, wavelength, edges)
    return r, h, n2


def scalar_modes(fibre, wavelength, l_values=None, r_max=None, n_points=3000, max_modes=None):
    """Guided scalar (LP) modes of a fibre by finite volumes in r.

    Parameters
    ----------
    fibre : Fibre
    wavelength : number
        Vacuum wavelength (µm).
    l_values : iterable of int, optional
        Azimuthal orders to solve (default: l = 0, 1, ... while modes exist).
    r_max : number, optional
        Outer radius of the window (Dirichlet wall), default 6 a (the tail
        of a single-mode fibre's mode near cutoff reaches beyond 4 a).
    n_points : int, optional
        Number of cells.
    max_modes : int, optional
        Largest number of modes per l.

    Returns
    -------
    modes : list of ScalarMode
        Sorted by decreasing n_eff.
    """

    k = 2*np.pi/wavelength
    if r_max is None:
        r_max = 6*fibre.a
    r, h, n2 = _radial_operator(fibre, wavelength, r_max, n_points)
    n_clad2 = float(fibre.n_clad(wavelength))**2
    lower = k**2*n_clad2
    upper = k**2*n2.max()*(1 + 1E-12)
    r_face = np.arange(1, n_points + 1)*h   # r_{j+1/2}
    # (r u')' with zero flux at r = 0 and u = 0 beyond the window
    main_flux = -(r_face + np.r_[0.0, r_face[:-1]])/h**2
    off = r_face[:-1]/h**2
    out = []
    l = 0
    auto = l_values is None
    l_iter = iter(l_values) if not auto else None
    while True:
        if not auto:
            try:
                l = next(l_iter)
            except StopIteration:
                break
        diag = (main_flux + k**2*n2*r - l**2/r)/r
        offd = off/np.sqrt(r[:-1]*r[1:])
        try:
            vals, vecs = eigh_tridiagonal(diag, offd, select="v", select_range=(lower, upper))
        except Exception:
            vals, vecs = np.array([]), np.zeros((n_points, 0))
        if len(vals) == 0 and auto:
            break
        order = np.argsort(vals)[::-1]
        if max_modes is not None:
            order = order[:max_modes]
        for m, i in enumerate(order, start=1):
            u = vecs[:, i]/np.sqrt(r)
            u = u/np.sqrt(np.sum(u**2*r)*h)
            j0 = np.argmax(np.abs(u) > 0.5*np.abs(u).max())
            if u[j0] < 0:
                u = -u
            beta = np.sqrt(vals[i])
            out.append(ScalarMode(l, m, beta, beta/k, wavelength, r, u))
        if auto:
            l += 1
    return sorted(out, key=lambda mode: -mode.n_eff)


def scalar_mode(fibre, wavelength, l=0, m=1, **kwargs):
    """The single scalar mode LP_lm (see `scalar_modes()`).

    Raises
    ------
    ValueError
        If LP_lm is not guided.
    """

    for mode in scalar_modes(fibre, wavelength, l_values=[l], max_modes=m, **kwargs):
        if mode.m == m:
            return mode
    raise ValueError(f"LP{l}{m} is not guided at {wavelength} µm")


def parabolic_betas(a, n_core, delta, wavelength, q):
    """Propagation constants of the infinite parabolic profile
    n**2 = n_core**2 (1 - 2 delta (r/a)**2) for the principal mode numbers
    q = 2p + l + 1 (= 2m + l - 1 with m = p + 1)."""
    k = 2*np.pi/wavelength
    q = np.asarray(q, dtype=float)
    return np.sqrt(k**2*n_core**2 - 2*k*n_core*np.sqrt(2*delta)/a*q)


VectorMode = namedtuple("VectorMode", ["kind", "nu", "m", "U", "W", "V", "n_eff"])
VectorMode.__doc__ = """Exact mode of a step-index fibre: kind is "TE", "TM", "HE"
or "EH", nu the azimuthal order, m the radial index; U, W, V as for LP
modes; n_eff the effective index."""


def _vector_function(kind, nu, V, r):
    """Pole free eigenvalue function of the exact step-index modes, a
    function of U (r = n_clad**2/n_core**2)."""

    def f(U):
        W = np.sqrt(np.maximum(V**2 - U**2, 1E-300))
        Kt = -(kve(nu - 1, W) + kve(nu + 1, W))/(2*W*kve(nu, W))   # K'_nu/(W K_nu)
        if kind == "TE":
            # J1/(U J0) = -K1/(W K0)  ->  J1(U) W K0 + U J0 K1 = 0
            return jv(1, U)/U + jv(0, U)*kve(1, W)/(W*kve(0, W))
        if kind == "TM":
            return jv(1, U)/U + r*jv(0, U)*kve(1, W)/(W*kve(0, W))
        S = (1/U**2 + 1/W**2)*(1/U**2 + r/W**2)
        root = np.sqrt(((1 - r)*Kt/2)**2 + nu**2*S)
        X = -(1 + r)*Kt/2 + (root if kind == "EH" else -root)   # J'_nu/(U J_nu)
        # J'_nu = J_{nu-1} - nu J_nu/U: X J_nu = J_{nu-1}/U - nu J_nu/U**2
        return jv(nu - 1, U)/U - (nu/U**2 + X)*jv(nu, U)
    return f


def vector_modes(a, n_core, n_clad, wavelength, nu_max=None, samples=4000):
    """Exact guided modes of a step-index fibre (TE, TM, HE, EH).

    Parameters
    ----------
    a : number
        Core radius (µm).
    n_core, n_clad : numbers
    wavelength : number (µm)
    nu_max : int, optional
        Largest azimuthal order of the hybrid modes.
    samples : int, optional
        Samples of U for bracketing the roots.

    Returns
    -------
    modes : list of VectorMode, sorted by decreasing n_eff.
    """

    V = 2*np.pi*a/wavelength*np.sqrt(n_core**2 - n_clad**2)
    r = n_clad**2/n_core**2
    k = 2*np.pi/wavelength
    U = np.linspace(V*1E-5, V*(1 - 1E-9), samples)
    if nu_max is None:
        nu_max = int(np.ceil(V)) + 1
    out = []
    jobs = [("TE", 0), ("TM", 0)] + [(kind, nu) for nu in range(1, nu_max + 1) for kind in ("HE", "EH")]
    for kind, nu in jobs:
        f = _vector_function(kind, nu, V, r)
        with np.errstate(all="ignore"):
            vals = f(U)
        ok = np.isfinite(vals)
        m = 0
        for i in np.nonzero(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]:
            Ur = brentq(f, U[i], U[i + 1], xtol=1E-14, rtol=1E-14)
            # reject sign changes through a jump
            if abs(f(Ur)) > 1E-6*(abs(vals[i]) + abs(vals[i + 1])):
                continue
            m += 1
            W = np.sqrt(V**2 - Ur**2)
            beta = np.sqrt(k**2*n_core**2 - (Ur/a)**2)
            out.append(VectorMode(kind, nu, m, Ur, W, V, beta/k))
    return sorted(out, key=lambda mode: -mode.n_eff)


def petermann_mfd(mode):
    """Petermann II mode field diameter 2 w (µm), with
    w**2 = 2 int u**2 r dr/int (du/dr)**2 r dr.

    Parameters
    ----------
    mode : ScalarMode or LPMode with a core radius as (mode, a)
        A ScalarMode, or a tuple (r, u) of radial samples.
    """

    if isinstance(mode, ScalarMode):
        r, u = mode.r, mode.u
    else:
        r, u = (np.asarray(v, dtype=float) for v in mode)
    du = np.gradient(u, r)
    return 2*np.sqrt(2*_trapz(u**2*r, r)/_trapz(du**2*r, r))


def bend_loss(fibre, wavelength, R, mode=None):
    """Pure bend loss of LP01 after Marcuse (1976), in dB/m.

    alpha = sqrt(pi) kappa**2 exp(-2 gamma**3 R/(3 beta**2))
            /(2 gamma**1.5 V**2 sqrt(R) K_1(gamma a)**2)

    (power loss per length), with kappa = U/a, gamma = W/a. Only the
    radiation from the bend; transition losses and the change of the mode
    shape by the bend (which raises the effective radius) are not included.

    Parameters
    ----------
    fibre : Fibre (step index)
    wavelength : number (µm)
    R : number or array
        Bend radius in mm.
    mode : LPMode, optional
        LP01 (computed if not given).
    """

    n1, n2 = float(fibre.n_core(wavelength)), float(fibre.n_clad(wavelength))
    if mode is None:
        mode = lp_modes(core_radius=fibre.a, n_core=n1, n_clad=n2, wavelength=wavelength, l_max=0)[0]
    a = fibre.a
    k = 2*np.pi/wavelength
    beta = mode.n_eff*k
    kappa, gamma = mode.U/a, mode.W/a
    Rm = np.asarray(R, dtype=float)*1E3   # µm
    alpha = (np.sqrt(np.pi)*kappa**2*np.exp(-2*gamma**3*Rm/(3*beta**2))
             /(2*gamma**1.5*mode.V**2*np.sqrt(Rm)*kv(1, gamma*a)**2))   # 1/µm
    return 10*np.log10(np.e)*alpha*1E6


# ---------------------------------------------------------------- dispersion

def effective_index(fibre, wavelength, l=0, m=1, **kwargs):
    """n_eff of LP_lm at one or several wavelengths (µm)."""
    wl = np.asarray(wavelength, dtype=float)
    out = [scalar_mode(fibre, float(w), l, m, **kwargs).n_eff for w in wl.ravel()]
    return np.reshape(out, wl.shape)[()]


def _derivatives(func, wavelength, step):
    wl = np.asarray(wavelength, dtype=float)
    f0 = func(wl)
    fp, fm = func(wl + step), func(wl - step)
    fpp, fmm = func(wl + 2*step), func(wl - 2*step)
    d1 = (8*(fp - fm) - (fpp - fmm))/(12*step)
    d2 = (-(fpp + fmm) + 16*(fp + fm) - 30*f0)/(12*step**2)
    return f0, d1, d2


def group_index(fibre, wavelength, l=0, m=1, step=0.005, **kwargs):
    """Group index N_g = n_eff - lambda dn_eff/dlambda of LP_lm."""
    f0, d1, _ = _derivatives(lambda w: effective_index(fibre, w, l, m, **kwargs), wavelength, step)
    return f0 - np.asarray(wavelength)*d1


def dispersion(fibre, wavelength, l=0, m=1, step=0.01, **kwargs):
    """Chromatic dispersion D = -(lambda/c) d**2 n_eff/dlambda**2 of LP_lm
    in ps/(nm km) (material, waveguide and profile dispersion together)."""
    _, _, d2 = _derivatives(lambda w: effective_index(fibre, w, l, m, **kwargs), wavelength, step)
    return _D_FACTOR*np.asarray(wavelength)*d2


def dispersion_slope(fibre, wavelength, l=0, m=1, step=0.02, **kwargs):
    """dD/dlambda in ps/(nm² km)."""
    wl = np.asarray(wavelength, dtype=float)
    return (dispersion(fibre, wl + step, l, m, **kwargs)
            - dispersion(fibre, wl - step, l, m, **kwargs))/(2*step*1E3)


def material_dispersion(n, wavelength, step=0.005):
    """Material dispersion -(lambda/c) n''(lambda) of a glass n(wavelength
    in µm), in ps/(nm km); also returns the group index.

    Returns
    -------
    D : array
    N_g : array
    """

    f0, d1, d2 = _derivatives(n, wavelength, step)
    wl = np.asarray(wavelength, dtype=float)
    return _D_FACTOR*wl*d2, f0 - wl*d1


def vb_derivatives(V, dV=1E-3):
    """b, d(Vb)/dV and V d**2(Vb)/dV**2 of LP01 of the step-index fibre
    (from `modes.lp_modes()`), the functions of Gloge's dispersion theory."""
    V = np.atleast_1d(np.asarray(V, dtype=float))

    def vb(v):
        return np.array([vv*lp_modes(vv, l_max=0)[0].b for vv in v])

    b = vb(V)/V
    p, mm = vb(V + dV), vb(V - dV)
    return b, (p - mm)/(2*dV), V*(p + mm - 2*b*V)/dV**2


def waveguide_dispersion(fibre, wavelength):
    """Gloge's waveguide dispersion of LP01 of a step-index fibre,
    D_wg = -(N_clad Delta/(c lambda)) V d**2(Vb)/dV**2, in ps/(nm km),
    with the group index N_clad of the cladding glass."""
    wl = np.asarray(wavelength, dtype=float)
    _, Ng2 = material_dispersion(fibre.n_clad, wl)
    V = fibre.V(wl)
    _, _, vd2 = vb_derivatives(np.atleast_1d(V))
    # in s/m² (lambda in m), 1 ps/(nm km) = 1E-6 s/m²
    out = -Ng2*fibre.delta(wl)/(2.99792458E8*wl*1E-6)*vd2.reshape(np.shape(V))*1E6
    return out[()]


def beta2_from_D(D, wavelength_nm):
    """beta_2 (ps²/km) from D (ps/(nm km)): -D lambda**2/(2 pi c)."""
    return -np.asarray(D)*np.asarray(wavelength_nm)**2/(2*np.pi*2.99792458E5)


def D_from_beta2(beta2, wavelength_nm):
    """D (ps/(nm km)) from beta_2 (ps²/km)."""
    return -np.asarray(beta2)*2*np.pi*2.99792458E5/np.asarray(wavelength_nm)**2


def wkb_group_delay(delta_mode, alpha, Delta, N_core, length_km=1.0, eps=0.0):
    """WKB group delay (ps) of the modes of a power law profile.

    Parameters
    ----------
    delta_mode : number or array
        Normalized depth (k**2 n_core**2 - beta**2)/(2 Delta k**2 n_core**2)
        of the mode, 0 on the axis, 1 at cutoff.
    alpha : number
        Profile exponent.
    Delta : number
        Relative index difference.
    N_core : number
        Group index of the core glass on the axis.
    length_km : number, optional
    eps : number, optional
        Profile dispersion parameter.
    """

    d = np.asarray(delta_mode, dtype=float)
    return length_km*_PS_PER_KM*N_core*(1 + Delta*(alpha - 2 - eps)/(alpha + 2)*d
                                        + Delta**2*(3*alpha - 2 - 2*eps)/(2*(alpha + 2))*d**2)


def wkb_delay_spread(alpha, Delta, N_core, length_km=1.0, eps=0.0, kind="rms", n=4001):
    """Intermodal delay spread (ps) of a power law fibre with all modes
    equally excited (m/M uniform): rms or full width ("max")."""
    u = (np.arange(n) + 0.5)/n
    tau = wkb_group_delay(u**(alpha/(alpha + 2)), alpha, Delta, N_core, length_km, eps)
    if kind == "rms":
        return np.std(tau)
    return tau.max() - tau.min()


def optimal_alpha(Delta, eps=0.0):
    """Profile exponent with the least rms intermodal spread (Olshansky and
    Keck): 2 + eps - Delta (4 + eps)(3 + eps)/(5 + 2 eps)."""
    return 2 + eps - Delta*(4 + eps)*(3 + eps)/(5 + 2*eps)


def pmd_dgd(n_sections, dgd_section, n_fibres=1, rng=None):
    """Differential group delay of fibres made of randomly oriented
    birefringent sections (Monte Carlo model of polarization mode
    dispersion).

    Each section is a retarder with the delay dgd_section between its
    eigenpolarizations, a random orientation and a random phase retardation.
    The DGD of the chain follows from the Jones matrix T(omega) as
    2 sqrt(det(dT/domega)). For many sections it is Maxwell distributed
    with the mean sqrt(8 n_sections/(3 pi)) dgd_section.

    Parameters
    ----------
    n_sections : int
    dgd_section : number
        DGD of one section (ps).
    n_fibres : int, optional
        Number of random fibres.
    rng : numpy Generator, optional

    Returns
    -------
    dgd : array (n_fibres,)
        In the unit of dgd_section.
    """

    rng = np.random.default_rng(rng)
    out = np.empty(n_fibres)
    for f in range(n_fibres):
        theta = rng.uniform(0, np.pi, n_sections)
        phi = rng.uniform(0, 2*np.pi, n_sections)
        T = np.eye(2, dtype=complex)
        dT = np.zeros((2, 2), dtype=complex)
        for th, ph in zip(theta, phi):
            c, s = np.cos(th), np.sin(th)
            Rm = np.array([[c, -s], [s, c]])
            D = np.diag([np.exp(0.5j*ph), np.exp(-0.5j*ph)])
            dD = np.diag([0.5j*dgd_section, -0.5j*dgd_section])@D
            J = Rm@D@Rm.T
            dJ = Rm@dD@Rm.T
            T, dT = J@T, dJ@T + J@dT
        out[f] = 2*np.sqrt(abs(np.linalg.det(dT)))
    return out


# ---------------------------------------------------------------- attenuation

_OH_BANDS = ((1.383, 0.016, 1.0), (1.240, 0.020, 0.045), (0.950, 0.012, 0.025))


def attenuation_components(wavelength, rayleigh=0.90, x_ge=0.036, oh_peak=0.0):
    """Components of the loss of a silica fibre in dB/km.

    Parameters
    ----------
    wavelength : number or array (µm)
    rayleigh : number, optional
        Rayleigh coefficient A in dB µm⁴/km (about 0.63 for pure silica,
        0.8 to 0.95 for a GeO2-doped single-mode core).
    x_ge : number, optional
        Mole fraction of GeO2 for the UV tail.
    oh_peak : number, optional
        Excess loss at the 1383 nm OH peak in dB/km (0.2 to 2 in older
        fibres, 0 for "zero water peak" fibres).

    Returns
    -------
    dict with the arrays "rayleigh", "uv", "ir", "oh" and "total".
    """

    wl = np.asarray(wavelength, dtype=float)
    out = {"rayleigh": rayleigh/wl**4,
           "uv": 154.2*x_ge/(46.6*x_ge + 60)*1E-4*np.exp(4.63/wl),
           "ir": 7.81E11*np.exp(-48.48/wl),
           "oh": sum(oh_peak*h*np.exp(-0.5*((wl - c)/w)**2) for c, w, h in _OH_BANDS)}
    out["total"] = out["rayleigh"] + out["uv"] + out["ir"] + out["oh"]
    return out


def attenuation(wavelength, rayleigh=0.90, x_ge=0.036, oh_peak=0.0):
    """Total loss of a silica fibre in dB/km, see `attenuation_components()`."""
    return attenuation_components(wavelength, rayleigh, x_ge, oh_peak)["total"]


# ---------------------------------------------------------------- Bragg gratings

def bragg_wavelength(period, n_eff, dn_dc=0.0):
    """Bragg wavelength 2 (n_eff + dn_dc) Lambda (units of the period)."""
    return 2*(n_eff + dn_dc)*period


def _section_matrix(sigma_hat, kappa, dz):
    """Transfer matrix of a uniform section, [R, S](z + dz) = T [R, S](z)."""
    sigma_hat = np.asarray(sigma_hat, dtype=complex)
    g = np.sqrt(abs(kappa)**2 - sigma_hat**2 + 0j)
    gz = g*dz
    small = np.abs(g) < 1E-12
    gs = np.where(small, 1.0, g)
    ch = np.cosh(gz)
    sh_g = np.where(small, dz, np.sinh(gz)/gs)
    T = np.empty(sigma_hat.shape + (2, 2), dtype=complex)
    T[..., 0, 0] = ch + 1j*sigma_hat*sh_g
    T[..., 0, 1] = 1j*kappa*sh_g
    T[..., 1, 0] = -1j*np.conj(kappa)*sh_g
    T[..., 1, 1] = ch - 1j*sigma_hat*sh_g
    return T


def fbg_uniform(wavelength, length, n_eff, dn_ac, period=None, bragg=None, dn_dc=0.0):
    """Reflection and transmission of a uniform fibre Bragg grating
    (analytic coupled mode solution).

    Parameters
    ----------
    wavelength : number or array
        Vacuum wavelength (µm).
    length : number
        Grating length (µm).
    n_eff : number
        Effective index of the core mode.
    dn_ac : number
        Amplitude of the index modulation of the mode (already multiplied by
        the overlap of the mode with the modulated core).
    period : number, optional
        Grating period (µm).
    bragg : number, optional
        Design wavelength 2 n_eff Lambda instead of the period.
    dn_dc : number, optional
        Mean index change, which shifts the reflection to 2 (n_eff + dn_dc) Lambda.

    Returns
    -------
    r, t : complex arrays
        Amplitude reflection (at the input, z = 0) and transmission.
    """

    if period is None:
        period = bragg/(2*n_eff)
    wl = np.asarray(wavelength, dtype=float)
    kappa = np.pi*dn_ac/wl
    sigma_hat = 2*np.pi*n_eff/wl - np.pi/period + 2*np.pi*dn_dc/wl
    g = np.sqrt(kappa**2 - sigma_hat**2 + 0j)
    gL = g*length
    den = g*np.cosh(gL) - 1j*sigma_hat*np.sinh(gL)
    return (1j*kappa*np.sinh(gL)/den)[()], (g/den)[()]


def fbg_transfer_matrix(wavelength, z_edges, n_eff, period, dn_ac, dn_dc=0.0, phase_shifts=None):
    """Reflection and transmission of a nonuniform fibre Bragg grating
    (piecewise uniform transfer matrices).

    Parameters
    ----------
    wavelength : number or array (µm)
    z_edges : array (N + 1,)
        Section boundaries along the grating (µm).
    n_eff : number
        Effective index of the core mode.
    period : number or array (N,)
        Grating period of each section (µm); a varying period is a chirp.
    dn_ac : number or array (N,)
        Modulation amplitude of each section (apodization).
    dn_dc : number or array (N,), optional
        Mean index change of each section.
    phase_shifts : dict, optional
        {section index i: phase}: the grating phase jumps by phase before
        section i (pi: a quarter-wave shifted grating).

    Returns
    -------
    r, t : complex arrays
    """

    wl = np.atleast_1d(np.asarray(wavelength, dtype=float))
    z_edges = np.asarray(z_edges, dtype=float)
    N = len(z_edges) - 1
    dz = np.diff(z_edges)
    period = np.broadcast_to(np.asarray(period, dtype=float), (N,))
    dn_ac = np.broadcast_to(np.asarray(dn_ac, dtype=float), (N,))
    dn_dc = np.broadcast_to(np.asarray(dn_dc, dtype=float), (N,))
    phase = np.zeros(N)
    for i, ph in (phase_shifts or {}).items():
        phase[i:] += ph
    M = np.broadcast_to(np.eye(2, dtype=complex), wl.shape + (2, 2)).copy()
    for i in range(N):
        kappa = np.pi*dn_ac[i]/wl*np.exp(1j*phase[i])
        sigma_hat = 2*np.pi*(n_eff + dn_dc[i])/wl - np.pi/period[i]
        T = _section_matrix(sigma_hat, kappa, dz[i])
        M = T@M
    r = -M[..., 1, 0]/M[..., 1, 1]
    t = 1/M[..., 1, 1]
    if np.ndim(wavelength) == 0:
        return r[0], t[0]
    return r, t


def fbg_delay(wavelength, r):
    """Group delay (ps) and dispersion (ps/nm) of a reflection r(lambda).

    tau = d arg(r)/d omega = -lambda**2/(2 pi c) d arg(r)/d lambda; the
    dispersion is d tau/d lambda.

    Parameters
    ----------
    wavelength : array (µm), uniformly sampled
    r : complex array

    Returns
    -------
    tau, d_tau : arrays
    """

    wl = np.asarray(wavelength, dtype=float)
    theta = np.unwrap(np.angle(r))
    tau = -wl**2/(2*np.pi*C_UM_PER_PS)*np.gradient(theta, wl)
    return tau, np.gradient(tau, wl*1E3)


def bragg_shift(bragg, strain=0.0, dT=0.0, p_e=0.22, alpha=0.55E-6, xi=6.7E-6):
    """Shift of the Bragg wavelength with axial strain and temperature,
    d lambda = lambda_B ((1 - p_e) strain + (alpha + xi) dT).

    Parameters
    ----------
    bragg : number
        Bragg wavelength (any unit, the result is in the same unit).
    strain : number or array
        Axial strain (1E-6 is one microstrain).
    dT : number or array
        Temperature change (K).
    p_e : number, optional
        Effective photo-elastic constant of silica.
    alpha, xi : numbers, optional
        Thermal expansion and thermo-optic coefficient (1/K).
    """

    return bragg*((1 - p_e)*np.asarray(strain) + (alpha + xi)*np.asarray(dT))
