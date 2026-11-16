#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optics of AR/VR displays and head-up displays: waveguide combiners with
surface relief and volume holographic gratings, pancake lenses and the
windshield of a head-up display.

Waveguide combiners
-------------------
A waveguide combiner carries the image of a small projector to the eye
inside a thin glass plate. Every field angle of the image is a plane wave.
Its transverse wave vector, in units of k_0, ::

    k = (k_x, k_y) = (sin theta, ...),   |k| <= 1 in air,

is conserved at the plate surfaces and changed by a grating of period
Lambda and groove direction phi by the grating vector ::

    G = m*lambda/Lambda*(cos phi, sin phi).

In a plate of index n the wave is guided by total internal reflection when
``1 < |k| < n``; in practice ``|k| <= n*sin(theta_max)`` with theta_max about
75° (steeper rays bounce too seldom to fill the eye box). The field of view
is the set of k that the in-coupler moves into this annulus and the
out-coupler moves back into the unit disk (`field_to_k()`, `diffract()`,
`is_guided()`, `max_fov_1d()`). Along one dimension the full field of view
is ::

    FOV = 2*arcsin((n*sin(theta_max) - 1)/2),

which makes the index of the plate the key parameter of AR glasses.

The pupil of the projector is replicated along the plate: a guided wave at
the angle theta_g hits the out-coupler every ``2*t*tan(theta_g)``
(`bounce_spacing()`), which couples out a fraction eta each time. Equal
brightness of the copies needs efficiencies that rise along the plate,
``eta_j = 1/(N - j)`` without losses (`uniform_outcoupling()`); the eye box
efficiency for an eye pupil at a given position is `eyebox_efficiency()`.

Surface relief gratings are computed rigorously with `litho.rcwa`; the
layers come from `binary_grating()`, `slanted_grating()` (staircase of
shifted ridges) and `volume_grating()` (a slanted sinusoidal index
modulation n0 + n1*cos(K.r)). Kogelnik's coupled wave theory of thick
volume holographic gratings, ::

    eta_T = sin^2(sqrt(nu^2 + xi^2))/(1 + xi^2/nu^2)            (transmission),
    eta_R = 1/(1 + (1 - xi^2/nu^2)/sinh^2(sqrt(nu^2 - xi^2)))  (reflection),

with nu = pi*n1*d/(lambda*sqrt(|c_R c_S|)) and xi = theta_d*d/(2 c_S), is
`kogelnik()`; `bragg_angle()` gives the Bragg condition.

Pancake lenses
--------------
A pancake lens folds the light path three times between a half mirror and
a reflective polarizer, with a quarter wave plate in between. With ideal
parts 25 % of the light reaches the eye along the folded path; any error
of the retarders (dispersion, angle of incidence) lets light pass the
reflective polarizer directly (a ghost, not folded and so not focused) or
fold five times instead of three. `pancake_jones()` follows both paths
with Jones matrices in a fixed lab frame (mirrors are -1, the handedness
flips by the reversed propagation). `quartz_indices()` (Ghosh 1999) and
`aplate_retardance()` (exact retardance of a plate with its optic axis in
the plane, at any angle of incidence and azimuth) give real retarders,
`achromatic_qwp()` the Pancharatnam half plus quarter wave combination.

Head-up displays
----------------
The windshield reflects the image of the HUD mirror into the eye. Its two
surfaces give two images; a wedge between them (a wedged PVB interlayer)
makes the two reflected rays meet at the virtual image.
`windshield_double_image()` traces both reflections in 2D and returns the
angular separation seen by the eye, for a given wedge and image distance;
`wedge_for_distance()` solves for the wedge that cancels it. For the
imaging of a folded HUD traced with `raytrace`, `aim_ray()` finds the ray
from a display point to an eye point and `vergence()` the 2x2 vergence
matrix of the light that enters the eye there: its eigenvalues are the
inverse image distances along the two principal directions (their
difference is the astigmatism, in diopters for mm), and the direction of
the aimed ray is where the eye sees the point (the difference between two
eyes is the binocular disparity).

References
----------
B. C. Kress, "Optical Architectures for Augmented-, Virtual-, and Mixed-Reality
Headsets", SPIE Press (2020).

H. Kogelnik, "Coupled wave theory for thick hologram gratings", Bell Syst.
Tech. J. 48, 2909-2947 (1969).

G. Ghosh, "Dispersion-equation coefficients for the refractive index and
birefringence of calcite and quartz crystals", Opt. Commun. 163, 95-102
(1999).

S. Pancharatnam, "Achromatic combinations of birefringent plates", Proc.
Indian Acad. Sci. A 41, 130-136 (1955).

T. L. Wong, Z. Yun, G. Ambur and J. Etter, "Folded optics with birefringent
reflective polarizers", Proc. SPIE 10335, 103350E (2017).

J. Wood, "Head-up displays", in: Handbook of Visual Display Technology,
Springer (2012).
"""

import numpy as np

from .litho.rcwa import Layer
from .polarization.jones import J_WP, J_polarizer


__all__ = ["field_to_k", "k_to_field", "diffract", "direction", "is_guided", "guide_angle",
           "max_fov_1d", "symmetric_period", "bounce_spacing", "uniform_outcoupling",
           "replicated_powers", "eyebox_efficiency",
           "binary_grating", "slanted_grating", "volume_grating",
           "bragg_angle", "kogelnik",
           "quartz_indices", "aplate_retardance", "achromatic_qwp", "pancake_jones",
           "windshield_double_image", "wedge_for_distance", "hud_system", "aim_ray", "vergence"]


# --------------------------------------------------------------------------
# waveguide combiners: k-space
# --------------------------------------------------------------------------

def field_to_k(theta_x, theta_y=0.0):
    """Transverse wave vector (in units of k_0) of a field direction.

    Parameters
    ----------
    theta_x, theta_y : numbers or arrays
        Horizontal and vertical field angles (radians): the direction is
        along (tan theta_x, tan theta_y, 1).

    Returns
    -------
    k_x, k_y : arrays
        Direction cosines along x and y (k/k_0 in air).
    """

    tx, ty = np.tan(theta_x), np.tan(theta_y)
    norm = np.sqrt(1 + tx**2 + ty**2)

    return tx/norm, ty/norm


def k_to_field(k_x, k_y, n=1.0):
    """Field angles of the transverse wave vector (k_x, k_y) in a medium of
    index n (inverse of `field_to_k()` for n = 1).

    Parameters
    ----------
    k_x, k_y : numbers or arrays
        Transverse wave vector in units of k_0.
    n : number, optional
        Index of the medium.

    Returns
    -------
    theta_x, theta_y : arrays
        NaN where the wave is evanescent.
    """

    k_x, k_y = np.asarray(k_x, dtype=float), np.asarray(k_y, dtype=float)
    with np.errstate(invalid="ignore"):
        k_z = np.sqrt(n**2 - k_x**2 - k_y**2)

    return np.arctan2(k_x, k_z), np.arctan2(k_y, k_z)


def diffract(k_x, k_y, period, wavelength, order=1, groove_angle=0.0):
    """Add the grating vector to the transverse wave vector.

    Parameters
    ----------
    k_x, k_y : numbers or arrays
        Transverse wave vector in units of k_0.
    period, wavelength : numbers
        In the same unit.
    order : int, optional
    groove_angle : number, optional
        Direction of the grating vector (radians from x); the grooves run
        perpendicular to it.

    Returns
    -------
    k_x, k_y : arrays
        k + order*wavelength/period*(cos(groove_angle), sin(groove_angle)).
    """

    g = order*wavelength/period

    return (np.asarray(k_x, dtype=float) + g*np.cos(groove_angle),
            np.asarray(k_y, dtype=float) + g*np.sin(groove_angle))


def direction(k_x, k_y, n=1.0, sign=1.0):
    """Unit direction vector of a plane wave with the transverse wave
    vector (k_x, k_y) in a medium of index n.

    Parameters
    ----------
    k_x, k_y : numbers or arrays
    n : number, optional
    sign : {1, -1}, optional
        Direction of travel along z.

    Returns
    -------
    d : array, shape (..., 3)
        NaN for evanescent waves.
    """

    k_x, k_y = np.broadcast_arrays(np.asarray(k_x, dtype=float), np.asarray(k_y, dtype=float))
    with np.errstate(invalid="ignore"):
        k_z = sign*np.sqrt(n**2 - k_x**2 - k_y**2)

    d = np.stack([k_x, k_y, k_z], axis=-1)/n
    d[np.isnan(k_z)] = np.nan

    return d


def is_guided(k_x, k_y, n, max_angle=np.radians(75.0)):
    """Whether a wave with the transverse wave vector k is guided by total
    internal reflection in a plate of index n in air, at most max_angle
    from the normal.

    Parameters
    ----------
    k_x, k_y : numbers or arrays
    n : number
    max_angle : number, optional
        Largest useful angle in the plate (radians).

    Returns
    -------
    guided : bool array
    """

    k = np.hypot(k_x, k_y)

    return (k > 1.0) & (k <= n*np.sin(max_angle))


def guide_angle(k_x, k_y, n):
    """Angle of a guided wave to the normal of the plate.

    Parameters
    ----------
    k_x, k_y : numbers or arrays
    n : number

    Returns
    -------
    theta : array
        arcsin(|k|/n), NaN if |k| > n.
    """

    with np.errstate(invalid="ignore"):
        return np.arcsin(np.hypot(k_x, k_y)/n)


def max_fov_1d(n, max_angle=np.radians(75.0)):
    """Largest full field of view along the grating vector of a 1D
    waveguide combiner (in-coupler and out-coupler with opposite grating
    vectors).

    The field maps onto the interval of k of length ``2*sin(FOV/2)``, which
    must fit between the TIR limit 1 and ``n*sin(max_angle)``.

    Parameters
    ----------
    n : number or array
        Index of the plate.
    max_angle : number, optional

    Returns
    -------
    fov : array
        Full field of view (radians); NaN if nothing is guided.
    """

    with np.errstate(invalid="ignore"):
        return 2*np.arcsin((np.asarray(n, dtype=float)*np.sin(max_angle) - 1)/2)


def symmetric_period(n, wavelength, max_angle=np.radians(75.0)):
    """Grating period that centres the field of view in the guided annulus,
    lambda/period = (1 + n*sin(max_angle))/2.

    Parameters
    ----------
    n : number
    wavelength : number
    max_angle : number, optional

    Returns
    -------
    period : number
        In the unit of the wavelength.
    """

    return wavelength/((1 + n*np.sin(max_angle))/2)


def bounce_spacing(thickness, theta):
    """Distance between two hits of the same surface by a guided wave.

    Parameters
    ----------
    thickness : number
    theta : number or array
        Angle to the normal in the plate.

    Returns
    -------
    spacing : array
        2*thickness*tan(theta).
    """

    return 2*thickness*np.tan(theta)


def uniform_outcoupling(n_out, loss=0.0, last=1.0):
    """Out-coupling efficiencies that give n_out copies of equal power.

    The copy j couples out the fraction eta_j of the power left in the
    guide; between two hits a fraction ``loss`` is lost (absorption,
    scatter). Solved backwards from the last hit, whose efficiency is
    ``last``.

    Parameters
    ----------
    n_out : int
        Number of out-coupling hits.
    loss : number, optional
        Fractional loss per bounce.
    last : number, optional
        Efficiency of the last hit (1 uses all the light).

    Returns
    -------
    eta : array (n_out,)
    """

    eta = np.empty(n_out)
    eta[-1] = last
    # power arriving at hit j relative to the equal output P: p_j*eta_j = P,
    # p_{j+1} = p_j*(1 - eta_j)*(1 - loss)
    p_next = 1.0/last
    for j in range(n_out - 2, -1, -1):
        p_j = 1.0 + p_next/(1 - loss)
        eta[j] = 1.0/p_j
        p_next = p_j

    return eta


def replicated_powers(eta, loss=0.0):
    """Powers of the copies of the pupil for the out-coupling efficiencies
    eta (unit power in the guide).

    Parameters
    ----------
    eta : sequence
        Efficiency of each hit.
    loss : number, optional
        Fractional loss per bounce.

    Returns
    -------
    powers : array
        Power coupled out at each hit.
    """

    eta = np.asarray(eta, dtype=float)
    left = np.concatenate([[1.0], np.cumprod((1 - eta[:-1])*(1 - loss))])

    return left*eta


def eyebox_efficiency(x_eye, pupil, positions, powers, beam_width):
    """Fraction of the power of a field that enters an eye pupil (1D).

    The copies of the pupil are beams of width beam_width centred at
    positions (in the plane of the eye, i.e. shifted by eye_relief*tan of
    the field angle), each with uniform irradiance powers/beam_width. The
    eye pupil is the interval x_eye +- pupil/2.

    Parameters
    ----------
    x_eye : number or array
        Centre of the eye pupil.
    pupil : number
        Eye pupil diameter.
    positions : array
        Centres of the copies.
    powers : array
        Powers of the copies.
    beam_width : number
        Width of each copy (the projector pupil).

    Returns
    -------
    fraction : array
        Power entering the pupil (in units of the power in the guide).
    """

    x_eye = np.asarray(x_eye, dtype=float)[..., None]
    lo = np.maximum(x_eye - pupil/2, positions - beam_width/2)
    hi = np.minimum(x_eye + pupil/2, positions + beam_width/2)

    return np.sum(np.clip(hi - lo, 0.0, None)*powers/beam_width, axis=-1)


# --------------------------------------------------------------------------
# grating layers for litho.rcwa
# --------------------------------------------------------------------------

def binary_grating(period, depth, fill, n_ridge, n_groove=1.0):
    """Binary surface relief grating as one lamellar layer.

    Parameters
    ----------
    period, depth : numbers
    fill : number
        Fraction of the period filled by the ridge.
    n_ridge, n_groove : complex, optional
        Indices of the ridge and of the grooves.

    Returns
    -------
    layers : list of litho.rcwa.Layer
    """

    return [Layer(depth, n_groove, [(0.0, fill*period, n_ridge)])]


def slanted_grating(period, depth, fill, slant, n_ridge, n_groove=1.0, n_slices=16,
                    top_down=True):
    """Slanted surface relief grating as a staircase of lamellar layers.

    The ridges (width fill*period) lean by the slant angle: at the depth z
    below the top, the ridge centre is at z*tan(slant). Slanted ridges
    break the symmetry between the orders +1 and -1, which is how
    surface relief in-couplers send most light one way along the guide.

    Parameters
    ----------
    period, depth, fill : numbers
    slant : number
        Slant of the ridges from the normal (radians), positive towards +x
        with depth.
    n_ridge, n_groove : complex, optional
    n_slices : int, optional
        Number of layers of the staircase.
    top_down : bool, optional
        Order of the layers: True from the top of the ridges (light
        incident from the air side of the grating), False from the base.

    Returns
    -------
    layers : list of litho.rcwa.Layer
    """

    dz = depth/n_slices
    z = (np.arange(n_slices) + 0.5)*dz
    layers = [Layer(dz, n_groove, [(zi*np.tan(slant), fill*period, n_ridge)]) for zi in z]

    return layers if top_down else layers[::-1]


def volume_grating(period_x, thickness, n0, n1, slant_phase=0.0, n_slices=32, n_segments=32):
    """Volume grating n(x, z) = n0 + n1*cos(K_x x + K_z z) as a stack of
    lamellar layers with piecewise constant index.

    Parameters
    ----------
    period_x : number
        Period along the surface, 2*pi/K_x.
    thickness : number
    n0, n1 : numbers
        Mean index and amplitude of the modulation.
    slant_phase : number, optional
        K_z*thickness: the phase by which the fringes shift from the top to
        the bottom (0 for fringes perpendicular to the surface).
    n_slices : int, optional
        Layers along z (one is enough without slant).
    n_segments : int, optional
        Segments per period.

    Returns
    -------
    layers : list of litho.rcwa.Layer
        From z = 0 (top) down.
    """

    if slant_phase == 0.0:
        n_slices = 1
    dz = thickness/n_slices
    w = period_x/n_segments
    xc = (np.arange(n_segments) + 0.5)*w
    layers = []
    for s in range(n_slices):
        phase = slant_phase*(s + 0.5)/n_slices
        n_seg = n0 + n1*np.cos(2*np.pi*xc/period_x + phase)
        layers.append(Layer(dz, n0, [(c, w*(1 - 1e-9), nn) for c, nn in zip(xc, n_seg)]))

    return layers


# --------------------------------------------------------------------------
# Kogelnik's coupled wave theory
# --------------------------------------------------------------------------

def bragg_angle(period, wavelength, n, slant=0.0):
    """Angle of incidence in the medium (from the surface normal) that
    satisfies the Bragg condition cos(slant - theta) = K/(2 beta).

    Parameters
    ----------
    period : number
        Fringe spacing Lambda (along the grating vector).
    wavelength : number
        Vacuum wavelength.
    n : number
        Mean index of the grating.
    slant : number, optional
        Angle phi of the grating vector to the surface normal (pi/2 for an
        unslanted transmission grating, 0 for an unslanted reflection
        grating).

    Returns
    -------
    theta : number
        The solution theta = slant - arccos(lambda/(2 n Lambda)).
    """

    return slant - np.arccos(wavelength/(2*n*period))


def kogelnik(theta, wavelength, period, thickness, n0, n1, slant=np.pi/2, polarization="s"):
    """Diffraction efficiency of a thick lossless volume grating after
    Kogelnik (1969).

    Two-wave coupled wave theory: the incident wave R at theta in the
    medium (index n0, no boundary reflections) and the signal S = R - K,
    with the grating vector K = 2 pi/Lambda (sin slant, cos slant) and the
    sinusoidal modulation n1. Transmission (c_S > 0) and reflection
    (c_S < 0) gratings are distinguished automatically.

    Parameters
    ----------
    theta : number or array
        Angle of incidence in the medium (radians from the normal).
    wavelength : number or array
        Vacuum wavelength.
    period : number
        Fringe spacing Lambda.
    thickness : number
    n0, n1 : numbers
    slant : number, optional
        Angle of K from the surface normal (default pi/2: unslanted
        transmission grating).
    polarization : {"s", "p"}, optional
        p multiplies the coupling by cos(2 (theta - slant)) (the angle
        between R and S).

    Returns
    -------
    eta : array
        Diffraction efficiency (power in S / power in R, with the
        projections c_S/c_R).
    """

    theta = np.asarray(theta, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    beta = 2*np.pi*n0/wavelength
    K = 2*np.pi/period
    c_R = np.cos(theta)
    c_S = np.cos(theta) - K/beta*np.cos(slant)
    dephase = K*np.cos(slant - theta) - K**2*wavelength/(4*np.pi*n0)
    kappa = np.pi*n1/wavelength
    if polarization == "p":
        kappa = kappa*np.cos(2*(theta - slant))
    elif polarization != "s":
        raise ValueError("polarization must be 's' or 'p'")
    nu = kappa*thickness/np.sqrt(np.abs(c_R*c_S) + 0j)
    xi = dephase*thickness/(2*c_S)
    with np.errstate(invalid="ignore", divide="ignore"):
        trans = np.sin(np.sqrt(nu**2 + xi**2))**2/(1 + xi**2/nu**2)
        s = np.sinh(np.sqrt(nu**2 - xi**2 + 0j))**2
        refl = 1/(1 + (1 - xi**2/nu**2)/s)
        # exactly on the edge nu = xi: limit nu^2/(1 + nu^2)
        refl = np.where(np.abs(s) < 1e-300, nu**2/(1 + nu**2), refl)

    return np.real(np.where(c_S > 0, trans, refl))


# --------------------------------------------------------------------------
# retarders and pancake lenses
# --------------------------------------------------------------------------

def quartz_indices(wavelength_nm):
    """Ordinary and extraordinary indices of crystalline quartz.

    Sellmeier fit of G. Ghosh, Opt. Commun. 163, 95 (1999), valid from 0.198
    to 2.05 µm (via refractiveindex.info).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    n_o, n_e : arrays
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2
    n_o = np.sqrt(1.28604141 + 1.07044083*l2/(l2 - 1.00585997e-2) + 1.10202242*l2/(l2 - 100.0))
    n_e = np.sqrt(1.28851804 + 1.09509924*l2/(l2 - 1.02101864e-2) + 1.15662475*l2/(l2 - 100.0))

    return n_o, n_e


def aplate_retardance(thickness, wavelength, n_o, n_e, sin_theta=0.0, azimuth=0.0):
    """Retardance of a plate with its optic axis in the plane of the plate
    (an A-plate), for any angle of incidence.

    With the optic axis along x and the plane of incidence at the azimuth
    psi from it, the tangential wave vector is s (cos psi, sin psi) with
    s = n_in*sin(theta_in); the normal components of the two eigenwaves
    are ::

        k_z,o = sqrt(n_o^2 - s^2),
        k_z,e = n_e*sqrt(1 - s^2 cos^2 psi/n_o^2 - s^2 sin^2 psi/n_e^2),

    and the retardance is 2 pi d/lambda (k_z,e - k_z,o).

    Parameters
    ----------
    thickness, wavelength : numbers or arrays
        In the same unit.
    n_o, n_e : numbers or arrays
    sin_theta : number or array, optional
        Tangential wave vector s (sine of the angle of incidence in air).
    azimuth : number or array, optional
        Angle psi between the optic axis and the plane of incidence.

    Returns
    -------
    delta : array
        Retardance (radians), extraordinary minus ordinary.
    """

    s2 = np.asarray(sin_theta, dtype=float)**2
    kz_o = np.sqrt(n_o**2 - s2)
    kz_e = n_e*np.sqrt(1 - s2*np.cos(azimuth)**2/n_o**2 - s2*np.sin(azimuth)**2/n_e**2)

    return 2*np.pi*np.asarray(thickness)/np.asarray(wavelength)*(kz_e - kz_o)


def achromatic_qwp(delta_hwp, delta_qwp, angle=0.0):
    """Jones matrix of Pancharatnam's achromatic quarter wave plate: a half
    wave plate at 15° followed by a quarter wave plate at 75° (both
    relative to the axis of the combination at angle).

    Parameters
    ----------
    delta_hwp, delta_qwp : numbers
        Retardances of the two plates (pi and pi/2 at the design
        wavelength).
    angle : number, optional
        Orientation of the combination (radians).

    Returns
    -------
    J : (2, 2) complex array
        Turns light polarized linearly along `angle` into circular light,
        and circular light back into linear, almost independently of the
        wavelength. It is not a pure linear retarder: the linear output
        is rotated, so its orientation in a system is found by
        optimization (e.g. -30° in the pancake of `pancake_jones()`).
    """

    return J_WP(delta_qwp, angle + np.radians(75.0)) @ J_WP(delta_hwp, angle + np.radians(15.0))


def pancake_jones(J_qwp, J_input=None, r_half=np.sqrt(0.5), t_half=np.sqrt(0.5), leak=0.0,
                  extinction=0.0):
    """Jones amplitudes of the folded image and of the ghosts of a pancake
    lens.

    Lab frame (x, y fixed for both directions of travel): a retarder
    passed backwards has the transposed Jones matrix (reciprocity; the
    same matrix for a single linear retarder, the reversed order for a
    stack of plates). The display
    emits linear polarization along x, which J_input (the quarter wave
    plate on the display, default ideal at 45°) turns circular. The light
    then passes:

    1. the half mirror (amplitude transmission t_half),
    2. the quarter wave plate J_qwp, which should turn the circular light
       linear along x,
    3. the reflective polarizer, which reflects x (amplitude -1, minus
       ``leak``) and transmits y (amplitude 1, transmission of x: leak),
    4. back through J_qwp, reflected by the half mirror (-r_half), through
       J_qwp again, now along y, and through the reflective polarizer and
       a clean-up absorbing polarizer along y (amplitude extinction for x)
       to the eye.

    Light that passes the reflective polarizer in step 3 is the direct
    ghost (one pass, no fold, not focused by the half mirror); light that
    is reflected again at the end of step 4 makes another round trip
    (the 5-pass ghost, focused by the mirror twice).

    Parameters
    ----------
    J_qwp : (2, 2) complex array
        Quarter wave plate between half mirror and reflective polarizer.
    J_input : (2, 2) complex array, optional
        Retarder on the display (ideal quarter wave plate at 45° if None).
    r_half, t_half : numbers, optional
        Amplitude reflection and transmission of the half mirror.
    leak : number, optional
        Amplitude transmission of the reflective polarizer for x (and
        reduction of its reflection).
    extinction : number, optional
        Amplitude transmission of the clean-up polarizer for x.

    Returns
    -------
    dict with keys "signal", "direct", "five_pass" (Jones vectors at the eye)
    and "T_signal", "T_direct", "T_five_pass" (their powers, relative to
    the display).
    """

    if J_input is None:
        J_input = J_WP(np.pi/2, np.pi/4)
    E0 = J_input @ np.array([1.0, 0.0], dtype=complex)
    RP_t = J_polarizer(leak, 1.0).astype(complex)                  # transmits y
    RP_r = -J_polarizer(np.sqrt(max(1 - leak**2, 0.0)), 0.0).astype(complex)  # reflects x
    clean = J_polarizer(extinction, 1.0).astype(complex)
    HM_r = -r_half
    J_back = J_qwp.T                               # reciprocity: reversed order of the plates
    E1 = J_qwp @ (t_half*E0)                       # at the reflective polarizer, first time
    direct = clean @ RP_t @ E1
    E2 = J_qwp @ (HM_r*(J_back @ (RP_r @ E1)))     # at the reflective polarizer, second time
    signal = clean @ RP_t @ E2
    E3 = J_qwp @ (HM_r*(J_back @ (RP_r @ E2)))
    five = clean @ RP_t @ E3

    def power(E):
        return float(np.sum(np.abs(E)**2))

    return {"signal": signal, "direct": direct, "five_pass": five, "T_signal": power(signal),
            "T_direct": power(direct), "T_five_pass": power(five)}


# --------------------------------------------------------------------------
# head-up displays
# --------------------------------------------------------------------------

def _reflect2(d, nrm):
    return d - 2*np.dot(d, nrm)*nrm


def _refract2(d, nrm, n1, n2):
    """Refract the unit vector d at the unit normal nrm (any orientation)."""

    if np.dot(d, nrm) > 0:
        nrm = -nrm
    c = -np.dot(d, nrm)
    r = n1/n2
    k = 1 - r**2*(1 - c**2)
    if k < 0:
        return np.full(2, np.nan)

    return r*d + (r*c - np.sqrt(k))*nrm


def _hit_line(p, d, q, t):
    """Intersection of the ray p + s d with the line through q along t."""

    A = np.array([[d[0], -t[0]], [d[1], -t[1]]])
    s, _ = np.linalg.solve(A, q - p)

    return p + s*d


def _windshield_rays(theta_in, thickness, n, wedge, distance, eye_distance):
    """Return the functions mapping an eye ray elevation to the image points
    (primary and secondary) in the 2D windshield model."""

    # eye at the origin; inner surface through (0, eye_distance) rotated so
    # that the central eye ray along +y meets it at theta_in
    q0 = np.array([0.0, eye_distance])
    t_in = np.array([np.cos(theta_in), np.sin(theta_in)])      # along the inner surface
    nrm_in = np.array([-np.sin(theta_in), np.cos(theta_in)])    # towards the outside
    # outer surface: through q0 + thickness*nrm_in, rotated by the wedge
    # about that point (thickness measured at the central hit)
    rot = np.array([[np.cos(wedge), -np.sin(wedge)], [np.sin(wedge), np.cos(wedge)]])
    t_out, nrm_out = rot @ t_in, rot @ nrm_in
    q1 = q0 + thickness*nrm_in

    def primary(a):
        d = np.array([np.sin(a), np.cos(a)])
        p = _hit_line(np.zeros(2), d, q0, t_in)
        return p, _reflect2(d, nrm_in)

    def secondary(a):
        d = np.array([np.sin(a), np.cos(a)])
        p = _hit_line(np.zeros(2), d, q0, t_in)
        d1 = _refract2(d, nrm_in, 1.0, n)
        p1 = _hit_line(p, d1, q1, t_out)
        d2 = _reflect2(d1, nrm_out)
        p2 = _hit_line(p1, d2, q0, t_in)
        d3 = _refract2(d2, nrm_in, n, 1.0)
        return p2, d3

    return primary, secondary


def windshield_double_image(theta_in, thickness, n, wedge, distance, eye_distance=800.0):
    """Angular separation of the two reflections of a windshield, seen by
    the eye (2D).

    The HUD forms a virtual image at the distance `distance` from the eye
    (measured along the reflected chief ray, i.e. eye to windshield plus
    windshield to image). The light reaching the windshield from the HUD
    therefore diverges from a point Q, the mirror image of the virtual
    image in the inner surface. The eye sees Q along the ray reflected at
    the inner surface and, displaced, along the ray that is refracted into
    the glass, reflected at the outer surface (tilted by the wedge) and
    refracted back. This function aims both rays at Q and returns the
    angle between them at the eye.

    Parameters
    ----------
    theta_in : number
        Angle of incidence of the central eye ray on the inner surface
        (radians), typically 60° to 65°.
    thickness : number
        Glass thickness at the central hit.
    n : number
        Index of the glass (laminated glass and PVB are close, about 1.52).
    wedge : number
        Wedge angle between the surfaces (radians); positive thickens the
        glass upwards (towards the roof for the usual layout).
    distance : number
        Virtual image distance from the eye (same unit as thickness);
        np.inf for an image at infinity.
    eye_distance : number, optional
        Distance from the eye to the windshield along the central ray.

    Returns
    -------
    separation : float
        Angle of the secondary image minus the angle of the primary
        (radians, positive: the ghost appears above in the rotated frame
        of the model).
    """

    from scipy.optimize import brentq

    primary, secondary = _windshield_rays(theta_in, thickness, n, wedge, distance, eye_distance)
    # Q: the virtual image point seen along the central ray, mirrored into
    # the incident space: the reflected central ray leaves the hit point
    p0, d0 = primary(0.0)
    if np.isinf(distance):
        def miss(f, a):
            p, d = f(a)
            return d[0]*d0[1] - d[1]*d0[0]      # parallel to the primary direction
    else:
        Q = p0 + (distance - eye_distance)*d0

        def miss(f, a):
            p, d = f(a)
            return d[0]*(Q - p)[1] - d[1]*(Q - p)[0]
    a_p = 0.0
    a_s = brentq(lambda a: miss(secondary, a), -0.05, 0.05, xtol=1e-14)

    return a_s - a_p


def wedge_for_distance(theta_in, thickness, n, distance, eye_distance=800.0):
    """Wedge angle that superimposes the two reflections of a windshield
    for a virtual image at `distance` (see `windshield_double_image()`).

    Parameters
    ----------
    theta_in, thickness, n, distance, eye_distance : numbers
        As in `windshield_double_image()`.

    Returns
    -------
    wedge : float
        Radians.
    """

    from scipy.optimize import brentq

    return brentq(lambda w: windshield_double_image(theta_in, thickness, n, w, distance,
                                                    eye_distance), -2e-3, 2e-3, xtol=1e-12)


def hud_system(R_mirror=-700.0, R_windshield=8000.0, mirror_zernike=None, mirror_angle=np.radians(20.0),
               windshield_angle=np.radians(62.0), display_distance=250.0, mirror_to_windshield=400.0,
               windshield_to_eye=900.0, mirror_radius=150.0, norm_radius=100.0, wavelength=550e-6,
               windshield_zernike=None):
    """Ray traced model of a windshield head-up display (mm).

    Display -> concave HUD mirror (a `raytrace.ZernikeSurface`, the
    freeform) -> windshield (a mirror, its inner surface) -> the plane of
    the eye box, folded in the y-z plane. The fold angles are those of the
    chief ray of the base layout (the HUD mirror a sphere of radius
    R_mirror without Zernike terms); changing the radius or the Zernike
    terms afterwards keeps the layout, as in a real package, and moves the
    image instead. The image plane is the eye box, its local z axis
    pointing into the eye; use `vergence()` to see what the eye sees.

    Parameters
    ----------
    R_mirror : number, optional
        Vertex radius of the HUD mirror (negative: concave for the light
        coming from the display).
    R_windshield : number, optional
        Radius of the windshield (positive: concave as seen from the
        driver, as real windshields are).
    mirror_zernike : dict, optional
        Fringe Zernike departures {j: c_j} (mm) of the HUD mirror, over
        norm_radius.
    mirror_angle, windshield_angle : numbers, optional
        Angles of incidence of the chief ray (radians).
    display_distance, mirror_to_windshield, windshield_to_eye : numbers,
        optional
        Distances along the chief ray (mm).
    mirror_radius : number, optional
        Semi diameter of the HUD mirror (the stop).
    norm_radius : number, optional
        Normalization radius of the Zernike terms.
    wavelength : number, optional
        In mm.
    windshield_zernike : dict, optional
        Fringe Zernike departures of the windshield (over 300 mm).

    Returns
    -------
    raytrace.System
        Object: the display at display_distance in front of the mirror
        (global start points (x, y, -display_distance)).
    """

    from . import raytrace as rt
    from .spectrometers import fold_after

    def fold(surfaces, element, tilt, distance, sigma):
        surfaces = surfaces + [rt.CoordinateBreak(tilt=(tilt, 0.0, 0.0)), element]
        a = fold_after(surfaces, 1, wavelength, display_distance, sigma)
        return surfaces + [rt.CoordinateBreak(tilt=(a, 0.0, 0.0), thickness=sigma*distance)]

    base = rt.Surface(R_mirror, mirror=True, semi_diameter=mirror_radius)
    surfaces = fold([], base, mirror_angle, mirror_to_windshield, -1)
    surfaces = fold(surfaces, rt.Surface(R_windshield, mirror=True), windshield_angle,
                    windshield_to_eye, 1)
    surfaces[1] = rt.ZernikeSurface(R_mirror, mirror=True, semi_diameter=mirror_radius,
                                    zernike=mirror_zernike or {}, norm_radius=norm_radius,
                                    label="HUD mirror")
    surfaces[4] = rt.ZernikeSurface(R_windshield, mirror=True, zernike=windshield_zernike or {},
                                    norm_radius=300.0, label="windshield")

    return rt.System(surfaces, 1, wavelength, object_distance=display_distance, ray_aiming=False)


def _image_local(system, bundle):
    o_img, R_img = system.frames()[1]
    P = (bundle.points[-1] - o_img) @ R_img
    d = bundle.directions[-1] @ R_img

    return P, d


def aim_ray(system, start, target, d0, tol=1e-9, max_iter=30):
    """Direction from a point in object space through `system` to a point
    of its image plane.

    Parameters
    ----------
    system : raytrace.System
    start : 3 vector
        Start point (global coordinates, in front of the first surface).
    target : 2 vector
        (x, y) in the local frame of the image plane.
    d0 : 3 vector
        Initial guess of the direction.
    tol : number, optional
        Tolerance on the image position.
    max_iter : int, optional

    Returns
    -------
    d : (3,) array
        Unit direction of the aimed ray.
    """

    from .raytrace import Rays

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    d = np.asarray(d0, dtype=float)/np.linalg.norm(d0)
    # two directions perpendicular to d
    u = np.cross(d, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(d, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    h = 1e-6
    a = np.zeros(2)
    for _ in range(max_iter):
        dirs = np.array([d + a[0]*u + a[1]*v, d + (a[0] + h)*u + a[1]*v, d + a[0]*u + (a[1] + h)*v])
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        b = system.trace(Rays(np.tile(start, (3, 1)), dirs, np.zeros(3)), clip=False)
        P = _image_local(system, b)[0][:, :2]
        if not np.all(np.isfinite(P)):
            raise ValueError("The aimed ray is lost.")
        err = P[0] - target
        if np.hypot(*err) < tol:
            break
        J = np.column_stack([(P[1] - P[0])/h, (P[2] - P[0])/h])
        a = a - np.linalg.solve(J, err)
    dd = d + a[0]*u + a[1]*v

    return dd/np.linalg.norm(dd)


def vergence(system, start, target, d0, h=1e-4):
    """Vergence matrix and viewing direction of the light from a point that
    enters the eye at a point of the image plane.

    The image plane of `system` is the plane of the eye pupil, its local z
    axis pointing into the eye. The rays around the aimed ray arrive with
    the local slopes (u, v) = (d_x/d_z, d_y/d_z) at the positions (x, y);
    the derivative V = d(u, v)/d(x, y) is the vergence matrix: light that
    diverges from a point at the distance D in front of the eye has
    V = identity/D.

    Parameters
    ----------
    system : raytrace.System
    start : 3 vector
        Object point (global).
    target : 2 vector
        Eye point in the local frame of the image plane.
    d0 : 3 vector
        Initial direction guess for `aim_ray()`.
    h : number, optional
        Angular step of the finite differences (radians).

    Returns
    -------
    V : (2, 2) array
        Vergence matrix (1/length).
    slope : (2,) array
        (u, v) of the aimed ray: the eye sees the point in the direction
        opposite to it.
    """

    from .raytrace import Rays

    d = aim_ray(system, start, target, d0)
    u = np.cross(d, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(d, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    dirs = np.array([d, d + h*u, d + h*v, d - h*u, d - h*v])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    b = system.trace(Rays(np.tile(np.asarray(start, float), (5, 1)), dirs, np.zeros(5)), clip=False)
    P, D = _image_local(system, b)
    slope = D[:, :2]/D[:, 2:3]
    A = np.column_stack([(P[1, :2] - P[3, :2])/(2*h), (P[2, :2] - P[4, :2])/(2*h)])
    B = np.column_stack([(slope[1] - slope[3])/(2*h), (slope[2] - slope[4])/(2*h)])

    return B @ np.linalg.inv(A), slope[0]
