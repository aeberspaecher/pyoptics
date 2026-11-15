#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ray traced spectrometers: the Czerny-Turner and the Offner imaging
spectrometer, built from `raytrace.GratingSurface` (notebooks 110 and 114).

The builders return `raytrace.System` objects whose object is the
entrance slit (a finite object at the distance of the collimator's focal
length; object points (x, y) with x along the slit) and whose image plane
is the detector. The dispersion lies in the y-z plane of every frame:
the spectrum spreads along y of the detector, the slit is imaged along x.
A system is built for one design wavelength, which fixes the geometry
(the fold angles of the coordinate breaks); `System.with_wavelength()`
then traces any other wavelength through the same hardware, so the chief
rays of different wavelengths land at different detector positions.

Folds follow the pattern of `raytrace`: a `CoordinateBreak(tilt=a)` in front
of a mirror or grating, and one behind it that turns the axis frame onto
the reflected (or diffracted) chief ray at the design wavelength, with
the thickness sign of the unfolded system (negative after an odd number
of reflections). The angle of the second break is found by tracing the
chief ray (`fold_after()`), so curved gratings and arbitrary angles need no
special formulas.

Closed forms used in the notebooks and tests:

- Grating equation (reflection, `echelle` sign convention) ::

      m lambda = d (sin(alpha) + sin(beta)),

  in terms of the grating rotation theta and the half included angle K
  (alpha = theta + K, beta = theta - K): m lambda = 2 d sin(theta) cos(K).
- Linear dispersion on the detector ::

      dy/dlambda = m f_cam/(d cos(beta)),

  and the anamorphic magnification of the slit width cos(alpha)/cos(beta).
- Shafer's condition for zero coma of a Czerny-Turner at the design
  wavelength (Shafer, Mégill and Droppleman 1964) ::

      sin(phi_2)/sin(phi_1) = (R_2/R_1)**2 (cos(phi_2)/cos(phi_1))**3
                               (cos(alpha)/cos(beta))**3,

  with the off-axis angles phi_1, phi_2 of the collimator and camera
  mirrors (half the fold angle at each mirror) and their radii R_1, R_2.
  The coma of a mirror grows as W**3 sin(phi)/R**2 with the beam width
  W, and the grating changes the width of the collimated beam by
  cos(beta)/cos(alpha), hence the last factor.

Units: millimetres, wavelengths in mm in the systems (as `raytrace`),
groove densities in lines per mm.

References
----------
M. Czerny and A. F. Turner, "Über den Astigmatismus bei Spiegelspektrometern",
Z. Phys. 61, 792-797 (1930).

A. B. Shafer, L. R. Megill and L. Droppleman, "Optimization of the
Czerny-Turner spectrometer", J. Opt. Soc. Am. 54, 879-887 (1964).

A. Offner, "New concepts in projection mask aligners", Opt. Eng. 14,
130-132 (1975); P. Mouroulis, R. O. Green and T. G. Chrien, "Design of
pushbroom imaging spectrometers for optimum recovery of spectroscopic and
spatial information", Appl. Opt. 39, 2210-2220 (2000).

C. Palmer, "Diffraction Grating Handbook", 8th ed., MKS Newport (2020).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt


def grating_rotation(wavelength, groove_density, order=1, included_angle=0.0):
    """Grating angles for a wavelength at a fixed included angle.

    Parameters
    ----------
    wavelength : number
        Wavelength in mm.
    groove_density : number
        Grooves per mm.
    order : int, optional
        Diffraction order m.
    included_angle : number, optional
        Angle 2K between the incident and the diffracted chief rays
        (radians); 0 is Littrow.

    Returns
    -------
    alpha, beta : floats
        Angles of incidence and diffraction (radians), alpha - beta = 2K,
        m lambda = d (sin(alpha) + sin(beta)).
    """

    K = included_angle/2
    s = order*wavelength*groove_density/(2*np.cos(K))
    if abs(s) > 1:
        raise ValueError("The order does not propagate at this included angle.")
    theta = np.arcsin(s)

    return theta + K, theta - K


def linear_dispersion(wavelength, groove_density, f_camera, order=1, included_angle=0.0):
    """Linear dispersion dy/dlambda = m f/(d cos(beta)) on the detector.

    Parameters
    ----------
    wavelength, groove_density, order, included_angle :
        See `grating_rotation()`.
    f_camera : number
        Focal length of the camera (mm).

    Returns
    -------
    dispersion : float
        mm on the detector per mm of wavelength (multiply by 1e-6 for mm/nm).
    """

    beta = grating_rotation(wavelength, groove_density, order, included_angle)[1]

    return order*groove_density*f_camera/np.cos(beta)


def shafer_angle(phi_1, f_collimator, f_camera, alpha, beta):
    """Camera off-axis angle phi_2 that cancels coma (Shafer's condition).

    Parameters
    ----------
    phi_1 : number
        Off-axis angle of the collimator (half its fold angle, radians).
    f_collimator, f_camera : numbers
        Focal lengths (R = 2 f).
    alpha, beta : numbers
        Angles of incidence and diffraction at the grating.

    Returns
    -------
    phi_2 : float
        Off-axis angle of the camera (radians, same sign as phi_1).
    """

    from scipy.optimize import brentq

    rhs = (f_camera/f_collimator)**2*(np.cos(alpha)/np.cos(beta))**3/np.cos(phi_1)**3*np.sin(phi_1)

    def f(p):
        return np.sin(p) - rhs*np.cos(p)**3

    return brentq(f, 0.0, np.pi/3) if phi_1 > 0 else -brentq(f, 0.0, np.pi/3) if phi_1 < 0 else 0.0


def fold_after(surfaces, stop_index, wavelength, object_distance, sigma):
    """Tilt of the coordinate break that follows the last surface of
    `surfaces` onto its outgoing chief ray.

    Parameters
    ----------
    surfaces : list of raytrace.Surface
        Surfaces up to the deflecting one (the last), starting with the
        first surface after the object.
    stop_index : int
        Index of a surface with a semi diameter (needed by `System`).
    wavelength : number
        Design wavelength (mm).
    object_distance : number
        Distance of the slit in front of the first vertex.
    sigma : {1, -1}
        Direction of the light along z of the new frame in the unfolded
        system (-1 after an odd number of reflections).

    Returns
    -------
    a_x : float
        Tilt about x of `CoordinateBreak(tilt=(a_x, 0, 0))`.
    """

    system = rt.System(surfaces, stop_index, wavelength, object_distance=object_distance,
                       ray_aiming=False)
    rays = rt.Rays(np.array([[0.0, 0.0, -object_distance]]), np.array([[0.0, 0.0, 1.0]]),
                   np.zeros(1))
    bundle = system.trace(rays, clip=False, last=len(surfaces) - 1)
    o, R = system.frames()[0][-1]
    d = bundle.directions[-1][0] @ R
    if not np.all(np.isfinite(d)):
        raise ValueError("The chief ray is lost before the fold.")
    if abs(d[0]) > 1e-9:
        raise ValueError("The chief ray leaves the y-z plane.")

    # sigma * (0, -sin a, cos a) = d:
    return float(np.arctan2(-sigma*d[1], sigma*d[2]))


def _chief_directions(surfaces, stop_index, wavelength, object_distance):
    """Directions of the chief ray before and after the last surface, in its
    local frame."""

    system = rt.System(surfaces, stop_index, wavelength, object_distance=object_distance,
                       ray_aiming=False)
    rays = rt.Rays(np.array([[0.0, 0.0, -object_distance]]), np.array([[0.0, 0.0, 1.0]]),
                   np.zeros(1))
    bundle = system.trace(rays, clip=False, last=len(surfaces) - 1)
    R = system.frames()[0][-1][1]

    return bundle.directions[-2][0] @ R, bundle.directions[-1][0] @ R


def _fold(surfaces, element, tilt_in, thickness, sigma, stop_index, wavelength, object_distance):
    """Append CoordinateBreak(tilt_in), element and the break onto the
    outgoing chief ray (thickness along the new axis)."""

    surfaces = surfaces + [rt.CoordinateBreak(tilt=(tilt_in, 0.0, 0.0)), element]
    a = fold_after(surfaces, stop_index, wavelength, object_distance, sigma)

    return surfaces + [rt.CoordinateBreak(tilt=(a, 0.0, 0.0), thickness=sigma*thickness)]


CzernyTurner = namedtuple("CzernyTurner", ["system", "alpha", "beta", "phi_1", "phi_2",
                                           "grating_index", "dispersion"])
CzernyTurner.__doc__ = """A Czerny-Turner spectrometer from `czerny_turner()`.

system : raytrace.System at the design wavelength (slit -> detector).
alpha, beta : angles of incidence and diffraction at the design wavelength.
phi_1, phi_2 : off-axis angles of collimator and camera.
grating_index : index of the grating in system.surfaces.
dispersion : linear dispersion on the detector, mm per nm.
"""


def czerny_turner(wavelength=550e-6, groove_density=1200.0, f_collimator=100.0,
                  f_camera=100.0, order=1, included_angle=np.radians(30.0),
                  phi_1=np.radians(8.0), phi_2=None, crossed=False, grating_size=25.0,
                  mirror_size=25.0, camera_size=50.0, slit_height=5.0, f_number=None, detector_tilt=0.0,
                  detector_shift=0.0, mirror_conic=0.0):
    """Czerny-Turner spectrometer with spherical mirrors and a plane grating.

    Slit -> concave collimator -> plane reflection grating -> concave
    camera -> detector, all folded in the y-z plane. The grating is the
    collimator is the aperture stop (a circle of diameter mirror_size, or
    of diameter f_collimator/f_number); the grating is in the collimated
    space, where the stop would put the entrance pupil at infinity.

    Parameters
    ----------
    wavelength : number, optional
        Design (centre) wavelength in mm.
    groove_density : number, optional
        Lines per mm.
    f_collimator, f_camera : numbers, optional
        Focal lengths of the mirrors (mm), radii -2f.
    order : int, optional
        Diffraction order.
    included_angle : number, optional
        Angle between the chief rays to and from the grating (radians).
    phi_1 : number, optional
        Off-axis angle of the collimator (half its fold angle).
    phi_2 : number, optional
        Off-axis angle of the camera; None: Shafer's coma-free angle.
    crossed : bool, optional
        False: the "W" (unfolded) layout, in which the coma of the two
        mirrors can cancel; True: the opposite sense of the camera fold
        (the coma adds).
    grating_size, mirror_size, camera_size : numbers, optional
        Clear diameters (mm) of the grating, the collimator and the camera
        mirror (larger, as the beams of the band walk over it).
    slit_height : number, optional
        Length of the slit (mm), the maximum object height (along x).
    f_number : number, optional
        Sets the stop instead of mirror_size: radius f_collimator/(2
        f_number) on the collimator.
    detector_tilt : number, optional
        Tilt of the detector about x (radians), to put it on the tilted
        focal surface.
    detector_shift : number, optional
        Detector distance minus f_camera (mm).
    mirror_conic : number, optional
        Conic constant of both mirrors (0 spheres, -1 paraboloids).

    Returns
    -------
    CzernyTurner
    """

    alpha, beta = grating_rotation(wavelength, groove_density, order, included_angle)
    if phi_2 is None:
        phi_2 = shafer_angle(abs(phi_1), f_collimator, f_camera, alpha, beta)*np.sign(phi_1)
    sense = -1.0 if crossed else 1.0
    period = 1.0/groove_density
    R1, R2 = -2*f_collimator, -2*f_camera
    surfaces = []
    # collimator (light travels +z before, -z after):
    stop_radius = mirror_size/2 if f_number is None else f_collimator/(2*f_number)
    collimator = rt.Surface(R1, 0.0, mirror=True, conic=mirror_conic,
                            semi_diameter=stop_radius, label="collimator")
    stop = 1
    surfaces = _fold([], collimator, phi_1, f_collimator, -1, stop, wavelength, f_collimator)
    # grating (the stop):
    grating = rt.GratingSurface(np.inf, 0.0, period=period, order=order, mirror=True,
                                semi_diameter=grating_size/2, label="grating")
    # the incident chief ray travels -z; a tilt of -alpha puts it at alpha
    # from the grating normal; choose the groove direction that sends the
    # order to beta:
    grating_index = len(surfaces) + 1
    trial = None
    for tilt_g in (alpha, -alpha):
        for groove in (0.0, np.pi):
            grating.groove_angle = groove
            d_in, d_out = _chief_directions(surfaces + [rt.CoordinateBreak(tilt=(tilt_g, 0, 0)),
                                                        grating], stop, wavelength, f_collimator)
            if not np.all(np.isfinite(d_out)):
                continue
            # incidence at alpha and the included angle between the chief rays:
            ok = (abs(abs(d_in[2]) - np.cos(alpha)) < 1e-9
                  and abs(np.dot(-d_in, d_out) - np.cos(alpha - beta)) < 1e-9)
            if ok:
                trial = (tilt_g, groove)
                break
        if trial is not None:
            break
    if trial is None:
        raise ValueError("No grating orientation found for this order.")
    grating.groove_angle = trial[1]
    surfaces = _fold(surfaces, grating, trial[0], f_collimator, 1, stop, wavelength,
                     f_collimator)
    # fix the distance grating -> camera (the collimator-grating distance
    # above is f_collimator; the beam is collimated so it does not matter
    # much, keep it equal):
    camera = rt.Surface(R2, 0.0, mirror=True, conic=mirror_conic,
                        semi_diameter=camera_size/2, label="camera")
    surfaces = _fold(surfaces, camera, sense*phi_2, f_camera + detector_shift, -1, stop,
                     wavelength, f_collimator)
    system = rt.System(surfaces, stop, wavelength, object_distance=f_collimator,
                       object_height=(slit_height/2, 0.0), image_tilt=(detector_tilt, 0.0, 0.0))
    disp = linear_dispersion(wavelength, groove_density, f_camera, order, included_angle)*1e-6

    return CzernyTurner(system, alpha, beta, phi_1, sense*phi_2, grating_index, disp)


def offner(wavelength=650e-6, R_primary=200.0, groove_density=40.0, order=-1,
           slit_offset=30.0, slit_length=12.0, grating_size=40.0, primary_size=120.0,
           grating_radius=None, image_shift=0.0):
    """Offner imaging spectrometer: concave primary, convex grating, concave
    tertiary (the primary used again), all concentric.

    The slit lies in the object plane through the common centre of
    curvature, offset by slit_offset in y and extended along x; its image
    (magnification -1) lies at -slit_offset, dispersed along y. The grating,
    of radius R_primary/2, is the aperture stop. The system is telecentric
    on both sides (the stop is in the focal plane of the primary), so the
    entrance pupil is at infinity: launch rays with `cone_rays()` rather than
    with `System.rays()`.

    Parameters
    ----------
    wavelength : number, optional
        Design wavelength (mm); only the cone of rays launched depends on
        it, the geometry is concentric for all wavelengths.
    R_primary : number, optional
        Radius of curvature of primary and tertiary (mm).
    groove_density : number, optional
        Lines per mm of the convex grating (along y).
    order : int, optional
        Diffraction order.
    slit_offset, slit_length : numbers, optional
        Height of the slit centre and its length (mm).
    grating_size, primary_size : numbers, optional
        Clear diameters (mm).
    grating_radius : number, optional
        Radius of the grating; default R_primary/2 (exact Offner).
    image_shift : number, optional
        Detector distance from the object plane (mm, focus adjustment).

    Returns
    -------
    raytrace.System
        Object: the slit plane (object_height = (slit_length/2,
        slit_offset)); image plane: the detector, in the object plane.
    """

    R1 = float(R_primary)
    R2 = R1/2 if grating_radius is None else float(grating_radius)
    surfaces = [rt.Surface(-R1, -(R1 - R2), mirror=True, semi_diameter=primary_size/2,
                           label="primary"),
                rt.GratingSurface(-R2, R1 - R2, period=1.0/groove_density, order=order,
                                  mirror=True, semi_diameter=grating_size/2, label="grating"),
                rt.Surface(-R1, -R1 + image_shift, mirror=True, semi_diameter=primary_size/2,
                           label="tertiary")]

    return rt.System(surfaces, 1, wavelength, object_distance=R1,
                     object_height=(slit_length/2, slit_offset))


def cone_rays(system, point, na, n=21, axis=(0.0, 0.0, 1.0)):
    """Rays from an object point filling a cone (a square grid of direction
    cosines inside the circle of radius na about the axis).

    Use it for telecentric systems (`offner()`), where `System.rays()`
    cannot aim at an entrance pupil at infinity; rays outside the stop are
    vignetted by the trace.

    Parameters
    ----------
    system : raytrace.System
        With a finite object distance.
    point : (x, y)
        Object point.
    na : number
        Sine of the half angle of the cone.
    n : int, optional
        Rays across the diameter.
    axis : 3-vector, optional
        Axis of the cone (unit vector).

    Returns
    -------
    raytrace.Rays
    """

    u = np.linspace(-na, na, n)
    U, V = np.meshgrid(u, u)
    inside = U**2 + V**2 <= na**2*(1 + 1e-12)
    U, V = U[inside], V[inside]
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    # an orthonormal frame with axis as its z:
    e1 = np.cross([0.0, 1.0, 0.0], axis)
    if np.linalg.norm(e1) < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    W = np.sqrt(1 - U**2 - V**2)
    d = U[:, None]*e1 + V[:, None]*e2 + W[:, None]*axis
    P = np.tile([point[0], point[1], -system.object_distance], (len(d), 1))

    return rt.Rays(P, d, np.zeros(len(d)))


def image_points(system, field, wavelengths, na=None, n=21, px=None, py=None):
    """Detector hits of the rays from one slit point at several wavelengths.

    Parameters
    ----------
    system : raytrace.System
        A spectrometer (finite object).
    field : (x, y)
        Slit point.
    wavelengths : sequence
        Wavelengths in mm.
    na : number, optional
        Launch a cone of this NA with `cone_rays()` (telecentric systems);
        default: rays through the entrance pupil (px, py) of `System.rays()`.
    n : int, optional
        Rays across the cone or pupil diameter.
    px, py : arrays, optional
        Pupil coordinates; default a square grid inside the unit circle.

    Returns
    -------
    list of (N, 2) arrays
        (x, y) in the detector frame for each wavelength, NaN for lost rays.
    """

    if na is None and px is None:
        u = np.linspace(-1, 1, n)
        PX, PY = np.meshgrid(u, u)
        inside = PX**2 + PY**2 <= 1 + 1e-12
        px, py = PX[inside], PY[inside]
    out = []
    for wl in np.atleast_1d(wavelengths):
        s = system.with_wavelength(float(wl))
        if na is None:
            bundle = s.trace_pupil(field, px, py)
        else:
            bundle = s.trace(cone_rays(s, field, na, n))
        P = s.to_image_frame(bundle.points[-1])
        out.append(P[:, :2])

    return out


def centroids(system, slit_points, wavelengths, **kwargs):
    """Centroids of the slit images: array (n_slit, n_wavelengths, 2).

    Parameters
    ----------
    system : raytrace.System
    slit_points : sequence of (x, y)
    wavelengths : sequence
    **kwargs :
        Passed to `image_points()` (na, n).

    Returns
    -------
    C : array
        Centroid (x, y) on the detector of every slit point and wavelength.
    """

    C = np.empty((len(slit_points), len(np.atleast_1d(wavelengths)), 2))
    for i, f in enumerate(slit_points):
        for j, P in enumerate(image_points(system, f, wavelengths, **kwargs)):
            C[i, j] = np.nanmean(P, axis=0)

    return C


def smile_keystone(C):
    """Smile and keystone of a pushbroom spectrometer from the centroids of
    `centroids()` (slit along x, dispersion along y).

    Smile: the bending of the image of the straight slit at one wavelength,
    y(x) - y(centre of the slit). Keystone: the change of the
    magnification along the slit with the wavelength, x(lambda) - x(centre
    wavelength), for each slit point.

    Parameters
    ----------
    C : array (n_slit, n_wavelengths, 2)
        The slit points ordered along the slit, with the centre one in the
        middle; the wavelengths in order, the centre one in the middle.

    Returns
    -------
    smile, keystone : arrays (n_slit, n_wavelengths)
        In detector length units.
    """

    i0, j0 = C.shape[0]//2, C.shape[1]//2
    smile = C[:, :, 1] - C[i0:i0 + 1, :, 1]
    keystone = C[:, :, 0] - C[:, j0:j0 + 1, 0]

    return smile, keystone
