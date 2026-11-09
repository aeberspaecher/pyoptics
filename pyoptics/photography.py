#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Photographic optics: the out-of-focus image of a point (bokeh), the lens
that makes it, the star that its iris makes of a bright point (sunstars),
the ghosts that its surfaces make of it by double reflection (flare), and
what is sharp: depth of field, diffraction and tilted lenses.

A camera lens focused at one distance images a point at another distance
as a blur, the shape and brightness distribution of which photographers
call bokeh. Geometrically, the blur is a shadow of the pupil: every ray
from the point crosses the sensor at a place set by where it passed the
pupil, and the brightness at a place in the blur is the number of rays per
area there. So the blur shows

- the outline of the pupil seen from the point: round on the axis, a
  "cat's eye" off the axis where the rims of the front and rear lenses cut
  into the beam (mechanical vignetting, `vignetting_circles()`),
- the spherical aberration of the lens, which bunches or spreads the rays
  towards the rim of the blur: a bright edge (soap bubble, nervous bokeh)
  on one side of the focus, a soft one on the other
  (`defocus_profile()`),
- small ripples of a surface (the "onion rings" of moulded aspheres), whose
  slopes redistribute the rays (`RippledSurface`, `onion_ring_contrast()`),
- the transmission of the pupil: an apodizing filter whose transmission
  falls towards the rim (Minolta/Sony STF, Fujifilm APD) gives a blur with
  soft edges (`gaussian_apodizer()`, `t_number()`).

Contents:

- `double_gauss()`: a fast double Gauss camera lens, f = 50 mm, f/1.8,
  six elements of N-LAK9 and N-SF5, designed for this module with
  `lens_optimization` (notebook 90), with clear apertures that vignette
  off-axis beams as real lenses of this type do; `focus()` moves the
  image plane to the smallest spot for an object distance (unit focusing).
- `blur_diameter()`: thin lens circle of confusion for a point at one
  distance with the lens focused at another; `blur_diameter_pupils()`:
  the same for a thick lens from its pupils (exact in first order).
- `point_footprint()`: rays from an object point through the whole
  aperture, with the vignetting of all clear apertures, an optional
  apodizer, their hits on the sensor and their weights (the irradiance of
  a point source on the entrance pupil plane, I cos^3(theta)/L^2);
  `splat()` bins them into an image; `relative_illumination()`.
- `vignetting_circles()`: the clear apertures seen from an object point,
  projected onto the entrance pupil plane (paraxially): circles whose
  intersection is the vignetted pupil, `vignetted_fraction()` its area.
- `defocus_profile()`: geometric irradiance across the blur of a pupil
  with defocus W020 and spherical aberration W040 (any apodization);
  `defocused_psf()`: the same blur from wave optics (FFT of the pupil).
- `RippledSurface`: a `raytrace.Surface` with a radial sinusoidal sag
  ripple; `onion_ring_contrast()` and `converging_beam_equivalent()`:
  the ring contrast of a weak phase ripple after propagation (Talbot) and
  the Fresnel scaling of a converging beam.
- `gaussian_apodizer()`, `apodizer_mean_transmission()`, `t_number()`,
  `apodizer_for_t_number()`: a Gaussian radial apodizer and the T-number
  it costs.
- Sunstars (notebook 91): `iris()`, the opening of an iris diaphragm with
  straight or rounded blades, `iris_apothem()` for a given area (F-number),
  `polygon_area()`, `polygon_perimeter()`, `iris_edge_length()`;
  `polygon_transform()`, the exact Fourier transform of a polygon (a sum
  over its edges), `iris_coverage()` and `iris_psf()`, the antialiased
  opening and its far field by FFT; the spike laws `spike_count()` (2 n
  for odd n, n for even n), `edge_spike()`, `curved_edge_spike()` and
  `rounded_edge_crossover()` for rounded blades, `power_outside()` (the
  wings depend on perimeter and area only); `sunstar()`, the image of the
  Sun (or a point) through an iris on a sensor, by wavelength.
- Flare and ghosts (notebook 92): `ghost_paths()`, every double
  reflection (j, i) of a lens, including those off the sensor;
  `ghost_system()`, a ghost path unfolded into a `raytrace.System` with
  the reflecting surfaces as mirrors; `flare_rays()` and `trace_flare()`,
  the polarization ray trace of a path with the thin film coatings of its
  surfaces; `AR_DESIGNS` and `ar_coatings()`, uncoated, single layer MgF2
  and a three layer antireflection coating on every air-glass surface;
  `ghost_footprint()`, an adaptively sampled ghost of a point or of the
  Sun, through a polygonal iris, with its spectrum; `ghost_spectra()` the
  same for several coatings at once; `ghost_analysis()`, the paraxial
  ghost analysis (blur, focus and irradiance relative to a white surface
  for every path).
- Depth of field and tilt (notebook 94): `hyperfocal_distance()`,
  `dof_limits()` (thin lens, consistent with `blur_diameter()`);
  `defocus_mtf()`, the wave optical MTF of a defocused lens (Hopkins),
  `geometric_defocus_mtf()`, the blur disk, and `optimum_f_number()`
  (Hansma); a tilted thin lens: `lens_normal()`, `tilted_conjugate()`,
  `hinge_distance()` (Merklinger), `plane_of_focus()` (with the
  Scheimpflug line), `tilted_blur()`; a real lens tilted with coordinate
  breaks, `tilt_lens()`, and `trace_point()` for finite object points.

Conventions are those of `raytrace`: lengths in mm, light along +z, the
optical path difference OPD > 0 where the wavefront leads, so that a sensor
behind the focus (a point farther away than the focused distance, i.e. the
background) sees defocus W020 < 0, and a positive lens with undercorrected
spherical aberration has W040 < 0. In `defocus_profile()` and
`defocused_psf()`, W020 and W040 are in waves and transverse coordinates in
units of lambda/NA (image space NA), so that a ray through the pupil radius
rho lands at r = dW/drho.

References
----------
H. H. Nasse, "Depth of field and bokeh", Carl Zeiss Camera Lens Division
(2010).

H. H. Nasse, "How to read MTF curves, part II", Carl Zeiss Camera Lens
Division (2009), on vignetting and the cat's eye.

R. Kingslake, "A History of the Photographic Lens", Academic Press (1989),
chapter 9 (the double Gauss).

W. J. Smith, "Modern Lens Design", 2nd ed., McGraw-Hill (2005), chapter 15
(double Gauss lenses).

J. W. Goodman, "Introduction to Fourier Optics", 3rd ed., Roberts (2005),
sections 4.5 (Talbot images) and 6.4 (apodization).

M. R. Teague, "Deterministic phase retrieval: a Green's function
solution", J. Opt. Soc. Am. 73, 1434-1441 (1983), on the transport of
intensity.

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), sections 8.5 (Fraunhofer diffraction at apertures), 8.8
(defocused images) and 9.2 (aberrations).

S.-W. Lee and R. Mittra, "Fourier transform of a polygonal shape function
and its application in electromagnetics", IEEE Trans. Antennas Propag. 31,
99-103 (1983).

J. B. Keller, "Geometrical theory of diffraction", J. Opt. Soc. Am. 52,
116-130 (1962).

J. E. Harvey and C. Ftaclas, "Diffraction effects of telescope secondary
mirror spiders on various image-quality criteria", Appl. Opt. 34,
6337-6349 (1995).

H. M. Merklinger, "Focusing the View Camera", self-published (1993).

H. H. Hopkins, "The frequency response of a defocused optical system",
Proc. R. Soc. Lond. A 231, 91-103 (1955).

H. A. Macleod, "Thin-Film Optical Filters", 4th ed., CRC Press (2010),
chapters 2 and 4 (antireflection coatings).

W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008),
section 14.6 (ghost images).

R. A. Chipman, W. S. T. Lam and G. Young, "Polarized Light and Optical
Systems", CRC Press (2018), chapters 9 and 10 (polarization ray tracing).

M. Hullin, E. Eisemann, H.-P. Seidel and S. Lee, "Physically-based
real-time lens flare rendering", ACM Trans. Graph. 30, 108 (2011).
"""

import copy
from collections import namedtuple

import numpy as np

from . import raytrace as rt


__all__ = ["double_gauss", "focus", "blur_diameter", "blur_diameter_pupils", "point_footprint", "splat",
           "relative_illumination", "vignetting_circles", "vignetted_fraction",
           "defocus_profile", "defocused_psf", "RippledSurface", "onion_ring_contrast",
           "converging_beam_equivalent", "gaussian_apodizer", "apodizer_mean_transmission",
           "t_number", "apodizer_for_t_number", "DOUBLE_GAUSS_F", "D_LINE",
           "iris", "polygon_area", "polygon_perimeter", "iris_apothem", "polygon_transform",
           "iris_coverage", "iris_psf", "edge_spike", "spike_count", "rounded_edge_crossover",
           "iris_edge_length", "power_outside", "curved_edge_spike", "sunstar", "SUN_RADIUS",
           "ghost_paths", "ghost_system", "flare_rays", "trace_flare", "AR_DESIGNS", "ar_coatings",
           "GhostFootprint", "ghost_footprint", "GhostInfo", "ghost_analysis", "ghost_spectra",
           "hyperfocal_distance", "dof_limits", "geometric_defocus_mtf", "defocus_mtf",
           "optimum_f_number", "lens_normal", "tilted_conjugate", "hinge_distance", "plane_of_focus",
           "tilted_blur", "tilt_lens", "trace_point"]


D_LINE = 587.5618e-6  # mm
DOUBLE_GAUSS_F = 50.0  # mm, focal length of double_gauss()

# The double Gauss of notebook 90 (f = 50 mm, f/1.8): radius, thickness to
# the next surface, glass behind ("A": air), clear semi-diameter (mm). The
# stop is surface 5; its semi-diameter is set from the F-number.
_DOUBLE_GAUSS = """
  34.135   7.030  N-LAK9  17.0
  62.192   0.300  A       16.0
  18.549   3.153  N-LAK9  13.8
  18.476   1.500  N-SF5   12.9
  14.669  10.566  A       11.7
     inf   3.858  A       None
 -23.316   1.500  N-SF5   10.9
  47.669  10.970  N-LAK9  12.8
 -29.340   0.300  A       13.6
  54.812   5.887  N-LAK9  13.6
-116.032  36.000  A       13.3
"""


def _glass(name):
    from . import glasses
    return glasses.get(name).n


def double_gauss(f_number=1.8, object_distance=np.inf, wavelength=D_LINE, apertures=True,
                 field=None):
    """A fast double Gauss camera lens, f = 50 mm, f/1.8 wide open.

    Six elements (N-LAK9 and N-SF5) in the classic arrangement: a positive
    meniscus and a cemented negative meniscus doublet in front of the stop,
    their mirror image behind it. Designed with
    `lens_optimization.DampedLeastSquares` for the F, d and C lines from a
    slower textbook double Gauss (notebook 90). Like the real lenses of
    this type it is soft wide open: undercorrected zonal spherical
    aberration (Seidel W040 = -32 waves, balanced by higher orders), an
    RMS spot radius (d line, best focus) of 37 um on the axis at f/1.8,
    10 um at f/2.8 and 4 um at f/4; off the axis it is softer than
    commercial lenses of the type.

    Parameters
    ----------
    f_number : number, optional
        F-number for an object at infinity; sets the semi-diameter of the
        stop, which stays when the lens is focused closer (the working
        F-number grows).
    object_distance : number, optional
        Distance of the object in front of the first vertex (mm); np.inf
        (default) for an object at infinity. The image plane stays at the
        paraxial focus for infinity; use `focus()` to focus the lens.
    wavelength : number, optional
        Vacuum wavelength in mm (default: the d line).
    apertures : bool, optional
        Give the lenses their clear apertures (default), which vignette
        off-axis beams at large apertures. False: only the stop limits.
    field : number, optional
        Maximum field: angle in radians for an object at infinity
        (default: 23.4 deg, the corner of a 36 x 24 mm sensor), object
        height in mm for a finite object (default: the same angle seen
        from the first vertex).

    Returns
    -------
    raytrace.System
    """

    surfaces = []
    for line in _DOUBLE_GAUSS.strip().splitlines():
        R, t, g, sd = line.split()
        R = float(R)
        sd = None if sd == "None" else float(sd)
        if not apertures:
            sd = None
        if g == "A":
            surfaces.append(rt.Surface(R, float(t), n_after=1.0, semi_diameter=sd))
        else:
            surfaces.append(rt.Surface(R, float(t), material=_glass(g), semi_diameter=sd))
    stop = 5
    surfaces[stop].n_after = None
    angle = np.deg2rad(23.4)
    inf_system = rt.System(surfaces, stop, wavelength,
                           entrance_pupil_radius=DOUBLE_GAUSS_F/(2*f_number),
                           field_angle=angle if field is None else field)
    if np.isinf(object_distance):
        return inf_system
    height = np.tan(angle)*object_distance if field is None else field

    return rt.System(inf_system.surfaces, stop, wavelength, object_distance=object_distance,
                     object_height=height)


def focus(system, object_distance=None, field=0.0, N=31):
    """Copy of a system focused on an object distance by moving the image
    plane (unit focusing: the whole lens moves away from the sensor).

    The image plane is placed at the smallest RMS spot (geometric best
    focus) of the given field, which for a lens with spherical aberration
    lies between the paraxial and the marginal focus.

    Parameters
    ----------
    system : raytrace.System
    object_distance : number, optional
        Object distance in front of the first vertex (np.inf for infinity).
        Default: that of the system.
    field : number, optional
        Field at which the spot is minimized (default: on the axis), as a
        fraction of the maximum field.
    N : int, optional
        Rays across the pupil of the spot diagram.

    Returns
    -------
    raytrace.System
    """

    from scipy.optimize import minimize_scalar

    if object_distance is None:
        object_distance = system.object_distance
    # the field as an angle seen from the first vertex:
    angle = (system.max_field if system.infinite_object
             else np.arctan(system.max_field/system.object_distance))
    if np.isinf(object_distance):
        s = rt.System(system.surfaces, system.stop_index, system.wavelength, field_angle=angle)
    else:
        s = rt.System(system.surfaces, system.stop_index, system.wavelength,
                      object_distance=object_distance, object_height=np.tan(angle)*object_distance)
    s = s.at_paraxial_focus()
    d0 = s.surfaces[-1].thickness
    f = field*s.max_field

    def rms(d):
        return s.with_image_distance(d).spot_diagram(f, N=N).rms

    # coarse scan over +-40 depths of focus (spherical aberration moves the best focus
    # far from the paraxial one), then refine around the best point
    width = 40*s.wavelength/s.paraxial().na**2
    grid = d0 + np.linspace(-width, width, 81)
    k = int(np.argmin([rms(d) for d in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(rms, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})

    return s.with_image_distance(res.x)


def blur_diameter(f, f_number, focus_distance, distance):
    """Thin lens circle of confusion: the diameter of the blur of a point at
    `distance` when the lens is focused at `focus_distance`.

    c = f**2/N * abs(s - s_f)/(s*(s_f - f)), with the distances s and s_f
    measured from the lens (the entrance pupil); np.inf is allowed for both.

    Parameters
    ----------
    f : number
        Focal length.
    f_number : number
        F-number N = f/D.
    focus_distance : number or array
        Distance the lens is focused at.
    distance : number or array
        Distance of the point.

    Returns
    -------
    c : number or array
        Blur diameter on the sensor (the unit of f).
    """

    s_f = np.asarray(focus_distance, float)
    s = np.asarray(distance, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        # D * |v - v_f|/v_f with v = f s/(s - f): written for s, s_f -> inf
        inv_s, inv_sf = np.where(np.isinf(s), 0.0, 1/s), np.where(np.isinf(s_f), 0.0, 1/s_f)
        c = f**2/f_number*np.abs(inv_sf - inv_s)/(1 - f*inv_sf)

    return c


def blur_diameter_pupils(f, ep_diameter, pupil_magnification, sensor_x, x):
    """Blur diameter of a thick lens in first order (exact paraxially).

    With Newton's distances, x of the point in front of the front focal
    point F and x_s of the sensor behind the rear focal point F', the point
    is imaged to x' = f**2/x behind F', and its cone of light, which fills
    the exit pupil (diameter m_p D at m_p f in front of F', m_p the pupil
    magnification), cuts the sensor in a disk of diameter

        c = m_p D abs(x_s - x')/(x' + m_p f).

    For m_p = 1 and the sensor focused at s_f this is the thin lens formula
    of `blur_diameter()` (distances from the principal planes).

    Parameters
    ----------
    f : number
        Focal length.
    ep_diameter : number
        Entrance pupil diameter D.
    pupil_magnification : number
        Exit over entrance pupil diameter.
    sensor_x : number
        Distance of the sensor behind F'.
    x : number or array
        Distance of the point in front of F (np.inf allowed).

    Returns
    -------
    c : number or array
    """

    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        x_img = np.where(np.isinf(x), 0.0, f**2/x)
    m = pupil_magnification

    return m*ep_diameter*np.abs(sensor_x - x_img)/(x_img + m*f)


Footprint = namedtuple("Footprint", ["x", "y", "weight", "stop_x", "stop_y", "pupil_x",
                                     "pupil_y", "cell_area"])
Footprint.__doc__ = """Rays of one object point that reach the sensor.

x, y : sensor coordinates of the rays (mm, image plane frame)
weight : power of each ray (sum: power through the lens, in units of the
    on-axis power per entrance pupil area times that area)
stop_x, stop_y : where the rays pass the stop, normalized to its
    semi-diameter (for apodizers)
pupil_x, pupil_y : where the rays cross the paraxial entrance pupil plane
    (mm)
cell_area : area of the entrance pupil plane per ray (mm^2)
"""


def _launch(system, field, X, Y):
    """Rays from the object point (field) through the points (X, Y) of the
    paraxial entrance pupil plane, and the relative irradiance of the point
    there (1 on the axis at the pupil centre)."""

    fx, fy = rt._field_xy(field)
    ep_z = system.paraxial().ep_z
    n = len(X)
    if system.infinite_object:
        d = np.array([np.tan(fx), np.tan(fy), 1.0])
        d = d/np.linalg.norm(d)
        E = np.column_stack([X, Y, np.full(n, ep_z)])
        z0 = min(0.0, ep_z) - 1.0
        P = E - ((ep_z - z0)/d[2])*d
        D = np.tile(d, (n, 1))
        w = np.full(n, d[2])  # projected area of the pupil plane
    else:
        O = np.array([fx, fy, -system.object_distance])
        E = np.column_stack([X, Y, np.full(n, ep_z)])
        D = E - O
        L = ep_z + system.object_distance
        r = np.linalg.norm(D, axis=1)
        D = D/r[:, None]
        P = np.tile(O, (n, 1))
        w = (L/r)**3  # I cos(theta)/r**2 = I cos(theta)**3/L**2, relative to I/L**2
    return rt.Rays(P, D, np.zeros(n)), w


def point_footprint(system, field, n_rays=20000, rng=None, apodization=None, margin=2.5):
    """Trace rays from one object point through the whole aperture.

    The rays fill the region of the paraxial entrance pupil plane through
    which light can pass (found with a coarse trace over margin times the
    pupil radius, then sampled with n_rays uniform random points), so the
    vignetting by every clear aperture and the pupil aberrations are those
    of the real rays. Each ray carries the irradiance that a point source
    gives on that plane, I cos(theta)**3/L**2 (object at infinity: the
    projected area cos(theta)), times the apodizer transmission.

    Parameters
    ----------
    system : raytrace.System
    field : number or pair
        Object point (see `raytrace.System.rays()`): field angle (object
        at infinity) or object height (finite object), a pair (x, y) for
        points off the y axis.
    n_rays : int, optional
        Rays in the sampled region.
    rng : numpy Generator, optional
    apodization : callable, optional
        Intensity transmission T(rho) of a filter at the stop, as a function
        of the normalized radius rho at which a ray passes the stop.
    margin : number, optional
        Radius of the coarse search region in units of the entrance pupil
        radius.

    Returns
    -------
    Footprint
    """

    rng = np.random.default_rng() if rng is None else rng
    p = system.paraxial()
    R = margin*p.ep_radius
    # coarse search for the region of the pupil plane that transmits
    g = np.linspace(-R, R, 61)
    X, Y = np.meshgrid(g, g)
    X, Y = X.ravel(), Y.ravel()
    inside = X**2 + Y**2 <= R**2
    X, Y = X[inside], Y[inside]
    rays, _ = _launch(system, field, X, Y)
    ok = system.trace(rays).valid
    if not np.any(ok):
        return Footprint(*[np.zeros(0)]*7, 0.0)
    h = g[1] - g[0]
    x0, x1 = X[ok].min() - 1.5*h, X[ok].max() + 1.5*h
    y0, y1 = Y[ok].min() - 1.5*h, Y[ok].max() + 1.5*h
    X = rng.uniform(x0, x1, n_rays)
    Y = rng.uniform(y0, y1, n_rays)
    cell = (x1 - x0)*(y1 - y0)/n_rays
    rays, w = _launch(system, field, X, Y)
    bundle = system.trace(rays)
    v = bundle.valid
    stop = system.stop_index
    o, Rm = system.frames()[0][stop]
    Q = (bundle.points[stop + 1][v] - o) @ Rm
    a_s = system.surfaces[stop].semi_diameter
    sx, sy = Q[:, 0]/a_s, Q[:, 1]/a_s
    w = w[v]*cell
    if apodization is not None:
        w = w*apodization(np.hypot(sx, sy))
    I = system.to_image_frame(bundle.points[-1][v])

    return Footprint(I[:, 0], I[:, 1], w, sx, sy, X[v], Y[v], cell)


def splat(footprint, x_edges, y_edges):
    """Bin the rays of a footprint into an image (power per bin area).

    Parameters
    ----------
    footprint : Footprint or sequence of Footprint
    x_edges, y_edges : arrays
        Bin edges on the sensor (mm).

    Returns
    -------
    E : (len(x_edges) - 1, len(y_edges) - 1) array
        Irradiance, indexed [x, y].
    """

    if isinstance(footprint, Footprint):
        footprint = [footprint]
    E = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for f in footprint:
        if len(f.x):
            E += np.histogram2d(f.x, f.y, [x_edges, y_edges], weights=f.weight)[0]
    area = np.diff(x_edges)[:, None]*np.diff(y_edges)[None, :]

    return E/area


def relative_illumination(system, fields, n_rays=20000, rng=None, apodization=None):
    """Power through the lens from object points at the given fields,
    relative to the axis (vignetting and the cos^3 or projection factor of
    the entrance pupil irradiance; the image side cos factor of the
    irradiance on the sensor is not included).

    Parameters
    ----------
    system : raytrace.System
    fields : sequence
        Fields (see `point_footprint()`).
    n_rays : int, optional
    rng : numpy Generator, optional
    apodization : callable, optional

    Returns
    -------
    array
    """

    P = np.array([point_footprint(system, f, n_rays, rng, apodization).weight.sum()
                  for f in [0.0] + list(fields)])

    return P[1:]/P[0]


def vignetting_circles(system, field):
    """The clear apertures seen from an object point, projected onto the
    paraxial entrance pupil plane.

    A paraxial ray from the object point through the point h of the
    entrance pupil plane reaches surface k at the height A_k h + B_k u (u its
    slope). It passes the clear aperture a_k if h lies inside a circle;
    the light that gets through is the intersection of the circles of all
    surfaces (the stop's circle is the entrance pupil). Off the axis the
    circles of the surfaces in front of and behind the stop shift in
    opposite directions and cut the pupil into the cat's eye.

    Parameters
    ----------
    system : raytrace.System
    field : number or pair
        Object point (see `point_footprint()`).

    Returns
    -------
    centres : (K, 2) array
        Circle centres (x, y) in the entrance pupil plane (mm), one per
        surface with a semi-diameter.
    radii : (K,) array
    surfaces : (K,) int array
        Indices of the surfaces.
    """

    p = system.paraxial()
    ep_z = p.ep_z
    fx, fy = rt._field_xy(field)
    centres, radii, index = [], [], []
    ys_h, _ = system._paraxial_trace(1.0, 0.0)               # h = 1, u = 0 at the EP plane...
    ys_u, _ = system._paraxial_trace(-ep_z, 1.0)             # ...h = 0, u = 1
    for k, s in enumerate(system.surfaces):
        if s.semi_diameter is None or isinstance(s, rt.CoordinateBreak):
            continue
        A, B = ys_h[k], ys_u[k]
        if system.infinite_object:
            u = np.array([np.tan(fx), np.tan(fy)])
            c = -B*u/A
            r = s.semi_diameter/abs(A)
        else:
            L = ep_z + system.object_distance
            y0 = np.array([fx, fy])
            G = A + B/L
            c = B*y0/L/G
            r = s.semi_diameter/abs(G)
        centres.append(c)
        radii.append(r)
        index.append(k)

    return np.array(centres), np.array(radii), np.array(index)


def vignetted_fraction(centres, radii, reference_radius, n=801):
    """Area of the intersection of circles relative to a circle of radius
    reference_radius (the unvignetted entrance pupil), on an n x n grid.

    Parameters
    ----------
    centres : (K, 2) array
    radii : (K,) array
    reference_radius : number
    n : int, optional

    Returns
    -------
    fraction : number
    mask, x, y : the intersection on the grid and its coordinates
    """

    i = np.argmin(radii)
    c0, r0 = centres[i], radii[i]
    x = np.linspace(c0[0] - r0, c0[0] + r0, n)
    y = np.linspace(c0[1] - r0, c0[1] + r0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = np.ones(X.shape, dtype=bool)
    for c, r in zip(centres, radii):
        mask &= (X - c[0])**2 + (Y - c[1])**2 <= r**2
    area = mask.sum()*(x[1] - x[0])*(y[1] - y[0])

    return area/(np.pi*reference_radius**2), mask, x, y


def _cubic_roots(r, W020, W040):
    """Pupil radii rho in [0, 1] with 2 W020 rho + 4 W040 rho**3 = +-r."""

    out = []
    for target in (r, -r):
        if W040 == 0:
            rho = np.array([target/(2*W020)]) if W020 != 0 else np.array([])
        else:
            rho = np.roots([4*W040, 0.0, 2*W020, -target])
            rho = rho[np.abs(rho.imag) < 1e-9].real
        out.extend(rho[(rho >= 0) & (rho <= 1)])

    return np.unique(np.round(out, 14))


def defocus_profile(r, W020, W040=0.0, apodization=None):
    """Geometric irradiance across the blur of a defocused point.

    The pupil has the wavefront W(rho) = W020 rho**2 + W040 rho**4 (waves)
    and the intensity transmission T(rho). A ray through rho lands at the
    radius abs(dW/drho) = abs(2 W020 rho + 4 W040 rho**3) (in lambda/NA), and the
    rays of the annulus 2 rho d rho spread over 2 pi r dr, so

        E(r) = sum over the rays that land at r of T(rho) rho/(pi r abs(dr/drho)),

    normalized to the power through the pupil (integral of E over the plane
    = 1). Where dr/drho = 0 (a caustic, when abs(W020) < 6 abs(W040) with opposite
    signs) E diverges.

    Parameters
    ----------
    r : array
        Radii in units of lambda/NA.
    W020, W040 : numbers
        Defocus and spherical aberration in waves (not both zero).
    apodization : callable, optional
        Intensity transmission T(rho).

    Returns
    -------
    E : array
        Irradiance in units of 1/(lambda/NA)**2.
    """

    r = np.atleast_1d(np.asarray(r, float))
    T = (lambda rho: np.ones_like(rho)) if apodization is None else apodization
    rho_q = np.linspace(0, 1, 2001)
    norm = np.trapezoid(T(rho_q)*2*rho_q, rho_q)
    E = np.zeros(len(r))
    for i, ri in enumerate(r):
        if ri <= 0:
            continue
        for rho in _cubic_roots(ri, W020, W040):
            slope = abs(2*W020 + 12*W040*rho**2)
            with np.errstate(divide="ignore"):
                E[i] += T(np.array([rho]))[0]*rho/(np.pi*ri*slope)

    return E/norm


def defocused_psf(W020, W040=0.0, amplitude=None, wavefront=None, n_pupil=512, n=2048):
    """Wave optical image of a point with defocus and spherical aberration
    (scalar, Fraunhofer: the FFT of the pupil).

    Parameters
    ----------
    W020, W040 : numbers
        Defocus and spherical aberration in waves.
    amplitude : callable, optional
        Amplitude transmission of the pupil as a function of rho (the
        square root of an intensity apodizer).
    wavefront : callable, optional
        Further wavefront W(rho) in waves (e.g. a ripple).
    n_pupil : int, optional
        Samples across the pupil diameter. The image spans n_pupil/2 lambda/NA,
        the blur radius is about 2 abs(W020) + 4 abs(W040) lambda/NA.
    n : int, optional
        FFT size (pupil zero padded to n x n).

    Returns
    -------
    x : (n,) array
        Image coordinates in lambda/NA.
    I : (n, n) array
        Irradiance, normalized to a unit integral over the plane (in units of
        1/(lambda/NA)**2), indexed [y, x].
    """

    from . import fft

    d = 2.0/n_pupil
    u = (np.arange(n) - n//2)*d
    PX, PY = np.meshgrid(u, u)
    rho = np.hypot(PX, PY)
    inside = rho <= 1
    A = np.where(inside, 1.0, 0.0)
    if amplitude is not None:
        A = A*np.where(inside, amplitude(np.minimum(rho, 1)), 0.0)
    W = W020*rho**2 + W040*rho**4
    if wavefront is not None:
        W = W + wavefront(rho)
    P = A*np.exp(2j*np.pi*W)
    U = np.fft.fftshift(fft.fft2(np.fft.ifftshift(P)))
    I = np.abs(U)**2
    x = (np.arange(n) - n//2)/(n*d)
    dx = x[1] - x[0]
    I = I/(I.sum()*dx**2)

    return x, I


class RippledSurface(rt.Surface):
    """A surface with a radial sinusoidal ripple added to its sag.

    z(r) = (conic and aspheric sag) + amplitude cos(2 pi r/period + phase),

    the mid-spatial frequency errors left by turning and polishing that
    show as "onion rings" in the out-of-focus images of lenses with moulded
    aspheres. The ripple's slope deflects the rays by about
    (n' - n) dz/dr, which is small, but in a defocused image the rays
    have travelled far enough to bunch into rings.

    Parameters
    ----------
    R : number, optional
    thickness : number, optional
    amplitude : number, optional
        Ripple amplitude (mm).
    period : number, optional
        Radial period of the ripple (mm).
    phase : number, optional
        Phase of the ripple at the vertex (radians).
    **kwargs
        Further arguments of `raytrace.Surface`.

    Rays through a RippledSurface are traced by the NumPy code of
    `raytrace` (the compiled kernel covers the standard surfaces only).
    """

    def __init__(self, R=np.inf, thickness=0.0, amplitude=0.0, period=1.0, phase=0.0, **kwargs):
        super().__init__(R, thickness, **kwargs)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def __repr__(self):
        return "Rippled" + super().__repr__()[:-1] + ", amplitude={}, period={})".format(
            self.amplitude, self.period)

    def ripple(self, r):
        """Sag of the ripple at the radius r."""

        return self.amplitude*np.cos(2*np.pi*r/self.period + self.phase)

    def sag(self, r2):
        return super().sag(r2) + self.ripple(np.sqrt(r2))

    def _slope_over_r(self, r2):
        r = np.sqrt(r2)
        k = 2*np.pi/self.period
        with np.errstate(invalid="ignore", divide="ignore"):
            s = -self.amplitude*k*np.sin(k*r + self.phase)/r
        # limit at the vertex (finite for phase = 0 or pi):
        s = np.where(r > 0, s, -self.amplitude*k**2*np.cos(self.phase))

        return super()._slope_over_r(r2) + s

    def intersect(self, P, d):
        t = super().intersect(P, d)
        dz = d[:, 2]
        for _ in range(20):
            Q = P + t[:, None]*d
            r2 = Q[:, 0]**2 + Q[:, 1]**2
            G = Q[:, 2] - self.sag(r2)
            dG = dz - self._slope_over_r(r2)*(Q[:, 0]*d[:, 0] + Q[:, 1]*d[:, 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                step = G/dG
            t = t - step
            with np.errstate(invalid="ignore"):
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-14*(1 + np.abs(t)))):
                    break

        return t


def onion_ring_contrast(opd_amplitude, period, distance, wavelength):
    """Relative amplitude of the irradiance ripple behind a weak sinusoidal
    phase ripple.

    A wavefront ripple W = A cos(2 pi x/p) (A << wavelength) makes the
    diffraction orders +-1 at the angles +-wavelength/p. After the distance
    z they lag by pi wavelength z/p**2 behind the zero order, and the
    irradiance is

        E/E_0 = 1 + 2 (2 pi A/wavelength) sin(pi wavelength z/p**2) cos(2 pi x/p),

    which grows linearly at first, as the transport of intensity equation
    and the rays predict (E/E_0 = 1 - z d^2W/dx^2, amplitude
    4 pi^2 A z/p**2), and oscillates (Talbot effect) once
    wavelength z/p**2 approaches 1/2.

    Parameters
    ----------
    opd_amplitude : number or array
        Wavefront ripple amplitude A (for a surface ripple a: (n' - n) a).
    period : number or array
    distance : number or array
        Propagation distance z of a collimated beam (for a converging beam
        the equivalent distance, see `converging_beam_equivalent()`).
    wavelength : number

    Returns
    -------
    wave, rays : arrays
        Amplitude of the relative irradiance ripple from waves and from
        rays (geometric limit).
    """

    A = np.asarray(opd_amplitude, float)
    z = np.asarray(distance, float)
    wave = 2*(2*np.pi*A/wavelength)*np.sin(np.pi*wavelength*z/period**2)
    rays = 4*np.pi**2*A*z/period**2

    return wave, rays


def converging_beam_equivalent(R, z):
    """Fresnel scaling of a converging spherical beam.

    A beam converging to a focus at the distance R that propagates the
    distance z has, apart from a magnification and a quadratic phase,
    the same field as a collimated beam that propagates

        z_eff = z R/(R - z), magnified by M = (R - z)/R.

    Behind the focus (z > R), M < 0 (the image is inverted) and z_eff < 0.
    A sensor at the distance Delta behind (or in front of, Delta < 0) the
    focus therefore sees the pupil plane ripples as if propagated by
    abs(z_eff) = R (R + Delta)/abs(Delta), about R**2/abs(Delta).

    Parameters
    ----------
    R : number
        Distance from the plane of the ripple to the focus.
    z : number or array
        Propagation distance.

    Returns
    -------
    z_eff, M : arrays
    """

    z = np.asarray(z, float)
    with np.errstate(divide="ignore"):
        return z*R/(R - z), (R - z)/R


def gaussian_apodizer(s):
    """Gaussian radial apodizer: intensity transmission exp(-(rho/s)**2) over
    the normalized pupil radius rho (a smooth version of the filters of the
    Minolta/Sony STF and Fujifilm APD lenses).

    Parameters
    ----------
    s : number
        1/e radius of the intensity transmission in units of the pupil
        radius.

    Returns
    -------
    T : callable
        T(rho).
    """

    def T(rho):
        return np.exp(-(np.asarray(rho, float)/s)**2)

    return T


def apodizer_mean_transmission(s):
    """Mean intensity transmission of `gaussian_apodizer(s)` over the pupil,
    s**2 (1 - exp(-1/s**2)).

    Parameters
    ----------
    s : number or array

    Returns
    -------
    tau : number or array
    """

    s = np.asarray(s, float)

    return s**2*(1 - np.exp(-1/s**2))


def t_number(f_number, mean_transmission):
    """T-number: the F-number of a clear lens that passes the same light,
    N/sqrt(tau).

    Parameters
    ----------
    f_number : number
    mean_transmission : number
        Mean intensity transmission tau over the pupil.

    Returns
    -------
    T : number
    """

    return f_number/np.sqrt(mean_transmission)


def apodizer_for_t_number(f_number, t):
    """1/e radius s of the Gaussian apodizer that makes an f/N lens a T-number
    t (> N), e.g. the f/2.8, T4.5 of the 135 mm STF lenses.

    Parameters
    ----------
    f_number : number
    t : number

    Returns
    -------
    s : number
    """

    from scipy.optimize import brentq

    tau = (f_number/t)**2
    if not 0 < tau < 1:
        raise ValueError("The T-number must be larger than the F-number.")

    return brentq(lambda s: apodizer_mean_transmission(s) - tau, 1e-3, 1e3)


# --- Sunstars: diffraction at the iris (notebook 91) ----------------------

def _iris_centres(n_blades, apothem, blade_radius, rotation):
    phi = rotation + 2*np.pi*np.arange(n_blades)/n_blades
    normals = np.c_[np.cos(phi), np.sin(phi)]

    return normals, -(blade_radius - apothem)*normals if np.isfinite(blade_radius) else None


def iris(n_blades, apothem, blade_radius=np.inf, rotation=0.0, n_arc=256):
    """Outline of the opening of an iris diaphragm with n_blades blades.

    Straight blades leave a regular polygon with the inscribed radius
    (apothem) h; the edge j has its outward normal at the angle
    rotation + 2 pi j/n_blades. Rounded blades have edges that are circular
    arcs of radius blade_radius (rho >= h): the opening is the intersection
    of n_blades disks of radius rho centred at -(rho - h) n_j, a polygon with
    bulged sides, round (a circle of radius h) for h = rho, i.e. at the
    opening the blades were made for, and more and more polygonal when the
    iris is closed further (h < rho).

    Parameters
    ----------
    n_blades : int
        Number of blades (>= 3).
    apothem : number
        Distance h of the edges from the centre.
    blade_radius : number, optional
        Radius of curvature rho of the blade edges, np.inf for straight
        blades.
    rotation : number, optional
        Angle of the normal of the first edge (radians).
    n_arc : int, optional
        Points per rounded edge.

    Returns
    -------
    vertices : array (M, 2)
        Counter-clockwise outline (not closed: the last point is not
        repeated). For straight blades the n_blades corners.
    """

    N = int(n_blades)
    if N < 3:
        raise ValueError("An iris needs at least three blades.")
    h = float(apothem)
    rho = float(blade_radius)
    if rho < h*(1 - 1e-12):
        raise ValueError("The blade radius must be at least the apothem.")
    normals, centres = _iris_centres(N, h, rho, rotation)
    if centres is None:
        psi = rotation + (2*np.arange(N) + 1)*np.pi/N
        R = h/np.cos(np.pi/N)

        return R*np.c_[np.cos(psi), np.sin(psi)]
    out = []
    for j in range(N):
        c = centres[j]
        ends = []
        for s in (-1, 1):
            psi = rotation + (2*j + s)*np.pi/N
            u = np.array([np.cos(psi), np.sin(psi)])
            cu = c @ u
            t = cu + np.sqrt(max(cu**2 - c @ c + rho**2, 0.0))
            p = t*u - c
            ends.append(np.arctan2(p[1], p[0]))
        a0, a1 = ends
        a1 = a0 + np.mod(a1 - a0, 2*np.pi)
        a = np.linspace(a0, a1, n_arc, endpoint=False)
        out.append(c + rho*np.c_[np.cos(a), np.sin(a)])

    return np.concatenate(out)


def polygon_area(vertices):
    """Area of a polygon given by its counter-clockwise vertices (shoelace)."""

    x, y = np.asarray(vertices, float).T

    return 0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)


def polygon_perimeter(vertices):
    """Perimeter of a closed polygon given by its vertices."""

    v = np.asarray(vertices, float)

    return np.sum(np.hypot(*(np.roll(v, -1, axis=0) - v).T))


def iris_apothem(n_blades, area, blade_radius=np.inf, n_arc=256):
    """Apothem h of the iris of `iris()` whose opening has the given area.
    Lenses calibrate their F-numbers by the area of the opening, so an f/N
    stop has the area of a circle of diameter f/N.

    Parameters
    ----------
    n_blades : int
    area : number
    blade_radius : number, optional
    n_arc : int, optional

    Returns
    -------
    h : number
    """

    from scipy.optimize import brentq

    N = int(n_blades)
    h_poly = np.sqrt(area/(N*np.tan(np.pi/N)))           # straight blades
    if not np.isfinite(blade_radius):
        return h_poly
    if area >= np.pi*blade_radius**2*(1 - 1e-12):
        raise ValueError("The opening is larger than the circle the blades are made for.")

    return brentq(lambda h: polygon_area(iris(N, h, blade_radius, n_arc=n_arc)) - area,
                  np.sqrt(area/np.pi), h_poly)


def polygon_transform(vertices, kx, ky):
    """Fourier transform of the indicator function of a polygon,
    F(k) = integral over the polygon of exp(-i k.r) dA, exactly.

    With exp(-i k.r) = div(i k exp(-i k.r)/k**2), the area integral becomes
    a sum over the edges (Gauss' theorem). The edge j from a_j to b_j, with
    the vector e_j = b_j - a_j, the outward normal times length
    N_j = (e_y, -e_x) and the midpoint m_j, contributes ::

        F(k) = i/k**2 sum_j (k.N_j) exp(-i k.m_j) sinc(k.e_j/2),

    sinc(x) = sin(x)/x. Along the normal of an edge (k parallel to N_j)
    the sinc is 1 and the edge gives a streak with amplitude L_j/k: the
    diffraction spikes. At k = 0, F is the area.

    Parameters
    ----------
    vertices : array (M, 2)
        Counter-clockwise vertices.
    kx, ky : arrays
        Spatial angular frequencies (any broadcastable shape).

    Returns
    -------
    F : complex array
    """

    v = np.asarray(vertices, float)
    kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    e = np.roll(v, -1, axis=0) - v
    m = v + 0.5*e
    k2 = kx**2 + ky**2
    size = np.max(np.hypot(*(v - v.mean(axis=0)).T))
    small = k2*size**2 < 1e-6
    k2s = np.where(small, 1.0, k2)
    F = np.zeros(kx.shape, complex)
    for (ex, ey), (mx, my) in zip(e, m):
        F += (kx*ey - ky*ex)*np.exp(-1j*(kx*mx + ky*my))*np.sinc((kx*ex + ky*ey)/(2*np.pi))
    F = 1j*F/k2s
    if np.any(small):
        A = polygon_area(v)
        w = v[:, 0]*np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1)*v[:, 1]
        c = np.sum((v + np.roll(v, -1, axis=0))*w[:, None], axis=0)/(6*A)      # centroid
        F = np.where(small, A*np.exp(-1j*(kx*c[0] + ky*c[1])), F)

    return F


def iris_coverage(x, y, n_blades, apothem, blade_radius=np.inf, rotation=0.0, dx=None):
    """Transmission of the iris opening on a grid, antialiased: the fraction
    of each pixel inside the opening, from the signed distance to the
    edges, so that a sampled opening has no staircase edges that would
    diffract on their own.

    Parameters
    ----------
    x, y : arrays
        Coordinates (broadcastable, e.g. from np.meshgrid).
    n_blades, apothem, blade_radius, rotation
        As in `iris()`.
    dx : number, optional
        Pixel size; default from x.

    Returns
    -------
    T : array
    """

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if dx is None:
        dx = np.max(np.abs(np.diff(x, axis=-1))) if x.shape[-1] > 1 else 1.0
    normals, centres = _iris_centres(int(n_blades), float(apothem), float(blade_radius), rotation)
    d = np.full(x.shape, -np.inf)
    for j, n in enumerate(normals):
        if centres is None:
            dj = x*n[0] + y*n[1] - apothem
        else:
            dj = np.hypot(x - centres[j, 0], y - centres[j, 1]) - blade_radius
        d = np.maximum(d, dj)

    return np.clip(0.5 - d/dx, 0.0, 1.0)


def edge_spike(k, edge_length, area, n_parallel=1, separation=0.0):
    """Normalized intensity I(k)/I(0) = abs(F(k)/A)**2 along the diffraction
    spike of a straight edge of length L of an opening of area A, at
    spatial angular frequency k along its normal: (L/(k A))**2 for one edge
    (odd number of blades), 4 sin**2(k s/2) (L/(k A))**2 for two parallel
    opposite edges a distance s apart (even number of blades), which on
    average is twice as bright.

    Parameters
    ----------
    k : array
    edge_length, area : numbers
    n_parallel : 1 or 2
    separation : number
        Distance of the two parallel edges (2 h for a regular polygon).

    Returns
    -------
    I : array
    """

    k = np.asarray(k, float)
    I1 = (edge_length/(k*area))**2
    if n_parallel == 1:
        return I1
    if n_parallel == 2:
        return 4*np.sin(k*separation/2)**2*I1
    raise ValueError("n_parallel must be 1 or 2.")


def spike_count(n_blades):
    """Number of diffraction spikes of an iris with n_blades straight blades:
    every edge streaks both ways along its normal, so 2 n for an odd number
    of blades; for an even number the opposite edges are parallel and their
    streaks coincide, n spikes."""

    n = int(n_blades)

    return n if n % 2 == 0 else 2*n


def rounded_edge_crossover(edge_length, blade_radius):
    """Spatial angular frequency k_c = 2 pi rho/L**2 up to which a rounded
    blade edge (chord L, radius rho) still diffracts like a straight edge:
    the straight edge streak L/k equals the stationary phase amplitude
    sqrt(2 pi rho/k)/k of a curved edge there. Beyond k_c the spike fades
    into a fan of angular width about L/rho and decays as 1/k**3, like the
    rings of a circular opening, instead of 1/k**2."""

    return 2*np.pi*blade_radius/edge_length**2


def iris_edge_length(n_blades, apothem, blade_radius=np.inf):
    """Chord length L of one edge of the iris of `iris()`, corner to corner:
    2 R_c sin(pi/n) with the distance R_c of the corners from the centre
    (2 h tan(pi/n) for straight blades)."""

    R_c = np.hypot(*iris(n_blades, apothem, blade_radius, n_arc=2)[0])

    return 2*R_c*np.sin(np.pi/int(n_blades))


def power_outside(k, perimeter, area):
    """Fraction of the power of the far field pattern of an opening outside
    the spatial angular frequency k (the radius r = k lambda f/(2 pi) in
    the focal plane), for k much larger than 2 pi over the size of the
    opening: P/(pi A k), with perimeter P and area A, whatever the shape.
    The far wings come from the edges, each length of edge sending the
    same power out; a polygon bunches it into spikes, a circle spreads it
    into rings (2/(pi k R), the familiar tail of the encircled energy of
    the Airy pattern).

    Parameters
    ----------
    k : array
    perimeter, area : numbers

    Returns
    -------
    fraction : array
    """

    return perimeter/(np.pi*area*np.asarray(k, float))


def iris_psf(n_blades, apothem, blade_radius=np.inf, rotation=0.0, n=1024, dx=None):
    """Far field (Fraunhofer) intensity of an iris opening by FFT, normalized
    to its centre, I(k)/I(0) = abs(F(k)/A)**2, on the centred grid of spatial
    angular frequencies k = 2 pi m/(n dx). In the focal plane of a lens of
    focal length f this is the point spread function at r = k lambda f/(2 pi).

    The opening is sampled with `iris_coverage()`, whose antialiased edges
    act like a box of width dx across each edge; its transfer
    sinc(k dx/2) is divided out. The spikes then agree with
    `polygon_transform()` to a few percent up to about a quarter of the
    sampling limit pi/dx; towards that limit the edge waves alias (about
    15 % low at half of it).

    Parameters
    ----------
    n_blades, apothem, blade_radius, rotation
        As in `iris()`.
    n : int, optional
        Grid size.
    dx : number, optional
        Pupil sampling; default 2.5 times the corner radius over n/2.

    Returns
    -------
    k : array (n,)
        Spatial angular frequencies (both axes).
    I : array (n, n)
        Normalized intensity, indexed [ky, kx].
    """

    from . import fft as _fft

    R_c = np.max(np.hypot(*iris(n_blades, apothem, blade_radius, rotation, n_arc=8).T))
    if dx is None:
        dx = 2.5*R_c/(n/2)
    x = (np.arange(n) - n//2)*dx
    T = iris_coverage(x[None, :], x[:, None], n_blades, apothem, blade_radius, rotation, dx=dx)
    A = T.sum()*dx**2
    k = 2*np.pi*(np.arange(n) - n//2)/(n*dx)
    K = np.hypot(k[None, :], k[:, None])
    F = _fft.FT(T)*dx**2/np.sinc(K*dx/(2*np.pi))

    return k, np.abs(F/A)**2


def curved_edge_spike(k, edge_length, blade_radius, area):
    """Normalized intensity I(k)/I(0) along the normal through the middle of
    a rounded blade edge (chord L, radius rho) of an opening of area A, from
    the edge integral with the phase of the arc in the parabolic
    (Fresnel) approximation, the edge alone ::

        F = 1/k int_{-L/2}^{L/2} exp(-i k t**2/(2 rho)) dt,

    a Fresnel integral: the straight edge streak (L/(k A))**2 for
    k << k_c = 2 pi rho/L**2 (`rounded_edge_crossover()`), the stationary
    phase limit 2 pi rho/(k**3 A**2) of a smooth curve for k >> k_c, with
    Fresnel wiggles in between. Straight blades are rho = np.inf.

    Parameters
    ----------
    k : array
    edge_length, blade_radius, area : numbers

    Returns
    -------
    I : array
    """

    from scipy.special import fresnel

    k = np.asarray(k, float)
    if not np.isfinite(blade_radius):
        return (edge_length/(k*area))**2
    w = np.sqrt(np.pi*blade_radius/k)
    S, C = fresnel(edge_length/2/w)
    amp = 2*w*np.hypot(C, S)

    return (amp/(k*area))**2


SUN_RADIUS = np.radians(0.533/2)  # angular radius of the Sun (rad)


def sunstar(n_blades, f_number, wavelength, focal_length=24.0, blade_radius=np.inf, rotation=0.0,
            source_radius=None, pixel=0.004, half_width=1.5, n=None, oversampling=2.2):
    """Image of a bright source through an iris: the diffraction pattern of
    the opening (`iris_psf()`, FFT), convolved with the image of the source
    (a uniform disk, the Sun by default) and with the sensor pixels (boxes),
    sampled at the pixel centres. The opening of an f/N stop has the area
    of a circle of diameter f/N.

    Parameters
    ----------
    n_blades : int
    f_number : number
    wavelength : number or array
        Wavelength(s) in mm.
    focal_length : number, optional
        In mm (default 24 mm, a wide angle lens).
    blade_radius : number, optional
        Radius of the blade edges in mm (np.inf: straight blades).
    rotation : number, optional
    source_radius : number, optional
        Radius of the geometric image of the source in mm; default the Sun,
        f tan(0.267 deg); 0 for a point source.
    pixel : number, optional
        Pixel pitch in mm.
    half_width : number, optional
        Half width of the image in mm.
    n : int, optional
        FFT grid (one for all wavelengths); the image plane sampling is
        min(wavelength) N/oversampling,
        so the grid spans n of those, which must exceed 2 half_width. Default:
        the smallest power of 2 that spans 4 half_width (at least 1024, at
        most 4096), so that the folded-back tail of the pattern (the FFT is
        periodic) stays below about 10 % at the edge of the image. (Below
        about f/8 the cap makes the grid smaller than that.)
    oversampling : number, optional

    Returns
    -------
    x : array
        Pixel centres in mm (both axes).
    E : array (n_wavelengths, n_pixels, n_pixels)
        Irradiance relative to that of the geometric image of the source
        (inside the disk of an extended source, about 1); for a point source
        relative to the centre of the diffraction pattern, I/I(0).
    """

    from scipy.special import j1
    from scipy.ndimage import map_coordinates
    from . import fft as _fft

    lam = np.atleast_1d(np.asarray(wavelength, float))
    f = float(focal_length)
    a = f*np.tan(SUN_RADIUS) if source_radius is None else float(source_radius)
    A_stop = np.pi*(f/(2*f_number))**2
    h = iris_apothem(n_blades, A_stop, blade_radius)
    step0 = lam.min()*f_number/oversampling            # image sampling at the shortest wavelength
    if n is None:
        n = int(np.clip(2**int(np.ceil(np.log2(4*half_width/step0))), 1024, 4096))
    if n*step0 < 2*half_width:
        raise ValueError("The FFT grid is too small for the image; increase n.")
    # One pupil grid for all wavelengths: the pattern in k is the same, only
    # its scale in the image, r = k lambda f/(2 pi), grows with the wavelength.
    dx = lam.min()*f/(n*step0)
    _, I0 = iris_psf(n_blades, h, blade_radius, rotation, n=n, dx=dx)     # I/I(0) on the k grid
    otf = _fft.FT(I0)
    m = np.arange(n) - n//2
    n_px = int(round(2*half_width/pixel))
    x = (np.arange(n_px) - (n_px - 1)/2)*pixel
    out = []
    for lw in lam:
        step = lw*f/(n*dx)
        nu = m/(n*step)                                # cycles/mm in the image
        transfer = np.sinc(nu[None, :]*pixel)*np.sinc(nu[:, None]*pixel)
        if a > 0:
            NU = np.hypot(nu[None, :], nu[:, None])
            arg = 2*np.pi*a*np.where(NU == 0, 1.0, NU)
            transfer = transfer*np.where(NU == 0, 1.0, 2*j1(arg)/arg)
        E = np.real(_fft.inv_FT(otf*transfer))         # I/I(0), averaged over source and pixel
        # I/I(0) times the peak of the unit power PSF, A/(lambda f)**2, is the
        # irradiance per unit power; relative to the source disk times pi a**2
        if a > 0:
            E = E*A_stop/(lw*f)**2*np.pi*a**2
        idx = x/step + n//2
        I, J = np.meshgrid(idx, idx, indexing="ij")
        out.append(map_coordinates(E, [I, J], order=1))

    return x, np.array(out)


# --- Flare and ghosts (notebook 92) ----------------------------------------

def _medium_index(system, k):
    """Refractive index in front of surface k of a system (k = len: image space)."""

    return float(np.real(system.n[k]))


def ghost_paths(system, sensor=True):
    """All double reflection ghost paths of a system: light passes the
    surfaces up to j, is reflected back at surface j, reflected forward again
    at an earlier surface i < j and then goes on to the image. Only surfaces
    that change the medium reflect (not stops or dummy surfaces).

    Parameters
    ----------
    system : raytrace.System
    sensor : bool, optional
        Include the sensor (image plane) as a reflecting surface j =
        len(system.surfaces).

    Returns
    -------
    list of (j, i)
    """

    N = len(system.surfaces)
    refl = [k for k in range(N) if system.n[k] != system.n[k + 1]]
    js = refl + ([N] if sensor else [])

    return [(j, i) for j in js for i in refl if i < j]


def ghost_system(system, j, i, coatings=None, sensor_index=1.65):
    """The unfolded system of the ghost path (j, i): the surfaces of `system`
    up to j, a reflection at j, the surfaces back to i in reverse order, a
    reflection at i and the surfaces from i + 1 to the image.

    The reflections and transmissions go through the coatings
    (`coatings.Coating`) of the surfaces, seen from the side the light comes
    from (`Coating.reversed()` on the way back); a reflection is the
    coating on the medium behind it as its substrate. The sensor (j = the
    number of surfaces) is a plane mirror at the image plane, a bare
    surface of index sensor_index (R = 6 % for 1.65) unless coatings[j]
    is given (with its substrate).

    Parameters
    ----------
    system : raytrace.System
        Monochromatic system (the indices are taken at its wavelength).
    j, i : int
        Ghost path, i < j (from `ghost_paths()`).
    coatings : dict {k: Coating}, optional
        Coatings of the surfaces k (in the orientation of the system); by
        default those of the surfaces.
    sensor_index : number, optional

    Returns
    -------
    ghost : raytrace.System
    stops : list of int
        Indices of the copies of the stop in the unfolded system (one to
        three, where the ray passes the stop).
    """

    from copy import copy
    from .coatings import Coating

    S = system.surfaces
    N = len(S)
    if not 0 <= i < j <= N:
        raise ValueError("A ghost path needs 0 <= i < j <= number of surfaces.")
    coatings = {} if coatings is None else coatings
    n = [_medium_index(system, k) for k in range(N + 1)]

    def coat(k):
        return coatings.get(k, S[k].coating if k < N else None)

    def layers(k):
        c = coat(k)
        return [] if c is None else list(c.layers)

    def plain(k, n_after=None):
        s = copy(S[k])
        s.material, s.index_offset = None, 0.0
        s.n_after = n_after if n_after is not None else (n[k + 1] if n[k + 1] != n[k] else None)
        s.coating = coat(k)
        return s

    out, stops = [], []
    for k in range(min(j, N)):
        out.append(plain(k))
        if k == system.stop_index:
            stops.append(len(out) - 1)
    # reflection at j, seen from the medium in front of it
    if j == N:
        c = coat(N)
        m = rt.Surface(np.inf, mirror=True, label="sensor",
                       coating=c if c is not None else Coating((), substrate=sensor_index))
    else:
        m = copy(S[j])
        m.material, m.n_after, m.index_offset, m.mirror = None, None, 0.0, True
        m.coating = Coating(layers(j), substrate=n[j + 1])
    m.thickness = -S[j - 1].thickness
    out.append(m)
    # back to i
    for k in range(j - 1, i, -1):
        s = plain(k, n_after=n[k] if n[k] != n[k + 1] else None)
        s.coating = None if coat(k) is None else coat(k).reversed()
        s.thickness = -S[k - 1].thickness
        out.append(s)
        if k == system.stop_index:
            stops.append(len(out) - 1)
    # reflection at i, from behind
    m = copy(S[i])
    m.material, m.n_after, m.index_offset, m.mirror = None, None, 0.0, True
    m.coating = Coating(layers(i)[::-1], substrate=n[i])
    m.thickness = S[i].thickness
    out.append(m)
    for k in range(i + 1, N):
        out.append(plain(k))
        if k == system.stop_index:
            stops.append(len(out) - 1)
    ghost = rt.System(out, stops[0] if stops else 0, system.wavelength, n_object=system.n_object,
                      entrance_pupil_radius=None)

    return ghost, stops


def flare_rays(system, field, n=201, radius=None):
    """A collimated beam from a distant bright source (field angle in
    radians, or (x, y) angles) that fills the whole front of the lens, not
    only the entrance pupil: ghost light can take any path. A square grid of
    n x n rays across radius (default the clear semi-diameter of the first
    surface plus the shift of an oblique beam).

    Returns
    -------
    rays : raytrace.Rays
    weight : array
        Power of each ray (irradiance 1 across the beam: area of a grid
        cell projected onto the beam cross section).
    """

    fx, fy = rt._field_xy(field)
    d = np.array([np.tan(fx), np.tan(fy), 1.0])
    d = d/np.linalg.norm(d)
    s0 = system.surfaces[0]
    if radius is None:
        sd = s0.semi_diameter if s0.semi_diameter is not None else 1.5*system.paraxial().ep_radius
        radius = sd + 10.0*np.hypot(d[0], d[1])/d[2]
    u = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(u, u)
    X, Y = X.ravel(), Y.ravel()
    z0 = -10.0
    P = np.column_stack([X - 10.0*d[0]/d[2], Y - 10.0*d[1]/d[2], np.full(X.size, z0)])
    D = np.tile(d, (X.size, 1))
    cell = (u[1] - u[0])**2

    return rt.Rays(P, D, np.zeros(X.size)), np.full(X.size, cell*d[2])


def _in_iris(bundle, stops, iris_shape):
    if iris_shape is None:
        return np.ones(len(bundle.valid), bool)
    n_blades, apothem, blade_radius, rotation = iris_shape
    ok = np.ones(len(bundle.valid), bool)
    for k in stops:
        P = bundle.points[k + 1]
        cov = iris_coverage(P[:, 0], P[:, 1], n_blades, apothem, blade_radius, rotation, dx=1e-6)
        ok &= np.nan_to_num(cov) > 0.5
    return ok


def trace_flare(system, rays, weight, path=None, coatings=None, iris_shape=None,
                sensor_index=1.65):
    """Trace a beam through the lens directly (path None) or along a ghost
    path (j, i) with the polarization ray trace (`System.trace_polarization`,
    unpolarized light), through the coatings.

    Parameters
    ----------
    system : raytrace.System
    rays, weight : from `flare_rays()`
    path : (j, i), optional
    coatings : dict {k: Coating}, optional
    iris_shape : (n_blades, apothem, blade_radius, rotation), optional
        An iris in the stop (see `iris()`): rays outside it are blocked at
        every pass through the stop (the stop's semi-diameter must not cut
        them first).
    sensor_index : number, optional

    Returns
    -------
    x, y : arrays
        Where the rays that get through hit the image plane (image frame).
    power : array
        Their power (weight times the transmittance of the path).
    """

    if path is None:
        sys_ = system
        if coatings:
            from copy import copy
            surf = [copy(s) for s in system.surfaces]
            for k, c in coatings.items():
                if k < len(surf):
                    surf[k].coating = c
            sys_ = rt.System(surf, system.stop_index, system.wavelength, n_object=system.n_object)
        stops = [system.stop_index]
    else:
        sys_, stops = ghost_system(system, *path, coatings=coatings, sensor_index=sensor_index)
    pb = sys_.trace_polarization(rays)
    T = pb.transmittance()
    b = pb.bundle if hasattr(pb, "bundle") else pb
    ok = np.isfinite(T) & _in_iris(b, stops, iris_shape)
    P = b.points[-1][ok]
    _, (o, R) = sys_.frames()
    Q = (P - o) @ R

    return Q[:, 0], Q[:, 1], weight[ok]*T[ok]


# Antireflection coatings of notebook 92, layers from the air side; indices
# of evaporated films (dispersion neglected): MgF2 1.38, ZrO2 2.05, Al2O3 1.63.
AR_DESIGNS = {
    "uncoated": [],
    "MgF2": [(1.38, 99.6e-6)],                                       # quarter wave at 550 nm
    "multilayer": [(1.38, 94.0e-6), (2.05, 126.7e-6), (1.63, 160.1e-6)],  # broadband, on n = 1.68
}


def ar_coatings(system, design="multilayer"):
    """Antireflection coatings on all air-glass surfaces of a system,
    oriented as `coatings.Coating` expects (layers from the medium in front
    of the surface): from the air side on surfaces light enters glass, in
    reverse on those where it leaves. Cemented surfaces stay bare.

    Designs (`AR_DESIGNS`, constant film indices, thicknesses in mm):
    "uncoated"; "MgF2", a quarter wave of MgF2 (n = 1.38) at 550 nm, about
    0.35 % at 550 nm and 0.7 % averaged over 420-680 nm on N-LAK9; and
    "multilayer", a three layer broadband coating (MgF2 / ZrO2 / Al2O3 from
    the air side, a quarter-half-quarter design optimized for n = 1.68),
    0.19 % averaged over 420-680 nm and below 0.4 % everywhere in it.

    Parameters
    ----------
    system : raytrace.System
    design : str or list of (index, thickness)

    Returns
    -------
    dict {k: Coating or None}
    """

    from .coatings import Coating, Layer

    layers = AR_DESIGNS[design] if isinstance(design, str) else design
    out = {}
    for k in range(len(system.surfaces)):
        n1, n2 = _medium_index(system, k), _medium_index(system, k + 1)
        if n1 == n2 or (n1 != 1.0 and n2 != 1.0):
            continue
        ls = [Layer(n, t) for n, t in layers]
        out[k] = Coating(ls if n1 == 1.0 else ls[::-1]) if ls else None

    return out


def _sp_spectra(sys_, bundle, ok, wavelengths_nm, n_table=181):
    """Transmittance of each ray along an (unfolded) system for s and p
    polarization separately at the given wavelengths, from the angle of
    incidence at every surface and the coatings' reflectance and
    transmittance (tabulated over the angle). Neglects the mixing of s and p
    between surfaces of different planes of incidence (skew rays) and the
    dispersion of the glasses; see `ghost_footprint()`."""

    from .coatings import Coating

    lam_nm = np.atleast_1d(np.asarray(wavelengths_nm, float))
    lam = lam_nm*1e-6
    n = [_medium_index(sys_, k) for k in range(len(sys_.surfaces) + 1)]
    frames = sys_.frames()[0]
    Ts = np.ones((int(ok.sum()), lam.size))
    Tp = np.ones_like(Ts)
    th_grid = np.linspace(0.0, np.pi/2*0.999, n_table)
    for m, s in enumerate(sys_.surfaces):
        if isinstance(s, rt.CoordinateBreak):
            continue
        c = s.coating
        if not s.mirror and n[m] == n[m + 1] and c is None:
            continue
        o, R = frames[m]
        Q = (bundle.points[m + 1][ok] - o) @ R
        cos_i = np.clip(np.abs(np.sum(s.normal(Q)*(bundle.directions[m][ok] @ R), axis=-1)), 0, 1)
        th = np.arccos(cos_i)
        c = Coating() if c is None else c
        for q, (lw, lwn) in enumerate(zip(lam, lam_nm)):
            if s.mirror:
                p = c.power(th_grid, n[m], None, lw, lwn)
                ks, kp = "R_s", "R_p"
            else:
                p = c.power(th_grid, n[m], n[m + 1], lw, lwn)
                ks, kp = "T_s", "T_p"
            Ts[:, q] *= np.interp(th, th_grid, np.nan_to_num(np.real(p[ks])))
            Tp[:, q] *= np.interp(th, th_grid, np.nan_to_num(np.real(p[kp])))

    return Ts, Tp


def _path_system(system, path, coatings, sensor_index, iris_shape):
    """The (unfolded) system of a path and the indices of its stop copies."""

    from copy import copy

    if path is None:
        surf = [copy(s) for s in system.surfaces]
        for k, c in (coatings or {}).items():
            if k < len(surf):
                surf[k].coating = c
        sys_ = rt.System(surf, system.stop_index, system.wavelength, n_object=system.n_object)
        stops = [system.stop_index]
    else:
        sys_, stops = ghost_system(system, *path, coatings=coatings, sensor_index=sensor_index)
    if iris_shape is not None:
        for k in stops:                     # the iris, not the round stop, limits the beam
            sys_.surfaces[k] = copy(sys_.surfaces[k])
            sys_.surfaces[k].semi_diameter = None

    return sys_, stops


def _flare_sample(system, sys_, stops, field, n_rays, rng, source_radius, iris_shape, n_coarse):
    """Rays of a distant source that fill only the part of the front of the
    lens that feeds the path of sys_ (coarse grid, then random points in the
    cells that pass and their neighbours), with directions over the source
    disk, and their powers (irradiance 1 across the beam)."""

    from scipy.ndimage import binary_dilation

    rays, _ = flare_rays(system, field, n=n_coarse)
    b = sys_.trace(rays)
    good = b.valid & _in_iris(b, stops, iris_shape)
    P0 = rays.points
    du = np.unique(P0[:, 0])[1] - np.unique(P0[:, 0])[0]
    grid = binary_dilation(good.reshape(n_coarse, n_coarse), iterations=1)
    cells = np.argwhere(grid)
    if len(cells) == 0:
        return None, None
    pick = cells[rng.integers(len(cells), size=n_rays)]
    base = P0.reshape(n_coarse, n_coarse, 3)[pick[:, 0], pick[:, 1]]
    X = base[:, 0] + (rng.random(n_rays) - 0.5)*du
    Y = base[:, 1] + (rng.random(n_rays) - 0.5)*du
    fx, fy = rt._field_xy(field)
    rr = source_radius*np.sqrt(rng.random(n_rays))
    phi = 2*np.pi*rng.random(n_rays)
    D = np.column_stack([np.tan(fx + rr*np.cos(phi)), np.tan(fy + rr*np.sin(phi)), np.ones(n_rays)])
    D = D/np.linalg.norm(D, axis=1)[:, None]
    z0 = base[0, 2]
    d0 = rays.directions[0]
    # each ray crosses z = 0 where the ray of the central direction would
    x0 = X - z0*d0[0]/d0[2]
    y0 = Y - z0*d0[1]/d0[2]
    P = np.column_stack([x0 + z0*D[:, 0]/D[:, 2], y0 + z0*D[:, 1]/D[:, 2], np.full(n_rays, z0)])
    weight = np.full(n_rays, len(cells)*du**2/n_rays*d0[2])

    return rt.Rays(P, D, np.zeros(n_rays)), weight


GhostFootprint = namedtuple("GhostFootprint", ["x", "y", "power", "spectrum", "rays"])
GhostFootprint.__doc__ = """Rays of a bright source along one path (direct or ghost).

- x, y: where they hit the sensor (mm, image frame).
- power: power of each ray at the wavelength of the system, from the
  polarization ray trace (unpolarized), in units of the irradiance of the
  source at the lens times mm^2.
- spectrum: (n, n_wavelengths) the same at other wavelengths (s and p
  chains), or None.
- rays: number of rays launched.
"""


def ghost_footprint(system, field, path=None, coatings=None, n_rays=20000, rng=None,
                    source_radius=0.0, iris_shape=None, wavelengths_nm=None, sensor_index=1.65,
                    n_coarse=81):
    """Trace a bright distant source (a disk of angular radius source_radius
    around the field angle, e.g. the Sun) along the direct path or a ghost
    path (j, i) of a lens, sampling only the part of the front aperture that
    feeds that path.

    A coarse grid over the whole front of the lens (`flare_rays()`) finds
    the cells whose rays reach the sensor along the path; n_rays rays then
    fill those cells (and their neighbours) at random, with directions
    spread over the source disk. They are traced with the polarization ray
    trace through the coatings (unpolarized light). With wavelengths_nm,
    the power of each ray is also evaluated at those wavelengths from s and
    p chains of the coating reflectances and transmittances at the traced
    angles of incidence (the ray geometry of the system's wavelength).

    Parameters
    ----------
    system : raytrace.System
    field : number or (x, y)
        Direction of the source (radians).
    path : (j, i), optional
        Ghost path; None: the direct image.
    coatings : dict {k: Coating}, optional
        E.g. from `ar_coatings()`.
    n_rays : int, optional
    rng : numpy Generator, optional
    source_radius : number, optional
        Angular radius of the source (`SUN_RADIUS` for the Sun).
    iris_shape : (n_blades, apothem, blade_radius, rotation), optional
        See `trace_flare()`.
    wavelengths_nm : array, optional
    sensor_index : number, optional
    n_coarse : int, optional

    Returns
    -------
    GhostFootprint
    """

    rng = np.random.default_rng() if rng is None else rng
    sys_, stops = _path_system(system, path, coatings, sensor_index, iris_shape)
    dense, weight = _flare_sample(system, sys_, stops, field, n_rays, rng, source_radius, iris_shape,
                                  n_coarse)
    if dense is None:
        return GhostFootprint(np.zeros(0), np.zeros(0), np.zeros(0),
                              None if wavelengths_nm is None else np.zeros((0, len(wavelengths_nm))),
                              0)
    pb = sys_.trace_polarization(dense)
    T = pb.transmittance()
    ok = np.isfinite(T) & _in_iris(pb.bundle, stops, iris_shape)
    _, (o, R) = sys_.frames()
    Q = (pb.bundle.points[-1][ok] - o) @ R
    spec = None
    if wavelengths_nm is not None:
        Ts, Tp = _sp_spectra(sys_, pb.bundle, ok, wavelengths_nm)
        spec = weight[ok][:, None]*0.5*(Ts + Tp)

    return GhostFootprint(Q[:, 0], Q[:, 1], weight[ok]*T[ok], spec, n_rays)


def _normal_RT(coating, n1, n2, wavelength, substrate=None):
    from .coatings import Coating
    c = Coating() if coating is None else coating
    if substrate is not None:
        c = Coating(c.layers, substrate=substrate)
        p = c.power(0.0, n1, None, wavelength)
        return float(np.real(p["R_s"]))
    p = c.power(0.0, n1, n2, wavelength)
    return float(np.real(p["R_s"])), float(np.real(p["T_s"]))


GhostInfo = namedtuple("GhostInfo", ["path", "R_j", "R_i", "transmittance", "magnification",
                                     "focus", "irradiance"])
GhostInfo.__doc__ = """Paraxial ghost analysis of one ghost path.

- path: (j, i).
- R_j, R_i: reflectances of the two ghost surfaces at normal incidence.
- transmittance: R_j R_i times the transmittances of all surfaces passed.
- magnification: a = y_image/y_in of a paraxial ray parallel to the axis
  at height y_in; the ghost of a collimated beam of radius h is a blur of
  radius abs(a) h on the sensor.
- focus: axial distance of the ghost focus from the sensor (mm, > 0
  behind it, inf for a collimated ghost).
- irradiance: ghost irradiance relative to the image of a white diffuse
  surface lit by the same source, 4 N**2 transmittance/(T a**2), with the
  transmittance T of the direct path.
"""


def ghost_analysis(system, coatings=None, sensor=True, sensor_index=1.65, f_number=None):
    """Paraxial ghost analysis (as in ghost focus programs): for every
    double reflection path, the reflectances and transmittances at normal
    incidence through the coatings, the paraxial blur of an axial
    collimated beam and the resulting ghost irradiance.

    A source of irradiance E_0 at the lens sends E_0 pi y_in**2 through a
    ghost path into a blur of radius abs(a) y_in, whatever limits y_in (the
    stop, at one of its passes): the ghost irradiance is E_0 R_i R_j T_g/a**2,
    independent of the aperture. The image of a white diffuse surface lit
    by the same source is E_0 T/(4 N**2). Their ratio grows as N**2: a lens
    shows its ghosts more when it is stopped down (and the exposure made
    longer).

    Parameters
    ----------
    system : raytrace.System
    coatings : dict {k: Coating}, optional
    sensor : bool, optional
    sensor_index : number, optional
    f_number : number, optional
        Default: the paraxial F-number of the system.

    Returns
    -------
    list of GhostInfo, and the transmittance T of the direct path
    """

    from .coatings import Coating

    coatings = {} if coatings is None else coatings
    S = system.surfaces
    N = len(S)
    n = [_medium_index(system, k) for k in range(N + 1)]
    wl = system.wavelength
    R, T = {}, {}
    for k in range(N):
        if n[k] != n[k + 1]:
            c = coatings.get(k, S[k].coating)
            R[k], T[k] = _normal_RT(c, n[k], n[k + 1], wl)
    if sensor:
        c = coatings.get(N)
        R[N] = _normal_RT(c if c is not None else Coating(), n[N], None, wl,
                          substrate=sensor_index if c is None else c.substrate)
    T_direct = np.prod([T[k] for k in T])
    F = system.paraxial().f_number if f_number is None else f_number
    out = []
    for j, i in ghost_paths(system, sensor):
        Tg = R[j]*R[i]
        for k in list(range(min(j, N))) + list(range(i + 1, min(j, N))) + list(range(i + 1, N)):
            Tg *= T.get(k, 1.0)
        g, _ = ghost_system(system, j, i, coatings=coatings, sensor_index=sensor_index)
        ys, us = g._paraxial_trace(1.0, 0.0)
        a = ys[-1] + us[-1]*g.surfaces[-1].thickness
        u = us[-1]
        focus = -a/u if u != 0 else np.inf
        irr = 4*F**2*Tg/(T_direct*a**2) if a != 0 else np.inf
        out.append(GhostInfo((j, i), R[j], R[i], Tg, a, focus, irr))

    return out, T_direct


def ghost_spectra(system, field, path, coating_sets, wavelengths_nm, n_rays=20000, rng=None,
                  source_radius=0.0, iris_shape=None, sensor_index=1.65, n_coarse=81):
    """Like `ghost_footprint()`, for several sets of coatings at once and
    only with the spectral s and p chains: the rays are traced once (the
    coatings do not change their paths) and weighted with the reflectances
    and transmittances of each set of coatings at each wavelength. Fast
    enough for rendering all ghosts of a lens.

    Parameters
    ----------
    system, field, path
        As in `ghost_footprint()`.
    coating_sets : list of dict {k: Coating}
    wavelengths_nm : array
    n_rays, rng, source_radius, iris_shape, sensor_index, n_coarse
        As in `ghost_footprint()`.

    Returns
    -------
    x, y : arrays
        Sensor coordinates of the rays that get through (mm).
    spectra : list of arrays (n, n_wavelengths)
        Power of each ray at each wavelength, one array per coating set.
    """

    rng = np.random.default_rng() if rng is None else rng
    sys0, stops = _path_system(system, path, coating_sets[0], sensor_index, iris_shape)
    dense, weight = _flare_sample(system, sys0, stops, field, n_rays, rng, source_radius, iris_shape,
                                  n_coarse)
    if dense is None:
        return np.zeros(0), np.zeros(0), [np.zeros((0, len(wavelengths_nm))) for _ in coating_sets]
    b = sys0.trace(dense)
    ok = b.valid & _in_iris(b, stops, iris_shape)
    _, (o, R) = sys0.frames()
    Q = (b.points[-1][ok] - o) @ R
    spectra = []
    for cs in coating_sets:
        sys_c, _ = _path_system(system, path, cs, sensor_index, iris_shape)
        Ts, Tp = _sp_spectra(sys_c, b, ok, wavelengths_nm)
        spectra.append(weight[ok][:, None]*0.5*(Ts + Tp))

    return Q[:, 0], Q[:, 1], spectra


# --- depth of field, diffraction and tilted lenses (notebook 94) ---

def hyperfocal_distance(f, f_number, c):
    """Thin lens hyperfocal distance f**2/(N c) + f: focused there, the far
    limit of the depth of field is at infinity (distances from the lens).

    Parameters
    ----------
    f : number
        Focal length.
    f_number : number
    c : number
        Acceptable circle of confusion (blur diameter on the sensor).
    """

    return f**2/(f_number*c) + f


def dof_limits(f, f_number, c, focus_distance):
    """Near and far limits of the depth of field of a thin lens: the
    distances at which `blur_diameter()` reaches c,
    1/s = 1/s_f +- (N c/f**2) (1 - f/s_f) (np.inf for a far limit beyond
    infinity).

    Returns
    -------
    near, far : numbers or arrays
    """

    s_f = np.asarray(focus_distance, float)
    inv_sf = np.where(np.isinf(s_f), 0.0, 1/s_f)
    k = f_number*c/f**2*(1 - f*inv_sf)
    inv_far = inv_sf - k
    with np.errstate(divide="ignore"):
        far = np.where(inv_far > 0, 1/np.where(inv_far > 0, inv_far, 1.0), np.inf)

    return 1/(inv_sf + k), far


def geometric_defocus_mtf(freq, c):
    """MTF of a uniform blur disk of diameter c: abs(2 J1(x)/x), x = pi c f
    (the geometric limit of large defocus)."""

    from scipy.special import j1

    x = np.pi*c*np.abs(np.asarray(freq, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.where(x == 0, 1.0, 2*j1(x)/np.where(x == 0, 1.0, x)))


def defocus_mtf(freq, f_number, defocus, wavelength, n_pupil=128):
    """Wave optical MTF of an aberration free lens with the sensor moved by
    `defocus` from the focus (Hopkins): the autocorrelation of the pupil
    with the defocus W020 = defocus/(8 N**2) (paraxial), computed by FFT.

    Parameters
    ----------
    freq : array
        Spatial frequencies on the sensor (cycles per unit of length).
    f_number : number
        Image space F-number N = 1/(2 NA).
    defocus : number
        Axial distance of the sensor from the focus (same unit).
    wavelength : number
    n_pupil : int, optional
        Samples across the pupil diameter.

    Returns
    -------
    array
        MTF at freq (radial; zero beyond the cut-off 1/(lambda N)).
    """

    R = n_pupil/2
    n = 4*int(R) + 8
    u = np.arange(n) - n/2
    X, Y = np.meshgrid(u, u)
    rho2 = (X**2 + Y**2)/R**2
    W = defocus/(8*f_number**2)/wavelength*rho2
    P = np.where(rho2 <= 1, np.exp(2j*np.pi*W), 0.0)
    A = np.fft.ifft2(np.abs(np.fft.fft2(P))**2)
    otf = np.abs(A[0, :n//2 + 1])/np.abs(A[0, 0])
    shift = np.arange(n//2 + 1)                     # pupil shift in samples, 2 R at the cut-off
    f_shift = shift/(2*R)/(wavelength*f_number)

    return np.interp(np.abs(np.asarray(freq, float)), f_shift, otf, right=0.0)


def optimum_f_number(image_depth, wavelength):
    """F-number that balances defocus and diffraction for a depth image_depth
    between the images of the nearest and farthest points (the sensor in
    the middle): the geometric blur image_depth/(2 N) equals the Airy
    diameter 2.44 lambda N, N = sqrt(image_depth/(4.88 lambda)) (Hansma)."""

    return np.sqrt(image_depth/(4.88*wavelength))


def lens_normal(tilt):
    """Axis of a lens tilted by `tilt` about x, (0, sin, cos): the front
    tips down, so that the plane of sharp focus turns towards the ground
    (y < 0)."""

    return np.array([0.0, np.sin(tilt), np.cos(tilt)])


def tilted_conjugate(points, f, tilt):
    """Conjugate points of a thin lens of focal length f at the origin,
    tilted by `tilt` (`lens_normal()`): a point at the axial distance a from
    the lens plane is imaged to a' = f a/(a - f) on the other side, on the
    line through the lens centre. The map takes object points (in front,
    z < 0) to image points and image points back to object points."""

    Q = np.atleast_2d(np.asarray(points, float))
    n = lens_normal(tilt)
    a = np.abs(Q @ n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ap = f*a/(a - f)
        return -(ap/a)[:, None]*Q


def hinge_distance(f, tilt):
    """Merklinger's hinge rule: the plane of sharp focus of a lens tilted by
    `tilt` always contains the line where the front focal plane meets the
    plane through the lens parallel to the sensor, J = f/sin(tilt) below
    the lens."""

    return f/np.sin(tilt)


def plane_of_focus(f, tilt, image_distance):
    """Plane of sharp focus of a thin lens tilted by `tilt` with the sensor
    plane z = image_distance (perpendicular to the camera axis z): the
    object side image of the sensor plane, in the y-z plane.

    Returns
    -------
    point : array (2,)
        (y, z) of the point where it meets the plane z = 0 of the lens
        centre (the hinge line, y = -J).
    angle : float
        Angle of the plane of focus to the camera axis (rad; 0: parallel
        to the axis, pi/2: parallel to the sensor), > 0 when it rises with
        distance in front of the camera (z < 0).
    scheimpflug_y : float
        Height of the line where the sensor and the lens plane meet,
        -image_distance/tan(tilt) (the plane of focus passes through it).
    """

    P = np.array([[0.0, 0.0, image_distance], [0.0, 10.0, image_distance]])
    Q = tilted_conjugate(P, f, tilt)
    d = Q[1, 1:] - Q[0, 1:]
    t = -Q[0, 2]/d[1]
    y0 = Q[0, 1] + t*d[0]
    angle = np.arctan2(d[0], -d[1])
    angle = (angle + np.pi/2) % np.pi - np.pi/2

    return np.array([y0, 0.0]), angle, -image_distance/np.tan(tilt)


def tilted_blur(points, f, f_number, tilt, image_distance, n=64):
    """Geometric blur of object points through a thin lens tilted by `tilt`
    (aperture f/N, round, in the lens plane) on the sensor plane
    z = image_distance: the rays from the aperture rim to the image point,
    cut by the sensor (an ellipse). Returns its largest diameter."""

    Q = np.atleast_2d(np.asarray(points, float))
    P = tilted_conjugate(Q, f, tilt)
    nrm = lens_normal(tilt)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(nrm, e1)
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    A = 0.5*f/f_number*(np.cos(phi)[:, None]*e1 + np.sin(phi)[:, None]*e2)     # (n, 3)
    t = (image_distance - A[None, :, 2])/(P[:, None, 2] - A[None, :, 2])
    S = A[None] + t[..., None]*(P[:, None, :] - A[None])
    half = n//2
    return np.max(np.linalg.norm(S[:, :half, :2] - S[:, half:, :2], axis=-1), axis=1)


def tilt_lens(system, tilt, pivot, image_distance=None):
    """Copy of a lens tilted by `tilt` about the x axis through the point
    `pivot` on its axis (mm behind the first vertex), the sensor staying
    perpendicular to the camera axis, image_distance behind the pivot
    (default: where it was). Built with coordinate breaks; the tilt follows
    `lens_normal()` (the front of the lens tips down for tilt > 0).

    Returns
    -------
    raytrace.System
        For finite object points traced with `trace_point()`.
    """

    S = [copy.copy(s) for s in system.surfaces]
    L = sum(s.thickness for s in S[:-1])
    if image_distance is None:
        image_distance = L + S[-1].thickness - pivot
    S[-1].thickness = pivot - L
    surfaces = ([rt.CoordinateBreak(thickness=pivot), rt.CoordinateBreak(tilt=(-tilt, 0.0, 0.0),
                                                                        thickness=-pivot)]
                + S + [rt.CoordinateBreak(tilt=(tilt, 0.0, 0.0), thickness=image_distance)])

    return rt.System(surfaces, system.stop_index + 2, system.wavelength, object_distance=1e4,
                     object_height=0.0, entrance_pupil_radius=None)


def trace_point(system, point, n_rays=4000, rng=None, search_radius=30.0, plane_z=0.0):
    """Rays from an object point (x, y, z), z < 0 in front of the first
    vertex, through a system (also a tilted one): a coarse search for the
    part of the plane z = plane_z that leads through the system, then
    n_rays random rays through it.

    Returns
    -------
    x, y : arrays
        Hits on the image plane (its local frame).
    """

    rng = np.random.default_rng() if rng is None else rng
    O = np.asarray(point, float)

    def launch(X, Y):
        E = np.column_stack([X, Y, np.full(len(X), plane_z)])
        D = E - O
        D = D/np.linalg.norm(D, axis=1)[:, None]
        return rt.Rays(np.tile(O, (len(X), 1)), D, np.zeros(len(X)))

    g = np.linspace(-search_radius, search_radius, 81)
    X, Y = [a.ravel() for a in np.meshgrid(g, g)]
    ok = system.trace(launch(X, Y)).valid
    if not np.any(ok):
        return np.zeros(0), np.zeros(0)
    h = g[1] - g[0]
    X = rng.uniform(X[ok].min() - h, X[ok].max() + h, n_rays)
    Y = rng.uniform(Y[ok].min() - h, Y[ok].max() + h, n_rays)
    b = system.trace(launch(X, Y))
    Q = system.to_image_frame(b.points[-1][b.valid])

    return Q[:, 0], Q[:, 1]
