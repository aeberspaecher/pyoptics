#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Photographic optics: the out-of-focus image of a point (bokeh) and the
lens that makes it.

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
Press (1999), sections 8.8 (defocused images) and 9.2 (aberrations).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt


__all__ = ["double_gauss", "focus", "blur_diameter", "blur_diameter_pupils", "point_footprint", "splat",
           "relative_illumination", "vignetting_circles", "vignetted_fraction",
           "defocus_profile", "defocused_psf", "RippledSurface", "onion_ring_contrast",
           "converging_beam_equivalent", "gaussian_apodizer", "apodizer_mean_transmission",
           "t_number", "apodizer_for_t_number", "DOUBLE_GAUSS_F", "D_LINE"]


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

        c = m_p D |x_s - x'|/(x' + m_p f).

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
    radius |dW/drho| = |2 W020 rho + 4 W040 rho**3| (in lambda/NA), and the
    rays of the annulus 2 rho d rho spread over 2 pi r dr, so

        E(r) = sum over the rays that land at r of T(rho) rho/(pi r |dr/drho|),

    normalized to the power through the pupil (integral of E over the plane
    = 1). Where dr/drho = 0 (a caustic, when |W020| < 6 |W040| with opposite
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
        the blur radius is about 2 |W020| + 4 |W040| lambda/NA.
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
    |z_eff| = R (R + Delta)/|Delta|, about R**2/|Delta|.

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
