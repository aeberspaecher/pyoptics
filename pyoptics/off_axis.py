#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Off-axis mirror systems: off-axis conic segments, field-biased and
aperture-offset layouts, unobscured three-mirror anastigmats with freeform
surfaces, ring fields and full-field displays
(notebook 155, off-axis mirrors and nodal aberrations).

An unobscured reflective system is usually built from a rotationally
symmetric parent (a coaxial system whose first order layout and third order
aberrations come from `raytrace`) of which only off-axis parts are used:

- *Aperture offset*: the stop is decentred, every mirror is used off its
  axis, the field stays centred (an off-axis paraboloid, the off-axis
  Gregorian). Here a plane dummy surface with `decenter` and a
  `semi_diameter` in front of the mirrors acts as the stop; ray aiming
  finds its centre (`off_axis_parabola()`).
- *Field bias*: the stop stays on the axis, the field is centred off the
  axis (`rectangular_field()` with a centre, `ring_field()`), and the mirrors
  in front of and behind the stop are used on opposite sides of the axis
  (the Cook three-mirror anastigmat, `cook_tma()`). The light then clears
  the mirrors if the field offset is large enough (`beam_clearance()`).

The surfaces stay segments of their parents, so `raytrace.System.seidel()`
and the nodal aberration theory of `nodal` apply to them. Freeform
departures over the used footprint are `raytrace.ZernikeSurface` terms with
a `centre` in the footprint (`freeform_mirrors()`); their coefficients,
the tilts and decentres of the mirrors and the image plane tilt are
variables of `lens_optimization.DampedLeastSquares`
(`lens_optimization.ZernikeTerm`, `Tilt`, `Decenter`, `ImageTilt`), and
`field_wavefront()`, `focal_length()` and `clearance_penalty()` are
operands over a two-dimensional field.

Closed forms used for checks (Malacara, "Optical Shop Testing", ch. 18):

- An off-axis segment of a conic of vertex radius R and conic constant K,
  centred at the height h, is tilted by the normal angle
  arctan(h/sqrt(R^2 - K h^2)) (`off_axis_conic()`) and has the local radii
  of curvature R_s = sqrt(R^2 - K h^2) (sagittal) and R_t = R_s^3/R^2
  (tangential, in the plane of the offset). In the tangent frame of the
  segment its sag over a disk of radius a is, to second order,
  Z4 a^2 (1/R_s + 1/R_t)/8 + Z5 a^2 (1/R_s - 1/R_t)/4 in Fringe Zernikes
  (`segment_zernikes()`): an off-axis paraboloid is an astigmatic (and
  comatic) cap, which is why it is hard to make.
- Field points: a rotationally symmetric parent has aberrations that depend
  only on |H|, so an arc of a ring around the axis (a ring field) sees the
  same correction along its whole length.

Functions:

- rectangular_field(), ring_field(): two-dimensional field grids.
- off_axis_conic(), segment_sag(), segment_zernikes(): the shape of an
  off-axis segment of a conic in its tangent frame.
- off_axis_parabola(): an aperture-offset paraboloid as a `System`.
- solve_conics(): conic constants from linear Seidel conditions.
- cook_tma(): a field-biased Cook three-mirror anastigmat.
- freeform_mirrors(): mirrors as `ZernikeSurface` with Zernikes centred
  on their footprints, ready for optimization.
- footprints(), beam_clearance(): footprints of the mirrors and the
  clearance of the other beams around them.
- field_zernikes(), field_rms(): Fringe Zernikes and RMS wavefront errors
  over a field grid; plot_full_field(): full-field displays of
  astigmatism, coma or RMS; draw_layout(): the meridional section with
  the mirrors drawn over their footprints only.
- field_wavefront(), focal_length(), clearance_penalty(): operands for
  `lens_optimization.DampedLeastSquares` over a two-dimensional field.

References
----------
D. R. Cook, "Three-mirror anastigmat used off-axis in aperture and field",
US 4,265,510 (1981).

L. G. Cook, "The last three-mirror anastigmat (TMA)?", Proc. SPIE CR41,
310-324 (1992).

K. P. Thompson, "Description of the third-order optical aberrations of
near-circular pupil optical systems without symmetry", J. Opt. Soc. Am. A
22, 1389-1401 (2005).

K. Fuerschbach, J. P. Rolland and K. P. Thompson, "A new family of optical
systems employing φ-polynomial surfaces", Opt. Express 19, 21919-21928
(2011).

A. Bauer, E. M. Schiesser and J. P. Rolland, "Starting geometry creation and
design method for freeform optics", Nat. Commun. 9, 1756 (2018).

D. Malacara (ed.), "Optical Shop Testing", 3rd ed., Wiley (2007), chapter 18
(aspheric surfaces).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt
from .bases import fringe_zernike


__all__ = ["rectangular_field", "ring_field", "OffAxisConic", "off_axis_conic", "segment_sag",
           "segment_zernikes", "off_axis_parabola", "solve_conics", "cook_tma",
           "freeform_mirrors", "Footprint", "footprints", "Clearance", "beam_clearance",
           "field_zernikes", "field_rms", "plot_full_field", "draw_layout", "field_wavefront",
           "focal_length", "clearance_penalty", "pupil_rings"]


# --- fields -----------------------------------------------------------------

def rectangular_field(centre=(0.0, 0.0), half_widths=(0.0, 0.0), n=(5, 5)):
    """Field points on a rectangular grid.

    Parameters
    ----------
    centre : (x, y), optional
        Centre of the field (radians for an object at infinity, length units
        for a finite object), e.g. the field bias of an off-axis system.
    half_widths : (hx, hy), optional
        Half widths of the field in x and y.
    n : (nx, ny), optional
        Points across x and y (1 gives the centre line).

    Returns
    -------
    fields : (nx*ny, 2) array
        (x, y) pairs, x running fastest.
    """

    nx, ny = (n, n) if np.isscalar(n) else n
    x = centre[0] + (np.linspace(-1, 1, nx) if nx > 1 else np.zeros(1))*half_widths[0]
    y = centre[1] + (np.linspace(-1, 1, ny) if ny > 1 else np.zeros(1))*half_widths[1]
    X, Y = np.meshgrid(x, y)

    return np.column_stack([X.ravel(), Y.ravel()])


def ring_field(radius, half_angle, width=0.0, n_arc=9, n_radial=1, azimuth=np.pi/2):
    """Field points on an arc of a ring around the axis (ring field).

    Parameters
    ----------
    radius : number
        Radius of the centre of the ring (an angle or an object height).
    half_angle : number
        Half the azimuthal extent of the arc (radians).
    width : number, optional
        Radial width of the ring (0: one arc).
    n_arc, n_radial : int, optional
        Points along the arc and across its width.
    azimuth : number, optional
        Azimuth of the middle of the arc, counted from +x towards +y
        (default pi/2: the arc is centred on the +y axis).

    Returns
    -------
    fields : (n_arc*n_radial, 2) array
        (x, y) pairs.
    """

    a = azimuth + (np.linspace(-1, 1, n_arc) if n_arc > 1 else np.zeros(1))*half_angle
    r = radius + (np.linspace(-0.5, 0.5, n_radial) if n_radial > 1 else np.zeros(1))*width
    A, Rr = np.meshgrid(a, r)

    return np.column_stack([(Rr*np.cos(A)).ravel(), (Rr*np.sin(A)).ravel()])


# --- off-axis segments of conics ---------------------------------------------

OffAxisConic = namedtuple("OffAxisConic", ["point", "normal_angle", "R_sagittal",
                                           "R_tangential"])
OffAxisConic.__doc__ = """Local geometry of a conic at an off-axis point.

Attributes
----------
point : (y, z) of the point on the parent surface (vertex at the origin)
normal_angle : angle of the surface normal with the axis (radians), the
    tilt of the tangent frame (positive for a surface that bends towards +z
    at +y, e.g. R > 0)
R_sagittal, R_tangential : local radii of curvature (signed like R)
"""


def off_axis_conic(R, K, h):
    """Point, normal and local radii of a conic at the height h.

    Parameters
    ----------
    R : number
        Vertex radius of curvature (signed as in `raytrace`).
    K : number
        Conic constant.
    h : number
        Height of the point above the axis (in the y-z plane).

    Returns
    -------
    OffAxisConic
    """

    c = 1/R
    z = c*h**2/(1 + np.sqrt(1 - (1 + K)*c**2*h**2))
    slope = c*h/np.sqrt(1 - (1 + K)*c**2*h**2)
    R_s = np.sign(R)*np.sqrt(R**2 - K*h**2)

    return OffAxisConic((h, z), np.arctan(slope), R_s, R_s**3/R**2)


def segment_sag(R, K, h, x, y):
    """Sag of an off-axis segment of a conic in its tangent frame.

    The tangent frame has its origin at the point (0, h) of the parent, its
    z axis along the surface normal there (tilted by `off_axis_conic()`'s
    normal angle about x) and its y axis in the plane of the offset.

    Parameters
    ----------
    R, K : numbers
        Vertex radius and conic constant of the parent.
    h : number
        Height of the segment centre on the parent.
    x, y : arrays
        Points of the tangent frame.

    Returns
    -------
    z : array
        Sag along the local normal (NaN where the local normal line misses
        the parent).
    """

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    g = off_axis_conic(R, K, h)
    a = g.normal_angle
    # local axes in the parent frame: x, y' = (cos a, sin a) in (y, z) and
    # z' = (-sin a, cos a) (rotation about x by -a in the raytrace convention):
    ey = np.array([0.0, np.cos(a), np.sin(a)])
    ez = np.array([0.0, -np.sin(a), np.cos(a)])
    origin = np.array([0.0, g.point[0], g.point[1]])
    big = 10*abs(R)
    P = (origin + x.ravel()[:, None]*np.array([1.0, 0, 0]) + y.ravel()[:, None]*ey
         - big*ez)
    d = np.tile(ez, (P.shape[0], 1))
    t = rt.Surface(R=R, conic=K).intersect(P, d)

    return (t - big).reshape(x.shape)


def segment_zernikes(R, K, h, a, n_terms=16, N=64):
    """Fringe Zernike coefficients of an off-axis segment in its tangent
    frame.

    Parameters
    ----------
    R, K : numbers
        Vertex radius and conic constant of the parent.
    h : number
        Height of the segment centre.
    a : number
        Radius of the segment (normalization radius of the Zernikes).
    n_terms : int, optional
        Number of Fringe Zernikes.
    N : int, optional
        Samples across the disk.

    Returns
    -------
    coeffs : array
        Coefficients in length units, coeffs[j - 1] belongs to Z_j: Z4 and
        Z9 the rotationally symmetric part (best fit sphere), Z5 the
        astigmatism and Z8 the coma of the cap.
    """

    u = np.linspace(-1, 1, N)
    U, V = np.meshgrid(u, u)
    ok = U**2 + V**2 <= 1
    z = segment_sag(R, K, h, a*U[ok], a*V[ok])
    A = np.column_stack([fringe_zernike(j, U[ok], V[ok]) for j in range(1, n_terms + 1)])

    return np.linalg.lstsq(A, z, rcond=None)[0]


def off_axis_parabola(f, offset, diameter, field_angle=0.0, wavelength=550e-6):
    """An off-axis paraboloid (aperture offset) focusing a collimated beam.

    Parameters
    ----------
    f : number
        Focal length of the parent paraboloid (> 0).
    offset : number
        Height of the centre of the aperture above the axis of the parent.
    diameter : number
        Diameter of the circular aperture (the stop, a plane in front of the
        mirror decentred by offset).
    field_angle : number or pair, optional
        Maximum field.
    wavelength : number, optional

    Returns
    -------
    raytrace.System
        Stop (plane, at f/2 in front of the vertex), mirror and the image
        plane through the parent focus, perpendicular to the parent axis.
    """

    d_stop = f/2
    s = [rt.Surface(thickness=d_stop, semi_diameter=diameter/2, decenter=(0.0, offset),
                    label="stop"),
         rt.Surface(R=-2*f, conic=-1.0, thickness=-f, mirror=True, label="OAP")]

    return rt.System(s, 0, wavelength, field_angle=field_angle)


# --- three-mirror anastigmats -------------------------------------------------

_TERMS = {"SI": 0, "SII": 1, "SIII": 2, "SIV": 3, "SV": 4}


def solve_conics(system, surfaces, terms=("SI", "SII", "SIII"), targets=None):
    """Conic constants for linear Seidel conditions.

    The Seidel sums are linear in the conic constants (each adds
    8 a4 (n' - n) y^4 ... with a4 = K c^3/8), so as many sums as conics can
    be set exactly.

    Parameters
    ----------
    system : raytrace.System
    surfaces : sequence of int
        Surfaces whose conic constants are solved.
    terms : sequence of str, optional
        Seidel sums ("SI" ... "SV") to set, as many as surfaces.
    targets : sequence, optional
        Their values (default 0).

    Returns
    -------
    raytrace.System
        Copy with the solved conic constants.
    """

    if len(terms) != len(surfaces):
        raise ValueError("Need as many Seidel terms as surfaces.")
    targets = np.zeros(len(terms)) if targets is None else np.asarray(targets, float)
    rows = [_TERMS[t] for t in terms]
    out = system.copy()
    for s in surfaces:
        out.surfaces[s].conic = 0.0
    S0 = out.seidel().total[rows]
    M = np.zeros((len(rows), len(surfaces)))
    for k, s in enumerate(surfaces):
        out.surfaces[s].conic = 1.0
        M[:, k] = out.seidel().total[rows] - S0
        out.surfaces[s].conic = 0.0
    K = np.linalg.solve(M, targets - S0)
    for s, k in zip(surfaces, K):
        out.surfaces[s].conic = float(k)

    return out


def cook_tma(efl=600.0, f_number=4.0, field_angle=np.radians(4.0), prescription=None,
             aperture_offset=0.0, wavelength=550e-6):
    """A Cook three-mirror anastigmat: the coaxial parent of a field-biased
    unobscured telescope.

    A concave primary, a convex secondary that is the aperture stop and a
    concave tertiary (no intermediate image). The radii follow the prescription scaled to the focal length,
    the tertiary is solved for a zero Petzval sum (c1 - c2 + c3 = 0, a flat
    field) and the conic constants for zero spherical aberration, coma and
    astigmatism (`solve_conics()`); the image plane is at the paraxial focus.

    Parameters
    ----------
    efl : number, optional
        Focal length (the prescription is scaled to it).
    f_number : number, optional
        Sets the entrance pupil diameter efl/f_number.
    field_angle : number or pair, optional
        Maximum field of the system (the parent is used over an off-axis
        part of it).
    prescription : (R1, d1, R2, d2), optional
        Radii and separations of primary and secondary, scaled to the
        focal length (default (-2/3, 0.2, -1/3, 4/9): primary and tertiary
        of equal radius, the image between primary and tertiary).
    aperture_offset : number, optional
        Decentre of the stop in units of its semi-diameter (a plane stop
        at the vertex of the secondary; 0: the secondary is the stop).
    wavelength : number, optional

    Returns
    -------
    raytrace.System
        Primary (0), secondary (1, the stop) and tertiary (2).
    """

    R1, d1, R2, d2 = (-2/3, 0.2, -1/3, 4/9) if prescription is None else prescription
    s = [rt.Surface(R=R1, thickness=-d1, mirror=True, label="M1"),
         rt.Surface(R=R2, thickness=d2, mirror=True, semi_diameter=1.0, label="M2"),
         rt.Surface(R=1/(1/R2 - 1/R1), thickness=-1.0, mirror=True, label="M3")]
    # scale with the focal length of the unscaled layout:
    k = efl/rt.System(s, 1, wavelength).paraxial().efl
    for surf in s:
        surf.R *= k
        surf.thickness *= k
    sys = rt.System(s, 1, wavelength, field_angle=field_angle,
                    entrance_pupil_radius=efl/(2*f_number))
    sys = solve_conics(sys.at_paraxial_focus(), [0, 1, 2])
    if aperture_offset:
        # the stop: a plane at the vertex of M2, decentred
        a = sys.surfaces[1].semi_diameter
        sys.surfaces[1].semi_diameter = None
        stop = rt.Surface(thickness=0.0, semi_diameter=a, decenter=(0.0, aperture_offset*a),
                          label="stop")
        S = sys.surfaces
        sys = rt.System([S[0], stop] + S[1:], 1, wavelength, field_angle=field_angle)
    else:
        sys.surfaces[1].label = "M2 (stop)"

    return sys


def freeform_mirrors(system, fields, surfaces=None, zernike=None, margin=1.0):
    """Mirrors as `raytrace.ZernikeSurface` with Zernikes centred on their
    footprints.

    Parameters
    ----------
    system : raytrace.System
    fields : (N, 2) array
        Field points of the footprints.
    surfaces : sequence of int, optional
        Mirrors to convert (default: all mirrors).
    zernike : dict, optional
        Initial coefficients {j: c_j} on every converted mirror (default
        none).
    margin : number, optional
        Normalization radius over the footprint radius.

    Returns
    -------
    raytrace.System
        Copy; the base sag (vertex radius, conic) is unchanged and the
        Zernikes are centred on the footprint centres with the footprint
        radii (times margin) as normalization radii.
    """

    surfaces = [i for i, s in enumerate(system.surfaces) if s.mirror] if surfaces is None \
        else list(surfaces)
    fp = footprints(system, fields)
    out = system.copy()
    for i in surfaces:
        s = out.surfaces[i]
        kw = {k: getattr(s, k) for k in ("conic", "aspheric", "semi_diameter", "mirror",
                                          "decenter", "tilt", "label")}
        out.surfaces[i] = rt.ZernikeSurface(s.R, s.thickness, zernike=dict(zernike or {}),
                                            norm_radius=margin*fp[i].radius,
                                            centre=fp[i].centre, **kw)

    return out


# --- footprints and clearance -------------------------------------------------

def pupil_rings(rings=(0.5, 1.0), spokes=8, centre=True):
    """Normalized pupil points on rings (full pupil).

    Parameters
    ----------
    rings : sequence, optional
        Radii of the rings.
    spokes : int, optional
        Points per unit radius on each ring (at least 4 per ring).
    centre : bool, optional
        Include the pupil centre first.

    Returns
    -------
    px, py : arrays
    """

    px, py = ([0.0], [0.0]) if centre else ([], [])
    for r in rings:
        n = max(4, int(round(spokes*r)))
        a = 2*np.pi*np.arange(n)/n
        px += list(r*np.sin(a))
        py += list(r*np.cos(a))

    return np.array(px), np.array(py)


def _trace_fields(system, fields, px, py, clip=False):
    """Trace the pupil points at each field; points (S + 2, F, P, 3) and
    directions, NaN for failed rays."""

    pts, dirs = [], []
    for f in fields:
        b = system.trace(system.rays(tuple(f), px, py), clip=clip)
        pts.append(b.points)
        dirs.append(b.directions)

    return np.stack(pts, axis=1), np.stack(dirs, axis=1)


def _local(system, i, P):
    o, R = system.frames()[0][i]
    return (P - o) @ R


def _hull(points):
    """Convex hull (counterclockwise) of 2D points (monotone chain)."""

    pts = sorted(set(map(tuple, np.round(points, 12))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


def _signed_distance(hull, q):
    """Signed distance of the points q to a convex polygon (negative inside)."""

    q = np.atleast_2d(q)
    a, b = hull, np.roll(hull, -1, axis=0)
    e = b - a
    L = np.hypot(e[:, 0], e[:, 1])
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)/L[:, None]  # outward for ccw
    rel = q[:, None, :] - a[None]
    side = np.einsum("qkj,kj->qk", rel, nrm)
    inside = np.all(side <= 0, axis=1)
    t = np.clip(np.einsum("qkj,kj->qk", rel, e)/L**2, 0, 1)
    closest = a[None] + t[..., None]*e[None]
    d = np.min(np.hypot(*(q[:, None, :] - closest).transpose(2, 0, 1)), axis=1)

    return np.where(inside, -d, d)


Footprint = namedtuple("Footprint", ["centre", "radius", "hull", "points"])
Footprint.__doc__ = """Footprint of the beams on a surface (local frame).

Attributes
----------
centre : (x, y) centre of the smallest box around the hits
radius : largest distance of a hit from the centre
hull : (K, 2) convex hull of the hits, counterclockwise
points : (N, 2) the hits
"""


def footprints(system, fields, rings=(0.5, 0.8, 1.0), spokes=16):
    """Footprints of the beams of all fields on every surface.

    Parameters
    ----------
    system : raytrace.System
    fields : (N, 2) array
        Field points (e.g. the corners and edges of the field).
    rings, spokes : optional
        Pupil sampling (`pupil_rings()`).

    Returns
    -------
    list of Footprint
        One per surface, in its local frame.
    """

    px, py = pupil_rings(rings, spokes)
    P = _trace_fields(system, fields, px, py)[0]
    out = []
    for i in range(len(system.surfaces)):
        q = _local(system, i, P[i + 1].reshape(-1, 3))[:, :2]
        q = q[np.all(np.isfinite(q), axis=1)]
        lo, hi = q.min(axis=0), q.max(axis=0)
        c = (lo + hi)/2
        out.append(Footprint((float(c[0]), float(c[1])),
                             float(np.max(np.hypot(*(q - c).T))), _hull(q), q))

    return out


Clearance = namedtuple("Clearance", ["obstacles", "beams", "distance", "blocked", "names"])
Clearance.__doc__ = """Clearance of the beams around the mirrors (and the image).

Attributes
----------
obstacles : indices of the obstacles (surfaces; len(surfaces) is the image
    plane)
beams : the beams by the surface they start from (-1: the incoming beam,
    i: from surface i to the next surface or the image)
distance : (obstacles, beams) array, smallest distance (length units) in
    the obstacle's local x-y plane from a crossing of the beam to the
    footprint of the obstacle grown by the rim width; negative if the beam
    hits the obstacle, inf if it never crosses it
blocked : (obstacles, beams) array, fraction of the rays of the beam that
    the obstacle blocks
names : labels of the obstacles
"""


def beam_clearance(system, fields, rim=0.0, image=True, rings=(0.5, 0.8, 1.0), spokes=16,
                   behind=1e4):
    """Does the light clear the mirrors? Distances of each beam from each
    mirror it passes.

    A beam is the bundle of rays between two consecutive surfaces (the
    incoming beam is extended to `behind` in front of its start). A beam
    is blocked by an obstacle if it crosses the obstacle's surface (its
    parent, in the local frame) inside the footprint of that obstacle grown
    by `rim` (a mechanical rim or margin). The beams that arrive at or
    leave an obstacle are not checked against it.

    Parameters
    ----------
    system : raytrace.System
    fields : (N, 2) array
        Field points (e.g. corners and edges of the field).
    rim : number, optional
        Margin added around every footprint (length units).
    image : bool, optional
        Include the image plane (its footprint, the detector) as an
        obstacle.
    rings, spokes : optional
        Pupil sampling.
    behind : number, optional
        Length by which the incoming beam is extended backwards.

    Returns
    -------
    Clearance
    """

    px, py = pupil_rings(rings, spokes)
    P = _trace_fields(system, fields, px, py)[0]
    fp = footprints(system, fields, rings, spokes)
    n_s = len(system.surfaces)
    obstacles = [i for i, s in enumerate(system.surfaces) if s.mirror]
    names = [system.surfaces[i].label or "surface {}".format(i) for i in obstacles]
    if image:
        obstacles.append(n_s)
        names.append("image")
    beams = list(range(-1, n_s))
    D = np.full((len(obstacles), len(beams)), np.inf)
    B = np.zeros_like(D)
    img_frame = system.frames()[1]
    for a, j in enumerate(obstacles):
        if j < n_s:
            o, R = system.frames()[0][j]
            surf = system.surfaces[j]
            hull = fp[j].hull
        else:
            o, R = img_frame
            surf = rt.Surface()
            q = ((P[-1].reshape(-1, 3) - o) @ R)[:, :2]
            hull = _hull(q[np.all(np.isfinite(q), axis=1)])
        for b, k in enumerate(beams):
            if k in (j - 1, j) or (j == n_s and k == n_s - 1):
                continue  # arrives at or leaves the obstacle
            A0 = P[k + 1].reshape(-1, 3)
            A1 = P[k + 2].reshape(-1, 3)
            ok = np.all(np.isfinite(A0), axis=1) & np.all(np.isfinite(A1), axis=1)
            A0, A1 = A0[ok], A1[ok]
            d = A1 - A0
            L = np.linalg.norm(d, axis=1)
            d = d/L[:, None]
            if k == -1:
                A0 = A0 - behind*d
                L = L + behind
            Pl, dl = (A0 - o) @ R, d @ R
            with np.errstate(invalid="ignore", divide="ignore"):
                if j < n_s:
                    t = surf.intersect(Pl, dl)
                else:
                    t = -Pl[:, 2]/dl[:, 2]
            cross = np.isfinite(t) & (t > 1e-9*L) & (t < L*(1 - 1e-9))
            if not np.any(cross):
                continue
            Q = Pl[cross, :2] + t[cross, None]*dl[cross, :2]
            dist = _signed_distance(hull, Q) - rim
            D[a, b] = float(np.min(dist))
            B[a, b] = np.sum(dist < 0)/len(t)

    return Clearance(obstacles, beams, D, B, names)


# --- field maps ----------------------------------------------------------------

def field_zernikes(system, fields, n_terms=9, N=32):
    """Fringe Zernike coefficients of the wavefront over a field grid.

    Parameters
    ----------
    system : raytrace.System
    fields : (F, 2) array
        Field points.
    n_terms : int, optional
    N : int, optional
        Pupil samples across (see `raytrace.System.opd()`).

    Returns
    -------
    (F, n_terms) array
        Coefficients in waves; Z5 + i Z6 is the astigmatism and Z7 + i Z8
        the coma (see `nodal`).
    """

    return np.array([system.zernike(tuple(f), n_terms=n_terms, N=N) for f in fields])


def field_rms(system, fields, N=32):
    """RMS wavefront error (waves, tilt removed) over a field grid.

    Parameters
    ----------
    system : raytrace.System
    fields : (F, 2) array
    N : int, optional

    Returns
    -------
    (F,) array
    """

    return np.array([system.rms_opd(tuple(f), N=N) for f in fields])


def plot_full_field(ax, fields, values, kind="astigmatism", scale=None, unit=1.0, color="C0",
                    lw=1.5):
    """Full-field display: astigmatism sticks, coma arrows or RMS circles
    over the field.

    Parameters
    ----------
    ax : matplotlib axis
    fields : (F, 2) array
        Field points (plotted divided by unit).
    values : array
        Complex astigmatism Z5 + i Z6 or coma Z7 + i Z8 per field point
        (kind "astigmatism" or "coma"), or a real value (kind "rms").
    kind : {"astigmatism", "coma", "rms"}, optional
        Astigmatism: a line along the axis of the astigmatism, at the
        azimuth arg(Z5 + i Z6)/2 (the direction of the larger wavefront
        curvature, Z5 + i Z6 positive real along x), length proportional
        to |Z5 + i Z6|. Coma: an arrow in the direction arg(Z7 + i Z8).
        RMS: circles with an area proportional to the value.
    scale : number, optional
        Length on the plot per unit of the value, for RMS the radius per
        square root of the value (default: the largest value fills about
        the grid spacing). Pass the returned scale to other plots to
        compare them.
    unit : number, optional
        Field unit (e.g. np.radians(1) to plot degrees).
    color, lw : optional
    Returns
    -------
    scale : number
    """

    F = np.asarray(fields)/unit
    v = np.asarray(values)
    spacing = np.min([np.min(np.abs(np.diff(np.unique(F[:, k])))) if len(np.unique(F[:, k])) > 1
                      else np.inf for k in range(2)])
    if not np.isfinite(spacing):
        spacing = 1.0
    if scale is None:
        top = np.nanmax(np.abs(v))
        if kind == "rms":
            scale = 0.45*spacing/np.sqrt(top) if top > 0 else 1.0
        else:
            scale = 0.9*spacing/top if top > 0 else 1.0
    for (x, y), val in zip(F, v):
        if kind == "astigmatism":
            L = scale*abs(val)
            a = np.angle(val)/2
            ax.plot([x - L/2*np.cos(a), x + L/2*np.cos(a)], [y - L/2*np.sin(a), y + L/2*np.sin(a)],
                    color=color, lw=lw, solid_capstyle="round")
        elif kind == "coma":
            L = scale*abs(val)
            a = np.angle(val)
            ax.annotate("", (x + L/2*np.cos(a), y + L/2*np.sin(a)),
                        (x - L/2*np.cos(a), y - L/2*np.sin(a)),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        elif kind == "rms":
            ax.add_patch(_circle((x, y), scale*np.sqrt(abs(val)), color))
        else:
            raise ValueError("kind must be 'astigmatism', 'coma' or 'rms'")
        ax.plot(x, y, ".", color="0.6", ms=2)
    ax.set_aspect("equal")

    return scale


def _circle(xy, r, color):
    from matplotlib.patches import Circle

    return Circle(xy, r, fc="none", ec=color, lw=1.2)


def draw_layout(system, fields, ax=None, n_rays=5, colors=None, lw=0.6, rim=0.0,
                mirror_lw=2.5, plane="yz", image_half_width=None):
    """Meridional section of an off-axis system with the mirrors drawn over
    their footprints only.

    Parameters
    ----------
    system : raytrace.System
    fields : sequence of pairs
        Fields to draw rays for.
    ax : matplotlib axis, optional
    n_rays : int, optional
        Rays per field across the pupil.
    colors : sequence, optional
        One color per field.
    lw : number, optional
        Ray line width.
    rim : number, optional
        Length added on both sides of each footprint.
    mirror_lw : number, optional
    plane : {"yz", "xz"}, optional
    image_half_width : number, optional
        Half width of the drawn image plane (default: its footprint).

    Returns
    -------
    ax : matplotlib axis
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    a = 1 if plane == "yz" else 0
    p = np.linspace(-1, 1, n_rays)
    px, py = (0*p, p) if a == 1 else (p, 0*p)
    allf = np.asarray(fields, float).reshape(-1, 2)
    fp = footprints(system, allf)
    for k, f in enumerate(allf):
        b = system.trace(system.rays(tuple(f), px, py), clip=False)
        color = colors[k] if colors is not None else "C{}".format(k)
        ax.plot(b.points[1:, :, 2], b.points[1:, :, a], color=color, lw=lw)
        # the incoming rays from a common start plane:
        z0 = np.nanmin(b.points[1:, :, 2]) - 0.15*np.ptp(b.points[1:, :, 2][np.isfinite(
            b.points[1:, :, 2])])
        s = (z0 - b.points[1, :, 2])/b.directions[0, :, 2]
        start = b.points[1] + s[:, None]*b.directions[0]
        ax.plot(np.stack([start[:, 2], b.points[1, :, 2]]),
                np.stack([start[:, a], b.points[1, :, a]]), color=color, lw=lw)
    frames, (o_img, R_img) = system.frames()
    for i, s in enumerate(system.surfaces):
        if isinstance(s, rt.CoordinateBreak):
            continue
        q = fp[i].points[:, a]
        t = np.linspace(q.min() - rim, q.max() + rim, 101)
        c = np.zeros((len(t), 2))
        c[:, 1 - a] = fp[i].centre[1 - a]
        loc = np.zeros((len(t), 3))
        loc[:, a] = t
        loc[:, 1 - a] = c[:, 1 - a]
        if isinstance(s, rt.ZernikeSurface):
            loc[:, 2] = s.sag_xy(loc[:, 0], loc[:, 1])
        else:
            loc[:, 2] = s.sag(loc[:, 0]**2 + loc[:, 1]**2)
        o, R = frames[i]
        g = loc @ R.T + o
        if s.mirror:
            ax.plot(g[:, 2], g[:, a], color="0.2", lw=mirror_lw, solid_capstyle="butt")
        elif s.semi_diameter is not None and i == system.stop_index:
            ax.plot(g[:, 2], g[:, a], color="0.5", lw=1.0, ls="--")
    hits = []
    for f in allf:
        bb = system.trace(system.rays(tuple(f), px, py), clip=False)
        hits.append(system.to_image_frame(bb.points[-1]))
    hits = np.concatenate(hits)
    if image_half_width is None:
        lo_, hi_ = np.nanmin(hits[:, a]), np.nanmax(hits[:, a])
        h, c0 = (hi_ - lo_)/2, (hi_ + lo_)/2
    else:
        h, c0 = image_half_width, 0.0
    t = np.linspace(c0 - 1.1*h, c0 + 1.1*h, 2)
    loc = np.zeros((2, 3))
    loc[:, a] = t
    g = loc @ R_img.T + o_img
    ax.plot(g[:, 2], g[:, a], color="0.35", lw=3)
    ax.set_aspect("equal")
    ax.set_xlabel("z")
    ax.set_ylabel(plane[0])

    return ax


# --- operands for lens_optimization ---------------------------------------------

def _clean(e, failed=1e3):
    return np.nan_to_num(np.asarray(e, float), nan=failed, posinf=failed, neginf=failed)


def field_wavefront(fields, rings=(0.5, 0.85, 1.0), spokes=8, tolerance=0.05,
                    remove_tilt=True):
    """Operand: OPD of rays over the full pupil at two-dimensional fields.

    Parameters
    ----------
    fields : (F, 2) array
        Field points (absolute, like `raytrace.System.rays()`).
    rings, spokes : optional
        Pupil sampling (`pupil_rings()`, full rings).
    tolerance : number, optional
        In waves.
    remove_tilt : bool, optional
        Remove the least squares tilt of the sampled OPD at each field (the
        image point is then the centroid, not the chief ray).

    Returns
    -------
    operand : callable
        operand(system) returns the OPDs / tolerance.
    """

    px, py = pupil_rings(rings, spokes, centre=True)
    A = np.column_stack([np.ones_like(px), px, py])

    def operand(system):
        out = []
        for f in fields:
            b = system.trace(system.rays(tuple(f), px, py), clip=False)
            chief = rt.RayBundle(b.points[:, :1], b.directions[:, :1], b.opl[:, :1],
                                 b.valid[:1], b.n)
            W = system._opd(b, chief)/system.wavelength
            if remove_tilt and np.all(np.isfinite(W)):
                W = W - A @ np.linalg.lstsq(A, W, rcond=None)[0]
            out.append(_clean(W/tolerance))

        return np.concatenate(out)

    return operand


def focal_length(efl, centre=(0.0, 0.0), delta=1e-3, tolerance=0.1, axes=(0, 1)):
    """Operand: real focal length from the chief rays around a field point,
    efl = d(image position)/d(tan field), in x and y.

    Parameters
    ----------
    efl : number
        Target (the magnitude of the image scale; the orientation of the
        image is free).
    centre : (x, y), optional
        Field point (radians).
    delta : number, optional
        Field step (radians).
    tolerance : number, optional
        Length units.
    axes : sequence, optional
        0 for x, 1 for y.

    Returns
    -------
    operand : callable
    """

    def operand(system):
        out = []
        for k in axes:
            e = np.zeros(2)
            e[k] = delta
            I = []
            for sgn in (-1, 1):
                f = np.asarray(centre, float) + sgn*e
                b = system.trace(system.rays(tuple(f), 0.0, 0.0), clip=False)
                I.append(system.to_image_frame(b.points[-1, 0]))
            t = np.tan(centre[k] + delta) - np.tan(centre[k] - delta)
            out.append((abs(I[1][k] - I[0][k])/t - efl)/tolerance)

        return _clean(out)

    return operand


def clearance_penalty(fields, minimum=5.0, tolerance=1.0, rim=0.0, image=True,
                      rings=(1.0,), spokes=12):
    """Operand: penalties for beams that pass a mirror closer than minimum.

    Parameters
    ----------
    fields : (F, 2) array
        Field points (corners and edges of the field).
    minimum : number, optional
        Required clearance (length units).
    tolerance : number, optional
        Length units.
    rim, image, rings, spokes : optional
        See `beam_clearance()`.

    Returns
    -------
    operand : callable
        operand(system) returns max(0, minimum - distance)/tolerance for
        every pair of obstacle and beam.
    """

    def operand(system):
        try:
            c = beam_clearance(system, fields, rim=rim, image=image, rings=rings, spokes=spokes)
        except (ValueError, IndexError):
            return np.full(1, 1e3)
        d = c.distance.ravel()

        return _clean(np.maximum(0.0, minimum - np.where(np.isfinite(d), d, 1e9))/tolerance)

    return operand
