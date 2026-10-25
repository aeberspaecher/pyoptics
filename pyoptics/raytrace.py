#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Sequential ray tracing through optical systems.

A `System` is a sequence of `Surface` objects (spheres, conics and even
aspheres, refracting or reflecting), optional `IdealLens` surfaces and
`CoordinateBreak` elements. Surfaces can be decentred and tilted. Rays
are traced exactly (exact intersections, vector law of refraction and
reflection); paraxial marginal and chief rays, the pupils and the Seidel
sums come from a paraxial y-nu trace, whose system matrix agrees with
`pyoptics.abcd`. The main result is the optical path difference (OPD) over
the pupil, which can be fitted with Fringe Zernikes and handed on to
`pyoptics.imaging` for PSFs, MTFs and Strehl ratios.

Sign conventions
----------------

- The optical axis is z, light enters the system travelling in +z. The
  vertex of the first surface lies at z = 0. The field lies in the
  meridional y-z plane (field points at +y).
- Radii R > 0 if the centre of curvature lies in +z of the vertex (as in
  `abcd`). A biconvex lens has R1 > 0 and R2 < 0; a concave mirror facing
  light that travels in +z has R < 0.
- The sag of a surface is ::

      z(r) = c r**2/(1 + sqrt(1 - (1 + K) c**2 r**2)) + A4 r**4 + A6 r**6 + ...

  with c = 1/R, the conic constant K (0 sphere, -1 paraboloid, < -1
  hyperboloid, (-1, 0) prolate ellipsoid, > 0 oblate ellipsoid) and the
  aspheric coefficients aspheric = (A4, A6, ...).
- thickness is the signed distance in z from a vertex to the next vertex
  (for the last surface: to the image plane). Behind a mirror, light travels
  in -z and thicknesses are negative. Refractive indices stay positive.
- Paraxial slopes u = dy/dz. Internally, the paraxial trace uses signed
  indices (-n while light travels in -z), so that mirrors are refractions
  with n' = -n.
- Field: for an object at infinity, the field is the angle in radians of the
  incoming rays with the z axis (rays point upwards, +y, for field > 0).
  For a finite object, it is the object height y.
  Fields can also be given as pairs (x, y): angles whose tangents are the
  direction slopes dx/dz and dy/dz, or object points (x, y).
- Pupil coordinates (P_x, P_y) are normalized coordinates in the paraxial
  entrance pupil (unit disk). With ray aiming (default), a ray with pupil
  coordinates (P_x, P_y) hits the stop at (P_x, P_y) times its semi
  diameter; the chief ray passes the centre of the stop.

Optical path difference
-----------------------
The OPD of a ray is its optical path to the reference sphere minus that of
the chief ray, OPD = OPL_ray - OPL_chief. The reference sphere is centred on
the point where the real chief ray hits the image plane and passes through
the point where the chief ray crosses the paraxial exit pupil plane. With the
pyoptics convention exp(i(k.r - omega*t)), the field on the reference
sphere is proportional to exp(+i k_0 OPL), so the pupil phase is ::

    phase = +2*pi*OPD/wavelength.

E.g. an image plane dz behind a perfect focus gives
OPD = n' dz (cos(alpha) - 1) < 0 at the rim, the same phase that the angular
spectrum propagator applies over dz. A positive lens with undercorrected
spherical aberration (marginal rays focus short) has OPD < 0 at the rim
(W040 < 0). The
transverse ray aberration is eps = (R/n') dW/dY (R: radius of the reference
sphere, Y: height in the exit pupil).

Tilts and decentres
-------------------
Every surface has a local frame in which it is the centred surface
described above. A surface's `decenter=(dx, dy)` and `tilt=(a_x, a_y, a_z)`
move it relative to the current axis frame: first the decentre, then the
rotations about the local x, y and z axes in this order (the local-to-axis
rotation is R_x(a_x) R_y(a_y) R_z(a_z), counterclockwise about the
respective axis, so a_x > 0 turns +z towards -y). Like Zemax's "decenter and return",
this only moves the surface itself; the axis for the following surfaces is
unchanged. A `CoordinateBreak` instead changes the axis frame for all
following surfaces (fold mirrors, tilted sub-assemblies). The image plane
can be decentred and tilted with `image_decenter` and `image_tilt` of
`System`.

First order properties (`paraxial()`, `seidel()`, the pupils used for ray
aiming and for the reference sphere) are those of the nominal system with
all tilts and decentres set to zero; a fold mirror is then an unfolded plane
mirror. Real rays always see the actual geometry. Spot diagrams, ray fans,
OPDs and PSFs are evaluated in the local frame of the image plane.

Units: all lengths (radii, thicknesses, wavelength) share one unit, set by
`units` of `System` ("mm" by default). It is only used to convert the
wavelength to nm for the dispersion of materials.

Limitations: monochromatic traces (build systems at several wavelengths for
colour), uniform pupil amplitude in the PSF, surfaces of revolution only (no
toroids or freeforms), paraxial data only for the nominal centred system.
"""

import copy
from collections import namedtuple

import numpy as np
from numpy import pi

from . import abcd
from . import imaging
from .bases import FringeZernikes
from .materials import abbe_dispersion


NM_PER_UNIT = {"nm": 1.0, "um": 1e3, "mm": 1e6, "cm": 1e7, "m": 1e9}


def _norm(v):
    return np.sqrt(np.sum(v**2, axis=-1))


def _dot(a, b):
    return np.sum(a*b, axis=-1)


def rotation(a_x=0.0, a_y=0.0, a_z=0.0):
    """Rotation matrix R_x(a_x) @ R_y(a_y) @ R_z(a_z) (angles in radians,
    counterclockwise about the respective axis)."""

    cx, sx = np.cos(a_x), np.sin(a_x)
    cy, sy = np.cos(a_y), np.sin(a_y)
    cz, sz = np.cos(a_z), np.sin(a_z)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return R_x @ R_y @ R_z


def _frame(origin, R, decenter=(0.0, 0.0), tilt=(0.0, 0.0, 0.0)):
    """Frame (origin, R) moved by a decentre in its x-y plane and then tilted."""

    origin = origin + R @ np.array([decenter[0], decenter[1], 0.0])

    return origin, R @ rotation(*tilt)


def _field_xy(field):
    """(x, y) components of a scalar (y) or pair field."""

    f = np.atleast_1d(np.asarray(field, dtype=float))
    if f.size == 1:
        return 0.0, float(f[0])

    return float(f[0]), float(f[1])


# surfaces:
class Surface:
    """Rotationally symmetric refracting or reflecting surface.

    Parameters
    ----------
    R : number, optional
        Vertex radius of curvature (R > 0: centre of curvature in +z).
        np.inf (default) for a plane.
    thickness : number, optional
        Signed axial distance to the next vertex (or to the image plane).
    n_after : number, optional
        Refractive index behind the surface. If neither n_after nor material
        are given, the medium does not change (dummy surfaces, stops,
        mirrors).
    material : tuple or callable, optional
        (n_d, V_d) for `materials.abbe_dispersion()`, or a function
        n(wavelength_nm). Overrides n_after.
    conic : number, optional
        Conic constant K.
    aspheric : sequence, optional
        Coefficients (A4, A6, A8, ...) of r**4, r**6, r**8, ...
    semi_diameter : number, optional
        Clear aperture radius; rays outside are vignetted. None: unlimited.
        On the stop surface it defines the aperture of the system.
    mirror : bool, optional
        Reflecting surface.
    decenter : (dx, dy), optional
        Decentre of the surface in the axis frame.
    tilt : (a_x, a_y, a_z), optional
        Tilt angles (radians) of the surface after the decentre, see the
        module docstring. Neither changes the axis of the following
        surfaces; the thickness is measured along the (untilted) axis from
        the undisplaced vertex.
    label : str, optional
    """

    def __init__(self, R=np.inf, thickness=0.0, n_after=None, material=None, conic=0.0,
                 aspheric=(), semi_diameter=None, mirror=False, decenter=(0.0, 0.0),
                 tilt=(0.0, 0.0, 0.0), label=None):
        if mirror and (n_after is not None or material is not None):
            raise ValueError("A mirror does not change the medium.")
        self.R = R
        self.thickness = thickness
        self.n_after = n_after
        self.material = material
        self.conic = conic
        self.aspheric = tuple(aspheric)
        self.semi_diameter = semi_diameter
        self.mirror = mirror
        self.decenter = tuple(decenter)
        self.tilt = tuple(tilt) + (0.0,)*(3 - len(tilt))
        self.label = label

    @property
    def displaced(self):
        return any(self.decenter) or any(self.tilt)

    def __repr__(self):
        kind = "Mirror" if self.mirror else "Surface"
        return ("{}(R={}, thickness={}, n_after={}, material={}, conic={}, aspheric={}, "
                "semi_diameter={})".format(kind, self.R, self.thickness, self.n_after,
                                           self.material, self.conic, self.aspheric,
                                           self.semi_diameter))

    @property
    def c(self):
        """Vertex curvature 1/R."""

        return 0.0 if np.isinf(self.R) else 1.0/self.R

    def index(self, wavelength_nm, n_before):
        """Refractive index behind the surface."""

        if self.material is not None:
            if callable(self.material):
                return float(self.material(wavelength_nm))
            n_d, V_d = self.material
            return float(abbe_dispersion(wavelength_nm, n_d, V_d))
        if self.n_after is not None:
            return float(self.n_after)

        return n_before

    def sag(self, r2):
        """Sag z at the squared radius r2 = x**2 + y**2 (NaN beyond the conic)."""

        c, K = self.c, self.conic
        with np.errstate(invalid="ignore"):
            z = c*r2/(1 + np.sqrt(1 - (1 + K)*c**2*r2))
        for j, A in enumerate(self.aspheric):
            z = z + A*r2**(j + 2)

        return z

    def _slope_over_r(self, r2):
        """(dz/dr)/r, i.e. the normal is (-x*s, -y*s, 1) up to normalization."""

        c, K = self.c, self.conic
        with np.errstate(invalid="ignore"):
            s = c/np.sqrt(1 - (1 + K)*c**2*r2)
        for j, A in enumerate(self.aspheric):
            s = s + 2*(j + 2)*A*r2**(j + 1)

        return s

    def intersect(self, P, d):
        """Distance t along the unit directions d from the points P (both
        (N, 3), P relative to the vertex) to the surface; NaN for misses."""

        c, K = self.c, self.conic
        x, y, z = P.T
        dx, dy, dz = d.T
        a = c*(dx**2 + dy**2 + (1 + K)*dz**2)
        b = 2*(c*(x*dx + y*dy + (1 + K)*z*dz) - dz)
        C = c*(x**2 + y**2 + (1 + K)*z**2) - 2*z
        with np.errstate(invalid="ignore", divide="ignore"):
            D = b**2 - 4*a*C
            # root on the branch through the vertex, stable for c -> 0:
            t = -2*C/(b + np.where(b >= 0, 1.0, -1.0)*np.sqrt(D))
        if self.aspheric:
            t_plane = -z/dz
            t = np.where(np.isfinite(t), t, t_plane)
            for _ in range(50):
                Q = P + t[:, None]*d
                r2 = Q[:, 0]**2 + Q[:, 1]**2
                G = Q[:, 2] - self.sag(r2)
                dG = dz - self._slope_over_r(r2)*(Q[:, 0]*dx + Q[:, 1]*dy)
                with np.errstate(invalid="ignore", divide="ignore"):
                    step = G/dG
                t = t - step
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-15*(1 + np.abs(t)))):
                    break
        Q = P + t[:, None]*d
        # the solution must lie on the real part of the conic:
        r2 = Q[:, 0]**2 + Q[:, 1]**2
        t = np.where(np.isfinite(self.sag(r2)), t, np.nan)

        return t

    def normal(self, Q):
        """Unit normals (pointing into +z) at the points Q relative to the vertex."""

        s = self._slope_over_r(Q[:, 0]**2 + Q[:, 1]**2)
        N = np.stack([-Q[:, 0]*s, -Q[:, 1]*s, np.ones(len(Q))], axis=-1)

        return N/_norm(N)[:, None]

    def deflect(self, Q, d, n1, n2):
        """Refract (or reflect) the directions d at the points Q.

        Returns
        -------
        d2 : array
            New unit directions, NaN under total internal reflection.
        dopl : array
            Extra optical path (zero for real surfaces).
        """

        N = self.normal(Q)
        cos_i = _dot(d, N)
        if self.mirror:
            return d - 2*cos_i[:, None]*N, np.zeros(len(d))
        # orient N along the ray:
        N = N*np.sign(cos_i)[:, None]
        cos_i = np.abs(cos_i)
        mu = n1/n2
        sin2_t = mu**2*(1 - cos_i**2)
        with np.errstate(invalid="ignore"):
            cos_t = np.sqrt(1 - sin2_t)
        d2 = mu*d + (cos_t - mu*cos_i)[:, None]*N

        return d2, np.zeros(len(d))

    def paraxial_power(self, n1, n2):
        """Power for the signed indices n1, n2: n2*u2 = n1*u1 - y*power."""

        return self.c*(n2 - n1)

    def abcd_matrix(self, n1, n2, sigma):
        """Unfolded ABCD matrix for light travelling in sigma*z (sigma = +-1)."""

        if self.mirror:
            return abcd.mirror(-sigma*self.R, n=n1)

        return abcd.refraction(sigma*self.R, n1, n2)


class IdealLens(Surface):
    """Ideal (paraxial) thin lens of focal length f in a plane.

    Every ray is bent towards the point where the parallel ray through the
    centre of the lens meets the focal plane, and the optical path is
    corrected such that collimated light is focused without aberrations at
    any field angle (like the paraxial surface of commercial codes). The
    medium does not change; f is the focal length in that medium.
    """

    def __init__(self, f, thickness=0.0, semi_diameter=None, decenter=(0.0, 0.0),
                 tilt=(0.0, 0.0, 0.0), label=None):
        super().__init__(np.inf, thickness, semi_diameter=semi_diameter, decenter=decenter,
                         tilt=tilt, label=label)
        self.f = f

    def __repr__(self):
        return "IdealLens(f={}, thickness={}, semi_diameter={})".format(
            self.f, self.thickness, self.semi_diameter)

    def deflect(self, Q, d, n1, n2):
        f = self.f
        Qf = f*d/np.abs(d[:, 2:3])  # focal point of the bundle with direction d
        v = Qf - Q
        dist = _norm(v)
        d2 = np.sign(f)*v/dist[:, None]
        dopl = n1*(np.sign(f)*(np.abs(f)/np.abs(d[:, 2]) - dist) - _dot(Q, d))

        return d2, dopl

    def paraxial_power(self, n1, n2):
        return abs(n1)/self.f

    def abcd_matrix(self, n1, n2, sigma):
        return abcd.thin_lens(self.f/n1)


class CoordinateBreak(Surface):
    """Change of the axis frame for all following surfaces.

    The new frame is the current one moved by decenter = (dx, dy) and then
    rotated by tilt = (a_x, a_y, a_z) (see the module docstring); thickness
    is measured along the new z axis. A coordinate break does not interact
    with the rays. For a fold mirror tilted by a about x, put
    CoordinateBreak(tilt=(a, 0, 0)), the plane mirror, CoordinateBreak(tilt=(a,
    0, 0), thickness=-d): behind the mirror, light travels in -z of the new
    frame. For a = pi/4, the beam is folded from +z to +y.
    """

    def __init__(self, decenter=(0.0, 0.0), tilt=(0.0, 0.0, 0.0), thickness=0.0, label=None):
        super().__init__(np.inf, thickness, decenter=decenter, tilt=tilt, label=label)

    def __repr__(self):
        return "CoordinateBreak(decenter={}, tilt={}, thickness={})".format(
            self.decenter, self.tilt, self.thickness)

    def paraxial_power(self, n1, n2):
        return 0.0

    def abcd_matrix(self, n1, n2, sigma):
        return np.eye(2)


# results:
Rays = namedtuple("Rays", ["points", "directions", "opl"])
Rays.__doc__ = """Rays: start points and unit directions, shape (N, 3), and
the optical path already accumulated, shape (N,)."""

RayBundle = namedtuple("RayBundle", ["points", "directions", "opl", "valid", "n"])
RayBundle.__doc__ = """Rays traced through a System.

points : (S + 2, N, 3) array
    Start points (plane 0), intersections with the S surfaces and with the
    image plane (plane S + 1). NaN after a ray was lost.
directions : (S + 2, N, 3) array
    Unit directions behind each plane.
opl : (S + 2, N) array
    Accumulated optical path length.
valid : (N,) bool array
    Rays that reached the image plane (not vignetted, no misses, no TIR).
n : (S + 1,) array
    Refractive indices in front of the first surface and behind each surface.
"""

Paraxial = namedtuple("Paraxial", ["efl", "bfl", "ffl", "cardinal_points", "ep_z", "ep_radius",
                                   "xp_z", "xp_radius", "image_distance", "image_height",
                                   "magnification", "f_number", "na", "lagrange", "marginal",
                                   "chief", "matrix"])
Paraxial.__doc__ = """Paraxial (first order) properties of a System.

Attributes
----------
efl, bfl, ffl : effective, back and front focal length (bfl from the last
    vertex, ffl in front of the first vertex)
cardinal_points : abcd.CardinalPoints along the unfolded axis, z = 0 at the
    first vertex
ep_z, ep_radius : z position and radius of the entrance pupil
xp_z, xp_radius : z position (global) and radius of the exit pupil
image_distance : signed z distance of the paraxial image from the last vertex
image_height : paraxial chief ray height in the paraxial image plane
magnification : lateral magnification (0 for an object at infinity)
f_number : efl/(2*ep_radius) (working F-number 1/(2*na) for finite objects)
na : image space numerical aperture ``n'*|u'|`` of the marginal ray
lagrange : Lagrange invariant H = n*(u_chief*y_marginal - u_marginal*y_chief)
marginal, chief : (S, 2) arrays of the ray height y at each vertex and the
    slope u = dy/dz behind each surface
matrix : ABCD matrix (reduced angles n*theta, unfolded) from the first to
    the last vertex
"""

OPDMap = namedtuple("OPDMap", ["px", "py", "W"])
OPDMap.__doc__ = """OPD over the pupil: W[i, j] at (P_x, P_y) = (px[j], py[i]),
NaN outside the pupil and for vignetted rays."""

Pupil = namedtuple("Pupil", ["NA", "k", "phase", "mask"])
Pupil.__doc__ = """Exit pupil for pyoptics.imaging: image space NA, vacuum
wavenumber k and callables phase(P_x, P_y) (radians) and mask(P_x, P_y) of
the normalized imaging pupil coordinates, e.g.
imaging.get_psf(x, y, pupil.NA, pupil.k, pupil.mask, pupil.phase)."""

Seidel = namedtuple("Seidel", ["S", "total", "W040", "W131", "W222", "W220", "W311"])
Seidel.__doc__ = """Seidel sums.

Attributes
----------
S : (S_surfaces, 5) array of the contributions S_I...S_V of each surface
total : the five sums S_I (spherical aberration), S_II (coma), S_III
    (astigmatism), S_IV (Petzval curvature), S_V (distortion)
W040, W131, W222, W220, W311 : wavefront coefficients in waves at full
    aperture and the given field, in the OPD convention of this module
    (Welford's sums S_I...S_V are positive for undercorrected systems, which
    have W040 < 0 here)::

        W = W040 rho**4 + W131 H rho**3 cos(theta) + W222 H**2 rho**2 cos(theta)**2
            + W220 H**2 rho**2 + W311 H**3 rho cos(theta),
        W040 = -S_I/8, W131 = -S_II/2, W222 = -S_III/2, W220 = -(S_III + S_IV)/4,
        W311 = -S_V/2

    (divided by the wavelength), theta measured from the y axis.
"""

SpotDiagram = namedtuple("SpotDiagram", ["x", "y", "rms", "centroid"])
SpotDiagram.__doc__ = """Image plane intersections x, y relative to the chief
ray, RMS radius about the centroid and the centroid (x, y)."""

RayFan = namedtuple("RayFan", ["p", "ey", "ex"])
RayFan.__doc__ = """Transverse ray aberrations relative to the chief ray:
tangential fan ey(P_y = p, P_x = 0) and sagittal fan ex(P_x = p, P_y = 0)."""


class System:
    """Sequential optical system.

    Parameters
    ----------
    surfaces : sequence of Surface
        Surfaces in the order the light meets them. The thickness of the
        last surface is the distance to the image plane.
    stop_index : int
        Index of the aperture stop surface. Its semi_diameter (or
        entrance_pupil_radius) sets the aperture.
    wavelength : number
        Vacuum wavelength in the length unit of the system.
    object_distance : number, optional
        Distance of the object in front of the first vertex, np.inf
        (default) for an object at infinity.
    field_angle : number or pair, optional
        Maximum field angle in radians (object at infinity); used as the
        default field of the methods.
    object_height : number or pair, optional
        Maximum object height (finite object).
    n_object : number, optional
        Refractive index of the object space.
    units : str, optional
        Length unit, one of NM_PER_UNIT ("mm" default).
    entrance_pupil_radius : number, optional
        If given, the semi diameter of the stop is set such that the
        paraxial entrance pupil has this radius.
    ray_aiming : bool, optional
        Aim the rays at the real stop (default) instead of the paraxial
        entrance pupil.
    image_decenter, image_tilt : optional
        Decentre (dx, dy) and tilt (a_x, a_y, a_z) of the image plane
        relative to the final axis frame.

    The surfaces are copied, later changes to the Surface objects passed in
    do not affect the system.
    """

    def __init__(self, surfaces, stop_index, wavelength, object_distance=np.inf, field_angle=0.0,
                 object_height=0.0, n_object=1.0, units="mm", entrance_pupil_radius=None,
                 ray_aiming=True, image_decenter=(0.0, 0.0), image_tilt=(0.0, 0.0, 0.0)):
        self.surfaces = [copy.copy(s) for s in surfaces]
        self.stop_index = stop_index % len(self.surfaces)
        self.wavelength = wavelength
        self.object_distance = object_distance
        self.max_field = field_angle if np.isinf(object_distance) else object_height
        self.n_object = n_object
        self.units = units
        self.ray_aiming = ray_aiming
        self.image_decenter = tuple(image_decenter)
        self.image_tilt = tuple(image_tilt) + (0.0,)*(3 - len(image_tilt))
        if entrance_pupil_radius is not None:
            A = self._ep_matrix()[0]
            self.surfaces[self.stop_index].semi_diameter = abs(entrance_pupil_radius*A)
        if self.surfaces[self.stop_index].semi_diameter is None:
            raise ValueError("The stop needs a semi_diameter (or give entrance_pupil_radius).")

    # bookkeeping:
    def __repr__(self):
        lines = ["System(wavelength={}, stop_index={}, object_distance={}, max_field={})"
                 .format(self.wavelength, self.stop_index, self.object_distance, self.max_field)]
        for i, (s, n) in enumerate(zip(self.surfaces, self.n[1:])):
            lines.append("  {:2d}{} {!r} n'={:.6f}".format(i, "*" if i == self.stop_index else " ",
                                                           s, n))

        return "\n".join(lines)

    @property
    def wavelength_nm(self):
        return self.wavelength*NM_PER_UNIT[self.units]

    @property
    def n(self):
        """Indices in object space and behind each surface, shape (S + 1,)."""

        n = [self.n_object]
        for s in self.surfaces:
            n.append(s.index(self.wavelength_nm, n[-1]))

        return np.array(n)

    @property
    def sigma(self):
        """Direction of travel (+1: +z, -1: -z) in object space and behind
        each surface."""

        sigma = [1]
        for s in self.surfaces:
            sigma.append(-sigma[-1] if s.mirror else sigma[-1])

        return np.array(sigma)

    @property
    def vertex_z(self):
        """z positions of the vertices."""

        return np.concatenate([[0.0], np.cumsum([s.thickness for s in self.surfaces[:-1]])])

    @property
    def image_z(self):
        """z position of the image plane."""

        return self.vertex_z[-1] + self.surfaces[-1].thickness

    @property
    def infinite_object(self):
        return np.isinf(self.object_distance)

    def frames(self):
        """Local frames (origin, R) of the surfaces and of the image plane.

        A point q in the local frame of surface i lies at origin + R @ q in
        global coordinates.

        Returns
        -------
        surface_frames : list of (origin, R)
        image_frame : (origin, R)
        """

        o, R = np.zeros(3), np.eye(3)
        frames = []
        for s in self.surfaces:
            if isinstance(s, CoordinateBreak):
                o, R = _frame(o, R, s.decenter, s.tilt)
                frames.append((o, R))
            else:
                frames.append(_frame(o, R, s.decenter, s.tilt))
            o = o + R @ np.array([0.0, 0.0, s.thickness])

        return frames, _frame(o, R, self.image_decenter, self.image_tilt)

    @property
    def tilted(self):
        """True if any surface or the image plane is decentred or tilted."""

        return (any(s.displaced for s in self.surfaces) or any(self.image_decenter)
                or any(self.image_tilt))

    def to_image_frame(self, points, directions=None):
        """Points (and directions) in the local frame of the image plane."""

        o, R = self.frames()[1]
        local = (np.asarray(points) - o) @ R
        if directions is None:
            return local

        return local, np.asarray(directions) @ R

    def copy(self):
        return copy.deepcopy(self)

    def with_wavelength(self, wavelength):
        """Copy of the system at another wavelength (same geometry)."""

        other = self.copy()
        other.wavelength = wavelength

        return other

    def with_image_distance(self, distance):
        """Copy of the system with the image plane at distance (signed, in
        z) behind the last vertex."""

        other = self.copy()
        other.surfaces[-1].thickness = distance

        return other

    def with_axial_shift(self, first, last, dz):
        """Copy with the surfaces first...last (inclusive, e.g. the two
        surfaces of a lens) moved by dz along the axis; all other surfaces,
        the object and the image plane stay in place.

        The gap in front of `first` (or the object distance for first = 0)
        grows by dz and the gap behind `last` shrinks by dz.
        """

        other = self.copy()
        if first == 0:
            # with an object at infinity, nothing in front needs to stay fixed
            if not self.infinite_object:
                other.object_distance += dz
        else:
            other.surfaces[first - 1].thickness += dz
        other.surfaces[last].thickness -= dz

        return other

    def at_paraxial_focus(self):
        """Copy of the system with the image plane in the paraxial image."""

        return self.with_image_distance(self.paraxial().image_distance)

    # paraxial optics:
    def _paraxial_trace(self, y, u, first=0, last=None):
        """Trace a paraxial ray (y, u) given in the vertex plane of surface
        `first` (in front of it) through surfaces first...last - 1.

        Returns heights at the vertices and slopes behind the surfaces.
        """

        last = len(self.surfaces) if last is None else last
        n_s = self.n*self.sigma
        ys, us = [], []
        for i in range(first, last):
            s = self.surfaces[i]
            u = (n_s[i]*u - y*s.paraxial_power(n_s[i], n_s[i + 1]))/n_s[i + 1]
            ys.append(y)
            us.append(u)
            y = y + u*s.thickness

        return np.array(ys), np.array(us)

    def abcd_matrix(self):
        """ABCD matrix from the first to the last vertex, assembled from
        `pyoptics.abcd` elements (reduced angles, mirrors unfolded)."""

        n, sigma = self.n, self.sigma
        elements = []
        for i, s in enumerate(self.surfaces):
            if i > 0:
                elements.append(abcd.free_space(abs(self.surfaces[i - 1].thickness), n[i]))
            elements.append(s.abcd_matrix(n[i], n[i + 1], sigma[i]))

        return abcd.compose(*elements)

    def to_abcd(self):
        """Equivalent `abcd.OpticalSystem` from the first vertex to the
        image plane (unfolded)."""

        n = self.n
        system = abcd.OpticalSystem(n=self.n_object)
        for i, s in enumerate(self.surfaces):
            system.element(s.abcd_matrix(n[i], n[i + 1], self.sigma[i]), n=n[i + 1], label=s.label)
            system.free_space(abs(s.thickness))

        return system

    def _matrix(self, first=0, last=None):
        """Matrix (reduced, unfolded) of the surfaces first...last - 1 from the
        vertex plane of `first` to the vertex plane of last - 1, from two
        paraxial traces."""

        n_s = self.n*self.sigma
        cols = []
        for y, nu in ((1.0, 0.0), (0.0, 1.0)):
            ys, us = self._paraxial_trace(y, nu/n_s[first], first, last)
            last_ = len(self.surfaces) if last is None else last
            cols.append([ys[-1], n_s[last_]*us[-1]])

        return np.array(cols).T

    def _ep_matrix(self):
        """(A, B) of the paraxial map (y, u) at z = 0 -> stop height, with
        true slopes u in object space."""

        ys1, _ = self._paraxial_trace(1.0, 0.0, 0, self.stop_index + 1)
        ys2, _ = self._paraxial_trace(0.0, 1.0, 0, self.stop_index + 1)

        return ys1[-1], ys2[-1]

    def _pupils(self):
        a_s = self.surfaces[self.stop_index].semi_diameter
        A, B = self._ep_matrix()
        if A == 0:  # object space telecentric
            ep_z, ep_r = np.inf, np.inf
        else:
            ep_z = B/A
            ep_r = abs(a_s/A)
        # exit pupil: image of the stop through the surfaces behind it
        z_last = self.vertex_z[-1]
        ys, us = self._paraxial_trace(0.0, 1.0, self.stop_index)
        y_L, u_L = ys[-1], us[-1]
        if u_L == 0:
            return ep_z, ep_r, np.inf, np.inf
        xp_z = z_last - y_L/u_L
        ys, us = self._paraxial_trace(a_s, 0.0, self.stop_index)
        xp_r = abs(ys[-1] + us[-1]*(xp_z - z_last))

        return ep_z, ep_r, xp_z, xp_r

    def _object_slopes(self):
        """Finite object: object space slopes of the paraxial marginal ray
        (to the rim of the stop) and of the chief ray per unit object height."""

        a_s = self.surfaces[self.stop_index].semi_diameter
        A, B = self._ep_matrix()
        D = A*self.object_distance + B

        return a_s/D, -A/D

    def _paraxial_rays(self, field=None):
        """Paraxial marginal and chief rays: start (y, u) at z = 0. Paraxial
        data use the y component of the field."""

        field = _field_xy(self.max_field if field is None else field)[1]
        if self.infinite_object:
            ep_z, ep_r = self._pupils()[:2]
            if not np.isfinite(ep_z):
                raise NotImplementedError("Object at infinity and object space telecentric.")
            marginal = (ep_r, 0.0)
            u = np.tan(field)
            chief = (-u*ep_z, u)
        else:
            s = self.object_distance
            u_m, u_c = self._object_slopes()
            marginal = (u_m*s, u_m)
            u = u_c*field
            chief = (field + u*s, u)

        return marginal, chief

    def paraxial(self, field=None):
        """First order properties, see `Paraxial`.

        field : number, optional
            Field for the chief ray (default: the maximum field).
        """

        n = self.n
        n_s = n*self.sigma
        M = self._matrix()
        length = np.sum(np.abs([s.thickness for s in self.surfaces[:-1]]))
        try:
            cp = abcd.cardinal_points(M, length, n[0], n[-1])
            efl = cp.f_prime
        except ValueError:
            cp, efl = None, np.inf
        ep_z, ep_r, xp_z, xp_r = self._pupils()
        (y_m, u_m), (y_c, u_c) = self._paraxial_rays(field)
        ym, um = self._paraxial_trace(y_m, u_m)
        yc, uc = self._paraxial_trace(y_c, u_c)
        image_distance = -ym[-1]/um[-1] if um[-1] != 0 else np.inf
        image_height = yc[-1] + uc[-1]*image_distance
        if self.infinite_object:
            magnification = 0.0
        else:
            magnification = abcd.image(M, self.object_distance, n[0], n[-1])[1]
        bfl = cp.bfl*self.sigma[-1] if cp is not None else np.inf
        ffl = cp.ffl if cp is not None else np.inf
        na = abs(n_s[-1]*um[-1])
        f_number = efl/(2*ep_r) if self.infinite_object else 1/(2*na)
        lagrange = n_s[0]*(u_c*y_m - u_m*y_c)

        return Paraxial(efl, bfl, ffl, cp, ep_z, ep_r, xp_z, xp_r, image_distance, image_height,
                        magnification, f_number, na, lagrange, np.column_stack([ym, um]),
                        np.column_stack([yc, uc]), M)

    def seidel(self, field=None):
        """Seidel sums of the surfaces for the paraxial marginal ray at full
        aperture and the chief ray at field (default: maximum field).

        Ideal lenses have no Seidel sums (NaN). See `Seidel`.
        """

        (y_m, u_m), (y_c, u_c) = self._paraxial_rays(field)
        ym, um = self._paraxial_trace(y_m, u_m)
        yc, uc = self._paraxial_trace(y_c, u_c)
        n_s = self.n*self.sigma
        u_in = np.concatenate([[u_m], um[:-1]])
        ub_in = np.concatenate([[u_c], uc[:-1]])
        H = n_s[0]*(u_c*y_m - u_m*y_c)
        S = np.zeros((len(self.surfaces), 5))
        for i, s in enumerate(self.surfaces):
            if isinstance(s, IdealLens):
                S[i] = np.nan
                continue
            n1, n2 = n_s[i], n_s[i + 1]
            c, y, yb = s.c, ym[i], yc[i]
            A = n1*(u_in[i] + y*c)
            Ab = n1*(ub_in[i] + yb*c)
            d_u = um[i]/n2 - u_in[i]/n1
            d_1n = 1/n2 - 1/n1
            d_1n2 = 1/n2**2 - 1/n1**2
            S_I = -A**2*y*d_u
            S_II = -A*Ab*y*d_u
            S_III = -Ab**2*y*d_u
            S_IV = -H**2*c*d_1n
            S_V = -Ab**3*y*d_1n2 + c*d_1n*Ab*yb*(2*y*Ab - yb*A)
            # departure of the surface from the sphere to fourth order:
            a4 = s.conic*c**3/8 + (s.aspheric[0] if s.aspheric else 0.0)
            dS = 8*a4*(n2 - n1)
            S[i] = [S_I + dS*y**4, S_II + dS*y**3*yb, S_III + dS*y**2*yb**2, S_IV,
                    S_V + dS*y*yb**3]
        tot = S.sum(axis=0)
        wl = self.wavelength

        # Welford's sums are positive for undercorrected systems, whose OPD
        # (OPL_ray - OPL_chief) is negative at the rim:
        return Seidel(S, tot, -tot[0]/8/wl, -tot[1]/2/wl, -tot[2]/2/wl, -(tot[2] + tot[3])/4/wl,
                      -tot[4]/2/wl)

    # real rays:
    def rays(self, field=None, px=0.0, py=0.0, aim=None):
        """Rays from the object point at field through the pupil points
        (px, py) (normalized entrance pupil coordinates, arrays or numbers).

        aim : bool, optional
            Ray aiming at the real stop (default: self.ray_aiming).
        """

        field = self.max_field if field is None else field
        px, py = np.broadcast_arrays(np.atleast_1d(np.asarray(px, float)),
                                     np.atleast_1d(np.asarray(py, float)))
        px, py = px.ravel(), py.ravel()
        fx, fy = _field_xy(field)
        if self.infinite_object:
            ep_r = self._pupils()[1]
            a, b = px*ep_r, py*ep_r
        else:
            u_m, u_c = self._object_slopes()
            a, b = u_c*fx + px*u_m, u_c*fy + py*u_m
        if self.ray_aiming if aim is None else aim:
            a, b = self._aim(field, a, b, px, py)

        return self._launch(field, a, b)

    def _launch(self, field, a, b):
        """Rays through the points (a, b) of the paraxial entrance pupil plane
        (object at infinity) or from the object point with the slopes
        (a, b) = (dx/dz, dy/dz) (finite object)."""

        n0 = self.n_object
        fx, fy = _field_xy(field)
        if self.infinite_object:
            ep_z = self._pupils()[0]
            E = np.stack([a, b, np.full(len(a), ep_z)], axis=-1)
            d = np.array([np.tan(fx), np.tan(fy), 1.0])
            d = np.tile(d/_norm(d), (len(a), 1))
            scale = max(np.sum(np.abs([s.thickness for s in self.surfaces])), 1.0)
            z0 = min(0.0, ep_z) - 0.1*scale
            P = E - ((ep_z - z0)/d[:, 2])[:, None]*d
            opl = n0*_dot(P - [0.0, 0.0, ep_z], d)
        else:
            O = np.array([fx, fy, -self.object_distance])
            d = np.stack([a, b, np.ones(len(a))], axis=-1)
            d /= _norm(d)[:, None]
            P = np.tile(O, (len(a), 1))
            opl = np.zeros(len(a))

        return Rays(P, d, opl)

    def _stop_hits(self, field, a, b):
        """Ray hits on the stop in its local frame."""

        bundle = self.trace(self._launch(field, a, b), clip=False, last=self.stop_index)
        o, R = self.frames()[0][self.stop_index]
        Q = (bundle.points[-1] - o) @ R

        return Q[:, 0], Q[:, 1]

    def _aim(self, field, a, b, px, py):
        a_s = self.surfaces[self.stop_index].semi_diameter
        sign = np.sign(self._ep_matrix()[0]) if self.infinite_object else 1.0
        tx, ty = sign*px*a_s, sign*py*a_s
        scale = max(a_s, 1e-12)
        eps = 1e-6*scale if self.infinite_object else 1e-6*max(abs(self._object_slopes()[0]), 1e-12)
        for _ in range(30):
            hx, hy = self._stop_hits(field, a, b)
            rx, ry = hx - tx, hy - ty
            bad = ~np.isfinite(rx + ry)
            if np.all(bad | (np.hypot(rx, ry) < 1e-13*scale)):
                break
            h1x, h1y = self._stop_hits(field, a + eps, b)
            h2x, h2y = self._stop_hits(field, a, b + eps)
            J00, J10 = (h1x - hx)/eps, (h1y - hy)/eps
            J01, J11 = (h2x - hx)/eps, (h2y - hy)/eps
            with np.errstate(invalid="ignore", divide="ignore"):
                det = J00*J11 - J01*J10
                da = (J11*rx - J01*ry)/det
                db = (-J10*rx + J00*ry)/det
            ok = np.isfinite(da + db)
            a = np.where(ok, a - da, a)
            b = np.where(ok, b - db, b)

        return a, b

    def trace(self, rays, clip=True, last=None):
        """Trace rays through the system.

        Parameters
        ----------
        rays : Rays
            Start points (in object space, in front of the first surface),
            unit directions and optical path, e.g. from `rays()`.
        clip : bool, optional
            Vignette rays outside the semi diameters.
        last : int, optional
            Stop at this surface (no image plane); default: trace to the
            image plane.

        Returns
        -------
        RayBundle
        """

        P = np.array(rays.points, dtype=float).reshape(-1, 3)
        d = np.array(rays.directions, dtype=float).reshape(-1, 3)
        opl = np.array(rays.opl, dtype=float).reshape(-1)
        n = self.n
        frames, (o_img, R_img) = self.frames()
        valid = np.ones(len(P), dtype=bool)
        points, directions, opls = [P], [d], [opl]
        n_surf = len(self.surfaces) if last is None else last + 1
        for i in range(n_surf):
            s = self.surfaces[i]
            if isinstance(s, CoordinateBreak):
                points.append(P)
                directions.append(d)
                opls.append(opl)
                continue
            o, R = frames[i]
            # local coordinates (row vectors: x @ R = R.T x):
            P_l, d_l = (P - o) @ R, d @ R
            t = s.intersect(P_l, d_l)
            Q = P_l + t[:, None]*d_l
            opl = opl + n[i]*t
            if clip and s.semi_diameter is not None:
                valid &= np.hypot(Q[:, 0], Q[:, 1]) <= s.semi_diameter*(1 + 1e-9)
            d_l, dopl = s.deflect(Q, d_l, n[i], n[i + 1])
            P, d = o + Q @ R.T, d_l @ R.T
            opl = opl + dopl
            valid &= np.isfinite(t) & np.all(np.isfinite(d), axis=-1)
            P, d, opl = self._lose(P, d, opl, valid)
            points.append(P)
            directions.append(d)
            opls.append(opl)
        if last is None:
            normal = R_img[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                t = _dot(o_img - P, normal)/_dot(d, normal)
            P = P + t[:, None]*d
            opl = opl + n[-1]*t
            valid &= np.isfinite(t)
            P, d, opl = self._lose(P, d, opl, valid)
            points.append(P)
            directions.append(d)
            opls.append(opl)

        return RayBundle(np.array(points), np.array(directions), np.array(opls), valid, n)

    @staticmethod
    def _lose(P, d, opl, valid):
        P, d, opl = P.copy(), d.copy(), opl.copy()
        P[~valid] = np.nan
        d[~valid] = np.nan
        opl[~valid] = np.nan

        return P, d, opl

    def trace_pupil(self, field=None, px=0.0, py=0.0, clip=True):
        """Trace rays through the pupil points (px, py) at field."""

        return self.trace(self.rays(field, px, py), clip=clip)

    def _image_hits(self, bundle):
        """Image plane points and directions in the local image frame."""

        return self.to_image_frame(bundle.points[-1], bundle.directions[-1])

    # OPD:
    def _reference(self, chief):
        """Centre, radius and side of the reference sphere (image frame)."""

        Ic, dc = (v[0] for v in self._image_hits(chief))
        xp_z = self._pupils()[2]
        size = max(np.sum(np.abs([s.thickness for s in self.surfaces])), 1.0)
        if np.isfinite(xp_z) and abs(xp_z) < 1e8*size:
            # exit pupil plane, nominally xp_z - image_z from the image plane:
            Qc = Ic + (xp_z - self.image_z - Ic[2])/dc[2]*dc
        else:
            Qc = self.to_image_frame(chief.points[-2, 0])
        L = _dot(Ic - Qc, dc)

        return Ic, _norm(Ic - Qc), np.sign(L) if L != 0 else 1.0

    def _opd(self, bundle, chief):
        """OPD (length units) of the rays of bundle relative to the chief ray."""

        Ic, R, side = self._reference(chief)
        n_img = self.n[-1]
        (I, d), opl = self._image_hits(bundle), bundle.opl[-1]
        w = I - Ic
        wd = _dot(w, d)
        with np.errstate(invalid="ignore"):
            s = wd + side*np.sqrt(wd**2 - _dot(w, w) + R**2)

        return (opl - n_img*s) - (chief.opl[-1, 0] - n_img*side*R)

    @staticmethod
    def _pupil_grid(N):
        p = np.linspace(-1.0, 1.0, N)
        PX, PY = np.meshgrid(p, p)
        inside = PX**2 + PY**2 <= 1.0 + 1e-12

        return p, PX, PY, inside

    def opd(self, field=None, N=64, units="waves"):
        """OPD map over the pupil.

        Parameters
        ----------
        field : number, optional
            Default: maximum field.
        N : int, optional
            Samples across the pupil (a grid on [-1, 1]**2).
        units : {"waves", "length"}, optional

        Returns
        -------
        OPDMap
        """

        p, PX, PY, inside = self._pupil_grid(N)
        bundle = self.trace_pupil(field, PX[inside], PY[inside])
        chief = self.trace_pupil(field, 0.0, 0.0)
        W = np.full(PX.shape, np.nan)
        W[inside] = self._opd(bundle, chief)
        if units == "waves":
            W /= self.wavelength
        elif units != "length":
            raise ValueError("units must be 'waves' or 'length'")

        return OPDMap(p, p, W)

    def zernike(self, field=None, n_terms=16, N=64):
        """Least squares fit of Fringe Zernikes Z_1...Z_n_terms (on the unit
        disk of pupil coordinates) to the OPD map.

        Returns
        -------
        coeffs : array
            Coefficients in waves, coeffs[j - 1] belongs to Z_j.
        """

        m = self.opd(field, N)
        PX, PY = np.meshgrid(m.px, m.py)
        ok = np.isfinite(m.W)
        Z = FringeZernikes(m.px, m.py)
        A = np.column_stack([Z.eval_single_scattered(PX[ok], PY[ok], j)
                             for j in range(1, n_terms + 1)])

        return np.linalg.lstsq(A, m.W[ok], rcond=None)[0]

    def _valid_opd(self, field, N, remove_tilt):
        m = self.opd(field, N)
        PX, PY = np.meshgrid(m.px, m.py)
        ok = np.isfinite(m.W)
        W = m.W[ok]
        if remove_tilt:
            A = np.column_stack([np.ones_like(W), PX[ok], PY[ok]])
            W = W - A @ np.linalg.lstsq(A, W, rcond=None)[0]

        return W

    def rms_opd(self, field=None, N=64, remove_tilt=True):
        """RMS OPD in waves with piston (and by default tilt) removed, with
        uniform weighting of the (vignetted) pupil."""

        W = self._valid_opd(field, N, remove_tilt)

        return np.sqrt(np.mean((W - W.mean())**2))

    def strehl(self, field=None, N=64, remove_tilt=True):
        """Strehl ratio ``|<exp(2*pi*i*W)>|**2`` over the (vignetted) pupil.

        With remove_tilt (default), the best fitting tilt is removed, i.e.
        the reference point is moved close to the diffraction focus. Without,
        it is the intensity at the chief ray image point, which is the centre
        of `psf()`.
        """

        W = self._valid_opd(field, N, remove_tilt)

        return np.abs(np.mean(np.exp(2j*pi*W)))**2

    # hand-off to imaging:
    def pupil(self, field=None, N=64):
        """Exit pupil in the coordinates of `pyoptics.imaging`.

        The imaging pupil is parametrized by the transverse wave vector,
        (P_x, P_y) = n' (d_x, d_y)/NA with the image space ray directions d
        taken relative to the chief ray. The OPD of the traced rays is
        interpolated there; vignetted and unsampled regions are masked. NA is
        the largest ``n'|d - d_chief|`` of all rays.
        """

        from scipy.interpolate import griddata

        _, PX, PY, inside = self._pupil_grid(N)
        # grid points plus a ring of rays on the rim:
        phi = np.linspace(0, 2*pi, 4*N, endpoint=False)
        px = np.concatenate([PX[inside], np.cos(phi)])
        py = np.concatenate([PY[inside], np.sin(phi)])
        bundle = self.trace_pupil(field, px, py)
        chief = self.trace_pupil(field, 0.0, 0.0)
        W = self._opd(bundle, chief)
        ok = bundle.valid
        n_img = self.n[-1]
        d, d_c = self._image_hits(bundle)[1], self._image_hits(chief)[1]
        q = n_img*(d[ok, :2] - d_c[0, :2])
        NA = np.max(_norm(q))
        q = q/NA
        phase_rays = 2*pi*W[ok]/self.wavelength

        def phase(P_x, P_y):
            val = griddata(q, phase_rays, (P_x, P_y), method="linear")
            return np.nan_to_num(val)

        def mask(P_x, P_y):
            val = griddata(q, phase_rays, (P_x, P_y), method="linear")
            return np.isfinite(val).astype(float)

        return Pupil(NA, 2*pi/self.wavelength, phase, mask)

    def psf(self, x, y, field=None, N=64, normalization="peak"):
        """Amplitude PSF on the image plane grid x, y centred on the chief
        ray, from `imaging.get_psf()` with the traced pupil.

        With normalization="peak", ``|psf|**2`` is relative to the peak of the
        aberration-free PSF of the same pupil (Strehl ratio at the centre).
        """

        pup = self.pupil(field, N)

        return imaging.get_psf(x, y, pup.NA, pup.k, pup.mask, pup.phase, normalization)

    def mtf(self, x, y, field=None, N=64):
        """MTF from `imaging.get_mtf()` on the frequency grid of x, y."""

        pup = self.pupil(field, N)

        return imaging.get_mtf(x, y, pup.NA, pup.k, pup.mask, pup.phase)

    # geometrical aberrations:
    def spot_diagram(self, field=None, N=31):
        """Image plane intersections of a square grid of N x N pupil rays
        (inside the unit disk) relative to the chief ray."""

        _, PX, PY, inside = self._pupil_grid(N)
        bundle = self.trace_pupil(field, PX[inside], PY[inside])
        chief = self.trace_pupil(field, 0.0, 0.0)
        I = self._image_hits(bundle)[0][bundle.valid] - self._image_hits(chief)[0][0]
        x, y = I[:, 0], I[:, 1]
        centroid = (x.mean(), y.mean())
        rms = np.sqrt(np.mean((x - centroid[0])**2 + (y - centroid[1])**2))

        return SpotDiagram(x, y, rms, centroid)

    def ray_fan(self, field=None, N=51):
        """Transverse ray aberrations of the tangential and sagittal fans."""

        p = np.linspace(-1.0, 1.0, N)
        hits = lambda px, py: self._image_hits(self.trace_pupil(field, px, py))[0]
        chief = hits(0.0, 0.0)[0]
        tan = hits(0.0*p, p)
        sag = hits(p, 0.0*p)

        return RayFan(p, tan[:, 1] - chief[1], sag[:, 0] - chief[0])

    # plots:
    def draw(self, ax=None, fields=None, n_rays=7, colors=None, lw=0.8):
        """Draw the meridional section of the system with ray fans.

        fields : sequence, optional
            Fields to trace (default: 0 and the maximum field).
        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if fields is None:
            fields = [0.0, self.max_field] if self.max_field else [0.0]
        p = np.linspace(-1, 1, n_rays)
        r_max = np.zeros(len(self.surfaces))
        for j, field in enumerate(fields):
            b = self.trace_pupil(field, 0*p, p)
            pts = b.points
            color = colors[j] if colors is not None else "C{}".format(j)
            ax.plot(pts[:, :, 2], pts[:, :, 1], color=color, lw=lw)
            r_max = np.fmax(r_max, np.nanmax(np.abs(pts[1:-1, :, 1]), axis=1))

        n = self.n
        frames, (o_img, R_img) = self.frames()

        def to_global(i, yy, zz):
            o, R = frames[i]
            q = np.stack([0*yy, yy, zz], axis=-1) @ R.T + o
            return q[..., 2], q[..., 1]

        heights = []
        for i, s in enumerate(self.surfaces):
            h = s.semi_diameter if s.semi_diameter is not None else 1.05*r_max[i]
            heights.append(h)
            if isinstance(s, CoordinateBreak):
                continue
            yy = np.linspace(-h, h, 101)
            ax.plot(*to_global(i, yy, s.sag(yy**2)), color="k", lw=2.5 if s.mirror else 1.2)
        # lens edges between surfaces in glass:
        for i in range(len(self.surfaces) - 1):
            s1, s2 = self.surfaces[i], self.surfaces[i + 1]
            if n[i + 1] != 1.0 and not s1.mirror and not isinstance(s2, CoordinateBreak):
                # rims joined by a flat bevel on the smaller surface:
                h1, h2 = heights[i], heights[i + 1]
                for sgn in (-1, 1):
                    z1, y1 = to_global(i, np.array([sgn*h1]), s1.sag(np.array([h1**2])))
                    z2, y2 = to_global(i + 1, np.array([sgn*h2]), s2.sag(np.array([h2**2])))
                    corner = (z2, y1) if h1 > h2 else (z1, y2)
                    ax.plot([z1[0], corner[0][0], z2[0]], [y1[0], corner[1][0], y2[0]], color="k",
                            lw=1.2)
        h_img = 1.1*np.nanmax(np.abs(r_max)) if np.any(np.isfinite(r_max)) else 1.0
        corners = np.array([[0, -h_img, 0], [0, h_img, 0]]) @ R_img.T + o_img
        ax.plot(corners[:, 2], corners[:, 1], color="0.5", lw=1, ls="--")
        ax.axhline(0, color="0.6", lw=0.6, ls="-.")
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")

        return ax

    def plot_spot_diagram(self, fields=None, N=31, axes=None, scale=None):
        """Spot diagrams for several fields side by side."""

        import matplotlib.pyplot as plt

        if fields is None:
            fields = [0.0, self.max_field] if self.max_field else [0.0]
        if axes is None:
            _, axes = plt.subplots(1, len(fields), figsize=(3*len(fields), 3), squeeze=False)
            axes = axes[0]
        spots = [self.spot_diagram(f, N) for f in fields]
        if scale is None:
            scale = 1.1*max(np.max(np.hypot(s.x, s.y)) for s in spots)
        for ax, f, s in zip(axes, fields, spots):
            ax.plot(s.x, s.y, ".", ms=2)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_aspect("equal")
            ax.set_title("field {:.4g}\nRMS {:.3g}".format(f, s.rms), fontsize=9)

        return axes

    def plot_ray_fan(self, fields=None, N=51, axes=None):
        """Tangential and sagittal ray fans for several fields."""

        import matplotlib.pyplot as plt

        if fields is None:
            fields = [0.0, self.max_field] if self.max_field else [0.0]
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(8, 3))
        for j, f in enumerate(fields):
            fan = self.ray_fan(f, N)
            axes[0].plot(fan.p, fan.ey, color="C{}".format(j), label="field {:.4g}".format(f))
            axes[1].plot(fan.p, fan.ex, color="C{}".format(j))
        axes[0].set_xlabel("$P_y$")
        axes[0].set_ylabel(r"$\epsilon_y$")
        axes[0].set_title("tangential")
        axes[1].set_xlabel("$P_x$")
        axes[1].set_ylabel(r"$\epsilon_x$")
        axes[1].set_title("sagittal")
        axes[0].legend(fontsize=8)
        for ax in axes:
            ax.axhline(0, color="0.6", lw=0.6)

        return axes


# example designs:
def fused_silica(wavelength_nm):
    """Refractive index of fused silica (Malitson's Sellmeier formula, valid
    from 0.21 to 3.7 um)."""

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2

    return np.sqrt(1 + 0.6961663*l2/(l2 - 0.0684043**2) + 0.4079426*l2/(l2 - 0.1162414**2)
                   + 0.8974794*l2/(l2 - 9.896161**2))


# Caldwell's lens: surface, radius, thickness to the next surface, medium
# behind ("S": fused silica, "A": air), semi diameter, all in mm.
_CALDWELL_KRF = """
 1  -617.8800 30.375 S  61.30
 2  -207.0830  0.934 A  64.20
 3   201.9739 68.636 S  64.75
 4  -416.6217  0.865 A  59.60
 5   460.0439  7.061 S  55.25
 6   179.6999 15.608 A  55.25
 7  -373.0162  6.952 S  54.90
 8   249.4960 30.983 A  54.35
 9 -2591.2000 11.541 S  55.90
10   229.2357 33.165 A  56.85
11   -82.3025 11.524 S  57.45
12   569.8191  9.159 A  74.85
13  5523.6000 36.703 S  79.45
14  -156.8200  0.889 A  85.05
15   610.3354 41.168 S 100.20
16  -221.8862  0.883 A 101.90
17   528.5938 26.903 S 104.20
18  -570.2004  0.883 A 104.05
19   423.5775 21.883 S 101.00
20 -1396.3000  0.883 A 100.00
21   203.9075 22.715 S  91.85
22   835.4548 67.972 A  89.70
23  -735.8990  8.386 S  57.50
24   104.6386 23.616 A  50.55
25  -184.6683 11.034 S  49.95
26   288.7053 58.171 A  46.10
27   -74.5663 11.343 S  51.85
28  2319.0000 11.371 A  63.05
29  -283.4504 22.211 S  64.75
30  -142.5176  1.323 A  69.90
31 -5670.5000 39.484 S  81.85
32  -146.6908  0.883 A  86.45
33   654.7531 37.168 S  94.75
34  -347.7071  0.883 A  96.35
35   254.9142 31.600 S  96.45
36  2133.2000  0.883 A  94.50
37   164.8042 27.885 S  89.95
38   349.3775  0.884 A  86.00
39   108.9816 73.045 S  77.70
40    75.6698 54.069 A  46.50
41    46.2841 16.956 S  25.70
42    99.3161 13.168 A  19.85
"""


def caldwell_krf_lens(wavelength=248.3e-6, aperture_margin=1.02, refocus=True):
    """All fused silica KrF lithography projection lens by J. B. Caldwell.

    J. Brian Caldwell, "All-fused-silica 248-nm lithographic projection
    lens", Optics & Photonics News 9(11), 40-41 (1998). Reduction 4x
    (magnification -0.25), image side NA 0.56, image field diameter 23.4 mm
    (object height up to 46.8 mm), 1 m from object to image, telecentric on
    both sides. The prescription is that of the paper as reproduced in the
    COMSOL application "Microlithography Lens" (lens data in mm).

    The aperture stop is not marked in the published table; it is placed at
    surface 30 (index 29), whose clear aperture limits the axial beam to the
    design NA 0.56 and which makes the chief rays telecentric to a few mrad.
    The other semi diameters of the table clip the rim of the full field
    beam by about 1%; they are enlarged by aperture_margin.

    Parameters
    ----------
    wavelength : number, optional
        Vacuum wavelength in mm (default: KrF, 248.3 nm). The index of fused
        silica follows Malitson's formula, n = 1.5084 at 248.3 nm.
    aperture_margin : number, optional
        Factor for the semi diameters of all surfaces but the stop.
    refocus : bool, optional
        Put the image plane into the paraxial image at this wavelength
        (default). The lens has a large axial colour (about 0.25 um per pm),
        so the image distance of the table (13.168 mm, for n = 1.5084)
        changes by micrometres with the wavelength.

    Returns
    -------
    System
        Lengths in mm, object 107.954 mm in front of the first surface,
        maximum field object_height = 46.8 mm.
    """

    stop = 29
    surfaces = []
    for i, line in enumerate(_CALDWELL_KRF.strip().splitlines()):
        _, R, t, medium, sd = line.split()
        surfaces.append(Surface(float(R), float(t),
                                material=fused_silica if medium == "S" else None,
                                n_after=None if medium == "S" else 1.0,
                                semi_diameter=float(sd)*(1 if i == stop else aperture_margin),
                                label=str(i + 1)))

    system = System(surfaces, stop, wavelength, object_distance=107.954, object_height=46.8)

    return system.at_paraxial_focus() if refocus else system
