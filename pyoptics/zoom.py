#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Zoom lenses: thin lens kinematics of mechanically compensated zooms and
real zoom lenses with variable air spaces.

A zoom lens changes its focal length by moving groups of lenses along the
axis while the image stays in focus. The image would move with a single
moving group; a second group that moves on a different, non-linear path
(the *compensator*, driven by a cam) puts it back. This is *mechanical
compensation*, the design of nearly all zoom lenses since the 1950s. In an
*optically compensated* zoom, by contrast, the moving groups are linked
and move together, and the image stays in place only at a few zoom
positions.

Thin lens tools
---------------

- `conjugate_distance()`: distance object to image of a thin lens for an
  object distance; `compensator_positions()` solves the quadratic for the
  two positions of a lens that image a given point onto a fixed point (the
  two branches of a compensator cam that meet at magnification -1).
- `two_group_zoom()`: the negative-positive (or positive-negative)
  two group zoom of wide angle zoom lenses: separation and back focus as a
  function of the focal length, f = f1 f2/(f1 + f2 - d).
- `four_group_zoom()`: the classic afocal four group zoom (focusing group,
  negative variator, positive compensator, fixed prime lens): the
  compensator position and the focal length for every variator position.
- `thin_system_efl()`: focal length and back focus of thin lenses.

Real zoom lenses
----------------

`ZoomLens` holds a `raytrace.System` and the table of its variable air
spaces at a few zoom positions (as a patent lists them); `ZoomLens.system()`
builds the system at a tabulated position or at an interpolated one,
optionally re-solving one air space (the compensator) for focus.
`ZoomLens.group_positions()` gives the vertex positions of the groups
relative to the image plane, i.e. the cam curves.
`olympus_zoom_us7583450()` is example 1 of US 7,583,450 (Olympus, 2009), a
four group zoom of focal length 12.3 to 49.1 mm for a Four Thirds camera.

References
----------
R. Kingslake, "A History of the Photographic Lens", Academic Press (1989),
chapter 13 (zoom lenses).

A. D. Clark, "Zoom Lenses", Adam Hilger (1973).

K. Yamaji, "Design of zoom lenses", in E. Wolf (ed.), Progress in Optics 6,
105-170 (1967).

W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008),
section 13.6 (zoom systems).

M. Kawamura (Olympus Imaging Corp.), "Zoom lens", US 7,583,450 B2 (2009),
example 1.
"""

import copy

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from . import raytrace as rt


def conjugate_distance(s, f):
    """Distance from object to image of a thin lens.

    Parameters
    ----------
    s : number or array
        Object position relative to the lens (negative for a real object
        in front of it).
    f : number
        Focal length.

    Returns
    -------
    T : number or array
        Image position minus object position, s' - s with 1/s' - 1/s = 1/f.
    """

    s = np.asarray(s, dtype=float)
    s_img = 1/(1/f + 1/s)

    return s_img - s


def thin_image(z_object, z_lens, f):
    """Image position of a thin lens at z_lens for an object point at z_object.

    Parameters
    ----------
    z_object : number or array
        Object position (np.inf or -np.inf: object at infinity).
    z_lens : number or array
        Lens position.
    f : number
        Focal length.

    Returns
    -------
    z_image : number or array
    """

    s = np.asarray(z_object, dtype=float) - z_lens
    with np.errstate(divide="ignore"):
        inv = 1/f + np.where(np.isinf(s), 0.0, 1/s)

    return z_lens + 1/inv


def compensator_positions(z_object, z_image, f):
    """Positions of a thin lens that images z_object onto z_image.

    With D = z_image - z_object the lens position z = z_object + x solves
    x**2 + D x + f D = 0 (real object and image, x = -s), so there are two
    solutions for D**2 >= 4 f D (abs(D) >= 4f for a positive lens with a real
    image), one with abs(magnification) < 1 and one with abs(magnification) > 1;
    they meet at magnification -1.

    Parameters
    ----------
    z_object, z_image : numbers or arrays
        Object and image positions.
    f : number
        Focal length.

    Returns
    -------
    z1, z2 : arrays
        The two lens positions (NaN where there is no solution), z1 <= z2.
    """

    zo = np.asarray(z_object, dtype=float)
    D = np.asarray(z_image, dtype=float) - zo
    # lens at z = zo + x: s = -x, s' = D - x; 1/(D - x) + 1/x = 1/f -> x^2 - D x + f D = 0
    disc = D**2 - 4*f*D
    with np.errstate(invalid="ignore"):
        r = np.sqrt(disc)
    x1, x2 = (D - r)/2, (D + r)/2

    return zo + np.minimum(x1, x2), zo + np.maximum(x1, x2)


def thin_system_efl(f, z):
    """Focal length and back focus of thin lenses in air.

    Parameters
    ----------
    f : sequence
        Focal lengths.
    z : sequence
        Lens positions (increasing).

    Returns
    -------
    efl : number
        Effective focal length.
    z_focus : number
        Position of the rear focus (image of an object at infinity).
    """

    y, u = 1.0, 0.0
    z_last = z[0]
    for fi, zi in zip(f, z):
        y = y + u*(zi - z_last)
        u = u - y/fi
        z_last = zi
    efl = -1.0/u

    return efl, z_last - y/u


def two_group_zoom(f1, f2, efl):
    """Two group zoom of thin lenses: separation and back focus.

    The focal length of two thin lenses at the separation d is
    f = f1 f2/(f1 + f2 - d), so d = f1 + f2 - f1 f2/f, and the back focus
    (second lens to image) is f (1 - d/f1). Both groups move: the
    separation shrinks and the back focus grows linearly with f, the rear
    group moves linearly, the front group on a hyperbola.

    Parameters
    ----------
    f1, f2 : numbers
        Focal lengths of the front and rear group.
    efl : number or array
        Focal lengths of the zoom.

    Returns
    -------
    d : array
        Separation of the groups.
    back : array
        Back focus.
    """

    efl = np.asarray(efl, dtype=float)
    d = f1 + f2 - f1*f2/efl

    return d, efl*(1 - d/f1)


def four_group_zoom(f1, f2, f3, f4, z2, z1=0.0, z4=None, z_target=None, branch=0):
    """Mechanically compensated four group zoom of thin lenses.

    Group 1 (positive, fixed, also the focusing group) forms an image of
    the object at infinity in its rear focus. Group 2 (the negative
    variator) moves and images it again, to a point a2 that moves
    non-linearly with the variator. Group 3 (the compensator) must take
    this moving point to a fixed point, which group 4 (the fixed prime or
    relay lens) images into the fixed image plane:

    - ``z_target=None`` (afocal zoom): the fixed point is at infinity, group
      3 sits one focal length behind a2, z3 = a2 + f3; groups 1 to 3 are an
      afocal telescope of variable magnification in front of the prime lens.
    - ``z_target`` given: group 3 images a2 onto z_target, a quadratic with
      two solutions (`compensator_positions()`), the two branches of the cam
      which meet where group 3 works at magnification -1.

    Parameters
    ----------
    f1, f2, f3, f4 : numbers
        Focal lengths of the groups.
    z2 : array
        Positions of the variator.
    z1 : number, optional
        Position of group 1.
    z4 : number, optional
        Position of the prime group; default far behind the others.
    z_target : number, optional
        Fixed image of group 3 (None: infinity).
    branch : {0, 1}, optional
        Branch of the compensator for a finite z_target (0: nearer to the
        variator).

    Returns
    -------
    z3 : array
        Compensator positions (NaN where there is no solution).
    efl : array
        Focal lengths of the whole zoom.
    z_image : array
        Image positions (constant where the compensation works).
    """

    z2 = np.atleast_1d(np.asarray(z2, dtype=float))
    if z4 is None:
        z4 = z1 + 4*(abs(f1) + abs(f2) + abs(f3))
    a1 = z1 + f1                        # image of infinity by group 1
    a2 = thin_image(a1, z2, f2)         # image by the variator
    if z_target is None:
        z3 = a2 + f3
    else:
        z3 = compensator_positions(a2, z_target, f3)[branch]
    efl = np.full(len(z2), np.nan)
    zi = np.full(len(z2), np.nan)
    for k in range(len(z2)):
        if np.isfinite(z3[k]):
            efl[k], zi[k] = thin_system_efl([f1, f2, f3, f4], [z1, z2[k], z3[k], z4])

    return z3, efl, zi


class ZoomLens:
    """A real zoom lens: a system with variable air spaces.

    Parameters
    ----------
    system : raytrace.System
        The lens at any zoom position (its variable thicknesses are
        replaced).
    variable : dict
        {surface index: sequence of thicknesses at the zoom positions} (the
        thickness behind the surface; the last surface's thickness is the
        distance to the image plane).
    focal_lengths : sequence
        Nominal focal lengths at the zoom positions (increasing).
    f_numbers : sequence, optional
        F-numbers at the zoom positions (the aperture follows them by
        setting the entrance pupil radius).
    groups : sequence of (first, last), optional
        Surface index ranges of the groups, for `group_positions()`.
    """

    def __init__(self, system, variable, focal_lengths, f_numbers=None, groups=None):
        self.base = system.copy()
        self.variable = {int(k): np.asarray(v, dtype=float) for k, v in variable.items()}
        self.focal_lengths = np.asarray(focal_lengths, dtype=float)
        self.f_numbers = None if f_numbers is None else np.asarray(f_numbers, dtype=float)
        self.groups = groups
        self._zeta = np.log(self.focal_lengths)

    def __len__(self):
        return len(self.focal_lengths)

    def _interp(self, values, zeta):
        k = len(self._zeta)
        if k == 1:
            return float(values[0])
        if k == 2:
            return float(np.interp(zeta, self._zeta, values))
        # quadratic (3 positions) or natural cubic spline in log(f)
        if k == 3:
            p = np.polyfit(self._zeta, values, 2)
            return float(np.polyval(p, zeta))

        return float(CubicSpline(self._zeta, values, bc_type="natural")(zeta))

    def spacings(self, position):
        """Variable thicknesses at a zoom position.

        Parameters
        ----------
        position : int or float
            Index of a tabulated position (int) or a nominal focal length
            (float; the spacings are interpolated over log f by a quadratic
            through three positions or a cubic spline).

        Returns
        -------
        spacings : dict
            {surface index: thickness}.
        """

        if isinstance(position, (int, np.integer)):
            return {i: float(v[position]) for i, v in self.variable.items()}
        zeta = np.log(float(position))

        return {i: self._interp(v, zeta) for i, v in self.variable.items()}

    def system(self, position, solve=None, field_angle=None):
        """The lens at a zoom position.

        Parameters
        ----------
        position : int or float
            Tabulated position (int) or nominal focal length (float), see
            `spacings()`.
        solve : (i, j), optional
            Gaps around a group that is moved to put the paraxial image
            into the image plane (the compensator), see `refocus()`.
        field_angle : number, optional
            Maximum field angle (radians) of the returned system.

        Returns
        -------
        system : raytrace.System
        """

        s = self.base.copy()
        for i, t in self.spacings(position).items():
            s.surfaces[i].thickness = t
        if self.f_numbers is not None:
            if isinstance(position, (int, np.integer)):
                F = self.f_numbers[position]
            else:
                F = self._interp(self.f_numbers, np.log(float(position)))
            efl = s.paraxial().efl
            s = rt.System(s.surfaces, s.stop_index, s.wavelength, s.object_distance,
                          s.max_field, s.max_field, s.n_object, s.units,
                          entrance_pupil_radius=efl/F/2)
        if field_angle is not None:
            s.max_field = field_angle
        if solve is not None:
            s = self.refocus(s, solve)[0]

        return s

    @staticmethod
    def refocus(system, gaps):
        """Move a group along the axis such that the paraxial image of the
        object lies in the image plane (the job of the compensator).

        Parameters
        ----------
        system : raytrace.System
        gaps : (i, j)
            The thickness behind surface i grows by dt and the one behind
            surface j (j > i) shrinks by dt: the surfaces i + 1 ... j move by
            dt, the image plane stays where it is.

        Returns
        -------
        system : raytrace.System
            Copy with the group moved.
        dt : number
            The shift.
        """

        i, j = gaps
        s = system.copy()
        target = s.surfaces[-1].thickness
        ti, tj = s.surfaces[i].thickness, s.surfaces[j].thickness

        def moved(dt):
            o = s.copy()
            o.surfaces[i].thickness = ti + dt
            o.surfaces[j].thickness = tj - dt
            return o

        def err(dt):
            return moved(dt).paraxial().image_distance - target

        # the root nearest to the current position (there can be two: the
        # two branches of a compensator cam)
        grid = np.linspace(-0.95*ti, 0.95*tj, 81)
        e = np.array([err(g) for g in grid])
        k = np.nonzero(np.sign(e[:-1]) != np.sign(e[1:]))[0]
        if len(k) == 0:
            raise ValueError("No focus within the gaps.")
        k = k[np.argmin(np.abs(grid[k] + grid[k + 1]))]
        dt = brentq(err, grid[k], grid[k + 1], xtol=1e-12)

        return moved(dt), dt

    def group_positions(self, system):
        """Vertex positions of the first surface of every group, relative to
        the image plane (negative: in front of it).

        Parameters
        ----------
        system : raytrace.System
            The lens at some zoom position.

        Returns
        -------
        z : array
            One position per group (needs `groups`).
        """

        if self.groups is None:
            raise ValueError("No groups given.")
        vz = system.vertex_z
        z_img = system.image_z

        return np.array([vz[a] - z_img for a, b in self.groups])

    def group_focal_lengths(self, system=None):
        """Focal lengths of the groups (thick, from the system matrix).

        Parameters
        ----------
        system : raytrace.System, optional
            Default: the lens at the first position.

        Returns
        -------
        f : array
        """

        s = self.system(0) if system is None else system

        P = np.array([-s._matrix(a, b + 1)[1, 0] for a, b in self.groups])
        with np.errstate(divide="ignore"):
            return np.where(P != 0, 1/np.where(P != 0, P, 1.0), np.inf)


# Example 1 of US 7,583,450 B2: r, d, (n_d, V_d); "d5" etc. are the variable spacings.
_US7583450 = [
    (109.4949, 2.5500, (1.84666, 23.78)), (65.6064, 6.5166, (1.56384, 60.67)),
    (250.3611, 0.2000, None), (56.3041, 5.2365, (1.77250, 49.60)), (129.4446, "d5", None),
    (56.6646, 2.0000, (1.88300, 40.76)), (15.7727, 2.9353, None),
    (59.6791, 1.9684, (1.80610, 40.88)), (20.8286, 6.1861, None),
    (-31.0694, 1.1413, (1.43875, 94.93)), (15.7046, 6.7088, (1.61293, 37.00)),
    (-29.7268, "d12", None), (-24.5029, 1.2783, (1.80610, 40.88)), (-92.4430, "d14", None),
    (np.inf, "d15", None),
    (67.0118, 2.4442, (1.76182, 26.52)), (-69.5017, 0.9379, None),
    (-27.8030, 1.1994, (1.75500, 52.32)), (-81.0550, "d19", None),
    (35.1459, 6.7687, (1.43875, 94.93)), (-17.3775, 1.2000, (1.88300, 40.76)),
    (-30.3574, 0.2000, None), (26.3652, 7.7834, (1.49700, 81.54)), (-24.7855, 0.1500, None),
    (-96.4756, 1.2057, (1.69680, 55.53)), (31.6869, 0.8737, None),
    (52.5247, 1.3000, (1.88300, 40.76)), (19.1738, 7.3100, (1.49700, 81.54)),
    (-32.1161, "d29", None),
    (np.inf, 4.6000, (1.51633, 64.14)), (np.inf, 1.0603, None),
]
_US7583450_ASPH = {
    8: (1.5703e-4, -6.2535e-7, 1.8023e-9, -4.7976e-12),
    9: (1.8640e-4, -3.0654e-7, 1.6790e-9, -1.1240e-11),
    13: (6.4830e-5, -1.4322e-7, 4.0045e-11),
    14: (5.2909e-5, -1.6207e-7),
    23: (-1.5009e-5, 3.8481e-8, -1.6118e-10),
    24: (2.6325e-5, -7.9830e-9, -7.1375e-11),
}
_US7583450_ZOOM = {
    "f": (12.31, 24.42, 49.14), "F": (2.88, 3.20, 3.57),
    "d5": (0.80000, 18.51118, 37.33684), "d12": (2.69354, 2.69354, 2.69354),
    "d14": (16.00294, 5.07006, 2.15164), "d15": (1.37426, 3.59390, 1.70000),
    "d19": (10.99914, 4.54263, 0.60000), "d29": (29.59504, 41.08752, 55.16241),
}
# object at 250 mm: the negative lens r13/r14 focuses (d12, d14 change)
_US7583450_NEAR = {
    "d5": (0.58742, 17.63032, 35.17989), "d12": (0.84541, 0.84541, 0.84541),
    "d14": (18.06366, 7.79904, 6.15671), "d15": (1.37426, 3.59390, 1.70000),
    "d19": (10.99914, 4.54263, 0.60000), "d29": (29.59504, 41.08752, 55.16241),
}


def olympus_zoom_us7583450(glasses="catalog", near=False, wavelength=587.56e-6,
                           image_height=10.8):
    """Example 1 of US 7,583,450 B2 (Olympus): a 4x zoom, f = 12.3 to
    49.1 mm, F/2.9 to F/3.6, for a Four Thirds camera.

    Groups (from the object): G1 positive (cemented doublet and a
    meniscus), G2 negative (meniscus, aspheric meniscus, cemented doublet
    with an S-FPL53 element) with the aspheric negative focusing lens
    r13/r14 behind it, the aperture stop, G3 (a weak positive-negative
    pair) and G4 positive (seven elements with two S-FPL53 and S-FPL51
    ED elements), then a 4.6 mm cover glass/filter plate. All groups move
    when zooming; focusing to 250 mm moves only the lens r13/r14.

    The patent gives n_d and V_d; with ``glasses="catalog"`` they are
    replaced by the nearest glasses of the OHARA catalogue (all within
    0.0002 in n_d), ``glasses="model"`` uses `materials.abbe_dispersion()`.
    The patent gives no field angle; image_height sets it from the paraxial
    focal length (10.8 mm: half the diagonal of a Four Thirds sensor).

    Parameters
    ----------
    glasses : {"catalog", "model"}, optional
    near : bool, optional
        Spacings for the object at 250 mm (else infinity).
    wavelength : number, optional
        In mm (d line default).
    image_height : number, optional
        Image height in mm for the maximum field.

    Returns
    -------
    zoom : ZoomLens
        Positions 0, 1, 2: wide, middle, tele. Groups: G1, G2, focusing
        lens, stop, G3, G4.
    """

    if glasses == "catalog":
        from .glasses import catalog
        cat = catalog(("ohara",))

        def material(m):
            return cat.nearest(*m)
    else:
        def material(m):
            return m

    table = _US7583450_NEAR if near else _US7583450_ZOOM
    surfaces, variable = [], {}
    for i, (R, d, m) in enumerate(_US7583450):
        if isinstance(d, str):
            variable[i] = table[d]
            d = table[d][0]
        surfaces.append(rt.Surface(R, d, material=None if m is None else material(m),
                                   n_after=1.0 if m is None else None,
                                   aspheric=_US7583450_ASPH.get(i + 1, ()),
                                   label="r{}".format(i + 1)))
    f = _US7583450_ZOOM["f"]
    F = _US7583450_ZOOM["F"]
    obj = 250.0 if near else np.inf
    s = rt.System(surfaces, 14, wavelength, object_distance=obj,
                  field_angle=np.arctan(image_height/f[0]) if not near else 0.0,
                  object_height=0.0, entrance_pupil_radius=f[0]/F[0]/2)
    groups = [(0, 4), (5, 11), (12, 13), (14, 14), (15, 18), (19, 28)]

    return ZoomLens(s, variable, f, F, groups)
