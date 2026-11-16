#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Monte Carlo tolerancing of `raytrace.System` lenses.

A tolerance analysis asks how good the *built* lenses are, given that every
radius, thickness, glass and element position is only made to within a
tolerance. The ingredients:

- **Tolerances** (`Radius`, `Power`, `Thickness`, `AxialShift`, `Index`,
  `Abbe`, `Decenter`, `ImagePlane`; `Tilt` of an element as a rigid body
  about a pivot, `Wedge` of a surface, `Irregularity` in fringes of a
  Fringe Zernike shape, `Inhomogeneity` of the index of a blank): one
  scalar error each, with a limit and a statistical distribution
  (`"normal"`: Gaussian with sigma = limit/2, cut at the limit;
  `"uniform"`; `"endpoints"`: plus or minus the limit).
  `element_tolerances()` builds the usual set for a list of elements.
- A **performance** vector whose sum of squares is the merit, block by
  block per field point: `SpotErrors` (transverse ray errors about the
  polychromatic centroid; a block's norm is the RMS spot radius) or
  `WavefrontErrors` (OPD with piston and tilt removed; a block's norm is the
  RMS wavefront error). Both trace the full pupil without vignetting, so
  that they stay valid for decentred systems and keep their length.
- **Compensators**: `lens_optimization.Variable` objects (the image
  distance `Thickness(-1)` for refocusing, `AirSpace` for axial element
  manipulators; here `LateralShift` for lateral ones, `ElementTilt`,
  `ImageTilt` for the levelling of a wafer or detector and `Wavelength`)
  that are adjusted after assembly to minimize the merit, optionally with
  soft ranges (`ranges=`: damped least squares, for manipulators with a
  limited stroke).
  Tolerances flagged `after_compensation` (such as the accuracy with which
  the focus is set, `ImagePlane`) are applied after the adjustment.
- Scalar **metrics** (e.g. the axial colour) computed for every system.

`ToleranceAnalysis` then gives

- `sensitivities()`: the change of the merit and the metrics for each
  tolerance at its limits (the inverse sensitivity table of lens design
  programs) and its share of the mean square merit,
- `monte_carlo()`: full ray traces of random lenses, each one compensated
  (Gauss-Newton steps with the compensator Jacobian of the nominal lens),
- `linear_model()`: the performance vector linearized in all tolerances
  and compensators, with the merit as the exact quadratic form of the
  linear model (the classical fast tolerancing of large lenses; nonlinear
  and cross terms are missing). `LinearModel.monte_carlo()` evaluates
  hundreds of thousands of lenses in seconds.
- `compensator_jacobian()` and `with_compensators()`: other compensators
  without tracing the tolerances again, e.g. to choose manipulators.
- `LinearModel.inverse_sensitivity()`: the error of each tolerance alone at
  which the merit (or a metric) reaches a target, and
  `LinearModel.allocate()`: tolerance limits that share an error budget
  (root sum square, equal or weighted shares, some limits fixed), applied
  with `with_limits()`.
- `yield_fraction()`: the fraction of lenses within the specification.

Monte Carlo runs can keep the performance vectors (`vectors=True`), e.g.
for `pyoptics.litho.budget`, which turns them into CD, placement and best
focus changes on the wafer.

Notebooks 70 (Monte Carlo tolerancing) and
150 (lens tolerances and wafer yield).

References
----------
W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008)
(tolerances, statistics, compensators).

R. R. Shannon, "The Art and Science of Optical Design", Cambridge University
Press (1997), chapter 7 (tolerancing).

H. H. Hopkins and H. J. Tiziani, "A theoretical and experimental study of
lens centring errors and their influence on optical image quality", Br. J.
Appl. Phys. 17, 33-54 (1966).
"""

import copy
from collections import namedtuple

import numpy as np

from .lens_optimization import Variable
from .bases import fringe_zernike, zernike_indices
from .raytrace import CoordinateBreak, IdealLens, Surface, ZernikeSurface


TEST_WAVELENGTH = 632.8e-6  # HeNe test plate wavelength in mm


# --- tolerances ----------------------------------------------------------------

def _truncated_normal_variance(cut=2.0):
    """Variance of a standard normal cut at +-cut sigma."""

    from math import erf, exp, pi, sqrt
    phi = exp(-cut**2/2)/sqrt(2*pi)
    Z = erf(cut/sqrt(2))

    return 1.0 - 2*cut*phi/Z


class Tolerance:
    """One scalar manufacturing or assembly error.

    Subclasses implement `apply()`, which changes a system in place by an
    error `value`.

    Parameters
    ----------
    limit : number
        Tolerance limit (the error lies in [-limit, limit]).
    distribution : {"normal", "uniform", "endpoints"}, optional
        "normal": Gaussian with sigma = limit/2, cut at the limit
        (default); "uniform" in [-limit, limit]; "endpoints": +-limit with
        equal probability.
    after_compensation : bool, optional
        Apply the error after the compensators have been adjusted (e.g. the
        accuracy of setting the focus).
    name : str, optional
    """

    def __init__(self, limit, distribution="normal", after_compensation=False, name=None):
        if distribution not in ("normal", "uniform", "endpoints"):
            raise ValueError("distribution must be 'normal', 'uniform' or 'endpoints'")
        self.limit = float(abs(limit))
        self.distribution = distribution
        self.after_compensation = after_compensation
        if name is not None:
            self.name = name

    name = "tolerance"
    group = "other"

    def apply(self, system, value):
        """Change the system in place by the error value.

        Parameters
        ----------
        system : raytrace.System
        value : number
        """

        raise NotImplementedError

    def sample(self, n, rng):
        """Random errors.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator

        Returns
        -------
        array
        """

        if self.distribution == "uniform":
            return rng.uniform(-self.limit, self.limit, n)
        if self.distribution == "endpoints":
            return self.limit*rng.choice((-1.0, 1.0), n)
        x = rng.normal(0.0, 0.5, n)
        bad = np.abs(x) > 1.0
        while np.any(bad):
            x[bad] = rng.normal(0.0, 0.5, bad.sum())
            bad = np.abs(x) > 1.0

        return self.limit*x

    @property
    def sigma(self):
        """Standard deviation of the distribution."""

        if self.distribution == "uniform":
            return self.limit/np.sqrt(3.0)
        if self.distribution == "endpoints":
            return self.limit

        return 0.5*self.limit*np.sqrt(_truncated_normal_variance(2.0))

    def __repr__(self):
        return "{}({}, ±{:.3g})".format(type(self).__name__, self.name, self.limit)


def _finite_radius(system, surface):
    s = system.surfaces[surface]
    if isinstance(s, (IdealLens, CoordinateBreak)):
        raise ValueError("Surface {} has no radius.".format(surface))

    return s


class Radius(Tolerance):
    """Error of the vertex radius of a surface, R -> R + value.

    Parameters
    ----------
    surface : int
    limit : number
        In the length unit.
    **kwargs
        See `Tolerance`.
    """

    group = "radius"

    def __init__(self, surface, limit, **kwargs):
        self.surface = surface
        self.name = "R{}".format(surface + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        s = _finite_radius(system, self.surface)
        s.R = s.R + value


class Power(Tolerance):
    """Radius error given in fringes of power against a test plate.

    A difference of the sag of N fringes, N*wavelength/2 at the rim h of
    the surface, is a change of curvature dc = N*wavelength/h**2 (works for
    plane surfaces too).

    Parameters
    ----------
    surface : int
    fringes : number
        Limit in fringes.
    rim : number
        Radius h of the tested aperture (e.g. the semi diameter).
    wavelength : number, optional
        Test wavelength in the length unit (default HeNe in mm).
    **kwargs
        See `Tolerance`.
    """

    group = "radius"

    def __init__(self, surface, fringes, rim, wavelength=TEST_WAVELENGTH, **kwargs):
        self.surface = surface
        self.rim = float(rim)
        self.test_wavelength = wavelength
        self.name = "R{}".format(surface + 1)
        super().__init__(fringes, **kwargs)

    def apply(self, system, value):
        s = _finite_radius(system, self.surface)
        c = s.c + value*self.test_wavelength/self.rim**2
        s.R = 1/c if c != 0 else np.inf


class Thickness(Tolerance):
    """Error of the axial thickness behind a surface (a centre thickness
    or an air space); all surfaces behind it move with it.

    Parameters
    ----------
    surface : int
    limit : number
    **kwargs
        See `Tolerance`.
    """

    group = "thickness"

    def __init__(self, surface, limit, **kwargs):
        self.surface = surface
        self.name = "t{}".format(surface + 1 if surface >= 0 else "img")
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        system.surfaces[self.surface].thickness += value


class AxialShift(Tolerance):
    """Axial position error of an element (surfaces first...last) between
    fixed neighbours: the gap in front grows, the gap behind shrinks
    (`System.with_axial_shift()`).

    Parameters
    ----------
    first, last : int
    limit : number
    **kwargs
        See `Tolerance`.
    """

    group = "axial position"

    def __init__(self, first, last, limit, **kwargs):
        self.first, self.last = first, last
        self.name = "z{}-{}".format(first + 1, last + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        if self.first == 0:
            if not system.infinite_object:
                system.object_distance += value
        else:
            system.surfaces[self.first - 1].thickness += value
        system.surfaces[self.last].thickness -= value


class ImagePlane(Tolerance):
    """Axial position error of the image plane (detector, wafer). By default
    applied after the compensators: the accuracy with which the focus is
    set or held.

    Parameters
    ----------
    limit : number
    after_compensation : bool, optional
    **kwargs
        See `Tolerance`.
    """

    group = "axial position"
    name = "z_img"

    def __init__(self, limit, after_compensation=True, **kwargs):
        super().__init__(limit, after_compensation=after_compensation, **kwargs)

    def apply(self, system, value):
        system.surfaces[-1].thickness += value


class Index(Tolerance):
    """Error of the refractive index of the medium behind a surface, the
    same at all wavelengths (`System.with_index_change()`).

    Parameters
    ----------
    surface : int
    limit : number
    **kwargs
        See `Tolerance`.
    """

    group = "index"

    def __init__(self, surface, limit, **kwargs):
        self.surface = surface
        self.name = "n{}".format(surface + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        s = system.surfaces[self.surface]
        if not s.changes_medium:
            raise ValueError("Surface {} does not start a new medium.".format(self.surface))
        s.index_offset = getattr(s, "index_offset", 0.0) + value


class Abbe(Tolerance):
    """Error of the Abbe number of a model glass material = (n_d, V_d).

    Parameters
    ----------
    surface : int
    limit : number
        Absolute limit, or relative (e.g. 0.008 for +-0.8 %) with relative.
    relative : bool, optional
    **kwargs
        See `Tolerance`.
    """

    group = "Abbe number"

    def __init__(self, surface, limit, relative=False, **kwargs):
        self.surface = surface
        self.relative = relative
        self.name = "V{}".format(surface + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        s = system.surfaces[self.surface]
        if not isinstance(s.material, tuple):
            raise ValueError("Abbe needs a model glass material=(n_d, V_d).")
        n_d, V_d = s.material
        s.material = (n_d, V_d*(1 + value) if self.relative else V_d + value)


class Decenter(Tolerance):
    """Lateral decentre of an element (surfaces first...last move together)
    along x or y.

    Parameters
    ----------
    first, last : int
    limit : number
    axis : {"x", "y"}, optional
    **kwargs
        See `Tolerance`.
    """

    group = "decenter"

    def __init__(self, first, last, limit, axis="y", **kwargs):
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.first, self.last, self.axis = first, last, axis
        self.name = "d{}{}-{}".format(axis, first + 1, last + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        k = 0 if self.axis == "x" else 1
        for i in range(self.first, self.last + 1):
            d = list(system.surfaces[i].decenter)
            d[k] += value
            system.surfaces[i].decenter = tuple(d)


def _element_vertices(system, first, last):
    """Axial positions of the vertices first...last relative to the first."""

    z = [0.0]
    for i in range(first, last):
        z.append(z[-1] + system.surfaces[i].thickness)

    return np.array(z)


def _tilt_element(system, first, last, angle, axis, pivot):
    """Rotate the surfaces first...last as a rigid body by an angle about the
    x or y axis through a point on the axis (pivot: its distance behind the
    first vertex in the untilted element; None: midway between the first and
    the last vertex). The tilt already present (that of the first surface
    about this axis, from earlier calls) is taken into account, so that
    rotations about the same pivot add up exactly."""

    if any(isinstance(system.surfaces[i], CoordinateBreak) for i in range(first, last + 1)):
        raise ValueError("Element tilts need an element without coordinate breaks.")
    k = 0 if axis == "x" else 1
    a0 = system.surfaces[first].tilt[k]
    a1 = a0 + angle
    z = _element_vertices(system, first, last)/np.cos(a0)  # along the element's own axis
    z = z - (0.5*z[-1] if pivot is None else pivot)
    for i, zi in zip(range(first, last + 1), z):
        s = system.surfaces[i]
        t = list(s.tilt)
        t[k] += angle
        s.tilt = tuple(t)
        # the vertex moves on a circle about the pivot (R_x(a): +z towards -y,
        # R_y(a): +z towards +x)
        d = list(s.decenter)
        if axis == "x":
            d[1] -= zi*(np.sin(a1) - np.sin(a0))
        else:
            d[0] += zi*(np.sin(a1) - np.sin(a0))
        s.decenter = tuple(d)
    # and along the axis by z (cos(a1) - cos(a0)): change the gaps, the
    # surfaces behind the element stay
    dz = z*(np.cos(a1) - np.cos(a0))
    if first == 0:
        if not system.infinite_object:
            system.object_distance += dz[0]
    else:
        system.surfaces[first - 1].thickness += dz[0]
    for i in range(first, last):
        system.surfaces[i].thickness += dz[i - first + 1] - dz[i - first]
    system.surfaces[last].thickness -= dz[-1]

class Tilt(Tolerance):
    """Tilt of an element (surfaces first...last rotate together as a rigid
    body) about the x or y axis through a pivot on the optical axis.

    Each surface is tilted by the angle and its vertex is moved on the
    circle about the pivot (to first order in the angle). A tilt about x
    (``axis="x"``) turns the element in the y-z plane, like a decentre
    along y.

    Parameters
    ----------
    first, last : int
    limit : number
        Angle in radians.
    axis : {"x", "y"}, optional
        Axis of the rotation.
    pivot : number, optional
        Axial position of the pivot behind the vertex of the first surface
        (default: midway between the first and the last vertex).
    **kwargs
        See `Tolerance`.
    """

    group = "tilt"

    def __init__(self, first, last, limit, axis="x", pivot=None, **kwargs):
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.first, self.last, self.axis, self.pivot = first, last, axis, pivot
        self.name = "t{}{}-{}".format(axis, first + 1, last + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        _tilt_element(system, self.first, self.last, value, self.axis, self.pivot)


class Wedge(Tolerance):
    """Tilt of one surface about its vertex, the wedge of an element (Zemax
    TSTX/TSTY).

    A wedge angle a of an element of diameter D is an edge thickness
    difference (ETD, TIR) of about a*D.

    Parameters
    ----------
    surface : int
    limit : number
        Angle in radians.
    axis : {"x", "y"}, optional
        Axis of the rotation.
    **kwargs
        See `Tolerance`.
    """

    group = "wedge"

    def __init__(self, surface, limit, axis="x", **kwargs):
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.surface, self.axis = surface, axis
        self.name = "w{}{}".format(axis, surface + 1)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        s = system.surfaces[self.surface]
        t = list(s.tilt)
        t[0 if self.axis == "x" else 1] += value
        s.tilt = tuple(t)


_PV_CACHE = {}


def _fringe_pv(j):
    """Peak-to-valley of the Fringe Zernike Z_j on the unit disk."""

    if j not in _PV_CACHE:
        r = np.sqrt(np.linspace(0.0, 1.0, 401))
        a = np.linspace(0.0, 2*np.pi, 361)
        R, A = np.meshgrid(r, a)
        Z = fringe_zernike(j, R*np.cos(A), R*np.sin(A))
        _PV_CACHE[j] = float(Z.max() - Z.min())

    return _PV_CACHE[j]


def _add_departure(system, surface, j, c, norm_radius):
    """Add c*Z_j(x/norm_radius, y/norm_radius) (Fringe) to the sag of a surface;
    a `raytrace.Surface` becomes a `raytrace.ZernikeSurface`."""

    s = system.surfaces[surface]
    if type(s) is Surface:
        z = ZernikeSurface.__new__(ZernikeSurface)
        z.__dict__.update(s.__dict__)
        z.zernike, z.norm_radius, z.ordering = (), float(norm_radius), "fringe"
        s = z
    elif type(s) is not ZernikeSurface or s.ordering != "fringe":
        raise ValueError("Surface {} is not a plain Surface or a ZernikeSurface with Fringe "
                         "ordering.".format(surface))
    # rescale to the surface's own normalization radius (exact only for the
    # terms whose radial order does not mix under scaling: use the same rim)
    if not np.isclose(s.norm_radius, norm_radius):
        raise ValueError("Surface {} has another Zernike normalization radius ({} against {})."
                         .format(surface, s.norm_radius, norm_radius))
    coeffs = dict(s.zernike)
    coeffs[int(j)] = coeffs.get(int(j), 0.0) + float(c)
    s.zernike = tuple(sorted(coeffs.items()))
    s._terms = tuple((n, m, scale*cj) for (n, m, scale), cj in
                     ((zernike_indices(jj, "fringe"), cj) for jj, cj in s.zernike))
    system.surfaces[surface] = s


class Irregularity(Tolerance):
    """Surface irregularity in fringes against a test plate: a figure error
    of the shape of one Fringe Zernike term (astigmatism Z5, Z6, spherical
    Z9, ...) whose peak-to-valley over the rim is N fringes, i.e.
    N*wavelength/2 of sag.

    The surface becomes a `raytrace.ZernikeSurface` (traced without the
    compiled kernel). The usual irregularity specification "N fringes,
    astigmatism and spherical" is modelled by one tolerance per term (5, 6
    and 9, see `element_tolerances()`).

    Parameters
    ----------
    surface : int
    fringes : number
        Limit in fringes (peak-to-valley of the departure).
    rim : number
        Radius of the tested aperture (normalization radius of the Zernikes).
    term : int, optional
        Fringe index (default 5, astigmatism at 0 degrees).
    wavelength : number, optional
        Test wavelength in the length unit (default HeNe in mm).
    **kwargs
        See `Tolerance`.
    """

    group = "irregularity"

    def __init__(self, surface, fringes, rim, term=5, wavelength=TEST_WAVELENGTH, **kwargs):
        if term < 2:
            raise ValueError("term must be at least 2 (Z1 is piston)")
        self.surface, self.rim, self.term = surface, float(rim), int(term)
        self.test_wavelength = wavelength
        self.name = "irr{}Z{}".format(surface + 1, term)
        super().__init__(fringes, **kwargs)

    def apply(self, system, value):
        c = value*self.test_wavelength/2/_fringe_pv(self.term)
        _add_departure(system, self.surface, self.term, c, self.rim)


class Inhomogeneity(Tolerance):
    """Index inhomogeneity of the glass of an element: an index variation
    dn*f(x, y) across the blank, with the shape f of a Fringe Zernike term
    scaled to a peak-to-valley of 1 over the rim (Z2, Z3: a linear gradient,
    Z4: a radial, quadratic variation, Z5, Z6: astigmatic, e.g. from
    stress).

    The glass is thin compared with the radii of the wavefront, so the error
    is the extra optical path t*dn*f(x, y) along the centre thickness t; it
    is put on the first surface of the element as a departure of the sag of
    -t*dn*f/(n' - n) (n, n': indices in front of and behind it), which adds
    this path to the rays.

    Parameters
    ----------
    first, last : int
        The element; its glass lies between the surfaces first and last.
    limit : number
        Peak-to-valley dn of the index variation (e.g. 2e-6).
    rim : number
        Radius over which the variation is given (normalization radius).
    term : int, optional
        Fringe index of the shape (default 4).
    **kwargs
        See `Tolerance`.
    """

    group = "inhomogeneity"

    def __init__(self, first, last, limit, rim, term=4, **kwargs):
        if term < 2:
            raise ValueError("term must be at least 2 (Z1 is a uniform index change, see Index)")
        self.first, self.last, self.rim, self.term = first, last, float(rim), int(term)
        self.name = "dn{}-{}Z{}".format(first + 1, last + 1, term)
        super().__init__(limit, **kwargs)

    def apply(self, system, value):
        n = system.n
        dn_surface = n[self.first + 1] - n[self.first]
        t = _element_vertices(system, self.first, self.last)[-1]
        c = -t*value/dn_surface/_fringe_pv(self.term)
        _add_departure(system, self.first, self.term, c, self.rim)


def element_tolerances(system, elements=None, fringes=None, radius=None, thickness=None,
                       index=None, abbe=None, decenter=None, air=None, shift=None, tilt=None,
                       wedge=None, irregularity=None, irregularity_terms=(5, 6, 9),
                       inhomogeneity=None, inhomogeneity_terms=(2, 3, 4), distribution="normal"):
    """The usual tolerances of a list of elements.

    Parameters
    ----------
    system : raytrace.System
    elements : list of (first, last), optional
        Default: `system.elements()`.
    fringes : number, optional
        Power tolerance of every surface in fringes (rim: semi diameter).
    radius : number, optional
        Alternatively an absolute radius tolerance.
    thickness : number, optional
        Centre thickness tolerance of the elements.
    index, abbe : number, optional
        Glass tolerances (abbe relative, e.g. 0.008).
    decenter : number, optional
        Decentre of each element, along x and y.
    air : number, optional
        Tolerance of the air spaces between the elements (despace).
    shift : number, optional
        Axial position tolerance of each element between its neighbours.
    tilt : number, optional
        Tilt of each element about x and y (radians, pivot midway between
        its vertices).
    wedge : number, optional
        Wedge of each element (radians): tilt of its last surface about x
        and y.
    irregularity : number, optional
        Irregularity of every surface in fringes, one tolerance per term of
        irregularity_terms (rim: semi diameter).
    irregularity_terms : sequence of int, optional
        Fringe Zernike terms of the irregularity (default astigmatism Z5,
        Z6 and spherical Z9).
    inhomogeneity : number, optional
        Peak-to-valley index inhomogeneity of each element, one tolerance
        per term of inhomogeneity_terms (rim: semi diameter of its first
        surface).
    inhomogeneity_terms : sequence of int, optional
        Shapes of the inhomogeneity (default linear gradients Z2, Z3 and
        radial Z4).
    distribution : str, optional

    Returns
    -------
    list of Tolerance
    """

    elements = system.elements() if elements is None else elements
    kw = dict(distribution=distribution)
    out = []
    for a, b in elements:
        for i in range(a, b + 1):
            s = system.surfaces[i]
            if fringes is not None:
                rim = s.semi_diameter if s.semi_diameter is not None else 1.0
                out.append(Power(i, fringes, rim, **kw))
            elif radius is not None and np.isfinite(s.R):
                out.append(Radius(i, radius, **kw))
    if thickness is not None:
        for a, b in elements:
            for i in range(a, b):
                out.append(Thickness(i, thickness, **kw))
    if air is not None:
        for _, b in elements[:-1]:
            out.append(Thickness(b, air, **kw))
    if shift is not None:
        out += [AxialShift(a, b, shift, **kw) for a, b in elements]
    if index is not None:
        for a, b in elements:
            for i in range(a, b):
                out.append(Index(i, index, **kw))
    if abbe is not None:
        for a, b in elements:
            for i in range(a, b):
                out.append(Abbe(i, abbe, relative=True, **kw))
    if decenter is not None:
        for a, b in elements:
            out += [Decenter(a, b, decenter, axis, **kw) for axis in ("x", "y")]
    if tilt is not None:
        for a, b in elements:
            out += [Tilt(a, b, tilt, axis, **kw) for axis in ("x", "y")]
    if wedge is not None:
        for _, b in elements:
            out += [Wedge(b, wedge, axis, **kw) for axis in ("x", "y")]

    def rim(i):
        s = system.surfaces[i]

        return s.semi_diameter if s.semi_diameter is not None else 1.0

    if irregularity is not None:
        for a, b in elements:
            for i in range(a, b + 1):
                out += [Irregularity(i, irregularity, rim(i), j, **kw) for j in irregularity_terms]
    if inhomogeneity is not None:
        for a, b in elements:
            out += [Inhomogeneity(a, b, inhomogeneity, rim(a), j, **kw)
                    for j in inhomogeneity_terms]

    return out


class LateralShift(Variable):
    """Compensator: lateral position (x or y) of an element, surfaces
    first...last (a lateral manipulator). The value is the decentre of
    the first surface.

    Parameters
    ----------
    first, last : int
    axis : {"x", "y"}, optional
    typical : number, optional
        Typical change (finite difference step 1e-4 of it).
    """

    def __init__(self, first, last, axis="y", typical=0.01):
        super().__init__()
        self.first, self.last = first, last
        self.k = 0 if axis == "x" else 1
        self._typical = typical
        self.name = "lateral {}{}-{}".format(axis, first + 1, last + 1)

    def get(self, system):
        return system.surfaces[self.first].decenter[self.k]

    def set(self, system, value):
        dv = value - self.get(system)
        for i in range(self.first, self.last + 1):
            d = list(system.surfaces[i].decenter)
            d[self.k] += dv
            system.surfaces[i].decenter = tuple(d)

    def typical(self, system):
        return self._typical



class ElementTilt(Variable):
    """Compensator: tilt of an element (surfaces first...last) about x or y
    through a pivot on the axis (a tilt manipulator, see `Tilt`). The value
    is the tilt of the first surface about that axis (radians).

    Parameters
    ----------
    first, last : int
    axis : {"x", "y"}, optional
    pivot : number, optional
        Axial position of the pivot behind the first vertex (default:
        midway between the first and the last vertex).
    typical : number, optional
        Typical change in radians (finite difference step 1e-4 of it).
    """

    def __init__(self, first, last, axis="x", pivot=None, typical=1e-3):
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.first, self.last, self.axis, self.pivot = first, last, axis, pivot
        self._typical = typical
        self.name = "tilt {}{}-{}".format(axis, first + 1, last + 1)

    def get(self, system):
        return system.surfaces[self.first].tilt[0 if self.axis == "x" else 1]

    def set(self, system, value):
        _tilt_element(system, self.first, self.last, value - self.get(system), self.axis,
                      self.pivot)

    def typical(self, system):
        return self._typical


class ImageTilt(Variable):
    """Compensator: tilt of the image plane about x or y (wafer or detector
    levelling), `System.image_tilt`.

    Parameters
    ----------
    axis : {"x", "y"}, optional
    typical : number, optional
        Typical change in radians.
    """

    def __init__(self, axis="y", typical=1e-4):
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.k = 0 if axis == "x" else 1
        self._typical = typical
        self.name = "image tilt {}".format(axis)

    def get(self, system):
        return system.image_tilt[self.k]

    def set(self, system, value):
        t = list(system.image_tilt)
        t[self.k] = value
        system.image_tilt = tuple(t)

    def typical(self, system):
        return self._typical


class Wavelength(Variable):
    """Compensator: the wavelength of the source (a line narrowed laser is
    tuned by fractions of a picometre; with the axial colour of a single
    glass lens this moves focus and spherical aberration).

    Parameters
    ----------
    typical : number, optional
        Typical change in the length unit (default 1e-9: 1 pm in mm).
    """

    name = "wavelength"

    def __init__(self, typical=1e-9):
        super().__init__()
        self._typical = typical

    def get(self, system):
        return system.wavelength

    def set(self, system, value):
        system.wavelength = value

    def typical(self, system):
        return self._typical

# --- performance ----------------------------------------------------------------

def _uncached(system):
    other = system.copy()
    other.__dict__.pop("_operand_cache", None)

    return other


def _full_pupil(rings, spokes):
    """Polar pupil points (centre plus rings) and their area weights (sum 1)."""

    radii = np.sort(np.asarray(rings, float))
    edges = np.concatenate([[0.0], (radii[1:] + radii[:-1])/2, [1.0]])
    c0 = edges[1]/2 if len(radii) else 0.5
    px, py, w = [0.0], [0.0], [c0**2]
    for r, e0, e1 in zip(radii, edges[:-1], edges[1:]):
        e0 = max(e0, c0)
        n = max(6, int(round(spokes*r)))
        a = 2*np.pi*np.arange(n)/n
        px += list(r*np.sin(a))
        py += list(r*np.cos(a))
        w += [(e1**2 - e0**2)/n]*n
    w = np.array(w)

    return np.array(px), np.array(py), w/w.sum()


def _field_points(system, fields):
    out = []
    for f in fields:
        f = np.atleast_1d(np.asarray(f, float))
        out.append(float(f[0])*system.max_field if f.size == 1
                   else (float(f[0])*system.max_field, float(f[1])*system.max_field))

    return out


class SpotErrors:
    """Transverse ray errors about the centroid of all wavelengths, per
    field; a block's norm is the polychromatic RMS spot radius.

    Parameters
    ----------
    fields : sequence, optional
        Field points as fractions of the maximum field: numbers (y) or
        pairs (x, y). Use both signs (and x fields) for decentred lenses.
    wavelengths : sequence, optional
        Default: the system's wavelength.
    weights : sequence, optional
        Weight per wavelength.
    rings : sequence, optional
        Pupil radii of the rings (plus the centre).
    spokes : int, optional
        Points per unit radius on a ring.
    """

    def __init__(self, fields=(0.0, 1.0, -1.0, (1.0, 0.0)), wavelengths=None, weights=None,
                 rings=(0.35, 0.6, 0.8, 0.95), spokes=16):
        self.fields = list(fields)
        self.wavelengths = wavelengths
        self.weights = weights
        self.px, self.py, self.w_pupil = _full_pupil(rings, spokes)
        self.units = "length"

    def __call__(self, system):
        """Performance vector of the system.

        Parameters
        ----------
        system : raytrace.System

        Returns
        -------
        array
        """

        wls = [system.wavelength] if self.wavelengths is None else list(self.wavelengths)
        wk = np.ones(len(wls)) if self.weights is None else np.asarray(self.weights, float)
        wk = wk/wk.sum()
        hits = []
        for w in wls:
            s = system if w == system.wavelength else system.with_wavelength(w)
            row = []
            for f in _field_points(s, self.fields):
                I = s._image_hits(s.trace(s.rays(f, self.px, self.py), clip=False))[0]
                row.append(I[:, :2])
            hits.append(row)
        out = []
        for j in range(len(self.fields)):
            P = np.stack([hits[k][j] for k in range(len(wls))])  # (wavelength, ray, 2)
            W = wk[:, None]*self.w_pupil[None, :]
            if not np.all(np.isfinite(P)):
                out.append(np.full(P.size, np.nan))
                continue
            c = np.einsum("kr,krd->d", W, P)
            out.append((np.sqrt(W)[..., None]*(P - c)).ravel())

        return np.concatenate(out)

    def blocks(self, size):
        """Slices of the vector per field.

        Parameters
        ----------
        size : int
            Length of the vector.

        Returns
        -------
        list of slice
        """

        n = size//len(self.fields)

        return [slice(j*n, (j + 1)*n) for j in range(len(self.fields))]


class WavefrontErrors:
    """OPD (waves) per field on a square grid of the pupil, with piston and
    tilt removed; a block's norm is the RMS wavefront error.

    Parameters
    ----------
    fields : sequence, optional
        Field points as fractions of the maximum field, numbers (y) or
        pairs (x, y).
    N : int, optional
        Grid points across the pupil.
    """

    def __init__(self, fields=(0.0, 0.7, 1.0, -1.0, (1.0, 0.0)), N=16):
        self.fields = list(fields)
        p = np.linspace(-1, 1, N)
        PX, PY = np.meshgrid(p, p)
        inside = PX**2 + PY**2 <= 1.0 + 1e-12
        self.px, self.py = PX[inside], PY[inside]
        A = np.column_stack([np.ones_like(self.px), self.px, self.py])
        self._proj = np.eye(len(self.px)) - A @ np.linalg.pinv(A)
        self.units = "waves"

    def __call__(self, system):
        """Performance vector of the system.

        Parameters
        ----------
        system : raytrace.System

        Returns
        -------
        array
        """

        out = []
        n = len(self.px)
        for f in _field_points(system, self.fields):
            bundle = system.trace(system.rays(f, self.px, self.py), clip=False)
            chief = system.trace(system.rays(f, 0.0, 0.0), clip=False)
            W = system._opd(bundle, chief)/system.wavelength
            out.append(self._proj @ W/np.sqrt(n) if np.all(np.isfinite(W)) else np.full(n, np.nan))

        return np.concatenate(out)

    def blocks(self, size):
        """Slices of the vector per field.

        Parameters
        ----------
        size : int
            Length of the vector.

        Returns
        -------
        list of slice
        """

        n = size//len(self.fields)

        return [slice(j*n, (j + 1)*n) for j in range(len(self.fields))]


def _solver(C, ranges=None):
    """Matrices M, K of the compensation: dc = -M y for a lens with the
    performance y, and a Gauss-Newton step -M y_c - K dc from a lens already
    compensated by dc (y_c): M is the pseudo-inverse of C, or
    (C^T C + R^-2)^-1 C^T with soft ranges R, and K = (C^T C + R^-2)^-1 R^-2
    (zero without ranges)."""

    if C.shape[1] == 0:
        return None, None
    if ranges is None:
        return np.linalg.pinv(C), np.zeros((C.shape[1], C.shape[1]))
    r = np.array([np.inf if v is None else float(v) for v in ranges])
    if len(r) != C.shape[1]:
        raise ValueError("One range per compensator.")
    A = C.T @ C + np.diag(1/r**2)

    return np.linalg.solve(A, C.T), np.linalg.solve(A, np.diag(1/r**2))


def _block_rms(Y, blocks):
    """RMS per block of the rows of Y (n, m) -> (n, n_blocks)."""

    Y = np.atleast_2d(Y)

    return np.sqrt(np.stack([np.sum(Y[:, b]**2, axis=1) for b in blocks], axis=1))


# --- analysis ------------------------------------------------------------------

MonteCarloResult = namedtuple("MonteCarloResult",
                              ["values", "rms", "metrics", "compensators", "vectors"],
                              defaults=(None,))
MonteCarloResult.__doc__ = """Monte Carlo lenses: the errors (n, n_tolerances),
the merit per field (n, n_fields) (RMS spot radius or RMS wavefront), the
metrics {name: (n,)}, the compensator changes (n, n_compensators) and, if
asked for, the performance vectors (n, m) (None otherwise)."""

Sensitivity = namedtuple("Sensitivity", ["names", "groups", "limits", "sigmas", "plus", "minus",
                                         "metrics_plus", "metrics_minus", "share"])
Sensitivity.__doc__ = """Inverse sensitivity table: per tolerance its name,
group, limit and standard deviation, the merit per field at +limit and
-limit (n_tol, n_fields, compensated), the metrics at +-limit
({name: (n_tol,)}) and the share of the tolerance in the increase of the
mean square merit (linear model, all fields together)."""


class ToleranceAnalysis:
    """Sensitivities and Monte Carlo analysis of a lens.

    Parameters
    ----------
    system : raytrace.System
        The nominal lens.
    tolerances : list of Tolerance
    performance : SpotErrors or WavefrontErrors (or a callable with blocks())
    compensators : list of lens_optimization.Variable, optional
        Adjusted after assembly, e.g. [lens_optimization.Thickness(-1)] for
        the focus.
    metrics : dict, optional
        {name: f(system) -> number}, evaluated for every (compensated)
        lens, e.g. the axial colour.
    steps : sequence, optional
        Finite difference steps of the compensators (default: 1e-4 of their
        typical change).
    iterations : int, optional
        Gauss-Newton steps of the compensation of a Monte Carlo lens.
    ranges : sequence, optional
        Soft range r_c of each compensator (None or inf: free): the
        compensation minimizes ``|y|**2 + sum_c (dc_c/r_c)**2`` instead of
        ``|y|**2`` (damped least squares), so that a compensator moves far only
        where it pays, like a manipulator with a limited stroke.
    """

    def __init__(self, system, tolerances, performance, compensators=(), metrics=None, steps=None,
                 iterations=1, ranges=None):
        self.system = _uncached(system)
        self.tolerances = list(tolerances)
        self.performance = performance
        self.compensators = list(compensators)
        self.metrics = dict(metrics or {})
        self.iterations = iterations
        self.y0 = performance(self.system)
        self.blocks = performance.blocks(len(self.y0))
        self.m0 = {k: f(self.system) for k, f in self.metrics.items()}
        self.C, self.G_comp = self.compensator_jacobian(self.compensators, steps)
        self.ranges = ranges
        self._C_pinv, self._C_damp = _solver(self.C, ranges)

    def compensator_jacobian(self, compensators, steps=None):
        """Jacobians of the performance vector and of the metrics with
        respect to compensators (central differences at the nominal lens).

        Parameters
        ----------
        compensators : list of lens_optimization.Variable
        steps : sequence, optional
            Default: 1e-4 of the typical change of each.

        Returns
        -------
        C : array (m, n_compensators)
        G : array (n_metrics, n_compensators)
        """

        if steps is None:
            steps = [1e-4*v.typical(self.system) for v in compensators]
        C, G = [], []
        for v, h in zip(compensators, steps):
            sp, sm = v.perturbed(self.system, h), v.perturbed(self.system, -h)
            C.append((self.performance(sp) - self.performance(sm))/(2*h))
            G.append([(f(sp) - f(sm))/(2*h) for f in self.metrics.values()])

        return (np.array(C).T.reshape(len(self.y0), len(compensators)),
                np.array(G).reshape(len(compensators), len(self.metrics)).T)

    def with_compensators(self, compensators, C=None, G=None, iterations=None, ranges=None):
        """The same analysis with other compensators, without tracing the
        nominal lens again.

        Parameters
        ----------
        compensators : list of lens_optimization.Variable
        C, G : arrays, optional
            Their Jacobians from `compensator_jacobian()` (traced if not
            given).
        iterations : int, optional
        ranges : sequence, optional
            Soft ranges of the compensators (see `ToleranceAnalysis`).

        Returns
        -------
        ToleranceAnalysis
        """

        other = ToleranceAnalysis.__new__(ToleranceAnalysis)
        other.__dict__.update(self.__dict__)
        other.compensators = list(compensators)
        if C is None:
            C, G = self.compensator_jacobian(other.compensators)
        nc = len(other.compensators)
        other.C = np.asarray(C, float).reshape(len(self.y0), nc)
        other.G_comp = np.asarray(G, float).reshape(len(self.metrics), nc)
        other.ranges = ranges
        other._C_pinv, other._C_damp = _solver(other.C, ranges)
        if iterations is not None:
            other.iterations = iterations

        return other

    @property
    def names(self):
        """Names of the tolerances."""

        return [t.name for t in self.tolerances]

    def nominal_rms(self):
        """Merit per field of the nominal lens (compensated).

        Returns
        -------
        array
        """

        return _block_rms(self._project(self.y0), self.blocks)[0]

    def _project(self, y):
        """Residual of y after the linear compensation."""

        if not self.compensators:
            return y

        return y - self.C @ (self._C_pinv @ y)

    def perturbed(self, values, compensate=True):
        """A lens with errors, compensated.

        Parameters
        ----------
        values : sequence
            One error per tolerance.
        compensate : bool, optional

        Returns
        -------
        system : raytrace.System
        y : array
            Performance vector.
        dc : array
            Compensator changes.
        """

        s = _uncached(self.system)
        post = []
        for t, v in zip(self.tolerances, values):
            if t.after_compensation:
                post.append((t, v))
            else:
                t.apply(s, v)
        dc = np.zeros(len(self.compensators))
        y = None
        if compensate and self.compensators:
            for _ in range(self.iterations):
                y = self.performance(s)
                if not np.all(np.isfinite(y)):
                    break
                step = -self._C_pinv @ y - self._C_damp @ dc
                for k, v in enumerate(self.compensators):
                    v.set(s, v.get(s) + step[k])
                dc += step
                s.__dict__.pop("_operand_cache", None)
            y = None
        for t, v in post:
            t.apply(s, v)
        if y is None:
            y = self.performance(s)

        return s, y, dc

    def sample(self, n, seed=None):
        """Random errors of n lenses.

        Parameters
        ----------
        n : int
        seed : int or numpy.random.Generator, optional

        Returns
        -------
        array (n, n_tolerances)
        """

        rng = np.random.default_rng(seed)

        return np.column_stack([t.sample(n, rng) for t in self.tolerances]) \
            if self.tolerances else np.zeros((n, 0))

    def monte_carlo(self, n=None, values=None, seed=None, compensate=True, vectors=False):
        """Full ray traces of random lenses.

        Parameters
        ----------
        n : int, optional
            Number of lenses (or give values).
        values : array (n, n_tolerances), optional
        seed : optional
        compensate : bool, optional
        vectors : bool, optional
            Keep the performance vectors (e.g. to evaluate them further).

        Returns
        -------
        MonteCarloResult
        """

        values = self.sample(n, seed) if values is None else np.atleast_2d(values)
        rms = np.zeros((len(values), len(self.blocks)))
        dcs = np.zeros((len(values), len(self.compensators)))
        mets = {k: np.zeros(len(values)) for k in self.metrics}
        Y = np.zeros((len(values), len(self.y0))) if vectors else None
        for i, v in enumerate(values):
            s, y, dc = self.perturbed(v, compensate)
            r = _block_rms(y, self.blocks)[0]
            rms[i] = np.where(np.isfinite(r), r, np.inf)
            dcs[i] = dc
            if vectors:
                Y[i] = y
            for k, f in self.metrics.items():
                mets[k][i] = f(s)

        return MonteCarloResult(values, rms, mets, dcs, Y)

    def with_limits(self, limits):
        """The same analysis with other tolerance limits (copies of the
        tolerances; e.g. from `LinearModel.allocate()`).

        Parameters
        ----------
        limits : sequence
            One limit per tolerance.

        Returns
        -------
        ToleranceAnalysis
        """

        other = ToleranceAnalysis.__new__(ToleranceAnalysis)
        other.__dict__.update(self.__dict__)
        other.tolerances = []
        for t, lim in zip(self.tolerances, limits):
            t = copy.copy(t)
            t.limit = float(abs(lim))
            other.tolerances.append(t)

        return other

    def _jacobian(self, fraction=1.0):
        """Central differences at +-fraction*limit of every tolerance:
        Jacobians of the performance vector and the metrics, and the
        vectors at the steps."""

        J = np.zeros((len(self.y0), len(self.tolerances)))
        G = np.zeros((len(self.metrics), len(self.tolerances)))
        Yp, Ym, Mp, Mm = [], [], [], []
        for k, t in enumerate(self.tolerances):
            h = fraction*t.limit
            pair = []
            for sign in (1.0, -1.0):
                s = _uncached(self.system)
                t.apply(s, sign*h)
                pair.append((self.performance(s), np.array([f(s) for f in self.metrics.values()])))
            (yp, mp), (ym, mm) = pair
            J[:, k] = (yp - ym)/(2*h)
            G[:, k] = (mp - mm)/(2*h)
            Yp.append(yp)
            Ym.append(ym)
            Mp.append(mp)
            Mm.append(mm)

        return J, G, np.array(Yp), np.array(Ym), np.array(Mp), np.array(Mm)

    def linear_model(self, fraction=1.0):
        """Performance vector and metrics linearized in the tolerances.

        Parameters
        ----------
        fraction : number, optional
            Finite difference step as a fraction of each limit.

        Returns
        -------
        LinearModel
        """

        J, G = self._jacobian(fraction)[:2]

        return LinearModel(self, J, G)

    def sensitivities(self, fraction=1.0):
        """Merit and metrics with each tolerance at its limits, compensated
        (linear compensation from the nominal lens), and the share of each
        tolerance in the increase of the mean square merit.

        Parameters
        ----------
        fraction : number, optional
            Evaluate at +-fraction*limit.

        Returns
        -------
        Sensitivity
        model : LinearModel
            The linear model from the same traces.
        """

        J, G, Yp, Ym, Mp, Mm = self._jacobian(fraction)
        model = LinearModel(self, J, G)
        post = np.array([t.after_compensation for t in self.tolerances], bool)
        y0c = self._project(self.y0)

        def comp(Y):
            out = np.array([self._project(y) for y in Y]) if len(Y) else np.zeros((0, len(y0c)))
            out[post] = y0c + (Y[post] - self.y0)

            return _block_rms(out, self.blocks) if len(Y) else np.zeros((0, len(self.blocks)))

        mp = {k: Mp[:, j] if len(Mp) else np.zeros(0) for j, k in enumerate(self.metrics)}
        mm = {k: Mm[:, j] if len(Mm) else np.zeros(0) for j, k in enumerate(self.metrics)}
        sig = np.array([t.sigma for t in self.tolerances])
        contrib = sig**2*np.sum(model.B**2, axis=0)
        share = contrib/contrib.sum() if contrib.sum() > 0 else contrib

        return Sensitivity(self.names, [t.group for t in self.tolerances],
                           np.array([t.limit for t in self.tolerances]), sig, comp(Yp), comp(Ym),
                           mp, mm, share), model


class LinearModel:
    """Performance vector and metrics of a `ToleranceAnalysis` linear in
    the errors, with the compensators solved by linear least squares.

    y = P (y0 + J_pre x_pre) + J_post x_post, P = 1 - C C^+ (the projection
    that removes what the compensators can correct), and m = m0 + G x +
    G_c c. The merit (block norms of y) is evaluated exactly from the linear
    y, i.e. it is quadratic in the errors.

    Parameters
    ----------
    analysis : ToleranceAnalysis
    J : array (m, n_tolerances)
        Jacobian of the performance vector (uncompensated).
    G : array (n_metrics, n_tolerances)
        Jacobian of the metrics.
    """

    def __init__(self, analysis, J, G):
        self.analysis = analysis
        self.J, self.G = J, G
        post = np.array([t.after_compensation for t in analysis.tolerances], bool)
        self.post = post
        B = np.array([analysis._project(J[:, k]) for k in range(J.shape[1])]).T \
            if J.shape[1] else np.zeros((len(analysis.y0), 0))
        B[:, post] = J[:, post]
        self.B = B  # compensated Jacobian
        self.a = analysis._project(analysis.y0)
        if analysis.compensators:
            Jpre = np.where(post[None, :], 0.0, J)
            self.D = -analysis._C_pinv @ Jpre  # compensator change per error
            self.c0 = -analysis._C_pinv @ analysis.y0
        else:
            self.D = np.zeros((0, J.shape[1]))
            self.c0 = np.zeros(0)

    def predict(self, values, vectors=False):
        """Linear model of lenses with the given errors.

        Parameters
        ----------
        values : array (n, n_tolerances)
        vectors : bool, optional
            Return the performance vectors too.

        Returns
        -------
        MonteCarloResult
        """

        X = np.atleast_2d(values)
        Y = self.a[None, :] + X @ self.B.T
        dc = self.c0[None, :] + X @ self.D.T
        mets = {}
        for j, (k, m0) in enumerate(self.analysis.m0.items()):
            mets[k] = m0 + X @ self.G[j] + dc @ self.analysis.G_comp[j]

        return MonteCarloResult(X, _block_rms(Y, self.analysis.blocks), mets, dc,
                                Y if vectors else None)

    def monte_carlo(self, n, seed=None, batch=20000, vectors=False):
        """Linear model of n random lenses.

        Parameters
        ----------
        n : int
        seed : optional
        batch : int, optional
            Lenses per batch (memory).
        vectors : bool, optional
            Return the performance vectors too (n x m numbers).

        Returns
        -------
        MonteCarloResult
        """

        rng = np.random.default_rng(seed)
        parts = [self.predict(self.analysis.sample(min(batch, n - i), rng), vectors)
                 for i in range(0, n, batch)]

        return MonteCarloResult(np.concatenate([p.values for p in parts]),
                                np.concatenate([p.rms for p in parts]),
                                {k: np.concatenate([p.metrics[k] for p in parts])
                                 for k in self.analysis.m0},
                                np.concatenate([p.compensators for p in parts]),
                                np.concatenate([p.vectors for p in parts]) if vectors else None)

    def mean_square(self, tolerances=None):
        """Expected mean square merit (all fields together) of the linear
        model: ``|a|**2 + sum_k sigma_k**2 |B_k|**2``.

        Parameters
        ----------
        tolerances : sequence of bool, optional
            Mask of the tolerances to include (default: all).

        Returns
        -------
        number
        """

        sig = np.array([t.sigma for t in self.analysis.tolerances])
        mask = np.ones(len(sig), bool) if tolerances is None else np.asarray(tolerances, bool)
        n_blocks = len(self.analysis.blocks)

        return (self.a @ self.a + np.sum(sig[mask]**2*np.sum(self.B[:, mask]**2, axis=0)))/n_blocks

    def with_compensators(self, compensators, C=None, G=None, ranges=None):
        """The same linear model with other compensators (the tolerances
        are not traced again).

        Parameters
        ----------
        compensators : list of lens_optimization.Variable
        C, G : arrays, optional
            Their Jacobians from `ToleranceAnalysis.compensator_jacobian()`.
        ranges : sequence, optional
            Soft ranges of the compensators (see `ToleranceAnalysis`).

        Returns
        -------
        LinearModel
        """

        return LinearModel(self.analysis.with_compensators(compensators, C, G, ranges=ranges),
                           self.J, self.G)

    def with_limits(self, limits):
        """The same linear model with other tolerance limits (the Jacobians
        are per unit error and stay).

        Parameters
        ----------
        limits : sequence
            One limit per tolerance.

        Returns
        -------
        LinearModel
        """

        return LinearModel(self.analysis.with_limits(limits), self.J, self.G)

    def _metric_jacobian(self, metric):
        """Change of a metric per unit error, compensators included."""

        j = list(self.analysis.m0).index(metric)

        return self.G[j] + self.analysis.G_comp[j] @ self.D if len(self.D) else self.G[j]

    def inverse_sensitivity(self, target, metric=None):
        """Inverse sensitivity: for each tolerance alone, the error at which
        the compensated merit reaches target (or a metric changes by
        target).

        For the merit, the RMS over the fields of the linear model,
        ``sqrt(|a + B_k x|**2/n_fields)``, is set to target and solved for the
        smaller ``|x|`` of both signs (zero if the nominal lens already misses
        the target); for a metric, ``|g_k| |x| = target``.

        Parameters
        ----------
        target : number
            Merit (RMS over the fields) or change of the metric.
        metric : str, optional
            Name of a metric instead of the merit.

        Returns
        -------
        array
            One error limit per tolerance (inf where it has no effect).
        """

        if metric is not None:
            g = np.abs(self._metric_jacobian(metric))
            with np.errstate(divide="ignore"):
                return np.where(g > 0, target/np.where(g > 0, g, 1.0), np.inf)
        nb = len(self.analysis.blocks)
        rest = nb*target**2 - self.a @ self.a
        out = np.full(self.B.shape[1], np.inf)
        if rest <= 0:
            return np.zeros(self.B.shape[1])
        for k in range(self.B.shape[1]):
            b = self.B[:, k]
            bb, ab = b @ b, self.a @ b
            if bb == 0:
                continue
            # bb x**2 + 2 ab x - rest = 0, roots for x > 0 and x < 0:
            root = np.sqrt(ab**2 + bb*rest)
            out[k] = min((-ab + root)/bb, (ab + root)/bb)

        return out

    def allocate(self, target, metric=None, weights=None, fixed=None):
        """Tolerance limits from an error budget (root sum square).

        The expected mean square merit of the linear model (`mean_square()`)
        or the variance of a metric is a sum over the tolerances of
        sigma_k**2 times a squared sensitivity. What the target leaves after
        the nominal lens and the fixed tolerances is shared out in proportion
        to the weights (equal shares by default), and each limit is chosen so
        that its tolerance takes exactly its share. A larger weight gives a
        looser tolerance, e.g. for what is expensive to make better.

        Parameters
        ----------
        target : number
            Expected RMS merit (root of `mean_square()`), or the standard
            deviation of the metric.
        metric : str, optional
            Name of a metric instead of the merit.
        weights : sequence, optional
            Share of each tolerance (default equal).
        fixed : sequence of bool, optional
            Tolerances that keep their limits.

        Returns
        -------
        array
            New limits (inf where a tolerance has no effect; pass them to
            `with_limits()` after capping them).
        """

        tols = self.analysis.tolerances
        ratio = np.array([t.sigma/t.limit for t in tols])  # sigma per unit limit
        sig = np.array([t.sigma for t in tols])
        if metric is None:
            s2 = np.sum(self.B**2, axis=0)/len(self.analysis.blocks)
            budget = target**2 - self.a @ self.a/len(self.analysis.blocks)
        else:
            s2 = self._metric_jacobian(metric)**2
            budget = target**2
        fixed = np.zeros(len(tols), bool) if fixed is None else np.asarray(fixed, bool)
        budget -= np.sum(sig[fixed]**2*s2[fixed])
        if budget <= 0:
            raise ValueError("The nominal lens and the fixed tolerances use up the budget.")
        w = np.ones(len(tols)) if weights is None else np.asarray(weights, float)
        free = ~fixed & (s2 > 0)
        out = np.array([t.limit for t in tols], float)
        out[~fixed & (s2 == 0)] = np.inf
        share = w[free]/w[free].sum()*budget
        out[free] = np.sqrt(share/s2[free])/ratio[free]

        return out


def yield_fraction(result, spec, metric_specs=None, fields=None):
    """Fraction of the lenses whose worst field merit is at most spec (and
    whose metrics are within their limits).

    Parameters
    ----------
    result : MonteCarloResult
    spec : number
        Limit of the merit (worst of the fields).
    metric_specs : dict, optional
        {name: limit of the absolute value}.
    fields : sequence of int, optional
        Fields to include (default: all).

    Returns
    -------
    number
    """

    rms = result.rms if fields is None else result.rms[:, list(fields)]
    ok = np.max(rms, axis=1) <= spec
    for k, lim in (metric_specs or {}).items():
        ok &= np.abs(result.metrics[k]) <= lim

    return ok.mean()
