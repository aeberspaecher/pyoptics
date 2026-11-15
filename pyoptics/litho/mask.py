#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Masks for lithographic imaging: layouts (geometry) and mask stacks
(materials and the model that turns them into transmissions).

Layout
------
A MaskLayout is one period (pitch_x, pitch_y) of a periodic mask that holds
rectangular features on a background (the field). A pitch of None makes the
mask invariant along that direction: lines along y have pitch_y = None. With
tone="clear" the field is clear and the features are absorber (lines on a
clear field mask), with tone="dark" the field is absorber and the features
are clear (contact holes, trenches). Isolated features are periodic features
with a pitch large enough that the neighbours do not interact (the coherence
length of the illumination, a few lambda/NA).

The diffraction orders are the Fourier coefficients of the transmission on
one period. They are computed analytically from the rectangles (products of
sinc functions), so there is no pixelation of the mask.

Manhattan polygons (MaskLayout.from_polygons()) are turned into the exact
union of non-overlapping rectangles, so L and T shapes, line ends with
hammerheads or serifs and OPC'd shapes have exact Fourier coefficients too.
Overlapping polygons are merged. The boundary layer model is not available
for polygon layouts (the rectangles of a polygon share internal edges).

Arbitrary polygons (edges at any angle: 45 degree wiring, octagonal or
round contacts, curvilinear ILT masks) are kept as polygons; their Fourier
coefficients are the exact transforms of the polygons (Lee and Mittra's
edge sum, `pyoptics.polygons.polygon_ft()`) sampled at the reciprocal
lattice, again without pixelation. Such polygons must not overlap (merge
them first, e.g. with gdstk.boolean(polygons, [], "or")); polygons with
holes are given as keyhole polygons, as in GDSII. The Kirchhoff models
(ThinMask, FilmMask) and Hopkins imaging work with them; the rigorous models
(RigorousMask, FDTDMask), the boundary layer model and the fragment based
OPC need rectangles. For the cut lines and CD metrics, features[0] is a
reference rectangle of the first polygon: the center of its bounding box
and the lengths of its chords along x and y through that center.

Layouts from GDSII/OASIS files: pyoptics does not read layout files, but
MaskLayout.from_gds() takes what the common Python layout libraries return
(gdstk and gdspy cells, polygon sets and lists of polygons, and dicts
{(layer, datatype): [polygons]} such as gdspy's get_polygons(by_spec=True)),
selects a layer, cuts a window (the period of the simulation) and scales
the coordinates, see `gds_polygons()`.

Mask stacks and approximations
------------------------------
A mask stack gives the transmissions of the clear and absorber regions as
Jones matrices for an illuminating plane wave (source point (s_x, s_y)):

- ThinMask: Kirchhoff (thin mask) approximation with given complex
  transmissions, e.g. binary (chrome on glass, t = 0) or attenuated phase
  shift masks (t = sqrt(T)*exp(i*phase)); independent of angle and
  polarization.
- FilmMask: the absorber is a stack of thin films (n + i*k, thickness) on
  the mask blank. Absorber and clear regions are taken as laterally infinite
  film stacks (a thin mask whose local transmissions are those of the films):
  their transmissions t_TE and t_TM are computed with pyoptics.thin_films for
  the angle of incidence of each source point at the mask, n*sin(theta) =
  ``NA*|s|/M`` (M: reduction ratio), and for TE and TM with respect to its plane
  of incidence. Phase and amplitude of the absorber, e.g. of an attenuated
  PSM, follow from the material data instead of being set by hand.
- BoundaryLayer (optional for both): thick mask (mask 3D) effects at the
  edges of the absorber are approximated by strips of width w centered on
  every edge, with a complex transmission that depends on whether the
  electric field is parallel (TE) or perpendicular (TM) to the edge
  (Tirapu-Azpiroz and Yablonovitch's boundary layer model). Its parameters
  come from a rigorous simulation or a calibration. At the corners of
  rectangles, the strips along y are taken to cover the corner squares.

Transmissions are relative to the clear area: the clear region has the
transmission 1 (FilmMask normalizes to the clear region unless
normalize=False), so that images can be normalized to the open frame.

References
----------
J. Tirapu-Azpiroz, P. Burchard and E. Yablonovitch, "Boundary layer model to
account for thick mask effects in photolithography", Proc. SPIE 5040, 1611
(2003).

A. Erdmann, "Optical and EUV Lithography: A Modeling Perspective", SPIE Press
(2021).

A. K.-K. Wong, "Resolution Enhancement Techniques in Optical Lithography", SPIE
Press (2001).
"""

import warnings

import numpy as np

from ..polarization import TE_TM_basis
from ..polygons import as_polygon, polygon_ft, contains, clip_to_rectangle
from ..polygons import polygon_area as _signed_area
from ..thin_films import _r_t


# layouts:

def _check_polygon(polygon):
    P = as_polygon(polygon)
    if len(P) < 4:
        raise ValueError("a Manhattan polygon is a sequence of at least four (x, y) vertices")
    d = np.roll(P, -1, axis=0) - P
    if np.any((d[:, 0] != 0) & (d[:, 1] != 0)):
        raise ValueError("polygon edges must be parallel to x or y (Manhattan)")

    return P


def is_manhattan(polygon):
    """True if all edges of the polygon are parallel to x or y.

    Parameters
    ----------
    polygon : (N, 2) array

    Returns
    -------
    bool
    """

    P = as_polygon(polygon)
    d = np.roll(P, -1, axis=0) - P

    return not np.any((d[:, 0] != 0) & (d[:, 1] != 0))


def polygon_area(polygon):
    """Signed area (positive for counterclockwise vertices).

    Parameters
    ----------
    polygon : (N, 2) array

    Returns
    -------
    A : float
    """

    return _signed_area(polygon)


def counterclockwise(polygon, manhattan=True):
    """The polygon with counterclockwise vertices, without repeated or
    collinear vertices.

    Parameters
    ----------
    polygon : (N, 2) array
    manhattan : bool, optional
        Require edges parallel to x or y.

    Returns
    -------
    polygon : (M, 2) array
    """

    P = _check_polygon(polygon) if manhattan else as_polygon(polygon)
    if polygon_area(P) < 0:
        P = P[::-1]
    P = P[np.any(P != np.roll(P, 1, axis=0), axis=1)]
    keep = []
    for i in range(len(P)):  # (exact tests: exact for edges parallel to x or y)
        a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
        if (b - a)[0]*(c - b)[1] - (b - a)[1]*(c - b)[0] != 0:  # not collinear
            keep.append(b)

    return np.array(keep)


def offset_polygon(polygon, distance):
    """Move every edge of a polygon outward along its normal (miter joins).

    A Manhattan polygon stays rectilinear. For other polygons the vertices
    move to the intersections of the shifted edges, which is exact for
    straight edges and a good approximation for finely sampled curves as
    long as the distance is small against the radii of curvature.

    Parameters
    ----------
    polygon : (N, 2) array
        Vertices (x, y).
    distance : number
        Shift of each edge along its outward normal (negative: inward).

    Returns
    -------
    polygon : (N, 2) array
        Offset polygon, counterclockwise.
    """

    P = counterclockwise(polygon, manhattan=is_manhattan(polygon))
    d = np.roll(P, -1, axis=0) - P
    normal = np.column_stack([d[:, 1], -d[:, 0]])/np.linalg.norm(d, axis=1)[:, None]
    out = []
    for i in range(len(P)):  # vertex i joins edge i - 1 and edge i
        n1, n2 = normal[i - 1], normal[i]
        out.append(P[i] + distance*(n1 + n2)/(1 + n1 @ n2))

    return np.array(out)


def _chord(P, x0, y0, axis):
    """Length of the chord of the polygon through (x0, y0) along the axis
    (0: x, 1: y), the run of the line inside the polygon that contains
    (x0, y0); 0 if the point is outside."""

    V = as_polygon(P)
    u, v = (V[:, 0], V[:, 1]) if axis == 0 else (V[:, 1], V[:, 0])
    u0, v0 = (x0, y0) if axis == 0 else (y0, x0)
    hits = []
    for i in range(len(V)):
        a_u, a_v, b_u, b_v = u[i], v[i], u[(i + 1) % len(V)], v[(i + 1) % len(V)]
        if (a_v <= v0 < b_v) or (b_v <= v0 < a_v):
            hits.append(a_u + (v0 - a_v)*(b_u - a_u)/(b_v - a_v))
    hits = np.sort(hits)
    for lo, hi in zip(hits[::2], hits[1::2]):
        if lo <= u0 <= hi:
            return float(hi - lo)

    return 0.0


def _reference_feature(P):
    """(x0, y0, w_x, w_y): center of the bounding box of the polygon and
    the chords through it (the bounding box sizes if the center is outside)."""

    V = as_polygon(P)
    lo, hi = V.min(axis=0), V.max(axis=0)
    x0, y0 = (lo + hi)/2
    w_x, w_y = _chord(V, x0, y0, 0), _chord(V, x0, y0, 1)
    if w_x == 0 or w_y == 0:
        w_x, w_y = hi - lo

    return (float(x0), float(y0), float(w_x), float(w_y))


def _check_overlaps(polygons, pitch_x, pitch_y, n=256):
    """Raise if polygons overlap each other or their periodic copies, tested
    on an n x n grid of points in one period (overlaps thinner than a grid
    step can go unnoticed)."""

    x = -pitch_x/2 + (np.arange(n) + 0.5)*pitch_x/n
    y = -pitch_y/2 + (np.arange(n) + 0.5)*pitch_y/n
    count = np.zeros((n, n), dtype=int)
    for P in polygons:
        lo, hi = P.min(axis=0), P.max(axis=0)
        # grid indices (unwrapped) covering the bounding box
        i = np.arange(int(np.floor((lo[0] + pitch_x/2)*n/pitch_x - 0.5)),
                      int(np.ceil((hi[0] + pitch_x/2)*n/pitch_x - 0.5)) + 1)
        j = np.arange(int(np.floor((lo[1] + pitch_y/2)*n/pitch_y - 0.5)),
                      int(np.ceil((hi[1] + pitch_y/2)*n/pitch_y - 0.5)) + 1)
        xs = -pitch_x/2 + (i + 0.5)*pitch_x/n
        ys = -pitch_y/2 + (j + 0.5)*pitch_y/n
        inside = contains(P, xs[None, :], ys[:, None]).astype(int)
        np.add.at(count, (np.mod(j, n)[:, None], np.mod(i, n)[None, :]), inside)
    if count.max() > 1:
        raise ValueError("polygons overlap (each other or their periodic copies); merge them "
                         "first, e.g. with gdstk.boolean(polygons, [], 'or')")


def polygon_rectangles(polygons):
    """The union of Manhattan polygons as non-overlapping rectangles (x0,
    y0, w_x, w_y) (exact; cells of the grid of all vertex coordinates,
    merged along x and then along y)."""

    polys = [_check_polygon(P) for P in polygons]
    if not polys:
        return []
    xs = np.unique(np.concatenate([P[:, 0] for P in polys]))
    ys = np.unique(np.concatenate([P[:, 1] for P in polys]))
    XC, YC = np.meshgrid((xs[:-1] + xs[1:])/2, (ys[:-1] + ys[1:])/2)
    inside = np.zeros(XC.shape, dtype=bool)
    for P in polys:  # even-odd rule with the vertical edges
        crossings = np.zeros(XC.shape, dtype=int)
        Q = np.roll(P, -1, axis=0)
        for (x_a, y_a), (x_b, y_b) in zip(P, Q):
            if x_a == x_b and y_a != y_b:
                crossings += (x_a > XC) & (YC > min(y_a, y_b)) & (YC < max(y_a, y_b))
        inside |= crossings % 2 == 1
    # runs along x in every row, then merge equal runs of consecutive rows
    rects, open_runs = [], {}
    for j in range(len(ys) - 1):
        runs, i = [], 0
        row = inside[j]
        while i < len(row):
            if row[i]:
                k = i
                while k + 1 < len(row) and row[k + 1]:
                    k += 1
                runs.append((xs[i], xs[k + 1]))
                i = k + 1
            else:
                i += 1
        new_open = {}
        for run in runs:
            y0 = open_runs.pop(run, ys[j])
            new_open[run] = y0
        for (x_a, x_b), y0 in open_runs.items():  # runs that ended below row j
            rects.append(((x_a + x_b)/2, (y0 + ys[j])/2, x_b - x_a, ys[j] - y0))
        open_runs = new_open
    for (x_a, x_b), y0 in open_runs.items():
        rects.append(((x_a + x_b)/2, (y0 + ys[-1])/2, x_b - x_a, ys[-1] - y0))

    return [tuple(float(v) for v in r) for r in rects]


def _rect_coeffs(m, P, center, size):
    """1D Fourier coefficients of a rect of width size at center on the
    period P; P None: invariant direction (only order 0, coefficient 1)."""

    m = np.asarray(m)
    if P is None:
        return np.ones(m.shape, dtype=complex)

    return size/P*np.sinc(m*size/P)*np.exp(-2j*np.pi*m*center/P)


class MaskLayout(object):
    """Geometry of one period of a mask.

    Parameters
    ----------
    pitch_x, pitch_y : numbers or None
        Periods; None for an invariant direction (at most one).
    features : sequence of (x0, y0, w_x, w_y)
        Centers and sizes of rectangles (must not overlap). Sizes along an
        invariant direction are ignored.
    tone : "clear" or "dark"
        Clear field (features are absorber) or dark field (features are
        clear).
    name : str, optional
    """

    def __init__(self, pitch_x, pitch_y=None, features=(), tone="clear", name=None):

        if pitch_x is None and pitch_y is None:
            raise ValueError("At least one direction must be periodic")
        if tone not in ("clear", "dark"):
            raise ValueError("tone must be 'clear' or 'dark'")
        self.pitch_x, self.pitch_y = pitch_x, pitch_y
        self.features = [tuple(float(v) for v in f) for f in features]
        self.tone = tone
        self.name = name
        self.polygons = None
        self.manhattan = True

    @classmethod
    def from_polygons(cls, pitch_x, pitch_y, polygons, tone="clear", name=None,
                      check_overlaps=True):
        """Layout of polygons in one period.

        If all edges are parallel to x or y (Manhattan polygons), the union
        of the polygons is cut into non-overlapping rectangles; overlapping
        polygons are merged, and the first polygon is the feature that
        cut_coordinates() and the CD metrics use (through its first
        rectangle). Otherwise the polygons are kept and their Fourier
        coefficients are computed with the exact polygon transform; they
        must not overlap, and features[0] is the reference rectangle of the
        first polygon (center of its bounding box, chords along x and y
        through it).

        The polygons should lie within one period of pitch_x x pitch_y (they
        are repeated periodically).

        Parameters
        ----------
        pitch_x, pitch_y : numbers
            Periods; both directions must be periodic.
        polygons : sequence of (N, 2) arrays
            Vertices (x, y) of each polygon.
        tone : {"clear", "dark"}, optional
            Clear field (polygons are absorber) or dark field (polygons are clear).
        name : str, optional
        check_overlaps : bool, optional
            Test non-Manhattan polygons for overlaps (on a grid of 256 x
            256 points per period).

        Returns
        -------
        MaskLayout
        """

        if pitch_x is None or pitch_y is None:
            raise ValueError("polygon layouts must be periodic in x and y")
        polygons = [as_polygon(P) for P in polygons]
        if not all(is_manhattan(P) for P in polygons):
            polys = [counterclockwise(P, manhattan=False) for P in polygons]
            if check_overlaps:
                _check_overlaps(polys, pitch_x, pitch_y)
            layout = cls(pitch_x, pitch_y, [_reference_feature(polys[0])], tone,
                         name or "polygons")
            layout.polygons = polys
            layout.manhattan = False
            return layout
        polys = [counterclockwise(P) for P in polygons]
        rects = polygon_rectangles(polys)
        if rects:  # the rectangle holding the center of the first polygon's first rectangle first
            x0, y0 = polygon_rectangles(polys[:1])[0][:2]
            i = next(i for i, (x, y, a, b) in enumerate(rects)
                     if abs(x0 - x) <= a/2 and abs(y0 - y) <= b/2)
            rects.insert(0, rects.pop(i))
        layout = cls(pitch_x, pitch_y, rects, tone, name or "polygons")
        layout.polygons = polys

        return layout

    @classmethod
    def from_gds(cls, source, window, layer=None, datatype=None, tone="clear", scale=1.0,
                 name=None, check_overlaps=True):
        """Layout from the polygons of a layout library (gdstk, gdspy, ...).

        The window is one period of the simulated mask: the polygons are
        clipped to it and the window center becomes the origin. A cell of
        a periodic array (SRAM, contact arrays) is simulated exactly with
        the window equal to the cell; for a clip from a random layout, the
        periodic copies are at least the window size away (choose a window
        that is a few lambda/NA larger than the region of interest).

        Parameters
        ----------
        source : layout object
            See `gds_polygons()`.
        window : (x0, y0, x1, y1)
            Window in the units of the source.
        layer, datatype : int or sequence of ints, optional
            Polygons of these layers and datatypes (default: all).
        tone : {"clear", "dark"}, optional
            Clear field (the polygons are absorber, e.g. a gate or metal
            layer printed in positive resist) or dark field (the polygons
            are clear, e.g. contacts and vias, or trenches).
        scale : number, optional
            Length unit of the source in the units of the simulation, e.g.
            1000 for a layout in um and a simulation in nm. Layout
            coordinates are taken at wafer scale (1x), as drawn.
        name : str, optional
        check_overlaps : bool, optional
            See `from_polygons()`.

        Returns
        -------
        MaskLayout
        """

        x0, y0, x1, y1 = (float(v) for v in window)
        if not (x1 > x0 and y1 > y0):
            raise ValueError("window must be (x0, y0, x1, y1) with x1 > x0 and y1 > y0")
        clipped = []
        for P in gds_polygons(source, layer, datatype):
            C = clip_to_rectangle(P, x0, y0, x1, y1)
            if C is not None:
                clipped.append((C - [(x0 + x1)/2, (y0 + y1)/2])*scale)
        if not clipped:
            raise ValueError("no polygons in the window")

        return cls.from_polygons((x1 - x0)*scale, (y1 - y0)*scale, clipped, tone,
                                 name or "layout clip", check_overlaps)

    # common layouts:

    @classmethod
    def lines_spaces(cls, pitch, cd, orientation="y", tone="clear", name=None):
        """Lines and spaces.

        Parameters
        ----------
        pitch : number
            Period.
        cd : number
            Width of the lines (absorber for tone="clear", clear trenches for
            tone="dark").
        orientation : {"y", "x"}, optional
            Direction of the lines: along y (pitch along x) or along x.
        tone : {"clear", "dark"}, optional
        name : str, optional

        Returns
        -------
        MaskLayout
        """

        name = name or "L/S {:g}/{:g}".format(cd, pitch - cd)
        if orientation == "y":
            return cls(pitch, None, [(0.0, 0.0, cd, 0.0)], tone, name)
        if orientation == "x":
            return cls(None, pitch, [(0.0, 0.0, 0.0, cd)], tone, name)
        raise ValueError("orientation must be 'x' or 'y'")

    @classmethod
    def iso_line(cls, cd, pitch=None, orientation="y", tone="clear", name=None):
        """Isolated line (or trench for tone="dark").

        Parameters
        ----------
        cd : number
            Line width.
        pitch : number, optional
            Period of the repeated line, default 20*cd.
        orientation : {"y", "x"}, optional
            Direction of the line.
        tone : {"clear", "dark"}, optional
        name : str, optional

        Returns
        -------
        MaskLayout
        """

        return cls.lines_spaces(20*cd if pitch is None else pitch, cd, orientation, tone,
                                name or "iso line {:g}".format(cd))

    @classmethod
    def contact_holes(cls, pitch, cd, pitch_y=None, cd_y=None, tone="dark", name=None):
        """Array of rectangular contact holes.

        Dark field (default): clear holes in absorber.

        Parameters
        ----------
        pitch : number
            Period in x.
        cd : number
            Hole width in x.
        pitch_y, cd_y : numbers, optional
            Period and hole width in y (defaults: square holes on a square grid).
        tone : {"dark", "clear"}, optional
        name : str, optional

        Returns
        -------
        MaskLayout
        """

        pitch_y = pitch if pitch_y is None else pitch_y
        cd_y = cd if cd_y is None else cd_y
        name = name or "contacts {:g} @ {:g}".format(cd, pitch)

        return cls(pitch, pitch_y, [(0.0, 0.0, cd, cd_y)], tone, name)

    @classmethod
    def iso_contact(cls, cd, pitch=None, cd_y=None, tone="dark", name=None):
        """Isolated contact hole.

        Parameters
        ----------
        cd : number
            Hole width in x.
        pitch : number, optional
            Period of the repeated hole, default 10*cd.
        cd_y : number, optional
            Hole width in y, default cd.
        tone : {"dark", "clear"}, optional
        name : str, optional

        Returns
        -------
        MaskLayout
        """

        return cls.contact_holes(10*cd if pitch is None else pitch, cd, None, cd_y, tone,
                                 name or "iso contact {:g}".format(cd))

    @classmethod
    def staggered_contacts(cls, pitch_x, pitch_y, cd, cd_y=None, tone="dark", name=None):
        """Staggered (brick) contact array.

        Rows at the distance pitch_y, every other row shifted by pitch_x/2.

        Parameters
        ----------
        pitch_x : number
            Period within a row.
        pitch_y : number
            Distance of the rows.
        cd : number
            Hole width in x.
        cd_y : number, optional
            Hole width in y, default cd.
        tone : {"dark", "clear"}, optional
        name : str, optional

        Returns
        -------
        MaskLayout
        """

        cd_y = cd if cd_y is None else cd_y

        return cls(pitch_x, 2*pitch_y, [(0.0, -pitch_y/2, cd, cd_y), (pitch_x/2, pitch_y/2, cd, cd_y)],
                   tone, name or "staggered contacts {:g}".format(cd))

    # properties:

    @property
    def is_1d(self):
        return self.pitch_x is None or self.pitch_y is None

    @property
    def feature_is_bright(self):
        """True if the features transmit (dark field mask)."""

        return self.tone == "dark"

    def scaled(self, factor_cd=1.0, bias=0.0):
        """Copy with scaled and biased features.

        Parameters
        ----------
        factor_cd : number, optional
            Factor for all feature sizes (polygon layouts: only 1).
        bias : number, optional
            Added to every feature size (each edge moves by bias/2), e.g. a mask
            CD error for MEEF.

        Returns
        -------
        MaskLayout
        """

        if self.polygons is not None:
            if factor_cd != 1.0:
                raise ValueError("polygon layouts can only be biased (factor_cd = 1)")
            return MaskLayout.from_polygons(self.pitch_x, self.pitch_y,
                                            [offset_polygon(P, bias/2) for P in self.polygons],
                                            self.tone, self.name, check_overlaps=False)
        feats = [(x0, y0, w*factor_cd + bias, h*factor_cd + bias) for x0, y0, w, h in self.features]

        return MaskLayout(self.pitch_x, self.pitch_y, feats, self.tone, self.name)

    # diffraction:

    def max_orders(self, NA, wavelength, sigma_max=1.0):
        """Largest diffraction order indices that can pass the pupil.

        Parameters
        ----------
        NA : number
            Numerical aperture (image side).
        wavelength : number
            Vacuum wavelength.
        sigma_max : number, optional
            Largest source point radius (partial coherence).

        Returns
        -------
        M, N : int
            Largest order indices in x and y (0 for an invariant direction).
        """

        f_max = NA*(1 + sigma_max)/wavelength
        M = 0 if self.pitch_x is None else int(np.floor(self.pitch_x*f_max)) + 1
        N = 0 if self.pitch_y is None else int(np.floor(self.pitch_y*f_max)) + 1

        return M, N

    def region_coefficients(self, m, n, boundary_width=0.0):
        """Fourier coefficients of the regions of the layout.

        Parameters
        ----------
        m, n : arrays of ints
            Order indices along x and y (1D each).
        boundary_width : number, optional
            Width of the boundary layer strips.

        Returns
        -------
        c : dict of arrays of shape (len(n), len(m))
            "field", "feature" and, for boundary_width > 0, "edge_y" (strips
            along edges parallel to y) and "edge_x" (edges parallel to x).
            The coefficients of all regions add up to the delta of order 0.
        """

        m, n = np.atleast_1d(m), np.atleast_1d(n)
        Px, Py, w = self.pitch_x, self.pitch_y, boundary_width
        if w > 0 and self.polygons is not None:
            raise ValueError("the boundary layer model is not available for polygon layouts")
        shape = (len(n), len(m))

        def rect(x0, y0, a, b):
            return np.outer(_rect_coeffs(n, Py, y0, b), _rect_coeffs(m, Px, x0, a))

        if not self.manhattan:  # exact transforms of the polygons at the reciprocal lattice
            k_x, k_y = 2*np.pi*m[None, :]/Px, 2*np.pi*n[:, None]/Py
            F = np.zeros(shape, dtype=complex)
            for P in self.polygons:
                F += polygon_ft(P, k_x, k_y)
            F /= Px*Py
            return {"feature": F, "field": np.outer(n == 0, m == 0).astype(complex) - F}
        c = {"feature": np.zeros(shape, dtype=complex)}
        outer = np.zeros(shape, dtype=complex)
        if w > 0:
            c["edge_y"] = np.zeros(shape, dtype=complex)
            c["edge_x"] = np.zeros(shape, dtype=complex)
        for x0, y0, a, b in self.features:
            if w <= 0:
                c["feature"] += rect(x0, y0, a, b)
                outer += rect(x0, y0, a, b)
                continue
            if (Px is not None and a <= w) or (Py is not None and b <= w):
                raise ValueError("Feature smaller than the boundary layer width")
            c["feature"] += rect(x0, y0, a - w, b - w)
            outer += rect(x0, y0, a + w, b + w)
            if Px is not None:  # edges parallel to y at x0 -+ a/2, including the corners
                for sgn in (-1, 1):
                    c["edge_y"] += rect(x0 + sgn*a/2, y0, w, b + w)
            if Py is not None:  # edges parallel to x, between the corners
                for sgn in (-1, 1):
                    c["edge_x"] += rect(x0, y0 + sgn*b/2, a - w, w)

        delta = np.outer(n == 0, m == 0).astype(complex)
        c["field"] = delta - outer

        return c

    def transmission(self, x, y=None, t_clear=1.0, t_absorber=0.0):
        """Sample the scalar thin mask transmission, e.g. for plots.

        Parameters
        ----------
        x : array
            x coordinates.
        y : array, optional
            y coordinates; None gives a cut at y = 0.
        t_clear, t_absorber : complex, optional
            Transmissions of the clear area and the absorber.

        Returns
        -------
        t : array
            Shape (len(y), len(x)), or (len(x),) for y=None.
        """

        X, Y = np.meshgrid(np.atleast_1d(x), np.atleast_1d(0.0 if y is None else y))
        inside = self.inside(X, Y)
        t_feat, t_field = (t_clear, t_absorber) if self.tone == "dark" else (t_absorber, t_clear)
        t = np.where(inside, t_feat, t_field)

        return t[0] if y is None else t

    def inside(self, x, y):
        """Points inside the features (periodically repeated).

        Parameters
        ----------
        x, y : arrays
            Coordinates, broadcast against each other.

        Returns
        -------
        inside : bool array
        """

        X, Y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if not self.manhattan:
            Px, Py = self.pitch_x, self.pitch_y
            Xw = (X + Px/2) % Px - Px/2  # into the period around the origin
            Yw = (Y + Py/2) % Py - Py/2
            inside = np.zeros(X.shape, dtype=bool)
            for P in self.polygons:  # with the copies of the polygon in the neighbouring periods
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        inside |= contains(P + [i*Px, j*Py], Xw, Yw)
            return inside
        inside = np.zeros(X.shape, dtype=bool)
        for x0, y0, a, b in self.features:
            in_x = True if self.pitch_x is None else (
                np.abs((X - x0 + self.pitch_x/2) % self.pitch_x - self.pitch_x/2) < a/2)
            in_y = True if self.pitch_y is None else (
                np.abs((Y - y0 + self.pitch_y/2) % self.pitch_y - self.pitch_y/2) < b/2)
            inside |= in_x & in_y

        return inside

    def __repr__(self):
        return "MaskLayout({}, pitch=({}, {}), tone={})".format(self.name, self.pitch_x,
                                                                 self.pitch_y, self.tone)


def _layer_match(value, wanted):
    if wanted is None:
        return True
    return int(value) in [int(w) for w in np.atleast_1d(wanted)]


def gds_polygons(source, layer=None, datatype=None):
    """Polygons from the objects of layout libraries, by duck typing (no
    layout library is imported).

    Understood are:

    - gdstk: a Cell (its get_polygons(), with references flattened), a
      Polygon or a list of Polygons (attributes points, layer, datatype);
      also gdstk.FlexPath and RobustPath through their to_polygons().
    - gdspy: a Cell or CellReference (get_polygons(by_spec=True)), a
      PolygonSet such as Rectangle or Polygon (attributes polygons, layers,
      datatypes).
    - A dict {(layer, datatype): [polygons]} as returned by gdspy's
      get_polygons(by_spec=True), or {layer: [polygons]}.
    - An (N, 2) array (one polygon) or a sequence of polygons; these have
      no layer and are always selected.

    Parameters
    ----------
    source : layout object
        See above.
    layer, datatype : int or sequence of ints, optional
        Select these layers and datatypes (default: all).

    Returns
    -------
    polygons : list of (N, 2) arrays
    """

    out = []

    def add(points, lay=None, dt=None):
        if lay is not None and not _layer_match(lay, layer):
            return
        if dt is not None and not _layer_match(dt, datatype):
            return
        out.append(as_polygon(points))

    def walk(obj):
        if isinstance(obj, dict):
            for key, polys in obj.items():
                lay, dt = (key if isinstance(key, tuple) else (key, None))
                for P in polys:
                    add(getattr(P, "points", P), lay, dt)
        elif hasattr(obj, "points") and hasattr(obj, "layer"):  # gdstk.Polygon
            add(obj.points, obj.layer, obj.datatype)
        elif hasattr(obj, "polygons") and hasattr(obj, "layers"):  # gdspy.PolygonSet
            for P, lay, dt in zip(obj.polygons, obj.layers, obj.datatypes):
                add(P, lay, dt)
        elif hasattr(obj, "to_polygons"):  # gdstk paths
            walk(obj.to_polygons())
        elif hasattr(obj, "get_polygons"):  # cells (gdspy: by_spec dict; gdstk: Polygon list)
            try:
                walk(obj.get_polygons(by_spec=True))
            except TypeError:
                walk(obj.get_polygons())
        else:
            try:
                V = np.asarray(obj, dtype=float)
            except (ValueError, TypeError):  # ragged: a sequence of polygons or layout objects
                V = None
            if V is not None and V.ndim == 2 and V.shape[1] == 2:
                add(V)
            elif V is not None and V.ndim == 3 and V.shape[2] == 2:
                for P in V:
                    add(P)
            elif V is not None and (V.ndim < 2 or V.size == 0):
                raise ValueError("not a polygon or layout object: {!r}".format(obj))
            else:
                for item in obj:
                    walk(item)

    walk(source)

    return out


# mask stacks:

class BoundaryLayer(object):
    """Boundary layer at the absorber edges.

    Parameters
    ----------
    width : number
        Width of the strip centered on each edge (wafer scale).
    t_TE, t_TM : complex numbers
        Transmission of the strip for the electric field parallel (TE) and
        perpendicular (TM) to the edge.
    """

    def __init__(self, width, t_TE, t_TM):
        self.width, self.t_TE, self.t_TM = width, complex(t_TE), complex(t_TM)

    def jones(self):
        """Jones matrices of the strips along y and along x."""

        return np.diag([self.t_TM, self.t_TE]), np.diag([self.t_TE, self.t_TM])


class ThinMask(object):
    """Kirchhoff (thin mask) model with given transmissions.

    Parameters
    ----------
    t_absorber : complex, optional
        Amplitude transmission of the absorber (default 0: binary mask).
    t_clear : complex, optional
    boundary_layer : BoundaryLayer, optional
    name : str, optional
    """

    def __init__(self, t_absorber=0.0, t_clear=1.0, boundary_layer=None, name=None):
        self.t_absorber, self.t_clear = complex(t_absorber), complex(t_clear)
        self.boundary_layer = boundary_layer
        self.name = name or "thin mask t={:.3g}".format(self.t_absorber)

    @classmethod
    def binary(cls, **kwargs):
        """Binary (chrome on glass) mask."""

        kwargs.setdefault("name", "binary")

        return cls(0.0, **kwargs)

    @classmethod
    def att_psm(cls, transmission=0.06, phase=180.0, **kwargs):
        """Attenuated phase shift mask.

        Parameters
        ----------
        transmission : number, optional
            Intensity transmission of the absorber.
        phase : number, optional
            Phase of the absorber in degrees.
        **kwargs
            Further arguments of `ThinMask` (e.g. boundary_layer, name).

        Returns
        -------
        ThinMask
        """

        kwargs.setdefault("name", "att. PSM {:g} %".format(100*transmission))

        return cls(np.sqrt(transmission)*np.exp(1j*np.radians(phase)), **kwargs)

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the clear area and the absorber for source points.

        Parameters
        ----------
        s_x, s_y : numbers or arrays
            Source point coordinates (sigma).
        NA : number
            Numerical aperture (unused by the thin mask).
        wavelength : number
            Vacuum wavelength (unused by the thin mask).
        magnification : number
            Reduction ratio of the projector (unused by the thin mask).

        Returns
        -------
        J_clear, J_absorber : arrays
            Shape np.shape(s_x) + (2, 2).
        """

        eye = np.broadcast_to(np.eye(2), np.shape(s_x) + (2, 2))

        return self.t_clear*eye, self.t_absorber*eye


class FilmMask(object):
    """Mask whose absorber is a thin film stack on the mask blank.

    Light comes from the blank (substrate_n), passes the absorber layers in
    the given order and leaves into the exit medium (air). The clear region
    is the bare blank; its transmission is taken at the same plane as the
    absorber's (bottom of the absorber), i.e. through exit medium of the same
    thickness.

    Parameters
    ----------
    n, k, t : sequences
        Refractive indices n + i*k and thicknesses of the absorber layers,
        from the blank outwards (mask scale thicknesses).
    substrate_n, substrate_k : numbers, optional
        Mask blank (fused silica at 193 nm: about 1.563).
    exit_n : number, optional
        Medium between mask and projection lens.
    normalize : bool, optional
        Divide by the clear transmission (per TE/TM), so that the clear
        region transmits 1.
    boundary_layer : BoundaryLayer, optional
    name : str, optional
    """

    def __init__(self, n, k, t, substrate_n=1.563, substrate_k=0.0, exit_n=1.0, normalize=True,
                 boundary_layer=None, name=None):
        self.n, self.k, self.t = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (n, k, t))
        if not len(self.n) == len(self.k) == len(self.t):
            raise ValueError("n, k and t must have the same length")
        self.substrate = substrate_n + 1j*substrate_k
        self.exit_n = exit_n
        self.normalize = normalize
        self.boundary_layer = boundary_layer
        self.name = name or "film mask"

    def transmissions(self, sin_theta, wavelength):
        """Transmissions of absorber and clear area for oblique incidence.

        Parameters
        ----------
        sin_theta : number or array
            Sine of the angle of incidence in the blank (substrate).
        wavelength : number
            Vacuum wavelength in the unit of the layer thicknesses.

        Returns
        -------
        t_abs, t_clear : arrays of shape (2,) + np.shape(sin_theta)
            [t_TE, t_TM] each (ratios of tangential electric fields).
        """

        sin_theta = np.asarray(sin_theta, dtype=float)
        n_abs = np.concatenate([[self.substrate], self.n + 1j*self.k, [self.exit_n]])
        n_clear = np.array([self.substrate, self.exit_n, self.exit_n], dtype=complex)
        t_abs = np.array([_r_t(sin_theta, n_abs, self.t, pol, wavelength)[1] for pol in ("TE", "TM")])
        t_clear = np.array([_r_t(sin_theta, n_clear, [self.t.sum()], pol, wavelength)[1]
                            for pol in ("TE", "TM")])

        return t_abs, t_clear

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the clear area and the absorber for source points.

        The angle of incidence at the mask follows from NA*sigma/magnification
        in the blank; TE and TM transmissions are combined on the TE/TM basis of
        each source point.

        Parameters
        ----------
        s_x, s_y : numbers or arrays
            Source point coordinates (sigma).
        NA : number
            Numerical aperture (image side).
        wavelength : number
            Vacuum wavelength.
        magnification : number
            Reduction ratio of the projector.

        Returns
        -------
        J_clear, J_absorber : arrays
            Shape np.shape(s_x) + (2, 2).
        """

        sin_theta = NA*np.hypot(s_x, s_y)/(magnification*np.real(self.substrate))
        t_abs, t_clear = self.transmissions(sin_theta, wavelength)
        if self.normalize:
            t_abs, t_clear = t_abs/t_clear, np.ones_like(t_clear)
        B = np.array(TE_TM_basis(s_x, s_y))  # B[0]: TE, B[1]: TM Jones vector

        def to_xy(t):  # sum over TE and TM of t*e e^T
            return np.einsum("ki...,k...,kj...->...ij", B, t, B)

        return to_xy(t_clear), to_xy(t_abs)
