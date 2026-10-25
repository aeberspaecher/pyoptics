#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical proximity correction (OPC) of Manhattan layouts.

The printed shape of a feature depends on its neighbourhood (pitch, line
ends, corners): OPC changes the mask so that the printed edges land on the
target edges (the design).

Rule-based OPC
--------------
bias_table() tabulates, through pitch, the mask bias that prints the target
CD of lines (or trenches) at one common dose, anchored on one pitch; the
bias is then looked up for every feature. insert_srafs() adds sub-resolution
assist features (scattering bars) to isolated or semi-dense lines, which
make them image more like dense lines and move the process window of the
forbidden pitches; sraf_margin() checks that the assists do not print.

Model-based OPC
---------------
The edges of the target polygons are cut into fragments (FragmentedLayout:
short fragments at the corners, the rest of an edge in pieces of at most
max_length). Every fragment has a site (its midpoint on the target edge,
on corner fragments nearer to the fragment's inner end) where the edge
placement error (EPE) is measured: the distance from the target edge to the
printed edge (threshold model) along the outward normal, positive if the
printed feature is too large. model_opc() moves the fragments along their
normals and re-images until the EPEs are small: either every fragment
against its own EPE ("local", the simplest scheme) or all fragments
together from the Jacobian of the EPEs (a MEEF matrix) by regularized least
squares ("jacobian", the default). Line ends grow hammerheads, convex
corners serifs and concave corners are cut back, as in the classic picture;
the rounding of the printed corners cannot be corrected, so the corner
sites get less weight. The image of every iteration is computed on a grid
of one period, with Abbe's method or, faster, with the Hopkins TCC (or its
SOCS kernels) of the target layout, and interpolated along the normals.

Only Manhattan polygons are supported, no curvilinear masks, and the
threshold model stands in for the resist.
"""

from collections import namedtuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq

from .mask import MaskLayout, counterclockwise
from . import hopkins


# rule-based OPC:

def _lines(layout, pitch, cd):
    """Lines (or trenches) like the 1D layout, on another pitch and CD."""

    orientation = "y" if layout.pitch_y is None else "x"

    return MaskLayout.lines_spaces(pitch, cd, orientation, layout.tone)


def bias_table(sim, pitches, cd_target, threshold=0.3, anchor=None, focus=0.0, max_bias=None,
               make_layout=None):
    """Mask bias of lines through pitch that prints the target CD at one
    dose (rule-based OPC).

    Parameters
    ----------
    sim : LithoSimulation
        Stack, illumination and projector; its layout (1D lines or
        trenches) gives orientation and tone.
    pitches : sequence of numbers
    cd_target : number
        Target CD on the wafer.
    threshold : number, optional
        Resist threshold (intensity relative to the open frame).
    anchor : number, optional
        Pitch that prints at cd_target without bias and so sets the dose
        (default: the first pitch).
    focus : number, optional
    max_bias : number, optional
        Largest absolute bias searched (default: cd_target).
    make_layout : callable, optional
        make_layout(pitch, cd) gives the layout of a pitch with the mask CD
        of the (first) line, e.g. with SRAFs (insert_srafs()). Default:
        lines and spaces with the orientation and tone of sim.layout.

    Returns
    -------
    bias : array
        Mask CD - target CD for every pitch (nan where no bias in the
        range prints the target).
    dose : number
        The common dose.
    """

    pitches = np.atleast_1d(np.asarray(pitches, dtype=float))
    anchor = pitches[0] if anchor is None else anchor
    max_bias = cd_target if max_bias is None else max_bias
    if make_layout is None:
        def make_layout(pitch, cd):
            return _lines(sim.layout, pitch, cd)
    dose = sim.with_(layout=make_layout(anchor, cd_target)).dose_to_size(cd_target, threshold, focus)

    def error(pitch, bias):  # nan where the line does not print (or merges)
        return sim.with_(layout=make_layout(pitch, cd_target + bias)).cd(threshold, dose,
                                                                         focus) - cd_target

    bias = np.full(len(pitches), np.nan)
    for i, pitch in enumerate(pitches):
        # bracket the root nearest to zero bias on a coarse grid, then refine
        grid = np.linspace(-max_bias, min(max_bias, pitch - cd_target - 1.0), 25)
        e = np.array([error(pitch, b) for b in grid])
        k = np.nonzero(np.isfinite(e[:-1]) & np.isfinite(e[1:]) & (np.sign(e[:-1]) != np.sign(e[1:])))[0]
        if len(k):
            k = k[np.argmin(np.abs(grid[k] + grid[k + 1]))]
            bias[i] = brentq(lambda b: error(pitch, b), grid[k], grid[k + 1], xtol=1e-3)

    return bias, dose


def insert_srafs(layout, width, distance, count=1, spacing=None, min_space=None):
    """Sub-resolution assist features next to the lines of a 1D layout.

    Parameters
    ----------
    layout : MaskLayout
        Lines (or trenches) along y or x with one feature per period.
    width : number
        Width of the assists (well below the resolution, so they do not
        print).
    distance : number
        Center to center distance of the first assist from the line.
    count : int, optional
        Assists on each side of the line.
    spacing : number, optional
        Center to center distance of further assists (default: distance).
    min_space : number, optional
        Smallest space between an assist and a line or another assist;
        assists from both sides of a space that come closer are merged into
        one in the middle (default: width).

    Returns
    -------
    MaskLayout
        The main line first, then the assists (same tone: absorber bars on
        a clear field, clear slits for a dark field).
    """

    if not layout.is_1d or len(layout.features) != 1:
        raise ValueError("SRAFs are inserted into 1D layouts with one line per period")
    spacing = distance if spacing is None else spacing
    min_space = width if min_space is None else min_space
    along_x = layout.pitch_y is None
    pitch = layout.pitch_x if along_x else layout.pitch_y
    x0, y0, a, b = layout.features[0]
    center, cd = (x0, a) if along_x else (y0, b)
    # positions in the space to the right of the line, from both of its lines
    pos = [distance + k*spacing for k in range(count)]
    pos = [p for p in pos if p + width/2 <= pitch - cd/2 - min_space and p - width/2 >= cd/2 + min_space]
    pos = sorted(pos + [pitch - p for p in pos])
    merged = []
    for p in pos:
        if merged and p - merged[-1][-1] < width + min_space:
            merged[-1].append(p)
        else:
            merged.append([p])
    pos = [np.mean(group) for group in merged]
    feats = [layout.features[0]]
    for p in pos:
        c = center + p
        if c > center + pitch/2:  # keep the assist in the period around the line
            c -= pitch
        feats.append((c, y0, width, b) if along_x else (x0, c, a, width))

    return MaskLayout(layout.pitch_x, layout.pitch_y, feats, layout.tone,
                      "{} + SRAF".format(layout.name))


def sraf_margin(sim, threshold=0.3, dose=1.0, foci=(0.0,)):
    """How far the assists (layout features after the first) are from
    printing: the smallest relative distance of the image at their centers
    from the threshold over the foci; positive: they do not print.
    """

    L = sim.layout
    along_x = L.pitch_y is None
    centers = [f[0] if along_x else f[1] for f in L.features[1:]]
    if not centers:
        return np.inf
    pts = np.asarray(centers)
    I = dose*(sim._images(pts, None, foci, 0.0)[0] if along_x else
              sim._images([L.features[0][0]], pts, foci, 0.0)[0][..., 0])
    margin = (threshold - I)/threshold if L.feature_is_bright else (I - threshold)/threshold

    return float(margin.min())


# fragments and edge placement errors:

class FragmentedLayout(object):
    """Target polygons with their edges cut into movable fragments.

    Parameters
    ----------
    layout : MaskLayout
        Target layout made with MaskLayout.from_polygons().
    max_length : number, optional
        Longest fragment.
    corner_length : number, optional
        Length of the fragments at the ends of an edge. Edges shorter than
        2*corner_length are one fragment, edges shorter than
        3*corner_length two.
    corner_site : number, optional
        Position of the site on the end fragments of edges with three or
        more fragments, as a fraction of the fragment length from the
        corner (0.5: the midpoint); away from the corner, where the
        rounding of the printed corner matters less.

    Attributes
    ----------
    start, end : arrays (n_fragments, 2)
        End points of the fragments on the target edges (counterclockwise).
    normal : array (n_fragments, 2)
        Outward unit normals.
    sites : array (n_fragments, 2)
        Midpoints, where the EPE is evaluated.
    """

    def __init__(self, layout, max_length=40.0, corner_length=15.0, corner_site=0.75):
        if layout.polygons is None:
            raise ValueError("the layout must be made with MaskLayout.from_polygons()")
        self.target = layout
        self.max_length, self.corner_length = max_length, corner_length
        self._edges = []  # per polygon, per edge: indices of its fragments
        start, end, normal, sites, corner = [], [], [], [], []
        c = corner_length
        for P in layout.polygons:
            edges = []
            for i in range(len(P)):
                a, b = P[i], P[(i + 1) % len(P)]
                L = np.linalg.norm(b - a)
                u = (b - a)/L
                if L < 2*c:
                    cuts = [0.0, L]
                elif L < 3*c:
                    cuts = [0.0, L/2, L]
                else:
                    k = int(np.ceil((L - 2*c)/max_length - 1e-9))
                    cuts = [0.0] + list(c + np.linspace(0, L - 2*c, k + 1)) + [L]
                edges.append(list(range(len(start), len(start) + len(cuts) - 1)))
                for j, (t0, t1) in enumerate(zip(cuts[:-1], cuts[1:])):
                    start.append(a + t0*u)
                    end.append(a + t1*u)
                    normal.append([u[1], -u[0]])
                    w = 0.5  # sites of corner fragments away from the corner
                    corner.append(len(cuts) > 3 and j in (0, len(cuts) - 2))
                    if corner[-1]:
                        w = corner_site if j == 0 else 1 - corner_site
                    sites.append(a + (t0 + w*(t1 - t0))*u)
            self._edges.append(edges)
        self.start, self.end = np.array(start), np.array(end)
        self.normal = np.array(normal)
        self.sites = np.array(sites)
        self.corner = np.array(corner, dtype=bool)
        # every site must be on the outline of the union (no shared edges)
        outside = self.sites + 1e-3*self.normal
        for x0, y0, a, b in layout.features:
            dx = (outside[:, 0] - x0 + layout.pitch_x/2) % layout.pitch_x - layout.pitch_x/2
            dy = (outside[:, 1] - y0 + layout.pitch_y/2) % layout.pitch_y - layout.pitch_y/2
            if np.any((np.abs(dx) < a/2) & (np.abs(dy) < b/2)):
                raise ValueError("the target polygons must not overlap or touch; draw them as one polygon")

    def __len__(self):
        return len(self.sites)

    @property
    def lengths(self):
        return np.linalg.norm(self.end - self.start, axis=1)

    def polygons(self, offsets):
        """The polygons with every fragment moved by its offset along its
        outward normal (jogs between the fragments of an edge, corners at
        the intersections of the moved end fragments)."""

        d = np.broadcast_to(np.asarray(offsets, dtype=float), (len(self),))
        # every moved fragment lies on the line u[a] = c (a: axis of its
        # normal); vertices are built from these exact values, so that the
        # edges stay parallel to x or y
        a = np.argmax(np.abs(self.normal), axis=1)
        c = self.start[np.arange(len(self)), a] + d*self.normal[np.arange(len(self)), a]

        def vertex(j, k):  # point on the lines of the perpendicular fragments j and k
            v = np.empty(2)
            v[a[j]], v[a[k]] = c[j], c[k]
            return v

        def jog(j, k):  # the two points at the cut between the fragments j, k of an edge
            v, w = self.end[j].copy(), self.end[j].copy()
            v[a[j]], w[a[k]] = c[j], c[k]
            return v, w

        out = []
        for edges in self._edges:
            V = []
            for i, frags in enumerate(edges):
                V.append(vertex(edges[i - 1][-1], frags[0]))
                for j, k in zip(frags[:-1], frags[1:]):
                    V.extend(jog(j, k))
            out.append(counterclockwise(V))

        return out

    def layout(self, offsets):
        """MaskLayout of the moved polygons."""

        T = self.target

        return MaskLayout.from_polygons(T.pitch_x, T.pitch_y, self.polygons(offsets), T.tone,
                                        T.name)


def image_grid(layout, spacing=2.0):
    """Grid of one period for images that edge_placement_errors()
    interpolates: returns x, y (cell [-pitch/2, pitch/2), periodic)."""

    def axis(P):
        n = int(np.ceil(P/spacing))
        return -P/2 + P*np.arange(n)/n

    return axis(layout.pitch_x), axis(layout.pitch_y)


def sample_image(I, x, y, points):
    """Periodic cubic spline interpolation of an image I (len(y), len(x))
    on the grid of image_grid() at points (..., 2)."""

    points = np.asarray(points, dtype=float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    coords = [(points[..., 1] - y[0])/dy, (points[..., 0] - x[0])/dx]

    return map_coordinates(I, coords, order=3, mode="grid-wrap")


def edge_placement_errors(I, x, y, sites, normals, threshold, bright=False, search=30.0, step=0.25):
    """Edge placement errors of the threshold contour of an image.

    Parameters
    ----------
    I : array (len(y), len(x))
        Image (times the dose) on the periodic grid of image_grid().
    x, y : arrays
    sites, normals : arrays (n, 2)
        Points on the target edges and outward unit normals.
    threshold : number
    bright : bool, optional
        The features are bright (dark field mask).
    search : number, optional
        Largest distance searched along the normals; an edge that is not
        found counts as -search (feature missing at the site) or +search.
    step : number, optional
        Sampling step along the normals.

    Returns
    -------
    array (n,)
        Distance of the printed edge from the site along the normal
        (positive: outside, the printed feature is larger).
    """

    s = np.arange(-search, search + step/2, step)
    pts = sites[:, None, :] + s[None, :, None]*normals[:, None, :]
    f = sample_image(I, x, y, pts) - threshold
    if bright:
        f = -f  # negative inside the feature, positive outside
    i0 = len(s)//2
    epe = np.empty(len(sites))
    for k, fk in enumerate(f):
        up = np.nonzero((fk[:-1] <= 0) & (fk[1:] > 0))[0]
        if len(up) == 0:
            epe[k] = -search if fk[i0] > 0 else search
            continue
        j = up[np.argmin(np.abs(up + 0.5 - i0))]
        epe[k] = s[j] + step*fk[j]/(fk[j] - fk[j + 1])

    return epe


# model-based OPC:

OPCResult = namedtuple("OPCResult", "layout fragments offsets epe history x y image")
OPCResult.__doc__ = """Result of model_opc().

layout : the corrected MaskLayout; fragments : FragmentedLayout of the
target; offsets, epe : arrays (n_fragments,); history : array (iterations
+ 1, 2) of the largest and the RMS ``abs(EPE)``, starting with the target
as the mask; x, y, image : the final image (times the dose) on the grid.
"""


def _imager(sim, target, imager, focus, energy):
    if imager not in ("abbe", "hopkins", "socs"):
        raise ValueError("imager must be 'abbe', 'hopkins' or 'socs'")
    if imager == "abbe":
        return lambda layout, x, y: sim.with_(layout=layout).image(x, y, focus)
    tcc = hopkins.TCC(sim.with_(layout=target), foci=[focus])
    if imager == "hopkins":
        return lambda layout, x, y: tcc.image(x, y, layout=layout)[0]
    socs = tcc.socs(energy=energy)

    return lambda layout, x, y: socs.image(x, y, layout=layout)[0]


def model_opc(sim, layout=None, threshold=0.3, dose=1.0, focus=0.0, method="jacobian", max_iter=20,
              damping=0.6, regularization=0.05, corner_weight=0.3, tol=0.25, max_offset=None,
              max_length=40.0, corner_length=15.0, corner_site=0.75, spacing=2.0, search=30.0,
              imager="hopkins", energy=0.999):
    """Model-based OPC: move the fragments of the target edges until the
    printed edges (threshold model) are on the target.

    Parameters
    ----------
    sim : LithoSimulation
        Stack, illumination, projector and wafer stack.
    layout : MaskLayout, optional
        Target made with MaskLayout.from_polygons() (default: the layout of
        sim).
    threshold, dose, focus : numbers, optional
        Threshold model at this dose and focus.
    method : "jacobian" or "local", optional
        "local": every fragment is moved against its own EPE, offset -=
        damping*EPE (fragments do not know about each other).
        "jacobian": the response of all EPEs to all fragment moves (the
        Jacobian, a MEEF matrix, by finite differences of the target) is
        computed once for the target, and the moves solve the regularized
        least squares problem ``min |W*(EPE + J*move)|**2 + mu*|offset +
        move|**2`` (W: weights of the sites, corner_weight for the corner
        fragments; mu: regularization times the mean of diag(J^T W**2 J)),
        damped by damping. The regularization keeps the mask close to the
        target where the EPEs hardly depend on the moves (e.g. alternating
        moves of short fragments).
        This handles the cross talk of short fragments (line ends, corners)
        and targets that cannot be reached everywhere (rounded corners).
        Updating the Jacobian on the way does not help: the EPEs near the
        corners are not smooth functions of the moves.
    max_iter : int, optional
    damping : number, optional
        Fraction of the computed move applied per iteration.
    regularization, corner_weight : numbers, optional
        See method.
    tol : number, optional
        Stop when every ``abs(EPE)`` is below tol.
    max_offset : number, optional
        Largest ``abs(offset)`` of a fragment (default: 2*corner_length).
    max_length, corner_length, corner_site : numbers, optional
        Fragmentation, see FragmentedLayout.
    spacing : number, optional
        Grid spacing of the images.
    search : number, optional
        Search range of the edges along the normals.
    imager : "abbe", "hopkins" or "socs", optional
        Abbe imaging of every mask, or the TCC of the target (exact for thin
        masks) or its SOCS kernels (energy fraction kept: energy).
    energy : number, optional

    Returns
    -------
    OPCResult
    """

    if method not in ("jacobian", "local"):
        raise ValueError("method must be 'jacobian' or 'local'")
    target = sim.layout if layout is None else layout
    frags = FragmentedLayout(target, max_length, corner_length, corner_site)
    max_offset = 2*corner_length if max_offset is None else max_offset
    x, y = image_grid(target, spacing)
    image = _imager(sim, target, imager, focus, energy)
    step = min(0.25, spacing/4)

    def epe_of(offsets):
        mask = frags.layout(offsets)
        I = dose*image(mask, x, y)
        return mask, I, edge_placement_errors(I, x, y, frags.sites, frags.normal, threshold,
                                              target.feature_is_bright, search, step)

    offsets = np.zeros(len(frags))
    history = []
    for it in range(max_iter + 1):
        mask, I, epe = epe_of(offsets)
        history.append((np.abs(epe).max(), np.sqrt(np.mean(epe**2))))
        if history[-1][0] < tol or it == max_iter:
            break
        if method == "local":
            move = -epe
        else:
            if it == 0:
                h = 1.0
                J = np.column_stack([(epe_of(h*np.eye(len(frags))[k])[2] - epe)/h
                                     for k in range(len(frags))])
                w2 = np.where(frags.corner, corner_weight, 1.0)**2
                JtJ = J.T @ (w2[:, None]*J)
                mu = regularization*np.mean(np.diag(JtJ))
                A = np.linalg.inv(JtJ + mu*np.eye(len(frags)))
            move = -A @ (J.T @ (w2*epe) + mu*offsets)
        offsets = np.clip(offsets + damping*move, -max_offset, max_offset)

    return OPCResult(mask, frags, offsets, epe, np.array(history), x, y, I)
