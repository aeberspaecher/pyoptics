#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Source optimization ("SO lite"): a pixelated source optimized for a set
of clips.

In Abbe's method the image is a sum over the source points, so it is linear
in their weights. The image of a source with weights W is the weighted sum of
the images of the single pixels (or, with a symmetry, of the symmetric groups
of pixels). SourceOptimizer computes these responses once, at the nominal
edges of every clip and for every focus, and then finds the weights by
linear programming:

- objective: the worst NILS over all clips and foci at the nominal feature
  edges, NILS = CD*slope/I with the image slope across the edge and the
  intensity I at the edge (the edge is printed at size when the threshold
  equals I, i.e. after a mask bias per clip, or at a common dose with
  dose_tolerance). Maximizing the minimum of these ratios is a
  quasi-concave problem, solved by bisection over the target value, each
  step a linear feasibility problem;
- constraints: non-negative weights, the symmetry of the source (pixel
  groups share a weight), a minimum pupil fill (an upper bound for the
  weight of a single pixel, as a scanner's illuminator cannot concentrate
  all light into a few pixels), and optionally a common dose: the
  intensities at the edges of all clips (their dose to size) agree within
  a tolerance.

Through focus the NILS at the nominal edges of the defocused images
measures the depth of focus. The result is an Illumination that works
everywhere else in the toolbox; SourceOptimizer.evaluate() rates any other
source (annular, dipole, quasar, ...) with the same metric.

This is source optimization only; the mask stays fixed apart from the
implicit bias per clip (no SMO, no OPC).
"""

from collections import namedtuple

import numpy as np

from .illumination import Illumination, _grid


class SourcePixels(object):
    """Pixels of a freeform source, grouped by a symmetry.

    Parameters
    ----------
    n_samples : int, optional
        Grid points across the unit circle (as for the standard shapes of
        Illumination, so that those lie on the same pixels).
    sigma_max : number, optional
        Largest sigma of the pixels.
    symmetry : {"none", "point", "xy", "xy-diagonal"}, optional
        Pixels that share a weight: none; s and -s (a symmetric source, no
        telecentricity error); mirror symmetry about the x and y axes;
        additionally about the diagonals (equal for x and y features).
    """

    _MAPS = {"none": [(1, 1, False)],
             "point": [(1, 1, False), (-1, -1, False)],
             "xy": [(sx, sy, False) for sx in (1, -1) for sy in (1, -1)],
             "xy-diagonal": [(sx, sy, swap) for sx in (1, -1) for sy in (1, -1) for swap in (False, True)]}

    def __init__(self, n_samples=25, sigma_max=1.0, symmetry="xy"):
        if symmetry not in self._MAPS:
            raise ValueError("symmetry must be one of {}".format(sorted(self._MAPS)))
        S_x, S_y = _grid(n_samples)
        keep = np.hypot(S_x, S_y) <= sigma_max + 1E-12
        self.s_x, self.s_y = S_x[keep], S_y[keep]
        self.n_samples, self.sigma_max, self.symmetry = n_samples, sigma_max, symmetry
        index = {(int(i), int(j)): k for k, (i, j) in enumerate(self._ij(self.s_x, self.s_y))}
        group_of = -np.ones(len(self.s_x), dtype=int)
        groups = []
        for k, (i, j) in enumerate(self._ij(self.s_x, self.s_y)):
            if group_of[k] >= 0:
                continue
            members = set()
            for sx, sy, swap in self._MAPS[symmetry]:
                a, b = (j, i) if swap else (i, j)
                members.add(index[(sx*a, sy*b)])
            members = np.array(sorted(members))
            group_of[members] = len(groups)
            groups.append(members)
        self.groups, self.group_of = groups, group_of

    def _ij(self, s_x, s_y):
        """Integer grid coordinates (symmetric about 0)."""

        step = 2.0/(self.n_samples - 1)
        return np.rint(np.column_stack([s_x, s_y])/step).astype(int)

    def __len__(self):
        return len(self.groups)

    @property
    def sizes(self):
        """Number of pixels in each group."""

        return np.array([len(g) for g in self.groups])

    def group_illumination(self, g, polarization="unpolarized", dop=1.0):
        """The pixels of group g with equal weights."""

        idx = self.groups[g]
        return Illumination(self.s_x[idx], self.s_y[idx], polarization=polarization, dop=dop,
                            name="group {}".format(g))

    def point_weights(self, W):
        """Weights of the single pixels for the group totals W."""

        return np.asarray(W, dtype=float)[self.group_of]/self.sizes[self.group_of]

    def illumination(self, W, polarization="unpolarized", dop=1.0, threshold=1E-6, name="freeform"):
        """Illumination with the group totals W; pixels below threshold times
        the largest pixel weight are dropped."""

        w = self.point_weights(W)
        keep = w > threshold*w.max()

        return Illumination(self.s_x[keep], self.s_y[keep], w[keep], polarization, dop, name)

    def fill(self, W, threshold=0.01):
        """Pupil fill: the fraction of the pupil area (sigma <= 1) that is
        lit with more than threshold times the largest pixel weight."""

        w = self.point_weights(W)
        n_disk = np.sum(np.hypot(*_grid(self.n_samples)) <= 1 + 1E-12)

        return np.sum(w > threshold*w.max())/n_disk


SourceResult = namedtuple("SourceResult", ["weights", "illumination", "nils", "worst", "fill"])
SourceResult.__doc__ = """Result of SourceOptimizer.optimize().

weights : group totals (sum 1) for SourcePixels.illumination()
illumination : the optimized Illumination
nils : NILS at the nominal edges, shape (clips, foci)
worst : its minimum (the objective)
fill : pupil fill of the result (SourcePixels.fill())
"""


def _edge(sim):
    """Cut direction, the positions of the nominal edge (outward from the
    feature center) and the sign that makes the slope across the edge
    positive for a dark feature in a bright surround and vice versa."""

    u, center, cd, direction = sim.cut_coordinates(3)
    sign = -1.0 if sim.layout.feature_is_bright else 1.0

    return direction, center + cd/2, cd, sign


def _edge_images(sim, foci, h):
    """Images at the nominal edge and at +-h across it, shape (foci, 3)."""

    direction, edge, cd, sign = _edge(sim)
    u = edge + np.array([-h, 0.0, h])
    x0, y0 = sim.layout.features[0][:2]
    if direction == "x":
        I = sim._images(u, None if sim.layout.pitch_y is None else [y0], foci, 0.0)[0]
    else:
        I = sim._images([x0], u, foci, 0.0)[0]

    return I.reshape(len(foci), 3)


class SourceOptimizer(object):
    """Pixelated source optimization for a set of clips.

    Parameters
    ----------
    clips : dict {name: LithoSimulation} or list
        The patterns to optimize for, each with its layout, mask stack and
        projector (and wafer stack); their illumination is ignored apart
        from its polarization. The first feature of each layout is
        evaluated, at its nominal edges along the cut direction
        (LithoSimulation.cut_coordinates()).
    pixels : SourcePixels
    foci : sequence, optional
        Foci at which the NILS counts (e.g. (0, +-dof/2)).
    polarization : str or array, optional
        Polarization of the source pixels (as for Illumination); default:
        that of the first clip's illumination.
    h : number, optional
        Step for the image slope across the edge.
    """

    def __init__(self, clips, pixels, foci=(0.0,), polarization=None, h=0.5):
        self.clips = dict(clips) if isinstance(clips, dict) else {
            "clip {}".format(i): c for i, c in enumerate(clips)}
        self.pixels = pixels
        self.foci = np.atleast_1d(np.asarray(foci, dtype=float))
        first = next(iter(self.clips.values()))
        self.polarization = first.illumination.polarization if polarization is None else polarization
        self.dop = first.illumination.dop if polarization is None else 1.0
        self.h = h
        self._responses = None

    @property
    def responses(self):
        """Per group (unit total weight, not normalized): intensity at the
        edge E and signed slope S, shape (clips, foci, groups), and the open
        frame O (groups,)."""

        if self._responses is None:
            G, K, F = len(self.pixels), len(self.clips), len(self.foci)
            E, S, O = np.zeros((K, F, G)), np.zeros((K, F, G)), np.zeros(G)
            for g in range(G):
                ill = self.pixels.group_illumination(g, self.polarization, self.dop)
                for k, sim in enumerate(self.clips.values()):
                    s = sim.with_(illumination=ill, normalize=False)
                    I = _edge_images(s, self.foci, self.h)
                    sign = _edge(s)[3]
                    E[k, :, g] = I[:, 1]
                    S[k, :, g] = sign*(I[:, 2] - I[:, 0])/(2*self.h)
                    if k == 0:
                        O[g] = s.open_frame()
            self._responses = E, S, O

        return self._responses

    @property
    def cds(self):
        return np.array([_edge(sim)[2] for sim in self.clips.values()])

    def nils(self, W):
        """NILS at the nominal edges for the group totals W, shape (clips,
        foci)."""

        E, S, _ = self.responses
        return self.cds[:, None]*(S @ W)/(E @ W)

    def edge_intensity(self, W):
        """Normalized intensity at the nominal edges (the threshold that
        prints each clip at size), shape (clips, foci)."""

        E, _, O = self.responses
        return (E @ W)/(O @ W)

    def _feasible(self, t, bounds, A_eq, b_eq):
        from scipy.optimize import linprog

        E, S, _ = self.responses
        A_ub = np.vstack([-(self.cds[:, None, None]*S - t*E).reshape(-1, E.shape[-1]), self._A_dose])
        res = linprog(np.zeros(E.shape[-1]), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq,
                      b_eq=b_eq, bounds=bounds, method="highs")

        return res.x if res.status == 0 else None

    def optimize(self, min_fill=0.1, dose_tolerance=None, tol=1E-3):
        """Weights that maximize the worst NILS over clips and foci.

        Parameters
        ----------
        min_fill : number, optional
            Minimum pupil fill: no pixel gets more than 1/(min_fill*N) of
            the light, N the number of grid points in the unit circle (0: no
            limit; the solution is then a few pixels).
        dose_tolerance : number, optional
            Common dose: the intensities at the edges of all clips (at the
            first focus) stay within (1 +- dose_tolerance) times that of the
            first clip, so that one dose prints all of them at size within
            about dose_tolerance/NILS relative CD. None (default): every
            clip is assumed to be biased to size on the mask.
        tol : number, optional
            Accuracy of the worst NILS (bisection).

        Returns
        -------
        SourceResult
        """

        E, S, O = self.responses
        sizes = self.pixels.sizes
        n_disk = np.sum(np.hypot(*_grid(self.pixels.n_samples)) <= 1 + 1E-12)
        cap = None if not min_fill else sizes/(min_fill*n_disk)
        if cap is not None and cap.sum() < 1:
            raise ValueError("min_fill too large for the pixels")
        bounds = [(0, None if cap is None else c) for c in (cap if cap is not None else sizes)]
        A_eq, b_eq = np.ones((1, len(sizes))), np.ones(1)
        A_dose = np.zeros((0, len(sizes)))
        if dose_tolerance is not None:  # (1 - tol) E_0 <= E_k <= (1 + tol) E_0
            d = dose_tolerance
            A_dose = np.array([row for k in range(1, len(E))
                               for row in (E[k, 0] - (1 + d)*E[0, 0], (1 - d)*E[0, 0] - E[k, 0])])
        self._A_dose = A_dose

        # bounds of the bisection: the best single group bounds every ratio
        ratio = self.cds[:, None, None]*S/np.where(E > 0, E, np.inf)
        hi = float(np.min(ratio.max(axis=-1)))
        x = self._feasible(0.0, bounds, A_eq, b_eq)
        if x is None:
            raise ValueError("no source fulfils the constraints (dose_tolerance, min_fill) "
                             "with positive image slopes at all edges")
        lo = float(np.min(self.nils(x)))
        while hi - lo > tol:
            t = 0.5*(lo + hi)
            y = self._feasible(t, bounds, A_eq, b_eq)
            if y is None:
                hi = t
            else:
                x, lo = y, max(t, float(np.min(self.nils(y))))
        W = np.clip(x, 0, None)
        W = W/W.sum()
        nils = self.nils(W)

        return SourceResult(W, self.pixels.illumination(W, self.polarization, self.dop), nils,
                            float(nils.min()), self.pixels.fill(W))

    def evaluate(self, illumination):
        """NILS at the nominal edges (clips, foci) for any Illumination,
        computed with the full simulation (normalized images)."""

        out = np.zeros((len(self.clips), len(self.foci)))
        for k, sim in enumerate(self.clips.values()):
            s = sim.with_(illumination=illumination)
            I = _edge_images(s, self.foci, self.h)
            sign, cd = _edge(s)[3], _edge(s)[2]
            out[k] = cd*sign*(I[:, 2] - I[:, 0])/(2*self.h)/I[:, 1]

        return out

    def project(self, illumination):
        """Group totals W of an Illumination whose points lie on the pixel
        grid (e.g. a standard shape with the same n_samples); points off the
        grid raise ValueError."""

        ij = self.pixels._ij(illumination.s_x, illumination.s_y)
        lookup = {tuple(p): k for k, p in enumerate(self.pixels._ij(self.pixels.s_x, self.pixels.s_y))}
        step = 2.0/(self.pixels.n_samples - 1)
        if not np.allclose(ij*step, np.column_stack([illumination.s_x, illumination.s_y])):
            raise ValueError("the source points are not on the pixel grid")
        W = np.zeros(len(self.pixels))
        for p, w in zip(map(tuple, ij), illumination.weights):
            W[self.pixels.group_of[lookup[p]]] += w

        return W


def scan(optimizer, make_illumination, parameters):
    """Rate a family of sources by the worst NILS of optimizer.evaluate().

    Parameters
    ----------
    optimizer : SourceOptimizer
    make_illumination : callable
        ``make_illumination(**p)`` returns an Illumination for each p in
        parameters.
    parameters : sequence of dict

    Returns
    -------
    list of (worst NILS, parameters, Illumination), best first.
    """

    out = []
    for p in parameters:
        try:
            ill = make_illumination(**p)
        except ValueError:
            continue
        out.append((float(np.min(optimizer.evaluate(ill))), p, ill))

    return sorted(out, key=lambda r: -r[0])


__all__ = ["SourcePixels", "SourceOptimizer", "SourceResult", "scan"]
