#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""From lens tolerances to the wafer: CD uniformity, best focus and
placement of a scanner lens built with manufacturing errors.

The chain joins `pyoptics.tolerancing` with `pyoptics.litho.lens`:

1. **Aberrations across the slit.** `SlitAberrations` is a performance
   vector for `tolerancing.ToleranceAnalysis`: per slit position x, the
   change against the nominal lens of the Fringe Zernikes (waves) and of the
   placement (distortion plus Zernike tilt) of a small feature, both averaged
   over the scan positions y. The placement is expressed as the equivalent
   tilt in waves, ``placement*NA/wavelength``, so that the vector reads as a
   wavefront; with ``scanner_correction`` the translation, magnification and
   rotation that a scanner corrects are removed first (`lens.linear_correction()`).
2. **Sensitivities of the features.** `slit_sensitivities()` gives, per slit
   position, the derivatives of CD, placement and best focus of a feature
   with respect to the Zernikes (`lens.zernike_sensitivity()`), taken about
   the nominal lens's own aberrations there, which change them.
3. **Lithographic errors.** `LithoErrors` turns the aberration vector into
   the linear changes of CD, placement and best focus of each feature per slit
   position, divided by their specifications, plus the aberrations
   themselves, weighted with the quadratic CD sensitivities (near best focus
   CD changes with the square of most aberrations) and a small floor (which
   keeps them recoverable and regularizes compensators that no feature
   sees). Its merit is the one a scanner
   minimizes with its lens manipulators and the wafer stage: the
   compensators of `ToleranceAnalysis` (axial and lateral element
   positions, element tilts, the wafer focus and tilt, the laser
   wavelength) are adjusted against it.
4. **Wafer errors.** `wafer_errors()` evaluates the Monte Carlo vectors
   (`monte_carlo(..., vectors=True)`) with the full quadratic
   sensitivities (`lens.predict()`): CD, placement and best focus changes
   per lens, feature and slit position; `wafer_summary()` gives the
   3 sigma CD uniformity, the overlay contribution and the best focus spread,
   `wafer_yield()` the fraction of lenses within the specifications.

Everything is linear in the Zernikes except the quadratic terms of step 4,
and the Zernikes are linear in the tolerances only to first order;
`ToleranceAnalysis.monte_carlo()` traces each lens completely and checks
the linear model (`ToleranceAnalysis.linear_model()`).

Notebook 150 (lens tolerances and wafer yield).

References
----------
T. Matsuyama, Y. Ohmura and D. M. Williamson, "The lithographic lens: its
history and evolution", Proc. SPIE 6154, 615403 (2006).

A. Erdmann, "Optical and EUV Lithography: A Modeling Perspective", SPIE Press
(2021), chapter 6 (aberrations and their impact).

C. Mack, "Fundamental Principles of Optical Lithography: The Science of
Microfabrication", Wiley (2007).
"""

from collections import namedtuple

import numpy as np

from . import lens


def _correction_projector(x, y):
    """Projections that remove the scanner's linear placement correction
    (dx = a0 + a1 x + a2 y, dy = b0 + b1 y + b2 x) from placements on the
    grid of slit positions x and scan positions y (flattened y-major)."""

    X, Y = np.meshgrid(x, y)
    X, Y = X.ravel(), Y.ravel()
    one = np.ones(len(X))
    Ax = np.column_stack([one, X, Y])
    Ay = np.column_stack([one, Y, X])
    I = np.eye(len(X))

    return I - Ax @ np.linalg.pinv(Ax), I - Ay @ np.linalg.pinv(Ay)


class SlitAberrations:
    """Changes of the scan averaged Zernikes and placements across the slit
    against a nominal lens, a performance vector for
    `tolerancing.ToleranceAnalysis`.

    One block per slit position x: the changes of the Fringe Zernikes of
    terms (waves), then of the placement (x, y) as an equivalent tilt in
    waves, ``placement*NA/wavelength``. Each value is the mean over the scan
    positions y (to first order the scanned image; the image fading by the
    variation along y is left out).

    Parameters
    ----------
    system : raytrace.System
        The nominal lens.
    x : sequence
        Slit positions (image side, units of the system).
    y : sequence, optional
        Scan positions.
    terms : sequence of int, optional
        Fringe Zernikes of the vector (default 4 to 25; the tilts Z2, Z3 are
        part of the placement).
    NA : number, optional
        NA of the pupil coordinates (default: `lens.lens_NA()` of the
        nominal lens).
    N : int, optional
        Rays across the pupil for the Zernike fit.
    n_terms : int, optional
        Terms of the fit (default: the largest of terms, at least 36).
    scanner_correction : bool, optional
        Remove the placement that a scanner corrects (translation,
        magnification, rotation, fitted over all x and y) before averaging.
    length_unit : number, optional
        Length unit of the lithography simulation in units of the system
        (1e-6: nm for a system in mm).
    """

    def __init__(self, system, x, y=(0.0,), terms=range(4, 26), NA=None, N=16, n_terms=None,
                 scanner_correction=True, length_unit=1e-6):
        self.x = np.atleast_1d(np.asarray(x, dtype=float))
        self.y = np.atleast_1d(np.asarray(y, dtype=float))
        self.terms = np.array(list(terms), dtype=int)
        self.NA = lens.lens_NA(system) if NA is None else float(NA)
        self.N = N
        self.n_terms = max(36, int(self.terms.max())) if n_terms is None else n_terms
        self.scanner_correction = scanner_correction
        self.length_unit = length_unit
        self.wavelength = system.wavelength/length_unit
        self.fields = list(self.x)
        self._proj = _correction_projector(self.x, self.y) if scanner_correction else None
        self.nominal_zernikes, self.nominal_placement, self.nominal_aberrations = \
            self._evaluate(system, keep=True)
        self.units = "waves"

    def _evaluate(self, system, keep=False):
        """Scan averaged Zernikes (n_x, n_terms) and placements (n_x, 2) of a
        lens (and the full nominal Zernikes per slit position)."""

        nx, ny = len(self.x), len(self.y)
        Z = np.zeros((ny, nx, len(self.terms)))
        P = np.zeros((ny, nx, 2))
        full = []
        for i, yi in enumerate(self.y):
            for k, xk in enumerate(self.x):
                ab = lens.lens_aberrations(system, (xk, yi), self.n_terms, self.N, self.NA,
                                           self.length_unit)
                Z[i, k] = [ab.zernikes.get(int(j), 0.0) for j in self.terms]
                P[i, k] = ab.placement
                if keep:
                    full.append(ab.zernikes)
        if self._proj is not None:
            P = np.stack([(self._proj[d] @ P[..., d].ravel()).reshape(ny, nx) for d in (0, 1)],
                         axis=-1)
        if keep:
            mean = [{j: np.mean([full[i*nx + k][j] for i in range(ny)]) for j in full[k]}
                    for k in range(nx)]
            return Z.mean(axis=0), P.mean(axis=0), mean

        return Z.mean(axis=0), P.mean(axis=0)

    def __call__(self, system):
        """Performance vector of a lens.

        Parameters
        ----------
        system : raytrace.System

        Returns
        -------
        array
        """

        Z, P = self._evaluate(system)
        dZ = Z - self.nominal_zernikes
        dP = (P - self.nominal_placement)*self.NA/self.wavelength

        return np.concatenate([np.concatenate([dZ[k], dP[k]]) for k in range(len(self.x))])

    @property
    def block_size(self):
        """Length of one block (one slit position)."""

        return len(self.terms) + 2

    def blocks(self, size):
        """Slices of the vector per slit position.

        Parameters
        ----------
        size : int
            Length of the vector.

        Returns
        -------
        list of slice
        """

        n = size//len(self.x)

        return [slice(k*n, (k + 1)*n) for k in range(len(self.x))]

    def unpack(self, Y):
        """Zernike and placement changes from performance vectors.

        Parameters
        ----------
        Y : array (n, m) or (m,)

        Returns
        -------
        dZ : array (n, n_x, n_terms)
            Zernike changes in waves.
        dP : array (n, n_x, 2)
            Placement changes in the length unit of the simulation.
        """

        Y = np.atleast_2d(Y).reshape(-1, len(self.x), self.block_size)

        return Y[..., :-2], Y[..., -2:]*self.wavelength/self.NA


FeatureSensitivity = namedtuple("FeatureSensitivity", ["name", "direction", "x", "sensitivities"])
FeatureSensitivity.__doc__ = """Zernike sensitivities of a feature across
the slit.

Attributes
----------
name : str
direction : "x" or "y", the cut direction (placement along it)
x : slit positions
sensitivities : list of lens.Sensitivity, one per slit position
"""


def slit_sensitivities(sim, aberrations, name="feature", direction=None, delta=0.01,
                       threshold=0.3, dose=1.0, foci=None, N=401):
    """Zernike sensitivities of a feature at each slit position, about the
    nominal lens's own (scan averaged) aberrations there.

    Parameters
    ----------
    sim : LithoSimulation
        Mask, illumination and projector (its Zernikes are replaced).
    aberrations : SlitAberrations
        Slit positions, terms and the nominal Zernikes.
    name : str, optional
    direction : {"x", "y"}, optional
        Cut direction, see LithoSimulation.cut_coordinates().
    delta : number, optional
        Step in waves of the central differences.
    threshold, dose, foci, N
        As for lens.feature_response().

    Returns
    -------
    FeatureSensitivity
    """

    P = sim.projector
    if not np.isclose(P.NA, aberrations.NA, rtol=1e-6):
        sim = sim.with_(projector=P.with_(NA=aberrations.NA))
        P = sim.projector
    if direction is None:
        direction = sim.cut_coordinates(3, 1.0)[3]
    out = []
    for z in aberrations.nominal_aberrations:
        z = {j: c for j, c in z.items() if j > 3}  # tilts only shift the feature
        s = sim.with_(projector=P.with_(zernikes=z))
        out.append(lens.zernike_sensitivity(s, aberrations.terms, delta, threshold, dose, foci, N,
                                            direction))

    return FeatureSensitivity(name, direction, aberrations.x.copy(), out)


WaferErrors = namedtuple("WaferErrors", ["cd", "placement", "best_focus"])
WaferErrors.__doc__ = """Changes of CD, placement (along the cut) and best
focus of a feature against the nominal lens, arrays (n_lenses, n_x) in the
length unit of the simulation."""


class LithoErrors:
    """Lithographic errors across the slit, a performance vector for
    `tolerancing.ToleranceAnalysis` whose merit a scanner minimizes.

    One block per slit position: for each feature the linear changes of CD,
    placement and best focus divided by their specifications, then the
    aberration vector of `SlitAberrations`, each entry times a weight w_j.
    The block norm is the specification-normalized error at that slit
    position; the compensators of a `ToleranceAnalysis` with this
    performance are adjusted like a scanner adjusts its manipulators.

    The weights cover what the linear rows miss: near best focus CD changes
    with the square of most aberrations (``Q_j c_j**2/2``, see
    `lens.zernike_sensitivity()`). With ``w_j**2 = w0**2 + max |Q_j|/(2 cd)``
    (the largest over the features at that slit position), a term reaches
    one specification unit where its quadratic CD change reaches the CD
    specification, so that the compensators and `ToleranceAnalysis` weigh
    it correctly. w0 (zernike_weight) keeps every entry, including the
    placement, in the vector.

    Parameters
    ----------
    aberrations : SlitAberrations
    features : sequence of FeatureSensitivity
        From `slit_sensitivities()`, at the slit positions of aberrations.
    cd, placement, focus : numbers
        Specifications (length unit of the simulation): the changes are
        divided by them.
    zernike_weight : number, optional
        Smallest weight w0 of the aberrations in the vector (per wave).
    quadratic : bool, optional
        Add the quadratic CD sensitivities to the weights (default);
        otherwise all weights are w0.
    """

    def __init__(self, aberrations, features, cd, placement, focus, zernike_weight=0.05,
                 quadratic=True):
        self.aberrations = aberrations
        self.features = list(features)
        self.specs = dict(cd=float(cd), placement=float(placement), best_focus=float(focus))
        self.zernike_weight = float(zernike_weight)
        if zernike_weight <= 0:
            raise ValueError("zernike_weight must be positive")
        ab = aberrations
        nt = len(ab.terms)
        self.fields = ab.fields
        self.units = "specification"
        # linear map of each block of the aberration vector
        self._T, self.weights = [], []
        for k in range(len(ab.x)):
            rows = []
            q = np.zeros(nt + 2)
            for f in self.features:
                if not np.allclose(f.x, ab.x):
                    raise ValueError("Feature {} has other slit positions.".format(f.name))
                s = f.sensitivities[k]
                pos = {int(j): i for i, j in enumerate(s.terms)}
                L = {key: np.array([s.linear[key][pos[int(j)]] if int(j) in pos else 0.0
                                    for j in ab.terms]) for key in s.linear}
                Q = np.array([s.quadratic["cd"][pos[int(j)]] if int(j) in pos else 0.0
                              for j in ab.terms])
                q[:nt] = np.maximum(q[:nt], np.abs(Q)/(2*self.specs["cd"]))
                d = 0 if f.direction == "x" else 1
                tilt = np.zeros(2)
                tilt[d] = ab.wavelength/ab.NA  # tilt waves back to length
                for key in ("cd", "placement", "best_focus"):
                    row = np.concatenate([L[key], tilt if key == "placement" else np.zeros(2)])
                    rows.append(row/self.specs[key])
            rows = np.array(rows).reshape(-1, nt + 2)
            w = np.sqrt(self.zernike_weight**2 + (q if quadratic else 0.0*q))
            self.weights.append(w)
            self._T.append(np.vstack([rows, np.diag(w)]))
        self.weights = np.array(self.weights)
        self._n_lin = 3*len(self.features)

    def from_aberrations(self, y):
        """Performance vector from an aberration vector of `SlitAberrations`.

        Parameters
        ----------
        y : array
            Vector of `aberrations`.

        Returns
        -------
        array
        """

        n = self.aberrations.block_size

        return np.concatenate([T @ y[k*n:(k + 1)*n] for k, T in enumerate(self._T)])

    def __call__(self, system):
        """Performance vector of a lens.

        Parameters
        ----------
        system : raytrace.System

        Returns
        -------
        array
        """

        return self.from_aberrations(self.aberrations(system))

    def blocks(self, size):
        """Slices of the vector per slit position.

        Parameters
        ----------
        size : int
            Length of the vector.

        Returns
        -------
        list of slice
        """

        n = size//len(self.fields)

        return [slice(k*n, (k + 1)*n) for k in range(len(self.fields))]

    def unpack(self, Y):
        """Aberration vectors (of `SlitAberrations`) from performance vectors.

        Parameters
        ----------
        Y : array (n, m) or (m,)

        Returns
        -------
        array (n, m_aberrations)
        """

        nb = self.aberrations.block_size
        Y = np.atleast_2d(Y).reshape(-1, len(self.fields), self._n_lin + nb)

        return (Y[..., self._n_lin:]/self.weights).reshape(len(Y), -1)


def _aberration_vectors(performance, vectors):
    if isinstance(performance, LithoErrors):
        return performance.aberrations, performance.unpack(vectors)

    return performance, np.atleast_2d(vectors)


def wafer_errors(performance, vectors, features=None, quadratic=True):
    """CD, placement and best focus changes of each feature from the
    performance vectors of Monte Carlo lenses.

    The Zernike changes go through the sensitivities of each slit position,
    ``sum_j L_j c_j + Q_j c_j**2/2`` (`lens.predict()`); the placement adds the
    change of the lens placement (distortion and tilt, after the scanner
    correction) along the cut.

    Parameters
    ----------
    performance : SlitAberrations or LithoErrors
        The performance of the analysis.
    vectors : array (n, m)
        Performance vectors, e.g. ``monte_carlo(..., vectors=True).vectors``.
    features : sequence of FeatureSensitivity, optional
        Default: those of a LithoErrors performance.
    quadratic : bool, optional
        Include the second derivatives.

    Returns
    -------
    dict
        {name: WaferErrors}
    """

    ab, Y = _aberration_vectors(performance, vectors)
    if features is None:
        features = performance.features
    dZ, dP = ab.unpack(Y)
    out = {}
    for f in features:
        res = {k: np.zeros(dZ.shape[:2]) for k in ("cd", "placement", "best_focus")}
        for k, s in enumerate(f.sensitivities):
            pos = {int(j): i for i, j in enumerate(ab.terms)}
            idx = [pos.get(int(j)) for j in s.terms]
            use = [i for i, p in enumerate(idx) if p is not None]
            c = dZ[:, k, [idx[i] for i in use]]
            for key in res:
                L, Q = s.linear[key][use], s.quadratic[key][use]
                res[key][:, k] = c @ L + (0.5*(c**2) @ Q if quadratic else 0.0)
        res["placement"] += dP[:, :, 0 if f.direction == "x" else 1]
        out[f.name] = WaferErrors(res["cd"], res["placement"], res["best_focus"])

    return out


def wafer_summary(errors):
    """Statistics of the wafer errors of Monte Carlo lenses.

    Parameters
    ----------
    errors : dict
        {name: WaferErrors} from `wafer_errors()`.

    Returns
    -------
    dict
        {name: {"cd_3sigma": 3 sigma of the CD change over lenses and slit
        positions, "cd_range": mean over the lenses of the CD range across
        the slit, "placement_3sigma": 3 sigma of the placement change,
        "focus_range": mean over the lenses of the best focus range across
        the slit, "focus_3sigma": 3 sigma of the best focus change}}
    """

    out = {}
    for name, e in errors.items():
        out[name] = {"cd_3sigma": 3*np.std(e.cd), "cd_range": np.mean(np.ptp(e.cd, axis=1)),
                     "placement_3sigma": 3*np.std(e.placement),
                     "focus_range": np.mean(np.ptp(e.best_focus, axis=1)),
                     "focus_3sigma": 3*np.std(e.best_focus)}

    return out


def wafer_yield(errors, cd=None, placement=None, focus=None):
    """Fraction of the lenses whose changes stay within the specifications
    at all slit positions, for all features.

    Parameters
    ----------
    errors : dict
        {name: WaferErrors} from `wafer_errors()`.
    cd, placement, focus : numbers, optional
        Limits of the absolute changes (None: not checked).

    Returns
    -------
    number
    """

    ok = None
    for e in errors.values():
        good = np.ones(len(e.cd), bool)
        for v, lim in ((e.cd, cd), (e.placement, placement), (e.best_focus, focus)):
            if lim is not None:
                good &= np.all(np.abs(v) <= lim, axis=1)
        ok = good if ok is None else ok & good

    return float(np.mean(ok))
