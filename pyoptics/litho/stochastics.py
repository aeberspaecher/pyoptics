#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Stochastic resist exposure: photon shot noise, discrete acid and
quencher, line edge and width roughness, local CD uniformity and missing or
merged contacts.

The resist chain of `resist` works with concentrations, the averages over
many molecules. In a cell of a few nanometres there are only a few photons,
acid generator, quencher and protecting groups, and their counts fluctuate.
This module follows the same chain on a top-down grid (x, y) with counts:

1. Photons. A dose D (mJ/cm**2) at the wavelength lambda delivers
   D*1e-17/E_ph photons per nm**2 (E_ph = h*c/lambda in J): about 9.7 per
   mJ/cm**2 at ArF (193 nm), 12.5 at KrF (248 nm) and 0.68 at EUV (13.5
   nm), 14 times fewer than ArF. A fraction 1 - exp(-alpha*d) of them is
   absorbed in a resist of thickness d. Each cell of area dx*dy (a column
   through the resist) absorbs a Poisson number n of photons with the mean
   n_bar from the image I (relative to the open frame).

2. Acid. On average a cell holds pag*V*(1 - exp(-C*D*I)) acids after the
   exposure (V = dx*dy*d, pag acid generator molecules per nm**3, Dill C of
   the resist model), so an absorbed photon makes on average y =
   pag*V*(1 - exp(-C*D*I))/n_bar acids, the acid yield. Given n photons
   the acid count is binomial, Binomial(n, y), for y <= 1 (each photon
   makes at most one acid, as in KrF and ArF resists), and Poisson with
   the mean y*n for y > 1 (several acids per photon from the secondary
   electrons of an EUV photon). Without photon noise the acid count is
   Poisson with the mean pag*V*(1 - exp(-C*D*I)) (random positions of the
   acid generator), without acid noise it is y*n.

   For y <= 1 photon and acid noise together give a Poisson acid count:
   the acids are a random subset of the photons, and the acid count alone
   sets the noise. The photon noise on its own has the variance y times the
   mean, so it matters little at ArF (y of a few tenths); for y > 1 the
   variance grows to (1 + y) times the mean, the photon limit of EUV.

3. Quencher. Poisson with the mean quencher*pag*V (quencher relative to the
   acid generator, as in resist.ResistModel).

4. Post-exposure bake. The counts, divided by pag*V, are the initial acid
   and quencher concentrations of resist.ResistModel.bake(): diffusion,
   neutralization and amplification as in the deterministic model, here
   periodic in x and y. This is the usual mean-field treatment of the bake
   (each molecule spreads into a Gaussian of the diffusion length; reaction
   rates use the local mean concentrations): the shot noise of the counts
   is kept, the noise of the random walks and reactions is not.

5. Deprotection. Of the blocking groups (blocking per nm**3) a fraction m
   stays protected; their count is Poisson with the mean blocking*V*m.

6. Development. The protected fraction is blurred with a Gaussian of
   development_length (the dissolution front does not follow single
   molecules) and the resist is removed where m < m_th (the threshold
   inhibitor concentration of the Mack model).

All noise sources can be switched off one by one (noise=...); without any
the chain gives the mean (deterministic) inhibitor. After the bake the
noise of a cell no longer depends on the grid as long as dx and dy are
small against the diffusion and development lengths.

Line edge roughness (LER) and line width roughness (LWR) are three
standard deviations of the edge positions and the widths along a line;
their power spectral density (PSD) follows the usual model (Palasantzas;
Mack) ::

    PSD(f) = PSD(0)/(1 + (2*pi*f*xi)**2)**(H + 1/2),

with the correlation length xi and the roughness exponent H, normalized so
that the one-sided PSD integrates to sigma**2. edge_psd() estimates the PSD
of measured edges, fit_psd() fits the model, random_edges() draws edges with
a model PSD.

For contact holes, contact_stats() measures every hole of a clip of many
periods (equivalent diameter from the cleared area), counts missing holes
(nothing cleared) and merged ones (a cleared region spanning several
cells), and gives the local CD uniformity (3 sigma of the hole CDs).

Lengths in nm, doses in mJ/cm**2, the wavelength in nm.

References
----------
C. A. Mack, "Fundamental Principles of Optical Lithography", Wiley (2007),
ch. 9 (line edge roughness).

C. A. Mack, "Reducing roughness in extreme ultraviolet lithography", J.
Micro/Nanolithogr. MEMS MOEMS 17, 041006 (2018).

G. Palasantzas, "Roughness spectrum and surface width of self-affine fractal
surfaces via the K-correlation model", Phys. Rev. B 48, 14472 (1993).

R. L. Brainard, P. Trefonas, J. H. Lammers, C. A. Cutler, J. F. Mackevich,
A. Trefonas and S. A. Robertson, "Shot noise, LER and quantum efficiency of
EUV photoresists", Proc. SPIE 5374, 74-85 (2004).

P. De Bisschop, "Stochastic effects in EUV lithography: random, local CD
variability, and printing failures", J. Micro/Nanolithogr. MEMS MOEMS 16,
041013 (2017).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import brentq, curve_fit
from scipy.special import gammaln

from .resist import ResistModel
from .metrics import feature_edges


PLANCK = 6.62607015e-34  # J s
SPEED_OF_LIGHT = 299792458.0  # m/s


def photon_energy(wavelength):
    """Photon energy h*c/lambda in J for a vacuum wavelength in nm."""

    return PLANCK*SPEED_OF_LIGHT/(np.asarray(wavelength, dtype=float)*1e-9)


def photons_per_area(dose, wavelength):
    """Photons per nm**2 in a dose (mJ/cm**2) at a wavelength (nm).

    1 mJ/cm**2 is 1e-17 J/nm**2: about 9.7 photons/nm**2 at 193 nm, 12.5 at
    248 nm and 0.68 at 13.5 nm.
    """

    return np.asarray(dose, dtype=float)*1e-17/photon_energy(wavelength)


NOISE_SOURCES = ("photon", "acid", "quencher", "deprotection")


class StochasticResult(object):
    """Result of StochasticResist.process().

    Attributes
    ----------
    I : (N_y, N_x) array
        Relative intensity.
    dose : number
    photons : (N_y, N_x) array
        Absorbed photons per cell (their mean without photon noise).
    acid_0, quencher_0 : (N_y, N_x) arrays
        Acid and quencher molecules per cell before the bake.
    acid : (N_y, N_x) array
        Acid concentration after the bake (relative to the acid generator).
    m : (N_y, N_x) array
        Protected fraction after the bake, the deprotection noise and the
        development blur.
    m_th : number
        Development threshold; the resist is removed where m < m_th.
    volume : number
        Cell volume dx*dy*thickness in nm**3.
    """

    def __init__(self, I, dose, photons, acid_0, quencher_0, acid, m, m_th, volume):
        self.I, self.dose = I, dose
        self.photons, self.acid_0, self.quencher_0 = photons, acid_0, quencher_0
        self.acid, self.m, self.m_th, self.volume = acid, m, m_th, volume

    @property
    def cleared(self):
        """Boolean map: True where the (positive) resist is removed."""

        return self.m < self.m_th


class StochasticResist(object):
    """Chemically amplified resist with discrete photons, acid, quencher and
    protecting groups on a top-down grid (see the module documentation).

    Parameters
    ----------
    model : resist.ResistModel, optional
        Chemistry: Dill C, acid and quencher diffusion lengths, quencher
        loading (relative to the acid generator), neutralization rate,
        amplification, threshold m_th and the number of bake steps.
    wavelength : number, optional
        Vacuum wavelength in nm (default ArF, 193.368 nm); sets the photon
        energy.
    thickness : number, optional
        Resist thickness d in nm.
    absorption : number, optional
        Absorption coefficient alpha of the resist in 1/nm (base e);
        the absorbed fraction is 1 - exp(-alpha*d).
    pag : number, optional
        Acid generator molecules per nm**3.
    blocking : number, optional
        Protecting (blocking) groups per nm**3.
    development_length : number, optional
        Gaussian blur (nm) of the protected fraction before the threshold.
    noise : sequence of str, optional
        Noise sources to include, any of "photon", "acid", "quencher",
        "deprotection"; () gives the mean chain.
    """

    def __init__(self, model=None, wavelength=193.368, thickness=90.0, absorption=1e-3, pag=0.07,
                 blocking=1.5, development_length=2.0, noise=NOISE_SOURCES):
        self.model = ResistModel(C=0.05, diffusion_length=10.0, amplification=3.0, quencher=0.15,
                                 quencher_diffusion_length=5.0, n_steps=20) if model is None else model
        self.wavelength, self.thickness, self.absorption = wavelength, thickness, absorption
        self.pag, self.blocking = pag, blocking
        self.development_length = development_length
        unknown = set(noise) - set(NOISE_SOURCES)
        if unknown:
            raise ValueError("unknown noise sources {}".format(sorted(unknown)))
        self.noise = tuple(noise)

    def with_(self, **kwargs):
        """Copy with some parameters exchanged."""

        p = dict(vars(self))
        p.update(kwargs)

        return StochasticResist(**p)

    @property
    def absorbed_fraction(self):
        """1 - exp(-alpha*d)."""

        return 1 - np.exp(-self.absorption*self.thickness)

    def absorbed_photons(self, dose, area=1.0):
        """Mean number of photons absorbed in a column of the resist of the
        given area (nm**2) at a dose (mJ/cm**2) in the open frame."""

        return photons_per_area(dose, self.wavelength)*self.absorbed_fraction*area

    def acid_yield(self, I, dose, dx, dy=None):
        """Mean acids per absorbed photon, pag*V*(1 - exp(-C*D*I))/n_bar
        (its limit C*pag*d/(photons per area and mJ/cm**2*absorbed fraction)
        where the intensity is zero)."""

        I = np.asarray(I, dtype=float)
        dy = dx if dy is None else dy
        n_pag = self.pag*dx*dy*self.thickness
        n_bar = self.absorbed_photons(dose, dx*dy)*I
        limit = self.model.C*n_pag/self.absorbed_photons(1.0, dx*dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n_bar > 0, n_pag*-np.expm1(-self.model.C*dose*I)/n_bar, limit)

    def _rng(self, rng):
        return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def process(self, I, dose, dx, dy=None, rng=None):
        """Expose, bake and develop with shot noise.

        Parameters
        ----------
        I : (N_y, N_x) array
            Relative intensity (image normalized to the open frame), periodic
            in both directions (whole periods of the layout).
        dose : number
            mJ/cm**2.
        dx, dy : numbers
            Grid spacings (dy defaults to dx).
        rng : numpy Generator or seed, optional

        Returns
        -------
        StochasticResult
        """

        I = np.atleast_2d(np.asarray(I, dtype=float))
        dy = dx if dy is None else dy
        rng = self._rng(rng)
        noise = set(self.noise)
        V = dx*dy*self.thickness
        n_pag = self.pag*V

        n_bar = self.absorbed_photons(dose, dx*dy)*I
        mean_acid = n_pag*(1 - np.exp(-self.model.C*dose*I))
        y = self.acid_yield(I, dose, dx, dy)
        if "photon" in noise:
            photons = rng.poisson(n_bar).astype(float)
            if "acid" in noise:
                low = y <= 1
                acid_0 = np.where(low, rng.binomial(photons.astype(np.int64), np.minimum(y, 1.0)),
                                  rng.poisson(np.where(low, 0.0, y*photons))).astype(float)
            else:
                acid_0 = y*photons
        else:
            photons = n_bar
            acid_0 = rng.poisson(mean_acid).astype(float) if "acid" in noise else mean_acid
        mean_q = np.full_like(I, self.model.quencher*n_pag)
        quencher_0 = rng.poisson(mean_q).astype(float) if "quencher" in noise else mean_q

        acid, m = self.model.bake(acid_0/n_pag, dx, dy, quencher=quencher_0/n_pag, periodic_z=True)
        if "deprotection" in noise:
            n_b = self.blocking*V
            m = rng.poisson(n_b*np.clip(m, 0.0, 1.0))/n_b
        if self.development_length > 0:
            m = gaussian_filter(m, (self.development_length/dy, self.development_length/dx),
                                mode="wrap")

        return StochasticResult(I, dose, photons, acid_0, quencher_0, acid, m, self.model.m_th, V)

    def mean(self, I, dose, dx, dy=None):
        """The chain without noise (process() with noise=())."""

        return self.with_(noise=()).process(I, dose, dx, dy)

    def mean_cd(self, x, I, dose, center=0.0, feature="line"):
        """CD of the mean chain for a periodic 1D cut.

        Parameters
        ----------
        x, I : arrays
            One or more whole periods (equidistant, end point excluded).
        dose : number
        center : number, optional
        feature : {"line", "space"}, optional
            A line of resist (stays where m >= m_th) or a space (cleared).

        Returns
        -------
        cd : number
            nan if the feature does not print.
        """

        x = np.asarray(x, dtype=float)
        m = self.mean(np.asarray(I, dtype=float)[None, :], dose, x[1] - x[0]).m[0]
        left, right = feature_edges(x, m, self.model.m_th, center, bright=_bright(feature))

        return right - left

    def dose_to_size(self, x, I, cd_target, center=0.0, feature="line", doses=(1.0, 300.0),
                     n_scan=40):
        """Dose at which the mean chain prints cd_target (see mean_cd())."""

        D = np.geomspace(doses[0], doses[1], n_scan)
        err = np.array([self.mean_cd(x, I, d, center, feature) - cd_target for d in D])
        ok = np.isfinite(err)
        for i in range(len(D) - 1):
            if ok[i] and ok[i + 1] and np.sign(err[i]) != np.sign(err[i + 1]):
                return brentq(lambda d: self.mean_cd(x, I, d, center, feature) - cd_target,
                              D[i], D[i + 1], xtol=1e-6*D[i])

        return np.nan


def _bright(feature):
    if feature not in ("line", "space"):
        raise ValueError("feature must be 'line' or 'space'")
    return feature == "line"


def line_edges(x, m, threshold, center=0.0, feature="line"):
    """Left and right edges of a line (or space) in every row of a
    developed resist.

    Parameters
    ----------
    x : (N_x,) array
        Increasing positions across the line.
    m : (N_y, N_x) array
        Protected fraction, e.g. StochasticResult.m.
    threshold : number
        Development threshold m_th.
    center : number, optional
        Nominal centre of the feature.
    feature : {"line", "space"}, optional
        A line of resist (m >= threshold) or a cleared space.

    Returns
    -------
    left, right : (N_y,) arrays
        nan in rows where the feature is broken (missing at the centre) or
        does not end inside the cut.
    """

    bright = _bright(feature)
    edges = np.array([feature_edges(x, row, threshold, center, bright) for row in np.atleast_2d(m)])

    return edges[:, 0], edges[:, 1]


class Roughness(object):
    """Line edge and width roughness of one or more lines.

    Attributes
    ----------
    cd : number
        Mean width.
    ler : number
        3 sigma of the edge positions about their means (both edges pooled).
    lwr : number
        3 sigma of the widths.
    correlation : number
        Correlation coefficient of the left and right edge deviations
        (LWR**2 = 2*LER**2*(1 - correlation)).
    broken : number
        Fraction of rows without a measurable feature.
    """

    def __init__(self, cd, ler, lwr, correlation, broken):
        self.cd, self.ler, self.lwr, self.correlation, self.broken = cd, ler, lwr, correlation, broken

    def __repr__(self):
        return "Roughness(cd={:.2f}, ler={:.2f}, lwr={:.2f}, correlation={:.2f}, broken={:.3f})".format(
            self.cd, self.ler, self.lwr, self.correlation, self.broken)


def roughness(left, right):
    """LER, LWR and mean CD from edges (arrays of shape (N_y,) or (lines,
    N_y)); rows with nan are left out, each line keeps its own mean.
    """

    left, right = np.atleast_2d(left), np.atleast_2d(right)
    ok = np.isfinite(left) & np.isfinite(right)
    dl, dr, w = [], [], []
    for l, r, k in zip(left, right, ok):
        if np.sum(k) < 2:
            continue
        dl.append(l[k] - np.mean(l[k]))
        dr.append(r[k] - np.mean(r[k]))
        w.append(r[k] - l[k])
    if not w:
        return Roughness(np.nan, np.nan, np.nan, np.nan, 1.0)
    dl, dr, w = np.concatenate(dl), np.concatenate(dr), np.concatenate(w)
    width = w - np.mean(w)
    ler = 3*np.sqrt(0.5*(np.mean(dl**2) + np.mean(dr**2)))
    corr = np.mean(dl*dr)/np.sqrt(np.mean(dl**2)*np.mean(dr**2))

    return Roughness(float(np.mean(w)), float(ler), float(3*np.sqrt(np.mean(width**2))), float(corr),
                     float(1 - np.mean(ok)))


def psd_model(f, sigma, xi, H=0.5):
    """One-sided edge PSD PSD(0)/(1 + (2*pi*f*xi)**2)**(H + 1/2), normalized
    to the variance sigma**2 (nm**3 for f in 1/nm, sigma and xi in nm)."""

    f = np.asarray(f, dtype=float)
    psd0 = sigma**2*4*np.sqrt(np.pi)*xi*np.exp(gammaln(H + 0.5) - gammaln(H))

    return psd0/(1 + (2*np.pi*f*xi)**2)**(H + 0.5)


def edge_psd(y, edges):
    """One-sided PSD of edge deviations along y, averaged over edges.

    Parameters
    ----------
    y : (N_y,) array
        Equidistant positions along the edge (periodic sampling).
    edges : (N_y,) or (n_edges, N_y) array
        Edge positions; each edge's mean is removed. Edges with nan are
        skipped.

    Returns
    -------
    f : (N_y//2,) array
        Frequencies 1/L ... 1/(2*dy) (no zero frequency).
    psd : array
        PSD in nm**3; sum(psd)*df is the variance of an edge.
    """

    edges = np.atleast_2d(np.asarray(edges, dtype=float))
    edges = edges[np.all(np.isfinite(edges), axis=1)]
    N = edges.shape[1]
    dy = y[1] - y[0]
    L = N*dy
    F = np.fft.rfft(edges - edges.mean(axis=1, keepdims=True), axis=1)*dy
    psd = np.mean(np.abs(F)**2, axis=0)/L
    psd[1:] *= 2
    if N % 2 == 0:
        psd[-1] /= 2

    return np.fft.rfftfreq(N, dy)[1:], psd[1:]


def fit_psd(f, psd, H=None, f_max=None):
    """Fit psd_model() to a PSD (least squares in log PSD).

    Parameters
    ----------
    f, psd : arrays
        From edge_psd().
    H : number, optional
        Fix the roughness exponent (0 < H <= 1); default: fit it as well.
    f_max : number, optional
        Fit only frequencies up to f_max (e.g. below the cut-off of the
        development blur, which the model does not describe).

    Returns
    -------
    sigma, xi, H : numbers
    """

    f, psd = np.asarray(f, dtype=float), np.asarray(psd, dtype=float)
    df = f[1] - f[0]
    if f_max is not None:
        f, psd = f[f <= f_max], psd[f <= f_max]
    sigma0 = np.sqrt(np.sum(psd)*df)
    half = psd[0]/2
    xi0 = 1/(2*np.pi*f[np.argmax(psd < half)]) if np.any(psd < half) else 1/(2*np.pi*f[-1])
    if H is None:
        fun = lambda f, ls, lx, h: np.log(psd_model(f, np.exp(ls), np.exp(lx), h))
        p, _ = curve_fit(fun, f, np.log(psd), p0=(np.log(sigma0), np.log(xi0), 0.5),
                         bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 1.0]))
        return float(np.exp(p[0])), float(np.exp(p[1])), float(p[2])
    fun = lambda f, ls, lx: np.log(psd_model(f, np.exp(ls), np.exp(lx), H))
    p, _ = curve_fit(fun, f, np.log(psd), p0=(np.log(sigma0), np.log(xi0)))

    return float(np.exp(p[0])), float(np.exp(p[1])), float(H)


def random_edges(N, dy, sigma, xi, H=0.5, n_edges=1, rng=None):
    """Random edges (deviations about 0) with the PSD of psd_model().

    Parameters
    ----------
    N : int
        Samples along the edge.
    dy : number
        Spacing.
    sigma, xi, H : numbers
        Model parameters.
    n_edges : int, optional
    rng : numpy Generator or seed, optional

    Returns
    -------
    edges : (n_edges, N) array
        Periodic in N*dy; the variance approaches sigma**2 when N*dy >> xi
        and dy << xi.
    """

    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    f = np.fft.rfftfreq(N, dy)
    L = N*dy
    amp = np.sqrt(psd_model(f, sigma, xi, H)*L/2)/dy
    amp[0] = 0.0
    if N % 2 == 0:
        amp[-1] *= np.sqrt(2)
    Z = (rng.standard_normal((n_edges, len(f))) + 1j*rng.standard_normal((n_edges, len(f))))/np.sqrt(2)
    if N % 2 == 0:
        Z[:, -1] = np.sqrt(2)*Z[:, -1].real
    return np.fft.irfft(amp*Z, n=N, axis=1)


class ContactStats(object):
    """Holes of a contact clip, see contact_stats().

    Attributes
    ----------
    cd : (n_y, n_x) array
        Equivalent diameter 2*sqrt(area/pi) of the cleared area in each
        cell (0 for a missing hole; merged holes count their share in the
        cell).
    missing : (n_y, n_x) bool array
    merged : (n_y, n_x) bool array
        The cleared region of the cell reaches into another cell.
    """

    def __init__(self, cd, missing, merged):
        self.cd, self.missing, self.merged = cd, missing, merged

    @property
    def n(self):
        return self.cd.size

    @property
    def missing_rate(self):
        return float(np.mean(self.missing))

    @property
    def merged_rate(self):
        return float(np.mean(self.merged))

    @property
    def good(self):
        """Cells with a single open, unmerged hole."""

        return ~self.missing & ~self.merged

    @property
    def mean_cd(self):
        return float(np.mean(self.cd[self.good])) if np.any(self.good) else np.nan

    @property
    def lcdu(self):
        """Local CD uniformity: 3 sigma of the CDs of the good holes."""

        return float(3*np.std(self.cd[self.good])) if np.sum(self.good) > 1 else np.nan

    def __repr__(self):
        return ("ContactStats(n={}, mean_cd={:.2f}, lcdu={:.2f}, missing={:.4f}, merged={:.4f})"
                .format(self.n, self.mean_cd, self.lcdu, self.missing_rate, self.merged_rate))


def contact_stats(x, y, cleared, pitch_x, pitch_y=None, x0=0.0, y0=0.0, min_area=0.0):
    """Measure the holes of a clip of contact holes.

    The cells are pitch_x by pitch_y and centred on x0 + i*pitch_x, y0 +
    j*pitch_y; the grid x, y must cover whole cells (e.g. n periods starting
    half a pitch before the first hole).

    Parameters
    ----------
    x, y : arrays
        Equidistant grid.
    cleared : (len(y), len(x)) bool array
        Developed resist, True where it is removed (StochasticResult.cleared).
    pitch_x, pitch_y : numbers
        Cell size (pitch_y defaults to pitch_x).
    x0, y0 : numbers, optional
        Centre of one hole.
    min_area : number, optional
        A hole with a cleared area of at most min_area (nm**2) is missing.

    Returns
    -------
    ContactStats
    """

    pitch_y = pitch_x if pitch_y is None else pitch_y
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cleared = np.asarray(cleared, dtype=bool)
    dA = (x[1] - x[0])*(y[1] - y[0])
    i = np.floor((x - x0)/pitch_x + 0.5).astype(int)
    j = np.floor((y - y0)/pitch_y + 0.5).astype(int)
    i -= i.min()
    j -= j.min()
    n_x, n_y = i.max() + 1, j.max() + 1
    cell = j[:, None]*n_x + i[None, :]
    area = np.bincount(cell[cleared], minlength=n_x*n_y)*dA

    labels, n_labels = label(cleared)
    merged = np.zeros(n_x*n_y, dtype=bool)
    if n_labels:
        lab, c = labels[cleared], cell[cleared]
        pairs = np.unique(lab*(n_x*n_y) + c)
        lab_of = pairs//(n_x*n_y)
        counts = np.bincount(lab_of, minlength=n_labels + 1)
        multi = counts[lab_of] > 1
        merged[pairs[multi] % (n_x*n_y)] = True
    missing = area <= min_area
    cd = 2*np.sqrt(area/np.pi)

    return ContactStats(cd.reshape(n_y, n_x), missing.reshape(n_y, n_x),
                        (merged & ~missing).reshape(n_y, n_x))


__all__ = ["photon_energy", "photons_per_area", "StochasticResist", "StochasticResult",
           "line_edges", "roughness", "Roughness", "psd_model", "edge_psd", "fit_psd",
           "random_edges", "contact_stats", "ContactStats"]
