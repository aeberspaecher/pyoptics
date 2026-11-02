#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical glass catalogues: dispersion formulas, glass maps and partial
dispersion.

The catalogues of SCHOTT, OHARA, HOYA, CDGM, SUMITA and HIKARI (about 1600
glasses) are shipped with pyoptics. They come from the refractiveindex.info
database (M. N. Polyanskiy, public domain, CC0 1.0), which took them from the
vendors' Zemax catalogues; `scripts/build_glass_catalog.py` rebuilds them.
Every glass keeps the vendor's dispersion formula (Sellmeier or the
Schott/Laurent power series) with its wavelength range, and the nominal
n_d, V_d, the deviation dP_gF of the partial dispersion from the normal line
(where the vendor gives it) and the production status.

A `Glass` is a callable n(wavelength_nm), so it can be used directly as the
material of a `raytrace.Surface`::

    from pyoptics import glasses
    bk7 = glasses.get("N-BK7")
    Surface(50.0, 5.0, material=bk7)

`catalog()` gives a `Catalog` (a dict name -> Glass) of one or several
vendors, filtered by status; `Catalog.nearest()` finds the glass nearest to
a point (n_d, V_d) of the glass map (to replace the model glasses of
`lens_optimization` by real glasses), `Catalog.table()` gives arrays for
glass maps and partial dispersion diagrams.

Partial dispersions and the normal line
---------------------------------------
The relative partial dispersion of the lines x, y is ::

    P_xy = (n_x - n_y)/(n_F - n_C).

For most glasses P_gF lies on a straight line over V_d, the *normal line*
through the glasses K7 and F2 (SCHOTT) ::

    P_gF = 0.6438 - 0.001682 V_d.

An achromat of two normal glasses brings F and C to one focus, but its
secondary spectrum (the focus at g relative to F and C) is
f (P_1 - P_2)/(V_1 - V_2); glasses with an anomalous partial dispersion
(dP_gF far from 0: fluor crowns, short flints) reduce it (apochromats).

Lines
-----
`LINES` holds the Fraunhofer and spectral lines (nm) i, h, g, F', F, e, d,
D, C', C, r, s, t.

References
----------
SCHOTT AG, "Optical Glass - Data Sheets" and TIE-29 "Refractive Index and
Dispersion" (Sellmeier formula, partial dispersion, normal line).

M. N. Polyanskiy, "Refractiveindex.info database of optical constants",
Sci. Data 11, 94 (2024).

R. Kingslake and R. B. Johnson, "Lens Design Fundamentals", 2nd ed.,
Academic Press (2010) (secondary spectrum, apochromats).
"""

import json
import os
from collections import namedtuple

import numpy as np


LINES = {"i": 365.01, "h": 404.66, "g": 435.83, "F'": 479.99, "F": 486.13, "e": 546.07,
         "d": 587.56, "D": 589.29, "C'": 643.85, "C": 656.27, "r": 706.52, "s": 852.11,
         "t": 1013.98}

VENDORS = ("schott", "ohara", "hoya", "cdgm", "sumita", "hikari")

NORMAL_LINE = (0.6438, -0.001682)  # P_gF = a + b V_d (SCHOTT, through K7 and F2)

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _wavelength_nm(w):
    return LINES[w] if isinstance(w, str) else float(w)


class Glass:
    """An optical glass with its dispersion formula.

    Parameters
    ----------
    name : str
    vendor : str
    formula : {"sellmeier", "sellmeier2", "polynomial"}
        Formula of refractiveindex.info, wavelengths in um:
        "sellmeier": n**2 - 1 = C1 + sum C_k l**2/(l**2 - C_k+1**2);
        "sellmeier2": n**2 - 1 = C1 + sum C_k l**2/(l**2 - C_k+1) (the
        SCHOTT form, B_i and C_i);
        "polynomial": n**2 = C1 + sum C_k l**C_k+1 (the Schott/Laurent power
        series of HOYA, SUMITA and others).
    coefficients : sequence
    range_um : (lower, upper), optional
        Wavelength range of the formula in um.
    nd, Vd : number, optional
        Nominal catalogue values (else computed).
    dPgF : number, optional
        Deviation of P_gF from the normal line given by the vendor.
    status : str, optional
        Production status ("standard", "preferred", "special", ...).
    code : str, optional
        Glass code.
    """

    def __init__(self, name, vendor, formula, coefficients, range_um=(0.0, np.inf), nd=None,
                 Vd=None, dPgF=None, status=None, code=None):
        if formula not in ("sellmeier", "sellmeier2", "polynomial"):
            raise ValueError("Unknown formula {!r}.".format(formula))
        self.name, self.vendor = name, vendor
        self.formula = formula
        self.coefficients = np.asarray(coefficients, float)
        self.range_um = tuple(range_um)
        self._nd, self._Vd = nd, Vd
        self.dPgF_catalog = dPgF
        self.status = status
        self.code = code

    def __repr__(self):
        return "Glass({!r}, {}, n_d={:.5f}, V_d={:.2f})".format(self.name, self.vendor, self.n_d,
                                                               self.V_d)

    def n(self, wavelength_nm):
        """Refractive index at the vacuum wavelength(s) in nm (or a line
        name such as "d", "F", "C", "g").

        Parameters
        ----------
        wavelength_nm : number, array or str

        Returns
        -------
        number or array
        """

        lam = np.asarray(_wavelength_nm(wavelength_nm) if isinstance(wavelength_nm, str)
                         else wavelength_nm, float)*1e-3
        C = self.coefficients
        l2 = lam**2
        if self.formula == "polynomial":
            n2 = C[0] + sum(C[k]*lam**C[k + 1] for k in range(1, len(C) - 1, 2))
        else:
            n2 = 1.0 + C[0]
            for k in range(1, len(C) - 1, 2):
                c = C[k + 1]**2 if self.formula == "sellmeier" else C[k + 1]
                n2 = n2 + C[k]*l2/(l2 - c)

        return np.sqrt(n2)

    def __call__(self, wavelength_nm):
        """Refractive index at wavelength_nm (a Glass is a material of
        `raytrace.Surface`)."""

        return self.n(wavelength_nm)

    def in_range(self, wavelength_nm):
        """Whether the wavelength lies in the range of the formula."""

        w = _wavelength_nm(wavelength_nm)*1e-3

        return self.range_um[0] <= w <= self.range_um[1]

    @property
    def n_d(self):
        """Catalogue n_d (computed from the formula if not given)."""

        return float(self._nd) if self._nd is not None else float(self.n("d"))

    @property
    def V_d(self):
        """Catalogue Abbe number V_d (computed if not given)."""

        return float(self._Vd) if self._Vd is not None else self.abbe_number()

    def abbe_number(self, center="d", short="F", long="C"):
        """Abbe number (n_center - 1)/(n_short - n_long) from the formula,
        e.g. V_e with center="e", short="F'", long="C'".

        Parameters
        ----------
        center, short, long : str or number
            Lines or wavelengths in nm.

        Returns
        -------
        float
        """

        return float((self.n(center) - 1)/(self.n(short) - self.n(long)))

    def partial_dispersion(self, x="g", y="F"):
        """Relative partial dispersion P_xy = (n_x - n_y)/(n_F - n_C).

        Parameters
        ----------
        x, y : str or number
            Lines or wavelengths in nm.

        Returns
        -------
        float
        """

        return float((self.n(x) - self.n(y))/(self.n("F") - self.n("C")))

    @property
    def dPgF(self):
        """Deviation of P_gF from the normal line (computed from the
        formula)."""

        return self.partial_dispersion("g", "F") - (NORMAL_LINE[0] + NORMAL_LINE[1]*self.V_d)

    @property
    def model(self):
        """The model glass (n_d, V_d) of `materials.abbe_dispersion()`."""

        return (self.n_d, self.V_d)


GlassTable = namedtuple("GlassTable", ["names", "vendors", "nd", "Vd", "PgF", "dPgF"])
GlassTable.__doc__ = """Arrays of a catalogue for plots: names, vendors,
n_d, V_d, P_gF and its deviation from the normal line."""


def glass_distance(nd_a, Vd_a, nd_b, Vd_b, index_weight=100.0):
    """Distance on the glass map with an index step of 1/index_weight
    counting as much as a step of 1 in the Abbe number (default 0.01 ~ 1).

    Parameters
    ----------
    nd_a, Vd_a, nd_b, Vd_b : number or array
    index_weight : number, optional

    Returns
    -------
    number or array
    """

    return np.hypot(index_weight*(np.asarray(nd_a) - nd_b), np.asarray(Vd_a) - Vd_b)


class Catalog(dict):
    """Glasses by name (a dict name -> Glass), with search and filters.
    Names that occur at several vendors are stored as "NAME (VENDOR)"."""

    def nearest(self, n_d, V_d, k=1, index_weight=100.0, dPgF=None, dPgF_weight=1000.0):
        """The glasses nearest to a point of the glass map.

        Parameters
        ----------
        n_d, V_d : number
            The point, e.g. a model glass of `lens_optimization`.
        k : int, optional
            Number of glasses.
        index_weight : number, optional
            See `glass_distance()`.
        dPgF : number, optional
            Also match this deviation from the normal line (for
            apochromats), weighted with dPgF_weight (0.001 ~ 1 in V_d).
        dPgF_weight : number, optional

        Returns
        -------
        Glass, or list of Glass for k > 1
        """

        t = self.table()
        d = glass_distance(t.nd, t.Vd, n_d, V_d, index_weight)
        if dPgF is not None:
            d = np.hypot(d, dPgF_weight*(t.dPgF - dPgF))
        order = np.argsort(d, kind="stable")[:k]  # ties (e.g. HT variants): first name
        out = [self[t.names[i]] for i in order]

        return out[0] if k == 1 else out

    def filter(self, predicate=None, status=None, n_range=None, V_range=None):
        """Sub catalogue.

        Parameters
        ----------
        predicate : callable, optional
            predicate(glass) -> bool.
        status : sequence of str, optional
            Keep glasses with one of these statuses.
        n_range, V_range : (lower, upper), optional

        Returns
        -------
        Catalog
        """

        def keep(g):
            if status is not None and g.status not in status:
                return False
            if n_range is not None and not n_range[0] <= g.n_d <= n_range[1]:
                return False
            if V_range is not None and not V_range[0] <= g.V_d <= V_range[1]:
                return False
            return predicate is None or predicate(g)

        return Catalog((k, g) for k, g in self.items() if keep(g))

    def table(self):
        """Arrays of all glasses (computed once), see `GlassTable`."""

        key = tuple(self)
        cached = self.__dict__.get("_table")
        if cached is not None and cached[0] == key:
            return cached[1]
        gl = list(self.values())
        with np.errstate(all="ignore"):
            PgF = np.array([g.partial_dispersion("g", "F") for g in gl])
        Vd = np.array([g.V_d for g in gl])
        t = GlassTable(list(self), [g.vendor for g in gl], np.array([g.n_d for g in gl]), Vd,
                       PgF, PgF - (NORMAL_LINE[0] + NORMAL_LINE[1]*Vd))
        self.__dict__["_table"] = (key, t)

        return t

    def __or__(self, other):
        out = Catalog(self)
        for k, g in other.items():
            out[k if k not in out else "{} ({})".format(g.name, g.vendor)] = g

        return out


_cache = {}


def load(vendor):
    """The full catalogue of one vendor.

    Parameters
    ----------
    vendor : str
        One of `VENDORS` (case insensitive).

    Returns
    -------
    Catalog
    """

    v = vendor.lower()
    if v not in VENDORS:
        raise ValueError("Unknown vendor {!r}, one of {}.".format(vendor, VENDORS))
    if v not in _cache:
        with open(os.path.join(_DATA, v + ".json"), encoding="utf-8") as f:
            doc = json.load(f)
        _cache[v] = Catalog((name, Glass(name, doc["vendor"], d["formula"], d["coefficients"],
                                         d["range_um"], d["nd"], d["Vd"], d["dPgF"], d["status"],
                                         d["code"]))
                            for name, d in sorted(doc["glasses"].items()))

    return Catalog(_cache[v])


def catalog(vendors=("schott",), status=None):
    """Glasses of one or several vendors.

    Parameters
    ----------
    vendors : str or sequence of str, optional
        Vendors (default SCHOTT), or "all".
    status : sequence of str, optional
        E.g. ("standard", "preferred") to leave out special and obsolete
        glasses (glasses without a status are kept only if None is in it).

    Returns
    -------
    Catalog
        Names that occur at several vendors get the vendor in brackets.
    """

    if isinstance(vendors, str):
        vendors = VENDORS if vendors == "all" else (vendors,)
    out = Catalog()
    for v in vendors:
        out = out | load(v)

    return out if status is None else out.filter(status=status)


def get(name, vendor=None):
    """A glass by name (case insensitive), from the given vendor or the
    first vendor of `VENDORS` that has it.

    Parameters
    ----------
    name : str
    vendor : str, optional

    Returns
    -------
    Glass
    """

    for v in ((vendor,) if vendor else VENDORS):
        c = load(v)
        for k, g in c.items():
            if k.upper() == name.upper():
                return g
    raise KeyError("No glass {!r}.".format(name))


def normal_line(V_d):
    """P_gF of the normal line at V_d."""

    return NORMAL_LINE[0] + NORMAL_LINE[1]*np.asarray(V_d)


def secondary_spectrum(glass_a, glass_b, f=1.0):
    """Secondary spectrum of a thin achromat of two glasses: the focus at g
    minus the common focus of F and C, f (P_a - P_b)/(V_a - V_b) (negative:
    g focus closer to the lens).

    Parameters
    ----------
    glass_a, glass_b : Glass
    f : number, optional
        Focal length.

    Returns
    -------
    float
    """

    Pa, Pb = glass_a.partial_dispersion(), glass_b.partial_dispersion()

    return -f*(Pa - Pb)/(glass_a.V_d - glass_b.V_d)
