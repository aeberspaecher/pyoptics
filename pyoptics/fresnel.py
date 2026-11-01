#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Fresnel lenses: design, ray tracing with real facets, and the way back.

A Fresnel lens keeps the surface slopes of a thick refracting lens (its
"parent", a sphere, conic or even asphere) and drops the material that only
holds the surface at its height: the sag is cut into annular zones and every
zone is lowered onto a common base plane. What remains is a plate with a
sawtooth relief of *facets* (the active, refracting parts) and *risers* (the
steps between the zones, also called drafts or groove walls). A. Fresnel
proposed it in 1822 for lighthouse lenses; today Fresnel lenses are moulded
in PMMA or polycarbonate for solar concentrators, overhead projectors,
spotlights, field lenses of cameras and VR headsets and LED optics.

Geometry
--------
In the local frame of a `FresnelSurface` the base plane is z = 0 and the
parent sag is z_p(r) (`raytrace.Surface.sag`). The zone radii are
0 = r_0 < r_1 < ... < r_K, for a constant groove pitch (moulded lenses),
a constant groove depth (diamond turned lenses, kinoforms) or given
explicitly (`zone_radii()`). The facet of zone k is ::

    curved facets:  f_k(r) = z_p(r) - z_p(r_k),
    flat facets:    f_k(r) = s_k (r - r_k),  s_k = (z_p(r_k+1) - z_p(r_k))/(r_k+1 - r_k),

so every facet starts on the base plane at the inner edge of its zone and
has the step height h_k = f_k(r_k+1) at the outer edge, where the riser
returns to the base plane. Flat facets are cones (a common way of cutting
moulds); the central zone always keeps the curved parent. A riser is
vertical (parallel to the axis), or, with a draft angle delta > 0 that
lets the mould release, a cone that leans by delta from the axis and covers
the start of the next facet over about ``|h_k| tan(delta)``. The last zone
continues beyond r_K without a step.

Ray tracing
-----------
Two models of a `FresnelSurface`:

- ``model="ideal"``: the usual idealization of lens design programs (e.g.
  the "Fresnel" surface of Zemax OpticStudio): the rays meet the flat base
  plane and are refracted with the normal of the parent at the same radius.
  The ray directions are those of an infinitely fine Fresnel lens; the
  optical path is that of a plane plate, so the OPD of such a lens is not
  the wavefront of any real lens (the rays are not normal to it).
- ``model="facets"``: the real sawtooth. Each ray is intersected with the
  facets and risers of the zones it can reach and refracted with the normal
  of the piece it hits. The optical path includes the step heights, so
  the pupil phase jumps by about (n - n') h_k at the zone edges. Rays are lost
  (vignetted, NaN) where a Fresnel lens loses light:

  - riser hits: the ray meets a riser first (draft loss, for oblique
    incidence also the shadow of the vertical riser);
  - shadowing: after refraction at a facet, the ray hits the relief again
    (the next riser) before it leaves the grooves, which is the main loss on
    the exit side of fast lenses;
  - total internal reflection at the facet.

  ``stray="refract"`` instead refracts riser hits with the riser normal
  and traces them on: the first order stray light, e.g. the halo around
  the focus of a solar concentrator. Rays that re-hit the relief are lost
  in either case (they would need a non sequential trace).

`FresnelSurface.classify()` and `facet_losses()` tell which of these
happens to each ray, `raytrace.System.trace_polarization()` adds the Fresnel
reflection losses at the facets.

The paraxial data (power, focal lengths, pupils) are those of the parent;
the Seidel sums of a Fresnel surface are not computed (NaN, as for an
ideal lens), since a plane carrying the parent's slopes does not have the
aberrations of the parent. A Fresnel surface is traced with NumPy (not by
the compiled kernel).

Design and the way back
-----------------------
`FresnelSurface.from_surface()` collapses a thick lens surface into zones
and `FresnelSurface.parent_surface()` returns the thick lens surface again;
the zone radii and facet slopes are the whole description of the lens.
`focusing_slopes()` gives the exact facet slopes for a perfect focus of
the ideal model by the vector law of refraction (the facet normal is
parallel to n d - n' d'); `surface_from_slopes()` fits an even asphere
(a parent) to slopes, e.g. to the facet angle table of a moulded lens or to
a design from `focusing_slopes()`. The parent of an aberration free Fresnel
lens is not the aberration free thick lens: a plano-convex lens that focuses
a collimated beam through its curved side without spherical aberration has
the hyperbola K = -n**2, the corresponding Fresnel lens a different conic
and aspheric terms, since its facets sit on the plane instead of on the
curved surface.

Waves: Fresnel lens vs. kinoform
--------------------------------
The steps of a Fresnel lens are many wavelengths deep (a 0.3 mm deep groove
in PMMA is several hundred waves of path); the phase jumps between the zones
are arbitrary and the zones add up incoherently: the focus is that of the
rays, blurred by diffraction at the zone width. A kinoform is a Fresnel lens
with the groove depth lambda_0/(n - 1) (constant depth zones): every step
is one wave, the zones add up in phase and the lens is diffractive. At
other wavelengths the step is ``alpha = lambda_0 (n(lambda) - 1)/(lambda (n_0 - 1))``
waves; the scalar efficiency of the diffraction order m is
``sinc**2(alpha - m)`` (`kinoform_efficiency()`), its focal length
f m lambda_0/lambda and its Abbe number lambda_d/(lambda_F - lambda_C) = -3.45
(`diffractive_abbe_number()`), strong and of opposite sign to glass.

References
----------
A. Fresnel, "Mémoire sur un nouveau système d'éclairage des phares", Paris
(1822).

R. Leutz and A. Suzuki, "Nonimaging Fresnel Lenses: Design and Performance of
Solar Concentrators", Springer (2001).

A. Davis and F. Kühnlenz, "Optical design using Fresnel lenses", Optik &
Photonik 2(4), 52-55 (2007).

W. T. Welford, "Aberrations of Optical Systems", Adam Hilger (1986), ch. 13
(Fresnel lenses and diffractive surfaces).

D. A. Buralli, G. M. Morris and J. R. Rogers, "Optical performance of
holographic kinoforms", Appl. Opt. 28, 976-983 (1989).

D. C. O'Shea, T. J. Suleski, A. D. Kathman and D. W. Prather, "Diffractive
Optics: Design, Fabrication, and Test", SPIE Press (2004), ch. 4.

N. Sultanova, S. Kasarova and I. Nikolov, "Dispersion properties of optical
polymers", Acta Phys. Pol. A 116, 585-587 (2009) (PMMA).

Zemax OpticStudio User Manual, "Fresnel" surface type.
"""

from collections import namedtuple

import numpy as np

from .raytrace import Surface, Rays, _dot, _norm


__all__ = ["FresnelSurface", "zone_radii", "focusing_slopes", "surface_from_slopes",
           "facet_losses", "FacetLosses", "vogel_spiral", "kinoform_efficiency", "kinoform_depth",
           "diffractive_abbe_number", "pmma", "FACET", "RISER", "SHADOWED", "TIR", "MISS"]

# ray classes of FresnelSurface.classify():
FACET, RISER, SHADOWED, TIR, MISS = 0, 1, 2, 3, 4
CLASS_NAMES = {FACET: "facet", RISER: "riser", SHADOWED: "shadowed", TIR: "TIR",
               MISS: "miss"}


def pmma(wavelength_nm):
    """Refractive index of PMMA (Sultanova et al. 2009, 437 to 1052 nm, n_d = 1.4906).

    Parameters
    ----------
    wavelength_nm : number or array
        Vacuum wavelength in nm.

    Returns
    -------
    n : number or array
    """

    lam2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2

    return np.sqrt(1 + 1.1819*lam2/(lam2 - 0.011313))


def zone_radii(surface, radius, pitch=None, depth=None, n_zones=None):
    """Zone radii of a Fresnel lens collapsed from a surface.

    Parameters
    ----------
    surface : raytrace.Surface or callable
        The parent (its sag(r2)) or a function z(r2).
    radius : number
        Outer radius of the lens.
    pitch : number, optional
        Constant zone width (groove pitch).
    depth : number, optional
        Constant groove depth: zone k ends where ``|z_p(r) - z_p(0)|`` reaches
        (k + 1) depth (needs a monotonic sag; the last zone may be shallower).
    n_zones : int, optional
        Number of zones of equal width (alternative to pitch).

    Returns
    -------
    r : array
        0 = r_0 < r_1 < ... < r_K = radius.
    """

    if sum(v is not None for v in (pitch, depth, n_zones)) != 1:
        raise ValueError("Give one of pitch, depth and n_zones.")
    if n_zones is not None:
        return np.linspace(0.0, radius, int(n_zones) + 1)
    if pitch is not None:
        r = np.arange(0.0, radius, pitch)
        return np.append(r, radius) if radius - r[-1] > 1e-9*pitch else np.append(r[:-1], radius)
    sag = surface.sag if hasattr(surface, "sag") else surface
    rr = np.linspace(0.0, radius, 20001)
    z = np.abs(sag(rr**2) - sag(np.zeros(1))[0])
    if np.any(np.diff(z) < 0):
        raise ValueError("Constant depth zones need a monotonic sag.")
    levels = np.arange(depth, z[-1], depth)
    r = np.interp(levels, z, rr)
    # refine on the sag (the interpolation is only linear between samples):
    for _ in range(4):
        h = 1e-7*radius
        zz = np.abs(sag(r**2) - sag(np.zeros(1))[0])
        dz = (np.abs(sag((r + h)**2) - sag(np.zeros(1))[0]) - zz)/h
        r = r - (zz - levels)/dz
    r = np.concatenate([[0.0], r, [radius]])
    if radius - r[-2] < 1e-9*radius:
        r = np.delete(r, -2)

    return r


class FresnelSurface(Surface):
    """Fresnel surface: a parent surface collapsed into zones on a base plane.

    See the module docstring for the geometry and the two ray trace models.

    Parameters
    ----------
    R, thickness : numbers, optional
        Vertex radius of the parent and distance to the next vertex (from
        the base plane).
    zones : sequence, optional
        Zone radii 0 = r_0 < ... < r_K (see `zone_radii()`).
    pitch, depth, n_zones : optional
        Instead of zones: build them with `zone_radii()` up to radius.
    radius : number, optional
        Outer radius for pitch, depth or n_zones (default: semi_diameter).
    model : {"facets", "ideal"}, optional
        Real sawtooth or the flat idealization.
    facets : {"curved", "flat"}, optional
        Facets that follow the parent, or cones through the zone edges of
        the parent (central zone curved).
    draft : number, optional
        Draft angle of the risers (radians) from the axis.
    stray : {"lose", "refract"}, optional
        Rays that hit a riser first are lost, or refracted at the riser.
    n_after, material, conic, aspheric, semi_diameter, mirror, decenter, tilt,
    coating, label, index_offset : optional
        As for `raytrace.Surface` (conic and aspheric describe the parent).
    """

    def __init__(self, R=np.inf, thickness=0.0, zones=None, pitch=None, depth=None,
                 n_zones=None, radius=None, model="facets", facets="curved", draft=0.0,
                 stray="lose", **kwargs):
        super().__init__(R, thickness, **kwargs)
        if model not in ("facets", "ideal"):
            raise ValueError("model must be 'facets' or 'ideal'")
        if facets not in ("curved", "flat"):
            raise ValueError("facets must be 'curved' or 'flat'")
        if stray not in ("lose", "refract"):
            raise ValueError("stray must be 'lose' or 'refract'")
        if not 0 <= draft < np.pi/2:
            raise ValueError("draft must lie in [0, pi/2)")
        if zones is None:
            radius = self.semi_diameter if radius is None else radius
            if radius is None:
                raise ValueError("Give zones, or radius (or semi_diameter) with pitch, depth "
                                 "or n_zones.")
            zones = zone_radii(lambda r2: Surface.sag(self, r2), radius, pitch=pitch, depth=depth,
                               n_zones=n_zones)
        zones = np.asarray(zones, dtype=float)
        if zones[0] != 0 or np.any(np.diff(zones) <= 0):
            raise ValueError("zones must start at 0 and increase.")
        self.zones = zones
        self.model = model
        self.facets = facets
        self.draft = float(draft)
        self.stray = stray

    # surfaces have a Seidel sum (raytrace.System.seidel()); a plane with the
    # slopes of the parent has not those of the parent:
    has_seidel = False

    def __repr__(self):
        return ("FresnelSurface(R={}, thickness={}, conic={}, aspheric={}, n_zones={}, "
                "model={!r}, facets={!r}, draft={})".format(
                    self.R, self.thickness, self.conic, self.aspheric, self.n_zones,
                    self.model, self.facets, self.draft))

    @classmethod
    def from_surface(cls, surface, pitch=None, depth=None, n_zones=None, zones=None,
                     radius=None, **kwargs):
        """Fresnel version of a (thick lens) surface.

        Parameters
        ----------
        surface : raytrace.Surface
            The parent: radius, conic, aspheric terms, thickness, medium,
            semi diameter, decentre, tilt, coating and label are copied.
        pitch, depth, n_zones, zones, radius : optional
            Zones, see `zone_radii()`.
        **kwargs
            model, facets, draft, stray, or overrides of the copied values
            (e.g. thickness).

        Returns
        -------
        FresnelSurface
        """

        args = dict(n_after=surface.n_after, material=surface.material, conic=surface.conic,
                    aspheric=surface.aspheric, semi_diameter=surface.semi_diameter,
                    mirror=surface.mirror, decenter=surface.decenter, tilt=surface.tilt,
                    coating=surface.coating, label=surface.label,
                    index_offset=getattr(surface, "index_offset", 0.0))
        thickness = kwargs.pop("thickness", surface.thickness)
        args.update(kwargs)

        return cls(surface.R, thickness, zones=zones, pitch=pitch, depth=depth, n_zones=n_zones,
                   radius=radius, **args)

    def parent_surface(self, thickness=None):
        """The thick lens surface (the parent) as a `raytrace.Surface`.

        Parameters
        ----------
        thickness : number, optional
            Distance to the next vertex (default: that of the Fresnel
            surface; a parent in place of a Fresnel surface on the exit side
            of a plate needs the thickness changed by its sag at the rim).

        Returns
        -------
        raytrace.Surface
        """

        return Surface(self.R, self.thickness if thickness is None else thickness,
                       n_after=self.n_after, material=self.material, conic=self.conic,
                       aspheric=self.aspheric, semi_diameter=self.semi_diameter,
                       mirror=self.mirror, decenter=self.decenter, tilt=self.tilt,
                       coating=self.coating, label=self.label,
                       index_offset=getattr(self, "index_offset", 0.0))

    # geometry of the relief:
    @property
    def n_zones(self):
        return len(self.zones) - 1

    def parent_sag(self, r):
        """Sag z_p of the parent at the radii r."""

        return Surface.sag(self, np.asarray(r, dtype=float)**2)

    def _geometry(self):
        """Facet offsets and slopes, step heights and riser feet (cached on
        the parent and the zones)."""

        key = (self.R, self.conic, self.aspheric, self.zones.tobytes(), self.facets, self.draft)
        cache = self.__dict__.get("_geom")
        if cache is not None and cache[0] == key:
            return cache[1]
        r = self.zones
        K = len(r) - 1
        zp = self.parent_sag(r)
        base = zp[:-1]                                  # z_p(r_k)
        slope = np.diff(zp)/np.diff(r)                  # chords, flat facets
        flat = np.zeros(K, dtype=bool)
        if self.facets == "flat":
            flat[1:] = True
        h = zp[1:] - base                               # step heights at r_k+1
        # riser feet e_k (riser k between zone k and k + 1 at r_k+1):
        e = r[1:-1].copy()
        if self.draft > 0 and K > 1:
            tan_d = np.tan(self.draft)
            hk, r0 = h[:-1], r[1:-1]
            sg = np.sign(hk)

            def gap(x):  # sg*(L(x) - f_k+1(x)) >= 0 on the riser
                f_next = np.where(flat[1:], slope[1:]*(x - r0),
                                  self.parent_sag(x) - base[1:])
                return sg*(hk - sg*(x - r0)/tan_d - f_next)

            lo, hi = r0.copy(), r0 + np.abs(hk)*tan_d*1.0001 + 1e-15
            for _ in range(60):
                mid = 0.5*(lo + hi)
                pos = gap(mid) > 0
                lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
            e = 0.5*(lo + hi)
            # the riser may not cover the whole next zone:
            e = np.minimum(e, r[2:])
        start = np.concatenate([[0.0], e])              # first point of facet k
        z_all = np.concatenate([[0.0], h])
        z_min, z_max = min(0.0, z_all.min()), max(0.0, z_all.max())
        parent = Surface(self.R, conic=self.conic, aspheric=self.aspheric)
        g = dict(parent=parent, r=r, K=K, base=base, slope=slope, flat=flat, h=h, e=e, start=start,
                 z_min=z_min, z_max=z_max, scale=max(r[-1], 1e-12))
        self._geom = (key, g)

        return g

    def facet_sag(self, k, r):
        """Height f_k(r) of the facet of zone k at the radii r (k may be an array)."""

        g = self._geometry()
        k = np.asarray(k)
        r = np.asarray(r, dtype=float)
        curved = self.parent_sag(r) - g["base"][k]
        return np.where(g["flat"][k], g["slope"][k]*(r - g["r"][k]), curved)

    def _riser_line(self, k, r):
        """Height of the drafted riser k at the radii r."""

        g = self._geometry()
        return g["h"][k] - np.sign(g["h"][k])*(r - g["r"][k + 1])/np.tan(self.draft)

    def profile(self, r):
        """Height z(r) of the relief (facets and risers) at the radii r."""

        g = self._geometry()
        r = np.abs(np.asarray(r, dtype=float))
        k = np.clip(np.searchsorted(g["r"], r, side="right") - 1, 0, g["K"] - 1)
        z = self.facet_sag(k, r)
        if self.draft > 0 and g["K"] > 1:
            j = np.clip(k - 1, 0, g["K"] - 2)
            on_riser = (k >= 1) & (r < g["e"][j])
            z = np.where(on_riser, self._riser_line(j, r), z)

        return z

    def sag(self, r2):
        """Sag at the squared radius r2: the relief (facets), 0 for the ideal model."""

        r2 = np.asarray(r2, dtype=float)
        if self.model == "ideal":
            return np.zeros(np.shape(r2))

        return self.profile(np.sqrt(r2))

    def section(self, h, n_per_zone=12):
        """Points (y, z) of the meridional section from -h to h, with the
        corners of the risers, for drawing.

        Parameters
        ----------
        h : number
            Half height of the section.
        n_per_zone : int, optional
            Points per facet.

        Returns
        -------
        y, z : arrays
        """

        if self.model == "ideal":
            y = np.array([-h, h])
            return y, np.zeros(2)
        g = self._geometry()
        pts = []
        for k in range(g["K"]):
            a = g["start"][k]
            b = g["r"][k + 1] if k < g["K"] - 1 else max(h, g["r"][-1])
            if a >= h:
                break
            b = min(b, h)
            rr = np.linspace(a, b, n_per_zone)
            pts.append(np.stack([rr, self.facet_sag(k, rr)], axis=-1))
            if k < g["K"] - 1 and b < h:
                pts.append(np.array([[g["r"][k + 1], g["h"][k]]]))  # top of the riser
        P = np.concatenate(pts)
        y = np.concatenate([-P[::-1, 0], P[:, 0]])
        z = np.concatenate([P[::-1, 1], P[:, 1]])

        return y, z

    # intersections with the pieces:
    @staticmethod
    def _quadratic(a, b, c):
        """Both roots of a t**2 + b t + c = 0 (NaN if complex; the linear
        root twice for a = 0)."""

        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.sqrt(b**2 - 4*a*c)
            q = -0.5*(b + np.where(b >= 0, 1.0, -1.0)*sq)
            t1, t2 = q/a, c/q
            lin = np.abs(a) < 1e-14*(np.abs(b) + np.abs(c) + 1e-300)
            t1 = np.where(lin, -c/b, t1)
            t2 = np.where(lin, -c/b, t2)

        return t1, t2

    def _cone_hits(self, P, d, m, z0, r0, r_lo, r_hi):
        """Hits of the cone z = z0 + m (r - r0), r in [r_lo, r_hi]."""

        x, y, z = P.T
        dx, dy, dz = d.T
        b0 = m*r0 - z0  # z + b0 = m r
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.isscalar(m) and m == 0:
                t = (z0 - z)/dz
                roots = (t, np.full_like(t, np.nan))
            else:
                a = dz**2 - m**2*(dx**2 + dy**2)
                b = 2*((z + b0)*dz - m**2*(x*dx + y*dy))
                c = (z + b0)**2 - m**2*(x**2 + y**2)
                roots = self._quadratic(a, b, c)
        out = []
        tol = 1e-9*self._geometry()["scale"]
        for t in roots:
            Q = P + t[:, None]*d
            rq = np.hypot(Q[:, 0], Q[:, 1])
            with np.errstate(invalid="ignore"):
                ok = (np.isfinite(t) & (rq >= r_lo - tol) & (rq <= r_hi + tol)
                      & (np.abs(Q[:, 2] - (z0 + m*(rq - r0))) <= tol + 1e-9*np.abs(Q[:, 2])))
            out.append(np.where(ok, t, np.nan))

        return out

    def _cylinder_hits(self, P, d, r0, z_lo, z_hi):
        """Hits of the cylinder r = r0 with z in [z_lo, z_hi]."""

        x, y, z = P.T
        dx, dy, dz = d.T
        roots = self._quadratic(dx**2 + dy**2, 2*(x*dx + y*dy), x**2 + y**2 - r0**2)
        tol = 1e-9*self._geometry()["scale"]
        out = []
        for t in roots:
            zq = z + t*dz
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(t) & (zq >= z_lo - tol) & (zq <= z_hi + tol)
            out.append(np.where(ok, t, np.nan))

        return out

    def _facet_hits(self, k, P, d):
        g = self._geometry()
        r_lo = g["start"][k]
        r_hi = g["r"][k + 1] if k < g["K"] - 1 else np.inf
        if g["flat"][k]:
            return self._cone_hits(P, d, g["slope"][k], 0.0, g["r"][k], r_lo, r_hi)
        Ps = P + np.array([0.0, 0.0, g["base"][k]])
        t = g["parent"].intersect(Ps, d)
        Q = P + t[:, None]*d
        rq = np.hypot(Q[:, 0], Q[:, 1])
        tol = 1e-9*g["scale"]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(t) & (rq >= r_lo - tol) & (rq <= r_hi + tol)

        return [np.where(ok, t, np.nan)]

    def _riser_hits(self, k, P, d):
        g = self._geometry()
        hk, r0 = g["h"][k], g["r"][k + 1]
        if self.draft == 0:
            return self._cylinder_hits(P, d, r0, min(0.0, hk), max(0.0, hk))
        m = -np.sign(hk)/np.tan(self.draft)

        return self._cone_hits(P, d, m, hk, r0, r0, g["e"][k])

    def _first_hit(self, P, d, t_min=None):
        """Distance to the first piece of the relief and its code (2k: facet
        of zone k, 2k + 1: riser k; -1: none).

        t_min: hits at t > t_min[i] count (default: from where the ray enters
        the slab of the relief)."""

        g = self._geometry()
        N = len(P)
        z_lo, z_hi = g["z_min"], g["z_max"]
        margin = (z_hi - z_lo) + 1e-9*g["scale"]
        dz = d[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            t_a = (z_lo - margin - P[:, 2])/dz
            t_b = (z_hi + margin - P[:, 2])/dz
        t0, t1 = np.fmin(t_a, t_b), np.fmax(t_a, t_b)
        if t_min is not None:
            t0 = np.fmax(t0, t_min)
        # radial range of the ray in the slab:
        dxy2 = d[:, 0]**2 + d[:, 1]**2
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = np.where(dxy2 > 0, -(P[:, 0]*d[:, 0] + P[:, 1]*d[:, 1])/dxy2, t0)
        ts = np.clip(ts, t0, t1)

        def rad(t):
            return np.hypot(P[:, 0] + t*d[:, 0], P[:, 1] + t*d[:, 1])

        r_lo = rad(ts)
        r_hi = np.fmax(rad(t0), rad(t1))
        k_lo = np.clip(np.searchsorted(g["r"], r_lo, side="right") - 2, 0, g["K"] - 1)
        k_hi = np.clip(np.searchsorted(g["r"], r_hi, side="right"), 0, g["K"] - 1)
        good = np.isfinite(r_lo) & np.isfinite(r_hi)
        best = np.full(N, np.inf)
        piece = np.full(N, -1)
        if not np.any(good):
            return np.full(N, np.nan), piece
        tol = 1e-12*g["scale"]
        for k in range(int(k_lo[good].min()), int(k_hi[good].max()) + 1):
            sel = np.flatnonzero(good & (k_lo <= k) & (k <= k_hi))
            if len(sel) == 0:
                continue
            Ps, ds = P[sel], d[sel]
            cands = [(2*k, t) for t in self._facet_hits(k, Ps, ds)]
            if k < g["K"] - 1:
                cands += [(2*k + 1, t) for t in self._riser_hits(k, Ps, ds)]
            for code, t in cands:
                with np.errstate(invalid="ignore"):
                    better = np.isfinite(t) & (t > t0[sel] + tol) & (t < best[sel])
                best[sel[better]] = t[better]
                piece[sel[better]] = code
        best[piece < 0] = np.nan

        return best, piece

    def intersect(self, P, d):
        """Distance along the rays to the base plane (ideal model) or to the
        first facet or riser they hit (facets model).

        Parameters
        ----------
        P : (N, 3) array
            Start points relative to the vertex, in the local frame of the surface.
        d : (N, 3) array
            Unit directions.

        Returns
        -------
        t : (N,) array
            NaN for rays that miss.
        """

        if self.model == "ideal":
            with np.errstate(invalid="ignore", divide="ignore"):
                return -P[:, 2]/d[:, 2]

        return self._first_hit(P, d)[0]

    def _piece_at(self, Q):
        """Code of the piece (2k facet, 2k + 1 riser) on which the points Q lie."""

        g = self._geometry()
        K = g["K"]
        r = np.hypot(Q[:, 0], Q[:, 1])
        z = Q[:, 2]
        k = np.clip(np.searchsorted(g["r"], r, side="right") - 1, 0, K - 1)
        cands, res = [], []
        for kk in (k - 1, k, k + 1):
            kc = np.clip(kk, 0, K - 1)
            ok = (kk >= 0) & (kk <= K - 1)
            cands.append(2*kc)
            res.append(np.where(ok, np.abs(z - self.facet_sag(kc, r)), np.inf))
        if K > 1:
            for kk in (k - 1, k):
                kc = np.clip(kk, 0, K - 2)
                ok = (kk >= 0) & (kk <= K - 2)
                if self.draft == 0:
                    rr = np.abs(r - g["r"][kc + 1])
                    lo, hi = np.minimum(0, g["h"][kc]), np.maximum(0, g["h"][kc])
                    rr = np.where((z >= lo - 1e-9) & (z <= hi + 1e-9), rr, np.inf)
                else:
                    rr = np.abs(z - self._riser_line(kc, r))
                    within = (r >= g["r"][kc + 1] - 1e-9) & (r <= g["e"][kc] + 1e-9)
                    rr = np.where(within, rr, np.inf)
                cands.append(2*kc + 1)
                res.append(np.where(ok, rr, np.inf))
        res = np.array(res)
        cands = np.array(cands)
        j = np.argmin(res, axis=0)

        return np.take_along_axis(cands, j[None], axis=0)[0]

    def _normal_piece(self, Q, piece):
        g = self._geometry()
        k = piece//2
        riser = piece % 2 == 1
        r = np.hypot(Q[:, 0], Q[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(r > 0, Q[:, 0]/r, 0.0)
            uy = np.where(r > 0, Q[:, 1]/r, 0.0)
        # dz/dr of the pieces:
        s_par = Surface._slope_over_r(self, r**2)*r
        kz = np.clip(k, 0, g["K"] - 1)
        s = np.where(g["flat"][kz], g["slope"][kz], s_par)
        N = np.stack([-ux*s, -uy*s, np.ones(len(Q))], axis=-1)
        if np.any(riser) and g["K"] > 1:
            kr = np.clip(k, 0, g["K"] - 2)
            if self.draft == 0:
                Nr = np.stack([ux, uy, np.zeros(len(Q))], axis=-1)
            else:
                m = -np.sign(g["h"][kr])/np.tan(self.draft)
                Nr = np.stack([-ux*m, -uy*m, np.ones(len(Q))], axis=-1)
            N = np.where(riser[:, None], Nr, N)

        return N/_norm(N)[:, None]

    def normal(self, Q):
        """Unit normals at the points Q: of the parent (ideal model) or of the
        facet or riser through Q.

        Parameters
        ----------
        Q : (N, 3) array
            Points on the surface, relative to the vertex.

        Returns
        -------
        N : (N, 3) array
            Normals with N_z >= 0 (radial for vertical risers).
        """

        if self.model == "ideal":
            return Surface.normal(self, Q)

        return self._normal_piece(Q, self._piece_at(Q))

    def _refract(self, N, d, n1, n2):
        cos_i = _dot(d, N)
        if self.mirror:
            return d - 2*cos_i[:, None]*N
        sgn = np.sign(cos_i)
        sgn[sgn == 0] = 1.0
        N = N*sgn[:, None]
        cos_i = np.abs(cos_i)
        mu = n1/n2
        with np.errstate(invalid="ignore"):
            cos_t = np.sqrt(1 - mu**2*(1 - cos_i**2))

        return mu*d + (cos_t - mu*cos_i)[:, None]*N

    def _deflect_classified(self, Q, d, n1, n2):
        piece = self._piece_at(Q)
        d2 = self._refract(self._normal_piece(Q, piece), d, n1, n2)
        riser = piece % 2 == 1
        code = np.where(np.all(np.isfinite(d2), axis=-1), FACET, TIR)
        code[riser] = RISER  # refracted or totally reflected at a riser: stray light
        # rays that hit the relief again on their way out of the grooves:
        idx = np.flatnonzero(np.all(np.isfinite(d2), axis=-1))
        again = np.zeros(len(idx), dtype=bool)
        if len(idx):
            eps = 1e-7*self._geometry()["scale"]
            t, p = self._first_hit(Q[idx], d2[idx], t_min=np.full(len(idx), eps))
            again = p >= 0
            code[idx[again & (code[idx] == FACET)]] = SHADOWED
        # (a stray ray from a riser that hits the relief again is lost, too)
        hits_again = np.zeros(len(Q), dtype=bool)
        hits_again[idx[again]] = True

        return d2, code, hits_again

    def deflect(self, Q, d, n1, n2):
        """Refract (or reflect) the rays at the base plane (ideal) or at the
        pieces they hit (facets: lost rays NaN, see the module docstring).

        Parameters
        ----------
        Q : (N, 3) array
            Intersection points in the local frame.
        d : (N, 3) array
            Unit directions of the incident rays.
        n1, n2 : numbers
            Refractive indices in front of and behind the surface.

        Returns
        -------
        d2 : array
            New unit directions.
        dopl : array
            Extra optical path (zero).
        """

        if self.model == "ideal":
            return Surface.deflect(self, Q, d, n1, n2)
        d2, code, again = self._deflect_classified(Q, d, n1, n2)
        lost = again | ~np.all(np.isfinite(d2), axis=-1)
        if self.stray == "lose":
            lost |= code == RISER
        d2 = d2.copy()
        d2[lost] = np.nan

        return d2, np.zeros(len(d))

    def classify(self, P, d, n1, n2):
        """What happens to rays at the surface.

        Parameters
        ----------
        P : (N, 3) array
            Points on the rays in the local frame (in front of the surface).
        d : (N, 3) array
            Unit directions.
        n1, n2 : numbers
            Refractive indices in front of and behind the surface.

        Returns
        -------
        code : (N,) int array
            FACET (refracted by a facet and out), RISER (hits a riser first),
            SHADOWED (refracted by a facet, then hits the relief again), TIR
            (total internal reflection at the first piece) or MISS.
        Q : (N, 3) array
            First hits (NaN for misses).
        """

        P = np.asarray(P, dtype=float).reshape(-1, 3)
        d = np.asarray(d, dtype=float).reshape(-1, 3)
        t = self.intersect(P, d)
        Q = P + t[:, None]*d
        code = np.full(len(P), MISS)
        hit = np.isfinite(t)
        if self.model == "ideal":
            d2 = Surface.deflect(self, Q[hit], d[hit], n1, n2)[0]
            code[hit] = np.where(np.all(np.isfinite(d2), axis=-1), FACET, TIR)
        elif np.any(hit):
            code[hit] = self._deflect_classified(Q[hit], d[hit], n1, n2)[1]

        return code, Q


FacetLosses = namedtuple("FacetLosses", ["px", "py", "code", "r", "fractions"])
FacetLosses.__doc__ = """Fate of the rays at a Fresnel surface (`facet_losses()`).

px, py : arrays
    Pupil coordinates of the rays.
code : int array
    FACET, RISER, SHADOWED, TIR or MISS for each ray (NaN rays in front of
    the surface: MISS).
r : array
    Radius of the first hit on the surface.
fractions : dict
    Fraction of the rays in each class, by name ("facet", "riser",
    "shadowed", "TIR", "miss").
"""


def facet_losses(system, index, field=None, n_rays=20000, px=None, py=None):
    """Classify the rays of a pupil grid at a Fresnel surface of a system.

    Parameters
    ----------
    system : raytrace.System
    index : int
        Index of the FresnelSurface in system.surfaces.
    field : optional
        Field (default: on axis).
    n_rays : int, optional
        Number of rays, on a Vogel (sunflower) spiral over the pupil: uniform
        in area and without the aliasing of a square grid against the narrow
        rings of the risers.
    px, py : arrays, optional
        Pupil coordinates instead of the spiral.

    Returns
    -------
    FacetLosses
    """

    s = system.surfaces[index]
    if not isinstance(s, FresnelSurface):
        raise TypeError("surface {} is not a FresnelSurface".format(index))
    field = 0.0 if field is None else field
    if px is None:
        px, py = vogel_spiral(n_rays)
    px, py = np.broadcast_arrays(np.asarray(px, float).ravel(), np.asarray(py, float).ravel())
    rays = system.rays(field, px, py)
    if index > 0:
        b = system.trace(rays, clip=True, last=index - 1)
        P, d = b.points[-1], b.directions[-1]
    else:
        P, d = np.asarray(rays.points), np.asarray(rays.directions)
    o, R = system.frames()[0][index]
    ok = np.all(np.isfinite(P), axis=-1)
    code = np.full(len(px), MISS)
    r = np.full(len(px), np.nan)
    n = system.n
    if np.any(ok):
        c, Q = s.classify((P[ok] - o) @ R, d[ok] @ R, n[index], n[index + 1])
        code[ok] = c
        r[ok] = np.hypot(Q[:, 0], Q[:, 1])
    fr = {name: float(np.mean(code == c)) for c, name in CLASS_NAMES.items()}

    return FacetLosses(px, py, code, r, fr)


def vogel_spiral(n):
    """n points spread uniformly over the unit disk (Vogel's sunflower spiral).

    Parameters
    ----------
    n : int

    Returns
    -------
    x, y : arrays
    """

    j = np.arange(n) + 0.5
    rho = np.sqrt(j/n)
    phi = j*np.pi*(3 - np.sqrt(5))

    return rho*np.cos(phi), rho*np.sin(phi)


def focusing_slopes(r, focus, n1, n2, d_in=(0.0, 0.0, 1.0), z=0.0):
    """Exact facet slopes dz/dr that send rays through a point on the axis.

    The facet normal is parallel to n1 d - n2 d' (vector law of refraction)
    for the incident direction d and the direction d' from the facet point to
    the focus. With z = 0, these are the slopes of an aberration free ideal
    Fresnel lens (flat base plane).

    Parameters
    ----------
    r : array
        Radii of the facet points (meridional, y = r).
    focus : number
        z of the focus on the axis (local frame), or np.inf/-np.inf for a
        collimated output along +z/-z.
    n1, n2 : numbers
        Indices in front of and behind the surface.
    d_in : (3,) or (N, 3) array, optional
        Incident directions (default: along the axis). A point source on the
        axis at z_s: d_in = (r, z - z_s)/norm, see the notebook.
    z : number or array, optional
        Heights of the facet points.

    Returns
    -------
    slope : array
        dz/dr at r (TIR or no solution: NaN).
    """

    r = np.asarray(r, dtype=float)
    z = np.broadcast_to(np.asarray(z, dtype=float), r.shape)
    d1 = np.broadcast_to(np.asarray(d_in, dtype=float), r.shape + (3,))
    d1 = d1/_norm(d1.reshape(-1, 3)).reshape(r.shape + (1,))
    if np.isinf(focus):
        d2 = np.zeros(r.shape + (3,))
        d2[..., 2] = np.sign(focus)
    else:
        d2 = np.stack([np.zeros_like(r), -r, focus - z], axis=-1)
        d2 = d2/np.linalg.norm(d2, axis=-1, keepdims=True)
    # the facet points lie at (0, r, z), the radial direction is y:
    N = n1*d1 - n2*d2
    Nr, Nz = N[..., 1], N[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = -Nr/Nz
    # check: the refraction must give d2 (no TIR, right sheet):
    Nn = np.stack([np.zeros_like(slope), -slope, np.ones_like(slope)], axis=-1)
    Nn = Nn/np.linalg.norm(Nn, axis=-1, keepdims=True)
    cos_i = np.sum(d1*Nn, axis=-1)
    with np.errstate(invalid="ignore"):
        tir = (n1/n2)**2*(1 - cos_i**2) > 1

    return np.where(tir, np.nan, slope)


def surface_from_slopes(r, slope, n_aspheric=4, conic=0.0, weights=None, **kwargs):
    """Fit a parent surface (sphere or conic plus even aspheric terms) to slopes.

    Least squares fit of dz/dr of `raytrace.Surface` (vertex curvature c and
    the coefficients A4, A6, ... for a fixed conic) to the slopes, e.g. the
    facet angle table of a Fresnel lens or the result of `focusing_slopes()`.

    Parameters
    ----------
    r, slope : arrays
        Radii and slopes dz/dr.
    n_aspheric : int, optional
        Number of aspheric coefficients (A4, ..., A_{2 n_aspheric + 2}).
    conic : number, optional
        Fixed conic constant of the fit.
    weights : array, optional
        Weights of the slopes.
    **kwargs
        Passed on to `raytrace.Surface` (thickness, n_after, material, ...).

    Returns
    -------
    surface : raytrace.Surface
    rms : number
        RMS slope residual.
    """

    from scipy.optimize import least_squares

    r = np.asarray(r, dtype=float)
    slope = np.asarray(slope, dtype=float)
    ok = np.isfinite(slope)
    r, slope = r[ok], slope[ok]
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)[ok]
    r_n = np.max(r)

    def model(p):
        c = p[0]/r_n
        q = 1 - (1 + conic)*c**2*r**2
        s = c*r/np.sqrt(np.maximum(q, 1e-300))
        for j in range(n_aspheric):
            s = s + 2*(j + 2)*p[j + 1]/r_n**(2*j + 3)*r**(2*j + 3)
        return np.where(q > 0, s, 1e3)

    # start: a paraxial fit of the slopes near the axis
    near = r <= 0.3*r_n
    c0 = np.polyfit(r[near], slope[near], 1)[0] if np.sum(near) > 2 else slope[-1]/r[-1]
    p0 = np.zeros(n_aspheric + 1)
    p0[0] = c0*r_n
    res = least_squares(lambda p: w*(model(p) - slope), p0, method="lm", xtol=1e-15,
                        ftol=1e-15, gtol=1e-15, max_nfev=20000)
    p = res.x
    c = p[0]/r_n
    A = tuple(p[j + 1]/r_n**(2*j + 3) for j in range(n_aspheric))
    s = Surface(np.inf if c == 0 else 1/c, conic=conic, aspheric=A, **kwargs)
    rms = float(np.sqrt(np.mean((model(p) - slope)**2)))

    return s, rms


def kinoform_depth(design_wavelength, n, n_out=1.0, order=1):
    """Groove depth of a kinoform: order*lambda_0/(n - n_out).

    Parameters
    ----------
    design_wavelength : number
    n, n_out : numbers
        Indices of the relief and of the medium in the grooves at lambda_0.
    order : int, optional
        Design order (multi-order diffractive lenses: order > 1).

    Returns
    -------
    depth : number
    """

    return order*design_wavelength/abs(n - n_out)


def kinoform_efficiency(order, wavelength, design_wavelength, n, n_design, n_out=1.0,
                        design_order=1):
    """Scalar diffraction efficiency sinc**2(alpha - m) of a kinoform.

    Parameters
    ----------
    order : int or array
        Diffraction order m.
    wavelength, design_wavelength : numbers or arrays
    n, n_design : numbers or arrays
        Index of the relief at wavelength and at the design wavelength.
    n_out : number, optional
        Index in the grooves.
    design_order : int, optional
        Design order (step of design_order waves at the design wavelength).

    Returns
    -------
    eta : array
        Efficiency with alpha = design_order lambda_0 (n - n_out)/((n_0 - n_out) lambda).
    """

    alpha = design_order*design_wavelength*(np.asarray(n) - n_out)/(
        (np.asarray(n_design) - n_out)*np.asarray(wavelength))

    return np.sinc(alpha - np.asarray(order))**2


def diffractive_abbe_number(lambda_d=587.56, lambda_F=486.13, lambda_C=656.27):
    """Abbe number lambda_d/(lambda_F - lambda_C) of a diffractive lens (-3.45).

    Parameters
    ----------
    lambda_d, lambda_F, lambda_C : numbers, optional
        Wavelengths (any common unit).

    Returns
    -------
    V : number
    """

    return lambda_d/(lambda_F - lambda_C)
