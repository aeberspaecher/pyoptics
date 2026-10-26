#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ray traced KrF illuminator: from the secondary source to the reticle.

The illuminator sets the pupil fill (sigma, annular) of a projection lens
and illuminates its field uniformly and telecentrically. `KrFIlluminator`
follows the architecture of the Zeiss illuminators of the 1990s (US
5,675,401 and US 5,982,558) and fits the Caldwell lens
(raytrace.caldwell_krf_lens(): NA 0.56, 4x, a 66 mm square of its 46.8 mm
object field radius):

1. Secondary source (plane 130 of US 5,675,401): a disk of radius
   source_radius that emits uniformly into the direction sines up to
   source_na. For a laser it stands for the beam shaper with a diffuser or
   diffractive element; for a lamp, for the image of the arc. (Invented.)
2. Zoom-axicon objective (raytrace.zeiss_zoom_axicon(), US 5,675,401
   Table 2, adapted to fused silica at 248 nm): maps the source directions
   onto positions in the pupil plane 30; the zoom sets the size of the
   illuminated disk (sigma), the gap between its axicons turns the disk
   into a ring (annular). Ray traced.
3. Diffuser in plane 30 (the "diffracting component 8" of US 5,675,401,
   Fig. 4d): adds directions uniformly within +-diffuser_angle (tangents,
   square) so that the rod entrance is filled. Without it the small spot
   on the rod entrance shows up as a grid of separate spots in the pupil
   (the mirror images of the spot that the rod makes, w/L apart).
   (Invented.)
4. Coupling lens: an ideal lens of focal length coupling_focal_length
   with plane 30 in its front and the rod entrance in its back focal plane,
   so that positions in plane 30 become directions in the rod. (Invented,
   the "objective 4" of US 5,675,401.)
5. Glass rod: a fused silica bar of square cross section rod_width and
   length rod_length; total internal reflection at its sides mixes the
   light (uniform field) and mirrors the directions (the pupil fill becomes
   symmetric under x -> -x and y -> -y). Traced exactly by unfolding the
   reflections. Its exit face carries the masking blades. (Invented: the
   patents use a rod of about 0.5 m.)
6. REMA objective (raytrace.zeiss_krf_rema(), US 5,982,558 Table 1):
   images the rod exit onto the reticle at -4.457x. Ray traced.

The default source NA (0.185) and coupling focal length (60 mm) give sigma
(radius with 95% of the energy) from 0.32 (zoom 0) to 0.83 (zoom 1) at the
reticle side NA 0.14 of the Caldwell lens; the REMA passes up to sigma
0.96. The axicons give rings from narrow ones near sigma 0.8 down to inner
radii of about 0.5 of the outer one.

Coordinates: field points are image (wafer) coordinates in mm, as in
pyoptics.litho.lens; the reticle point is field/magnification (magnification
-0.25 for Caldwell). Sigma coordinates follow pyoptics.litho.illumination:
a source point s illuminates the wafer with the transverse wave vector
k_0*NA*s, i.e. s = -d_xy/(NA*|magnification|) with the reticle side ray
directions d of the (inverting) projection lens.
"""

from collections import namedtuple

import numpy as np

from .illumination import Illumination
from ..raytrace import Rays, fused_silica, zeiss_krf_rema, zeiss_zoom_axicon


ReticleRays = namedtuple("ReticleRays", ["points", "directions", "sigma"])
ReticleRays.__doc__ = """Rays of the illuminator at the reticle.

points : (N, 2) array
    Positions in the reticle plane (mm).
directions : (N, 3) array
    Unit directions behind the reticle plane.
sigma : (N, 2) array
    Sigma coordinates (see the module docstring).
"""


def _fold(u, width):
    """Fold the unfolded coordinate u of a rod with mirror sides at
    +-width/2 back into the rod; returns the coordinate and the sign of the
    direction (-1 after an odd number of reflections)."""

    v = u + width/2
    k = np.floor(v/width)
    v = v - k*width
    odd = (k % 2) != 0

    return np.where(odd, width - v, v) - width/2, np.where(odd, -1.0, 1.0)


class KrFIlluminator(object):
    """Ray traced KrF illuminator with zoom-axicon, glass rod and REMA
    objective, see the module docstring.

    Parameters
    ----------
    zoom : number or pair, optional
        Zoom position(s) of the zoom-axicon in [0, 1] (0: small sigma).
    axicon_gap : number, optional
        Distance of the axicons in mm (0: conventional illumination).
    source_radius : number, optional
        Radius of the secondary source in mm.
    source_na : number, optional
        Largest direction sine of the secondary source.
    diffuser_angle : number, optional
        Half width of the diffuser in plane 30 (tangent of the direction
        added in x and y); default: rod_width/(2*coupling_focal_length), the
        rod entrance just filled. 0: no diffuser.
    coupling_focal_length : number, optional
        Focal length of the coupling lens in mm.
    rod_width, rod_length : numbers, optional
        Square cross section and length of the glass rod in mm.
    NA : number, optional
        Image side NA of the projection lens (sigma = 1).
    magnification : number, optional
        Magnification of the projection lens (reticle to wafer).
    wavelength : number, optional
        Vacuum wavelength in mm.
    n_rays : int, optional
        Rays traced from the source (the pupil fill at a field point uses
        those that arrive within a window around it).
    seed : int, optional
        Seed of the random rays.
    """

    def __init__(self, zoom=1.0, axicon_gap=0.0, source_radius=1.0, source_na=0.185,
                 diffuser_angle=None, coupling_focal_length=60.0, rod_width=14.8, rod_length=500.0, NA=0.56,
                 magnification=-0.25, wavelength=248.38e-6, n_rays=400000, seed=0):
        self.zoom = zoom
        self.axicon_gap = axicon_gap
        self.source_radius = source_radius
        self.source_na = source_na
        self.coupling_focal_length = coupling_focal_length
        if diffuser_angle is None:
            diffuser_angle = rod_width/(2*coupling_focal_length)
        self.diffuser_angle = diffuser_angle
        self.rod_width = rod_width
        self.rod_length = rod_length
        self.NA = NA
        self.magnification = magnification
        self.wavelength = wavelength
        self.n_rays = n_rays
        self.seed = seed
        self.rema = zeiss_krf_rema(wavelength)
        self._rod_exit = None

    def __repr__(self):
        return ("KrFIlluminator(zoom={}, axicon_gap={}, source_na={}, coupling_focal_length={}, "
                "rod={} x {} mm)".format(self.zoom, self.axicon_gap, self.source_na,
                                         self.coupling_focal_length, self.rod_width,
                                         self.rod_length))

    @property
    def zoom_axicon(self):
        """The zoom-axicon objective (raytrace.System) of the current setting."""

        return zeiss_zoom_axicon(self.zoom, self.axicon_gap, self.wavelength,
                                 source_radius=self.source_radius)

    @property
    def NA_reticle(self):
        """Reticle side NA of the projection lens (sigma = 1)."""

        return self.NA*abs(self.magnification)

    @property
    def rema_magnification(self):
        return self.rema.paraxial().magnification

    def set(self, zoom=None, axicon_gap=None):
        """Change the zoom and/or the axicon gap.

        Parameters
        ----------
        zoom : number or pair, optional
        axicon_gap : number, optional

        Returns
        -------
        KrFIlluminator
            self
        """

        if zoom is not None:
            self.zoom = zoom
        if axicon_gap is not None:
            self.axicon_gap = axicon_gap
        self._rod_exit = None

        return self

    # tracing:

    def _source_rays(self, n, rng):
        r = self.source_radius*np.sqrt(rng.random(n))
        phi = 2*np.pi*rng.random(n)
        q = self.source_na*np.sqrt(rng.random(n))
        psi = 2*np.pi*rng.random(n)
        sx, sy = q*np.cos(psi), q*np.sin(psi)
        P = np.stack([r*np.cos(phi), r*np.sin(phi),
                      np.full(n, -self.zoom_axicon.object_distance)], axis=-1)
        d = np.stack([sx, sy, np.sqrt(1 - sx**2 - sy**2)], axis=-1)

        return Rays(P, d, np.zeros(n))

    def pupil_plane_rays(self, n_rays=None):
        """Rays in the pupil plane 30 behind the zoom-axicon.

        Parameters
        ----------
        n_rays : int, optional
            Default: self.n_rays.

        Returns
        -------
        points, directions : (N, 3) arrays
            Of the rays that pass (the zoom-axicon has no clear apertures).
        """

        rng = np.random.default_rng(self.seed)
        n = self.n_rays if n_rays is None else n_rays
        b = self.zoom_axicon.trace(self._source_rays(n, rng), clip=False)

        return b.points[-1][b.valid], b.directions[-1][b.valid]

    def rod_exit_rays(self, n_rays=None):
        """Rays at the exit face of the rod (the object plane of the REMA).

        Parameters
        ----------
        n_rays : int, optional
            Default: self.n_rays.

        Returns
        -------
        points : (N, 2) array
            Positions on the exit face (mm, centre of the rod at 0).
        directions : (N, 3) array
            Directions in air behind the exit face.
        transmission : float
            Fraction of the source rays that enter the rod.
        """

        if n_rays is None and self._rod_exit is not None:
            return self._rod_exit
        P, d = self.pupil_plane_rays(n_rays)
        n_traced = self.n_rays if n_rays is None else n_rays
        f = self.coupling_focal_length
        tan = d[:, :2]/d[:, 2:3]
        if self.diffuser_angle:
            rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
            tan = tan + self.diffuser_angle*(2*rng.random(tan.shape) - 1)
        # ideal coupling lens f behind plane 30: a ray of direction tan ends
        # up at f*tan in the back focal plane (the rod entrance), with the
        # direction from its point in the lens plane to there:
        E = f*tan
        Q = P[:, :2] + f*tan
        v = np.concatenate([E - Q, np.full((len(E), 1), f)], axis=1)
        d_in = v/np.linalg.norm(v, axis=1)[:, None]
        w = self.rod_width
        inside = np.all(np.abs(E) <= w/2, axis=1)
        E, d_in = E[inside], d_in[inside]
        # in the glass the transverse direction sines are divided by n; the
        # side faces reflect totally (n sin > 1 there for all rays of NA < 0.9):
        n = fused_silica(self.wavelength*1e6)
        s = d_in[:, :2]/n
        s_z = np.sqrt(1 - np.sum(s**2, axis=1))
        X, sign_x = _fold(E[:, 0] + self.rod_length*s[:, 0]/s_z, w)
        Y, sign_y = _fold(E[:, 1] + self.rod_length*s[:, 1]/s_z, w)
        out = d_in.copy()
        out[:, 0] *= sign_x
        out[:, 1] *= sign_y
        result = (np.stack([X, Y], axis=-1), out, inside.sum()/n_traced)
        if n_rays is None:
            self._rod_exit = result

        return result

    def reticle_rays(self, field=None, window=1.0, n_rays=None):
        """Rays at the reticle, traced through the REMA objective.

        Parameters
        ----------
        field : pair, optional
            Image (wafer) field point in mm; None: the whole field.
        window : number, optional
            Half width (wafer scale, mm) of the square around the field point
            within which the rays are taken.
        n_rays : int, optional
            Rays from the source; default self.n_rays.

        Returns
        -------
        ReticleRays
        """

        X, d, _ = self.rod_exit_rays(n_rays)
        m = self.rema_magnification
        if field is not None:
            center = np.asarray(field, dtype=float)/self.magnification  # on the reticle
            half = window/abs(self.magnification)
            # preselect at the rod exit (REMA distortion < 0.5%):
            c_rod = center/m
            pre = np.all(np.abs(X - c_rod) <= half/abs(m)*1.05 + 0.005*np.abs(c_rod) + 1e-3,
                         axis=1)
            X, d = X[pre], d[pre]
        z0 = -self.rema.object_distance
        rays = Rays(np.concatenate([X, np.full((len(X), 1), z0)], axis=1), d, np.zeros(len(X)))
        b = self.rema.trace(rays, clip=True)
        R, D = b.points[-1][b.valid][:, :2], b.directions[-1][b.valid]
        if field is not None:
            keep = np.all(np.abs(R - center) <= half, axis=1)
            R, D = R[keep], D[keep]
        sigma = -D[:, :2]/self.NA_reticle

        return ReticleRays(R, D, sigma)

    # results:

    def pupil_fill(self, field=(0.0, 0.0), window=1.0, n_samples=41, polarization="unpolarized",
                   n_rays=None):
        """Pupil fill at a field point as an `Illumination` (histogram of the
        sigma coordinates of the rays on an n_samples grid across the unit
        circle).

        Parameters
        ----------
        field : pair, optional
            Image (wafer) field point in mm.
        window : number, optional
            Half width (wafer scale, mm) of the field window, see reticle_rays().
        n_samples : int, optional
            Grid points across the sigma range [-1, 1].
        polarization : optional
            See pyoptics.litho.illumination.
        n_rays : int, optional
            Rays from the source; default self.n_rays.

        Returns
        -------
        Illumination
        """

        s = self.reticle_rays(field, window, n_rays).sigma
        edges = np.linspace(-1, 1, n_samples + 1)
        H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[edges, edges])
        c = (edges[1:] + edges[:-1])/2
        S_x, S_y = np.meshgrid(c, c, indexing="ij")

        return Illumination.from_map(S_x, S_y, H, polarization=polarization,
                                     name="KrF illuminator")

    def sigma(self, field=None, fractions=(0.05, 0.95), window=1.0, n_rays=None):
        """Radii of the pupil fill that enclose the given fractions of the
        energy, e.g. (sigma_in, sigma_out).

        Parameters
        ----------
        field : pair, optional
            Image field point in mm; None (default): all rays of the field.
        fractions : sequence, optional
            Energy fractions.
        window : number, optional
            See reticle_rays().
        n_rays : int, optional

        Returns
        -------
        array
        """

        s = self.reticle_rays(field, window, n_rays).sigma
        if len(s) == 0:  # e.g. a ring beyond the stop of the REMA
            return np.full(np.shape(fractions), np.nan)

        return np.quantile(np.hypot(s[:, 0], s[:, 1]), fractions)

    def centroid(self, field=(0.0, 0.0), window=1.0, n_rays=None):
        """Centroid of the reticle side ray directions at a field point (the
        energetic telecentricity of the illumination), direction sines
        (d_x, d_y).

        Parameters
        ----------
        field : pair, optional
        window : number, optional
        n_rays : int, optional

        Returns
        -------
        array (2,)
        """

        return self.reticle_rays(field, window, n_rays).directions[:, :2].mean(axis=0)

    def telecentricity(self, lens, field=(0.0, 0.0), window=1.0, n_rays=None):
        """Mismatch between the illumination and a projection lens: the
        centroid of the illumination directions at the reticle minus the
        direction of the lens' object side chief ray, in direction sines
        (radians for small values). In sigma units it is the shift of the
        pupil fill relative to the pupil of the lens, divided by NA_reticle.

        Parameters
        ----------
        lens : pyoptics.raytrace.System
            Projection lens (object = reticle), e.g. caldwell_krf_lens().
        field : pair, optional
            Image field point in mm.
        window : number, optional
        n_rays : int, optional

        Returns
        -------
        array (2,)
        """

        obj = np.asarray(field, dtype=float)/self.magnification
        b = lens.trace_pupil(field=tuple(obj), px=0.0, py=0.0)
        chief = b.directions[0, 0, :2]

        return self.centroid(field, window, n_rays) - chief

    def uniformity(self, n_bins=8, n_rays=None):
        """Illumination of the field: ray counts on n_bins x n_bins cells of
        the reticle image of the rod and the non uniformity
        (max - min)/(max + min).

        Parameters
        ----------
        n_bins : int, optional
        n_rays : int, optional

        Returns
        -------
        counts : (n_bins, n_bins) array
        non_uniformity : float
        """

        R = self.reticle_rays(None, n_rays=n_rays).points
        half = 0.995*self.rod_width/2*abs(self.rema_magnification)
        H, _, _ = np.histogram2d(R[:, 0], R[:, 1], bins=n_bins, range=[[-half, half]]*2)

        return H, (H.max() - H.min())/(H.max() + H.min())

    def for_sigma(self, sigma_out, sigma_in=None, fractions=(0.05, 0.95), n_rays=40000):
        """Set zoom and axicon gap for a pupil fill.

        Conventional (sigma_in None): the zoom (both elements together) for
        sigma_out at gap 0; annular: zoom and gap. sigma_in and sigma_out
        are the radii that enclose the energy fractions.

        Parameters
        ----------
        sigma_out : number
        sigma_in : number, optional
        fractions : pair, optional
        n_rays : int, optional
            Rays per trial (sigma from the rod exit directions).

        Returns
        -------
        KrFIlluminator
            self, with the new setting.
        """

        from scipy.optimize import brentq, least_squares

        k = 1/(abs(self.rema_magnification)*self.NA_reticle)

        def sig(zoom, gap):
            self.set(zoom, gap)
            d = self.rod_exit_rays(n_rays)[1]
            return np.quantile(np.hypot(d[:, 0], d[:, 1])*k, fractions)

        if sigma_in is None:
            lo, hi = sig(0.0, 0.0)[1], sig(1.0, 0.0)[1]
            if not lo <= sigma_out <= hi:
                raise ValueError("sigma_out must lie in [{:.3f}, {:.3f}] for this source"
                                 .format(lo, hi))
            zoom = brentq(lambda z: sig(z, 0.0)[1] - sigma_out, 0.0, 1.0, xtol=1e-4)
            return self.set(zoom, 0.0)

        def residual(x):
            z = np.clip(x[0], 0, 1)
            return sig(z, max(x[1], 0.0)) - np.array([sigma_in, sigma_out])

        sol = least_squares(residual, [0.5, 20.0], bounds=([0, 0], [1, 165.0]),
                            diff_step=[0.02, 0.02])

        return self.set(float(sol.x[0]), float(sol.x[1]))
