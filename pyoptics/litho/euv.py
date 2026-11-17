#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""EUV lithography at 13.5 nm: Mo/Si multilayer mirrors, reflective masks
lit at an oblique chief ray, the mask-side geometry of NA 0.33 and of the
anamorphic high-NA projector, centrally obscured pupils and the numbers of
the EUV photon budget (notebooks 156 (EUV multilayers and reflective masks)
to 158 (EUV stochastics and flare)).

Every material absorbs at 13.5 nm and all refractive indices are close to
1 (n = 1 - delta + i*beta with delta and beta of 1e-3 to 1e-1), so EUV
optics are mirrors with multilayer coatings: about 40 pairs of molybdenum
and silicon with a period of about 6.9 nm, which reflect through
constructive interference of the weak reflections at many interfaces
(Bragg reflection), with a peak reflectance of about 70 %.

- `INDEX_13_5`: optical constants at 13.5 nm (Henke tables, as given by
  CXRO) of the usual mirror, capping, absorber and substrate materials.
- `Multilayer`: a periodic Mo/Si stack with an optional capping layer and
  interlayers: amplitude reflection coefficients (s and p) over the angle
  and the wavelength (`pyoptics.thin_films`), reflectance, bandwidth,
  the depth of the effective reflection plane, a design of the period for
  a wavelength and an angle, and the stack as a `coatings.Coating` for the
  polarization ray trace of mirror systems.
- `ReflectiveMask`: a mask stack for `LithoSimulation` that computes the
  reflected diffraction orders of a patterned absorber on a multilayer
  with RCWA (`litho.rcwa`) for every source point, at a chief ray angle
  (CRA) of incidence of 6 degrees in the y-z plane by default. Any reduction
  ratio, also an anamorphic one (M_x, M_y), sets the angles at the mask.
- `min_cra()`, `mask_cones()`: the chief ray angle that keeps the cones of
  illumination and imaging at the mask apart, and the cones themselves.
- `obscured_pupil()`: a pupil with a central obscuration as an
  apodization for `Projector`.
- `z_factor()`: the resolution, LWR and sensitivity (RLS) trade-off.

Geometry of the mask. The mask lies in the x-y plane, the light comes from
above (z < 0) and is reflected back. The chief ray of the illumination has
the direction (0, sin(CRA), cos(CRA)) (plane of incidence y-z; the scan
direction of scanners is y); the reflected chief ray (0, sin(CRA),
-cos(CRA)) is the axis of the projection optics. A source point s and a
diffraction order (m, n) of a layout with the wafer-scale pitches p_x, p_y
leave the mask with the transverse direction cosines ::

    (NA*s_x/M_x + m*lambda/(M_x*p_x),  sin(CRA) + NA*s_y/M_y + n*lambda/(M_y*p_y)),

so in the pupil (normalized to the NA, relative to the chief ray) they land
where a transmission mask would put them: the projector and the image
sum of `LithoSimulation` stay the same, only the mask diffraction differs.
Lines along x (pitch along y, across the plane of incidence) see the
oblique light across their edges, which throw shadows; lines along y see it
along their length.

Normalization: the reflected orders are divided by the reflection of the
bare multilayer at the chief ray (a complex number), so a clear mask
reflects about 1 near the chief ray. Its reflectance still varies over the
source points (the angular bandwidth of the multilayer), and
`LithoSimulation` normalizes images to that open frame.

Convention: fields vary as exp(i(k.r - omega*t)), absorbing materials have
n + i*k with k > 0; amplitude coefficients are ratios of the tangential
electric fields (as in `pyoptics.thin_films`, r_p = r_s at normal
incidence). Lengths in nm.

References
----------
B. L. Henke, E. M. Gullikson and J. C. Davis, "X-ray interactions:
photoabsorption, scattering, transmission, and reflection at E = 50-30,000
eV, Z = 1-92", Atomic Data and Nuclear Data Tables 54, 181-342 (1993);
https://henke.lbl.gov/optical_constants/.

D. Attwood and A. Sakdinawat, "X-Rays and Extreme Ultraviolet Radiation",
2nd ed., Cambridge University Press (2017).

E. Spiller, "Soft X-Ray Optics", SPIE Press (1994).

A. Erdmann, "Optical and EUV Lithography: A Modeling Perspective", SPIE Press
(2021), ch. 6.

A. Erdmann, P. Evanschitzky, F. Shao, T. Fühner, G. F. Lorusso, E.
Hendrickx, M. Goethals, R. Jonckheere, T. Bret and T. Hofmann,
"Predictive modeling of EUV-lithography: the role of mask, optics, and
photoresist effects", Proc. SPIE 8886, 888602 (2013).

J. van Schoot, E. van Setten, K. Troost, S. Lok, J. Stoeldraijer, R. Peeters,
J. Benschop and J. Zimmerman, "High-NA EUV lithography exposure tool:
advantages and program progress", Proc. SPIE 11517, 1151712 (2021).

P. Naulleau, C. N. Anderson, L.-M. Baclea-an, D. Chan, P. Denham,
S. George, K. A. Goldberg, B. Hoef, G. Jones, C. Koh, B. La Fontaine, B.
McClinton, R. Miyakawa, W. Montgomery, S. Rekawa and T. Wallow, "The SEMATECH
Berkeley microfield exposure tool", Proc. SPIE 7636, 76361J (2010).

T. Wallow, C. Higgins, R. Brainard, K. Petrillo, W. Montgomery, C.-S. Koay,
G. Denbeaux, O. Wood and Y. Wei, "Evaluation of EUV resist materials for use
at the 32 nm half-pitch node", Proc. SPIE 6921, 69211F (2008) (Z-factor).
"""

import numpy as np
from scipy.optimize import minimize_scalar

from ..thin_films import _r_t
from .rcwa import Layer, rcwa, _transposed


#: Wavelength of EUV lithography (nm).
EUV_WAVELENGTH = 13.5

#: Refractive indices n + i*k at 13.5 nm (Henke/CXRO tables for the bulk
#: densities). "low-n" is a model low-n absorber (about the n and k of RuTa
#: or PtMo alloys); TaBO is the oxide anti-reflection layer on TaBN.
INDEX_13_5 = {
    "Mo": 0.9238 + 0.00644j,
    "Si": 0.9990 + 0.00183j,
    "Ru": 0.8864 + 0.01706j,
    "MoSi2": 0.9689 + 0.00434j,
    "B4C": 0.9637 + 0.00517j,
    "SiO2": 0.9780 + 0.01080j,
    "TaBN": 0.9490 + 0.03100j,
    "TaBO": 0.9520 + 0.02600j,
    "Ta": 0.9430 + 0.04080j,
    "Ni": 0.9483 + 0.07270j,
    "low-n": 0.9000 + 0.03000j,
    "vacuum": 1.0 + 0.0j,
}


def index(material):
    """Complex refractive index at 13.5 nm: a name of INDEX_13_5 or a
    number."""

    if isinstance(material, str):
        return INDEX_13_5[material]

    return complex(material)


def fwhm(x, y):
    """Full width at half maximum of a single peak sampled at x (linear
    interpolation between the samples; nan if a half maximum point is
    outside x)."""

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    h = y[i]/2
    left = np.nonzero(y[:i] < h)[0]
    right = np.nonzero(y[i:] < h)[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan
    a, b = left[-1], i + right[0]
    x_l = x[a] + (h - y[a])*(x[a + 1] - x[a])/(y[a + 1] - y[a])
    x_r = x[b - 1] + (h - y[b - 1])*(x[b] - x[b - 1])/(y[b] - y[b - 1])

    return x_r - x_l


class Multilayer(object):
    """Periodic multilayer mirror (Mo/Si by default) with a capping layer.

    The stack from the top: the capping layer, then `pairs` periods of
    (spacer, absorber), i.e. Si then Mo, optionally with interlayers
    (interdiffused MoSi2) at both interfaces of each period, then the
    substrate.

    Parameters
    ----------
    pairs : int, optional
        Number of periods.
    period : number, optional
        Period d in nm (6.95 nm: the peak at 13.5 nm for 6 degrees).
    gamma : number, optional
        Fraction of the period taken by the absorber (Mo).
    capping : (material, thickness) or None, optional
        Capping layer (Ru, 2.5 nm by default, against oxidation).
    interlayer : number, optional
        Thickness (nm) of the interlayer at each interface, taken from the
        neighbouring layers half and half (the period stays the same).
    absorber, spacer, interlayer_material, substrate : material, optional
        Names of INDEX_13_5 or complex indices.
    """

    def __init__(self, pairs=40, period=6.95, gamma=0.4, capping=("Ru", 2.5), interlayer=0.0,
                 absorber="Mo", spacer="Si", interlayer_material="MoSi2", substrate="SiO2"):
        self.pairs, self.period, self.gamma = int(pairs), float(period), float(gamma)
        self.capping = capping
        self.interlayer = float(interlayer)
        self.absorber, self.spacer = absorber, spacer
        self.interlayer_material, self.substrate = interlayer_material, substrate

    def with_(self, **kwargs):
        """Copy with some parameters exchanged."""

        p = dict(vars(self))
        p.update(kwargs)

        return Multilayer(**p)

    def __repr__(self):
        return "Multilayer({} x {:.3f} nm, gamma={:.2f}, capping={})".format(
            self.pairs, self.period, self.gamma, self.capping)

    @property
    def substrate_index(self):
        return index(self.substrate)

    def layers(self):
        """Layers from the top as a list of (complex index, thickness)."""

        out = []
        if self.capping is not None and self.capping[1] > 0:
            out.append((index(self.capping[0]), float(self.capping[1])))
        d_abs, d_sp = self.gamma*self.period, (1 - self.gamma)*self.period
        il = self.interlayer
        if il > 0:
            if il >= min(d_abs, d_sp):
                raise ValueError("interlayers thicker than the layers")
            cell = [(index(self.spacer), d_sp - il), (index(self.interlayer_material), il),
                    (index(self.absorber), d_abs - il), (index(self.interlayer_material), il)]
        else:
            cell = [(index(self.spacer), d_sp), (index(self.absorber), d_abs)]

        return out + cell*self.pairs

    @property
    def thickness(self):
        return sum(t for _, t in self.layers())

    def _stack(self, n_in):
        lay = self.layers()
        n = np.array([n_in] + [n for n, _ in lay] + [self.substrate_index], dtype=complex)

        return n, [t for _, t in lay]

    def amplitudes(self, angle, wavelength=EUV_WAVELENGTH, n_in=1.0):
        """Amplitude reflection coefficients r_s and r_p.

        Parameters
        ----------
        angle : number or array
            Angle of incidence in radians.
        wavelength : number or array, optional
            Vacuum wavelength in nm (the optical constants stay those of
            13.5 nm).
        n_in : number, optional
            Medium above the stack.

        Returns
        -------
        r_s, r_p : complex arrays
            Broadcast shape of angle and wavelength.
        """

        angle, wavelength = np.broadcast_arrays(np.asarray(angle, dtype=float),
                                                np.asarray(wavelength, dtype=float))
        n, t = self._stack(n_in)
        r_s, r_p = np.empty(angle.shape, complex), np.empty(angle.shape, complex)
        sin_a = np.sin(angle)
        for lam in np.unique(wavelength):
            sel = wavelength == lam
            r_s[sel] = _r_t(sin_a[sel], n, t, "TE", lam)[0]
            r_p[sel] = _r_t(sin_a[sel], n, t, "TM", lam)[0]

        return r_s[()], r_p[()]

    def reflectance(self, angle, wavelength=EUV_WAVELENGTH, pol="unpolarized", n_in=1.0):
        """Reflectance for "s", "p" or unpolarized light (the mean)."""

        r_s, r_p = self.amplitudes(angle, wavelength, n_in)
        R = {"s": np.abs(r_s)**2, "p": np.abs(r_p)**2,
             "unpolarized": (np.abs(r_s)**2 + np.abs(r_p)**2)/2}

        return R[pol]

    def bandwidth(self, angle=np.radians(6.0), wavelengths=None, pol="unpolarized"):
        """FWHM (nm) of the reflectance over the wavelength at an angle."""

        wavelengths = np.linspace(12.5, 14.5, 801) if wavelengths is None else wavelengths

        return fwhm(wavelengths, self.reflectance(angle, wavelengths, pol))

    def effective_depth(self, angle=np.radians(6.0), wavelength=EUV_WAVELENGTH, pol="s",
                        delta=1e-4):
        """Depth below the top of the stack of the plane mirror that has the
        same change of the reflection phase with the angle.

        A mirror at the depth z adds 2*k*z*cos(theta) to the phase, so
        z_eff = (d phi/d cos(theta))/(2*k); for Mo/Si it lies about 40 to 50
        nm below the surface, which is where the image of a reflective mask
        appears to come from.
        """

        a = np.array([angle - delta, angle + delta])
        r = self.amplitudes(a, wavelength)[0 if pol == "s" else 1]
        dphi = np.angle(r[1]/r[0])
        dcos = np.cos(a[1]) - np.cos(a[0])

        return dphi/dcos/(2*2*np.pi/wavelength)

    @classmethod
    def design(cls, wavelength=EUV_WAVELENGTH, angle=np.radians(6.0), bounds=(6.0, 8.0), **kwargs):
        """Multilayer whose period puts the reflectance peak at the wavelength
        and angle (other parameters as in Multilayer)."""

        def neg_R(d):
            return -cls(period=d, **kwargs).reflectance(angle, wavelength)

        d0 = wavelength/(2*np.cos(angle))
        grid = np.linspace(bounds[0], bounds[1], 81)
        grid = grid[np.abs(grid - d0) < 1.0] if np.any(np.abs(grid - d0) < 1.0) else grid
        best = grid[int(np.argmin([neg_R(d) for d in grid]))]
        step = grid[1] - grid[0]
        res = minimize_scalar(neg_R, bounds=(best - step, best + step), method="bounded",
                              options={"xatol": 1e-5})

        return cls(period=float(res.x), **kwargs)

    def coating(self, profile=None, scale=1e-6):
        """The stack as a coatings.Coating for the mirrors of a
        raytrace.System.

        Parameters
        ----------
        profile : callable, optional
            Relative thickness f(x, y) of all layers on the local surface
            coordinates (a graded multilayer), e.g. coatings.RadialProfile.
        scale : number, optional
            Factor from nm to the length unit of the system (1e-6 for mm,
            1 for nm).

        Returns
        -------
        coatings.Coating
        """

        from ..coatings import Coating, Layer as CLayer

        return Coating([CLayer(n, t*scale, profile) for n, t in self.layers()],
                       substrate=self.substrate_index, label=repr(self))

    def jones(self, sin_x, sin_y, wavelength=EUV_WAVELENGTH):
        """Jones matrices of the reflection (tangential fields E_x, E_y) for
        the transverse direction cosines (sin_x, sin_y) of the incident
        wave: r_s along the s direction, r_p along the p direction.

        Returns
        -------
        J : array, shape np.shape(sin_x) + (2, 2)
        """

        sin_x, sin_y = np.broadcast_arrays(np.asarray(sin_x, float), np.asarray(sin_y, float))
        s = np.hypot(sin_x, sin_y)
        r_s, r_p = self.amplitudes(np.arcsin(np.clip(s, 0, 1)), wavelength)
        r_s, r_p = np.asarray(r_s), np.asarray(r_p)
        safe = s > 1e-12
        c = np.where(safe, sin_x/np.where(safe, s, 1), 1.0)
        d = np.where(safe, sin_y/np.where(safe, s, 1), 0.0)
        # p (in the plane of incidence) along (c, d), s along (-d, c)
        J = np.empty(sin_x.shape + (2, 2), dtype=complex)
        J[..., 0, 0] = r_p*c*c + r_s*d*d
        J[..., 1, 1] = r_p*d*d + r_s*c*c
        J[..., 0, 1] = J[..., 1, 0] = (r_p - r_s)*c*d

        return J


def _magnifications(magnification):
    M = np.broadcast_to(np.asarray(magnification, dtype=float), (2,))

    return float(M[0]), float(M[1])


class ReflectiveMask(object):
    """EUV mask: a patterned absorber on a multilayer, lit at an oblique chief
    ray; the reflected orders come from RCWA for every source point (see the
    module documentation for the geometry and the normalization).

    Use it as the stack of a LithoSimulation (1D layouts: lines along x or y)
    with a Projector of wavelength 13.5 nm; the magnification of the
    projector may be a pair (M_x, M_y) for an anamorphic projector.

    Parameters
    ----------
    absorber : sequence of (material, thickness) or (material, clear, thickness)
        Absorber layers from the top (e.g. [("TaBO", 2.0), ("TaBN", 58.0)]);
        clear is the material of the layer between the features (vacuum).
    multilayer : Multilayer, optional
        The blank below the absorber (its capping layer included).
    cra : number, optional
        Chief ray angle of incidence in degrees (6 for NA 0.33, 5.355 for the
        anamorphic NA 0.55 projector).
    n_orders : int, optional
        Orders -n_orders..n_orders of the RCWA (default: about twice those
        that propagate).
    factorization : "li" or "laurent", optional
    normalize : bool, optional
        Divide by the reflection of the bare multilayer at the chief ray.
    name : str, optional
    """

    def __init__(self, absorber=(("TaBN", 60.0),), multilayer=None, cra=6.0, n_orders=None,
                 factorization="li", normalize=True, name=None):
        self.absorber = []
        for layer in absorber:
            if len(layer) == 2:
                layer = (layer[0], "vacuum", layer[1])
            self.absorber.append((index(layer[0]), index(layer[1]), float(layer[2])))
        self.multilayer = Multilayer() if multilayer is None else multilayer
        self.cra = float(cra)
        self.n_orders, self.factorization = n_orders, factorization
        self.normalize = normalize
        self.boundary_layer = None
        self.name = name or "reflective mask"
        self._cache = {}

    cache_size = 16

    def with_(self, **kwargs):
        """Copy with some parameters exchanged (absorber as a list of
        (n_abs, n_clear, thickness))."""

        p = dict(absorber=self.absorber, multilayer=self.multilayer, cra=self.cra,
                 n_orders=self.n_orders, factorization=self.factorization,
                 normalize=self.normalize, name=self.name)
        p.update(kwargs)

        return ReflectiveMask(**p)

    def __repr__(self):
        return "ReflectiveMask({}, {:.0f} nm absorber, CRA {:.2f} deg)".format(
            self.name, self.thickness, self.cra)

    @property
    def thickness(self):
        """Absorber thickness."""

        return sum(t for _, _, t in self.absorber)

    def _blank_layers(self):
        return [Layer(t, n) for n, t in self.multilayer.layers()]

    def incidence(self, s_x, s_y, NA, magnification):
        """Transverse direction cosines (k_x, k_y)/k_0 of the incident waves
        of the source points at the mask."""

        M_x, M_y = _magnifications(magnification)

        return (NA*np.asarray(s_x, dtype=float)/M_x,
                np.sin(np.radians(self.cra)) + NA*np.asarray(s_y, dtype=float)/M_y)

    def _reference(self, wavelength):
        """Reflection of the clear area (the blank with the clear material of
        the absorber layers on top) at the chief ray, a complex scalar."""

        if not self.normalize:
            return 1.0
        r = self._films(0.0, np.sin(np.radians(self.cra)), wavelength, "clear")[0]

        return (r[0, 0] + r[1, 1])/2

    def _films(self, k_x, k_y, wavelength, region):
        if region == "clear":
            top = [Layer(t, n_c) for _, n_c, t in self.absorber]
        else:
            top = [Layer(t, n_a) for n_a, _, t in self.absorber]
        res = rcwa(1.0, top + self._blank_layers(), wavelength, k_x, k_y,
                   n_out=self.multilayer.substrate_index, n_orders=0)

        return res.r[:, 0]

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the unpatterned clear blank and of the absorber
        film on it, for the source points (the open frame and a film
        approximation).

        Returns
        -------
        J_clear, J_absorber : arrays, shape np.shape(s_x) + (2, 2)
        """

        shape = np.shape(s_x)
        k_x, k_y = self.incidence(np.ravel(s_x), np.ravel(s_y), NA, magnification)
        r0 = self._reference(wavelength)
        J_c = self._films(k_x, k_y, wavelength, "clear")/r0
        J_a = self._films(k_x, k_y, wavelength, "absorber")/r0

        return J_c.reshape(shape + (2, 2)), J_a.reshape(shape + (2, 2))

    def film_reflectance(self, wavelength=EUV_WAVELENGTH):
        """Reflectances (clear, absorber) at the chief ray, unpolarized."""

        k_y = np.sin(np.radians(self.cra))
        out = []
        for region in ("clear", "absorber"):
            r = self._films(0.0, k_y, wavelength, region)[0]
            out.append(float(np.sum(np.abs(r)**2)/2))

        return tuple(out)

    def _rcwa_layers(self, layout, M):
        segments = [(M*x0, M*a) for x0, y0, a, b in layout.features]
        absorber_features = layout.tone == "clear"
        out = []
        for n_abs, n_clear, t in self.absorber:
            n_feat, n_field = (n_abs, n_clear) if absorber_features else (n_clear, n_abs)
            out.append(Layer(t, n_field, [(c, w, n_feat) for c, w in segments]))

        return out + self._blank_layers()

    def order_jones(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the reflected diffraction orders of a 1D layout.

        Parameters
        ----------
        layout : MaskLayout
            1D layout (lines along x or y), wafer scale.
        m, n : int or arrays
            Order indices in x and y.
        s_x, s_y : numbers or arrays
            Source point coordinates (sigma).
        NA : number
            Numerical aperture (image side).
        wavelength : number
        magnification : number or (M_x, M_y)

        Returns
        -------
        J : array
            Shape (len(s_x), 2, 2, len(n), len(m)).
        """

        if not layout.is_1d:
            raise ValueError("ReflectiveMask supports 1D layouts (lines along x or y) only")
        s_x, s_y = np.atleast_1d(np.asarray(s_x, dtype=float)), np.atleast_1d(np.asarray(s_y, dtype=float))
        key = (layout.pitch_x, layout.pitch_y, tuple(map(tuple, layout.features)), layout.tone,
               tuple(np.atleast_1d(m)), tuple(np.atleast_1d(n)), s_x.tobytes(), s_y.tobytes(), NA,
               wavelength, _magnifications(magnification))
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self._order_jones(layout, m, n, s_x, s_y, NA, wavelength,
                                                 magnification)

        return self._cache[key]

    def _order_jones(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        axis = 0 if layout.pitch_x is not None else 1
        orders = np.atleast_1d(m if axis == 0 else n)
        M = _magnifications(magnification)[axis]
        period = M*(layout.pitch_x if axis == 0 else layout.pitch_y)
        k_x, k_y = self.incidence(s_x, s_y, NA, magnification)
        k_par, k_perp = (k_x, k_y) if axis == 0 else (k_y, k_x)
        n_orders = self.n_orders
        if n_orders is None:
            reach = 1.0 + np.max(np.abs(k_par))
            n_orders = max(10, 2*int(np.ceil(period*reach/wavelength)))
        n_orders = max(n_orders, int(np.abs(orders).max()))
        geometry = layout if axis == 0 else _transposed(layout)
        res = rcwa(period, self._rcwa_layers(geometry, M), wavelength, k_par, k_perp,
                   n_in=1.0, n_out=self.multilayer.substrate_index, n_orders=n_orders,
                   factorization=self.factorization)
        r = res.r[:, orders + n_orders]/self._reference(wavelength)  # (B, orders, 2, 2)
        if axis == 1:  # back from the transposed geometry: swap x and y components
            r = r[..., ::-1, ::-1]
            return r.transpose(0, 2, 3, 1)[..., :, None]

        return r.transpose(0, 2, 3, 1)[..., None, :]

    def kirchhoff(self, wavelength=EUV_WAVELENGTH):
        """The thin (Kirchhoff) mask with the same film reflections at the
        chief ray: a mask.ThinMask with the absorber amplitude relative to
        the clear blank."""

        from .mask import ThinMask

        k_y = np.sin(np.radians(self.cra))
        r_c = self._films(0.0, k_y, wavelength, "clear")[0]
        r_a = self._films(0.0, k_y, wavelength, "absorber")[0]
        t = (r_a[0, 0] + r_a[1, 1])/(r_c[0, 0] + r_c[1, 1])

        return ThinMask(t_absorber=complex(t), name="Kirchhoff " + self.name)


def shadow_width(thickness, cra=6.0):
    """Geometric shadow of an absorber edge at the mask, 2*thickness*tan(CRA)
    (in and out of the absorber), in the unit of the thickness."""

    return 2*thickness*np.tan(np.radians(cra))


def min_cra(NA, magnification, sigma=1.0):
    """Smallest chief ray angle (degrees) at which the illumination cone
    (sigma*NA/M) and the imaging cone (NA/M) at the mask do not overlap:
    sin(CRA) = (1 + sigma)*NA/(2*M) for the reduction ratio M in the plane
    of incidence."""

    return np.degrees(np.arcsin((1 + sigma)*NA/(2*np.asarray(magnification, dtype=float))))


def mask_cones(NA, magnification, cra=6.0, sigma=1.0, n=181):
    """Outlines of the cones at the mask in direction cosines, seen from
    above the mask (both cones pointing away from it).

    Returns
    -------
    imaging, illumination : (n, 2) arrays
        The reflected (imaging) cone, centred on (0, sin(CRA)), and the
        incident cone reversed, centred on (0, -sin(CRA)), each an ellipse
        with the half axes NA/M_x and NA/M_y (times sigma for the
        illumination).
    """

    M_x, M_y = _magnifications(magnification)
    t = np.linspace(0, 2*np.pi, n)
    c = np.sin(np.radians(cra))
    ell = np.stack([NA*np.cos(t)/M_x, NA*np.sin(t)/M_y], axis=1)

    return ell + [0.0, c], sigma*ell - [0.0, c]


def obscured_pupil(obscuration, shape="circle"):
    """Pupil transmission with a central obscuration, for
    Projector(apodization=...).

    Parameters
    ----------
    obscuration : number
        Radius of the obscuration relative to the pupil radius.
    shape : {"circle"}, optional

    Returns
    -------
    callable
        A(P_x, P_y): 0 inside the obscuration, 1 outside.
    """

    if shape != "circle":
        raise ValueError("only circular obscurations are supported")

    def A(P_x, P_y):
        return (np.hypot(P_x, P_y) >= obscuration).astype(float)

    A.obscuration = obscuration

    return A


def schwarzschild_objective(NA=0.3, reduction=5.0, R1=30.0, separation=80.0, field=0.05,
                            n_aspheric=4, wavelength=EUV_WAVELENGTH*1e-6, iterations=4):
    """A two-mirror Schwarzschild reduction objective of the kind of the EUV
    micro-exposure tools (NA 0.3, 5x; units mm).

    The light from the mask meets a small convex mirror M1 (the aperture
    stop) first, which sends it back to a large concave mirror M2 around
    the incoming beam; M2 focuses it past M1 onto the wafer. Both mirrors
    are concentric (common centre of curvature), which leaves little coma
    and astigmatism; the object distance gives the reduction, conic
    constants remove the third order spherical aberration and coma
    (off_axis.solve_conics()), and even aspheric terms on both mirrors and
    the image distance are then fitted by Gauss-Newton to the ray traced
    OPD on axis and at half and full field. The light passes the hole in
    M2 on its way in, and M1 blocks the centre of the beam on its way out:
    the pupil is centrally obscured (see `obscuration()`).

    Parameters
    ----------
    NA : number, optional
        Image-side numerical aperture.
    reduction : number, optional
        Reduction ratio (the magnification is -1/reduction).
    R1 : number, optional
        Radius of curvature of the convex mirror (mm).
    separation : number, optional
        Distance M1-M2 (mm); M2 has the radius R1 + separation.
    field : number, optional
        Image field radius (mm) over which the aspherics are fitted.
    n_aspheric : int, optional
        Aspheric terms r**4 ... per mirror.
    wavelength : number, optional
        In mm.
    iterations : int, optional
        Gauss-Newton steps.

    Returns
    -------
    raytrace.System
        M1 (0, the stop) and M2 (1); the object height is the field.
    """

    from scipy.optimize import brentq

    from .. import raytrace as rt
    from ..off_axis import solve_conics

    h = field*reduction

    def system(u, ep=1.0):
        s = [rt.Surface(R=R1, thickness=-separation, mirror=True, label="M1 (convex, stop)"),
             rt.Surface(R=R1 + separation, thickness=separation, mirror=True, label="M2 (concave)")]
        return rt.System(s, 0, wavelength, object_distance=u, object_height=h,
                         entrance_pupil_radius=ep).at_paraxial_focus()

    def mag_error(u):
        return system(u).paraxial().magnification + 1/reduction

    u_grid = separation + np.geomspace(1.0, 50*separation, 200)
    e = np.array([mag_error(u) for u in u_grid])
    k = np.nonzero(np.sign(e[:-1]) != np.sign(e[1:]))[0]
    if len(k) == 0:
        raise ValueError("no object distance gives this reduction")
    u = brentq(mag_error, u_grid[k[0]], u_grid[k[0] + 1])
    S0 = system(u)
    S0 = system(u, NA/S0.paraxial().na)
    S0 = solve_conics(S0, [0, 1], ("SI", "SII"))

    # aspherics in units of one wave of sag at the rim of each mirror
    b = S0.trace_pupil((0.0, 0.0), np.array([1.0]), np.array([0.0]))
    r1, r2 = abs(b.points[1, 0, 0]), abs(b.points[2, 0, 0])
    unit = np.array([wavelength/r1**(4 + 2*i) for i in range(n_aspheric)]
                    + [wavelength/r2**(4 + 2*i) for i in range(n_aspheric)] + [1e-4])
    fields = [(0.0, 0.0), (0.0, h/2), (0.0, h)]

    def apply(x):
        S = S0.copy()
        S.surfaces[0].aspheric = tuple(x[:n_aspheric]*unit[:n_aspheric])
        S.surfaces[1].aspheric = tuple(x[n_aspheric:2*n_aspheric]*unit[n_aspheric:2*n_aspheric])
        S.surfaces[1].thickness = S0.surfaces[1].thickness + x[-1]*unit[-1]
        return S

    def residual(x):
        S, out = apply(x), []
        for f in fields:
            W = S.opd(f, N=24)[2]
            W = W[np.isfinite(W)]
            out.append(W - W.mean())
        return np.concatenate(out)

    x = np.zeros(2*n_aspheric + 1)
    for _ in range(iterations):
        r0 = residual(x)
        J = np.array([residual(x + e) - r0 for e in np.eye(len(x))]).T
        x = x + np.linalg.lstsq(J, -r0, rcond=None)[0]

    return apply(x)


def obscuration(system, N=201):
    """Central obscuration of a Schwarzschild objective in normalized pupil
    radius: the rays from M2 to the image that pass M1 inside its rim
    (the beam footprint on M1), and the rays from M1 that fall into the
    hole M2 needs for the incoming beam (the beam from the edge of the
    field at the M2 plane).

    Returns
    -------
    rho : number
        Pupil radius below which the rays are blocked.
    """

    rho = np.linspace(0.0, 1.0, N)
    h = system.max_field
    b = system.trace_pupil((0.0, 0.0), rho, np.zeros_like(rho))
    r_m1 = np.max(np.abs(system.trace_pupil((0.0, h), np.array([0.0, 0.0, 1.0, -1.0]),
                                            np.array([1.0, -1.0, 0.0, 0.0])).points[1, :, :2]))
    # the returning beam at the plane of M1 (z = 0 of the global frame)
    p2, d2 = b.points[2], b.directions[2]
    t = -p2[:, 2]/d2[:, 2]
    r_back = np.hypot(*(p2[:, :2] + t[:, None]*d2[:, :2]).T)
    # the incoming beam at the plane of M2
    z2 = b.points[2, -1, 2]
    edge = system.trace_pupil((0.0, h), np.array([0.0, 0.0]), np.array([1.0, -1.0]))
    p0, d0 = edge.points[0], edge.directions[0]
    t0 = (z2 - p0[:, 2])/d0[:, 2]
    r_hole = np.max(np.abs((p0[:, :2] + t0[:, None]*d0[:, :2])[:, 1]))
    r_m2 = np.hypot(*b.points[2, :, :2].T)
    blocked = (r_back < r_m1) | (r_m2 < r_hole)

    return float(rho[blocked].max()) if np.any(blocked) else 0.0


def angles_of_incidence(system, surface, field=(0.0, 0.0), px=None, py=None):
    """Angles of incidence (radians) of rays on a mirror of a raytrace.System
    and the local radial coordinate of their hits.

    Parameters
    ----------
    system : raytrace.System
    surface : int
    field : pair, optional
        Object field point.
    px, py : arrays, optional
        Pupil coordinates (default: a radius 0..1).

    Returns
    -------
    r, aoi : arrays
    """

    if px is None:
        px = np.linspace(0.0, 1.0, 41)
        py = np.zeros_like(px)
    b = system.trace_pupil(tuple(field), np.asarray(px, float), np.asarray(py, float))
    o, R = system.frames()[0][surface]
    Q = (b.points[surface + 1] - o) @ R
    normal = system.surfaces[surface].normal(Q) @ R.T
    k1 = b.directions[surface]
    cos_i = np.abs(np.sum(k1*normal, axis=1))

    return np.hypot(Q[:, 0], Q[:, 1]), np.arccos(np.clip(cos_i, 0, 1))


def graded_profile(multilayer, r, aoi, angle=None, degree=2, wavelength=EUV_WAVELENGTH):
    """Relative thickness profile of a graded multilayer that keeps the
    Bragg peak at the local angle of incidence (period proportional to
    1/sqrt(1 - sin(aoi)**2/n_mean**2), fitted with even powers of r).

    Parameters
    ----------
    multilayer : Multilayer
        Designed for the angle `angle` (default: the AOI at r = 0).
    r, aoi : arrays
        Radial positions on the mirror and their angles of incidence
        (angles_of_incidence()).
    degree : int, optional
        Highest power of (r/r_max)**2.

    Returns
    -------
    coatings.RadialProfile
    """

    from ..coatings import RadialProfile

    r, aoi = np.asarray(r, float), np.asarray(aoi, float)
    ok = np.isfinite(r) & np.isfinite(aoi)
    r, aoi = r[ok], aoi[ok]
    angle = aoi[np.argmin(r)] if angle is None else angle
    d0 = multilayer.period
    periods = np.array([Multilayer.design(wavelength, a, bounds=(0.9*d0, 1.3*d0),
                                          **{k: v for k, v in vars(multilayer).items()
                                             if k != "period"}).period for a in
                        np.linspace(aoi.min(), aoi.max(), 5)])
    ratio = np.interp(aoi, np.linspace(aoi.min(), aoi.max(), 5), periods)/multilayer.design(
        wavelength, angle, bounds=(0.9*d0, 1.3*d0),
        **{k: v for k, v in vars(multilayer).items() if k != "period"}).period
    r_max = r.max()
    A = np.array([(r/r_max)**(2*k) for k in range(1, degree + 1)]).T
    coef = np.linalg.lstsq(A, ratio - 1, rcond=None)[0]

    return RadialProfile(tuple(coef), r_norm=r_max)


def z_factor(half_pitch, lwr, dose):
    """The RLS figure of merit Z = R**3*LWR**2*D (Wallow et al. 2008).

    For a family of resists, improving one of resolution R (half pitch),
    roughness LWR and dose D costs the others; lower is better.

    Parameters
    ----------
    half_pitch, lwr : numbers or arrays
        In nm.
    dose : number or array
        In mJ/cm**2.

    Returns
    -------
    Z : number or array
        In mJ*nm**3 (1 cm**2 = 1e14 nm**2), typically 1e-8 to 3e-8 for
        chemically amplified EUV resists.
    """

    return (np.asarray(half_pitch, float)**3*np.asarray(lwr, float)**2
            *np.asarray(dose, float)*1e-14)


def feature_center(x, I, threshold, center=0.0, bright=False):
    """Centre of the printed feature (mean of its two threshold edges)."""

    from .metrics import feature_edges

    left, right = feature_edges(x, I, threshold, center, bright=bright)

    return (left + right)/2


__all__ = ["EUV_WAVELENGTH", "INDEX_13_5", "index", "fwhm", "Multilayer", "ReflectiveMask",
           "shadow_width", "min_cra", "mask_cones", "obscured_pupil", "schwarzschild_objective",
           "obscuration", "angles_of_incidence", "graded_profile", "z_factor", "feature_center"]
