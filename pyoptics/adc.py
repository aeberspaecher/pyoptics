#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Atmospheric refraction and dispersion, and the atmospheric dispersion
corrector (ADC) of a telescope (notebook 101).

Air bends the light of a star towards the zenith by the refraction R(z),
about a minute of arc at the zenith angle z = 45°, and more for blue than
for red light: a star near the horizon is a short vertical spectrum. An
ADC, two counter-rotating Amici prisms, adds the opposite dispersion.

Air index (`edlen_index()`): Edlén's formula as revised by Birch and Downs
(1993, 1994) for the temperature t (°C), the pressure p (Pa) and the
partial pressure of water vapour f (Pa), with the wavenumber sigma = 1/lambda
in 1/um ::

    (n - 1)_s 1e8 = 8342.54 + 2406147/(130 - sigma^2) + 15998/(38.9 - sigma^2)
    (n - 1)_tp = (n - 1)_s p [1 + 1e-8 (0.601 - 0.00972 t) p]/(96095.43 (1 + 0.003661 t))
    n_tpf = n_tp - f (3.7345 - 0.0401 sigma^2) 1e-10

Refraction (all angles in radians):

- plane parallel layers: Snell's law between the ground and the vacuum
  gives sin(z_true) = n0 sin(z_app), so R = (n0 - 1) tan z to first
  order, whatever the profile n(h) (`refraction_flat()`);
- a spherical atmosphere, the classic two term formula ::

      R = A tan z - B tan^3 z,  A = (n0 - 1)(1 - beta),
                                B = (n0 - 1)(beta - (n0 - 1)/2),

  with beta = H/R_earth, H the scale height of an isothermal atmosphere
  (`refraction_laplace()`);
- the exact refraction of a spherically stratified isothermal atmosphere
  (n - 1 proportional to exp(-h/H)) from the invariant n r sin(psi) =
  const (Bouguer) as an integral over the height (`refraction_integral()`),
  and by tracing the ray through the atmosphere as a GRIN medium
  (`atmosphere_medium()`, `trace_refraction()`).

Dispersion: R(lambda) - R(lambda_ref) = (n(lambda) - n(lambda_ref)) tan z
(1 - beta) to first order (`atmospheric_dispersion()`).

ADC: an Amici prism is a thin prism of glass 1 cemented to an opposed one
of glass 2 with the same apex angle; it deviates by ::

    delta(lambda) = (n1(lambda) - n2(lambda)) alpha

(`AmiciPrism`, exact for a thick prism in `AmiciPrism.deviation_exact()`).
Two of them rotated by +phi and -phi about the optical axis add their
deviations along one direction, 2 delta cos(phi), and cancel across it.
In a collimated beam compressed by M = D_tel/D_beam a deviation delta in
the beam is delta/M on the sky. The ADC corrects the dispersion when ::

    2 cos(phi) (delta(lambda) - delta(lambda_ref))/M = -(R(lambda) - R(lambda_ref)),

for all wavelengths of the band only if the glass pair has the same
relative dispersion as air; `best_glass_pairs()` searches a catalogue for
the pairs that match it best, `adc_residual()` gives the best phi and the
residual dispersion for a zenith angle.

References
----------
B. Edlén, "The refractive index of air", Metrologia 2, 71-80 (1966).

K. P. Birch and M. J. Downs, "An updated Edlén equation for the
refractive index of air", Metrologia 30, 155-162 (1993); "Correction to
the updated Edlén equation for the refractive index of air", Metrologia
31, 315-316 (1994).

R. M. Green, "Spherical Astronomy", Cambridge University Press (1985),
chapter 4 (refraction, A tan z - B tan^3 z).

A. V. Filippenko, "The importance of atmospheric differential refraction
in spectrophotometry", Publ. Astron. Soc. Pac. 94, 715-721 (1982).

C. G. Wynne and S. P. Worswick, "Atmospheric dispersion correctors at
the Cassegrain focus", Mon. Not. R. Astron. Soc. 220, 657-670 (1986).
"""

from __future__ import division

import numpy as np

from pyoptics import grin


__all__ = ["ARCSEC", "R_EARTH", "edlen_index", "water_vapour_pressure", "scale_height",
           "refraction_flat", "refraction_laplace", "refraction_integral", "atmosphere_medium",
           "trace_refraction", "atmospheric_dispersion", "AmiciPrism", "adc_dispersion",
           "adc_residual", "best_glass_pairs", "smeared_star"]

ARCSEC = np.pi/180/3600
R_EARTH = 6.371e6                                      # m


def edlen_index(wavelength_nm, t=15.0, p=101325.0, f=0.0):
    """Refractive index of air after Edlén (Birch and Downs 1993/1994).

    Parameters
    ----------
    wavelength_nm : number or array
        Vacuum wavelength (nm).
    t : number, optional
        Temperature (°C).
    p : number, optional
        Pressure (Pa).
    f : number, optional
        Partial pressure of water vapour (Pa).

    Returns
    -------
    n : number or array
    """

    s2 = (1e3/np.asarray(wavelength_nm, float))**2
    ns = 1e-8*(8342.54 + 2406147/(130 - s2) + 15998/(38.9 - s2))
    ntp = ns*p*(1 + 1e-8*(0.601 - 0.00972*t)*p)/(96095.43*(1 + 0.003661*t))
    return (1 + ntp - f*(3.7345 - 0.0401*s2)*1e-10)[()]


def water_vapour_pressure(t, relative_humidity):
    """Partial pressure of water vapour (Pa) for a relative humidity (0 to 1)
    at t (°C), with the Magnus formula for the saturation pressure."""

    return relative_humidity*611.2*np.exp(17.62*t/(243.12 + t))


def scale_height(t):
    """Scale height R_air T/g (m) of an isothermal atmosphere at t (°C)."""

    return 287.05*(t + 273.15)/9.80665


def refraction_flat(z, n0):
    """Refraction (n0 - 1) tan z of plane parallel layers (first order).

    Parameters
    ----------
    z : number or array
        Apparent zenith angle (rad).
    n0 : number or array
        Index of the air at the observer.
    """

    return (n0 - 1)*np.tan(z)


def refraction_laplace(z, n0, H=8435.0, R=R_EARTH):
    """Refraction A tan z - B tan^3 z of a spherical atmosphere with the
    scale height H (see the module docstring)."""

    beta = H/R
    N = n0 - 1
    tz = np.tan(z)
    return N*(1 - beta)*tz - N*(beta - N/2)*tz**3


def refraction_integral(z, n0, H=8435.0, R=R_EARTH, n_nodes=400):
    """Exact refraction of a spherical isothermal atmosphere.

    With n(h) = 1 + (n0 - 1) exp(-h/H) and the invariant n r sin(psi) =
    n0 R sin(z), R(z) = integral of tan(psi) (-dn/n) from the ground to
    infinity. Gauss-Laguerre-like quadrature in u = exp(-h/H) (Gauss-Legendre
    on 0 < u < 1 with u = s^2 to resolve the ground).

    Parameters
    ----------
    z : number or array
        Apparent zenith angle (rad), below 90°.
    n0 : number
    H, R : numbers, optional
        Scale height and radius of the Earth (m).
    n_nodes : int, optional

    Returns
    -------
    refraction : number or array (rad)
    """

    z = np.atleast_1d(np.asarray(z, float))
    s, w = np.polynomial.legendre.leggauss(n_nodes)
    s = 0.5*(s + 1)
    w = 0.5*w
    u = s**2                                         # exp(-h/H)
    du = 2*s*w
    N = n0 - 1
    h = -H*np.log(u)
    n = 1 + N*u
    r = R + h
    out = []
    for zz in z:
        k = n0*R*np.sin(zz)
        sin_psi = k/(n*r)
        tan_psi = sin_psi/np.sqrt(1 - sin_psi**2)
        # dn = N du
        out.append(np.sum(tan_psi*N/n*du))
    out = np.array(out)
    return out[0] if out.size == 1 else out


def atmosphere_medium(n0, H=8435.0, R=R_EARTH):
    """A spherical isothermal atmosphere as a 2D GRIN medium (metres).

    The Earth's centre is at (x, y) = (0, -R), the observer at the origin,
    y is up: n = 1 + (n0 - 1) exp(-(r - R)/H).

    Returns
    -------
    grin.IndexField
    """

    N = n0 - 1
    c = np.array([0.0, -R])

    def n(r):
        rr = np.linalg.norm(r - c, axis=-1)
        return 1 + N*np.exp(-(rr - R)/H)

    def force(r):
        d = r - c
        rr = np.linalg.norm(d, axis=-1)
        e = N*np.exp(-(rr - R)/H)
        return ((1 + e)*(-e/H)/rr)[..., None]*d          # n grad n

    return grin.IndexField(n, force)


def trace_refraction(z, n0, H=8435.0, R=R_EARTH, top=150e3, step=50.0):
    """Refraction by tracing rays from the observer up through
    `atmosphere_medium()` with `grin.trace()`: the angle between the
    direction the ray leaves the observer (apparent zenith angle z) and
    its direction above the atmosphere.

    Parameters
    ----------
    z : array
        Apparent zenith angles (rad).
    n0, H, R : numbers
    top : number, optional
        Height where the trace stops (m).
    step : number, optional
        Step of the tracer (m).

    Returns
    -------
    refraction : array (rad)
    """

    z = np.atleast_1d(np.asarray(z, float))
    medium = atmosphere_medium(n0, H, R)
    P = np.zeros((z.size, 2))
    D = np.c_[np.sin(z), np.cos(z)]
    c = np.array([0.0, -R])
    stop = lambda r: (R + top) - np.linalg.norm(r - c, axis=-1)
    max_steps = int(4*top/np.cos(z.max())/step) + 100
    rays = grin.trace(medium, P, D, step=step, stop=stop, record=False, max_steps=max_steps)
    d = rays.direction
    return np.arctan2(d[:, 0], d[:, 1]) - z


def atmospheric_dispersion(wavelength_nm, z, wavelength_ref=550.0, t=15.0, p=101325.0, f=0.0,
                           exact=False):
    """Refraction at a wavelength relative to a reference wavelength (rad).

    The star appears higher (towards the zenith) at shorter wavelengths:
    the returned value is positive towards the zenith.

    Parameters
    ----------
    wavelength_nm : number or array
    z : number or array
        Zenith angle (rad).
    wavelength_ref : number, optional
    t, p, f : numbers, optional
        Air temperature (°C), pressure (Pa) and water vapour pressure (Pa).
    exact : bool, optional
        Use `refraction_integral()` instead of the A tan z - B tan^3 z
        formula.

    Returns
    -------
    dR : array, broadcast of wavelength and z
    """

    n = edlen_index(wavelength_nm, t, p, f)
    n_ref = edlen_index(wavelength_ref, t, p, f)
    H = scale_height(t)
    if exact:
        lam = np.atleast_1d(np.asarray(wavelength_nm, float))
        R_ref = refraction_integral(z, n_ref, H)
        return np.array([refraction_integral(z, nn, H) - R_ref
                         for nn in np.atleast_1d(n)]).reshape(np.broadcast(lam, z).shape)
    return refraction_laplace(z, n, H) - refraction_laplace(z, n_ref, H)


class AmiciPrism(object):
    """A thin Amici prism: glass 1 and glass 2 cemented with opposed wedges of
    the apex angle alpha.

    Parameters
    ----------
    glass1, glass2 : callables
        n(wavelength_nm), e.g. `glasses.Glass`.
    apex : number
        Apex angle (rad).
    """

    def __init__(self, glass1, glass2, apex):
        self.glass1, self.glass2 = glass1, glass2
        self.apex = apex

    def delta_n(self, wavelength_nm):
        """n1 - n2."""

        lam = np.asarray(wavelength_nm, float)
        return np.asarray(self.glass1(lam), float) - np.asarray(self.glass2(lam), float)

    def deviation(self, wavelength_nm):
        """Thin prism deviation (n1 - n2) alpha (rad)."""

        return self.delta_n(wavelength_nm)*self.apex

    def deviation_exact(self, wavelength_nm):
        """Exact deviation of a ray at normal incidence: undeviated into
        glass 1, refracted at the cemented face tilted by alpha, and at the
        exit face (normal to the axis) into air (rad)."""

        lam = np.asarray(wavelength_nm, float)
        n1 = np.asarray(self.glass1(lam), float)
        n2 = np.asarray(self.glass2(lam), float)
        t2 = np.arcsin(n1/n2*np.sin(self.apex))
        return np.arcsin(n2*np.sin(t2 - self.apex))


def adc_dispersion(prism, wavelength_nm, phi, M=1.0, wavelength_ref=550.0):
    """Dispersion of two Amici prisms counter-rotated by +/-phi, on the
    sky (rad, relative to the reference wavelength, along the axis where
    the deviations add): 2 cos(phi) (delta - delta_ref)/M."""

    d = prism.deviation(wavelength_nm) - prism.deviation(wavelength_ref)
    return 2*np.cos(phi)*d/M


def adc_residual(prism, wavelength_nm, dR, M=1.0, wavelength_ref=550.0, weights=None):
    """Best rotation and residual dispersion of an ADC for the atmospheric
    dispersion dR(lambda).

    cos(phi) is the weighted least squares scale of the ADC dispersion
    onto dR (means removed), clipped to 1 (beyond, the ADC is too weak);
    the ADC is oriented to oppose the atmosphere.

    Parameters
    ----------
    prism : AmiciPrism
    wavelength_nm : array
    dR : array
        Atmospheric dispersion relative to wavelength_ref (rad on the sky).
    M : number, optional
        Beam compression D_tel/D_beam.
    wavelength_ref : number, optional
    weights : array, optional

    Returns
    -------
    phi : number (rad)
    residual : array (rad), dR + ADC dispersion
    """

    w = np.ones_like(dR) if weights is None else np.asarray(weights, float)
    g = adc_dispersion(prism, wavelength_nm, 0.0, M, wavelength_ref)
    # remove the weighted mean (a common offset is a pointing offset)
    def centred(a):
        return a - np.sum(w*a)/np.sum(w)
    gc, dc = centred(g), centred(dR)
    c = np.sum(w*gc*dc)/np.sum(w*gc*gc)
    # the ADC is turned so that it opposes the atmosphere (a rotation by
    # 180 degrees flips the sign of its dispersion)
    cos_phi = min(abs(c), 1.0)
    return np.arccos(cos_phi), dR - np.sign(c)*cos_phi*g


def best_glass_pairs(catalog, wavelength_nm, wavelength_ref=550.0, max_delta_n=0.02, min_scale=0.0,
                     k=10, t=15.0, p=101325.0, f=0.0):
    """Glass pairs whose difference n1 - n2 has the shape of the dispersion
    of air.

    For each pair the difference g = (n1 - n2)(lambda) - (n1 - n2)(lambda_ref)
    is scaled onto the air dispersion a = n_air(lambda) - n_air(lambda_ref)
    by least squares (after removing the means); the relative rms residual
    measures how well an ADC of this pair can correct all wavelengths at
    once. Pairs with abs(n1 - n2) > max_delta_n at lambda_ref (a large
    deviation) are left out.

    Parameters
    ----------
    catalog : dict name -> Glass
    wavelength_nm : array
    wavelength_ref : number, optional
    max_delta_n : number, optional
    min_scale : number, optional
        Leave out pairs whose dispersion difference is less than min_scale
        times that of air (they would need large apex angles).
    k : int, optional
        Number of pairs returned.
    t, p, f : numbers, optional
        Air conditions.

    Returns
    -------
    list of (relative residual, name1, name2, scale, delta_n_ref), best
    first; scale is (n1 - n2) dispersion per unit air dispersion, positive
    (glass 1 is the more dispersive one).
    """

    lam = np.asarray(wavelength_nm, float)
    names, rows = [], []
    for name, g in catalog.items():
        if hasattr(g, "in_range") and not (g.in_range(lam.min()) and g.in_range(lam.max())):
            continue
        names.append(name)
        rows.append(np.r_[g(lam), g(wavelength_ref)])
    N = np.array(rows)                                   # (glasses, nlam + 1)
    a = edlen_index(lam, t, p, f) - edlen_index(wavelength_ref, t, p, f)
    a = a - a.mean()
    an = a/np.linalg.norm(a)
    disp = N[:, :-1] - N[:, -1:]                        # n(lambda) - n(lambda_ref) per glass
    disp = disp - disp.mean(axis=1, keepdims=True)
    G = disp[:, None, :] - disp[None, :, :]            # pair differences
    nref = N[:, -1]
    dn = nref[:, None] - nref[None, :]
    proj = G @ an                                       # component along the air shape
    norm = np.linalg.norm(G, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.sqrt(np.clip(norm**2 - proj**2, 0, None))/np.abs(proj)
    scale = proj/np.linalg.norm(a)
    rel[(np.abs(dn) > max_delta_n) | (proj <= 0) | (scale < min_scale)] = np.inf
    order = np.argsort(rel, axis=None)[:k]
    out = []
    for idx in order:
        i, j = np.unravel_index(idx, rel.shape)
        if not np.isfinite(rel[i, j]):
            break
        out.append((rel[i, j], names[i], names[j], scale[i, j], dn[i, j]))
    return out


def smeared_star(x, y, wavelength_nm, weights, dR, fwhm):
    """Image of a star with the dispersion dR(lambda) along y: the sum of
    Gaussian seeing images (FWHM in the unit of x, y) displaced by dR.

    Parameters
    ----------
    x, y : 1d arrays
        Image coordinates (arcsec or rad, the unit of dR and fwhm).
    wavelength_nm : array
    weights : array
        Relative photon flux per wavelength sample.
    dR : array
        Displacement per wavelength (towards +y).
    fwhm : number or array
        Seeing FWHM, per wavelength if an array.

    Returns
    -------
    image : (len(y), len(x)) array, sum 1
    """

    X, Y = np.meshgrid(x, y)
    s = np.broadcast_to(np.asarray(fwhm, float)/2.3548, np.shape(dR))
    img = np.zeros_like(X)
    for w, d, si in zip(weights, dR, s):
        img += w*np.exp(-(X**2 + (Y - d)**2)/(2*si**2))/si**2
    return img/img.sum()
