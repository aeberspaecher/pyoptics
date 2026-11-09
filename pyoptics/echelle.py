#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Echelle spectrographs: the grating equation in high orders, the blaze,
the free spectral range, cross-dispersion and the two-dimensional
echellogram, and the resolving power against the slit width (notebook
100).

An echelle is a coarse grating (tens of grooves per mm) with right-angle
grooves, used close to the Littrow configuration at a steep blaze angle
theta_B (tan theta_B = 2 for an "R2" echelle, 4 for "R4"). Light falls on
the facets of width b = d cos(theta_B), and orders around ::

    m_B = 2 d cos(gamma) sin(theta_B)/lambda

(fifty to two hundred in the visible) are bright. Wavelengths are in nm and
the groove spacing d in nm too; lengths at the detector in mm.

Closed form results used for checks:

- Grating equation with the off-plane angle gamma (`Echelle.beta()`) ::

      m lambda = d cos(gamma) (sin(alpha) + sin(beta)).

- Blaze function (`Echelle.efficiency()`, scalar theory, facet fully
  illuminated, no shadowing) ::

      eta = sinc^2(nu),  nu = b cos(gamma) (sin(alpha - theta_B) +
                              sin(beta - theta_B))/lambda,

  with sinc(x) = sin(pi x)/(pi x) (numpy's). In Littrow on the blaze
  (alpha = theta_B) nu = m - m_B to first order in beta - theta_B, so the
  orders share the light as sinc^2(m - m_B), which sums to 1, and the blaze
  peak of order m is at lambda_B = 2 d cos(gamma) sin(theta_B)/m.
- Free spectral range lambda/m: the part of order m that does not overlap
  its neighbours. Its length on the detector, f Delta beta = f lambda/(d
  cos beta) = 2 f tan(theta_B)/m (Littrow), grows with the wavelength.
- Angular dispersion m/(d cos(gamma) cos(beta)) = 2 tan(theta_B)/lambda in
  Littrow, the same for all orders: R2 or R4 is what sets it.
- Resolving power: the diffraction limit of a grating of N illuminated
  grooves is R = m N (Rayleigh), the FWHM of the line m N/0.886. A slit of
  width s behind a collimator of focal length f_coll limits it to ::

      R = 2 tan(theta_B) f_coll/s = 2 tan(theta_B) D_coll/(phi D_tel)

  for a slit of angular width phi on the sky, a collimated beam D_coll and
  a telescope D_tel: the R phi product grows with the size of the echelle,
  not with the telescope (`slit_resolving_power()`).
- Cross-dispersion: the orders are stacked across the dispersion with a
  low-dispersion grating or a prism. Adjacent orders differ by lambda/m in
  wavelength, so a grating separates them by f m_cd lambda/(m d_cd cos)
  ~ lambda^2: crowded in the blue, while a prism with dn/dlambda ~
  lambda^-3 spaces them more evenly.

Functions and classes:

- Echelle: grating equation, Littrow order, blaze wavelength, efficiency,
  free spectral range, dispersion.
- GratingCrossDisperser, PrismCrossDisperser: the cross-dispersion angle.
- order_position(), echellogram(): positions of wavelengths in the
  echellogram and an image of a spectrum.
- staircase_far_field(), order_efficiencies(): the wavefront reflected by
  the facets in Littrow (a staircase of steps of 2 d sin(theta_B)) and its
  far field by FFT, for a numerical check of the blaze function and of
  R = m N.
- instrument_profile(), fwhm(), slit_resolving_power(): the line profile
  for a slit and a diffraction limited beam, and its width.
- solar_like_spectrum(): a blackbody with Fraunhofer lines.

References
----------
D. J. Schroeder, "Astronomical Optics", 2nd ed., Academic Press (2000),
chapters 13-15 (grating equation, blaze, echelles, R phi product).

C. Palmer, "Diffraction Grating Handbook", 8th ed., MKS Newport (2020).

F. Pepe et al., "HARPS: ESO's coming planet searcher", The Messenger 110,
9-14 (2002) (R4 echelle of 31.6 grooves/mm, orders 89 to 161).
"""

from __future__ import division

import numpy as np

from pyoptics.ultrafast import prism_angles


__all__ = ["Echelle", "GratingCrossDisperser", "PrismCrossDisperser", "order_position",
           "echellogram", "staircase_far_field", "order_efficiencies", "instrument_profile",
           "fwhm", "slit_resolving_power", "solar_like_spectrum", "FRAUNHOFER_LINES"]


class Echelle(object):
    """An echelle grating used near Littrow.

    Parameters
    ----------
    grooves_per_mm : number
    blaze : number
        Blaze angle theta_B (rad).
    gamma : number, optional
        Off-plane angle (rad); the incident and diffracted beams are tilted
        by gamma out of the plane perpendicular to the grooves.
    alpha : number, optional
        In-plane angle of incidence (rad), default theta_B (Littrow).
    """

    def __init__(self, grooves_per_mm, blaze, gamma=0.0, alpha=None):
        self.d = 1e6/grooves_per_mm                      # nm
        self.blaze = blaze
        self.gamma = gamma
        self.alpha = blaze if alpha is None else alpha

    @property
    def facet(self):
        """Facet width b = d cos(theta_B) (nm)."""

        return self.d*np.cos(self.blaze)

    def littrow_order(self, wavelength):
        """Continuous order m_B = 2 d cos(gamma) sin(theta_B)/lambda."""

        return 2*self.d*np.cos(self.gamma)*np.sin(self.blaze)/np.asarray(wavelength, float)

    def blaze_wavelength(self, m):
        """Blaze wavelength 2 d cos(gamma) sin(theta_B)/m of order m (nm)."""

        return 2*self.d*np.cos(self.gamma)*np.sin(self.blaze)/np.asarray(m, float)

    def central_order(self, wavelength):
        """Integer order nearest to the blaze for a wavelength."""

        return np.rint(self.littrow_order(wavelength)).astype(int)

    def beta(self, m, wavelength):
        """In-plane diffraction angle from m lambda = d cos(gamma)(sin alpha
        + sin beta); NaN where the order does not propagate."""

        s = m*np.asarray(wavelength, float)/(self.d*np.cos(self.gamma)) - np.sin(self.alpha)
        with np.errstate(invalid="ignore"):
            return np.arcsin(np.where(np.abs(s) <= 1, s, np.nan))

    def wavelength(self, m, beta):
        """Wavelength d cos(gamma)(sin alpha + sin beta)/m diffracted into
        the angle beta by order m (nm)."""

        return self.d*np.cos(self.gamma)*(np.sin(self.alpha) + np.sin(beta))/np.asarray(m, float)

    def efficiency(self, m, wavelength):
        """Scalar blaze efficiency sinc^2(nu) of order m (see the module
        docstring)."""

        wavelength = np.asarray(wavelength, float)
        b = self.beta(m, wavelength)
        nu = self.facet*np.cos(self.gamma)*(np.sin(self.alpha - self.blaze)
                                            + np.sin(b - self.blaze))/wavelength
        return np.sinc(nu)**2

    def free_spectral_range(self, m):
        """Free spectral range lambda_B/m of order m (nm)."""

        return self.blaze_wavelength(m)/np.asarray(m, float)

    def fsr_limits(self, m):
        """Wavelengths where order m meets its neighbours: the blaze
        wavelengths of the half orders m -/+ 1/2 (nm, short, long)."""

        m = np.asarray(m, float)
        return self.blaze_wavelength(m + 0.5), self.blaze_wavelength(m - 0.5)

    def dispersion(self, m, wavelength):
        """Angular dispersion d beta/d lambda = m/(d cos(gamma) cos(beta))
        (rad/nm)."""

        return m/(self.d*np.cos(self.gamma)*np.cos(self.beta(m, wavelength)))

    def orders(self, wavelength_min, wavelength_max):
        """Orders whose free spectral range overlaps [wavelength_min,
        wavelength_max], in decreasing order (increasing wavelength)."""

        m_hi = int(np.floor(self.littrow_order(wavelength_min) + 0.5))
        m_lo = int(np.ceil(self.littrow_order(wavelength_max) - 0.5))
        return np.arange(m_hi, m_lo - 1, -1)


class GratingCrossDisperser(object):
    """A low dispersion grating that stacks the echelle orders.

    Parameters
    ----------
    grooves_per_mm : number
    order : int, optional
    alpha : number, optional
        Angle of incidence (rad).
    """

    def __init__(self, grooves_per_mm, order=1, alpha=0.0):
        self.d = 1e6/grooves_per_mm
        self.order = order
        self.alpha = alpha

    def angle(self, wavelength):
        """Diffraction angle sin(phi) = m lambda/d - sin(alpha) (rad)."""

        return np.arcsin(self.order*np.asarray(wavelength, float)/self.d - np.sin(self.alpha))

    def dispersion(self, wavelength):
        """d phi/d lambda = m/(d cos phi) (rad/nm)."""

        return self.order/(self.d*np.cos(self.angle(wavelength)))


class PrismCrossDisperser(object):
    """A prism that stacks the echelle orders, used at a fixed angle of
    incidence.

    Parameters
    ----------
    n : callable
        n(wavelength_nm) of the glass, e.g. a `glasses.Glass`.
    apex : number
        Apex angle (rad).
    theta_in : number, optional
        Angle of incidence (rad), default the minimum deviation angle at
        `wavelength_ref`.
    wavelength_ref : number, optional
    passes : int, optional
        1 or 2 (double pass): the deviation is multiplied by the number of
        passes, the usual approximation for a prism traversed twice.
    """

    def __init__(self, n, apex, theta_in=None, wavelength_ref=500.0, passes=1):
        self.n = n
        self.apex = apex
        if theta_in is None:
            theta_in = np.arcsin(float(n(wavelength_ref))*np.sin(apex/2))
        self.theta_in = theta_in
        self.passes = passes

    def angle(self, wavelength):
        """Deviation theta_in + theta_out - apex (times the passes, rad)."""

        n = np.asarray(self.n(np.asarray(wavelength, float)), float)
        _, _, t_out = prism_angles(self.apex, n, self.theta_in)
        return self.passes*(self.theta_in + t_out - self.apex)

    def dispersion(self, wavelength, step=0.01):
        """d(deviation)/d lambda by central differences (rad/nm)."""

        wavelength = np.asarray(wavelength, float)
        return (self.angle(wavelength + step) - self.angle(wavelength - step))/(2*step)


def order_position(echelle, cross, m, wavelength, f_camera, wavelength_ref=None):
    """Position of a wavelength of order m in the echellogram.

    x (along the orders) is f tan(beta - theta_B), y (across) is
    f tan(phi - phi_ref) with the cross-disperser angle phi and phi_ref its
    value at `wavelength_ref`.

    Parameters
    ----------
    echelle : Echelle
    cross : GratingCrossDisperser or PrismCrossDisperser
    m : int or array
    wavelength : number or array (nm)
    f_camera : number
        Camera focal length (mm).
    wavelength_ref : number, optional
        Wavelength at y = 0 (nm), default 500.

    Returns
    -------
    x, y : arrays (mm)
    """

    if wavelength_ref is None:
        wavelength_ref = 500.0
    x = f_camera*np.tan(echelle.beta(m, wavelength) - echelle.blaze)
    y = f_camera*np.tan(cross.angle(wavelength) - cross.angle(wavelength_ref))
    return x, y


def echellogram(spectrum, echelle, cross, f_camera, orders, pixel, shape, slit_height=0.15,
                line_width=None, n_samples=6000, wavelength_ref=500.0, overlap=0.1):
    """Image of a spectrum in an echelle spectrograph.

    Each order is sampled over its free spectral range (extended by
    `overlap` on both sides, where the neighbouring order shows the same
    wavelengths), weighted by the spectrum and the blaze efficiency and
    deposited on the pixels; the slit image is a top hat of `slit_height`
    across the order and a Gaussian of FWHM `line_width` along it.

    Parameters
    ----------
    spectrum : callable
        Spectral flux per nm, spectrum(wavelength_nm).
    echelle : Echelle
    cross : GratingCrossDisperser or PrismCrossDisperser
    f_camera : number (mm)
    orders : sequence of int
    pixel : number
        Pixel size (mm).
    shape : (ny, nx)
        Detector size in pixels, centred on (0, 0).
    slit_height : number, optional
        Height of the slit image across the orders (mm).
    line_width : number, optional
        FWHM of the slit image along the orders (mm), default 2 pixels.
    n_samples : int, optional
        Wavelength samples per order.
    wavelength_ref : number, optional
    overlap : number, optional
        Fraction of the free spectral range added on both sides.

    Returns
    -------
    image : (ny, nx) array (flux per pixel)
    extent : (x0, x1, y0, y1) in mm for imshow
    """

    from scipy.ndimage import convolve1d, gaussian_filter1d

    ny, nx = shape
    x_edges = (np.arange(nx + 1) - nx/2)*pixel
    y_edges = (np.arange(ny + 1) - ny/2)*pixel
    image = np.zeros((ny, nx))
    for m in orders:
        lo, hi = echelle.fsr_limits(m)
        pad = overlap*(hi - lo)
        lam = np.linspace(lo - pad, hi + pad, n_samples)
        dlam = lam[1] - lam[0]
        x, y = order_position(echelle, cross, m, lam, f_camera, wavelength_ref)
        w = spectrum(lam)*echelle.efficiency(m, lam)*dlam
        ok = np.isfinite(x) & np.isfinite(w)
        # linear interpolation between the two nearest pixel columns,
        # nearest pixel row (the slit height spreads it afterwards)
        fx = (x[ok] - x_edges[0])/pixel - 0.5
        i0 = np.floor(fx).astype(int)
        t = fx - i0
        j = np.floor((y[ok] - y_edges[0])/pixel).astype(int)
        for ii, ww in ((i0, w[ok]*(1 - t)), (i0 + 1, w[ok]*t)):
            inside = (ii >= 0) & (ii < nx) & (j >= 0) & (j < ny)
            np.add.at(image, (j[inside], ii[inside]), ww[inside])
    if line_width is None:
        line_width = 2*pixel
    image = gaussian_filter1d(image, line_width/pixel/2.3548, axis=1)
    n_h = max(int(round(slit_height/pixel)), 1)
    image = convolve1d(image, np.ones(n_h)/n_h, axis=0, mode="constant")
    return image, (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])


def staircase_far_field(echelle, wavelength, n_facets=200, samples=16, pad=8):
    """Far field of the wavefront reflected by an echelle in Littrow.

    In Littrow the facets face the beam: the reflected wavefront is a
    staircase of n_facets steps of width b = d cos(theta_B) (across the
    beam), each delayed by the round trip 2 d sin(theta_B) = m_B lambda
    against its neighbour. Its FFT gives the intensity against the
    deviation beta - theta_B (in-plane, gamma = 0).

    Parameters
    ----------
    echelle : Echelle
    wavelength : number (nm)
    n_facets : int, optional
    samples : int, optional
        Samples per facet.
    pad : int, optional
        Zero padding factor.

    Returns
    -------
    delta : array
        sin(beta - theta_B), sorted.
    intensity : array
        Normalized to a total of 1 (energy fractions per sample).
    """

    b = echelle.facet
    step = 2*np.pi*2*echelle.d*np.sin(echelle.blaze)/wavelength
    j = np.repeat(np.arange(n_facets), samples)
    field = np.exp(1j*step*j)
    n = field.size*pad
    F = np.fft.fftshift(np.fft.fft(field, n))
    du = b/samples
    freq = np.fft.fftshift(np.fft.fftfreq(n, du))
    # the facet delays are the reflected wavefront: a tilt of the wavefront
    # by delta adds exp(-2 pi i delta u/lambda)
    delta = -freq*wavelength
    order = np.argsort(delta)
    intensity = np.abs(F[order])**2
    return delta[order], intensity/intensity.sum()


def order_efficiencies(echelle, wavelength, orders, n_facets=200, samples=16, pad=8):
    """Energy fractions of the orders from `staircase_far_field()`: the
    power within half an order spacing of each order's direction. In the
    staircase model (small deviations) order m leaves at sin(beta -
    theta_B) = (m - m_B) lambda/b.

    Parameters
    ----------
    echelle : Echelle
    wavelength : number (nm)
    orders : sequence of int

    Returns
    -------
    eta : array
    """

    delta, I = staircase_far_field(echelle, wavelength, n_facets, samples, pad)
    spacing = wavelength/echelle.facet
    eta = []
    for m in orders:
        d0 = (m - echelle.littrow_order(wavelength))*spacing
        eta.append(I[np.abs(delta - d0) < spacing/2].sum())
    return np.array(eta)


def instrument_profile(delta, slit_angle, beam_width, wavelength):
    """Line profile in angle: the slit image (a top hat of full width
    slit_angle = s/f_coll, Littrow) convolved with the diffraction pattern
    sinc^2(W delta/lambda) of the collimated beam of width W.

    Parameters
    ----------
    delta : array
        Equally spaced angles (rad), centred on 0.
    slit_angle : number (rad)
    beam_width : number
        W, in the unit of the wavelength.
    wavelength : number

    Returns
    -------
    profile : array, peak 1
    """

    psf = np.sinc(beam_width*delta/wavelength)**2
    if slit_angle > 0:
        dd = delta[1] - delta[0]
        half = int(np.ceil(slit_angle/(2*dd))) + 1
        k = np.arange(-half, half + 1)*dd
        # overlap of each sample cell with the slit [-s/2, s/2]
        kernel = np.clip(np.minimum(k + dd/2, slit_angle/2) - np.maximum(k - dd/2, -slit_angle/2),
                         0, None)
        psf = np.convolve(psf, kernel, mode="same")
    return psf/psf.max()


def fwhm(x, y):
    """Full width at half maximum of a single peak, by linear interpolation."""

    y = np.asarray(y, float)
    h = y.max()/2
    above = np.nonzero(y >= h)[0]
    i0, i1 = above[0], above[-1]
    xl = x[i0 - 1] + (h - y[i0 - 1])*(x[i0] - x[i0 - 1])/(y[i0] - y[i0 - 1])
    xr = x[i1] + (h - y[i1])*(x[i1 + 1] - x[i1])/(y[i1 + 1] - y[i1])
    return xr - xl


def slit_resolving_power(blaze, f_coll=None, slit_width=None, D_coll=None, D_tel=None, phi=None):
    """Slit limited resolving power of an echelle in Littrow.

    Either 2 tan(theta_B) f_coll/s (slit width s at the collimator focus)
    or, with the collimated beam D_coll, the telescope aperture D_tel and
    the angular slit width phi on the sky, 2 tan(theta_B) D_coll/(phi
    D_tel).

    Returns
    -------
    R : number or array
    """

    if phi is not None:
        return 2*np.tan(blaze)*D_coll/(np.asarray(phi, float)*D_tel)
    return 2*np.tan(blaze)*f_coll/np.asarray(slit_width, float)


#: Some strong Fraunhofer lines (nm, air) with depths and widths (nm) for
#: `solar_like_spectrum()`.
FRAUNHOFER_LINES = (
    ("Ca II K", 393.37, 0.85, 0.6), ("Ca II H", 396.85, 0.8, 0.5),
    ("H delta", 410.17, 0.6, 0.12), ("H gamma", 434.05, 0.65, 0.13),
    ("Fe I", 438.35, 0.6, 0.03), ("H beta", 486.13, 0.7, 0.12),
    ("Mg I b1", 518.36, 0.75, 0.05), ("Mg I b2", 517.27, 0.75, 0.045),
    ("Mg I b4", 516.73, 0.6, 0.03), ("Fe I", 527.04, 0.6, 0.02),
    ("Na I D2", 588.995, 0.8, 0.025), ("Na I D1", 589.592, 0.75, 0.022),
    ("H alpha", 656.28, 0.8, 0.1),
)


def solar_like_spectrum(wavelength, temperature=5800.0, lines=FRAUNHOFER_LINES):
    """A blackbody of `temperature` (photons per nm, peak 1) with Lorentzian
    absorption lines (name, centre, depth, FWHM).

    Parameters
    ----------
    wavelength : array (nm)

    Returns
    -------
    flux : array
    """

    lam = np.asarray(wavelength, float)
    hc_k = 1.4387769e7                                # nm K
    photons = lam**-4/np.expm1(hc_k/(lam*temperature))
    lam_peak = hc_k/(3.92069*temperature)             # peak of the photon spectrum
    photons /= lam_peak**-4/np.expm1(hc_k/(lam_peak*temperature))
    absorb = np.ones_like(lam)
    for _, c, depth, w in lines:
        absorb *= 1 - depth/(1 + ((lam - c)/(w/2))**2)
    return photons*absorb
