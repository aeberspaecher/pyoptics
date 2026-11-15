#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Trigonometric (Fourier) interpolation of images and fields, with spectral
windows against the Gibbs phenomenon.

N equidistant samples f_l = f(x_0 + l*dx) of a periodic function with the
period L = N*dx define the trigonometric polynomial ::

    p(x) = sum_m c_m*w_m*exp(i*2*pi*f_m*(x - x_0)),   c_m = 1/N sum_l f_l*exp(-i*2*pi*m*l/N),

with the frequencies f_m = m/L, m = -N/2 ... N/2 (for even N the Nyquist
coefficient is split in halves onto -N/2 and +N/2, so real samples give a
real p(x)), and with all weights w_m = 1, p(x_l) = f_l. If f is band limited
to below the Nyquist frequency 1/(2*dx) (all its harmonics are among the
f_m), p(x) = f(x) everywhere: the interpolation is exact. Aerial images are
such functions: the intensity behind a lens of numerical aperture NA holds
the spatial frequencies up to 2*NA/wavelength only, so a grid with
dx <= wavelength/(4*NA) carries the whole image and everything between the
samples follows from the samples (Whittaker-Shannon). Fields (amplitudes)
are band limited to NA/wavelength.

Data that are not band limited (a binary mask, a thresholded resist pattern,
a non-periodic image whose ends do not match) make the series ring next to
every jump, however fine the grid (Gibbs phenomenon): the Fourier series of
a step overshoots by 9 % of the jump, the interpolant through samples of a
step by 14 %, and the ripples spread over the whole period. Spectral windows
w_m that fall smoothly to zero at the cutoff trade the ringing for a broader
edge; then p(x) no longer passes through the samples. Windows of
smoothing_window(), as functions of nu = f/f_cutoff (overshoot of the
Fourier series of a step, 10-90 % rise in units of 1/(2*f_cutoff)):

- "none" (or None): w = 1, the plain interpolation (Dirichlet kernel);
  9 %, rise 0.9,
- "lanczos": Lanczos' sigma factors ``w = sinc(nu)**order``; one factor
  averages p over one period of the highest harmonic: 1.2 %, rise 1.6;
  order 2: 0.2 %, rise 2.2,
- "fejer": ``w = 1 - |nu|``, the Cesaro mean of the partial sums. Its kernel is
  positive: no overshoot at all, rise 2.1. (The OTF of an incoherent 1D
  imaging system is such a triangle.)
- "hann": ``w = cos(pi*nu/2)**2``, raised cosine: 0.6 %, rise 1.9,
- "gaussian": ``w = exp(-nu**2/(2*s**2))`` with s = 0.4/order, the transfer
  function of a Gaussian blur (e.g. acid diffusion in a resist): 0.1 %,
  rise 2.0.

In 2D the windows are products of the 1D windows (separable, the usual
choice for sigma factors on a rectangular spectrum) or functions of the
radial frequency (radial=True, round cutoff as set by a circular pupil).

Functions and classes:

- smoothing_window(), lanczos_sigma(): the window weights,
- trig_interpolate(): along one axis, at arbitrary points,
- trig_interpolate_2d(): images and fields on arbitrary (rectilinear)
  output grids or at scattered points,
- fourier_upsample(): refinement by an integer factor with FFTs (zero
  padding of the spectrum), the fast way to a finer equidistant grid,
- TrigInterpolator: an image with an optional frequency cutoff (e.g. the
  incoherent cutoff 2*NA/wavelength), callable at any points,
- level_crossings(): positions where the interpolant of a periodic 1D cut
  crosses a level, located on the trigonometric polynomial itself (for
  threshold CDs, see litho.metrics.threshold_cd(interpolation="fourier")).

Conventions: x must be equidistant and represents one period; the sample at
x_0 + L is not included (utils.grid1d(..., assume_periodicity=True)
conventions). Frequencies are spatial frequencies (cycles per length unit).
Complex input gives complex output, real input real output.

References
----------
C. Lanczos, "Applied Analysis", Prentice Hall (1956), ch. IV.

J. P. Boyd, "Chebyshev and Fourier Spectral Methods", 2nd ed., Dover (2001),
ch. 2 and 11 (trigonometric interpolation, filters against Gibbs).

D. Gottlieb and C.-W. Shu, "On the Gibbs phenomenon and its resolution",
SIAM Review 39, 644-668 (1997).

J. W. Goodman, "Introduction to Fourier Optics", 3rd ed., Roberts (2005),
ch. 6 (band limits of coherent and incoherent images).
"""

import numpy as np

from . import fft as _fft
from .utils import freq_grid


WINDOWS = ("none", "lanczos", "fejer", "hann", "gaussian")


def smoothing_window(nu, kind="lanczos", order=1):
    """Spectral window against the Gibbs phenomenon.

    Parameters
    ----------
    nu : array
        Frequencies relative to the cutoff, f/f_cutoff (the window is zero
        for ``|nu| > 1``).
    kind : {"none", "lanczos", "fejer", "hann", "gaussian"} or None, optional
        See the module docstring.
    order : number, optional
        Power of the Lanczos sigma factors (1: Lanczos' sigma factors, 2:
        twice smoothed); for "gaussian", the standard deviation of the
        Gaussian is 0.4/order in units of the cutoff (order 1: w = 0.044 at
        the cutoff). Ignored by the other windows.

    Returns
    -------
    w : array
        Window weights, w(0) = 1.
    """

    nu = np.abs(np.asarray(nu, dtype=float))
    inside = nu <= 1.0
    if kind is None or kind == "none":
        w = np.ones_like(nu)
    elif kind == "lanczos":
        w = np.sinc(nu)**order
    elif kind == "fejer":
        w = 1.0 - nu
    elif kind == "hann":
        w = np.cos(np.pi*nu/2)**2
    elif kind == "gaussian":
        s = 0.4/order
        w = np.exp(-nu**2/(2*s**2))
    else:
        raise ValueError("Unknown window {!r}, use one of {}".format(kind, WINDOWS))

    if kind in ("lanczos", "fejer", "hann"):
        inside = nu < 1.0  # exactly zero at the cutoff
    return np.where(inside, w, 0.0)


def lanczos_sigma(N, order=1, normal_order=False):
    """Lanczos sigma factors for the DFT frequencies of N samples.

    ``sigma_m = sinc(m/(N/2))**order``, m the frequency index (numpy.fft.fftfreq
    order unless normal_order); the Nyquist coefficient of even N gets zero.

    Parameters
    ----------
    N : int
        Number of samples.
    order : number, optional
        Power of the sigma factors.
    normal_order : bool, optional
        Zero frequency at index N//2 (pyoptics.fft.FT() order).

    Returns
    -------
    sigma : array of length N
        Multiply a spectrum with it.
    """

    m = np.fft.fftfreq(N)*N
    sigma = smoothing_window(m/(N/2), "lanczos", order)

    return np.fft.fftshift(sigma) if normal_order else sigma


def _check_grid(x, N, name="x"):
    """Start and step of an equidistant coordinate array of length N."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) != N:
        raise ValueError("{} must be 1d with {} entries (the length of the data axis)".format(name, N))
    if N == 1:
        raise ValueError("At least two samples are needed")
    dx = (x[-1] - x[0])/(N - 1)
    if not np.allclose(np.diff(x), dx, rtol=1E-6, atol=1E-12*abs(dx)):
        raise ValueError("{} must be equidistant".format(name))

    return x[0], dx


def _split_nyquist(C, axis):
    """Spectrum in fftfreq order -> (coefficients, integer frequencies) with
    the Nyquist coefficient of even lengths split onto -N/2 and +N/2."""

    C = np.moveaxis(C, axis, 0)
    N = C.shape[0]
    m = np.fft.fftfreq(N)*N
    if N % 2 == 0:
        C = np.concatenate([C, C[N//2:N//2 + 1]], axis=0)
        C[N//2] *= 0.5
        C[N] *= 0.5
        m = np.concatenate([m, [N/2]])

    return np.moveaxis(C, 0, axis), np.rint(m).astype(int)


def _nu(f, f_cutoff, f_nyquist):
    """Frequencies relative to the cutoff (Nyquist if None)."""

    return f/(f_nyquist if f_cutoff is None else f_cutoff)


def trig_interpolate(f, x, x_out, axis=-1, smoothing=None, order=1, f_cutoff=None):
    """Trigonometric interpolation along one axis at arbitrary points.

    Parameters
    ----------
    f : array
        Samples (real or complex), periodic along axis.
    x : 1d array
        Equidistant sample positions along axis (one period, without the
        repeated end point).
    x_out : array_like
        Output positions (any, also outside the period: the interpolant is
        periodic).
    axis : int, optional
    smoothing : str or None, optional
        Window against the Gibbs phenomenon, see `smoothing_window()`.
    order : number, optional
        See `smoothing_window()`.
    f_cutoff : number, optional
        Frequency cutoff (cycles per length unit); the window is scaled to
        it and the harmonics beyond are dropped. Default: the Nyquist
        frequency 1/(2*dx).

    Returns
    -------
    p : array
        f with the axis replaced by np.shape(x_out) (scalar x_out: axis
        removed).
    """

    f = np.asarray(f)
    axis = axis % f.ndim
    N = f.shape[axis]
    x0, dx = _check_grid(x, N)
    C, m = _split_nyquist(_fft.fft(f, axis=axis)/N, axis)
    freq = m/(N*dx)
    w = smoothing_window(_nu(freq, f_cutoff, 0.5/dx), smoothing, order)
    keep = w != 0
    x_out = np.asarray(x_out, dtype=float)
    E = np.exp(2j*np.pi*np.outer(x_out.ravel() - x0, freq[keep]))*w[keep]
    C = np.moveaxis(np.compress(keep, C, axis=axis), axis, -1)
    p = (C @ E.T).reshape(C.shape[:-1] + x_out.shape)
    k = x_out.ndim
    p = np.moveaxis(p, list(range(f.ndim - 1, f.ndim - 1 + k)), list(range(axis, axis + k)))

    return p.real if np.isrealobj(f) else p


def _spectrum_2d(f, x, y, smoothing, order, f_cutoff, radial):
    """Weighted, Nyquist split coefficients of the nonzero window entries:
    (coefficients (Ny', Nx', ...), f_x (Nx'), f_y (Ny'), x0, y0)."""

    Ny, Nx = f.shape[:2]
    x0, dx = _check_grid(x, Nx, "x")
    y0, dy = _check_grid(y, Ny, "y")
    C = _fft.fft(_fft.fft(f, axis=0), axis=1)/(Nx*Ny)
    C, m = _split_nyquist(C, 1)
    C, n = _split_nyquist(C, 0)
    fx, fy = m/(Nx*dx), n/(Ny*dy)
    if f_cutoff is None:
        fcx, fcy = 0.5/dx, 0.5/dy
    else:
        fcx, fcy = np.broadcast_to(np.asarray(f_cutoff, dtype=float), (2,))
    if radial:
        nu = np.hypot(fx[None, :]/fcx, fy[:, None]/fcy)
        W = smoothing_window(nu, smoothing, order)
    else:
        W = np.outer(smoothing_window(fy/fcy, smoothing, order),
                     smoothing_window(fx/fcx, smoothing, order))
    rows, cols = np.any(W != 0, axis=1), np.any(W != 0, axis=0)
    W = W[np.ix_(rows, cols)]
    C = C[np.ix_(rows, cols)]*W.reshape(W.shape + (1,)*(C.ndim - 2))

    return C, fx[cols], fy[rows], x0, y0


def trig_interpolate_2d(f, x, y, x_out, y_out, smoothing=None, order=1, f_cutoff=None, radial=False,
                        grid=True, chunk=4096):
    """Trigonometric interpolation of an image or field.

    Parameters
    ----------
    f : array (Ny, Nx, ...)
        Samples on the periodic grid x, y; trailing axes (e.g. field
        components) are interpolated alike.
    x, y : 1d arrays
        Equidistant coordinates (one period each).
    x_out, y_out : arrays
        grid=True: 1d arrays of output coordinates (any spacing); grid=False:
        scattered points, arrays of the same shape.
    smoothing, order : optional
        Window against the Gibbs phenomenon, see `smoothing_window()`.
    f_cutoff : number or (f_x, f_y), optional
        Cutoff frequency (cycles per length unit). Default: the Nyquist
        frequencies of x and y.
    radial : bool, optional
        Window and cutoff as functions of the radial frequency
        ``hypot(f_x/f_cx, f_y/f_cy)`` (e.g. a round pupil) instead of
        products of 1D windows.
    grid : bool, optional
    chunk : int, optional
        Scattered points evaluated per block.

    Returns
    -------
    p : array
        grid=True: (len(y_out), len(x_out), ...); else np.shape(x_out) + (...).
    """

    f = np.asarray(f)
    C, fx, fy, x0, y0 = _spectrum_2d(f, x, y, smoothing, order, f_cutoff, radial)
    tail = C.shape[2:]
    C2 = C.reshape(C.shape[:2] + (-1,))
    if grid:
        Ex = np.exp(2j*np.pi*np.outer(np.asarray(x_out, dtype=float) - x0, fx))
        Ey = np.exp(2j*np.pi*np.outer(np.asarray(y_out, dtype=float) - y0, fy))
        p = np.einsum("jm,mnk,ln->jlk", Ey, C2, Ex)
        p = p.reshape(p.shape[:2] + tail)
    else:
        xo, yo = np.broadcast_arrays(np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float))
        xs, ys = xo.ravel() - x0, yo.ravel() - y0
        p = np.empty((len(xs), C2.shape[2]), dtype=complex)
        for s in range(0, len(xs), chunk):
            Ex = np.exp(2j*np.pi*np.outer(xs[s:s + chunk], fx))
            Ey = np.exp(2j*np.pi*np.outer(ys[s:s + chunk], fy))
            p[s:s + chunk] = np.einsum("pm,mnk,pn->pk", Ey, C2, Ex)
        p = p.reshape(xo.shape + tail)

    return p.real if np.isrealobj(f) else p


def upsampled_grid(x, factor):
    """Coordinates of fourier_upsample() for the input coordinates x.

    Parameters
    ----------
    x : 1d array
        Equidistant input coordinates (one period).
    factor : int

    Returns
    -------
    x_fine : 1d array
        len(x)*factor points from x[0] with the step dx/factor.
    """

    x0, dx = _check_grid(x, len(x))

    return x0 + dx/factor*np.arange(len(x)*factor)


def fourier_upsample(f, factor, axes=(0, 1), smoothing=None, order=1):
    """Refine a periodic image by an integer factor with FFTs.

    The spectrum is zero padded (the Nyquist coefficient of even lengths
    split symmetrically), so the result equals trig_interpolate_2d() on the
    grid of upsampled_grid() for every axis, at FFT speed.

    Parameters
    ----------
    f : array
        Samples, periodic along axes.
    factor : int or sequence of ints
        Refinement per axis.
    axes : int or sequence of ints, optional
    smoothing, order : optional
        Window against the Gibbs phenomenon (product over the axes), see
        `smoothing_window()`.

    Returns
    -------
    g : array
        Shape of f with each of the axes multiplied by its factor; g[::factor]
        equals f (without smoothing).
    """

    f = np.asarray(f)
    axes = np.atleast_1d(axes)
    factors = np.broadcast_to(np.atleast_1d(factor), axes.shape)
    g = f
    for ax, q in zip(axes, factors):
        q = int(q)
        N = g.shape[ax]
        G, m = _split_nyquist(_fft.fft(g, axis=ax), ax)
        w = smoothing_window(m/(N/2), smoothing, order)
        shape = [1]*G.ndim
        shape[ax] = len(m)
        G = G*w.reshape(shape)
        M = N*q
        P = np.zeros(G.shape[:ax] + (M,) + G.shape[ax + 1:], dtype=complex)
        idx = np.mod(m, M)  # m >= -N/2 > -M/2 for q >= 2; for q = 1 the split halves add up
        np.add.at(np.moveaxis(P, ax, 0), idx, np.moveaxis(G, ax, 0))
        g = _fft.ifft(P, axis=ax)*q

    return g.real if np.isrealobj(f) else g


def level_crossings(x, f, level, smoothing=None, order=1, f_cutoff=None, refine=8, tol=1E-12):
    """Positions where the trigonometric interpolant of a periodic cut
    crosses a level.

    The crossings are bracketed on a grid refined by fourier_upsample() and
    then located on the trigonometric polynomial itself by bisection and
    secant steps (Brent), so they are exact for band-limited data sampled
    above Nyquist, however coarse the grid.

    Parameters
    ----------
    x : 1d array
        Equidistant sample positions (one period).
    f : 1d array
        Real samples.
    level : number
    smoothing, order, f_cutoff : optional
        See `trig_interpolate()`.
    refine : int, optional
        Refinement for bracketing (crossings closer than dx/refine may be
        missed).
    tol : number, optional
        Tolerance of the positions relative to the sample spacing.

    Returns
    -------
    crossings : 1d array
        Sorted positions in [x[0], x[0] + L).
    rising : 1d array of bool
        True where f increases through the level.
    """

    from scipy.optimize import brentq

    f = np.asarray(f, dtype=float)
    N = len(f)
    x0, dx = _check_grid(x, N)
    C, m = _split_nyquist(_fft.fft(f)/N, 0)
    freq = m/(N*dx)
    w = smoothing_window(_nu(freq, f_cutoff, 0.5/dx), smoothing, order)
    keep = w != 0
    c, fr = C[keep]*w[keep], freq[keep]

    def p(u):
        """Interpolant minus the level at the points u."""
        return np.real(np.exp(2j*np.pi*np.outer(np.atleast_1d(u) - x0, fr)) @ c) - level

    M = N*refine
    u = x0 + dx/refine*np.arange(M + 1)
    g = p(u)
    sg = np.sign(g)
    out, rising = [], []
    for i in range(M):
        if sg[i] == 0 and sg[i + 1] != 0:
            out.append(u[i])
            rising.append(sg[i + 1] > 0)
        elif sg[i]*sg[i + 1] < 0:
            try:
                r = brentq(lambda s: p(s)[0], u[i], u[i + 1], xtol=tol*abs(dx))
            except ValueError:  # |p| at the roundoff level at an end of the bracket
                r = u[i] + (u[i + 1] - u[i])*g[i]/(g[i] - g[i + 1])
            out.append(r)
            rising.append(sg[i + 1] > 0)
    out = np.array(out)
    rising = np.array(rising, dtype=bool)
    L = N*dx
    out = x0 + np.mod(out - x0, L)
    order_ = np.argsort(out)

    return out[order_], rising[order_]


class ImageInterpolator(object):
    """Base class: an image on an equidistant, periodic grid.

    Parameters
    ----------
    image : array (len(y), len(x), ...)
    x, y : 1d arrays
    """

    def __init__(self, image, x, y):
        self.image = np.asarray(image)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        _check_grid(self.x, self.image.shape[1], "x")
        _check_grid(self.y, self.image.shape[0], "y")
        self.full_spectrum = _fft.FT(self.image) if self.image.ndim == 2 else None

    def __call__(self, x, y, grid=True):
        raise NotImplementedError


class TrigInterpolator(ImageInterpolator):
    """Trigonometric interpolation for images recorded through an imaging
    system, optionally with a frequency cutoff and a window.

    A typical cutoff frequency is 2*NA/wavelength for intensities (image
    space NA, any coherence), NA/wavelength for coherent fields. Spectral
    content beyond the cutoff (noise, aliasing) is removed.

    Parameters
    ----------
    image : array (len(y), len(x), ...)
        Samples on one period (real or complex).
    x, y : 1d arrays
        Equidistant coordinates.
    f_cutoff : number, optional
        Cutoff frequency (cycles per length unit). None: Nyquist (plain
        interpolation).
    assume_circular_NA : bool, optional
        Round cutoff (frequencies with ``f_x**2 + f_y**2 > f_cutoff**2`` are
        dropped) instead of a square one.
    smoothing, order : optional
        Window against the Gibbs phenomenon, see `smoothing_window()`,
        scaled to the cutoff.

    Attributes
    ----------
    F_X, F_Y : 1d arrays
        Spatial frequencies inside the square of the cutoff (normal order),
        flattened; truncated_spectrum holds the FT() coefficients there.
    """

    def __init__(self, image, x, y, f_cutoff=None, assume_circular_NA=True, smoothing=None, order=1):
        super(TrigInterpolator, self).__init__(image, x, y)
        self.assume_circular_NA = assume_circular_NA
        self.smoothing, self.order = smoothing, order
        self.f_cutoff = f_cutoff

    @property
    def f_cutoff(self):
        """Cutoff frequency (cycles per length unit), None for Nyquist."""
        return self._f_cutoff

    @f_cutoff.setter
    def f_cutoff(self, value):
        """Set the cutoff and recompute the truncated spectrum."""
        self._f_cutoff = value
        self._recompute_spectrum()

    def _recompute_spectrum(self):
        if self.full_spectrum is None:
            self.F_X = self.F_Y = self.truncated_spectrum = None
            return
        F_X, F_Y = freq_grid(self.x, self.y, wavenumbers=False, normal_order=True)
        spectrum = self.full_spectrum.copy()
        f_c = np.inf if self.f_cutoff is None else self.f_cutoff
        if self.assume_circular_NA:
            spectrum[F_X**2 + F_Y**2 > f_c**2] = 0.0
        allowed = (np.abs(F_X) <= f_c) & (np.abs(F_Y) <= f_c)
        self.F_X = F_X[allowed]
        self.F_Y = F_Y[allowed]
        self.truncated_spectrum = spectrum[allowed]

    def __call__(self, x, y, grid=True):
        """Interpolated values.

        Parameters
        ----------
        x, y : arrays
            grid=True: 1d output coordinates; else scattered points.
        grid : bool, optional

        Returns
        -------
        values : array
            (len(y), len(x), ...) or np.shape(x) + (...).
        """

        radial = self.assume_circular_NA and self.f_cutoff is not None
        f_c = self.f_cutoff
        if f_c is not None:
            dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
            # never beyond Nyquist (the window would not reach zero there):
            f_c = (min(f_c, 0.5/abs(dx)), min(f_c, 0.5/abs(dy))) if not radial else f_c

        return trig_interpolate_2d(self.image, self.x, self.y, x, y, self.smoothing, self.order,
                                   f_c, radial=radial, grid=grid)
