#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Propagation with analytic handling of the wavefront phase.

Behind a lens, near a point source or far from a waist, a field has a
strongly curved wavefront. Sampled as a complex array, its phase must change
by less than pi between neighbouring samples: a beam of width L with the
radius of curvature R needs about L**2/(lambda*|R|) samples per axis, often
thousands, although its amplitude is smooth and needs only a few dozen.
Wyrowski and coworkers therefore write the field as ::

    u(x, y) = A(x, y)*exp(i*psi(x, y)),

with a smooth, sampled amplitude A and a phase psi that is known
analytically, and carry psi through the operators of the propagation
(Fourier transforms, multiplication with the transfer function) in closed
form (Wang et al. 2019, 2020; Wyrowski and Kuhn 2011). Here psi is a
separable quadratic polynomial ::

    psi = alpha_x*x**2 + alpha_y*y**2 + beta_x*x + beta_y*y + gamma,

i.e. a spherical or astigmatic wavefront (alpha = k/(2R)) with a tilt
(beta, the carrier) and a constant. QuadraticPhaseField holds A, the grid,
the coefficients and the wavelength; the same class holds spectra, with
k_x, k_y in place of x, y.

Semi-analytical Fourier transform (SFT, Wang et al. 2019). Per axis, with
kappa = k - beta and x_kappa = kappa/(2*alpha), completing the square gives
the exact identity ::

    int A(x) exp(i*(alpha*x**2 + beta*x)) exp(-i*k*x) dx
        = exp(-i*kappa**2/(4*alpha)) (A * c)(x_kappa),    c(u) = exp(i*alpha*u**2).

The spectrum is again "amplitude times quadratic phase", with the analytic
phase -kappa**2/(4*alpha) and the amplitude (A * c) on the grid
k_j = beta + 2*alpha*x_j. The convolution with the chirp c is done with FFTs
(its spectrum is known in closed form, sqrt(i*pi/alpha) exp(-i*nu**2/(4*alpha))).
This is exact, and the sampling of the chirps is the only condition: with N
samples of spacing dx, the chirp exp(i*alpha*x**2) is sampled in x if
rho = |alpha|*N*dx**2/pi < 1 and its counterpart exp(-i*nu**2/(4*alpha)) in
the frequency domain if rho > 1. The transforms here pick per axis: for
rho < 1 the quadratic phase is multiplied into A and an ordinary (chirp-z)
FFT is used, for rho >= 1 the SFT ("mode" forces one of them). In the limit
rho >> 1 the chirp convolution becomes the identity: the SFT turns into the
pointwise (homeomorphic) transform below.

Propagation (propagate()). The transfer function exp(i*k_z*z) is split into
its paraxial part, exp(i*k*z)*exp(-i*z*(k_x**2 + k_y**2)/(2*k)), which is
added to the analytic quadratic phase of the spectrum, and the non-paraxial
remainder, which multiplies the sampled amplitude (paraxial=False, exact
angular spectrum propagation) or is dropped (paraxial=True, exact Fresnel
propagation). One forward and one inverse SFT, each with N samples per axis,
propagate a converging or diverging beam to any distance; the output grid
follows the beam (it scales with (R + z)/R like the geometric footprint), and
near a focus the inverse transform switches to the FFT and can be evaluated
on a chosen window (x_out, y_out, chirp-z transform).

Homeomorphic Fourier transform (HFT, Wang et al. 2020). If the phase
dominates (rho >> 1), the Fourier integral is given by its stationary point
x_s, grad psi(x_s) = k, alone ::

    U(k) = A(x_s) 2*pi/sqrt(|det H|) exp(i*pi*sigma(H)/4) exp(i*(psi(x_s) - k.x_s)),

with the Hessian H of psi and its signature sigma: each sample is mapped to
one point of the spectrum (homeomorphically), in O(N) operations and without
any FFT. homeomorphic_FT() evaluates it for any smooth phase given as an
unwrapped sampled array, geometric_propagation() chains it with the transfer
function and the inverse transform, which reduces exactly to geometric optics
with the transport of the amplitude: the sample at x moves to
x + z*grad(psi)/k_z, its amplitude scales with 1/sqrt(|det(I - z H G)|)
(G: Hessian of k_z) and it collects the phase z*k**2/k_z plus the Gouy phase
-pi/2 per crossed caustic. It fails near foci and caustics, where the SFT
(or the FFT) is required.

Tilted planes (tilted_plane()). Propagation onto a plane with a rotated
normal is a coordinate transformation of the spectrum: the plane wave with
wave vector k reaches the tilted plane with k^ = R^T k (Matsushima, Schimmel
and Wyrowski 2003, see pyoptics.tilted). The interpolation of the spectrum
onto a uniform k^ grid is the expensive part of the standard method, because
the spectrum of a curved beam is a strongly oscillating chirp. Here the
spectrum's phase is analytic: only the smooth amplitude is interpolated, the
phase psi~(k(k^)) + k.r0 is evaluated exactly at the new points, its best
separable quadratic (incl. the linear carrier, cf. Zhang et al. 2016) is kept
analytic and only the small remainder is sampled, followed by an inverse SFT
in the plane's own coordinates. So yes: the approach carries over to tilted
planes, with the same small N as for parallel planes.

Functions and classes:

- QuadraticPhaseField: amplitude, grid and analytic phase of a field or of
  a spectrum; field(), phase(), intensity(), resample(), required_samples()
- semi_analytical_FT(), inverse_semi_analytical_FT(): the transforms
- propagate(): propagation between parallel planes (angular spectrum or
  Fresnel)
- tilted_plane(): propagation onto a tilted plane
- homeomorphic_FT(): pointwise Fourier transform for any smooth phase
- geometric_propagation(): pointwise propagation (HFT, transfer function,
  inverse HFT)

Conventions as in pyoptics.propagators: fields vary as exp(i(k.r - omega*t)),
U(k) = int u(x) exp(-i*k.x) d^2x and u(x) = 1/(4*pi**2) int U(k) exp(+i*k.x)
d^2k (continuous transforms, i.e. including dx*dy). Grids must be linear and
equidistant; they need not be centered.

References
----------
Z. Wang, S. Zhang, O. Baladron-Zorita, C. Hellmann and F. Wyrowski,
"Application of the semi-analytical Fourier transform to electromagnetic
modeling", Opt. Express 27, 15335-15350 (2019).

Z. Wang, O. Baladron-Zorita, C. Hellmann and F. Wyrowski, "Theory and
algorithm of the homeomorphic Fourier transform for optical simulations",
Opt. Express 28, 10552-10571 (2020).

Z. Wang, O. Baladron-Zorita, C. Hellmann and F. Wyrowski, "Generalized Debye
integral", Opt. Express 28, 24459-24470 (2020).

F. Wyrowski and M. Kuhn, "Introduction to field tracing", J. Mod. Opt. 58,
449-466 (2011).

S. Zhang, D. Asoubar, C. Hellmann and F. Wyrowski, "Propagation of
electromagnetic fields between non-parallel planes: a fully vectorial
formulation and an efficient implementation", Appl. Opt. 55, 529-538 (2016).

K. Matsushima, H. Schimmel and F. Wyrowski, "Fast calculation method for
optical diffraction on tilted planes by use of the angular spectrum of plane
waves", J. Opt. Soc. Am. A 20, 1755-1762 (2003).
"""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates

from . import fft as _fft
from .fft import _zoom_dft
from .tilted import _rotation, _origin
from .utils import wavenumber


__all__ = ["QuadraticPhaseField", "semi_analytical_FT", "inverse_semi_analytical_FT",
           "propagate", "tilted_plane", "homeomorphic_FT", "geometric_propagation"]


def _pair(value):
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if value.shape == (1,):
        return float(value[0]), float(value[0])
    if value.shape != (2,):
        raise ValueError("expected a number or a pair of numbers")
    return float(value[0]), float(value[1])


def _grid(x, name):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise ValueError("{} must be a linear (1d) coordinate array".format(name))
    if len(x) > 1:
        dx = (x[-1] - x[0])/(len(x) - 1)
        if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1E-6, atol=1E-12*abs(dx)):
            raise ValueError("{} must be ascending and equidistant".format(name))
    return x


def _spacing(x):
    return (x[-1] - x[0])/(len(x) - 1)


class QuadraticPhaseField(object):
    """A field (or spectrum) u = A*exp(i*psi) with a sampled amplitude A and the
    analytic phase psi = alpha_x*x**2 + alpha_y*y**2 + beta_x*x + beta_y*y + gamma.

    Parameters
    ----------
    amplitude : array
        Complex amplitude A of shape (Ny, Nx). It must be smooth (sampled
        according to its own bandwidth); the wavefront goes into psi.
    x, y : arrays
        Linear, ascending, equidistant coordinates (need not be centered).
        For a spectrum: the wavenumbers k_x, k_y.
    wavelength : number
        Vacuum wavelength.
    alpha, beta : pairs of numbers, optional
        Quadratic and linear coefficients of psi in x and y.
    gamma : number, optional
        Constant phase.
    n : number, optional
        Refractive index of the medium.
    domain : str, optional
        "space" (default) or "spectrum".
    """

    def __init__(self, amplitude, x, y, wavelength, alpha=(0.0, 0.0), beta=(0.0, 0.0),
                 gamma=0.0, n=1.0, domain="space"):
        self.x = _grid(x, "x")
        self.y = _grid(y, "y")
        self.amplitude = np.asarray(amplitude, dtype=complex)
        if self.amplitude.shape != (len(self.y), len(self.x)):
            raise ValueError("amplitude must have the shape (len(y), len(x)) = ({}, {})"
                             .format(len(self.y), len(self.x)))
        self.wavelength = float(wavelength)
        self.n = n
        self.alpha = np.array(_pair(alpha))
        self.beta = np.array(_pair(beta))
        self.gamma = float(gamma)
        if domain not in ("space", "spectrum"):
            raise ValueError("domain must be 'space' or 'spectrum'")
        self.domain = domain

    @classmethod
    def from_curvature(cls, amplitude, x, y, wavelength, R=np.inf, carrier=(0.0, 0.0), n=1.0):
        """Field A*exp(i*k*(x**2/(2*R_x) + y**2/(2*R_y)))*exp(i*(c_x*x + c_y*y)).

        R > 0 is a diverging, R < 0 a converging wave (focus at z = -R),
        R = inf a plane wavefront; R may be a pair (R_x, R_y). carrier is the
        tilt (c_x, c_y) as transverse wavenumbers.
        """

        k = wavenumber(wavelength, n)
        alpha = [0.0 if np.isinf(r) else k/(2*r) for r in _pair(R)]
        return cls(amplitude, x, y, wavelength, alpha, carrier, 0.0, n)

    @classmethod
    def from_field(cls, field, x, y, wavelength, R=np.inf, carrier=(0.0, 0.0), n=1.0):
        """Split a sampled field into the analytic phase given by R and carrier
        (see from_curvature()) and the remaining amplitude."""

        out = cls.from_curvature(field, x, y, wavelength, R, carrier, n)
        out.amplitude = out.amplitude*np.exp(-1j*out.phase())
        return out

    @property
    def k(self):
        """Wavenumber in the medium."""
        return wavenumber(self.wavelength, self.n)

    @property
    def radius_of_curvature(self):
        """(R_x, R_y) = k/(2*alpha) of a field (inf for alpha = 0)."""
        if self.domain != "space":
            raise ValueError("a spectrum has no radius of curvature")
        return tuple(np.inf if a == 0 else self.k/(2*a) for a in self.alpha)

    def copy(self):
        return QuadraticPhaseField(self.amplitude.copy(), self.x.copy(), self.y.copy(),
                                   self.wavelength, self.alpha, self.beta, self.gamma,
                                   self.n, self.domain)

    def phase(self, x=None, y=None):
        """Analytic phase psi on the grid (or on the linear coordinates x, y)."""
        x = self.x if x is None else np.asarray(x, dtype=float)
        y = self.y if y is None else np.asarray(y, dtype=float)
        px = self.alpha[0]*x**2 + self.beta[0]*x
        py = self.alpha[1]*y**2 + self.beta[1]*y
        return py[:, None] + px[None, :] + self.gamma

    def field(self):
        """The complete field A*exp(i*psi) on the grid. Aliased where psi
        changes by more than pi between samples (see required_samples())."""
        return self.amplitude*np.exp(1j*self.phase())

    def intensity(self):
        """``|A|**2``: the intensity does not depend on psi."""
        return np.abs(self.amplitude)**2

    def required_samples(self):
        """Samples (N_x, N_y) that the complete field A*exp(i*psi) needs on the
        same window: the window length times the largest local frequency of
        psi, over pi, plus the samples of A."""
        out = []
        for c, a, b in ((self.x, self.alpha[0], self.beta[0]), (self.y, self.alpha[1], self.beta[1])):
            L = c[-1] - c[0] + (_spacing(c) if len(c) > 1 else 0.0)
            f_max = max(abs(2*a*c[0] + b), abs(2*a*c[-1] + b))
            out.append(int(np.ceil(L*f_max/np.pi)) + len(c))
        return tuple(out)

    def chirp_ratio(self):
        """rho = |alpha|*N*d**2/pi per axis: the transforms use the SFT for
        rho >= 1 and the FFT (phase multiplied into A) for rho < 1."""
        return tuple(_chirp_ratio(a, c) for a, c in ((self.alpha[0], self.x), (self.alpha[1], self.y)))

    def resample(self, x, y):
        """The same field on other linear coordinates: the amplitude is
        interpolated (cubic splines, zero outside the grid), the phase is
        evaluated analytically. Accurate as long as A is smooth (a
        RuntimeWarning is issued if its phase changes by more than pi/4
        between samples, e.g. behind an FFT step that multiplied the
        quadratic phase into A: pass the output grid to the transform
        instead)."""
        x, y = _grid(x, "x"), _grid(y, "y")
        A = self.amplitude
        step_phase = np.zeros(A.shape)
        step_phase[:, 1:] = np.cumsum(np.angle(A[:, 1:]*np.conj(A[:, :-1])), axis=1)
        _check_residual(step_phase, np.abs(A)**2, "resample()", axes=(1,))
        step_phase[1:, :] = np.cumsum(np.angle(A[1:, :]*np.conj(A[:-1, :])), axis=0)
        step_phase[0, :] = 0
        _check_residual(step_phase, np.abs(A)**2, "resample()", axes=(0,))
        A = _interp_axis(self.amplitude, self.x, x, axis=1)
        A = _interp_axis(A, self.y, y, axis=0)
        return QuadraticPhaseField(A, x, y, self.wavelength, self.alpha, self.beta,
                                   self.gamma, self.n, self.domain)

    def __repr__(self):
        return ("QuadraticPhaseField(domain={!r}, shape={}, alpha={}, beta={}, wavelength={:g})"
                .format(self.domain, self.amplitude.shape, self.alpha, self.beta, self.wavelength))


def _chirp_ratio(alpha, x):
    if len(x) < 2:
        return 0.0
    return abs(alpha)*len(x)*_spacing(x)**2/np.pi


def _interp_axis(A, x, x_new, axis):
    if len(x) == len(x_new) and np.allclose(x, x_new, rtol=0, atol=1E-12*max(1.0, np.max(np.abs(x)))):
        return A
    if len(x) < 4:
        raise ValueError("resampling needs at least 4 samples per axis")
    spline = CubicSpline(x, A, axis=axis, extrapolate=False)
    out = spline(x_new)
    return np.where(np.isnan(out), 0, out)


def _shape(axis, ndim, N):
    s = [1]*ndim
    s[axis] = N
    return s


def _transform_axis(A, x, alpha, beta, sign, axis, mode, out, pad):
    """int A(x) exp(i*(alpha*x**2 + beta*x)) exp(i*sign*k*x) dx along axis.

    Returns the amplitude B on the output grid k, the coefficients
    (alpha', beta', gamma') of the analytic phase of the result, and whether
    the SFT was used. With out given, the FFT branch evaluates the sum there
    (chirp-z); the SFT branch ignores it (its grid is fixed, the caller
    resamples).
    """

    N = len(x)
    dx = _spacing(x)
    xc = x[N//2]
    xi = x - xc
    if mode not in ("auto", "sft", "fft"):
        raise ValueError("mode must be 'auto', 'sft' or 'fft'")
    use_sft = alpha != 0 and (mode == "sft" or (mode == "auto" and _chirp_ratio(alpha, x) >= 1))
    shp = _shape(axis, A.ndim, N)

    if not use_sft:
        # (beta + s*k)*xc is analytic; the rest is a centered sum over xi.
        g = A*np.exp(1j*alpha*x**2).reshape(shp)
        if out is None:
            dk = 2*np.pi/(N*dx)
            q = dk*(np.arange(N) - N//2)
            if sign < 0:
                B = _fft.fftshift(_fft.fft(_fft.ifftshift(g, axes=axis), axis=axis), axes=axis)
            else:
                B = _fft.fftshift(_fft.ifft(_fft.ifftshift(g, axes=axis), axis=axis), axes=axis)*N
        else:
            q = np.asarray(out, dtype=float) + sign*beta
            B = _zoom_dft(g, xi, q, sign, axis)
        k = q - sign*beta
        return B*dx, k, (0.0, sign*xc, beta*xc), False

    # SFT: (A * c)(x_kappa) by FFT convolution with the analytic chirp spectrum
    Np = _fft._next_fast_len(int(np.ceil(pad*N))) if pad and pad > 1 else N
    off = (Np - N)//2
    widths = [(0, 0)]*A.ndim
    widths[axis] = (off, Np - N - off)
    Ap = np.pad(A, widths)
    nu = 2*np.pi*np.fft.fftfreq(Np, dx)
    c_hat = (np.sqrt(np.pi/abs(alpha))*np.exp(1j*np.sign(alpha)*np.pi/4)
             *np.exp(-1j*nu**2/(4*alpha)))
    conv = _fft.ifft(_fft.fft(Ap, axis=axis)*c_hat.reshape(_shape(axis, A.ndim, Np)), axis=axis)
    conv = np.take(conv, np.arange(off, off + N), axis=axis)
    k = -sign*(2*alpha*x + beta)
    if k[-1] < k[0]:
        k = k[::-1]
        conv = np.flip(conv, axis=axis)
    return conv, k, (-1/(4*alpha), -sign*beta/(2*alpha), -beta**2/(4*alpha)), True


def _transform(field, sign, x_out, y_out, mode, pad):
    A = field.amplitude
    coords, alpha, beta, gamma = [None, None], [0.0, 0.0], [0.0, 0.0], field.gamma
    sft_axes = []
    for i, (axis, c, out) in enumerate(((1, field.x, x_out), (0, field.y, y_out))):
        A, k, (a, b, g), used_sft = _transform_axis(A, c, field.alpha[i], field.beta[i], sign,
                                                    axis, mode, out, pad)
        coords[i], alpha[i], beta[i] = k, a, b
        gamma += g
        if used_sft and out is not None:
            sft_axes.append(i)
    if sign > 0:
        A = A/(4*np.pi**2)
    result = QuadraticPhaseField(A, coords[0], coords[1], field.wavelength, alpha, beta, gamma,
                                 field.n, "spectrum" if sign < 0 else "space")
    if sft_axes:
        result = result.resample(x_out if 0 in sft_axes else result.x,
                                 y_out if 1 in sft_axes else result.y)
    return result


def semi_analytical_FT(field, kx_out=None, ky_out=None, mode="auto", pad=2):
    """Spectrum U(k) = int u(x) exp(-i*k.x) d^2x of a QuadraticPhaseField.

    Parameters
    ----------
    field : QuadraticPhaseField
        Field in the space domain.
    kx_out, ky_out : arrays, optional
        Output wavenumbers (linear, equidistant). Axes transformed with the
        FFT are evaluated there directly (chirp-z transform), axes
        transformed with the SFT are interpolated. Default: the natural grid
        of each transform (N samples).
    mode : str, optional
        "auto" (default: SFT where the chirp ratio rho >= 1, see the module
        docstring), "sft" or "fft".
    pad : number, optional
        Zero padding factor of the chirp convolution in the SFT.

    Returns
    -------
    spectrum : QuadraticPhaseField
        Amplitude and analytic phase of U on the wavenumbers x = k_x,
        y = k_y (domain "spectrum").
    """

    if field.domain != "space":
        raise ValueError("semi_analytical_FT() needs a field in the space domain")
    return _transform(field, -1, kx_out, ky_out, mode, pad)


def inverse_semi_analytical_FT(spectrum, x_out=None, y_out=None, mode="auto", pad=2):
    """Field u(x) = 1/(4*pi**2) int U(k) exp(+i*k.x) d^2k of a spectrum; the
    inverse of semi_analytical_FT() (see there for the parameters)."""

    if spectrum.domain != "spectrum":
        raise ValueError("inverse_semi_analytical_FT() needs a spectrum")
    return _transform(spectrum, +1, x_out, y_out, mode, pad)


def _kz(k, KX, KY):
    return np.sqrt((k**2 - KX**2 - KY**2).astype(complex))


def _check_residual(phase, weight, what, axes=(0, 1)):
    """Warn if a sampled phase changes by more than pi/4 between neighbours
    where the amplitude matters."""
    significant = weight > 1E-4*np.max(weight) if np.max(weight) > 0 else weight > 0
    steps = []
    for axis in axes:
        if phase.shape[axis] > 1:
            d = np.abs(np.diff(phase, axis=axis))
            m = (np.take(significant, np.arange(phase.shape[axis] - 1), axis=axis)
                 & np.take(significant, np.arange(1, phase.shape[axis]), axis=axis))
            steps.append(np.max(d[m]) if np.any(m) else 0.0)
    step = max(steps) if steps else 0.0
    if step > np.pi/4:
        warnings.warn("{}: the sampled remainder of the phase changes by up to {:.2f} rad "
                      "between samples; refine the grid (more samples or a smaller window)"
                      .format(what, step), RuntimeWarning, stacklevel=3)
    return step


def propagate(field, z, paraxial=False, x_out=None, y_out=None, mode="auto", pad=2):
    """Propagate a QuadraticPhaseField by z between parallel planes.

    The forward transform, the transfer function and the inverse transform
    all keep the quadratic phase analytic (see the module docstring), so
    that N samples per axis suffice at any distance if the amplitude is
    smooth. The output grid follows the beam: away from a focus it is the
    input grid scaled by about (R + z)/R; near a focus (chirp ratio < 1) the
    inverse transform is an FFT whose window can be chosen with x_out, y_out.

    Parameters
    ----------
    field : QuadraticPhaseField
        Field in the plane z = 0.
    z : number
        Propagation distance (negative: backwards).
    paraxial : boolean, optional
        If True, propagate with the Fresnel transfer function
        exp(i*k*z)*exp(-i*z*(k_x**2 + k_y**2)/(2*k)), which is handled
        entirely analytically. If False (default), the non-paraxial
        remainder of exp(i*k_z*z) multiplies the sampled spectrum: exact
        angular spectrum (Rayleigh-Sommerfeld) propagation, as long as the
        remainder is sampled (a RuntimeWarning is issued if it changes by more
        than pi/4 between samples).
        Evanescent waves decay.
    x_out, y_out : arrays, optional
        Output coordinates (linear, equidistant), see
        semi_analytical_FT().
    mode, pad : optional
        See semi_analytical_FT().

    Returns
    -------
    field_z : QuadraticPhaseField
        Field in the plane z.
    """

    S = semi_analytical_FT(field, mode=mode, pad=pad)
    k = field.k
    if paraxial:
        S.alpha = S.alpha - z/(2*k)
        S.gamma = S.gamma + k*z
    else:
        KX, KY = np.meshgrid(S.x, S.y)
        k_perp2 = KX**2 + KY**2
        kz = _kz(k, KX, KY)
        propagating = k_perp2 < k**2
        weight = np.abs(S.amplitude)**2*propagating
        # analytic part: the best separable quadratic of k_z*z over the
        # spectrum (the paraxial expansion if the spectrum is narrow)
        if np.count_nonzero(weight > 1E-12*np.max(weight)) >= 5:
            a, b, c, d, e = _fit_separable_quadratic(z*np.real(kz), KX, KY,
                                                     weight*(weight > 1E-12*np.max(weight)))
        else:
            a, b, c, d, e = -z/(2*k), -z/(2*k), 0.0, 0.0, k*z
        quadratic = a*KX**2 + b*KY**2 + c*KX + d*KY + e
        S.alpha = S.alpha + (a, b)
        S.beta = S.beta + (c, d)
        S.gamma = S.gamma + e
        residual = np.where(propagating, z*np.real(kz) - quadratic, 0.0)
        _check_residual(residual, weight, "propagate()")
        factor = np.where(propagating, np.exp(1j*residual),
                          np.exp(-abs(z)*np.abs(np.imag(kz)))*np.exp(-1j*quadratic))
        S.amplitude = S.amplitude*factor
    return inverse_semi_analytical_FT(S, x_out, y_out, mode=mode, pad=pad)


def _fit_separable_quadratic(phase, KX, KY, weight):
    """Weighted least squares fit of a*kx**2 + b*ky**2 + c*kx + d*ky + e."""
    w = np.sqrt(weight.ravel())
    m = w > 0
    kx, ky = KX.ravel()[m], KY.ravel()[m]
    # scale the columns for a well conditioned system
    sx = max(np.max(np.abs(kx)), 1E-300)
    sy = max(np.max(np.abs(ky)), 1E-300)
    u, v = kx/sx, ky/sy
    M = np.stack([u**2, v**2, u, v, np.ones_like(u)], axis=1)*w[m, None]
    coeffs, *_ = np.linalg.lstsq(M, phase.ravel()[m]*w[m], rcond=None)
    a, b, c, d, e = coeffs
    return a/sx**2, b/sy**2, c/sx, d/sy, e


def tilted_plane(field, z0, R=None, angles=None, x_out=None, y_out=None, N_out=None,
                 k_margin=0.05, mode="auto", pad=2, threshold=1E-6):
    """Propagate a QuadraticPhaseField from z = 0 onto a tilted plane.

    Geometry as in pyoptics.tilted: the target plane passes through
    r0 = (0, 0, z0) (or r0 = z0 for a 3-vector), its local axes x^, y^ and
    normal are the columns of the rotation matrix R, and the field is
    returned in the local coordinates (x^, y^).

    Steps (see the module docstring): SFT of the source; a uniform grid of
    target wavenumbers k^ covering the rotated spectrum; k = R k^ and the
    Jacobian k_z/k^_z; cubic interpolation of the smooth spectral amplitude
    only; the exact phase psi~(k) + k.r0 at the new points, of which the best
    separable quadratic (incl. the carrier) is kept analytic; inverse SFT in
    the plane's coordinates. Only plane waves with k_z > 0 and k^_z > 0 are
    kept.

    Parameters
    ----------
    field : QuadraticPhaseField
        Field in the plane z = 0.
    z0 : number or array
        Distance of the target plane's origin, or its position (x0, y0, z0).
    R : array, optional
        Rotation matrix (e.g. geometry.R_y(theta)).
    angles : tuple, optional
        (theta_x, theta_y) instead of R, see tilted.rotation_from_angles().
    x_out, y_out : arrays, optional
        Output coordinates in the plane (linear, equidistant), see
        semi_analytical_FT(). Default: the natural grid of the inverse
        transform.
    N_out : int or pair, optional
        Samples of the target spectrum (and field) per axis. Default: twice
        the samples of the source (the remainder of the phase that is not
        quadratic in k^ must be sampled; a RuntimeWarning is issued
        otherwise).
    k_margin : number, optional
        Relative margin of the target spectrum around the rotated support.
    mode, pad : optional
        See semi_analytical_FT().
    threshold : number, optional
        Relative intensity of the spectrum that defines its support.

    Returns
    -------
    field_t : QuadraticPhaseField
        The field on the tilted plane, in its local coordinates.
    """

    R = _rotation(R, angles)
    r0 = _origin(z0)
    k = field.k
    S = semi_analytical_FT(field, mode=mode, pad=pad)
    KX, KY = np.meshgrid(S.x, S.y)
    KZ = np.real(_kz(k, KX, KY))
    P = np.abs(S.amplitude)**2
    support = (P > threshold*np.max(P)) & (KX**2 + KY**2 < k**2)
    if not np.any(support):
        raise ValueError("tilted_plane(): the field has no propagating plane waves")
    K_hat = np.einsum("ji,jab->iab", R, np.stack([KX, KY, KZ]))  # R^T k
    if np.any(K_hat[2][support] <= 0):
        warnings.warn("tilted_plane(): part of the spectrum travels away from the target "
                      "plane (k^_z <= 0) and is dropped", RuntimeWarning, stacklevel=2)
    Nx, Ny = (2*len(S.x), 2*len(S.y)) if N_out is None else tuple(int(v) for v in _pair(N_out))

    def target_axis(values, N):
        lo, hi = np.min(values), np.max(values)
        margin = k_margin*(hi - lo) + 1E-12*k
        return np.linspace(lo - margin, hi + margin, N)

    kx_t = target_axis(K_hat[0][support & (K_hat[2] > 0)], Nx)
    ky_t = target_axis(K_hat[1][support & (K_hat[2] > 0)], Ny)
    KXt, KYt = np.meshgrid(kx_t, ky_t)
    KZt2 = k**2 - KXt**2 - KYt**2
    KZt = np.sqrt(np.maximum(KZt2, 0.0))
    Kb = np.einsum("ij,jab->iab", R, np.stack([KXt, KYt, KZt]))  # back to source frame
    valid = (KZt2 > 0) & (Kb[2] > 0)

    # interpolate the smooth amplitude only
    ix = (Kb[0] - S.x[0])/_spacing(S.x)
    iy = (Kb[1] - S.y[0])/_spacing(S.y)
    coords = np.stack([iy, ix])
    B = (map_coordinates(S.amplitude.real, coords, order=3, mode="constant", cval=0.0)
         + 1j*map_coordinates(S.amplitude.imag, coords, order=3, mode="constant", cval=0.0))
    inside = (ix >= 0) & (ix <= len(S.x) - 1) & (iy >= 0) & (iy <= len(S.y) - 1)
    valid &= inside
    jac = np.where(valid, Kb[2]/np.where(valid, KZt, 1.0), 0.0)
    B = np.where(valid, B*jac, 0.0)

    # exact phase at the new points, best separable quadratic kept analytic
    phase = (S.alpha[0]*Kb[0]**2 + S.alpha[1]*Kb[1]**2 + S.beta[0]*Kb[0] + S.beta[1]*Kb[1]
             + Kb[0]*r0[0] + Kb[1]*r0[1] + Kb[2]*r0[2])
    weight = np.abs(B)**2
    a, b, c, d, e = _fit_separable_quadratic(phase, KXt, KYt, weight)
    residual = np.where(valid, phase - (a*KXt**2 + b*KYt**2 + c*KXt + d*KYt + e), 0.0)
    _check_residual(residual, weight, "tilted_plane()")
    St = QuadraticPhaseField(B*np.exp(1j*residual), kx_t, ky_t, field.wavelength, (a, b), (c, d),
                             S.gamma + e, field.n, "spectrum")
    return inverse_semi_analytical_FT(St, x_out, y_out, mode=mode, pad=pad)


def _hessian(phase, x, y):
    gy, gx = np.gradient(phase, y, x, edge_order=2)
    gxy, gxx = np.gradient(gx, y, x, edge_order=2)
    gyy, gyx = np.gradient(gy, y, x, edge_order=2)
    return gx, gy, gxx, 0.5*(gxy + gyx), gyy


def homeomorphic_FT(amplitude, phase, x, y):
    """Pointwise (homeomorphic) Fourier transform of A*exp(i*phase).

    Every sample x_j is mapped to the wavenumber k_j = grad phase(x_j), where
    it is the stationary point of the Fourier integral, and ::

        U(k_j) = A_j 2*pi/sqrt(|det H_j|) exp(i*pi*sigma_j/4) exp(i*(phase_j - k_j.x_j))

    (H: Hessian of the phase, sigma: its signature). Valid if the phase
    dominates (many oscillations across the beam, no caustic: det H != 0
    and A slowly varying on the scale of the stationary zone).

    Parameters
    ----------
    amplitude : array
        Smooth amplitude A, shape (Ny, Nx).
    phase : array
        Smooth, unwrapped phase on the same grid (it need not be sampled
        below pi per sample: unwrapped values are smooth).
    x, y : arrays
        Linear coordinates.

    Returns
    -------
    kx, ky : arrays
        Wavenumbers (non-uniform, shape (Ny, Nx)) of the spectrum samples.
    U_amplitude : array
        Complex amplitude (incl. 2*pi/sqrt|det H| and the signature phase).
    U_phase : array
        Unwrapped phase phase - k.x of the spectrum.
    """

    x, y = _grid(x, "x"), _grid(y, "y")
    X, Y = np.meshgrid(x, y)
    gx, gy, hxx, hxy, hyy = _hessian(np.asarray(phase, dtype=float), x, y)
    det = hxx*hyy - hxy**2
    trace = hxx + hyy
    # signature: both eigenvalues positive (+2), negative (-2) or mixed (0)
    sigma = np.where(det > 0, 2*np.sign(trace), 0)
    with np.errstate(divide="ignore"):
        U_amp = np.asarray(amplitude)*2*np.pi/np.sqrt(np.abs(det))*np.exp(1j*np.pi*sigma/4)
    return gx, gy, U_amp, phase - gx*X - gy*Y


def geometric_propagation(amplitude, phase, x, y, z, wavelength, n=1.0):
    """Pointwise propagation: HFT, transfer function exp(i*k_z*z), inverse HFT.

    The chain of the two stationary phase evaluations is evaluated in closed
    form (no division by det H, so plane wavefronts are allowed): the sample
    at x moves along its ray to x + z*k_perp/k_z with k_perp = grad phase,
    and ::

        u = A/sqrt(|det M|) exp(i*(phase + z*k**2/k_z)) exp(-i*pi*n_neg/2),
        M = I - z*H*G,

    with the Hessians H of the phase and G of k_z(k_x, k_y), and the number
    n_neg of negative eigenvalues of M (foci crossed: the Gouy phase). This
    is geometrical optics with amplitude transport; it diverges at foci and
    caustics (det M = 0) and ignores diffraction at the edges of A.

    Parameters
    ----------
    amplitude, phase : arrays
        Smooth amplitude and unwrapped phase, shape (Ny, Nx).
    x, y : arrays
        Linear coordinates.
    z : number
        Distance.
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.

    Returns
    -------
    x_z, y_z : arrays
        Positions (shape (Ny, Nx), non-uniform) of the samples in the plane z.
    u_amplitude : array
        Complex amplitude there (incl. the Gouy phase).
    u_phase : array
        Unwrapped phase there, phase + z*k**2/k_z.
    """

    x, y = _grid(x, "x"), _grid(y, "y")
    k = wavenumber(wavelength, n)
    X, Y = np.meshgrid(x, y)
    phase = np.asarray(phase, dtype=float)
    kx, ky, hxx, hxy, hyy = _hessian(phase, x, y)
    kz2 = k**2 - kx**2 - ky**2
    if np.any(kz2 <= 0):
        raise ValueError("geometric_propagation(): the phase gradient exceeds k (evanescent)")
    kz = np.sqrt(kz2)
    # Hessian of k_z
    gxx = -1/kz - kx**2/kz**3
    gyy = -1/kz - ky**2/kz**3
    gxy = -kx*ky/kz**3
    # M = I - z*H*G
    m11 = 1 - z*(hxx*gxx + hxy*gxy)
    m12 = -z*(hxx*gxy + hxy*gyy)
    m21 = -z*(hxy*gxx + hyy*gxy)
    m22 = 1 - z*(hxy*gxy + hyy*gyy)
    det = m11*m22 - m12*m21
    tr = m11 + m22
    disc = np.sqrt(np.maximum(tr**2/4 - det, 0.0))
    n_neg = (tr/2 - disc < 0).astype(int) + (tr/2 + disc < 0).astype(int)
    with np.errstate(divide="ignore"):
        u_amp = np.asarray(amplitude)/np.sqrt(np.abs(det))*np.exp(-1j*np.pi*n_neg/2)
    return X + z*kx/kz, Y + z*ky/kz, u_amp, phase + z*k**2/kz
