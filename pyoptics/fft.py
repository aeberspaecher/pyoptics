#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Helper module to deal with FFTs.

Convention: fields vary as exp(i(k.r - omega*t)). A field is a superposition
of plane waves exp(+i*(k_x*x + k_y*y)); the forward transform FT() has the
kernel exp(-i*(k_x*x + k_y*y)) and yields their amplitudes, the inverse
transform inv_FT() sums them up again::

    FT(f)[m, n] = sum_jl f[j, l]*exp(-i*(k_x[n]*x[l] + k_y[m]*y[j])),
    inv_FT(F)[j, l] = 1/(Nx*Ny) sum_mn F[m, n]*exp(+i*(k_x[n]*x[l] + k_y[m]*y[j])).

Both take and return centered arrays: x = 0 at index N//2 in position space
and k = 0 at index N//2 in frequency space, the latter matching
utils.frequencies() and utils.freq_grid() with normal_order=True. For even N,
utils.grid1d(-L/2, L/2, N) gives such a centered x. For odd N use e.g.
dx*(np.arange(N) - N//2). (If x is not centered, FT() equals the sum above up
to the constant phase factor exp(+i*k*x[N//2]), which inv_FT() removes again.)

FT() and inv_FT() are unnormalized like numpy.fft.fft2() and ifft2(); multiply
FT() by dx*dy to approximate the continuous Fourier integral. FT_unitary()
and inv_FT_unitary() are scaled by 1/sqrt(Nx*Ny) and sqrt(Nx*Ny) such that
FT_unitary() conserves the sum of ``|f|**2``.

rFT() and inv_rFT() are the versions for real input. Their spectra only hold
the non-negative k_x (the columns N//2, N//2 + 1, ... of the FT() spectrum,
for even N followed by the Nyquist frequency); only the k_y axis is centered.

czt() is Bluestein's chirp-z transform. It evaluates sums of the above kind
for arbitrary, equidistant output frequencies (zoomed or shifted spectra)
with FFTs, in O((N + M)*log(N + M)) operations. zoom_FT() and zoom_inv_FT()
use it to evaluate FT() and inv_FT() on any equidistant grid of output
frequencies or positions, e.g. a fine grid over a small part of a spectrum.

If installed, pyFFTW is used (via transparent_pyfftw or pyfftw's NumPy
interface); its transforms use the same sign convention as numpy.fft.
"""

import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift

try:
    from transparent_pyfftw.numpy_fft import fft, ifft, fft2, ifft2, rfft2, irfft2
    from transparent_pyfftw import save_wisdom
except ImportError:
    try:  # the 'fftw' extra installs pyfftw itself
        from pyfftw.interfaces.numpy_fft import fft, ifft, fft2, ifft2, rfft2, irfft2
    except ImportError:  # no FFTW at all: use NumPy's FFTs
        from numpy.fft import fft, ifft, fft2, ifft2, rfft2, irfft2

    def save_wisdom():
        """No-op: there is no FFTW wisdom to save without transparent_pyfftw."""
        pass


FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: fftshift(ifft2(ifftshift(x)))

FT_unitary = lambda x: FT(x)/np.sqrt(np.prod(np.shape(x)))
inv_FT_unitary = lambda x: inv_FT(x)*np.sqrt(np.prod(np.shape(x)))


def rFT(x):
    """FT() for real x. Returns the spectrum for k_x >= 0 only (shape
    (Ny, Nx//2 + 1)); k_y is centered.
    """

    return fftshift(rfft2(ifftshift(x)), axes=0)


def inv_rFT(X, shape=None):
    """Inverse of rFT(). shape is the shape of the real array and needs to be
    given for an odd number of x samples.
    """

    if shape is None:
        shape = (np.shape(X)[0], 2*(np.shape(X)[1] - 1))

    return fftshift(irfft2(ifftshift(X, axes=0), s=shape))


def rFT_unitary(x):
    return rFT(x)/np.sqrt(np.prod(np.shape(x)))


def inv_rFT_unitary(X, shape=None):
    x = inv_rFT(X, shape)

    return x*np.sqrt(np.prod(np.shape(x)))


# TODO: add more routines that avoid fftshift?


def _next_fast_len(n):
    """Smallest product of powers of 2, 3 and 5 that is >= n."""

    best = 2**int(np.ceil(np.log2(n)))
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5

    return best


def czt(x, M=None, w=None, a=1.0, axis=-1):
    """Chirp-z transform (Bluestein's algorithm).

    Computes ::

        X[m] = sum_{n=0}^{N-1} x[n]*a**(-n)*w**(n*m),   m = 0, ..., M-1,

    i.e. the z transform of x on the points z_m = a*w**(-m), with FFTs of
    length >= N + M - 1. With a = 1, w = exp(-2j*pi/N) and M = N, this is the
    DFT numpy.fft.fft(x). For a = exp(1j*theta_0) and w = exp(-1j*dtheta),
    the points z_m lie on the unit circle at the angles theta_0 + m*dtheta
    (a zoomed DFT); this is the case used in pyoptics. (scipy.signal.czt()
    uses the same definition.)

    Parameters
    ----------
    x : array
        Input data; the transform runs along axis.
    M : int, optional
        Number of output points. Defaults to the length N of x along axis.
    w : complex, optional
        Ratio between subsequent points. Defaults to exp(-2j*pi/M).
    a : complex, optional
        Starting point. Defaults to 1.
    axis : int, optional
        Axis to transform.

    Returns
    -------
    X : array
        Complex array with M points along axis.

    Notes
    -----
    Bluestein's identity n*m = (n**2 + m**2 - (m - n)**2)/2 turns the sum into
    a convolution with the chirp w**(-n**2/2), evaluated cyclically by FFT.
    The chirps are computed as exp(log(w)*n**2/2) from a single logarithm,
    which keeps the product of the three chirps equal to w**(n*m) for any
    branch of the logarithm. For ``|w|`` != 1 the chirps grow or decay
    exponentially; this is accurate only for moderate N and M.

    References
    ----------
    L. Bluestein, "A linear filtering approach to the computation of discrete
    Fourier transform," IEEE Trans. Audio Electroacoust. 18, 451-455 (1970)
    L. Rabiner, R. Schafer and C. Rader, "The chirp z-transform algorithm,"
    IEEE Trans. Audio Electroacoust. 17, 86-92 (1969)
    """

    x = np.moveaxis(np.asarray(x, dtype=complex), axis, -1)
    N = x.shape[-1]
    if M is None:
        M = N
    if w is None:
        w = np.exp(-2j*np.pi/M)
    if N < 1 or M < 1:
        raise ValueError("czt() needs N >= 1 input and M >= 1 output points")

    log_w, log_a = np.log(complex(w)), np.log(complex(a))
    n = np.arange(N, dtype=float)
    m = np.arange(M, dtype=float)
    L = _next_fast_len(N + M - 1)

    # x[n]*a**(-n)*w**(n**2/2), convolved with w**(-j**2/2), j = -(N-1) ... M-1:
    y = np.zeros(x.shape[:-1] + (L,), dtype=complex)
    y[..., :N] = x*np.exp(-log_a*n + log_w*n**2/2)
    j = np.arange(L, dtype=float)
    j[L - N + 1:] -= L  # negative lags at the end (cyclic convolution)
    v = np.exp(-log_w*j**2/2)
    v[M:L - N + 1] = 0.0

    X = ifft(fft(y, axis=-1)*fft(v), axis=-1)[..., :M]*np.exp(log_w*m**2/2)

    return np.moveaxis(X, -1, axis)


def _equidistant(values, name):
    """Return start and step of a linear, equidistant coordinate array."""

    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.ndim != 1:
        raise ValueError("{} must be a linear (1d) coordinate array".format(name))
    if len(values) == 1:
        return values[0], 0.0, 1
    step = (values[-1] - values[0])/(len(values) - 1)
    if not np.allclose(np.diff(values), step, rtol=1E-6, atol=1E-12*abs(step)):
        raise ValueError("{} must be equidistant".format(name))

    return values[0], step, len(values)


def _zoom_dft(f, x, k_out, sign, axis):
    """sum_l f[l]*exp(sign*1j*k_out[m]*x[l]) along axis, for equidistant x
    and k_out, via czt().
    """

    x0, dx, N = _equidistant(x, "x")
    k0, dk, M = _equidistant(k_out, "k_out")
    if np.shape(f)[axis] != N:
        raise ValueError("Coordinate array of length {} does not match the field "
                         "axis of length {}".format(N, np.shape(f)[axis]))

    # exp(s*i*(k0 + m*dk)*(x0 + l*dx)) = exp(s*i*k_m*x0)*exp(s*i*k0*dx*l)*exp(s*i*dk*dx*l*m):
    F = czt(f, M, w=np.exp(sign*1j*dk*dx), a=np.exp(-sign*1j*k0*dx), axis=axis)
    shape = [1]*np.ndim(F)
    shape[axis] = M
    k_m = k0 + dk*np.arange(M)

    return F*np.exp(sign*1j*k_m*x0).reshape(shape)


def zoom_FT(f, x, y, kx_out, ky_out):
    """FT() evaluated on an arbitrary equidistant frequency grid.

    Computes ::

        F[m, n] = sum_jl f[j, l]*exp(-i*(kx_out[n]*x[l] + ky_out[m]*y[j]))

    with two chirp-z transforms. The frequencies may be finer or coarser
    than 2*pi/L, shifted and of any number; the sum is periodic in k_x with
    the period 2*pi/dx. For centered x, y (see the module docstring) and the
    frequencies kx_out, ky_out of utils.frequencies(), F equals FT(f). Unlike
    FT(), the sum always uses the coordinates as given: no grid centering is
    needed and the phase of the result refers to the origin x = y = 0.

    Parameters
    ----------
    f : array
        Field of shape (Ny, Nx).
    x, y : arrays
        Linear, equidistant coordinates of f.
    kx_out, ky_out : arrays
        Linear, equidistant output wavenumbers (angular spatial frequencies).

    Returns
    -------
    F : array
        Spectrum of shape (len(ky_out), len(kx_out)). Multiply by dx*dy to
        approximate the Fourier integral.
    """

    F = _zoom_dft(f, x, kx_out, -1, axis=1)

    return _zoom_dft(F, y, ky_out, -1, axis=0)


def zoom_inv_FT(F, kx, ky, x_out, y_out):
    """inv_FT() evaluated on an arbitrary equidistant grid of positions.

    Computes the plane wave sum ::

        f[j, l] = 1/(Nx*Ny) sum_mn F[m, n]*exp(+i*(kx[n]*x_out[l] + ky[m]*y_out[j]))

    with two chirp-z transforms, i.e. the trigonometric interpolation of
    inv_FT(F) on any equidistant grid (the result has the period 2*pi/dk_x in
    x). For x_out, y_out equal to the centered grid that belongs to kx, ky,
    f equals inv_FT(F).

    Parameters
    ----------
    F : array
        Plane wave amplitudes of shape (Ny, Nx), e.g. from FT().
    kx, ky : arrays
        Linear, equidistant wavenumbers of F, e.g. utils.frequencies(x).
    x_out, y_out : arrays
        Linear, equidistant output coordinates.

    Returns
    -------
    f : array
        Field of shape (len(y_out), len(x_out)).
    """

    f = _zoom_dft(F, kx, x_out, +1, axis=1)
    f = _zoom_dft(f, ky, y_out, +1, axis=0)

    return f/np.prod(np.shape(F)[:2])
