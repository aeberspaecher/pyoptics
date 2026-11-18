#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Image restoration with measured or computed PSFs: colour channels,
field dependent blur and deconvolution.

A camera image is the scene blurred by the point spread function of the
lens, integrated over the pixels, filtered by the colour filters and
corrupted by photon and read noise. When the PSFs are known (measured with
a point source or a noise target, or computed from the lens design and
perhaps refined by phase retrieval), much of the blur can be undone. This
module holds the pieces used in
notebook 160 (image restoration and PSF deconvolution):

PSFs of a lens

- `cfa_sensitivity()`: model spectral sensitivities of the red, green
  and blue pixels of a phone sensor behind an IR cut filter.
- `field_for_image_point()`: field angles whose chief ray lands at a
  given point of the sensor.
- `psf_samples()`: pixel integrated intensity PSFs of a `raytrace.System`
  at points of the sensor and wavelengths, centred on the chief ray of a
  reference wavelength (so lateral colour is a shift of the PSF);
  `channel_psfs()` weights them with the channel sensitivities into
  polychromatic PSFs of the colour channels.

Forward models

- `SpaceVariantBlur`: blur by a PSF that changes over the image, as a sum
  ``y = sum_k h_k * (w_k x)`` of convolutions with kernels h_k of
  weighted copies of the image (Nagy and O'Leary 1998; with an SVD of the
  PSFs, the low rank "eigen-PSF" model of Flicker and Rigaut 2005 and
  Denis et al. 2015). It has the forward and the adjoint operator, and an
  optional margin by which the scene is larger than the image (the
  boundary problem of deconvolution). A shift invariant blur is the case
  of one kernel and constant weight.

Restoration

- `wiener()`: the Wiener (Tikhonov) filter for one PSF;
  `blended_deconvolve()` applies a shift invariant method with the PSF of
  each node of a grid and blends the results with the interpolation
  weights of the nodes.
- `richardson_lucy()`: the Richardson-Lucy iteration, the maximum
  likelihood estimate for Poisson data, for any `SpaceVariantBlur`, with a
  background (read noise, dark signal) and a mask of valid pixels (e.g.
  the pixels of one colour of a Bayer mosaic).
- `deconvolve_tv()`: least squares with total variation and, optionally,
  the cross-channel prior of Heide et al. (2013) that ties the edges of a
  colour channel to those of a reference channel, solved by the
  primal-dual algorithm of Condat (2013) and Vu (2013).
- `unrolled_hqs()`: a fixed number of half-quadratic splitting steps
  (data step by FFT, denoising step by soft thresholding of Haar wavelet
  coefficients) whose step parameters can be learned from training
  images, the simplest kind of an "unrolled network".

Test scenes and metrics

- `dead_leaves()`: the dead leaves texture of occluding discs with a
  power law of radii, scale invariant like natural images and used for
  camera texture tests (ISO 19567-2).
- `siemens_star()`, `psnr()`.

Images are arrays (n_y, n_x) with the row index along y; PSFs are odd
sized arrays (p, p) centred on the middle pixel and summing to 1. Lengths
in mm, wavelengths in mm unless the name says nm.

References
----------
N. Wiener, "Extrapolation, interpolation, and smoothing of stationary
time series", MIT Press (1949).

W. H. Richardson, "Bayesian-based iterative method of image restoration",
J. Opt. Soc. Am. 62, 55-59 (1972). L. B. Lucy, "An iterative technique
for the rectification of observed distributions", Astron. J. 79, 745-754
(1974).

L. I. Rudin, S. Osher and E. Fatemi, "Nonlinear total variation based
noise removal algorithms", Physica D 60, 259-268 (1992).

J. G. Nagy and D. P. O'Leary, "Restoring images degraded by spatially
variant blur", SIAM J. Sci. Comput. 19, 1063-1082 (1998).

D. Flicker and F. Rigaut, "Anisoplanatic deconvolution of adaptive optics
images", J. Opt. Soc. Am. A 22, 504-513 (2005).

L. Denis, E. Thiebaut, F. Soulez, J.-M. Becker and R. Mourya, "Fast
approximations of shift-variant blur", Int. J. Comput. Vis. 115, 253-278
(2015).

F. Heide, M. Rouf, M. B. Hullin, B. Labitzke, W. Heidrich and A. Kolb,
"High-quality computational imaging through simple lenses", ACM Trans.
Graph. 32, 149 (2013).

L. Condat, "A primal-dual splitting method for convex optimization
involving Lipschitzian, proximable and linear composite terms", J. Optim.
Theory Appl. 158, 460-479 (2013). B. C. Vu, "A splitting algorithm for
dual monotone inclusions involving cocoercive operators", Adv. Comput.
Math. 38, 667-681 (2013).

K. Zhang, L. Van Gool and R. Timofte, "Deep unfolding network for image
super-resolution", CVPR (2020).
"""

import numpy as np

from . import fft as _fft

__all__ = ["cfa_sensitivity", "chief_ray_point", "field_for_image_point", "psf_samples",
           "channel_psfs", "hat_weights", "SpaceVariantBlur", "wiener", "blended_deconvolve",
           "richardson_lucy", "grad", "grad_adjoint", "deconvolve_tv", "haar2", "inverse_haar2",
           "unrolled_hqs", "dead_leaves", "siemens_star", "psnr", "encircled_energy_radius"]


# ---------------------------------------------------------------------------
# PSFs of a lens

def cfa_sensitivity(wavelength_nm, ir_cut=650.0, uv_cut=410.0):
    """Model spectral sensitivities of the red, green and blue pixels.

    Gaussian dye transmissions (centres 600, 535 and 460 nm; the red one
    stays open to the long wavelengths) times the silicon quantum
    efficiency, falling towards the blue, and an IR cut filter (logistic
    edge at `ir_cut`, 10 nm wide) and its UV edge. The curves are a model
    of the general shape, not of a particular sensor.

    Parameters
    ----------
    wavelength_nm : array
    ir_cut, uv_cut : number, optional
        50 % points of the IR cut filter [nm].

    Returns
    -------
    array (3, ...)
        Relative sensitivities of R, G and B.
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    g = lambda mu, s: np.exp(-0.5*((lam - mu)/s)**2)
    R = np.where(lam < 600.0, g(600.0, 28.0), 1.0)
    G = g(535.0, 38.0)
    B = g(460.0, 32.0)
    qe = np.clip(0.35 + 0.65*(lam - 400.0)/200.0, 0.35, 1.0)
    filt = 1/(1 + np.exp((lam - ir_cut)/10.0))/(1 + np.exp(-(lam - uv_cut)/6.0))

    return np.array([R, G, B])*qe*filt


def chief_ray_point(system, field):
    """Image plane point (x, y) of the chief ray of a field.

    Parameters
    ----------
    system : raytrace.System
    field : number or pair
        See `raytrace.System.rays()`.

    Returns
    -------
    array (2,)
    """

    b = system.trace_pupil(field, 0.0, 0.0)

    return system._image_hits(b)[0][0, :2]


def field_for_image_point(system, x, y, tol=1e-9):
    """Field (pair of angles or object heights) whose chief ray at the
    wavelength of the system lands at the sensor point (x, y).

    Starts from the radial chief ray height of a rotationally symmetric
    system and refines with Newton steps on the traced chief ray.

    Parameters
    ----------
    system : raytrace.System
    x, y : number
        Sensor coordinates [mm].
    tol : number, optional
        Tolerance on the image point [mm].

    Returns
    -------
    array (2,)
        Field (x, y) for `raytrace.System.rays()`.
    """

    r = np.hypot(x, y)
    if r == 0:
        return np.zeros(2)
    a = np.linspace(0, 1.05*system.max_field, 41)
    h = np.array([chief_ray_point(system, (0.0, ai))[1] for ai in a])
    ok = np.isfinite(h)
    a, h = a[ok], h[ok]
    sign = 1.0 if h[-1] > 0 else -1.0     # -1: a real, inverted image of a finite object
    a0 = sign*np.interp(r, sign*h, a)
    f = np.array([a0*x/r, a0*y/r])
    eps = 1e-6*max(system.max_field, 1e-12)
    for _ in range(20):
        p = chief_ray_point(system, f)
        e = p - (x, y)
        if np.max(np.abs(e)) < tol:
            break
        J = np.column_stack([(chief_ray_point(system, f + d) - p)/eps
                             for d in (np.array([eps, 0]), np.array([0, eps]))])
        f = f - np.linalg.solve(J, e)

    return f


def _fourier_shift(img, sy, sx):
    """Shift a (band limited) image by (sy, sx) samples."""

    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]

    return np.real(np.fft.ifft2(np.fft.fft2(img)*np.exp(-2j*np.pi*(ky*sy + kx*sx))))


def psf_samples(system, points, wavelengths, pitch, n_pix=31, oversample=4, N=64,
                reference_wavelength=None):
    """Pixel integrated intensity PSFs at points of the sensor.

    For every sensor point the field of the chief ray at the reference
    wavelength is found (`field_for_image_point()`), and for every
    wavelength the PSF (`raytrace.System.psf()`) is computed on a grid
    `oversample` times finer than the pixels, shifted by the offset of the
    chief ray of that wavelength from the reference one (lateral colour),
    integrated over the pixel areas and normalized to a sum of 1. The
    pixel n_pix//2 of each PSF is centred on the reference chief ray.

    Parameters
    ----------
    system : raytrace.System
    points : array (n, 2)
        Sensor points (x, y) [mm].
    wavelengths : sequence
        [mm].
    pitch : number
        Pixel pitch [mm].
    n_pix : int, optional
        PSF size in pixels (odd).
    oversample : int, optional
        Sub-samples per pixel and axis.
    N : int, optional
        Pupil samples, see `raytrace.System.psf()`.
    reference_wavelength : number, optional
        Default: the wavelength of the system.

    Returns
    -------
    psfs : array (n, n_wavelengths, n_pix, n_pix)
    shifts : array (n, n_wavelengths, 2)
        Chief ray offsets (x, y) from the reference [mm].
    """

    ref = system.copy()
    if reference_wavelength is not None:
        ref.wavelength = reference_wavelength
    M = n_pix*oversample
    dx = pitch/oversample
    xf = (np.arange(M) - M//2)*dx
    # the centre of pixel n_pix//2 lies (oversample - 1)/2 fine samples above M//2:
    c0 = (n_pix//2)*oversample + (oversample - 1)/2 - M//2
    points = np.atleast_2d(points)
    out = np.zeros((len(points), len(wavelengths), n_pix, n_pix))
    shifts = np.zeros((len(points), len(wavelengths), 2))
    for i, (px, py) in enumerate(points):
        field = field_for_image_point(ref, px, py)
        p0 = chief_ray_point(ref, field)
        for j, wl in enumerate(wavelengths):
            s = system.copy()
            s.wavelength = wl
            shifts[i, j] = chief_ray_point(s, field) - p0
            I = np.abs(s.psf(xf, xf, field=field, N=N, normalization=None))**2
            I = _fourier_shift(I, c0 + shifts[i, j, 1]/dx, c0 + shifts[i, j, 0]/dx)
            I = np.maximum(I, 0.0).reshape(n_pix, oversample, n_pix, oversample).sum(axis=(1, 3))
            out[i, j] = I/I.sum()

    return out, shifts


def channel_psfs(psfs, wavelength_nm, sensitivity=None, spectrum=None):
    """Polychromatic PSFs of the colour channels: the monochromatic PSFs
    weighted with sensitivity times spectrum and normalized.

    Parameters
    ----------
    psfs : array (..., n_wavelengths, p, p)
    wavelength_nm : array (n_wavelengths,)
    sensitivity : array (n_channels, n_wavelengths), optional
        Default: `cfa_sensitivity()`.
    spectrum : array (n_wavelengths,), optional
        Spectrum of the light (default: flat).

    Returns
    -------
    array (..., n_channels, p, p)
    """

    S = cfa_sensitivity(wavelength_nm) if sensitivity is None else np.asarray(sensitivity)
    w = S*(1.0 if spectrum is None else np.asarray(spectrum))
    w = w/w.sum(axis=1, keepdims=True)

    return np.einsum("cw,...wij->...cij", w, psfs)


def encircled_energy_radius(psf, fraction=0.8, pitch=1.0):
    """Radius of the circle around the centroid that holds `fraction` of
    the PSF.

    Parameters
    ----------
    psf : array (p, p)
    fraction : number, optional
    pitch : number, optional
        Pixel size (units of the result).

    Returns
    -------
    number
    """

    yy, xx = np.indices(psf.shape)
    s = psf.sum()
    cy, cx = (psf*yy).sum()/s, (psf*xx).sum()/s
    r = np.hypot(yy - cy, xx - cx).ravel()
    o = np.argsort(r)
    c = np.cumsum(psf.ravel()[o])/s

    return r[o][np.searchsorted(c, fraction)]*pitch


# ---------------------------------------------------------------------------
# forward model

def hat_weights(nodes, n):
    """Piecewise linear interpolation weights (hat functions) of nodes
    along an axis of n pixels; constant beyond the first and last node.

    Parameters
    ----------
    nodes : array (m,)
        Increasing pixel coordinates of the nodes.
    n : int

    Returns
    -------
    array (m, n)
        Weights that sum to 1 at every pixel.
    """

    nodes = np.asarray(nodes, dtype=float)
    t = np.arange(n, dtype=float)
    W = np.zeros((len(nodes), n))
    if len(nodes) == 1:
        W[0] = 1.0
        return W
    for i in range(len(nodes)):
        e = np.zeros(len(nodes))
        e[i] = 1.0
        W[i] = np.interp(t, nodes, e)

    return W


class SpaceVariantBlur:
    """Space variant blur ``y = crop(sum_k h_k * (w_k x))``.

    Parameters
    ----------
    kernels : array (K, p, p)
        Kernels h_k, centred (p odd).
    weights : array (K, n_y + 2 margin, n_x + 2 margin)
        Weight maps w_k on the scene grid.
    margin : int, optional
        The scene extends by margin pixels beyond the image on every side;
        the forward operator crops it (the image is a window of a larger
        scene, without assumptions on what lies outside).
    """

    def __init__(self, kernels, weights, margin=0):
        self.kernels = np.asarray(kernels, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.margin = int(margin)
        self.scene_shape = self.weights.shape[1:]
        m = self.margin
        self.shape = (self.scene_shape[0] - 2*m, self.scene_shape[1] - 2*m)
        p = self.kernels.shape[-1]
        self._full = (self.scene_shape[0] + p - 1, self.scene_shape[1] + p - 1)
        self._full = tuple(_fft._next_fast_len(n) for n in self._full)
        self._H = _fft.rfft2(self.kernels, s=self._full)
        self._Hflip = _fft.rfft2(self.kernels[:, ::-1, ::-1], s=self._full)
        self._norm = None

    @classmethod
    def invariant(cls, psf, shape, margin=0):
        """Shift invariant blur by one PSF on images of `shape`.

        Parameters
        ----------
        psf : array (p, p)
        shape : (n_y, n_x)
            Image shape.
        margin : int, optional

        Returns
        -------
        SpaceVariantBlur
        """

        s = (shape[0] + 2*margin, shape[1] + 2*margin)

        return cls(np.asarray(psf)[None], np.ones((1,) + s), margin)

    @classmethod
    def from_grid(cls, psfs, node_y, node_x, shape, rank=None, margin=0):
        """Blur from PSFs given on a grid of nodes, interpolated
        bilinearly between them.

        With rank=None the kernels are the node PSFs and the weights their
        hat functions (interpolation of the PSFs, Nagy and O'Leary). With a
        rank K the node PSFs are replaced by their best rank K
        approximation (SVD): K "eigen-PSFs" as kernels, and the weights are
        the interpolated expansion coefficients (Flicker and Rigaut).

        Parameters
        ----------
        psfs : array (n_node_y, n_node_x, p, p)
        node_y, node_x : arrays
            Pixel coordinates of the nodes in the image (not the scene).
        shape : (n_y, n_x)
            Image shape.
        rank : int, optional
        margin : int, optional

        Returns
        -------
        SpaceVariantBlur
        """

        psfs = np.asarray(psfs, dtype=float)
        ny, nx, p, _ = psfs.shape
        Wy = hat_weights(np.asarray(node_y) + margin, shape[0] + 2*margin)
        Wx = hat_weights(np.asarray(node_x) + margin, shape[1] + 2*margin)
        hats = np.einsum("iy,jx->ijyx", Wy, Wx).reshape(ny*nx, len(Wy[0]), len(Wx[0]))
        P = psfs.reshape(ny*nx, p*p)
        if rank is None:
            return cls(psfs.reshape(ny*nx, p, p), hats, margin)
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        K = min(rank, len(S))
        kernels = (S[:K, None]*Vt[:K]).reshape(K, p, p)
        weights = np.einsum("nk,nyx->kyx", U[:, :K], hats)
        op = cls(kernels, weights, margin)
        op.singular_values = S

        return op

    def _crop(self, a):
        m = self.margin
        return a[..., m:m + self.shape[0], m:m + self.shape[1]] if m else a

    def _conv(self, a, adjoint=False):
        """Linear convolution of a (K, scene) stack with the kernels (or,
        for the adjoint, with the flipped kernels), cropped to the scene."""

        c = self.kernels.shape[-1]//2
        H = self._Hflip if adjoint else self._H
        full = _fft.irfft2(_fft.rfft2(a, s=self._full)*H, s=self._full)

        return full[..., c:c + self.scene_shape[0], c:c + self.scene_shape[1]]

    def forward(self, x):
        """Blurred image of a scene x (scene shape).

        Parameters
        ----------
        x : array

        Returns
        -------
        array
            Image (image shape).
        """

        return self._crop(self._conv(self.weights*x).sum(axis=0))

    def adjoint(self, y):
        """Adjoint operator, from image to scene.

        Parameters
        ----------
        y : array

        Returns
        -------
        array
        """

        m = self.margin
        Y = np.zeros(self.scene_shape)
        Y[m:m + self.shape[0], m:m + self.shape[1]] = y

        return (self.weights*self._conv(np.broadcast_to(Y, self.kernels.shape[:1] + Y.shape),
                                         adjoint=True)).sum(axis=0)

    def norm(self, n_iter=30, rng=0):
        """Operator norm (largest singular value) by power iteration.

        Parameters
        ----------
        n_iter : int, optional
        rng : int or Generator, optional

        Returns
        -------
        number
        """

        if self._norm is None:
            x = np.random.default_rng(rng).random(self.scene_shape)
            for _ in range(n_iter):
                x = self.adjoint(self.forward(x))
                s = np.linalg.norm(x)
                x /= s
            self._norm = np.sqrt(s)

        return self._norm


# ---------------------------------------------------------------------------
# restoration

def _otf(psf, shape):
    """OTF on an rfft grid of `shape` of a centred PSF."""

    p = psf.shape[0]
    big = np.zeros(shape)
    big[:p, :p] = psf
    big = np.roll(big, (-(p//2), -(p//2)), axis=(0, 1))

    return np.fft.rfft2(big)


def wiener(image, psf, nsr, pad=None):
    """Wiener deconvolution with one PSF.

    ``X = conj(H) Y / (|H|**2 + NSR)``. The image is extended by mirroring
    by `pad` pixels on every side (default: the PSF size), which avoids the
    wrap around of the periodic FFT at the borders.

    Parameters
    ----------
    image : array (n_y, n_x)
    psf : array (p, p)
    nsr : number or callable
        Noise to signal power ratio; a callable gets the radial spatial
        frequency (cycles per pixel) on the rfft grid and returns it.
    pad : int, optional

    Returns
    -------
    array (n_y, n_x)
    """

    p = psf.shape[0]
    pad = p if pad is None else pad
    big = np.pad(image, pad, mode="symmetric")
    H = _otf(psf, big.shape)
    if callable(nsr):
        fy = np.fft.fftfreq(big.shape[0])[:, None]
        fx = np.fft.rfftfreq(big.shape[1])[None, :]
        nsr = nsr(np.hypot(fx, fy))
    X = np.conj(H)*np.fft.rfft2(big)/(np.abs(H)**2 + nsr)
    out = np.fft.irfft2(X, s=big.shape)

    return out[pad:pad + image.shape[0], pad:pad + image.shape[1]]


def blended_deconvolve(image, psfs, node_y, node_x, solver, margin=None):
    """Space variant restoration from a shift invariant method: every node
    PSF restores a window around its node, and the results are blended
    with the bilinear interpolation weights of the nodes (which sum to 1).

    Parameters
    ----------
    image : array (n_y, n_x)
    psfs : array (n_node_y, n_node_x, p, p)
    node_y, node_x : arrays
        Pixel coordinates of the nodes.
    solver : callable
        solver(window, psf) -> restored window.
    margin : int, optional
        Extra pixels around the support of each node's weight (default
        2 p).

    Returns
    -------
    array (n_y, n_x)
    """

    ny, nx = image.shape
    p = psfs.shape[-1]
    margin = 2*p if margin is None else margin
    Wy, Wx = hat_weights(node_y, ny), hat_weights(node_x, nx)
    out = np.zeros(image.shape)
    for i in range(len(node_y)):
        ys = np.nonzero(Wy[i])[0]
        y0, y1 = max(ys[0] - margin, 0), min(ys[-1] + 1 + margin, ny)
        for j in range(len(node_x)):
            xs = np.nonzero(Wx[j])[0]
            x0, x1 = max(xs[0] - margin, 0), min(xs[-1] + 1 + margin, nx)
            r = solver(image[y0:y1, x0:x1], psfs[i, j])
            out[y0:y1, x0:x1] += np.outer(Wy[i, y0:y1], Wx[j, x0:x1])*r

    return out


def richardson_lucy(image, op, n_iter=30, background=0.0, mask=None, x0=None, callback=None,
                    eps=1e-12):
    """Richardson-Lucy deconvolution.

    The multiplicative update ::

        x <- x A^T(m y/(A x + b)) / A^T m

    increases the Poisson likelihood of the data y in every step; b is a
    known background and m a mask of the valid pixels. It keeps x >= 0 and,
    without a mask, the total flux. Without regularization it amplifies
    the noise after enough iterations: the number of iterations is the
    regularization parameter.

    Parameters
    ----------
    image : array
        Data y (counts, or anything proportional).
    op : SpaceVariantBlur
    n_iter : int, optional
    background : number or array, optional
    mask : array, optional
        1 for measured pixels, 0 for missing ones.
    x0 : array, optional
        Start (scene shape); default: the mean of the data.
    callback : callable, optional
        callback(i, x) after every iteration.
    eps : number, optional

    Returns
    -------
    array
        Restored scene (scene shape).
    """

    y = np.asarray(image, dtype=float)
    m = np.ones_like(y) if mask is None else np.asarray(mask, dtype=float)
    norm = np.maximum(op.adjoint(m), eps)
    mean = np.sum(m*np.maximum(y - background, 0))/np.sum(m)
    x = np.full(op.scene_shape, max(mean, eps)) if x0 is None else np.array(x0, dtype=float)
    floor = max(eps, 1e-6*np.max(y))
    for i in range(n_iter):
        ratio = m*y/np.maximum(op.forward(x) + background, floor)
        x = np.maximum(x*op.adjoint(ratio)/norm, 0.0)
        if callback is not None:
            callback(i, x)

    return x


def grad(x):
    """Forward differences (periodic) along y and x, shape (2, ...).

    Parameters
    ----------
    x : array

    Returns
    -------
    array
    """

    return np.array([np.roll(x, -1, axis=-2) - x, np.roll(x, -1, axis=-1) - x])


def grad_adjoint(g):
    """Adjoint of `grad()` (minus the divergence).

    Parameters
    ----------
    g : array (2, ...)

    Returns
    -------
    array
    """

    return (np.roll(g[0], 1, axis=-2) - g[0]) + (np.roll(g[1], 1, axis=-1) - g[1])


def deconvolve_tv(image, op, lam, n_iter=200, reference=None, cross=0.0, mask=None, x0=None,
                  nonnegative=True, callback=None):
    """Regularized deconvolution with total variation and the
    cross-channel prior.

    Minimizes ::

        1/2 |m (A x - y)|**2 + lam |grad x|_1
                             + cross |r grad x - x grad r|_1

    over x >= 0, with the (anisotropic) total variation and the
    cross-channel term of Heide et al. (2013): with r a reference channel
    (e.g. the restored green), the term vanishes where the edges of x and r
    are at the same places and in proportion to the intensities, i.e.
    where the colour (x/r) is constant. The solver is the primal-dual
    algorithm of Condat and Vu with steps from the operator norms.

    Parameters
    ----------
    image : array
        Data y.
    op : SpaceVariantBlur
    lam : number
        Weight of the total variation.
    n_iter : int, optional
    reference : array, optional
        Reference channel r (scene shape) for the cross-channel term.
    cross : number, optional
        Weight of the cross-channel term.
    mask : array, optional
        1 for measured pixels.
    x0 : array, optional
        Start (default: the adjoint image).
    nonnegative : bool, optional
    callback : callable, optional
        callback(i, x).

    Returns
    -------
    array
        Restored scene.
    """

    y = np.asarray(image, dtype=float)
    m = np.ones_like(y) if mask is None else np.asarray(mask, dtype=float)
    terms = [(grad, grad_adjoint, lam, np.sqrt(8.0))]
    if reference is not None and cross > 0:
        r = np.asarray(reference, dtype=float)
        gr = grad(r)
        K = lambda x: r*grad(x) - gr*x
        Kt = lambda q: grad_adjoint(r*q) - np.sum(gr*q, axis=0)
        terms.append((K, Kt, cross, np.sqrt(8.0)*np.max(np.abs(r)) + np.max(np.abs(gr))*np.sqrt(2.0)))
    L = op.norm()**2
    Knorm2 = sum(t[3]**2 for t in terms)
    tau = 1.0/L
    sigma = 0.5/(tau*Knorm2)   # 1/tau - sigma |K|^2 >= L/2
    x = op.adjoint(m*y) if x0 is None else np.array(x0, dtype=float)
    duals = [np.zeros((2,) + op.scene_shape) for _ in terms]
    for i in range(n_iter):
        g = op.adjoint(m*(op.forward(x) - y))
        for (K, Kt, w, _), q in zip(terms, duals):
            g = g + Kt(q)
        xn = x - tau*g
        if nonnegative:
            xn = np.maximum(xn, 0.0)
        xb = 2*xn - x
        for k, (K, Kt, w, _) in enumerate(terms):
            duals[k] = np.clip(duals[k] + sigma*K(xb), -w, w)
        x = xn
        if callback is not None:
            callback(i, x)

    return x


def haar2(x, levels):
    """Orthonormal 2D Haar wavelet transform (in place layout).

    Parameters
    ----------
    x : array (n_y, n_x)
        Sizes divisible by 2**levels.
    levels : int

    Returns
    -------
    array
        Coefficients, the coarse approximation in the top left corner.
    """

    c = np.array(x, dtype=float)
    ny, nx = c.shape
    for _ in range(levels):
        a = c[:ny, :nx]
        s = (a[0::2] + a[1::2])/np.sqrt(2)
        d = (a[0::2] - a[1::2])/np.sqrt(2)
        a = np.vstack([s, d])
        s = (a[:, 0::2] + a[:, 1::2])/np.sqrt(2)
        d = (a[:, 0::2] - a[:, 1::2])/np.sqrt(2)
        c[:ny, :nx] = np.hstack([s, d])
        ny, nx = ny//2, nx//2

    return c


def inverse_haar2(c, levels):
    """Inverse of `haar2()`.

    Parameters
    ----------
    c : array
    levels : int

    Returns
    -------
    array
    """

    x = np.array(c, dtype=float)
    NY, NX = x.shape
    for lev in reversed(range(levels)):
        ny, nx = NY >> lev, NX >> lev
        a = x[:ny, :nx]
        s, d = a[:, :nx//2], a[:, nx//2:]
        b = np.empty_like(a)
        b[:, 0::2] = (s + d)/np.sqrt(2)
        b[:, 1::2] = (s - d)/np.sqrt(2)
        s, d = b[:ny//2], b[ny//2:]
        a = np.empty_like(b)
        a[0::2] = (s + d)/np.sqrt(2)
        a[1::2] = (s - d)/np.sqrt(2)
        x[:ny, :nx] = a

    return x


def _haar_denoise(x, t, levels):
    """Soft thresholding of the Haar detail coefficients, averaged over the
    four half pixel shifts (cycle spinning against blocking)."""

    out = np.zeros_like(x)
    ny, nx = x.shape[0] >> levels, x.shape[1] >> levels
    for sy in (0, 1):
        for sx in (0, 1):
            c = haar2(np.roll(x, (sy, sx), axis=(0, 1)), levels)
            coarse = c[:ny, :nx].copy()
            c = np.sign(c)*np.maximum(np.abs(c) - t, 0.0)
            c[:ny, :nx] = coarse
            out += np.roll(inverse_haar2(c, levels), (-sy, -sx), axis=(0, 1))

    return out/4


def unrolled_hqs(image, psf, params, levels=3, pad=None):
    """Half-quadratic splitting with a fixed number of steps.

    Step k solves the data problem ``min |h * x - y|**2 + rho_k |x - z|**2``
    in closed form by FFT and then denoises, z = soft thresholding of the
    Haar detail coefficients of x with the threshold t_k. The parameters
    (rho_k, t_k) are the "weights" of an unrolled network: they can be
    learned by minimizing the error on training images.

    Parameters
    ----------
    image : array (n_y, n_x)
    psf : array (p, p)
    params : array (K, 2)
        log10 rho_k and log10 t_k of the K steps.
    levels : int, optional
        Haar levels of the denoiser.
    pad : int, optional
        Mirror padding (default: the PSF size, rounded up so that the
        padded size is divisible by 2**levels).

    Returns
    -------
    array (n_y, n_x)
    """

    p = psf.shape[0]
    pad = p if pad is None else pad
    q = 2**levels
    py = pad + (-(image.shape[0] + 2*pad)) % q
    px = pad + (-(image.shape[1] + 2*pad)) % q
    big = np.pad(image, ((pad, py), (pad, px)), mode="symmetric")
    H = _otf(psf, big.shape)
    HY = np.conj(H)*np.fft.rfft2(big)
    H2 = np.abs(H)**2
    z = big.copy()
    for lr, lt in np.asarray(params):
        rho = 10.0**lr
        x = np.fft.irfft2((HY + rho*np.fft.rfft2(z))/(H2 + rho), s=big.shape)
        z = _haar_denoise(x, 10.0**lt, levels)

    return z[pad:pad + image.shape[0], pad:pad + image.shape[1]]


# ---------------------------------------------------------------------------
# test scenes and metrics

def dead_leaves(shape, r_min=1.0, r_max=None, colours=None, n_max=200000, oversample=2, rng=None):
    """Dead leaves texture: discs with radii of density ~ 1/r**3 between
    r_min and r_max and random colours, falling on top of each other until
    the image is covered (rendered front to back).

    Parameters
    ----------
    shape : (n_y, n_x)
    r_min, r_max : number, optional
        Radii in pixels (default r_max: a quarter of the smaller side).
    colours : callable, optional
        colours(rng, n) -> array (n, c) of disc colours; default: grey
        levels uniform in [0.1, 0.9].
    n_max : int, optional
        Largest number of discs.
    oversample : int, optional
        The texture is rendered on a finer grid and averaged over the
        pixels (pixel integrated, anti-aliased edges).
    rng : int or Generator, optional

    Returns
    -------
    array (c, n_y, n_x) or (n_y, n_x) for grey levels
    """

    rng = np.random.default_rng(rng)
    ny, nx = shape[0]*oversample, shape[1]*oversample
    r_max = min(shape)/4 if r_max is None else r_max
    a, b = r_min*oversample, r_max*oversample
    colours = (lambda g, n: g.uniform(0.1, 0.9, (n, 1))) if colours is None else colours
    cols = colours(rng, n_max)
    C = cols.shape[1]
    img = np.zeros((C, ny, nx))
    free = np.ones((ny, nx), dtype=bool)
    left = ny*nx
    for k in range(n_max):
        # inverse transform sampling of p(r) ~ r**-3 on [a, b]
        u = rng.random()
        r = 1/np.sqrt(1/a**2 - u*(1/a**2 - 1/b**2))
        cy, cx = rng.uniform(-r, ny + r), rng.uniform(-r, nx + r)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 2, ny)
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 2, nx)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        hit = ((yy - cy)**2 + (xx - cx)**2 <= r*r) & free[y0:y1, x0:x1]
        n = hit.sum()
        if n == 0:
            continue
        img[:, y0:y1, x0:x1][:, hit] = cols[k][:, None]
        free[y0:y1, x0:x1][hit] = False
        left -= n
        if left == 0:
            break
    img = img.reshape(C, shape[0], oversample, shape[1], oversample).mean(axis=(2, 4))

    return img[0] if C == 1 else img


def siemens_star(n, spokes=36, oversample=4):
    """Siemens star of `spokes` black and white sectors filling an n x n
    image, pixel integrated; values 0.1 and 0.9.

    Parameters
    ----------
    n : int
    spokes : int, optional
    oversample : int, optional

    Returns
    -------
    array (n, n)
    """

    t = (np.arange(n*oversample) + 0.5)/oversample - n/2
    X, Y = np.meshgrid(t, t)
    s = 0.1 + 0.8*(np.sin(spokes/2*np.arctan2(Y, X)) > 0)

    return s.reshape(n, oversample, n, oversample).mean(axis=(1, 3))


def psnr(x, ref, peak=1.0, border=0):
    """Peak signal to noise ratio in dB.

    Parameters
    ----------
    x, ref : arrays
    peak : number, optional
    border : int, optional
        Pixels left out at the borders of the last two axes.

    Returns
    -------
    number
    """

    if border:
        x = x[..., border:-border, border:-border]
        ref = ref[..., border:-border, border:-border]

    return 10*np.log10(peak**2/np.mean((np.asarray(x) - ref)**2))
