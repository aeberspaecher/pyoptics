#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Lensless imaging: pinhole cameras, coded apertures and diffuser cameras.

A camera without a lens trades the focusing element for computation. Three
designs are covered:

- The pinhole camera (camera obscura). A small pinhole of diameter d at the
  distance v in front of the screen forms an image whose blur is the larger
  of the geometric shadow of the pinhole (about d) and its diffraction spot
  (about 2.44*wavelength*v/d). The best compromise is near
  d = 1.9*sqrt(wavelength*v) (Rayleigh 1891): `rayleigh_pinhole_diameter()`,
  `pinhole_on_axis_intensity()` (exact on-axis Fresnel intensity behind a
  circular hole), `pinhole_blur()` (the geometric and diffraction blur added
  in quadrature, Petzval's estimate).
- The coded aperture. A mask with many holes casts overlapping shadows of
  every source point; correlating the recorded picture with a decoding
  array recovers the scene. For the modified uniformly redundant array
  (MURA, Gottesman and Fenimore 1989) the periodic correlation of mask and
  decoder is a delta, so the decoding is exact without noise: `mura()`,
  `mura_decoder()`, `coded_aperture_image()`, `coded_aperture_decode()`.
  The signal-to-noise ratio for photon noise is `coded_aperture_snr()`
  (Fenimore 1978), against `pinhole_snr()` for a single hole.
- The diffuser camera (DiffuserCam, Antipa et al. 2018). A smooth random
  phase plate a few millimetres in front of a bare sensor turns every point
  source into a pattern of caustics. The pattern shifts with the lateral
  position of the source and scales with its distance, so the picture is a
  convolution of the scene with one point spread function (PSF) per depth.
  `point_source_psf()` computes the PSF by wave propagation,
  `psf_magnification()` and `scale_psf()` move it to another depth,
  `caustic_distance()` estimates where the caustics are sharpest. The scene
  is recovered by deconvolution: `wiener_deconvolve()` or
  `admm_tv_deconvolve()`, the ADMM solver of Antipa et al. with total
  variation regularization, positivity and a sensor smaller than the
  blurred scene (`convolve_crop()` is the forward model).

All lengths are in the same unit; the arrays are 2d with the axis order
(y, x) of `pyoptics.utils.grid2d()`. PSFs are centred: the point at the
array centre (index N//2) is the origin.

References
----------
Lord Rayleigh, "On pin-hole photography", Phil. Mag. 31, 87-99 (1891).

M. Young, "Pinhole optics", Appl. Opt. 10, 2763-2767 (1971).

K. D. Mielenz, "On the diffraction limit for lensless imaging", J. Res.
Natl. Inst. Stand. Technol. 104, 479-485 (1999).

E. E. Fenimore and T. M. Cannon, "Coded aperture imaging with uniformly
redundant arrays", Appl. Opt. 17, 337-347 (1978).

E. E. Fenimore, "Coded aperture imaging: predicted performance of uniformly
redundant arrays", Appl. Opt. 17, 3562-3570 (1978).

S. R. Gottesman and E. E. Fenimore, "New family of binary arrays for coded
aperture imaging", Appl. Opt. 28, 4344-4352 (1989).

N. Antipa, G. Kuo, R. Heckel, B. Mildenhall, E. Bostan, R. Ng and L. Waller,
"DiffuserCam: lensless single-exposure 3D imaging", Optica 5, 1-9 (2018).

S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed
optimization and statistical learning via the alternating direction method
of multipliers", Found. Trends Mach. Learn. 3, 1-122 (2011).

M. V. Berry, "Disruption of images: the caustic-touching theorem", J. Opt.
Soc. Am. A 4, 561-569 (1987).
"""

import numpy as np

from .propagators import rayleigh_sommerfeld_I_TF
from .utils import ensure_meshgrid


# ---------------------------------------------------------------------------
# pinhole camera
# ---------------------------------------------------------------------------

def _effective_distance(image_distance, object_distance):
    """1/z = 1/v + 1/u: the distance that sets the Fresnel number of a hole
    lit by a point source at u and observed at v."""

    v = np.asarray(image_distance, dtype=float)
    u = np.asarray(object_distance, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0/(1.0/v + np.where(np.isinf(u), 0.0, 1.0/u))


def rayleigh_pinhole_diameter(wavelength, image_distance, object_distance=np.inf, factor=1.9):
    """Optimum pinhole diameter d = factor*sqrt(wavelength*z).

    With z = image_distance for a distant object, and 1/z = 1/u + 1/v for an
    object at the distance u. Rayleigh's value factor = 1.9 makes the radius
    r = sqrt(0.9*wavelength*z): the path difference between the centre and
    the rim of the hole is 0.45 wavelengths, a little less than the first
    Fresnel zone (factor 2), which maximizes the on-axis intensity. Other
    criteria give factors between about 1.5 and 2 (see `pinhole_blur()`).

    Parameters
    ----------
    wavelength : number
    image_distance : number
        Distance v from the pinhole to the screen.
    object_distance : number, optional
        Distance u from the object to the pinhole (default: infinity).
    factor : number, optional
        Defaults to Rayleigh's 1.9.

    Returns
    -------
    d : number
        Pinhole diameter.
    """

    return factor*np.sqrt(wavelength*_effective_distance(image_distance, object_distance))


def pinhole_on_axis_intensity(diameter, wavelength, image_distance, object_distance=np.inf):
    """On-axis intensity behind a circular hole, relative to free space.

    For a point source at the distance u (a plane wave for u = infinity)
    the Fresnel integral over a hole of radius a gives on the axis at the
    distance v exactly ::

        I/I_free = 4*sin(pi*N/2)**2,   N = a**2/(wavelength*z),

    with 1/z = 1/u + 1/v and N the number of Fresnel zones in the hole. The
    intensity is largest (four times the unobstructed value) when the hole
    is the first Fresnel zone, d = 2*sqrt(wavelength*z), and vanishes for two
    zones.

    Parameters
    ----------
    diameter : number or array
    wavelength : number
    image_distance : number
    object_distance : number, optional

    Returns
    -------
    I : number or array
        On-axis intensity in units of the intensity without the screen.
    """

    a = 0.5*np.asarray(diameter, dtype=float)
    N = a**2/(wavelength*_effective_distance(image_distance, object_distance))
    return 4.0*np.sin(0.5*np.pi*N)**2


def pinhole_blur(diameter, wavelength, image_distance, object_distance=np.inf):
    """Blur diameter of a pinhole camera: geometric and diffraction parts.

    The geometric shadow of the hole has the diameter d*(u + v)/u; the
    diffraction spot (Airy disk) has the diameter 2.44*wavelength*v/d. Added
    in quadrature (Petzval's estimate), the blur is smallest for
    d = sqrt(2.44*wavelength*z) = 1.56*sqrt(wavelength*z), with
    1/z = 1/u + 1/v.

    Parameters
    ----------
    diameter : number or array
    wavelength : number
    image_distance : number
    object_distance : number, optional

    Returns
    -------
    blur, geometric, diffraction : numbers or arrays
        Total blur diameter and its two parts, on the screen.
    """

    d = np.asarray(diameter, dtype=float)
    v = float(image_distance)
    u = float(object_distance)
    geometric = d*(1.0 + (0.0 if np.isinf(u) else v/u))
    diffraction = 2.44*wavelength*v/d
    return np.hypot(geometric, diffraction), geometric, diffraction


# ---------------------------------------------------------------------------
# coded apertures
# ---------------------------------------------------------------------------

def _is_prime(p):
    if p < 2:
        return False
    return all(p % q for q in range(2, int(np.sqrt(p)) + 1))


def mura_orders(p_max):
    """The orders p <= p_max for which square MURAs exist: primes p = 4m + 1.

    Parameters
    ----------
    p_max : int

    Returns
    -------
    orders : list of int
    """

    return [p for p in range(5, int(p_max) + 1) if p % 4 == 1 and _is_prime(p)]


def mura(p):
    """Square modified uniformly redundant array (MURA) of prime order p = 4m + 1.

    Following Gottesman and Fenimore (1989), with C_i = +1 if i is a
    quadratic residue modulo p and -1 otherwise ::

        A[0, j] = 0,   A[i, 0] = 1 (i != 0),
        A[i, j] = 1 if C_i*C_j = +1 else 0.

    (p**2 - 1)/2 of the p**2 elements are open, almost exactly half.

    Parameters
    ----------
    p : int
        A prime with p % 4 == 1 (5, 13, 17, 29, 37, 41, 53, 61, ...).

    Returns
    -------
    A : array of int, shape (p, p)
        1 for open, 0 for opaque elements. Element [i, j] is row i (y), column
        j (x).
    """

    p = int(p)
    if not (_is_prime(p) and p % 4 == 1):
        raise ValueError("A MURA needs a prime order p = 4m + 1, e.g. {}.".format(mura_orders(70)))
    residue = np.zeros(p, dtype=bool)
    residue[(np.arange(1, p)**2) % p] = True
    C = np.where(residue, 1, -1)
    A = np.zeros((p, p), dtype=int)
    A[1:, 0] = 1
    A[1:, 1:] = np.outer(C[1:], C[1:]) == 1
    return A


def mura_decoder(A):
    """Decoding array G of a MURA: +1 where A is open, -1 where it is closed,
    and G[0, 0] = +1.

    The periodic correlation of A and G is (p**2 - 1)/2 at zero shift and
    exactly zero elsewhere, and G sums to 1.

    Parameters
    ----------
    A : array
        A MURA from `mura()`.

    Returns
    -------
    G : array of int
    """

    G = np.where(np.asarray(A) == 1, 1, -1)
    G[0, 0] = 1
    return G


def coded_aperture_image(scene, mask):
    """Detector picture of a coded aperture camera in the periodic model.

    Every scene element (a source direction) casts the shadow of the mask,
    shifted by its position: D[i] = sum_k scene[k]*mask[i - k], indices
    modulo the mask size. This is exact for a mask mosaicked 2 x 2 (or a
    cyclic mask of size 2p - 1) in front of a detector of p x p elements,
    each the size of one mask element, for sources within the fully coded
    field of view (geometric shadows, no diffraction).

    Parameters
    ----------
    scene : array, shape (p, p)
        Photons per element through a single open element, i.e. what a
        pinhole camera of one element would record.
    mask : array, shape (p, p)

    Returns
    -------
    D : array, shape (p, p)
        Expected detector counts.
    """

    return np.real(np.fft.ifft2(np.fft.fft2(scene)*np.fft.fft2(mask)))


def coded_aperture_decode(picture, decoder, n_open=None):
    """Decode a coded aperture picture by periodic correlation with G.

    O[j] = sum_i picture[i]*G[i - j]/n_open. For a MURA and the picture of
    `coded_aperture_image()` this returns the scene exactly, plus the mean
    background per detector element (the decoder sums to 1, so a constant
    background B per element becomes B/n_open in every scene element).

    Parameters
    ----------
    picture : array, shape (p, p)
    decoder : array, shape (p, p)
        The decoding array, e.g. `mura_decoder()`.
    n_open : number, optional
        Normalization, the number of open mask elements. Defaults to
        (p**2 - 1)/2, the value of a MURA.

    Returns
    -------
    scene : array, shape (p, p)
    """

    p_y, p_x = np.shape(decoder)
    if n_open is None:
        n_open = 0.5*(p_y*p_x - 1)
    corr = np.real(np.fft.ifft2(np.fft.fft2(picture)*np.conj(np.fft.fft2(decoder))))
    return corr/n_open


def coded_aperture_snr(signal, total, n_open, background=0.0, n_pixels=None):
    """Signal-to-noise ratio of one decoded scene element (photon noise).

    Fenimore (1978): with psi_j photons per open element from source j, a
    total of psi_T = sum_k psi_k from all sources and a background of B per
    detector element, the correlation with a +-1 decoder sums the variance of
    every detector element, so ::

        SNR_j = n_open*psi_j/sqrt(n_open*psi_T + n_pixels*B).

    A bright source among dark ones gains sqrt(n_open) over a single hole;
    in a crowded or extended scene the noise of all sources lands in every
    element.

    Parameters
    ----------
    signal : number or array
        psi_j, photons per open element from the scene element.
    total : number or array
        psi_T, the sum of psi over all scene elements.
    n_open : number
        Number of open mask elements.
    background : number or array, optional
        Background photons per detector element.
    n_pixels : number, optional
        Detector elements; defaults to 2*n_open + 1 (a MURA).

    Returns
    -------
    snr : number or array
    """

    if n_pixels is None:
        n_pixels = 2*n_open + 1
    return n_open*np.asarray(signal, dtype=float)/np.sqrt(n_open*np.asarray(total, dtype=float)
                                                           + n_pixels*np.asarray(background, dtype=float))


def pinhole_snr(signal, background=0.0):
    """Signal-to-noise ratio of one element of a single-hole camera: psi/sqrt(psi + B).

    Parameters
    ----------
    signal : number or array
        Photons through the hole from the scene element.
    background : number or array, optional
        Background photons per detector element.

    Returns
    -------
    snr : number or array
    """

    signal = np.asarray(signal, dtype=float)
    return signal/np.sqrt(signal + background)


# ---------------------------------------------------------------------------
# diffuser camera: PSF
# ---------------------------------------------------------------------------

def caustic_distance(correlation_length, sigma, wavelength):
    """Typical focal length of the bumps of a smooth random phase plate.

    For a Gaussian random phase with standard deviation sigma and the
    autocorrelation sigma**2*exp(-r**2/l**2) (as made by
    `wavefront_shaping.random_phase_screen()`), the curvature d2(phi)/dx2 has
    the standard deviation sqrt(12)*sigma/l**2. A bump of that curvature
    focuses at f = k/phi'' ::

        f = 2*pi*l**2/(sqrt(12)*sigma*wavelength).

    The caustic network behind the plate is sharpest at about this distance:
    much closer the intensity is still smooth, much farther the caustics of
    neighbouring bumps overlap.

    Parameters
    ----------
    correlation_length : number
    sigma : number
        Standard deviation of the phase (rad).
    wavelength : number

    Returns
    -------
    f : number
    """

    return 2*np.pi*correlation_length**2/(np.sqrt(12.0)*sigma*wavelength)


def point_source_psf(transmission, x, y, z, wavelength, source_distance, source_x=0.0,
                     source_y=0.0, propagator=rayleigh_sommerfeld_I_TF, normalize=True, **kwargs):
    """Intensity PSF of a mask or diffuser for a point source in front of it.

    The spherical wave exp(i*k*R) of a point source at (source_x, source_y)
    and the distance source_distance in front of the mask is multiplied with
    the transmission and propagated over the distance z to the sensor.

    Parameters
    ----------
    transmission : array
        Complex transmission of the mask (including its aperture).
    x, y : arrays
        Linear or 2d coordinates of the mask (and sensor) grid.
    z : number
        Mask to sensor distance.
    wavelength : number
    source_distance : number
        Distance of the source in front of the mask; np.inf for a plane wave.
    source_x, source_y : numbers, optional
        Lateral position of the source (for an infinitely distant source: the
        tangents of the field angles).
    propagator : callable, optional
        A propagator that keeps the grid; defaults to
        `rayleigh_sommerfeld_I_TF`.
    normalize : bool, optional
        If True, the PSF sums to 1.
    kwargs
        Passed on to the propagator.

    Returns
    -------
    psf : array
        Intensity on the sensor grid.
    """

    X, Y = ensure_meshgrid(x, y)
    k = 2*np.pi/wavelength
    if np.isinf(source_distance):
        incident = np.exp(1j*k*(source_x*X + source_y*Y)/np.sqrt(1.0 + source_x**2 + source_y**2))
    else:
        R = np.sqrt((X - source_x)**2 + (Y - source_y)**2 + source_distance**2)
        incident = np.exp(1j*k*(R - source_distance))
    field, _, _ = propagator(incident*transmission, x, y, z, wavelength, **kwargs)
    psf = np.abs(field)**2
    if normalize:
        psf = psf/psf.sum()
    return psf


def psf_magnification(z, source_distance):
    """Size of the PSF pattern on the sensor relative to the mask: 1 + z/u.

    The rays from a point at the distance u through the mask features reach
    the sensor at the distance z behind it with the magnification (u + z)/u.
    A point at the lateral position s moves the pattern by -s*z/u.

    Parameters
    ----------
    z : number
        Mask to sensor distance.
    source_distance : number or array
        Distance u of the source in front of the mask (np.inf allowed).

    Returns
    -------
    m : number or array
    """

    u = np.asarray(source_distance, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 + np.where(np.isinf(u), 0.0, z/u)


def scale_psf(psf, factor, order=1):
    """Magnify a centred PSF about the array centre by factor (sum preserved).

    Parameters
    ----------
    psf : array
    factor : number
        Magnification, e.g. psf_magnification(z, u_new)/psf_magnification(z, u_old).
    order : int, optional
        Spline order of the interpolation (scipy.ndimage.map_coordinates).

    Returns
    -------
    scaled : array
        The scaled PSF on the same grid, normalized to the sum of psf.
    """

    from scipy.ndimage import map_coordinates

    psf = np.asarray(psf, dtype=float)
    ny, nx = psf.shape
    cy, cx = ny//2, nx//2
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = np.array([cy + (i - cy)/factor, cx + (j - cx)/factor])
    scaled = map_coordinates(psf, coords, order=order, mode="constant", cval=0.0)
    scaled = np.clip(scaled, 0.0, None)
    return scaled*psf.sum()/scaled.sum()


# ---------------------------------------------------------------------------
# diffuser camera: forward model and deconvolution
# ---------------------------------------------------------------------------

def _otf(psf):
    """OTF of a centred PSF, in FFT order."""

    return np.fft.rfft2(np.fft.ifftshift(psf))


def _crop_slices(full_shape, crop_shape):
    return tuple(slice((n - m)//2, (n - m)//2 + m) for n, m in zip(full_shape, crop_shape))


def convolve_crop(scene, psf, sensor_shape=None):
    """Forward model of a diffuser camera: periodic convolution and sensor crop.

    Parameters
    ----------
    scene : array
        Scene on the sensor grid (projected through the pinhole of the
        mask centre: inverted and scaled by z/u).
    psf : array
        Centred PSF of the same shape.
    sensor_shape : tuple of int, optional
        Shape of the sensor, centred in the grid; defaults to the full grid.
        Pad the scene by at least the PSF size, so that the periodic
        convolution equals the linear one on the sensor.

    Returns
    -------
    picture : array
    """

    picture = np.fft.irfft2(np.fft.rfft2(scene)*_otf(psf), s=np.shape(scene))
    if sensor_shape is not None:
        picture = picture[_crop_slices(np.shape(scene), sensor_shape)]
    return picture


def wiener_deconvolve(picture, psf, regularization):
    """Wiener (Tikhonov) deconvolution: X = B*conj(H)/(|H|**2 + K).

    Parameters
    ----------
    picture : array
        Measured picture on the full grid of the PSF.
    psf : array
        Centred PSF.
    regularization : number
        K, the noise-to-signal power ratio. Relative to the PSF summing to 1
        (|H(0)| = 1).

    Returns
    -------
    scene : array
    """

    psf = np.asarray(psf, dtype=float)
    H = _otf(psf/psf.sum())
    return np.fft.irfft2(np.fft.rfft2(picture)*np.conj(H)/(np.abs(H)**2 + regularization),
                         s=np.shape(psf))


def _grad(x):
    return np.stack((np.roll(x, -1, 0) - x, np.roll(x, -1, 1) - x))


def _grad_adj(g):
    return (np.roll(g[0], 1, 0) - g[0]) + (np.roll(g[1], 1, 1) - g[1])


def total_variation(x):
    """Anisotropic total variation sum(|dx| + |dy|) with periodic differences.

    Parameters
    ----------
    x : array

    Returns
    -------
    tv : number
    """

    return np.abs(_grad(x)).sum()


def admm_tv_deconvolve(picture, psf, tau, n_iter=300, mu1=None, mu2=None, mu3=None,
                       nonnegative=True, x0=None, callback=None):
    """Deconvolution with total variation and a cropped sensor, by ADMM.

    Solves (Antipa et al. 2018) ::

        min_x 1/2*||b - C H x||**2 + tau*||grad x||_1   subject to x >= 0,

    with H the periodic convolution with the PSF on the full grid and C the
    crop to the (smaller, centred) sensor. The splitting v = H x, u = grad x,
    w = x makes every update closed form: v and w pixelwise, u by soft
    thresholding, x by one division in Fourier space.

    Parameters
    ----------
    picture : array
        Measured picture b, the full grid or a centred crop of it.
    psf : array
        Centred PSF on the full grid (pad the sensor by about the PSF size).
    tau : number
        Weight of the total variation, relative to a PSF summing to 1.
    n_iter : int, optional
    mu1, mu2, mu3 : numbers, optional
        Penalty parameters of the three splittings (H x, grad x, x); see
        Notes. Defaults: 1e-2, 1e-3, 1e-2.
    nonnegative : bool, optional
    x0 : array, optional
        Initial scene (default zero).
    callback : callable, optional
        Called as callback(i, x) after every iteration.

    Returns
    -------
    x : array
        The scene on the full grid.

    Notes
    -----
    The minimizer does not depend on mu1, mu2, mu3; they set how fast ADMM
    gets there. The defaults suit a PSF normalized to sum 1 (|H| <= 1) and
    tau of 1e-5 to 1e-3 for pictures of order 0.1 to 1. Too large values
    make the iteration creep, too small ones make it oscillate.
    """

    psf = np.asarray(psf, dtype=float)
    shape = psf.shape
    H = _otf(psf/psf.sum())
    b = np.asarray(picture, dtype=float)
    sl = _crop_slices(shape, b.shape)
    Ctb = np.zeros(shape)
    Ctb[sl] = b
    CtC = np.zeros(shape)
    CtC[sl] = 1.0
    mu1 = 1e-2 if mu1 is None else mu1
    mu2 = 1e-3 if mu2 is None else mu2
    mu3 = 1e-2 if mu3 is None else mu3

    # |grad|^2 in Fourier space for periodic forward differences
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    lap = (2 - 2*np.cos(2*np.pi*ky)) + (2 - 2*np.cos(2*np.pi*kx))
    denom = mu1*np.abs(H)**2 + mu2*lap + (mu3 if nonnegative else 0.0)
    if not nonnegative:
        denom = denom + 1e-12

    conv = lambda a, F: np.fft.irfft2(np.fft.rfft2(a)*F, s=shape)
    x = np.zeros(shape) if x0 is None else np.array(x0, dtype=float)
    Hx = conv(x, H)
    gx = _grad(x)
    xi = np.zeros(shape)
    eta = np.zeros((2,) + shape)
    rho = np.zeros(shape)
    for i in range(n_iter):
        # u: soft thresholding of the gradient
        t = gx + eta/mu2
        u = np.sign(t)*np.maximum(np.abs(t) - tau/mu2, 0.0)
        # v: data term with the crop, pixelwise
        v = (xi + mu1*Hx + Ctb)/(CtC + mu1)
        # w: positivity
        if nonnegative:
            w = np.maximum(rho/mu3 + x, 0.0)
            r = mu3*w - rho
        else:
            r = np.zeros(shape)
        r = r + _grad_adj(mu2*u - eta) + conv(mu1*v - xi, np.conj(H))
        x = np.fft.irfft2(np.fft.rfft2(r)/denom, s=shape)
        Hx = conv(x, H)
        gx = _grad(x)
        xi += mu1*(Hx - v)
        eta += mu2*(gx - u)
        if nonnegative:
            rho += mu3*(x - w)
        if callback is not None:
            callback(i, x)
    return x
