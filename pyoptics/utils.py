#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Utils for pyoptics.
"""

from math import pi

import numpy as np
from numpy.fft import fftfreq, fftshift
from numpy.lib.scimath import sqrt as complex_sqrt
from scipy.signal import savgol_coeffs

from .fft import FT_unitary, inv_FT_unitary


# TODO: add convenience routines the create complete grids? like a routine
# that returns the unit square from [-1, +1, -1, +1] input

TWOPI = 2*pi
epsilon_0 = 8.854187817620E-12  #  A^2 s^4 kg^−1 m^−3
mu_0 = 4*pi*1E-7  # V s / A m
Z_0 = np.sqrt(mu_0/epsilon_0)  # vacuum impedance

deg_to_rad = lambda d: d/180.0*pi
rad_to_deg = lambda r: r/pi*180.0
sin_cos = lambda phi: (np.sin(phi), np.cos(phi))
kronecker_delta = lambda n, m : (1.0 if n == m else 0.0)


def get_dx(x):
    """Get dx using a formulation that does not suffer from numerical
    cancellation of significant digits.
    """

    Nx = len(x)
    dx = (x[-1] - x[0])/(Nx - 1.)

    return dx


def complex_average(z_vals):
    """Compute average of complex numbers from r, phi instead of real and imaginary
    part.

    Parameters
    ----------
    z_vals : iterable

    Returns
    -------
    z_avg : complex
    """

    r_vals = np.abs(z_vals)
    phi_vals = np.arctan2(np.imag(z_vals), np.real(z_vals))

    avg = r_vals.mean()*np.exp(1j*phi_vals.mean())

    return avg


def ensure_meshgrid(x, y):
    """Return meshed coordinates for linear or meshed input.

    Parameters
    ----------
    x, y : arrays
        Both 1d (meshed with np.meshgrid) or both 2d (returned as they are).

    Returns
    -------
    XX, YY : 2d arrays

    Raises
    ------
    ValueError
        If x and y are neither both 1d nor both 2d.
    """

    d_x, d_y = len(np.asarray(x).shape), len(np.asarray(y).shape)

    if (d_x == 1) and (d_y == 1):
        XX, YY = np.meshgrid(x, y)
    elif (d_x == 2) and (d_y == 2):
        XX, YY = np.asarray(x), np.asarray(y)
    else:
        raise ValueError("Dimensions of both x and y are unequal to 1 or 2")

    return XX, YY


def _pad_offset(N, N_pad):
    """Offset of the original samples in a padded axis such that the sample
    at index N//2 lands at index N_pad//2 (a centered grid stays centered)."""

    if N_pad < N:
        raise ValueError("Padded size {} is smaller than the size {}".format(N_pad, N))

    return N_pad//2 - N//2


def pad(field, x, y, shape=None, factor=None, value=0.0):
    """Pad a field symmetrically and extend its coordinate arrays.

    The sample at index N//2 is moved to index N_pad//2 of the padded array,
    so a grid that is centered as pyoptics.fft expects (x = 0 at index N//2)
    stays centered and FT() of the padded field has no additional phase.

    Parameters
    ----------
    field : array
        Field of shape (Ny, Nx) or a stack of shape (Ny, Nx, ...) (images
        along the last axes, as in pyoptics.processing).
    x, y : arrays
        Linear, equidistant coordinates of the field.
    shape : tuple of ints, optional
        New shape (Ny_pad, Nx_pad) of the first two axes.
    factor : number, optional
        Alternatively to shape: enlarge both axes by this factor (rounded up
        to the next integer number of samples). Exactly one of shape and
        factor must be given.
    value : number, optional
        Value to pad with. Defaults to 0.

    Returns
    -------
    field_pad : array
        Padded field of the same dtype.
    x_pad, y_pad : arrays
        Coordinates of the padded field with the same spacing.
    """

    if (shape is None) == (factor is None):
        raise ValueError("Give exactly one of shape and factor")

    field = np.asarray(field)
    Ny, Nx = field.shape[:2]
    if factor is not None:
        shape = (int(np.ceil(factor*Ny)), int(np.ceil(factor*Nx)))
    Ny_pad, Nx_pad = shape
    off_y, off_x = _pad_offset(Ny, Ny_pad), _pad_offset(Nx, Nx_pad)

    field_pad = np.full((Ny_pad, Nx_pad) + field.shape[2:], value, dtype=field.dtype)
    field_pad[off_y:off_y + Ny, off_x:off_x + Nx] = field

    x_pad = x[Nx//2] + get_dx(x)*(np.arange(Nx_pad) - Nx_pad//2)
    y_pad = y[Ny//2] + get_dx(y)*(np.arange(Ny_pad) - Ny_pad//2)

    return field_pad, x_pad, y_pad


def unpad(field, x, y, shape):
    """Crop a field to the central region of a given shape; inverse of pad().

    Parameters
    ----------
    field : array
        Field of shape (Ny_pad, Nx_pad) or a stack (Ny_pad, Nx_pad, ...).
    x, y : arrays
        Coordinates of the field.
    shape : tuple of ints
        Shape (Ny, Nx) of the cropped field.

    Returns
    -------
    field_cropped : array
    x_cropped, y_cropped : arrays
    """

    field = np.asarray(field)
    Ny_pad, Nx_pad = field.shape[:2]
    Ny, Nx = shape
    off_y, off_x = _pad_offset(Ny, Ny_pad), _pad_offset(Nx, Nx_pad)

    return (field[off_y:off_y + Ny, off_x:off_x + Nx],
            np.asarray(x)[off_x:off_x + Nx], np.asarray(y)[off_y:off_y + Ny])


def sgn(x):
    """Sign function using the convention sgn(0) = 1.0.
    """

    return np.where(np.asarray(x) >= 0, 1.0, -1.0)[()]


def Z(n):
    """Impedance of medium of (possbily complex) refractive index.
    """

    impedance = Z_0/n   # TODO: is it really abs(n) or rather complex_sqrt(epsilon)?

    return impedance


def I(E, n=1.0):
    """Get intensity from field amplitude. Returns correct result as obtained
    by time-avergaing the Poynting vector.

    Parameters
    ----------
    E : array
        Field amplitude
    n : double, optional
        Refractive index of medium

    Returns
    -------
    I : array
        Field amplitude E**2*n/(2*Z_0)
    """

    intensity = np.abs(E)**2*n/(2*Z_0)

    return intensity


def wavenumber(wavelength, n=1.0):
    """Wavenumber k = 2*pi*n/wavelength.

    Parameters
    ----------
    wavelength : number or array
        Vacuum wavelength.
    n : number, optional
        Refractive index (default: vacuum).

    Returns
    -------
    k : number or array
    """

    return TWOPI*n/wavelength


def k_z(k, k_x, k_y):
    """Return z component of k vector from k_x and k_y.

    Parameters
    ----------
    k : number
    k_x, k_y : arrays

    Returns
    -------
    k_z : array

    Notes
    -----
    The function carefully handles evanescent waves: the branch with
    Im(k_z) >= 0 is returned, so that exp(+i*k_z*z) decays for z > 0
    (exp(-i*omega*t) time dependence). The same holds for complex k in
    absorbing media (k = k_0*(n + i*kappa), kappa > 0).
    """

    return complex_sqrt(k**2 - k_x**2 - k_y**2)


def grid2d(x_min, x_max, y_min, y_max, Nx, Ny=None, assume_periodicity=True):
    """Generate a two-dimensional grid from boundary values and number of
    samples.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : numbers
        Boundary values.
    Nx, Ny : ints
        Number of samples (Ny is optional and defaults to a value equal to Nx).
    assume_periodicity : boolean, optional
        If True, the endpoints are not included in the sampled values. This
        matches FFT conventions.

    Returns
    -------
    x, y : arrays
        Linear coordinate arrays.
    X, Y : arrays
        Meshed coordinate arrays.
    """

    if Ny is None:
        Ny = Nx

    x = grid1d(x_min, x_max, Nx, assume_periodicity=assume_periodicity)
    y = grid1d(y_min, y_max, Ny, assume_periodicity=assume_periodicity)

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y


def grid1d(x_min, x_max, N, assume_periodicity=True):
    """Generate a one-dimensional coordinate array from boundary values and
    number of samples.

    Parameters
    ----------
    x_min, x_max : numbers
        Boundary values.
    N  : int
        Number of samples.
    assume_periodicity : boolean, optional
        If True, the endpoints are not included in the sampled values. This
        matches FFT conventions.

    Returns
    -------
    x : array
        Linear coordinate array. With assume_periodicity=True, x_min = -x_max
        and even N, x = 0 lies at index N//2 as pyoptics.fft.FT() expects; for
        odd N it does not.
        Linear coordinate array.
    """
    x = np.linspace(x_min, x_max, N, endpoint=(not assume_periodicity))

    return x


def frequencies(x, wavenumbers=True, normal_order=True):
    """Return frequencies as used in DFTs.

    Parameters
    ----------
    x : array
    wavenumbers, boolean, optional
        If True, wavenumbers instead of spatial frequencies will be returned.
    normal_order : bool, optional
        If True, the grid is  returned in 'normal' order with zero frequency
        components centered and increasing frequencies. If False, the zero
        frequency component is to the 'left', which is sometimes referred to
        as 'standard order'.

    Returns
    -------
    freq_x : arrays
        Arrays of frequencies/wavenumbers. In normal order, the zero
        frequency is at index N//2, which matches the spectra of
        pyoptics.fft.FT().
    """

    N_x = len(x)
    dx = get_dx(x)
    freq_x = fftfreq(N_x, dx)

    if wavenumbers:
        freq_x = 2*pi*freq_x

    if normal_order:
        freq_x = fftshift(freq_x)

    return freq_x


def freq_grid(x, y, wavenumbers=True, normal_order=True):
    """Compute a Fourier frequency grid corresponding to given x, y.

    Parameters
    ----------
    x,y : arrays
        x and y values to use.
    wavenumbers, boolean, optional
        If True, wavenumbers instead of spatial frequencies will be returned.
    normal_order : bool, optional
        If True, the grid is  returned in 'normal' order with zero frequency
        components centered and increasing frequencies. If False, the zero
        frequency component is to the 'left', which is sometimes referred to
        as 'standard order'.

    Returns
    -------
    freq_x, freq_y : arrays
        Meshed spatial frequencies/wavenumbers.
    """

    freq_x, freq_y = (frequencies(x, wavenumbers, normal_order),
                      frequencies(y, wavenumbers, normal_order),
                     )

    out = np.meshgrid(freq_x, freq_y)

    return out


def get_length_scales(wavelength, NA, n=1.0, use_small_NA=False):
    """Return two length scales for an imaging system: the resolution
    limit/Airy diameter and depth of focus (DOF, for small NA known as the
    Rayleigh unit).

    Parameters
    ----------
    wavelength : double
        The vacuum wavelength.
    NA : double
        Numerical aperture n*sin(alpha) (already includes the refractive
        index).
    n : double, optional
        Refractive index of the medium. Defaults to 1 (air/vacuum).
    use_small_NA : boolean, optional
        If True, use the small NA approximation commonly used in microscopy for
        the DOF.

    Returns
    -------
    airy : double
        The Airy *diameter* (not radius!).
    DOF : double
        The Rayleigh unit.
    """

    airy_diameter = 1.22*wavelength/NA
    DOF = n*wavelength/NA**2  # TODO: better formula for DOF, case distinction

    return airy_diameter, DOF


def get_z_derivative(stack, dz, order):
    r"""Get the derivative \partial_z I for an image stack. This quantity may
    be used in transport of intensity computations.

    Finite difference quotients are used in the computation. Finite differences
    amplfiy noise. To avoid this, a Savitzky-Golay filter is used here for a
    noise smoothing derivative. The idea is to approximate the the derivative
    by a low order polynomial and use a larger number of samples (here: images)
    to define the polynomial in a least squares sense.

    Parameters
    ----------
    stack : array
        Intensity stack. An odd number of equidistant image plane centered
        around the plane of interest is expected.
    dz : double
        Distance of individual image planes.
    order : int
        Order of approximating polynomial. Must be smaller than the number of
        images.

    Returns
    -------
    dIdZ : array
        z-derivative in the central z-plane.
    """

    # TODO: implement default behaviour for order

    num_images = np.size(stack, 2)
    deriv_coeffs = savgol_coeffs(num_images, order, 1, delta=dz, use="dot")

    return np.einsum("ijk,k->ij", stack, deriv_coeffs)


def add_camera_noise(stack, model="camera", **kwargs):
    """Apply a camera noise model to an image or an image stack.

    Shortcut for ``processing.CameraNoiser(model, **kwargs)(stack)``; see there
    for the models and parameters.

    Parameters
    ----------
    stack : array
        Image (2d) or image stack (3d, last index numbering the images). For
        the "poisson" and "camera" models: expected photons per pixel.
    model : str, optional
        "gaussian", "poisson" or "camera".

    Returns
    -------
    noisy : array
    """

    from .processing import CameraNoiser  # processing imports utils

    return CameraNoiser(model, **kwargs)(stack)


def center_of_mass(img, x, y, threshold=0.0):
    """Return the center of mass (centroid) of a (non-negative) image.

    Parameters
    ----------
    img : array
        Image of shape (len(y), len(x)), usually an intensity.
    x, y : arrays
        1d coordinate grids.
    threshold : float, optional
        Pixels below threshold*max(img) are ignored (set to zero).

    Returns
    -------
    x_c, y_c : float
        Coordinates of the centroid.
    """

    img = np.asarray(img, dtype=float)
    if threshold > 0:
        img = np.where(img >= threshold*np.max(img), img, 0.0)

    total = np.sum(img)
    if total == 0:
        raise ValueError("cannot compute the center of mass of an empty image")

    x_c = np.sum(np.sum(img, axis=0)*x)/total
    y_c = np.sum(np.sum(img, axis=1)*y)/total

    return x_c, y_c


def scalar_product(field1, field2, x, y, weight_func=None):
    """Compute the scalar product ``<field_1|field_2>`` (using the physicist's
    convention of complex conjugation for field1).

    Parameters
    ----------
    field1, field2 : arrays
        Sampled fields.
    x, y : arrays
        Linear x and y arrays.
    weight_func : callable, optional
        Function w = w(n) return weights for the one-dimensional integration
        scheme to use. If omitted, a simple midpoint rule is used.

    Returns
    -------
    s : complex
        Scalar product.
    """

    if weight_func is None:
        weights = np.ones(np.shape(field1))
    else:
        weights = weight_grid(weight_func, len(x), len(y))

    prod = scalar_product_with_weights(field1, field2, x, y, weights)

    return prod


def scalar_product_with_weights(field1, field2, x, y, weights):
    """Compute the scalar product ``<field_1|field_2>`` (using the physicist's
    convention of complex conjugation for field1).

    Parameters
    ----------
    field1, field2 : arrays
        Sampled fields.
    x, y : arrays
        Linear x and y arrays.
    weights : array
        Weights array of same shape as field1, field2.

    Returns
    -------
    s : complex
        Scalar product.
    """

    dx, dy = get_dx(x), get_dx(y)
    prod = np.sum(weights*np.conj(field1)*field2)*dx*dy

    return prod


def scalar_product_without_weights(field1, field2, x, y):
    """Compute the scalar product ``<field_1|field_2>`` (using the physicist's
    convention of complex conjugation for field1).

    Parameters
    ----------
    field1, field2 : arrays
        Sampled fields.
    x, y : arrays
        Linear x and y arrays.

    Returns
    -------
    s : complex
        Scalar product.
    """

    dx, dy = get_dx(x), get_dx(y)
    prod = np.sum(np.conj(field1)*field2)*dx*dy

    return prod


def local_momentum(field, wavelength):
    # TODO: implement a sensible notion of local momentum / direction cosines
    # use FT and get angles from frequencies? use dispersion for z-component?
    """Local momentum of a field (not implemented yet, returns None).

    Parameters
    ----------
    field : array
    wavelength : number
    """

    pass


def simpson_weights(N):
    """Return Simpson integration weights (for 1d integration).

    Parameters
    ----------
    N : int

    Returns
    -------
    w : array
        Simpson weigths.

    Notes
    -----
    If N is even, the right-most indices will be treated by a different
    interpolation formula (of same order). This allows Simpson integration for
    even length sampled data.
    """

    w = np.zeros(N)

    # catch even number of integration points - if even, make l index of last
    # "odd" point - compare https://github.com/scipy/scipy/issues/5618
    l = N-1 if N%2 == 1 else N-2

    w[0], w[l] = 2.0, 2.0
    w[1:l:2] = 8.0
    w[2:l-1:2] = 4.0

    if N % 2 == 0:
        w[-3:] += np.array([-1.0, 8, 5])/2
        #w[:3] += np.array([5, 8, -1])/2

        # TODO: also treat "left" interval similarly?
        # or return average of left and right treatment?

    w /= 6.0

    return w


#def trapz_weights(N):


def weight_grid(func, Nx, Ny):
    """Meshed weight grid from 1D weights.

    Parameters
    ----------
    func : callable
        func(N) returns a weight vector of length N, e.g. `simpson_weights`.
    Nx, Ny : int
        Number of samples in x and y.

    Returns
    -------
    w : 2d array
        Shape (Ny, Nx), the outer product of the y and x weights.
    """

    wx = func(Nx)
    wy = func(Ny)

    w = np.einsum("i,j -> ij", wy, wx)
    # TODO: np.outer() seems faster from some N on...

    return w


def Ez(x, y, Ex, Ey, wl, n=1.0):
    """Compute z-component of electrical field from x&y components.
    Assumes charge-free region.

    Parameters
    ----------
    x, y : arrays
        x and y coordinates of sampled fields.
    Ex, Ey : arrays
        x and y field components.
    wl : number
        Wavelength in same units as x and y.
    n : number, optional
        Refractive index of medium.
    """

    # TODO: do we have any reason to allow charges?
    # TODO: test case! (diffraction of linearly polarized Gaussian?)

    # use div(E) = 0: in k-space this equation gives k \cdot E(k) = 0, which needs to
    # be true for all k.

    E_tilde_x, E_tilde_y = FT_unitary(Ex), FT_unitary(Ey)
    # FT_unitary() returns a centered spectrum, so use centered frequencies:
    Nu_x, Nu_y = freq_grid(x, y, wavenumbers=False, normal_order=True)
    Nu_z = k_z(n/wl, Nu_x, Nu_y)

    E_tilde_z = (-Nu_x*E_tilde_x - Nu_y*E_tilde_y)/Nu_z

    E_z = inv_FT_unitary(E_tilde_z)

    return E_z


def super_gaussian(x, y, x0, y0, sigma_x, sigma_y, N):
    """Super Gaussian exp(-``|x - x0|**N``/(2*sigma_x**N) - ``|y - y0|**N``/(2*sigma_y**N)).

    Parameters
    ----------
    x, y : arrays
        Linear or meshed coordinates.
    x0, y0 : numbers
        Centre.
    sigma_x, sigma_y : numbers
        Widths.
    N : number
        Order (N = 2: Gaussian).

    Returns
    -------
    g : 2d array
    """

    XX, YY = ensure_meshgrid(x, y)

    g = np.exp(-np.abs(XX - x0)**N/(2*sigma_x**N) - np.abs(YY - y0)**N/(2*sigma_y**N))

    return g


def image_energy(img):
    """Return energy in an image.

    Parameters
    ----------
    img : array

    Returns
    -------
    E : number
    """

    # TODO: shouldn't physical engery be of function of intensity (using e.g. I() as defined in this module)?
    # TODO: refractive index?

    return np.sum(np.abs(img)**2)


def normalize_img_to_energy(img, E):
    """Normalize a given field to a given energy.

    Parameters
    ----------
    img : array
    E : number

    Returns
    -------
    img_prime : array
    """

    # TODO: I()?
    # TODO: refractive index?

    return img*np.sqrt(E)/np.sqrt(image_energy(img))
