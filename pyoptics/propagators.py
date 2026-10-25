#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Propagators for electromagnetic fields.

The following propagators are available:

- rayleigh_sommerfeld_I_IR()
- rayleigh_sommerfeld_I_TF()
- fresnel_TF()
- fresnel_IR()
- fresnel_rescaling()
- fraunhofer()
- rayleigh_sommerfeld_I_DI()

Propagators onto a free output window (center, extent and sampling chosen
independently of the input grid, evaluated with chirp-z transforms, see
fft.czt()):

- shifted_angular_spectrum(): exact (Rayleigh-Sommerfeld) propagation to an
  off-axis or rescaled window, band limited after Matsushima (2010)
- fresnel_scaled(): Fresnel integral (single transform) on any window
- fraunhofer_scaled(): Fraunhofer integral on any window
- far_field(): far field amplitude in chosen directions

Propagation onto tilted planes (implemented in pyoptics.tilted, re-exported
here):

- tilted_plane(): scalar field on a plane with a rotated normal, or from a
  tilted plane to a parallel one
- vectorial_tilted_plane(): the same for vector fields, in the plane's axes

Vectorial propagation (fields of shape (Ny, Nx, 2) or (Ny, Nx, 3) with the
components [E_x, E_y(, E_z)]):

- vectorial_rayleigh_sommerfeld_TF(): exact angular spectrum propagation
- propagate_vectorial(): any of the scalar propagators for E_x, E_y, plus E_z
- vectorial_mansuripur(): Mansuripur's high NA (aplanatic lens) diffraction
- longitudinal_component(): E_z from E_x and E_y (div E = 0)

The routines with _TF in their name use a transfer function approach, whereas
the routines whose names end in _IR use formulations with using the impulse
response. Mathematically, those approaches are identical; however, they may
differ in their sampling properties.

The propagators share a common interface:

propagated_field, x, y = propagator_func(field, x_prime, y_prime, z, wavelength),

but individual propagators may accept further arguments.

The input grid must have x = 0 and y = 0 at index N//2 (FFT convention, e.g.
utils.grid1d(-L/2, L/2, N) for even N, dx*(np.arange(N) - N//2) for any N;
see pyoptics.fft); the impulse response based propagators,
fresnel_rescaling() and fraunhofer() raise a ValueError otherwise.

FFT based convolutions are cyclic. Use propagate_padded() to zero pad the
field if it spreads beyond the computational window, and band_limit=True in
rayleigh_sommerfeld_I_TF() for large propagation distances.

Convention: fields vary as exp(i(k.r - omega*t)), the time dependence
exp(-i*omega*t) being implicit. Propagation by z > 0 thus multiplies a plane
wave by exp(+i*k_z*z), the Green's function is exp(+i*k*r)/r and evanescent
waves decay as exp(-|k_z|*z). FT() (kernel exp(-i*k_x*x)) yields the
amplitudes of the plane waves exp(+i*(k_x*x + k_y*y)), inv_FT() sums them up.
Negative z propagates backwards. In rayleigh_sommerfeld_I_TF(), absorbing
media can be described by a complex n = n' + i*kappa with kappa > 0.
"""

# TODO: can from_spectrum switches be implemented? this might save Fourier transforms.

import warnings
from math import pi, ceil

import numpy as np

from .fft import (fft2, ifft2, fftfreq, fftshift, ifftshift, FT, inv_FT, FT_unitary,
                  inv_FT_unitary, zoom_FT, zoom_inv_FT)
from .utils import (freq_grid, frequencies, wavenumber, k_z, TWOPI, simpson_weights,
                    weight_grid, ensure_meshgrid, get_dx, pad, unpad)


def _linear_coords(x, y):
    """Return linear coordinate arrays from linear or meshed x, y."""

    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 2:
        x, y = x[0, :], y[:, 0]

    return x, y


def _check_centered(x, y, func_name):
    """Raise ValueError if the grid does not have x = 0 and y = 0 at index
    N//2. Propagators that sample a kernel or a chirp on the grid rely on this
    FFT convention (fftshift/ifftshift).
    """

    for name, coords in zip(("x", "y"), _linear_coords(x, y)):
        N = len(coords)
        if abs(coords[N//2]) > 1E-6*abs(get_dx(coords)):
            raise ValueError("{}() needs {} = 0 at index N//2 (here {} = {:g}). "
                             "Use e.g. utils.grid1d(-L/2, L/2, N) with even N "
                             "(it omits the right end point) or "
                             "dx*(np.arange(N) - N//2).".format(func_name, name, name, coords[N//2]))


def propagate_padded(propagator, field, x_prime, y_prime, z, wavelength, n=1.0,
                     pad_factor=2, **kwargs):
    """Call propagator on a zero padded field and crop the result.

    FFT based convolutions are cyclic: light leaving the computational window
    on one side reenters on the other. Zero padding enlarges the window to
    pad_factor times its size and thereby suppresses this wrap-around.

    Parameters
    ----------
    propagator : callable
        One of the propagators that keep the grid (rayleigh_sommerfeld_I_TF,
        rayleigh_sommerfeld_I_IR, rayleigh_sommerfeld_I_DI, fresnel_TF,
        fresnel_IR) or of the vectorial ones (vectorial_rayleigh_sommerfeld_TF,
        vectorial_mansuripur). Vector fields of shape (Ny, Nx, 2) or (Ny, Nx, 3)
        are padded componentwise; the vectorial propagators compute E_z in
        the padded window.
    field, x_prime, y_prime, z, wavelength, n
        As for the propagator.
    pad_factor : number, optional
        Size of the padded window relative to the original one.
    kwargs
        Passed on to propagator.

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (the same as given).
    """

    if propagator in (fresnel_rescaling, fraunhofer):
        raise ValueError("Padding changes the output grid of {}(); pad the "
                         "field yourself instead.".format(propagator.__name__))

    x, y = _linear_coords(x_prime, y_prime)
    field_pad, x_pad, y_pad = pad(np.asarray(field, dtype=complex), x, y, factor=pad_factor)

    propagated_pad, _, _ = propagator(field_pad, x_pad, y_pad, z, wavelength, n=n, **kwargs)

    return unpad(propagated_pad, x_pad, y_pad, np.shape(field)[:2])[0], x_prime, y_prime


# NOTE: wavelength arguments are vacuum wavelengths; inside a medium of
# refractive index n the propagators use the wavelength wavelength/n.

def suggest_N_critical(L, z, wl):
    """Suggest number of samples N for "critical sampling" -- for this number
    of samples, both the impulse response and the transfer function for the
    Fresnel propagator are well sampled.

    Parameters
    ----------
    L : number
        Sampling box side length.
    z : number
        Propagation distance.
    wl : number
        Wavelength.

    Returns
    -------
    N_crit : integer
    """

    # dx_crit = wl*z/L
    N_crit = round(L**2/(wl*z)) + 1 

    return round(N_crit)


def rayleigh_sommerfeld_I_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Compute the Rayleigh-Sommerfeld integral of first kind using FFTs. The
    convolution with Green's function is performed in k-space, with the Green's
    function being explicitly taken to k-space by FFT.

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    _check_centered(x_prime, y_prime, "rayleigh_sommerfeld_I_IR")

    k = wavenumber(wavelength, n)
    dx_prime, dy_prime = get_dx(x_prime), get_dx(y_prime)
    X_prime, Y_prime = ensure_meshgrid(x_prime, y_prime)
    r = np.sqrt(z**2 + X_prime**2 + Y_prime**2)
    wl_medium = wavelength/n

    IR = z/(1j*wl_medium)*np.exp(1j*k*r)/r**2

    field_propagated = inv_FT(FT(field)*FT(IR)*dx_prime*dy_prime)

    return field_propagated, x_prime, y_prime


def rayleigh_sommerfeld_I_TF(field, x_prime, y_prime, z, wavelength, n=1.0, band_limit=False):
    """Compute the Rayleigh-Sommerfeld integral of first kind using FFTs. The
    convolution with Green's function is performed in k-space, with the Green's
    function being analytically transformed to k-space.

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional
    band_limit : boolean, optional
        If True, remove the frequencies for which the transfer function is
        undersampled (band-limited angular spectrum method of Matsushima and
        Shimobaba, Opt. Express 17, 19662 (2009)). This avoids aliasing for
        large z; combine with propagate_padded() as in the paper.

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    k = wavenumber(wavelength, n)
    K_X, K_Y = freq_grid(x_prime, y_prime, wavenumbers=True, normal_order=True)
    KZ = k_z(k, K_X, K_Y)

    TF = np.exp(1j*KZ*z)

    if band_limit:
        wl_medium = wavelength/n
        x, y = _linear_coords(x_prime, y_prime)
        # frequency spacing 1/(window size) in x and y:
        du, dv = 1/(len(x)*get_dx(x)), 1/(len(y)*get_dx(y))
        u_limit = 1/(wl_medium*np.sqrt((2*du*z)**2 + 1))
        v_limit = 1/(wl_medium*np.sqrt((2*dv*z)**2 + 1))
        TF[(np.abs(K_X/TWOPI) > u_limit) | (np.abs(K_Y/TWOPI) > v_limit)] = 0.0

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated, x_prime, y_prime


def rayleigh_sommerfeld_I_DI(field, x_prime, y_prime, z, wavelength, n=1.0, use_simpson=True):
    """Implement the Rayleigh-Sommerfeld direct integration method of Shen and
    Wang.

    This method expressed a numerical integration of the Rayleigh-Sommerfeld
    diffraction formula using Simpon's rule as a convolution product which can
    be computed by means of FFTs.
    
    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional
    use_simpson : boolean, optional
        If True (default), use a Simpson integration; if False, use a midpoint
        rule.

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    # TODO: take out g(x,y,z) and decide whether to numexpress it on import...

    N = np.shape(field)
    assert(N[0] == N[1])  # TODO: raise more expressive exception
    N = N[0]

    k = wavenumber(wavelength, n)
    dx, dy = get_dx(x_prime), get_dx(y_prime)

    if use_simpson:
        weights = weight_grid(simpson_weights, N, N)
    else:
        weights = np.ones([N, N])

    U = np.block([ [weights*field, np.zeros([N, N-1])],
                   [np.zeros([N-1, N]), np.zeros([N-1, N-1])]
                 ])

    # adapt Shen/Wang's indexing to zero-based indexing:
    x_vec = np.array( [x_prime[0] - x_prime[N - j] for j in range(1, N) ] +  [x_prime[j - N] - x_prime[0] for j in range(N, 2*N) ] )
    y_vec = np.array( [y_prime[0] - y_prime[N - j] for j in range(1, N) ] +  [y_prime[j - N] - y_prime[0] for j in range(N, 2*N) ] )
    X, Y = ensure_meshgrid(x_vec, y_vec)

    r = np.sqrt(X**2 + Y**2 + z**2)
    H = 1/TWOPI*np.exp(1j*k*r)/r**2*z*(1/r - 1j*k)   # TODO: two pi? last terms in parens?

    val = ifft2(fft2(U)*fft2(H))*dx*dy
    field_propagated = val[-N:, -N:]  # lower right sub-matrix

    return field_propagated, x_prime, y_prime


def fresnel_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Compute the Fresnel difraction integral using FFTs. The convolution with
    Green's function is performed in k-space, with the Green's function being
    explicitly taken to k-space by FFT.

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    _check_centered(x_prime, y_prime, "fresnel_IR")

    k = wavenumber(wavelength, n)

    dx_prime, dy_prime = get_dx(x_prime), get_dx(y_prime)
    X_prime, Y_prime = ensure_meshgrid(x_prime, y_prime)

    wl_medium = wavelength/n

    IR = np.exp(1j*k*z)/(1j*wl_medium*z)*np.exp(1j*k/(2.0*z)*(X_prime**2 + Y_prime**2))

    field_propagated = inv_FT(FT(field)*FT(IR)*dx_prime*dy_prime)

    return field_propagated, x_prime, y_prime


def fresnel_TF(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Compute the Fresnel diffraction integral using FFTs. The
    convolution with the Fresnel Green's function is performed in k-space, with
    the Green's function being analytically transformed to k-space.

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    k = wavenumber(wavelength, n)
    F_x, F_y = freq_grid(x_prime, y_prime, wavenumbers=False, normal_order=True)

    wl_medium = wavelength/n

    TF = np.exp(1j*k*z)*np.exp(-1j*pi*wl_medium*z*(F_x**2 + F_y**2))

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated, x_prime, y_prime


def fresnel_rescaling(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Compute the Fresnel integral (paraxial approximation) on a rescaled
    coordinate grid.
    
    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    k = wavenumber(wavelength, n)
    wl_medium = wavelength/n

    X_prime, Y_prime, x_new, y_new, X_new, Y_new = _new_coordinates_mesh(x_prime, y_prime, z, wl_medium)

    prefactor = np.exp(1j*k*z)/(1j*wl_medium*z)*np.exp(1j*k/(2*z)*(X_new**2 + Y_new**2))
    dx, dy = get_dx(x_prime), get_dx(y_prime)

    field_propagated = prefactor*FT(field*np.exp(1j*k/(2*z)*(X_prime**2 + Y_prime**2)))*dx*dy

    return field_propagated, x_new, y_new


def fraunhofer(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Compute the Fraunhofer integral (farfield approximation) using FFTs.
    
    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
       Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional

    Returns
    -------
    field_propagated : array
    x, y : arrays
        Coordinates (2d) in final plane.
    """

    k = wavenumber(wavelength, n)
    wl_medium = wavelength/n

    X_prime, Y_prime, x_new, y_new, X_new, Y_new = _new_coordinates_mesh(x_prime, y_prime, z, wl_medium)

    prefactor = np.exp(1j*k*z)/(1j*wl_medium*z)*np.exp(1j*k/(2*z)*(X_new**2 + Y_new**2))
    dx, dy = get_dx(x_prime), get_dx(y_prime)

    field_propagated = prefactor*FT(field)*dx*dy

    return field_propagated, x_new, y_new


def _new_coordinates_mesh(x_prime, y_prime, z, wavelength):
    _check_centered(x_prime, y_prime, "fraunhofer/fresnel_rescaling")

    # create coordinates meshes for both "object" (primed coordinates) and
    # "image" space ("new" coordinates):
    X_prime, Y_prime = ensure_meshgrid(x_prime, y_prime)
    x_new, y_new = _fraunhofer_coord_scale(x_prime, y_prime, z, wavelength)
    X_new, Y_new = ensure_meshgrid(x_new, y_new)

    return X_prime, Y_prime, x_new, y_new, X_new, Y_new


def _fraunhofer_coord_scale(x, y, z, wavelength):
    """Scaled coordinates for Fraunhofer & coordinate scaling Fresnel propagator.

    Uses the fact that x = f_x*lambda*z with f_x as usual in FFT computations.
    """

    f_x, f_y = map(lambda item: fftshift(fftfreq(len(item), d=get_dx(item))), (x, y))
    x_new, y_new = wavelength*z*f_x, wavelength*z*f_y

    return x_new, y_new


# Propagation onto free output windows
# ------------------------------------
#
# The following propagators evaluate the final field on an output grid
# x_out, y_out of their own: any center, extent and sampling (equidistant).
# Fourier sums are evaluated with chirp-z transforms (fft.zoom_FT() and
# zoom_inv_FT()), which cost about as much as a few FFTs of size N + M.

def _window_coords(x_out, y_out, func_name):
    """Linear output coordinates from linear or meshed x_out, y_out."""

    if x_out is None or y_out is None:
        raise ValueError("{}() needs output coordinates x_out and y_out".format(func_name))

    return _linear_coords(np.atleast_1d(x_out), np.atleast_1d(y_out))


def _extent(coords):
    """Distance between the first and the last sample (at least one sample
    spacing, also for a single sample)."""

    return max(abs(coords[-1] - coords[0]), abs(get_dx(coords)) if len(coords) > 1 else 0.0)


def shifted_angular_spectrum(field, x_prime, y_prime, z, wavelength, x_out, y_out,
                             n=1.0, band_limit=True, pad_factor="auto"):
    """Angular spectrum propagation onto a shifted (and rescaled) window.

    Evaluates the exact solution of the Helmholtz equation (as
    rayleigh_sommerfeld_I_TF())

        U(x, y, z) = 1/(Nx*Ny) sum_mn A[m, n]*exp(i*(k_x*x + k_y*y + k_z*z))

    with the plane wave amplitudes A of the field in the initial plane, at
    the points of an output window x_out, y_out that may lie anywhere (e.g.
    off-axis, beside the input window) and may have a sampling of its own.
    The sum is evaluated with chirp-z transforms (fft.zoom_inv_FT()).

    The discrete spectrum describes a periodic continuation of the input
    field. To keep the replicas out of the output window, the input is zero
    padded to a window of at least the size of input plus output window, and
    (band_limit=True) only plane waves that travel from the input window into
    the output window are kept: a plane wave with the wavevector (k_x, k_y,
    k_z) moves laterally by z*k_x/k_z, which must lie between
    min(x_out) - max(x_prime) and max(x_out) - min(x_prime) (the same for y).
    This also guarantees that the transfer function is sampled finely enough
    (Matsushima's shifted angular spectrum method; for an output window
    equal to the input window it is the band limit of
    rayleigh_sommerfeld_I_TF(band_limit=True)). Evanescent waves are dropped
    if band_limit is True.

    The band shrinks with the distance: the spectral spacing 2*pi/L_pad
    (L_pad: padded window) must resolve it. A warning is issued if fewer
    than four plane waves are kept along x or y; increase pad_factor then, or
    use fresnel_scaled() or far_field(), which have no such limit (in the
    Fresnel and far field regime).

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
        Linear or 2d equidistant coordinates of the field. They need not be
        centered.
    z : number
        Propagation distance (negative z propagates backwards).
    wavelength : number
        Vacuum wavelength.
    x_out, y_out : arrays
        Linear or 2d equidistant coordinates of the output window.
    n : number, optional
        Refractive index.
    band_limit : boolean, optional
        Apply the band limit described above (default). Without it, light
        that leaves the output window may reenter from the periodic replicas.
    pad_factor : "auto" or number, optional
        Size of the padded input window relative to the input window. "auto"
        (default) pads to at least input plus output window in x and y. Use
        1 to switch padding off.

    Returns
    -------
    field_propagated : array
        Field of shape (len(y_out), len(x_out)).
    x_out, y_out : arrays
        The output coordinates as given.

    Sources
    -------
    K. Matsushima, "Shifted angular spectrum method for off-axis numerical
    propagation," Opt. Express 18, 18453-18463 (2010)
    K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for
    numerical simulation of free-space propagation in far and near fields,"
    Opt. Express 17, 19662-19673 (2009)
    """

    x_in, y_in = _linear_coords(x_prime, y_prime)
    x_o, y_o = _window_coords(x_out, y_out, "shifted_angular_spectrum")
    field = np.asarray(field, dtype=complex)
    Ny, Nx = field.shape

    if pad_factor == "auto":
        shape = tuple(int(ceil((_extent(c_in) + _extent(c_o))/abs(get_dx(c_in)))) + 1
                      for c_in, c_o in ((y_in, y_o), (x_in, x_o)))
        shape = (max(shape[0], Ny), max(shape[1], Nx))
        field_pad, x_pad, y_pad = pad(field, x_in, y_in, shape=shape)
    else:
        field_pad, x_pad, y_pad = pad(field, x_in, y_in, factor=pad_factor)

    # plane wave amplitudes referring to the origin (the grid need not be centered):
    kx, ky = frequencies(x_pad), frequencies(y_pad)
    K_X, K_Y = np.meshgrid(kx, ky)
    A = FT(field_pad)*np.exp(-1j*(K_X*x_pad[len(x_pad)//2] + K_Y*y_pad[len(y_pad)//2]))

    k = wavenumber(wavelength, n)
    KZ = k_z(k, K_X, K_Y)
    TF = np.exp(1j*KZ*z)

    if band_limit:
        propagating = (np.real(KZ) > 0) & (np.abs(np.imag(KZ)) < 1E-12*np.abs(k))
        KZ_safe = np.where(propagating, np.real(KZ), 1.0)
        keep = propagating.copy()
        for K, c_in, c_o in ((K_X, x_in, x_o), (K_Y, y_in, y_o)):
            shift = z*K/KZ_safe
            keep &= (shift >= c_o.min() - c_in.max()) & (shift <= c_o.max() - c_in.min())
        TF = np.where(keep, TF, 0.0)
        N_band = min(np.count_nonzero(keep.any(axis=0)), np.count_nonzero(keep.any(axis=1)))
        if N_band < 4:
            warnings.warn("shifted_angular_spectrum(): the band limit keeps only {} "
                          "plane wave(s) along x or y; the angular spectrum is too coarse "
                          "for this distance. Increase pad_factor or use fresnel_scaled() "
                          "or far_field() for large distances.".format(N_band), stacklevel=2)

    field_propagated = zoom_inv_FT(A*TF, kx, ky, x_o, y_o)

    return field_propagated, x_out, y_out


def _check_fresnel_sampling(x, y, z, wl_medium, x_out, y_out, chirp, func_name):
    """Warn if the input chirp exp(i*k*r**2/(2z)) is undersampled or if the
    output window is larger than the period wl*z/dx of the discrete Fresnel
    transform.
    """

    for name, c, c_o in (("x", x, x_out), ("y", y, y_out)):
        dx = abs(get_dx(c))
        if chirp:
            r_max = np.max(np.abs(c))
            # local frequency of the chirp k*r/z must stay below pi/dx:
            if r_max/(wl_medium*abs(z)) > 1/(2*dx):
                warnings.warn("{}(): the Fresnel chirp is undersampled at the edge "
                              "of the input window in {} (|z| = {:g} < 2*max|{}|*d{}/wavelength "
                              "= {:g}); use shifted_angular_spectrum() for short distances"
                              .format(func_name, name, abs(z), name, name,
                                      2*r_max*dx/wl_medium), stacklevel=3)
        if _extent(c_o) > wl_medium*abs(z)/dx:
            warnings.warn("{}(): the output window in {} is larger than the period "
                          "wavelength*z/d{} = {:g} of the discrete transform; it "
                          "contains replicas".format(func_name, name, name,
                                                     wl_medium*abs(z)/dx), stacklevel=3)


def _fresnel_single_transform(field, x_prime, y_prime, z, wavelength, x_out, y_out, n,
                              chirp, func_name):
    x_in, y_in = _linear_coords(x_prime, y_prime)
    x_o, y_o = _window_coords(x_out, y_out, func_name)

    k = wavenumber(wavelength, n)
    wl_medium = wavelength/n
    _check_fresnel_sampling(x_in, y_in, z, wl_medium, x_o, y_o, chirp, func_name)

    X_in, Y_in = np.meshgrid(x_in, y_in)
    X_o, Y_o = np.meshgrid(x_o, y_o)
    if chirp:
        field = field*np.exp(1j*k/(2.0*z)*(X_in**2 + Y_in**2))

    prefactor = np.exp(1j*k*z)/(1j*wl_medium*z)*np.exp(1j*k/(2.0*z)*(X_o**2 + Y_o**2))
    dx, dy = get_dx(x_in), get_dx(y_in)

    # the kernel exp(-i*k*(x*x' + y*y')/z) is FT() at k_x = k*x/z:
    field_propagated = prefactor*zoom_FT(field, x_in, y_in, k*x_o/z, k*y_o/z)*dx*dy

    return field_propagated, x_out, y_out


def fresnel_scaled(field, x_prime, y_prime, z, wavelength, x_out, y_out, n=1.0):
    """Fresnel diffraction integral on a free output window.

    Evaluates the single transform form of the Fresnel integral

        U(x, y) = exp(i*k*z)/(i*wavelength*z)*exp(i*k*(x**2 + y**2)/(2*z))
                  * sum U'(x', y')*exp(i*k*(x'**2 + y'**2)/(2*z))
                               *exp(-i*k*(x*x' + y*y')/z)*dx'*dy'

    with chirp-z transforms (fft.zoom_FT()), on any equidistant output grid.
    In contrast to fresnel_rescaling() (output spacing wavelength*z/L), the
    output spacing, center and number of samples are free; e.g. the focus of
    a lens can be sampled finely without padding.

    The sum is exact for the sampled input, so the result does not suffer
    from wrap-around. It is accurate as long as the input chirp
    exp(i*k*r'**2/(2z)) times the field is sampled well; a warning is issued
    if the chirp alone is undersampled at the edge of the input window
    (|z| < 2*max|x'|*dx/wavelength) or if the output window exceeds the
    period wavelength*z/dx of the result (it then contains replicas). For
    short distances use shifted_angular_spectrum().

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
        Linear or 2d equidistant coordinates of the field. They need not be
        centered: the phases refer to x' = y' = 0.
    z : number
        Propagation distance (z != 0, negative z propagates backwards).
    wavelength : number
        Vacuum wavelength.
    x_out, y_out : arrays
        Linear or 2d equidistant output coordinates.
    n : number, optional
        Refractive index.

    Returns
    -------
    field_propagated : array
        Field of shape (len(y_out), len(x_out)).
    x_out, y_out : arrays
        The output coordinates as given.

    Note
    ----
    With x_out, y_out the output grid of fresnel_rescaling(), the results of
    both agree.
    """

    return _fresnel_single_transform(field, x_prime, y_prime, z, wavelength, x_out, y_out,
                                     n, True, "fresnel_scaled")


def fraunhofer_scaled(field, x_prime, y_prime, z, wavelength, x_out, y_out, n=1.0):
    """Fraunhofer integral on a free output window.

    As fresnel_scaled(), but without the input chirp (far field
    approximation, Fresnel number of the input much smaller than one). With
    x_out, y_out the output grid of fraunhofer(), both agree.

    Parameters and returns as for fresnel_scaled().
    """

    return _fresnel_single_transform(field, x_prime, y_prime, z, wavelength, x_out, y_out,
                                     n, False, "fraunhofer_scaled")


def far_field(field, x_prime, y_prime, wavelength, sigma_x, sigma_y, n=1.0):
    """Far field amplitude of a field in chosen directions.

    For r -> infinity in the direction of the unit vector s = (sigma_x,
    sigma_y, sigma_z), the Rayleigh-Sommerfeld (first kind) field of the
    input plane becomes

        U(r*s) = f(s)*exp(i*k*r)/r,
        f(s) = k*sigma_z/(2*pi*i) * sum U'(x', y')*exp(-i*k*(sigma_x*x' + sigma_y*y'))*dx'*dy'

    (stationary phase evaluation of the angular spectrum; exact in this
    limit, no paraxial approximation). f is evaluated on an equidistant grid
    of direction cosines with chirp-z transforms, so the angular range and
    resolution are free (e.g. a fine grid over one diffraction order).
    Directions with sigma_x**2 + sigma_y**2 > 1 get f = 0.

    Parameters
    ----------
    field : array
    x_prime, y_prime : arrays
        Linear or 2d equidistant coordinates of the field; the phase of f
        refers to x' = y' = 0.
    wavelength : number
        Vacuum wavelength.
    sigma_x, sigma_y : arrays
        Linear, equidistant direction cosines, sigma_x = sin(theta_x) for a
        direction in the x-z plane at the angle theta_x to the z axis (inside
        the medium of index n).
    n : number, optional
        Refractive index.

    Returns
    -------
    f : array
        Far field amplitude of shape (len(sigma_y), len(sigma_x)); |f|**2 is
        the radiant intensity (power per solid angle up to the medium's
        impedance factor).
    sigma_x, sigma_y : arrays
        The directions as given.
    """

    x_in, y_in = _linear_coords(x_prime, y_prime)
    s_x, s_y = _linear_coords(np.atleast_1d(sigma_x), np.atleast_1d(sigma_y))

    k = wavenumber(wavelength, n)
    S_X, S_Y = np.meshgrid(s_x, s_y)
    sigma_z = np.sqrt(np.clip(1.0 - S_X**2 - S_Y**2, 0.0, None))
    A = zoom_FT(field, x_in, y_in, k*s_x, k*s_y)*get_dx(x_in)*get_dx(y_in)
    f = np.where(S_X**2 + S_Y**2 <= 1.0, k*sigma_z/(1j*TWOPI)*A, 0.0)

    return f, sigma_x, sigma_y


# Vectorial propagation
# ---------------------
#
# Vector fields are arrays of shape (Ny, Nx, 2) with the components [E_x, E_y]
# or of shape (Ny, Nx, 3) with [E_x, E_y, E_z]. In a homogeneous, charge-free
# medium div(E) = 0 holds, i.e. k.E = 0 for every plane wave
# exp(i(k_x*x + k_y*y + k_z*z)) of the angular spectrum. The transverse
# components E_x, E_y in one plane thus fix the field completely: they
# propagate like scalar fields, and E_z follows from
# E_z(k) = -(k_x*E_x(k) + k_y*E_y(k))/k_z.

def _split_vector_field(field, func_name):
    """Return E_x, E_y of a vector field of shape (Ny, Nx, 2) or (Ny, Nx, 3)."""

    field = np.asarray(field)
    if field.ndim != 3 or field.shape[2] not in (2, 3):
        raise ValueError("{}() needs a vector field of shape (Ny, Nx, 2) or "
                         "(Ny, Nx, 3), got shape {}".format(func_name, field.shape))

    return field[:, :, 0], field[:, :, 1]


def longitudinal_component(E_x, E_y, x, y, wavelength, n=1.0):
    """Return E_z of a field from its transverse components E_x, E_y.

    Uses div(E) = 0 in a homogeneous, charge-free medium: for every plane wave
    of the angular spectrum k.E = 0, hence E_z(k) = -(k_x*E_x(k) + k_y*E_y(k))/k_z.
    Evanescent waves (imaginary k_z) are included.

    The factor 1/k_z diverges for grazing waves (|k_perp| -> k). The
    singularity is integrable, but a frequency sample that happens to lie
    very close to the circle |k_perp| = k would get an arbitrarily large
    weight. |k_z| is therefore bounded from below by sqrt(k*dk) (dk: the
    smaller frequency spacing of the grid), which is the order of magnitude
    of the mean of 1/k_z over one frequency cell at the circle. Only samples
    within about dk/2 of the circle are affected.

    Parameters
    ----------
    E_x, E_y : arrays
        Transverse field components sampled on x, y.
    x, y : arrays
        Linear or 2d coordinates, centered as pyoptics.fft expects.
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.

    Returns
    -------
    E_z : array
    """

    k = wavenumber(wavelength, n)
    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    KZ = k_z(k, K_x, K_y)

    x_lin, y_lin = _linear_coords(x, y)
    dk = TWOPI/max(len(x_lin)*abs(get_dx(x_lin)), len(y_lin)*abs(get_dx(y_lin)))
    kz_min = np.sqrt(np.abs(k)*dk)
    near_grazing = np.abs(KZ) < kz_min
    KZ = np.where(near_grazing, kz_min*np.exp(1j*np.angle(KZ)), KZ)

    return inv_FT((-K_x*FT(E_x) - K_y*FT(E_y))/KZ)


def vectorial_rayleigh_sommerfeld_TF(field, x_prime, y_prime, z, wavelength, n=1.0,
                                     band_limit=False):
    """Propagate a vector field with the angular spectrum method.

    This is the vectorial counterpart of rayleigh_sommerfeld_I_TF() and an
    exact solution of Maxwell's equations in a homogeneous, charge-free
    medium: E_x and E_y are propagated like scalar fields, E_z is computed in
    the final plane from div(E) = 0 (see longitudinal_component()). The
    angular spectrum is the vectorial Rayleigh-Sommerfeld solution for given
    transverse fields in the initial plane.

    Parameters
    ----------
    field : array
        Vector field of shape (Ny, Nx, 2) ([E_x, E_y]) or (Ny, Nx, 3)
        ([E_x, E_y, E_z]); a given E_z is ignored as it follows from E_x and
        E_y.
    x_prime, y_prime : arrays
        Linear or 2d arrays in the initial plane.
    z : number
        Propagation distance in the same units as x_prime, y_prime and wavelength.
    wavelength : number
    n : number, optional
    band_limit : boolean, optional
        See rayleigh_sommerfeld_I_TF().

    Returns
    -------
    field_propagated : array
        Field [E_x, E_y, E_z] of shape (Ny, Nx, 3).
    x, y : arrays
        Coordinates in final plane (the same as given).

    Note
    ----
    Use propagate_padded() to suppress the wrap-around of cyclic convolutions;
    it accepts vector fields.
    """

    E_x, E_y = _split_vector_field(field, "vectorial_rayleigh_sommerfeld_TF")

    return propagate_vectorial(rayleigh_sommerfeld_I_TF, np.stack([E_x, E_y], axis=-1),
                               x_prime, y_prime, z, wavelength, n=n, band_limit=band_limit)


def propagate_vectorial(propagator, field, x_prime, y_prime, z, wavelength, n=1.0,
                        **kwargs):
    """Propagate a vector field with any of the scalar propagators.

    E_x and E_y are propagated independently by the scalar propagator, which
    is exact for rayleigh_sommerfeld_I_TF() and an approximation of the same
    order as the scalar propagator otherwise. E_z is computed in the final
    plane:

    - for propagators that keep the grid from div(E) = 0 with
      longitudinal_component(),
    - for fraunhofer() and fresnel_rescaling() from the transversality of the
      far field, E.r = 0, i.e. E_z = -(x*E_x + y*E_y)/z.

    Parameters
    ----------
    propagator : callable
        A scalar propagator of this module.
    field : array
        Vector field of shape (Ny, Nx, 2) or (Ny, Nx, 3) (a given E_z is
        ignored).
    x_prime, y_prime, z, wavelength, n
        As for the propagator.
    kwargs
        Passed on to propagator.

    Returns
    -------
    field_propagated : array
        Field [E_x, E_y, E_z] of shape (Ny, Nx, 3).
    x, y : arrays
        Coordinates in final plane.
    """

    E_x, E_y = _split_vector_field(field, "propagate_vectorial")

    E_x_prop, x, y = propagator(E_x, x_prime, y_prime, z, wavelength, n=n, **kwargs)
    E_y_prop, _, _ = propagator(E_y, x_prime, y_prime, z, wavelength, n=n, **kwargs)

    if propagator in (fraunhofer, fresnel_rescaling):
        X, Y = ensure_meshgrid(x, y)
        E_z_prop = -(X*E_x_prop + Y*E_y_prop)/z
    else:
        E_z_prop = longitudinal_component(E_x_prop, E_y_prop, x, y, wavelength, n)

    return np.stack([E_x_prop, E_y_prop, E_z_prop], axis=-1), x, y


def aplanatic_rotation(sigma_x, sigma_y):
    """Polarization rotation of a ray bent into the direction (sigma_x,
    sigma_y, sigma_z) by an aplanatic lens.

    A ray parallel to the optical axis with transverse polarization [p_x, p_y]
    leaves an aplanatic lens (or any system obeying the sine condition) in the
    direction s = (sigma_x, sigma_y, sigma_z), sigma_z = sqrt(1 - sigma_x**2 -
    sigma_y**2). Its field is rotated about the axis z x s, i.e. the s
    component (perpendicular to the meridional plane) is kept and the p
    component is tilted along with the ray (Mansuripur's "psi" factors):

        E = R @ [p_x, p_y],
        R = [[1 - sigma_x**2/(1 + sigma_z), -sigma_x*sigma_y/(1 + sigma_z)],
             [-sigma_x*sigma_y/(1 + sigma_z), 1 - sigma_y**2/(1 + sigma_z)],
             [-sigma_x, -sigma_y]].

    R has orthonormal columns that are perpendicular to s.

    Parameters
    ----------
    sigma_x, sigma_y : numbers or arrays
        Direction cosines of the ray; sigma_x**2 + sigma_y**2 <= 1.

    Returns
    -------
    R : array
        Matrix of shape (3, 2) + np.shape(sigma_x).
    """

    sigma_x, sigma_y = np.broadcast_arrays(np.asarray(sigma_x, dtype=float),
                                           np.asarray(sigma_y, dtype=float))
    sigma_z = np.sqrt(np.clip(1.0 - sigma_x**2 - sigma_y**2, 0.0, None))

    R = np.empty((3, 2) + sigma_x.shape)
    R[0, 0] = 1 - sigma_x**2/(1 + sigma_z)
    R[0, 1] = R[1, 0] = -sigma_x*sigma_y/(1 + sigma_z)
    R[1, 1] = 1 - sigma_y**2/(1 + sigma_z)
    R[2, 0] = -sigma_x
    R[2, 1] = -sigma_y

    return R


def vectorial_mansuripur(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Mansuripur's vectorial diffraction: each plane wave of the angular
    spectrum of a (paraxially described) field is given the polarization of a
    ray bent by an aplanatic lens.

    The angular spectrum T(sigma_x, sigma_y) of the transverse input field
    [E_x, E_y] is interpreted as the field in the exit pupil of an aplanatic
    lens: the pupil point that belongs to the direction sigma = k_perp/k is
    turned into a plane wave whose polarization is aplanatic_rotation() of
    [T_x, T_y], weighted by 1/sqrt(sigma_z). The weight conserves energy: the
    power flowing through every plane z = const equals the power of the
    input field (sum of |E_x|**2 + |E_y|**2 on the grid). The plane waves are
    then propagated exactly to z.

    For a field whose spectrum is a pupil function, e.g. field =
    inv_FT(P) with P(sigma_x, sigma_y) the pupil, the result is the
    focus of an aplanatic high NA lens (Richards-Wolf theory) at the distance
    z from the focal plane. In the paraxial limit (small sigma) it reduces to
    scalar propagation of E_x and E_y. The transverse components at z = 0 do
    not equal the input field for non-paraxial spectra: the input is not the
    field in a physical plane, but the scalar ("paraxial") representative of
    the vector field. Use vectorial_rayleigh_sommerfeld_TF() to propagate a
    field given in a physical plane.

    Only propagating waves (sigma_x**2 + sigma_y**2 < 1) are kept; a lens
    does not produce evanescent waves.

    Parameters
    ----------
    field : array
        Input field of shape (Ny, Nx, 2); the last index holds the x and y
        polarization components.
    x_prime, y_prime : arrays
        Linear or 2d coordinates, centered as pyoptics.fft expects.
    z : number
        Distance from the focal plane.
    wavelength : number
    n : number, optional
        Refractive index of the image space; sigma = k_perp/(k_0*n).

    Returns
    -------
    field_propagated : array
        Field [E_x, E_y, E_z] of shape (Ny, Nx, 3).
    x, y : arrays
        Coordinates (the same as given).

    Sources
    -------
    M. Mansuripur, "Certain computational aspects of vector diffraction problems,"
    J. Opt. Soc. Am. A 6, 786-805 (1989), erratum J. Opt. Soc. Am. A 10, 382 (1993)
    B. Richards and E. Wolf, "Electromagnetic diffraction in optical systems II,"
    Proc. R. Soc. Lond. A 253, 358-379 (1959)
    """

    E_x, E_y = _split_vector_field(field, "vectorial_mansuripur")
    if np.shape(field)[2] != 2:
        raise ValueError("vectorial_mansuripur() needs a field of shape (Ny, Nx, 2)")

    k = wavenumber(wavelength, n)
    K_x, K_y = freq_grid(x_prime, y_prime, wavenumbers=True, normal_order=True)
    Sigma_x, Sigma_y = K_x/k, K_y/k
    propagating = Sigma_x**2 + Sigma_y**2 < 1.0
    Sigma_z = np.sqrt(np.where(propagating, 1.0 - Sigma_x**2 - Sigma_y**2, 1.0))

    R = aplanatic_rotation(np.where(propagating, Sigma_x, 0.0),
                           np.where(propagating, Sigma_y, 0.0))
    weight = np.where(propagating, np.exp(1j*k*z*Sigma_z)/np.sqrt(Sigma_z), 0.0)

    T_x, T_y = FT(E_x), FT(E_y)
    field_propagated = np.stack([inv_FT(weight*(R[b, 0]*T_x + R[b, 1]*T_y))
                                 for b in range(3)], axis=-1)

    return field_propagated, x_prime, y_prime


def vectorial_fresnel(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Deprecated alias of vectorial_mansuripur().

    The name was misleading: the function does not use the Fresnel
    approximation.
    """

    warnings.warn("vectorial_fresnel() is deprecated, use vectorial_mansuripur() "
                  "(same physics) or vectorial_rayleigh_sommerfeld_TF() "
                  "(propagation of a field given in a plane)",
                  DeprecationWarning, stacklevel=2)

    return vectorial_mansuripur(field, x_prime, y_prime, z, wavelength, n=n)



# TODO: also provide error term for Fresnel approxmiation?
def fresnel_number(outer_aperture_diameter, z, wavelength, wavefront_curvature=None, inner_aperture_diameter_factor=0.0):
    """Return Fresnel number for diffraction of circular or annular apertures.

    Parameters
    ----------
    outer_aperture_diameter : number
        Diameter of outer boundary of aperture.
    z : number
        Propagation distance
    wavelength : number
        Wavelength (in same units as propgation distance z)
    wavefront_curvature : number
        Curvature of wavefront in the aperture. Use 1/f for a wavefront that
        (geometrically) focusses at distance f along the optical axis. Defaults
        to None, which is interpreted as infinite curvature, i.e. a collimated
        wavefront.
    inner_aperture_diameter_factor : number, optional
        Factor defining the annulus' inner diameter as
        inner_aperture_diameter_factor * outer_aperture_diameter. Defaults to
        0, i.e. a circular aperture.

    Returns
    -------
    F : number
        Fresnel number, i.e. number of half-wave zones of the aperture as seen
        from the on-axis point at distance z: F = a**2*(1 - eps**2)/wavelength
        * |1/z - c| with a the outer radius, eps the inner diameter factor and
        c the wavefront curvature.
    """

    a = outer_aperture_diameter/2.0
    c = 0.0 if wavefront_curvature is None else wavefront_curvature

    F = a**2*(1 - inner_aperture_diameter_factor**2)/wavelength*np.abs(1.0/z - c)

    return F


def z_from_fresnel_number(F, aperture_diameter, wavelength):
    """Inverse of fresnel_number() for a collimated beam and a circular
    aperture.
    """

    z = (aperture_diameter/2.0)**2/(wavelength*F)

    # TODO: extend to new signature of fresnel_number

    return z


def suggest_lateral_sampling(L, n, wl):
    """Suggest number of samples N such that all propagating frequencies are
    covered.

    Parameters
    ----------
    L : double or array_like
        Lateral extent of one or dimensions.
    n : double
        Refractive index.
    wl : double
        Wavelength (in same units as L).

    Returns
    -------
    N : int or tuple
        Suggested number of samples.
    """

    # df = 1/L; question to ask: for which N is N*df >= n/lambda?
    # suggest something like 2N + 1.
    N = ceil(2.0*n*L/wl) + 1

    # TODO: extend by an optional keyword spatial_extent.
    # use spatial_extent to estimate bandwith using Heisenberg's uncertainty principle.
    # if given, lower bound for relevant frequencies (from all propagating frequencies to all relevant frequencies)

    return N


# propagation onto tilted planes, see pyoptics.tilted:
from .tilted import tilted_plane, vectorial_tilted_plane  # noqa: E402
