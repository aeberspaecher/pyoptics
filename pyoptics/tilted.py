#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Propagation onto tilted planes.

The field u(x, y) is given in the source plane z = 0. The target plane passes
through the point r0 (usually (0, 0, z0)) and is spanned by the columns of a
rotation matrix R: its local axes x^, y^ and its normal n^ are ::

    x^ = R[:, 0],  y^ = R[:, 1],  n^ = R[:, 2]

in source coordinates, and the target point with local coordinates (x^, y^)
lies at r = r0 + x^*R[:, 0] + y^*R[:, 1]. geometry.R_y(theta) tilts the plane
about the y axis, its normal towards +x.

Every plane wave exp(i*k.r) of the source spectrum reaches the target plane
as exp(i*k.r0)*exp(i*(k^_x*x^ + k^_y*y^)) with the rotated wave vector
k^ = R^T k. The field on the target plane is thus again a superposition of
plane waves in its own coordinates, but on a non-uniform frequency grid.
Two methods evaluate it:

- "exact": the non-uniform plane wave sum, evaluated directly. For a rotation
  about the x or y axis, k^_y (or k^_x) depends on one source frequency
  only, and the sum separates into one matrix product per row of the
  spectrum (O(N³)). For a general rotation, the type 3 non-uniform FFT of
  finufft is used if it is installed, otherwise a direct sum (O(N⁴), only
  sensible for small N). No interpolation, no Jacobian.
- "rotation" (Matsushima, Schimmel and Wyrowski, JOSA A 20, 1755 (2003)):
  the target spectrum is sampled on a uniform k^ grid, each k^ is mapped
  back to k = R k^, the (zero padded) source spectrum is interpolated there
  and multiplied by the Jacobian ``|d(k_x, k_y)/d(k^_x, k^_y)|`` = k_z/k^_z,
  followed by one inverse FFT. Fast (O(N² log N)); the accuracy is set by
  the interpolation (cubic by default) and improves with padding. The
  result is periodic in the output window: the field must decay inside it,
  otherwise it wraps around (the "exact" sum does not).

The Jacobian is not optional: without it, the field on the tilted plane is
off by a factor of about cos(theta) (an error of 1 - cos(theta), 50 % at 60°).

Carrier. A beam travelling along z arrives on a plane tilted by theta with
the transverse wavenumber k^_x = -k*sin(theta): its phase ramps by one
wavelength per lambda/sin(theta). With remove_carrier=True (default) this
mean tilt (the power weighted mean of k^_x, k^_y over the spectrum) is
removed and returned as k_carrier, the field on the plane being ::

    u_true(x^, y^) = u_t(x^, y^)*exp(i*(k_carrier[0]*x^ + k_carrier[1]*y^)).

k_carrier_in tells the routines about a carrier that was removed from an
input field in the same way, e.g. for the way back.

Further notes:

- Only plane waves that cross the target plane in the direction of its
  normal (k^_z > 0) are kept; these are all waves if the source spectrum
  contains no angles beyond 90° - theta. The result is thus a valid field
  for propagation onwards from the tilted plane. Evanescent waves of the
  source are dropped.
- The footprint of a beam stretches by 1/cos(theta) along x^, and the target
  plane spans z0 -/+ (L/2)*sin(theta) along z. Choose the output window
  (x_out, y_out) and the padding accordingly. A ValueError is raised if the
  plane crosses the source plane inside the window.
- The target plane may also lie behind the source plane (all points z < 0):
  the forward travelling waves are then propagated backwards. This is the
  way back from a tilted plane, see reverse.
- ``|u_t|²`` integrated over the tilted plane is not conserved (a tilted
  detector catches the beam on a larger area); the power flux through the
  plane, power_flux(), is.

Functions:

- tilted_plane(): scalar field on a tilted plane (or from a tilted plane to a
  parallel one)
- vectorial_tilted_plane(): the same for [E_x, E_y], returning the field
  components in the plane's own axes
- rotate_angular_spectrum(): low level building block of the "rotation"
  method
- tilted_beam_widths(): ISO 11146 beam widths on an inclined plane (detector)
- power_flux(): power through a plane from the angular spectrum
- rotation_from_angles(): R for tilts about x and y

Convention: fields vary as exp(i(k.r - omega*t)), see pyoptics.propagators.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from .fft import FT, inv_FT, zoom_inv_FT
from .geometry import R_x, R_y
from .utils import frequencies, wavenumber, get_dx, pad as pad_field
from .beam_quality import beam_widths

try:
    import finufft
except ImportError:  # optional speed-up for general rotations in "exact"
    finufft = None


__all__ = ["tilted_plane", "vectorial_tilted_plane", "rotate_angular_spectrum",
           "tilted_beam_widths", "power_flux", "rotation_from_angles"]


_TOL = 1E-12
_CHUNK_ELEMENTS = 2**22  # complex numbers in temporary arrays of the direct sums


def rotation_from_angles(theta_x=0.0, theta_y=0.0):
    """Rotation matrix R = R_x(theta_x) R_y(theta_y).

    The plane is first tilted by theta_y about the y axis (normal towards +x),
    then by theta_x about the x axis (normal towards -y).

    Parameters
    ----------
    theta_x, theta_y : numbers
        Angles in radians.

    Returns
    -------
    R : array
        3x3 rotation matrix.
    """

    return np.dot(R_x(theta_x), R_y(theta_y))


def _rotation(R, angles):
    if R is not None and angles is not None:
        raise ValueError("Give either R or angles, not both")
    if angles is not None:
        R = rotation_from_angles(*angles)
    if R is None:
        R = np.eye(3)
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.allclose(np.dot(R.T, R), np.eye(3), atol=1E-9):
        raise ValueError("R must be a 3x3 rotation matrix")

    return R


def _origin(z0):
    r0 = np.atleast_1d(np.asarray(z0, dtype=float))
    if r0.shape == (1,):
        r0 = np.array([0.0, 0.0, r0[0]])
    if r0.shape != (3,):
        raise ValueError("z0 must be a number or a vector (x0, y0, z0)")

    return r0


def _lin(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 2:
        x, y = x[0, :], y[:, 0]

    return x, y


def _is_centered(x):
    N = len(x)
    return N > 1 and abs(x[N//2]) <= 1E-6*abs(get_dx(x))


def _source_spectra(fields, x, y, pad, k_carrier_in):
    """Zero pad the fields and return their spectra (FT(), unnormalized) with
    the wavenumbers of the samples (carrier included) and the spacings."""

    x, y = _lin(x, y)
    if not (_is_centered(x) and _is_centered(y)):
        raise ValueError("The source grid must have x = 0 and y = 0 at index N//2 "
                         "(see pyoptics.fft)")
    dx, dy = get_dx(x), get_dx(y)

    spectra = []
    for f in fields:
        f = np.asarray(f, dtype=complex)
        if pad is not None and pad > 1:
            f, x_p, y_p = pad_field(f, x, y, factor=pad)
        else:
            x_p, y_p = x, y
        spectra.append(FT(f))
    kx = frequencies(x_p) + k_carrier_in[0]
    ky = frequencies(y_p) + k_carrier_in[1]

    return spectra, kx, ky, dx, dy


def _check_side(r0, R, x_out, y_out, func_name):
    """The target plane must not cross the source plane inside the window."""

    corners_z = [r0[2] + R[2, 0]*xc + R[2, 1]*yc
                 for xc in (np.min(x_out), np.max(x_out))
                 for yc in (np.min(y_out), np.max(y_out))]
    if not (min(corners_z) > 0 or max(corners_z) < 0):
        raise ValueError("{}(): the target plane crosses the source plane z = 0 inside "
                         "the output window (z from {:g} to {:g}); move it further away "
                         "or reduce the window".format(func_name, min(corners_z), max(corners_z)))


def _carrier(spectra, KX, KY, KZ, valid, R):
    """Power weighted mean transverse wavenumber in the target frame."""

    P = sum(np.abs(F)**2 for F in spectra)*valid
    total = np.sum(P)
    if total == 0:
        return np.zeros(2)
    k_mean = np.array([np.sum(P*KX), np.sum(P*KY), np.sum(P*KZ)])/total

    return np.dot(R.T, k_mean)[:2]


def _separable_sum(G, A, b, x_out, y_out):
    """sum_mn G[m, n]*exp(i*(A[m, n]*x_out[p] + b[m]*y_out[q])) for b
    depending on the row m only: one matrix product per block of rows.
    """

    rows = np.nonzero(np.any(G != 0, axis=1))[0]
    equidistant = len(x_out) > 1 and np.allclose(np.diff(x_out), x_out[1] - x_out[0], rtol=1E-9)
    dx_out = x_out[1] - x_out[0] if len(x_out) > 1 else 0.0
    H = np.zeros((len(rows), len(x_out)), dtype=complex)
    chunk = max(1, _CHUNK_ELEMENTS//(G.shape[1]*len(x_out)))
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        if equidistant:
            # exp(i*a*(x_0 + p*dx)) as a cumulative product, much cheaper than exp():
            E = np.empty((len(r), G.shape[1], len(x_out)), dtype=complex)
            E[:, :, 0] = np.exp(1j*A[r]*x_out[0])
            E[:, :, 1:] = np.exp(1j*A[r]*dx_out)[:, :, None]
            np.cumprod(E, axis=2, out=E)
        else:
            E = np.exp(1j*A[r, :, None]*x_out[None, None, :])
        H[start:start + chunk] = np.matmul(G[r][:, None, :], E)[:, 0, :]

    return np.dot(np.exp(1j*np.outer(y_out, b[rows])), H)


def _general_sum(G, A, B, x_out, y_out):
    """sum_mn G[m, n]*exp(i*(A[m, n]*x_out[p] + B[m, n]*y_out[q]))."""

    nz = G != 0
    g, a, b = G[nz], A[nz], B[nz]
    if finufft is not None:
        X, Y = np.meshgrid(x_out, y_out)
        u = finufft.nufft2d3(a, b, g, X.ravel(), Y.ravel(), isign=1, eps=1E-12)
        return u.reshape(X.shape)

    u = np.zeros((len(y_out), len(x_out)), dtype=complex)
    chunk = max(1, _CHUNK_ELEMENTS//max(len(x_out), len(y_out)))
    for start in range(0, len(g), chunk):
        s = slice(start, start + chunk)
        E_y = np.exp(1j*np.outer(y_out, b[s]))*g[s]
        E_x = np.exp(1j*np.outer(a[s], x_out))
        u += np.dot(E_y, E_x)

    return u


def _exact(spectra, kx, ky, k, r0, R, x_out, y_out, remove_carrier):
    KX, KY = np.meshgrid(kx, ky)
    KZ = np.sqrt(np.maximum(k**2 - KX**2 - KY**2, 0.0))
    K_hat = np.einsum("ji,jmn->imn", R, np.array([KX, KY, KZ]))  # R^T k
    valid = (KX**2 + KY**2 < k**2) & (K_hat[2] > _TOL*k)

    c = _carrier(spectra, KX, KY, KZ, valid, R) if remove_carrier else np.zeros(2)
    A, B = K_hat[0] - c[0], K_hat[1] - c[1]
    phase = np.exp(1j*(KX*r0[0] + KY*r0[1] + KZ*r0[2]))
    norm = 1.0/KX.size

    y_separable = abs(R[0, 1]) < _TOL and abs(R[2, 1]) < _TOL  # k^_y = R[1, 1]*k_y
    x_separable = abs(R[1, 0]) < _TOL and abs(R[2, 0]) < _TOL  # k^_x = R[0, 0]*k_x

    results = []
    for F in spectra:
        G = np.where(valid, F*phase*norm, 0.0)
        if y_separable:
            u = _separable_sum(G, A, B[:, 0], x_out, y_out)
        elif x_separable:
            u = _separable_sum(G.T, B.T, A[0, :], y_out, x_out).T
        else:
            u = _general_sum(G, A, B, x_out, y_out)
        results.append(u)

    return results, c


def _rotate(F, kx, ky, k, R, kx_hat, ky_hat, kind="cubic", jacobian=True):
    """Interpolate the spectrum F(kx, ky) at k = R k^ for the uniform k^ grid
    and apply the Jacobian. Returns the rotated spectrum (as a density), the
    source wave vectors at the samples and the validity mask.
    """

    if kind not in ("linear", "cubic"):
        raise ValueError('kind must be "linear" or "cubic", not {!r}'.format(kind))
    order = {"linear": 1, "cubic": 3}[kind]
    KXh, KYh = np.meshgrid(kx_hat, ky_hat)
    KZh2 = k**2 - KXh**2 - KYh**2
    KZh = np.sqrt(np.maximum(KZh2, 0.0))
    KX, KY, KZ = np.einsum("ij,jmn->imn", R, np.array([KXh, KYh, KZh]))
    valid = (KZh2 > 0) & (KZ > _TOL*k)

    # fractional indices into the source grid:
    i_x = (KX - kx[0])/(kx[1] - kx[0])
    i_y = (KY - ky[0])/(ky[1] - ky[0])
    # snap round-off just outside the grid back onto its edge nodes:
    i_x = np.where((i_x > -1E-9) & (i_x < len(kx) - 1 + 1E-9), np.clip(i_x, 0, len(kx) - 1), i_x)
    i_y = np.where((i_y > -1E-9) & (i_y < len(ky) - 1 + 1E-9), np.clip(i_y, 0, len(ky) - 1), i_y)
    coords = np.array([i_y, i_x])
    F_int = (map_coordinates(np.real(F), coords, order=order, mode="constant", cval=0.0)
             + 1j*map_coordinates(np.imag(F), coords, order=order, mode="constant", cval=0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(valid, KZ/KZh, 0.0) if jacobian else valid.astype(float)

    return F_int*J, (KX, KY, KZ), valid


def _rotation_method(spectra, kx, ky, dx, dy, k, r0, R, x_out, y_out, remove_carrier, kind):
    for name, coords in (("x_out", x_out), ("y_out", y_out)):
        if len(coords) < 2 or not np.allclose(np.diff(coords), coords[1] - coords[0], rtol=1E-6):
            raise ValueError('method="rotation" needs equidistant {}; use method="exact" '
                             'for arbitrary output points'.format(name))
    dx_out, dy_out = get_dx(x_out), get_dx(y_out)

    # the uniform k^ grid is always centered on the carrier, which a coarse
    # output grid could not resolve otherwise; without remove_carrier, the
    # carrier is multiplied back onto the result:
    KX, KY = np.meshgrid(kx, ky)
    KZ = np.sqrt(np.maximum(k**2 - KX**2 - KY**2, 0.0))
    valid = (KX**2 + KY**2 < k**2) & (np.einsum("j,jmn->mn", R[:, 2], np.array([KX, KY, KZ])) > 0)
    c = _carrier(spectra, KX, KY, KZ, valid, R)

    kx_rel = frequencies(np.arange(len(x_out))*dx_out)
    ky_rel = frequencies(np.arange(len(y_out))*dy_out)
    kx_hat, ky_hat = kx_rel + c[0], ky_rel + c[1]

    centered = _is_centered(x_out) and _is_centered(y_out)
    results = []
    for F in spectra:
        F_rot, (KX, KY, KZ), valid = _rotate(F, kx, ky, k, R, kx_hat, ky_hat, kind)
        F_hat = F_rot*np.exp(1j*(KX*r0[0] + KY*r0[1] + KZ*r0[2]))*valid*(dx*dy)/(dx_out*dy_out)
        if centered:
            u = inv_FT(F_hat)
        else:
            u = zoom_inv_FT(F_hat, kx_rel, ky_rel, x_out, y_out)
        if not remove_carrier:
            u = u*np.exp(1j*(c[0]*x_out[None, :] + c[1]*y_out[:, None]))
        results.append(u)

    return results, (c if remove_carrier else np.zeros(2))


def _tilted(fields, x, y, z0, wavelength, n, R, angles, method, x_out, y_out,
            remove_carrier, pad, kind, k_carrier_in, reverse, func_name, spectra_hook=None):
    R = _rotation(R, angles)
    r0 = _origin(z0)
    if reverse:
        # the source plane is tilted by R, the target plane is the plane
        # z = z0 of the unrotated frame; in the source frame its origin is
        # R^T r0 and its axes are R^T:
        r0, R = np.dot(R.T, r0), R.T.copy()

    x, y = _lin(x, y)
    x_out = x.copy() if x_out is None else np.atleast_1d(np.asarray(x_out, dtype=float))
    y_out = y.copy() if y_out is None else np.atleast_1d(np.asarray(y_out, dtype=float))
    _check_side(r0, R, x_out, y_out, func_name)

    k = np.real(wavenumber(wavelength, n))
    spectra, kx, ky, dx, dy = _source_spectra(fields, x, y, pad, k_carrier_in)
    if spectra_hook is not None:
        spectra = spectra_hook(spectra, kx, ky, k)

    if method is None:
        separable = ((abs(R[0, 1]) < _TOL and abs(R[2, 1]) < _TOL)
                     or (abs(R[1, 0]) < _TOL and abs(R[2, 0]) < _TOL))
        N = max(len(x), len(y), len(x_out), len(y_out))
        method = "exact" if N <= 256 and (separable or finufft is not None or N <= 64) else "rotation"

    if method == "exact":
        results, c = _exact(spectra, kx, ky, k, r0, R, x_out, y_out, remove_carrier)
    elif method == "rotation":
        results, c = _rotation_method(spectra, kx, ky, dx, dy, k, r0, R, x_out, y_out,
                                      remove_carrier, kind)
    else:
        raise ValueError('method must be "exact" or "rotation", not {!r}'.format(method))

    return results, x_out, y_out, c, R


def tilted_plane(field, x, y, z0, wavelength, n=1.0, R=None, angles=None, method=None,
                 x_out=None, y_out=None, remove_carrier=True, pad=2, kind="cubic",
                 k_carrier_in=(0.0, 0.0), reverse=False):
    """Propagate a scalar field from the plane z = 0 onto a tilted plane.

    See the module docstring for the geometry, the methods and the carrier.

    Parameters
    ----------
    field : array
        Field of shape (Ny, Nx) in the source plane.
    x, y : arrays
        Linear or meshed coordinates of the source plane, centered (x = 0 at
        index N//2).
    z0 : number or array
        Distance of the target plane's origin (0, 0, z0), or its position
        (x0, y0, z0).
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index (real) of the medium.
    R : array, optional
        Rotation matrix whose columns are the target axes x^, y^ and the
        normal, e.g. geometry.R_y(theta). Default: no rotation.
    angles : tuple, optional
        (theta_x, theta_y) instead of R, see rotation_from_angles().
    method : str, optional
        "exact" or "rotation". By default "exact" for N <= 256 (input and
        output) if the rotation is about the x or y axis or finufft is
        installed (or for N <= 64), and "rotation" otherwise. "exact" takes
        about a second for N = 256 (with pad=2), "rotation" a few
        hundredths; with cubic interpolation they agree to about 1E-6 if
        the field fits into the output window.
    x_out, y_out : arrays, optional
        Coordinates in the target plane (linear). Default: the source
        coordinates (same N and dx). "exact" accepts any points, "rotation"
        needs equidistant ones.
    remove_carrier : boolean, optional
        If True (default), remove the mean phase tilt, see the module
        docstring.
    pad : number, optional
        Zero padding factor of the source field. Padding refines the sampling
        of the source spectrum: it improves the interpolation of "rotation"
        and pushes the periodic copies of the source (both methods) away.
    kind : str, optional
        Interpolation of the source spectrum for "rotation": "cubic"
        (default, error about 1E-6 to 1E-8 relative to the peak) or "linear"
        (about 1E-3 without padding, decreasing with pad**2).
    k_carrier_in : tuple, optional
        Carrier (k_x, k_y) removed from the input field, which is
        field*exp(i*(k_x*x + k_y*y)) in truth.
    reverse : boolean, optional
        If True, the source plane is the tilted one: the field is given on a
        plane rotated by R (or angles), and the target is the plane z = z0
        (or through the point z0) of the unrotated frame, parallel to its
        x-y plane, with x_out, y_out along its x and y. This is the
        transformation with R^-1.

    Returns
    -------
    u_t : array
        Field on the target plane, shape (len(y_out), len(x_out)), with the
        carrier removed if remove_carrier.
    x_t, y_t : arrays
        Linear coordinates in the target plane.
    k_carrier : array
        The removed carrier (k^_x, k^_y), zeros if remove_carrier is False.
    """

    results, x_t, y_t, c, _ = _tilted([field], x, y, z0, wavelength, n, R, angles, method,
                                      x_out, y_out, remove_carrier, pad, kind,
                                      k_carrier_in, reverse, "tilted_plane")

    return results[0], x_t, y_t, c


def vectorial_tilted_plane(field, x, y, z0, wavelength, n=1.0, R=None, angles=None,
                           method=None, x_out=None, y_out=None, remove_carrier=True, pad=2,
                           kind="cubic", k_carrier_in=(0.0, 0.0), reverse=False):
    """Propagate a vector field from the plane z = 0 onto a tilted plane.

    E_z of the source follows from div(E) = 0 (as in
    propagators.longitudinal_component(), a given E_z is ignored). The three
    Cartesian components are propagated onto the target plane like scalar
    fields, and are then expressed in the plane's own axes: the local
    components are Ê = R^T E, i.e. Ê_x^ = E.x^, Ê_y^ = E.y^ and Ê_n = E.n^
    (the component normal to the plane).

    Parameters
    ----------
    field : array
        Vector field [E_x, E_y] or [E_x, E_y, E_z] of shape (Ny, Nx, 2) or
        (Ny, Nx, 3), components along the source axes.
    x, y, z0, wavelength, n, R, angles, method, x_out, y_out, remove_carrier, pad, kind, k_carrier_in, reverse
        See tilted_plane().

    Returns
    -------
    E_t : array
        Field [Ê_x^, Ê_y^, Ê_n] on the target plane, shape
        (len(y_out), len(x_out), 3).
    x_t, y_t : arrays
    k_carrier : array
    """

    field = np.asarray(field)
    if field.ndim != 3 or field.shape[2] not in (2, 3):
        raise ValueError("vectorial_tilted_plane() needs a vector field of shape (Ny, Nx, 2) "
                         "or (Ny, Nx, 3), got shape {}".format(field.shape))

    def add_E_z(spectra, kx, ky, k):
        # E_z from div(E) = 0 for every plane wave (as in
        # propagators.longitudinal_component(), with 1/k_z bounded for grazing waves):
        F_x, F_y = spectra
        KX, KY = np.meshgrid(kx, ky)
        KZ = np.sqrt(np.maximum(k**2 - KX**2 - KY**2, 0.0))
        KZ = np.maximum(KZ, np.sqrt(k*min(kx[1] - kx[0], ky[1] - ky[0])))
        F_z = np.where(KX**2 + KY**2 < k**2, -(KX*F_x + KY*F_y)/KZ, 0.0)
        return [F_x, F_y, F_z]

    results, x_t, y_t, c, R_used = _tilted([field[:, :, 0], field[:, :, 1]], x, y, z0,
                                           wavelength, n, R, angles, method, x_out, y_out,
                                           remove_carrier, pad, kind, k_carrier_in, reverse,
                                           "vectorial_tilted_plane", spectra_hook=add_E_z)
    E = np.stack(results, axis=-1)
    E_t = np.einsum("ji,...j->...i", R_used, E)  # local components R^T E

    return E_t, x_t, y_t, c


def rotate_angular_spectrum(A, kx, ky, k, R, kx_out=None, ky_out=None, kind="cubic",
                            jacobian=True):
    """Express an angular spectrum in a rotated frame.

    A(k_x, k_y) are the amplitudes of the plane waves exp(i*k.r) (k_z > 0)
    of a field, sampled on the uniform grid kx, ky. The same field, written in
    the frame whose axes are the columns of R (same origin), has the angular
    spectrum ::

        Â(k^_x, k^_y) = A(k_x, k_y)*k_z/k^_z,  k = R k^,

    k_z/k^_z being the Jacobian ``|d(k_x, k_y)/d(k^_x, k^_y)|``. A is
    interpolated at k = R k^ ("linear" or "cubic", outside of the grid it is
    zero). Waves with k^_z <= 0 or k_z <= 0 are set to zero.

    A is treated as a spectral density. For discrete spectra (FT() samples)
    multiply the result by (dk^_x*dk^_y)/(dk_x*dk_y) if the frequency
    spacings differ.

    Parameters
    ----------
    A : array
        Spectrum of shape (len(ky), len(kx)).
    kx, ky : arrays
        Linear, equidistant wavenumbers of A.
    k : number
        Wavenumber in the medium.
    R : array
        Rotation matrix.
    kx_out, ky_out : arrays, optional
        Wavenumbers k^_x, k^_y of the output grid. Default: kx, ky.
    kind : str, optional
        "cubic" (default) or "linear" interpolation.
    jacobian : boolean, optional
        Apply the Jacobian (default). jacobian=False is wrong and only meant
        to demonstrate the error it causes.

    Returns
    -------
    A_rot : array
        Spectrum of shape (len(ky_out), len(kx_out)).
    """

    kx, ky = np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    kx_out = kx if kx_out is None else np.asarray(kx_out, dtype=float)
    ky_out = ky if ky_out is None else np.asarray(ky_out, dtype=float)
    A_rot, _, valid = _rotate(np.asarray(A, dtype=complex), kx, ky, k,
                              np.asarray(R, dtype=float), kx_out, ky_out, kind, jacobian)

    return np.where(valid, A_rot, 0.0)


def power_flux(field, x, y, wavelength, n=1.0, k_carrier=(0.0, 0.0)):
    """Power flux through a plane, computed from the angular spectrum.

    P = sum over the propagating plane waves of ``|A|²`` k_z/k (in the units of
    the integral of ``|u|²``; for a paraxial field P equals the integral of ``|u|²``
    over the plane). Unlike the integral of ``|u|²``, P is the same on every
    plane the field crosses, tilted or not.

    Parameters
    ----------
    field : array
    x, y : arrays
        Linear or meshed, equidistant coordinates.
    wavelength : number
    n : number, optional
    k_carrier : tuple, optional
        Carrier removed from the field (see tilted_plane()).

    Returns
    -------
    P : number
    """

    x, y = _lin(x, y)
    k = np.real(wavenumber(wavelength, n))
    field = np.asarray(field, dtype=complex)
    F = FT(field)
    kx = frequencies(x) + k_carrier[0]
    ky = frequencies(y) + k_carrier[1]
    KX, KY = np.meshgrid(kx, ky)
    KZ = np.sqrt(np.maximum(k**2 - KX**2 - KY**2, 0.0))

    return np.sum(np.abs(F)**2*KZ/k)*abs(get_dx(x)*get_dx(y))/field.size


def tilted_beam_widths(field, x, y, z0, wavelength, n=1.0, iterate=True, window=3.0,
                       **kwargs):
    """ISO 11146 beam widths of a beam on an inclined plane (e.g. a tilted
    detector).

    The field is propagated with tilted_plane() and beam_quality.beam_widths()
    is evaluated with the intensity ``|u_t|²`` in the plane's own coordinates.
    For a collimated beam on a plane tilted by theta about y, d_x grows as
    d/cos(theta).

    Parameters
    ----------
    field, x, y, z0, wavelength, n
        See tilted_plane().
    iterate, window : optional
        See beam_quality.beam_widths().
    kwargs
        Further arguments of tilted_plane() (R, angles, x_out, ...).

    Returns
    -------
    widths : beam_quality.BeamWidths
        Widths, azimuth and centroid in the plane's coordinates.
    u_t, x_t, y_t
        Field and coordinates on the plane, see tilted_plane().
    """

    u_t, x_t, y_t, _ = tilted_plane(field, x, y, z0, wavelength, n=n, **kwargs)
    widths = beam_widths(np.abs(u_t)**2, x_t, y_t, iterate=iterate, window=window)

    return widths, u_t, x_t, y_t
