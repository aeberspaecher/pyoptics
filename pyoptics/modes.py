#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Mode analysis of beams and step-index fibre modes.

Hermite- and Laguerre-Gaussian decomposition
--------------------------------------------
decompose_hg() and decompose_lg() expand a sampled field in the paraxial
modes of beams.gauss_hermite() and beams.gauss_laguerre(), ::

    E(x, y) = sum_j c_j u_j(x, y; w0, z),

for a basis with waist radius w0 whose waist lies a distance z in front of
the plane of the field (z > 0: the basis diverges in the plane, R(z) > 0).
The modes include the Gouy phase, so the coefficients do not change when the
field and the basis are propagated together. The modes are normalized
numerically on the grid, ``|c_j|**2/P`` is the fraction of the power P in mode
j, and the expansion is truncated at the mode order N = l + m (HG) or
N = 2p + ``|l|`` (LG) <= max_order.

The basis (w0, z and the centre x0, y0) can be given or fitted
(fit_basis()). The choice matters: any w0 gives a complete basis, but a basis
that is too wide or too narrow needs many modes for the same field. The fit
maximizes the power that is captured by the truncated basis.

M² from the mode content
------------------------
The Hermite functions are the eigenstates of the harmonic oscillator. With
xi = sqrt(2)*x/w0 the position xi and the (dimensionless) direction p are
xi = (a + a^+)/sqrt(2) and p = i*(a^+ - a)/sqrt(2), so the second moments of
any coherent superposition follow from <a>, <a^+ a> and <a^2>, which are
sums over the HG coefficients. Then ::

    M²_x = 2*sqrt(Var(xi)*Var(p) - Cov(xi, p)**2),

which is the M² of ISO 11146 (second moments, beam_quality.fit_caustic())
along x, and analogously along y. For a pure HG_lm mode, M²_x = 2l + 1,
M²_y = 2m + 1. The result is exact for the truncated expansion; if the
captured power is not close to 1, the field has content beyond max_order.

Step-index fibres
-----------------
In the weakly guiding approximation (n_core - n_clad << n_core), the modes
of a step-index fibre with core radius a are linearly polarized LP_lm modes
with the field ::

    E(r, phi) = J_l(U*r/a)/J_l(U) * cos(l*phi)   for r <= a,
    E(r, phi) = K_l(W*r/a)/K_l(W) * cos(l*phi)   for r >  a

(or sin(l*phi), or exp(+-i*l*phi)). U and W follow from the characteristic
equation ::

    U*J_{l-1}(U)/J_l(U) = -W*K_{l-1}(W)/K_l(W),   U**2 + W**2 = V**2,

with the V number V = 2*pi*a/wavelength*NA and NA = sqrt(n_core**2 -
n_clad**2). The normalized propagation constant b = W**2/V**2 gives
n_eff**2 = n_clad**2 + b*(n_core**2 - n_clad**2). LP_lm is cut off at V =
U_c, the m-th zero of J_{l-1} (for l = 0 counting U_c = 0 as the first
zero): LP01 has no cutoff, LP11 is guided above V = 2.405. LP_0m modes are
twofold degenerate (polarization), LP_lm with l >= 1 fourfold (cos/sin and
polarization). Vectorial effects (splitting into HE, EH, TE, TM modes) are
not modelled.

Marcuse's Gaussian approximation of LP01 has the mode field radius ::

    w/a = 0.65 + 1.619/V**1.5 + 2.879/V**6

(D. Marcuse, Bell Syst. Tech. J. 56, 703 (1977)), better than 1 % for 1.2 <
V < 2.4.

Coupling efficiency
-------------------
coupling_efficiency() is the power coupling efficiency of a field into a mode
(or any other field), the normalized overlap integral ::

    eta = |<u|E>|**2/(<u|u>*<E|E>).

It generalizes abcd.coupling_efficiency(), which only handles coaxial
Gaussian beams. coupling_vs_offset(), coupling_vs_tilt() and
coupling_vs_defocus() sweep a misalignment of the field relative to the
mode.

All functions use the pyoptics convention exp(i(k.r - omega*t)), and the
wavelength is the wavelength in the medium in which the beams are given (the
beams module uses k = 2*pi/wavelength).
"""

from collections import namedtuple
from math import pi

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import eval_hermite, jn_zeros, jv, kve

from .beams import gauss_hermite, gauss_laguerre
from .utils import center_of_mass, ensure_meshgrid, freq_grid, get_dx, scalar_product, wavenumber
from .fft import FT, inv_FT


# Gaussian basis:
BasisParameters = namedtuple("BasisParameters", ["w0", "z", "x0", "y0", "w", "R"])
BasisParameters.__doc__ = """Parameters of a Gaussian mode basis.

w0: waist radius; z: distance of the plane from the waist (z > 0 behind the
waist, diverging); x0, y0: centre; w, R: beam radius and wavefront radius of
curvature in the plane (R = inf at the waist).
"""


def waist_from_plane(w, R, wavelength):
    """Waist of a Gaussian beam from its radius and wavefront radius in a plane.

    Uses 1/q = 1/R - i*wavelength/(pi*w**2), q = z + i*z_R.

    Parameters
    ----------
    w : number
        Beam radius (1/e**2 intensity) in the plane.
    R : number
        Wavefront radius of curvature in the plane (np.inf: flat, R > 0:
        diverging).
    wavelength : number
        Wavelength in the medium.

    Returns
    -------
    w0 : number
        Waist radius.
    z : number
        Distance of the plane from the waist (z > 0: behind the waist).
    """

    inv_q = (0.0 if np.isinf(R) else 1.0/R) - 1j*wavelength/(pi*w**2)
    q = 1.0/inv_q
    z, z_R = q.real, q.imag

    return np.sqrt(wavelength*z_R/pi), z


def plane_from_waist(w0, z, wavelength):
    """Beam radius and wavefront radius a distance z from a waist.

    Parameters
    ----------
    w0 : number
        Waist radius.
    z : number
        Distance from the waist.
    wavelength : number
        Wavelength in the medium.

    Returns
    -------
    w : number
        Beam radius (1/e**2 intensity).
    R : number
        Wavefront radius of curvature, np.inf at the waist.
    """

    z_R = pi*w0**2/wavelength
    w = w0*np.sqrt(1 + (z/z_R)**2)
    R = np.inf if z == 0 else z*(1 + (z_R/z)**2)

    return w, R


def _basis_parameters(w0, z, x0, y0, wavelength):
    w, R = plane_from_waist(w0, z, wavelength)

    return BasisParameters(w0, z, x0, y0, w, R)


def hg_mode(l, m, x, y, w0, wavelength, z=0.0, x0=0.0, y0=0.0):
    """Hermite-Gaussian mode HG_lm of `beams.gauss_hermite`.

    Parameters
    ----------
    l, m : int
        Mode indices in x and y.
    x, y : arrays
        Linear or meshed coordinates.
    w0 : number
        Waist radius.
    wavelength : number
        Wavelength in the medium.
    z : number, optional
        Distance from the waist.
    x0, y0 : numbers, optional
        Centre of the mode.

    Returns
    -------
    field : 2d array
        Mode field (including the Gouy phase).
    """

    X, Y = ensure_meshgrid(x, y)
    z_r = pi*w0**2/wavelength

    return gauss_hermite(l, m, X - x0, Y - y0, z, w0, z_r, wavelength)


def lg_mode(p, l, x, y, w0, wavelength, z=0.0, x0=0.0, y0=0.0):
    """Laguerre-Gaussian mode LG_pl of `beams.gauss_laguerre`.

    l > 0 has the helical phase exp(+i*l*phi).

    Parameters
    ----------
    p, l : int
        Radial and azimuthal index.
    x, y : arrays
        Linear or meshed coordinates.
    w0 : number
        Waist radius.
    wavelength : number
        Wavelength in the medium.
    z : number, optional
        Distance from the waist.
    x0, y0 : numbers, optional
        Centre of the mode.

    Returns
    -------
    field : 2d array
        Mode field (including the Gouy phase).
    """

    X, Y = ensure_meshgrid(x, y)
    z_r = pi*w0**2/wavelength

    return gauss_laguerre(p, l, X - x0, Y - y0, z, w0, z_r, wavelength)


def hg_indices(max_order):
    """Indices (l, m) of all HG modes with l + m <= max_order, by order."""

    return [(l, N - l) for N in range(max_order + 1) for l in range(N, -1, -1)]


def lg_indices(max_order):
    """Indices (p, l) of all LG modes with 2p + ``|l|`` <= max_order, by order."""

    return [(p, l) for N in range(max_order + 1) for l in range(-N, N + 1, 2)
            for p in [(N - abs(l))//2]]


_FAMILIES = {"hg": (hg_mode, hg_indices), "lg": (lg_mode, lg_indices)}


def _normalized(u, x, y):
    return u/np.sqrt(scalar_product(u, u, x, y).real)


def _power(field, x, y):
    return scalar_product(field, field, x, y).real


def _initial_basis(field, x, y, wavelength):
    """Centre, radius and curvature from the moments of the field: w from the
    second moments, and 1/R from the intensity weighted mean of r*dphi/dr,
    ``sum((x*dphi/dx + y*dphi/dy)*|E|**2) = k/R*sum(r**2*|E|**2)``.
    """

    X, Y = ensure_meshgrid(x, y)
    I = np.abs(field)**2
    x0, y0 = center_of_mass(I, x, y)
    U, V = X - x0, Y - y0
    r2 = np.sum((U**2 + V**2)*I)/np.sum(I)
    w = np.sqrt(2*r2)  # <r**2> = w**2/2 for a TEM_00 beam

    dE_dy, dE_dx = np.gradient(field, y, x)
    # |E|**2*grad(phi) = Im(conj(E)*grad(E)):
    flux = np.sum(U*np.imag(np.conj(field)*dE_dx) + V*np.imag(np.conj(field)*dE_dy))
    k = wavenumber(wavelength)
    curvature = flux/(k*np.sum((U**2 + V**2)*I))

    return x0, y0, w, curvature


def fit_basis(field, x, y, wavelength, max_order=0, family="hg", method="overlap",
              x0=None, y0=None):
    """Fit the parameters of a Gaussian mode basis to a field.

    Parameters
    ----------
    field : array
    x, y : arrays
    wavelength : number
    max_order : int, optional
        Highest mode order of the truncated basis whose captured power is
        maximized (method "overlap"). 0 (default) fits the fundamental mode.
    family : str, optional
        "hg" or "lg".
    method : str, optional
        "overlap" (default) maximizes the power in the truncated basis,
        starting from "moments", which takes the radius from the second
        moments and the curvature from the phase gradient. "moments" alone
        is a good basis for near Gaussian beams; for multimode beams it gives
        a basis that is about sqrt(M²) too wide.
    x0, y0 : numbers, optional
        Centre of the basis. If omitted, fitted as well ("overlap"), or the
        intensity centroid ("moments").

    Returns
    -------
    basis : BasisParameters

    Notes
    -----
    The basis is coaxial: a tilted beam needs many modes (or remove the tilt
    first).
    """

    x0_c, y0_c, w, curvature = _initial_basis(field, x, y, wavelength)
    fit_centre = (x0 is None, y0 is None)
    x0 = x0_c if x0 is None else x0
    y0 = y0_c if y0 is None else y0

    if method == "overlap":
        # HG and LG modes up to the same order span the same space, so the
        # captured power does not depend on the family. HG modes separate,
        # u_lm(x, y) = h_l(x)*h_m(y), and the overlaps of all modes are one
        # matrix product with 1D modes (constant phases do not matter here)
        P = _power(field, x, y)
        k = wavenumber(wavelength)
        x_lin, y_lin = np.ravel(ensure_meshgrid(x, y)[0][0]), np.ravel(ensure_meshgrid(x, y)[1][:, 0])
        dx, dy = get_dx(x_lin), get_dx(y_lin)
        order = np.add.outer(np.arange(max_order + 1), np.arange(max_order + 1))

        def modes_1d(coord, w_fit, c_fit):
            xi = np.sqrt(2)*coord/w_fit
            h = np.array([eval_hermite(l, xi) for l in range(max_order + 1)])
            h = h*np.exp(-xi**2/2 + 1j*k*c_fit*coord**2/2)
            return h/np.sqrt(np.sum(np.abs(h)**2, axis=1, keepdims=True))

        def unpack(params):
            # radius relative to w, curvature 1/R in units of the far field
            # angle, centre relative to w (the centroid is not the centre of
            # the basis for an asymmetric mode mixture):
            w_fit = w*np.exp(params[0])
            c_fit = params[1]/w**2*2/k
            x_fit = x0 + w*params[2] if fit_centre[0] else x0
            y_fit = y0 + w*params[3] if fit_centre[1] else y0
            return w_fit, c_fit, x_fit, y_fit

        def lost_power(params):
            w_fit, c_fit, x_fit, y_fit = unpack(params)
            h_x, h_y = modes_1d(x_lin - x_fit, w_fit, c_fit), modes_1d(y_lin - y_fit, w_fit, c_fit)
            C = np.conj(h_y) @ field @ np.conj(h_x).T*np.sqrt(dx*dy)  # C[m, l]
            return 1.0 - np.sum(np.abs(C[order <= max_order])**2)/P

        # the second moment radius is too large by up to sqrt(M²) for a mode
        # mixture: try several start radii
        results = [minimize(lost_power, [np.log(scale), curvature*w**2*k/2, 0.0, 0.0],
                            method="Nelder-Mead",
                            options={"xatol": 1E-7, "fatol": 1E-13, "maxiter": 2000})
                   for scale in (1.0, 0.7, 0.5)]
        best = min(results, key=lambda result: result.fun)
        w_fit, c_fit, x0, y0 = unpack(best.x)
        w0, z = waist_from_plane(w_fit, np.inf if c_fit == 0 else 1.0/c_fit, wavelength)
    elif method == "moments":
        w0, z = waist_from_plane(w, np.inf if curvature == 0 else 1.0/curvature, wavelength)
    else:
        raise ValueError(f"Unknown method '{method}', use 'overlap' or 'moments'.")

    return _basis_parameters(w0, z, x0, y0, wavelength)


class ModeDecomposition:
    """Expansion of a field in Hermite- or Laguerre-Gaussian modes.

    Attributes
    ----------
    family : str
        "hg" or "lg".
    coefficients : dict
        Complex coefficients c_j of the numerically normalized modes, keyed by
        (l, m) for HG and (p, l) for LG modes.
    basis : BasisParameters
        Waist, distance from the waist and centre of the basis.
    wavelength : number
    power : number
        Power ``<E|E>`` of the decomposed field.
    reconstruction_error : number
        Relative rms error ``||E - sum_j c_j u_j||/||E||`` on the grid.
    """

    def __init__(self, family, coefficients, basis, wavelength, power, reconstruction_error):
        self.family = family
        self.coefficients = coefficients
        self.basis = basis
        self.wavelength = wavelength
        self.power = power
        self.reconstruction_error = reconstruction_error

    @property
    def power_fractions(self):
        """Fractions ``|c_j|**2/P`` of the power in each mode (dict)."""

        return {key: abs(c)**2/self.power for key, c in self.coefficients.items()}

    @property
    def captured_power(self):
        """Fraction of the power in the truncated basis."""

        return sum(self.power_fractions.values())

    @property
    def max_order(self):
        return max(_order(self.family, key) for key in self.coefficients)

    def order_fractions(self):
        """Power fractions summed per mode order N (array, index N)."""

        fractions = np.zeros(self.max_order + 1)
        for key, f in self.power_fractions.items():
            fractions[_order(self.family, key)] += f

        return fractions

    def mode(self, key, x, y):
        """The normalized basis mode of a given key.

        Parameters
        ----------
        key : tuple of int
            (l, m) for HG or (p, l) for LG modes.
        x, y : arrays
            Linear or meshed coordinates.

        Returns
        -------
        field : 2d array
            Mode, normalized to unit power on the grid.
        """

        mode_func = _FAMILIES[self.family][0]
        b = self.basis

        return _normalized(mode_func(*key, x, y, b.w0, self.wavelength, b.z, b.x0, b.y0), x, y)

    def reconstruct(self, x, y, keys=None):
        """Sum of the modes with their coefficients.

        Parameters
        ----------
        x, y : arrays
            Linear or meshed coordinates.
        keys : iterable of tuples, optional
            Modes to include. Defaults to all modes of the decomposition.

        Returns
        -------
        field : 2d array
            Reconstructed field.
        """

        keys = self.coefficients.keys() if keys is None else keys
        if self.family == "hg":
            return _hg_synthesis({key: self.coefficients[key] for key in keys}, x, y, self.basis,
                                 self.wavelength)
        X, _ = ensure_meshgrid(x, y)
        field = np.zeros(X.shape, dtype=complex)
        for key in keys:
            field += self.coefficients[key]*self.mode(key, x, y)

        return field

    def propagate(self, distance):
        """The same expansion a distance further along z (paraxial free
        propagation): the modes contain the Gouy phase, so only the basis
        changes. Use reconstruct() on the result to get the propagated field.
        """

        b = self.basis
        basis = _basis_parameters(b.w0, b.z + distance, b.x0, b.y0, self.wavelength)

        return ModeDecomposition(self.family, dict(self.coefficients), basis, self.wavelength,
                                 self.power, self.reconstruction_error)

    def to_hg(self, x, y):
        """HG decomposition of the same (truncated) field in the same basis.

        HG and LG modes up to the same order span the same space.

        Parameters
        ----------
        x, y : arrays
            Grid on which the LG expansion is reconstructed and decomposed again.

        Returns
        -------
        ModeDecomposition
            The decomposition itself if it already uses HG modes.
        """

        if self.family == "hg":
            return self

        return decompose_hg(self.reconstruct(x, y), x, y, self.basis.w0, self.wavelength,
                            self.basis.z, self.max_order, self.basis.x0, self.basis.y0)

    def M2(self, x=None, y=None):
        """Beam propagation ratios from the mode content.

        Second moments of the truncated expansion, see the module docstring. An
        LG decomposition is converted with `to_hg()`, which needs x and y.

        Parameters
        ----------
        x, y : arrays, optional
            Grid for the conversion of an LG decomposition.

        Returns
        -------
        M2_x, M2_y : numbers
            M² along x and y.

        Raises
        ------
        ValueError
            For an LG decomposition without x and y.
        """

        if self.family == "lg":
            if x is None or y is None:
                raise ValueError("An LG decomposition needs x, y for the conversion to HG modes.")
            return self.to_hg(x, y).M2()

        return m2_from_hg_coefficients(self.coefficients)


def _order(family, key):
    return key[0] + key[1] if family == "hg" else 2*key[0] + abs(key[1])


def m2_from_hg_coefficients(coefficients):
    """M²_x and M²_y of a coherent superposition of HG modes with coefficients
    {(l, m): c} (normalized modes, any common basis).
    """

    N = max(max(key) for key in coefficients) + 1
    c = np.zeros((N + 2, N + 2), dtype=complex)
    for (l, m), value in coefficients.items():
        c[l, m] = value
    c /= np.sqrt(np.sum(np.abs(c)**2))
    n = np.arange(N + 2)

    def m2(c):
        # c[l, :] for the oscillator along axis 0; a c_l = sqrt(l) c_{l-1}
        a = np.sum(np.conj(c[:-1])*np.sqrt(n[1:])[:, None]*c[1:])
        a2 = np.sum(np.conj(c[:-2])*np.sqrt(n[2:]*n[1:-1])[:, None]*c[2:])
        num = np.sum(n[:, None]*np.abs(c)**2)
        var_xi = a2.real + num + 0.5 - 2*a.real**2
        var_p = -a2.real + num + 0.5 - 2*a.imag**2
        cov = a2.imag - 2*a.real*a.imag
        return 2*np.sqrt(max(var_xi*var_p - cov**2, 0.0))

    return m2(c), m2(c.T)


def _hg_modes_1d(x, y, basis, wavelength, max_order):
    """The HG modes separate, u_lm = h_l(x)*h_m(y)*exp(i*theta_lm), with the
    1D modes h_l (numerically normalized, real amplitude, curvature phase) and
    the constant phase theta_lm = k*z - (l + m + 1)*psi of
    beams.gauss_hermite(). Returns h_x[l], h_y[m] and theta[m, l].
    """

    X, Y = ensure_meshgrid(x, y)
    x_lin, y_lin = X[0], Y[:, 0]
    k = wavenumber(wavelength)
    z_r = pi*basis.w0**2/wavelength
    psi = np.arctan2(basis.z, z_r)
    # same branch for the curvature as beams.gauss_hermite():
    inv_R = 1.0/(basis.z*(1.0 + (z_r/basis.z)**2)) if np.abs(basis.z) >= 1E-14 else 0.0

    def modes_1d(coord):
        xi = np.sqrt(2)*coord/basis.w
        h = np.array([eval_hermite(l, xi) for l in range(max_order + 1)])
        h = h*np.exp(-xi**2/2 + 1j*k*inv_R*coord**2/2)
        return h/np.sqrt(np.sum(np.abs(h)**2, axis=1, keepdims=True)*get_dx(coord))

    n = np.arange(max_order + 1)
    theta = k*basis.z - (np.add.outer(n, n) + 1)*psi

    return modes_1d(x_lin - basis.x0), modes_1d(y_lin - basis.y0), theta


def _hg_projection(field, x, y, basis, wavelength, max_order):
    """Overlaps with all normalized HG modes as one matrix product."""

    h_x, h_y, theta = _hg_modes_1d(x, y, basis, wavelength, max_order)
    X, Y = ensure_meshgrid(x, y)
    C = np.conj(h_y) @ field @ np.conj(h_x).T*get_dx(X[0])*get_dx(Y[:, 0])*np.exp(-1j*theta)

    return {(l, m): C[m, l] for l, m in hg_indices(max_order)}


def _hg_synthesis(coefficients, x, y, basis, wavelength):
    """sum_lm c_lm*u_lm as one matrix product."""

    max_order = max(l + m for l, m in coefficients)
    h_x, h_y, theta = _hg_modes_1d(x, y, basis, wavelength, max_order)
    C = np.zeros((max_order + 1, max_order + 1), dtype=complex)
    for (l, m), c in coefficients.items():
        C[m, l] = c

    return h_y.T @ (C*np.exp(1j*theta)) @ h_x


def _decompose(family, field, x, y, w0, wavelength, z, max_order, x0, y0, method,
               fit_kwargs=None):
    if w0 is None:
        basis = fit_basis(field, x, y, wavelength, max_order, family, x0=x0, y0=y0,
                          **(fit_kwargs or {}))
    else:
        basis = _basis_parameters(w0, z, 0.0 if x0 is None else x0, 0.0 if y0 is None else y0,
                                  wavelength)

    mode_func, indices = _FAMILIES[family]
    keys = indices(max_order)

    def mode(key):
        return _normalized(mode_func(*key, x, y, basis.w0, wavelength, basis.z, basis.x0, basis.y0),
                           x, y)

    if method == "projection" and family == "hg":
        coefficients = _hg_projection(field, x, y, basis, wavelength, max_order)
    elif method == "projection":
        coefficients = {key: scalar_product(mode(key), field, x, y) for key in keys}
    elif method == "lstsq":
        # the sampled modes are not exactly orthogonal on a coarse or small
        # grid: solve the normal equations with the Gram matrix
        modes = np.array([mode(key).ravel() for key in keys])
        c, *_ = np.linalg.lstsq(modes.T, np.ravel(field), rcond=None)
        coefficients = dict(zip(keys, c))
    else:
        raise ValueError(f"Unknown method '{method}', use 'projection' or 'lstsq'.")

    P = _power(field, x, y)
    decomposition = ModeDecomposition(family, coefficients, basis, wavelength, P, None)
    residual = field - decomposition.reconstruct(x, y)
    decomposition.reconstruction_error = np.sqrt(_power(residual, x, y)/P)

    return decomposition


def decompose_hg(field, x, y, w0=None, wavelength=1.0, z=0.0, max_order=10, x0=None, y0=None,
                 method="projection", fit_kwargs=None):
    """Decompose a field into Hermite-Gaussian modes HG_lm, l + m <= max_order.

    Parameters
    ----------
    field : array
    x, y : arrays
        Linear or meshed coordinates.
    w0 : number, optional
        Waist radius of the basis. If omitted, the basis (w0, z, x0, y0) is
        fitted with fit_basis().
    wavelength : number
        Wavelength in the medium.
    z : number, optional
        Distance of the plane from the waist of the basis (z > 0: diverging
        basis). Use waist_from_plane() to get w0, z from a beam radius and a
        radius of curvature in the plane.
    max_order : int, optional
    x0, y0 : numbers, optional
        Centre of the basis (default 0, or the centroid if the basis is
        fitted).
    method : str, optional
        "projection" (default): overlap integrals with the numerically
        normalized modes, one mode at a time. "lstsq": least squares fit,
        accounts for the residual non-orthogonality of the sampled modes, but
        holds all modes in memory.
    fit_kwargs : dict, optional
        Keyword arguments for fit_basis() (e.g. method="moments").

    Returns
    -------
    decomposition : ModeDecomposition
    """

    return _decompose("hg", field, x, y, w0, wavelength, z, max_order, x0, y0, method, fit_kwargs)


def decompose_lg(field, x, y, w0=None, wavelength=1.0, z=0.0, max_order=10, x0=None, y0=None,
                 method="projection", fit_kwargs=None):
    """Decompose a field into Laguerre-Gaussian modes LG_pl, 2p + ``|l|`` <= max_order.

    The coefficients are keyed by (p, l).

    Parameters
    ----------
    field : array
    x, y : arrays
        Linear or meshed coordinates.
    w0 : number, optional
        Waist radius of the basis. If omitted, the basis is fitted with
        `fit_basis()`.
    wavelength : number
        Wavelength in the medium.
    z : number, optional
        Distance of the plane from the waist of the basis.
    max_order : int, optional
    x0, y0 : numbers, optional
        Centre of the basis.
    method : {"projection", "lstsq"}, optional
        See `decompose_hg()`.
    fit_kwargs : dict, optional
        Keyword arguments for `fit_basis()`.

    Returns
    -------
    decomposition : ModeDecomposition
    """

    return _decompose("lg", field, x, y, w0, wavelength, z, max_order, x0, y0, method, fit_kwargs)


# step-index fibres:
def numerical_aperture(n_core, n_clad):
    """Numerical aperture of a step-index fibre.

    Parameters
    ----------
    n_core, n_clad : numbers
        Refractive indices of core and cladding.

    Returns
    -------
    NA : number
        sqrt(n_core**2 - n_clad**2).
    """

    return np.sqrt(n_core**2 - n_clad**2)


def v_number(core_radius, n_core, n_clad, wavelength):
    """V number of a step-index fibre.

    Parameters
    ----------
    core_radius : number
        Core radius a.
    n_core, n_clad : numbers
        Refractive indices of core and cladding.
    wavelength : number
        Vacuum wavelength.

    Returns
    -------
    V : number
        2*pi*a/wavelength*NA.
    """

    return 2*pi*core_radius/wavelength*numerical_aperture(n_core, n_clad)


def lp_cutoff(l, m):
    """Cutoff V number of the LP_lm mode.

    Parameters
    ----------
    l : int
        Azimuthal index.
    m : int
        Radial index, m >= 1.

    Returns
    -------
    V_c : number
        The m-th zero of J_{l-1}, counting U = 0 as the first zero for l = 0.
    """

    if l == 0:
        return 0.0 if m == 1 else jn_zeros(1, m - 1)[-1]

    return jn_zeros(l - 1, m)[-1]


class LPMode(namedtuple("LPMode", ["l", "m", "U", "W", "V", "b", "n_eff"])):
    """Linearly polarized mode LP_lm of a weakly guiding step-index fibre.

    l, m: azimuthal and radial index; U, W: normalized transverse wavenumbers
    in core and cladding; V: V number; b = W**2/V**2: normalized propagation
    constant; n_eff: effective index (None if the indices are unknown).
    """

    @property
    def degeneracy(self):
        """Number of degenerate modes: 2 for l = 0, 4 for l >= 1."""

        return 2 if self.l == 0 else 4

    def radial(self, r, core_radius):
        """Radial profile of the mode.

        Parameters
        ----------
        r : number or array
            Distance from the fibre axis.
        core_radius : number
            Core radius a.

        Returns
        -------
        profile : array
            J_l in the core, K_l in the cladding, 1 at the core-cladding interface.
        """

        rho = np.abs(np.asarray(r, dtype=float))/core_radius
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            outside = kve(self.l, self.W*rho)/kve(self.l, self.W)*np.exp(-self.W*(rho - 1))
        return np.where(rho <= 1, jv(self.l, self.U*rho)/jv(self.l, self.U), np.nan_to_num(outside))

    def field(self, x, y, core_radius, orientation="cos", x0=0.0, y0=0.0):
        """Mode field on a grid, normalized to unit power.

        Parameters
        ----------
        x, y : arrays
            Linear or meshed coordinates.
        core_radius : number
            Core radius.
        orientation : {"cos", "sin", "+", "-"}, optional
            Azimuthal dependence, see `lp_mode_field()`.
        x0, y0 : numbers, optional
            Position of the fibre axis.

        Returns
        -------
        field : 2d array
        """

        return lp_mode_field(self, x, y, core_radius, orientation, x0, y0)


def _characteristic(W, l, V):
    # U*J_{l-1}(U)*K_l(W) + W*K_{l-1}(W)*J_l(U) = 0 without poles, divided by
    # K_l(W) (scaled Bessel functions cancel the exponentials). W is the
    # variable: near cutoff, W goes to zero much faster than V - U (for l = 0
    # exponentially), and U would not be resolved in floating point.
    U = np.sqrt(V**2 - W**2)
    return U*jv(l - 1, U) + W*kve(l - 1, W)/kve(l, W)*jv(l, U)


def lp_modes(V=None, core_radius=None, n_core=None, n_clad=None, wavelength=None, l_max=None,
             samples_per_unit=200):
    """Guided LP modes of a weakly guiding step-index fibre.

    Parameters
    ----------
    V : number, optional
        V number. If omitted, computed from core_radius, n_core, n_clad and
        the vacuum wavelength, which also give n_eff.
    core_radius, n_core, n_clad, wavelength : numbers, optional
    l_max : int, optional
        Largest azimuthal index (default: all guided modes).
    samples_per_unit : int, optional
        Sampling density of U for bracketing the roots.

    Returns
    -------
    modes : list of LPMode
        Sorted by decreasing b (n_eff). Modes extremely close to their
        cutoff (W below about 1E-300, e.g. LP01 for V < 0.04) are not found.
    """

    if V is None:
        V = v_number(core_radius, n_core, n_clad, wavelength)
    have_indices = n_core is not None and n_clad is not None

    # samples of W: uniform in U from U = 0 to close to V, then down to tiny
    # W on a logarithmic scale for the modes close to cutoff
    U_samples = np.linspace(V*1E-4, V*(1 - 1E-6), max(int(V*samples_per_unit), 400))
    W_uniform = np.sqrt(V**2 - U_samples**2)
    W_samples = np.concatenate([W_uniform, np.logspace(np.log10(W_uniform[-1]), -300, 600)[1:]])

    modes = []
    l = 0
    while lp_cutoff(l, 1) < V and (l_max is None or l <= l_max):
        with np.errstate(over="ignore", invalid="ignore"):
            f = _characteristic(W_samples, l, V)
        # K_l(W) overflows for tiny W and l >= 2 (these modes need no tail):
        W_l, f = W_samples[np.isfinite(f)], f[np.isfinite(f)]
        m = 0
        for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
            if W_l[i] > 1E-6*V:
                W = brentq(_characteristic, W_l[i + 1], W_l[i], args=(l, V), xtol=1E-15, rtol=1E-14)
            else:
                log_W = brentq(lambda t: _characteristic(np.exp(t), l, V), np.log(W_l[i + 1]),
                               np.log(W_l[i]), xtol=1E-12)
                W = np.exp(log_W)
            m += 1
            U = np.sqrt(V**2 - W**2)
            b = (W/V)**2
            n_eff = np.sqrt(n_clad**2 + b*(n_core**2 - n_clad**2)) if have_indices else None
            modes.append(LPMode(l, m, U, W, V, b, n_eff))
        l += 1

    return sorted(modes, key=lambda mode: -mode.b)


def lp_mode(l, m, V=None, **kwargs):
    """The single guided mode LP_lm.

    Parameters
    ----------
    l, m : int
        Mode indices.
    V : number, optional
        V number.
    **kwargs
        Further arguments of `lp_modes()`, e.g. the fibre parameters instead of
        V.

    Returns
    -------
    LPMode

    Raises
    ------
    ValueError
        If LP_lm is not guided.
    """

    for mode in lp_modes(V, l_max=l, **kwargs):
        if mode.l == l and mode.m == m:
            return mode
    raise ValueError(f"LP{l}{m} is not guided (cutoff at V = {lp_cutoff(l, m):.4f}).")


def lp_mode_field(mode, x, y, core_radius, orientation="cos", x0=0.0, y0=0.0):
    """Field of an LP mode on a grid, normalized numerically to unit power.

    Parameters
    ----------
    mode : LPMode
    x, y : arrays
    core_radius : number
    orientation : str, optional
        "cos" (default) or "sin" for the real modes cos(l*phi), sin(l*phi),
        "+" or "-" for the helical modes exp(+-i*l*phi). Ignored for l = 0.
    x0, y0 : numbers, optional
        Position of the fibre axis.

    Returns
    -------
    field : array
    """

    X, Y = ensure_meshgrid(x, y)
    U, V = X - x0, Y - y0
    phi = np.arctan2(V, U)
    azimuthal = {"cos": np.cos(mode.l*phi), "sin": np.sin(mode.l*phi),
                 "+": np.exp(1j*mode.l*phi), "-": np.exp(-1j*mode.l*phi)}[orientation]
    field = mode.radial(np.sqrt(U**2 + V**2), core_radius)*azimuthal

    return _normalized(field, x, y)


def marcuse_mode_field_radius(V, core_radius):
    """Mode field radius of LP01 after Marcuse.

    w/a = 0.65 + 1.619/V**1.5 + 2.879/V**6, the 1/e**2 intensity radius of the
    best matching Gaussian, better than 1 % for 1.2 < V < 2.4. The mode field
    diameter is 2*w.

    Parameters
    ----------
    V : number or array
        V number.
    core_radius : number
        Core radius a.

    Returns
    -------
    w : number or array
        Mode field radius.
    """

    return core_radius*(0.65 + 1.619/V**1.5 + 2.879/V**6)


# coupling:
def coupling_efficiency(field, mode, x, y):
    """Power coupling efficiency of a field into a mode (or any other field).

    eta = ``|<mode|field>|**2/(<mode|mode>*<field|field>)``.

    Parameters
    ----------
    field : 2d array
        Incident field.
    mode : 2d array
        Mode field on the same grid (need not be normalized).
    x, y : arrays
        Coordinates.

    Returns
    -------
    eta : number
        Efficiency between 0 and 1.
    """

    return (abs(scalar_product(mode, field, x, y))**2
            /(_power(mode, x, y)*_power(field, x, y)))


def mode_power_fractions(field, modes, x, y):
    """Fractions of the power of a field coupled into each of a set of modes.

    Parameters
    ----------
    field : 2d array
        Incident field.
    modes : sequence of 2d arrays
        (Orthogonal) mode fields, e.g. the fields of all LP modes of a fibre.
    x, y : arrays
        Coordinates.

    Returns
    -------
    fractions : 1d array
        Coupling efficiency into each mode.
    """

    P = _power(field, x, y)

    return np.array([abs(scalar_product(mode, field, x, y))**2/(_power(mode, x, y)*P)
                     for mode in modes])


def coupling_vs_offset(field, mode, x, y, offsets, axis="x"):
    """Coupling efficiency with the field shifted laterally.

    The field is shifted with a Fourier shift; keep it away from the grid
    edges.

    Parameters
    ----------
    field : 2d array
        Incident field.
    mode : 2d array
        Mode field on the same grid.
    x, y : arrays
        Coordinates.
    offsets : sequence of numbers
        Lateral shifts of the field.
    axis : {"x", "y"}, optional
        Direction of the shift.

    Returns
    -------
    eta : 1d array
        Coupling efficiency for each offset.
    """

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    K = K_X if axis == "x" else K_Y
    spectrum = FT(field)

    return np.array([coupling_efficiency(inv_FT(spectrum*np.exp(-1j*K*d)), mode, x, y)
                     for d in offsets])


def coupling_vs_tilt(field, mode, x, y, wavelength, angles, axis="x", n=1.0):
    """Coupling efficiency with the field tilted.

    The field is multiplied by exp(i*k*sin(theta)*x) (or y).

    Parameters
    ----------
    field : 2d array
        Incident field.
    mode : 2d array
        Mode field on the same grid.
    x, y : arrays
        Coordinates.
    wavelength : number
        Vacuum wavelength.
    angles : sequence of numbers
        Tilt angles in radians in the medium, in the xz or yz plane.
    axis : {"x", "y"}, optional
        Plane of the tilt.
    n : number, optional
        Refractive index of the medium.

    Returns
    -------
    eta : 1d array
        Coupling efficiency for each angle.
    """

    X, Y = ensure_meshgrid(x, y)
    coord = X if axis == "x" else Y
    k = wavenumber(wavelength, n)

    return np.array([coupling_efficiency(field*np.exp(1j*k*np.sin(theta)*coord), mode, x, y)
                     for theta in angles])


def _propagating_angular_spectrum(field, x, y, z, wavelength, n=1.0):
    # angular spectrum propagation that drops the evanescent waves, which
    # would grow without bound for z < 0
    k = wavenumber(wavelength, n)
    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    kz2 = k**2 - K_X**2 - K_Y**2
    TF = np.where(kz2 > 0, np.exp(1j*np.sqrt(np.abs(kz2))*z), 0.0)

    return inv_FT(FT(field)*TF), x, y


def coupling_vs_defocus(field, mode, x, y, wavelength, distances, n=1.0, propagator=None):
    """Coupling efficiency with the field propagated along z.

    The mode stays in the plane of the field, so positive distances mean the
    fibre end is behind the plane of the field.

    Parameters
    ----------
    field : 2d array
        Incident field.
    mode : 2d array
        Mode field on the same grid.
    x, y : arrays
        Coordinates.
    wavelength : number
        Vacuum wavelength.
    distances : sequence of numbers
        Propagation distances (positive or negative).
    n : number, optional
        Refractive index of the medium.
    propagator : callable, optional
        propagator(field, x, y, z, wavelength, n) returning the propagated
        field first. The default is the angular spectrum method without the
        evanescent waves; pass e.g. `propagators.fresnel_TF` to compare with
        paraxial (ABCD) results.

    Returns
    -------
    eta : 1d array
        Coupling efficiency for each distance.
    """

    propagator = _propagating_angular_spectrum if propagator is None else propagator

    return np.array([coupling_efficiency(propagator(field, x, y, dz, wavelength, n)[0], mode, x, y)
                     for dz in distances])
