#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module for basis sets.

In optics, different basis sets are used to describe fields in the plane, e.g.
Zernike polynomials in the description of aberrations in systems with circular
aperture stops. This module offers an object oriented approach to such basis
sets.
"""

from abc import ABCMeta, abstractmethod
from functools import lru_cache
from math import sqrt, floor, ceil

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from scipy.special import factorial as fac  # tolerant to negative arguments!
from scipy.linalg import lstsq, qr

from .utils import (kronecker_delta, weight_grid, simpson_weights, sgn,
                    scalar_product_without_weights,
                    scalar_product, ensure_meshgrid,
                    )
from .masks import (circular_mask, rectangluar_mask, annular_mask, to_bool)

# TODO: reintroduce scalar/inner product in BasisSet? this allowed for inner products that respected the mask/support automatically


def local_coordinates(x, y, x0=0.0, y0=0.0, angle=0.0):
    """Coordinates in a frame that is shifted to (x0, y0) and rotated by angle.

    A basis function f(u, v) evaluated at the local coordinates (u, v) of the
    points (x, y) is the function f moved to (x0, y0) and rotated
    counterclockwise (from x towards y) by angle about that point.

    Parameters
    ----------
    x, y : arrays or numbers
        Coordinates of the same shape.
    x0, y0 : numbers, optional
        Origin of the local frame.
    angle : number, optional
        Rotation angle in radians, counterclockwise.

    Returns
    -------
    u, v : arrays
        u = cos(angle)*(x - x0) + sin(angle)*(y - y0),
        v = -sin(angle)*(x - x0) + cos(angle)*(y - y0).
    """

    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    c, s = np.cos(angle), np.sin(angle)

    return c*dx + s*dy, -s*dx + c*dy


class BasisSet(object):
    """Base class for all basis sets. Should not be instantiated directly.
    """

    mask = None  # binary mask that defines the basis functions' support
    # this is needed for basis functions that are defined on non-rectangular
    # domains, e.g. the Zernike polynomials that have a circular domain of
    # definition. the specific mask is used in scalar products.

    # TODO: give __mul__() to evaluated objects (scalar product)?

    def __init__(self, x, y):
        """Define grid the basis functions are defined on.
        """
        self.x = x
        self.y = y
        self.XX, self.YY = ensure_meshgrid(x, y)
        self._has_norm = False

    def __call__(self, indices, coeffs, x=None, y=None):
        """Return basis functions for given indices.

        Parameters
        ----------
        indices : iterable
        coeffs : iterable
        x, y : arrays, optional
            If given, the basis is evaluated for the given (scattered) x and y
            points.

        Returns
        -------
        val : array
            Sampled data. The shape is identical to the scattered x, y data if
            x and y are given; otherwise val is of the same shape as x and y
            given to the class' constructor.
        """

        if (x is not None) and (y is not None):  # scattered data
            val = self.eval_scattered(indices, coeffs, x, y)
        else:  # data on x, y known to the class since instanciation
            val = sum([coeffs[i]*self.eval_single(ind) for i, ind in enumerate(indices)])

        return val

    def eval_single(self, index):
        """Return a single basis function.
        """

        raise NotImplementedError("Do not call this function in base class.")

    def eval_single_scattered(self, x, y, i):
        """Return i-th basis function at scattered (x, y) points.
        """

        # used in fitting routine for scattered data

        raise NotImplementedError("Do not call this function in base class.")

    def eval_scattered(self, inds, coeffs, x, y):
        """Evaluate basis for scattered x, y data.

        Parameters
        ----------
        inds : iterable
            Indices to evaluate.
        coeffs : iterable
        x, y : arrays

        Returns
        -------
        val : array
        """

        val = np.zeros_like(x)
        for ind, coeff in zip(inds, coeffs):
            val += coeff*self.eval_single_scattered(x, y, ind)

        return val

    def normalization_factor(self, i):
        """Return norm of i-th basis function. This is used in basis expansions.
        """

        raise NotImplementedError("Do not call this function in base class.")

    def coeffs_from_field(self, field, indices, method="LSTSQ"):
        """Expand a given field in the basis set.
        """

        if method == "LSTSQ":
            coeffs = self._coeffs_from_lstsq(field, indices)
        elif method == "DI":
            coeffs = self._coeffs_from_integration(field, indices)
        else:
            raise ValueError("Unknown method. Must be one of 'LSTSQ' or 'DI'")

        return coeffs

    def _coeffs_from_lstsq(self, field, indices, do_scale=False):
        """Determine basis expansion coefficients by least-square fitting.

        Parameters
        ----------
        field : array
            Sampled data to expand in basis.
        indices : array, integer
            Indices to consider in fit.
        do_scale : boolean, optional
            If True, apply preconditioning to least square problem. Defaults to
            False.

        Returns
        -------
        coeffs : array
            Expansion coefficients.

        Notes
        -----
        Least square fitting may be better than direct integration as it tries
        to optimally describe the given field. However, balancing the basis
        functions for minimum residual can be unphysical, specifically if the
        sampling is questionable. However, least squares generally give very
        good results and are recommended generally.
        """

        # TODO: preconditioning?
        Z_i = [self.eval_single(i) for i in indices]
        N_samples = len(Z_i[0].flatten())
        N_indices = len(indices)
        M = np.zeros([N_samples, N_indices])
        for i in range(N_indices):
            M[:, i] = Z_i[i].flatten()
        if do_scale:
            scale = np.sqrt((M*M).sum(axis=0))
            scale = np.log2(scale)
            M /= 2.0**(scale)
        coeffs, _, _, s = lstsq(M, field.flatten())
        if do_scale:
            coeffs /= 2.0**scale

        return coeffs

    def _coeffs_from_integration(self, field, indices):
        """Determine basis expansion coefficients by direct overlap integral
        computation.

        Parameters
        ----------
        field : array
            Sampled data to expand in basis.
        indices : array, integer
            Indices to consider in fit.

        Returns
        -------
        coeffs : array
            Expansion coefficients.
        """

        if self._has_norm:
            norms = [self.normalization_factor(i) for i in indices]
        else:
            norms = len(indices)*(1.0,)

        # TODO: weights are hardcoded, how to change that?
        coeffs = 1.0/np.array(norms)*np.array([scalar_product(self.eval_single(i),
                                                              self.mask*field,
                                                              self.x, self.y,
                                                              simpson_weights)
                                               for i in indices
                                               ])

        return coeffs


    def coeffs_from_scattered(self, x, y, data, indices, do_scale=False):
        """Fit basis to given scattered data.

        Parameters
        ----------
        x, y : arrays
        data : arrays
           Data sampled at x, y. Of identical shape as each x and y.
        indices : array or list
           Indices to be used in fit.
        do_scale : boolean, optional
            If True, apply preconditioning to least squares problem. Defaults
            to False.

        Returns
        -------
        coeffs : array
        """

        Z_i = [self.eval_single_scattered(x, y, i) for i in indices]
        N_samples = len(Z_i[0])
        N_indices = len(indices)
        M = np.zeros([N_samples, N_indices])
        for i in range(N_indices):
            M[:, i] = Z_i[i]
        if do_scale:
            scale = np.sqrt((M*M).sum(axis=0))
            scale = np.int64(np.log2(scale))
            M /= 2.0**(scale)
        coeffs, _, _, s = lstsq(M, data)
        if do_scale:
            coeffs /= 2.0**scale

        return coeffs


class PresampledBasisSet(BasisSet):
    """Sampled basis functions.

    Substitutes run-time recomputation of data by lookups of precomputed data.
    """

    def __init__(self, x, y, basis_type=None, indices=None, **kwargs):
        """Presample given basis function and store away for fast evaluation.

        Parameters
        ----------
        x, y : arrays
        basis_type : object
            BasisSet class to presample.
        indices : array-like
            Array of indices to presample.
        kwargs : dict
            Keyword arguments to hand over basis_type on instantiation.
        """

        super(PresampledBasisSet, self).__init__(x, y)

        self.basis = basis_type(x, y, **kwargs)
        self.sampled_funcs = {}

        for ind in indices:
            sampled_func = self.basis.eval_single(ind)
            self.sampled_funcs[ind] = sampled_func

        # redirect norm & mask issues to contained basis:
        self._has_norm = self.basis._has_norm
        self.normalization_factor = self.basis.normalization_factor
        self.mask = self.basis.mask
        # TODO: is mask_x and mask_y are elements of self.basis, expose those too

    def eval_single(self, index):
        if index not in self.sampled_funcs.keys():
            raise ValueError("Index {} not amongst the sampled indices".format(index))

        sampled_func = self.sampled_funcs[index]

        return sampled_func


def fringe_zernike_norm(j, R_norm=1.0):
    n, m = fringe_to_n_m(j)  # radial and azimuthal index

    norm = R_norm**2*np.pi*(1.+kronecker_delta(m, 0))/(2.*(n + 1.))

    return norm


def fringe_to_n_m(j):
    """Map Fringe Zernike index j to Zernike indices n, m.
    """

    d = floor(sqrt(j-1)) + 1
    m = floor((((d**2-j)/2)) if (((int(d)**2 - j) % 2) == 0) else ceil((-(d**2) + j - 1)/2.))
    n = round((2.0*(d-1) - abs(m)))

    return int(n), int(m)


@lru_cache(maxsize=None)
def _fringe_zernike_R_nm_coeffs(n, m):
    """Coefficients of R_nm (cached, read-only).

    Parameters
    ----------
    n, m : ints

    Returns
    -------
    coeffs : array
        Coefficients, coeffs[0] contains coefficients of highest power.
    """

    N, M = int(n), int(m)

    coeffs = np.zeros(N+1)
    for k in range((N-M)//2 + 1):
        # TODO: implement formula with binomials using mpmath
        denom = fac(k) * fac((N+M)/2.0 - k) * fac((N-M)/2.0 - k)
        if np.isclose(denom, 0.0):  # avoid division by zero
            coeffs[N-2*k] = 0.0
        else:
            coeffs[N-2*k] = (-1.0)**k * fac(N-k) / denom

    coeffs = coeffs[::-1]  # highest power first, for np.polyval()
    coeffs.flags.writeable = False  # shared by all callers through the cache

    return coeffs


def fringe_zernike_R_nm(n, m, rho):
    """Radial polynomial for Zernikes.

    Parameters
    ----------
    n : int
    m : int
    rho : array or number

    Returns
    -------
    R_mm : number or array
    """

    coeffs = _fringe_zernike_R_nm_coeffs(int(n), int(m))
    R = np.polyval(coeffs, rho)

    return R


def fringe_zernike_Y_m(m, phi):
    """Angle dependence of Zernike polynomials.

    Parameters
    ----------
    m : int
    phi : array or number

    Returns
    -------
    Y_m : number or array
    """

    # TODO: factor out?
    if m > 0:
        val = np.cos(m*phi)
    elif m == 0:
        val = np.ones(np.shape(phi))  # TODO: this assumes array-valued phi. change that?
    else:
        val = np.sin(-m*phi)  # minus sign to get |m| for m < 0, otherwise there'll be a sign flip

    return val


def fringe_zernike(j, x, y):
    """Fringe Zernike Z_j at the points (x, y) of the unit disk (no mask is
    applied, peak value 1 on the unit circle).

    Parameters
    ----------
    j : int
        Fringe index.
    x, y : arrays or numbers
        Coordinates normalized to the radius of the disk.

    Returns
    -------
    Z_j : array
    """

    n, m = fringe_to_n_m(j)
    rho, phi = np.hypot(x, y), np.arctan2(y, x)

    return fringe_zernike_R_nm(n, abs(m), rho)*fringe_zernike_Y_m(m, phi)


def fringe_zernike_sum(coeffs, x, y):
    """sum_j c_j Z_j at the points (x, y) of the unit disk for the
    coefficients {j: c_j} (real or complex); zeros for no coefficients."""

    out = np.zeros(np.broadcast(x, y).shape)
    for j, c in (coeffs or {}).items():
        out = out + c*fringe_zernike(j, x, y)

    return out


class FringeZernikes(BasisSet):
    """Fringe Zernike polyonimals as described in Gross' Handbook of Optics
    (2nd vol) or
    http://www.jcmwave.com/JCMsuite/doc/html/ParameterReference/0c19949d2f03c5a96890075a6695b258.html
    """

    def __init__(self, x, y, R_norm=1.0, x0=0.0, y0=0.0, angle=0.0):
        """Fringe Zernikes on the disk of radius R_norm around (x0, y0).

        Parameters
        ----------
        x, y : arrays
            Linear coordinates of the grid.
        R_norm : number, optional
            Normalization radius (radius of the unit disk).
        x0, y0 : numbers, optional
            Center of the disk.
        angle : number, optional
            Rotation of the basis functions in radians, counterclockwise. The
            azimuth phi of the polynomials is measured from the rotated x axis,
            e.g. Z_2 (x tilt) rotated by pi/2 equals Z_3 (y tilt).
        """

        super(FringeZernikes, self).__init__(x-x0, y-y0)

        self.R_norm = R_norm
        self.x0, self.y0, self.angle = x0, y0, angle
        U, V = local_coordinates(self.XX, self.YY, 0.0, 0.0, angle)
        self.Rho = np.sqrt(U**2 + V**2)/R_norm
        self.Phi = np.arctan2(V, U)

        self.mask = circular_mask(x-x0, y-y0, R_norm)
        self._has_norm = True

    def eval_single(self, j):
        n, m = fringe_to_n_m(j)
        R = fringe_zernike_R_nm(n, abs(m), self.Rho)
        Y = fringe_zernike_Y_m(m, self.Phi)

        return R*Y*self.mask

    def eval_single_scattered(self, x, y, j):
        """Return the j-th Fringe Zernike at scattered points (x, y) (no
        mask is applied)."""

        u, v = local_coordinates(x, y, self.x0, self.y0, self.angle)

        return fringe_zernike(j, u/self.R_norm, v/self.R_norm)

    def normalization_factor(self, i):
        """Norm ``<Z_i|Z_i>`` of i-th Fringe-Zernike polynomial.

        Parameters
        ----------
        i : int

        Returns
        -------
        norm : double

        Notes
        -----
        The Fringe Zernikes get their normalization from the choice of
        R(R_max) = 1. This means the norm over the entire (unit) disk may
        differ with the Fringe index.
        """

        norm = fringe_zernike_norm(i, self.R_norm)

        return norm


def legendre_norm_1d(n, a=1.0):
    """Return norm of L_n(x).

    Parameters
    ----------
    n : int
        Index to L_n(x).
    a : number
        Scaling parameter.

    Returns
    -------
    norm : number
    """

    norm = a*2./(2*n + 1.)

    return norm


def legendre_norm_2d(j, a=1.0, b=1.0):
    """Return norm of tensor product basis state L_n(x)*L_m(y).

    Parameters
    ----------
    j : int
        Single index to tensor product state. Maps to n, m as in
        LegendrePolynomials class.
    a, b : numbers
        x and y scaling parameters.

    Returns
    -------
    norm : number
    """

    n, m = _poly_single_index_to_n_m(j)
    norm = legendre_norm_1d(n, a)*legendre_norm_1d(m, b)

    return norm



class TensorProductBasis(BasisSet, metaclass=ABCMeta):
    """Abstract base class for tensor product bases P_n(u/a)*P_m(v/b) on the
    rectangle ``|u|`` <= a, ``|v|`` <= b, built from a family P_n of 1D polynomials.

    (u, v) are the local coordinates of the rectangle that is centered at
    (x0, y0) and rotated counterclockwise by angle (see local_coordinates()).
    Subclasses implement eval_1d(); a subclass that knows its norms sets
    _has_norm = True and implements normalization_factor().

    The single index j enumerates the pairs (n, m) by total order n + m:
    (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), ...
    """

    def __init__(self, x, y, a=1.0, b=1.0, x0=0.0, y0=0.0, angle=0.0):
        """Parameters
        ----------
        x, y : arrays
            Linear coordinates of the grid.
        a, b : numbers, optional
            Half side lengths of the rectangle along the (rotated) x and y
            axes.
        x0, y0 : numbers, optional
            Center of the rectangle.
        angle : number, optional
            Rotation of the rectangle and the basis functions in radians,
            counterclockwise.
        """

        super(TensorProductBasis, self).__init__(x, y)
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.angle = angle
        self._has_norm = False

        # scaled coordinates (the rectangle is [-1, +1] x [-1, +1]):
        U, V = local_coordinates(self.XX, self.YY, x0, y0, angle)
        self.XX_scaled, self.YY_scaled = U/a, V/b

        if angle == 0.0:
            # axis parallel: the basis functions are outer products of 1D
            # arrays, which is faster to evaluate
            self.x_scaled, self.y_scaled = (x-x0)/a, (y-y0)/b
            self.mask, self.mask_x, self.mask_y = rectangluar_mask(x, y, a, b, x0, y0, True)
        else:
            self.x_scaled = self.y_scaled = None
            self.mask_x = self.mask_y = None
            self.mask = ((np.abs(self.XX_scaled) <= 1) & (np.abs(self.YY_scaled) <= 1)).astype(int)

    @abstractmethod
    def eval_1d(self, t, n):
        """Evaluate the n-th 1D polynomial at t (scaled coordinates)."""

    def eval_single(self, index):
        n, m = self.single_index_to_n_m(index)

        if self.x_scaled is not None:
            xy_poly = np.outer(self.eval_1d(self.y_scaled, m), self.eval_1d(self.x_scaled, n))
        else:
            xy_poly = self.eval_1d(self.XX_scaled, n)*self.eval_1d(self.YY_scaled, m)

        return self.mask*xy_poly

    def eval_single_scattered(self, x, y, i):
        """Return i-th basis function at scattered (x, y) points (no mask is
        applied).
        """

        n, m = self.single_index_to_n_m(i)
        u, v = local_coordinates(x, y, self.x0, self.y0, self.angle)

        return self.eval_1d(u/self.a, n)*self.eval_1d(v/self.b, m)

    def single_index_to_n_m(self, index):
        """Convert single index to indices n, m for P_n(x) and P_m(y).
        """

        n, m = _poly_single_index_to_n_m(index)

        return n, m


class LegendrePolynomials(TensorProductBasis):
    """Legendre polynomials are a suitable basis set for fields defined on
    rectangular apertures.

    The basis set is the product of two Legendre polynomials L_n(u/a)*L_m(v/b);
    see TensorProductBasis for the parameters.
    """

    def __init__(self, x, y, a=1.0, b=1.0, x0=0.0, y0=0.0, angle=0.0):
        super(LegendrePolynomials, self).__init__(x, y, a, b, x0, y0, angle)
        self._has_norm = True

    def eval_1d(self, t, n):
        c = np.zeros(n+1)
        c[n] = 1.0

        return legval(t, c)

    def normalization_factor(self, i):
        """Norm ``<L_i|L_i>`` of i-th Legendre basis function.

        Parameters
        ----------
        i : int

        Returns
        -------
        norm : double
        """

        # the Legendres are a product basis L_n(x)*L_m(y) - using Fubini's
        # theorem, the normalization can thus be written as the product of
        # norms norm(L_n)*norm(L_m)
        norm = legendre_norm_2d(i, self.a, self.b)

        return norm


class ChebyshevPolynomials(TensorProductBasis):
    """Chebyshev polynomials are a suitable basis set for fields defined on
    rectangular apertures.

    The basis set is the product of two Chebyshev polynomials T_n(u/a)*T_m(v/b);
    see TensorProductBasis for the parameters. The basis is orthogonal with
    respect to a specific scalar product with weight function 1/sqrt(1-x**2)
    for each dimension.
    """

    def eval_1d(self, t, n):
        c = np.zeros(n+1)
        c[n] = 1.0

        return chebval(t, c)


def _poly_single_index_to_n_m(single_ind):
    """Map single index to x, y indices/powers.

    n and m may be used as powers in monomials (Polynomials class) or as indices
    for e.g. Legendre polynomials.
    """

    # find order of monomial: single_ind >= (order+1)*(order+2)
    # solve expression for number of monomials up to given order for order:
    order_fract = int(-3./2 + sqrt(9./4 - 2*(1-single_ind)) + 1)
    order = int(floor(order_fract))

    element = single_ind - poly_basis_size(order-1)  # which element in the sequence of monomials up to given order?

    n = order - element
    m = element

    return n, m


def poly_basis_size(order):
    """Number of monomials up to given order.
    """

    if order < 0:
        s = 0  # account for order 0
    else:
        s = (order+1)*(order+2)//2

    return s


class Polynomials(BasisSet):
    """Polynomials in x and y.
    """

    pass


class NumericallyOrthogonalized(PresampledBasisSet):
    """Represents a basis set that is numerically orthogonalized through a
    Gram-Schmidt procedure on the given grid.
    """

    def __init__(self, x, y, basis_type=None, indices=None, new_mask=None, weight_func=None,
                 norm_func=None, **kwargs):
        """Construct numerically orthogonalized basis function set.

        Parameters
        ----------
        x, y : arrays
        basis_type : BasisSet class
            Base class to instantiate, sample and orthogonalize.
        indices : array-like
            Array of indices to sample.
        new_mask : array
            New mask on which the basis functions should be numerically
            orthogonalized on.
        weight_func : callable, optional
            Function that returns 1d weights for inner product summation. If
            None, a standard midpoint rule will be used. The weights are
            also used in norm computations.
        norm_func : callable, optional
            Function norm(i) that returns the desired norm for the i-th basis
            function. Defaults to None, which will keep the old norm (even if
            new_mask defines a new support).

        kwargs : dict
            Handed over to basis_type object.

        Notes
        -----
        As the QR decomposition used in the orthogonalization process may change
        signs, a rescaling to prevent overall sign is performed. In this process
        the signs of the first element within the mask are evaluated. These
        should not be zero.

        If a new mask is given, it is advised to choose the basis_type such that
        the new mask is 'filled' even with the original basis functions, e.g.
        if the new mask is an annulus with outer radius R_outer and the
        basis_type are Fringe Zernike polynomials, it is recommended to set the
        Fringe Zernikes R_norm parameter to R_outer.

        If a new mask is given without a normalization function, the old the
        new results can be difficult the compare as the old norms are kept. This
        may be undesirable for different spatial support, e.g. may smaller supports
        lead to higher values. Use RescaledBasis to rescale.
        """

        super(NumericallyOrthogonalized, self).__init__(x, y, basis_type=basis_type,
                                                        indices=indices,
                                                        **kwargs)

        if new_mask is None:
            mask = to_bool(self.basis.mask)
        else:
            mask = to_bool(new_mask)
            self.mask = new_mask

        if weight_func is None:
            weight_func = lambda N: np.zeros(N) + 1.0
        weights = weight_grid(weight_func, len(self.x), len(self.y))
        weights_masked = weights[mask]

        # run orthogonalisation:

        # determine size needed for basis matrix:
        num_samples_mask = np.sum(mask)  # number of rows
        num_basis_funcs = len(indices)  # number of columns

        # build matrix for orthogonalization procedure: put data within mask in columns
        basis_funcs = np.zeros([num_samples_mask, num_basis_funcs], order="C")  # TODO: check order for best performance
        for i, ind in enumerate(indices):
            curr_sampled_func = self.sampled_funcs[ind]
            basis_funcs[:, i] = (np.sqrt(weights_masked)*curr_sampled_func[mask])

        # QR:
        Q, R = qr(basis_funcs, mode="economic")

        # dictionary that holds the coefficients of the linear combination that
        # makes the new basis orthogonal
        self.transformation_coeffs = {}

        for i, ind in enumerate(indices):
            curr_orthogonalized_basis_func = Q[:, i]
            self.transformation_coeffs[ind] = R[:, i]
            # compute norm renormalized basis function. as the square root of
            # the weights to use are already included in curr_orthogonalized_basis_func,
            # we can use the scalar product that does not expect any weights:
            curr_norm = scalar_product_without_weights(curr_orthogonalized_basis_func,
                                                       curr_orthogonalized_basis_func,
                                                       self.x, self.y)
            if norm_func is None:
                # keep old norm:
                old_basis_func = basis_funcs[:, i]
                desired_norm = scalar_product_without_weights(old_basis_func,
                                                              old_basis_func,
                                                              self.x, self.y,
                                                              )
            else:
                desired_norm = norm_func(ind)

            # overwrite sampled_funcs by orthogonalized data,
            # renormalize to old norm and apply scaling factor (preserve sign of first vector elements - QR decomposition may change that):
            scale = sgn(curr_orthogonalized_basis_func[0])*sgn(basis_funcs[0, i])  # sign scale
            factor = scale/np.sqrt(curr_norm)*np.sqrt(desired_norm)/np.sqrt(weights_masked)  # renormalization

            factor_transform = scale/np.sqrt(curr_norm)*np.sqrt(desired_norm)/len(curr_orthogonalized_basis_func)  # accomodate sign and change in norm;
                                                                                                                   # divide by number of samples to compensate for missing dx*dy
            self.sampled_funcs[ind][mask] = factor*curr_orthogonalized_basis_func
            self.sampled_funcs[ind][~mask] = 0.0
            self.transformation_coeffs[ind] *= factor_transform  # also rescale the transform


    def eval_single_scattered(self, x, y, i, interpolation="linear"):
        # TODO: implement

        # generate interpolator (if not yet present)
        # use interpolator to interpolate to scattered x, y

        raise NotImplementedError()


class RescaledBasis(BasisSet):
    """Rescale basis functions on a new aperture/mask.

    Use this for e.g. normalizing FringeZernikes an non-circular aperture such
    that coefficients can again be read as maximum values on the aperture.
    """

    def __init__(self, x, y, basis_to_rescale_type=None, mask=None, scale_func=None, scale_val_func=None, **kwargs):
        """Create RescaledBasis instance.

        Parameters
        ----------
        x, y : arrays
            Linear coordinate arrays
        basis_to_rescale_type : BasisSet subclass object
            The basis to use with a new mask
        mask : array
            Sampled mask or aperture to use with evaluated basis functions.
        scale_func : callable
            Callable that returns a scale for evaluated basis function, e.g.
            lambda val: np.nanmax(np.abs(val))
        scale_val_func : callable
            Function that computes the new scale for a given index, e.g.
            lambda i : 1 for an index-independent scale of 1.
        kwargs : dict
            Handed to basis_type on instantiation.
        """

        super(RescaledBasis, self).__init__(x, y)
        self.mask = mask  # store new mask, used in evaluation
        self._basis = basis_to_rescale_type(x, y, **kwargs)  # instantiate basis for new aperture
        self.scale_func = scale_func
        self.scale_val_func = scale_val_func

    def get_rescaling_factor(self, index):
        """Get factor to rescale with.
        """

        func = self._basis.eval_single(index)
        rescaling_factor = self.get_rescaling_factor_from_sampled_func(index, func)

        return rescaling_factor

    def get_rescaling_factor_from_sampled_func(self, index, sampled_func):
        """Get rescaling factor for given index and sampled base function.
        """

        old_scale = self.scale_func(sampled_func)
        new_scale = self.scale_val_func(index)
        rescaling_factor = new_scale/old_scale

        return rescaling_factor


    def eval_single(self, index):
        val = self._basis.eval_single(index)
        val *= self.mask
        val *= self.get_rescaling_factor_from_sampled_func(index, val)

        return val


def basis_from_generating_function(index_func, field_func, **kwargs):
    """Generate a BasisSet object from a given field function.

    Parameters
    ----------
    index_func : callable
        Maps a single integer n to one or several indices i, j, k...
    field_func : callable
        Field or basis function generating function with signature
        ``field_func(i, j, k, ..., x, y, **kwargs)``.

    Returns
    -------
    generated_basis : BasisSet object
    """

    pass
