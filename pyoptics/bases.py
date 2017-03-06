#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module for basis sets.

In optics, different basis sets are used to describe fields in the plane, e.g.
Zernike polynomials in the description of aberrations in systems with circular
aperture stops. This module offers an object oriented approach to such basis
sets.
"""

from math import sqrt, floor, ceil

import numpy as np
from numpy.polynomial.legendre import legval2d, legval
from scipy.misc import factorial as fac  # tolerant to negative arguments!
from scipy.linalg import lstsq, qr

from pyoptics.utils import (kronecker_delta, weight_grid, simpson_weights, sgn,
                            scalar_product_without_weights,
                            scalar_product
                           )
from pyoptics.masks import circular_mask, rectangluar_mask

# TODO: reintroduce scalar/inner product in BasisSet? this allowed for inner products that respected the mask/support automatically


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
        self.XX, self.YY = np.meshgrid(x, y)
        self._has_norm = False

    def __call__(self, indices, coeffs=None):
        """Return basis functions for given indices.
        """

        if coeffs is None:
            coeffs = np.ones(len(indices))

        val = sum([coeffs[i]*self.eval_single(ind) for i, ind in enumerate(indices)])

        return val

    def eval_single(self, index):
        """Return a single basis function.
        """

        pass

    def normalization_factor(self, i):
        """Return norm of i-th basis function. This is used in basis expansions.
        """

        raise NotImplementedError("Do not call this function in base class.")

    def coeffs_from_field(self, field, indices, method="LSTSQ"):
        """Expand a given field in the basis set.
        """

        if method == "LSTSQ":
            coeffs = self._coeffs_from_field_lstsq(field, indices)
        elif method == "DI":
            coeffs = self._coeffs_from_field_integration(field, indices)
        else:
            raise ValueError("Unknown method. Must be one of 'LSTSQ' or 'DI'")

        return coeffs

    def _coeffs_from_field_lstsq(self, field, indices):
        """Determine basis expansion coefficients by least-square fitting.

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

        Note
        ----
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
        coeffs, _, _, _ = lstsq(M, field.flatten())

        return coeffs

    def _coeffs_from_field_integration(self, field, indices):
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


class PresampledBasisSet(BasisSet):
    """Sampled basis functions.

    Substitutes run-time recomputation of data by lookups of precomputed data.
    """

    def __init__(self, x, y, basis_class, indices, **kwargs):
        """Presample given basis function and store away for fast evaluation.

        Parameters
        ----------
        x, y : arrays
        basis_class : object
            BasisSet class to presample.
        indices : array-like
            Array of indices to presample.
        kwargs : dict
            Keyword arguments to hand over basis_class on instantiation.
        """

        super(PresampledBasisSet, self).__init__(x, y)

        self.basis = basis_class(x, y, **kwargs)
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

    return n, m


def _fringe_zernike_R_nm_coeffs(n, m):
    """Coefficients of R_nm.

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
    for k in range((N-M)/2 + 1):
        coeffs[N-2*k] = (-1.0)**k * fac(N-k) / (fac(k) * fac((N+M)/2.0 - k) * fac((N-M)/2.0 - k))

    return coeffs[::-1]  # highest power first, for np.polyval()


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

    coeffs = _fringe_zernike_R_nm_coeffs(n, m)
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
        val = np.ones(np.shape(phi))
    else:
        val = np.sin(m*phi)

    return val


class FringeZernikes(BasisSet):
    """Fringe Zernike polyonimals as described in Gross' Handbook of Optics (2nd vol) or
    http://www.jcmwave.com/JCMsuite/doc/html/ParameterReference/0c19949d2f03c5a96890075a6695b258.html
    """

    def __init__(self, x, y, R_norm):
        super(FringeZernikes, self).__init__(x, y)
        self.R_norm = R_norm
        self.Rho = np.sqrt(self.XX**2 + self.YY**2)/R_norm
        self.Phi = np.arctan2(self.YY, self.XX)

        self.mask = circular_mask(x, y, R_norm)  # TODO: introduce x0, y0!
        self._has_norm = True

    def eval_single(self, j):
        n, m = fringe_to_n_m(j)
        R = fringe_zernike_R_nm(n, abs(m), self.Rho)
        Y = fringe_zernike_Y_m(m, self.Phi)

        return R*Y*self.mask

    def normalization_factor(self, i):
        """Norm <Z_i|Z_i> of i-th Fringe-Zernike polynomial.

        Parameters
        ----------
        i : int

        Returns
        -------
        norm : double

        Note
        ----
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



class LegendrePolynomials(BasisSet):
    """Legendre polynomials are a suitable basis set for fields defined on
    rectangular apertures.

    The basis set is the product of two Legendre polynomials L_i(x)*L_j(y).

    """

    def __init__(self, x, y, a, b, x0=0.0, y0=0.0):
        super(LegendrePolynomials, self).__init__(x, y)
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self._has_norm = True

        # store scaled x and y arrays (should contain interval (-1, +1))
        # for NumPy legval calls:
        self.x_scaled, self.y_scaled = (x-x0)/a, (y-y0)/b
        self.XX_scaled, self.YY_scaled = np.meshgrid(self.x_scaled, self.y_scaled)

        # mask is a rectangle x_scaled = [-1, +1], y_scaled = [-1, +1]
        self.mask, self.mask_x, self.mask_y = rectangluar_mask(x, y, a, b, x0, y0, True)

    def __call__(self, indices, coeffs=None):
        """Return basis functions for given indices.
        """

        if coeffs is None:
            coeffs = np.ones(len(indices))

        n, m = self.single_index_to_n_m(index)

        #c = np.zeros(2*(max(n, m)+1,))
        c = np.zeros([n, m])
        # using legval2d, row index goes to x polynomial, column index to y polynmomial:
        c[n, m] = 1.0
        xy_poly = legval2d(self.XX_scaled, self.YY_scaled, c)

        # TODO: fix this routine!

        return xy_poly

    def eval_single(self, index):
        n, m = self.single_index_to_n_m(index)

        # outer product approach:
        cx = np.zeros(n+1)
        cx[n] = 1.0
        cy = np.zeros(m+1)
        cy[m] = 1.0
        xy_poly = np.outer(legval(self.y_scaled, cy), legval(self.x_scaled, cx))  # TODO: order?

        return self.mask*xy_poly

    def single_index_to_n_m(self, index):
        """Convert single index to indices n, m for L_n(x) and L_m(y).
        """

        n, m = _poly_single_index_to_n_m(index)

        return n, m

    def normalization_factor(self, i):
        """Norm <L_i|L_i> of i-th Legendre basis function.

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
        s = (order+1)*(order+2)/2

    return s


class Polynomials(BasisSet):
    """Polynomials in x and y.
    """

    pass


class NumericallyOrthogonalized(PresampledBasisSet):
    """Represents a basis set that is numerically orthogonalized through a
    Gram-Schmidt procedure on the given grid.
    """

    def __init__(self, x, y, base_type, indices, new_mask=None, weight_func=None,
                 norm_func=None, **kwargs):
        """Construct numerically orthogonalized basis function set.

        Parameters
        ----------
        x, y : arrays
        base_type : BasisSet class
            Base class to instantiate, sample and orthogonalize.
        indices : array-like
            Array of indices to sample.
        new_mask : array
            New mask on which the basis functions should be numerically
            orthogonalized on.
        weights_func : callable, optional
            Function that returns 1d weights for inner product summation. If
            None, a standard midpoint rule will be used. The weights are
            also used in norm computations.
        norm_func : callable, optional
            Function norm(i) that returns the desired norm for the i-th basis
            function. Defaults to None, which will keep the old norm (even if
            new_mask defines a new support).

        kwargs : dict
            Handed over to base_type object.

        Note
        ----
        As the QR decomposition used in the orthogonalization process may change
        signs, a rescaling to prevent overall sign is performed. In this process
        the signs of the first element within the mask are evaluated. These
        should not be zero.

        If a new mask is given, it is advised to choose the base_type such that
        the new mask is 'filled' even with the original basis functions, e.g.
        if the new mask is an annulus witzh outer radius R_outer and the
        base_type are Fringe Zernike polynomials, it is recommended to set the
        Fringe Zernikes R_norm parameter to R_outer.

        If a new mask is given without a normalization function, the old the
        new results can be diffult the compare as the old norms are kept. This
        may be undesirable for different spatial support.
        """

        super(NumericallyOrthogonalized, self).__init__(x, y, base_type, indices, **kwargs)

        if new_mask is None:
            mask = self.basis.mask > 0.0 # TODO: why > 0 necessary?
        else:
            mask = new_mask > 0.0
            self.mask = new_mask

        if weight_func is None:
            weight_func = lambda N: np.zeros(N) + 1.0
        weights = weight_grid(weight_func, len(self.x), len(self.y))
        weights_masked = weights[mask]

        # run orthogonalisation:

        # determine size needed for basis matrix:
        num_samples_mask = np.sum(mask)  # number of rows
        num_basis_funcs = len(indices)  # number of columns

        # build matrix for orthogonalizatgion procedure: put data within mask in columns
        basis_funcs = np.zeros([num_samples_mask, num_basis_funcs], order="C")  # TODO: check order for best performance
        for i, ind in enumerate(indices):
            curr_sampled_func = self.sampled_funcs[ind]
            basis_funcs[:, i] = (np.sqrt(weights_masked)*curr_sampled_func[mask])

        # QR:
        Q, _ = qr(basis_funcs, mode="economic")

        for i, ind in enumerate(indices):
            curr_orthogonalized_basis_func = Q[:, i]

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
            scale = sgn(curr_orthogonalized_basis_func[0])*sgn(basis_funcs[0, i])
            self.sampled_funcs[ind][mask] = \
                scale*curr_orthogonalized_basis_func/np.sqrt(curr_norm)*np.sqrt(desired_norm)/np.sqrt(weights_masked)
            self.sampled_funcs[ind][~mask] = 0.0


if(__name__ == '__main__'):
    from pyoptics.utils import grid1d
    from pyoptics.plot_tools import plot_intensity
    x = grid1d(-2, +2, 512)
    y = grid1d(-2, +2, 512)
    R_norm = 1.5
    Z = FringeZernikes(x, y, R_norm)
    for j in range(1, 16+1):
        Z_sampled = Z.eval_single(j)
        print(Z_sampled)
        plot_intensity(Z_sampled, x, y, title="$Z_{%s}$"%j)

    # compute inner product matrix numerically:
