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
from scipy.linalg import lstsq

from pyoptics.utils import scalar_product, kronecker_delta

# TODO: reintroduce scalar/inner product in BasisSet? this allowed for inner products that respected the mask/support automatically
# TODO: in case of scaled x, y supports (Zernike R_norm or Legdendre rectangle size): which x, y to store? Scaled or unscaled? Compute scalar products in which coordinates?


# probably keep original coordinates and use those in all computations, but also store transform x, y for eval_single().
# this allows reuse of x, y by the user and will shift responsibilty away from the user to the developer!


class BasisSet(object):
    basis_size = 0  # number of coefficients to use
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
        coeffs = 1.0/np.array(norms)*np.array([scalar_product(self.eval_single(i), self.mask*field, self.x, self.y, simpson_weights) for i in indices])

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
            self.sampled_funcs[str(ind)] = sampled_func

        # redirect norm issues to contained basis:
        self._has_norm = self.basis._has_norm
        self.normalization_factor = self.basis.normalization_factor

    def eval_single(self, index):
        if str(index) not in self.sampled_funcs.keys():
            raise ValueError("Index {} not amongst the sampled indices".format(index))

        sampled_func = self.sampled_funcs[str(index)]

        return sampled_func


class FringeZernikes(BasisSet):
    """Fringe Zernike polyonimals as described in Gross' Handbook of Optics (2nd vol) or
    http://www.jcmwave.com/JCMsuite/doc/html/ParameterReference/0c19949d2f03c5a96890075a6695b258.html
    """

    def __init__(self, x, y, R_norm):
        super(FringeZernikes, self).__init__(x, y)
        self.R_norm = R_norm
        self.Rho = np.sqrt(self.XX**2 + self.YY**2)/R_norm
        self.Phi = np.arctan2(self.YY, self.XX)

        mask = np.zeros(np.shape(self.XX))
        mask[self.Rho <= 1.0] = 1.0
        self.mask = mask
        self._has_norm = True

    def R_nm(self, n, m):
        # TODO: factor out?
        N, M = int(n), int(m)

        R = np.zeros_like(self.Rho)
        for k in range((N-M)/2 + 1):
            R += (-1.0)**k * fac(N-k) / (fac(k) * fac((N+M)/2.0 - k) * fac((N-M)/2.0 - k )) * self.Rho**(n-2.0*k)

        return R

    def Y_m(self, m):
        # TODO: factor out?
        if m > 0:
            val = np.cos(m*self.Phi)
        elif m == 0:
            val = np.ones(np.shape(self.Phi))
        else:
            val = np.sin(m*self.Phi)

        return val

    def fringe_to_n_m(self, j):
        d = floor(sqrt(j-1)) + 1
        m = floor((((d**2-j)/2)) if (((int(d)**2 - j) % 2) == 0) else ceil((-(d**2) + j - 1)/2.))
        n = round((2.0*(d-1) - abs(m)))

        return n, m

    def eval_single(self, j):
        n, m = self.fringe_to_n_m(j)
        R = self.R_nm(n, abs(m))
        Y = self.Y_m(m)

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
        differs with the Fringe index.
        """

        n, m = self.fringe_to_n_m(i)

        norm = self.R_norm**2*np.pi*(1.+kronecker_delta(m, 0))/(2.*(n + 1.))

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

        # store scaled x and y arrays (should contain (-1, +1)) for NumPy legval
        # calls:
        self.x_scaled, self.y_scaled = (x-x0)/a, (y-y0)/b
        self.XX_scaled, self.YY_scaled = np.meshgrid(self.x_scaled, self.y_scaled)

        # mask is a rectangle x_scaled = [-1, +1], y_scaled = [-1, +1]
        self.mask_x = np.zeros_like(x)
        self.mask_x[(self.x_scaled >= -1) & (self.x_scaled <= +1)] = 1.0
        self.mask_y = np.zeros_like(y)
        self.mask_y[(self.y_scaled >= -1) & (self.y_scaled <= +1)] = 1.0

        #mask[
             #(self.XX_scaled >= -1) & (self.XX_scaled <= +1)
             #& (self.YY_scaled >= -1) & (self.YY_scaled <= +1)
            #] = 1.0
        self.mask = np.outer(self.mask_y, self.mask_x)

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

        n, m = self.single_index_to_n_m(i)

        # the Legendres are a product basis L_n(x)*L_m(y) - using Fubini's theorem, the
        # normalization can thus be written as the product of norms norm(L_n)*norm(L_m)
        norm = self.a*self.b*4./((2*n + 1.)*(2*m + 1.))

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

    # TODO: use Gram-Schmidt directly? Or rather use QR?

    def __init__(self, base_type, x, y):
        # compute all basis functions and store those as a class member

        # run orthogonalisation:

        pass

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
