#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module for basis sets.

In optics, different basis sets are used to describe fields in the plane, e.g.
Zernike polynomials in the description of aberrations in systems with circular
aperture stops. This module offers an object oriented approach such basis sets.
"""

from math import sqrt, floor, ceil

import numpy as np
from scipy.misc import factorial as fac  # tolerant to negative arguments!

from pyoptics.utils import scalar_product


class BasisSet(object):
    basis_size = 0  # number of coefficients to use
    mask = None  # binary mask that defines the basis functions' support
    # this is needed for basis functions that are defined on non-rectangular
    # domains, e.g. the Zernike polynomials that have a circular domain of
    # definition. the specific mask is used in scalar products.

    def __init__(self, x, y):
        """Define grid the basis functions are defined on.
        """
        # TODO: implement here or in subclasses?
        self.x = x
        self.y = y
        self.XX, self.YY = np.meshgrid(x, y)

    def __call__(self, coeffs):
        """Return a field computed from the given expansion coefficients.
        """

        # distinguish scalar coeff (np.isscalar) and array

        # TODO: give __mul__() to evaluated objects (scalar product)?

        pass

    def eval_single(self, index):
        """Return a single basis function.
        """

        pass

    def coeffs_from_field(self, field, method="LSTSQ"):
        """Expand a given field in the basis set.
        """

        # TODO: offer expansion by least square fits and expansion by evaluation of scalar products

        pass

    def scalar_product(self, field, index):
        """Compute the scalar product of a given field and a given basis
        function.

        The implemenation here allows to use scalar products with different
        weight functions.
        """

        # TODO: all scalar products are the same - except for the weight function!
        # implement only the weight function and extend the scalar_product() in
        # the utils module by an optional weight function

        # TODO: implement standard scalar product in base class, override in
        # subclasses

        # scalar_product() is available in utils

        pass


class PreSampledBasisSet(BasisSet):
    pass


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
        mask[self.Rho < 1.0] = 1.0
        self.mask = mask

    def R_nm(self, n, m):
        # TODO: factor out?
        #summands = [(-1)**k * binom(n-k, k) * binom(n-2*k, (n-m)//2.-k) * self.Rho**(n-2.0*k) for k in range(0, int((n-m)//2 + 1))]
        #R = sum(summands)

        N, M = int(n), int(m)

        R = np.zeros_like(self.Rho)
        for k in range((N-M)/2 + 1):
            R += (-1.0)**k * fac(N-k) / ( fac(k) * fac( (N+M)/2.0 - k ) * fac( (N-M)/2.0 - k ) ) * self.Rho**(n-2.0*k)

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

        print("j = {}: n = {}; m = {}; d = {}".format(j, n, m, d))

        return n, m

    def eval_single(self, j):
        n, m = self.fringe_to_n_m(j)
        R = self.R_nm(n, abs(m))
        Y = self.Y_m(m)

        return R*Y*self.mask


class Polynomials(BasisSet):
    """Polynomials in x and y.
    """

    pass


def poly_basis_size(order):
    # Number of nonomials up to given order.

    pass


class NumericallyOrthogonalized(BasisSet):
    """Represents a basis set that is numerically orthogonalized through a
    Gram-Schmidt procedure on the given grid.
    """

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
