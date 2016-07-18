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
from scipy.linalg import lstsq

from pyoptics.utils import scalar_product, kronecker_delta, simpson_weights


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
        M  = np.zeros([N_samples, N_indices])
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
            norms = [self.normalization_factor(i)  for i in indices]
        else:
            norms = len(indices)*(1.0,)

        coeffs = 1.0/np.array(norms)*np.array([scalar_product(self.eval_single(i), self.mask*field, self.x, self.y, simpson_weights) for i in indices])

        return coeffs


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
