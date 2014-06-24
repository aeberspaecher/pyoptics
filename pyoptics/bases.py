#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module for basis sets.

In optics, different basis sets are used to describe fields in the plane, e.g.
Zernike polynomials in the description of aberrations in systems with circular
aperture stops. This module offers an object oriented approach such basis sets.
"""


from .utils import scalar_product


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

        pass

    def __call__(self, coeffs):
        """Return a field computed from the given expansion coefficients.
        """

        # distinguish scalar coeff (np.isscalar) and array

        pass

    def eval_single(self, index):
        """Return a single basis function.
        """

        pass

    def coeffs_from_field(self, field, method="LSTSQ"):
        """Expand a given field in the basis set.
        """

        # TODO: offer expansion by least square fits and expansion by evalutaion of scalar products

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


class FringeZernikes(BasisSet):
    def R_nm(self, n, m):
        pass

    # TODO: find map (n, m) --> fringe index


class NumericallyOrthogonalized(BasisSet):
    """Represents a basis set that is numerically orthogonalized through a
    Gram-Schmidt procedure on the given grid.
    """

    def __init__(self, base_type, x, y):
        # compute all basis functions and store those as a class member

        # run orthogonalisation:

        pass
