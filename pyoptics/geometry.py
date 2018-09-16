#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Geometry helper functions.
"""

import numpy as np
from scipy.linalg import norm

from .utils import sin_cos


def cross_prod(a, b):
    """Cross product of two vectors a and b.

    Parameters
    ----------
    a, b : arrays

    Returns
    -------
    res : array
        Cross-product of a and b (using right-hand rule).
    """

    v_a = cross_prod_matrix(a)

    return np.dot(v_a, b)


def cross_prod_matrix(v):
    """Compute the cross product matrix from given vector.

    Parameters
    ----------
    v : array

    Returns
    -------
    v_x : array
        Matrix v_x with dot(v_x, b) = v x b.
    """

    v_x = np.array([ [0., -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]
                   ])

    return v_x


def R_onto(a, b):
    """Rotation matrix that rotations vector a onto vector b.

    a and b are assumed the be unit vectors, however the routine also works
    a and b of arbitraty length.

    Parameters
    ----------
    a, b : arrays

    Returns
    -------
    R : array
        Rotation matrix that maps a unnit vector along a onto the direction
        of b.
    """
    #http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    # TODO: catch case of a, b (anti-)parallel?

    # guarantee a and b are unit vectors:
    a /= np.sqrt(norm(a))
    b /= np.sqrt(norm(b))

    v = cross_prod(a, b)
    s = norm(v)
    c = np.dot(a, b)
    v_x = cross_prod_matrix(v)

    R = np.eye(3) + v_x + 1./(1+c)*np.dot(v_x, v_x)

    return R


def R_x(theta):
    """Elementary rotation about x axis.

    Parameters
    ----------
    theta : number
        Rotation angle

    Returns
    -------
    R_x : array
    """

    s, c = sin_cos(theta)
    Rx = np.array([ [1., 0, 0],
                    [0, c, -s],
                    [0, s, c]
                  ])
    return Rx


def R_y(theta):
    """Elementary rotation about y axis.

    Parameters
    ----------
    theta : number
        Rotation angle

    Returns
    -------
    R_y : array
    """

    s, c = sin_cos(theta)
    Ry = np.array([ [c, 0, s],
                    [0, 1., 0],
                    [-s, 0, c]
                  ])
    return Ry


def R_z(theta):
    """Elementary rotation about z axis.

    Parameters
    ----------
    theta : number
        Rotation angle

    Returns
    -------
    R_z : array
    """

    s, c = sin_cos(theta)
    Rz = np.array([ [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                  ])
    return Rz
