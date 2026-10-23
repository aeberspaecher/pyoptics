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
    """Rotation matrix that rotates vector a onto vector b.

    a and b do not need to be unit vectors; the routine normalizes copies of
    them.

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

    a = np.asarray(a, dtype=float)/norm(a)
    b = np.asarray(b, dtype=float)/norm(b)

    v = cross_prod(a, b)
    c = np.dot(a, b)

    if np.isclose(c, -1.0):  # antiparallel: rotate by pi about any axis perpendicular to a
        u = cross_prod(a, np.eye(3)[np.argmin(np.abs(a))])
        u /= norm(u)
        return 2*np.outer(u, u) - np.eye(3)

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
