#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for Jones calculus.

Conventions are the same as in mueller.py: Jones vectors are [E_x, E_y], the
Stokes parameter S_3 = 2 Im(conj(E_x)*E_y) is positive for J_RHCP, and an
element with Jones matrix J rotated by an angle phi has the Jones matrix
J_rot(phi).T @ J @ J_rot(phi). Retarders delay the y component, i.e. their
fast axis is along x before rotation.
"""

import numpy as np

# Jones vectors:
J_H = np.array([1., 0])  # horizontal
J_V = np.array([0, 1.])  # vertical
J_Lp45 = 1./np.sqrt(2)*np.array([1., 1])  # diagonal; linear +45°
J_Lm45 = 1./np.sqrt(2)*np.array([1., -1])  # diagonal; linear -45°
J_RHCP = 1./np.sqrt(2)*np.array([1., +1j])  # right-hand circular polarized (S_3 = +1, see mueller.S_RCP)
J_LHCP = 1./np.sqrt(2)*np.array([1., -1j])  # left-hand circular polarized (S_3 = -1, see mueller.S_LCP)

# Jones matrix for rotator:
J_rot = lambda phi: np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])


def rotate(J, phi):
    """Return Jones matrix of an element J rotated by the angle phi.
    """

    return J_rot(phi).T @ J @ J_rot(phi)


def J_WP(delta, phi=0.0):
    """Jones matrix for a wave plate (retarder).

    Parameters
    ----------
    delta : number
        Phase retardance [rad].
    phi : number, optional
        Angle of fast axis.

    Returns
    -------
    J : array
    """

    return rotate(np.array([[1., 0], [0, np.exp(1j*delta)]]), phi)


J_QWP = J_WP(np.pi/2)  # Jones matrix for quarter-wave plate (fast axis horizontal)
J_HWP = J_WP(np.pi)  # Jones matrix for half-wave plate (fast axis horizontal)


def J_polarizer(p_H, p_V):
    """Jones matrix for polarizer.

    Parameters
    ----------
    p_H, p_V : numbers
        Amplitude transmission for horizontal and vertical polarization
        (0 <= p_i <= 1).

    Returns
    -------
    J : array
    """

    return np.array([[p_H, 0.], [0., p_V]])


# Jones matrix for ideal linear polarizer with transmission axis at angle phi:
J_rot_polarizer = lambda phi: rotate(J_polarizer(1., 0.), phi)
