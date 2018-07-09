#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for Jones calculus.
"""

# Jones vectors:
J_H = np.array([1., 0])  # horizontal
J_V = np.array([0, 1.])  # vertical
J_Lp45 = 1./np.sqrt(2)*np.array([1., 1])  # diagonal; linear +45°
J_Lm45 = 1./np.sqrt(2)*np.array([1., -1])  # diagonal; linear -45°
J_RHCP = 1./np.sqrt(2)*np.array([1., -1j])  # right-hand circular polarized
J_LHCP = 1./np.sqrt(2)*np.array([1., +1j])  # left-hand circular polatized



J_WP = np.array([1., 0], [0, np.exp(-1j*)])  # Jones matrix for wave plate
J_QWP = np.array()  # Jones matrix for quarter-wave plate
J_HWP = np.array()  # Jones matrix for half-wave plate

# Jones matrix for rotator:
J_rot = lambda phi: np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

J_rot_polarizer = lambda phi:



def J_polarizer(p_H, p_V):
    """Jones matrix for polarizer.

    Parameters
    ----------
    p_H, p_V : numbers
        Polarization (0 <= p_i <= 1).

    Returns
    -------
    """

    pass
