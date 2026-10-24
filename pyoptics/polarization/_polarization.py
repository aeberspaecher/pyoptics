#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for computations involving polarization: TE/TM (s/p) bases for
plane waves, decomposition of partially polarized light into Jones vectors,
degree of polarization and intensity in the preferred state.

Conventions as in jones.py and mueller.py: Jones vectors are [E_x, E_y],
fields vary as exp(i(k.r - omega*t)).

TE/TM refer to the plane of incidence of a plane wave with the transverse
direction (sigma_x, sigma_y) relative to the z axis (e.g. a point in the
pupil of a lens): the TE (s) unit vector is perpendicular to the plane of
incidence, e_TE = (-sin(phi), cos(phi)), the TM (p) unit vector lies in it,
e_TM = (cos(phi), sin(phi)), phi = atan2(sigma_y, sigma_x). Both are the
projections onto the x-y plane, i.e. the Jones vectors in the pupil that
aplanatic focusing (propagators.aplanatic_rotation()) turns into the s and p
polarized plane waves. On the axis (sigma = 0) the plane of incidence is
undefined; there phi = 0 is used (TE = y, TM = x).
"""

import numpy as np

from .mueller import stokes_from_complex_E


def TE_TM_basis(sigma_x, sigma_y):
    """Return the TE and TM Jones unit vectors for the direction (sigma_x,
    sigma_y).

    Parameters
    ----------
    sigma_x, sigma_y : numbers or arrays
        Transverse direction (only its azimuth matters).

    Returns
    -------
    e_TE, e_TM : arrays
        Shape (2,) + np.shape(sigma_x): e_TE = (-sin(phi), cos(phi)),
        e_TM = (cos(phi), sin(phi)).
    """

    phi = np.arctan2(sigma_y, sigma_x)
    s, c = np.sin(phi), np.cos(phi)

    return np.array([-s, c]), np.array([c, s])


def lin_pol_to_TE_TM(field, sigma_x, sigma_y):
    """Convert [E_x, E_y] data to [E_TE, E_TM].

    Parameters
    ----------
    field : array
        Jones vector(s) with the components along the first axis, shape
        (2, ...), or a field of shape (Ny, Nx, 2) when sigma_x, sigma_y have
        the shape (Ny, Nx).
    sigma_x, sigma_y : numbers or arrays
        Transverse direction of the plane wave(s), see TE_TM_basis().

    Returns
    -------
    field_TE_TM : array
        Same shape as field.
    """

    E_x, E_y, stack = _components(field, sigma_x)
    e_TE, e_TM = TE_TM_basis(sigma_x, sigma_y)

    return _assemble(e_TE[0]*E_x + e_TE[1]*E_y, e_TM[0]*E_x + e_TM[1]*E_y, stack)


def TE_TM_pol_to_lin(field, sigma_x, sigma_y):
    """Convert [E_TE, E_TM] data to [E_x, E_y]; inverse of lin_pol_to_TE_TM().
    """

    E_TE, E_TM, stack = _components(field, sigma_x)
    e_TE, e_TM = TE_TM_basis(sigma_x, sigma_y)

    return _assemble(e_TE[0]*E_TE + e_TM[0]*E_TM, e_TE[1]*E_TE + e_TM[1]*E_TM, stack)


def _components(field, sigma_x):
    """Split field into its two components; stack is True for (Ny, Nx, 2)."""

    field = np.asarray(field)
    stack = (field.ndim == np.ndim(sigma_x) + 1 and field.ndim >= 3 and field.shape[-1] == 2)
    if stack:
        return field[..., 0], field[..., 1], True

    return field[0], field[1], False


def _assemble(a, b, stack):
    return np.stack([a, b], axis=-1 if stack else 0)


def coherency_matrix(S):
    """Return the 2x2 coherency matrix <E E^H> of a Stokes vector S.
    """

    S0, S1, S2, S3 = S

    # S_2 = 2 Re(E_x conj(E_y)), S_3 = 2 Im(conj(E_x) E_y), see mueller.py:
    return 0.5*np.array([[S0 + S1, S2 - 1j*S3],
                         [S2 + 1j*S3, S0 - S1]])


def jones_from_stokes(S):
    """Decompose (partially polarized) light into two mutually incoherent,
    orthogonal Jones vectors.

    The coherency matrix of S is diagonalized; its eigenvectors are the two
    polarization states and its eigenvalues their intensities. Unpolarized
    light S = [1, 0, 0, 0] gives two orthogonal states with weight 1/2 each
    (x and y), fully polarized light one state with weight S_0 and one with
    weight 0.

    Parameters
    ----------
    S : array
        Stokes vector [S_0, S_1, S_2, S_3].

    Returns
    -------
    weights : array
        Intensities of the two states, shape (2,), largest first; they add
        up to S_0.
    J : array
        Normalized Jones vectors, shape (2, 2): J[i] belongs to weights[i].
        Summing weights[i]*mueller.stokes_from_complex_E(J[i]) gives S.
    """

    w, v = np.linalg.eigh(coherency_matrix(S))
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)
    J = v[:, order].T

    # remove the arbitrary global phase: make the larger component real
    for i in range(2):
        big = np.argmax(np.abs(J[i]))
        J[i] *= np.exp(-1j*np.angle(J[i][big]))

    return w, J


def DOP(field):
    """Degree of polarization of a field.

    Parameters
    ----------
    field : array
        Jones vectors of shape (2, ...) (e.g. a field [E_x, E_y] sampled on a
        grid, with the components along the first axis); the Stokes
        parameters are summed (incoherent average) over all other axes.

    Returns
    -------
    DOP : number
    """

    S = stokes_from_complex_E(np.asarray(field))
    S = S.reshape(4, -1).sum(axis=1)

    return np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)/S[0]


def IPS(field, pref_state):
    """Intensity in the preferred state: the fraction of the intensity
    that a perfect analyzer for the Jones vector pref_state transmits.

    Parameters
    ----------
    field : array
        Jones vectors of shape (2, ...), summed incoherently as in DOP().
        A Stokes vector (shape (4,)) is accepted as well.
    pref_state : array
        Jones vector of the preferred state.

    Returns
    -------
    IPS : number
        Between 0 and 1; 1/2 for unpolarized light.
    """

    p = np.asarray(pref_state, dtype=complex)
    p = p/np.linalg.norm(p)
    field = np.asarray(field)

    if field.shape == (4,):
        C = coherency_matrix(field)
        return np.real(np.conj(p) @ C @ p)/np.real(np.trace(C))

    E = field.reshape(2, -1)
    projected = np.abs(np.conj(p[0])*E[0] + np.conj(p[1])*E[1])**2

    return np.sum(projected)/np.sum(np.abs(E)**2)
