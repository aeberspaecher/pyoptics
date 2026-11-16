#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Crystal optics: plane waves in anisotropic media, the 4 x 4 matrix
method for stacks of anisotropic layers, double refraction at crystal
surfaces, polarizing prisms, optical activity and Faraday rotation
(notebook 138, crystal optics; used by notebooks
113 (Fabry–Perot and filters), 139 (electro-optic modulators) and 140
(liquid crystals)).

Conventions as in the rest of pyoptics: fields vary as
exp(i(k.r - omega*t)), wave vectors are given in units of the vacuum wave
number k0 = 2 pi/lambda (so that the length of k is the index of the
wave), lengths in any unit consistent with the wavelength. A medium is
described by its relative dielectric tensor eps (3 x 3, complex for
absorbing media, Hermitian for magneto-optic ones); mu = 1.

Plane waves
-----------
A plane wave with the unit wave normal s in a medium with the tensor
eps has the displacement D perpendicular to s. Its index n follows from
the eigenvalue problem of the index ellipsoid,

    P eps^-1 P D = D/n**2,   P = 1 - s s^T,

restricted to the plane perpendicular to s: two waves with orthogonal D,
the field E = eps^-1 D (not perpendicular to s in general) and the
Poynting vector S ~ E x (s x E), which deviates from s by the walk-off
angle (`wave_indices()`, `walkoff()`). `uniaxial_eps()` and
`biaxial_eps()` build the tensors, `rotate_eps()` turns them.

The 4 x 4 matrix method
-----------------------
For stratified media (layers parallel to the x-y plane) and a plane of
incidence x-z, the tangential wave vector xi = n_in sin(theta_in) is
conserved, and the tangential fields psi = (E_x, H_y, E_y, H_x) (H in
units of 1/Z0) obey Berreman's equation

    d psi/dz = i k0 Delta(eps, xi) psi,

whose four eigenvalues q are the normal components k_z/k0 of the two
forward and two backward waves (`delta_matrix()`, `modes()`). A layer of
thickness d multiplies psi by exp(i k0 Delta d). `stack_jones()` solves a
stack between two isotropic media for the Jones matrices of transmission
and reflection, either coherently (all multiple reflections) or
forward-only (each interface transmits once, reflections are lost: the
"extended Jones matrix" method, right for thick plates and for slices of
a liquid crystal, where the multiple reflections between plate surfaces
would only add Fabry–Perot fringes). The Jones matrices act on
(p, s) amplitudes (p first, so that at normal incidence the order is
(x, y) as in `polarization.jones`); with `azimuth` they are projected to
the lab x-y basis. Amplitudes are the lengths of E, the transmitted power
is abs(t)**2 q_out/q_in with q = sqrt(n**2 - xi**2) (`power_factor()`).
In reflection the p amplitude is counted along +x for both directions,
so r_p = r_s at normal incidence, as in `thin_films`.

Interfaces and prisms
---------------------
`refract()` splits a plane wave at a plane interface between any two
media into reflected and transmitted waves with their power fractions,
found with the same eigenmodes (an evanescent wave carries no power: total
internal reflection). `trace_interfaces()` follows all branches through a
sequence of plane interfaces, which is all a polarizing prism is
(`glan_prism()`, `wollaston()`, `rochon()`).

Optical activity and Faraday rotation
-------------------------------------
A rotation by the angle a is the Jones matrix [[cos a, -sin a], [sin a,
cos a]] in the lab frame, counterclockwise from x to y as seen by an
observer looking towards the source (z towards the observer). Natural
optical activity is reciprocal: in the lab frame a backward wave is
rotated back, and a round trip cancels. Faraday rotation (the magnetic
field B along z) is not: the rotation adds up on the way back
(`rotator()`). In the 4 x 4 method the Faraday effect is the gyrotropic
tensor eps_xy = -eps_yx = i g (`gyrotropic_eps()`, `faraday_g()`).
Dextrorotatory (right-handed) quartz turns the plane clockwise for the
observer, i.e. by -rho d in the convention above (`quartz_rotatory_power()`).
A plate with both linear and circular birefringence (quartz off its axis)
has elliptical eigenstates (`elliptical_retarder()`).

References
----------
D. W. Berreman, "Optics in stratified and anisotropic media: 4 x 4-matrix
formulation", J. Opt. Soc. Am. 62, 502-510 (1972).

P. Yeh, "Optical Waves in Layered Media", Wiley (1988), chapter 9.

A. Lien, "Extended Jones matrix representation for the twisted nematic
liquid-crystal display at oblique incidence", Appl. Phys. Lett. 57,
2767-2769 (1990).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), chapter 15.

A. Yariv and P. Yeh, "Optical Waves in Crystals", Wiley (1984), chapters
4 and 5.

J. M. Bennett, "Polarizers", in "Handbook of Optics", Vol. II, 2nd ed.,
McGraw-Hill (1995), chapter 3.

G. Ghosh, "Dispersion-equation coefficients for the refractive index and
birefringence of calcite and quartz crystals", Opt. Commun. 163, 95-102
(1999).
"""

from collections import namedtuple

import numpy as np

from .arvr import quartz_indices


__all__ = ["calcite_indices", "quartz_indices", "quartz_rotatory_power", "tgg_verdet",
           "uniaxial_eps", "biaxial_eps", "rotation_matrix", "rotate_eps", "gyrotropic_eps",
           "faraday_g", "director", "wave_indices", "walkoff", "delta_matrix", "modes",
           "isotropic_modes", "stack_jones", "power_factor", "refract", "trace_interfaces",
           "glan_prism", "wollaston", "rochon", "rotator", "elliptical_retarder", "isolator",
           "Wave"]


# --------------------------------------------------------------------------
# materials

def calcite_indices(wavelength_nm):
    """Ordinary and extraordinary indices of calcite.

    Sellmeier fit of G. Ghosh, Opt. Commun. 163, 95 (1999), valid from 0.2
    to 2.2 µm.

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    n_o, n_e : arrays
        1.6583 and 1.4861 at 589.3 nm (handbook values 1.6584 and 1.4864).
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2
    n_o = np.sqrt(1.73358749 + 0.96464345*l2/(l2 - 1.94325203e-2) + 1.82831454*l2/(l2 - 120.0))
    n_e = np.sqrt(1.35859695 + 0.82427830*l2/(l2 - 1.06689543e-2) + 0.14429128*l2/(l2 - 120.0))

    return n_o, n_e


def quartz_rotatory_power(wavelength_nm):
    """Rotatory power of crystalline quartz along its optic axis.

    A Drude formula rho = A/(lambda**2 - lambda_0**2) fitted to 21.72°/mm
    at 589.3 nm and 17.32°/mm at 656.3 nm (A = 7.13 °µm²/mm, lambda_0 =
    0.137 µm); it gives 49.2°/mm at 404.7 nm (measured 48.9).

    Parameters
    ----------
    wavelength_nm : number or array

    Returns
    -------
    rho : array
        Degrees per mm, positive for right-handed quartz (clockwise for an
        observer looking towards the source).
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2

    return 7.133/(l2 - 0.0189)


def tgg_verdet(wavelength_nm, temperature=293.0):
    """Verdet constant of terbium gallium garnet (TGG).

    A/(lambda**2 - lambda_0**2) through -134 rad/(T m) at 632.8 nm and -40
    rad/(T m) at 1064 nm (lambda_0 = 0.30 µm, the 4f-5d transitions of
    Tb3+), scaled as 1/T for the paramagnetic crystal.

    Parameters
    ----------
    wavelength_nm : number or array
    temperature : number, optional
        Kelvin (293 K for the values above).

    Returns
    -------
    V : array
        rad/(T m); the sign follows the usual tables (negative: the
        rotation is counterclockwise about B, seen with B pointing at the
        observer). Only the magnitude matters for an isolator.
    """

    l2 = (np.asarray(wavelength_nm, dtype=float)*1e-3)**2

    return -41.7/(l2 - 0.089)*293.0/temperature


# --------------------------------------------------------------------------
# tensors

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def uniaxial_eps(n_o, n_e, axis=(0.0, 0.0, 1.0)):
    """Dielectric tensor of a uniaxial medium.

    Parameters
    ----------
    n_o, n_e : numbers or arrays
        Ordinary and extraordinary (complex) indices.
    axis : (..., 3) array, optional
        Optic axis (normalized here).

    Returns
    -------
    eps : (..., 3, 3) complex array
        n_o**2 (1 - a a^T) + n_e**2 a a^T.
    """

    a = _unit(axis)
    n_o = np.asarray(n_o, dtype=complex)[..., None, None]
    n_e = np.asarray(n_e, dtype=complex)[..., None, None]
    aa = a[..., :, None]*a[..., None, :]

    return n_o**2*(np.eye(3) - aa) + n_e**2*aa


def director(tilt, twist):
    """Unit vector (cos tilt cos twist, cos tilt sin twist, sin tilt): the
    director of a liquid crystal with the tilt from the x-y plane and the
    twist (azimuth) from x.

    Parameters
    ----------
    tilt, twist : numbers or arrays
        Radians.

    Returns
    -------
    n : (..., 3) array
    """

    tilt, twist = np.broadcast_arrays(np.asarray(tilt, dtype=float), np.asarray(twist, dtype=float))

    return np.stack([np.cos(tilt)*np.cos(twist), np.cos(tilt)*np.sin(twist), np.sin(tilt)], axis=-1)


def rotation_matrix(axis, angle):
    """Rotation by `angle` (radians, right-handed) about `axis`.

    Parameters
    ----------
    axis : 3 vector
    angle : number or array

    Returns
    -------
    R : (..., 3, 3) array
    """

    k = _unit(axis)
    angle = np.asarray(angle, dtype=float)[..., None, None]
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def rotate_eps(eps, R):
    """Tensor of a medium turned by the rotation R: R eps R^T.

    Parameters
    ----------
    eps : (..., 3, 3) array
    R : (..., 3, 3) array

    Returns
    -------
    eps_rotated : (..., 3, 3) array
    """

    R = np.asarray(R)

    return R @ np.asarray(eps) @ np.swapaxes(R, -1, -2)


def biaxial_eps(n_x, n_y, n_z, R=None):
    """Dielectric tensor of a biaxial medium with the principal indices
    n_x, n_y, n_z, optionally turned by the rotation R.

    Parameters
    ----------
    n_x, n_y, n_z : numbers
    R : (3, 3) array, optional

    Returns
    -------
    eps : (3, 3) complex array
    """

    eps = np.diag(np.array([n_x, n_y, n_z], dtype=complex)**2)

    return eps if R is None else rotate_eps(eps, R)


def gyrotropic_eps(eps, g, direction=(0.0, 0.0, 1.0)):
    """Add the magneto-optic (Faraday) gyration to a tensor.

    eps_ij + i g e_ijk b_k: for b along z, eps_xy = i g, eps_yx = -i g. The
    circular eigenwaves along b have n**2 = eps -+ g; a positive g rotates
    linear polarization counterclockwise (from x to y) for a wave along +b
    and also for a wave along -b (non-reciprocal).

    Parameters
    ----------
    eps : (..., 3, 3) array
    g : number or array
        Gyration (see `faraday_g()`).
    direction : 3 vector, optional
        Direction b of the magnetic field.

    Returns
    -------
    eps_g : (..., 3, 3) complex array
    """

    b = _unit(direction)
    G = np.array([[0, b[2], -b[1]], [-b[2], 0, b[0]], [b[1], -b[0], 0]])

    return np.asarray(eps, dtype=complex) + 1j*np.asarray(g)[..., None, None]*G


def faraday_g(rotation_per_length, wavelength, n):
    """Gyration g that rotates linear polarization by `rotation_per_length`
    (radians per unit length, positive counterclockwise from x to y for
    propagation along the field) in a medium of index n:
    g = 2 n rotation_per_length/k0 (to first order in g).

    Parameters
    ----------
    rotation_per_length : number
        V B for a Verdet constant V (consistent length units).
    wavelength : number
    n : number

    Returns
    -------
    g : number
    """

    return 2*n*rotation_per_length*wavelength/(2*np.pi)


# --------------------------------------------------------------------------
# plane waves in a homogeneous medium

def _transverse_basis(s):
    s = _unit(s)
    helper = np.where(np.abs(s[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    u = _unit(np.cross(helper, s))
    v = np.cross(s, u)
    return s, u, v


def wave_indices(eps, direction):
    """Indices and fields of the two plane waves with the wave normal
    `direction` (the index ellipsoid).

    Parameters
    ----------
    eps : (..., 3, 3) array
    direction : (..., 3) array
        Wave normal (normalized here).

    Returns
    -------
    n : (..., 2) array
        Indices, sorted ascending (the fast wave first).
    D : (..., 2, 3) complex array
        Unit displacement vectors (perpendicular to the wave normal).
    E : (..., 2, 3) complex array
        Electric fields eps^-1 D, normalized.
    S : (..., 2, 3) array
        Unit Poynting vectors.
    """

    eps = np.asarray(eps, dtype=complex)
    s, u, v = _transverse_basis(direction)
    B = np.stack([u, v], axis=-1).astype(complex)                     # (..., 3, 2)
    inv = np.linalg.inv(eps)
    M = np.swapaxes(B, -1, -2) @ inv @ B                              # (..., 2, 2)
    w, c = np.linalg.eig(M)                                           # w = 1/n**2
    n = 1/np.sqrt(w)
    order = np.argsort(n.real, axis=-1)
    n = np.take_along_axis(n, order, axis=-1)
    c = np.take_along_axis(c, order[..., None, :], axis=-1)
    D = np.swapaxes(B @ c, -1, -2)                                    # (..., 2, 3)
    D = D/np.linalg.norm(D, axis=-1, keepdims=True)
    E = np.einsum("...ij,...mj->...mi", inv, D)
    E = E/np.linalg.norm(E, axis=-1, keepdims=True)
    H = np.cross(s[..., None, :], E)*n[..., None]
    S = np.real(np.cross(E, np.conj(H)))
    S = S/np.linalg.norm(S, axis=-1, keepdims=True)

    return n, D, E, S


def walkoff(eps, direction):
    """Walk-off angles between the wave normal and the Poynting vectors of
    the two waves of `wave_indices()`.

    Parameters
    ----------
    eps : (..., 3, 3) array
    direction : (..., 3) array

    Returns
    -------
    rho : (..., 2) array
        Radians (0 for an ordinary wave).
    """

    s = _unit(direction)
    S = wave_indices(eps, s)[3]

    return np.arccos(np.clip(np.einsum("...j,...mj->...m", s, S), -1.0, 1.0))


# --------------------------------------------------------------------------
# the 4 x 4 matrix method

def delta_matrix(eps, xi):
    """Berreman's matrix for psi = (E_x, H_y, E_y, H_x), plane of
    incidence x-z: d psi/dz = i k0 Delta psi.

    Parameters
    ----------
    eps : (..., 3, 3) array
    xi : number or array
        Tangential wave vector k_x/k0.

    Returns
    -------
    Delta : (..., 4, 4) complex array
    """

    e = np.asarray(eps, dtype=complex)
    xi = np.asarray(xi, dtype=complex)
    shape = np.broadcast_shapes(e.shape[:-2], xi.shape)
    e = np.broadcast_to(e, shape + (3, 3))
    xi = np.broadcast_to(xi, shape)
    ezz = e[..., 2, 2]
    D = np.zeros(shape + (4, 4), dtype=complex)
    D[..., 0, 0] = -xi*e[..., 2, 0]/ezz
    D[..., 0, 1] = 1 - xi**2/ezz
    D[..., 0, 2] = -xi*e[..., 2, 1]/ezz
    D[..., 1, 0] = e[..., 0, 0] - e[..., 0, 2]*e[..., 2, 0]/ezz
    D[..., 1, 1] = -e[..., 0, 2]*xi/ezz
    D[..., 1, 2] = e[..., 0, 1] - e[..., 0, 2]*e[..., 2, 1]/ezz
    D[..., 2, 3] = -1.0
    D[..., 3, 0] = -e[..., 1, 0] + e[..., 1, 2]*e[..., 2, 0]/ezz
    D[..., 3, 1] = e[..., 1, 2]*xi/ezz
    D[..., 3, 2] = -e[..., 1, 1] + e[..., 1, 2]*e[..., 2, 1]/ezz + xi**2

    return D


def _flux(V):
    """z component of the Poynting vector Re(E_x H_y* - E_y H_x*)/2 of the
    columns of V (..., 4, m)."""
    return 0.5*np.real(V[..., 0, :]*np.conj(V[..., 1, :]) - V[..., 2, :]*np.conj(V[..., 3, :]))


def modes(eps, xi, normalize=True):
    """Eigenwaves of a homogeneous layer for the tangential wave vector xi.

    Parameters
    ----------
    eps : (..., 3, 3) array
    xi : number or array
    normalize : bool, optional
        Scale propagating modes to unit power flux along z (evanescent ones
        to a unit field vector).

    Returns
    -------
    q : (..., 4) complex array
        k_z/k0 of the waves, ordered (forward, forward, backward,
        backward); the forward waves sorted by Re q (the faster wave,
        smaller q, first).
    V : (..., 4, 4) complex array
        The columns are the tangential fields psi of the waves.
    """

    q, V = np.linalg.eig(delta_matrix(eps, xi))
    S = _flux(V)
    evan = np.abs(q.imag) > 1e-9*np.maximum(np.abs(q), 1.0)
    fwd = np.where(evan, q.imag > 0, S > 0)
    # forward first (sorted by Re q), backward after (sorted by -Re q)
    key = np.where(fwd, 0.0, 1e6) + np.where(fwd, q.real, -q.real)
    order = np.argsort(key, axis=-1, kind="stable")
    q = np.take_along_axis(q, order, axis=-1)
    V = np.take_along_axis(V, order[..., None, :], axis=-1)
    if normalize:
        S = np.abs(_flux(V))
        evan = np.abs(q.imag) > 1e-9*np.maximum(np.abs(q), 1.0)
        scale = np.where(evan | (S < 1e-30), np.linalg.norm(V, axis=-2), np.sqrt(np.where(S > 0, S, 1.0)))
        V = V/scale[..., None, :]

    return q, V


def isotropic_modes(n, xi):
    """Waves (p+, s+, p-, s-) of an isotropic medium, amplitudes the
    lengths of E (p counted along +x for both directions).

    Parameters
    ----------
    n : number or array
        Complex index.
    xi : number or array

    Returns
    -------
    q : (..., 4) complex array
    V : (..., 4, 4) complex array
    """

    n = np.asarray(n, dtype=complex)
    xi = np.asarray(xi, dtype=complex)
    n, xi = np.broadcast_arrays(n, xi)
    q = np.sqrt(n**2 - xi**2)
    q = np.where(q.imag < 0, -q, q)
    c = q/n                                                       # cos(theta)
    V = np.zeros(n.shape + (4, 4), dtype=complex)
    # p forward: E = (c, 0, -s), H_y = n
    V[..., 0, 0], V[..., 1, 0] = c, n
    # s forward: E_y = 1, H_x = -q
    V[..., 2, 1], V[..., 3, 1] = 1.0, -q
    # p backward: E_x = c, H_y = -n
    V[..., 0, 2], V[..., 1, 2] = c, -n
    # s backward: E_y = 1, H_x = +q
    V[..., 2, 3], V[..., 3, 3] = 1.0, q

    return np.stack([q, q, -q, -q], axis=-1), V


def _layer_eps(layer):
    eps = np.asarray(layer, dtype=complex)
    if eps.ndim == 0 or eps.shape[-2:] != (3, 3):
        eps = eps[..., None, None]**2*np.eye(3)
    return eps


def _rotate_z(eps, phi):
    """Tensor in the frame of a plane of incidence at the azimuth phi."""
    return rotate_eps(eps, rotation_matrix((0.0, 0.0, 1.0), -np.asarray(phi)))


def _solve(A, b):
    return np.linalg.solve(A, b)


def stack_jones(layers, wavelength, xi=0.0, n_in=1.0, n_out=1.0, coherent=True, azimuth=None):
    """Jones matrices of a stack of anisotropic layers between isotropic
    media (the 4 x 4 matrix method).

    Parameters
    ----------
    layers : list of (eps, thickness)
        From the incidence side. eps is a (..., 3, 3) tensor or an
        isotropic (complex) index; it broadcasts with xi and azimuth.
    wavelength : number or array
        Same unit as the thicknesses (an array broadcasts like xi).
    xi : number or array, optional
        n_in sin(theta_in).
    n_in, n_out : numbers, optional
        Indices of the incidence and exit media.
    coherent : bool, optional
        True: all multiple reflections (Berreman). False: forward-only,
        every interface transmits once and reflected light is lost (thick
        plates, sliced liquid crystals); r is then the reflection of the
        first interface only.
    azimuth : number or array, optional
        Azimuth of the plane of incidence from the lab x axis (radians).
        Given, the layers are turned into the plane of incidence and t, r
        are projected to the lab x-y basis (p along (cos, sin), s along
        (-sin, cos)); otherwise t, r are in the (p, s) basis.

    Returns
    -------
    t, r : (..., 2, 2) complex arrays
        Jones matrices of transmission and reflection (output component,
        input component); amplitudes are lengths of E, see
        `power_factor()` for powers.
    """

    k0 = (2*np.pi/np.asarray(wavelength, dtype=float))[..., None]
    xi = np.asarray(xi, dtype=float)
    eps_list = [_layer_eps(L[0]) for L in layers]
    if azimuth is not None:
        eps_list = [_rotate_z(e, azimuth) for e in eps_list]
    q_in, A_in = isotropic_modes(n_in, xi)
    q_out, A_out = isotropic_modes(n_out, xi)
    if coherent:
        shape = np.broadcast_shapes(xi.shape, *[e.shape[:-2] for e in eps_list])
        M = np.broadcast_to(np.eye(4, dtype=complex), shape + (4, 4))
        for eps, (_, d) in zip(eps_list, layers):
            q, V = modes(eps, xi, normalize=False)
            ph = np.exp(1j*k0*q*d)
            P = (V*ph[..., None, :]) @ np.linalg.inv(V)
            M = P @ M
        MA = M @ A_in
        G = np.concatenate([MA[..., 2:], -np.broadcast_to(A_out[..., :2], MA[..., :2].shape)], axis=-1)
        X = _solve(G, -MA[..., :2])
        r, t = X[..., :2, :], X[..., 2:, :]
    else:
        V_prev = A_in
        a = None
        r = None
        for eps, (_, d) in zip(eps_list + [None], layers + [(None, 0.0)]):
            if eps is None:
                q, V = q_out, A_out
            else:
                q, V = modes(eps, xi, normalize=False)
            shape = np.broadcast_shapes(V_prev.shape, V.shape)
            Vp = np.broadcast_to(V_prev, shape)
            Vn = np.broadcast_to(V, shape)
            G = np.concatenate([Vp[..., 2:], -Vn[..., :2]], axis=-1)
            rhs = -Vp[..., :2] if a is None else -(Vp[..., :2] @ a)
            X = _solve(G, rhs)
            if r is None:
                r = X[..., :2, :]
            a = X[..., 2:, :]
            if eps is not None:
                a = a*np.exp(1j*k0*q[..., :2]*d)[..., :, None]
            V_prev = V
        t = a
    if azimuth is not None:
        phi = np.asarray(azimuth, dtype=float)
        c, s = np.cos(phi), np.sin(phi)
        B = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)   # columns p, s
        Bt = np.swapaxes(B, -1, -2)
        t = B @ t @ Bt
        # reflected waves: p along +x as well (tangential convention)
        r = B @ r @ Bt

    return t, r


def power_factor(xi, n_in=1.0, n_out=1.0):
    """Factor q_out/q_in = Re sqrt(n_out**2 - xi**2)/sqrt(n_in**2 - xi**2)
    between abs(t)**2 and the transmitted power.

    Parameters
    ----------
    xi : number or array
    n_in, n_out : numbers

    Returns
    -------
    factor : array
    """

    xi = np.asarray(xi, dtype=complex)

    return np.real(np.sqrt(n_out**2 - xi**2))/np.real(np.sqrt(n_in**2 - xi**2))


# --------------------------------------------------------------------------
# interfaces and polarizing prisms

Wave = namedtuple("Wave", ["k", "power", "S", "E", "label"])
Wave.__doc__ = """A plane wave after `refract()` or `trace_interfaces()`.

k : 3 vector, wave vector in units of k0 (its length is the index);
power : fraction of the incident power (flux through the interface);
S : unit Poynting vector; E : unit electric field (complex);
label : str, history of the branch ("o", "e" or "p", "s" per medium,
"R" for a reflection).
"""


def _is_isotropic(eps):
    return np.allclose(eps, eps[0, 0]*np.eye(3), atol=1e-12*abs(eps[0, 0]))


def _local_modes(eps, xi):
    if _is_isotropic(eps):
        n = np.sqrt(eps[0, 0])
        q, V = isotropic_modes(n, xi)
        V = V/np.sqrt(np.maximum(np.abs(_flux(V)), 1e-300))[None, :]
        return q, V, ("p", "s")
    q, V = modes(eps, xi)
    return q, V, ("e", "o")      # labels fixed below from the fields


def _mode_label(eps, k, E):
    """'o' if E is perpendicular to the optic axis of a uniaxial medium."""
    w, v = np.linalg.eigh(np.real(eps))
    # the axis is the eigenvector with the distinct eigenvalue
    d = np.abs(np.diff(w))
    if d.max() < 1e-12:
        return "?"
    axis = v[:, 2] if d[1] > d[0] else v[:, 0]
    if min(d) > 1e-9*abs(w).max():
        # biaxial: label by the slow/fast wave instead
        return None
    ka = np.cross(np.real(k), axis)
    if np.linalg.norm(ka) < 1e-9*np.linalg.norm(k):
        return "o"
    # the ordinary wave has E (and D) along k x axis
    return "o" if abs(np.vdot(ka/np.linalg.norm(ka), E)) > 0.7*np.linalg.norm(E) else "e"


def _fields(eps, xi, q, psi, normalize=True):
    """Full E of a wave from its tangential fields."""
    Ex, Hy, Ey, Hx = psi
    Ez = -(xi*Hy + eps[2, 0]*Ex + eps[2, 1]*Ey)/eps[2, 2]
    E = np.array([Ex, Ey, Ez])
    return E/np.linalg.norm(E) if normalize else E


def refract(eps1, eps2, normal, k, E=None):
    """Reflected and transmitted plane waves at a plane interface.

    Parameters
    ----------
    eps1, eps2 : (3, 3) arrays or numbers
        Media before and after the interface (an isotropic index is
        accepted).
    normal : 3 vector
        Interface normal pointing into medium 2.
    k : 3 vector
        Incident wave vector in units of k0; it must be one of the waves
        of medium 1 (the nearest one is taken).
    E : 3 vector, optional
        Incident field for an isotropic medium 1 (default: unpolarized,
        split equally into p and s).

    Returns
    -------
    transmitted, reflected : lists of Wave
        Up to two waves each, with their power fractions (0 for evanescent
        waves, which are left out).
    """

    eps1, eps2 = _layer_eps(eps1), _layer_eps(eps2)
    N = _unit(normal)
    k = np.asarray(k, dtype=float)
    kt = k - (k @ N)*N
    xi = np.linalg.norm(kt)
    x = kt/xi if xi > 1e-12 else _transverse_basis(N)[1]
    y = np.cross(N, x)
    R = np.array([x, y, N])                                   # global -> local
    e1, e2 = rotate_eps(eps1, R), rotate_eps(eps2, R)
    q1, V1, lab1 = _local_modes(e1, xi)
    q2, V2, lab2 = _local_modes(e2, xi)
    kz = k @ N
    G = np.concatenate([V1[:, 2:], -V2[:, :2]], axis=1)
    if _is_isotropic(e1):
        if E is None:
            incident = [(0, np.sqrt(0.5)), (1, np.sqrt(0.5))]
            coherent_sum = False
        else:
            El = R @ np.asarray(E, dtype=complex)
            c = kz/np.linalg.norm(k)
            # p amplitude along (c, 0, -s), s along y
            sn = xi/np.linalg.norm(k)
            amp_p = El[0]*c - El[2]*sn
            amp_s = El[1]
            norm = np.sqrt(abs(amp_p)**2 + abs(amp_s)**2)
            incident = [(None, np.array([amp_p, amp_s])/norm)]
            coherent_sum = True
    else:
        near = np.abs(q1[:2] - kz) < 1e-7*max(abs(kz), 1.0)
        if E is not None and near.all():
            # degenerate waves (along an optic axis): decompose the field
            El = R @ np.asarray(E, dtype=complex)
            F = np.stack([_fields(e1, xi, q1[i], V1[:, i], normalize=False) for i in range(2)], axis=1)
            a = np.linalg.lstsq(F, El, rcond=None)[0]
            incident = [(None, a/np.linalg.norm(a))]
            coherent_sum = True
        else:
            j = int(np.argmin(np.abs(q1[:2] - kz)))
            incident = [(j, 1.0)]
            coherent_sum = False

    def waves(qs, Vs, eps_loc, labs, amps, sign, extra):
        out = []
        if abs(qs[0] - qs[1]) < 1e-7*max(abs(qs[0]), 1.0) and abs(qs[0].imag) < 1e-9:
            # degenerate pair (isotropic, or along an optic axis): one wave
            psi = Vs @ amps
            p = float(abs(_flux(psi[:, None])[0]))
            if p < 1e-14:
                return out
            kl = np.array([xi, 0.0, qs[0].real])
            Ef = _fields(eps_loc, xi, qs[0], psi)
            S = R.T @ (kl/np.linalg.norm(kl))
            if labs[0] in "ps":
                lab = "p" if abs(amps[0]) >= abs(amps[1]) else "s"
            else:
                lab = "o"
            out.append(Wave(R.T @ kl, p, S, R.T @ Ef, extra + lab))
            return out
        for i in range(2):
            qi = qs[i]
            if abs(qi.imag) > 1e-9:
                continue
            p = float(np.abs(amps[i])**2)
            if p < 1e-14:
                continue
            kl = np.array([xi, 0.0, qi.real])
            Ef = _fields(eps_loc, xi, qi, Vs[:, i])
            H = np.cross(kl, Ef)
            S = np.real(np.cross(Ef, np.conj(H)))
            S = R.T @ (S/np.linalg.norm(S))
            Eg = R.T @ Ef
            lab = labs[i] if labs[0] in "ps" else (_mode_label(eps_loc, kl, Ef) or ("f" if i == 0 else "s"))
            out.append(Wave(R.T @ kl, p, S, Eg, extra + lab))
        return out

    trans, refl = [], []
    if coherent_sum:
        _, a = incident[0]
        X = np.linalg.solve(G, -(V1[:, :2] @ a))
        trans = waves(q2[:2], V2[:, :2], e2, lab2, X[2:], 1, "")
        refl = waves(q1[2:], V1[:, 2:], e1, lab1, X[:2], -1, "R")
    else:
        for j, w in incident:
            X = np.linalg.solve(G, -V1[:, j])*w
            trans += waves(q2[:2], V2[:, :2], e2, lab2, X[2:], 1, "")
            refl += waves(q1[2:], V1[:, 2:], e1, lab1, X[:2], -1, "R")

    return trans, refl


def trace_interfaces(media, normals, k, E=None, min_power=1e-9):
    """Follow all transmitted branches of a plane wave through plane
    interfaces (reflections are dropped).

    Parameters
    ----------
    media : list of (3, 3) arrays or numbers
        len(normals) + 1 media, from the incidence side.
    normals : list of 3 vectors
        Interface normals, pointing forward.
    k : 3 vector
        Incident wave vector in medium 0 (units of k0).
    E : 3 vector, optional
        Incident field (default unpolarized).
    min_power : number, optional
        Branches with less power are dropped.

    Returns
    -------
    waves : list of Wave
        The waves leaving the last interface, with the labels of their
        history (e.g. "o e p": ordinary in the first crystal, extraordinary
        in the second).
    """

    current = [Wave(np.asarray(k, dtype=float), 1.0, None, E, "")]
    for i, N in enumerate(normals):
        nxt = []
        for w in current:
            Ein = E if i == 0 else w.E
            tr, _ = refract(media[i], media[i + 1], N, w.k, E=Ein)
            for t in tr:
                p = w.power*t.power
                if p > min_power:
                    nxt.append(Wave(t.k, p, t.S, t.E, (w.label + " " + t.label).strip()))
        current = nxt

    return current


def _deg_dir(theta_deg):
    th = np.radians(theta_deg)
    return np.array([np.sin(th), 0.0, np.cos(th)])


def glan_prism(theta_in, n_o, n_e, apex, n_gap=1.0, axis="taylor", n_ambient=1.0):
    """Transmission of a Glan-type polarizer (two crystal prisms with a gap
    or cement at the apex angle).

    The rays lie in the x-z plane; the entrance face is normal to z, the
    gap is tilted by `apex` about y. The ordinary wave meets the gap beyond
    its critical angle and is reflected; the extraordinary wave (n_e < n_o,
    as in calcite) passes.

    Parameters
    ----------
    theta_in : number or array
        Angle of incidence in the x-z plane (degrees, positive towards +x,
        towards the side where the gap angle grows).
    n_o, n_e : numbers
    apex : number
        Angle of the gap to the entrance face (degrees).
    n_gap : number, optional
        1 for an air gap (Glan-Taylor, Glan-Foucault), the cement index
        for a Glan-Thompson.
    axis : {"taylor", "foucault"}, optional
        Optic axis along x (in the plane of incidence, the transmitted wave
        is p polarized at the gap: Glan-Taylor) or along y (s polarized:
        Glan-Foucault).
    n_ambient : number, optional

    Returns
    -------
    T_e, T_o : arrays
        Power transmission of the extraordinary and the ordinary wave
        through all four surfaces (unpolarized input counts each as 1).
    """

    a = {"taylor": (1.0, 0.0, 0.0), "foucault": (0.0, 1.0, 0.0)}[axis]
    eps = uniaxial_eps(n_o, n_e, a)
    al = np.radians(apex)
    gap_normal = np.array([np.sin(al), 0.0, np.cos(al)])
    normals = [np.array([0.0, 0.0, 1.0]), gap_normal, gap_normal, np.array([0.0, 0.0, 1.0])]
    media = [n_ambient, eps, n_gap, eps, n_ambient]
    th = np.atleast_1d(np.asarray(theta_in, dtype=float))
    T_e, T_o = np.zeros(th.shape), np.zeros(th.shape)
    for i, t in enumerate(th):
        for w in trace_interfaces(media, normals, n_ambient*_deg_dir(t)):
            lab = w.label.split()
            if lab[0] == "e":
                T_e[i] += 2*w.power
            elif lab[0] == "o":
                T_o[i] += 2*w.power
    if np.ndim(theta_in) == 0:
        return T_e[0], T_o[0]
    return T_e, T_o


def _split_prisms(axes, n_o, n_e, wedge, theta_in, n_ambient=1.0):
    eps1 = uniaxial_eps(n_o, n_e, axes[0])
    eps2 = uniaxial_eps(n_o, n_e, axes[1])
    w = np.radians(wedge)
    N = np.array([np.sin(w), 0.0, np.cos(w)])
    z = np.array([0.0, 0.0, 1.0])
    waves = trace_interfaces([n_ambient, eps1, eps2, n_ambient], [z, N, z],
                             n_ambient*_deg_dir(theta_in))
    return waves


def wollaston(n_o, n_e, wedge, theta_in=0.0, n_ambient=1.0):
    """The two beams of a Wollaston prism: optic axes along x (first prism)
    and y (second), a cemented-free interface at the wedge angle.

    Parameters
    ----------
    n_o, n_e : numbers
    wedge : number
        Wedge angle in degrees.
    theta_in : number, optional
        Angle of incidence (degrees, x-z plane).
    n_ambient : number, optional

    Returns
    -------
    waves : list of Wave
        Two output waves; their angles are degrees(arctan2(k_x, k_z)). For
        small wedges the split is about 2 arcsin((n_o - n_e) tan(wedge)).
    """

    return _split_prisms([(1.0, 0, 0), (0, 1.0, 0)], n_o, n_e, wedge, theta_in, n_ambient)


def rochon(n_o, n_e, wedge, theta_in=0.0, n_ambient=1.0):
    """The two beams of a Rochon prism: optic axis along z (the beam) in the
    first prism, along y in the second; the ordinary wave passes
    undeviated.

    Parameters
    ----------
    n_o, n_e : numbers
    wedge : number
        Wedge angle in degrees.
    theta_in : number, optional
    n_ambient : number, optional

    Returns
    -------
    waves : list of Wave
    """

    return _split_prisms([(0, 0, 1.0), (0, 1.0, 0)], n_o, n_e, wedge, theta_in, n_ambient)


# --------------------------------------------------------------------------
# Jones-level rotators

def rotator(angle, faraday=False, backward=False):
    """Jones matrix (lab x-y frame) of a rotator.

    Parameters
    ----------
    angle : number
        Rotation for a forward wave, counterclockwise from x to y as seen
        by an observer looking towards the source (radians).
    faraday : bool, optional
        True for Faraday rotation (non-reciprocal), False for natural
        optical activity (reciprocal).
    backward : bool, optional
        The wave travels along -z. Natural activity then rotates by
        -angle in the lab frame (a round trip cancels), Faraday rotation
        by +angle (a round trip doubles it).

    Returns
    -------
    J : (2, 2) array
    """

    a = angle if (faraday or not backward) else -angle
    c, s = np.cos(a), np.sin(a)

    return np.array([[c, -s], [s, c]])


def elliptical_retarder(delta, rotation, axis=0.0):
    """Jones matrix of a plate with linear and circular birefringence.

    The plate is exp(N) with N = i delta/2 diag(-1, 1) + rotation [[0, -1],
    [1, 0]] in the frame of its fast axis (Jones' differential matrix for a
    homogeneous medium): a linear retardance delta (slow axis y, as
    `polarization.jones.J_WP`) and a rotation that alone would turn the
    plane by `rotation` (counterclockwise). Its eigenstates are elliptical;
    quartz cut at an angle to its axis is the classic example.

    Parameters
    ----------
    delta : number
        Linear retardance (radians).
    rotation : number
        Circular part as a rotation angle (radians).
    axis : number, optional
        Azimuth of the fast axis (radians).

    Returns
    -------
    J : (2, 2) complex array
    """

    g = np.sqrt((delta/2)**2 + rotation**2 + 0j)
    N = np.array([[-1j*delta/2, -rotation], [rotation, 1j*delta/2]])
    # N**2 = -g**2 I, exp(N) = cos g I + sin(g)/g N
    J = np.cos(g)*np.eye(2) + (np.sinc(g/np.pi) if g != 0 else 1.0)*N
    c, s = np.cos(axis), np.sin(axis)
    Rr = np.array([[c, s], [-s, c]])

    return Rr.T @ J @ Rr


def isolator(rotation_error=0.0, extinction=np.inf, input_angle=0.0, rotation=np.pi/4):
    """Forward and backward transmission of a Faraday isolator: polarizer at
    `input_angle`, Faraday rotator, analyzer at input_angle + 45°.

    Parameters
    ----------
    rotation_error : number or array, optional
        Deviation of the Faraday rotation from 45° (radians).
    extinction : number, optional
        Power extinction ratio of the polarizers (inf: ideal).
    input_angle : number, optional
    rotation : number, optional
        Nominal rotation (pi/4).

    Returns
    -------
    T_forward, T_backward : arrays
        Power transmissions for light polarized along the input polarizer
        (forward) and along the analyzer (backward); the isolation is
        -10 log10(T_backward).
    """

    err = np.atleast_1d(np.asarray(rotation_error, dtype=float))
    leak = 0.0 if np.isinf(extinction) else 1/np.sqrt(extinction)

    def pol(angle):
        c, s = np.cos(angle), np.sin(angle)
        Rr = np.array([[c, s], [-s, c]])
        return Rr.T @ np.diag([1.0, leak]) @ Rr

    P1, P2 = pol(input_angle), pol(input_angle + np.pi/4)
    Tf, Tb = [], []
    for e in err:
        F = rotator(rotation + e, faraday=True)
        Ein = np.array([np.cos(input_angle), np.sin(input_angle)])
        Tf.append(np.sum(np.abs(P2 @ F @ P1 @ Ein)**2))
        Eb = np.array([np.cos(input_angle + np.pi/4), np.sin(input_angle + np.pi/4)])
        Tb.append(np.sum(np.abs(P1 @ F @ P2 @ Eb)**2))
    Tf, Tb = np.array(Tf), np.array(Tb)
    if np.ndim(rotation_error) == 0:
        return Tf[0], Tb[0]
    return Tf, Tb
