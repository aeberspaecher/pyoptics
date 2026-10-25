#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Jones pupils: the polarization properties of a projection lens, their
representations and decompositions.

A Jones pupil J(P) is a 2 x 2 complex matrix at every point P = (P_x, P_y)
of the normalized pupil. It maps the Jones vector (E_x, E_y) of a plane wave
entering the pupil to that leaving it. In LithoSimulation it acts on every
diffraction order before the aplanatic rotation into the image space
(Projector(jones_pupil=...)), on top of the scalar pupil (Zernike phase and
apodization) of the Projector.

Representations from the literature
-----------------------------------
Eight real functions describe a Jones pupil. The literature uses these
representations:

- Jones-Zernike: each of the four complex elements J_jk expanded into
  Zernikes. Simple, but the elements mix physical effects and depend on the
  orientation of the coordinate system.
- Pauli decomposition (McIntyre, Kye, Levinson and Neureuther, J.
  Micro/Nanolith. MEMS MOEMS 5(3), 033001 (2006)), ::

      J = a_0*s_0 + a_1*s_1 + a_2*s_2 + a_3*s_3,   a_k = tr(s_k J)/2,

  with the Pauli matrices s_0 = I, s_1 = diag(1, -1), s_2 = [[0, 1], [1, 0]]
  and s_3 = [[0, -i], [i, 0]]. a_0 is the scalar part (apodization and
  wavefront), a_1 and a_2 are linear retardance (imaginary parts) and
  diattenuation (real parts) along x/y and +-45 degrees, and a_3 is circular
  retardance (rotation) and circular diattenuation. McIntyre et al. found
  this the most useful format for lithographers. Expanding each a_k into
  Zernikes gives the Pauli-Zernike representation (PauliZernikePupil).
- Physical decomposition (e.g. Totzeck, Graeupner, Heil et al., "Polarization
  influence on imaging", J. Micro/Nanolith. MEMS MOEMS 4(3), 031108 (2005);
  Geh et al.): the polar decomposition J = U H into a retarder U (unitary)
  and a diattenuator H (Hermitian, positive). It gives scalar transmission
  and phase, the magnitude and orientation of the retardance and of the
  diattenuation, and the circular retardance (rotation). decompose() does
  this; PhysicalJonesPupil builds a Jones pupil from these parts.
- Orientation Zernike polynomials (OZP; Ruoff and Totzeck, "Orientation
  Zernike polynomials: a useful way to describe the polarization effects of
  optical imaging systems", J. Micro/Nanolith. MEMS MOEMS 8(3), 031404
  (2009); US patent 8,126,669). Retardance and diattenuation are
  orientation fields: an orientation psi is defined modulo 180 degrees, so
  it is described by an orientator a*(cos(2*psi), sin(2*psi)). OZPs are a
  complete set of vector polynomials for orientator fields: the Zernike
  radial polynomials times angular orientators. Like scalar Zernikes they
  separate the symmetries (a rotationally symmetric radial or tangential
  retardance, typical for coatings and stress, is a single OZP), and they
  serve for tolerancing via linear and quadratic sensitivities of CD and
  placement. They are used for lens specification and matching (Zeiss and
  ASML).

Orientation Zernike convention used here
----------------------------------------
An orientator field is written as the complex function o = o_1 + i*o_2. The
OZPs are R_n^abs(m)(rho)*exp(i*m*phi) and i*R_n^abs(m)(rho)*exp(i*m*phi) for all
(n, m) with n - abs(m) even. They are labelled by +-j with the Fringe Zernike
index j of (n, abs(m)) such that the first component o_1 of OZ +j is the Fringe
Zernike Z_j (the patent's convention: the J_11 element corresponds to the
scalar Zernike abs(OZ)), and OZ -j differs in the sign of the second component:

- Z_j = R*cos(m*phi), m >= 0:  OZ +j = R*exp(+i*m*phi), OZ -j = R*exp(-i*m*phi),
- Z_j = R*sin(m*phi), m > 0:   OZ +j = i*R*exp(-i*m*phi), OZ -j = -i*R*exp(+i*m*phi),
- m = 0: OZ +j = R (along x/y), OZ -j = i*R (along +-45 degrees).

A rotation of the lens by alpha multiplies exp(i*m*phi) by exp(i*(2 - m)*alpha):
the rotationally symmetric ones are OZ +5 = rho**2*exp(2*i*phi) (radial
orientation; a negative coefficient is tangential) and OZ -6 =
-i*rho**2*exp(2*i*phi) (a spiral orientation at -45 degrees to the radius:
i*o rotates an orientator by 45 degrees). The patent writes the
angular parts with half angles, (cos(m*phi/2), -sin(m*phi/2)); that is the
same set of functions with m relabelled.

Physical Jones pupil used here
------------------------------
PhysicalJonesPupil is J = U H with ::

    U = exp(-i*pi*(g_1*s_1 + g_2*s_2 + g_3*s_3)),
    H = exp(e_1*s_1 + e_2*s_2 + e_3*s_3)/sqrt(cosh(2*abs(e))),

the retardance orientator (g_1, g_2) in waves (the retardance between the
eigenpolarizations is 2*pi*abs(g) radians at the orientation psi of the fast
axis, with g_1 + i*g_2 = abs(g)*exp(2*i*psi)), the circular retardance g_3 in
waves (a rotation of the polarization by pi*g_3), and the diattenuation
orientator d = tanh(2*abs(e))*e/abs(e), whose magnitude is
(T_max - T_min)/(T_max + T_min). H is normalized to a mean intensity
transmission of 1, so that the scalar transmission and phase stay with the
Projector (apodization, Zernikes).

Convention: fields vary as exp(i(k.r - omega*t)).
"""

import numpy as np

from ..bases import fringe_to_n_m, fringe_zernike_R_nm, fringe_zernike, fringe_zernike_sum


SIGMA = np.array([[[1, 0], [0, 1]],
                  [[1, 0], [0, -1]],
                  [[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]]], dtype=complex)


# Pauli algebra:

def pauli_matrix(a):
    """Jones matrices sum_k a[k]*s_k for Pauli coefficients a of shape
    (4,) + shape; returns shape + (2, 2)."""

    return np.einsum("k...,kij->...ij", np.asarray(a, dtype=complex), SIGMA)


def pauli_coefficients(J):
    """Pauli coefficients a_k = tr(s_k J)/2 of Jones matrices J (shape
    (..., 2, 2)); returns shape (4, ...)."""

    return np.einsum("kji,...ij->k...", SIGMA, np.asarray(J, dtype=complex))/2


def _expm_pauli(c):
    """exp(c_1 s_1 + c_2 s_2 + c_3 s_3) for complex vectors c of shape (3,
    ...): cosh(z) I + sinh(z)/z c.s with z = sqrt(c.c)."""

    c = np.asarray(c, dtype=complex)
    z = np.sqrt(np.sum(c**2, axis=0))
    small = np.abs(z) < 1e-8
    z_safe = np.where(small, 1.0, z)
    shz = np.where(small, 1 + z**2/6, np.sinh(z_safe)/z_safe)

    return pauli_matrix(np.concatenate([np.cosh(z)[None], shz*c]))


# scalar and orientation Zernikes:

zernike = fringe_zernike  # Fringe Zernike Z_j(P_x, P_y), see pyoptics.bases


def zernike_sum(coefficients, P_x, P_y):
    """sum_j c_j Z_j (complex coefficients allowed)."""

    return fringe_zernike_sum(coefficients, P_x, P_y).astype(complex)


def orientation_zernike(oz, P_x, P_y):
    """Orientation Zernike polynomial OZ +-j as the complex orientator field
    o_1 + i*o_2 (see the module documentation)."""

    j = abs(int(oz))
    if j < 1 or oz == 0:
        raise ValueError("orientation Zernike indices are +-1, +-2, ...")
    n, m = fringe_to_n_m(j)
    rho, phi = np.hypot(P_x, P_y), np.arctan2(P_y, P_x)
    R = fringe_zernike_R_nm(n, abs(m), rho)
    if m == 0:
        return R*(1.0 if oz > 0 else 1j)
    if m > 0:
        return R*np.exp(1j*m*phi*(1 if oz > 0 else -1))

    return R*(1j*np.exp(-1j*abs(m)*phi) if oz > 0 else -1j*np.exp(1j*abs(m)*phi))


def orientation_zernike_sum(coefficients, P_x, P_y):
    """Orientator field sum_oz c_oz OZ_oz as o_1 + i*o_2 (real
    coefficients)."""

    out = np.zeros(np.broadcast(P_x, P_y).shape, dtype=complex)
    for oz, c in (coefficients or {}).items():
        out = out + c*orientation_zernike(oz, P_x, P_y)

    return out


def oz_indices(n_terms):
    """OZ indices +-1 ... +-n_terms in the order +1, -1, +2, -2, ..."""

    return [s*j for j in range(1, n_terms + 1) for s in (1, -1)]


def _lstsq(columns, values):
    A = np.column_stack(columns)
    c, *_ = np.linalg.lstsq(A, values, rcond=None)

    return c


def fit_zernikes(P_x, P_y, values, n_terms=36, start=1):
    """Fringe Zernike coefficients {j: c} of (complex or real) values sampled
    at the pupil points (P_x, P_y) (1D arrays inside the unit circle)."""

    js = list(range(start, n_terms + 1))
    c = _lstsq([zernike(j, P_x, P_y) for j in js], values)

    return {j: (complex(v) if np.iscomplexobj(values) else float(np.real(v))) for j, v in zip(js, c)}


def fit_orientation_zernikes(P_x, P_y, orientator, n_terms=36):
    """OZP coefficients {+-j: c} of an orientator field o_1 + i*o_2 sampled
    at the pupil points (real least squares)."""

    ozs = oz_indices(n_terms)
    cols = [orientation_zernike(oz, P_x, P_y) for oz in ozs]
    A = np.vstack([np.column_stack([c.real for c in cols]), np.column_stack([c.imag for c in cols])])
    b = np.concatenate([np.real(orientator), np.imag(orientator)])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    return dict(zip(ozs, (float(v) for v in c)))


# Jones pupils:

class JonesPupil(object):
    """Base class: a callable J(P_x, P_y) returning Jones matrices of shape
    np.broadcast(P_x, P_y).shape + (2, 2). Any callable with this signature
    can serve as Projector(jones_pupil=...)."""

    def __call__(self, P_x, P_y):
        raise NotImplementedError

    def pauli(self, P_x, P_y):
        """Pauli coefficients, shape (4,) + shape."""

        return pauli_coefficients(self(P_x, P_y))

    def __matmul__(self, other):
        return ProductPupil(self, other)


class ProductPupil(JonesPupil):
    """The Jones pupil first @ second (second acts first)."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def __call__(self, P_x, P_y):
        return self.first(P_x, P_y) @ self.second(P_x, P_y)


class PauliZernikePupil(JonesPupil):
    """J = sum_k a_k s_k with Zernike expansions of the Pauli coefficients.

    Parameters
    ----------
    a0, a1, a2, a3 : dict {j: complex}, optional
        Fringe Zernike coefficients of the Pauli coefficients; a0 defaults to
        {1: 1} (identity), the others to 0.
    """

    def __init__(self, a0=None, a1=None, a2=None, a3=None):
        self.a = [dict({1: 1.0} if a0 is None else a0), dict(a1 or {}), dict(a2 or {}), dict(a3 or {})]

    def __call__(self, P_x, P_y):
        return pauli_matrix([zernike_sum(a, P_x, P_y) for a in self.a])

    @classmethod
    def fit(cls, P_x, P_y, J, n_terms=36):
        """Fit to Jones matrices J (shape (N, 2, 2)) sampled at the pupil
        points."""

        a = pauli_coefficients(J)

        return cls(*[fit_zernikes(P_x, P_y, a_k, n_terms) for a_k in a])


class PhysicalJonesPupil(JonesPupil):
    """Retarder times diattenuator, J = U H (see the module documentation).

    Parameters
    ----------
    retardance : dict {+-j: c}, optional
        OZP coefficients of the linear retardance orientator (waves).
    diattenuation : dict {+-j: c}, optional
        OZP coefficients of the linear diattenuation orientator
        ((T_max - T_min)/(T_max + T_min)).
    circular_retardance : dict {j: c}, optional
        Fringe Zernike coefficients of the circular retardance (waves; a
        rotation by pi*g_3).
    circular_diattenuation : dict {j: c}, optional
        Fringe Zernike coefficients of the circular diattenuation.
    """

    def __init__(self, retardance=None, diattenuation=None, circular_retardance=None,
                 circular_diattenuation=None):
        self.retardance = dict(retardance or {})
        self.diattenuation = dict(diattenuation or {})
        self.circular_retardance = dict(circular_retardance or {})
        self.circular_diattenuation = dict(circular_diattenuation or {})

    def fields(self, P_x, P_y):
        """Retardance generator g (3, ...) in waves and diattenuation vector
        d (3, ...) (magnitude (T_max - T_min)/(T_max + T_min))."""

        o_r = orientation_zernike_sum(self.retardance, P_x, P_y)
        o_d = orientation_zernike_sum(self.diattenuation, P_x, P_y)
        g = np.array([o_r.real, o_r.imag, zernike_sum(self.circular_retardance, P_x, P_y).real])
        d = np.array([o_d.real, o_d.imag, zernike_sum(self.circular_diattenuation, P_x, P_y).real])

        return g, d

    def __call__(self, P_x, P_y):
        g, d = self.fields(P_x, P_y)

        return compose(g, d)

    @classmethod
    def fit(cls, P_x, P_y, J, n_terms=36):
        """Fit to Jones matrices J sampled at the pupil points: the polar
        decomposition of each J, then OZP fits of the orientators and
        Zernike fits of the circular parts. Returns (pupil, scalar) with
        the scalar part {"phase": Zernikes in waves, "transmission":
        Zernikes of the mean intensity transmission} of the decomposition."""

        parts = decompose(J)
        g, d = parts["retardance"], parts["diattenuation"]
        pupil = cls(fit_orientation_zernikes(P_x, P_y, g[0] + 1j*g[1], n_terms),
                    fit_orientation_zernikes(P_x, P_y, d[0] + 1j*d[1], n_terms),
                    fit_zernikes(P_x, P_y, g[2], n_terms), fit_zernikes(P_x, P_y, d[2], n_terms))
        scalar = {"phase": fit_zernikes(P_x, P_y, parts["phase"]/(2*np.pi), n_terms),
                  "transmission": fit_zernikes(P_x, P_y, parts["transmission"], n_terms)}

        return pupil, scalar


def compose(g, d, transmission=1.0, phase=0.0):
    """Jones matrices sqrt(T)*exp(i*phase)*U H from the retardance generator
    g (3, ...) in waves, the diattenuation vector d (3, ...), the mean
    intensity transmission T and the phase (radians)."""

    g, d = np.asarray(g, dtype=float), np.asarray(d, dtype=float)
    U = _expm_pauli(-1j*np.pi*g)
    d_abs = np.sqrt(np.sum(d**2, axis=0))
    if np.any(d_abs >= 1):
        raise ValueError("diattenuation must be smaller than 1")
    e_abs = np.arctanh(d_abs)/2
    scale = np.where(d_abs > 0, e_abs/np.where(d_abs > 0, d_abs, 1.0), 0.5)  # |e|/|d| (-> 1/2 for d -> 0)
    H = _expm_pauli(scale*d)/np.sqrt(np.cosh(2*e_abs))[..., None, None]
    J = U @ H

    return (np.sqrt(transmission)*np.exp(1j*np.asarray(phase)))[..., None, None]*J


def decompose(J):
    """Polar decomposition J = sqrt(T)*exp(i*phase)*U H of Jones matrices
    (shape (..., 2, 2)) as in PhysicalJonesPupil.

    Returns
    -------
    dict
        "transmission" T (mean intensity transmission, (s_1**2 + s_2**2)/2
        with the singular values s), "phase" (radians, of det(U)/2),
        "retardance" g (3, ...) in waves (linear: g[0] + i*g[1] is the
        orientator, circular: g[2]), "diattenuation" d (3, ...), and "U"
        (unitary, including exp(i*phase)) and "H" (normalized as in
        PhysicalJonesPupil), J = sqrt(T)*U H.
    """

    J = np.asarray(J, dtype=complex)
    W, s, Vh = np.linalg.svd(J)
    U = W @ Vh
    H = np.swapaxes(Vh.conj(), -1, -2) @ (s[..., :, None]*Vh)
    T = (s[..., 0]**2 + s[..., 1]**2)/2
    # unitary part: exp(i*phase) times SU(2), b_0 >= 0 (retardance <= half a wave)
    phase = np.angle(np.linalg.det(U))/2
    V = U*np.exp(-1j*phase)[..., None, None]
    b = pauli_coefficients(V)
    flip = b[0].real < 0
    b = np.where(flip, -b, b)
    phase = np.where(flip, phase + np.pi, phase)
    phase = np.angle(np.exp(1j*phase))
    n_sin = (1j*b[1:]).real  # sin(delta/2)*n
    half = np.arctan2(np.sqrt(np.sum(n_sin**2, axis=0)), b[0].real)  # delta/2
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(half > 1e-12, n_sin*half/np.sin(np.where(half > 1e-12, half, 1.0)), n_sin)/np.pi
    # Hermitian part: H/sqrt(T) = exp(e.s)/sqrt(cosh 2|e|), d = tanh(2|e|) e/|e|
    Hn = H/np.sqrt(T)[..., None, None]
    c = pauli_coefficients(Hn).real  # cosh|e|/N, sinh|e| e_hat/N
    d = 2*c[0]*c[1:]/(c[0]**2 + np.sum(c[1:]**2, axis=0))  # tanh(2|e|) e_hat

    return {"transmission": T, "phase": phase, "retardance": g, "diattenuation": d, "U": U, "H": Hn}


def scalar_free(J):
    """Jones matrices divided by their scalar part sqrt(T)*exp(i*phase) (the
    polarization part that PhysicalJonesPupil describes)."""

    parts = decompose(J)

    return J/(np.sqrt(parts["transmission"])*np.exp(1j*parts["phase"]))[..., None, None]


__all__ = ["SIGMA", "pauli_matrix", "pauli_coefficients", "zernike", "zernike_sum",
           "orientation_zernike", "orientation_zernike_sum", "oz_indices", "fit_zernikes",
           "fit_orientation_zernikes", "JonesPupil", "ProductPupil", "PauliZernikePupil",
           "PhysicalJonesPupil", "compose", "decompose", "scalar_free"]
