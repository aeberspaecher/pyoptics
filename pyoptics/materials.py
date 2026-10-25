#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Simple dispersion models for optical materials.

Glass catalogues characterize a glass by its refractive index n_d at the
helium d line and its Abbe number ::

    V_d = (n_d - 1)/(n_F - n_C),

with the hydrogen F and C lines. The functions here build a dispersion curve
n(wavelength) from these two numbers using the two-term Cauchy formula ::

    n(wavelength) = A + B/wavelength**2,

whose coefficients are chosen such that n(lambda_d) = n_d and
n(lambda_F) - n(lambda_C) = (n_d - 1)/V_d hold exactly. This is a fair
approximation in the visible for normally dispersive glasses, but not in the
UV or IR, where Sellmeier data should be used instead.

Wavelengths are in nanometres by default. Other units work as long as the
line wavelengths are passed in the same unit as the evaluation wavelengths.

Media for the polarization ray trace
------------------------------------
`Material` objects describe a medium by its complex refractive index
n + i*k (k > 0 for absorption, as in `pyoptics.thin_films`) and, for
anisotropic media, by the index seen by the two polarization eigenstates of a
ray:

- `Isotropic`: n and the extinction coefficient k (see
  `extinction_from_absorption()` for an absorption coefficient alpha).
- `Uniaxial`: ordinary and extraordinary indices n_o + i*k_o and
  n_e + i*k_e with an optic axis. n_e != n_o gives retardance (linear
  birefringence), k_e != k_o dichroism (diattenuation).
- `CubicCrystal`: the intrinsic birefringence of CaF2 and other cubic
  crystals at short wavelengths.

Rays are refracted with the real part of a reference index
(`Material.n_geometric()`, n_o for uniaxial media); the anisotropic and
absorbing part of the index only acts on the polarization along the ray
(`Material.propagation()`). This is the usual model for weak anisotropy
(``|n_e - n_o| << 1``): the splitting of the ordinary and extraordinary rays
and their walk-off are neglected, their phase difference is not. Index
data are evaluated at wavelengths in nm; a value can be a number, a function
n(wavelength_nm) or a pair (n_d, V_d) for `abbe_dispersion()`.
"""

import numpy as np

from .geometry import R_z


# Fraunhofer lines [nm]:
LAMBDA_d = 587.5618  # helium d line
LAMBDA_F = 486.1327  # hydrogen F line
LAMBDA_C = 656.2725  # hydrogen C line


def abbe_number(n_d, n_F, n_C):
    """Abbe number V_d = (n_d - 1)/(n_F - n_C).
    """

    return (n_d - 1.0)/(n_F - n_C)


def cauchy_coeffs_from_abbe(n_d, V_d, lambda_d=LAMBDA_d, lambda_F=LAMBDA_F,
                            lambda_C=LAMBDA_C):
    """Coefficients A, B of n = A + B/wavelength**2 for given n_d and V_d.

    Parameters
    ----------
    n_d : double
        Refractive index at lambda_d.
    V_d : double
        Abbe number, must be positive. np.inf gives a dispersion free medium.
    lambda_d, lambda_F, lambda_C : double, optional
        Wavelengths of the d, F and C lines. Defaults are in nm; pass them in
        another unit to get B in that unit squared.

    Returns
    -------
    A, B : double
        Cauchy coefficients; B has the unit of wavelength squared.
    """

    if not V_d > 0:
        raise ValueError("V_d must be positive")

    # n_F - n_C = B*(1/lambda_F**2 - 1/lambda_C**2) = (n_d - 1)/V_d:
    B = (n_d - 1.0)/V_d/(1.0/lambda_F**2 - 1.0/lambda_C**2)
    A = n_d - B/lambda_d**2

    return A, B


def cauchy_n(wavelength, A, B):
    """Two-term Cauchy formula n = A + B/wavelength**2.
    """

    return A + B/np.asarray(wavelength, dtype=float)**2


def abbe_dispersion(wavelength, n_d, V_d, lambda_d=LAMBDA_d,
                    lambda_F=LAMBDA_F, lambda_C=LAMBDA_C):
    """Refractive index n(wavelength) of a glass given by n_d and V_d.

    See cauchy_coeffs_from_abbe() for the parameters.

    Examples
    --------
    >>> n_bk7 = abbe_dispersion([450, 550, 650], 1.5168, 64.17)
    """

    A, B = cauchy_coeffs_from_abbe(n_d, V_d, lambda_d, lambda_F, lambda_C)

    return cauchy_n(wavelength, A, B)


# media for the polarization ray trace:

def extinction_from_absorption(alpha, wavelength):
    """Extinction coefficient k = alpha*wavelength/(4*pi) of a medium whose
    intensity decays as exp(-alpha*z) (alpha in the inverse unit of
    wavelength)."""

    return alpha*wavelength/(4*np.pi)


def absorption_from_extinction(k, wavelength):
    """Intensity absorption coefficient alpha = 4*pi*k/wavelength."""

    return 4*np.pi*k/wavelength


def _value(value, wavelength_nm):
    """A number, a function of the wavelength in nm or a pair (n_d, V_d)."""

    if callable(value):
        if wavelength_nm is None:
            raise ValueError("a dispersive material needs the wavelength in nm")
        return value(wavelength_nm)
    if isinstance(value, tuple):
        if wavelength_nm is None:
            raise ValueError("a dispersive material needs the wavelength in nm")
        return abbe_dispersion(wavelength_nm, *value)

    return value


def complex_index(material, wavelength_nm=None):
    """Complex refractive index n + i*k of a Material, a number (complex
    allowed), a function n(wavelength_nm) or a pair (n_d, V_d)."""

    if isinstance(material, Material):
        return complex(material.index(wavelength_nm))

    return complex(_value(material, wavelength_nm))


def expm_2x2(A):
    """Matrix exponentials of a stack of complex 2 x 2 matrices (..., 2, 2).

    With A = a*I + B, a = trace(A)/2 and B**2 = d**2*I (d**2 = -det(B)),
    exp(A) = exp(a)*(cosh(d)*I + sinh(d)/d*B), which holds for defective
    matrices too.
    """

    A = np.asarray(A, dtype=complex)
    a = (A[..., 0, 0] + A[..., 1, 1])/2
    B = A - a[..., None, None]*np.eye(2)
    d = np.sqrt(B[..., 0, 0]**2 + B[..., 0, 1]*B[..., 1, 0])
    small = np.abs(d) < 1e-6
    d_safe = np.where(small, 1.0, d)
    sinhc = np.where(small, 1 + d**2/6, np.sinh(d_safe)/d_safe)

    return np.exp(a)[..., None, None]*(np.cosh(d)[..., None, None]*np.eye(2)
                                       + sinhc[..., None, None]*B)


def transverse_basis(k):
    """Orthonormal vectors (N, 3, 2) perpendicular to the unit directions k
    (N, 3): k x z (k x x along z) and k x (k x z)."""

    z = np.array([0.0, 0.0, 1.0])
    s = np.cross(k, z)
    small = np.linalg.norm(s, axis=-1) < 1e-12
    s[small] = np.cross(k[small], [1.0, 0.0, 0.0])
    s = s/np.linalg.norm(s, axis=-1)[:, None]

    return np.stack([s, np.cross(k, s)], axis=-1)


class Material(object):
    """Medium for the polarization ray trace (see the module docstring).

    Subclasses define index() and, if anisotropic, delta_index(). Directions
    are given in the local frame of the surface the medium belongs to (the
    surface in front of it).
    """

    def index(self, wavelength_nm=None):
        """Mean complex refractive index n + i*k."""

        raise NotImplementedError

    def n_geometric(self, wavelength_nm=None):
        """Real index used for the ray directions and the optical path."""

        return float(np.real(self.index(wavelength_nm)))

    def delta_index(self, k, basis, wavelength_nm=None):
        """Complex index matrices (N, 2, 2) on the transverse bases (N, 3, 2)
        of the rays with unit directions k (N, 3), minus n_geometric()."""

        dn = complex(self.index(wavelength_nm)) - self.n_geometric(wavelength_nm)

        return np.tile(dn*np.eye(2), (len(k), 1, 1))

    def propagation(self, k, length, wavelength, wavelength_nm=None):
        """Polarization ray tracing matrices (N, 3, 3) of paths of the given
        lengths along the unit directions k (N, 3) beyond the geometric
        optical path n_geometric()*length: retardance, diattenuation and
        absorption. wavelength (vacuum) is in the unit of the lengths."""

        B = transverse_basis(k)
        dN = self.delta_index(k, B, wavelength_nm)
        U = expm_2x2(2j*np.pi/wavelength*np.asarray(length)[:, None, None]*dN)

        return np.einsum("nij,njk,nlk->nil", B, U, B) + k[:, :, None]*k[:, None, :]


class Isotropic(Material):
    """Isotropic, possibly absorbing medium of index n + i*k.

    Parameters
    ----------
    n : number, callable or (n_d, V_d)
        Refractive index (may be complex).
    k : number or callable, optional
        Extinction coefficient, added as i*k.
    """

    def __init__(self, n, k=0.0):
        self.n, self.k = n, k

    def __repr__(self):
        return "Isotropic(n={!r}, k={!r})".format(self.n, self.k)

    def index(self, wavelength_nm=None):
        return _value(self.n, wavelength_nm) + 1j*_value(self.k, wavelength_nm)


class Uniaxial(Material):
    """Uniaxial medium: retardance (n_e != n_o) and dichroism (k_e != k_o).

    A ray along the unit vector k at the angle theta to the optic axis has
    the ordinary eigenstate (D perpendicular to the plane of k and the axis)
    with n_o + i*k_o and the extraordinary one with ::

        1/n(theta)**2 = cos(theta)**2/(n_o + i*k_o)**2 + sin(theta)**2/(n_e + i*k_e)**2.

    Rays are refracted with n_o. A quarter wave plate cut with the axis in
    the plate has the thickness wavelength/(4*abs(n_e - n_o)).

    Parameters
    ----------
    n_o, n_e : number, callable or (n_d, V_d)
        Ordinary and extraordinary index.
    axis : 3 vector, optional
        Optic axis in the local frame of the surface in front of the medium
        (default z, along the axis of a centred system).
    k_o, k_e : number or callable, optional
        Extinction coefficients of the two eigenstates (k_e > k_o: a
        dichroic polarizer passing the ordinary polarization).
    """

    def __init__(self, n_o, n_e, axis=(0.0, 0.0, 1.0), k_o=0.0, k_e=0.0):
        self.n_o, self.n_e, self.k_o, self.k_e = n_o, n_e, k_o, k_e
        axis = np.asarray(axis, dtype=float)
        self.axis = axis/np.linalg.norm(axis)

    def __repr__(self):
        return "Uniaxial(n_o={!r}, n_e={!r}, axis={}, k_o={!r}, k_e={!r})".format(
            self.n_o, self.n_e, tuple(self.axis), self.k_o, self.k_e)

    def _indices(self, wavelength_nm):
        n_o = _value(self.n_o, wavelength_nm) + 1j*_value(self.k_o, wavelength_nm)
        n_e = _value(self.n_e, wavelength_nm) + 1j*_value(self.k_e, wavelength_nm)

        return complex(n_o), complex(n_e)

    def index(self, wavelength_nm=None):
        n_o, n_e = self._indices(wavelength_nm)

        return (2*n_o + n_e)/3

    def n_geometric(self, wavelength_nm=None):
        return float(np.real(self._indices(wavelength_nm)[0]))

    def delta_index(self, k, basis, wavelength_nm=None):
        n_o, n_e = self._indices(wavelength_nm)
        o = np.cross(k, self.axis)
        norm = np.linalg.norm(o, axis=-1)
        along = norm < 1e-12  # along the axis: only the ordinary index
        o[along] = basis[along, :, 0]
        o = o/np.linalg.norm(o, axis=-1)[:, None]
        e = np.cross(k, o)
        cos2 = np.sum(k*self.axis, axis=-1)**2
        n_theta = 1/np.sqrt(cos2/n_o**2 + (1 - cos2)/n_e**2)
        # index tensor on the transverse plane, then on the basis:
        M = n_o*o[:, :, None]*o[:, None, :] + n_theta[:, None, None]*e[:, :, None]*e[:, None, :]
        N = np.einsum("nik,nij,njl->nkl", basis, M, basis)

        return N - self.n_geometric(wavelength_nm)*np.eye(2)


class CubicCrystal(Material):
    """Intrinsic (spatial dispersion induced) birefringence of a cubic
    crystal such as CaF2.

    Cubic crystals are isotropic to lowest order, but at short wavelengths
    the finite wave vector makes them weakly birefringent (Burnett, Levine
    and Shirley, Phys. Rev. B 64, 241102 (2001)). The lowest order term
    allowed by cubic symmetry changes the index for a ray along the unit
    vector k (crystal axes) by ::

        dn = -2*delta_n*(B^T diag(k_x**2, k_y**2, k_z**2) B - trace/2)

    on its transverse polarization basis B. It vanishes along the <100> and
    <111> axes and is largest along the <110> axes, where delta_n =
    n([001] polarization) - n([1-10] polarization). Burnett et al. measured
    abs(delta_n) = 3.4e-7 (3.4 nm/cm) at 193 nm and 11.2e-7 at 157 nm for
    CaF2, which made it hard to use at 157 nm. The pattern over the pupil
    has the symmetry of the axis along the lens axis (threefold for [111],
    fourfold for [100]), so clocking elements against each other (e.g. two
    [111] elements by 60 degrees) cancels the non rotationally symmetric
    part.

    Parameters
    ----------
    delta_n : number
        Birefringence along <110> (see above), e.g. -3.4e-7.
    axis : {"111", "100", "110"}, optional
        Crystal axis along the local z axis (the lens axis).
    clocking : number, optional
        Rotation of the crystal about the local z axis, radians.
    n : number, callable or (n_d, V_d), optional
        Mean refractive index, needed when the crystal is the material of a
        raytrace.Surface (not as a medium override that keeps the index of
        the system).
    """

    _AXES = {"100": np.eye(3),
             "111": np.array([[1, -1, 0]/np.sqrt(2), [1, 1, -2]/np.sqrt(6), [1, 1, 1]/np.sqrt(3)]),
             "110": np.array([[0, 0, 1.0], [1, -1, 0]/np.sqrt(2), [1, 1, 0]/np.sqrt(2)])}

    def __init__(self, delta_n, axis="111", clocking=0.0, n=None):
        if axis not in self._AXES:
            raise ValueError("axis must be one of {}".format(sorted(self._AXES)))
        self.delta_n, self.axis, self.clocking, self.n = delta_n, axis, clocking, n
        # _AXES rows: the lens x, y, z axes in crystal coordinates (unclocked)
        self.to_crystal = self._AXES[axis].T @ R_z(-clocking)

    def __repr__(self):
        return "CubicCrystal(delta_n={}, axis={!r}, clocking={}, n={!r})".format(
            self.delta_n, self.axis, self.clocking, self.n)

    def index(self, wavelength_nm=None):
        if self.n is None:
            raise ValueError("CubicCrystal needs n to be used as the material of a surface")

        return _value(self.n, wavelength_nm)

    def index_change(self, k, basis):
        """Change of the refractive index (N, 2, 2) on the transverse basis
        (N, 3, 2) of the rays with unit directions k (N, 3), local frame."""

        k_c = k @ self.to_crystal.T
        B_c = np.einsum("ij,njk->nik", self.to_crystal, basis)
        N = np.einsum("nik,ni,nil->nkl", B_c, k_c**2, B_c)
        N = N - np.trace(N, axis1=1, axis2=2)[:, None, None]/2*np.eye(2)

        return -2*self.delta_n*N

    def delta_index(self, k, basis, wavelength_nm=None):
        dn = 0.0 if self.n is None else complex(self.index(wavelength_nm)) - self.n_geometric(wavelength_nm)

        return self.index_change(k, basis) + dn*np.eye(2)
