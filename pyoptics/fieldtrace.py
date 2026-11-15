#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Field tracing: sequential chains of propagation steps and components.

Field tracing (Wyrowski and Kuhn 2011) generalizes ray tracing: a harmonic
field, not a ray bundle, is traced through the system, and every part of
the system is modelled with the technique that suits it, e.g. a thin element
for a weak phase plate, the angular spectrum of plane waves for a short
free-space step, the far-field operator for a long one and geometrical optics
for a strongly curved wavefront. In the convective single-neighbour
approximation (one source, every component lit by its predecessor only, no
multiple reflections) the output field is the product ::

    V_out = P_{J,J-1} C_{J-1} P_{J-1,J-2} ... C_2 P_{2,1} V_source

of free-space operators P and component operators C. A `Chain` is exactly
this product, written as a list of steps::

    from pyoptics import fieldtrace as ft

    src = ft.gaussian(w0=1.0, wavelength=633e-6, x=x, y=x)
    chain = ft.Chain([ft.ThinLens(100.0), ft.Propagate(100.0), ft.Detector("focus")])
    result = chain.run(src)
    print(result.report())
    I = result["focus"].irradiance

Fields
------
A `Field` stores the transverse electric field as a smooth sampled
amplitude times an analytic phase (see pyoptics.analytic_phase) ::

    E_x,y(x, y) = A_x,y(x, y) exp(i psi(x, y)),
    psi = alpha_x x**2 + alpha_y y**2 + beta_x x + beta_y y + gamma,

so that spherical wavefronts behind lenses or near foci cost no samples. A
scalar (globally polarized) field holds one amplitude A and a Jones vector J,
(E_x, E_y) = J A; a vector field holds A_x and A_y. In a homogeneous
isotropic medium the two transverse components fix the field: E_z, the
magnetic field H and the Poynting vector are computed on demand
(`Field.Ez()`, `Field.H()`, `Field.poynting()`), from the angular spectrum
when the complete field can be sampled on the grid and in the local plane
wave approximation otherwise. Fields can be one-dimensional (y=None): they do
not depend on y (x-z problems such as slits and cylindrical lenses).

Polarization: scalar, vector or chosen automatically
----------------------------------------------------
Propagation in a homogeneous isotropic medium never changes the Jones vector
of a globally polarized field, so free-space steps always run at scalar cost.
`Chain.run(polarization="auto")` (the default) keeps a field scalar as long
as it is globally polarized: a component that makes the polarization vary
across the beam (a spatially varying `Jones` element, a user `Function`)
promotes it to a vector field, and it is demoted again as soon as it is
globally polarized up to the tolerance (e.g. behind a polarizer).
Detectors include E_z in the irradiance when its share of the power exceeds
the tolerance. "scalar" forces one component throughout (cross-polarized
power that a component creates is dropped and reported), "vector" carries
both components throughout. `ChainResult.report()` lists every change.

Free-space operators and their automatic selection
--------------------------------------------------
`Propagate(z, method=...)` offers

- "spw": the spectrum of plane waves (angular spectrum, rigorous), evaluated
  with the semi-analytical Fourier transform (analytic_phase.propagate()):
  the quadratic part of k_z*z stays analytic, so that, unlike the classical
  SPW operator, the sampling need not grow with the curvature of the field;
- "fresnel": the same in the paraxial approximation (exact Fresnel
  propagation, the quadratic phase entirely analytic);
- "fresnel_ft": Fresnel propagation as a single Fourier transform: the
  far-field operator applied to the field times the Fresnel chirp
  exp(i k (x**2 + y**2)/(2 z)) (kept analytic). Its output grid is chosen
  independently of the input's, so a small, strongly diffracting field
  (an obstruction, a pinhole) spreads onto a large window at no cost;
- "farfield": the stationary phase evaluation of the angular spectrum,
  ``u(r) = k z/(2 pi i r) U(k x/r, k y/r) exp(i k r)/r`` (a spherical wave modulated
  by the angular spectrum U, evaluated with the semi-analytical Fourier
  transform);
- "geometric": geometrical optics with amplitude transport
  (analytic_phase.geometric_propagation()), fails at caustics;
- "auto" (default): the selection of Wyrowski and Kuhn (2011, section 3.3).
  The x- and the y-cut of the field with the largest power are propagated
  as one-dimensional fields with every candidate. Each candidate's sampling
  is refined (doubled) until its result changes by less than tol/4; the
  converged number of samples is its numerical effort, and its deviation
  from the converged "spw" result, ``||u - u_spw||/||u_spw||`` (L2 norms), is its
  error. The candidate with the smallest effort N_x*N_y among those with an
  error below tol is used (ties go to the more rigorous one); the 2D field
  is resampled to its N_x, N_y and propagated. The decision is logged
  (`ChainResult.report()`, `Propagate.last_estimate`).

Components
----------
`ThinLens` (analytic: changes alpha only), `ThinElement` (thin element
approximation for a height profile between two media), `PhaseScreen`,
`Aperture`, `Interface` (plane interface into another medium), `Jones`
(constant or spatially varying Jones matrix), `Function` (any callable on a
Field) and `Detector` (records the field and derived quantities).

Lens systems
------------
`Trace(system)` takes the field through the surfaces of a raytrace.System
with geometrical optics: one ray per sample along the local wave vector,
an exact polarization ray trace (Fresnel coefficients or coatings at every
surface) and ray field summation in the output plane with the energy law of
the ray tubes. `Chain.from_system(system, gap=...)` hands the field over to
wave optics a short distance behind the last surface, so that foci and
caustics are propagated with the automatically chosen free-space operator::

    chain = ft.Chain.from_system(ball_lens, gap=0.5) + [ft.Detector("focus")]

Parallel arms and fields in parts
---------------------------------
`Parallel(arm_1, arm_2, ..., weights=...)` splits the field into arms (an
interferometer between a beam splitter and a combiner); `Obstruction` (a
small particle or defect) splits it by Babinet's principle into the
unobstructed field and a small, finely sampled field of opposite sign.
Both give a `FieldSum`: every part keeps its own grid, analytic phase and
automatically chosen operator (with an error budget relative to the whole
sum), and the parts are added coherently only in a `Detector` (on its grid)
or in `Combine`::

    chain = ft.Chain([
        ft.Parallel([ft.Propagate(50), ft.Trace(plate), ft.Propagate(50)],
                    [ft.Propagate(50), ft.Trace(lens),
                     ft.Obstruction(radius=0.1, center=(1, 0.5)), ft.Propagate(50)],
                    weights=(0.5, 0.5)),
        ft.Propagate(50), ft.Detector("camera", x=x, y=x)])

Conventions as in pyoptics.propagators: fields vary as exp(i(k.r - omega*t)),
lengths in one unit (mm throughout pyoptics), wavelengths in vacuum.

References
----------
F. Wyrowski and M. Kuhn, "Introduction to field tracing", J. Mod. Opt. 58,
449-466 (2011).

Z. Wang, S. Zhang, O. Baladron-Zorita, C. Hellmann and F. Wyrowski,
"Application of the semi-analytical Fourier transform to electromagnetic
modeling", Opt. Express 27, 15335-15350 (2019).
"""

import time
import warnings

import numpy as np
from scipy.interpolate import CubicSpline, griddata
from scipy.ndimage import map_coordinates

from . import analytic_phase as _ap
from .analytic_phase import QuadraticPhaseField, _fit_separable_quadratic
from .utils import wavenumber


__all__ = ["Field", "gaussian", "from_array", "Chain", "ChainResult", "Step", "Propagate",
           "ThinLens", "ThinElement", "PhaseScreen", "Aperture", "Interface", "Jones", "Function",
           "Trace", "FieldSum", "Parallel", "Obstruction", "Combine",
           "Detector", "DetectorRecord", "PropagationFailure", "estimate_propagation", "METHODS", "Z0"]


Z0 = 376.730313668  # impedance of free space (Ohm)

METHODS = ("spw", "fresnel", "fresnel_ft", "farfield", "geometric")  # most rigorous first

# rows of the strip that represents a field independent of y
_STRIP_Y = np.array([-1.0, 0.0, 1.0])


def _pair(value):
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if value.shape == (1,):
        return np.array([value[0], value[0]])
    if value.shape != (2,):
        raise ValueError("expected a number or a pair of numbers")
    return value.copy()


def _linear(x, name):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("{} must be a linear coordinate array with at least two samples".format(name))
    dx = (x[-1] - x[0])/(len(x) - 1)
    if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1E-6, atol=1E-12*abs(dx)):
        raise ValueError("{} must be ascending and equidistant".format(name))
    return x


def _spacing(x):
    return (x[-1] - x[0])/(len(x) - 1)


def _window(lo, hi, N):
    """N equidistant samples covering [lo, hi]."""
    return np.linspace(lo, hi, int(N))


# --------------------------------------------------------------------------
# Field
# --------------------------------------------------------------------------

class Field(object):
    """Transverse electric field = sampled amplitude times analytic phase.

    Parameters
    ----------
    amplitude : array
        Shape (Ny, Nx) (scalar, globally polarized along jones) or
        (Ny, Nx, 2) (vector: A_x, A_y). For a one-dimensional field (y=None)
        shape (Nx,) or (Nx, 2).
    x : array
        Linear, equidistant, ascending coordinates.
    y : array or None
        The same for y; None for a field that does not depend on y.
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index of the medium.
    alpha, beta, gamma : optional
        Coefficients of the analytic phase psi (see the module docstring).
    jones : pair, optional
        Jones vector of a scalar field (normalized; default (1, 0)).
    z : number, optional
        Position along the axis (book keeping).
    """

    def __init__(self, amplitude, x, y=None, wavelength=None, n=1.0, alpha=(0.0, 0.0),
                 beta=(0.0, 0.0), gamma=0.0, jones=None, z=0.0):
        if wavelength is None:
            raise ValueError("Field needs a wavelength")
        self.x = _linear(x, "x")
        self.one_d = y is None
        A = np.asarray(amplitude, dtype=complex)
        if self.one_d:
            self._y = _STRIP_Y.copy()
            if A.ndim == 1:
                A = A[:, None]
            if A.ndim != 2 or A.shape[0] != len(self.x) or A.shape[1] not in (1, 2):
                raise ValueError("a one-dimensional field needs an amplitude of shape (Nx,) or (Nx, 2)")
            A = np.repeat(A[None, :, :], len(self._y), axis=0)
        else:
            self._y = _linear(y, "y")
            if A.ndim == 2:
                A = A[:, :, None]
            if A.ndim != 3 or A.shape[:2] != (len(self._y), len(self.x)) or A.shape[2] not in (1, 2):
                raise ValueError("amplitude must have the shape (len(y), len(x)) = ({}, {}) or "
                                 "(len(y), len(x), 2)".format(len(self._y), len(self.x)))
        self._A = A
        self.wavelength = float(wavelength)
        self.n = n
        self.alpha = _pair(alpha)
        self.beta = _pair(beta)
        if self.one_d:
            self.alpha[1] = self.beta[1] = 0.0
        self.gamma = float(gamma)
        if jones is None:
            jones = (1.0, 0.0)
        J = np.asarray(jones, dtype=complex)
        if J.shape != (2,) or not np.linalg.norm(J) > 0:
            raise ValueError("jones must be a non-zero pair")
        self.jones = J/np.linalg.norm(J)
        self.z = float(z)

    # -- basic properties ---------------------------------------------------

    @property
    def y(self):
        """y coordinates (None for a one-dimensional field)."""
        return None if self.one_d else self._y

    @property
    def is_vector(self):
        return self._A.shape[2] == 2

    @property
    def k(self):
        """Wavenumber in the medium."""
        return wavenumber(self.wavelength, self.n)

    @property
    def shape(self):
        """(Nx,) or (Ny, Nx)."""
        return (len(self.x),) if self.one_d else (len(self._y), len(self.x))

    def _user(self, a):
        """Internal (Ny, Nx, ...) array -> the user's shape."""
        return a[1] if self.one_d else a

    def copy(self):
        out = Field.__new__(Field)
        out.__dict__.update(self.__dict__)
        out._A = self._A.copy()
        out.alpha = self.alpha.copy()
        out.beta = self.beta.copy()
        out.jones = self.jones.copy()
        return out

    def _with(self, A, x=None, y=None, alpha=None, beta=None, gamma=None, jones=None, n=None):
        out = self.copy()
        out._A = A
        if x is not None:
            out.x = x
        if y is not None:
            out._y = y
        if alpha is not None:
            out.alpha = np.array(alpha, dtype=float)
        if beta is not None:
            out.beta = np.array(beta, dtype=float)
        if gamma is not None:
            out.gamma = float(gamma)
        if jones is not None:
            J = np.asarray(jones, dtype=complex)
            out.jones = J/np.linalg.norm(J)
        if n is not None:
            out.n = n
        return out

    def __repr__(self):
        return ("Field({}, {}, shape={}, z={:g}, wavelength={:g}, n={}, R={})".format(
            "1d" if self.one_d else "2d", "vector" if self.is_vector else "scalar",
            self.shape, self.z, self.wavelength, self.n, self.radius_of_curvature))

    @property
    def radius_of_curvature(self):
        """(R_x, R_y) of the analytic phase (inf for a plane wavefront)."""
        return tuple(np.inf if a == 0 else self.k/(2*a) for a in self.alpha)

    # -- amplitudes and complete fields --------------------------------------

    def _phase(self):
        """Analytic phase psi, internal shape (Ny, Nx)."""
        px = self.alpha[0]*self.x**2 + self.beta[0]*self.x
        py = self.alpha[1]*self._y**2 + self.beta[1]*self._y
        return py[:, None] + px[None, :] + self.gamma

    def phase(self):
        """Analytic phase psi on the grid."""
        return self._user(self._phase())

    def _components(self):
        """Internal amplitudes (Ny, Nx, 2) of E_x, E_y."""
        if self.is_vector:
            return self._A
        return self._A*self.jones[None, None, :]

    def amplitudes(self):
        """Amplitudes (A_x, A_y) relative to the analytic phase."""
        C = self._components()
        return self._user(C[:, :, 0]), self._user(C[:, :, 1])

    def E(self):
        """Complete transverse field (E_x, E_y) = (A_x, A_y)*exp(i*psi).
        Aliased where psi changes by more than pi between samples (see
        required_samples())."""
        e = np.exp(1j*self.phase())
        Ax, Ay = self.amplitudes()
        return Ax*e, Ay*e

    @property
    def scalar_amplitude(self):
        """Amplitude A of a scalar field (E = jones*A*exp(i*psi))."""
        if self.is_vector:
            raise ValueError("a vector field has no scalar amplitude; see scalarized()")
        return self._user(self._A[:, :, 0])

    def required_samples(self):
        """Samples (N_x, N_y) that the complete field needs on the same window."""
        q = self._qpf(0)
        Nx, Ny = q.required_samples()
        return (Nx, 1) if self.one_d else (Nx, Ny)

    def _qpf(self, c):
        """QuadraticPhaseField of component c (internal arrays)."""
        if c >= self._A.shape[2]:
            raise IndexError("component out of range")
        return QuadraticPhaseField(self._A[:, :, c], self.x, self._y, self.wavelength,
                                   self.alpha, self.beta, self.gamma, self.n)

    @classmethod
    def _from_qpfs(cls, qs, template, z=None):
        """Field from one QuadraticPhaseField per component of template: all are
        expressed on the grid and analytic phase of the first."""
        base = qs[0]
        As = [base.amplitude]
        for q in qs[1:]:
            As.append(_rebase(q, base))
        A = np.stack(As, axis=2)
        out = template._with(A, x=base.x, y=(None if template.one_d else base.y),
                             alpha=base.alpha, beta=base.beta, gamma=base.gamma)
        if template.one_d:
            out.alpha[1] = out.beta[1] = 0.0
        if z is not None:
            out.z = float(z)
        return out

    def resample(self, x, y=None):
        """The field on other linear coordinates (amplitude interpolated,
        analytic phase kept; zero outside the old window)."""
        x = _linear(x, "x")
        y = _STRIP_Y if self.one_d else _linear(y, "y")
        qs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in range(self._A.shape[2]):
                q = self._qpf(c)
                if self.one_d:
                    A = _interp1(q.amplitude[1], self.x, x)
                    qs.append(QuadraticPhaseField(np.repeat(A[None, :], 3, axis=0), x, y,
                                                  self.wavelength, self.alpha, self.beta,
                                                  self.gamma, self.n))
                else:
                    qs.append(q.resample(x, y))
        A = np.stack([q.amplitude for q in qs], axis=2)
        return self._with(A, x=x, y=(None if self.one_d else y))

    # -- polarization ---------------------------------------------------------

    def vectorized(self):
        """The same field with two components (A_x, A_y)."""
        if self.is_vector:
            return self.copy()
        return self._with(self._components().copy())

    def scalarized(self):
        """Closest globally polarized field and the share of the power lost.

        The Jones vector is the dominant singular vector of the 2 x N matrix
        of the samples (A_x, A_y); the lost share is the power in the
        orthogonal polarization relative to the total power (0 for a
        globally polarized field).

        Returns
        -------
        field : Field
            Scalar field.
        lost : float
            Lost power share.
        """
        if not self.is_vector:
            return self.copy(), 0.0
        M = self._A.reshape(-1, 2).T
        total = np.sum(np.abs(M)**2)
        if total == 0:
            return self._with(self._A[:, :, :1].copy(), jones=(1.0, 0.0)), 0.0
        G = M @ M.conj().T
        w, V = np.linalg.eigh(G)
        J = V[:, -1]
        # phase convention: the larger component real and positive
        i = int(np.argmax(np.abs(J)))
        J = J*np.exp(-1j*np.angle(J[i]))
        A = np.tensordot(self._A, J.conj(), axes=([2], [0]))
        lost = max(float(w[0].real/total), 0.0)
        return self._with(A[:, :, None], jones=J), lost

    # -- derived quantities -----------------------------------------------------

    def _local_k(self, A):
        """Local transverse wave vector grad(psi + arg A) (internal shape)."""
        X, Y = np.meshgrid(self.x, self._y)
        kx = 2*self.alpha[0]*X + self.beta[0]
        ky = 2*self.alpha[1]*Y + self.beta[1]
        dx = _spacing(self.x)
        gx = np.zeros(A.shape)
        gx[:, 1:-1] = np.angle(A[:, 2:]*np.conj(A[:, :-2]))/(2*dx)
        gx[:, 0], gx[:, -1] = gx[:, 1], gx[:, -2]
        kx = kx + gx
        if not self.one_d:
            dy = _spacing(self._y)
            gy = np.zeros(A.shape)
            gy[1:-1] = np.angle(A[2:]*np.conj(A[:-2]))/(2*dy)
            gy[0], gy[-1] = gy[1], gy[-2]
            ky = ky + gy
        return kx, ky

    def _sampled(self):
        """True if the complete field is sampled on the grid (psi changes by
        less than pi/2 per sample)."""
        Nx, Ny = self.required_samples()
        return Nx <= 2*len(self.x) and (self.one_d or Ny <= 2*len(self._y))

    def _derived(self, method):
        """Internal amplitudes (relative to psi) of E_z and of k x E/k_0
        (= Z0*H), shape (Ny, Nx) and (Ny, Nx, 3)."""
        C = self._components()
        k = self.k
        if method == "auto":
            method = "spectral" if self._sampled() else "local"
        if method == "local":
            # local plane wave approximation
            A_ref = C[:, :, 0] if np.sum(np.abs(C[:, :, 0])**2) >= np.sum(np.abs(C[:, :, 1])**2) \
                else C[:, :, 1]
            kx, ky = self._local_k(A_ref)
            kz = np.sqrt(np.maximum(np.real(k)**2 - kx**2 - ky**2, 0.0)).astype(complex)
            kz = np.where(kz.real > 0, kz, np.inf)
            Ex, Ey = C[:, :, 0], C[:, :, 1]
            Ez = -(kx*Ex + ky*Ey)/kz
            kz = np.where(np.isinf(kz), 0.0, kz)
            KxE = np.stack([ky*Ez - kz*Ey, kz*Ex - kx*Ez, kx*Ey - ky*Ex], axis=2)
            return Ez, KxE/wavenumber(self.wavelength)
        if method != "spectral":
            raise ValueError("method must be 'auto', 'spectral' or 'local'")
        # angular spectrum on the (zero padded) grid
        e = np.exp(1j*self._phase())
        Ex, Ey = C[:, :, 0]*e, C[:, :, 1]*e
        if self.one_d:
            Ex, Ey, e = Ex[1:2], Ey[1:2], e[1:2]
        Ny, Nx = Ex.shape
        Px, Py = Nx, Ny  # periodic window, as the propagators
        kxs = 2*np.pi*np.fft.fftfreq(Px, _spacing(self.x))
        kys = np.zeros(1) if self.one_d else 2*np.pi*np.fft.fftfreq(Py, _spacing(self._y))
        KX, KY = np.meshgrid(kxs, kys)
        KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
        dk = 2*np.pi/(Px*_spacing(self.x))
        kz_min = np.sqrt(np.abs(k)*dk)
        KZ = np.where(np.abs(KZ) < kz_min, kz_min*np.exp(1j*np.angle(KZ)), KZ)
        fx = np.fft.fft2(Ex, s=(Py, Px))
        fy = np.fft.fft2(Ey, s=(Py, Px))
        fz = -(KX*fx + KY*fy)/KZ
        k0 = wavenumber(self.wavelength)
        parts = [fz, (KY*fz - KZ*fy)/k0, (KZ*fx - KX*fz)/k0, (KX*fy - KY*fx)/k0]
        back = [np.fft.ifft2(p)[:Ny, :Nx]/e for p in parts]
        if self.one_d:
            back = [np.repeat(b, 3, axis=0) for b in back]
        return back[0], np.stack(back[1:], axis=2)

    def Ez(self, method="auto"):
        """Amplitude of E_z relative to the analytic phase, from div E = 0:
        E_z(k) = -(k_x E_x(k) + k_y E_y(k))/k_z. method: "spectral" (angular
        spectrum, needs the complete field sampled on the grid), "local"
        (local plane waves along the local phase gradient) or "auto"."""
        return self._user(self._derived(method)[0])

    def H(self, method="auto"):
        """Amplitudes (H_x, H_y, H_z) relative to the analytic phase, in A per
        unit of E (V/unit): H = k x E/(k_0 Z0) for every plane wave."""
        KxE = self._derived(method)[1]/Z0
        return tuple(self._user(KxE[:, :, i]) for i in range(3))

    def poynting(self, method="auto"):
        """Time averaged Poynting vector (S_x, S_y, S_z) = Re(E x H*)/2."""
        Ez, KxE = self._derived(method)
        C = self._components()
        E = np.stack([C[:, :, 0], C[:, :, 1], Ez], axis=2)
        S = 0.5*np.real(np.cross(E, np.conj(KxE/Z0)))
        return tuple(self._user(S[:, :, i]) for i in range(3))

    def irradiance(self, include_z=False):
        """|E_x|**2 + |E_y|**2 (+ |E_z|**2 if include_z)."""
        C = self._components()
        I = np.sum(np.abs(C)**2, axis=2)
        if include_z:
            I = I + np.abs(self._derived("auto")[0])**2
        return self._user(I)

    def power(self):
        """Integral of |E_x|**2 + |E_y|**2 over the grid (per unit length in y
        for a one-dimensional field)."""
        dA = _spacing(self.x)*(1.0 if self.one_d else _spacing(self._y))
        return float(np.sum(self.irradiance())*dA)

    def ez_share(self):
        """Share of |E_z|**2 in the total |E|**2 (cheap, local plane waves)."""
        Ez = self._derived("local")[0]
        Ez2 = np.sum(np.abs(Ez)**2)
        return float(Ez2/(Ez2 + np.sum(np.abs(self._components())**2) + 1E-300))


def _interp1(A, x, x_new):
    """Cubic interpolation of complex samples, zero outside."""
    out = np.zeros(len(x_new), dtype=complex)
    inside = (x_new >= x[0]) & (x_new <= x[-1])
    if np.any(inside):
        cs = CubicSpline(x, A)
        out[inside] = cs(x_new[inside])
    return out


def _rebase(q, base):
    """Amplitude of the QuadraticPhaseField q on the grid and analytic phase of
    base."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if not (len(q.x) == len(base.x) and np.allclose(q.x, base.x)
                and len(q.y) == len(base.y) and np.allclose(q.y, base.y)):
            q = q.resample(base.x, base.y)
    return q.amplitude*np.exp(1j*(q.phase() - base.phase()))


def _normalize_jones(polarization):
    named = {"x": (1, 0), "y": (0, 1), "45": (1, 1), "-45": (1, -1),
             "rcp": (1, -1j), "lcp": (1, 1j)}
    if isinstance(polarization, str):
        if polarization not in named:
            raise ValueError("unknown polarization {!r}".format(polarization))
        return np.array(named[polarization], dtype=complex)
    return np.asarray(polarization, dtype=complex)


def from_array(field, x, y=None, wavelength=None, n=1.0, R=np.inf, polarization="x", z=0.0):
    """Field from a sampled complex array.

    Parameters
    ----------
    field : array
        Complete field, shape (Ny, Nx) (scalar, polarized along polarization),
        (Ny, Nx, 2) (E_x, E_y), or (Nx,) / (Nx, 2) with y=None.
    x, y, wavelength, n : see Field
    R : number or pair, optional
        Radius of curvature that is split off into the analytic phase
        (R > 0 diverging, R < 0 converging, inf: none).
    polarization : str or pair, optional
        Jones vector of a scalar field: "x", "y", "45", "-45", "rcp", "lcp"
        or a pair.
    z : number, optional
    """
    k = wavenumber(wavelength, n)
    alpha = [0.0 if np.isinf(r) else k/(2*r) for r in _pair(R)]
    f = Field(field, x, y, wavelength, n, alpha=alpha, jones=_normalize_jones(polarization), z=z)
    f._A = f._A*np.exp(-1j*f._phase())[:, :, None]
    return f


def gaussian(w0, wavelength, x, y=None, n=1.0, distance=0.0, polarization="x", power=None, z=0.0):
    """Gaussian beam (TEM00) a given distance behind its waist.

    Parameters
    ----------
    w0 : number
        Waist radius (1/e**2 of the irradiance).
    wavelength : number
        Vacuum wavelength.
    x, y : arrays
        Coordinates (y=None: a one-dimensional beam, Gaussian in x only).
    n : number, optional
        Refractive index.
    distance : number, optional
        Distance from the waist (negative: in front of it, converging).
    polarization : str or pair, optional
        "x", "y", "45", "-45", "rcp", "lcp", a Jones vector, or "radial" /
        "azimuthal" (a vector field: the doughnut of the TEM01* modes
        ``r/w*exp(-r**2/w**2)`` with radial or azimuthal polarization).
    power : number, optional
        Normalize to this power (Field.power()).
    z : number, optional
        Position of the plane (book keeping).
    """
    k = wavenumber(wavelength, n)
    zR = k*w0**2/2
    d = float(distance)
    w = w0*np.sqrt(1 + (d/zR)**2)
    R = np.inf if d == 0 else d*(1 + (zR/d)**2)
    one_d = y is None
    gouy = np.arctan(d/zR)*(0.5 if one_d else 1.0)
    x = _linear(x, "x")
    X, Y = (x, 0.0*x) if one_d else np.meshgrid(x, _linear(y, "y"))
    amp = np.sqrt(w0/w) if one_d else w0/w
    G = amp*np.exp(-(X**2 + Y**2)/w**2)
    alpha = (0.0 if np.isinf(R) else k/(2*R),)*2
    gamma = k*d - gouy
    if isinstance(polarization, str) and polarization in ("radial", "azimuthal"):
        if one_d:
            raise ValueError("radial and azimuthal polarization need a two-dimensional field")
        G = G*np.sqrt(2)/w
        gamma -= np.arctan(d/zR)  # the TEM01* doughnut has twice the Gouy phase
        if polarization == "radial":
            A = np.stack([X*G, Y*G], axis=-1)
        else:
            A = np.stack([-Y*G, X*G], axis=-1)
        f = Field(A, x, y, wavelength, n, alpha=alpha, gamma=gamma, z=z)
    else:
        f = Field(G, x, y, wavelength, n, alpha=alpha, gamma=gamma,
                  jones=_normalize_jones(polarization), z=z)
    if power is not None:
        f._A = f._A*np.sqrt(power/f.power())
    return f


# --------------------------------------------------------------------------
# free-space operators
# --------------------------------------------------------------------------

class PropagationFailure(Exception):
    """A free-space operator cannot be applied (e.g. geometric optics through
    a caustic, the far field for z <= 0)."""


def _spectrum(q, one_d, oversample=4):
    """Spectrum of q (semi-analytical FT), oversampled so that its amplitude
    can be interpolated."""
    def dense(n):
        return max(1, min(oversample, 4096//n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = _ap.semi_analytical_FT(q)
        ox = dense(len(S.x))
        oy = 1 if one_d else dense(len(S.y))
        if ox > 1 or oy > 1:
            kx = _window(S.x[0], S.x[-1], (len(S.x) - 1)*ox + 1)
            ky = S.y if oy == 1 else _window(S.y[0], S.y[-1], (len(S.y) - 1)*oy + 1)
            S = _ap.semi_analytical_FT(q, kx, ky)
    return S


def _support(values, coords, eps=1E-6):
    """Interval of coords that holds all but the share eps of the (marginal)
    energy values at either end."""
    c = np.cumsum(values)
    if c[-1] <= 0:
        return coords[0], coords[-1]
    c = c/c[-1]
    i0 = int(np.searchsorted(c, eps))
    i1 = int(np.searchsorted(c, 1 - eps))
    return coords[min(i0, len(coords) - 1)], coords[min(i1, len(coords) - 1)]


def _spectrum_support(q, one_d, eps=1E-6):
    """The spectrum of q and (kx_min, kx_max, ky_min, ky_max) of the support
    of its propagating part."""
    S = _spectrum(q, one_d)
    KX, KY = np.meshgrid(S.x, S.y)
    I = np.abs(S.amplitude)**2*(KX**2 + KY**2 < np.real(q.k)**2)
    kx0, kx1 = _support(np.sum(I, axis=0), S.x, eps)
    ky0, ky1 = _support(np.sum(I, axis=1), S.y, eps)
    return S, (kx0, kx1, ky0, ky1)


def _footprint(f, z, eps=1E-6):
    """Window [x_lo, x_hi, y_lo, y_hi] that holds the field after the distance
    z: the input support swept by the range of directions of its spectrum."""
    q = f._qpf(int(np.argmax(np.sum(np.abs(f._A)**2, axis=(0, 1)))))
    I = np.abs(q.amplitude)**2
    x_lo, x_hi = _support(np.sum(I, axis=0), f.x, eps)
    S, (kx0, kx1, ky0, ky1) = _spectrum_support(q, f.one_d, eps)
    k = np.real(f.k)

    def sweep(lo, hi, k0, k1):
        t0 = k0/np.sqrt(max(k**2 - k0**2, 1E-30*k**2))
        t1 = k1/np.sqrt(max(k**2 - k1**2, 1E-30*k**2))
        ends = [lo + z*t0, lo + z*t1, hi + z*t0, hi + z*t1]
        return min(ends), max(ends)

    xs = sweep(x_lo, x_hi, kx0, kx1)
    if f.one_d:
        return xs + (f._y[0], f._y[-1])
    y_lo, y_hi = _support(np.sum(I, axis=1), f._y, eps)
    return xs + sweep(y_lo, y_hi, ky0, ky1)


def _pad_to(f, window, margin=0.1):
    """Zero pad f (same spacing) so that its grid covers window, if needed."""
    x_lo, x_hi, y_lo, y_hi = window

    def grow(c, lo, hi):
        d = _spacing(c)
        span = hi - lo
        lo, hi = lo - margin*span, hi + margin*span
        n_lo = max(int(np.ceil((c[0] - lo)/d)), 0)
        n_hi = max(int(np.ceil((hi - c[-1])/d)), 0)
        return np.concatenate([c[0] - d*np.arange(n_lo, 0, -1), c, c[-1] + d*np.arange(1, n_hi + 1)]), n_lo, n_hi

    x, nxl, nxh = grow(f.x, x_lo, x_hi)
    if f.one_d:
        y, nyl, nyh = f._y, 0, 0
    else:
        y, nyl, nyh = grow(f._y, y_lo, y_hi)
    if nxl == nxh == nyl == nyh == 0:
        return f
    A = np.pad(f._A, ((nyl, nyh), (nxl, nxh), (0, 0)))
    if f.one_d:
        A[:] = A[1:2]
    return f._with(A, x=x, y=(None if f.one_d else y))


def _farfield_q(q, z, one_d, N=None, paraxial=False):
    """Far-field operator for one component (QuadraticPhaseField); paraxial:
    Fraunhofer form (k x/z, exp(i k (z + rho**2/(2 z))))."""
    if z <= 0:
        raise PropagationFailure("the far-field operator needs z > 0")
    k = q.k
    kr = np.real(k)
    S, (kx0, kx1, ky0, ky1) = _spectrum_support(q, one_d)
    lim = 0.999*kr

    def window(k0, k1, kgrid, Nw):
        s0, s1 = np.clip(k0, -lim, lim)/kr, np.clip(k1, -lim, lim)/kr
        if paraxial:
            a, b = z*s0, z*s1
        else:
            a, b = z*s0/np.sqrt(1 - s0**2), z*s1/np.sqrt(1 - s1**2)
        span = max(b - a, 1E-12*abs(z))
        pad = 0.1*span + z*abs(_spacing(kgrid))/kr
        return _window(a - pad, b + pad, Nw)

    Nx = len(q.x) if N is None else N[0]
    xo = window(kx0, kx1, S.x, Nx)
    if one_d:
        yo = _STRIP_Y
        X = xo[None, :]
        r = z + X**2/(2*z) if paraxial else np.sqrt(X**2 + z**2)
        KX, KY = k*X/(z if paraxial else r), np.zeros_like(X)
    else:
        Ny = len(q.y) if N is None else N[1]
        yo = window(ky0, ky1, S.y, Ny)
        X, Y = np.meshgrid(xo, yo)
        r = z + (X**2 + Y**2)/(2*z) if paraxial else np.sqrt(X**2 + Y**2 + z**2)
        rm = z if paraxial else r
        KX, KY = k*X/rm, k*Y/rm
    # spectrum amplitude at (KX, KY): cubic interpolation of the smooth
    # amplitude, analytic phase exact
    ix = (np.real(KX) - S.x[0])/_spacing(S.x)
    if one_d:
        row = int(np.argmin(np.abs(S.y)))
        Arow = S.amplitude[row]
        amp = (map_coordinates(Arow.real[None, :], [np.zeros_like(ix), ix], order=3, cval=0.0)
               + 1j*map_coordinates(Arow.imag[None, :], [np.zeros_like(ix), ix], order=3, cval=0.0))
        # the strip's transform holds the integral over its rows
        amp = amp/(len(q.y)*_spacing(q.y))
        prefactor = ((1.0 if paraxial else z/r)*np.sqrt(k/(2*np.pi*(z if paraxial else r)))
                     *np.exp(-1j*np.pi/4))
    else:
        iy = (np.real(KY) - S.y[0])/_spacing(S.y)
        amp = (map_coordinates(S.amplitude.real, [iy, ix], order=3, cval=0.0)
               + 1j*map_coordinates(S.amplitude.imag, [iy, ix], order=3, cval=0.0))
        prefactor = k/(2j*np.pi*z) if paraxial else k*(z/r)/(2j*np.pi)/r
    phi = (S.alpha[0]*KX**2 + S.beta[0]*KX + S.alpha[1]*KY**2 + S.beta[1]*KY + S.gamma
           + k*r)
    phi = np.real(phi)
    u = prefactor*amp
    W = np.abs(u)**2
    if one_d:
        Xf, Yf = np.broadcast_to(X, (1, len(xo))), np.zeros((1, len(xo)))
    else:
        Xf, Yf = X, Y
    a, b, c, d, e = _fit_separable_quadratic(phi, Xf, Yf, W*(W > 1E-12*W.max()))
    quad = a*Xf**2 + b*Yf**2 + c*Xf + d*Yf + e
    A = u*np.exp(1j*(phi - quad))
    if one_d:
        A = np.repeat(A, 3, axis=0)
        b = d = 0.0
    return QuadraticPhaseField(A, xo, yo, q.wavelength, (a, b), (c, d), e, q.n)


def _fresnel_ft_q(q, z, one_d, N=None):
    """Single Fourier transform Fresnel operator for one component."""
    if z <= 0:
        raise PropagationFailure("the single-FT Fresnel operator needs z > 0")
    c = np.real(q.k)/(2*z)
    alpha = (q.alpha[0] + c, 0.0 if one_d else q.alpha[1] + c)
    qc = QuadraticPhaseField(q.amplitude, q.x, q.y, q.wavelength, alpha, q.beta, q.gamma, q.n)
    return _farfield_q(qc, z, one_d, N, paraxial=True)


def _unwrapped_phase(q):
    """psi + unwrapped arg(A) of a QuadraticPhaseField (2D unwrapping along
    the brightest row and column)."""
    A = q.amplitude
    I = np.abs(A)**2
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    ph = np.unwrap(np.angle(A), axis=1)
    col = np.unwrap(ph[:, ix])
    ph = ph + (col - ph[:, ix])[:, None]
    ph = ph - 2*np.pi*np.round((ph[iy, ix] - np.angle(A[iy, ix]))/(2*np.pi))
    return q.phase() + ph


def _geometric_q(q, z, one_d, threshold=1E-4, strong=1E-3):
    """Geometrical optics operator for one component."""
    phase = _unwrapped_phase(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            Xz, Yz, uA, uphi = _ap.geometric_propagation(np.abs(q.amplitude), phase, q.x, q.y, z,
                                                         q.wavelength, q.n)
        except ValueError as err:
            raise PropagationFailure(str(err))
    I = np.abs(q.amplitude)**2
    lit = (I > threshold*I.max()) & np.isfinite(uA) & np.isfinite(Xz) & np.isfinite(Yz)
    N = len(q.x)
    if one_d:
        r = 1
        xs, a, p = Xz[r], uA[r], uphi[r]
        m = lit[r]

        def monotone(mask):
            d = np.diff(xs[mask])
            return len(d) == 0 or np.all(d > 0) or np.all(d < 0)

        if not monotone(m):
            # weak samples (e.g. ringing with random phase) may fold over:
            # keep the strong ones
            m = lit[r] & (I[r] > strong*I.max())
            if not monotone(m):
                raise PropagationFailure("geometric propagation crosses a caustic")
        order = np.argsort(xs[m])
        xs_m, a_m, p_m = xs[m][order], a[m][order], p[m][order]
        span = xs_m[-1] - xs_m[0]
        xo = _window(xs_m[0] - 0.02*span, xs_m[-1] + 0.02*span, N)
        # the quadratic part of the phase stays analytic; only the smooth
        # remainder and the amplitude are interpolated (cubic)
        W = np.abs(a_m)**2
        a2, b2, c2, d2, e2 = _fit_separable_quadratic(p_m, xs_m, np.zeros_like(xs_m), W)
        rest = p_m - (a2*xs_m**2 + c2*xs_m + e2)
        inside = (xo >= xs_m[0]) & (xo <= xs_m[-1])
        A_o = np.zeros(N, dtype=complex)
        if len(xs_m) >= 4:
            spline = CubicSpline(xs_m, np.stack([a_m.real, a_m.imag, rest], axis=1))
            v = spline(xo[inside])
            A_o[inside] = (v[:, 0] + 1j*v[:, 1])*np.exp(1j*v[:, 2])
        A = np.repeat(A_o[None, :], 3, axis=0)
        return QuadraticPhaseField(A, xo, _STRIP_Y, q.wavelength, (a2, 0.0), (c2, 0.0), e2, q.n)
    # two dimensions: the map must keep its orientation (no caustic)
    jx = np.gradient(Xz, q.y, q.x)
    jy = np.gradient(Yz, q.y, q.x)
    det = jx[1]*jy[0] - jx[0]*jy[1]
    def oriented(mask):
        sg = np.sign(det[mask])
        return len(sg) == 0 or np.all(sg > 0) or np.all(sg < 0)

    if not oriented(lit):
        lit = lit & (I > strong*I.max())
        if not oriented(lit):
            raise PropagationFailure("geometric propagation crosses a caustic")
    pts = np.stack([Xz[lit], Yz[lit]], axis=1)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    xo = _window(x0 - 0.02*(x1 - x0), x1 + 0.02*(x1 - x0), N)
    yo = _window(y0 - 0.02*(y1 - y0), y1 + 0.02*(y1 - y0), len(q.y))
    Xo, Yo = np.meshgrid(xo, yo)
    # the quadratic part of the phase stays analytic; only the smooth
    # remainder and the amplitude are interpolated (cubic, scattered points)
    Xl, Yl, Pl, Al = Xz[lit], Yz[lit], uphi[lit], uA[lit]
    a2, b2, c2, d2, e2 = _fit_separable_quadratic(Pl, Xl, Yl, np.abs(Al)**2)
    rest = Pl - (a2*Xl**2 + b2*Yl**2 + c2*Xl + d2*Yl + e2)
    vals = np.stack([Al.real, Al.imag, rest], axis=1)
    out = griddata(pts, vals, (Xo, Yo), method="cubic")
    out = np.nan_to_num(out, nan=0.0)
    A = (out[..., 0] + 1j*out[..., 1])*np.exp(1j*out[..., 2])
    return QuadraticPhaseField(A, xo, yo, q.wavelength, (a2, b2), (c2, d2), e2, q.n)


def _covers(out, window, slack=0.05):
    """Does the output grid cover the window (up to slack times its span on
    each side: the window is an energy quantile estimate)?"""
    x_lo, x_hi, y_lo, y_hi = window
    sx = slack*(x_hi - x_lo)
    ok = out.x[0] <= x_lo + sx and out.x[-1] >= x_hi - sx
    if not out.one_d:
        sy = slack*(y_hi - y_lo)
        ok = ok and out._y[0] <= y_lo + sy and out._y[-1] >= y_hi - sy
    return ok


def _apply_operator(f, z, method, N):
    """One free-space operator on every component of f."""
    outs = []
    for c in range(f._A.shape[2]):
        q = f._qpf(c)
        if method == "spw":
            outs.append(_ap.propagate(q, z, paraxial=False))
        elif method == "fresnel":
            outs.append(_ap.propagate(q, z, paraxial=True))
        elif method == "farfield":
            outs.append(_farfield_q(q, z, f.one_d, N))
        elif method == "fresnel_ft":
            outs.append(_fresnel_ft_q(q, z, f.one_d, N))
        elif method == "geometric":
            outs.append(_geometric_q(q, z, f.one_d))
        else:
            raise ValueError("unknown method {!r}; use one of {}".format(method, METHODS + ("auto",)))
    if f.one_d:
        for o in outs:
            o.amplitude[:] = o.amplitude[1:2]
    return Field._from_qpfs(outs, f, z=f.z + z)


def _eps(tol):
    """Energy share outside the footprint that an error budget tol allows."""
    return max(1E-6, (tol/10)**2)


def _propagate(f, z, method, N=None, eps=1E-6):
    """Apply one free-space operator to a Field.

    The spectral operators ("spw", "fresnel") keep the grid near the input
    plane (FFT branch of the transforms); if the beam then leaves the window,
    the input is zero padded (same spacing) to the window the beam reaches.

    Returns
    -------
    field : Field
    notes : list of str
        Warnings of the transforms.
    used : Field
        The input actually propagated (padded or not).
    """
    notes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out = _apply_operator(f, z, method, N)
        if method in ("spw", "fresnel"):
            window = _footprint(f, z, eps)
            if not _covers(out, window):
                f = _pad_to(f, window)
                del caught[:]
                out = _apply_operator(f, z, method, N)
    for w in caught:
        msg = str(w.message).split("\n")[0]
        if msg not in notes:
            notes.append(msg)
    return out, notes, f


# --------------------------------------------------------------------------
# automatic selection
# --------------------------------------------------------------------------

def _cuts(f):
    """One-dimensional fields along the x and the y cut with the largest
    power (through the dominant component)."""
    c = int(np.argmax(np.sum(np.abs(f._A)**2, axis=(0, 1))))
    A = f._A[:, :, c]
    I = np.abs(A)**2
    iy = int(np.argmax(np.sum(I, axis=1)))
    y0 = f._y[iy]
    gx = f.gamma + f.alpha[1]*y0**2 + f.beta[1]*y0
    cut_x = Field(A[iy], f.x, None, f.wavelength, f.n, alpha=(f.alpha[0], 0.0),
                  beta=(f.beta[0], 0.0), gamma=gx, z=f.z)
    if f.one_d:
        return cut_x, None
    ix = int(np.argmax(np.sum(I, axis=0)))
    x0 = f.x[ix]
    gy = f.gamma + f.alpha[0]*x0**2 + f.beta[0]*x0
    cut_y = Field(A[:, ix], f._y, None, f.wavelength, f.n, alpha=(f.alpha[1], 0.0),
                  beta=(f.beta[1], 0.0), gamma=gy, z=f.z)
    return cut_x, cut_y


def _deviation(a, ref):
    """``||a - ref||/||ref||`` on the grid of ref (one-dimensional fields)."""
    fa = a.resample(ref.x)
    ea = fa.scalar_amplitude*np.exp(1j*fa.phase())
    er = ref.scalar_amplitude*np.exp(1j*ref.phase())
    nr = np.linalg.norm(er)
    return float(np.linalg.norm(ea - er)/nr) if nr > 0 else np.inf


def _refine(f, method, fx, fy=1):
    """Refine the sampling of an operator by the factors fx, fy (per axis).

    What limits each operator differs: the spectral operators ("spw",
    "fresnel") sample the spectrum with the spacing 2*pi/window, so their
    input window is enlarged (zero padding, same spacing); the far-field
    operator gets more output samples; geometrical optics gets a denser
    input grid. Returns (field, N) with N the output samples of the far
    field (None otherwise)."""
    if fx == 1 and fy == 1:
        return f, None
    if method in ("spw", "fresnel"):
        def grown(c, fac):
            span = (c[-1] - c[0])*fac
            mid = 0.5*(c[0] + c[-1])
            return mid - span/2, mid + span/2
        x_lo, x_hi = grown(f.x, fx)
        y_lo, y_hi = (f._y[0], f._y[-1]) if f.one_d else grown(f._y, fy)
        return _pad_to(f, (x_lo, x_hi, y_lo, y_hi), margin=0.0), None
    if method in ("farfield", "fresnel_ft"):
        return f, (len(f.x)*fx, len(f._y)*fy)
    x = _window(f.x[0], f.x[-1], (len(f.x) - 1)*fx + 1)
    y = None if f.one_d else _window(f._y[0], f._y[-1], (len(f._y) - 1)*fy + 1)
    return f.resample(x, y), None


def _converge(cut, z, method, tol, max_N):
    """Propagate the cut with method, doubling its sampling (see _refine())
    until the result changes by less than tol/4. Returns (result, effort N,
    refinement factor, converged)."""
    prev, prev_N = None, None
    factor = 1
    while True:
        f, N_out = _refine(cut, method, factor)
        try:
            out, _, used = _propagate(f, z, method, N_out, _eps(tol))
        except PropagationFailure:
            return None, np.inf, factor, False
        N = len(out.x) if method in ("farfield", "fresnel_ft") else len(used.x)
        if prev is not None:
            # on the finer grid: a coarse result must interpolate well
            change = _deviation(prev, out)
            if change < tol/4:
                return prev, prev_N, factor//2, True
        if N*2 > max_N:
            return out, N, factor, False
        prev, prev_N = out, N
        factor *= 2


def estimate_propagation(field, z, candidates=METHODS, tol=1E-2, max_N=2**14):
    """Error and effort of free-space operators for one step (1D cuts).

    Parameters
    ----------
    field : Field
    z : number
        Distance.
    candidates : sequence of str, optional
        Operators to judge (from METHODS).
    tol : number, optional
        Accepted relative L2 deviation from the converged SPW result.
    max_N : int, optional
        Largest number of samples per cut.

    Returns
    -------
    choice : str
        The selected operator.
    table : dict
        Per operator: {"error": deviation (max over the cuts), "N": (N_x, N_y),
        "effort": N_x*N_y, "converged": bool}. table["reference"] tells
        whether the SPW reference converged.
    """
    cuts = [c for c in _cuts(field) if c is not None]
    table = {m: {"error": 0.0, "N": [], "factor": [], "converged": True} for m in candidates}
    ref_ok = True
    for cut in cuts:
        ref, N_ref, fac_ref, ok = _converge(cut, z, "spw", tol, max_N)
        ref_ok = ref_ok and ok
        for m in candidates:
            if m == "spw":
                res, N, fac, conv = ref, N_ref, fac_ref, ok
            else:
                res, N, fac, conv = _converge(cut, z, m, tol, max_N)
            if res is None or ref is None:
                err = np.inf
            else:
                err = 0.0 if m == "spw" else _deviation(res, ref)
            table[m]["error"] = max(table[m]["error"], err)
            table[m]["N"].append(N)
            table[m]["factor"].append(fac)
            table[m]["converged"] = table[m]["converged"] and conv
    for m in candidates:
        Ns = table[m]["N"]
        if field.one_d:
            Ns = Ns + [1]
        table[m]["N"] = tuple(Ns)
        table[m]["factor"] = tuple(table[m]["factor"]) + ((1,) if field.one_d else ())
        table[m]["effort"] = float(np.prod([float(v) for v in Ns]))
    ok = [m for m in candidates if table[m]["error"] <= tol and np.isfinite(table[m]["effort"])]
    rank = {m: i for i, m in enumerate(METHODS)}
    if ok:
        choice = min(ok, key=lambda m: (table[m]["effort"], rank[m]))
    else:
        choice = min(candidates, key=lambda m: (table[m]["error"], rank[m]))
    table["reference"] = {"converged": ref_ok}
    return choice, table


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

class Step(object):
    """A step of a chain: apply(field, ctx) returns (field, info), info a dict
    with optional keys "method", "error", "notes"."""

    label = "step"

    def apply(self, field, ctx):
        raise NotImplementedError

    def __repr__(self):
        return self.label


class Propagate(Step):
    """Free-space propagation by z in the field's medium.

    Parameters
    ----------
    z : number
        Distance (negative: backwards; not with "farfield").
    method : str, optional
        "auto" (default), "spw", "fresnel", "fresnel_ft", "farfield" or
        "geometric"; see
        the module docstring.
    tol : number, optional
        Error budget of the automatic selection (default: the chain's tol).
    candidates : sequence, optional
        Operators the selection may choose from.
    max_samples : int, optional
        Largest grid the automatic selection may use; beyond it the step
        fails with PropagationFailure instead of exhausting the memory.
    """

    def __init__(self, z, method="auto", tol=None, candidates=METHODS, max_samples=2**24):
        self.z = float(z)
        self.method = method
        self.tol = tol
        self.candidates = tuple(candidates)
        self.max_samples = int(max_samples)
        self.last_estimate = None
        self.label = "propagate {:g}".format(self.z)

    def apply(self, field, ctx):
        info = {"notes": []}
        method = self.method
        if method == "auto":
            tol = ctx.get("tol_part", ctx.get("tol", 1E-2)) if self.tol is None else self.tol
            method, table = estimate_propagation(field, self.z, self.candidates, tol)
            self.last_estimate = table
            info["error"] = table[method]["error"]
            info["candidates"] = {m: (table[m]["error"], table[m]["effort"]) for m in self.candidates}
            fx, fy = table[method]["factor"]
            size = int(np.prod([int(np.ceil(v*f)) for v, f in zip(field.shape[::-1], (fx, fy))]))
            if size > self.max_samples:
                raise PropagationFailure(
                    "propagate {}: {} needs about {:.1e} samples (more than max_samples={:.1e}); "
                    "hard edges or fine structure in a large field? Smooth the edges, split the "
                    "field (FieldSum, Obstruction) or loosen tol".format(
                        self.z, method, size, self.max_samples))
            field, N_out = _refine(field, method, fx, fy)
            out, notes, _ = _propagate(field, self.z, method, N_out, _eps(tol))
            if not table["reference"]["converged"]:
                info["notes"].append("SPW reference not converged: errors are estimates")
            if table[method]["error"] > tol:
                info["notes"].append("no operator met tol={:g}".format(tol))
        else:
            out, notes, _ = _propagate(field, self.z, method)
        info["method"] = method
        info["notes"] += notes
        return out, info


class ThinLens(Step):
    """Ideal thin lens of focal length f (in the field's medium): psi gets
    -k*(x**2 + y**2)/(2*f), analytically (no samples needed).

    Parameters
    ----------
    f : number
        Focal length (> 0 converging).
    axis : str, optional
        "xy" (default), "x" or "y" (cylindrical lens).
    """

    def __init__(self, f, axis="xy"):
        self.f = float(f)
        self.axis = axis
        self.label = "thin lens f={:g}".format(self.f)

    def apply(self, field, ctx):
        da = -field.k/(2*self.f)
        alpha = field.alpha.copy()
        if "x" in self.axis:
            alpha[0] += da
        if "y" in self.axis and not field.one_d:
            alpha[1] += da
        return field._with(field._A.copy(), alpha=alpha), {"method": "analytic"}


def _evaluate(func_or_array, field):
    """Values of a callable f(x, y) (f(x) for 1D fields) or an array on the
    field's grid, internal shape (Ny, Nx)."""
    if callable(func_or_array):
        if field.one_d:
            v = np.asarray(func_or_array(field.x))
            return np.broadcast_to(v, (len(field._y), len(field.x)))
        X, Y = np.meshgrid(field.x, field._y)
        return np.asarray(func_or_array(X, Y))
    v = np.asarray(func_or_array)
    if field.one_d:
        if v.shape != (len(field.x),):
            raise ValueError("array does not match the grid; pass a callable to resample")
        return np.broadcast_to(v, (len(field._y), len(field.x)))
    if v.shape != (len(field._y), len(field.x)):
        raise ValueError("array does not match the grid; pass a callable to resample")
    return v


class PhaseScreen(Step):
    """Multiply by exp(i*phase(x, y)) (callable or array on the grid)."""

    def __init__(self, phase, label="phase screen"):
        self.phase = phase
        self.label = label

    def apply(self, field, ctx):
        ph = _evaluate(self.phase, field)
        return field._with(field._A*np.exp(1j*ph)[:, :, None]), {"method": "TEA"}


class ThinElement(Step):
    """Thin element approximation (TEA) of a height profile h(x, y).

    The element of index n_element replaces the medium n_medium over the
    height h; the field gets exp(i*k_0*(n_element - n_medium)*h) (absorption
    for complex indices) and optionally the Fresnel transmission of both
    faces at normal incidence, 4*n*n_0/(n + n_0)**2.

    Parameters
    ----------
    height : callable or array
        h(x, y) (h(x) for one-dimensional fields) or values on the grid.
    n_element : number
    n_medium : number, optional
        Default: the field's medium.
    fresnel : bool, optional
        Include the Fresnel losses.
    """

    def __init__(self, height, n_element, n_medium=None, fresnel=False, label="thin element"):
        self.height = height
        self.n_element = n_element
        self.n_medium = n_medium
        self.fresnel = fresnel
        self.label = label

    def apply(self, field, ctx):
        n0 = field.n if self.n_medium is None else self.n_medium
        h = _evaluate(self.height, field)
        t = np.exp(1j*wavenumber(field.wavelength)*(self.n_element - n0)*h)
        if self.fresnel:
            t = t*4*self.n_element*n0/(self.n_element + n0)**2
        return field._with(field._A*t[:, :, None]), {"method": "TEA"}


class Aperture(Step):
    """Aperture: circular (radius), rectangular (half_widths) or any
    transmission mask(x, y)."""

    def __init__(self, radius=None, half_widths=None, center=(0.0, 0.0), mask=None,
                 label="aperture"):
        if sum(v is not None for v in (radius, half_widths, mask)) != 1:
            raise ValueError("give exactly one of radius, half_widths, mask")
        self.radius, self.half_widths, self.center, self.mask = radius, half_widths, center, mask
        self.label = label

    def apply(self, field, ctx):
        X, Y = np.meshgrid(field.x - self.center[0], field._y - self.center[1])
        if field.one_d:
            Y = np.zeros_like(X)
        if self.mask is not None:
            T = _evaluate(self.mask, field)
        elif self.radius is not None:
            T = (X**2 + Y**2 <= self.radius**2).astype(float)
        else:
            hx, hy = _pair(self.half_widths)
            T = ((np.abs(X) <= hx) & (np.abs(Y) <= hy)).astype(float)
        return field._with(field._A*T[:, :, None]), {"method": "mask"}


class Interface(Step):
    """Plane interface (normal to z) into a medium of index n2. The phase is
    continuous; with fresnel=True the field is multiplied by the normal
    incidence transmission 2*n1/(n1 + n2)."""

    def __init__(self, n2, fresnel=False, label=None):
        self.n2 = n2
        self.fresnel = fresnel
        self.label = label or "interface n={}".format(n2)

    def apply(self, field, ctx):
        t = 2*field.n/(field.n + self.n2) if self.fresnel else 1.0
        return field._with(field._A*t, n=self.n2), {"method": "plane interface"}


class Jones(Step):
    """Jones matrix: a constant (2, 2) array or a callable J(x, y) (J(x) for
    one-dimensional fields) returning shape (..., 2, 2). A constant matrix
    keeps a scalar field scalar."""

    def __init__(self, matrix, label="Jones element"):
        self.matrix = matrix
        self.label = label

    def apply(self, field, ctx):
        if not callable(self.matrix):
            J = np.asarray(self.matrix, dtype=complex)
            if J.shape != (2, 2):
                raise ValueError("a constant Jones matrix must have the shape (2, 2)")
            if not field.is_vector:
                v = J @ field.jones
                s = np.linalg.norm(v)
                if s == 0:
                    return field._with(field._A*0.0), {"method": "Jones"}
                return field._with(field._A*s, jones=v/s), {"method": "Jones"}
            A = np.einsum("ij,yxj->yxi", J, field._A)
            return field._with(A), {"method": "Jones"}
        if field.one_d:
            M = np.asarray(self.matrix(field.x), dtype=complex)
            M = np.broadcast_to(M, (len(field._y),) + M.shape)
        else:
            X, Y = np.meshgrid(field.x, field._y)
            M = np.asarray(self.matrix(X, Y), dtype=complex)
        A = np.einsum("yxij,yxj->yxi", M, field._components())
        return field._with(A), {"method": "Jones"}


class Function(Step):
    """Any callable func(field) -> field (e.g. a rigorous solver)."""

    def __init__(self, func, label=None):
        self.func = func
        self.label = label or getattr(func, "__name__", "function")

    def apply(self, field, ctx):
        return self.func(field), {"method": "user"}


class Trace(Step):
    """Geometrical optics field tracing through a raytrace.System.

    The field (in the object space medium, in a plane normal to the axis a
    distance in front of the first vertex) is converted into a structured
    grid of rays along its local phase gradient (local plane waves). The rays
    are traced exactly through the surfaces with their polarization
    (System.trace_polarization(): Fresnel coefficients or coatings at every
    surface, i.e. the local plane interface approximation) to the image
    plane of the system. There the field is assembled by ray field summation
    (raywave.rays_to_field()): every triangle of neighbouring rays is a tube
    whose irradiance follows from the energy law, ::

        n_out cos_out |E_out|**2 dA_out = T n_in cos_in |E_in|**2 dA_in

    (T: power transmittance of the ray), its phase from the optical path,
    and the tubes of all sheets covering a sample are added (caustics, with
    the Maslov phase). The quadratic part of the output phase is kept
    analytic. The polarization is traced vectorially: an x polarized input
    generally leaves with E_y (and E_z) components.

    Parameters
    ----------
    system : raytrace.System
        The surfaces; its image plane is the output plane (the thickness of
        the last surface).
    distance : number, optional
        Distance of the field plane in front of the first vertex.
    n_rays : int, optional
        Rays per axis (over the lit part of the field; default: the field's
        samples, at most 256).
    x_out, y_out : arrays, optional
        Output grid in the image plane (default: the footprint of the rays
        with the input's number of samples, or more where the residual
        phase needs them, at most max_N per axis).
    clip : bool, optional
        Vignette rays outside the semi diameters.
    coatings, media : dicts, optional
        Passed to System.trace_polarization().
    """

    def __init__(self, system, distance=0.0, n_rays=None, x_out=None, y_out=None, clip=True,
                 coatings=None, media=None, max_N=2048, label=None):
        self.system = system
        self.max_N = int(max_N)
        self.distance = float(distance)
        self.n_rays = n_rays
        self.x_out, self.y_out = x_out, y_out
        self.clip = clip
        self.coatings, self.media = coatings, media
        self.label = label or "trace {} surfaces".format(len(system.surfaces))

    def apply(self, field, ctx):
        from .raytrace import Rays
        from .raywave import rays_to_field

        if field.one_d:
            raise ValueError("Trace needs a two-dimensional field")
        sysm = self.system
        notes = []
        n_in = sysm.n[0]
        if not np.isclose(np.real(field.n), n_in):
            notes.append("field medium n={} differs from the object space n={}".format(field.n, n_in))
        k0 = wavenumber(field.wavelength)
        if not np.isclose(field.wavelength, sysm.wavelength, rtol=1E-9):
            notes.append("field wavelength differs from the system's: tracing at the system's indices")
        C = field._components()
        I = np.sum(np.abs(C)**2, axis=2)
        lit = I > 1E-8*I.max()
        rows, cols = np.flatnonzero(np.any(lit, axis=1)), np.flatnonzero(np.any(lit, axis=0))
        # one ray beyond the lit region on each side
        r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 1, I.shape[0] - 1)
        c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 1, I.shape[1] - 1)
        nr = self.n_rays or min(256, max(r1 - r0, c1 - c0) + 1)
        iy = np.unique(np.round(np.linspace(r0, r1, nr)).astype(int))
        ix = np.unique(np.round(np.linspace(c0, c1, nr)).astype(int))
        # local wave vector of the dominant component: gradient of the
        # analytic phase plus Im(A* grad A)/|A|**2 (no phase unwrapping); in
        # the dark the analytic phase alone
        dom = int(np.argmax(np.sum(np.abs(C)**2, axis=(0, 1))))
        Ad = field._A[:, :, dom if field.is_vector else 0]
        X, Y = np.meshgrid(field.x, field._y)
        dAy, dAx = np.gradient(Ad, field._y, field.x)
        Ad2 = np.abs(Ad)**2
        wet = Ad2 > 1E-8*Ad2.max()
        with np.errstate(invalid="ignore", divide="ignore"):
            gx = np.where(wet, np.imag(np.conj(Ad)*dAx)/Ad2, 0.0)
            gy = np.where(wet, np.imag(np.conj(Ad)*dAy)/Ad2, 0.0)
        gx = gx + 2*field.alpha[0]*X + field.beta[0]
        gy = gy + 2*field.alpha[1]*Y + field.beta[1]
        # complete (unwrapped) phase: the optical path of the rays
        q = field._qpf(dom) if field.is_vector else field._qpf(0)
        Phi = np.where(lit, _unwrapped_phase(q), field._phase())
        sel = np.ix_(iy, ix)
        Xs, Ys, kx, ky = X[sel], Y[sel], gx[sel], gy[sel]
        k = np.real(field.k)
        kz2 = k**2 - kx**2 - ky**2
        bad = kz2 <= 0
        if np.any(bad):
            share = np.sum(I[sel][bad])/np.sum(I[sel])
            if share > 1E-3:
                raise PropagationFailure("Trace: evanescent local wave vectors in the input "
                                         "({:.1e} of the power)".format(share))
            notes.append("{:.1e} of the power in evanescent local waves: dropped".format(share))
            kx, ky = np.where(bad, 0.0, kx), np.where(bad, 0.0, ky)
            kz2 = k**2 - kx**2 - ky**2
        kz = np.sqrt(np.maximum(kz2, 1E-12*k**2))
        D = np.stack([kx, ky, kz], axis=-1)/k
        # input field vectors relative to exp(i*Phi), E_z from k.E = 0
        rel = np.exp(1j*(field._phase()[sel] - Phi[sel]))
        Ex, Ey = C[:, :, 0][sel]*rel, C[:, :, 1][sel]*rel
        Ez = -(kx*Ex + ky*Ey)/kz
        E_in = np.stack([Ex, Ey, Ez], axis=-1)
        # rays from the field plane
        o0 = sysm.frames()[0][0][0]
        z0 = o0[2] - self.distance
        P0 = np.stack([Xs, Ys, np.full_like(Xs, z0)], axis=-1)
        shape = Xs.shape
        rays = Rays(P0.reshape(-1, 3), D.reshape(-1, 3), (Phi[sel]/k0).reshape(-1))
        pb = sysm.trace_polarization(rays, clip=self.clip, coatings=self.coatings, media=self.media)
        b = pb.bundle
        valid = b.valid
        lost = 1 - np.sum(I[sel].reshape(-1)*b.valid)/np.sum(I[sel])
        if lost > 1E-6:
            notes.append("{:.1e} of the power vignetted or lost".format(lost))
        _, R_img = sysm.frames()[1]
        P_out, D_out = sysm.to_image_frame(b.points[-1], b.directions[-1])
        n_out = float(np.real(sysm.n[-1]))
        E_out = np.einsum("nij,nj->ni", pb.P[-1], E_in.reshape(-1, 3)) @ R_img
        cos_in = np.abs(D.reshape(-1, 3)[:, 2])
        cos_out = np.abs(D_out[:, 2])
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.real(n_in)*cos_in/(n_out*cos_out))
        amp = E_out[:, :2]*scale[:, None]
        amp[~valid] = np.nan
        S = b.opl[-1].copy()
        S[~valid] = np.nan
        Pxy = P_out[:, :2].copy()
        Pxy[~valid] = np.nan
        # analytic quadratic part of the output phase k0*S
        ok = valid & np.all(np.isfinite(amp), axis=1)
        w = np.sum(np.abs(amp[ok])**2, axis=1) + 1E-300
        a2, b2, c2, d2, e2 = _fit_separable_quadratic(k0*S[ok], Pxy[ok, 0], Pxy[ok, 1], w)
        S_rest = S - (a2*Pxy[:, 0]**2 + b2*Pxy[:, 1]**2 + c2*Pxy[:, 0] + d2*Pxy[:, 1] + e2)/k0
        D_rest = D_out.copy()
        D_rest[:, 0] -= (2*a2*Pxy[:, 0] + c2)/(k0*n_out)
        D_rest[:, 1] -= (2*b2*Pxy[:, 1] + d2)/(k0*n_out)
        # output grid
        if self.x_out is not None:
            xo, yo = _linear(self.x_out, "x_out"), _linear(self.y_out, "y_out")
        else:
            # footprint of the rays that carry power
            wr = np.sum(np.abs(amp)**2, axis=1)
            bright = ok & (np.nan_to_num(wr) > 1E-4*np.nanmax(wr))
            pts = Pxy[bright]
            x0, x1 = pts[:, 0].min(), pts[:, 0].max()
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            mx, my = 0.1*(x1 - x0) + 1E-12, 0.1*(y1 - y0) + 1E-12
            x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
            # samples: the input's number, or more if the residual phase
            # (beyond the analytic quadratic) needs them (two samples per
            # period of its largest local frequency)
            kr = k0*n_out*np.abs(D_rest[bright, :2])
            kxm, kym = kr.max(axis=0) if len(kr) else (0.0, 0.0)
            Nx = max(len(field.x), int(np.ceil((x1 - x0)*kxm/np.pi*1.25)) + 1)
            Ny = max(len(field._y), int(np.ceil((y1 - y0)*kym/np.pi*1.25)) + 1)
            if max(Nx, Ny) > self.max_N:
                notes.append("output grid limited to {} samples (residual phase needs {}x{})"
                             .format(self.max_N, Nx, Ny))
                Nx, Ny = min(Nx, self.max_N), min(Ny, self.max_N)
            xo, yo = _window(x0, x1, Nx), _window(y0, y1, Ny)
        launch = np.stack([Xs, Ys], axis=-1)
        u = rays_to_field(xo, yo, Pxy.reshape(shape + (2,)), D_rest.reshape(shape + (3,)),
                          S_rest.reshape(shape), field.wavelength, launch=launch, irradiance=1.0,
                          n=n_out, amplitude=amp.reshape(shape + (2,)))
        out = Field(u, xo, yo, field.wavelength, n_out, alpha=(a2, b2), beta=(c2, d2), gamma=e2,
                    z=field.z + self.distance + float(sysm.frames()[1][0][2] - o0[2]))
        return out, {"method": "geometric (ray trace)", "notes": notes}


class DetectorRecord(object):
    """What a Detector saw: the field and the quantities it evaluated."""

    def __init__(self, name, field, values):
        self.name = name
        self.field = field
        self.values = values

    def __getattr__(self, item):
        values = self.__dict__.get("values", {})
        if item in values:
            return values[item]
        raise AttributeError(item)

    def __getitem__(self, item):
        return self.values[item]

    def __repr__(self):
        return "DetectorRecord({!r}, {})".format(self.name, sorted(self.values))


class Detector(Step):
    """Record the field and derived quantities.

    Parameters
    ----------
    name : str
    quantities : sequence of str, optional
        Any of "irradiance" (|E|**2, incl. |E_z|**2 if the chain is vectorial,
        or, in "auto" mode, if E_z carries more than tol of the power),
        "E" (E_x, E_y), "Ez", "H", "poynting", "power".
    x, y : arrays, optional
        Detector grid (default: the field's; for a FieldSum the union of the
        part windows, see FieldSum.default_grid()). The parts of a FieldSum
        are added coherently on it.
    """

    def __init__(self, name, quantities=("irradiance",), x=None, y=None):
        self.name = name
        self.quantities = tuple(quantities)
        self.x, self.y = x, y
        self.label = "detector {!r}".format(name)

    def apply(self, field, ctx):
        passed = field
        if isinstance(field, FieldSum):
            field = field.combined(self.x, self.y)
        elif self.x is not None:
            field = field.resample(self.x, self.y)
        notes = []
        mode = ctx.get("polarization", "auto")
        tol = ctx.get("tol", 1E-2)
        if mode == "vector":
            include_z = True
        elif mode == "scalar":
            include_z = False
        else:
            share = field.ez_share()
            include_z = share > tol
            if include_z:
                notes.append("E_z carries {:.1e} of |E|^2: included".format(share))
        values = {}
        for q in self.quantities:
            if q == "irradiance":
                values[q] = field.irradiance(include_z=include_z)
            elif q == "E":
                values[q] = field.E()
            elif q == "Ez":
                values[q] = field.Ez()
            elif q == "H":
                values[q] = field.H()
            elif q == "poynting":
                values[q] = field.poynting()
            elif q == "power":
                values[q] = field.power()
            else:
                raise ValueError("unknown quantity {!r}".format(q))
        ctx.setdefault("detectors", {})[self.name] = DetectorRecord(self.name, field.copy(), values)
        return passed, {"method": "detector", "notes": notes}


# --------------------------------------------------------------------------
# fields in several parts: parallel arms, Babinet obstructions
# --------------------------------------------------------------------------

class FieldSum(object):
    """Coherent sum of fields, each on its own grid with its own analytic
    phase (and, in a chain, its own propagation operator).

    A chain treats every part as a field of its own; the parts are added
    only where the sum is needed: in a Detector, in Combine or by
    combined(). This keeps a large smooth field and a small strongly
    diffracting one (an obstruction, see Obstruction) cheap at the same
    time, and holds the arms of an interferometer (see Parallel).
    """

    def __init__(self, parts):
        flat = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, FieldSum) else [p])
        if not flat:
            raise ValueError("a FieldSum needs at least one part")
        if len({(p.wavelength, p.one_d) for p in flat}) > 1:
            raise ValueError("parts must share wavelength and dimensionality")
        self.parts = flat

    def __len__(self):
        return len(self.parts)

    def copy(self):
        return FieldSum([p.copy() for p in self.parts])

    @property
    def one_d(self):
        return self.parts[0].one_d

    @property
    def is_vector(self):
        return any(p.is_vector for p in self.parts)

    @property
    def z(self):
        return self.parts[0].z

    @property
    def wavelength(self):
        return self.parts[0].wavelength

    @property
    def shape(self):
        return self.parts[0].shape

    def default_grid(self, max_N=4096):
        """Union of the part windows, sampled with the finest part spacing
        (at most max_N per axis)."""
        def axis(coords):
            lo = min(c[0] for c in coords)
            hi = max(c[-1] for c in coords)
            d = min(_spacing(c) for c in coords)
            N = min(int(np.ceil((hi - lo)/d)) + 1, int(max_N))
            return _window(lo, hi, N)
        x = axis([p.x for p in self.parts])
        y = None if self.one_d else axis([p._y for p in self.parts])
        return x, y

    def combined(self, x=None, y=None, max_N=4096):
        """The sum as one Field on the grid x, y (default: default_grid()),
        with the analytic phase of the strongest part. The rest of the phase
        is sampled on that grid; choose it fine enough."""
        if x is None:
            x, y = self.default_grid(max_N)
        x = _linear(x, "x")
        y = None if self.one_d else _linear(y, "y")
        powers = [p.power() for p in self.parts]
        base = self.parts[int(np.argmax(powers))].resample(x, y)
        e_base = np.exp(-1j*base._phase())
        A = np.zeros(base._A.shape[:2] + (2,), dtype=complex)
        for p in self.parts:
            g = p.resample(x, y)
            A += g._components()*(np.exp(1j*g._phase())*e_base)[:, :, None]
        out = base.vectorized()._with(A)
        if not self.is_vector:
            out, _ = out.scalarized()
        return out

    def E(self, x=None, y=None):
        return self.combined(x, y).E()

    def irradiance(self, x=None, y=None, include_z=False):
        return self.combined(x, y).irradiance(include_z=include_z)

    def power(self, x=None, y=None):
        return self.combined(x, y).power()

    def __repr__(self):
        return "FieldSum({} parts: {})".format(
            len(self.parts), ", ".join("x".join(str(v) for v in p.shape[::-1]) for p in self.parts))


class Parallel(Step):
    """Parallel arms between a beam splitter and a combiner.

    Each arm is a Chain (or a list of steps) that gets the incoming field
    times its weight; the result is the FieldSum of the arm outputs, which
    later steps (the common path behind the combiner) treat part by part and
    a Detector adds coherently. The weights are amplitude factors, e.g.
    (0.5, 0.5) for the bright port of a Mach-Zehnder interferometer with two
    50/50 splitters (1/sqrt(2) at each). The arms are run with the chain's
    polarization mode and tolerance; their steps show up in the chain's
    report, prefixed with the arm number.

    Parameters
    ----------
    arms : Chains or lists of steps
    weights : sequence of complex, optional
        Amplitude factors (default: 1).
    """

    def __init__(self, *arms, weights=None, label="parallel"):
        if len(arms) < 1:
            raise ValueError("give at least one arm")
        self.arms = [a if isinstance(a, Chain) else Chain(a) for a in arms]
        self.weights = (np.ones(len(arms), dtype=complex) if weights is None
                        else np.asarray(weights, dtype=complex))
        if len(self.weights) != len(self.arms):
            raise ValueError("one weight per arm")
        self.label = label

    def apply(self, field, ctx):
        parts = field.parts if isinstance(field, FieldSum) else [field]
        outs = []
        index = ctx["_index"]
        for i, (arm, w) in enumerate(zip(self.arms, self.weights)):
            for p in parts:
                f = p._with(p._A*w)
                outs.append(Chain._run_steps(arm, f, ctx, "{}.{}.".format(index, i + 1)))
        ctx["_index"] = index
        return FieldSum(outs), {"method": "{} arms".format(len(self.arms))}


class Obstruction(Step):
    """Small obstruction or defect, by Babinet's principle.

    A small opaque particle (or a defect of transmission t) in a large field
    E turns it into E + (t - 1) E M, M the indicator of the obstruction. The
    second part lives on a small, finely sampled grid around the obstruction
    and diffracts strongly, the first stays smooth: as two parts of a
    FieldSum each is propagated with its own grid and operator. This is the
    "virtual stop" of the field tracing literature: a second source of the
    opposite phase.

    Parameters
    ----------
    radius, half_widths : number, optional
        Circular or rectangular obstruction.
    center : (x, y), optional
    transmission : complex, optional
        Amplitude transmission of the obstruction (0: opaque; exp(i phi) for
        a pure phase defect).
    samples : int, optional
        Samples across the obstruction (its grid adds a 25% margin). They
        also limit its angular spectrum to about wavelength/(2 spacing):
        diffraction beyond that leaves any detector of the size of the main
        beam, and a coarse grid keeps it from dominating the error budget.
        The indicator is area weighted (anti-aliased).
    """

    def __init__(self, radius=None, half_widths=None, center=(0.0, 0.0), transmission=0.0,
                 samples=24, label="obstruction"):
        if (radius is None) == (half_widths is None):
            raise ValueError("give exactly one of radius, half_widths")
        self.radius, self.half_widths = radius, half_widths
        self.center = tuple(float(c) for c in center)
        self.transmission = complex(transmission)
        self.samples = int(samples)
        self.label = label

    def apply(self, field, ctx):
        if isinstance(field, FieldSum):
            raise ValueError("Obstruction acts on a single field; place it inside an arm")
        hx, hy = ((self.radius, self.radius) if self.radius is not None
                  else _pair(self.half_widths))
        cx, cy = self.center
        N = int(np.ceil(self.samples*1.25)) | 1
        x = _window(cx - 1.25*hx, cx + 1.25*hx, N)
        y = None if field.one_d else _window(cy - 1.25*hy, cy + 1.25*hy, N)
        small = field.resample(x, y)
        # area-weighted (anti-aliased) indicator: 5x5 subsamples per sample
        sub = (np.arange(5) - 2)/5
        dx = _spacing(small.x)
        dy = 0.0 if field.one_d else _spacing(small._y)
        X, Y = np.meshgrid(small.x - cx, small._y - cy)
        if field.one_d:
            Y = np.zeros_like(X)
        M = np.zeros(X.shape)
        for sx in sub:
            for sy in (sub if not field.one_d else [0.0]):
                Xs, Ys = X + sx*dx, Y + sy*dy
                if self.radius is not None:
                    M += (Xs**2 + Ys**2 <= self.radius**2)
                else:
                    M += (np.abs(Xs) <= hx) & (np.abs(Ys) <= hy)
        M /= len(sub)*(1 if field.one_d else len(sub))
        small = small._with(small._A*((self.transmission - 1)*M)[:, :, None])
        if field.one_d:
            notes = []
        else:
            notes = ["Babinet part: {} samples on {:.3g} x {:.3g}".format(N, 2.5*hx, 2.5*hy)]
        return FieldSum([field, small]), {"method": "Babinet", "notes": notes}


class Combine(Step):
    """Add the parts of a FieldSum into one Field on the grid x, y (default:
    FieldSum.default_grid())."""

    def __init__(self, x=None, y=None, label="combine"):
        self.x, self.y = x, y
        self.label = label

    def apply(self, field, ctx):
        if not isinstance(field, FieldSum):
            return field, {"method": "single field"}
        return field.combined(self.x, self.y), {"method": "{} parts".format(len(field))}


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------

class ChainResult(object):
    """Result of Chain.run(): the output field, the detector records, the
    fields at all boundaries (if kept) and a log of every step."""

    def __init__(self, field, detectors, fields, log):
        self.field = field
        self.detectors = detectors
        self.fields = fields
        self.log = log

    def __getitem__(self, name):
        return self.detectors[name]

    def report(self):
        """One line per step: operator, method, estimated error, samples,
        polarization, time and notes."""
        lines = ["{:>3s}  {:28s} {:15s} {:>9s} {:>11s} {:7s} {:>8s}  {}".format(
            "#", "step", "method", "error", "samples", "pol.", "ms", "notes")]
        for row in self.log:
            err = "" if row.get("error") is None else "{:.1e}".format(row["error"])
            lines.append("{:>3s}  {:28s} {:15s} {:>9s} {:>11s} {:7s} {:>8.1f}  {}".format(
                str(row["index"]), row["step"][:28], row.get("method", "")[:15], err, row["samples"],
                row["polarization"], 1E3*row["time"], "; ".join(row.get("notes", []))))
            if row.get("candidates"):
                cand = ", ".join("{} {:.1e}/{:.0f}".format(m, e, n)
                                 for m, (e, n) in row["candidates"].items())
                lines.append("     {:28s} candidates (error/effort): {}".format("", cand))
        return "\n".join(lines)

    def __repr__(self):
        return self.report()


class Chain(object):
    """Sequence of steps, traced in order (convective field tracing).

    A Chain behaves like a list: it can be indexed, sliced, appended to and
    concatenated with +.
    """

    def __init__(self, steps=()):
        self.steps = list(steps)
        for s in self.steps:
            if not isinstance(s, (Step, Chain)):
                raise TypeError("chain elements must be Steps or Chains, got {!r}".format(s))

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Chain(self.steps[i])
        return self.steps[i]

    def __add__(self, other):
        other = other.steps if isinstance(other, Chain) else list(other)
        return Chain(self.steps + other)

    def append(self, step):
        self.steps.append(step)

    @classmethod
    def from_system(cls, system, distance=0.0, gap=None, method="auto", tol=None, **trace_kw):
        """Chain for a raytrace.System: geometrical optics through the
        surfaces, wave optics in the free space behind them.

        The field is traced (Trace) from a plane `distance` in front of the
        first vertex to `gap` behind the last vertex, and then propagated
        (Propagate, operator chosen automatically by default) the rest of the
        way to the image plane of the system (the thickness of its last
        surface). This is the split of the field tracing paper: rays where
        the field is smooth and the surfaces are large, diffraction where it
        matters (a focus, a caustic). Without gap the trace goes all the way
        to the image plane.

        Parameters
        ----------
        system : raytrace.System
        distance : number, optional
            Distance of the input plane in front of the first vertex.
        gap : number, optional
            Distance behind the last vertex at which the rays hand over to
            the wave optical propagation.
        method, tol : optional
            For the Propagate step.
        trace_kw : optional
            Passed to Trace (n_rays, clip, coatings, media, max_N, ...).
        """
        import copy
        last = system.surfaces[-1].thickness
        if gap is None or np.isclose(gap, last):
            return cls([Trace(system, distance=distance, **trace_kw)])
        if abs(gap) > abs(last) or gap*last < 0:
            raise ValueError("gap must lie between the last vertex and the image plane")
        sysm = copy.copy(system)
        sysm.surfaces = [copy.copy(s) for s in system.surfaces]
        sysm.surfaces[-1].thickness = gap
        return cls([Trace(sysm, distance=distance, **trace_kw),
                    Propagate(last - gap, method=method, tol=tol)])

    def _flat(self):
        for s in self.steps:
            if isinstance(s, Chain):
                yield from s._flat()
            else:
                yield s

    def __repr__(self):
        return "Chain([{}])".format(", ".join(repr(s) for s in self._flat()))

    @staticmethod
    def _run_steps(chain, field, ctx, prefix):
        """Apply the steps of chain to field (a Field or FieldSum, part by
        part), with the polarization decisions and the log in ctx."""
        polarization, tol = ctx["polarization"], ctx["tol"]
        log = ctx["_log"]
        for i, step in enumerate(chain._flat()):
            t0 = time.perf_counter()
            index = "{}{}".format(prefix, i)
            ctx["_index"] = index
            if isinstance(field, FieldSum) and not isinstance(step, (Parallel, Combine, Detector)):
                parts = field.parts
            else:
                parts = [field]
            outs = []
            if len(parts) > 1:
                # error budget relative to the whole sum: a weak part may
                # be less accurate by the ratio of the amplitudes
                powers = np.array([p.power() for p in parts])
                budgets = tol*np.sqrt(np.sum(powers)/np.maximum(powers, 1E-300))
                budgets = np.minimum(budgets, 3*tol)
            for j, part in enumerate(parts):
                ctx["tol_part"] = budgets[j] if len(parts) > 1 else tol
                was_vector = part.is_vector
                out, info = step.apply(part, ctx)
                notes = list(info.get("notes", []))
                if isinstance(out, Field) and out.is_vector and polarization != "vector":
                    s, lost = out.scalarized()
                    if polarization == "scalar":
                        if lost > 1E-12:
                            notes.append("scalar: dropped {:.1e} cross-polarized power".format(lost))
                        out = s
                    elif lost <= tol:
                        if was_vector:
                            notes.append("globally polarized again (residual {:.1e}): scalar"
                                         .format(lost))
                        out = s
                    elif not was_vector:
                        notes.append("polarization varies (cross share {:.1e}): vector"
                                     .format(lost))
                outs.append(out)
                label = step.label if len(parts) == 1 else "{} [part {}]".format(step.label, j + 1)
                shape = out.shape if isinstance(out, Field) else "{} parts".format(len(out))
                log.append({"index": index, "step": label,
                            "method": info.get("method", ""),
                            "error": info.get("error"), "candidates": info.get("candidates"),
                            "samples": ("x".join(str(v) for v in shape[::-1])
                                        if isinstance(shape, tuple) else shape),
                            "polarization": "vector" if out.is_vector else "scalar",
                            "time": time.perf_counter() - t0, "notes": notes})
                t0 = time.perf_counter()
            field = outs[0] if len(outs) == 1 else FieldSum(outs)
            if ctx["_keep"] and not prefix:
                ctx["_fields"].append(field.copy())
        return field

    def run(self, source, polarization="auto", tol=1E-2, keep=False):
        """Trace the source field through the chain.

        Parameters
        ----------
        source : Field
        polarization : str, optional
            "auto" (default), "scalar" or "vector"; see the module docstring.
        tol : number, optional
            Error budget of the automatic operator selection and the
            threshold for polarization and E_z decisions.
        keep : bool, optional
            Keep the field at every boundary (ChainResult.fields).

        Returns
        -------
        ChainResult
        """
        if polarization not in ("auto", "scalar", "vector"):
            raise ValueError("polarization must be 'auto', 'scalar' or 'vector'")
        ctx = {"polarization": polarization, "tol": tol, "detectors": {}}
        field = source.copy()
        log = []
        if polarization == "vector":
            field = field.vectorized()
        elif polarization == "scalar" and field.is_vector:
            field, lost = field.scalarized()
            if lost > 0:
                warnings.warn("scalar chain: the source loses {:.1e} of its power "
                              "(cross polarization)".format(lost), RuntimeWarning)
        ctx["_log"], ctx["_keep"] = log, keep
        fields = [field.copy()] if keep else []
        ctx["_fields"] = fields
        field = Chain._run_steps(self, field, ctx, "")
        return ChainResult(field, ctx["detectors"], fields, log)
