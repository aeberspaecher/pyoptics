#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Media: permittivity, conductivity and dispersion.

In the normalized units of the FDTD (eps_0 = mu_0 = c = 1), frequencies are
angular frequencies in units of c divided by the length unit: a vacuum
wavelength wl has omega = 2*pi/wl. With the time dependence exp(-i omega t)
of all of pyoptics, a medium has the complex permittivity ::

    eps(omega) = eps_inf + i*sigma/omega + sum_k chi_k(omega)

with the susceptibilities of its poles,

- Lorentz: chi = delta_eps*omega_0**2/(omega_0**2 - omega**2 - i*gamma*omega)
  (bound charges, resonances of glasses and absorbers),
- Drude: chi = -omega_p**2/(omega**2 + i*gamma*omega) (free electrons:
  metals, doped semiconductors),
- Debye: chi = delta_eps/(1 - i*omega*tau) (orientational relaxation).

The FDTD advances one polarization density P per pole with its auxiliary
differential equation (ADE), P'' + gamma P' + omega_0**2 P =
delta_eps*omega_0**2 E (Drude: omega_0 = 0 and omega_p**2 on the right,
Debye: tau P' + P = delta_eps E), and the electric field sees dP/dt as a
current. The Lorentz and Drude updates are second order in time, the
Debye update (an exponential integrator) first order.

Photon energies convert with omega = E/(hbar c) per length unit:
`omega_from_eV()` (1.2398/E µm for E in eV).
"""

import numpy as np


HBAR_C_EV_UM = 0.19732697    # hbar*c in eV*µm


def omega_from_eV(energy, length_unit=1e-6):
    """Normalized angular frequency of a photon energy in eV, for lengths in
    units of length_unit metres (default µm)."""

    return np.asarray(energy)/(HBAR_C_EV_UM*1e-6/length_unit)


class Pole(object):
    """Base class of a dispersive term. Subclasses define chi(omega) and the
    ADE update coefficients (a, b, c) of P(n+1) = a P(n) + b P(n-1) + c E(n)
    (per unit filling)."""

    def chi(self, omega):
        raise NotImplementedError

    def coefficients(self, dt):
        raise NotImplementedError

    def chi_wl(self, wavelength):
        """Susceptibility at vacuum wavelengths."""

        return self.chi(2*np.pi/np.asarray(wavelength, dtype=float))


class Lorentz(Pole):
    """Lorentz oscillator: delta_eps*omega_0**2/(omega_0**2 - omega**2 -
    i gamma omega)."""

    def __init__(self, delta_eps, omega_0, gamma):
        self.delta_eps, self.omega_0, self.gamma = float(delta_eps), float(omega_0), float(gamma)

    def chi(self, omega):
        w0 = self.omega_0
        return self.delta_eps*w0**2/(w0**2 - omega**2 - 1j*self.gamma*omega)

    def _strength(self):
        return self.delta_eps*self.omega_0**2

    def coefficients(self, dt):
        g = self.gamma*dt/2
        return ((2 - self.omega_0**2*dt**2)/(1 + g), (g - 1)/(1 + g), self._strength()*dt**2/(1 + g))

    def __repr__(self):
        return "Lorentz(delta_eps=%g, omega_0=%g, gamma=%g)" % (self.delta_eps, self.omega_0, self.gamma)


class Drude(Lorentz):
    """Free electron (Drude) term: -omega_p**2/(omega**2 + i gamma omega)."""

    def __init__(self, omega_p, gamma):
        self.omega_p, self.gamma = float(omega_p), float(gamma)
        self.omega_0 = 0.0
        self.delta_eps = np.inf

    def chi(self, omega):
        return -self.omega_p**2/(omega**2 + 1j*self.gamma*omega)

    def _strength(self):
        return self.omega_p**2

    def __repr__(self):
        return "Drude(omega_p=%g, gamma=%g)" % (self.omega_p, self.gamma)


class Debye(Pole):
    """Debye relaxation: delta_eps/(1 - i omega tau)."""

    def __init__(self, delta_eps, tau):
        self.delta_eps, self.tau = float(delta_eps), float(tau)

    def chi(self, omega):
        return self.delta_eps/(1 - 1j*omega*self.tau)

    def coefficients(self, dt):
        # exponential integrator over one step with E(n) held constant:
        # stable for any tau, but first order in time
        e = np.exp(-dt/self.tau)
        return (e, 0.0, self.delta_eps*(1 - e))

    def __repr__(self):
        return "Debye(delta_eps=%g, tau=%g)" % (self.delta_eps, self.tau)


class Medium(object):
    """Medium with permittivity, conductivity and optional dispersion.

    Parameters
    ----------
    eps : number
        Relative permittivity (eps_inf for a dispersive medium; >= 1 for
        stability with the default time step).
    sigma : number, optional
        Conductivity in normalized units (absorption).
    poles : sequence of Pole, optional
        Lorentz, Drude and Debye terms.
    """

    def __init__(self, eps=1.0, sigma=0.0, poles=()):
        self.eps = float(eps)
        self.sigma = float(sigma)
        self.poles = tuple(poles)

    @classmethod
    def from_index(cls, n, k=0.0, wavelength=None):
        """Non-dispersive medium with the complex index n + ik at one vacuum
        wavelength (exact there; the conductivity model makes k vary with
        wavelength)."""

        if k != 0 and wavelength is None:
            raise ValueError("an absorbing medium needs the wavelength")
        sigma = 2*n*k*2*np.pi/wavelength if k else 0.0
        return cls(n**2 - k**2, sigma)

    @classmethod
    def fit(cls, wavelengths, index, n_lorentz=2, drude=False, eps_inf=None):
        """Fit a dispersive medium to complex refractive indices.

        Parameters
        ----------
        wavelengths : array
            Vacuum wavelengths (in the length unit of the simulation).
        index : array or callable
            Complex indices n + ik at the wavelengths, or a function of the
            wavelength (e.g. ``lambda wl: materials.complex_index(m, wl*1e3)``
            for wavelengths in µm).
        n_lorentz : int
            Number of Lorentz poles.
        drude : bool
            Add a Drude term (metals).
        eps_inf : number, optional
            Fix eps_inf (default: fitted, >= 1).

        Returns
        -------
        Medium
            Its `fit_error` attribute is the maximum absolute error of eps over the
            fitted wavelengths.
        """

        from scipy.optimize import least_squares
        wl = np.asarray(wavelengths, dtype=float)
        n = index(wl) if callable(index) else np.asarray(index)
        target = np.asarray(n, dtype=complex)**2
        w = 2*np.pi/wl
        wmax = w.max()
        scale = np.max(np.abs(target))

        def build(p):
            k = 0
            if eps_inf is None:
                e_inf = 1 + p[0]**2
                k = 1
            else:
                e_inf = eps_inf
            poles = []
            if drude:
                poles.append(Drude(abs(p[k]), abs(p[k + 1])))
                k += 2
            for _ in range(n_lorentz):
                poles.append(Lorentz(abs(p[k]), abs(p[k + 1]), abs(p[k + 2])))
                k += 3
            return cls(e_inf, poles=poles)

        def residual(p):
            r = (build(p).complex_eps_omega(w) - target)/scale
            return np.concatenate([r.real, r.imag])

        best = None
        rng = np.random.default_rng(0)
        for trial in range(12):
            p0 = [] if eps_inf is None else []
            if eps_inf is None:
                p0.append(np.sqrt(max(np.real(target).min() - 1, 0.1)) if not drude else 1.0)
            if drude:
                p0 += [np.sqrt(max(-np.real(target).min(), 1.0))*wmax, 0.05*wmax]
            for j in range(n_lorentz):
                w0 = wmax*(1.5 + 2*j) if trial == 0 else rng.uniform(0.3, 4)*wmax
                p0 += [rng.uniform(0.1, 2) if trial else 1.0, w0, 0.1*w0 if trial == 0 else rng.uniform(0.01, 1)*w0]
            res = least_squares(residual, np.asarray(p0, dtype=float), max_nfev=4000)
            if best is None or res.cost < best.cost:
                best = res
        medium = build(best.x)
        medium.fit_error = np.max(np.abs(medium.complex_eps_omega(w) - target))
        return medium

    @property
    def dispersive(self):
        return bool(self.poles)

    def complex_eps_omega(self, omega):
        """Complex relative permittivity at angular frequencies."""

        omega = np.asarray(omega, dtype=float)
        eps = self.eps + 1j*self.sigma/omega
        for p in self.poles:
            eps = eps + p.chi(omega)
        return eps

    def complex_eps(self, wavelength):
        """Complex relative permittivity at vacuum wavelengths."""

        return self.complex_eps_omega(2*np.pi/np.asarray(wavelength, dtype=float))

    def index(self, wavelength=None):
        """Complex refractive index n + ik (at a wavelength if conductive or
        dispersive; the root with k >= 0)."""

        if self.sigma == 0 and not self.poles:
            return np.sqrt(self.eps) + 0j
        if wavelength is None:
            raise ValueError("a conductive or dispersive medium needs the wavelength")
        n = np.sqrt(self.complex_eps(wavelength) + 0j)
        return np.where(np.imag(n) < 0, -n, n)

    def __repr__(self):
        s = "Medium(eps=%g, sigma=%g" % (self.eps, self.sigma)
        if self.poles:
            s += ", poles=%r" % (list(self.poles),)
        return s + ")"


VACUUM = Medium(1.0)
