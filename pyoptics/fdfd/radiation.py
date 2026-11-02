#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Far field of a 2D solution from outgoing cylindrical waves.

Outside a circle of radius rho that contains all sources and scatterers,
the field along z in a homogeneous medium of index n is a sum of outgoing
cylindrical waves,

    u(r, phi) = sum_m a_m H_m(k r) exp(i m phi),   k = n omega,

with the Hankel functions H_m of the first kind. The coefficients follow
from the field on the circle by an FFT over phi, and for k r -> infinity
H_m(k r) -> sqrt(2/(pi k r)) exp(i (k r - m pi/2 - pi/4)), so the far-field
amplitude in the direction phi is

    F(phi) = sum_m a_m (-i)**m exp(i m phi)

(times the common factor sqrt(2/(pi k r)) exp(i (k r - pi/4))). For a
resonance, omega is complex and so is k; the field then grows with r
(the light left the cavity earlier), but the angular distribution is
that of F.
"""

import numpy as np


class FarField(object):
    """Far-field pattern: angles phi (radians, from +x), complex amplitude
    F(phi), intensity abs(F)**2 normalized to a mean of 1 over the circle,
    and the cylindrical harmonics (m, a_m) of the outgoing field."""

    def __init__(self, phi, amplitude, m, a_m):
        self.phi = phi
        self.amplitude = amplitude
        I = np.abs(amplitude)**2
        self.intensity = I/I.mean()
        self.m, self.a_m = m, a_m

    def directionality(self, direction=0.0):
        """U = <cos(phi - direction)> weighted by the intensity: 1 for all
        light into the direction, 0 for symmetric emission, -1 for all of
        it into the opposite direction (Wiersig and Hentschel 2008)."""

        I = self.intensity
        return float(np.sum(I*np.cos(self.phi - direction))/np.sum(I))

    def peak(self):
        """Direction (radians) of the largest intensity."""

        return float(self.phi[np.argmax(self.intensity)])

    def harmonic_power(self):
        """abs(a_m)**2 normalized to a sum of 1: the angular momentum spectrum
        of the outgoing wave."""

        p = np.abs(self.a_m)**2
        return p/p.sum()


def _sample(sol, X, Y):
    """Bilinear interpolation of the field along z at points."""

    from scipy.interpolate import RegularGridInterpolator
    x, y = sol.coords[sol.sim.components[0]]
    u = sol.main
    re = RegularGridInterpolator((x, y), u.real)
    im = RegularGridInterpolator((x, y), u.imag)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return (re(pts) + 1j*im(pts)).reshape(X.shape)


def circle_harmonics(solution, radius, center=(0.0, 0.0), num_orders=60):
    """Fourier coefficients c_m of the field along z on a circle,
    u(radius, phi) = sum_m c_m exp(i m phi): e.g. the azimuthal number of a
    whispering gallery mode (on a circle just inside the boundary).

    Returns
    -------
    m : array
        -num_orders..num_orders.
    c_m : complex array
    """

    N = 4*num_orders + 4
    t = 2*np.pi*np.arange(N)/N
    u = _sample(solution, center[0] + radius*np.cos(t), center[1] + radius*np.sin(t))
    c = np.fft.fft(u)/N
    m = np.fft.fftfreq(N, 1.0/N).astype(int)
    keep = np.abs(m) <= num_orders
    order = np.argsort(m[keep])
    return m[keep][order], c[keep][order]


def farfield(solution, radius, center=(0.0, 0.0), n_medium=None, num_orders=None, n_angles=720):
    """Far-field pattern of a 2D solution (driven or a resonance).

    Parameters
    ----------
    solution : Solution
        From `Simulation.solve()` or `Resonance.field`. With a PlaneWave
        source the field outside its box is the scattered field, so the
        circle has to lie outside the box.
    radius : number
        Radius of the circle on which the field is expanded; it must
        enclose all structures and sources and lie in the homogeneous
        background, inside the PML.
    center : (x, y), optional
    n_medium : complex, optional
        Index of the medium on the circle; default: the background of the
        simulation at the wavelength.
    num_orders : int, optional
        Highest abs(m); default about k radius + 20.
    n_angles : int, optional
        Directions of the result.

    Returns
    -------
    FarField
    """

    from scipy.special import hankel1

    sim = solution.sim
    omega = complex(solution.omega)
    if n_medium is None:
        n_medium = np.sqrt(complex(np.asarray(sim.background.complex_eps(2*np.pi/omega.real))))
    k = n_medium*omega
    if num_orders is None:
        num_orders = int(abs(k)*radius + 20)
    m, c = circle_harmonics(solution, radius, center, num_orders)
    a_m = c/hankel1(m, k*radius)
    phi = 2*np.pi*np.arange(n_angles)/n_angles - np.pi
    F = np.exp(1j*np.outer(phi, m)) @ (a_m*(-1j)**m)
    return FarField(phi, F, m, a_m)
