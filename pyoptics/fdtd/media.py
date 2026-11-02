#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Media: relative permittivity and conductivity.

In the normalized units of the FDTD (eps_0 = mu_0 = c = 1), a medium with
relative permittivity eps and conductivity sigma has the complex
permittivity eps + i*sigma/omega (time dependence exp(-i omega t), as in all
of pyoptics), with omega = 2*pi/wl for the vacuum wavelength wl.
Dispersive (Drude, Lorentz, Debye) media are planned.
"""

import numpy as np


class Medium(object):
    """Non-dispersive, possibly conductive medium.

    Parameters
    ----------
    eps : number
        Relative permittivity (>= 1 for stability with the default time step).
    sigma : number, optional
        Conductivity in normalized units (absorption).
    """

    def __init__(self, eps=1.0, sigma=0.0):
        self.eps = float(eps)
        self.sigma = float(sigma)

    @classmethod
    def from_index(cls, n, k=0.0, wavelength=None):
        """Medium with the complex index n + ik at one vacuum wavelength
        (exact there; the conductivity model makes k vary with wavelength)."""

        if k != 0 and wavelength is None:
            raise ValueError("an absorbing medium needs the wavelength")
        sigma = 2*n*k*2*np.pi/wavelength if k else 0.0
        return cls(n**2 - k**2, sigma)

    def complex_eps(self, wavelength):
        """Complex relative permittivity at a vacuum wavelength."""

        return self.eps + 1j*self.sigma*np.asarray(wavelength)/(2*np.pi)

    def index(self, wavelength=None):
        """Complex refractive index n + ik (at a wavelength if conductive)."""

        if self.sigma == 0:
            return np.sqrt(self.eps) + 0j
        return np.sqrt(self.complex_eps(wavelength))

    def __repr__(self):
        return "Medium(eps=%g, sigma=%g)" % (self.eps, self.sigma)


VACUUM = Medium(1.0)
