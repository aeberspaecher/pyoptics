#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Flare: long-range scattered light in the projection lens.

Surface roughness of the lens elements (or mirrors) scatters a small part of
the light, the total integrated scatter (TIS), into a halo around every
image point. In the Born (Rayleigh-Rice) approximation a surface with the
height PSD S(f) (2D, isotropic, normalized so that 2*pi*int S(f) f df =
sigma**2) scatters a plane wave by the angle lambda*f into a direction that
lands at the distance r = lambda*f*L from the image point, L the lever arm
of the surface (its image-side distance to the wafer, divided by the
magnification for surfaces before the stop). The flare point spread
function is then ::

    PSF(r) = TIS/sigma**2*S(r/(lambda*L))/(lambda*L)**2,

with 2*pi*int PSF(r) r dr = TIS and, per surface, ::

    TIS = (4*pi*sigma*cos(theta)/lambda)**2           (mirror),
    TIS = (2*pi*(n - 1)*sigma/lambda)**2              (refracting surface).

Both grow as 1/lambda**2: a surface that scatters 0.01 % at 193 nm scatters
2 % at 13.5 nm. A fractal (power-law) PSD S ~ f**-beta gives a power-law
PSF ~ r**-beta. Flare.power_law() sets the PSF directly from a total and
an exponent, the way flare is usually specified; Flare.from_roughness()
derives it from an ABC (K-correlation) PSD of the surfaces.

The image with flare is ::

    I_flare = (1 - TIS)*I + PSF * I_mask,

where I_mask is the light that leaves the mask over a large area. The PSF
reaches micrometres to millimetres, much further than one clip, so its
convolution with the image is a constant over a clip: TIS times the pattern
density (the open fraction of the mask) averaged with the PSF. A clear
field stays at 1. Flare.level() gives the level for a uniform density,
Flare.level_map() for a density map of a die (FFT convolution), and
Flare.apply() adds it to an image. The flare level adds a constant
background to every image: it lowers the contrast and the image log slope
and shifts the dose to size, by an amount that depends on the
surrounding pattern density (the effect a flare test with dark pads in a
clear field shows).

Lengths in nm (PSF), the density map in any length unit with the PSF in
the same unit.

References
----------
J. M. Bennett and L. Mattsson, "Introduction to Surface Roughness and
Scattering", OSA (1989).

E. L. Church, H. A. Jenkinson and J. M. Zavada, "Relationship between surface
scattering and microtopographic features", Opt. Eng. 18, 125-136 (1979).

J. P. Kirk, "Scattered light in photolithographic lenses", Proc. SPIE 2197,
566-572 (1994).

F. M. Schellenberg, J. Word and O. Toublan, "Layout compensation for EUV
flare", Proc. SPIE 5751, 320-329 (2005).

C. Krautschik, M. Ito, I. Nishiyama and K. Otaki, "Impact of EUV light scatter
on CD control as a result of mask density changes", Proc. SPIE 4688,
289-301 (2002).
"""

import numpy as np
from scipy.integrate import quad


def tis(sigma, wavelength, n=None, cos_theta=1.0):
    """Total integrated scatter of a smooth surface (Born approximation).

    Parameters
    ----------
    sigma : number
        RMS roughness (same unit as the wavelength).
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index of a refracting surface between glass and air (or
        n_1/n_2 contrast as n - 1); None for a mirror.
    cos_theta : number, optional
        Cosine of the angle of incidence (mirror).

    Returns
    -------
    TIS : number
    """

    if n is None:
        return (4*np.pi*sigma*cos_theta/wavelength)**2

    return (2*np.pi*(n - 1)*sigma/wavelength)**2


class ABCRoughness(object):
    """Isotropic 2D roughness PSD of the ABC (K-correlation) model ::

        S(f) = A/(1 + (B*f)**2)**((C + 1)/2),

    normalized to the RMS roughness sigma over the band f_min ... f_max
    (2*pi*int S(f) f df = sigma**2): B is the correlation length, C the
    slope (S ~ f**-(C + 1) for f >> 1/B).

    Parameters
    ----------
    sigma : number
        RMS roughness in the band.
    B : number
        Correlation length (1/f units).
    C : number
        Power-law slope parameter (C > 0).
    f_min, f_max : numbers, optional
        Band of spatial frequencies that scatter into the flare range.
    """

    def __init__(self, sigma, B, C=1.5, f_min=0.0, f_max=np.inf):
        self.sigma, self.B, self.C = sigma, B, C
        self.f_min, self.f_max = f_min, f_max
        g = lambda f: f/(1 + (B*f)**2)**((C + 1)/2)
        self.A = sigma**2/(2*np.pi*quad(g, f_min, f_max, limit=200)[0])

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        inside = (f >= self.f_min) & (f <= self.f_max)
        return np.where(inside, self.A/(1 + (self.B*f)**2)**((self.C + 1)/2), 0.0)


class Flare(object):
    """Flare point spread function and its use on images.

    Parameters
    ----------
    psf : callable
        Radial PSF psf(r) (1/length**2), normalized so that 2*pi*int psf(r)
        r dr = total.
    total : number
        Total integrated scatter (fraction of the light in the flare).
    r_min, r_max : numbers
        Range of the PSF (zero outside).
    """

    def __init__(self, psf, total, r_min, r_max):
        self.psf, self.total, self.r_min, self.r_max = psf, total, r_min, r_max

    @classmethod
    def power_law(cls, total, exponent=1.5, r_min=1e3, r_max=1e6):
        """PSF ~ r**-exponent between r_min and r_max, holding total.

        Parameters
        ----------
        total : number
            Flare fraction (e.g. 0.01 to 0.03 for DUV lenses, 0.05 to 0.1
            for early EUV tools).
        exponent : number, optional
        r_min, r_max : numbers, optional
            Range (default 1 um to 1 mm in nm).
        """

        a = 2 - exponent
        norm = (np.log(r_max/r_min) if abs(a) < 1e-12 else (r_max**a - r_min**a)/a)
        K = total/(2*np.pi*norm)

        def psf(r):
            r = np.asarray(r, dtype=float)
            inside = (r >= r_min) & (r <= r_max)
            return np.where(inside, K*np.where(inside, r, 1.0)**-exponent, 0.0)

        return cls(psf, total, r_min, r_max)

    @classmethod
    def from_roughness(cls, psd, wavelength, lever_arm, n=None, n_surfaces=1, cos_theta=1.0):
        """Flare of n_surfaces equal surfaces with the roughness PSD psd.

        Parameters
        ----------
        psd : ABCRoughness or callable
            2D isotropic PSD S(f) with an attribute sigma (RMS roughness);
            f in 1/length, S in length**4.
        wavelength : number
            Vacuum wavelength (same length unit).
        lever_arm : number
            L: scattering by the angle lambda*f lands at r = lambda*f*L on the
            wafer (same unit as the PSF radius).
        n : number, optional
            Refractive index (refracting surfaces); None for mirrors.
        n_surfaces : int, optional
        cos_theta : number, optional
            For mirrors.
        """

        total = n_surfaces*tis(psd.sigma, wavelength, n, cos_theta)
        scale = wavelength*lever_arm

        def psf(r):
            return total/psd.sigma**2*psd(np.asarray(r, dtype=float)/scale)/scale**2

        f_min, f_max = getattr(psd, "f_min", 0.0), getattr(psd, "f_max", np.inf)

        return cls(psf, total, f_min*scale, f_max*scale)

    def encircled(self, r):
        """Fraction of the light scattered within the radius r (of total)."""

        lo = max(self.r_min, 1e-12)
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(r)
        for k, rk in enumerate(r):
            if rk > lo:
                hi = min(rk, self.r_max) if np.isfinite(self.r_max) else rk
                out[k] = 2*np.pi*quad(lambda s: self.psf(s)*s, lo, hi, limit=200)[0]
        return out

    def level(self, density):
        """Flare level for a uniform pattern density (open fraction of the
        mask): total*density."""

        return self.total*np.asarray(density, dtype=float)

    def kernel(self, dx, shape):
        """The PSF integrated over the cells of a grid of spacing dx, shape
        (N_y, N_x), centred on [0, 0] with wrap-around: the centre cell holds
        the light within a disc of the same area, the others are averaged
        over 4 x 4 sub-points. The sum approaches total when the grid holds
        the whole range."""

        N_y, N_x = shape
        u = np.fft.fftfreq(N_x, 1.0/N_x)*dx
        v = np.fft.fftfreq(N_y, 1.0/N_y)*dx
        s = (np.arange(4) + 0.5)/4 - 0.5
        K = np.zeros(shape)
        for a in s:
            for b in s:
                R = np.hypot(u[None, :] + a*dx, v[:, None] + b*dx)
                K += self.psf(R)
        K *= dx**2/16
        K[0, 0] = self.encircled(dx/np.sqrt(np.pi))[0]

        return K

    def level_map(self, density, dx, pad=True):
        """Flare level across a die: the PSF convolved with a density map.

        Parameters
        ----------
        density : (N_y, N_x) array
            Open fraction of the mask on a coarse grid (0: chrome, 1: clear).
        dx : number
            Spacing of the map (same unit as the PSF).
        pad : bool, optional
            Pad with darkness (the reticle blades) to the size of the PSF
            instead of a periodic map.

        Returns
        -------
        level : (N_y, N_x) array
        """

        density = np.asarray(density, dtype=float)
        N_y, N_x = density.shape
        if pad:
            extra = int(np.ceil(min(self.r_max, 2*max(N_x, N_y)*dx)/dx))
            shape = (N_y + extra, N_x + extra)
        else:
            shape = (N_y, N_x)
        D = np.zeros(shape)
        D[:N_y, :N_x] = density
        K = self.kernel(dx, shape)
        out = np.fft.irfft2(np.fft.rfft2(D)*np.fft.rfft2(K), s=shape)

        return out[:N_y, :N_x]

    def apply(self, I, level):
        """Image with flare, (1 - total)*I + level.

        Parameters
        ----------
        I : array
            Image normalized to the open frame.
        level : number or array
            Flare level, e.g. level(density) or a value of level_map().
        """

        return (1 - self.total)*np.asarray(I, dtype=float) + level

    def __repr__(self):
        return "Flare(total={:.4g}, r_min={:.3g}, r_max={:.3g})".format(self.total, self.r_min,
                                                                      self.r_max)


__all__ = ["tis", "ABCRoughness", "Flare"]
