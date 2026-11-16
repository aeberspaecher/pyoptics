#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Focus and image blur: laser bandwidth times axial colour, and stage
vibration.

A scanner image is not taken in one focal plane. Two effects spread the
focus of every point over a distribution, and one blurs it sideways:

- Laser bandwidth. A refractive projection lens has axial colour: the
  focus moves by dz/dlambda (for the all fused silica lens of
  raytrace.caldwell_krf_lens() about 0.25 um per pm; lens.axial_colour()
  computes it from the ray tracer with System.with_wavelength()). A line
  narrowed excimer laser has a bandwidth of a few tenths of a pm (FWHM),
  so the image is the incoherent sum of images over a focus range of tens
  of nanometres, weighted with the laser spectrum (each wavelength is
  incoherent with the others).
- Vertical stage vibration. The wafer moves along z while a point crosses
  the slit; the moving standard deviation (MSD, z) of its height over the
  exposure time is a Gaussian spread of the focus. The moving average (MA)
  is a focus offset.
- Lateral stage vibration. The MSD in x and y blurs the image with a
  Gaussian (and the MA shifts it).

FocusBlur combines the laser spectrum (Lorentzian, Gaussian or a tabulated
one) and the vertical MSD into one focus distribution, evaluates the images
in all its foci in one batched call of LithoSimulation.through_focus() and
sums them; the lateral MSD is applied to the summed image. The focus spread
acts like a reduced depth of focus and lowers the contrast at best focus,
which is why scanners specify the bandwidth (FWHM and E95) and the MSD,
and why an intentionally broadened spectrum (or a focus drilling scan)
can extend the depth of focus of contacts.

Convention: a wavelength offset dlambda puts the image plane at focus -
dz_dlambda*dlambda relative to the focal plane at that wavelength (focus as
in LithoSimulation.image()); for symmetric spectra the sign does not matter.

Lengths in nm, the bandwidth in pm, dz_dlambda in nm per pm.

References
----------
H. J. Levinson, "Principles of Lithography", 4th ed., SPIE Press (2019)
(excimer laser bandwidth and E95, stage MSD and MA in the focus and overlay
budgets).

C. Mack, "Fundamental Principles of Optical Lithography: The Science of
Microfabrication", Wiley (2007).

A. Erdmann, "Optical and EUV Lithography: A Modeling Perspective", SPIE Press
(2021).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf


def _spectrum_cdf(kind, fwhm):
    """CDF of a line shape of the given FWHM centred on 0."""

    if kind == "lorentzian":
        g = 0.5*fwhm
        return lambda u: 0.5 + np.arctan(u/g)/np.pi
    if kind == "gaussian":
        s = fwhm/(2*np.sqrt(2*np.log(2)))
        return lambda u: 0.5*(1 + erf(u/(np.sqrt(2)*s)))
    raise ValueError("spectrum must be 'lorentzian', 'gaussian' or (offsets, weights)")


def e95(fwhm, spectrum="lorentzian"):
    """Width that holds 95 % of the energy of a line of the given FWHM
    (Lorentzian: 12.7*FWHM, Gaussian: 1.66*FWHM)."""

    if spectrum == "lorentzian":
        return fwhm*np.tan(0.475*np.pi)
    if spectrum == "gaussian":
        return fwhm/(2*np.sqrt(2*np.log(2)))*2*1.959963984540054
    raise ValueError("spectrum must be 'lorentzian' or 'gaussian'")


class FocusBlur(object):
    """Focus distribution from the laser spectrum and the stage MSD, and the
    blurred images.

    Parameters
    ----------
    dz_dlambda : number, optional
        Axial colour of the lens, nm of focus per pm of wavelength.
    bandwidth : number, optional
        FWHM of the laser line in pm.
    spectrum : {"lorentzian", "gaussian"} or (offsets, weights), optional
        Line shape; a tabulated spectrum gives wavelength offsets in pm and
        their weights (bandwidth is then ignored).
    z_msd : number, optional
        Vertical moving standard deviation of the stage (nm).
    xy_msd : number or pair, optional
        Lateral moving standard deviation (nm), the same or (x, y).
    n_foci : int, optional
        Planes that sample the focus distribution.
    extent : number, optional
        Half range of focus offsets that is sampled; default: 4 standard
        deviations of the Gaussian parts plus 2 FWHM of a Lorentzian line.
        The light of a Lorentzian line outside this range (`truncated`) is
        left out and the rest renormalized.
    """

    def __init__(self, dz_dlambda=0.0, bandwidth=0.0, spectrum="lorentzian", z_msd=0.0, xy_msd=0.0,
                 n_foci=15, extent=None):
        self.dz_dlambda, self.bandwidth, self.spectrum = dz_dlambda, bandwidth, spectrum
        self.z_msd, self.xy_msd = z_msd, xy_msd
        self.n_foci, self.extent = n_foci, extent

    def with_(self, **kwargs):
        """Copy with some parameters exchanged."""

        p = {k: v for k, v in vars(self).items() if k != "truncated"}
        p.update(kwargs)

        return FocusBlur(**p)

    @property
    def focus_fwhm(self):
        """FWHM of the chromatic focus spread (nm)."""

        return abs(self.dz_dlambda)*self.bandwidth

    def focus_distribution(self):
        """Focus offsets and weights (summing to 1).

        Returns
        -------
        offsets, weights : arrays
            Weighted means of n_foci equal bins and their probabilities
            (a tabulated spectrum without z_msd: its own focus offsets).
        truncated : number
            Probability outside the sampled range (also stored as
            self.truncated).
        """

        tabulated = not isinstance(self.spectrum, str)
        s_g = self.z_msd
        fwhm_z = self.focus_fwhm
        if tabulated:
            dl, w = (np.asarray(a, dtype=float) for a in self.spectrum)
            z_tab, w_tab = -self.dz_dlambda*dl, w/np.sum(w)
        elif self.spectrum == "gaussian":
            s_g = np.hypot(s_g, fwhm_z/(2*np.sqrt(2*np.log(2))))
            fwhm_z = 0.0
        else:
            _spectrum_cdf(self.spectrum, 1.0)  # validates the name

        if tabulated and s_g == 0:
            self.truncated = 0.0
            return z_tab, w_tab, 0.0
        if self.extent is not None:
            E = self.extent
        elif tabulated:
            E = np.max(np.abs(z_tab)) + 4*s_g
        else:
            E = 4*s_g + 2*fwhm_z
        if E == 0:
            self.truncated = 0.0
            return np.zeros(1), np.ones(1), 0.0

        # fine grid of the density (cell centres, symmetric), then n_foci bins
        M = 64*self.n_foci
        edges = np.linspace(-E, E, M + 1)
        z = 0.5*(edges[1:] + edges[:-1])
        dz = edges[1] - edges[0]
        gauss_cdf = lambda u, mu: 0.5*(1 + erf((u - mu)/(np.sqrt(2)*s_g)))
        if tabulated:
            p = np.zeros(M)
            for zt, wt in zip(z_tab, w_tab):
                p += wt*np.diff(gauss_cdf(edges, zt))
        elif fwhm_z > 0:
            p = np.diff(_spectrum_cdf("lorentzian", fwhm_z)(edges))
            if s_g > 0:
                n_k = int(np.ceil(5*s_g/dz))
                u = np.arange(-n_k - 0.5, n_k + 1)*dz
                k = np.diff(gauss_cdf(u, 0.0))
                p = np.convolve(p, k/np.sum(k), mode="same")
        else:
            p = np.diff(gauss_cdf(edges, 0.0))
        total = np.sum(p)
        self.truncated = float(max(0.0, 1 - total))
        b = np.arange(M)//64
        W = np.bincount(b, weights=p, minlength=self.n_foci)
        Z = np.bincount(b, weights=p*z, minlength=self.n_foci)
        keep = W > 1e-12*total
        offsets, weights = Z[keep]/W[keep], W[keep]/total

        return offsets, weights, self.truncated

    def blur_xy(self, I, dx, dy=None):
        """Gaussian blur of an image (whole periods, 1D cut along x or 2D
        (N_y, N_x)) with the lateral MSD."""

        s_x, s_y = np.broadcast_to(np.asarray(self.xy_msd, dtype=float), (2,))
        I = np.asarray(I, dtype=float)
        if s_x == 0 and s_y == 0:
            return I
        if I.ndim == 1:
            return gaussian_filter(I, s_x/dx, mode="wrap")
        dy = dx if dy is None else dy

        return gaussian_filter(I, (s_y/dy, s_x/dx), mode="wrap")

    def through_focus(self, sim, x, foci, y=None, depth=0.0):
        """Blurred images for several (mean) foci.

        Parameters
        ----------
        sim : LithoSimulation
        x, y : arrays
            As for LithoSimulation.image(); for a lateral MSD they must
            cover whole periods with equidistant points.
        foci : sequence of numbers
        depth : number, optional

        Returns
        -------
        I : array
            Shape (len(foci),) + image shape.
        """

        offsets, weights, _ = self.focus_distribution()
        foci = np.atleast_1d(np.asarray(foci, dtype=float))
        planes = (foci[:, None] + offsets[None, :]).ravel()
        I = sim.through_focus(x, planes, y=y, depth=depth)
        I = I.reshape((len(foci), len(offsets)) + I.shape[1:])
        I = np.tensordot(weights, I, axes=(0, 1))
        if np.any(np.asarray(self.xy_msd) != 0):
            dx = x[1] - x[0]
            dy = None if y is None else y[1] - y[0]
            I = np.array([self.blur_xy(Ik, dx, dy) for Ik in I])

        return I

    def image(self, sim, x, y=None, focus=0.0, depth=0.0):
        """Blurred image at one (mean) focus, see through_focus()."""

        return self.through_focus(sim, x, [focus], y, depth)[0]

    def __repr__(self):
        return ("FocusBlur(dz_dlambda={}, bandwidth={}, spectrum={!r}, z_msd={}, xy_msd={})"
                .format(self.dz_dlambda, self.bandwidth,
                        self.spectrum if isinstance(self.spectrum, str) else "tabulated",
                        self.z_msd, self.xy_msd))


__all__ = ["FocusBlur", "e95"]
