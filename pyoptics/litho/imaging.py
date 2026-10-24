#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Projection lens and lithography simulation (Abbe's method).

For every source point the mask (a MaskLayout with a mask stack) diffracts
the illuminating plane wave into orders with the transverse wave vectors
k_0*NA*s + 2*pi*(m/p_x, n/p_y) and Jones matrices given by the Fourier
coefficients of its regions times their transmissions. The projector passes
the orders inside its pupil, applies aberrations and defocus and, in the
vectorial model, turns each order into a plane wave in the immersion medium
with the aplanatic polarization rotation (as vectorial_imaging does). The
optional WaferStack then gives the field inside the resist. Fields are
summed over the orders on any rectilinear grid by two matrix products
(exp(i*k_y*y) @ A @ exp(i*k_x*x).T), which is exact for any pitch and any
grid; intensities are summed over the mutually incoherent polarization
states and source points.

Images are normalized to the open frame (a clear mask) in the same plane, so
a threshold of e.g. 0.3 means 30 % of the clear field intensity; a dose D
prints where D*I exceeds the threshold.

Convention: fields vary as exp(i(k.r - omega*t)); a focus z > 0 places the
image plane (or the top of the wafer stack) behind the focal plane, each
plane wave gets exp(+i*k_z*z).
"""

import numpy as np

from ..bases import fringe_to_n_m, fringe_zernike_R_nm, fringe_zernike_Y_m
from ..propagators import aplanatic_rotation
from . import metrics


class Projector(object):
    """Projection lens.

    Parameters
    ----------
    NA : number
        Image-side numerical aperture (NA <= n).
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index of the image space (immersion liquid; water at 193
        nm: 1.437).
    magnification : number, optional
        Reduction ratio M (4 for scanners); sets the angles at the mask.
    vectorial : bool, optional
        Vectorial (aplanatic polarization rotation) or scalar model.
    obliquity : bool, optional
        Radiometric factor 1/sqrt(cos(theta)) (vectorial model only).
    zernikes : dict, optional
        Fringe Zernike coefficients of the wavefront in waves, e.g. {9:
        0.02} (spherical aberration), on the normalized pupil.
    phase : callable, optional
        Additional pupil phase phase(P_x, P_y) in radians.
    """

    def __init__(self, NA, wavelength, n=1.0, magnification=4.0, vectorial=True, obliquity=True,
                 zernikes=None, phase=None):
        if NA > n:
            raise ValueError("NA = {} exceeds the refractive index n = {}".format(NA, n))
        self.NA, self.wavelength, self.n = NA, wavelength, n
        self.magnification = magnification
        self.vectorial, self.obliquity = vectorial, obliquity
        self.zernikes = dict(zernikes or {})
        self.phase = phase

    @property
    def k0(self):
        return 2*np.pi/self.wavelength

    def pupil_phase(self, P_x, P_y):
        """Wavefront phase in radians on normalized pupil coordinates."""

        W = np.zeros(np.shape(P_x))
        if self.zernikes:
            rho, phi = np.hypot(P_x, P_y), np.arctan2(P_y, P_x)
            for j, c in self.zernikes.items():
                n, m = fringe_to_n_m(j)
                W = W + c*fringe_zernike_R_nm(n, abs(m), rho)*fringe_zernike_Y_m(m, phi)
        W = 2*np.pi*W
        if self.phase is not None:
            W = W + self.phase(P_x, P_y)

        return W

    def with_(self, **kwargs):
        """Copy with changed parameters, e.g. projector.with_(vectorial=False)."""

        p = dict(NA=self.NA, wavelength=self.wavelength, n=self.n,
                 magnification=self.magnification, vectorial=self.vectorial,
                 obliquity=self.obliquity, zernikes=self.zernikes, phase=self.phase)
        p.update(kwargs)

        return Projector(**p)

    @property
    def resolution_unit(self):
        """lambda/NA, e.g. for k_1 = hp*NA/lambda."""

        return self.wavelength/self.NA

    def __repr__(self):
        return "Projector(NA={}, wavelength={}, n={}, {})".format(
            self.NA, self.wavelength, self.n, "vectorial" if self.vectorial else "scalar")


class LithoSimulation(object):
    """Imaging of a mask with an illumination, a projector and an optional
    wafer stack.

    Parameters
    ----------
    layout : MaskLayout
    stack : ThinMask or FilmMask
    illumination : Illumination
    projector : Projector
    wafer : WaferStack, optional
        None: aerial image in the immersion medium.
    normalize : bool, optional
        Normalize images to the open frame.
    """

    def __init__(self, layout, stack, illumination, projector, wafer=None, normalize=True):
        self.layout, self.stack = layout, stack
        self.illumination, self.projector = illumination, projector
        self.wafer = wafer
        self.normalize = normalize

    def with_(self, **kwargs):
        """Copy with some parts exchanged, e.g. sim.with_(illumination=...)."""

        p = dict(layout=self.layout, stack=self.stack, illumination=self.illumination,
                 projector=self.projector, wafer=self.wafer, normalize=self.normalize)
        p.update(kwargs)

        return LithoSimulation(**p)

    # orders and Jones matrices:

    def orders(self):
        """Order indices m, n and the transverse spatial angular frequencies
        of the mask (without the shift by the source point)."""

        P = self.projector
        M, N = self.layout.max_orders(P.NA, P.wavelength, self.illumination.sigma_max)
        m, n = np.arange(-M, M + 1), np.arange(-N, N + 1)
        f_x = np.zeros(len(m)) if self.layout.pitch_x is None else 2*np.pi*m/self.layout.pitch_x
        f_y = np.zeros(len(n)) if self.layout.pitch_y is None else 2*np.pi*n/self.layout.pitch_y

        return m, n, f_x, f_y

    def _region_jones(self, s_x, s_y):
        """Jones matrices of the regions of the layout for the source point."""

        P = self.projector
        J_clear, J_abs = self.stack.jones(s_x, s_y, P.NA, P.wavelength, P.magnification)
        if self.layout.tone == "clear":
            J = {"field": J_clear, "feature": J_abs}
        else:
            J = {"field": J_abs, "feature": J_clear}
        J["clear"] = J_clear
        bl = getattr(self.stack, "boundary_layer", None)
        if bl is not None:
            J["edge_y"], J["edge_x"] = bl.jones()

        return J

    def _boundary_width(self):
        bl = getattr(self.stack, "boundary_layer", None)

        return 0.0 if bl is None else bl.width

    # images:

    def _intensity(self, coeffs, f_x, f_y, x, y, focus, depth, return_components):
        P = self.projector
        k0, kNA = P.k0, P.k0*P.NA
        k = k0*P.n
        x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        W_x, W_y = np.exp(1j*np.outer(x, f_x)), np.exp(1j*np.outer(y, f_y))
        I_xyz = np.zeros((3, len(y), len(x)))

        for s_x, s_y, w, states in self.illumination:
            K_x, K_y = np.meshgrid(f_x + kNA*s_x, f_y + kNA*s_y)
            passed = K_x**2 + K_y**2 <= kNA**2*(1 + 1E-12)
            if not np.any(passed):
                continue
            rows, cols = np.any(passed, axis=1), np.any(passed, axis=0)
            K_x, K_y, passed = K_x[rows][:, cols], K_y[rows][:, cols], passed[rows][:, cols]
            Wx, Wy = W_x[:, cols], W_y[:, rows]

            J = self._region_jones(s_x, s_y)
            T = sum(J[r][:, :, None, None]*c[rows][:, cols] for r, c in coeffs.items())

            Sigma_x, Sigma_y = np.where(passed, K_x/k, 0.0), np.where(passed, K_y/k, 0.0)
            cos_theta = np.sqrt(np.clip(1 - Sigma_x**2 - Sigma_y**2, 1E-12, None))
            pupil = passed*np.exp(1j*(P.pupil_phase(K_x/kNA, K_y/kNA) + k*cos_theta*focus))
            if P.vectorial:
                R = aplanatic_rotation(Sigma_x, Sigma_y)
                if P.obliquity:
                    R = R/np.sqrt(cos_theta)

            for weight, jones in states:
                E_pupil = np.einsum("ijnm,j->inm", T, jones)*pupil
                if P.vectorial:
                    E = np.einsum("ijnm,jnm->inm", R, E_pupil)
                else:
                    E = np.concatenate([E_pupil, np.zeros((1,) + E_pupil.shape[1:])])
                if self.wafer is not None:
                    E = self.wafer.transmit(E, K_x, K_y, k0, P.n, depth, P.vectorial)
                for i in range(3):
                    I_xyz[i] += w*weight*np.abs(Wy @ E[i] @ Wx.T)**2

        I = I_xyz.sum(axis=0)

        return (I, I_xyz) if return_components else I

    def open_frame(self, focus=0.0, depth=0.0):
        """Intensity of a clear mask in the same plane (normalization)."""

        clear = {"clear": np.ones((1, 1), dtype=complex)}
        I = self._intensity(clear, np.zeros(1), np.zeros(1), [0.0], [0.0], focus, depth, False)

        return I[0, 0]

    def image(self, x, y=None, focus=0.0, depth=0.0, return_components=False):
        """Image intensity.

        Parameters
        ----------
        x : array
            x positions (image side).
        y : array, optional
            y positions; None gives the cut at y = 0 as a 1D array.
        focus : number, optional
            Position of the image plane (with a wafer stack: of the top of
            the stack) relative to the focal plane.
        depth : number, optional
            Depth below the resist top (wafer stack only).
        return_components : bool, optional
            Also return |E_x|**2, |E_y|**2, |E_z|**2.

        Returns
        -------
        I : array
            Shape (len(y), len(x)), or (len(x),) for y=None; normalized to
            the open frame if self.normalize.
        I_xyz : array, optional
            Shape (3,) + I.shape.
        """

        m, n, f_x, f_y = self.orders()
        coeffs = self.layout.region_coefficients(m, n, self._boundary_width())
        I, I_xyz = self._intensity(coeffs, f_x, f_y, x, [0.0] if y is None else y, focus, depth,
                                   True)
        if self.normalize:
            I0 = self.open_frame(focus, depth)
            I, I_xyz = I/I0, I_xyz/I0
        if y is None:
            I, I_xyz = I[0], I_xyz[:, 0]

        return (I, I_xyz) if return_components else I

    def through_focus(self, x, foci, y=None, depth=0.0):
        """Images for all foci, shape (len(foci),) + image shape."""

        return np.array([self.image(x, y, f, depth) for f in foci])

    # helpers for metrics:

    def cut_coordinates(self, N=201, periods=1.0, direction=None):
        """A cut through the center of the first feature, across it: returns
        (positions, center, cd, direction) with direction "x" or "y"."""

        L = self.layout
        if direction is None:
            direction = "y" if L.pitch_x is None else "x"
        x0, y0, a, b = L.features[0]
        if direction == "x":
            pitch, center, cd = L.pitch_x, x0, a
        else:
            pitch, center, cd = L.pitch_y, y0, b
        u = center + np.linspace(-periods*pitch/2, periods*pitch/2, N)

        return u, center, cd, direction

    def cut(self, N=401, periods=1.0, direction=None, focus=0.0, depth=0.0):
        """Image along a cut through the first feature: returns (u, I)."""

        u, center, cd, direction = self.cut_coordinates(N, periods, direction)
        x0, y0 = self.layout.features[0][:2]
        if direction == "x":
            I = self.image(u, None if self.layout.pitch_y is None else [y0], focus, depth)
        else:
            I = self.image([x0], u, focus, depth)
        I = np.ravel(I)

        return u, I

    def cd(self, threshold=0.3, dose=1.0, focus=0.0, depth=0.0, N=401, direction=None):
        """Printed CD of the first feature for the threshold model."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        _, I = self.cut(N, 1.0, direction, focus, depth)

        return metrics.threshold_cd(u, dose*I, threshold, center, self.layout.feature_is_bright)

    def fem(self, foci, doses, threshold=0.3, depth=0.0, N=401, direction=None):
        """Focus-exposure matrix: CD of the first feature, shape
        (len(doses), len(foci))."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        cuts = [self.cut(N, 1.0, direction, f, depth)[1] for f in foci]

        return metrics.fem_cd(u, cuts, doses, threshold, center, self.layout.feature_is_bright)

    def dose_to_size(self, cd_target=None, threshold=0.3, focus=0.0, depth=0.0, N=401,
                     direction=None):
        """Dose that prints the first feature at cd_target (default: its
        nominal size)."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        _, I = self.cut(N, 1.0, direction, focus, depth)

        return metrics.dose_to_size(u, I, cd if cd_target is None else cd_target, threshold, center)

    def meef(self, threshold=0.3, dose=1.0, delta=1.0, focus=0.0, depth=0.0, N=801,
             direction=None):
        """Mask error enhancement factor dCD_wafer/dCD_mask (both at wafer
        scale) from mask CDs biased by -+delta/2."""

        cds = [self.with_(layout=self.layout.scaled(bias=b)).cd(threshold, dose, focus, depth, N,
                                                                 direction)
               for b in (-delta/2, delta/2)]

        return (cds[1] - cds[0])/delta

    def nils(self, threshold=None, dose=1.0, focus=0.0, depth=0.0, N=801, direction=None):
        """NILS of the first feature: cd*|d ln(I)/du| at its nominal edges
        (threshold=None) or at the printed edges for the threshold."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        _, I = self.cut(N, 1.0, direction, focus, depth)
        if threshold is None:
            return np.mean([metrics.nils(u, I, center + s*cd/2, cd) for s in (-1, 1)])

        return metrics.nils_at_threshold(u, dose*I, threshold, center, self.layout.feature_is_bright)
