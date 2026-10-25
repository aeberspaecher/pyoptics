#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Projection lens and lithography simulation (Abbe's method).

For every source point the mask (a MaskLayout with a mask stack) diffracts
the illuminating plane wave into orders with the transverse wave vectors
k_0*NA*s + 2*pi*(m/p_x, n/p_y) and Jones matrices given by the Fourier
coefficients of its regions times their transmissions (a RigorousMask
gives the Jones matrices of the orders directly, computed with RCWA for each
source point). The projector passes
the orders inside its pupil, applies apodization, aberrations and defocus
and, in the vectorial model, turns each order into a plane wave in the
immersion medium with the aplanatic polarization rotation (as
vectorial_imaging does). The
optional WaferStack then gives the field inside the resist. Fields are
summed over the orders on any rectilinear grid by two matrix products
(exp(i*k_y*y) @ A @ exp(i*k_x*x).T), which is exact for any pitch and any
grid; intensities are summed over the mutually incoherent polarization
states and source points.

All source points, polarization states and foci are computed together in
batches of array operations. Usually the intensities are not summed field by
field but via the mutual intensities of the orders, sum_b w_b*A_b*A_b^H over
all fields A_b, which are then synthesized into the image once; this is the
same sum in another order (no approximation as in Hopkins' SOCS
decompositions) and is much faster unless there are many orders and few
image points.

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


def _synthesize(M, W_x, W_y):
    """Intensity sum_{o, o'} M[..., o, o'] exp(i*(f_o - f_o').r) on the grid
    of W_x = exp(i*x*f_x), W_y = exp(i*y*f_y) from mutual intensities M of
    shape (..., n_y*n_x, n_y*n_x) of the orders (n, m) (n along y); returns
    shape (..., len(y), len(x))."""

    n_x, n_y = W_x.shape[1], W_y.shape[1]
    n_o = n_x*n_y
    lead = M.shape[:-2]
    # M[(n, m), (n', m')] -> M[(n, n'), (m, m')], then U_y @ M @ U_x.T
    M = M.reshape(lead + (n_y, n_x, n_y, n_x)).swapaxes(-3, -2).reshape(lead + (n_y**2, n_x**2))
    U_x = (W_x[:, :, None]*W_x.conj()[:, None, :]).reshape(len(W_x), n_x**2)
    U_y = (W_y[:, :, None]*W_y.conj()[:, None, :]).reshape(len(W_y), n_y**2)
    if len(W_y)*n_o**2 + len(W_y)*n_x**2*len(W_x) <= n_o**2*len(W_x) + len(W_y)*n_y**2*len(W_x):
        return ((U_y @ M) @ U_x.T).real

    return (U_y @ (M @ U_x.T)).real


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
    apodization_zernikes : dict, optional
        Apodization (pupil amplitude transmission) as Fringe Zernike
        coefficients a_j. With apodization_model="linear" (default) the
        amplitude transmission is A = 1 + sum_j a_j*Z_j(P_x, P_y), so the
        pupil function is A*exp(i*W); e.g. {4: -0.1} lets the transmission
        fall off towards the rim of the pupil (Z_4 = 2*rho**2 - 1), {4: -0.1,
        9: 0.05} adds a fourth order term, and a constant (a_1) cancels in
        normalized images. A must not become negative inside the pupil
        (ValueError). With apodization_model="log" the coefficients describe
        the log amplitude instead: A = exp(sum_j a_j*Z_j), which is positive
        for any coefficients and makes the coefficients of two apodizing
        elements add up; both agree to first order in a_j.
    apodization_model : {"linear", "log"}, optional
        How apodization_zernikes are turned into the amplitude (see above).
    apodization : callable, optional
        Additional amplitude transmission apodization(P_x, P_y) (real, >= 0)
        on normalized pupil coordinates, multiplied with the Zernike part.
    jones_pupil : callable, optional
        Polarization of the lens: jones_pupil(P_x, P_y) returns Jones
        matrices of shape np.shape(P_x) + (2, 2) that act on the Jones
        vector (E_x, E_y) of every order before the aplanatic rotation into
        the image space, on top of the scalar pupil above (see
        pyoptics.litho.jones_pupil, e.g. PauliZernikePupil or
        PhysicalJonesPupil). None: a polarization-free lens.
    """

    def __init__(self, NA, wavelength, n=1.0, magnification=4.0, vectorial=True, obliquity=True,
                 zernikes=None, phase=None, apodization_zernikes=None, apodization=None,
                 apodization_model="linear", jones_pupil=None):
        if NA > n:
            raise ValueError("NA = {} exceeds the refractive index n = {}".format(NA, n))
        self.NA, self.wavelength, self.n = NA, wavelength, n
        self.magnification = magnification
        self.vectorial, self.obliquity = vectorial, obliquity
        self.zernikes = dict(zernikes or {})
        self.phase = phase
        self.apodization_zernikes = dict(apodization_zernikes or {})
        self.apodization = apodization
        if apodization_model not in ("linear", "log"):
            raise ValueError("apodization_model must be 'linear' or 'log', not {!r}".format(apodization_model))
        self.apodization_model = apodization_model
        self.jones_pupil = jones_pupil

    @property
    def k0(self):
        return 2*np.pi/self.wavelength

    @staticmethod
    def _zernike_sum(coeffs, P_x, P_y):
        W = np.zeros(np.shape(P_x))
        if coeffs:
            rho, phi = np.hypot(P_x, P_y), np.arctan2(P_y, P_x)
            for j, c in coeffs.items():
                n, m = fringe_to_n_m(j)
                W = W + c*fringe_zernike_R_nm(n, abs(m), rho)*fringe_zernike_Y_m(m, phi)

        return W

    def pupil_phase(self, P_x, P_y):
        """Wavefront phase in radians on normalized pupil coordinates."""

        W = 2*np.pi*self._zernike_sum(self.zernikes, P_x, P_y)
        if self.phase is not None:
            W = W + self.phase(P_x, P_y)

        return W

    def pupil_amplitude(self, P_x, P_y):
        """Amplitude transmission of the pupil (apodization) on normalized
        pupil coordinates; 1 without apodization."""

        S = self._zernike_sum(self.apodization_zernikes, P_x, P_y)
        if self.apodization_model == "log":
            A = np.exp(S)
        else:
            A = 1.0 + S
            inside = np.broadcast_to(np.hypot(P_x, P_y) <= 1.0, np.shape(A))
            if np.any(A[inside] < 0):
                raise ValueError("apodization_zernikes give a negative amplitude transmission "
                                 "inside the pupil (min {:.3g})".format(np.min(A[inside])))
        if self.apodization is not None:
            A = A*self.apodization(P_x, P_y)

        return A

    def with_(self, **kwargs):
        """Copy with changed parameters, e.g. projector.with_(vectorial=False)."""

        p = dict(NA=self.NA, wavelength=self.wavelength, n=self.n,
                 magnification=self.magnification, vectorial=self.vectorial,
                 obliquity=self.obliquity, zernikes=self.zernikes, phase=self.phase,
                 apodization_zernikes=self.apodization_zernikes, apodization=self.apodization,
                 apodization_model=self.apodization_model, jones_pupil=self.jones_pupil)
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
    stack : ThinMask, FilmMask or RigorousMask
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
        """Jones matrices of the regions of the layout for the source
        point(s), shape np.shape(s_x) + (2, 2)."""

        P = self.projector
        try:
            J_clear, J_abs = self.stack.jones(s_x, s_y, P.NA, P.wavelength, P.magnification)
        except Exception:  # a stack for single points only; errors reappear below
            J_clear = None
        if np.shape(J_clear) != np.shape(s_x) + (2, 2):
            J_clear, J_abs = (np.array(J) for J in zip(*[
                self.stack.jones(a, b, P.NA, P.wavelength, P.magnification)
                for a, b in zip(np.ravel(s_x), np.ravel(s_y))]))
            J_clear = J_clear.reshape(np.shape(s_x) + (2, 2))
            J_abs = J_abs.reshape(np.shape(s_x) + (2, 2))
        if self.layout.tone == "clear":
            J = {"field": J_clear, "feature": J_abs}
        else:
            J = {"field": J_abs, "feature": J_clear}
        J["clear"] = J_clear
        bl = getattr(self.stack, "boundary_layer", None)
        if bl is not None:
            J["edge_y"], J["edge_x"] = (np.broadcast_to(J_bl, np.shape(s_x) + (2, 2))
                                        for J_bl in bl.jones())

        return J

    def _boundary_width(self):
        bl = getattr(self.stack, "boundary_layer", None)

        return 0.0 if bl is None else bl.width

    # images:

    # Memory for the fields of one batch of source points (the sum over the
    # source is done in batches of points, polarization states and foci with
    # a few large array operations instead of many small ones), and for the
    # mutual intensities of the orders (see _intensity):
    batch_bytes = 32*2**20
    mutual_bytes = 256*2**20

    def _polarization_table(self):
        """Weights (shape (points, states)) and Jones vectors (points, states,
        2) of the polarization states of all source points; points with
        fewer states are padded with zero weights."""

        ill = self.illumination
        n_points = len(ill.s_x)
        if not callable(ill.polarization):  # same states everywhere, TE/TM vectors vary
            states = ill.polarization_states(ill.s_x, ill.s_y)
            w = np.array([np.broadcast_to(weight, n_points) for weight, J in states]).T
            jones = np.array([np.broadcast_to(np.reshape(J, (2, -1)).T, (n_points, 2))
                              for weight, J in states]).transpose(1, 0, 2)
            return ill.weights[:, None]*w, jones

        states = [ill.polarization_states(s_x, s_y) for s_x, s_y in zip(ill.s_x, ill.s_y)]
        n_states = max(len(st) for st in states)
        w = np.zeros((n_points, n_states))
        jones = np.zeros((n_points, n_states, 2), dtype=complex)
        for i, st in enumerate(states):
            for j, (weight, J) in enumerate(st):
                w[i, j], jones[i, j] = weight, J

        return ill.weights[:, None]*w, jones

    def _intensity(self, coeffs, f_x, f_y, x, y, foci, depth):
        """``|E_x|**2``, ``|E_y|**2``, ``|E_z|**2`` summed over the source, shape
        (len(foci), 3, len(y), len(x))."""

        x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        foci = np.atleast_1d(np.asarray(foci, dtype=float))
        W_x, W_y = np.exp(1j*np.outer(x, f_x)), np.exp(1j*np.outer(y, f_y))
        n_x, n_y = len(f_x), len(f_y)
        I_xyz = np.zeros((len(foci), 3, len(y), len(x)))

        rigorous = coeffs is None  # the mask stack gives the Jones matrices of the orders
        if rigorous:
            m, n = self.orders()[:2]
        else:
            regions = list(coeffs)
            C = np.array([coeffs[r] for r in regions])  # (regions, n_y, n_x)
        weights, jones = self._polarization_table()
        s_x_all, s_y_all = self.illumination.s_x, self.illumination.s_y
        n_states = weights.shape[1]
        n_f, n_o = len(foci)*n_states*3, n_y*n_x  # fields per source point, orders
        n_b = len(s_x_all)*n_f

        # Two exact ways to sum the intensities: directly, as |W_y @ F @
        # W_x.T|**2 for each field F of each source point, polarization state
        # and focus, or via the mutual intensities M = sum_b w_b*F_b*F_b^H of
        # the orders (shape (orders, orders), summed over the source first),
        # I(x, y) = sum W_y[y, n]*W_y[y, n']^* M[(n, m), (n', m')]
        # W_x[x, m]*W_x[x, m']^*. The latter is much cheaper unless there are
        # many orders and few image points.
        cost_direct = n_b*min(len(y)*n_o + len(y)*n_x*len(x), n_o*len(x) + len(y)*n_y*len(x))
        cost_mutual = n_b*n_o**2 + len(foci)*3*min(len(y)*n_o**2 + len(y)*n_x**2*len(x),
                                                    n_o**2*len(x) + len(y)*n_y**2*len(x))
        mutual = (cost_mutual < cost_direct
                  and 16*len(foci)*3*n_o**2 <= self.mutual_bytes)
        if mutual:
            M = np.zeros((len(foci), 3, n_o, n_o), dtype=complex)
            per_point = 16*n_f*n_o*4
        else:
            per_point = 16*n_f*(n_o + len(y)*len(x) + max(len(y)*n_x, n_y*len(x)))
        batch = max(1, int(self.batch_bytes//per_point))

        for start in range(0, len(s_x_all), batch):
            sl = slice(start, start + batch)
            s_x, s_y = s_x_all[sl], s_y_all[sl]
            passed, pupil, G = self._transfer(s_x, s_y, f_x, f_y, foci, depth)
            if not np.any(passed):
                continue

            # Jones vectors behind the mask: sum over the regions of their
            # Jones matrices times the incident states times their coefficients
            if rigorous:
                P_ = self.projector
                J_o = self.stack.order_jones(self.layout, m, n, s_x, s_y, P_.NA, P_.wavelength,
                                             P_.magnification)  # (points, 2, 2, n_y, n_x)
                E_pupil = np.einsum("pijnm,psj->psinm", J_o, jones[sl])
            else:
                J = self._region_jones(s_x, s_y)
                J = np.stack([J[r] for r in regions], axis=1)  # (points, regions, 2, 2)
                V = np.einsum("prij,psj->psri", J, jones[sl])
                E_pupil = np.einsum("psri,rnm->psinm", V, C)  # (points, states, 2, n_y, n_x)

            H = np.einsum("ijpnm,psjnm->psinm", G, E_pupil)  # (points, states, 3, n_y, n_x)
            F = pupil[:, :, None, None]*H[None]  # (foci, points, states, 3, n_y, n_x)
            w = weights[sl]

            if mutual:
                F = F.transpose(0, 3, 4, 5, 1, 2).reshape(len(foci), 3, n_o, -1)*np.sqrt(w.ravel())
                M += F @ F.conj().swapaxes(-1, -2)
                continue

            # W_y @ F[b] @ W_x.T for all b as two matrix products
            F = F.reshape(-1, n_y, n_x)
            if len(y)*n_x <= n_y*len(x):
                A = np.tensordot(W_y, F, axes=(1, 1))  # (len(y), b, n_x)
                A = (A.reshape(-1, n_x) @ W_x.T).reshape(len(y), -1, len(x))
            else:
                A = (F.reshape(-1, n_x) @ W_x.T).reshape(-1, n_y, len(x))
                A = np.tensordot(W_y, A, axes=(1, 1))  # (len(y), b, len(x))
            A = A.reshape((len(y), len(foci)) + w.shape + (3, len(x)))
            I_xyz += np.tensordot(A.real**2 + A.imag**2, w,
                                  axes=([2, 3], [0, 1])).transpose(1, 2, 0, 3)

        if mutual:
            I_xyz = _synthesize(M, W_x, W_y)

        return I_xyz

    def _transfer(self, s_x, s_y, f_x, f_y, foci, depth):
        """Pupil and polarization transfer of the orders (spatial angular
        frequencies f_x, f_y) for the source points s_x, s_y.

        Returns
        -------
        passed : bool array (points, n_y, n_x)
            Orders inside the pupil.
        pupil : array (foci, points, n_y, n_x)
            Apodization, aberrations and defocus (0 outside the pupil).
        G : array (3, 2, points, n_y, n_x)
            Maps pupil Jones vectors to the fields in image space (or in the
            resist).
        """

        P = self.projector
        k0, kNA = P.k0, P.k0*P.NA
        k = k0*P.n
        K_x, K_y = np.broadcast_arrays(f_x[None, None, :] + kNA*s_x[:, None, None],
                                       f_y[None, :, None] + kNA*s_y[:, None, None])
        passed = K_x**2 + K_y**2 <= kNA**2*(1 + 1E-12)  # (points, n_y, n_x)
        Sigma_x, Sigma_y = np.where(passed, K_x/k, 0.0), np.where(passed, K_y/k, 0.0)
        cos_theta = np.sqrt(np.clip(1 - Sigma_x**2 - Sigma_y**2, 1E-12, None))
        P_x, P_y = K_x/kNA, K_y/kNA
        amplitude = passed*P.pupil_amplitude(P_x, P_y)
        phase = P.pupil_phase(P_x, P_y)
        pupil = amplitude*np.exp(1j*(phase + k*cos_theta*np.asarray(foci)[:, None, None, None]))

        if P.vectorial:
            G = aplanatic_rotation(Sigma_x, Sigma_y)
            if P.obliquity:
                G = G/np.sqrt(cos_theta)
        else:
            G = np.zeros((3, 2) + passed.shape)
            G[0, 0], G[1, 1] = 1.0, 1.0
        if P.jones_pupil is not None:
            J = P.jones_pupil(np.where(passed, P_x, 0.0), np.where(passed, P_y, 0.0))
            G = np.einsum("ij...,...jk->ik...", G, J)
        if self.wafer is not None:
            K_x_in, K_y_in = np.where(passed, K_x, 0.0), np.where(passed, K_y, 0.0)
            G = np.where(passed, self.wafer.transmit(G.astype(complex), K_x_in, K_y_in, k0,
                                                     P.n, depth, P.vectorial), 0.0)

        return passed, pupil, G

    def _open_frames(self, foci, depth):
        clear = {"clear": np.ones((1, 1), dtype=complex)}
        I_xyz = self._intensity(clear, np.zeros(1), np.zeros(1), [0.0], [0.0], foci, depth)

        return I_xyz.sum(axis=1)[:, 0, 0]

    def open_frame(self, focus=0.0, depth=0.0):
        """Intensity of a clear mask in the same plane (normalization)."""

        return self._open_frames([focus], depth)[0]

    def _images(self, x, y, foci, depth):
        """Images and components for all foci, shapes (len(foci),) + image
        shape and (len(foci), 3) + image shape."""

        m, n, f_x, f_y = self.orders()
        coeffs = (None if hasattr(self.stack, "order_jones")
                  else self.layout.region_coefficients(m, n, self._boundary_width()))
        I_xyz = self._intensity(coeffs, f_x, f_y, x, [0.0] if y is None else y, foci, depth)
        if self.normalize:
            I_xyz = I_xyz/self._open_frames(foci, depth)[:, None, None, None]
        if y is None:
            I_xyz = I_xyz[:, :, 0]

        return I_xyz.sum(axis=1), I_xyz

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
            Also return ``|E_x|**2``, ``|E_y|**2``, ``|E_z|**2``.

        Returns
        -------
        I : array
            Shape (len(y), len(x)), or (len(x),) for y=None; normalized to
            the open frame if self.normalize.
        I_xyz : array, optional
            Shape (3,) + I.shape.
        """

        I, I_xyz = self._images(x, y, [focus], depth)

        return (I[0], I_xyz[0]) if return_components else I[0]

    def through_focus(self, x, foci, y=None, depth=0.0):
        """Images for all foci, shape (len(foci),) + image shape."""

        return self._images(x, y, foci, depth)[0]

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

    def _cuts(self, N, periods, direction, foci, depth):
        """Images along the cut for all foci: returns (u, I), I of shape
        (len(foci), N)."""

        u, center, cd, direction = self.cut_coordinates(N, periods, direction)
        x0, y0 = self.layout.features[0][:2]
        if direction == "x":
            I = self._images(u, None if self.layout.pitch_y is None else [y0], foci, depth)[0]
        else:
            I = self._images([x0], u, foci, depth)[0]

        return u, I.reshape(len(foci), -1)

    def cut(self, N=401, periods=1.0, direction=None, focus=0.0, depth=0.0):
        """Image along a cut through the first feature: returns (u, I)."""

        u, I = self._cuts(N, periods, direction, [focus], depth)

        return u, I[0]

    def cd(self, threshold=0.3, dose=1.0, focus=0.0, depth=0.0, N=401, direction=None):
        """Printed CD of the first feature for the threshold model."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        _, I = self.cut(N, 1.0, direction, focus, depth)

        return metrics.threshold_cd(u, dose*I, threshold, center, self.layout.feature_is_bright)

    def fem(self, foci, doses, threshold=0.3, depth=0.0, N=401, direction=None):
        """Focus-exposure matrix: CD of the first feature, shape
        (len(doses), len(foci))."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        cuts = self._cuts(N, 1.0, direction, foci, depth)[1]

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
        """NILS of the first feature: ``cd*|d ln(I)/du|`` at its nominal edges
        (threshold=None) or at the printed edges for the threshold."""

        u, center, cd, direction = self.cut_coordinates(N, 1.0, direction)
        _, I = self.cut(N, 1.0, direction, focus, depth)
        if threshold is None:
            return np.mean([metrics.nils(u, I, center + s*cd/2, cd) for s in (-1, 1)])

        return metrics.nils_at_threshold(u, dose*I, threshold, center, self.layout.feature_is_bright)
