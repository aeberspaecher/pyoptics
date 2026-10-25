#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Hopkins imaging: transmission cross coefficients (TCC) and their sum of
coherent systems (SOCS) decomposition.

Abbe's method (LithoSimulation) sums the image over the source points, so
every new mask costs a full source integration. Hopkins' method moves the
source integration in front: if the mask diffracts the same way for all
source points (the Hopkins approximation), the image is

    I(r) = sum_{o, o'} a_o a_o'^* TCC[o, o'] exp(i*(f_o - f_o').r)

with the mask spectrum a_o (the diffraction orders o = (m, n) of one period)
and the transmission cross coefficients

    TCC[o, o'] = sum_{source points s, polarization states, field components}
                 w_s H_o(s) H_o'(s)^*,

where H_o(s) is the field in the image (or in the resist) that a unit order
o produces for the source point s: pupil (apodization, aberrations, defocus)
and the polarization transfer of the projector and the wafer stack. The TCC
depends on the illumination, the projector, the wafer stack, the focus and
the pitch, but not on the shapes on the mask: computed once, it images any
layout with the same pitches by a sum over pairs of orders.

Polarization: the mask spectrum is a field of Jones matrices, a_o =
sum_regions J_region*C_region[o]. If all Jones matrices are multiples of
the identity (thin masks, film masks at normal incidence), the mask is
scalar and the TCC is indexed by the orders o. Otherwise (e.g. a boundary
layer with different TE and TM transmissions) each used element (j, k) of
the Jones matrices (j: pupil component, k: incident component) is a
"channel" c, and the TCC is indexed by (o, c).

The TCC is Hermitian and positive semidefinite, so its eigenvectors phi_k
and eigenvalues lambda_k >= 0 give the SOCS decomposition

    I(r) = sum_k lambda_k |sum_o a_o phi_k[o] exp(i*f_o.r)|^2,

a sum of coherent images through the kernels phi_k. Few kernels usually
carry almost all of the energy (the sum of the eigenvalues), so truncating
the sum makes imaging very fast: the basis of fast OPC and source mask
optimization.

Approximations against Abbe's method: the mask Jones matrices are those at
normal incidence (s = 0). For a ThinMask, whose transmissions do not depend
on the angle of incidence, Hopkins and Abbe agree exactly (up to rounding);
for a FilmMask, the angle dependence of the absorber is dropped. Truncated
SOCS adds the energy of the dropped kernels as a further error.
"""

import numpy as np

from .imaging import _synthesize


class TCC(object):
    """Transmission cross coefficients of a lithographic system.

    Parameters
    ----------
    sim : LithoSimulation
        Illumination, projector, wafer stack and mask stack; its layout sets
        the pitches (and so the orders). Any layout with the same pitches
        can be imaged afterwards.
    foci : sequence of numbers, optional
        Foci (one TCC each), see LithoSimulation.image().
    depth : number, optional
        Depth in the resist (wafer stack only).

    Attributes
    ----------
    m, n : arrays
        Order indices along x and y.
    f_x, f_y : arrays
        Spatial angular frequencies of the orders.
    channels : list
        "scalar" for a scalar mask, otherwise the Jones matrix elements
        (j, k) of the mask that are used.
    matrix : array
        Shape (len(foci), N, N), N = len(n)*len(m)*len(channels), index
        ((n, m), c) in C order.
    """

    batch_bytes = 32*2**20

    def __init__(self, sim, foci=(0.0,), depth=0.0):
        self.sim = sim
        self.foci = np.atleast_1d(np.asarray(foci, dtype=float))
        self.depth = depth
        self.m, self.n, self.f_x, self.f_y = sim.orders()
        J = self._jones(sim)
        if all(Jr[0, 1] == 0 and Jr[1, 0] == 0 and Jr[0, 0] == Jr[1, 1] for Jr in J.values()):
            self.channels = ["scalar"]
            self._basis = np.eye(2)[None]
        else:
            self.channels = [(j, k) for j in range(2) for k in range(2)
                             if any(abs(Jr[j, k]) > 0 for Jr in J.values())]
            self._basis = np.zeros((len(self.channels), 2, 2))
            for c, (j, k) in enumerate(self.channels):
                self._basis[c, j, k] = 1.0
        self.matrix = self._compute()
        a_clear = self.mask_vector(_clear_coefficients(len(self.n), len(self.m)), {"clear": J["clear"]})
        self._open = np.einsum("i,fij,j->f", a_clear, self.matrix, a_clear.conj()).real

    @staticmethod
    def _jones(sim):
        """Region Jones matrices at normal incidence (Hopkins)."""

        J = sim._region_jones(np.zeros(1), np.zeros(1))
        return {r: np.asarray(v)[0] for r, v in J.items()}

    @property
    def shape(self):
        return len(self.n), len(self.m), len(self.channels)

    def _compute(self):
        sim = self.sim
        n_y, n_x, n_c = self.shape
        N = n_y*n_x*n_c
        weights, jones = sim._polarization_table()
        s_x_all, s_y_all = sim.illumination.s_x, sim.illumination.s_y
        n_states = weights.shape[1]
        T = np.zeros((len(self.foci), N, N), dtype=complex)
        per_point = 16*len(self.foci)*n_states*3*N*3
        batch = max(1, int(self.batch_bytes//per_point))
        for start in range(0, len(s_x_all), batch):
            sl = slice(start, start + batch)
            passed, pupil, G = sim._transfer(s_x_all[sl], s_y_all[sl], self.f_x, self.f_y,
                                             self.foci, self.depth)
            if not np.any(passed):
                continue
            # field component i of a unit order o in channel c for the
            # incident state v: pupil*(G E_c v)_i
            Ev = np.einsum("cjk,psk->pscj", self._basis, jones[sl])  # (points, states, channels, 2)
            H = np.einsum("ijpnm,pscj->psinmc", G, Ev)  # (points, states, 3, n_y, n_x, channels)
            F = pupil[:, :, None, None, :, :, None]*H[None]
            w = np.sqrt(weights[sl])  # (points, states)
            F = F*w[None, :, :, None, None, None, None]
            F = F.transpose(0, 4, 5, 6, 1, 2, 3).reshape(len(self.foci), N, -1)
            T += F @ F.conj().swapaxes(-1, -2)

        return T

    def mask_vector(self, coefficients, jones=None):
        """Mask spectrum a[(n, m), c] from region coefficients.

        Parameters
        ----------
        coefficients : dict
            Fourier coefficients of the regions on the order grid, e.g. from
            MaskLayout.region_coefficients().
        jones : dict, optional
            Jones matrices of the regions; default: those of the mask stack
            of the simulation at normal incidence.
        """

        J = self._jones(self.sim) if jones is None else jones
        n_y, n_x, n_c = self.shape
        a = np.zeros((n_y, n_x, n_c), dtype=complex)
        for r, C in coefficients.items():
            for i, c in enumerate(self.channels):
                a[..., i] += (J[r][0, 0] if c == "scalar" else J[r][c])*C

        return a.ravel()

    def layout_vector(self, layout):
        """Mask spectrum of a layout with the pitches of the TCC (tone and
        boundary layer as in LithoSimulation)."""

        L0 = self.sim.layout
        if layout.pitch_x != L0.pitch_x or layout.pitch_y != L0.pitch_y:
            raise ValueError("The layout must have the pitches of the TCC ({}, {})"
                             .format(L0.pitch_x, L0.pitch_y))
        sim = self.sim.with_(layout=layout)
        J = self._jones(sim)
        coeffs = layout.region_coefficients(self.m, self.n, sim._boundary_width())

        return self.mask_vector(coeffs, J)

    def open_frame(self):
        """Intensity of a clear mask for all foci."""

        return self._open

    def _grids(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray([0.0] if y is None else y, dtype=float))

        return np.exp(1j*np.outer(x, self.f_x)), np.exp(1j*np.outer(y, self.f_y))

    def image(self, x, y=None, layout=None, mask_vector=None, normalize=True):
        """Image for all foci.

        Parameters
        ----------
        x, y : arrays
            Image grid; y=None gives the cut at y = 0.
        layout : MaskLayout, optional
            Default: the layout of the simulation.
        mask_vector : array, optional
            A mask spectrum (e.g. from mask_vector()) instead of a layout.
        normalize : bool, optional
            Divide by the open frame.

        Returns
        -------
        array
            Shape (len(foci), len(y), len(x)), or (len(foci), len(x)) for
            y=None.
        """

        a = self._vector(layout, mask_vector)
        n_y, n_x, n_c = self.shape
        # mutual intensities of the orders, summed over the channels:
        T = self.matrix.reshape(len(self.foci), n_y*n_x, n_c, n_y*n_x, n_c)
        A = a.reshape(n_y*n_x, n_c)
        M = np.einsum("oc,fockd,kd->fok", A, T, A.conj())
        W_x, W_y = self._grids(x, y)
        I = _synthesize(M, W_x, W_y)
        if normalize:
            I = I/self._open[:, None, None]

        return I[:, 0] if y is None else I

    def _vector(self, layout, mask_vector):
        if mask_vector is not None:
            return np.asarray(mask_vector)

        return self.layout_vector(self.sim.layout if layout is None else layout)

    def socs(self, n_kernels=None, energy=None):
        """SOCS decomposition.

        Parameters
        ----------
        n_kernels : int, optional
            Number of kernels to keep (largest eigenvalues).
        energy : number, optional
            Keep the fewest kernels whose eigenvalues sum to at least this
            fraction of the total (e.g. 0.999); applied per focus, the
            largest count is used for all. Default (neither given): all
            kernels with positive eigenvalues.

        Returns
        -------
        SOCS
        """

        lam, phi = np.linalg.eigh(self.matrix)
        lam, phi = lam[:, ::-1], phi[:, :, ::-1]  # decreasing
        lam = np.clip(lam, 0.0, None)
        if n_kernels is None:
            if energy is None:
                n_kernels = int(np.max(np.sum(lam > 1e-12*lam[:, :1], axis=1)))
            else:
                frac = np.cumsum(lam, axis=1)/np.sum(lam, axis=1, keepdims=True)
                n_kernels = int(np.max(np.argmax(frac >= energy, axis=1))) + 1
        n_kernels = max(1, min(int(n_kernels), lam.shape[1]))

        return SOCS(self, lam[:, :n_kernels], phi[:, :, :n_kernels], lam.sum(axis=1))


class SOCS(object):
    """Sum of coherent systems: the leading eigenvalues and eigenvectors
    (kernels) of a TCC.

    Attributes
    ----------
    tcc : TCC
    eigenvalues : array (len(foci), n_kernels)
    kernels : array (len(foci), N, n_kernels)
    energy : array (len(foci),)
        Fraction of the TCC's trace carried by the kept kernels.
    """

    def __init__(self, tcc, eigenvalues, kernels, total):
        self.tcc = tcc
        self.eigenvalues, self.kernels = eigenvalues, kernels
        self.energy = eigenvalues.sum(axis=1)/total

    @property
    def n_kernels(self):
        return self.eigenvalues.shape[1]

    def kernel_fields(self, x, y=None, focus_index=0):
        """Coherent amplitude kernels in real space: the fields of the
        kernels for a unit mask spectrum, sum_o phi_k[o, c] exp(i f_o.r),
        summed over the channels; shape (n_kernels, len(y), len(x))."""

        n_y, n_x, n_c = self.tcc.shape
        phi = self.kernels[focus_index].reshape(n_y, n_x, n_c, -1).sum(axis=2)
        W_x, W_y = self.tcc._grids(x, y)

        return W_y @ phi.transpose(2, 0, 1) @ W_x.T

    def image(self, x, y=None, layout=None, mask_vector=None, normalize=True):
        """Image for all foci as the sum of the coherent images of the
        kernels; arguments and shape as for TCC.image()."""

        tcc = self.tcc
        a = tcc._vector(layout, mask_vector)
        n_y, n_x, n_c = tcc.shape
        W_x, W_y = tcc._grids(x, y)
        # order amplitudes of every kernel, summed over the channels:
        B = (a[None, :, None]*self.kernels).reshape(len(tcc.foci), n_y, n_x, n_c, -1).sum(axis=3)
        E = W_y @ B.transpose(0, 3, 1, 2) @ W_x.T  # (foci, kernels, len(y), len(x))
        I = np.einsum("fk,fkyx->fyx", self.eigenvalues, E.real**2 + E.imag**2)
        if normalize:
            I = I/tcc.open_frame()[:, None, None]

        return I[:, 0] if y is None else I


def _clear_coefficients(n_y, n_x):
    C = np.zeros((n_y, n_x), dtype=complex)
    C[n_y//2, n_x//2] = 1.0

    return {"clear": C}
