#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Rigorous mask diffraction with the FDTD (pyoptics.fdtd), for 1D and 2D
layouts.

FDTDMask is a mask stack for LithoSimulation and hopkins.TCC with the same
description and interface as rcwa.RigorousMask, but it solves Maxwell's
equations on a grid instead of with Fourier modes. That makes it slower
than RCWA for lines and spaces, but it handles what the RCWA of
pyoptics.litho.rcwa does not: 2D layouts (contact holes, line ends,
any Manhattan pattern of MaskLayout features).

Set-up (mask scale, lengths in the unit of the wavelength, z along the
optical axis): one period of the layout (pitch times the reduction ratio)
with Bloch-periodic boundaries in x and y, the mask blank below z = 0, the
absorber layers from z = 0 to their total thickness, the exit medium above,
PML at both ends in z. A plane current sheet in the blank launches a
pulsed plane wave with the transverse wave vector of the source point
(k_t = k0 NA s/M), once for each of two polarizations, and a
DiffractionMonitor in the exit medium expands the transmitted field into
the orders (m, n). An invariant direction of a 1D layout is one cell wide
(the run is then as cheap as a 2D one). Each order is moved back to the
bottom of the absorber with its discrete propagation constant, and with
normalize=True the result is divided by the transmission of the unpatterned
stack of clear materials (a one-cell run at the same angle), as RigorousMask
does: the open frame transmits 1, and the source and the grid dispersion
drop out.

Absorbers with Re(eps) < 1 (chrome at 193 nm) are Drude media that are
exact at the wavelength (fdtd.Medium.from_index).

Accuracy is set by the cell size: with the default of wavelength/40 the
orders of chrome and MoSi lines agree with RCWA to better than 1e-2. The
cell size is adjusted (by up to 25 %, never coarser) so that the feature
edges and the absorber interfaces lie on grid nodes: the subpixel
averaging of a Drude metal is poor, and edges between nodes cost several
1e-2 for chrome in TM.

References
----------
A. K. Wong and A. R. Neureuther, "Mask topography effects in projection
printing of phase-shifting masks", IEEE Trans. Electron Devices 41, 895-902
(1994).

A. Erdmann, "Optical and EUV Lithography: A Modeling Perspective", SPIE
Press (2021).

A. Taflove and S. C. Hagness, "Computational Electrodynamics: The
Finite-Difference Time-Domain Method", 3rd ed., Artech House (2005), chapter
13.

M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of
planar-grating diffraction", J. Opt. Soc. Am. 71, 811-818 (1981).
"""

import numpy as np

from .rcwa import RigorousMask


class _LayoutRegion(object):
    """Absorber (or clear) region of a periodic layout in a slab of z, at
    mask scale: contains(x, y, z) for the FDTD."""

    def __init__(self, layout, magnification, z0, z1, absorber=True):
        self.layout, self.M = layout, magnification
        self.z0, self.z1 = z0, z1
        self.absorber = absorber

    def contains(self, x, y, z):
        lay, M = self.layout, self.M
        x, y, z = np.broadcast_arrays(x, y, z)
        inside = np.zeros(x.shape, dtype=bool)
        for x0, y0, a, b in lay.features:
            in_x = True if lay.pitch_x is None else (
                np.abs((x/M - x0 + lay.pitch_x/2) % lay.pitch_x - lay.pitch_x/2) < a/2)
            in_y = True if lay.pitch_y is None else (
                np.abs((y/M - y0 + lay.pitch_y/2) % lay.pitch_y - lay.pitch_y/2) < b/2)
            inside |= in_x & in_y
        feature_absorbs = lay.tone == "clear"
        region = inside if feature_absorbs == self.absorber else ~inside
        return region & (z >= self.z0) & (z < self.z1)


def _cells(L, edges, h):
    """Number of cells across a period L near L/h (not fewer) that puts the
    feature edges as close to grid nodes as possible: the subpixel
    averaging of a metal edge (a Drude medium) is poor, and an edge on a
    node avoids it. Up to 25 % more cells are allowed."""

    n0 = max(1, int(round(L/h)))
    edges = np.asarray(edges, dtype=float)
    if len(edges) == 0:
        return n0
    best, best_cost = n0, np.inf
    for n in range(n0, int(np.ceil(1.25*n0)) + 1):
        f = ((edges + L/2)/(L/n)) % 1.0
        cost = np.max(np.minimum(f, 1 - f))
        if cost < best_cost - 0.02:          # prefer fewer cells unless clearly better
            best, best_cost = n, cost
        if best_cost < 0.02:
            break
    return best


class FDTDMask(RigorousMask):
    """Mask stack computed rigorously with the FDTD, for 1D and 2D layouts.

    Parameters
    ----------
    layers : sequence of (n_absorber, thickness) or (n_absorber, n_clear, thickness)
        Complex refractive indices and thicknesses (mask scale) from the
        blank outwards, as for RigorousMask.
    substrate_n : number, optional
        Mask blank (fused silica at 193 nm: about 1.563).
    exit_n : number, optional
        Medium between mask and projection lens.
    spacing : number, optional
        FDTD cell size at mask scale; default wavelength/40.
    incidence : "per_point" or "normal"
        "per_point" runs the FDTD for every source point (Abbe, as
        RigorousMask); "normal" uses the orders at normal incidence for all
        of them (the Hopkins approximation of the mask, two runs in total).
    time : number, optional
        Length of each run in periods of the wavelength (default 40).
    normalize : bool, optional
        Normalize to the unpatterned stack of clear materials (the open
        frame transmits 1).
    backend : str, optional
        FDTD kernels ("auto", "numpy", "numba").
    name : str, optional

    Notes
    -----
    Each source point costs four runs: two polarizations through the
    patterned cell and two through a one-cell column of the clear stack
    (negligible). The results of the last calls are cached, as for
    RigorousMask; `runs` counts the patterned runs so far.
    """

    def __init__(self, layers, substrate_n=1.563, exit_n=1.0, spacing=None, incidence="per_point",
                 time=40.0, normalize=True, backend="auto", name=None):
        super(FDTDMask, self).__init__(layers, substrate_n, exit_n, normalize=normalize,
                                       name=name or "FDTD mask")
        if incidence not in ("per_point", "normal"):
            raise ValueError("incidence must be 'per_point' or 'normal'")
        self.spacing, self.incidence, self.time = spacing, incidence, time
        self.backend = backend
        self.runs = 0
        self.pml_cells = 12

    def order_jones(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the diffraction orders of a layout (1D or 2D).

        Parameters
        ----------
        layout : MaskLayout
        m, n : int or arrays
            Order indices in x and y.
        s_x, s_y : numbers or arrays
            Source point coordinates (sigma).
        NA : number
            Numerical aperture (image side).
        wavelength : number
            Vacuum wavelength.
        magnification : number
            Reduction ratio of the projector.

        Returns
        -------
        J : array
            Shape (len(s_x), 2, 2, len(n), len(m)): the tangential field
            (E_x, E_y) of each order at the bottom of the absorber per unit
            incident tangential field, relative to the clear stack.
        """

        s_x, s_y = np.atleast_1d(np.asarray(s_x, dtype=float)), np.atleast_1d(np.asarray(s_y, dtype=float))
        m, n = np.atleast_1d(m), np.atleast_1d(n)
        key = (layout.pitch_x, layout.pitch_y, tuple(map(tuple, layout.features)), layout.tone,
               tuple(m), tuple(n), s_x.tobytes(), s_y.tobytes(), NA, wavelength, magnification,
               self.incidence)
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            if self.incidence == "normal":
                J0 = self._point(layout, m, n, 0.0, 0.0, NA, wavelength, magnification)
                J = np.broadcast_to(J0, (len(s_x),) + J0.shape).copy()
            else:
                J = np.array([self._point(layout, m, n, a, b, NA, wavelength, magnification)
                              for a, b in zip(s_x, s_y)])
            self._cache[key] = J
        return self._cache[key]

    # the FDTD runs:

    def _geometry(self, wavelength):
        h = self.spacing or wavelength/40.0
        T = self.thickness
        z_src = -0.35*wavelength                 # current sheet in the blank
        z_mon = T + 0.25*wavelength              # orders in the exit medium
        z_lo = z_src - 3*h
        z_hi = z_mon + 3*h
        return h, T, z_src, z_mon, z_lo, z_hi

    def _structure(self, layout, magnification, patterned, wavelength):
        from .. import fdtd
        med = lambda nn: fdtd.Medium.from_index(np.real(nn), np.imag(nn), wavelength)
        st = [(fdtd.Slab(-1e9, 0.0, axis=2), med(self.substrate_n))]
        z = 0.0
        for n_abs, n_clear, t in self.layers:
            if n_clear != self.exit_n:
                st.append((fdtd.Slab(z, z + t, axis=2), med(n_clear)))
            if patterned:
                st.append((_LayoutRegion(layout, magnification, z, z + t), med(n_abs)))
            z += t
        return st

    def simulation(self, layout, NA, wavelength, magnification, s_x=0.0, s_y=0.0, component="Ex",
                   patterned=True, monitors=(), max_order=10):
        """The FDTD simulation of one run, set up but not run (e.g. to plot
        the cell or to record near fields with more monitors).

        Parameters
        ----------
        layout : MaskLayout
        NA, wavelength, magnification : numbers
            As for order_jones().
        s_x, s_y : numbers, optional
            Source point (sigma).
        component : "Ex" or "Ey"
            Current of the plane source (the polarization of the incident
            wave for normal incidence).
        patterned : bool
            False: the unpatterned stack of clear materials, one cell wide.
        monitors : sequence, optional
            Additional monitors.
        max_order : int
            Highest order of the DiffractionMonitor (the first monitor).

        Returns
        -------
        sim : fdtd.Simulation
        """

        from .. import fdtd
        k0 = 2*np.pi/wavelength
        k_t = (k0*NA*s_x/magnification, k0*NA*s_y/magnification)
        h, T, z_src, z_mon, z_lo, z_hi = self._geometry(wavelength)
        P = self.pml_cells
        L, d = [], []
        feats = np.asarray(layout.features, dtype=float).reshape(-1, 4)
        for i, pitch in enumerate((layout.pitch_x, layout.pitch_y)):
            if not patterned or pitch is None:
                L.append(h)
                d.append(h)
            else:
                L.append(magnification*pitch)
                edges = magnification*np.concatenate([feats[:, i] - feats[:, i + 2]/2,
                                                      feats[:, i] + feats[:, i + 2]/2])
                d.append(L[-1]/_cells(L[-1], edges, h))
        # along z: the blank/absorber interface and the top of the absorber on nodes
        hz = T/max(1, round(T/h)) if T > 0 else h
        lo, hi = np.floor(z_lo/hz - P)*hz, np.ceil(z_hi/hz + P)*hz
        grid = fdtd.Grid((L[0], L[1], hi - lo), (d[0], d[1], hz), center=(0.0, 0.0, (lo + hi)/2))
        src = fdtd.PlaneSource(z_src, fdtd.GaussianPulse(wavelength, 0.2), axis=2, component=component)
        mon = fdtd.DiffractionMonitor(z_mon, [wavelength], axis=2, max_order=max_order)
        return fdtd.Simulation(grid, self._structure(layout, magnification, patterned, wavelength),
                               [src], [mon] + list(monitors),
                               boundaries=(fdtd.Bloch(k_t[0]), fdtd.Bloch(k_t[1]), fdtd.PML(P)),
                               background=fdtd.Medium.from_index(np.real(self.exit_n)),
                               backend=self.backend, dtype=np.complex64)

    def _run(self, layout, magnification, NA, s, component, patterned, wavelength, max_order):
        """One FDTD run; returns the DiffractionOrders of the transmitted
        field and the distance from the bottom of the absorber to the
        monitor."""

        sim = self.simulation(layout, NA, wavelength, magnification, s[0], s[1], component, patterned,
                              max_order=max_order)
        sim.run(time=self.time*wavelength)
        if patterned:
            self.runs += 1
        mon = sim.monitors[0]
        z_mon = self._geometry(wavelength)[3]
        z_node = sim.coords("Ex")[2][sim.grid.index((0, 0, z_mon), "Ex", sim.mode)[2]]
        return mon.orders(), z_node - self.thickness

    def _point(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        """Order Jones matrices (2, 2, len(n), len(m)) for one source point."""

        max_order = int(max(np.abs(m).max(), np.abs(n).max()))
        out = np.zeros((2, 2, len(n), len(m)), dtype=complex)
        T = np.zeros((2, 2) + out.shape[2:], dtype=complex)
        R = np.zeros((2, 2), dtype=complex)
        for j, comp in enumerate(("Ex", "Ey")):
            o, d = self._run(layout, magnification, NA, (s_x, s_y), comp, True, wavelength, max_order)
            amp = o.amplitude[0]*np.exp(-1j*o.ka[0]*d)        # back to the absorber, (2, N, M)
            amp = np.where(o.propagating[0][None], amp, 0.0)
            for a_, mm in enumerate(m):
                for b_, nn in enumerate(n):
                    if mm in o.m and nn in o.n:
                        i_m, i_n = int(np.nonzero(o.m == mm)[0][0]), int(np.nonzero(o.n == nn)[0][0])
                        T[:, j, b_, a_] = amp[:, i_n, i_m]
            if self.normalize:
                o_r, d_r = self._run(layout, magnification, NA, (s_x, s_y), comp, False, wavelength, 0)
                R[:, j] = o_r.amplitude[0][:, 0, 0]*np.exp(-1j*o_r.ka[0][0, 0]*d_r)
        if not self.normalize:
            raise NotImplementedError("FDTDMask needs normalize=True (the source amplitude drops out "
                                      "only relative to the clear stack)")
        Rinv = np.linalg.inv(R)
        return np.einsum("ijnm,jk->iknm", T, Rinv)


__all__ = ["FDTDMask"]
