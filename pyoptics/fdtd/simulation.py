#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""The Simulation: structure, sources, monitors and the time stepping.

Update equations (normalized units, mu = 1), Ez mode in 2D ::

    dHx/dt = -dEz/dy,   dHy/dt = dEz/dx,
    eps dEz/dt + sigma Ez = dHy/dx - dHx/dy - Jz,

Hz mode ::

    dHz/dt = dEx/dy - dEy/dx - Mz,
    eps dEx/dt + sigma Ex = dHz/dy - Jx,
    eps dEy/dt + sigma Ey = -dHz/dx - Jy,

and in 1D (propagation along x) the Ez mode with Ez and Hy only. H is
updated from E with forward differences, E from H with backward
differences; E lives at integer, H at half-integer time steps. The
conductive term is discretized semi-implicitly,
E <- Ca E + Cb (curl H - J) with Ca = (1 - s)/(1 + s), Cb = dt/eps/(1 + s),
s = sigma dt/(2 eps).
"""

import time as _time

import numpy as np

from . import boundaries as _bnd
from .grid import E_COMPONENTS, H_COMPONENTS, OFFSETS
from .media import Medium, VACUUM


class Simulation(object):
    """FDTD simulation in 1D or 2D.

    Parameters
    ----------
    grid : Grid
    structure : sequence of (Shape, Medium)
        Drawn in order on top of the background, later ones on top.
    sources : sequence of Source
    monitors : sequence of Monitor
    mode : "Ez" or "Hz"
        2D polarization: Ez has the electric field along z (out of the
        plane, fields Ez, Hx, Hy), Hz the magnetic field along z (Ex, Ey,
        Hz). Ignored in 1D, which has Ez and Hy. See the module docstring
        of pyoptics.fdtd for the TE/TM names.
    boundaries : str, PML or sequence of them
        Per axis "pml" (or a PML instance), "pec" or "periodic"; a single
        entry applies to all axes. Default: PML() everywhere.
    background : Medium
    courant : number
        Time step as a fraction of the stability limit.
    dt : number, optional
        Explicit time step (overrides courant).
    subpixel : int, optional
        Sub-samples per cell and axis for averaging the permittivity across
        material boundaries (1: staircase). Default 64 in 1D, 8 in 2D. The
        average is arithmetic, which is the right one for E parallel to the
        boundaries (1D, and the Ez mode in 2D).
    dtype : numpy dtype
        float64 (default) or float32.
    """

    def __init__(self, grid, structure=(), sources=(), monitors=(), mode="Ez", boundaries="pml",
                 background=VACUUM, courant=0.99, dt=None, subpixel=None, dtype=np.float64):
        if grid.ndim == 1:
            mode = "Ez"
        if mode not in ("Ez", "Hz"):
            raise ValueError("mode must be 'Ez' or 'Hz'")
        self.grid = grid
        self.ndim = grid.ndim
        self.mode = mode
        self.structure = list(structure)
        self.sources = list(sources)
        self.monitors = list(monitors)
        self.boundaries = _bnd.parse(boundaries, grid.ndim)
        self.background = background
        self.dt = grid.time_step(courant) if dt is None else dt
        self.subpixel = int(subpixel) if subpixel else (64 if grid.ndim == 1 else 8)
        self.dtype = np.dtype(dtype)
        self.step_count = 0
        self.track_energy = False
        self._e_prev = None
        self._ready = False

    # ------------------------------------------------------------------ setup

    @property
    def e_components(self):
        return E_COMPONENTS[(self.ndim, self.mode)]

    @property
    def h_components(self):
        return H_COMPONENTS[(self.ndim, self.mode)]

    @property
    def components(self):
        return self.e_components + self.h_components

    @property
    def time(self):
        """Time of the electric field (H is half a step behind)."""

        return self.step_count*self.dt

    def coords(self, component):
        return self.grid.coords(component, self.mode)

    def rasterize(self, component):
        """Permittivity and conductivity at the nodes of a component,
        averaged over subpixel**ndim samples per cell."""

        grid = self.grid
        s = self.subpixel
        base = self.coords(component)
        sub = (np.arange(s) + 0.5)/s - 0.5
        eps = np.zeros(grid.shape)
        sigma = np.zeros(grid.shape)
        shifts = np.stack(np.meshgrid(*[sub]*self.ndim, indexing="ij"), -1).reshape(-1, self.ndim)
        for shift in shifts:
            pts = np.meshgrid(*[c + d*h for c, d, h in zip(base, shift, grid.spacing)], indexing="ij")
            e = np.full(grid.shape, self.background.eps)
            g = np.full(grid.shape, self.background.sigma)
            for shape, medium in self.structure:
                inside = shape.contains(*pts)
                e[inside] = medium.eps
                g[inside] = medium.sigma
            eps += e
            sigma += g
        n = len(shifts)
        return eps/n, sigma/n

    def setup(self):
        """Rasterize the structure, allocate the fields and set up sources and
        monitors (done by run() if needed)."""

        grid = self.grid
        dt = self.dt
        self.fields = {c: np.zeros(grid.shape, self.dtype) for c in self.components}
        self.eps, self.sigma = {}, {}
        self._Ca, self._Cb = {}, {}
        for c in self.e_components:
            eps, sigma = self.rasterize(c)
            s = sigma*dt/(2*eps)
            self.eps[c] = eps
            self.sigma[c] = sigma
            self._Ca[c] = ((1 - s)/(1 + s)).astype(self.dtype) if np.any(sigma) else None
            self._Cb[c] = (dt/eps/(1 + s)).astype(self.dtype)
        # PML coefficients per axis: for H updates the derivatives sit at
        # half nodes, for E updates at integer nodes
        self._pml = {}
        for a, b in enumerate(self.boundaries):
            if isinstance(b, _bnd.PML):
                if 2*b.cells >= grid.shape[a]:
                    raise ValueError("PML thicker than the domain along axis %d" % a)
                for kind, off in (("h", 0.5), ("e", 0.0)):
                    coeffs = b.coefficients(grid.shape[a], grid.spacing[a], dt, off)
                    shaped = []
                    for bb, cc, ik in coeffs:
                        shp = [1]*self.ndim
                        shp[a] = -1
                        shaped.append(tuple(v.reshape(shp).astype(self.dtype) for v in (bb, cc, ik)))
                    self._pml[(a, kind)] = (b.cells, shaped)
        self._psi = {}
        self.step_count = 0
        for s in self.sources:
            s.setup(self)
        for m in self.monitors:
            m.setup(self)
        self._ready = True

    # ---------------------------------------------------------- derivatives

    def _periodic(self, a):
        return self.boundaries[a] == _bnd.PERIODIC

    def _sl(self, a, s):
        idx = [slice(None)]*self.ndim
        idx[a] = s
        return tuple(idx)

    def _stretch(self, d, a, kind, key):
        """Apply the CPML to a derivative d along axis a (in place)."""

        if (a, kind) not in self._pml:
            return d
        n, sides = self._pml[(a, kind)]
        N = self.grid.shape[a]
        for side, (b, c, ik) in zip((0, 1), sides):
            sl = self._sl(a, slice(0, n) if side == 0 else slice(N - n, N))
            ds = d[sl]
            psi = self._psi.get((key, side))
            if psi is None:
                psi = self._psi[(key, side)] = np.zeros(ds.shape, self.dtype)
            psi *= b
            psi += c*ds
            d[sl] = ds*ik + psi
        return d

    def _dfwd(self, F, a, key):
        """Forward difference along a (result at half nodes) / spacing."""

        d = np.empty_like(F)
        d[self._sl(a, slice(0, -1))] = F[self._sl(a, slice(1, None))] - F[self._sl(a, slice(0, -1))]
        last = self._sl(a, -1)
        if self._periodic(a):
            d[last] = F[self._sl(a, 0)] - F[last]
        else:
            d[last] = -F[last]     # the tangential E on the outer wall is 0
        d /= self.dtype.type(self.grid.spacing[a])
        return self._stretch(d, a, "h", key)

    def _dbwd(self, F, a, key):
        """Backward difference along a (result at integer nodes) / spacing."""

        d = np.empty_like(F)
        d[self._sl(a, slice(1, None))] = F[self._sl(a, slice(1, None))] - F[self._sl(a, slice(0, -1))]
        first = self._sl(a, 0)
        d[first] = F[first] - F[self._sl(a, -1)]    # periodic; overwritten by the wall otherwise
        d /= self.dtype.type(self.grid.spacing[a])
        return self._stretch(d, a, "e", key)

    # ---------------------------------------------------------------- update

    def _update_h(self):
        f = self.fields
        dt = self.dtype.type(self.dt)
        if self.ndim == 1:
            f["Hy"] += dt*self._dfwd(f["Ez"], 0, "Hy_x")
        elif self.mode == "Ez":
            f["Hx"] -= dt*self._dfwd(f["Ez"], 1, "Hx_y")
            f["Hy"] += dt*self._dfwd(f["Ez"], 0, "Hy_x")
        else:
            f["Hz"] += dt*(self._dfwd(f["Ex"], 1, "Hz_y") - self._dfwd(f["Ey"], 0, "Hz_x"))
        t = self.step_count*self.dt
        for s in self.sources:
            s.inject_h(self, t)

    def _e_step(self, c, curl):
        F = self.fields[c]
        Ca = self._Ca[c]
        if Ca is not None:
            F *= Ca
        F += self._Cb[c]*curl

    def _update_e(self):
        f = self.fields
        if self.track_energy:
            self._e_prev = {c: f[c].copy() for c in self.e_components}
        if self.ndim == 1:
            self._e_step("Ez", self._dbwd(f["Hy"], 0, "Ez_x"))
        elif self.mode == "Ez":
            self._e_step("Ez", self._dbwd(f["Hy"], 0, "Ez_x") - self._dbwd(f["Hx"], 1, "Ez_y"))
        else:
            self._e_step("Ex", self._dbwd(f["Hz"], 1, "Ex_y"))
            self._e_step("Ey", -self._dbwd(f["Hz"], 0, "Ey_x"))
        t = (self.step_count + 0.5)*self.dt
        for s in self.sources:
            s.inject_e(self, t)
        # tangential E on the walls at index 0 (the walls at the far ends
        # are taken care of by the forward differences)
        for a in range(self.ndim):
            if self._periodic(a):
                continue
            for c in self.e_components:
                if OFFSETS[self.ndim][self.mode][c][a] == 0.0:
                    f[c][self._sl(a, 0)] = 0

    def _advance(self):
        self.step_count += 1

    def step(self):
        """One time step: H from n - 1/2 to n + 1/2, E from n to n + 1."""

        self._update_h()
        self._update_e()
        self._advance()
        for m in self.monitors:
            m.record(self)

    def run(self, time=None, steps=None, callback=None, every=None, progress=False):
        """Advance the simulation by a time span or a number of steps.

        Parameters
        ----------
        time : number, optional
            Simulated time span (rounded up to whole steps).
        steps : int, optional
        callback : callable, optional
            callback(sim), called every `every` steps (e.g. live plots).
        every : int, optional
        progress : bool
            Print the number of steps and the run time at the end.
        """

        if not self._ready:
            self.setup()
        if steps is None:
            if time is None:
                raise ValueError("give time or steps")
            steps = int(np.ceil(time/self.dt - 1e-9))
        t0 = _time.perf_counter()
        for n in range(steps):
            self.step()
            if callback is not None and every and (n + 1) % every == 0:
                callback(self)
        if progress:
            dt = _time.perf_counter() - t0
            cells = np.prod(self.grid.shape)
            print("%d steps in %.2f s (%.1f Mcell updates/s)" % (steps, dt, steps*cells/dt/1e6))
        return self

    # ------------------------------------------------------------- analysis

    def energy(self):
        """Electromagnetic energy 1/2 sum(eps E**2 + H**2) dV.

        E and H are half a step apart, so this oscillates slightly around
        the true energy. With track_energy = True (set before stepping), the
        E term is eps E(n)*E(n + 1) instead, which makes it the quantity that
        the Yee scheme conserves exactly in lossless media inside PEC walls.
        """

        if not self._ready:
            return 0.0
        dV = np.prod(self.grid.spacing)
        if self._e_prev is not None and self.track_energy:
            w = sum(np.sum(self.eps[c]*self.fields[c]*self._e_prev[c].astype(float))
                    for c in self.e_components)
        else:
            w = sum(np.sum(self.eps[c]*self.fields[c].astype(float)**2) for c in self.e_components)
        w += sum(np.sum(self.fields[c].astype(float)**2) for c in self.h_components)
        return 0.5*w*dV

    def permittivity(self, component=None):
        """Rasterized permittivity at the nodes of a component (default: the
        first electric component)."""

        if not self._ready:
            self.setup()
        return self.eps[component or self.e_components[0]]

    def pml_cells(self):
        """Number of PML cells per axis (0 where there is none)."""

        return tuple(b.cells if isinstance(b, _bnd.PML) else 0 for b in self.boundaries)
