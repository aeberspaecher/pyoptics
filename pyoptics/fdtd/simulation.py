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
from .media import VACUUM


def stability_omega(medium):
    """Highest natural frequency of the polarization of a medium, which
    lowers the stable time step: omega_p**2/eps for a Drude pole,
    omega_0**2*(1 + delta_eps/eps) for a Lorentz pole (squares summed over
    the poles, eps the background permittivity eps_inf); Debye poles and
    conductivity do not count."""

    from .media import Drude, Lorentz
    eps = max(medium.eps, 1e-12)
    om2 = 0.0
    for p in medium.poles:
        if isinstance(p, Drude):
            om2 += p.omega_p**2/eps
        elif isinstance(p, Lorentz):
            om2 += p.omega_0**2*(1 + p.delta_eps/eps)
    return np.sqrt(om2)


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
    boundaries : str, PML, Bloch or sequence of them
        Per axis "pml" (or a PML instance), "pec", "periodic" or Bloch(k)
        (Bloch-periodic with wave number k; makes the fields complex); a
        single entry applies to all axes. Default: PML() everywhere.
    background : Medium
    courant : number
        Time step as a fraction of the stability limit. With Drude and
        Lorentz media the limit is lower than in vacuum: the scheme is
        stable for (dt/dt0)**2 + (Omega dt/2)**2 <= 1, with dt0 the vacuum
        limit and Omega**2 = omega_p**2/eps (Drude) or
        omega_0**2 (1 + delta_eps/eps) (Lorentz), summed over the poles of
        the medium (stability_omega()); the default time step includes it.
    dt : number, optional
        Explicit time step (overrides courant).
    subpixel : int, optional
        Sub-samples per cell and axis for averaging the permittivity across
        material boundaries (1: staircase). Default 64 in 1D, 8 in 2D.
    averaging : "anisotropic" or "arithmetic"
        How the in-plane E components of the Hz mode see material
        boundaries (see rasterize()); the Ez mode and 1D are unaffected.
    backend : "auto", "numpy" or "numba"
        Update kernels; "auto" uses numba if it is installed
        (``pip install pyoptics[fast]``), which is 10 to 20 times faster.
        NumPy is the reference implementation.
    dtype : numpy dtype
        float64 (default) or float32 (precision of the coefficients and
        fields); complex128 or complex64 make the fields complex, which
        Bloch boundaries with k != 0 do automatically.
    """

    def __init__(self, grid, structure=(), sources=(), monitors=(), mode="Ez", boundaries="pml",
                 background=VACUUM, courant=0.99, dt=None, subpixel=None, averaging="anisotropic",
                 backend="auto", dtype=np.float64):
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
        if dt is None:
            Om2 = max(stability_omega(m)**2 for m in [background] + [m for _, m in self.structure])
            dt = courant/np.sqrt(np.sum(1.0/grid.spacing**2) + Om2/4)
        self.dt = dt
        self.subpixel = int(subpixel) if subpixel else (64 if grid.ndim == 1 else 8)
        self.averaging = averaging
        self.backend = backend
        self.rdtype = np.empty(0, dtype).real.dtype       # coefficients
        bloch = any(isinstance(b, _bnd.Bloch) for b in self.boundaries)
        if bloch or np.dtype(dtype).kind == "c":
            self.dtype = np.result_type(self.rdtype, np.complex64)  # fields
        else:
            self.dtype = self.rdtype
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

    def _media(self):
        """Distinct media: the background first, then the structure's."""

        media = [self.background]
        for _, m in self.structure:
            if not any(m is x for x in media):
                media.append(m)
        return media

    def rasterize(self, component):
        """Effective permittivity, conductivity and filling fractions of the
        media at the nodes of a component, from subpixel**ndim samples per
        cell.

        Returns
        -------
        eps, sigma : arrays
            For E parallel to the material boundaries the arithmetic mean of
            eps is right, for E normal to them the harmonic mean. With
            averaging="anisotropic", the in-plane components of the Hz mode
            get 1/eps = n_a**2 <1/eps> + (1 - n_a**2)/<eps>, with the
            component n_a of the boundary normal estimated from the first
            moment of eps over the samples (Farjadpour et al., diagonal
            approximation). E along z is always parallel to the boundaries.
        fractions : array (n_media, ...)
            Filling fraction of each medium of _media().
        """

        grid = self.grid
        s = self.subpixel
        base = self.coords(component)
        sub = (np.arange(s) + 0.5)/s - 0.5
        media = self._media()
        eps_of = np.array([m.eps for m in media])
        frac = np.zeros((len(media),) + grid.shape)
        inv = np.zeros(grid.shape)
        moment = np.zeros((self.ndim,) + grid.shape)
        shifts = np.stack(np.meshgrid(*[sub]*self.ndim, indexing="ij"), -1).reshape(-1, self.ndim)
        for shift in shifts:
            pts = np.meshgrid(*[c + d*h for c, d, h in zip(base, shift, grid.spacing)], indexing="ij")
            ids = np.zeros(grid.shape, dtype=int)
            for shape, medium in self.structure:
                k = next(i for i, m in enumerate(media) if m is medium)
                ids[shape.contains(*pts)] = k
            e = eps_of[ids]
            for k in range(len(media)):
                frac[k] += ids == k
            inv += 1/e
            for a in range(self.ndim):
                moment[a] += shift[a]*e
        n = len(shifts)
        frac /= n
        inv /= n
        eps = np.tensordot(eps_of, frac, axes=1)
        sigma = np.tensordot(np.array([m.sigma for m in media]), frac, axes=1)
        axis = {"Ex": 0, "Ey": 1}.get(component) if self.ndim == 2 else None
        if self.averaging == "anisotropic" and axis is not None:
            norm2 = np.sum(moment**2, axis=0)
            has = norm2 > 1e-24*np.max(eps)**2
            na2 = np.where(has, moment[axis]**2/np.where(has, norm2, 1), 0.0)
            eps = 1/(na2*inv + (1 - na2)/eps)
        return eps, sigma, frac

    def setup(self):
        """Rasterize the structure, allocate the fields and set up sources and
        monitors (done by run() if needed)."""

        grid = self.grid
        dt = self.dt
        self.fields = {c: np.zeros(grid.shape, self.dtype) for c in self.components}
        self.eps, self.sigma = {}, {}
        self._Ca, self._Cb = {}, {}
        media = self._media()
        poles = [(k, p) for k, m in enumerate(media) for p in m.poles]
        self._pole_coeffs = np.array([p.coefficients(dt) for _, p in poles],
                                     dtype=self.rdtype).reshape(-1, 3)
        self._P, self._Pp, self._W = {}, {}, {}
        for c in self.e_components:
            eps, sigma, frac = self.rasterize(c)
            s = sigma*dt/(2*eps)
            self.eps[c] = eps
            self.sigma[c] = sigma
            self._Ca[c] = ((1 - s)/(1 + s)).astype(self.rdtype) if np.any(sigma) else None
            self._Cb[c] = (dt/eps/(1 + s)).astype(self.rdtype)
            K = len(poles)
            self._W[c] = np.array([frac[k] for k, _ in poles], dtype=self.rdtype).reshape((K,) + grid.shape)
            self._P[c] = np.zeros((K,) + grid.shape, self.dtype)
            self._Pp[c] = np.zeros((K,) + grid.shape, self.dtype)
        self._backend = self._resolve_backend()
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
                        shaped.append(tuple(v.reshape(shp).astype(self.rdtype) for v in (bb, cc, ik)))
                    self._pml[(a, kind)] = (b.cells, shaped)
        self._psi = {}
        self.step_count = 0
        if self._backend == "numba":
            self._prepare_numba()
        for s in self.sources:
            s.setup(self)
        for m in self.monitors:
            m.setup(self)
        self._ready = True

    def _resolve_backend(self):
        from . import _kernels_numba
        if self.backend == "numpy":
            return "numpy"
        if self.backend == "numba":
            if not _kernels_numba.available:
                raise ImportError("the numba backend needs numba: pip install pyoptics[fast]")
            return "numba"
        if self.backend == "auto":   # 1D grids are too small to gain from threads
            return "numba" if _kernels_numba.available and self.ndim == 2 else "numpy"
        raise ValueError("backend must be 'auto', 'numpy' or 'numba'")

    def _prepare_numba(self):
        """Full-length CPML coefficient arrays per axis and the 2D views
        (1D runs as a 2D Ez grid with one periodic cell along y)."""

        dt = self.rdtype.type
        shape2 = self.grid.shape if self.ndim == 2 else (self.grid.shape[0], 1)
        self._nb_shape = shape2
        coef = {}
        for a in range(2):
            N = shape2[a]
            b_ = self.boundaries[a] if a < self.ndim else _bnd.PERIODIC
            for kind, off in (("h", 0.5), ("e", 0.0)):
                f = np.zeros(N, dtype=np.bool_)
                bb = np.zeros(N, self.rdtype)
                cc = np.zeros(N, self.rdtype)
                ik = np.ones(N, self.rdtype)
                if isinstance(b_, _bnd.PML):
                    n = b_.cells
                    (bl, cl, kl), (br, cr, kr) = b_.coefficients(N, self.grid.spacing[a], self.dt, off)
                    for sl, (b1, c1, k1) in ((slice(0, n), (bl, cl, kl)), (slice(N - n, N), (br, cr, kr))):
                        bb[sl], cc[sl], ik[sl], f[sl] = b1, c1, k1, True
                coef[(a, kind)] = (f, bb, cc, ik)
        self._nb_coef = coef
        self._nb_periodic = tuple(self._periodic(a) if a < self.ndim else True for a in range(2))
        self._nb_spacing = tuple(dt(self.grid.spacing[a]) if a < self.ndim else dt(1.0) for a in range(2))
        need = lambda a, kind: coef[(a, kind)][0].any()
        self._nb_psi = {}
        for key, a, kind in (("h0", 0, "h"), ("h1", 1, "h"), ("e0", 0, "e"), ("e1", 1, "e")):
            self._nb_psi[key] = np.zeros(shape2 if need(a, kind) else (1, 1), self.dtype)
        self._nb_ones = {c: np.ones(shape2, self.rdtype) for c in self.e_components}
        ph = [self._phase(a) if a < self.ndim else 1.0 for a in range(2)]
        self._nb_ph = tuple(self.dtype.type(p) for p in ph)
        self._nb_cph = tuple(self.dtype.type(np.conj(p)) for p in ph)
        self._nb_dummy = np.zeros(shape2, self.dtype)

    def _v(self, a):
        """2D view of a field or per-component array for the kernels."""

        return a if self.ndim == 2 else a.reshape(self._nb_shape)

    def _vk(self, a):
        return a if self.ndim == 2 else a.reshape((a.shape[0],) + self._nb_shape)

    def _numba_h(self):
        from . import _kernels_numba as kn
        f = self.fields
        dt = self.rdtype.type(self.dt)
        dx, dy = self._nb_spacing
        px, py = self._nb_periodic
        cx, cy = self._nb_coef[(0, "h")], self._nb_coef[(1, "h")]
        psi = self._nb_psi
        if self.mode == "Ez":
            Hx = f["Hx"] if self.ndim == 2 else self._nb_dummy
            kn.h_ez(self._v(f["Ez"]), Hx, self._v(f["Hy"]), dt, dx, dy, px, py, *self._nb_ph, *cx, *cy,
                    psi["h1"], psi["h0"])
        else:
            kn.h_hz(f["Ex"], f["Ey"], f["Hz"], dt, dx, dy, px, py, *self._nb_ph, *cx, *cy, psi["h1"],
                    psi["h0"])

    def _numba_e(self):
        from . import _kernels_numba as kn
        f = self.fields
        dt = self.rdtype.type(self.dt)
        dx, dy = self._nb_spacing
        px, py = self._nb_periodic
        cx, cy = self._nb_coef[(0, "e")], self._nb_coef[(1, "e")]
        psi = self._nb_psi
        pa, pb, pc = (np.ascontiguousarray(self._pole_coeffs[:, i]) for i in range(3))
        Ca = {c: self._v(self._Ca[c]) if self._Ca[c] is not None else self._nb_ones[c]
              for c in self.e_components}
        if self.mode == "Ez":
            Hx = f["Hx"] if self.ndim == 2 else self._nb_dummy
            kn.e_ez(self._v(f["Ez"]), Hx, self._v(f["Hy"]), Ca["Ez"], self._v(self._Cb["Ez"]), dt,
                    dx, dy, px, py, *self._nb_cph, *cx, *cy, psi["e0"], psi["e1"], self._vk(self._P["Ez"]),
                    self._vk(self._Pp["Ez"]), self._vk(self._W["Ez"]), pa, pb, pc)
        else:
            kn.e_hz(f["Ex"], f["Ey"], f["Hz"], Ca["Ex"], self._Cb["Ex"], Ca["Ey"], self._Cb["Ey"], dt,
                    dx, dy, px, py, *self._nb_cph, *cx, *cy, psi["e1"], psi["e0"], self._P["Ex"], self._Pp["Ex"],
                    self._W["Ex"], self._P["Ey"], self._Pp["Ey"], self._W["Ey"], pa, pb, pc)

    # ---------------------------------------------------------- derivatives

    def _periodic(self, a):
        b = self.boundaries[a]
        return b == _bnd.PERIODIC or isinstance(b, _bnd.Bloch)

    def _phase(self, a):
        """exp(i k L) of a Bloch boundary along axis a (1 otherwise)."""

        b = self.boundaries[a]
        if isinstance(b, _bnd.Bloch):
            return np.exp(1j*b.k*self.grid.size[a])
        return 1.0

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
            d[last] = F[self._sl(a, 0)]*self.dtype.type(self._phase(a)) - F[last]
        else:
            d[last] = -F[last]     # the tangential E on the outer wall is 0
        d /= self.rdtype.type(self.grid.spacing[a])
        return self._stretch(d, a, "h", key)

    def _dbwd(self, F, a, key):
        """Backward difference along a (result at integer nodes) / spacing."""

        d = np.empty_like(F)
        d[self._sl(a, slice(1, None))] = F[self._sl(a, slice(1, None))] - F[self._sl(a, slice(0, -1))]
        first = self._sl(a, 0)
        # periodic (Bloch: F(-1) = F(N - 1) exp(-i k L)); overwritten by
        # the wall otherwise
        d[first] = F[first] - F[self._sl(a, -1)]*self.dtype.type(np.conj(self._phase(a)))
        d /= self.rdtype.type(self.grid.spacing[a])
        return self._stretch(d, a, "e", key)

    # ---------------------------------------------------------------- update

    def _update_h(self):
        f = self.fields
        dt = self.rdtype.type(self.dt)
        if self._backend == "numba":
            self._numba_h()
        elif self.ndim == 1:
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
        if len(self._pole_coeffs):
            P, Pp, W = self._P[c], self._Pp[c], self._W[c]
            dP = np.zeros_like(F)
            for k, (pa, pb, pc) in enumerate(self._pole_coeffs):
                Pn = (pa*P[k] + pb*Pp[k] + pc*W[k]*F).astype(self.dtype)
                dP += Pn - P[k]
                Pp[k] = P[k]
                P[k] = Pn
            curl = curl - dP/self.rdtype.type(self.dt)
        Ca = self._Ca[c]
        if Ca is not None:
            F *= Ca
        F += self._Cb[c]*curl

    def _update_e(self):
        f = self.fields
        if self.track_energy:
            self._e_prev = {c: f[c].copy() for c in self.e_components}
        if self._backend == "numba":
            self._numba_e()
        elif self.ndim == 1:
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
        f = {c: self.fields[c].astype(np.result_type(self.fields[c], np.float64))
             for c in self.components}
        if self._e_prev is not None and self.track_energy:
            w = sum(np.sum(self.eps[c]*np.real(f[c]*np.conj(self._e_prev[c])))
                    for c in self.e_components)
        else:
            w = sum(np.sum(self.eps[c]*np.abs(f[c])**2) for c in self.e_components)
        w += sum(np.sum(np.abs(f[c])**2) for c in self.h_components)
        return 0.5*float(w)*dV

    def permittivity(self, component=None):
        """Rasterized permittivity at the nodes of a component (default: the
        first electric component)."""

        if not self._ready:
            self.setup()
        return self.eps[component or self.e_components[0]]

    def pml_cells(self):
        """Number of PML cells per axis (0 where there is none)."""

        return tuple(b.cells if isinstance(b, _bnd.PML) else 0 for b in self.boundaries)
