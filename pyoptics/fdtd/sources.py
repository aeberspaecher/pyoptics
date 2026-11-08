#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Sources: time profiles, soft current sources and the total-field /
scattered-field (TFSF) plane wave.

Point and line (plane in 3D) sources are soft current sources: an
electric current density J (or magnetic M for Hz) is added to the update
equation, so the source does not scatter the fields passing through it. A
source's amplitude is the dipole moment (point) or the current per unit
length or area (line, plane), so the radiated field does not depend on the
cell size.

References
----------
K. R. Umashankar and A. Taflove, "A novel method to analyze electromagnetic
scattering of complex objects", IEEE Trans. Electromagn. Compat. 24, 397-405
(1982).

A. Taflove and S. C. Hagness, "Computational Electrodynamics: The
Finite-Difference Time-Domain Method", 3rd ed., Artech House (2005), chapter
5.

J. B. Schneider, "Understanding the Finite-Difference Time-Domain Method",
Washington State University (2010), www.eecs.wsu.edu/~schneidj/ufdtd,
chapter 3 (TFSF) and chapter 5 (sources).
"""

import numpy as np

from .boundaries import Bloch
from .grid import OFFSETS


class GaussianPulse(object):
    """Sine carrier under a Gaussian envelope (no DC component).

    s(t) = amplitude*exp(-(t - delay)**2/(2 tau**2))*sin(omega_0 (t - delay))

    Parameters
    ----------
    wavelength : number
        Vacuum wavelength of the carrier.
    bandwidth : number
        Spectral standard deviation relative to the carrier frequency;
        tau = 1/(2 pi bandwidth f_0).
    delay : number, optional
        Centre of the pulse, 5 tau by default (the pulse starts smoothly).
    """

    def __init__(self, wavelength, bandwidth=0.2, delay=None, amplitude=1.0):
        self.wavelength = wavelength
        self.frequency = 1.0/wavelength
        self.bandwidth = bandwidth
        self.tau = 1.0/(2*np.pi*bandwidth*self.frequency)
        self.delay = 5*self.tau if delay is None else delay
        self.amplitude = amplitude

    def __call__(self, t):
        t = t - self.delay
        return self.amplitude*np.exp(-t**2/(2*self.tau**2))*np.sin(2*np.pi*self.frequency*t)

    @property
    def end(self):
        """Time after which the pulse is negligible (< 4e-6 of its peak)."""

        return self.delay + 5*self.tau

    def __repr__(self):
        return "GaussianPulse(wavelength=%g, bandwidth=%g)" % (self.wavelength, self.bandwidth)


class ContinuousWave(object):
    """sin(omega t), switched on smoothly over a ramp time (raised cosine).

    Parameters
    ----------
    wavelength : number
    ramp : number, optional
        Rise time, 3 periods by default.
    """

    def __init__(self, wavelength, ramp=None, amplitude=1.0):
        self.wavelength = wavelength
        self.frequency = 1.0/wavelength
        self.ramp = 3*wavelength if ramp is None else ramp
        self.amplitude = amplitude
        self.end = np.inf

    def __call__(self, t):
        env = np.where(t < self.ramp, 0.5*(1 - np.cos(np.pi*np.clip(t, 0, None)/self.ramp)), 1.0)
        return self.amplitude*env*np.sin(2*np.pi*self.frequency*t)

    def __repr__(self):
        return "ContinuousWave(wavelength=%g)" % self.wavelength


class Source(object):
    """Base class. Subclasses implement setup(sim), inject_h(sim, t) and
    inject_e(sim, t)."""

    def setup(self, sim):
        pass

    def inject_h(self, sim, t):
        pass

    def inject_e(self, sim, t):
        pass


def _default_component(sim):
    return sim.default_component


class _CurrentSource(Source):

    def _apply(self, sim, value, electric):
        F = sim.fields[self.component]
        if electric:
            F[self._idx] -= sim._Cb[self.component][self._idx]*self._weight*value
        else:
            F[self._idx] -= sim.dt*self._weight*value

    def inject_e(self, sim, t):
        if self.component[0] == "E":
            self._apply(sim, self.profile(t), True)

    def inject_h(self, sim, t):
        if self.component[0] == "H":
            self._apply(sim, self.profile(t), False)


class PointSource(_CurrentSource):
    """Point dipole (a line current along z in 2D, a sheet in 1D, a point
    dipole in 3D).

    Parameters
    ----------
    position : number or tuple
    profile : callable
        Time profile, e.g. GaussianPulse or ContinuousWave.
    component : str, optional
        Field component the current drives: Ez (default in 1D, the Ez mode
        and 3D), Ex or Ey (electric), Hz (magnetic current, default in the
        Hz mode); in 3D any of the six.
    amplitude : number
    """

    def __init__(self, position, profile, component=None, amplitude=1.0):
        self.position = position
        self.profile = profile
        self.component = component
        self.amplitude = amplitude

    def setup(self, sim):
        if self.component is None:
            self.component = _default_component(sim)
        self._idx = sim.grid.index(self.position, self.component, sim.mode)
        self._weight = self.amplitude/np.prod(sim.grid.spacing)


class LineSource(_CurrentSource):
    """Current sheet across the domain at a position along an axis: a line
    in 2D, a plane in 3D (also available as PlaneSource), a point in 1D.
    With periodic boundaries along the line it launches plane waves to both
    sides. With a Bloch boundary along the line (Bloch(k)),
    the current carries exp(i k s) along the line and launches plane waves
    with the transverse wave number k: oblique incidence at
    sin(theta) = k/(n k0), an angle that varies across the band of a pulse.

    Parameters
    ----------
    position : number
        Coordinate of the line along `axis`.
    profile : callable
        Time profile.
    axis : int
        Normal of the line (0: a line x = position, 1: y = position; 2: a
        plane z = position in 3D).
    component : str, optional
        As for PointSource; in 3D the default is Ex (Ey for planes normal to
        x).
    extent : (lo, hi), optional
        Part of the line that radiates (default: all of it); in 3D
        ((lo, hi), (lo, hi)) along the two other axes (in the order x, y,
        z), entries may be None.
    transverse : callable, optional
        Amplitude across the line as a function of the coordinate along it,
        e.g. a Gaussian beam profile; in 3D a function of the two other
        coordinates.
    angle : number, optional
        Tilt of the launched waves (radians, from the axis towards the
        positive coordinate along the line): each point is delayed by
        n*sin(angle)*(s - center), which launches plane waves (or a tilted
        beam with `transverse`) at that angle in a medium of index n. 2D
        only.
    n : number
        Refractive index at the line (for `angle`).
    center : number
        Coordinate along the line where the delay is zero.
    amplitude : number
    """

    def __init__(self, position, profile, axis=0, component=None, extent=None, transverse=None,
                 angle=0.0, n=1.0, center=0.0, amplitude=1.0):
        self.position = position
        self.profile = profile
        self.axis = axis
        self.component = component
        self.extent = extent
        self.transverse = transverse
        self.angle = angle
        self.n = n
        self.center = center
        self.amplitude = amplitude
        self._delay = None

    def setup(self, sim):
        if self.component is None:
            self.component = ("Ey" if self.axis == 0 else "Ex") if sim.ndim == 3 else \
                _default_component(sim)
        if sim.ndim == 3:
            return self._setup3(sim)
        grid = sim.grid
        coords = grid.coords(self.component, sim.mode)
        pos = np.zeros(grid.ndim)
        pos[self.axis] = self.position
        i = grid.index(pos, self.component, sim.mode)[self.axis]
        weight = self.amplitude/grid.spacing[self.axis]
        if grid.ndim == 1:
            self._idx = (i,)
            self._weight = weight
            return
        other = 1 - self.axis
        t = coords[other]
        w = np.full(t.shape, weight)
        if self.extent is not None:
            w = w*((t >= self.extent[0]) & (t <= self.extent[1]))
        if self.transverse is not None:
            w = w*self.transverse(t)
        self._idx = (i, slice(None)) if self.axis == 0 else (slice(None), i)
        b = sim.boundaries[other]
        if isinstance(b, Bloch):
            # Bloch-periodic along the line: the current carries exp(i k t)
            w = w*np.exp(1j*b.k*t)
        self._weight = w
        if self.angle:
            self._delay = self.n*np.sin(self.angle)*(t - self.center)

    def _setup3(self, sim):
        grid = sim.grid
        a = self.axis
        coords = grid.coords(self.component, sim.mode)
        pos = np.zeros(3)
        pos[a] = self.position
        i = grid.index(pos, self.component, sim.mode)[a]
        others = [b for b in range(3) if b != a]
        t = np.meshgrid(coords[others[0]], coords[others[1]], indexing="ij")
        w = np.full(t[0].shape, self.amplitude/grid.spacing[a], dtype=complex)
        for q, b in enumerate(others):
            if self.extent is not None and self.extent[q] is not None:
                lo, hi = self.extent[q]
                w = w*((t[q] >= lo) & (t[q] <= hi))
            if isinstance(sim.boundaries[b], Bloch):
                w = w*np.exp(1j*sim.boundaries[b].k*t[q])
        if self.transverse is not None:
            w = w*self.transverse(*t)
        if self.angle:
            raise ValueError("LineSource(angle=...) is 2D only; use Bloch boundaries in 3D")
        if not np.iscomplexobj(sim.fields[self.component]):
            w = w.real
        idx = [slice(None)]*3
        idx[a] = i
        self._idx = tuple(idx)
        self._weight = w

    def _value(self, t):
        return self.profile(t) if self._delay is None else self.profile(t - self._delay)

    def inject_e(self, sim, t):
        if self.component[0] == "E":
            self._apply(sim, self._value(t), True)

    def inject_h(self, sim, t):
        if self.component[0] == "H":
            self._apply(sim, self._value(t), False)


class TFSF(Source):
    """Plane wave injected on the boundary of a box (total-field /
    scattered-field formulation): in 2D (both modes) travelling towards +x,
    in 3D towards +z, polarized along x or y.

    Inside the box the total field is computed, outside only the scattered
    field: without scatterers the field outside the box is zero up to
    round-off. The incident wave comes from an auxiliary 1D grid with the
    same cell size and time step, whose numerical dispersion along x is
    identical to that of the 2D grid.

    Parameters
    ----------
    box : (xmin, xmax, ymin, ymax[, zmin, zmax])
        Total field region; it should lie in the background medium near
        its edges and stay clear of the PML.
    profile : callable
        Time profile of the incident field (its E at the source of the
        auxiliary grid; the wave reaches x = xmin a bit later).
    n_background : number
        Refractive index of the background medium.
    polarization : "x" or "y"
        Direction of E in 3D (H along y or -x).
    wavelengths : array, optional
        Record the DFT of the incident field at these vacuum wavelengths
        (convention as DFTField), for incident_phasor(): the reference that
        turns DFT fields of a pulsed run into fields per unit incident
        amplitude.
    """

    def __init__(self, box, profile, n_background=1.0, polarization="x", wavelengths=None):
        self.box = box
        self.profile = profile
        self.n_background = n_background
        self.polarization = polarization
        self.wavelengths = None if wavelengths is None else \
            np.atleast_1d(np.asarray(wavelengths, dtype=float))

    def _aux(self, sim, h, n_box):
        """The auxiliary 1D grid; returns the aux index of the box start."""

        from .simulation import Simulation
        from .grid import Grid
        from .media import Medium
        from .boundaries import PML
        pml = 20
        k0 = pml + 8    # auxiliary index of the first main index in the box
        n_aux = k0 + n_box + 4 + pml
        aux_grid = Grid(n_aux*h, h, center=n_aux*h/2)
        self.aux = Simulation(aux_grid, boundaries=PML(pml), courant=None, dt=sim.dt,
                              background=Medium(self.n_background**2), dtype=sim.dtype,
                              sources=[PointSource(aux_grid.axis(0)[pml + 4], self.profile)])
        self.aux.setup()
        if self.wavelengths is not None:
            self._omega = 2*np.pi/self.wavelengths
            self._acc = np.zeros((len(self._omega), n_aux), complex)
        return k0

    def _record(self):
        if self.wavelengths is not None:
            ph = np.exp(1j*self._omega*self.aux.time)*self.aux.dt
            self._acc += np.outer(ph, self.aux.fields["Ez"])

    def incident_phasor(self, position):
        """DFT of the incident E (Ez in the Ez mode, Ey in the Hz mode, the
        polarization direction in 3D) at the node nearest to a position
        along the propagation axis, per wavelength (needs wavelengths)."""

        if self.wavelengths is None:
            raise ValueError("TFSF(wavelengths=...) records the incident field")
        return self._acc[:, self._aux_index(position)]

    def _aux_index(self, position):
        if hasattr(self, "z0"):
            return int(round((position - self.z0)/self.aux.grid.spacing[0])) + self.koff + \
                self.box_indices[4]
        return int(round((position - self.x0)/self.aux.grid.spacing[0])) + self.koff + self.i0

    def setup(self, sim):
        if sim.ndim == 3:
            return self._setup3(sim)
        if sim.ndim != 2:
            raise ValueError("TFSF needs a 2D or 3D simulation")
        grid = sim.grid
        ex = grid.axis(0)
        xmin, xmax, ymin, ymax = self.box
        self.i0 = int(np.ceil((xmin - grid.origin[0])/grid.spacing[0] - 1e-9))
        self.i1 = int(np.floor((xmax - grid.origin[0])/grid.spacing[0] + 1e-9))
        self.j0 = int(np.ceil((ymin - grid.origin[1])/grid.spacing[1] - 1e-9))
        self.j1 = int(np.floor((ymax - grid.origin[1])/grid.spacing[1] + 1e-9))
        if not (0 < self.i0 < self.i1 < grid.shape[0] - 1 and 0 < self.j0 < self.j1 < grid.shape[1] - 1):
            raise ValueError("TFSF box must lie inside the domain")
        k0 = self._aux(sim, grid.spacing[0], self.i1 - self.i0)
        self.koff = k0 - self.i0     # aux index = main x index + koff
        self.x0 = ex[self.i0]
        self.box_indices = (self.i0, self.i1, self.j0, self.j1)

    def _setup3(self, sim):
        """Plan of the corrections: for every curl term that couples an
        incident component across the box surface, the faces where the
        updated node and its neighbour lie on different sides."""

        grid = sim.grid
        lo = [int(np.ceil((self.box[2*a] - grid.origin[a])/grid.spacing[a] - 1e-9)) for a in range(3)]
        hi = [int(np.floor((self.box[2*a + 1] - grid.origin[a])/grid.spacing[a] + 1e-9)) for a in range(3)]
        if not all(0 < lo[a] < hi[a] < grid.shape[a] - 1 for a in range(3)):
            raise ValueError("TFSF box must lie inside the domain")
        self.box_indices = tuple(v for a in range(3) for v in (lo[a], hi[a]))
        self.koff = self._aux(sim, grid.spacing[2], hi[2] - lo[2]) - lo[2]
        self.z0 = grid.axis(2)[lo[2]]
        # incident components (sign relative to the aux Ez, Hy) and the
        # curl terms (target, source, axis, sign) that involve them
        if self.polarization == "x":
            inc = {"Ex": 1.0, "Hy": -1.0}
            terms = [("Hy", "Ex", 2, -1), ("Hz", "Ex", 1, 1), ("Ex", "Hy", 2, -1), ("Ez", "Hy", 0, 1)]
        elif self.polarization == "y":
            inc = {"Ey": 1.0, "Hx": 1.0}
            terms = [("Hx", "Ey", 2, 1), ("Hz", "Ey", 0, -1), ("Ey", "Hx", 2, 1), ("Ez", "Hx", 1, -1)]
        else:
            raise ValueError("polarization must be 'x' or 'y'")
        off = OFFSETS[3]["3D"]

        def inside(pos, a):
            return (pos >= lo[a] - 1e-9) & (pos <= hi[a] + 1e-9)

        self._plan = {"h": [], "e": []}
        for T, S, a, sgn in terms:
            kind = T[0].lower()
            oT = off[T]
            sl = []
            for b in range(3):
                pos = np.arange(grid.shape[b]) + oT[b]
                sel = np.nonzero(inside(pos, b))[0]
                sl.append(slice(sel[0], sel[-1] + 1))
            pos = np.arange(grid.shape[a]) + oT[a]
            inT = inside(pos, a).astype(int)
            sp = inT - inside(pos + 0.5, a)
            sm = inT - inside(pos - 0.5, a)
            for i in np.nonzero((sp != 0) | (sm != 0))[0]:
                idx = list(sl)
                idx[a] = i
                plus = i + 1 if kind == "h" else i       # index of the S node at +h/2
                coef = sgn/grid.spacing[a]
                self._plan[kind].append((T, S, inc[S], a, tuple(idx), int(sp[i]), int(sm[i]), plus,
                                         coef, sl[2]))

    def _apply3(self, sim, kind):
        aux = {"E": self.aux.fields["Ez"], "H": self.aux.fields["Hy"]}
        k = self.koff
        for T, S, sS, a, idx, sp, sm, plus, coef, zsl in self._plan[kind]:
            F = aux[S[0]]
            if a == 2:
                val = sp*F[plus + k] - sm*F[plus - 1 + k]
            else:   # the incident value at the node's own z
                val = (sp - sm)*F[zsl.start + k:zsl.stop + k]
            c = sim.dt*coef if kind == "h" else sim._Cb[T][idx]*coef
            sim.fields[T][idx] += (c*sS*val).astype(sim.dtype)

    def inject_h(self, sim, t):
        # called after the H update of the main grid (aux E at the same time)
        if sim.ndim == 3:
            self._apply3(sim, "h")
            self.aux._update_h()
            return
        Ez1 = self.aux.fields["Ez"]
        i0, i1, j0, j1 = self.box_indices
        k = self.koff
        dt = sim.dt
        dx, dy = sim.grid.spacing
        if sim.mode == "Ez":
            Hx, Hy = sim.fields["Hx"], sim.fields["Hy"]
            Hy[i0 - 1, j0:j1 + 1] -= dt/dx*Ez1[i0 + k]
            Hy[i1, j0:j1 + 1] += dt/dx*Ez1[i1 + k]
            Hx[i0:i1 + 1, j0 - 1] += dt/dy*Ez1[i0 + k:i1 + 1 + k]
            Hx[i0:i1 + 1, j1] -= dt/dy*Ez1[i0 + k:i1 + 1 + k]
        else:
            Hz = sim.fields["Hz"]
            Hz[i0 - 1, j0:j1] += dt/dx*Ez1[i0 + k]
            Hz[i1, j0:j1] -= dt/dx*Ez1[i1 + k]
        self.aux._update_h()

    def inject_e(self, sim, t):
        # called after the E update of the main grid (aux H half a step ahead)
        if sim.ndim == 3:
            self._apply3(sim, "e")
            self.aux._update_e()
            self.aux._advance()
            self._record()
            return
        Hy1 = self.aux.fields["Hy"]
        i0, i1, j0, j1 = self.box_indices
        k = self.koff
        dx, dy = sim.grid.spacing
        if sim.mode == "Ez":
            Cb = sim._Cb["Ez"]
            Ez = sim.fields["Ez"]
            Ez[i0, j0:j1 + 1] -= Cb[i0, j0:j1 + 1]/dx*Hy1[i0 - 1 + k]
            Ez[i1, j0:j1 + 1] += Cb[i1, j0:j1 + 1]/dx*Hy1[i1 + k]
        else:
            Ex, Ey = sim.fields["Ex"], sim.fields["Ey"]
            CbX, CbY = sim._Cb["Ex"], sim._Cb["Ey"]
            Hz_inc_l = -Hy1[i0 - 1 + k]
            Hz_inc_r = -Hy1[i1 + k]
            Ey[i0, j0:j1] += CbY[i0, j0:j1]/dx*Hz_inc_l
            Ey[i1, j0:j1] -= CbY[i1, j0:j1]/dx*Hz_inc_r
            Hz_inc = -Hy1[i0 + k:i1 + k]      # at x half nodes i0..i1-1
            Ex[i0:i1, j0] -= CbX[i0:i1, j0]/dy*Hz_inc
            Ex[i0:i1, j1] += CbX[i0:i1, j1]/dy*Hz_inc
        self.aux._update_e()
        self.aux._advance()
        self._record()

    def incident(self, x, t=None):
        """Incident Ez (Ez mode) or Ey (Hz mode) of the auxiliary grid at x,
        or in 3D the incident E (along the polarization) at z = x, at the
        current time of the simulation (nearest node)."""

        k = self._aux_index(x)
        return self.aux.fields["Ez"][k]


PlaneSource = LineSource
"""A LineSource in 3D: a current sheet on a plane."""
