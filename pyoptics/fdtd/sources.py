#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Sources: time profiles, soft current sources and the total-field /
scattered-field (TFSF) plane wave.

Point and line sources are soft current sources: an electric current
density J (or magnetic M for Hz) is added to the update equation, so the
source does not scatter the fields passing through it. A source's
amplitude is the dipole moment (point) or the current per unit length
(line), so the radiated field does not depend on the cell size.
"""

import numpy as np


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
    return "Ez" if sim.ndim == 1 or sim.mode == "Ez" else "Hz"


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
    """Point dipole (a line current along z in 2D, a sheet in 1D).

    Parameters
    ----------
    position : number or tuple
    profile : callable
        Time profile, e.g. GaussianPulse or ContinuousWave.
    component : str, optional
        Field component the current drives: Ez (default in 1D and the Ez
        mode), Ex or Ey (electric), Hz (magnetic current, default in the Hz
        mode).
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
    """Current sheet across the domain at a position along an axis (2D), or
    a sheet in 1D. With periodic boundaries along the line it launches plane
    waves to both sides.

    Parameters
    ----------
    position : number
        Coordinate of the line along `axis`.
    profile : callable
        Time profile.
    axis : int
        Normal of the line (0: a line x = position, 1: y = position).
    component : str, optional
        As for PointSource.
    extent : (lo, hi), optional
        Part of the line that radiates (default: all of it).
    transverse : callable, optional
        Amplitude across the line as a function of the coordinate along it,
        e.g. a Gaussian beam profile.
    amplitude : number
    """

    def __init__(self, position, profile, axis=0, component=None, extent=None, transverse=None,
                 amplitude=1.0):
        self.position = position
        self.profile = profile
        self.axis = axis
        self.component = component
        self.extent = extent
        self.transverse = transverse
        self.amplitude = amplitude

    def setup(self, sim):
        if self.component is None:
            self.component = _default_component(sim)
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
        self._weight = w


class TFSF(Source):
    """Plane wave travelling towards +x, injected on the boundary of a box
    (total-field / scattered-field formulation, 2D, both modes).

    Inside the box the total field is computed, outside only the scattered
    field: without scatterers the field outside the box is zero up to
    round-off. The incident wave comes from an auxiliary 1D grid with the
    same cell size and time step, whose numerical dispersion along x is
    identical to that of the 2D grid.

    Parameters
    ----------
    box : (xmin, xmax, ymin, ymax)
        Total field region; it should lie in the background medium near
        its edges and stay clear of the PML.
    profile : callable
        Time profile of the incident field (its E at the source of the
        auxiliary grid; the wave reaches x = xmin a bit later).
    n_background : number
        Refractive index of the background medium.
    """

    def __init__(self, box, profile, n_background=1.0):
        self.box = box
        self.profile = profile
        self.n_background = n_background

    def setup(self, sim):
        from .simulation import Simulation
        from .grid import Grid
        from .media import Medium
        from .boundaries import PML
        if sim.ndim != 2:
            raise ValueError("TFSF needs a 2D simulation")
        grid = sim.grid
        ex = grid.axis(0)
        xmin, xmax, ymin, ymax = self.box
        self.i0 = int(np.ceil((xmin - grid.origin[0])/grid.spacing[0] - 1e-9))
        self.i1 = int(np.floor((xmax - grid.origin[0])/grid.spacing[0] + 1e-9))
        self.j0 = int(np.ceil((ymin - grid.origin[1])/grid.spacing[1] - 1e-9))
        self.j1 = int(np.floor((ymax - grid.origin[1])/grid.spacing[1] + 1e-9))
        if not (0 < self.i0 < self.i1 < grid.shape[0] - 1 and 0 < self.j0 < self.j1 < grid.shape[1] - 1):
            raise ValueError("TFSF box must lie inside the domain")
        dx = grid.spacing[0]
        pml = 20
        k0 = pml + 8    # auxiliary index of main x index i0
        n_aux = k0 + (self.i1 - self.i0) + 4 + pml
        aux_grid = Grid(n_aux*dx, dx, center=n_aux*dx/2)
        self.aux = Simulation(aux_grid, boundaries=PML(pml), courant=None, dt=sim.dt,
                              background=Medium(self.n_background**2), dtype=sim.dtype,
                              sources=[PointSource(aux_grid.axis(0)[pml + 4], self.profile)])
        self.aux.setup()
        self.koff = k0 - self.i0     # aux index = main x index + koff
        self.x0 = ex[self.i0]
        self.box_indices = (self.i0, self.i1, self.j0, self.j1)

    def inject_h(self, sim, t):
        # called after the H update of the main grid (aux E at the same time)
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

    def incident(self, x, t=None):
        """Incident Ez (Ez mode) or Ey (Hz mode) of the auxiliary grid at x,
        at the current time of the simulation (nearest node)."""

        k = int(round((x - self.x0)/self.aux.grid.spacing[0])) + self.koff + self.i0
        return self.aux.fields["Ez"][k]
