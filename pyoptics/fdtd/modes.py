#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Waveguide modes on the Yee grid, mode sources and mode monitors.

Mode solver
-----------
A waveguide along axis a (0: x, 1: y) has a cross section along the other
axis t. Its modes vary as exp(i(beta s - omega t)) along s. On the Yee
grid the transverse operator is a three-point finite difference, and the
eigenvalue problem reads ::

    Ez mode:  D_tt Ez + k0~**2 eps_z Ez = beta~**2 Ez,
    Hz mode:  eps_t D_t (1/eps_l) D_t Hz + k0~**2 eps_t Hz = beta~**2 Hz,

with eps_l the permittivity of the E component along the waveguide and
eps_t that of the transverse E component. For the grid of a simulation
k0~ = 2 sin(omega dt/2)/dt and beta~ = 2 sin(beta ds/2)/ds include the
numerical dispersion of the leapfrog in time and of the differences along
s, so the modes are exact eigenmodes of the discretized Maxwell equations
of a straight waveguide: they propagate without changing shape and without
spurious reflections at the source. With dt = ds = 0 (waveguide_modes()
without a simulation) the time and the propagation direction are
continuous.

The operator is symmetric (after scaling with sqrt(eps_t) in the Hz mode)
and tridiagonal, so the guided modes come from a tridiagonal eigensolver in
O(n) per mode. The window around the waveguide has Dirichlet walls (the
guided field decays before it reaches them); it must not contain other
waveguides unless their supermodes are wanted.

Normalization and field relations
---------------------------------
Each mode carries unit power, P = 1/2 int (E x H) . s dt = 1. The
transverse fields of a mode travelling towards +s are ::

    Ez mode:  E_t = Ez = u,  H_t = -+ n Ez     (Hy = -n Ez for s = x,
                                                Hx = +n Ez for s = y)
    Hz mode:  H_t = Hz = u,  E_t = +- n Hz/eps_t (Ey = +n Hz/eps_y for x,
                                                 Ex = -n Hz/eps_x for y)

with n = beta~/k0~. A mode travelling towards -s has the same E_t and the
opposite H_t.

Mode amplitudes
---------------
A ModeMonitor records the DFT of E_t and H_t on a line (H interpolated to
the E nodes as in Flux) and projects them on the modes at each wavelength:
with E = (a+ + a-) E_m, H = (a+ - a-) H_m, ::

    a+- = (int E x H_m* +- int E_m* x H)/4

(for unit-power modes with real profiles), and ``|a+-|**2`` is the power in
the forward and backward mode. Ratios of amplitudes at two monitors are
S-parameters (s_parameters()).

Mode source
-----------
ModeSource injects a mode on a line with the total-field / scattered-field
method (a one-sided TFSF): on the side the mode travels to, the total field
is computed, behind it only the scattered field, i.e. what the structure
sends back. The incident field is u(t) g(s, t), where g comes from an
auxiliary 1D grid filled with eps = n**2 at the centre wavelength. There
the 1D wave and the mode have identical phase velocities (to round-off),
so nothing leaks backwards; away from the centre wavelength the mode
dispersion differs slightly from that of the uniform 1D medium and a small
fraction leaks (about 1e-6 in the Ez and 2e-4 in the Hz mode of a
high-index waveguide at +-6 % of the centre frequency).
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .monitors import Monitor
from .sources import Source

# per (mode, axis): (transverse E, transverse H, longitudinal E, c) with c
# the sign of the updates along the axis: H_t += c dt dE_t/ds,
# E_t += c Cb dH_t/ds
_COMP = {
    ("Ez", 0): ("Ez", "Hy", None, 1),
    ("Ez", 1): ("Ez", "Hx", None, -1),
    ("Hz", 0): ("Ey", "Hz", "Ex", -1),
    ("Hz", 1): ("Ex", "Hz", "Ey", 1),
}


class Mode(object):
    """A guided mode at one wavelength.

    Attributes
    ----------
    wavelength : float
    n_eff : float
        Effective index beta/k0 (with the discretization, beta and k0 are
        the physical ones whose finite differences give beta~ and k0~).
    beta : float
    coords : array
        Positions along the cross section (E_t nodes).
    E, H : arrays
        Transverse E and H of the mode travelling towards +s, normalized to
        unit power.
    mode : "Ez" or "Hz"
    axis : int
        Propagation axis.
    n_tilde : float
        beta~/k0~, the ratio of H_t to E_t (n_eff for a continuous grid).
    """

    def __init__(self, wavelength, n_eff, beta, coords, E, H, mode, axis, n_tilde, order):
        self.wavelength = wavelength
        self.n_eff = n_eff
        self.beta = beta
        self.coords = coords
        self.E = E
        self.H = H
        self.mode = mode
        self.axis = axis
        self.n_tilde = n_tilde
        self.order = order

    @property
    def field(self):
        """The field along z (Ez or Hz) of the mode."""

        return self.E if self.mode == "Ez" else self.H

    def __repr__(self):
        return "Mode(%s, order=%d, wavelength=%g, n_eff=%.6f)" % (self.mode, self.order,
                                                                 self.wavelength, self.n_eff)


def _sign(mode, axis):
    """Sign s with S_axis = s*E_t*H_t."""

    # Sx = Ey Hz - Ez Hy, Sy = Ez Hx - Ex Hz
    return {("Ez", 0): -1, ("Ez", 1): 1, ("Hz", 0): 1, ("Hz", 1): -1}[(mode, axis)]


def waveguide_modes(eps, spacing, wavelength, mode="Ez", num_modes=1, eps_long=None, axis=0,
                    coords=None, dt=0.0, ds=0.0):
    """Guided modes of a cross section sampled on a regular grid.

    Parameters
    ----------
    eps : array (n,)
        Permittivity across the waveguide: of Ez (Ez mode) or of the
        transverse E component (Hz mode), at its nodes.
    spacing : float
        Sample distance across the waveguide.
    wavelength : float
        Vacuum wavelength.
    mode : "Ez" or "Hz"
    num_modes : int
        Number of modes, by decreasing effective index.
    eps_long : array (n + 1,), optional
        Hz mode: permittivity of the E component along the waveguide, at
        the nodes between the samples of eps (and one beyond each end);
        default: the mean of the neighbours of eps.
    axis : int
        Propagation axis (sets the signs of the field relations).
    coords : array (n,), optional
        Positions of the samples (default: spacing*arange(n)).
    dt, ds : float
        Time step and cell size along the waveguide of a simulation, for
        modes that are exact on its grid; 0: continuous.

    Returns
    -------
    list of Mode
        The guided modes found (the effective index above the largest
        permittivity at the window edges); fewer than num_modes if fewer
        are guided.
    """

    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    h = float(spacing)
    omega = 2*np.pi/wavelength
    k0t = 2*np.sin(omega*dt/2)/dt if dt else omega
    if coords is None:
        coords = h*np.arange(n)
    if mode == "Ez":
        d = -2/h**2 + k0t**2*eps
        e = np.full(n - 1, 1/h**2)
        scale = np.ones(n)
    elif mode == "Hz":
        if eps_long is None:
            ep = np.concatenate([eps[:1], eps, eps[-1:]])
            eps_long = 0.5*(ep[:-1] + ep[1:])
        el = np.asarray(eps_long, dtype=float)
        if len(el) != n + 1:
            raise ValueError("eps_long needs len(eps) + 1 samples")
        # A = eps_t (L + k0^2), L u_j = ((u_{j+1}-u_j)/el_{j+1} - (u_j-u_{j-1})/el_j)/h^2;
        # symmetric form S^-1 A S with S = sqrt(eps_t)
        d = -eps*(1/el[1:] + 1/el[:-1])/h**2 + k0t**2*eps
        e = np.sqrt(eps[:-1]*eps[1:])/el[1:-1]/h**2
        scale = np.sqrt(eps)
    else:
        raise ValueError("mode must be 'Ez' or 'Hz'")
    m = min(num_modes, n)
    w, v = eigh_tridiagonal(d, e, select="i", select_range=(n - m, n - 1))
    w, v = w[::-1], v[:, ::-1]
    cutoff = k0t**2*max(eps[0], eps[-1])
    s = _sign(mode, axis)
    modes = []
    for k in range(m):
        lam = w[k]
        if lam <= cutoff:
            break
        bt = np.sqrt(lam)
        beta = 2/ds*np.arcsin(min(bt*ds/2, 1.0)) if ds else bt
        u = v[:, k]*scale
        nt = bt/k0t
        if mode == "Ez":
            E = u
            H = s*nt*u
        else:
            H = u
            E = s*nt*u/eps
        P = 0.5*s*np.sum(E*H)*h
        norm = np.sqrt(P)
        E, H = E/norm, H/norm
        # sign convention: largest lobe of the z field positive
        f = E if mode == "Ez" else H
        if f[np.argmax(np.abs(f))] < 0:
            E, H = -E, -H
        modes.append(Mode(wavelength, beta/omega, beta, np.asarray(coords), E, H, mode, axis,
                          nt, k))
    return modes


def _line(sim, position, axis, extent):
    """Index of the line along `axis` and the transverse slice for a
    component pair; returns (i, tsl, coords along t)."""

    if sim.ndim != 2:
        raise ValueError("mode sources and monitors need a 2D simulation")
    cE = _COMP[(sim.mode, axis)][0]
    pos = np.zeros(2)
    pos[axis] = position
    i = sim.grid.index(pos, cE, sim.mode)[axis]
    t = sim.coords(cE)[1 - axis]
    if extent is None:
        tsl = slice(0, len(t))
    else:
        sel = np.nonzero((t >= extent[0] - 1e-9) & (t <= extent[1] + 1e-9))[0]
        if len(sel) < 3:
            raise ValueError("mode window too small")
        tsl = slice(sel[0], sel[-1] + 1)
    return i, tsl, t[tsl]


def _take(arr, axis, i, tsl):
    return arr[i, tsl] if axis == 0 else arr[tsl, i]


def find_modes(sim, position, wavelength, axis=0, extent=None, num_modes=1):
    """Modes of the cross section of a simulation's structure on the line
    normal to `axis` at `position`, exact on its Yee grid.

    Parameters
    ----------
    sim : Simulation
        2D simulation (set up if needed, which rasterizes the structure).
    position : float
        Coordinate of the line along `axis`.
    wavelength : float
    axis : int
        Propagation axis (0: x, 1: y).
    extent : (lo, hi), optional
        Window along the other axis (default: the whole line). It should
        contain one waveguide and some cladding on both sides.
    num_modes : int

    Returns
    -------
    list of Mode
    """

    if not hasattr(sim, "eps"):
        sim.setup()
    cE, cH, cL, c = _COMP[(sim.mode, axis)]
    i, tsl, t = _line(sim, position, axis, extent)
    eps = _take(sim.eps[cE], axis, i, tsl)
    h = sim.grid.spacing[1 - axis]
    ds = sim.grid.spacing[axis]
    eps_long = None
    if cL is not None:
        # longitudinal E sits half a cell below the transverse E nodes
        # (offset 0 vs 0.5 along t): samples tsl.start .. tsl.stop
        full = sim.eps[cL][i, :] if axis == 0 else sim.eps[cL][:, i]
        lo, hi = tsl.start, tsl.stop
        idx = np.clip(np.arange(lo, hi + 1), 0, len(full) - 1)
        eps_long = full[idx]
    return waveguide_modes(eps, h, wavelength, sim.mode, num_modes, eps_long, axis, t,
                           dt=sim.dt, ds=ds)


def section_modes(structure, wavelength, extent, spacing, mode="Ez", background=None,
                  num_modes=1, axis=0, position=0.0, courant=0.99):
    """Modes of a waveguide cross section drawn with shapes, without a
    simulation of the whole device.

    The structure is rasterized on a thin strip of cells across `extent`
    at `position` (with the sub-cell averaging of Simulation) and solved as
    in find_modes(), with the time step of a simulation with the same
    spacing and Courant number, so the modes are the ones such a simulation
    supports (e.g. supermodes of coupled waveguides for different gaps).

    Parameters
    ----------
    structure : list of (Shape, Medium)
    wavelength : float
    extent : (lo, hi)
        Window across the waveguide(s).
    spacing : float or (dx, dy)
    mode : "Ez" or "Hz"
    background : Medium, optional
    num_modes : int
    axis : int
        Propagation axis.
    position : float
        Coordinate along the axis of the cross section.
    courant : float

    Returns
    -------
    list of Mode
    """

    from .simulation import Simulation
    from .grid import Grid
    from .media import VACUUM
    h = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    lo, hi = extent
    size = [0.0, 0.0]
    center = [0.0, 0.0]
    size[axis] = 4*h[axis]
    center[axis] = position
    size[1 - axis] = (hi - lo) + 2*h[1 - axis]
    center[1 - axis] = 0.5*(lo + hi)
    grid = Grid(tuple(size), tuple(h), center=tuple(center))
    bnd = ["pec", "pec"]
    bnd[axis] = "periodic"
    sim = Simulation(grid, structure, mode=mode, boundaries=tuple(bnd),
                     background=background or VACUUM, dt=Grid((1.0, 1.0), tuple(h)).time_step(courant),
                     backend="numpy")
    sim.setup()
    return find_modes(sim, position, wavelength, axis, extent, num_modes)


class ModeSource(Source):
    """Injects a waveguide mode travelling along +-axis from a line
    (one-sided total-field / scattered-field source).

    Parameters
    ----------
    position : float
        Coordinate of the injection line along `axis`.
    profile : callable
        Time profile, e.g. GaussianPulse or ContinuousWave.
    axis : int
        Propagation axis (0: x, 1: y).
    direction : +1 or -1
        Direction of travel.
    extent : (lo, hi), optional
        Window across the waveguide for the mode solver and the source.
    mode_index : int
        0: fundamental mode.
    wavelength : float, optional
        Wavelength of the mode profile (default: profile.wavelength).

    The fields on the side the mode travels to are total fields; behind
    the line only what the structure reflects remains. After setup, the
    injected Mode is in `mode`.
    """

    def __init__(self, position, profile, axis=0, direction=1, extent=None, mode_index=0,
                 wavelength=None, amplitude=1.0):
        self.position = position
        self.profile = profile
        self.axis = axis
        self.direction = 1 if direction > 0 else -1
        self.extent = extent
        self.mode_index = mode_index
        self.wavelength = wavelength
        self.amplitude = amplitude

    def setup(self, sim):
        from .simulation import Simulation
        from .grid import Grid
        from .media import Medium
        from .boundaries import PML
        from .sources import PointSource
        wl = self.wavelength or self.profile.wavelength
        modes = find_modes(sim, self.position, wl, self.axis, self.extent, self.mode_index + 1)
        if len(modes) <= self.mode_index:
            raise ValueError("the waveguide has no mode %d at this wavelength" % self.mode_index)
        self.mode = m = modes[self.mode_index]
        cE, cH, cL, c = _COMP[(sim.mode, self.axis)]
        self._cE, self._cH, self._c = cE, cH, c
        i, tsl, _ = _line(sim, self.position, self.axis, self.extent)
        self._i, self._tsl = i, tsl
        if not (1 <= i < sim.grid.shape[self.axis] - 1):
            raise ValueError("mode source line too close to the wall")
        # profiles: E_t = alpha_E*Ez1, H_t = c*alpha_H*Hy1 (forward)
        if sim.mode == "Ez":
            u = m.E
            self._aE = self.amplitude*u
            self._aH = self.amplitude*u
        else:
            u = m.H
            eps_t = _take(sim.eps[cE], self.axis, i, tsl)
            self._aH = self.amplitude*u
            self._aE = self.amplitude*m.n_tilde**2*u/eps_t
        ds = sim.grid.spacing[self.axis]
        pml = 20
        k0 = pml + 8
        n_aux = k0 + 4 + pml
        aux_grid = Grid(n_aux*ds, ds, center=n_aux*ds/2)
        self.aux = Simulation(aux_grid, boundaries=PML(pml), courant=None, dt=sim.dt,
                              background=Medium(m.n_tilde**2), dtype=sim.dtype, backend="numpy",
                              sources=[PointSource(aux_grid.axis(0)[pml + 4], self.profile)])
        self.aux.setup()
        self._k0 = k0
        self._ds = ds

    def _idx(self, i):
        return (i, self._tsl) if self.axis == 0 else (self._tsl, i)

    def inject_h(self, sim, t):
        Ez1 = self.aux.fields["Ez"]
        d = self.direction
        iH = self._i - 1 if d > 0 else self._i
        H = sim.fields[self._cH]
        H[self._idx(iH)] -= d*sim.dt*self._c*self._aE*Ez1[self._k0]/self._ds
        self.aux._update_h()

    def inject_e(self, sim, t):
        Hy1 = self.aux.fields["Hy"]
        E = sim.fields[self._cE]
        idx = self._idx(self._i)
        E[idx] -= sim._Cb[self._cE][idx]*self._aH*Hy1[self._k0 - 1]/self._ds
        self.aux._update_e()
        self.aux._advance()


class ModeMonitor(Monitor):
    """Amplitudes of a waveguide mode travelling in both directions through
    a line, per wavelength.

    Parameters
    ----------
    position : float
        Coordinate of the line along `axis`.
    wavelengths : array
    axis : int
        Propagation axis (0: x, 1: y).
    extent : (lo, hi), optional
        Window across the waveguide.
    mode_index : int
    every : int
        Sample every n-th step for the DFT. With a pulse source, a few
        samples per period of the shortest wavelength in the signal are
        enough (the fields contain no higher frequencies), e.g.
        every = int(wavelength_min/(8*dt)).

    After the run, amplitudes() gives a+ and a- (complex, per wavelength);
    ``|a|**2`` is the power in the mode, in the units of the Flux monitor (the
    Flux monitor itself reads cos(beta ds/2) of it for a guided mode, from
    the interpolation of H).
    """

    def __init__(self, position, wavelengths, axis=0, extent=None, mode_index=0, every=1):
        self.every = int(every)
        self.position = position
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        self.axis = axis
        self.extent = extent
        self.mode_index = mode_index

    def setup(self, sim):
        a = self.axis
        cE, cH, _, _ = _COMP[(sim.mode, a)]
        i, tsl, t = _line(sim, self.position, a, self.extent)
        if i < 1:
            raise ValueError("mode monitor too close to the wall")
        self._cE, self._cH = cE, cH
        self._sE = (i, tsl) if a == 0 else (tsl, i)
        self._sH = (slice(i - 1, i + 1), tsl) if a == 0 else (tsl, slice(i - 1, i + 1))
        self._h = sim.grid.spacing[1 - a]
        self.coords = t
        self._sign = _sign(sim.mode, a)
        self._omega = 2*np.pi/self.wavelengths
        nt = len(t)
        self._E = np.zeros((len(self._omega), nt), complex)
        self._H = np.zeros((len(self._omega), nt), complex)
        # samples are buffered and transformed in blocks (one matrix
        # product per block instead of an outer product per step)
        block = 256
        self._bE = np.zeros((block, nt), np.result_type(sim.dtype, np.float64))
        self._bH = np.zeros((block, nt), np.result_type(sim.dtype, np.float64))
        self._bt = np.zeros(block)
        self._fill = 0
        self._n = 0
        self._dt = sim.dt
        self._hshift = np.exp(-0.5j*self._omega*sim.dt)[:, None]
        self.modes = []
        for wl in self.wavelengths:
            ms = find_modes(sim, self.position, wl, a, self.extent, self.mode_index + 1)
            self.modes.append(ms[self.mode_index] if len(ms) > self.mode_index else None)
        # H interpolated from s +- ds/2 to the E nodes is cos(beta ds/2) H
        ds = sim.grid.spacing[a]
        self._cos = np.array([np.cos(m.beta*ds/2) if m is not None else 1.0 for m in self.modes])

    def record(self, sim):
        self._n += 1
        if self._n % self.every:
            return
        k = self._fill
        self._bE[k] = sim.fields[self._cE][self._sE]
        Hp = sim.fields[self._cH][self._sH]
        self._bH[k] = 0.5*(Hp[0] + Hp[1]) if self.axis == 0 else 0.5*(Hp[:, 0] + Hp[:, 1])
        self._bt[k] = sim.time
        self._fill += 1
        if self._fill == len(self._bt):
            self._flush()

    def _flush(self):
        n = self._fill
        if n == 0:
            return
        t = self._bt[:n]
        w = self.every*self._dt
        pE = np.exp(1j*np.outer(self._omega, t))*w
        pH = pE*self._hshift
        self._E += pE@self._bE[:n]
        self._H += pH@self._bH[:n]
        self._fill = 0

    def amplitudes(self):
        """Forward and backward mode amplitudes (a+, a-), each (n_wavelengths,);
        nan where the mode is not guided."""

        self._flush()
        ap = np.full(len(self.wavelengths), np.nan, complex)
        am = np.full(len(self.wavelengths), np.nan, complex)
        s, h = self._sign, self._h
        for k, m in enumerate(self.modes):
            if m is None:
                continue
            cs = self._cos[k]
            p1 = s*np.sum(self._E[k]*m.H)*h       # (a+ + a-)*2
            p2 = s*np.sum(m.E*self._H[k])*h/cs    # (a+ - a-)*2
            ap[k] = (p1 + p2)/4
            am[k] = (p1 - p2)/4
        return ap, am

    def power(self):
        """Power in the forward and backward mode, (``|a+|**2``, ``|a-|**2``)."""

        ap, am = self.amplitudes()
        return np.abs(ap)**2, np.abs(am)**2

    def fields(self):
        """DFT amplitudes of E_t and H_t (interpolated) along the line."""

        self._flush()
        return self._E, self._H


def s_parameters(incident, outgoing, direction_in=1, direction_out=1):
    """S-parameter between two mode monitors, per wavelength:
    a_out/a_in, with a_in the amplitude travelling in `direction_in` at the
    `incident` monitor and a_out that travelling in `direction_out` at the
    `outgoing` monitor (+1: towards +axis)."""

    ain = incident.amplitudes()[0 if direction_in > 0 else 1]
    aout = outgoing.amplitudes()[0 if direction_out > 0 else 1]
    return aout/ain
