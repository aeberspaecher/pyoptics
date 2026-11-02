#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Periodic structures: diffraction orders of gratings and Bloch modes
(band structures) of photonic crystals.

Diffraction orders
------------------
A structure that is periodic along t (period L, the length of the domain,
with periodic or Bloch(k) boundaries) scatters a plane wave with the
transverse wave number k into the orders ::

    k_m = k + 2 pi m/L,

which propagate along the other axis (the axis of the monitor) where
k_m**2 < n**2 k0**2. DiffractionMonitor takes the DFT of the transverse E
and H on a line across the period and expands both in the orders,
E(t) = sum E_m exp(i k_m t). The power of order m through the line is
1/2 Re(E_m H_m*) L (with the sign of the Poynting vector along the axis),
and with the incident power (from a run without the structure) these are
the diffraction efficiencies. H is interpolated to the E nodes along the
axis, which multiplies each order by cos(k_a,m h/2); orders() divides it
out with k_a,m from the discrete dispersion relation of the grid, so the
efficiencies carry only the discretization error of the structure.

For reflected orders the incident field has to be removed: pass the
monitor of a reference run without the structure (same source, same
grid; the incident medium everywhere) as `reference`; its fields are
subtracted.

Bloch modes
-----------
bloch_modes() finds the frequencies of the modes of a periodic medium at
one Bloch wave vector: a unit cell with Bloch boundaries on all periodic
axes, a few dipoles at random positions excited by a short pulse, and the
ring-down at a few probes analysed with the matrix pencil method
(analysis.resonances). Sweeping k along the edges of the Brillouin zone
gives the band diagram.
"""

import numpy as np

from . import boundaries as _bnd
from .monitors import Monitor, _FLUX


def _bloch_k(sim, axis):
    b = sim.boundaries[axis]
    if isinstance(b, _bnd.Bloch):
        return b.k
    if b == _bnd.PERIODIC:
        return 0.0
    raise ValueError("DiffractionMonitor needs periodic or Bloch boundaries along the line")


class DiffractionOrders(object):
    """Result of DiffractionMonitor.orders(): per wavelength (rows) and
    order (columns).

    Attributes
    ----------
    wavelengths : (W,)
    m : (M,)
        Order numbers.
    kt : (M,)
        Transverse wave numbers k + 2 pi m/L.
    power : (W, M)
        Power of each order through the line, positive along +axis, per
        unit length along z (0 for evanescent orders).
    propagating : (W, M) bool
    angle : (W, M)
        Angle from the axis (radians, nan for evanescent orders) in the
        medium at the line, sin(angle) = kt/(n k0) with the sign of kt.
    amplitude : (W, M) complex
        Complex amplitude of each order of the transverse E on the line
        (the DFT of E expanded in exp(i kt t), referred to the time origin
        of the run). Ratios of two runs with the same source give
        transmission and reflection coefficients with their phases, e.g.
        the phase library of a metasurface.
    """

    def __init__(self, wavelengths, m, kt, power, propagating, angle, amplitude=None):
        self.wavelengths, self.m, self.kt = wavelengths, m, kt
        self.power, self.propagating, self.angle = power, propagating, angle
        self.amplitude = amplitude

    def order(self, m):
        """Power of order m, (W,)."""

        return self.power[:, int(np.nonzero(self.m == m)[0][0])]

    def coefficient(self, m):
        """Complex amplitude of order m, (W,)."""

        return self.amplitude[:, int(np.nonzero(self.m == m)[0][0])]


class DiffractionMonitor(Monitor):
    """Diffraction orders through a line across one period.

    Parameters
    ----------
    position : float
        Coordinate of the line along `axis`.
    wavelengths : array
    axis : int
        Normal of the line, the direction in which the orders leave (the
        other axis must be periodic or Bloch-periodic).
    max_order : int
        Orders -max_order..max_order are evaluated.
    """

    def __init__(self, position, wavelengths, axis=1, max_order=10):
        self.position = position
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        self.axis = axis
        self.max_order = int(max_order)

    def setup(self, sim):
        if sim.ndim != 2:
            raise ValueError("DiffractionMonitor needs a 2D simulation")
        a, t = self.axis, 1 - self.axis
        self._kb = _bloch_k(sim, t)
        cE, cH, self._sign = _FLUX[(sim.ndim, sim.mode, a)]
        pos = np.zeros(2)
        pos[a] = self.position
        i = sim.grid.index(pos, cE, sim.mode)[a]
        if i < 1:
            raise ValueError("diffraction monitor too close to the wall")
        self._sE = (i, slice(None)) if a == 0 else (slice(None), i)
        self._sH = (slice(i - 1, i + 1), slice(None)) if a == 0 else (slice(None), slice(i - 1, i + 1))
        self._cE, self._cH = cE, cH
        self.coords = sim.coords(cE)[t]
        self._L = sim.grid.size[t]
        self._h = sim.grid.spacing
        self._dt = sim.dt
        eps_line = sim.eps[cE][self._sE]
        self.eps = float(np.mean(eps_line))
        if np.ptp(eps_line) > 1e-9*self.eps:
            raise ValueError("the monitor line must lie in a homogeneous medium")
        self._omega = 2*np.pi/self.wavelengths
        n = len(self.coords)
        self._E = np.zeros((len(self._omega), n), complex)
        self._H = np.zeros((len(self._omega), n), complex)

    def record(self, sim):
        E = sim.fields[self._cE][self._sE]
        Hp = sim.fields[self._cH][self._sH]
        H = 0.5*(Hp[0] + Hp[1]) if self.axis == 0 else 0.5*(Hp[:, 0] + Hp[:, 1])
        pE = np.exp(1j*self._omega*sim.time)*sim.dt
        pH = pE*np.exp(-0.5j*self._omega*sim.dt)
        self._E += np.outer(pE, E)
        self._H += np.outer(pH, H)

    def fields(self):
        """DFT amplitudes of E_t and H_t (interpolated) along the line."""

        return self._E, self._H

    def orders(self, reference=None):
        """Power in the diffraction orders, optionally of the fields minus
        those of a reference run (for the reflected orders).

        Returns
        -------
        DiffractionOrders
        """

        E, H = self._E, self._H
        if reference is not None:
            E = E - reference._E
            H = H - reference._H
        m = np.arange(-self.max_order, self.max_order + 1)
        kt = self._kb + 2*np.pi*m/self._L
        basis = np.exp(-1j*np.outer(kt, self.coords))/len(self.coords)
        Em = E @ basis.T
        Hm = H @ basis.T
        # discrete dispersion: (2/dt sin(w dt/2))^2 eps = sum_a (2/h_a sin(k_a h_a/2))^2
        a, t = self.axis, 1 - self.axis
        ha, ht = self._h[a], self._h[t]
        k0t = 2/self._dt*np.sin(self._omega*self._dt/2)
        rhs = k0t[:, None]**2*self.eps - (2/ht*np.sin(kt[None, :]*ht/2))**2
        s = rhs*(ha/2)**2
        prop = (rhs > 0) & (s < 1)
        ka = np.where(prop, 2/ha*np.arcsin(np.sqrt(np.clip(s, 0, 1))), 0.0)
        cosf = np.where(prop, np.cos(ka*ha/2), 1.0)
        P = 0.5*np.real(self._sign*Em*np.conj(Hm/cosf))*self._L
        P = np.where(prop, P, 0.0)
        n = np.sqrt(self.eps)
        with np.errstate(invalid="ignore"):
            sin_t = kt[None, :]/(n*self._omega[:, None])
            angle = np.where(prop & (np.abs(sin_t) <= 1), np.arcsin(np.clip(sin_t, -1, 1)), np.nan)
        return DiffractionOrders(self.wavelengths, m, kt, P, prop, angle, amplitude=Em)


def bloch_modes(grid, structure, k, band, mode="Ez", background=None, time=None, n_sources=5,
                n_probes=5, seed=1, subpixel=None, backend="auto", region=None,
                **resonance_options):
    """Frequencies of the Bloch modes of a periodic medium at one wave
    vector.

    Parameters
    ----------
    grid : Grid
        The unit cell (1D or 2D); all its axes are Bloch-periodic.
    structure : list of (Shape, Medium)
    k : float or (kx, ky)
        Bloch wave vector (rad per length unit).
    band : (wavelength_min, wavelength_max)
        Vacuum wavelengths to look for modes in; the exciting pulse covers
        them.
    mode : "Ez" or "Hz"
    background : Medium, optional
    time : float, optional
        Length of the run (default: 60 periods at the longest wavelength
        after the pulse).
    n_sources, n_probes : int
        Dipoles and probes at random positions (a mode with a node at one of
        them still shows at the others).
    seed : int
    region : ((x0, x1), (y0, y1)), optional
        Where the dipoles and probes go (default: the whole cell); in a
        supercell with a defect, the defect region picks out its modes.
    resonance_options
        Passed to analysis.resonances (e.g. min_q, max_poles).

    Returns
    -------
    analysis.Resonances
        The modes in the band, sorted by frequency; .frequency in units of
        c over the length unit (a/lambda for a lattice constant a = 1).
    """

    from .simulation import Simulation
    from .sources import GaussianPulse, PointSource
    from .monitors import Probe
    from .media import VACUUM
    from .analysis import resonances
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if len(k) != grid.ndim:
        raise ValueError("need one Bloch wave number per axis")
    rng = np.random.default_rng(seed)
    lo, hi = min(band), max(band)
    f0 = 0.5*(1/lo + 1/hi)
    pulse = GaussianPulse(1/f0, bandwidth=0.5*(1/lo - 1/hi)/f0)
    if region is None:
        lo_c, size = np.asarray(grid.origin, float), np.asarray(grid.size, float)
    else:
        reg = np.atleast_2d(np.asarray(region, dtype=float))
        lo_c, size = reg[:, 0], reg[:, 1] - reg[:, 0]
    pos = lambda: tuple(lo_c + rng.random(grid.ndim)*size)
    pts = [pos() for _ in range(n_sources)]
    sources = [PointSource(p if grid.ndim == 2 else p[0], pulse,
                           amplitude=np.exp(2j*np.pi*rng.random())) for p in pts]
    probes = [Probe(p if grid.ndim == 2 else p[0]) for p in [pos() for _ in range(n_probes)]]
    bnd = tuple(_bnd.Bloch(kk) for kk in k)
    sim = Simulation(grid, structure, sources, probes, mode=mode, boundaries=bnd,
                     background=background or VACUUM, dtype=np.complex128, subpixel=subpixel,
                     backend=backend)
    if time is None:
        time = pulse.end + 60*hi
    sim.run(time=time)
    sel = probes[0].times > pulse.end
    sig = sum(np.real(p.values[sel]) for p in probes)
    opts = dict(band=(lo, hi), min_q=50.0)
    opts.update(resonance_options)
    return resonances(sig, sim.dt, t0=probes[0].times[sel][0], **opts)
