#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Monitors record data during a run, so the full time history never has
to be stored.

- Probe: time series of one component at a point (spectra, ring-downs).
- Snapshots: decimated frames of a component for plots and animations,
  optionally cropped, subsampled, stored as float16, kept as a ring buffer
  (keep_last) or streamed to an HDF5 file.
- DFTField: running discrete Fourier transform at a few wavelengths. The
  complex amplitudes follow the pyoptics convention (time dependence
  exp(-i omega t)): a field f(t) gives F = sum f(t) exp(+i omega t) dt, so
  a wave travelling towards +x has the phase exp(+ikx). For a continuous
  wave, Re{F exp(-i omega t)} animates the steady state over one period.
- Flux: power through a plane (3D), a line (2D) or a point (1D) per
  wavelength, from the DFT of E and H; ratios of two runs (with and
  without a structure) give reflectance and transmittance.

Monitor regions are given as ((xmin, xmax), (ymin, ymax)[, (zmin, zmax)])
(None for the whole domain); each component keeps the nodes that fall
inside. A number instead of a range picks the single node layer nearest
to it, e.g. ((-1, 1), 0.0, (-2, 2)) for the plane y = 0 in 3D (the arrays
keep a length-1 axis there).

References
----------
A. Taflove and S. C. Hagness, "Computational Electrodynamics: The
Finite-Difference Time-Domain Method", 3rd ed., Artech House (2005), chapter
5 (DFT and flux).

A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and S.
G. Johnson, "MEEP: A flexible free-software package for electromagnetic
simulations by the FDTD method", Comput. Phys. Commun. 181, 687-702 (2010).
"""

import collections

import numpy as np

from . import storage
from .grid import OFFSETS, mode_key


def _region_index(sim, component, region, stride=1):
    """Index tuple (slices) and coordinates of a component inside a region."""

    coords = sim.coords(component)
    idx, cs = [], []
    for a, c in enumerate(coords):
        if region is None or region[a] is None:
            sl = slice(None, None, stride)
        elif np.ndim(region[a]) == 0:
            i = int(np.argmin(np.abs(c - region[a])))
            sl = slice(i, i + 1)
        else:
            lo, hi = region[a]
            sel = np.nonzero((c >= lo - 1e-9) & (c <= hi + 1e-9))[0]
            if len(sel) == 0:
                raise ValueError("monitor region misses the grid along axis %d" % a)
            sl = slice(sel[0], sel[-1] + 1, stride)
        idx.append(sl)
        cs.append(c[sl])
    return tuple(idx), tuple(cs)


def _time_of(sim, component):
    """Time of the current values of a component (H lags by dt/2)."""

    return sim.time - (0.5*sim.dt if component[0] == "H" else 0.0)


class Monitor(object):
    """Base class: setup(sim) once, record(sim) after every step."""

    def setup(self, sim):
        pass

    def record(self, sim):
        pass


class Probe(Monitor):
    """Time series of one field component at the node nearest to a point.

    Attributes after a run: times, values (arrays).
    """

    def __init__(self, position, component=None):
        self.position = position
        self.component = component

    def setup(self, sim):
        if self.component is None:
            self.component = sim.default_component
        self._idx = sim.grid.index(self.position, self.component, sim.mode)
        self._t, self._v = [], []

    def record(self, sim):
        self._t.append(_time_of(sim, self.component))
        self._v.append(sim.fields[self.component][self._idx].item())

    @property
    def times(self):
        return np.array(self._t)

    @property
    def values(self):
        return np.array(self._v)

    def spectrum(self, wavelengths):
        """DFT of the recorded series at vacuum wavelengths (convention as
        DFTField)."""

        w = 2*np.pi/np.atleast_1d(np.asarray(wavelengths, dtype=float))
        dt = self._t[1] - self._t[0] if len(self._t) > 1 else 1.0
        return np.exp(1j*np.outer(w, self.times)) @ self.values*dt


class SnapshotStack(object):
    """Frames of one component: times (n,), coords (one array per axis),
    frames (n, ...) and the component name."""

    def __init__(self, times, coords, frames, component, mode="Ez"):
        self.times = np.asarray(times)
        self.coords = tuple(np.asarray(c) for c in coords)
        self.frames = np.asarray(frames)
        self.component = component
        self.mode = mode

    def __len__(self):
        return len(self.times)

    @property
    def nbytes(self):
        return self.frames.nbytes

    def save(self, path):
        """Save to .npz or (with h5py) .h5."""

        arrays = {"times": self.times, "frames": self.frames}
        for i, c in enumerate(self.coords):
            arrays["xyz"[i]] = c
        storage.save(path, arrays, {"component": self.component, "mode": self.mode})

    @classmethod
    def load(cls, path):
        arrays, attrs = storage.load(path)
        coords = tuple(arrays[k] for k in "xyz" if k in arrays)
        return cls(arrays["times"], coords, arrays["frames"], attrs.get("component", ""),
                   attrs.get("mode", "Ez"))


class Snapshots(Monitor):
    """Decimated frames of one field component.

    Parameters
    ----------
    component : str, optional
        Default: Ez in the Ez mode, Hz in the Hz mode.
    every : int
        Record every `every` steps.
    region : tuple, optional
        Crop, ((xmin, xmax), (ymin, ymax)).
    stride : int
        Spatial subsampling.
    dtype : numpy dtype
        Storage type of the frames; float16 halves the memory of float32
        and is plenty for plots. Complex (Bloch) fields are stored as their
        real part unless dtype is complex.
    keep_last : int, optional
        Keep only the last frames (ring buffer).
    start : number
        Time at which recording starts.
    path : str, optional
        Stream the frames to this HDF5 file (needs h5py) instead of keeping
        them in memory.
    """

    def __init__(self, component=None, every=10, region=None, stride=1, dtype=np.float16,
                 keep_last=None, start=0.0, path=None):
        self.component = component
        self.every = int(every)
        self.region = region
        self.stride = int(stride)
        self.dtype = np.dtype(dtype)
        self.keep_last = keep_last
        self.start = start
        self.path = path

    def setup(self, sim):
        if self.component is None:
            self.component = sim.default_component
        self._idx, self.coords = _region_index(sim, self.component, self.region, self.stride)
        self._mode = sim.mode
        self._frames = collections.deque(maxlen=self.keep_last)
        self._times = collections.deque(maxlen=self.keep_last)
        self._writer = None
        if self.path is not None:
            shape = tuple(len(c) for c in self.coords)
            self._writer = storage.FrameWriter(self.path, shape, self.dtype, self.coords,
                                               {"component": self.component, "mode": sim.mode})

    def record(self, sim):
        if sim.step_count % self.every or sim.time < self.start:
            return
        frame = sim.fields[self.component][self._idx]
        if np.iscomplexobj(frame) and self.dtype.kind != "c":
            frame = frame.real      # complex (Bloch) fields: the real part
        frame = frame.astype(self.dtype)
        t = _time_of(sim, self.component)
        if self._writer is not None:
            self._writer.append(t, frame)
        else:
            self._frames.append(frame)
            self._times.append(t)

    def close(self):
        """Close the HDF5 file of a streaming recorder."""

        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def stack(self):
        """The recorded frames as a SnapshotStack (read back from the file
        when streaming)."""

        if self.path is not None:
            self.close()
            return SnapshotStack.load(self.path)
        frames = np.array(self._frames) if self._frames else \
            np.zeros((0,) + tuple(len(c) for c in self.coords), self.dtype)
        return SnapshotStack(np.array(self._times), self.coords, frames, self.component, self._mode)


class DFTFieldData(object):
    """Complex field amplitudes at some wavelengths: fields[component] has
    the shape (n_wavelengths, ...), coords[component] one array per axis."""

    def __init__(self, wavelengths, fields, coords, mode="Ez"):
        self.wavelengths = np.asarray(wavelengths)
        self.fields = fields
        self.coords = coords
        self.mode = mode

    def __getitem__(self, component):
        return self.fields[component]

    def save(self, path):
        arrays = {"wavelengths": self.wavelengths}
        for c, F in self.fields.items():
            arrays[c] = F
            for i, x in enumerate(self.coords[c]):
                arrays["%s_%s" % (c, "xyz"[i])] = x
        storage.save(path, arrays, {"mode": self.mode, "components": ",".join(self.fields)})

    @classmethod
    def load(cls, path):
        arrays, attrs = storage.load(path)
        comps = attrs["components"].split(",")
        fields = {c: arrays[c] for c in comps}
        coords = {c: tuple(arrays["%s_%s" % (c, a)] for a in "xyz" if "%s_%s" % (c, a) in arrays)
                  for c in comps}
        return cls(arrays["wavelengths"], fields, coords, attrs.get("mode", "Ez"))


class DFTField(Monitor):
    """Running DFT of field components at vacuum wavelengths.

    Parameters
    ----------
    wavelengths : number or array
    components : sequence of str, optional
        Default: all components of the mode.
    region : tuple, optional
        ((xmin, xmax), (ymin, ymax)); default: the whole grid.
    start : number
        Time from which on to accumulate (e.g. after a CW source has
        reached the steady state).
    stride : int
        Spatial subsampling.
    """

    def __init__(self, wavelengths, components=None, region=None, start=0.0, stride=1):
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        self.components = components
        self.region = region
        self.start = start
        self.stride = stride

    def setup(self, sim):
        comps = self.components or sim.components
        self._omega = 2*np.pi/self.wavelengths
        self._idx, self._coords, self._acc = {}, {}, {}
        for c in comps:
            idx, coords = _region_index(sim, c, self.region, self.stride)
            self._idx[c], self._coords[c] = idx, coords
            self._acc[c] = np.zeros((len(self.wavelengths),) + tuple(len(x) for x in coords), complex)
        self._mode = sim.mode
        self._offsets = OFFSETS[sim.ndim][mode_key(sim.ndim, sim.mode)]
        self.duration = 0.0

    def record(self, sim):
        if sim.time < self.start:
            return
        self.duration += sim.dt
        for c, acc in self._acc.items():
            F = sim.fields[c][self._idx[c]]
            phase = np.exp(1j*self._omega*_time_of(sim, c))*sim.dt
            for k in range(len(phase)):
                acc[k] += phase[k]*F

    def field(self, component):
        """Complex amplitudes (n_wavelengths, ...) of a component."""

        return self._acc[component]

    def coords(self, component):
        return self._coords[component]

    def colocated(self, components=None, wavelength_index=0):
        """Components moved to common nodes (the integer nodes of the Yee
        grid, by averaging the two neighbours along each axis where a
        component sits half a cell off), on the nodes all of them share:
        for intensities or vector plots. Along the normal of a plane monitor
        (one node layer) the components are taken as they are, half a cell
        apart.

        Returns
        -------
        coords : tuple of arrays
        fields : complex array (n_components, ...)
        """

        comps = list(components or self._acc)
        out, crds = [], []
        for c in comps:
            F = self._acc[c][wavelength_index]
            cs = list(self._coords[c])
            for a, o in enumerate(self._offsets[c]):
                if o and F.shape[a] > 1:
                    lo = [slice(None)]*F.ndim
                    hi = [slice(None)]*F.ndim
                    lo[a], hi[a] = slice(0, -1), slice(1, None)
                    F = 0.5*(F[tuple(lo)] + F[tuple(hi)])
                    cs[a] = 0.5*(cs[a][:-1] + cs[a][1:])
            out.append(F)
            crds.append(cs)
        common, sel = [], [[] for _ in comps]
        for a in range(len(crds[0])):
            ref = crds[0][a]
            if all(len(cs[a]) == 1 for cs in crds):   # a plane monitor: the nearest layers
                common.append(ref)
                for q in range(len(comps)):
                    sel[q].append(np.zeros(1, dtype=int))
                continue
            keep = np.ones(len(ref), dtype=bool)
            for cs in crds[1:]:
                keep &= np.min(np.abs(ref[:, None] - cs[a][None, :]), axis=1) < 1e-6
            common.append(ref[keep])
            for q, cs in enumerate(crds):
                sel[q].append(np.array([np.argmin(np.abs(cs[a] - v)) for v in ref[keep]], dtype=int))
        fields = np.array([F[np.ix_(*sel[q])] for q, F in enumerate(out)])
        return tuple(common), fields

    def data(self):
        """The result as DFTFieldData (e.g. to save it)."""

        return DFTFieldData(self.wavelengths, {c: a.copy() for c, a in self._acc.items()},
                            dict(self._coords), self._mode)

    def steady_state(self, component):
        """Complex amplitudes normalized to a continuous wave: for a CW run
        accumulated over whole periods, this is the phasor of the field
        (F*2/duration)."""

        return self._acc[component]*2/self.duration


# component pairs (E, H, sign) for the Poynting vector along an axis:
# S_axis = sign*E*H
_FLUX = {
    (1, "Ez", 0): ("Ez", "Hy", -1),
    (2, "Ez", 0): ("Ez", "Hy", -1), (2, "Ez", 1): ("Ez", "Hx", 1),
    (2, "Hz", 0): ("Ey", "Hz", 1), (2, "Hz", 1): ("Ex", "Hz", -1),
}
# 3D: S_x = Ey Hz - Ez Hy, S_y = Ez Hx - Ex Hz, S_z = Ex Hy - Ey Hx; each
# H differs from its E partner by half a cell along the normal only
_FLUX3 = {0: (("Ey", "Hz", 1), ("Ez", "Hy", -1)), 1: (("Ez", "Hx", 1), ("Ex", "Hz", -1)),
          2: (("Ex", "Hy", 1), ("Ey", "Hx", -1))}


class Flux(Monitor):
    """Spectral power flux through a plane (3D), a line (2D) or a point
    (1D).

    The line or plane is normal to `axis` at `position`, optionally
    limited to `extent` along the other axis ((lo, hi)) or, in 3D, axes
    (((lo, hi), (lo, hi)) in the order x, y, z, entries may be None); the
    flux counts positive along +axis. H is interpolated to the E nodes.
    flux() returns 1/2 Re(sum E H*) dA per wavelength (with a unit
    amplitude source, in arbitrary units: divide by the flux of a reference
    run).
    """

    def __init__(self, position, wavelengths, axis=0, extent=None):
        self.position = position
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        self.axis = axis
        self.extent = extent

    def setup(self, sim):
        if sim.ndim == 3:
            self._parts = []
            for cE, cH, sign in _FLUX3[self.axis]:
                part = Flux(self.position, self.wavelengths, self.axis, self.extent)
                part._setup3(sim, cE, cH, sign)
                self._parts.append(part)
            return
        self._parts = None
        a = self.axis
        cE, cH, self._sign = _FLUX[(sim.ndim, sim.mode, a)]
        grid = sim.grid
        pos = np.zeros(sim.ndim)
        pos[a] = self.position
        i = grid.index(pos, cE, sim.mode)[a]
        if i < 1:
            raise ValueError("flux line too close to the wall")
        if sim.ndim == 1:
            sE, sH = (i,), (slice(i - 1, i + 1),)
            self._dA = 1.0
        else:
            other = 1 - a
            t = sim.coords(cE)[other]
            if self.extent is None:
                tsl = slice(None)
            else:
                sel = np.nonzero((t >= self.extent[0]) & (t <= self.extent[1]))[0]
                tsl = slice(sel[0], sel[-1] + 1)
            sE = [None, None]
            sH = [None, None]
            sE[a], sE[other] = slice(i, i + 1), tsl
            sH[a], sH[other] = slice(i - 1, i + 1), tsl
            sE, sH = tuple(sE), tuple(sH)
            self._dA = grid.spacing[other]
            self.coords = t[tsl]
        self._cE, self._cH = cE, cH
        self._sE, self._sH = sE, sH
        self._omega = 2*np.pi/self.wavelengths
        nE = np.asarray(sim.fields[cE][sE]).size
        self._E = np.zeros((len(self._omega), nE), complex)
        self._H = np.zeros((len(self._omega), nE), complex)

    def _setup3(self, sim, cE, cH, sign):
        self._parts = None
        a = self.axis
        grid = sim.grid
        pos = np.zeros(3)
        pos[a] = self.position
        i = grid.index(pos, cE, sim.mode)[a]
        if i < 1:
            raise ValueError("flux plane too close to the wall")
        sE, sH = [None]*3, [None]*3
        others = [b for b in range(3) if b != a]
        for q, b in enumerate(others):
            t = sim.coords(cE)[b]
            ext = None if self.extent is None else self.extent[q]
            if ext is None:
                sE[b] = sH[b] = slice(None)
            else:
                sel = np.nonzero((t >= ext[0]) & (t <= ext[1]))[0]
                sE[b] = sH[b] = slice(sel[0], sel[-1] + 1)
        sE[a], sH[a] = slice(i, i + 1), slice(i - 1, i + 1)
        self._sign = sign
        self._cE, self._cH = cE, cH
        self._sE, self._sH = tuple(sE), tuple(sH)
        self._dA = grid.spacing[others[0]]*grid.spacing[others[1]]
        self._omega = 2*np.pi/self.wavelengths
        nE = np.asarray(sim.fields[cE][self._sE]).size
        self._E = np.zeros((len(self._omega), nE), complex)
        self._H = np.zeros((len(self._omega), nE), complex)

    def record(self, sim):
        if self._parts:
            for p in self._parts:
                p.record(sim)
            return
        E = np.asarray(sim.fields[self._cE][self._sE]).ravel()
        Hpair = np.asarray(sim.fields[self._cH][self._sH])
        if sim.ndim == 1:
            H = 0.5*(Hpair[0] + Hpair[1])
        else:
            H = 0.5*Hpair.sum(axis=self.axis).ravel()
        pE = np.exp(1j*self._omega*sim.time)*sim.dt
        pH = np.exp(1j*self._omega*(sim.time - 0.5*sim.dt))*sim.dt
        self._E += np.outer(pE, E)
        self._H += np.outer(pH, H)

    def flux(self):
        """Power flux per wavelength (n_wavelengths,)."""

        if self._parts:
            return sum(p.flux() for p in self._parts)
        return 0.5*np.real(self._sign*np.sum(self._E*np.conj(self._H), axis=1))*self._dA

    def fields(self):
        """DFT amplitudes of E and H (interpolated) along the line (in 3D a
        list of the (E, H) pairs of the two terms of the Poynting vector)."""

        if self._parts:
            return [(p._E, p._H) for p in self._parts]
        return self._E, self._H
