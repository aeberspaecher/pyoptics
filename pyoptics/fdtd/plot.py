#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Plots and animations of FDTD simulations (needs matplotlib).

- plot_structure(): permittivity map with the PML, sources and monitors;
  a periodic unit cell can be repeated to show the structure it stands for.
- plot_field(): a component of the current fields, of a DFT monitor or of
  a snapshot, with the outlines of the structure on top.
- animate(): a matplotlib animation of a SnapshotStack (time domain) or of
  a DFT field (the steady state Re{F exp(-i omega t)} over one period,
  which needs no stored frames at all). In notebooks, show it with
  IPython.display.HTML(anim.to_jshtml()) or as a GIF with gif(anim); save
  it with anim.save().
- xt_diagram(): a 1D run as an image over x and t.

3D runs are shown as sections: plane=("y", 0.0) (the default) shows the
xz plane through y = 0, ("z", 1.0) the xy plane at z = 1; monitors
recorded on a single plane are shown as they are.

Real fields use a diverging colormap with symmetric limits, intensities
inferno. The colours of sources and monitors are those of pyoptics.sketch.
2D arrays are indexed [ix, iy] and shown with x to the right.
"""

import numpy as np

from . import sources as _src
from . import modes as _modes
from . import monitors as _mon
from .boundaries import PML, PERIODIC, Bloch


FIELD_CMAP = "RdBu_r"
INTENSITY_CMAP = "inferno"
SOURCE_COLOR = "#e0a000"   # sketch.ILLUMINATION
MONITOR_COLOR = "#2a7ab9"  # sketch.IMAGING
PML_COLOR = "0.5"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _ax(ax):
    if ax is None:
        ax = _plt().gca()
    return ax


def _extent(coords):
    """imshow extent of node coordinates (cells centred on the nodes)."""

    out = []
    for c in coords:
        d = c[1] - c[0] if len(c) > 1 else 1.0
        out += [c[0] - d/2, c[-1] + d/2]
    return out


AXIS_NAMES = "xyz"


def _plane(plane):
    """(axis index, position) of a section plane of a 3D run."""

    if plane is None:
        plane = ("y", 0.0)
    a, pos = plane
    return (AXIS_NAMES.index(a) if isinstance(a, str) else int(a)), float(pos)


def _section(F, coords, plane=None):
    """2D section of a 3D array: the given plane, else a length-1 axis
    (a monitor on a single plane), else the default plane. Returns the
    array, the coordinates and the axis names that remain."""

    if F.ndim != 3:
        return F, coords, list(AXIS_NAMES[:F.ndim])
    flat = [a for a in range(3) if F.shape[a] == 1]
    if plane is None and flat:
        a = flat[0]
        i = 0
    else:
        a, pos = _plane(plane)
        i = int(np.argmin(np.abs(coords[a] - pos)))
    keep = [b for b in range(3) if b != a]
    return np.take(F, i, axis=a), [coords[b] for b in keep], [AXIS_NAMES[b] for b in keep]


def _labels(ax, names):
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])


def _structure_outline(ax, sim, color="k", lw=0.6, alpha=0.6, plane=None):
    """Outline of each shape of the structure on top of a field plot."""

    if not sim.structure or sim.ndim == 1:
        return
    c = sim.e_components[0]
    if sim.ndim == 3:
        a, pos = _plane(plane)
        keep = [b for b in range(3) if b != a]
        x, y = (sim.coords(c)[b] for b in keep)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = [None]*3
        pts[keep[0]], pts[keep[1]], pts[a] = X, Y, np.full(X.shape, pos)
    else:
        x, y = sim.coords(c)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = (X, Y)
    for shape, _ in sim.structure:
        inside = shape.contains(*pts).astype(float)
        if 0 < inside.mean() < 1:
            ax.contour(x, y, inside.T, levels=[0.5], colors=color, linewidths=lw, alpha=alpha)


def _overlays3(ax, sim, plane, show_pml=True, show_sources=True, show_monitors=True):
    """PML, sources and monitors of a 3D run on a section plane (sources
    and monitors projected onto it)."""

    plt = _plt()
    g = sim.grid
    a, pos = _plane(plane)
    keep = [b for b in range(3) if b != a]
    ext = g.extent
    lo = [ext[2*b] for b in keep]
    hi = [ext[2*b + 1] for b in keep]

    def span(q, v0, v1, **kw):
        """A band v0..v1 along shown axis q (0: horizontal) across the plot."""
        if q == 0:
            return plt.Rectangle((v0, lo[1]), v1 - v0, hi[1] - lo[1], **kw)
        return plt.Rectangle((lo[0], v0), hi[0] - lo[0], v1 - v0, **kw)

    if show_pml:
        for q, b in enumerate(keep):
            bnd = sim.boundaries[b]
            if isinstance(bnd, PML):
                w = bnd.cells*g.spacing[b]
                for v0 in (lo[q], hi[q] - w):
                    r = span(q, v0, v0 + w)
                    r.set(facecolor="none", edgecolor=PML_COLOR, hatch="///", lw=0, alpha=0.5)
                    ax.add_patch(r)

    def line(axis, position, color, **kw):
        if axis in keep:
            q = keep.index(axis)
            if q == 0:
                ax.plot([position]*2, [lo[1], hi[1]], color=color, **kw)
            else:
                ax.plot([lo[0], hi[0]], [position]*2, color=color, **kw)

    if show_sources:
        for s in sim.sources:
            if isinstance(s, _src.PointSource):
                p = np.asarray(s.position, dtype=float)
                ax.plot(p[keep[0]], p[keep[1]], "o", ms=6, mec="k", mew=0.6, color=SOURCE_COLOR)
            elif isinstance(s, _src.LineSource):
                line(s.axis, s.position, SOURCE_COLOR, lw=2)
            elif isinstance(s, _src.TFSF):
                b = s.box
                x0, x1 = b[2*keep[0]], b[2*keep[0] + 1]
                y0, y1 = b[2*keep[1]], b[2*keep[1] + 1]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           ec=SOURCE_COLOR, lw=1.2, ls="--"))
                if 2 in keep:   # the wave travels along +z
                    if keep[1] == 2:
                        tail, tip = ((x0 + x1)/2, y0), ((x0 + x1)/2, y0 + 0.25*(y1 - y0))
                    else:
                        tail, tip = (x0, (y0 + y1)/2), (x0 + 0.25*(x1 - x0), (y0 + y1)/2)
                    ax.annotate("", xy=tip, xytext=tail,
                                arrowprops=dict(arrowstyle="-|>", color=SOURCE_COLOR, lw=1.5))
    if show_monitors:
        for m in sim.monitors:
            if isinstance(m, _mon.Flux):
                line(m.axis, m.position, MONITOR_COLOR, lw=1.2, ls="--")
            elif isinstance(m, _mon.Probe):
                p = np.asarray(m.position, dtype=float)
                ax.plot(p[keep[0]], p[keep[1]], "x", ms=6, color=MONITOR_COLOR, mew=1.5)


def _overlays(ax, sim, show_pml=True, show_sources=True, show_monitors=True, plane=None):
    plt = _plt()
    g = sim.grid
    if sim.ndim == 3:
        return _overlays3(ax, sim, plane, show_pml, show_sources, show_monitors)
    if sim.ndim == 1:
        if show_pml:
            for a, b in enumerate(sim.boundaries):
                if isinstance(b, PML):
                    w = b.cells*g.spacing[0]
                    x0, x1 = g.origin[0], g.origin[0] + g.size[0]
                    ax.axvspan(x0, x0 + w, color=PML_COLOR, alpha=0.2, lw=0)
                    ax.axvspan(x1 - w, x1, color=PML_COLOR, alpha=0.2, lw=0)
        if show_sources:
            for s in sim.sources:
                pos = np.atleast_1d(s.position)[0]
                ax.axvline(pos, color=SOURCE_COLOR, lw=1.5)
        if show_monitors:
            for m in sim.monitors:
                if isinstance(m, (_mon.Flux, _mon.Probe)):
                    ax.axvline(np.atleast_1d(m.position)[0], color=MONITOR_COLOR, lw=1, ls="--")
        return
    x0, x1, y0, y1 = g.extent
    if show_pml:
        for a, b in enumerate(sim.boundaries):
            if not isinstance(b, PML):
                continue
            w = b.cells*g.spacing[a]
            for lo in ((x0, x1 - w) if a == 0 else (y0, y1 - w)):
                if a == 0:
                    rect = plt.Rectangle((lo, y0), w, y1 - y0)
                else:
                    rect = plt.Rectangle((x0, lo), x1 - x0, w)
                rect.set(facecolor="none", edgecolor=PML_COLOR, hatch="///", lw=0, alpha=0.5)
                ax.add_patch(rect)
    if show_sources:
        for s in sim.sources:
            if isinstance(s, _src.PointSource):
                ax.plot(*s.position, "o", ms=6, mec="k", mew=0.6, color=SOURCE_COLOR)
            elif isinstance(s, _src.LineSource):
                lo, hi = s.extent if s.extent is not None else \
                    ((y0, y1) if s.axis == 0 else (x0, x1))
                if s.axis == 0:
                    ax.plot([s.position]*2, [lo, hi], color=SOURCE_COLOR, lw=2)
                else:
                    ax.plot([lo, hi], [s.position]*2, color=SOURCE_COLOR, lw=2)
            elif isinstance(s, _modes.ModeSource):
                lo, hi = s.extent if s.extent is not None else \
                    ((y0, y1) if s.axis == 0 else (x0, x1))
                L = 0.3*(hi - lo)
                if s.axis == 0:
                    ax.plot([s.position]*2, [lo, hi], color=SOURCE_COLOR, lw=2)
                    tip = (s.position + s.direction*L, (lo + hi)/2)
                    tail = (s.position, (lo + hi)/2)
                else:
                    ax.plot([lo, hi], [s.position]*2, color=SOURCE_COLOR, lw=2)
                    tip = ((lo + hi)/2, s.position + s.direction*L)
                    tail = ((lo + hi)/2, s.position)
                ax.annotate("", xy=tip, xytext=tail,
                            arrowprops=dict(arrowstyle="-|>", color=SOURCE_COLOR, lw=1.5))
            elif isinstance(s, _src.TFSF):
                bx0, bx1, by0, by1 = s.box
                ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                           ec=SOURCE_COLOR, lw=1.2, ls="--"))
                ax.annotate("", xy=(bx0 + 0.25*(bx1 - bx0), (by0 + by1)/2),
                            xytext=(bx0, (by0 + by1)/2),
                            arrowprops=dict(arrowstyle="-|>", color=SOURCE_COLOR, lw=1.5))
    if show_monitors:
        for m in sim.monitors:
            if isinstance(m, (_mon.Flux, _modes.ModeMonitor)) or type(m).__name__ == "DiffractionMonitor":
                lo, hi = getattr(m, "extent", None) or ((y0, y1) if m.axis == 0 else (x0, x1))
                if m.axis == 0:
                    ax.plot([m.position]*2, [lo, hi], color=MONITOR_COLOR, lw=1.2, ls="--")
                else:
                    ax.plot([lo, hi], [m.position]*2, color=MONITOR_COLOR, lw=1.2, ls="--")
            elif isinstance(m, _mon.Probe):
                ax.plot(*m.position, "x", ms=6, color=MONITOR_COLOR, mew=1.5)
            elif isinstance(m, (_mon.DFTField, _mon.Snapshots)) and m.region is not None:
                (rx0, rx1), (ry0, ry1) = m.region
                ax.add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                                           ec=MONITOR_COLOR, lw=0.8, ls=":"))


def plot_structure(sim, ax=None, component=None, cmap="Greys", show_pml=True, show_sources=True,
                   show_monitors=True, colorbar=False, periods=1, plane=None):
    """Refractive index map (2D, a section in 3D, sqrt of the permittivity)
    or permittivity profile (1D) with PML, sources and monitors.

    Sources are drawn in amber (points, lines, the TFSF box with an arrow
    for the direction), monitors in blue (flux lines dashed, DFT regions
    dotted, probes as crosses), the PML hatched (2D) or shaded (1D).

    periods : int
        Repeat the cell this many times (odd numbers centre the simulated
        cell) along each periodic or Bloch-periodic axis in the plot, to
        show the structure a unit cell stands for. The simulated cell is
        framed with dashed lines; sources and monitors are drawn in it
        only.
    plane : (axis, position), optional
        3D: the section, ("y", 0.0) by default (see the module docstring).
    """

    ax = _ax(ax)
    c = component or sim.e_components[0]
    eps = sim.permittivity(c)
    coords = sim.coords(c)
    if sim.ndim == 1:
        ax.plot(coords[0], eps, color="k", lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\varepsilon_r$")
        ax.set_xlim(*sim.grid.extent)
    else:
        names = list(AXIS_NAMES[:2])
        shown = [0, 1]
        if sim.ndim == 3:
            a3, _ = _plane(plane)
            eps, coords, names = _section(eps, coords, (a3, _plane(plane)[1]))
            shown = [b for b in range(3) if b != a3]
        n = np.sqrt(np.maximum(eps, 0))
        ext = list(_extent(coords))
        periodic = [sim.boundaries[b] == PERIODIC or isinstance(sim.boundaries[b], Bloch)
                    for b in shown]
        reps = [int(periods) if p else 1 for p in periodic]
        if max(reps) > 1:
            n_t = np.tile(n, reps)
            for a in range(2):
                L, k0 = sim.grid.size[shown[a]], (reps[a] - 1)//2
                ext[2*a] -= k0*L
                ext[2*a + 1] += (reps[a] - 1 - k0)*L
        else:
            n_t = n
        im = ax.imshow(n_t.T, origin="lower", extent=ext, cmap=cmap,
                       vmin=min(1.0, n.min()), vmax=1 + 1.4*max(n.max() - 1, 0.3),
                       interpolation="nearest")
        if max(reps) > 1:
            gext = sim.grid.extent
            gx0, gx1, gy0, gy1 = (gext[2*b + e] for b in shown for e in (0, 1))
            ax.add_patch(_plt().Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec="k",
                                          lw=0.8, ls="--"))
        if colorbar:
            _plt().colorbar(im, ax=ax, label="n")
        _labels(ax, names)
        ax.set_aspect("equal")
    _overlays(ax, sim, show_pml, show_sources, show_monitors, plane=plane)
    return ax


def _resolve(data, component, wavelength_index):
    """Values and coordinates of a field from a Simulation, a DFT monitor,
    DFTFieldData or a SnapshotStack (last frame)."""

    from .simulation import Simulation
    if isinstance(data, Simulation):
        c = component or data.default_component
        F = data.fields[c]
        return (F.real if np.iscomplexobj(F) else F).astype(float), data.coords(c), c
    if isinstance(data, _mon.DFTField):
        data = data.data()
    if isinstance(data, _mon.DFTFieldData):
        c = component or list(data.fields)[0]
        return data.fields[c][wavelength_index], data.coords[c], c
    if isinstance(data, _mon.SnapshotStack):
        return data.frames[-1].astype(float), data.coords, data.component
    raise TypeError("cannot plot %r" % (data,))


def _part(F, part):
    if part == "real":
        return np.real(F)
    if part == "imag":
        return np.imag(F)
    if part == "abs":
        return np.abs(F)
    if part == "intensity":
        return np.abs(F)**2
    raise ValueError("part must be real, imag, abs or intensity")


def plot_field(data, component=None, ax=None, sim=None, part="real", wavelength_index=0,
               vmax=None, log=False, cmap=None, colorbar=True, title=None, plane=None):
    """Plot a field component.

    Parameters
    ----------
    data : Simulation, DFTField, DFTFieldData or SnapshotStack
        A Simulation shows its current (real) fields, a DFT its complex
        amplitudes at wavelength_index, a SnapshotStack its last frame.
    component : str, optional
    sim : Simulation, optional
        Draw the outlines of its structure on top (used automatically if
        data is a Simulation).
    part : "real", "imag", "abs" or "intensity"
    log : bool
        Logarithmic intensity in dB (part "abs" or "intensity").
    plane : (axis, position), optional
        3D: the section (see the module docstring).
    """

    from .simulation import Simulation
    plt = _plt()
    ax = _ax(ax)
    F, coords, c = _resolve(data, component, wavelength_index)
    if isinstance(data, Simulation) and sim is None:
        sim = data
    names = list(AXIS_NAMES[:2])
    if np.ndim(F) == 3:
        if plane is None and 1 in F.shape:
            a = list(F.shape).index(1)
            plane = (a, coords[a][0])
        F, coords, names = _section(F, coords, plane)
    V = _part(F, part)
    signed = part in ("real", "imag")
    if log and not signed:
        V = 10*np.log10(np.maximum(V/np.max(V), 1e-12)) if part == "intensity" else \
            20*np.log10(np.maximum(V/np.max(V), 1e-12))
    if len(coords) == 1:
        ax.plot(coords[0], V, lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel(c)
        if sim is not None:
            _overlays(ax, sim, show_monitors=False)
        return ax
    if cmap is None:
        cmap = FIELD_CMAP if signed else INTENSITY_CMAP
    if vmax is None:
        vmax = np.max(np.abs(V)) if signed else (0 if log else np.max(V))
    vmin = -vmax if signed else (-40 if log else 0)
    im = ax.imshow(V.T, origin="lower", extent=_extent(coords), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    if sim is not None:
        _structure_outline(ax, sim, plane=plane)
    ax.set_xlim(_extent(coords)[:2])
    ax.set_ylim(_extent(coords)[2:])
    _labels(ax, names)
    ax.set_aspect("equal")
    if colorbar:
        label = {"real": "Re ", "imag": "Im ", "abs": "|%s|", "intensity": "|%s|$^2$"}[part]
        label = label % c if "%s" in label else label + c
        plt.colorbar(im, ax=ax, label=label + (" (dB)" if log and not signed else ""), shrink=0.8)
    if title:
        ax.set_title(title)
    return ax


def animate(data, component=None, sim=None, ax=None, frames=24, wavelength_index=0, interval=60,
            vmax=None, cmap=FIELD_CMAP, title=True, plane=None):
    """Animation of a SnapshotStack, or of a DFT field over one period.

    Parameters
    ----------
    data : SnapshotStack, DFTField or DFTFieldData
        A DFT field F is animated as Re{F exp(-i 2 pi k/frames)},
        k = 0..frames - 1 (one period of the steady state).
    sim : Simulation, optional
        Draw the outlines of its structure on top.
    frames : int
        Frames per period (DFT fields only).
    vmax : number, optional
        Colour limit; default: the maximum over all frames (snapshots: the
        99.5 % percentile, so that the source does not saturate the scale).
    plane : (axis, position), optional
        3D: the section (see the module docstring).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """

    plt = _plt()
    from matplotlib.animation import FuncAnimation
    if isinstance(data, _mon.DFTField):
        data = data.data()
    if isinstance(data, _mon.DFTFieldData):
        c = component or list(data.fields)[0]
        F = data.fields[c][wavelength_index]
        coords = data.coords[c]
        names = list(AXIS_NAMES[:2])
        if F.ndim == 3:
            if plane is None and 1 in F.shape:
                a = list(F.shape).index(1)
                plane = (a, coords[a][0])
            F, coords, names = _section(F, coords, plane)
        phases = np.exp(-2j*np.pi*np.arange(frames)/frames)
        get = lambda k: np.real(F*phases[k])
        n = frames
        labels = ["phase %d/%d" % (k, frames) for k in range(frames)]
        if vmax is None:
            vmax = np.max(np.abs(F))
    else:
        c = data.component
        coords = data.coords
        names = list(AXIS_NAMES[:2])
        get = lambda k: data.frames[k].astype(float)
        if data.frames.ndim == 4:
            if plane is None and 1 in data.frames.shape[1:]:
                a = list(data.frames.shape[1:]).index(1)
                plane = (a, coords[a][0])
            coords3 = coords
            _, coords, names = _section(data.frames[0], coords3, plane)
            get = lambda k: _section(data.frames[k].astype(float), coords3, plane)[0]
        n = len(data)
        labels = ["t = %.2f" % t for t in data.times]
        if vmax is None:
            vmax = np.percentile(np.abs(data.frames.astype(float)), 99.5) or 1.0
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if len(coords) == 1:
        (line,) = ax.plot(coords[0], get(0), lw=1.2)
        ax.set_ylim(-1.1*vmax, 1.1*vmax)
        ax.set_xlim(coords[0][0], coords[0][-1])
        ax.set_xlabel("x")
        ax.set_ylabel(c)
        if sim is not None:
            _overlays(ax, sim, show_monitors=False)
        artist = line
        update_data = lambda k: line.set_ydata(get(k))
    else:
        artist = ax.imshow(get(0).T, origin="lower", extent=_extent(coords), cmap=cmap, vmin=-vmax,
                           vmax=vmax, interpolation="bilinear", animated=True)
        if sim is not None:
            _structure_outline(ax, sim, plane=plane)
        ax.set_xlim(_extent(coords)[:2])
        ax.set_ylim(_extent(coords)[2:])
        _labels(ax, names)
        ax.set_aspect("equal")
        update_data = lambda k: artist.set_data(get(k).T)
    txt = ax.set_title(labels[0] if title else "")

    def update(k):
        update_data(k)
        if title:
            txt.set_text(labels[k])
        return artist, txt

    return FuncAnimation(fig, update, frames=n, interval=interval, blit=False)


def gif(anim, fps=12, dpi=None):
    """Render an animation to an animated GIF; returns an IPython Image (the
    notebook stores the GIF itself, typically much smaller than the
    frames of to_jshtml())."""

    import os
    import tempfile
    from IPython.display import Image
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        os.remove(path)
    return Image(data=data, format="gif")


def xt_diagram(stack, ax=None, sim=None, vmax=None, cmap=FIELD_CMAP, colorbar=True):
    """A 1D SnapshotStack as an image over x (horizontal) and t (vertical):
    pulses show up as lines with slope 1/v, reflections as their mirror
    images."""

    plt = _plt()
    ax = _ax(ax)
    V = stack.frames.astype(float)
    if vmax is None:
        vmax = np.percentile(np.abs(V), 99.5) or 1.0
    x = stack.coords[0]
    im = ax.imshow(V, origin="lower", aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=_extent([x, stack.times]), interpolation="nearest")
    if sim is not None:
        eps = sim.permittivity()
        xe = sim.coords(sim.e_components[0])[0]
        edges = xe[1:][np.abs(np.diff(eps)) > 1e-6]
        for e in edges:
            ax.axvline(e, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if colorbar:
        plt.colorbar(im, ax=ax, label=stack.component)
    return ax
