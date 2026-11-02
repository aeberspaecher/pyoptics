#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Plots and animations of FDTD simulations (needs matplotlib).

- plot_structure(): permittivity map with the PML, sources and monitors.
- plot_field(): a component of the current fields, of a DFT monitor or of
  a snapshot, with the outlines of the structure on top.
- animate(): a matplotlib animation of a SnapshotStack (time domain) or of
  a DFT field (the steady state Re{F exp(-i omega t)} over one period,
  which needs no stored frames at all). In notebooks, show it with
  IPython.display.HTML(anim.to_jshtml()); save it with anim.save().
- xt_diagram(): a 1D run as an image over x and t.

Real fields use a diverging colormap with symmetric limits, intensities
inferno. The colours of sources and monitors are those of pyoptics.sketch.
2D arrays are indexed [ix, iy] and shown with x to the right.
"""

import numpy as np

from .grid import OFFSETS
from . import sources as _src
from . import monitors as _mon
from .boundaries import PML


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


def _structure_outline(ax, sim, region=None, color="k", lw=0.6, alpha=0.6):
    """Contours of the permittivity on top of a field plot."""

    c = sim.e_components[0]
    eps = sim.permittivity(c)
    if np.ptp(eps) == 0:
        return
    x, y = sim.coords(c)
    levels = np.unique(np.round(eps, 6))
    if len(levels) > 6:
        levels = np.linspace(eps.min(), eps.max(), 4)[1:-1]
    else:
        levels = (levels[:-1] + levels[1:])/2
    ax.contour(x, y, eps.T, levels=levels, colors=color, linewidths=lw, alpha=alpha)


def _overlays(ax, sim, show_pml=True, show_sources=True, show_monitors=True):
    plt = _plt()
    g = sim.grid
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
            elif isinstance(s, _src.TFSF):
                bx0, bx1, by0, by1 = s.box
                ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                           ec=SOURCE_COLOR, lw=1.2, ls="--"))
                ax.annotate("", xy=(bx0 + 0.25*(bx1 - bx0), (by0 + by1)/2),
                            xytext=(bx0, (by0 + by1)/2),
                            arrowprops=dict(arrowstyle="-|>", color=SOURCE_COLOR, lw=1.5))
    if show_monitors:
        for m in sim.monitors:
            if isinstance(m, _mon.Flux):
                lo, hi = m.extent if m.extent is not None else \
                    ((y0, y1) if m.axis == 0 else (x0, x1))
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
                   show_monitors=True, colorbar=False):
    """Permittivity map (2D) or profile (1D) with PML, sources and monitors.

    Sources are drawn in amber (points, lines, the TFSF box with an arrow
    for the direction), monitors in blue (flux lines dashed, DFT regions
    dotted, probes as crosses), the PML hatched (2D) or shaded (1D).
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
        im = ax.imshow(eps.T, origin="lower", extent=_extent(coords), cmap=cmap,
                       vmin=min(1.0, eps.min()), vmax=1 + 2*max(eps.max() - 1, 0.5),
                       interpolation="nearest")
        if colorbar:
            _plt().colorbar(im, ax=ax, label=r"$\varepsilon_r$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    _overlays(ax, sim, show_pml, show_sources, show_monitors)
    return ax


def _resolve(data, component, wavelength_index):
    """Values and coordinates of a field from a Simulation, a DFT monitor,
    DFTFieldData or a SnapshotStack (last frame)."""

    from .simulation import Simulation
    if isinstance(data, Simulation):
        c = component or data.e_components[0] if data.mode == "Ez" else component or "Hz"
        return data.fields[c].astype(float), data.coords(c), c
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
               vmax=None, log=False, cmap=None, colorbar=True, title=None):
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
    """

    from .simulation import Simulation
    plt = _plt()
    ax = _ax(ax)
    F, coords, c = _resolve(data, component, wavelength_index)
    if isinstance(data, Simulation) and sim is None:
        sim = data
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
        _structure_outline(ax, sim)
    ax.set_xlim(_extent(coords)[:2])
    ax.set_ylim(_extent(coords)[2:])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if colorbar:
        label = {"real": "Re ", "imag": "Im ", "abs": "|%s|", "intensity": "|%s|$^2$"}[part]
        label = label % c if "%s" in label else label + c
        plt.colorbar(im, ax=ax, label=label + (" (dB)" if log and not signed else ""), shrink=0.8)
    if title:
        ax.set_title(title)
    return ax


def animate(data, component=None, sim=None, ax=None, frames=24, wavelength_index=0, interval=60,
            vmax=None, cmap=FIELD_CMAP, title=True):
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
        phases = np.exp(-2j*np.pi*np.arange(frames)/frames)
        get = lambda k: np.real(F*phases[k])
        n = frames
        labels = ["phase %d/%d" % (k, frames) for k in range(frames)]
        if vmax is None:
            vmax = np.max(np.abs(F))
    else:
        c = data.component
        coords = data.coords
        get = lambda k: data.frames[k].astype(float)
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
            _structure_outline(ax, sim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        update_data = lambda k: artist.set_data(get(k).T)
    txt = ax.set_title(labels[0] if title else "")

    def update(k):
        update_data(k)
        if title:
            txt.set_text(labels[k])
        return artist, txt

    return FuncAnimation(fig, update, frames=n, interval=interval, blit=False)


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
