#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Plots for lithography simulations (needs matplotlib).

- plot_source(): source points in the sigma plane with their polarization,
- plot_mask(): mask transmission (one or more periods),
- plot_pupil_fill(): diffraction orders of the mask in the pupil for the
  whole source (the classic "pupil fill" diagram),
- plot_image(): 1D cut with the mask and the threshold, or 2D image with
  feature outlines and the printed contour,
- plot_bossung(): CD vs. focus for several doses,
- plot_process_window(): focus-dose plane with the CD spec contours and
  exposure latitude vs. depth of focus,
- plot_orientation(): orientation fields on the pupil (retardance,
  diattenuation, orientation Zernikes of a Jones pupil).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from .metrics import el_dof, process_window


def _ax(ax):
    return plt.gca() if ax is None else ax


def plot_source(illumination, ax=None, show_polarization=True, every=1, color="C0",
                title=None):
    """Source points (marker area ~ weight) and polarization ticks.

    Polarization ticks show the Jones vector of the dominant state of each
    point (real part of the components); unpolarized points get none.
    """

    ax = _ax(ax)
    ill = illumination
    ax.add_patch(Circle((0, 0), 1, fill=False, ec="gray", lw=1))
    ax.scatter(ill.s_x, ill.s_y, s=12*ill.weights/ill.weights.max(), color=color, alpha=0.5)
    if show_polarization:
        sel = slice(None, None, every)
        u, v, X, Y = [], [], [], []
        for s_x, s_y in zip(ill.s_x[sel], ill.s_y[sel]):
            states = ill.polarization_states(s_x, s_y)
            w, J = max(states, key=lambda st: st[0])
            if len(states) > 1 and abs(states[0][0] - states[1][0]) < 1E-9:
                continue  # unpolarized
            J = J*np.exp(-1j*np.angle(J[np.argmax(np.abs(J))]))
            X.append(s_x), Y.append(s_y), u.append(np.real(J[0])), v.append(np.real(J[1]))
        if X:
            ax.quiver(X, Y, u, v, pivot="middle", headwidth=1, headlength=0, headaxislength=0,
                      width=0.005, scale=28, color="k")
    ax.set_xlim(-1.08, 1.08), ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("$s_x$"), ax.set_ylabel("$s_y$")
    ax.set_title(ill.name if title is None else title, fontsize=10)

    return ax


def plot_mask(layout, ax=None, periods=2, N=400, t_absorber=0.0, cmap="gray", title=None):
    """Transmission amplitude of the layout over periods x periods."""

    ax = _ax(ax)
    Px = layout.pitch_x or layout.pitch_y
    Py = layout.pitch_y or layout.pitch_x
    x = np.linspace(-periods*Px/2, periods*Px/2, N)
    y = np.linspace(-periods*Py/2, periods*Py/2, N)
    t = layout.transmission(x, y, 1.0, t_absorber)
    ax.imshow(np.abs(t), extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap=cmap,
              vmin=0, vmax=1)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title(layout.name if title is None else title, fontsize=10)

    return ax


def plot_pupil_fill(sim, ax=None, max_points=2000, title=None):
    """Positions of all diffraction orders that pass the pupil for all
    source points (normalized pupil coordinates), colored by order, marker
    area ~ ``|c|**2*source`` weight."""

    ax = _ax(ax)
    P = sim.projector
    kNA = P.k0*P.NA
    m, n, f_x, f_y = sim.orders()
    c = sim.layout.region_coefficients(m, n)
    # orders that carry light: those of the features, and order 0
    amp = np.abs(c["feature"]) + np.outer(n == 0, m == 0)
    M, N_ = np.meshgrid(m, n)
    ill = sim.illumination
    step = max(1, len(ill)//max_points)
    ax.add_patch(Circle((0, 0), 1, fill=True, fc="C1", alpha=0.12, ec="k", lw=1.5))
    cmap = plt.get_cmap("tab10")
    for s_x, s_y, w in zip(ill.s_x[::step], ill.s_y[::step], ill.weights[::step]):
        P_x, P_y = f_x[None, :]/kNA + s_x, f_y[:, None]/kNA + s_y
        P_x, P_y = np.broadcast_to(P_x, amp.shape), np.broadcast_to(P_y, amp.shape)
        ok = (P_x**2 + P_y**2 <= 1) & (amp > 1E-6)
        ax.scatter(P_x[ok], P_y[ok], s=4, color=[cmap((abs(a) + abs(b)) % 10) for a, b in
                                               zip(M[ok], N_[ok])], alpha=0.6, lw=0)
    ax.set_xlim(-1.08, 1.08), ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("$k_x/(k_0$NA)"), ax.set_ylabel("$k_y/(k_0$NA)")
    ax.set_title("pupil fill" if title is None else title, fontsize=10)

    return ax


def plot_image(sim, x, I, y=None, ax=None, threshold=None, dose=1.0, show_mask=True, label=None,
               cmap="inferno", colorbar=True, **kwargs):
    """Plot a 1D image (with shaded features and the threshold) or a 2D
    image (with feature outlines and the printed contour at
    threshold/dose)."""

    ax = _ax(ax)
    L = sim.layout
    if y is None:
        line, = ax.plot(x, I, label=label, **kwargs)
        if show_mask and L.pitch_x is not None:
            x0, _, a, _ = L.features[0]
            k_min, k_max = np.floor((x[0] - x0)/L.pitch_x) - 1, np.ceil((x[-1] - x0)/L.pitch_x) + 1
            for k in np.arange(k_min, k_max + 1):
                for xf, _, af, _ in L.features:
                    ax.axvspan(xf + k*L.pitch_x - af/2, xf + k*L.pitch_x + af/2, color="gray",
                               alpha=0.12, lw=0)
            ax.set_xlim(x[0], x[-1])
        if threshold is not None:
            ax.axhline(threshold/dose, color="gray", ls="--", lw=1)
        ax.set_xlabel("x"), ax.set_ylabel("intensity")
        return line

    im = ax.imshow(I, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap=cmap, **kwargs)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)
    if show_mask:
        Px, Py = L.pitch_x or np.inf, L.pitch_y or np.inf
        for xf, yf, a, b in L.features:
            for i in range(-3, 4):
                for j in range(-3, 4):
                    xc, yc = xf + i*Px, yf + j*Py
                    if x[0] - a < xc < x[-1] + a and y[0] - b < yc < y[-1] + b:
                        ax.add_patch(Rectangle((xc - a/2, yc - b/2), a, b, fill=False, ec="w",
                                               lw=0.8, ls=":"))
    if threshold is not None:
        ax.contour(x, y, I, [threshold/dose], colors="c", linewidths=1.2)
    ax.set_xlim(x[0], x[-1]), ax.set_ylim(y[0], y[-1])
    ax.set_xlabel("x"), ax.set_ylabel("y")

    return im


def plot_bossung(foci, doses, cd, ax=None, cd_target=None, tolerance=0.1, cmap="viridis",
                 every=1, colorbar=True):
    """Bossung curves: CD vs. focus, one curve per dose (colored by dose,
    with a colorbar)."""

    ax = _ax(ax)
    cm = plt.get_cmap(cmap)
    norm = plt.Normalize(np.min(doses), np.max(doses))
    for i in range(0, len(doses), every):
        ax.plot(foci, cd[i], color=cm(norm(doses[i])), lw=1.5)
    if cd_target is not None:
        ax.axhline(cd_target, color="k", lw=1)
        ax.axhspan(cd_target*(1 - tolerance), cd_target*(1 + tolerance), color="gray", alpha=0.15)
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        plt.colorbar(sm, ax=ax, label="dose")
    ax.set_xlabel("focus"), ax.set_ylabel("CD")

    return ax


def plot_process_window(foci, doses, cd, cd_target, tolerance=0.1, axes=None, el_min=None,
                        color="C0", label=None):
    """Focus-dose process window (CD spec contours and in-spec doses) and
    exposure latitude vs. depth of focus.

    Returns
    -------
    dof, el : arrays (see metrics.el_dof())
    """

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    low, high = process_window(foci, doses, cd, cd_target, tolerance)
    ax = axes[0]
    F, D = np.meshgrid(foci, doses)
    cd_ = np.where(np.isnan(cd), -1.0, cd)
    ax.contour(F, D, cd_, [cd_target*(1 - tolerance), cd_target*(1 + tolerance)], colors=color,
               linestyles=["--", "-"], linewidths=1)
    ax.fill_between(foci, low, high, color=color, alpha=0.2, label=label)
    ax.set_xlabel("focus"), ax.set_ylabel("dose")
    ax.set_title("CD within ±{:g} %".format(100*tolerance), fontsize=10)

    dof, el = el_dof(foci, low, high)
    axes[1].plot(dof, 100*el, color=color, lw=2, label=label)
    if el_min is not None:
        axes[1].axhline(100*el_min, color="gray", ls=":")
    axes[1].set_xlabel("depth of focus"), axes[1].set_ylabel("exposure latitude [%]")
    if len(el):
        top = max(100*np.max(el)*1.05, axes[1].get_ylim()[1] if len(axes[1].lines) > 1 else 0)
        axes[1].set_ylim(0, top)

    return dof, el


def plot_orientation(orientator, ax=None, N=41, n_ticks=13, scale=None, cmap="viridis", title=None,
                     colorbar=True, label=None):
    """Orientation field on the pupil, such as a retardance or diattenuation
    (see jones_pupil): color = magnitude, ticks along the orientation psi
    (``o = |o|*exp(2*i*psi)``) with lengths ~ magnitude.

    Parameters
    ----------
    orientator : callable or tuple
        orientator(P_x, P_y) returning o_1 + i*o_2, e.g.
        lambda P_x, P_y: jp.orientation_zernike(5, P_x, P_y); or samples
        (P_x, P_y, o) on scattered pupil points.
    N, n_ticks : int, optional
        Grid for the color map and the number of ticks across the pupil.
    scale : number, optional
        Magnitude of a full length tick; default: the maximum.
    """

    ax = _ax(ax)
    p = np.linspace(-1, 1, N)
    q = np.linspace(-1, 1, n_ticks)
    if callable(orientator):
        PX, PY = np.meshgrid(p, p)
        QX, QY = np.meshgrid(q, q)
        o, o_t = orientator(PX, PY), orientator(QX, QY)
        o = np.where(PX**2 + PY**2 <= 1, o, np.nan)
        im = ax.pcolormesh(p, p, np.abs(o), cmap=cmap, shading="auto")
    else:
        from scipy.interpolate import griddata
        P_x, P_y, values = orientator
        QX, QY = np.meshgrid(q, q)
        o_t = griddata((P_x, P_y), values, (QX, QY))
        im = ax.tripcolor(P_x, P_y, np.abs(values), cmap=cmap, shading="gouraud")
    inside = (QX**2 + QY**2 <= 1) & np.isfinite(o_t)
    a = np.abs(o_t[inside])
    scale = (np.max(a) if a.size and np.max(a) > 0 else 1.0) if scale is None else scale
    psi = np.angle(o_t[inside])/2
    length = 0.9*(q[1] - q[0])*a/scale
    ax.quiver(QX[inside], QY[inside], length*np.cos(psi), length*np.sin(psi), angles="xy",
              scale_units="xy", scale=1, pivot="middle", headwidth=0, headlength=0,
              headaxislength=0, width=0.006, color="w")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec="k", lw=1))
    ax.set_xlim(-1.05, 1.05), ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=10)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label=label)

    return ax
