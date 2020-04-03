#!/usr/bin/env python
#-*- coding:utf-8 -*-


import matplotlib as mpl
mpl.rcParams["font.size"] = 16
mpl.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

from .utils import I, ensure_meshgrid


default_cmap = plt.cm.inferno
default_cmap_phase = plt.cm.bwr

def plot_intensity(psi, x, y, xlabel=None, ylabel=None, title=None, colorbar=True,
                   use_intensity=False, n=1.0, ax=None, **kwargs):
    """Plot an intensity image.

    Parameters
    ----------
    psi : array
        Complex field or intensity image. psi is considered to be an intensity
        already is if is of complex type, otherwise it's considered to be an
        in intensity.
    x, y : arrays
        Linear coordinate arrays the image is sampled on.
    xlabel, ylabel, title : strings, optional
        Matplotlib options.
    colorbar : boolean, optional
        If True, a colorbar is added.
    use_intensity : boolean, optional
        If True, use physical intensity for the plot. Otherwise,
        use abs(psi)**2 [default].
    n : double, optional
        Refractive index of embedding medium. Defaults to n=1.0. Used if psi is
        complex.
    ax : matplotlib axes, optional
         If given, use this axes to plot to.
    **kwargs : keyword arguments
        Propagated to matplotlib imshow() call.
    """

    if np.iscomplexobj(psi):
        psi = I(psi, n) if use_intensity else np.abs(psi)**2

    if "cmap" not in kwargs:
        kwargs["cmap"] = default_cmap
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"

    if ax is None:
        fig = plt.figure(facecolor="white")
        ax = plt.gca()
    else:
        fig = ax.figure

    ax.locator_params(nbins=5)

    im = ax.imshow(psi, extent=(x[0], x[-1], y[0], y[-1]), **kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if colorbar is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # ax.colorbar()
    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_field(field, x, y, xlabel=None, ylabel=None, title=None, colorbar=True,
               use_intensity=False, amp_title=None, phase_title=None,
               horizontal_layout=True,
               use_rad=False, n=1.0, **kwargs):
    """Plot a complex field.

    Parameters
    ----------
    psi : array
        Complex field.
    x, y : arrays
        Linear coordinate arrays the field is sampled on.
    xlabel, ylabel, title : strings, optional
        Matplotlib options.
    colorbar : boolean, optional
        If True, a colorbar is added.
    use_intensity : boolean, optional
        If True, use intensity for the amplitude plot. Otherwise,
        use abs(psi) [default].
    n : double, optional
        Refractive index of embedding medium. Defaults to n=1.0. Used if psi is
        complex.
    **kwargs : keyword arguments
        Propagated to matplotlib imshow() call. If given, phase_cmap is mapped
        to cmap for the phase plot.
    """

    # FIXME: aspect ratio & extent - images can be truncated!
    # TODO: make x, y optionally linear or 2d arrays input-wise; make internal use of ensure_meshgrid()

    fig = plt.figure(facecolor="white")

    if colorbar and horizontal_layout:
        gs = gridspec.GridSpec(1, 2,
                               #width_ratios=[1, 1],
                               #height_ratios=[1],
                               wspace=0, hspace=0
                              )

        amp_ax = fig.add_subplot(gs[0])
        amp_div = make_axes_locatable(amp_ax)
        amp_cbar_ax = amp_div.append_axes("right", size="10%", pad=0.05)

        phase_ax = fig.add_subplot(gs[1], sharey=amp_ax)
        phase_div = make_axes_locatable(phase_ax)
        phase_cbar_ax = phase_div.append_axes("right", size="10%", pad=0.05)
        plt.setp(phase_ax.get_yticklabels(), visible=False)
    if (not colorbar) and horizontal_layout:
        gs = gridspec.GridSpec(1, 2)
        amp_ax = fig.add_subplot(gs[0, 0])
        phase_ax = fig.add_subplot(gs[0, 1], sharey=amp_ax)
        plt.setp(phase_ax.get_yticklabels(), visible=False)
    if colorbar and (not horizontal_layout):
        gs = gridspec.GridSpec(2, 1)
        amp_ax = fig.add_subplot(gs[0, 0])
        phase_ax = fig.add_subplot(gs[1, 0], sharex=amp_ax)
        amp_div = make_axes_locatable(amp_ax)
        amp_cbar_ax = amp_div.append_axes("right", size="10%", pad=0.05)
        phase_div = make_axes_locatable(phase_ax)
        phase_cbar_ax = phase_div.append_axes("right", size="10%", pad=0.05)
        plt.setp(amp_ax.get_xticklabels(), visible=False)
    if (not colorbar) and (not horizontal_layout):
        gs = gridspec.GridSpec(2, 1)
        amp_ax = fig.add_subplot(gs[0, 0])
        phase_ax = fig.add_subplot(gs[1, 0], sharex=amp_ax)
        plt.setp(amp_ax.get_xticklabels(), visible=False)

    if not np.iscomplexobj(field):
        raise ValueError("psi must be complex-valued")

    amp_ax.locator_params(nbins=5)
    phase_ax.locator_params(nbins=5)

    amp = I(field, n) if use_intensity else np.abs(field)
    phase = np.angle(field) if use_rad else np.angle(field)/np.pi

    if "cmap" not in kwargs:
        kwargs["cmap"] = default_cmap

    im1 = amp_ax.imshow(amp, extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
                        aspect="equal", **kwargs)

    if "phase_cmap" not in kwargs:
        kwargs["cmap"] = default_cmap_phase
    else:
        kwargs["cmap"] = kwargs.pop("phase_cmap")

    im2 = phase_ax.imshow(phase, extent=(x[0], x[-1], y[0], y[-1]), aspect="equal",
                          origin="lower", **kwargs)

    if amp_title is not None:
        amp_ax.set_title(amp_title)
    if phase_title is not None:
        phase_ax.set_title(phase_title)

    if xlabel is not None:
        if horizontal_layout:
            amp_ax.set_xlabel(xlabel)
        phase_ax.set_xlabel(xlabel)
    if ylabel is not None:
        if (not horizontal_layout):
            phase_ax.set_ylabel(ylabel)
        amp_ax.set_ylabel(ylabel)

    if title is not None:
        plt.suptitle(title)
    if colorbar:
        plt.colorbar(im1, cax=amp_cbar_ax, use_gridspec=True)
        plt.colorbar(im2, cax=phase_cbar_ax, use_gridspec=True)

    # use full phase range for colormap:
    if use_rad:
        im2.set_clim((-np.pi, +np.pi))
    else:
        im2.set_clim((-1, +1))

    amp_ax.set(adjustable='box-forced', aspect="equal")
    phase_ax.set(adjustable='box-forced', aspect="equal")

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    plt.show()


def plot_imstack_alternative(imstack, x, y, z_vals, cmap=plt.cm.Blues, normalize=False):
    """Plot an image stack.

    Parameters
    ----------
    imstack : array
       Array of dimension (Nz, Nx, Ny).
    x, y : arrays
        Linear or 2d coordinate arrays.
    z_vals : array
    cmap : MPL colormap, optional
        Colormap to use. Defaults to cm.Blues.
    normalize : boolean, optional
        If True, normalize each image slice individually to [0, 1]. Use this if
        the individual images differ a lot in maximum intensity. Defaults to
        False.
    """

    # TODO: implement ax option

    X, Y = ensure_meshgrid(x, y)
    Z = np.zeros_like(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # cmp. some StackOverflow post about alpha and colormaps:
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # normalize whole image stack to [0, 1]:
    imstack_mapped = imstack - np.min(imstack)
    imstack_mapped = imstack_mapped/np.max(imstack_mapped)

    for z, I in zip(z_vals, imstack_mapped):
        if normalize:
            I /= np.max(I)
        ax.plot_surface(X, Y, Z+z, rstride=1, cstride=1,
                        facecolors=my_cmap(I), shade=False)
        # ax.contourf(X, Y, I, zdir='z')
        # plt.imshow(np.abs(field)**2); plt.show()

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_imstack(imstack, x, y, z_vals, cmap=plt.cm.Blues,
                             normalize=False, alpha=0.5):
    """Plot an image stack.

    Parameters
    ----------
    imstack : array
       Array of dimension (Nz, Nx, Ny).
    x, y : arrays
        Linear or 2d coordinate arrays.
    z_vals : array
    cmap : MPL colormap, optional
        Colormap to use. Defaults to cm.Blues.
    normalize : boolean, optional
        If True, normalize each image slice individually to [0, 1]. Use this if
        the individual images differ a lot in maximum intensity. Defaults to
        False.
    alpha : float, optional
    """

    # TODO: implement ax option

    X, Y = ensure_meshgrid(x, y)
    Z = np.zeros_like(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # cmp. some StackOverflow post about alpha and colormaps:
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # normalize whole image stack to [0, 1]:
    imstack_mapped = imstack - np.min(imstack)
    imstack_mapped = imstack_mapped/np.max(imstack_mapped)

    # levels = np.linspace(imstack_mapped.min(), imstack_mapped.max(), 50)
    levels = np.linspace(0, 1, 40)*(z_vals.max() - z_vals.min())

    for z, I in zip(z_vals, imstack_mapped):
        if normalize:
            I /= np.max(I)
            ax.contourf(X, Y, I, zdir="z", cmap=cmap, offset=z, alpha=alpha,)

    ax.set_zlim((z_vals.min(), z_vals.max()))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
if(__name__ == '__main__'):
    import numpy as np
    from pyoptics.beams import gauss_laguerre, gauss_hermite
    from pyoptics.utils import grid1d
    wl = 0.630
    x, y = grid1d(-2, +2, 512, False), grid1d(-2, 2, 512, False)
    psi = gauss_laguerre(5, 10, x, y, z=-0.01, w_0=0.5, z_r=1.1, wl=wl)

    plot_intensity(psi, x, y)
    plot_field(psi, x, y, title="Test", amp_title="Intensity",
               phase_title=r"Phase [$\pi$]", use_rad=False,
               xlabel="$x$ [arb. units]", ylabel="$y$ [nm]",
               colorbar=True, horizontal_layout=True, use_intensity=True)
