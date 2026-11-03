#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Optical vortices: helical phases, topological charge and orbital angular
momentum (OAM) of scalar paraxial beams.

A field u(r, phi) = A(r)*exp(i*l*phi) has a helical wavefront and a phase
singularity on the axis, where the amplitude vanishes. The integer l is the
topological charge: the number of times the phase winds through 2*pi on a
loop around the singularity, counted positive for an anticlockwise winding
(x to the right, y up, looking against the direction of propagation, i.e.
the beam comes towards the viewer). With the pyoptics convention
exp(i(k.r - omega*t)) a beam with exp(+i*l*phi) carries the orbital angular
momentum l*hbar per photon along +z (Allen et al. 1992), like the modes of
`modes.lg_mode()`.

Elements that make vortices:

- spiral_phase_plate(): the transmission exp(i*l*phi) of a spiral phase
  plate, optionally as a staircase of N levels or with a fractional charge.
- spiral_plate_height(): the height profile l*wavelength*phi/(2*pi*(n - 1))
  of a refractive plate.
- fork_grating(): the hologram of a vortex with a tilted plane wave, a
  grating with a fork of l extra lines at its centre (binary or sinusoidal
  amplitude, blazed or binary phase). The diffraction order m carries the
  charge m*l.

Analysis:

- topological_charge(): the winding number of the phase on a circle.
- phase_singularities(): all vortices of a sampled field, with positions
  and signed charges (winding around each 2x2 cell of the grid).
- oam_spectrum(): the spiral spectrum, the share P_l of the power in each
  azimuthal harmonic exp(i*l*phi) about a chosen axis.
- mean_oam(): the OAM per photon in units of hbar,
  ``<L_z> = Im(int conj(u)*d(u)/d(phi) dA)/int |u|**2 dA``, about a chosen axis.
- orbital_current(): the transverse orbital current density
  ``Im(conj(u)*grad(u))``, proportional to the transverse Poynting vector of
  the paraxial beam (without the spin part).

References
----------
L. Allen, M. W. Beijersbergen, R. J. C. Spreeuw and J. P. Woerdman, "Orbital
angular momentum of light and the transformation of Laguerre-Gaussian laser
modes", Phys. Rev. A 45, 8185-8189 (1992).

N. R. Heckenberg, R. McDuff, C. P. Smith and A. G. White, "Generation of
optical phase singularities by computer-generated holograms", Opt. Lett. 17,
221-223 (1992).

M. W. Beijersbergen, R. P. C. Coerwinkel, M. Kristensen and J. P. Woerdman,
"Helical-wavefront laser beams produced with a spiral phaseplate", Opt.
Commun. 112, 321-327 (1994).

L. Torner, J. P. Torres and S. Carrasco, "Digital spiral imaging", Opt.
Express 13, 873-881 (2005).

J. B. Götte, K. O'Holleran, D. Preece, F. Flossmann, S. Franke-Arnold, S. M.
Barnett and M. J. Padgett, "Light beams with fractional orbital angular
momentum and their vortex structure", Opt. Express 16, 993-1006 (2008).
"""

import numpy as np

from .utils import ensure_meshgrid, get_dx


__all__ = ["spiral_phase_plate", "spiral_plate_height", "fork_grating", "topological_charge",
           "phase_singularities", "oam_spectrum", "mean_oam", "orbital_current"]


def _azimuth(x, y, x0, y0, angle=0.0):
    # azimuth in [0, 2*pi), measured from the direction `angle`
    X, Y = ensure_meshgrid(x, y)
    return np.mod(np.arctan2(Y - y0, X - x0) - angle, 2*np.pi)


def spiral_phase_plate(x, y, l, levels=None, x0=0.0, y0=0.0, angle=0.0):
    """Transmission exp(i*l*phi) of a spiral phase plate.

    Parameters
    ----------
    x, y : arrays
        Linear or meshed coordinates.
    l : number
        Topological charge. A non-integer l gives a plate whose phase step
        2*pi*l at the edge (along the direction `angle`) is not a multiple
        of 2*pi: a fractional vortex.
    levels : int, optional
        If given, the phase is a staircase of `levels` steps of 2*pi*l/levels
        each (a multilevel plate). Defaults to a continuous ramp.
    x0, y0 : numbers, optional
        Centre of the plate.
    angle : number, optional
        Direction of the edge (phase step) in rad from the x axis.

    Returns
    -------
    t : 2d array
        Complex transmission.
    """

    phi = _azimuth(x, y, x0, y0, angle)
    if levels is not None:
        phi = 2*np.pi/levels*np.floor(levels*phi/(2*np.pi))

    return np.exp(1j*l*phi)


def spiral_plate_height(x, y, l, wavelength, n, x0=0.0, y0=0.0, angle=0.0, n_medium=1.0):
    """Height profile of a refractive spiral phase plate of charge l.

    The height rises linearly with the azimuth to
    ``l*wavelength/(n - n_medium)`` at the edge.

    Parameters
    ----------
    x, y : arrays
    l : number
        Design charge.
    wavelength : number
        Design (vacuum) wavelength.
    n : number
        Refractive index of the plate at the design wavelength.
    x0, y0, angle : numbers, optional
        Centre and direction of the edge, as in spiral_phase_plate().
    n_medium : number, optional
        Index of the surrounding medium.

    Returns
    -------
    h : 2d array
        Height in the units of the wavelength.
    """

    phi = _azimuth(x, y, x0, y0, angle)

    return l*wavelength*phi/(2*np.pi*(n - n_medium))


def fork_grating(x, y, l, period, kind="binary", x0=0.0, y0=0.0, angle=0.0):
    """Fork grating: the hologram of a vortex of charge l with a tilted plane
    wave.

    The grating phase is ``psi = 2*pi*s/period + l*phi`` with s the
    coordinate along the grating vector (direction `angle`). The diffraction
    order m leaves in the direction ``m*wavelength/period`` and carries the
    charge m*l.

    Parameters
    ----------
    x, y : arrays
        Linear or meshed coordinates.
    l : int
        Charge of the first diffraction order (the number of extra lines of
        the fork).
    period : number
        Grating period far from the fork.
    kind : {"binary", "sinusoidal", "blazed", "binary_phase"}, optional
        "binary" (default): amplitude 0 or 1 where cos(psi) < 0 or > 0, as
        printed on film (Heckenberg et al. 1992): 25 % of the power in the
        zero order, 1/pi**2 = 10.1 % in each first order.
        "sinusoidal": amplitude (1 + cos(psi))/2, orders 0 and +-1 only.
        "blazed": phase mod(psi, 2*pi), all power in the order +1.
        "binary_phase": phase 0 or pi, 4/pi**2 = 40.5 % in each first order
        and no zero order.
    x0, y0 : numbers, optional
        Position of the fork.
    angle : number, optional
        Direction of the grating vector in rad from the x axis.

    Returns
    -------
    t : 2d array
        Complex transmission.
    """

    X, Y = ensure_meshgrid(x, y)
    s = (X - x0)*np.cos(angle) + (Y - y0)*np.sin(angle)
    psi = 2*np.pi*s/period + l*np.arctan2(Y - y0, X - x0)
    if kind == "binary":
        return (np.cos(psi) > 0).astype(complex)
    if kind == "sinusoidal":
        return (0.5*(1 + np.cos(psi))).astype(complex)
    if kind == "blazed":
        return np.exp(1j*np.mod(psi, 2*np.pi))
    if kind == "binary_phase":
        return np.where(np.cos(psi) > 0, 1.0, -1.0).astype(complex)
    raise ValueError("kind must be 'binary', 'sinusoidal', 'blazed' or 'binary_phase'")


def _interpolate(field, x, y, xs, ys, order=3):
    # complex field at the points (xs, ys), cubic spline interpolation
    from scipy.ndimage import map_coordinates

    x = np.asarray(x)
    y = np.asarray(y)
    x = x[0] if x.ndim == 2 else x
    y = y[:, 0] if y.ndim == 2 else y
    if x.ndim == 2 or y.ndim == 2:
        raise ValueError("x and y must be linear or meshed coordinates")
    j = (np.asarray(xs) - x[0])/(x[1] - x[0])
    i = (np.asarray(ys) - y[0])/(y[1] - y[0])
    coords = np.array([i.ravel(), j.ravel()])
    re = map_coordinates(field.real, coords, order=order, mode="constant")
    im = map_coordinates(field.imag, coords, order=order, mode="constant")

    return (re + 1j*im).reshape(np.shape(xs))


def _linear(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (x[0] if x.ndim == 2 else x), (y[:, 0] if y.ndim == 2 else y)


def topological_charge(field, x, y, radius, x0=0.0, y0=0.0, n=720):
    """Winding number of the phase of a field on a circle.

    ``(1/(2*pi))*sum of the phase increments`` along the circle, anticlockwise.
    It is the total charge of all vortices inside the circle. The field must
    not vanish on the circle and its phase must change by less than pi
    between neighbouring points.

    Parameters
    ----------
    field : 2d array
    x, y : arrays
        Linear or meshed equidistant coordinates.
    radius : number
        Radius of the circle.
    x0, y0 : numbers, optional
        Centre of the circle.
    n : int, optional
        Number of points on the circle.

    Returns
    -------
    charge : float
        Close to an integer if the circle is well sampled.
    """

    x, y = _linear(x, y)
    t = 2*np.pi*np.arange(n + 1)/n
    u = _interpolate(field, x, y, x0 + radius*np.cos(t), y0 + radius*np.sin(t))

    return float(np.sum(np.angle(u[1:]/u[:-1]))/(2*np.pi))


def phase_singularities(field, x, y, threshold=1E-14, mask=None):
    """Find the phase singularities (vortices) of a sampled field.

    The phase increments around each cell of four neighbouring samples are
    summed anticlockwise (x to the right, y up). A sum of +-2*pi marks a
    vortex of charge +-1 inside the cell. A vortex that sits exactly on a
    sample (an analytic zero, u == 0) is found by winding around that sample
    through its eight neighbours. A degenerate vortex of charge l can show
    up as one cell of charge l or as several cells close together; the total
    charge in a region is right in either case. A line of zeros, such as
    the dark rings of the modes LG_pl with p > 0, is not generic: the
    field changes sign across it, and the cells on the line show pairs of
    opposite charges that add up to zero (exclude them with `mask`).

    Parameters
    ----------
    field : 2d array
    x, y : arrays
        Linear or meshed equidistant coordinates.
    threshold : number, optional
        Cells whose largest corner intensity is below threshold times the
        largest intensity of the field are skipped; there the phase is noise.
        The core of a vortex of charge l is dark as ``r**(2*|l|)``, so the
        default is low; raise it for noisy (e.g. measured) fields.
    mask : 2d bool array, optional
        Cells to search (shape of the field minus one in both axes, or the
        shape of the field, then the cell takes the value of its lower left
        corner).

    Returns
    -------
    xs, ys : 1d arrays
        Positions of the cell centres.
    charges : 1d int array
        Signed charges.
    """

    x, y = _linear(x, y)
    u = np.asarray(field)
    I = np.abs(u)**2
    I_min = threshold*I.max()
    a, b, c, d = u[:-1, :-1], u[:-1, 1:], u[1:, 1:], u[1:, :-1]
    # increments as arguments of products, which stay finite at exact zeros
    winding = (np.angle(b*np.conj(a)) + np.angle(c*np.conj(b)) + np.angle(d*np.conj(c))
               + np.angle(a*np.conj(d)))/(2*np.pi)
    charge = np.rint(winding).astype(int)
    I_corner = np.maximum.reduce([np.abs(a)**2, np.abs(b)**2, np.abs(c)**2, np.abs(d)**2])
    zero = u == 0
    touches_zero = zero[:-1, :-1] | zero[:-1, 1:] | zero[1:, 1:] | zero[1:, :-1]
    keep = (charge != 0) & (I_corner > I_min) & ~touches_zero
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == u.shape:
            mask = mask[:-1, :-1]
        keep &= mask
    i, j = np.nonzero(keep)
    xc = 0.5*(x[:-1] + x[1:])
    yc = 0.5*(y[:-1] + y[1:])
    xs, ys, qs = list(xc[j]), list(yc[i]), list(charge[i, j])
    # a vortex exactly on a sample (an analytic zero): wind around the node
    # through its eight neighbours
    for i0, j0 in zip(*np.nonzero(zero[1:-1, 1:-1])):
        i0, j0 = i0 + 1, j0 + 1
        ring = u[[i0, i0 + 1, i0 + 1, i0 + 1, i0, i0 - 1, i0 - 1, i0 - 1, i0],
                 [j0 + 1, j0 + 1, j0, j0 - 1, j0 - 1, j0 - 1, j0, j0 + 1, j0 + 1]]
        if np.any(ring == 0) or np.abs(ring).max()**2 <= I_min:
            continue
        q = int(np.rint(np.sum(np.angle(ring[1:]*np.conj(ring[:-1])))/(2*np.pi)))
        if q != 0 and (mask is None or mask[min(i0, mask.shape[0] - 1), min(j0, mask.shape[1] - 1)]):
            xs.append(x[j0])
            ys.append(y[i0])
            qs.append(q)

    return np.array(xs), np.array(ys), np.array(qs, dtype=int)


def oam_spectrum(field, x, y, l_max, x0=0.0, y0=0.0, r_max=None, n_r=None, n_phi=None):
    """Spiral spectrum: the share of the power in each azimuthal harmonic.

    The field is resampled on a polar grid about (x0, y0) and expanded in
    ``exp(i*l*phi)`` on each circle, ``u(r, phi) = sum_l a_l(r)*exp(i*l*phi)``.
    The weight of the harmonic l is ``P_l = int |a_l(r)|**2 r dr`` divided by
    the power in the disc r < r_max (Torner et al. 2005).

    Parameters
    ----------
    field : 2d array
    x, y : arrays
        Linear or meshed equidistant coordinates.
    l_max : int
        Largest ``|l|`` returned.
    x0, y0 : numbers, optional
        The axis about which the OAM is measured.
    r_max : number, optional
        Outer radius. Defaults to the largest circle inside the window.
    n_r : int, optional
        Number of radii. Defaults to the number of samples across r_max.
    n_phi : int, optional
        Number of azimuths. Defaults to max(256, 4*l_max).

    Returns
    -------
    ls : 1d int array
        -l_max, ..., l_max.
    P : 1d array
        Power fractions (they add up to 1 if l_max covers all harmonics).
    """

    x, y = _linear(x, y)
    dx = get_dx(x)
    if r_max is None:
        r_max = min(x0 - x[0], x[-1] - x0, y0 - y[0], y[-1] - y0)
    if n_r is None:
        n_r = max(int(np.ceil(r_max/dx)), 16)
    if n_phi is None:
        n_phi = max(256, 4*l_max)
    if n_phi < 2*l_max + 1:
        raise ValueError("n_phi must be at least 2*l_max + 1")
    r = (np.arange(n_r) + 0.5)*r_max/n_r
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    u = _interpolate(field, x, y, x0 + R*np.cos(PHI), y0 + R*np.sin(PHI))
    a = np.fft.fft(u, axis=1)/n_phi                   # a_l(r) in FFT order
    w = np.sum(np.abs(a)**2*r[:, None], axis=0)
    w /= np.sum(w)
    ls = np.arange(-l_max, l_max + 1)

    return ls, w[np.mod(ls, n_phi)]


def _derivatives(field, x, y, method):
    x, y = _linear(x, y)
    if method == "fft":
        kx = 2*np.pi*np.fft.fftfreq(len(x), get_dx(x))
        ky = 2*np.pi*np.fft.fftfreq(len(y), get_dx(y))
        F = np.fft.fft2(field)
        du_dx = np.fft.ifft2(1j*kx[None, :]*F)
        du_dy = np.fft.ifft2(1j*ky[:, None]*F)
    elif method == "difference":
        du_dy, du_dx = np.gradient(field, get_dx(y), get_dx(x), edge_order=2)
    else:
        raise ValueError("method must be 'fft' or 'difference'")
    return x, y, du_dx, du_dy


def mean_oam(field, x, y, x0=0.0, y0=0.0, method="fft"):
    """Orbital angular momentum per photon in units of hbar, about the axis
    through (x0, y0).

    ``<L_z> = Im(int conj(u)*(X*du/dy - Y*du/dx) dA)/int |u|**2 dA`` with
    X = x - x0 and Y = y - y0, the expectation value of ``-i*d/dphi``. For a
    beam without a mean tilt it does not depend on the axis.

    Parameters
    ----------
    field : 2d array
    x, y : arrays
        Linear or meshed equidistant coordinates.
    x0, y0 : numbers, optional
        Axis.
    method : {"fft", "difference"}, optional
        Spectral derivatives (default; accurate, needs a field that is
        smooth and decays towards the edges of the window) or second order
        finite differences.

    Returns
    -------
    l : float
    """

    x, y, du_dx, du_dy = _derivatives(field, x, y, method)
    X, Y = np.meshgrid(x - x0, y - y0)
    dphi = X*du_dy - Y*du_dx

    return float(np.sum(np.imag(np.conj(field)*dphi))/np.sum(np.abs(field)**2))


def orbital_current(field, x, y, method="fft"):
    """Transverse orbital current density ``Im(conj(u)*grad(u))``.

    It equals ``|u|**2*grad(arg u)``; divided by ``k*|u|**2`` it is the
    local direction (tangent of the angle) of the energy flow against the
    z axis. Around a vortex of charge l its azimuthal part is ``l/r*|u|**2``.

    Parameters
    ----------
    field : 2d array
    x, y : arrays
        Linear or meshed equidistant coordinates.
    method : {"fft", "difference"}, optional
        As in mean_oam().

    Returns
    -------
    jx, jy : 2d arrays
    """

    _, _, du_dx, du_dy = _derivatives(field, x, y, method)
    c = np.conj(field)

    return np.imag(c*du_dx), np.imag(c*du_dy)
