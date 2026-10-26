#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Shack-Hartmann wavefront sensor: spot images, centroids and reconstruction.

A Shack-Hartmann sensor is a square array of n x n lenslets of pitch p and
focal length f in front of a camera in their focal plane. Each lenslet
focuses its part of the beam to a spot; a local tilt of the wavefront moves
the spot by f times the mean slope of the wavefront over the lenslet. From
the spot displacements, the wavefront is reconstructed either modally (least
squares fit of the gradients of Fringe Zernikes) or zonally (Southwell's
least squares integration on the lenslet grid).

`ShackHartmann` covers the whole chain:

- `ShackHartmann.field()`: the field on the lenslet array from a pupil phase
  and amplitude given on normalized pupil coordinates,
- `ShackHartmann.spots()`: the camera image, a Fraunhofer (Fourier) transform
  of the field over each lenslet, evaluated at the camera pixels,
- `ShackHartmann.slopes()`: thresholded centre of gravity of each spot,
  relative to a reference (the lenslet axes or a calibration image),
- `ShackHartmann.zernike_matrix()`, `ShackHartmann.reconstruct_zernikes()`
  and `ShackHartmann.reconstruct_zonal()`: wavefront reconstruction.

Noise is not part of the model; apply `processing.CameraNoiser` to the
image, scaled to photon numbers.

Conventions
-----------
Lengths (pitch, focal length, wavelength, pixel size, pupil radius) share
one unit. The lenslet array is centred on the optical axis; lenslet (i, j)
is row i (y) and column j (x), as for images. The beam footprint on the
lenslets is a disk of radius pupil_radius, which maps to the unit disk of
normalized pupil coordinates (P_x, P_y) = (x, y)/pupil_radius. Phases are
in radians, with the field exp(i*phase); the OPD in length units is
W = phase*wavelength/(2*pi), as in raytrace. A wavefront with a positive
slope dW/dx (the beam turns towards +x) moves the spots by +f*dW/dx.
Slopes are dimensionless (dW/dx in length per length). Fringe Zernike
coefficients are in waves on the unit pupil, as in phase_retrieval and
raytrace.

The centroid of the Fraunhofer pattern of a lenslet equals f times the
intensity weighted mean gradient of W over the lenslet (exactly, for an
infinite detector). `zernike_matrix()` uses this as the model, so partially
illuminated lenslets at the rim are described correctly. Deviations come
from the finite camera window of each lenslet, the threshold and the
discrete pixels.

Limitations: no crosstalk between lenslets (each spot is computed in the
window of its own lenslet only; light leaving the window is lost, which
sets the dynamic range), scalar Fraunhofer diffraction with ideal thin
lenslets, square lenslets without dead zones.
"""

from collections import namedtuple

import numpy as np
from numpy import pi

from .bases import fringe_zernike


Slopes = namedtuple("Slopes", ["sx", "sy", "valid", "flux"])
Slopes.__doc__ = """Measured wavefront slopes of a Shack-Hartmann sensor.

sx, sy: slopes dW/dx and dW/dy per lenslet, shape (n, n), NaN where not
valid. valid: lenslets with enough light. flux: light per lenslet window,
relative to the brightest one."""


def _as_array(value, P_x, P_y, default):
    """Evaluate value (callable, array, number or None) on the pupil grid."""

    if value is None:
        return default
    if callable(value):
        return np.asarray(value(P_x, P_y))

    return np.broadcast_to(np.asarray(value), P_x.shape)


class ShackHartmann(object):
    """Shack-Hartmann sensor with n x n square lenslets.

    Parameters
    ----------
    n_lenslets : int
        Number of lenslets across the array.
    pitch : number
        Lenslet pitch (width of a lenslet).
    focal_length : number
        Focal length of the lenslets; the camera sits in their focal plane.
    wavelength : number
        Wavelength.
    pixels : int, optional
        Camera pixels per lenslet across (default 16). The pixel size is
        pitch/pixels.
    samples : int, optional
        Samples of the field per lenslet across. The default is the smallest
        even number >= max(16, 2*N_F) with the lenslet Fresnel number
        N_F = pitch**2/(wavelength*focal_length), so that the Fourier sum of
        a lenslet does not alias over the camera window.
    pupil_radius : number, optional
        Radius of the beam footprint on the array, default n_lenslets*pitch/2
        (the circle inscribed in the array).

    Attributes
    ----------
    x, y : arrays
        Field sample coordinates on the array (length n_lenslets*samples).
    P_x, P_y : arrays
        Normalized pupil coordinates of the field samples, shape (N, N).
    centers : array
        Lenslet centre coordinates along x and y (length n_lenslets).
    u : array
        Camera pixel coordinates relative to the lenslet centre (length
        pixels).
    """

    def __init__(self, n_lenslets, pitch, focal_length, wavelength, pixels=16, samples=None,
                 pupil_radius=None):
        self.n_lenslets = int(n_lenslets)
        self.pitch = float(pitch)
        self.focal_length = float(focal_length)
        self.wavelength = float(wavelength)
        self.pixels = int(pixels)
        if samples is None:
            N_F = self.pitch**2/(self.wavelength*self.focal_length)
            samples = 2*int(np.ceil(max(16, 2*N_F)/2))
        self.samples = int(samples)
        self.pupil_radius = (self.n_lenslets*self.pitch/2 if pupil_radius is None
                             else float(pupil_radius))

        n, s = self.n_lenslets, self.samples
        self.centers = (np.arange(n) - (n - 1)/2)*self.pitch
        self.dx = self.pitch/s
        self._local = (np.arange(s) - (s - 1)/2)*self.dx
        self.x = (self.centers[:, None] + self._local[None, :]).ravel()
        self.y = self.x.copy()
        X, Y = np.meshgrid(self.x, self.y)
        self.P_x, self.P_y = X/self.pupil_radius, Y/self.pupil_radius
        self.pixel_size = self.pitch/self.pixels
        self.u = (np.arange(self.pixels) - (self.pixels - 1)/2)*self.pixel_size

    def __repr__(self):
        return ("ShackHartmann({0} x {0} lenslets, pitch {1:g}, f {2:g}, {3} pixels, "
                "pupil radius {4:g})".format(self.n_lenslets, self.pitch, self.focal_length,
                                             self.pixels, self.pupil_radius))

    @property
    def fresnel_number(self):
        """Fresnel number pitch**2/(wavelength*focal_length) of a lenslet."""

        return self.pitch**2/(self.wavelength*self.focal_length)

    @property
    def max_slope(self):
        """Largest slope that keeps a spot centre in its lenslet window."""

        return self.pitch/(2*self.focal_length)

    @property
    def slope_per_pixel(self):
        """Slope that moves a spot by one camera pixel."""

        return self.pixel_size/self.focal_length

    def _blocks(self, a):
        """Split an (N, N) array into (n, n, s, s) lenslet blocks."""

        n, s = self.n_lenslets, self.samples

        return a.reshape(n, s, n, s).transpose(0, 2, 1, 3)

    def amplitude(self, amplitude=None):
        """Field amplitude on the samples; default: the unit disk."""

        disk = (self.P_x**2 + self.P_y**2 <= 1.0).astype(float)

        return _as_array(amplitude, self.P_x, self.P_y, disk)

    def field(self, phase=None, amplitude=None):
        """Complex field on the lenslet array.

        Parameters
        ----------
        phase : callable, array or number, optional
            Pupil phase in radians: phase(P_x, P_y) on normalized pupil
            coordinates (e.g. raytrace.Pupil.phase), or an array on the
            sample grid of shape (N, N). Default: a plane wave.
        amplitude : callable, array or number, optional
            Field amplitude, as phase (e.g. raytrace.Pupil.mask). Default:
            1 on the unit disk, 0 outside.

        Returns
        -------
        E : array
            Field of shape (N, N) with N = n_lenslets*samples.
        """

        phi = _as_array(phase, self.P_x, self.P_y, np.zeros(self.P_x.shape))

        return self.amplitude(amplitude)*np.exp(1j*phi)

    def spots(self, field):
        """Camera image of the spots.

        Each lenslet window of pixels x pixels camera pixels receives the
        Fraunhofer pattern of the field over that lenslet, ::

            E(u, v) ~ sum_xy E(x, y)*exp(-i*k*(u*x + v*y)/f),

        with the local coordinates x, y and u, v relative to the lenslet
        centre.

        Parameters
        ----------
        field : array
            Field from field(), shape (N, N).

        Returns
        -------
        image : array
            Power per pixel, shape (n_lenslets*pixels,)*2, normalized to the
            total power of the field (a sum of 1 if no light leaves the
            windows).
        """

        k = 2*pi/self.wavelength
        M = np.exp(-1j*k*np.outer(self.u, self._local)/self.focal_length)
        E = np.einsum("ml,abls,ns->abmn", M, self._blocks(np.asarray(field)), M, optimize=True)
        # Fraunhofer: power per pixel = |dx**2/(lambda*f) sum|**2*pixel**2:
        I = np.abs(E)**2*(self.dx**2*self.pixel_size/(self.wavelength*self.focal_length))**2
        I /= np.sum(np.abs(field)**2)*self.dx**2
        n, q = self.n_lenslets, self.pixels

        return I.transpose(0, 2, 1, 3).reshape(n*q, n*q)

    def centroids(self, image, threshold=0.1):
        """Spot centroids and flux per lenslet window.

        Parameters
        ----------
        image : array
            Camera image from spots() (possibly with noise).
        threshold : number, optional
            Fraction of the peak of each window that is subtracted before the
            centre of gravity is computed (values below are set to zero).
            This suppresses the noise of the dark pixels.

        Returns
        -------
        cx, cy : arrays
            Centroids relative to the lenslet centres, shape (n, n), NaN for
            empty windows.
        flux : array
            Total of each window, shape (n, n).
        """

        n, q = self.n_lenslets, self.pixels
        W = np.asarray(image, dtype=float).reshape(n, q, n, q).transpose(0, 2, 1, 3)
        flux = W.sum(axis=(2, 3))
        peak = W.max(axis=(2, 3), keepdims=True)
        W = np.clip(W - threshold*peak, 0, None)
        total = W.sum(axis=(2, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = np.einsum("abmn,n->ab", W, self.u)/total
            cy = np.einsum("abmn,m->ab", W, self.u)/total

        return cx, cy, flux

    def slopes(self, image, reference=None, threshold=0.1, min_flux=0.5):
        """Wavefront slopes from a camera image.

        Parameters
        ----------
        image : array
            Camera image from spots().
        reference : array, optional
            Camera image of a reference (plane) wave, e.g. from a
            calibration; its centroids are subtracted. Default: the lenslet
            centres.
        threshold : number, optional
            See centroids().
        min_flux : number, optional
            Lenslets whose window holds less than min_flux times the flux of
            the brightest window are not valid (partially illuminated
            lenslets at the rim).

        Returns
        -------
        Slopes
        """

        cx, cy, flux = self.centroids(image, threshold)
        if reference is not None:
            rx, ry, _ = self.centroids(reference, threshold)
            cx, cy = cx - rx, cy - ry
        flux = flux/np.max(flux)
        valid = (flux >= min_flux) & np.isfinite(cx) & np.isfinite(cy)
        sx = np.where(valid, cx/self.focal_length, np.nan)
        sy = np.where(valid, cy/self.focal_length, np.nan)

        return Slopes(sx, sy, valid, flux)

    def model_slopes(self, phase=None, amplitude=None):
        """Slopes that an ideal sensor measures.

        The intensity weighted mean phase gradient over each lenslet, converted to
        dW/dx and dW/dy. The gradient is taken by central differences on the
        sample grid, so the phase must be continuous (unwrapped). Samples whose
        difference stencil reaches a dark sample are left out, so that a phase
        that is only defined on the pupil (e.g. zero outside) does not produce
        spurious slopes at the rim.

        Parameters
        ----------
        phase, amplitude : callable, array or number, optional
            As for `field()`.

        Returns
        -------
        sx, sy : arrays
            Shape (n, n); NaN for dark lenslets.
        """

        phi = _as_array(phase, self.P_x, self.P_y, np.zeros(self.P_x.shape))
        amp = self.amplitude(amplitude)
        gy, gx = np.gradient(np.where(amp > 0, phi, np.nan), self.dx)
        scale = self.wavelength/(2*pi)
        result = []
        for g in (gx, gy):
            ok = np.isfinite(g)
            w = self._blocks(np.where(ok, amp**2, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                result.append(np.sum(w*self._blocks(np.where(ok, g, 0.0)), axis=(2, 3))
                              /np.sum(w, axis=(2, 3))*scale)

        return tuple(result)

    def zernike_matrix(self, indices, amplitude=None):
        """Slopes of Fringe Zernikes of one wave each.

        Parameters
        ----------
        indices : sequence of int
            Fringe indices j.
        amplitude : callable, array or number, optional
            Pupil amplitude (weights of the gradients), as for field().

        Returns
        -------
        Gx, Gy : arrays
            Shape (len(indices), n, n): model slopes dW/dx, dW/dy of
            1 wave of Z_j (see model_slopes()).
        """

        G = [self.model_slopes(2*pi*fringe_zernike(j, self.P_x, self.P_y), amplitude)
             for j in indices]

        return np.array([g[0] for g in G]), np.array([g[1] for g in G])

    def reconstruct_zernikes(self, slopes, n_zernike=15, amplitude=None, indices=None):
        """Modal reconstruction: least squares fit of Zernike gradients.

        Parameters
        ----------
        slopes : Slopes
            Measured slopes from slopes().
        n_zernike : int, optional
            Fit the Fringe Zernikes 2, ..., n_zernike (piston is invisible).
        amplitude : callable, array or number, optional
            Pupil amplitude for the model, as for field().
        indices : sequence of int, optional
            Fringe indices to fit instead of 2...n_zernike.

        Returns
        -------
        coeffs : dict
            Fringe Zernike coefficients {j: c_j} in waves.
        """

        indices = list(range(2, n_zernike + 1)) if indices is None else list(indices)
        Gx, Gy = self.zernike_matrix(indices, amplitude)
        v = slopes.valid & np.all(np.isfinite(Gx) & np.isfinite(Gy), axis=0)
        A = np.concatenate([Gx[:, v].T, Gy[:, v].T])
        b = np.concatenate([slopes.sx[v], slopes.sy[v]])
        c = np.linalg.lstsq(A, b, rcond=None)[0]

        return dict(zip(indices, c))

    def reconstruct_zonal(self, slopes):
        """Zonal reconstruction (Southwell 1980) on the lenslet grid.

        Solves (W_{i,j+1} - W_{i,j})/pitch = (s_x[i,j+1] + s_x[i,j])/2 and
        the same along y for all pairs of neighbouring valid lenslets in the
        least squares sense (the mean of W is zero).

        Parameters
        ----------
        slopes : Slopes
            Measured slopes from slopes().

        Returns
        -------
        W : array
            Wavefront at the lenslet centres in waves, shape (n, n), NaN where
            not valid.
        """

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr

        v = slopes.valid
        idx = -np.ones(v.shape, dtype=int)
        idx[v] = np.arange(np.count_nonzero(v))
        rows, cols, vals, b = [], [], [], []

        def add(i1, i2, s1, s2):
            ok = (i1 >= 0) & (i2 >= 0)
            r = len(b) + np.arange(np.count_nonzero(ok))
            rows.extend(np.repeat(r, 2))
            cols.extend(np.column_stack([i2[ok], i1[ok]]).ravel())
            vals.extend(np.tile([1.0, -1.0], len(r)))
            b.extend(self.pitch*(s1[ok] + s2[ok])/2)

        add(idx[:, :-1], idx[:, 1:], slopes.sx[:, :-1], slopes.sx[:, 1:])
        add(idx[:-1, :], idx[1:, :], slopes.sy[:-1, :], slopes.sy[1:, :])
        n_v = np.count_nonzero(v)
        # zero mean as an extra equation:
        rows.extend([len(b)]*n_v)
        cols.extend(range(n_v))
        vals.extend([1.0]*n_v)
        b.append(0.0)
        A = coo_matrix((vals, (rows, cols)), shape=(len(b), n_v)).tocsr()
        w = lsqr(A, np.array(b), atol=1e-14, btol=1e-14, iter_lim=20*n_v)[0]
        W = np.full(v.shape, np.nan)
        W[v] = (w - w.mean())/self.wavelength

        return W
