#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Beam widths, divergence and beam propagation ratio M² after ISO 11146.

The beam width is defined by the second moments of the intensity
distribution (ISO 11146-1). For a beam with centroid (x_c, y_c), the
variances sigma_x**2, sigma_y**2 and the covariance sigma_xy**2 give the
widths along the principal axes, ::

    d_x = 2*sqrt(2)*sqrt(sigma_x**2 + sigma_y**2 + g*sqrt((sigma_x**2 - sigma_y**2)**2 + 4*sigma_xy**4)),
    d_y = 2*sqrt(2)*sqrt(sigma_x**2 + sigma_y**2 - g*sqrt(...)),

with g = sgn(sigma_x**2 - sigma_y**2), and the azimuth ::

    phi = 1/2*arctan(2*sigma_xy**2/(sigma_x**2 - sigma_y**2))

of the principal axis "x" (the one closer to the lab x axis, ``|phi|`` <= pi/4).
For a TEM_00 beam with 1/e**2 intensity radius w, d_x = d_y = 2*w.

The second moments are very sensitive to noise and background far from the
beam. ISO 11146-3 therefore restricts the integration to a rectangle of
three times the beam widths, centered on the centroid and aligned with the
principal axes, and iterates until the widths converge (beam_widths() with
iterate=True). Any background offset has to be subtracted beforehand.

The beam propagation ratio follows from a hyperbolic fit to the widths d(z)
measured along the beam (ISO 11146-1, section 9): d(z)**2 = a + b*z + c*z**2
gives the waist position z_0 = -b/(2c), the waist width d_0, the (full)
divergence angle Theta = sqrt(c), the Rayleigh length z_R = d_0/Theta and
M² = pi/(4*wavelength)*d_0*Theta. ISO asks for at least 10 positions, about
half of them within one Rayleigh length of the waist and about half beyond
two Rayleigh lengths. The fit is done separately for each principal axis
(for simple astigmatic beams).
"""

import warnings
from collections import namedtuple

import numpy as np

from .bases import local_coordinates
from .utils import ensure_meshgrid


SecondMoments = namedtuple("SecondMoments", ["x_c", "y_c", "sigma_x2", "sigma_y2", "sigma_xy2"])
SecondMoments.__doc__ = """First and second moments of an intensity distribution.

x_c, y_c: centroid; sigma_x2, sigma_y2: variances sigma_x**2, sigma_y**2;
sigma_xy2: covariance sigma_xy**2 (all in units of the coordinates
(squared)).
"""


class BeamWidths(namedtuple("BeamWidths", ["d_x", "d_y", "phi", "x_c", "y_c", "moments"])):
    """Second moment beam widths after ISO 11146-1.

    d_x, d_y: widths (diameters) along the principal axes; phi: azimuth of
    the principal x axis in radians (counterclockwise from the lab x axis);
    x_c, y_c: centroid; moments: the underlying SecondMoments.
    """

    __slots__ = ()

    @property
    def d_r(self):
        """Generalized beam diameter 2*sqrt(2)*sqrt(sigma_x**2 + sigma_y**2)
        (equals d_x = d_y for round beams)."""

        return np.sqrt((self.d_x**2 + self.d_y**2)/2)

    @property
    def ellipticity(self):
        """Ratio min(d_x, d_y)/max(d_x, d_y); ISO 11146 regards beams with an
        ellipticity above 0.87 as round."""

        return min(self.d_x, self.d_y)/max(self.d_x, self.d_y)


class CausticFit(namedtuple("CausticFit", ["d_0", "z_0", "theta", "z_R", "M2", "coeffs"])):
    """Result of fit_caustic().

    d_0: waist width (diameter); z_0: waist position; theta: full far field
    divergence angle; z_R: Rayleigh length; M2: beam propagation ratio M²;
    coeffs: (a, b, c) of the fit d(z)**2 = a + b*z + c*z**2.
    """

    __slots__ = ()

    def d(self, z):
        """Fitted beam width d(z) = sqrt(d_0**2 + theta**2*(z - z_0)**2)."""

        return np.sqrt(self.d_0**2 + self.theta**2*(np.asarray(z) - self.z_0)**2)


def second_moments(I, x, y, background=0.0, mask=None):
    """Centroid and second moments of an intensity distribution.

    Parameters
    ----------
    I : array
        Intensity, shape (Ny, Nx).
    x, y : arrays
        Linear or meshed coordinates.
    background : number or array, optional
        Background subtracted from I before computing the moments.
    mask : array, optional
        Weights (e.g. a boolean integration area) that multiply the intensity.

    Returns
    -------
    moments : SecondMoments
    """

    X, Y = ensure_meshgrid(x, y)
    I = np.asarray(I, dtype=float) - background
    if mask is not None:
        I = I*mask

    P = np.sum(I)
    if P <= 0:
        raise ValueError("The (background corrected) intensity has no positive total power")

    x_c, y_c = np.sum(X*I)/P, np.sum(Y*I)/P
    sigma_x2 = np.sum((X - x_c)**2*I)/P
    sigma_y2 = np.sum((Y - y_c)**2*I)/P
    sigma_xy2 = np.sum((X - x_c)*(Y - y_c)*I)/P

    return SecondMoments(x_c, y_c, sigma_x2, sigma_y2, sigma_xy2)


def widths_from_moments(moments):
    """ISO 11146-1 beam widths and azimuth from second moments.

    Parameters
    ----------
    moments : SecondMoments

    Returns
    -------
    widths : BeamWidths
    """

    sx2, sy2, sxy2 = moments.sigma_x2, moments.sigma_y2, moments.sigma_xy2
    diff = sx2 - sy2

    if abs(diff) + abs(sxy2) <= 1E-12*(sx2 + sy2):
        # round beam: the azimuth is undefined, report 0
        gamma, phi = 1.0, 0.0
    elif diff != 0:
        gamma = np.sign(diff)
        phi = 0.5*np.arctan(2*sxy2/diff)
    else:
        # principal axes along the diagonals; the principal x axis at
        # phi = +-pi/4 is the major axis:
        gamma = 1.0
        phi = np.sign(sxy2)*np.pi/4

    root = np.sqrt(diff**2 + 4*sxy2**2)
    d_x = 2*np.sqrt(2)*np.sqrt(sx2 + sy2 + gamma*root)
    d_y = 2*np.sqrt(2)*np.sqrt(max(sx2 + sy2 - gamma*root, 0.0))

    return BeamWidths(d_x, d_y, phi, moments.x_c, moments.y_c, moments)


def beam_widths(I, x, y, background=0.0, iterate=True, window=3.0, max_iter=50, rtol=1E-6):
    """Second moment beam widths after ISO 11146.

    Parameters
    ----------
    I : array
        Intensity, shape (Ny, Nx).
    x, y : arrays
        Linear or meshed coordinates.
    background : number or array, optional
        Background (offset) subtracted from I.
    iterate : boolean, optional
        If True (default), restrict the integration to a rectangle of window
        times the beam widths around the centroid, aligned with the principal
        axes, and iterate until the widths change by less than rtol (ISO
        11146-3); for round beams (ellipticity > 0.87) the window is a square
        of window times d_r aligned with x and y. The iteration starts from the moments of the region above half
        the (background corrected) maximum. If False, integrate over
        the whole image.
    window : number, optional
        Size of the integration area in units of the beam widths. ISO
        prescribes 3.
    max_iter : int, optional
        Maximum number of iterations.
    rtol : number, optional
        Relative tolerance on d_x and d_y for convergence.

    Returns
    -------
    widths : BeamWidths
    """

    X, Y = ensure_meshgrid(x, y)

    if iterate:
        # start from the moments of the region above half the maximum: with
        # noisy images, the moments of the whole image can be far off or even
        # undefined (negative variances), and the iteration would not recover
        # from such a start. The half maximum region underestimates the widths,
        # and the iteration converges from below, which is robust against
        # noise far from the beam:
        I_corrected = np.asarray(I, dtype=float) - background
        widths = widths_from_moments(second_moments(I_corrected, X, Y,
                                                    mask=I_corrected > 0.5*I_corrected.max()))
    else:
        widths = widths_from_moments(second_moments(I, X, Y, background))

    if iterate:
        seen_areas = set()
        for _ in range(max_iter):
            if widths.ellipticity > 0.87:
                # round beam (ISO 11146-1): the azimuth is not defined and
                # would jump from iteration to iteration with noise, so use a
                # square window aligned with x and y
                U, V = X - widths.x_c, Y - widths.y_c
                half_x = half_y = window*widths.d_r/2
            else:
                U, V = local_coordinates(X, Y, widths.x_c, widths.y_c, widths.phi)
                half_x, half_y = window*widths.d_x/2, window*widths.d_y/2
            area = (np.abs(U) <= half_x) & (np.abs(V) <= half_y)
            key = np.packbits(area).tobytes()
            if key in seen_areas:
                # the same pixels as in an earlier step: the iteration has
                # converged or (with noise) runs in a cycle between a few
                # windows that differ by single pixels
                break
            seen_areas.add(key)
            with np.errstate(invalid="ignore"):
                new = widths_from_moments(second_moments(I, X, Y, background, area))
            if not np.all(np.isfinite([new.d_x, new.d_y])):
                raise ValueError("Beam width iteration failed (negative variance in the window); "
                                 "the signal to noise ratio is too low, average several frames")
            converged = np.allclose([new.d_x, new.d_y], [widths.d_x, widths.d_y], rtol=rtol, atol=0)
            widths = new
            if converged:
                break
        else:
            warnings.warn("Beam width iteration did not converge within {} steps".format(max_iter))

    return widths


def fit_caustic(z, d, wavelength):
    """Fit the hyperbolic caustic d(z)**2 = a + b*z + c*z**2 to measured beam
    widths and derive waist, divergence and M² (ISO 11146-1).

    Parameters
    ----------
    z : array
        Positions along the beam.
    d : array
        Beam widths (diameters, e.g. BeamWidths.d_x) at these positions.
    wavelength : number
        Wavelength in the medium the beam propagates in.

    Returns
    -------
    fit : CausticFit
    """

    z, d = np.asarray(z, dtype=float), np.asarray(d, dtype=float)
    if len(z) < 3:
        raise ValueError("Need at least 3 positions (ISO 11146 asks for 10)")

    c, b, a = np.polyfit(z, d**2, 2)
    if c <= 0 or 4*a*c - b**2 <= 0:
        raise ValueError("The widths do not describe a beam waist (fit gives c = {}, "
                         "4ac - b**2 = {}); measure further away from the waist".format(c, 4*a*c - b**2))

    z_0 = -b/(2*c)
    theta = np.sqrt(c)
    d_0 = np.sqrt(4*a*c - b**2)/(2*np.sqrt(c))
    z_R = d_0/theta
    M2 = np.pi/(4*wavelength)*d_0*theta

    return CausticFit(d_0, z_0, theta, z_R, M2, (a, b, c))
