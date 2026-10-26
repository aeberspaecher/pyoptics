#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Interferometry: simulation of interferograms, phase-shifting
interferometry (PSI) and Fourier-transform fringe analysis (Takeda).

Conventions
-----------
Fields vary as exp(i(k.r - omega*t)) (see the README). The test beam carries
the phase phi = k*W of its wavefront error (or optical path difference) W,
so a wavefront that runs ahead has phi > 0. Superposing it with a reference
beam E_ref gives the two-beam interferogram ::

    I = |E_test + E_ref*exp(i*delta)|**2
      = I_0*(1 + gamma*cos(phi - delta)),

where phi = arg(E_test*conj(E_ref)) is the phase of the test beam relative to
the reference and gamma the visibility (fringe contrast). A phase shift delta
> 0 delays the reference by delta (its optical path gets longer by
delta/k), so the fringes move towards larger phi. All algorithms here return
phi, i.e. the phase of the test beam relative to the reference, with the sign
convention of pyoptics: phase = +k*OPD. Note that many textbooks write
cos(phi + delta) instead; their formulas then give -phi.

In reflection (Twyman-Green, Fizeau), the light passes the surface twice, so
a surface height h gives phi = 2*k*h.

A tilt between the beams adds a linear carrier phase k_c.r = k_cx*x + k_cy*y
(given here as a wavenumber in rad per length unit, like the k of
utils.freq_grid()).

Image stacks follow the convention of pyoptics.processing: the last index
numbers the images, i.e. frames[:, :, j] is the j-th phase-shifted frame (any
number of leading dimensions is fine). Stacks can thus be passed through
processing.Scaler and processing.CameraNoiser directly.

The phase-shifting algorithms return the wrapped phase in (-pi, pi] and the
modulation gamma = b/a, where I = a + b*cos(phi - delta), i.e. the local
fringe visibility (between 0 and 1 for physical intensities). Pixels without
light (a = 0) get gamma = 0. Unwrap the phase with pyoptics.unwrap.

Contents
--------
Simulation:

- interferogram(): interferogram (or stack) from a phase map
- interfere(): interferogram (or stack) from two complex fields
- zernike_phase(): phase map from Fringe Zernike coefficients in waves
- phase_shifts(): nominal or miscalibrated phase shifts

Phase-shifting algorithms:

- psi_n_step(): least squares for N frames with arbitrary known shifts
- psi_4_step(): four-bucket algorithm (shifts 0, pi/2, pi, 3*pi/2)
- psi_carre(): Carré, four frames with an unknown but constant step
- psi_hariharan(): Hariharan's five-frame algorithm, insensitive to linear
  shift errors
- shift_error_sensitivity(): rms phase error vs. phase shifter
  miscalibration

Single-frame analysis:

- fourier_fringe_analysis(): Takeda's Fourier-transform method for
  interferograms with a carrier
- find_carrier(): locate the carrier sideband in the spectrum

Spherical surfaces (radius of curvature tests):

- aplanatic_directions(): ray directions of a converging test wave
- sphere_displacement_phase(): phase of a displaced test sphere
- fit_sphere_displacement(): displacement of a test sphere from the null
  position (confocal or cat's eye) from its measured phase

A sphere tested with a converging spherical wave (Fizeau transmission
sphere, Twyman-Green with a diverger lens) gives a null interferogram in two
positions: confocal, with its centre of curvature in the focus, and cat's
eye, with its vertex in the focus. The radius of curvature is the distance
between the two, R = z_cat - z_confocal (signed as in pyoptics.raytrace,
R > 0 for a centre of curvature in +z, i.e. downstream). Near either
position, a small displacement d of the sphere changes the optical path of
the ray with the direction n by d.n per pass, i.e. the phase by ::

    phi = passes*k*(d_x*n_x + d_y*n_y + d_z*n_z),

tilt fringes for a lateral and power (defocus) fringes for an axial
displacement. Fitting this model to the measured phase gives the null
positions from slightly defocused interferograms. In cat's eye, lateral
displacements cancel to first order (the beam returns on the opposite side
of the pupil).
"""

import warnings

import numpy as np
from numpy import pi

from .bases import FringeZernikes
from .fft import FT, inv_FT
from .utils import ensure_meshgrid, freq_grid, get_dx


def _wrap(phi):
    """Wrap phases to (-pi, pi]."""

    return np.angle(np.exp(1j*np.asarray(phi)))


def _modulation(a, b):
    """gamma = b/a, with gamma = 0 where a = 0."""

    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(a != 0, b/np.where(a != 0, a, 1.0), 0.0)

    return gamma


def _frames(frames, n=None, name="algorithm"):
    """Return frames as float array with the frame index last and check
    their number."""

    frames = np.asarray(frames, dtype=float)
    if frames.ndim < 1:
        raise ValueError("frames must have at least one dimension")
    if n is not None and frames.shape[-1] != n:
        raise ValueError("{} needs {} frames (last axis), got {}".format(
            name, n, frames.shape[-1]))

    return frames


# --- simulation --------------------------------------------------------------

def zernike_phase(x, y, coeffs, R_norm, indices=None, x0=0.0, y0=0.0,
                  passes=1):
    """Phase map 2*pi*passes*W of a wavefront W given by Fringe Zernike
    coefficients in waves.

    Parameters
    ----------
    x, y : arrays
        Linear coordinates of the grid.
    coeffs : sequence of floats
        Fringe Zernike coefficients in waves (units of the wavelength).
    R_norm : float
        Radius of the pupil (normalization radius of the Zernikes).
    indices : sequence of ints, optional
        Fringe indices that belong to coeffs. Defaults to 1, 2, ...,
        len(coeffs).
    x0, y0 : floats, optional
        Center of the pupil.
    passes : int, optional
        Number of passes through the test object, e.g. 2 for a surface
        measured in reflection (the coefficients then describe the surface
        height in waves).

    Returns
    -------
    phase : array
        Phase in radians, zero outside the pupil.
    mask : array of bool
        Pupil mask.
    """

    if indices is None:
        indices = range(1, len(coeffs) + 1)
    indices = list(indices)
    if len(indices) != len(coeffs):
        raise ValueError("coeffs and indices must have the same length")

    Z = FringeZernikes(x, y, R_norm=R_norm, x0=x0, y0=y0)
    W = Z(indices, list(coeffs))
    mask = np.asarray(Z.mask, dtype=bool)

    return 2*pi*passes*W*mask, mask


def phase_shifts(N, step=None, start=None, linear_error=0.0,
                 quadratic_error=0.0):
    """Phase shifts of a phase-shifting sequence, optionally with a
    miscalibrated phase shifter.

    The nominal shifts are delta_j = start + j*step, j = 0, ..., N - 1. A
    phase shifter with the relative linear calibration error epsilon_1 and a
    quadratic nonlinearity epsilon_2 produces ::

        delta_j' = delta_j*(1 + epsilon_1) + epsilon_2*delta_j**2/pi,

    the error being taken relative to the shifter's zero position (delta =
    0).

    Parameters
    ----------
    N : int
        Number of frames.
    step : float, optional
        Nominal phase step in radians. Defaults to 2*pi/N.
    start : float, optional
        Nominal first shift. Defaults to 0, i.e. the sequence 0, step, ...
        Use start=-(N - 1)*step/2 for a sequence centered at 0 (as for
        Carré's and Hariharan's algorithms).
    linear_error : float, optional
        Relative linear calibration error epsilon_1, e.g. 0.05 for a shifter
        that moves 5 % too far.
    quadratic_error : float, optional
        Quadratic nonlinearity epsilon_2 (relative error of a shift of pi).

    Returns
    -------
    shifts : array of length N
    """

    if step is None:
        step = 2*pi/N
    if start is None:
        start = 0.0

    nominal = start + step*np.arange(N)

    return nominal*(1 + linear_error) + quadratic_error*nominal**2/pi


def interferogram(phase, x=None, y=None, visibility=1.0, carrier=(0.0, 0.0),
                  shift=0.0, I0=1.0, noiser=None):
    """Two-beam interferogram I = I0*(1 + visibility*cos(phase + k_c.r -
    shift)) of a test phase.

    Parameters
    ----------
    phase : array
        Phase of the test beam relative to the reference in radians (e.g.
        from zernike_phase()).
    x, y : arrays, optional
        Linear coordinates of the grid; only needed for a carrier.
    visibility : float or array, optional
        Fringe visibility gamma (may vary across the field).
    carrier : tuple of floats, optional
        Carrier wavenumbers (k_cx, k_cy) in rad per length unit of a tilt
        between the beams, e.g. (2*pi/period, 0) for vertical fringes.
    shift : float or sequence of floats, optional
        Phase shift(s) of the reference in radians. For a sequence, the
        result is a stack with the frame index last.
    I0 : float or array, optional
        Mean intensity (bias).
    noiser : callable, optional
        Applied to the result, e.g. a processing.CameraNoiser (in that case,
        I0 is the mean number of photons per pixel).

    Returns
    -------
    I : array
        Interferogram, or stack of interferograms of shape phase.shape +
        (len(shift),).
    """

    phase = np.asarray(phase, dtype=float)
    k_cx, k_cy = carrier
    if k_cx != 0 or k_cy != 0:
        if x is None or y is None:
            raise ValueError("x and y are needed for a carrier")
        X, Y = ensure_meshgrid(x, y)
        phase = phase + k_cx*X + k_cy*Y

    shift = np.asarray(shift, dtype=float)
    if shift.ndim > 0:
        phase = phase[..., np.newaxis]
        I0 = np.asarray(I0, dtype=float)
        visibility = np.asarray(visibility, dtype=float)
        if I0.ndim > 0:
            I0 = I0[..., np.newaxis]
        if visibility.ndim > 0:
            visibility = visibility[..., np.newaxis]

    I = I0*(1 + visibility*np.cos(phase - shift))
    if noiser is not None:
        I = noiser(I)

    return I


def interfere(E_test, E_ref, shift=0.0):
    """Interferogram ``|E_test + E_ref*exp(i*shift)|**2`` of two fields.

    The algorithms of this module recover arg(E_test*conj(E_ref)) from it.

    Parameters
    ----------
    E_test, E_ref : arrays (or scalar for E_ref)
        Complex fields of the test and reference beams.
    shift : float or sequence of floats, optional
        Phase shift(s) of the reference beam. For a sequence, the result is a
        stack with the frame index last.

    Returns
    -------
    I : array
    """

    E_test = np.asarray(E_test)
    E_ref = np.asarray(E_ref)
    shift = np.asarray(shift, dtype=float)
    if shift.ndim > 0:
        E_test = E_test[..., np.newaxis]
        if E_ref.ndim > 0:
            E_ref = E_ref[..., np.newaxis]

    return np.abs(E_test + E_ref*np.exp(1j*shift))**2


# --- phase-shifting algorithms ----------------------------------------------

def psi_n_step(frames, shifts=None, return_bias=False):
    """Least-squares phase-shifting algorithm for N frames with known
    shifts.

    Fits I_j = a + c*cos(delta_j) + s*sin(delta_j) per pixel (Bruning et al.
    1974, Greivenkamp 1984), which gives phi = atan2(s, c) and b =
    sqrt(c**2 + s**2). For equally spaced shifts over one period, this is the
    classical N-bucket (synchronous detection) algorithm.

    Parameters
    ----------
    frames : array
        Image stack, frame index last. At least 3 frames.
    shifts : sequence of floats, optional
        Phase shifts of the frames in radians. Defaults to 2*pi*j/N.
    return_bias : bool, optional
        If True, also return the bias a.

    Returns
    -------
    phase : array
        Wrapped phase phi in (-pi, pi].
    modulation : array
        gamma = b/a.
    bias : array
        Only if return_bias: the mean intensity a.
    """

    frames = _frames(frames)
    N = frames.shape[-1]
    if shifts is None:
        shifts = 2*pi*np.arange(N)/N
    shifts = np.asarray(shifts, dtype=float)
    if len(shifts) != N:
        raise ValueError("need one shift per frame")
    if N < 3:
        raise ValueError("need at least 3 frames")

    A = np.stack([np.ones(N), np.cos(shifts), np.sin(shifts)], axis=1)
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("the shifts do not determine the phase (rank < 3)")
    a, c, s = np.moveaxis(frames @ np.linalg.pinv(A).T, -1, 0)

    phase = np.arctan2(s, c)
    modulation = _modulation(a, np.hypot(c, s))
    if return_bias:
        return phase, modulation, a

    return phase, modulation


def psi_4_step(frames):
    """Four-bucket algorithm for the shifts 0, pi/2, pi and 3*pi/2.

    The phase is ::

        phi = atan2(I_2 - I_4, I_1 - I_3).

    Parameters
    ----------
    frames : array
        Stack of 4 frames, frame index last.

    Returns
    -------
    phase, modulation : arrays
        Wrapped phase and modulation gamma = b/a.
    """

    I1, I2, I3, I4 = np.moveaxis(_frames(frames, 4, "psi_4_step"), -1, 0)
    s, c = I2 - I4, I1 - I3
    a = (I1 + I2 + I3 + I4)/4

    return np.arctan2(s, c), _modulation(a, np.hypot(s, c)/2)


def psi_carre(frames, return_step=False):
    """Carré's algorithm: four frames with the shifts -3*alpha/2,
    -alpha/2, alpha/2, 3*alpha/2 and an unknown step alpha.

    With D_1 = I_1 - I_4 and D_2 = I_2 - I_3, ::

        tan(alpha/2)**2 = (3*D_2 - D_1)/(D_1 + D_2),
        tan(phi) = sgn(-D_2)*sqrt(|(3*D_2 - D_1)*(D_1 + D_2)|)
                   / (I_2 + I_3 - I_1 - I_4).

    The step is estimated per pixel; the modulation uses the median step.
    The estimate of alpha (and with it the phase) is ill-conditioned where
    sin(phi) or cos(phi) is close to 0, so noise hurts more than with
    algorithms for known shifts. Works for 0 < alpha < pi, best near alpha =
    110 degrees.

    Parameters
    ----------
    frames : array
        Stack of 4 frames, frame index last.
    return_step : bool, optional
        If True, also return the median of the estimated step alpha.

    Returns
    -------
    phase, modulation : arrays
        Wrapped phase and modulation gamma = b/a.
    alpha : float
        Only if return_step: the estimated phase step in radians.
    """

    I1, I2, I3, I4 = np.moveaxis(_frames(frames, 4, "psi_carre"), -1, 0)
    D1, D2 = I1 - I4, I2 - I3
    S = I2 + I3 - I1 - I4  # 2*b*cos(phi)*(cos(alpha/2) - cos(3*alpha/2))

    num = np.sqrt(np.abs((3*D2 - D1)*(D1 + D2)))*np.sign(-D2)
    phase = np.arctan2(num, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (3*D2 - D1)/(D1 + D2)
    ratio = ratio[np.isfinite(ratio) & (ratio >= 0)]
    if ratio.size == 0:
        raise ValueError("cannot estimate the phase step (no fringes?)")
    alpha = 2*np.arctan(np.sqrt(np.median(ratio)))

    # D_2 = -2*b*sin(phi)*sin(alpha/2)
    b_sin = -D2/(2*np.sin(alpha/2))
    b_cos = S/(2*(np.cos(alpha/2) - np.cos(3*alpha/2)))
    a = (I1 + I2 + I3 + I4)/4 \
        - b_cos*(np.cos(alpha/2) + np.cos(3*alpha/2))/2
    modulation = _modulation(a, np.hypot(b_sin, b_cos))

    if return_step:
        return phase, modulation, alpha

    return phase, modulation


def psi_hariharan(frames, return_step=False):
    """Hariharan's five-frame algorithm (Hariharan, Oreb, Eiju 1987) for the
    shifts -2*alpha, -alpha, 0, alpha, 2*alpha with alpha = pi/2::

        phi = atan2(2*(I_4 - I_2), 2*I_3 - I_1 - I_5).

    A linear miscalibration alpha = pi/2*(1 + epsilon) only causes a phase
    error of second order in epsilon (compared to first order for the four
    bucket algorithm).

    Parameters
    ----------
    frames : array
        Stack of 5 frames, frame index last.
    return_step : bool, optional
        If True, also return the median of the step estimated from
        cos(alpha) = (I_5 - I_1)/(2*(I_4 - I_2)) (Hariharan's calibration
        formula).

    Returns
    -------
    phase, modulation : arrays
        Wrapped phase and modulation gamma = b/a.
    alpha : float
        Only if return_step: the estimated phase step in radians.
    """

    I1, I2, I3, I4, I5 = np.moveaxis(_frames(frames, 5, "psi_hariharan"),
                                     -1, 0)
    s = 2*(I4 - I2)  # 4*b*sin(phi)*sin(alpha)
    c = 2*I3 - I1 - I5  # 2*b*cos(phi)*(1 - cos(2*alpha))
    phase = np.arctan2(s, c)

    # for alpha = pi/2: s = 4 b sin(phi), c = 4 b cos(phi),
    # a = (I_1 + 2 I_2 + 2 I_4 + I_5)/6 + ... : use the mean of I_2, I_4 and
    # of I_1, I_3, I_5 (each b*cos/sin cancels to first order)
    a = ((I2 + I4)/2 + (I1 + 2*I3 + I5)/4)/2
    modulation = _modulation(a, np.hypot(s, c)/4)

    if return_step:
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_alpha = (I5 - I1)/(2*(I4 - I2))
        good = np.isfinite(cos_alpha) & (np.abs(I4 - I2) > 0.1*np.max(np.abs(I4 - I2)))
        alpha = np.arccos(np.clip(np.median(cos_alpha[good]), -1, 1))
        return phase, modulation, alpha

    return phase, modulation


_ALGORITHMS = {
    "4-step": (psi_4_step, 4, pi/2, False),
    "carre": (psi_carre, 4, pi/2, True),
    "hariharan": (psi_hariharan, 5, pi/2, True),
}


def shift_error_sensitivity(algorithm, linear_errors, quadratic_error=0.0,
                            N=None, step=None, N_phase=360):
    """Rms phase error of a phase-shifting algorithm vs. the calibration
    error of the phase shifter.

    Frames with miscalibrated shifts (see phase_shifts()) of noise-free
    fringes with visibility 1 are evaluated for N_phase test phases evenly
    spread over one period; the rms of the wrapped phase error is returned
    for each linear error.

    Parameters
    ----------
    algorithm : str or callable
        "4-step", "carre", "hariharan", "n-step" (least squares with the
        nominal shifts, use N and step) or a callable f(frames) -> (phase,
        modulation) (then N and step are needed).
    linear_errors : sequence of floats
        Relative linear calibration errors epsilon_1.
    quadratic_error : float, optional
        Quadratic nonlinearity epsilon_2 (see phase_shifts()).
    N : int, optional
        Number of frames ("n-step" and callables).
    step : float, optional
        Nominal phase step; defaults to 2*pi/N for "n-step" and callables.
    N_phase : int, optional
        Number of test phases.

    Returns
    -------
    rms_error : array
        Rms phase error in radians for each linear error.
    """

    centered = True
    if callable(algorithm):
        func = algorithm
        centered = False
    elif algorithm == "n-step":
        if N is None:
            raise ValueError("N is needed for 'n-step'")
        step = 2*pi/N if step is None else step
        nominal = phase_shifts(N, step)

        def func(frames):
            return psi_n_step(frames, nominal)
        centered = False
    elif algorithm in _ALGORITHMS:
        func, N, step, centered = _ALGORITHMS[algorithm]
    else:
        raise ValueError("unknown algorithm {}".format(algorithm))

    if N is None:
        raise ValueError("N is needed for a callable algorithm")
    if step is None:
        step = 2*pi/N
    start = -(N - 1)*step/2 if centered else 0.0

    phi = 2*pi*(np.arange(N_phase) + 0.5)/N_phase - pi
    rms = []
    for eps in np.atleast_1d(linear_errors):
        shifts = phase_shifts(N, step, start, eps, quadratic_error)
        frames = interferogram(phi, shift=shifts)
        phase = func(frames)[0]
        rms.append(np.sqrt(np.mean(_wrap(phase - phi)**2)))

    return np.array(rms)


# --- Fourier-transform method -----------------------------------------------

def _tukey(N, alpha):
    """1D Tukey window with taper fraction alpha."""

    if alpha <= 0:
        return np.ones(N)
    n = np.arange(N)/(N - 1)
    w = np.ones(N)
    edge = alpha/2
    lo, hi = n < edge, n > 1 - edge
    w[lo] = 0.5*(1 - np.cos(pi*n[lo]/edge))
    w[hi] = 0.5*(1 - np.cos(pi*(1 - n[hi])/edge))

    return w


def _spectral_window(KX, KY, center, radius, kind):
    """Window around center in k space."""

    r = np.hypot(KX - center[0], KY - center[1])/radius
    if kind == "tophat":
        return (r <= 1).astype(float)
    if kind == "hann":
        return np.where(r <= 1, np.cos(pi*r/2)**2, 0.0)
    if kind == "gaussian":  # 1/e**2 radius equals radius
        return np.exp(-2*r**2)
    raise ValueError("window must be 'tophat', 'hann', 'gaussian' or an array")


def find_carrier(I, x, y, exclude=None, mask=None):
    """Locate the carrier, i.e. the +1 sideband of a fringe pattern.

    Searches the spectral peak in the half plane k_x > 0 (or k_x = 0,
    k_y > 0) outside a disk around k = 0 and returns the centroid of the
    power spectrum ``|S|**2`` in a disk of radius ``|k|/2`` around the current
    estimate k, starting at the peak and iterated until it settles.
    For a test phase without net tilt (e.g. defocus or astigmatism, which
    spread the sideband symmetrically), the centroid is the carrier; a tilt
    of the test phase is included in it.

    Parameters
    ----------
    I : array
        Interferogram.
    x, y : arrays
        Linear coordinates.
    exclude : float, optional
        Radius around k = 0 (in rad per length unit) that is ignored.
        Defaults to 3 frequency samples.
    mask : array of bool, optional
        Region with fringes (e.g. the pupil). Without it, the edge of a
        pupil can outshine the sideband.

    Returns
    -------
    carrier : tuple of floats
        (k_cx, k_cy).
    """

    I = np.asarray(I, dtype=float)
    KX, KY = freq_grid(x, y, wavenumbers=True, normal_order=True)
    dkx = abs(KX[0, 1] - KX[0, 0])
    dky = abs(KY[1, 0] - KY[0, 0])
    if exclude is None:
        exclude = 3*max(dkx, dky)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        I = np.where(mask, I - np.mean(I[mask]), 0.0)
    else:
        I = I - np.mean(I)
    P = np.abs(FT(I))**2
    half = (KX > 0) | ((KX == 0) & (KY > 0))
    P_search = np.where(half & (np.hypot(KX, KY) > exclude), P, 0.0)
    m, n = np.unravel_index(np.argmax(P_search), P.shape)
    if P_search[m, n] == 0:
        raise ValueError("no carrier found")

    # centroid in a disk of radius |k|/2, recentred until it settles (the
    # peak of a broad sideband need not lie at its centre)
    k = np.array([KX[m, n], KY[m, n]])
    for _ in range(20):
        near = np.hypot(KX - k[0], KY - k[1]) <= np.hypot(*k)/2
        P_near = np.where(near, P_search, 0.0)
        k_new = np.array([np.sum(KX*P_near), np.sum(KY*P_near)])/np.sum(P_near)
        if np.hypot(*(k_new - k)) < 1E-3*max(dkx, dky):
            break
        k = k_new

    return (k_new[0], k_new[1])


def fourier_fringe_analysis(I, x, y, carrier=None, window="tophat",
                            radius=None, mask=None, apodize=0.0,
                            remove_carrier=True, return_sideband=False):
    """Takeda's Fourier-transform method (Takeda, Ina, Kobayashi 1982) for a
    single interferogram with a carrier.

    The interferogram I = a + b*cos(phi + k_c.r) = a + c*exp(i k_c.r) + c.c.,
    with c = b/2*exp(i*phi), has a sideband around k_c in its spectrum. A
    window around k_c isolates it; its inverse transform, multiplied with
    exp(-i k_c.r), gives c and hence phi and b. The bias a is the low-pass
    filtered interferogram (same window around k = 0).

    The method needs the sideband to be separated from the zero order and
    the -1 order: the local fringe frequency ``|grad phi|`` must stay below the
    window radius (by default ``|k_c|/2)``. A warning is issued if the window
    holds less than 90 % of the power in the half plane of the carrier, or
    if it reaches beyond the Nyquist frequency.

    Parameters
    ----------
    I : array
        Interferogram.
    x, y : arrays
        Linear coordinates.
    carrier : tuple of floats, optional
        Carrier (k_cx, k_cy) in rad per length unit. Found with
        find_carrier() if not given; then the tilt of the test phase is
        removed along with the carrier (remove a residual tilt with a fit if
        needed).
    window : str or array, optional
        "tophat" (default), "hann" or "gaussian" window around k_c, or a
        window array on the centered frequency grid of I. The tapered
        windows ring less but attenuate the outer parts of a broad sideband,
        which biases the modulation low; the bias is filtered with the same
        window around k = 0 (for a window array, the global mean is used).
    radius : float, optional
        Window radius in rad per length unit. Defaults to ``|k_c|/2``.
    mask : array of bool, optional
        Region with fringes (e.g. the pupil). Outside, I is replaced by its
        mean inside before the transform, which reduces ringing at the edge,
        and the outputs are set to 0.
    apodize : float, optional
        Taper fraction of a separable Tukey window applied to I - mean(I)
        before the transform (0: no apodization, 1: Hann window). It reduces
        the ringing that the field edges cause in the interior, at the cost
        of larger errors in the tapered border. The modulation is corrected
        for it.
    remove_carrier : bool, optional
        If False, the carrier phase k_c.r is kept in the result.
    return_sideband : bool, optional
        If True, also return the complex sideband c (after carrier removal).

    Returns
    -------
    phase : array
        Wrapped phase in (-pi, pi].
    modulation : array
        gamma = b/a.
    sideband : array
        Only if return_sideband.
    """

    I = np.asarray(I, dtype=float)
    X, Y = ensure_meshgrid(x, y)
    KX, KY = freq_grid(x, y, wavenumbers=True, normal_order=True)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        I = np.where(mask, I, np.mean(I[mask]))

    if carrier is None:
        carrier = find_carrier(I, x, y)
    k_c = np.array(carrier, dtype=float)
    k_c_abs = np.hypot(*k_c)
    if k_c_abs == 0:
        raise ValueError("the Fourier-transform method needs a carrier")
    if radius is None:
        radius = k_c_abs/2

    k_nyq = pi/max(get_dx(x), get_dx(y))
    if np.max(np.abs(k_c)) + radius > k_nyq:
        warnings.warn("the sideband window reaches beyond the Nyquist "
                      "frequency: the carrier is too large for the sampling")
    dk = 2*pi/(len(x)*get_dx(x))
    if k_c_abs < 4*dk:
        warnings.warn("the carrier is less than 4 fringes across the field: "
                      "zero order and sideband overlap")

    mean = np.mean(I)
    apod = np.outer(_tukey(I.shape[0], apodize), _tukey(I.shape[1], apodize))
    S = FT((I - mean)*apod)

    if isinstance(window, str):
        W_side = _spectral_window(KX, KY, k_c, radius, window)
        W_zero = _spectral_window(KX, KY, (0.0, 0.0), radius, window)
    else:
        W_side = np.asarray(window)
        W_zero = None

    c = inv_FT(S*W_side)*np.exp(-1j*(k_c[0]*X + k_c[1]*Y))
    if W_zero is not None:
        # bias and sideband both carry the apodization window
        a = np.real(inv_FT(S*W_zero)) + mean*apod
    else:
        a = mean*apod

    phase = np.angle(c)
    modulation = _modulation(a, 2*np.abs(c))

    # check for spectral overlap: the window must hold the sideband, i.e.
    # most of the power in the half plane of the carrier
    P = np.abs(S)**2
    half = (KX*k_c[0] + KY*k_c[1] > 0) & (np.hypot(KX, KY) > 3*dk)
    inside = half & (np.hypot(KX - k_c[0], KY - k_c[1]) <= radius)
    if np.sum(P[half]) > 0:
        captured = np.sum(P[inside])/np.sum(P[half])
        if captured < 0.9:
            warnings.warn("the sideband window holds only {:.0f} % of the "
                          "sideband power: orders overlap or the window is "
                          "too small, increase the carrier".format(100*captured))

    if not remove_carrier:
        phase = _wrap(phase + k_c[0]*X + k_c[1]*Y)
    if mask is not None:
        phase = np.where(mask, phase, 0.0)
        modulation = np.where(mask, modulation, 0.0)
        c = np.where(mask, c, 0.0)

    if return_sideband:
        return phase, modulation, c

    return phase, modulation


# --- spherical surfaces -------------------------------------------------------

def aplanatic_directions(x, y, R_pupil, NA, x0=0.0, y0=0.0):
    """Directions of the rays of a converging spherical wave from an
    aplanatic transmission sphere (sine condition: sin(theta) = NA*rho).

    The pupil point (x, y) (rho = r/R_pupil) belongs to the ray that
    converges towards the focus on the axis with the angle theta, i.e. it
    travels in +z and towards the axis.

    Parameters
    ----------
    x, y : arrays
        Linear coordinates of the grid (e.g. of the detector, on which the
        pupil is imaged).
    R_pupil : float
        Radius of the pupil on the grid.
    NA : float
        Numerical aperture of the test wave.
    x0, y0 : floats, optional
        Centre of the pupil.

    Returns
    -------
    directions : array of shape (len(y), len(x), 3)
        Unit vectors (n_x, n_y, n_z), NaN outside the pupil.
    """

    X, Y = ensure_meshgrid(x, y)
    u, v = NA*(X - x0)/R_pupil, NA*(Y - y0)/R_pupil
    s2 = u**2 + v**2
    with np.errstate(invalid="ignore"):
        n = np.stack([-u, -v, np.sqrt(1 - s2)], axis=-1)
    n[s2 > NA**2*(1 + 1e-12)] = np.nan

    return n


def sphere_displacement_phase(directions, displacement, wavelength, passes=2):
    """Phase passes*k*(d.n) of a test sphere displaced by d from its null
    position (first order in d), see the module docstring.

    Parameters
    ----------
    directions : array of shape (..., 3)
        Unit directions n of the test wave (e.g. from aplanatic_directions()
        or raytrace.System.ray_map()).
    displacement : (d_x, d_y, d_z)
        Displacement of the sphere in length units.
    wavelength : float
    passes : int, optional
        2 for a surface in reflection.

    Returns
    -------
    phase : array
    """

    d = np.asarray(displacement, dtype=float)

    return passes*2*pi/wavelength*np.tensordot(directions, d, axes=([-1], [0]))


def fit_sphere_displacement(phase, directions, wavelength, mask=None,
                            lateral=True, passes=2):
    """Least squares fit of the displacement of a test sphere from its null
    position to its unwrapped phase.

    Fits phi = c_0 + passes*k*(d.n) (see sphere_displacement_phase()). The
    axial component d_z from the confocal and the cat's eye interferograms
    gives the null positions and the radius of curvature: with the stage
    positions z_1 (near confocal) and z_2 (near cat's eye),
    R = (z_2 - d_z2) - (z_1 - d_z1).

    Parameters
    ----------
    phase : array
        Unwrapped phase of the test beam relative to the reference.
    directions : array of shape phase.shape + (3,)
        Unit directions of the test wave at each pixel.
    wavelength : float
    mask : array of bool, optional
        Pixels to use (default: all with finite phase and directions).
    lateral : bool, optional
        Fit the lateral displacement (d_x, d_y) as well. In cat's eye, set it
        to False and fit tilt terms instead (lateral displacements do not
        show there; a tilt of the reference does).
    passes : int, optional

    Returns
    -------
    displacement : array (d_x, d_y, d_z)
        (d_x, d_y are 0 for lateral=False.)
    residual : array
        phase minus the fit (0 outside the mask).
    """

    phase = np.asarray(phase, dtype=float)
    n = np.asarray(directions, dtype=float)
    valid = np.isfinite(phase) & np.all(np.isfinite(n), axis=-1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    s = passes*2*pi/wavelength
    nv = n[valid]
    if lateral:
        A = np.column_stack([np.ones(len(nv)), s*nv[:, 0], s*nv[:, 1], s*nv[:, 2]])
    else:
        # tilt of the reference: linear in the transverse direction components
        A = np.column_stack([np.ones(len(nv)), nv[:, 0], nv[:, 1], s*nv[:, 2]])
    coeffs = np.linalg.lstsq(A, phase[valid], rcond=None)[0]

    fit = np.zeros_like(phase)
    fit[valid] = A @ coeffs
    residual = np.where(valid, phase - fit, 0.0)
    if lateral:
        d = coeffs[1:4]
    else:
        d = np.array([0.0, 0.0, coeffs[3]])

    return d, residual
