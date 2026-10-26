#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Phase retrieval: recover a phase from intensity measurements.

Two problems are covered:

1. Support-based phase retrieval (coherent diffractive imaging): an object g
   of limited extent (its support) is to be found from the magnitude ``|FT(g)|``
   of its far field alone. ER() implements Fienup's error reduction, HIO()
   the hybrid input-output algorithm, both with support and optionally
   positivity constraints and a shrink-wrap support update (ShrinkWrap,
   Marchesini et al. 2003). With a distance z, the detector sits in the near
   field instead (Fresnel coherent diffractive imaging): the Fourier
   transforms are replaced by a pyoptics propagator to z and back.

2. Pupil phase retrieval from through-focus PSFs (phase diversity): the
   pupil phase of an imaging system is found from intensity PSFs measured at
   known defocus values. phase_diversity() offers a multi-plane
   Gerchberg-Saxton (Misell) iteration and a least squares fit of Fringe
   Zernike coefficients using the PSF model of imaging.get_psf().

Conventions
-----------
Fields vary as exp(i(k.r - omega*t)). Far field magnitudes are those of
fft.FT() (centered, unnormalized): magnitude = abs(FT(g)). Pupils live on the
centered frequency grid of the image plane grid x, y, as in imaging, with
normalized pupil coordinates (P_x, P_y) = (k_x, k_y)/(k*NA); the PSF is
inv_FT() of the pupil function. Pupil phases are in radians, Zernike
coefficients (Fringe numbering, R_norm = 1 on the unit pupil) in waves, as in
litho.

Ambiguities
-----------
abs(FT(g)) does not change if g is shifted cyclically by whole pixels,
multiplied by a constant phase factor or replaced by its twin
conj(g(-x, -y)). Support-based retrieval therefore can recover g only up to
these operations; align() removes them for a comparison with a reference.
In the near field (z given), only the constant phase factor remains: the
curvature of the Fresnel kernel distinguishes the object from its twin, and
a shift of the object shifts the measured magnitude.
Likewise, a single in-focus PSF cannot distinguish a pupil phase phi(P) from
-phi(-P), so the sign of the even Zernikes (defocus, astigmatism, spherical
aberration, ...) is undetermined. A known defocus applied to some of the
measurements (phase diversity) removes this ambiguity.
"""

from dataclasses import dataclass, field as dc_field

import numpy as np
from numpy import pi
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .fft import FT, inv_FT
from .propagators import rayleigh_sommerfeld_I_TF
from .utils import freq_grid, get_dx, k_z
from .bases import fringe_zernike
from .imaging import pupil_coordinates, pupil_function
from .unwrap import unwrap_quality_guided


@dataclass
class RetrievalResult:
    """Result of an iterative phase retrieval.

    Attributes
    ----------
    obj : array
        The retrieved object (ER(), HIO()) or pupil function
        (phase_diversity()).
    support : array of bool
        Final support (ER(), HIO()) or pupil (phase_diversity()).
    errors : array
        Error metric per iteration (see the function that returned the
        result).
    errors_object : array
        Object domain error per iteration (ER(), HIO() only): the fraction of
        the energy of the Fourier projected object that violates the object
        constraints.
    phase : array
        Pupil phase in radians on the frequency grid, zero outside the pupil
        (phase_diversity() only).
    zernikes : dict
        Fringe Zernike coefficients of the pupil phase in waves, {j: c_j} for
        j = 2, ..., n_zernike (phase_diversity() only).
    """

    obj: np.ndarray
    support: np.ndarray
    errors: np.ndarray
    errors_object: np.ndarray = None
    phase: np.ndarray = None
    zernikes: dict = dc_field(default_factory=dict)


# support-based phase retrieval:

def fourier_error(obj, magnitude):
    """Normalized rms deviation of abs(FT(obj)) from the measured magnitude.

    E_F = sqrt(sum((abs(FT(obj)) - magnitude)**2)/sum(magnitude**2)).

    Parameters
    ----------
    obj : 2d array
        Object estimate.
    magnitude : 2d array
        Measured Fourier magnitude (centered, as FT() returns it).

    Returns
    -------
    error : number
    """

    return np.sqrt(np.sum((np.abs(FT(obj)) - magnitude)**2)/np.sum(magnitude**2))


def autocorrelation_support(magnitude, threshold=0.04, sigma=1.0):
    """Initial support estimate from the autocorrelation of the object.

    The inverse transform of magnitude**2 is the autocorrelation of the
    object, whose support contains the support of the object (it is twice as
    wide). It is a safe, loose start for ShrinkWrap.

    Parameters
    ----------
    magnitude : array
        Far field magnitude abs(FT(g)).
    threshold : number, optional
        Fraction of the maximum of the (smoothed) autocorrelation above which
        a pixel belongs to the support.
    sigma : number, optional
        Width in pixels of a Gaussian smoothing applied before thresholding.

    Returns
    -------
    support : array of bool
    """

    autocorr = np.abs(inv_FT(magnitude**2))
    if sigma > 0:
        autocorr = gaussian_filter(autocorr, sigma, mode="wrap")

    return autocorr > threshold*np.max(autocorr)


class ShrinkWrap(object):
    """Shrink-wrap support update after Marchesini et al., Phys. Rev. B 68,
    140101 (2003).

    Every `every` iterations, the modulus of the current object estimate is
    smoothed by a Gaussian of width sigma (in pixels) and thresholded at
    `threshold` times its maximum; this is the new support. sigma is
    multiplied by `decay` after each update until it reaches sigma_min, so
    the support tightens gradually around the object.

    Parameters
    ----------
    every : int, optional
        Number of iterations between two support updates.
    sigma : number, optional
        Initial width of the Gaussian smoothing in pixels.
    sigma_min : number, optional
        Final width of the Gaussian smoothing.
    decay : number, optional
        Factor applied to sigma after every update.
    threshold : number, optional
        Fraction of the maximum of the smoothed modulus.
    """

    def __init__(self, every=20, sigma=3.0, sigma_min=1.0, decay=0.95, threshold=0.2):
        self.every = every
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay
        self.threshold = threshold

    def __call__(self, obj, support, iteration):
        """Return the (possibly) updated support after iteration
        `iteration` (counting from 1)."""

        if iteration % self.every != 0:
            return support

        blurred = gaussian_filter(np.abs(obj), self.sigma, mode="wrap")
        self.sigma = max(self.sigma*self.decay, self.sigma_min)

        return blurred > self.threshold*np.max(blurred)


def _start_object(magnitude, support, start, positivity, real, rng):
    if start is not None:
        return np.array(start, dtype=complex)

    rng = np.random.default_rng(rng)
    shape = np.shape(magnitude)
    if positivity or real:
        g = rng.uniform(0, 1, shape).astype(complex)
    else:
        g = rng.uniform(0, 1, shape)*np.exp(2j*pi*rng.uniform(0, 1, shape))

    return g*support


def _transforms(z, x, y, wavelength, n, propagator):
    """Forward (object to detector) and backward transforms."""

    if z is None:
        return FT, inv_FT

    if x is None or y is None or wavelength is None:
        raise ValueError("A detector distance z needs the grid x, y and the wavelength")
    if propagator is None:
        propagator = rayleigh_sommerfeld_I_TF
        if min(get_dx(x), get_dx(y)) < wavelength/(2*np.real(n)):
            # back propagation by -z would amplify evanescent waves
            raise ValueError("The grid contains evanescent spatial frequencies "
                             "(dx < wavelength/(2*n)); use dx >= wavelength/(2*n)")

    forward = lambda u: propagator(u, x, y, z, wavelength, n)[0]
    backward = lambda u: propagator(u, x, y, -z, wavelength, n)[0]

    return forward, backward


def _iterate(magnitude, support, N_iter, beta, hio, start, positivity, real,
             shrinkwrap, rng, callback, z, x, y, wavelength, n, propagator):
    """Common loop of ER and HIO."""

    forward, backward = _transforms(z, x, y, wavelength, n, propagator)
    magnitude = np.asarray(magnitude, dtype=float)
    if support is None:
        if z is not None:
            raise ValueError("In the near field (z given), a support is needed")
        support = autocorrelation_support(magnitude)
    support = np.asarray(support, dtype=bool)
    if support.shape != magnitude.shape:
        raise ValueError("support and magnitude need the same shape")
    if shrinkwrap is True:
        shrinkwrap = ShrinkWrap()
    real = real or positivity

    g = _start_object(magnitude, support, start, positivity, real, rng)
    errors, errors_obj = np.empty(N_iter), np.empty(N_iter)

    for i in range(N_iter):
        G = forward(g)
        errors[i] = np.sqrt(np.sum((np.abs(G) - magnitude)**2)/np.sum(magnitude**2))
        g_prime = backward(magnitude*np.exp(1j*np.angle(G)))  # Fourier projection
        if real:
            g_prime = g_prime.real.astype(complex)

        valid = support.copy()
        if positivity:
            valid &= (g_prime.real >= 0)

        E = np.sum(np.abs(g_prime)**2)
        errors_obj[i] = np.sqrt(np.sum(np.abs(g_prime[~valid])**2)/E) if E > 0 else 0.0

        if hio:
            g = np.where(valid, g_prime, g - beta*g_prime)
        else:
            g = np.where(valid, g_prime, 0.0)

        if shrinkwrap is not None:
            # the estimate is the part of g' that meets the constraints:
            support = shrinkwrap(np.where(valid, g_prime, 0.0), support, i + 1)

        if callback is not None:
            callback(i + 1, np.where(valid, g_prime, 0.0), support)

    # the estimate: last Fourier projection with the object constraints
    g_prime = backward(magnitude*np.exp(1j*np.angle(forward(g))))
    if real:
        g_prime = g_prime.real.astype(complex)
    valid = support & (g_prime.real >= 0) if positivity else support
    obj = np.where(valid, g_prime, 0.0)
    if real:
        obj = obj.real

    return RetrievalResult(obj=obj, support=support, errors=errors, errors_object=errors_obj)


def ER(magnitude, support, N_iter=200, start=None, positivity=False, real=False,
       shrinkwrap=None, rng=None, callback=None, z=None, x=None, y=None,
       wavelength=None, n=1.0, propagator=None):
    """Fienup's error reduction algorithm.

    Alternates the projection onto the measured Fourier magnitude,
    g' = inv_FT(magnitude*exp(i*angle(FT(g)))), with the object constraints:
    g_new = g' where g' meets the constraints (inside the support and, with
    positivity, Re(g') >= 0) and 0 elsewhere. The Fourier error never
    increases, but ER stagnates easily; it is mostly used to polish the
    result of HIO().

    Parameters
    ----------
    magnitude : array
        Measured far field magnitude abs(FT(g)) (the square root of the
        intensity), centered as returned by fft.FT().
    support : array of bool or None
        Pixels in which the object may be nonzero. None: start from
        autocorrelation_support() (use with shrinkwrap).
    N_iter : int, optional
        Number of iterations.
    start : array, optional
        Start object, e.g. the result of a previous run. Defaults to random
        values inside the support.
    positivity : bool, optional
        Require a real, nonnegative object.
    real : bool, optional
        Require a real object (implied by positivity).
    shrinkwrap : ShrinkWrap, True or None, optional
        Update the support during the iteration; True uses ShrinkWrap() with
        default parameters.
    rng : int or numpy.random.Generator, optional
        Seed for the random start object.
    callback : callable, optional
        Called after every iteration as callback(iteration, estimate,
        support), with the iteration counted from 1 and the current estimate
        (the Fourier projection with the object constraints applied), e.g.
        to record snapshots.
    z : number, optional
        Distance of the detector from the object. None (default): far field,
        magnitude = abs(fft.FT(g)). Otherwise magnitude is the modulus of the
        field propagated by z on the same grid, magnitude =
        abs(propagator(g, x, y, z, wavelength, n)[0]), and the projection
        propagates back by -z (Fresnel coherent diffractive imaging). The
        support is then required.
    x, y : arrays, optional
        Linear grid coordinates (needed with z).
    wavelength : number, optional
        Vacuum wavelength (needed with z).
    n : number, optional
        Refractive index between object and detector.
    propagator : callable, optional
        Grid preserving pyoptics propagator with the signature
        propagator(field, x, y, z, wavelength, n) -> (field, x, y), used for
        +z and -z, e.g. propagators.fresnel_TF or
        functools.partial(propagators.propagate_padded,
        propagators.rayleigh_sommerfeld_I_TF). Defaults to the angular
        spectrum method propagators.rayleigh_sommerfeld_I_TF, which needs a
        grid without evanescent frequencies, dx >= wavelength/(2*n), so that
        the back propagation is the inverse of the forward one.

    Returns
    -------
    result : RetrievalResult
        obj: retrieved object (real if positivity or real), support: final
        support, errors: normalized rms deviation of the modulus in the
        detector plane from magnitude, before each projection (in the far
        field this is fourier_error()), errors_object: object domain error.

    Notes
    -----
    In the far field, the object is found only up to a cyclic shift by whole
    pixels, a constant phase factor and the twin image conj(g(-x, -y)); see
    align(). In the near field, only the constant phase factor remains.
    """

    return _iterate(magnitude, support, N_iter, 0.0, False, start, positivity,
                    real, shrinkwrap, rng, callback, z, x, y, wavelength, n,
                    propagator)


def HIO(magnitude, support, beta=0.9, N_iter=500, start=None, positivity=False,
        real=False, shrinkwrap=None, rng=None, callback=None, z=None, x=None,
        y=None, wavelength=None, n=1.0, propagator=None):
    """Fienup's hybrid input-output algorithm.

    Like ER(), but outside the valid region the iterate is not set to zero:
    g_new = g' where g' meets the object constraints and g - beta*g'
    elsewhere (Fienup, Appl. Opt. 21, 2758 (1982)). This negative feedback
    pushes the iterate out of the local minima in which ER stagnates. The
    Fourier error of HIO does not decrease monotonically; a few ER()
    iterations with start=result.obj usually finish the reconstruction.

    Parameters
    ----------
    magnitude, support, N_iter, start, positivity, real, shrinkwrap, rng,
    callback, z, x, y, wavelength, n, propagator :
        See ER().
    beta : number, optional
        Feedback parameter, typically 0.5 to 1.

    Returns
    -------
    result : RetrievalResult
        See ER(). obj is the last Fourier projection with the object
        constraints applied (the iterate g itself is not an estimate of the
        object outside the support).
    """

    return _iterate(magnitude, support, N_iter, beta, True, start, positivity,
                    real, shrinkwrap, rng, callback, z, x, y, wavelength, n,
                    propagator)


def twin(obj):
    """Twin image conj(g(-x, -y)) on a centered grid; it has the same far
    field magnitude as g."""

    return np.conj(np.roll(obj[::-1, ::-1], (1, 1), axis=(0, 1)))


def align(obj, reference, allow_twin=True):
    """Remove the trivial ambiguities of support-based phase retrieval.

    Finds the cyclic shift by whole pixels, the constant phase factor and, if
    allow_twin, the choice between obj and its twin() that bring obj closest
    to reference.

    Parameters
    ----------
    obj : array
        Retrieved object.
    reference : array
        True object.
    allow_twin : bool, optional
        Also try the twin image.

    Returns
    -------
    aligned : array
        Shifted, phase corrected (and possibly twinned) obj.
    error : number
        Normalized rms difference norm(aligned - reference)/norm(reference).
    """

    best = None
    candidates = (obj, twin(obj)) if allow_twin else (obj,)
    for g in candidates:
        xcorr = np.fft.ifft2(np.fft.fft2(reference)*np.conj(np.fft.fft2(g)))
        shift = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
        g_shifted = np.roll(g, shift, axis=(0, 1))
        overlap = np.vdot(g_shifted, reference)
        if abs(overlap) > 0:
            g_shifted = g_shifted*overlap/abs(overlap)
        error = np.linalg.norm(g_shifted - reference)/np.linalg.norm(reference)
        if best is None or error < best[1]:
            best = (g_shifted, error)

    return best


# pupil phase retrieval with phase diversity:

def defocus_to_waves(dz, NA, k):
    """Fringe Zernike Z4 coefficient (in waves) of a paraxial defocus dz.

    Observing the PSF a distance dz behind the focal plane (dz > 0 further
    downstream, in the direction of propagation) multiplies the pupil by
    exp(i*k_z*dz); paraxially k_z = k - k*NA**2*P**2/2 with the normalized
    pupil radius P. Up to piston this is the Fringe Zernike
    Z4 = 2*P**2 - 1 with the coefficient ::

        c_4 = -k*dz*NA**2/(8*pi)  (waves).

    Parameters
    ----------
    dz : number or array
        Defocus distance.
    NA : number
        Numerical aperture as used in imaging (the pupil edge is at
        ``|k_perp|`` = k*NA).
    k : number
        Wavenumber in the image space.

    Returns
    -------
    c_4 : number or array
        Z4 coefficient in waves.
    """

    return -k*np.asarray(dz)*NA**2/(8*pi)


def _diversity_factors(x, y, NA, k, defocus_waves, defocus_z):
    """Pupil factors of the diversity planes, shape (M, Ny, Nx)."""

    if (defocus_waves is None) == (defocus_z is None):
        raise ValueError("Give either defocus_waves or defocus_z")
    if defocus_z is not None:
        # exact: the angular spectrum propagator exp(i*k_z*dz), piston removed
        K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=True)
        KZ = np.real(k_z(k, K_X, K_Y)) - k
        return np.exp(1j*np.atleast_1d(defocus_z).astype(float)[:, None, None]*KZ)

    Z4 = _pupil_zernikes(x, y, NA, k, [4])[0]
    return np.exp(2j*pi*np.atleast_1d(defocus_waves).astype(float)[:, None, None]*Z4)


def _pupil_zernikes(x, y, NA, k, indices):
    """Fringe Zernikes on the normalized pupil coordinates."""

    P_x, P_y = pupil_coordinates(x, y, NA, k)

    return np.array([fringe_zernike(j, P_x, P_y) for j in indices])


def diversity_psfs(x, y, NA, k, defocus_waves=None, pupil_mask=None, phase=None,
                   defocus_z=None):
    """Intensity PSFs at a series of known defoci.

    Parameters
    ----------
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Numerical aperture.
    k : number
        Wavenumber.
    defocus_waves : sequence of numbers, optional
        Fringe Z4 coefficient (waves) added to the pupil phase for each PSF,
        see defocus_to_waves() (paraxial defocus).
    pupil_mask, phase : arrays or callables, optional
        See imaging.pupil_function(); phase is the unknown pupil phase in
        radians.
    defocus_z : sequence of numbers, optional
        Instead of defocus_waves: distances of the observation planes from
        the focal plane (positive downstream). The pupil is multiplied by
        the exact angular spectrum propagator exp(i*(k_z - k)*dz), so the
        PSFs equal propagators.rayleigh_sommerfeld_I_TF() of the in-focus
        PSF.

    Returns
    -------
    psfs : array
        Intensity PSFs ``|inv_FT(P*D_m)|**2`` with the diversity factors D_m,
        shape (M, Ny, Nx), each normalized to a total of 1.
    """

    P = pupil_function(x, y, NA, k, pupil_mask, phase)
    D = _diversity_factors(x, y, NA, k, defocus_waves, defocus_z)
    psfs = np.abs(inv_FT(P[None]*D))**2

    return psfs/np.sum(psfs, axis=(1, 2), keepdims=True)


def _fit_zernikes_to_phase(phase, P_abs, Z, indices):
    """Weighted least squares fit of Zernikes to a phase on the pupil."""

    inside = P_abs > 0
    A = Z[:, inside].T*np.sqrt(P_abs[inside])[:, None]
    b = phase[inside]*np.sqrt(P_abs[inside])
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    return c


def phase_diversity(psfs, defocus_waves, x, y, NA, k, n_zernike=15, method="fit",
                    pupil_mask=None, N_iter=None, start=None, tol=1e-12, solver="trf",
                    defocus_z=None):
    """Retrieve the pupil phase from intensity PSFs at known defoci.

    Parameters
    ----------
    psfs : array
        Measured intensity PSFs, shape (M, Ny, Nx), sampled on x, y with the
        PSF at the grid center as in imaging.get_psf(). Their scale does not
        matter; each is normalized to the energy of the model PSF.
    defocus_waves : sequence of M numbers or None
        Known defocus of each PSF as Fringe Z4 coefficient in waves (see
        defocus_to_waves()), i.e. paraxial defocus. None if defocus_z is
        given. At least one value should be nonzero: with a
        single in-focus PSF, the sign of the even aberrations is ambiguous.
        Values of about +-0.5 to +-1 wave work well.
    x, y : arrays
        Linear image plane coordinates.
    NA : number
        Numerical aperture.
    k : number
        Wavenumber.
    n_zernike : int, optional
        Fringe Zernikes 2, ..., n_zernike describe the phase (piston is
        irrelevant).
    method : {"fit", "gs"}, optional
        "fit" (default): least squares fit of the Zernike coefficients to the
        measured intensities (scipy.optimize.least_squares with the analytic
        Jacobian). Fast and accurate if the start is within the basin of
        attraction; noise is handled in the least squares sense. "gs":
        multi-plane Gerchberg-Saxton (Misell) iteration, which cycles through
        the planes and replaces the modulus of the PSF by the measured one.
        It needs no basis, so it also finds phases that are not described by
        n_zernike Zernikes. Its phase is unwrapped on the pupil
        (unwrap.unwrap_quality_guided()) and returned as is; the Zernike
        coefficients are fitted to it. Convergence is slower than "fit",
        and a "gs" result can serve as start of a "fit".
    pupil_mask : array or callable, optional
        Known pupil amplitude, see imaging.pupil_function().
    N_iter : int, optional
        "gs": number of iterations (default 200). "fit": maximum number of
        function evaluations (default 200).
    start : dict or array, optional
        Start phase: Fringe Zernike coefficients {j: c_j} in waves or a phase
        array in radians on the frequency grid. Defaults to zero.
    tol : number, optional
        "fit" only: tolerances ftol, xtol and gtol of least_squares.
    solver : {"trf", "lm", "dogbox"}, optional
        "fit" only: algorithm of scipy.optimize.least_squares. The trust
        region default "trf" is robust; Levenberg-Marquardt ("lm") is
        somewhat faster on well posed problems but can run away on ill
        posed ones (e.g. a single in-focus PSF).
    defocus_z : sequence of M numbers, optional
        Instead of defocus_waves: the distances of the PSF planes from the
        focal plane (positive downstream, in the units of x and 2*pi/k), e.g.
        the positions of a focus stage. The defocus is applied exactly
        (non-paraxially) as the angular spectrum propagator, see
        diversity_psfs().

    Returns
    -------
    result : RetrievalResult
        obj: pupil function, support: pupil (bool), phase: pupil phase in
        radians (zero outside the pupil, piston removed), zernikes: Fringe
        Zernike coefficients in waves {j: c_j}, errors: normalized rms
        deviation of the model amplitudes sqrt(I) from the measured ones,
        per iteration ("gs") or per function evaluation ("fit").

    Notes
    -----
    The PSF grid has to sample the intensity: dx <= wavelength/(4*NA). The
    pupil is sampled with about L*NA/wavelength pixels across its radius (L:
    width of the window); a PSF window much larger than the PSF itself thus
    resolves the pupil better, as in imaging.
    """

    psfs = np.asarray(psfs, dtype=float)
    if psfs.ndim == 2:
        psfs = psfs[None]
    diversity = _diversity_factors(x, y, NA, k, defocus_waves, defocus_z)
    if len(psfs) != len(diversity):
        raise ValueError("Need one defocus value per PSF")
    if method not in ("fit", "gs"):
        raise ValueError("method must be 'fit' or 'gs'")

    indices = list(range(2, n_zernike + 1))
    P0 = pupil_function(x, y, NA, k, pupil_mask)
    P_abs = np.abs(P0)
    inside = P_abs > 0
    Z = _pupil_zernikes(x, y, NA, k, indices)

    # normalize the measurements to the energy of the model (Parseval: the
    # energy of inv_FT(P*exp(i*phi)) does not depend on phi):
    E_model = np.sum(P_abs**2)/np.size(P0)
    I_meas = np.clip(psfs, 0, None)
    I_meas = I_meas*E_model/np.sum(I_meas, axis=(1, 2), keepdims=True)
    A_meas = np.sqrt(I_meas)
    norm_A = np.sqrt(np.sum(I_meas))

    if start is None:
        phi0 = np.zeros_like(P_abs)
    elif isinstance(start, dict):
        phi0 = 2*pi*sum(c*Z[indices.index(j)] for j, c in start.items())
    else:
        phi0 = np.asarray(start, dtype=float)

    if method == "gs":
        N_iter = 200 if N_iter is None else N_iter
        P = P0*np.exp(1j*phi0)
        errors = np.empty(N_iter)
        for i in range(N_iter):
            err = 0.0
            for d, A in zip(diversity, A_meas):
                psf = inv_FT(P*d)
                err += np.sum((np.abs(psf) - A)**2)
                P_new = FT(A*np.exp(1j*np.angle(psf)))*np.conj(d)
                P = P0*np.exp(1j*np.angle(P_new))
            errors[i] = np.sqrt(err)/norm_A

        # unwrap on the pupil, fit the Zernikes and remove the piston:
        phase = np.zeros_like(P_abs)
        phase[inside] = unwrap_quality_guided(np.angle(P), mask=inside)[inside]
        phase = (phase - np.sum(phase*P_abs)/np.sum(P_abs))*inside
        c = _fit_zernikes_to_phase(phase, P_abs, Z, indices)/(2*pi)
    else:
        N_iter = 200 if N_iter is None else N_iter
        c0 = _fit_zernikes_to_phase(phi0, P_abs, Z, indices)/(2*pi)
        errors = []

        def model(c):
            phi = 2*pi*np.tensordot(c, Z, axes=1)
            P = P0*np.exp(1j*phi)
            return P, inv_FT(P[None]*diversity)

        def residuals(c):
            _, psf = model(c)
            r = (np.abs(psf)**2 - I_meas).ravel()
            errors.append(np.sqrt(np.sum((np.abs(psf) - A_meas)**2))/norm_A)
            return r

        def jacobian(c):
            P, psf = model(c)
            J = np.empty((psf.size, len(c)))
            for m, Z_m in enumerate(Z):
                dpsf = inv_FT(2j*pi*Z_m*P[None]*diversity)
                J[:, m] = (2*np.real(np.conj(psf)*dpsf)).ravel()
            return J

        res = least_squares(residuals, c0, jac=jacobian, method=solver,
                            ftol=tol, xtol=tol, gtol=tol, max_nfev=N_iter)
        c = res.x
        phase = 2*pi*np.tensordot(c, Z, axes=1)*inside
        mean = np.sum(phase*P_abs)/np.sum(P_abs)
        phase = (phase - mean)*inside
        errors = np.array(errors)

    return RetrievalResult(obj=P0*np.exp(1j*phase), support=inside, errors=errors,
                           phase=phase, zernikes=dict(zip(indices, c)))
