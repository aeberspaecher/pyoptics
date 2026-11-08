#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wavefront shaping: focusing light through scattering media.

A scattering medium mixes the N segments of a spatial light modulator (SLM)
in front of it into a speckle pattern behind it. As long as the medium does
not change, this mixing is linear: the field in an output sample m is
E_m = sum_n t_mn A_n exp(i phi_n), with the transmission matrix (TM) t_mn
and the complex amplitudes A_n exp(i phi_n) of the segments. Shaping the
phases phi_n so that all contributions to one sample arrive in phase makes a
focus whose intensity is about (pi/4) N times the mean speckle intensity
(Vellekoop and Mosk 2007).

This module contains

- random media and SLM segments: `random_phase_screen()`,
  `segment_labels()`, `segment_field()`, `transmission_matrix()` (probes any
  linear propagator segment by segment), `random_unitary()`,
- focusing with a known TM: `phase_conjugation()` for full, phase-only,
  P-level phase and binary amplitude modulation, `enhancement()` and the
  theoretical values `expected_enhancement()`, and `tm_gerchberg_saxton()`,
  which shapes several foci or a pattern through the TM with the
  Gerchberg-Saxton loop of `pyoptics.ifta`,
- feedback algorithms that only measure the intensity in the target
  (Vellekoop and Mosk 2008): `stepwise_sequential()` (SSA and the
  continuous variant CSA) and `partitioning()` (RPA),
- measuring the TM with phase-stepping interferometry against a reference
  (Popoff et al. 2010): `measure_transmission_matrix()`, with the canonical
  or a Hadamard basis,
- random matrix theory for the singular values of a TM:
  `marchenko_pastur_pdf()`.

The feedback algorithms and the TM measurement talk to the experiment only
through a callable that takes the complex SLM settings (one value per
segment) and returns the measured intensity, a number for the feedback
algorithms or an array of camera pixels for the TM measurement. Noise, a
reference beam or a medium that changes over time are part of that callable.

References
----------
I. M. Vellekoop and A. P. Mosk, "Focusing coherent light through opaque
strongly scattering media", Opt. Lett. 32, 2309-2311 (2007).

I. M. Vellekoop and A. P. Mosk, "Phase control algorithms for focusing light
through turbid media", Opt. Commun. 281, 3071-3080 (2008).

S. M. Popoff, G. Lerosey, R. Carminati, M. Fink, A. C. Boccara and
S. Gigan, "Measuring the transmission matrix in optics: an approach to the
study and control of light propagation in disordered media", Phys. Rev.
Lett. 104, 100601 (2010).

D. Akbulut, T. J. Huisman, E. G. van Putten, W. L. Vos and A. P. Mosk,
"Focusing light through random photonic media by binary amplitude
modulation", Opt. Express 19, 4017-4029 (2011).

I. M. Vellekoop, "Feedback-based wavefront shaping", Opt. Express 23,
12189-12206 (2015).

R. Di Leonardo, F. Ianni and G. Ruocco, "Computer generation of optimal
holograms for optical trap arrays", Opt. Express 15, 1913-1922 (2007).

V. A. Marchenko and L. A. Pastur, "Distribution of eigenvalues for some sets
of random matrices", Math. USSR-Sbornik 1, 457-483 (1967).
"""

import numpy as np

from .ifta import IFTABase


# random media and SLM segments

def random_phase_screen(shape, correlation_length, sigma, dx=1.0, rng=None):
    """Random phase screen with Gaussian statistics and Gaussian correlation.

    White Gaussian noise is filtered with a Gaussian in the Fourier domain
    (periodic boundaries), so that the autocorrelation of the screen is
    sigma**2*exp(-r**2/correlation_length**2).

    Parameters
    ----------
    shape : tuple of int
        Shape of the screen, (Ny, Nx) (or any number of axes).
    correlation_length : number
        1/e radius of the autocorrelation, in the units of dx.
    sigma : number
        Standard deviation of the phase (rad). sigma >> 2 pi makes a ground
        glass diffuser without an unscattered (ballistic) part.
    dx : number, optional
        Sample spacing (the same along all axes).
    rng : numpy.random.Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    phase : array
        Phase screen, mean 0 and standard deviation sigma (exactly, after
        normalization).
    """

    rng = np.random.default_rng(rng)
    noise = rng.standard_normal(shape)
    # Gaussian autocorrelation exp(-r^2/l^2) <=> filter exp(-r^2/(l^2/2)) in
    # real space, i.e. exp(-k^2 l^2/8) in frequency space
    k2 = 0.0
    for axis, n in enumerate(shape):
        k = 2*np.pi*np.fft.fftfreq(n, dx)
        k2 = k2 + (k**2).reshape([-1 if i == axis else 1 for i in range(len(shape))])
    filtered = np.fft.ifftn(np.fft.fftn(noise)*np.exp(-k2*correlation_length**2/8)).real
    filtered -= filtered.mean()
    return sigma*filtered/filtered.std()


def segment_labels(x, y, n_x, width, height=None, n_y=None, x0=0.0, y0=0.0):
    """Label the samples of a grid by the SLM segment (macro pixel) they belong to.

    The rectangle width x height centered at (x0, y0) is divided into
    n_x x n_y equal segments, numbered row by row (along x first) from the
    lower left corner.

    Parameters
    ----------
    x, y : arrays
        Linear coordinates of the grid.
    n_x : int
        Number of segments along x.
    width : number
        Width of the segmented rectangle.
    height : number, optional
        Height (default: width).
    n_y : int, optional
        Number of segments along y (default: n_x).
    x0, y0 : numbers, optional
        Center of the rectangle.

    Returns
    -------
    labels : int array, shape (len(y), len(x))
        Segment index of each sample, -1 outside the rectangle.
    """

    height = width if height is None else height
    n_y = n_x if n_y is None else n_y
    ix = np.floor((np.asarray(x) - x0 + width/2)/width*n_x).astype(int)
    iy = np.floor((np.asarray(y) - y0 + height/2)/height*n_y).astype(int)
    IX, IY = np.meshgrid(ix, iy)
    inside = (IX >= 0) & (IX < n_x) & (IY >= 0) & (IY < n_y)
    return np.where(inside, IY*n_x + IX, -1)


def segment_field(values, labels):
    """Field of an SLM: one complex value per segment.

    Parameters
    ----------
    values : array, shape (N,) or (K, N)
        Complex value of each segment (A_n exp(i phi_n)), or K settings.
    labels : int array
        Segment labels, see `segment_labels()`; samples labelled -1 are 0.

    Returns
    -------
    field : array, shape labels.shape (or (K,) + labels.shape)
    """

    values = np.asarray(values)
    padded = np.concatenate([values, np.zeros(values.shape[:-1] + (1,), dtype=values.dtype)], axis=-1)
    return padded[..., labels]          # label -1 picks the appended zero


def transmission_matrix(propagate, labels, n_segments=None, output=None, batch=1):
    """Transmission matrix of a linear optical system, segment by segment.

    Column n is the output field with only segment n switched on (amplitude
    1, phase 0).

    Parameters
    ----------
    propagate : callable
        Maps an input field (shape labels.shape) to the output field. With
        batch > 1 it is called with a stack of fields, shape
        (k,) + labels.shape, and has to return a stack.
    labels : int array
        Segment labels of the input plane, see `segment_labels()`.
    n_segments : int, optional
        Number of segments (default: labels.max() + 1).
    output : bool array, int index array or callable, optional
        Selects the output samples that make the rows: a boolean mask or
        an index array into the flattened output field, or a callable
        that maps the output field to a 1d array. Default: all samples.
    batch : int, optional
        Number of segments propagated together.

    Returns
    -------
    T : complex array, shape (M, N)
    """

    n_segments = int(labels.max()) + 1 if n_segments is None else n_segments

    def select(field):
        if output is None:
            return field.reshape(-1)
        if callable(output):
            return np.asarray(output(field)).reshape(-1)
        output_ = np.asarray(output)
        if output_.dtype == bool:
            return field[output_]
        return field.reshape(-1)[output_]

    columns = []
    for start in range(0, n_segments, batch):
        stop = min(start + batch, n_segments)
        settings = np.zeros((stop - start, n_segments), dtype=complex)
        settings[np.arange(stop - start), np.arange(start, stop)] = 1.0
        fields = segment_field(settings, labels)
        if batch == 1:
            columns.append(select(propagate(fields[0])))
        else:
            out = propagate(fields)
            columns.extend(select(o) for o in out)
    return np.array(columns).T


def random_unitary(n, rng=None):
    """Random unitary matrix, uniformly distributed (Haar measure).

    The Q factor of the QR decomposition of a complex Gaussian matrix, with
    the phases of R's diagonal moved into Q (Mezzadri 2007).

    Parameters
    ----------
    n : int
        Size.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    U : complex array, shape (n, n)
    """

    rng = np.random.default_rng(rng)
    Z = (rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))/np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q*(d/np.abs(d))


# focusing with a known TM

def _check_modulation(modulation, levels):
    if modulation not in ("full", "phase", "binary_amplitude"):
        raise ValueError(f"Unknown modulation '{modulation}', use 'full', 'phase' or 'binary_amplitude'.")
    if levels is not None and modulation != "phase":
        raise ValueError("levels only applies to modulation='phase'.")
    if levels is not None and levels < 2:
        raise ValueError("levels must be at least 2.")


def phase_conjugation(t, modulation="phase", levels=None):
    """SLM settings that focus into the output sample(s) of a known TM row.

    Parameters
    ----------
    t : array, shape (N,)
        Row of the transmission matrix that belongs to the target (or any
        linear combination of rows, e.g. sum_m w_m t_m for several foci).
    modulation : {"phase", "full", "binary_amplitude"}, optional
        "phase": unit amplitudes, phases -arg(t_n) (optionally quantized to
        `levels` equidistant values). "full": amplitude and phase,
        ``conj(t)/|t|*sqrt(N)``, the same input power as N unit segments.
        "binary_amplitude": segments on (1) or off (0), on where
        Re(t_n exp(-i theta)) > 0, with the reference phase theta that
        gives the largest focus (as with a digital micromirror device).
    levels : int, optional
        Number of phase levels for modulation="phase" (default:
        continuous).

    Returns
    -------
    values : complex array, shape (N,)
        SLM settings, one per segment.
    """

    _check_modulation(modulation, levels)
    t = np.asarray(t)
    N = len(t)
    if modulation == "full":
        return np.conj(t)/np.linalg.norm(t)*np.sqrt(N)
    if modulation == "phase":
        phase = -np.angle(t)
        if levels is not None:
            step = 2*np.pi/levels
            phase = step*np.round(phase/step)
        return np.exp(1j*phase)
    # binary amplitude: try the half planes that start at each arg(t_n)
    # (the optimum is one of them) and keep the best
    thetas = np.angle(t)
    best, best_I = None, -1.0
    for theta in np.concatenate([thetas - np.pi/2 + 1e-12, thetas + np.pi/2 - 1e-12]):
        on = np.real(t*np.exp(-1j*theta)) > 0
        intensity = np.abs(np.sum(t[on]))**2
        if intensity > best_I:
            best, best_I = on, intensity
    return best.astype(complex)


def enhancement(t, values):
    """Enhancement of a focus: its intensity over the mean speckle intensity.

    The mean speckle intensity is the ensemble average over random SLM phases
    for the same input power spread evenly over the segments,
    ``sum_n |t_n|**2 * sum_n |A_n|**2/N``, i.e. the intensity this output sample
    gets on average without shaping (Vellekoop and Mosk 2007). For
    phase-only settings this is ``sum_n |t_n|**2``.

    Parameters
    ----------
    t : array, shape (N,) or (M, N)
        TM row(s) of the target sample(s).
    values : array, shape (N,)
        SLM settings.

    Returns
    -------
    eta : number or array, shape (M,)
    """

    t = np.asarray(t)
    values = np.asarray(values)
    power = np.sum(np.abs(values)**2)/len(values)
    return np.abs(t @ values)**2/(power*np.sum(np.abs(t)**2, axis=-1))


def expected_enhancement(N, modulation="phase", levels=None):
    """Expected enhancement for a TM with independent circular Gaussian elements.

    With ``E|t|^2 = 1`` and Rayleigh distributed ``|t_n|``:

    - full (amplitude and phase) control: eta = N,
    - phase only: eta = 1 + (N - 1) pi/4, because ``(E|t|)^2 = pi/4``,
    - P phase levels: eta = 1 + (N - 1) pi/4 sinc^2(pi/P), with
      sinc(x) = sin(x)/x (1 + (N - 1)/pi for binary phase),
    - binary amplitude: eta = 1 + (N - 1)/(2 pi), relative to the mean
      speckle for the same input power (the same number of segments
      switched on), for the segments in a fixed half plane
      Re(t_n exp(-i theta)) > 0. Choosing the best theta, as
      `phase_conjugation()` does, adds a fluctuation term that decays as
      1/sqrt(N) (about +18 % for N = 256, +9 % for N = 1024).

    Parameters
    ----------
    N : int or array
        Number of segments.
    modulation : {"phase", "full", "binary_amplitude"}, optional
    levels : int, optional
        Number of phase levels for modulation="phase".

    Returns
    -------
    eta : number or array
    """

    _check_modulation(modulation, levels)
    N = np.asarray(N, dtype=float)
    if modulation == "full":
        return N
    if modulation == "binary_amplitude":
        return 1 + (N - 1)/(2*np.pi)
    factor = 1.0 if levels is None else np.sinc(1.0/levels)**2   # np.sinc(x) = sin(pi x)/(pi x)
    return 1 + (N - 1)*np.pi/4*factor


def tm_gerchberg_saxton(T, target_amplitude, n_iter=50, start=None, weighted=True):
    """Phase-only SLM settings for a pattern of several foci or an image.

    A Gerchberg-Saxton loop with `pyoptics.ifta.IFTABase` between the SLM
    (object space, unit amplitudes) and the output samples (image space,
    target amplitudes with free phases), with the TM as forward transform
    and its conjugate transpose as backward transform. The output phases are
    free, so the result does not depend on the phase of a reference that
    multiplies the rows of a measured TM.

    Plain Gerchberg-Saxton stagnates with foci of unequal height. The
    weighted variant (GSW, Di Leonardo et al. 2007) multiplies the target
    amplitude of every output sample by the ratio of the mean output
    amplitude to its own in every iteration, which pushes weak foci up and
    strong ones down until they are equal.

    Parameters
    ----------
    T : complex array, shape (M, N)
        Transmission matrix to the output samples.
    target_amplitude : array, shape (M,)
        Wanted amplitudes (square root of the intensities).
    n_iter : int, optional
        Number of iterations.
    start : complex array, shape (M,), optional
        Start field in the output plane (default: the target amplitude with
        zero phase, whose first backward transform is the phase conjugate).
    weighted : bool, optional
        Use the weighted algorithm (default) instead of plain
        Gerchberg-Saxton.

    Returns
    -------
    values : complex array, shape (N,)
        Phase-only SLM settings.
    output : complex array, shape (M,)
        Output field of these settings.
    """

    T = np.asarray(T)
    target_amplitude = np.asarray(target_amplitude, dtype=float)
    TH = T.conj().T
    unit = lambda u: np.exp(1j*np.angle(u))
    on = target_amplitude > 0
    weights = np.ones_like(target_amplitude)

    def image_constraint(obj, img, old_obj, old_img, i):
        if weighted and i > 1:
            ratio = np.abs(img[on])/target_amplitude[on]
            weights[on] *= np.mean(ratio)/np.maximum(ratio, 1e-12*np.mean(ratio))
        return weights*target_amplitude*unit(img)

    algorithm = IFTABase(
        to_image_space=lambda obj, img, old_obj, old_img, i: T @ obj,
        to_object_space=lambda obj, img, old_obj, old_img, i: TH @ img,
        image_space_constrainer=image_constraint,
        object_space_constrainer=lambda obj, img, old_obj, old_img, i: unit(obj),
        stopping_condition=lambda obj, img, old_obj, old_img, i: i > n_iter)
    start = target_amplitude.astype(complex) if start is None else np.asarray(start, dtype=complex)
    values, output, _ = algorithm.iterate(start)
    return values, output


# feedback algorithms

def _fit_phase(phases, intensities):
    """Phase that maximizes I(phi) = a + b cos(phi - phi_0), from samples."""

    return np.angle(np.sum(np.asarray(intensities)*np.exp(1j*np.asarray(phases))))


def stepwise_sequential(measure, n_segments, n_phases=8, continuous=False, n_sweeps=1,
                        start=None, callback=None):
    """Sequential feedback algorithm: optimize one segment after the other.

    The phase of segment n is stepped through n_phases values, the target
    intensity measured for each, and the phase of the maximum of the fitted
    cosine I(phi) = a + b cos(phi - phi_0) kept. In the stepwise sequential
    algorithm (SSA) all segments are measured against the same (initial)
    background and all phases are set at the end of a sweep; in the
    continuous sequential algorithm (CSA) every phase is set right after it
    has been measured, so the signal grows during the sweep (Vellekoop and
    Mosk 2008).

    Parameters
    ----------
    measure : callable
        measure(values) -> intensity in the target for the complex SLM
        settings values (shape (n_segments,)). May include noise.
    n_segments : int
    n_phases : int, optional
        Phase steps per segment (at least 3).
    continuous : bool, optional
        CSA instead of SSA.
    n_sweeps : int, optional
        Number of passes over all segments.
    start : complex array, optional
        Initial SLM settings (default: all 1).
    callback : callable, optional
        callback(values, n_measurements) after every update of the
        settings, e.g. to record the noise-free enhancement.

    Returns
    -------
    values : complex array, shape (n_segments,)
        Final SLM settings (unit amplitudes).
    n_measurements : int
        Number of calls of measure().
    """

    if n_phases < 3:
        raise ValueError("n_phases must be at least 3 to fit the cosine.")
    values = np.ones(n_segments, dtype=complex) if start is None else np.array(start, dtype=complex)
    phases = 2*np.pi*np.arange(n_phases)/n_phases
    count = 0
    for _ in range(n_sweeps):
        reference = values.copy()
        new_phase = np.angle(values)
        for n in range(n_segments):
            trial = (values if continuous else reference).copy()
            base = np.angle(trial[n])
            intensities = []
            for p in phases:
                trial[n] = np.exp(1j*(base + p))
                intensities.append(measure(trial))
                count += 1
            new_phase[n] = base + _fit_phase(phases, intensities)
            if continuous:
                values[n] = np.exp(1j*new_phase[n])
                if callback is not None:
                    callback(values, count)
        if not continuous:
            values = np.exp(1j*new_phase)
            if callback is not None:
                callback(values, count)
    return values, count


def partitioning(measure, n_segments, n_iterations, n_phases=8, start=None, rng=None,
                 callback=None):
    """Random partitioning algorithm (RPA): optimize random halves of the SLM.

    In every iteration a random half of the segments gets a common phase
    offset, stepped through n_phases values; the best offset (from the fitted
    cosine) is kept. Each step changes half of the field, so the signal
    changes much more than with a single segment, which makes the algorithm
    robust against noise, at the cost of a slower final convergence
    (Vellekoop and Mosk 2008).

    Parameters
    ----------
    measure : callable
        measure(values) -> intensity in the target.
    n_segments : int
    n_iterations : int
        Number of partitions tried.
    n_phases : int, optional
        Phase steps per partition (at least 3).
    start : complex array, optional
        Initial SLM settings (default: all 1).
    rng : numpy.random.Generator or int, optional
    callback : callable, optional
        callback(values, n_measurements) after every iteration.

    Returns
    -------
    values : complex array, shape (n_segments,)
    n_measurements : int
    """

    if n_phases < 3:
        raise ValueError("n_phases must be at least 3 to fit the cosine.")
    rng = np.random.default_rng(rng)
    values = np.ones(n_segments, dtype=complex) if start is None else np.array(start, dtype=complex)
    phases = 2*np.pi*np.arange(n_phases)/n_phases
    count = 0
    for _ in range(n_iterations):
        half = rng.permutation(n_segments)[:n_segments//2]
        intensities = []
        for p in phases:
            trial = values.copy()
            trial[half] *= np.exp(1j*p)
            intensities.append(measure(trial))
            count += 1
        values[half] *= np.exp(1j*_fit_phase(phases, intensities))
        if callback is not None:
            callback(values, count)
    return values, count


# measuring the TM

def hadamard_basis(n):
    """Hadamard basis of +-1 patterns (n a power of 2).

    Parameters
    ----------
    n : int

    Returns
    -------
    H : array, shape (n, n)
        Columns are the patterns; H @ H.T = n*I.
    """

    if n < 1 or n & (n - 1):
        raise ValueError("The Hadamard basis needs a power of 2.")
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def measure_transmission_matrix(measure, n_segments, n_steps=4, basis="canonical"):
    """Measure a transmission matrix by phase-stepping interferometry.

    For every input pattern b_j (a column of the basis) the phase of the
    pattern is stepped, phi_k = 2 pi k/n_steps, and the camera measures
    ``I_k = |r + exp(i phi_k) T b_j|**2`` with a reference field r that the
    experiment adds (e.g. light from an unmodulated part of the SLM, as in
    Popoff et al. 2010). Then sum_k I_k exp(-i phi_k)/n_steps = conj(r) T b_j
    in every pixel at once, and the TM follows by inverting the basis. The
    result is the observed TM, diag(conj(r)) T: every row carries the
    unknown reference of its pixel, which does not matter for focusing into
    that pixel.

    Parameters
    ----------
    measure : callable
        measure(values) -> intensities in the M camera pixels (shape (M,))
        for the complex SLM settings values (shape (n_segments,)),
        including the reference field and any noise.
    n_segments : int
    n_steps : int, optional
        Phase steps per pattern (at least 3).
    basis : "canonical", "hadamard" or array, optional
        Input patterns: one segment at a time ("canonical"), the +-1 patterns
        of `hadamard_basis()` (all segments on, a much larger signal against
        the noise), or any invertible (n_segments, n_segments) matrix whose
        columns are the patterns.

    Returns
    -------
    T_obs : complex array, shape (M, n_segments)
        Observed TM, rows multiplied by the conjugate reference field.
    n_measurements : int
    """

    if n_steps < 3:
        raise ValueError("n_steps must be at least 3.")
    if isinstance(basis, str):
        if basis == "canonical":
            B = np.eye(n_segments)
        elif basis == "hadamard":
            B = hadamard_basis(n_segments)
        else:
            raise ValueError(f"Unknown basis '{basis}'.")
    else:
        B = np.asarray(basis)
    phases = 2*np.pi*np.arange(n_steps)/n_steps
    columns = []
    for j in range(B.shape[1]):
        acc = 0.0
        for p in phases:
            acc = acc + np.asarray(measure(B[:, j]*np.exp(1j*p)))*np.exp(-1j*p)
        columns.append(acc/n_steps)
    T_basis = np.array(columns).T            # conj(r) T B
    if isinstance(basis, str) and basis == "canonical":
        T_obs = T_basis
    elif isinstance(basis, str) and basis == "hadamard":
        T_obs = T_basis @ B.T/n_segments
    else:
        T_obs = T_basis @ np.linalg.inv(B)
    return T_obs, n_steps*B.shape[1]


# random matrix theory

def marchenko_pastur_pdf(s, ratio=1.0):
    """Density of the singular values of a large random matrix.

    For an M x N matrix (N <= M) with independent elements of zero mean and
    variance 1/M, the eigenvalues lambda = s**2 of T^H T follow the
    Marchenko-Pastur law with gamma = N/M,
    ``p(lambda) = sqrt((l_+ - lambda)(lambda - l_-))/(2 pi gamma lambda)``,
    ``l_+- = (1 +- sqrt(gamma))**2``; this function returns the density of
    the singular values, 2 s p(s**2). For a square matrix it is the quarter
    circle sqrt(4 - s**2)/pi on [0, 2].

    Parameters
    ----------
    s : number or array
        Singular values.
    ratio : number, optional
        gamma = N/M, between 0 and 1.

    Returns
    -------
    pdf : number or array
    """

    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1].")
    s = np.asarray(s, dtype=float)
    lam = s**2
    lo, hi = (1 - np.sqrt(ratio))**2, (1 + np.sqrt(ratio))**2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.sqrt(np.clip((hi - lam)*(lam - lo), 0, None))/(2*np.pi*ratio*lam)
    return np.where((lam > lo) & (lam < hi), 2*s*p, 0.0)
