#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Digital holography: recording, reconstruction, refocusing and autofocus.

A hologram is the intensity of an object wave O and a reference wave R on a
camera, ::

    H = |O + R|**2 = |O|**2 + |R|**2 + O*conj(R) + conj(O)*R.

The first two terms are the zero order, O*conj(R) is the object (+1 order)
and conj(O)*R its twin (-1 order). Recording and reconstruction:

- hologram(), phase_shifted_holograms(), reference_wave(): simulate off-axis
  and in-line holograms of a given object field (add camera noise with
  processing.CameraNoiser).
- off_axis_angle_range(), max_object_bandwidth(): the sampling condition that
  separates the orders on the camera.
- reconstruct_off_axis(): single shot reconstruction by sideband filtering.
- reconstruct_phase_shifting(): in-line or off-axis phase-shifting holography
  from N frames with known reference phase shifts (least squares, via
  interferometry.psi_n_step()).
- reconstruct_gabor(): in-line (Gabor) holography with plane wave
  illumination, optionally with iterative twin image suppression.
- suppress_zero_order(): remove the zero order from a hologram.
- find_carrier(): locate the +1 order in the hologram spectrum.

Working with the reconstructed field:

- refocus(): numerical refocusing with the angular spectrum or Fresnel
  propagator (padded).
- focus_metric(), autofocus(): sharpness metrics (amplitude variance, Tamura
  coefficient, gradient, Laplacian) and a search for the best focus.
- quantitative_phase(), remove_tilt(), optical_path_difference(): unwrapped
  phase maps (with pyoptics.unwrap), background compensation and OPD.

Conventions: fields vary as exp(i(k.r - omega*t)). The reference wave tilted
by the angles (theta_x, theta_y) is R = A*exp(i*(k_x*x + k_y*y + delta)) with
(k_x, k_y) = k*(sin(theta_x), sin(theta_y)), its "carrier". The object term
O*conj(R) thus sits at -(k_x, k_y) in the spectrum FT(H) and the twin at
+(k_x, k_y). All reconstructions return the object field O in the hologram
plane (scaled to the reference amplitude). An object at the distance d in
front of the camera is brought back into focus with refocus(O, x, y, -d, ...):
propagation by z > 0 multiplies each plane wave by exp(+i*k_z*z), so the
camera plane is reached from the object with +d and the object from the
camera with -d. Wavelengths are vacuum wavelengths; n is the refractive index
of the medium between object and camera.
"""

import warnings

import numpy as np

from .fft import FT, inv_FT
from .interferometry import psi_n_step
from .propagators import (propagate_padded, fresnel_TF,
                          _linear_coords)
from .utils import freq_grid, wavenumber, ensure_meshgrid, get_dx, k_z


def _carrier(ref_angle, wavelength, n=1.0):
    """Transverse wave vector (k_x, k_y) of a reference wave tilted by
    ref_angle = (theta_x, theta_y) (radians).
    """

    k = wavenumber(wavelength, n)
    theta_x, theta_y = ref_angle

    return k*np.sin(theta_x), k*np.sin(theta_y)


def _nyquist(x, y):
    """Largest wave numbers pi/dx, pi/dy the grid can represent."""

    x, y = _linear_coords(x, y)

    return np.pi/get_dx(x), np.pi/get_dx(y)


def reference_wave(x, y, wavelength, ref_angle=(0.0, 0.0), ref_amp=1.0,
                   phase_shift=0.0, n=1.0):
    """Tilted plane reference wave on the camera.

    Parameters
    ----------
    x, y : arrays
        Linear or meshed coordinates of the camera.
    wavelength : number
    ref_angle : tuple of two numbers, optional
        Tilt angles (theta_x, theta_y) in radians: the reference has the
        transverse wave vector k*(sin(theta_x), sin(theta_y)).
    ref_amp : number, optional
        Amplitude A of the reference.
    phase_shift : number, optional
        Additional phase delta (e.g. of a phase shifter).
    n : number, optional
        Refractive index.

    Returns
    -------
    R : array
        A*exp(i*(k_x*x + k_y*y + delta)).
    """

    X, Y = ensure_meshgrid(x, y)
    k_x, k_y = _carrier(ref_angle, wavelength, n)

    return ref_amp*np.exp(1j*(k_x*X + k_y*Y + phase_shift))


def max_object_bandwidth(x, y):
    """Largest object bandwidth for which an off-axis hologram on this grid
    can separate the orders.

    The +1 order (radius B) has to stay clear of the zero order (``|O|**2`` has
    radius 2B), so the carrier must satisfy ``|k_c|`` >= 3B, and it has to stay
    below the Nyquist frequency, ``|k_c|`` + B <= pi/dx. Both are possible for
    B <= pi/(4*dx) with the carrier along one axis of the grid. (A diagonal
    carrier allows somewhat more, but the axis-aligned value is the simple
    and safe bound.)

    Parameters
    ----------
    x, y : arrays
        Camera coordinates.

    Returns
    -------
    B_max : number
        Maximum object bandwidth k*NA (radius in k space) in rad per length
        unit, along the coarser axis.
    """

    k_ny_x, k_ny_y = _nyquist(x, y)

    return min(k_ny_x, k_ny_y)/4


def off_axis_angle_range(bandwidth, x, y, wavelength, n=1.0, axis="x"):
    """Range of reference tilt angles that separate the orders of an
    off-axis hologram.

    For an object with the bandwidth B = k*NA (radius of its angular
    spectrum), the +1 order must not overlap the zero order, ``|k_c|`` >= 3B, and
    must be sampled by the camera, ``|k_c|`` + B <= pi/dx, where k_c = k*sin(theta)
    is the carrier.

    Parameters
    ----------
    bandwidth : number
        Object bandwidth B (radius in k space, rad per length unit), e.g.
        k*NA of the imaging optics in front of the camera.
    x, y : arrays
        Camera coordinates.
    wavelength : number
    n : number, optional
    axis : "x" or "y", optional
        Direction of the tilt.

    Returns
    -------
    theta_min, theta_max : numbers
        Smallest and largest tilt in radians. Raises ValueError if no angle
        works (theta_min > theta_max); then the object has to be band limited
        further or the camera pixels have to be smaller.
    """

    k = wavenumber(wavelength, n)
    k_ny = _nyquist(x, y)[0 if axis == "x" else 1]

    kc_min, kc_max = 3*bandwidth, min(k_ny - bandwidth, k)
    if kc_min > kc_max:
        raise ValueError("No reference angle separates the orders: the object "
                         "bandwidth {:.4g} exceeds pi/(4*d) = {:.4g} (or k/3). "
                         "Reduce the NA or the pixel pitch.".format(bandwidth, k_ny/4))
    if kc_min > k:
        raise ValueError("The carrier would exceed the wave number k.")

    return np.arcsin(kc_min/k), np.arcsin(kc_max/k)


def hologram(obj_field, x, y, wavelength, ref_angle=(0.0, 0.0), ref_amp=1.0,
             phase_shift=0.0, n=1.0):
    """Record the hologram ``|O + R|**2`` of an object field O.

    Parameters
    ----------
    obj_field : array
        Object wave on the camera (i.e. already propagated there).
    x, y : arrays
        Camera coordinates.
    wavelength : number
    ref_angle : tuple of two numbers, optional
        Reference tilt (theta_x, theta_y) in radians; (0, 0) gives an in-line
        hologram, see reference_wave().
    ref_amp : number, optional
        Reference amplitude.
    phase_shift : number, optional
        Phase shift of the reference (phase-shifting holography).
    n : number, optional

    Returns
    -------
    H : array
        Hologram intensity. Apply processing.CameraNoiser for a noisy
        recording (after scaling to photon numbers with processing.Scaler).
        Warns if the carrier exceeds the Nyquist frequency of the camera.
    """

    k_x, k_y = _carrier(ref_angle, wavelength, n)
    k_ny_x, k_ny_y = _nyquist(x, y)
    if abs(k_x) >= k_ny_x or abs(k_y) >= k_ny_y:
        warnings.warn("The reference carrier exceeds the Nyquist frequency of "
                      "the camera: the fringes are aliased.")

    R = reference_wave(x, y, wavelength, ref_angle, ref_amp, phase_shift, n)

    return np.abs(obj_field + R)**2


def phase_shifted_holograms(obj_field, x, y, wavelength, shifts, ref_angle=(0.0, 0.0),
                            ref_amp=1.0, n=1.0):
    """Record a stack of holograms with phase-shifted references.

    Parameters
    ----------
    obj_field, x, y, wavelength, ref_angle, ref_amp, n
        As for hologram().
    shifts : sequence of numbers
        Reference phase shifts delta_j in radians, e.g.
        interferometry.phase_shifts(4) for the 4-step method.

    Returns
    -------
    frames : array of shape (Ny, Nx, len(shifts))
        Frame index last, as in pyoptics.processing and
        pyoptics.interferometry.
    """

    return np.stack([hologram(obj_field, x, y, wavelength, ref_angle, ref_amp, delta, n)
                     for delta in shifts], axis=-1)


def suppress_zero_order(H, x=None, y=None, method="mean", I_ref=None, I_obj=None,
                        radius=None):
    """Remove the zero order ``|O|**2`` + ``|R|**2`` from a hologram.

    Parameters
    ----------
    H : array
        Hologram.
    x, y : arrays, optional
        Camera coordinates (needed for method="highpass").
    method : str, optional
        - "mean": subtract the mean of H (removes the DC peak only).
        - "highpass": subtract a Gaussian low pass of H with the 1/e radius
          `radius` in k space (rad per length unit), which removes the slowly
          varying ``|O|**2`` + ``|R|**2`` for objects with little bandwidth.
        - "recorded": subtract separately recorded intensities I_ref = ``|R|**2``
          and I_obj = ``|O|**2`` (exact).
    I_ref, I_obj : arrays or numbers, optional
        Intensities of reference and object alone (method="recorded"; a
        missing I_obj is treated as zero).
    radius : number, optional
        Low pass radius for method="highpass".

    Returns
    -------
    H_filtered : array
        H without the zero order: ideally O*conj(R) + conj(O)*R.
    """

    H = np.asarray(H, dtype=float)

    if method == "mean":
        return H - np.mean(H)

    if method == "recorded":
        if I_ref is None:
            raise ValueError("method='recorded' needs I_ref")
        return H - I_ref - (0.0 if I_obj is None else I_obj)

    if method == "highpass":
        if x is None or y is None or radius is None:
            raise ValueError("method='highpass' needs x, y and radius")
        K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
        lowpass = np.exp(-(K_x**2 + K_y**2)/radius**2)
        return H - np.real(inv_FT(FT(H)*lowpass))

    raise ValueError("Unknown method '{}'".format(method))


def find_carrier(H, x, y, exclude_radius=None):
    """Locate the off-axis +1 order in the spectrum of a hologram.

    Parameters
    ----------
    H : array
        Hologram.
    x, y : arrays
        Camera coordinates.
    exclude_radius : number, optional
        Radius around k = 0 (rad per length unit) that is ignored, so that the
        zero order is not picked. Defaults to a tenth of the Nyquist frequency.

    Returns
    -------
    k_x, k_y : numbers
        Carrier of the reference (the +1 order O*conj(R) sits at -(k_x, k_y)),
        refined to sub-pixel precision by a parabola fit of ``log|FT(H)|``.

    Notes
    -----
    The spectrum of a real hologram is symmetric, ``|FT(H)(k)|`` = ``|FT(H)(-k)|``,
    so the +1 order cannot be told from the twin. The carrier with k_x > 0
    (or k_y > 0 if k_x = 0) is returned. If the true reference has the
    opposite sign, the reconstruction gives conj(O), which is in focus at +d
    instead of -d. Pass the known reference angle whenever possible.
    """

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    S = np.abs(FT(np.asarray(H, dtype=float) - np.mean(H)))

    if exclude_radius is None:
        exclude_radius = 0.1*min(_nyquist(x, y))
    S[K_x**2 + K_y**2 <= exclude_radius**2] = 0.0
    # pick one of the two symmetric orders: k_x > 0, or k_x = 0 and k_y > 0
    S[(K_x < 0) | ((K_x == 0) & (K_y <= 0))] = 0.0

    j, l = np.unravel_index(np.argmax(S), S.shape)
    k_x, k_y = K_x[0, :], K_y[:, 0]

    def refine(values, idx, axis_k):
        if idx == 0 or idx == len(values) - 1:
            return axis_k[idx]
        a, b, c = np.log(values[idx - 1:idx + 2] + 1e-300)
        denom = a - 2*b + c
        offset = 0.5*(a - c)/denom if denom != 0 else 0.0
        return axis_k[idx] + offset*(axis_k[1] - axis_k[0])

    # the +1 order of the spectrum of H sits at -k_c; the order picked here
    # (k_x > 0) is the twin, i.e. at +k_c
    return refine(S[j, :], l, k_x), refine(S[:, l], j, k_y)


def _window(K_r, radius, window):
    """Radial filter window of the given radius."""

    rho = K_r/radius
    if window in ("circle", "circular", None):
        return (rho <= 1).astype(float)
    if window == "hann":
        return np.where(rho <= 1, np.cos(0.5*np.pi*rho)**2, 0.0)
    if window == "tukey":
        # flat up to rho = 0.5, cosine taper to zero at rho = 1
        taper = np.cos(np.pi*(rho - 0.5))**2
        return np.where(rho <= 0.5, 1.0, np.where(rho <= 1, taper, 0.0))
    if window == "gaussian":
        return np.exp(-2*rho**2)

    raise ValueError("Unknown window '{}'".format(window))


def reconstruct_off_axis(H, x, y, wavelength, ref_angle=None, carrier=None, window="circle",
                         radius=None, ref_amp=1.0, n=1.0, reference=None, subtract_mean=True,
                         return_spectrum=False):
    """Reconstruct the object field from an off-axis hologram by sideband
    filtering.

    The hologram is multiplied by the (numerical) reference wave R, which
    moves the object term ``O*|R|**2`` to k = 0, the zero order to the carrier
    k_c and the twin to 2*k_c. A window of the given radius around k = 0 then
    selects the object.

    Parameters
    ----------
    H : array
        Off-axis hologram.
    x, y : arrays
        Camera coordinates.
    wavelength : number
    ref_angle : tuple of two numbers, optional
        Reference tilt (theta_x, theta_y) in radians.
    carrier : tuple of two numbers, optional
        Carrier (k_x, k_y) instead of ref_angle. If neither is given, it is
        found with find_carrier() (with the sign ambiguity described there).
    window : str or array, optional
        "circle" (default; hard edge: keeps the full object spectrum inside
        the radius, but rings if the spectrum does not vanish there),
        "tukey" (flat up to half the radius, cosine edge), "hann",
        "gaussian" (smooth: less ringing and noise, but they attenuate the
        object spectrum like an apodized pupil, so use a radius well above
        the object bandwidth), or an array on the k grid (centred at k = 0)
        that multiplies the shifted spectrum.
    radius : number, optional
        Window radius in rad per length unit; the object bandwidth k*NA if it
        is known. Defaults to ``|k_c|/3``, the largest radius that keeps the zero
        order out for an object fulfilling ``|k_c|`` >= 3B.
    ref_amp : number, optional
        Reference amplitude A; the result is divided by A.
    n : number, optional
    reference : array, optional
        The reference wave R on the camera, if it is not a plane wave (e.g.
        a spherical reference). It replaces the plane wave built from
        ref_angle or carrier, so that its curvature is removed before
        filtering; ref_angle or carrier still set the default window radius.
        A curved reference that is treated as a plane wave leaves a lens
        phase on O, which shifts the focus and the magnification on
        refocusing.
    subtract_mean : bool, optional
        If True, subtract the mean of H first. This removes the strong DC
        peak, whose spectral leakage (the reference carrier is in general not
        periodic in the window) would otherwise spread into the object order.
    return_spectrum : bool, optional
        If True, also return the shifted spectrum FT(H*R) and the window.

    Returns
    -------
    O : array
        Object field in the hologram plane.
    spectrum, window : arrays
        Only if return_spectrum is True.
    """

    if carrier is None:
        carrier = (_carrier(ref_angle, wavelength, n) if ref_angle is not None
                   else find_carrier(H, x, y))
    k_cx, k_cy = carrier
    k_c = np.hypot(k_cx, k_cy)

    if reference is None:
        X, Y = ensure_meshgrid(x, y)
        R = ref_amp*np.exp(1j*(k_cx*X + k_cy*Y))
    else:
        R = np.asarray(reference)
    H = np.asarray(H, dtype=float)
    if subtract_mean:
        H = H - np.mean(H)
    spectrum = FT(H*R)

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    if isinstance(window, np.ndarray):
        W = window
    else:
        if radius is None:
            if k_c == 0:
                raise ValueError("In-line hologram (carrier 0): give a radius, "
                                 "or use reconstruct_phase_shifting()")
            radius = k_c/3
        if radius > k_c/2:
            warnings.warn("The window radius exceeds half the carrier: the zero "
                          "order leaks into the reconstruction.")
        W = _window(np.hypot(K_x, K_y), radius, window)

    O = inv_FT(spectrum*W)/np.abs(R)**2

    if return_spectrum:
        return O, spectrum, W

    return O


def reconstruct_phase_shifting(frames, shifts=None, x=None, y=None, wavelength=None,
                               ref_angle=(0.0, 0.0), ref_amp=1.0, n=1.0, reference=None):
    """Reconstruct the object field from phase-shifted holograms.

    Each frame is H_j = ``|O + R*exp(i*delta_j)|**2`` = a + b*cos(phi - delta_j)
    with phi = arg(O*conj(R)) and b = ``2*|O*conj(R)|``. The least squares
    phase-shifting algorithm interferometry.psi_n_step() gives phi, a and the
    modulation b/a per pixel, hence C = O*conj(R) = (b/2)*exp(i*phi) and
    O = ``C*R/|R|**2``. This is exact for any N >= 3 distinct known shifts and
    removes the zero order and the twin image, also for in-line holograms.

    Parameters
    ----------
    frames : array of shape (Ny, Nx, N)
        Phase-shifted holograms, frame index last.
    shifts : sequence of N numbers, optional
        Reference phase shifts delta_j in radians. Defaults to 2*pi*j/N.
    x, y, wavelength : optional
        Camera coordinates and wavelength, needed to build the reference from
        ref_angle (not needed for an in-line reference).
    ref_angle : tuple of two numbers, optional
        Reference tilt (theta_x, theta_y); (0, 0) for in-line holography.
    ref_amp : number, optional
    n : number, optional
    reference : array, optional
        Reference wave R (at delta = 0) on the camera, overrides ref_angle and
        ref_amp, e.g. for a spherical reference.

    Returns
    -------
    O : array
        Object field in the hologram plane.
    """

    phase, modulation, bias = psi_n_step(frames, shifts, return_bias=True)
    C = 0.5*modulation*bias*np.exp(1j*phase)

    if reference is None:
        if tuple(ref_angle) == (0.0, 0.0):
            reference = ref_amp
        else:
            reference = reference_wave(x, y, wavelength, ref_angle, ref_amp, 0.0, n)

    return C*reference/np.abs(reference)**2


def reconstruct_gabor(H, x, y, z, wavelength, n_iter=0, support=None, ref_amp=1.0, n=1.0,
                      propagator="asm", pad_factor=2, return_errors=False):
    """Reconstruct an in-line (Gabor) hologram recorded with plane wave
    illumination.

    The object, a transmission t = 1 + o at the distance z in front of the
    camera, is illuminated by a plane wave of amplitude A. The camera records
    H = ``A**2*|1 + O|**2`` with O the scattered wave on the camera. Propagating
    H/A**2 = 1 + O + conj(O) + ``|O|**2`` back by -z gives 1 + o (in focus) plus
    the twin image, conj(O) propagated by -z, which is o* defocused by 2z,
    and the small ``|O|**2`` term.

    With n_iter > 0, the twin image is suppressed iteratively (Gerchberg-Saxton
    type, cf. Latychevskaia and Fink, PRL 98, 233901 (2007)): in the object
    plane the transmission is forced to ``|t|`` <= 1 (absorption, no gain) and to
    t = 1 outside the support; in the camera plane the amplitude is replaced
    by the measured sqrt(H)/A.

    Parameters
    ----------
    H : array
        In-line hologram.
    x, y : arrays
        Camera coordinates.
    z : number
        Object-camera distance (> 0).
    wavelength : number
    n_iter : int, optional
        Number of twin image suppression iterations (0: plain back
        propagation).
    support : bool array, optional
        Pixels that may contain the object; outside, t = 1. Without support
        only the absorption constraint is used.
    ref_amp : number, optional
        Illumination amplitude A.
    n, propagator, pad_factor
        As for refocus().
    return_errors : bool, optional
        If True, also return the rms amplitude mismatch in the camera plane
        for each iteration.

    Returns
    -------
    t : array
        Object transmission in the object plane (not in the camera plane, as
        the other reconstructions).
    errors : array
        Only if return_errors is True.
    """

    H = np.asarray(H, dtype=float)
    amplitude = np.sqrt(np.maximum(H, 0.0))/ref_amp
    prop = lambda U, dz: refocus(U, x, y, dz, wavelength, n, propagator, pad_factor)

    t = prop(H/ref_amp**2 + 0j, -z)
    errors = []
    for _ in range(n_iter):
        A_t = np.abs(t)
        t = np.where(A_t > 1, t/np.where(A_t > 0, A_t, 1.0), t)
        if support is not None:
            t = np.where(support, t, 1.0)
        U = prop(t, z)
        errors.append(np.sqrt(np.mean((np.abs(U) - amplitude)**2)))
        U = amplitude*np.exp(1j*np.angle(U))
        t = prop(U, -z)

    if return_errors:
        return t, np.array(errors)

    return t


def _angular_spectrum(field, x, y, z, wavelength, n=1.0, band_limit=True):
    """rayleigh_sommerfeld_I_TF() without evanescent waves. Backward
    propagation (z < 0) would amplify them exponentially; a camera does not
    record them anyway. The band limit is Matsushima's, as in
    rayleigh_sommerfeld_I_TF().
    """

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)
    k = wavenumber(wavelength, n)
    propagating = K_x**2 + K_y**2 < np.real(k)**2
    if band_limit:
        x_lin, y_lin = _linear_coords(x, y)
        wl_medium = wavelength/np.real(n)
        du, dv = 1/(len(x_lin)*get_dx(x_lin)), 1/(len(y_lin)*get_dx(y_lin))
        u_limit = 1/(wl_medium*np.sqrt((2*du*z)**2 + 1))
        v_limit = 1/(wl_medium*np.sqrt((2*dv*z)**2 + 1))
        propagating &= (np.abs(K_x/(2*np.pi)) <= u_limit) & (np.abs(K_y/(2*np.pi)) <= v_limit)

    TF = np.zeros(K_x.shape, dtype=complex)
    TF[propagating] = np.exp(1j*k_z(k, K_x[propagating], K_y[propagating])*z)

    return inv_FT(FT(field)*TF), x, y


def refocus(field, x, y, z, wavelength, n=1.0, propagator="asm", pad_factor=2, **kwargs):
    """Refocus a reconstructed field numerically.

    Parameters
    ----------
    field : array
        Complex field, e.g. the output of reconstruct_off_axis().
    x, y : arrays
        Coordinates (centred grid, see pyoptics.propagators).
    z : number
        Propagation distance. Positive z propagates downstream (away from
        the object); an object at distance d in front of the camera is in
        focus for z = -d.
    wavelength : number
    n : number, optional
    propagator : "asm", "fresnel" or callable, optional
        "asm": band-limited angular spectrum (rayleigh_sommerfeld_I_TF with
        band_limit=True), exact for all angles; evanescent waves are
        removed, because backward propagation would amplify them. "fresnel": paraxial transfer
        function (fresnel_TF). A callable is used as a pyoptics propagator.
    pad_factor : number, optional
        Zero padding factor (1 for none) against wrap-around.
    kwargs
        Passed on to the propagator.

    Returns
    -------
    field_refocused : array
    """

    if propagator == "asm":
        propagator = _angular_spectrum
        kwargs.setdefault("band_limit", True)
    elif propagator == "fresnel":
        propagator = fresnel_TF
    elif not callable(propagator):
        raise ValueError("propagator must be 'asm', 'fresnel' or a callable")

    if pad_factor is None or pad_factor <= 1:
        return propagator(field, x, y, z, wavelength, n=n, **kwargs)[0]

    return propagate_padded(propagator, field, x, y, z, wavelength, n=n,
                            pad_factor=pad_factor, **kwargs)[0]


focus_metrics = ("variance", "tamura", "gradient", "laplacian")


def focus_metric(field, metric="tamura", mask=None):
    """Sharpness metric of the amplitude of a field.

    Parameters
    ----------
    field : array
        Complex field (only its amplitude ``|field|`` enters).
    metric : str, optional
        - "variance": variance of the normalized amplitude, ``var(|U|)/mean(|U|)**2``
        - "tamura": Tamura coefficient ``sqrt(std(|U|)/mean(|U|))``
        - "gradient": mean squared gradient of ``|U|``, normalized by ``mean(|U|)**2``
        - "laplacian": mean squared Laplacian of ``|U|``, normalized likewise
    mask : bool array, optional
        Evaluate the metric only where mask is True.

    Returns
    -------
    value : float

    Notes
    -----
    For amplitude objects all metrics are largest in focus. Pure phase
    objects have a uniform amplitude in focus, the metrics are smallest
    there (they rise on both sides, with phase contrast of opposite sign).
    """

    A = np.abs(field)

    if metric in ("gradient", "laplacian"):
        if metric == "gradient":
            gy, gx = np.gradient(A)
            values = gx**2 + gy**2
        else:
            values = (np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1)
                      + np.roll(A, -1, 1) - 4*A)**2
        if mask is not None:
            values, A = values[mask], A[mask]
        return np.mean(values)/np.mean(A)**2

    if mask is not None:
        A = A[mask]
    if metric == "variance":
        return np.var(A)/np.mean(A)**2
    if metric == "tamura":
        return np.sqrt(np.std(A)/np.mean(A))

    raise ValueError("metric must be one of {}".format(focus_metrics))


def autofocus(field, x, y, z_range, wavelength, metric="tamura", object_type="amplitude",
              n=1.0, propagator="asm", pad_factor=2, mask=None, refine=True):
    """Find the best focus of a reconstructed field.

    The field is refocused to all distances in z_range and the focus metric
    is evaluated in each plane.

    Parameters
    ----------
    field : array
        Field in the hologram plane.
    x, y : arrays
    z_range : sequence of numbers
        Candidate refocus distances (e.g. np.linspace(-d_max, -d_min, 41)).
    wavelength : number
    metric : str or callable, optional
        One of focus_metrics, or a function of the field returning a number.
    object_type : "amplitude" or "phase", optional
        Amplitude objects are in focus at the maximum of the metric, phase
        objects at its minimum.
    n, propagator, pad_factor
        As for refocus().
    mask : bool array, optional
        Region in which the metric is evaluated.
    refine : bool, optional
        If True, refine the best distance by a parabola through the best
        sample and its neighbours (for equidistant z_range).

    Returns
    -------
    z_best : float
        Best refocus distance.
    values : array
        Metric for each entry of z_range.
    """

    if object_type not in ("amplitude", "phase"):
        raise ValueError("object_type must be 'amplitude' or 'phase'")

    if callable(metric):
        evaluate = metric
    else:
        evaluate = lambda U: focus_metric(U, metric, mask)

    z_range = np.asarray(z_range, dtype=float)
    values = np.array([evaluate(refocus(field, x, y, z, wavelength, n, propagator, pad_factor))
                       for z in z_range])

    score = values if object_type == "amplitude" else -values
    i = int(np.argmax(score))
    z_best = z_range[i]

    if refine and 0 < i < len(z_range) - 1:
        a, b, c = score[i - 1:i + 2]
        denom = a - 2*b + c
        if denom < 0:  # a proper maximum
            dz = 0.5*(z_range[i + 1] - z_range[i - 1])
            z_best = z_range[i] + 0.5*(a - c)/denom*dz

    return z_best, values


def remove_tilt(phase, x, y, mask=None, piston=True):
    """Subtract a least squares plane (tilt and optionally piston) from a
    phase map.

    Parameters
    ----------
    phase : array
        Unwrapped phase.
    x, y : arrays
    mask : bool array, optional
        Pixels used for the fit (e.g. the background around a sample).
    piston : bool, optional
        If True, also subtract the mean (over mask).

    Returns
    -------
    phase_corrected : array
    coeffs : array
        (piston, a_x, a_y) of the fitted plane piston + a_x*x + a_y*y.
    """

    X, Y = ensure_meshgrid(x, y)
    if mask is None:
        mask = np.isfinite(phase)
    M = np.column_stack([np.ones(np.count_nonzero(mask)), X[mask], Y[mask]])
    coeffs, *_ = np.linalg.lstsq(M, phase[mask], rcond=None)
    if not piston:
        coeffs[0] = 0.0

    return phase - (coeffs[0] + coeffs[1]*X + coeffs[2]*Y), coeffs


def quantitative_phase(field, x=None, y=None, reference_field=None, unwrap="quality", mask=None,
                       tilt_mask=None, weights=None):
    """Quantitative phase map of a reconstructed (and refocused) field.

    Parameters
    ----------
    field : array
        Complex object field.
    x, y : arrays, optional
        Coordinates, needed for tilt removal (tilt_mask).
    reference_field : array, optional
        Field reconstructed without sample (or a fitted background). The
        phase of field*conj(reference_field) is free of the aberrations and
        the residual carrier tilt common to both ("double exposure" or
        background compensation).
    unwrap : "quality", "ls", None or callable, optional
        Phase unwrapping: "quality" (default) uses
        pyoptics.unwrap.unwrap_quality_guided (with the amplitude as quality
        if no weights are given), which stays congruent with the wrapped
        phase; "ls" uses pyoptics.unwrap.unwrap_ls (weighted with `weights`
        if given), which is robust against residues but smooths steep
        phases. None returns the wrapped phase, a callable is called with
        the wrapped phase.
    mask : bool array, optional
        Valid pixels, passed to the unwrapper.
    tilt_mask : bool array, optional
        If given, remove_tilt() is applied with this mask (e.g. the sample
        free background), which also fixes the piston there.
    weights : array, optional
        Weights or quality map for the unwrapper.

    Returns
    -------
    phase : array
        Phase in radians (unwrapped unless unwrap is None).
    """

    if reference_field is not None:
        field = field*np.conj(reference_field)

    phase = np.angle(field)

    if unwrap is not None:
        if callable(unwrap):
            phase = unwrap(phase)
        else:
            from . import unwrap as unwrap_module
            if unwrap == "ls":
                phase = unwrap_module.unwrap_ls(phase, weights=weights, mask=mask)
            elif unwrap == "quality":
                quality = np.abs(field) if weights is None else weights
                phase = unwrap_module.unwrap_quality_guided(phase, quality=quality, mask=mask)
            else:
                raise ValueError("unwrap must be 'ls', 'quality', None or a callable")

    if tilt_mask is not None:
        if x is None or y is None:
            raise ValueError("tilt removal needs x and y")
        phase = remove_tilt(phase, x, y, mask=tilt_mask)[0]

    return phase


def optical_path_difference(phase, wavelength):
    """Optical path difference phase/k_0.

    The thickness of a sample with index n_s in a medium n_m is
    OPD/(n_s - n_m).

    Parameters
    ----------
    phase : number or array
        Phase in radians.
    wavelength : number
        Vacuum wavelength.

    Returns
    -------
    opd : number or array
        In the unit of the wavelength.
    """

    return phase*wavelength/(2*np.pi)
