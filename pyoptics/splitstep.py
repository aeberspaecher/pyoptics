#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Split-step solvers for nonlinear propagation: pulses in fibres and
crystals, coupled envelopes of several frequencies, and beams in Kerr media
(notebooks 121 (SPM and solitons in fibres),
145 (second harmonic generation and phase matching),
146 (parametric amplification and OPOs) and
147 (Kerr effect and self-focusing)).

Units are free as long as they are consistent: the fibre notebooks use ps,
km and W (gamma in 1/(W km), beta_2 in ps²/km), the crystal notebooks fs
or ps and mm or m. Fields follow the pyoptics convention
exp(i(kz - omega t)), as in `pyoptics.fibrecomm` and `pyoptics.lasers`.

Spectra
-------
The spectrum of an envelope A(t) is A~(Omega) = int A(t) exp(i Omega t) dt,
with Omega = omega - omega_0 the offset from the carrier: positive Omega is
a higher optical frequency, a shorter wavelength. numpy's ifft has the
kernel exp(+i...), so A~ is ifft(A) N dt, ordered like
`angular_frequency()`; `spectrum()` returns it.

Pulses
------
The generalized nonlinear Schrödinger equation (GNLSE) for the envelope in
the frame moving with the group velocity (Agrawal's form, the pyoptics
sign convention) is, in the frequency domain, ::

    dA~/dz = (i sum_k beta_k Omega**k/k! - alpha/2) A~
             + i gamma (1 + Omega/omega_0) FT[A (R * abs(A)**2)],

with the propagation constants beta_k = d^k beta/d omega^k (k >= 2), the
power loss alpha, the nonlinear coefficient gamma = n2 omega_0/(c A_eff),
the shock (self-steepening) term Omega/omega_0, and the nonlinear response
R(t) = (1 - f_R) delta(t) + f_R h_R(t): instantaneous Kerr effect plus the
delayed Raman response h_R of silica (`raman_response()`). Without the
last two terms it is the nonlinear Schrödinger equation (NLSE) ::

    dA/dz = -alpha/2 A - i beta_2/2 d²A/dt² + beta_3/6 d³A/dt³ + i gamma abs(A)**2 A.

`nlse()` solves the NLSE with the symmetric split-step Fourier method:
half a dispersion step in the frequency domain, the nonlinear phase
gamma abs(A)**2 dz in the time domain (exact, the power does not change
in it), the other half of the dispersion step. Its error is of second
order in dz. `gnlse()` solves the full equation with the fourth order
Runge–Kutta method in the interaction picture (RK4IP, Hult 2007),
optionally with an adaptive step from the local error (step doubling).

Coupled envelopes
-----------------
`rk4ip()` is the RK4IP step for any number of envelopes, each with its
own linear operator in the frequency domain (dispersion, group velocity
mismatch, a phase mismatch, loss) and a common nonlinear term given as a
function of the envelopes in the time domain. `gnlse()` uses it for one
field; `pyoptics.nonlinear` uses it for second harmonic generation and
parametric amplification of pulses (two and three fields).

Beams
-----
`beam_propagate()` solves the paraxial wave equation with a Kerr term
for a beam in one or two transverse dimensions, ::

    dE/dz = i/(2k) laplace_perp E + i k0 n2 abs(E)**2 E (- beta_mpa/2 abs(E)**(2 m - 2) E),

with abs(E)**2 the intensity and k = n0 k0, again by the symmetric split
step; the step is chosen so that the nonlinear phase per step stays
small, which resolves self-focusing up to the point where the grid can no
longer hold the collapsing beam.

Soliton and SPM formulas
------------------------
dispersion_length() L_D = T0**2/abs(beta_2), nonlinear_length()
L_NL = 1/(gamma P0), soliton_order() N = sqrt(L_D/L_NL), soliton() the
fundamental (or N-th order) soliton N sqrt(P0) sech(t/T0) with
P0 = abs(beta_2)/(gamma T0**2), soliton_period() z0 = pi L_D/2,
fission_length() L_D/N, spm_phase() the peak nonlinear phase gamma P0
L_eff with the effective length (1 - exp(-alpha L))/alpha,
spm_broadening() the rms spectral broadening of a Gaussian pulse,
sqrt(1 + 4 phi_max**2/(3 sqrt(3))), and ssfs_rate() Gordon's soliton
self-frequency shift.

References
----------
G. P. Agrawal, "Nonlinear Fiber Optics", 6th ed., Academic Press (2019),
chapters 2 to 5 and 12.

J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for
simulating supercontinuum generation in optical fibers", J. Lightwave
Technol. 25, 3770-3775 (2007).

O. V. Sinkin, R. Holzlöhner, J. Zweck and C. R. Menyuk, "Optimization of
the split-step Fourier method in modeling optical-fiber communications
systems", J. Lightwave Technol. 21, 61-68 (2003).

K. J. Blow and D. Wood, "Theoretical description of transient stimulated
Raman scattering in optical fibers", IEEE J. Quantum Electron. 25,
2665-2673 (1989).

J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in
photonic crystal fiber", Rev. Mod. Phys. 78, 1135-1184 (2006).

J. P. Gordon, "Theory of the soliton self-frequency shift", Opt. Lett. 11,
662-664 (1986).
"""

import numpy as np


__all__ = ["angular_frequency", "spectrum", "dispersion_operator", "raman_response",
           "raman_spectrum", "nlse", "rk4ip", "gnlse", "beam_propagate", "dispersion_length",
           "nonlinear_length", "soliton_order", "soliton", "soliton_period", "fission_length",
           "effective_length", "spm_phase", "spm_broadening", "ssfs_rate"]


# --- grids and spectra ---------------------------------------------------------

def angular_frequency(t):
    """Angular frequency offsets Omega (in the inverse units of t) of the
    FFT grid that belongs to the equally spaced times t, in FFT order
    (2 pi fftfreq(N, dt))."""
    t = np.asarray(t, dtype=float)
    return 2*np.pi*np.fft.fftfreq(len(t), t[1] - t[0])


def spectrum(t, A, shift=True):
    """Spectrum A~(Omega) = int A(t) exp(i Omega t) dt of an envelope.

    Parameters
    ----------
    t : array
        Equally spaced times.
    A : array
        Envelope, time along the last axis.
    shift : bool, optional
        Return Omega ascending (fftshift), else in FFT order.

    Returns
    -------
    Omega : array
        Angular frequency offsets from the carrier.
    S : complex array
        Spectrum; abs(S)**2 is the energy spectral density (energy per
        unit angular frequency is abs(S)**2/(2 pi)).
    """

    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    Om = angular_frequency(t)
    S = np.fft.ifft(A, axis=-1)*len(t)*dt
    if shift:
        return np.fft.fftshift(Om), np.fft.fftshift(S, axes=-1)
    return Om, S


def _to_freq(A):
    return np.fft.ifft(A, axis=-1)


def _to_time(S):
    return np.fft.fft(S, axis=-1)


def dispersion_operator(Omega, betas=(), alpha=0.0, beta=None):
    """Linear operator D(Omega) of the propagation, dA~/dz = D A~.

    Parameters
    ----------
    Omega : array
        Angular frequency offsets.
    betas : sequence, optional
        (beta_2, beta_3, ...): Taylor coefficients of the propagation
        constant from the second order on.
    alpha : number, optional
        Power loss coefficient (1/length, not dB).
    beta : callable, optional
        Full dispersion beta(Omega) - beta_0 - beta_1 Omega instead of (or
        added to) the Taylor series.

    Returns
    -------
    D : complex array
        i sum_k beta_k Omega**k/k! - alpha/2.
    """

    Om = np.asarray(Omega, dtype=float)
    ph = np.zeros_like(Om)
    fact = 1.0
    for k, b in enumerate(betas, start=2):
        fact *= k
        ph = ph + b*Om**k/fact
    if beta is not None:
        ph = ph + beta(Om)
    return 1j*ph - alpha/2


# --- Raman response --------------------------------------------------------------

def raman_response(t, tau1=0.0122, tau2=0.032):
    """Delayed Raman response h_R(t) of silica (Blow and Wood): a damped
    oscillation (tau1**2 + tau2**2)/(tau1 tau2**2) exp(-t/tau2) sin(t/tau1)
    for t > 0, zero before, normalized to unit area.

    Parameters
    ----------
    t : array
    tau1, tau2 : numbers, optional
        Defaults are 12.2 fs and 32 fs in ps.

    Returns
    -------
    h : array
        In the inverse units of t.
    """

    t = np.asarray(t, dtype=float)
    tp = np.where(t > 0, t, 0.0)
    return np.where(t > 0, (tau1**2 + tau2**2)/(tau1*tau2**2)*np.exp(-tp/tau2)*np.sin(tp/tau1), 0.0)


def raman_spectrum(Omega, tau1=0.0122, tau2=0.032):
    """Fourier transform int h_R(t) exp(i Omega t) dt of `raman_response()`
    (1 at Omega = 0; its imaginary part is the Raman gain, peaking at
    13.2 THz, 2 pi 13.2 rad/ps, below the pump for silica).

    Parameters
    ----------
    Omega : array
    tau1, tau2 : numbers, optional

    Returns
    -------
    H : complex array
    """

    Om = np.asarray(Omega, dtype=float)
    a = 1/tau2 - 1j*Om
    return (tau1**2 + tau2**2)/(tau1*tau2**2)*(1/tau1)/(a**2 + 1/tau1**2)


# --- the NLSE: symmetric split-step Fourier ---------------------------------------

def _save_points(length, n_save):
    return np.linspace(0.0, length, int(n_save))


def nlse(t, A0, length, betas=(), gamma=0.0, alpha=0.0, steps=1000, n_save=None):
    """Nonlinear Schrödinger equation by the symmetric split-step Fourier
    method (time window periodic).

    Parameters
    ----------
    t : array
        Equally spaced times (frame moving with the group velocity).
    A0 : complex array
        Input envelope, abs(A)**2 is the power.
    length : number
    betas : sequence, optional
        (beta_2, beta_3, ...).
    gamma : number, optional
        Nonlinear coefficient (1/(power length)).
    alpha : number, optional
        Power loss coefficient.
    steps : int, optional
        Number of equal steps.
    n_save : int, optional
        Number of equally spaced z at which to keep the field (including
        0 and `length`); rounded so that each falls on a step.

    Returns
    -------
    A : complex array
        Output envelope.
    z, history : arrays
        Only with n_save: the kept positions and fields (n_save, len(t)).
    """

    Om = angular_frequency(t)
    D = dispersion_operator(Om, betas, alpha)
    dz = length/steps
    half = np.exp(D*dz/2)
    A = np.asarray(A0, dtype=complex).copy()
    keep = None
    if n_save is not None:
        idx = np.unique(np.round(np.linspace(0, steps, int(n_save))).astype(int))
        keep = set(idx.tolist())
        zs, hist = [0.0], [A.copy()]
    for k in range(1, steps + 1):
        A = _to_time(_to_freq(A)*half)
        A = A*np.exp(1j*gamma*np.abs(A)**2*dz)
        A = _to_time(_to_freq(A)*half)
        if keep is not None and k in keep:
            zs.append(k*dz)
            hist.append(A.copy())
    if keep is None:
        return A
    return A, np.array(zs), np.array(hist)


# --- RK4 in the interaction picture -----------------------------------------------

def _rk4ip_step(S, D, N, z, h):
    """One RK4IP step of the spectra S (list) with the linear operators D
    (list) and the nonlinear function N(z, fields) -> list of time domain
    terms dA/dz."""
    E = [np.exp(d*h/2) for d in D]

    def nl(zz, spectra):
        fields = [_to_time(s) for s in spectra]
        return [_to_freq(x) for x in N(zz, fields)]

    SI = [e*s for e, s in zip(E, S)]
    k1 = [e*k for e, k in zip(E, nl(z, S))]
    k2 = nl(z + h/2, [si + h/2*k for si, k in zip(SI, k1)])
    k3 = nl(z + h/2, [si + h/2*k for si, k in zip(SI, k2)])
    k4 = nl(z + h, [e*(si + h*k) for e, si, k in zip(E, SI, k3)])
    return [e*(si + h/6*(a + 2*b + 2*c)) + h/6*d
            for e, si, a, b, c, d in zip(E, SI, k1, k2, k3, k4)]


def rk4ip(t, fields, linear, nonlinear, length, steps=1000, tol=None, n_save=None, max_steps=200000):
    """Coupled envelopes by the fourth order Runge–Kutta method in the
    interaction picture (RK4IP).

    Each envelope A_j obeys dA~_j/dz = D_j(Omega) A~_j + FT[N_j(z, A)],
    with the linear operator D_j applied exactly and the nonlinear terms
    N_j (in the time domain) by Runge–Kutta.

    Parameters
    ----------
    t : array
        Equally spaced times.
    fields : list of complex arrays
        Input envelopes (one array for a single field is fine too).
    linear : list of complex arrays
        D_j(Omega) in FFT order (`angular_frequency()`), e.g. from
        `dispersion_operator()`.
    nonlinear : callable
        nonlinear(z, fields) -> list of arrays dA_j/dz (time domain).
    length : number
    steps : int, optional
        Number of equal steps; with `tol` the initial number.
    tol : number, optional
        Relative local error per step for adaptive steps (step doubling,
        Sinkin et al.); the step grows or shrinks to keep the error between
        tol/2 and tol.
    n_save : int, optional
        Number of equally spaced z at which to keep the fields.
    max_steps : int, optional
        Safety limit of the adaptive stepper.

    Returns
    -------
    fields : list of complex arrays
        Output envelopes (a single array if a single array was given).
    z, history : array, list of arrays
        Only with n_save: positions and, per field, the kept fields
        (n_save, len(t)).
    """

    single = not isinstance(fields, (list, tuple))
    if single:
        fields = [fields]
        linear = [linear]
        f_nl = nonlinear
        nonlinear = lambda z, F: [f_nl(z, F[0])]
    S = [_to_freq(np.asarray(f, dtype=complex)) for f in fields]
    D = [np.asarray(d, dtype=complex) for d in linear]
    z_targets = _save_points(length, n_save) if n_save is not None else np.array([0.0, length])
    hist = [[f.copy() for f in fields]] if n_save is not None else None
    h = length/steps
    z = 0.0
    count = 0
    for z_next in z_targets[1:]:
        while z < z_next*(1 - 1E-12):
            h_try = min(h, z_next - z)
            if tol is None:
                S = _rk4ip_step(S, D, nonlinear, z, h_try)
                z += h_try
            else:
                coarse = _rk4ip_step(S, D, nonlinear, z, h_try)
                half = _rk4ip_step(S, D, nonlinear, z, h_try/2)
                fine = _rk4ip_step(half, D, nonlinear, z + h_try/2, h_try/2)
                num = np.sqrt(sum(np.sum(np.abs(f - c)**2) for f, c in zip(fine, coarse)))
                den = np.sqrt(sum(np.sum(np.abs(f)**2) for f in fine)) + 1E-300
                err = num/den
                if err > 2*tol:
                    h = h_try/2
                    continue
                # Richardson extrapolation of the fifth order
                S = [(16*f - c)/15 for f, c in zip(fine, coarse)]
                z += h_try
                if err > tol:
                    h = h_try/2**0.2
                elif err < tol/2:
                    h = h_try*2**0.2
                else:
                    h = max(h, h_try)
            count += 1
            if count > max_steps:
                raise RuntimeError("rk4ip: more than max_steps steps")
        if hist is not None:
            hist.append([_to_time(s) for s in S])
    out = [_to_time(s) for s in S]
    if single:
        out = out[0]
    if hist is None:
        return out
    hist_f = [np.array([hh[j] for hh in hist]) for j in range(len(S))]
    if single:
        hist_f = hist_f[0]
    return out, z_targets, hist_f


def gnlse(t, A0, length, betas=(), gamma=0.0, alpha=0.0, omega0=None, raman=None, steps=1000,
          tol=None, n_save=None, beta=None):
    """Generalized nonlinear Schrödinger equation (dispersion of any order,
    loss, Kerr effect, self-steepening and Raman scattering) by RK4IP.

    Parameters
    ----------
    t : array
        Equally spaced times.
    A0 : complex array
        Input envelope (abs(A)**2 the power).
    length : number
    betas : sequence, optional
        (beta_2, beta_3, ...).
    gamma : number, optional
        Nonlinear coefficient.
    alpha : number, optional
        Power loss coefficient.
    omega0 : number, optional
        Carrier angular frequency (inverse units of t): adds
        self-steepening, (1 + Omega/omega0) in front of the nonlinear term.
    raman : number or tuple, optional
        f_R (0.18 for silica) or (f_R, tau1, tau2), in the units of t;
        the defaults of `raman_response()` are in ps.
    steps : int, optional
        Number of steps (initial number with `tol`).
    tol : number, optional
        Relative local error for adaptive steps.
    n_save : int, optional
        Number of equally spaced z to keep.
    beta : callable, optional
        Full dispersion beta(Omega) - beta_0 - beta_1 Omega.

    Returns
    -------
    A : complex array
    z, history : arrays
        Only with n_save.
    """

    t = np.asarray(t, dtype=float)
    Om = angular_frequency(t)
    D = dispersion_operator(Om, betas, alpha, beta)
    shock = 1.0 if omega0 is None else 1 + Om/omega0
    if raman is None:
        f_R, H = 0.0, None
    else:
        if np.ndim(raman) == 0:
            f_R, pars = float(raman), ()
        else:
            f_R, pars = raman[0], tuple(raman[1:])
        H = raman_spectrum(Om, *pars)

    def nonlinear(z, A):
        P = np.abs(A)**2
        if H is not None:
            P = (1 - f_R)*P + f_R*np.real(_to_time(H*_to_freq(P)))
        term = _to_freq(A*P)
        return _to_time(1j*gamma*shock*term)

    return rk4ip(t, np.asarray(A0, dtype=complex), D, nonlinear, length, steps=steps, tol=tol,
                 n_save=n_save)


# --- beams in Kerr media ------------------------------------------------------------

def beam_propagate(coords, E0, length, wavelength, n0=1.0, n2=0.0, max_phase=0.05,
                   steps=None, n_save=None, mpa=None, stop_intensity=None):
    """Paraxial beam in a Kerr medium by the symmetric split-step method.

    Parameters
    ----------
    coords : array or tuple of two arrays
        x, or (x, y): equally spaced transverse coordinates (field periodic
        over the window, which should be wide enough).
    E0 : complex array
        Input field with abs(E0)**2 the intensity (shape (len(x),) or
        (len(y), len(x))).
    length : number
    wavelength : number
        Vacuum wavelength (units of x).
    n0, n2 : numbers, optional
        Linear index and Kerr coefficient (n = n0 + n2 I, in the inverse
        units of the intensity).
    max_phase : number, optional
        Largest nonlinear phase k0 n2 max(I) dz per step (rad); sets the
        step size adaptively.
    steps : int, optional
        Fixed number of equal steps instead.
    n_save : int, optional
        Number of equally spaced z at which to keep the field.
    mpa : tuple, optional
        (beta_m, m): multiphoton absorption dI/dz = -beta_m I**m, which
        stops a collapse.
    stop_intensity : number, optional
        Stop when the peak intensity exceeds this (a collapse).

    Returns
    -------
    E : complex array
        Field at the end (or where it stopped).
    z : array
        Positions of all steps taken (with n_save: the kept positions).
    peak : array
        Peak intensity at the positions in z.
    history : array
        Only with n_save: the kept fields.
    """

    k0 = 2*np.pi/wavelength
    k = n0*k0
    two_d = isinstance(coords, (tuple, list))
    if two_d:
        x, y = coords
        kx = 2*np.pi*np.fft.fftfreq(len(x), x[1] - x[0])
        ky = 2*np.pi*np.fft.fftfreq(len(y), y[1] - y[0])
        K2 = kx[None, :]**2 + ky[:, None]**2
        fwd, bwd = np.fft.fft2, np.fft.ifft2
    else:
        x = coords
        kx = 2*np.pi*np.fft.fftfreq(len(x), x[1] - x[0])
        K2 = kx**2
        fwd, bwd = np.fft.fft, np.fft.ifft
    E = np.asarray(E0, dtype=complex).copy()

    def linear(E, dz):
        return bwd(fwd(E)*np.exp(-1j*K2*dz/(2*k)))

    def nonlinear(E, dz):
        I = np.abs(E)**2
        ph = k0*n2*I*dz
        if mpa is None:
            return E*np.exp(1j*ph)
        b, m = mpa
        # exact intensity loss over dz for dI/dz = -b I**m, phase with the start intensity
        I_new = I/(1 + (m - 1)*b*np.maximum(I, 1E-300)**(m - 1)*dz)**(1/(m - 1))
        return E*np.sqrt(I_new/np.maximum(I, 1E-300))*np.exp(1j*ph)

    z = 0.0
    zs, peaks = [0.0], [np.max(np.abs(E)**2)]
    save_at = _save_points(length, n_save) if n_save is not None else None
    hist = [E.copy()] if n_save is not None else None
    i_save = 1
    dz_fixed = None if steps is None else length/steps
    while z < length*(1 - 1E-12):
        I_max = np.max(np.abs(E)**2)
        if dz_fixed is not None:
            dz = dz_fixed
        else:
            dz = max_phase/(k0*abs(n2)*I_max) if n2 != 0 and I_max > 0 else length/100
            dz = min(dz, length/100)
        if save_at is not None and i_save < len(save_at):
            dz = min(dz, save_at[i_save] - z)
        dz = min(dz, length - z)
        E = linear(E, dz/2)
        E = nonlinear(E, dz)
        E = linear(E, dz/2)
        z += dz
        if save_at is not None and i_save < len(save_at) and z >= save_at[i_save]*(1 - 1E-12):
            hist.append(E.copy())
            i_save += 1
        zs.append(z)
        peaks.append(np.max(np.abs(E)**2))
        if stop_intensity is not None and peaks[-1] > stop_intensity:
            break
        if len(zs) > 10**6:
            raise RuntimeError("beam_propagate: too many steps")
    zs, peaks = np.array(zs), np.array(peaks)
    if hist is None:
        return E, zs, peaks
    return E, save_at[:len(hist)], peaks, np.array(hist)


# --- formulas --------------------------------------------------------------------------

def dispersion_length(T0, beta2):
    """Dispersion length T0**2/abs(beta_2) (T0 the 1/e half width of the
    power of a Gaussian, the sech parameter of a soliton)."""
    return T0**2/np.abs(beta2)


def nonlinear_length(P0, gamma):
    """Nonlinear length 1/(gamma P0): the length over which the peak
    acquires one radian of nonlinear phase."""
    return 1/(gamma*P0)


def soliton_order(T0, P0, gamma, beta2):
    """Soliton order N = sqrt(L_D/L_NL) = sqrt(gamma P0 T0**2/abs(beta_2))."""
    return np.sqrt(gamma*P0*T0**2/np.abs(beta2))


def soliton(t, T0, beta2, gamma, N=1.0, t0=0.0):
    """N-th order soliton N sqrt(P0) sech((t - t0)/T0) with the fundamental
    soliton power P0 = abs(beta_2)/(gamma T0**2) (beta_2 < 0); FWHM =
    1.763 T0.

    Parameters
    ----------
    t : array
    T0 : number
    beta2 : number
    gamma : number
    N : number, optional
    t0 : number, optional
        Centre.

    Returns
    -------
    A : complex array
    """

    P0 = np.abs(beta2)/(gamma*T0**2)
    return (N*np.sqrt(P0)/np.cosh((np.asarray(t) - t0)/T0)).astype(complex)


def soliton_period(T0, beta2):
    """Soliton period z0 = pi L_D/2, after which a higher order soliton
    returns to its input shape (and a fundamental soliton has acquired a
    phase of pi/4)."""
    return np.pi/2*dispersion_length(T0, beta2)


def fission_length(T0, beta2, N):
    """Length L_D/N at which a higher order soliton (N >> 1) is compressed
    most and breaks up into fundamental solitons when perturbed by
    third-order dispersion or Raman scattering (Dudley, Genty and Coen)."""
    return dispersion_length(T0, beta2)/N


def effective_length(length, alpha=0.0):
    """Effective length (1 - exp(-alpha L))/alpha of a lossy fibre (alpha
    the power loss coefficient); L for alpha = 0."""
    if alpha == 0:
        return np.asarray(length, dtype=float)
    return (1 - np.exp(-alpha*np.asarray(length)))/alpha


def spm_phase(P0, gamma, length, alpha=0.0):
    """Peak nonlinear phase phi_max = gamma P0 L_eff of self-phase
    modulation."""
    return gamma*P0*effective_length(length, alpha)


def spm_broadening(phi_max):
    """Rms spectral broadening factor sqrt(1 + 4 phi_max**2/(3 sqrt(3))) of
    an unchirped Gaussian pulse by self-phase modulation without dispersion
    (Agrawal, eq. 4.1.15)."""
    return np.sqrt(1 + 4/(3*np.sqrt(3))*np.asarray(phi_max)**2)


def ssfs_rate(T0, beta2, T_R=0.003):
    """Soliton self-frequency shift d Omega/dz = -8 T_R abs(beta_2)/(15 T0**4)
    of a fundamental soliton (Gordon; Agrawal eq. 5.5.21), in rad per unit
    time per unit length: negative, towards longer wavelengths.

    Parameters
    ----------
    T0 : number
        Soliton width parameter.
    beta2 : number
    T_R : number, optional
        Slope of the Raman gain, int t R(t) dt (about 3 fs for silica,
        default in ps).

    Returns
    -------
    rate : number
    """

    return -8*T_R*np.abs(beta2)/(15*T0**4)
