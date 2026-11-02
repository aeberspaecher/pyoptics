#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Resonances from ring-down signals (harmonic inversion).

After a short pulse has excited a resonator, a probe inside it records a
sum of damped oscillations ::

    f(t) = sum_k A_k exp(-gamma_k t) cos(omega_k t + phi_k).

resonances() finds omega_k, gamma_k and the quality factors
Q_k = omega_k/(2 gamma_k) with the matrix pencil method (Hua and Sarkar
1990): the samples f(t_j) = sum c_k z_k**j with z_k = exp((-gamma_k + i
omega_k) dtau) form a Hankel matrix of rank 2K; the z_k are the
eigenvalues of the shift between its leading right singular vectors. It
resolves resonances closer than the Fourier resolution 1/T of the record
and measures Q far beyond the record length, as long as the signal stays
above the noise (round-off for FDTD).

The signal is decimated before the analysis, so the FDTD time step does not
set the matrix size; the sampling must stay above twice the highest
frequency in the signal (with a pulse source: the upper band edge).

References
----------
Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating parameters of
exponentially damped/undamped sinusoids in noise", IEEE Trans. Acoust.
Speech Signal Process. 38, 814-824 (1990).

T. K. Sarkar and O. Pereira, "Using the matrix pencil method to estimate the
parameters of a sum of complex exponentials", IEEE Antennas Propag. Mag. 37,
48-55 (1995).

V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and
its applications", J. Chem. Phys. 107, 6756-6769 (1997), the related filter
diagonalization used by MEEP's Harminv.
"""

import numpy as np


class Resonances(object):
    """Result of resonances(): arrays sorted by frequency.

    frequency, wavelength, omega, decay (gamma), Q, amplitude, phase, and
    error (the relative rms residual of the fit, a single number).
    """

    def __init__(self, omega, decay, amplitude, phase, error):
        order = np.argsort(omega)
        self.omega = omega[order]
        self.decay = decay[order]
        self.amplitude = amplitude[order]
        self.phase = phase[order]
        self.error = error

    @property
    def frequency(self):
        return self.omega/(2*np.pi)

    @property
    def wavelength(self):
        return 2*np.pi/self.omega

    @property
    def Q(self):
        # lossless modes (decay ~ 0 up to round-off, either sign): inf
        with np.errstate(divide="ignore"):
            return np.where(self.decay > 0, self.omega/(2*np.maximum(self.decay, 1e-300)), np.inf)

    def __len__(self):
        return len(self.omega)

    def __repr__(self):
        rows = ["%-12s %-12s %-12s %-10s" % ("wavelength", "Q", "amplitude", "decay")]
        for w, q, a, g in zip(self.wavelength, self.Q, self.amplitude, self.decay):
            rows.append("%-12.6g %-12.5g %-12.4g %-10.4g" % (w, q, a, g))
        return "\n".join(rows)


def resonances(signal, dt, t0=0.0, band=None, max_poles=40, rtol=1e-8, samples=1200,
               min_amplitude=1e-3, min_q=0.0):
    """Frequencies, decay rates and Q of the damped oscillations in a
    real signal.

    Parameters
    ----------
    signal : array
        Samples at times t0 + j*dt, e.g. Probe.values after the source has
        died out (start the record after the pulse).
    dt : float
        Sample interval.
    t0 : float
        Time of the first sample (sets the phases and amplitudes).
    band : (wavelength_min, wavelength_max), optional
        Keep only resonances in this range of vacuum wavelengths.
    max_poles : int
        Upper limit of the model order (complex exponentials).
    rtol : float
        Singular values below rtol times the largest are noise.
    samples : int
        The signal is decimated to about this many samples.
    min_amplitude : float
        Drop resonances whose amplitude is below this fraction of the
        largest.
    min_q : float
        Drop poles with a lower Q, e.g. the strongly damped ones that
        describe what is left of the source pulse.

    Returns
    -------
    Resonances
    """

    x = np.asarray(signal, dtype=float)
    step = max(1, len(x)//samples)
    if band is not None:
        # keep the sampling above 4x the highest frequency
        step = min(step, max(1, int(band[0]/(4*dt))))
    y = x[::step]
    tau = dt*step
    N = len(y)
    L = N//3
    H = np.lib.stride_tricks.sliding_window_view(y, L + 1)
    _, s, Vh = np.linalg.svd(H, full_matrices=False)
    M = int(min(max_poles, np.sum(s > rtol*s[0])))
    M = max(M, 2)
    V = Vh[:M].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1])@V[1:])
    # amplitudes by least squares on the decimated signal
    j = np.arange(N)
    Z = z[None, :]**j[:, None]
    c, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    err = np.linalg.norm(Z@c - y)/np.linalg.norm(y)
    lz = np.log(z)/tau
    omega = lz.imag
    decay = -lz.real
    keep = omega > 0
    # a real cosine A cos(wt + phi) = A/2 e^{i phi} e^{iwt} + c.c.
    amp = 2*np.abs(c)*np.exp(decay*t0)
    phase = np.angle(c) - omega*t0
    omega, decay, amp, phase = omega[keep], decay[keep], amp[keep], phase[keep]
    if band is not None:
        wl = 2*np.pi/omega
        sel = (wl >= band[0]) & (wl <= band[1])
        omega, decay, amp, phase = omega[sel], decay[sel], amp[sel], phase[sel]
    sel = omega/(2*np.maximum(decay, 1e-300)) >= min_q
    omega, decay, amp, phase = omega[sel], decay[sel], amp[sel], phase[sel]
    if len(amp):
        sel = amp >= min_amplitude*np.max(amp)
        omega, decay, amp, phase = omega[sel], decay[sel], amp[sel], phase[sel]
    return Resonances(omega, decay, amp, np.angle(np.exp(1j*phase)), err)
