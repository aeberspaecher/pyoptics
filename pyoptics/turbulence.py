#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Atmospheric turbulence: phase screens, layers and the classical results
of Kolmogorov theory.

Turbulence mixes air of slightly different temperature, and with it of
slightly different refractive index. In the inertial range between the inner
scale l0 (millimetres) and the outer scale L0 (tens of metres) the index
fluctuations follow Kolmogorov's 2/3 law with the strength Cn**2. A wave
passing a layer of thickness dh picks up a random phase whose statistics only
depend on the Fried parameter r0 of the layer,
``r0 = (0.423*k**2*Cn**2*dh)**(-3/5)``, and the r0 of several layers combine as
``r0**(-5/3) = sum(r0_i**(-5/3))``. r0 is the diameter over which the RMS phase
error is about 1 rad; it grows with the wavelength as wavelength**(6/5).

Phase screens and their statistics:

- phase_psd(): the von Kármán phase power spectrum
  ``0.0229*r0**(-5/3)*(f**2 + 1/L0**2)**(-11/6)`` (f in cycles per length), with an
  optional Gaussian inner scale cutoff; Kolmogorov for L0 = inf.
- phase_screen(): random screens by filtering white noise in Fourier space
  (FFT method), with Lane's subharmonics for the low frequencies that the
  periodic grid misses. Without subharmonics the screens are periodic (as
  needed for split-step propagation on a periodic grid), but lack tilt.
- structure_function(): the measured phase structure function
  ``D(r) = <(phi(x + r) - phi(x))**2>``, and
  structure_function_theory(): ``6.88*(r/r0)**(5/3)`` (Kolmogorov) or the
  von Kármán form that saturates at large r.
- noll_variance(), noll_residual(): the variance of the Zernike coefficients
  of Kolmogorov turbulence (per radial order) and the residual variance after
  the first J modes are corrected (Noll 1976).

Imaging through turbulence:

- atmospheric_otf(): the long exposure transfer function
  ``exp(-D(wavelength*nu)/2)``, and the tilt-corrected short exposure OTF of
  Fried (1966).
- seeing_fwhm(): the width of the long exposure PSF, ``0.976*wavelength/r0``,
  reduced by a finite outer scale (Tokovinin 2002).
- lucky_probability(): Fried's (1978) probability of a short exposure with
  less than 1 rad**2 of phase variance (tilt removed).

Vertical profiles and layers:

- hufnagel_valley(): the Hufnagel-Valley Cn**2(h) model,
  layers_from_profile(): its integral over slabs as discrete layers,
- fried_parameter(), isoplanatic_angle(), coherence_time(),
  scintillation_index(): r0, theta0, tau0 and the weak fluctuation
  (Rytov) scintillation index of a star for a point detector or a circular
  aperture (aperture averaging).

Propagation and time evolution:

- split_step(): a field through a sequence of phase screens and free space
  sections (angular spectrum method).
- FrozenFlow: Taylor's frozen flow hypothesis; every layer moves as a rigid
  screen with its wind velocity, sampled with cubic splines.

Conventions
-----------
SI units are not required, but lengths (r0, L0, dx, heights, wavelength)
share one unit; Cn**2 is in length**(-2/3), Cn**2*dh in length**(1/3). Phases
are in radians at the wavelength for which r0 is given. Heights are along the
line of sight (zenith observations; for a zenith angle z multiply Cn**2*dh and
the heights by 1/cos(z)). Spatial frequencies are in cycles per length, angular
frequencies in cycles per radian.

References
----------
D. L. Fried, "Optical resolution through a randomly inhomogeneous medium for
very long and very short exposures", J. Opt. Soc. Am. 56, 1372-1379 (1966).

R. J. Noll, "Zernike polynomials and atmospheric turbulence", J. Opt. Soc. Am.
66, 207-211 (1976).

D. L. Fried, "Probability of getting a lucky short-exposure image through
turbulence", J. Opt. Soc. Am. 68, 1651-1658 (1978).

F. Roddier, "The effects of atmospheric turbulence in optical astronomy",
Progress in Optics 19, 281-376 (1981).

E. M. Johansson and D. T. Gavel, "Simulation of stellar speckle imaging",
Proc. SPIE 2200, 372-383 (1994).

R. G. Lane, A. Glindemann and J. C. Dainty, "Simulation of a Kolmogorov phase
screen", Waves Random Media 2, 209-224 (1992).

A. Tokovinin, "From differential image motion to seeing", Publ. Astron. Soc.
Pac. 114, 1156-1166 (2002).

J. D. Schmidt, "Numerical simulation of optical wave propagation", SPIE Press
(2010), chapter 9.
"""

import numpy as np
from numpy import pi
from scipy.special import gamma, j1, kv

from .fft import fft2, ifft2
from .propagators import rayleigh_sommerfeld_I_TF


# 2*(24/5*Gamma(6/5))**(5/6): D(r) = 6.88*(r/r0)**(5/3)
KOLMOGOROV_SF = 2*(24/5*gamma(6/5))**(5/6)
# phase PSD constant: Phi(f) = 0.0229*r0**(-5/3)*f**(-11/3) (f in cycles per length)
KOLMOGOROV_PSD = gamma(11/6)**2/(2*pi**(11/3))*(24/5*gamma(6/5))**(5/6)
# piston removed phase variance over a circular aperture, in (D/r0)**(5/3)
# (Noll gives 1.0299; the difference is the rounded PSD constant 0.023)
NOLL_PISTON_REMOVED = 1.0324216


def phase_psd(f, r0, L0=np.inf, l0=0.0):
    """von Kármán phase power spectral density.

    ``Phi(f) = 0.0229*r0**(-5/3)*(f**2 + 1/L0**2)**(-11/6)*exp(-(f/f_m)**2)``
    with ``f_m = 5.92/(2*pi*l0)``.

    Parameters
    ----------
    f : number or array
        Spatial frequency in cycles per length.
    r0 : number
        Fried parameter.
    L0 : number, optional
        Outer scale (default inf: Kolmogorov spectrum).
    l0 : number, optional
        Inner scale (default 0: no cutoff).

    Returns
    -------
    psd : number or array
        Phase PSD in rad**2*length**2 (the variance per frequency area). At
        f = 0 with L0 = inf the value is inf.
    """

    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        psd = KOLMOGOROV_PSD*r0**(-5/3)*(f**2 + 1/L0**2)**(-11/6)
    if l0 > 0:
        psd = psd*np.exp(-(2*pi*l0*f/5.92)**2)

    return psd


_N_CELL_AVERAGE = 4      # FFT frequencies up to this index get cell averaged PSDs
_N_SUB = 8               # sub-samples per cell and axis for the average


def _cell_average_psd(fx, fy, dfx, dfy, r0, L0, l0):
    """Effective PSD of the cells of size dfx x dfy around (fx, fy).

    The Kolmogorov spectrum is so steep that its value at the centre of the
    cells next to f = 0 misrepresents them badly. Each cell is replaced by
    one plane wave at its centre f_c; its weight is the mean of
    Phi(f)*f**2/f_c**2 over the cell, so that the cell contributes the right
    amount to the structure function D(r) at r << 1/f (the gradient
    variance). The plain variance (mean of Phi) would overestimate D(r).
    """

    u = (np.arange(_N_SUB) + 0.5)/_N_SUB - 0.5
    fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    fc2 = fx**2 + fy**2
    total = 0.0
    for a in u:
        for b in u:
            f2 = (fx + a*dfx)**2 + (fy + b*dfy)**2
            total = total + phase_psd(np.sqrt(f2), r0, L0, l0)*f2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fc2 > 0, total/_N_SUB**2/np.where(fc2 > 0, fc2, 1.0), 0.0)


def phase_screen(N, dx, r0, L0=np.inf, l0=0.0, subharmonics=6, rng=None, M=None):
    """Random phase screen with von Kármán statistics.

    The FFT method filters complex white noise with sqrt(Phi(f)*df_x*df_y)
    and takes the real part of the inverse transform. The grid lacks all
    frequencies below 1/(N*dx), which carry most of the phase variance
    (tilt). With subharmonics > 0, Lane's method adds them: for p = 1, ...,
    subharmonics a 3 x 3 grid of frequencies with the spacing 1/(3**p*N*dx)
    (without its centre) is added with the matching weights. The PSD of the
    lowest frequencies is averaged over their cells in frequency space
    (weighted with f**2, see _cell_average_psd()), which removes most of the
    lack of large scale power of the plain method (in the spirit of
    Johansson and Gavel 1994).

    Parameters
    ----------
    N : int
        Number of samples along y (rows).
    dx : number
        Sample spacing.
    r0 : number
        Fried parameter.
    L0, l0 : numbers, optional
        Outer and inner scale, see phase_psd().
    subharmonics : int, optional
        Number of subharmonic levels (default 6; 0 gives a periodic screen).
        Each level extends the spectrum by a factor of 3 towards low
        frequencies; for Kolmogorov turbulence six levels reproduce the
        structure function within a few per cent up to half the screen
        size, three levels give about 10 % less at a third of the size.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.
    M : int, optional
        Number of samples along x (columns), default N.

    Returns
    -------
    phi : array
        Phase in radians, shape (N, M), with zero mean.
    """

    rng = np.random.default_rng(rng)
    M = N if M is None else int(M)
    Ly, Lx = N*dx, M*dx
    fy = np.fft.fftfreq(N, dx)[:, None]
    fx = np.fft.fftfreq(M, dx)[None, :]
    psd = phase_psd(np.hypot(fx, fy), r0, L0, l0)
    # the lowest frequencies: average the steep PSD over their cells
    n_avg = min(_N_CELL_AVERAGE, N//2, M//2)
    iy = np.r_[0:n_avg + 1, N - n_avg:N]
    ix = np.r_[0:n_avg + 1, M - n_avg:M]
    psd[np.ix_(iy, ix)] = _cell_average_psd(fx[0, ix][None, :], fy[iy, 0][:, None],
                                            1/Lx, 1/Ly, r0, L0, l0)
    psd[0, 0] = 0.0
    noise = rng.standard_normal((N, M)) + 1j*rng.standard_normal((N, M))
    phi = np.real(ifft2(noise*np.sqrt(psd/(Lx*Ly))))*N*M

    if subharmonics > 0:
        y = (np.arange(N) - N//2)*dx
        x = (np.arange(M) - M//2)*dx
        low = np.zeros((N, M), dtype=complex)
        for p in range(1, subharmonics + 1):
            dfx, dfy = 1/(3**p*Lx), 1/(3**p*Ly)
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i == 0 and j == 0:
                        continue
                    c = (rng.standard_normal() + 1j*rng.standard_normal())
                    c *= np.sqrt(_cell_average_psd(j*dfx, i*dfy, dfx, dfy, r0, L0, l0)*dfx*dfy)
                    low += c*np.outer(np.exp(2j*pi*i*dfy*y), np.exp(2j*pi*j*dfx*x))
        phi = phi + np.real(low)

    return phi - np.mean(phi)


def structure_function(phi, dx, max_shift, mask=None):
    """Phase structure function of a screen along x and y.

    ``D(r) = <(phi(x + r) - phi(x))**2>``, averaged over all pairs of samples
    a distance r = s*dx apart along the rows and the columns.

    Parameters
    ----------
    phi : array or sequence of arrays
        Phase screen(s) of shape (N, M); several screens are averaged.
    dx : number
        Sample spacing.
    max_shift : int
        Largest shift s in samples.
    mask : array of bool, optional
        Only pairs with both samples inside the mask are used (e.g. a pupil).

    Returns
    -------
    r : array
        Separations dx, 2*dx, ..., max_shift*dx.
    D : array
        Structure function at r, in rad**2.
    """

    screens = [phi] if np.ndim(phi) == 2 else list(phi)
    r = dx*np.arange(1, max_shift + 1)
    total = np.zeros(max_shift)
    count = np.zeros(max_shift)
    for p in screens:
        m = np.ones(np.shape(p), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        for s in range(1, max_shift + 1):
            for a, b, ma, mb in ((p[:, s:], p[:, :-s], m[:, s:], m[:, :-s]),
                                 (p[s:, :], p[:-s, :], m[s:, :], m[:-s, :])):
                ok = ma & mb
                total[s - 1] += np.sum((a - b)[ok]**2)
                count[s - 1] += np.count_nonzero(ok)

    return r, total/count


def structure_function_theory(r, r0, L0=np.inf):
    """Theoretical phase structure function.

    Kolmogorov: ``D(r) = 6.88*(r/r0)**(5/3)``. von Kármán (finite L0)::

        D(r) = 0.1726*(L0/r0)**(5/3)*(1 - 2**(1/6)/Gamma(5/6)*a**(5/6)*K_5/6(a))

    with a = 2*pi*r/L0. It follows the Kolmogorov law for r << L0 and
    saturates at twice the phase variance, 0.1726*(L0/r0)**(5/3), for r >> L0.

    Parameters
    ----------
    r : number or array
        Separation.
    r0 : number
        Fried parameter.
    L0 : number, optional
        Outer scale (default inf).

    Returns
    -------
    D : number or array
        In rad**2.
    """

    r = np.asarray(r, dtype=float)
    if np.isinf(L0):
        return KOLMOGOROV_SF*(r/r0)**(5/3)

    a = 2*pi*r/L0
    with np.errstate(invalid="ignore"):
        shape = np.where(a > 0, 2**(1/6)/gamma(5/6)*a**(5/6)*kv(5/6, np.where(a > 0, a, 1.0)), 1.0)

    return 4*pi*KOLMOGOROV_PSD*3/5*(L0/r0)**(5/3)*(1 - shape)


def noll_index_to_n(j):
    """Radial order n of the Zernike polynomial with Noll index j (j >= 1)."""

    j = np.asarray(j)

    return np.floor((np.sqrt(8*j - 7) - 1)/2).astype(int)


def noll_variance(n, D_over_r0=1.0):
    """Variance of one Zernike coefficient of Kolmogorov turbulence.

    All Zernike modes of radial order n >= 1 have the same variance (Noll
    1976). The modes are normalized to an RMS of 1 over the pupil (Noll's
    normalization), so the variance is the phase variance in that mode:
    0.449, 0.0233, 0.00622, 0.00247, ... rad**2 for n = 1, 2, 3, 4, ...
    times (D/r0)**(5/3). It is the integral of the phase PSD times the
    squared Fourier transform of the mode, in closed form by the
    Weber-Schafheitlin integral.

    Parameters
    ----------
    n : int or array
        Radial order (>= 1).
    D_over_r0 : number, optional
        Aperture diameter in units of r0.

    Returns
    -------
    var : number or array
        In rad**2.
    """

    n = np.asarray(n, dtype=float)
    nu = n + 1
    # int_0^inf J_nu(t)**2 t**(-14/3) dt by Weber-Schafheitlin:
    integral = gamma(14/3)*gamma(nu - 11/6)/(2**(14/3)*gamma(17/6)**2*gamma(nu + 17/6))
    # 2*pi*f*Phi(f)*(n + 1)*(J_{n+1}(2 pi f R)/(pi f R))**2 integrated over f, with D = 2R = 1:
    var = 2*pi*KOLMOGOROV_PSD*(n + 1)/(pi/2)**2*pi**(11/3)*integral

    return var*D_over_r0**(5/3)


def noll_residual(J, D_over_r0=1.0):
    """Residual phase variance after correcting the first J Zernike modes.

    Delta_J of Noll (1976) for Noll indices 1 ... J (J = 1: piston removed,
    J = 3: tilt removed as well, ...), computed as the piston removed variance
    1.0324*(D/r0)**(5/3) minus the variances of modes 2 ... J. For large J,
    ``Delta_J`` approaches ``0.2944*J**(-sqrt(3)/2)*(D/r0)**(5/3)``. The
    residual only depends on how many modes of each radial order are removed,
    so it also holds for Fringe Zernikes when complete radial orders are
    corrected.

    Parameters
    ----------
    J : int or array of int
        Number of corrected modes (>= 1).
    D_over_r0 : number, optional
        Aperture diameter in units of r0.

    Returns
    -------
    Delta : number or array
        In rad**2.
    """

    J = np.asarray(J, dtype=int)
    j_max = int(np.max(J))
    var = noll_variance(noll_index_to_n(np.arange(2, j_max + 1)))
    removed = np.concatenate([[0.0], np.cumsum(var)])

    return (NOLL_PISTON_REMOVED - removed[J - 1])*D_over_r0**(5/3)


def atmospheric_otf(nu, r0, wavelength, L0=np.inf, D=None):
    """Atmospheric transfer function of long or short exposures.

    Long exposure: ``exp(-D_phi(wavelength*nu)/2)``, for Kolmogorov
    turbulence ``exp(-3.44*(wavelength*nu/r0)**(5/3))``. The total OTF is
    this times the OTF of the telescope.

    Short exposure (D given, Kolmogorov only): Fried's tilt-corrected result
    ``exp(-3.44*(wavelength*nu/r0)**(5/3)*(1 - (wavelength*nu/D)**(1/3)))``
    for ``wavelength*nu <= D`` (the "near field" case alpha = 1).

    Parameters
    ----------
    nu : number or array
        Angular frequency in cycles per radian.
    r0 : number
        Fried parameter at wavelength.
    wavelength : number
    L0 : number, optional
        Outer scale (long exposure only).
    D : number, optional
        Telescope diameter; if given, the short exposure OTF is returned.

    Returns
    -------
    otf : number or array
    """

    rho = wavelength*np.asarray(nu, dtype=float)
    if D is None:
        return np.exp(-structure_function_theory(rho, r0, L0)/2)

    u = np.clip(rho/D, 0, 1)

    return np.exp(-KOLMOGOROV_SF/2*(rho/r0)**(5/3)*(1 - u**(1/3)))


def seeing_fwhm(r0, wavelength, L0=np.inf):
    """FWHM of the long exposure PSF (seeing disk) in radians.

    ``0.976*wavelength/r0`` for Kolmogorov turbulence. A finite outer scale
    reduces it by ``sqrt(1 - 2.183*(r0/L0)**0.356)`` (Tokovinin 2002, for
    L0/r0 > 20).

    Parameters
    ----------
    r0 : number
        Fried parameter at wavelength.
    wavelength : number
    L0 : number, optional

    Returns
    -------
    fwhm : number
    """

    fwhm = 0.976*wavelength/r0
    if np.isfinite(L0):
        fwhm *= np.sqrt(1 - 2.183*(r0/L0)**0.356)

    return fwhm


def r0_at_wavelength(r0, wavelength_ref, wavelength):
    """Scale r0 from wavelength_ref to wavelength: r0 ~ wavelength**(6/5)."""

    return r0*(wavelength/wavelength_ref)**(6/5)


def lucky_probability(D_over_r0):
    """Fried's probability of a lucky short exposure.

    The probability that the phase over the aperture, with piston and tilt
    removed, has a variance of at most 1 rad**2:
    ``P = 5.6*exp(-0.1557*(D/r0)**2)``, Fried's (1978) fit for D/r0 >= 3.5.

    Parameters
    ----------
    D_over_r0 : number or array

    Returns
    -------
    P : number or array
    """

    return 5.6*np.exp(-0.1557*np.asarray(D_over_r0, dtype=float)**2)


def hufnagel_valley(h, v_rms=21.0, A=1.7e-14):
    """Hufnagel-Valley profile Cn**2(h) in m**(-2/3), h in metres.

    ``Cn2 = 0.00594*(v_rms/27)**2*(1e-5*h)**10*exp(-h/1000)
    + 2.7e-16*exp(-h/1500) + A*exp(-h/100)``.
    With the defaults (HV 5/7) r0 is about 5 cm and theta0 about 7 urad at
    500 nm.

    Parameters
    ----------
    h : number or array
        Height above the ground in metres.
    v_rms : number, optional
        RMS wind speed of the high altitude winds in m/s (the tropopause
        peak).
    A : number, optional
        Strength of the ground layer.

    Returns
    -------
    Cn2 : number or array
    """

    h = np.asarray(h, dtype=float)

    return (0.00594*(v_rms/27)**2*(1e-5*h)**10*np.exp(-h/1000)
            + 2.7e-16*np.exp(-h/1500) + A*np.exp(-h/100))


def layers_from_profile(profile, edges, samples=2000):
    """Discrete layers from a Cn**2 profile.

    Each slab between two edges becomes one layer with the integrated
    Cn**2*dh, at the Cn**2-weighted mean height of the slab.

    Parameters
    ----------
    profile : callable
        Cn**2(h).
    edges : array
        Slab boundaries (increasing heights).
    samples : int, optional
        Samples per slab for the trapezoidal integration.

    Returns
    -------
    heights : array
        Layer heights.
    cn2_dh : array
        Integrated strength of each layer.
    """

    heights, cn2_dh = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        h = np.linspace(a, b, samples)
        c = profile(h)
        w = np.trapezoid(c, h)
        cn2_dh.append(w)
        heights.append(np.trapezoid(c*h, h)/w)

    return np.array(heights), np.array(cn2_dh)


def fried_parameter(cn2_dh, wavelength):
    """Fried parameter of one or several layers together.

    ``r0 = (0.423*k**2*sum(Cn2*dh))**(-3/5)``.

    Parameters
    ----------
    cn2_dh : number or array
        Integrated Cn**2*dh of the layers.
    wavelength : number

    Returns
    -------
    r0 : number
    """

    k = 2*pi/wavelength

    return (0.423*k**2*np.sum(cn2_dh))**(-3/5)


def isoplanatic_angle(heights, cn2_dh, wavelength):
    """Isoplanatic angle ``theta0 = (2.914*k**2*sum(Cn2*dh*h**(5/3)))**(-3/5)``.

    Two stars theta0 apart see phases that differ by 1 rad**2 (Fried 1982),
    theta0 = 0.314*r0/h_bar with the effective turbulence height h_bar.

    Parameters
    ----------
    heights, cn2_dh : arrays
        Layer heights and strengths.
    wavelength : number

    Returns
    -------
    theta0 : number
        In radians.
    """

    k = 2*pi/wavelength

    return (2.914*k**2*np.sum(np.asarray(cn2_dh)*np.asarray(heights)**(5/3)))**(-3/5)


def coherence_time(cn2_dh, speeds, wavelength):
    """Coherence time ``tau0 = 0.314*r0/v_eff`` of frozen flow layers.

    v_eff is the average ``(sum(Cn2*dh*v**(5/3))/sum(Cn2*dh))**(3/5)``. A
    perfect correction delayed by tau leaves the phase variance
    ``(tau/tau0)**(5/3)`` rad**2 (the structure function at v*tau).

    Parameters
    ----------
    cn2_dh, speeds : arrays
        Layer strengths and wind speeds.
    wavelength : number

    Returns
    -------
    tau0 : number
    """

    cn2_dh = np.asarray(cn2_dh, dtype=float)
    v_eff = (np.sum(cn2_dh*np.abs(speeds)**(5/3))/np.sum(cn2_dh))**(3/5)

    return KOLMOGOROV_SF**(-3/5)*fried_parameter(cn2_dh, wavelength)/v_eff


def scintillation_index(heights, cn2_dh, wavelength, D=0.0, f_max=None):
    """Weak fluctuation scintillation index of a star (plane wave).

    The relative variance of the intensity at the ground (or of the flux
    through a circular aperture of diameter D) in the Rytov approximation:
    each layer at height h turns phase into log-amplitude with the Fresnel
    filter sin(pi*wavelength*h*f**2), so that ::

        sigma_I**2 = 4*sum_i int 2*pi*f*Phi_i(f)*sin(pi*wavelength*h_i*f**2)**2*A(f) df

    with the phase PSD Phi_i of the layer and the aperture filter
    ``A(f) = (2*J1(pi*D*f)/(pi*D*f))**2``. For a point detector this is
    ``19.12*wavelength**(-7/6)*sum(Cn2*dh*h**(5/6))``; for D much larger than
    the Fresnel scale sqrt(wavelength*h) it falls as D**(-7/3).

    Parameters
    ----------
    heights, cn2_dh : arrays
        Layer heights and strengths.
    wavelength : number
    D : number, optional
        Aperture diameter (default 0: point detector).
    f_max : number, optional
        Upper integration limit in cycles per length (e.g. the Nyquist
        frequency 1/(2*dx) of a simulation grid). Default: no limit.

    Returns
    -------
    sigma2 : number
    """

    heights = np.atleast_1d(np.asarray(heights, dtype=float))
    cn2_dh = np.atleast_1d(np.asarray(cn2_dh, dtype=float))
    if D == 0 and f_max is None:
        return 19.12*wavelength**(-7/6)*np.sum(cn2_dh*heights**(5/6))

    sigma2 = 0.0
    for h, c in zip(heights, cn2_dh):
        if h <= 0:
            continue
        r0_i = fried_parameter(c, wavelength)
        f_F = 1/np.sqrt(wavelength*h)
        top = 60*f_F if f_max is None else f_max
        f = np.geomspace(1e-4*f_F, top, 200000)
        integrand = 2*pi*f*phase_psd(f, r0_i)*np.sin(pi*wavelength*h*f**2)**2
        if D > 0:
            a = pi*D*f
            integrand *= (2*j1(a)/a)**2
        sigma2 += 4*np.trapezoid(integrand, f)

    return sigma2


def split_step(field, x, y, screens, distances, wavelength):
    """Propagate a field through phase screens and free space sections.

    The field is multiplied by exp(i*screens[0]), propagated by
    distances[0], multiplied by exp(i*screens[1]), and so on (angular
    spectrum method, `propagators.rayleigh_sommerfeld_I_TF()`). For a star,
    start with a plane wave at the highest layer and propagate downwards.
    The grid is periodic: use periodic screens (no subharmonics) or keep
    the region of interest away from the edges.

    Parameters
    ----------
    field : array
        Input field.
    x, y : arrays
        Linear coordinates.
    screens : sequence of arrays
        Phase screens in radians (at wavelength).
    distances : sequence of numbers
        Distance after each screen (0 for the last one at the ground).
    wavelength : number

    Returns
    -------
    field : array
        Field after the last section.
    """

    u = np.asarray(field, dtype=complex)
    for phi, z in zip(screens, distances):
        u = u*np.exp(1j*phi)
        if z != 0:
            u = rayleigh_sommerfeld_I_TF(u, x, y, z, wavelength)[0]

    return u


class FrozenFlow(object):
    """Phase screens moving with the wind (Taylor's frozen flow).

    Every layer is a large screen that moves rigidly with its velocity. The
    phase in the pupil at time t is the sum over the layers of the screen,
    sampled with cubic splines on a window that starts at
    origin + velocity*t. The screens are treated as periodic; screens
    with subharmonics are not, so keep the windows inside the screens over
    the simulated time.

    Parameters
    ----------
    screens : sequence of arrays
        Phase (or OPD) screens, shape (N, M) each.
    dx : number
        Sample spacing of the screens.
    velocities : sequence of (vx, vy)
        Wind velocity of each layer (length per time unit).
    shape : (int, int)
        Shape of the output window (samples with spacing dx).
    origins : sequence of (x0, y0), optional
        Lower left corner of each window at t = 0, in length units relative
        to the first sample of the screen. Default: windows centred at
        t = t_mid (see below), so that they stay inside the screens best.
    t_mid : number, optional
        Time at which the default windows are centred.

    Notes
    -----
    Sub-sample shifts by cubic spline interpolation damp the highest
    spatial frequencies slightly (by less than 10 % in amplitude up to
    half the Nyquist frequency).
    """

    def __init__(self, screens, dx, velocities, shape, origins=None, t_mid=0.0):
        from scipy.ndimage import spline_filter

        self.dx = float(dx)
        self.velocities = np.array(velocities, dtype=float).reshape(-1, 2)
        self.shape = tuple(shape)
        self._coeffs = [spline_filter(np.asarray(s, dtype=float), order=3, mode="grid-wrap")
                        for s in screens]
        if origins is None:
            origins = []
            for s, v in zip(screens, self.velocities):
                N, M = np.shape(s)
                origins.append(((M - self.shape[1])*self.dx/2 - v[0]*t_mid,
                                (N - self.shape[0])*self.dx/2 - v[1]*t_mid))
        self.origins = np.array(origins, dtype=float).reshape(-1, 2)
        self._iy, self._ix = np.indices(self.shape, dtype=float)

    def layer(self, i, t):
        """Window of layer i at time t, shape `shape`."""

        from scipy.ndimage import map_coordinates

        x0, y0 = self.origins[i] + self.velocities[i]*t
        coords = (self._iy + y0/self.dx, self._ix + x0/self.dx)

        return map_coordinates(self._coeffs[i], coords, order=3, mode="grid-wrap", prefilter=False)

    def __call__(self, t):
        """Sum of all layers at time t."""

        return sum(self.layer(i, t) for i in range(len(self._coeffs)))
