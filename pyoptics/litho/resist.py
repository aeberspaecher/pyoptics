#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Resist model for chemically amplified (DUV) resists in an x-z cross
section: exposure, post-exposure bake (PEB) and development.

The constant threshold model of metrics prints where the image exceeds a
threshold. This module follows the physical chain instead, on a grid that
spans one period across the feature (x, periodic) and the resist thickness
(z, below the resist top):

1. Image in the resist (intensity_in_resist()): ``|E|**2`` inside the resist of
   a LithoSimulation with a WaferStack, including the standing waves,
   relative to the open frame intensity incident on the stack (the aerial
   open frame in the immersion medium).

2. Exposure: the photo acid generator turns into acid with the Dill C
   parameter, ``h_0 = 1 - exp(-C*D*I)`` for the dose D (mJ/cm**2) and C
   (cm**2/mJ). DUV resists bleach little, so the optical constants stay
   fixed (no Dill A and B).

3. PEB: the acid h (relative to the acid generator concentration) diffuses
   with the diffusion length sigma = sqrt(2*D_acid*t_PEB), is neutralized by
   a base quencher q (initial concentration quencher, diffusion length
   quencher_diffusion_length) and deprotects the polymer, ::

       dh/dtau = sigma**2/2*Laplace(h) - kappa*h*q,
       dq/dtau = sigma_q**2/2*Laplace(q) - kappa*h*q,
       dm/dtau = -K*h*m,

   over the normalized bake time tau from 0 to 1 with m = 1 (fully
   protected) at the start, the amplification K = k_amp*t_PEB and the
   neutralization rate kappa (None: instantaneous). Diffusion is spectral
   (FFT along x, periodic; cosine transform along z, no flux through the top
   and the bottom of the resist), the reactions are integrated exactly over
   each of n_steps time steps (operator splitting).

4. Development with the Mack rate model, ::

       r(m) = r_max*(a + 1)*(1 - m)**n/(a + (1 - m)**n) + r_min,
       a = (n + 1)/(n - 1)*(1 - m_th)**n,

   (nm/s; m_th is the threshold inhibitor concentration, n the dissolution
   selectivity), and the development front from the arrival times T(x, z)
   of the eikonal equation ``|grad T| = 1/r`` with T = 0 at the resist top,
   solved by fast marching. The resist is removed where T <= develop_time.

5. Dose: ResistModel.dose_to_size() finds the dose for a target CD at a
   given depth, ResistModel.dose_to_clear() the dose at which the open frame
   (open_frame_in_resist()) develops through, E_0, whose swing with the
   resist thickness is the classic swing curve.

6. Profile: the CD of the first feature of the layout through the resist
   thickness (a line of resist for a dark mask feature, an opening for a
   bright one), CDs at the top, middle and bottom (10, 50 and 90 % of the
   thickness), sidewall angle and top loss.

The geometry is a cut across the feature: diffusion along the feature
(y) is neglected, which is exact for lines and spaces.

Lengths in nm, times in s, doses in mJ/cm**2.
"""

import heapq

import numpy as np
from scipy.fft import dct, idct, irfft, rfft
from scipy.optimize import brentq



class LatentImage(object):
    """Intensity in the resist on the grid x (one period, periodic), z
    (cell centers of the resist thickness); I has shape (len(z), len(x)).
    center is the feature center, opening True if the feature is an
    opening in the resist (a bright feature on the mask)."""

    def __init__(self, x, z, I, center, opening, thickness, pitch):
        self.x, self.z, self.I = x, z, I
        self.center, self.opening = center, opening
        self.thickness, self.pitch = thickness, pitch

    @property
    def dx(self):
        return self.pitch/len(self.x)

    @property
    def dz(self):
        return self.thickness/len(self.z)


def intensity_in_resist(sim, focus=0.0, N_x=128, N_z=40, direction=None):
    """Image in the resist across the first feature of the layout.

    Parameters
    ----------
    sim : LithoSimulation
        With a WaferStack.
    focus : number, optional
        Position of the resist top relative to the focal plane.
    N_x, N_z : int, optional
        Grid points across one period and through the resist thickness.
    direction : "x" or "y", optional
        Direction of the cut (default: as LithoSimulation.cut()).

    Returns
    -------
    LatentImage
        Intensity relative to the open frame intensity incident on the
        stack.
    """

    if sim.wafer is None:
        raise ValueError("intensity_in_resist needs a LithoSimulation with a WaferStack")
    _, center, _, direction = sim.cut_coordinates(3, 1.0, direction)
    L = sim.layout
    pitch = L.pitch_x if direction == "x" else L.pitch_y
    if pitch is None:
        raise ValueError("the cut must run along a periodic direction")
    x = center + pitch*(np.arange(N_x)/N_x - 0.5)
    d = sim.wafer.thickness
    z = (np.arange(N_z) + 0.5)*d/N_z
    raw = sim.with_(normalize=False)
    I_0 = sim.with_(wafer=None, normalize=False).open_frame(focus)
    x0, y0 = L.features[0][:2]
    I = []
    for depth in z:
        if direction == "x":
            I.append(raw.image(x, None if L.pitch_y is None else [y0], focus, depth).ravel())
        else:
            I.append(raw.image([x0], x, focus, depth).ravel())

    return LatentImage(x, z, np.array(I)/I_0, center, L.feature_is_bright, d, pitch)


def open_frame_in_resist(sim, focus=0.0, N_z=40):
    """Intensity in the resist for a clear mask (the open frame).

    Use it e.g. for `ResistModel.dose_to_clear()`.

    Parameters
    ----------
    sim : LithoSimulation
        Simulation with a WaferStack.
    focus : number, optional
        Position of the resist top relative to the focal plane.
    N_z : int, optional
        Number of cells across the resist thickness.

    Returns
    -------
    LatentImage
        A single column, normalized to the open frame without the stack.

    Raises
    ------
    ValueError
        If the simulation has no wafer stack.
    """

    if sim.wafer is None:
        raise ValueError("open_frame_in_resist needs a LithoSimulation with a WaferStack")
    d = sim.wafer.thickness
    z = (np.arange(N_z) + 0.5)*d/N_z
    raw = sim.with_(normalize=False)
    I_0 = sim.with_(wafer=None, normalize=False).open_frame(focus)
    I = np.array([[raw.open_frame(focus, depth)] for depth in z])/I_0

    return LatentImage(np.zeros(1), z, I, 0.0, True, d, 1.0)


def _diffuse(u, kx2, kz2, s2):
    """Gaussian blur with variance s2 of u (shape (N_z, N_x)), periodic in x
    and with no flux through the top and bottom (cosine transform)."""

    if s2 == 0:
        return u
    U = rfft(dct(u, type=2, axis=0, norm="ortho"), axis=1)
    U = U*np.exp(-0.5*s2*(kz2[:, None] + kx2[None, :]))

    return idct(irfft(U, n=u.shape[1], axis=1), type=2, axis=0, norm="ortho")


def _neutralize(h, q, rate, dtau):
    """Acid-base neutralization dh = dq = -rate*h*q*dtau, exact (rate None:
    instantaneous)."""

    if rate is None:
        return np.maximum(h - q, 0.0), np.maximum(q - h, 0.0)
    c = h - q  # conserved
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = np.exp(-rate*c*dtau)
        h_new = np.where(np.abs(c) > 1e-12, c/(1 - (1 - c/h)*e), h/(1 + rate*h*dtau))
    h_new = np.where(h > 0, h_new, 0.0)

    return h_new, h_new - c


def mack_rate(m, r_max=100.0, r_min=0.1, n=5.0, m_th=0.5):
    """Development rate of the Mack model.

    Parameters
    ----------
    m : number or array
        Relative inhibitor concentration in [0, 1].
    r_max, r_min : numbers, optional
        Largest and smallest rate (nm/s).
    n : number, optional
        Dissolution selectivity.
    m_th : number, optional
        Threshold inhibitor concentration.

    Returns
    -------
    rate : number or array
        Development rate in nm/s.
    """

    a = (n + 1)/(n - 1)*(1 - m_th)**n
    u = (1 - np.asarray(m, dtype=float))**n

    return r_max*(a + 1)*u/(a + u) + r_min


def develop(rate, dx, dz):
    """Arrival times T of the development front, ``|grad T| = 1/rate`` with
    T = 0 at the resist top (half a cell above the first row), periodic
    along x (axis 1), by first order fast marching.

    Parameters
    ----------
    rate : array, shape (N_z, N_x)
        Development rate in the cells (positive).
    dx, dz : numbers
        Grid spacings.
    """

    N_z, N_x = rate.shape
    slowness = (1.0/np.asarray(rate, dtype=float)).ravel().tolist()
    inf = float("inf")
    T = [inf]*(N_z*N_x)
    known = bytearray(N_z*N_x)

    def update(j, i):
        k = j*N_x + i
        f = slowness[k]
        l, r = j*N_x + (i - 1) % N_x, j*N_x + (i + 1) % N_x
        a = min(T[l] if known[l] else inf, T[r] if known[r] else inf)
        # along z: the row above (or the resist top at dz/2) and the row below
        cands = [(0.0, dz/2)] if j == 0 else ([(T[k - N_x], dz)] if known[k - N_x] else [])
        if j < N_z - 1 and known[k + N_x]:
            cands.append((T[k + N_x], dz))
        b, h_z = min(cands, key=lambda c: c[0] + f*c[1]) if cands else (inf, dz)
        if a == inf:
            return b + f*h_z
        if b == inf:
            return a + f*dx
        A = 1/dx**2 + 1/h_z**2
        B = a/dx**2 + b/h_z**2
        C = a**2/dx**2 + b**2/h_z**2 - f**2
        disc = B**2 - A*C
        if disc >= 0:
            t = (B + disc**0.5)/A
            if t >= max(a, b):
                return t

        return min(a + f*dx, b + f*h_z)

    heap = []
    for i in range(N_x):
        T[i] = update(0, i)
        heap.append((T[i], 0, i))
    heapq.heapify(heap)
    while heap:
        t, j, i = heapq.heappop(heap)
        k = j*N_x + i
        if known[k]:
            continue
        known[k] = 1
        for jj, ii in ((j - 1, i), (j + 1, i), (j, (i - 1) % N_x), (j, (i + 1) % N_x)):
            if 0 <= jj < N_z and not known[jj*N_x + ii]:
                t_new = update(jj, ii)
                if t_new < T[jj*N_x + ii]:
                    T[jj*N_x + ii] = t_new
                    heapq.heappush(heap, (t_new, jj, ii))

    return np.array(T).reshape(N_z, N_x)


class Profile(object):
    """Resist profile of the feature: z (rows), left, right and cd per row
    (nan where the feature is not resolved), cd_top, cd_middle and
    cd_bottom (at 10, 50 and 90 % of the thickness), sidewall_angle
    (degrees, 90 for vertical walls, < 90 for a resist line that widens
    towards the bottom or an opening that narrows) and top_loss (thickness
    lost at the top of the resist between the features)."""

    def __init__(self, z, left, right, thickness, top_loss, opening):
        self.z, self.left, self.right = z, left, right
        self.cd = right - left
        self.thickness, self.top_loss, self.opening = thickness, top_loss, opening
        self.cd_top, self.cd_middle, self.cd_bottom = (self.cd_at(f*thickness) for f in (0.1, 0.5, 0.9))

    def cd_at(self, z):
        """CD at the depth z (linear interpolation, nan if not resolved)."""

        return np.interp(z, self.z, self.cd)

    @property
    def sidewall_angle(self):
        rows = (self.z >= 0.1*self.thickness) & (self.z <= 0.9*self.thickness) & np.isfinite(self.cd)
        if rows.sum() < 2:
            return np.nan
        slope = np.polyfit(self.z[rows], self.cd[rows]/2, 1)[0]  # d(half width)/dz

        return 90.0 - np.degrees(np.arctan(-slope if self.opening else slope))


def _front(u_a, u_b, T_a, T_b, r_a, r_b, t):
    """Position of the development front at the time t between the sample
    a (developed, T_a <= t) and b (T_b > t): piecewise linear in time with
    the break at the cell boundary halfway, reached after a share of
    T_b - T_a in proportion to the slowness of cell a (linear
    interpolation for equal rates, and the cell boundary for very
    different ones)."""

    tau_a, tau_b = 1/r_a, 1/r_b
    t_mid = T_a + (T_b - T_a)*tau_a/(tau_a + tau_b)
    if t <= t_mid:
        f = 0.5*(t - T_a)/(t_mid - T_a)
    else:
        f = 0.5 + 0.5*(t - t_mid)/(T_b - t_mid)

    return u_a + f*(u_b - u_a)


def _edges(x, T, rate, t, i_c, opening):
    """Edges of the feature at the index i_c in one row (nan if it is not
    resolved or reaches the end of the row)."""

    inside = T <= t if opening else T > t
    if not inside[i_c]:
        return np.nan, np.nan
    out = np.nonzero(~inside)[0]
    left, right = out[out < i_c], out[out > i_c]
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan
    edges = []
    for i_out, i_in in ((left[-1], left[-1] + 1), (right[0], right[0] - 1)):
        a, b = (i_in, i_out) if opening else (i_out, i_in)  # a: developed side
        edges.append(_front(x[a], x[b], T[a], T[b], rate[a], rate[b], t))

    return edges[0], edges[1]


def profile(latent, arrival, develop_time, rate=None):
    """Resist profile of the feature from the arrival times of the development front.

    Without the rate, the front is interpolated linearly in time between the
    samples.

    Parameters
    ----------
    latent : LatentImage
        Grid and feature of the simulation.
    arrival : (N_z, N_x) array
        Arrival times of the development front at the cell centers.
    develop_time : number
        Develop time (s).
    rate : (N_z, N_x) array, optional
        Development rate (nm/s).

    Returns
    -------
    Profile
        Edges per depth and the top loss.
    """

    x, z = latent.x, latent.z
    if rate is None:
        rate = np.ones_like(arrival)
    i_c = int(np.argmin(np.abs(x - latent.center)))
    edges = np.array([_edges(x, T, r, develop_time, i_c, latent.opening)
                      for T, r in zip(arrival, rate)])
    # top loss: the resist column farthest from the openings (line center, or half a period from
    # the center of an opening, the first column); the resist top is part of the first cell
    i = 0 if latent.opening else i_c
    zs, Ts = np.concatenate([[0.0], z]), np.concatenate([[0.0], arrival[:, i]])
    rs = np.concatenate([[rate[0, i]], rate[:, i]])
    if Ts[-1] <= develop_time:
        top_loss = latent.thickness
    else:
        k = int(np.argmax(Ts > develop_time))
        top_loss = _front(zs[k - 1], zs[k], Ts[k - 1], Ts[k], rs[k - 1], rs[k], develop_time)

    return Profile(z, edges[:, 0], edges[:, 1], latent.thickness, top_loss, latent.opening)


class ResistResult(object):
    """Result of ResistModel.process(): the latent image, the dose, the acid
    after exposure (acid_0) and after the bake (acid), the inhibitor
    concentration m, the development rate, the arrival times of the
    development front, the develop time and the profile; all arrays of
    shape (N_z, N_x)."""

    def __init__(self, latent, dose, acid_0, acid, inhibitor, rate, arrival, develop_time, profile):
        self.latent, self.dose = latent, dose
        self.acid_0, self.acid, self.inhibitor = acid_0, acid, inhibitor
        self.rate, self.arrival, self.develop_time = rate, arrival, develop_time
        self.profile = profile

    @property
    def resist(self):
        """True where resist remains after development."""

        return self.arrival > self.develop_time


class ResistModel(object):
    """Chemically amplified resist: exposure, PEB and Mack development (see
    the module documentation).

    Parameters
    ----------
    C : number, optional
        Dill C (cm**2/mJ).
    diffusion_length : number, optional
        Acid diffusion length sqrt(2*D*t_PEB) (nm).
    amplification : number, optional
        k_amp*t_PEB.
    quencher : number, optional
        Initial base quencher concentration (relative to the acid generator).
    quencher_diffusion_length : number, optional
    neutralization : number or None, optional
        kappa*t_PEB; None for instantaneous neutralization.
    r_max, r_min, n, m_th : numbers, optional
        Mack development rate parameters (nm/s).
    develop_time : number, optional
        s.
    n_steps : int, optional
        Time steps of the bake.
    """

    def __init__(self, C=0.05, diffusion_length=20.0, amplification=2.0, quencher=0.0,
                 quencher_diffusion_length=0.0, neutralization=None, r_max=100.0, r_min=0.1, n=5.0,
                 m_th=0.5, develop_time=60.0, n_steps=40):
        self.C = C
        self.diffusion_length, self.amplification = diffusion_length, amplification
        self.quencher, self.quencher_diffusion_length = quencher, quencher_diffusion_length
        self.neutralization = neutralization
        self.r_max, self.r_min, self.n, self.m_th = r_max, r_min, n, m_th
        self.develop_time = develop_time
        self.n_steps = n_steps

    def with_(self, **kwargs):
        """Copy with some parameters exchanged."""

        p = dict(vars(self))
        p.update(kwargs)

        return ResistModel(**p)

    def expose(self, I, dose):
        """Relative acid concentration after exposure, 1 - exp(-C*dose*I).

        Parameters
        ----------
        I : array
            Relative intensity in the resist.
        dose : number
            Dose in mJ/cm**2.

        Returns
        -------
        acid : array
        """

        return 1 - np.exp(-self.C*dose*np.asarray(I))

    def bake(self, acid, dx, dz):
        """Post exposure bake: acid diffusion, quenching and amplification.

        Parameters
        ----------
        acid : (N_z, N_x) array
            Acid after exposure.
        dx : number
            Grid spacing in x (periodic).
        dz : number
            Spacing of the cell centers in z.

        Returns
        -------
        acid : (N_z, N_x) array
            Acid after the bake.
        m : (N_z, N_x) array
            Relative inhibitor concentration.
        """

        N_z, N_x = acid.shape
        kx2 = (2*np.pi*np.fft.rfftfreq(N_x, dx))**2
        kz2 = (np.pi*np.arange(N_z)/(N_z*dz))**2
        dtau = 1.0/self.n_steps
        s2_h = self.diffusion_length**2*dtau
        s2_q = self.quencher_diffusion_length**2*dtau
        h = np.array(acid, dtype=float)
        q = np.full_like(h, float(self.quencher))
        m = np.ones_like(h)
        has_q = self.quencher > 0
        if has_q and self.neutralization is None:
            h, q = _neutralize(h, q, None, 0.0)
        for _ in range(self.n_steps):
            m *= np.exp(-0.5*self.amplification*h*dtau)
            h = _diffuse(h, kx2, kz2, s2_h)
            if has_q:
                q = _diffuse(q, kx2, kz2, s2_q)
                h, q = _neutralize(h, q, self.neutralization, dtau)
            m *= np.exp(-0.5*self.amplification*h*dtau)

        return h, m

    def rate(self, m):
        """Mack development rate (nm/s)."""

        return mack_rate(m, self.r_max, self.r_min, self.n, self.m_th)

    def process(self, latent, dose):
        """Expose, bake and develop.

        Parameters
        ----------
        latent : LatentImage
            From intensity_in_resist().
        dose : number
            mJ/cm**2.

        Returns
        -------
        ResistResult
        """

        acid_0 = self.expose(latent.I, dose)
        acid, m = self.bake(acid_0, latent.dx, latent.dz)
        rate = self.rate(m)
        T = develop(rate, latent.dx, latent.dz)
        return ResistResult(latent, dose, acid_0, acid, m, rate, T, self.develop_time,
                            profile(latent, T, self.develop_time, rate))

    def dose_to_size(self, latent, cd_target, depth=0.5, doses=(1.0, 300.0), n_scan=40):
        """Dose that prints cd_target at a relative depth of the resist.

        A geometric scan of n_scan doses brackets the solution, which is
        then refined by root finding. Between doses where the feature
        appears or disappears at that depth the scan is repeated more
        finely, since the window of doses that resolve it can be narrow
        (standing waves). The lowest dose found is returned.

        Parameters
        ----------
        latent : LatentImage
            From intensity_in_resist().
        cd_target : number
            Target CD (nm).
        depth : number, optional
            Relative depth, 0: top, 1: bottom of the resist.
        doses : (number, number), optional
            Range of doses in mJ/cm**2.
        n_scan : int, optional
            Doses of the initial scan.

        Returns
        -------
        dose : number
            Dose in mJ/cm**2; nan if no dose in the range prints cd_target.
        """

        z = depth*latent.thickness

        def error(dose):
            return self.process(latent, dose).profile.cd_at(z) - cd_target

        def search(lo, hi, n, level):
            scan = np.geomspace(lo, hi, n)
            e = np.array([error(D) for D in scan])
            for k in range(n - 1):
                if np.isfinite(e[k]) and np.isfinite(e[k + 1]) and e[k]*e[k + 1] <= 0:
                    return brentq(error, scan[k], scan[k + 1], xtol=1e-4*scan[k])
            if level < 2:
                for k in range(n - 1):
                    if np.isfinite(e[k]) != np.isfinite(e[k + 1]):
                        D = search(scan[k], scan[k + 1], 8, level + 1)
                        if np.isfinite(D):
                            return D
            return np.nan

        return search(doses[0], doses[1], n_scan, 0)

    def dose_to_clear(self, latent, doses=(0.1, 1000.0)):
        """Dose to clear, E_0.

        The dose (mJ/cm**2) at which the development front reaches the bottom of
        the resist within the develop time everywhere; for the open frame
        (`open_frame_in_resist()`) this is the classic E_0.

        Parameters
        ----------
        latent : LatentImage
            Intensity in the resist.
        doses : (number, number), optional
            Range of doses in mJ/cm**2.

        Returns
        -------
        dose : number
            nan if the resist does not clear within the range of doses.
        """

        dz = latent.dz

        def error(log_dose):
            dose = np.exp(log_dose)
            acid, m = self.bake(self.expose(latent.I, dose), latent.dx, dz)
            rate = self.rate(m)
            T = develop(rate, latent.dx, dz)
            return np.log((T[-1] + 0.5*dz/rate[-1]).max()/self.develop_time)

        lo, hi = np.log(doses[0]), np.log(doses[1])
        if error(lo)*error(hi) > 0:
            return np.nan

        return np.exp(brentq(error, lo, hi, xtol=1e-5))


__all__ = ["LatentImage", "intensity_in_resist", "open_frame_in_resist", "mack_rate", "develop", "Profile", "profile",
           "ResistResult", "ResistModel"]
