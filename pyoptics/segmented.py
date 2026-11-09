#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Segmented telescope mirrors: hexagonal segments, their diffraction
pattern, segment piston and tip-tilt errors, and phasing (notebook 98).

A segmented primary is described by a stack of segment masks (one
antialiased transmission map per segment) on a pupil grid in metres. The
field in the pupil is the sum of the segments, each with its own piston
and tip-tilt (`segment_field()`); the far field is computed with the matrix
Fourier transform of `coronagraph.mft()` at angles in radians
(`far_field()`).

Closed form results used for checks:

- Fourier transform of a polygon (`polygon_ft()`): by the divergence
  theorem the integral over a polygon of exp(-i k.r) is a sum over its
  edges, ::

      F(k) = sum_j (i k.n_j/k^2) L_j exp(-i k.m_j) sinc(k.e_j/2),

  with the outward normals n_j, lengths L_j, midpoints m_j and edge vectors
  e_j, sinc(x) = sin(x)/x. The intensity of a hexagon falls off as k^-4
  except along the six edge normals, where it falls off as k^-2: the
  diffraction spikes.
- Identical segments at the centres c_j: the far field is the segment's
  times the array factor sum_j exp(-i k.c_j).
- On axis, a segment with the wavefront tilt t (radians) contributes the
  transform of its hexagon at k = -2 pi t/lambda, and a piston p the factor
  exp(2 pi i p/lambda). For small errors the Strehl ratio is close to
  exp(-sigma^2) with the rms phase sigma (Maréchal); for independent
  random pistons the mean PSF is the segment PSF times
  exp(-sigma^2) abs(AF)^2 + (1 - exp(-sigma^2)) N.
- A regular hexagon of side s (flat-to-flat 2 h = sqrt(3) s) has the area
  3 sqrt(3) s^2/2 and the polar moment of area 5 s^2/12 per unit area
  (`hexagon_rms_radius()`).
- Dispersed fringes of two segments with the piston difference p and the
  baseline B (centre to centre): across the spectrum the intensity at the
  angle theta along the baseline varies as 1 + V cos(2 pi (p - B theta)/
  lambda), so the number of fringes across the band [sigma_1, sigma_2] of
  wavenumbers sigma = 1/lambda is p (sigma_2 - sigma_1), and the fringes
  tilt with the sign of p (`dispersed_fringes()`, `dfs_piston()`).

Functions:

- hexagon_vertices(), hex_centres(), polygon_mask(), segment_masks(),
  strut_mask(), jwst_like_pupil(): segmented apertures.
- polygon_ft(), array_factor(), hexagon_rms_radius(): closed forms.
- segment_field(), far_field(): fields with segment errors.
- dispersed_fringes(), dfs_piston(): coarse phasing with a dispersed
  fringe sensor.
- diversity_psfs(), misell(), fit_segments(): fine phasing by phase
  retrieval from defocused images (multi-plane Gerchberg-Saxton) and the
  least squares fit of segment piston and tip-tilt.

References
----------
G. Chanan, M. Troy, F. Dekens, S. Michaels, J. Nelson, T. Mast and D.
Kirkman, "Phasing the mirror segments of the Keck telescopes: the
broadband phasing algorithm", Appl. Opt. 37, 140-155 (1998).

F. Shi, D. C. Redding, A. E. Lowman, C. W. Bowers, L. A. Burns, P. Petrone,
C. M. Ohara and S. A. Basinger, "Segmented mirror coarse phasing with a
dispersed fringe sensor: experiment on NGST's wavefront control testbed",
Proc. SPIE 4850, 318-328 (2003).

D. S. Acton, J. S. Knight, A. Contos, S. Grimaldi, J. Terry, P. Lightsey et
al., "Wavefront sensing and controls for the James Webb Space Telescope",
Proc. SPIE 8442, 84422H (2012).

R. A. Gonsalves, "Phase retrieval and diversity in adaptive optics",
Opt. Eng. 21, 829-832 (1982); D. L. Misell, "An examination of an iterative
method for the solution of the phase problem in optics and electron optics",
J. Phys. D 6, 2200-2216 (1973).

S.-W. Lee and R. Mittra, "Fourier transform of a polygonal shape
function and its application in electromagnetics", IEEE Trans. Antennas
Propag. 31, 99-103 (1983).
"""

from collections import namedtuple

import numpy as np

from .coronagraph import mft


__all__ = ["hexagon_vertices", "hex_centres", "polygon_mask", "segment_masks", "strut_mask",
           "SegmentedPupil", "jwst_like_pupil", "polygon_ft", "array_factor",
           "hexagon_rms_radius", "segment_field", "far_field", "dispersed_fringes",
           "dfs_piston", "diversity_psfs", "misell", "fit_segments"]


def hexagon_vertices(centre, flat, angle=0.0):
    """Vertices of a regular hexagon (counter-clockwise).

    Parameters
    ----------
    centre : (float, float)
        Centre (x, y).
    flat : float
        Flat-to-flat width.
    angle : float, optional
        Direction of the first vertex from the x axis in radians (0: flat
        top and bottom).

    Returns
    -------
    V : (6, 2) array
    """

    s = flat/np.sqrt(3)                   # side = circumradius
    a = angle + np.arange(6)*np.pi/3
    return np.c_[centre[0] + s*np.cos(a), centre[1] + s*np.sin(a)]


def hex_centres(n_rings, pitch, angle=0.0, skip_centre=True):
    """Centres of a hexagonal arrangement of hexagonal segments.

    Parameters
    ----------
    n_rings : int
        Number of rings around the centre (2 gives 18 segments plus the
        centre, as on JWST).
    pitch : float
        Centre-to-centre distance of neighbours (segment flat-to-flat width
        plus the gap).
    angle : float, optional
        Vertex direction of the segments as in `hexagon_vertices()`; the
        neighbours lie across the flats, at angle + 30° + k 60°.
    skip_centre : bool, optional
        Leave out the central segment.

    Returns
    -------
    C : (n, 2) array
        Centres, ring by ring.
    """

    a1 = pitch*np.array([np.cos(angle + np.pi/6), np.sin(angle + np.pi/6)])
    a2 = pitch*np.array([np.cos(angle + 5*np.pi/6), np.sin(angle + 5*np.pi/6)])
    out = []
    for ring in range(0 if not skip_centre else 1, n_rings + 1):
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                if max(abs(q), abs(r), abs(q - r)) == ring:
                    out.append(q*a1 + r*a2)
    C = np.array(out).reshape(-1, 2)
    order = np.lexsort((np.arctan2(C[:, 1], C[:, 0]), np.round(np.hypot(*C.T)/pitch, 6)))
    return C[order]


def polygon_mask(x, y, vertices, oversample=4):
    """Antialiased transmission of a convex polygon on a grid.

    Parameters
    ----------
    x, y : arrays
        Linear, equidistant coordinates.
    vertices : (n, 2) array
        Counter-clockwise vertices.
    oversample : int, optional
        Sub-samples per pixel and direction.

    Returns
    -------
    M : array of shape (len(y), len(x))
        Open fraction of each pixel.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    V = np.asarray(vertices, float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    M = np.zeros((len(y), len(x)))
    ix = np.nonzero((x >= V[:, 0].min() - dx) & (x <= V[:, 0].max() + dx))[0]
    iy = np.nonzero((y >= V[:, 1].min() - dy) & (y <= V[:, 1].max() + dy))[0]
    if len(ix) == 0 or len(iy) == 0:
        return M
    xs, ys = x[ix], y[iy]
    s = (np.arange(oversample) + 0.5)/oversample - 0.5
    E = np.roll(V, -1, axis=0) - V
    acc = np.zeros((len(ys), len(xs)))
    for a in s:
        for b in s:
            X, Y = np.meshgrid(xs + a*dx, ys + b*dy)
            inside = np.ones(X.shape, bool)
            for (vx, vy), (ex, ey) in zip(V, E):
                inside &= ex*(Y - vy) - ey*(X - vx) >= 0
            acc += inside
    M[np.ix_(iy, ix)] = acc/oversample**2
    return M


def segment_masks(x, y, centres, flat, angle=0.0, oversample=4):
    """Stack of hexagonal segment masks.

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates.
    centres : (n, 2) array
        Segment centres.
    flat : float
        Flat-to-flat width of a segment.
    angle : float, optional
        Vertex direction, see `hexagon_vertices()`.
    oversample : int, optional

    Returns
    -------
    S : array of shape (n, len(y), len(x))
    """

    return np.array([polygon_mask(x, y, hexagon_vertices(c, flat, angle), oversample)
                     for c in centres])


def strut_mask(x, y, angles, width, length, oversample=4):
    """Transmission of straight struts (spider vanes) from the centre.

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates.
    angles : sequence of float
        Directions of the struts from the x axis (radians).
    width : float
        Width of each strut.
    length : float
        Length from the centre.
    oversample : int, optional

    Returns
    -------
    T : array
        1 in the open pupil, 0 under the struts.
    """

    T = np.ones((len(y), len(x)))
    for a in angles:
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])*width/2
        V = np.array([-n, -n + length*d, n + length*d, n])       # counter-clockwise
        T *= 1 - polygon_mask(x, y, V, oversample)
    return T


SegmentedPupil = namedtuple("SegmentedPupil", ["masks", "centres", "flat", "angle", "struts",
                                               "pupil"])
SegmentedPupil.__doc__ = """A segmented aperture on a grid.

Attributes
----------
masks : (n, ny, nx) array
    Segment masks (without struts).
centres : (n, 2) array
    Segment centres.
flat, angle : float
    Flat-to-flat width and vertex direction of the segments.
struts : (ny, nx) array
    Transmission of the struts (all ones without struts).
pupil : (ny, nx) array
    The whole aperture, masks.sum(0)*struts.
"""


def jwst_like_pupil(x, y, flat=1.32, gap=0.007, n_rings=2, strut_width=0.1,
                    strut_angles=(np.pi/2, -2*np.pi/3, -np.pi/3), oversample=4):
    """A JWST-like aperture: 18 hexagonal segments in two rings, no central
    segment, flat tops, and three secondary mirror struts.

    The struts point up (along +y) and down at 30° on both sides of -y. The
    two lower ones are perpendicular to edges of the segments, so their
    diffraction spikes fall on top of the six spikes of the hexagons; the
    upper one adds the two horizontal spikes (the 6 + 2 pattern).

    Parameters
    ----------
    x, y : arrays
        Pupil coordinates in metres.
    flat : float, optional
        Flat-to-flat width of a segment (m).
    gap : float, optional
        Gap between segments (m).
    n_rings : int, optional
        Rings of segments around the empty centre.
    strut_width : float, optional
        Width of the struts (m); 0 for none.
    strut_angles : sequence, optional
        Strut directions (radians).
    oversample : int, optional

    Returns
    -------
    SegmentedPupil
    """

    C = hex_centres(n_rings, flat + gap)
    S = segment_masks(x, y, C, flat, 0.0, oversample)
    R = np.max(np.hypot(*C.T)) + flat
    T = strut_mask(x, y, strut_angles, strut_width, R, oversample) if strut_width > 0 \
        else np.ones(S.shape[1:])
    return SegmentedPupil(S, C, flat, 0.0, T, S.sum(0)*T)


def polygon_ft(vertices, kx, ky):
    """Fourier transform of a polygon: integral of exp(-i (kx x + ky y)).

    Parameters
    ----------
    vertices : (n, 2) array
        Counter-clockwise vertices.
    kx, ky : arrays
        Angular spatial frequencies (rad per length unit), broadcast.

    Returns
    -------
    F : complex array
        The transform; at k = 0 the area.
    """

    V = np.asarray(vertices, float)
    kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    k2 = kx**2 + ky**2
    F = np.zeros(kx.shape, complex)
    for j in range(len(V)):
        a, b = V[j], V[(j + 1) % len(V)]
        e = b - a
        L = np.hypot(*e)
        n = np.array([e[1], -e[0]])/L       # outward for counter-clockwise
        m = (a + b)/2
        ke = (kx*e[0] + ky*e[1])/2
        F += 1j*(kx*n[0] + ky*n[1])*L*np.exp(-1j*(kx*m[0] + ky*m[1]))*np.sinc(ke/np.pi)
    small = k2 < 1e-30
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.asarray(F/k2)
    if np.any(small):
        area = 0.5*np.sum(V[:, 0]*np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1)*V[:, 1])
        F[small] = area
    return F


def array_factor(centres, kx, ky):
    """Array factor sum_j exp(-i k.c_j) of segments at the centres."""

    C = np.asarray(centres, float)
    kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    return np.sum(np.exp(-1j*(kx[..., None]*C[:, 0] + ky[..., None]*C[:, 1])), axis=-1)


def hexagon_rms_radius(flat):
    """RMS distance from the centre over a regular hexagon,
    sqrt(5/12) s with the side s = flat/sqrt(3)."""

    return np.sqrt(5/12)*flat/np.sqrt(3)


def segment_field(pupil, x, y, wavelength, piston=None, tip=None, tilt=None, phase=None):
    """Pupil field of a segmented aperture with segment errors.

    Parameters
    ----------
    pupil : SegmentedPupil
    x, y : arrays
        Pupil coordinates (m).
    wavelength : float
        Wavelength (m).
    piston : array, optional
        Wavefront piston of each segment (m).
    tip, tilt : arrays, optional
        Wavefront slope of each segment in x and y (radians); the image of
        the segment moves by this angle.
    phase : array, optional
        Additional phase map (radians) common to all segments.

    Returns
    -------
    E : complex array of shape (len(y), len(x))
    """

    n = len(pupil.centres)
    piston = np.zeros(n) if piston is None else np.asarray(piston, float)
    tip = np.zeros(n) if tip is None else np.asarray(tip, float)
    tilt = np.zeros(n) if tilt is None else np.asarray(tilt, float)
    X, Y = np.meshgrid(x, y)
    k = 2*np.pi/wavelength
    E = np.zeros(X.shape, complex)
    for S, c, p, a, b in zip(pupil.masks, pupil.centres, piston, tip, tilt):
        sel = S > 0
        E[sel] += S[sel]*np.exp(1j*k*(p + a*(X[sel] - c[0]) + b*(Y[sel] - c[1])))
    E *= pupil.struts
    if phase is not None:
        E *= np.exp(1j*phase)
    return E


def far_field(E, x, y, theta_x, theta_y, wavelength):
    """Far field amplitude at angles (radians), by the matrix Fourier
    transform; the intensity on axis of an unaberrated pupil is its area
    squared over wavelength**2 (the normalization of `mft()` divided by
    the wavelength).

    Parameters
    ----------
    E : array
        Pupil field (or a stack).
    x, y : arrays
        Pupil coordinates (m).
    theta_x, theta_y : arrays
        Angles (radians).
    wavelength : float

    Returns
    -------
    F : complex array (..., len(theta_y), len(theta_x))
    """

    return mft(E, x, y, np.asarray(theta_x)/wavelength, np.asarray(theta_y)/wavelength)/wavelength


def dispersed_fringes(pupil, x, y, piston, wavelengths, theta, pair=(0, 1), tilt_offset=0.0,
                      n_sub=1):
    """Dispersed fringe sensor: the spectrum of the image of two segments.

    A grism disperses the image of two neighbouring segments (masked out of
    the pupil) perpendicular to their baseline; each wavelength gives a
    one-dimensional cut of the two-segment PSF along the baseline.

    Parameters
    ----------
    pupil : SegmentedPupil
    x, y : arrays
        Pupil coordinates (m).
    piston : float
        Wavefront piston difference, second segment minus first (m).
    wavelengths : array
        Wavelengths (m) along the dispersion.
    theta : array
        Angles along the baseline (radians).
    pair : (int, int), optional
        Segment indices.
    tilt_offset : float, optional
        Common pointing along the baseline (radians).
    n_sub : int, optional
        Wavenumbers averaged per spectral channel. With n_sub > 1 every
        channel integrates over its width in wavenumber, the spacing of
        1/wavelengths (a finite spectral resolution), which reduces the
        fringe visibility by sinc(p dsigma).

    Returns
    -------
    I : array of shape (len(wavelengths), len(theta))
        Intensity, normalized to the peak of an unpistoned pair at each
        wavelength.
    baseline : float
        Centre-to-centre distance (m).
    """

    i, j = pair
    c = pupil.centres
    d = c[j] - c[i]
    B = np.hypot(*d)
    u = d/B
    X, Y = np.meshgrid(x, y)
    S1 = pupil.masks[i]*pupil.struts
    S2 = pupil.masks[j]*pupil.struts
    sel = (S1 + S2) > 0
    Xs, Ys, A1, A2 = X[sel], Y[sel], S1[sel], S2[sel]
    # cut along the baseline through the image centre
    t = np.asarray(theta, float) + tilt_offset
    proj = np.outer(t, u[0]*Xs + u[1]*Ys)
    F0 = np.sum(A1 + A2)
    sig = 1/np.asarray(wavelengths, float)
    d_sig = np.abs(np.gradient(sig)) if len(sig) > 1 else np.zeros(1)
    out = []
    for s0, ds in zip(sig, d_sig):
        subs = s0 + ds*((np.arange(n_sub) + 0.5)/n_sub - 0.5) if n_sub > 1 else [s0]
        acc = 0.0
        for s_ in subs:
            E = A1 + A2*np.exp(2j*np.pi*piston*s_)
            F = np.exp(-2j*np.pi*proj*s_) @ E
            acc = acc + np.abs(F)**2/F0**2
        out.append(acc/len(subs))
    return np.array(out), B


def dfs_piston(I, wavelengths, theta, baseline, p_max, n=4001):
    """Piston difference from a dispersed fringe pattern.

    The fringes vary as cos(2 pi (p - B theta)/lambda); the piston p is
    found by correlating the mean-subtracted pattern with
    exp(-2 pi i (p - B theta)/lambda) for trial values of p and refining the
    maximum. The sign of p is set by the tilt of the fringes.

    Parameters
    ----------
    I : array (len(wavelengths), len(theta))
    wavelengths, theta : arrays
    baseline : float
    p_max : float
        Search range -p_max...p_max (m). With channels equally spaced by
        dsigma in wavenumber, p and p + 1/dsigma give the same pattern up to
        a constant phase, so the range should not exceed 1/(2 dsigma).
    n : int, optional
        Trial values in the first search.

    Returns
    -------
    p : float
        Estimated piston difference (m).
    """

    sig = 1/np.asarray(wavelengths)
    J = I - I.mean(axis=1, keepdims=True)
    Bt = -baseline*np.asarray(theta)

    def score(p):
        return np.abs(np.sum(J*np.exp(-2j*np.pi*np.outer(sig, p + Bt))))

    ps = np.linspace(-p_max, p_max, n)
    s = np.array([score(p) for p in ps])
    k = int(np.argmax(s))
    lo, hi = ps[max(k - 1, 0)], ps[min(k + 1, n - 1)]
    for _ in range(40):                    # golden section refinement
        a = hi - 0.618*(hi - lo)
        b = lo + 0.618*(hi - lo)
        if score(a) > score(b):
            hi = b
        else:
            lo = a
    return (lo + hi)/2


def diversity_psfs(E, defocus_waves, D, x, y, pad=2):
    """PSFs of a pupil field with added defocus (weak lenses), by FFT.

    The defocus phase is 2 pi a (2 rho^2 - 1) with rho = r/(D/2), i.e. a
    waves of the Fringe Zernike Z4 (peak-to-valley 2a waves).

    Parameters
    ----------
    E : array (n, n)
        Pupil field.
    defocus_waves : sequence of float
        Defocus coefficients a.
    D : float
        Diameter that normalizes rho.
    x, y : arrays
        Pupil coordinates.
    pad : int, optional
        Zero padding factor (2: Nyquist sampled images).

    Returns
    -------
    psfs : array (len(defocus_waves), pad n, pad n)
        Intensities, each normalized to the total power.
    phases : array (len(defocus_waves), n, n)
        The diversity phases.
    """

    X, Y = np.meshgrid(x, y)
    rho2 = (X**2 + Y**2)/(D/2)**2
    phases = np.array([2*np.pi*a*(2*rho2 - 1) for a in defocus_waves])
    N = pad*E.shape[0]
    out = []
    for ph in phases:
        F = np.fft.fft2(E*np.exp(1j*ph), s=(N, N))
        I = np.abs(F)**2
        out.append(I/I.sum())
    return np.array(out), phases


def misell(psfs, phases, amplitude, n_iter=200, start=None):
    """Phase retrieval from defocused PSFs: multi-plane Gerchberg-Saxton
    (Misell) iteration.

    In each plane the current pupil estimate times the known diversity
    phase is transformed to the image, its modulus replaced by the measured
    one and transformed back; the diversity phase is removed, the estimates
    of all planes averaged and the known pupil amplitude imposed.

    Parameters
    ----------
    psfs : array (m, N, N)
        Measured intensities (FFT ordering, as from `diversity_psfs()`).
    phases : array (m, n, n)
        Diversity phases.
    amplitude : array (n, n)
        Known pupil amplitude.
    n_iter : int, optional
    start : array, optional
        Initial phase (radians).

    Returns
    -------
    phase : array (n, n)
        Retrieved pupil phase (radians), piston of the pupil removed.
    error : array
        Relative modulus error per iteration.
    """

    m, N, _ = psfs.shape
    n = amplitude.shape[0]
    mod = np.sqrt(psfs)
    phi = np.zeros(amplitude.shape) if start is None else np.array(start, float)
    G = amplitude*np.exp(1j*phi)
    scale = np.sqrt(np.sum(np.abs(amplitude)**2)*N*N)
    err = []
    for _ in range(n_iter):
        acc = np.zeros(amplitude.shape, complex)
        e = 0.0
        for k in range(m):
            F = np.fft.fft2(G*np.exp(1j*phases[k]), s=(N, N))
            target = mod[k]*scale
            e += np.sum((np.abs(F) - target)**2)/np.sum(target**2)
            F = target*np.exp(1j*np.angle(F))
            g = np.fft.ifft2(F)[:n, :n]
            acc += g*np.exp(-1j*phases[k])
        err.append(np.sqrt(e/m))
        G = amplitude*np.exp(1j*np.angle(acc))
    w = np.abs(amplitude)**2
    phase = np.angle(G*np.exp(-1j*np.angle(np.sum(w*G))))
    return phase, np.array(err)


def fit_segments(phase, pupil, x, y, wavelength, weights=None):
    """Least squares piston and tip-tilt of every segment from a phase map.

    Parameters
    ----------
    phase : array
        Pupil phase (radians).
    pupil : SegmentedPupil
    x, y : arrays
    wavelength : float
    weights : array, optional
        Pixel weights (default: the pupil transmission).

    Returns
    -------
    piston, tip, tilt : arrays
        Wavefront piston (m) and slopes (radians) per segment, in the
        convention of `segment_field()`.
    """

    X, Y = np.meshgrid(x, y)
    W = pupil.pupil if weights is None else weights
    k = 2*np.pi/wavelength
    out = []
    for S, c in zip(pupil.masks, pupil.centres):
        sel = (S*W) > 0.5
        A = np.c_[np.ones(sel.sum()), X[sel] - c[0], Y[sel] - c[1]]
        coef = np.linalg.lstsq(A, phase[sel]/k, rcond=None)[0]
        out.append(coef)
    out = np.array(out)
    return out[:, 0], out[:, 1], out[:, 2]
