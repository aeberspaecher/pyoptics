"""Image sensors: what happens to the image of a lens between the cover glass
and the pixel values (notebook 93).

A sensor samples the irradiance in the image plane, but it does not sample
it at points. Each pixel integrates over its light sensitive area (the
pixel aperture), most sensors put a mosaic of colour filters in front of
the pixels (the Bayer pattern), many add an optical low pass filter
(OLPF) of birefringent plates, and every pixel has a microlens that funnels
the light onto a photodiode smaller than the pixel. Each of these shapes
the image:

- the pixel aperture is a box filter, with the MTF sinc(f a) of an
  aperture of width a; frequencies above the Nyquist frequency 1/(2 p) of
  the pixel pitch p fold back as aliases (`pixel_mtf()`,
  `diffraction_mtf()`, `sensor_image()`), and the slanted edge method of
  ISO 12233 measures the MTF of the sampled system below and above
  Nyquist (`slanted_edge_mtf()`),
- the Bayer mosaic samples red and blue at twice the pitch and green on a
  quincunx; colour information is carried on the carriers (1/2p, 0),
  (0, 1/2p) and (1/2p, 1/2p), where fine luminance detail aliases into false
  colour (`bayer_masks()`, `mosaic()`, `demosaic_bilinear()`,
  `demosaic_malvar()`, `cfa_carriers()`, `false_colour()`); a zone plate
  `zone_plate()` maps every spatial frequency to a place in the image
  (`zone_plate_position()`),
- a birefringent plate with its optic axis tilted to the plate normal
  displaces the extraordinary ray sideways (walk-off, `walkoff_angle()`,
  `poynting_direction()`); two plates with a quarter wave plate between
  them split a point into four, a low pass filter with the MTF
  abs(cos(pi f_x d) cos(pi f_y d)) and zeros at 1/(2 d) (`OLPFPlate`,
  `WavePlate`, `olpf_spots()`, `olpf_otf()`, `olpf_thickness()`),
- a microlens images the exit pupil of the lens onto the photodiode; as
  the chief ray angle grows, the image walks off the photodiode, the pixel
  loses light and some of it lands in the neighbour (crosstalk); shifting
  the microlenses towards the centre of the frame compensates for one
  exit pupil distance (`PixelStack`, `pixel_stack()`, `trace_pixel()`,
  `microlens_shift()`, `pupil_image()`, `disk_square_overlap()`).

Units: lengths in mm (pitches of a few micrometres are 0.00x mm), spatial
frequencies in cycles per mm (line pairs per mm), angles in radians.

References
----------
G. C. Holst and T. S. Lomheim, "CMOS/CCD Sensors and Camera Systems", 2nd
ed., SPIE Press (2011), chapters 10 (MTF) and 11 (sampling).

ISO 12233:2017, "Photography - Electronic still picture imaging -
Resolution and spatial frequency responses" (slanted edge SFR).

P. D. Burns, "Slanted-edge MTF for digital camera and scanner analysis",
Proc. IS&T PICS Conference, 135-138 (2000).

B. E. Bayer, "Color imaging array", US patent 3,971,065 (1976).

D. Alleysson, S. Süsstrunk and J. Hérault, "Linear demosaicing inspired by
the human visual system", IEEE Trans. Image Process. 14, 439-449 (2005),
on luminance and chrominance carriers of the Bayer mosaic.

H. S. Malvar, L.-W. He and R. Cutler, "High-quality linear interpolation
for demosaicing of Bayer-patterned color images", Proc. IEEE ICASSP 3,
485-488 (2004).

J. E. Greivenkamp, "Color dependent optical prefilter for the suppression
of aliasing artifacts", Appl. Opt. 29, 676-684 (1990).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), section 15.3 (light in uniaxial crystals, walk-off).

P. B. Catrysse and B. A. Wandell, "Optical efficiency of image sensor
pixels", J. Opt. Soc. Am. A 19, 1610-1620 (2002).
"""

from collections import namedtuple

import numpy as np


__all__ = ["pixel_mtf", "diffraction_mtf", "nyquist", "zone_plate", "zone_plate_position",
           "sensor_image", "slanted_edge", "slanted_edge_mtf", "bayer_masks", "mosaic",
           "demosaic_bilinear", "demosaic_malvar", "cfa_carriers", "false_colour",
           "walkoff_angle", "extraordinary_index", "poynting_direction", "olpf_thickness",
           "OLPFPlate", "WavePlate", "olpf_spots", "olpf_otf", "PixelStack", "pixel_stack",
           "trace_pixel", "microlens_shift", "pupil_image", "disk_square_overlap", "QUARTZ", "LITHIUM_NIOBATE"]


# ordinary and extraordinary indices at 550 nm
QUARTZ = (1.5462, 1.5554)
LITHIUM_NIOBATE = (2.3124, 2.2238)


# --- pixels and sampling ---

def nyquist(pitch):
    """Nyquist frequency 1/(2 p) of a pixel pitch p (cycles per mm)."""

    return 0.5/pitch


def pixel_mtf(f, pitch, fill=1.0):
    """MTF of a square pixel aperture along one axis: abs(sinc(f a)) with the
    aperture width a = sqrt(fill) p.

    Parameters
    ----------
    f : array
        Spatial frequency (cycles per mm).
    pitch : number
        Pixel pitch p (mm).
    fill : number, optional
        Fill factor: the fraction of the pixel area that is sensitive (a
        centred square).
    """

    return np.abs(np.sinc(np.asarray(f, dtype=float)*pitch*np.sqrt(fill)))


def diffraction_mtf(f, wavelength, f_number):
    """MTF of an aberration free lens with a round pupil: the overlap of two
    pupil circles, 2/pi (arccos(v) - v sqrt(1 - v**2)) with
    v = f lambda N, zero from the cut-off 1/(lambda N) on.
    """

    v = np.clip(np.abs(np.asarray(f, dtype=float))*wavelength*f_number, 0.0, 1.0)

    return 2/np.pi*(np.arccos(v) - v*np.sqrt(1 - v**2))


def zone_plate(x, y, r0):
    """Zone plate test chart 0.5 (1 + cos(pi r**2/r0**2)): a chirp whose
    local frequency r/r0**2 grows linearly with the distance from the
    centre, so that every spatial frequency (f_x, f_y) appears at the
    place (x, y) = r0**2 (f_x, f_y)."""

    return 0.5*(1 + np.cos(np.pi*(x**2 + y**2)/r0**2))


def zone_plate_position(fx, fy, r0):
    """Where the zone plate `zone_plate()` has the local frequency
    (fx, fy): (x, y) = r0**2 (fx, fy)."""

    return r0**2*np.asarray(fx), r0**2*np.asarray(fy)


def _fine_frequencies(shape, dx):
    fy = np.fft.fftfreq(shape[-2], dx)
    fx = np.fft.fftfreq(shape[-1], dx)

    return fx[None, :], fy[:, None]


def sensor_image(scene, oversample, pitch, wavelength=None, f_number=None, spots=None, fill=1.0):
    """Image of a scene on a sensor: the scene, given on a grid oversample
    times finer than the pixels, is filtered by the diffraction MTF of the
    lens and by an optical low pass filter (both in the Fourier domain, so
    periodic), then integrated over the pixel apertures and sampled.

    Parameters
    ----------
    scene : array (..., n_y*oversample, n_x*oversample)
        Irradiance on the fine grid (any number of leading channels).
    oversample : int
        Fine grid points per pixel and axis.
    pitch : number
        Pixel pitch (mm).
    wavelength, f_number : number, optional
        Lens diffraction (`diffraction_mtf()`, applied radially); none if
        not given.
    spots : array (n, 3), optional
        OLPF spots (dx, dy, weight) from `olpf_spots()`.
    fill : number, optional
        Fill factor: the centred square of side round(sqrt(fill)
        oversample) fine points is integrated.

    Returns
    -------
    array (..., n_y, n_x)
    """

    scene = np.asarray(scene, dtype=float)
    dx = pitch/oversample
    if wavelength is not None or spots is not None:
        fx, fy = _fine_frequencies(scene.shape, dx)
        H = np.ones((len(fy), fx.shape[1]), dtype=complex)
        if wavelength is not None:
            H = H*diffraction_mtf(np.hypot(fx, fy), wavelength, f_number)
        if spots is not None:
            H = H*olpf_otf(fx, fy, spots)
        scene = np.real(np.fft.ifft2(np.fft.fft2(scene)*H))
    ny, nx = scene.shape[-2]//oversample, scene.shape[-1]//oversample
    k = max(1, int(round(np.sqrt(fill)*oversample)))
    s0 = (oversample - k)//2
    blocks = scene.reshape(scene.shape[:-2] + (ny, oversample, nx, oversample))

    return blocks[..., s0:s0 + k, :, s0:s0 + k].mean(axis=(-3, -1))


def slanted_edge(n, angle, oversample=16, contrast=(0.1, 0.9), pitch=1.0, wavelength=None,
                 f_number=None, fill=1.0, pad=2, spots=None):
    """A slanted edge target imaged on a sensor of n x n pixels: dark left,
    bright right, the edge through the centre and tilted by `angle` from the
    pixel columns, sampled with `sensor_image()` from a fine grid padded by
    the factor `pad` (so that the periodic filtering does not reach the
    window), optionally through an OLPF (`olpf_spots()`).

    Returns
    -------
    array (n, n)
    """

    N = pad*n
    u = (np.arange(N*oversample) + 0.5)/oversample - N/2
    X, Y = np.meshgrid(u, u)
    s = X*np.cos(angle) - Y*np.sin(angle)
    lo, hi = contrast
    scene = np.where(s > 0, hi, lo)
    # a straight edge across the whole padded window; the wrap at its border is far away
    img = sensor_image(scene, oversample, pitch, wavelength, f_number, spots=spots, fill=fill)
    i0 = (N - n)//2

    return img[i0:i0 + n, i0:i0 + n]


def slanted_edge_mtf(image, oversampling=4):
    """MTF from the image of a slightly tilted, near vertical edge (the
    slanted edge method of ISO 12233, after Burns): the edge is located in
    every row by the centroid of the row derivative, a line is fitted, all
    pixels are binned by their distance to the line into a 4 times
    oversampled edge spread function, differentiated, windowed and Fourier
    transformed; the box of the bins and the finite difference are
    divided out.

    Parameters
    ----------
    image : array (n_y, n_x)
    oversampling : int, optional
        Bins per pixel.

    Returns
    -------
    f : array
        Frequencies in cycles per pixel, up to 2.
    mtf : array
    angle : float
        Fitted edge angle from the columns (rad).
    """

    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    d = np.diff(img, axis=1)
    xc = np.arange(nx - 1) + 1.0                       # between pixel centres j and j + 1 (centres at j + 0.5)
    w = np.abs(d)
    c = (w*xc).sum(axis=1)/w.sum(axis=1)
    rows = np.arange(ny) + 0.5
    b, a = np.polyfit(rows, c, 1)
    X, Y = np.meshgrid(np.arange(nx) + 0.5, rows)
    s = (X - (a + b*Y))/np.sqrt(1 + b**2)
    dlt = 1.0/oversampling
    half = 0.5*nx/np.sqrt(1 + b**2) - 2
    sel = np.abs(s) < half
    idx = np.floor((s[sel] + half)/dlt).astype(int)
    nb = int(np.ceil(2*half/dlt))
    cnt = np.bincount(idx, minlength=nb)[:nb]
    tot = np.bincount(idx, weights=img[sel], minlength=nb)[:nb]
    good = cnt > 0
    centers = np.arange(nb)
    esf = np.interp(centers, centers[good], tot[good]/cnt[good])
    lsf = np.diff(esf)
    L = len(lsf)
    k0 = (np.arange(L)*np.abs(lsf)).sum()/np.abs(lsf).sum()
    win = np.where(np.abs(np.arange(L) - k0) < L/2,
                   0.54 + 0.46*np.cos(2*np.pi*(np.arange(L) - k0)/L), 0.0)
    F = np.abs(np.fft.rfft(lsf*win))
    f = np.fft.rfftfreq(L, dlt)
    mtf = F/F[0]/np.sinc(f*dlt)**2
    keep = f <= 2.0

    return f[keep], mtf[keep], np.arctan(b)


# --- the Bayer mosaic ---

_PATTERNS = {"RGGB": ((0, 0), (1, 1)), "BGGR": ((1, 1), (0, 0)),
             "GRBG": ((0, 1), (1, 0)), "GBRG": ((1, 0), (0, 1))}


def bayer_masks(shape, pattern="RGGB"):
    """Boolean masks (3, n_y, n_x) of the red, green and blue pixels of a
    Bayer mosaic; pattern names the colours of the top left 2 x 2 cell, row
    by row."""

    (ry, rx), (by, bx) = _PATTERNS[pattern]
    yy, xx = np.indices(shape)
    R = (yy % 2 == ry) & (xx % 2 == rx)
    B = (yy % 2 == by) & (xx % 2 == bx)

    return np.array([R, ~(R | B), B])


def mosaic(rgb, pattern="RGGB"):
    """Raw image of a Bayer sensor: of every pixel, only the channel of its
    filter. rgb has shape (3, n_y, n_x)."""

    rgb = np.asarray(rgb, dtype=float)

    return (rgb*bayer_masks(rgb.shape[1:], pattern)).sum(axis=0)


def _conv(img, k):
    from scipy.ndimage import convolve

    return convolve(img, np.asarray(k, dtype=float), mode="mirror")


def demosaic_bilinear(raw, pattern="RGGB"):
    """Bilinear demosaicing: every missing value is the mean of its nearest
    neighbours of the same colour (two or four)."""

    M = bayer_masks(raw.shape, pattern)
    kg = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])/4
    krb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/4

    return np.array([_conv(raw*M[0], krb), _conv(raw*M[1], kg), _conv(raw*M[2], krb)])


def demosaic_malvar(raw, pattern="RGGB"):
    """Gradient corrected linear demosaicing of Malvar, He and Cutler: the
    bilinear estimate plus a Laplacian of the colour that was measured at
    the pixel, with 5 x 5 kernels."""

    M = bayer_masks(raw.shape, pattern)
    (ry, rx), (by, bx) = _PATTERNS[pattern]
    yy, xx = np.indices(raw.shape)
    g_at_rb = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])/8
    row = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]])/8           # the colour sits left and right
    col = row.T
    diag = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                     [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]])/8
    G = np.where(M[1], raw, _conv(raw, g_at_rb))
    c_row, c_col, c_diag = _conv(raw, row), _conv(raw, col), _conv(raw, diag)
    out = [None, G, None]
    for ch, (cy, cx), other in ((0, (ry, rx), M[2]), (2, (by, bx), M[0])):
        in_row = M[1] & (yy % 2 == cy)          # green pixel in a row of this colour
        in_col = M[1] & (xx % 2 == cx)
        out[ch] = np.where(M[ch], raw, np.where(in_row, c_row, np.where(in_col, c_col,
                                                                        np.where(other, c_diag, 0.0))))

    return np.array(out)


def cfa_carriers(pitch):
    """Spatial frequencies (cycles per mm) of the chrominance carriers of a
    Bayer mosaic: the colour differences are modulated onto (1/2p, 0),
    (0, 1/2p) and (1/2p, 1/2p) (and their mirror images); luminance detail
    near these frequencies is decoded as false colour."""

    q = 0.5/pitch

    return np.array([[q, 0.0], [0.0, q], [q, q]])


def false_colour(rgb):
    """rms deviation of the channels from their mean, per pixel: zero for a
    grey image, so for a grey scene all of it is false colour."""

    rgb = np.asarray(rgb, dtype=float)

    return np.sqrt(((rgb - rgb.mean(axis=0))**2).mean(axis=0))


# --- birefringent low pass filters ---

def extraordinary_index(n_o, n_e, theta):
    """Index of the extraordinary wave whose wave vector makes the angle
    theta with the optic axis: 1/n**2 = cos**2/n_o**2 + sin**2/n_e**2."""

    return 1/np.sqrt(np.cos(theta)**2/n_o**2 + np.sin(theta)**2/n_e**2)


def walkoff_angle(n_o, n_e, theta=np.pi/4):
    """Walk-off angle between the wave vector and the Poynting vector of the
    extraordinary wave, for a wave vector at theta to the optic axis:
    tan(theta_S) = (n_o/n_e)**2 tan(theta), rho = theta - theta_S (> 0:
    the energy flows towards the optic axis, as in a positive crystal like
    quartz; < 0 away from it, as in lithium niobate)."""

    return theta - np.arctan((n_o/n_e)**2*np.tan(theta))


def poynting_direction(n_o, n_e, axis, k):
    """Unit Poynting vector of the extraordinary wave with wave vector
    direction k in a uniaxial crystal with optic axis `axis`, from the
    dielectric tensor (independent of `walkoff_angle()`): D of the e-wave
    lies in the plane of k and the axis, E = eps^-1 D, and
    S ~ E x (k x E)."""

    a = np.asarray(axis, dtype=float)
    a = a/np.linalg.norm(a)
    k = np.asarray(k, dtype=float)
    k = k/np.linalg.norm(k)
    eps = n_o**2*np.eye(3) + (n_e**2 - n_o**2)*np.outer(a, a)
    D = a - (a @ k)*k
    if np.linalg.norm(D) < 1e-12:            # along the optic axis: no walk-off
        return k
    E = np.linalg.solve(eps, D)
    S = np.cross(E, np.cross(k, E))

    return S/np.linalg.norm(S)


def olpf_thickness(d, n_o=QUARTZ[0], n_e=QUARTZ[1], theta=np.pi/4):
    """Plate thickness that displaces the extraordinary ray by d at normal
    incidence, for the optic axis at theta to the plate normal."""

    return abs(d/np.tan(walkoff_angle(n_o, n_e, theta)))


OLPFPlate = namedtuple("OLPFPlate", ["thickness", "azimuth", "n_o", "n_e", "theta"])
OLPFPlate.__new__.__defaults__ = (QUARTZ[0], QUARTZ[1], np.pi/4)
OLPFPlate.__doc__ = """A birefringent plate of an optical low pass filter.

- thickness: plate thickness (mm).
- azimuth: direction in the sensor plane of the optic axis' projection
  (rad from x); the extraordinary ray (polarized along it) is displaced
  along it by thickness tan(rho).
- n_o, n_e: ordinary and extraordinary indices (default quartz).
- theta: angle of the optic axis to the plate normal (default 45 deg,
  close to the largest walk-off).
"""

WavePlate = namedtuple("WavePlate", ["retardance", "azimuth"])
WavePlate.__doc__ = """A wave plate (depolarizer) between OLPF plates.

- retardance: phase delay of the slow axis (rad), pi/2 for a quarter wave
  plate.
- azimuth: direction of the fast axis (rad from x).
"""


def olpf_spots(elements, wavelength=550e-6, tol=1e-9):
    """Spots that a stack of birefringent plates and wave plates makes of
    one ray of unpolarized light.

    Each input polarization (x and y, incoherent) is followed as a set of
    displaced copies with Jones vectors: a plate splits every copy into an
    ordinary part (not displaced) and an extraordinary part polarized along
    the azimuth and displaced along it, with their phases; copies that end
    at the same place add coherently. Copies at different places are added
    in intensity.

    Parameters
    ----------
    elements : sequence of OLPFPlate and WavePlate
    wavelength : number, optional
    tol : number, optional
        Displacements closer than this are the same place (mm).

    Returns
    -------
    array (n, 3)
        Rows (dx, dy, weight), weights summing to 1.
    """

    k0 = 2*np.pi/wavelength
    spots = {}
    for J0 in (np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)):
        comps = [(np.zeros(2), J0)]
        for el in elements:
            u = np.array([np.cos(el.azimuth), np.sin(el.azimuth)])
            v = np.array([-u[1], u[0]])
            new = []
            if isinstance(el, OLPFPlate):
                rho = walkoff_angle(el.n_o, el.n_e, el.theta)
                ne = extraordinary_index(el.n_o, el.n_e, el.theta)
                shift = el.thickness*np.tan(rho)*u
                for off, J in comps:
                    new.append((off + shift, (J @ u)*u*np.exp(1j*k0*ne*el.thickness)))
                    new.append((off, (J @ v)*v*np.exp(1j*k0*el.n_o*el.thickness)))
            else:
                for off, J in comps:
                    new.append((off, (J @ u)*u + np.exp(1j*el.retardance)*(J @ v)*v))
            merged = []
            for off, J in new:
                for m in merged:
                    if np.all(np.abs(m[0] - off) < tol):
                        m[1] += J
                        break
                else:
                    merged.append([off, J.copy()])
            comps = [(m[0], m[1]) for m in merged]
        for off, J in comps:
            key = tuple(np.round(off/tol).astype(np.int64))
            spots[key] = spots.get(key, (off, 0.0))
            spots[key] = (off, spots[key][1] + 0.5*np.vdot(J, J).real)
    out = np.array([[o[0], o[1], w] for o, w in spots.values() if w > 1e-12])

    return out


def olpf_otf(fx, fy, spots):
    """Optical transfer function of an OLPF: the Fourier transform of its
    spots, sum of w exp(-2 pi i (fx dx + fy dy)). For four equal spots on a
    square of side d its modulus is abs(cos(pi fx d) cos(pi fy d))."""

    fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    H = 0.0
    for dx, dy, w in spots:
        H = H + w*np.exp(-2j*np.pi*(fx*dx + fy*dy))

    return H


# --- microlenses and the chief ray angle ---

PixelStack = namedtuple("PixelStack", ["pitch", "lens_radius", "R", "n", "depth", "pd_half_width",
                                       "shift"])
PixelStack.__doc__ = """Geometry of a pixel with a microlens (all in mm).

- pitch: pixel pitch p.
- lens_radius: radius a of the round microlens footprint (p/2: touching
  lenses; the corners are flat).
- R: radius of curvature of the microlens (convex towards the light).
- n: index of the microlens and of the stack below it.
- depth: distance of the photodiode plane below the microlens vertex.
- pd_half_width: half width of the square photodiode (centred in the pixel).
- shift: lateral offset (x) of the microlens from the pixel centre.
"""


def pixel_stack(pitch=0.004, R=None, n=1.55, depth=None, pd_fraction=0.5, shift=0.0):
    """A `PixelStack` with touching round microlenses (a = p/2), by default
    R = 0.7 p and the photodiode, of width pd_fraction p, in the paraxial
    focal plane n R/(n - 1) of the microlens."""

    R = 0.7*pitch if R is None else R
    depth = n*R/(n - 1) if depth is None else depth

    return PixelStack(pitch, 0.5*pitch, R, n, depth, 0.5*pd_fraction*pitch, shift)


def _refract(d, nrm, n1, n2):
    """Snell's law in vector form; nrm points against the incident rays."""

    c = -np.sum(d*nrm, axis=-1)
    r = n1/n2
    k = 1 - r**2*(1 - c**2)
    t = r*d + (r*c - np.sqrt(np.clip(k, 0, None)))[:, None]*nrm

    return t/np.linalg.norm(t, axis=-1)[:, None]


def trace_pixel(stack, cra, f_number, n_rays=20000, rng=None, azimuth=0.0):
    """Ray trace of the light of a lens through one pixel: rays uniformly
    over the cell of the pixel's microlens (and colour filter), their
    directions filling the cone of the exit
    pupil (image space NA 1/(2N), a distant pupil) around the chief ray at
    the angle cra (in the plane at `azimuth` from x); refraction at the
    spherical microlens or at the flat gaps between microlenses, then
    straight to the photodiode plane.

    Returns
    -------
    efficiency : float
        Fraction of the light through the pixel's microlens cell that
        reaches its photodiode.
    crosstalk : float
        Fraction that reaches the photodiodes of the eight neighbours (and
        so passes the colour filter of the wrong pixel).
    hits : array (n, 2)
        Where the rays cross the photodiode plane (x, y).
    """

    rng = np.random.default_rng() if rng is None else rng
    p, a, R, n = stack.pitch, stack.lens_radius, stack.R, stack.n
    x = stack.shift + (rng.random(n_rays) - 0.5)*p      # the cell of this pixel's microlens
    y = (rng.random(n_rays) - 0.5)*p
    NA = 1/(2*f_number)
    rr = NA*np.sqrt(rng.random(n_rays))
    ph = 2*np.pi*rng.random(n_rays)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    al = np.sin(cra)*ca + rr*np.cos(ph)
    be = np.sin(cra)*sa + rr*np.sin(ph)
    d = np.stack([al, be, np.sqrt(1 - al**2 - be**2)], axis=-1)
    h_cap = R - np.sqrt(R**2 - a**2)
    # start above the lens, on the line through (x, y, 0)
    P = np.stack([x, y, np.zeros(n_rays)], axis=-1) - ((h_cap + 1e-6*p)/d[:, 2])[:, None]*d
    # the microlens array is periodic: every ray meets the (shifted) lens of the cell it enters
    cell = np.stack([np.round((x - stack.shift)/p), np.round(y/p)], axis=-1)*p
    C = np.stack([stack.shift + cell[:, 0], cell[:, 1], np.full(n_rays, R)], axis=-1)
    oc = P - C
    bq = np.sum(oc*d, axis=-1)
    cq = np.sum(oc**2, axis=-1) - R**2
    disc = bq**2 - cq
    t = -bq - np.sqrt(np.clip(disc, 0, None))
    Q = P + t[:, None]*d
    on_lens = (disc > 0) & (np.hypot(Q[:, 0] - C[:, 0], Q[:, 1] - C[:, 1]) <= a) & (Q[:, 2] <= h_cap)
    nrm = np.where(on_lens[:, None], (Q - C)/R, np.array([0.0, 0.0, -1.0]))
    tf = (h_cap - P[:, 2])/d[:, 2]
    Q = np.where(on_lens[:, None], Q, P + tf[:, None]*d)
    d2 = _refract(d, nrm, 1.0, n)
    t2 = (stack.depth - Q[:, 2])/d2[:, 2]
    H = Q[:, :2] + t2[:, None]*d2[:, :2]
    w = stack.pd_half_width
    own = (np.abs(H[:, 0]) <= w) & (np.abs(H[:, 1]) <= w)
    gx, gy = np.round(H[:, 0]/p), np.round(H[:, 1]/p)
    nb = ((np.abs(H[:, 0] - gx*p) <= w) & (np.abs(H[:, 1] - gy*p) <= w) & ~own
          & (np.abs(gx) <= 1) & (np.abs(gy) <= 1))

    return own.mean(), nb.mean(), H


def microlens_shift(stack, cra):
    """Shift of the microlens (towards the centre of the frame, so < 0 for
    cra > 0) that puts the chief ray, refracted at the microlens vertex, on
    the centre of the photodiode: -depth tan(arcsin(sin(cra)/n))."""

    return -stack.depth*np.tan(np.arcsin(np.sin(cra)/stack.n))


def pupil_image(stack, cra, f_number):
    """Paraxial image of the exit pupil on the photodiode plane (photodiode
    in the focal plane): a disk of radius f tan(u) centred at
    shift + f tan(cra), with the front focal length f = R/(n - 1) and
    sin(u) = 1/(2N).

    Returns
    -------
    xc, radius : float
    """

    f = stack.R/(stack.n - 1)

    return stack.shift + f*np.tan(cra), f*np.tan(np.arcsin(1/(2*f_number)))


def disk_square_overlap(xc, yc, radius, half_width, n=400):
    """Fraction of a uniformly lit disk (centre (xc, yc), radius) that falls
    into the square abs(x), abs(y) <= half_width (Gauss-Legendre quadrature
    over x of the exact chord lengths)."""

    x0, x1 = max(xc - radius, -half_width), min(xc + radius, half_width)
    if x1 <= x0:
        return 0.0
    g, wg = np.polynomial.legendre.leggauss(n)
    xs = 0.5*(x1 - x0)*g + 0.5*(x1 + x0)
    h = np.sqrt(np.clip(radius**2 - (xs - xc)**2, 0, None))
    L = np.clip(np.minimum(yc + h, half_width) - np.maximum(yc - h, -half_width), 0, None)

    return 0.5*(x1 - x0)*np.sum(wg*L)/(np.pi*radius**2)
