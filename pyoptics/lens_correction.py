#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Correction of distortion, lateral colour and lens shading from a traced
lens, as an image signal processor (ISP) does it.

A camera lens maps the field angle theta onto the image height h(theta)
instead of the rectilinear f tan(theta) (distortion), the map is slightly
different for every wavelength and so for every colour channel (lateral
colour), and the irradiance falls towards the corners (relative
illumination: vignetting, the cos^4 law and its modification by
distortion and pupil aberrations). On the sensor the pixels accept the
light less well at large chief ray angles, and the interference IR cut
filter moves its edge to shorter wavelengths at oblique incidence, which
makes the shading different for the colour channels (colour shading).
This module derives all these maps from a `raytrace.System` and applies
them to images, for notebook 161 (distortion, lateral colour and lens shading correction):

Geometry

- `radial_mapping()`: real chief ray image heights h(theta) for several
  wavelengths and the rectilinear heights f tan(theta);
  `channel_heights()` weights them with the channel sensitivities.
- `brown_radial()`, `fit_brown_radial()`: the radial polynomial
  distortion model of Brown and Conrady (also that of OpenCV and of the
  Android camera API), r_d = r_u (1 + k_1 r_u^2 + k_2 r_u^4 + ...).
- `warp_coordinates()`: for every pixel of the corrected (rectilinear)
  image, the position on the sensor to sample, from a radial mapping;
  `remap()` resamples with a spline of the given order.

Shading

- `ir_cut_transmission()`: an interference IR cut filter whose edge
  moves to shorter wavelengths at oblique incidence,
  lambda_c(theta) = lambda_c(0) sqrt(1 - sin^2(theta)/n_eff^2), averaged
  over the cone of the light from the exit pupil (`cone_angles()`).
- `channel_shading()`: shading of the R, G and B pixels at the fields of
  a lens: relative illumination (`photography.relative_illumination()`),
  pixel acceptance against the chief ray angle (`sensor.trace_pixel()`),
  and the IR cut filter at its angles of incidence.
- `radial_map()`: a function of the image height evaluated on a pixel grid;
  `gain_table()` and `apply_gain_table()`: the coarse grid of gains of an
  ISP's lens shading correction, bilinearly interpolated.

Lengths in mm, angles in radians, wavelengths in mm unless the name says
nm. Images are arrays (n_y, n_x) with the row index along y; the optical
axis is at the centre of the sensor.

References
----------
D. C. Brown, "Decentering distortion of lenses", Photogramm. Eng. 32,
444-462 (1966).

Android Open Source Project, `CameraCharacteristics.LENS_DISTORTION` and
`CaptureRequest.SHADING_MODE` (developer.android.com).

H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., CRC Press (2010),
for the angle shift of interference filters.
"""

import numpy as np

from . import photography as _ph
from . import sensor as _se

__all__ = ["radial_mapping", "channel_heights", "brown_radial", "fit_brown_radial",
           "invert_radial", "warp_coordinates", "remap", "cone_angles", "ir_cut_transmission",
           "chief_ray_angles", "pixel_efficiency", "channel_shading", "radial_map", "gain_table",
           "apply_gain_table"]


# ---------------------------------------------------------------------------
# geometry

def radial_mapping(system, fields, wavelengths=None):
    """Real chief ray image heights against the field.

    Parameters
    ----------
    system : raytrace.System
        Object at infinity: fields are angles.
    fields : array
        Field angles (y) [rad].
    wavelengths : sequence, optional
        Default: the wavelength of the system.

    Returns
    -------
    h : array (n_wavelengths, n_fields)
        Image heights (y) [mm].
    ideal : array (n_fields,)
        Rectilinear heights efl tan(field) of the paraxial focal length at
        the system wavelength.
    """

    wavelengths = [system.wavelength] if wavelengths is None else wavelengths
    fields = np.asarray(fields, dtype=float)
    h = []
    for wl in wavelengths:
        s = system.copy()
        s.wavelength = wl
        h.append(_ph.chief_ray_mapping(s, fields)[0])
    efl = system.paraxial().efl

    return np.array(h), efl*np.tan(fields)


def channel_heights(h, sensitivity):
    """Image heights of the colour channels: the heights of the wavelengths
    weighted with the channel sensitivities.

    Parameters
    ----------
    h : array (n_wavelengths, n_fields)
    sensitivity : array (n_channels, n_wavelengths)

    Returns
    -------
    array (n_channels, n_fields)
    """

    w = np.asarray(sensitivity, dtype=float)
    w = w/w.sum(axis=1, keepdims=True)

    return w @ np.asarray(h)


def brown_radial(r, k):
    """Radial distortion model r_d = r (1 + k_1 r^2 + k_2 r^4 + ...).

    Parameters
    ----------
    r : array
        Undistorted (normalized) radii.
    k : sequence
        k_1, k_2, ...

    Returns
    -------
    array
        Distorted radii.
    """

    r = np.asarray(r, dtype=float)

    return r*(1 + sum(kj*r**(2*(j + 1)) for j, kj in enumerate(k)))


def fit_brown_radial(r_ideal, r_real, n_terms=3):
    """Least squares fit of `brown_radial()` to a radial mapping.

    Parameters
    ----------
    r_ideal, r_real : arrays
        Undistorted and distorted radii (same normalization).
    n_terms : int, optional

    Returns
    -------
    k : array (n_terms,)
    residual : array
        r_real - brown_radial(r_ideal, k).
    """

    r = np.asarray(r_ideal, dtype=float)
    A = np.column_stack([r**(2*j + 3) for j in range(n_terms)])
    k = np.linalg.lstsq(A, np.asarray(r_real) - r, rcond=None)[0]

    return k, np.asarray(r_real) - brown_radial(r, k)


def invert_radial(r, r_from, r_to):
    """Interpolate a monotonic radial mapping r_from -> r_to at r.

    Parameters
    ----------
    r : array
    r_from, r_to : arrays
        Samples of the mapping (r_from increasing).

    Returns
    -------
    array
    """

    return np.interp(r, r_from, r_to)


def warp_coordinates(shape, pitch, r_ideal, r_sensor, out_pitch=None, centre=None):
    """Sensor pixel coordinates to sample for every pixel of the corrected
    image.

    The corrected image has pixels of `out_pitch` on a rectilinear grid
    centred on the axis; a pixel at the radius r_u (mm) shows the scene
    point that the lens images at r_sensor(r_u) on the sensor, along the
    same azimuth.

    Parameters
    ----------
    shape : (n_y, n_x)
        Shape of the corrected image (and of the sensor image, if
        out_pitch is the pitch).
    pitch : number
        Sensor pixel pitch [mm].
    r_ideal, r_sensor : arrays
        Samples of the mapping (rectilinear height -> sensor height) [mm].
    out_pitch : number, optional
        Pixel pitch of the corrected image (default: pitch).
    centre : (row, col), optional
        Position of the axis in pixels, on the sensor and in the corrected
        image alike (default: the centre, (n_y - 1)/2, (n_x - 1)/2). With
        it, a crop away from the axis is corrected on its own.

    Returns
    -------
    rows, cols : arrays (n_y, n_x)
        Fractional sensor pixel coordinates for `remap()`.
    """

    ny, nx = shape
    out_pitch = pitch if out_pitch is None else out_pitch
    cy, cx = ((ny - 1)/2, (nx - 1)/2) if centre is None else centre
    y = (np.arange(ny) - cy)*out_pitch
    x = (np.arange(nx) - cx)*out_pitch
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    rs = invert_radial(r, r_ideal, r_sensor)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(r > 0, rs/r, (r_sensor[1] - r_sensor[0])/(r_ideal[1] - r_ideal[0]))

    return cy + Y*m/pitch, cx + X*m/pitch


def remap(image, rows, cols, order=3, cval=0.0):
    """Resample an image at fractional pixel coordinates.

    Parameters
    ----------
    image : array (..., n_y, n_x)
        Leading axes (channels) are resampled with the same coordinates.
    rows, cols : arrays
    order : int, optional
        Spline order: 0 nearest, 1 bilinear, 3 cubic spline.
    cval : number, optional
        Value outside the image.

    Returns
    -------
    array (..., rows.shape)
    """

    from scipy.ndimage import map_coordinates

    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        return map_coordinates(image, [rows, cols], order=order, mode="constant", cval=cval)

    return np.array([remap(im, rows, cols, order, cval) for im in image])


# ---------------------------------------------------------------------------
# shading

def cone_angles(cra, f_number, n=2000, rng=0):
    """Angles of incidence (to the normal of a flat filter or sensor) of
    rays filling the cone of a distant exit pupil around the chief ray.

    Parameters
    ----------
    cra : number
        Chief ray angle [rad].
    f_number : number
        Image space NA 1/(2N).
    n : int, optional
        Number of rays (uniform over the pupil).
    rng : int or Generator, optional

    Returns
    -------
    array (n,)
    """

    rng = np.random.default_rng(rng)
    NA = 1/(2*f_number)
    rr = NA*np.sqrt(rng.random(n))
    ph = 2*np.pi*rng.random(n)
    a = np.sin(cra) + rr*np.cos(ph)
    b = rr*np.sin(ph)

    return np.arcsin(np.clip(np.hypot(a, b), 0, 1))


def ir_cut_transmission(wavelength_nm, aoi=0.0, cut_nm=650.0, width_nm=10.0, n_eff=1.8):
    """Transmission of an interference IR cut filter: a logistic edge at
    cut_nm for normal incidence, moving to shorter wavelengths as
    cut_nm sqrt(1 - sin^2(aoi)/n_eff^2) at the angle of incidence aoi;
    averaged over aoi when it is an array.

    Parameters
    ----------
    wavelength_nm : array
    aoi : number or array, optional
        Angle(s) of incidence [rad]; an array (e.g. from `cone_angles()`) is
        averaged over.
    cut_nm, width_nm : number, optional
    n_eff : number, optional
        Effective index of the multilayer.

    Returns
    -------
    array, shape of wavelength_nm
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    aoi = np.atleast_1d(np.asarray(aoi, dtype=float))
    lc = cut_nm*np.sqrt(1 - np.sin(aoi)**2/n_eff**2)
    T = 1/(1 + np.exp((lam[..., None] - lc)/width_nm))

    return T.mean(axis=-1)


def chief_ray_angles(system, fields):
    """Angles of the real chief rays with the axis in image space.

    Parameters
    ----------
    system : raytrace.System
    fields : array
        y fields.

    Returns
    -------
    array [rad]
    """

    from .lens_optimization import _aimed_chief

    D = _aimed_chief(system, np.asarray(fields, dtype=float))[2]

    return np.arctan2(np.hypot(D[:, 0], D[:, 1]), D[:, 2])


def pixel_efficiency(stack, cra, f_number, shifts=None, n_rays=20000, rng=0):
    """Fraction of the light on a pixel that reaches its photodiode, for
    chief ray angles cra and microlens shifts (`sensor.trace_pixel()`).

    Parameters
    ----------
    stack : sensor.PixelStack
    cra : array
        Chief ray angles [rad].
    f_number : number
    shifts : array, optional
        Microlens shift for every angle (default: the shift of the stack).
    n_rays : int, optional
    rng : int, optional
        Seed (the same rays for every angle).

    Returns
    -------
    array
    """

    cra = np.atleast_1d(cra)
    shifts = np.full(len(cra), stack.shift) if shifts is None else np.asarray(shifts)

    return np.array([_se.trace_pixel(stack._replace(shift=s), c, f_number, n_rays,
                                     np.random.default_rng(rng))[0] for c, s in zip(cra, shifts)])


def channel_shading(system, fields, wavelength_nm, sensitivity, stack=None, shifts=None,
                    ir_cut=None, n_rays=20000, rng=0):
    """Shading of the colour channels against the field, relative to the
    axis.

    The product of three factors: the relative illumination of the lens on
    the sensor (`photography.relative_illumination()` with the image side
    factor: vignetting, the obliquity of pupil and image, distortion), the
    acceptance of the pixel stack at the chief ray angle (if a stack is
    given) and the spectral weight of each channel through the IR cut
    filter at the angles of the cone of light (if `ir_cut` is given).

    Parameters
    ----------
    system : raytrace.System
    fields : array
        y fields (> 0).
    wavelength_nm : array
    sensitivity : array (n_channels, n_wavelengths)
        Channel sensitivities without the IR cut filter.
    stack : sensor.PixelStack, optional
    shifts : array, optional
        Microlens shifts at the fields.
    ir_cut : dict, optional
        Keyword arguments of `ir_cut_transmission()` (cut_nm, width_nm,
        n_eff); None for no angle dependent filter.
    n_rays : int, optional
    rng : int, optional

    Returns
    -------
    shading : array (n_channels, n_fields)
    parts : dict
        "illumination" (n_fields,), "pixel" (n_fields,), "filter"
        (n_channels, n_fields) and "cra" (n_fields,) [rad].
    """

    fields = np.asarray(fields, dtype=float)
    S = np.asarray(sensitivity, dtype=float)
    ri = _ph.relative_illumination(system, fields, n_rays=n_rays, rng=np.random.default_rng(rng),
                                   image_side=True)
    cra = chief_ray_angles(system, fields)
    N = system.paraxial().efl/(2*system.paraxial().ep_radius)
    if stack is not None:
        eta = pixel_efficiency(stack, cra, N, shifts, n_rays=n_rays, rng=rng)
        eta0 = pixel_efficiency(stack, [0.0], N, None if shifts is None else [0.0], n_rays=n_rays, rng=rng)[0]
        pix = eta/eta0
    else:
        pix = np.ones(len(fields))
    if ir_cut is not None:
        w0 = S @ ir_cut_transmission(wavelength_nm, cone_angles(0.0, N), **ir_cut)
        filt = np.array([S @ ir_cut_transmission(wavelength_nm, cone_angles(c, N), **ir_cut)
                         for c in cra]).T/w0[:, None]
    else:
        filt = np.ones((len(S), len(fields)))

    return ri*pix*filt, {"illumination": ri, "pixel": pix, "filter": filt, "cra": cra}


def radial_map(shape, pitch, heights, values, centre=None):
    """A function of the image height on a pixel grid (linear interpolation,
    held constant beyond the last height).

    Parameters
    ----------
    shape : (n_y, n_x)
    pitch : number
    heights : array
        Increasing image heights [mm] (start at 0 for a value on the axis).
    values : array (..., n_heights)
    centre : (row, col), optional

    Returns
    -------
    array (..., n_y, n_x)
    """

    ny, nx = shape
    cy, cx = ((ny - 1)/2, (nx - 1)/2) if centre is None else centre
    yy, xx = np.indices(shape)
    r = np.hypot((yy - cy)*pitch, (xx - cx)*pitch)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(r, heights, values)

    return np.array([radial_map(shape, pitch, heights, v, centre) for v in values])


def gain_table(gain, grid=(13, 17)):
    """Coarse table of a gain map, sampled at grid points that include the
    corners (as the lens shading tables of ISPs).

    Parameters
    ----------
    gain : array (..., n_y, n_x)
    grid : (rows, cols), optional

    Returns
    -------
    array (..., rows, cols)
    """

    ny, nx = gain.shape[-2:]
    iy = np.round(np.linspace(0, ny - 1, grid[0])).astype(int)
    ix = np.round(np.linspace(0, nx - 1, grid[1])).astype(int)

    return gain[..., iy[:, None], ix[None, :]]


def apply_gain_table(image, table):
    """Multiply an image with the bilinear interpolation of a gain table
    (`gain_table()`) over its full size.

    Parameters
    ----------
    image : array (..., n_y, n_x)
    table : array (..., rows, cols)

    Returns
    -------
    image : array
    gain : array
        The interpolated gain.
    """

    ny, nx = np.shape(image)[-2:]
    rows, cols = table.shape[-2:]
    gy = np.linspace(0, rows - 1, ny)
    gx = np.linspace(0, cols - 1, nx)
    Gy, Gx = np.meshgrid(gy, gx, indexing="ij")
    from scipy.ndimage import map_coordinates

    if table.ndim == 2:
        g = map_coordinates(table, [Gy, Gx], order=1)
    else:
        g = np.array([map_coordinates(t, [Gy, Gx], order=1) for t in table])

    return np.asarray(image)*g, g
