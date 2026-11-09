#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""The light field (plenoptic) camera: a microlens array in the image plane
of a main lens and a sensor one microlens focal length behind it, after
Adelson and Wang (1992) and Ng et al. (2005), as in the Lytro cameras
(notebook 95).

Every microlens images the main lens aperture onto the n x n pixels behind
it, so a pixel records the light that arrives at its microlens (position x
on the microlens array) from one part of the aperture (position u on the
main lens): the camera samples the 4D light field L(x, y, u, v). The
pixels at the same place under every microlens form a sub-aperture view,
the image seen through one part of the aperture; shifting the views
against each other and adding them refocuses the photograph after the
exposure.

Model: a thin main lens of focal length f at f/N focused at an object
distance s_f, so that the microlens array lies in the image plane at
v = f s_f/(s_f - f); thin microlenses of pitch p and focal length f_m,
square and touching; a sensor of n x n square pixels per microlens at the
focal plane of the microlenses. All directions are paraxial. A ray that
leaves the aperture at u and crosses the microlens array at x, inside the
microlens centred at c, reaches the sensor at

    x_s = c + f_m (x - u)/v,

so the microlens images the aperture (reversed and scaled by f_m/v); with
the image side F-numbers matched, f_m/p = v/D, its image just fills the
n x n pixels.

- PlenopticCamera, plenoptic_camera(): the camera.
- pixel_pitch(): the sensor pixels matched to the grid of microlens
  images.
- view_step(): the step between the views on the aperture.
- disparity(), depth_from_disparity(): shift between neighbouring
  sub-aperture views for a point at a given distance, and back.
- render_raw(): the raw sensor image of a scene of textured planes,
  sampled with rays.
- subaperture_views(), raw_from_views(): rearrange the raw image into the
  n x n views and back.
- refocus(), aperture_weights(): shift-and-add refocusing, with sub-pixel
  Fourier shifts, over the views inside the aperture; view_weights(): the
  views that are not cut by the rim of the aperture.
- view_shift(): sub-pixel shift between two images (phase correlation and
  Gauss-Newton refinement);
  disparity_map(): local disparity between views from patches.
- refocused_mtf(), mtf50(): the geometric MTF of a refocused image and its
  50 % frequency.
- microlens_images(): images of the exit pupil of a real lens
  (raytrace.System) behind the microlenses, for one field point.
"""

from collections import namedtuple

import numpy as np

__all__ = ["PlenopticCamera", "plenoptic_camera", "pixel_pitch", "view_step", "disparity", "depth_from_disparity",
           "Layer", "render_raw", "subaperture_views", "raw_from_views", "aperture_weights", "view_weights", "refocus",
           "view_shift", "disparity_map", "refocused_mtf", "mtf50", "microlens_images"]


PlenopticCamera = namedtuple("PlenopticCamera", ["focal_length", "f_number", "focus_distance",
                                                 "image_distance", "aperture", "pitch",
                                                 "microlens_focal_length", "n_pixels",
                                                 "n_lenslets"])
PlenopticCamera.__doc__ = """A plenoptic camera with a thin main lens (lengths in mm).

- focal_length: focal length f of the main lens.
- f_number: F-number N of the main lens, f/D.
- focus_distance: object distance s_f imaged onto the microlens array.
- image_distance: distance v of the microlens array behind the main lens.
- aperture: diameter D of the main lens aperture.
- pitch: pitch p of the (square) microlenses.
- microlens_focal_length: focal length f_m, also the gap between the
  microlens array and the sensor.
- n_pixels: pixels n under every microlens along x and along y.
- n_lenslets: microlenses along x and along y.
"""


def plenoptic_camera(focal_length=50.0, f_number=4.0, focus_distance=1000.0, pitch=0.1,
                     n_pixels=7, n_lenslets=100, microlens_f_number=None):
    """A plenoptic camera.

    Parameters
    ----------
    focal_length, f_number, focus_distance : number
        Main lens (mm).
    pitch : number
        Microlens pitch p (mm).
    n_pixels : int
        Pixels n per microlens along x and along y (pixel pitch p/n).
    n_lenslets : int
        Microlenses along x and along y.
    microlens_f_number : number, optional
        f_m/p. Default: the image side F-number v/D of the main lens, so
        that the microlens images just fill their cells.

    Returns
    -------
    PlenopticCamera
    """

    v = focal_length*focus_distance/(focus_distance - focal_length)
    D = focal_length/f_number
    Nm = v/D if microlens_f_number is None else microlens_f_number

    return PlenopticCamera(focal_length, f_number, focus_distance, v, D, pitch, Nm*pitch,
                           int(n_pixels), int(n_lenslets))


def pixel_pitch(camera):
    """Pitch of the sensor pixels, p (1 + f_m/v)/n: seen from the centre of
    the main lens, the microlens images lie on a grid of pitch
    p (1 + f_m/v), slightly larger than the microlens pitch, and the
    sensor is matched to it so that every image covers n x n pixels (a
    real camera is calibrated instead, see `microlens_images()`)."""

    c = camera

    return c.pitch*(1 + c.microlens_focal_length/c.image_distance)/c.n_pixels


def view_step(camera):
    """Distance between the parts of the aperture that neighbouring views
    see, Delta u = q v/f_m = D (1 + f_m/v)/n for the pixel pitch q of
    `pixel_pitch()` (a pixel width seen through a microlens from the main
    lens)."""

    c = camera

    return pixel_pitch(c)*c.image_distance/c.microlens_focal_length


def disparity(camera, distance):
    """Shift d between neighbouring sub-aperture views (in microlenses) of
    a point at the object distance s: d = (1 - v/v_s) Delta u/p, where v_s
    is the image distance of s and Delta u = D (1 + f_m/v)/n the step
    between views on the aperture (`view_step()`); d is linear in 1/s,
    d = v Delta u (1/s - 1/s_f)/p. View j (pixel j from the centre of each
    microlens, j = -(n-1)/2 ... (n-1)/2) sees the point at x_0 - j d; the
    point is in focus on the microlenses for d = 0, behind it (closer to
    the camera than s_f) for d > 0.
    """

    c = camera
    s = np.asarray(distance, float)
    with np.errstate(divide="ignore"):
        inv_vs = 1/c.focal_length - 1/s

    return (1 - c.image_distance*inv_vs)*view_step(c)/c.pitch


def depth_from_disparity(camera, d):
    """Object distance with the disparity d (inverse of `disparity()`);
    negative values (beyond infinity) where d is too negative."""

    c = camera
    inv_vs = (1 - np.asarray(d, float)*c.pitch/view_step(c))/c.image_distance
    with np.errstate(divide="ignore"):
        return 1/(1/c.focal_length - inv_vs)


Layer = namedtuple("Layer", ["distance", "texture"])
Layer.__doc__ = """A plane of the scene perpendicular to the axis.

- distance: object distance s of the plane (mm).
- texture: callable texture(a, b) of the angular coordinates a = x/s and
  b = y/s (rad, x and y on the plane) returning (radiance, opacity),
  opacity in [0, 1]; the layers in front cover the ones behind.
"""


def _radiance(camera, layers, x, y, u, w):
    """Radiance along the rays through the main lens at (u, w) that cross
    the microlens array at (x, y): the object side ray passes through
    (u, w) and the conjugate point of (x, y) at s_f."""

    c = camera
    m = -c.focus_distance/c.image_distance
    X, Y = m*x, m*y
    out = np.zeros_like(x)
    T = np.ones_like(x)
    for L in sorted(layers, key=lambda L: L.distance):
        t = L.distance/c.focus_distance
        a = (u + (X - u)*t)/L.distance
        b = (w + (Y - w)*t)/L.distance
        val, alpha = L.texture(a, b)
        alpha = np.broadcast_to(alpha, a.shape)
        out = out + T*alpha*val
        T = T*(1 - alpha)

    return out


def render_raw(camera, layers, samples=8, rng=None, neighbours=1, rows=None, chunk=2**20):
    """Raw sensor image of a scene of textured planes.

    The sensor has n n_lenslets pixels along x and along y with the pitch
    `pixel_pitch()`. Every pixel is sampled with `samples` random rays: a
    random point x_s on the pixel and a random point x on a microlens
    (centre c) give the aperture point u = x - (x_s - c) v/f_m; rays that miss the circular aperture carry no
    light. Microlenses up to `neighbours` cells away are included, so the
    microlens images overlap when the main lens is faster than the
    microlenses.

    Parameters
    ----------
    camera : PlenopticCamera
    layers : sequence of Layer
    samples : int, optional
    rng : numpy Generator, optional
    neighbours : int, optional
    rows : (start, stop), optional
        Render only these pixel rows (counted from the top).
    chunk : int, optional
        Rays per batch.

    Returns
    -------
    array (n n_lenslets, n n_lenslets)
        Mean radiance per pixel; 1 for a white plane that fills the
        aperture.
    """

    c = camera
    rng = np.random.default_rng() if rng is None else rng
    n_tot = c.n_pixels*c.n_lenslets
    q = pixel_pitch(c)
    cell = q*c.n_pixels            # pitch of the microlens images
    r0, r1 = (0, n_tot) if rows is None else rows
    iy, ix = np.mgrid[r0:r1, 0:n_tot]
    iy, ix = iy.ravel(), ix.ravel()
    out = np.zeros(iy.size)
    k = c.microlens_focal_length/c.image_distance
    R2 = (c.aperture/2)**2
    half_s = n_tot*q/2
    half = c.n_lenslets*c.pitch/2
    step = max(1, chunk//(samples*(2*neighbours + 1)**2))
    for i0 in range(0, iy.size, step):
        sl = slice(i0, i0 + step)
        py, px = iy[sl], ix[sl]
        acc = np.zeros(py.size)
        for _ in range(samples):
            # x_s on the pixel (x to the right, y up, origin on the axis)
            xs = (px + rng.random(px.size))*q - half_s
            ys = half_s - (py + rng.random(py.size))*q
            lx = np.floor((xs + half_s)/cell)
            ly = np.floor((half_s - ys)/cell)
            for dy in range(-neighbours, neighbours + 1):
                for dx in range(-neighbours, neighbours + 1):
                    cx = (lx + dx + 0.5)*c.pitch - half
                    cy = half - (ly + dy + 0.5)*c.pitch
                    x = cx + (rng.random(px.size) - 0.5)*c.pitch
                    y = cy + (rng.random(px.size) - 0.5)*c.pitch
                    u = x - (xs - cx)/k
                    w = y - (ys - cy)/k
                    ok = u**2 + w**2 <= R2
                    if np.any(ok):
                        acc[ok] += _radiance(c, layers, x[ok], y[ok], u[ok], w[ok])
        out[sl] = acc/samples

    return out.reshape(r1 - r0, n_tot)


def subaperture_views(raw, n_pixels, white=None):
    """Rearrange a raw image into sub-aperture views.

    Parameters
    ----------
    raw : array
    n_pixels : int
    white : array, optional
        Raw image of a white scene (flat field) to divide by, as cameras
        are calibrated; pixels darker than 1e-3 of its maximum become 0.
        Rendered with the same random numbers (the same seed of `rng` in
        `render_raw()`), it also removes the sampling noise of the pixels
        on the rim of the microlens images.

    Returns
    -------
    array (n, n, M, M)
        views[i, j] holds pixel (i, j) under every microlens (row i,
        column j from the top left), i.e. the image seen through one part
        of the aperture. Pixel j = (n-1)/2 + k sees the aperture at
        u = -k Delta u (`view_step()`): the microlens reverses the
        aperture.
    """

    n = int(n_pixels)
    if white is not None:
        ok = white > 1e-3*white.max()
        raw = np.where(ok, raw/np.where(ok, white, 1), 0.0)
    M0, M1 = raw.shape[0]//n, raw.shape[1]//n
    r = raw[:M0*n, :M1*n].reshape(M0, n, M1, n)

    return r.transpose(1, 3, 0, 2).copy()


def raw_from_views(views):
    """Inverse of `subaperture_views()`."""

    n0, n1, M0, M1 = views.shape

    return views.transpose(2, 0, 3, 1).reshape(M0*n0, M1*n1)


def _fourier_shift(img, sy, sx, pad):
    p = np.pad(img, pad, mode="edge")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.fftfreq(p.shape[1])[None, :]
    P = np.fft.ifft2(np.fft.fft2(p)*np.exp(-2j*np.pi*(fy*sy + fx*sx))).real

    return P[pad:pad + img.shape[0], pad:pad + img.shape[1]]


def refocus(views, d, aperture=None):
    """Shift-and-add refocusing (Ng 2005): view (i, j) is shifted by
    (i - c, j - c) d microlenses with c = (n-1)/2 (Fourier shift on an
    edge padded copy) and the views are averaged, which brings the points
    with disparity d (`disparity()`) into focus. d = 0 gives the
    conventional photograph focused on the microlens array.

    Parameters
    ----------
    views : array (n, n, M0, M1)
    d : number
        Disparity of the plane to focus on (microlenses per view).
    aperture : array (n, n) of weights, optional
        Synthetic aperture: weights of the views (default: all equal, but
        views whose mean is below 0.2 of the brightest view are left out,
        such as the dark corners outside the circular aperture).

    Returns
    -------
    array (M0, M1)
    """

    if aperture is None:
        aperture = aperture_weights(views)
    out = 0
    for w, v in _sheared(views, d, aperture):
        out = out + w*v

    return out/aperture.sum()


def view_shift(a, b, iterations=8, margin=None):
    """Shift (dy, dx) of image b relative to image a, in pixels; b(x) =
    a(x - dx) gives dx > 0.

    The whole-pixel shift is the peak of the phase correlation; it is
    refined by Gauss-Newton steps (Lucas and Kanade) that shift b back with
    sub-pixel Fourier shifts and fit the remaining difference with the
    gradient of a, inside a margin (default: a quarter of the size) that
    keeps out the edges.
    """

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a0, b0 = a - a.mean(), b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.conj(np.fft.fft2(a0*w))*np.fft.fft2(b0*w)
    r = np.fft.ifft2(F/(np.abs(F) + 1e-12*np.abs(F).max())).real
    i, j = np.unravel_index(np.argmax(r), r.shape)
    ny, nx = r.shape
    dy = float(i - ny if i > ny/2 else i)
    dx = float(j - nx if j > nx/2 else j)
    m = max(ny, nx)//4 if margin is None else margin
    sl = (slice(m, ny - m), slice(m, nx - m))
    gy, gx = np.gradient(a)
    gy, gx = gy[sl].ravel(), gx[sl].ravel()
    G = np.array([[gy @ gy, gy @ gx], [gx @ gy, gx @ gx]])
    pad = int(np.ceil(max(abs(dy), abs(dx)))) + 4
    for _ in range(iterations):
        bb = _fourier_shift(b, -dy, -dx, pad)
        e = (bb - a)[sl].ravel()
        # bb(x) = a(x - delta) ~ a - grad a . delta
        step = np.linalg.solve(G, -np.array([gy @ e, gx @ e]))
        dy += step[0]
        dx += step[1]
        if np.max(np.abs(step)) < 1e-4:
            break

    return dy, dx


def aperture_weights(views, threshold=0.2):
    """Weights of the views for `refocus()`: 1 for the views whose mean is
    at least `threshold` times that of the brightest view, 0 for the rest
    (such as the dark corners outside the circular aperture)."""

    m = views.mean(axis=(2, 3))
    return (m > threshold*m.max()).astype(float)


def view_weights(white, n_pixels, threshold=0.9):
    """Views that see the whole of their part of the aperture: 1 where the
    white raw image puts at least `threshold` of the light of the
    brightest view into the view, 0 elsewhere. The views on the rim of the
    circular aperture see only a part of their square of the aperture,
    nearer to the centre than the nominal u; leaving them out of
    `disparity_map()` removes the bias they cause."""

    m = subaperture_views(white, n_pixels).mean(axis=(2, 3))

    return (m >= threshold*m.max()).astype(float)


def _sheared(views, d, aperture):
    """The views shifted to bring disparity d into register (generator)."""

    n = views.shape[0]
    cen = (n - 1)/2
    pad = int(np.ceil(abs(d)*cen)) + 4
    for i in range(n):
        for j in range(n):
            if aperture[i, j]:
                yield aperture[i, j], _fourier_shift(views[i, j], (i - cen)*d, (j - cen)*d, pad)


def disparity_map(views, candidates, patch=9, aperture=None):
    """Disparity of every microlens by photo-consistency (after Tao et al.
    2013): for every candidate disparity the views are sheared as in
    `refocus()`; where the candidate is right, all views see the same
    radiance. The variance across the views, normalized by the brightness
    of each view and averaged over patch x patch microlenses, is minimal
    there; a parabola through the minimum and its neighbours refines it.

    Parameters
    ----------
    views : array (n, n, M0, M1)
    candidates : array
        Equally spaced disparities to try (microlenses per view).
    patch : int, optional
    aperture : array (n, n), optional
        Weights of the views, as in `refocus()`.

    Returns
    -------
    d : array (M0, M1)
    cost : array (len(candidates), M0, M1)
    """

    from scipy.ndimage import uniform_filter

    if aperture is None:
        aperture = aperture_weights(views)
    gain = views.mean(axis=(2, 3))
    gain = np.where(gain > 0, gain, 1)
    norm = views/gain[:, :, None, None]*gain.max()
    cost = np.zeros((len(candidates),) + views.shape[2:])
    for k, d in enumerate(candidates):
        s1 = s2 = W = 0
        for w, v in _sheared(norm, d, aperture):
            s1 = s1 + w*v
            s2 = s2 + w*v**2
            W += w
        var = s2/W - (s1/W)**2
        cost[k] = uniform_filter(var, patch, mode="nearest")
    k = np.clip(np.argmin(cost, axis=0), 1, len(candidates) - 2)
    i0, i1 = np.indices(k.shape)
    c0, cm, cp = cost[k, i0, i1], cost[k - 1, i0, i1], cost[k + 1, i0, i1]
    den = cm - 2*c0 + cp
    off = np.where(den > 0, 0.5*(cm - cp)/np.where(den > 0, den, 1), 0.0)
    step = candidates[1] - candidates[0]

    return np.asarray(candidates)[k] + np.clip(off, -1, 1)*step, cost


def refocused_mtf(freq, d, d_focus=None, weights=None):
    """Geometric MTF of a refocused image along x, frequency in cycles per
    microlens.

    Every view integrates over one microlens (a box of width 1) and, since
    each pixel sees a part Delta u of the aperture, over a box of width abs(d)
    for a point with disparity d. Refocused on d_focus, view column j is
    displaced by (j - c)(d - d_focus) microlenses, c = (n-1)/2; the
    columns, weighted by the number of views in them, add the factor
    abs(sum_j w_j exp(2 pi i (j - c)(d - d_focus) f))/sum_j w_j.

    Parameters
    ----------
    freq : array
    d : number
        Disparity of the object.
    d_focus : number, optional
        Disparity refocused on (default: d).
    weights : array (n,), optional
        Views per column of the synthetic aperture (needed when d_focus
        differs from d).

    Returns
    -------
    array
    """

    f = np.asarray(freq, float)
    m = np.abs(np.sinc(f)*np.sinc(d*f))
    if d_focus is not None and d_focus != d:
        w = np.asarray(weights, float)
        j = np.arange(w.size) - (w.size - 1)/2
        m = m*np.abs(np.exp(2j*np.pi*np.multiply.outer(f, j)*(d - d_focus)) @ w)/w.sum()

    return m


def mtf50(freq, mtf):
    """First frequency at which the MTF drops to 0.5 (linear
    interpolation); nan if it never does."""

    f = np.asarray(freq, float)
    m = np.asarray(mtf, float)
    below = np.nonzero(m < 0.5)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]

    return f[i - 1] + (m[i - 1] - 0.5)*(f[i] - f[i - 1])/(m[i - 1] - m[i])


def microlens_images(system, field, pitch, microlens_focal_length, n_rays=4000, rng=None,
                     margin=1.4):
    """Images of the exit pupil of a real lens behind the microlenses.

    Rays from one field point are traced through `system` to its image
    plane, where the microlens array lies; a thin microlens of focal length
    f_m sends a ray with the direction (k_x, k_y, k_z) to x_s = c +
    f_m k_x/k_z on the sensor, c being the centre of the microlens, so the
    distribution of x_s - c is the image of the (vignetted) exit pupil.

    Parameters
    ----------
    system : raytrace.System
    field : number
        Field (see `raytrace.System.rays()`), in the y direction.
    pitch : number
        Microlens pitch (mm), only to say which microlens a ray passes;
        the result does not depend on it, since every microlens maps a
        direction to the same offset from its centre.
    microlens_focal_length : number
    n_rays : int, optional
        Rays on a random set of points of the entrance pupil, out to
        `margin` times its paraxial radius (the stop and the clear
        apertures cut off the rest).
    rng : numpy Generator, optional

    Returns
    -------
    xs, ys : arrays
        Sensor positions relative to the centre of the microlens that the
        ray passes (mm).
    hx, hy : arrays
        Ray positions in the image plane (mm).
    """

    rng = np.random.default_rng() if rng is None else rng
    r = margin*np.sqrt(rng.random(n_rays))
    t = 2*np.pi*rng.random(n_rays)
    rays = system.rays(field, r*np.cos(t), r*np.sin(t))
    b = system.trace(rays)
    v = b.valid
    P = system.to_image_frame(b.points[-1][v])
    k = b.directions[-1][v]
    xs = microlens_focal_length*k[:, 0]/k[:, 2]
    ys = microlens_focal_length*k[:, 1]/k[:, 2]

    return xs, ys, P[:, 0], P[:, 1]
