#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Aberrations of a ray traced projection lens in lithographic imaging:
from the lens design to CD, placement and best focus across the slit, and
the sensitivities of these to Zernike coefficients.

Lens to projector
-----------------
lens_aberrations() traces a pyoptics.raytrace.System at an image field
point and fits Fringe Zernikes to its OPD on the pupil coordinates of
Projector, (P_x, P_y) = n' (d - d_chief)_xy/NA with the image space ray
directions d, so that the coefficients can be handed to Projector directly
(projector_from_system()). Besides the wavefront it returns two properties of
the chief ray that the pupil does not describe:

- distortion: the chief ray image point minus the paraxial image point
  m*object (m: paraxial magnification); in a scanner the uniform part of it
  (magnification and shift) is corrected by the scanner, the rest is an
  overlay error,
- telecentricity: the chief ray direction; an image in a defocused plane
  moves sideways by focus*tan(theta_chief).

The Zernike tilts Z_2, Z_3 are relative to the chief ray and shift the
image on top of the distortion.

Scanner slit
------------
A scanner exposes through a slit (e.g. 26 x 8 mm on the wafer, the long
side along x) while mask and wafer move along y; every point of the wafer
sees all scan positions y of the slit. through_slit() therefore computes the
image at a slit position x as the average of the images at the scan
positions y, each one with its own aberrations and placement. Aberrations
that vary along y blur the image (image fading); along x they give the CD
and placement variation across the slit.

Sensitivities
-------------
zernike_sensitivity() gives the first and second derivatives of CD,
placement and best focus with respect to each Zernike coefficient (central
differences), the classic way to compare imaging conditions and to set lens
tolerances. predict() turns them into the changes expected for a set of
coefficients (linear and diagonal quadratic terms, cross terms neglected).

Polarization
------------
lens_jones_pupil() traces the polarization through the lens: Fresnel
transmission of every surface, optional thin film coatings
(quarter_wave_coating()) and birefringent elements such as CaF2 with its
intrinsic birefringence (CubicCrystal). The result is fitted with
pyoptics.litho.jones_pupil and handed to Projector(jones_pupil=...).

Caldwell's lens is a dry KrF design (NA 0.56). It is the ray traced model
of a projection lens here; it cannot reach the NA 1.35 of ArF immersion.

Units: the system (e.g. caldwell_krf_lens()) is in mm, the lithography
simulation in its own length unit; length_unit is that unit in mm (1e-6
for nm, the default). Field points are image (wafer) coordinates in the
units of the system.
"""

from collections import namedtuple

import numpy as np

from . import metrics
from ..bases import fringe_zernike
from ..geometry import R_z
from .imaging import Projector
from .jones_pupil import decompose


LensAberrations = namedtuple("LensAberrations", ["field", "zernikes", "rms_residual", "distortion",
                                                 "telecentricity", "placement", "NA"])
LensAberrations.__doc__ = """Aberrations of a lens at one image field point.

Attributes
----------
field : (x, y) image field point (units of the system)
zernikes : dict {j: c_j} of Fringe Zernike coefficients in waves (j >= 2)
    on the pupil of Projector
rms_residual : RMS of the OPD left by the fit, in waves
distortion : (dx, dy) chief ray image point minus the paraxial image point,
    in the length unit of the simulation
telecentricity : (t_x, t_y) x and y components of the chief ray direction
    (radians for small angles)
placement : (dx, dy) placement of a small feature, distortion plus the
    shift by the Zernike tilts, -(c_2, c_3)*lambda/NA (first order: the
    centroid of the PSF), in the length unit of the simulation
NA : the NA that normalizes the pupil coordinates
"""

Response = namedtuple("Response", ["cd", "placement", "best_focus", "foci", "cd_focus",
                                   "placement_focus", "nils_focus"])
Response.__doc__ = """Printed feature for the threshold model.

Attributes
----------
cd, placement : CD and placement (center of the printed feature minus its
    nominal center) at focus 0
best_focus : focus of the largest NILS at the printed edges (vertex of a
    parabola through the five foci around the maximum); unlike the vertex
    of a Bossung curve it is well defined at the isofocal dose too
foci, cd_focus, placement_focus, nils_focus : CD, placement and NILS
    through focus
"""

Sensitivity = namedtuple("Sensitivity", ["terms", "delta", "nominal", "linear", "quadratic"])
Sensitivity.__doc__ = """Zernike sensitivities.

Attributes
----------
terms : array of Fringe indices j
delta : step in waves of the central differences
nominal : Response without the steps
linear, quadratic : dicts with the keys "cd", "placement", "best_focus" of
    arrays (one entry per term): first derivatives d/dc_j (length unit per
    wave) and second derivatives d^2/dc_j^2 (length unit per wave^2)
"""

SlitResult = namedtuple("SlitResult", ["x", "y", "cd", "placement", "best_focus", "aberrations"])
SlitResult.__doc__ = """Feature across the slit.

Attributes
----------
x, y : slit positions and scan positions (units of the system)
cd, placement, best_focus : arrays over x, from the scan averaged images
aberrations : LensAberrations, list (over x) of lists (over y)
"""


def _pupil_rays(system, field, N):
    """OPD (waves) and directions of rays on an N x N pupil grid and the
    chief ray, at the object point belonging to the image field point."""

    m = system.paraxial().magnification
    obj = np.asarray(field, dtype=float)/m
    _, PX, PY, inside = system._pupil_grid(N)
    bundle = system.trace_pupil(obj, PX[inside], PY[inside])
    chief = system.trace_pupil(obj, 0.0, 0.0)
    W = system._opd(bundle, chief)/system.wavelength
    d = system._image_hits(bundle)[1]
    I_c, d_c = (v[0] for v in system._image_hits(chief))
    ok = bundle.valid & np.isfinite(W)

    return W[ok], system.n[-1]*(d[ok, :2] - d_c[:2]), I_c, d_c, m*obj


def lens_NA(system, N=48):
    """Image space NA of the lens on axis: the largest ``n'*|d_xy|`` of the
    traced rays."""

    q = _pupil_rays(system, (0.0, 0.0), N)[1]

    return float(np.max(np.hypot(q[:, 0], q[:, 1])))


def lens_aberrations(system, field=(0.0, 0.0), n_terms=36, N=48, NA=None, length_unit=1e-6):
    """Fringe Zernikes, distortion and telecentricity of a lens at an image
    field point.

    Parameters
    ----------
    system : pyoptics.raytrace.System
        Finite conjugate projection lens.
    field : pair, optional
        Image (wafer) field point (x, y) in the units of the system.
    n_terms : int, optional
        Fit Z_1 ... Z_n_terms (Z_1 is dropped from the result).
    N : int, optional
        Rays across the pupil.
    NA : number, optional
        Normalization of the pupil coordinates; rays outside it are not
        used, so a smaller NA than the lens's describes the stopped down
        lens. Default: lens_NA(system).
    length_unit : number, optional
        Length unit of the simulation in units of the system (1e-6: nm
        for a system in mm).

    Returns
    -------
    LensAberrations
    """

    if NA is None:
        NA = lens_NA(system, N)
    W, q, I_c, d_c, ideal = _pupil_rays(system, field, N)
    P = q/NA
    use = np.hypot(P[:, 0], P[:, 1]) <= 1.0
    A = np.column_stack([fringe_zernike(j, P[use, 0], P[use, 1]) for j in range(1, n_terms + 1)])
    c, *_ = np.linalg.lstsq(A, W[use], rcond=None)
    residual = W[use] - A @ c
    zernikes = {j: float(c[j - 1]) for j in range(2, n_terms + 1)}
    distortion = tuple(float(v) for v in (I_c[:2] - ideal)/length_unit)
    telecentricity = (float(d_c[0]), float(d_c[1]))
    wavelength = system.wavelength/length_unit
    placement = (distortion[0] - zernikes.get(2, 0.0)*wavelength/NA,
                 distortion[1] - zernikes.get(3, 0.0)*wavelength/NA)

    return LensAberrations(tuple(float(f) for f in field), zernikes,
                           float(np.sqrt(np.mean(residual**2))), distortion, telecentricity,
                           placement, NA)


def linear_correction(aberrations):
    """The placement errors that a scanner corrects: translation,
    magnification and rotation fitted to the placements of a set of field
    points,

        dx = a_0 + a_1*x + a_2*y,   dy = b_0 + b_1*y + b_2*x

    (magnification along x is set in the lens, along y by the scan speed;
    a_2 and b_2 are the rotations of mask and wafer).

    Parameters
    ----------
    aberrations : sequence of LensAberrations
        Field points spread over the slit (at least three not on a line).

    Returns
    -------
    correction : callable
        correction(x, y) -> (dx, dy), the fitted placement in the length
        unit of the simulation; pass it to through_slit().
    """

    f = np.array([ab.field for ab in aberrations])
    d = np.array([ab.placement for ab in aberrations])
    one = np.ones(len(f))
    a = np.linalg.lstsq(np.column_stack([one, f[:, 0], f[:, 1]]), d[:, 0], rcond=None)[0]
    b = np.linalg.lstsq(np.column_stack([one, f[:, 1], f[:, 0]]), d[:, 1], rcond=None)[0]

    def correction(x, y):
        return a[0] + a[1]*x + a[2]*y, b[0] + b[1]*y + b[2]*x

    return correction


def projector_from_system(system, field=(0.0, 0.0), n_terms=36, N=48, NA=None, length_unit=1e-6,
                          **kwargs):
    """Projector with the NA, wavelength, reduction ratio and Zernikes of a
    ray traced lens at an image field point.

    Parameters
    ----------
    system, field, n_terms, N, NA, length_unit
        As for lens_aberrations().
    kwargs
        Further arguments of Projector (n, vectorial, apodization, ...).

    Returns
    -------
    Projector
    """

    ab = lens_aberrations(system, field, n_terms, N, NA, length_unit)
    kwargs.setdefault("magnification", 1/abs(system.paraxial().magnification))

    return Projector(ab.NA, system.wavelength/length_unit, zernikes=ab.zernikes, **kwargs)


# polarization ray trace:

LensJonesPupil = namedtuple("LensJonesPupil", ["P_x", "P_y", "J", "transmission", "field", "NA"])
LensJonesPupil.__doc__ = """Jones pupil of a ray traced lens.

Attributes
----------
P_x, P_y : (N,) arrays
    Normalized pupil coordinates of the rays (as in lens_aberrations()).
J : (N, 2, 2) array
    Jones matrices from the object space to the image space, both in the
    aplanatic bases of the ray directions (the identity for an ideal lens).
transmission : number
    Mean intensity transmission of the chief ray (that J is normalized to
    if normalize was True).
field, NA : image field point and the NA of the pupil coordinates
"""


def quarter_wave_coating(n_glass, wavelength, n_coating=None):
    """Single layer antireflection coating: a quarter wave layer at normal
    incidence of index n_coating (default sqrt(n_glass), zero reflection;
    MgF2 has about 1.40 at 248 nm and 1.43 at 193 nm).

    Returns
    -------
    list of (n, thickness)
        The layers from the air side, thickness in the units of wavelength,
        for lens_jones_pupil(coating=...).
    """

    n_c = np.sqrt(n_glass) if n_coating is None else n_coating

    return [(n_c, wavelength/(4*n_c))]


def _ray_basis(k, normal):
    """Unit s vectors k x normal (any perpendicular vector at normal
    incidence) and the p vectors k x s, for rows of k."""

    s = np.cross(k, normal)
    small = np.linalg.norm(s, axis=-1) < 1e-12
    s[small] = np.cross(k[small], [1.0, 0.0, 0.0])
    s = s/np.linalg.norm(s, axis=-1)[:, None]

    return s, np.cross(k, s)


def _interface(k1, k2, normal, n1, n2, layers, wavelength):
    """Polarization ray tracing matrices (Chipman's P, shape (N, 3, 3)) of the
    refraction of the rays k1 -> k2 at a surface with the unit normals,
    for the transmitted amplitude: ``|P E|**2`` is the transmitted power fraction
    (Fresnel transmission times n2*cos_t/(n1*cos_i))."""

    from ..thin_films import _r_t

    cos_i = np.abs(np.sum(k1*normal, axis=-1))
    cos_t = np.abs(np.sum(k2*normal, axis=-1))
    sin_i = np.sqrt(np.clip(1 - cos_i**2, 0, None))
    if n1 < n2:  # layers on the side of the lower index
        n_stack, t_stack = [n1] + [n for n, _ in layers] + [n2], [t for _, t in layers]
    else:
        n_stack, t_stack = [n1] + [n for n, _ in layers[::-1]] + [n2], [t for _, t in layers[::-1]]
    n_stack = np.array(n_stack, dtype=complex)
    t_s = _r_t(sin_i, n_stack, t_stack, "TE", wavelength)[1]
    t_p = _r_t(sin_i, n_stack, t_stack, "TM", wavelength)[1]*cos_i/cos_t  # tangential -> full field
    power = np.sqrt(n2*cos_t/(n1*cos_i))
    s, p1 = _ray_basis(k1, normal)
    p2 = np.cross(k2, s)

    return power[:, None, None]*(t_s[:, None, None]*s[:, :, None]*s[:, None, :]
                                 + t_p[:, None, None]*p2[:, :, None]*p1[:, None, :])


class CubicCrystal(object):
    """Intrinsic (spatial dispersion induced) birefringence of a cubic
    crystal such as CaF2, for lens_jones_pupil(birefringent=...).

    Cubic crystals are isotropic to lowest order, but at short wavelengths
    the finite wave vector makes them weakly birefringent (Burnett, Levine
    and Shirley, Phys. Rev. B 64, 241102 (2001)). The lowest order term
    allowed by cubic symmetry changes the index for a ray along the unit
    vector k (crystal axes) by ::

        dn = -2*delta_n*(B^T diag(k_x**2, k_y**2, k_z**2) B - trace/2)

    on its transverse polarization basis B. It vanishes along the <100> and
    <111> axes and is largest along the <110> axes, where delta_n =
    n([001] polarization) - n([1-10] polarization). Burnett et al. measured
    abs(delta_n) = 3.4e-7 (3.4 nm/cm) at 193 nm and 11.2e-7 at 157 nm for
    CaF2, which made it hard to use at 157 nm. The pattern over the pupil
    has the symmetry of the axis along the lens axis (threefold for [111],
    fourfold for [100]), so clocking elements against each other (e.g. two
    [111] elements by 60 degrees) cancels the non rotationally symmetric
    part.

    Parameters
    ----------
    delta_n : number
        Birefringence along <110> (see above), e.g. -3.4e-7.
    axis : {"111", "100", "110"}, optional
        Crystal axis along the lens axis (z).
    clocking : number, optional
        Rotation of the crystal about the lens axis, radians.
    """

    _AXES = {"100": np.eye(3),
             "111": np.array([[1, -1, 0]/np.sqrt(2), [1, 1, -2]/np.sqrt(6), [1, 1, 1]/np.sqrt(3)]),
             "110": np.array([[0, 0, 1.0], [1, -1, 0]/np.sqrt(2), [1, 1, 0]/np.sqrt(2)])}

    def __init__(self, delta_n, axis="111", clocking=0.0):
        if axis not in self._AXES:
            raise ValueError("axis must be one of {}".format(sorted(self._AXES)))
        self.delta_n, self.axis, self.clocking = delta_n, axis, clocking
        # _AXES rows: the lens x, y, z axes in crystal coordinates (unclocked)
        self.to_crystal = self._AXES[axis].T @ R_z(-clocking)

    def __repr__(self):
        return "CubicCrystal(delta_n={}, axis={!r}, clocking={})".format(self.delta_n, self.axis,
                                                                       self.clocking)

    def index_change(self, k, basis):
        """Change of the refractive index (N, 2, 2) on the transverse basis
        (N, 3, 2) of the rays with unit directions k (N, 3), lens frame."""

        k_c = k @ self.to_crystal.T
        B_c = np.einsum("ij,njk->nik", self.to_crystal, basis)
        N = np.einsum("nik,ni,nil->nkl", B_c, k_c**2, B_c)
        N = N - np.trace(N, axis1=1, axis2=2)[:, None, None]/2*np.eye(2)

        return -2*self.delta_n*N

    def propagation(self, k, length, wavelength):
        """Polarization ray tracing matrices (N, 3, 3) of a path of the
        given lengths along the unit directions k (lens frame)."""

        s, p = _ray_basis(k, np.tile([0.0, 0.0, 1.0], (len(k), 1)))
        B = np.stack([s, p], axis=-1)
        w, V = np.linalg.eigh(self.index_change(k, B))
        U = np.einsum("nij,nj,nkj->nik", V, np.exp(2j*np.pi/wavelength*length[:, None]*w), V)

        return np.einsum("nij,njk,nlk->nil", B, U, B) + k[:, :, None]*k[:, None, :]


def lens_jones_pupil(system, field=(0.0, 0.0), N=32, NA=None, coating=None, birefringent=None,
                     normalize=True):
    """Polarization ray trace of a lens: its Jones pupil at an image field
    point.

    Every refraction contributes the Fresnel transmission of its s and p
    components (with an optional thin film coating), as a 3 x 3 polarization
    ray tracing matrix (Chipman); the product over the surfaces, between the
    aplanatic bases of the object and image space ray directions (see
    propagators.aplanatic_rotation), is the Jones matrix. The bases use the
    absolute ray directions, which for a lens telecentric on both sides
    differ little from those relative to the chief ray that the pupil
    coordinates use. An uncoated surface is a weak diattenuator (s is
    reflected more, so p is transmitted better) with radial orientation;
    coatings add retardance. The geometric optical path is not included
    (that is the wavefront of lens_aberrations()), and neither is the
    birefringence of the glass (stress, or the intrinsic birefringence of
    CaF2) unless given as birefringent media (CubicCrystal).

    Parameters
    ----------
    system : pyoptics.raytrace.System
        Finite conjugate projection lens without mirrors.
    field : pair, optional
        Image field point (units of the system).
    N : int, optional
        Rays across the pupil.
    NA : number, optional
        Normalization of the pupil coordinates; default lens_NA(system).
    coating : list of (n, thickness) or callable, optional
        Thin film layers on every glass-air surface, from the air side (e.g.
        quarter_wave_coating()), thickness in the units of the system; or a
        function coating(i, n_glass) of the surface index returning such a
        list (or None). None: uncoated.
    birefringent : dict {i: medium}, optional
        Birefringent media behind the surfaces i (index into
        system.surfaces), e.g. {38: CubicCrystal(-3.4e-7, "111")}. A medium
        has a method propagation(k, length, wavelength) returning the
        polarization ray tracing matrices of the paths (see CubicCrystal);
        its refractive index is still that of the system.
    normalize : bool, optional
        Divide by the square root of the transmission of the chief ray and
        by its scalar phase (see jones_pupil.decompose()), so that the Jones
        pupil leaves the overall dose unchanged.

    Returns
    -------
    LensJonesPupil
        Fit it with jones_pupil.PauliZernikePupil.fit() or
        PhysicalJonesPupil.fit() to use it in Projector(jones_pupil=...).
    """

    from ..propagators import aplanatic_rotation
    from ..raytrace import CoordinateBreak

    if any(s.mirror for s in system.surfaces):
        raise NotImplementedError("polarization ray tracing of mirrors")
    if NA is None:
        NA = lens_NA(system, N)
    m = system.paraxial().magnification
    obj = np.asarray(field, dtype=float)/m
    _, PX, PY, inside = system._pupil_grid(N)
    px = np.concatenate([[0.0], PX[inside]])
    py = np.concatenate([[0.0], PY[inside]])
    bundle = system.trace_pupil(obj, px, py)
    ok = bundle.valid
    n = bundle.n
    frames = system.frames()[0]
    P = np.tile(np.eye(3, dtype=complex), (len(px), 1, 1))
    for i, s in enumerate(system.surfaces):
        if isinstance(s, CoordinateBreak) or n[i] == n[i + 1]:
            continue
        o, R = frames[i]
        k1, k2 = bundle.directions[i][ok], bundle.directions[i + 1][ok]
        normal = s.normal((bundle.points[i + 1][ok] - o) @ R) @ R.T
        layers = coating(i, max(n[i], n[i + 1])) if callable(coating) else coating
        P_i = np.tile(np.eye(3, dtype=complex), (len(px), 1, 1))
        P_i[ok] = _interface(k1, k2, normal, n[i], n[i + 1], layers or [], system.wavelength)
        P = P_i @ P
        if birefringent and i in birefringent:
            length = np.linalg.norm(bundle.points[i + 2][ok] - bundle.points[i + 1][ok], axis=-1)
            P_i[ok] = birefringent[i].propagation(k2, length, system.wavelength)
            P = P_i @ P
    d_in, d_out = bundle.directions[0], system._image_hits(bundle)[1]
    B_in, B_out = aplanatic_rotation(d_in[:, 0], d_in[:, 1]), aplanatic_rotation(d_out[:, 0], d_out[:, 1])
    J = np.einsum("jip,pjk,klp->pil", B_out, P, B_in)
    transmission = float(np.sum(np.abs(J[0])**2)/2)
    if normalize:
        chief = decompose(J[0])
        J = J/(np.sqrt(chief["transmission"])*np.exp(1j*chief["phase"]))
    q = n[-1]*(d_out[:, :2] - d_out[0, :2])/NA
    ok[0] = False

    return LensJonesPupil(q[ok, 0], q[ok, 1], J[ok], transmission, tuple(float(f) for f in field), NA)


# feature response:

def _shifted_cuts(sim, foci, N, direction, shift=(0.0, 0.0), telecentricity=(0.0, 0.0)):
    """Cuts over two periods through focus, moved by shift +
    focus*tan(telecentricity); also returns these shifts."""

    u, center, cd, direction = sim.cut_coordinates(2*N - 1, 2.0, direction)
    I = sim._cuts(2*N - 1, 2.0, direction, foci, 0.0)[1]
    i = 0 if direction == "x" else 1
    t = telecentricity[i]
    s = shift[i] + np.asarray(foci)*t/np.sqrt(1 - t**2)
    if np.any(s != 0):
        period = (u[-1] - u[0])/2
        I = np.array([np.interp(u - s_f, u[:N - 1], I_f[:N - 1], period=period)
                      for s_f, I_f in zip(s, I)])

    return u, center, direction, I, s


def _response(sim, u, center, I, foci, threshold, dose, expected=0.0):
    """CD and placement of the feature at center + expected (one value per
    focus): its extremum is searched within a quarter period of there, so
    that large shifts are followed; nan where the extremum is not inside
    (no feature, e.g. after a contrast reversal)."""

    bright = sim.layout.feature_is_bright
    period = (u[-1] - u[0])/2
    edges = []
    for I_f, e in zip(I, np.broadcast_to(expected, len(I))):
        near = np.nonzero(np.abs(u - center - e) <= period/4)[0]
        i = near[np.argmax(I_f[near]) if bright else np.argmin(I_f[near])]
        if i in (near[0], near[-1]):
            edges.append((np.nan, np.nan))
            continue
        edges.append(metrics.feature_edges(u, dose*I_f, threshold, u[i], bright))
    edges = np.array(edges)
    cd = edges[:, 1] - edges[:, 0]
    placement = edges.mean(axis=1) - center
    nils = np.array([np.nan if np.isnan(e[0]) else
                     w*np.mean([metrics.image_log_slope(u, I_f, x) for x in e])
                     for I_f, e, w in zip(I, edges, cd)])
    i0 = int(np.argmin(np.abs(foci)))

    return Response(cd[i0], placement[i0], metrics.best_focus(foci, nils, n_fit=5),
                    np.asarray(foci), cd, placement, nils)


def _default_foci(projector):
    return projector.wavelength/projector.NA**2*np.linspace(-0.5, 0.5, 11)


def feature_response(sim, threshold=0.3, dose=1.0, foci=None, N=401, direction=None):
    """CD, placement and best focus of the first feature of a simulation.

    The feature is searched within a quarter period around its nominal
    center plus the shift by the projector's Zernike tilt, so that large
    placement errors are followed; where no feature is found (e.g. behind a
    contrast reversal far out of focus) CD and placement are nan.

    Parameters
    ----------
    sim : LithoSimulation
    threshold, dose : numbers, optional
        Threshold model.
    foci : array, optional
        Foci for best focus and the through focus values (should contain
        0). Default: 11 foci over +-lambda/(2 NA^2).
    N : int, optional
        Samples of the cut over one pitch (the cut covers two pitches).
    direction : {"x", "y"}, optional
        Cut direction, see LithoSimulation.cut_coordinates().

    Returns
    -------
    Response
    """

    foci = _default_foci(sim.projector) if foci is None else np.asarray(foci, dtype=float)
    u, center, direction, I, _ = _shifted_cuts(sim, foci, N, direction)
    # follow the shift by the Zernike tilt:
    P = sim.projector
    tilt = -P.zernikes.get(2 if direction == "x" else 3, 0.0)*P.wavelength/P.NA

    return _response(sim, u, center, I, foci, threshold, dose, tilt)


def zernike_sensitivity(sim, terms=range(2, 26), delta=0.02, threshold=0.3, dose=1.0, foci=None,
                        N=401, direction=None):
    """Derivatives of CD, placement and best focus with respect to Zernike
    coefficients.

    Each coefficient is changed by +-delta (waves) on top of the
    projector's own Zernikes; first and second derivatives follow from
    central differences.

    Parameters
    ----------
    sim : LithoSimulation
    terms : sequence of int, optional
        Fringe indices.
    delta : number, optional
        Step in waves; small enough for the quadratic terms to describe the
        response (default 0.02, i.e. 20 milliwaves).
    threshold, dose, foci, N, direction
        As for feature_response().

    Returns
    -------
    Sensitivity
    """

    terms = np.array(list(terms))
    foci = _default_foci(sim.projector) if foci is None else np.asarray(foci, dtype=float)
    nominal = feature_response(sim, threshold, dose, foci, N, direction)
    keys = ("cd", "placement", "best_focus")
    linear = {k: np.zeros(len(terms)) for k in keys}
    quadratic = {k: np.zeros(len(terms)) for k in keys}
    z0 = sim.projector.zernikes
    for i, j in enumerate(terms):
        r = []
        for s in (-1, 1):
            z = dict(z0)
            z[j] = z.get(j, 0.0) + s*delta
            r.append(feature_response(sim.with_(projector=sim.projector.with_(zernikes=z)),
                                      threshold, dose, foci, N, direction))
        for k in keys:
            v_m, v_0, v_p = getattr(r[0], k), getattr(nominal, k), getattr(r[1], k)
            linear[k][i] = (v_p - v_m)/(2*delta)
            quadratic[k][i] = (v_p - 2*v_0 + v_m)/delta**2

    return Sensitivity(terms, delta, nominal, linear, quadratic)


def predict(sensitivity, zernikes):
    """Changes of CD, placement and best focus predicted from sensitivities.

    ``sum_j L_j c_j + Q_j c_j**2/2`` with the first and second derivatives
    L_j, Q_j; mixed terms (e.g. defocus and spherical aberration together)
    are neglected.

    Parameters
    ----------
    sensitivity : Sensitivity
    zernikes : dict
        Coefficients {j: c_j} in waves; terms without sensitivities are
        ignored.

    Returns
    -------
    dict
        Predicted changes for "cd", "placement" and "best_focus".
    """

    c = np.array([zernikes.get(int(j), 0.0) for j in sensitivity.terms])

    return {k: float(np.sum(sensitivity.linear[k]*c + sensitivity.quadratic[k]*c**2/2))
            for k in sensitivity.linear}


def through_slit(sim, system, x, y=(0.0,), threshold=0.3, dose=1.0, foci=None, correction=None,
                 n_terms=36, N_pupil=32, N=401, direction=None, length_unit=1e-6):
    """CD, placement and best focus across the slit of a scanner.

    At each slit position x, the cuts through focus for all scan positions
    y (with the Zernikes, distortion and telecentricity of the lens at (x,
    y)) are averaged, which is the image a scanned wafer point receives.
    The placement is relative to the nominal position of the feature at the
    paraxial image point (minus the correction).

    Parameters
    ----------
    sim : LithoSimulation
        Mask, illumination and projector; the Zernikes of the projector
        are replaced by those of the lens, its NA normalizes the pupil.
    system : pyoptics.raytrace.System
        The lens, at the wavelength of the projector.
    x : array
        Slit positions (image side, units of the system).
    y : array, optional
        Scan positions, e.g. np.linspace(-4, 4, 5) for an 8 mm slit; one
        position (default 0) gives the static image.
    threshold, dose, foci, N, direction
        As for feature_response().
    correction : callable, optional
        correction(x, y) -> (dx, dy): placement removed by the scanner,
        e.g. from linear_correction().
    n_terms, N_pupil, length_unit
        As for lens_aberrations() (n_terms, N, length_unit).

    Returns
    -------
    SlitResult
    """

    P = sim.projector
    if not np.isclose(system.wavelength/length_unit, P.wavelength, rtol=1e-3):
        raise ValueError("The wavelength of the system ({} in simulation units) differs from that "
                         "of the projector ({}); use system.with_wavelength()"
                         .format(system.wavelength/length_unit, P.wavelength))
    foci = _default_foci(P) if foci is None else np.asarray(foci, dtype=float)
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    out, aberrations = [], []
    for x_i in x:
        row, cuts, expected = [], [], []
        for y_j in y:
            ab = lens_aberrations(system, (x_i, y_j), n_terms, N_pupil, P.NA, length_unit)
            row.append(ab)
            shift = np.array(ab.distortion)
            if correction is not None:
                shift = shift - np.array(correction(x_i, y_j))
            s = sim.with_(projector=P.with_(zernikes=ab.zernikes))
            u, center, d, I, s_f = _shifted_cuts(s, foci, N, direction, shift, ab.telecentricity)
            i = 0 if d == "x" else 1
            cuts.append(I)
            expected.append(s_f + ab.placement[i] - ab.distortion[i])
        aberrations.append(row)
        out.append(_response(sim, u, center, np.mean(cuts, axis=0), foci, threshold, dose,
                             np.mean(expected, axis=0)))

    return SlitResult(x, y, np.array([r.cd for r in out]), np.array([r.placement for r in out]),
                      np.array([r.best_focus for r in out]), aberrations)
