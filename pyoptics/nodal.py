#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Nodal aberration theory: the third order aberrations of optical
systems whose surfaces are decentred and tilted
(notebook 155, off-axis mirrors and nodal aberrations).

In a rotationally symmetric system every surface contributes aberrations
centred on the common axis, and the wavefront of the field point H
(normalized, a two-dimensional vector) is the Seidel expansion in H and the
pupil vector rho. Buchroeder and Thompson noticed that a tilted or
decentred surface still contributes the same aberrations, only centred on
its own field point sigma_j, the field point whose chief ray passes
through the surface's centre of curvature (the spherical part) or its
vertex (the aspheric part). The total wavefront is ::

    W(H, rho) = sum_j [ W040_j (rho.rho)^2
                        + W131_j ((H - sigma_j).rho)(rho.rho)
                        + 1/2 W222_j ((H - sigma_j)^2 . rho^2)
                        + W220M_j ((H - sigma_j).(H - sigma_j))(rho.rho)
                        + W311_j ((H - sigma_j).(H - sigma_j))((H - sigma_j).rho) ]

with the medial field curvature W220M = W220 + W222/2 and the vector
products of Thompson (H^2 is the vector with twice the angle and the
squared length). Writing vectors as complex numbers H = H_x + i H_y, this
module evaluates the sums in closed form (Thompson 2005):

- Coma: Z7 + i Z8 = (W131 H - A131)/3 with A131 = sum W131_j sigma_j. If
  the system is aplanatic (W131 = 0, a Ritchey-Chrétien), a misalignment
  gives coma that is the same over the whole field.
- Astigmatism: Z5 + i Z6 = 1/2 (W222 H^2 - 2 A222 H + B222) with A222 =
  sum W222_j sigma_j and B222 = sum W222_j sigma_j^2. It vanishes at the
  two roots of the quadratic, the nodes a222 +- sqrt(a222^2 - b222^2)
  (a222 = A222/W222, b222^2 = B222/W222): binodal astigmatism.
- Focus: Z4 = 1/2 (W220M ``|H|^2`` - 2 A220M . H + B220M) + W040/2 (a tilted
  and curved image surface), A220M = sum W220M_j sigma_j. On a fixed
  image plane the misaligned beam adds a focus linear in the field, from
  its tilt against the plane (`boresight_focus()`).

(Fringe Zernikes in waves, Z5 + i Z6 positive real for astigmatism along
x, `raytrace.System.zernike()` conventions.) The field displacement
vectors, in units of the maximum field of the system (H = 1 at the
maximum field), are ::

    sigma_sphere = n c delta_C / A_bar,  sigma_asphere = delta_v / y_bar,

with the lateral displacement delta_C = d + R (a_y, -a_x) of the centre of
curvature and delta_v = d of the vertex for a decentre d and tilts a_x,
a_y (`raytrace` conventions), the paraxial chief ray height y_bar and the
refraction invariant A_bar = n (u_bar + y_bar c) of the chief ray of the
maximum field (`aberration_field_centres()`).

The aberrations of a freeform departure follow from the footprint of the
beam on the surface: at the field H the beam covers the disk around
y_bar H with the radius y (marginal ray height); the departure Delta(x, y)
there adds (n - n') Delta to the OPD (signed indices, 2 Delta for a mirror
hit by light travelling in +z). A Zernike astigmatism on a surface in the
stop gives astigmatism that is constant over the field, a Zernike coma
away from the stop gives constant coma and field-linear astigmatism
(Fuerschbach et al. 2014; `freeform_field_zernikes()`).

Functions:

- aberration_field_centres(): sigma vectors of all surfaces from their
  decentres and tilts.
- nodal_coefficients(): the sums W..., A131, A222, B222, A220M, B220M.
- nat_zernikes(): Fringe Zernikes Z1...Z9 predicted at field points (with
  an aperture offset by sampling the wavefront over the sub-aperture).
- astigmatic_nodes(), coma_free_pivot(): the nodes of binodal astigmatism
  and the point about which a mirror tilts without adding coma.
- fit_field(): the same coefficients fitted to ray traced Zernikes over a
  field; boresight_focus(): the focus from the tilt of the beam against
  the image plane.
- freeform_field_zernikes(): field dependent Zernikes of the departure of
  a `raytrace.ZernikeSurface`.

Limits: third order only; displacements of the stop surface and
coordinate breaks are not supported (move the other surfaces instead); the
paraxial data are those of the nominal system (`raytrace`).

References
----------
R. A. Buchroeder, "Tilted component optical systems", PhD thesis,
University of Arizona (1976).

R. V. Shack and K. P. Thompson, "Influence of alignment errors of a
telescope system on its aberration field", Proc. SPIE 251, 146-153 (1980).

K. P. Thompson, "Description of the third-order optical aberrations of
near-circular pupil optical systems without symmetry", J. Opt. Soc. Am. A
22, 1389-1401 (2005).

K. P. Thompson, T. Schmid, O. Cakmakci and J. P. Rolland, "Real-ray-based
method for locating individual surface aberration field centers in
imaging optical systems without symmetry", J. Opt. Soc. Am. A 26,
1503-1517 (2009).

T. Schmid, J. P. Rolland, A. Rakich and K. P. Thompson, "Separation of the
effects of astigmatic figure error from misalignments using nodal
aberration theory (NAT)", Opt. Express 18, 17433-17447 (2010).

K. Fuerschbach, J. P. Rolland and K. P. Thompson, "Theory of aberration
fields for general optical systems with freeform surfaces", Opt. Express
22, 26585-26606 (2014).
"""

import warnings
from collections import namedtuple

import numpy as np

from . import raytrace as rt
from .bases import fringe_zernike


__all__ = ["aberration_field_centres", "NodalCoefficients", "nodal_coefficients", "nat_zernikes",
           "astigmatic_nodes", "coma_free_pivot", "FieldFit", "fit_field",
           "freeform_field_zernikes", "complex_field", "boresight_focus"]


def complex_field(fields, max_field):
    """Field points (x, y) as normalized complex numbers H = (x + i y)/max_field.

    Parameters
    ----------
    fields : (F, 2) array
    max_field : number

    Returns
    -------
    (F,) complex array
    """

    f = np.atleast_2d(np.asarray(fields, float))

    return (f[:, 0] + 1j*f[:, 1])/max_field


def _paraxial(system, field):
    """Marginal and chief ray data in front of every surface: y, u_in,
    y_bar, u_bar_in, signed indices n_s (before) and n_s' (after)."""

    (y_m, u_m), (y_c, u_c) = system._paraxial_rays(field)
    ym, um = system._paraxial_trace(y_m, u_m)
    yc, uc = system._paraxial_trace(y_c, u_c)
    n_s = system.n*system.sigma
    u_in = np.concatenate([[u_m], um[:-1]])
    ub_in = np.concatenate([[u_c], uc[:-1]])

    return ym, u_in, yc, ub_in, n_s


def _check(system, field):
    f = rt._field_xy(system.max_field if field is None else field)[1]
    if f == 0:
        raise ValueError("Needs a field (the system has max_field = 0).")
    for s in system.surfaces:
        if isinstance(s, rt.CoordinateBreak) and s.displaced:
            raise NotImplementedError("Coordinate breaks are not supported.")


def optical_axis_ray(system):
    """The optical axis ray (OAR): the real ray from the centre of the field
    through the centre of the stop, where it meets each surface.

    The stop is taken at its nominal centre (a decentred stop that only
    selects an off-axis part of the pupil, an aperture offset, is centred
    for this ray).

    Parameters
    ----------
    system : raytrace.System

    Returns
    -------
    y_o, u_o : complex arrays
        Lateral position x + i y of the ray in the vertex plane of every
        surface and its slope (dx + i dy)/dz in front of the surface.
    """

    s = system.copy()
    stop = s.surfaces[s.stop_index]
    if stop.R == np.inf:
        stop.decenter = (0.0, 0.0)
    b = s.trace(s.rays((0.0, 0.0), 0.0, 0.0), clip=False)
    P, D = b.points[1:-1, 0], b.directions[:-2, 0]
    z_v = s.vertex_z
    u = (D[:, 0] + 1j*D[:, 1])/D[:, 2]
    y = P[:, 0] + 1j*P[:, 1] - u*(P[:, 2] - z_v)

    return y, u


def _displacements(system):
    """Complex displacements of every surface relative to the optical axis
    ray: of the vertex, d - y_o, and of the centre of curvature times c,
    c (d - y_o) + (a_y - i a_x) - u_o."""

    y_o, u_o = optical_axis_ray(system)
    out = []
    for i, s in enumerate(system.surfaces):
        d = s.decenter[0] + 1j*s.decenter[1]
        a = s.tilt[1] - 1j*s.tilt[0]
        if i == system.stop_index and s.R == np.inf:
            d = 0.0  # an aperture offset, not a misalignment
        dv = d - y_o[i]
        out.append((dv, s.c*dv + a - u_o[i]))

    # numerical noise of the trace in an aligned system:
    scale = max(1.0, np.max(np.abs(system.vertex_z)))

    return [(0j if abs(dv) < 1e-12*scale else dv, 0j if abs(cd) < 1e-14 else cd)
            for dv, cd in out]


def aberration_field_centres(system, field=None):
    """Field displacement vectors sigma of the spherical and aspheric parts
    of every surface.

    Parameters
    ----------
    system : raytrace.System
        Surfaces with `decenter` and `tilt` (the nominal paraxial data are
        those of the centred system).
    field : number, optional
        Field of H = 1 (default: the maximum field of the system).

    Returns
    -------
    sigma_sphere, sigma_asphere : complex arrays
        Per surface, in units of the field; 0 for undisplaced surfaces, inf
        where the paraxial chief ray passes the vertex (aspheric part) or
        the centre of curvature (spherical part) of a displaced surface.
    """

    _check(system, field)
    ym, u_in, yc, ub_in, n_s = _paraxial(system, field)
    s_sph = np.zeros(len(system.surfaces), complex)
    s_asph = np.zeros_like(s_sph)
    disp = _displacements(system)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, s in enumerate(system.surfaces):
            d, cdC = disp[i]
            Ab = n_s[i]*(ub_in[i] + yc[i]*s.c)
            if cdC != 0:
                s_sph[i] = n_s[i]*cdC/Ab if Ab != 0 else np.inf
            if d != 0:
                s_asph[i] = d/yc[i] if yc[i] != 0 else np.inf

    return s_sph, s_asph


NodalCoefficients = namedtuple("NodalCoefficients", [
    "W040", "W131", "W222", "W220M", "W311", "A131", "A222", "B222", "A220M", "B220M",
    "sigma_sphere", "sigma_asphere", "W_sphere", "W_asphere", "max_field", "distortion"])
NodalCoefficients.__doc__ = """Coefficients of nodal aberration theory
(waves, H = 1 at max_field).

Attributes
----------
W040, W131, W222, W220M, W311 : sums over all surfaces (as for the centred
    system, W220M = W220 + W222/2 the medial field curvature)
A131 : sum W131_j sigma_j (complex), the constant coma is -A131/3 in Z7 + i Z8
A222, B222 : sum W222_j sigma_j and sum W222_j sigma_j^2 (complex)
A220M : sum W220M_j sigma_j (complex), the tilt of the medial image surface
B220M : sum W220M_j ``|sigma_j|^2`` (real), its shift
sigma_sphere, sigma_asphere : field displacements of each surface
W_sphere, W_asphere : (S, 5) per surface coefficients W040, W131, W222,
    W220M, W311 of the spherical and aspheric parts
max_field : the field of H = 1
distortion : (C1, C2, C3, C4) sums W311_j sigma_j, W311_j ``|sigma_j|^2``,
    W311_j sigma_j^2, W311_j sigma_j^2 conj(sigma_j) of the distortion terms
"""


def _products(system, field):
    """Per surface and part: the coefficients and the sums of W_k times the
    powers of sigma that appear, evaluated without dividing by zero where
    the aspheric part sits in the stop (y_bar = 0)."""

    ym, u_in, yc, ub_in, n_s = _paraxial(system, field)
    lam = system.wavelength
    sd = system.seidel(field)
    S, S_a = np.nan_to_num(sd.S), np.nan_to_num(sd.aspheric)
    S_s = S - S_a
    to_w = np.array([-1/8, -1/2, -1/2, -1/4, -1/2])/lam  # W040, W131, W222, W220M, W311

    def waves(Sx):
        # W220M = -(2 S_III + S_IV)/4
        W = Sx*to_w
        W[..., 3] = -(2*Sx[..., 2] + Sx[..., 3])/4/lam
        return W

    W_s, W_a = waves(S_s), waves(S_a)
    s_sph, s_asph = np.zeros(len(S), complex), np.zeros(len(S), complex)
    # sums: coma C, astigmatism A, B, focus F1, F2, distortion D1..D4
    out = dict(A131=0j, A222=0j, B222=0j, A220M=0j, B220M=0.0, C1=0j, C2=0.0, C3=0j, C4=0j)

    def add(W, s):
        # W: per surface coefficients, s: complex sigma (finite)
        out["A131"] += W[1]*s
        out["A222"] += W[2]*s
        out["B222"] += W[2]*s**2
        out["A220M"] += W[3]*s
        out["B220M"] += W[3]*abs(s)**2
        out["C1"] += W[4]*s
        out["C2"] += W[4]*abs(s)**2
        out["C3"] += W[4]*s**2
        out["C4"] += W[4]*s**2*np.conj(s)

    disp = _displacements(system)
    for i, surf in enumerate(system.surfaces):
        if isinstance(surf, rt.CoordinateBreak):
            continue
        d, cdC = disp[i]
        # spherical part
        Ab = n_s[i]*(ub_in[i] + yc[i]*surf.c)
        if cdC != 0:
            if Ab == 0:
                warnings.warn("Surface {}: the chief ray passes the centre of curvature, "
                              "sigma is infinite.".format(i))
                s_sph[i] = np.inf
            else:
                s_sph[i] = n_s[i]*cdC/Ab
                add(W_s[i], s_sph[i])
        # aspheric part: S_k = dS y^(4 - k) y_bar^k, sigma = d/y_bar
        if d != 0 and np.any(S_a[i] != 0):
            if yc[i] != 0:
                s_asph[i] = d/yc[i]
                add(W_a[i], s_asph[i])
            else:
                s_asph[i] = np.inf
                dS = S_a[i, 0]/ym[i]**4
                y = ym[i]
                # only the terms with as many sigmas as powers of y_bar survive
                out["A131"] += -dS*y**3*d/2/lam
                out["B222"] += -dS*y**2*d**2/2/lam
                out["B220M"] += -2*dS*y**2*abs(d)**2/4/lam
                out["C4"] += -dS*y*d**2*np.conj(d)/2/lam

    return W_s, W_a, s_sph, s_asph, out


def nodal_coefficients(system, field=None):
    """Coefficients of nodal aberration theory of a system with decentred
    and tilted surfaces.

    Parameters
    ----------
    system : raytrace.System
    field : number, optional
        Field of H = 1 (default: the maximum field).

    Returns
    -------
    NodalCoefficients
    """

    _check(system, field)
    W_s, W_a, s_sph, s_asph, o = _products(system, field)
    W = (W_s + W_a).sum(axis=0)
    max_field = rt._field_xy(system.max_field if field is None else field)[1]

    return NodalCoefficients(W[0], W[1], W[2], W[3], W[4], o["A131"], o["A222"], o["B222"],
                             o["A220M"], o["B220M"], s_sph, s_asph, W_s, W_a, max_field,
                             (o["C1"], o["C2"], o["C3"], o["C4"]))


def _wavefront(coef, H, rho):
    """Third order NAT wavefront at complex field points H (F,) and complex
    pupil points rho (P,): (F, P) array in waves."""

    H = np.asarray(H, complex)[:, None]
    r = np.asarray(rho, complex)[None, :]
    rr = np.abs(r)**2
    W = coef.W040*rr**2 + 0j
    coma = coef.W131*H - coef.A131
    W = W + np.real(coma*np.conj(r))*rr
    ast = 0.5*(coef.W222*H**2 - 2*coef.A222*H + coef.B222)
    W = W + np.real(ast*np.conj(r)**2)
    foc = coef.W220M*np.abs(H)**2 - 2*np.real(np.conj(coef.A220M)*H) + coef.B220M
    W = W + foc*rr
    C1, C2, C3, C4 = coef.distortion
    dist = (coef.W311*H**2*np.conj(H) - H**2*np.conj(C1) - 2*np.abs(H)**2*C1 + 2*H*C2
            + np.conj(H)*C3 - C4)
    W = W + np.real(dist*np.conj(r))

    return np.real(W)


def nat_zernikes(coef, H, n_terms=9, pupil_offset=0.0, pupil_scale=1.0, N=32):
    """Fringe Zernikes predicted by nodal aberration theory at field points.

    Without an aperture offset the result is in closed form (Z4 to Z9);
    with one, the wavefront of the parent pupil is sampled over the
    sub-aperture (centre pupil_offset, radius pupil_scale, in units of the
    parent pupil radius) and fitted.

    Parameters
    ----------
    coef : NodalCoefficients
    H : complex array
        Normalized field points (`complex_field()`).
    n_terms : int, optional
    pupil_offset : complex, optional
        Centre of the used aperture in the parent pupil.
    pupil_scale : number, optional
        Radius of the used aperture in the parent pupil.
    N : int, optional
        Samples across the sub-aperture.

    Returns
    -------
    (F, n_terms) array
        Coefficients in waves; Z1 (piston), Z2 and Z3 (tilts, including the
        distortion) are those of the third order wavefront, which
        `raytrace.System.zernike()` gives relative to the real chief ray.
    """

    H = np.atleast_1d(np.asarray(H, complex))
    if pupil_offset == 0 and pupil_scale == 1 and n_terms <= 9:
        Z = np.zeros((len(H), 9))
        coma = (coef.W131*H - coef.A131)/3
        ast = 0.5*(coef.W222*H**2 - 2*coef.A222*H + coef.B222)
        foc = coef.W220M*np.abs(H)**2 - 2*np.real(np.conj(coef.A220M)*H) + coef.B220M
        Z[:, 3] = foc/2 + coef.W040/2
        Z[:, 4], Z[:, 5] = ast.real, ast.imag
        Z[:, 6], Z[:, 7] = coma.real, coma.imag
        Z[:, 8] = coef.W040/6
        # piston and tilts from the closed forms of the Fringe expansion:
        Z[:, 0] = foc/2 + coef.W040/3
        C1, C2, C3, C4 = coef.distortion
        dist = (coef.W311*H**2*np.conj(H) - H**2*np.conj(C1) - 2*np.abs(H)**2*C1 + 2*H*C2
                + np.conj(H)*C3 - C4)
        tilt = dist + 2*coma
        Z[:, 1], Z[:, 2] = tilt.real, tilt.imag

        return Z[:, :n_terms]

    u = np.linspace(-1, 1, N)
    U, V = np.meshgrid(u, u)
    ok = U**2 + V**2 <= 1
    q = U[ok] + 1j*V[ok]
    W = _wavefront(coef, H, pupil_offset + pupil_scale*q)
    A = np.column_stack([fringe_zernike(j, U[ok], V[ok]) for j in range(1, n_terms + 1)])

    return np.linalg.lstsq(A, W.T, rcond=None)[0].T


def astigmatic_nodes(coef):
    """The two nodes of binodal astigmatism (complex, normalized field).

    Parameters
    ----------
    coef : NodalCoefficients or FieldFit

    Returns
    -------
    (2,) complex array
        a222 -+ sqrt(a222^2 - b222^2); both at the centre of the field for
        a centred system.
    """

    a = coef.A222/coef.W222
    b2 = coef.B222/coef.W222
    r = np.sqrt(a**2 - b2 + 0j)

    return np.array([a - r, a + r])


def coma_free_pivot(system, surface, axis=0, field=None):
    """Point on the axis of a surface about which it can be tilted without
    changing the coma (the coma-free pivot of a telescope secondary).

    A tilt a about a point z_p behind the vertex (along +z) decentres the
    vertex by z_p a; the pivot is where the coma of the decentre cancels
    that of the tilt about the vertex.

    Parameters
    ----------
    system : raytrace.System
    surface : int
    axis : int, optional
        0: tilt about x (in the y-z plane), 1: about y.
    field : number, optional

    Returns
    -------
    z_p : float
        Signed distance of the pivot from the vertex along the axis (length
        units).
    """

    def a131(decenter, tilt):
        s = system.copy()
        s.surfaces[surface].decenter = decenter
        s.surfaces[surface].tilt = tilt
        return nodal_coefficients(s, field).A131

    eps = 1e-6
    if axis == 0:
        k_d = a131((0.0, eps), (0.0, 0.0, 0.0))/eps    # per unit decentre in y
        k_a = a131((0.0, 0.0), (eps, 0.0, 0.0))/eps    # per unit tilt about x
        # rotation a_x about the pivot moves the vertex by +z_p a_x in y
        return float(np.real(-k_a/k_d))
    k_d = a131((eps, 0.0), (0.0, 0.0, 0.0))/eps
    k_a = a131((0.0, 0.0), (0.0, eps, 0.0))/eps
    # rotation a_y about the pivot moves the vertex by -z_p a_y in x
    return float(np.real(k_a/k_d))


FieldFit = namedtuple("FieldFit", ["W131", "A131", "W222", "A222", "B222", "W220M", "A220M",
                                   "B220M", "residual"])
FieldFit.__doc__ = """Nodal aberration coefficients fitted to Zernikes over
a field (see `fit_field()`); residual is the RMS of the fit residuals of
Z4 to Z8 (waves)."""


def fit_field(H, Z):
    """Fit the nodal field dependence to Zernikes over the field.

    Z7 + i Z8 = (W131 H - A131)/3, Z5 + i Z6 = (W222 H^2 - 2 A222 H + B222)/2
    and Z4 = (W220M ``|H|^2`` - 2 A220M.H + B220M)/2 + const.

    Parameters
    ----------
    H : complex array
        Normalized field points.
    Z : (F, >=8) array
        Fringe Zernikes (waves) at the field points, e.g. from
        `off_axis.field_zernikes()`.

    Returns
    -------
    FieldFit
        The constant of Z4 (which includes W040/2) is returned in B220M.
    """

    H = np.asarray(H, complex)
    Z = np.asarray(Z, float)
    coma = Z[:, 6] + 1j*Z[:, 7]
    ast = Z[:, 4] + 1j*Z[:, 5]
    M1 = np.column_stack([np.ones_like(H), H])
    p, res1 = np.linalg.lstsq(M1, coma, rcond=None)[:2]
    M2 = np.column_stack([np.ones_like(H), H, H**2])
    q = np.linalg.lstsq(M2, ast, rcond=None)[0]
    M3 = np.column_stack([np.abs(H)**2, H.real, H.imag, np.ones(len(H))])
    r = np.linalg.lstsq(M3, Z[:, 3], rcond=None)[0]
    resid = np.concatenate([np.abs(coma - M1 @ p), np.abs(ast - M2 @ q), np.abs(Z[:, 3] - M3 @ r)])

    return FieldFit(3*p[1], -3*p[0], 2*q[2], -q[1], 2*q[0], 2*r[0], -(r[1] + 1j*r[2]), 2*r[3],
                    float(np.sqrt(np.mean(resid**2))))


def boresight_focus(system, fields):
    """Z4 (waves) from the tilt of the beam against the image plane.

    A misaligned system also tilts the beam of the field centre against the
    image plane by the small angle eps (the boresight error); the image
    surface, perpendicular to the beam, then lies the distance eps.r in
    front of the plane at the image point r, which adds
    -n' (eps.r) NA^2/(4 lambda) to Z4, linear in the field. This first
    order effect is not part of the third order wavefront of
    `nat_zernikes()`; add it to compare the focus with
    `raytrace.System.zernike()`.

    Parameters
    ----------
    system : raytrace.System
    fields : (F, 2) array
        Field points (absolute).

    Returns
    -------
    (F,) array
    """

    def chief(f):
        b = system.trace(system.rays(tuple(f), 0.0, 0.0), clip=False)
        return system.to_image_frame(b.points[-1, 0], b.directions[-1, 0])

    p0, d0 = chief((0.0, 0.0))
    eps = d0[:2]/d0[2]
    n_img = abs(system.n[-1])
    na = system.paraxial().na
    out = []
    for f in np.atleast_2d(fields):
        r = chief(f)[0][:2] - p0[:2]
        out.append(-n_img*np.dot(eps, r)*na**2/(4*system.wavelength))

    return np.array(out)


def freeform_field_zernikes(system, surface, H, n_terms=9, field=None, pupil_offset=None, N=32):
    """Field dependent Zernikes of the departure of a `raytrace.ZernikeSurface`
    (freeform nodal aberration theory, to first order in the departure).

    At the field H the paraxial beam covers the disk of radius ``|y|`` (marginal
    ray height) around y_bar H (chief ray height) on the surface, shifted by
    y times the pupil offset; the departure there, times (n - n') (signed
    indices), is the change of the OPD, fitted with Fringe Zernikes of the
    entrance pupil coordinates.

    Parameters
    ----------
    system : raytrace.System
    surface : int
        Index of a ZernikeSurface.
    H : complex array
        Normalized field points.
    n_terms : int, optional
    field : number, optional
        Field of H = 1 (default: the maximum field).
    pupil_offset : complex, optional
        Centre of the used aperture in the parent pupil (units of the stop
        semi-diameter); default: from the decentre of the stop surface.
    N : int, optional
        Pupil samples across.

    Returns
    -------
    (F, n_terms) array
        Waves.
    """

    s = system.surfaces[surface]
    if not isinstance(s, rt.ZernikeSurface):
        raise TypeError("Surface {} is not a ZernikeSurface.".format(surface))
    ym, u_in, yc, ub_in, n_s = _paraxial(system, field)
    if pupil_offset is None:
        stop = system.surfaces[system.stop_index]
        pupil_offset = (stop.decenter[0] + 1j*stop.decenter[1])/stop.semi_diameter
    H = np.atleast_1d(np.asarray(H, complex))
    u = np.linspace(-1, 1, N)
    U, V = np.meshgrid(u, u)
    ok = U**2 + V**2 <= 1
    q = U[ok] + 1j*V[ok]
    y, yb = ym[surface], yc[surface]
    A = np.column_stack([fringe_zernike(j, U[ok], V[ok]) for j in range(1, n_terms + 1)])
    out = []
    for h in H:
        p = yb*h + y*(pupil_offset + q)
        dep = s.departure(p.real, p.imag)
        W = (n_s[surface] - n_s[surface + 1])*dep/system.wavelength
        out.append(np.linalg.lstsq(A, W, rcond=None)[0])

    return np.array(out)
