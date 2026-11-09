#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Reflecting telescopes: the Newtonian, the classical Cassegrain, the
Ritchey-Chrétien and the Schmidt camera as `raytrace.System` objects, with
the closed form third order aberrations of conic mirrors to check them
against (notebook 97).

All builders return systems for an object at infinity, light entering in
+z, lengths in mm. The aperture stop is the primary mirror (the corrector
plate for the Schmidt camera). Following the conventions of `raytrace`, a
concave mirror facing the incoming light has R < 0, and behind a mirror
thicknesses are negative.

Two-mirror telescopes are described by the focal length f1 of the primary,
the system focal length f and the back focal distance b (from the primary
vertex to the focus, positive behind the primary). The secondary
magnification is m = f/f1, beta = b/f1, and the secondary sits at the
distance d = f1 - p in front of the primary, where p = (f1 + b)/(m + 1) is
its distance to the primary focus (`two_mirror_geometry()`).

Closed form results (Schroeder 2000, chapter 6; Wilson 2007), for a field
angle theta (radians), F = f/D, stop at the primary:

- Single conic mirror of focal length f, stop at the mirror: wavefront
  coefficients W040 = -(1 + K) y^4/(32 f^3) (y = D/2), W131 = y^3 theta/
  (4 f^2) and W222 = -y^2 theta^2/(2 f); the sagittal image surface is
  flat, the medial one has the radius f and the tangential one f/2
  (`mirror_aberrations()`).
- Angular tangential coma ACC = 3 W131/y; for a paraboloid and for the
  classical Cassegrain (paraboloid primary, hyperboloid secondary with
  K2 = -((m + 1)/(m - 1))^2) it is 3 theta/(16 F^2), the coma of a
  paraboloid of the same focal ratio.
- The Ritchey-Chrétien removes spherical aberration and coma with the
  Schwarzschild conics (`rc_conics()`) ::

      K1 = -1 - 2 (1 + beta)/(m^2 (m - beta)),
      K2 = -((m + 1)/(m - 1))^2 - 2 m (m + 1)/((m - beta)(m - 1)^3).

- Angular astigmatism (blur diameter at the medial focus) AAS = 2 W222/y:
  -theta^2/(2F) (m^2 + beta)/(m (1 + beta)) for the classical Cassegrain and
  -theta^2/(2F) (m (2m + 1) + beta)/(2 m (1 + beta)) for the
  Ritchey-Chrétien.
- Petzval curvature 1/R_P = -(2/R1 - 2/R2) of the two mirrors (vertex
  radii R1, R2 in the raytrace convention); the medial surface has the
  curvature 1/R_P + 2 A with the astigmatic curvature A = AAS/(y theta^2)
  (`two_mirror_aberrations()`).
- Schmidt camera: a spherical mirror of radius R = 2 f with the stop at its
  centre of curvature has no coma and no astigmatism; a thin aspheric
  plate of index n there, of thickness t(r) = (r^4 - k r_m^2 r^2)/(4 (n - 1)
  R^3), removes its spherical aberration (`schmidt_plate()`). The image
  surface is a sphere of radius f concentric with the mirror. At another
  wavelength the plate is off by the fraction dn/(n - 1); with the
  neutral zone at sqrt(k/2) r_m = 0.866 r_m (k = 3/2) the residual
  aberration is a Chebyshev polynomial and the chromatic blur has the
  angular diameter dn/(128 (n - 1) F^3) (`schmidt_chromatic_blur()`).

Functions:

- two_mirror_geometry(), rc_conics(), cassegrain_conic(): first order
  layout and conic constants of two-mirror telescopes.
- newtonian(), two_mirror_telescope(), schmidt_camera(): ray traceable
  telescopes; schmidt_plate(): the corrector.
- mirror_aberrations(), two_mirror_aberrations(),
  schmidt_chromatic_blur(): closed form third order results.
- spot(): a spot diagram with a central obscuration (annular pupil),
  best_focus(): the image distance with the smallest RMS spot.

References
----------
K. Schwarzschild, "Untersuchungen zur geometrischen Optik II",
Astronomische Mitteilungen der Königlichen Sternwarte zu Göttingen 10
(1905).

D. J. Schroeder, "Astronomical Optics", 2nd ed., Academic Press (2000).

R. N. Wilson, "Reflecting Telescope Optics I", 2nd ed., Springer (2007).

B. Schmidt, "Ein lichtstarkes komafreies Spiegelsystem", Mitteilungen der
Hamburger Sternwarte in Bergedorf 7, 15-17 (1932).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt


__all__ = ["TwoMirror", "two_mirror_geometry", "rc_conics", "cassegrain_conic", "newtonian",
           "two_mirror_telescope", "schmidt_plate", "schmidt_camera", "MirrorAberrations",
           "mirror_aberrations", "TwoMirrorAberrations", "two_mirror_aberrations",
           "schmidt_chromatic_blur", "Spot", "spot", "best_focus"]


TwoMirror = namedtuple("TwoMirror", ["m", "beta", "d", "p", "q", "R1", "R2", "obscuration"])
TwoMirror.__doc__ = """First order layout of a two-mirror (Cassegrain) telescope.

Attributes
----------
m : float
    Secondary magnification f/f1.
beta : float
    Back focal distance over primary focal length, b/f1.
d : float
    Separation of the mirrors.
p : float
    Distance from the secondary to the primary focus.
q : float
    Distance from the secondary to the final focus (d + b).
R1, R2 : float
    Vertex radii in the raytrace convention (both negative for a
    Cassegrain: concave primary, convex secondary).
obscuration : float
    Diameter of the on-axis beam on the secondary over D, p/f1.
"""


def two_mirror_geometry(f1, f, b):
    """First order layout of a Cassegrain telescope.

    Parameters
    ----------
    f1 : float
        Focal length of the primary.
    f : float
        Focal length of the system (f > f1 for a Cassegrain).
    b : float
        Back focal distance, from the primary vertex to the focus (positive
        behind the primary).

    Returns
    -------
    TwoMirror
        Magnification, separations and radii.
    """

    m = f/f1
    p = (f1 + b)/(m + 1)
    d = f1 - p
    q = d + b
    R2 = 2*p*q/(q - p)
    return TwoMirror(m, b/f1, d, p, q, -2*f1, -R2, p/f1)


def rc_conics(m, beta):
    """Conic constants of the Ritchey-Chrétien telescope (Schwarzschild).

    Spherical aberration and coma vanish to third order for ::

        K1 = -1 - 2 (1 + beta)/(m^2 (m - beta)),
        K2 = -((m + 1)/(m - 1))^2 - 2 m (m + 1)/((m - beta)(m - 1)^3).

    Parameters
    ----------
    m : float
        Secondary magnification f/f1.
    beta : float
        b/f1.

    Returns
    -------
    K1, K2 : float
        Conic constants of the primary and the secondary (hyperboloids).
    """

    K1 = -1 - 2*(1 + beta)/(m**2*(m - beta))
    K2 = -((m + 1)/(m - 1))**2 - 2*m*(m + 1)/((m - beta)*(m - 1)**3)
    return K1, K2


def cassegrain_conic(m):
    """Conic constant of the secondary of a classical Cassegrain.

    The hyperboloid with its foci at the primary focus and the final focus,
    K2 = -((m + 1)/(m - 1))^2.

    Parameters
    ----------
    m : float
        Secondary magnification.

    Returns
    -------
    K2 : float
        Conic constant.
    """

    return -((m + 1)/(m - 1))**2


def newtonian(D, F, conic=-1.0, field_angle=np.radians(0.5), wavelength=550e-6):
    """A Newtonian telescope: a paraboloid (or another conic) primary.

    The flat diagonal only folds the beam and is left out.

    Parameters
    ----------
    D : float
        Aperture in mm.
    F : float
        Focal ratio f/D.
    conic : float, optional
        Conic constant of the primary (-1: paraboloid, 0: sphere).
    field_angle : float, optional
        Maximum field angle in radians.
    wavelength : float, optional
        Wavelength in mm.

    Returns
    -------
    system : raytrace.System
        The primary with the image plane in its paraxial focus.
    """

    f = F*D
    s = [rt.Surface(R=-2*f, thickness=-f, conic=conic, mirror=True, semi_diameter=D/2,
                    label="primary")]
    return rt.System(s, 0, wavelength, field_angle=field_angle)


def two_mirror_telescope(D, f1, f, b, kind="rc", conics=None, field_angle=np.radians(0.25),
                         wavelength=550e-6):
    """A Cassegrain type telescope: classical Cassegrain or Ritchey-Chrétien.

    Parameters
    ----------
    D : float
        Aperture in mm.
    f1 : float
        Focal length of the primary.
    f : float
        System focal length.
    b : float
        Back focal distance behind the primary vertex.
    kind : str, optional
        "rc" (Ritchey-Chrétien, `rc_conics()`) or "classical" (paraboloid
        primary, `cassegrain_conic()`). Ignored if conics are given.
    conics : (K1, K2), optional
        Conic constants (e.g. a Dall-Kirkham or a misfigured mirror).
    field_angle : float, optional
        Maximum field angle in radians.
    wavelength : float, optional
        Wavelength in mm.

    Returns
    -------
    system : raytrace.System
        Primary (stop) and secondary, image plane in the paraxial focus.
    """

    g = two_mirror_geometry(f1, f, b)
    if conics is None:
        if kind == "rc":
            conics = rc_conics(g.m, g.beta)
        elif kind == "classical":
            conics = (-1.0, cassegrain_conic(g.m))
        else:
            raise ValueError("kind must be 'rc' or 'classical'")
    s = [rt.Surface(R=g.R1, thickness=-g.d, conic=conics[0], mirror=True, semi_diameter=D/2,
                    label="primary"),
         rt.Surface(R=g.R2, thickness=g.q, conic=conics[1], mirror=True, label="secondary")]
    return rt.System(s, 0, wavelength, field_angle=field_angle)


def schmidt_plate(D, F, n, k=1.5):
    """Thin Schmidt corrector for a spherical mirror, as a surface sag.

    The plate thickness t(r) = (r^4 - k r_m^2 r^2)/(4 (n - 1) R^3) + const,
    with R = 2 f = 2 F D and r_m = D/2, cancels the third order spherical
    aberration of the mirror; k sets the neutral zone (where the plate is
    thickest) at r = r_m sqrt(k/2). On the front surface of a plate with a
    flat back the sag is z = -t, i.e. the vertex radius
    R_p = 2 (n - 1) R^3/(k r_m^2) and A4 = -1/(4 (n - 1) R^3).

    Parameters
    ----------
    D : float
        Aperture (plate diameter).
    F : float
        Focal ratio of the mirror.
    n : float
        Refractive index of the plate at the design wavelength.
    k : float, optional
        Neutral zone parameter (1.5 minimizes the chromatic blur, 0 gives a
        plate that is thinnest at the centre).

    Returns
    -------
    R_p : float
        Vertex radius of the aspheric front surface, np.inf for k = 0.
    A4 : float
        Coefficient of r^4 of its sag.
    """

    R = 2*F*D
    rm = D/2
    A4 = -1/(4*(n - 1)*R**3)
    if k == 0:
        return np.inf, A4
    return 2*(n - 1)*R**3/(k*rm**2), A4


def schmidt_camera(D, F, material=(1.5168, 64.17), k=1.5, plate_thickness=10.0,
                   mirror_diameter=None, field_angle=np.radians(3.0), wavelength=550e-6, refine=2):
    """A Schmidt camera: aspheric corrector plate at the centre of curvature
    of a spherical mirror.

    The plate is the stop; its front surface carries the asphere of
    `schmidt_plate()` (for the index at `wavelength`), its back is flat.
    Optionally, the higher order terms are then fitted to the traced
    on-axis wavefront.
    The mirror (radius R = -2 F D) is R behind the plate, and the image
    plane lies in the paraxial focus, about f in front of the mirror. The
    focal surface is curved (radius f); use `best_focus()` or
    `System.with_image_distance()` for points off the axis.

    Parameters
    ----------
    D : float
        Aperture.
    F : float
        Focal ratio f/D.
    material : tuple or callable, optional
        Plate glass, (n_d, V_d) or n(wavelength_nm) (default N-BK7 like).
    k : float, optional
        Neutral zone parameter of the plate.
    plate_thickness : float, optional
        Thickness of the plate at the centre.
    mirror_diameter : float, optional
        Clear diameter of the mirror (default: unlimited, no vignetting).
    field_angle : float, optional
        Maximum field angle in radians.
    wavelength : float, optional
        Design wavelength in mm.
    refine : int, optional
        Number of corrections of the plate's A4, A6 and A8 from the traced
        on-axis OPD (default 2; 0 keeps the thin plate formula, which
        leaves the fifth and higher order spherical aberration of a fast
        mirror). The r^2 term, and with it the neutral zone, is kept.

    Returns
    -------
    system : raytrace.System
        Plate (2 surfaces), mirror and image plane in the paraxial focus.
    """

    f = F*D
    R = 2*f
    plate = rt.Surface(R=np.inf, thickness=plate_thickness, material=material, semi_diameter=D/2)
    probe = rt.System([plate, rt.Surface(thickness=1.0, n_after=1.0)], 0, wavelength)
    n = probe.n[1]
    R_p, A4 = schmidt_plate(D, F, n, k)
    sd_m = None if mirror_diameter is None else mirror_diameter/2
    s = [rt.Surface(R=R_p, thickness=plate_thickness, material=material, aspheric=(A4,),
                    semi_diameter=D/2, label="corrector"),
         rt.Surface(R=np.inf, thickness=R - plate_thickness, n_after=1.0),
         rt.Surface(R=-R, thickness=-f, mirror=True, semi_diameter=sd_m, label="mirror")]
    sys = rt.System(s, 0, wavelength, field_angle=field_angle)
    sys = sys.with_image_distance(sys.paraxial().image_distance)
    for _ in range(refine):
        sys = _refine_plate(sys, D/2)
    return sys


def _onaxis_polynomial(sys):
    """Fit of the on-axis OPD (length units) with 1, r^2, r^4, r^6, r^8."""
    m = sys.opd(0.0, 64, units="length")
    PX, PY = np.meshgrid(m.px, m.py)
    r2 = PX**2 + PY**2
    ok = np.isfinite(m.W)
    A = np.column_stack([r2[ok]**j for j in range(5)])
    return np.linalg.lstsq(A, m.W[ok], rcond=None)[0]


def _refine_plate(sys, rm):
    """One correction of A4, A6, A8 of the plate from the traced on-axis OPD."""
    c = _onaxis_polynomial(sys)
    n = sys.n[1]
    s0 = sys.surfaces[0]
    asp = list(s0.aspheric) + [0.0]*(3 - len(s0.aspheric))
    for j, cj in enumerate(c[2:]):
        asp[j] += cj/((n - 1)*rm**(4 + 2*j))
    sys.surfaces[0].aspheric = tuple(asp)
    return sys.with_image_distance(sys.paraxial().image_distance)


MirrorAberrations = namedtuple("MirrorAberrations", ["W040", "W131", "W222", "ACC", "AAS",
                                                     "petzval_radius", "sagittal_radius",
                                                     "medial_radius", "tangential_radius"])
MirrorAberrations.__doc__ = """Third order aberrations of a conic mirror, stop at the mirror.

Attributes
----------
W040, W131, W222 : float
    Wavefront coefficients at the rim of the pupil (length units), with the
    raytrace sign convention (OPD < 0 for undercorrected spherical
    aberration).
ACC : float
    Angular tangential coma 3 W131/y (radians).
AAS : float
    Angular astigmatism 2 W222/y, the blur diameter at the medial focus
    (radians, negative as in Schroeder).
petzval_radius, sagittal_radius, medial_radius, tangential_radius : float
    Radii of the image surfaces in the raytrace convention (inf: flat).
"""


def mirror_aberrations(D, f, K, theta):
    """Closed form third order aberrations of a conic mirror, stop at the mirror.

    Parameters
    ----------
    D : float
        Aperture.
    f : float
        Focal length.
    K : float
        Conic constant.
    theta : float
        Field angle in radians.

    Returns
    -------
    MirrorAberrations
        W040 = -(1 + K) y^4/(32 f^3), W131 = y^3 theta/(4 f^2),
        W222 = -y^2 theta^2/(2 f), angular coma 3 theta/(16 F^2),
        astigmatism -theta^2/(2F); image surfaces: Petzval -f, sagittal
        flat, medial f, tangential f/2 (raytrace convention: the mirror
        faces -z, so positive radii curve towards it).
    """

    y = D/2
    W040 = -(1 + K)*y**4/(32*f**3)
    W131 = y**3*theta/(4*f**2)
    W222 = -y**2*theta**2/(2*f)
    return MirrorAberrations(W040, W131, W222, 3*W131/y, 2*W222/y, -f, np.inf, f, f/2)


TwoMirrorAberrations = namedtuple("TwoMirrorAberrations", ["ACC", "AAS", "petzval_radius",
                                                           "medial_radius", "conics"])
TwoMirrorAberrations.__doc__ = """Closed form third order field aberrations of a two-mirror telescope.

Attributes
----------
ACC : float
    Angular tangential coma (radians); 0 for the Ritchey-Chrétien.
AAS : float
    Angular astigmatism, blur diameter at the medial focus (radians).
petzval_radius : float
    Radius of the Petzval surface (raytrace convention).
medial_radius : float
    Radius of the medial image surface (the best focus over the field).
conics : (K1, K2)
    The conic constants.
"""


def two_mirror_aberrations(D, f1, f, b, theta, kind="rc"):
    """Closed form third order field aberrations of a Cassegrain telescope
    (Schroeder 2000, table 6.4), stop at the primary.

    Parameters
    ----------
    D : float
        Aperture.
    f1, f, b : float
        Primary focal length, system focal length, back focal distance.
    theta : float
        Field angle in radians.
    kind : str, optional
        "rc" or "classical".

    Returns
    -------
    TwoMirrorAberrations
        Coma, astigmatism, Petzval and medial radii.
    """

    g = two_mirror_geometry(f1, f, b)
    m, beta = g.m, g.beta
    F = f/D
    if kind == "rc":
        ACC = 0.0
        A = -1/(2*F)*(m*(2*m + 1) + beta)/(2*m*(1 + beta))
        conics = rc_conics(m, beta)
    elif kind == "classical":
        ACC = 3*theta/(16*F**2)
        A = -1/(2*F)*(m**2 + beta)/(m*(1 + beta))
        conics = (-1.0, cassegrain_conic(m))
    else:
        raise ValueError("kind must be 'rc' or 'classical'")
    c_P = -(2/g.R1 - 2/g.R2)
    y = D/2
    c_A = A/y
    c_m = c_P + 2*c_A
    return TwoMirrorAberrations(ACC, A*theta**2, 1/c_P, np.inf if c_m == 0 else 1/c_m, conics)


def schmidt_chromatic_blur(F, n, dn):
    """Angular diameter of the chromatic blur of a Schmidt camera.

    For a plate with the neutral zone at 0.866 of the aperture (k = 3/2)
    the residual at another wavelength is (dn/(n - 1)) times the plate's
    aberration, a (4 rho^3 - 3 rho) slope that stays within its value at the
    rim; the blur at the design focus has the angular diameter ::

        dn/(128 (n - 1) F^3).

    Parameters
    ----------
    F : float
        Focal ratio.
    n : float
        Plate index at the design wavelength.
    dn : float
        Index difference to the design wavelength.

    Returns
    -------
    beta : float
        Angular diameter in radians.
    """

    return np.abs(dn)/(128*(n - 1)*F**3)


Spot = namedtuple("Spot", ["x", "y", "rms", "centroid", "px", "py"])
Spot.__doc__ = """Spot diagram relative to the chief ray.

Attributes
----------
x, y : arrays
    Image plane intersections relative to the chief ray.
rms : float
    RMS radius about the centroid.
centroid : (float, float)
    Centroid relative to the chief ray.
px, py : arrays
    Pupil coordinates of the rays.
"""


def spot(system, field=None, obscuration=0.0, n_rings=12, n_per_ring=6):
    """Spot diagram over an annular pupil (with a central obscuration).

    Rays on equally spaced rings between the obscuration and the rim, with
    a number of rays proportional to the radius (about n_per_ring*i on ring
    i of an unobscured pupil, hexapolar sampling), so that every ray
    stands for the same pupil area.

    Parameters
    ----------
    system : raytrace.System
        The telescope.
    field : float or pair, optional
        Field (default: the maximum field).
    obscuration : float, optional
        Diameter of the central obscuration relative to the pupil.
    n_rings : int, optional
        Number of rings.
    n_per_ring : int, optional
        Rays on the first ring.

    Returns
    -------
    Spot
        Intersections, RMS radius and centroid.
    """

    px, py = [], []
    e = obscuration
    for i in range(1, n_rings + 1):
        r = e + (1 - e)*(i - 0.5)/n_rings
        k = max(3, int(round(n_per_ring*n_rings*r)))
        a = 2*np.pi*(np.arange(k) + 0.5)/k
        px.append(r*np.cos(a))
        py.append(r*np.sin(a))
    px, py = np.concatenate(px), np.concatenate(py)
    bundle = system.trace_pupil(field, px, py)
    chief = system.trace_pupil(field, 0.0, 0.0)
    I = system.to_image_frame(bundle.points[-1])
    Ic = system.to_image_frame(chief.points[-1])[0]
    ok = bundle.valid
    x, y = I[ok, 0] - Ic[0], I[ok, 1] - Ic[1]
    c = (x.mean(), y.mean())
    rms = np.sqrt(np.mean((x - c[0])**2 + (y - c[1])**2))
    return Spot(x, y, rms, c, px[ok], py[ok])


def best_focus(system, field=None, obscuration=0.0, search=1.0, n=41, iterations=3):
    """Image distance with the smallest RMS spot at a field.

    Parameters
    ----------
    system : raytrace.System
        The telescope.
    field : float or pair, optional
        Field (default: the maximum field).
    obscuration : float, optional
        Central obscuration, see `spot()`.
    search : float, optional
        Half width of the first search interval around the current image
        plane (length units).
    n : int, optional
        Samples per search.
    iterations : int, optional
        Number of refinements (each one narrows the interval tenfold).

    Returns
    -------
    dz : float
        Shift of the image plane (signed z, along the axis) from the current
        position.
    rms : float
        RMS spot radius there.
    """

    t0 = system.surfaces[-1].thickness
    centre, half = 0.0, search
    for _ in range(iterations):
        dzs = centre + np.linspace(-half, half, n)
        rms = [spot(system.with_image_distance(t0 + dz), field, obscuration, 6, 6).rms for dz in dzs]
        k = int(np.argmin(rms))
        centre, half = dzs[k], 2*half/(n - 1)*2
    return centre, spot(system.with_image_distance(t0 + centre), field, obscuration).rms
