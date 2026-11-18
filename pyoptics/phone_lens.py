#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Smartphone camera lenses: plastics, operands and a five element design.

A phone main camera lens is a stack of four to seven injection moulded
plastic elements in front of a sensor a few millimetres across. It has to
be short (the total track TTL of 1.3 to 1.6 times the image height), fast
(f/1.5 to f/2) and wide (about 35 to 40 degrees half field), and it may
not send the light steeply onto the sensor, whose micro lenses and pixel
stacks accept chief ray angles (CRA) of about 30 to 35 degrees at the
corner. Only two families of optical plastics are available: cyclo-olefin
polymers (n_d about 1.53 to 1.545, V_d about 56) for the positive elements
and polycarbonate or polyester resins (n_d about 1.58 to 1.67, V_d 19 to
30) as the flint. What makes such a lens possible are the aspheres: every
surface is aspheric, and the last element, which sits close to the sensor
and acts as a field flattener and a "telecentricity corrector", has a
strongly curved centre and a rim that bends back, with inflection points.

This module holds what notebook 159
(phone camera lens with Forbes Q polynomials) needs:

- `PLASTICS`: model glasses (n_d, V_d) of the two plastic families
  (nominal values of cyclo-olefin polymers of the E48R/APL5014 kind and of
  the polyester/polycarbonate flints; no vendor data are reproduced).
- Operands and constraint functions for `lens_optimization`:
  `distortion()` (against f tan theta), `chief_ray_angle()`,
  `lateral_colour()` and `min_thickness()` (thickness between two
  surfaces sampled from the axis to the rim, which catches the thin
  zones of strongly aspheric elements that the paraxial edge thickness
  misses); `clear_apertures()` gives the real ray semi apertures.
- `phone_lens()`: the five element f/1.8 design of the notebook as Q-bfs
  surfaces, optimized for infinity and 100 mm together (`DESIGN`), or
  the earlier design for infinity only (`DESIGN_INFINITY`);
  `close_focus()` focuses it on a near object by moving the whole lens,
  and `to_monomials()` refits it as even aspheres.

Lengths in mm, angles in radians unless the name says degrees.
"""

import numpy as np

from . import lens_optimization as lo
from . import qaspheres as qa
from . import raytrace as rt

__all__ = ["PLASTICS", "IR_FILTER", "WAVELENGTHS", "distortion", "chief_ray_angle",
           "lateral_colour", "clear_apertures", "min_thickness", "phone_lens",
           "lens_surfaces", "to_monomials", "close_focus", "DESIGN", "DESIGN_INFINITY",
           "CLOSE_DISTANCE"]

# Nominal (n_d, V_d) of the two families of optical plastics
PLASTICS = {
    "COP": (1.531, 56.0),     # cyclo-olefin polymer (E48R kind)
    "COC": (1.544, 56.0),     # cyclo-olefin copolymer (APL5014 kind)
    "PC": (1.585, 29.9),      # polycarbonate
    "OKP": (1.632, 23.4),     # fluorene polyester (OKP kind)
}
# IR cut filter: a thin borosilicate plate
IR_FILTER = (1.5168, 64.2)

# Design wavelengths: F, d and C lines [mm]
WAVELENGTHS = (486.1e-6, 587.6e-6, 656.3e-6)


def _chief(system, fraction):
    return lo._chief(system, fraction*system.max_field)


def distortion(fields=(0.3, 0.5, 0.7, 0.85, 1.0)):
    """Function system -> relative distortion y/(f tan theta) - 1 of the
    real chief ray at fractions of the maximum field (for a Constraint).

    Parameters
    ----------
    fields : sequence, optional
        Fractions of the maximum field angle.

    Returns
    -------
    callable
    """

    def f(system):
        efl = system.paraxial().efl
        return np.array([_chief(system, fr)[0][1]/(efl*np.tan(fr*system.max_field)) - 1
                         for fr in fields])
    f.__name__ = "distortion"

    return f


def chief_ray_angle(fields=(0.5, 0.7, 0.85, 1.0)):
    """Function system -> angle of the real chief ray with the axis at the
    image (degrees), at fractions of the maximum field.

    Parameters
    ----------
    fields : sequence, optional

    Returns
    -------
    callable
    """

    def f(system):
        out = []
        for fr in fields:
            D = _chief(system, fr)[1]
            out.append(np.degrees(np.arctan2(np.hypot(D[0], D[1]), D[2])))
        return np.array(out)
    f.__name__ = "cra"

    return f


def lateral_colour(fields=(0.5, 0.7, 1.0), wavelengths=(WAVELENGTHS[0], WAVELENGTHS[2]),
                   tolerance=1e-3):
    """Operand: difference of the chief ray heights at two wavelengths
    (lateral colour), divided by the tolerance.

    Parameters
    ----------
    fields : sequence, optional
    wavelengths : (short, long), optional
    tolerance : number, optional
        In length units (default 1 um).

    Returns
    -------
    callable
    """

    def op(system):
        a = lo._at_wavelength(system, wavelengths[0])
        b = lo._at_wavelength(system, wavelengths[1])
        return np.array([(_chief(a, fr)[0][1] - _chief(b, fr)[0][1])/tolerance for fr in fields])

    return op


def clear_apertures(system, fields=(0.0, 0.5, 0.7, 0.85, 1.0), n=33, margin=1.0):
    """Real ray semi apertures of all surfaces: the largest ray height of
    the rim rays of the pupil at the fields, times margin.

    Parameters
    ----------
    system : raytrace.System
    fields : sequence, optional
        Fractions of the maximum field.
    n : int, optional
        Rim rays per field.
    margin : number, optional

    Returns
    -------
    array
        One radius per surface (0 where no ray arrives).
    """

    m = len(system.surfaces)
    r = np.zeros(m)
    a = np.linspace(0, 2*np.pi, n)
    for fr in fields:
        b = system.trace_pupil(fr*system.max_field, np.sin(a), np.cos(a), clip=False)
        h = np.hypot(b.points[1:m + 1, :, 0], b.points[1:m + 1, :, 1])
        r = np.maximum(r, np.max(np.where(np.isfinite(h), h, -1.0), axis=1))

    return np.maximum(r, 0.0)*margin


def min_thickness(surfaces, rims, n=24):
    """Function system -> smallest axial distance between surface i and
    i + 1 for each i in surfaces, sampled from the axis to the larger of
    their rims, each surface continued as a flat flange beyond its own rim
    (for a Constraint).

    Parameters
    ----------
    surfaces : sequence of int
    rims : sequence
        Semi aperture of every surface (`clear_apertures()`).
    n : int, optional
        Samples along the radius.

    Returns
    -------
    callable
    """

    def f(system):
        S = system.surfaces
        out = []
        for i in surfaces:
            r = np.linspace(0, max(rims[i], rims[i + 1]), n)
            # beyond its own rim a surface continues as a flat flange
            ra, rb = np.minimum(r, rims[i]), np.minimum(r, rims[i + 1])
            t = S[i].thickness + S[i + 1].sag(rb**2) - S[i].sag(ra**2)
            out.append(np.min(t) if np.all(np.isfinite(t)) else -1.0)
        return np.array(out)
    f.__name__ = "thickness"

    return f


# The designs of notebook 159 (phone camera lens with Forbes Q polynomials):
# f = 4 mm, f/1.8, image height 3 mm (36.9 degrees half field), stop on the
# first surface. Per lens surface: radius of the best fit sphere, thickness
# behind it, normalization radius and Q-bfs coefficients a_0... [mm]; the
# material behind it. DESIGN_INFINITY was optimized for an object at
# infinity only.
DESIGN_INFINITY = (
    (1.736534522, 0.6942330612, 1.15, (-0.0349623726, 0.0110453317, -0.0076436966, 0.00789869534, -0.00833045177, 0.00547031863, -0.00368482575, 0.00178887284),
     PLASTICS["COP"]),
    (-49.23207608, 0.1006915118, 1.15, (-0.0290756888, 0.00576805075, -0.0259554759, 0.0227734083, -0.0160440975, 0.00808290163, -0.00300552421, 0.000535078087),
     None),
    (25.36262006, 0.25, 1.15, (0.112469956, -0.013305263, -0.00753084217, 0.00425992235, 0.00298835042, -0.00531807631, 0.00356968968, -0.00118822454),
     PLASTICS["OKP"]),
    (2.047625519, 0.2443185225, 1.15, (-0.0543681367, 0.0298387684, -0.00411957966, -0.00189554561, 0.00749470694, -0.00599810718, 0.00244922518, -0.000438140251),
     None),
    (5.124912532, 0.534668617, 1.25, (-0.0248587793, -0.0430557735, 0.0369934837, -0.0121274221, 0.00569602713, -0.00424957472, 0.00209020276, -0.000460818479),
     PLASTICS["COP"]),
    (12.56990677, 0.5386191452, 1.4, (-0.150925818, -0.010850284, 0.0137125379, 0.0231731332, -0.0302705171, 0.0176934139, -0.0058856696, 0.00101536031),
     None),
    (-6.8165732, 1.360852281, 1.55, (0.275372124, 0.00474561045, -0.0140190038, -0.0109419877, 0.0243893438, -0.0193022081, 0.010179815, -0.00323730589),
     PLASTICS["COP"]),
    (-2.090668297, 0.03166018844, 1.9, (-0.613281538, 0.311951056, -0.0661901353, 0.0747772774, 0.140629526, 0.00702632912, 0.0457228581, 0.024885865, -0.00956363615, 0.0138327194),
     None),
    (-2.770967334, 0.8134182669, 2.2, (-0.754396244, 0.235998632, 0.161987357, 0.0997922236, 0.0928220302, 0.135405865, 0.0281421529, 0.0606096384, 0.00327330054, 0.000837625288),
     PLASTICS["COP"]),
    (111.5548828, 0.2607692031, 2.9, (2.23676352, -0.501585057, 0.373320897, 0.00176133391, 0.0566734885, 0.0352446014, -0.000685025694, 0.040473863, -0.0113689252, -0.0147499604),
     None),
    (np.inf, 0.21, None, (), IR_FILTER),
    (np.inf, 0.2607692031, None, (), None),
)

# DESIGN: the same lens polished for infinity and CLOSE_DISTANCE together
# (the lens moves away from the sensor to focus), with 21 fields from the
# axis to the corner, the rim of the pupil and the gap between L5 and the
# IR filter across the aperture in the merit function, then scaled to
# f = 4 mm. (With six fields the optimizer had hidden up to 1.7 waves
# between them.)
DESIGN = (
    (1.72324768, 0.7376464301, 1.14162, (-0.031326627, 0.00872610903, -0.00564880629, 0.00550699917, -0.00492730852, 0.0034655955, -0.00247089425, 0.00107149624),
     PLASTICS["COP"]),
    (-65.67508311, 0.08118598742, 1.14162, (-0.0297173444, 0.00433432297, -0.0253905693, 0.0215593289, -0.0131357384, 0.00739340995, -0.00310851938, 0.00135872982),
     None),
    (26.26031145, 0.2668537284, 1.14162, (0.109778685, -0.0135264109, -0.00801368315, 0.00521094894, 0.00404453923, -0.00424483018, 0.0041107008, -0.00162676005),
     PLASTICS["OKP"]),
    (2.019202286, 0.251272361, 1.14162, (-0.0520219401, 0.0301029068, -0.00391185758, -0.00011236511, 0.00734102273, -0.00655496518, 0.00477953708, -0.00203774668),
     None),
    (5.116483169, 0.5077234414, 1.24089, (-0.0236647927, -0.0410324538, 0.0395656319, -0.0109436667, 0.00440100036, -0.00295567762, 0.00251894848, -0.00135662908),
     PLASTICS["COP"]),
    (12.55308919, 0.5264580166, 1.38979, (-0.157319105, -0.00947657088, 0.0168082242, 0.0225923098, -0.0293409199, 0.0190557086, -0.00673231626, 0.00079511382),
     None),
    (-6.185412554, 1.251031967, 1.5387, (0.244928441, -0.00284880418, -0.00278592411, -0.00664373046, 0.0177191037, -0.0149270102, 0.00637248503, -0.0020448618),
     PLASTICS["COP"]),
    (-2.080091831, 0.05985433071, 1.88615, (-0.666900253, 0.312828321, -0.0561614354, 0.0611572194, 0.119006904, -0.014232915, 0.0293286951, 0.0173298083, -0.00250038734, 0.00910089003),
     None),
    (-2.721005592, 0.7992552563, 2.18396, (-0.767876546, 0.235679467, 0.176391208, 0.100074037, 0.067016628, 0.113061226, 0.019029025, 0.0392847156, -0.0183914678, 0.0208408181),
     PLASTICS["COP"]),
    (-10.92129804, 0.3441506786, 2.87885, (2.44109925, -0.444189001, 0.352477463, -0.0154331684, 0.11054673, 0.0171751673, 0.0285408378, 0.00241340118, 0.0105574318, -0.00100676544),
     None),
    (np.inf, 0.1529575156, None, (),
     IR_FILTER),
    (np.inf, 0.2832674459, None, (),
     None),
)

EFL, F_NUMBER, IMAGE_HEIGHT = 4.0, 1.8, 3.0
CLOSE_DISTANCE = 100.0
LENS_SURFACES = tuple(range(10))


def lens_surfaces():
    """Indices of the ten lens surfaces (the IR filter and image follow)."""

    return LENS_SURFACES


def phone_lens(design=None, wavelength=WAVELENGTHS[1]):
    """The five element f/1.8 phone camera lens of
    notebook 159 (phone camera lens with Forbes Q polynomials) as a `raytrace.System` of
    Q-bfs surfaces, an IR filter and the sensor.

    Parameters
    ----------
    design : sequence, optional
        Prescription rows (R_bfs, thickness, norm_radius, q, material); the
        default is `DESIGN`.
    wavelength : number, optional

    Returns
    -------
    raytrace.System
    """

    rows = DESIGN if design is None else design
    S = []
    for R, t, rn, q, mat in rows:
        kw = dict(n_after=1.0) if mat is None else dict(material=mat)
        if rn is None:
            S.append(rt.Surface(R, t, **kw))
        else:
            S.append(qa.QbfsSurface(R, t, q=q, norm_radius=rn, **kw))

    return rt.System(S, 0, wavelength, field_angle=np.arctan(IMAGE_HEIGHT/EFL),
                     entrance_pupil_radius=EFL/F_NUMBER/2)


def close_focus(system, distance=CLOSE_DISTANCE, shifts=np.linspace(0.0, 0.3, 61)):
    """The lens focused on an object at a finite distance by moving it as
    a whole away from the sensor (as the voice coil motor of a phone
    camera does).

    The object height is ``distance*tan(field angle)``, so that the image
    height stays about the same; the shift of the lens is the one of
    `shifts` with the smallest mean rms wavefront at 0, 0.5 and 0.85 field.

    Parameters
    ----------
    system : raytrace.System
        Focused at infinity (`phone_lens()`).
    distance : number, optional
        From the object to the first vertex [mm].
    shifts : array, optional
        Lens shifts to try [mm].

    Returns
    -------
    system : raytrace.System
    shift : float
        How far the lens moved [mm] (the paraxial estimate is
        f**2/(distance - f)).
    """

    s = system.copy()
    s.object_distance = distance
    s.max_field = distance*np.tan(system.max_field)
    best = None
    for dz in shifts:
        t = s.copy()
        t.surfaces[-1].thickness = system.surfaces[-1].thickness + dz
        with np.errstate(all="ignore"):
            w = np.mean([t.rms_opd(f*t.max_field, N=24) for f in (0.0, 0.5, 0.85)])
        if np.isfinite(w) and (best is None or w < best[0]):
            best = (w, dz, t)

    return best[2], float(best[1])


def to_monomials(system, orders=(4, 6, 8, 10, 12, 14, 16), surfaces=LENS_SURFACES, rims=None):
    """Copy of a system with its Q-bfs surfaces refit as even aspheres
    (the vertex curvature on a paraboloid, K = -1, which has a sag at every
    radius, plus A_4 r**4 ...) by least squares over their clear
    apertures.

    Parameters
    ----------
    system : raytrace.System
    orders : sequence of int, optional
    surfaces : sequence of int, optional
    rims : sequence, optional
        Fit radius per surface (default: the normalization radii).

    Returns
    -------
    other : raytrace.System
    errors : array
        Largest sag error of each fit.
    """

    other = system.copy()
    errors = []
    for i in surfaces:
        Q = system.surfaces[i]
        rim = Q.norm_radius if rims is None else rims[i]
        r = rim*np.sqrt((1 + np.cos(np.linspace(0, np.pi, 400)))/2)
        target = Q.sag(r**2) - qa.even_asphere_sag(r, Q.c, -1.0)
        M = np.stack([(r/rim)**k for k in orders], axis=1)
        a = np.linalg.lstsq(M, target, rcond=None)[0]
        A = tuple(ak/rim**k for ak, k in zip(a, orders))
        kw = dict(material=Q.material) if Q.material is not None else dict(n_after=Q.n_after)
        other.surfaces[i] = rt.Surface(1/Q.c, Q.thickness, conic=-1.0, aspheric=_pad(A, orders),
                                       semi_diameter=Q.semi_diameter, **kw)
        errors.append(np.max(np.abs(M @ a - target)))

    return other, np.array(errors)


def _pad(A, orders):
    """Coefficients A_4, A_6, ... in the order of `raytrace.Surface`."""

    out = [0.0]*((max(orders) - 4)//2 + 1)
    for a, k in zip(A, orders):
        out[(k - 4)//2] = a

    return tuple(out)
