#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Schematic drawings of optical layouts with matplotlib: thin lenses, stops,
prisms, mirrors, detectors and paraxial rays, and ready-made sketches of
microscopes, interferometers, wavefront sensors and other setups of the
example notebooks.

The drawings are 2D meridional sections with the optical axis along the
horizontal coordinate z and the height y vertical, in arbitrary layout
units. Rays are traced paraxially through thin lenses (paraxial_ray()), so
conjugate planes, foci and parallel beams in the sketches are where they
belong; the proportions are chosen for clarity, not taken from real
instruments.

Primitives (all draw onto a given matplotlib Axes):

- lens(), stop(), plane(), prism(), mirror(), detector(), point_source(),
  polarizer(), label(): optical elements and annotations.
- paraxial_ray(), draw_ray(): trace a ray through a sequence of thin lenses
  (in either direction along z) and draw it.

Microscope sketches used by the example notebooks 52 to 55:

- koehler_microscope(): a transmitted light microscope with Köhler
  illumination, the illumination and image forming ray paths and the two
  sets of conjugate planes.
- abbe_4f(): Abbe's picture of image formation, diffraction orders of a
  grating focused in the back focal plane, optionally filtered.
- phase_contrast_microscope(): condenser annulus and phase ring, direct and
  diffracted light.
- dic_microscope(): polarizer, Nomarski prisms, sheared beams, analyzer.
- confocal_microscope(): laser scanning confocal microscope with dichroic
  mirror and pinhole, in-focus and out-of-focus emission.
- fpm_microscope(): LED array microscope for Fourier ptychography with a
  bright field and a dark field LED.

Primitives for folded layouts: rotated_lens(), rotated_detector(),
beam_splitter(), laser(), sphere(), lenslet_array(), band() (a beam between
two edge rays), arrow(), wavefront().

Setup sketches for the other example notebooks:

- fold_mirror(): coated 45° fold mirror with s and p (notebook 14).
- twyman_green(): Twyman-Green interferometer with a PZT phase shifter
  (notebook 23).
- fizeau_interferometer(): Fizeau interferometer with a transmission
  sphere, confocal and cat's eye positions (notebook 24).
- shack_hartmann_sensor(), phase_diversity_setup(): two ways to measure a
  lens (notebook 25).
- holography_setup(): off-axis digital holography, lensless or with a
  microscope (notebooks 26 and 27).
- krf_illuminator(): the KrF illuminator with its pupil and field planes
  (notebook 42).
- spatial_filter(): beam expander with a pinhole that removes a ghost
  (notebook 56).
- vortex_interferometer(): Mach-Zehnder with a spiral phase plate, with a
  tilted or a spherical reference wave; mode_converter(): the pi/2
  cylindrical lens converter between HG and LG modes (notebook 69).
- axicon_bessel(): an axicon makes a Bessel zone, an obstacle in it casts a
  short shadow (notebook 68).
- cubic_phase_airy(): a cubic phase plate and a lens make an Airy beam,
  whose rays touch a parabolic caustic (notebook 68).
- young_extended_source(), vcz_setup(), stellar_interferometer(),
  michelson_interferometer(), hbt_interferometer(): Young's double pinhole
  with an extended source, the van Cittert-Zernike theorem, Michelson's
  stellar interferometer, a Michelson interferometer for temporal coherence
  and a Hanbury Brown-Twiss intensity interferometer (notebook 74).
- rainbow_observer(): sun behind the observer, the rain, the antisolar
  point and the cones of the primary and secondary bow (notebook 71).
- glory_aircraft(): the glory and the cloud bow around the shadow of an
  aircraft on a cloud deck (notebook 71).
- twinkling_star(): starlight through turbulent layers, shadow bands at
  the ground, an eye and a telescope; adaptive_optics_system(): telescope,
  deformable mirror, dichroic, Shack-Hartmann sensor, science camera and
  the real-time loop (notebook 72).
- wavefront_shaping_setup(): an SLM imaged onto a diffuser, a thick slab or
  a multimode fibre, a camera behind it and the feedback loop, optionally
  with the flat reference frame for measuring the transmission matrix
  (notebook 73).
- foucault_tester(): the Foucault knife-edge (or wire, or Ronchi) test of
  a concave mirror at its centre of curvature; schlieren_system(): a lens
  type Toepler schlieren system with a slit source and a knife edge, and a
  direct shadowgraph (notebook 76).
- sim_microscope(), sted_microscope(): structured illumination with a
  grating and an order mask, and a STED microscope with a vortex plate,
  dichroic mirrors and a pinhole (notebook 78).
- grin_rod_imaging(): sine rays in a GRIN rod, inverted and upright
  images at half and full pitch; road_mirage(): direct and bent rays from
  a car to the eye over a hot road, the inverted image (notebook 77).
- lensless_camera(): a pinhole camera, a coded aperture camera and a
  diffuser camera (notebook 79).
- oct_setup(): a time-domain and a spectral-domain optical coherence
  tomograph with a layered sample (notebook 82).
- optical_tweezers(): laser, beam expander overfilling a water immersion
  objective, sample chamber with a bead, condenser and quadrant photodiode;
  ashkin_rays(): two rays refracted by a sphere near the focus and the
  forces they exert; momentum_balance(): the force from the momentum flux
  through a sphere far away (notebook 75).
- pulse_through_lens(): a pulse through a singlet lags behind the phase
  front, most on the axis (Bor's pulse-front delay), not through an
  achromat; pulse_front_tilt(): the tilted pulse front behind a prism or a
  grating (notebook 81).
- approximation_map(): the levels of approximation from Maxwell's
  equations to rays and Fraunhofer diffraction and the modules for each;
  sign_conventions(): axes, radii, thicknesses, ray and field angles and
  the two mirror conventions; diffraction_geometry(): the Rayleigh-Sommerfeld
  integral; eikonal_wavefronts(): wavefronts of constant optical path and
  the rays normal to them (notebook 00).

Importing this module does not change matplotlib's settings; matplotlib is
imported when a function is called.

References
----------
A. Köhler, "Ein neues Beleuchtungsverfahren für mikrophotographische
Zwecke", Z. wiss. Mikrosk. 10, 433-440 (1893).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), section 8.6 (phase contrast).

D. Malacara (ed.), "Optical Shop Testing", 3rd ed., Wiley (2007), chapters 2
(Fizeau) and 14 (phase shifting).

J. Durnin, J. J. Miceli and J. H. Eberly, "Diffraction-free beams", Phys.
Rev. Lett. 58, 1499-1501 (1987).

G. A. Siviloglou, J. Broky, A. Dogariu and D. N. Christodoulides,
"Observation of accelerating Airy beams", Phys. Rev. Lett. 99, 213901
(2007).

J. Mertz, "Introduction to Optical Microscopy", 2nd ed., Cambridge
University Press (2019).

D. B. Murphy and M. W. Davidson, "Fundamentals of Light Microscopy and
Electronic Imaging", 2nd ed., Wiley-Blackwell (2012).

I. M. Vellekoop and A. P. Mosk, "Focusing coherent light through opaque
strongly scattering media", Opt. Lett. 32, 2309-2311 (2007).

G. S. Settles, "Schlieren and Shadowgraph Techniques", Springer (2001).
"""

import numpy as np


__all__ = ["lens", "stop", "plane", "prism", "mirror", "detector", "point_source",
           "polarizer", "label", "paraxial_ray", "draw_ray", "koehler_microscope",
           "abbe_4f", "phase_contrast_microscope", "dic_microscope",
           "confocal_microscope", "fpm_microscope", "rotated_lens", "rotated_detector",
           "beam_splitter", "laser", "sphere", "lenslet_array", "band", "arrow", "wavefront",
           "twyman_green", "fizeau_interferometer", "shack_hartmann_sensor",
           "phase_diversity_setup", "holography_setup", "fold_mirror", "krf_illuminator",
           "spatial_filter", "vortex_interferometer", "mode_converter",
           "axicon_bessel", "cubic_phase_airy", "young_extended_source", "vcz_setup",
           "stellar_interferometer", "michelson_interferometer", "hbt_interferometer",
           "rainbow_observer", "glory_aircraft", "twinkling_star", "adaptive_optics_system",
           "wavefront_shaping_setup", "foucault_tester", "schlieren_system",
           "sim_microscope", "sted_microscope",
           "grin_rod_imaging", "road_mirage", "lensless_camera", "oct_setup", "optical_tweezers", "ashkin_rays",
           "momentum_balance", "pulse_through_lens", "pulse_front_tilt",
           "approximation_map", "sign_conventions", "diffraction_geometry", "eikonal_wavefronts"]


GLASS = "#cfe3f3"
EDGE = "#4a6f8a"
METAL = "0.25"
ILLUMINATION = "#e0a000"
IMAGING = "#2a7ab9"
EXCITATION = "#1f77b4"
EMISSION = "#2ca02c"
STRAY = "#d62728"


def _patches():
    import matplotlib.patches as mpatches
    return mpatches


def lens(ax, z, h, thickness=None, color=GLASS, edgecolor=EDGE, text=None, text_y=None,
         fontsize=9, zorder=3):
    """Draw a thin biconvex lens of half height h at the position z.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
        Axial position.
    h : number
        Half height (semi diameter).
    thickness : number, optional
        Centre thickness of the drawn shape. Defaults to 0.18*h.
    color, edgecolor : colors, optional
    text : str, optional
        Label below the lens.
    text_y : number, optional
        Height of the label. Defaults to -1.15*h.
    fontsize : number, optional
    zorder : number, optional

    Returns
    -------
    patch : matplotlib Polygon
    """

    mpatches = _patches()
    t = 0.18*h if thickness is None else thickness
    yy = np.linspace(-h, h, 41)
    bulge = t/2*(1 - (yy/h)**2)
    pts = np.r_[np.c_[z + bulge, yy], np.c_[z - bulge[::-1], yy[::-1]]]
    patch = mpatches.Polygon(pts, closed=True, facecolor=color, edgecolor=edgecolor,
                             lw=1, zorder=zorder)
    ax.add_patch(patch)
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")
    return patch


def stop(ax, z, h, openings=((0.0, 0.5),), width=None, color=METAL, text=None, text_y=None,
         fontsize=9, zorder=4):
    """Draw an aperture stop (diaphragm) at z between -h and h.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
        Axial position.
    h : number
        Half height of the stop.
    openings : sequence of (r_inner, r_outer), optional
        Transparent zones as radii (abs(y) between r_inner and r_outer, on both
        sides of the axis). ((0, r),) is an iris of radius r, ((r1, r2),) an
        annulus. Defaults to ((0, 0.5),).
    width : number, optional
        Drawn thickness. Defaults to 0.04*h.
    color : color, optional
    text : str, optional
        Label below the stop.
    text_y : number, optional
    fontsize : number, optional
    zorder : number, optional
    """

    mpatches = _patches()
    w = 0.04*h if width is None else width
    edges = [0.0]
    for r1, r2 in sorted(openings):
        edges += [r1, r2]
    edges.append(h)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a <= 0:
            continue
        for sgn in (1, -1):
            y0 = a if sgn > 0 else -b
            ax.add_patch(mpatches.Rectangle((z - w/2, y0), w, b - a, facecolor=color,
                                            edgecolor="none", zorder=zorder))
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")


def plane(ax, z, h, text=None, color="0.55", ls=":", text_y=None, fontsize=9, **kwargs):
    """Draw a plane (e.g. a focal or conjugate plane) as a dotted line.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
    h : number
        Half height.
    text : str, optional
        Label above the line.
    color, ls : optional
    text_y : number, optional
        Label height. Defaults to 1.08*h.
    fontsize : number, optional
    **kwargs
        Passed to ax.plot().
    """

    ax.plot([z, z], [-h, h], color=color, ls=ls, lw=1, **kwargs)
    if text:
        label(ax, z, 1.08*h if text_y is None else text_y, text, fontsize=fontsize, va="bottom",
              color=color)


def prism(ax, z, h, width=None, text=None, text_y=None, fontsize=9, zorder=3):
    """Draw a Wollaston/Nomarski prism: a rectangle split along its diagonal.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
    h : number
        Half height.
    width : number, optional
        Defaults to 0.3*h.
    text : str, optional
    text_y : number, optional
    fontsize : number, optional
    zorder : number, optional
    """

    mpatches = _patches()
    w = 0.3*h if width is None else width
    ax.add_patch(mpatches.Polygon([(z - w/2, -h), (z + w/2, -h), (z - w/2, h)], closed=True,
                                  facecolor="#e8d9f3", edgecolor="#6a4c93", lw=1, zorder=zorder))
    ax.add_patch(mpatches.Polygon([(z + w/2, -h), (z + w/2, h), (z - w/2, h)], closed=True,
                                  facecolor="#d6c2ea", edgecolor="#6a4c93", lw=1, zorder=zorder))
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")


def mirror(ax, z, y, length, angle, color="0.3", lw=3, text=None, text_dy=None, fontsize=9,
           dichroic=False):
    """Draw a flat (or dichroic) mirror centred at (z, y), tilted by angle
    (radians against the y axis).

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
        Centre.
    length : number
    angle : number
        Tilt against the vertical, radians (pi/4: a 45° fold).
    color : color, optional
    lw : number, optional
    text : str, optional
    text_dy : number, optional
        Vertical offset of the label. Defaults to -0.7*length.
    fontsize : number, optional
    dichroic : bool, optional
        Draw a coated beam splitter (light blue) instead of a mirror.
    """

    dz, dy = length/2*np.sin(angle), length/2*np.cos(angle)
    ax.plot([z - dz, z + dz], [y - dy, y + dy], color="#7fb7d9" if dichroic else color,
            lw=lw, solid_capstyle="butt", zorder=4)
    if text:
        label(ax, z, y + (-0.7*length if text_dy is None else text_dy), text, fontsize=fontsize,
              va="top")


def detector(ax, z, h, text=None, width=None, color="0.35", text_y=None, fontsize=9):
    """Draw a detector (camera or photomultiplier) face at z.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
    h : number
        Half height of the sensitive area.
    text : str, optional
    width : number, optional
        Defaults to 0.12*h.
    color : color, optional
    text_y : number, optional
    fontsize : number, optional
    """

    mpatches = _patches()
    w = 0.12*h if width is None else width
    ax.add_patch(mpatches.Rectangle((z, -h), w, 2*h, facecolor=color, edgecolor="none", zorder=4))
    if text:
        label(ax, z + w/2, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")


def point_source(ax, z, y, color=ILLUMINATION, size=60, text=None, text_dy=-0.25, fontsize=9,
                 marker="*"):
    """Draw a point source (lamp filament, LED, fluorophore) at (z, y).

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
    color : color, optional
    size : number, optional
        Marker size.
    text : str, optional
    text_dy : number, optional
    fontsize : number, optional
    marker : str, optional
    """

    ax.scatter([z], [y], s=size, color=color, marker=marker, zorder=6, edgecolors="none")
    if text:
        label(ax, z, y + text_dy, text, fontsize=fontsize, va="top")


def polarizer(ax, z, h, direction="vertical", text=None, text_y=None, fontsize=9):
    """Draw a linear polarizer (a thin plate with its transmission direction).

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
    h : number
        Half height.
    direction : {"vertical", "out of plane", "45"}, optional
        Transmission direction symbol: a double arrow in the plane of the
        drawing, a dot (normal to it) or a cross for 45°.
    text : str, optional
    text_y : number, optional
    fontsize : number, optional
    """

    mpatches = _patches()
    ax.add_patch(mpatches.Rectangle((z - 0.03*h, -h), 0.06*h, 2*h, facecolor="#f2e6c9",
                                    edgecolor="#8c6d1f", lw=1, zorder=3))
    sym_y = 1.25*h
    if direction == "vertical":
        ax.annotate("", xy=(z, sym_y + 0.2*h), xytext=(z, sym_y - 0.2*h),
                    arrowprops=dict(arrowstyle="<->", color="#8c6d1f", lw=1.2))
    elif direction == "out of plane":
        ax.plot([z], [sym_y], "o", color="#8c6d1f", ms=4)
    else:
        ax.plot([z - 0.12*h, z + 0.12*h], [sym_y - 0.12*h, sym_y + 0.12*h], color="#8c6d1f", lw=1.2)
        ax.plot([z - 0.12*h, z + 0.12*h], [sym_y + 0.12*h, sym_y - 0.12*h], color="#8c6d1f", lw=1.2)
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")


def label(ax, z, y, text, fontsize=9, ha="center", va="center", color="0.15", **kwargs):
    """Write a label at (z, y)."""

    ax.text(z, y, text, fontsize=fontsize, ha=ha, va=va, color=color, **kwargs)


def paraxial_ray(lenses, z0, y0, u0, z_end):
    """Trace a paraxial ray through thin lenses.

    The ray starts at (z0, y0) with the slope u0 = dy/dz and runs towards
    z_end (in +z or -z direction); it is refracted by every lens (z_L, f)
    between z0 and z_end, in the order met. A converging lens (f > 0) bends
    the ray towards the axis in either direction of travel.

    Parameters
    ----------
    lenses : sequence of (z_L, f)
        Positions and focal lengths of thin lenses.
    z0, y0 : numbers
        Start point.
    u0 : number
        Start slope dy/dz.
    z_end : number
        End position.

    Returns
    -------
    z, y : 1d arrays
        Vertices of the ray (start, lenses, end).
    """

    direction = np.sign(z_end - z0) or 1.0
    met = sorted([(zl, f) for zl, f in lenses
                  if (zl - z0)*direction > 0 and (z_end - zl)*direction >= 0],
                 key=lambda e: e[0]*direction)
    zs, ys = [z0], [y0]
    z, y, u = z0, y0, u0
    for zl, f in met:
        y = y + u*(zl - z)
        z = zl
        zs.append(z)
        ys.append(y)
        # slope along the direction of travel: v = direction*u; lens: v' = v - y/f
        u = u - direction*y/f
    zs.append(z_end)
    ys.append(y + u*(z_end - z))
    return np.array(zs), np.array(ys)


def draw_ray(ax, lenses, z0, y0, u0, z_end, color=IMAGING, lw=1.2, ls="-", alpha=0.9,
             arrow=True, zorder=2):
    """Trace a paraxial ray with paraxial_ray() and draw it.

    Parameters
    ----------
    ax : matplotlib Axes
    lenses, z0, y0, u0, z_end
        See paraxial_ray().
    color, lw, ls, alpha : optional
        Line style.
    arrow : bool, optional
        Draw a small arrow head on the first segment to show the direction.
    zorder : number, optional

    Returns
    -------
    z, y : 1d arrays
        Vertices of the ray.
    """

    z, y = paraxial_ray(lenses, z0, y0, u0, z_end)
    ax.plot(z, y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    if arrow and len(z) > 1:
        zm, ym = (z[0] + z[1])/2, (y[0] + y[1])/2
        dz, dy = (z[1] - z[0])*0.01, (y[1] - y[0])*0.01
        ax.annotate("", xy=(zm + dz, ym + dy), xytext=(zm - dz, ym - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                    mutation_scale=9), zorder=zorder)
    return z, y


def _finish(ax, zlim, ylim, title=None):
    ax.set_xlim(*zlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def koehler_microscope(ax, rays="both", title=None):
    """Sketch of a transmitted light microscope with Köhler illumination.

    Lamp, collector, field diaphragm, condenser aperture diaphragm (in the
    front focal plane of the condenser), condenser, specimen, objective with
    its back focal plane (pupil), tube lens and camera. The illumination
    rays (from a point of the lamp filament) are focused in the aperture
    diaphragm and the back focal plane and pass the specimen as a plane
    wave; the image forming rays (from a point of the specimen) are focused
    on the camera. Aperture planes (filament, condenser aperture, back focal
    plane) and field planes (field diaphragm, specimen, camera) are labelled.

    Parameters
    ----------
    ax : matplotlib Axes
    rays : {"both", "illumination", "imaging"}, optional
        Which ray sets to draw. Defaults to "both".
    title : str, optional
    """

    z_lamp, z_col, f_col = 0.0, 1.5, 1.0
    z_fd = 2.6                             # imaged into the specimen by the condenser
    z_cond, f_cond = 5.5, 1.0
    z_ap = z_cond - f_cond                 # 4.5, the image of the filament
    z_spec = z_cond + 1/(1/f_cond - 1/(z_cond - z_fd))
    z_obj, f_obj = z_spec + 1.0, 1.0
    z_bfp = z_obj + f_obj
    z_tube, f_tube = z_bfp + 2.4, 2.4
    z_cam = z_tube + f_tube
    lenses = [(z_col, f_col), (z_cond, f_cond), (z_obj, f_obj), (z_tube, f_tube)]
    h = 0.95

    lens(ax, z_col, h, text="collector")
    stop(ax, z_fd, 1.1, openings=((0, 0.55),), text="field\ndiaphragm", text_y=-1.3)
    stop(ax, z_ap, 1.1, openings=((0, 0.45),), text="aperture\ndiaphragm", text_y=-1.3)
    lens(ax, z_cond, h, text="condenser")
    ax.plot([z_spec, z_spec], [-0.9, 0.9], color="#8fbf8f", lw=4, solid_capstyle="butt", zorder=3)
    label(ax, z_spec, -1.09, "specimen", va="top", fontsize=9)
    lens(ax, z_obj, 0.85, text="objective", text_y=-1.09)
    plane(ax, z_bfp, 0.9, text="back focal\nplane (pupil)")
    lens(ax, z_tube, h, text="tube lens")
    detector(ax, z_cam, 1.0, text="camera", text_y=-1.15)
    point_source(ax, z_lamp, 0.0, text="lamp", text_dy=-0.3)

    if rays in ("both", "illumination"):
        # a point of the filament: focused in the aperture diaphragm and the pupil
        for u in np.linspace(-0.3, 0.3, 5):
            draw_ray(ax, lenses, z_lamp, 0.0, u, z_cam, color=ILLUMINATION, lw=1.0, alpha=0.8)
    if rays in ("both", "imaging"):
        # a point of the specimen above the axis: focused on the camera
        for u in np.linspace(-0.3, 0.3, 5):
            draw_ray(ax, lenses, z_spec, 0.22, u, z_cam, color=IMAGING, lw=1.0)
    ax.plot([], [], color=ILLUMINATION, label="illumination (point of the filament)")
    ax.plot([], [], color=IMAGING, label="image forming rays (point of the specimen)")
    ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.0))
    for z_, txt in [(z_lamp, "A"), (z_ap, "A"), (z_bfp, "A"), (z_fd, "F"), (z_spec, "F"),
                    (z_cam, "F")]:
        label(ax, z_ + (0.06 if z_ == z_cam else 0), -1.95, txt, fontsize=9,
              color="#b35900" if txt == "A" else "#1f5f8b", weight="bold")
    label(ax, (z_lamp + z_cam)/2, -2.25, "A: aperture planes (conjugate to the filament)    "
          "F: field planes (conjugate to the specimen)", fontsize=8, color="0.3")
    _finish(ax, (-0.4, z_cam + 0.4), (-2.4, 1.6), title)


def abbe_4f(ax, period_orders=2, blocked=(), title=None, show_mask=True):
    """Abbe's picture: a grating diffracts a plane wave into orders that the
    objective focuses into points of its back focal plane; the image is the
    interference of the orders that pass.

    Parameters
    ----------
    ax : matplotlib Axes
    period_orders : int, optional
        Highest order m drawn (orders -m to m). Defaults to 2.
    blocked : sequence of int, optional
        Orders blocked by a mask in the back focal plane (drawn dashed after
        the mask). Defaults to none.
    title : str, optional
    show_mask : bool, optional
        Draw the mask pieces for the blocked orders. Defaults to True.
    """

    f = 2.0
    z_g, z_l1, z_bfp, z_l2, z_img = 0.0, f, 2*f, 3*f, 4*f
    lenses = [(z_l1, f), (z_l2, f)]
    h = 1.4
    lens(ax, z_l1, h, text="objective")
    lens(ax, z_l2, h, text="tube lens")
    plane(ax, z_bfp, 1.5, text="back focal plane")
    ax.plot([z_g, z_g], [-1.0, 1.0], color="0.2", lw=2, ls=(0, (1, 1.5)))
    label(ax, z_g, -1.15, "grating", va="top")
    plane(ax, z_img, 1.5, text="image plane")
    sin_1 = 0.44/max(period_orders, 1)
    colors = {0: "#444444", 1: "#2a7ab9", -1: "#2a7ab9", 2: "#9467bd", -2: "#9467bd",
              3: "#8c564b", -3: "#8c564b"}
    for z0 in np.linspace(-1.6, -0.2, 4):          # incident plane wave
        ax.annotate("", xy=(z_g - 0.02, 0.0 + (z0 + 0.9)), xytext=(-1.2, (z0 + 0.9)),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1, mutation_scale=8))
    for m in range(-period_orders, period_orders + 1):
        u = m*sin_1
        c = colors.get(m, "0.4")
        for y0 in (-0.5, 0.0, 0.5):
            z, y = paraxial_ray(lenses, z_g, y0, u, z_img)
            if m in blocked:
                # stopped by the mask in the back focal plane
                ax.plot(z[:2], y[:2], color=c, lw=1, alpha=0.8)
                ax.plot([z[1], z_bfp], [y[1], y[1] + (z_bfp - z[1])*(u - y[1]/f)],
                        color=c, lw=1, alpha=0.8)
            else:
                ax.plot(z, y, color=c, lw=1, alpha=0.8)
        y_f = f*u
        ax.plot([z_bfp], [y_f], "o", color=c, ms=5, zorder=5)
        ax.text(z_bfp + 0.12, y_f, f"{m:+d}" if m else "0", fontsize=8, va="center", color=c)
        if m in blocked and show_mask:
            ax.add_patch(_patches().Rectangle((z_bfp - 0.04, y_f - 0.12), 0.08, 0.24,
                                              facecolor=METAL, zorder=6))
    _finish(ax, (-1.3, z_img + 0.5), (-1.9, 1.8), title)


def phase_contrast_microscope(ax, title=None):
    """Sketch of a Zernike phase contrast microscope.

    An annulus in the front focal plane of the condenser illuminates the
    specimen with a cone of plane waves. The direct light (solid) is focused
    by the objective onto the phase ring in its back focal plane, an image
    of the annulus; light diffracted by the specimen (dashed) spreads over
    the pupil and mostly misses the ring.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    f_c, f_o, f_t = 1.2, 1.2, 2.0
    z_ann = 0.0
    z_cond = f_c
    z_spec = 2*f_c + 0.0
    z_obj = z_spec + f_o
    z_bfp = z_obj + f_o
    z_tube = z_bfp + 1.0
    z_img = z_tube + f_t
    lenses = [(z_cond, f_c), (z_obj, f_o), (z_tube, f_t)]
    r_ann = 0.65
    stop(ax, z_ann, 1.1, openings=((r_ann - 0.07, r_ann + 0.07),), text="condenser\nannulus",
         text_y=-1.3)
    lens(ax, z_cond, 1.0, text="condenser")
    ax.plot([z_spec, z_spec], [-0.9, 0.9], color="#8fbf8f", lw=4, solid_capstyle="butt", zorder=3)
    label(ax, z_spec, -1.09, "specimen", va="top")
    lens(ax, z_obj, 1.0, text="objective")
    # phase ring: a grey band in the pupil plane where the annulus is imaged
    plane(ax, z_bfp, 1.0)
    for sgn in (1, -1):
        ax.add_patch(_patches().Rectangle((z_bfp - 0.05, sgn*r_ann*f_o/f_c - 0.07), 0.1, 0.14,
                                          facecolor="#c49a3a", edgecolor="#7a5c12", zorder=5))
    label(ax, z_bfp, 1.15, "phase ring in the\nback focal plane", va="bottom")
    lens(ax, z_tube, 1.0, text="tube lens")
    detector(ax, z_img, 1.0, text="camera")
    # direct light from both sides of the annulus
    for y0 in (r_ann, -r_ann):
        for u in np.linspace(-0.25, 0.25, 3):
            draw_ray(ax, lenses, z_ann, y0, u, z_img, color=ILLUMINATION, lw=1.0, arrow=(u == 0))
    # light diffracted by a point of the specimen
    for u in (-0.55, -0.35, -0.15, 0.15, 0.35):
        draw_ray(ax, lenses, z_spec, 0.15, u, z_img, color=IMAGING, lw=1.0, ls="--", arrow=False)
    ax.plot([], [], color=ILLUMINATION, label="direct (undiffracted) light")
    ax.plot([], [], color=IMAGING, ls="--", label="light diffracted by the specimen")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.03))
    _finish(ax, (-0.4, z_img + 0.4), (-1.6, 1.7), title)


def dic_microscope(ax, shear=0.25, title=None):
    """Sketch of a Nomarski DIC microscope.

    Polarizer, Nomarski prism in the front focal plane of the condenser
    (splits each ray into two orthogonal polarizations with a small angle),
    condenser (turns the angle into a lateral shear in the specimen),
    objective, second Nomarski prism in its back focal plane (recombines the
    beams; its lateral position sets the bias), analyzer and camera. The
    shear is exaggerated.

    Parameters
    ----------
    ax : matplotlib Axes
    shear : number, optional
        Drawn shear in layout units. Defaults to 0.25.
    title : str, optional
    """

    f_c, f_o, f_t = 1.2, 1.2, 1.8
    z_pol = -1.0
    z_p1 = 0.0
    z_cond = f_c
    z_spec = 2*f_c
    z_obj = z_spec + f_o
    z_p2 = z_obj + f_o
    z_an = z_p2 + 1.0
    z_tube = z_an + 1.0
    z_img = z_tube + f_t
    lenses = [(z_cond, f_c), (z_obj, f_o), (z_tube, f_t)]
    polarizer(ax, z_pol, 0.9, direction="45", text="polarizer\n(45°)")
    prism(ax, z_p1, 0.9, text="Nomarski\nprism 1")
    lens(ax, z_cond, 1.0, text="condenser")
    ax.plot([z_spec, z_spec], [-0.9, 0.9], color="#8fbf8f", lw=4, solid_capstyle="butt", zorder=3)
    label(ax, z_spec, -1.09, "specimen", va="top")
    lens(ax, z_obj, 1.0, text="objective")
    prism(ax, z_p2, 0.9, text="Nomarski\nprism 2\n(shift: bias)")
    polarizer(ax, z_an, 0.9, direction="45", text="analyzer\n(-45°)")
    lens(ax, z_tube, 1.0, text="tube lens")
    detector(ax, z_img, 1.0, text="camera")
    a = shear/(2*f_c)                       # half split angle
    for y0 in (-0.5, 0.0, 0.5):
        # unpolarized/45° light before the first prism
        ax.plot([z_pol - 0.5, z_p1], [y0, y0], color="0.45", lw=1)
        # the prism splits the ray into two polarizations with the angles +-a;
        # the condenser turns the angle into a lateral offset (the shear),
        # the objective focuses both into the same point of the second prism
        for sgn, c in ((1, "#d62728"), (-1, "#1f77b4")):
            z, y = paraxial_ray(lenses, z_p1, y0, sgn*a, z_p2)
            ax.plot(z, y, color=c, lw=1, alpha=0.85)
        # recombined by the second prism: one ray onwards to the image
        y_p2 = -y0*f_o/f_c
        draw_ray(ax, lenses, z_p2, y_p2, 0.0, z_img, color="0.45", lw=1, arrow=False)
    # the shear in the specimen
    ax.annotate("", xy=(z_spec + 0.25, shear/2), xytext=(z_spec + 0.25, -shear/2),
                arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.8, mutation_scale=6))
    label(ax, z_spec + 0.32, 0.0, "shear", ha="left", fontsize=8, color="0.2")
    ax.plot([], [], color="#d62728", label="polarization 1")
    ax.plot([], [], color="#1f77b4", label="polarization 2 (sheared)")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.03))
    _finish(ax, (z_pol - 0.7, z_img + 0.4), (-1.75, 1.7), title)


def confocal_microscope(ax, out_of_focus=0.45, title=None):
    """Sketch of a laser scanning confocal fluorescence microscope.

    The laser beam is expanded, reflected by a dichroic mirror and focused
    by the objective into the specimen (the scan mirrors that move the focus
    are omitted). Fluorescence from the focus (green) returns through the
    objective, passes the dichroic mirror and is focused by the tube lens
    onto the pinhole in front of the detector. Fluorescence from a plane
    deeper in the specimen (red, dashed) is focused before the pinhole and
    mostly blocked by it.

    Parameters
    ----------
    ax : matplotlib Axes
    out_of_focus : number, optional
        Depth of the out-of-focus emitter below the focus (layout units).
        Defaults to 0.45.
    title : str, optional
    """

    z_pin = 0.0
    z_det = -0.9
    z_tube, f_t = 2.0, 2.0
    z_dic = 3.6
    z_obj, f_o = 6.6, 1.0
    z_foc = z_obj + f_o
    lenses_back = [(z_obj, f_o), (z_tube, f_t)]
    h = 0.85
    detector(ax, z_det, 0.5, text="detector\n(PMT)", text_y=-0.65)
    stop(ax, z_pin, 0.7, openings=((0, 0.04),), width=0.05, text="pinhole", text_y=-0.85)
    lens(ax, z_tube, h, text="tube lens")
    mirror(ax, z_dic, 0.0, 1.5, np.pi/4, dichroic=True, text="dichroic\nmirror", text_dy=-0.75)
    lens(ax, z_obj, h, text="objective")
    ax.add_patch(_patches().Rectangle((z_foc - 0.25, -1.0), 1.1, 2.0, facecolor="#e9f5e9",
                                      edgecolor="#8fbf8f", zorder=1))
    label(ax, z_foc + 0.3, -1.09, "thick specimen", va="top")
    # laser from above, expanded, folded by the dichroic towards the objective
    y_laser, y_exp, r_beam = 3.0, 2.2, 0.45
    ax.add_patch(_patches().Rectangle((z_dic - 0.25, y_laser), 0.5, 0.35, facecolor="0.3",
                                      zorder=4))
    label(ax, z_dic + 0.35, y_laser + 0.17, "laser", ha="left")
    ax.add_patch(_patches().Ellipse((z_dic, y_exp), 2.2*r_beam, 0.12, facecolor=GLASS,
                                    edgecolor=EDGE, zorder=3))
    label(ax, z_dic - 1.2*r_beam, y_exp, "beam\nexpander", ha="right", fontsize=8)
    for sgn in (-1, 1):
        ax.plot([z_dic + sgn*0.06, z_dic + sgn*r_beam], [y_laser, y_exp], color=EXCITATION, lw=1)
        # down to the dichroic (the line y = z - z_dic), then along +z to the objective
        zb = z_dic + sgn*r_beam
        ax.plot([zb, zb], [y_exp, sgn*r_beam], color=EXCITATION, lw=1)
        z, y = paraxial_ray([(z_obj, f_o)], zb, sgn*r_beam, 0.0, z_foc)
        ax.plot(z, y, color=EXCITATION, lw=1)
    point_source(ax, z_foc, 0.0, color=EMISSION, size=40, marker="o")
    label(ax, z_foc + 0.05, 0.2, "focus", ha="left", fontsize=8)
    for u in (-0.55, -0.3, 0.3, 0.55):
        draw_ray(ax, lenses_back, z_foc, 0.0, u, z_det, color=EMISSION, lw=1.0, arrow=False)
    z_oof = z_foc + out_of_focus
    point_source(ax, z_oof, 0.0, color=STRAY, size=30, marker="o")
    label(ax, z_oof + 0.1, -0.22, "out of\nfocus", ha="left", fontsize=8, color=STRAY)
    for u in (-0.4, -0.22, 0.22, 0.4):
        z, y = paraxial_ray(lenses_back, z_oof, 0.0, u, z_pin)
        ax.plot(z, y, color=STRAY, lw=1.0, ls="--", alpha=0.9)
    ax.plot([], [], color=EXCITATION, label="excitation laser")
    ax.plot([], [], color=EMISSION, label="fluorescence from the focus: passes the pinhole")
    ax.plot([], [], color=STRAY, ls="--", label="fluorescence from out of focus: blocked")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.0))
    _finish(ax, (z_det - 0.4, z_foc + 1.1), (-1.5, 3.6), title)


def fpm_microscope(ax, title=None):
    """Sketch of an LED array microscope for Fourier ptychography.

    A matrix of LEDs below the specimen replaces the condenser; one LED at a
    time illuminates the specimen with a plane wave from its direction. The
    low NA objective accepts the direct light of the central LEDs (bright
    field, orange) but not that of outer LEDs (dark field, red), whose
    direct light misses the pupil; only the light diffracted by the specimen
    to large angles reaches the camera then.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    z_led = 0.0
    z_spec = 2.8
    z_obj, f_o = 4.0, 1.0
    z_bfp = z_obj + f_o
    z_tube, f_t = 6.5, 2.2
    z_cam = z_tube + f_t
    lenses = [(z_obj, f_o), (z_tube, f_t)]
    leds = np.linspace(-1.6, 1.6, 9)
    for y in leds:
        ax.add_patch(_patches().Rectangle((z_led - 0.08, y - 0.08), 0.16, 0.16,
                                          facecolor="#f2f2f2", edgecolor="0.5", zorder=4))
    label(ax, z_led, 1.8, "LED matrix", va="bottom")
    ax.plot([z_spec, z_spec], [-0.9, 0.9], color="#8fbf8f", lw=4, solid_capstyle="butt", zorder=3)
    label(ax, z_spec, -1.05, "specimen", va="top")
    lens(ax, z_obj, 0.95, text="4x objective\n(low NA)", text_y=-1.05)
    stop(ax, z_bfp, 1.0, openings=((0, 0.18),), text="pupil", text_y=-1.15)
    lens(ax, z_tube, 0.9, text="tube lens")
    detector(ax, z_cam, 1.0, text="camera")
    # bright field LED (centre): plane wave nearly along the axis
    y_bf = leds[4]
    ax.add_patch(_patches().Rectangle((z_led - 0.08, y_bf - 0.08), 0.16, 0.16,
                                      facecolor=ILLUMINATION, edgecolor="0.3", zorder=5))
    for ys in (-0.35, 0.0, 0.35):
        u = (ys - y_bf)/(z_spec - z_led)
        ax.plot([z_led, z_spec], [y_bf, ys], color=ILLUMINATION, lw=1)
        draw_ray(ax, lenses, z_spec, ys, u, z_cam, color=ILLUMINATION, lw=1, arrow=False)
    # dark field LED: its direct light misses the pupil
    y_df = leds[1]
    ax.add_patch(_patches().Rectangle((z_led - 0.08, y_df - 0.08), 0.16, 0.16,
                                      facecolor=STRAY, edgecolor="0.3", zorder=5))
    for ys in (-0.35, 0.0, 0.2):
        u = (ys - y_df)/(z_spec - z_led)
        ax.plot([z_led, z_spec], [y_df, ys], color=STRAY, lw=1)
        z, y = paraxial_ray(lenses, z_spec, ys, u, z_bfp)
        ax.plot(z, y, color=STRAY, lw=1, ls="-")
    # light diffracted by the specimen back into the objective
    for u in (-0.12, 0.0):
        draw_ray(ax, lenses, z_spec, 0.0, u, z_cam, color=STRAY, lw=1, ls="--", arrow=False)
    ax.plot([], [], color=ILLUMINATION, label="bright field LED: direct light passes the pupil")
    ax.plot([], [], color=STRAY, label="dark field LED: direct light blocked by the pupil")
    ax.plot([], [], color=STRAY, ls="--", label="... only light diffracted by the specimen gets through")
    ax.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02), ncol=2)
    _finish(ax, (-0.5, z_cam + 0.4), (-1.9, 2.1), title)


# ---------------------------------------------------------------------------
# further primitives for folded layouts: interferometers, holography, sensors
# ---------------------------------------------------------------------------

LASER = "#d6604d"
REFERENCE = "#2a7ab9"
TEST = "#c0392b"


def _transform(ax, z, y, rotate):
    """Transform from local element coordinates (axis along z, centred at the
    origin) into data coordinates: rotation by rotate degrees, then a shift
    to (z, y)."""

    import matplotlib.transforms as mtransforms
    return mtransforms.Affine2D().rotate_deg(rotate).translate(z, y) + ax.transData


def _local_to_data(zl, yl, z, y, rotate):
    a = np.radians(rotate)
    return z + zl*np.cos(a) - yl*np.sin(a), y + zl*np.sin(a) + yl*np.cos(a)


def _side_label(ax, z, y, h, rotate, text, fontsize, text_offset=None):
    # a label "below" an element, rotated with it: below for an element across
    # a horizontal beam, to the right for one across a vertical beam
    d = 1.15*h if text_offset is None else text_offset
    zl, yl = _local_to_data(0.0, -d, z, y, rotate)
    if abs(np.sin(np.radians(rotate))) > 0.7:
        ha = "left" if zl > z else "right"
        label(ax, zl, yl, text, fontsize=fontsize, ha=ha, va="center")
    else:
        label(ax, zl, yl, text, fontsize=fontsize, va="top" if yl < y else "bottom")


def rotated_lens(ax, z, y, h, rotate=90, thickness=None, text=None, text_offset=None, fontsize=9,
                 color=GLASS, edgecolor=EDGE, zorder=3):
    """Draw a thin biconvex lens centred at (z, y) with its axis rotated by
    rotate degrees against z (90: the lens sits across a vertical beam).

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
        Centre.
    h : number
        Half height.
    rotate : number, optional
        Rotation of the lens axis in degrees. Defaults to 90.
    thickness : number, optional
        Defaults to 0.18*h.
    text : str, optional
        Label beside the lens.
    text_offset : number, optional
        Distance of the label from the centre. Defaults to 1.15*h.
    fontsize : number, optional
    color, edgecolor : colors, optional
    zorder : number, optional
    """

    mpatches = _patches()
    t = 0.18*h if thickness is None else thickness
    yy = np.linspace(-h, h, 41)
    bulge = t/2*(1 - (yy/h)**2)
    pts = np.r_[np.c_[bulge, yy], np.c_[-bulge[::-1], yy[::-1]]]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=color, edgecolor=edgecolor, lw=1,
                                  zorder=zorder, transform=_transform(ax, z, y, rotate)))
    if text:
        _side_label(ax, z, y, h, rotate, text, fontsize, text_offset)


def rotated_detector(ax, z, y, h, rotate=90, width=None, text=None, text_offset=None, fontsize=9,
                     color="0.35"):
    """Draw a detector face centred at (z, y), rotated by rotate degrees
    (90: facing a beam that travels along -y or +y).

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
    h : number
        Half size of the sensitive area.
    rotate : number, optional
        Defaults to 90.
    width : number, optional
        Drawn thickness. Defaults to 0.12*h.
    text : str, optional
    text_offset : number, optional
    fontsize : number, optional
    color : color, optional
    """

    mpatches = _patches()
    w = 0.12*h if width is None else width
    ax.add_patch(mpatches.Rectangle((-w/2, -h), w, 2*h, facecolor=color, edgecolor="none",
                                    zorder=4, transform=_transform(ax, z, y, rotate)))
    if text:
        _side_label(ax, z, y, h, rotate, text, fontsize, text_offset)


def beam_splitter(ax, z, y, size, orientation="/", cube=False, color="#9fc5e0", text=None,
                  text_dy=None, fontsize=9):
    """Draw a 50/50 beam splitter at (z, y), a plate or a cube, at 45°.

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
        Centre.
    size : number
        Length of the splitting face (plate) or edge length (cube).
    orientation : {"/", "\\\\"}, optional
        Direction of the splitting face. "/" reflects light travelling along
        +z into +y, "\\\\" reflects it into -y.
    cube : bool, optional
        Draw a cube instead of a plate.
    color : color, optional
    text : str, optional
    text_dy : number, optional
        Vertical offset of the label. Defaults to -0.75*size.
    fontsize : number, optional
    """

    mpatches = _patches()
    s = 1 if orientation == "/" else -1
    if cube:
        ax.add_patch(mpatches.Rectangle((z - size/2, y - size/2), size, size, facecolor=GLASS,
                                        edgecolor=EDGE, lw=1, zorder=3))
        d = size/2
    else:
        d = size/(2*np.sqrt(2))
        # a thin plate: a parallelogram along the diagonal
        w = 0.025*size
        pts = [(z - d - w, y - s*d + w*s), (z + d - w, y + s*d + w*s),
               (z + d + w, y + s*d - w*s), (z - d + w, y - s*d - w*s)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=GLASS, edgecolor=EDGE, lw=0.8,
                                      zorder=3))
    ax.plot([z - d, z + d], [y - s*d, y + s*d], color=color, lw=2.2, zorder=4,
            solid_capstyle="butt")
    if text:
        label(ax, z, y + (-0.75*size if text_dy is None else text_dy), text, fontsize=fontsize,
              va="top")


def laser(ax, z, y, length=1.0, height=0.3, rotate=0, text="laser", fontsize=9, color="0.3"):
    """Draw a laser head as a box whose output end is at (z, y).

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
        Position of the output aperture.
    length, height : numbers, optional
        Size of the box.
    rotate : number, optional
        Direction of the emitted beam in degrees against +z. Defaults to 0.
    text : str, optional
        Label inside the box.
    fontsize : number, optional
    color : color, optional
    """

    mpatches = _patches()
    ax.add_patch(mpatches.Rectangle((-length, -height/2), length, height, facecolor=color,
                                    edgecolor="none", zorder=4,
                                    transform=_transform(ax, z, y, rotate)))
    if text:
        zc, yc = _local_to_data(-length/2, 0.0, z, y, rotate)
        label(ax, zc, yc, text, fontsize=fontsize, color="white",
              rotation=rotate if abs(rotate) <= 90 else rotate - 180, zorder=5)


def sphere(ax, z_vertex, y, R, h, color="0.25", lw=2.0, ls="-", glass=None, text=None,
           text_y=None, fontsize=9, zorder=4):
    """Draw a spherical surface (a mirror or a lens surface) in section.

    Parameters
    ----------
    ax : matplotlib Axes
    z_vertex, y : numbers
        Vertex position; y is the height of the axis.
    R : number
        Radius of curvature, positive for a centre at z_vertex + R (a convex
        surface facing -z).
    h : number
        Half height (semi diameter), smaller than abs(R).
    color, lw, ls : optional
        Line style of the surface.
    glass : number, optional
        If given, the thickness of a glass body behind the surface (towards
        +z), drawn with a flat back.
    text : str, optional
    text_y : number, optional
    fontsize : number, optional
    zorder : number, optional

    Returns
    -------
    z, yy : 1d arrays
        The drawn surface.
    """

    yy = np.linspace(-h, h, 81)
    zz = z_vertex + R - np.sign(R)*np.sqrt(R**2 - yy**2)
    if glass is not None:
        z_back = z_vertex + glass
        pts = np.r_[np.c_[zz, y + yy], [[z_back, y + h], [z_back, y - h]]]
        ax.add_patch(_patches().Polygon(pts, closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1,
                                        zorder=zorder - 1))
    ax.plot(zz, y + yy, color=color, lw=lw, ls=ls, zorder=zorder)
    if text:
        label(ax, z_vertex, y - 1.15*h if text_y is None else text_y, text, fontsize=fontsize,
              va="top")
    return zz, y + yy


def lenslet_array(ax, z, h, n, focal_length=None, text=None, text_y=None, fontsize=9):
    """Draw a lenslet array of n lenslets between -h and h at z.

    Parameters
    ----------
    ax : matplotlib Axes
    z : number
    h : number
        Half height of the array.
    n : int
        Number of lenslets.
    focal_length : number, optional
        Not drawn; accepted for symmetry with the sketches that use it.
    text : str, optional
    text_y : number, optional
    fontsize : number, optional

    Returns
    -------
    centres : 1d array
        Heights of the lenslet centres.
    """

    p = 2*h/n
    centres = -h + p*(np.arange(n) + 0.5)
    ax.add_patch(_patches().Rectangle((z - 0.02*h, -h), 0.04*h, 2*h, facecolor=GLASS,
                                      edgecolor=EDGE, lw=0.8, zorder=3))
    for c in centres:
        yy = np.linspace(-p/2, p/2, 15)
        ax.plot(z + 0.02*h + 0.35*p*(1 - (2*yy/p)**2)/2, c + yy, color=EDGE, lw=0.8, zorder=3)
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")
    return centres


def band(ax, z_top, y_top, z_bottom, y_bottom, color=LASER, alpha=0.22, edges=True, lw=0.8,
         zorder=1):
    """Fill a beam between two edge rays (polylines).

    Parameters
    ----------
    ax : matplotlib Axes
    z_top, y_top, z_bottom, y_bottom : 1d arrays
        Vertices of the two edge rays, in the same direction of travel.
    color : color, optional
    alpha : number, optional
        Opacity of the fill.
    edges : bool, optional
        Draw the edge rays as lines too.
    lw : number, optional
    zorder : number, optional
    """

    pts = np.r_[np.c_[z_top, y_top], np.c_[z_bottom, y_bottom][::-1]]
    ax.add_patch(_patches().Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                                    alpha=alpha, zorder=zorder))
    if edges:
        ax.plot(z_top, y_top, color=color, lw=lw, zorder=zorder + 0.5)
        ax.plot(z_bottom, y_bottom, color=color, lw=lw, zorder=zorder + 0.5)


def arrow(ax, z0, y0, z1, y1, color="0.2", lw=1.0, style="-|>", text=None, text_pos=None,
          fontsize=8, ha="center", va="center", mutation_scale=9):
    """Draw an arrow from (z0, y0) to (z1, y1) with an optional label at
    text_pos (default: the middle)."""

    ax.annotate("", xy=(z1, y1), xytext=(z0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=mutation_scale, shrinkA=0, shrinkB=0))
    if text:
        zt, yt = ((z0 + z1)/2, (y0 + y1)/2) if text_pos is None else text_pos
        label(ax, zt, yt, text, fontsize=fontsize, ha=ha, va=va, color=color)


def wavefront(ax, z, y, h, amplitude=0.0, vertical=True, color="0.2", lw=1.3, n_waves=1.5,
              zorder=5):
    """Draw a wavefront across a beam: a straight line (amplitude 0) or a
    wavy one.

    Parameters
    ----------
    ax : matplotlib Axes
    z, y : numbers
        Centre.
    h : number
        Half width of the beam.
    amplitude : number, optional
        Deviation from a plane wavefront.
    vertical : bool, optional
        True for a beam along z (the wavefront is vertical), False for a beam
        along y.
    color, lw : optional
    n_waves : number, optional
        Number of undulations over the beam.
    zorder : number, optional
    """

    s = np.linspace(-h, h, 61)
    d = amplitude*(np.sin(np.pi*n_waves*s/h + 0.4) + 0.6*(s/h)**2 - 0.3)
    if vertical:
        ax.plot(z + d, y + s, color=color, lw=lw, zorder=zorder)
    else:
        ax.plot(z + s, y + d, color=color, lw=lw, zorder=zorder)


def _expander(ax, z_out, z_l1, f1, z_l2, f2, w, z_end, y=0.0, pinhole=True, color=LASER):
    """A Kepler beam expander (with a pinhole in the common focus) from the
    laser output at z_out to z_end along +z; returns the output half width."""

    lenses = [(z_l1, f1), (z_l2, f2)]
    zt, yt = paraxial_ray(lenses, z_out, w, 0.0, z_end)
    zb, yb = paraxial_ray(lenses, z_out, -w, 0.0, z_end)
    band(ax, zt, y + yt, zb, y + yb, color=color)
    lens(ax, z_l1, 1.7*w)
    lens(ax, z_l2, 1.15*w*f2/f1)
    if pinhole:
        stop(ax, z_l1 + f1, 1.5*w, openings=((0, 0.12*w),), width=0.03)
    return w*f2/f1


def twyman_green(ax, title=None):
    """Sketch of a Twyman-Green interferometer for phase shifting and Fourier
    fringe analysis.

    A HeNe laser beam is expanded (with a pinhole as spatial filter) and
    split by a beam splitter. The reference arm ends on a flat mirror that a
    piezo actuator (PZT) moves by fractions of a wavelength: the phase shift
    delta of phase shifting interferometry. Tilting the same mirror instead
    gives the carrier fringes of the single frame Fourier method. The test
    arm carries the wavefront error W of the surface under test (drawn
    exaggerated). Both returning beams are recombined by the splitter, and a
    telescope images the test surface onto the camera.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    w = 0.1
    z_bs = 2.6
    h = _expander(ax, -0.4, 0.0, 0.3, 1.2, 0.9, w, z_bs)
    laser(ax, -0.4, 0.0, length=1.3, height=0.3, text="HeNe laser")
    label(ax, 0.6, -0.55, "beam expander\nand pinhole", va="top")
    # test arm along +z
    z_test = 4.6
    band(ax, [z_bs, z_test], [h, h], [z_bs, z_test], [-h, -h], color=LASER)
    yy = np.linspace(-0.5, 0.5, 61)
    surf = z_test + 0.03*np.sin(2.5*np.pi*yy + 0.5) + 0.05*yy**2
    ax.add_patch(_patches().Polygon(np.r_[np.c_[surf, yy], [[z_test + 0.25, 0.5],
                                                          [z_test + 0.25, -0.5]]],
                                    closed=True, facecolor="0.8", edgecolor="0.3", zorder=3))
    label(ax, z_test + 0.12, -0.6, "surface under test\n(wavefront error W)", va="top")
    wavefront(ax, 3.7, 0.0, h, amplitude=0.06, color=TEST)
    arrow(ax, 4.15, 0.42, 3.55, 0.42, color=TEST, text="test wave", text_pos=(3.85, 0.55),
          va="bottom")
    # reference arm along +y
    y_ref = 2.0
    band(ax, [z_bs + h, z_bs + h], [0.0, y_ref], [z_bs - h, z_bs - h], [0.0, y_ref],
         color=LASER)
    ax.plot([z_bs - 0.5, z_bs + 0.5], [y_ref, y_ref], color="0.25", lw=4, solid_capstyle="butt",
            zorder=4)
    ax.add_patch(_patches().Rectangle((z_bs - 0.3, y_ref + 0.04), 0.6, 0.22, facecolor="#e6d3a3",
                                      edgecolor="#8c6d1f", hatch="////", zorder=4))
    label(ax, z_bs + 0.6, y_ref + 0.15, "reference mirror on a PZT:\n"
          "shift by λ/8 → phase step δ = π/2 (PSI),\ntilt → carrier fringes (Fourier method)",
          ha="left", fontsize=8)
    arrow(ax, z_bs - 0.65, y_ref - 0.15, z_bs - 0.65, y_ref + 0.15, style="<->", color="#8c6d1f")
    wavefront(ax, z_bs, 1.2, h, vertical=False, color=REFERENCE)
    arrow(ax, z_bs - 0.42, 1.55, z_bs - 0.42, 0.95, color=REFERENCE, text="reference\nwave",
          text_pos=(z_bs - 0.48, 1.25), ha="right")
    # recombined beams towards the camera along -y, through a telescope
    lenses = [(-0.9, 0.5), (-1.65, 0.25)]
    yt, zt = paraxial_ray(lenses, 0.0, h, 0.0, -2.25)
    yb, zb = paraxial_ray(lenses, 0.0, -h, 0.0, -2.25)
    band(ax, z_bs + zt, yt, z_bs + zb, yb, color="#8e5ba8")
    rotated_lens(ax, z_bs, -0.9, 1.15*h, text="imaging optics\n(test surface → camera)")
    rotated_lens(ax, z_bs, -1.65, 0.75*h)
    rotated_detector(ax, z_bs, -2.3, 0.3, text="camera:  I = a + b cos(φ − δ)", rotate=90,
                     text_offset=0.4)
    beam_splitter(ax, z_bs, 0.0, 0.9, "/", text="beam\nsplitter", text_dy=-0.35)
    _finish(ax, (-1.8, 6.4), (-2.6, 2.5), title)


def fizeau_interferometer(ax, cat_eye=True, title=None):
    """Sketch of a Fizeau interferometer with a transmission sphere measuring
    the radius of a lens surface.

    The laser light, focused into a pinhole, diverges through a beam
    splitter and is collimated. The transmission sphere (TS) turns the plane
    wave into a spherical wave; its last surface, concentric with the focus,
    reflects part of the light back as the reference wave. The test surface
    on a length measuring stage reflects the test wave. In the confocal
    position (solid) the centre of curvature of the test surface lies in
    the focus, in the cat's eye position (dashed) its vertex; the stage
    travel between the two nulls is the radius R. The returning waves are
    focused through the beam splitter into a small stop and imaged onto the
    camera.

    Parameters
    ----------
    ax : matplotlib Axes
    cat_eye : bool, optional
        Draw the cat's eye position of the test surface as well. Defaults to
        True.
    title : str, optional
    """

    z_ph, z_col, f_col = -0.7, 1.6, 2.3
    z_bs = 0.35
    h = 0.8
    # laser, focusing lens, pinhole
    lasers_z = -1.05
    zl, yl = paraxial_ray([(lasers_z, 0.35)], lasers_z - 0.3, 0.08, 0.0, z_ph)
    zl2, yl2 = paraxial_ray([(lasers_z, 0.35)], lasers_z - 0.3, -0.08, 0.0, z_ph)
    band(ax, zl, yl, zl2, yl2)
    laser(ax, lasers_z - 0.3, 0.0, length=1.2, height=0.28, text="HeNe laser")
    lens(ax, lasers_z, 0.15)
    stop(ax, z_ph, 0.3, openings=((0, 0.02),), width=0.03)
    label(ax, z_ph, -0.38, "pinhole", va="top", fontsize=8)
    # diverging to the collimator, collimated to the TS
    z_ts, f_ts = 2.4, 2.0
    z_f = z_ts + f_ts
    u = h/(z_col - z_ph)
    band(ax, [z_ph, z_col, z_ts - 0.2], [0, h, h], [z_ph, z_col, z_ts - 0.2], [0, -h, -h])
    lens(ax, z_col, 1.12*h, text="collimator", text_y=-1.0)
    # transmission sphere: front surface weakly convex, back surface concentric with the focus
    R_ref = f_ts - 0.2
    yy = np.linspace(-0.95, 0.95, 61)
    front = z_ts + 0.1 - 0.3*(1 - (yy/0.95)**2)
    back = z_f - np.sqrt(R_ref**2 - yy**2)
    ax.add_patch(_patches().Polygon(np.r_[np.c_[front, yy], np.c_[back, yy][::-1]], closed=True,
                                    facecolor=GLASS, edgecolor=EDGE, lw=1, zorder=3))
    ax.plot(back, yy, color=REFERENCE, lw=2, zorder=4)
    label(ax, z_ts + 0.1, -1.05, "transmission sphere\n(last surface: reference)", va="top")
    # converging test wave to the focus
    R_t = 0.9
    z_conf = z_f - R_t
    for y0 in np.linspace(-h, h, 7):
        z_start = z_f - np.sqrt(R_ref**2 - y0**2)
        # straight to the focus, ending on the test surface (a sphere around the focus)
        d = np.hypot(z_f - z_start, y0)
        frac = 1 - R_t/d
        ax.plot([z_start, z_start + frac*(z_f - z_start)], [y0, y0*(1 - frac)], color=LASER,
                lw=0.9, zorder=2)
        ax.plot([z_ts - 0.2, z_start], [y0, y0], color=LASER, lw=0.9, zorder=3.5, alpha=0.6)
    sphere(ax, z_conf, 0.0, R_t, 0.55, glass=0.35, color=TEST, lw=2.2)
    label(ax, z_conf + 0.15, -0.65, "test surface,\nconfocal", va="top", color=TEST)
    ax.plot([z_f], [0.0], "o", color="0.2", ms=3, zorder=6)
    label(ax, z_f, 0.12, "focus", fontsize=8, va="bottom")
    if cat_eye:
        sphere(ax, z_f, 0.0, R_t, 0.55, color=TEST, lw=1.2, ls="--")
        label(ax, z_f + 0.3, 0.62, "cat's eye", fontsize=8, color=TEST, ha="left")
        arrow(ax, z_conf, -1.2, z_f, -1.2, style="<->", text="stage travel = R",
              text_pos=((z_conf + z_f)/2, -1.27), va="top")
        ax.plot([z_conf, z_conf], [-0.6, -1.25], color="0.5", lw=0.6, ls=":")
        ax.plot([z_f, z_f], [-0.1, -1.25], color="0.5", lw=0.6, ls=":")
    # returning waves: reflected up by the beam splitter, focused into a stop, imaged on the camera
    w_bs = u*(z_bs - z_ph)
    y_foc = z_bs - z_ph
    band(ax, [z_bs - w_bs, z_bs], [0.0, y_foc], [z_bs + w_bs, z_bs], [0.0, y_foc],
         color="#8e5ba8")
    ax.add_patch(_patches().Rectangle((z_bs - 0.3, y_foc - 0.015), 0.27, 0.03, facecolor=METAL,
                                      zorder=4))
    ax.add_patch(_patches().Rectangle((z_bs + 0.03, y_foc - 0.015), 0.27, 0.03, facecolor=METAL,
                                      zorder=4))
    y_il, f_il = y_foc + 0.45, 0.45
    band(ax, [z_bs, z_bs + 0.2, z_bs + 0.2], [y_foc, y_il, y_il + 0.5],
         [z_bs, z_bs - 0.2, z_bs - 0.2], [y_foc, y_il, y_il + 0.5], color="#8e5ba8")
    rotated_lens(ax, z_bs, y_il, 0.26, text="imaging lens", text_offset=0.35)
    rotated_detector(ax, z_bs, y_il + 0.55, 0.3, text="camera: test against\nreference wave",
                     text_offset=0.4)
    beam_splitter(ax, z_bs, 0.0, 0.7, "\\")
    label(ax, z_bs + 0.35, -0.6, "beam\nsplitter", va="top", fontsize=8)
    _finish(ax, (-2.4, 5.6), (-1.6, y_il + 1.0), title)


def _lens_under_test(ax, z_lens, f, h, n_rays=9, spherical=0.25, z_end=None, color=LASER):
    """Collimated beam through a plano-convex singlet (plane side first) with
    spherical aberration: the marginal rays cross the axis before the
    paraxial focus. Returns the rays as (z, y, slope) after the lens."""

    yy = np.linspace(-h, h, n_rays)
    t = 0.3*h
    pts_y = np.linspace(-1.12*h, 1.12*h, 41)
    back = z_lens + t/2 - t*(1 - (pts_y/(1.12*h))**2)*0 + t*((1 - (pts_y/(1.12*h))**2) - 0.5)
    ax.add_patch(_patches().Polygon(np.r_[[[z_lens - t/2, -1.12*h], [z_lens - t/2, 1.12*h]],
                                          np.c_[back, pts_y][::-1]],
                                    closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1, zorder=3))
    rays = []
    for y0 in yy:
        z_cross = z_lens + f*(1 - spherical*(y0/h)**2)
        u = -y0/(z_cross - z_lens)
        rays.append((z_lens, y0, u))
        ax.plot([z_lens - 1.2, z_lens], [y0, y0], color=color, lw=0.9, zorder=2)
        if z_end is not None:
            ax.plot([z_lens, z_end], [y0, y0 + u*(z_end - z_lens)], color=color, lw=0.9, zorder=2)
    return rays


def shack_hartmann_sensor(ax, title=None):
    """Sketch of a Shack-Hartmann measurement of a lens.

    A collimated beam passes the lens under test (a plano-convex singlet,
    plane side first, with spherical aberration: the rim rays focus short).
    An aplanatic collimator re-collimates the light behind the focus; the
    lenslet array cuts the beam into patches and focuses each onto the
    camera. A spot moves from its reference position (the lenslet axis) by
    the lenslet focal length times the local wavefront slope (exaggerated).

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    z_lens, f, h = 0.0, 3.0, 0.8
    z_col, f_col = z_lens + f + 1.6, 1.6
    z_la, f_la = z_col + 1.2, 0.6
    z_cam = z_la + f_la
    rays = _lens_under_test(ax, z_lens, f, h, n_rays=9, spherical=0.12)
    label(ax, z_lens, -1.05, "lens under test\n(plane side first)", va="top")
    label(ax, z_lens - 1.0, 0.95, "collimated\ntest beam", fontsize=8)
    out = []
    for z0, y0, u in rays:
        zr, yr = paraxial_ray([(z_col, f_col)], z0, y0, u, z_la)
        ax.plot(zr, yr, color=LASER, lw=0.9, zorder=2)
        out.append((yr[-1], (yr[-1] - yr[-2])/(zr[-1] - zr[-2])))
    lens(ax, z_col, 1.0, text="collimator\n(aplanatic)")
    plane(ax, z_lens + f, 0.4, text="paraxial\nfocus", text_y=0.45)
    n = 8
    hl = 0.85
    centres = lenslet_array(ax, z_la, hl, n, text="lenslet\narray")
    detector(ax, z_cam, hl, width=0.08, text="camera")
    # spot displacements: f_la times the mean slope of the rays over each lenslet, exaggerated
    y_out = np.array([o[0] for o in out])
    s_out = np.array([o[1] for o in out])
    p = 2*hl/n
    for c in centres:
        slope = np.interp(c, y_out[::-1] if y_out[0] > y_out[-1] else y_out,
                          s_out[::-1] if y_out[0] > y_out[-1] else s_out)
        y_spot = c + 1.0*f_la*slope
        for e in (-p/2.5, p/2.5):
            ax.plot([z_la, z_cam], [c + e, y_spot], color=LASER, lw=0.7, zorder=2)
        ax.plot([z_la + 0.04, z_cam], [c, c], color="0.6", lw=0.5, ls=":", zorder=2)
        ax.plot([z_cam], [y_spot], "o", color=TEST, ms=3.5, zorder=6)
    label(ax, z_cam + 0.25, 0.0, "spot shift =\nf × local slope", ha="left", fontsize=8)
    _finish(ax, (-1.4, z_cam + 1.5), (-1.5, 1.3), title)


def phase_diversity_setup(ax, planes=(-1.0, 0.0, 1.0), title=None):
    """Sketch of a phase diversity measurement of a lens: the same lens and
    beam as in shack_hartmann_sensor(), with a camera on a focus stage that
    records the PSF in a few planes around the focus.

    Parameters
    ----------
    ax : matplotlib Axes
    planes : sequence of numbers, optional
        Camera positions relative to the best focus, in units of the drawn
        stage travel. Defaults to (-1, 0, 1).
    title : str, optional
    """

    z_lens, f, h = 0.0, 3.0, 0.8
    z_best = z_lens + f*(1 - 0.12*0.6)
    dz = 0.45
    _lens_under_test(ax, z_lens, f, h, n_rays=9, spherical=0.12, z_end=z_best + 1.3*dz*max(planes))
    label(ax, z_lens, -1.05, "lens under test\n(plane side first)", va="top")
    label(ax, z_lens - 1.0, 0.95, "collimated\ntest beam", fontsize=8)
    for p in planes:
        z = z_best + p*dz
        ax.add_patch(_patches().Rectangle((z - 0.03, -0.3), 0.06, 0.6, facecolor="0.35" if p == 0
                                          else "none", edgecolor="0.35", ls="-" if p == 0 else "--",
                                          zorder=4))
    zs = [z_best + p*dz for p in planes]
    arrow(ax, min(zs), -0.5, max(zs), -0.5, style="<->", text="camera on a focus stage:\n"
          "PSFs at known defocus", text_pos=(z_best, -0.6), va="top")
    _finish(ax, (-1.4, z_best + 2.2), (-1.5, 1.3), title)


def holography_setup(ax, microscope=False, title=None):
    """Sketch of an off-axis digital holography setup (Mach-Zehnder).

    The expanded laser beam is split into an object arm, which passes the
    sample, and a reference arm. A second beam splitter recombines them on
    the camera, the reference tilted by the angle theta that puts the
    carrier fringes into the hologram. Without a microscope (notebook 26)
    the camera records the object wave a distance d behind the sample, with
    a microscope (notebook 27) objective and tube lens image the sample onto
    the camera (with an optional defocus).

    Parameters
    ----------
    ax : matplotlib Axes
    microscope : bool, optional
        Draw an objective and a tube lens in the object arm. Defaults to
        False.
    title : str, optional
    """

    w = 0.1
    z_bs1 = 1.0
    h = _expander(ax, -1.0, -0.7, 0.25, 0.05, 0.75, w, z_bs1)
    laser(ax, -1.0, 0.0, length=1.2, height=0.28, text="laser")
    y_ref = 2.0
    if microscope:
        z_s, z_obj, f_o, z_tube, f_t = 2.0, 2.5, 0.5, 3.8, 1.0
        z_cam = z_tube + f_t
        z_bs2 = z_cam - 0.6
    else:
        z_s = 2.6
        z_bs2, z_cam = 3.5, 4.2
    # object arm
    band(ax, [z_bs1, z_s], [h, h], [z_bs1, z_s], [-h, -h], color=LASER)
    ax.plot([z_s, z_s], [-0.45, 0.45], color="#8fbf8f", lw=4, solid_capstyle="butt", zorder=3)
    label(ax, z_s, -0.52, "sample", va="top")
    if microscope:
        lenses = [(z_obj, f_o), (z_tube, f_t)]
        zt, yt = paraxial_ray(lenses, z_s, h, 0.0, z_cam)
        zb, yb = paraxial_ray(lenses, z_s, -h, 0.0, z_cam)
        band(ax, zt, yt, zb, yb, color=LASER, alpha=0.12)
        for u in (-0.35, 0.0, 0.35):
            draw_ray(ax, lenses, z_s, 0.12, u, z_cam, color=TEST, lw=1.0, arrow=False)
        lens(ax, z_obj, 0.45, text="objective", text_y=-0.52)
        lens(ax, z_tube, 0.75, text="tube lens", text_y=-0.85)
        label(ax, z_obj + 0.55, 0.62, "object wave: image\nof the sample", color=TEST, fontsize=8,
              va="bottom")
        h_cam = 0.75
    else:
        # light scattered by the sample spreads towards the camera
        band(ax, [z_s, z_cam], [h, h], [z_s, z_cam], [-h, -h], color=LASER, alpha=0.12)
        for u in (-0.3, 0.0, 0.3):
            ax.plot([z_s, z_cam], [0.1, 0.1 + u*(z_cam - z_s)], color=TEST, lw=1.0)
        arrow(ax, z_s, -0.75, z_cam, -0.75, style="<->", text="d", text_pos=((z_s + z_cam)/2, -0.8),
              va="top")
        label(ax, (z_s + z_bs2)/2, 0.75, "object wave", color=TEST, fontsize=8, va="bottom")
        h_cam = 0.5
    # reference arm: up, along, down, then tilted onto the camera
    band(ax, [z_bs1 - h, z_bs1 - h, z_bs2 + h, z_bs2 + h], [0.0, y_ref + h, y_ref + h, 0.0],
         [z_bs1 + h, z_bs1 + h, z_bs2 - h, z_bs2 - h], [0.0, y_ref - h, y_ref - h, 0.0],
         color=REFERENCE, alpha=0.15)
    theta = np.radians(12)
    d = z_cam - z_bs2
    band(ax, [z_bs2, z_cam], [h, d*np.tan(theta) + h], [z_bs2, z_cam], [-h, d*np.tan(theta) - h],
         color=REFERENCE, alpha=0.15)
    mirror(ax, z_bs1, y_ref, 0.6, np.pi/4)
    mirror(ax, z_bs2, y_ref, 0.6, -np.pi/4)
    label(ax, (z_bs1 + z_bs2)/2, y_ref + h + 0.08, "reference wave", color=REFERENCE,
          va="bottom", fontsize=9)
    beam_splitter(ax, z_bs1, 0.0, 0.6, "/")
    beam_splitter(ax, z_bs2, 0.0, 0.6, "\\")
    detector(ax, z_cam, h_cam, width=0.08)
    label(ax, z_cam + 0.15, -h_cam - 0.05, "camera: hologram\n|O + R|²", va="top", ha="left")
    label(ax, z_cam - 0.1, h_cam + 0.12, "θ: off-axis\nreference", color=REFERENCE, fontsize=8,
          va="bottom", ha="right")
    label(ax, -0.4, -0.45, "beam expander", va="top", fontsize=8)
    _finish(ax, (-2.3, z_cam + 1.4), (-1.2, y_ref + 0.75), title)


def fold_mirror(ax, title=None):
    """Sketch of a coated 45° fold mirror in a laser head, with the s and p
    directions.

    A collimated, circularly polarized beam meets the mirror at 45° and is
    folded towards the focusing objective. The plane of incidence is the
    plane of the drawing: p lies in it (arrows), s is normal to it (dots).
    The coating, a stack of high and low index layers, reflects s and p with
    different phases (retardance), which turns circular polarization
    elliptical.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    h = 0.35
    z_m, y_m = 2.6, 0.0
    y_obj, f_obj = 1.4, 0.8
    y_f = y_obj + f_obj
    # incident beam up to the mirror face (the line y = z - z_m), reflected beam from it
    band(ax, [-0.3, z_m + h], [h, h], [-0.3, z_m - h], [-h, -h], color=LASER)
    band(ax, [z_m + h, z_m + h, z_m], [h, y_obj, y_f], [z_m - h, z_m - h, z_m], [-h, y_obj, y_f],
         color=LASER)
    rotated_lens(ax, z_m, y_obj, 1.2*h, text="objective", text_offset=0.55)
    ax.plot([z_m], [y_f], "o", color=TEST, ms=4, zorder=6)
    label(ax, z_m + 0.12, y_f + 0.05, "focus", ha="left", fontsize=8)
    # mirror substrate with the layer stack on its front face ("/" orientation)
    L = 1.3
    a = np.pi/4
    t_dir = np.array([np.sin(a), np.cos(a)])           # along the face
    n_dir = np.array([np.cos(a), -np.sin(a)])          # into the substrate
    c = np.array([z_m, y_m])
    sub = [c - t_dir*L/2, c + t_dir*L/2, c + t_dir*L/2 + 0.25*n_dir, c - t_dir*L/2 + 0.25*n_dir]
    ax.add_patch(_patches().Polygon(sub, closed=True, facecolor="0.85", edgecolor="0.4", lw=1,
                                    zorder=3))
    for k in range(6):
        d0, d1 = -0.012*(k + 1), -0.012*k
        quad = [c - t_dir*L/2 + d0*n_dir, c + t_dir*L/2 + d0*n_dir, c + t_dir*L/2 + d1*n_dir,
                c - t_dir*L/2 + d1*n_dir]
        ax.add_patch(_patches().Polygon(quad, closed=True, facecolor="#3b6ea5" if k % 2 == 0
                                        else "#cfe0f1", edgecolor="none", zorder=4))
    label(ax, z_m + 0.55, -0.55, "45° mirror: (HL)$^N$H stack\non fused silica", ha="left",
          va="top")
    # p (in plane) and s (out of plane) on the incident and reflected beam
    pc = "#6a3d9a"

    def s_symbol(z, y):
        ax.plot([z], [y], "o", color=pc, ms=8, mfc="white", mew=1.2, zorder=6)
        ax.plot([z], [y], "o", color=pc, ms=2.5, zorder=7)

    for zz in (0.4, 1.3):
        arrow(ax, zz, -0.25, zz, 0.25, style="<->", color=pc)
        s_symbol(zz + 0.2, 0.0)
    arrow(ax, z_m - 0.25, 0.85, z_m + 0.25, 0.85, style="<->", color=pc)
    s_symbol(z_m, 1.05)
    label(ax, -0.3, -0.45, "collimated beam, circularly polarized\n"
          "p: in the plane of incidence (arrows)\ns: normal to it (circles)", va="top", ha="left",
          fontsize=8, color=pc)
    label(ax, z_m + 0.5, 0.95, "r$_p$/r$_s$ = |r$_p$/r$_s$| e$^{iΔ}$:\nretardance Δ turns\n"
          "circular into elliptical", ha="left", fontsize=8, color=pc)
    _finish(ax, (-0.5, z_m + 2.0), (-1.0, y_f + 0.3), title)


def _fold(y, h):
    """Unfold the reflections in a light pipe of half width h."""

    t = np.mod(y + h, 4*h)
    return np.where(t < 2*h, t - h, 3*h - t)


def krf_illuminator(ax, title=None):
    """Sketch of the KrF illuminator of notebook 42, unfolded and not to
    scale.

    Secondary source (laser, beam shaper and diffuser), zoom-axicon
    objective (its directions become positions in the pupil plane 30, where
    the size of the illuminated disk is sigma and the axicons make it a
    ring), coupling lens (pupil plane 30 to the rod entrance), the glass rod
    (a kaleidoscope that makes the field uniform) with the masking blades at
    its exit, and the REMA objective that images the rod exit onto the
    reticle with its diaphragm in a pupil plane. Pupil planes (P) and field
    planes (F) are marked. The rays are paraxial, every group drawn as one
    thin lens.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    z_src, f_z = 0.0, 1.5
    z_zoom = z_src + f_z
    z_p30 = z_zoom + f_z
    f_c = 1.0
    z_c = z_p30 + f_c
    z_r0, z_r1, h_rod = z_c + f_c, z_c + f_c + 4.0, 0.3
    f_a, f_b = 0.6, 1.2
    z_a = z_r1 + f_a
    z_d = z_a + f_a
    z_b = z_d + f_b
    z_ret = z_b + f_b
    UV = "#7b3fa0"
    # parts
    ax.add_patch(_patches().Rectangle((z_src - 1.4, -0.2), 0.9, 0.4, facecolor="0.3", zorder=4))
    label(ax, z_src - 0.95, 0.0, "KrF\nlaser", color="white", fontsize=8, zorder=5)
    ax.annotate("", xy=(z_src - 0.08, 0.0), xytext=(z_src - 0.5, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=UV, lw=1.2, mutation_scale=9))
    ax.plot([z_src, z_src], [-0.15, 0.15], color=UV, lw=4, solid_capstyle="butt", zorder=4)
    label(ax, z_src - 0.5, -0.6, "beam shaper,\ndiffuser:\nsecondary source", va="top", fontsize=8)
    ax.add_patch(_patches().FancyBboxPatch((z_zoom - 0.75, -0.95), 1.5, 1.9,
                                           boxstyle="round,pad=0.02", facecolor="none",
                                           edgecolor="0.6", ls="--", lw=0.8, zorder=1))
    lens(ax, z_zoom - 0.5, 0.75)
    lens(ax, z_zoom + 0.5, 0.75)
    # axicon pair in contact (conventional setting)
    zz = z_zoom
    ax.add_patch(_patches().Polygon([(zz - 0.3, -0.75), (zz - 0.3, 0.75), (zz - 0.12, 0.75),
                                     (zz + 0.06, 0.0), (zz - 0.12, -0.75)],
                                    closed=True, facecolor=GLASS, edgecolor=EDGE, zorder=3))
    ax.add_patch(_patches().Polygon([(zz - 0.12, 0.75), (zz + 0.28, 0.75), (zz + 0.28, -0.75),
                                     (zz - 0.12, -0.75), (zz + 0.06, 0.0)],
                                    closed=True, facecolor="#dceaf6", edgecolor=EDGE, zorder=3))
    label(ax, z_zoom, -1.0, "zoom-axicon objective\n(σ zoom; axicons\napart: annular)",
          va="top", fontsize=8)
    plane(ax, z_p30, 0.9, text="pupil plane 30\n+ diffuser", fontsize=8)
    lens(ax, z_c, 0.9, text="coupling\nlens", text_y=-1.0, fontsize=8)
    ax.add_patch(_patches().Rectangle((z_r0, -h_rod), z_r1 - z_r0, 2*h_rod, facecolor=GLASS,
                                      edgecolor=EDGE, lw=1, zorder=1))
    label(ax, (z_r0 + z_r1)/2, -h_rod - 0.1, "glass rod: kaleidoscope, uniform field", va="top",
          fontsize=8)
    for sgn in (1, -1):
        ax.add_patch(_patches().Rectangle((z_r1 + 0.03, sgn*h_rod - (0.2 if sgn < 0 else 0)),
                                          0.05, 0.2, facecolor=METAL, zorder=4))
    label(ax, z_r1 + 0.05, h_rod + 0.25, "masking\nblades", fontsize=8, va="bottom")
    lens(ax, z_a, 0.7)
    stop(ax, z_d, 0.9, openings=((0, 0.45),), width=0.04)
    lens(ax, z_b, 1.1)
    label(ax, (z_a + z_b)/2, -1.05, "REMA objective\n(diaphragm in plane 14)", va="top",
          fontsize=8)
    ax.plot([z_ret, z_ret], [-0.75, 0.75], color="0.2", lw=2, ls=(0, (2, 1.5)), zorder=4)
    label(ax, z_ret, -0.82, "reticle", va="top", fontsize=8)
    ax.annotate("", xy=(z_ret + 0.9, 0.0), xytext=(z_ret + 0.1, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1, mutation_scale=9))
    label(ax, z_ret + 0.5, 0.12, "projection\nlens", fontsize=8, va="bottom", color="0.4")
    # paraxial rays: source points with a fan of directions
    for y_s, col in ((0.0, UV), (0.12, "#c51b7d")):
        for u in np.linspace(-0.25, 0.25, 5):
            z1, y1 = paraxial_ray([(z_zoom, f_z), (z_c, f_c)], z_src, y_s, u, z_r0)
            ax.plot(z1, y1, color=col, lw=0.8, alpha=0.85, zorder=2)
            v = (y1[-1] - y1[-2])/(z1[-1] - z1[-2])
            zz = np.linspace(z_r0, z_r1, 200)
            ax.plot(zz, _fold(y1[-1] + v*(zz - z_r0), h_rod), color=col, lw=0.8, alpha=0.85,
                    zorder=2)
            # exit: position folded, direction flipped with every reflection
            y_unf = y1[-1] + v*(z_r1 - z_r0)
            n_ref = np.floor((y_unf + h_rod)/(2*h_rod))
            v_out = v*(-1)**n_ref
            z2, y2 = paraxial_ray([(z_a, f_a), (z_b, f_b)], z_r1, float(_fold(y_unf, h_rod)), v_out,
                                  z_ret)
            ax.plot(z2, y2, color=col, lw=0.8, alpha=0.85, zorder=2)
    for z_, txt in [(z_p30, "P"), (z_d, "P"), (z_r1, "F"), (z_ret, "F")]:
        label(ax, z_, -1.6, txt, fontsize=10, color="#b35900" if txt == "P" else "#1f5f8b",
              weight="bold")
    label(ax, (z_src + z_ret)/2, -1.9, "P: pupil planes (positions = directions on the reticle,"
          " σ)    F: field planes (conjugate to the reticle)", fontsize=8, color="0.3")
    _finish(ax, (z_src - 1.5, z_ret + 1.0), (-2.05, 1.3), title)


def spatial_filter(ax, pinhole=True, title=None):
    """Sketch of a spatial filter: a 3x beam expander with a pinhole in the
    common focus (the setup of notebook 56).

    The clean part of the input beam (orange) is focused onto the axis and
    passes the pinhole. Light that travels in other directions, from a
    ghost reflection or from phase ripple (red), is focused beside the axis
    and blocked. The second lens collimates the filtered beam.

    Parameters
    ----------
    ax : matplotlib Axes
    pinhole : bool, optional
        Draw the pinhole (otherwise the ghost passes and crosses the output
        beam). Defaults to True.
    title : str, optional
    """

    f1, f2 = 1.2, 3.6
    z1 = 0.0
    z_ph = z1 + f1
    z2 = z_ph + f2
    z_end = z2 + 1.5
    w = 0.35
    lenses = [(z1, f1), (z2, f2)]
    zt, yt = paraxial_ray(lenses, z1 - 1.5, w, 0.0, z_end)
    zb, yb = paraxial_ray(lenses, z1 - 1.5, -w, 0.0, z_end)
    band(ax, zt, yt, zb, yb, color=LASER)
    # a ghost: a weaker beam at a small angle, focused beside the pinhole
    u_g = 0.08
    end_g = z_ph if pinhole else z_end
    for y0 in (-0.6*w, 0.0, 0.6*w):
        z, y = paraxial_ray(lenses, z1 - 1.5, y0 - u_g*1.5, u_g, end_g)
        ax.plot(z, y, color=TEST, lw=0.9, ls="--", zorder=2)
    lens(ax, z1, 1.3*w, text="lens 1")
    lens(ax, z2, 3.3*w, text="lens 2")
    if pinhole:
        stop(ax, z_ph, 0.9, openings=((0, 0.06),), width=0.05, text="pinhole in the\n"
             "back focal plane", text_y=-1.0)
    else:
        plane(ax, z_ph, 0.9, text="no pinhole")
    label(ax, z1 - 0.9, 0.55, "clean beam\n+ ghost (dashed)", fontsize=8, va="bottom")
    label(ax, z_end - 0.3, -3.1*w - 0.12, "filtered and\nexpanded beam" if pinhole else
          "ghost crosses the\nexpanded beam", fontsize=8, va="top")
    arrow(ax, z_ph + 0.15, f1*u_g + 0.25, z_ph + 0.02, f1*u_g + 0.02, color=TEST,
          text="x = f₁ θ", text_pos=(z_ph + 0.2, f1*u_g + 0.3), ha="left", va="bottom")
    _finish(ax, (z1 - 1.6, z_end + 0.1), (-1.6, 1.4), title)


def _spiral_plate(ax, z, h, step=0.24, base=0.06, text=None, text_y=None, fontsize=9):
    # meridional section of a spiral phase plate: the azimuths pi/2 (top) and
    # 3*pi/2 (bottom) have a quarter and three quarters of the full step, and
    # the step of the plate edge runs through the axis
    mpatches = _patches()
    for y0, y1, t in ((0.0, h, base + step/4), (-h, 0.0, base + 3*step/4)):
        ax.add_patch(mpatches.Rectangle((z - base/2, y0), t, y1 - y0, facecolor=GLASS,
                                        edgecolor=EDGE, lw=1, zorder=3))
    ax.plot([z - base/2 - 0.05, z + base/2 + step + 0.05], [0, 0], color="0.4", lw=0.6, ls=":",
            zorder=4)
    if text:
        label(ax, z + step/2, -1.15*h if text_y is None else text_y, text, fontsize=fontsize,
              va="top")


def _profile_image(ax, z, y, size, image, cmap=None):
    # a small intensity pattern drawn into the sketch, centred at (z, y), on
    # a white background (magma_r fading to white at zero, as in the notebooks)
    if cmap is None:
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap
        t = np.linspace(0, 1, 256)[:, None]
        cmap = ListedColormap(1 - (1 - colormaps["magma_r"](t[:, 0])[:, :3])*np.clip(t/0.15, 0, 1))
    ax.imshow(image/image.max(), extent=[z - size/2, z + size/2, y - size/2, y + size/2],
              origin="lower", cmap=cmap, zorder=6, interpolation="bilinear")
    ax.add_patch(_patches().Rectangle((z - size/2, y - size/2), size, size, facecolor="none",
                                      edgecolor="0.4", lw=0.8, zorder=7))


def vortex_interferometer(ax, reference="tilted", title=None):
    """Sketch of a Mach-Zehnder interferometer that measures the charge of an
    optical vortex (notebook 69).

    The expanded laser beam is split. In one arm a spiral phase plate makes
    the vortex beam, the other arm is the reference. A second beam splitter
    superposes both on a camera. A tilted plane reference gives straight
    fringes with a fork of ``|l|`` extra fringes at the vortex; a spherical
    reference (a lens in the reference arm) gives l spiral arms whose sense
    of rotation is the sign of l.

    Parameters
    ----------
    ax : matplotlib Axes
    reference : {"tilted", "spherical"}, optional
        Kind of reference wave. Defaults to "tilted".
    title : str, optional
    """

    w = 0.1
    z_bs1, z_bs2, z_cam = 1.0, 3.5, 4.2
    h = _expander(ax, -1.0, -0.7, 0.25, 0.05, 0.75, w, z_bs1)
    laser(ax, -1.0, 0.0, length=1.2, height=0.28, text="laser")
    y_ref = 2.0
    z_spp = 2.0
    # vortex arm
    band(ax, [z_bs1, z_bs2], [h, h], [z_bs1, z_bs2], [-h, -h], color=LASER)
    band(ax, [z_bs2, z_cam], [h, h], [z_bs2, z_cam], [-h, -h], color=LASER)
    _spiral_plate(ax, z_spp, 0.45, text="spiral phase plate\nexp(i l φ)", text_y=-0.55)
    label(ax, (z_spp + z_bs2)/2 + 0.1, 0.42, "vortex beam", color=TEST, fontsize=8, va="bottom")
    # reference arm: up, along, down
    band(ax, [z_bs1 - h, z_bs1 - h, z_bs2 + h, z_bs2 + h], [0.0, y_ref + h, y_ref + h, 0.0],
         [z_bs1 + h, z_bs1 + h, z_bs2 - h, z_bs2 - h], [0.0, y_ref - h, y_ref - h, 0.0],
         color=REFERENCE, alpha=0.15)
    mirror(ax, z_bs1, y_ref, 0.6, np.pi/4)
    mirror(ax, z_bs2, y_ref, 0.6, -np.pi/4)
    d = z_cam - z_bs2
    if reference == "spherical":
        z_l = (z_bs1 + z_bs2)/2
        rotated_lens(ax, z_l, y_ref, 1.8*h, rotate=0)
        label(ax, z_l, y_ref + 1.8*h + 0.06, "lens: spherical reference wave", color=REFERENCE,
              va="bottom", fontsize=9)
        band(ax, [z_bs2, z_cam], [h, 1.6*h], [z_bs2, z_cam], [-h, -1.6*h], color=REFERENCE,
             alpha=0.15)
        cam_text = "camera: spiral with\n|l| arms"
    else:
        theta = np.radians(12)
        band(ax, [z_bs2, z_cam], [h, d*np.tan(theta) + h], [z_bs2, z_cam],
             [-h, d*np.tan(theta) - h], color=REFERENCE, alpha=0.15)
        label(ax, (z_bs1 + z_bs2)/2, y_ref + h + 0.08, "tilted plane reference wave",
              color=REFERENCE, va="bottom", fontsize=9)
        cam_text = "camera: fork with\n|l| extra fringes"
    beam_splitter(ax, z_bs1, 0.0, 0.6, "/")
    beam_splitter(ax, z_bs2, 0.0, 0.6, "\\")
    detector(ax, z_cam, 0.5, width=0.08)
    label(ax, z_cam + 0.15, -0.55, cam_text, va="top", ha="left")
    label(ax, -0.4, -0.45, "beam expander", va="top", fontsize=8)
    _finish(ax, (-2.3, z_cam + 1.4), (-1.2, y_ref + 0.75), title)


def mode_converter(ax, title=None, images=None):
    """Sketch of the pi/2 cylindrical lens mode converter of Beijersbergen et
    al. (notebook 69).

    Two cylindrical lenses of focal length f (power in the plane of the
    drawing) a distance d = sqrt(2)*f apart turn a Hermite-Gaussian mode
    HG_n0 whose axes are at 45° to the cylinder axes into the
    Laguerre-Gaussian mode LG_0n, and back. The beam has its waist in the
    middle between the lenses with the Rayleigh length (1 + 1/sqrt(2))*f.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    images : tuple of two 2d arrays, optional
        Intensity patterns drawn before and behind the converter (e.g. an
        HG mode at 45° and the LG mode).
    """

    f = 1.4
    d = np.sqrt(2)*f
    z1, z2 = 0.0, d
    zc = d/2
    zR = (1 + 1/np.sqrt(2))*f
    w0 = 0.28
    zz = np.linspace(z1 - 1.6, z2 + 1.6, 200)
    # the Gaussian envelope in the plane of the lenses (the waist between them)
    ww = w0*np.sqrt(1 + ((zz - zc)/zR)**2)
    band(ax, zz, ww, zz, -ww, color=LASER)
    for z_ in (z1, z2):
        ax.add_patch(_patches().Rectangle((z_ - 0.05, -0.6), 0.1, 1.2, facecolor=GLASS,
                                          edgecolor=EDGE, lw=1, zorder=3))
        label(ax, z_, -0.68, "cylindrical\nlens f", va="top", fontsize=8)
    plane(ax, zc, 0.5, text="waist", text_y=0.55)
    arrow(ax, z1, 0.85, z2, 0.85, style="<->", text="d = √2 f", text_pos=(zc, 0.92), va="bottom")
    label(ax, zc, -1.15, "Rayleigh length of the beam: z_R = (1 + 1/√2) f", fontsize=8,
          color="0.3")
    if images is not None:
        _profile_image(ax, z1 - 1.0, 0.0, 0.9, images[0])
        _profile_image(ax, z2 + 1.0, 0.0, 0.9, images[1])
        label(ax, z1 - 1.0, -0.6, "HG$_{n0}$ at 45°", fontsize=8, va="top")
        label(ax, z2 + 1.0, -0.6, "LG$_{0n}$", fontsize=8, va="top")
    _finish(ax, (z1 - 1.6, z2 + 1.6), (-1.3, 1.25), title)


def axicon_bessel(ax, obstacle=True, title=None):
    """Sketch of a Bessel beam made by an axicon, optionally with an
    obstacle in the Bessel zone (the setup of notebook 68).

    A collimated beam of radius w falls on an axicon (a glass cone), which
    tilts all rays towards the axis by the same angle θ. The two halves of
    the beam cross in a diamond, the Bessel zone, which ends at
    z_max = w/tan θ. An opaque disk of radius a in the zone blocks the rays
    that would have crossed the axis behind it; they are missing only in a
    triangle of length a/tan θ (the shadow, white), and the rays that pass
    beside the obstacle fill the axis again behind it.

    Parameters
    ----------
    ax : matplotlib Axes
    obstacle : bool, optional
        Draw the obstacle and its shadow. Defaults to True.
    title : str, optional
    """

    mpatches = _patches()
    w, u = 0.6, 0.16           # beam radius, deflection angle (exaggerated)
    z_max = w/u
    z_end = 1.25*z_max
    # input beam
    band(ax, [-1.4, 0.0], [w, w], [-1.4, 0.0], [-w, -w], color=LASER)
    # the two halves of the cone: rays of the upper half go down and vice versa
    for sgn in (1, -1):
        band(ax, [0.0, z_end], [sgn*w, sgn*(w - u*z_end)], [0.0, z_end], [0.0, -sgn*u*z_end],
             color=LASER, alpha=0.18)
    zone = mpatches.Polygon([(0.0, 0.0), (z_max/2, w/2), (z_max, 0.0), (z_max/2, -w/2)],
                            closed=True, facecolor=LASER, edgecolor="none", alpha=0.25, zorder=1.2)
    ax.add_patch(zone)
    z_shadow = (0.38*z_max, 0.38*z_max + 0.14/u) if obstacle else (z_max, z_max)
    ax.plot([0.0, z_shadow[0]], [0.0, 0.0], color=LASER, lw=2.2, zorder=2)
    ax.plot([z_shadow[1], z_max], [0.0, 0.0], color=LASER, lw=2.2, zorder=2)
    # the axicon: flat entrance face, conical exit face
    ha, t0, t1 = 1.25*w, 0.06, 0.22
    ax.add_patch(mpatches.Polygon([(-t0, -ha), (-t0, ha), (t0, ha), (t1, 0.0), (t0, -ha)],
                                  closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1, zorder=3))
    label(ax, 0.0, -1.15*ha, "axicon", va="top")
    label(ax, -0.7, w + 0.12, "collimated beam,\nradius w", fontsize=8, va="bottom")
    arrow(ax, 0.55, 0.62, 0.36, w - 0.36*u - 0.01, text="θ", text_pos=(0.62, 0.66),
          ha="left", va="bottom")
    label(ax, z_max, -0.12, "z_max = w / tan θ", fontsize=8, va="top", ha="left")
    if obstacle:
        z0, a = 0.38*z_max, 0.14
        for sgn in (1, -1):  # rays blocked by the obstacle: a missing strip in each half
            dz = z_end - z0
            ax.add_patch(mpatches.Polygon([(z0, a), (z_end, a - sgn*u*dz), (z_end, -a - sgn*u*dz),
                                           (z0, -a)], closed=True, facecolor="white",
                                          edgecolor="none", alpha=0.55, zorder=1.6))
        ax.add_patch(mpatches.Polygon([(z0, a), (z0 + a/u, 0.0), (z0, -a)], closed=True,
                                      facecolor="white", edgecolor="0.4", ls="--", lw=0.8,
                                      zorder=1.7))
        ax.add_patch(mpatches.Rectangle((z0 - 0.03, -a), 0.06, 2*a, facecolor=METAL,
                                        edgecolor="none", zorder=4))
        label(ax, z0, 0.95*w, "obstacle,\nradius a", fontsize=8, va="bottom")
        ax.plot([z0, z0], [a + 0.03, 0.93*w], color="0.5", lw=0.6)
        arrow(ax, z0, -0.5*w, z0 + a/u, -0.5*w, style="<|-|>", color="0.3")
        label(ax, z0 + a/(2*u), -0.5*w - 0.06, "shadow\na / tan θ", fontsize=8, va="top",
              color="0.3")
        label(ax, z0 + a/u + 0.9, 0.12, "healed", fontsize=8, va="bottom")
    else:
        label(ax, z_max/2, 0.08, "Bessel zone", fontsize=8, va="bottom")
    _finish(ax, (-1.5, z_end + 0.1), (-1.0, 1.05), title)


def cubic_phase_airy(ax, title=None):
    """Sketch of an Airy beam made by a cubic phase plate and a lens (the
    setup of notebook 68).

    The cubic phase φ ∝ y³ tilts the rays by an angle ∝ y², the same for
    both halves of the beam. Behind the lens they no longer meet in a focus
    but touch a parabola, the caustic, on which the main lobe of the Airy
    beam travels. The outer rays reach the caustic last: they carry the
    light that builds the main lobe further down the line, which is why an
    Airy beam heals.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    f, w, c = 2.0, 0.7, -0.18  # focal length, beam radius, strength of the cubic phase
    z_plate, z_lens = 0.0, 0.8
    z_f = z_lens + f
    z_end = z_f + 2.0
    band(ax, [-1.3, z_plate], [w, w], [-1.3, z_plate], [-w, -w], color=LASER)
    for y0 in np.linspace(-w, w, 13):
        u = c*y0**2                # tilt from the cubic phase
        y1 = y0 + u*(z_lens - z_plate)
        u2 = u - y1/f              # thin lens
        zz = np.array([z_plate, z_lens, z_end])
        yy = np.array([y0, y1, y1 + u2*(z_end - z_lens)])
        ax.plot(zz, yy, color=LASER, lw=0.8, alpha=0.8, zorder=2)
    # the caustic: envelope of the ray family behind the lens, where d/dt of
    # y(z; t) = y1(t) + slope(t) (z - z_lens) vanishes
    d = z_lens - z_plate
    t = np.linspace(-w, w, 801)
    y1 = t + c*t**2*d
    slope = c*t**2 - y1/f
    dz = -(1 + 2*c*t*d)/(2*c*t - (1 + 2*c*t*d)/f)
    zc, yc = z_lens + dz, y1 + slope*dz
    keep = (zc > z_lens + 0.3) & (zc < z_end)
    zc, yc = zc[keep], yc[keep]
    order = np.argsort(zc)
    zc, yc = zc[order], yc[order]
    ax.plot(zc, yc, color="0.15", lw=1.6, ls="--", zorder=3)
    # plate with a cubic surface
    mpatches = _patches()
    yy = np.linspace(-1.2*w, 1.2*w, 61)
    face = z_plate + 0.08 + 0.05*(yy/w)**3
    pts = np.r_[np.c_[np.full_like(yy, z_plate - 0.08), yy], np.c_[face, yy][::-1]]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1,
                                  zorder=3))
    label(ax, z_plate, -1.38*w, "cubic phase\nplate (or SLM)", va="top")
    lens(ax, z_lens, 1.25*w, text="lens, focal\nlength f", text_y=-1.38*w)
    plane(ax, z_f, 1.2*w, text="back focal plane")
    label(ax, zc[-1] - 0.05, yc[-1] + 0.12, "caustic: the main lobe\nof the Airy beam",
          fontsize=8, ha="right", va="bottom")
    _finish(ax, (-1.4, z_end + 0.1), (-1.35, 1.25), title)


STAR = "#e0a000"
STAR2 = "#8e5ba8"


def _fringe_curve(ax, z0, y, phase, period, amplitude, visibility=1.0, color="0.3", lw=1.0,
                  ls="-"):
    """A fringe profile drawn sideways: z = z0 + amplitude*(1 + V cos(2 pi y/period + phase))."""
    ax.plot(z0 + amplitude*(1 + visibility*np.cos(2*np.pi*y/period + phase)), y, color=color,
            lw=lw, ls=ls, zorder=5)


def young_extended_source(ax, title=None):
    """Sketch of Young's double pinhole lit by an extended incoherent source
    (section 1 of notebook 74).

    The source sits in the front focal plane of a collimator. Each source
    point sends a plane wave onto the pinholes, tilted by its height over
    the focal length, and makes complete fringes on the screen behind a
    second lens, shifted by its tilt. The points are mutually incoherent, so
    their fringe intensities add up and wash out (black curve) once the
    shifts reach half a period.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    f_c, z_ph, f2 = 2.0, 3.0, 1.6
    z_l2 = z_ph + 0.8
    z_scr = z_l2 + f2
    y_p = 0.45
    lenses = [(f_c, f_c), (z_l2, f2)]
    colours = [STAR, IMAGING, STAR2]
    y_src = [0.3, 0.0, -0.3]
    # the extended source
    ax.add_patch(_patches().Rectangle((-0.06, -0.38), 0.12, 0.76, facecolor="#fbe7b0",
                                      edgecolor=ILLUMINATION, lw=1, zorder=3))
    for ys, col in zip(y_src, colours):
        point_source(ax, 0.0, ys, color=col, size=55)
        for yp in (y_p, -y_p):
            h = yp + ys/f_c*(z_ph - f_c)           # height on the collimator
            draw_ray(ax, lenses, 0.0, ys, (h - ys)/f_c, z_scr, color=col, lw=0.9, alpha=0.8,
                     arrow=False)
    lens(ax, f_c, 1.0, text="collimator f$_c$")
    stop(ax, z_ph, 1.0, openings=((y_p - 0.04, y_p + 0.04),), width=0.05,
         text="two pinholes\nd apart", text_y=-1.45)
    lens(ax, z_l2, 1.0, text="lens f")
    ax.plot([z_scr, z_scr], [-1.0, 1.0], color="0.35", lw=2.5, zorder=4)
    label(ax, z_scr, -1.15, "screen", va="top")
    label(ax, 0.0, -0.5, "incoherent source,\ndiameter D", va="top")
    # fringes of each source point and their sum
    y = np.linspace(-0.95, 0.95, 300)
    period = 0.5
    for ys, col in zip(y_src, colours):
        _fringe_curve(ax, z_scr + 0.15, y, 2*np.pi*ys/0.75, period, 0.22, color=col, lw=0.9)
    phases = np.linspace(-0.3, 0.3, 21)*2*np.pi/0.75
    total = np.mean([1 + np.cos(2*np.pi*y/period + p) for p in phases], axis=0)
    ax.plot(z_scr + 0.15 + 0.22*total, y, color="k", lw=1.8, zorder=6)
    label(ax, z_scr + 0.4, 1.05, "fringes of each point\nand their sum (black)", va="bottom",
          fontsize=8)
    label(ax, z_ph + 0.4, 0.95, "each point: a plane wave\ntilted by ξ/f$_c$", fontsize=8,
          va="bottom")
    _finish(ax, (-0.6, z_scr + 0.9), (-1.55, 1.45), title)


def vcz_setup(ax, title=None):
    """Sketch of the van Cittert-Zernike theorem (section 2 of notebook 74).

    An incoherent source of diameter D: every point radiates a spherical
    wave with its own random phase (drawn in different colours). A distance
    z away the fields at two points P1 and P2 are correlated as long as
    their separation is small compared with lambda*z/D; the degree of
    coherence is the Fourier transform of the source intensity.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    from matplotlib.patches import Arc

    z_obs = 4.2
    ys = [0.32, 0.12, -0.08, -0.28]
    colours = [STAR, IMAGING, STAR2, EMISSION]
    ax.add_patch(_patches().Rectangle((-0.06, -0.4), 0.12, 0.8, facecolor="#fbe7b0",
                                      edgecolor=ILLUMINATION, lw=1, zorder=3))
    for i, (y0, col) in enumerate(zip(ys, colours)):
        point_source(ax, 0.0, y0, color=col, size=45)
        for r in np.arange(0.6 + 0.2*i, z_obs - 0.1, 0.9):
            theta = np.degrees(np.arcsin(min(1.1/r, 1.0)))
            ax.add_patch(Arc((0.0, y0), 2*r, 2*r, theta1=-theta, theta2=theta, color=col,
                             lw=0.7, alpha=0.6, zorder=2))
    ax.plot([z_obs, z_obs], [-1.3, 1.3], color="0.55", ls=":", lw=1)
    for y0, name in [(0.55, "P$_1$"), (0.15, "P$_2$")]:
        ax.plot([z_obs], [y0], "o", color="k", ms=5, zorder=6)
        label(ax, z_obs + 0.12, y0, name, ha="left")
    arrow(ax, z_obs + 0.5, 0.15, z_obs + 0.5, 0.55, style="<|-|>", color="0.3")
    label(ax, z_obs + 0.6, 0.35, "Δr", ha="left", fontsize=9)
    arrow(ax, z_obs + 0.25, -0.75, z_obs + 0.25, 0.75, style="<|-|>", color=STRAY, lw=0.8)
    label(ax, z_obs + 0.35, -0.6, "coherence width\n≈ λz/D", fontsize=8, color=STRAY,
          va="center", ha="left")
    arrow(ax, 0.0, -1.0, z_obs, -1.0, style="<|-|>", color="0.4", text="z",
          text_pos=(z_obs/2, -1.08), va="top")
    label(ax, 0.0, -0.5, "incoherent source,\ndiameter D,\nrandom phases", va="top", fontsize=8)
    label(ax, z_obs, 1.35, "observation plane:\nμ(Δr) = FT{I(ξ)} at Δr/λz", va="bottom",
          fontsize=8)
    _finish(ax, (-0.8, z_obs + 1.6), (-1.4, 1.75), title)


def stellar_interferometer(ax, title=None):
    """Sketch of Michelson's stellar interferometer (section 3 of notebook 74).

    Light of a star falls on two outer mirrors a baseline B apart. They and
    two inner mirrors a fixed distance h apart send it into a telescope,
    which focuses both beams into one image with fringes of period
    lambda*f/h. Light from a second point of the stellar disk, at the angle
    alpha, arrives at one outer mirror with the extra path B*alpha and
    shifts its fringes by the phase k*B*alpha; the baseline sets the
    visibility.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    B, hh = 4.4, 1.6
    y_m, y_tel, y_foc = 3.0, 1.9, -0.1
    w = 0.12
    f = y_tel - y_foc
    # starlight onto the outer mirrors
    for zc in (-B/2, B/2):
        band(ax, [zc - w, zc - w], [4.3, y_m], [zc + w, zc + w], [4.3, y_m], color=STAR,
             alpha=0.3)
    # the folded paths to the telescope
    for s in (-1, 1):
        band(ax, [s*B/2, s*hh/2], [y_m + w, y_m + w], [s*B/2, s*hh/2], [y_m - w, y_m - w],
             color=STAR, alpha=0.3)
        band(ax, [s*hh/2 - w, s*hh/2 - w, 0.0], [y_m, y_tel, y_foc],
             [s*hh/2 + w, s*hh/2 + w, 0.0], [y_m, y_tel, y_foc], color=STAR, alpha=0.3)
        mirror(ax, s*B/2, y_m, 0.55, s*np.pi/4)
        mirror(ax, s*hh/2, y_m, 0.55, s*np.pi/4)
    rotated_lens(ax, 0.0, y_tel, 1.25, rotate=90)
    label(ax, 1.4, y_tel, "telescope, focal length f", ha="left")
    # wavefront of the tilted star across the baseline
    ax.plot([-B/2 - 0.2, B/2 + 0.2], [4.0 + 0.06*(-B/2 - 0.2), 4.0 + 0.06*(B/2 + 0.2)],
            color=STAR2, lw=1.2)
    label(ax, B/2 + 0.3, 4.3, "star point at angle α:\nextra path B·α", ha="left", fontsize=8,
          color=STAR2)
    label(ax, -B/2 - 0.1, 4.45, "starlight", fontsize=8, va="bottom")
    arrow(ax, -B/2, 2.4, B/2, 2.4, style="<|-|>", color="0.3")
    label(ax, 0.0, 2.47, "baseline B: sets the visibility", va="bottom", fontsize=9)
    arrow(ax, -hh/2, 3.4, hh/2, 3.4, style="<|-|>", color="0.3")
    label(ax, 0.0, 3.47, "h", va="bottom", fontsize=9)
    label(ax, -B/2 - 0.35, y_m, "outer\nmirror", ha="right", fontsize=8)
    label(ax, 0.0, y_m, "inner mirrors", fontsize=8)
    # fringes in the focal plane
    zz = np.linspace(-0.6, 0.6, 400)
    env = np.exp(-(zz/0.3)**2)
    ax.plot(zz, y_foc - 0.65 + 0.45*env*(1 + np.cos(2*np.pi*zz/0.1))/2, color=STAR, lw=1.2)
    ax.plot([-0.65, 0.65], [y_foc - 0.65]*2, color="0.6", lw=0.6)
    label(ax, 0.75, y_foc - 0.5, "fringes in the focus,\nperiod λf/h (set by h)", ha="left",
          fontsize=8)
    _finish(ax, (-B/2 - 1.0, B/2 + 2.2), (-0.9, 4.8), title)


def michelson_interferometer(ax, title=None):
    """Sketch of a Michelson interferometer measuring temporal coherence
    (section 4 of notebook 74).

    A collimated beam from a broadband source is split; one mirror is fixed,
    the other moves by Delta/2, which gives the optical path difference
    Delta between the arms. The detector sees I(Delta) = 1 + Re gamma(Delta/c):
    fringes under the envelope abs(gamma), the Fourier transform of the
    spectrum.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    w = 0.3
    z_bs = 2.4
    point_source(ax, -0.6, 0.0, color=ILLUMINATION, size=90, text="LED, lamp\nor star")
    for y0 in (w, -w):
        ax.plot([-0.6, 0.4], [0.0, y0], color=ILLUMINATION, lw=0.8)
    band(ax, [0.4, z_bs], [w, w], [0.4, z_bs], [-w, -w], color=ILLUMINATION)
    lens(ax, 0.4, 0.45, text="collimator")
    # fixed arm up, moving arm to the right, output down
    band(ax, [z_bs - w, z_bs - w], [0.0, 2.0], [z_bs + w, z_bs + w], [0.0, 2.0],
         color=ILLUMINATION)
    band(ax, [z_bs, 4.6], [w, w], [z_bs, 4.6], [-w, -w], color=ILLUMINATION)
    band(ax, [z_bs - w, z_bs - w], [0.0, -1.6], [z_bs + w, z_bs + w], [0.0, -1.6],
         color="#8e5ba8")
    mirror(ax, z_bs, 2.05, 0.9, np.pi/2, text="fixed mirror", text_dy=0.35)
    mirror(ax, 4.65, 0.0, 0.9, 0.0)
    label(ax, 4.65, -0.6, "moving mirror", va="top")
    arrow(ax, 4.85, 0.6, 5.35, 0.6, style="<|-|>", color="0.3")
    label(ax, 5.1, 0.68, "Δ/2", va="bottom")
    beam_splitter(ax, z_bs, 0.0, 0.9, "/", text="beam\nsplitter", text_dy=-0.4)
    rotated_detector(ax, z_bs, -1.7, 0.35)
    label(ax, z_bs - 0.5, -1.75, "detector:\nI(Δ) = 1 + Re γ(Δ/c)", ha="right", fontsize=8)
    # a sketch of the interferogram
    d = np.linspace(-1.1, 1.1, 500)
    ax.plot(4.0 + 0.9*d, -1.5 + 0.35*np.exp(-(d/0.4)**2)*np.cos(2*np.pi*d/0.09), color="#8e5ba8",
            lw=0.8)
    label(ax, 4.0, -1.95, "interferogram vs. Δ:\nenvelope = abs(γ), width ~ λ²/Δλ", fontsize=8,
          va="top")
    _finish(ax, (-1.1, 5.7), (-2.5, 2.6), title)


def hbt_interferometer(ax, title=None):
    """Sketch of a Hanbury Brown-Twiss intensity interferometer (section 5
    of notebook 74).

    Two light collectors a baseline B apart focus starlight onto
    photodetectors. Their photocurrents are band-pass filtered (electronic
    bandwidth Delta f), one is delayed to equalize the paths, and the
    product of the fluctuations is averaged. The normalized correlation is
    abs(mu(B))**2, reduced by the ratio of the optical coherence time to the
    detector response time. Only paths good to about c/Delta f (centimetres)
    are needed, and atmospheric phases drop out.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    B = 4.4
    y_col, y_det = 2.4, 1.2
    for s in (-1, 1):
        zc = s*B/2
        band(ax, [zc - 0.55, zc - 0.55, zc], [4.0, y_col, y_det], [zc + 0.55, zc + 0.55, zc],
             [4.0, y_col, y_det], color=STAR, alpha=0.3)
        sphere_pts = np.linspace(-0.6, 0.6, 50)
        ax.plot(zc + sphere_pts, y_col - 0.12 + 0.25*sphere_pts**2, color="0.3", lw=3)
        rotated_detector(ax, zc, y_det - 0.05, 0.18, rotate=90)
        label(ax, zc + s*0.7, y_col + 0.1, "light\ncollector", ha="left" if s > 0 else "right",
              fontsize=8)
        label(ax, zc + s*0.3, y_det - 0.05, "photo-\ndetector", ha="left" if s > 0 else "right",
              fontsize=8)
        ax.plot([zc, zc, s*0.35], [y_det - 0.15, 0.2, 0.2], color="0.3", lw=1)
        ax.add_patch(mpatches.FancyBboxPatch((zc - 0.45, 0.5), 0.9, 0.35,
                                             boxstyle="round,pad=0.02", facecolor="white",
                                             edgecolor="0.3", zorder=4))
        label(ax, zc, 0.675, "filter Δf" if s < 0 else "filter Δf, delay", fontsize=8, zorder=5)
    ax.add_patch(mpatches.Circle((0.0, 0.2), 0.3, facecolor="white", edgecolor="0.3",
                                 zorder=4))
    label(ax, 0.0, 0.2, "×", fontsize=14, zorder=5)
    ax.plot([0.0, 0.0], [-0.1, -0.5], color="0.3", lw=1)
    ax.add_patch(mpatches.FancyBboxPatch((-1.3, -1.05), 2.6, 0.5, boxstyle="round,pad=0.02",
                                         facecolor="white", edgecolor="0.3", zorder=4))
    label(ax, 0.0, -0.8, "average ⟨ΔI₁ΔI₂⟩ / ⟨I₁⟩⟨I₂⟩ ∝ abs(μ(B))²", fontsize=8, zorder=5)
    arrow(ax, -B/2, 4.25, B/2, 4.25, style="<|-|>", color="0.3")
    label(ax, 0.0, 4.32, "baseline B", va="bottom")
    label(ax, 0.0, 3.3, "no beam combination:\npaths need only match\nto ~ c/Δf (cm)",
          fontsize=8)
    _finish(ax, (-B/2 - 1.6, B/2 + 1.6), (-1.3, 4.7), title)

SUN = "#e8a33d"
BOW_RED = "#c0392b"
BOW_VIOLET = "#6a3d9a"


def _sun(ax, z, y, r=0.32):
    mpatches = _patches()
    ax.add_patch(mpatches.Circle((z, y), r, facecolor=SUN, edgecolor="#b97a1d", lw=1, zorder=4))
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax.plot([z + 1.25*r*np.cos(a), z + 1.6*r*np.cos(a)],
                [y + 1.25*r*np.sin(a), y + 1.6*r*np.sin(a)], color=SUN, lw=1.2, zorder=4)


def _person(ax, z, y, height=0.9, color="0.2"):
    mpatches = _patches()
    head = height/7
    ax.add_patch(mpatches.Circle((z, y + height - head), head, facecolor=color,
                                 edgecolor="none", zorder=5))
    ax.plot([z, z], [y + 0.42*height, y + height - 2*head], color=color, lw=2.2, zorder=5)
    for dz in (-0.12, 0.12):
        ax.plot([z, z + dz*height], [y + 0.42*height, y], color=color, lw=2.0, zorder=5)
        ax.plot([z, z + 1.4*dz*height], [y + 0.72*height, y + 0.5*height], color=color, lw=1.6,
                zorder=5)

    return z, y + height - head


def rainbow_observer(ax, sun_elevation=25.0, title=None):
    """Sketch of where a rainbow appears (the geometry of notebook 71).

    The Sun shines from behind the observer onto a curtain of rain. The
    line from the Sun through the observer's eye meets the ground at the
    antisolar point, the centre of the bows. Drops that send their primary
    rainbow ray to the eye lie on a cone of about 42 deg around this line
    (red outside, violet inside), those of the secondary bow on a cone of
    about 51 deg (colours reversed). Between the two cones no light of
    either order reaches the eye: Alexander's dark band.

    Parameters
    ----------
    ax : matplotlib Axes
    sun_elevation : number, optional
        Elevation of the Sun in degrees.
    title : str, optional
    """

    mpatches = _patches()
    h = np.radians(sun_elevation)
    z_rain, z_end = 7.0, 8.4
    ax.plot([-3.6, z_end], [0, 0], color="0.35", lw=1.2, zorder=2)
    ax.add_patch(mpatches.Rectangle((-3.6, -1.6), z_end + 3.6, 1.6, facecolor="#e7efe0",
                                    edgecolor="none", zorder=0))
    eye = _person(ax, 0.0, 0.0)
    # sun rays travel along (cos h, -sin h)
    d = np.array([np.cos(h), -np.sin(h)])
    sun = np.array(eye) - 3.0*d
    _sun(ax, *sun)
    # rain curtain
    ax.add_patch(mpatches.Rectangle((z_rain - 0.6, 0), 1.2, 4.6, facecolor="#dfe7ef",
                                    edgecolor="none", zorder=0.5))
    rng = np.random.default_rng(3)
    for zr, yr in zip(rng.uniform(z_rain - 0.55, z_rain + 0.55, 70), rng.uniform(0.1, 4.4, 70)):
        ax.plot([zr, zr - 0.03], [yr, yr - 0.16], color="#7f9db9", lw=0.8, zorder=1)
    label(ax, z_rain, 4.75, "rain", fontsize=9, va="bottom")
    # antisolar line
    t_ground = eye[1]/np.sin(h)
    anti = np.array(eye) + t_ground*d
    ax.plot([sun[0] + 0.5*d[0], anti[0]], [sun[1] + 0.5*d[1], anti[1]], color="0.4", lw=0.9,
            ls="--", zorder=2)
    ax.plot(*anti, marker="x", color="0.2", ms=7, mew=1.5, zorder=5)
    label(ax, anti[0], anti[1] - 0.12, "antisolar\npoint", fontsize=8, va="top")

    def ray_to_eye(rho_deg, color, lw=1.4):
        a = -h + np.radians(rho_deg)     # direction from the eye to the drop
        e = np.array([np.cos(a), np.sin(a)])
        t = (z_rain - eye[0])/e[0]
        drop = np.array(eye) + t*e
        ax.plot([eye[0], drop[0]], [eye[1], drop[1]], color=color, lw=lw, zorder=3)
        ax.plot([drop[0] - 2.4*d[0], drop[0]], [drop[1] - 2.4*d[1], drop[1]], color=SUN, lw=0.9,
                alpha=0.9, zorder=2)
        ax.add_patch(mpatches.Circle(drop, 0.07, facecolor="#9fc5e0", edgecolor=EDGE, lw=0.8,
                                     zorder=4))
        return drop

    # Alexander's band between the red edge of the primary and the red edge of the secondary
    a1, a2 = -h + np.radians(42.3), -h + np.radians(50.1)
    r_far = (z_rain - eye[0])/np.cos(a2)
    ax.add_patch(mpatches.Polygon([eye, (eye[0] + r_far*np.cos(a2), eye[1] + r_far*np.sin(a2)),
                                   (z_rain, eye[1] + (z_rain - eye[0])*np.tan(a1))],
                                  closed=True, facecolor="0.55", edgecolor="none", alpha=0.25,
                                  zorder=1.5))
    p_red = ray_to_eye(42.3, BOW_RED)
    p_vio = ray_to_eye(40.6, BOW_VIOLET)
    s_red = ray_to_eye(50.1, BOW_RED)
    s_vio = ray_to_eye(53.5, BOW_VIOLET)
    label(ax, z_rain + 0.75, (p_red[1] + p_vio[1])/2, "primary bow\n(red outside)", fontsize=8,
          ha="left")
    label(ax, z_rain + 0.75, (s_red[1] + s_vio[1])/2, "secondary bow\n(red inside)", fontsize=8,
          ha="left")
    label(ax, z_rain + 0.75, (p_red[1] + s_red[1])/2, "Alexander's\ndark band", fontsize=8,
          ha="left", color="0.35")
    # angles at the eye
    r_arc = 2.0
    for rho, txt in ((42, "42°"), (51, "51°")):
        aa = np.linspace(-h, -h + np.radians(rho), 40)
        ax.plot(eye[0] + r_arc*np.cos(aa), eye[1] + r_arc*np.sin(aa), color="0.3", lw=0.8)
        am = -h + np.radians(rho) - 0.06
        label(ax, eye[0] + (r_arc + 0.3)*np.cos(am), eye[1] + (r_arc + 0.3)*np.sin(am), txt,
              fontsize=8)
        r_arc += 0.9
    label(ax, sun[0], sun[1] + 0.6, "Sun behind\nthe observer", fontsize=8, va="bottom")
    label(ax, -1.2, eye[1] + 1.2*np.tan(h) + 0.12, "sunlight", fontsize=8, color="#b97a1d",
          rotation=-sun_elevation, ha="center", va="bottom")
    _finish(ax, (-3.6, z_end + 1.6), (-1.6, 5.2), title)


def glory_aircraft(ax, title=None):
    """Sketch of the glory seen from an aircraft (notebook 71).

    The Sun shines from behind the aircraft onto a deck of cloud. Its shadow
    falls on the cloud around the antisolar point; the cloud droplets
    (radius ~10 um) scatter light straight back to the observer in coloured
    rings of a few degrees around the shadow, the glory, and further out,
    at about 40 deg, in the broad white cloud bow (a fogbow).

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    h = np.radians(60.0)
    d = np.array([np.cos(h), -np.sin(h)])     # sunlight direction
    plane_pos = np.array([0.0, 3.0])
    # cloud deck
    zz = np.linspace(-3.0, 6.5, 200)
    top = 0.0 + 0.08*np.sin(3.1*zz) + 0.05*np.sin(7.3*zz + 1)
    ax.fill_between(zz, -1.3, top, color="#eef1f4", edgecolor="0.6", lw=0.8, zorder=1)
    rng = np.random.default_rng(5)
    ax.scatter(rng.uniform(-2.9, 6.4, 150), rng.uniform(-1.2, -0.1, 150), s=3, color="#9fb3c8",
               zorder=1.2)
    label(ax, 5.6, -0.65, "cloud\n(droplets ~10 µm)", fontsize=8)
    # aircraft (side view)
    pz, py = plane_pos
    body = [(pz - 0.9, py), (pz + 0.8, py + 0.02), (pz + 1.0, py - 0.06), (pz + 0.8, py - 0.12),
            (pz - 0.8, py - 0.1)]
    ax.add_patch(mpatches.Polygon(body, closed=True, facecolor="0.3", edgecolor="none", zorder=5))
    ax.add_patch(mpatches.Polygon([(pz - 0.75, py - 0.02), (pz - 0.95, py + 0.35), (pz - 0.82, py + 0.35),
                                   (pz - 0.55, py - 0.02)], closed=True, facecolor="0.3",
                                  edgecolor="none", zorder=5))
    ax.add_patch(mpatches.Polygon([(pz - 0.1, py - 0.05), (pz + 0.25, py - 0.05), (pz - 0.35, py - 0.3),
                                   (pz - 0.55, py - 0.3)], closed=True, facecolor="0.45",
                                  edgecolor="none", zorder=5))
    _sun(ax, *(plane_pos - 1.95*d), r=0.24)
    for off in (-0.5, 0.0, 0.5):
        p0 = plane_pos - 1.5*d + off*np.array([d[1], -d[0]])
        ax.plot([p0[0], p0[0] + 1.0*d[0]], [p0[1], p0[1] + 1.0*d[1]], color=SUN, lw=1.0)
    # antisolar direction and the shadow
    t = (py - 0.0)/np.sin(h)
    anti = plane_pos + t*d
    ax.plot([pz + 0.4*d[0], anti[0]], [py + 0.4*d[1], anti[1]], color="0.4", ls="--", lw=0.9,
            zorder=2)
    ax.add_patch(mpatches.Ellipse(anti, 0.55, 0.1, facecolor="0.35", edgecolor="none", zorder=3))
    label(ax, anti[0] + 0.38, anti[1] + 0.06, "shadow", fontsize=8, ha="left", va="bottom")

    def cone(rho_deg, colors, lw=1.3):
        for j, c in enumerate(colors):
            for sgn in (1, -1):
                a = np.arctan2(d[1], d[0]) + sgn*np.radians(rho_deg + 0.8*j)
                e = np.array([np.cos(a), np.sin(a)])
                tt = (py - 0.0)/(-e[1])
                ax.plot([pz, pz + tt*e[0]], [py, py + tt*e[1]], color=c, lw=lw, zorder=2.5)

    cone(4.0, ["#3b6fb6", "#2ca02c", BOW_RED], lw=1.0)
    cone(40.0, ["0.55"], lw=2.0)
    label(ax, anti[0] + 0.45, 1.25, "glory: coloured\nrings of a few\ndegrees", fontsize=8,
          ha="left")
    a = np.arctan2(d[1], d[0]) + np.radians(40.0)
    e = np.array([np.cos(a), np.sin(a)])
    q = plane_pos + 1.6*e
    label(ax, q[0] + 0.15, q[1] + 0.05, "cloud bow, ~40°\n(white fogbow)", fontsize=8, ha="left")
    label(ax, -1.9, 4.75, "Sun behind\nthe aircraft", fontsize=8)
    _finish(ax, (-3.0, 6.5), (-1.3, 5.3), title)


def _wavy(s, amplitude, n_waves, phase=0.0):
    """A smooth random-looking undulation along s (for turbulent wavefronts)."""

    s = np.asarray(s, dtype=float)
    span = np.ptp(s) if np.ptp(s) > 0 else 1.0
    u = (s - s.min())/span
    return amplitude*(np.sin(2*np.pi*n_waves*u + phase) + 0.5*np.sin(2*np.pi*2.3*n_waves*u + 1.7*phase + 0.9)
                      + 0.3*np.sin(2*np.pi*0.45*n_waves*u + 2.1))/1.8


def twinkling_star(ax, title=None):
    """Sketch of starlight passing turbulent layers (the setup of notebook 72).

    A plane wave from a star (at the top) crosses a few thin turbulent layers
    at different heights, each moving with its own wind. Every layer
    corrugates the wavefront; over the kilometres down to the ground the
    corrugations focus and defocus the light into a moving pattern of
    shadow bands with a typical size of the Fresnel scale sqrt(λh) (a few
    centimetres). A small pupil (the eye) sees the intensity fluctuate, the
    star twinkles; a large telescope averages over many bands but sees a
    wavefront corrugated on the scale r0, which spreads the image into a
    speckle cloud (seeing).

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    x = np.linspace(-2.6, 2.6, 300)
    point_source(ax, 0.0, 5.1, color=ILLUMINATION, size=260)
    label(ax, 0.35, 5.1, "star", ha="left", fontsize=9)
    # plane wavefronts above the atmosphere
    for y0 in (4.45, 4.15):
        ax.plot(x, np.full_like(x, y0), color="0.2", lw=1.2)
    label(ax, 2.7, 4.3, "plane\nwavefronts", ha="left", fontsize=8)
    layers = [(3.4, "jet stream, ~10 km", 25, 0.0), (2.2, "~4 km", 10, 1.3),
              (1.05, "boundary layer", 5, 2.6)]
    amp = 0.0
    for i, (y0, text, v, ph) in enumerate(layers):
        band(ax, x, y0 + 0.13 + _wavy(x, 0.05, 5, ph), x, y0 - 0.13 + _wavy(x, 0.05, 5, ph + 1),
             color=IMAGING, alpha=0.18, edges=False)
        label(ax, -2.75, y0, text, ha="right", fontsize=8)
        arrow(ax, 1.7, y0 + 0.27, 1.7 + 0.025*v, y0 + 0.27, color=IMAGING, lw=1.0)
        label(ax, 1.7 + 0.025*v + 0.08, y0 + 0.27, "wind", fontsize=7, ha="left",
              color=IMAGING)
        # the wavefront below the layer gets more and more corrugated
        amp += 0.05
        y_w = y0 - 0.5
        ax.plot(x, y_w + _wavy(x, amp, 6 + 2*i, 0.7*i), color="0.2", lw=1.1)
    # focused and defocused light near the ground: shadow bands
    xs = np.linspace(-2.6, 2.6, 400)
    pattern = 0.5 + 0.5*np.tanh(2.5*(_wavy(xs, 1.0, 14, 0.4) + 0.3*_wavy(xs, 1.0, 31, 1.1)))
    for x0, x1, p in zip(xs[:-1], xs[1:], pattern[:-1]):
        ax.add_patch(mpatches.Rectangle((x0, -0.1), x1 - x0, 0.22,
                                        facecolor=(1, 0.85 - 0.25*p, 0.35 + 0.5*(1 - p)),
                                        edgecolor="none", alpha=0.3 + 0.7*p, zorder=2))
    label(ax, 2.7, 0.0, "shadow bands\n(size ~ √(λh))", ha="left", fontsize=8)
    ax.plot([-2.8, 2.8], [-0.18, -0.18], color="0.35", lw=1.5)
    # an eye and a telescope at the ground
    ax.add_patch(mpatches.Ellipse((-1.6, -0.48), 0.42, 0.22, facecolor="white", edgecolor="0.2",
                                  lw=1, zorder=4))
    ax.add_patch(mpatches.Circle((-1.6, -0.48), 0.07, facecolor="0.15", zorder=5))
    label(ax, -1.6, -0.7, "eye: small pupil,\nthe star twinkles", fontsize=8, va="top")
    ax.add_patch(mpatches.Polygon([(0.55, -0.25), (1.75, -0.25), (1.35, -0.85), (0.95, -0.85)],
                                  closed=True, facecolor="0.85", edgecolor="0.3", lw=1, zorder=4))
    ax.plot([0.55, 1.75], [-0.25, -0.25], color=METAL, lw=3, zorder=5)
    label(ax, 1.15, -0.95, "telescope: averages the bands,\nbut images a speckle cloud",
          fontsize=8, va="top")
    _finish(ax, (-4.2, 3.9), (-1.5, 5.5), title)


def adaptive_optics_system(ax, title=None):
    """Sketch of an adaptive optics system (the setup of notebook 72).

    The telescope (drawn as a lens) images the star; a collimator makes the
    beam parallel and images the pupil onto the deformable mirror (DM). A
    dichroic beam splitter sends the visible light to a Shack-Hartmann
    wavefront sensor and passes the infrared light to the science camera.
    A real-time computer turns the spot shifts into actuator commands
    (reconstructor and integrator) and pushes the DM to the conjugate of
    the measured wavefront, closing the loop: the sensor sees the residual
    behind the DM.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    h_t, f_t = 1.0, 2.4
    z_col, f_col = f_t + 0.8, 0.8
    w = h_t*f_col/f_t
    z_dm = z_col + 1.1
    y_bs = -1.6
    # turbulent input wavefronts
    for z0 in (-1.3, -0.95, -0.6):
        s = np.linspace(-h_t, h_t, 61)
        ax.plot(z0 + _wavy(s, 0.07, 2.5, 3*z0), s, color="0.2", lw=1.1)
    label(ax, -0.95, 1.2, "distorted\nwavefront", fontsize=8, va="bottom")
    # telescope focus and collimator
    z, yt = paraxial_ray([(0.0, f_t), (z_col, f_col)], -1.4, h_t, 0.0, z_dm - w)
    z, yb = paraxial_ray([(0.0, f_t), (z_col, f_col)], -1.4, -h_t, 0.0, z_dm - w)
    band(ax, z, yt, z, yb, color=LASER)
    lens(ax, 0.0, 1.1*h_t, text="telescope")
    lens(ax, z_col, 1.25*w, text="collimator", text_y=-0.75)
    # beam from the DM down to the dichroic
    band(ax, [z_dm - w, z_dm - w], [-w + 2*w, y_bs + w], [z_dm + w, z_dm + w], [w, y_bs - w],
         color=LASER, edges=False)
    ax.add_patch(mpatches.Polygon([(z_dm - w, w), (z_dm + w, -w), (z_dm + w, w)], closed=True,
                                  facecolor=LASER, edgecolor="none", alpha=0.22, zorder=1))
    # deformable mirror: a wavy face at 45° with actuators behind
    s = np.linspace(-1.25*w, 1.25*w, 61)
    d = _wavy(s, 0.035, 2.0, 0.8)
    e = np.array([1, -1])/np.sqrt(2)
    n = np.array([1, 1])/np.sqrt(2)
    ax.plot(z_dm + s*e[0] + d*n[0], s*e[1] + d*n[1], color=METAL, lw=3, zorder=4)
    for si in np.linspace(-1.0*w, 1.0*w, 6):
        p0 = np.array([z_dm, 0.0]) + si*e + 0.06*n
        p1 = p0 + 0.28*n
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.45", lw=1.6, zorder=4)
    label(ax, z_dm - 0.1, 0.78, "deformable\nmirror (DM)", ha="right", fontsize=9)
    # dichroic: visible to the Shack-Hartmann (along +z), infrared down to the camera
    beam_splitter(ax, z_dm, y_bs, 2.6*w, orientation="\\")
    label(ax, z_dm - 0.5, y_bs - 0.05, "dichroic", ha="right", fontsize=9)
    z_la = z_dm + 1.3
    band(ax, [z_dm - w, z_la], [y_bs + w, y_bs + w], [z_dm + w, z_la], [y_bs - w, y_bs - w],
         color="#e0a000", edges=False, alpha=0.25)
    ax.add_patch(mpatches.Polygon([(z_dm - w, y_bs + w), (z_dm + w, y_bs + w), (z_dm + w, y_bs - w)],
                                  closed=True, facecolor="#e0a000", edgecolor="none", alpha=0.25,
                                  zorder=1))
    hl = 1.1*w
    p = 2*hl/6
    c = np.linspace(-hl + p/2, hl - p/2, 6)
    ax.add_patch(mpatches.Rectangle((z_la - 0.02, y_bs - hl), 0.04, 2*hl, facecolor=GLASS,
                                    edgecolor=EDGE, lw=0.8, zorder=3))
    for ci in c:
        yy = np.linspace(-p/2, p/2, 15)
        ax.plot(z_la + 0.02 + 0.35*p*(1 - (2*yy/p)**2)/2, y_bs + ci + yy, color=EDGE, lw=0.8,
                zorder=3)
        ax.plot([z_la, z_la + 0.45], [y_bs + ci + p/2.5, y_bs + ci + 0.02*np.sin(5*ci)],
                color="#e0a000", lw=0.7)
        ax.plot([z_la, z_la + 0.45], [y_bs + ci - p/2.5, y_bs + ci + 0.02*np.sin(5*ci)],
                color="#e0a000", lw=0.7)
    ax.add_patch(mpatches.Rectangle((z_la + 0.45, y_bs - hl), 0.06, 2*hl, facecolor="0.35",
                                    edgecolor="none", zorder=4))
    label(ax, z_la + 0.25, y_bs - hl - 0.1, "Shack-Hartmann\n(visible)", fontsize=9, va="top")
    # science path: infrared down to a camera lens and the camera
    y_cam_lens, y_cam = y_bs - 1.0, y_bs - 2.0
    ax.add_patch(mpatches.Polygon([(z_dm - w, y_bs - w), (z_dm - w, y_cam_lens), (z_dm, y_cam),
                                   (z_dm + w, y_cam_lens), (z_dm + w, y_bs + w)],
                                  closed=True, facecolor=LASER, edgecolor="none", alpha=0.22,
                                  zorder=1))
    ax.plot([z_dm - w, z_dm, z_dm + w], [y_cam_lens, y_cam, y_cam_lens], color=LASER, lw=0.8)
    wavefront(ax, z_dm, -0.75, w, vertical=False)
    label(ax, z_dm + w + 0.08, -0.75, "corrected", fontsize=8, ha="left")
    rotated_lens(ax, z_dm, y_cam_lens, 1.2*w, text="camera lens", text_offset=-1.5*w)
    rotated_detector(ax, z_dm, y_cam, 0.5*w, width=0.07)
    label(ax, z_dm, y_cam - 0.12, "science camera (infrared):\nsharp image", fontsize=9, va="top")
    # loop: computer from the sensor to the DM
    zc, yc = z_la + 1.75, 0.75
    ax.add_patch(mpatches.FancyBboxPatch((zc - 0.75, yc - 0.32), 1.5, 0.64,
                                         boxstyle="round,pad=0.03", facecolor="#eef3e8",
                                         edgecolor="0.35", lw=1, zorder=4))
    label(ax, zc, yc, "real-time computer\nslopes → commands\n(integrator)", fontsize=8,
          zorder=5)
    arrow(ax, z_la + 0.55, y_bs + 0.2, zc, yc - 0.34, color=EMISSION, lw=1.3)
    arrow(ax, zc - 0.77, yc, z_dm + 0.42, 0.42, color=EMISSION, lw=1.3)
    label(ax, z_la + 1.25, y_bs + 0.9, "spot\nshifts", fontsize=8, color=EMISSION, ha="left")
    label(ax, (zc - 0.77 + z_dm + 0.42)/2, 0.68, "actuator\ncommands", fontsize=8,
          color=EMISSION, va="bottom")
    _finish(ax, (-1.6, zc + 0.9), (y_cam - 0.6, 1.6), title)


def _slm(ax, z, h, n=10, reference=False, text=None, text_y=None, fontsize=9):
    # transmissive SLM drawn as a column of segments with random grey phases;
    # with a reference, the outer segments stay flat (light grey, hatched)
    mpatches = _patches()
    rng = np.random.default_rng(7)
    width = 0.14
    for k in range(n):
        y0 = -h + 2*h*k/n
        outer = reference and (k < 2 or k >= n - 2)
        ax.add_patch(mpatches.Rectangle((z - width/2, y0), width, 2*h/n,
                                        facecolor="0.93" if outer else str(0.35 + 0.5*rng.random()),
                                        edgecolor="0.3", lw=0.5, hatch="///" if outer else None,
                                        zorder=3))
    if text:
        label(ax, z, -1.15*h if text_y is None else text_y, text, fontsize=fontsize, va="top")


def wavefront_shaping_setup(ax, medium="diffuser", reference=False, feedback=True, title=None):
    """Sketch of wavefront shaping through a scattering medium (notebook 73).

    A collimated laser beam falls onto a segmented phase SLM, which a 4f
    relay images onto the entrance face of the medium. A lens behind the
    medium images its far field (or, for a fibre, its output facet) onto a
    camera. Scattered light (thin red lines) leaves the medium in all
    directions; the shaped part (red band) converges onto the target pixel,
    whose intensity is fed back to the SLM.

    Parameters
    ----------
    ax : matplotlib Axes
    medium : {"diffuser", "slab", "fibre"}, optional
        A thin ground glass diffuser, a thick random slab, or a multimode
        fibre with input and output coupling lenses.
    reference : bool, optional
        Keep the outer SLM segments flat as an internal reference (for
        measuring the transmission matrix by phase stepping).
    feedback : bool, optional
        Draw the feedback loop from the target pixel to the SLM.
    title : str, optional
    """

    if medium not in ("diffuser", "slab", "fibre"):
        raise ValueError("medium must be 'diffuser', 'slab' or 'fibre'.")
    mpatches = _patches()
    h = 0.6                                   # beam half height at the SLM
    # relay: SLM -> medium, magnification 1/2 (1/7 onto the fibre core)
    z_slm, f1, f2 = 0.0, 1.4, (0.2 if medium == "fibre" else 0.7)
    z_l1 = z_slm + f1
    z_l2 = z_l1 + f1 + f2
    z_med = z_l2 + f2
    lenses = [(z_l1, f1), (z_l2, f2)]
    band(ax, [-1.3, z_slm], [h, h], [-1.3, z_slm], [-h, -h], color=LASER)
    zt, yt = paraxial_ray(lenses, z_slm, h, 0.0, z_med)
    zb, yb = paraxial_ray(lenses, z_slm, -h, 0.0, z_med)
    band(ax, zt, yt, zb, yb, color=LASER, alpha=0.16)
    _slm(ax, z_slm, h, reference=reference,
         text="phase SLM\n(N segments" + (",\nflat frame:\nreference)" if reference else ")"))
    label(ax, -0.75, h + 0.1, "laser", fontsize=8, va="bottom")
    lens(ax, z_l1, 1.1*h)
    lens(ax, z_l2, 0.75*h if medium != "fibre" else 0.35*h)
    label(ax, (z_l1 + z_l2)/2, -1.15, "4f relay: SLM imaged\nonto the medium", fontsize=8, va="top")
    hm = h*f2/f1                              # beam half height on the medium
    rng = np.random.default_rng(3)
    if medium == "diffuser":
        thick = 0.12
        ax.add_patch(mpatches.Rectangle((z_med, -0.75), thick, 1.5, facecolor="0.9", edgecolor="0.35",
                                        hatch="....", zorder=3))
        text = "ground glass\ndiffuser"
    elif medium == "slab":
        thick = 0.9
        ax.add_patch(mpatches.Rectangle((z_med, -0.75), thick, 1.5, facecolor="0.94", edgecolor="0.35",
                                        zorder=3))
        zz, yy = rng.uniform(z_med + 0.04, z_med + thick - 0.04, 90), rng.uniform(-0.71, 0.71, 90)
        ax.scatter(zz, yy, s=10, color="0.55", zorder=3.5, linewidths=0)
        # a multiply scattered path through the slab
        zp = np.r_[z_med, np.sort(rng.uniform(z_med, z_med + thick, 6)), z_med + thick]
        yp = np.r_[0.1, rng.uniform(-0.5, 0.5, 6), -0.15]
        ax.plot(zp, yp, color=LASER, lw=0.9, zorder=4)
        text = "thick random slab\n(WPM)"
    else:
        thick = 2.2
        zc = np.linspace(z_med, z_med + thick, 200)
        yc = 0.4*np.sin(2*np.pi*(zc - z_med)/thick*1.5)*np.sin(np.pi*(zc - z_med)/thick)**0.3
        ax.plot(zc, yc, color="0.35", lw=6, solid_capstyle="butt", zorder=3)
        ax.plot(zc, yc, color="#f3e6c8", lw=3.5, solid_capstyle="butt", zorder=3.1)
        text = "multimode fibre\n(M modes, mode mixing)"
    label(ax, z_med + thick/2, -0.95, text, fontsize=8, va="top")
    z_out = z_med + thick
    f3 = 1.5
    z_l3 = z_out + 0.9
    z_cam = z_l3 + f3
    if medium == "fibre":
        hm = 0.08                             # light leaves the small core
    # scattered light: random directions
    for _ in range(22):
        y0 = rng.uniform(-hm, hm)
        y1 = y0 + rng.uniform(-0.75, 0.75)
        ax.plot([z_out, z_l3], [y0, y1], color=LASER, lw=0.5, alpha=0.45, zorder=2)
    # shaped part: a band from the medium to the target pixel
    y_t = 0.3
    band(ax, [z_out, z_l3, z_cam], [hm, hm, y_t], [z_out, z_l3, z_cam], [-hm, -hm, y_t], color=LASER,
         alpha=0.3)
    lens(ax, z_l3, 0.8)
    label(ax, z_l3, -0.95, "lens", fontsize=8, va="top")
    detector(ax, z_cam, 0.75, text="camera:\n" + ("far field" if medium != "fibre" else "output facet"))
    ax.plot(z_cam - 0.02, y_t, "o", color=LASER, ms=5, zorder=5)
    label(ax, z_cam + 0.17, y_t - 0.08, "target", fontsize=8, ha="left", va="top")
    if feedback:
        y_fb = 1.45
        ax.plot([z_cam + 0.1, z_cam + 0.55, z_cam + 0.55, z_slm, z_slm],
                [y_t, y_t, y_fb, y_fb, h + 0.05], color="0.3", lw=1, zorder=2)
        arrow(ax, z_slm, h + 0.3, z_slm, h + 0.05, color="0.3")
        label(ax, (z_slm + z_cam)/2, y_fb + 0.08,
              "intensity in the target → SLM phases (feedback)" if not reference else
              "phase-stepped camera frames → transmission matrix", fontsize=8, va="bottom")
    _finish(ax, (-1.4, z_cam + 0.8), (-1.7 if not reference else -2.0, 1.75), title)


def _knife(ax, z, y_edge, h, from_below=True, color=METAL, zorder=6):
    """A knife blade in section, its edge at height y_edge."""

    s = -1 if from_below else 1
    ax.add_patch(_patches().Polygon([(z - 0.05, y_edge + s*h), (z + 0.05, y_edge + s*h),
                                     (z + 0.05, y_edge + s*0.08), (z, y_edge)],
                                    closed=True, facecolor=color, edgecolor="none",
                                    zorder=zorder))


def _ronchi_bars(ax, z, y, h, n=9, zorder=6):
    for yb in np.linspace(y - h, y + h, n):
        ax.add_patch(_patches().Rectangle((z - 0.025, yb - h/(2*n)), 0.05, h/n, facecolor=METAL,
                                          edgecolor="none", zorder=zorder))


def foucault_tester(ax, test="knife", title=None):
    """Sketch of the Foucault knife-edge test of a concave mirror (notebook
    76).

    A pinhole source next to the centre of curvature C of the mirror is
    imaged back onto the other side of C. A knife edge (or a wire, a slit,
    a Ronchi ruling) is moved into this image, and the eye or a camera
    behind it, focused on the mirror, sees the mirror lit by the light that
    passes. A ray from a zone that bends the wavefront (red) misses the focus
    and is blocked or passed depending on its side: the zone turns dark or
    bright.

    Parameters
    ----------
    ax : matplotlib Axes
    test : {"knife", "wire", "ronchi"}, optional
        The element in the focus. Defaults to "knife".
    title : str, optional
    """

    R = 5.6
    z_c = 0.0
    z_m = z_c + R
    h = 1.4
    s = 0.12
    sphere(ax, z_m, 0.0, -R, h, color="0.3", lw=3, text="concave mirror\nunder test",
           text_y=-h - 0.15)
    yy = np.linspace(-h, h, 9)
    z_surf = z_m - R + np.sqrt(R**2 - yy**2)
    for yi, zi in zip(yy, z_surf):
        ax.plot([z_c, zi], [-s, yi], color=ILLUMINATION, lw=0.7, alpha=0.7, zorder=2)
        ax.plot([zi, z_c, z_c - 1.2], [yi, s, s - (yi - s)/(zi - z_c)*1.2], color=IMAGING,
                lw=0.7, alpha=0.7, zorder=2)
    # a zone with a slope error: its reflected ray misses the focus
    yz = 0.9
    zz = z_m - R + np.sqrt(R**2 - yz**2)
    y_miss = s + 0.2
    u_err = (y_miss - yz)/(z_c - zz)
    ax.plot([zz, z_c], [yz, y_miss], color=STRAY, lw=1.3, zorder=3)
    if test == "knife":
        _knife(ax, z_c, s, 0.7, from_below=False)
        label(ax, z_c + 0.12, s + 0.75, "knife edge", ha="left", fontsize=8)
    elif test == "wire":
        ax.plot([z_c], [s], "o", color=METAL, ms=5, zorder=6)
        ax.plot([z_c, z_c], [s, s + 0.7], color=METAL, lw=1, zorder=6)
        label(ax, z_c + 0.12, s + 0.75, "wire", ha="left", fontsize=8)
    else:
        _ronchi_bars(ax, z_c, s + 0.25, 0.45)
        label(ax, z_c + 0.12, s + 0.75, "Ronchi ruling", ha="left", fontsize=8)
    if test == "ronchi":
        ax.plot([z_c, z_c - 1.2], [y_miss, y_miss + u_err*(-1.2)], color=STRAY, lw=1.3,
                zorder=3)
    point_source(ax, z_c, -s, size=50)
    ax.add_patch(_patches().Rectangle((z_c + 0.03, -0.75), 0.04, 0.75 - s - 0.03,
                                      facecolor=METAL, edgecolor="none", zorder=5))
    label(ax, z_c + 0.12, -0.8, "pinhole source", ha="left", va="top", fontsize=8)
    # eye / camera focused on the mirror
    lens(ax, z_c - 1.2, 0.4, text="camera lens\n(focused on\nthe mirror)", text_y=-0.5)
    detector(ax, z_c - 1.9, 0.3)
    arrow(ax, z_c, -h - 0.35, z_m, -h - 0.35, style="<->", text="R",
          text_pos=((z_c + z_m)/2, -h - 0.42), va="top")
    _finish(ax, (z_c - 2.3, z_m + 0.5), (-h - 0.7, h + 0.25), title)


def _candle_plume(ax, z, y_base, height, width=0.3, color="#c0392b"):
    mpatches = _patches()
    ax.add_patch(mpatches.Rectangle((z - 0.1, y_base - 0.6), 0.2, 0.6, facecolor="#f3ead3",
                                    edgecolor="0.4", lw=0.8, zorder=5))
    ax.add_patch(mpatches.Ellipse((z, y_base + 0.12), 0.1, 0.26, facecolor="#f5a623",
                                  edgecolor="none", zorder=6))
    t = np.linspace(0, 1, 60)
    for s in (-1, 1):
        ax.plot(z + s*width*(0.25 + 0.75*np.sqrt(t)) + 0.08*np.sin(9*t)*t, y_base + 0.25 + t*height,
                color=color, lw=0.8, alpha=0.7, zorder=5)


def schlieren_system(ax, method="schlieren", title=None):
    """Sketch of a lens type Toepler schlieren system and of a direct
    shadowgraph (notebook 76).

    A lamp and condenser light a slit source in the focus of the first lens,
    which makes a parallel beam through the test section (a candle plume).
    The second lens forms an image of the slit on the knife edge, and the
    camera images the test section. A ray deflected by the angle eps in the
    plume (red) arrives f2*eps beside the slit image and passes or is
    blocked. In the shadowgraph the parallel beam falls directly onto a
    screen at the distance L, where the deflected rays have moved by L*eps.

    Parameters
    ----------
    ax : matplotlib Axes
    method : {"schlieren", "shadowgraph"}, optional
        Defaults to "schlieren".
    title : str, optional
    """

    f1, f2 = 2.0, 2.6
    z_s = 0.0
    z_l1 = z_s + f1
    z_t = z_l1 + 1.6
    z_l2 = z_t + 1.6
    z_k = z_l2 + f2
    h = 0.75
    point_source(ax, z_s - 0.9, 0.0, size=70, text="lamp", text_dy=-0.2)
    lens(ax, z_s - 0.5, 0.25)
    stop(ax, z_s, 0.45, openings=((0, 0.04),), width=0.04, text="slit\nsource", text_y=-0.5)
    band(ax, [z_s, z_l1, z_t + (1.6 if method == "schlieren" else 2.6)], [0.0, h, h],
         [z_s, z_l1, z_t + (1.6 if method == "schlieren" else 2.6)], [0.0, -h, -h],
         color=ILLUMINATION, alpha=0.18)
    lens(ax, z_l1, 1.1*h, text="lens 1 (f₁)")
    _candle_plume(ax, z_t, -h - 0.15, 2*h + 0.3)
    label(ax, z_t, -h - 0.85, "test section:\ncandle plume", va="top", fontsize=8)
    eps = 0.09
    yr = 0.25
    if method == "schlieren":
        z_cam = z_k + 1.4
        band(ax, [z_l2, z_k, z_cam], [h, 0.0, -h*1.4/f2], [z_l2, z_k, z_cam],
             [-h, 0.0, h*1.4/f2], color=ILLUMINATION, alpha=0.18)
        lens(ax, z_l2, 1.1*h, text="lens 2 (f₂)")
        y_l2 = yr + eps*(z_l2 - z_t)
        u2 = eps - y_l2/f2
        ax.plot([z_t, z_l2, z_cam], [yr, y_l2, y_l2 + u2*(z_cam - z_l2)], color=STRAY, lw=1.3,
                zorder=4)
        ax.plot([z_l1 - 0.6, z_t], [yr, yr], color=STRAY, lw=1.3, zorder=4)
        _knife(ax, z_k, 0.0, 0.6)
        label(ax, z_k, -0.7, "knife edge in\nthe slit image", va="top", fontsize=8)
        arrow(ax, z_k + 0.3, eps*f2 + 0.25, z_k + 0.03, eps*f2 + 0.02, color=STRAY,
              text="f₂ ε", text_pos=(z_k + 0.33, eps*f2 + 0.27), ha="left", va="bottom")
        lens(ax, z_cam, 0.6, text="camera lens", text_y=0.95)
        detector(ax, z_cam + 1.0, 0.5, text="camera\n(focused on the\ntest section)",
                 text_y=-0.6)
        _finish(ax, (z_s - 1.2, z_cam + 1.6), (-1.8, 1.2), title)
    else:
        z_scr = z_t + 2.6
        ax.plot([z_l1 - 0.6, z_t, z_scr], [yr, yr, yr + eps*(z_scr - z_t)], color=STRAY, lw=1.3,
                zorder=4)
        detector(ax, z_scr, 1.1*h, text="screen or camera", text_y=-0.95)
        arrow(ax, z_t, h + 0.3, z_scr, h + 0.3, style="<->", text="L",
              text_pos=((z_t + z_scr)/2, h + 0.35), va="bottom")
        _finish(ax, (z_s - 1.2, z_scr + 0.6), (-1.8, 1.2), title)


def _fold_down_slash(z_m, z_unf, y_unf):
    # a "/" mirror at (z_m, 0): light travelling along -z is sent along -y;
    # unfolded (z_m + dz, dy), dz < 0, lands at (z_m + dy, dz)
    return z_m + np.asarray(y_unf), np.asarray(z_unf) - z_m


def _fold_up_backslash(z_m, z_unf, y_unf):
    # a "\" mirror at (z_m, 0): light travelling along -z is sent along +y;
    # unfolded (z_m + dz, dy), dz < 0, lands at (z_m - dy, -dz)
    return z_m - np.asarray(y_unf), z_m - np.asarray(z_unf)


def sim_microscope(ax, title=None):
    """Sketch of a structured illumination microscope with a grating (notebook 78).

    A phase grating in a plane conjugate to the specimen diffracts the
    collimated laser beam. A mask in the following pupil plane blocks the
    zeroth order and passes the +-1st orders, which are focused near the
    edge of the objective pupil and leave the objective as two plane waves
    that cross in the specimen: their interference is the sinusoidal
    excitation pattern (inset). The fluorescence is reflected by a dichroic
    mirror onto the camera. Rotating and shifting the grating gives the
    angles and phases of the pattern.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    u = 0.5                                  # diffraction angle of the +-1st orders
    g = 0.2                                  # half height of the beam on the grating
    z_g = 0.0
    z_l1, f1 = 1.0, 1.0                      # grating -> order mask (pupil plane)
    z_mask = 2.0
    z_l2, f2 = 2.7, 0.7                      # mask -> intermediate image at 3.4
    z_l3, f3 = 4.3, 0.9                      # image -> objective pupil at 5.2
    z_obj, f_o = 5.8, 0.6
    z_dic = 4.85
    z_foc = z_obj + f_o
    lenses = [(z_l1, f1), (z_l2, f2), (z_l3, f3), (z_obj, f_o)]
    # laser, grating
    laser(ax, -0.8, 0.0, length=0.9, height=0.3, text="laser")
    band(ax, [-0.8, z_g], [g, g], [-0.8, z_g], [-g, -g], color=EXCITATION, alpha=0.18)
    for k in range(9):
        y0 = -0.42 + 0.84*k/8
        ax.add_patch(mpatches.Rectangle((z_g - 0.03, y0), 0.06 if k % 2 else 0.1, 0.84/9,
                                        facecolor=GLASS, edgecolor=EDGE, lw=0.6, zorder=3))
    label(ax, z_g, -0.55, "phase grating\n(rotate, shift)", fontsize=8, va="top")
    # the +-1st orders as collimated beams
    for sgn in (-1, 1):
        zt, yt = paraxial_ray(lenses, z_g, g, sgn*u, z_foc + 0.3)
        zb, yb = paraxial_ray(lenses, z_g, -g, sgn*u, z_foc + 0.3)
        band(ax, zt, yt, zb, yb, color=EXCITATION, alpha=0.16, lw=0.6)
    # the blocked zeroth order
    zt, yt = paraxial_ray(lenses[:1], z_g, g, 0.0, z_mask)
    zb, yb = paraxial_ray(lenses[:1], z_g, -g, 0.0, z_mask)
    band(ax, zt, yt, zb, yb, color="0.6", alpha=0.2, lw=0.5, zorder=0.5)
    lens(ax, z_l1, 0.75)
    y_o = f1*u
    stop(ax, z_mask, 0.8, openings=((y_o - 0.07, y_o + 0.07),), width=0.05,
         text="order mask:\n0 blocked, ±1 pass", text_y=-0.9)
    lens(ax, z_l2, 0.75)
    plane(ax, z_l2 + f2, 0.6)
    label(ax, z_l2 + f2, 0.68, "image\nplane", fontsize=7, va="bottom")
    lens(ax, z_l3, 0.75)
    plane(ax, z_obj - f_o, 0.75)
    label(ax, z_obj - f_o, -0.82, "pupil", fontsize=7, va="top")
    mirror(ax, z_dic, 0.0, 1.1, -np.pi/4, dichroic=True)
    label(ax, z_dic + 0.1, -0.85, "dichroic", fontsize=8, va="top")
    lens(ax, z_obj, 0.85, text="objective", text_y=-0.95)
    ax.add_patch(mpatches.Rectangle((z_foc - 0.12, -0.6), 0.24, 1.2, facecolor="#e9f5e9",
                                    edgecolor="#8fbf8f", zorder=1))
    label(ax, z_foc, -0.68, "specimen", fontsize=8, va="top")
    # fluorescence from a point in the specimen, back to the camera above
    z_t, f_t = z_dic - 1.2, 0.9               # camera tube lens (unfolded position)
    z_cam = z_t - f_t
    for v in (-0.4, -0.2, 0.2, 0.4):
        zz, yy = paraxial_ray([(z_obj, f_o), (z_t, f_t)], z_foc, 0.0, v, z_cam)
        sel = zz <= z_dic
        z_cross = z_dic
        y_cross = np.interp(z_cross, zz[::-1], yy[::-1])
        ax.plot(np.r_[zz[~sel], z_cross], np.r_[yy[~sel], y_cross], color=EMISSION, lw=0.9,
                zorder=2)
        zf, yf = _fold_up_backslash(z_dic, np.r_[z_cross, zz[sel]], np.r_[y_cross, yy[sel]])
        ax.plot(zf, yf, color=EMISSION, lw=0.9, zorder=2)
    zc, yc = _fold_up_backslash(z_dic, z_t, 0.0)
    rotated_lens(ax, zc, yc, 0.4, rotate=90, text="tube lens", fontsize=8)
    zc, yc = _fold_up_backslash(z_dic, z_cam, 0.0)
    rotated_detector(ax, zc, yc, 0.4, rotate=-90, text="camera", fontsize=8)
    # inset: the fringes in the specimen
    s = np.linspace(-1, 1, 64)
    S, T = np.meshgrid(s, s)
    _profile_image(ax, z_foc + 0.85, 0.75, 0.8, 1 + np.cos(2*np.pi*3*(0.85*T + 0.5*S)))
    label(ax, z_foc + 0.85, 0.27, "excitation\nfringes", fontsize=8, va="top")
    ax.plot([], [], color=EXCITATION, label="±1st orders: two plane waves in the specimen")
    ax.plot([], [], color=EMISSION, label="fluorescence")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _finish(ax, (-1.9, z_foc + 1.4), (-1.3, 2.7), title)


def sted_microscope(ax, images=None, title=None):
    """Sketch of a STED microscope (notebook 78).

    An excitation beam (blue) and a depletion beam (red) are combined by a
    dichroic mirror and focused by the same objective. The depletion beam
    passes a spiral (vortex) phase plate and a quarter wave plate, so its
    focus is a doughnut with a zero in the centre, and switches off the
    fluorophores everywhere but there. The fluorescence (green) is
    separated by a second dichroic mirror and detected through a pinhole.
    Insets show the excitation spot, the doughnut and the effective spot.

    Parameters
    ----------
    ax : matplotlib Axes
    images : sequence of three 2d arrays, optional
        Excitation, depletion and effective PSF for the insets. Defaults to
        simple analytic shapes.
    title : str, optional
    """

    mpatches = _patches()
    w = 0.32
    z_las = -0.5
    z_spp, z_qwp = 0.3, 1.5
    z_d1, z_d2 = 2.4, 3.6
    z_obj, f_o = 4.8, 0.9
    z_foc = z_obj + f_o
    laser(ax, z_las, 0.0, length=1.0, height=0.3, text="STED", color="#a03020")
    for top in (True, False):
        y_edge = w if top else -w
        zz, yy = paraxial_ray([(z_obj, f_o)], z_las, y_edge, 0.0, z_foc)
        if top:
            zt, yt = zz, yy
        else:
            zb, yb = zz, yy
    band(ax, zt, yt, zb, yb, color=STRAY, alpha=0.15)
    _spiral_plate(ax, z_spp, 0.45, text="vortex plate\nexp(iφ)", text_y=-0.55)
    ax.add_patch(mpatches.Rectangle((z_qwp - 0.04, -0.45), 0.08, 0.9, facecolor="#e8e0f5",
                                    edgecolor="#6a5a8a", lw=0.8, zorder=3))
    label(ax, z_qwp, -0.55, "λ/4: circular,\nspin ∥ vortex", fontsize=8, va="top")
    # excitation from above, reflected along +z by the first dichroic ("\")
    w_e = 0.24
    y_ex = 1.3
    laser(ax, z_d1, y_ex, length=0.8, height=0.3, rotate=-90, text="exc.", fontsize=8,
          color="#1a5a8a")
    band(ax, [z_d1 - w_e, z_d1 - w_e], [y_ex, -w_e], [z_d1 + w_e, z_d1 + w_e],
         [y_ex, w_e], color=EXCITATION, alpha=0.18)
    zz_t, yy_t = paraxial_ray([(z_obj, f_o)], z_d1, w_e, 0.0, z_foc)
    zz_b, yy_b = paraxial_ray([(z_obj, f_o)], z_d1, -w_e, 0.0, z_foc)
    band(ax, zz_t, yy_t, zz_b, yy_b, color=EXCITATION, alpha=0.18)
    mirror(ax, z_d1, 0.0, 1.0, -np.pi/4, dichroic=True)
    label(ax, z_d1 - 0.1, 0.62, "dichroic", fontsize=8, ha="right", va="bottom")
    # fluorescence: back through the objective, reflected down by the second dichroic ("/")
    z_t, f_t = z_d2 - 1.0, 0.9
    z_pin = z_t - f_t
    for v in (-0.3, -0.16, 0.16, 0.3):
        zz, yy = paraxial_ray([(z_obj, f_o), (z_t, f_t)], z_foc, 0.0, v, z_pin - 0.25)
        y_cross = np.interp(z_d2, zz[::-1], yy[::-1])
        keep = zz > z_d2
        ax.plot(np.r_[zz[keep], z_d2], np.r_[yy[keep], y_cross], color=EMISSION, lw=0.9, zorder=2)
        zf, yf = _fold_down_slash(z_d2, np.r_[z_d2, zz[~keep]], np.r_[y_cross, yy[~keep]])
        ax.plot(zf, yf, color=EMISSION, lw=0.9, zorder=2)
    mirror(ax, z_d2, 0.0, 1.0, np.pi/4, dichroic=True)
    label(ax, z_d2 + 0.12, 0.62, "dichroic", fontsize=8, ha="left", va="bottom")
    zc, yc = _fold_down_slash(z_d2, z_t, 0.0)
    rotated_lens(ax, zc, yc, 0.35, rotate=90)
    zc, yc = _fold_down_slash(z_d2, z_pin, 0.0)
    ax.plot([zc - 0.35, zc - 0.04], [yc, yc], color=METAL, lw=3, solid_capstyle="butt", zorder=4)
    ax.plot([zc + 0.04, zc + 0.35], [yc, yc], color=METAL, lw=3, solid_capstyle="butt", zorder=4)
    label(ax, zc + 0.42, yc, "pinhole", fontsize=8, ha="left")
    rotated_detector(ax, zc, yc - 0.25, 0.3, rotate=-90)
    label(ax, zc + 0.42, yc - 0.3, "detector", fontsize=8, ha="left")
    lens(ax, z_obj, 0.85, text="objective", text_y=-0.95)
    ax.add_patch(mpatches.Rectangle((z_foc - 0.1, -0.55), 0.2, 1.1, facecolor="#e9f5e9",
                                    edgecolor="#8fbf8f", zorder=1))
    label(ax, z_foc, -0.62, "specimen\n(scanned)", fontsize=8, va="top")
    # insets
    if images is None:
        s = np.linspace(-1.5, 1.5, 81)
        S, T = np.meshgrid(s, s)
        R2 = S**2 + T**2
        exc = np.exp(-R2/0.5)
        donut = R2*np.exp(-R2/0.5)
        images = (exc, donut, exc*np.exp(-np.log(2)*6*donut/donut.max()))
    names = ("excitation", "doughnut", "effective spot")
    size = 0.75
    y_i = 1.75
    for i, (img, name) in enumerate(zip(images, names)):
        zi = z_d2 + 0.1 + i*1.0
        _profile_image(ax, zi, y_i, size, np.asarray(img, dtype=float))
        label(ax, zi, y_i + size/2 + 0.05, name, fontsize=8, va="bottom")
    label(ax, z_d2 + 0.6, y_i, "×", fontsize=12)
    label(ax, z_d2 + 1.6, y_i, "→", fontsize=12)
    label(ax, z_d2 + 1.1, y_i - size/2 - 0.08, "survival exp(−ln2·I/I_s)", fontsize=7, va="top")
    ax.plot([], [], color=EXCITATION, label="excitation")
    ax.plot([], [], color=STRAY, label="STED (depletion) beam")
    ax.plot([], [], color=EMISSION, label="fluorescence")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _finish(ax, (-1.7, z_foc + 0.8), (-2.2, 2.4), title)


ROAD = "#5b5b5b"
HOT_AIR = "#f4c99a"


def grin_rod_imaging(ax, title=None):
    """Sketch of a GRIN rod lens: an object in contact with the front face,
    rays that run on sine curves about the axis, an inverted image at half a
    pitch and an upright one after a full pitch (notebook 77).

    The index falls from the axis outwards (shaded); a point on the axis is
    focused again every half pitch, a parallel bundle at a quarter pitch.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    P, h = 8.0, 1.0
    zz = np.linspace(0, P, 400)
    # the index profile as horizontal bands, darker on the axis
    for k, yy in enumerate(np.linspace(h, 0.0, 9)[:-1]):
        ax.add_patch(mpatches.Rectangle((0, -yy), P, 2*yy, facecolor=GLASS, alpha=0.35,
                                        edgecolor="none", zorder=0.5 + 0.01*k))
    ax.add_patch(mpatches.Rectangle((0, -h), P, 2*h, facecolor="none", edgecolor=EDGE, lw=1.2,
                                    zorder=3))
    ax.plot([-0.9, P + 0.9], [0, 0], color="0.55", lw=0.7, ls="-.", zorder=1)
    # rays from an axial point (blue) and an off-axis point (orange)
    for slope in (-0.55, -0.3, 0.3, 0.55):
        ax.plot(zz, slope/(2*np.pi/P)*np.sin(2*np.pi*zz/P), color=IMAGING, lw=1.1, zorder=2)
    y0 = 0.45
    for slope in (-0.5, 0.0, 0.5):
        ax.plot(zz, y0*np.cos(2*np.pi*zz/P) + slope/(2*np.pi/P)*np.sin(2*np.pi*zz/P),
                color=ILLUMINATION, lw=1.1, zorder=2)
    ax.annotate("", xy=(0, y0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6), zorder=4)
    ax.annotate("", xy=(P/2, -y0), xytext=(P/2, 0),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6), zorder=4)
    ax.annotate("", xy=(P, y0), xytext=(P, 0),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6), zorder=4)
    for z, txt in ((0, "object"), (P/4, "P/4"), (P/2, "P/2: inverted"), (3*P/4, "3P/4"),
                   (P, "P: upright")):
        ax.plot([z, z], [-h - 0.1, h + 0.1], color="0.4", lw=0.8, ls=":", zorder=1)
        label(ax, z, -h - 0.3, txt, fontsize=8, va="top")
    label(ax, P/2, h + 0.35, r"$n(\rho)$ highest on the axis", fontsize=8, va="bottom",
          color=EDGE)
    # index profile to the right
    rr = np.linspace(-h, h, 80)
    ax.plot(P + 1.3 + 0.7*(1 - rr**2), rr, color=EDGE, lw=1.2)
    ax.plot([P + 1.3, P + 1.3], [-h, h], color="0.6", lw=0.7)
    label(ax, P + 1.65, -h - 0.3, r"$n$", fontsize=9, va="top")
    _finish(ax, (-1.0, P + 2.4), (-h - 0.9, h + 0.9), title)


def road_mirage(ax, title=None):
    """Sketch of the inferior mirage over a hot road (notebook 77).

    The air just above the road is hotter and has a lower index. Rays from
    the top of a distant car reach the eye on a direct, nearly straight
    path, and on a path that dips into the hot layer and is bent back
    upwards; the second arrives from below the horizon, so the eye sees an
    inverted image of the car (and of the sky) apparently lying on the
    road, like a reflection in water. Vertical scale strongly exaggerated.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    L, y_eye = 10.0, 1.6
    ax.add_patch(mpatches.Rectangle((-0.8, -1.0), L + 1.6, 1.0, facecolor=ROAD, edgecolor="none",
                                    zorder=1))
    for k, a in enumerate(np.linspace(0.45, 0.08, 6)):
        ax.add_patch(mpatches.Rectangle((-0.8, 0.06*k), L + 1.6, 0.06, facecolor=HOT_AIR, alpha=a,
                                        edgecolor="none", zorder=0.8))
    for z in np.arange(-0.4, L + 0.8, 1.2):
        ax.plot([z, z + 0.5], [-0.12, -0.12], color="w", lw=2, zorder=1.5)
    label(ax, 0.3*L, 0.55, "hot air: lower index", fontsize=8, color="#b5651d")
    eye = _person(ax, 0.0, 0.0, height=y_eye/0.86*0.95)
    # the car at the far end
    zc = L
    ax.add_patch(mpatches.FancyBboxPatch((zc - 0.5, 0.12), 1.0, 0.45, boxstyle="round,pad=0.02",
                                         facecolor="#c0392b", edgecolor="0.2", lw=0.8, zorder=4))
    ax.add_patch(mpatches.Polygon([(zc - 0.3, 0.57), (zc - 0.15, 0.85), (zc + 0.25, 0.85),
                                   (zc + 0.4, 0.57)], facecolor="#c0392b", edgecolor="0.2",
                                  lw=0.8, zorder=4))
    for dz in (-0.28, 0.28):
        ax.add_patch(mpatches.Circle((zc + dz, 0.12), 0.11, facecolor="0.15", zorder=5))
    top = np.array([zc + 0.05, 0.85])
    # direct ray
    ax.plot([eye[0], top[0]], [eye[1], top[1]], color=IMAGING, lw=1.3, zorder=3)
    # bent ray: a hyperbola with straight asymptotes, turning in the hot layer
    s1 = (-top[1] - eye[1])/(zc - eye[0])           # towards the mirror image of the car top
    z1 = eye[0] + (0.0 - eye[1])/s1
    s2 = top[1]/(top[0] - z1)
    zz = np.linspace(eye[0], top[0], 300)
    yy = 0.5*(s1 + s2)*(zz - z1) + 0.5*(s2 - s1)*np.sqrt((zz - z1)**2 + 0.5**2)
    ax.plot(zz, yy, color=LASER, lw=1.3, zorder=3)
    z_t, y_t = zz[np.argmin(yy)], yy.min()
    # the apparent direction of the bent ray and the inverted image (the car mirrored)
    ax.plot([eye[0], top[0]], [eye[1], -top[1]], color=LASER, lw=0.9, ls="--", zorder=2)
    ghost = dict(facecolor="#e8a09a", edgecolor="none", alpha=0.85, zorder=2)
    ax.add_patch(mpatches.Rectangle((zc - 0.5, -0.57), 1.0, 0.45, **ghost))
    ax.add_patch(mpatches.Polygon([(zc - 0.3, -0.57), (zc - 0.15, -0.85), (zc + 0.25, -0.85),
                                   (zc + 0.4, -0.57)], **ghost))
    z_img, y_img = zc, -0.85
    label(ax, z_img - 0.75, y_img + 0.45, "inverted image\n(seen on the road)", fontsize=8,
          ha="right", color="w")
    label(ax, 0.45*L, 1.45, "direct ray", fontsize=8, color=IMAGING)
    label(ax, z_t + 0.3, y_t + 0.3, "turning point", fontsize=8, color=LASER, ha="left")
    ax.plot([eye[0], L + 0.8], [eye[1], eye[1]], color="0.6", lw=0.7, ls=":", zorder=1)
    label(ax, L + 0.8, eye[1] + 0.12, "horizon", fontsize=8, ha="right", va="bottom", color="0.4")
    _finish(ax, (-0.8, L + 0.8), (-1.0, 2.2), title)


def lensless_camera(ax, kind="pinhole", title=None):
    """Sketch of a lensless camera (notebook 79).

    - "pinhole": an object, a pinhole of diameter d and a screen at the
      distance v. Each object point casts a blurred spot (the shadow of the
      hole widened by diffraction) and the image is inverted.
    - "coded": two distant sources and a mask with many holes. Every source
      casts a shifted shadow of the whole mask; the detector records their
      sum, from which a correlation with the decoding array recovers the
      sources.
    - "diffuser": a point source, a smooth random phase plate (the diffuser)
      and a bare sensor a few millimetres behind it. The bumps of the
      diffuser act as weak random lenslets whose rays cross on the sensor in
      a network of bright caustics, the PSF.

    Parameters
    ----------
    ax : matplotlib Axes
    kind : {"pinhole", "coded", "diffuser"}, optional
    title : str, optional
    """

    if kind not in ("pinhole", "coded", "diffuser"):
        raise ValueError("kind must be 'pinhole', 'coded' or 'diffuser'.")
    mpatches = _patches()
    if kind == "pinhole":
        z_obj, z_hole, z_scr = -3.0, 0.0, 2.0
        r = 0.06
        ax.annotate("", xy=(z_obj, 0.85), xytext=(z_obj, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=ILLUMINATION, lw=2.5, mutation_scale=14))
        label(ax, z_obj, -0.15, "object", va="top")
        stop(ax, z_hole, 1.1, openings=((0.0, r),), width=0.06)
        label(ax, z_hole, -1.25, "pinhole, diameter d", va="top")
        for y0, col in ((0.85, ILLUMINATION), (0.0, IMAGING)):
            # geometric cone from the object point through the hole
            y_scr = [y0 + (yh - y0)*(z_scr - z_obj)/(z_hole - z_obj) for yh in (r, -r)]
            band(ax, [z_obj, z_hole, z_scr], [y0, r, y_scr[0]], [z_obj, z_hole, z_scr],
                 [y0, -r, y_scr[1]], color=col, alpha=0.25, lw=0.6)
            yc = np.mean(y_scr)
            yy = np.linspace(-0.22, 0.22, 80)
            spot = np.exp(-(yy/0.08)**2)
            ax.plot(z_scr + 0.08 + 0.3*spot, yc + yy, color=col, lw=1.4, zorder=5)
        ax.plot([z_scr, z_scr], [-1.1, 1.1], color="0.35", lw=3, zorder=4)
        label(ax, z_scr, -1.25, "screen", va="top")
        arrow(ax, z_hole, 0.95, z_scr, 0.95, style="<->", text="v", text_pos=((z_hole + z_scr)/2, 1.07),
              fontsize=9)
        label(ax, z_scr + 0.75, 0.0, "blurred,\ninverted\nimage\n(blur ≈ d and\n2.44 λv/d)",
              fontsize=8)
        _finish(ax, (z_obj - 0.5, z_scr + 1.35), (-1.6, 1.3), title)
    elif kind == "coded":
        z_src, z_mask, z_det = -3.2, 0.0, 2.2
        holes = [(-0.95, -0.8), (-0.65, -0.5), (-0.4, -0.25), (0.05, 0.25), (0.45, 0.55), (0.75, 0.95)]
        sources = ((0.7, STAR), (-0.45, STAR2))
        for ys, col in sources:
            point_source(ax, z_src, ys, color=col, size=90)
            for a, b in holes:
                ya = ys + (a - ys)*(z_det - z_src)/(z_mask - z_src)
                yb = ys + (b - ys)*(z_det - z_src)/(z_mask - z_src)
                band(ax, [z_src, z_mask, z_det], [ys, b, yb], [z_src, z_mask, z_det], [ys, a, ya],
                     color=col, alpha=0.13, edges=False)
                ax.add_patch(mpatches.Rectangle((z_det + 0.07 + (0.0 if col == STAR else 0.12), ya),
                                                0.1, yb - ya, facecolor=col, alpha=0.8, zorder=5,
                                                edgecolor="none"))
        # mask: opaque between the holes
        edges = [-1.15] + [e for h in holes for e in h] + [1.15]
        for a, b in zip(edges[::2], edges[1::2]):
            ax.add_patch(mpatches.Rectangle((z_mask - 0.03, a), 0.06, b - a, facecolor=METAL,
                                            edgecolor="none", zorder=4))
        label(ax, z_src, -0.75, "distant\nsources", va="top")
        label(ax, z_mask - 0.2, -1.3, "coded mask\n(MURA, ~50 % open)", va="top")
        ax.plot([z_det, z_det], [-1.15, 1.15], color="0.35", lw=3, zorder=4)
        label(ax, z_det + 0.45, -1.3, "detector: sum of\nshifted shadows", va="top")
        label(ax, z_det + 1.05, 0.0, "decode:\ncorrelate\nwith G", fontsize=8)
        _finish(ax, (z_src - 0.5, z_det + 1.5), (-1.75, 1.3), title)
    else:
        z_src, z_dif, z_sen = -3.0, 0.0, 1.8
        y = np.linspace(-0.9, 0.9, 400)
        bumps = 0.05*np.sin(7.1*y + 0.4) + 0.035*np.sin(13.3*y + 1.9) + 0.02*np.sin(23.0*y + 0.3)
        ax.fill_betweenx(y, z_dif - 0.08, z_dif + 0.04 + bumps, color=GLASS, zorder=3)
        ax.plot(z_dif + 0.04 + bumps, y, color=EDGE, lw=1.0, zorder=3.5)
        point_source(ax, z_src, 0.25, color=ILLUMINATION, size=90)
        slope = np.gradient(bumps, y)
        hits = []
        for yr in np.linspace(-0.85, 0.85, 41):
            ax.plot([z_src, z_dif], [0.25, yr], color=ILLUMINATION, lw=0.5, alpha=0.6, zorder=2)
            u = (yr - 0.25)/(z_dif - z_src) - 0.3*np.interp(yr, y, slope)
            y_end = yr + u*(z_sen - z_dif)
            hits.append(y_end)
            ax.plot([z_dif, z_sen], [yr, y_end], color=STRAY, lw=0.6, alpha=0.8, zorder=2)
        ax.plot([z_sen, z_sen], [-1.1, 1.1], color="0.35", lw=3, zorder=4)
        # intensity on the sensor: density of ray hits, smoothed
        ys = np.linspace(-1.1, 1.1, 300)
        dens = sum(np.exp(-((ys - h)/0.025)**2) for h in hits)
        ax.plot(z_sen + 0.08 + 0.12*dens/dens.max()*3, ys, color="k", lw=1.0, zorder=5)
        label(ax, z_src, 0.0, "point source\nat distance u", va="top")
        label(ax, z_dif - 0.35, -1.05, "smooth random\nphase plate", va="top")
        label(ax, z_sen + 0.5, -1.2, "bare sensor\n(z behind)", va="top")
        label(ax, z_sen + 0.95, 0.75, "caustics:\nthe PSF", fontsize=8)
        _finish(ax, (z_src - 0.6, z_sen + 1.4), (-1.6, 1.2), title)


def oct_setup(ax, kind="spectral", title=None):
    """Sketch of an optical coherence tomograph (notebook 82).

    A superluminescent diode (SLD) feeds a Michelson interferometer: the
    reference arm ends in a mirror, the sample arm focuses into a layered
    sample. The returning beams are recombined on the detection side.

    - "time": the reference mirror is scanned over the depth range and a
      single photodiode records fringes wherever the reference path matches
      the path to a reflector within the coherence length.
    - "spectral": the reference mirror is fixed, and a spectrometer (grating
      and line camera) records the spectrum. Each reflector modulates it with
      fringes whose frequency in k is proportional to its depth.

    Parameters
    ----------
    ax : matplotlib Axes
    kind : {"spectral", "time"}, optional
    title : str, optional
    """

    if kind not in ("spectral", "time"):
        raise ValueError("kind must be 'spectral' or 'time'.")
    mpatches = _patches()
    w = 0.25
    z_bs = 2.2
    laser(ax, 0.2, 0.0, length=1.0, height=0.45, text="SLD", color="0.3")
    band(ax, [0.2, z_bs], [w, w], [0.2, z_bs], [-w, -w], color=ILLUMINATION)
    # reference arm (up)
    y_ref = 2.0
    band(ax, [z_bs - w, z_bs - w], [0.0, y_ref], [z_bs + w, z_bs + w], [0.0, y_ref], color=REFERENCE,
         alpha=0.2)
    mirror(ax, z_bs, y_ref + 0.05, 0.8, np.pi/2)
    if kind == "time":
        arrow(ax, z_bs + 0.6, y_ref - 0.25, z_bs + 0.6, y_ref + 0.35, style="<|-|>", color="0.3")
        label(ax, z_bs + 0.75, y_ref + 0.05, "scanned\n$z_R$", ha="left", fontsize=8)
    else:
        label(ax, z_bs + 0.55, y_ref + 0.05, "fixed\n$z_R$", ha="left", fontsize=8)
    label(ax, z_bs, y_ref + 0.35, "reference mirror", va="bottom", fontsize=9)
    # sample arm (right): lens focusing into a layered sample
    z_l, z_s = 3.6, 4.6
    band(ax, [z_bs, z_l, z_s], [w, w, 0.03], [z_bs, z_l, z_s], [-w, -w, -0.03], color=ILLUMINATION)
    lens(ax, z_l, 0.4, text=None)
    layers = (("#f2e6d8", 0.18), ("#e6cfb8", 0.22), ("#d9b99b", 0.14), ("#cfa98a", 0.35))
    z0 = z_s - 0.05
    for col, d in layers:
        ax.add_patch(mpatches.Rectangle((z0, -0.75), d, 1.5, facecolor=col, edgecolor="0.5",
                                        lw=0.6, zorder=3))
        z0 += d
    label(ax, z_s + 0.4, -0.9, "layered sample", va="top")
    arrow(ax, z_s - 0.05, 0.95, z0, 0.95, style="-|>", color="0.3")
    label(ax, (z_s + z0)/2, 1.05, "depth z", va="bottom", fontsize=8)
    beam_splitter(ax, z_bs, 0.0, 0.8, "/", text=None)
    label(ax, z_bs - 0.55, 0.45, "beam\nsplitter", ha="right", fontsize=8)
    # detection (down)
    if kind == "time":
        band(ax, [z_bs - w, z_bs - w], [0.0, -1.5], [z_bs + w, z_bs + w], [0.0, -1.5], color="#8e5ba8")
        rotated_detector(ax, z_bs, -1.6, 0.3)
        label(ax, z_bs - 0.45, -1.6, "photodiode", ha="right", fontsize=8)
        d = np.linspace(-1.0, 1.0, 600)
        env = sum(a*np.exp(-((d - c)/0.06)**2) for a, c in ((1.0, -0.6), (0.45, -0.1), (0.7, 0.5)))
        ax.plot(z_bs + 1.6 + 0.9*d, -1.6 + 0.3*env*np.cos(2*np.pi*d/0.035), color="#8e5ba8", lw=0.6)
        label(ax, z_bs + 1.6, -2.0, "signal vs. $z_R$: fringe packets\nat the reflector depths",
              va="top", fontsize=8)
    else:
        band(ax, [z_bs - w, z_bs - w], [0.0, -1.1], [z_bs + w, z_bs + w], [0.0, -1.1], color="#8e5ba8")
        # transmission grating and a fan of colours onto a line camera
        ax.add_patch(mpatches.Rectangle((z_bs - 0.4, -1.18), 0.8, 0.08, facecolor="0.6",
                                        edgecolor="0.3", hatch="||||", zorder=4))
        label(ax, z_bs - 0.5, -1.15, "grating", ha="right", fontsize=8)
        colours = ("#3a6fd8", "#2ca7a0", "#5aa469", "#e0a000", "#d6604d")
        for i, c in enumerate(colours):
            zc = z_bs + 0.25 + 0.38*i
            ax.plot([z_bs, zc], [-1.18, -2.2], color=c, lw=1.6, alpha=0.9)
        ax.add_patch(mpatches.Rectangle((z_bs + 0.1, -2.32), 1.85, 0.12, facecolor="0.35",
                                        edgecolor="none", zorder=4))
        label(ax, z_bs + 1.0, -2.4, "line camera: $I(k)$", va="top", fontsize=8)
        k = np.linspace(0, 1, 500)
        spec = np.exp(-((k - 0.5)/0.22)**2)*(1 + 0.5*np.cos(2*np.pi*14*k) + 0.3*np.cos(2*np.pi*31*k))
        ax.plot(z_bs + 2.4 + 1.6*k, -2.0 + 0.4*spec/spec.max(), color="#8e5ba8", lw=0.8)
        label(ax, z_bs + 3.2, -2.25, "spectrum: fringes,\nfrequency ∝ depth", va="top", fontsize=8)
    _finish(ax, (-1.0, 6.2), (-3.0, 2.9), title)


BEAD = "#e8eef5"


def _refract2d(d, n_hat, n1, n2):
    """Refract the unit direction d at a surface with unit normal n_hat
    (pointing against d) from index n1 into n2 (None on total reflection)."""

    cos_i = -np.dot(d, n_hat)
    mu = n1/n2
    k = 1 - mu**2*(1 - cos_i**2)
    if k < 0:
        return None
    return mu*d + (mu*cos_i - np.sqrt(k))*n_hat


def _ray_through_sphere(p0, d, c, R, n_rel):
    """Points of a ray from p0 along d through a circle (centre c, radius R)
    and its exit direction (refraction only, no reflections), or None if it
    misses."""

    p0, d, c = (np.asarray(v, dtype=float) for v in (p0, d, c))
    oc = p0 - c
    b = np.dot(oc, d)
    disc = b**2 - (np.dot(oc, oc) - R**2)
    if disc <= 0:
        return None
    p1 = p0 + (-b - np.sqrt(disc))*d
    t = _refract2d(d, (p1 - c)/R, 1.0, n_rel)
    p2 = p1 - 2*np.dot(p1 - c, t)*t
    d_out = _refract2d(t, -(p2 - c)/R, n_rel, 1.0)
    return p1, p2, d_out


def optical_tweezers(ax, title=None):
    """Sketch of an optical tweezers setup (notebook 75).

    A laser beam is expanded so that it overfills the back aperture of a
    high NA water immersion objective, which focuses it into a sample
    chamber. A bead near the focus is trapped slightly beyond it. A
    condenser collects the light behind the chamber; a quadrant photodiode
    in its back focal plane measures the bead's displacement (back focal
    plane interferometry). A dichroic mirror sends the image of the sample
    to a camera.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    z_l1, f1 = 0.0, 0.45
    z_l2, f2 = 1.35, 0.9
    z_dic = 2.35
    z_obj, f_o = 3.4, 0.75
    z_foc = z_obj + f_o
    z_cond, f_c = z_foc + 0.75, 0.75
    z_qpd = z_cond + 1.0
    w = 0.22
    laser(ax, z_l1 - 0.35, 0.0, length=1.0, height=0.34, text="laser\n1064 nm", fontsize=8)
    lenses = [(z_l1, f1), (z_l2, f2)]
    zt, yt = paraxial_ray(lenses, z_l1 - 0.35, w, 0.0, z_obj)
    zb, yb = paraxial_ray(lenses, z_l1 - 0.35, -w, 0.0, z_obj)
    h_obj = 0.38                                       # back aperture, overfilled by the beam
    band(ax, zt, yt, zb, yb, color=LASER, alpha=0.2)
    ax.plot([z_obj, z_foc], [h_obj, 0.0], color=LASER, lw=0.8, zorder=2)
    ax.plot([z_obj, z_foc], [-h_obj, 0.0], color=LASER, lw=0.8, zorder=2)
    ax.add_patch(mpatches.Polygon([(z_obj, h_obj), (z_foc, 0), (z_obj, -h_obj)], closed=True,
                                  facecolor=LASER, alpha=0.3, edgecolor="none", zorder=1))
    # behind the focus: the diverging beam is collected by the condenser
    ax.add_patch(mpatches.Polygon([(z_foc, 0), (z_cond, 0.75*h_obj/f_o*f_c + 0.0),
                                   (z_qpd, 0.75*h_obj/f_o*f_c), (z_qpd, -0.75*h_obj/f_o*f_c),
                                   (z_cond, -0.75*h_obj/f_o*f_c)], closed=True,
                                  facecolor=LASER, alpha=0.14, edgecolor="none", zorder=1))
    lens(ax, z_l1, 0.3, text="beam\nexpander", text_y=-0.5, fontsize=8)
    lens(ax, z_l2, 0.6)
    mirror(ax, z_dic, 0.0, 1.1, np.pi/4, dichroic=True)
    label(ax, z_dic, -0.62, "dichroic", fontsize=8, va="top")
    stop(ax, z_obj - 0.22, 0.75, openings=((0, h_obj),), width=0.04)
    label(ax, z_obj - 0.22, 0.8, "back aperture\n(overfilled)", fontsize=8, va="bottom")
    ax.add_patch(mpatches.Polygon([(z_obj - 0.15, -0.6), (z_obj + 0.25, -0.45), (z_obj + 0.25, 0.45),
                                   (z_obj - 0.15, 0.6)], closed=True, facecolor="0.8", edgecolor="0.4",
                                  zorder=3))
    label(ax, z_obj + 0.05, -0.7, "objective\nNA 1.2 (water)", fontsize=8, va="top")
    # sample chamber: coverslip, water, slide
    ax.add_patch(mpatches.Rectangle((z_obj + 0.4, -0.75), 0.08, 1.5, facecolor=GLASS, edgecolor=EDGE,
                                    lw=0.8, zorder=2))
    ax.add_patch(mpatches.Rectangle((z_obj + 0.48, -0.75), 0.55, 1.5, facecolor="#eaf4fb",
                                    edgecolor="none", zorder=0.5))
    ax.add_patch(mpatches.Rectangle((z_obj + 1.03, -0.75), 0.08, 1.5, facecolor=GLASS, edgecolor=EDGE,
                                    lw=0.8, zorder=2))
    ax.add_patch(mpatches.Circle((z_foc + 0.05, 0.0), 0.09, facecolor=BEAD, edgecolor="0.3", lw=1,
                                 zorder=5))
    label(ax, z_obj + 0.75, 0.8, "sample in water", fontsize=8, va="bottom")
    label(ax, z_foc + 0.05, -0.16, "bead", fontsize=8, va="top")
    lens(ax, z_cond, 0.55)
    label(ax, z_cond, -0.7, "condenser", fontsize=8, va="top")
    ax.add_patch(mpatches.Rectangle((z_qpd, -0.32), 0.07, 0.64, facecolor="0.35", edgecolor="none",
                                    zorder=4))
    ax.plot([z_qpd + 0.035]*2, [-0.32, 0.32], color="white", lw=0.8, zorder=5)
    label(ax, z_qpd + 0.04, -0.4, "QPD\n(back focal\nplane)", fontsize=8, va="top")
    # imaging path: via the dichroic to a camera
    ax.add_patch(mpatches.Rectangle((z_dic - 0.25, 1.15), 0.5, 0.1, facecolor="0.35", edgecolor="none",
                                    zorder=4))
    label(ax, z_dic + 0.32, 1.2, "camera", fontsize=8, ha="left")
    for sgn in (-1, 1):
        ax.plot([z_dic + sgn*0.12, z_dic + sgn*0.12], [sgn*0.12, 1.15], color=IMAGING, lw=0.8,
                ls="--", zorder=2)
    label(ax, z_dic - 0.2, 0.75, "image", fontsize=8, ha="right", color=IMAGING)
    _finish(ax, (z_l1 - 1.45, z_qpd + 0.45), (-1.25, 1.45), title)


def ashkin_rays(ax, offset=(0.6, 0.0), n_rel=1.35, title=None):
    """Ashkin's picture of the forces on a sphere in a focus (notebook 75).

    Two rays a and b from the edge of the objective's aperture converge
    towards the focus f, are refracted by a sphere (index ratio n_rel; the
    weaker reflections are omitted) and leave it in new directions. Each
    ray loses the momentum d_in - d_out, which pushes the sphere (thin
    arrows F_a, F_b); their sum (thick arrow) points back towards the focus.

    Parameters
    ----------
    ax : matplotlib Axes
    offset : (dz, dy), optional
        Position of the sphere's centre relative to the focus, in sphere
        radii. (0.6, 0): beyond the focus, (-0.6, 0): before it, (0, 0.6):
        beside it.
    n_rel : number, optional
    title : str, optional
    """

    mpatches = _patches()
    R = 1.0
    c = np.array(offset, dtype=float)*R
    ax.add_patch(mpatches.Circle(tuple(c), R, facecolor=BEAD, edgecolor="0.3", lw=1.2, zorder=1))
    ax.plot(0, 0, "+", color="0.2", ms=9, mew=1.5, zorder=5)
    label(ax, -0.1, -0.12, "f", fontsize=11, ha="right", va="top", style="italic")
    z_start = -2.6
    F_sum = np.zeros(2)
    for name, ang, col in [("a", 0.6, LASER), ("b", -0.6, "#b0413e")]:
        d = np.array([np.cos(ang), np.sin(ang)])
        p0 = -d*(-z_start/np.cos(ang))
        res = _ray_through_sphere(p0, d, c, R, n_rel)
        if res is None:
            ax.plot([p0[0], 2.4], [p0[1], p0[1] + d[1]/d[0]*(2.4 - p0[0])], color=col, lw=1.4)
            continue
        p1, p2, d_out = res
        if d_out is None:
            continue
        p3 = p2 + d_out*(2.6 - p2[0])/max(d_out[0], 0.2)
        ax.plot([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], color=col, lw=1.4, zorder=3)
        ax.plot([p1[0], 0], [p1[1], 0], color=col, lw=0.8, ls=":", zorder=2)
        arrow(ax, *(p0 + 0.35*d), *(p0 + 0.6*d), color=col, lw=1.4)
        label(ax, p0[0] - 0.05, p0[1] + (0.18 if ang > 0 else -0.18), name, fontsize=10, color=col,
              style="italic")
        F = d - d_out
        F_sum += F
        tip = c + 3.0*F
        arrow(ax, c[0], c[1], tip[0], tip[1], color=col, lw=1.3, mutation_scale=11)
        e = F/np.linalg.norm(F)
        label(ax, tip[0] + 0.2*e[0], tip[1] + 0.2*e[1], "F$_{}$".format(name), fontsize=10, color=col)
    tip = c + 3.0*F_sum
    arrow(ax, c[0], c[1], tip[0], tip[1], color="0.1", lw=2.6, mutation_scale=15)
    ax.plot(*c, "o", color="0.1", ms=3, zorder=6)
    e = F_sum/np.linalg.norm(F_sum)
    label(ax, tip[0] + 0.22*e[0], tip[1] + 0.22*e[1], "F", fontsize=12, color="0.1")
    _finish(ax, (z_start - 0.1, 2.7), (-2.0, 2.0), title)


def momentum_balance(ax, title=None):
    """The force on a scatterer from the momentum flux far away (notebook 75).

    The focused beam comes in through the left part of a large sphere
    around the focus and leaves through the right part. A bead near the
    focus scatters light into all directions and changes the outgoing
    beam; the force on the bead is the momentum that comes in minus the
    momentum that goes out, the Maxwell stress tensor integrated over the
    large sphere.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    Rs = 2.0
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(Rs*np.cos(th), Rs*np.sin(th), color="0.45", lw=1, ls="--")
    a = np.deg2rad(55)
    ax.add_patch(mpatches.Polygon([(-Rs*np.cos(a), Rs*np.sin(a)), (0, 0), (-Rs*np.cos(a), -Rs*np.sin(a))],
                                  closed=True, facecolor=LASER, alpha=0.28, edgecolor="none"))
    b = np.deg2rad(46)
    ax.add_patch(mpatches.Polygon([(0, 0), (Rs*np.cos(b), Rs*np.sin(b)), (Rs*np.cos(b), -Rs*np.sin(b))],
                                  closed=True, facecolor=LASER, alpha=0.22, edgecolor="none"))
    ax.add_patch(mpatches.Circle((0.18, 0), 0.22, facecolor=BEAD, edgecolor="0.3", lw=1, zorder=4))
    for r in (0.55, 0.85, 1.15, 1.45):
        ax.plot(0.18 + r*np.cos(th), r*np.sin(th), color="#2a7ab9", lw=0.7, alpha=0.6, zorder=2)
    for ang in (180 - 35, 180, 180 + 35):
        t = np.deg2rad(ang)
        arrow(ax, 1.55*Rs*np.cos(t)*0.7 + 0.0, 1.55*Rs*np.sin(t)*0.7, 1.05*Rs*np.cos(t)*0.7,
              1.05*Rs*np.sin(t)*0.7, color=LASER, lw=1.4)
    for ang in (-28, 0, 28):
        t = np.deg2rad(ang)
        arrow(ax, 1.03*Rs*np.cos(t), 1.03*Rs*np.sin(t), 1.33*Rs*np.cos(t), 1.33*Rs*np.sin(t),
              color=LASER, lw=1.4)
    for ang in (75, 120, -75, -120, 160):
        t = np.deg2rad(ang)
        arrow(ax, 1.03*Rs*np.cos(t), 1.03*Rs*np.sin(t), 1.22*Rs*np.cos(t), 1.22*Rs*np.sin(t),
              color="#2a7ab9", lw=1.0)
    label(ax, -2.95, 1.5, "incoming\nbeam", fontsize=9, color=LASER)
    label(ax, 2.95, 1.25, "outgoing beam\n+ scattered", fontsize=9, color=LASER)
    label(ax, 0.0, 2.55, "scattered light (all directions)", fontsize=9, color="#2a7ab9")
    label(ax, 0.0, -2.5, "F = (momentum in) − (momentum out)\n= ∮ T·n dA on a sphere far away",
          fontsize=9, va="top")
    label(ax, 0.18, -0.32, "bead", fontsize=8, va="top")
    _finish(ax, (-3.6, 3.6), (-3.1, 2.8), title)


PULSE = "#d6604d"
PHASE_FRONT = "0.45"


def pulse_through_lens(ax, achromat=False, title=None):
    """Sketch of Bor's pulse-front delay in a lens (notebook 81).

    A collimated beam carries a short pulse (red band) on a plane phase
    front. Behind a singlet the phase fronts (grey arcs) converge on the
    focus, but the pulse travels with the group velocity: it lags behind
    the phase front, most on the axis, where the glass is thickest, and
    reaches the focus spread in time by the delay Delta T. Behind an
    achromat (achromat=True, a cemented crown and flint doublet) pulse front
    and phase front coincide.

    Parameters
    ----------
    ax : matplotlib Axes
    achromat : bool, optional
    title : str, optional
    """

    mpatches = _patches()
    h, f = 1.0, 3.2
    z_l = 0.0
    if achromat:
        yy = np.linspace(-h, h, 41)
        b1 = 0.16*(1 - (yy/h)**2)
        crown = np.r_[np.c_[z_l - 0.02 - b1, yy], np.c_[np.full_like(yy, z_l + 0.02), yy[::-1]]]
        ax.add_patch(mpatches.Polygon(crown, closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1,
                                      zorder=3))
        flint = np.r_[np.c_[np.full_like(yy, z_l + 0.02), yy],
                      np.c_[z_l + 0.12 - 0.06*(1 - (yy[::-1]/h)**2), yy[::-1]]]
        ax.add_patch(mpatches.Polygon(flint, closed=True, facecolor="#9fbfdc", edgecolor=EDGE,
                                      lw=1, zorder=3))
        label(ax, z_l, -1.2, "achromat (crown + flint)", va="top")
    else:
        lens(ax, z_l, h, thickness=0.36)
        label(ax, z_l, -1.2, "singlet", va="top")
    # incoming beam, phase fronts and the pulse
    band(ax, [-2.6, z_l], [h, h], [-2.6, z_l], [-h, -h], color=PULSE, alpha=0.08, edges=False)
    for z in (-2.3, -1.9, -1.5, -0.7, -0.3):
        ax.plot([z, z], [-h, h], color=PHASE_FRONT, lw=0.8, zorder=2)
    ax.add_patch(mpatches.Rectangle((-1.15, -h), 0.12, 2*h, facecolor=PULSE, alpha=0.85,
                                    edgecolor="none", zorder=4))
    label(ax, -1.1, 1.22, "pulse on a\nphase front", fontsize=8, va="bottom")
    # converging beam
    band(ax, [z_l, f], [h, 0.0], [z_l, f], [-h, 0.0], color=PULSE, alpha=0.08, edges=False)
    for zc in (1.0, 1.7, 2.3):
        rr = f - zc
        y_edge = h*rr/f
        phi = np.linspace(-np.arcsin(y_edge/np.hypot(rr, 0) if rr else 0), np.arcsin(y_edge/rr), 41)
        ax.plot(f - rr*np.cos(phi), rr*np.sin(phi), color=PHASE_FRONT, lw=0.8, zorder=2)
    # the pulse front behind the lens, on (or behind) the phase front at z = 1.45
    rr = f - 1.45
    y_edge = h*rr/f
    phi = np.linspace(-np.arcsin(y_edge/rr), np.arcsin(y_edge/rr), 61)
    zp, yp = f - rr*np.cos(phi), rr*np.sin(phi)
    ax.plot(zp, yp, color=PHASE_FRONT, lw=0.8, ls="--", zorder=2)
    lag = 0.0 if achromat else 0.38*(1 - (yp/y_edge)**2)
    ax.plot(zp - lag, yp, color=PULSE, lw=4.5, alpha=0.9, solid_capstyle="round", zorder=4)
    if achromat:
        label(ax, 1.45, 0.9, "pulse front =\nphase front", fontsize=8, va="bottom")
    else:
        arrow(ax, zp[30], 0.0, zp[30] - 0.38, 0.0, color="0.15", style="<->",
              text="ΔT", text_pos=(zp[30] - 0.19, 0.13), fontsize=9)
        label(ax, 1.25, 0.9, "pulse front lags behind\nthe phase front (n_g > n)",
              fontsize=8, va="bottom")
    point_source(ax, f, 0.0, color="k", size=18)
    label(ax, f + 0.1, -0.22, "focus", fontsize=8, ha="left")
    _finish(ax, (-2.7, f + 0.7), (-1.55, 1.6), title)


def pulse_front_tilt(ax, element="prism", title=None):
    """Sketch of the pulse-front tilt behind a prism or a grating (notebook
    81).

    The colours leave the element at different angles (angular dispersion
    d theta/d lambda). The phase fronts stay normal to the beam, but the
    pulse front (red band) is tilted against them by the angle gamma with
    tan gamma = lambda d theta/d lambda: behind a prism the part of the beam
    that crossed the thick base lags; behind a grating the part that hit
    the grating later lags.

    Parameters
    ----------
    ax : matplotlib Axes
    element : {"prism", "grating"}, optional
    title : str, optional
    """

    if element not in ("prism", "grating"):
        raise ValueError("element must be 'prism' or 'grating'.")
    mpatches = _patches()
    w = 0.32
    if element == "prism":
        apex = np.deg2rad(60)
        top, base = (0.0, 1.0), 1.7
        a2 = apex/2
        pts = [top, (-base*np.tan(a2), top[1] - base), (base*np.tan(a2), top[1] - base)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=GLASS, edgecolor=EDGE, lw=1,
                                      zorder=3))
        z_in, y_in, ang_in = -0.5, 0.15, np.deg2rad(22)
        p_out, ang_out = (0.5, 0.15), -np.deg2rad(22)
        label(ax, 0.0, -0.95, "prism", va="top")
        tilt = -np.deg2rad(28)       # the base side lags; drawn exaggerated
        colours = ((-0.12, "#2a7ab9"), (0.0, "#5aa469"), (0.12, "#d6604d"))
    else:
        z_g = 0.0
        ax.plot([z_g, z_g], [-1.0, 1.0], color="0.25", lw=2.5, zorder=3)
        for y in np.linspace(-0.95, 0.95, 20):
            ax.plot([z_g, z_g + 0.06], [y, y + 0.04], color="0.25", lw=0.8, zorder=3)
        z_in, y_in, ang_in = z_g, 0.0, 0.0
        p_out, ang_out = (z_g, 0.0), -np.deg2rad(30)
        label(ax, 0.0, -1.1, "grating", va="top")
        tilt = np.deg2rad(30)        # pulse front parallel to the grating
        colours = ((0.12, "#2a7ab9"), (0.0, "#5aa469"), (-0.12, "#d6604d"))
    # incoming beam (from the left, at angle ang_in) with a normal pulse front
    d_in = np.array([np.cos(ang_in), np.sin(ang_in)])
    n_in = np.array([-d_in[1], d_in[0]])
    P0 = np.array([z_in, y_in]) - 2.2*d_in
    e1, e2 = P0 + w*n_in, P0 - w*n_in
    f1, f2 = np.array([z_in, y_in]) + w*n_in, np.array([z_in, y_in]) - w*n_in
    band(ax, [e1[0], f1[0]], [e1[1], f1[1]], [e2[0], f2[0]], [e2[1], f2[1]], color=PULSE,
         alpha=0.1, edges=False)
    c = P0 + 1.1*d_in
    ax.plot([c[0] + w*n_in[0], c[0] - w*n_in[0]], [c[1] + w*n_in[1], c[1] - w*n_in[1]],
            color=PULSE, lw=4.5, alpha=0.9, zorder=4)
    for s in (0.5, 0.8, 1.4, 1.7):
        q = P0 + s*d_in
        ax.plot([q[0] + w*n_in[0], q[0] - w*n_in[0]], [q[1] + w*n_in[1], q[1] - w*n_in[1]],
                color=PHASE_FRONT, lw=0.8, zorder=2)
    # dispersed colours
    for da, col in colours:
        a = ang_out + da
        ax.plot([p_out[0], p_out[0] + 2.4*np.cos(a)], [p_out[1], p_out[1] + 2.4*np.sin(a)],
                color=col, lw=1.0, zorder=2)
    # outgoing beam: phase fronts normal to it, tilted pulse front
    d_o = np.array([np.cos(ang_out), np.sin(ang_out)])
    n_o = np.array([-d_o[1], d_o[0]])
    O = np.array(p_out)
    band(ax, [O[0] + w*n_o[0], O[0] + w*n_o[0] + 2.4*d_o[0]],
         [O[1] + w*n_o[1], O[1] + w*n_o[1] + 2.4*d_o[1]],
         [O[0] - w*n_o[0], O[0] - w*n_o[0] + 2.4*d_o[0]],
         [O[1] - w*n_o[1], O[1] - w*n_o[1] + 2.4*d_o[1]], color=PULSE, alpha=0.1, edges=False)
    for s in (0.6, 0.95, 1.95, 2.25):
        q = O + s*d_o
        ax.plot([q[0] + w*n_o[0], q[0] - w*n_o[0]], [q[1] + w*n_o[1], q[1] - w*n_o[1]],
                color=PHASE_FRONT, lw=0.8, zorder=2)
    q = O + 1.45*d_o
    m = np.array([np.cos(np.arctan2(n_o[1], n_o[0]) + tilt), np.sin(np.arctan2(n_o[1], n_o[0]) + tilt)])
    L = w/np.cos(tilt)
    ax.plot([q[0] + L*m[0], q[0] - L*m[0]], [q[1] + L*m[1], q[1] - L*m[1]], color=PULSE, lw=4.5,
            alpha=0.9, zorder=4)
    ax.plot([q[0] + w*n_o[0], q[0] - w*n_o[0]], [q[1] + w*n_o[1], q[1] - w*n_o[1]],
            color=PHASE_FRONT, lw=0.8, ls="--", zorder=3)
    label(ax, q[0] + 0.55, q[1] + 0.55, "tilted pulse front\ntan γ = λ dθ/dλ", fontsize=8,
          ha="left")
    label(ax, P0[0] + 1.0, P0[1] + 0.95, "pulse front =\nphase front", fontsize=8)
    _finish(ax, (P0[0] - 0.3, O[0] + 3.2), (-1.5, 1.45), title)


# ---------------------------------------------------------------------------
# Foundations and conventions (notebook 00)
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, title, lines=(), modules=None, color="#eef4fa", edgecolor=EDGE,
         fontsize=8.5):
    """A rounded box with a bold title, some lines of text and a line of
    module names in a monospaced font (the approximation map)."""

    mpatches = _patches()
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                         boxstyle="round,pad=0.02,rounding_size=0.15",
                                         facecolor=color, edgecolor=edgecolor, lw=1.1, zorder=3))
    rows = [(title, dict(weight="bold", fontsize=fontsize + 0.5))]
    rows += [(t, dict(fontsize=fontsize)) for t in lines]
    if modules:
        rows.append((modules, dict(fontsize=fontsize - 0.5, family="monospace", color="#1d4f7a")))
    n = len(rows)
    for i, (t, kw) in enumerate(rows):
        yy = y + h/2*(0.62 - 1.24*i/max(n - 1, 1)) if n > 1 else y
        ax.text(x, yy, t, ha="center", va="center", zorder=4, **kw)


def approximation_map(ax, title=None):
    """The levels of approximation from Maxwell's equations down to rays and
    Fraunhofer diffraction, with the pyoptics modules that work on each
    level (notebook 00).

    Every arrow names the assumption that leads from one level to the next.
    Polarization (Jones and Stokes vectors) branches off where the field is
    a sum of transverse plane waves, interfaces and thin films where the
    medium is piecewise homogeneous in planar layers; both are exact
    solutions of Maxwell's equations under these assumptions.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    W, H = 4.7, 1.25
    boxes = {
        "maxwell": (8.0, 10.0, "Maxwell's equations",
                    ("∇×E = −∂B/∂t,  ∇×H = J + ∂D/∂t",),
                    "fdtd  (time domain, any structure)"),
        "vector": (8.0, 7.9, "vector Helmholtz equation",
                   ("∇×∇×E − k₀² n²(r) E = 0",),
                   "fdfd  litho.rcwa  analytical (Mie)"),
        "layers": (2.2, 7.9, "planar interfaces and layers",
                   ("Fresnel coefficients, 2×2 matrices",),
                   "interfaces  thin_films  coatings"),
        "jones": (13.6, 7.9, "polarization",
                  ("E⊥ = (E_x, E_y): Jones, Stokes, Mueller",),
                  "polarization  litho.jones_pupil"),
        "scalar": (8.0, 5.8, "scalar Helmholtz equation",
                   ("(∇² + k₀² n²(r)) U = 0",),
                   "wpm  fdfd (2D: Ez or Hz exactly)"),
        "angular": (4.6, 3.7, "angular spectrum = Rayleigh-Sommerfeld",
                    ("U = Σ A(k⊥) exp(i(k⊥·r⊥ + k_z z))",),
                    "propagators (also vectorial)  tilted"),
        "eikonal": (11.4, 3.7, "eikonal and rays",
                    ("|∇S|² = n²,  d/ds(n dr/ds) = ∇n",),
                    "raytrace  grin  raywave"),
        "fresnel": (4.6, 1.6, "Fresnel (paraxial) diffraction",
                    ("2ik ∂_z u + ∇⊥² u = 0: Gaussian beams",),
                    "propagators.fresnel_*  beams  modes"),
        "abcd": (11.4, 1.6, "paraxial rays",
                 ("(y, nu) → ABCD,  q-law for beams",),
                 "abcd  delano  raytrace.paraxial()"),
        "fraun": (4.6, -0.5, "Fraunhofer diffraction, imaging",
                  ("far field = Fourier transform of the pupil",),
                  "propagators.fraunhofer  imaging  litho"),
    }
    colors = {"maxwell": "#fde9c9", "layers": "#e6f2e3", "jones": "#efe6f5"}
    widths = {"layers": 4.2, "jones": 4.2}
    for key, (x, y, t, lines, mods) in boxes.items():
        _box(ax, x, y, widths.get(key, W), H, t, lines, mods,
             color=colors.get(key, "#eef4fa"))

    def link(a, b, text, side="right", dx=0.0, ls="-", rad=0.0, fontsize=7.8):
        xa, ya = boxes[a][:2]
        xb, yb = boxes[b][:2]
        if abs(ya - yb) < 1e-9:      # horizontal link
            sgn = np.sign(xb - xa)
            p0, p1 = (xa + sgn*widths.get(a, W)/2, ya), (xb - sgn*widths.get(b, W)/2, yb)
        else:
            p0, p1 = (xa + dx, ya - H/2), (xb + dx, yb + H/2)
        ax.annotate("", xy=p1, xytext=p0, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.0, ls=ls,
                                    mutation_scale=10, shrinkA=1, shrinkB=1,
                                    connectionstyle="arc3,rad=%g" % rad))
        xm, ym = (p0[0] + p1[0])/2, (p0[1] + p1[1])/2
        if abs(ya - yb) < 1e-9:
            ax.text(xm, ym - 0.1, text, fontsize=fontsize, color="0.25", style="italic",
                    ha="center", va="top", zorder=4)
            return
        off = 0.12 if side == "right" else -0.12
        ax.text(xm + off, ym, text, fontsize=fontsize, color="0.25", style="italic",
                ha="left" if side == "right" else "right", va="center", zorder=4)

    link("maxwell", "vector", "time harmonic exp(−iωt),\nlinear isotropic medium")
    link("maxwell", "jones", "transverse plane waves", rad=-0.15)
    link("vector", "layers", "planar\nlayers")
    link("vector", "scalar", "n varies slowly over λ:\ncomponents decouple")
    link("scalar", "angular", "homogeneous n", side="left")
    link("scalar", "eikonal", "λ → 0:  U = A exp(ik₀S)")
    link("angular", "fresnel", "paraxial: k_z ≈ k − k⊥²/2k", side="left")
    link("eikonal", "abcd", "small angles and heights")
    link("fresnel", "fraun", "far field: N_F = a²/λz ≪ 1\n(or the focal plane of a lens)",
         side="left")
    ax.annotate("", xy=(11.4 - W/2, 1.6), xytext=(4.6 + W/2, 1.6), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color="0.45", lw=1.0, ls="--",
                                mutation_scale=10))
    ax.text(8.0, 1.75, "same matrices", fontsize=7.8, color="0.35", style="italic",
            ha="center", va="bottom")
    ax.annotate("", xy=(11.4 - W/2, 3.7), xytext=(4.6 + W/2, 3.7), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color="0.45", lw=1.0, ls="--",
                                mutation_scale=10))
    ax.text(8.0, 3.85, "rays ↔ waves:\nstationary phase", fontsize=7.8, color="0.35",
            style="italic", ha="center", va="bottom")
    _finish(ax, (-0.1, 16.1), (-1.4, 10.9), title)


def sign_conventions(ax, title=None):
    """Coordinates and sign conventions of pyoptics in one meridional
    section (notebook 00): the axes, the direction of propagation, the
    signs of radii, thicknesses, ray angles and field angles, and the two
    mirror conventions of `abcd` (unfolded) and `raytrace` (real geometry).

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    mpatches = _patches()
    # axes
    z0, y0 = -1.7, -1.85
    arrow(ax, z0, y0, z0 + 1.0, y0, color="0.1", lw=1.4)
    arrow(ax, z0, y0, z0, y0 + 1.0, color="0.1", lw=1.4)
    ax.add_patch(mpatches.Circle((z0, y0), 0.09, facecolor="white", edgecolor="0.1", lw=1.2,
                                 zorder=5))
    ax.plot(z0, y0, "o", color="0.1", ms=2.5, zorder=6)
    label(ax, z0 + 1.12, y0, "z", fontsize=11, ha="left")
    label(ax, z0, y0 + 1.15, "y", fontsize=11, va="bottom")
    label(ax, z0 - 0.15, y0 - 0.2, "x (towards you)", fontsize=8, ha="right", va="top")
    ax.plot([-1.2, 13.3], [0, 0], color="0.6", lw=0.8, ls="-.", zorder=1)
    label(ax, 9.6, 0.12, "optical axis", fontsize=8, ha="right", va="bottom", color="0.4")
    # incoming plane wave with tilted field
    field = np.deg2rad(12)
    for yy in (0.55, 1.0, 1.45):
        ax.plot([-1.0, 1.4], [yy - 0.9 - 2.4*np.tan(field), yy - 0.9], color=ILLUMINATION, lw=1.0)
    label(ax, 0.6, -1.15, "field > 0: rays go up (+y)", fontsize=8, color="#8a6300")
    # wave crest moving in +z
    zz = np.linspace(-1.0, 1.5, 200)
    ax.plot(zz, 1.85 + 0.18*np.cos(2*np.pi*zz/0.5), color=IMAGING, lw=1.0)
    arrow(ax, 0.9, 2.25, 1.6, 2.25, color=IMAGING, lw=1.2)
    label(ax, 0.25, 2.55, "exp(i(kz − ωt)) travels in +z", fontsize=8.5, color=IMAGING)
    # biconvex lens with its two centres of curvature
    zv1, zv2, h, R1, R2 = 3.0, 3.9, 1.3, 2.2, -2.2
    yy = np.linspace(-h, h, 61)
    s1 = zv1 + R1 - np.sqrt(R1**2 - yy**2)
    s2 = zv2 + R2 + np.sqrt(R2**2 - yy**2)
    ax.add_patch(mpatches.Polygon(np.r_[np.c_[s1, yy], np.c_[s2[::-1], yy[::-1]]], closed=True,
                                  facecolor=GLASS, edgecolor=EDGE, lw=1.1, zorder=3))
    for zv, R, name, yl in ((zv1, R1, "R₁ > 0", -0.45), (zv2, R2, "R₂ < 0", -0.95)):
        c = zv + R
        ax.plot(c, 0, "+", color="0.2", ms=8, mew=1.2, zorder=5)
        arrow(ax, zv, yl, c, yl, color="0.3", lw=0.9)
        label(ax, (zv + c)/2, yl - 0.18, name, fontsize=8.5, va="top")
    label(ax, zv1 + R1, 0.2, "C₁", fontsize=8, va="bottom")
    label(ax, zv2 + R2, 0.2, "C₂", fontsize=8, va="bottom")
    label(ax, 3.45, 1.75, "R > 0: centre of\ncurvature in +z", fontsize=8.5)
    # thickness
    arrow(ax, zv1, -1.55, zv2, -1.55, style="<->", color="0.3", lw=0.9)
    label(ax, 3.45, -1.75, "d > 0", fontsize=8.5, va="top")
    # a ray with positive slope behind the lens
    zr0, zr1 = 6.0, 8.2
    ax.plot([zr0, zr1], [-0.3, 0.7], color=IMAGING, lw=1.4, zorder=4)
    ax.plot([zr0, zr0 + 1.2], [-0.3, -0.3], color="0.5", lw=0.7, ls=":")
    a = np.arctan2(1.0, zr1 - zr0)
    arc = np.linspace(0, a, 20)
    ax.plot(zr0 + 0.9*np.cos(arc), -0.3 + 0.9*np.sin(arc), color=IMAGING, lw=0.8)
    label(ax, 7.4, 1.25, "u = dy/dz > 0\n(ABCD: reduced angle n·u)", fontsize=8, color=IMAGING)
    # concave mirror at the end: two conventions
    zm, Rm, hm = 12.2, -2.6, 1.3
    ym = np.linspace(-hm, hm, 61)
    sm = zm + Rm + np.sqrt(Rm**2 - ym**2)
    ax.plot(sm, ym, color="0.2", lw=2.4, zorder=4)
    for yh in np.linspace(-hm, hm, 9):
        zs = zm + Rm + np.sqrt(Rm**2 - yh**2)
        ax.plot([zs, zs + 0.15], [yh, yh + 0.1], color="0.2", lw=0.7)
    for yb in (-0.8, 0.8):
        zs = zm + Rm + np.sqrt(Rm**2 - yb**2)
        ax.plot([9.6, zs], [yb, yb], color=LASER, lw=1.0)
        ax.plot([zs, zm + Rm/2], [yb, 0], color=LASER, lw=1.0, ls="--")
    ax.plot(zm + Rm, 0, "+", color="0.2", ms=8, mew=1.2, zorder=5)
    label(ax, zm + Rm, -0.2, "C", fontsize=8, va="top")
    label(ax, 11.4, 2.0, "concave mirror:\nraytrace R < 0 (real geometry,\nthicknesses behind it < 0)\n"
          "abcd R > 0 (unfolded, f = R/2)", fontsize=8)
    _finish(ax, (-3.2, 13.5), (-2.4, 2.9), title)


def diffraction_geometry(ax, title=None):
    """Geometry of the Rayleigh-Sommerfeld diffraction integral (notebook
    00): a plane wave on an aperture in the plane z = 0, secondary wavelets
    exp(ikr)/r from its points, the distance r and the angle theta from a
    point Q of the aperture to the observation point P.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str, optional
    """

    a = 1.0
    # screen with an opening
    for y_a, y_b in ((a, 2.0), (-2.0, -a)):
        ax.add_patch(_patches().Rectangle((-0.06, y_a), 0.12, y_b - y_a, facecolor=METAL,
                                          edgecolor="none", zorder=3))
    # incident plane wave
    for z in (-2.2, -1.7, -1.2, -0.7):
        ax.plot([z, z], [-1.6, 1.6], color=ILLUMINATION, lw=1.0)
    arrow(ax, -2.4, 1.85, -1.3, 1.85, color="#8a6300", lw=1.2)
    label(ax, -1.85, 2.1, "U₀ exp(ikz)", fontsize=9, color="#8a6300")
    # wavelets
    for yq in np.linspace(-0.8, 0.8, 5):
        for rr in (0.35, 0.7, 1.05):
            th = np.linspace(-np.pi/2, np.pi/2, 60)
            ax.plot(rr*np.cos(th), yq + rr*np.sin(th), color=IMAGING, lw=0.6, alpha=0.45)
    # Q, P, r, theta
    Q, P = np.array([0.0, -0.45]), np.array([5.2, 1.4])
    ax.plot(*Q, "o", color="0.1", ms=4, zorder=6)
    ax.plot(*P, "o", color=LASER, ms=6, zorder=6)
    arrow(ax, Q[0], Q[1], P[0], P[1], color="0.2", lw=1.1)
    label(ax, Q[0] - 0.15, Q[1] - 0.05, "Q = (x′, y′, 0)", fontsize=9, ha="right")
    label(ax, P[0] + 0.15, P[1] + 0.05, "P = (x, y, z)", fontsize=9, ha="left", color=LASER)
    label(ax, 2.5, 0.8, "r", fontsize=11, style="italic")
    ax.plot([Q[0], Q[0] + 2.0], [Q[1], Q[1]], color="0.5", lw=0.8, ls=":")
    th = np.arctan2(P[1] - Q[1], P[0] - Q[0])
    arc = np.linspace(0, th, 20)
    ax.plot(Q[0] + 1.4*np.cos(arc), Q[1] + 1.4*np.sin(arc), color="0.3", lw=0.8)
    label(ax, Q[0] + 1.65, Q[1] + 0.2, "θ", fontsize=10)
    label(ax, 0.0, -2.25, "aperture, z = 0", fontsize=9, va="top")
    ax.plot([-2.6, 6.4], [0, 0], color="0.6", lw=0.7, ls="-.", zorder=1)
    label(ax, 6.4, -0.15, "z", fontsize=10, ha="right", va="top")
    label(ax, 3.1, -1.25, "U(P) = (1/iλ) ∫ U(Q) cos θ exp(ikr)/r dA\n"
          "(Rayleigh-Sommerfeld I, kr ≫ 1)", fontsize=9)
    label(ax, 3.1, -1.95, "Green's function of (∇² + k²): exp(ikr)/(4πr)", fontsize=8.5, color="0.35")
    _finish(ax, (-2.8, 6.6), (-2.5, 2.4), title)


def eikonal_wavefronts(ax, n2=1.5, title=None):
    """Rays and wavefronts of a point source in front of a glass block
    (notebook 00): the wavefronts are the surfaces of constant optical path
    S, computed along the refracted rays, and the rays are their normals
    (grad S = n s). In the glass the wavefronts are closer by 1/n, and they
    are no longer spheres: the virtual source of the refracted rays is
    smeared out (spherical aberration of a plane surface).

    Parameters
    ----------
    ax : matplotlib Axes
    n2 : number, optional
        Index of the block.
    title : str, optional
    """

    zs, zi, z_end, h = 0.0, 2.0, 6.0, 1.7
    ax.add_patch(_patches().Rectangle((zi, -h - 0.15), z_end - zi, 2*h + 0.3, facecolor=GLASS,
                                      edgecolor=EDGE, lw=1.0, zorder=1))
    point_source(ax, zs, 0.0, color=ILLUMINATION, size=50)
    angles = np.deg2rad(np.linspace(-38, 38, 13))
    rays = []
    for a in angles:
        y_i = (zi - zs)*np.tan(a)
        a2 = np.arcsin(np.sin(a)/n2)
        y_e = y_i + (z_end - zi)*np.tan(a2)
        rays.append((a, y_i, a2, y_e))
        ax.plot([zs, zi, z_end], [0, y_i, y_e], color=IMAGING, lw=0.8, alpha=0.8, zorder=2)
    # wavefronts: points of equal optical path S along the rays
    step = 0.45
    for S in np.arange(step, 8.0, step):
        pts = []
        for a, y_i, a2, y_e in rays:
            L1 = (zi - zs)/np.cos(a)
            if S <= L1:
                pts.append((zs + S*np.cos(a), S*np.sin(a)))
            else:
                s2 = (S - L1)/n2
                if zi + s2*np.cos(a2) > z_end:
                    pts = None
                    break
                pts.append((zi + s2*np.cos(a2), y_i + s2*np.sin(a2)))
        if pts is None:
            continue
        pts = np.array(pts)
        th = np.linspace(angles[0], angles[-1], 60)
        L1 = (zi - zs)/np.cos(th)
        if S <= (zi - zs):          # whole wavefront still in air: a circle arc
            ax.plot(zs + S*np.cos(th), S*np.sin(th), color="0.25", lw=1.0, zorder=3)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="0.25", lw=1.0, zorder=3)
    label(ax, 1.0, 1.85, "S = const", fontsize=9)
    label(ax, 4.0, 1.95, "n = %.1f: wavefronts closer by 1/n" % n2, fontsize=9, va="bottom")
    label(ax, 4.0, -1.95, "rays ⊥ wavefronts:  ∇S = n ŝ,  |∇S| = n", fontsize=9, va="top")
    _finish(ax, (-0.5, 6.3), (-2.6, 2.5), title)
