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
- axicon_bessel(): an axicon makes a Bessel zone, an obstacle in it casts a
  short shadow (notebook 68).
- cubic_phase_airy(): a cubic phase plate and a lens make an Airy beam,
  whose rays touch a parabolic caustic (notebook 68).

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
"""

import numpy as np


__all__ = ["lens", "stop", "plane", "prism", "mirror", "detector", "point_source",
           "polarizer", "label", "paraxial_ray", "draw_ray", "koehler_microscope",
           "abbe_4f", "phase_contrast_microscope", "dic_microscope",
           "confocal_microscope", "fpm_microscope", "rotated_lens", "rotated_detector",
           "beam_splitter", "laser", "sphere", "lenslet_array", "band", "arrow", "wavefront",
           "twyman_green", "fizeau_interferometer", "shack_hartmann_sensor",
           "phase_diversity_setup", "holography_setup", "fold_mirror", "krf_illuminator",
           "spatial_filter", "axicon_bessel", "cubic_phase_airy"]


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

