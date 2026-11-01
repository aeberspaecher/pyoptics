#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Schematic drawings of optical layouts with matplotlib: thin lenses, stops,
prisms, mirrors, detectors and paraxial rays, and ready-made sketches of
microscopes.

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

Importing this module does not change matplotlib's settings; matplotlib is
imported when a function is called.

References
----------
A. Köhler, "Ein neues Beleuchtungsverfahren für mikrophotographische
Zwecke", Z. wiss. Mikrosk. 10, 433-440 (1893).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), section 8.6 (phase contrast).

J. Mertz, "Introduction to Optical Microscopy", 2nd ed., Cambridge
University Press (2019).

D. B. Murphy and M. W. Davidson, "Fundamentals of Light Microscopy and
Electronic Imaging", 2nd ed., Wiley-Blackwell (2012).
"""

import numpy as np


__all__ = ["lens", "stop", "plane", "prism", "mirror", "detector", "point_source",
           "polarizer", "label", "paraxial_ray", "draw_ray", "koehler_microscope",
           "abbe_4f", "phase_contrast_microscope", "dic_microscope",
           "confocal_microscope", "fpm_microscope"]


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
