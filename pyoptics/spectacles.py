#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Spectacle lenses: presbyopia, progressive addition lenses and their
power maps, by surface curvature and by ray tracing as worn.

Units: lengths in mm, powers in dioptres (1/m). A surface of curvature
kappa (1/mm) between air and glass of index n has the power
1000 (n - 1) kappa.

Presbyopia
----------
`accommodation_amplitude()` gives Hofstetter's fits of Duane's
measurements of the amplitude of accommodation against age. A presbyope
needs an *addition* (add) of power for near work: with half of the
amplitude held in reserve, add = 1000/d_near - A/2.

Progressive addition lenses
---------------------------
A progressive lens has a distance zone in its upper part, a near zone
below, and a *corridor* between them along which the power rises
smoothly by the add. The progressive surface (usually the front) is a
freeform: its mean curvature must change along the corridor, which by
the Minkwitz theorem forces astigmatism beside it: on an umbilic line
(no astigmatism) the surface astigmatism grows sideways as ::

    d(cyl)/dx = 2 d(P_mean)/dy

so a short corridor or a large add means narrow zones and strong
astigmatism in the lower periphery. Designs differ only in where they
put it ("hard" designs: wide clear zones, steep astigmatism gradients;
"soft" designs: spread out, lower maxima).

`progressive_departure()` designs the departure of the progressive
surface from its base sphere on a grid by linear least squares on its
second derivatives (thin, small slope departures, a common model of PAL
design): the mean curvature should follow a target (zero in the distance
zone, the add in the near zone and the add profile along the corridor),
the surface astigmatism (the difference of the principal curvatures)
should vanish, each with a weight map. `surface_power()` computes the
mean power and the cylinder of any sag z(x, y) exactly from its first
and second fundamental forms.

`ProgressiveLens` puts the progressive front surface (a
`freeform.TabulatedSurface`) and a spherical back surface in front of
the eye and traces the lens *as worn*: for every direction of gaze, a
narrow pencil from the eye's centre of rotation is traced backwards
through the lens, and the curvature matrix of its wavefront behind the
lens gives sphere, cylinder and axis (`ProgressiveLens.power()`,
`ProgressiveLens.power_maps()`).

References
----------
J. Sheedy, R. F. Hardy and J. R. Hayes, "Progressive addition lenses --
measurements and ratings", Optometry 77, 23-39 (2006).

G. Minkwitz, "Über den Flächenastigmatismus bei gewissen symmetrischen
Asphären", Optica Acta 10, 223-227 (1963).

D. J. Meister and S. W. Fisher, "Progress in the spectacle correction of
presbyopia. Part 1: Design and development of progressive lenses", Clin.
Exp. Optom. 91, 240-250 (2008).

J. Wang, R. Gulliver and F. Santosa, "Analysis of a variational approach
to progressive lens design", SIAM J. Appl. Math. 64, 277-296 (2003).

H. W. Hofstetter, "Useful age-amplitude formula", Optom. World 38, 42-45
(1950); A. Duane, "Studies in monocular and binocular accommodation with
their clinical applications", Am. J. Ophthalmol. 5, 865-877 (1922).

ISO 8980-2:2017, "Ophthalmic optics -- Uncut finished spectacle lenses --
Part 2: Specifications for power-variation lenses".
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .freeform import TabulatedSurface
from .raytrace import Surface


def accommodation_amplitude(age, which="mean"):
    """Hofstetter's fits of Duane's amplitude of accommodation.

    Parameters
    ----------
    age : number or array
        Age in years.
    which : {"mean", "min", "max"}, optional
        Mean: 18.5 - 0.3 age, minimum: 15 - 0.25 age, maximum:
        25 - 0.4 age (dioptres, clipped at 0).

    Returns
    -------
    A : number or array
        Amplitude in dioptres.
    """

    age = np.asarray(age, dtype=float)
    a, b = {"mean": (18.5, 0.3), "min": (15.0, 0.25), "max": (25.0, 0.4)}[which]

    return np.clip(a - b*age, 0.0, None)


def required_add(age, near_distance=400.0, reserve=0.5, which="min"):
    """Near addition for a working distance, keeping part of the
    accommodation in reserve.

    Parameters
    ----------
    age : number or array
        Age in years.
    near_distance : number, optional
        Working distance in mm (400 mm: 2.5 D).
    reserve : number, optional
        Fraction of the amplitude held in reserve (0.5: use half).
    which : str, optional
        Amplitude fit, see `accommodation_amplitude()`.

    Returns
    -------
    add : number or array
        Dioptres, rounded up to 0.25 D steps, at least 0.
    """

    need = 1000.0/near_distance - (1 - reserve)*accommodation_amplitude(age, which)

    return np.clip(np.ceil(need*4 - 1e-9)/4, 0.0, None)


def add_profile(y, add, y_fit=4.0, y_near=-14.0):
    """Mean power along the corridor: 0 above the fitting cross height
    y_fit, the add below the near reference height y_near, a smooth
    cubic step in between.

    Parameters
    ----------
    y : array
        Height on the lens (mm, + up).
    add : number
        Addition in dioptres.
    y_fit, y_near : numbers, optional
        Start and end of the corridor (its length is y_fit - y_near).

    Returns
    -------
    P : array
    """

    t = np.clip((y_fit - np.asarray(y, float))/(y_fit - y_near), 0.0, 1.0)

    return add*t**2*(3 - 2*t)


def _second_differences(n, h):
    """Sparse operators of d2/dx2, d2/dy2 and d2/dxdy on an n x n grid
    (index [i, j] = (x_i, y_j), flattened row-major), central
    differences at the interior points; returns the operators and the
    flat indices of the interior points."""

    N = n*n
    idx = np.arange(N).reshape(n, n)
    I = idx[1:-1, 1:-1].ravel()

    def op(entries):
        rows, cols, vals = [], [], []
        for (di, dj), v in entries:
            rows.append(np.arange(len(I)))
            cols.append((idx[1 + di:n - 1 + di, 1 + dj:n - 1 + dj]).ravel())
            vals.append(np.full(len(I), v))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                           np.concatenate(cols))),
                                 shape=(len(I), N))

    Dxx = op([((-1, 0), 1/h**2), ((0, 0), -2/h**2), ((1, 0), 1/h**2)])
    Dyy = op([((0, -1), 1/h**2), ((0, 0), -2/h**2), ((0, 1), 1/h**2)])
    Dxy = op([((1, 1), 1/(4*h*h)), ((-1, -1), 1/(4*h*h)), ((1, -1), -1/(4*h*h)),
              ((-1, 1), -1/(4*h*h))])

    return Dxx, Dyy, Dxy, I


def progressive_departure(x, add, n=1.5, y_fit=4.0, y_near=-14.0, inset=2.5,
                          mean_weight=None, astig_weight=None):
    """Departure of a progressive surface from its base sphere on a square grid.

    Least squares on the second derivatives of the departure d(x, y)
    (small slopes): minimize ::

        sum w_m ((d_xx + d_yy)/2 - H)**2 + w_a ((d_xx - d_yy)**2/4 + d_xy**2)

    with the target mean curvature H = P/(1000 (n - 1)) from
    `add_profile()` along the umbilic line, which runs vertically down to
    y_fit and then bends linearly towards the nasal inset at y_near, and
    the weights w_m (mean power) and w_a (astigmatism, the squared half
    difference of the principal curvatures). The departure and its slopes
    vanish at the fitting cross (0, y_fit) (no prism there).

    Parameters
    ----------
    x : 1d array
        Grid coordinates (mm), the same for x and y, equally spaced.
    add : number
        Addition in dioptres.
    n : number, optional
        Refractive index.
    y_fit, y_near : numbers, optional
        Corridor start (fitting cross) and near reference height.
    inset : number, optional
        Nasal (here: -x) inset of the near zone (mm).
    mean_weight, astig_weight : callables, optional
        w(X, Y, Xc) with Xc the x of the umbilic line at height Y; default:
        `default_weights()` ("hard" design).

    Returns
    -------
    d : 2d array
        Departure d[i, j] at (x[i], x[j]) in mm (axis 0 is x).
    """

    x = np.asarray(x, float)
    m = len(x)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xc = umbilic_line(Y, y_fit, y_near, inset)
    H = add_profile(Y, add, y_fit, y_near)/(1000*(n - 1))
    if mean_weight is None or astig_weight is None:
        wm, wa = default_weights()
        mean_weight = mean_weight or wm
        astig_weight = astig_weight or wa
    Dxx, Dyy, Dxy, I = _second_differences(m, h)
    Wm = np.sqrt(mean_weight(X, Y, Xc).ravel()[I])
    Wa = np.sqrt(astig_weight(X, Y, Xc).ravel()[I])
    blocks = [sparse.diags(Wm) @ (0.5*(Dxx + Dyy)),
              sparse.diags(Wa) @ (0.5*(Dxx - Dyy)),
              sparse.diags(Wa) @ Dxy]
    rhs = [Wm*H.ravel()[I], np.zeros(len(I)), np.zeros(len(I))]
    # gauge: d = 0 and zero slopes at the fitting cross
    i0 = np.argmin(np.abs(x))
    j0 = np.argmin(np.abs(x - y_fit))
    k = i0*m + j0
    big = 1e3*max(1.0, np.max(Wm))/h**2
    G = sparse.lil_matrix((3, m*m))
    G[0, k] = big
    G[1, k + m], G[1, k - m] = big/(2*h), -big/(2*h)
    G[2, k + 1], G[2, k - 1] = big/(2*h), -big/(2*h)
    A = sparse.vstack(blocks + [G.tocsr()]).tocsr()
    b = np.concatenate(rhs + [np.zeros(3)])
    # normal equations, solved directly (a tiny ridge removes the null
    # space of the boundary rows)
    AtA = (A.T @ A).tocsc() + 1e-10*sparse.identity(A.shape[1], format="csc")*float(np.max(Wm))**2/h**4
    sol = spsolve(AtA, A.T @ b)

    return sol.reshape(m, m)


def umbilic_line(Y, y_fit=4.0, y_near=-14.0, inset=2.5):
    """x of the umbilic line at heights Y: 0 above the fitting cross, then
    linear towards -inset at the near reference height and below."""

    t = np.clip((y_fit - np.asarray(Y, float))/(y_fit - y_near), 0.0, 1.0)

    return -inset*t


def default_weights(kind="hard", y_fit=4.0, y_near=-14.0):
    """Weight maps for `progressive_departure()`.

    Parameters
    ----------
    kind : {"hard", "soft"}, optional
        "hard": large weights on the mean power and the astigmatism in a
        wide distance zone and a wide near zone, the astigmatism pushed into
        the lower periphery. "soft": smaller weights on the zones and a
        uniform astigmatism weight everywhere, so it spreads out.
    y_fit, y_near : numbers, optional
        Corridor ends, as for `progressive_departure()`.

    Returns
    -------
    mean_weight, astig_weight : callables w(X, Y, Xc)
    """

    def corridor(X, Y, Xc, width):
        return np.exp(-((X - Xc)/width)**2)

    if kind == "hard":
        def wm(X, Y, Xc):
            dist = 1/(1 + np.exp(-(Y - y_fit - 2.0)/1.0))
            near = (Y < y_near + 2.0)*np.exp(-((X - Xc)/9.0)**4)
            return 30*dist + 30*near + 20*corridor(X, Y, Xc, 2.0)*(Y <= y_fit + 2) + 0.01

        def wa(X, Y, Xc):
            dist = 1/(1 + np.exp(-(Y - y_fit - 1.0)/1.0))
            return 0.2 + 60*dist + 40*corridor(X, Y, Xc, 2.0)*(Y <= y_fit + 2)

        return wm, wa
    if kind == "soft":
        def wm(X, Y, Xc):
            dist = 1/(1 + np.exp(-(Y - y_fit - 3.0)/2.5))
            near = (Y < y_near + 2.0)*np.exp(-((X - Xc)/6.0)**2)
            return 10*dist + 10*near + 20*corridor(X, Y, Xc, 2.0)*(Y <= y_fit + 2) + 0.01

        def wa(X, Y, Xc):
            dist = 1/(1 + np.exp(-(Y - y_fit - 4.0)/3.0))
            return 1.0 + 15*dist*np.exp(-(X/18.0)**2) + 15*corridor(X, Y, Xc, 1.5)*(Y <= y_fit + 2)

        return wm, wa
    raise ValueError("kind must be 'hard' or 'soft'")


def surface_power(x, y, z, n=1.5):
    """Mean power and cylinder of a surface z(x, y) from its fundamental forms.

    Parameters
    ----------
    x, y : 1d arrays
        Grid (axis 0 of z is x), equally spaced.
    z : 2d array
        Sag (mm).
    n : number, optional
        Refractive index (power = 1000 (n - 1) curvature).

    Returns
    -------
    mean : 2d array
        Mean surface power (D).
    cyl : 2d array
        Surface astigmatism (D), 1000 (n - 1) abs(k1 - k2).
    """

    hx, hy = x[1] - x[0], y[1] - y[0]
    zx, zy = np.gradient(z, hx, hy)
    zxx = np.gradient(zx, hx, axis=0)
    zyy = np.gradient(zy, hy, axis=1)
    zxy = np.gradient(zx, hy, axis=1)
    w = 1 + zx**2 + zy**2
    H = ((1 + zy**2)*zxx - 2*zx*zy*zxy + (1 + zx**2)*zyy)/(2*w**1.5)
    K = (zxx*zyy - zxy**2)/w**2
    disc = np.sqrt(np.clip(H**2 - K, 0.0, None))
    s = 1000*(n - 1)

    return s*H, s*2*disc


def sphero_cylinder(P):
    """Sphere, cylinder (negative) and axis of a symmetric 2 x 2 power matrix.

    Parameters
    ----------
    P : (..., 2, 2) array
        Dioptric power matrix in the (horizontal, vertical) basis.

    Returns
    -------
    sphere, cyl, axis : arrays
        Minus cylinder form; axis in degrees (0 to 180, from the horizontal).
    """

    P = np.asarray(P, float)
    a, b, d = P[..., 0, 0], 0.5*(P[..., 0, 1] + P[..., 1, 0]), P[..., 1, 1]
    mean = 0.5*(a + d)
    r = np.sqrt((0.5*(a - d))**2 + b**2)
    sphere = mean + r
    cyl = -2*r
    # the axis of a minus cylinder is the meridian of the larger power
    axis = np.degrees(0.5*np.arctan2(2*b, a - d)) % 180

    return sphere, cyl, axis


class ProgressiveLens:
    """A spectacle lens with a progressive front surface, traced as worn.

    The lens frame: x horizontal (+x temporal for a right lens, the near
    zone is inset towards -x), y up, z from the object towards the eye.
    The front vertex is at the fitting cross, at z = 0; the eye's centre
    of rotation lies on the z axis at z_cre = thickness + vertex distance +
    13.5 mm (the usual sighting centre). The lens is not tilted
    (no pantoscopic tilt, no wrap).

    Parameters
    ----------
    x : 1d array
        Grid of the progressive departure (mm, the same for x and y).
    departure : 2d array or None
        Departure d[i, j] at (x[i], x[j]) from the front base sphere (None:
        a single vision lens).
    front_power : number, optional
        Base curve: surface power of the front sphere (D).
    distance_power : number, optional
        Back vertex power at the fitting cross (D); the back radius is
        solved for it.
    n : number, optional
        Refractive index (CR-39: 1.498).
    thickness : number, optional
        Centre thickness (mm).
    vertex_distance : number, optional
        Back vertex to cornea (mm).
    """

    def __init__(self, x, departure=None, front_power=5.0, distance_power=0.0, n=1.498,
                 thickness=3.0, vertex_distance=12.0):
        self.x = np.asarray(x, float)
        self.n = n
        self.thickness = thickness
        self.vertex_distance = vertex_distance
        self.z_cre = thickness + vertex_distance + 13.5
        self.R1 = 1000*(n - 1)/front_power if front_power else np.inf
        self.departure = None if departure is None else np.asarray(departure, float)
        # back radius for the distance power (thick lens, back vertex power)
        F1 = front_power/1000
        F1v = F1/(1 - thickness/n*F1)              # front power reduced to the back vertex
        F2 = distance_power/1000 - F1v
        self.R2 = (1 - n)/F2 if F2 != 0 else np.inf
        X, Y = np.meshgrid(self.x, self.x, indexing="ij")
        c1 = 0.0 if np.isinf(self.R1) else 1/self.R1
        base = c1*(X**2 + Y**2)/(1 + np.sqrt(1 - c1**2*(X**2 + Y**2)))
        self.front_sag = base + (0.0 if departure is None else self.departure)
        self.front = TabulatedSurface(x=self.x, y=self.x, z=self.front_sag)
        self.back = Surface(self.R2, 0.0)

    def surface_maps(self):
        """Mean power and cylinder of the front surface (D), see `surface_power()`."""

        return surface_power(self.x, self.x, self.front_sag, self.n)

    def _trace_back(self, P, d):
        """Trace rays from the eye side backwards through back and front
        surface; returns points and directions in object space."""

        # back surface: vertex at z = thickness
        o2 = np.array([0.0, 0.0, self.thickness])
        Q = P - o2
        t = self.back.intersect(Q, d)
        Q = Q + t[:, None]*d
        d, _ = self.back.deflect(Q, d, 1.0, self.n)
        P = Q + o2
        t = self.front.intersect(P, d)
        P = P + t[:, None]*d
        d, _ = self.front.deflect(P, d, self.n, 1.0)

        return P, d

    def gaze_direction(self, x, y):
        """Unit vector from the centre of rotation towards the point (x, y)
        of the front vertex plane (the direction of gaze through it)."""

        g = np.array([x, y, -self.z_cre], float)
        return g/np.linalg.norm(g)

    def power(self, x, y, pupil=1.0, n_rays=16):
        """Dioptric power matrix of the lens seen by the eye looking
        through the point (x, y) of the front vertex plane.

        A pencil of rays parallel to the line of sight, of radius `pupil`
        about it at the centre of rotation, is traced backwards out of the
        lens; the curvature of its wavefront in object space, measured on
        the plane normal to the central ray where it leaves the lens, is
        the power the lens adds to the eye along this line of sight
        (positive: plus power).

        Parameters
        ----------
        x, y : numbers
            Point in the front vertex plane (mm).
        pupil : number, optional
            Radius of the pencil (mm).
        n_rays : int, optional
            Rays on the ring around the central ray.

        Returns
        -------
        P : (2, 2) array
            Power matrix (D) in the horizontal/vertical basis of the plane.
        prism : (2,) array
            Deviation of the line of sight (prism dioptres, 100 tan), +x, +y.
        """

        g = self.gaze_direction(x, y)
        cre = np.array([0.0, 0.0, self.z_cre])
        # basis perpendicular to g
        e1 = np.cross([0.0, 1.0, 0.0], g)
        e1 = e1/np.linalg.norm(e1)
        e2 = np.cross(g, e1)
        a = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        off = np.r_[[[0.0, 0.0]], np.c_[pupil*np.cos(a), pupil*np.sin(a)],
                    np.c_[0.5*pupil*np.cos(a), 0.5*pupil*np.sin(a)]]
        P0 = cre + off[:, :1]*e1 + off[:, 1:]*e2
        d0 = np.tile(g, (len(P0), 1))
        P, d = self._trace_back(P0, d0)
        # plane normal to the central ray at its exit point
        n0 = d[0]
        f1 = np.cross([0.0, 1.0, 0.0], n0)
        f1 = f1/np.linalg.norm(f1)
        f2 = np.cross(n0, f1)
        t = ((P[0] - P) @ n0)/(d @ n0)
        Q = P + t[:, None]*d - P[0]
        xi = np.c_[Q @ f1, Q @ f2]
        sl = np.c_[(d @ f1)/(d @ n0), (d @ f2)/(d @ n0)]
        # slopes = M xi; converging rays (plus power) have M negative
        M = np.linalg.lstsq(xi[1:], sl[1:] - sl[0], rcond=None)[0].T
        Pm = -1000*0.5*(M + M.T)
        # the frame (f1, f2) looks along -z: flip x so that +x stays +x
        Pm[0, 1] = Pm[1, 0] = -Pm[0, 1]
        dev = d[0]
        prism = 100*np.array([dev[0]/(-dev[2]) - g[0]/(-g[2]), dev[1]/(-dev[2]) - g[1]/(-g[2])])

        return Pm, prism

    def power_maps(self, xs, ys, pupil=1.0):
        """Mean power and cylinder as worn over a grid of points of the
        front vertex plane.

        Parameters
        ----------
        xs, ys : 1d arrays
            Grid (mm).
        pupil : number, optional
            Radius of the traced pencils (mm).

        Returns
        -------
        mean, cyl, axis : 2d arrays
            Indexed [i, j] at (xs[i], ys[j]); cylinder as a positive
            number; NaN where the pencil misses the lens.
        """

        mean = np.full((len(xs), len(ys)), np.nan)
        cyl = np.full_like(mean, np.nan)
        axis = np.full_like(mean, np.nan)
        for i, xx in enumerate(xs):
            for j, yy in enumerate(ys):
                with np.errstate(invalid="ignore"):
                    Pm, _ = self.power(xx, yy, pupil)
                if np.all(np.isfinite(Pm)):
                    s, c, ax = sphero_cylinder(Pm)
                    mean[i, j], cyl[i, j], axis[i, j] = s + c/2, -c, ax

        return mean, cyl, axis
