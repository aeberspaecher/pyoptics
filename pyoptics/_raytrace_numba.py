"""Compiled ray trace kernel for raytrace.System.trace() (optional, needs numba).

The kernel traces each ray through all surfaces in one compiled loop instead
of one set of NumPy array operations per surface, which removes the Python
and temporary array overhead that dominates the trace of small and medium
bundles (ray aiming traces many of them). It covers Surface (conics, even
aspheres, mirrors), IdealLens, Axicon and CoordinateBreak with the same
formulas as their NumPy methods; systems with other surface types are
traced by NumPy.

``available`` tells whether numba could be imported.
"""

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover - depends on the installation
    nb = None

available = nb is not None

# surface kinds:
STANDARD, BREAK, IDEAL, AXICON = 0, 1, 2, 3

def _sag_slope(kind, c, K, A, n_A, cone, r2):
    """Sag z(r2) and (dz/dr)/r of a conic with even aspheric terms or of a
    cone (slope cone)."""

    if kind == AXICON:
        return cone*np.sqrt(r2), cone/np.sqrt(max(r2, 1e-300))
    q = 1.0 - (1.0 + K)*c*c*r2
    sq = np.sqrt(q)  # NaN beyond the conic
    z = c*r2/(1.0 + sq)
    s = c/sq
    p = r2
    for j in range(n_A):
        z += A[j]*p*r2  # A_j r2**(j + 2)
        s += 2*(j + 2)*A[j]*p  # 2 (j + 2) A_j r2**(j + 1)
        p *= r2

    return z, s


def _intersect_conic(c, K, A, n_A, x, y, z, ex, ey, ez):
    a = c*(ex*ex + ey*ey + (1.0 + K)*ez*ez)
    b = 2.0*(c*(x*ex + y*ey + (1.0 + K)*z*ez) - ez)
    C = c*(x*x + y*y + (1.0 + K)*z*z) - 2.0*z
    D = b*b - 4.0*a*C
    # root on the branch through the vertex, stable for c -> 0:
    t = -2.0*C/(b + (1.0 if b >= 0 else -1.0)*np.sqrt(D))
    if (1.0 + K)*c*(z + t*ez) > 1.0:
        t = C/(a*t)
    if n_A > 0:
        if not np.isfinite(t):
            t = -z/ez
        prev = np.inf
        for _ in range(50):
            qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
            zs, s = _sag_slope(STANDARD, c, K, A, n_A, 0.0, qx*qx + qy*qy)
            step = (qz - zs)/(ez - s*(qx*ex + qy*ey))
            t = t - step
            tol = 1.0 + abs(t)
            # converged, or the steps no longer shrink at the rounding level:
            if not np.isfinite(step) or abs(step) < 1e-15*tol or \
                    (abs(step) >= prev and abs(step) < 1e-12*tol):
                break
            prev = abs(step)
    # the solution must lie on the real part of the conic:
    qx, qy = x + t*ex, y + t*ey
    zs, _ = _sag_slope(STANDARD, c, K, A, n_A, 0.0, qx*qx + qy*qy)
    if not np.isfinite(zs):
        t = np.nan

    return t


def _intersect_cone(cone, x, y, z, ex, ey, ez):
    t_plane = -z/ez
    if cone == 0.0:
        return t_plane
    k2 = cone*cone
    a = ez*ez - k2*(ex*ex + ey*ey)
    b = 2.0*(z*ez - k2*(x*ex + y*ey))
    C = z*z - k2*(x*x + y*y)
    sq = np.sqrt(b*b - 4.0*a*C)
    if abs(a) < 1e-14*(abs(b) + abs(C) + 1.0):
        r0 = r1 = -C/b
    else:
        r0, r1 = (-b - sq)/(2.0*a), (-b + sq)/(2.0*a)
    sgn = 1.0 if cone > 0 else -1.0
    best, t = np.inf, np.nan
    for r in (r0, r1):
        if np.isfinite(r) and (z + r*ez)*sgn >= -1e-12*(1.0 + abs(z)):
            dist = abs(r - t_plane)
            if dist < best:
                best, t = dist, r

    return t


def _trace_rays(P0, d0, opl0, O, R, kind, c, K, A, n_A, cone, f, mirror, sd, n, clip,
                out_P, out_d, out_opl, valid):
    S = len(kind)
    for r in nb.prange(P0.shape[0]):
        px, py, pz = P0[r, 0], P0[r, 1], P0[r, 2]
        dx, dy, dz = d0[r, 0], d0[r, 1], d0[r, 2]
        opl = opl0[r]
        ok = True
        out_P[0, r, 0], out_P[0, r, 1], out_P[0, r, 2] = px, py, pz
        out_d[0, r, 0], out_d[0, r, 1], out_d[0, r, 2] = dx, dy, dz
        out_opl[0, r] = opl
        for i in range(S):
            if kind[i] != BREAK and ok:
                # local coordinates: (P - o) @ R
                ux, uy, uz = px - O[i, 0], py - O[i, 1], pz - O[i, 2]
                x = ux*R[i, 0, 0] + uy*R[i, 1, 0] + uz*R[i, 2, 0]
                y = ux*R[i, 0, 1] + uy*R[i, 1, 1] + uz*R[i, 2, 1]
                z = ux*R[i, 0, 2] + uy*R[i, 1, 2] + uz*R[i, 2, 2]
                ex = dx*R[i, 0, 0] + dy*R[i, 1, 0] + dz*R[i, 2, 0]
                ey = dx*R[i, 0, 1] + dy*R[i, 1, 1] + dz*R[i, 2, 1]
                ez = dx*R[i, 0, 2] + dy*R[i, 1, 2] + dz*R[i, 2, 2]
                if kind[i] == AXICON:
                    t = _intersect_cone(cone[i], x, y, z, ex, ey, ez)
                else:
                    t = _intersect_conic(c[i], K[i], A[i], n_A[i], x, y, z, ex, ey, ez)
                qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
                opl += n[i]*t
                r2 = qx*qx + qy*qy
                if clip and sd[i] >= 0 and not np.sqrt(r2) <= sd[i]*(1 + 1e-9):
                    ok = False
                if kind[i] == IDEAL:
                    # towards the focal point of the bundle with direction e
                    fi = f[i]
                    aez = abs(ez)
                    vx, vy, vz = fi*ex/aez - qx, fi*ey/aez - qy, fi*ez/aez - qz
                    dist = np.sqrt(vx*vx + vy*vy + vz*vz)
                    sgn = 1.0 if fi > 0 else (-1.0 if fi < 0 else 0.0)
                    opl += n[i]*(sgn*(abs(fi)/aez - dist) - (qx*ex + qy*ey + qz*ez))
                    ex, ey, ez = sgn*vx/dist, sgn*vy/dist, sgn*vz/dist
                else:
                    _, s = _sag_slope(kind[i], c[i], K[i], A[i], n_A[i], cone[i], r2)
                    Nx, Ny = -qx*s, -qy*s
                    nn = np.sqrt(Nx*Nx + Ny*Ny + 1.0)
                    Nx, Ny, Nz = Nx/nn, Ny/nn, 1.0/nn
                    cos_i = ex*Nx + ey*Ny + ez*Nz
                    if mirror[i]:
                        ex, ey, ez = ex - 2*cos_i*Nx, ey - 2*cos_i*Ny, ez - 2*cos_i*Nz
                    else:
                        # orient N along the ray:
                        if cos_i < 0:
                            Nx, Ny, Nz = -Nx, -Ny, -Nz
                        elif not cos_i > 0:
                            Nx, Ny, Nz = 0.0*Nx, 0.0*Ny, 0.0*Nz  # np.sign(0) = 0, NaN stays NaN
                        cos_i = abs(cos_i)
                        mu = n[i]/n[i + 1]
                        cos_t = np.sqrt(1 - mu*mu*(1 - cos_i*cos_i))  # NaN: total reflection
                        g = cos_t - mu*cos_i
                        ex, ey, ez = mu*ex + g*Nx, mu*ey + g*Ny, mu*ez + g*Nz
                # back to global coordinates: o + Q @ R.T
                px = O[i, 0] + R[i, 0, 0]*qx + R[i, 0, 1]*qy + R[i, 0, 2]*qz
                py = O[i, 1] + R[i, 1, 0]*qx + R[i, 1, 1]*qy + R[i, 1, 2]*qz
                pz = O[i, 2] + R[i, 2, 0]*qx + R[i, 2, 1]*qy + R[i, 2, 2]*qz
                dx = R[i, 0, 0]*ex + R[i, 0, 1]*ey + R[i, 0, 2]*ez
                dy = R[i, 1, 0]*ex + R[i, 1, 1]*ey + R[i, 1, 2]*ez
                dz = R[i, 2, 0]*ex + R[i, 2, 1]*ey + R[i, 2, 2]*ez
                if not (np.isfinite(t) and np.isfinite(dx) and np.isfinite(dy)
                        and np.isfinite(dz)):
                    ok = False
            if ok:
                out_P[i + 1, r, 0], out_P[i + 1, r, 1], out_P[i + 1, r, 2] = px, py, pz
                out_d[i + 1, r, 0], out_d[i + 1, r, 1], out_d[i + 1, r, 2] = dx, dy, dz
                out_opl[i + 1, r] = opl
            else:
                for j in range(3):
                    out_P[i + 1, r, j] = np.nan
                    out_d[i + 1, r, j] = np.nan
                out_opl[i + 1, r] = np.nan
        valid[r] = ok


if available:
    _options = dict(cache=True, error_model="numpy")
    _sag_slope = nb.njit(**_options)(_sag_slope)
    _intersect_conic = nb.njit(**_options)(_intersect_conic)
    _intersect_cone = nb.njit(**_options)(_intersect_cone)
    _kernel = nb.njit(parallel=True, **_options)(_trace_rays)


def surface_kind(surface, types):
    """Kernel code of a surface, or None if the kernel does not cover it.

    types : (Surface, IdealLens, Axicon, CoordinateBreak) classes."""

    Surface, IdealLens, Axicon, CoordinateBreak = types
    t = type(surface)
    if t is Surface:
        return STANDARD
    if t is CoordinateBreak:
        return BREAK
    if t is IdealLens:
        return IDEAL
    if t is Axicon:
        return AXICON

    return None


def trace(P, d, opl, frames, surfaces, kinds, n, clip):
    """Trace rays through the surfaces.

    Parameters
    ----------
    P, d : (N, 3) arrays
        Start points and unit directions in global coordinates.
    opl : (N,) array
    frames : list of (origin, R)
        Frames of the surfaces.
    surfaces : list of Surface
    kinds : list of int
        From surface_kind().
    n : array
        Indices in front of the first surface and behind each surface.
    clip : bool

    Returns
    -------
    points, directions : (S + 1, N, 3) arrays
    opls : (S + 1, N) array
    valid : (N,) bool array
        NaN and False for rays that missed a surface, were totally
        reflected or (clip) vignetted, from that surface on.
    """

    S, N = len(surfaces), len(P)
    O = np.zeros((S, 3))
    R = np.zeros((S, 3, 3))
    c, K, cone, f = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
    n_A = np.zeros(S, dtype=np.int64)
    mirror = np.zeros(S, dtype=np.bool_)
    sd = np.full(S, -1.0)
    for i, (s, kind, (o, Ri)) in enumerate(zip(surfaces, kinds, frames)):
        O[i], R[i] = o, Ri
        n_A[i] = len(s.aspheric)
        if kind == BREAK:
            continue
        c[i], K[i], mirror[i] = s.c, s.conic, s.mirror
        if s.semi_diameter is not None:
            sd[i] = s.semi_diameter
        if kind == IDEAL:
            f[i] = s.f
        elif kind == AXICON:
            cone[i] = s.slope
    A = np.zeros((S, max(1, int(n_A.max(initial=0)))))
    for i, s in enumerate(surfaces):
        A[i, :n_A[i]] = s.aspheric
    out_P = np.empty((S + 1, N, 3))
    out_d = np.empty((S + 1, N, 3))
    out_opl = np.empty((S + 1, N))
    valid = np.empty(N, dtype=np.bool_)
    _kernel(np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(d, dtype=float),
           np.ascontiguousarray(opl, dtype=float), O, R, np.asarray(kinds, dtype=np.int64), c, K, A,
           n_A, cone, f, mirror, sd, np.asarray(n, dtype=float), bool(clip), out_P, out_d, out_opl,
           valid)

    return out_P, out_d, out_opl, valid
