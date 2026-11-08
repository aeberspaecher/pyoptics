"""Compiled kernel for illumination.CPC.trace() (optional, needs numba).

Every ray is followed through all its wall reflections in one compiled loop,
with the same steps as the NumPy code of `CPC.trace()`: the wall hit is
bracketed by sampling the ray segment up to the next aperture plane and
refined by bisection, the wall radius and slope are interpolated linearly
on a uniform grid in z. Rays that bounce many times (skew rays near the
rim, rejected light) make the NumPy version slow; here they cost little.

``available`` tells whether numba could be imported.
"""

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover - depends on the installation
    nb = None

available = nb is not None
prange = nb.prange if available else range


def _radius(z, z_step, r_grid):
    m = r_grid.shape[0]
    u = z/z_step
    if u <= 0.0:
        return r_grid[0]
    if u >= m - 1:
        return r_grid[m - 1]
    i = int(u)
    f = u - i
    return (1.0 - f)*r_grid[i] + f*r_grid[i + 1]


def _trace(P, d, fate, k, w, z_step, r_grid, s_grid, length, kind3d, mirror, c_tir,
           reflectivity, max_reflections, n_steps, t_cap):
    N = P.shape[0]
    for j in prange(N):
        if fate[j] >= 0:
            continue
        x, y, z = P[j, 0], P[j, 1], P[j, 2]
        ex, ey, ez = d[j, 0], d[j, 1], d[j, 2]
        while True:
            if ez > 0:
                t_end = (length - z)/ez
            elif ez < 0:
                t_end = -z/ez
            else:
                t_end = np.inf
            if t_end > t_cap:
                t_end = t_cap
            # bracket the first sample outside the wall:
            lo = 0.0
            hi = -1.0
            for i in range(1, n_steps + 1):
                t = t_end*i/n_steps
                qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
                rho = np.sqrt(qx*qx + qy*qy) if kind3d else abs(qx)
                zc = min(max(qz, 0.0), length)
                if _radius(zc, z_step, r_grid) - rho < 0.0:
                    hi = t
                    break
                lo = t
            if hi < 0.0:
                x, y, z = x + t_end*ex, y + t_end*ey, z + t_end*ez
                fate[j] = 0 if ez > 0 else 1
                break
            for _ in range(50):
                mid = 0.5*(lo + hi)
                qx, qy, qz = x + mid*ex, y + mid*ey, z + mid*ez
                rho = np.sqrt(qx*qx + qy*qy) if kind3d else abs(qx)
                zc = min(max(qz, 0.0), length)
                if _radius(zc, z_step, r_grid) - rho >= 0.0:
                    lo = mid
                else:
                    hi = mid
            x, y, z = x + lo*ex, y + lo*ey, z + lo*ez
            s = _radius(min(max(z, 0.0), length), z_step, s_grid)
            if kind3d:
                rho = np.sqrt(x*x + y*y)
                ux, uy = x/rho, y/rho
            else:
                ux, uy = (1.0 if x >= 0 else -1.0), 0.0
            nx, ny, nz = -ux, -uy, s
            nn = np.sqrt(nx*nx + ny*ny + nz*nz)
            nx, ny, nz = nx/nn, ny/nn, nz/nn
            cos_i = -(ex*nx + ey*ny + ez*nz)
            k[j] += 1
            if mirror:
                w[j] *= reflectivity
            elif abs(cos_i) >= c_tir:
                fate[j] = 2
                break
            ex, ey, ez = ex + 2*cos_i*nx, ey + 2*cos_i*ny, ez + 2*cos_i*nz
            ne = np.sqrt(ex*ex + ey*ey + ez*ez)
            ex, ey, ez = ex/ne, ey/ne, ez/ne
            if k[j] > max_reflections:
                fate[j] = 3
                break
        P[j, 0], P[j, 1], P[j, 2] = x, y, z
        d[j, 0], d[j, 1], d[j, 2] = ex, ey, ez


if available:
    _radius = nb.njit(cache=True)(_radius)
    trace = nb.njit(cache=True, parallel=True)(_trace)
else:  # pragma: no cover
    trace = None
