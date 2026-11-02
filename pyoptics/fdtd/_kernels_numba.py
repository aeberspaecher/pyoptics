#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Compiled update kernels (optional, need numba).

Each kernel updates all H or all E components of a mode in one parallel
loop over the grid, including the CPML (psi arrays of the full grid size,
touched only where the per-axis flag is set), periodic or PEC ends and
the dispersive polarization densities. They follow the NumPy updates in
simulation.py exactly; 1D runs as a 2D Ez grid with one periodic cell
along y. ``available`` tells whether numba could be imported.
"""

try:
    import numba as nb
except ImportError:  # pragma: no cover - depends on the installation
    nb = None

available = nb is not None


def _jit(f):
    if nb is None:
        return f
    return nb.njit(parallel=True, cache=True, fastmath=False)(f)


if nb is not None:
    prange = nb.prange
else:  # pragma: no cover
    prange = range


@_jit
def h_ez(Ez, Hx, Hy, dt, dx, dy, px, py, fxh, bxh, cxh, kxh, fyh, byh, cyh, kyh, psi_hx_y, psi_hy_x):
    nx, ny = Ez.shape
    for i in prange(nx):
        for j in range(ny):
            e0 = Ez[i, j]
            # Hx -= dt dEz/dy (forward)
            if j < ny - 1:
                e1 = Ez[i, j + 1]
            elif py:
                e1 = Ez[i, 0]
            else:
                e1 = 0.0
            d = (e1 - e0)/dy
            if fyh[j]:
                psi_hx_y[i, j] = byh[j]*psi_hx_y[i, j] + cyh[j]*d
                d = d*kyh[j] + psi_hx_y[i, j]
            Hx[i, j] -= dt*d
            # Hy += dt dEz/dx (forward)
            if i < nx - 1:
                e1 = Ez[i + 1, j]
            elif px:
                e1 = Ez[0, j]
            else:
                e1 = 0.0
            d = (e1 - e0)/dx
            if fxh[i]:
                psi_hy_x[i, j] = bxh[i]*psi_hy_x[i, j] + cxh[i]*d
                d = d*kxh[i] + psi_hy_x[i, j]
            Hy[i, j] += dt*d


@_jit
def e_ez(Ez, Hx, Hy, Ca, Cb, dt, dx, dy, px, py, fxe, bxe, cxe, kxe, fye, bye, cye, kye,
         psi_ez_x, psi_ez_y, P, Pp, W, pa, pb, pc):
    nx, ny = Ez.shape
    K = pa.shape[0]
    for i in prange(nx):
        for j in range(ny):
            # dHy/dx (backward)
            if i > 0:
                h0 = Hy[i - 1, j]
            else:
                h0 = Hy[nx - 1, j]
            d1 = (Hy[i, j] - h0)/dx
            if fxe[i]:
                psi_ez_x[i, j] = bxe[i]*psi_ez_x[i, j] + cxe[i]*d1
                d1 = d1*kxe[i] + psi_ez_x[i, j]
            # dHx/dy (backward)
            if j > 0:
                h0 = Hx[i, j - 1]
            else:
                h0 = Hx[i, ny - 1]
            d2 = (Hx[i, j] - h0)/dy
            if fye[j]:
                psi_ez_y[i, j] = bye[j]*psi_ez_y[i, j] + cye[j]*d2
                d2 = d2*kye[j] + psi_ez_y[i, j]
            curl = d1 - d2
            e = Ez[i, j]
            dP = 0.0
            for k in range(K):
                w = W[k, i, j]
                if w == 0.0:
                    continue
                p = P[k, i, j]
                pn = pa[k]*p + pb[k]*Pp[k, i, j] + pc[k]*w*e
                dP += pn - p
                Pp[k, i, j] = p
                P[k, i, j] = pn
            Ez[i, j] = Ca[i, j]*e + Cb[i, j]*(curl - dP/dt)


@_jit
def h_hz(Ex, Ey, Hz, dt, dx, dy, px, py, fxh, bxh, cxh, kxh, fyh, byh, cyh, kyh, psi_hz_y, psi_hz_x):
    nx, ny = Hz.shape
    for i in prange(nx):
        for j in range(ny):
            # dEx/dy (forward)
            if j < ny - 1:
                e1 = Ex[i, j + 1]
            elif py:
                e1 = Ex[i, 0]
            else:
                e1 = 0.0
            d1 = (e1 - Ex[i, j])/dy
            if fyh[j]:
                psi_hz_y[i, j] = byh[j]*psi_hz_y[i, j] + cyh[j]*d1
                d1 = d1*kyh[j] + psi_hz_y[i, j]
            # dEy/dx (forward)
            if i < nx - 1:
                e1 = Ey[i + 1, j]
            elif px:
                e1 = Ey[0, j]
            else:
                e1 = 0.0
            d2 = (e1 - Ey[i, j])/dx
            if fxh[i]:
                psi_hz_x[i, j] = bxh[i]*psi_hz_x[i, j] + cxh[i]*d2
                d2 = d2*kxh[i] + psi_hz_x[i, j]
            Hz[i, j] += dt*(d1 - d2)


@_jit
def e_hz(Ex, Ey, Hz, CaX, CbX, CaY, CbY, dt, dx, dy, px, py, fxe, bxe, cxe, kxe, fye, bye, cye, kye,
         psi_ex_y, psi_ey_x, PX, PpX, WX, PY, PpY, WY, pa, pb, pc):
    nx, ny = Hz.shape
    K = pa.shape[0]
    for i in prange(nx):
        for j in range(ny):
            # Ex: dHz/dy (backward)
            if j > 0:
                h0 = Hz[i, j - 1]
            else:
                h0 = Hz[i, ny - 1]
            d = (Hz[i, j] - h0)/dy
            if fye[j]:
                psi_ex_y[i, j] = bye[j]*psi_ex_y[i, j] + cye[j]*d
                d = d*kye[j] + psi_ex_y[i, j]
            e = Ex[i, j]
            dP = 0.0
            for k in range(K):
                w = WX[k, i, j]
                if w == 0.0:
                    continue
                p = PX[k, i, j]
                pn = pa[k]*p + pb[k]*PpX[k, i, j] + pc[k]*w*e
                dP += pn - p
                PpX[k, i, j] = p
                PX[k, i, j] = pn
            Ex[i, j] = CaX[i, j]*e + CbX[i, j]*(d - dP/dt)
            # Ey: -dHz/dx (backward)
            if i > 0:
                h0 = Hz[i - 1, j]
            else:
                h0 = Hz[nx - 1, j]
            d = (Hz[i, j] - h0)/dx
            if fxe[i]:
                psi_ey_x[i, j] = bxe[i]*psi_ey_x[i, j] + cxe[i]*d
                d = d*kxe[i] + psi_ey_x[i, j]
            e = Ey[i, j]
            dP = 0.0
            for k in range(K):
                w = WY[k, i, j]
                if w == 0.0:
                    continue
                p = PY[k, i, j]
                pn = pa[k]*p + pb[k]*PpY[k, i, j] + pc[k]*w*e
                dP += pn - p
                PpY[k, i, j] = p
                PY[k, i, j] = pn
            Ey[i, j] = CaY[i, j]*e + CbY[i, j]*(-d - dP/dt)
