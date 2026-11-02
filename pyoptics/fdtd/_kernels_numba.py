#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Compiled update kernels (optional, need numba).

Each kernel updates all H or all E components of a mode in one parallel
loop over the grid, including the CPML (psi arrays of the full grid size,
touched only where the per-axis flag is set), periodic, Bloch-periodic
(phase factors ph = exp(i k L) at the far end, conj(ph) at the near end;
1 for plain periodicity) or PEC ends and the dispersive polarization
densities. With Bloch phases the fields are complex. They follow the NumPy updates in
simulation.py exactly; 1D runs as a 2D Ez grid with one periodic cell
along y. ``available`` tells whether numba could be imported.

The 3D kernels keep the CPML auxiliary fields only inside the absorbing
layers: psi along axis a has 2n cells along a (n per side), and the map
m[i] gives the slab index of grid index i (-1 outside the layers).
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
def h_ez(Ez, Hx, Hy, dt, dx, dy, px, py, phx, phy, fxh, bxh, cxh, kxh, fyh, byh, cyh, kyh, psi_hx_y, psi_hy_x):
    nx, ny = Ez.shape
    for i in prange(nx):
        for j in range(ny):
            e0 = Ez[i, j]
            # Hx -= dt dEz/dy (forward)
            if j < ny - 1:
                e1 = Ez[i, j + 1]
            elif py:
                e1 = Ez[i, 0]*phy
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
                e1 = Ez[0, j]*phx
            else:
                e1 = 0.0
            d = (e1 - e0)/dx
            if fxh[i]:
                psi_hy_x[i, j] = bxh[i]*psi_hy_x[i, j] + cxh[i]*d
                d = d*kxh[i] + psi_hy_x[i, j]
            Hy[i, j] += dt*d


@_jit
def e_ez(Ez, Hx, Hy, Ca, Cb, dt, dx, dy, px, py, cphx, cphy, fxe, bxe, cxe, kxe, fye, bye, cye, kye,
         psi_ez_x, psi_ez_y, P, Pp, W, pa, pb, pc):
    nx, ny = Ez.shape
    K = pa.shape[0]
    for i in prange(nx):
        for j in range(ny):
            # dHy/dx (backward)
            if i > 0:
                h0 = Hy[i - 1, j]
            else:
                h0 = Hy[nx - 1, j]*cphx
            d1 = (Hy[i, j] - h0)/dx
            if fxe[i]:
                psi_ez_x[i, j] = bxe[i]*psi_ez_x[i, j] + cxe[i]*d1
                d1 = d1*kxe[i] + psi_ez_x[i, j]
            # dHx/dy (backward)
            if j > 0:
                h0 = Hx[i, j - 1]
            else:
                h0 = Hx[i, ny - 1]*cphy
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
def h_hz(Ex, Ey, Hz, dt, dx, dy, px, py, phx, phy, fxh, bxh, cxh, kxh, fyh, byh, cyh, kyh, psi_hz_y, psi_hz_x):
    nx, ny = Hz.shape
    for i in prange(nx):
        for j in range(ny):
            # dEx/dy (forward)
            if j < ny - 1:
                e1 = Ex[i, j + 1]
            elif py:
                e1 = Ex[i, 0]*phy
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
                e1 = Ey[0, j]*phx
            else:
                e1 = 0.0
            d2 = (e1 - Ey[i, j])/dx
            if fxh[i]:
                psi_hz_x[i, j] = bxh[i]*psi_hz_x[i, j] + cxh[i]*d2
                d2 = d2*kxh[i] + psi_hz_x[i, j]
            Hz[i, j] += dt*(d1 - d2)


@_jit
def e_hz(Ex, Ey, Hz, CaX, CbX, CaY, CbY, dt, dx, dy, px, py, cphx, cphy, fxe, bxe, cxe, kxe, fye, bye, cye, kye,
         psi_ex_y, psi_ey_x, PX, PpX, WX, PY, PpY, WY, pa, pb, pc):
    nx, ny = Hz.shape
    K = pa.shape[0]
    for i in prange(nx):
        for j in range(ny):
            # Ex: dHz/dy (backward)
            if j > 0:
                h0 = Hz[i, j - 1]
            else:
                h0 = Hz[i, ny - 1]*cphy
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
                h0 = Hz[nx - 1, j]*cphx
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


def _jit_serial(f):
    if nb is None:
        return f
    return nb.njit(cache=True, fastmath=False)(f)


@_jit_serial
def _poles(P, Pp, W, pa, pb, pc, i, j, k, e):
    """Advance the polarization densities at a node; returns their change."""

    dP = 0.0
    for q in range(pa.shape[0]):
        w = W[q, i, j, k]
        if w == 0.0:
            continue
        p = P[q, i, j, k]
        pn = pa[q]*p + pb[q]*Pp[q, i, j, k] + pc[q]*w*e
        dP += pn - p
        Pp[q, i, j, k] = p
        P[q, i, j, k] = pn
    return dP


@_jit
def h_3d(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz, px, py, pz, phx, phy, phz,
         mx, bx, cx, kx, my, by, cy, ky, mz, bz, cz, kz,
         psi_hx_y, psi_hx_z, psi_hy_x, psi_hy_z, psi_hz_x, psi_hz_y):
    nx, ny, nz = Ex.shape
    for i in prange(nx):
        ia = mx[i]
        for j in range(ny):
            ja = my[j]
            for k in range(nz):
                ka = mz[k]
                ex = Ex[i, j, k]
                ey = Ey[i, j, k]
                ez = Ez[i, j, k]
                # forward neighbours (wrapped, or the zero tangential E on the far wall)
                if i < nx - 1:
                    ey_x = Ey[i + 1, j, k]
                    ez_x = Ez[i + 1, j, k]
                elif px:
                    ey_x = Ey[0, j, k]*phx
                    ez_x = Ez[0, j, k]*phx
                else:
                    ey_x = 0.0*ey
                    ez_x = 0.0*ez
                if j < ny - 1:
                    ex_y = Ex[i, j + 1, k]
                    ez_y = Ez[i, j + 1, k]
                elif py:
                    ex_y = Ex[i, 0, k]*phy
                    ez_y = Ez[i, 0, k]*phy
                else:
                    ex_y = 0.0*ex
                    ez_y = 0.0*ez
                if k < nz - 1:
                    ex_z = Ex[i, j, k + 1]
                    ey_z = Ey[i, j, k + 1]
                elif pz:
                    ex_z = Ex[i, j, 0]*phz
                    ey_z = Ey[i, j, 0]*phz
                else:
                    ex_z = 0.0*ex
                    ey_z = 0.0*ey
                # Hx += dt (dEy/dz - dEz/dy)
                d1 = (ey_z - ey)/dz
                if ka >= 0:
                    psi_hx_z[i, j, ka] = bz[k]*psi_hx_z[i, j, ka] + cz[k]*d1
                    d1 = d1*kz[k] + psi_hx_z[i, j, ka]
                d2 = (ez_y - ez)/dy
                if ja >= 0:
                    psi_hx_y[i, ja, k] = by[j]*psi_hx_y[i, ja, k] + cy[j]*d2
                    d2 = d2*ky[j] + psi_hx_y[i, ja, k]
                Hx[i, j, k] += dt*(d1 - d2)
                # Hy += dt (dEz/dx - dEx/dz)
                d1 = (ez_x - ez)/dx
                if ia >= 0:
                    psi_hy_x[ia, j, k] = bx[i]*psi_hy_x[ia, j, k] + cx[i]*d1
                    d1 = d1*kx[i] + psi_hy_x[ia, j, k]
                d2 = (ex_z - ex)/dz
                if ka >= 0:
                    psi_hy_z[i, j, ka] = bz[k]*psi_hy_z[i, j, ka] + cz[k]*d2
                    d2 = d2*kz[k] + psi_hy_z[i, j, ka]
                Hy[i, j, k] += dt*(d1 - d2)
                # Hz += dt (dEx/dy - dEy/dx)
                d1 = (ex_y - ex)/dy
                if ja >= 0:
                    psi_hz_y[i, ja, k] = by[j]*psi_hz_y[i, ja, k] + cy[j]*d1
                    d1 = d1*ky[j] + psi_hz_y[i, ja, k]
                d2 = (ey_x - ey)/dx
                if ia >= 0:
                    psi_hz_x[ia, j, k] = bx[i]*psi_hz_x[ia, j, k] + cx[i]*d2
                    d2 = d2*kx[i] + psi_hz_x[ia, j, k]
                Hz[i, j, k] += dt*(d1 - d2)


@_jit
def e_3d(Ex, Ey, Ez, Hx, Hy, Hz, CaX, CbX, CaY, CbY, CaZ, CbZ, lossy, dt, dx, dy, dz,
         cphx, cphy, cphz, mx, bx, cx, kx, my, by, cy, ky, mz, bz, cz, kz,
         psi_ex_y, psi_ex_z, psi_ey_x, psi_ey_z, psi_ez_x, psi_ez_y,
         PX, PpX, WX, PY, PpY, WY, PZ, PpZ, WZ, pa, pb, pc):
    nx, ny, nz = Ex.shape
    K = pa.shape[0]
    for i in prange(nx):
        ia = mx[i]
        im = i - 1 if i > 0 else nx - 1
        for j in range(ny):
            ja = my[j]
            jm = j - 1 if j > 0 else ny - 1
            for k in range(nz):
                ka = mz[k]
                km = k - 1 if k > 0 else nz - 1
                hx = Hx[i, j, k]
                hy = Hy[i, j, k]
                hz = Hz[i, j, k]
                # backward neighbours (wrapped with the Bloch phase; on
                # walls the result is overwritten by the boundary condition)
                if i > 0:
                    hy_x = Hy[im, j, k]
                    hz_x = Hz[im, j, k]
                else:
                    hy_x = Hy[im, j, k]*cphx
                    hz_x = Hz[im, j, k]*cphx
                if j > 0:
                    hx_y = Hx[i, jm, k]
                    hz_y = Hz[i, jm, k]
                else:
                    hx_y = Hx[i, jm, k]*cphy
                    hz_y = Hz[i, jm, k]*cphy
                if k > 0:
                    hx_z = Hx[i, j, km]
                    hy_z = Hy[i, j, km]
                else:
                    hx_z = Hx[i, j, km]*cphz
                    hy_z = Hy[i, j, km]*cphz
                # Ex: dHz/dy - dHy/dz
                d1 = (hz - hz_y)/dy
                if ja >= 0:
                    psi_ex_y[i, ja, k] = by[j]*psi_ex_y[i, ja, k] + cy[j]*d1
                    d1 = d1*ky[j] + psi_ex_y[i, ja, k]
                d2 = (hy - hy_z)/dz
                if ka >= 0:
                    psi_ex_z[i, j, ka] = bz[k]*psi_ex_z[i, j, ka] + cz[k]*d2
                    d2 = d2*kz[k] + psi_ex_z[i, j, ka]
                e = Ex[i, j, k]
                dP = _poles(PX, PpX, WX, pa, pb, pc, i, j, k, e) if K else 0.0
                Ex[i, j, k] = (CaX[i, j, k]*e if lossy else e) + CbX[i, j, k]*(d1 - d2 - dP/dt)
                # Ey: dHx/dz - dHz/dx
                d1 = (hx - hx_z)/dz
                if ka >= 0:
                    psi_ey_z[i, j, ka] = bz[k]*psi_ey_z[i, j, ka] + cz[k]*d1
                    d1 = d1*kz[k] + psi_ey_z[i, j, ka]
                d2 = (hz - hz_x)/dx
                if ia >= 0:
                    psi_ey_x[ia, j, k] = bx[i]*psi_ey_x[ia, j, k] + cx[i]*d2
                    d2 = d2*kx[i] + psi_ey_x[ia, j, k]
                e = Ey[i, j, k]
                dP = _poles(PY, PpY, WY, pa, pb, pc, i, j, k, e) if K else 0.0
                Ey[i, j, k] = (CaY[i, j, k]*e if lossy else e) + CbY[i, j, k]*(d1 - d2 - dP/dt)
                # Ez: dHy/dx - dHx/dy
                d1 = (hy - hy_x)/dx
                if ia >= 0:
                    psi_ez_x[ia, j, k] = bx[i]*psi_ez_x[ia, j, k] + cx[i]*d1
                    d1 = d1*kx[i] + psi_ez_x[ia, j, k]
                d2 = (hx - hx_y)/dy
                if ja >= 0:
                    psi_ez_y[i, ja, k] = by[j]*psi_ez_y[i, ja, k] + cy[j]*d2
                    d2 = d2*ky[j] + psi_ez_y[i, ja, k]
                e = Ez[i, j, k]
                dP = _poles(PZ, PpZ, WZ, pa, pb, pc, i, j, k, e) if K else 0.0
                Ez[i, j, k] = (CaZ[i, j, k]*e if lossy else e) + CbZ[i, j, k]*(d1 - d2 - dP/dt)
