#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Carpet cloaks from quasi-conformal maps (transformation optics in 2D).

A map w = (u, v)(x, y) from the physical plane to a "virtual" plane turns
vacuum in the virtual plane into a medium in the physical plane that
reproduces the virtual fields exactly (Ward and Pendry 1996, Pendry,
Schurig and Smith 2006). With the Jacobian K = d(u, v)/d(x, y) and
J = K^-1, for E along z (``mode="Ez"``) ::

    eps_z = 1/det(J) = det(K),    mu (in-plane) = J J^T/det(J).

If the map is conformal (u + i v analytic), J J^T = det(J) I and the
medium is isotropic and non-magnetic: n**2 = det(K) = abs(dw/dz)**2. For a
carpet cloak (Li and Pendry 2008) the physical region lies above a bump
y = h(x) on a mirror, inside the box abs(x) < a, y < H; the virtual
region is the rectangle above a flat mirror. A conformal map between the
two does not exist in general (the conformal modules differ), but with
"sliding" boundaries (points may move along the edges) the map that is
harmonic in the physical plane is almost conformal: the in-plane
anisotropy is second order in the bump height (the first order term is
exactly conformal, see `CarpetCloak.linear_n2`). Dropping it leaves an
isotropic, all-dielectric graded index n(x, y), which an observer above
the mirror sees as a flat mirror. Where the map stretches (the feet of the
bump) n < 1.

`CarpetCloak` solves the two Laplace problems

- v: v = 0 on the bump, v = H on the top, dv/dn = 0 on the sides
  (sliding),
- u: u = -a, a on the sides, du/dn = 0 on the bump and the top (sliding),

with second-order finite differences on a boundary-fitted grid
(xi, eta) = (x, (y - h(x))/(H - h(x))), and gives n**2, the anisotropy
alpha = sigma_1/sigma_2 >= 1 (the ratio of the principal indices of the
full anisotropic medium; the singular values of K), the virtual
coordinates, and an index field for `pyoptics.grin` (rays) and
`pyoptics.fdfd` (waves). ``boundary="fixed"`` instead pins all boundary
points (u = x, v = y on the edges): no sliding, a strongly anisotropic map.

Functions
---------
- cos2_bump(), gaussian_bump(): bump profiles.
- beam_sources(): a line of `fdfd.PointSource` that launches a Gaussian
  beam in one direction (a current sheet with a precompensated angular
  spectrum).
- angular_spectrum(): plane wave amplitudes of a field on a line.

References
----------
J. Li and J. B. Pendry, "Hiding under the carpet: a new strategy for
cloaking", Phys. Rev. Lett. 101, 203901 (2008).

J. B. Pendry, D. Schurig and D. R. Smith, "Controlling electromagnetic
fields", Science 312, 1780-1782 (2006).

A. J. Ward and J. B. Pendry, "Refraction and geometry in Maxwell's
equations", J. Mod. Opt. 43, 773-793 (1996).

U. Leonhardt, "Optical conformal mapping", Science 312, 1777-1780 (2006).

J. F. Thompson, Z. U. A. Warsi and C. W. Mastin, "Numerical Grid
Generation: Foundations and Applications", North-Holland (1985), chapter 5
(elliptic grid generation).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RectBivariateSpline

from .grin import IndexField


__all__ = ["cos2_bump", "gaussian_bump", "CarpetCloak", "beam_sources", "angular_spectrum"]


def cos2_bump(height, half_width):
    """h(x) = height cos(pi x/(2 half_width))**2 for abs(x) < half_width,
    0 outside (continuous slope, compact support)."""

    def h(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.abs(x) < half_width, height*np.cos(np.pi*x/(2*half_width))**2, 0.0)
    return h


def gaussian_bump(height, width):
    """h(x) = height exp(-x**2/width**2)."""

    return lambda x: height*np.exp(-np.asarray(x, dtype=float)**2/width**2)


def _derivs(h, x, d=1e-4):
    h0 = h(x)
    return h0, (h(x + d) - h(x - d))/(2*d), (h(x + d) - 2*h0 + h(x - d))/d**2


class CarpetCloak(IndexField):
    """Quasi-conformal carpet cloak above a bump on a mirror.

    Parameters
    ----------
    bump : callable
        Height h(x) of the mirror (0 near abs(x) = a, below H).
    half_width : number
        a: the cloak fills abs(x) < a, h(x) < y < H.
    height : number
        H.
    n_xi, n_eta : int, optional
        Grid points along x and across (from the bump to the top).
    boundary : {"sliding", "fixed"}
        Boundary conditions of the map (see the module docstring).
    n_min : number, optional
        Clip the index from below (e.g. 1 for a cloak without n < 1).

    Attributes
    ----------
    xi, eta : 1D arrays
        The computational grid.
    x, y : 2D arrays
        Its physical points.
    u, v : 2D arrays
        Virtual coordinates there.
    n2_grid, alpha : 2D arrays
        det(K) (unclipped) and the anisotropy sigma_1/sigma_2.
    """

    def __init__(self, bump, half_width, height, n_xi=601, n_eta=121, boundary="sliding", n_min=0.0):
        if boundary not in ("sliding", "mirror", "fixed"):
            raise ValueError("boundary must be 'sliding', 'mirror' or 'fixed'")
        self.bump, self.a, self.H = bump, float(half_width), float(height)
        self.boundary, self.n_min = boundary, float(n_min)
        self.h = 1e-6
        self.xi = np.linspace(-self.a, self.a, n_xi)
        self.eta = np.linspace(0.0, 1.0, n_eta)
        XI, ETA = np.meshgrid(self.xi, self.eta, indexing="ij")
        hh, h1, h2 = _derivs(bump, XI)
        D = self.H - hh
        self._D = D
        self.x, self.y = XI, hh + ETA*D
        # metric of eta = (y - h)/D:  eta_x, eta_y = 1/D, eta_xx (eta_yy = 0)
        ex = h1*(ETA - 1)/D
        exx = (h2*(ETA - 1) + h1*ex)/D + h1*(ETA - 1)*h1/D**2
        self._ex, self._ey = ex, 1/D
        coef = dict(fxx=np.ones_like(XI), fxe=2*ex, fee=ex**2 + 1/D**2, fe=exx)
        self.u, self.v = self._solve(coef, h1, D)
        K = self.jacobian()
        self.n2_grid = K[0, 0]*K[1, 1] - K[0, 1]*K[1, 0]
        fro = np.sum(K**2, axis=(0, 1))
        det = np.abs(self.n2_grid)
        s1 = np.sqrt(np.maximum(fro/2 + np.sqrt(np.maximum(fro**2/4 - det**2, 0)), 0))
        self.alpha = s1**2/det
        n2c = np.maximum(self.n2_grid, self.n_min**2)
        self._spline = RectBivariateSpline(self.xi, self.eta, n2c, kx=3, ky=3)

    # the two Laplace problems

    def _solve(self, c, h1, D):
        nx, ne = len(self.xi), len(self.eta)
        dx, de = self.xi[1] - self.xi[0], self.eta[1] - self.eta[0]
        idx = np.arange(nx*ne).reshape(nx, ne)
        I, Jn = np.meshgrid(np.arange(nx), np.arange(ne), indexing="ij")
        side = (I == 0) | (I == nx - 1)
        bottom, top = Jn == 0, Jn == ne - 1

        def assemble(kind):
            rows, cols, vals = [], [], []
            b = np.zeros(nx*ne)
            done = np.zeros((nx, ne), dtype=bool)

            def put(mask, di, dj, w):
                ii, jj = I[mask], Jn[mask]
                rows.append(idx[ii, jj]); cols.append(idx[ii + di, jj + dj])
                vals.append(np.broadcast_to(w, ii.shape) if np.ndim(w) == 0 else w[mask])

            def dirichlet(mask, value):
                m = mask & ~done
                put(m, 0, 0, 1.0)
                b[idx[m]] = np.broadcast_to(value, (nx, ne))[m]
                done[m] = True

            def onesided(mask, axis, sign):
                # second-order one-sided first derivative into the domain (zero normal derivative)
                m = mask & ~done
                e = (sign, 0) if axis == 0 else (0, sign)
                put(m, 0, 0, -3.0); put(m, *e, 4.0); put(m, 2*e[0], 2*e[1], -1.0)
                done[m] = True
                return m

            if kind == "v":
                dirichlet(bottom, 0.0)
                dirichlet(top, self.H)
                if self.boundary == "sliding":       # dv/dx = 0 (h' = 0 at the sides)
                    onesided(side & (I == 0), 0, 1); onesided(side & (I == nx - 1), 0, -1)
                else:
                    dirichlet(side, self.y)
            else:
                dirichlet(side, np.sign(self.x)*self.a)
                if self.boundary == "fixed":
                    dirichlet(bottom | top, self.x)
                else:
                    if self.boundary == "mirror":
                        dirichlet(top, self.x)
                    else:                             # du/dy = u_eta/D = 0
                        onesided(top, 1, -1)
                    # -h' u_x + u_y = 0  <=>  u_eta = h' D u_xi/(1 + h'^2)
                    m = onesided(bottom, 1, 1)
                    q = h1*D/(1 + h1**2)*de/dx
                    put(m, 1, 0, -q); put(m, -1, 0, q)
            # interior: f_xixi + 2 eta_x f_xieta + (eta_x^2 + eta_y^2) f_etaeta + eta_xx f_eta = 0
            m = ~done
            a_, ce, d_ = c["fxx"]/dx**2, c["fee"]/de**2, c["fe"]/(2*de)
            w = c["fxe"]/(4*dx*de)
            put(m, -1, 0, a_); put(m, 1, 0, a_)
            put(m, 0, -1, ce - d_); put(m, 0, 1, ce + d_)
            put(m, 0, 0, -2*a_ - 2*ce)
            put(m, 1, 1, w); put(m, -1, -1, w); put(m, 1, -1, -w); put(m, -1, 1, -w)
            A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(nx*ne, nx*ne))
            return spla.spsolve(A, b).reshape(nx, ne)

        return assemble("u"), assemble("v")

    def jacobian(self):
        """K = d(u, v)/d(x, y) on the grid, shape (2, 2, n_xi, n_eta)."""

        out = np.empty((2, 2) + self.u.shape)
        for k, f in enumerate((self.u, self.v)):
            fxi, feta = np.gradient(f, self.xi, self.eta, edge_order=2)
            out[k, 0] = fxi + self._ex*feta
            out[k, 1] = self._ey*feta
        return out

    # geometry

    def inside(self, x, y):
        """True in the cloak (abs(x) < a, h(x) <= y <= H)."""

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return (np.abs(x) < self.a) & (y >= self.bump(x)) & (y <= self.H)

    def _xe(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        xc = np.clip(x, -self.a, self.a)
        hh = self.bump(xc)
        return xc, np.clip((y - hh)/(self.H - hh), 0.0, 1.0)

    def virtual(self, x, y):
        """Virtual coordinates (u, v) of physical points (identity outside)."""

        xc, e = self._xe(x, y)
        su = RectBivariateSpline(self.xi, self.eta, self.u, kx=3, ky=3)
        sv = RectBivariateSpline(self.xi, self.eta, self.v, kx=3, ky=3)
        ins = self.inside(x, y)
        return (np.where(ins, su(xc, e, grid=False), x), np.where(ins, sv(xc, e, grid=False), y))

    def permittivity(self, x, y):
        """n**2 at physical points (1 outside the cloak and below the
        bump), clipped at n_min**2; for `fdfd.Simulation(permittivity=...)`."""

        xc, e = self._xe(x, y)
        return np.where(self.inside(x, y), self._spline(xc, e, grid=False), 1.0)

    def n2(self, r):
        r = np.asarray(r, dtype=float)
        return self.permittivity(r[..., 0], r[..., 1])

    def n(self, r):
        return np.sqrt(self.n2(r))

    def force(self, r):
        """grad(n**2)/2 (chain rule through the boundary-fitted grid)."""

        r = np.asarray(r, dtype=float)
        x, y = r[..., 0], r[..., 1]
        xc, e = self._xe(x, y)
        hh, h1, _ = _derivs(self.bump, xc)
        D = self.H - hh
        fx = self._spline(xc, e, dx=1, grid=False)
        fe = self._spline(xc, e, dy=1, grid=False)
        ins = self.inside(x, y)
        out = np.zeros_like(r)
        out[..., 0] = np.where(ins, 0.5*(fx + h1*(e - 1)/D*fe), 0.0)
        out[..., 1] = np.where(ins, 0.5*fe/D, 0.0)
        return out

    def linear_n2(self, x, y, period=None, n=4096):
        """First order theory in the bump height (an infinite strip
        0 < y < H): with v = y - phi, u = x - chi, phi = h at y = 0 and
        chi_y = -h' there, phi and chi are harmonic conjugates and ::

            n**2 = 1 + 2 sum_k abs(k) h_k cosh(k (H - y))/sinh(abs(k) H) exp(i k x)

        (h_k: Fourier coefficients of h over a period, default 8 a)."""

        L = 8*self.a if period is None else period
        xs = (np.arange(n) - n//2)*L/n
        hk = np.fft.fft(np.fft.ifftshift(self.bump(xs)))/n
        k = 2*np.pi*np.fft.fftfreq(n, L/n)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        kk, ck = np.abs(k[1:]), hk[1:]
        xf, yf = x.ravel(), y.ravel()
        out = np.empty(xf.shape)
        for s in range(0, len(xf), 256):
            yy = yf[s:s + 256, None]      # cosh(k(H - y))/sinh(kH) without overflow
            kern = kk*(np.exp(-kk*yy) + np.exp(-kk*(2*self.H - yy)))/(1 - np.exp(-2*kk*self.H))
            out[s:s + 256] = np.real(np.sum(ck*kern*np.exp(1j*k[1:]*xf[s:s + 256, None]), axis=-1))
        return 1 + 2*out.reshape(x.shape)

    def trace(self, x_start, y_start, angle, y_end, step=0.01):
        """Rays through the isotropic cloak (pyoptics.grin) onto the mirror
        and back: straight lines outside, Snell refraction at the top y = H
        (the index jumps there unless the map is exact), reflection on the
        bump. Rays start at (x_start, y_start) above the cloak, at `angle`
        from the -y axis (towards +x), and end on the line y = y_end.

        Returns (x_end, exit_angle, path_in), the end points, the angles of
        the outgoing rays from the +y axis (towards +x) and the paths inside
        the cloak (a list of arrays (n_points, 2) per ray).
        """

        from .grin import trace as grin_trace

        x0 = np.atleast_1d(np.asarray(x_start, dtype=float))
        s, c = np.sin(angle), np.cos(angle)
        xH = x0 + (y_start - self.H)*s/c
        R = np.c_[xH, np.full_like(xH, self.H*(1 - 1e-12))]
        nin = self.n(R)
        d = np.c_[np.full_like(xH, s), -np.sqrt(nin**2 - s**2)]      # T_x = sin(angle) conserved
        legs = []
        hit = grin_trace(self, R, d, step, stop=lambda r: r[:, 1] - self.bump(r[:, 0]))
        legs.append(hit.path)
        Rm = hit.position
        hh, h1, _ = _derivs(self.bump, Rm[:, 0])
        nrm = np.c_[-h1, np.ones_like(h1)]/np.sqrt(1 + h1**2)[:, None]
        T = hit.T - 2*np.sum(hit.T*nrm, axis=1)[:, None]*nrm
        Rm = Rm + 1e-9*nrm
        out = grin_trace(self, Rm, T, step, stop=lambda r: r[:, 1] - self.H)
        legs.append(out.path)
        tx = out.T[:, 0]                                                # = sin(exit angle) outside
        ex = np.arcsin(np.clip(tx, -1, 1))
        x_end = out.position[:, 0] + (y_end - self.H)*np.tan(ex)
        paths = [np.concatenate([legs[0][:, k], legs[1][:, k]]) for k in range(len(x0))]
        return x_end, ex, paths

    def boundary_mismatch(self):
        """Largest sliding of the outer boundary: max abs(u - x) on the top
        and max abs(v - y) on the sides (zero for a perfect cloak)."""

        return (np.abs(self.u[:, -1] - self.xi).max(),
                max(np.abs(self.v[0] - self.y[0]).max(), np.abs(self.v[-1] - self.y[-1]).max()))


def beam_sources(x, y_source, wavelength, angle, waist, focus=(0.0, 0.0), n_bg=1.0, downward=True,
                 spacing=None):
    """Point sources on the grid nodes x of a horizontal line that launch a
    Gaussian beam towards -y (or +y): a current sheet K(x) with
    K_k = -2 k_y u_k/omega, so that the wave it sends towards the mirror has
    the angular spectrum u_k of a Gaussian beam with the given waist, focused
    at `focus`, at `angle` from the -y axis (positive: towards +x). The
    sheet also radiates the other way; that wave leaves through the PML.

    With the grid `spacing` (the same along x and y) the wave numbers k_y
    and the sheet's Green's function are those of the grid (the discrete
    dispersion of the five-point Laplacian), so that u_inc is the incident
    field the FDFD actually computes; otherwise those of the continuum.

    Returns a list of fdfd.PointSource, and the incident field
    u_inc(X, Y) as a function (angular spectrum, for comparison).
    """

    from .fdfd import PointSource

    x = np.asarray(x, dtype=float)
    k = 2*np.pi*n_bg/wavelength
    omega = 2*np.pi/wavelength
    hx = x[1] - x[0]
    npad = 4*len(x)
    kx = 2*np.pi*np.fft.fftfreq(npad, hx)
    prop = np.abs(kx) < k
    if spacing is None:
        ky = np.sqrt(np.maximum(k**2 - kx**2, 0))
        gy = 2*ky                                   # sheet: u_k = -omega K_k/(2 k_y)
    else:
        hh = float(spacing)
        q = np.maximum(k**2 - (2/hh*np.sin(kx*hh/2))**2, 0)
        ky = 2/hh*np.arcsin(np.minimum(hh/2*np.sqrt(q), 1))
        gy = 2/hh*np.sin(ky*hh)
    th = np.arcsin(np.clip(kx/k, -1, 1))
    spec = np.where(prop, np.exp(-((th - angle)*k*waist/2)**2), 0.0)
    s = -1.0 if downward else 1.0

    def u_inc(X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        ph = np.exp(1j*(kx[prop]*(X[..., None] - focus[0]) + s*ky[prop]*(Y[..., None] - focus[1])))
        return np.sum(spec[prop]*ph, axis=-1)*(kx[1] - kx[0])/(2*np.pi)

    # wanted field on the source line, spectrum in exp(i kx x)
    uk = spec*np.exp(-1j*kx*focus[0] + 1j*s*ky*(y_source - focus[1]))
    Kk = np.where(prop, -gy*uk/omega, 0.0)
    K = np.array([np.sum(Kk*np.exp(1j*kx*xx)) for xx in x])*(kx[1] - kx[0])/(2*np.pi)
    keep = np.abs(K) > 1e-6*np.abs(K).max()
    return [PointSource((xx, y_source), KK*hx) for xx, KK in zip(x[keep], K[keep])], u_inc


def angular_spectrum(u, x, wavelength, n_pad=8, n_bg=1.0):
    """Plane wave amplitudes of a field u(x) on a line: angles
    arcsin(kx/k) in radians and the spectrum (FFT, zero padded), for the
    propagating part."""

    u = np.asarray(u)
    hx = x[1] - x[0]
    N = n_pad*len(u)
    U = np.fft.fftshift(np.fft.fft(u, N))*hx
    kx = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, hx))
    U = U*np.exp(-1j*kx*x[0])
    k = 2*np.pi*n_bg/wavelength
    m = np.abs(kx) < k
    return np.arcsin(kx[m]/k), U[m]
