"""Compiled Newton intersection of the Forbes Q surfaces (optional, needs numba).

`qaspheres` intersects rays with Q-bfs, Q-con, Q2D and Q-con + Q2D surfaces
by Newton steps on the full sag. In NumPy every step is a set of array
operations over all rays and every polynomial group, which costs mostly
Python and temporary arrays for the small bundles of ray aiming. Here every
ray is iterated in one compiled loop, with Clenshaw sums for the Chebyshev
series and the same formulas as the NumPy methods.

``available`` tells whether numba could be imported.
"""

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover - depends on the installation
    nb = None

available = nb is not None

# radial part: QBFS (sphere c_bfs plus u^2 (1 - u^2) S / root), QCON (conic
# plus u^4 S), SPHERE (sphere c only, the base of a Q2D surface)
QBFS, QCON, SPHERE = 0, 1, 2
# freeform part: none, added to the sag (Q-con + Q2D), divided by the base
# sphere root (Q2D surface)
NONE, ADDED, OVER_ROOT = 0, 1, 2


def _clenshaw(t, a, n):
    """Chebyshev series a[:n] at t."""

    b1 = 0.0
    b2 = 0.0
    for k in range(n - 1, 0, -1):
        b1, b2 = 2.0*t*b1 - b2 + a[k], b1
    return t*b1 - b2 + a[0]


def _radial(kind, c, K, rn2, a, na, da, r2):
    """Sag and (dz/dr)/r of the rotationally symmetric part."""

    if kind == QCON:
        arg = 1.0 - (1.0 + K)*c*c*r2
    else:
        arg = 1.0 - c*c*r2
    if arg < 0.0:
        return np.nan, np.nan
    root = np.sqrt(arg)
    z = c*r2/(1.0 + root)
    s = c/root
    if kind == SPHERE or na == 0:
        return z, s
    x = r2/rn2
    t = 2.0*x - 1.0
    S = _clenshaw(t, a, na)
    dS = 2.0*_clenshaw(t, da, max(na - 1, 1)) if na > 1 else 0.0
    if kind == QCON:
        return z + x*x*S, s + 2.0*(2.0*x*S + x*x*dS)/rn2
    # Q-bfs: c is c_bfs
    dD = ((1.0 - 2.0*x)*S + x*(1.0 - x)*dS)/rn2/root + x*(1.0 - x)*S*c*c/(2.0*root**3)
    return z + x*(1.0 - x)*S/root, s + 2.0*dD


def _q2d(x, y, rf, ms, kinds, A, dA, lens):
    """Q2D departure D and (dD/dx, dD/dy) at (x, y) relative to its centre;
    groups sorted by m."""

    wr = x/rf
    wi = y/rf
    X = wr*wr + wi*wi
    t = 2.0*X - 1.0
    D = 0.0
    Dx = 0.0
    Dy = 0.0
    pr, pi = 1.0, 0.0      # w^m
    qr, qi = 0.0, 0.0      # w^(m - 1)
    mc = 0
    for g in range(ms.shape[0]):
        m = ms[g]
        while mc < m:
            qr, qi = pr, pi
            pr, pi = pr*wr - pi*wi, pr*wi + pi*wr
            mc += 1
        n = lens[g]
        P = _clenshaw(t, A[g], n)
        dP = 2.0*_clenshaw(t, dA[g], n - 1) if n > 1 else 0.0    # dP/dX
        if m == 0:
            f = X*(1.0 - X)*P
            df = (1.0 - 2.0*X)*P + X*(1.0 - X)*dP
            D += f
            Dx += df*2.0*wr/rf
            Dy += df*2.0*wi/rf
            continue
        if kinds[g] == 0:      # cos: Re w^m
            Av, Ax, Ay = pr, m*qr, -m*qi
        else:                  # sin: Im w^m
            Av, Ax, Ay = pi, m*qi, m*qr
        D += P*Av
        Dx += dP*2.0*wr/rf*Av + P*Ax/rf
        Dy += dP*2.0*wi/rf*Av + P*Ay/rf
    return D, Dx, Dy


def _sag_gradient(kind, c, K, rn2, a, na, da, free, rf, x0, y0, ms, kinds, A, dA, lens, x, y):
    r2 = x*x + y*y
    z, s = _radial(kind, c, K, rn2, a, na, da, r2)
    gx = x*s
    gy = y*s
    if free == ADDED:
        D, Dx, Dy = _q2d(x - x0, y - y0, rf, ms, kinds, A, dA, lens)
        z += D
        gx += Dx
        gy += Dy
    elif free == OVER_ROOT:
        D, Dx, Dy = _q2d(x, y, rf, ms, kinds, A, dA, lens)
        arg = 1.0 - c*c*r2
        if arg < 0.0:
            return np.nan, np.nan, np.nan
        root = np.sqrt(arg)
        k = D*c*c/root**3
        z += D/root
        gx += Dx/root + k*x
        gy += Dy/root + k*y
    return z, gx, gy


def _intersect(P, d, t0, kind, c, K, rn2, a, na, da, free, rf, x0, y0, ms, kinds, A, dA, lens,
               tol):
    """Newton steps on the full sag from t0 (NaN: from the vertex plane);
    the rays where the sag is undefined at the end are NaN."""

    N = P.shape[0]
    out = np.empty(N)
    for i in range(N):
        t = t0[i]
        if not np.isfinite(t):
            t = -P[i, 2]/d[i, 2]
        for _ in range(60):
            qx = P[i, 0] + t*d[i, 0]
            qy = P[i, 1] + t*d[i, 1]
            qz = P[i, 2] + t*d[i, 2]
            z, gx, gy = _sag_gradient(kind, c, K, rn2, a, na, da, free, rf, x0, y0,
                                      ms, kinds, A, dA, lens, qx, qy)
            step = (qz - z)/(d[i, 2] - gx*d[i, 0] - gy*d[i, 1])
            if not np.isfinite(step):
                break
            t -= step
            if abs(step) < tol*(1.0 + abs(t)):
                break
        qx = P[i, 0] + t*d[i, 0]
        qy = P[i, 1] + t*d[i, 1]
        z, gx, gy = _sag_gradient(kind, c, K, rn2, a, na, da, free, rf, x0, y0,
                                  ms, kinds, A, dA, lens, qx, qy)
        out[i] = t if np.isfinite(z) else np.nan
    return out


if available:
    _options = dict(cache=True, error_model="numpy")
    _clenshaw = nb.njit(**_options)(_clenshaw)
    _radial = nb.njit(**_options)(_radial)
    _q2d = nb.njit(**_options)(_q2d)
    _sag_gradient = nb.njit(**_options)(_sag_gradient)
    # serial: the bundles of ray aiming are small, and optimizers run in parallel processes
    _kernel = nb.njit(**_options)(_intersect)
else:  # pragma: no cover
    _kernel = None


_EMPTY_I = np.zeros(0, np.int64)
_EMPTY_2 = np.zeros((0, 1))


def pack_groups(groups):
    """Arrays of the Q2D groups {(kind, m): Chebyshev series} for the kernel,
    sorted by m: ms, kinds (0 cos, 1 sin), series A and derivative series
    dA (rows padded with zeros), lengths."""

    keys = sorted(groups, key=lambda k: (k[1], k[0]))
    if not keys:
        return _EMPTY_I, _EMPTY_I, _EMPTY_2, _EMPTY_2, _EMPTY_I
    L = max(len(groups[k]) for k in keys)
    A = np.zeros((len(keys), L))
    dA = np.zeros((len(keys), max(L - 1, 1)))
    lens = np.zeros(len(keys), np.int64)
    for j, k in enumerate(keys):
        coef = np.asarray(groups[k], float)
        A[j, :len(coef)] = coef
        lens[j] = len(coef)
        if len(coef) > 1:
            dA[j, :len(coef) - 1] = np.polynomial.chebyshev.chebder(coef)
    ms = np.array([k[1] for k in keys], np.int64)
    kinds = np.array([0 if k[0] == "cos" else 1 for k in keys], np.int64)
    return ms, kinds, A, dA, lens


def series(a):
    """Series and derivative series (radial part) for the kernel."""

    a = np.ascontiguousarray(a, float)
    da = np.polynomial.chebyshev.chebder(a) if len(a) > 1 else np.zeros(1)
    return a, len(a), np.ascontiguousarray(da, float)


def intersect(P, d, t0, radial, c, K, rn2, a, free=NONE, rf=1.0, centre=(0.0, 0.0),
              groups=None, tol=1e-14):
    """Distances along the rays (P, d) to the surface, from the start t0.

    radial : QBFS, QCON or SPHERE; c, K, rn2 and the series a (from
    `series()`) describe it. free : NONE, ADDED or OVER_ROOT, with the
    normalization radius rf, the centre and the packed groups (from
    `pack_groups()`) of the Q2D part."""

    a, na, da = a
    ms, kinds, A, dA, lens = groups if groups is not None else pack_groups({})
    return _kernel(np.ascontiguousarray(P, float), np.ascontiguousarray(d, float),
                   np.ascontiguousarray(t0, float), radial, float(c), float(K), float(rn2),
                   a, na, da, free, float(rf), float(centre[0]), float(centre[1]),
                   ms, kinds, A, dA, lens, float(tol))
