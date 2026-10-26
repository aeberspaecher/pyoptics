#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Phase unwrapping in one and two dimensions.

Measured or simulated phases (np.angle() of a field, the output of a phase
shifting algorithm, a reconstructed hologram) are only known modulo 2π: they
are *wrapped* into (-π, π]. Unwrapping adds the multiples of 2π back, under
the assumption that the true phase changes by less than π between
neighbouring pixels.

Two families of 2D algorithms are implemented:

- `unwrap_ls()`: least squares unwrapping (Ghiglia and Romero 1994). The
  unwrapped phase φ is the function whose discrete gradient is closest, in a
  (weighted) least squares sense, to the wrapped differences of the data. The
  unweighted problem is a Poisson equation with Neumann boundary conditions
  that a single DCT solves. Weights (e.g. a modulation or amplitude map) and
  masks are handled by preconditioned conjugate gradients, with the DCT
  solver as preconditioner. Least squares solutions are smooth and robust,
  but in general *not congruent* to the data: wrap(φ) differs from the input
  wherever the data contain residues. `congruent=True` snaps the result to
  the nearest congruent phase.
- `unwrap_quality_guided()`: path following in the order of a reliability
  (quality) map (Herráez et al. 2002). The best pixel pairs are joined first,
  so errors are confined to unreliable regions. The result is congruent by
  construction.

Independent information about the phase avoids path following altogether:

- `unwrap_with_reference()`: snaps the wrapped phase to the nearest
  congruent phase of a coarse but unwrapped estimate, e.g. a phase from the
  transport of intensity equation (pyoptics.TIE), a model or a measurement
  at a longer wavelength. Every pixel is unwrapped on its own, so residues
  and disconnected regions do not matter; the estimate has to be correct to
  better than ±π.
- `unwrap_two_wavelength()`: two-wavelength (synthetic wavelength)
  unwrapping of phases measured at two wavelengths, e.g. in dual-wavelength
  holography or interferometry.

`residues()` locates the phase singularities (vortices, noise points) that
make unwrapping path dependent, `reliability()` computes Herráez's second
difference quality map, and `unwrap_1d()` wraps numpy.unwrap.

Conventions
-----------

Arrays are indexed [y, x] as from np.meshgrid() / utils.grid2d(); the x
direction is the last axis. Pixels outside a mask (or NaN in the input) are
returned as NaN. The piston of the unwrapped phase is arbitrary; all
functions fix it such that the result agrees with the wrapped input modulo 2π
on average (and exactly, for congruent results), with the mean value of the
result in (-π, π].

References
----------

D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
unweighted phase unwrapping that uses fast transforms and iterative methods",
J. Opt. Soc. Am. A 11, 107 (1994).

J. Gass, A. Dakoff and M. K. Kim, "Phase imaging without 2π ambiguity by
multiwavelength digital holography", Opt. Lett. 28, 1141 (2003).

M. A. Herráez, D. R. Burton, M. J. Lalor and M. A. Gdeisat, "Fast
two-dimensional phase-unwrapping algorithm based on sorting by reliability
following a noncontinuous path", Appl. Opt. 41, 7437 (2002).

D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory,
Algorithms, and Software", Wiley (1998).
"""

import numpy as np
from scipy.fft import dctn, idctn


TWO_PI = 2*np.pi


def wrap(phi):
    """Wrap a phase into [-π, π).

    Parameters
    ----------
    phi : array or number
        Phase in radians.

    Returns
    -------
    phi_wrapped : array or number
        phi minus the multiple of 2π that brings it into [-π, π). NaNs stay
        NaN.
    """

    return np.mod(np.asarray(phi) + np.pi, TWO_PI) - np.pi


def unwrap_1d(phase_wrapped, axis=-1, discont=np.pi):
    """Unwrap along one axis (numpy.unwrap()).

    Parameters
    ----------
    phase_wrapped : array
        Wrapped phase.
    axis : int, optional
        Axis along which to unwrap.
    discont : float, optional
        Maximum jump between neighbouring samples that is considered
        physical.

    Returns
    -------
    phase : array
        Unwrapped phase; the first sample along axis is left unchanged.
    """

    return np.unwrap(phase_wrapped, discont=discont, axis=axis)


def _valid_mask(phase_wrapped, mask):
    phase_wrapped = np.asarray(phase_wrapped, dtype=float)
    if phase_wrapped.ndim != 2:
        raise ValueError("Expected a 2D phase array.")
    valid = np.isfinite(phase_wrapped)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != phase_wrapped.shape:
            raise ValueError("mask must have the shape of the phase.")
        valid &= mask

    return phase_wrapped, valid


def _wrapped_gradients(psi):
    """Wrapped forward differences along x (last axis) and y."""

    dx = wrap(np.diff(psi, axis=1))
    dy = wrap(np.diff(psi, axis=0))

    return dx, dy


def residues(phase_wrapped):
    """Residue map (phase singularities) of a wrapped phase.

    The residue of an elementary loop of 2×2 pixels is the sum of the wrapped
    phase differences around it, counterclockwise (in x, then y; i.e. with y
    pointing up), divided by 2π. It is an integer: 0 for a smooth phase, +1
    at the core of a vortex exp(+iφ) (φ the azimuth), -1 for exp(-iφ). Wherever
    residues exist, unwrapping depends on the path, and a congruent
    unwrapped phase has to contain cuts between residues of opposite sign.

    Parameters
    ----------
    phase_wrapped : array (Ny, Nx)
        Wrapped phase.

    Returns
    -------
    res : integer array (Ny-1, Nx-1)
        res[i, j] belongs to the loop with corners [i, j], [i, j+1],
        [i+1, j+1], [i+1, j], i.e. it sits at the point between those four
        pixels. Loops that touch a NaN are 0.
    """

    psi = np.asarray(phase_wrapped, dtype=float)
    dx, dy = _wrapped_gradients(psi)
    # counterclockwise: +x along the lower edge, +y along the right edge,
    # -x along the upper edge, -y along the left edge:
    circ = dx[:-1, :] + dy[:, 1:] - dx[1:, :] - dy[:, :-1]
    res = np.rint(circ/TWO_PI)
    res[~np.isfinite(res)] = 0

    return res.astype(int)


def reliability(phase_wrapped, mask=None):
    """Reliability (quality) map after Herráez et al. (2002).

    The reliability of a pixel is 1/D with the second difference ::

        D² = H² + V² + D₁² + D₂²,

    H = wrap(ψ[i, j-1] − ψ[i, j]) − wrap(ψ[i, j] − ψ[i, j+1]) and likewise V
    (along y) and D₁, D₂ (along the diagonals). Second differences are small
    for smooth (low curvature) phases and large for noise and near residues.

    Parameters
    ----------
    phase_wrapped : array (Ny, Nx)
        Wrapped phase.
    mask : boolean array, optional
        Valid pixels.

    Returns
    -------
    R : array
        Reliability; border pixels, pixels outside the mask and pixels with
        an invalid neighbour have R = 0.
    """

    psi, valid = _valid_mask(phase_wrapped, mask)
    p = np.where(valid, psi, 0.0)
    c = p[1:-1, 1:-1]

    def d2(a, b):
        return wrap(a - c) - wrap(c - b)

    H = d2(p[1:-1, :-2], p[1:-1, 2:])
    V = d2(p[:-2, 1:-1], p[2:, 1:-1])
    D1 = d2(p[:-2, :-2], p[2:, 2:])
    D2 = d2(p[:-2, 2:], p[2:, :-2])
    D = np.sqrt(H**2 + V**2 + D1**2 + D2**2)

    # a pixel is reliable only if its whole 3×3 neighbourhood is valid:
    v = valid.astype(int)
    n_valid = sum(v[1 + di:v.shape[0] - 1 + di, 1 + dj:v.shape[1] - 1 + dj]
                  for di in (-1, 0, 1) for dj in (-1, 0, 1))
    R = np.zeros_like(psi)
    with np.errstate(divide="ignore"):
        R[1:-1, 1:-1] = np.where(n_valid == 9, 1.0/D, 0.0)
    R[~np.isfinite(R)] = np.finfo(float).max  # D = 0: perfectly smooth

    return R


def _poisson_eigenvalues(shape):
    Ny, Nx = shape
    ky = 2*np.cos(np.pi*np.arange(Ny)/Ny) - 2
    kx = 2*np.cos(np.pi*np.arange(Nx)/Nx) - 2

    return ky[:, None] + kx[None, :]


def _solve_poisson_neumann(rho, eig):
    """Solve the discrete Poisson equation ∇²φ = rho with Neumann boundary
    conditions by a DCT-II (Ghiglia and Romero 1994). The mean of φ is 0.
    """

    rho_hat = dctn(rho, type=2, norm="ortho")
    eig = eig.copy()
    eig[0, 0] = 1.0
    phi_hat = rho_hat/eig
    phi_hat[0, 0] = 0.0

    return idctn(phi_hat, type=2, norm="ortho")


def _divergence(gx, gy, shape):
    """Discrete divergence (backward difference) of edge fields gx (Ny, Nx-1)
    and gy (Ny-1, Nx), the negative adjoint of the forward difference: the
    Laplacian with Neumann boundaries is _divergence(diff_x(φ), diff_y(φ)).
    """

    rho = np.zeros(shape)
    rho[:, :-1] += gx
    rho[:, 1:] -= gx
    rho[:-1, :] += gy
    rho[1:, :] -= gy

    return rho


def _weighted_laplacian(phi, wx, wy):
    gx = wx*np.diff(phi, axis=1)
    gy = wy*np.diff(phi, axis=0)

    return _divergence(gx, gy, phi.shape)


def _fix_piston(phi, psi, valid, congruent):
    """Shift phi such that it agrees with psi modulo 2π on average, optionally
    snap it to the nearest congruent phase, and set invalid pixels to NaN.
    """

    out = np.full(phi.shape, np.nan)
    if not np.any(valid):
        return out
    d = phi[valid] - psi[valid]
    offset = np.angle(np.mean(np.exp(1j*d)))  # circular mean of phi - psi
    phi = phi - offset
    if congruent:
        phi = psi + TWO_PI*np.rint((phi - psi)/TWO_PI)
    # bring the mean into (-π, π] by a multiple of 2π (keeps congruence):
    phi = phi - TWO_PI*np.rint(np.mean(phi[valid])/TWO_PI)
    out[valid] = phi[valid]

    return out


def unwrap_ls(phase_wrapped, weights=None, mask=None, method="auto",
              congruent=False, tol=1E-10, max_iter=500, return_info=False):
    """Least squares phase unwrapping (Ghiglia and Romero 1994).

    Minimizes ::

        Σ w_x (φ[i, j+1] − φ[i, j] − Δ_x ψ)² + Σ w_y (φ[i+1, j] − φ[i, j] − Δ_y ψ)²

    with the wrapped differences Δψ = wrap(ψ[neighbour] − ψ) of the data ψ.
    If the true phase changes by less than π between neighbours, the
    wrapped differences are the true ones and the solution is exact (up to
    a piston). Edge weights are the smaller of the two pixel weights squared,
    w_x = min(w[i, j]², w[i, j+1]²), as in Ghiglia and Pritt (1998).

    Parameters
    ----------
    phase_wrapped : array (Ny, Nx)
        Wrapped phase. NaNs are treated as masked pixels.
    weights : array (Ny, Nx), optional
        Pixel weights ≥ 0, e.g. a modulation, amplitude or quality map
        (normalized or not). Pixels with weight 0 are excluded.
    mask : boolean array (Ny, Nx), optional
        True for valid pixels.
    method : str, optional
        "dct": unweighted least squares, solved directly by one DCT (only
        without weights and mask). "pcg": weighted least squares by
        preconditioned conjugate gradients, preconditioned with the DCT
        solver. "auto" (default): "dct" if there are no weights, mask or
        NaNs, otherwise "pcg".
    congruent : bool, optional
        If True, return the congruent phase closest to the least squares
        solution, ψ + 2π·round((φ_LS − ψ)/2π). It is exact wherever φ_LS is
        within π of the truth, and keeps the full detail (and noise) of the
        data.
    tol : float, optional
        Relative residual at which "pcg" stops.
    max_iter : int, optional
        Maximum number of "pcg" iterations.
    return_info : bool, optional
        If True, also return a dict with the number of iterations and the
        final relative residual ("pcg"; 0 and 0.0 for "dct").

    Returns
    -------
    phase : array (Ny, Nx)
        Unwrapped phase, NaN outside the valid region. The piston is chosen
        such that phase agrees with phase_wrapped modulo 2π on average, and
        the mean of phase is in (-π, π].
    info : dict
        Only if return_info is True.
    """

    psi, valid = _valid_mask(phase_wrapped, mask)
    use_weights = weights is not None or not np.all(valid)
    if method == "auto":
        method = "pcg" if use_weights else "dct"
    if method not in ("dct", "pcg"):
        raise ValueError("Unknown method '%s'." % method)
    if method == "dct" and use_weights:
        raise ValueError("method='dct' is unweighted; use 'pcg' for weights, "
                         "masks and NaNs.")

    p = np.where(valid, psi, 0.0)
    dx, dy = _wrapped_gradients(p)
    eig = _poisson_eigenvalues(psi.shape)
    info = {"iterations": 0, "residual": 0.0}

    if method == "dct":
        phi = _solve_poisson_neumann(_divergence(dx, dy, psi.shape), eig)
    else:
        w = np.ones(psi.shape) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != psi.shape:
            raise ValueError("weights must have the shape of the phase.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        w2 = np.where(valid, w, 0.0)**2
        wx = np.minimum(w2[:, :-1], w2[:, 1:])
        wy = np.minimum(w2[:-1, :], w2[1:, :])

        # Q φ = c with Q the weighted Laplacian; Q is negative semidefinite,
        # so solve -Q φ = -c by CG (Ghiglia and Romero 1994, Ghiglia and
        # Pritt 1998, ch. 5.3):
        def A(v):
            return -_weighted_laplacian(v, wx, wy)

        def M_inv(r):
            return -_solve_poisson_neumann(r, eig)

        b = -_divergence(wx*dx, wy*dy, psi.shape)
        b_norm = np.linalg.norm(b)
        phi = np.zeros(psi.shape)
        if b_norm > 0:
            r = b.copy()
            z = M_inv(r)
            d = z.copy()
            rz = np.vdot(r, z)
            for it in range(1, max_iter + 1):
                Ad = A(d)
                alpha = rz/np.vdot(d, Ad)
                phi += alpha*d
                r -= alpha*Ad
                res = np.linalg.norm(r)/b_norm
                if res < tol:
                    break
                z = M_inv(r)
                rz_new = np.vdot(r, z)
                d = z + (rz_new/rz)*d
                rz = rz_new
            info = {"iterations": it, "residual": res}

    phi = _fix_piston(phi, psi, valid, congruent)

    if return_info:
        return phi, info

    return phi


def _find(parent, offset, p):
    """Root of p and the offset of p relative to the root (in units of 2π),
    with path compression.
    """

    path = []
    while parent[p] != p:
        path.append(p)
        p = parent[p]
    root = p
    # compress: walk back from the node closest to the root
    acc = 0
    for q in reversed(path):
        acc += offset[q]
        offset[q] = acc
        parent[q] = root

    return root


def unwrap_quality_guided(phase_wrapped, quality=None, mask=None):
    """Quality guided unwrapping by reliability sorting (Herráez et al. 2002).

    Every pair of horizontally or vertically neighbouring pixels (an edge)
    gets the reliability q_1 + q_2 of its pixels. Edges are processed from the
    most to the least reliable one; each joins the groups of its two pixels,
    adding to the smaller group the multiple of 2π that makes the step
    across the edge smaller than π. Unreliable regions (noise, residues) are
    thus unwrapped last and cannot spoil the good parts. The path is not
    continuous, which makes the algorithm fast (O(N log N) for the sort).

    Parameters
    ----------
    phase_wrapped : array (Ny, Nx)
        Wrapped phase. NaNs are treated as masked pixels.
    quality : array (Ny, Nx), optional
        Pixel quality, larger is better (e.g. modulation or amplitude). The
        default is reliability(), the inverse second difference.
    mask : boolean array (Ny, Nx), optional
        True for valid pixels.

    Returns
    -------
    phase : array (Ny, Nx)
        Unwrapped phase, congruent with phase_wrapped (phase − phase_wrapped
        is a multiple of 2π everywhere), NaN outside the valid region.
        Regions that are not connected to each other get independent
        pistons. The mean of the result is in (-π, π].
    """

    psi, valid = _valid_mask(phase_wrapped, mask)
    Ny, Nx = psi.shape
    if quality is None:
        q = reliability(psi, valid)
        # the border has no second differences; give it the reliability of
        # its nearest inner pixel so that it is not unwrapped last:
        if Ny > 2 and Nx > 2:
            q[0, :], q[-1, :] = q[1, :], q[-2, :]
            q[:, 0], q[:, -1] = q[:, 1], q[:, -2]
        q = np.minimum(q, 1E300)  # avoid inf + inf below
    else:
        q = np.asarray(quality, dtype=float)
        if q.shape != psi.shape:
            raise ValueError("quality must have the shape of the phase.")
    p = np.where(valid, psi, 0.0)

    idx = np.arange(Ny*Nx).reshape(Ny, Nx)
    # edges: (pixel a, pixel b, reliability, number of 2π to add to b - a)
    e_a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    e_b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    e_q = np.concatenate([(q[:, :-1] + q[:, 1:]).ravel(),
                          (q[:-1, :] + q[1:, :]).ravel()])
    e_ok = np.concatenate([(valid[:, :-1] & valid[:, 1:]).ravel(),
                           (valid[:-1, :] & valid[1:, :]).ravel()])
    p_flat = p.ravel()
    raw = p_flat[e_b] - p_flat[e_a]
    e_k = np.rint((wrap(raw) - raw)/TWO_PI).astype(int)

    order = np.argsort(-e_q[e_ok], kind="stable")
    e_a, e_b, e_k = e_a[e_ok][order], e_b[e_ok][order], e_k[e_ok][order]

    # union-find with offsets: the unwrapped phase of pixel p is
    # ψ_p + 2π·(offset of p relative to its root + ... up to the root)
    parent = list(range(Ny*Nx))
    offset = [0]*(Ny*Nx)
    size = [1]*(Ny*Nx)
    for a, b, k in zip(e_a.tolist(), e_b.tolist(), e_k.tolist()):
        ra = _find(parent, offset, a)
        rb = _find(parent, offset, b)
        if ra == rb:
            continue
        oa = offset[a] if a != ra else 0
        ob = offset[b] if b != rb else 0
        # need n_b - n_a = k, i.e. shift group b by m = k + oa - ob:
        m = k + oa - ob
        if size[ra] >= size[rb]:
            parent[rb] = ra
            offset[rb] = m
            size[ra] += size[rb]
        else:
            parent[ra] = rb
            offset[ra] = -m
            size[rb] += size[ra]

    n = np.empty(Ny*Nx, dtype=int)
    for i in range(Ny*Nx):
        r = _find(parent, offset, i)
        n[i] = offset[i] if i != r else 0
    phi = p + TWO_PI*n.reshape(Ny, Nx)

    # independent pistons for unconnected regions: bring each region's mean
    # into (-π, π]
    roots = np.array([_find(parent, offset, i) for i in range(Ny*Nx)]).reshape(Ny, Nx)
    out = np.full(psi.shape, np.nan)
    v_roots = roots[valid]
    v_phi = phi[valid]
    uniq, inv = np.unique(v_roots, return_inverse=True)
    means = np.bincount(inv, weights=v_phi)/np.bincount(inv)
    out[valid] = v_phi - TWO_PI*np.rint(means/TWO_PI)[inv]

    return out


def unwrap_with_reference(phase_wrapped, reference, mask=None):
    """Unwrap a phase with an unwrapped estimate of it.

    Each pixel gets the multiple of 2π that brings it closest to the
    reference, ::

        φ = ψ + 2π·round((reference − ψ)/2π).

    The result is congruent with the data and exact wherever the reference
    is within ±π of the true phase, independently of all other pixels: no
    path is followed, so residues, noise and disconnected regions do not
    spread errors. Typical references are a TIE phase (pyoptics.TIE), the
    phase at a synthetic wavelength (unwrap_two_wavelength()), a model of
    the sample, or a smoothed least squares solution.

    Parameters
    ----------
    phase_wrapped : array
        Wrapped phase ψ (any shape). NaNs are treated as masked pixels.
    reference : array or number
        Unwrapped estimate of the phase, with the same piston as the data
        (e.g. both zero on the background).
    mask : boolean array, optional
        True for valid pixels.

    Returns
    -------
    phase : array
        Unwrapped phase, NaN outside the mask. Unlike the path following
        methods, the piston is not changed: it is the reference's.
    """

    psi = np.asarray(phase_wrapped, dtype=float)
    phi = psi + TWO_PI*np.rint((reference - psi)/TWO_PI)
    if mask is not None:
        phi = np.where(mask, phi, np.nan)

    return phi


def synthetic_wavelength(wavelength1, wavelength2):
    """Synthetic (beat) wavelength λ1·λ2/|λ1 − λ2| of two wavelengths."""

    return wavelength1*wavelength2/abs(wavelength1 - wavelength2)


def unwrap_two_wavelength(phase1, phase2, wavelength1, wavelength2, mask=None,
                          refine=True):
    """Optical path difference from wrapped phases at two wavelengths.

    Both phases belong to the same optical path difference (OPD),
    ψ_j = wrap(2π·OPD/λ_j). Their difference is the phase at the synthetic
    wavelength Λ = λ1·λ2/|λ1 − λ2| (synthetic_wavelength()), ::

        OPD_Λ = Λ_s·wrap(ψ1 − ψ2)/2π,   Λ_s = λ1·λ2/(λ2 − λ1),

    which is unambiguous for |OPD| < Λ/2, without any path following. Its
    noise is that of the single phases times Λ/λ, so with refine=True the
    coarse OPD is only used as reference to unwrap each phase
    (unwrap_with_reference()), and the two results are averaged. That needs
    the coarse OPD to be within ±λ/2 of the truth (hierarchical unwrapping,
    Gass et al. 2003).

    Parameters
    ----------
    phase1, phase2 : arrays
        Wrapped phases at wavelength1 and wavelength2, in radians, with a
        common zero (e.g. both background compensated).
    wavelength1, wavelength2 : numbers
        The (vacuum) wavelengths, which must differ.
    mask : boolean array, optional
        True for valid pixels.
    refine : bool, optional
        If False, return the coarse synthetic wavelength OPD.

    Returns
    -------
    opd : array
        Optical path difference in the unit of the wavelengths, NaN outside
        the mask. The phase at λ_j is 2π·opd/λ_j. Dispersion of the sample
        (a different OPD at the two wavelengths) shows up as an error of
        Λ/λ times the OPD difference in the coarse result.
    """

    if wavelength1 == wavelength2:
        raise ValueError("The wavelengths must differ.")

    psi1 = np.asarray(phase1, dtype=float)
    psi2 = np.asarray(phase2, dtype=float)
    Lambda_s = wavelength1*wavelength2/(wavelength2 - wavelength1)
    opd = Lambda_s*wrap(psi1 - psi2)/TWO_PI

    if refine:
        opd1 = unwrap_with_reference(psi1, TWO_PI*opd/wavelength1)*wavelength1/TWO_PI
        opd2 = unwrap_with_reference(psi2, TWO_PI*opd/wavelength2)*wavelength2/TWO_PI
        opd = 0.5*(opd1 + opd2)
    if mask is not None:
        opd = np.where(mask, opd, np.nan)

    return opd


def unwrap(phase_wrapped, method="quality", **kwargs):
    """Unwrap a 1D or 2D phase.

    Parameters
    ----------
    phase_wrapped : array
        Wrapped phase.
    method : str, optional
        "quality" (unwrap_quality_guided(), default), "ls" (unwrap_ls()),
        "reference" (unwrap_with_reference(), needs reference=...) or "1d"
        (unwrap_1d(); the default for 1D input without a reference).
    kwargs :
        Passed on to the selected function.

    Returns
    -------
    phase : array
        Unwrapped phase.
    """

    if np.ndim(phase_wrapped) == 1 and method != "reference":
        method = "1d"
    funcs = {"quality": unwrap_quality_guided, "ls": unwrap_ls, "1d": unwrap_1d,
             "reference": unwrap_with_reference}
    if method not in funcs:
        raise ValueError("Unknown method '%s'." % method)

    return funcs[method](phase_wrapped, **kwargs)
