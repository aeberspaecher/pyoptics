#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Rigorous coupled wave analysis (RCWA, Fourier modal method) of 1D
lamellar gratings, for rigorous (thick) mask diffraction.

Geometry: the grating is periodic along x (period P) and invariant along y.
A plane wave comes from the incidence medium (z < 0, refractive index
n_in, the mask blank), passes the layers in the given order (each one
lamellar: a background permittivity with rectangular segments of other
permittivities) and leaves into the exit medium (n_out, air below the
mask). Any direction of incidence is allowed (conical diffraction for
k_y != 0).

Method: in each layer the tangential fields are expanded into the orders
m = -M..M with the transverse wave numbers (normalized to k_0)
K_x,m = k_x/k_0 + m*lambda/P and K_y = k_y/k_0. With z' = k_0*z,
h = Z_0*H and the convention exp(i(k.r - omega*t)), Maxwell's equations
become ::

    d/dz' [E_x, E_y] = i*P [h_x, h_y],   d/dz' [h_x, h_y] = i*Q [E_x, E_y],

    P = [[K_x e^-1 K_y,     I - K_x e^-1 K_x],
         [K_y e^-1 K_y - I, -K_y e^-1 K_x   ]],
    Q = [[-K_x K_y,         K_x^2 - [eps]   ],
         [[1/eps]^-1 - K_y^2, K_y K_x        ]],

with the Toeplitz matrices [eps] and [1/eps] of the Fourier coefficients of
the permittivity and its inverse and e = [eps]. The factorization follows
Li's rules (Laurent's rule for E_y and E_z, which are continuous across the
vertical interfaces, the inverse rule for E_x, whose D_x is), which makes
TM and metallic gratings converge quickly. The modes of a layer vary as
exp(+-i*gamma*z') with gamma**2 the eigenvalues of P Q (Im(gamma) >= 0),
eigenvectors W (E) and V = Q W/gamma (h). The layers are stacked with the
enhanced transmittance matrix method (Moharam et al., JOSA A 12, 1077
(1995)), which is stable for thick and absorbing layers.

calibrate_boundary_layer() fits the boundary layer model of the thin and
film masks (mask.BoundaryLayer) to the rigorous orders.

RigorousMask is a mask stack for LithoSimulation and hopkins.TCC that uses
rcwa() for 1D layouts (lines along x or y): the absorber layers take the
shape of the layout (features and field), and every source point gets its
own diffracted orders, so the mask diffraction depends on the angle of
incidence and the polarization as it should (Abbe). The Hopkins TCC uses
the orders at normal incidence.

Results are Jones matrices of each transmitted and reflected order: they map
the tangential field (E_x, E_y) of the incident wave at the top of the
layers to the tangential field of the order at the bottom of the layers
(transmission) or at the top (reflection). Diffraction efficiencies are
``Re(K_z,m)*|E_m|**2/(Re(K_z,inc)*|E_inc|**2)`` with the full fields (E_z from
transversality).
"""

import numpy as np

from .mask import FilmMask, _rect_coeffs


class Layer(object):
    """A lamellar layer.

    Parameters
    ----------
    thickness : number
    background : complex
        Refractive index n + i*k of the layer outside the segments.
    segments : sequence of (center, width, index), optional
        Segments of other refractive indices (must not overlap, width < P).
    """

    def __init__(self, thickness, background, segments=()):
        self.thickness = float(thickness)
        self.background = complex(background)
        self.segments = [(float(c), float(w), complex(n)) for c, w, n in segments]

    def coefficients(self, period, m, inverse=False):
        """Fourier coefficients of eps (or 1/eps) for the orders m."""

        f = (lambda n: 1/n**2) if inverse else (lambda n: n**2)
        c = f(self.background)*(np.asarray(m) == 0).astype(complex)
        for center, width, n in self.segments:
            c = c + (f(n) - f(self.background))*_rect_coeffs(m, period, center, width)

        return c

    @property
    def is_uniform(self):
        return len(self.segments) == 0


def _toeplitz(c, N):
    """Toeplitz matrix T[i, j] = c[i - j] from coefficients c of the orders
    -(N - 1)..(N - 1)."""

    i = np.arange(N)
    return c[(i[:, None] - i[None, :]) + N - 1]


def _sqrt_branch(g2):
    """sqrt with Im >= 0 (decaying for exp(i*gamma*z)), and Re >= 0 for
    real arguments."""

    g = np.sqrt(g2.astype(complex))
    flip = (g.imag < 0) | ((g.imag == 0) & (g.real < 0))

    return np.where(flip, -g, g)


def _homogeneous(eps, Kx, Ky):
    """W, V, gamma of a homogeneous medium (batched over the leading axis):
    W = I, gamma = K_z, V = Q/gamma."""

    B, N = Kx.shape
    Kz = _sqrt_branch(eps - Kx**2 - Ky[:, None]**2)
    gamma = np.concatenate([Kz, Kz], axis=1)
    V = np.zeros((B, 2*N, 2*N), dtype=complex)
    d = np.arange(N)
    # Q = [[-Kx Ky, Kx^2 - eps], [eps - Ky^2, Ky Kx]] (diagonal blocks), V = Q/gamma
    V[:, d, d] = -Kx*Ky[:, None]/Kz
    V[:, d, N + d] = (Kx**2 - eps)/Kz
    V[:, N + d, d] = (eps - Ky[:, None]**2)/Kz
    V[:, N + d, N + d] = Ky[:, None]*Kx/Kz
    W = np.broadcast_to(np.eye(2*N, dtype=complex), (B, 2*N, 2*N))

    return W, V, gamma, Kz


def _layer_modes(layer, period, m_all, Kx, Ky, factorization):
    """W, V, gamma of a lamellar layer (batched).

    The modes of a 1D grating are the TE and TM modes of the planar problem
    in the plane spanned by x and the direction of propagation in the y-z
    plane, with propagation constants beta**2 from two N x N eigenproblems
    that depend on k_x only: TE (E perpendicular to x, tangential field
    (0, u)) from E - K_x**2, TM (h perpendicular to x, tangential h (0, v))
    from [1/eps]^-1 (I - K_x [eps]^-1 K_x). Then gamma**2 = beta**2 - k_y**2,
    V_TE = Q W_TE/gamma and W_TM = P V_TM/gamma. Source points that share
    k_x share the eigenproblems."""

    B, N = Kx.shape
    if layer.is_uniform:
        W, V, gamma, _ = _homogeneous(layer.background**2, Kx, Ky)
        return W, V, gamma
    E = _toeplitz(layer.coefficients(period, m_all), N)
    E_inv = np.linalg.inv(E)
    E_x = np.linalg.inv(_toeplitz(layer.coefficients(period, m_all, inverse=True), N)) \
        if factorization == "li" else E
    I = np.eye(N)

    # eigenproblems for the distinct k_x
    kx_unique, index = np.unique(np.round(Kx[:, 0], 12), return_inverse=True)
    KXu = Kx[np.unique(index, return_index=True)[1]][:, :, None]*I
    b2_TE, U = np.linalg.eig(E - KXu @ KXu)
    b2_TM, Vh = np.linalg.eig(E_x @ (I - KXu @ E_inv @ KXu))
    b2_TE, U, b2_TM, Vh = b2_TE[index], U[index], b2_TM[index], Vh[index]

    ky = Ky[:, None]
    g_TE, g_TM = _sqrt_branch(b2_TE - ky**2), _sqrt_branch(b2_TM - ky**2)
    KX = Kx[:, :, None]*I
    k3 = Ky[:, None, None]
    KeK = KX @ E_inv
    # V_TE = Q [0; u]/gamma, W_TM = P [0; v]/gamma
    V_TE = np.concatenate([KX @ KX @ U - E @ U, k3*(KX @ U)], axis=1)/g_TE[:, None, :]
    W_TM = np.concatenate([(I - KeK @ KX) @ Vh, -k3*(E_inv @ KX @ Vh)], axis=1)/g_TM[:, None, :]
    zeros = np.zeros_like(U)
    W = np.concatenate([np.concatenate([zeros, U], axis=1), W_TM], axis=2)
    V = np.concatenate([V_TE, np.concatenate([zeros, Vh], axis=1)], axis=2)

    return W, V, np.concatenate([g_TE, g_TM], axis=1)


class RCWAResult(object):
    """Result of rcwa(): orders m, K_x (shape (B, N)), K_y (B,), K_z in the
    incidence and exit media (B, N), and the Jones matrices of the
    reflected and transmitted orders, r and t of shape (B, N, 2, 2) (out
    component, in component; tangential E_x, E_y). B is the number of
    directions of incidence."""

    def __init__(self, m, Kx, Ky, Kz_in, Kz_out, r, t, n_in, n_out):
        self.m, self.Kx, self.Ky = m, Kx, Ky
        self.Kz_in, self.Kz_out = Kz_in, Kz_out
        self.r, self.t = r, t
        self.n_in, self.n_out = n_in, n_out

    def _efficiency(self, J, Kz, jones):
        jones = np.asarray(jones, dtype=complex)
        E = np.einsum("bnij,j->bni", J, jones)
        E_z = -(self.Kx*E[..., 0] + self.Ky[:, None]*E[..., 1])/Kz
        power = np.abs(E[..., 0])**2 + np.abs(E[..., 1])**2 + np.abs(E_z)**2
        i0 = int(np.argmin(np.abs(self.m)))
        Kz0 = self.Kz_in[:, i0]
        E_inc_z = -(self.Kx[:, i0]*jones[0] + self.Ky*jones[1])/Kz0
        p_inc = np.sum(np.abs(jones)**2) + np.abs(E_inc_z)**2

        return np.where(np.abs(Kz.imag) < 1e-12, Kz.real, 0.0)*power/(Kz0.real*p_inc)[:, None]

    def efficiencies(self, jones):
        """Diffraction efficiencies (R, T) of all orders, shape (B, N) each,
        for the incident tangential field jones = (E_x, E_y) (lossless
        incidence and exit media)."""

        return self._efficiency(self.r, self.Kz_in, jones), self._efficiency(self.t, self.Kz_out, jones)


def rcwa(period, layers, wavelength, k_x=0.0, k_y=0.0, n_in=1.0, n_out=1.0, n_orders=10,
         factorization="li"):
    """Diffraction of plane waves by a stack of lamellar layers.

    Parameters
    ----------
    period : number
    layers : sequence of Layer
        From the incidence side to the exit side.
    wavelength : number
    k_x, k_y : numbers or 1D arrays
        Transverse wave vector components of the incident waves, in units of
        k_0 (n_in*sin(theta)*cos(phi), ...); arrays are computed together.
    n_in, n_out : numbers
        Refractive indices of the incidence and exit media.
    n_orders : int, optional
        Orders -n_orders..n_orders are kept.
    factorization : "li" or "laurent", optional
        Li's rules (default) or Laurent's rule for all components (slower
        convergence for TM, for comparison).

    Returns
    -------
    RCWAResult
    """

    m = np.arange(-n_orders, n_orders + 1)
    N = len(m)
    m_all = np.arange(-2*n_orders, 2*n_orders + 1)
    k_x, k_y = np.broadcast_arrays(np.atleast_1d(np.asarray(k_x, dtype=float)),
                                   np.atleast_1d(np.asarray(k_y, dtype=float)))
    B = len(k_x)
    Kx = k_x[:, None] + m[None, :]*wavelength/period
    Ky = np.asarray(k_y, dtype=float)
    k0 = 2*np.pi/wavelength

    W_I, V_I, _, Kz_in = _homogeneous(complex(n_in)**2, Kx, Ky)
    _, V_II, _, Kz_out = _homogeneous(complex(n_out)**2, Kx, Ky)

    # enhanced transmittance matrices from the exit side up
    f, g = np.broadcast_to(np.eye(2*N, dtype=complex), (B, 2*N, 2*N)), V_II
    steps = []
    for layer in reversed(list(layers)):
        W, V, gamma = _layer_modes(layer, period, m_all, Kx, Ky, factorization)
        X = np.exp(1j*gamma*k0*layer.thickness)  # (B, 2N), |X| <= 1
        Wf, Vg = np.linalg.solve(W, f), np.linalg.solve(V, g)
        a, b = (Wf + Vg)/2, (Wf - Vg)/2
        a_inv_X = np.linalg.inv(a)*X[:, None, :]  # a^-1 X
        XbaX = X[:, :, None]*(b @ a_inv_X)
        f = W + W @ XbaX
        g = V - V @ XbaX
        steps.append(a_inv_X)

    # top: e_inc + W_I r = f T_1, h_inc - V_I r = g T_1, for unit E_x and E_y of order 0
    i0 = n_orders
    e_inc = np.zeros((B, 2*N, 2), dtype=complex)
    e_inc[:, i0, 0] = 1.0
    e_inc[:, N + i0, 1] = 1.0
    h_inc = V_I @ e_inc
    A = np.concatenate([np.concatenate([-W_I, f], axis=2),
                        np.concatenate([V_I, g], axis=2)], axis=1)
    sol = np.linalg.solve(A, np.concatenate([e_inc, h_inc], axis=1))
    r, T = sol[:, :2*N], sol[:, 2*N:]
    for a_inv_X in reversed(steps):  # T_{l+1} = a_l^-1 X_l T_l, from the top layer down
        T = a_inv_X @ T

    def jones(F):  # (B, 2N, 2) -> (B, N, 2, 2)
        return F.reshape(B, 2, N, 2).transpose(0, 2, 1, 3)

    return RCWAResult(m, Kx, Ky, Kz_in, Kz_out, jones(r), jones(T), n_in, n_out)


class RigorousMask(object):
    """Mask stack computed rigorously (RCWA) for 1D layouts.

    The absorber is a stack of layers on the mask blank, as for FilmMask,
    but patterned: in every layer the absorber regions of the layout (the
    features for tone="clear", the field for tone="dark") have the
    absorber material and the clear regions the clear material of that
    layer (the exit medium by default).

    Parameters
    ----------
    layers : sequence of (n_absorber, thickness) or (n_absorber, n_clear, thickness)
        Complex refractive indices and thicknesses (mask scale) from the
        blank outwards.
    substrate_n : number, optional
        Mask blank (fused silica at 193 nm: about 1.563).
    exit_n : number, optional
        Medium between mask and projection lens.
    n_orders : int, optional
        Orders -n_orders..n_orders of the RCWA (default: about twice those
        that propagate in the blank, at least 10).
    factorization : "li" or "laurent", optional
    normalize : bool, optional
        Normalize to the clear area (the unpatterned stack of the clear
        materials) at the same angle of incidence, as FilmMask does, so that
        the open frame transmits 1.
    name : str, optional
    """

    def __init__(self, layers, substrate_n=1.563, exit_n=1.0, n_orders=None, factorization="li",
                 normalize=True, name=None):
        self.layers = []
        for layer in layers:
            if len(layer) == 2:
                layer = (layer[0], exit_n, layer[1])
            self.layers.append((complex(layer[0]), complex(layer[1]), float(layer[2])))
        self.substrate_n, self.exit_n = substrate_n, exit_n
        self.n_orders, self.factorization = n_orders, factorization
        self.normalize = normalize
        self.boundary_layer = None
        self.name = name or "rigorous mask"
        self._cache = {}

    # RCWA results of the last calls (source points, layouts), since images, cuts and
    # focus-exposure matrices of one simulation need the same orders again
    cache_size = 16

    @property
    def thickness(self):
        return sum(t for _, _, t in self.layers)

    def film_mask(self):
        """The FilmMask of the same layers (laterally infinite films)."""

        n_abs = np.array([a for a, _, _ in self.layers])
        if any(c != self.exit_n for _, c, _ in self.layers):
            raise ValueError("film_mask() needs the exit medium as the clear material")
        return FilmMask(n_abs.real, n_abs.imag, [t for _, _, t in self.layers],
                        substrate_n=np.real(self.substrate_n), substrate_k=np.imag(self.substrate_n),
                        exit_n=self.exit_n, normalize=self.normalize)

    def jones(self, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices (J_clear, J_absorber) of laterally infinite films
        (clear: identity if normalized), for the open frame and as a film
        approximation; the patterned mask is order_jones()."""

        eye = np.broadcast_to(np.eye(2, dtype=complex), np.shape(s_x) + (2, 2))
        J_abs = (self.film_mask().jones(s_x, s_y, NA, wavelength, magnification)[1]
                 if all(c == self.exit_n for _, c, _ in self.layers) else np.full_like(eye, np.nan))
        if self.normalize:
            return eye, J_abs
        raise NotImplementedError("jones() of a RigorousMask with normalize=False")

    def _rcwa_layers(self, layout, magnification):
        """Layers across the lines of a layout periodic along x, at mask
        scale."""

        segments = [(magnification*x0, magnification*a) for x0, y0, a, b in layout.features]
        absorber_features = layout.tone == "clear"
        out = []
        for n_abs, n_clear, t in self.layers:
            n_feat, n_field = (n_abs, n_clear) if absorber_features else (n_clear, n_abs)
            out.append(Layer(t, n_field, [(c, w, n_feat) for c, w in segments]))

        return out

    def order_jones(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        """Jones matrices of the diffraction orders (m, n) of the layout for
        the source points, shape (len(s_x), 2, 2, len(n), len(m)); the
        results of the last calls are cached."""

        if not layout.is_1d:
            raise ValueError("RigorousMask supports 1D layouts (lines along x or y) only")
        s_x, s_y = np.atleast_1d(np.asarray(s_x, dtype=float)), np.atleast_1d(np.asarray(s_y, dtype=float))
        key = (layout.pitch_x, layout.pitch_y, tuple(map(tuple, layout.features)), layout.tone,
               tuple(np.atleast_1d(m)), tuple(np.atleast_1d(n)), s_x.tobytes(), s_y.tobytes(), NA,
               wavelength, magnification)
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self._order_jones(layout, m, n, s_x, s_y, NA, wavelength, magnification)

        return self._cache[key]

    def _order_jones(self, layout, m, n, s_x, s_y, NA, wavelength, magnification):
        axis = 0 if layout.pitch_x is not None else 1
        orders = np.atleast_1d(m if axis == 0 else n)
        M = magnification
        period = M*(layout.pitch_x if axis == 0 else layout.pitch_y)
        n_orders = self.n_orders
        if n_orders is None:
            n_orders = max(10, 2*int(np.ceil(period*(abs(self.substrate_n) + NA/M)/wavelength)))
        n_orders = max(n_orders, int(np.abs(orders).max()))
        k_par, k_perp = (np.asarray(s_x, dtype=float), np.asarray(s_y, dtype=float))
        if axis == 1:
            k_par, k_perp = k_perp, k_par
        k_par, k_perp = NA*np.atleast_1d(k_par)/M, NA*np.atleast_1d(k_perp)/M
        geometry = layout if axis == 0 else _transposed(layout)
        res = rcwa(period, self._rcwa_layers(geometry, M), wavelength, k_par, k_perp,
                   n_in=self.substrate_n, n_out=self.exit_n, n_orders=n_orders,
                   factorization=self.factorization)
        t = res.t[:, orders + n_orders]  # (B, orders, 2, 2)
        if self.normalize:
            clear = [Layer(t_l, n_clear) for _, n_clear, t_l in self.layers]
            ref = rcwa(period, clear, wavelength, k_par, k_perp, n_in=self.substrate_n,
                       n_out=self.exit_n, n_orders=0).t[:, 0]
            t = t @ np.linalg.inv(ref)[:, None]
        if axis == 1:  # back from the transposed geometry: swap x and y components
            t = t[..., ::-1, ::-1]
            out = t.transpose(0, 2, 3, 1)[..., :, None]  # (B, 2, 2, n, 1)
        else:
            out = t.transpose(0, 2, 3, 1)[..., None, :]  # (B, 2, 2, 1, m)

        return out


class Calibration(object):
    """Result of calibrate_boundary_layer(): the boundary_layer, the stack
    with the boundary layer, the scanned widths, the relative rms residual
    of the order amplitudes for each width (TE and TM together) and the
    residual of the base stack without a boundary layer
    (residual_base)."""

    def __init__(self, boundary_layer, stack, widths, residual, residual_base):
        self.boundary_layer, self.stack = boundary_layer, stack
        self.widths, self.residual = widths, residual
        self.residual_base = residual_base


def calibrate_boundary_layer(rigorous, layouts, NA, wavelength, magnification=4.0, base=None,
                             widths=None):
    """Fit a boundary layer model to rigorous mask diffraction.

    The boundary layer model (mask.BoundaryLayer) adds strips of width w
    with the transmissions t_TE (E along the edge) and t_TM (E across) at
    the edges of the absorber of a thin or film mask. For a given width the
    order amplitudes are linear in t_TE and t_TM, so these follow by linear
    least squares from the rigorous orders at normal incidence; the width is
    scanned.

    Parameters
    ----------
    rigorous : RigorousMask
    layouts : sequence of MaskLayout
        1D layouts with lines along y (pitch_x), e.g. several pitches and
        CDs; all orders that can reach the pupil (``|m|*lambda/p <= 2*NA``) are
        fitted.
    NA, wavelength, magnification : numbers
        Of the projector.
    base : ThinMask or FilmMask, optional
        The stack that gets the boundary layer (default: the FilmMask of the
        layers of the rigorous mask).
    widths : array, optional
        Widths to scan (wafer scale; default 0.5 to 15 nm).

    Returns
    -------
    Calibration
    """

    import copy

    from .mask import BoundaryLayer

    base = rigorous.film_mask() if base is None else base
    widths = np.linspace(0.5, 15.0, 30) if widths is None else np.asarray(widths, dtype=float)
    zero = np.zeros(1)
    J_clear, J_abs = (np.asarray(J)[0] for J in base.jones(zero, zero, NA, wavelength, magnification))
    data = []
    for layout in layouts:
        if layout.pitch_x is None or layout.pitch_y is not None:
            raise ValueError("calibrate_boundary_layer expects 1D layouts with lines along y")
        M = int(np.floor(2*NA*layout.pitch_x/wavelength))
        m = np.arange(-M, M + 1)
        J_o = rigorous.order_jones(layout, m, [0], zero, zero, NA, wavelength, magnification)[0][:, :, 0]
        J_region = ({"field": J_clear, "feature": J_abs} if layout.tone == "clear"
                    else {"field": J_abs, "feature": J_clear})
        data.append((layout, m, J_o, J_region))

    def fit(w):
        t, res2, norm2 = {}, 0.0, 0.0
        for comp, pol in ((0, "TM"), (1, "TE")):  # E_x across the lines along y: TM
            A, b = [], []
            for layout, m, J_o, J_region in data:
                c = layout.region_coefficients(m, [0], w)
                known = sum(c[r][0]*J_region[r][comp, comp] for r in ("field", "feature"))
                A.append(c["edge_y"][0])
                b.append(J_o[comp, comp] - known)
            A, b = np.concatenate(A), np.concatenate(b)
            t[pol] = np.vdot(A, b)/np.vdot(A, A).real
            res2 += np.sum(np.abs(b - A*t[pol])**2)
            norm2 += np.sum(np.abs(np.concatenate([J_o[comp, comp] for _, _, J_o, _ in data]))**2)
        return t, np.sqrt(res2/norm2)

    results = [fit(w) for w in widths]
    base_res2, norm2 = 0.0, 0.0
    for layout, m, J_o, J_region in data:
        c = layout.region_coefficients(m, [0])
        for comp in (0, 1):
            known = sum(c[r][0]*J_region[r][comp, comp] for r in ("field", "feature"))
            base_res2 += np.sum(np.abs(J_o[comp, comp] - known)**2)
            norm2 += np.sum(np.abs(J_o[comp, comp])**2)
    residual = np.array([r for _, r in results])
    i = int(np.argmin(residual))
    bl = BoundaryLayer(widths[i], results[i][0]["TE"], results[i][0]["TM"])
    stack = copy.copy(base)
    stack.boundary_layer = bl

    return Calibration(bl, stack, widths, residual, np.sqrt(base_res2/norm2))


def _transposed(layout):
    """The layout with x and y exchanged."""

    from .mask import MaskLayout

    return MaskLayout(layout.pitch_y, layout.pitch_x, [(y0, x0, b, a) for x0, y0, a, b in layout.features],
                      tone=layout.tone, name=layout.name)


def is_rigorous(stack):
    """True for mask stacks that give the Jones matrices of the orders
    themselves (order_jones())."""

    return hasattr(stack, "order_jones")


__all__ = ["Layer", "rcwa", "RCWAResult", "RigorousMask", "is_rigorous", "Calibration",
           "calibrate_boundary_layer"]
