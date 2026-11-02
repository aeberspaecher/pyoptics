#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""The FDFD operator on the 2D Yee grid, driven solves and resonances.

The field along z (u = Ez or Hz) obeys, with the stretched-coordinate PML
factors s_x, s_y and exp(-i omega t),

- Ez: (1/s_x) d/dx (1/s_x) d/dx u + (1/s_y) d/dy (1/s_y) d/dy u
  + omega**2 eps_z u = -i omega J_z,
- Hz: (1/s_x) d/dx (1/(eps_y s_x)) d/dx u + (1/s_y) d/dy (1/(eps_x s_y)) d/dy u
  + omega**2 u = -i omega M_z,

discretized with the forward and backward differences of the Yee cell
(the same staggering as the FDTD), so that the in-plane fields follow
from u by one difference each: Hx = (1/(i omega s_y)) dEz/dy,
Hy = -(1/(i omega s_x)) dEz/dx; Ex = -(1/(i omega eps_x s_y)) dHz/dy,
Ey = (1/(i omega eps_y s_x)) dHz/dx.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fdtd.grid import OFFSETS, E_COMPONENTS, H_COMPONENTS
from ..fdtd.boundaries import PML, PERIODIC, Bloch, parse
from ..fdtd.media import VACUUM
from ..fdtd.monitors import DFTFieldData


class PointSource(object):
    """Line current along z through a point (2D): an electric current for
    mode "Ez", a magnetic one for "Hz", on the nearest node of the field
    along z.

    Parameters
    ----------
    position : (x, y)
    amplitude : complex, optional
        Current (per unit length along z).
    """

    def __init__(self, position, amplitude=1.0):
        self.position = tuple(float(p) for p in position)
        self.amplitude = amplitude


class PlaneWave(object):
    """Plane wave injected on the boundary of a box: the solution is the
    total field inside the box and the scattered field outside (TFSF).

    The incident wave u_inc = amplitude*exp(i k.r) (phase zero at the
    origin) uses the wave number of the grid, so it is injected without
    leakage; the box has to lie in the background medium, inside the PML.

    Parameters
    ----------
    direction : (dx, dy) or number
        Direction of propagation, or its angle from the +x axis in radians.
    box : ((x0, x1), (y0, y1))
        The total-field region.
    amplitude : complex, optional
        Amplitude of the field along z.
    """

    def __init__(self, direction, box, amplitude=1.0):
        if np.ndim(direction) == 0:
            direction = (np.cos(direction), np.sin(direction))
        d = np.asarray(direction, dtype=float)
        self.direction = d/np.hypot(*d)
        self.box = tuple(tuple(float(v) for v in b) for b in box)
        self.amplitude = amplitude


class Solution(DFTFieldData):
    """Fields of a driven solve or a resonance: `fields[c]` has the shape
    (1, nx, ny) (one wavelength, as DFTFieldData), `coords[c]` the
    coordinates; `main` is the field along z, `omega` the (complex) angular
    frequency, `sim` the simulation."""

    def __init__(self, sim, omega, fields, coords):
        wl = 2*np.pi/np.real(omega)
        super(Solution, self).__init__([wl], fields, coords, sim.mode)
        self.sim, self.omega = sim, omega

    @property
    def main(self):
        return self.fields[self.sim.components[0]][0]


class Resonance(object):
    """A quasi-normal mode: complex angular frequency omega
    (Im(omega) < 0), its vacuum wavelength 2 pi/Re(omega), Q and field
    (a Solution, scaled to a largest abs(u) of 1 with real phase there)."""

    def __init__(self, omega, field):
        self.omega, self.field = complex(omega), field

    @property
    def wavelength(self):
        return 2*np.pi/self.omega.real

    @property
    def Q(self):
        return self.omega.real/(-2*self.omega.imag) if self.omega.imag < 0 else np.inf

    def overlap(self, other):
        """abs(<u, v>)/(norm(u) norm(v)) of the fields along z (the same grid)."""
        u, v = self.field.main.ravel(), other.field.main.ravel()
        return abs(np.vdot(u, v))/(np.linalg.norm(u)*np.linalg.norm(v))

    def __repr__(self):
        return "Resonance(wavelength=%.6g, Q=%.4g)" % (self.wavelength, self.Q)


def _diff(n, h, bc):
    """Forward difference (n x n) with the boundary condition of an axis:
    Dirichlet beyond the last node, or (Bloch-)periodic."""

    D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n), format="lil", dtype=complex)
    if bc == PERIODIC or isinstance(bc, Bloch):
        D[n - 1, 0] = np.exp(1j*bc.k*n*h) if isinstance(bc, Bloch) else 1.0
    return (D/h).tocsr()


def _stretch(n, bc, offset, omega, h):
    """PML factors s = kappa + i sigma/omega at nodes i + offset of an axis."""

    s = np.ones(n, dtype=complex)
    if not isinstance(bc, PML):
        return s
    m = bc.cells
    p = np.arange(n) + offset
    d = np.clip(np.maximum(m - p, p - (n - m))/m, 0.0, 1.0)
    sigma_max = -(bc.order + 1)*np.log(bc.reflection)/(2*m*h)
    return 1 + (bc.kappa_max - 1)*d**bc.order + 1j*sigma_max*d**bc.order/omega


class Simulation(object):
    """A 2D FDFD problem: grid, structure and boundaries, at any
    wavelength.

    Parameters
    ----------
    grid : fdtd.Grid
        2D grid (the PML lies inside it).
    structure : sequence of (shape, Medium)
        Drawn in order, later ones on top; media are evaluated at the
        wavelength (Medium.from_index(n, k) for a lossy medium).
    mode : {"Ez", "Hz"}
        Field along z.
    boundaries : per axis PML, "pec", "periodic" or Bloch, optional
        Default: PML(16) on both axes.
    background : Medium, optional
    subpixel : int, optional
        Sub-samples per cell and axis for averaging the permittivity
        (1: plain staircase).
    """

    def __init__(self, grid, structure=(), mode="Ez", boundaries=None, background=VACUUM, subpixel=4):
        if grid.ndim != 2:
            raise ValueError("the FDFD solver is 2D")
        if mode not in ("Ez", "Hz"):
            raise ValueError("mode must be 'Ez' or 'Hz'")
        self.grid, self.mode = grid, mode
        self.structure = list(structure)
        self.background = background
        self.boundaries = parse(boundaries if boundaries is not None else (PML(16), PML(16)), 2)
        self.subpixel = int(subpixel)
        self.ndim = 2
        self.e_components = E_COMPONENTS[(2, mode)]
        h_comps = H_COMPONENTS[(2, mode)]
        self.components = (("Ez",) + h_comps) if mode == "Ez" else (("Hz",) + self.e_components)
        self.sources, self.monitors = [], []   # for fdtd.plot
        self._eps_cache = {}
        self._wavelength = None

    # geometry

    def coords(self, component):
        off = OFFSETS[2][self.mode][component]
        g = self.grid
        return tuple(g.origin[a] + (np.arange(g.shape[a]) + off[a])*g.spacing[a] for a in range(2))

    def _medium_eps(self, wavelength):
        media = [m for _, m in self.structure] + [self.background]
        return np.array([complex(np.asarray(m.complex_eps(wavelength))) for m in media])

    def _fine(self, component, wavelength):
        """Permittivity on the sub-samples of the cells around the nodes of
        a component, shape (nx, S, ny, S)."""

        S = self.subpixel
        g = self.grid
        x, y = self.coords(component)
        t = ((np.arange(S) + 0.5)/S - 0.5)
        xf = (x[:, None] + t[None, :]*g.spacing[0]).ravel()
        yf = (y[:, None] + t[None, :]*g.spacing[1]).ravel()
        X, Y = np.meshgrid(xf, yf, indexing="ij")
        idx = np.full(X.shape, len(self.structure), dtype=np.int32)
        for i, (shape, _) in enumerate(self.structure):
            idx[np.asarray(shape.contains(X, Y), dtype=bool)] = i
        return self._medium_eps(wavelength)[idx].reshape(len(x), S, len(y), S)

    def epsilon(self, wavelength):
        """Averaged complex permittivity at the nodes of the E components:
        {"Ez": eps_z} (arithmetic mean over the cell, exact for E along all
        interfaces) or {"Ex": eps_x, "Ey": eps_y} (anisotropic: harmonic
        mean along the interface normal, arithmetic across)."""

        key = float(wavelength)
        if key in self._eps_cache:
            return self._eps_cache[key]
        out = {}
        for c in self.e_components:
            F = self._fine(c, wavelength)
            mean = F.mean(axis=(1, 3))
            if self.mode == "Ez":
                out[c] = mean
                continue
            hmean = 1/(1/F).mean(axis=(1, 3))
            from scipy.ndimage import gaussian_filter
            gx, gy = np.gradient(gaussian_filter(np.abs(mean), 1.0), *self.grid.spacing)
            norm = np.hypot(gx, gy)
            nn = np.where(norm > 0, (gx if c == "Ex" else gy)/np.where(norm > 0, norm, 1), 0.0)**2
            out[c] = nn*hmean + (1 - nn)*mean
        self._eps_cache = {key: out}
        self._wavelength = wavelength
        return out

    def permittivity(self, component=None, wavelength=None):
        """Real part of the averaged permittivity at the nodes of an E
        component (for plots), at the wavelength of the last solve unless
        given."""

        wl = wavelength or self._wavelength or 1.0
        return np.real(self.epsilon(wl)[component or self.e_components[0]])

    # operators

    def _parts(self, omega_pml):
        """Difference matrices with the PML factors folded in:
        dxf, dyf (to half nodes, forward), dxb, dyb (to integer nodes,
        backward), for the field along z."""

        g = self.grid
        nx, ny = g.shape
        hx, hy = g.spacing
        Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        # positions of the field along z: integer nodes (Ez) or half nodes (Hz)
        o = 0.0 if self.mode == "Ez" else 0.5
        out = {}
        for a, (n, h, bc) in enumerate(zip((nx, ny), (hx, hy), self.boundaries)):
            De = _diff(n, h, bc)
            Dh = -De.conj().T.tocsr()
            if self.mode == "Ez":   # u at integer nodes: forward -> half, backward -> integer
                f = sp.diags(1/_stretch(n, bc, o + 0.5, omega_pml, h)) @ De
                b = sp.diags(1/_stretch(n, bc, o, omega_pml, h)) @ Dh
            else:                   # u at half nodes: backward -> integer, forward -> half
                b = sp.diags(1/_stretch(n, bc, 0.0, omega_pml, h)) @ Dh
                f = sp.diags(1/_stretch(n, bc, 0.5, omega_pml, h)) @ De
            if a == 0:
                out["xf"], out["xb"] = sp.kron(f, Iy, format="csr"), sp.kron(b, Iy, format="csr")
            else:
                out["yf"], out["yb"] = sp.kron(Ix, f, format="csr"), sp.kron(Ix, b, format="csr")
        return out

    def _curlcurl(self, wavelength, omega_pml):
        """The spatial part L of the operator (A = L + omega**2 eps for Ez,
        L + omega**2 for Hz) and the difference matrices."""

        eps = self.epsilon(wavelength)
        d = self._parts(omega_pml)
        if self.mode == "Ez":
            L = d["xb"] @ d["xf"] + d["yb"] @ d["yf"]
        else:
            L = (d["xf"] @ sp.diags(1/eps["Ey"].ravel()) @ d["xb"] +
                 d["yf"] @ sp.diags(1/eps["Ex"].ravel()) @ d["yb"])
        return L.tocsr(), d, eps

    def operator(self, wavelength):
        """Sparse matrix A of the driven problem A u = b at a vacuum
        wavelength."""

        omega = 2*np.pi/wavelength
        L, _, eps = self._curlcurl(wavelength, omega)
        w2 = omega**2*(eps["Ez"].ravel() if self.mode == "Ez" else np.ones(L.shape[0]))
        return (L + sp.diags(w2)).tocsc()

    # derived fields

    def _solution(self, u, omega, d, eps):
        nx, ny = self.grid.shape
        main = self.components[0]
        fields = {main: u.reshape(1, nx, ny)}
        if self.mode == "Ez":
            fields["Hx"] = (d["yf"] @ u/(1j*omega)).reshape(1, nx, ny)
            fields["Hy"] = (-(d["xf"] @ u)/(1j*omega)).reshape(1, nx, ny)
        else:
            fields["Ex"] = (-(d["yb"] @ u)/(1j*omega*eps["Ex"].ravel())).reshape(1, nx, ny)
            fields["Ey"] = ((d["xb"] @ u)/(1j*omega*eps["Ey"].ravel())).reshape(1, nx, ny)
        coords = {c: self.coords(c) for c in fields}
        return Solution(self, omega, fields, coords)

    # driven problems

    def _incident(self, src, wavelength, omega):
        """Incident plane wave on the nodes of the field along z, with the
        wave number of the grid."""

        eps_b = complex(np.asarray(self.background.complex_eps(wavelength)))
        hx, hy = self.grid.spacing
        dx, dy = src.direction
        K = omega*np.sqrt(eps_b)
        for _ in range(50):   # Newton on the discrete dispersion relation
            f = (2/hx*np.sin(K*dx*hx/2))**2 + (2/hy*np.sin(K*dy*hy/2))**2 - omega**2*eps_b
            fp = (2/hx*dx*np.sin(K*dx*hx)) + (2/hy*dy*np.sin(K*dy*hy))
            step = f/fp
            K = K - step
            if abs(step) < 1e-15*abs(K):
                break
        x, y = self.coords(self.components[0])
        X, Y = np.meshgrid(x, y, indexing="ij")
        return src.amplitude*np.exp(1j*K*(dx*X + dy*Y)), X, Y

    def rhs(self, sources, wavelength, A=None):
        """Right-hand side b of A u = b for a list of sources."""

        omega = 2*np.pi/wavelength
        nx, ny = self.grid.shape
        b = np.zeros(nx*ny, dtype=complex)
        x, y = self.coords(self.components[0])
        hx, hy = self.grid.spacing
        for s in sources:
            if isinstance(s, PointSource):
                i, j = int(np.argmin(np.abs(x - s.position[0]))), int(np.argmin(np.abs(y - s.position[1])))
                b[i*ny + j] += -1j*omega*s.amplitude/(hx*hy)
            elif isinstance(s, PlaneWave):
                A = self.operator(wavelength) if A is None else A
                f, X, Y = self._incident(s, wavelength, omega)
                (x0, x1), (y0, y1) = s.box
                q = (~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))).astype(float).ravel()
                Q = sp.diags(q)
                b += (Q @ A - A @ Q) @ f.ravel()
            else:
                raise TypeError("unknown source %r" % (s,))
        return b

    def solve(self, wavelength, sources):
        """Fields of the sources at a vacuum wavelength.

        Parameters
        ----------
        wavelength : number
        sources : PointSource, PlaneWave or a list of them

        Returns
        -------
        Solution
            fields of the mode (the field along z first) as DFTFieldData
            for `pyoptics.fdtd.plot`; with a PlaneWave the total field in
            its box and the scattered field outside.
        """

        if isinstance(sources, (PointSource, PlaneWave)):
            sources = [sources]
        omega = 2*np.pi/wavelength
        L, d, eps = self._curlcurl(wavelength, omega)
        w2 = omega**2*(eps["Ez"].ravel() if self.mode == "Ez" else np.ones(L.shape[0]))
        A = (L + sp.diags(w2)).tocsc()
        b = self.rhs(sources, wavelength, A)
        u = spla.splu(A).solve(b)
        return self._solution(u, omega, d, eps)

    # resonances

    def _eigs(self, wavelength, omega_pml, omega_guess, n, v0=None):
        L, d, eps = self._curlcurl(wavelength, omega_pml)
        C = (sp.diags(1/eps["Ez"].ravel()) @ L).tocsc() if self.mode == "Ez" else L.tocsc()
        mu, V = spla.eigs(C, k=n, sigma=-complex(omega_guess)**2, which="LM", v0=v0)
        omega = np.sqrt(-mu.astype(complex))
        omega = np.where(omega.real < 0, -omega, omega)
        return omega, V, d, eps

    def resonances(self, wavelength, n=4, refine=1, v0=None):
        """Quasi-normal modes with vacuum wavelengths near a guess.

        The eigenvalues -omega**2 of the operator nearest to the guess, by
        shift-invert Arnoldi. The PML stretching depends on the frequency;
        it is set at the guess and then, for each mode more than 1e-3 away
        from it, at its own Re(omega) (`refine` times). Media are
        evaluated at the guess.

        Parameters
        ----------
        wavelength : number or complex
            Guess of the vacuum wavelength (a complex one, 2 pi/omega of a
            known resonance, shifts closer to a mode with low Q).
        n : int, optional
            Number of modes.
        refine : int, optional
            Re-solves with the PML at the mode's own frequency.
        v0 : array, optional
            Start vector of the Arnoldi iteration (e.g. the field of a
            nearby mode).

        Returns
        -------
        list of Resonance
            Sorted by the distance of omega to the guess. Modes in the PML
            (spurious, with large abs(Im(omega))) can appear among them; see
            `Resonance.Q` and the fields.
        """

        omega0 = 2*np.pi/complex(wavelength)
        wl_r = 2*np.pi/omega0.real
        om, V, d, eps = self._eigs(wl_r, omega0.real, omega0, n, v0)
        modes = []
        for k in range(len(om)):
            w, v, dd = om[k], V[:, k], d
            for _ in range(refine):
                if abs(w.real - omega0.real)/omega0.real < 1e-3:   # the PML hardly changes
                    break
                w2, V2, dd, eps = self._eigs(wl_r, w.real, w, 1, v)
                w, v = w2[0], V2[:, 0]
            i = int(np.argmax(np.abs(v)))
            v = v/v[i]
            modes.append(Resonance(w, self._solution(v, w, dd, eps)))
        modes.sort(key=lambda r: abs(r.omega - omega0))
        return modes

    def resonance(self, wavelength=None, like=None, n=3, refine=1, min_overlap=0.6):
        """One mode: the nearest to a wavelength, or the one most like a
        given Resonance (largest overlap of the fields; for tracing a mode
        through a gradual change of the structure on the same grid, which
        starts from its complex frequency and field).

        Parameters
        ----------
        wavelength : number, optional
            Guess; default: that of `like`.
        like : Resonance, optional
        n, refine :
            See resonances().
        min_overlap : number, optional
            With `like`: search more modes (up to 4 n + 4) until one
            overlaps it at least this much, else warn.

        Returns
        -------
        Resonance
        """

        if like is None:
            return self.resonances(wavelength, n=n, refine=refine)[0]
        guess = 2*np.pi/like.omega if wavelength is None else wavelength
        for k in (n, 2*n + 2, 4*n + 4):   # widen the search if no mode is alike
            modes = self.resonances(guess, n=k, refine=refine, v0=like.field.main.ravel())
            best = max(modes, key=like.overlap)
            if like.overlap(best) >= min_overlap:
                break
        else:
            import warnings
            warnings.warn("no mode overlaps the given one by %.2f or more (best %.2f)"
                          % (min_overlap, like.overlap(best)))
        return best
