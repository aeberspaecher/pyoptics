#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for phase retrieval (phase_retrieval) and the IFTA module (ifta).
"""

import numpy as np
import pytest

from pyoptics import ifta
from pyoptics import phase_retrieval as pr
from pyoptics.bases import FringeZernikes
from pyoptics.fft import FT, FT_unitary
from pyoptics.imaging import get_psf, pupil_coordinates
from pyoptics.propagators import rayleigh_sommerfeld_I_TF
from pyoptics.utils import grid1d


# support-based phase retrieval:

N = 64
_x = np.arange(N) - N//2
X, Y = np.meshgrid(_x, _x)
# a real, nonnegative object: a rectangle and a weaker disk
OBJ = (((np.abs(X + 3) < 6) & (np.abs(Y - 2) < 4)).astype(float)
       + 0.7*((X - 4)**2 + (Y + 5)**2 < 16))
SUPPORT = (np.abs(X) < 14) & (np.abs(Y) < 14)


def test_twin_has_same_fourier_magnitude():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))

    assert np.allclose(np.abs(FT(pr.twin(g))), np.abs(FT(g)))
    assert np.allclose(pr.twin(pr.twin(g)), g)


def test_align_removes_shift_phase_and_twin():
    g = OBJ*np.exp(1j*0.3*X/N)
    distorted = 1j*np.roll(pr.twin(g), (5, -7), axis=(0, 1))
    aligned, error = pr.align(distorted, g)

    assert error < 1e-12
    assert np.allclose(aligned, g)


def test_er_keeps_constraints_and_lowers_the_error():
    magnitude = np.abs(FT(OBJ))
    result = pr.ER(magnitude, SUPPORT, N_iter=50, positivity=True, rng=1)

    assert np.all(result.obj[~SUPPORT] == 0)
    assert np.all(result.obj >= 0)
    assert np.isrealobj(result.obj)
    # the error of error reduction never increases:
    assert np.all(np.diff(result.errors) <= 1e-12)
    assert result.errors[-1] < result.errors[0]


def test_hio_reconstructs_real_object_up_to_ambiguities():
    magnitude = np.abs(FT(OBJ))
    hio = pr.HIO(magnitude, SUPPORT, beta=0.9, N_iter=500, positivity=True, rng=1)
    er = pr.ER(magnitude, SUPPORT, N_iter=50, start=hio.obj, positivity=True)
    _, error = pr.align(er.obj, OBJ)

    assert er.errors[-1] < 1e-4
    assert error < 1e-3
    assert pr.fourier_error(er.obj, magnitude) < 1e-4


def test_hio_with_shrinkwrap_reconstructs_complex_object():
    obj = OBJ*np.exp(1j*0.8*np.cos(X/4.0)*np.sin(Y/5.0))
    magnitude = np.abs(FT(obj))
    hio = pr.HIO(magnitude, None, N_iter=1000, shrinkwrap=True, rng=2)
    er = pr.ER(magnitude, hio.support, N_iter=100, start=hio.obj)
    _, error = pr.align(er.obj, obj)

    assert error < 1e-2
    # shrink-wrap has tightened the support from the autocorrelation:
    assert hio.support.sum() < pr.autocorrelation_support(magnitude).sum()/2


def test_callback_sees_every_iteration():
    seen = []
    pr.ER(np.abs(FT(OBJ)), SUPPORT, N_iter=7, rng=0,
          callback=lambda i, g, support: seen.append((i, g.shape, support.sum())))

    assert [s[0] for s in seen] == list(range(1, 8))
    assert all(s[1] == OBJ.shape and s[2] == SUPPORT.sum() for s in seen)


def test_autocorrelation_support_contains_object():
    support = pr.autocorrelation_support(np.abs(FT(OBJ)))
    # the object is shifted by the autocorrelation, but the support of the
    # autocorrelation contains a copy of it around the center:
    assert np.all(support[OBJ > 0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pr.ER(np.ones((8, 8)), np.ones((4, 4), dtype=bool))


# phase diversity:

WL = 0.5
K = 2*np.pi/WL


def _pupil_setup(NA=0.25, N_grid=128):
    L = N_grid*WL/(4*NA)  # dx = wavelength/(4*NA)
    x = grid1d(-L/2, L/2, N_grid)
    P_x, P_y = pupil_coordinates(x, x, NA, K)
    basis = FringeZernikes(P_x[0], P_y[:, 0])

    return x, P_x, P_y, basis


def _rms_waves(phase, reference, inside):
    d = (phase - reference)[inside]
    d -= d.mean()

    return np.sqrt(np.mean(d**2))/(2*np.pi)


def test_defocus_to_waves_matches_propagation():
    # a PSF with spherical aberration propagated by dz equals the PSF with the
    # Z4 coefficient from defocus_to_waves(), but not with the opposite sign
    NA, N_grid = 0.1, 256
    L = N_grid*WL/(4*NA)
    x = grid1d(-L/2, L/2, N_grid)
    spherical = lambda P_x, P_y: 2*np.pi*0.3*(6*(P_x**2 + P_y**2)**2 - 6*(P_x**2 + P_y**2) + 1)
    psf = get_psf(x, x, NA, K, phase=spherical)

    for dz in (100.0, -100.0):
        I = np.abs(rayleigh_sommerfeld_I_TF(psf, x, x, dz, WL)[0])**2
        I /= I.sum()
        c4 = pr.defocus_to_waves(dz, NA, K)
        right = pr.diversity_psfs(x, x, NA, K, [c4], phase=spherical)[0]
        wrong = pr.diversity_psfs(x, x, NA, K, [-c4], phase=spherical)[0]

        assert abs(c4) == pytest.approx(0.5)
        assert np.max(np.abs(I - right)) < 5e-3*I.max()  # paraxial approximation
        assert np.max(np.abs(I - wrong)) > 0.1*I.max()


@pytest.mark.parametrize("method", ["fit", "gs"])
def test_phase_diversity_recovers_random_zernike_phase(method):
    x, P_x, P_y, basis = _pupil_setup()
    rng = np.random.default_rng(3)
    true = {j: rng.normal(0, 0.08) for j in range(2, 16)}
    phase = 2*np.pi*sum(c*basis.eval_single_scattered(P_x, P_y, j) for j, c in true.items())
    inside = P_x**2 + P_y**2 <= 1
    defocus = [-0.75, 0.0, 0.75]

    psfs = pr.diversity_psfs(x, x, 0.25, K, defocus, phase=phase)
    result = pr.phase_diversity(psfs, defocus, x, x, 0.25, K, n_zernike=15, method=method)

    assert _rms_waves(result.phase, phase, inside) < 0.01  # below wavelength/100
    assert max(abs(result.zernikes[j] - true[j]) for j in true) < 0.005
    assert result.errors[-1] < 1e-6
    assert np.allclose(np.abs(result.obj), inside)


def test_phase_diversity_is_scale_invariant_and_accepts_start():
    x, P_x, P_y, basis = _pupil_setup(N_grid=64)
    true = {4: 0.05, 7: -0.1, 9: 0.08}
    phase = 2*np.pi*sum(c*basis.eval_single_scattered(P_x, P_y, j) for j, c in true.items())
    defocus = [-0.5, 0.5]
    psfs = 1e4*pr.diversity_psfs(x, x, 0.25, K, defocus, phase=phase)

    result = pr.phase_diversity(psfs, defocus, x, x, 0.25, K, n_zernike=11,
                                start={4: 0.04, 9: 0.1})
    for j in range(2, 12):
        assert result.zernikes[j] == pytest.approx(true.get(j, 0.0), abs=1e-8)


def test_single_in_focus_psf_has_sign_ambiguity():
    x, P_x, P_y, basis = _pupil_setup(N_grid=64)
    Z5 = basis.eval_single_scattered(P_x, P_y, 5)  # astigmatism, even
    psf_plus = pr.diversity_psfs(x, x, 0.25, K, [0.0], phase=2*np.pi*0.1*Z5)
    psf_minus = pr.diversity_psfs(x, x, 0.25, K, [0.0], phase=-2*np.pi*0.1*Z5)
    assert np.allclose(psf_plus, psf_minus)

    # with defocus diversity, the two signs give different PSFs:
    psf_plus = pr.diversity_psfs(x, x, 0.25, K, [0.5], phase=2*np.pi*0.1*Z5)
    psf_minus = pr.diversity_psfs(x, x, 0.25, K, [0.5], phase=-2*np.pi*0.1*Z5)
    assert not np.allclose(psf_plus, psf_minus)


def test_phase_diversity_input_checks():
    x, _, _, _ = _pupil_setup(N_grid=32)
    psfs = np.ones((2, 32, 32))
    with pytest.raises(ValueError):
        pr.phase_diversity(psfs, [0.0], x, x, 0.25, K)
    with pytest.raises(ValueError):
        pr.phase_diversity(psfs, [0.0, 1.0], x, x, 0.25, K, method="foo")


# ifta:

_A_IN = np.exp(-(X**2 + Y**2)/(2*8.0**2))
_PHI = 0.02*X**2 + 0.3*np.sin(Y/5.0)
_I_TARGET = np.abs(FT_unitary(_A_IN*np.exp(1j*_PHI)))**2  # reachable target


def _amplitude_error(img):
    return np.linalg.norm(np.abs(img) - np.sqrt(_I_TARGET))/np.linalg.norm(np.sqrt(_I_TARGET))


def _random_start(seed=0):
    rng = np.random.default_rng(seed)
    return np.sqrt(_I_TARGET)*np.exp(2j*np.pi*rng.uniform(size=_I_TARGET.shape))


def test_gs_error_does_not_increase_and_converges():
    start = _random_start()
    errors = [_amplitude_error(ifta.GS(_A_IN**2, _I_TARGET, n).iterate(start)[1])
              for n in (1, 2, 5, 10, 20, 50, 100, 200)]

    assert np.all(np.diff(errors) <= 1e-12)
    assert errors[-1] < 0.2*errors[0]


def test_gs_finds_exact_solution_from_nearby_start():
    start = np.sqrt(_I_TARGET)*np.exp(1j*(np.angle(FT_unitary(_A_IN*np.exp(1j*_PHI)))
                                          + 0.3*np.random.default_rng(1).standard_normal(_I_TARGET.shape)))
    obj, img, n = ifta.GS(_A_IN**2, _I_TARGET, 200).iterate(start)

    assert n == 201
    assert _amplitude_error(img) < 5e-3  # vs. about 0.04 from a random start
    # the object has the prescribed amplitude, and energy is conserved:
    assert np.allclose(np.abs(obj), _A_IN)
    assert np.sum(np.abs(img)**2) == pytest.approx(np.sum(_A_IN**2))


@pytest.mark.parametrize("cls", [ifta.IFTA_HOO, ifta.IFTA_OverCompensation])
def test_ifta_variants_improve_and_keep_object_amplitude(cls):
    start = _random_start()
    _, img_1, _ = cls(_A_IN**2, _I_TARGET, 1).iterate(start)
    obj, img, _ = cls(_A_IN**2, _I_TARGET, 100).iterate(start)

    assert np.allclose(np.abs(obj), _A_IN)
    assert _amplitude_error(img) < 0.5*_amplitude_error(img_1)


def test_digitize_phase():
    phase = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
    field = 2.0*np.exp(1j*phase)
    quantized = ifta.digitize_phase(field, 4)

    assert np.allclose(np.abs(quantized), 2.0)
    assert np.allclose(np.unique(np.round(np.angle(quantized), 12)),
                       np.round(np.array([-3, -1, 1, 3])*np.pi/4, 12))
