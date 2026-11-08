#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the wavefront shaping tools: statistics of random TMs against the
enhancement theory and the Marchenko-Pastur law, the feedback algorithms and
the phase-stepping TM measurement.
"""

import numpy as np
import pytest

from pyoptics import wavefront_shaping as ws
from pyoptics.utils import grid2d


def gaussian_tm(M, N, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((M, N)) + 1j*rng.standard_normal((M, N)))/np.sqrt(2)


def test_phase_screen_statistics():
    phase = ws.random_phase_screen((256, 256), 4.0, 2.0, rng=1)
    assert phase.std() == pytest.approx(2.0)
    assert phase.mean() == pytest.approx(0.0, abs=1e-12)
    # autocorrelation exp(-r^2/l^2) at r = l along x
    c = np.mean(phase*np.roll(phase, 4, axis=1))/phase.var()
    assert c == pytest.approx(np.exp(-1), abs=0.05)


def test_segments_and_field():
    x, y, _, _ = grid2d(-8, 8, -8, 8, 64)
    labels = ws.segment_labels(x, y, 4, 8.0)
    assert labels.max() == 15
    assert np.sum(labels == -1) == 64*64 - 32*32
    assert np.all(np.bincount(labels[labels >= 0]) == 64)
    values = np.arange(16) + 1j
    field = ws.segment_field(values, labels)
    assert np.all(field[labels == 5] == values[5])
    assert np.all(field[labels == -1] == 0)
    assert ws.segment_field(np.ones((3, 16)), labels).shape == (3, 64, 64)


def test_transmission_matrix_of_linear_map():
    x, y, _, _ = grid2d(-8, 8, -8, 8, 32)
    labels = ws.segment_labels(x, y, 4, 12.0)
    screen = np.exp(1j*ws.random_phase_screen((32, 32), 1.0, 10.0, rng=2))
    propagate = lambda u: np.fft.fft2(u*screen, axes=(-2, -1))
    mask = np.zeros((32, 32), bool)
    mask[:5, :7] = True
    T = ws.transmission_matrix(propagate, labels, output=mask)
    T_batch = ws.transmission_matrix(propagate, labels, output=mask, batch=5)
    assert T.shape == (35, 16)
    assert np.allclose(T, T_batch)
    values = np.exp(1j*np.linspace(0, 3, 16))
    assert np.allclose(T @ values, propagate(ws.segment_field(values, labels))[mask])


def test_random_unitary():
    U = ws.random_unitary(20, rng=3)
    assert np.allclose(U.conj().T @ U, np.eye(20))


@pytest.mark.parametrize("modulation, levels", [("full", None), ("phase", None), ("phase", 2),
                                                ("phase", 4), ("binary_amplitude", None)])
def test_enhancement_statistics(modulation, levels):
    N = 400
    T = gaussian_tm(300, N, seed=4)
    eta = [ws.enhancement(t, ws.phase_conjugation(t, modulation, levels)) for t in T]
    expected = ws.expected_enhancement(N, modulation, levels)
    if modulation == "binary_amplitude":
        # the theory is for a fixed half plane; the best one is a bit better
        assert np.mean(eta) > expected
        eta = [ws.enhancement(t, (t.real > 0).astype(complex)) for t in T]
    assert np.mean(eta) == pytest.approx(expected, rel=0.03)


def test_enhancement_full_is_exact():
    t = gaussian_tm(1, 50)[0]
    assert ws.enhancement(t, ws.phase_conjugation(t, "full")) == pytest.approx(50)


def test_feedback_algorithms_reach_phase_conjugation():
    N = 64
    t = gaussian_tm(1, N, seed=5)[0]
    measure = lambda v: np.abs(t @ v)**2
    optimum = ws.enhancement(t, ws.phase_conjugation(t))
    for continuous in (False, True):
        values, count = ws.stepwise_sequential(measure, N, n_phases=5, continuous=continuous)
        assert count == 5*N
        assert ws.enhancement(t, values) == pytest.approx(optimum, rel=2e-2)
    values, _ = ws.partitioning(measure, N, 600, rng=6)
    assert ws.enhancement(t, values) > 0.95*optimum


def test_callback_counts():
    t = gaussian_tm(1, 16, seed=7)[0]
    seen = []
    ws.stepwise_sequential(lambda v: np.abs(t @ v)**2, 16, n_phases=4, continuous=True,
                           callback=lambda v, n: seen.append(n))
    assert seen == list(range(4, 65, 4))


@pytest.mark.parametrize("basis", ["canonical", "hadamard"])
def test_measured_tm(basis):
    M, N = 40, 32
    T = gaussian_tm(M, N, seed=8)
    r = gaussian_tm(1, M, seed=9)[0]*3
    measure = lambda v: np.abs(r + T @ v)**2
    T_obs, count = ws.measure_transmission_matrix(measure, N, n_steps=4, basis=basis)
    assert count == 4*N
    assert np.allclose(T_obs, np.conj(r)[:, None]*T)


def test_marchenko_pastur():
    s = np.linspace(0, 3, 30001)
    for ratio in (1.0, 0.5, 0.2):
        pdf = ws.marchenko_pastur_pdf(s, ratio)
        assert np.trapezoid(pdf, s) == pytest.approx(1, abs=2e-3)
        assert np.trapezoid(s**2*pdf, s) == pytest.approx(1, abs=2e-3)     # E[lambda] = 1
    assert ws.marchenko_pastur_pdf(1.0, 1.0) == pytest.approx(np.sqrt(3)/np.pi)
    # a large Gaussian matrix
    T = gaussian_tm(800, 400, seed=10)/np.sqrt(800)
    sv = np.linalg.svd(T, compute_uv=False)
    assert sv.max() == pytest.approx(1 + np.sqrt(0.5), abs=0.05)
    assert sv.min() == pytest.approx(1 - np.sqrt(0.5), abs=0.05)


def test_gerchberg_saxton_foci():
    T = gaussian_tm(200, 256, seed=11)
    targets = np.zeros(200)
    targets[[3, 50, 120]] = 1.0
    values, out = ws.tm_gerchberg_saxton(T, targets, n_iter=30)
    assert np.allclose(np.abs(values), 1)
    assert np.allclose(out, T @ values)
    I = np.abs(out)**2
    background = np.delete(I, [3, 50, 120]).mean()
    assert I[[3, 50, 120]].min() > 30*background
    assert I[[3, 50, 120]].std()/I[[3, 50, 120]].mean() < 0.1


def test_weighted_gerchberg_saxton_equalizes():
    T = gaussian_tm(9, 256, seed=12)
    I_plain = np.abs(ws.tm_gerchberg_saxton(T, np.ones(9), 100, weighted=False)[1])**2
    I_w = np.abs(ws.tm_gerchberg_saxton(T, np.ones(9), 100)[1])**2
    assert I_w.std()/I_w.mean() < 1e-2 < I_plain.std()/I_plain.mean()
    assert I_w.mean() > 0.9*I_plain.mean()
