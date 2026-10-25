#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.litho.source_opt: pixel groups, linearity of the
responses against the full simulation, optimality against standard shapes,
the pupil fill and the common dose constraints.
"""

import numpy as np
import pytest

from pyoptics.litho import Illumination, LithoSimulation, MaskLayout, Projector, ThinMask
from pyoptics.litho import source_opt as so


@pytest.mark.parametrize("symmetry, largest", [("none", 1), ("point", 2), ("xy", 4), ("xy-diagonal", 8)])
def test_pixel_groups(symmetry, largest):
    px = so.SourcePixels(15, symmetry=symmetry)
    assert np.array_equal(np.sort(np.concatenate(px.groups)), np.arange(len(px.s_x)))
    assert px.sizes.max() == largest
    for g in px.groups:  # a group is closed under the symmetry
        ij = {tuple(p) for p in px._ij(px.s_x[g], px.s_y[g])}
        if symmetry != "none":
            assert {(-i, -j) for i, j in ij} == ij
    W = np.random.default_rng(0).uniform(size=len(px))
    W /= W.sum()
    assert np.isclose(px.point_weights(W).sum(), 1.0)
    with pytest.raises(ValueError):
        so.SourcePixels(15, symmetry="sixfold")


@pytest.fixture(scope="module")
def optimizer():
    P = Projector(1.35, 193.0, n=1.437)
    ill = Illumination.conventional(0.5, polarization="TE")
    clips = {p: LithoSimulation(MaskLayout.lines_spaces(p, 45.0), ThinMask.att_psm(), ill, P)
             for p in (90.0, 130.0, 300.0)}
    return so.SourceOptimizer(clips, so.SourcePixels(17, symmetry="xy"), foci=(0.0, 40.0))


def test_responses_are_linear(optimizer):
    for ill in (Illumination.annular(0.9, 0.6, 17, polarization="TE"),
                Illumination.dipole(0.9, 0.5, n_samples=17, polarization="TE")):
        W = optimizer.project(ill)
        assert np.isclose(W.sum(), 1.0)
        assert np.allclose(optimizer.nils(W), optimizer.evaluate(ill), rtol=1e-9)
    with pytest.raises(ValueError):
        optimizer.project(Illumination.annular(0.9, 0.6, 16))


def test_optimum_beats_standard_shapes(optimizer):
    r = optimizer.optimize(min_fill=0)
    assert np.allclose(optimizer.evaluate(r.illumination), r.nils, rtol=1e-6)
    for ill in (Illumination.annular(0.9, 0.6, 17, polarization="TE"),
                Illumination.dipole(0.9, 0.5, n_samples=17, polarization="TE"),
                Illumination.conventional(0.7, 17, polarization="TE")):
        assert r.worst >= np.min(optimizer.nils(optimizer.project(ill))) - 2e-3
    filled = optimizer.optimize(min_fill=0.2)
    assert filled.worst <= r.worst + 1e-9 and filled.fill >= 0.18
    assert filled.weights.max() <= 1/(0.2*np.sum(np.hypot(optimizer.pixels.s_x, optimizer.pixels.s_y) <= 1))*optimizer.pixels.sizes.max() + 1e-9


def test_common_dose(optimizer):
    clips = {k: v for k, v in list(optimizer.clips.items())[:2]}
    opt = so.SourceOptimizer(clips, optimizer.pixels, foci=optimizer.foci)
    free = opt.optimize(min_fill=0.1)
    fixed = opt.optimize(min_fill=0.1, dose_tolerance=0.0)
    I = opt.edge_intensity(fixed.weights)[:, 0]
    assert np.isclose(I[0], I[1], rtol=1e-6) and fixed.worst <= free.worst + 1e-9
    with pytest.raises(ValueError):
        opt.optimize(min_fill=1.5)  # more pixels than there are


def test_scan(optimizer):
    params = [dict(sigma_out=o, sigma_in=0.5) for o in (0.7, 0.9)] + [dict(sigma_out=0.4, sigma_in=0.6)]
    out = so.scan(optimizer, lambda **p: Illumination.annular(n_samples=17, polarization="TE", **p), params)
    assert len(out) == 2 and out[0][0] >= out[-1][0]  # the empty annulus is skipped
