#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.litho.hopkins: TCC and SOCS imaging against Abbe's method
(LithoSimulation), which agree exactly for masks that diffract the same way
for all source points (ThinMask), the properties of the TCC (Hermitian,
positive semidefinite, rank of coherent illumination) and the convergence of
truncated SOCS.
"""

import numpy as np
import pytest

from pyoptics.litho import (BoundaryLayer, FilmMask, Illumination, LithoSimulation, MaskLayout,
                            Projector, ThinMask, WaferStack, hopkins)


WL = 193.0
N_WATER = 1.437
NA = 1.35


def immersion(**kwargs):
    return Projector(NA, WL, n=N_WATER, **kwargs)


CASES = {
    "L/S dipole y": (MaskLayout.lines_spaces(90.0, 45.0), ThinMask.binary(),
                     Illumination.dipole(0.95, 0.7, polarization="y")),
    "holes att. PSM quasar TE": (MaskLayout.contact_holes(130.0, 65.0), ThinMask.att_psm(0.06),
                                 Illumination.quasar(0.95, 0.7, polarization="TE")),
    "staggered unpolarized": (MaskLayout.staggered_contacts(160.0, 120.0, 60.0),
                              ThinMask.att_psm(0.06), Illumination.annular(0.9, 0.6)),
    "boundary layer": (MaskLayout.contact_holes(130.0, 65.0),
                       ThinMask(0.0, boundary_layer=BoundaryLayer(5.0, 0.3 + 0.1j, 0.5)),
                       Illumination.annular(0.9, 0.6, polarization="TE")),
}


def grid(layout):
    x = np.linspace(-layout.pitch_x/2 if layout.pitch_x else -100, 60, 23)
    y = None if layout.is_1d else np.linspace(-50, 70, 19)
    return x, y


@pytest.mark.parametrize("name", list(CASES))
def test_tcc_equals_abbe_for_thin_masks(name):
    layout, stack, ill = CASES[name]
    sim = LithoSimulation(layout, stack, ill, immersion(zernikes={7: 0.03, 9: 0.02}))
    foci = [0.0, 60.0]
    tcc = hopkins.TCC(sim, foci)
    x, y = grid(layout)
    abbe = np.array([sim.image(x, y, focus=f) for f in foci])
    assert np.allclose(tcc.image(x, y), abbe, rtol=0, atol=1e-12)
    assert np.allclose(tcc.socs().image(x, y), abbe, rtol=0, atol=1e-12)
    assert tcc.channels == (["scalar"] if name != "boundary layer" else [(0, 0), (1, 1)])


def test_tcc_with_wafer_stack_and_scalar_projector():
    layout, stack, ill = CASES["holes att. PSM quasar TE"]
    wafer = WaferStack((1.70, 0.03, 100.0), below=[(1.82, 0.34, 40.0)])
    for proj in (immersion(), immersion(vectorial=False)):
        sim = LithoSimulation(layout, stack, ill, proj, wafer=wafer)
        tcc = hopkins.TCC(sim, [20.0], depth=30.0)
        x, y = grid(layout)
        assert np.allclose(tcc.image(x, y)[0], sim.image(x, y, focus=20.0, depth=30.0), atol=1e-12)


def test_other_layouts_with_the_same_pitch():
    layout, stack, ill = CASES["holes att. PSM quasar TE"]
    sim = LithoSimulation(layout, stack, ill, immersion())
    tcc = hopkins.TCC(sim)
    x, y = grid(layout)
    for other in (layout.scaled(bias=6.0), MaskLayout.contact_holes(130.0, 50.0, cd_y=80.0)):
        assert np.allclose(tcc.image(x, y, layout=other)[0], sim.with_(layout=other).image(x, y),
                           atol=1e-12)
    with pytest.raises(ValueError):
        tcc.image(x, y, layout=MaskLayout.contact_holes(140.0, 65.0))


def test_tcc_is_hermitian_and_positive():
    layout, stack, ill = CASES["staggered unpolarized"]
    tcc = hopkins.TCC(LithoSimulation(layout, stack, ill, immersion(zernikes={8: 0.05})))
    T = tcc.matrix[0]
    assert np.allclose(T, T.conj().T, atol=1e-14*np.abs(T).max())
    assert np.linalg.eigvalsh(T).min() > -1e-12*np.abs(T).max()


def test_coherent_illumination_has_one_kernel():
    """One source point and one polarization state: the TCC is the outer
    product of one coherent transfer function, rank 1 in the scalar model
    (three field components in the vectorial model: rank up to 3)."""

    layout = MaskLayout.contact_holes(300.0, 120.0)
    for vectorial, rank in ((False, 1), (True, 3)):
        sim = LithoSimulation(layout, ThinMask.binary(), Illumination.coherent(0.2, 0.1, polarization="x"),
                              immersion(vectorial=vectorial))
        tcc = hopkins.TCC(sim)
        socs = tcc.socs()
        assert socs.n_kernels <= rank
        x, y = grid(layout)
        assert np.allclose(socs.image(x, y)[0], sim.image(x, y), atol=1e-12)


def test_socs_truncation_converges():
    layout, stack, ill = CASES["holes att. PSM quasar TE"]
    sim = LithoSimulation(layout, stack, ill, immersion())
    tcc = hopkins.TCC(sim)
    x, y = grid(layout)
    exact = tcc.image(x, y)[0]
    errors = [np.abs(tcc.socs(n_kernels=k).image(x, y)[0] - exact).max() for k in (1, 2, 4, 8, 16)]
    assert errors[-1] < 1e-3 and all(b <= a + 1e-12 for a, b in zip(errors, errors[1:]))
    s = tcc.socs(energy=0.99)
    assert s.energy[0] >= 0.99 and tcc.socs(n_kernels=s.n_kernels - 1).energy[0] < 0.99
    lam = tcc.socs().eigenvalues[0]
    assert np.all(np.diff(lam) <= 1e-12) and np.all(lam >= 0)


def test_kernel_fields_reproduce_the_image():
    """The kernel fields are the coherent images of a mask whose orders are
    all 1 (a comb of points): its image is their eigenvalue weighted sum."""

    layout, stack, ill = CASES["holes att. PSM quasar TE"]
    tcc = hopkins.TCC(LithoSimulation(layout, stack, ill, immersion()))
    socs = tcc.socs()
    x, y = grid(layout)
    K = socs.kernel_fields(x, y)
    assert K.shape == (socs.n_kernels, len(y), len(x))
    comb = np.ones(tcc.matrix.shape[-1])
    I = np.einsum("k,kyx->yx", socs.eigenvalues[0], np.abs(K)**2)
    assert np.allclose(tcc.image(x, y, mask_vector=comb, normalize=False)[0], I)
    # a clear mask (DC order only) images as the open frame:
    clear = np.zeros(tcc.matrix.shape[-1])
    clear[len(clear)//2] = 1.0
    assert np.allclose(socs.image(x, y, mask_vector=clear)[0], 1.0)


def test_hopkins_approximation_for_film_masks():
    """FilmMask transmissions depend weakly on the angle of incidence at the
    mask (NA/4): Hopkins (normal incidence) is close to Abbe, not equal."""

    layout = MaskLayout.lines_spaces(180.0, 45.0)
    stack = FilmMask([2.343], [0.586], [68.0])
    sim = LithoSimulation(layout, stack, Illumination.annular(0.9, 0.6, polarization="TE"), immersion())
    x, _ = grid(layout)
    diff = np.abs(hopkins.TCC(sim).image(x)[0] - sim.image(x)).max()
    assert 1e-6 < diff < 1e-2
