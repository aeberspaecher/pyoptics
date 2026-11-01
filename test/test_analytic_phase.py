#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.analytic_phase (semi-analytical and homeomorphic Fourier
transforms, propagation with analytic quadratic phase)."""

import warnings

import numpy as np
import pytest

from pyoptics import tilted
from pyoptics.analytic_phase import (QuadraticPhaseField, semi_analytical_FT,
                                     inverse_semi_analytical_FT, propagate, tilted_plane,
                                     homeomorphic_FT, geometric_propagation)
from pyoptics.geometry import R_y
from pyoptics.propagators import rayleigh_sommerfeld_I_TF
from pyoptics.utils import grid1d


WL = 0.5
K = 2*np.pi/WL


def _gaussian_lens(N, L, w, f, y_shift=0.0):
    x = grid1d(-L/2, L/2, N)
    y = x + y_shift
    A = np.exp(-(x[None, :]**2 + y[:, None]**2)/w**2)
    return QuadraticPhaseField.from_curvature(A, x, y, WL, R=-f), x, y


def _fresnel_gaussian(X, Y, a_x, a_y, z):
    """Fresnel propagation of exp(-a_x*x**2 - a_y*y**2) (complex a) by z."""
    ap_x = 1/(1/a_x + 2j*z/K)
    ap_y = 1/(1/a_y + 2j*z/K)
    return (np.sqrt(ap_x/a_x)*np.sqrt(ap_y/a_y)*np.exp(-ap_x*X**2 - ap_y*Y**2)
            *np.exp(1j*K*z))


def test_sft_of_curved_gaussian_is_exact():
    w, f = 200.0, 5000.0
    F, x, y = _gaussian_lens(128, 1800.0, w, f, y_shift=3.0)
    assert min(F.chirp_ratio()) > 1                     # SFT branch
    assert min(F.required_samples()) > 10*128           # the full field would need > 1280
    S = semi_analytical_FT(F)
    a = 1/w**2 - 1j*F.alpha[0]
    KX, KY = np.meshgrid(S.x, S.y)
    U = np.pi/a*np.exp(-(KX**2 + KY**2)/(4*a))
    assert np.abs(S.field() - U).max() < 1E-8*np.abs(U).max()
    # the plain FFT on the same coarse grid is aliased
    S_fft = semi_analytical_FT(F, mode="fft")
    KX, KY = np.meshgrid(S_fft.x, S_fft.y)
    assert np.abs(S_fft.field() - np.pi/a*np.exp(-(KX**2 + KY**2)/(4*a))).max() > 0.1*abs(np.pi/a)


def test_fft_and_sft_agree_on_a_fine_grid():
    F, x, y = _gaussian_lens(512, 600.0, 60.0, 4000.0)
    U = F.field()
    S_sft = semi_analytical_FT(F, mode="sft")
    S_fft = semi_analytical_FT(F, mode="fft", kx_out=S_sft.x, ky_out=S_sft.y)
    assert np.abs(S_sft.field() - S_fft.field()).max() < 1E-8*np.abs(S_fft.field()).max()
    # and the FFT branch equals the plain sum
    assert np.isclose(S_fft.field()[len(S_sft.y)//2, len(S_sft.x)//2],
                      np.sum(U*np.exp(-1j*(S_sft.x[len(S_sft.x)//2]*x[None, :]
                                           + S_sft.y[len(S_sft.y)//2]*y[:, None])))
                      *(x[1] - x[0])**2, rtol=1E-9)


def test_round_trip():
    F, x, y = _gaussian_lens(96, 1800.0, 200.0, -3000.0)
    B = inverse_semi_analytical_FT(semi_analytical_FT(F))
    assert np.allclose(B.x, x) and np.allclose(B.y, y)
    assert np.abs(B.field() - F.field()).max() < 1E-8


@pytest.mark.parametrize("z", [1000.0, 4000.0, 4900.0, 5000.0, 5100.0, 9000.0, -3000.0])
def test_paraxial_propagation_of_focused_gaussian_is_exact(z):
    w, f = 200.0, 5000.0
    F, x, y = _gaussian_lens(128, 1800.0, w, f)
    G = propagate(F, z, paraxial=True)
    X, Y = np.meshgrid(G.x, G.y)
    a = 1/w**2 - 1j*F.alpha[0]
    U = _fresnel_gaussian(X, Y, a, a, z)
    assert np.abs(G.field() - U).max() < 1E-8*np.abs(U).max()


def test_astigmatic_beam_with_tilt():
    w, N, L = 150.0, 128, 1600.0
    x = grid1d(-L/2, L/2, N)
    A = np.exp(-(x[None, :]**2 + x[:, None]**2)/w**2)
    F = QuadraticPhaseField.from_curvature(A, x, x, WL, R=(-4000.0, -6000.0))
    G = propagate(F, 3000.0, paraxial=True)
    X, Y = np.meshgrid(G.x, G.y)
    U = _fresnel_gaussian(X, Y, 1/w**2 - 1j*F.alpha[0], 1/w**2 - 1j*F.alpha[1], 3000.0)
    assert np.abs(G.field() - U).max() < 1E-8*np.abs(U).max()
    # a carrier shifts the beam by z*c/k (paraxial)
    c = 0.02*K
    Ft = QuadraticPhaseField.from_curvature(A, x, x, WL, R=(-4000.0, -6000.0), carrier=(c, 0.0))
    Gt = propagate(Ft, 3000.0, paraxial=True)
    I = Gt.intensity()
    x_mean = np.sum(I*Gt.x[None, :])/np.sum(I)
    assert x_mean == pytest.approx(3000.0*c/K, rel=1E-6)


def test_nonparaxial_propagation_matches_angular_spectrum():
    # NA about 0.15: a fully sampled reference needs 1024**2 samples
    w, f, z = 25.0, 160.0, 100.0
    Nr, Lr = 1024, 200.0
    xr = grid1d(-Lr/2, Lr/2, Nr)
    R2 = xr[None, :]**2 + xr[:, None]**2
    U0 = np.exp(-R2/w**2)*np.exp(-1j*K*R2/(2*f))
    Ur, _, _ = rayleigh_sommerfeld_I_TF(U0, xr, xr, z, WL)
    F, x, y = _gaussian_lens(192, Lr, w, f)
    sel = (xr >= -40) & (xr < 40)
    x_out = xr[sel][::5]
    ref = Ur[np.ix_(sel, sel)][::5, ::5]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        G = propagate(F, z, x_out=x_out, y_out=x_out)
    assert np.abs(G.field() - ref).max() < 1E-4*np.abs(ref).max()


def test_focus_on_a_chosen_window():
    w, f = 200.0, 5000.0
    F, x, y = _gaussian_lens(128, 1800.0, w, f)
    x_out = grid1d(-20, 20, 81)
    G = propagate(F, f, paraxial=True, x_out=x_out, y_out=x_out)
    X, Y = np.meshgrid(x_out, x_out)
    a = 1/w**2 - 1j*F.alpha[0]
    U = _fresnel_gaussian(X, Y, a, a, f)
    assert np.abs(G.field() - U).max() < 1E-8*np.abs(U).max()


def test_tilted_plane_matches_exact_tilted_propagation():
    w, f, z0 = 40.0, 200.0, 120.0
    R = R_y(np.radians(30))
    Nr, Lr = 512, 220.0
    xr = grid1d(-Lr/2, Lr/2, Nr)
    R2 = xr[None, :]**2 + xr[:, None]**2
    U0 = np.exp(-R2/w**2)*np.exp(-1j*K*R2/(2*f))
    x_out = grid1d(-30, 30, 48)
    ut, _, _, _ = tilted.tilted_plane(U0, xr, xr, z0, WL, R=R, x_out=x_out, y_out=x_out,
                                      method="exact", remove_carrier=False)
    F, x, y = _gaussian_lens(64, Lr, w, f)
    T = tilted_plane(F, z0, R=R, x_out=x_out, y_out=x_out, N_out=256)
    assert np.abs(T.field() - ut).max() < 5E-4*np.abs(ut).max()


def test_untilted_plane_equals_propagate():
    F, x, y = _gaussian_lens(96, 1800.0, 200.0, 5000.0)
    x_out = grid1d(-300, 300, 50)
    T = tilted_plane(F, 2000.0, x_out=x_out, y_out=x_out)
    G = propagate(F, 2000.0, x_out=x_out, y_out=x_out)
    assert np.abs(T.field() - G.field()).max() < 1E-4*np.abs(G.field()).max()


def test_homeomorphic_FT_error_falls_with_the_fresnel_number():
    errors = []
    for f in (20000.0, 5000.0, 1250.0):
        w, N, L = 200.0, 64, 1800.0
        x = grid1d(-L/2, L/2, N)
        X, Y = np.meshgrid(x, x)
        A = np.exp(-(X**2 + Y**2)/w**2)
        psi = -K*(X**2 + Y**2)/(2*f)
        kx, ky, Ua, Up = homeomorphic_FT(A, psi, x, x)
        a = 1/w**2 + 1j*K/(2*f)
        U = np.pi/a*np.exp(-(kx**2 + ky**2)/(4*a))
        errors.append(np.abs(Ua*np.exp(1j*Up) - U).max()/np.abs(U).max())
        # leading order: relative error 2*f/(k*w**2) (inverse Fresnel number)
        assert errors[-1] == pytest.approx(2*f/(K*w**2), rel=0.1)
    assert errors[0] > errors[1] > errors[2]


def test_geometric_propagation_far_from_focus():
    # NA about 0.03, Fresnel number k*w**2/(2*f) = 50: paraxial and geometric regime
    w, f, N, L = 400.0, 20000.0, 64, 2400.0
    x = grid1d(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    A = np.exp(-(X**2 + Y**2)/w**2)
    psi = -K*(X**2 + Y**2)/(2*f)
    a = 1/w**2 + 1j*K/(2*f)
    for z in (5000.0, 60000.0):                # before and after the focus
        xz, yz, ua, ph = geometric_propagation(A, psi, x, x, z, WL)
        U = _fresnel_gaussian(xz, yz, a, a, z)
        mask = A > 1E-2
        err = np.abs(ua*np.exp(1j*ph) - U)[mask].max()/np.abs(U).max()
        assert err < 3E-2
        # rays: straight lines through the focus
        kx, ky = -K*X/f, -K*Y/f
        assert np.allclose(xz, X + z*kx/np.sqrt(K**2 - kx**2 - ky**2))
        assert np.allclose(xz[mask], (X*(1 - z/f))[mask], rtol=1E-2, atol=1E-6*L)
        # the power is conserved (area of each ray tube scales with det M)
        dA = np.abs(np.gradient(xz, axis=1)*np.gradient(yz, axis=0))
        assert np.sum(np.abs(ua)**2*dA) == pytest.approx(np.sum(A**2)*(x[1] - x[0])**2, rel=1E-2)


def test_resample_warns_for_a_chirped_amplitude():
    F, x, y = _gaussian_lens(64, 1800.0, 200.0, 5000.0)
    G = QuadraticPhaseField(F.field(), x, y, WL)            # phase multiplied into A
    with pytest.warns(RuntimeWarning):
        G.resample(grid1d(-100, 100, 32), grid1d(-100, 100, 32))


def test_from_field_splits_off_the_wavefront():
    x = grid1d(-50, 50, 128)
    X, Y = np.meshgrid(x, x)
    U = np.exp(-(X**2 + Y**2)/20.0**2)*np.exp(1j*K*(X**2 + Y**2)/(2*800.0))
    F = QuadraticPhaseField.from_field(U, x, x, WL, R=800.0)
    assert np.allclose(F.amplitude, np.exp(-(X**2 + Y**2)/20.0**2))
    assert F.radius_of_curvature == pytest.approx((800.0, 800.0))
    assert np.allclose(F.field(), U)
