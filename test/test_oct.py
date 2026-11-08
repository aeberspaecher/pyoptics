#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the OCT tools against analytical results: the axial resolution of a
Gaussian spectrum, the sensitivity roll-off of a spectrometer, the
equivalence of time-domain and spectral-domain A-scans, the reflection
amplitudes of a layer stack, dispersion broadening and its numerical
compensation, and full range (mirror free) A-scans.
"""

import numpy as np
import pytest

from pyoptics import oct as oc


WL0, DWL = 0.84, 0.05
K0 = 2*np.pi/WL0
W = oc.bandwidth_k(WL0, DWL)


def _k(n=4096):
    return np.linspace(K0 - 4*W, K0 + 4*W, n, endpoint=False)


def _peak(z, A, sign=1):
    sel = sign*z > 0
    i = np.argmax(np.abs(A)*sel)
    return z[i], np.abs(A[i])


def test_axial_resolution_formula():
    assert oc.axial_resolution(0.84, 0.05) == pytest.approx(2*np.log(2)/np.pi*0.84**2/0.05)
    z = np.linspace(-30, 30, 20001)
    assert oc.fwhm(z, oc.coherence_envelope(z, W)) == pytest.approx(oc.axial_resolution(WL0, DWL),
                                                                   rel=2e-3)


@pytest.mark.parametrize("z_s", [80.0, 300.0])
def test_mirror_a_scan(z_s):
    k = _k()
    S = oc.gaussian_spectrum(k, K0, W)
    r = 0.1
    I = oc.spectral_interferogram(k, S, r, z_sample=z_s)
    z, A = oc.a_scan(I, k, background=S*(1 + r**2), norm=S.sum(), n_fft=8*len(k))
    for sign in (1, -1):
        zp, ap = _peak(z, A, sign)
        assert zp == pytest.approx(sign*z_s, abs=0.1)
        assert ap == pytest.approx(r, rel=1e-3)
    width = oc.fwhm(z, np.abs(A)*(z > 0))
    assert width == pytest.approx(oc.axial_resolution(WL0, DWL), rel=0.03)


def test_rolloff_matches_spectrometer():
    n_pix, m = 1024, 16
    kf = _k(n_pix*m)
    S = oc.gaussian_spectrum(kf, K0, W)
    _, Sp = oc.spectrometer(kf, S, n_pix)
    dk = (kf[-1] - kf[0] + kf[1] - kf[0])/n_pix
    res = 2.5*dk
    zmax = np.pi/(2*dk)
    for frac in (0.2, 0.5, 0.8):
        z_s = frac*zmax
        I = oc.spectral_interferogram(kf, S, 0.1, z_sample=z_s)
        kp, Ip = oc.spectrometer(kf, I, n_pix, resolution=res)
        _, Sp_r = oc.spectrometer(kf, S, n_pix, resolution=res)
        z, A = oc.a_scan(Ip, kp, background=Sp_r*1.01, norm=Sp.sum(), n_fft=8*n_pix)
        assert _peak(z, A)[1]/0.1 == pytest.approx(oc.sensitivity_rolloff(z_s, dk, res), rel=0.01)
    assert oc.max_depth(kp) == pytest.approx(zmax)


def test_time_domain_equals_spectral_domain():
    k = _k(2048)
    S = oc.gaussian_spectrum(k, K0, W)
    r = oc.stack_reflection(k, [1.45], [40.0], 1.0, 1.0)
    z_r = np.arange(0, 120, 0.05)
    td = oc.time_domain_signal(k, S, r, z_r, z_sample=30.0)
    env = np.abs(oc.demodulate(td))
    z, A = oc.a_scan(oc.spectral_interferogram(k, S, r, z_sample=30.0), k, background=S*(1 + np.abs(r)**2),
                     norm=S.sum(), n_fft=16*len(k))
    # the envelope is the A-scan, both peaks: surface and back side of the film
    for zp in (30.0, 30.0 + 1.45*40):
        i_td = np.argmin(np.abs(z_r - zp))
        a_sd = np.interp(zp, z, np.abs(A))
        assert env[i_td] == pytest.approx(a_sd, rel=0.02)
    assert env[np.argmin(np.abs(z_r - 30.0))] == pytest.approx(0.45/2.45, rel=0.02)


def test_stack_reflection_single_interface_and_amplitudes():
    k = _k(64)
    r = oc.stack_reflection(k, [], [], 1.0, 1.5)
    assert np.allclose(r, -0.2)
    n, t = [1.5, 1.38], [100.0, 50.0]
    z, amp = oc.single_reflection_amplitudes(n, t, 1.0, 1.42)
    assert np.allclose(z, [0, 150, 219])
    r_if = np.array([-0.2, (1.5 - 1.38)/2.88, (1.38 - 1.42)/2.80])
    assert np.allclose(amp, r_if*np.array([1, 0.96, 0.96*(1 - r_if[1]**2)]))
    kf = _k(4096)
    S = oc.gaussian_spectrum(kf, K0, W)
    rs = oc.stack_reflection(kf, n, t, 1.0, 1.42)
    I = oc.spectral_interferogram(kf, S, rs, z_sample=50.0)
    zz, A = oc.a_scan(oc.analytic_interferogram(I, background=S*(1 + np.abs(rs)**2)),
                      kf, background=0, norm=S.sum(), n_fft=4*len(kf))
    for zi, ai in zip(z, amp):
        assert np.interp(zi + 50.0, zz, np.abs(A)) == pytest.approx(abs(ai), rel=0.01)


def test_dispersion_broadening_and_compensation():
    k = _k(4096)
    S = oc.gaussian_spectrum(k, K0, W)
    phi2, phi3 = 60.0, 150.0
    q = k - K0
    r = 0.1*np.exp(1j*(phi2*q**2/2 + phi3*q**3/6))
    I = oc.spectral_interferogram(k, S, r, z_sample=200.0)
    c = oc.analytic_interferogram(I, background=S*1.01)
    z, A0 = oc.a_scan(oc.compensate_dispersion(c, k, K0, [0, phi3]), k, background=0,
                      n_fft=8*len(k))
    ratio = oc.fwhm(z, np.abs(A0))/oc.axial_resolution(WL0, DWL)
    assert ratio == pytest.approx(oc.dispersion_broadening(phi2, W), rel=0.02)
    found = oc.optimize_dispersion(c, k, K0, W, order=3)
    assert found[0] == pytest.approx(phi2, rel=0.02)
    assert found[1] == pytest.approx(phi3, rel=0.1)


def test_dispersion_coefficients():
    k = _k(512)
    phase = 3.0 + 2.0*(k - K0) + 0.5*7.0*(k - K0)**2 + 11.0/6*(k - K0)**3
    assert np.allclose(oc.dispersion_coefficients(k, phase, K0), [3, 2, 7, 11])


def test_full_range_removes_mirror():
    k = _k()
    S = oc.gaussian_spectrum(k, K0, W)
    I0 = oc.spectral_interferogram(k, S, 0.1, z_sample=-50.0)
    I90 = oc.spectral_interferogram(k, S, 0.1, z_sample=-50.0, r_ref=1j)
    z, A = oc.a_scan(oc.full_range_interferogram(I0, I90, S*1.01), k, background=0, norm=S.sum())
    assert _peak(z, A, -1)[0] == pytest.approx(-50, abs=0.5)
    assert _peak(z, A, -1)[1] == pytest.approx(0.1, rel=0.01)
    assert _peak(z, A, 1)[1] < 1e-6


def test_snr_ratio():
    assert oc.sd_td_snr_ratio(1000.0, 10.0) == pytest.approx(1000/(10*np.sqrt(np.pi/(8*np.log(2)))))
