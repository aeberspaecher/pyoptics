"""Tests of pyoptics.ultrafast."""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyoptics import glasses as gl
from pyoptics import raytrace as rt
from pyoptics import ultrafast as uf

BK7 = gl.get("N-BK7", "schott")


def test_dispersion_against_literature():
    # refractiveindex.info (SCHOTT N-BK7, Malitson fused silica) at 800 nm
    assert uf.group_index(BK7, 800.0) == pytest.approx(1.5267, abs=2e-4)
    assert uf.gvd(BK7, 800.0) == pytest.approx(44.65, abs=0.05)
    assert uf.gvd(uf.FUSED_SILICA, 800.0) == pytest.approx(36.16, abs=0.05)
    assert uf.tod(uf.FUSED_SILICA, 800.0) == pytest.approx(27.47, abs=0.1)
    assert float(uf.FUSED_SILICA(632.8)) == pytest.approx(1.4570, abs=1e-4)


def test_derivatives_of_a_polynomial():
    n = lambda lam: 1.5 + 1e-4*lam + 1e-7*lam**2 + 1e-10*lam**3
    lam = 700.0
    assert uf.dn_dlambda(n, lam, 1) == pytest.approx(1e-4 + 2e-7*lam + 3e-10*lam**2, rel=1e-9)
    assert uf.dn_dlambda(n, lam, 2) == pytest.approx(2e-7 + 6e-10*lam, rel=1e-6)
    assert uf.dn_dlambda(n, lam, 3) == pytest.approx(6e-10, rel=1e-4)


def test_gaussian_broadening_numeric():
    t, om = uf.pulse_grid(800.0, 2000.0, 512)
    om0 = om[256]
    for tau, gdd in ((20.0, 0.0), (20.0, 500.0), (40.0, -800.0)):
        tt = np.linspace(-400, 400, 4001)
        E = uf.to_time(uf.gaussian_spectrum(om, om0, tau, gdd), om, tt, om0)
        assert uf.fwhm(tt, np.abs(E)**2) == pytest.approx(uf.gaussian_broadening(tau, gdd), rel=2e-3)
        if gdd == 0:
            assert np.abs(E).max() == pytest.approx(1.0, rel=1e-6)
    gdd = 450.0
    taus = np.linspace(10, 80, 2001)
    best = taus[np.argmin(uf.gaussian_broadening(taus, gdd))]
    assert best == pytest.approx(uf.optimal_input_duration(gdd), abs=0.05)
    assert uf.gaussian_broadening(best, gdd) == pytest.approx(np.sqrt(2)*best, rel=1e-3)


def test_material_phase_taylor():
    om0 = float(uf.omega_from_wavelength(800.0))
    dw = 1e-3
    ph = uf.material_phase(BK7, om0 + dw*np.arange(-2, 3), 10.0, om0)
    gdd = (ph[3] - 2*ph[2] + ph[1])/dw**2
    assert ph[2] == pytest.approx(0.0, abs=1e-9)
    assert gdd == pytest.approx(10*uf.gvd(BK7, 800.0), rel=1e-3)


def _aspheric_singlet(f=100.0, epd=20.0, wl=800.0):
    n = float(BK7(wl))

    def system(K):
        S = [rt.Surface((n - 1)*f, 5.0, material=BK7, conic=K), rt.Surface(np.inf, f, n_after=1.0)]
        return rt.System(S, 0, wl*1e-6, entrance_pupil_radius=epd/2).at_paraxial_focus()

    K = brentq(lambda K: system(K).trace_pupil(0.0, 0.0, 1.0).points[-1][0, 1], -5, 0)
    return system(K)


def test_bor_delay_ray_trace():
    s = _aspheric_singlet()
    d = uf.ray_delays(s, np.linspace(0, 1, 6))
    lag = d.group - d.phase
    traced = lag[0] - lag
    bor = uf.bor_delay(d.height, 800.0, uf.system_power_derivative(s))
    thin = uf.bor_delay(10.0, 800.0, uf.thin_lens_power_derivative(100.0, BK7, 800.0))
    assert thin == pytest.approx(51.8, abs=0.1)
    assert np.allclose(traced, bor, rtol=0.01, atol=0.05)
    # the phase delay to the focus is constant (no spherical aberration)
    assert np.ptp(d.phase) < 0.1
    # GDD on the axis: the glass of the centre thickness
    assert d.gdd[0] == pytest.approx(5.0*uf.gvd(BK7, 800.0), rel=1e-3)


def test_focal_spectrum_airy_and_axis_model():
    s = _aspheric_singlet()
    r = np.linspace(0, 0.01, 41)
    E = uf.focal_spectrum(s, [800.0], r, n_rho=300)[0, 0]
    I = np.abs(E)**2/np.abs(E[0])**2
    NA = 10.0/np.hypot(10.0, 100.0)
    first_zero = 0.61*800e-6/NA
    i = np.argmin(I[r < 1.3*first_zero])
    assert r[i] == pytest.approx(first_zero, rel=0.05)
    # pulse on the axis: Bor's box model
    t, om = uf.pulse_grid(800.0, 800.0, 160)
    om0 = om[80]
    A0 = uf.gaussian_spectrum(om, om0, 20.0)
    A = A0*np.exp(-1j*uf.material_phase(BK7, om, 5.0, om0))     # axial dispersion precompensated
    lam = uf.wavelength_from_omega(om)
    E0 = uf.focal_spectrum(s, lam, [0.0], n_rho=300)[:, 0, 0]
    T0 = uf.ray_delays(s, [0.0]).group[0]
    E0 = E0*np.exp(-1j*(om - om0)*T0)
    tt = np.linspace(-150, 100, 501)
    I_num = np.abs(uf.to_time(E0*A, om, tt, om0))**2
    d = uf.ray_delays(s, [0.0, 1.0])
    delay = (d.group - d.phase)[0] - (d.group - d.phase)[1]
    model = uf.bor_axis_spectrum(om, A0, delay, d.gdd[1] - d.gdd[0])
    I_mod = np.abs(uf.to_time(model, om, tt, om0))**2
    assert uf.fwhm(tt, I_num) == pytest.approx(uf.fwhm(tt, I_mod), rel=0.01)
    assert tt[np.argmax(I_num)] == pytest.approx(tt[np.argmax(I_mod)], abs=3.0)


def test_prism_tilt():
    sf11 = gl.get("N-SF11", "schott")
    apex = np.deg2rad(60)
    th = uf.minimum_deviation_incidence(apex, float(sf11(800.0)))
    t1, t2, t_out = uf.prism_angles(apex, float(sf11(800.0)), th)
    assert t_out == pytest.approx(th) and t1 == pytest.approx(apex/2)
    D = uf.prism_dispersion(apex, sf11, 800.0, th)
    D_num = (uf.prism_angles(apex, float(sf11(800.5)), th)[2]
             - uf.prism_angles(apex, float(sf11(799.5)), th)[2])/1.0
    assert D == pytest.approx(D_num, rel=1e-5)
    d = uf.prism_ray_delays(apex, sf11, 800.0, th, np.linspace(-2, 2, 21))
    assert np.ptp(d.phase) < 1e-6
    slope = np.polyfit(d.x, d.group, 1)[0]
    assert abs(np.arctan(uf.C_MM_PER_FS*slope)) == pytest.approx(abs(uf.tilt_angle(800.0, D)),
                                                                 rel=1e-6)


def test_dispersed_beam_tilt_grating():
    period = 1e6/600
    t, om = uf.pulse_grid(800.0, 8000.0, 1024)
    om0 = om[512]
    A = uf.gaussian_spectrum(om, om0, 30.0)
    theta = uf.grating_angle(period, 1, uf.wavelength_from_omega(om))
    x = np.linspace(-0.2, 0.2, 41)
    tt = np.linspace(-500, 500, 2001)
    I = np.abs(uf.dispersed_beam_xt(x, tt, om, A, theta, 0.15))**2
    slope = np.polyfit(x, tt[np.argmax(I, axis=0)], 1)[0]
    gamma = uf.tilt_angle(800.0, uf.grating_dispersion(period, 1, 800.0))
    assert abs(np.arctan(uf.C_MM_PER_FS*slope)) == pytest.approx(gamma, rel=0.01)


def test_beamlet_and_sstf():
    f, w, lam = 50.0, 0.4, 800.0
    x = np.linspace(-0.2, 0.2, 2001)
    E = uf.beamlet_focus_1d(x, 0.0, lam, 0.0, w, f)
    w0 = lam*1e-6*f/(np.pi*w)
    assert np.interp(w0, x, np.abs(E))/np.abs(E).max() == pytest.approx(np.exp(-1), rel=1e-4)
    # power conservation (1D): int |E|^2 dx = sqrt(pi/2) w
    assert np.trapezoid(np.abs(E)**2, x) == pytest.approx(np.sqrt(np.pi/2)*w, rel=1e-3)
    # off-centre beamlets meet in the focus
    E1 = uf.beamlet_focus_1d(x, 0.0, lam, 1.0, w, f)
    assert x[np.argmax(np.abs(E1))] == pytest.approx(0.0, abs=1e-3)
    # on-axis duration away from the focus against the closed form
    tau = 30.0
    t, om = uf.pulse_grid(lam, 3000.0, 256)
    om0 = om[128]
    A = uf.gaussian_spectrum(om, om0, tau)
    chirp = 3.0/(np.sqrt(8*np.log(2))/tau)
    tt = np.linspace(-400, 400, 801)
    for z in (0.0, 0.5, 2.0):
        Ez = np.array([uf.beamlet_focus_1d([0.0], z, l, chirp*(o - om0), w, f)[0]
                       for l, o in zip(uf.wavelength_from_omega(om), om)])
        Ez = Ez*np.exp(-1j*(om - om0)*(f + z)/uf.C_MM_PER_FS)
        I = np.abs(uf.to_time(Ez*A, om, tt, om0))**2
        assert uf.fwhm(tt, I) == pytest.approx(uf.sstf_axis_pulse(z, tau, chirp, w, f, lam)[0],
                                               rel=0.01)
