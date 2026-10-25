#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test digital holography: recording, off-axis, phase-shifting and Gabor
reconstruction, refocusing, autofocus and quantitative phase. All lengths in
micrometres.
"""

import numpy as np
import pytest

from pyoptics import holography as holo
from pyoptics.fft import FT, inv_FT
from pyoptics.interferometry import phase_shifts
from pyoptics.propagators import propagate_padded, rayleigh_sommerfeld_I_TF
from pyoptics.utils import freq_grid, grid2d


WL = 0.5
N, L = 256, 128.0
D = 200.0  # object-camera distance


@pytest.fixture(scope="module")
def grid():
    x, y, X, Y = grid2d(-L/2, L/2, -L/2, L/2, N)
    return x, y, X, Y


def _band_limit(field, x, y, B):
    K_x, K_y = freq_grid(x, y)
    return inv_FT(FT(field)*(K_x**2 + K_y**2 <= B**2))


def _propagate(field, x, y, z):
    return propagate_padded(rayleigh_sommerfeld_I_TF, field, x, y, z, WL,
                            band_limit=True)[0]


def _phase_amplitude_object(X, Y):
    """Gaussian phase bump on a partly absorbing, smoothly bounded object."""
    r2 = X**2 + Y**2
    amplitude = 0.6 + 0.4*np.exp(-((X - 10)**2 + Y**2)/6**2)
    phase = 2.5*np.exp(-r2/8**2)
    return amplitude*np.exp(1j*phase)*np.exp(-(r2/40**2)**4)


@pytest.fixture(scope="module")
def off_axis(grid):
    x, y, X, Y = grid
    B = 0.8*holo.max_object_bandwidth(x, y)
    obj = _band_limit(_phase_amplitude_object(X, Y), x, y, B)
    O = _propagate(obj, x, y, D)
    theta_min, theta_max = holo.off_axis_angle_range(B, x, y, WL)
    ref_angle = (0.5*(theta_min + theta_max), 0.0)
    H = holo.hologram(O, x, y, WL, ref_angle=ref_angle)
    return dict(obj=obj, O=O, H=H, B=B, ref_angle=ref_angle)


# recording

def test_hologram_of_plane_wave(grid):
    """Two plane waves give cos fringes with the difference wave vector."""
    x, y, X, Y = grid
    k = 2*np.pi/WL
    theta = 0.05
    O = 0.5*np.exp(1j*0.3)*np.ones_like(X)
    H = holo.hologram(O, x, y, WL, ref_angle=(theta, 0.0), ref_amp=2.0)
    expected = 0.25 + 4 + 2*0.5*2*np.cos(0.3 - k*np.sin(theta)*X)
    assert np.allclose(H, expected)


def test_hologram_warns_above_nyquist(grid):
    x, y, X, Y = grid
    theta = np.arcsin(1.2*np.pi/(L/N)/(2*np.pi/WL))
    with pytest.warns(UserWarning, match="Nyquist"):
        holo.hologram(np.ones_like(X), x, y, WL, ref_angle=(theta, 0.0))


def test_off_axis_angle_range(grid):
    x, y, X, Y = grid
    k, k_ny = 2*np.pi/WL, np.pi/(L/N)
    B = 0.2*k_ny
    theta_min, theta_max = holo.off_axis_angle_range(B, x, y, WL)
    assert np.isclose(k*np.sin(theta_min), 3*B)
    assert np.isclose(k*np.sin(theta_max), k_ny - B)
    assert np.isclose(holo.max_object_bandwidth(x, y), k_ny/4)
    with pytest.raises(ValueError):
        holo.off_axis_angle_range(0.3*k_ny, x, y, WL)


def test_phase_shifted_holograms_stack(grid):
    x, y, X, Y = grid
    O = np.exp(1j*X/10)
    shifts = phase_shifts(4)
    frames = holo.phase_shifted_holograms(O, x, y, WL, shifts)
    assert frames.shape == X.shape + (4,)
    assert np.allclose(frames[..., 2], holo.hologram(O, x, y, WL, phase_shift=shifts[2]))


# off-axis reconstruction

def test_off_axis_reconstruction_in_hologram_plane(grid, off_axis):
    x, y, X, Y = grid
    O, H = off_axis["O"], off_axis["H"]
    O_rec = holo.reconstruct_off_axis(H, x, y, WL, ref_angle=off_axis["ref_angle"],
                                      radius=off_axis["B"], window="circle")
    c = slice(N//4, 3*N//4)
    assert np.max(np.abs(O_rec - O)[c, c]) < 1e-3*np.max(np.abs(O))
    assert np.max(np.abs(O_rec - O)) < 0.02*np.max(np.abs(O))


def test_off_axis_reconstruction_and_refocusing(grid, off_axis):
    """Record at distance D, reconstruct, refocus by -D: the object returns."""
    x, y, X, Y = grid
    O_rec = holo.reconstruct_off_axis(off_axis["H"], x, y, WL,
                                      ref_angle=off_axis["ref_angle"],
                                      radius=off_axis["B"], window="circle")
    obj_rec = holo.refocus(O_rec, x, y, -D, WL)
    c = slice(N//4, 3*N//4)
    assert np.max(np.abs(obj_rec - off_axis["obj"])[c, c]) < 5e-3
    # the twin is in focus at +D instead: there the object is blurred
    assert np.max(np.abs(holo.refocus(np.conj(O_rec), x, y, -D, WL)
                         - off_axis["obj"])[c, c]) > 0.1


def test_off_axis_carrier_from_angle_or_k(grid, off_axis):
    x, y, X, Y = grid
    carrier = holo._carrier(off_axis["ref_angle"], WL)
    a = holo.reconstruct_off_axis(off_axis["H"], x, y, WL, ref_angle=off_axis["ref_angle"])
    b = holo.reconstruct_off_axis(off_axis["H"], x, y, WL, carrier=carrier)
    assert np.allclose(a, b)


def test_find_carrier(grid, off_axis):
    x, y, X, Y = grid
    k_x, k_y = holo.find_carrier(off_axis["H"], x, y)
    k_cx, k_cy = holo._carrier(off_axis["ref_angle"], WL)
    dk = 2*np.pi/L
    assert abs(k_x - k_cx) < 0.1*dk and abs(k_y - k_cy) < 0.1*dk


def test_find_carrier_diagonal(grid):
    x, y, X, Y = grid
    ref_angle = (0.15, -0.1)
    H = holo.hologram(0.5*np.exp(-(X**2 + Y**2)/20**2) + 0j, x, y, WL, ref_angle=ref_angle)
    k_x, k_y = holo.find_carrier(H, x, y)
    k_cx, k_cy = holo._carrier(ref_angle, WL)
    dk = 2*np.pi/L
    assert abs(k_x - k_cx) < 0.2*dk and abs(k_y - k_cy) < 0.2*dk


def test_window_radius_warning(grid, off_axis):
    x, y, X, Y = grid
    k_c = holo._carrier(off_axis["ref_angle"], WL)[0]
    with pytest.warns(UserWarning, match="zero"):
        holo.reconstruct_off_axis(off_axis["H"], x, y, WL, ref_angle=off_axis["ref_angle"],
                                  radius=0.6*k_c)


def test_window_array_and_spectrum(grid, off_axis):
    x, y, X, Y = grid
    O, S, W = holo.reconstruct_off_axis(off_axis["H"], x, y, WL,
                                        ref_angle=off_axis["ref_angle"], window="hann",
                                        return_spectrum=True)
    assert S.shape == W.shape == X.shape
    assert np.allclose(holo.reconstruct_off_axis(off_axis["H"], x, y, WL,
                                                 ref_angle=off_axis["ref_angle"], window=W), O)


# zero order

def test_suppress_zero_order_recorded(grid):
    x, y, X, Y = grid
    O = 0.3*np.exp(-(X**2 + Y**2)/30**2)*np.exp(1j*X/7)
    R = holo.reference_wave(x, y, WL, (0.1, 0.0), 1.5)
    H = holo.hologram(O, x, y, WL, (0.1, 0.0), 1.5)
    H_f = holo.suppress_zero_order(H, method="recorded", I_ref=np.abs(R)**2, I_obj=np.abs(O)**2)
    assert np.allclose(H_f, 2*np.real(O*np.conj(R)))
    assert np.isclose(np.mean(holo.suppress_zero_order(H)), 0.0)


def test_suppress_zero_order_highpass(grid):
    x, y, X, Y = grid
    slow = 3 + np.exp(-(X**2 + Y**2)/30**2)
    fringes = np.cos(2*np.pi*80/L*X)  # periodic in the window
    H_f = holo.suppress_zero_order(slow + fringes, x, y, method="highpass", radius=1.0)
    assert np.max(np.abs(H_f - fringes)) < 1e-2


# phase-shifting holography

@pytest.mark.parametrize("shifts", [phase_shifts(4), phase_shifts(3),
                                    np.array([0.0, 0.7, 1.9, 2.6, 4.4])])
def test_phase_shifting_in_line(grid, shifts):
    """In-line phase shifting recovers O exactly (no zero order, no twin)."""
    x, y, X, Y = grid
    O = _phase_amplitude_object(X, Y)
    frames = holo.phase_shifted_holograms(O, x, y, WL, shifts, ref_amp=1.3)
    O_rec = holo.reconstruct_phase_shifting(frames, shifts, ref_amp=1.3)
    assert np.max(np.abs(O_rec - O)) < 1e-10


def test_phase_shifting_off_axis(grid):
    x, y, X, Y = grid
    O = _phase_amplitude_object(X, Y)
    shifts = phase_shifts(4)
    frames = holo.phase_shifted_holograms(O, x, y, WL, shifts, ref_angle=(0.02, 0.01))
    O_rec = holo.reconstruct_phase_shifting(frames, shifts, x, y, WL, ref_angle=(0.02, 0.01))
    assert np.max(np.abs(O_rec - O)) < 1e-10
    R = holo.reference_wave(x, y, WL, (0.02, 0.01))
    assert np.allclose(holo.reconstruct_phase_shifting(frames, shifts, reference=R), O_rec)


# Gabor holography

def test_gabor_twin_image_suppression(grid):
    """Iterating with absorption and support constraints reduces the twin
    image of an absorbing object."""
    x, y, X, Y = grid
    support = (np.abs(X) < 12) & (np.abs(Y) < 12)
    obj = 1 - 0.6*np.exp(-(X**2 + Y**2)/4**2) - 0.4*np.exp(-((X - 6)**2 + (Y + 5)**2)/2**2)
    obj = _band_limit(obj + 0j, x, y, 2.0)
    H = np.abs(_propagate(obj, x, y, D))**2

    plain = holo.reconstruct_gabor(H, x, y, D, WL)
    iterated, errors = holo.reconstruct_gabor(H, x, y, D, WL, n_iter=30, support=support,
                                              return_errors=True)
    c = slice(N//4, 3*N//4)
    err_plain = np.sqrt(np.mean(np.abs(plain - obj)[c, c]**2))
    err_iter = np.sqrt(np.mean(np.abs(iterated - obj)[c, c]**2))
    assert err_iter < 0.6*err_plain
    assert errors[-1] < errors[0]


# refocusing

@pytest.mark.parametrize("propagator", ["asm", "fresnel"])
def test_refocus_round_trip(grid, propagator):
    x, y, X, Y = grid
    obj = _band_limit(_phase_amplitude_object(X, Y), x, y, 1.0)
    there = holo.refocus(obj, x, y, 150.0, WL, propagator=propagator)
    back = holo.refocus(there, x, y, -150.0, WL, propagator=propagator)
    c = slice(N//4, 3*N//4)
    assert np.max(np.abs(back - obj)[c, c]) < 1e-3


def test_refocus_matches_propagator(grid):
    x, y, X, Y = grid
    obj = _phase_amplitude_object(X, Y)
    assert np.allclose(holo.refocus(obj, x, y, 50.0, WL, pad_factor=1),
                       rayleigh_sommerfeld_I_TF(obj, x, y, 50.0, WL, band_limit=True)[0])
    assert np.allclose(holo.refocus(obj, x, y, 50.0, WL, propagator=rayleigh_sommerfeld_I_TF,
                                    pad_factor=1),
                       rayleigh_sommerfeld_I_TF(obj, x, y, 50.0, WL)[0])


# autofocus

@pytest.mark.parametrize("metric", holo.focus_metrics)
def test_autofocus_amplitude_object(grid, metric):
    """An absorbing bar target is found within one depth step. The
    illumination has soft edges and the metric is evaluated in the centre, so
    that diffraction at the window edges does not count."""
    x, y, X, Y = grid
    bars = np.exp(-((X**2 + Y**2)/45**2)**4)
    for x0 in (-12, -4, 4, 12):
        bars[(np.abs(X - x0) < 2) & (np.abs(Y) < 15)] *= 0.2
    obj = _band_limit(bars + 0j, x, y, 1.2)
    O = _propagate(obj, x, y, D)
    z_range = np.linspace(-280, -120, 17)
    dz = z_range[1] - z_range[0]
    mask = (np.abs(X) < 30) & (np.abs(Y) < 30)
    z_best, values = holo.autofocus(O, x, y, z_range, WL, metric=metric, mask=mask)
    assert values.shape == z_range.shape
    assert abs(z_best + D) < dz


def test_autofocus_phase_object(grid):
    """A pure phase object has the most uniform amplitude in focus."""
    x, y, X, Y = grid
    obj = _band_limit(np.exp(1j*1.0*np.exp(-(X**2 + Y**2)/5**2)), x, y, 1.5)
    O = _propagate(obj, x, y, D)
    z_range = np.linspace(-280, -120, 17)
    z_best, _ = holo.autofocus(O, x, y, z_range, WL, metric="variance", object_type="phase")
    assert abs(z_best + D) < z_range[1] - z_range[0]


def test_autofocus_callable_metric(grid):
    x, y, X, Y = grid
    O = np.exp(-(X**2 + Y**2)/10**2) + 0j
    z_best, values = holo.autofocus(O, x, y, [-10, 0, 10], WL, metric=lambda U: np.max(np.abs(U)),
                                    refine=False)
    assert z_best == 0 and len(values) == 3


def test_focus_metric_mask(grid):
    x, y, X, Y = grid
    U = 1 + 0.5*(X > 0)
    assert holo.focus_metric(U, "variance", mask=X > 0) == 0
    assert holo.focus_metric(U, "variance") > 0
    with pytest.raises(ValueError):
        holo.focus_metric(U, "unknown")


# quantitative phase

def test_quantitative_phase_with_unwrapping(grid):
    """A phase bump of 6 pi with a residual carrier tilt: background
    compensation, unwrapping and tilt removal give the phase."""
    x, y, X, Y = grid
    phase = 6*np.pi*np.exp(-(X**2 + Y**2)/20**2)
    tilt = 0.05*X - 0.03*Y
    field = np.exp(1j*(phase + tilt))
    background = np.exp(1j*tilt)
    for method in ("ls", "quality"):
        result = holo.quantitative_phase(field, reference_field=background, unwrap=method)
        result -= result[0, 0]
        assert np.max(np.abs(result - (phase - phase[0, 0]))) < 1e-8

    tilt_mask = X**2 + Y**2 > 50**2
    result = holo.quantitative_phase(field, x, y, tilt_mask=tilt_mask)
    assert np.max(np.abs(result - phase)) < 1e-3*np.max(phase)
    assert np.allclose(holo.quantitative_phase(field, unwrap=None), np.angle(field))


def test_remove_tilt_and_opd(grid):
    x, y, X, Y = grid
    plane = 0.2 + 0.01*X - 0.02*Y
    corrected, coeffs = holo.remove_tilt(plane, x, y)
    assert np.allclose(corrected, 0, atol=1e-12)
    assert np.allclose(coeffs, [0.2, 0.01, -0.02])
    assert np.isclose(holo.optical_path_difference(2*np.pi, WL), WL)


def test_refocus_backwards_with_evanescent_corners():
    """With dx < wavelength/sqrt(2) the corners of the k grid are evanescent;
    backward refocusing must not amplify them."""
    x, y, X, Y = grid2d(-16, 16, -16, 16, 128)   # dx = 0.25 = wavelength/2
    obj = np.exp(-(X**2 + Y**2)/3**2)*np.exp(1j*np.exp(-(X**2 + Y**2)/2**2)) + 0j
    obj = _band_limit(obj, x, y, 0.9*2*np.pi/WL)
    there = holo.refocus(obj, x, y, 40.0, WL)
    back = holo.refocus(there, x, y, -40.0, WL)
    assert np.all(np.isfinite(back))
    assert np.max(np.abs(back - obj)) < 1e-3


def test_off_axis_spherical_reference(grid, off_axis):
    """A curved reference is removed exactly if it is passed as reference;
    treated as a plane wave it leaves a lens phase on O."""
    x, y, X, Y = grid
    k = 2*np.pi/WL
    R = holo.reference_wave(x, y, WL, off_axis["ref_angle"])*np.exp(1j*k*(X**2 + Y**2)/(2*5e3))
    H = np.abs(off_axis["O"] + R)**2
    kwargs = dict(ref_angle=off_axis["ref_angle"], radius=off_axis["B"], window="circle")
    O_ref = holo.reconstruct_off_axis(H, x, y, WL, reference=R, **kwargs)
    O_plane = holo.reconstruct_off_axis(H, x, y, WL, **kwargs)
    c = slice(N//4, 3*N//4)
    err = lambda O: np.max(np.abs(O - off_axis["O"])[c, c])
    assert err(O_ref) < 2e-3*np.max(np.abs(off_axis["O"]))
    assert err(O_plane) > 0.1*np.max(np.abs(off_axis["O"]))
