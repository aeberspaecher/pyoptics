import numpy as np
import pytest

from pyoptics import sensor as se


P = 0.004


def test_pixel_and_diffraction_mtf():
    assert se.pixel_mtf(se.nyquist(P), P) == pytest.approx(2/np.pi)
    assert se.pixel_mtf(1/P, P) == pytest.approx(0.0, abs=1e-12)
    assert se.pixel_mtf(1/P, P, fill=0.25) == pytest.approx(2/np.pi)
    assert se.diffraction_mtf(0.0, 550e-6, 8) == pytest.approx(1.0)
    assert se.diffraction_mtf(1/(550e-6*8), 550e-6, 8) == pytest.approx(0.0, abs=1e-12)


def test_sensor_image_box_average():
    os_ = 4
    fine = np.arange(64, dtype=float).reshape(8, 8)
    img = se.sensor_image(fine, os_, 1.0)
    assert img.shape == (2, 2)
    assert img[0, 0] == pytest.approx(fine[:4, :4].mean())


def test_slanted_edge_matches_sinc():
    img = se.slanted_edge(64, np.radians(5))
    f, m, ang = se.slanted_edge_mtf(img)
    assert np.degrees(ang) == pytest.approx(5.0, abs=0.02)
    for q in (0.1, 0.25, 0.5, 0.75):
        assert np.interp(q, f, m) == pytest.approx(abs(np.sinc(q)), abs=0.01)


def test_aliasing_folds():
    os_, n = 8, 128
    x = (np.arange(n*os_) + 0.5)/os_
    for f_in in (0.3, 0.8, 1.3):
        row = se.sensor_image(np.tile(np.cos(2*np.pi*f_in*x), (os_, 1)), os_, 1.0)[0]
        k = np.argmax(np.abs(np.fft.rfft(row*np.hanning(n))))
        assert np.fft.rfftfreq(n)[k] == pytest.approx(abs(f_in - round(f_in)), abs=1/n)


def test_bayer_mosaic_and_demosaic_flat():
    rgb = np.ones((3, 16, 16))*np.array([0.2, 0.5, 0.8])[:, None, None]
    raw = se.mosaic(rgb)
    M = se.bayer_masks(raw.shape)
    assert M.sum(axis=0).max() == 1 and M[1].sum() == 128
    assert raw[0, 0] == 0.2 and raw[1, 1] == 0.8
    for fn in (se.demosaic_bilinear, se.demosaic_malvar):
        assert np.allclose(fn(raw), rgb)
    for pat in ("BGGR", "GRBG", "GBRG"):
        assert np.allclose(se.demosaic_malvar(se.mosaic(rgb, pat), pat), rgb)


def test_zone_plate_false_colour_at_carriers():
    n, os_ = 128, 4
    u = (np.arange(n*os_) + 0.5)/os_ - n/2
    U, V = np.meshgrid(u, u)
    r0 = np.sqrt(n/2)
    grey = se.sensor_image(se.zone_plate(U, V, r0), os_, 1.0)
    rgb = se.demosaic_malvar(se.mosaic(np.array([grey]*3)))
    fc = se.false_colour(rgb)
    x0, _ = se.zone_plate_position(0.5, 0.0, r0)
    c = n//2
    near = fc[c - 3:c + 4, c + int(x0) - 3:c + int(x0) + 4].mean()
    centre = fc[c - 8:c + 8, c - 8:c + 8].mean()          # low frequencies: almost no false colour
    assert near > 0.1 and centre < 0.15*near
    assert np.allclose(se.cfa_carriers(P), [[125, 0], [0, 125], [125, 125]])


def test_walkoff_formula_matches_tensor():
    for no, ne in (se.QUARTZ, se.LITHIUM_NIOBATE):
        for th in np.radians([10, 45, 70]):
            S = se.poynting_direction(no, ne, [np.sin(th), 0, np.cos(th)], [0, 0, 1])
            assert np.arctan2(S[0], S[2]) == pytest.approx(se.walkoff_angle(no, ne, th), abs=1e-12)
    t = se.olpf_thickness(P)
    assert t*np.tan(se.walkoff_angle(*se.QUARTZ)) == pytest.approx(P)
    assert 0.6 < t < 0.75


def test_olpf_four_spots_and_mtf():
    t = se.olpf_thickness(P)
    two = se.olpf_spots([se.OLPFPlate(t, 0.0), se.OLPFPlate(t, np.pi/2)])
    assert len(two) == 2 and np.allclose(two[:, 2], 0.5)
    four = se.olpf_spots([se.OLPFPlate(t, 0.0), se.WavePlate(np.pi/2, np.pi/4), se.OLPFPlate(t, np.pi/2)])
    assert len(four) == 4 and np.allclose(four[:, 2], 0.25)
    assert sorted(map(tuple, np.round(four[:, :2]/P, 6))) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    f = np.linspace(0, 300, 31)
    H = se.olpf_otf(f, 40.0, four)
    assert np.allclose(np.abs(H), np.abs(np.cos(np.pi*f*P)*np.cos(np.pi*40.0*P)))


def test_microlens_on_axis_and_shift():
    st = se.pixel_stack(P)
    rng = np.random.default_rng(0)
    e, x, _ = se.trace_pixel(st, 0.0, 8.0, 20000, rng)
    assert e == pytest.approx(np.pi/4, abs=0.015)       # the round microlens's share of the pixel
    assert x == 0.0
    xc, rad = se.pupil_image(st, 0.0, 8.0)
    assert np.pi/4*se.disk_square_overlap(xc, 0.0, rad, st.pd_half_width) == pytest.approx(np.pi/4)
    cra = np.radians(15)
    e_off = se.trace_pixel(st, cra, 8.0, 20000, rng)[0]
    e_shift = se.trace_pixel(st._replace(shift=se.microlens_shift(st, cra)), cra, 8.0, 20000, rng)[0]
    assert e_off < 0.2 and e_shift > 0.7


def test_disk_square_overlap():
    assert se.disk_square_overlap(0, 0, 1.0, 10.0) == pytest.approx(1.0)
    assert se.disk_square_overlap(0, 0, 1.0, 1/np.sqrt(2)) == pytest.approx(2/np.pi, rel=1e-6)
    assert se.disk_square_overlap(5, 0, 1.0, 1.0) == 0.0
    assert se.disk_square_overlap(1.0, 0, 0.5, 1.0) == pytest.approx(0.5, rel=1e-6)
