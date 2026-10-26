"""The compiled ray trace kernel (numba, optional) against the NumPy trace."""

import numpy as np
import pytest

from pyoptics import raytrace as rt
from pyoptics import _raytrace_numba

pytestmark = pytest.mark.skipif(not _raytrace_numba.available, reason="numba not installed")


@pytest.fixture
def numpy_only():
    old = rt.use_numba
    rt.use_numba = False
    yield
    rt.use_numba = old


def _mixed_system():
    # mirror with conic and asphere, tilts, decentre, ideal lens and cone
    surfaces = [rt.Surface(R=50, thickness=5, n_after=1.5, semi_diameter=10),
                rt.Surface(R=-50, thickness=20, n_after=1.0, semi_diameter=10),
                rt.CoordinateBreak(tilt=(0.05, 0.0, 0.0)),
                rt.Surface(R=-200, thickness=-30, mirror=True, conic=-0.5, aspheric=(1e-6,),
                           semi_diameter=15, decenter=(0.5, 0.0)),
                rt.IdealLens(-40, thickness=-50, semi_diameter=20, tilt=(0.0, 0.02, 0.0)),
                rt.Axicon(0.3, thickness=-10, n_after=1.4, semi_diameter=20),
                rt.Surface(thickness=-10, n_after=1.0)]
    return rt.System(surfaces, 0, 0.5e-3, field_angle=0.02)


SYSTEMS = {
    "zeiss_arfi": rt.zeiss_arfi_lens,  # aspheres, high NA
    "caldwell_krf": rt.caldwell_krf_lens,  # finite object
    "zoom_axicon": lambda: rt.zeiss_zoom_axicon(axicon_gap=20.0),
    "mixed": _mixed_system,
}


@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("clip", [True, False])
def test_kernel_matches_numpy(name, clip):
    system = SYSTEMS[name]()
    _, PX, PY, inside = system._pupil_grid(24)
    # beyond the rim: vignetted rays and rays that miss surfaces
    rays = system.rays(None, 1.1*PX[inside], 1.1*PY[inside], aim=False)
    old = rt.use_numba
    try:
        rt.use_numba = False
        ref = system.trace(rays, clip=clip)
        rt.use_numba = True
        new = system.trace(rays, clip=clip)
    finally:
        rt.use_numba = old
    assert np.array_equal(new.valid, ref.valid)
    assert 0 < np.count_nonzero(ref.valid)
    for a, b in ((new.points, ref.points), (new.directions, ref.directions), (new.opl, ref.opl)):
        assert np.array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a[np.isfinite(b)], b[np.isfinite(b)], rtol=0, atol=1e-10)


def test_kernel_stops_at_last_surface():
    system = rt.zeiss_arfi_lens()
    rays = system.rays((0.0, 0.0), np.linspace(-1, 1, 7), 0.0)
    a = system.trace(rays, clip=False, last=system.stop_index)
    assert a.points.shape[0] == system.stop_index + 2


def test_aimed_rays_agree(numpy_only):
    system = rt.caldwell_krf_lens()
    px = np.linspace(-1, 1, 9)
    ref = system.rays(None, px, 0.5*px)
    rt.use_numba = True
    new = system.rays(None, px, 0.5*px)
    np.testing.assert_allclose(new.directions, ref.directions, rtol=0, atol=1e-12)
