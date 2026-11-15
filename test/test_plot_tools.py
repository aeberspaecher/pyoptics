import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyoptics.plot_tools import show, complex_to_rgb


def test_show_symmetric_and_colorbar():
    fig, ax = plt.subplots()
    img = np.array([[-1.0, 0.5], [np.nan, 2.0]])
    im = show(ax, img, "t", symmetric=True, ticks=False)
    assert im.get_clim() == (-2.0, 2.0)
    assert ax.get_title() == "t" and len(ax.get_xticks()) == 0
    assert len(fig.axes) == 2    # the colorbar
    plt.close(fig)


def test_complex_to_rgb():
    U = np.array([[1.0, -1.0], [1j, 0.0]])
    rgb = complex_to_rgb(U)
    assert rgb.shape == (2, 2, 3)
    assert np.allclose(rgb[1, 1], 0.0)          # zero amplitude is black
    assert np.isclose(rgb[0, 0].max(), 1.0)      # full amplitude, full value


def test_import_keeps_rcparams():
    import importlib
    import pyoptics.plot_tools as pt
    before = dict(matplotlib.rcParams)
    importlib.reload(pt)
    assert matplotlib.rcParams["font.size"] == before["font.size"]
