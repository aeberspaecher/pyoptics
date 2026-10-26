"""Make the overview figure at the top of README.md.

Run from the repository root: python docs/images/make_readme_figure.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyoptics import propagators as prop
from pyoptics import raytrace as rt
from pyoptics.masks import rectangluar_mask
from pyoptics.utils import grid2d
from pyoptics.litho import Illumination, MaskLayout, ThinMask, Projector, LithoSimulation
from pyoptics.litho.plot import plot_image

plt.rcParams.update({"font.size": 10, "image.origin": "lower"})
fig = plt.figure(figsize=(15, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.4, 1], wspace=0.3)

# Fresnel diffraction behind a square aperture, Fresnel number 5
wl, a, N = 0.5, 250.0, 1024
dx = 2*a/251
x, y, _, _ = grid2d(-N*dx/2, N*dx/2, -N*dx/2, N*dx/2, N)
z = a**2/(wl*5)
U, _, _ = prop.propagate_padded(prop.fresnel_TF, rectangluar_mask(x, y, a, a).astype(complex),
                                x, y, z, wl, pad_factor=2)
ax = fig.add_subplot(gs[0])
ax.imshow(np.abs(U)**2, extent=(x[0], x[-1], y[0], y[-1]), cmap="inferno")
ax.set_xlim(-400, 400); ax.set_ylim(-400, 400)
ax.set_title("Fresnel diffraction")
ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")

# Caldwell's KrF lithography lens, ray traced
ax = fig.add_subplot(gs[1])
rt.caldwell_krf_lens().draw(ax, n_rays=5)
ax.set_title("Ray trace: Caldwell's KrF projection lens (NA 0.56)")
ax.set_aspect("auto"); ax.set_ylim(-130, 130)
ax.set_xlabel("z [mm]"); ax.set_ylabel("y [mm]")

# aerial image of dense contact holes, ArF water immersion
sim = LithoSimulation(MaskLayout.contact_holes(130.0, 65.0), ThinMask.binary(),
                      Illumination.quasar(0.95, 0.7, polarization="TE"),
                      Projector(1.35, 193.0, n=1.437))
xs = np.linspace(-130, 130, 121)
ax = fig.add_subplot(gs[2])
plot_image(sim, xs, sim.image(xs, xs), xs, ax=ax, threshold=0.3, dose=sim.dose_to_size(threshold=0.3),
           colorbar=False)
ax.set_title("Aerial image, 65 nm contacts")
ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readme_overview.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
