# pyoptics

[![tests](https://github.com/aeberspaecher/pyoptics/actions/workflows/tests.yml/badge.svg)](https://github.com/aeberspaecher/pyoptics/actions/workflows/tests.yml)
[![docs](https://github.com/aeberspaecher/pyoptics/actions/workflows/docs.yml/badge.svg)](https://aeberspaecher.github.io/pyoptics/)
![Python 3.9 to 3.13](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

<!-- docs:about-start -->
**Wave optics, ray optics and lithography simulation in Python.**

pyoptics is a toolbox for optical simulation built on NumPy and SciPy: scalar
and vectorial wave propagation, Gaussian beams and modes, thin films and
polarization, imaging and aberrations, sequential ray tracing, wavefront
metrology and a full lithography imaging stack. Its blocks are small and
independent, so a propagator, a Zernike basis or a ray traced lens can be
reused in new contexts. The numerics are checked against analytic results
throughout.
<!-- docs:about-end -->

![Fresnel diffraction behind a square aperture, a ray traced lithography lens, and the aerial image of contact holes](docs/images/readme_overview.png)

**[Documentation](https://aeberspaecher.github.io/pyoptics/)** ·
**[Example notebooks](notebooks/README.md)** ·
**[API reference](https://aeberspaecher.github.io/pyoptics/api/index.html)** ·
**[Conventions](https://aeberspaecher.github.io/pyoptics/conventions.html)**

## Features

| Area | What you get | Modules |
|---|---|---|
| **Wave propagation** | Rayleigh-Sommerfeld, Fresnel and Fraunhofer propagators, band-limited and padded propagation, free output windows (shifted angular spectrum, chirp-z), vectorial fields and high NA focusing, propagation onto tilted planes, wave propagation method (WPM) for inhomogeneous media | `propagators`, `tilted`, `fft`, `wpm` |
| **Beams and modes** | Hermite- and Laguerre-Gaussian beams, ABCD matrices and resonators, Collins integral, M² and beam widths after ISO 11146, mode decomposition, LP fibre modes and coupling efficiency | `beams`, `abcd`, `beam_quality`, `modes` |
| **Thin films and polarization** | Fresnel coefficients, coherent multilayer stacks, Jones and Mueller calculus, Stokes parameters, dispersion and crystal materials | `interfaces`, `thin_films`, `polarization`, `materials` |
| **Imaging** | PSF, OTF/MTF and Strehl ratio with pupil masks and aberrations, Zernike, Legendre and Chebyshev bases, partially coherent vectorial imaging | `imaging`, `bases`, `vectorial_imaging`, `analytical` |
| **Ray tracing and lens design** | Sequential ray trace with conics and aspheres, tilts and decentres, paraxial data and Seidel sums, OPD maps and Zernike fits, spot diagrams, polarization ray trace through coatings and crystals, damped least squares optimization, Delano diagrams | `raytrace`, `coatings`, `lens_optimization`, `delano` |
| **Wavefront metrology** | Phase-shifting and Fourier fringe analysis, digital holography, Shack-Hartmann sensors, phase retrieval (Fienup, phase diversity), phase unwrapping, TIE, IFTA beam shaping | `interferometry`, `holography`, `shack_hartmann`, `phase_retrieval`, `unwrap`, `TIE`, `ifta` |
| **Lithography** | Sources, masks and projectors (Abbe and Hopkins/SOCS imaging), thin and rigorous (RCWA) masks, resist and wafer stacks, CD, NILS, Bossung curves and process windows, ray traced projection lenses and illuminators, Jones pupils, source optimization and OPC | `litho` |

The [module overview](https://aeberspaecher.github.io/pyoptics/modules.html)
in the documentation describes every module in detail.

## Installation

pyoptics is not on PyPI yet. It has one Cython extension, so you need a C
compiler (`build-essential` on Debian/Ubuntu, `xcode-select --install` on
macOS, the Microsoft C++ Build Tools on Windows). Then:

```bash
git clone https://github.com/aeberspaecher/pyoptics.git
cd pyoptics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[plot]"
```

Optional extras can be combined, for example `".[plot,fast,test]"`:

| Extra | Installs | For |
|---|---|---|
| `plot` | matplotlib | plotting tools |
| `fast` | numba | compiled ray trace kernel, several times faster ray traces and ray aiming |
| `nufft` | finufft | faster exact propagation onto planes tilted about a general axis |
| `tie` | sympy | the TIE module |
| `notebooks` | matplotlib, JupyterLab, ipywidgets | running the example notebooks |
| `test` | pytest, matplotlib, sympy | running the test suite |
| `docs` | Sphinx and extensions | building the documentation |

The [installation guide](https://aeberspaecher.github.io/pyoptics/install.html)
walks through every step on Linux, macOS and Windows and covers common build
errors.

## Quickstart

Propagate a Gaussian beam by one Rayleigh range with the angular spectrum
method and compare it with the analytic beam (lengths in µm):

```python
import numpy as np
from pyoptics import propagators as prop
from pyoptics.beams import gauss_laguerre
from pyoptics.utils import grid2d

wl, w0 = 0.5, 5.0                      # wavelength and waist [µm]
z_R = np.pi*w0**2/wl                   # Rayleigh range
x, y, X, Y = grid2d(-80, 80, -80, 80, 512)

U0 = gauss_laguerre(0, 0, x, y, 0.0, w0, z_R, wl)          # waist at z = 0
U, _, _ = prop.propagate_padded(prop.rayleigh_sommerfeld_I_TF, U0, x, y, z_R, wl)
U_exact = gauss_laguerre(0, 0, x, y, z_R, w0, z_R, wl)

print(f"max. error: {np.abs(U - U_exact).max()/np.abs(U_exact).max():.1e}")
# max. error: 2.5e-04
```

Ray trace a plano-convex BK7 singlet (f = 100 mm, f/5) and look at its
spherical aberration (lengths in mm):

```python
import numpy as np
from pyoptics import raytrace as rt

BK7 = (1.5168, 64.17)                  # (n_d, V_d)
lens = rt.System([rt.Surface(51.68, 5.0, material=BK7, semi_diameter=10.5),
                  rt.Surface(np.inf, 95.0, n_after=1.0, semi_diameter=10.5)],
                 0, 587.6e-6, entrance_pupil_radius=10.0).at_paraxial_focus()

print(f"EFL:           {lens.paraxial().efl:.2f} mm")
print(f"W040 (Seidel): {lens.seidel(0.0).W040:.2f} waves")
print(f"RMS OPD:       {lens.rms_opd(0.0):.3f} waves, Strehl {lens.strehl(0.0):.2f}")
# EFL:           100.00 mm
# W040 (Seidel): -4.62 waves
# RMS OPD:       1.388 waves, Strehl 0.03
```

Image 45 nm lines and spaces with an ArF water immersion scanner (lengths in
nm):

```python
from pyoptics.litho import Illumination, MaskLayout, ThinMask, Projector, LithoSimulation

sim = LithoSimulation(MaskLayout.lines_spaces(pitch=90.0, cd=45.0),
                      ThinMask.binary(),
                      Illumination.dipole_for_pitch(90.0, 193.0, 1.35, polarization="y"),
                      Projector(NA=1.35, wavelength=193.0, n=1.437))
print(f"NILS {sim.nils():.2f}, dose to size {sim.dose_to_size(threshold=0.3):.2f}")
# NILS 2.82, dose to size 0.85
```

## Example notebooks

The [`notebooks/`](notebooks/README.md) folder holds 43 executed Jupyter
notebooks, one per topic, from free space propagation and Gaussian beams over
thin films, imaging, ray tracing and interferometry to a complete lithography
workflow with lens design, source optimization and OPC. They are stored with
their outputs, so the figures show directly on GitHub, and they are rendered
in the [documentation](https://aeberspaecher.github.io/pyoptics/examples.html).
Most of them compare the numerics with analytic results.

To run them yourself, install the `notebooks` extra and start JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/
```

## Status

<!-- docs:status-start -->
pyoptics is version 0.1, a research and teaching toolbox rather than a
polished library. It runs on Python 3.9 to 3.13. The API may still change
between versions, and there is no release on PyPI yet.

The numerics are tested against analytical results by about 900 pytest tests,
which run in GitHub Actions on Python 3.9, 3.11 and 3.13 for every push and
pull request. The few unfinished parts are listed in the module overview,
planned work in
[`WIP.md`](https://github.com/aeberspaecher/pyoptics/blob/master/WIP.md).
<!-- docs:status-end -->

## Conventions

pyoptics uses the time dependence exp(−iωt) throughout, so a plane wave
travelling in +z is exp(+ikz), and `fft.FT()` works on centered arrays. Sign
conventions for lenses, ABCD matrices, ray tracing, absorbing media and
polarization are collected on the
[conventions page](https://aeberspaecher.github.io/pyoptics/conventions.html);
read it before comparing results with a textbook.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest test
```

The test suite (about 900 tests) takes one to two minutes. After editing `pyoptics/wpm.pyx`,
run the install command again to recompile the extension. Docstrings follow
the NumPy style; the documentation is built with Sphinx from `docs/`:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The figure at the top of this README is made by
[`docs/images/make_readme_figure.py`](docs/images/make_readme_figure.py).

## License

pyoptics is released under the [BSD 3-Clause License](LICENSE).
Copyright 2014-2017 Alexander Eberspächer.
