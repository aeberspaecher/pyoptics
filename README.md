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
| **Wave propagation** | Rayleigh-Sommerfeld, Fresnel and Fraunhofer propagators, band-limited and padded propagation, free output windows (shifted angular spectrum, chirp-z), vectorial fields and high NA focusing, propagation onto tilted planes, analytic handling of curved wavefronts (semi-analytical and homeomorphic Fourier transforms after Wyrowski, few samples at any distance and on tilted planes), wave propagation method (WPM) for inhomogeneous media, field tracing chains with automatic choice of the propagation operator per step, ray tracing through lens systems inside the chain, interferometer arms and Babinet obstructions on their own grids | `propagators`, `tilted`, `analytic_phase`, `fieldtrace`, `fft`, `wpm` |
| **Maxwell solvers** | Finite-difference time-domain (FDTD) solver in 1D, 2D and 3D with PML, TFSF plane waves, dispersive metals (Drude/Lorentz), optional numba kernels, DFT and flux monitors and animations, waveguide modes, mode sources and S-parameters for couplers and ring resonators, Bloch boundaries for gratings (diffraction orders) and photonic crystal band structures, photonic-crystal waveguides and cavities, grating couplers and metalenses, a 3D sphere against the vector Mie series and the photonic nanojet, a rigorous litho mask model for contact holes; finite-difference frequency-domain (FDFD) solver in 2D: driven solves and resonances (quasi-normal modes, Q) with PML and Bloch boundaries, mode tracing, far fields and directionality, a microdisk deformed into the limaçon of Pascal and its ray chaos, a quasi-conformal carpet cloak | `fdtd`, `fdfd`, `billiard`, `cloak` |
| **Beams and modes** | Hermite- and Laguerre-Gaussian beams, ABCD matrices and resonators, Collins integral, M² and beam widths after ISO 11146, mode decomposition, LP fibre modes and coupling efficiency, optical vortices and orbital angular momentum, Bessel and Airy beams, ultrashort pulses through lenses, prisms and gratings | `beams`, `abcd`, `beam_quality`, `modes`, `vortices`, `ultrafast` |
| **Fibre optics and communication** | Step and graded-index fibres in the ray picture, slab and fibre modes with the single-mode condition, mode field diameter and bend loss, material, waveguide and intermodal dispersion, pulse broadening and the bit-rate limit, fibre Bragg gratings from coupled modes, a fibre link with receiver noise, eye diagrams and the bit error rate; self-phase modulation, solitons, soliton fission and supercontinuum, the Raman self-frequency shift and SPM in a 10 Gb/s link with a general split-step (NLSE/GNLSE, RK4IP) solver | `fibres`, `fibrecomm`, `splitstep` |
| **Thin films and polarization** | Fresnel coefficients, coherent multilayer stacks, Jones and Mueller calculus, Stokes parameters, dispersion and crystal materials | `interfaces`, `thin_films`, `polarization`, `materials` |
| **Imaging and microscopy** | PSF, OTF/MTF and Strehl ratio with pupil masks and aberrations, Zernike, Legendre and Chebyshev bases, partially coherent vectorial imaging; microscopy: partially coherent bright field imaging (Abbe, Köhler), weak object transfer functions, Zernike phase contrast, Nomarski DIC, confocal PSFs and optical sectioning, Fourier ptychography with pupil recovery, super-resolution (SIM, STED, single molecule localization), optical coherence tomography, optical tweezers | `imaging`, `bases`, `vectorial_imaging`, `analytical`, `microscopy`, `superresolution`, `oct`, `tweezers` |
| **Illumination and radiometry** | Radiometry and photometry, the radiometry of images, étendue and phase space, nonimaging concentrators (CPC, edge-ray principle), Köhler and critical illumination, homogenizers, LED optics with Monte Carlo illuminance, freeform beam shaping, speckle reduction, colours of spectra (CIE XYZ, sRGB) | `radiometry`, `illumination`, `freeform`, `speckle`, `colour` |
| **Cameras and photography** | Bokeh, sunstars, flare and ghosts of real camera lenses, the sensor (pixels, Bayer mosaic, low pass filter, microlenses), depth of field and tilt-shift, the light field camera, lensless imaging with pinholes, coded apertures and diffuser cameras | `photography`, `sensor`, `lightfield`, `lensless` |
| **Astronomy and atmosphere** | Turbulence, scintillation and adaptive optics, coherence and stellar interferometry, coronagraphs, a telescope zoo, segmented mirrors, gravitational lensing, the atmospheric dispersion corrector, rainbows and glories | `turbulence`, `coherence`, `coronagraph`, `telescopes`, `segmented`, `lensing`, `adc`, `rainbow` |
| **Spectroscopy** | Grating and prism spectrometers (Czerny–Turner traced with grating surfaces), the echelle spectrograph, Fourier transform spectroscopy, Fabry–Perot interferometers and interference filters, spectral lines and absorption, imaging spectroscopy | `spectrometers`, `spectroscopy`, `echelle` |
| **AR/VR and head-up displays** | Waveguide combiners in k-space (field of view against the index, colour), in-couplers and out-couplers with RCWA (binary and slanted gratings), exit pupil expansion and the eye box, the rainbow artefact, fold gratings; volume holographic gratings with Kogelnik's coupled waves against RCWA; pancake lenses with real retarders and their ghosts; windshield HUDs: reflectance, the double image and the wedge, a ray traced freeform HUD mirror with vergence, astigmatism and disparity over the eye box | `arvr` |
| **3D sensing and LiDAR** | Time-of-flight LiDAR link budgets (Lambertian targets, fog, sunlight), linear APDs with thresholds and false alarms, SPAD histograms with pile-up and Coates' correction, class 1 eye safety at 905 and 1550 nm for pulse trains and scanned beams; FMCW LiDAR with range and Doppler, the shot noise limited coherent receiver, speckle fading, laser linewidth and chirp nonlinearity correction; beam scanners (galvos, polygons, resonant MEMS and Lissajous patterns, optical phased arrays, Risley prisms); laser triangulation and its speckle limit, fringe projection with temporal phase unwrapping, and the dot projector of depth cameras (IFTA fan-out gratings, block matching) | `lidar`, `structured_light` |
| **Surface and thin-film metrology** | Coherence scanning interferometry (white light interferometry, WLI): correlograms, envelope detection and frequency domain analysis with phase gap and fringe order, the obliquity factor, metals and transparent films, film thickness from the spectral signature, batwings at steps; spectroscopic ellipsometry: Psi and Delta of layered samples in the Nebraska conventions, the pseudo-dielectric function, rotating analyzer and rotating compensator instruments, Cauchy, Tauc-Lorentz, Lorentz, Drude and Bruggeman models, model fits with MSE and correlations; scatterometry (OCD) with RCWA: trapezoidal line profiles, Psi-Delta and Mueller signatures in planar and conical mounts, library search and regression; phase measuring deflectometry: screen and camera geometry, fringe decoding, iterative slope integration, the height-slope ambiguity | `csi`, `ellipsometry`, `scatterometry`, `deflectometry` |
| **Crystal optics and light modulators** | Plane waves in anisotropic crystals (the index ellipsoid, double refraction and walk-off), the 4×4 (Berreman) matrix method for stacks of anisotropic and gyrotropic layers at any angle, polarizing prisms (Glan–Taylor, Glan–Thompson, Wollaston, Rochon) traced through their interfaces, optical activity and Faraday isolators; Pockels and Kerr cells, the Mach–Zehnder modulator with chirp, extinction and travelling-wave bandwidth; nematic liquid crystals (Fréedericksz transition, director profiles from the Frank energy, retarders, TN cells with Gooch–Tarry and their viewing angle); acousto-optic modulators, deflectors and tunable filters (Raman–Nath and Bragg regimes, coupled orders, figure of merit, AOTF passband) | `crystals`, `electrooptics`, `liquid_crystals`, `acoustooptics` |
| **Lasers** | Einstein coefficients, cross sections and gain media, saturation and amplifiers (Frantz–Nodvik), three- and four-level rate equations, threshold, slope efficiency and optimum output coupling (Rigrod), longitudinal and transverse modes, the Schawlow–Townes linewidth, relaxation oscillations; active and passive Q-switching, phase-locked modes and frequency combs, active and soliton mode-locking from the Haus master equation; laser diodes from their rate equations: L–I curve, modulation bandwidth, chirp (α_H), RIN and linewidth with Langevin noise | `lasers` |
| **Nonlinear optics** | χ⁽²⁾ and χ⁽³⁾; second harmonic generation from the coupled amplitude equations (with depletion), birefringent type I and II phase matching in BBO and KDP with d_eff and walk-off from the index ellipsoid, quasi-phase matching in MgO:PPLN (temperature dependent Sellmeier, real domain patterns, higher orders, duty cycle), acceptance bandwidths in angle, wavelength and temperature, Boyd–Kleinman focusing with walk-off, femtosecond SHG with group velocity mismatch; sum and difference frequency generation, parametric gain and bandwidth, OPO tuning, SRO and DRO thresholds and the pump depletion of a singly resonant OPO; the Kerr effect and Kerr lens, self-focusing and collapse against Marburger, small-scale self-focusing and the B-integral, SPM, and Kerr-lens mode-locking with the laser master equation | `nonlinear`, `splitstep` |
| **Detectors and photon statistics** | Responsivity from the quantum efficiency, band gaps (Varshni, Hansen for HgCdTe) and cutoffs, absorption coefficients of Si, InGaAs and HgCdTe and the quantum efficiency of absorbing layers, photodiode bandwidth from capacitance and transit time, shot, thermal and APD excess noise, NEP, D*, background limited (BLIP) and R₀A limited detectivity, BER and receiver sensitivity, SPADs with dead time, Poisson, thermal and number state statistics, Mandel Q and g2, chaotic light, binomial losses, the beam splitter with number states (HBT, Hong–Ou–Mandel) | `detectors` |
| **Ray tracing and lens design** | Sequential ray trace with conics and aspheres, tilts and decentres, tolerance sensitivities (decentre, axial shift, radius, index), paraxial data and Seidel sums, OPD maps and Zernike fits, spot diagrams, polarization ray trace through coatings and crystals, curved rays in gradient-index media, damped least squares optimization (variables incl. model glasses, solves, Lagrange multiplier constraints, several configurations such as zoom positions with shared variables; the landscape of local minima: random starts, saddle points and their construction, basin hopping), Monte Carlo tolerancing (sensitivity tables, yields, linear model, compensators and manipulators with limited strokes, tilt, wedge, irregularity and inhomogeneity tolerances, inverse sensitivity and tolerance allocation), glass catalogues of six vendors with partial dispersions, Delano diagrams; conversion between ray ensembles and wave fields (ray field summation with Gouy phases, Gaussian beam decomposition through real lens surfaces, phase gradient and Wigner rays); Fresnel lenses with real facets (draft, shadowing and riser losses), exact facet design and kinoforms; diffractive surfaces in the ray trace (local grating equation, power, Seidel sums and colour of hybrid lenses, kinoform efficiency); lens design classics: mechanically compensated zoom lenses (a patent zoom with cams and inner focusing), Forbes Q-bfs and Q-con aspheres, progressive spectacle lenses traced as worn, F-theta and telecentric scan lenses behind XY galvo scanners with a lookup table correction; off-axis mirror systems (off-axis conic segments, aperture offset and field bias, beam clearance, ring fields, an unobscured three-mirror anastigmat with freeform mirrors) and nodal aberration theory (misalignment: constant coma, binodal astigmatism, the coma-free pivot; freeform terms) | `raytrace`, `raywave`, `fresnel`, `coatings`, `lens_optimization`, `lens_landscape`, `off_axis`, `nodal`, `tolerancing`, `glasses`, `delano`, `grin`, `zoom`, `qaspheres`, `spectacles`, `scan_lenses` |
| **Wavefront metrology** | Phase-shifting and Fourier fringe analysis, digital holography, Shack-Hartmann sensors, phase retrieval (Fienup, phase diversity), phase unwrapping, TIE, IFTA beam shaping, wavefront shaping through scattering media, Schlieren and the knife-edge test | `interferometry`, `holography`, `shack_hartmann`, `phase_retrieval`, `unwrap`, `TIE`, `ifta`, `wavefront_shaping`, `schlieren` |
| **Lithography** | Sources, masks and projectors (Abbe and Hopkins/SOCS imaging), thin and rigorous masks (RCWA for lines, 3D FDTD for any layout), resist and wafer stacks, CD, NILS, Bossung curves and process windows, ray traced projection lenses and illuminators, lens tolerances to CD uniformity, best focus and overlay with scanner manipulators, Jones pupils, source optimization and OPC, stochastics (photon and acid shot noise, LER and LWR with their power spectra, local CDU and missing contacts), flare from lens roughness and pattern density, focus blur from laser bandwidth and stage vibration, EUV (Mo/Si multilayer mirrors, reflective masks with shadowing and mask 3D effects, anamorphic high-NA pupils with central obscuration, EUV stochastics and flare) | `litho` |

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
| `hdf5` | h5py | streaming large FDTD snapshot stacks to compressed HDF5 files |
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

The [`notebooks/`](notebooks/README.md) folder holds 129 executed Jupyter
notebooks, one per topic, from free space propagation and Gaussian beams over
thin films, imaging, ray tracing and interferometry, Maxwell solvers,
microscopy, illumination, photography, astronomy, spectroscopy and fibre
optics to a complete lithography workflow with lens design, source
optimization and OPC. They are stored with
their outputs, so the figures show directly on GitHub, and they are rendered
in the [documentation](https://aeberspaecher.github.io/pyoptics/examples.html).
Most of them compare the numerics with analytic results. The
[learning paths](https://aeberspaecher.github.io/pyoptics/learning_paths.html)
suggest reading orders for different backgrounds.

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

The numerics are tested against analytical results by more than 2000 pytest tests,
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

The test suite (more than 2000 tests) takes several minutes. After editing `pyoptics/wpm.pyx`,
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
Copyright 2014–2026 Alexander Eberspächer.
