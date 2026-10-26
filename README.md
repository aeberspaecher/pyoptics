pyoptics
========

About
-----

<!-- docs:about-start -->
pyoptics is a collection of tools for wave optical simulation. It contains
classes for imaging simulations and wave propagation. pyoptics has a modular
structure that easily allows reuse of individual blocks in new contexts.
<!-- docs:about-end -->

Status
------

<!-- docs:status-start -->
pyoptics is version 0.1, a research and teaching toolbox rather than a
polished library. It runs on Python 3.9 to 3.13. The API may still change
between versions, and there is no release on PyPI; install it from source as
described below.

The core is tested: a suite of about 630 pytest tests checks the numerics
against analytical results (Fresnel and Fraunhofer diffraction, Gaussian beam
propagation, Mahajan's on-axis intensity, Airy pattern and OTF, Fresnel
equations, thin film stacks, Goos-Hänchen shift, beam widths and M² of
Hermite-Gaussian beams, Mueller and Jones calculus, FFT conventions,
Richards-Wolf integrals for high NA foci, two-beam interference of TE and
TM waves, in air and in a resist, standing waves in a resist on silicon,
threshold CDs and process windows of analytic images, apodized two-beam
images, ABCD matrices against lens and resonator formulas and the Collins
integral against Gaussian beams, fields on tilted planes against plane
waves and Gaussian beams at the plane's 3D points, power flux through
tilted planes, phase unwrapping of wrapped Zernike
wavefronts, vortices and noisy wavefronts, phase-shifting and Fourier
fringe analysis of known Zernike phases, holograms of known objects
reconstructed, refocused and autofocused, exact ray traces against Snell's
law, perfect conics, Seidel sums and the Airy pattern, tilted plates and
fold mirrors, Fienup's HIO and phase diversity recovering known objects and
Zernike pupil phases, HG/LG mode decompositions and M² from the mode content
against ISO caustic fits, LP fibre modes against cutoffs and the Marcuse
mode field radius, fibre coupling against the Gaussian formulas) and runs in GitHub
Actions on Python 3.9, 3.11 and 3.13 for every push and pull request. A
consistent phase convention is used throughout (see Conventions).

Status of the modules:

- `propagators`: mature and tested (Rayleigh-Sommerfeld, Fresnel, Fraunhofer,
  band-limited and padded propagation; vectorial propagation of [E_x, E_y]
  with the longitudinal component E_z, any scalar propagator vectorized,
  Mansuripur's high NA focusing, checked against the Richards-Wolf integrals;
  propagation onto free output windows: shifted angular spectrum method with
  Matsushima's band limit, scaled Fresnel and Fraunhofer integrals, far field
  in chosen directions)
- `tilted`: propagation onto tilted planes, tested: scalar and vectorial
  fields from z = 0 onto a plane with any rotated normal and back (tilted
  source to parallel target), sampled in the plane's own coordinates; exact
  non-uniform plane wave sum (separable for tilts about x or y, finufft or a
  direct sum otherwise) and Matsushima's spectrum rotation with the
  Jacobian; carrier removal, power flux and ISO 11146 beam widths on
  inclined planes (detectors); `tilted_plane()` and
  `vectorial_tilted_plane()` are also available from `propagators`
- `fft`, `utils`, `masks`, `geometry`, `analytical`: mature and tested
  (`fft.czt()`: Bluestein's chirp-z transform; `fft.zoom_FT()` and
  `zoom_inv_FT()` evaluate `FT()` and `inv_FT()` on any equidistant grid)
  (`utils.pad()`/`unpad()` for centered zero padding; `analytical` holds the
  Airy pattern, its encircled energy and the diffraction limited OTF)
- `bases`: mature (Zernike, Legendre, Chebyshev bases, numerical
  orthogonalization, fits); all bases can be shifted and rotated, Legendre and
  Chebyshev derive from the abstract `TensorProductBasis`; tested
- `beams`: tested (plane, spherical, Hermite-Gaussian and Laguerre-Gaussian
  beams)
- `interfaces`, `thin_films`: mature and tested (Fresnel coefficients,
  coherent multilayer stacks; `stack_transfer_functions()` gives r and t on
  the angular spectrum grid of a field)
- `materials`: Abbe (n_d, V_d) dispersion model via a two-term Cauchy
  formula, tested
- `polarization.mueller`, `polarization.jones`: Mueller calculus and basic
  Jones vectors and matrices, tested
- `vectorial_imaging`: vectorial, partially coherent imaging (Abbe's method)
  of 1D and 2D masks with conventional, annular, dipole and quasar sources,
  TE/TM/unpolarized/partially polarized illumination, contrast and NILS,
  tested (two-beam contrast of TE and TM waves)
- `litho`: lithography imaging toolbox built on `vectorial_imaging`,
  tested: illumination sources (conventional, annular, dipole, quadrupole,
  quasar, multipole with sector or circular poles, freeform maps) for any
  sigma and any polarization (x, y, TE/azimuthal, TM/radial, unpolarized,
  Jones, Stokes, degree of polarization); mask layouts with analytic
  diffraction orders (lines and spaces, isolated lines, trenches, dense,
  isolated and staggered contact holes, any rectangles, clear or dark
  field); mask stacks in the thin mask approximation (binary, attenuated
  PSM), with absorber transmissions from `thin_films` (angle and TE/TM
  dependent) and an optional boundary layer model for thick mask edges;
  projector with immersion (or dry), vectorial or scalar model, Fringe
  Zernike aberrations and pupil apodization (amplitude transmission
  1 + sum of Fringe Zernikes, optionally as log amplitude, or any
  transmission function); images in the resist on a thin film wafer stack (BARC, top
  coat, substrate; TE/TM Fresnel transmission, standing waves); metrics
  (contrast, ILS/NILS, threshold CD, dose to size, MEEF, focus-exposure
  matrix, Bossung curves, process window, exposure latitude vs. depth of
  focus, Gaussian resist blur) and plots (source, mask, pupil fill, images,
  Bossung curves, process windows); `litho.lens` hands a ray traced lens
  (`raytrace`) to the projector (Fringe Zernikes on the projector's pupil,
  distortion, telecentricity, placement of a small feature), computes CD,
  placement and best focus across a scanner slit with scan averaging and
  the linear scanner correction, and Zernike sensitivities with a quick
  aberration budget; `litho.hopkins` computes transmission cross
  coefficients (TCC) and their SOCS decomposition, equal to Abbe's method
  for thin masks and fast for many masks with the same optics; swing
  curves and BARC optimization (`litho.wafer`, absorbed energy exact
  against `thin_films`) and `litho.resist`, a chemically amplified resist
  in an x-z cross section (exposure, PEB with acid diffusion, quencher and
  amplification, Mack development with a fast marching front; profile, CD
  at top/middle/bottom, sidewall angle, top loss, dose to size, E0);
  `litho.rcwa`: rigorous mask diffraction of 1D layouts by RCWA (Li's
  factorization, conical incidence, enhanced transmittance matrices; exact
  against `thin_films`, energy conserving), `RigorousMask` for Abbe and
  Hopkins imaging and a calibration of the boundary layer model;
  `litho.jones_pupil`: polarization of the lens as Jones pupils in the
  representations of the literature (Pauli-Zernike, polar decomposition,
  retardance and diattenuation in orientation Zernike polynomials) with
  fits, used by `Projector(jones_pupil=...)` in Abbe and Hopkins imaging,
  and a polarization ray trace of coated or uncoated lenses with optional
  CaF2 elements and their intrinsic birefringence
  (`litho.lens.lens_jones_pupil` on top of `raytrace.System.trace_polarization`,
  `materials.CubicCrystal`);
  `litho.source_opt`: source optimization of a pixelated source with
  symmetry, pupil fill and common dose constraints, maximizing the worst
  NILS over clips and foci by linear programming on the Abbe responses;
  `litho.opc`: optical proximity correction of Manhattan polygons
  (`MaskLayout.from_polygons`, exact unions of rectangles): bias tables
  through pitch, SRAF rules with a printability check, fragmentation, edge
  placement errors and model-based OPC with a regularized Jacobian update
  (Abbe, Hopkins or SOCS imaging)
- `polarization`: TE/TM bases, Stokes to Jones decomposition, DOP and IPS
- `imaging`: scalar imaging, tested: coherent imaging through a circular
  pupil, PSF with pupil masks (obscuration, apodization) and phases (also
  zoomed onto any output window via the chirp-z transform), OTF and
  MTF, Strehl ratio and Maréchal approximations
- `abcd`: ABCD (ray transfer) matrices, tested: thin and thick lenses,
  spherical interfaces, tilted mirrors (tangential/sagittal), GRIN media;
  `OpticalSystem` builds systems element by element and traces rays and
  Gaussian beams (beam radius, wavefront radius, Gouy phase) anywhere, also
  inside elements; cardinal points and images; complex beam parameter q and
  the ABCD law (also in media), mode matching efficiency; resonator
  stability, eigenmodes and round trip Gouy phase; `collins()` propagates any
  field through an ABCD system (Collins integral, separate x and y matrices
  for astigmatic systems)
- `unwrap`: 2D phase unwrapping, tested: least squares unwrapping with a
  DCT Poisson solver (Ghiglia-Romero), weighted least squares with
  modulation or amplitude weights and masks (conjugate gradients
  preconditioned by the DCT solver), quality guided unwrapping by
  reliability sorting (Herráez), unwrapping with a reference estimate
  (e.g. a TIE phase) and two-wavelength (synthetic wavelength) unwrapping,
  residue maps, `wrap()` and a 1D helper
- `interferometry`: fringe analysis, tested: simulated interferograms
  (Zernike test wavefronts, tilt carrier, visibility, camera noise via
  `processing.CameraNoiser`, miscalibrated phase shifters); phase-shifting
  algorithms (N-step least squares for arbitrary known shifts, 4-step,
  Carré with unknown step, Hariharan's 5-step), each returning the wrapped
  phase and the modulation; Takeda's Fourier-transform method for single
  interferograms with a carrier (sideband window, automatic carrier search,
  masks, apodization, overlap warning); phase error vs. shifter
  miscalibration
- `holography`: digital holography, tested: off-axis and in-line holograms
  (tilted reference, phase shifts; camera noise via `processing.CameraNoiser`),
  sampling condition for the reference angle; reconstruction by sideband
  filtering (circular, Tukey, Hann, Gaussian windows, automatic carrier
  search), phase-shifting holography (via `interferometry.psi_n_step()`) and
  Gabor holography with iterative twin image suppression; zero order
  suppression; numerical refocusing (angular spectrum or Fresnel, padded);
  autofocus (amplitude variance, Tamura coefficient, gradient, Laplacian);
  quantitative phase with background compensation, tilt removal and
  unwrapping (`unwrap`)
- `raytrace`: sequential ray tracing, tested: spherical, conic and even
  aspheric surfaces, lenses and mirrors, ideal (paraxial) lenses, apertures
  and vignetting, indices from constants, `materials.abbe_dispersion` or any
  function (fused silica after Malitson); decentred and tilted surfaces,
  coordinate breaks (fold mirrors), tilted image planes, fields in any
  direction, object space telecentric systems; exact intersections and
  vector Snell's law, total internal reflection; ray aiming at the real
  stop; paraxial
  marginal and chief rays, cardinal points, entrance and exit pupils (system
  matrix identical to `abcd`, `to_abcd()`), Seidel sums; OPD maps over the
  pupil relative to the reference sphere on the chief ray, Fringe Zernike
  fits, RMS OPD and Strehl ratio; PSF and MTF via `imaging`; spot diagrams,
  ray fans and layout plots; example: Caldwell's all fused silica KrF
  lithography projection lens (NA 0.56, 4x) with decentre and axial shift
  sensitivities of its elements (`with_axial_shift()`); polarization ray
  trace with energy (`System.trace_polarization()`, Chipman's 3x3 matrices):
  Fresnel refraction and reflection through thin film coatings on any
  surface (`coatings.Coating`, `coatings.Layer`) with layer thickness
  variations across the surface (radial, xy or Zernike polynomials),
  coated metal mirrors, absorbing (`materials.Isotropic`, n + ik) and
  birefringent or dichroic media (`materials.Uniaxial`,
  `materials.CubicCrystal`); power along every ray, transmittance, output
  fields and Jones matrices; vectorial focus of the traced rays
  (`System.vectorial_focus()`, Debye-Wolf integral with the coatings and the
  OPD)
- `delano`: Delano's y-ybar diagram, tested: the diagram of any centred
  `raytrace.System` (powers, spacings, magnification, NA, pupils and
  cardinal points agree with the paraxial trace and `abcd`), thin lens
  systems from a drawn polygon, thick singlets with the same first order
  properties (principal planes, `thicken()`), `refine()` for hand drawn
  layouts (Petzval sum, spaces, beam heights, power signs)
- `lens_optimization`: damped least squares lens optimization, tested
  (singlet bending for minimum spherical aberration, aspheres, solves):
  curvatures, thicknesses, object distance and even aspheric terms as
  variables, solves for total length and object side NA, operands for
  transverse ray errors, distortion, telecentricity, paraxial targets and
  clearances (centre and edge thicknesses at the rim of the full beam),
  stages of growing aperture and field
- `beam_quality`: beam widths, azimuth and M² after ISO 11146 (second
  moments, iterated integration window, caustic fit), tested
- `modes`: mode analysis and fibres, tested: decomposition of any field into
  Hermite-Gaussian or Laguerre-Gaussian modes (given or fitted waist,
  curvature and centre; power fractions, reconstruction error, propagation
  of the expansion), M² from the mode content (exact second moments of
  coherent superpositions); LP modes of weakly guiding step-index fibres
  (characteristic equation, cutoffs, b(V), n_eff, mode fields, Marcuse's
  Gaussian approximation); coupling efficiency of any field into any mode
  (generalizes `abcd.coupling_efficiency()`), sweeps over offset, tilt and
  defocus, power per fibre mode
- `TIE`: basic transport of intensity solver, partly tested
- `ifta`: basic Gerchberg-Saxton and variants for beam shaping, tested
  (convergence of GS, HIO-HOO and over-compensation on a reachable target,
  phase quantization)
- `phase_retrieval`: phase retrieval, tested: Fienup's error reduction and
  hybrid input-output with support, reality and positivity constraints,
  shrink-wrap support and the autocorrelation support, error histories,
  removal of the trivial ambiguities (shift, constant phase, twin image);
  pupil phase retrieval from through-focus PSFs (phase diversity) by a
  multi-plane Gerchberg-Saxton iteration (with unwrapping via `unwrap`) or a
  least squares fit of Fringe Zernikes with the analytic Jacobian, with
  paraxial (Z4) or exact (angular spectrum) defocus; near-field (Fresnel)
  coherent diffractive imaging with any pyoptics propagator in place of the
  Fourier transform
- `shack_hartmann`: Shack-Hartmann wavefront sensor, tested: spot images
  (Fraunhofer pattern of each lenslet on its camera window) from a pupil
  phase, e.g. a ray-traced `raytrace.Pupil`, thresholded centroids with
  optional reference image, intensity weighted model slopes, modal
  reconstruction (least squares fit of Fringe Zernike gradients) and zonal
  reconstruction (Southwell); sensitivity and dynamic range of the design
- `wpm`: wave propagation method (Cython extension) for inhomogeneous, also
  absorbing media, 2D and y-invariant 1D fields, multi-step driver; direct
  O(N^2) sum (OpenMP) or the fast "levels" variant (one FFT pair per distinct
  index), chosen automatically; tested against Rayleigh-Sommerfeld, plane
  waves (tilted, evanescent, absorbed), slabs and a GRIN mode
- `processing`: image processors for single images and image stacks, tested:
  `Scaler` (peak, mean or total of field strength or intensity),
  `CameraNoiser` (Gaussian, Poisson shot noise, linear camera model after
  EMVA 1288 with dark current, read noise, full well, gain, ADC and fixed
  pattern noise), `Shifter` (subpixel Fourier shift) and `Centerer`
  (centering on the intensity centroid, `utils.center_of_mass()`)
- Unfinished stubs: `interpolator.TrigInterpolator` cannot be evaluated yet,
  the functions in `polarization._polarization` are empty

See `WIP.md` for planned work.
<!-- docs:status-end -->

Installation
------------

<!-- docs:install-start -->
pyoptics is a Python package with one Cython extension (`wpm`). It is built
with setuptools via `pyproject.toml`. Cython and NumPy are downloaded
automatically while building, but you need a C compiler on your machine.

### Step-by-step installation into a virtual environment

A virtual environment ("venv") is a private folder of Python packages that
belongs to one project. Installing pyoptics into a venv keeps it apart from
your system Python and from your other projects. If something goes wrong,
you can simply delete the folder and start again.

**1. Install the prerequisites**

You need Python 3.9 or newer, git, and a C compiler:

- Debian/Ubuntu: `sudo apt install python3 python3-venv python3-dev build-essential git`
- Fedora: `sudo dnf install python3 python3-devel gcc git`
- macOS: install Python from [python.org](https://www.python.org/downloads/)
  or Homebrew, then run `xcode-select --install` to get the compiler and git.
- Windows: install Python from [python.org](https://www.python.org/downloads/)
  (tick "Add python.exe to PATH" in the installer), install
  [git](https://git-scm.com/download/win), and install the
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  with the "Desktop development with C++" workload.

Check that Python works by opening a terminal (on Windows: PowerShell) and
running `python3 --version` (on Windows: `py --version`).

**2. Download pyoptics**

    git clone https://github.com/aeberspaecher/pyoptics.git
    cd pyoptics

All following commands are run inside this `pyoptics` folder.

**3. Create the virtual environment**

This creates a folder called `.venv` in the project:

    python3 -m venv .venv        # Linux and macOS
    py -m venv .venv             # Windows

**4. Activate the virtual environment**

    source .venv/bin/activate    # Linux and macOS
    .venv\Scripts\Activate.ps1   # Windows PowerShell
    .venv\Scripts\activate.bat   # Windows cmd.exe

Your prompt now starts with `(.venv)`. While it does, `python` and `pip`
refer to the venv. You have to activate the venv again in every new terminal.
If PowerShell refuses to run the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**5. Install pyoptics**

    python -m pip install --upgrade pip
    python -m pip install -e ".[plot]"

This compiles the `wpm` extension, which can take a minute. `-e` means
"editable": changes to the Python files take effect immediately without
reinstalling. Only after editing `wpm.pyx` do you have to run the install
command again to recompile it.

The part in brackets selects optional extras, which you can combine, for
example `".[plot,test]"`:

- `plot`: matplotlib, for the plotting tools
- `nufft`: finufft, for faster exact propagation onto tilted planes
  (`tilted`, rotations about a general axis)
- `tie`: sympy and matplotlib, for the TIE module
- `notebooks`: matplotlib, JupyterLab and ipywidgets, for running the example
  notebooks (including the interactive one)
- `test`: pytest (and matplotlib, which some modules import), for running the tests

Use `python -m pip install -e .` if you want no extras at all.

**6. Check that it works**

    python -c "import pyoptics.wpm; print('pyoptics works')"

If it prints `pyoptics works`, you are done. With the `test` extra you can
also run the test suite with `python -m pytest test`; it takes about ten
seconds.

When you are finished, leave the venv with `deactivate`.

### Troubleshooting

- `error: command 'gcc' failed` or "Microsoft Visual C++ 14.0 or greater is
  required": the C compiler from step 1 is missing.
- `Python.h: No such file or directory`: install the Python development
  headers (`python3-dev` or `python3-devel`, see step 1).
- `No module named venv` / `ensurepip is not available`: install
  `python3-venv` (Debian/Ubuntu).
- `ModuleNotFoundError: No module named 'pyoptics'`: the venv is not
  activated in this terminal; repeat step 4.

### Notes on OpenMP

On Linux and Windows the `wpm` extension is compiled with OpenMP so it can
use several CPU cores. On macOS it is built without OpenMP.
<!-- docs:install-end -->

Example notebooks
-----------------

The directory `notebooks/` contains executed Jupyter notebooks that
illustrate the individual modules with visual, physically intuitive examples:
free space propagation, Gaussian beams, interfaces and thin films,
polarization, imaging and aberrations, TIE, IFTA, WPM, basis functions, an
interactive layer stack designer, image processing (scaling, camera noise,
centering), beam quality, lithography and ABCD matrices (rays, Gaussian
beams, resonators), propagation onto tilted planes, phase unwrapping, interferometry, digital
holography, ray tracing to OPD maps and phase retrieval. See
`notebooks/README.md` for an overview.

Conventions
-----------

<!-- docs:conventions-start -->
All of pyoptics uses the time dependence exp(-iωt), which is implicit and
never written out, and positive spatial phases for propagation, i.e. fields
vary as exp(i(**k**·**r** − ωt)). In particular:

- A plane wave travelling in +z is exp(+ikz). Propagating a field by a
  distance z > 0 multiplies each of its plane wave components by
  exp(+ik_z z); evanescent components decay as exp(−|k_z| z).
- `fft.FT()` has the kernel exp(−ik_x x) and returns the amplitudes of the
  plane waves exp(+ik_x x); `fft.inv_FT()` sums them up. Both work on
  centered arrays (x = 0 and k = 0 at index N//2, as `utils.freq_grid()`
  returns them), see the docstring of `pyoptics.fft`.
- A diverging spherical wave is exp(+ikr)/r. Beams (`beams.gauss_laguerre()`,
  `beams.gauss_hermite()`) carry the phase k z + k r²/(2R(z)) − ψ(z) with the
  Gouy phase ψ(z) = (N + 1) arctan(z/z_R); Laguerre-Gaussian modes have the
  helical phase exp(+ilφ).
- A converging thin lens of focal length f > 0 has the transmission
  exp(−ik r²/(2f)) (`masks.thin_lens()`).
- ABCD matrices (`abcd`) act on rays (y, nθ) with the reduced angle nθ, so
  det M = 1. Radii are positive if the centre of curvature lies in +z,
  concave mirrors have R > 0. The complex beam parameter is the textbook
  q = z + i z_R with 1/q = 1/R − iλ₀/(π n w²); the field is
  ∝ exp(+ik r²/(2R)) exp(−r²/w²) and the Gouy phase enters as exp(−iψ).
- Ray tracing (`raytrace`) uses the same radius convention; a concave mirror
  facing the incoming light has R < 0 there (real geometry, not unfolded),
  and thicknesses behind mirrors are negative. Tilts are rotations
  R_x(a_x) R_y(a_y) R_z(a_z), counterclockwise, after the decentre. The OPD is
  OPL_ray − OPL_chief on the reference sphere, so the pupil phase is
  +2π·OPD/λ: an image plane behind the focus gives OPD < 0 at the rim, as
  the propagators' exp(+ik_z z).
- Absorbing media have the complex refractive index n + iκ with κ > 0. Under
  total internal reflection, the transmitted wave decays away from the
  interface.
- Wave plates delay the slow (y) component by exp(+iδ). The Jones vector
  [1, i]/√2 rotates from x to y in time and has the Stokes parameter
  S₃ = 2 Im(E_x* E_y) = +1.
<!-- docs:conventions-end -->

Documentation
-------------

The documentation is built with Sphinx from `docs/` and published at
<https://aeberspaecher.github.io/pyoptics/>: installation, conventions, the
example notebooks with their figures and an API reference generated from the
docstrings. Docstrings follow the NumPy style. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The sections About, Status, Installation and Conventions of this README are
included in the documentation; edit them here.

License
-------

Copyright 2014-2017 Alexander Eberspächer

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
