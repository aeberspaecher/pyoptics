pyoptics
========

About
-----

pyoptics is a collection of tools for wave optical simulation. It contains
classes for imaging simulations and wave propagation. pyoptics has a modular
structure that easily allows reuse of individual blocks in new contexts.

Status
------

pyoptics is version 0.1, a research and teaching toolbox rather than a
polished library. It runs on Python 3.9 to 3.13. The API may still change
between versions, and there is no release on PyPI; install it from source as
described below.

The core is tested: a suite of about 450 pytest tests checks the numerics
against analytical results (Fresnel and Fraunhofer diffraction, Gaussian beam
propagation, Mahajan's on-axis intensity, Airy pattern and OTF, Fresnel
equations, thin film stacks, Goos-Hänchen shift, beam widths and M² of
Hermite-Gaussian beams, Mueller and Jones calculus, FFT conventions,
Richards-Wolf integrals for high NA foci, two-beam interference of TE and
TM waves, in air and in a resist, standing waves in a resist on silicon,
threshold CDs and process windows of analytic images, apodized two-beam
images, ABCD matrices against lens and resonator formulas and the Collins
integral against Gaussian beams, phase unwrapping of wrapped Zernike
wavefronts, vortices and noisy wavefronts, phase-shifting and Fourier
fringe analysis of known Zernike phases) and runs in GitHub
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
  Bossung curves, process windows)
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
  reliability sorting (Herráez), residue maps, `wrap()` and a 1D helper
- `interferometry`: fringe analysis, tested: simulated interferograms
  (Zernike test wavefronts, tilt carrier, visibility, camera noise via
  `processing.CameraNoiser`, miscalibrated phase shifters); phase-shifting
  algorithms (N-step least squares for arbitrary known shifts, 4-step,
  Carré with unknown step, Hariharan's 5-step), each returning the wrapped
  phase and the modulation; Takeda's Fourier-transform method for single
  interferograms with a carrier (sideband window, automatic carrier search,
  masks, apodization, overlap warning); phase error vs. shifter
  miscalibration
- `beam_quality`: beam widths, azimuth and M² after ISO 11146 (second
  moments, iterated integration window, caustic fit), tested
- `TIE`: basic transport of intensity solver, partly tested
- `ifta`: basic Gerchberg-Saxton and variants, only shown in a notebook, no
  tests
- `wpm`: wave propagation method (Cython extension), basic; one test, more
  examples in the notebook
- `processing`: image processors for single images and image stacks, tested:
  `Scaler` (peak, mean or total of field strength or intensity),
  `CameraNoiser` (Gaussian, Poisson shot noise, linear camera model after
  EMVA 1288 with dark current, read noise, full well, gain, ADC and fixed
  pattern noise), `Shifter` (subpixel Fourier shift) and `Centerer`
  (centering on the intensity centroid, `utils.center_of_mass()`)
- Unfinished stubs: `interpolator.TrigInterpolator` cannot be evaluated yet,
  the functions in `polarization._polarization` are empty

See `WIP.md` for planned work.

Installation
------------

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
example `".[plot,fftw,test]"`:

- `plot`: matplotlib, for the plotting tools
- `fftw`: pyFFTW, for faster FFTs
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

Example notebooks
-----------------

The directory `notebooks/` contains executed Jupyter notebooks that
illustrate the individual modules with visual, physically intuitive examples:
free space propagation, Gaussian beams, interfaces and thin films,
polarization, imaging and aberrations, TIE, IFTA, WPM, basis functions, an
interactive layer stack designer, image processing (scaling, camera noise,
centering), beam quality, lithography and ABCD matrices (rays, Gaussian
beams, resonators), phase unwrapping and interferometry. See
`notebooks/README.md` for an overview.

Conventions
-----------

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
- Absorbing media have the complex refractive index n + iκ with κ > 0. Under
  total internal reflection, the transmitted wave decays away from the
  interface.
- Wave plates delay the slow (y) component by exp(+iδ). The Jones vector
  [1, i]/√2 rotates from x to y in time and has the Stokes parameter
  S₃ = 2 Im(E_x* E_y) = +1.

Documentation
-------------

There is no separate manual yet. The best starting points are the example
notebooks in `notebooks/`, the conventions above and the docstrings of the
individual functions and modules.

License
-------

Copyright 2014-2017 Alexander Eberspächer

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
