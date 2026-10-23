pyoptics
========

About
-----

pyoptics is a collection of tools for wave optical simulation. It contains
classes for imaging simulations and wave propagation. pyoptics has a modular
structure that easily allows reuse of individual blocks in new contexts.

Status
------

pyoptics is currently in an early pre-beta stadium.

Vague status of some of the modules:

- bases: mature
- thin_films: rather mature
- propagators: basic, missing documentation
- utils: mature
- beams: basic
- IFTA: basic
- TIE: basic
- imaging: missing completely
- processing: missing
- polarization: basic Müller calculus support, Jones calculus missing
- wpm: basic, untested

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

You need Python 3.8 or newer, git, and a C compiler:

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
- `test`: pytest (and matplotlib, which some modules import), for running the tests

Use `python -m pip install -e .` if you want no extras at all.

**6. Check that it works**

    python -c "import pyoptics.wpm; print('pyoptics works')"

If the last line printed is `pyoptics works`, you are done. A message like
"Loading NumPy's fft routines, tfftw probably not present" is normal and only
means the optional FFTW backend is not installed. With the `test` extra you can also run the test suite with
`python -m pytest test`.

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

Conventions
-----------

All of pyoptics uses the time dependence exp(-iωt), which is implicit and
never written out, and positive spatial phases for propagation, i.e. fields
vary as exp(i(**k**·**r** − ωt)). In particular:

- A plane wave travelling in +z is exp(+ikz). Propagating a field by a
  distance z > 0 multiplies each of its plane wave components by
  exp(+ik_z z); evanescent components decay as exp(−|k_z| z).
- `fft.FT()` has the kernel exp(−ik_x x) and returns the amplitudes of the
  plane waves exp(+ik_x x); `fft.inv_FT()` sums them up.
- A diverging spherical wave is exp(+ikr)/r. Beams (`beams.gauss_laguerre()`,
  `beams.gauss_hermite()`) carry the phase k z + k r²/(2R(z)) − ψ(z) with the
  Gouy phase ψ(z) = (N + 1) arctan(z/z_R); Laguerre-Gaussian modes have the
  helical phase exp(+ilφ).
- A converging thin lens of focal length f > 0 has the transmission
  exp(−ik r²/(2f)) (`masks.thin_lens()`).
- Absorbing media have the complex refractive index n + iκ with κ > 0. Under
  total internal reflection, the transmitted wave decays away from the
  interface.
- Wave plates delay the slow (y) component by exp(+iδ). The Jones vector
  [1, i]/√2 rotates from x to y in time and has the Stokes parameter
  S₃ = 2 Im(E_x* E_y) = +1.

Documentation
-------------

Documentation is currently mostly absent. Please refer to individual functions'
docstrings.

License
-------

Copyright 2014-2017 Alexander Eberspächer

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
