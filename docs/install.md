# Installation

pyoptics is a Python package with one Cython extension (`wpm`). It is built
with setuptools via `pyproject.toml`. Cython and NumPy are downloaded
automatically while building, but you need a C compiler on your machine.

## Step-by-step installation into a virtual environment

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
- `fast`: numba, for a compiled ray trace kernel (`raytrace`, several times
  faster ray traces and ray aiming, most for large lenses)
- `nufft`: finufft, for faster exact propagation onto tilted planes
  (`tilted`, rotations about a general axis)
- `tie`: sympy and matplotlib, for the TIE module
- `notebooks`: matplotlib, JupyterLab and ipywidgets, for running the example
  notebooks (including the interactive one)
- `test`: pytest (and matplotlib, which some modules import), for running the tests
- `docs`: Sphinx and its extensions, for building this documentation

Use `python -m pip install -e .` if you want no extras at all.

**6. Check that it works**

    python -c "import pyoptics.wpm; print('pyoptics works')"

If it prints `pyoptics works`, you are done. With the `test` extra you can
also run the test suite with `python -m pytest test`; it takes about ten
seconds.

When you are finished, leave the venv with `deactivate`.

## Troubleshooting

- `error: command 'gcc' failed` or "Microsoft Visual C++ 14.0 or greater is
  required": the C compiler from step 1 is missing.
- `Python.h: No such file or directory`: install the Python development
  headers (`python3-dev` or `python3-devel`, see step 1).
- `No module named venv` / `ensurepip is not available`: install
  `python3-venv` (Debian/Ubuntu).
- `ModuleNotFoundError: No module named 'pyoptics'`: the venv is not
  activated in this terminal; repeat step 4.

## Notes on OpenMP

On Linux and Windows the `wpm` extension is compiled with OpenMP so it can
use several CPU cores. On macOS it is built without OpenMP.

## Building this documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The notebooks are rendered with the outputs stored in them; re-execute them
with `jupyter nbconvert` (see {doc}`examples`) to update the figures.
