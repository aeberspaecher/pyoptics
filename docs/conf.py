"""Sphinx configuration for the pyoptics documentation.

Build with

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

The notebooks are rendered with their stored outputs (they are not executed
during the build). They live in notebooks/ at the top of the repository and
are copied into docs/notebooks/ on every build.
"""

import shutil
from importlib.metadata import version as _version
from pathlib import Path

DOCS = Path(__file__).parent
ROOT = DOCS.parent

# Sphinx only reads sources below docs/, so copy the notebooks in.
_nb_target = DOCS/"notebooks"
shutil.rmtree(_nb_target, ignore_errors=True)
shutil.copytree(ROOT/"notebooks", _nb_target, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "README.md"))


project = "pyoptics"
author = "Alexander Eberspächer"
copyright = "2014-2026, Alexander Eberspächer"
release = _version("pyoptics")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# `name` in docstrings links to the Python object of that name.
default_role = "py:obj"

# API reference: one page per module, generated from api/*.rst.
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_xref_param_type = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Notebooks and Markdown pages.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3
# Sections included from README.md start at level 3 below the page title.
suppress_warnings = ["myst.header"]

html_theme = "furo"
html_title = f"pyoptics {release}"
html_theme_options = {
    "source_repository": "https://github.com/aeberspaecher/pyoptics",
    "source_branch": "master",
    "source_directory": "docs/",
}
