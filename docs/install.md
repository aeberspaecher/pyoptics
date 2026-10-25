# Installation

```{include} ../README.md
:start-after: <!-- docs:install-start -->
:end-before: <!-- docs:install-end -->
```

## Building this documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The notebooks are rendered with the outputs stored in them; re-execute them
with `jupyter nbconvert` (see {doc}`examples`) to update the figures.
