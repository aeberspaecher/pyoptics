#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check that notebooks are referred to by number and name, never by number alone.

A bare "see notebook NN" says nothing to a reader who does not know the
numbers by heart. Every reference must carry the notebook's short name from
``notebooks/names.json``, in parentheses, after a comma or simply after the
number::

    notebook 47 (rays and waves) showed ...
    ... as before (notebook 47, rays and waves).
    notebooks 117 (dispersion in fibres) and 118 (pulse broadening)
    notebook [52 (Abbe image formation)](52_abbe_image_formation.ipynb)
    [47 rays and waves](47_rays_and_waves.ipynb)

The name may also stand just before the reference, as in
"the Zernike aberration gallery (notebook 45)". The check covers the
markdown cells and code comments of all notebooks, the Markdown files of
the repository and the docs, and the comments and docstrings of the
package and the tests.
"""

import io
import json
import re
import tokenize
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT/"notebooks").glob("*.ipynb"))
with open(ROOT/"notebooks"/"names.json", encoding="utf-8") as _f:
    NAMES = json.load(_f)

NUM = r"\d{2,3}"
# a number, optionally followed by a name in parentheses
ITEM = NUM + r"(?![\d.,]\d)(?:\s*\([^()]*\))?"
SEP = r"(?:\s*,\s*|\s+(?:and|or|to)\s+|\s*[–-]\s*|\s*,\s*(?:and|or)\s+)"
# "notebook NN", "notebooks NN and MM", "nbNN", "notebooks NN to MM"
REF = re.compile(r"(?<![\w\[/])(?:[Nn]otebooks?|[Nn][Bb]s?)\s*(?P<list>" + ITEM
                 + r"(?:" + SEP + ITEM + r")*)(?!\w)")
ITEM_NUM = re.compile(r"(?<![\w(])" + NUM + r"(?![\d.,]\d)")
# a link or Sphinx role whose text is a bare number:
# "[NN](NN_name.ipynb)", {doc}`NN <notebooks/NN_name>`
BARE_LINK = re.compile(r"\[(?P<n>" + NUM + r")\]\([^)\s]*\.ipynb[^)\s]*\)")
BARE_DOC = re.compile(r"\{doc\}`(?P<n>" + NUM + r")\s*<")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _strip_links(text):
    return LINK.sub(r"\1", text)


def _named(num, before, after):
    """Whether the name of notebook `num` stands next to the reference."""
    name = NAMES[num].lower()
    if name in before[-(len(name) + 12):].lower():
        return True
    return re.match(r"\s*(?:,\s*|\(\s*|,?\s*\*)?" + re.escape(name), after.lower()) is not None


def bare_references(text):
    """The references in `text` that give a notebook's number without its name."""
    found = [m.group(0) for m in BARE_LINK.finditer(text) if m.group("n") in NAMES]
    found += [m.group(0) for m in BARE_DOC.finditer(text)]
    text = _strip_links(text)
    for m in REF.finditer(text):
        refs = m.group("list")
        for item in ITEM_NUM.finditer(refs):
            num = item.group(0)
            if num in NAMES and not _named(num, text[:m.start()], refs[item.end():] + text[m.end():]):
                found.append(f"{m.group(0)!r} (no name for {num})")
    return found


def _python_text(source, strings=True):
    """Comments (and string literals) of Python source."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [line.split("#", 1)[1] for line in source.splitlines() if "#" in line]
    kinds = (tokenize.COMMENT, tokenize.STRING) if strings else (tokenize.COMMENT,)
    return [t.string for t in tokens if t.type in kinds]


def _notebook_text(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    texts = []
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            texts.append(source)
        elif cell["cell_type"] == "code":
            # magics are not Python, so take comments line by line there
            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                texts += [line.split("#", 1)[1] for line in source.splitlines() if "#" in line]
            else:
                texts += _python_text(source, strings=False)
    return texts


def _sources():
    md = [ROOT/"README.md", ROOT/"WIP.md", ROOT/"CLAUDE.md", ROOT/"notebooks"/"README.md"]
    md += sorted((ROOT/"docs").glob("*.md"))
    py = sorted((ROOT/"pyoptics").rglob("*.py")) + sorted((ROOT/"pyoptics").glob("*.pyx"))
    py += [p for p in sorted((ROOT/"test").glob("*.py")) if p.name != Path(__file__).name]
    return [p for p in md if p.exists()] + py + NOTEBOOKS


def _texts(path):
    if path.suffix == ".ipynb":
        return _notebook_text(path)
    source = path.read_text(encoding="utf-8")
    if path.suffix in (".py", ".pyx"):
        return _python_text(source)
    return [source]


def test_every_notebook_has_a_name():
    numbers = {p.name.split("_", 1)[0] for p in NOTEBOOKS}
    assert numbers <= set(NAMES), f"add to notebooks/names.json: {sorted(numbers - set(NAMES))}"
    assert set(NAMES) <= numbers, f"no such notebook: {sorted(set(NAMES) - numbers)}"


# the cases are assembled so that the check does not trip over this file
_N = "note" + "book"


@pytest.mark.parametrize("text, n_bare", [
    (f"as in {_N} 47.", 1),
    (f"as in {_N} 47 (rays and waves).", 0),
    (f"({_N} 47, rays and waves)", 0),
    (f"{_N} 47 rays and waves", 0),
    ("(nb47)", 1),
    (f"{_N}s 117 (dispersion in fibres) and 118.", 1),
    (f"{_N}s 117 (dispersion in fibres) and 118 (pulse broadening)", 0),
    (f"{_N}s 52 (Abbe image formation) to 55.", 1),
    (f"{_N} [52](52_abbe_image_formation.ipynb)", 2),
    (f"{_N} [52 (Abbe image formation)](52_abbe_image_formation.ipynb)", 0),
    (f"the Zernike aberration gallery ({_N} 45)", 0),
    ("see [47](47_rays_and_waves.ipynb)", 1),
    ("{doc}`47 <notebooks/47_rays_and_waves>`", 1),
    (f"({_N} 05, section 12)", 1),
    (f"this {_N} uses 64 rays", 0),
])
def test_detector(text, n_bare):
    assert len(bare_references(text)) == n_bare


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_references_name_the_notebook(path):
    bare = [ref for text in _texts(path) for ref in bare_references(text)]
    assert not bare, ("refer to notebooks by number and name (see notebooks/names.json): "
                      + "; ".join(bare))
