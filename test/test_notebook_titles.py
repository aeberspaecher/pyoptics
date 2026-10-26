#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check that every example notebook starts with a single, unique H1 title.

The docs take a notebook's page title from its first heading. A notebook
that starts with code, or with a numbered section, renders without a proper
title and garbles the table of contents.
"""

import json
import re
from pathlib import Path

import pytest


NOTEBOOKS = sorted((Path(__file__).parents[1]/"notebooks").glob("*.ipynb"))
H1 = re.compile(r"^#\s+\S")


def _h1s(nb):
    """All level 1 headings in the markdown cells, in order."""
    headings, fence = [], False
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
            elif not fence and H1.match(line):
                headings.append(line[1:].strip())
    return headings


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_starts_with_single_title(path):
    nb = _load(path)
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown", "first cell must be the markdown title"
    assert H1.match("".join(first["source"])), "first cell must start with '# Title'"
    assert len(_h1s(nb)) == 1, "use '##' and below for sections"


def test_titles_unique():
    titles = [_h1s(_load(p))[0] for p in NOTEBOOKS]
    assert len(titles) == len(set(titles))
