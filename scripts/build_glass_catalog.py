#!/usr/bin/env python
"""Build the glass catalogue data of `pyoptics.glasses` from the
refractiveindex.info database (CC0, https://github.com/polyanskiy/refractiveindex.info-database).

Usage::

    python scripts/build_glass_catalog.py PATH/TO/refractiveindex.info-database

writes pyoptics/glasses/data/<vendor>.json with the dispersion formula of
every optical glass of the vendors below (needs PyYAML).
"""

import glob
import json
import os
import sys

import yaml

VENDORS = {"schott": "SCHOTT", "ohara": "OHARA", "hoya": "HOYA", "cdgm": "CDGM",
           "sumita": "SUMITA", "hikari": "HIKARI"}
FORMULAS = {"formula 1": "sellmeier", "formula 2": "sellmeier2", "formula 3": "polynomial"}


def convert(root, vendor):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "database", "data", "specs", vendor, "optical",
                                           "*.yml"))):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        data = [x for x in d.get("DATA", []) if x["type"] in FORMULAS]
        if not data:
            continue
        x = data[0]
        p = d.get("PROPERTIES", {}) or {}
        coeffs = [float(c) for c in str(x["coefficients"]).split()]
        lo, hi = (float(w) for w in str(x["wavelength_range"]).split())
        name = os.path.splitext(os.path.basename(f))[0]
        out[name] = {"formula": FORMULAS[x["type"]], "coefficients": coeffs,
                     "range_um": [lo, hi], "nd": p.get("nd"), "Vd": p.get("Vd"),
                     "dPgF": p.get("dPgF"), "status": p.get("glass_status"),
                     "code": str(p["glass_code"]) if p.get("glass_code") is not None else None}

    return out


def main(root):
    here = os.path.join(os.path.dirname(__file__), "..", "pyoptics", "glasses", "data")
    for vendor, label in VENDORS.items():
        glasses = convert(root, vendor)
        doc = {"vendor": label,
               "source": "refractiveindex.info database (CC0 1.0), database/data/specs/"
                         + vendor + "/optical, from the vendor's Zemax catalogue",
               "glasses": glasses}
        with open(os.path.join(here, vendor + ".json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=0, sort_keys=True)
            fh.write("\n")
        print(vendor, len(glasses))


if __name__ == "__main__":
    main(sys.argv[1])
