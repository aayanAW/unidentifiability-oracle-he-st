"""Confirm two Xenium replicates use the IDENTICAL gene panel (preregistration.md sec 11-A caveat ii).

Usage:  python scripts/check_panel.py data/rep1 data/rep2
Looks for a Xenium panel file (gene_panel.json / *panel*.json / features.tsv[.gz]) in each dir and diffs
the gene sets. Exits non-zero if the panels differ -- the noise floor is only valid on the intersection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_genes(d: Path) -> set[str]:
    for pat in ("gene_panel.json", "*panel*.json", "*.json"):
        for f in sorted(d.glob(pat)):
            try:
                obj = json.loads(f.read_text())
            except Exception:
                continue
            genes = _genes_from_json(obj)
            if genes:
                print(f"  {d.name}: read {len(genes)} genes from {f.name}")
                return genes
    for pat in ("features.tsv", "features.tsv.gz", "*features*.tsv*"):
        for f in sorted(d.glob(pat)):
            import gzip

            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fh:
                genes = {line.split("\t")[0].strip() for line in fh if line.strip()}
            if genes:
                print(f"  {d.name}: read {len(genes)} genes from {f.name}")
                return genes
    raise FileNotFoundError(f"no panel/features file found in {d}")


def _genes_from_json(obj) -> set[str]:
    # Xenium gene_panel.json nests target names under payload/targets/.../type/data/name
    found = set()

    def walk(o):
        if isinstance(o, dict):
            if "name" in o and isinstance(o["name"], str):
                found.add(o["name"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


def main(a: str, b: str) -> int:
    ga, gb = _load_genes(Path(a)), _load_genes(Path(b))
    inter = ga & gb
    only_a, only_b = ga - gb, gb - ga
    print(f"intersection: {len(inter)} genes")
    if only_a or only_b:
        print(
            f"  WARNING: panels differ -- only in {a}: {len(only_a)}, only in {b}: {len(only_b)}"
        )
        print(
            "  Build the noise floor on the intersection only; report the restriction."
        )
        return 1
    print("  OK: identical panels. Noise floor valid on the full panel.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
