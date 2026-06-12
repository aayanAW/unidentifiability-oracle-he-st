"""Phase-1 gating control (preregistration.md sec 6): the Xenium-replicate noise floor, run FIRST.

This is the cheapest decisive REAL-data check and needs ONLY the two Xenium replicates -- no UNI, no
H&E, no Visium -- so it is unblocked by the UNI/Virchow2 gating problem. It answers the load-bearing
feasibility question behind kill-criterion K4 ("noise floor dominates"):

  After matching the serial replicates at niche resolution and z-scoring each gene, sigma2_xen(g) =
  0.5*mean((z1-z2)^2) is the technical-noise fraction of that gene's variance (= 1 - cross-replicate
  concordance). A gene whose sigma2_xen ~ 1 is pure measurement noise on this panel+binning and is NOT
  auditable for unidentifiability; a gene with low sigma2_xen has reproducible spatial signal that the
  oracle can then test against morphology. If essentially NO gene clears the floor, K4 fires and the
  project stops (reported as a negative result), independent of any morphology model.

This is a PRECONDITION readout, not the H2 CONFIRM/KILL gate (that needs morphology -> feasibility_breast).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loaders import load_xenium_noise_floor  # noqa: E402
from src.oracle import noise_floor  # noqa: E402

# Pre-registered readout thresholds (a gene "clears the floor" if its technical-noise fraction is below
# CLEAR_TAU, i.e. cross-replicate concordance > 1-CLEAR_TAU). FEASIBLE if >= MIN_CLEARED of panel clears.
CLEAR_TAU = 0.70
MIN_CLEARED_FRACTION = (
    0.15  # ties to the H3 utility floor (>=15% of panel genes auditable)
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "data_dir", help="dir with rep1/ and rep2/ Xenium sections (from fetch_data.sh)"
    )
    ap.add_argument("--bin-um", type=float, default=100.0)
    ap.add_argument("--min-cells", type=int, default=20)
    args = ap.parse_args()

    try:
        z1, z2, centers, genes = load_xenium_noise_floor(
            args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
        )
    except FileNotFoundError as e:
        print(f"[noise_floor] data not ready:\n  {e}")
        return 3
    s2 = noise_floor(z1, z2)  # per-gene technical-noise fraction in z-units
    cleared = s2 < CLEAR_TAU
    frac_cleared = float(cleared.mean())

    print("=" * 66)
    print("Phase-1 noise-floor control -- Xenium technical replicate (breast)")
    print("=" * 66)
    print(f"niches matched (both reps, >= {args.min_cells} cells): {len(centers)}")
    print(f"panel-intersection genes: {len(genes)}   bin: {args.bin_um:.0f} um")
    print(
        f"sigma2_xen (z-units)  median={np.median(s2):.3f}  "
        f"p25={np.percentile(s2, 25):.3f}  p75={np.percentile(s2, 75):.3f}"
    )
    print(
        f"genes clearing the floor (sigma2_xen < {CLEAR_TAU}): "
        f"{int(cleared.sum())}/{len(genes)}  ({frac_cleared:.3f})"
    )
    _top_table(genes, s2)

    feasible = frac_cleared >= MIN_CLEARED_FRACTION
    print("-" * 66)
    if feasible:
        print(
            f"READOUT: FLOOR-CLEARED -- {frac_cleared:.1%} of panel genes retain reproducible signal "
            f"above the technical noise floor (>= {MIN_CLEARED_FRACTION:.0%}). K4 does NOT fire on this "
            "control; morphology-conditioned U is worth computing (run feasibility_breast.py --real)."
        )
        return 0
    print(
        f"READOUT: FLOOR-DOMINATES -- only {frac_cleared:.1%} of panel genes clear the floor "
        f"(< {MIN_CLEARED_FRACTION:.0%}). K4 risk HIGH: the Xenium technical noise consumes most niche-level "
        "variance at this binning. Try a coarser bin or report as a negative result (preregistration.md sec 8)."
    )
    return 2


def _top_table(genes, s2, k: int = 12) -> None:
    order = np.argsort(s2)
    print("  most reproducible genes (lowest sigma2_xen, best oracle candidates):")
    for i in order[:k]:
        print(f"    {genes[i]:<16} sigma2_xen={s2[i]:.3f}  concordance~{1 - s2[i]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
