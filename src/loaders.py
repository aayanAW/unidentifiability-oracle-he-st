"""Real-data swap point. Wire GSE243280 (breast Xenium Rep1/Rep2) + paired Visium + H&E here.

Locked noise-floor source (preregistration.md sec 11-A):
  - sigma2_xen : Janesick GSE243280 GSM7780153 (Rep1) + GSM7780154 (Rep2)
                 breast Sample #1, same FFPE block, serial 5um sections, same 313-gene panel.
  - morphology : UNI (MahmoodLab/UNI) or Virchow2 (paige-ai/Virchow2) tile embeddings (pre-cached).
  - target     : Xenium-denoised expression on the panel-intersection genes.

Returns the SAME `TriadData` interface as the simulator, so experiments/feasibility_breast.py is
unchanged between synthetic and real. Until wired, this raises with the exact next step.
"""

from __future__ import annotations

from .simulator import TriadData


def load_breast_triad(data_dir: str) -> TriadData:
    raise NotImplementedError(
        "Real loader not wired yet. Next steps (Phase 0/1):\n"
        "  1. bash scripts/fetch_data.sh          # GSE243280 Rep1/Rep2 + paired Visium + H&E\n"
        "  2. python scripts/check_panel.py data/rep1 data/rep2   # confirm identical 313-gene panel\n"
        "  3. cache UNI/Virchow2 tile embeddings -> morph; register Visium<->Xenium (cell/niche resolution);\n"
        "     intersect panel genes; populate TriadData(coords, morph, xen_rep1, xen_rep2, visium, gene_class=None).\n"
        "  Estimate sigma2_reg from the +/-1-spot registration-perturbation sensitivity (preregistration.md sec 11-D)."
    )
