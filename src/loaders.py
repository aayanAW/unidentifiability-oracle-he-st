"""Real-data swap point for GSE243280 (breast Xenium Rep1/Rep2 + post-Xenium H&E + paired Visium).

Two entry points:

  load_xenium_noise_floor(data_dir)  -- the preregistration.md sec 6 gating control. Needs ONLY the two
      Xenium replicates (no UNI, no H&E, no Visium): it returns the matched-niche replicate matrices so
      oracle.noise_floor can estimate sigma2_xen and we can ask whether ANY variance survives it (K4).
      This is the cheapest decisive real-data check and is unblocked by the UNI gating problem.

  load_breast_triad(data_dir)        -- the full TriadData for experiments/feasibility_breast.py --real:
      adds morphology embeddings (UNI/Virchow2, ungated fallback) and the Visium dropout target.

Locked noise-floor source (preregistration.md sec 11-A):
  sigma2_xen : Janesick GSE243280 GSM7780153 (Rep1) + GSM7780154 (Rep2) -- breast Sample #1, same FFPE
               block, serial 5um sections, same 313-gene panel. Serial sections => niche-level matching.

Expected layout under `data_dir` (produced by scripts/fetch_data.sh):
  data_dir/rep1/{cell_feature_matrix.h5, cells.parquet|cells.csv.gz, gene_panel.json}
  data_dir/rep2/{...same...}
  data_dir/rep1/Post-Xenium_HE_Rep1.ome.tif[.gz]   + *_homography.csv.gz   (only for the full triad)
  data_dir/visium/ (10x spatial outs: filtered_feature_bc_matrix.h5 + spatial/)  (only for the full triad)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import xenium_io as xio
from .simulator import TriadData

NICHE_UM = 100.0  # niche resolution (preregistration.md sec 11-D: never raw 55um Visium-spot resolution)
MIN_CELLS_PER_BIN = 20


def load_xenium_noise_floor(
    data_dir: str, bin_um: float = NICHE_UM, min_cells: int = MIN_CELLS_PER_BIN
):
    """Return (z_rep1, z_rep2, centers, genes) at matched niches -- the sec 6 noise-floor control input.

    z_rep1/z_rep2 are per-gene z-scored log1p-CPM pseudobulk over co-occupied niches in a shared frame.
    Plug into oracle.noise_floor(z_rep1, z_rep2). No morphology / Visium / UNI required.
    """
    root = Path(data_dir)
    sec1 = xio.read_section(_section_dir(root, "rep1"))
    sec2 = xio.read_section(_section_dir(root, "rep2"))
    z1, z2, centers, genes = xio.matched_bins(
        sec1, sec2, bin_um=bin_um, min_cells=min_cells
    )
    return z1, z2, centers, genes


def load_breast_triad(data_dir: str) -> TriadData:
    """Full real triad. Requires both Xenium replicates + post-Xenium H&E (+ homography) + paired Visium.

    morph = H&E embeddings (UNI if access granted, else ungated fallback -> the run is EXPLORATORY).
    visium = dropout-corrupted target binned to the SAME niches. gene_class = None (no planted labels).
    """
    root = Path(data_dir)
    z1, z2, centers, genes = load_xenium_noise_floor(data_dir)

    morph = _load_morph(root, centers)
    visium = _load_visium_binned(root, centers, genes)

    return TriadData(
        coords=centers,
        morph=morph,
        xen_rep1=z1,
        xen_rep2=z2,
        visium=visium,
        gene_class=None,
    )


def _load_morph(root: Path, centers: np.ndarray) -> np.ndarray:
    from .embeddings import embed_bins

    he = xio._first_existing(  # noqa: SLF001 -- shared file-glob helper
        root / "rep1",
        (
            "Post-Xenium_HE_Rep1.ome.tif",
            "Post-Xenium_HE_Rep1.ome.tif.gz",
            "*HE*.ome.tif*",
        ),
    )
    if he is None:
        raise FileNotFoundError(
            f"no post-Xenium H&E under {root / 'rep1'} -- needed for morphology embeddings.\n"
            "Fetch GSM7780153_Post-Xenium_HE_Rep1.ome.tif.gz (1.3 GB) and its *_homography.csv.gz."
        )
    homog = xio._first_existing(
        root / "rep1", ("*HE*homography.csv.gz", "*homography.csv*")
    )  # noqa: SLF001
    mf = embed_bins(he, centers, homog, encoder="uni")
    if not mf.is_confirmatory:
        print(
            f"[loaders] morphology encoder = '{mf.encoder_name}' (FALLBACK). This triad run is "
            "EXPLORATORY; a confirmatory U claim needs UNI/Virchow2 access."
        )
    return mf.features


def _load_visium_binned(
    root: Path, centers: np.ndarray, genes: list[str]
) -> np.ndarray:
    """Bin Visium counts onto the Xenium niches; return (n_bins, n_genes) on the shared gene set.

    Visium spots are registered into the Rep1 niche frame by rigid tissue-mask alignment (same method as
    the replicate registration). If no Visium dir is present, raises with the exact file to fetch.
    """
    vdir = root / "visium"
    if not vdir.exists():
        raise FileNotFoundError(
            f"no {vdir} -- the full triad needs the paired breast Visium (CytAssist FFPE) outs.\n"
            "The noise-floor control (load_xenium_noise_floor) does NOT need this and can run now."
        )
    import scanpy as sc

    h5 = xio._first_existing(vdir, ("*filtered_feature_bc_matrix.h5", "*.h5"))  # noqa: SLF001
    ad = sc.read_10x_h5(str(h5))
    ad.var_names_make_unique()
    pos = _read_visium_positions(vdir)
    spot_xy = pos.loc[ad.obs_names, ["x", "y"]].to_numpy(float)

    aff = xio.rigid_register(spot_xy, centers, NICHE_UM)
    spot_reg = xio.apply_affine(spot_xy, aff)

    counts = (
        np.asarray(ad.X.todense()) if hasattr(ad.X, "todense") else np.asarray(ad.X)
    )
    gidx = {g: i for i, g in enumerate(ad.var_names)}
    cols = [gidx.get(g) for g in genes]
    out = np.zeros((len(centers), len(genes)), np.float64)
    bin_of = _nearest_bin(spot_reg, centers, NICHE_UM)
    for s, b in enumerate(bin_of):
        if b < 0:
            continue
        for j, c in enumerate(cols):
            if c is not None:
                out[b, j] += counts[s, c]
    return out


def _read_visium_positions(vdir: Path):
    import pandas as pd

    spatial = vdir / "spatial"
    f = xio._first_existing(  # noqa: SLF001
        spatial if spatial.exists() else vdir,
        ("tissue_positions.csv", "tissue_positions_list.csv", "*tissue_positions*.csv"),
    )
    df = pd.read_csv(f, header=0 if _has_header(f) else None)
    df.columns = [str(c).lower() for c in df.columns]
    bc = xio._pick(df.columns, ("barcode", "0"))  # noqa: SLF001
    xc = xio._pick(df.columns, ("pxl_col_in_fullres", "x", "4"))  # noqa: SLF001
    yc = xio._pick(df.columns, ("pxl_row_in_fullres", "y", "5"))  # noqa: SLF001
    return df.set_index(df[bc].astype(str)).rename(columns={xc: "x", yc: "y"})[
        ["x", "y"]
    ]


def _nearest_bin(pts: np.ndarray, centers: np.ndarray, bin_um: float) -> np.ndarray:
    """Assign each point to the niche whose center it falls in; -1 if outside any occupied niche."""
    key_to_bin = {
        tuple(np.round(c / bin_um).astype(int)): i for i, c in enumerate(centers)
    }
    keys = np.round(pts / bin_um).astype(int)
    return np.array([key_to_bin.get(tuple(k), -1) for k in keys])


def _has_header(path: Path) -> bool:
    with open(path) as fh:
        first = fh.readline()
    return "barcode" in first.lower() or "pxl" in first.lower()


def _section_dir(root: Path, name: str) -> Path:
    d = root / name
    if not d.exists():
        raise FileNotFoundError(
            f"missing {d}. Run scripts/fetch_data.sh to populate {root}/rep1 and {root}/rep2 "
            "(cell_feature_matrix.h5 + cells.parquet + gene_panel.json per replicate)."
        )
    return d
