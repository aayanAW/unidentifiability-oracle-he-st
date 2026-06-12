"""Low-level Xenium output readers + niche binning + serial-section registration.

Flexible across 10x Xenium Onboard Analysis output vintages (the Janesick 2023 breast bundle and
later releases): the cell-by-gene matrix may be `cell_feature_matrix.h5`, the MEX directory
`cell_feature_matrix/`, or a tarball; cell centroids may be `cells.parquet` or `cells.csv.gz`. We try
each in turn (preregistration.md: "no hard-coded data formats -- flexible loaders").

The two breast replicates (GSM7780153 Rep1 / GSM7780154 Rep2) are SERIAL sections, not the same cells,
so the technical noise floor sigma2_xen is estimated at NICHE (spatial-bin) resolution after a rigid
registration of the two cell point-clouds into a shared frame -- never cell-to-cell. Niche binning is
the resolution mandated by preregistration.md sec 11-D (compute U at cell/niche resolution, never at raw
55um Visium-spot resolution) and is coarse enough that serial-section + registration error inflate
sigma2_xen (the safe direction: it shrinks U).
"""

from __future__ import annotations

import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Xenium non-gene features to drop before any expression analysis (control/QC probes).
_NONGENE_PREFIXES = (
    "NegControlProbe",
    "NegControlCodeword",
    "antisense",
    "BLANK",
    "Blank",
    "UnassignedCodeword",
    "DeprecatedCodeword",
    "Intergenic",
)


@dataclass(frozen=True)
class XeniumSection:
    """One Xenium section: per-cell centroids + cell-by-gene counts on real (non-control) genes."""

    coords: np.ndarray  # (n_cells, 2) micron centroids
    counts: np.ndarray  # (n_cells, n_genes) raw integer counts
    genes: list[str]  # (n_genes,) gene names, controls removed


# --------------------------------------------------------------------------------------------------
# Selective zip extraction (disk-aware: the breast outs.zip is ~9.2 GB; we pull only what we need)
# --------------------------------------------------------------------------------------------------


def list_zip_members(zip_path: str | Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def extract_members(
    zip_path: str | Path, dest: str | Path, wanted_suffixes: tuple[str, ...]
) -> list[Path]:
    """Extract only members whose name ends with one of `wanted_suffixes`. Returns extracted paths.

    Lets a caller pull `cell_feature_matrix.h5` + `cells.parquet` + `gene_panel.json` out of the 9.2 GB
    bundle without unpacking transcripts/morphology, then delete the zip (see scripts/fetch_data.sh).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if any(name.endswith(suf) for suf in wanted_suffixes):
                flat = dest / Path(name).name
                with zf.open(name) as src, open(flat, "wb") as dst:
                    dst.write(src.read())
                out.append(flat)
    return out


# --------------------------------------------------------------------------------------------------
# Cell-by-gene matrix readers (h5 -> MEX -> fail), flexible across vintages
# --------------------------------------------------------------------------------------------------


def _read_matrix(section_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (counts (n_cells,n_genes), gene_names, cell_ids). Tries .h5 then MEX directory."""
    h5 = _first_existing(
        section_dir, ("cell_feature_matrix.h5", "*cell_feature_matrix.h5")
    )
    if h5 is not None:
        return _read_h5(h5)
    mex = section_dir / "cell_feature_matrix"
    if mex.is_dir():
        return _read_mex(mex)
    for mex_name in ("cell_feature_matrix.tar.gz",):
        if (section_dir / mex_name).exists():
            raise NotImplementedError(
                f"{mex_name} found -- untar it to a cell_feature_matrix/ dir first (or extract the .h5)."
            )
    raise FileNotFoundError(
        f"no cell_feature_matrix.h5 or cell_feature_matrix/ MEX dir under {section_dir}"
    )


def _read_h5(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    import scanpy as sc

    ad = sc.read_10x_h5(str(path), gex_only=False)
    counts = (
        np.asarray(ad.X.todense()) if hasattr(ad.X, "todense") else np.asarray(ad.X)
    )
    genes = [str(g) for g in ad.var_names]
    cell_ids = np.asarray(ad.obs_names)
    return counts.astype(np.float64), genes, cell_ids


def _read_mex(mex_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    import scanpy as sc

    ad = sc.read_mtx(str(_first_existing(mex_dir, ("matrix.mtx.gz", "matrix.mtx")))).T
    feats = _first_existing(mex_dir, ("features.tsv.gz", "features.tsv"))
    barcodes = _first_existing(mex_dir, ("barcodes.tsv.gz", "barcodes.tsv"))
    genes = [
        line.split("\t")[1] if "\t" in line else line for line in _read_lines(feats)
    ]
    cell_ids = np.asarray(_read_lines(barcodes))
    counts = (
        np.asarray(ad.X.todense()) if hasattr(ad.X, "todense") else np.asarray(ad.X)
    )
    return counts.astype(np.float64), genes, cell_ids


# --------------------------------------------------------------------------------------------------
# Cell centroid readers (parquet -> csv.gz)
# --------------------------------------------------------------------------------------------------


def _read_centroids(section_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (n_cells,2) microns, cell_ids) from cells.parquet or cells.csv.gz."""
    pq = _first_existing(section_dir, ("cells.parquet", "*cells.parquet"))
    if pq is not None:
        import pandas as pd

        df = pd.read_parquet(pq)
    else:
        csv = _first_existing(section_dir, ("cells.csv.gz", "cells.csv", "*cells.csv*"))
        if csv is None:
            raise FileNotFoundError(
                f"no cells.parquet or cells.csv.gz under {section_dir}"
            )
        import pandas as pd

        df = pd.read_csv(csv)
    xcol = _pick(df.columns, ("x_centroid", "x", "cell_centroid_x"))
    ycol = _pick(df.columns, ("y_centroid", "y", "cell_centroid_y"))
    idcol = _pick(df.columns, ("cell_id", "cell", "barcode"))
    coords = df[[xcol, ycol]].to_numpy(dtype=np.float64)
    cell_ids = df[idcol].to_numpy().astype(str)
    return coords, cell_ids


def read_section(section_dir: str | Path) -> XeniumSection:
    """Read one Xenium section dir into aligned centroids + counts on real genes only."""
    section_dir = Path(section_dir)
    counts, genes, mat_ids = _read_matrix(section_dir)
    coords, cell_ids = _read_centroids(section_dir)
    counts, coords = _align_cells(counts, mat_ids, coords, cell_ids)
    keep = _real_gene_mask(genes)
    genes = [g for g, k in zip(genes, keep) if k]
    counts = counts[:, np.asarray(keep, dtype=bool)]
    return XeniumSection(coords=coords, counts=counts, genes=genes)


def read_panel(path: str | Path) -> set[str]:
    """Read the Xenium gene set from gene_panel.json(.gz) or panel.tsv(.gz)."""
    path = Path(path)
    text = _read_text_maybe_gz(path)
    if path.name.endswith(".json") or path.name.endswith(".json.gz"):
        return _genes_from_panel_json(json.loads(text))
    return {line.split("\t")[0].strip() for line in text.splitlines() if line.strip()}


# --------------------------------------------------------------------------------------------------
# Niche binning + serial-section registration
# --------------------------------------------------------------------------------------------------


def bin_pseudobulk(
    section: XeniumSection, bin_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate cells into square `bin_um` niches. Returns (bin_centers(m,2), pb_counts(m,G), n_cells(m,)).

    pb_counts = summed raw counts per bin; library-size + log1p normalization is applied later, after
    the two replicates' bins are matched, so both are normalized on a shared footing.
    """
    xy = section.coords
    origin = xy.min(0)
    ij = np.floor((xy - origin) / bin_um).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    m, g = keys.shape[0], section.counts.shape[1]
    pb = np.zeros((m, g), dtype=np.float64)
    np.add.at(pb, inv, section.counts)
    ncell = np.bincount(inv, minlength=m).astype(np.float64)
    centers = origin + (keys + 0.5) * bin_um
    return centers, pb, ncell


def rigid_register(
    src_pts: np.ndarray, dst_pts: np.ndarray, bin_um: float
) -> np.ndarray:
    """Estimate a rigid (rotation+translation) map taking src cloud onto dst, resolving the 180-deg flip.

    Serial-section alignment at niche resolution: a PCA principal-axis + 4-flip initialization (scored by
    binned tissue-density overlap), then ICP refinement on the tissue point clouds. PCA fixes the gross
    rotation/flip from the (anisotropic) tissue shape; ICP tightens it using the boundary -- this is what
    makes the alignment robust on real serial sections (which have an irregular outline). The
    +/-1-bin perturbation sensitivity pre-registered in preregistration.md sec 11-D guards residual error.
    Returns a (2,3) affine [R|t] mapping src -> dst frame.
    """
    sc_, dc = src_pts.mean(0), dst_pts.mean(0)
    Rs = _principal_axes(src_pts - sc_)
    Rd = _principal_axes(dst_pts - dc)
    best, best_score = None, -np.inf
    for flip in (np.eye(2), np.diag([1.0, -1.0]), np.diag([-1.0, 1.0]), -np.eye(2)):
        R = Rd @ flip @ Rs.T
        t = dc - R @ sc_
        moved = (R @ src_pts.T).T + t
        score = _density_overlap(moved, dst_pts, bin_um)
        if score > best_score:
            best, best_score = (R, t), score
    R, t = best
    R, t = _icp_refine(src_pts, dst_pts, R, t)
    return np.hstack([R, t[:, None]])


def _icp_refine(
    src: np.ndarray,
    dst: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    iters: int = 12,
    sample: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative-closest-point refine of an initial rigid (R,t) using nearest-neighbor correspondences."""
    from scipy.spatial import cKDTree

    si = (
        src
        if len(src) <= sample
        else src[np.linspace(0, len(src) - 1, sample).astype(int)]
    )
    dtree = cKDTree(dst)
    for _ in range(iters):
        moved = (R @ si.T).T + t
        _, idx = dtree.query(moved)
        R_new, t_new = _procrustes_rigid(si, dst[idx])
        if np.allclose(R_new, R, atol=1e-6) and np.allclose(t_new, t, atol=1e-3):
            R, t = R_new, t_new
            break
        R, t = R_new, t_new
    return R, t


def _procrustes_rigid(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid (rotation+translation, no scale) mapping A->B via SVD of the cross-covariance."""
    muA, muB = A.mean(0), B.mean(0)
    H = (A - muA).T @ (B - muB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection guard
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, muB - R @ muA


def apply_affine(pts: np.ndarray, aff: np.ndarray) -> np.ndarray:
    return (aff[:, :2] @ pts.T).T + aff[:, 2]


def matched_bins(
    sec1: XeniumSection,
    sec2: XeniumSection,
    bin_um: float = 100.0,
    min_cells: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Match serial replicates at niche resolution in a shared frame.

    Returns (z1(k,G), z2(k,G), centers(k,2), genes) where z* are per-gene z-scored log1p-CPM pseudobulk
    over the SHARED panel genes and the k bins occupied (>= min_cells) in BOTH replicates after Rep2 is
    rigidly registered into Rep1's frame. z1/z2 plug straight into oracle.noise_floor.
    """
    genes = [g for g in sec1.genes if g in set(sec2.genes)]
    g1 = {g: i for i, g in enumerate(sec1.genes)}
    g2 = {g: i for i, g in enumerate(sec2.genes)}
    idx1 = np.array([g1[g] for g in genes])
    idx2 = np.array([g2[g] for g in genes])

    aff = rigid_register(sec2.coords, sec1.coords, bin_um)
    sec2_reg = XeniumSection(apply_affine(sec2.coords, aff), sec2.counts, sec2.genes)

    c1, pb1, n1 = bin_pseudobulk(sec1, bin_um)
    c2, pb2, n2 = bin_pseudobulk(sec2_reg, bin_um)

    key1 = _bin_keys(c1, bin_um)
    key2 = _bin_keys(c2, bin_um)
    map2 = {tuple(k): i for i, k in enumerate(key2)}
    rows1, rows2, centers = [], [], []
    for i, k in enumerate(key1):
        j = map2.get(tuple(k))
        if j is not None and n1[i] >= min_cells and n2[j] >= min_cells:
            rows1.append(i)
            rows2.append(j)
            centers.append(c1[i])
    if not rows1:
        raise RuntimeError(
            "no co-occupied niches after registration -- check bin_um/min_cells or registration quality."
        )
    pb1m = pb1[np.array(rows1)][:, idx1]
    pb2m = pb2[np.array(rows2)][:, idx2]
    z1, z2 = _normalize_pair(pb1m, pb2m)
    return z1, z2, np.array(centers), genes


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _normalize_pair(pb1: np.ndarray, pb2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Library-size (CPM-like) + log1p, then per-gene z-score using stats POOLED over both replicates."""
    l1 = _log1p_cpm(pb1)
    l2 = _log1p_cpm(pb2)
    mu = 0.5 * (l1.mean(0) + l2.mean(0))
    sd = 0.5 * (l1.std(0) + l2.std(0)) + 1e-8
    return (l1 - mu) / sd, (l2 - mu) / sd


def _log1p_cpm(pb: np.ndarray) -> np.ndarray:
    lib = pb.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(pb / lib * 1e4)


def _principal_axes(centered: np.ndarray) -> np.ndarray:
    """Columns = principal axes (2x2 rotation), ordered by descending variance."""
    cov = np.cov(centered.T)
    w, V = np.linalg.eigh(cov)
    return V[:, np.argsort(w)[::-1]]


def _density_overlap(a: np.ndarray, b: np.ndarray, bin_um: float) -> float:
    """Correlation of binned point densities -- higher = better registration."""
    lo = np.minimum(a.min(0), b.min(0))
    hi = np.maximum(a.max(0), b.max(0))
    nb = np.maximum(((hi - lo) / bin_um).astype(int) + 1, 1)
    ha = _hist2d(a, lo, bin_um, nb)
    hb = _hist2d(b, lo, bin_um, nb)
    if ha.std() < 1e-8 or hb.std() < 1e-8:
        return -np.inf
    return float(np.corrcoef(ha.ravel(), hb.ravel())[0, 1])


def _hist2d(p: np.ndarray, lo: np.ndarray, bin_um: float, nb: np.ndarray) -> np.ndarray:
    ij = np.clip(((p - lo) / bin_um).astype(int), 0, nb - 1)
    h = np.zeros(tuple(nb))
    np.add.at(h, (ij[:, 0], ij[:, 1]), 1.0)
    return h


def _bin_keys(centers: np.ndarray, bin_um: float) -> np.ndarray:
    return np.round(centers / bin_um).astype(np.int64)


def _real_gene_mask(genes: list[str]) -> list[bool]:
    return [not g.startswith(_NONGENE_PREFIXES) for g in genes]


def _align_cells(
    counts: np.ndarray, mat_ids: np.ndarray, coords: np.ndarray, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Order counts and coords by a common cell_id set (h5 and cells file can differ in order)."""
    if np.array_equal(mat_ids, cell_ids):
        return counts, coords
    pos = {c: i for i, c in enumerate(cell_ids)}
    keep = [i for i, c in enumerate(mat_ids) if c in pos]
    counts = counts[keep]
    order = [pos[mat_ids[i]] for i in keep]
    return counts, coords[order]


def _genes_from_panel_json(obj) -> set[str]:
    found: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("name"), str):
                found.add(o["name"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return {g for g in found if not g.startswith(_NONGENE_PREFIXES)}


def _first_existing(d: Path, patterns: tuple[str, ...]) -> Path | None:
    for pat in patterns:
        hits = (
            sorted(d.glob(pat))
            if any(c in pat for c in "*?[")
            else ([d / pat] if (d / pat).exists() else [])
        )
        for h in hits:
            if h.exists():
                return h
    return None


def _read_text_maybe_gz(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def _read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _pick(columns, candidates: tuple[str, ...]) -> str:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"none of {candidates} in columns {list(columns)}")
