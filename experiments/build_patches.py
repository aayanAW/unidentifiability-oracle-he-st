"""Build the END-TO-END training input: H&E image patches at the niche centers (the cluster step's data).

Writes data/cache/patches_breast.npz with X=(N,3,224,224) uint8 patches, Y=z_est (cleaner-reference
target), coords, and the encoder/confirmatory labels. With this present, the end-to-end fine-tune is a
one-line submit:  BACKBONE=vit_small_patch14_dinov2.lvd142m DATA=data/cache/patches_breast.npz \
                  sbatch scripts/train_dualhead.sbatch
(src.train builds the backbone, ImageNet-normalizes the uint8 patches, and fine-tunes end-to-end.)

Cluster-free prep: runs on this Mac (only reads patch windows lazily; no GPU). EXPLORATORY substrate
(ungated DINOv2-S downstream), breast only.
Run:  python3 experiments/build_patches.py data --bin-um 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import xenium_io as xio  # noqa: E402
from src.embeddings import _extract_patches, _load_homography  # noqa: E402
from src.loaders import XENIUM_UM_PER_PX, load_xenium_noise_floor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--bin-um", type=float, default=300.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--patch-px", type=int, default=224)
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    args = ap.parse_args()
    root = Path(args.data_dir)

    z1, z2, centers, genes = load_xenium_noise_floor(
        args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
    )
    y = 0.5 * (z1 + z2)
    he = xio._first_existing(  # noqa: SLF001
        root / "rep1", ("Post-Xenium_HE_Rep1.ome.tif", "*HE*.ome.tif", "*HE*ome.tif.gz")
    )
    if he is None:
        print("[build_patches] no H&E under data/rep1.")
        return 3
    homog = _load_homography(
        xio._first_existing(root / "rep1", ("*HE*homography.csv.gz",))  # noqa: SLF001
    )
    patches = _extract_patches(
        he, centers, homog, args.patch_px, XENIUM_UM_PER_PX
    )  # (N,H,W,3) uint8
    X = np.moveaxis(patches, -1, 1).astype(np.uint8)  # -> (N,3,H,W) for the backbone

    out = Path(__file__).resolve().parents[1] / "data" / "cache" / "patches_breast.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        Y=y.astype(np.float32),
        coords=centers,
        genes=np.array(genes),
        # the patches are the SUBSTRATE for an end-to-end fine-tune of this ungated backbone; label it so a
        # downstream run can never mistake it for a confirmatory UNI run (preregistration.md sec 10).
        encoder_name=np.array(f"end-to-end:{args.backbone}"),
        is_confirmatory=np.bool_(False),
    )
    nz = (X.reshape(len(X), -1).max(1) > 0).sum()
    print(
        f"[build_patches] wrote {out}  X={X.shape} uint8  Y={y.shape}  "
        f"non-zero patches={nz}/{len(X)}  (-> BACKBONE=... DATA={out} sbatch scripts/train_dualhead.sbatch)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
