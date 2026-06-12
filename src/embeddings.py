"""Morphology featurization: H&E patch embeddings at niche-bin centers.

Pre-registered encoders are UNI (MahmoodLab/UNI) and Virchow2 (paige-ai/Virchow2) -- both GATED on
Hugging Face and (UNI) restricted to institutional-email accounts. Until access is granted, an UNGATED
fallback (timm DINOv2-ViT, or a deterministic intensity-texture descriptor if timm/torch are absent)
lets the whole oracle pipeline run end-to-end so the machinery is validated on real tissue. Fallback
runs are EXPLORATORY (preregistration.md sec 10) -- a real confirmatory U claim must use the frozen UNI
embeddings. `encoder_name` is recorded with every cache so no fallback run is ever mistaken for the real one.

Embeddings are cached to disk (.npz) keyed by encoder + patch geometry; the 1.3 GB H&E OME-TIFF is read
once, tiled, embedded, and can then be deleted (the cache is a few MB).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GATED_ENCODERS = {
    "uni": "MahmoodLab/UNI",
    "uni2": "MahmoodLab/UNI2-h",
    "virchow2": "paige-ai/Virchow2",
}
DEFAULT_FALLBACK = (
    "vit_small_patch14_dinov2.lvd142m"  # ungated DINOv2-S, the correct timm model name
)


@dataclass(frozen=True)
class MorphFeatures:
    features: np.ndarray  # (n_bins, d) per-bin embedding
    encoder_name: (
        str  # exact encoder used -- "uni" (confirmatory) vs a fallback (exploratory)
    )
    is_confirmatory: bool  # True only for the pre-registered gated encoders


def embed_bins(
    he_image_path: str | Path,
    bin_centers_um: np.ndarray,
    homography_csv: str | Path | None,
    encoder: str = "uni",
    patch_px: int = 224,
    um_per_px: float = 0.2125,
    cache_dir: str | Path = "data/cache",
) -> MorphFeatures:
    """Embed an H&E patch at each niche-bin center. Falls back to an ungated encoder if UNI is unavailable.

    `homography_csv` is the 10x Post-Xenium H&E->Xenium homography (GEO `*_homography.csv.gz`): it maps
    Xenium micron coordinates into H&E pixel space so the patch is drawn at the right place. um_per_px is
    the Xenium pixel size (0.2125 um) used only if no homography is supplied.
    """
    requested_gated = encoder in GATED_ENCODERS
    conf_cache = _cache_path(
        cache_dir, he_image_path, bin_centers_um, encoder, patch_px, tag="confirmatory"
    )
    fb_cache = _cache_path(
        cache_dir, he_image_path, bin_centers_um, encoder, patch_px, tag="fallback"
    )
    # audit H7: a GATED request never reuses a fallback cache, so a confirmatory run is retried once UNI
    # access is fixed (instead of silently returning the stale ungated embeddings forever).
    if conf_cache.exists():
        return _load_cache(conf_cache)
    if not requested_gated and fb_cache.exists():
        return _load_cache(fb_cache)

    H = _load_homography(homography_csv)
    patches = _extract_patches(he_image_path, bin_centers_um, H, patch_px, um_per_px)
    feats, name, confirmatory = _run_encoder(patches, encoder)

    cache = conf_cache if confirmatory else fb_cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, features=feats, encoder_name=name, is_confirmatory=confirmatory
    )
    return MorphFeatures(feats, name, confirmatory)


def _load_cache(path: Path) -> MorphFeatures:
    d = np.load(path, allow_pickle=True)
    return MorphFeatures(
        d["features"], str(d["encoder_name"]), bool(d["is_confirmatory"])
    )


def _run_encoder(patches: np.ndarray, encoder: str) -> tuple[np.ndarray, str, bool]:
    """patches (n,H,W,3) uint8 -> (features, encoder_name, is_confirmatory)."""
    if encoder in GATED_ENCODERS:
        try:
            return _embed_gated(patches, encoder)
        except (
            Exception
        ) as e:  # gating / auth / network failure -> exploratory fallback
            print(
                f"[embeddings] gated encoder '{encoder}' ({GATED_ENCODERS[encoder]}) unavailable: {e}\n"
                f"[embeddings] falling back to ungated '{DEFAULT_FALLBACK}' -- run is EXPLORATORY, "
                f"not a confirmatory U claim (preregistration.md sec 10)."
            )
            return _embed_fallback(patches)
    return _embed_fallback(patches, encoder)


def _embed_gated(patches: np.ndarray, encoder: str) -> tuple[np.ndarray, str, bool]:
    import timm
    import torch

    repo = GATED_ENCODERS[encoder]
    model = timm.create_model(f"hf-hub:{repo}", pretrained=True, num_classes=0)
    model.eval()
    feats = _forward_timm(model, patches, torch)
    return feats, encoder, True


def _embed_fallback(
    patches: np.ndarray, encoder: str = DEFAULT_FALLBACK
) -> tuple[np.ndarray, str, bool]:
    try:
        import timm
        import torch

        # dynamic_img_size lets ViT/DINOv2 interpolate position embeddings to the patch size we feed
        # (DINOv2-lvd142m natively expects 518px); fall back to a plain create_model for non-ViT names.
        try:
            model = timm.create_model(
                encoder, pretrained=True, num_classes=0, dynamic_img_size=True
            )
        except TypeError:
            model = timm.create_model(encoder, pretrained=True, num_classes=0)
        model.eval()
        return _forward_timm(model, patches, torch), encoder, False
    except Exception as e:
        print(
            f"[embeddings] timm fallback unavailable ({e}); using deterministic descriptor."
        )
        return _descriptor(patches), "intensity-texture-descriptor", False


def _forward_timm(model, patches: np.ndarray, torch) -> np.ndarray:
    """Mean/std ImageNet-normalize, batch through a timm model, return pooled features."""
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            chunk = patches[i : i + 64].astype(np.float32) / 255.0
            chunk = (chunk - mean) / std
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, 0)


def _descriptor(patches: np.ndarray) -> np.ndarray:
    """Torch-free fallback: per-channel mean/std + simple gradient-energy texture. Deterministic, ungated.

    Not a research encoder -- only keeps the pipeline runnable when neither UNI nor timm is installed.
    """
    p = patches.astype(np.float32) / 255.0
    mean = p.mean((1, 2))
    std = p.std((1, 2))
    gx = np.abs(np.diff(p, axis=2)).mean((1, 2))
    gy = np.abs(np.diff(p, axis=1)).mean((1, 2))
    return np.concatenate([mean, std, gx, gy], axis=1)


# --------------------------------------------------------------------------------------------------
# H&E reading + patch extraction
# --------------------------------------------------------------------------------------------------


def _extract_patches(
    he_path: str | Path,
    centers_um: np.ndarray,
    H: np.ndarray | None,
    patch_px: int,
    um_per_px: float,
) -> np.ndarray:
    """Return (n_bins, patch_px, patch_px, 3) uint8 patches centered on each niche.

    Memory-safe: the breast Post-Xenium H&E is ~1.3 GB; we map each niche to its H&E pixel (SB4 inverse
    homography) and read only the patch_px window via a lazy zarr store, never the whole image into RAM.
    """
    he_path = _ensure_gunzipped(Path(he_path))
    centers_px = _um_to_px(np.asarray(centers_um, dtype=float), H, um_per_px)
    patches, _in_bounds = _read_windows_lazy(he_path, centers_px, patch_px)
    return patches


def _ensure_gunzipped(path: Path) -> Path:
    """A .gz OME-TIFF cannot be opened as a lazy zarr store; gunzip it once to a sibling and reuse it."""
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")  # drop the trailing .gz
    if not out.exists():
        import gzip
        import shutil

        with gzip.open(path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
    return out


def _read_windows_lazy(
    he_path: str | Path, centers_px: np.ndarray, patch_px: int
) -> tuple[np.ndarray, int]:
    """Read patch_px windows at the given H&E pixel centers via a lazy zarr store (audit SB4 port).

    Returns ((n, patch_px, patch_px, 3) uint8 patches, n_in_bounds). Out-of-bounds centers stay zero-padded.
    """
    import tifffile
    import zarr

    store = tifffile.imread(str(he_path), aszarr=True)
    z = zarr.open(store, mode="r")
    arr = (
        z[0] if isinstance(z, zarr.hierarchy.Group) else z
    )  # pyramidal series -> level 0
    h, w = int(arr.shape[0]), int(arr.shape[1])
    half = patch_px // 2
    out = np.zeros((len(centers_px), patch_px, patch_px, 3), np.uint8)
    in_bounds = 0
    for k, (cx, cy) in enumerate(centers_px):
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        xs, ys = max(0, x0), max(0, y0)
        xe, ye = min(w, x0 + patch_px), min(h, y0 + patch_px)
        if xe <= xs or ye <= ys:
            continue
        tile = np.asarray(arr[ys:ye, xs:xe])
        if tile.ndim == 2:  # grayscale -> stack to 3 channels
            tile = np.stack([tile] * 3, axis=-1)
        if tile.dtype != np.uint8:
            tile = (255 * (tile / (tile.max() + 1e-8))).astype(np.uint8)
        out[k, ys - y0 : ye - y0, xs - x0 : xe - x0] = tile[..., :3]
        in_bounds += 1
    return out, in_bounds


def _um_to_px(
    centers_um: np.ndarray, H: np.ndarray | None, um_per_px: float
) -> np.ndarray:
    """Map niche centers (microns) to H&E pixels.

    The 10x Post-Xenium homography H maps H&E pixels -> Xenium pixels. To place a patch at a Xenium niche
    we need the inverse, applied to the niche expressed in XENIUM PIXELS (microns / um_per_px). Applying
    the forward H to microns (the pre-SB4 bug) sent ~20% of breast niches out of bounds and depressed
    morph->ST R^2; the inverse map lands 399/399 in-bounds (audit SB4 fix, rollout 14).
    """
    centers_um = np.asarray(centers_um, dtype=float)
    if H is None:
        return centers_um / um_per_px
    xen_px = centers_um / um_per_px
    pts = np.hstack([xen_px, np.ones((len(xen_px), 1))])
    proj = (np.linalg.inv(H) @ pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def _load_homography(csv: str | Path | None) -> np.ndarray | None:
    if csv is None:
        return None
    import pandas as pd

    df = pd.read_csv(csv, header=None)
    vals = df.to_numpy(dtype=np.float64)
    if vals.size == 9:
        return vals.reshape(3, 3)
    return None


def _cache_path(
    cache_dir, he_path, centers, encoder, patch_px, tag="confirmatory"
) -> Path:
    # key includes patch geometry + a content fingerprint of the bin centers so a different patch size or
    # niche layout never silently reuses stale embeddings (audit H7). `tag` separates confirmatory (gated)
    # caches from fallback ones so a gated request never returns the fallback.
    fp = f"{float(centers.sum()):.3f}|{float((centers**2).sum()):.3f}"
    key = hashlib.sha1(
        f"{Path(he_path).name}|{centers.shape}|{fp}|{encoder}|{patch_px}|{tag}".encode()
    ).hexdigest()[:16]
    return Path(cache_dir) / f"morph_{encoder}_{tag}_{key}.npz"
