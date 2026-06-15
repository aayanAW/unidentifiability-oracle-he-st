"""Download-free tests for the H&E morphology pipeline: the SB4 registration fix + lazy window reader.

The audit (rollout 9, SB4; fixed in the exploratory script rollout 14) found the H&E<->Xenium mapping was
inverted: the 10x Post-Xenium homography maps H&E pixels -> Xenium pixels, so to draw a patch at a Xenium
niche we must apply `inv(H)` to the niche expressed in XENIUM PIXELS (microns / um_per_px), not the
forward `H` to microns. With the wrong mapping only ~80% of niches landed in-bounds and morph->ST R^2
collapsed; with the fix, 399/399 breast niches map in-bounds and R^2 max went 0.27 -> 0.38. These tests
pin the corrected mapping (`src/embeddings._um_to_px`) and the memory-safe windowed reader so the real
pipeline (loaders.load_breast_triad -> embeddings.embed_bins) never regresses to the broken mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import _read_windows_lazy, _um_to_px  # noqa: E402


def test_um_to_px_inverts_homography_into_he_pixels():
    """`_um_to_px` must return H&E pixels p such that the forward homography H @ p == niche Xenium pixels.

    H&E-px -> Xenium-px is the 10x convention; we need the inverse mapped onto microns/um_per_px (SB4)."""
    rng = np.random.default_rng(0)
    H = np.array([[1.05, 0.02, 30.0], [-0.01, 0.98, -12.0], [0.0, 0.0, 1.0]])
    um_per_px = 0.2125
    centers_um = rng.uniform(0.0, 500.0, size=(50, 2))

    he_px = _um_to_px(centers_um, H, um_per_px)

    pts = np.hstack([he_px, np.ones((len(he_px), 1))])
    forward = (H @ pts.T).T
    recovered_xen_px = forward[:, :2] / forward[:, 2:3]
    assert np.allclose(recovered_xen_px, centers_um / um_per_px, atol=1e-6), (
        "forward-mapping the returned H&E pixels must recover the niche Xenium pixels (SB4 inverse map)"
    )


def test_um_to_px_no_homography_scales_microns_to_pixels():
    centers_um = np.array([[100.0, 200.0], [0.0, 50.0]])
    px = _um_to_px(centers_um, None, 0.2125)
    assert np.allclose(px, centers_um / 0.2125)


def test_lazy_window_reader_reads_at_requested_pixel(tmp_path):
    """The windowed reader must read patch_px windows centered on the requested pixel WITHOUT loading the
    whole image, and report how many landed in-bounds."""
    import tifffile

    img = np.zeros((400, 400, 3), np.uint8)
    img[184:216, 184:216] = 255  # bright 32px square centered at pixel (200, 200)
    p = tmp_path / "toy.ome.tif"
    tifffile.imwrite(str(p), img)

    centers_px = np.array([[200.0, 200.0], [10_000.0, 10_000.0]])  # one in, one out
    patches, in_bounds = _read_windows_lazy(p, centers_px, patch_px=64)

    assert patches.shape == (2, 64, 64, 3)
    assert in_bounds == 1, "exactly one center is inside the 400x400 image"
    assert patches[0, 32, 32].max() == 255, (
        "in-bounds patch centered on the bright square"
    )
    assert patches[1].max() == 0, "out-of-bounds center stays zero-padded"


def test_channels_first_ome_tiff_is_transposed_not_zeroed(tmp_path):
    """A (C,Y,X) OME-TIFF must be detected and transposed so patches carry tissue, not all-zero (review HIGH)."""
    import tifffile

    img_cf = np.zeros((3, 400, 400), np.uint8)
    img_cf[:, 184:216, 184:216] = (
        200  # bright square at pixel (200,200), channels-first
    )
    p = tmp_path / "cf.ome.tif"
    tifffile.imwrite(str(p), img_cf)

    patches, in_bounds = _read_windows_lazy(p, np.array([[200.0, 200.0]]), patch_px=64)
    assert in_bounds == 1
    assert patches[0, 32, 32].max() == 200, (
        "channels-first image must not yield all-zero patches"
    )


def test_extract_patches_raises_when_all_out_of_bounds(tmp_path):
    """The SB4 invariant: 0 in-bounds means a bad homography -> refuse to embed all-zero patches (review HIGH)."""
    import tifffile

    from src.embeddings import _extract_patches

    p = tmp_path / "small.ome.tif"
    tifffile.imwrite(str(p), np.zeros((300, 300, 3), np.uint8))
    far = np.array([[1e6, 1e6], [2e6, 2e6]])  # microns far outside, H=None scales to px
    try:
        _extract_patches(p, far, None, patch_px=64, um_per_px=0.2125)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on 0 in-bounds niches")


def test_non_uint8_image_rejected(tmp_path):
    import tifffile

    p = tmp_path / "u16.ome.tif"
    tifffile.imwrite(str(p), (np.ones((128, 128, 3)) * 1000).astype(np.uint16))
    try:
        _read_windows_lazy(p, np.array([[64.0, 64.0]]), patch_px=32)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-uint8 H&E")


if __name__ == "__main__":
    import tempfile

    test_um_to_px_inverts_homography_into_he_pixels()
    test_um_to_px_no_homography_scales_microns_to_pixels()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        test_lazy_window_reader_reads_at_requested_pixel(d)
        test_channels_first_ome_tiff_is_transposed_not_zeroed(d)
        test_extract_patches_raises_when_all_out_of_bounds(d)
        test_non_uint8_image_rejected(d)
    print("ALL embeddings TESTS PASS")
