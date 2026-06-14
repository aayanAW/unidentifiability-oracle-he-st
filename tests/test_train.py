"""CPU smoke tests for the dual-head training harness (train.py).

The full run is multi-GPU/DDP on the cluster; here we prove the mechanics on CPU with tiny synthetic data:
  - device selection returns a usable torch.device;
  - the distributed guard is a no-op when WORLD_SIZE is unset (the Mac path);
  - fit_dual_head reduces held-out loss and the trained mean head tracks the target;
  - checkpoint save -> load round-trips predictions exactly;
  - the `--smoke` CLI entrypoint trains a couple of epochs and writes a checkpoint, returning 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictor import DualHeadOracle  # noqa: E402
from src.train import (  # noqa: E402
    TrainConfig,
    fit_dual_head,
    is_distributed,
    load_checkpoint,
    main,
    pick_device,
    save_checkpoint,
)


def _linear_gaussian(n=256, d=16, g=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, g)).astype(np.float32)
    Y = (X @ W + 0.1 * rng.standard_normal((n, g))).astype(np.float32)
    return X, Y


def test_pick_device_and_distributed_guard():
    dev = pick_device("cpu")
    assert isinstance(dev, torch.device) and dev.type == "cpu"
    assert is_distributed() is False  # WORLD_SIZE unset in the test env


def test_fit_dual_head_reduces_val_loss():
    X, Y = _linear_gaussian()
    idx = np.arange(len(X))
    val = idx[:64]
    train = idx[64:]
    model = DualHeadOracle(
        n_genes=Y.shape[1], in_dim=X.shape[1], trunk_dims=(32,), dropout=0.0
    )
    cfg = TrainConfig(
        epochs=80, lr=1e-2, batch_size=64, device="cpu", amp=False, seed=0
    )
    model, history = fit_dual_head(model, X, Y, train, val, cfg)
    assert history["val_loss"][-1] < history["val_loss"][0], "val loss did not improve"
    mean, _ = model.predict(torch.from_numpy(X[val]))
    r = np.mean(
        [np.corrcoef(mean[:, j].numpy(), Y[val][:, j])[0, 1] for j in range(Y.shape[1])]
    )
    assert r > 0.8, f"mean head failed to fit held-out (avg r={r:.2f})"


def test_fit_dual_head_image_mode_with_backbone():
    """End-to-end wiring: 4D image patches flow through a backbone in fit_dual_head and the loss drops."""

    class StubBackbone(torch.nn.Module):
        num_features = 12

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 12, 3, padding=1)

        def forward(self, x):  # (N,3,H,W) -> (N,12)
            return self.proj(x).mean(dim=(2, 3))

    rng = np.random.default_rng(0)
    n, g = 48, 3
    X = (rng.random((n, 3, 16, 16)) * 255).astype(np.float32)  # uint8-range patches
    # target is a function of the patch mean so the backbone+head can learn it
    Y = (X.reshape(n, 3, -1).mean(-1) @ rng.standard_normal((3, g))).astype(np.float32)
    model = DualHeadOracle(
        n_genes=g, in_dim=12, backbone=StubBackbone(), trunk_dims=(16,), dropout=0.0
    )
    cfg = TrainConfig(
        epochs=40, lr=1e-2, batch_size=24, device="cpu", amp=False, seed=0
    )
    idx = np.arange(n)
    _, history = fit_dual_head(model, X, Y, idx[12:], idx[:12], cfg)
    assert history["val_loss"][-1] < history["val_loss"][0], (
        "image-mode loss did not improve"
    )


def test_checkpoint_round_trip(tmp_path):
    X, Y = _linear_gaussian(seed=1)
    model = DualHeadOracle(
        n_genes=Y.shape[1], in_dim=X.shape[1], trunk_dims=(16,), dropout=0.0
    )
    ckpt = tmp_path / "model.pt"
    meta = {"n_genes": Y.shape[1], "in_dim": X.shape[1], "trunk_dims": (16,)}
    save_checkpoint(ckpt, model, meta=meta, epoch=3)

    fresh = DualHeadOracle(
        n_genes=Y.shape[1], in_dim=X.shape[1], trunk_dims=(16,), dropout=0.0
    )
    loaded_meta = load_checkpoint(ckpt, fresh)
    assert loaded_meta["epoch"] == 3 and loaded_meta["meta"]["n_genes"] == Y.shape[1]

    xb = torch.from_numpy(X[:8])
    m0, v0 = model.predict(xb)
    m1, v1 = fresh.predict(xb)
    assert torch.allclose(m0, m1) and torch.allclose(v0, v1)


def test_smoke_cli_writes_checkpoint(tmp_path):
    out = tmp_path / "run"
    rc = main(["--smoke", "--epochs", "2", "--device", "cpu", "--out", str(out)])
    assert rc == 0
    assert (out / "model.pt").exists(), "smoke run must write a checkpoint"


if __name__ == "__main__":
    import tempfile

    test_pick_device_and_distributed_guard()
    test_fit_dual_head_reduces_val_loss()
    test_fit_dual_head_image_mode_with_backbone()
    with tempfile.TemporaryDirectory() as d:
        test_checkpoint_round_trip(Path(d))
        test_smoke_cli_writes_checkpoint(Path(d))
    print("ALL train TESTS PASS")
