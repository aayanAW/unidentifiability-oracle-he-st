"""Download-free CPU tests for the trainable dual-head oracle (HYBRID substrate, predictor.py).

Validates the module contract and that it can actually learn, without touching timm/the network:
  - frozen-embedding mode (backbone=None) returns (mean, logvar) of the right shape, logvar clamped;
  - image mode runs through a tiny stub backbone (stands in for the ungated DINOv2-S);
  - beta-NLL matches its closed form and stays finite; beta=0 reduces to plain Gaussian NLL;
  - a few SGD steps on a linear-Gaussian problem reduce the loss and the mean head tracks the target
    (the smoke that the dual head trains -- the cluster run only scales this up).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictor import DualHeadOracle, beta_nll_loss  # noqa: E402


def test_forward_embedding_mode_shapes_and_logvar_clamp():
    torch.manual_seed(0)
    model = DualHeadOracle(n_genes=5, in_dim=16, min_logvar=-4.0, max_logvar=4.0)
    x = torch.randn(8, 16)
    mean, logvar = model(x)
    assert mean.shape == (8, 5) and logvar.shape == (8, 5)
    assert torch.isfinite(mean).all() and torch.isfinite(logvar).all()
    assert logvar.min() >= -4.0 - 1e-5 and logvar.max() <= 4.0 + 1e-5


def test_forward_image_mode_runs_through_backbone():
    """A (N,3,H,W) batch must route through the backbone; embeddings (N,d) must bypass it."""

    class StubBackbone(torch.nn.Module):
        num_features = 16

        def forward(self, x):  # (N,3,H,W) -> (N,16)
            return x.mean(dim=(2, 3)).repeat(1, 16 // 3 + 1)[:, :16]

    model = DualHeadOracle(n_genes=3, in_dim=16, backbone=StubBackbone())
    img = torch.randn(4, 3, 28, 28)
    mean, logvar = model(img)
    assert mean.shape == (4, 3) and logvar.shape == (4, 3)


def test_image_mode_imagenet_normalizes_before_backbone():
    """uint8-range patches must be ImageNet-normalized before the backbone (end-to-end correctness)."""

    class RecordBackbone(torch.nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return x.mean(dim=(2, 3)).repeat(1, 8 // 3 + 1)[:, :8]

    bb = RecordBackbone()
    model = DualHeadOracle(n_genes=2, in_dim=8, backbone=bb)
    white = torch.full((1, 3, 16, 16), 255.0)  # all-white patch in [0,255]
    model(white)
    # (255/255 - mean)/std per channel
    expected = (1.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    assert torch.allclose(bb.seen[0, :, 0, 0], expected, atol=1e-5)


def test_beta_nll_matches_closed_form_and_beta_zero_is_plain_nll():
    torch.manual_seed(1)
    mean = torch.randn(6, 3)
    logvar = torch.randn(6, 3)
    target = torch.randn(6, 3)
    var = logvar.exp()
    plain = 0.5 * ((target - mean) ** 2 / var + logvar)
    # beta=0 -> plain Gaussian NLL (mean over all elements)
    assert torch.allclose(beta_nll_loss(mean, logvar, target, beta=0.0), plain.mean())
    # beta=0.5 -> each term weighted by stop-grad(var)^0.5
    weighted = (plain * var.detach() ** 0.5).mean()
    assert torch.allclose(beta_nll_loss(mean, logvar, target, beta=0.5), weighted)
    assert torch.isfinite(beta_nll_loss(mean, logvar, target, beta=0.5))


def test_dual_head_learns_linear_gaussian():
    torch.manual_seed(0)
    n, d, g = 256, 16, 4
    X = torch.randn(n, d)
    W = torch.randn(d, g)
    Y = X @ W + 0.1 * torch.randn(n, g)

    model = DualHeadOracle(n_genes=g, in_dim=d, trunk_dims=(32,), dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    with torch.no_grad():
        m0, lv0 = model(X)
        loss0 = beta_nll_loss(m0, lv0, Y, beta=0.5).item()
    for _ in range(150):
        opt.zero_grad()
        mean, logvar = model(X)
        loss = beta_nll_loss(mean, logvar, Y, beta=0.5)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mean, _ = model(X)
        loss1 = beta_nll_loss(*model(X), Y, beta=0.5).item()
    assert loss1 < loss0, f"loss did not decrease: {loss0:.3f} -> {loss1:.3f}"
    # mean head tracks the target (per-gene Pearson r averaged)
    r = np.mean(
        [np.corrcoef(mean[:, j].numpy(), Y[:, j].numpy())[0, 1] for j in range(g)]
    )
    assert r > 0.8, f"mean head failed to fit (avg r={r:.2f})"


if __name__ == "__main__":
    test_forward_embedding_mode_shapes_and_logvar_clamp()
    test_forward_image_mode_runs_through_backbone()
    test_image_mode_imagenet_normalizes_before_backbone()
    test_beta_nll_matches_closed_form_and_beta_zero_is_plain_nll()
    test_dual_head_learns_linear_gaussian()
    print("ALL predictor TESTS PASS")
