"""Trainable dual-head oracle `f` (HYBRID stage): ungated backbone + H&E->ST mean head + variance/U head.

This is the trained substrate that replaces the frozen RF/Ridge of the C1 feasibility vehicle
(architecture.md staged C1 -> HYBRID; plan.md "Direction update", rollout 12). A higher-capacity trained
`f` leaves only irreducible residual, tightening U toward true unidentifiability -- directly addressing
audit C1 (a low-capacity `f` under-fits and inflates U: identifiable genes keep U~=0.4 instead of ~0). The
variance head is the LEARNED aleatoric estimate the selective layer defers on; comparing its selective-risk
efficiency against the raw ensemble-variance baseline is the HYBRID dual-head's headline value-add.

Two operating modes share ONE module:
  - frozen-embedding mode (runs on this Mac): `backbone=None`; `forward` consumes pre-cached DINOv2
    embeddings and trains only trunk + heads -> the FIRST trained-oracle result, cluster-free.
  - end-to-end mode (cluster): `backbone` = the ungated DINOv2-S (`vit_small_patch14_dinov2.lvd142m`);
    `forward` consumes H&E image patches and fine-tunes the whole stack. Ungated => no UNI/Virchow2 blocker.

Loss = beta-NLL (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic Uncertainty Estimation"): the
per-sample Gaussian NLL is weighted by stop-gradient(var)**beta. beta=0 is plain NLL (the variance head
can collapse onto easy points and starve the mean head's gradient); beta~=0.5 keeps the heteroscedastic
signal while restoring a near-MSE gradient scale, which is what makes the dual head train stably.
"""

from __future__ import annotations

import torch
from torch import nn

UNGATED_BACKBONE = "vit_small_patch14_dinov2.lvd142m"  # ungated DINOv2-S; correct timm name (rollout 13)


def beta_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """beta-NLL: Gaussian NLL weighted by stop-gradient(var)**beta. Mean-reduced over all elements.

    beta=0 recovers the plain heteroscedastic Gaussian NLL; beta=1 recovers an MSE-scale gradient. The
    additive constant 0.5*log(2*pi) is dropped (it does not affect the gradient or model comparison).
    """
    var = logvar.exp()
    nll = 0.5 * ((target - mean) ** 2 / var + logvar)
    if beta > 0.0:
        nll = nll * var.detach() ** beta
    return nll.mean()


class DualHeadOracle(nn.Module):
    """Shared trunk -> (mean head, log-variance head). Optional image backbone in front of the trunk."""

    def __init__(
        self,
        n_genes: int,
        in_dim: int,
        backbone: nn.Module | None = None,
        trunk_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        min_logvar: float = -10.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        if not trunk_dims:
            raise ValueError("trunk_dims must have at least one layer")
        self.backbone = backbone
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        dims = [in_dim, *trunk_dims]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        hidden = trunk_dims[-1]
        self.mean_head = nn.Linear(hidden, n_genes)
        self.logvar_head = nn.Linear(hidden, n_genes)
        # start the variance head near unit variance so early training is well-conditioned
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Route (N,3,H,W) image batches through the backbone; pass (N,d) embeddings straight through."""
        if self.backbone is not None and x.dim() == 4:
            return self.backbone(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.features(x))
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(self.min_logvar, self.max_logvar)
        return mean, logvar

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns (mean, variance) with the module in eval mode."""
        was_training = self.training
        self.eval()
        mean, logvar = self.forward(x)
        if was_training:
            self.train()
        return mean, logvar.exp()


def make_backbone(
    name: str = UNGATED_BACKBONE, pretrained: bool = True, freeze: bool = False
) -> tuple[nn.Module, int]:
    """Build an ungated timm backbone for end-to-end mode. Returns (model, embed_dim).

    `dynamic_img_size=True` lets the DINOv2 ViT interpolate its position embeddings to the 224px patch we
    feed (it natively expects 518px). `freeze=True` caches it as a fixed feature extractor (then prefer the
    embedding-mode path, which skips the backbone entirely and is far cheaper).
    """
    import timm

    try:
        model = timm.create_model(
            name, pretrained=pretrained, num_classes=0, dynamic_img_size=True
        )
    except TypeError:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    embed_dim = int(model.num_features)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    return model, embed_dim
