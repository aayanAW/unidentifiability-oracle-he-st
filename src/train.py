"""Training harness for the dual-head oracle: AMP, checkpointing, deep ensembles, torchrun/DDP-ready.

`fit_dual_head` is the single training loop used by BOTH the local frozen-embedding experiment
(experiments/trained_oracle_breast.py, spatial-block OOF on this Mac) and the cluster CLI
(`torchrun -m src.train ...`, see scripts/train_dualhead.sbatch). It degrades cleanly:

  - device: CUDA -> MPS -> CPU (auto), overridable.
  - AMP: real mixed precision only on CUDA (GradScaler/autocast); a no-op on MPS/CPU so the same code runs
    on this Mac for the smoke test and on H100s for the full run.
  - DDP: activated only when WORLD_SIZE>1 (torchrun sets it); single-process on the Mac. Rank 0 checkpoints.
  - deep ensemble: `fit_ensemble` trains N seeds -> the epistemic term + the independent-f' arm (audit C3).

No data formats are hard-coded: the CLI reads an .npz of {X, Y[, coords]} (cached DINOv2 embeddings or any
feature matrix), so swapping in UNI/Virchow2 or Xenium-5K embeddings is a file swap, not a code change.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .predictor import DualHeadOracle, beta_nll_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    beta: float = 0.5
    device: str | None = None  # None -> auto (cuda > mps > cpu)
    amp: bool = True  # honored only on CUDA
    grad_clip: float = 5.0
    num_workers: int = 0
    seed: int = 0


# --------------------------------------------------------------------------------------------------
# device + distributed
# --------------------------------------------------------------------------------------------------


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def maybe_init_distributed() -> tuple[bool, int, int]:
    """Init the process group iff torchrun launched us (WORLD_SIZE>1). Returns (distributed, local_rank, rank)."""
    if not is_distributed():
        return False, 0, 0
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_rank()


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------


def fit_dual_head(
    model: DualHeadOracle,
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    cfg: TrainConfig,
) -> tuple[DualHeadOracle, dict]:
    """Train `model` on X[train_idx]->Y[train_idx] with beta-NLL. Returns (model, history) on its device.

    X may be (N, d) cached embeddings or (N, 3, H, W) image patches (the module routes images through its
    backbone). val_idx (optional) is evaluated each epoch for early-stopping diagnostics.
    """
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = model.to(device)

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    Yt = torch.as_tensor(np.asarray(Y, dtype=np.float32))
    train_idx = np.asarray(train_idx)

    ds = TensorDataset(Xt[train_idx], Yt[train_idx])
    distributed = is_distributed()
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    train_module: nn.Module = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        train_module = DDP(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    opt = torch.optim.AdamW(
        train_module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_module.train()
        running, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda")
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                mean, logvar = train_module(xb)
                loss = beta_nll_loss(mean, logvar, yb, beta=cfg.beta)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(train_module.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            nb += 1
        history["train_loss"].append(running / max(nb, 1))
        history["val_loss"].append(
            _eval_loss(model, Xt, Yt, val_idx, device, cfg.beta)
            if val_idx is not None and len(val_idx)
            else float("nan")
        )
    return model, history


@torch.no_grad()
def _eval_loss(model, Xt, Yt, idx, device, beta) -> float:
    model.eval()
    idx = np.asarray(idx)
    mean, logvar = model(Xt[idx].to(device))
    return float(beta_nll_loss(mean, logvar, Yt[idx].to(device), beta=beta).item())


def fit_ensemble(
    build_model: Callable[[], DualHeadOracle],
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    cfg: TrainConfig,
    n_models: int = 5,
) -> list[tuple[DualHeadOracle, dict]]:
    """Train a deep ensemble (different seeds) -> epistemic spread + the independent-f' arm (audit C3)."""
    out = []
    for m in range(n_models):
        out.append(
            fit_dual_head(
                build_model(), X, Y, train_idx, val_idx, replace(cfg, seed=cfg.seed + m)
            )
        )
    return out


# --------------------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------------------


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    meta: dict | None = None,
    epoch: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": _unwrap(model).state_dict(),
        "epoch": int(epoch),
        "meta": meta or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(path, map_location=map_location)
    _unwrap(model).load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return {"epoch": state.get("epoch", 0), "meta": state.get("meta", {})}


# --------------------------------------------------------------------------------------------------
# CLI (cluster entrypoint + local smoke)
# --------------------------------------------------------------------------------------------------


def _smoke_data(n: int = 256, d: int = 32, g: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, g)).astype(np.float32)
    Y = (X @ W + 0.1 * rng.standard_normal((n, g))).astype(np.float32)
    return X, Y


def _load_yaml_overrides(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the dual-head oracle (frozen-embedding or end-to-end)."
    )
    ap.add_argument(
        "--data", type=str, default=None, help=".npz with arrays X (N,d) and Y (N,G)"
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional yaml of TrainConfig overrides",
    )
    ap.add_argument("--out", type=str, default="runs/dualhead")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run (CI/dev)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--trunk", type=int, nargs="+", default=[512, 256])
    args = ap.parse_args(argv)

    distributed, _local_rank, rank = maybe_init_distributed()

    if args.smoke:
        X, Y = _smoke_data()
    elif args.data:
        d = np.load(args.data)
        X, Y = d["X"], d["Y"]
    else:
        ap.error("provide --data <npz> or --smoke")

    ov = _load_yaml_overrides(args.config)
    cfg = TrainConfig(
        epochs=args.epochs or int(ov.get("epochs", 5 if args.smoke else 200)),
        lr=args.lr or float(ov.get("lr", 1e-3)),
        batch_size=args.batch_size or int(ov.get("batch_size", 128)),
        beta=args.beta if args.beta is not None else float(ov.get("beta", 0.5)),
        device=args.device,
    )

    n = len(X)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    out = Path(args.out)
    for m in range(args.ensemble):
        model = DualHeadOracle(
            n_genes=int(Y.shape[1]),
            in_dim=int(X.shape[1]),
            trunk_dims=tuple(args.trunk),
        )
        model, hist = fit_dual_head(
            model, X, Y, train_idx, val_idx, replace(cfg, seed=cfg.seed + m)
        )
        if rank == 0:
            ckpt = out / ("model.pt" if args.ensemble == 1 else f"model_{m}.pt")
            save_checkpoint(
                ckpt,
                model,
                meta={
                    "n_genes": int(Y.shape[1]),
                    "in_dim": int(X.shape[1]),
                    "trunk": list(args.trunk),
                    "final_val_loss": hist["val_loss"][-1],
                },
                epoch=cfg.epochs,
            )
            print(
                f"[train] member {m}: final val_loss={hist['val_loss'][-1]:.4f} -> {ckpt}"
            )

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
