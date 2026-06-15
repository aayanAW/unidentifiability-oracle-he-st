"""Render the paper figures from the saved result arrays (cluster-free; Phase-6 design-system plots).

Reads experiments/results/{trained_oracle_breast.npz, conformal_breast.npz} (written by the respective
experiments) and writes PNGs to experiments/results/figures/. Code computes every number; this script only
draws them (CLAUDE.md: code computes, Claude never invents numbers).

Run:  python3 experiments/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGS = RESULTS / "figures"


def _selective_risk_figure() -> str | None:
    f = RESULTS / "trained_oracle_breast.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for key, label, style in (
        ("curve_oracle_ddh", "oracle (defer true-worst)", "k--"),
        ("curve_rf", "RF raw-variance U", "C0-"),
        ("curve_ddh", "trained dual-head U", "C3-"),
        ("curve_rawvar_ddh", "dual-head raw variance", "C1:"),
    ):
        if key in d.files:
            c = d[key]
            ax.plot(c[:, 0], c[:, 1], style, label=label, lw=2)
    ax.set_xlabel("coverage (retained gene fraction)")
    ax.set_ylabel("selective risk (mean OOF error on retained)")
    ax.set_title(
        "Selective-risk-coverage — breast 300µm (frozen DINOv2-S, exploratory)"
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIGS / "fig_selective_risk_coverage.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def _conformal_figure() -> str | None:
    f = RESULTS / "conformal_breast.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    target = 1.0 - float(d["alpha"])
    cn, cm = np.sort(d["cov_naive"]), np.sort(d["cov_mond"])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = np.arange(len(cn))
    ax.bar(x - 0.2, cn, 0.4, label="naive split-conformal", color="C0")
    ax.bar(x + 0.2, cm, 0.4, label="spatial-Mondrian", color="C2")
    ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target:.2f}")
    ax.set_xlabel("spatial block (sorted by coverage)")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(min(cn.min(), cm.min()) - 0.03, 1.0)
    ax.set_title(
        f"H1 spatial coverage — breast (naive Moran's I={float(d['morans_i']):.2f}, p={float(d['morans_p']):.3f})"
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = FIGS / "fig_conformal_coverage.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def _selective_conformal_figure() -> str | None:
    f = RESULTS / "selective_conformal_breast.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    t = d["table"]  # [retained_fraction, coverage, mean_width]
    target = 1.0 - float(d["alpha"])
    fig, ax1 = plt.subplots(figsize=(5.2, 4.0))
    ax1.plot(t[:, 0], t[:, 1], "C0-o", lw=2, label="coverage")
    ax1.axhline(target, color="C0", ls=":", lw=1)
    ax1.set_xlabel("retained gene fraction (abstain on highest-U first)")
    ax1.set_ylabel("marginal coverage", color="C0")
    ax1.set_ylim(target - 0.05, 1.0)
    ax2 = ax1.twinx()
    ax2.plot(t[:, 0], t[:, 2], "C3-s", lw=2, label="interval width")
    ax2.set_ylabel("mean interval width", color="C3")
    ax1.set_title("Coverage-guaranteed selective prediction — breast (exploratory)")
    fig.tight_layout()
    out = FIGS / "fig_selective_conformal.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = [
        p
        for p in (
            _selective_risk_figure(),
            _conformal_figure(),
            _selective_conformal_figure(),
        )
        if p
    ]
    if not made:
        print("[figures] no result .npz found -- run the experiments first.")
        return 1
    for p in made:
        print(f"[figures] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
