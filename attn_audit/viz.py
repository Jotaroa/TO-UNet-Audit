"""
viz.py
======
Figures for the attention-vs-mechanics study. All functions save a PNG and
return the path. Matplotlib only.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .alignment import downsample_to, _norm01


def _pnorm(a: np.ndarray, p: float = 99.0) -> np.ndarray:
    """Percentile-clipped [0,1] normalization so a few extreme cells (e.g. the
    load point in a sensitivity field) don't crush the rest of the dynamic
    range to black."""
    a = a.astype(float)
    hi = np.percentile(a, p)
    lo = a.min()
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def qualitative_panel(analysis: dict, path: str, title: str = "") -> str:
    """Row 1 = network I/O (the two input channels X_n and dX, then the target).
    Row 2 = the audit: the converged sensitivity (the reference, NOT a network
    input) next to each attention gate. Keeping the sensitivity out of the input
    row makes explicit that attention is compared to a field the network never
    receives."""
    s = analysis["sample"]
    alphas = analysis["alphas"]
    S = np.abs(analysis["S"])
    ng = len(alphas)

    X_n = np.asarray(s["x_in"])          # input channel 0
    dX = np.asarray(s["S_in"])           # input channel 1 = X_n - X_{n-1} (signed)
    x_out = np.asarray(s["x_out"])       # target

    ncols = max(3, ng + 1)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4))
    for ax in axes.ravel():
        ax.axis("off")

    # --- Row 1: network inputs and target ---
    axes[0, 0].imshow(X_n, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("input ch0:  Xₙ\n(density snapshot)")
    m = float(np.abs(dX).max() + 1e-9)
    axes[0, 1].imshow(dX, cmap="RdBu_r", vmin=-m, vmax=m)
    axes[0, 1].set_title("input ch1:  δX = Xₙ−Xₙ₋₁\n(last density update)")
    axes[0, 2].imshow(x_out, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 2].set_title("target:  converged ρ")

    # --- Row 2: audit reference + attention gates ---
    axes[1, 0].imshow(_pnorm(S), cmap="magma")
    axes[1, 0].set_title("audit ref:  |∂c/∂ρ|\n(NOT a network input)")
    for gi, a in enumerate(alphas):
        axes[1, gi + 1].imshow(a, cmap="viridis", vmin=0, vmax=1)
        axes[1, gi + 1].set_title(f"attention gate {gi}\n({a.shape[0]}×{a.shape[1]})")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def overlay_scatter(analysis: dict, gate: int, path: str) -> str:
    """Scatter of attention vs |S| for one gate (+ Spearman annotation)."""
    a = analysis["alphas"][gate]
    S = _norm01(np.abs(downsample_to(analysis["S"], a.shape))).ravel()
    av = _norm01(a).ravel()
    al = analysis["per_gate"][gate]["align"]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(S, av, s=8, alpha=0.4, edgecolors="none")
    ax.set_xlabel("|∂c/∂ρ|  (normalized)")
    ax.set_ylabel("attention α  (normalized)")
    ax.set_title(f"gate {gate}: ρ_s={al['spearman']:.2f}, "
                 f"partial(|ρ)={al['partial_spearman_rho']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def metric_bars(agg: dict, path: str) -> str:
    """Per-gate Q1 / Q2 / Q2b / Q3 correlations side by side."""
    gates = [g["gate"] for g in agg["per_gate"]]
    sp = [g["align_mean"]["spearman"] for g in agg["per_gate"]]
    pc = [g["align_mean"]["partial_spearman_rho"] for g in agg["per_gate"]]
    pf = [g["align_mean"]["partial_spearman_full"] for g in agg["per_gate"]]
    im = [g["align_mean"]["in_material_spearman"] for g in agg["per_gate"]]
    x = np.arange(len(gates)); w = 0.2

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - 1.5 * w, sp, w, label="Q1 raw  ρ_s(α,|S|)", color="#1f77b4")
    ax.bar(x - 0.5 * w, pc, w, label="Q2 partial  (·|ρ)", color="#ff7f0e")
    ax.bar(x + 0.5 * w, pf, w, label="Q2b partial  (·|ρ,grad)", color="#9467bd")
    ax.bar(x + 1.5 * w, im, w, label="Q3 in-material", color="#2ca02c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"gate {g}\n({agg['per_gate'][i]['res']}²)"
                                          for i, g in enumerate(gates)])
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Attention–mechanics alignment by gate")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def competition_bars(agg: dict, path: str) -> str:
    """What does attention align with best? (overall, averaged across gates)"""
    comp = agg["overall_competition"]
    order = ["sensitivity", "input_gradient", "density", "edges",
             "dist_to_bc", "random"]
    names = [k for k in order if k in comp]
    vals = [comp[k] for k in names]
    # sensitivity = the mechanical reference (red); input_gradient = the
    # "echo your own input" rival (orange); the rest are shape/control (grey).
    cmap = {"sensitivity": "#c0392b", "input_gradient": "#e67e22"}
    colors = [cmap.get(n, "#7f8c8d") for n in names]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman(α, target)")
    ax.set_title("Competition: which field does attention track?")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def randomization_curve(rand: dict, path: str) -> str:
    """Alignment decay under cascading weight randomization."""
    y = rand["alignment"]
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(x, y, "-o", ms=5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(rand["stages"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("module randomized (deep → shallow)")
    ax.set_ylabel("in-material ρ_s(α, |S|)  (learned mechanics)")
    ax.set_title("Sanity check: cascading weight randomization")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return path
