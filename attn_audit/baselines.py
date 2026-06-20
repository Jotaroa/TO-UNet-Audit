"""
baselines.py
============
Alternative explanations for what an attention map "is". The central claim of
the study -- attention attends to MECHANICS, not an artifact -- only holds if
attention aligns with the sensitivity field MORE than with trivial,
non-mechanical fields that any segmentation network might latch onto:

  density        rho        -- "attention just lights up where material is"
  edges          |grad rho| -- "attention is a boundary detector"
  dist_to_bc     geometry   -- "attention is a fixed geometric prior near
                               loads/supports, independent of the field"
  random         shuffled   -- null control; correlation should vanish

`competing_targets` builds these fields. `competition` then measures how
strongly a given attention map correlates (Spearman) with EACH candidate target
on a common grid. If `sensitivity` wins, the mechanical reading survives; if
`edges` or `density` win, the skeptical reading does.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, sobel
from scipy.stats import spearmanr

from .alignment import downsample_to, _norm01


def competing_targets(sensitivity: np.ndarray, density: np.ndarray,
                      bc, rng: np.random.Generator | None = None,
                      input_gradient: np.ndarray | None = None
                      ) -> dict[str, np.ndarray]:
    """Build the set of candidate fields attention might be tracking.

    All returned at the native element resolution; the comparison routine
    downsamples each to the attention resolution.
    """
    rng = rng or np.random.default_rng(0)
    gx = sobel(density, axis=0); gy = sobel(density, axis=1)
    edges = np.hypot(gx, gy)

    # distance-to-BC: small near loads/supports, large far away -> invert so
    # "high = near BC" matches the high-attention convention.
    marker = np.zeros_like(density)
    if getattr(bc, "support_mask", None) is not None:
        marker += bc.support_mask
    if getattr(bc, "load_mask", None) is not None:
        marker += bc.load_mask
    if marker.max() == 0:
        marker[density.shape[0] // 2, 0] = 1.0
    dist = distance_transform_edt(marker == 0)
    near_bc = dist.max() - dist

    targets = {
        "sensitivity": np.abs(sensitivity),
        "density": density,
        "edges": edges,
        "dist_to_bc": near_bc,
        "random": rng.standard_normal(density.shape),
    }
    if input_gradient is not None:
        # |dc/drho| handed to the network as channel 1: "attention just echoes
        # its own gradient input" -- the key trivial explanation to beat when
        # the SIMP gradient is itself a network input.
        targets["input_gradient"] = np.abs(input_gradient)
    return targets


def competition(alpha: np.ndarray, targets: dict[str, np.ndarray]
                ) -> dict[str, float]:
    """Spearman(alpha, target) for every candidate, on alpha's grid."""
    a = _norm01(alpha).ravel()
    out = {}
    for name, fld in targets.items():
        t = _norm01(downsample_to(fld, alpha.shape)).ravel()
        out[name] = float(spearmanr(a, t).statistic)
    return out


def shuffle_control(alpha: np.ndarray, sensitivity: np.ndarray,
                    n: int = 200, rng: np.random.Generator | None = None
                    ) -> tuple[float, float]:
    """Permutation null for spearman(alpha, |S|).

    Returns (observed, p_value) where p is the fraction of spatially-shuffled
    attention maps whose |correlation| meets or exceeds the observed one.
    """
    rng = rng or np.random.default_rng(0)
    S = _norm01(np.abs(downsample_to(sensitivity, alpha.shape))).ravel()
    a = _norm01(alpha).ravel()
    obs = abs(float(spearmanr(a, S).statistic))
    count = 0
    for _ in range(n):
        perm = rng.permutation(a)
        if abs(float(spearmanr(perm, S).statistic)) >= obs:
            count += 1
    return obs, (count + 1) / (n + 1)
