"""
sanity.py
=========
Cascading weight-randomization sanity check (Adebayo et al., 2018) -- GENERIC
version that works on any model (incl. NN4TopOptUNet), not just the demo net.

The skeptical hypothesis to defeat: the attention-vs-mechanics alignment comes
from the architecture/input, not from anything the network LEARNED. If so,
scrambling the trained weights would leave the alignment intact. We randomize
parameter-bearing top-level blocks from the OUTPUT side toward the INPUT side,
recomputing the in-material Spearman(alpha,|S|) after each step; a genuine,
learning-dependent alignment should decay toward chance.

`extract_fn(model, x) -> list of (H,W) alpha maps` is injected so this module
stays model-agnostic (pass attention.extract_attention_masks for your model).
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from .alignment import downsample_to, _norm01


def _reinit_recursive(block: nn.Module, rng_seed: int):
    torch.manual_seed(rng_seed)
    for m in block.modules():
        if hasattr(m, "reset_parameters") and list(m.parameters(recurse=False)):
            m.reset_parameters()


def default_block_order(model) -> list[tuple[str, nn.Module]]:
    """Output->input order over parameter-bearing top-level children.
    named_children() is in definition order (input->output); we reverse it so
    randomization proceeds deepest(output)-first."""
    children = [(n, m) for n, m in model.named_children()
                if list(m.parameters(recurse=True))]
    return list(reversed(children))


def _mean_alignment(alphas, sensitivity, density=None,
                    material_thresh: float = 0.5) -> float:
    vals = []
    S_full = np.abs(sensitivity)
    for a in alphas:
        S = downsample_to(S_full, a.shape)
        av = _norm01(a).ravel()
        sv = _norm01(S).ravel()
        if density is not None:
            rho = downsample_to(density, a.shape).ravel()
            mask = rho > material_thresh
            if mask.sum() >= 8:
                vals.append(float(spearmanr(av[mask], sv[mask]).statistic))
                continue
        vals.append(float(spearmanr(av, sv).statistic))
    return float(np.nanmean(vals))


@torch.no_grad()
def cascading_randomization(model, x: torch.Tensor, sensitivity: np.ndarray,
                            extract_fn, density: np.ndarray | None = None,
                            seed: int = 0, block_order=None) -> dict:
    """Randomize blocks output->input, tracking alignment decay.

    extract_fn  : callable(model, x) -> list of (H,W) alpha maps.
    block_order : optional list of (name, module); default = output->input over
                  parameter-bearing top-level children.
    """
    model = copy.deepcopy(model).eval()
    a0 = extract_fn(model, x)
    stages = ["trained"]
    align = [_mean_alignment(a0, sensitivity, density)]

    ordered = block_order or default_block_order(model)
    for i, (name, mod) in enumerate(ordered):
        _reinit_recursive(mod, seed + i)
        a = extract_fn(model, x)
        stages.append(name)
        align.append(_mean_alignment(a, sensitivity, density))

    return {"stages": stages, "alignment": np.array(align)}
