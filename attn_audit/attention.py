"""
attention.py
============
Capture attention-gate coefficient maps alpha from a forward pass.

Two ways in:

  1. `extract_from_agresunet(model, x)` -- for the reference AGResUNet whose
     AttentionGate modules already cache `last_alpha`. Zero setup.

  2. `AttentionExtractor(model, module_filter)` -- generic forward hooks for
     YOUR OWN trained network. Point `module_filter` at your attention modules
     (by type or by name substring) and we grab whatever single-channel map
     they emit. This is how you plug the IoU 93-99% model into the study.

In both cases the output is a list of alpha images, one per gate, each a numpy
array of shape (H_s, W_s) at that gate's native skip resolution, values in
[0,1] (a batch element is selected; default index 0).
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import torch
import torch.nn as nn

from .ag_resunet import AGResUNet, AttentionGate


def _to_alpha_img(t: torch.Tensor, batch_idx: int = 0) -> np.ndarray:
    """Reduce a (B,C,H,W) attention tensor to a single (H,W) map in [0,1].

    If the gate emits multi-channel attention, average over channels -- the
    interpretable quantity is "how much spatial location (i,j) is attended to".
    """
    t = t.detach().float().cpu()
    if t.dim() == 4:
        t = t[batch_idx]            # (C,H,W)
    if t.dim() == 3:
        t = t.mean(0)               # (H,W)
    a = t.numpy()
    # normalize defensively into [0,1] if a gate emitted pre-sigmoid logits
    if a.min() < 0 or a.max() > 1:
        rng = a.max() - a.min()
        a = (a - a.min()) / (rng + 1e-12)
    return a


@torch.no_grad()
def extract_from_agresunet(model: AGResUNet, x: torch.Tensor,
                           batch_idx: int = 0) -> list[np.ndarray]:
    """Run the reference model, return alpha maps shallow->deep (skip order)."""
    model.eval()
    _ = model(x)
    # model.gates is in decoder order (deep skip first). Reorder to
    # shallowest-skip-first so index 0 == highest resolution.
    alphas = [g.last_alpha for g in model.gates]
    alphas = list(reversed(alphas))
    return [_to_alpha_img(a, batch_idx) for a in alphas]


class AttentionExtractor:
    """Generic forward-hook extractor for arbitrary attention modules.

    Example (your own model):
        ext = AttentionExtractor(my_model,
                                 module_filter=lambda n, m: "att" in n.lower())
        with ext:
            my_model(x)
        alphas = ext.alphas()        # list of (H,W) maps, registration order
    """

    def __init__(self, model: nn.Module,
                 module_filter: Callable[[str, nn.Module], bool] | None = None,
                 output_selector: Callable[[object], torch.Tensor] | None = None):
        self.model = model
        self.output_selector = output_selector or (lambda out: out)
        if module_filter is None:
            module_filter = lambda n, m: isinstance(m, AttentionGate)
        self._targets = [(n, m) for n, m in model.named_modules()
                         if module_filter(n, m)]
        if not self._targets:
            raise ValueError("module_filter matched no modules. Pass a filter "
                             "that selects your attention gates by name/type.")
        self._handles = []
        self._captured: dict[str, torch.Tensor] = {}

    def _make_hook(self, name):
        def hook(module, inp, out):
            # prefer an explicit cached alpha if the module exposes one
            cached = getattr(module, "last_alpha", None)
            t = cached if cached is not None else self.output_selector(out)
            self._captured[name] = t
        return hook

    def __enter__(self):
        self._captured.clear()
        for n, m in self._targets:
            self._handles.append(m.register_forward_hook(self._make_hook(n)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def names(self) -> list[str]:
        return [n for n, _ in self._targets]

    def alphas(self, batch_idx: int = 0) -> list[np.ndarray]:
        return [_to_alpha_img(self._captured[n], batch_idx)
                for n, _ in self._targets if n in self._captured]


# --------------------------------------------------------------------------- #
# Convenience: extract spatial-attention masks from an arbitrary model by      #
# hooking submodules whose qualified name ends with a given suffix (e.g. the   #
# ".psi" sigmoid mask inside each AttentionGate of NN4TopOptUNet).             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_attention_masks(model, x, name_endswith=".psi",
                            reverse_to_hi_res=True, batch_idx=0):
    """Return alpha maps (list of (H,W) arrays in [0,1]) by hooking every
    module whose name ends with `name_endswith`. By default reorders so index 0
    is the highest-resolution gate."""
    ext = AttentionExtractor(
        model, module_filter=lambda n, m: n.endswith(name_endswith))
    with ext:
        model(x)
    alphas = ext.alphas(batch_idx)
    if reverse_to_hi_res and len(alphas) >= 2 \
            and alphas[0].shape[0] < alphas[-1].shape[0]:
        alphas = list(reversed(alphas))
    return alphas
