"""
audit.py
========
Per-sample analysis + aggregation for the attention-vs-mechanics study, written
to be model-agnostic. Works on NN4TopOptUNet (2-channel input, logits output)
via injected `extract_fn` and the `net_input` field produced by dataset/adapter.
"""
from __future__ import annotations
import numpy as np
import torch

from .alignment import align_maps, Alignment
from .baselines import competing_targets, competition


def _input_tensor(sample):
    """Build the network input tensor. Uses the 2-channel `net_input` if present
    (NN4TopOptUNet), else a 1-channel `x_in`."""
    if "net_input" in sample:
        arr = np.asarray(sample["net_input"], np.float32)      # (2,H,W)
        return torch.from_numpy(arr)[None]                     # (1,2,H,W)
    arr = np.asarray(sample["x_in"], np.float32)
    return torch.from_numpy(arr)[None, None]                   # (1,1,H,W)


def _mean_iou(pred, target, thr=0.5):
    p = (pred > thr).float(); t = (target > thr).float()
    inter = (p * t).sum(); union = ((p + t) > 0).float().sum()
    return float(inter / (union + 1e-6))


@torch.no_grad()
def analyze_sample(model, sample, extract_fn, sens_state="out",
                   apply_sigmoid=True, use_input_gradient=True):
    """Run the model on one sample, extract attention, align each gate to |S|,
    and run the competing-target competition.

    extract_fn         : callable(model, x_tensor) -> list of (H,W) alpha maps.
    sens_state         : "out" (default) compares attention to the CONVERGED
                         sensitivity. Use "out" when the EARLY sensitivity is a
                         network input (NN4TopOptUNet channel 1), otherwise the
                         comparison is tautological. "in" reproduces the
                         input-echo baseline on purpose.
    apply_sigmoid      : True if the model returns logits (NN4TopOptUNet does).
    use_input_gradient : add the channel-1 |dc/drho| as a competing target and
                         as a second confound control.
    """
    rho = np.asarray(sample["x_in"], float)
    S = sample["S_in"] if sens_state == "in" else sample["S_out"]
    if sens_state == "out":
        rho = np.asarray(sample["x_out"], float)

    x_t = _input_tensor(sample)
    out = model(x_t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    pred = torch.sigmoid(out) if apply_sigmoid else out
    tgt = torch.from_numpy(np.asarray(sample["x_out"], np.float32))[None, None]
    iou = _mean_iou(pred, tgt)

    alphas = extract_fn(model, x_t)

    igrad = None
    if use_input_gradient and "net_input" in sample:
        ni = np.asarray(sample["net_input"], float)
        if ni.shape[0] >= 2:
            igrad = ni[1]                       # channel-1 = |dc/drho| input
    targets = competing_targets(S, rho, sample["bc"], input_gradient=igrad)

    per_gate = []
    for gi, a in enumerate(alphas):
        al: Alignment = align_maps(a, S, rho, input_grad=igrad)
        comp = competition(a, targets)
        per_gate.append({"gate": gi, "res": a.shape[0],
                         "align": al.as_dict(), "competition": comp})
    return {"iou": iou, "alphas": alphas, "per_gate": per_gate,
            "S": S, "rho": rho, "sample": sample}


def aggregate(analyses: list[dict]) -> dict:
    n_gates = len(analyses[0]["per_gate"])
    keys = list(analyses[0]["per_gate"][0]["align"].keys())
    comp_keys = list(analyses[0]["per_gate"][0]["competition"].keys())

    per_gate = []
    for gi in range(n_gates):
        amean = {k: np.nanmean([a["per_gate"][gi]["align"][k] for a in analyses])
                 for k in keys}
        astd = {k: np.nanstd([a["per_gate"][gi]["align"][k] for a in analyses])
                for k in keys}
        cmean = {k: np.nanmean([a["per_gate"][gi]["competition"][k]
                                for a in analyses]) for k in comp_keys}
        per_gate.append({"gate": gi,
                         "res": analyses[0]["per_gate"][gi]["res"],
                         "align_mean": amean, "align_std": astd,
                         "competition_mean": cmean})

    overall = {k: np.nanmean([a["per_gate"][gi]["align"][k]
                              for a in analyses for gi in range(n_gates)])
               for k in keys}
    overall_comp = {k: np.nanmean([a["per_gate"][gi]["competition"][k]
                                   for a in analyses for gi in range(n_gates)])
                    for k in comp_keys}
    iou = float(np.mean([a["iou"] for a in analyses]))
    return {"per_gate": per_gate, "overall": overall,
            "overall_competition": overall_comp, "mean_iou": iou,
            "n_samples": len(analyses)}
