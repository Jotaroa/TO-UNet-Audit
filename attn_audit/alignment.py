"""
alignment.py
============
Quantify how well an attention map alpha aligns with the mechanics field
S = dc/drho. This module is where the research question becomes numbers.

Design principle -- separate THREE questions that a naive correlation conflates:

  Q1  Does attention track |S| at all?                 -> spearman(alpha, |S|)
  Q2  Does attention track |S| BEYOND just tracking
      where material is (rho)?                          -> partial corr (alpha,|S| | rho)
  Q3  Inside the structure, does attention rank-order
      elements by their mechanical importance?         -> in-material spearman

Q2 and Q3 are the decisive tests. In topology optimization rho and |S| are
intrinsically correlated (the optimizer pushes material toward high-sensitivity
regions), so a strong Q1 correlation can be a pure artifact of both fields
co-varying with rho. Only Q2/Q3 isolate genuinely mechanical content.

All maps are brought to a common resolution (the attention map's), then
per-sample min-max normalized before comparison. Sensitivity is used as |S|.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from scipy.stats import spearmanr, rankdata
from skimage.metrics import structural_similarity as ssim
from skimage.measure import block_reduce
from skimage.transform import resize


# --------------------------------------------------------------------------- #
# resolution matching + normalization                                          #
# --------------------------------------------------------------------------- #
def downsample_to(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-average `field` (nely,nelx) down to `shape`.

    Average pooling (not subsampling) so a coarse cell reflects the mean
    mechanical content of the fine cells under the attention receptive field.
    """
    H, W = shape
    fH, fW = field.shape
    if (fH, fW) == (H, W):
        return field.astype(float)
    if fH % H == 0 and fW % W == 0:
        return block_reduce(field, (fH // H, fW // W), np.mean)
    return resize(field, shape, order=1, anti_aliasing=True, preserve_range=True)


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    rng = a.max() - a.min()
    return (a - a.min()) / (rng + 1e-12)


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #
def _partial_spearman(a: np.ndarray, b: np.ndarray, ctrl: np.ndarray) -> float:
    """Spearman partial correlation of a,b controlling for ctrl.

    Implemented as Pearson correlation of the residuals of rank(a) and rank(b)
    after linear regression on rank(ctrl). This is the standard rank-based
    partial correlation.
    """
    ra = rankdata(a); rb = rankdata(b); rc = rankdata(ctrl)
    ra = (ra - ra.mean()); rb = (rb - rb.mean()); rc = (rc - rc.mean())
    # residualize on rc
    denom = (rc @ rc) + 1e-12
    res_a = ra - (ra @ rc) / denom * rc
    res_b = rb - (rb @ rc) / denom * rc
    num = res_a @ res_b
    den = np.sqrt((res_a @ res_a) * (res_b @ res_b)) + 1e-12
    return float(num / den)


def _partial_spearman_multi(a: np.ndarray, b: np.ndarray,
                            ctrls: list[np.ndarray]) -> float:
    """Spearman partial correlation of a,b controlling for SEVERAL covariates
    simultaneously (rank-based, multiple linear regression on the ranks)."""
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    C = np.column_stack([rankdata(c) for c in ctrls]).astype(float)
    C = C - C.mean(0)
    CtC = C.T @ C + 1e-9 * np.eye(C.shape[1])
    res_a = ra - C @ np.linalg.solve(CtC, C.T @ ra)
    res_b = rb - C @ np.linalg.solve(CtC, C.T @ rb)
    den = np.sqrt((res_a @ res_a) * (res_b @ res_b)) + 1e-12
    return float((res_a @ res_b) / den)


def _topk_iou(a: np.ndarray, b: np.ndarray, k: float = 0.2) -> float:
    """IoU of the top-k fraction masks of a and b."""
    n = a.size
    ka = int(round(k * n))
    if ka == 0:
        return 0.0
    ia = set(np.argsort(a.ravel())[-ka:])
    ib = set(np.argsort(b.ravel())[-ka:])
    inter = len(ia & ib)
    union = len(ia | ib)
    return inter / (union + 1e-12)


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    """Discretized mutual information, normalized to [0,1] by min entropy."""
    ja = np.clip((a * bins).astype(int), 0, bins - 1)
    jb = np.clip((b * bins).astype(int), 0, bins - 1)
    pab = np.zeros((bins, bins))
    np.add.at(pab, (ja.ravel(), jb.ravel()), 1.0)
    pab /= pab.sum()
    pa = pab.sum(1); pb = pab.sum(0)
    nz = pab > 0
    mi = (pab[nz] * np.log(pab[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / (min(ha, hb) + 1e-12))


# --------------------------------------------------------------------------- #
# the alignment record                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Alignment:
    spearman: float          # Q1: raw rank correlation alpha vs |S|
    pearson: float
    ssim: float
    mutual_info: float
    topk_iou: float
    partial_spearman_rho: float    # Q2: controls for density
    partial_spearman_full: float   # Q2b: controls for density AND input gradient
    in_material_spearman: float    # Q3: restricted to rho > thresh
    material_frac: float           # fraction of cells in material phase
    mas: float                     # Mechanics Alignment Score (composite)

    def as_dict(self):
        return asdict(self)


def align_maps(alpha: np.ndarray, sensitivity: np.ndarray, density: np.ndarray,
               material_thresh: float = 0.5,
               input_grad: np.ndarray = None) -> Alignment:
    """Compare ONE attention map against the mechanics field.

    Parameters
    ----------
    alpha       : (Ha,Wa) attention coefficient map in [0,1].
    sensitivity : (nely,nelx) signed dc/drho field used as the REFERENCE
                  (will be |.|-ed). For NN4TopOptUNet use the CONVERGED-state
                  sensitivity, since the early-iteration sensitivity is a
                  network input and comparing to it would be tautological.
    density     : (nely,nelx) density rho.
    input_grad  : (nely,nelx) the |dc/drho| handed to the network as channel 1.
                  When given, an extra partial correlation controls for BOTH
                  density and this input gradient, isolating mechanical content
                  the network did NOT simply receive as input.
    """
    shape = alpha.shape
    S = np.abs(downsample_to(sensitivity, shape))
    rho = downsample_to(density, shape)

    a = _norm01(alpha).ravel()
    s = _norm01(S).ravel()
    r = _norm01(rho).ravel()

    sp = float(spearmanr(a, s).statistic)
    pe = float(np.corrcoef(a, s)[0, 1])
    ss = float(ssim(_norm01(alpha), _norm01(S), data_range=1.0))
    mi = _mutual_information(_norm01(alpha), _norm01(S))
    iou = _topk_iou(a, s, k=0.2)
    pcorr = _partial_spearman(a, s, r)

    if input_grad is not None:
        ig = _norm01(np.abs(downsample_to(input_grad, shape))).ravel()
        pcorr_full = _partial_spearman_multi(a, s, [r, ig])
    else:
        pcorr_full = pcorr

    mask = rho.ravel() > material_thresh
    mfrac = float(mask.mean())
    if mask.sum() >= 8:
        im_sp = float(spearmanr(a[mask], s[mask]).statistic)
    else:
        im_sp = float("nan")

    # Composite Mechanics Alignment Score: reward genuine mechanical content.
    # Weighted toward the confound-controlled signals (Q2b, Q3) over raw Q1.
    parts = []
    parts.append(0.20 * max(sp, 0.0))
    parts.append(0.50 * max(pcorr_full, 0.0))
    if not np.isnan(im_sp):
        parts.append(0.30 * max(im_sp, 0.0))
    mas = float(sum(parts))

    return Alignment(sp, pe, ss, mi, iou, pcorr, pcorr_full, im_sp, mfrac, mas)
