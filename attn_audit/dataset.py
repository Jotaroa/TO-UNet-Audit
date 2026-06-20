"""
dataset.py
==========
Synthetic SIMP dataset for NN4TopOptUNet, generated with the pseudo-random
problem-sampling strategy of Sosnovik & Oseledets (nn4topopt):

  * N_x ~ Poisson(2)  fixed-x nodes,  N_y ~ Poisson(1)  fixed-y nodes,
    N_L ~ Poisson(1)  loaded nodes
  * nodes drawn on the grid with boundary nodes 100x more likely than inner ones
  * load values = -1 (vertical)
  * volume fraction ~ Normal(0.5, 0.1)
  * 100 iterations of standard SIMP per problem

One problem -> one training object:
    INPUT  X = [rho_init, filtered dc/drho at rho_init]   (2,H,W)
    TARGET Y = converged layout (binarized)               (1,H,W)

To scale to many thousands of objects we DO NOT keep the full 100-frame history;
we store only what training and the audit need (init frame + its update
gradient, the converged layout, the converged sensitivity, and the BC masks).
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np

from .simp import SIMPSolver, SIMPConfig, BoundaryConditions


# --------------------------------------------------------------------------- #
# 2-channel input assembly (shared by training and audit)                     #
# --------------------------------------------------------------------------- #
def build_net_input(rho, grad, grad_mode="raw"):
    """nn4topopt input: channel 0 = density snapshot X_n, channel 1 = last
    density update dX = X_n - X_{n-1} (signed). 'raw' keeps dX as-is (paper
    convention); 'signednorm'/'absnorm' are legacy options."""
    rho = np.asarray(rho, float)
    g = np.asarray(grad, float)
    if grad_mode == "raw":
        pass                                   # keep signed dX
    elif grad_mode == "signednorm":
        g = g / (np.abs(g).max() + 1e-12)
    elif grad_mode == "absnorm":
        g = np.abs(g); g = g / (g.max() + 1e-12)
    return np.stack([rho, g], axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# pseudo-random problem sampling (nn4topopt strategy)                          #
# --------------------------------------------------------------------------- #
def _boundary_prob(nelx, nely):
    """Node-sampling probabilities: boundary nodes 100x inner ones.
    Node id = ix*(nely+1)+iy, ix in [0,nelx], iy in [0,nely]."""
    NX, NY = nelx + 1, nely + 1
    w = np.ones((NX, NY))
    w[0, :] = w[-1, :] = 100.0
    w[:, 0] = w[:, -1] = 100.0
    w = w.ravel()
    return w / w.sum()


def sample_problem(nelx, nely, rng, cap=20):
    """Sample one random (BoundaryConditions, volfrac) per the strategy."""
    p = _boundary_prob(nelx, nely)
    nodes = np.arange((nelx + 1) * (nely + 1))

    Nx = min(max(int(rng.poisson(2)), 1), cap)   # >=1 to avoid free x-translation
    Ny = min(max(int(rng.poisson(1)), 2), cap)   # >=2 to pin y-translation+rotation
    NL = min(max(int(rng.poisson(1)), 1), cap)   # >=1 load

    fx = rng.choice(nodes, size=Nx, replace=False, p=p)
    fy = rng.choice(nodes, size=Ny, replace=False, p=p)
    ld = rng.choice(nodes, size=NL, replace=False, p=p)

    fixed = np.concatenate([2 * fx, 2 * fy + 1])
    load_dofs = np.array([d for d in (2 * ld + 1) if d not in set(fixed)], int)
    vf = float(np.clip(rng.normal(0.5, 0.1), 0.2, 0.8))

    sup = np.zeros((nely, nelx)); ldm = np.zeros((nely, nelx))
    def mark(mask, ids):
        for nid in ids:
            ix, iy = nid // (nely + 1), nid % (nely + 1)
            mask[min(iy, nely - 1), min(ix, nelx - 1)] = 1.0
    mark(sup, np.concatenate([fx, fy])); mark(ldm, ld)

    bc = BoundaryConditions("rand", fixed, load_dofs,
                            -np.ones(load_dofs.size), sup, ldm)
    return bc, vf


# --------------------------------------------------------------------------- #
# generation                                                                  #
# --------------------------------------------------------------------------- #
def _is_valid_object(x_init, x_final, compliance=None,
                     min_solid=0.10, max_solid=0.90,
                     max_intermediate=0.45,
                     min_input_target_diff=0.02):
    """Reject degenerate / non-converged SIMP solutions.

    Returns (ok: bool, reason: str). A solution is rejected if:
      - the converged layout is almost empty or almost full
        (solid fraction outside [min_solid, max_solid]);
      - too many cells stay grey / un-binarized (not converged);
      - input and target are essentially identical (trivial problem).

    Note: x_init is an nn4topopt snapshot X_n (a partly-formed structure), so we
    do NOT noise-check it — with projection it is legitimately near-binary. The
    genuinely-failed cases (speckle, non-convergence) are caught by the x_final
    checks below and by the singular-BC detection in the solver."""
    xf = np.asarray(x_final); xi = np.asarray(x_init)

    solid = float((xf > 0.5).mean())
    if solid < min_solid:
        return False, f"empty (solid={solid:.2f})"
    if solid > max_solid:
        return False, f"full (solid={solid:.2f})"

    intermediate = float(((xf > 0.2) & (xf < 0.8)).mean())
    if intermediate > max_intermediate:
        return False, f"not converged (grey={intermediate:.2f})"

    diff = float(np.abs(xi - xf).mean())
    if diff < min_input_target_diff:
        return False, f"trivial (|init-final|={diff:.3f})"

    if compliance is not None and not np.all(np.isfinite(compliance)):
        return False, "non-finite compliance"
    return True, "ok"


def generate_objects(n_objects=1000, nelx=40, nely=40, max_iter=100, seed=0,
                     n_init=1, keep_history=False, verbose=True,
                     quality_guard=True, use_projection=True, iter_lambda=6.0):
    """Generate `n_objects` training objects. Resamples on singular/failed FE
    and (if quality_guard) on degenerate/non-converged solutions."""
    rng = np.random.default_rng(seed)
    objs = []
    attempts = 0
    rejected = 0
    while len(objs) < n_objects and attempts < n_objects * 60:
        attempts += 1
        bc, vf = sample_problem(nelx, nely, rng)
        if bc.load_dofs.size == 0:
            continue
        cfg = SIMPConfig(nelx=nelx, nely=nely, volfrac=vf, penal=3.0,
                         rmin=1.5, max_iter=max_iter, tol=0.0,
                         use_projection=use_projection)   # crisp 0/1 designs
        solver = SIMPSolver(cfg, bc)
        try:
            res = solver.optimize(seed=int(rng.integers(1 << 30)))
        except Exception:
            continue
        hist, xf = res["history"], res["x_final"]
        if not (np.all(np.isfinite(hist)) and np.all(np.isfinite(xf))):
            continue

        # nn4topopt input: a snapshot X_n at an intermediate iteration n, plus
        # the last density update dX = X_n - X_{n-1}. The target is the final
        # converged structure. We sample n early (where the projection beta is
        # constant) so dX is the genuine optimization update, not a beta jump.
        T = hist.shape[0]
        n_max = max(1, min(T - 2, int(cfg.beta_iters) - 1))
        if n_max < 1:
            continue
        n_iter = 1 + int(rng.poisson(iter_lambda))
        n_iter = int(np.clip(n_iter, 1, n_max))
        X_n = hist[n_iter]
        dX = hist[n_iter] - hist[n_iter - 1]      # last density update (signed)

        if quality_guard:
            ok, _reason = _is_valid_object(X_n, xf,
                                           compliance=res.get("compliance"))
            if not ok:
                rejected += 1
                continue

        rec = {"x_init": X_n.astype(np.float32),          # channel 0 = X_n
               "grad_init": dX.astype(np.float32),        # channel 1 = X_n - X_{n-1}
               "x_final": xf.astype(np.float32),          # target = converged
               "sensitivity_final": res["sensitivity_final"].astype(np.float32),
               "support_mask": bc.support_mask.astype(np.float32),
               "load_mask": bc.load_mask.astype(np.float32),
               "volfrac": vf, "n_iter": n_iter}
        if n_init > 1:
            rec["extra_inits"] = [
                (hist[t].astype(np.float32),
                 (hist[t] - hist[t - 1]).astype(np.float32))
                for t in range(1, min(n_init, n_max + 1))]
        if keep_history:
            rec["history"] = hist.astype(np.float32)
        objs.append(rec)
        if verbose and len(objs) % max(1, n_objects // 10) == 0:
            print(f"      generated {len(objs)}/{n_objects} "
                  f"(accept {len(objs)/attempts:.0%}, rejected {rejected} low-quality)")
    if len(objs) < n_objects:
        print(f"      WARNING: only {len(objs)}/{n_objects} objects "
              f"(rejected {rejected} low-quality; increase attempts or relax guard)")
    return objs


# --------------------------------------------------------------------------- #
# training pairs                                                              #
# --------------------------------------------------------------------------- #
def make_training_pairs(objects, grad_mode="raw", binarize_target=True):
    Xs, Ys = [], []
    for o in objects:
        y = o["x_final"]
        if binarize_target:
            y = (y > 0.5).astype(np.float32)
        Xs.append(build_net_input(o["x_init"], o["grad_init"], grad_mode))
        Ys.append(y[None])
        for ri, gi in o.get("extra_inits", []):
            Xs.append(build_net_input(ri, gi, grad_mode)); Ys.append(y[None])
    return np.stack(Xs), np.stack(Ys)


# --------------------------------------------------------------------------- #
# persistence (compact: fixed-size per-object fields, no full history)         #
# --------------------------------------------------------------------------- #
_AUDIT_KEYS = ["x_init", "grad_init", "x_final", "sensitivity_final",
               "support_mask", "load_mask"]


def save_dataset(path, objects, X, Y):
    d = {"X": X, "Y": Y, "n": len(objects),
         "volfrac": np.array([o["volfrac"] for o in objects], np.float32)}
    for k in _AUDIT_KEYS:
        d[k] = np.stack([o[k] for o in objects]).astype(np.float32)
    np.savez_compressed(path, **d)


def load_audit_objects(path, max_n=None, indices=None):
    """Reload per-object fields needed for the audit (no solver required:
    grad_init and sensitivity_final were precomputed at generation time).

    indices : optional list of object indices to load (e.g. the held-out
              validation set saved by train.py). If given, only those objects
              are materialized (then capped by max_n). If None, the first
              max_n objects are loaded.

    Memory-safe: each compressed array is decompressed ONCE and only the
    selected rows are kept."""
    z = np.load(path)
    n = int(z["n"])
    if indices is not None:
        sel = [int(i) for i in indices if 0 <= int(i) < n]
    else:
        sel = list(range(n))
    if max_n is not None:
        sel = sel[:int(max_n)]

    arr = {}
    for k in _AUDIT_KEYS:                 # x_init, grad_init, x_final, sensitivity_final, masks
        full = z[k]                        # decompress this array exactly once
        arr[k] = full[sel].copy()          # keep only the selected rows
        del full
    z.close()

    out = []
    for j in range(len(sel)):
        bc = SimpleNamespace(name="rand",
                             support_mask=arr["support_mask"][j],
                             load_mask=arr["load_mask"][j])
        out.append({"x_init": arr["x_init"][j], "grad_init": arr["grad_init"][j],
                    "x_final": arr["x_final"][j],
                    "sensitivity_final": arr["sensitivity_final"][j], "bc": bc})
    return out


# --------------------------------------------------------------------------- #
# sharding: merge many dataset_seed*.npz shards into one dataset.npz           #
# --------------------------------------------------------------------------- #
_CONCAT_KEYS = ["X", "Y", "volfrac"] + _AUDIT_KEYS


def merge_datasets(paths, out):
    """Concatenate several shard .npz files (each saved by save_dataset) into a
    single dataset .npz. Returns (out_path, total_objects)."""
    buffers = {k: [] for k in _CONCAT_KEYS}
    total = 0
    for p in paths:
        z = np.load(p)
        for k in _CONCAT_KEYS:
            buffers[k].append(z[k])
        total += int(z["n"])
    merged = {k: np.concatenate(buffers[k], axis=0) for k in _CONCAT_KEYS}
    merged["n"] = int(merged["x_init"].shape[0])
    np.savez_compressed(out, **merged)
    return out, total
