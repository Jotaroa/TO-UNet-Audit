"""
PHASE 3 -- audit a trained NN4TopOptUNet: do its attention gates attend to
MECHANICS (converged SIMP sensitivity) or just shape/the input gradient?
Independent of phases 1 and 2.

Requires (same folder): conv_model.py, data/dataset.npz, model.pt
Example:  python audit.py --max-audit 500
Outputs:  audit_out/results.json + figures
"""
import argparse, os, json
import numpy as np
import torch

from conv_model import NN4TopOptUNet
import attn_audit as aa
from attn_audit.attention import extract_attention_masks
from attn_audit.dataset import build_net_input, load_audit_objects
from attn_audit import viz


def _structure_score(xf):
    """Higher = more truss-like (thin distributed members); lower = a chunky
    blob. Uses perimeter/area of the converged structure."""
    solid = (np.asarray(xf) > 0.5)
    area = float(solid.sum())
    if area < 10:
        return -1.0
    perim = float(np.abs(np.diff(solid.astype(float), axis=0)).sum()
                  + np.abs(np.diff(solid.astype(float), axis=1)).sum())
    return perim / area


def sample_from_object(o, grad_mode="raw"):
    net = build_net_input(o["x_init"], o["grad_init"], grad_mode)
    return {"net_input": net, "x_in": o["x_init"], "x_out": o["x_final"],
            "S_in": o["grad_init"], "S_out": o["sensitivity_final"], "bc": o["bc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--weights", default="model.pt")
    ap.add_argument("--outdir", default="audit_out")
    ap.add_argument("--max-audit", type=int, default=500, help="objects to audit")
    ap.add_argument("--val-objs", default=None,
                    help="JSON of validation object indices "
                         "(default: <weights>.val_objs.json if it exists)")
    ap.add_argument("--all-objs", action="store_true",
                    help="ignore the val-objs file and audit the first --max-audit objects")
    ap.add_argument("--sample-index", type=int, default=0,
                    help="which audited sample to draw in fig_qualitative (0-based)")
    ap.add_argument("--auto-sample", action="store_true",
                    help="auto-pick a non-chunky (truss-like) sample for fig_qualitative")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    model = NN4TopOptUNet(in_channels=2, out_channels=1, base_filters=32)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()

    # Prefer the held-out validation objects saved by train.py, so we audit
    # data the model never trained on.
    val_objs = None
    vpath = a.val_objs or (a.weights + ".val_objs.json")
    if not a.all_objs and os.path.exists(vpath):
        val_objs = json.load(open(vpath))
        print(f"auditing on {min(len(val_objs), a.max_audit)} held-out val "
              f"objects from {vpath}")
    else:
        print(f"auditing on first {a.max_audit} objects "
              f"({'--all-objs set' if a.all_objs else 'no val-objs file found'})")

    objects = load_audit_objects(a.data, max_n=a.max_audit, indices=val_objs)
    samples = [sample_from_object(o) for o in objects]
    extract = lambda m, x: extract_attention_masks(m, x, name_endswith=".psi")

    # reference = converged sensitivity (early gradient is a network input)
    analyses = [aa.analyze_sample(model, s, extract, sens_state="out")
                for s in samples]
    agg = aa.aggregate(analyses)

    s0 = samples[0]
    x0 = torch.from_numpy(np.asarray(s0["net_input"], np.float32))[None]
    rand = aa.sanity.cascading_randomization(model, x0, s0["S_out"], extract,
                                             density=s0["x_out"], seed=0)

    # pick which sample to draw in the qualitative panel
    qi = a.sample_index
    if a.auto_sample:
        scores = [_structure_score(an["sample"]["x_out"]) for an in analyses]
        qi = int(np.argmax(scores))
    qi = max(0, min(qi, len(analyses) - 1))
    print(f"fig_qualitative uses audited sample #{qi}"
          + (" (auto-picked truss-like)" if a.auto_sample else ""))
    viz.qualitative_panel(analyses[qi], os.path.join(a.outdir, "fig_qualitative.png"),
                          title="Attention vs mechanics (NN4TopOptUNet)")
    viz.metric_bars(agg, os.path.join(a.outdir, "fig_metric_bars.png"))
    viz.competition_bars(agg, os.path.join(a.outdir, "fig_competition.png"))
    viz.randomization_curve(rand, os.path.join(a.outdir, "fig_randomization.png"))

    results = {"mean_iou": agg["mean_iou"], "n_samples": agg["n_samples"],
               "overall": agg["overall"],
               "overall_competition": agg["overall_competition"],
               "per_gate": [{"gate": g["gate"], "res": g["res"],
                             "align_mean": g["align_mean"],
                             "competition_mean": g["competition_mean"]}
                            for g in agg["per_gate"]],
               "randomization": {"stages": rand["stages"],
                                 "alignment": rand["alignment"].tolist()}}
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    o = agg["overall"]
    print(f"mean IoU {agg['mean_iou']:.3f}  | {agg['n_samples']} objects "
          f"| reference = converged sensitivity")
    print(f"Q1 raw {o['spearman']:+.3f}  Q2 |rho {o['partial_spearman_rho']:+.3f}  "
          f"Q2b |rho,grad {o['partial_spearman_full']:+.3f}  "
          f"Q3 in-mat {o['in_material_spearman']:+.3f}")
    print("competition:", {k: round(v, 3) for k, v in agg["overall_competition"].items()})
    print(f"-> {a.outdir}/results.json + figures")


if __name__ == "__main__":
    main()
