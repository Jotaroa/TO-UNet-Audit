"""
PHASE 2 -- train NN4TopOptUNet on data/dataset.npz using your combined loss
(BCE + ToleranceBandLoss + 0.3*LovaszHingeLoss). Independent of phases 1 and 3.

Requires (same folder): conv_model.py, loss.py, data/dataset.npz
Examples:
    python train.py --epochs 60 --batch 32                 # auto GPU if available
    python train.py --epochs 60 --batch 64 --device cuda   # force GPU
    python train.py --device cpu                           # force CPU
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from conv_model import NN4TopOptUNet
from loss import LovaszHingeLoss, ToleranceBandLoss


def resolve_device(name="auto"):
    """name: 'auto' | 'cuda' | 'cpu' (also accepts 'cuda:0', etc.)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available -> falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def combined_loss(logits, y, bce, tol, lov, vol_coeff=1.0):
    prob = torch.sigmoid(logits)
    l_bce = bce(prob, y)
    l_vol = tol(prob.mean(dim=(1, 2, 3)), y.mean(dim=(1, 2, 3)))
    l_lov = lov(logits.squeeze(1), y.squeeze(1))
    return l_bce + vol_coeff * l_vol + 0.3 * l_lov


def iou(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float(); t = (y > thr).float()
    i = (p * t).sum((1, 2, 3)); u = ((p + t) > 0).float().sum((1, 2, 3))
    return float((i / (u + 1e-6)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vol-coeff", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="'auto' (default), 'cuda', 'cpu', or e.g. 'cuda:0'")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (set >0 on Linux for speed)")
    a = ap.parse_args()

    device = resolve_device(a.device)
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(device)})"
             if device.type == "cuda" else ""))

    torch.manual_seed(a.seed)
    z = np.load(a.data)
    X = torch.from_numpy(z["X"]); Y = torch.from_numpy(z["Y"])
    print(f"dataset: X={tuple(X.shape)}  Y={tuple(Y.shape)}")

    full = TensorDataset(X, Y)
    n_val = max(1, int(len(full) * a.val_frac))
    tr, va = random_split(full, [len(full) - n_val, n_val],
                          generator=torch.Generator().manual_seed(a.seed))
    # Save the validation object indices so Phase 3 can audit EXACTLY the
    # held-out set (objects the model never trained on). With n_init=1 (one
    # training pair per object), these indices are also object indices.
    val_objs_path = a.out + ".val_objs.json"
    json.dump(sorted(int(i) for i in va.indices), open(val_objs_path, "w"))
    print(f"val objects: {len(va)}  -> {val_objs_path}")
    pin = (device.type == "cuda")
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=pin)
    vl = DataLoader(va, batch_size=a.batch,
                    num_workers=a.workers, pin_memory=pin)

    model = NN4TopOptUNet(in_channels=2, out_channels=1, base_filters=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCELoss(); tol = ToleranceBandLoss(epsilon=0.1); lov = LovaszHingeLoss()

    best = -1.0
    for ep in range(a.epochs):
        model.train()
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=pin); yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            combined_loss(model(xb), yb, bce, tol, lov, a.vol_coeff).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vi = float(np.mean([iou(model(xb.to(device)), yb.to(device))
                                for xb, yb in vl]))
        if vi > best:
            best = vi; torch.save(model.state_dict(), a.out)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  val IoU {vi:.3f}  (best {best:.3f})")
    print(f"done. best val IoU {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
