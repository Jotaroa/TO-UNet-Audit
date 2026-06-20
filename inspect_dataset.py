"""
inspect_dataset.py
==================
Kiểm tra trực quan dữ liệu sinh ra từ Phase 1 (data/dataset.npz).

Vẽ một lưới các object, mỗi hàng = một object với 5 cột:
  [input ch0 = rho_init] [input ch1 = |grad| chuẩn hóa] [target = layout hội tụ]
  [|dc/drho| hội tụ (mốc audit)] [BC: gối=xanh, tải=đỏ]

Cũng in thống kê tổng quát (số object, volfrac, dải giá trị từng kênh).

Ví dụ:
  python inspect_dataset.py                       # 6 object đầu -> dataset_preview.png
  python inspect_dataset.py --n 12 --random       # 12 object ngẫu nhiên
  python inspect_dataset.py --index 0 5 10 99      # các object chỉ định
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--n", type=int, default=6, help="số object hiển thị")
    ap.add_argument("--random", action="store_true", help="bốc ngẫu nhiên")
    ap.add_argument("--index", type=int, nargs="+", default=None,
                    help="chỉ số object cụ thể (ghi đè --n/--random)")
    ap.add_argument("--out", default="dataset_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.data)
    n_total = int(z["n"])
    print(f"dataset: {a.data}")
    print(f"  số object (n)      : {n_total}")
    print(f"  X (train input)    : {z['X'].shape}  = (N, 2, 40, 40)")
    print(f"  Y (train target)   : {z['Y'].shape}  = (N, 1, 40, 40)")
    print(f"  volfrac            : mean {float(z['volfrac'].mean()):.3f}  "
          f"[{float(z['volfrac'].min()):.2f}, {float(z['volfrac'].max()):.2f}]")
    print(f"  x_init  range      : [{float(z['x_init'].min()):.3f}, {float(z['x_init'].max()):.3f}]")
    print(f"  x_final range      : [{float(z['x_final'].min()):.3f}, {float(z['x_final'].max()):.3f}]")

    # chọn object
    if a.index is not None:
        idx = [i for i in a.index if 0 <= i < n_total]
    elif a.random:
        rng = np.random.default_rng(a.seed)
        idx = sorted(rng.choice(n_total, size=min(a.n, n_total), replace=False).tolist())
    else:
        idx = list(range(min(a.n, n_total)))
    print(f"  hiển thị object    : {idx}")

    x_init = z["x_init"]; grad = z["grad_init"]; x_final = z["x_final"]
    sfin = z["sensitivity_final"]; sup = z["support_mask"]; ld = z["load_mask"]
    vf = z["volfrac"]

    cols = 5
    rows = len(idx)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    if rows == 1:
        axes = axes[None, :]
    titles = ["input ch0\nX_n (snapshot)", "input ch1\nδX = Xₙ−Xₙ₋₁",
              "target\nlayout hội tụ", "|∂c/∂ρ| hội tụ\n(mốc audit)",
              "BC\ngối=xanh, tải=đỏ"]

    def pnorm(im):  # percentile clip cho dễ nhìn trường sensitivity
        lo, hi = np.percentile(im, 1), np.percentile(im, 99)
        return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

    for r, i in enumerate(idx):
        axes[r, 0].imshow(x_init[i], cmap="gray_r", vmin=0, vmax=1)
        axes[r, 1].imshow(np.abs(grad[i]) / (np.abs(grad[i]).max() + 1e-9),
                          cmap="magma")
        axes[r, 2].imshow(x_final[i], cmap="gray_r", vmin=0, vmax=1)
        axes[r, 3].imshow(pnorm(np.abs(sfin[i])), cmap="magma")
        # BC: nền layout mờ + gối xanh + tải đỏ
        axes[r, 4].imshow(x_final[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.35)
        ys, xs = np.where(sup[i] > 0); axes[r, 4].scatter(xs, ys, s=14, c="blue", marker="s")
        yl, xl = np.where(ld[i] > 0);  axes[r, 4].scatter(xl, yl, s=24, c="red", marker="v")
        axes[r, 0].set_ylabel(f"obj {i}\nvf={vf[i]:.2f}", fontsize=8)
        for c in range(cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)

    plt.tight_layout()
    plt.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"-> đã lưu {a.out}")


if __name__ == "__main__":
    main()
