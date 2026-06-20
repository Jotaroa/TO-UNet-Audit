"""
PHASE 1a -- generate ONE dataset shard (run many in parallel with different seeds).

Each shard is a self-contained .npz (same format as the merged dataset):
    X,Y (training pairs) + per-object audit fields.

Examples
--------
    python gen_shard.py --seed 0 --n 2500 --out data/shards/shard_0.npz
    python gen_shard.py --seed 1 --n 2500 --out data/shards/shard_1.npz
    ...launch as many as you have CPU cores, each with a DIFFERENT --seed.
"""
import argparse, os
from attn_audit import dataset as ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="unique per shard")
    ap.add_argument("--n", type=int, default=2500, help="objects in this shard")
    ap.add_argument("--nelx", type=int, default=40)
    ap.add_argument("--nely", type=int, default=40)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=1, help=">1 augments per object")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-projection", action="store_true",
                    help="disable Heaviside projection (use legacy sensitivity filtering)")
    a = ap.parse_args()

    out = a.out or f"data/shards/shard_seed{a.seed}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"[shard seed={a.seed}] generating {a.n} problems "
          f"({a.nelx}x{a.nely}, {a.max_iter} iters, "
          f"projection={'off' if a.no_projection else 'on'})...")
    objs = ds.generate_objects(n_objects=a.n, nelx=a.nelx, nely=a.nely,
                               max_iter=a.max_iter, seed=a.seed,
                               n_init=a.n_init, verbose=True,
                               use_projection=not a.no_projection)
    X, Y = ds.make_training_pairs(objs)
    ds.save_dataset(out, objs, X, Y)
    print(f"[shard seed={a.seed}] wrote {out}  "
          f"({os.path.getsize(out)/1e6:.1f} MB, X={X.shape})")


if __name__ == "__main__":
    main()
