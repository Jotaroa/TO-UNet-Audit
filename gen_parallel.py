"""
PHASE 1a (parallel) -- generate many shards across CPU cores in ONE command,
optionally merging them into data/dataset.npz at the end.

Example
-------
    python gen_parallel.py --total 10000 --shards 10 --workers 10 --merge

This spawns `shards` worker processes (each a different seed) writing
data/shards/shard_seed*.npz, then (with --merge) concatenates them.
"""
import argparse, os, math
from multiprocessing import Pool
from attn_audit import dataset as ds


def _gen_one(task):
    seed, n, nelx, nely, max_iter, n_init, outdir, use_proj = task
    objs = ds.generate_objects(n_objects=n, nelx=nelx, nely=nely,
                               max_iter=max_iter, seed=seed,
                               n_init=n_init, verbose=False,
                               use_projection=use_proj)
    X, Y = ds.make_training_pairs(objs)
    path = os.path.join(outdir, f"shard_seed{seed}.npz")
    ds.save_dataset(path, objs, X, Y)
    return path, X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=10000, help="total objects")
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--nelx", type=int, default=40)
    ap.add_argument("--nely", type=int, default=40)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=1)
    ap.add_argument("--outdir", default="data/shards")
    ap.add_argument("--merge", action="store_true", help="merge -> data/dataset.npz")
    ap.add_argument("--no-projection", action="store_true",
                    help="disable Heaviside projection (legacy sensitivity filtering)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    per = math.ceil(a.total / a.shards)
    use_proj = not a.no_projection
    tasks = [(seed, per, a.nelx, a.nely, a.max_iter, a.n_init, a.outdir, use_proj)
             for seed in range(a.shards)]
    print(f"generating {a.shards} shards x {per} objects "
          f"on {a.workers} workers ({a.total} total target, "
          f"projection={'on' if use_proj else 'off'})...")

    with Pool(processes=a.workers) as pool:
        results = pool.map(_gen_one, tasks)

    paths = [p for p, _ in results]
    got = sum(nx for _, nx in results)
    print(f"done: {len(paths)} shards, {got} training pairs total")
    for p, nx in results:
        print(f"  {p}  ({nx} pairs)")

    if a.merge:
        out, total = ds.merge_datasets(paths, "data/dataset.npz")
        print(f"merged -> {out}  ({total} objects, "
              f"{os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
