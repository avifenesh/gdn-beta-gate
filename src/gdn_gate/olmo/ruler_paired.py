#!/usr/bin/env python3
"""RULER per arm vs stock, paired per item (C1 readout). Reads the vLLM battery sample dumps
(<root>/<arm>/samples_ruler<L>/<task>.jsonl, score under key str(L)) and reports, per length and task, the arm and stock
means, the paired difference, and a two-sided paired sign-flip permutation p (20k draws, fixed seed). Also the per-length
macro mean over tasks with an item-bootstrap 95% interval of the difference.
--base names the reference job (default stock; e.g. ctrl_s0 for the Qwen3.5 counting probe, vllm/vcount.py); the
per-row field "stock" holds that reference's mean. When both dumps carry lm-eval's doc_hash (vcount.py writes it),
items whose hashes differ are left out of the pairing and counted in the row's "hash_mismatch".
usage: ruler_paired.py ROOT OUT.json [--arms half,erase] [--lengths 4096,8192,16384] [--base stock]"""
import argparse
import json
import os

import numpy as np


def load(root, arm, L, task, hashes=None):
    p = f"{root}/{arm}/samples_ruler{L}/{task}.jsonl"
    if not os.path.exists(p):
        return None
    rows = list(map(json.loads, open(p)))
    if hashes is not None:
        hashes.update({r["doc_id"]: r["doc_hash"] for r in rows if r.get("doc_hash") is not None})
    return {r["doc_id"]: float(r[str(L)]) for r in rows}


def perm_p(d, rng, n=20000):
    d = np.asarray(d)
    if not d.any():
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--arms", default="half,erase")
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--base", default="stock", help="reference job directory under ROOT")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {"root": a.root, "rows": [], "macro": []}
    if a.base != "stock":
        out["base"] = a.base
    for L in map(int, a.lengths.split(",")):
        d0 = f"{a.root}/{a.base}/samples_ruler{L}"
        if not os.path.isdir(d0):
            continue
        tasks = sorted(f[:-6] for f in os.listdir(d0) if f.endswith(".jsonl"))
        for arm in a.arms.split(","):
            diffs = []
            for t in tasks:
                hs, hx = {}, {}
                s, x = load(a.root, a.base, L, t, hs), load(a.root, arm, L, t, hx)
                if x is None:
                    continue
                ids = sorted(set(s) & set(x))
                bad = [i for i in ids if i in hs and i in hx and hs[i] != hx[i]]
                if bad:
                    ids = [i for i in ids if i not in set(bad)]
                d = np.array([x[i] - s[i] for i in ids])
                diffs.append(d)
                row = {"L": L, "arm": arm, "task": t, "n": len(ids),
                       "stock": round(float(np.mean([s[i] for i in ids])), 4),
                       "arm_mean": round(float(np.mean([x[i] for i in ids])), 4),
                       "diff": round(float(d.mean()), 4), "p": round(perm_p(d, rng), 4),
                       "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
                if hs or hx:
                    row["hash_mismatch"] = len(bad)
                out["rows"].append(row)
            if len(diffs) == len(tasks) and len({d.size for d in diffs}) > 1:
                # the item bootstrap resamples one item index across tasks, so it needs equal n (dropped hashes or
                # mixed limits break that; this case raised before)
                out["macro"].append({"L": L, "arm": arm, "tasks": len(tasks), "skipped": "tasks differ in n",
                                     "n": [int(d.size) for d in diffs]})
            elif len(diffs) == len(tasks):
                D = np.stack(diffs)  # tasks x items (RULER tasks share n)
                boot = [D[:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(5000)]
                out["macro"].append({"L": L, "arm": arm, "tasks": len(tasks), "diff": round(float(D.mean()), 4),
                                     "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                              round(float(np.percentile(boot, 97.5)), 4)],
                                     "p": round(perm_p(D.mean(0), rng), 4)})
    json.dump(out, open(a.out, "w"), indent=1)
    for m in out["macro"]:
        print("macro", m)
    for r in out["rows"]:
        if r["p"] < 0.05:
            print("sig", r)


if __name__ == "__main__":
    main()
