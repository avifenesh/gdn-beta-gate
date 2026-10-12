#!/usr/bin/env python3
"""Summarise lesion_nll.py: per-domain NLL per arm and paired arm-minus-stock deltas with chunk bootstrap CIs.

Bootstrap: 95%, 10,000 draws, seed 20261006, resampling chunks (paired: the same chunks for both arms, stock
restricted to the arm's chunk prefix). Deltas are in nats per token, token-weighted within a domain; "all" weights
the four domains equally. Noise: the stock CI half-width per domain, and the stock first-half vs second-half gap.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
arms = {}
for line in open(f"{a.run}/arms.jsonl"):
    r = json.loads(line)
    arms[r["name"]] = r
assert "stock" in arms
rng = np.random.default_rng(20261006)
B = 10_000


def arr(rec, dom, n):
    s = np.array(rec["domains"][dom]["sum"])[:n]  # [n, buckets]
    c = np.array(rec["domains"][dom]["count"], dtype=float)
    return s, c


doms = list(arms["stock"]["domains"])
out = {"noise": {}, "arms": {}}
for d in doms:
    s, c = arr(arms["stock"], d, arms["stock"]["n_chunks"])
    per = s.sum(1) / c.sum()
    idx = rng.integers(0, len(per), (B, len(per)))
    bs = per[idx].mean(1)
    h = len(per) // 2
    out["noise"][d] = {"stock_nll": per.mean(), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       "half_gap": float(per[:h].mean() - per[h:].mean()), "n_chunks": len(per)}
for name, rec in arms.items():
    if name == "stock":
        continue
    n = rec["n_chunks"]
    row = {"desc": rec["desc"], "n_chunks": n, "tok_per_s": rec["tok_per_s"], "beta_gt1_frac": rec["beta_gt1_frac"]}
    deltas_all = []
    for d in doms:
        s1, c = arr(rec, d, n)
        s0, _ = arr(arms["stock"], d, n)
        diff = (s1 - s0)  # [n, nb]
        tot = diff.sum(1) / c.sum()
        idx = rng.integers(0, n, (B, n))
        bs = tot[idx].mean(1)
        deltas_all.append(bs)
        row[d] = {"delta": float(tot.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "by_bucket": [float(diff[:, j].sum() / (n * c[j])) for j in range(len(c))]}
    allb = np.mean(deltas_all, 0)
    row["all"] = {"delta": float(np.mean([row[d]["delta"] for d in doms])),
                  "ci": [float(np.percentile(allb, 2.5)), float(np.percentile(allb, 97.5))]}
    out["arms"][name] = row
json.dump(out, open(a.out, "w"), indent=1)
print("noise", json.dumps({d: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()} for d, x in out["noise"].items()}))
for name, row in out["arms"].items():
    print(f"{name:10s} all {row['all']['delta']:+.5f} [{row['all']['ci'][0]:+.5f},{row['all']['ci'][1]:+.5f}] " +
          " ".join(f"{d} {row[d]['delta']:+.4f}" for d in doms) + f" b>1 {row['beta_gt1_frac']:.4f}")
