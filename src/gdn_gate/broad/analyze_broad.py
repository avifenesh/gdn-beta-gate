#!/usr/bin/env python3
"""Explore / confirm analysis of step 6 battery results.

Input: --root holding battery JSONs named {arm}_s{seed}_step{N}[_strip].json (battery.py --out).
Unit of replication is the seed: for a contrast X - Y at one bench and checkpoint, d_s is the mean item score of X
minus that of Y at seed s, over one split half (for a strip contrast X and Y are the same weights). Seeds pair arms
because an arm pair at one seed saw the same examples in the same order. Test: one-sample t on {d_s} (two-sided),
with an exact sign-flip permutation p reported beside it.

explore (half A): every (bench, checkpoint, contrast) with at least --min_seeds seeds; Benjamini-Hochberg at
  q < 0.10; a hit also needs every seed's d_s to share the sign of the mean.
confirm (half B): at most 3 hypotheses from --hyp (JSON list of {bench, step, contrast}); Holm at 0.05.
Lower is better for nll (it is a loss); every other bench is higher-is-better.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

CONTRASTS = {
    "wide-ctrl": ("wide", "ctrl"), "wide1-ctrl": ("wide1", "ctrl"), "wide-wide1": ("wide", "wide1"),
    "wide-wide_strip": ("wide", "wide_strip"), "wide1-wide1_strip": ("wide1", "wide1_strip"),
}
NAME = re.compile(r"^(wide1|wide|ctrl)_s(\d+)_step(\d+)(_strip)?\.json$")


def load(root):
    res = {}  # (variant, seed, step) -> bench -> {"items": {...}, "split": {...}}
    for p in sorted(Path(root).glob("*.json")):
        m = NAME.match(p.name)
        if not m:
            continue
        arm, seed, step, strip = m.group(1), int(m.group(2)), int(m.group(3)), bool(m.group(4))
        j = json.loads(p.read_text())
        assert j["load"].get("strip", False) == strip, p
        res[(arm + ("_strip" if strip else ""), seed, step)] = j["bench"]
    return res


def half_mean(b, half):
    v = [s for k, s in b["items"].items() if b["split"][k] == half]
    return float(np.mean(v)) if v else float("nan"), len(v)


def signflip_p(d):
    d = np.asarray(d)
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([f >= obs - 1e-15 for f in flips]))


def tests(res, half, min_seeds):
    rows = []
    steps = sorted({k[2] for k in res})
    benches = sorted({b for v in res.values() for b in v})
    for step, bench, (cname, (x, y)) in itertools.product(steps, benches, CONTRASTS.items()):
        seeds = sorted({k[1] for k in res if k[2] == step and k[0] == x} & {k[1] for k in res if k[2] == step and k[0] == y})
        seeds = [s for s in seeds if bench in res[(x, s, step)] and bench in res[(y, s, step)]]
        if len(seeds) < min_seeds:
            continue
        d, n_items = [], 0
        for s in seeds:
            mx, n_items = half_mean(res[(x, s, step)][bench], half)
            my, _ = half_mean(res[(y, s, step)][bench], half)
            d.append(mx - my)
        d = np.array(d)
        t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
        rows.append({"step": step, "bench": bench, "contrast": cname, "n_seeds": len(d), "n_items_half": n_items,
                     "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                     "per_seed": [round(float(v), 5) for v in d], "p_t": float(t.pvalue) if t is not None else 1.0,
                     "p_signflip": signflip_p(d), "same_sign": bool(np.all(np.sign(d) == np.sign(d.mean())) and d.mean() != 0)})
    return rows


def bh(ps, q):
    ps = np.asarray(ps)
    m = len(ps)
    order = np.argsort(ps)
    thr = q * np.arange(1, m + 1) / m
    passed = ps[order] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    adj = np.minimum.accumulate((ps[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.minimum(adj, 1.0)
    return out, qv


def holm(ps, alpha):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if ps[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["explore", "confirm"], required=True)
    ap.add_argument("--hyp", default="", help="confirm: JSON file, list of {bench, step, contrast}, at most 3")
    ap.add_argument("--min_seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = load(a.root)
    if a.mode == "explore":
        rows = tests(res, "A", a.min_seeds)
        if rows:
            sig, qv = bh([r["p_t"] for r in rows], 0.10)
            for r, s, q in zip(rows, sig, qv):
                r["q_bh"] = float(q)
                r["hit"] = bool(s and r["same_sign"])
        out = {"mode": "explore", "half": "A", "n_tests": len(rows), "hits": [r for r in rows if r.get("hit")],
               "tests": sorted(rows, key=lambda r: r["p_t"])}
    else:
        hyp = json.loads(Path(a.hyp).read_text())
        assert 1 <= len(hyp) <= 3, "confirm at most 3 hypotheses"
        allrows = tests(res, "B", a.min_seeds)
        rows = [next(r for r in allrows if (r["bench"], r["step"], r["contrast"]) == (h["bench"], h["step"], h["contrast"]))
                for h in hyp]
        rej = holm([r["p_t"] for r in rows], 0.05)
        for r, h, z in zip(rows, hyp, rej):
            r["confirmed"] = bool(z and (("sign" not in h) or np.sign(r["mean_diff"]) == h["sign"]))
        out = {"mode": "confirm", "half": "B", "hypotheses": hyp, "results": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
