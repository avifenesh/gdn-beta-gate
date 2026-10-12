#!/usr/bin/env python3
"""Paired comparison for the wide1 control on the damaging main retrofit recipe (Qwen3.5-4B-Base).

Question: is the forgetting protection of the wide arm (HumanEval kept at 0.21-0.31 while ctrl seeds fall to
0.09-0.27) caused by the negative range or by the extra gate parameters? wide1 has the same gate parameters, init
and learning rate as wide with the beta ceiling at 1, so:
  wide1 - ctrl  > 0 and wide - wide1 ~ 0   -> the gate parameters carry the protection
  wide1 - ctrl ~ 0  and wide - wide1 > 0   -> the negative range carries it

Inputs are rescore_4b.py receipts (one JSON per checkpoint, item-level). Names:
  main_<ctrl|wide|wide1>_s<seed>[_<gb10|rtx5090|box2>][_strip]
(box2 = the second rented box, cloud 8x RTX PRO 6000, rental 1.) A name without a hardware tag is a banked run; its training hardware comes from its run_dir (the rented 8x RTX PRO
6000 box for /box/, GB10 for <work>/runs). Contrasts are paired by seed (same task and replay stream; wide and
wide1 also share the LoRA draw):
  wide1 - ctrl, wide1 - wide, wide - ctrl (reference), each gated arm - its strip twin,
  and same-stack checks: a rerun <arm>_s<k>_<hw> minus the banked <arm>_s<k> (what the hardware alone moves).
Per seed: accuracy difference and exact McNemar p over items. Pooled: mean of the per-seed differences with
(a) a 95% item bootstrap (10,000 draws, seed 20261010; ignores seed variance) and (b) a seed-level 95% t interval
(n - 1 df; the honest one at 3 seeds), plus a pooled McNemar over all item x seed pairs.

Step-0 identity: at step 0 every arm is the base model (LoRA B = 0, gate W = 0) on the seed's first task and replay
batch, so a new run's step-0 task and replay losses must equal the banked box run's of the same seed up to kernel
noise (bound 1e-3 relative; GB10 vs box measured 8e-5 in the probe4b stage). This proves the data order, base
revision, replay file and recipe are the banked ones. Recipe identity: the new run's train.py args equal the banked
twin's except model/replay/out paths, arm and --no_grad_ckpt.

The rule output ("reading") is mechanical; the verdict is the owner's.
Usage: compare.py --new_dir D [--banked_dir D ...] [--box_runs D] [--new_runs D] --out O.json
       compare.py --identity NAME [--identity NAME ...] [--box_runs D] [--new_runs D] [--out O.json]
                  (step-0 and recipe identity of trained runs only, no receipts needed; exit 1 unless all pass)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

NAME = re.compile(r"^main_(?P<arm>ctrl|wide1|wide)_s(?P<seed>\d+)(?:_(?P<hw>gb10|rtx5090|box2))?(?P<strip>_strip)?$")
ITEM_PARTS = (("he", "pass1"), ("cloze", "acc"), ("letter", "acc"))
STEP0_REL = 1e-3
PATH_ARGS = {"model", "replay", "out", "arm", "no_grad_ckpt", "init_from"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def parse(name, rec):
    m = NAME.match(name)
    if not m:
        return None
    hw = m["hw"]
    if hw is None:
        rd = rec.get("run_dir", "")
        hw = "box" if "/box/" in rd else ("gb10" if "/gdn-gate/runs/" in rd else "unknown")
    return {"arm": m["arm"], "seed": int(m["seed"]), "hw": hw, "strip": bool(m["strip"]), "banked": m["hw"] is None}


def load(dirs):
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            n = os.path.basename(p)[:-5]
            try:
                r = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(r, dict) or r.get("smoke") or "name" not in r:
                continue
            k = parse(n, r)
            if k is None:
                continue
            if n in out:
                raise SystemExit(f"duplicate receipt {n} in {d} and {out[n]['_path']}")
            r["_key"], r["_path"] = k, p
            out[n] = r
    return out


def mcnemar(a, b):
    """Exact two-sided McNemar p from paired 0/1 vectors."""
    b10 = int(((a == 1) & (b == 0)).sum())
    b01 = int(((a == 0) & (b == 1)).sum())
    n = b10 + b01
    if n == 0:
        return 1.0, b10, b01
    k = min(b10, b01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), b10, b01


def items(r, part):
    it = r.get(part, {}).get("items")
    return {k: int(v) for k, v in it.items()} if it else None


def pair_vec(ra, rb, part):
    ia, ib = items(ra, part), items(rb, part)
    if ia is None or ib is None:
        return None
    keys = sorted(set(ia) & set(ib))
    if len(keys) != len(ia) or len(keys) != len(ib):
        raise SystemExit(f"{part}: item sets differ between {ra['name']} and {rb['name']} ({len(ia)} vs {len(ib)})")
    return np.array([ia[k] for k in keys]), np.array([ib[k] for k in keys])


def boot_mean_of_seed_means(diffs, n=10_000, seed=20261010):
    # resampling items with replacement only changes how often each distinct value appears, so draw those counts
    # (multinomial): the same distribution as index resampling, O(draws) instead of O(draws x items)
    g = np.random.default_rng(seed)
    acc = np.zeros(n)
    for d in diffs:
        vals, cnt = np.unique(d, return_counts=True)
        acc += g.multinomial(len(d), cnt / cnt.sum(), size=n) @ vals / len(d)
    acc /= len(diffs)
    return [float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))]


def seed_t(ds):
    if len(ds) < 2:
        return None
    m = float(np.mean(ds))
    s = float(np.std(ds, ddof=1))
    h = T975.get(len(ds) - 1, 1.96) * s / math.sqrt(len(ds))
    return [m - h, m + h]


def contrast(R, pairs):
    """pairs: list of (name_a, name_b), one per seed. Returns per-part per-seed and pooled stats for a - b."""
    res = {}
    for part, key in ITEM_PARTS:
        per, diffs, A, B = [], [], [], []
        for na, nb in pairs:
            v = pair_vec(R[na], R[nb], part)
            if v is None:
                continue
            a, b = v
            p, b10, b01 = mcnemar(a, b)
            d = (a - b).astype(float)
            per.append({"a": na, "b": nb, "diff": float(d.mean()), "a_acc": float(a.mean()), "b_acc": float(b.mean()),
                        "a_only": b10, "b_only": b01, "mcnemar_p": p})
            diffs.append(d)
            A.append(a)
            B.append(b)
        if not per:
            continue
        ds = [x["diff"] for x in per]
        p_all, b10, b01 = mcnemar(np.concatenate(A), np.concatenate(B))
        res[part] = {"per_seed": per, "mean": float(np.mean(ds)), "all_seeds_positive": all(x > 0 for x in ds),
                     "all_seeds_negative": all(x < 0 for x in ds), "boot95_items": boot_mean_of_seed_means(diffs),
                     "t95_seeds": seed_t(ds), "pooled_mcnemar_p": p_all, "pooled_a_only": b10, "pooled_b_only": b01,
                     "n_seeds": len(per)}
    for k in ("wikitext", "clean"):
        per = []
        for na, nb in pairs:
            xa, xb = R[na].get("nll", {}).get(k), R[nb].get("nll", {}).get(k)
            if xa and xb:
                per.append(xa["mean"] - xb["mean"])
        if per:
            res[f"nll_{k}"] = {"per_seed": per, "mean": float(np.mean(per)), "t95_seeds": seed_t(per)}
    return res


def table_row(r):
    o = {**r["_key"], "run_dir": r.get("run_dir"), "scored_on": r.get("gpu")}
    for part, key in ITEM_PARTS:
        if part in r:
            o[part] = r[part][key]
            if part == "cloze":
                o["cloze_stride"] = r[part].get("stride")
    for k, v in r.get("nll", {}).items():
        o[f"nll_{k}"] = v["mean"]
    if "tier1" in r:
        s = r["tier1"]["summary"]
        o["tier1@256"] = sum(s[f"{t}@256"] for t in ("parity", "swap", "codeswap")) / 3
    if "check_gate" in r:
        o["check_gate"] = r["check_gate"]
    return o


def step0(path):
    for f in ("train_log.json",):
        p = os.path.join(path, f)
        if os.path.exists(p):
            log = json.load(open(p))
            if log and log[0].get("step") == 0:
                return log[0]
    return None


def run_args(path):
    for f in ("result.json", "result_sweep.json"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            return json.load(open(p))["meta"]["args"]
    return None


def identity_one(n, k, rd, box_runs):
    """Step-0 loss and recipe identity of one new training run (dir rd) against the banked box run of its seed."""
    twin = next((os.path.join(box_runs, f"main_{a}_s{k['seed']}") for a in ("ctrl", "wide")
                 if os.path.isdir(os.path.join(box_runs, f"main_{a}_s{k['seed']}"))), None)
    if twin is None:
        return {"pass": None, "why": f"no banked box run for seed {k['seed']}"}
    s_new, s_box = step0(rd), step0(twin)
    a_new, a_box = run_args(rd), run_args(twin)
    if s_new is None or s_box is None or a_new is None or a_box is None:
        return {"pass": False, "why": f"missing train_log.json or result.json in {rd} or {twin}"}
    rel = {f: abs(s_new[f] - s_box[f]) / abs(s_box[f]) for f in ("task_loss", "replay_loss")}
    diff = {f: (a_new.get(f), a_box.get(f)) for f in set(a_new) | set(a_box)
            if f not in PATH_ARGS and a_new.get(f) != a_box.get(f)}
    if a_new.get("arm") != k["arm"] or a_new.get("seed") != k["seed"]:
        diff["name_vs_args"] = (n, a_new.get("arm"), a_new.get("seed"))
    return {"twin": twin, "step0_rel": rel, "recipe_diff": diff,
            "pass": all(v <= STEP0_REL for v in rel.values()) and not diff}


def identity_checks(R, new_runs, box_runs):
    """Step-0 loss and recipe identity of each new training run against the banked box run of the same seed."""
    out = {}
    for n, r in R.items():
        k = r["_key"]
        if k["banked"] or k["strip"]:
            continue
        rd = os.path.join(new_runs, n) if new_runs else r.get("run_dir", "")
        out[n] = identity_one(n, k, rd, box_runs)
    return out


def identity_only(names, new_runs, box_runs, out_path):
    """--identity: the step-0 check of trained runs before any receipt exists (the box scheduler runs it per run)."""
    res = {}
    for n in names:
        m = NAME.match(n)
        if not m or m["hw"] is None or m["strip"]:
            raise SystemExit(f"--identity takes trained new-run names main_<arm>_s<seed>_<hw>, got {n}")
        k = {"arm": m["arm"], "seed": int(m["seed"]), "hw": m["hw"], "strip": False, "banked": False}
        res[n] = {"run_dir": os.path.join(new_runs, n), **identity_one(n, k, os.path.join(new_runs, n), box_runs)}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=1)
    for n, v in res.items():
        print(f"identity {n}: pass {v.get('pass')} step0_rel {v.get('step0_rel')} recipe_diff {v.get('recipe_diff')}"
              f" twin {v.get('twin')} {v.get('why', '')}")
    return 0 if all(v.get("pass") for v in res.values()) else 1


def reading(c_w1_ctrl, c_wide_w1):
    """Mechanical rule on HumanEval (seed-paired): an effect counts when every seed has the same sign and the pooled
    McNemar p < 0.05."""
    def eff(c):
        h = (c or {}).get("he")
        if not h or h["n_seeds"] < 3:
            return None
        return h["all_seeds_positive"] and h["pooled_mcnemar_p"] < 0.05
    a, b = eff(c_w1_ctrl), eff(c_wide_w1)
    if a is None or b is None:
        return "incomplete (needs 3 seeds of wide1, wide and ctrl)"
    if a and not b:
        return "gate parameters: wide1 protects like wide (wide1 > ctrl, wide ~ wide1)"
    if b and not a:
        return "negative range: wide1 does not protect (wide1 ~ ctrl, wide > wide1)"
    if a and b:
        return "both: wide1 protects part, the range adds more"
    return "unresolved: neither contrast is consistent across seeds at p < 0.05"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_dir", help="receipts of the new runs (results/w1main/rescore4b)")
    ap.add_argument("--identity", action="append", default=[], help="identity check only, of these trained runs")
    ap.add_argument("--banked_dir", action="append", default=[], help="banked receipts (default: results/rescore/rescore4b_rtx5090 + rescore4b_gb10)")
    ap.add_argument("--box_runs", default=os.path.join(os.environ.get("WG_RUNS", "runs"), "retrofit/box"))
    ap.add_argument("--new_runs", default=os.path.join(os.environ.get("WG_RUNS", "runs"), "w1main"))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.identity:
        raise SystemExit(identity_only(a.identity, a.new_runs, a.box_runs, a.out))
    if not a.new_dir or not a.out:
        ap.error("--new_dir and --out are required (unless --identity)")
    wt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    banked = a.banked_dir or [os.path.join(wt, "results/rescore/rescore4b_rtx5090"),
                              os.path.join(wt, "results/rescore/rescore4b_gb10")]
    R = load(banked + [a.new_dir])
    gated = {n: r for n, r in R.items() if not r["_key"]["strip"]}

    def find(arm, seed, new):
        """the new run of (arm, seed) on any new hardware, or the banked one"""
        c = [n for n, r in gated.items() if r["_key"]["arm"] == arm and r["_key"]["seed"] == seed
             and r["_key"]["banked"] != new]
        return sorted(c)[0] if c else None

    seeds = sorted({r["_key"]["seed"] for r in gated.values() if r["_key"]["arm"] == "wide1"})
    pairs = {"wide1 - ctrl": [], "wide1 - wide": [], "wide - ctrl (banked)": []}
    for s in seeds:
        w1, c, w = find("wide1", s, True), find("ctrl", s, False), find("wide", s, False)
        if w1 and c:
            pairs["wide1 - ctrl"].append((w1, c))
        if w1 and w:
            pairs["wide1 - wide"].append((w1, w))
    # the same contrasts against reruns on the wide1 run's own hardware, where they exist (no stack difference)
    pairs["wide1 - ctrl (same hw)"], pairs["wide1 - wide (same hw)"] = [], []
    for n, r in gated.items():
        k = r["_key"]
        if k["arm"] == "wide1" and not k["banked"]:
            for arm in ("ctrl", "wide"):
                t = f"main_{arm}_s{k['seed']}_{k['hw']}"
                if t in gated:
                    pairs[f"wide1 - {arm} (same hw)"].append((n, t))
    for s in sorted({r["_key"]["seed"] for r in gated.values()}):
        c, w = find("ctrl", s, False), find("wide", s, False)
        if c and w and R[c]["_key"]["hw"] == R[w]["_key"]["hw"]:
            pairs["wide - ctrl (banked)"].append((w, c))
    strip_pairs = [(n, n + "_strip") for n in gated if n + "_strip" in R]
    same_stack = []
    for n, r in gated.items():
        k = r["_key"]
        if not k["banked"]:
            b = f"main_{k['arm']}_s{k['seed']}"
            if b in R:
                same_stack.append((n, b))
    contrasts = {name: contrast(R, p) for name, p in pairs.items() if p}
    contrasts["gated - strip twin"] = {f"{x} - {y}": contrast(R, [(x, y)]) for x, y in strip_pairs}
    contrasts["same-stack rerun - banked"] = {f"{x} - {y}": contrast(R, [(x, y)]) for x, y in same_stack}

    def wide_minus(cs):
        return contrasts.get(cs, {}).get("he", {}).get("mean")
    share = None
    d_w1, d_w = wide_minus("wide1 - ctrl"), None
    if pairs["wide1 - ctrl"]:  # the wide - ctrl gap on the same seeds as wide1 - ctrl
        same = [(find("wide", int(NAME.match(c)["seed"]), False), c) for _, c in pairs["wide1 - ctrl"]]
        if all(w for w, _ in same):
            d_w = contrast(R, same).get("he", {}).get("mean")
    if d_w1 is not None and d_w:
        share = d_w1 / d_w
    arms = {}
    for arm in ("ctrl", "wide", "wide1"):
        for strip in (False, True):
            xs = [r for r in R.values() if r["_key"]["arm"] == arm and r["_key"]["strip"] == strip]
            if xs:
                key = arm + ("_strip" if strip else "")
                arms[key] = {p: {"mean": float(np.mean([r[p][k] for r in xs if p in r])),
                                 "n": sum(1 for r in xs if p in r),
                                 "by_run": {r["name"]: r[p][k] for r in xs if p in r}}
                             for p, k in ITEM_PARTS}
    out = {"table": {n: table_row(r) for n, r in sorted(R.items())},
           "arm_means": arms, "contrasts": contrasts,
           "he_protection_share_wide1": share,
           "identity": identity_checks(R, a.new_runs, a.box_runs),
           "reading": reading(contrasts.get("wide1 - ctrl"), {"he": _flip(contrasts.get("wide1 - wide", {}).get("he"))}),
           "inputs": {"banked_dirs": banked, "new_dir": a.new_dir, "box_runs": a.box_runs, "new_runs": a.new_runs},
           "note": "seed-paired; pooled = mean of per-seed differences; boot95 over items (no seed variance), "
                   "t95 over seeds (n-1 df). Rule output is mechanical; the verdict is the owner's."}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    cols = ("he", "cloze", "letter", "nll_wikitext", "nll_clean", "tier1@256")
    print("name".ljust(28), "hw".ljust(5), *[c.rjust(12) for c in cols])
    for n, o in out["table"].items():
        print(n.ljust(28), o["hw"].ljust(5), *[(f"{o[c]:.4f}" if isinstance(o.get(c), float) else "-").rjust(12) for c in cols])
    for name in ("wide1 - ctrl", "wide1 - wide", "wide - ctrl (banked)"):
        h = contrasts.get(name, {}).get("he")
        if h:
            print(f"{name}: HE per seed {[round(x['diff'], 3) for x in h['per_seed']]} mean {h['mean']:+.3f} "
                  f"t95 {h['t95_seeds']} pooled McNemar p {h['pooled_mcnemar_p']:.3g}")
    print("identity:", json.dumps({n: v.get("pass") for n, v in out["identity"].items()}))
    print("reading:", out["reading"], "| HE protection share of wide1:", share)


def _flip(h):
    """wide - wide1 from a wide1 - wide contrast (sign flipped) for the rule."""
    if not h:
        return None
    ds = [-x["diff"] for x in h["per_seed"]]
    return {**h, "all_seeds_positive": all(x > 0 for x in ds), "n_seeds": len(ds)}


if __name__ == "__main__":
    main()
