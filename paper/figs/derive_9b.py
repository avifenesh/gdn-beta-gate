"""Collect the Qwen3.5-9B numbers of the paper (Section 4.4 and the compute statement). CPU only.

Reads the 9B readouts and the frozen AIME 2025 analysis receipt, plus the 9B training logs, and writes
paper/data/r9b.json. Every block names its source files.

1. GSM8K validation slice (500 items x 2 samples): per arm and seed the readout summary (area, accuracy at 8,192,
   length, median, p90, unfinished share); Gate(0,1) - LoRA per seed and pooled over seeds, with the paired item
   bootstrap of the decision rule (items resampled, seeds paired by index, 10,000 draws, numpy default_rng(20261009)),
   a hierarchical bootstrap and a seed-level t-interval (as paper/figs/stats_extra.py).
2. AIME 2025 (30 items x 8 samples, read once): per arm the values recomputed from the readout rows and checked
   against the analysis receipt (the script stops if they differ), and the hierarchical and t intervals of the two
   pre-registered contrasts and of accuracy at 8,192.
3. Training: forced (cut at the 512 cap) share and reasoning length over the last 20 and last 10 steps of each run,
   the per-step forced share for the figure, the one resume, and the trainable parameter counts.
In this repository the 9B readouts, the AIME 2025 receipts and the 9B run logs must be present under results/; the
launcher log behind wall/wide1_s0_job_limit_hours and the plan record behind the wall-clock spans are lab files that are
not released, so a full rerun needs WG_JOBLOG and the plan file. Run: uv run --with numpy python3 paper/figs/derive_9b.py
"""
import gzip
import json
import math
import os
import re
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
RUNS = RES / "run_logs/ml7_9b/grpo"  # the small training logs and meta files the paper reads, copied from the run dirs
OUT = ROOT / "paper/data/r9b.json"
JOBLOG = Path(os.environ.get("WG_JOBLOG", "arm_wide1.log"))  # launcher log of wide1_s0, not released (lab only)
CAPS = [256, 512, 1024, 2048, 4096, 8192]
DRAWS, RULE_SEED, HIER_SEED = 10000, 20261009, 20261012
T975_2 = 4.302652729911275  # t quantile, 2 degrees of freedom
SEEDS = (0, 1, 2)
ARMS = ("ctrl", "wide1")


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def norm_auc(accs):
    x = [math.log2(c) for c in CAPS]
    return sum((x[j + 1] - x[j]) * (accs[j] + accs[j + 1]) / 2 for j in range(len(x) - 1)) / (x[-1] - x[0])


def val_path(arm, seed=None):
    if arm == "untouched":
        return RES / "ml7_9b/step9_curve_ans1280_Qwen3.5-9B_gb10.json"
    return RES / f"ml7_9b/step9_pilot_{arm}_s{seed}_cap512_r9b_gb10_step150_ans1280_Qwen3.5-9B_gb10.json"


def aime_path(arm, seed=None):
    tag = "untouched" if arm == "untouched" else f"{arm}_{seed}"
    return RES / f"ml7_9b/aime2025/readout_aime2025_{tag}.json"


_cache = {}


def load(p):
    """Released readouts are gzipped (results/**/readout_*.json.gz): read FILE or FILE.gz."""
    if p not in _cache:
        _cache[p] = json.loads(gzip.open(str(p) + ".gz", "rt").read() if not p.exists() else p.read_text())
    return _cache[p]


def per_item(rows, metric):
    by = {}
    for r in rows:
        if metric == "area":
            v = norm_auc([r["caps"][str(c)][0] for c in CAPS])
        elif metric == "acc8k":
            v = r["caps"]["8192"][0]
        else:
            v = min(r["think_len"], 8192)
        by.setdefault(r["id"], []).append(v)
    return {k: sum(v) / len(v) for k, v in by.items()}


def contrast(pathf, metric, a="wide1", b="ctrl"):
    """Seed-paired a - b: per-seed differences, the rule's item interval, a hierarchical and a t-interval."""
    ids, A, B = None, [], []
    for s in SEEDS:
        pa, pb = per_item(load(pathf(a, s))["rows"], metric), per_item(load(pathf(b, s))["rows"], metric)
        if ids is None:
            ids = sorted(pa)
        assert sorted(pa) == ids == sorted(pb)
        A.append([pa[i] for i in ids])
        B.append([pb[i] for i in ids])
    D = np.array(A) - np.array(B)  # seeds x items
    k, n = D.shape
    per_seed = D.mean(axis=1)
    mean = float(per_seed.mean())
    rng = np.random.default_rng(RULE_SEED)
    idx = rng.integers(0, n, size=(DRAWS, n))
    item = np.quantile(D.mean(axis=0)[idx].mean(axis=1), [0.025, 0.975])
    rng = np.random.default_rng(HIER_SEED)
    m = np.empty(DRAWS)
    for j in range(DRAWS):
        m[j] = D[np.ix_(rng.integers(0, k, size=k), rng.integers(0, n, size=n))].mean()
    hier = np.quantile(m, [0.025, 0.975])
    half = T975_2 * float(per_seed.std(ddof=1)) / math.sqrt(k)
    signs = bool((per_seed > 0).all() or (per_seed < 0).all())
    return {"contrast": f"{a} - {b}", "metric": metric, "n_items": n, "seed_mean_diff": mean,
            "per_seed_diff": [float(x) for x in per_seed], "item_ci95": [float(x) for x in item],
            "hier_ci95": [float(x) for x in hier], "t_ci95": [mean - half, mean + half],
            "signs_agree": signs, "passes_rule": bool(signs and (item[0] > 0 or item[1] < 0))}


def val_block():
    out = {"source": [rel(val_path("untouched"))] + [rel(val_path(a, s)) for a in ARMS for s in SEEDS], "arms": {}}
    keys = ("area", "acc8k", "length", "median", "p90", "unfinished")

    def summ(p):
        d = load(p)
        assert d["env"]["gpu"] == "NVIDIA GB10" and d["args"]["samples"] == 2 and d["args"]["answer_max"] == 1280
        S = d["summary"]
        n = S["natural"]
        return dict(zip(keys, (S["norm_auc_log2"], S["per_cap"]["8192"]["acc"], n["reasoning_mean_at_max"],
                               n["median"], n["p90"], n["cut_off_share_at_max"]))), S["n_items"], len(d["rows"])

    u, n_items, n_rows = summ(val_path("untouched"))
    out["n_items"], out["n_rows"] = n_items, n_rows
    out["arms"]["untouched"] = u
    for a in ARMS:
        per = [summ(val_path(a, s))[0] for s in SEEDS]
        out["arms"][a] = {k: {"per_seed": [p[k] for p in per], "seed_mean": st.mean(p[k] for p in per),
                              "seed_sd": st.stdev(p[k] for p in per)} for k in keys}
    L = contrast(val_path, "length")
    L["pct_shorter_per_seed"] = [1 - out["arms"]["wide1"]["length"]["per_seed"][i] / out["arms"]["ctrl"]["length"]["per_seed"][i]
                                 for i in range(3)]
    L["pct_shorter_seed_mean"] = 1 - out["arms"]["wide1"]["length"]["seed_mean"] / out["arms"]["ctrl"]["length"]["seed_mean"]
    out["gate_minus_lora"] = {"length": L, "acc8k": contrast(val_path, "acc8k"), "area": contrast(val_path, "area")}
    return out


def aime_block():
    an_p = RES / "ml7_9b/aime2025/analysis_fresh_9b.json"
    an = load(an_p)
    out = {"source": rel(an_p), "analysis_sha256_in_final": load(RES / "ml7_9b/aime2025/final.json")["analysis"]["9b"]["sha256"],
           "test_touches": load(RES / "ml7_9b/aime2025/final.json")["test_touches"], "n_items": an["n_items"],
           "samples_per_item": an["samples_per_item"], "arms": {}}

    def recompute(p):
        d = load(p)
        assert d["env"]["gpu"] == "NVIDIA GB10"
        rs = d["rows"]
        acc = {c: st.mean(r["caps"][str(c)][0] for r in rs) for c in CAPS}
        return {"area": st.mean(per_item(rs, "area").values()), "acc": acc,
                "length": st.mean(min(r["think_len"], 8192) for r in rs),
                "unfinished": st.mean(0 if (r["closed"] == 1 and r["think_len"] <= 8192) else 1 for r in rs),
                "answer_unended_8k": st.mean(1 - r["caps"]["8192"][3] for r in rs)}

    b = recompute(aime_path("untouched"))
    B = an["baseline"]
    assert abs(b["area"] - B["norm_auc_log2"]) < 1e-9 and abs(b["length"] - B["natural_len"]) < 1e-9
    assert abs(b["unfinished"] - B["cut_off_share_at_max"]) < 1e-9
    assert all(abs(b["acc"][c] - B["acc"][str(c)]) < 1e-9 for c in CAPS)
    assert abs(b["answer_unended_8k"] - B["checks"]["8192"]["answer_unended_frac"]) < 1e-9
    out["arms"]["untouched"] = {"area": b["area"], "acc1k": b["acc"][1024], "acc4k": b["acc"][4096], "acc8k": b["acc"][8192],
                                "length": b["length"], "unfinished": b["unfinished"], "answer_unended_8k": b["answer_unended_8k"]}
    for a in ARMS:
        per = [recompute(aime_path(a, s)) for s in SEEDS]
        R = an["arms"][a]
        for i, p in enumerate(per):
            assert abs(p["area"] - R["norm_auc_log2"]["per_seed"][i]) < 1e-9
            assert abs(p["length"] - R["natural_len"]["per_seed"][i]) < 1e-9
            assert abs(p["unfinished"] - R["cut_off_share_at_max"]["per_seed"][i]) < 1e-9
            assert all(abs(p["acc"][c] - R["acc"][str(c)]["per_seed"][i]) < 1e-9 for c in CAPS)
            assert abs(p["answer_unended_8k"] - R["checks"]["8192"]["answer_unended_frac"][i]) < 1e-9
        out["arms"][a] = {k: st.mean(p[k] for p in per) for k in ("area", "length", "unfinished", "answer_unended_8k")}
        out["arms"][a].update({"acc1k": st.mean(p["acc"][1024] for p in per), "acc4k": st.mean(p["acc"][4096] for p in per),
                               "acc8k": st.mean(p["acc"][8192] for p in per),
                               "budget_to_match": R["tokens_to_base_max_acc"]["from_seed_mean_curve"],
                               "budget_to_match_target": R["tokens_to_base_max_acc"]["target"]})
    out["gate_minus_lora"] = {}
    for metric, blk in (("area", "effects"), ("length", "length_effects"), ("acc8k", None)):
        c = contrast(aime_path, metric)
        ref = an[blk]["wide1 - ctrl"] if blk else an["acc_diffs_descriptive"]["8192"]["wide1 - ctrl"]
        assert abs(c["seed_mean_diff"] - ref["seed_mean_diff"]) < 1e-9
        assert all(abs(x - y) < 1e-9 for x, y in zip(c["per_seed_diff"], ref["per_seed_diff"]))
        # the rule's item interval is the receipt's (same draw seed and method); keep the receipt value
        c["item_ci95_receipt"] = ref["ci95"]
        out["gate_minus_lora"][metric] = c
    out["selftest_pass"] = {f"{a}_{s}": bool(load(aime_path(a, s))["parity"]["pass"]) for a in ARMS for s in SEEDS}
    out["all_selftests_pass"] = all(out["selftest_pass"].values())
    for h in ("H5", "H6"):
        out[h] = {k: an["preregistered_hypotheses"][h][k] for k in ("verdict", "seed_mean_diff", "ci95", "per_seed_diff")}
    return out


def train_block():
    out = {"source": str(RUNS), "runs": {}}
    for d in ["ctrl_s0_cap512_fmt9b_gb10"] + [f"{a}_s{s}_cap512_r9b_gb10" for a in ARMS for s in SEEDS]:
        rows = [json.loads(line) for line in open(RUNS / d / "train_log.jsonl") if line.strip()]
        assert [r["step"] for r in rows] == list(range(len(rows)))
        m = json.load(open(RUNS / d / "meta.json"))
        r = {"steps": len(rows), "gpu": m["env"]["gpu"], "n_lora": m["n_lora"], "n_w": m["n_w"],
             "micro_tokens": m["args"]["micro_tokens"], "init_sha256": (m.get("init_from") or {}).get("sha256") if isinstance(m.get("init_from"), dict) else None,
             "logged_train_hours": sum(x.get("step_s", 0.0) for x in rows) / 3600,
             "forced_per_step": [x["forced_frac"] for x in rows]}
        for n in (10, 20):
            t = rows[-n:]
            r[f"last{n}"] = {"forced": st.mean(x["forced_frac"] for x in t), "reasoning": st.mean(x["reasoning_mean"] for x in t),
                             "acc": st.mean(x["acc"] for x in t)}
        res = RUNS / d / "meta_resume125.json"
        if res.exists():
            mr = json.load(open(res))
            r["resume"] = {"resumed_from": mr["resumed_from"], "dropped_log_lines": mr["dropped_log_lines"]["train_log.jsonl"],
                           "opt_start_step": mr["opt_at_start"]["start_step"], "source": str(res)}
        out["runs"][d] = r
    for a in ARMS:
        runs = [out["runs"][f"{a}_s{s}_cap512_r9b_gb10"] for s in SEEDS]
        out[a] = {f"last{n}_{k}": [x[f"last{n}"][k] for x in runs] for n in (10, 20) for k in ("forced", "reasoning", "acc")}
    out["params"] = {"lora": out["runs"]["ctrl_s0_cap512_r9b_gb10"]["n_lora"],
                     "gate_w": out["runs"]["wide1_s0_cap512_r9b_gb10"]["n_w"]}
    out["params"]["gated_total"] = out["params"]["lora"] + out["params"]["gate_w"]
    m4 = RES / "run_logs/ml7_4b_runs/wide1_s0_cap512_r_box/meta.json"
    if not m4.exists():  # the release keeps the 4B run logs under run_logs/ml7_4b/grpo
        m4 = RES / "run_logs/ml7_4b/grpo/wide1_s0_cap512_r_box/meta.json"
    out["micro_tokens_4b"] = {"value": json.load(open(m4))["args"]["micro_tokens"], "source": str(m4)}
    return out


def wall_block():
    """Wall-clock spans of the 9B work, from the plan's records and the AIME read's own step log."""
    if not (RES / "V2_GENERAL_BENCH_PLAN.md").exists() or not JOBLOG.exists():
        # release tree: the plan record and the launcher log are lab files, so keep the shipped wall block as data
        return json.load(open(OUT))["wall"]
    plan = (RES / "V2_GENERAL_BENCH_PLAN.md").read_text()  # lab plan record, not released
    steps = (RES / "ml7_9b/aime2025/steps.log").read_text()
    s12 = re.search(r"2026-10-14 (\d\d):(\d\d) to 2026-10-15 (\d\d):(\d\d) Israel", plan).groups()
    h12 = (24 + int(s12[2]) + int(s12[3]) / 60) - (int(s12[0]) + int(s12[1]) / 60)
    t0 = re.search(r"=== readouts 2026-10-16T(\d\d):(\d\d)", steps).groups()
    t1 = re.search(r"=== analysis start 2026-10-16T(\d\d):(\d\d)", steps).groups()
    ha = (int(t1[0]) + int(t1[1]) / 60) - (int(t0[0]) + int(t0[1]) / 60)
    s0 = re.search(r"Two Sparks, about (\d+) h wall", plan).group(1)
    # the job log of the gated seed 0 run: its time limit, and exit 124 (stopped at the limit) before the resume
    jl = JOBLOG.read_text()
    limit = int(re.search(r"max (\d+)s", jl).group(1))
    assert "rc=124" in jl
    return {"seed0_hours": float(s0), "seeds12_hours": h12, "aime_read_hours": ha,
            "wide1_s0_job_limit_hours": limit / 3600, "wide1_s0_job_rc": 124, "job_log": str(JOBLOG),
            "source": [rel(RES / "V2_GENERAL_BENCH_PLAN.md"), rel(RES / "ml7_9b/aime2025/steps.log")]}


def main():
    out = {"note": __doc__.strip().splitlines()[0], "val": val_block(), "aime": aime_block(), "train": train_block(),
           "wall": wall_block()}
    OUT.write_text(json.dumps(out, indent=1))
    v, a, t = out["val"], out["aime"], out["train"]
    print("val length", v["gate_minus_lora"]["length"]["seed_mean_diff"], v["gate_minus_lora"]["length"]["item_ci95"])
    print("val acc8k", v["gate_minus_lora"]["acc8k"]["seed_mean_diff"], v["gate_minus_lora"]["acc8k"]["item_ci95"])
    for m, c in a["gate_minus_lora"].items():
        print("aime", m, c["seed_mean_diff"], c["item_ci95"], c["item_ci95_receipt"], c["hier_ci95"], c["t_ci95"])
    print("train last20 forced", t["ctrl"]["last20_forced"], t["wide1"]["last20_forced"])
    print("wall", out["wall"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
