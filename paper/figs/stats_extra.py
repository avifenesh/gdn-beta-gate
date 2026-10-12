"""Extra statistics for the revised paper, computed from the per-sample rows of the readout files. CPU only.

1. For the main arm contrasts: the analysis receipt's own seed-mean difference (reproduced here from the rows as a
   check), a seed-level t-interval (per-seed differences, t with k-1 degrees of freedom) and a hierarchical bootstrap
   interval (seeds resampled with replacement, paired by index, then items resampled with replacement; 10,000 draws).
2. Censoring-aware length on the fresh set and on MATH-500 (4B): the restricted mean at 8,192 (each open trace counts
   as 8,192; this equals the paper's natural length), the median reasoning length (open traces are right-censored at
   8,192, so the median is defined only when fewer than half are open), the share closed by 2,048, 4,096 and 8,192,
   and the mean length of correct traces.
3. The item split of MATH-500 and fresh samples in both directions: conditioned on whether LoRA closes, and on
   whether Gate(0,1) closes (samples matched by item, seed and sample index).
4. A minimum detectable effect for the Olmo RULER tasks (100 items per task), from the observed discordance.
Writes paper/data/extra.json. Run: python3 paper/figs/stats_extra.py
"""
import gzip
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper/data/extra.json"
CAPS = [256, 512, 1024, 2048, 4096, 8192]
DRAWS, SEED = 10000, 20261012
T975 = {2: 4.302652729911275, 5: 2.570581835636314}


def norm_auc(accs):
    x = [math.log2(c) for c in CAPS]
    a = sum((x[j + 1] - x[j]) * (accs[j] + accs[j + 1]) / 2 for j in range(len(x) - 1))
    return a / (x[-1] - x[0])


def path_of(model, setname, arm, seed):
    if setname == "fresh":
        tag = "untouched" if arm == "untouched" else ("warmup" if arm == "warmup" else f"{arm}_{seed}")
        return RES / f"fresh/readout_fresh_{model}_{tag}.json"
    if model == "4b":
        if arm == "ctrlpm":
            return RES / f"ml7_4b/step14_ctrlpm/readout_{setname}_ctrlpm_{seed}.json"
        tag = arm if arm in ("untouched", "warmup") else f"{arm}_{seed}"
        return RES / f"ml7_4b/step14/readout_{setname}_{tag}.json"
    d = "step14" if (arm in ("untouched", "warmup") or seed < 3) else "step14_ext"
    tag = arm if arm in ("untouched", "warmup") else f"{arm}_{seed}"
    return RES / f"ml7/{d}/readout_{setname}_{tag}.json"


_cache = {}


def rows(model, setname, arm, seed=None):
    p = path_of(model, setname, arm, seed)
    if p not in _cache:
        _cache[p] = json.loads(gzip.open(str(p) + ".gz").read() if not p.exists() else p.read_text())["rows"]
    return _cache[p]


def per_item(rs, metric):
    by = {}
    for r in rs:
        if metric == "area":
            v = norm_auc([r["caps"][str(c)][0] for c in CAPS])
        else:
            v = min(r["think_len"], 8192)
        by.setdefault(r["id"], []).append(v)
    return {k: sum(v) / len(v) for k, v in by.items()}


def contrast(model, setname, metric, a, b, seeds):
    ids = None
    A, B = [], []
    for s in seeds:
        pa, pb = per_item(rows(model, setname, a, s), metric), per_item(rows(model, setname, b, s), metric)
        if ids is None:
            ids = sorted(pa)
        A.append(np.array([pa[i] for i in ids]))
        B.append(np.array([pb[i] for i in ids]))
    D = np.array(A) - np.array(B)  # seeds x items
    k, n = D.shape
    per_seed = D.mean(axis=1)
    mean = float(per_seed.mean())
    sd = float(per_seed.std(ddof=1))
    t = T975[k - 1]
    rng = np.random.default_rng(SEED)
    m = np.empty(DRAWS)
    for j in range(DRAWS):
        si = rng.integers(0, k, size=k)
        ii = rng.integers(0, n, size=n)
        m[j] = D[np.ix_(si, ii)].mean()
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"model": model, "set": setname, "metric": metric, "contrast": f"{a} - {b}", "seeds": list(seeds),
            "n_items": n, "seed_mean_diff": mean, "per_seed_diff": [float(x) for x in per_seed],
            "n_pos": int((per_seed > 0).sum()), "n_neg": int((per_seed < 0).sum()),
            "t_ci95": [mean - t * sd / math.sqrt(k), mean + t * sd / math.sqrt(k)],
            "t_excludes_0": bool(abs(mean) > t * sd / math.sqrt(k)),
            "hier_ci95": [float(lo), float(hi)], "hier_excludes_0": bool(lo > 0 or hi < 0)}


def length_profile(model, setname, arm, seeds):
    out = {"restricted_mean_8192": [], "median": [], "closed_by_2048": [], "closed_by_4096": [], "closed_by_8192": [],
           "correct_mean": []}
    for s in seeds:
        rs = rows(model, setname, arm, s)
        L = [min(r["think_len"], 8192) for r in rs]
        closed = [r["closed"] == 1 and r["think_len"] <= 8192 for r in rs]
        out["restricted_mean_8192"].append(st.mean(L))
        srt = sorted(x if c else math.inf for x, c in zip(L, closed))
        med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
        out["median"].append(med if math.isfinite(med) else None)
        for c in (2048, 4096, 8192):
            out[f"closed_by_{c}"].append(sum(1 for x, cl in zip(L, closed) if cl and x <= c) / len(L))
        cor = [x for x, r in zip(L, rs) if r["caps"]["8192"][0] == 1]
        out["correct_mean"].append(st.mean(cor) if cor else None)
    res = {}
    for k, v in out.items():
        vv = [x for x in v if x is not None]
        res[k] = {"per_seed": v, "seed_mean": (st.mean(vv) if len(vv) == len(v) else None)}
    return res


def split(model, setname, cond_arm, arms, seeds):
    """Accuracy at 8,192 of each arm on samples where cond_arm closes vs leaves open (matched item, seed, sample)."""
    res = {"closed": {}, "open": {}, "share_open": []}
    for s in seeds:
        key = {(r["id"], r["sample"]): (r["closed"] == 1 and r["think_len"] <= 8192) for r in rows(model, setname, cond_arm, s)}
        res["share_open"].append(sum(1 for v in key.values() if not v) / len(key))
        for a in arms:
            for grp, want in (("closed", True), ("open", False)):
                v = [r["caps"]["8192"][0] for r in rows(model, setname, a, s) if key[(r["id"], r["sample"])] == want]
                res[grp].setdefault(a, []).append(st.mean(v) if v else None)
    for grp in ("closed", "open"):
        res[grp] = {a: {"per_seed": v, "seed_mean": st.mean(v)} for a, v in res[grp].items()}
    res["share_open"] = {"per_seed": res["share_open"], "seed_mean": st.mean(res["share_open"])}
    return res


def olmo_mde():
    d = json.load(open(RES / "olmo/vllm2/ruler_paired.json"))
    out = []
    for r in d["rows"]:
        if r["arm"] != "erase":
            continue
        n = r["n"]
        disc = (r["better"] + r["worse"]) / n
        var = max(disc - r["diff"] ** 2, 0.0)
        se = math.sqrt(var / n)
        se_floor = math.sqrt(2 * r["stock"] * (1 - r["stock"]) / n)
        out.append({"L": r["L"], "task": r["task"], "n": n, "stock": r["stock"], "diff": r["diff"],
                    "mde_paired_2p8se": 2.8 * se, "mde_unpaired_2p8se": 2.8 * se_floor})
    noncw = [x for x in out if x["task"] != "ruler_cwe" and x["stock"] < 0.99]
    return {"rows": out, "source": str(Path(RES / "olmo/vllm2/ruler_paired.json").relative_to(ROOT)),
            "noncwe_below_ceiling_unpaired_mde": {"min": min(x["mde_unpaired_2p8se"] for x in noncw),
                                                  "max": max(x["mde_unpaired_2p8se"] for x in noncw)},
            "noncwe_below_ceiling_paired_mde": {"min": min(x["mde_paired_2p8se"] for x in noncw),
                                                "max": max(x["mde_paired_2p8se"] for x in noncw)},
            "cwe_unpaired_mde": {L: x["mde_unpaired_2p8se"] for x in out if x["task"] == "ruler_cwe" for L in [x["L"]]}}


def g1_decomp():
    """Validation (G1): split the gap between a gated checkpoint and the LoRA arm into the part the trained gate
    carries at inference (full - strip) and the part the co-trained LoRA carries without the gate (strip - LoRA)."""
    g = json.load(open(RES / "mech/g1g2_summary.json"))["g1_condition_means"]
    out = {"source": str(Path(RES / "mech/g1g2_summary.json").relative_to(ROOT))}
    for arm in ("wide1", "wide"):
        full, strip, ctrl = (g[f"{arm}_full"]["natural_len"], g[f"{arm}_strip"]["natural_len"], g["ctrl"]["natural_len"])
        gap = full - ctrl
        out[arm] = {"gap_full_minus_ctrl": gap, "gate_at_inference": full - strip, "cotrained_lora": strip - ctrl,
                    "share_gate_at_inference": (full - strip) / gap, "share_cotrained_lora": (strip - ctrl) / gap}
    return out


def derived_ratios():
    """Length cuts quoted as percentages, and the 4B budget-to-match values rounded to 2 significant figures."""
    a4m = json.load(open(RES / "ml7_4b/step14_ctrlpm/analysis_math500.json"))
    a4g = json.load(open(RES / "ml7_4b/step14_ctrlpm/analysis_gsm8k.json"))
    a2g = json.load(open(RES / "ml7/step14/analysis_gsm8k.json"))
    nl = lambda d, a: d["arms"][a]["natural_len"]["seed_mean"]  # noqa: E731
    return {
        "source": "analysis receipts of the 4B (ml7_4b/step14_ctrlpm) and 2B read-1 (ml7/step14) test reads",
        "cut_4b_math500_wide1_vs_ctrl": 1 - nl(a4m, "wide1") / nl(a4m, "ctrl"),
        "cut_4b_gsm8k_wide1_vs_ctrl": 1 - nl(a4g, "wide1") / nl(a4g, "ctrl"),
        "cut_4b_gsm8k_wide_vs_ctrl": 1 - nl(a4g, "wide") / nl(a4g, "ctrl"),
        "cut_2b_gsm8k_wide_vs_ctrl_read1": 1 - nl(a2g, "wide") / nl(a2g, "ctrl"),
        "humaneval_hw_gap_ctrl_s0": None,
    }


def check_against(receipt, res, block, name):
    d = json.load(open(receipt))
    e = d[block][res["contrast"]]
    assert abs(e["seed_mean_diff"] - res["seed_mean_diff"]) < 1e-9, (name, e["seed_mean_diff"], res["seed_mean_diff"])
    return {"receipt": str(Path(receipt).relative_to(ROOT)), "receipt_ci95_item": e["ci95"], "receipt_verdict": e["verdict"]}


def main():
    S3, S35, S6 = [0, 1, 2], [3, 4, 5], [0, 1, 2, 3, 4, 5]
    A4M = RES / "ml7_4b/step14_ctrlpm/analysis_math500.json"
    A4G = RES / "ml7_4b/step14_ctrlpm/analysis_gsm8k.json"
    F4, F2 = RES / "fresh/analysis_fresh_4b.json", RES / "fresh/analysis_fresh_2b.json"
    A2M, A2G = RES / "ml7/step14/analysis_math500.json", RES / "ml7/step14/analysis_gsm8k.json"
    AXM, AXG = RES / "ml7/step14_ext/analysis_math500.json", RES / "ml7/step14_ext/analysis_gsm8k.json"
    plan = [
        ("4b_math500_area_wide-ctrl", "4b", "math500", "area", "wide", "ctrl", S3, A4M, "effects"),
        ("4b_math500_area_wide1-ctrl", "4b", "math500", "area", "wide1", "ctrl", S3, A4M, "effects"),
        ("4b_math500_area_wide1-ctrlpm", "4b", "math500", "area", "wide1", "ctrlpm", S3, A4M, "effects"),
        ("4b_math500_area_wide-wide1", "4b", "math500", "area", "wide", "wide1", S3, A4M, "effects"),
        ("4b_math500_area_ctrlpm-ctrl", "4b", "math500", "area", "ctrlpm", "ctrl", S3, A4M, "effects"),
        ("4b_math500_len_wide1-ctrl", "4b", "math500", "len", "wide1", "ctrl", S3, A4M, "length_effects"),
        ("4b_math500_len_wide-wide1", "4b", "math500", "len", "wide", "wide1", S3, A4M, "length_effects"),
        ("4b_gsm8k_len_wide-ctrl", "4b", "gsm8k", "len", "wide", "ctrl", S3, A4G, "length_effects"),
        ("4b_gsm8k_len_wide1-ctrl", "4b", "gsm8k", "len", "wide1", "ctrl", S3, A4G, "length_effects"),
        ("4b_gsm8k_len_wide1-ctrlpm", "4b", "gsm8k", "len", "wide1", "ctrlpm", S3, A4G, "length_effects"),
        ("4b_gsm8k_len_wide-wide1", "4b", "gsm8k", "len", "wide", "wide1", S3, A4G, "length_effects"),
        ("4b_gsm8k_area_ctrlpm-ctrl", "4b", "gsm8k", "area", "ctrlpm", "ctrl", S3, A4G, "effects"),
        ("4b_fresh_area_wide1-ctrl", "4b", "fresh", "area", "wide1", "ctrl", S3, F4, "effects"),
        ("4b_fresh_len_wide1-ctrl", "4b", "fresh", "len", "wide1", "ctrl", S3, F4, "length_effects"),
        ("4b_fresh_area_wide1-ctrlpm", "4b", "fresh", "area", "wide1", "ctrlpm", S3, F4, "effects"),
        ("4b_fresh_len_wide1-ctrlpm", "4b", "fresh", "len", "wide1", "ctrlpm", S3, F4, "length_effects"),
        ("4b_fresh_area_wide-wide1", "4b", "fresh", "area", "wide", "wide1", S3, F4, "effects"),
        ("4b_fresh_len_wide-wide1", "4b", "fresh", "len", "wide", "wide1", S3, F4, "length_effects"),
        ("2b_math500_area_wide-ctrl_r1", "2b", "math500", "area", "wide", "ctrl", S3, A2M, "effects"),
        ("2b_math500_area_wide1-ctrl_r1", "2b", "math500", "area", "wide1", "ctrl", S3, A2M, "effects"),
        ("2b_math500_area_wide-ctrl_r2", "2b", "math500", "area", "wide", "ctrl", S35, AXM, "effects"),
        ("2b_math500_area_wide1-ctrl_r2", "2b", "math500", "area", "wide1", "ctrl", S35, AXM, "effects"),
        ("2b_gsm8k_len_wide-ctrl_r2", "2b", "gsm8k", "len", "wide", "ctrl", S35, AXG, "length_effects"),
        ("2b_fresh_area_wide1-ctrl", "2b", "fresh", "area", "wide1", "ctrl", S6, F2, "effects"),
        ("2b_fresh_area_wide-wide1", "2b", "fresh", "area", "wide", "wide1", S6, F2, "effects"),
        ("2b_fresh_len_wide-ctrl", "2b", "fresh", "len", "wide", "ctrl", S6, F2, "length_effects"),
    ]
    out = {"note": __doc__.strip().splitlines()[0], "draws": DRAWS, "rng_seed": SEED, "contrasts": {},
           "length_profile": {}, "split": {}}
    for name, model, setname, metric, a, b, seeds, rec, block in plan:
        r = contrast(model, setname, metric, a, b, seeds)
        r.update(check_against(rec, r, block, name))
        out["contrasts"][name] = r
        print(f"{name:32s} {r['seed_mean_diff']:+9.4f} item {r['receipt_ci95_item']} t {r['t_ci95']} "
              f"hier {r['hier_ci95']} {r['n_pos']}+/{r['n_neg']}-")
    for arm in ("untouched", "warmup", "ctrl", "ctrlpm", "wide1", "wide"):
        seeds = [None] if arm in ("untouched", "warmup") else S3
        out["length_profile"][f"4b_fresh_{arm}"] = length_profile("4b", "fresh", arm, seeds)
        out["length_profile"][f"4b_math500_{arm}"] = length_profile("4b", "math500", arm, seeds)
    for arm in ("untouched", "ctrl", "wide1", "wide"):
        seeds = [None] if arm == "untouched" else S6
        out["length_profile"][f"2b_fresh_{arm}"] = length_profile("2b", "fresh", arm, seeds)
    arms4 = ["ctrl", "ctrlpm", "wide1", "wide"]
    for setname in ("math500", "fresh"):
        out["split"][f"4b_{setname}_by_ctrl"] = split("4b", setname, "ctrl", arms4, S3)
        out["split"][f"4b_{setname}_by_wide1"] = split("4b", setname, "wide1", arms4, S3)
    out["olmo_mde"] = olmo_mde()
    out["g1_decomp"] = g1_decomp()
    out["derived"] = derived_ratios()
    der = json.load(open(ROOT / "paper/data/derived.json"))["forgetting_humaneval"]
    out["derived"]["humaneval_hw_gap_ctrl_s0"] = der["4B_main_ctrl"]["s0"]["pass1"] - der["4B_same_hw_s0"]["ctrl"]["pass1"]
    print("derived", out["derived"])
    print("g1", json.dumps(out["g1_decomp"]))
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out["length_profile"].items():
        print(k, {m: (None if x["seed_mean"] is None else round(x["seed_mean"], 3)) for m, x in v.items()},
              "median per seed", v["median"]["per_seed"])
    for k, v in out["split"].items():
        print(k, "open share", round(v["share_open"]["seed_mean"], 3),
              {g: {a: round(x["seed_mean"], 3) for a, x in v[g].items()} for g in ("closed", "open")})
    m = out["olmo_mde"]
    print("olmo mde", m["noncwe_below_ceiling_unpaired_mde"], m["noncwe_below_ceiling_paired_mde"], m["cwe_unpaired_mde"])


if __name__ == "__main__":
    main()
