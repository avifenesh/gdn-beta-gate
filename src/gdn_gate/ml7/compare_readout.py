#!/usr/bin/env python3
"""Agreement test between two accuracy-vs-budget readouts of the same model on the same items (readout_vllm.py
JSON and/or broad/readout_len.py JSON, several files per side = several samples per item), plus the parity verdict
for step 7: vLLM readout vs HF readout must agree, a red arm must not.

Tests (paired by item, over the items both sides read), each a z statistic under H0 "same per-item distribution":
* accuracy at every budget, and the cut-off share at the largest budget (binary): D = mean_i(a_i - b_i), a_i / b_i =
  the item's mean over that side's samples; Var(D) = sum_i q_i (1 - q_i) (1/nA_i + 1/nB_i) / n^2 with q_i the pooled
  per-item rate.
* normalized area under accuracy vs log2 budget, and natural length (reasoning tokens at the largest budget):
  Var(D) = sum_i s_i^2 (1/nA_i + 1/nB_i) / n^2, s_i^2 the within-item variance pooled over both sides' samples
  (items with no within-item df take the mean s^2).
Family-wise alpha 0.05, Bonferroni over all tests: AGREE iff every |z| <= z_crit. Descriptive: per-item rank of
side B's natural length among side A's samples (mean 0.5 under H0).

Parity verdict (main): PASS iff green (--a vs --b) AGREES, every --red file DISAGREES with --b, and every --self file
(the --a engine, another seed) AGREES with --a.
Usage: compare_readout.py --a V.json --b HF1.json [HF2.json] [--red R.json] [--self V2.json] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from readout_vllm import SCHEMA, norm_auc, summarize  # noqa: E402

ALPHA = 0.05


def load_rows(paths: list[str]) -> tuple[list[dict], list[int], dict]:
    """Normalized rows from readout_vllm.py (schema wg-readout-v1) or readout_len.py JSON files. Several files =
    several samples per item (sample index offset per file)."""
    rows, caps, meta = [], None, []
    off = 0
    for p in paths:
        d = json.loads(Path(p).read_text())
        if d.get("schema") == SCHEMA:
            c = d["caps"]
            rr = [dict(r, sample=r["sample"] + off) for r in d["rows"]]
            meta.append({"file": p, "kind": "vllm", "arm": d["arm"], "red": d.get("red", ""), "seed": d["args"]["seed"],
                         "samples": d["args"]["samples"], "gpu": d["env"]["gpu"]})
            off += d["args"]["samples"]
        elif "items" in d and "think_len" in d:  # readout_len.py (HF)
            c = sorted(int(k) for k in d["items"])
            rr = []
            for iid, tl in d["think_len"].items():
                rec = {"id": iid, "sample": off, "think_len": tl, "closed": None, "silent_stop": None, "caps": {}}
                for k in c:
                    x = d["items"][str(k)][iid]
                    rec["caps"][str(k)] = [x["correct"], x["reasoning"], x["forced"], x["answer_ended"], None]
                rr.append(rec)
            meta.append({"file": p, "kind": "hf", "seed": d["args"]["sample_seed"], "gpu": d.get("gpu")})
            off += 1
        else:
            raise SystemExit(f"{p}: neither readout_vllm nor readout_len JSON")
        if caps is not None and c != caps:
            raise SystemExit(f"{p}: budgets {c} differ from {caps}")
        caps = c
        rows += rr
    return rows, caps, {"files": meta}


def _by_item(rows, val):
    by: dict = {}
    for r in rows:
        by.setdefault(r["id"], []).append(float(val(r)))
    return by


def _z(D, var):
    if var <= 0:
        return 0.0 if D == 0 else math.inf
    return D / math.sqrt(var)


def z_binary(A, B, ids):
    n = len(ids)
    D = sum(sum(A[i]) / len(A[i]) - sum(B[i]) / len(B[i]) for i in ids) / n
    var = 0.0
    for i in ids:
        q = (sum(A[i]) + sum(B[i])) / (len(A[i]) + len(B[i]))
        var += q * (1 - q) * (1 / len(A[i]) + 1 / len(B[i]))
    return D, _z(D, var / n ** 2)


def z_cont(A, B, ids):
    n = len(ids)
    D = sum(sum(A[i]) / len(A[i]) - sum(B[i]) / len(B[i]) for i in ids) / n
    s2 = {}
    for i in ids:
        ss, df = 0.0, 0
        for v in (A[i], B[i]):
            m = sum(v) / len(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
        if df > 0:
            s2[i] = ss / df
    if not s2:
        return D, math.nan
    fill = sum(s2.values()) / len(s2)
    var = sum(s2.get(i, fill) * (1 / len(A[i]) + 1 / len(B[i])) for i in ids)
    return D, _z(D, var / n ** 2)


def compare(ra: list[dict], rb: list[dict], caps: list[int]) -> dict:
    big = str(max(caps))
    ids = sorted(set(r["id"] for r in ra) & set(r["id"] for r in rb))
    if not ids:
        raise SystemExit("no common items")
    ra = [r for r in ra if r["id"] in set(ids)]
    rb = [r for r in rb if r["id"] in set(ids)]
    tests = {}
    for c in caps:
        k = str(c)
        D, z = z_binary(_by_item(ra, lambda r: r["caps"][k][0]), _by_item(rb, lambda r: r["caps"][k][0]), ids)
        tests[f"acc@{c}"] = {"diff_a_minus_b": D, "z": z}
    D, z = z_binary(_by_item(ra, lambda r: r["caps"][big][2]), _by_item(rb, lambda r: r["caps"][big][2]), ids)
    tests["cut_off_share@max"] = {"diff_a_minus_b": D, "z": z}
    auc = lambda r: norm_auc(caps, [r["caps"][str(c)][0] for c in caps])  # noqa: E731
    D, z = z_cont(_by_item(ra, auc), _by_item(rb, auc), ids)
    tests["norm_auc_log2"] = {"diff_a_minus_b": D, "z": z}
    nat = lambda r: min(r["think_len"], int(big))  # noqa: E731
    D, z = z_cont(_by_item(ra, nat), _by_item(rb, nat), ids)
    tests["natural_len@max"] = {"diff_a_minus_b": D, "z": z}
    zc = NormalDist().inv_cdf(1 - ALPHA / (2 * len(tests)))
    for t in tests.values():
        t["pass"] = bool(abs(t["z"]) <= zc) if not math.isnan(t["z"]) else False
    A, B = _by_item(ra, nat), _by_item(rb, nat)
    ranks = []
    for i in ids:
        for b in B[i]:
            ranks.append((sum(a < b for a in A[i]) + 0.5 * sum(a == b for a in A[i])) / len(A[i]))
    agree = all(t["pass"] for t in tests.values())
    # a red arm counts as caught only by a finite failing test (an untestable statistic is not a detection)
    disagree = any((not t["pass"]) and not math.isnan(t["z"]) for t in tests.values())
    return {"agree": agree, "disagree": disagree, "z_crit": zc, "family_alpha": ALPHA, "n_items": len(ids), "tests": tests,
            "natural_len_rank_of_b_in_a": {"mean": sum(ranks) / len(ranks), "expected": 0.5,
                                           "se_if_uniform": math.sqrt(1 / 12 / len(ranks))},
            "summary_a": summarize(ra, caps), "summary_b": summarize(rb, caps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="reference side (vLLM readout, several samples per item)")
    ap.add_argument("--b", nargs="+", required=True, help="side under test (HF readout_len.py files)")
    ap.add_argument("--red", nargs="*", default=[], help="red-arm readouts that must DISAGREE with --b")
    ap.add_argument("--diag", nargs="*", default=[], help="extra arms compared with --b, reported only (power diagnostics)")
    ap.add_argument("--self", nargs="*", default=[], dest="self_", help="--a engine, another seed: must AGREE with --a")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ra, caps, ma = load_rows(a.a)
    rb, cb, mb = load_rows(a.b)
    assert caps == cb, (caps, cb)
    res = {"a": ma, "b": mb, "green": compare(ra, rb, caps), "red": [], "self": []}
    for f in a.red:
        rr, cr, mr = load_rows([f])
        res["red"].append({"file": f, **compare(rr, rb, caps)})
    res["diag"] = []
    for f in a.diag:
        rd, _, _ = load_rows([f])
        res["diag"].append({"file": f, **compare(rd, rb, caps)})
    for f in a.self_:
        rs, cs, ms = load_rows([f])
        res["self"].append({"file": f, **compare(ra, rs, caps)})
    ok = res["green"]["agree"] and all(r["disagree"] for r in res["red"]) and all(s["agree"] for s in res["self"])
    res["verdict"] = "PASS" if ok else "FAIL"
    res["verdict_rule"] = "green agrees, every red disagrees, every self-twin agrees"
    Path(a.out).write_text(json.dumps(res, indent=1))
    show = lambda c: {k: round(v["z"], 2) for k, v in c["tests"].items()}  # noqa: E731
    print(json.dumps({"verdict": res["verdict"], "z_crit": round(res["green"]["z_crit"], 3), "green": show(res["green"]),
                      "red": [show(r) for r in res["red"]], "diag": [show(r) for r in res["diag"]], "self": [show(s) for s in res["self"]]}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
