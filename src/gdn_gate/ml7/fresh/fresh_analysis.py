#!/usr/bin/env python3
"""fresh_analysis.py: the frozen step 7 analysis on the Phase 5 fresh held-out set (AIME 2024/2025/2026, AMC 2023;
pre-registration in ml_pipeline_4b/PIPELINE.md, "Fresh test set").

Nothing of the frozen scripts is edited: step14_analysis.py (S) and ctrlpm_analysis.py (P) are imported and only these
module constants are set in this process:
  S.TEST_FILES        ("test_fresh.jsonl",)  so --split test accepts the fresh file (its sha256 in --manifest,
                      ml_pipeline_4b/fresh_test/split_manifest.json)
  S.TEST_ID_PREFIXES  ("fresh:",)
  hypotheses          + H5 (fresh: wide1 - ctrl area > 0) and H6 (fresh: wide1 - ctrl natural length < 0), the
                      frozen H1-H4 stay and read "not applicable" (their sets are MATH-500 / GSM8K)
Record 4b (--record 4b): four arms through P.run (ctrlpm = readout --arm ctrl --rank 20, P's own checks and its
three-arm self-check), seeds 0,1,2, warm-up role. Record 2b (--record 2b): the three frozen arms through S.validate /
S.analyze, seeds 0-5, no warm-up. The hypotheses' rows name the record (H5 / H6 confirmatory at 4B; at 2B the same
rows are the secondary replication H5-2B / H6-2B).
Per source (descriptive only, never a verdict): every readout restricted to the ids of one source, S.analyze again,
seed means of area and natural length and the effect blocks, marked descriptive.

Usage: fresh_analysis.py --record 4b|2b --baseline U.json [--warmup W.json] ARM_SEED=path ... --manifest M --out OUT
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ctrlpm_analysis as P  # noqa: E402
import step14_analysis as S  # noqa: E402

FRESH_FILE = "test_fresh.jsonl"
SOURCES = ("aime2024", "aime2025", "aime2026", "amc2023")
H56 = {
    "H5": {"set": "fresh", "metric": "norm_auc_log2", "block": "effects", "contrast": "wide1 - ctrl",
           "direction": "positive", "text": "fresh set (AIME 2024/2025/2026, AMC 2023): wide1 - ctrl area > 0 under "
           "the decision rule"},
    "H6": {"set": "fresh", "metric": "natural_len", "block": "length_effects", "contrast": "wide1 - ctrl",
           "direction": "negative", "text": "fresh set: wide1 reasons less than ctrl at the largest budget (natural "
           "length, wide1 - ctrl < 0) under the decision rule"},
}
SEEDS = {"4b": "0,1,2", "2b": "0,1,2,3,4,5"}


def setup() -> None:
    S.TEST_FILES = (FRESH_FILE,)
    S.TEST_ID_PREFIXES = ("fresh:",)
    S.HYPOTHESES = {**S.HYPOTHESES, **H56}
    P.FROZEN["HYPOTHESES"] = {**P.FROZEN["HYPOTHESES"], **H56}
    P.EXT["HYPOTHESES"] = {**P.EXT["HYPOTHESES"], **H56}


def source_of(i: str) -> str:
    return i.split(":", 1)[1].split("-", 1)[0]


def restrict(d: dict, src: str) -> dict:
    e = copy.copy(d)
    e["rows"] = [r for r in d["rows"] if source_of(r["id"]) == src]
    return e


def per_source(base, slots, warm, manifest) -> dict:
    out = {}
    for src in SOURCES:
        b = restrict(base, src)
        if not b["rows"]:
            continue
        w = restrict(warm, src) if warm is not None else None
        sl = {k: restrict(d, src) for k, d in slots.items()}
        r = S.analyze(b, sl, {}, None, manifest, w, "test")
        out[src] = {"descriptive": "per-source restriction, no verdict (the pre-registered rule applies to the pooled "
                                   "set only)", "n_items": r["n_items"],
                    "baseline_norm_auc_log2": r["baseline"].get("norm_auc_log2") if isinstance(r["baseline"], dict) else None,
                    "arms": {a: {"norm_auc_log2": v["norm_auc_log2"], "natural_len": v["natural_len"],
                                 "acc_8192": v["acc"].get("8192")} for a, v in r["arms"].items()},
                    "effects": {k: {x: e[x] for x in ("per_seed_diff", "seed_mean_diff", "ci95")} for k, e in r["effects"].items()},
                    "length_effects": {k: {x: e[x] for x in ("per_seed_diff", "seed_mean_diff", "ci95")}
                                       for k, e in r["length_effects"].items()}}
    return out


def run(a) -> dict:
    setup()
    seeds = a.seeds or SEEDS[a.record]
    if a.record == "4b":
        ns = argparse.Namespace(split="test", baseline=a.baseline, warmup=a.warmup, slots=a.slots, out=a.out,
                                manifest=a.manifest, seeds=seeds, pm_rank=a.pm_rank, touch1=None)
        res = P.run(ns)
        # per-source blocks with the four arms (P's constants are active after run)
        base = S.load(a.baseline)
        warm = S.load(a.warmup) if a.warmup else None
        slots, _ = S.parse_slots(a.slots)
        P.relabel(slots)
    else:
        S.set_seeds(seeds.split(","))
        base = S.load(a.baseline)
        warm = S.load(a.warmup) if a.warmup else None
        slots, diag = S.parse_slots(a.slots)
        if diag:
            raise S.Refuse("fresh_analysis: no diagnostic step readouts")
        bad = S.validate(base, slots, {}, warm, "test", False, a.manifest)
        bad += [f"{k[0]}_{k[1]}: args.rank {d['args'].get('rank')!r}, need 16" for k, d in slots.items()
                if d["args"].get("rank") != 16]
        if bad:
            print("REFUSED, inputs differ:\n  " + "\n  ".join(bad), file=sys.stderr)
            raise S.Refuse(2)
        res = S.analyze(base, slots, {}, None, a.manifest, warm, "test")
    ids = sorted({r["id"] for r in base["rows"]})
    if not ids or any(not i.startswith("fresh:") for i in ids):
        raise S.Refuse("fresh_analysis: the baseline holds non-fresh ids")
    for h in ("H5", "H6"):
        res["preregistered_hypotheses"][h]["record"] = a.record
        res["preregistered_hypotheses"][h]["role"] = "confirmatory" if a.record == "4b" else "secondary replication (H%s-2B)" % h[1]
    res["per_source_descriptive"] = per_source(base, slots, warm, a.manifest)
    res["fresh"] = {"record": a.record, "seeds": seeds, "file": FRESH_FILE,
                    "sources": {s: sum(source_of(i) == s for i in ids) for s in SOURCES}}
    res["receipt_fresh"] = {"argv": sys.argv, "script_sha256": S._sha_file(__file__),
                            "frozen_analysis_sha256": S._sha_file(S.__file__), "ctrlpm_analysis_sha256": S._sha_file(P.__file__),
                            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                            "python": platform.python_version(), "numpy": S.np.__version__}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True, choices=("4b", "2b"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--warmup", default=None)
    ap.add_argument("slots", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--pm_rank", type=int, default=P.PM_RANK)
    a = ap.parse_args(argv)
    res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for h in ("H5", "H6"):
        r = res["preregistered_hypotheses"][h]
        print(f"{h} ({a.record}): {r['verdict']} diff {r.get('seed_mean_diff')} ci {r.get('ci95')} seeds {r.get('per_seed_diff')}")
    print(f"fresh receipt: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
