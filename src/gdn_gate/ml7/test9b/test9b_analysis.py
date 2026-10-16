#!/usr/bin/env python3
"""test9b_analysis.py: the frozen step 7 analysis on AIME 2025 for the 9B record (Qwen3.5-9B; pre-registration in
ml_pipeline_9b/PIPELINE.md, "AIME 2025 test read at 9B (pre-registration").

Nothing of the frozen scripts is edited. fresh_analysis.py (FA) and through it step14_analysis.py (S) are imported;
FA.setup() runs unchanged (the fresh file name, the fresh: id prefix, H5/H6), then this process sets these module
constants of S:
  S.TEST_FILES  ("test_aime2025.jsonl",)  --split test accepts the AIME 2025 file (its sha256 in --manifest,
                ml_pipeline_9b/aime2025_test/split_manifest.json); the ids keep their fresh:aime2025-... form
  S.ARMS        ("ctrl", "wide1")         the two arms the 9B record trained (LoRA, Gate(0,1)); wide and ctrlpm
                were not trained at 9B
  S.CONTRASTS   (("wide1", "ctrl"),)       Gate(0,1) - LoRA, seed s minus seed s; each arm - untouched follows from
                ARMS as in the frozen script
  S.HYPOTHESES  {H5, H6}                   the 4B record's fresh-set hypotheses (FA.H56: same set key, metric, block,
                contrast, direction), their text naming AIME 2025 at 9B; the frozen H1-H4 are dropped (their sets are
                MATH-500 / GSM8K and their contrasts involve arms this record does not have)
Why fresh_analysis.py is not run as is: its --record accepts 4b (four arms through ctrlpm_analysis) or 2b (three arms
incl. wide), and its file constant is test_fresh.jsonl, so a 30-item AIME 2025 file and a two-arm record are refused.
The path taken here is FA's 2b path (S.validate, S.analyze, the rank check) with the constants above: seeds 0,1,2, no
warm-up, the untouched model as the baseline (the frozen S.validate needs a baseline readout of arm stock: it is the
reference for caps, items, samples, sampling, GPU and data sha256, and it sets the tokens-to-base target).
Per-source blocks are not computed: the file holds one source, so they would repeat the pooled result.
Extra refusals (beyond S.validate): every readout must carry args.rank 16, probe_effect 0.05, answer_max 1280, samples
8, and only fresh:aime2025- ids.

Usage: test9b_analysis.py --baseline U.json ctrl_0=P ... wide1_2=P --manifest M --out OUT
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fresh"))
sys.path.insert(0, str(HERE.parent))
import fresh_analysis as FA  # noqa: E402
import step14_analysis as S  # noqa: E402

FILE = "test_aime2025.jsonl"
ID_PREFIX = "fresh:aime2025-"
ARMS9 = ("ctrl", "wide1")
CONTRASTS9 = (("wide1", "ctrl"),)
SEEDS9 = "0,1,2"
WANT = {"rank": 16, "probe_effect": 0.05, "answer_max": 1280, "samples": 8}
TEXT = {"H5": "AIME 2025 (30 items x 8 samples), Qwen3.5-9B: wide1 (Gate(0,1)) - ctrl (LoRA) area > 0 under the "
              "decision rule",
        "H6": "AIME 2025, Qwen3.5-9B: wide1 reasons less than ctrl at the largest budget (natural length, wide1 - ctrl "
              "< 0) under the decision rule"}


def setup() -> None:
    FA.setup()
    S.TEST_FILES = (FILE,)
    S.ARMS = ARMS9
    S.CONTRASTS = CONTRASTS9
    S.HYPOTHESES = {h: {**FA.H56[h], "text": TEXT[h]} for h in ("H5", "H6")}


def run(baseline: str, slot_args: list[str], manifest: str) -> dict:
    setup()
    S.set_seeds(SEEDS9.split(","))
    base = S.load(baseline)
    slots, diag = S.parse_slots(slot_args)
    if diag:
        raise S.Refuse("test9b_analysis: no diagnostic step readouts")
    bad = S.validate(base, slots, {}, None, "test", False, manifest)
    for name, d in [("baseline", base)] + [(f"{a}_{s}", d) for (a, s), d in sorted(slots.items())]:
        for k, v in WANT.items():
            if d.get("args", {}).get(k) != v:
                bad.append(f"{name}: args.{k} {d.get('args', {}).get(k)!r}, the 9B AIME 2025 read needs {v!r}")
    ids = sorted({r["id"] for r in base["rows"]})
    if not ids or any(not i.startswith(ID_PREFIX) for i in ids):
        bad.append(f"baseline: ids outside AIME 2025 ({ID_PREFIX}...)")
    if bad:
        print("REFUSED, inputs differ:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise S.Refuse(2)
    res = S.analyze(base, slots, {}, None, manifest, None, "test")
    for h in ("H5", "H6"):
        res["preregistered_hypotheses"][h]["record"] = "9b"
        res["preregistered_hypotheses"][h]["role"] = "confirmatory (9B AIME 2025, owner approval 2026-10-15)"
    res["aime2025_9b"] = {"record": "9b", "model": "Qwen3.5-9B", "seeds": SEEDS9, "file": FILE, "n_items": len(ids),
                          "note": "30 items: the paired item bootstrap 95% interval is the main uncertainty"}
    res["receipt"] = {
        "argv": sys.argv, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": S._sha_file(__file__), "fresh_analysis_sha256": S._sha_file(FA.__file__),
        "frozen_analysis_sha256": S._sha_file(S.__file__), "python": platform.python_version(),
        "numpy": S.np.__version__,
        "inputs": {"baseline": {"file": base["_path"], "sha256": base["_sha256"], "arm": base["arm"], "role": "untouched"},
                   **{f"{x}_{s}": {"file": d["_path"], "sha256": d["_sha256"], "arm": d["arm"], "ckpt": d.get("ckpt"),
                                   "ckpt_sha256": d["env"]["sha256"].get("ckpt")} for (x, s), d in sorted(slots.items())}},
        "readout": {"seed": base["args"]["seed"], "samples": base["args"]["samples"],
                    "answer_max": base["args"].get("answer_max"), "sampling": base.get("sampling"),
                    "data_sha256": base["env"]["sha256"].get("data"), "gpu": base["env"].get("gpu")}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("slots", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.baseline, a.slots, a.manifest)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    S.print_table(res)
    for h in ("H5", "H6"):
        r = res["preregistered_hypotheses"][h]
        print(f"{h} (9b): {r['verdict']} diff {r.get('seed_mean_diff')} ci {r.get('ci95')} seeds {r.get('per_seed_diff')}")
    print(f"AIME 2025 9B receipt: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
