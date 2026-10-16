#!/usr/bin/env python3
"""test9b_guard.py: preflight and the guard.final_test() wrapper for the 9B AIME 2025 read (test9b_spark.sh).

fresh_guard.py (FG) is reused unchanged: this process sets its module constants
  FG.FILE    "test_aime2025.jsonl"   (frame, coverage and manifest checks read the AIME 2025 file)
  FG.BLOCK   "## AIME 2025 test read at 9B (pre-registration"
  FG.LABELS  {"4b": the 7 labels of this read, "2b": []}  (FG's label check is keyed by record name; the 9B labels sit
             under its first key, its messages say "4b labels" for them)
and calls FG.preflight_reasons and FG.cmd_final. So the guard behaviour is FG's: one guard.final_test() (the shared
ml_pipeline/guard.py) around PHASE=readouts, the coverage check (every row read by every planned model x 8 samples,
planned caps, seed, answer cap, rank, probe, arm, data sha256), PHASE=analysis, and the touch recorded only after the
analysis receipt exists; the runner checks FRESH_UNDER_GUARD as fresh_test_box.sh does.

Added here, before FG's checks (preflight refuses on any):
  * the plan holds exactly the pre-registered labels (untouched, ctrl_0-2, wide1_0-2), record 9b, answer_max 1280,
    probe 0.05, rank 16, arm per label (untouched stock, ctrl_* ctrl, wide1_* wide1);
  * every trained checkpoint's run meta.json: args arm, seed, cap 512, answer_max 1280, rank 16, probe_effect 0.05,
    alpha 0.2, model .../Qwen3.5-9B, and init_from.sha256 = the shared warm-up checkpoint's sha256
    (ckpt_expect_9b.json "warmup") with applied true;
  * frozen code: every "<sha256>  <path>" line inside the pre-registration block of PIPELINE.md matches the file
    (paths relative to the worktree), and the block lists at least the readout, analysis and guard files;
  * the guard state shows test touches 0, with no exception: this is the single read (FG alone would accept a
    later touch under an override line).
After FG.cmd_final: final.json gets the field "what" set to this read (FG writes its Phase 5 label) and "record_9b".

preflight --gd DIR --md PIPELINE.md --data DIR --plan PLAN.tsv --expect ckpt_expect_9b.json
final     --gd DIR --md PIPELINE.md --data DIR --plan PLAN.tsv --samples 8 --runner SCRIPT --res DIR --out FINAL.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "fresh"))
import fresh_guard as FG  # noqa: E402

FILE = "test_aime2025.jsonl"
BLOCK = "## AIME 2025 test read at 9B (pre-registration"
LABELS9 = ["untouched"] + [f"{a}_{s}" for a in ("ctrl", "wide1") for s in range(3)]
PLAN_WANT = {"record": "9b", "answer_max": "1280", "probe": "0.05", "rank": "16"}
META_WANT = {"cap": 512, "answer_max": 1280, "rank": 16, "probe_effect": 0.05, "alpha": 0.2}
MODEL_NAME = "Qwen3.5-9B"
MUST_FREEZE = ("readout_vllm.py", "train_grpo_vllm.py", "gate_vllm.py", "rl_common.py", "arms.py", "packing.py",
               "step14_analysis.py", "ctrlpm_analysis.py", "fresh_analysis.py", "test9b_analysis.py", "guard.py",
               "fresh_guard.py", "test9b_guard.py")
SHA_LINE = re.compile(r"^\s*([0-9a-f]{64})  (\S+)\s*$", re.M)


def setup() -> None:
    FG.FILE = FILE
    FG.BLOCK = BLOCK
    FG.LABELS = {"4b": list(LABELS9), "2b": []}


def block_text(md: str) -> str:
    """The pre-registration section: from its heading to the next '## ' heading."""
    i = md.find(BLOCK)
    if i < 0:
        return ""
    j = md.find("\n## ", i + len(BLOCK))
    return md[i:] if j < 0 else md[i:j]


def frozen_reasons(md: str, wt: Path) -> list[str]:
    bad = []
    lines = SHA_LINE.findall(block_text(md))
    if not lines:
        return ["no frozen-code sha256 lines in the pre-registration block"]
    names = {Path(p).name for _, p in lines}
    for f in MUST_FREEZE:
        if f not in names:
            bad.append(f"frozen-code list in the pre-registration block lacks {f}")
    for want, rel in lines:
        p = wt / rel
        got = FG.sha(p) if p.is_file() else "absent"
        if got != want:
            bad.append(f"frozen code {rel}: sha256 {got[:16]} differs from the pre-registered {want[:16]}")
    return bad


def arm_of(label: str) -> str:
    return "stock" if label == "untouched" else label.rsplit("_", 1)[0]


def extra_reasons(gd: Path, md: Path, plan: list[dict], expect: dict, wt: Path = WT) -> list[str]:
    bad = []
    labels = [r["label"] for r in plan]
    if sorted(labels) != sorted(LABELS9) or len(set(labels)) != len(labels):
        bad.append(f"plan labels {labels}: need exactly {LABELS9}")
    for r in plan:
        for k, v in PLAN_WANT.items():
            if r.get(k) != v:
                bad.append(f"{r['label']}: plan {k} {r.get(k)!r}, pre-registered {v!r}")
        if r["label"] in LABELS9 and r.get("arm") != arm_of(r["label"]):
            bad.append(f"{r['label']}: plan arm {r.get('arm')!r}, need {arm_of(r['label'])!r}")
        if r["label"] == "untouched" and r.get("ckpt") != "none":
            bad.append("untouched: plan carries a checkpoint")
    warm = (expect.get("warmup") or {}).get("sha256")
    if not warm:
        bad.append("ckpt_expect: no warm-up sha256 (the arms' shared init)")
    for r in plan:
        if r["ckpt"] == "none":
            continue
        rd = Path(r["ckpt"]).parent
        try:
            meta = json.loads((rd / "meta.json").read_text())
        except (OSError, ValueError) as e:
            bad.append(f"{r['label']}: no readable meta.json in {rd} ({type(e).__name__})")
            continue
        args, init = meta.get("args", {}), meta.get("init_from") or {}
        want = {**META_WANT, "arm": arm_of(r["label"]), "seed": int(r["label"].rsplit("_", 1)[1])}
        for k, v in want.items():
            if args.get(k) != v:
                bad.append(f"{r['label']}: meta.json args.{k} = {args.get(k)!r}, need {v!r}")
        if Path(str(args.get("model", ""))).name != MODEL_NAME:
            bad.append(f"{r['label']}: meta.json args.model {args.get('model')!r} is not {MODEL_NAME}")
        if warm and (init.get("sha256") != warm or not init.get("applied", False)):
            bad.append(f"{r['label']}: init_from sha256 {str(init.get('sha256'))[:16]} applied {init.get('applied')!r}, "
                       f"need the warm-up {warm[:16]} applied")
    txt = md.read_text() if md.is_file() else ""
    bad += frozen_reasons(txt, wt) if BLOCK in txt else []
    guard = FG.import_guard()
    try:
        st = json.loads((gd / guard.STATE_FILE).read_text())
        if st.get("test_touches") != 0:
            bad.append(f"guard: {gd} shows test touches {st.get('test_touches')}; this read happens once, at touch 0")
    except (OSError, ValueError) as e:
        bad.append(f"guard state unreadable in {gd}: {e}")
    return bad


def preflight(a) -> int:
    setup()
    plan = FG.read_plan(a.plan)
    expect = json.loads(Path(a.expect).read_text())
    bad = extra_reasons(Path(a.gd), Path(a.md), plan, expect)
    bad += [b.replace("4b labels missing", "9b labels missing")
            for b in FG.preflight_reasons(Path(a.gd), Path(a.md), Path(a.data), plan, expect)]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(dict.fromkeys(bad)), file=sys.stderr)
        return 2
    print("preflight: OK (pre-registration and frozen code, AIME 2025 file and fingerprint, touches 0, checkpoints "
          "sha256 and provenance, plan arguments)")
    return 0


def final(a) -> int:
    setup()
    rc = FG.cmd_final(a)
    out = Path(a.out)
    d = json.loads(out.read_text())
    d["what"] = "AIME 2025 (30 items) read by the 9B record (Qwen3.5-9B: untouched, LoRA ctrl and Gate(0,1) wide1 x 3)"
    d["record_9b"] = {"file": FILE, "guard_dir": str(a.gd), "pipeline_md": str(a.md)}
    out.write_text(json.dumps(d, indent=1))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "final"):
        p = sub.add_parser(name)
        for k in ("--gd", "--md", "--data", "--plan"):
            p.add_argument(k, required=True)
        if name == "preflight":
            p.add_argument("--expect", required=True)
        else:
            p.add_argument("--samples", type=int, required=True)
            p.add_argument("--runner", required=True)
            p.add_argument("--res", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return preflight(a) if a.cmd == "preflight" else final(a)


if __name__ == "__main__":
    sys.exit(main())
