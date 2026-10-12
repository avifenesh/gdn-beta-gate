#!/usr/bin/env python3
"""Preflight and guard.final_test() wrapper for step14_test_local.sh (ml-pipeline step 14, step 7).

preflight --ml DIR --data DIR --plan PLAN.tsv
    Refuses (exit 2, printing every reason) unless:
    * every checkpoint in the plan exists, its run dir has a DONE marker at the planned step (warm-up 40, arms 150),
      and every arm's meta.json says it was initialized from the warm-up checkpoint (sha256) with its slot's arm,
      seed, cap 512 and answer_max 1792;
    * test_gsm8k.jsonl and test_math500.jsonl match ml_pipeline/split_manifest.json;
    * the test frame (columns id, problem, answer; GSM8K test then MATH-500, file order; .guard_state.json
      "test_frame") matches the frozen guard fingerprint, and the guard state shows no touch (or PIPELINE.md carries
      the owner's final-test override line, read with guard._override_allows);
    * PIPELINE.md shows the owner's sign-off on the third gate (C), read with the ml-pipeline hook's own rules
      (hooks/guard_training.py gate_state: a line naming the gate with an approval word and no todo/pending/not).

final --ml DIR --data DIR --plan PLAN.tsv --samples N --runner SCRIPT --res DIR --out FINAL.json
    guard.final_test() wrapped around the whole test evaluation, so the guard's single touch covers it:
    predict_fn runs `bash SCRIPT` with PHASE=readouts (every readout, resumable), checks that every test row was read
    by every planned model for N samples, runs `bash SCRIPT` with PHASE=analysis (step14_analysis.py per test set),
    and returns the per-row count of (model, sample) reads. metric_fn = share of test rows read by all models; any
    failure raises inside predict_fn, so the touch is recorded only after every readout and both analyses finished.
    final_test appends `- Step 14 final test: test_rows_read_by_all_models=... (touch N, date)` to PIPELINE.md; the
    science is in the analysis receipts.

Plan TSV columns: set, label, arm, ckpt ("none" for the untouched model), out (readout JSON), run_dir, step.
--arm_cap (default 512) and --answer_max (default 1792) are the record's training cap and answer cap (the step 7
record's values by default; the Qwen3.5-4B record passes its own, set by its step 9 answer-cap rule and cap pilots).
ctrlpm slots (the parameter-matched control of the Qwen3.5-4B record, owner approval 2026-10-11 ~23:55 Israel): the
run's meta.json must show args.arm ctrl (no gate) and args.rank PM_RANK (20), and its init_from must be a pad_rank.py
output whose receipt (<init>.pad.json, beside the resolved init path) names the warm-up checkpoint's sha256 as its
source, the file's own sha256 as recorded in meta.json, and the rank PM_RANK; the other checks are the same.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
SETS = {"gsm8k": "test_gsm8k.jsonl", "math500": "test_math500.jsonl"}
ARM_CAP, ARM_ANSWER_MAX = 512, 1792
READOUT = {"caps": "256,512,1024,2048,4096,8192", "answer_max": 1792, "seed": 20261006}
PM_ARM, PM_RANK = "ctrlpm", 20  # parameter-matched control: trainer arm ctrl, LoRA rank 20, widened warm-up init

# ml-pipeline hook (hooks/guard_training.py) gate reading, verbatim
GATE_LINE = re.compile(r"^.*\bgate\s*[\[(]?([abcd])\b.*$", re.I | re.M)
APPROVED = re.compile(r"\b(?:approved|approved-gate|overridden|override)\b", re.I)
NEGATED = re.compile(r"\b(?:not|pending|awaiting|waiting|todo|unapproved|blocked|in progress)\b", re.I)
OVERRIDE_C_LINE = re.compile(r"^.*\boverrid(?:e|den)\b.*\b(?:gate\s*c|final test|test set)\b.*$", re.I | re.M)


def third_gate_open(md: str) -> bool:
    for m in GATE_LINE.finditer(md):
        if m.group(1).upper() == "C" and APPROVED.search(m.group(0)) and not NEGATED.search(m.group(0)):
            return True
    return any(not NEGATED.search(m.group(0)) for m in OVERRIDE_C_LINE.finditer(md))


def import_guard():
    """guard.py from the repo's ml_pipeline (the code); its state and PIPELINE.md come from --ml."""
    sys.path.insert(0, str(WT / "ml_pipeline"))
    import guard  # noqa: PLC0415

    return guard


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_plan(p) -> list[dict]:
    keys = ("set", "label", "arm", "ckpt", "out", "run_dir", "step")
    return [dict(zip(keys, line.split("\t"))) for line in Path(p).read_text().splitlines() if line.strip()]


def test_ids(data: Path) -> dict:
    return {s: [json.loads(x)["id"] for x in open(data / f) if x.strip()] for s, f in SETS.items()}


def test_frame(data: Path):
    import pandas as pd  # noqa: PLC0415

    rows = [json.loads(x) for f in SETS.values() for x in open(data / f) if x.strip()]
    return pd.DataFrame([{k: r[k] for k in ("id", "problem", "answer")} for r in rows])


def preflight_reasons(ml: Path, data: Path, plan: list[dict], arm_cap: int = ARM_CAP,
                      answer_max: int = ARM_ANSWER_MAX) -> list[str]:
    bad = []
    # checkpoints, DONE markers, provenance
    warm_sha, seen = None, set()
    for r in plan:
        if r["ckpt"] == "none" or r["label"] in seen:
            continue
        seen.add(r["label"])
        ck, rd = Path(r["ckpt"]), Path(r["run_dir"])
        if not ck.is_file():
            bad.append(f"{r['label']}: checkpoint missing: {ck}")
            continue
        try:
            done = json.loads((rd / "DONE").read_text())
            if int(done.get("steps", -1)) != int(r["step"]):
                bad.append(f"{r['label']}: {rd}/DONE says steps {done.get('steps')}, planned {r['step']}")
        except (OSError, ValueError) as e:
            bad.append(f"{r['label']}: run not finished, no readable DONE in {rd} ({type(e).__name__})")
        if r["label"] == "warmup":
            warm_sha = sha(ck)
    has_warm = any(r["label"] == "warmup" for r in plan)
    for r in plan:
        if r["label"] in ("untouched", "warmup") or r["set"] != "gsm8k" or not Path(r["ckpt"]).is_file():
            continue
        rd = Path(r["run_dir"])
        try:
            meta = json.loads((rd / "meta.json").read_text())
        except (OSError, ValueError) as e:
            bad.append(f"{r['label']}: no readable meta.json in {rd} ({type(e).__name__})")
            continue
        args, init = meta.get("args", {}), meta.get("init_from") or {}
        arm, seed = r["label"].rsplit("_", 1)
        want = {"arm": arm, "seed": int(seed), "cap": arm_cap, "answer_max": answer_max}
        if arm == PM_ARM:
            want.update(arm="ctrl", rank=PM_RANK)
        for k, v in want.items():
            if args.get(k) != v:
                bad.append(f"{r['label']}: meta.json args.{k} = {args.get(k)!r}, slot needs {v!r}")
        if arm == PM_ARM and has_warm and warm_sha:
            pr = Path(str(init.get("path", "")) + ".pad.json")
            try:
                pad = json.loads(pr.read_text())
                if pad.get("source_sha256") != warm_sha or pad.get("sha256") != init.get("sha256") \
                        or pad.get("rank_to") != PM_RANK or not pad.get("verified"):
                    bad.append(f"{r['label']}: pad receipt {pr} does not tie init {str(init.get('sha256'))[:16]} to the "
                               f"warm-up checkpoint {warm_sha[:16]} at rank {PM_RANK}")
            except (OSError, ValueError) as e:
                bad.append(f"{r['label']}: no readable pad receipt {pr} ({type(e).__name__})")
        elif has_warm and warm_sha and init.get("sha256") != warm_sha:
            bad.append(f"{r['label']}: initialized from sha256 {str(init.get('sha256'))[:16]}, "
                       f"not the warm-up checkpoint {warm_sha[:16]}")
        if not init.get("applied", False):
            bad.append(f"{r['label']}: meta.json init_from.applied is not true")
    # frozen test files
    files_ok = True
    try:
        man = json.loads((ml / "split_manifest.json").read_text())
        for f in SETS.values():
            got = sha(data / f) if (data / f).is_file() else "absent"
            if got != man["files"][f]:
                files_ok = False
                bad.append(f"{f}: sha256 {got[:16]} does not match split_manifest.json {man['files'][f][:16]}")
    except (OSError, ValueError, KeyError) as e:
        files_ok = False
        bad.append(f"split manifest unreadable: {e}")
    # guard state: fingerprint and touches
    md_path = ml / "PIPELINE.md"
    md = md_path.read_text() if md_path.is_file() else ""
    guard = import_guard()
    try:
        state = json.loads((ml / guard.STATE_FILE).read_text())
        if files_ok:
            fp = guard._fingerprint(test_frame(data))
            if fp != state["test_fingerprint"]:
                bad.append(f"test frame fingerprint {fp[:16]} differs from the frozen {state['test_fingerprint'][:16]}")
        if state["test_touches"] >= 1 and not guard._override_allows(md_path, "final test"):
            bad.append(f"guard: the test set was already evaluated (test_touches {state['test_touches']}); a second "
                       "evaluation needs the owner's final-test override line in PIPELINE.md")
    except (OSError, ValueError, KeyError) as e:
        bad.append(f"guard state unreadable in {ml}: {e}")
    if not third_gate_open(md):
        row = next((ln.strip() for ln in md.splitlines() if re.search(r"\bgate\s*c\b", ln, re.I)), "no line for it")
        bad.append(f"gate C sign-off not found in {md_path} ({row!r}): step 14 runs only after the owner signs "
                   "off gate C (ml-pipeline checklist order: steps 9-12, gate C, step 13, step 14)")
    return bad


def cmd_preflight(a) -> int:
    bad = preflight_reasons(Path(a.ml), Path(a.data), read_plan(a.plan), a.arm_cap, a.answer_max)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    print("preflight: OK (checkpoints, provenance, test hashes, guard fingerprint, no earlier touch, gate C)")
    return 0


def coverage(plan: list[dict], data: Path, samples: int, answer_max: int = ARM_ANSWER_MAX) -> tuple[dict, list[str]]:
    """Per test id, the number of (model, sample) reads over every planned readout of its set; plus problems."""
    ids = test_ids(data)
    counts = {i: 0 for s in ids for i in ids[s]}
    bad = []
    for r in plan:
        p = Path(r["out"])
        if not p.is_file():
            bad.append(f"{r['set']} {r['label']}: readout missing ({p})")
            continue
        d = json.loads(p.read_text())
        ar = d.get("args", {})
        got = {"caps": ar.get("caps"), "answer_max": ar.get("answer_max"), "seed": ar.get("seed")}
        want = {**READOUT, "answer_max": answer_max}
        if got != want or ar.get("samples") != samples or d.get("red") or d.get("arm") != r["arm"]:
            bad.append(f"{r['set']} {r['label']}: readout args {got} samples {ar.get('samples')} arm {d.get('arm')} "
                       f"red {d.get('red')!r} differ from the plan")
        by: dict = {}
        for row in d.get("rows", []):
            by.setdefault(row["id"], set()).add(row["sample"])
        extra = set(by) - set(ids[r["set"]])
        if extra:
            bad.append(f"{r['set']} {r['label']}: {len(extra)} ids outside the test set, e.g. {sorted(extra)[0]}")
        for i in ids[r["set"]]:
            counts[i] += len(by.get(i, ()))
    n_models = {s: len({r["label"] for r in plan if r["set"] == s}) for s in ids}
    short = [i for s in ids for i in ids[s] if counts[i] != n_models[s] * samples]
    if short:
        bad.append(f"{len(short)} test rows not read by every model x {samples} samples, e.g. {short[0]} "
                   f"({counts[short[0]]} reads)")
    return counts, bad


def run_phase(runner: str, phase: str) -> None:
    env = dict(os.environ, PHASE=phase, STEP14_UNDER_GUARD="1")
    rc = subprocess.run(["bash", runner], env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase {phase} failed (rc {rc}); the guard touch is not recorded, rerun to resume")


def cmd_final(a) -> int:
    ml, data = Path(a.ml), Path(a.data)
    guard = import_guard()
    plan = read_plan(a.plan)
    frame = test_frame(data)
    n_models = len({r["label"] for r in plan})
    expected = n_models * a.samples
    analyses = {s: Path(a.res) / f"analysis_{s}.json" for s in SETS}

    def run_everything(features):
        run_phase(a.runner, "readouts")
        counts, bad = coverage(plan, data, a.samples, a.answer_max)
        if bad:
            raise RuntimeError("incomplete readouts, touch not recorded:\n  " + "\n  ".join(bad))
        run_phase(a.runner, "analysis")
        miss = [str(p) for p in analyses.values() if not p.is_file()]
        if miss:
            raise RuntimeError(f"analysis receipts missing: {miss}")
        return [counts[i] for i in features["id"]]

    def test_rows_read_by_all_models(y_true, y_pred):
        return float(sum(int(v) == expected for v in y_pred) / len(y_pred))

    score = guard.final_test(run_everything, frame, target="answer", metric_fn=test_rows_read_by_all_models,
                             state_dir=ml, pipeline_md=ml / "PIPELINE.md")
    state = json.loads((ml / guard.STATE_FILE).read_text())
    out = {"step": 14, "metric": "test_rows_read_by_all_models", "score": score, "test_touches": state["test_touches"],
           "models": n_models, "samples": a.samples,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "analysis": {s: {"file": str(p), "sha256": sha(p)} for s, p in analyses.items()},
           "readouts": {f"{r['set']}_{r['label']}": {"file": r["out"], "sha256": sha(r["out"]), "arm": r["arm"],
                                                       "ckpt": r["ckpt"]} for r in plan}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"step 14 final test recorded: {out['metric']}={score:.4f}, test_touches {state['test_touches']}; "
          f"analysis {', '.join(str(p) for p in analyses.values())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "final"):
        p = sub.add_parser(name)
        p.add_argument("--ml", required=True, help="ml_pipeline dir: .guard_state.json, PIPELINE.md, split_manifest.json")
        p.add_argument("--data", required=True)
        p.add_argument("--plan", required=True)
        p.add_argument("--arm_cap", type=int, default=ARM_CAP, help="the record's training cap (meta.json args.cap)")
        p.add_argument("--answer_max", type=int, default=ARM_ANSWER_MAX,
                       help="the record's answer cap: rollouts (meta.json args.answer_max) and every readout")
        if name == "final":
            p.add_argument("--samples", type=int, required=True)
            p.add_argument("--runner", required=True)
            p.add_argument("--res", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_preflight(a) if a.cmd == "preflight" else cmd_final(a)


if __name__ == "__main__":
    sys.exit(main())
