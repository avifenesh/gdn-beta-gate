#!/usr/bin/env python3
"""fresh_guard.py: preflight and guard.final_test() wrapper for the Phase 5 fresh held-out set (fresh_test_box.sh).

preflight --gd GUARD_DIR --md PIPELINE.md --data DATA_DIR --plan PLAN.tsv --expect ckpt_expect.json
    Refuses (exit 2, every reason printed) unless:
    * the pre-registration block ("## Fresh test set (pre-registration") is in PIPELINE.md;
    * test_fresh.jsonl matches GUARD_DIR/split_manifest.json and the guard fingerprint (frame id, problem, answer, file
      order), and the guard state shows test touches 0;
    * every checkpoint in the plan exists, has the sha256 its label has in ckpt_expect.json (the sha256 the earlier
      test readouts recorded for that run), and its run dir's DONE marker names the planned step (150; warm-up 40);
    * every plan label is a pre-registered label (4b: untouched, warmup, ctrl/wide/wide1/ctrlpm x 0-2; 2b: untouched,
      ctrl/wide/wide1 x 0-5) and the 4b labels are all present (the 2b block is all or nothing).

final --gd GUARD_DIR --md PIPELINE.md --data DATA_DIR --plan PLAN.tsv --samples N --runner SCRIPT --res DIR --out F
    guard.final_test() around the whole evaluation: predict_fn runs `bash SCRIPT` PHASE=readouts, checks that every
    fresh row was read by every planned model x N samples with the planned arguments, runs PHASE=analysis, and returns
    per-row read counts; the touch is recorded only after every readout and every analysis exists.

Plan TSV columns: label, record, arm, ckpt ("none" = untouched), out, model, answer_max, probe, rank.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
FILE = "test_fresh.jsonl"
BLOCK = "## Fresh test set (pre-registration"
READOUT = {"caps": "256,512,1024,2048,4096,8192", "seed": 20261006}
KEYS = ("label", "record", "arm", "ckpt", "out", "model", "answer_max", "probe", "rank")
LABELS = {"4b": ["4b_untouched", "4b_warmup"] + [f"4b_{a}_{s}" for a in ("ctrl", "wide", "wide1", "ctrlpm") for s in range(3)],
          "2b": ["2b_untouched"] + [f"2b_{a}_{s}" for a in ("ctrl", "wide", "wide1") for s in range(6)]}


def import_guard():
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
    return [dict(zip(KEYS, ln.split("\t"))) for ln in Path(p).read_text().splitlines() if ln.strip()]


def frame(data: Path):
    import pandas as pd  # noqa: PLC0415

    rows = [json.loads(x) for x in open(data / FILE) if x.strip()]
    return pd.DataFrame([{k: r[k] for k in ("id", "problem", "answer")} for r in rows])


def preflight_reasons(gd: Path, md: Path, data: Path, plan: list[dict], expect: dict) -> list[str]:
    bad = []
    txt = md.read_text() if md.is_file() else ""
    if BLOCK not in txt:
        bad.append(f"no pre-registration block {BLOCK!r} in {md}: the fresh set is read only after it is written")
    labels = [r["label"] for r in plan]
    if len(set(labels)) != len(labels):
        bad.append("plan has duplicate labels")
    allowed = set(LABELS["4b"]) | set(LABELS["2b"])
    for x in labels:
        if x not in allowed:
            bad.append(f"{x}: not a pre-registered label")
    if not set(LABELS["4b"]) <= set(labels):
        bad.append(f"4b labels missing: {sorted(set(LABELS['4b']) - set(labels))}")
    two = set(labels) & set(LABELS["2b"])
    if two and two != set(LABELS["2b"]):
        bad.append(f"2b block incomplete (all or nothing): missing {sorted(set(LABELS['2b']) - two)}")
    for r in plan:
        if r["ckpt"] == "none":
            if r["arm"] != "stock":
                bad.append(f"{r['label']}: no checkpoint but arm {r['arm']}")
            continue
        e = expect.get(r["label"])
        ck = Path(r["ckpt"])
        if e is None:
            bad.append(f"{r['label']}: no expected sha256 in ckpt_expect.json")
        elif not ck.is_file():
            bad.append(f"{r['label']}: checkpoint missing: {ck}")
        else:
            got = sha(ck)
            if got != e["sha256"]:
                bad.append(f"{r['label']}: checkpoint sha256 {got[:16]} differs from the recorded {e['sha256'][:16]}")
            step = int(ck.stem.split("step")[-1])
            try:
                done = json.loads((ck.parent / "DONE").read_text())
                if int(done.get("steps", -1)) != step:
                    bad.append(f"{r['label']}: DONE says steps {done.get('steps')}, checkpoint step {step}")
            except (OSError, ValueError) as ex:
                bad.append(f"{r['label']}: no readable DONE beside {ck} ({type(ex).__name__})")
    guard = import_guard()
    try:
        man = json.loads((gd / "split_manifest.json").read_text())
        got = sha(data / FILE) if (data / FILE).is_file() else "absent"
        if got != man["files"][FILE]:
            bad.append(f"{FILE}: sha256 {got[:16]} differs from {gd}/split_manifest.json {man['files'][FILE][:16]}")
        else:
            st = json.loads((gd / guard.STATE_FILE).read_text())
            fp = guard._fingerprint(frame(data))
            if fp != st["test_fingerprint"]:
                bad.append(f"fresh frame fingerprint {fp[:16]} differs from the frozen {st['test_fingerprint'][:16]}")
            if st["test_touches"] >= 1 and not guard._override_allows(md, "final test fresh"):
                bad.append(f"guard: the fresh set was already evaluated (test_touches {st['test_touches']})")
    except (OSError, ValueError, KeyError) as ex:
        bad.append(f"guard state or manifest unreadable in {gd}: {ex}")
    return bad


def coverage(plan: list[dict], data: Path, samples: int) -> tuple[dict, list[str]]:
    ids = [json.loads(x)["id"] for x in open(data / FILE) if x.strip()]
    counts = {i: 0 for i in ids}
    bad = []
    for r in plan:
        p = Path(r["out"])
        if not p.is_file():
            bad.append(f"{r['label']}: readout missing ({p})")
            continue
        d = json.loads(p.read_text())
        ar = d.get("args", {})
        want = {"caps": READOUT["caps"], "seed": READOUT["seed"], "answer_max": int(r["answer_max"]), "samples": samples,
                "rank": int(r["rank"]), "probe_effect": float(r["probe"])}
        got = {k: ar.get(k) for k in want}
        if got != want or d.get("red") or d.get("arm") != r["arm"] or d["env"]["sha256"].get("data") != sha(data / FILE):
            bad.append(f"{r['label']}: readout args {got} arm {d.get('arm')} red {d.get('red')!r} differ from the plan {want}")
        by: dict = {}
        for row in d.get("rows", []):
            by.setdefault(row["id"], set()).add(row["sample"])
        extra = set(by) - set(ids)
        if extra:
            bad.append(f"{r['label']}: {len(extra)} ids outside the fresh set")
        for i in ids:
            counts[i] += len(by.get(i, ()))
    short = [i for i in ids if counts[i] != len(plan) * samples]
    if short:
        bad.append(f"{len(short)} fresh rows not read by every model x {samples}, e.g. {short[0]} ({counts[short[0]]})")
    return counts, bad


def run_phase(runner: str, phase: str) -> None:
    rc = subprocess.run(["bash", runner], env=dict(os.environ, PHASE=phase, FRESH_UNDER_GUARD="1")).returncode
    if rc != 0:
        raise RuntimeError(f"phase {phase} failed (rc {rc}); the guard touch is not recorded, rerun to resume")


def cmd_final(a) -> int:
    gd, data, md = Path(a.gd), Path(a.data), Path(a.md)
    guard = import_guard()
    plan = read_plan(a.plan)
    expected = len(plan) * a.samples
    records = sorted({r["record"] for r in plan})
    analyses = {rec: Path(a.res) / f"analysis_fresh_{rec}.json" for rec in records}

    def run_everything(features):
        run_phase(a.runner, "readouts")
        counts, bad = coverage(plan, data, a.samples)
        if bad:
            raise RuntimeError("incomplete readouts, touch not recorded:\n  " + "\n  ".join(bad))
        run_phase(a.runner, "analysis")
        miss = [str(p) for p in analyses.values() if not p.is_file()]
        if miss:
            raise RuntimeError(f"analysis receipts missing: {miss}")
        return [counts[i] for i in features["id"]]

    def fresh_rows_read_by_all_models(y_true, y_pred):
        return float(sum(int(v) == expected for v in y_pred) / len(y_pred))

    score = guard.final_test(run_everything, frame(data), target="answer", metric_fn=fresh_rows_read_by_all_models,
                             state_dir=gd, pipeline_md=md)
    st = json.loads((gd / guard.STATE_FILE).read_text())
    out = {"what": "Phase 5 fresh held-out set", "metric": "fresh_rows_read_by_all_models", "score": score,
           "test_touches": st["test_touches"], "models": len(plan), "samples": a.samples, "records": records,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "analysis": {k: {"file": str(p), "sha256": sha(p)} for k, p in analyses.items()},
           "readouts": {r["label"]: {"file": r["out"], "sha256": sha(r["out"]), "arm": r["arm"], "ckpt": r["ckpt"]}
                        for r in plan}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"fresh final test recorded: {out['metric']}={score:.4f}, touches {st['test_touches']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "final"):
        p = sub.add_parser(name)
        p.add_argument("--gd", required=True)
        p.add_argument("--md", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--plan", required=True)
        if name == "preflight":
            p.add_argument("--expect", required=True)
        else:
            p.add_argument("--samples", type=int, required=True)
            p.add_argument("--runner", required=True)
            p.add_argument("--res", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        bad = preflight_reasons(Path(a.gd), Path(a.md), Path(a.data), read_plan(a.plan), json.loads(Path(a.expect).read_text()))
        if bad:
            print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 2
        print("preflight: OK (pre-registration, fresh file and fingerprint, touches 0, checkpoints and sha256)")
        return 0
    return cmd_final(a)


if __name__ == "__main__":
    sys.exit(main())
