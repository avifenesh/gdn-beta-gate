#!/usr/bin/env python3
"""build_aime2025.py: the AIME 2025 test file of the 9B record (lane wg, owner direction 2026-10-15 ~23:55 Israel:
"go with generic aime"), cut from the frozen Phase 5 fresh set by source tag, plus its own guard state (touches 0).

Selection: every line of DATA/test_fresh.jsonl (sha256 9ce51dff..., the 4B record's frozen fresh file) whose "id" is
fresh:aime2025-<contest>-<n>, in file order, written byte for byte (the line is not re-serialized). Only the id field
is looked at; the problem and answer text are copied, never printed or inspected. Refuses unless the source file has
its frozen sha256, exactly 30 lines match (15 AIME I, 15 AIME II, numbers 1-15 each, as fresh_manifest.json counts
for MathArena/aime_2025 c94da77e), and every selected id is unique.

Outputs (refuses to overwrite a file whose content would change):
  DATA/test_aime2025.jsonl          the 30 rows
  DATA/aime2025_manifest.json       source file and sha256, rule, ids in order, output sha256
  --guard DIR: DIR/split_manifest.json ({"files": {"test_aime2025.jsonl": sha256}, "counts": ...}, the manifest the
              analysis and guard read) and DIR/.guard_state.json (guard._fingerprint of the frame id, problem, answer
              in file order, test touches 0; needs pandas, so run it with the workstation vLLM venv python)

Usage: build_aime2025.py --data DATA_DIR [--guard ml_pipeline_9b/aime2025_test]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
SRC = "test_fresh.jsonl"
SRC_SHA = "9ce51dff11355439f073e4e9d62e22eb6ae45f40bb8d3b4db2b9df78cf0f50fa"
OUT = "test_aime2025.jsonl"
ID = re.compile(r"^fresh:aime2025-(I|II)-([0-9]+)$")
N = 30


class Refuse(SystemExit):
    pass


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def select(src: bytes, src_sha: str = SRC_SHA) -> tuple[bytes, list[str]]:
    """The selected lines (bytes, file order) and their ids; refuses on any deviation from the rule."""
    got = sha_bytes(src)
    if got != src_sha:
        raise Refuse(f"{SRC}: sha256 {got[:16]} is not the frozen fresh file {src_sha[:16]}")
    keep, ids = [], []
    for line in src.splitlines(keepends=True):
        if not line.strip():
            continue
        i = json.loads(line)["id"]
        if ID.match(i):
            keep.append(line if line.endswith(b"\n") else line + b"\n")
            ids.append(i)
    want = {f"fresh:aime2025-{c}-{n}" for c in ("I", "II") for n in range(1, 16)}
    if len(ids) != N or len(set(ids)) != N or set(ids) != want:
        raise Refuse(f"selection: {len(ids)} lines ({len(set(ids))} unique ids); need exactly the {N} ids "
                     "fresh:aime2025-I-1..15 and -II-1..15")
    return b"".join(keep), ids


def write_frozen(p: Path, data: bytes) -> None:
    if p.exists() and p.read_bytes() != data:
        raise Refuse(f"{p} exists with different content (sha256 {sha_bytes(p.read_bytes())[:16]}); not overwritten")
    if not p.exists():
        p.write_bytes(data)


def guard_state(guard_dir: Path, out_file: Path, out_sha: str) -> dict:
    sys.path.insert(0, str(WT / "ml_pipeline"))
    import guard  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    rows = [json.loads(x) for x in out_file.read_bytes().splitlines() if x.strip()]
    frame = pd.DataFrame([{k: r[k] for k in ("id", "problem", "answer")} for r in rows])
    st = {"test_fingerprint": guard._fingerprint(frame), "n_test": len(frame), "test_touches": 0,
          "duplicates_dropped": 0, "split": "fixed_external_test (AIME 2025, cut from the Phase 5 fresh set)",
          "created": datetime.date.today().isoformat(),
          "test_frame": f"columns id, problem, answer; rows of {OUT}, file order",
          "record": "ml_pipeline_9b (Qwen3.5-9B, lane wg): AIME 2025 read once by the 9B record, touch count 0"}
    guard_dir.mkdir(parents=True, exist_ok=True)
    man = json.dumps({"files": {OUT: out_sha}, "counts": {OUT: len(frame)}}, indent=1) + "\n"
    write_frozen(guard_dir / "split_manifest.json", man.encode())
    gp = guard_dir / guard.STATE_FILE
    if gp.exists():
        old = json.loads(gp.read_text())
        if old.get("test_fingerprint") != st["test_fingerprint"] or old.get("n_test") != st["n_test"]:
            raise Refuse(f"{gp} exists with another fingerprint; not overwritten")
        return old
    gp.write_text(json.dumps(st, indent=2))
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--guard", default=None)
    a = ap.parse_args(argv)
    data = Path(a.data)
    body, ids = select((data / SRC).read_bytes())
    out_sha = sha_bytes(body)
    write_frozen(data / OUT, body)
    man = {"what": "AIME 2025 (30 problems, AIME I + II) for the 9B record of lane wg, cut by source tag from the "
                   "Phase 5 fresh set (owner direction 2026-10-15 ~23:55 Israel)",
           "source_file": str(data / SRC), "source_sha256": SRC_SHA, "rule": f"lines whose id matches {ID.pattern}, "
           "file order, bytes unchanged", "ids": ids, "file": str(data / OUT), "sha256": out_sha, "n_items": len(ids),
           "upstream": "MathArena/aime_2025 c94da77eb22bbd6439e62a323bec18493a421302 (fresh_manifest.json)",
           "script_sha256": sha_bytes(Path(__file__).read_bytes())}
    mp = data / "aime2025_manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if old.get("sha256") != out_sha or old.get("ids") != ids:
            raise Refuse(f"{mp} exists for other content; not overwritten")
    else:
        mp.write_text(json.dumps({**man, "created_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds")}, indent=1) + "\n")
    print(f"{OUT}: {len(ids)} rows, sha256 {out_sha}")
    if a.guard:
        st = guard_state(Path(a.guard), data / OUT, out_sha)
        print(f"guard state {a.guard}: fingerprint {st['test_fingerprint']}, test touches {st['test_touches']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
