#!/usr/bin/env python3
"""Code-lock amendment ledger for the readout instrument (stdlib only; runs on the workstation and the box with /usr/bin/python3).

A reuse check (step12_val_box.sh / step14_test_box.sh with STEP12_REUSE / STEP14_REUSE) adopts readouts of a finished
set only when the instrument code equals that set's code lock. An owner-approved amendment changes a locked file
without changing what the reused readouts measured; it is accepted only through the ledger amendments/ledger.json:

  {"schema": "wg-code-amendments-v1", "amendments": [
     {"file": "readout_vllm.py", "old": <sha256 before>, "new": <sha256 after>, "diff": "<file in amendments/>",
      "diff_sha256": <sha256 of the diff file>, "approval": "<owner approval: when, where, quote>", "what": "..."}]}

`accept --file PATH --want SHA` exits 0 when sha256(PATH) == SHA, or when the current file can be walked back to SHA
through ledger entries for that file name: at each link the entry's diff file must match its diff_sha256, the
current content must hash to the entry's `new`, and reverse-applying the diff (normal `diff OLD NEW` format, every
context line checked) must give content that hashes to the entry's `old`. Anything else (no entry, a changed diff, a
diff that does not reproduce the old file, an extra edit beside a documented one) exits 1 with the reason: an
undocumented change still refuses.

Published copy: the released files were path-scrubbed, so their sha256 differ from the lab's. A hash in the ledger
(lab file) also matches when results/SCRUB_LEDGER.json maps it to the sha256 of the published content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "amendments" / "ledger.json"
SCHEMA = "wg-code-amendments-v1"
SCRUB_LEDGER = HERE.parents[2] / "results" / "SCRUB_LEDGER.json"


def _published() -> dict:
    """lab sha256 -> published sha256, from results/SCRUB_LEDGER.json (empty when absent)."""
    try:
        return {f["sha256_original"]: f["sha256_released"] for f in json.loads(SCRUB_LEDGER.read_text())["files"]
                if f.get("sha256_original")}
    except (OSError, ValueError, KeyError):
        return {}


PUB = _published()


def same(content_sha: str, lab_sha: str) -> bool:
    return content_sha == lab_sha or PUB.get(lab_sha) == content_sha


HUNK = re.compile(r"^(\d+)(?:,(\d+))?([acd])(\d+)(?:,(\d+))?$")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_normal_diff(text: str) -> list[dict]:
    """Hunks of a normal-format diff: old range, new range, old lines (<), new lines (>)."""
    hunks, cur = [], None
    for ln in text.splitlines(keepends=True):
        m = HUNK.match(ln.rstrip("\n"))
        if m:
            o1, o2, op, n1, n2 = m.groups()
            o1, n1 = int(o1), int(n1)
            cur = {"op": op, "o1": o1, "o2": int(o2) if o2 else o1, "n1": n1, "n2": int(n2) if n2 else n1,
                   "old": [], "new": []}
            hunks.append(cur)
        elif cur is not None and ln.startswith("< "):
            cur["old"].append(ln[2:])
        elif cur is not None and ln.startswith("> "):
            cur["new"].append(ln[2:])
        elif ln.rstrip("\n") == "---" or ln.startswith("\\ No newline"):
            continue
        else:
            raise ValueError(f"not a normal-format diff line: {ln!r}")
    for h in hunks:
        no = 0 if h["op"] == "a" else h["o2"] - h["o1"] + 1
        nn = 0 if h["op"] == "d" else h["n2"] - h["n1"] + 1
        if len(h["old"]) != no or len(h["new"]) != nn:
            raise ValueError(f"hunk {h['op']} at old {h['o1']} new {h['n1']}: line counts do not match its header")
    return hunks


def reverse_apply(new_text: str, diff_text: str) -> str:
    """The OLD file from the NEW file and `diff OLD NEW`; every '>' line must be at its place in NEW."""
    lines = new_text.splitlines(keepends=True)
    out, pos = [], 0
    for h in parse_normal_diff(diff_text):
        start = h["n1"] if h["op"] == "d" else h["n1"] - 1  # 'd': N = the line after which the deleted lines were
        if start < pos:
            raise ValueError("diff hunks out of order")
        out.extend(lines[pos:start])
        got = lines[start:start + len(h["new"])]
        if got != h["new"]:
            raise ValueError(f"diff does not match the current file at line {start + 1}")
        out.extend(h["old"])
        pos = start + len(h["new"])
    out.extend(lines[pos:])
    return "".join(out)


def load_ledger(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r}, not {SCHEMA}")
    for e in d["amendments"]:
        for k in ("file", "old", "new", "diff", "diff_sha256", "approval"):
            if not str(e.get(k, "")).strip():
                raise ValueError(f"{path}: an amendment of {e.get('file')} has no {k}")
    return d["amendments"]


def accept(path: Path, want: str, ledger: Path = LEDGER) -> tuple[bool, str]:
    data = path.read_bytes()
    cur = sha_bytes(data)
    if same(cur, want):
        return True, "equal"
    try:
        entries = [e for e in load_ledger(ledger) if e["file"] == path.name]
    except (OSError, ValueError, KeyError) as e:
        return False, f"{path.name} sha256 {cur[:16]} differs from the lock's {want[:16]} and the ledger is unreadable: {e}"
    text, chain = data.decode(), []
    for _ in range(len(entries) + 1):
        if same(cur, want):
            return True, "documented amendments: " + " <- ".join(chain)
        e = next((x for x in entries if same(cur, x["new"])), None)
        if e is None:
            return False, (f"{path.name} sha256 {cur[:16]} differs from the lock's {want[:16]} with no ledger amendment "
                           f"ending at {cur[:16]} (undocumented change)")
        dp = ledger.parent / e["diff"]
        try:
            diff = dp.read_bytes()
        except OSError:
            return False, f"{path.name}: amendment diff {dp} missing"
        if not same(sha_bytes(diff), e["diff_sha256"]):
            return False, f"{path.name}: amendment diff {dp.name} sha256 {sha_bytes(diff)[:16]} != ledger {e['diff_sha256'][:16]}"
        try:
            text = reverse_apply(text, diff.decode())
        except ValueError as x:
            return False, f"{path.name}: amendment {dp.name} does not reproduce the change: {x}"
        cur = sha_bytes(text.encode())
        if not same(cur, e["old"]):
            return False, (f"{path.name}: reverse-applying {dp.name} gives sha256 {cur[:16]}, not the ledger's old "
                           f"{e['old'][:16]} (an edit beside the documented one)")
        chain.append(f"{e['new'][:12]}<-{e['old'][:12]} ({dp.name})")
    return False, f"{path.name}: ledger chain does not reach {want[:16]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accept")
    a.add_argument("--file", required=True)
    a.add_argument("--want", required=True)
    a.add_argument("--ledger", default=str(LEDGER))
    x = ap.parse_args(argv)
    ok, why = accept(Path(x.file), x.want, Path(x.ledger))
    print(why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
