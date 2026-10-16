#!/usr/bin/env python3
"""Rerun the frozen step 12 / step 14 analyses on the released readouts and compare with the released receipts.

For each analysis receipt (results/**/analysis_*.json) this script takes the command line recorded in its
`receipt.argv` (`receipt_fresh.argv` for the fresh held-out set, whose analysis fresh_analysis.py imports the frozen
step 14 analyses; the 9B AIME 2025 receipt names test9b/test9b_analysis.py, which imports both), decompresses the readouts it names (readout_*.json.gz, into a scratch directory), reruns the same
analysis script from src/gdn_gate/ml7/ on CPU, and compares the new output with the released file key by key. The
`receipt` block (timestamps, hashes, paths) is excluded; every other key must be equal, except that a sha256 of an
input file may differ when results/SCRUB_LEDGER.json maps the released value (the lab file) to the new value (the
published, path-scrubbed file). Exit 1 on any other difference.

Usage: python scripts/rerun_analysis.py [RECORD ...]
  RECORD defaults to the four test reads, the fresh held-out set, the 9B AIME 2025 read and the validation reads,
  e.g. results/fresh
Needs: numpy (CPU only, a few minutes in total).
"""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["results/ml7/step14", "results/ml7/step14_ext", "results/ml7_4b/step14", "results/ml7_4b/step14_ctrlpm",
           "results/fresh", "results/ml7_9b/aime2025",
           "results/ml7/step12", "results/ml7/step12_box", "results/ml7/step12_box_ext", "results/ml7_4b/step12_box",
           "results/ml7_4b/step12_ctrlpm"]


# Keys step14_analysis.py gained on 2026-10-11, before any 4B or seeds 3-5 data; the 2B read-1 and early validation
# receipts were written by the frozen version without them (see the script's docstring).
ADDED_LATER = {"/length_effects: only in new", "/preregistered_hypotheses: only in new"}
LEDGER = {f["sha256_original"]: f["sha256_released"]
          for f in json.loads((ROOT / "results/SCRUB_LEDGER.json").read_text())["files"]}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if path == "" and k in ("receipt", "receipt_fresh"):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in b else 'released'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}/{i}")]
    if a == b:
        return []
    if path.endswith("sha256") and LEDGER.get(a) == b:
        return []  # hash of an input file that was path-scrubbed for publication; the ledger maps old to new
    return [f"{path}: released {a!r} new {b!r}"]


def rerun(receipt: Path, scratch: Path) -> list[str]:
    old = json.loads(receipt.read_text())
    argv = (old.get("receipt_fresh") or old["receipt"])["argv"]
    script = ROOT / argv[0].replace("<repo>/", "")
    out = scratch / receipt.name
    args = []
    skip = False
    for i, a in enumerate(argv[1:]):
        if skip:
            skip = False
            continue
        if a == "--out":
            args += ["--out", str(out)]
            skip = True
            continue
        key, eq, val = a.partition("=")
        target = val if eq else a
        if target.startswith("<repo>/"):
            rel = target.replace("<repo>/", "")
            src = ROOT / rel
            if not src.exists() and (ROOT / (rel + ".gz")).exists():
                dst = scratch / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(gzip.open(ROOT / (rel + ".gz")).read())
                src = dst
            target = str(src)
        args.append(f"{key}={target}" if eq else target)
    r = subprocess.run([sys.executable, str(script)] + args, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return [f"exit {r.returncode}: {r.stderr.strip()[-2000:]}"]
    new = json.loads(out.read_text().replace(str(scratch) + "/", "<repo>/").replace(str(ROOT) + "/", "<repo>/"))
    return diff(old, new)


def main():
    records = sys.argv[1:] or DEFAULT
    bad = 0
    scratch = Path(tempfile.mkdtemp(prefix="rerun_analysis_"))
    try:
        for rec in records:
            for receipt in sorted((ROOT / rec).glob("analysis_*.json")):
                d = rerun(receipt, scratch)
                rel = receipt.relative_to(ROOT)
                note = ""
                if d and set(d) <= ADDED_LATER:
                    d, note = [], " (released file predates the keys length_effects and preregistered_hypotheses; every released key is equal)"
                print(f"{'OK  ' if not d else 'DIFF'} {rel}" + (note if not d else f" ({len(d)} differences)"))
                for line in d[:10]:
                    print("     ", line)
                bad += bool(d)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
