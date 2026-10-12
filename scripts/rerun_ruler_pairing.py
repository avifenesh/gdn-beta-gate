#!/usr/bin/env python3
"""Recompute the paired RULER receipts of Section 8 from the released per-item scores and compare.

results/olmo/ruler_items/ and results/vcount/ruler_items/ hold, per arm, length and task, one line per item with its
doc_id, doc_hash (when lm-eval provided one) and score; the model outputs are not released. This script runs
src/gdn_gate/olmo/ruler_paired.py on them with the arguments of each receipt and compares every key but "root".
Usage: python scripts/rerun_ruler_pairing.py      (CPU, numpy, under a minute; exit 1 on any difference)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ("results/olmo/vllm2/ruler_paired.json", "results/olmo/ruler_items/battery_vllm2", ["--arms", "half,erase"]),
    ("results/olmo/vllm2/counting_paired.json", "results/olmo/ruler_items/battery_vllm2_cwe", ["--arms", "erase"]),
    ("results/olmo/vllm2/counting_localization.json", "results/olmo/ruler_items/battery_vllm2_loc",
     ["--arms", "erase_g0,erase_g1,erase_g2,erase_g3", "--lengths", "8192"]),
    ("results/vcount/step6_32k_paired.json", "results/vcount/ruler_items/step6_32k",
     ["--arms", "wide_s0,wide1_s0,base", "--lengths", "32768", "--base", "ctrl_s0"]),
]


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for receipt, items, args in JOBS:
            out = Path(td) / "out.json"
            r = subprocess.run([sys.executable, str(ROOT / "src/gdn_gate/olmo/ruler_paired.py"), str(ROOT / items),
                                str(out)] + args, capture_output=True, text=True)
            if r.returncode != 0:
                print("FAIL", receipt, r.stderr[-500:])
                bad += 1
                continue
            new, old = json.loads(out.read_text()), json.loads((ROOT / receipt).read_text())
            new.pop("root"), old.pop("root")
            ok = new == old
            bad += not ok
            print(f"{'OK  ' if ok else 'DIFF'} {receipt}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
