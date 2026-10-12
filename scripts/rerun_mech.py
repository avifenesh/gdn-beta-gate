#!/usr/bin/env python3
"""Rerun the CPU summaries of the mechanism section from released data and compare with the released receipts.

  g1g2_summary.py   results/mech/ablate + dose (G1 ablation and G2 dose readouts, per item) -> g1g2_summary.json
  gs_summary.py     results/mech/gate_stats (G3 per-checkpoint beta statistics) -> gate_stats_summary_fixed.json
                    (decodes token ids with the Qwen3.5-4B tokenizer at the pinned revision, downloaded if absent)
  item_match.py     results/ml7_4b/step14{,_ctrlpm} readouts (4B test reads) -> item_match.json
  traj_extra.py     results/mech/traj.json -> traj_extra.json

Each script runs on CPU in a scratch directory; its JSON output must equal the released file. The GPU readouts
(ablate, dose, gate_stats), traj.py (needs the unreleased rollout logs), text_stats.py (needs the unreleased
generation traces), w_stats.py and param_drift.py (need the checkpoints, published as adapters) are not rerun here.
Usage: python scripts/rerun_mech.py      Exit 1 on any difference. Needs: numpy, matplotlib, tokenizers.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MECH = ROOT / "results/mech"
SRC = ROOT / "src/gdn_gate/mech"
QWEN4 = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def run(args, cwd, env=None):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    return r.returncode, (r.stdout + r.stderr)[-2000:]


def same(a: Path, b: Path) -> str:
    return "OK  " if json.loads(a.read_text()) == json.loads(b.read_text()) else "DIFF"


def main():
    bad = 0
    tmp = Path(tempfile.mkdtemp(prefix="rerun_mech_"))
    try:
        # G1 / G2
        m = tmp / "mech"
        for d in ("ablate", "dose"):
            shutil.copytree(MECH / d, m / d)
        rc, log = run([SRC / "g1g2_summary.py", m], ROOT)
        st = same(m / "g1g2_summary.json", MECH / "g1g2_summary.json") if rc == 0 else "FAIL"
        print(st, "results/mech/g1g2_summary.json (g1g2_summary.py)", "" if rc == 0 else log)
        bad += st != "OK  "
        # G3
        hf = os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
        from huggingface_hub import hf_hub_download

        hf_hub_download(QWEN4[0], "tokenizer.json", revision=QWEN4[1], cache_dir=hf)
        out = tmp / "gss"
        rc, log = run([SRC / "gs_summary.py", MECH / "gate_stats", out, "fixed"], ROOT, {"WG_HF_CACHE": hf, "MPLBACKEND": "Agg"})
        st = same(out / "gate_stats_summary_fixed.json", MECH / "gate_stats_summary/gate_stats_summary_fixed.json") if rc == 0 else "FAIL"
        print(st, "results/mech/gate_stats_summary/gate_stats_summary_fixed.json (gs_summary.py)", "" if rc == 0 else log)
        bad += st != "OK  "
        # item-matched decomposition from the 4B test readouts
        w = tmp / "im"
        for d in ("step14", "step14_ctrlpm"):
            (w / "results/ml7_4b" / d).mkdir(parents=True)
            for p in (ROOT / "results/ml7_4b" / d).glob("readout_*.json.gz"):
                (w / "results/ml7_4b" / d / p.name[:-3]).write_bytes(gzip.open(p).read())
        (w / "out").mkdir()
        rc, log = run([SRC / "item_match.py", w / "out"], w)
        st = same(w / "out/item_match.json", MECH / "item_match.json") if rc == 0 else "FAIL"
        print(st, "results/mech/item_match.json (item_match.py)", "" if rc == 0 else log)
        bad += st != "OK  "
        # training-dynamics windows
        t = tmp / "tj"
        t.mkdir()
        shutil.copyfile(MECH / "traj.json", t / "traj.json")
        rc, log = run([SRC / "traj_extra.py", t], ROOT)
        st = same(t / "traj_extra.json", MECH / "traj_extra.json") if rc == 0 else "FAIL"
        print(st, "results/mech/traj_extra.json (traj_extra.py)", "" if rc == 0 else log)
        bad += st != "OK  "
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
