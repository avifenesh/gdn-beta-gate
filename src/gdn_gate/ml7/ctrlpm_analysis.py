#!/usr/bin/env python3
"""ctrlpm_analysis.py: the frozen step 7 analysis (step14_analysis.py, imported, never edited) with a fourth arm,
ctrlpm, the parameter-matched control of the Qwen3.5-4B record (owner approval 2026-10-11 ~23:55 Israel; design and
pre-registration in ml_pipeline_4b/PIPELINE.md, "Extension: parameter-matched control ctrlpm").

ctrlpm = the ctrl arm (no gate, library forward) with LoRA rank 20 instead of 16 on the same GDN projections, so its
trainable parameter count (10,844,160) matches wide1's (LoRA r16 8,675,328 + gate W 1,966,080 = 10,641,408) within
+1.9%. Its readouts are readout_vllm.py --arm ctrl --rank 20 files; this wrapper checks that (file arm ctrl, args.rank
20, checkpoint path in a ctrlpm_ run dir) and then labels them arm ctrlpm in memory, so the frozen validate() and
analyze() read four arms with the same rule, the same bootstrap draws (one index matrix) and the same blocks.

What changes against the frozen script, all by setting its module constants in this process:
  ARMS        + ctrlpm
  CONTRASTS   + wide1 - ctrlpm, wide - ctrlpm, ctrlpm - ctrl (each arm - warm-up / - untouched includes ctrlpm)
  LABEL       accepts ctrlpm_SEED
  HYPOTHESES  + H3 (MATH-500 test: wide1 - ctrlpm area > 0) and H4 (GSM8K test: wide1 - ctrlpm natural length < 0),
              named before any ctrlpm data; H1 and H2 are evaluated as before (their contrasts do not involve ctrlpm)
Self-check (always run, the receipt records it): analyze() of the three original arms alone, with the frozen constants
restored, must equal the four-arm result on every key the three-arm result has (effects, length_effects, arms,
hypotheses H1/H2): adding an arm changes nothing already computed. With --touch1 RECEIPT (the frozen script's own
analysis of the same set, test touch 1) the original contrasts must also equal that receipt bit for bit, which proves
the reused untouched, warm-up, ctrl, wide and wide1 readouts are the touch-1 files and the instrument is unchanged.

Usage: ctrlpm_analysis.py --split test|val --baseline B.json --warmup W.json ctrl_0=P ... wide1_2=P ctrlpm_0=P ...
                          --out OUT.json [--manifest M] [--seeds 0,1,2] [--pm_rank 20] [--touch1 ANALYSIS.json]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step14_analysis as S  # noqa: E402  (frozen; sha256 0d1f341b... at the 4B touch-1 lock)

PM = "ctrlpm"
PM_RANK = 20
FROZEN = {"ARMS": S.ARMS, "CONTRASTS": S.CONTRASTS, "LABEL": S.LABEL, "HYPOTHESES": dict(S.HYPOTHESES)}
EXT = {
    "ARMS": S.ARMS + (PM,),
    "CONTRASTS": S.CONTRASTS + (("wide1", PM), ("wide", PM), (PM, "ctrl")),
    "LABEL": re.compile(r"^(ctrlpm|ctrl|wide1|wide)_([0-9]+)(?:_step([0-9]+))?$"),
    "HYPOTHESES": {
        **S.HYPOTHESES,
        "H3": {"set": "math500", "metric": "norm_auc_log2", "block": "effects", "contrast": "wide1 - ctrlpm",
               "direction": "positive", "text": "MATH-500 test: wide1 - ctrlpm area > 0 under the decision rule "
               "(the gate beats the same number of extra LoRA parameters)"},
        "H4": {"set": "gsm8k_test", "metric": "natural_len", "block": "length_effects", "contrast": "wide1 - ctrlpm",
               "direction": "negative", "text": "GSM8K test: wide1 reasons less than ctrlpm at the largest budget "
               "(natural length, wide1 - ctrlpm < 0) under the decision rule"},
    },
}


def use(consts: dict) -> None:
    for k, v in consts.items():
        setattr(S, k, v)


def pm_reasons(slots: dict, pm_rank: int) -> list[str]:
    """ctrlpm readouts must be ctrl-engine readouts of a rank-pm_rank checkpoint from a ctrlpm run; the other arms'
    readouts must be rank 16 (the record's)."""
    bad = []
    for (a, s), d in sorted(slots.items()):
        rank = (d.get("args") or {}).get("rank")
        if a == PM:
            if d.get("arm") != "ctrl":
                bad.append(f"{a}_{s}: file arm {d.get('arm')!r}, a ctrlpm readout is readout_vllm.py --arm ctrl")
            if rank != pm_rank:
                bad.append(f"{a}_{s}: args.rank {rank!r}, ctrlpm needs {pm_rank}")
            if f"/{PM}_s{s}_" not in str(d.get("ckpt")):
                bad.append(f"{a}_{s}: checkpoint {d.get('ckpt')!r} is not in a {PM}_s{s}_ run dir")
        elif rank != 16:
            bad.append(f"{a}_{s}: args.rank {rank!r}, the record's arms are rank 16")
    return bad


def relabel(slots: dict) -> None:
    for (a, s), d in slots.items():
        if a == PM:
            d["_file_arm"] = d["arm"]
            d["arm"] = PM


def same(x, y, path="") -> list[str]:
    """Paths where y differs from x, for keys of x (y may have more keys)."""
    if isinstance(x, dict):
        if not isinstance(y, dict):
            return [path or "/"]
        out = []
        for k, v in x.items():
            if k not in y:
                out.append(f"{path}/{k} missing")
            else:
                out += same(v, y[k], f"{path}/{k}")
        return out
    if isinstance(x, (list, tuple)):
        if not isinstance(y, (list, tuple)) or len(x) != len(y):
            return [path]
        out = []
        for j, (u, v) in enumerate(zip(x, y)):
            out += same(u, v, f"{path}[{j}]")
        return out
    return [] if x == y or (x != x and y != y) else [path]  # NaN == NaN here


def run(a) -> dict:
    S.set_seeds(a.seeds.split(","))
    use(EXT)
    base = S.load(a.baseline)
    warm = S.load(a.warmup) if a.warmup else None
    slots, diag = S.parse_slots(a.slots)
    if diag:
        raise S.Refuse("ctrlpm_analysis: diagnostic step readouts are not part of this analysis")
    bad = pm_reasons(slots, a.pm_rank)
    relabel(slots)
    bad += S.validate(base, slots, {}, warm, a.split, False, a.manifest)
    if bad:
        print("REFUSED, inputs differ:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise S.Refuse(2)
    res = S.analyze(base, slots, {}, None, a.manifest, warm, a.split)
    # self-check: the three original arms alone, frozen constants, must equal the four-arm result on their keys
    use(FROZEN)
    three = {k: v for k, v in slots.items() if k[0] != PM}
    res3 = S.analyze(base, three, {}, None, a.manifest, warm, a.split)
    use(EXT)
    keys = ("effects", "length_effects", "arms", "acc_diffs_descriptive", "baseline", "warmup")
    diffs = [p for k in keys for p in same(res3[k], res[k], k)]
    diffs += [p for h in ("H1", "H2") for p in same(res3["preregistered_hypotheses"][h],
                                                     res["preregistered_hypotheses"][h], f"hyp/{h}")]
    t1 = None
    if a.touch1:
        ref = json.loads(Path(a.touch1).read_text())
        td = [p for k in ("effects", "length_effects", "arms") for p in same(ref[k], res3[k], k)]
        t1 = {"file": a.touch1, "sha256": S._sha_file(a.touch1), "equal_on_original_arms": not td, "diffs": td[:20]}
        if td:
            print("REFUSED: the original arms do not reproduce the touch-1 analysis:\n  " + "\n  ".join(td[:20]),
                  file=sys.stderr)
            raise S.Refuse(3)
    if diffs:
        print("REFUSED: adding ctrlpm changed results of the original arms:\n  " + "\n  ".join(diffs[:20]),
              file=sys.stderr)
        raise S.Refuse(3)
    res["ctrlpm"] = {"arm": PM, "engine_arm": "ctrl", "lora_rank": a.pm_rank,
                     "trainable_params": {"ctrl": 8_675_328, "wide1": 8_675_328 + 1_966_080, PM: 10_844_160},
                     "added_contrasts": [f"{x} - {y}" for x, y in EXT["CONTRASTS"][len(FROZEN["CONTRASTS"]):]],
                     "self_check_three_arms_equal": True, "touch1_check": t1}
    res["receipt"] = {
        "argv": sys.argv, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": S._sha_file(__file__), "frozen_analysis_sha256": S._sha_file(S.__file__),
        "python": platform.python_version(), "numpy": S.np.__version__, "split": a.split,
        "inputs": {"baseline": {"file": base["_path"], "sha256": base["_sha256"]},
                   **({"warmup": {"file": warm["_path"], "sha256": warm["_sha256"],
                                  "ckpt_sha256": warm["env"]["sha256"].get("ckpt")}} if warm is not None else {}),
                   **{f"{x}_{s}": {"file": d["_path"], "sha256": d["_sha256"], "file_arm": d.get("_file_arm", d["arm"]),
                                   "rank": d["args"].get("rank"), "ckpt": d.get("ckpt"),
                                   "ckpt_sha256": d["env"]["sha256"].get("ckpt")} for (x, s), d in sorted(slots.items())}},
    }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(S.SPLITS))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--warmup", default=None)
    ap.add_argument("slots", nargs="+", help="ARM_SEED=path for ctrl/wide/wide1/ctrlpm x the seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pm_rank", type=int, default=PM_RANK)
    ap.add_argument("--touch1", default=None, help="the frozen script's analysis of the same set (touch 1)")
    a = ap.parse_args(argv)
    res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    S.print_table(res)
    print(f"\n{res['split_label']} receipt (with ctrlpm): {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
