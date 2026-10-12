#!/usr/bin/env python3
"""Arm analysis for step 7, length-cost GRPO: step 14 (final test, touched once, --split test, run once per
test set) and step 12 (model evaluation on the validation slice, --split val). One script, one rule, so the
validation readout and the final test are read the same way; every output is labeled with its split.

Contract: ml_pipeline/PIPELINE.md, "Step 3: problem contract" (metric and decision rule), "Gate B preparation" (metric
line, answer cap rule), the design note of 2026-10-09 00:05 (correctness-conditioned length report) and the design
change of 2026-10-09 10:00 (shared format warm-up; arms initialized from it; its readout is the baseline the arms are
compared against).

Inputs: the untouched-model readout (--baseline, role untouched, arm stock), the shared format warm-up readout
(--warmup, role warmup: arm ctrl with a checkpoint, ctrl_s0_cap512_fmt_gb10 ckpt_step40), and the readout of the final
checkpoint for every arm (ctrl, wide, wide1) and seed (0, 1, 2), all readout_vllm.py JSON (schema wg-readout-v1)
on the same data. Optional: readouts of the last checkpoints of a run (ARM_SEED_stepN=path) for the checkpoint-noise
diagnostic only.

Metrics:
* Primary: normalized area under accuracy vs log2 budget (caps 256..8192), per arm = mean over the 3 seeds, per-seed
  values listed. Per item the value is the mean over its samples of the per-sample area (the area is linear in the
  accuracies, so the mean of these equals the area of the mean curve).
* Arm effect rule, pre-registered: an effect counts only if all 3 per-seed differences have the same nonzero sign AND
  the paired item bootstrap 95% interval of the seed-mean difference excludes 0. Contrasts: arm vs arm (wide - ctrl,
  wide1 - ctrl, wide - wide1; seed s of arm A minus seed s of arm B), each arm - warm-up and each arm - untouched
  (every seed paired with the one baseline readout). Bootstrap: resample items with replacement, 10,000 draws, numpy
  default_rng(seed 20261009), percentile interval. One index matrix is shared by every contrast. warm-up - untouched is
  reported descriptively (one readout per side, so the 3-seed rule does not apply).
* Secondary: accuracy at each cap; tokens to reach the untouched model's 8k accuracy (log2-interpolated,
  readout_vllm.tokens_at_acc, None = never reached); natural length (mean reasoning tokens at the largest cap) and
  cut-off share at the largest cap; checks beside them (silent stops, missing \\boxed{}, answer not ended).
* Correctness-conditioned length: at each cap (the largest cap = natural length), mean and median reasoning tokens
  among correct samples and among incorrect samples; per arm the seed-mean of the per-seed statistic.
* Answer tokens per budget: from the rows when they carry them (a 6th element of each per-cap list, or a row field
  "answer_tokens" {cap: n}); else, with --tokenizer, by re-tokenizing the answer texts of the OUT.traces.jsonl.gz
  sidecar (as answer_cap.py does; the decoded text has special tokens stripped, so the stop token is not counted);
  else reported as unavailable.
* Natural length under the rule (length_effects): per item the mean over its samples of the reasoning tokens at the
  largest cap (min(think_len, largest cap), the per-row quantity of readout_vllm.summarize's natural length, so the
  item mean of an arm equals its natural_len), every contrast of the area rule (arm vs arm, arm - warm-up, arm -
  untouched), the same decision rule, the same bootstrap draws and the same index matrix. A negative difference = the
  first arm reasons less.
* Pre-registered hypotheses (preregistered_hypotheses), named in ml_pipeline_4b/PIPELINE.md from the 2B record's
  touch 1 before any 4B or seeds 3-5 data: H1, MATH-500 test, wide - ctrl area > 0; H2, GSM8K test, wide - ctrl
  natural length < 0 (wide reasons less); each needs the rule (all seeds of the set agree in sign, interval excludes 0)
  in the named direction. Evaluated only on the test split and on the hypothesis's own set; otherwise listed as not
  applicable.
* Checkpoint noise (diagnostic, never used for selection): per (arm, seed) the area of each extra checkpoint readout
  and of the final one, and their spread (max - min).
* Test split, MATH-500 only: the primary metric and effect rule also on the 450 items the pilot never read
  (ml_pipeline/split_manifest.json pilot50_math500_ids), as a check.

Splits: --split test needs every arm and seed and data whose sha256 is a test file of the split manifest; --split val
needs the validation slice (val.jsonl sha256 in the manifest, no test ids). --allow_partial (validation only) accepts
missing arm runs while the 9 runs are still training: every contrast then uses the seeds present, and a contrast with
fewer than 3 seeds is reported as incomplete with no verdict; the receipt says partial and lists the runs present.

Refuses to run (exit 2, printing every difference) if any file is not schema wg-readout-v1, is a red-arm readout,
caps differ, item sets or per-item sample counts differ, readout seed, sample count, answer cap, sampling, GPU or data
sha256 differ from the baseline, the baseline is not arm stock, the warm-up is not arm ctrl with a checkpoint, a file's
arm field does not match its label, an arm or seed is missing (test split, or validation without --allow_partial), two
slots (warm-up included) carry the same checkpoint sha256, or the data does not belong to the declared split.

--seeds (default 0,1,2, the step 7 record): the seed set the rule needs. With --seeds 3,4,5 the same rule reads a
pre-registered extension set (all of its seeds must agree in sign); with --seeds 0,1,2,3,4,5 it reads the pooled set.
The default reproduces every key of the frozen step 14 analysis (touch 1) exactly apart from the receipt block; the
keys length_effects and preregistered_hypotheses were added on 2026-10-11 (before any 4B or seeds 3-5 data) and are
new.

Usage: step14_analysis.py --split test|val --baseline BASE.json [--warmup WARM.json] ctrl_0=P ... wide1_2=P
                          [ctrl_0_step140=P ...] --out RECEIPT.json [--allow_partial] [--tokenizer DIR]
                          [--manifest split_manifest.json] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import platform
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from readout_vllm import SCHEMA, norm_auc, summarize, tokens_at_acc  # noqa: E402

ARMS = ("ctrl", "wide", "wide1")
SEEDS = (0, 1, 2)
CONTRASTS = (("wide", "ctrl"), ("wide1", "ctrl"), ("wide", "wide1"))
BOOT_DRAWS = 10_000
BOOT_SEED = 20261009
DEFAULT_MANIFEST = HERE.parents[2] / "ml_pipeline" / "split_manifest.json"
LABEL = re.compile(r"^(ctrl|wide1|wide)_([0-9]+)(?:_step([0-9]+))?$")
SPLITS = {"val": "validation", "test": "test"}
STEP_OF_SPLIT = {"val": 12, "test": 14}
VAL_FILE = "val.jsonl"
TEST_FILES = ("test_gsm8k.jsonl", "test_math500.jsonl")
TEST_ID_PREFIXES = ("gsm8k_test:", "math500:")
WARM_ARM = "ctrl"
DESCRIPTIVE = "descriptive only: one readout per side, the 3-seed rule does not apply"
# pre-registered confirmatory hypotheses (ml_pipeline_4b/PIPELINE.md step 3, from the 2B record's touch 1, 2026-10-11)
HYPOTHESES = {
    "H1": {"set": "math500", "metric": "norm_auc_log2", "block": "effects", "contrast": "wide - ctrl",
           "direction": "positive", "text": "MATH-500 test: wide - ctrl area > 0 under the decision rule"},
    "H2": {"set": "gsm8k_test", "metric": "natural_len", "block": "length_effects", "contrast": "wide - ctrl",
           "direction": "negative", "text": "GSM8K test: wide reasons less than ctrl at the largest budget (natural "
           "length, wide - ctrl < 0) under the decision rule"},
}


def set_seeds(seeds) -> None:
    """The seed set every check, contrast and text uses (--seeds). The default (0, 1, 2) leaves every output as the
    frozen step 14 analysis wrote it."""
    global SEEDS, DESCRIPTIVE
    seeds = tuple(int(x) for x in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        raise Refuse(f"--seeds {seeds}: need distinct non-negative seeds")
    SEEDS = tuple(sorted(seeds))
    DESCRIPTIVE = f"descriptive only: one readout per side, the {len(SEEDS)}-seed rule does not apply"


class Refuse(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ loading and checks
def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    d["_path"] = str(path)
    d["_sha256"] = _sha_file(path)
    return d


def item_samples(d: dict) -> dict:
    by: dict = {}
    for r in d["rows"]:
        by.setdefault(r["id"], set()).add(r["sample"])
    return by


def _has_ckpt(d: dict) -> bool:
    return d.get("ckpt") not in (None, "", "none") and bool(d["env"]["sha256"].get("ckpt"))


def validate(base: dict, slots: dict, diag: dict, warm: dict | None = None, split: str = "test",
             allow_partial: bool = False, manifest: str | Path | None = None) -> list[str]:
    """Every difference that makes the comparison invalid; empty list = OK. slots: {(arm, seed): d}; diag:
    {(arm, seed, step): d}; warm: the warm-up readout (role warmup) or None."""
    bad = []
    if split not in SPLITS:
        return [f"split {split!r}: need one of {sorted(SPLITS)}"]
    if allow_partial and split != "val":
        bad.append("--allow_partial is for the validation split only: the test split needs every arm and seed")
    files = [("baseline", base)] + ([("warmup", warm)] if warm is not None else []) + \
            [(f"{a}_{s}", d) for (a, s), d in sorted(slots.items())] + \
            [(f"{a}_{s}_step{t}", d) for (a, s, t), d in sorted(diag.items())]
    for name, d in files:
        if d.get("schema") != SCHEMA:
            bad.append(f"{name}: schema {d.get('schema')!r}, need {SCHEMA!r} ({d['_path']})")
        if d.get("red"):
            bad.append(f"{name}: red-arm readout (red={d['red']!r}), never a result ({d['_path']})")
    if any("schema" in b for b in bad):
        return bad
    if base.get("arm") != "stock":
        bad.append(f"baseline: arm {base.get('arm')!r}, need 'stock' (the untouched model)")
    if warm is not None and (warm.get("arm") != WARM_ARM or not _has_ckpt(warm)):
        bad.append(f"warmup: arm {warm.get('arm')!r}, ckpt {warm.get('ckpt')!r}: the warm-up role needs arm "
                   f"{WARM_ARM!r} with a checkpoint (the shared format warm-up, {warm['_path']})")
    for (a, s) in [(a, s) for a in ARMS for s in SEEDS]:
        if (a, s) not in slots and not allow_partial:
            bad.append(f"{a}_{s}: missing (need every arm in {ARMS} x seed in {SEEDS})")
    extra = sorted(set(slots) - {(a, s) for a in ARMS for s in SEEDS})
    for a, s in extra:
        bad.append(f"{a}_{s}: seed {s} outside {SEEDS}")
    for (a, s, t) in sorted(diag):
        if (a, s) not in slots:
            bad.append(f"{a}_{s}_step{t}: diagnostic readout without the final readout {a}_{s}")
    for name, d in files:
        if name in ("baseline", "warmup"):
            continue
        arm = name.split("_")[0]
        if d.get("arm") != arm:
            bad.append(f"{name}: file arm field {d.get('arm')!r} does not match the label {arm!r} ({d['_path']})")
    ref_caps = base["caps"]
    ref_items = item_samples(base)

    def key_of(d):
        return {"readout seed": d["args"]["seed"], "samples per item": d["args"]["samples"],
                "answer_max": d["args"].get("answer_max"), "sampling": d.get("sampling"),
                "data sha256": d["env"]["sha256"].get("data"), "GPU": d["env"].get("gpu")}
    ref = key_of(base)
    for name, d in files[1:]:
        if d["caps"] != ref_caps:
            bad.append(f"{name}: caps {d['caps']} differ from baseline {ref_caps}")
        for k, v in key_of(d).items():
            if v != ref[k]:
                bad.append(f"{name}: {k} {v!r} differs from baseline {ref[k]!r}")
        its = item_samples(d)
        if set(its) != set(ref_items):
            only_f = sorted(set(its) - set(ref_items))
            only_b = sorted(set(ref_items) - set(its))
            bad.append(f"{name}: items differ from baseline: {len(only_f)} only in file {only_f[:3]}, "
                       f"{len(only_b)} only in baseline {only_b[:3]}")
        else:
            diff = [i for i in its if its[i] != ref_items[i]]
            if diff:
                bad.append(f"{name}: per-item samples differ from baseline on {len(diff)} items, e.g. {diff[0]}: "
                           f"{sorted(its[diff[0]])} vs {sorted(ref_items[diff[0]])}")
    n_samp = {len(v) for v in ref_items.values()}
    if len(n_samp) != 1:
        bad.append(f"baseline: unequal samples per item {sorted(n_samp)}")
    ck: dict = {}
    for name, d in [("warmup", warm)] * (warm is not None) + [(f"{a}_{s}", d) for (a, s), d in slots.items()]:
        h = d["env"]["sha256"].get("ckpt")
        if h:
            ck.setdefault(h, []).append(name)
    for h, who in ck.items():
        if len(who) > 1:
            bad.append(f"{' and '.join(sorted(who))}: same checkpoint sha256 {h[:12]}")
    # the data belongs to the declared split (split manifest sha256; validation also refuses test ids)
    mp = Path(manifest) if manifest else DEFAULT_MANIFEST
    try:
        man = json.loads(mp.read_text())["files"]
    except (OSError, ValueError, KeyError) as e:
        return bad + [f"split manifest unreadable ({mp}): {type(e).__name__} {e}"]
    dsha = str(base["env"]["sha256"].get("data"))
    if split == "val":
        want = man.get(VAL_FILE)
        if dsha != want:
            bad.append(f"baseline: data sha256 {dsha[:16]} is not the validation slice ({VAL_FILE} "
                       f"{str(want)[:16]} in {mp}); --split val reads validation readouts only")
        tid = sorted(i for i in ref_items if i.startswith(TEST_ID_PREFIXES))
        if tid:
            bad.append(f"baseline: {len(tid)} test ids in a validation analysis, e.g. {tid[0]}")
    else:
        want = {f: man.get(f) for f in TEST_FILES}
        if dsha not in want.values():
            what = " (it is the validation slice: run with --split val)" if dsha == man.get(VAL_FILE) else ""
            bad.append(f"baseline: data sha256 {dsha[:16]} is not a test file of {mp} "
                       f"({', '.join(f'{f} {str(h)[:16]}' for f, h in want.items())}){what}")
    return bad


# ------------------------------------------------------------------------------------------------ per-item matrices
def item_auc(d: dict, ids: list[str]) -> np.ndarray:
    caps = d["caps"]
    by: dict = {}
    for r in d["rows"]:
        by.setdefault(r["id"], []).append(norm_auc(caps, [r["caps"][str(c)][0] for c in caps]))
    return np.array([sum(by[i]) / len(by[i]) for i in ids])


def item_acc(d: dict, ids: list[str], c: int) -> np.ndarray:
    by: dict = {}
    for r in d["rows"]:
        by.setdefault(r["id"], []).append(r["caps"][str(c)][0])
    return np.array([sum(by[i]) / len(by[i]) for i in ids])


def item_natural(d: dict, ids: list[str]) -> np.ndarray:
    """Per item, the mean over its samples of the reasoning tokens at the largest cap (readout_vllm.summarize's
    natural length row quantity, min(think_len, largest cap))."""
    big = max(d["caps"])
    by: dict = {}
    for r in d["rows"]:
        by.setdefault(r["id"], []).append(min(r["think_len"], big))
    return np.array([sum(by[i]) / len(by[i]) for i in ids], dtype=float)


def boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_DRAWS, n))


def boot_ci(diff_items: np.ndarray, idx: np.ndarray) -> list[float]:
    """Percentile 95% interval of the mean of diff_items under item resampling (rows of idx = draws)."""
    m = np.empty(idx.shape[0])
    for k in range(0, idx.shape[0], 1000):
        m[k:k + 1000] = diff_items[idx[k:k + 1000]].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return [float(lo), float(hi)]


def effect(per_seed_items_a: list[np.ndarray], per_seed_items_b: list[np.ndarray], idx: np.ndarray,
           seeds: list | None = None, need: int | None = None) -> dict:
    """Decision rule on per-item values: seeds paired by index (b may be the one baseline repeated). With fewer than
    `need` paired seeds (validation while runs are still training) the contrast is incomplete: no verdict."""
    need = len(SEEDS) if need is None else need
    d_items = [a - b for a, b in zip(per_seed_items_a, per_seed_items_b)]
    per_seed = [float(x.mean()) for x in d_items]
    mean_items = np.mean(d_items, axis=0)
    ci = boot_ci(mean_items, idx)
    agree = all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed)
    excl = ci[0] > 0 or ci[1] < 0
    complete = len(per_seed) >= need
    eff = complete and agree and excl
    if eff:
        verdict = "effect, " + ("positive" if per_seed[0] > 0 else "negative")
    elif complete:
        verdict = "no effect"
    else:
        verdict = f"incomplete ({len(per_seed)} of {need} seeds), no verdict"
    return {"per_seed_diff": per_seed, "seeds": list(seeds) if seeds is not None else None, "n_seeds": len(per_seed),
            "seed_mean_diff": float(mean_items.mean()), "ci95": ci, "signs_agree": bool(agree),
            "ci_excludes_0": bool(excl), "complete": bool(complete), "effect": bool(eff), "rule": f"{need}/{need} seeds",
            "verdict": verdict}


def descriptive(e: dict) -> dict:
    return {**e, "complete": None, "effect": None, "rule": "descriptive", "verdict": DESCRIPTIVE}


def contrasts(vals: dict, present: dict, idx: np.ndarray, has_warm: bool) -> tuple[dict, list[str]]:
    """Every contrast of the rule on per-item values. vals: {"baseline": v, "warmup": v, (arm, seed): v}."""
    out, skipped = {}, []
    for a, b in CONTRASTS:
        common = [s for s in present[a] if s in present[b]]
        if not common:
            skipped.append(f"{a} - {b}: no seed present in both arms")
            continue
        out[f"{a} - {b}"] = effect([vals[(a, s)] for s in common], [vals[(b, s)] for s in common], idx, common)
    for ref in ("untouched",) + (("warmup",) if has_warm else ()):
        rv = vals["baseline" if ref == "untouched" else "warmup"]
        for a in ARMS:
            if not present[a]:
                skipped.append(f"{a} - {ref}: no {a} run present")
                continue
            out[f"{a} - {ref}"] = effect([vals[(a, s)] for s in present[a]], [rv] * len(present[a]), idx, present[a])
    if has_warm:
        out["warmup - untouched"] = descriptive(effect([vals["warmup"]], [vals["baseline"]], idx))
    return out, skipped


# ------------------------------------------------------------------------------------------------ descriptive blocks
def cond_length(d: dict) -> dict:
    """Reasoning tokens among correct and among incorrect samples, at each cap."""
    out = {}
    for c in d["caps"]:
        k = str(c)
        cor = [r["caps"][k][1] for r in d["rows"] if r["caps"][k][0]]
        inc = [r["caps"][k][1] for r in d["rows"] if not r["caps"][k][0]]
        out[k] = {g: {"n": len(v), "mean": (sum(v) / len(v)) if v else None,
                      "median": float(statistics.median(v)) if v else None}
                  for g, v in (("correct", cor), ("incorrect", inc))}
    return out


def answer_tokens_from_rows(d: dict) -> dict | None:
    caps = [str(c) for c in d["caps"]]
    vals: dict = {k: [] for k in caps}
    for r in d["rows"]:
        for k in caps:
            if isinstance(r.get("answer_tokens"), dict) and k in r["answer_tokens"]:
                vals[k].append(r["answer_tokens"][k])
            elif len(r["caps"][k]) >= 6:
                vals[k].append(r["caps"][k][5])
            else:
                return None
    return vals


def answer_tokens_from_traces(d: dict, tok) -> dict | None:
    p = Path(d["_path"] + ".traces.jsonl.gz")
    if tok is None or not p.exists():
        return None
    caps = [str(c) for c in d["caps"]]
    texts: dict = {k: [] for k in caps}
    with gzip.open(p, "rt") as f:
        for line in f:
            t = json.loads(line)
            ans = t["answers"]
            if isinstance(ans, str):
                import ast
                ans = ast.literal_eval(ans)
            for k in caps:
                texts[k].append(ans[k])
    if any(len(v) != len(d["rows"]) for v in texts.values()):
        return None
    return {k: [len(x) for x in tok(v, add_special_tokens=False)["input_ids"]] for k, v in texts.items()}


def _stats(v: list) -> dict:
    s = sorted(v)
    return {"mean": sum(s) / len(s), "median": float(statistics.median(s)), "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
            "n": len(s)}


def answer_block(d: dict, tok) -> dict:
    v = answer_tokens_from_rows(d)
    src = "rows"
    if v is None:
        v = answer_tokens_from_traces(d, tok)
        src = "traces sidecar, re-tokenized"
    if v is None:
        why = "rows carry no answer tokens; " + ("no traces sidecar" if tok is not None else "no --tokenizer given")
        return {"available": False, "why": why}
    return {"available": True, "source": src, "per_cap": {k: _stats(x) for k, x in v.items()}}


def _seed_mean(vals: list):
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v)) if len(v) == len(vals) and v else None


# ------------------------------------------------------------------------------------------------ analysis
def single_block(d: dict, sm: dict, target: float, tok) -> dict:
    """Untouched model or warm-up: one readout."""
    caps = d["caps"]
    big = str(max(caps))
    acc = [sm["per_cap"][str(c)]["acc"] for c in caps]
    out = {"file": d["_path"], "arm": d.get("arm"), "ckpt": d.get("ckpt"),
           "norm_auc_log2": sm["norm_auc_log2"], "norm_auc_log2_se": sm["norm_auc_log2_se"],
           "acc": {str(c): sm["per_cap"][str(c)]["acc"] for c in caps},
           "reasoning_mean": {str(c): sm["per_cap"][str(c)]["reasoning_mean"] for c in caps},
           "tokens_to_base_max_acc": tokens_at_acc(caps, acc, target),
           "natural_len": sm["natural"]["reasoning_mean_at_max"],
           "cut_off_share_at_max": sm["natural"]["cut_off_share_at_max"],
           "checks": {str(c): {k: sm["per_cap"][str(c)][k]
                               for k in ("forced_frac", "boxed_missing_frac", "answer_unended_frac")} for c in caps},
           "silent_stops": sm["natural"]["silent_stops"]}
    out["cond_length"] = cond_length(d)
    out["cond_length"]["natural"] = out["cond_length"][big]
    out["answer_tokens"] = answer_block(d, tok)
    return out


def analyze(base: dict, slots: dict, diag: dict | None = None, tok=None, manifest: str | None = None,
            warm: dict | None = None, split: str = "test") -> dict:
    diag = diag or {}
    caps = base["caps"]
    big = str(max(caps))
    ids = sorted({r["id"] for r in base["rows"]})
    idx = boot_index(len(ids))
    present = {a: [s for s in SEEDS if (a, s) in slots] for a in ARMS}
    n_runs = sum(len(v) for v in present.values())
    summ = {"baseline": summarize(base["rows"], caps)}
    if warm is not None:
        summ["warmup"] = summarize(warm["rows"], caps)
    for (a, s), d in slots.items():
        summ[f"{a}_{s}"] = summarize(d["rows"], caps)
    base_acc = [summ["baseline"]["per_cap"][str(c)]["acc"] for c in caps]
    target = base_acc[-1]

    arms: dict = {}
    for a in ARMS:
        sd = present[a]
        if not sd:
            continue
        n = len(sd)
        ss = [summ[f"{a}_{s}"] for s in sd]
        accs = {str(c): [x["per_cap"][str(c)]["acc"] for x in ss] for c in caps}
        mean_curve = [sum(accs[str(c)]) / n for c in caps]
        tok_seed = [tokens_at_acc(caps, [x["per_cap"][str(c)]["acc"] for c in caps], target) for x in ss]
        arms[a] = {
            "seeds": sd,
            "norm_auc_log2": {"seed_mean": sum(x["norm_auc_log2"] for x in ss) / n,
                              "per_seed": [x["norm_auc_log2"] for x in ss],
                              "per_seed_se": [x["norm_auc_log2_se"] for x in ss]},
            "acc": {k: {"seed_mean": sum(v) / len(v), "per_seed": v} for k, v in accs.items()},
            "reasoning_mean": {str(c): {"seed_mean": sum(x["per_cap"][str(c)]["reasoning_mean"] for x in ss) / n,
                                        "per_seed": [x["per_cap"][str(c)]["reasoning_mean"] for x in ss]} for c in caps},
            "tokens_to_base_max_acc": {"target": target, "per_seed": tok_seed,
                                       "seed_mean_of_per_seed": _seed_mean(tok_seed),
                                       "from_seed_mean_curve": tokens_at_acc(caps, mean_curve, target)},
            "natural_len": {"seed_mean": sum(x["natural"]["reasoning_mean_at_max"] for x in ss) / n,
                            "per_seed": [x["natural"]["reasoning_mean_at_max"] for x in ss]},
            "cut_off_share_at_max": {"seed_mean": sum(x["natural"]["cut_off_share_at_max"] for x in ss) / n,
                                     "per_seed": [x["natural"]["cut_off_share_at_max"] for x in ss]},
            "checks": {str(c): {k: [x["per_cap"][str(c)][k] for x in ss]
                                for k in ("forced_frac", "boxed_missing_frac", "answer_unended_frac")} for c in caps},
            "silent_stops": [x["natural"]["silent_stops"] for x in ss],
        }
        cl = [cond_length(slots[(a, s)]) for s in sd]
        arms[a]["cond_length"] = {
            k: {g: {"mean_seed_mean": _seed_mean([x[k][g]["mean"] for x in cl]),
                    "median_seed_mean": _seed_mean([x[k][g]["median"] for x in cl]),
                    "n_per_seed": [x[k][g]["n"] for x in cl],
                    "mean_per_seed": [x[k][g]["mean"] for x in cl],
                    "median_per_seed": [x[k][g]["median"] for x in cl]} for g in ("correct", "incorrect")}
            for k in map(str, caps)}
        arms[a]["cond_length"]["natural"] = arms[a]["cond_length"][big]
        ab = [answer_block(slots[(a, s)], tok) for s in sd]
        if all(x["available"] for x in ab):
            arms[a]["answer_tokens"] = {"available": True, "source": ab[0]["source"], "per_cap": {
                k: {st: {"seed_mean": sum(x["per_cap"][k][st] for x in ab) / n,
                         "per_seed": [x["per_cap"][k][st] for x in ab]} for st in ("mean", "median", "p90")}
                for k in map(str, caps)}}
        else:
            arms[a]["answer_tokens"] = {"available": False, "why": next(x.get("why") for x in ab if not x["available"])}

    baseline = single_block(base, summ["baseline"], target, tok)
    warmup = single_block(warm, summ["warmup"], target, tok) if warm is not None else None

    def effects_on(sel: list[int] | None) -> tuple[dict, list[str]]:
        ii = ids if sel is None else [ids[j] for j in sel]
        ix = idx if sel is None else boot_index(len(ii))
        au = {"baseline": item_auc(base, ii), **{k: item_auc(d, ii) for k, d in slots.items()}}
        if warm is not None:
            au["warmup"] = item_auc(warm, ii)
        return contrasts(au, present, ix, warm is not None)

    effects, skipped = effects_on(None)
    # natural length under the same rule, pairing and index matrix (H2)
    nl = {"baseline": item_natural(base, ids), **{k: item_natural(d, ids) for k, d in slots.items()}}
    if warm is not None:
        nl["warmup"] = item_natural(warm, ids)
    length_effects, _ = contrasts(nl, present, idx, warm is not None)
    # descriptive (not the decision rule): accuracy differences per cap, same pairing and draws
    acc_diffs = {}
    for c in caps:
        ac = {"baseline": item_acc(base, ids, c), **{k: item_acc(d, ids, c) for k, d in slots.items()}}
        if warm is not None:
            ac["warmup"] = item_acc(warm, ids, c)
        row, _ = contrasts(ac, present, idx, warm is not None)
        acc_diffs[str(c)] = {k: {x: e[x] for x in ("per_seed_diff", "seeds", "seed_mean_diff", "ci95")}
                             for k, e in row.items()}

    subset = None
    if split == "test" and ids and all(i.startswith("math500:") for i in ids):
        mp = Path(manifest) if manifest else DEFAULT_MANIFEST
        if mp.exists():
            pilot = set(json.loads(mp.read_text()).get("pilot50_math500_ids", []))
            sel = [j for j, i in enumerate(ids) if i not in pilot]
            si = [ids[j] for j in sel]
            subset = {"what": "MATH-500 items the pilot never read (check, not the primary set)",
                      "manifest": str(mp), "n_items": len(sel), "n_pilot_excluded": len(ids) - len(sel),
                      "norm_auc_log2": {
                          "baseline": float(item_auc(base, si).mean()),
                          **({"warmup": float(item_auc(warm, si).mean())} if warm is not None else {}),
                          **{a: {"per_seed": [float(item_auc(slots[(a, s)], si).mean()) for s in present[a]]}
                             for a in ARMS if present[a]}},
                      "effects": effects_on(sel)[0]}
            for a in ARMS:
                if present[a]:
                    v = subset["norm_auc_log2"][a]["per_seed"]
                    subset["norm_auc_log2"][a]["seed_mean"] = sum(v) / len(v)
        else:
            subset = {"what": "MATH-500 unseen-450 check", "unavailable": f"manifest not found: {mp}"}

    noise = {}
    for (a, s, t), d in sorted(diag.items()):
        key = f"{a}_{s}"
        e = noise.setdefault(key, {"final": {"file": slots[(a, s)]["_path"], "norm_auc_log2": summ[key]["norm_auc_log2"]},
                                   "steps": {}})
        e["steps"][str(t)] = {"file": d["_path"], "norm_auc_log2": summarize(d["rows"], caps)["norm_auc_log2"]}
    for key, e in noise.items():
        v = [e["final"]["norm_auc_log2"]] + [x["norm_auc_log2"] for x in e["steps"].values()]
        e["spread_max_minus_min"] = max(v) - min(v)
        e["sd"] = statistics.stdev(v) if len(v) > 1 else None

    sets = sorted({i.split(":")[0] for i in ids})
    complete = n_runs == len(ARMS) * len(SEEDS)
    blocks = {"effects": effects, "length_effects": length_effects}
    hyp = {}
    for h, spec in HYPOTHESES.items():
        e = blocks[spec["block"]].get(spec["contrast"])
        applies = split == "test" and sets == [spec["set"]]
        row = {**spec, "applies": applies}
        if not applies:
            row["verdict"] = f"not applicable here (needs the test split of {spec['set']}; this analysis: {split} {sets})"
        elif e is None:
            row["verdict"] = f"not evaluated: contrast {spec['contrast']} absent"
        else:
            sign_ok = (e["seed_mean_diff"] > 0) if spec["direction"] == "positive" else (e["seed_mean_diff"] < 0)
            ok = bool(e["effect"]) and sign_ok
            row.update({"per_seed_diff": e["per_seed_diff"], "seeds": e["seeds"], "seed_mean_diff": e["seed_mean_diff"],
                        "ci95": e["ci95"], "rule_verdict": e["verdict"], "supported": ok,
                        "verdict": "supported" if ok else ("rule effect in the other direction" if e["effect"]
                                                           else "not supported (no effect by the rule)")})
        hyp[h] = row
    res = {"step": STEP_OF_SPLIT[split], "split": split, "split_label": SPLITS[split], "eval_sets": sets,
           "caps": caps, "n_items": len(ids), "samples_per_item": base["args"]["samples"],
           "complete": complete, "partial": not complete, "n_arm_runs": n_runs,
           "n_arm_runs_planned": len(ARMS) * len(SEEDS), "arms_present": present,
           "partial_note": None if complete else
           (f"partial: {n_runs} of {len(ARMS) * len(SEEDS)} arm runs present; contrasts use the seeds present and a "
            f"contrast with fewer than {len(SEEDS)} seeds has no verdict"),
           "primary_metric": f"normalized area under accuracy vs log2 budget, per arm the mean of {len(SEEDS)} seeds",
           "decision_rule": f"effect iff all {len(SEEDS)} per-seed differences share a nonzero sign AND the paired item bootstrap "
                            f"95% percentile interval of the seed-mean difference excludes 0 ({BOOT_DRAWS} draws, "
                            f"numpy default_rng({BOOT_SEED}), items resampled with replacement, seeds paired by index)",
           "baseline": baseline, "warmup": warmup, "arms": arms, "effects": effects, "contrasts_skipped": skipped,
           "length_effects": length_effects, "preregistered_hypotheses": hyp,
           "acc_diffs_descriptive": acc_diffs, "math500_unseen450": subset,
           "checkpoint_noise_diagnostic": {"note": "diagnostic only, never used for selection; the final checkpoint "
                                                   "is the one read out", "runs": noise} if noise else None,
           "per_file_summary": summ}
    if split == "test":
        res["test_set"] = sets
    return res


# ------------------------------------------------------------------------------------------------ CLI
def parse_slots(args: list[str]) -> tuple[dict, dict]:
    slots, diag, bad = {}, {}, []
    for x in args:
        if "=" not in x:
            bad.append(f"{x!r}: expected ARM_SEED=path or ARM_SEED_stepN=path")
            continue
        lab, p = x.split("=", 1)
        m = LABEL.match(lab)
        if not m:
            bad.append(f"{lab!r}: label must be ARM_SEED or ARM_SEED_stepN with ARM in {ARMS}")
            continue
        a, s, t = m.group(1), int(m.group(2)), m.group(3)
        key = (a, s, int(t)) if t else (a, s)
        tgt = diag if t else slots
        if key in tgt:
            bad.append(f"{lab}: given twice")
        tgt[key] = load(p)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise Refuse(2)
    return slots, diag


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_table(res: dict) -> None:
    caps = res["caps"]
    big = str(max(caps))
    B, W, A = res["baseline"], res.get("warmup"), res["arms"]
    print(f"\nstep {res['step']}, {res['split_label'].upper()} split, sets {','.join(res['eval_sets'])}: "
          f"{res['n_items']} items x {res['samples_per_item']} samples, caps {caps}")
    if res["partial"]:
        print(res["partial_note"].replace("partial:", "PARTIAL:", 1) + "; present: " +
              ", ".join(f"{a} seeds {v}" for a, v in res["arms_present"].items() if v))
    hdr = ["model", "area", "area per seed"] + [f"acc@{c}" for c in caps] + \
          ["tok@base8k", "natural", f"cut@{big}", f"nobox@{big}", "silent"]
    tk = lambda v: "not reached" if v is None else fmt(v, 0)  # noqa: E731
    singles = [("untouched", B)] + ([("warm-up", W)] if W else [])
    rows = [[name, fmt(x["norm_auc_log2"]), "-"] + [fmt(x["acc"][str(c)], 3) for c in caps] +
            [tk(x["tokens_to_base_max_acc"]), fmt(x["natural_len"], 0), fmt(x["cut_off_share_at_max"], 3),
             fmt(x["checks"][big]["boxed_missing_frac"], 3), str(x["silent_stops"])] for name, x in singles]
    for a in ARMS:
        if a not in A:
            rows.append([a, "absent"] + [""] * (len(hdr) - 2))
            continue
        x = A[a]
        rows.append([f"{a} (seeds {','.join(map(str, x['seeds']))})", fmt(x["norm_auc_log2"]["seed_mean"]),
                     " ".join(fmt(v) for v in x["norm_auc_log2"]["per_seed"])] +
                    [fmt(x["acc"][str(c)]["seed_mean"], 3) for c in caps] +
                    [tk(x["tokens_to_base_max_acc"]["from_seed_mean_curve"]), fmt(x["natural_len"]["seed_mean"], 0),
                     fmt(x["cut_off_share_at_max"]["seed_mean"], 3),
                     fmt(_seed_mean(x["checks"][big]["boxed_missing_frac"]), 3), "/".join(map(str, x["silent_stops"]))])
    _grid(hdr, rows)
    print(f"\narm effects on the area, {res['split_label']} split (rule: {len(SEEDS)}/{len(SEEDS)} seeds same sign AND bootstrap 95% "
          "interval excludes 0)")
    hdr = ["contrast", "seeds", "per-seed diff", "seed-mean diff", "95% interval", "signs", "excl 0", "verdict"]
    rows = []
    for k, e in res["effects"].items():
        desc = e["rule"] == "descriptive"
        rows.append([k, "-" if e["seeds"] is None else ",".join(map(str, e["seeds"])),
                     " ".join(f"{v:+.4f}" for v in e["per_seed_diff"]), f"{e['seed_mean_diff']:+.4f}",
                     f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]",
                     "-" if desc else ("agree" if e["signs_agree"] else "mixed/0"),
                     "yes" if e["ci_excludes_0"] else "no", e["verdict"]])
    _grid(hdr, rows)
    for k in res.get("contrasts_skipped") or []:
        print(f"  skipped: {k}")
    print(f"\nnatural length (reasoning tokens at {big}) under the same rule; negative = the first arm reasons less")
    rows = []
    for k, e in (res.get("length_effects") or {}).items():
        desc = e["rule"] == "descriptive"
        rows.append([k, "-" if e["seeds"] is None else ",".join(map(str, e["seeds"])),
                     " ".join(f"{v:+.0f}" for v in e["per_seed_diff"]), f"{e['seed_mean_diff']:+.1f}",
                     f"[{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]",
                     "-" if desc else ("agree" if e["signs_agree"] else "mixed/0"),
                     "yes" if e["ci_excludes_0"] else "no", e["verdict"]])
    if rows:
        _grid(hdr, rows)
    for h, row in (res.get("preregistered_hypotheses") or {}).items():
        print(f"  {h} ({row['text']}): {row['verdict']}")
    print("\nreasoning tokens among correct / incorrect samples (mean, median; arms = seed mean)")
    hdr = ["model"] + [f"{c} cor" for c in caps] + [f"{c} inc" for c in caps]
    rows = []
    for name, cl in [(n, x["cond_length"]) for n, x in singles] + [(a, A[a]["cond_length"]) for a in ARMS if a in A]:
        r = [name]
        for g in ("correct", "incorrect"):
            for c in caps:
                z = cl[str(c)][g]
                mean = z.get("mean", z.get("mean_seed_mean"))
                med = z.get("median", z.get("median_seed_mean"))
                r.append(f"{fmt(mean, 0)}/{fmt(med, 0)}")
        rows.append(r)
    _grid(hdr, rows)
    print("\nanswer tokens per budget (mean)")
    hdr = ["model"] + [str(c) for c in caps]
    rows = []
    blocks = [(n, x["answer_tokens"]) for n, x in singles] + [(a, A[a]["answer_tokens"]) for a in ARMS if a in A]
    for name, ab in blocks:
        if not ab["available"]:
            rows.append([name] + ["unavailable"] + [""] * (len(caps) - 1))
        else:
            ms = [ab["per_cap"][str(c)]["mean"] for c in caps]
            rows.append([name] + [fmt(m["seed_mean"] if isinstance(m, dict) else m, 0) for m in ms])
    _grid(hdr, rows)
    for name, ab in blocks:
        if not ab["available"]:
            print(f"  {name}: {ab['why']}")
    sub = res.get("math500_unseen450")
    if sub and "effects" in sub:
        print(f"\nMATH-500 unseen check ({sub['n_items']} items): area untouched {sub['norm_auc_log2']['baseline']:.4f}, " +
              ", ".join(f"{a} {sub['norm_auc_log2'][a]['seed_mean']:.4f}" for a in ARMS if a in sub["norm_auc_log2"]))
        for k, e in sub["effects"].items():
            print(f"  {k}: {e['seed_mean_diff']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] {e['verdict']}")
    elif sub:
        print(f"\nMATH-500 unseen check: {sub['unavailable']}")
    nd = res.get("checkpoint_noise_diagnostic")
    if nd:
        print("\ncheckpoint noise (diagnostic only, never selection): area spread over the last checkpoints")
        for k, e in nd["runs"].items():
            print(f"  {k}: spread {e['spread_max_minus_min']:.4f} over final + steps {sorted(e['steps'], key=int)}")


def _grid(hdr, rows):
    w = [max(len(str(r[j])) for r in [hdr] + rows) for j in range(len(hdr))]
    for r in [hdr] + rows:
        print("  ".join(str(v).ljust(w[j]) for j, v in enumerate(r)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(SPLITS),
                    help="val = step 12 (validation slice), test = step 14 (final test); labels every output")
    ap.add_argument("--baseline", required=True, help="untouched-model readout (role untouched, arm stock)")
    ap.add_argument("--warmup", default=None, help="shared format warm-up readout (role warmup, arm ctrl + checkpoint)")
    ap.add_argument("slots", nargs="+", help="ARM_SEED=path for ctrl/wide/wide1 x 0/1/2; ARM_SEED_stepN=path diagnostic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_partial", action="store_true",
                    help="validation only: accept missing arm runs (contrasts with fewer than 3 seeds get no verdict)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir for answer tokens from the traces sidecar")
    ap.add_argument("--manifest", default=None, help=f"split manifest (default {DEFAULT_MANIFEST})")
    ap.add_argument("--seeds", default="0,1,2", help="seed set of the rule (default 0,1,2: the step 7 record)")
    a = ap.parse_args(argv)
    set_seeds(a.seeds.split(","))
    base = load(a.baseline)
    warm = load(a.warmup) if a.warmup else None
    slots, diag = parse_slots(a.slots)
    bad = validate(base, slots, diag, warm, a.split, a.allow_partial, a.manifest)
    if bad:
        print("REFUSED, inputs differ:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise Refuse(2)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    res = analyze(base, slots, diag, tok, a.manifest, warm, a.split)
    res["receipt"] = {
        "argv": sys.argv if argv is None else ["step14_analysis.py"] + list(argv),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": _sha_file(__file__), "python": platform.python_version(), "numpy": np.__version__,
        "split": a.split,
        "inputs": {"baseline": {"file": base["_path"], "sha256": base["_sha256"], "arm": base["arm"], "role": "untouched"},
                   **({"warmup": {"file": warm["_path"], "sha256": warm["_sha256"], "arm": warm["arm"], "role": "warmup",
                                  "ckpt": warm.get("ckpt"), "ckpt_sha256": warm["env"]["sha256"].get("ckpt")}}
                      if warm is not None else {}),
                   **{f"{x}_{s}": {"file": d["_path"], "sha256": d["_sha256"], "arm": d["arm"],
                                   "ckpt": d.get("ckpt"), "ckpt_sha256": d["env"]["sha256"].get("ckpt")}
                      for (x, s), d in sorted(slots.items())},
                   **{f"{x}_{s}_step{t}": {"file": d["_path"], "sha256": d["_sha256"]} for (x, s, t), d in sorted(diag.items())}},
        "readout": {"seed": base["args"]["seed"], "samples": base["args"]["samples"],
                    "answer_max": base["args"].get("answer_max"), "sampling": base.get("sampling"),
                    "data_sha256": base["env"]["sha256"].get("data"), "gpu": base["env"].get("gpu")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print_table(res)
    print(f"\n{res['split_label']} receipt: {a.out}")
    return res


if __name__ == "__main__":
    main()
